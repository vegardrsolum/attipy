from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._quatops import _normalize, _quatprod
from ._transforms import _matrix_from_quat
from ._vectorops import _skew_symmetric


def _signed_smallest_angle(angle: float, degrees: bool = True) -> float:
    """
    Convert the given angle to the smallest angle between [-180., 180) degrees.

    Parameters
    ----------
    angle : float
        Value of angle.
    degrees : bool, default True
        Specify whether ``angle`` is given degrees or radians.

    Returns
    -------
    float
        The smallest angle between [-180., 180) degrees (or  [-pi, pi] radians).
    """
    base = 180.0 if degrees else np.pi
    return (angle + base) % (2.0 * base) - base


def _h_head(q: NDArray[np.float64]) -> float:
    """
    Compute yaw angle from unit quaternion.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    float
        Yaw angle in the NED reference frame.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.251, John Wiley & Sons, 2021.
    """
    q_w, q_x, q_y, q_z = q
    u_y = 2.0 * (q_x * q_y + q_z * q_w)
    u_x = 1.0 - 2.0 * (q_y**2 + q_z**2)
    return np.arctan2(u_y, u_x)  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _dhda_head(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute yaw angle gradient wrt to the unit quaternion.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (3,)
        Unit quaternion.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Yaw angle gradient vector.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.254, John Wiley & Sons, 2021.
    """
    q_w, q_x, q_y, q_z = q
    u_y = 2.0 * (q_x * q_y + q_z * q_w)
    u_x = 1.0 - 2.0 * (q_y**2 + q_z**2)
    u = u_y / u_x

    duda_scale = 1.0 / u_x**2
    duda_x = -(q_w * q_y) * (1.0 - 2.0 * q_w**2) - (2.0 * q_w**2 * q_x * q_z)
    duda_y = (q_w * q_x) * (1.0 - 2.0 * q_z**2) + (2.0 * q_w**2 * q_y * q_z)
    duda_z = q_w**2 * (1.0 - 2.0 * q_y**2) + (2.0 * q_w * q_x * q_y * q_z)
    duda = duda_scale * np.array([duda_x, duda_y, duda_z])

    dhda = 1.0 / (1.0 + u**2) * duda

    return dhda  # type: ignore[no-any-return]


class AHRS:
    """
    Attitude and heading reference system (AHRS).

    The internal filter is a multiplicative extended Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    x0_prior : array-like, shape (7,), default (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        Initial (a priori) 7-element AHRS state estimate:

        * Attitude (unit quaternion) - 4 elements
        * Gyroscope bias (x, y, z) - 3 elements

        Defaults to attitude as the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e.,
        no rotation), and no bias.
    P0_prior : array-like, shape (6, 6), default np.eye(6) * 1e-6
        Initial (a priori) estimate of the error covariance matrix, **P**. Defaults
        to a small diagonal matrix (np.eye(6) * 1e-6).
    err_gyro : dict of {str: float}, default :const:`smsfusion.constants.ERR_GYRO_MOTION2`
        Dictionary containing gyroscope noise parameters with keys:

        * ``N``: White noise power spectral density in (rad/s)/sqrt(Hz).
        * ``B``: Bias stability in rad/s.
        * ``tau_cb``: Bias correlation time in seconds.

        Defaults to {'N': 0.0001, 'B': 0.00005, 'tau_cb': 50.0} which are typical
        values for low-cost MEMS gyroscopes.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up). The body's (or IMU/AHRS
        sensor's) degrees of freedom will be expressed relative to this frame.
        Furthermore, the aiding heading angle is also interpreted relative to this
        frame according to the right-hand rule.
    """

    def __init__(
        self,
        fs: float,
        q0: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg0: ArrayLike = (0.0, 0.0, 0.0),
        P0_prior: ArrayLike = 1e-6 * np.eye(6),
        err_gyro: dict[str, float] = {"N": 0.0001, "B": 0.00005, "tau_cb": 50.0},
        nav_frame: str = "NED",
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._err_gyro = err_gyro
        self._dq_prealloc = np.array([2.0, 0.0, 0.0, 0.0])  # Preallocation
        self._nav_frame = nav_frame.lower()

        # Strapdown algorithm / INS state
        self._att = Attitude(q0)
        self._bg = np.asarray_chkfinite(bg0).reshape(3)

        # Gravity reference vector
        if self._nav_frame == "ned":
            self._vg_ref_n = np.array([0.0, 0.0, 1.0])
        elif self._nav_frame == "enu":
            self._vg_ref_n = np.array([0.0, 0.0, -1.0])
        else:
            raise ValueError(f"Unknown navigation frame: {self._nav_frame}")

        # Error state estimate (after reset)
        self._dx_prealloc = np.zeros(6)  # always zero, but used in sequential update

        # Initialize Kalman filter
        self._P_prior = np.asarray_chkfinite(P0_prior).copy(order="C")
        self._P = self._P_prior.copy(order="C")

        # Prepare system matrices
        self._F = self._prep_F(err_gyro)
        self._G = self._prep_G()
        self._H = self._prep_H()
        self._W = self._prep_W(err_gyro)
        self._I = np.eye(6, order="C")

        # Error-state estimate (before reset)
        self._dx = np.empty_like(self._dx_prealloc)  # needed for smoothing only

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Error covariance matrix, **P**. I.e., the error covariance matrix associated with
        the Kalman filter's updated (a posteriori) error-state estimate.
        """
        P = self._P.copy()
        return P

    @property
    def P_prior(self) -> NDArray[np.float64]:
        """
        Next (a priori) estimate of the error covariance matrix, **P**. I.e., the error
        covariance matrix associated with the Kalman filter's projected (a priori)
        error-state estimate.
        """
        P_prior = self._P_prior.copy()
        return P_prior

    @staticmethod
    def _prep_F(err_gyro: dict[str, float]) -> NDArray[np.float64]:
        """
        Prepare linearized state matrix, F.
        """

        beta_gyro = 1.0 / err_gyro["tau_cb"]

        # Temporary placeholder vectors (to be replaced each timestep)
        w_ins = np.array([0.0, 0.0, 0.0])

        S = _skew_symmetric  # alias skew symmetric matrix

        # State transition matrix
        F = np.zeros((6, 6))
        F[0:3, 0:3] = -S(w_ins)  # NB! update each time step
        F[0:3, 3:6] = -np.eye(3)
        F[3:6, 3:6] = -beta_gyro * np.eye(3)

        return F

    def _update_F(self, w_ins: NDArray[np.float64]) -> None:
        """Update linearized state transition matrix, F."""
        S = _skew_symmetric  # alias skew symmetric matrix

        # Update matrix
        self._F[0:3, 0:3] = -S(w_ins)  # NB! update each time step

    @staticmethod
    def _prep_G() -> NDArray[np.float64]:
        """Prepare (white noise) input matrix, G."""

        # Input (white noise) matrix
        G = np.zeros((6, 6))
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = np.eye(3)
        return G

    @staticmethod
    def _prep_H() -> NDArray[np.float64]:
        """Prepare linearized measurement matrix, H. Values are placeholders only"""
        H = np.zeros((4, 6))
        return H

    def _update_H_g_ref(self, R_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for g_ref aiding."""
        S = _skew_symmetric
        self._H[0:3, 0:3] = S(R_nm.T @ self._vg_ref_n)
        return self._H[0:3]

    def _update_H_head(self, q_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update and return part of H matrix relevant for heading aiding."""
        self._H[3:4, 0:3] = _dhda_head(q_nm)
        return self._H[3:4]

    @staticmethod
    def _prep_W(err_gyro: dict[str, float]) -> NDArray[np.float64]:
        """Prepare white noise power spectral density matrix"""
        N_gyro = err_gyro["N"]
        sigma_gyro = err_gyro["B"]
        beta_gyro = 1.0 / err_gyro["tau_cb"]

        # White noise power spectral density matrix
        W = np.eye(6)
        W[0:3, 0:3] *= N_gyro**2
        W[3:6, 3:6] *= 2.0 * sigma_gyro**2 * beta_gyro
        return W

    def _reset_ins(self, dx: NDArray[np.float64]) -> None:
        """Combine states and reset INS"""
        da = dx[0:3]
        self._dq_prealloc[1:4] = da
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * self._dq_prealloc
        self._att._q = _quatprod(self._att._q, dq)
        self._att._q = _normalize(self._att._q)
        self._bg = self._bg + dx[3:6]
        self._dx_prealloc[:] = np.zeros(dx.size)

    @staticmethod
    @njit  # type: ignore[misc]
    def _update_dx_P(
        dx: NDArray[np.float64],
        P: NDArray[np.float64],
        dz: NDArray[np.float64],
        var: NDArray[np.float64],
        H: NDArray[np.float64],
        I_: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        for i, (dz_i, var_i) in enumerate(zip(dz, var)):
            H_i = np.ascontiguousarray(H[i, :])
            K_i = P @ H_i.T / (H_i @ P @ H_i.T + var_i)
            dx += K_i * (dz_i - H_i @ dx)
            K_i = np.ascontiguousarray(K_i[:, np.newaxis])  # as 2D array
            H_i = np.ascontiguousarray(H_i[np.newaxis, :])  # as 2D array
            P = (I_ - K_i @ H_i) @ P @ (I_ - K_i @ H_i).T + var_i * K_i @ K_i.T
        return dx, P

    def update(
        self,
        f_imu: ArrayLike,
        w_imu: ArrayLike,
        degrees: bool = False,
        head: float | None = None,
        head_var: float | None = None,
        head_degrees: bool = True,
        g_ref: bool = False,
        g_var: ArrayLike | None = None,
    ) -> Self:
        """
        Update/correct the AHRS' state estimate with aiding measurements, and project
        ahead using IMU measurements.

        If no aiding measurements are provided, the AHRS is simply propagated ahead
        using dead reckoning with the IMU measurements.

        Parameters
        ----------
        f_imu : array-like, shape (3,)
            Specific force measurements (i.e., accelerations + gravity), given
            as [f_x, f_y, f_z]^T where f_x, f_y and f_z are
            acceleration measurements in x-, y-, and z-direction, respectively.
        w_imu : array-like, shape (3,)
            Angular rate measurements, given as [w_x, w_y, w_z]^T where
            w_x, w_y and w_z are angular rates about the x-, y-,
            and z-axis, respectively.
        degrees : bool, default False
            Specifies whether the unit of ``w_imu`` are in degrees or radians.
        head : float, optional
            Heading measurement. I.e., the yaw angle of the 'body' frame relative to the
            assumed 'navigation' frame ('NED' or 'ENU') specified during initialization.
            If ``None``, compass aiding is not used. See ``head_degrees`` for units.
        head_var : float, optional
            Variance of heading measurement noise. Units must be compatible with ``head``.
             See ``head_degrees`` for units. Required for ``head``.
        head_degrees : bool, default False
            Specifies whether the unit of ``head`` and ``head_var`` are in degrees and degrees^2,
            or radians and radians^2. Default is in radians and radians^2.
        g_ref : bool, optional, default False
            Specifies whether the gravity reference vector is used as an aiding measurement.
        g_var : array-like, shape (3,), optional
            Variance of gravitational reference vector measurement noise. Required for
            ``g_ref``.

        Returns
        -------
        AHRS
            A reference to the instance itself after the update.
        """

        f_imu = np.asarray(f_imu, dtype=float)
        w_imu = np.asarray(w_imu, dtype=float)

        if degrees:
            w_imu = (np.pi / 180.0) * w_imu

        # Bias compensated IMU measurements
        w_ins = w_imu - self._bg

        # Current INS state estimates
        q_ins_nm = self._att._q
        R_ins_nm = _matrix_from_quat(q_ins_nm)  # body-to-inertial rot matrix

        # Aliases
        dx = self._dx_prealloc  # zeros
        dt = self._dt
        F = self._F
        G = self._G
        W = self._W
        P = self._P_prior
        I_ = self._I

        # Update system matrices
        self._update_F(w_ins)

        if g_ref:
            if g_var is None:
                raise ValueError("'g_var' not provided.")
            vg_meas_m = -_normalize(f_imu)
            g_var = np.asarray(g_var, dtype=float, order="C")
            dz_g = vg_meas_m - R_ins_nm.T @ self._vg_ref_n
            H_g = self._update_H_g_ref(R_ins_nm)
            dx, P = self._update_dx_P(dx, P, dz_g, g_var, H_g, I_)

        if head is not None:
            if head_var is None:
                raise ValueError("'head_var' not provided.")

            if head_degrees:
                head = (np.pi / 180.0) * head
                head_var = (np.pi / 180.0) ** 2 * head_var

            head_var_ = np.asarray([head_var], dtype=float, order="C")
            dz_head = np.asarray(
                [_signed_smallest_angle(head - _h_head(q_ins_nm), degrees=False)],
                dtype=float,
                order="C",
            )

            H_head = self._update_H_head(q_ins_nm)
            dx, P = self._update_dx_P(dx, P, dz_head, head_var_, H_head, I_)

        self._dx[:] = dx.ravel().copy()

        # Reset INS state
        if dx.any():
            self._reset_ins(dx.ravel())

        # Update current state
        self._P[:] = P

        # Discretize system
        phi = I_ + dt * F  # state transition matrix
        Q = dt * G @ W @ G.T  # process noise covariance matrix

        # Project ahead
        self._att.update(w_ins * dt, degrees=False)
        self._P_prior[:] = phi @ P @ phi.T + Q

        return self
