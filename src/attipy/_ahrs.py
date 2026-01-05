from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._quatops import _quatprod
from ._transforms import _matrix_from_quat
from ._vectorops import _normalize, _skew_symmetric


def _ssa(angle: float, degrees: bool = True) -> float:
    """
    Convert the given angle to the smallest signed angle between [-180., 180) degrees.

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


def _state_matrix(f_corr, w_corr, R_nm, gbc) -> NDArray[np.float64]:
    """
    Setup linearized state matrix, dfdx.
    """

    beta_gyro = 1.0 / gbc

    S = _skew_symmetric  # alias skew symmetric matrix

    # State transition matrix
    dfdx = np.zeros((9, 9))
    dfdx[0:3, 0:3] = -S(w_corr)  # NB! update each time step
    dfdx[0:3, 3:6] = -np.eye(3)
    dfdx[3:6, 3:6] = -beta_gyro * np.eye(3)
    dfdx[6:9, 0:3] = -R_nm @ S(f_corr)  # NB! update each time step

    return dfdx


def _wn_input_matrix(R_nm):
    """Setup linearized (white noise) input matrix, dfdw."""

    # Input (white noise) matrix
    dfdw = np.zeros((9, 9))
    dfdw[0:3, 0:3] = -np.eye(3)
    dfdw[3:6, 3:6] = np.eye(3)
    dfdw[6:9, 6:9] = -R_nm  # NB! update each time step

    return dfdw


def _wn_psd_matrix(vrw, arw, gbs, gbc) -> NDArray[np.float64]:
    """Setup white noise (process noise) power spectral density matrix, W."""
    N_acc = vrw
    N_gyro = arw
    sigma_gyro = gbs
    beta_gyro = 1.0 / gbc

    # White noise power spectral density matrix
    W = np.eye(9)
    W[0:3, 0:3] *= N_gyro**2
    W[3:6, 3:6] *= 2.0 * sigma_gyro**2 * beta_gyro
    W[6:9, 6:9] *= N_acc**2

    return W


@njit  # type: ignore[misc]
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


def _measurement_matrix(vg_ref_n, q_nm) -> None:
    """Setup linearized measurement matrix, dhdx."""
    S = _skew_symmetric

    R_nm = _matrix_from_quat(q_nm)

    dhdx = np.zeros((7, 9))
    dhdx[0:3, 0:3] = S(R_nm.T @ vg_ref_n)  # gravity reference vector
    dhdx[3:4, 0:3] = _dhda_head(q_nm)  # heading
    dhdx[4:7, 6:9] = np.eye(3)  # velocity

    return dhdx


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


class AHRS:
    """
    Attitude and Heading Reference System (AHRS).

    The internal filter is a multiplicative extended Kalman filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    q0 : Attitude or array_like, shape (4,), default (1.0, 0.0, 0.0, 0.0)
        Initial (a priori) attitude state estimate, given as a unit quaternion
        or :class:`~attipy.Attitude` instance. Defaults to the identity quaternion
        (1.0, 0.0, 0.0, 0.0), i.e., no rotation.
    bg0 : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial (a priori) gyroscope bias estimate. Defaults to zero bias.
    P0 : array_like, shape (6, 6), default np.eye(6) * 1e-6
        Initial (a priori) estimate of the error covariance matrix, **P**. Defaults
        to a small diagonal matrix (np.eye(6) * 1e-6).
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like 'navigation' frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up). The body's (or IMU/AHRS
        sensor's) degrees of freedom will be expressed relative to this frame.
        Furthermore, the aiding heading angle is also interpreted relative to this
        frame according to the right-hand rule.
    """

    _I = np.eye(9)
    _dx = np.zeros(9)  # (da, dbg, dv), always zero after reset
    _dq_prealloc = np.array([2.0, 0.0, 0.0, 0.0])  # Preallocation

    def __init__(
        self,
        fs: float,
        q0: ArrayLike | Attitude = (1.0, 0.0, 0.0, 0.0),
        bg0: ArrayLike = (0.0, 0.0, 0.0),
        v0: ArrayLike = (0.0, 0.0, 0.0),
        P0: ArrayLike = 1e-6 * np.eye(9),
        g: float = 9.80665,
        nav_frame: str = "NED",
        acc_noise_density: float = 0.001,
        gyro_noise_density: float = 0.0001,
        gyro_bias_stability: float = 0.00005,
        bias_corr_time: float = 50.0,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._nav_frame = nav_frame.lower()
        self._g = g
        self._g_n = self._gravity_nav(self._nav_frame)
        self._vg_ref_n = _normalize(self._g_n)

        # IMU noise parameters
        self._vrw = acc_noise_density  # velocity random walk
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._att = q0 if isinstance(q0, Attitude) else Attitude(q0)
        self._bg = np.asarray_chkfinite(bg0).reshape(3)
        self._v = np.asarray_chkfinite(v0).reshape(3)
        self._P = np.asarray_chkfinite(P0).copy()

        # Additional state variables
        self._f = -self._g_n.copy()
        self._w = np.zeros(3)
        self._R_nm = self._att.as_matrix()  # avoiding repeated calls

        # Prepare system matrices
        self._dfdx = _state_matrix(self._f, self._w, self._R_nm, self._gbc)
        self._dfdw = _wn_input_matrix(self._R_nm)
        self._dhdx = _measurement_matrix(self._vg_ref_n, self._att._q)
        self._W = _wn_psd_matrix(self._vrw, self._arw, self._gbs, self._gbc)

    def _gravity_nav(self, nav_frame) -> NDArray[np.float64]:
        """
        Gravity vector direction in the navigation frame (NED or ENU).
        """
        if nav_frame == "ned":
            g_n = np.array([0.0, 0.0, self._g])
        elif nav_frame == "enu":
            g_n = np.array([0.0, 0.0, -self._g])
        else:
            raise ValueError(f"Unknown navigation frame: {self._nav_frame}.")
        return g_n

    @property
    def attitude(self) -> Attitude:
        """
        Attitude estimate.
        """
        return self._att

    def bias_gyro(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Gyroscope bias estimate.
        """
        bg = self._bg.copy()
        if degrees:
            bg = np.degrees(bg)
        return bg

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Error covariance matrix estimate.
        """
        P = self._P.copy()
        return P

    def _dhdx_head(self, q_nm):
        """
        Heading measurement matrix.
        """
        self._dhdx[3:4, 0:3] = _dhda_head(q_nm)
        return self._dhdx[3:4]

    def _dhdx_g_ref(self, R_nm):
        """
        Gravity reference vector measurement matrix.
        """
        S = _skew_symmetric
        self._dhdx[0:3, 0:3] = S(R_nm.T @ self._vg_ref_n)
        return self._dhdx[0:3]

    def _dhdx_vel(self):
        """
        Velocity measurement matrix.
        """
        return self._dhdx[4:7]

    def _reset(self) -> None:
        """Reset state (regulating error state to zero)."""
        dx = self._dx

        if not dx.any():
            return

        da = dx[0:3]
        self._dq_prealloc[1:4] = da
        dq = (1.0 / np.sqrt(4.0 + da.T @ da)) * self._dq_prealloc
        self._att._q[:] = _quatprod(self._att._q, dq)
        self._att._q[:] = _normalize(self._att._q)
        self._bg[:] = self._bg + dx[3:6]
        self._v[:] = self._v + dx[6:9]
        self._dx[:] = np.zeros(dx.size)

    def _aiding_update_head(self, head_meas, head_var, head_degrees):
        """
        Update with heading measurement.
        """
        dx, P = self._dx, self._P

        if head_meas is None:
            return dx, P

        if head_var is None:
            raise ValueError("'head_var' not provided.")

        if head_degrees:
            head_meas = (np.pi / 180.0) * head_meas
            head_var = (np.pi / 180.0) ** 2 * head_var

        head = _h_head(self._att._q)  # heading estimate

        var = np.asarray([head_var], dtype=float)
        dz = np.asarray([_ssa(head_meas - head, degrees=False)], dtype=float)
        dhdx = self._dhdx_head(self._att._q)

        self._dx[:], self._P[:] = _update_dx_P(dx, P, dz, var, dhdx, self._I)

    def _aiding_update_vel(self, vel_meas, vel_var):
        """
        Update with velocity vector measurement.
        """
        dx, P = self._dx, self._P

        if vel_meas is None:
            return dx, P

        if vel_var is None:
            raise ValueError("'vel_var' not provided.")

        vel_meas = np.asarray(vel_meas, dtype=float)
        var = np.asarray(vel_var, dtype=float)
        dz = vel_meas - self._v
        dhdx = self._dhdx_vel()

        self._dx[:], self._P[:] = _update_dx_P(dx, P, dz, var, dhdx, self._I)

    def _aiding_update_g_ref(self, f, g_var, g_ref):
        """
        Update with gravity reference vector measurement.
        """
        dx, P = self._dx, self._P

        if g_ref is False:
            return dx, P

        if g_var is None:
            raise ValueError("'g_var' not provided.")

        R_nm = self._att.as_matrix()

        var = np.asarray(g_var, dtype=float)
        vg_meas_m = -_normalize(f)
        dz = vg_meas_m - R_nm.T @ self._vg_ref_n
        dhdx = self._dhdx_g_ref(R_nm)

        self._dx[:], self._P[:] = _update_dx_P(dx, P, dz, var, dhdx, self._I)

    def _phi(self, dt):
        """
        State transition matrix.
        """

        dfdx = self._dfdx
        I_ = self._I
        f_corr = self._f
        w_corr = self._w - self._bg

        # Update
        S = _skew_symmetric
        dfdx[0:3, 0:3] = -S(w_corr)
        dfdx[6:9, 0:3] = -self._R_nm @ S(f_corr)

        # Discretize
        phi = I_ + dt * dfdx  # first-order approximation

        return phi

    def _Q(self, dt):
        """
        Process noise covariance matrix.
        """
        dfdw = self._dfdw
        W = self._W

        # Update
        dfdw[6:9, 6:9] = -self._R_nm

        # Discretize
        Q = dt * dfdw @ W @ dfdw.T

        return Q

    def _project_ahead(self, dt, f, w):
        """
        Project state ahead using dead reckoning.
        """

        f_corr = f
        w_corr = w - self._bg
        f_corr_prev = self._f
        w_corr_prev = self._w - self._bg

        # Discretized (error) state space
        phi, Q = self._phi(dt), self._Q(dt)

        # Velocity
        dv = 0.5 * (f_corr + f_corr_prev) * dt  # trapezoidal rule
        dv_corr = dt * self._g_n
        self._v += self._R_nm @ dv + dv_corr

        # Attitude
        dtheta = 0.5 * (w_corr + w_corr_prev) * dt  # trapezoidal rule
        self._att.update(dtheta, degrees=False)

        # Covariance
        self._P = phi @ self._P @ phi.T + Q

    def update(
        self,
        f: ArrayLike,
        w: ArrayLike,
        degrees: bool = False,
        v: ArrayLike | None = (0.0, 0.0, 0.0),
        v_var: ArrayLike | None = (100.0, 100.0, 100.0),
        hdg: float | None = None,
        hdg_var: float | None = None,
        hdg_degrees: bool = True,
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
        f : array-like, shape (3,)
            Specific force (i.e., accelerations + gravity) measurement (fx, fy, fz).
        w : array-like, shape (3,)
            Angular rate measurement (wx, wy, wz).
        degrees : bool, default False
            Specifies whether the unit of ``w`` are in degrees or radians.
        v : array-like, shape (3,), optional
            Velocity measurement (vx, vy, vz). If ``None``, velocity aiding is not used.
        v_var : array-like, shape (3,), optional
            Variance of the velocity measurement noise. Required for ``v``.
        hdg : float, optional
            Heading measurement. I.e., the yaw angle of the 'body' frame relative to the
            assumed 'navigation' frame ('NED' or 'ENU') specified during initialization.
            If ``None``, compass aiding is not used. See ``head_degrees`` for units.
        hdg_var : float, optional
            Variance of heading measurement noise. Units must be compatible with ``hdg``.
            See ``hdg_degrees`` for units. Required for ``hdg``.
        hdg_degrees : bool, default False
            Specifies whether the unit of ``hdg`` and ``hdg_var`` are in degrees and degrees^2,
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

        f = np.asarray(f, dtype=float)
        w = np.asarray(w, dtype=float)

        if degrees:
            w = (np.pi / 180.0) * w

        # Project state and covariance ahead
        self._project_ahead(self._dt, f, w)

        # Update state and covariance with aiding measurements
        self._aiding_update_vel(v, v_var)
        self._aiding_update_head(hdg, hdg_var, hdg_degrees)
        self._aiding_update_g_ref(f, g_var, g_ref)

        # Reset state estimates (regulating error state to zero)
        self._reset()

        self._f = f
        self._w = w
        self._R_nm = self._att.as_matrix()  # avoiding repeated calls

        return self
