from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._quatops import _quatprod
from ._vectorops import _normalize, _skew_symmetric


def _gravity_nav(g, nav_frame) -> NDArray[np.float64]:
    """
    Gravity vector direction in the navigation frame (NED or ENU).
    """
    if nav_frame == "ned":
        g_n = np.array([0.0, 0.0, g])
    elif nav_frame == "enu":
        g_n = np.array([0.0, 0.0, -g])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return g_n


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


def _state_matrix(f_b_corr, w_b_corr, R_nb, gbc) -> NDArray[np.float64]:
    """
    Setup linearized state matrix, dfdx.
    """

    beta_gyro = 1.0 / gbc

    S = _skew_symmetric  # alias skew symmetric matrix

    # State transition matrix
    dfdx = np.zeros((9, 9))
    dfdx[0:3, 0:3] = -S(w_b_corr)  # NB! update each time step
    dfdx[0:3, 3:6] = -np.eye(3)
    dfdx[3:6, 3:6] = -beta_gyro * np.eye(3)
    dfdx[6:9, 0:3] = -R_nb @ S(f_b_corr)  # NB! update each time step

    return dfdx


def _wn_input_matrix(R_nb):
    """Setup linearized (white noise) input matrix, dfdw."""

    # Input (white noise) matrix
    dfdw = np.zeros((9, 9))
    dfdw[0:3, 0:3] = -np.eye(3)
    dfdw[3:6, 3:6] = np.eye(3)
    dfdw[6:9, 6:9] = -R_nb  # NB! update each time step

    return dfdw


def _wn_psd_matrix(vrw, arw, gbs, gbc) -> NDArray[np.float64]:
    """Setup white noise (process noise) power spectral density matrix, W."""

    # White noise power spectral density matrix
    W = np.eye(9)
    W[0:3, 0:3] *= arw**2
    W[3:6, 3:6] *= 2.0 * gbs**2 / gbc
    W[6:9, 6:9] *= vrw**2

    return W


@njit  # type: ignore[misc]
def _yaw_from_quat(q_nb: NDArray[np.float64]) -> float:
    """
    Compute yaw angle from unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    float
        Yaw angle in radians.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.251, John Wiley & Sons, 2021.
    """
    qw, qx, qy, qz = q_nb
    u_y = 2.0 * (qx * qy + qz * qw)
    u_x = 1.0 - 2.0 * (qy**2 + qz**2)
    return np.arctan2(u_y, u_x)  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _dyawda(q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute yaw angle gradient wrt to the scaled Gibbs vector.

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
    qw, qx, qy, qz = q_nb
    u_y = 2.0 * (qx * qy + qz * qw)
    u_x = 1.0 - 2.0 * (qy**2 + qz**2)
    u = u_y / u_x

    duda_scale = 1.0 / u_x**2
    duda_x = -(qw * qy) * (1.0 - 2.0 * qw**2) - (2.0 * qw**2 * qx * qz)
    duda_y = (qw * qx) * (1.0 - 2.0 * qz**2) + (2.0 * qw**2 * qy * qz)
    duda_z = qw**2 * (1.0 - 2.0 * qy**2) + (2.0 * qw * qx * qy * qz)
    duda = duda_scale * np.array([duda_x, duda_y, duda_z])

    dyawda = 1.0 / (1.0 + u**2) * duda

    return dyawda  # type: ignore[no-any-return]


def _measurement_matrix(q_nb) -> None:
    """Setup linearized measurement matrix, dhdx."""
    dhdx = np.zeros((7, 9))
    dhdx[0:3, 6:9] = np.eye(3)  # velocity
    dhdx[3:4, 0:3] = _dyawda(q_nb)  # heading
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

    The internal fusion filter is an (error-state) multiplicative extended Kalman
    filter (MEKF).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    q_nb : Attitude or array_like, shape (4,), default (1.0, 0.0, 0.0, 0.0)
        Initial attitude estimate represented as a unit quaternion (qw, qx, qy, qz)
        or an Attitude object. Defaults to no rotation (identity quaternion).
    bg_b : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial gyroscope bias estimate (bgx, bgy, bgz). Defaults to zero bias.
    v_n : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial linear velocity estimate (vx, vy, vz) expressed in the navigation
        frame. Defaults to zero velocity (stationary).
    w_b : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial angular rate estimate (wx, wy, wz) expressed in the body frame.
        Defaults to zero angular rate (stationary).
    a_n : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial linear acceleration estimate (ax, ay, az) expressed in the navigation
        frame. Defaults to zero linear acceleration (stationary).
    P : array_like, shape (9, 9), default 1e-6 * np.eye(9)
        Initial error covariance matrix estimate. Defaults to a small diagonal matrix
        (1e-6 * np.eye(9)). The internal error-state Kalman filter's state vector
        is ordered as: dx = (da, dbg, dv), where da is the attitude error (3-parameter
        2xGibbs vector), dbg is the gyroscope bias error, and dv is the velocity
        error.
    g : float, default 9.80665
        The gravitational acceleration. Default is the 'standard gravity' 9.80665.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like navigation frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up). The body's (or IMU/AHRS
        sensor's) degrees of freedom will be expressed relative to this frame.
        Furthermore, the aiding heading angle is also interpreted relative to this
        frame according to the right-hand rule.
    acc_noise_density : float, default 0.001
        Accelerometer noise density (velocity random walk) in (m/s)/√Hz. Defaults
        to 0.001 (typical value for low-cost MEMS IMUs).
    gyro_noise_density : float, default 0.0001
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.0001 (typical value for low-cost MEMS IMUs).
    gyro_bias_stability : float, default 0.00005
        Gyroscope bias stability (1-sigma) in rad/s. Defaults to 0.00005 (typical
        value for low-cost MEMS IMUs).
    bias_corr_time : float, default 50.0
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    """

    _I = np.eye(9)
    _dx = np.zeros(9)  # error state estimate (da, dbg, dv) (always zero after reset)
    _dq = np.array([1.0, 0.0, 0.0, 0.0])  # error quaternion preallocation

    def __init__(
        self,
        fs: float,
        q_nb: ArrayLike | Attitude = (1.0, 0.0, 0.0, 0.0),
        bg_b: ArrayLike = (0.0, 0.0, 0.0),
        v_n: ArrayLike = (0.0, 0.0, 0.0),
        w_b: ArrayLike = (0.0, 0.0, 0.0),
        a_n: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = 1e-6 * np.eye(9),
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
        self._g_n = _gravity_nav(self._g, self._nav_frame)

        # IMU noise parameters
        self._vrw = acc_noise_density  # velocity random walk
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._att_nb = q_nb if isinstance(q_nb, Attitude) else Attitude(q_nb)
        self._R_nb = self._att_nb.as_matrix()  # avoiding repeated calls
        self._bg_b = np.asarray_chkfinite(bg_b).reshape(3).copy()
        self._v_n = np.asarray_chkfinite(v_n).reshape(3).copy()
        self._w_b = np.asarray_chkfinite(w_b).reshape(3).copy()
        self._a_n = np.asarray_chkfinite(a_n).reshape(3).copy()
        self._f_b = self._R_nb.T @ (self._a_n - self._g_n)
        self._P = np.asarray_chkfinite(P).reshape(9, 9).copy()

        # Continuous time state space model (updated each time step)
        # TODO: avoid continuous time state space by computing phi and Q directly
        self._dfdx = _state_matrix(self._f_b, self._w_b, self._R_nb, self._gbc)
        self._dfdw = _wn_input_matrix(self._R_nb)
        self._dhdx = _measurement_matrix(self._att_nb._q)
        self._W = _wn_psd_matrix(self._vrw, self._arw, self._gbs, self._gbc)

        # Discretized state space model (updated each time step)
        self._phi = self._I + self._dt * self._dfdx  # first-order approximation
        self._Q = self._dt * self._dfdw @ self._W @ self._dfdw.T

    @property
    def attitude(self) -> Attitude:
        """
        Attitude estimate (no copy).
        """
        return self._att_nb

    @property
    def q_nb(self) -> NDArray[np.float64]:
        """
        Copy of the attitude estimate (represented as a unit quaternion).
        """
        return self._att_nb._q.copy()

    @property
    def bg_b(self) -> NDArray[np.float64]:
        """
        Copy of the gyroscope bias estimate (rad/s) expressed in the body frame.
        """
        return self._bg_b.copy()

    @property
    def v_n(self) -> NDArray[np.float64]:
        """
        Copy of the velocity estimate (m/s) expressed in the navigation frame.
        """
        return self._v_n.copy()

    @property
    def w_b(self) -> NDArray[np.float64]:
        """
        Copy of the bias corrected angular rate measurement (rad/s) expressed in
        the body frame.
        """
        return self._w_b.copy()

    @property
    def f_b(self) -> NDArray[np.float64]:
        """
        Copy of the specific force measurement (m/s^2) expressed in the body frame.
        """
        return self._f_b.copy()


    @property
    def a_n(self) -> NDArray[np.float64]:
        """
        Copy of the linear acceleration estimate (m/s^2) expressed in the navigation frame.
        """
        return self._a_n.copy()

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Copy of the error covariance matrix estimate.
        """
        return self._P.copy()

    def _dhdx_vel(self):
        """
        Velocity measurement matrix.
        """
        return self._dhdx[0:3]

    def _dhdx_yaw(self, q_nb):
        """
        Heading (yaw angle) measurement matrix.
        """
        self._dhdx[3:4, 0:3] = _dyawda(q_nb)
        return self._dhdx[3:4]

    def _reset(self) -> None:
        """Reset state (regulating error state to zero)."""
        dx = self._dx

        if not dx.any():
            return

        da = dx[0:3]
        self._dq[:] = (2.0, *da) / np.sqrt(4.0 + da.T @ da)
        self._att_nb._q[:] = _normalize(_quatprod(self._att_nb._q, self._dq))
        self._bg_b[:] = self._bg_b + dx[3:6]
        self._v_n[:] = self._v_n + dx[6:9]
        self._dx[:] = np.zeros(dx.size)

    def _aiding_update_vel(self, v_meas, v_var):
        """
        Update with velocity vector measurement.
        """
        dx, P = self._dx, self._P

        if v_meas is None:
            return dx, P

        if v_var is None:
            raise ValueError("'vel_var' not provided.")

        dz = v_meas - self._v_n
        var = np.asarray(v_var, dtype=float)
        dhdx = self._dhdx_vel()

        dx[:], P[:] = _update_dx_P(dx, P, dz, var, dhdx, self._I)

    def _aiding_update_yaw(self, yaw_meas, yaw_var, yaw_degrees):
        """
        Update with heading measurement.
        """
        dx, P = self._dx, self._P

        if yaw_meas is None:
            return dx, P

        if yaw_var is None:
            raise ValueError("'yaw_var' not provided.")

        if yaw_degrees:
            yaw_meas = (np.pi / 180.0) * yaw_meas
            yaw_var = (np.pi / 180.0) ** 2 * yaw_var

        yaw = _yaw_from_quat(self._att_nb._q)  # heading estimate

        var = np.asarray([yaw_var], dtype=float)
        dz = np.asarray([_ssa(yaw_meas - yaw, degrees=False)], dtype=float)
        dhdx = self._dhdx_yaw(self._att_nb._q)
        dx[:], P[:] = _update_dx_P(dx, P, dz, var, dhdx, self._I)

    def _project_ahead(self):
        """
        Project state and covariance ahead.
        """

        # Velocity (dead reckoning)
        self._v_n[:] += self._a_n * self._dt

        # Attitude (dead reckoning)
        dtheta = self._w_b * self._dt
        self._att_nb.update(dtheta, degrees=False)

        # Covariance
        self._P[:] = self._phi @ self._P @ self._phi.T + self._Q

    def _update_state(self, f_b: NDArray[np.float64], w_b: NDArray[np.float64]) -> None:
        """
        Update states and state space matrices.
        """

        # States
        self._R_nb[:] = self._att_nb.as_matrix()  # avoiding repeated calls
        self._f_b[:] = f_b
        self._a_n[:] = self._R_nb @ self._f_b + self._g_n
        self._w_b[:] = w_b - self._bg_b

        # Continuous time state space
        S = _skew_symmetric
        self._dfdx[0:3, 0:3] = -S(self._w_b)
        self._dfdx[6:9, 0:3] = -self._R_nb @ S(self._f_b)
        self._dfdw[6:9, 6:9] = -self._R_nb

        # Discretized state space
        self._phi[:] = self._I + self._dt * self._dfdx  # first-order approximation
        self._Q[:] = self._dt * self._dfdw @ self._W @ self._dfdw.T

    def update(
        self,
        f_b: ArrayLike,
        w_b: ArrayLike,
        degrees: bool = False,
        v_n: ArrayLike | None = (0.0, 0.0, 0.0),
        v_var: ArrayLike | None = (100.0, 100.0, 100.0),
        yaw: float | None = None,
        yaw_var: float | None = None,
        yaw_degrees: bool = False,
    ) -> Self:
        """
        Update the AHRS state estimates with IMU and aiding measurements.

        Parameters
        ----------
        f_b : array_like, shape (3,)
            Specific force (i.e., acceleration + gravity) measurement (fx, fy, fz).
        w_b : array_like, shape (3,)
            Angular rate measurement (wx, wy, wz).
        degrees : bool, default False
            Specifies whether the unit of the rotation rate, ``w``, are in degrees
            or radians (default).
        v_n : array_like, shape (3,), optional
            Velocity measurement (vx, vy, vz). If ``None``, velocity aiding is not used.
        v_var : array_like, shape (3,), optional
            Variance of the velocity measurement noise. Required for ``v_n``.
        yaw : float, optional
            Heading (yaw angle) measurement. See ``yaw_degrees`` for units. If ``None``,
            heading aiding is not used.
        yaw_var : float, optional
            Variance of heading (yaw angle) measurement noise. Units must be compatible
            with ``yaw``. See ``yaw_degrees`` for units. Required for ``yaw``.
        yaw_degrees : bool, default False
            Specifies whether the unit of ``yaw`` and ``yaw_var`` are in degrees
            and degrees^2, or radians and radians^2 (default).

        Returns
        -------
        AHRS
            A reference to the instance itself after the update.
        """

        f_b = np.asarray(f_b, dtype=float)
        w_b = np.asarray(w_b, dtype=float)

        if degrees:
            w_b = (np.pi / 180.0) * w_b

        # Project state and covariance estimates ahead (a priori)
        self._project_ahead()

        # Update state and covariance estimates with aiding measurements (a posteriori)
        self._aiding_update_vel(v_n, v_var)
        self._aiding_update_yaw(yaw, yaw_var, yaw_degrees)

        # Reset state estimates (regulating error state estimate to zero)
        self._reset()
        self._update_state(f_b, w_b)

        return self
