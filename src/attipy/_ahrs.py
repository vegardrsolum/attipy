from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._statespace import _dyawda, _measurement_matrix
from ._statespace import _process_noise_cov as _setup_Q
from ._statespace import _state_transition as _setup_phi
from ._statespace import _update_state_transition as _update_phi
from ._transforms import _quat_from_gibbs2, _yaw_from_quat


def _gravity_nav(g, nav_frame) -> NDArray[np.float64]:
    """
    Gravity vector in the navigation frame ('NED' or 'ENU').
    """
    if nav_frame.lower() == "ned":
        g_n = np.array([0.0, 0.0, g])
    elif nav_frame.lower() == "enu":
        g_n = np.array([0.0, 0.0, -g])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return g_n


def _ssa(angle: float, degrees: bool = False) -> float:
    """
    Convert the given angle to the smallest signed angle between [-pi., pi) radians.

    Parameters
    ----------
    angle : float
        Angle value.
    degrees : bool, default False
        Specifies whether ``angle`` is given degrees or radians (default).

    Returns
    -------
    float
        The smallest angle between [-pi, pi] radians (or [-180., 180) degrees).
    """
    base = 180.0 if degrees else np.pi
    return (angle + base) % (2.0 * base) - base


@njit  # type: ignore[misc]
def _kalman_update(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    
    n = x.size

    for i in range(z.shape[0]):
        h_i = H[i, :]
        z_i = z[i]
        v_i = var[i]

        # Kalman gain
        PHt = P @ h_i
        S = h_i @ PHt + v_i
        K = PHt / S  # shape (n,)

        # State update
        hx = 0.0
        for j in range(n):
            hx += h_i[j] * x[j]
        x += K * (z_i - hx)

        # Covariance update (Joseph form)
        A = I_ - np.outer(K, h_i)
        P = A @ P @ A.T + v_i * np.outer(K, K)

    return x, P


# @njit  # type: ignore[misc]
# def _kalman_update(
#     x: NDArray[np.float64],
#     P: NDArray[np.float64],
#     z: NDArray[np.float64],
#     var: NDArray[np.float64],
#     H: NDArray[np.float64],
#     I_: NDArray[np.float64],
# ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

#     for i in range(z.shape[0]):
#         h_i = H[i, :]
#         z_i = z[i]
#         v_i = var[i]

#         # Kalman gain
#         PHt = P @ h_i
#         S = h_i @ PHt + v_i
#         K = PHt / S  # shape (n,)

#         # State update
#         x += K * (z_i - h_i @ x)

#         # Covariance update (Joseph form)
#         A = I_ - np.outer(K, h_i)
#         P = A @ P @ A.T + v_i * np.outer(K, K)

#     return x, P


# @njit  # type: ignore[misc]
# def _update_dx_P(
#     dx: NDArray[np.float64],
#     P: NDArray[np.float64],
#     dz: NDArray[np.float64],
#     var: NDArray[np.float64],
#     H: NDArray[np.float64],
# ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     """
#     Update state and covariance estimates with a series of measurements.
#     """
#     n = dx.shape[0]

#     # Preallocation
#     PH = np.empty(n, dtype=np.float64)  # P @ H.T
#     K = np.empty(n, dtype=np.float64)

#     for i in range(dz.shape[0]):
#         z = dz[i]
#         h = H[i, :]
#         v = var[i]

#         # P @ H.T
#         for a in range(n):
#             s = 0.0
#             for b in range(n):
#                 s += P[a, b] * h[b]
#             PH[a] = s

#         # S = H @ P @ H.T + v, hx = H @ dx
#         S = v
#         hx = 0.0
#         for a in range(n):
#             S += h[a] * PH[a]
#             hx += h[a] * dx[a]

#         # Precalculate inverse, since multiply is faster than divide
#         invS = 1.0 / S

#         # Kalman gain: K = P @ H.T / S
#         for a in range(n):
#             K[a] = PH[a] * invS

#         # State update: dx += K * (dz - H @ dx)
#         r = z - hx
#         for a in range(n):
#             dx[a] += K[a] * r

#         # Covariance update: P = P - K H P - P H.T K.T + S K K.T (Joseph, expanded)
#         for a in range(n):
#             Ka = K[a]
#             PHa = PH[a]
#             for b in range(n):
#                 P[a, b] = P[a, b] - Ka * PH[b] - PHa * K[b] + S * Ka * K[b]

#     return dx, P


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
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    v_n : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial linear velocity estimate (vx, vy, vz) in m/s expressed in the navigation
        frame. Defaults to zero velocity (stationary).
    w_b : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial angular rate estimate (wx, wy, wz) in rad/s expressed in the body frame.
        Defaults to zero angular rate (stationary).
    a_n : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial linear acceleration estimate (ax, ay, az) in m/s^2 expressed in
        the navigation frame. Defaults to zero linear acceleration (stationary).
    P : array_like, shape (9, 9), default 1e-6 * np.eye(9)
        Initial error covariance matrix estimate. Defaults to a small diagonal matrix
        (1e-6 * np.eye(9)). The order of the (error) states is: dx = (da, dbg, dv),
        where da is the attitude error (3-parameter 2xGibbs vector), dbg is the
        gyroscope bias error, and dv is the velocity error.
    g : float, default 9.80665
        The gravitational acceleration. Default is the 'standard gravity' 9.80665.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like navigation frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up).
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

    _I9x9 = np.eye(9)
    _I3x3 = np.eye(3)
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

        # Discretized state space model (updated each time step)
        self._phi = _setup_phi(self._dt, self._f_b, self._w_b, self._R_nb, self._gbc)
        self._Q = _setup_Q(self._dt, self._vrw, self._arw, self._gbs, self._gbc)
        self._dhdx = _measurement_matrix(self._att_nb._q)

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
        Velocity part of the measurement matrix.
        """
        return self._dhdx[0:3]

    def _dhdx_yaw(self, q_nb):
        """
        Heading (yaw angle) part of the measurement matrix.
        """
        self._dhdx[3:4, 0:3] = _dyawda(q_nb)
        return self._dhdx[3:4]

    def _reset(self) -> None:
        """
        Reset state (regulating error state to zero).
        """
        dx = self._dx

        if not dx.any():
            return

        self._att_nb._correct_dq(_quat_from_gibbs2(dx[0:3]))
        self._bg_b[:] = self._bg_b + dx[3:6]
        self._v_n[:] = self._v_n + dx[6:9]
        self._dx[:] = np.zeros(dx.size)

    def _aiding_update_vel(self, v_meas, v_var):
        """
        Update with velocity vector aiding measurement.
        """
        dx, P = self._dx, self._P

        if v_meas is None:
            return dx, P

        if v_var is None:
            raise ValueError("'vel_var' not provided.")

        dz = v_meas - self._v_n
        var = np.asarray(v_var, dtype=float)
        dhdx = self._dhdx_vel()

        dx[:], P[:] = _kalman_update(dx, P, dz, var, dhdx, self._I9x9)

    def _aiding_update_yaw(self, yaw_meas, yaw_var, yaw_degrees):
        """
        Update with heading aiding measurement.
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
        dx[:], P[:] = _kalman_update(dx, P, dz, var, dhdx, self._I9x9)

    def _project_ahead(self):
        """
        Project state and covariance estimates ahead.
        """

        # Velocity (dead reckoning)
        self._v_n[:] += self._a_n * self._dt

        # Attitude (dead reckoning)
        dtheta = self._w_b * self._dt
        self._att_nb._correct_dtheta(dtheta)

        # Covariance
        self._P[:] = self._phi @ self._P @ self._phi.T + self._Q

    def _update_state(self, f_b: NDArray[np.float64], w_b: NDArray[np.float64]) -> None:
        """
        Update state vectors and state space matrices.
        """
        self._R_nb[:] = self._att_nb.as_matrix()  # avoiding repeated calls
        self._f_b[:] = f_b
        self._a_n[:] = self._R_nb @ self._f_b + self._g_n
        self._w_b[:] = w_b - self._bg_b
        _update_phi(self._phi, self._dt, self._f_b, self._w_b, self._R_nb, self._I3x3)

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
            Specific force (i.e., acceleration + gravity) measurement (fx, fy, fz)
            in m/s^2.
        w_b : array_like, shape (3,)
            Angular rate measurement (wx, wy, wz) in rad/s (default) or deg/s. See
            ``degrees`` parameter for units.
        degrees : bool, default False
            Specifies whether the unit of the rotation rate, ``w_b``, are deg/s
            or rad/s (default).
        v_n : array_like, shape (3,), optional
            Velocity measurement (vx, vy, vz) in m/s. If ``None``, velocity aiding
            is not used.
        v_var : array_like, shape (3,), optional
            Variance of the velocity measurement noise in (m/s)^2. Required for ``v_n``.
        yaw : float, optional
            Heading (yaw angle) measurement in rad (default) or deg. See ``yaw_degrees``
            for units. If ``None``, heading aiding is not used.
        yaw_var : float, optional
            Variance of heading (yaw angle) measurement noise in rad^2 (default)
            or deg^2. Units must be compatible with ``yaw``. See ``yaw_degrees``
            for units. Required for ``yaw``.
        yaw_degrees : bool, default False
            Specifies whether the unit of ``yaw`` and ``yaw_var`` are deg and deg^2
            or rad and rad^2 (default).

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
