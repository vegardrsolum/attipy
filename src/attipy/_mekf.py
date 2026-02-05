from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._kalman import _kalman_update_scalar, _kalman_update_sequential
from ._statespace import (
    _dyawda,
    _measurement_matrix,
    _process_noise_cov,
    _state_transition,
    _update_state_transition,
)
from ._transforms import _yaw_from_quat


def _gravity_nav(g, nav_frame) -> NDArray[np.float64]:
    """
    Gravity vector in the navigation frame ('NED' or 'ENU').

    Parameters
    ----------
    g : float
        Gravitational acceleration in m/s^2.
    nav_frame : {'NED', 'ENU'}
        Navigation frame.

    Returns
    -------
    NDArray[np.float64], shape (3,)
        Gravity vector expressed in the navigation frame.
    """
    if nav_frame.lower() == "ned":
        g_n = np.array([0.0, 0.0, g])
    elif nav_frame.lower() == "enu":
        g_n = np.array([0.0, 0.0, -g])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return g_n


def _signed_smallest_angle(angle: float, degrees: bool = False) -> float:
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


class MEKF:
    """
    Multiplicative extended Kalman filter (MEKF) for position, velocity and attitude
    (PVA) estimation.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    att : Attitude or array_like, shape (4,)
        Initial attitude estimate as an Attitude instance or a unit quaternion,
        (qw, qx, qy, qz).
    pos : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial position estimate (px, py, pz) in m expressed in the navigation frame.
        Defaults to zero position.
    vel : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial linear velocity estimate (vx, vy, vz) in m/s expressed in the navigation
        frame. Defaults to zero velocity (stationary).
    acc : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial linear acceleration estimate (ax, ay, az) in m/s^2 expressed in
        the navigation frame. Defaults to zero linear acceleration (stationary).
    ba : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Accelerometer bias estimate (bax, bay, baz) in m/s^2. Defaults to zero bias.
    bg : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    w : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial angular rate estimate (wx, wy, wz) in rad/s expressed in the body frame.
        Defaults to zero angular rate (stationary).
    P : array_like, shape (12, 12), default 1e-6 * np.eye(12)
        Initial error covariance matrix estimate. Defaults to a small diagonal matrix
        (1e-6 * np.eye(12)). The order of the (error) states is: dx = (dp, dv, da, dbg),
        where dp is the position error, dv is the velocity error, da is the attitude
        error (3-parameter 2xGibbs vector), and dbg is the gyroscope bias error.
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
    gyro_bias_corr_time : float, default 50.0
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    """

    _I12 = np.eye(12)
    _dx = np.zeros(3)

    def __init__(
        self,
        fs: float,
        att: Attitude | ArrayLike,
        pos: ArrayLike = (0.0, 0.0, 0.0),
        vel: ArrayLike = (0.0, 0.0, 0.0),
        acc: ArrayLike = (0.0, 0.0, 0.0),
        ba: ArrayLike = (0.0, 0.0, 0.0),
        bg: ArrayLike = (0.0, 0.0, 0.0),
        w: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = 1e-6 * np.eye(12),
        g: float = 9.80665,
        nav_frame: str = "NED",
        acc_noise_density: float = 0.001,
        gyro_noise_density: float = 0.0001,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
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
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._att_nb = att if isinstance(att, Attitude) else Attitude(att)
        self._R_nb = self._att_nb.as_matrix()  # avoiding repeated calls
        self._epsilon = np.concatenate([pos, vel, bg]).reshape(9)
        self._a_n = np.asarray_chkfinite(acc).reshape(3)
        self._f_b = self._R_nb.T @ (self._a_n - self._g_n)
        self._w_b = np.asarray_chkfinite(w).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(12, 12).copy()

        # Discretized state space model (updated each time step)
        self._phi = _state_transition(
            self._dt, self._f_b, self._w_b, self._R_nb, self._gbc
        )
        self._Q = _process_noise_cov(
            self._dt, self._vrw, self._arw, self._gbs, self._gbc
        )
        self._dhdx = _measurement_matrix(self._att_nb._q)

    @property
    def attitude(self) -> Attitude:
        """
        Attitude estimate (no copy).
        """
        return self._att_nb

    @property
    def position(self) -> NDArray[np.float64]:
        """
        Copy of the position estimate (m) expressed in the navigation frame.
        """
        return self._epsilon[0:3].copy()

    @property
    def velocity(self) -> NDArray[np.float64]:
        """
        Copy of the linear velocity estimate (m/s) expressed in the navigation frame.
        """
        return self._epsilon[3:6].copy()

    @property
    def acceleration(self) -> NDArray[np.float64]:
        """
        Copy of the linear acceleration estimate (m/s^2) expressed in the navigation frame.
        """
        return self._a_n.copy()

    @property
    def bias_gyro(self) -> NDArray[np.float64]:
        """
        Copy of the gyroscope bias estimate (rad/s) expressed in the body frame.
        """
        return self._epsilon[6:9].copy()

    @property
    def bias_acc(self) -> NDArray[np.float64]:
        """
        Copy of the accelerometer bias estimate (m/s^2) expressed in the body frame.
        """
        return self._ba_b.copy()

    @property
    def angular_rate(self) -> NDArray[np.float64]:
        """
        Copy of the bias corrected angular rate measurement (rad/s) expressed in
        the body frame.
        """
        return self._w_b.copy()

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Copy of the error covariance matrix estimate.
        """
        return self._P.copy()

    def _dhdx_pos(self):
        """
        Position part of the measurement matrix, shape (3, 12).
        """
        return self._dhdx[0:3]

    def _dhdx_vel(self):
        """
        Velocity part of the measurement matrix, shape (3, 12).
        """
        return self._dhdx[3:6]

    def _dhdx_yaw(self, q_nb):
        """
        Heading (yaw angle) part of the measurement matrix, shape (12,).
        """
        self._dhdx[6:7, 6:9] = _dyawda(q_nb)
        return self._dhdx[6]

    def _reset(self) -> None:
        """
        Reset state (regulating error-state to zero).
        """
        dx = self._dx

        if not dx.any():
            return

        self._p_n[:] += dx[0:3]
        self._v_n[:] += dx[3:6]
        self._att_nb._correct_da(dx[6:9])
        self._bg_b[:] += dx[9:12]
        self._dx[:] = np.zeros(dx.size)

    def _aiding_update_pos(self, p_meas, p_var):
        """
        Update with position vector aiding measurement.
        """

        if p_meas is None:
            return None

        if p_var is None:
            raise ValueError("'pos_var' not provided.")

        dz = p_meas - self._p_n
        var = p_var
        dhdx = self._dhdx_pos()
        dx = self._dx
        P = self._P

        _kalman_update_sequential(dx, P, dz, var, dhdx, self._I12)

    def _aiding_update_vel(self, v_meas, v_var):
        """
        Update with velocity vector aiding measurement.
        """

        if v_meas is None:
            return None

        if v_var is None:
            raise ValueError("'vel_var' not provided.")

        dz = v_meas - self._v_n
        var = v_var
        dhdx = self._dhdx_vel()
        dx = self._dx
        P = self._P

        _kalman_update_sequential(dx, P, dz, var, dhdx, self._I12)

    def _aiding_update_yaw(self, yaw_meas, yaw_var, yaw_degrees):
        """
        Update with heading aiding measurement.
        """

        if yaw_meas is None:
            return None

        if yaw_var is None:
            raise ValueError("'yaw_var' not provided.")

        if yaw_degrees:
            yaw_meas = (np.pi / 180.0) * yaw_meas
            yaw_var = (np.pi / 180.0) ** 2 * yaw_var

        yaw = _yaw_from_quat(self._att_nb._q)  # heading estimate

        var = yaw_var
        dz = _signed_smallest_angle(yaw_meas - yaw, degrees=False)
        dhdx = self._dhdx_yaw(self._att_nb._q)
        dx = self._dx
        P = self._P

        _kalman_update_scalar(dx, P, dz, var, dhdx, self._I12)

    def _project_ahead(self):
        """
        Project state and covariance estimates ahead.
        """

        # Position (dead reckoning)
        self._p_n[:] += self._v_n * self._dt

        # Velocity (dead reckoning)
        self._v_n[:] += self._a_n * self._dt

        # Attitude (dead reckoning)
        self._att_nb._project_ahead(self._w_b, self._dt)

        # Covariance
        self._P[:] = self._phi @ self._P @ self._phi.T + self._Q

    def update(
        self,
        f: ArrayLike,
        w: ArrayLike,
        degrees: bool = False,
        pos: ArrayLike | None = (0.0, 0.0, 0.0),
        pos_var: ArrayLike | None = (1000.0, 1000.0, 1000.0),
        vel: ArrayLike | None = (0.0, 0.0, 0.0),
        vel_var: ArrayLike | None = (100.0, 100.0, 100.0),
        yaw: float | None = None,
        yaw_var: float | None = None,
        yaw_degrees: bool = False,
    ) -> Self:
        """
        Update the MEKF state estimates with IMU and aiding measurements.

        Parameters
        ----------
        f : array_like, shape (3,)
            Specific force (i.e., acceleration + gravity) measurement (fx, fy, fz)
            in m/s^2.
        w : array_like, shape (3,)
            Angular rate measurement (wx, wy, wz) in rad/s (default) or deg/s. See
            ``degrees`` parameter for units.
        degrees : bool, default False
            Specifies whether the unit of the rotation rate, ``w``, are deg/s
            or rad/s (default).
        pos : array_like, shape (3,), optional
            Position measurement (px, py, pz) in m. If ``None``, position aiding is not used.
        pos_var : array_like, shape (3,), optional
            Variance of the position measurement noise in m^2. Required for ``pos``.
        vel : array_like, shape (3,), optional
            Velocity measurement (vx, vy, vz) in m/s. If ``None``, velocity aiding
            is not used.
        vel_var : array_like, shape (3,), optional
            Variance of the velocity measurement noise in (m/s)^2. Required for ``vel``.
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
        MEKF
            A reference to the instance itself after the update.
        """

        f_b = np.asarray(f, dtype=float)
        w_b = np.asarray(w, dtype=float)

        if degrees:
            w_b = (np.pi / 180.0) * w_b

        # Project (a priori) state and covariance estimates ahead
        self._project_ahead()

        # Update (a posteriori) state and covariance estimates with aiding measurements
        self._aiding_update_pos(pos, pos_var)
        self._aiding_update_vel(vel, vel_var)
        self._aiding_update_yaw(yaw, yaw_var, yaw_degrees)

        # Reset state estimates (regulating error-state to zero)
        self._reset()

        # Update model
        self._R_nb[:] = self._att_nb.as_matrix()  # avoiding repeated calls
        self._w_b[:] = w_b - self._bg_b
        self._f_b[:] = f_b - self._ba_b
        self._a_n[:] = self._R_nb @ self._f_b + self._g_n
        _update_state_transition(self._phi, self._dt, self._f_b, self._w_b, self._R_nb)

        return self
