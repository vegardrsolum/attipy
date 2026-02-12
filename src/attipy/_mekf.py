from turtle import pos
from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._kalman import _kalman_update_scalar, _kalman_update_sequential
from ._statespace import (
    ATT_IDX,
    BA_IDX,
    BG_IDX,
    POS_IDX,
    VEL_IDX,
    _dyawda,
    _measurement_matrix,
    _process_noise_cov,
    _state_transition,
    _update_state_transition,
)
from ._transforms import _yaw_from_quat
from ._vectorops import _skew_symmetric as S


def _gref_nav(nav_frame) -> NDArray[np.float64]:
    """
    Gravity reference unity vector in the navigation frame ('NED' or 'ENU').

    Parameters
    ----------
    nav_frame : {'NED', 'ENU'}
        Navigation frame.

    Returns
    -------
    NDArray[np.float64], shape (3,)
        Gravity reference vector expressed in the navigation frame.
    """
    if nav_frame.lower() == "ned":
        gref_n = np.array([0.0, 0.0, 1.0])
    elif nav_frame.lower() == "enu":
        gref_n = np.array([0.0, 0.0, -1.0])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return gref_n


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


def _signed_smallest_angle(angle: float) -> float:
    """
    Convert the given angle to the smallest signed angle between [-pi., pi) radians.

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        The smallest angle between [-pi, pi] radians.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


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
        Initial accelerometer bias estimate (bax, bay, baz) in m/s^2. Defaults to zero bias.
    bg : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    w : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial angular rate estimate (wx, wy, wz) in rad/s expressed in the body frame.
        Defaults to zero angular rate (stationary).
    P : array_like, shape (15, 15), default 1e-6 * np.eye(15)
        Initial error covariance matrix estimate. Defaults to a small diagonal matrix
        (1e-6 * np.eye(15)). The order of the (error) states is: dx = (dp, dv, da, dba, dbg),
        where dp is the position error, dv is the velocity error, da is the attitude
        error (3-parameter 2xGibbs vector), dba is the accelerometer bias error,
        and dbg is the gyroscope bias error.
    g : float, default 9.80665
        The gravitational acceleration. Default is the 'standard gravity' 9.80665.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like navigation frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up).
    acc_noise_density : float, default 0.001
        Accelerometer noise density (velocity random walk) in (m/s)/√Hz. Defaults
        to 0.001 (typical value for low-cost MEMS IMUs).
    acc_bias_stability : float, default 0.0005
        Accelerometer bias stability (1-sigma) in m/s^2. Defaults to 0.0005 (typical
        value for low-cost MEMS IMUs).
    acc_bias_corr_time : float, default 50.0
        Accelerometer bias correlation time in seconds. Defaults to 50.0 s.
    gyro_noise_density : float, default 0.0001
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.0001 (typical value for low-cost MEMS IMUs).
    gyro_bias_stability : float, default 0.00005
        Gyroscope bias stability (1-sigma) in rad/s. Defaults to 0.00005 (typical
        value for low-cost MEMS IMUs).
    gyro_bias_corr_time : float, default 50.0
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    """

    _I15: NDArray[np.float64] = np.eye(15)

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
        P: ArrayLike = 1e-6 * np.eye(15),
        g: float = 9.80665,
        nav_frame: str = "NED",
        acc_noise_density: float = 0.001,
        acc_bias_stability: float = 0.0005,
        acc_bias_corr_time: float = 50.0,
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
        self._abs = acc_bias_stability  # accelerometer bias stability
        self._abc = acc_bias_corr_time  # accelerometer bias correlation time
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._att_nb = att if isinstance(att, Attitude) else Attitude(att)
        self._R_nb = self._att_nb.as_matrix()  # avoiding repeated calls
        self._p_n = np.asarray_chkfinite(pos).reshape(3).copy()
        self._v_n = np.asarray_chkfinite(vel).reshape(3).copy()
        self._a_n = np.asarray_chkfinite(acc).reshape(3).copy()
        self._ba_b = np.asarray_chkfinite(ba).reshape(3).copy()
        self._bg_b = np.asarray_chkfinite(bg).reshape(3).copy()
        self._f_b = self._R_nb.T @ (self._a_n - self._g_n)
        self._w_b = np.asarray_chkfinite(w).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(15, 15).copy()
        self._dx = np.zeros(15, dtype=np.float64)

        # Discretized state space model (updated each time step)
        self._phi = _state_transition(
            self._dt, self._f_b, self._w_b, self._R_nb, self._abc, self._gbc
        )
        self._Q = _process_noise_cov(
            self._dt, self._vrw, self._arw, self._abs, self._abc, self._gbs, self._gbc
        )
        self._dhdx = _measurement_matrix(self._att_nb._q)

    @property
    def attitude(self) -> Attitude:
        """Attitude estimate (no copy)."""
        return self._att_nb

    @property
    def position(self) -> NDArray[np.float64]:
        """
        Copy of the position estimate (m) expressed in the navigation frame.
        """
        return self._p_n.copy()

    @property
    def velocity(self) -> NDArray[np.float64]:
        """
        Copy of the linear velocity estimate (m/s) expressed in the navigation frame.
        """
        return self._v_n.copy()

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
        return self._bg_b.copy()

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
        self._dhdx[6:7, ATT_IDX] = _dyawda(q_nb)
        return self._dhdx[6]

    def _reset(self) -> None:
        """
        Reset state (regulating error-state to zero).
        """

        if not self._dx.any():
            return

        self._p_n[:] += self._dx[POS_IDX]
        self._v_n[:] += self._dx[VEL_IDX]
        self._att_nb._correct_da(self._dx[ATT_IDX])
        self._ba_b[:] += self._dx[BA_IDX]
        self._bg_b[:] += self._dx[BG_IDX]
        self._dx[:] = 0.0

    def _aiding_update_pos(self, pos_meas, pos_var):
        """
        Update with position vector aiding measurement.
        """

        if pos_meas is None:
            return None

        if pos_var is None:
            raise ValueError("'pos_var' not provided.")

        dz = pos_meas - self._p_n
        dhdx = self._dhdx_pos()
        _kalman_update_sequential(self._dx, self._P, dz, pos_var, dhdx, self._I15)

    def _aiding_update_vel(self, vel_meas, vel_var):
        """
        Update with velocity vector aiding measurement.
        """

        if vel_meas is None:
            return None

        if vel_var is None:
            raise ValueError("'vel_var' not provided.")

        dz = vel_meas - self._v_n
        dhdx = self._dhdx_vel()
        _kalman_update_sequential(self._dx, self._P, dz, vel_var, dhdx, self._I15)

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
        dz = _signed_smallest_angle(yaw_meas - yaw)
        dhdx = self._dhdx_yaw(self._att_nb._q)
        _kalman_update_scalar(self._dx, self._P, dz, yaw_var, dhdx, self._I15)

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
        Update state estimates with IMU and aiding measurements.

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

        if degrees:
            w = (np.pi / 180.0) * np.asarray(w)

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
        self._w_b[:] = w - self._bg_b
        self._f_b[:] = f - self._ba_b
        self._a_n[:] = self._R_nb @ self._f_b + self._g_n
        _update_state_transition(self._phi, self._dt, self._f_b, self._w_b, self._R_nb)

        return self


class MiniMEKF:
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
    bg : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    w : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial angular rate estimate (wx, wy, wz) in rad/s expressed in the body frame.
        Defaults to zero angular rate (stationary).
    P : array_like, shape (6, 6), default 1e-6 * np.eye(6)
        Initial error covariance matrix estimate. Defaults to a small diagonal matrix
        (1e-6 * np.eye(6)). The order of the (error) states is: dx = (da, dbg),
        where da is the attitude error (3-parameter 2xGibbs vector), and dbg is
        the gyroscope bias error.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like navigation frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up).
    gyro_noise_density : float, default 0.0001
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.0001 (typical value for low-cost MEMS IMUs).
    gyro_bias_stability : float, default 0.00005
        Gyroscope bias stability (1-sigma) in rad/s. Defaults to 0.00005 (typical
        value for low-cost MEMS IMUs).
    gyro_bias_corr_time : float, default 50.0
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    """

    _I6: NDArray[np.float64] = np.eye(6)

    _state_idx = np.r_[ATT_IDX, BG_IDX]

    def __init__(
        self,
        fs: float,
        att: Attitude | ArrayLike,
        bg: ArrayLike = (0.0, 0.0, 0.0),
        w: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = 1e-6 * np.eye(6),
        nav_frame: str = "NED",
        gyro_noise_density: float = 0.0001,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._nav_frame = nav_frame.lower()
        self._gref_n = _gref_nav(self._nav_frame)

        # IMU noise parameters
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._att_nb = att if isinstance(att, Attitude) else Attitude(att)
        self._R_nb = self._att_nb.as_matrix()  # avoiding repeated calls
        self._bg_b = np.asarray_chkfinite(bg).reshape(3).copy()
        self._w_b = np.asarray_chkfinite(w).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(6, 6).copy()
        self._dx = np.zeros(6, dtype=np.float64)

        # Discretized state space model (updated each time step)
        f_b = np.zeros(3)
        vrw = 0.0
        abs = 0.0
        abc = 50.0
        self._phi = _state_transition(
            self._dt, f_b, self._w_b, self._R_nb, abc, self._gbc
        )[np.ix_(self._state_idx, self._state_idx)]
        self._Q = _process_noise_cov(
            self._dt, vrw, self._arw, abs, abc, self._gbs, self._gbc
        )[np.ix_(self._state_idx, self._state_idx)]

        self._dhdx = np.zeros((4, 6))
        self._dhdx[0:1, 0:3] = _dyawda(self._att_nb._q)
        self._dhdx[1:4, 0:3] = S(self._R_nb.T @ self._gref_n)

    @property
    def attitude(self) -> Attitude:
        """Attitude estimate (no copy)."""
        return self._att_nb

    @property
    def bias_gyro(self) -> NDArray[np.float64]:
        """
        Copy of the gyroscope bias estimate (rad/s) expressed in the body frame.
        """
        return self._bg_b.copy()

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

    def _dhdx_yaw(self, q_nb):
        """
        Heading (yaw angle) part of the measurement matrix, shape (6,).
        """
        self._dhdx[0:1, 0:3] = _dyawda(q_nb)
        return self._dhdx[0]

    def _dhdx_gref(self, R_nb):
        """
        Gravity reference vector part of the measurement matrix, shape (3, 6).
        """
        self._dhdx[1:4, 0:3] = S(R_nb.T @ self._gref_n)
        return self._dhdx[1:4]

    def _reset(self) -> None:
        """
        Reset state (regulating error-state to zero).
        """

        if not self._dx.any():
            return

        self._att_nb._correct_da(self._dx[0:3])
        self._bg_b[:] += self._dx[3:6]
        self._dx[:] = 0.0

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
        dz = _signed_smallest_angle(yaw_meas - yaw)
        dhdx = self._dhdx_yaw(self._att_nb._q)
        _kalman_update_scalar(self._dx, self._P, dz, yaw_var, dhdx, self._I6)

    def _aiding_update_gref(self, f_b, gref_var):
        """
        Update with gravity reference vector aiding measurement.
        """

        if f_b is None:
            return None

        if gref_var is None:
            raise ValueError("'gref_var' not provided.")

        R_nb = self._att_nb.as_matrix()

        gref_meas = -f_b / np.linalg.norm(f_b)
        dz = gref_meas - R_nb.T @ self._gref_n
        dhdx = self._dhdx_gref(R_nb)
        _kalman_update_sequential(self._dx, self._P, dz, gref_var, dhdx, self._I6)

    def _project_ahead(self):
        """
        Project state and covariance estimates ahead.
        """

        # Attitude (dead reckoning)
        self._att_nb._project_ahead(self._w_b, self._dt)

        # Covariance
        self._P[:] = self._phi @ self._P @ self._phi.T + self._Q

    @staticmethod
    @njit  # type: ignore[misc]
    def _update_state_transition(
        phi: NDArray[np.float64],
        dt: float,
        w_b: NDArray[np.float64],
    ):
        """
        Update the state transition matrix, phi, in place:

            phi[0:3, 0:3] = I - dt * S(w_b)

        Parameters
        ----------
        phi : ndarray, shape (6, 6)
            State transition matrix to be updated in place.
        dt : float
            Time step.
        w_b : ndarray, shape (3,)
            Angular rate measurement (bias corrected) in body frame.

        Notes
        -----
        Assuming the first order approximation:

            phi = I + dt * dfdx

        where dfdx denotes the linearized state matrix.
        """
        wx, wy, wz = w_b

        # phi[0:3, 0:3] = np.eye(3) - dt * S(w_b)
        phi[0, 1] = dt * wz
        phi[0, 2] = -dt * wy
        phi[1, 0] = -dt * wz
        phi[1, 2] = dt * wx
        phi[2, 0] = dt * wy
        phi[2, 1] = -dt * wx

    def update(
        self,
        f: ArrayLike,
        w: ArrayLike,
        degrees: bool = False,
        yaw: float | None = None,
        yaw_var: float | None = None,
        yaw_degrees: bool = False,
        gref: bool = True,
        gref_var: ArrayLike | None = (0.01, 0.01, 0.01),
    ) -> Self:
        """
        Update state estimates with IMU and aiding measurements.

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

        if degrees:
            w = (np.pi / 180.0) * np.asarray(w)

        # Project (a priori) state and covariance estimates ahead
        self._project_ahead()

        # Update (a posteriori) state and covariance estimates with aiding measurements
        self._aiding_update_gref(f if gref else None, gref_var)
        self._aiding_update_yaw(yaw, yaw_var, yaw_degrees)

        # Reset state estimates (regulating error-state to zero)
        self._reset()

        # Update model
        self._R_nb[:] = self._att_nb.as_matrix()  # avoiding repeated calls
        self._w_b[:] = w - self._bg_b
        self._update_state_transition(self._phi, self._dt, self._w_b)

        return self
