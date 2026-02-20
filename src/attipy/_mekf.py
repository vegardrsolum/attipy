from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._kalman import (
    _kalman_update_scalar,
    _kalman_update_sequential,
    _project_cov_ahead,
)
from ._statespace import _dyawda
from ._transforms import _yaw_from_quat
from ._vectorops import _normalize_vec
from ._vectorops import _skew_symmetric as S


def _gravity_nav(g: float, nav_frame: str) -> NDArray[np.float64]:
    """
    Gravity vector expressed in the navigation frame ('NED' or 'ENU').

    Parameters
    ----------
    g : float
        Gravitational acceleration in m/s^2.
    nav_frame : {'NED', 'ENU'}
        Navigation frame in which the gravity vector is expressed.

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
    Multiplicative extended Kalman filter (MEKF) for attitude estimation.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    att : Attitude or array_like, shape (4,)
        Initial attitude estimate as an Attitude instance or a unit quaternion (qw, qx, qy, qz).
    bg : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    w : array_like, shape (3,), default (0.0, 0.0, 0.0)
        Initial angular rate estimate (wx, wy, wz) in rad/s expressed in the body frame.
        Defaults to zero angular rate (stationary).
    P : array_like, shape (6, 6), default 1e-6 * np.eye(6)
        Initial error covariance matrix estimate. Defaults to a small diagonal matrix
        (1e-6 * np.eye(6)). The order of the (error) states is: dx = (da, dbg),
        where da is the attitude error, and dbg is the gyroscope bias error.
    gyro_noise_density : float, default 0.0001
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.0001 (typical value for low-cost MEMS IMUs).
    gyro_bias_stability : float, default 0.00005
        Gyroscope bias stability (1-sigma) in rad/s. Defaults to 0.00005 (typical
        value for low-cost MEMS IMUs).
    gyro_bias_corr_time : float, default 50.0
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    nav_frame : {'NED', 'ENU'}, default 'NED'
        Specifies the assumed inertial-like navigation frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up).
    """

    _ATT_IDX: slice = slice(0, 3)  # attitude error state indices
    _BG_IDX: slice = slice(3, 6)  # gyro bias error state indices
    _I: NDArray[np.float64] = np.eye(6)

    def __init__(
        self,
        fs: float,
        att: Attitude | ArrayLike,
        bg: ArrayLike = (0.0, 0.0, 0.0),
        w: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = 1e-6 * np.eye(6),
        gyro_noise_density: float = 0.0001,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
        nav_frame: str = "NED",
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._nav_frame = nav_frame.lower()
        self._z2g = np.sign(_gravity_nav(1.0, self._nav_frame)[2])

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

        # Discrete state-space model (phi is updated each time step)
        self._phi = self._prep_state_transition_matrix()
        self._Q = self._prep_process_noise_cov_matrix()
        self._dhdx = self._prep_measurement_matrix()

    @property
    def _vg_b(self):
        """Gravity reference vector (unit vector) expressed in the body frame."""
        return self._z2g * self._att_nb.as_matrix()[2, :]

    @property
    def _yaw(self) -> float:
        """
        Heading (yaw angle) estimate in radians.
        """
        return _yaw_from_quat(self._att_nb._q)

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

    def _prep_state_transition_matrix(self) -> NDArray[np.float64]:
        """
        Setup state transition matrix, phi, using the first-order approximation:

            phi = I + dt * dfdx

        where dfdx denotes the linearized state matrix.

        Returns
        -------
        phi : ndarray, shape (6, 6)
            State transition matrix.
        """
        phi = np.eye(6)
        phi[self._ATT_IDX, self._ATT_IDX] -= self._dt * S(self._w_b)  # NB! update
        phi[self._ATT_IDX, self._BG_IDX] -= self._dt * np.eye(3)
        phi[self._BG_IDX, self._BG_IDX] -= self._dt * np.eye(3) / self._gbc
        return phi

    @staticmethod
    @njit  # type: ignore[misc]
    def _update_state_transition(
        phi: NDArray[np.float64],
        dt: float,
        w_b: NDArray[np.float64],
    ) -> None:
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
        phi[0, 1] = dt * wz
        phi[0, 2] = -dt * wy
        phi[1, 0] = -dt * wz
        phi[1, 2] = dt * wx
        phi[2, 0] = dt * wy
        phi[2, 1] = -dt * wx

    def _prep_process_noise_cov_matrix(self) -> NDArray[np.float64]:
        """
        Setup process noise covariance matrix, Q, using the first-order approximation:

            Q = dt @ dfdw @ W @ dfdw.T

        Returns
        -------
        Q : ndarray, shape (6, 6)
            Process noise covariance matrix.
        """
        Q = np.zeros((6, 6))
        Q[self._ATT_IDX, self._ATT_IDX] = self._dt * self._arw**2 * np.eye(3)
        Q[self._BG_IDX, self._BG_IDX] = (
            self._dt * (2.0 * self._gbs**2 / self._gbc) * np.eye(3)
        )
        return Q

    def _prep_measurement_matrix(self) -> NDArray[np.float64]:
        """
        Setup linearized measurement matrix, dhdx.

        Parameters
        ----------
        q_nb : ndarray, shape (4,)
            Unit quaternion.
        vg_b : ndarray, shape (3,)
            Gravity reference (unit) vector expressed in the body frame.

        Returns
        -------
        dhdx : ndarray, shape (4, 6)
            Linearized measurement matrix.
        """
        dhdx = np.zeros((4, 6))
        dhdx[0:3, self._ATT_IDX] = S(self._vg_b)  # NB! update
        dhdx[3:4, self._ATT_IDX] = _dyawda(self._att_nb._q)  # NB! update
        return dhdx

    def _dhdx_gref(self, vg_b: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Gravity reference vector part of the measurement matrix, shape (3, 6).
        """
        self._dhdx[0:3, self._ATT_IDX] = S(vg_b)
        return self._dhdx[0:3]

    def _dhdx_yaw(self, q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Heading (yaw angle) part of the measurement matrix, shape (6,).
        """
        self._dhdx[3:4, self._ATT_IDX] = _dyawda(q_nb)
        return self._dhdx[3]

    def _reset(self) -> None:
        """
        Reset state (regulating error-state to zero).
        """

        if not self._dx.any():
            return

        self._att_nb._correct_da(self._dx[self._ATT_IDX])
        self._bg_b[:] += self._dx[self._BG_IDX]
        self._dx[:] = 0.0

    def _aiding_update_gref(
        self, vg_meas: ArrayLike | None, vg_var: ArrayLike | None
    ) -> None:
        """
        Update with gravity reference vector aiding measurement.
        """

        if vg_meas is None:
            return None

        if vg_var is None:
            raise ValueError("'vg_var' not provided.")

        vg_b = self._vg_b
        dz = vg_meas - vg_b
        dhdx = self._dhdx_gref(vg_b)
        _kalman_update_sequential(self._dx, self._P, dz, vg_var, dhdx, self._I)

    def _aiding_update_yaw(
        self, yaw_meas: float | None, yaw_var: float | None, yaw_degrees: bool
    ) -> None:
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

        dz = _signed_smallest_angle(yaw_meas - self._yaw)
        dhdx = self._dhdx_yaw(self._att_nb._q)
        _kalman_update_scalar(self._dx, self._P, dz, yaw_var, dhdx, self._I)

    def _project_ahead(self) -> None:
        """
        Project state and covariance estimates ahead.
        """

        # Attitude (dead reckoning)
        self._att_nb._project_ahead(self._w_b, self._dt)

        # Covariance
        _project_cov_ahead(self._P, self._phi, self._Q)

    def update(
        self,
        f: ArrayLike,
        w: ArrayLike,
        degrees: bool = False,
        yaw: float | None = None,
        yaw_var: float | None = None,
        yaw_degrees: bool = False,
        gref: bool = True,
        gref_var: ArrayLike | None = (0.001, 0.001, 0.001),
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
        gref : bool, default True
            Specifies whether to use the gravity reference vector aiding measurement.
            If ``False``, gravity reference aiding is not used.
        gref_var : array_like, shape (3,), default (0.001, 0.001, 0.001)
            Variance of gravity reference vector measurement noise.

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
        self._aiding_update_gref(-_normalize_vec(f) if gref else None, gref_var)
        self._aiding_update_yaw(yaw, yaw_var, yaw_degrees)

        # Reset attitude
        self._reset()

        # Update model
        self._w_b[:] = w - self._bg_b
        self._update_state_transition(self._phi, self._dt, self._w_b)

        return self
