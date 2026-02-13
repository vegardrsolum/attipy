from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._attitude import Attitude
from ._kalman import _covariance_update, _kalman_gain
from ._mekf import _signed_smallest_angle
from ._statespace import _dyawda
from ._transforms import _yaw_from_quat
from ._vectorops import _normalize_vec
from ._vectorops import _skew_symmetric as S


def _state_transition_matrix(dt, w_b, gbc):
    phi = np.eye(6)
    phi[0:3, 0:3] -= dt * S(w_b)  # NB! update each time step
    phi[0:3, 3:6] -= dt * np.eye(3)
    phi[3:6, 3:6] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _update_state_transition_matrix(
    phi: NDArray[np.float64],
    dt: float,
    w_b: NDArray[np.float64],
):
    """
    Update the state transition matrix in place.

    Parameters
    ----------
    phi : ndarray, shape (6, 6)
        State transition matrix to be updated in place.
    dt : float
        Time step.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    """
    wx, wy, wz = w_b
    phi[0, 1] = dt * wz
    phi[0, 2] = -dt * wy
    phi[1, 0] = -dt * wz
    phi[1, 2] = dt * wx
    phi[2, 0] = dt * wy
    phi[2, 1] = -dt * wx


def _process_noise_cov_matrix(dt, arw, gbs, gbc):
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = dt * arw**2 * np.eye(3)
    Q[3:6, 3:6] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _measurement_matrix(q_nb, vg_b):
    dhdx = np.zeros((4, 6))
    dhdx[0:1, 0:3] = _dyawda(q_nb)  # NB! update each time step
    dhdx[1:4, 0:3] = S(vg_b)  # NB! update each time step
    return dhdx


@njit  # type: ignore[misc]
def _update_measurement_matrix_yaw(dhdx, q_nb):
    """
    Heading (yaw angle) part of the measurement matrix, shape (6,).
    """
    dhdx[0:1, 0:3] = _dyawda(q_nb)
    return dhdx[0]


@njit  # type: ignore[misc]
def _update_measurement_matrix_gref(dhdx, vg_b):
    """
    Gravity reference vector part of the measurement matrix, shape (3, 6).
    """
    dhdx[1:4, 0:3] = S(vg_b)
    return dhdx[1:4]


def _z_down(nav_frame) -> NDArray[np.float64]:
    if nav_frame.lower() == "ned":
        return 1
    elif nav_frame.lower() == "enu":
        return -1
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")


@njit  # type: ignore[misc]
def _kalman_update_scalar(da, bg_b, P, z, r, h, I_):

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    y = z - np.dot(h[0:3], da)
    da[:] += k[0:3] * y
    bg_b[:] += k[3:6] * y

    # Updated (a posteriori) covariance estimate (Joseph form)
    _covariance_update(P, k, h, r, I_)


@njit  # type: ignore[misc]
def _kalman_update_sequential(da, bg_b, P, z, var, H, I_):
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar(da, bg_b, P, z[i], var[i], H[i], I_)


@njit  # type: ignore[misc]
def _project_cov_ahead(P, phi, Q):
    P[:, :] = phi @ P @ phi.T + Q


class MEKF_:
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
        self._z_down = _z_down(self._nav_frame)

        # IMU noise parameters
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._att_nb = att if isinstance(att, Attitude) else Attitude(att)
        self._vg_b = self._z_down * self._att_nb.as_matrix()[2, :]  # preallocation
        self._bg_b = np.asarray_chkfinite(bg).reshape(3).copy()
        self._w_b = np.asarray_chkfinite(w).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(6, 6).copy()
        self._da = np.zeros(3)  # attitude error state (2xGibbs vector)

        # Discretized state space model (updated each time step)
        self._phi = _state_transition_matrix(self._dt, self._w_b, self._gbc)
        self._Q = _process_noise_cov_matrix(self._dt, self._arw, self._gbs, self._gbc)
        self._dhdx = _measurement_matrix(self._att_nb._q, self._vg_b)

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

    def _reset(self) -> None:
        """
        Reset state (regulating error-state to zero).
        """

        if not self._da.any():
            return

        self._att_nb._correct_da(self._da)
        self._da[:] = 0.0

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

        yaw = _yaw_from_quat(self._att_nb._q)
        z = _signed_smallest_angle(yaw_meas - yaw)
        dhdx = _update_measurement_matrix_yaw(self._dhdx, self._att_nb._q)
        _kalman_update_scalar(self._da, self._bg_b, self._P, z, yaw_var, dhdx, self._I6)

    def _aiding_update_gref(self, f_b, gref_var):
        """
        Update with gravity reference vector aiding measurement.
        """

        if f_b is None:
            return None

        if gref_var is None:
            raise ValueError("'gref_var' not provided.")

        self._vg_b[:] = self._z_down * self._att_nb.as_matrix()[2, :]
        z = -_normalize_vec(f_b) - self._vg_b
        dhdx = _update_measurement_matrix_gref(self._dhdx, self._vg_b)
        _kalman_update_sequential(
            self._da, self._bg_b, self._P, z, gref_var, dhdx, self._I6
        )

    def _project_ahead(self):
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
        gref_var : array_like, shape (3,), default (0.01, 0.01, 0.01)
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
        self._aiding_update_gref(f if gref else None, gref_var)
        self._aiding_update_yaw(yaw, yaw_var, yaw_degrees)

        # Reset attitude
        self._reset()

        # Update model
        self._w_b[:] = w - self._bg_b
        _update_state_transition_matrix(self._phi, self._dt, self._w_b)

        return self
