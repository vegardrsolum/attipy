from typing import Self

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from attipy._quatops import _correct_quat_with_gibbs2, _correct_quat_with_rotvec

from ._attitude import Attitude
from ._kalman_fast import (
    _kalman_update_scalar_fast,
    _kalman_update_sequential_fast,
    _project_cov_ahead_fast,
)
from ._statespace import (
    _dyawda,
    _measurement_matrix,
    _process_noise_cov,
    _state_transition,
    _update_state_transition,
)
from ._transforms import _nz_b_from_quat, _yaw_from_quat
from ._vectorops import _normalize_vec, _skew_symmetric


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
    ndarray, shape (3,)
        Gravity vector expressed in the navigation frame.
    """
    if nav_frame.lower() == "ned":
        g_n = np.array([0.0, 0.0, g])
    elif nav_frame.lower() == "enu":
        g_n = np.array([0.0, 0.0, -g])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return g_n


def _nz2vg(nav_frame: str) -> float:
    """
    Gravity direction along the navigation frame's z-axis. Transforms the z-axis
    of the navigation frame to a gravity reference vector (unit vector).

    Parameters
    ----------
    nav_frame : {'NED', 'ENU'}
        Navigation frame.

    Returns
    -------
    float
        Gravity direction along the navigation frame's z-axis. +1.0 for 'NED' and
        -1.0 for 'ENU'.
    """
    if nav_frame.lower() == "ned":
        return 1.0
    elif nav_frame.lower() == "enu":
        return -1.0
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")


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


@njit  # type: ignore[misc]
def _reset(q, bg, dx) -> None:
    """
    Reset states (regulating error-states to zero).

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion to be updated in place.
    bg : ndarray, shape (3,)
        Gyroscope bias to be updated in place.
    dx : ndarray, shape (6,)
        Error-state vector to be reset in place.
    """
    _correct_quat_with_gibbs2(q, dx[0:3])
    bg[:] += dx[3:6]
    dx[:] = 0.0


class MEKF:
    """
    Multiplicative extended Kalman filter (MEKF) for attitude estimation.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    q : Attitude or array_like, shape (4,), optional
        Initial attitude estimate as an Attitude instance or a unit quaternion (qw, qx, qy, qz).
        Defaults to the identity quaternion (1.0, 0.0, 0.0, 0.0) (i.e., no rotation).
    bg : array_like, shape (3,), optional
        Initial gyroscope bias estimate (bgx, bgy, bgz) in rad/s. Defaults to zero bias.
    P : array_like, shape (6, 6), optional
        Initial error covariance matrix estimate. Defaults to a small diagonal matrix
        (1e-6 * np.eye(6)). The order of the (error) states is: dx = (da, dbg),
        where da is the attitude error, and dbg is the gyroscope bias error.
    dtheta : array_like, shape (3,), optional
        Previous attitude increment (coning integral) in radians. Defaults to zero.
    gyro_noise_density : float, optional
        Gyroscope noise density (angular random walk) in (rad/s)/√Hz. Defaults to
        0.0001 (typical value for low-cost MEMS IMUs).
    gyro_bias_stability : float, optional
        Gyroscope bias stability (1-sigma) in rad/s. Defaults to 0.00005 (typical
        value for low-cost MEMS IMUs).
    gyro_bias_corr_time : float, optional
        Gyroscope bias correlation time in seconds. Defaults to 50.0 s.
    nav_frame : {'NED', 'ENU'}, optional
        Specifies the assumed inertial-like navigation frame. Should be 'NED'
        (North-East-Down) (default) or 'ENU' (East-North-Up).
    """

    def __init__(
        self,
        fs: float,
        q: Attitude | ArrayLike = (1.0, 0.0, 0.0, 0.0),
        bg: ArrayLike = (0.0, 0.0, 0.0),
        P: ArrayLike = 1e-6 * np.eye(6),
        dtheta: ArrayLike = (0.0, 0.0, 0.0),
        gyro_noise_density: float = 0.0001,
        gyro_bias_stability: float = 0.00005,
        gyro_bias_corr_time: float = 50.0,
        nav_frame: str = "NED",
    ) -> None:
        self._fs = fs
        self._dt = 1.0 / fs
        self._nav_frame = nav_frame.lower()
        self._nz2vg = _nz2vg(self._nav_frame)
        self._tmp = np.empty((6, 6))  # preallocated workspace

        # IMU noise parameters
        self._arw = gyro_noise_density  # angular random walk
        self._gbs = gyro_bias_stability  # gyro bias stability
        self._gbc = gyro_bias_corr_time  # gyro bias correlation time

        # State and covariance estimates
        self._att_nb = q if isinstance(q, Attitude) else Attitude(q)
        self._bg_b = np.asarray_chkfinite(bg).reshape(3).copy()
        self._dtheta = np.asarray_chkfinite(dtheta).reshape(3).copy()
        self._P = np.asarray_chkfinite(P).reshape(6, 6).copy()
        self._dx = np.zeros(6)

        # Discrete state-space model
        self._phi = _state_transition(self._dt, self._dtheta, self._gbc)
        self._Q = _process_noise_cov(self._dt, self._arw, self._gbs, self._gbc)
        self._dhdx = _measurement_matrix(self._att_nb._q, self._vg_b)

    @property
    def _vg_b(self):
        """Gravity reference vector (unit vector) expressed in the body frame."""
        return self._nz2vg * _nz_b_from_quat(self._att_nb._q)

    @property
    def _yaw(self) -> float:
        """
        Heading (yaw angle) estimate in radians.
        """
        return _yaw_from_quat(self._att_nb._q)

    @property
    def P(self) -> NDArray[np.float64]:
        """
        Copy of the error covariance matrix estimate.
        """
        return self._P.copy()

    @property
    def dtheta(self) -> NDArray[np.float64]:
        """Copy of the previous attitude increment (coning integral) in radians."""
        return self._dtheta.copy()

    @property
    def attitude(self) -> Attitude:
        """Attitude estimate (no copy)."""
        return self._att_nb

    def bias_gyro(self, degrees=False) -> NDArray[np.float64]:
        """
        Gyroscope bias estimate expressed in the body frame.

        Parameters
        ----------
        degrees : bool, optional
            Specifies whether to return the bias estimate in deg/s or rad/s. Defaults
            to rad/s.

        Returns
        -------
        ndarray, shape (3,)
            Copy of the gyroscope bias estimate.
        """
        if degrees:
            return np.degrees(self._bg_b.copy())
        return self._bg_b.copy()

    def _dhdx_gref(self, vg_b: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Gravity reference vector part of the measurement matrix, shape (3, 6).
        """
        self._dhdx[0:3, 0:3] = _skew_symmetric(vg_b)
        return self._dhdx[0:3]

    def _dhdx_yaw(self, q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Heading (yaw angle) part of the measurement matrix, shape (6,).
        """
        self._dhdx[3:4, 0:3] = _dyawda(q_nb)
        return self._dhdx[3]

    def _aiding_update_gref(
        self, vg_meas: ArrayLike | None, vg_var: ArrayLike | None
    ) -> None:
        """
        Update state and covariance with gravity reference vector aiding measurement.
        """

        if vg_meas is None:
            return None

        if vg_var is None:
            raise ValueError("'vg_var' not provided.")

        vg_b = self._vg_b
        dz = vg_meas - vg_b
        dhdx = self._dhdx_gref(vg_b)
        _kalman_update_sequential_fast(
            self._dx, self._P, dz, vg_var, dhdx, self._tmp[0], self._tmp[1]
        )

    def _aiding_update_yaw(
        self, yaw_meas: float | None, yaw_var: float | None, yaw_degrees: bool
    ) -> None:
        """
        Update state and covariance with heading (yaw angle) aiding measurement.
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
        _kalman_update_scalar_fast(
            self._dx, self._P, dz, yaw_var, dhdx, self._tmp[0], self._tmp[1]
        )

    def _project_ahead(self, dtheta: NDArray[np.float64]) -> None:
        """
        Project state and covariance estimates ahead.
        """

        # Attitude update (strapdown algorithm)
        _correct_quat_with_rotvec(self._att_nb._q, dtheta)

        # Covariance projection
        _project_cov_ahead_fast(self._P, self._phi, self._Q, self._tmp)

    def update(
        self,
        dv: ArrayLike,
        dtheta: ArrayLike,
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
        dv : array_like, shape (3,)
            Sculling integral in m/s. I.e., the integral of specific force, f, over
            the sampling interval, dt. The simple approximation dv = f * dt can be
            used if sculling-corrected integrals are not available.
        dtheta : array_like, shape (3,)
            Coning integral (see ``degrees`` parameter for units). I.e., the integral
            of angular velocity, w, over the sampling interval, dt. The simple approximation
            dtheta = w * dt can be used if coning-corrected integrals are not available.
        degrees : bool, optional
            Specifies whether ``dtheta`` is given in degrees or radians. Defaults to radians.
        yaw : float, optional
            Heading (yaw angle) aiding measurement (see ``yaw_degrees`` for units).
            Defaults to ``None`` (no yaw aiding).
        yaw_var : float, optional
            Variance of heading (yaw angle) measurement (see ``yaw_degrees`` for units).
            Required for ``yaw``.
        yaw_degrees : bool, optional
            Specifies whether the units of ``yaw`` and ``yaw_var`` are deg and deg^2
            or rad and rad^2 (default).
        gref : bool, optional
            Specifies whether to use accelerometer measurements (dv) and the known
            direction of gravity as aiding. Defaults to ``True``.
        gref_var : array_like, shape (3,), optional
            Variance of gravity reference vector measurement noise (dimensionless).
            Required for ``gref``. Defaults to (0.001, 0.001, 0.001).

        Returns
        -------
        MEKF
            A reference to the instance itself after the update.
        """
        dv = np.asarray(dv)
        dtheta = np.asarray(dtheta)

        if degrees:
            dtheta = np.radians(dtheta)

        dtheta = dtheta - self._dt * self._bg_b

        # Project (a priori) state and covariance estimates ahead
        self._project_ahead(dtheta)

        # Update (a posteriori) state and covariance estimates with aiding measurements
        self._aiding_update_gref(-_normalize_vec(dv) if gref else None, gref_var)
        self._aiding_update_yaw(yaw, yaw_var, yaw_degrees)

        # Reset state (regulating error-state to zero)
        _reset(self._att_nb._q, self._bg_b, self._dx)

        # Update state space model
        self._dtheta[:] = dtheta
        _update_state_transition(self._phi, dtheta)

        return self
