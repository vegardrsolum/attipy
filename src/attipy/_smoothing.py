from typing import Self
from warnings import warn

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._ahrs import AHRS
from ._quatops import _quatprod
from ._statespace import _update_state_transition as _update_phi
from ._transforms import _matrix_from_quat
from ._vectorops import _normalize


class FixedIntervalSmoother:
    """
    Fixed-interval smoothing for AidedINS.

    This class wraps an instance of AidedINS (or a subclass like AHRS or VRU),
    and maintains a time-ordered buffer of state and error covariance estimates
    as measurements are processed via the ``update()`` method. A backward sweep
    over the buffered data using the Rauch-Tung-Striebel (RTS) algorithm [1] is
    performed to refine the filter estimates.

    Parameters
    ----------
    ains : AidedINS or AHRS or VRU
        The underlying AidedINS instance used for forward filtering.
    cov_smoothing : bool, default True
        Whether to include the error covariance matrix, `P`, in the smoothing process.
        Disabling the covariance smoothing has no effect on the smoothed state estimates,
        and can reduce computation time if smoothed covariances are not required.

    References
    ----------
    [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
        filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
    """

    def __init__(self, ahrs: AHRS, cov_smoothing: bool = True) -> None:
        warn(
            "FixedIntervalSmoother is experimental and may change or be removed in the future.",
            UserWarning,
        )
        self._ahrs = ahrs
        self._cov_smoothing = cov_smoothing

        # Buffers for storing state and covariance estimates from forward sweep
        self._P_buf = []  # error covariance estimates (w/o smoothing)
        self._dx_buf = []  # error-state estimates (w/o smoothing)
        self._q_buf = []  # quaternion estimates (w/o smoothing)
        self._bg_buf = []  # gyro bias estimates (w/o smoothing)
        self._v_buf = []  # velocity estimates (w/o smoothing)
        self._w_buf = []  # angular rate measurements (bias corrected)
        self._f_buf = []  # specific force measurements

        # Smoothed state and covariance estimates
        self._q_nb = np.empty((0, 4), dtype="float64")
        self._bg_b = np.empty((0, 3), dtype="float64")
        self._v_n = np.empty((0, 3), dtype="float64")
        self._P = np.empty((0, *self._ahrs.P.shape), dtype="float64")

    @property
    def ahrs(self) -> AHRS:
        """
        The underlying AidedINS instance used for forward filtering.

        Returns
        -------
        AidedINS or AHRS or VRU
            The AidedINS instance.
        """
        return self._ahrs

    def update(self, *args, **kwargs) -> Self:
        """
        Update the AINS with measurements, and append the current AINS state to
        the smoother's internal buffer.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed on to ``ains.update()``.
        **kwargs : dict
            Keyword arguments to be passed on to ``ains.update()``.
        """
        self._ahrs.update(*args, **kwargs)
        # self._x_buf.append(self._ains.x)
        self._P_buf.append(self._ahrs.P)
        self._dx_buf.append(self._ahrs._dx.copy())

        # State
        self._q_buf.append(self._ahrs.q_nb)
        self._bg_buf.append(self._ahrs.bg_b)
        self._v_buf.append(self._ahrs.v_n)
        self._w_buf.append(self._ahrs.w_b)
        self._f_buf.append(self._ahrs.f_b)

        return self

    def clear(self) -> None:
        """
        Clear the internal buffer of state estimates. This resets the smoother,
        and prepares for a new interval of measurements.
        """
        self._dx_buf.clear()
        self._P_buf.clear()
        self._q_buf.clear()
        self._bg_buf.clear()
        self._v_buf.clear()
        self._w_buf.clear()
        self._f_buf.clear()

    def _smooth(self):
        n_samples = len(self._q_buf)
        if n_samples == 0:
            self._q_nb = np.empty((0, 4), dtype="float64")
            self._bg_b = np.empty((0, 3), dtype="float64")
            self._v_n = np.empty((0, 3), dtype="float64")
            self._P = np.empty((0, *self._ains.P.shape), dtype="float64")
        elif n_samples == 1:
            self._q_nb = np.asarray(self._q_buf)
            self._bg_b = np.asarray(self._bg_buf)
            self._v_n = np.asarray(self._v_buf)
            self._P = np.asarray(self._P_buf)
        elif n_samples != len(self._q_nb):
            q_nb, bg_b, v_n, dx, P = _rts_backward_sweep(
                self._q_buf,
                self._bg_buf,
                self._v_buf,
                self._w_buf,
                self._f_buf,
                self._dx_buf,
                self._P_buf,
                self._cov_smoothing,
                self._ahrs._phi,
                self._ahrs._Q,
                self._ahrs._dt,
            )
            self._q_nb = np.asarray(q_nb)
            self._bg_b = np.asarray(bg_b)
            self._v_n = np.asarray(v_n)
            self._P = np.asarray(P)

    @property
    def q_nb(self) -> NDArray:
        self._smooth()
        return self._q_nb.copy()

    @property
    def bg_b(self) -> NDArray:
        self._smooth()
        return self._bg_b.copy()

    @property
    def v_n(self) -> NDArray:
        self._smooth()
        return self._v_n.copy()

    @property
    def x(self) -> NDArray:
        """
        Smoothed state vector estimates.

        Returns
        -------
        np.ndarray, shape (N, 15) or (N, 12)
            State estimates for each of the N time steps where the smoother has
            been updated with measurements.
        """
        self._smooth()
        return self._x.copy()

    @property
    def P(self) -> NDArray:
        """
        Error covariance matrix estimates.

        If ``cov_smoothing=True``, smoothed error covariance estimates are returned.
        Otherwise, the forward filter covariance estimates are returned.

        Returns
        -------
        np.ndarray, shape (N, 15, 15) or (N, 12, 12)
            Error covariance matrix estimates for each of the N time steps where
            the smoother has been updated with measurements.
        """
        self._smooth()
        return self._P.copy()


@njit  # type: ignore[misc]
def _rts_backward_sweep(
    q_nb: list[NDArray],
    bg_b: list[NDArray],
    v_n: list[NDArray],
    w_b: list[NDArray],
    f_b: list[NDArray],
    dx: list[NDArray],
    P: list[NDArray],
    cov_smoothing: bool,
    phi_k: NDArray,
    Q_k: NDArray,
    dt: float,
) -> tuple[list[NDArray], list[NDArray]]:
    """
    Perform a backward sweep with the RTS algorithm [1].

    Parameters
    ----------
    x : NDArray, shape (n_samples, 16)
        The state vector.
    dx : NDArray, shape (n_samples, 15) or (n_samples, 12)
        The error state vector.
    P : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
        The covariance matrix.
    P_prior : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
        The a priori covariance matrix.
    phi : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
        The state transition matrix.
    cov_smoothing : bool
        Whether to include the error covariance matrix in the smoothing process.

    Returns
    -------
    x_smth : NDArray, shape (n_samples, 15) or (n_samples, 12)
        The smoothed state vector.
    P_smth : NDArray, shape (n_samples, 15, 15) or (n_samples, 12, 12)
        The smoothed covariance matrix if include_cov is True, otherwise None.

    References
    ----------
    [1] R. G. Brown and P. Y. C. Hwang, "Random signals and applied Kalman
        filtering with MATLAB exercises", 4th ed. Wiley, pp. 208-212, 2012.
    """
    I3x3 = np.eye(3)

    q_nb = q_nb.copy()
    bg_b = bg_b.copy()
    v_n = v_n.copy()
    dx = dx.copy()
    P = P.copy()

    q_prealloc = np.array([2.0, 0.0, 0.0, 0.0])  # Preallocation

    # Backward sweep
    n = len(q_nb)
    for k in range(n - 2, -1, -1):

        # Update step k state space
        R_nb_k = _matrix_from_quat(q_nb[k])
        phi_k[:] = _update_phi(phi_k, dt, I3x3, f_b[k], w_b[k], R_nb_k)
        P_prior_kp1 = phi_k @ P[k] @ phi_k.T + Q_k

        # Smoothed error-state estimate and corresponding covariance
        A = P[k] @ phi_k.T @ np.linalg.inv(P_prior_kp1)
        ddx = A @ dx[k + 1]
        dx[k] += ddx
        if cov_smoothing:
            P[k] += A @ (P[k + 1] - P_prior_kp1) @ A.T

        # Reset
        dda = ddx[0:3]
        q_prealloc[1:] = dda
        ddq = (1.0 / np.sqrt(4.0 + dda.T @ dda)) * q_prealloc
        q_nb[k][:] = _normalize(_quatprod(q_nb[k], ddq))
        bg_b[k][:] = bg_b[k] + ddx[3:6]
        v_n[k][:] = v_n[k] + ddx[6:9]

    return q_nb, bg_b, v_n, dx, P
