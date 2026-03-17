import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._kalman import _kalman_gain


@njit  # type: ignore[misc]
def _state_update_fast(
    x: NDArray[np.float64], z: float, k: NDArray[np.float64], h: NDArray[np.float64]
) -> None:
    """
    Update state with scalar measurement.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State vector to be updated in place.
    z : float
        Scalar measurement.
    k : ndarray, shape (n,)
        Kalman gain vector.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    """
    y = z - np.sum(h * x)
    for i in range(len(x)):
        x[i] += k[i] * y


@njit  # type: ignore[misc]
def _covariance_update_fast(
    P: NDArray[np.float64],
    k: NDArray[np.float64],
    h: NDArray[np.float64],
    r: float,
    tmp: NDArray[np.float64],
) -> None:
    """
    Compute the updated state error covariance matrix estimate (Joseph form):

        P = (I - k @ h) @ P @ (I - k @ h).T + r * k @ k.T

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    k : ndarray, shape (n,)
        Kalman gain vector.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    r : float
        Scalar measurement noise variance.
    tmp : ndarray, shape (n,)
        Temporary workspace array for intermediate calculations, to avoid repeated
        allocations.
    """
    n = P.shape[0]

    hP = tmp
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += h[j] * P[j, i]
        hP[i] = s

    for i in range(n):
        ki = k[i]
        for j in range(n):
            P[i, j] -= ki * hP[j]

    Th = tmp
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += P[i, j] * h[j]
        Th[i] = s

    for i in range(n):
        c = r * k[i] - Th[i]
        for j in range(n):
            P[i, j] += c * k[j]


@njit  # type: ignore[misc]
def _kalman_update_scalar_fast(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: float,
    r: float,
    h: NDArray[np.float64],
    tmp: NDArray[np.float64],
) -> None:
    """
    Scalar Kalman filter measurement update.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    z : float
        Scalar measurement.
    r : float
        Scalar measurement noise variance.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    tmp : ndarray, shape (n,)
        Temporary workspace array for intermediate calculations, to avoid repeated
        allocations.
    """

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    _state_update_fast(x, z, k, h)

    # Updated (a posteriori) covariance estimate (Joseph form)
    _covariance_update_fast(P, k, h, r, tmp)


@njit  # type: ignore[misc]
def _kalman_update_sequential_fast(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    tmp: NDArray[np.float64],
) -> None:
    """
    Sequential (one-at-a-time) Kalman filter measurement update.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    z : ndarray, shape (m,)
        Measurement vector.
    var : ndarray, shape (m,)
        Measurement noise variances corresponding to each scalar measurement.
    H : ndarray, shape (m, n)
        Measurement matrix where each row corresponds to a scalar measurement model.
    tmp : ndarray, shape (n,)
        Temporary workspace array for intermediate calculations, to avoid repeated
        allocations.
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar_fast(x, P, z[i], var[i], H[i], tmp)
