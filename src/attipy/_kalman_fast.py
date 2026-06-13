import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_update_scalar_fast(
    z: float,
    r: float,
    h: NDArray[np.float64],
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    tmp: NDArray[np.float64],
) -> None:
    """
    Scalar Kalman filter measurement update (loop-based, zero heap allocation).

    Parameters
    ----------
    z : float
        Scalar measurement.
    r : float
        Scalar measurement noise variance.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    tmp : ndarray, shape (n,)
        Temporary workspace array.
    """
    n = h.shape[0]

    # Innovation (pre-fit residual) covariance
    s = r
    for i in range(n):
        Ph_i = 0.0
        for j in range(n):
            Ph_i += P[i, j] * h[j]
        tmp[i] = Ph_i
        s += h[i] * Ph_i

    s_inv = 1.0 / s

    # Updated (a posteriori) state estimate
    y = z
    for i in range(n):
        y -= h[i] * x[i]
    ky = s_inv * y
    for i in range(n):
        x[i] += tmp[i] * ky

    # Updated (a posteriori) covariance estimate (rank-1 downdate, upper triangle + mirror)
    for i in range(n):
        for j in range(i, n):
            p = P[i, j] - tmp[i] * tmp[j] * s_inv
            P[i, j] = p
            P[j, i] = p


@njit  # type: ignore[misc]
def _kalman_update_sequential_fast(
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    tmp: NDArray[np.float64],
) -> None:
    """
    Sequential (one-at-a-time) Kalman filter measurement update.

    Parameters
    ----------
    z : ndarray, shape (m,)
        Measurement vector.
    var : ndarray, shape (m,)
        Measurement noise variances corresponding to each scalar measurement.
    H : ndarray, shape (m, n)
        Measurement matrix where each row corresponds to a scalar measurement model.
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    tmp : ndarray, shape (n,)
        Temporary workspace array.
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar_fast(z[i], var[i], H[i], x, P, tmp)


@njit  # type: ignore[misc]
def _project_cov_ahead_fast(
    P: NDArray[np.float64],
    phi: NDArray[np.float64],
    Q: NDArray[np.float64],
    tmp: NDArray[np.float64],
) -> None:
    """
    Project the error covariance ahead:

        P = phi @ P @ phi.T + Q

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    phi : ndarray, shape (n, n)
        State transition matrix.
    Q : ndarray, shape (n, n)
        Process noise covariance matrix.
    tmp : ndarray, shape (n, n)
        Temporary workspace matrix.
    """
    n = P.shape[0]

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += phi[i, k] * P[k, j]
            tmp[i, j] = s

    # upper triangle + mirror
    for i in range(n):
        for j in range(i, n):
            p = Q[i, j]
            for k in range(n):
                p += tmp[i, k] * phi[j, k]
            P[i, j] = p
            P[j, i] = p
