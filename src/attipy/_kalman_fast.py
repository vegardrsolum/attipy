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
    tmp_k: NDArray[np.float64],
    tmp_cov: NDArray[np.float64],
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
    tmp_k : ndarray, shape (n,)
        Temporary workspace; holds the Kalman gain on output.
    tmp_cov : ndarray, shape (n,)
        Temporary workspace; holds Ph = P @ h on output.
    """
    n = h.shape[0]

    # Compute Ph = P @ h into tmp_cov and s = h @ Ph + r
    s = r
    for i in range(n):
        Ph_i = 0.0
        for j in range(n):
            Ph_i += P[i, j] * h[j]
        tmp_cov[i] = Ph_i
        s += h[i] * Ph_i

    # Kalman gain k = Ph / s into tmp_k
    s_inv = 1.0 / s
    for i in range(n):
        tmp_k[i] = tmp_cov[i] * s_inv

    # State update: x += k * (z - h @ x)
    y = z
    for i in range(n):
        y -= h[i] * x[i]
    for i in range(n):
        x[i] += tmp_k[i] * y

    # Joseph-form covariance update:
    #   P = P - outer(k, Ph) - outer(Ph, k) + s * outer(k, k)
    # Implemented in-place using tmp_cov = Ph and tmp_k = k.
    for i in range(n):
        ki = tmp_k[i]
        c = r * ki
        for j in range(n):
            P[i, j] -= ki * tmp_cov[j]
            c -= P[i, j] * h[j]
        for j in range(n):
            P[i, j] += c * tmp_k[j]


@njit  # type: ignore[misc]
def _kalman_update_sequential_fast(
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    tmp_k: NDArray[np.float64],
    tmp_cov: NDArray[np.float64],
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
    tmp_k : ndarray, shape (n,)
        Temporary workspace array for the Kalman gain vector.
    tmp_cov : ndarray, shape (n,)
        Temporary workspace array for the covariance update.
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar_fast(z[i], var[i], H[i], x, P, tmp_k, tmp_cov)


@njit  # type: ignore[misc]
def _project_cov_ahead_fast(
    P: NDArray[np.float64],
    phi: NDArray[np.float64],
    Q: NDArray[np.float64],
    tmp: NDArray[np.float64],
) -> None:
    """
    Project the error covariance ahead: P = phi @ P @ phi.T + Q

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

    # tmp = phi @ P
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += phi[i, k] * P[k, j]
            tmp[i, j] = s

    # P = tmp @ phi.T + Q (exploit symmetry)
    for i in range(n):
        for j in range(i, n):
            p = Q[i, j]
            for k in range(n):
                p += tmp[i, k] * phi[j, k]
            P[i, j] = p
            P[j, i] = p
