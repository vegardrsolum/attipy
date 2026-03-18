import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_gain_fast(
    P: NDArray[np.float64],
    h: NDArray[np.float64],
    r: float,
    out: NDArray[np.float64],
) -> None:
    """
    Compute the Kalman gain for a scalar measurement:

        k = P @ h.T / (h @ P @ h.T + r)

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    r : float
        Scalar measurement noise variance.
    out : ndarray, shape (n,)
        Output array for the Kalman gain vector, written in place.
    """
    n = len(h)  # number of states

    s = 0.0
    for i in range(n):
        v = 0.0
        for j in range(n):
            v += P[i, j] * h[j]
        out[i] = v
        s += h[i] * v

    s_inv = 1.0 / (s + r)
    for i in range(n):
        out[i] *= s_inv


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
    n = len(x)  # number of states
    y = 0.0
    for i in range(n):
        y += h[i] * x[i]
    y = z - y
    for i in range(n):
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
    Joseph-form covariance update (in place):

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
        Temporary workspace array.
    """
    n = len(h)  # number of states

    for j in range(n):
        s = 0.0
        for i in range(n):
            s += h[i] * P[i, j]
        tmp[j] = s

    for i in range(n):
        ki = k[i]

        dot = 0.0
        for j in range(n):
            P[i, j] -= ki * tmp[j]
            dot += P[i, j] * h[j]

        c = r * ki - dot
        for j in range(n):
            P[i, j] += c * k[j]


@njit  # type: ignore[misc]
def _kalman_update_scalar_fast(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: float,
    r: float,
    h: NDArray[np.float64],
    k: NDArray[np.float64],
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
    k : ndarray, shape (n,)
        Temporary workspace array for the Kalman gain vector.
    tmp : ndarray, shape (n,)
        Temporary workspace array for the covariance update.
    """

    # Kalman gain
    _kalman_gain_fast(P, h, r, k)

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
    k: NDArray[np.float64],
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
    k : ndarray, shape (n,)
        Temporary workspace array for the Kalman gain vector.
    tmp : ndarray, shape (n,)
        Temporary workspace array for the covariance update.
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar_fast(x, P, z[i], var[i], H[i], k, tmp)


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

    # tmp = phi @ P
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += phi[i, k] * P[k, j]
            tmp[i, j] = s

    # P = tmp @ phi.T + Q
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += tmp[i, k] * phi[j, k]
            P[i, j] = s + Q[i, j]
