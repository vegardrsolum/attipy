import numpy as np
from numba import njit
from numpy.typing import NDArray


def _kalman_update(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    R: NDArray[np.float64],
    H: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Kalman filter measurement update.

    Used as reference implementation for testing the fast versions.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated.
    z : ndarray, shape (m,)
        Measurement vector.
    R : ndarray, shape (m, m)
        Measurement noise covariance matrix.
    H : ndarray, shape (m, n)
        Measurement matrix where each row corresponds to a scalar measurement model.

    Returns
    -------
    x : ndarray, shape (n,)
        Updated state estimate.
    P : ndarray, shape (n, n)
        Updated state error covariance matrix.
    """
    x = np.asarray(x)
    P = np.asarray(P)
    z = np.asarray(z)
    H = np.asarray(H)
    R = np.asarray(R)
    I_ = np.eye(x.size)

    # Innovation (pre-fit residual) covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # Updated (a posteriori) state estimate
    x = x + K @ (z - H @ x)

    # Updated (a posteriori) covariance estimate (Joseph form)
    P = (I_ - K @ H) @ P @ (I_ - K @ H).T + K @ R @ K.T

    return x, P


@njit  # type: ignore[misc]
def _kalman_gain(
    P: NDArray[np.float64], h: NDArray[np.float64], r: float
) -> NDArray[np.float64]:
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

    Returns
    -------
    k : ndarray, shape (n,)
        Kalman gain vector.
    """

    # Innovation covariance (inverse)
    Ph = np.dot(P, h)
    s_inv = 1.0 / (np.dot(h, Ph) + r)

    # Kalman gain
    k = Ph * s_inv

    return k


@njit  # type: ignore[misc]
def _covariance_update(
    P: NDArray[np.float64],
    k: NDArray[np.float64],
    h: NDArray[np.float64],
    r: float,
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
    """
    A = np.eye(k.size) - np.outer(k, h)
    P[:, :] = A @ P @ A.T + r * np.outer(k, k)


@njit  # type: ignore[misc]
def _kalman_update_scalar(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: float,
    r: float,
    h: NDArray[np.float64],
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
    """

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    x[:] += k * (z - np.dot(h, x))

    # Updated (a posteriori) covariance estimate (Joseph form)
    _covariance_update(P, k, h, r)


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
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
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar(x, P, z[i], var[i], H[i])


@njit  # type: ignore[misc]
def _project_cov_ahead(
    P: NDArray[np.float64], phi: NDArray[np.float64], Q: NDArray[np.float64]
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
    """
    P[:, :] = phi @ P @ phi.T + Q
