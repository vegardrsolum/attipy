import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


def _kalman_update(
    z: ArrayLike,
    R: ArrayLike,
    H: ArrayLike,
    x: ArrayLike,
    P: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Kalman filter measurement update.

    Used as reference implementation for testing the fast versions.

    Parameters
    ----------
    z : array_like, shape (m,)
        Measurement vector.
    R : array_like, shape (m, m)
        Measurement noise covariance matrix.
    H : array_like, shape (m, n)
        Measurement matrix where each row corresponds to a scalar measurement model.
    x : array_like, shape (n,)
        State estimate to be updated.
    P : array_like, shape (n, n)
        State error covariance matrix to be updated.

    Returns
    -------
    x : ndarray, shape (n,)
        Updated state estimate.
    P : ndarray, shape (n, n)
        Updated state error covariance matrix.
    """
    z = np.asarray(z)
    H = np.asarray(H)
    R = np.asarray(R)
    x = np.asarray(x)
    P = np.asarray(P)
    I_ = np.eye(x.size)

    # Innovation (pre-fit residual) covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # Updated (a posteriori) state estimate
    x = x + K @ (z - H @ x)

    # Updated (a posteriori) covariance estimate (Joseph form)
    P = (I_ - K @ H) @ P @ (I_ - K @ H).T + K @ R @ K.T

    return x, P  # type: ignore[return-value]


@njit  # type: ignore[misc]
def _kalman_update_scalar(
    z: float,
    r: float,
    h: NDArray[np.float64],
    x: NDArray[np.float64],
    P: NDArray[np.float64],
) -> None:
    """
    Scalar Kalman filter measurement update.

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
    """
    Ph = np.dot(P, h)

    # Innovation (pre-fit residual) covariance
    s = np.dot(h, Ph) + r

    # Kalman gain
    k = Ph / s

    # Updated (a posteriori) state estimate
    x[:] += k * (z - np.dot(h, x))

    # Updated (a posteriori) covariance estimate (Joseph form expanded)
    P[:, :] = P - np.outer(k, Ph) - np.outer(Ph, k) + s * np.outer(k, k)


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    x: NDArray[np.float64],
    P: NDArray[np.float64],
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
        Measurement matrix where each row corresponds to a scalar
        measurement model.
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar(z[i], var[i], H[i], x, P)


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
