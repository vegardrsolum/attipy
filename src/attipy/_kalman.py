import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_gain(P, h, r):
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
def _covariance_update(P, k, h, r, I_):
    """
    Update covariance estimate (Joseph form):

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
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """
    A = I_ - np.outer(k, h)
    P[:, :] = A @ P @ A.T + r * np.outer(k, k)


@njit  # type: ignore[misc]
def _kalman_update_scalar(da, bg_b, P, z, r, h, I_):
    """
    Scalar Kalman filter measurement update.

    Assumes the following error-state order:

        dx = [da, dbg_b]

    where da is the attitude error and dbg_b is the gyroscope bias error.

    Only the attitude error (da) is assumed to be non-zero, as the other states
    are updated (reset) directly.

    Parameters
    ----------
    da : ndarray, shape (3,)
        Attitude error estimate to be updated in place.
    bg_b : ndarray, shape (3,)
        Gyroscope bias estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    z : float
        Scalar measurement.
    r : float
        Scalar measurement noise variance.
    h : ndarray, shape (n,)
        Measurement matrix (row vector).
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    y = z - np.dot(h[0:3], da)  # only attitude error is non-zero
    da[:] += k[0:3] * y
    bg_b[:] += k[3:6] * y

    # Updated (a posteriori) covariance estimate (Joseph form)
    _covariance_update(P, k, h, r, I_)


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    da: NDArray[np.float64],
    bg_b: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Sequential (one-at-a-time) Kalman filter measurement update.

    Parameters
    ----------
    da : ndarray, shape (3,)
        Attitude error estimate to be updated in place.
    bg_b : ndarray, shape (3,)
        Gyroscope bias estimate to be updated in place.
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    z : ndarray, shape (m,)
        Measurement vector.
    var : ndarray, shape (m,)
        Measurement noise variances corresponding to each scalar measurement.
    H : ndarray, shape (m, n)
        Measurement matrix where each row corresponds to a scalar measurement model.
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """
    m = z.shape[0]
    for i in range(m):
        _kalman_update_scalar(da, bg_b, P, z[i], var[i], H[i], I_)


@njit  # type: ignore[misc]
def _project_cov_ahead(P, phi, Q):
    """
    Project the error covariance ahead (in place):

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
