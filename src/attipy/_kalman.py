import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_gain(P, h, r):
    """
    Compute the Kalman gain for a scalar measurement.

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix.
    h : ndarray, shape (n,)
        Measurement sensitivity matrix (row vector).
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
    Update covariance estimate (Joseph form).

    Parameters
    ----------
    P : ndarray, shape (n, n)
        State error covariance matrix to be updated in place.
    k : ndarray, shape (n,)
        Kalman gain vector.
    h : ndarray, shape (n,)
        Measurement sensitivity matrix (row vector).
    r : float
        Scalar measurement noise variance.
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """
    A = I_ - np.outer(k, h)
    P[:] = A @ P @ A.T + r * np.outer(k, k)


@njit  # type: ignore[misc]
def _kalman_update_scalar(x, P, z, r, h, I_):
    """
    Scalar Kalman filter measurement update.

    Assumes the following scalar measurement model:

        z = h x + v,    v ~ N(0, r)

    where:
    - x is the state vector,
    - z is the scalar measurement,
    - h is the measurement (row) vector,
    - v is zero-mean Gaussian measurement noise,
    - r is the measurement noise variance.

    The Kalman update equations are given below. They are expressed in terms of
    matrix (2D array) operations, but implemented using 1D array operations for
    computational efficiency. See the parameter descriptions for expected array
    shapes.

    Innovation covariance:

        s = h @ P @ h.T + r

    Kalman gain:

        k = P @ h.T / s

    Updated (a posteriori) state estimate

        x = x + k * (z - h @ x)

    Updated (a posteriori) covariance estimate (Joseph form)

        P = (I - k @ h) @ P @ (I - k @ h).T + r @ k @ k.T

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
        Measurement sensitivity matrix (row vector).
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    x[:] += k * (z - np.dot(h, x))

    # Updated (a posteriori) covariance estimate (Joseph form)
    _covariance_update(P, k, h, r, I_)


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Sequential Kalman filter measurement update. Updates (in place) the state, x,
    and covariance matrix, P, using the Kalman filter update equations given below.

    Innovation covariance:

        S = H @ P @ H.T + R

    Kalman gain:

        K = P @ H.T @ inv(S)

    Updated (a posteriori) state estimate:

        x = x + K @ (z - H @ x)

    Updated (a posteriori) covariance estimate (Joseph form):

        P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    The update is applied sequentially (one measurement at a time) and uses the
    Joseph stabilized form for the covariance update to preserve symmetry and positive
    semi-definiteness.

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
        Measurement matrix; each row corresponds to a scalar measurement model.
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """

    for i in range(z.shape[0]):
        _kalman_update_scalar(x, P, z[i], var[i], H[i], I_)
