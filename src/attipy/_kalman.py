import numpy as np
from numba import njit
from numpy.typing import NDArray


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

        x = x + k @ (z - h @ x)

    Updated (a posteriori) covariance estimate (Joseph form)

        P = (I - k @ h) @ P @ (I - k @ h).T + r @ k @ k.T

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State covariance matrix to be updated in place.
    z : float
        Measurement.
    r : float
        Measurement noise variance.
    h : ndarray, shape (n,)
        Measurement matrix.
    I_ : ndarray, shape (n, n)
        Identity matrix.
    """
    # TODO: speed up by writing out the operations explicitly

    # Innovation covariance (inverse)
    Ph = np.dot(P, h)
    s_inv = 1.0 / (np.dot(h, Ph) + r)

    # Kalman gain
    k = Ph * s_inv

    # Updated (a posteriori) state estimate
    x += k * (z - np.dot(h, x))

    # Updated (a posteriori) covariance estimate (Joseph form)
    A = I_ - np.outer(k, h)
    P[:, :] = A @ P @ A.T + r * np.outer(k, k)


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
    Sequential Kalman filter measurement update.

    Performs a series of scalar measurement updates of the state vector `x` and
    covariance matrix `P` using the corresponding measurement matrix rows in `H`,
    measurements `z`, and measurement variances `var`.

    The update is applied sequentially (one measurement at a time) and uses the
    Joseph stabilized form for the covariance update to preserve symmetry and positive
    semi-definiteness.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate to be updated in place.
    P : ndarray, shape (n, n)
        State covariance matrix to be updated in place.
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
        _kalman_update_scalar(x, P, z[i], var[i], H[i], I_)
