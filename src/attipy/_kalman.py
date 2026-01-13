import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_scalar(x, P, z, r, h, I_):
    """
    Scalar Kalman filter measurement update.

    Assuming the following measurement relationship:

        z = h x + v ,    v ~ N(0, r)

    where:
    - x is the state vector.
    - z is the scalar measurement.
    - v is the measurement noise.
    - h is the measurement matrix.
    - r is the measurement noise variance.

    The update equations are given below. They are expressed in terms of 2D array
    (column vector) operations, but implemented as 1D array operations for efficiency.
    See Parameters for shapes.

    Innovation covariance:

        S = h @ P @ h.T + r

    Kalman gain:

        k = P @ h.T / S

    State update (a posteriori)

        x = x + k @ (z - h @ x)

    Covariance update (a posteriori) (Joseph form)

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

    # Kalman gain
    Ph = np.dot(P, h)
    k = Ph / (np.dot(h, Ph) + r)

    # State update
    x += k * (z - np.dot(h, x))

    # Covariance update (Joseph form)
    A = I_ - np.outer(k, h)
    P[:, :] = A @ P @ A.T + r * np.outer(k, k)

    return x, P


@njit  # type: ignore[misc]
def _kalman_sequential(
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
        Measurement matrix; each row corresponds to a scalar measurement model.
    I_ : ndarray, shape (n, n)
        Identity matrix.

    Returns
    -------
    x : ndarray, shape (n,)
        Updated state estimate.
    P : ndarray, shape (n, n)
        Updated state covariance matrix.
    """

    for i in range(z.shape[0]):
        x[:], P[:, :] = _kalman_scalar(x, P, z[i], var[i], H[i], I_)

    return x, P
