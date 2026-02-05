import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_gain(P, h, r):
    """
    Compute the Kalman gain for a scalar measurement.
    """

    # Innovation covariance (inverse)
    Ph = np.dot(P, h)
    s_inv = 1.0 / (np.dot(h, Ph) + r)

    # Kalman gain
    k = Ph * s_inv

    return k


@njit  # type: ignore[misc]
def _state_update(da, p, v, bg, k, z):
    """
    Update state estimates:
        x = x + k * z
    """
    p[0] += k[0] * z
    p[1] += k[1] * z
    p[2] += k[2] * z
    v[0] += k[3] * z
    v[1] += k[4] * z
    v[2] += k[5] * z
    da[0] += k[6] * z
    da[1] += k[7] * z
    da[2] += k[8] * z
    bg[0] += k[9] * z
    bg[1] += k[10] * z
    bg[2] += k[11] * z


@njit  # type: ignore[misc]
def _covariance_update(P, k, h, r, I_):
    """
    Update covariance estimate (Joseph form).
    """
    A = I_ - np.outer(k, h)
    P[:, :] = A @ P @ A.T + r * np.outer(k, k)


@njit  # type: ignore[misc]
def _kalman_update_scalar(da, p, v, bg, P, z, r, h, I_):
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

        x = x + k * z

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

    # Kalman gain
    k = _kalman_gain(P, h, r)

    # Updated (a posteriori) state estimate
    _state_update(da, p, v, bg, k, z)

    # Updated (a posteriori) covariance estimate (Joseph form)
    _covariance_update(P, k, h, r, I_)


@njit  # type: ignore[misc]
def _kalman_update_sequential(
    da: NDArray[np.float64],
    p: NDArray[np.float64],
    v: NDArray[np.float64],
    bg: NDArray[np.float64],
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
        _kalman_update_scalar(da, p, v, bg, P, z[i], var[i], H[i], I_)
