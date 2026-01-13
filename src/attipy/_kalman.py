import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_update_old(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
    Ph: NDArray[np.float64],
    k: NDArray[np.float64],
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    for i in range(z.shape[0]):
        hi = H[i]
        vi = var[i]
        zi = z[i]

        # Kalman gain
        Ph[:] = P @ hi
        S = hi @ Ph + vi
        if S <= 1e-20:  # numerical safety
            S = 1e-20
        k[:] = Ph / S

        # State update
        x += k * (zi - hi @ x)

        # Covariance update (Joseph form)
        A[:] = I_ - np.outer(k, hi)
        P[:, :] = A @ P @ A.T + vi * np.outer(k, k)
    return x, P


@njit  # type: ignore[misc]
def _kalman_update_old2(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
    Ph: NDArray[np.float64],
    k: NDArray[np.float64],
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    for i in range(z.shape[0]):
        hi = H[i]
        vi = var[i]
        zi = z[i]

        # Kalman gain
        for j in range(9):
            Ph[j] = np.sum(P[j, :] * hi[:])
        S = np.dot(hi, Ph) + vi
        if S <= 1e-20:  # numerical safety
            S = 1e-20
        k[:] = Ph / S

        # State update
        x += k * (zi - np.dot(hi, x))

        # A = I - K @ H
        A[:] = I_
        for r in range(9):
            for c in range(9):
                A[r, c] -= k[r] * hi[c]

        # Covariance update (Joseph form)
        P[:, :] = A @ P @ A.T + vi * np.outer(k, k)
    return x, P


@njit  # type: ignore[misc]
def _kalman_update_old(
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

    The following measurement model is assumed (for each scalar measurement):

        z = h x + v ,    v ~ N(0, r)

    and the update equations are given below.

    Innovation covariance:

        S = h @ P @ h.T + R

    Kalman gain:

        K = P @ h.T / S

    State update (a posteriori)

        x = x + K @ (z - h @ x)

    Covariance update (a posteriori) (Joseph form)

        P = (I - K @ h) @ P @ (I - K @ h).T + R @ K @ K.T

    where:
    - h is the i-th row of `H`
    - r = var[i] is the i-th measurement noise variance
    - I is the identity matrix

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
        hi = H[i]
        vi = var[i]
        zi = z[i]

        # Kalman gain
        Ph = np.dot(P, hi)
        k = Ph / (np.dot(hi, Ph) + vi)

        # State update
        x += k * (zi - np.dot(hi, x))

        # Covariance update (Joseph form)
        A = I_ - np.outer(k, hi)
        P[:, :] = A @ P @ A.T + vi * np.outer(k, k)
    return x, P


@njit  # type: ignore[misc]
def _kalman_update(
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

    The following measurement model is assumed (for each scalar measurement):

        z = h x + v ,    v ~ N(0, r)

    and the update equations are given below.

    Innovation covariance:

        S = h @ P @ h.T + R

    Kalman gain:

        K = P @ h.T / S

    State update (a posteriori)

        x = x + K @ (z - h @ x)

    Covariance update (a posteriori) (Joseph form)

        P = (I - K @ h) @ P @ (I - K @ h).T + R @ K @ K.T

    where:
    - h is the i-th row of `H`
    - r = var[i] is the i-th measurement noise variance
    - I is the identity matrix

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
        h_i = np.ascontiguousarray(H[i, np.newaxis])
        r_i = np.ascontiguousarray(var[i, np.newaxis])
        z_i = np.ascontiguousarray(z[i, np.newaxis])

        # Innovation covariance
        S = h_i @ P @ h_i.T + r_i

        # Kalman gain
        K = P @ h_i.T / S

        # State update
        x = x + K @ (z_i - h_i @ x)

        # Covariance update (Joseph form)
        P[:, :] = (I_ - K @ h_i) @ P @ (I_ - K @ h_i).T + r_i * K @ K.T

    return x, P
