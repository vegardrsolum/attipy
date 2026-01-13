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
def _kalman_update(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

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
