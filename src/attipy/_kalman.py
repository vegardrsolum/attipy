import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_update(
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
        hi = H[i, :]
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
