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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    for i in range(z.shape[0]):
        hi = H[i]
        vi = var[i]
        zi = z[i]

        # Kalman gain
        Ph = P @ hi
        S = hi @ Ph + vi
        if S <= 1e-20:  # optional numerical safety
            S = 1e-20
        K = Ph / S

        # State update
        x += K * (zi - hi @ x)

        # Covariance update (Joseph form)
        A = I_ - np.outer(K, hi)
        P[:, :] = A @ P @ A.T + vi * np.outer(K, K)

    return x, P
