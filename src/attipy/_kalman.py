import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _kalman_update_v1(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    for i in range(z.shape[0]):
        h_i = H[i, :]
        z_i = z[i]
        v_i = var[i]

        # Kalman gain
        PHt = P @ h_i
        S = h_i @ PHt + v_i
        K = PHt / S  # shape (n,)

        # State update
        x += K * (z_i - h_i @ x)

        # Covariance update (Joseph form)
        A = I_ - np.outer(K, h_i)
        P = A @ P @ A.T + v_i * np.outer(K, K)

    return x, P


@njit  # type: ignore[misc]
def _kalman_update_v2(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    PH: NDArray[np.float64],
    k: NDArray[np.float64],
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Update state and covariance estimates with a series of measurements, using the
    sequential scalar Joseph form:

        P = (I - k @ h.T) @ P @ (I - k @ h.T).T + v * k @ k.T
    """

    n = 9  # number of states

    for i in range(z.size):
        h_i = H[i, :]
        z_i = z[i]
        v_i = var[i]

        for a in range(n):
            s = 0.0
            for b in range(n):
                s += P[a, b] * h_i[b]
            PH[a] = s
            
        S = v_i
        hx = 0.0
        for a in range(n):
            S += h_i[a] * PH[a]
            hx += h_i[a] * x[a]

        invS = 1.0 / S

        # Kalman gain
        for a in range(n):
            k[a] = PH[a] * invS

        # State update
        r = z_i - hx
        for a in range(n):
            x[a] += k[a] * r

        # A = I - k @ h.T
        for a in range(n):
            for b in range(n):
                A[a, b] = 0.0
            A[a, a] = 1.0
        for a in range(n):
            ka = k[a]
            for b in range(n):
                A[a, b] -= ka * h_i[b]

        P[:, :] = A @ P @ A.T

        for a in range(n):
            ka = k[a]
            for b in range(n):
                P[a, b] += v_i * ka * k[b]

    return x, P


@njit  # type: ignore[misc]
def _kalman_update_v3(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    Ph: NDArray[np.float64],   # P @ h
    k: NDArray[np.float64],    # Kalman gain
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    n = 9  # number of states

    for i in range(z.size):
        v_i = var[i]

        # Ph = P @ h
        for a in range(n):
            s = 0.0
            for b in range(n):
                s += P[a, b] * H[i, b]
            Ph[a] = s

        # Innovation covariance and predicted measurement
        S = v_i
        hx = 0.0
        for a in range(n):
            hia = H[i, a]
            S += hia * Ph[a]
            hx += hia * x[a]

        # Safety (optional)
        if S <= 1e-20:
            S = 1e-20

        invS = 1.0 / S

        # Kalman gain
        for a in range(n):
            k[a] = Ph[a] * invS

        # State update
        r = z[i] - hx
        for a in range(n):
            x[a] += k[a] * r

        # Joseph covariance update:
        # P = P - k Ph^T - Ph k^T + S k k^T
        for a in range(n):
            ka = k[a]
            pha = Ph[a]
            for b in range(n):
                P[a, b] = P[a, b] - ka * Ph[b] - pha * k[b] + S * ka * k[b]

        # Optional symmetry enforcement
        for a in range(n):
            for b in range(a + 1, n):
                s = 0.5 * (P[a, b] + P[b, a])
                P[a, b] = s
                P[b, a] = s

    return x, P
