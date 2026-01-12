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
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    n = x.size

    PH = np.empty(n, dtype=np.float64)
    K = np.empty(n, dtype=np.float64)

    for i in range(z.shape[0]):
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
            K[a] = PH[a] * invS

        # State update
        r = z_i - hx
        for a in range(n):
            x[a] += K[a] * r

        # Covariance update (Joseph form)
        A = I_ - np.outer(K, h_i)
        P = A @ P @ A.T + v_i * np.outer(K, K)

    return x, P


@njit  # type: ignore[misc]
def _kalman_update_v3(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    n = x.size

    PH = np.empty(n, dtype=np.float64)
    K = np.empty(n, dtype=np.float64)

    A = np.empty((n, n), dtype=np.float64)

    for i in range(z.shape[0]):
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
            K[a] = PH[a] * invS

        # State update
        r = z_i - hx
        for a in range(n):
            x[a] += K[a] * r

        for i in range(n):
            for j in range(n):
                A[i, j] = 0.0
            A[i, i] = 1.0
        for i in range(n):
            ki = K[i]
            for j in range(n):
                A[i, j] -= ki * h_i[j]

        P[:, :] = A @ P @ A.T

        for i in range(n):
            ki = K[i]
            for j in range(n):
                P[i, j] += v_i * ki * K[j]

        # # Symmetrize (optional but recommended)
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         s = 0.5 * (P_new[i, j] + P_new[j, i])
        #         P_new[i, j] = s
        #         P_new[j, i] = s

    return x, P


@njit  # type: ignore[misc]
def _kalman_update_v4(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    PH: NDArray[np.float64],
    K: NDArray[np.float64],
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    n = x.size

    for i in range(z.shape[0]):
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
            K[a] = PH[a] * invS

        # State update
        r = z_i - hx
        for a in range(n):
            x[a] += K[a] * r

        for i in range(n):
            for j in range(n):
                A[i, j] = 0.0
            A[i, i] = 1.0
        for i in range(n):
            ki = K[i]
            for j in range(n):
                A[i, j] -= ki * h_i[j]

        P[:, :] = A @ P @ A.T

        for i in range(n):
            ki = K[i]
            for j in range(n):
                P[i, j] += v_i * ki * K[j]

        # # Symmetrize (optional but recommended)
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         s = 0.5 * (P_new[i, j] + P_new[j, i])
        #         P_new[i, j] = s
        #         P_new[j, i] = s

    return x, P


@njit  # type: ignore[misc]
def _kalman_update_v5(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    PH: NDArray[np.float64],
    k: NDArray[np.float64],
    A: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

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

        # A = I - k @ h.T, shape (n, n)
        for i in range(n):
            for j in range(n):
                A[i, j] = 0.0
            A[i, i] = 1.0

        for i in range(n):
            ki = k[i]
            for j in range(n):
                A[i, j] -= ki * h_i[j]

        P[:, :] = A @ P @ A.T

        for i in range(n):
            ki = k[i]
            for j in range(n):
                P[i, j] += v_i * ki * k[j]

    return x, P


_kalman_update = _kalman_update_v3


# @njit  # type: ignore[misc]
# def _update_dx_P(
#     dx: NDArray[np.float64],
#     P: NDArray[np.float64],
#     dz: NDArray[np.float64],
#     var: NDArray[np.float64],
#     H: NDArray[np.float64],
# ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     """
#     Update state and covariance estimates with a series of measurements.
#     """
#     n = dx.shape[0]

#     # Preallocation
#     PH = np.empty(n, dtype=np.float64)  # P @ H.T
#     K = np.empty(n, dtype=np.float64)

#     for i in range(dz.shape[0]):
#         z = dz[i]
#         h = H[i, :]
#         v = var[i]

#         # P @ H.T
#         for a in range(n):
#             s = 0.0
#             for b in range(n):
#                 s += P[a, b] * h[b]
#             PH[a] = s

#         # S = H @ P @ H.T + v, hx = H @ dx
#         S = v
#         hx = 0.0
#         for a in range(n):
#             S += h[a] * PH[a]
#             hx += h[a] * dx[a]

#         # Precalculate inverse, since multiply is faster than divide
#         invS = 1.0 / S

#         # Kalman gain: K = P @ H.T / S
#         for a in range(n):
#             K[a] = PH[a] * invS

#         # State update: dx += K * (dz - H @ dx)
#         r = z - hx
#         for a in range(n):
#             dx[a] += K[a] * r

#         # Covariance update: P = P - K H P - P H.T K.T + S K K.T (Joseph, expanded)
#         for a in range(n):
#             Ka = K[a]
#             PHa = PH[a]
#             for b in range(n):
#                 P[a, b] = P[a, b] - Ka * PH[b] - PHa * K[b] + S * Ka * K[b]

#     return dx, P
