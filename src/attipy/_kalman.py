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
    Ph: NDArray[np.float64],
    k: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Sequential scalar Kalman measurement update using the Joseph-stabilized
    covariance form.

    Each measurement z[i] with variance var[i] and measurement row hᵢ is
    processed sequentially according to

        S   = hᵢᵀ P hᵢ + vᵢ
        k   = P hᵢ / S
        x⁺  = x + k (zᵢ − hᵢᵀ x)
        P⁺  = (I − k hᵢᵀ) P (I − k hᵢᵀ)ᵀ + vᵢ k kᵀ

    The Joseph form guarantees symmetry and positive semi-definiteness of the
    covariance matrix under finite-precision arithmetic.

    This implementation avoids explicit matrix products and performs the update
    using rank-1 operations, making it suitable for high-performance `numba`
    compilation and sequential (scalar) measurement processing.

    Parameters
    ----------
    x : ndarray, shape (n,)
        State estimate (updated in place).
    P : ndarray, shape (n, n)
        State covariance matrix (updated in place).
    z : ndarray, shape (m,)
        Measurement vector.
    var : ndarray, shape (m,)
        Measurement variances (assumed independent).
    H : ndarray, shape (m, n)
        Measurement matrix (one row per scalar measurement).
    Ph : ndarray, shape (n,)
        Preallocated workspace for P @ hᵢ.
    k : ndarray, shape (n,)
        Preallocated workspace for the Kalman gain.

    Returns
    -------
    x : ndarray
        Updated state estimate.
    P : ndarray
        Updated covariance matrix.

    Notes
    -----
    - Measurements are processed sequentially (scalar updates).
    - A small lower bound is applied to the innovation covariance S for numerical
      safety.
    - The covariance matrix is explicitly symmetrized after each update to
      counteract numerical drift.
    - The state dimension n is assumed fixed (n = 9).
    """

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


@njit  # type: ignore[misc]
def _kalman_update_v1p5(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    Ph: NDArray[np.float64],
    hP: NDArray[np.float64],
    K: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    n = 9  # number of states

    for i in range(z.shape[0]):
        v_i = var[i]

        # Ph = P @ h
        for a in range(n):
            s = 0.0
            for b in range(n):
                s += P[a, b] * H[i, b]
            Ph[a] = s

        # hP = h.T @ P
        for b in range(n):
            s = 0.0
            for a in range(n):
                s += H[i, a] * P[a, b]
            hP[b] = s

        # S and predicted measurement
        S = v_i
        hx = 0.0
        for a in range(n):
            S += H[i, a] * Ph[a]
            hx += H[i, a] * x[a]

        if S <= 1e-20:
            S = 1e-20
        invS = 1.0 / S

        # K = Ph / S
        for a in range(n):
            K[a] = Ph[a] * invS

        # x update
        r = z[i] - hx
        for a in range(n):
            x[a] += K[a] * r

        # Joseph expansion (no A, no matmul, no outer)
        for a in range(n):
            Ka = K[a]
            Pha = Ph[a]
            for b in range(n):
                P[a, b] = P[a, b] - Ka * hP[b] - Pha * K[b] + S * Ka * K[b]

    return x, P


@njit  # type: ignore[misc]
def _kalman_update_v4(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    z: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    for i in range(z.shape[0]):

        # Kalman gain
        Ph = P @ H[i, :]
        S = H[i, :] @ Ph + var[i]
        K = Ph / S

        # State update
        x += K * (z[i] - H[i, :] @ x)

        # Covariance update (Joseph form)
        A = I_ - np.outer(K, H[i, :])
        P = A @ P @ A.T + var[i] * np.outer(K, K)

    return x, P


@njit  # type: ignore[misc]
def _kalman_update_org(
    dx: NDArray[np.float64],
    P: NDArray[np.float64],
    dz: NDArray[np.float64],
    var: NDArray[np.float64],
    H: NDArray[np.float64],
    I_: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Sequential Kalman filter measurement update of error state, dx, and covariance
    matrix, P.
    """
    # Note: all arrays must be C-contiguous for numba njit
    for i, (dz_i, var_i) in enumerate(zip(dz, var)):
        H_i = H[i, :]  # must be C-contiguous
        K_i = P @ H_i.T / (H_i @ P @ H_i.T + var_i)
        dx += K_i * (dz_i - H_i @ dx)
        K_i = np.ascontiguousarray(K_i[:, np.newaxis])  # as C-contiguous 2D array
        H_i = np.ascontiguousarray(H_i[np.newaxis, :])  # as C-contiguous 2D array
        P = (I_ - K_i @ H_i) @ P @ (I_ - K_i @ H_i).T + var_i * K_i @ K_i.T
    return dx, P


@njit  # type: ignore[misc]
def _kalman_update_v5(
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
