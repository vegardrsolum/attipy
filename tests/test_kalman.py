import numpy as np

from attipy._kalman import _kalman_update_scalar, _kalman_update_sequential


def test_kalman_sequential():

    rng = np.random.default_rng(42)

    m = 4  # number of measurements
    n = 9  # state dimension

    x = rng.random(n)
    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    H = rng.random((m, n))
    var = rng.random(m)
    z = rng.random(m)

    x_upd = x.copy()
    P_upd = P.copy()
    _kalman_update_sequential(x_upd, P_upd, z, var, H, np.eye(n))

    R = np.diag(var)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_expect = x + K @ (z - H @ x)
    P_expect = (np.eye(9) - K @ H) @ P @ (np.eye(9) - K @ H).T + K @ R @ K.T

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_kalman_scalar():

    n = 9  # state dimension

    rng = np.random.default_rng(42)

    x = rng.random(n)
    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    h = rng.random(n)
    r = rng.random(1)
    z = rng.random(1)

    x_upd = x.copy()
    P_upd = P.copy()
    _kalman_update_scalar(x_upd, P_upd, z, r, h, np.eye(n))

    x = np.ascontiguousarray(x[:, np.newaxis])  # (n, 1)
    h = np.ascontiguousarray(h[np.newaxis, :])  # (1, n)
    z = np.ascontiguousarray(z[:, np.newaxis])  # (1, 1)

    S = h @ P @ h.T + r
    K = P @ h.T / S
    x = x + K @ (z - h @ x)
    P[:, :] = (np.eye(n) - K @ h) @ P @ (np.eye(n) - K @ h).T + r * K @ K.T

    np.testing.assert_allclose(x_upd, x.ravel())
    np.testing.assert_allclose(P_upd, P)
