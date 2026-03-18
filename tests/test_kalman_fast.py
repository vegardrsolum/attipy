import numpy as np

from attipy._kalman import _kalman_update
from attipy._kalman_fast import (
    _covariance_update_fast,
    _kalman_update_scalar_fast,
    _kalman_update_sequential_fast,
)


def test_kalman_update_sequential_fast():

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
    _kalman_update_sequential_fast(x_upd, P_upd, z, var, H, np.empty(2 * n))

    x_expect, P_expect = _kalman_update(x, P, z, np.diag(var), H)

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_kalman_scalar():

    n = 9  # state dimension

    rng = np.random.default_rng(42)

    x = rng.random(n)
    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    h = rng.random(n)
    r = rng.random()
    z = rng.random()

    x_upd = x.copy()
    P_upd = P.copy()
    _kalman_update_scalar_fast(x_upd, P_upd, z, r, h, np.empty(2 * n))

    x_expect, P_expect = _kalman_update(x, P, z, np.array([[r]]), h.reshape(1, n))

    np.testing.assert_allclose(x_upd, x_expect.ravel())
    np.testing.assert_allclose(P_upd, P_expect)


def test_covariance_update_fast():

    n = 9  # state dimension

    rng = np.random.default_rng(42)

    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    h = rng.random(n)
    k = rng.random(n)
    r = rng.random()

    P_upd = P.copy()
    _covariance_update_fast(P_upd, k, h, r, np.empty(n))

    k = np.ascontiguousarray(k[:, np.newaxis])  # (n, 1)
    h = np.ascontiguousarray(h[np.newaxis, :])  # (1, n)

    P[:, :] = (np.eye(n) - k @ h) @ P @ (np.eye(n) - k @ h).T + r * k @ k.T

    np.testing.assert_allclose(P_upd, P)
