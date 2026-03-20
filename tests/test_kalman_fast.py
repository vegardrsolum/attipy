import numpy as np

from attipy._kalman import _kalman_update
from attipy._kalman_fast import (
    _covariance_update_fast,
    _kalman_update_scalar_fast,
    _kalman_update_sequential_fast,
    _project_cov_ahead_fast,
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
    _kalman_update_sequential_fast(z, var, H, x_upd, P_upd, np.empty(n), np.empty(n))

    x_expect, P_expect = _kalman_update(z, np.diag(var), H, x, P)

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
    _kalman_update_scalar_fast(z, r, h, x_upd, P_upd, np.empty(n), np.empty(n))

    x_expect, P_expect = _kalman_update(z, np.array([[r]]), h.reshape(1, n), x, P)

    np.testing.assert_allclose(x_upd, x_expect)
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


def test_project_cov_ahead():

    rng = np.random.default_rng(42)

    n = 9  # state dimension

    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    phi = rng.random((n, n))
    A = rng.random((n, n))
    Q = A @ A.T + np.eye(n)  # positive semi-definite

    P_proj = P.copy()
    _project_cov_ahead_fast(P_proj, phi, Q, np.empty((n, n)))

    P_expect = phi @ P @ phi.T + Q

    np.testing.assert_allclose(P_proj, P_expect)
