import numpy as np

from attipy._kalman import (
    _kalman_update,
    _kalman_update_scalar,
    _kalman_update_sequential,
    _project_cov_ahead,
)


def test_kalman_update():

    rng = np.random.default_rng(42)

    m = 4  # number of measurements
    n = 9  # state dimension

    x = rng.random(n)
    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    H = rng.random((m, n))
    R = rng.random((m, m))
    z = rng.random(m)

    x_upd, P_upd = _kalman_update(x, P, z, R, H)

    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_expect = x + K @ (z - H @ x)
    P_expect = (np.eye(n) - K @ H) @ P @ (np.eye(n) - K @ H).T + K @ R @ K.T

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_kalman_update_sequential():

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
    _kalman_update_sequential(x_upd, P_upd, z, var, H)

    x_expect, P_expect = _kalman_update(x, P, z, np.diag(var), H)

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_kalman_update_scalar():

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
    _kalman_update_scalar(x_upd, P_upd, z, r, h)

    x_expect, P_expect = _kalman_update(x, P, z, np.array([[r]]), h.reshape(1, n))

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_project_cov_ahead():

    rng = np.random.default_rng(42)

    n = 9  # state dimension

    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    phi = rng.random((n, n))
    A = rng.random((n, n))
    Q = A @ A.T + np.eye(n)  # positive semi-definite

    P_proj = P.copy()
    _project_cov_ahead(P_proj, phi, Q)

    P_expect = phi @ P @ phi.T + Q

    np.testing.assert_allclose(P_proj, P_expect)
