import numpy as np

from attipy._kalman import _kalman_update
from attipy._kalman_fast import (
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
    _kalman_update_sequential_fast(z, var, H, x_upd, P_upd, np.empty(n))

    x_expect, P_expect = _kalman_update(z, np.diag(var), H, x, P)

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)
    np.testing.assert_allclose(P_upd, P_upd.T)


def test_kalman_update_scalar_fast():

    rng = np.random.default_rng(42)

    n = 9  # state dimension

    x = rng.random(n)
    A = rng.random((n, n))
    P = A @ A.T + np.eye(n)  # positive semi-definite
    h = rng.random(n)
    r = rng.random()
    z = rng.random()

    x_upd = x.copy()
    P_upd = P.copy()
    _kalman_update_scalar_fast(z, r, h, x_upd, P_upd, np.empty(n))

    x_expect, P_expect = _kalman_update(z, np.array([[r]]), h.reshape(1, n), x, P)

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)
    np.testing.assert_allclose(P_upd, P_upd.T)


def test_kalman_update_scalar_fast_joseph_form_stability():
    """
    Repeated, highly informative, near-collinear measurement updates in single
    precision are a worst case for cancellation error in the covariance update.
    The Joseph form must keep the covariance matrix symmetric and (numerically)
    positive semi-definite even under this stress, unlike the algebraically
    equivalent but less stable P - K(Ph)^T shortcut, which drifts far from PSD
    in the same scenario.
    """
    rng = np.random.default_rng(1)
    n = 6

    h0 = rng.standard_normal(n).astype(np.float32)
    h0 /= np.linalg.norm(h0)

    P = np.eye(n, dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)
    tmp = np.empty(n, dtype=np.float32)

    for _ in range(20_000):
        h = h0 + rng.standard_normal(n).astype(np.float32) * np.float32(1e-6)
        r = np.float32(1e-8)
        z = np.float32(0.0)
        _kalman_update_scalar_fast(z, r, h.astype(np.float32), x, P, tmp)

    np.testing.assert_allclose(P, P.T)

    eigvals = np.linalg.eigvalsh(P.astype(np.float64))
    assert eigvals.min() > -1e-4


def test_project_cov_ahead_fast():

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
    np.testing.assert_allclose(P_proj, P_proj.T)
