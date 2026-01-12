import numpy as np

import attipy as ap
from attipy._kalman import _kalman_update
from attipy._statespace import _measurement_matrix
from attipy._transforms import _quat_from_euler_zyx


def test_kalman_update():

    rng = np.random.default_rng(42)

    x = np.zeros(9)
    P = np.eye(9) + 0.01 * rng.random((9, 9))
    q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))  # arbitrary attitude
    H = _measurement_matrix(q_nb)
    m = H.shape[0]
    var = rng.random(m)
    z = rng.random(m)

    x_upd, P_upd = _kalman_update(x.copy(), P.copy(), z, var, H, np.eye(9))

    R = np.diag(var)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_expect = x + K @ (z - H @ x)
    P_expect = (np.eye(9) - K @ H) @ P @ (np.eye(9) - K @ H).T + K @ R @ K.T

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)


def test_kalman_update_v5():

    rng = np.random.default_rng(42)

    x = rng.random(9)
    P = np.eye(9) + 0.01 * rng.random((9, 9))
    q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))  # arbitrary attitude
    H = _measurement_matrix(q_nb)
    m = H.shape[0]
    var = rng.random(m)
    z = rng.random(m)

    n = H.shape[1]
    PH = np.empty(n, dtype=np.float64)
    k = np.empty(n, dtype=np.float64)
    A = np.empty((n, n), dtype=np.float64)
    
    x_upd, P_upd = ap._kalman._kalman_update_v5(x.copy(), P.copy(), z, var, H, PH, k, A)

    R = np.diag(var)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_expect = x + K @ (z - H @ x)
    P_expect = (np.eye(9) - K @ H) @ P @ (np.eye(9) - K @ H).T + K @ R @ K.T

    np.testing.assert_allclose(x_upd, x_expect)
    np.testing.assert_allclose(P_upd, P_expect)
