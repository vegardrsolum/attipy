import numpy as np

import attipy as ap
from attipy._vectorops import _skew_symmetric


def test_state_matrix():
    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    gbc = 0.04

    dfdx = ap._statespace._state_matrix(f_b_corr, w_b_corr, R_nb, gbc)

    S = _skew_symmetric  # alias skew symmetric matrix

    beta_gyro = 1.0 / gbc

    # Linearized state matrix
    dfdx_expect = np.zeros((9, 9))
    dfdx_expect[0:3, 0:3] = -S(w_b_corr)
    dfdx_expect[0:3, 3:6] = -np.eye(3)
    dfdx_expect[3:6, 3:6] = -beta_gyro * np.eye(3)
    dfdx_expect[6:9, 0:3] = -R_nb @ S(f_b_corr)

    assert np.allclose(dfdx, dfdx_expect)


def test_state_transition():
    dt = 0.1
    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    gbc = 0.04

    phi = ap._statespace._state_transition(dt, f_b_corr, w_b_corr, R_nb, gbc)
    dfdx = ap._statespace._state_matrix(f_b_corr, w_b_corr, R_nb, gbc)

    # First order approximation
    phi_approx = np.eye(9) + dt * dfdx

    assert np.allclose(phi, phi_approx)
