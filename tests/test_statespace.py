import numpy as np

import attipy as ap


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

    assert np.allclose(phi, phi_approx, atol=1e-8)
