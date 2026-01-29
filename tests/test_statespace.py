import numpy as np
import pytest

import attipy as ap
from attipy._statespace import (
    _process_noise_cov,
    _process_noise_psd,
    _state_matrix,
    _state_transition,
    _update_state_transition,
    _wn_input_matrix,
)
from attipy._vectorops import _skew_symmetric


@pytest.fixture
def noise_params():
    vrw = 0.001
    arw = 0.0001
    gbs = 0.00005
    gbc = 50.0
    return vrw, arw, gbs, gbc


def test_state_matrix(noise_params):
    *_, gbc = noise_params

    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    dfdx_out = _state_matrix(f_b_corr, w_b_corr, R_nb, gbc)

    S = _skew_symmetric  # alias skew symmetric matrix

    # Linearized state matrix
    dfdx = np.zeros((9, 9))
    dfdx[0:3, 0:3] = -S(w_b_corr)
    dfdx[0:3, 3:6] = -np.eye(3)
    dfdx[6:9, 0:3] = -R_nb @ S(f_b_corr)
    dfdx[3:6, 3:6] = -np.eye(3) / gbc

    np.testing.assert_allclose(dfdx_out, dfdx)


def test_wn_input_matrix():
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    dfdw_out = _wn_input_matrix(R_nb)

    # Input (white noise) matrix
    dfdw = np.zeros((9, 9))
    dfdw[0:3, 0:3] = -np.eye(3)
    dfdw[6:9, 6:9] = -R_nb
    dfdw[3:6, 3:6] = np.eye(3)

    np.testing.assert_allclose(dfdw_out, dfdw)


def test_process_noise_psd(noise_params):
    vrw, arw, gbs, gbc = noise_params

    W_out = _process_noise_psd(vrw, arw, gbs, gbc)

    # White noise power spectral density matrix
    W = np.eye(9)
    W[0:3, 0:3] *= arw**2
    W[6:9, 6:9] *= vrw**2
    W[3:6, 3:6] *= 2.0 * gbs**2 / gbc

    np.testing.assert_allclose(W_out, W)


def test_state_transition(noise_params):
    *_, gbc = noise_params

    dt = 0.1
    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    phi_out = _state_transition(dt, f_b_corr, w_b_corr, R_nb, gbc)

    dfdx = _state_matrix(f_b_corr, w_b_corr, R_nb, gbc)
    phi = np.eye(9) + dt * dfdx  # first order approximation

    np.testing.assert_allclose(phi_out, phi)


def test_update_state_transition(noise_params):
    *_, gbc = noise_params

    dt = 0.1

    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    phi = _state_transition(dt, f_b_corr, w_b_corr, R_nb, gbc)

    f_b_corr = np.array([0.15, 0.25, 9.6])
    w_b_corr = np.array([0.015, 0.025, 0.035])
    R_nb = ap.Attitude.from_euler([0.15, 0.25, 0.35]).as_matrix()
    _update_state_transition(phi, dt, f_b_corr, w_b_corr, R_nb)

    phi_expected = _state_transition(dt, f_b_corr, w_b_corr, R_nb, gbc)

    np.testing.assert_allclose(phi, phi_expected)


def test_process_noise_cov(noise_params):
    dt = 0.1
    vrw, arw, gbs, gbc = noise_params

    Q_out = _process_noise_cov(dt, vrw, arw, gbs, gbc)

    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    W = _process_noise_psd(vrw, arw, gbs, gbc)
    dfdw = _wn_input_matrix(R_nb)

    Q = dt * dfdw @ W @ dfdw.T

    np.testing.assert_allclose(Q_out, Q, atol=1e-12)
