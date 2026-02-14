import numpy as np
import pytest

import attipy as ap
from attipy._statespace import (
    _process_noise_cov_matrix,
    _process_noise_psd,
    _state_matrix,
    _state_transition_matrix,
    _update_state_transition_matrix,
    _wn_input_matrix,
)
from attipy._vectorops import _skew_symmetric


@pytest.fixture
def noise_params():
    arw = 0.0001
    gbs = 0.00005
    gbc = 50.0
    return arw, gbs, gbc


def test_state_matrix(noise_params):
    *_, gbc = noise_params

    w_b_corr = np.array([0.01, 0.02, 0.03])
    dfdx_out = _state_matrix(w_b_corr, gbc)

    S = _skew_symmetric  # alias skew symmetric matrix

    # Linearized state matrix
    dfdx = np.zeros((6, 6))
    dfdx[0:3, 0:3] = -S(w_b_corr)  # NB! update each time step
    dfdx[0:3, 3:6] = -np.eye(3)
    dfdx[3:6, 3:6] = -np.eye(3) / gbc
    np.testing.assert_allclose(dfdx_out, dfdx)


def test_wn_input_matrix():
    dfdw_out = _wn_input_matrix()

    # Input (white noise) matrix
    dfdw = np.zeros((6, 6))
    dfdw[0:3, 0:3] = -np.eye(3)
    dfdw[3:6, 3:6] = np.eye(3)

    np.testing.assert_allclose(dfdw_out, dfdw)


def test_process_noise_psd(noise_params):
    arw, gbs, gbc = noise_params

    W_out = _process_noise_psd(arw, gbs, gbc)

    # White noise power spectral density matrix
    W = np.eye(6)
    W[0:3, 0:3] *= arw**2
    W[3:6, 3:6] *= 2.0 * gbs**2 / gbc
    np.testing.assert_allclose(W_out, W)


def test_state_transition(noise_params):
    *_, gbc = noise_params

    dt = 0.1
    w_b_corr = np.array([0.01, 0.02, 0.03])

    phi_out = _state_transition_matrix(dt, w_b_corr, gbc)

    dfdx = _state_matrix(w_b_corr, gbc)
    phi = np.eye(6) + dt * dfdx  # first order approximation

    np.testing.assert_allclose(phi_out, phi)


def test_update_state_transition(noise_params):
    *_, gbc = noise_params

    dt = 0.1

    w_b_corr = np.array([0.01, 0.02, 0.03])
    phi = _state_transition_matrix(dt, w_b_corr, gbc)

    w_b_corr = np.array([0.015, 0.025, 0.035])
    _update_state_transition_matrix(phi, dt, w_b_corr)

    phi_expected = _state_transition_matrix(dt, w_b_corr, gbc)

    np.testing.assert_allclose(phi, phi_expected)


def test_process_noise_cov(noise_params):
    dt = 0.1
    arw, gbs, gbc = noise_params

    Q_out = _process_noise_cov_matrix(dt, arw, gbs, gbc)

    W = _process_noise_psd(arw, gbs, gbc)
    dfdw = _wn_input_matrix()

    Q = dt * dfdw @ W @ dfdw.T

    np.testing.assert_allclose(Q_out, Q, atol=1e-12)
