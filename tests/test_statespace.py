import numpy as np
import pytest

import attipy as ap
from attipy._statespace import (
    ATT_IDX,
    BG_IDX,
    _process_noise_cov,
    _process_noise_cov_full,
    _process_noise_psd_full,
    _state_matrix_full,
    _state_transition,
    _state_transition_full,
    _update_state_transition_full,
    _wn_input_matrix_full,
)
from attipy._vectorops import _skew_symmetric


@pytest.fixture
def noise_params():
    vrw = 0.001
    arw = 0.0001
    abs = 0.0005
    abc = 100.0
    gbs = 0.00005
    gbc = 50.0
    return vrw, arw, abs, abc, gbs, gbc


def test_state_matrix_full(noise_params):
    *_, abc, _, gbc = noise_params

    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    dfdx_out = _state_matrix_full(f_b_corr, w_b_corr, R_nb, abc, gbc)

    S = _skew_symmetric  # alias skew symmetric matrix

    # Linearized state matrix
    dfdx = np.zeros((15, 15))
    dfdx[0:3, 3:6] = np.eye(3)
    dfdx[3:6, 6:9] = -R_nb @ S(f_b_corr)
    dfdx[3:6, 9:12] = -R_nb
    dfdx[6:9, 6:9] = -S(w_b_corr)
    dfdx[6:9, 12:15] = -np.eye(3)
    dfdx[9:12, 9:12] = -np.eye(3) / abc
    dfdx[12:15, 12:15] = -np.eye(3) / gbc

    np.testing.assert_allclose(dfdx_out, dfdx)


def test_wn_input_matrix_full():
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    dfdw_out = _wn_input_matrix_full(R_nb)

    # Input (white noise) matrix
    dfdw = np.zeros((15, 12))
    dfdw[3:6, 0:3] = -R_nb
    dfdw[6:9, 3:6] = -np.eye(3)
    dfdw[9:12, 6:9] = np.eye(3)
    dfdw[12:15, 9:12] = np.eye(3)

    np.testing.assert_allclose(dfdw_out, dfdw)


def test_process_noise_psd_full(noise_params):
    vrw, arw, abs, abc, gbs, gbc = noise_params

    W_out = _process_noise_psd_full(vrw, arw, abs, abc, gbs, gbc)

    # White noise power spectral density matrix
    W = np.eye(12)
    W[0:3, 0:3] *= vrw**2
    W[3:6, 3:6] *= arw**2
    W[6:9, 6:9] *= 2.0 * abs**2 / abc
    W[9:12, 9:12] *= 2.0 * gbs**2 / gbc
    np.testing.assert_allclose(W_out, W)


def test_state_transition_full(noise_params):
    *_, abc, _, gbc = noise_params

    dt = 0.1
    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    phi_out = _state_transition_full(dt, f_b_corr, w_b_corr, R_nb, abc, gbc)

    dfdx = _state_matrix_full(f_b_corr, w_b_corr, R_nb, abc, gbc)
    phi = np.eye(15) + dt * dfdx  # first order approximation

    np.testing.assert_allclose(phi_out, phi)


def test_update_state_transition_full(noise_params):
    *_, abc, _, gbc = noise_params

    dt = 0.1

    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    phi = _state_transition_full(dt, f_b_corr, w_b_corr, R_nb, abc, gbc)

    f_b_corr = np.array([0.15, 0.25, 9.6])
    w_b_corr = np.array([0.015, 0.025, 0.035])
    R_nb = ap.Attitude.from_euler([0.15, 0.25, 0.35]).as_matrix()
    _update_state_transition_full(phi, dt, f_b_corr, w_b_corr, R_nb)

    phi_expected = _state_transition_full(dt, f_b_corr, w_b_corr, R_nb, abc, gbc)

    np.testing.assert_allclose(phi, phi_expected)


def test_process_noise_cov_full(noise_params):
    dt = 0.1
    vrw, arw, abs, abc, gbs, gbc = noise_params

    Q_out = _process_noise_cov_full(dt, vrw, arw, abs, abc, gbs, gbc)

    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    W = _process_noise_psd_full(vrw, arw, abs, abc, gbs, gbc)
    dfdw = _wn_input_matrix_full(R_nb)

    Q = dt * dfdw @ W @ dfdw.T

    np.testing.assert_allclose(Q_out, Q, atol=1e-12)


def test_state_transition(noise_params):
    *_, abc, _, gbc = noise_params

    dt = 0.1
    f_b_corr = np.array([0.1, 0.2, 9.7])
    w_b_corr = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    phi_out = _state_transition(dt, w_b_corr, gbc)

    sx = np.r_[ATT_IDX, BG_IDX]
    sxx = np.ix_(sx, sx)
    phi_expect = _state_transition_full(dt, f_b_corr, w_b_corr, R_nb, abc, gbc)[sxx]

    np.testing.assert_allclose(phi_out, phi_expect)


def test_process_noise_cov(noise_params):
    dt = 0.1
    vrw, arw, abs, abc, gbs, gbc = noise_params

    Q_out = _process_noise_cov(dt, arw, gbs, gbc)

    sx = np.r_[ATT_IDX, BG_IDX]
    sxx = np.ix_(sx, sx)
    Q_expect = _process_noise_cov_full(dt, vrw, arw, abs, abc, gbs, gbc)[sxx]

    np.testing.assert_allclose(Q_out, Q_expect)
