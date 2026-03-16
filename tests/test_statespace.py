import numpy as np
import pytest

import attipy as ap
from attipy._statespace import (
    _process_noise_cov,
    _process_noise_cov_full,
    _process_noise_psd,
    _process_noise_psd_full,
    _state_matrix,
    _state_matrix_full,
    _state_transition,
    _state_transition_full,
    _update_state_transition_full,
    _wn_input_matrix,
    _wn_input_matrix_full,
)
from attipy._vectorops import _skew_symmetric


@pytest.fixture
def gyro_noise_params():
    arw = 0.0001
    gbs = 0.00005
    gbc = 50.0
    return arw, gbs, gbc


@pytest.fixture
def acc_noise_params():
    vrw = 0.001
    abs = 0.0005
    abc = 100.0
    return vrw, abs, abc


def test_state_matrix_full(gyro_noise_params, acc_noise_params):
    *_, abc = acc_noise_params
    *_, gbc = gyro_noise_params

    f_b = np.array([0.1, 0.2, 9.7])
    w_b = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    dfdx_out = _state_matrix_full(f_b, w_b, R_nb, abc, gbc)

    S = _skew_symmetric  # alias skew symmetric matrix

    # Linearized state matrix
    dfdx = np.zeros((15, 15))
    dfdx[0:3, 3:6] = np.eye(3)
    dfdx[3:6, 6:9] = -R_nb @ S(f_b)
    dfdx[3:6, 9:12] = -R_nb
    dfdx[6:9, 6:9] = -S(w_b)
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


def test_process_noise_psd_full(gyro_noise_params, acc_noise_params):
    vrw, abs, abc = acc_noise_params
    arw, gbs, gbc = gyro_noise_params

    W_out = _process_noise_psd_full(vrw, arw, abs, abc, gbs, gbc)

    # White noise power spectral density matrix
    W = np.eye(12)
    W[0:3, 0:3] *= vrw**2
    W[3:6, 3:6] *= arw**2
    W[6:9, 6:9] *= 2.0 * abs**2 / abc
    W[9:12, 9:12] *= 2.0 * gbs**2 / gbc
    np.testing.assert_allclose(W_out, W)


def test_state_transition_full(gyro_noise_params, acc_noise_params):
    *_, abc = acc_noise_params
    *_, gbc = gyro_noise_params

    dt = 0.1
    f_b = np.array([0.1, 0.2, 9.7])
    w_b = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()

    phi_out = _state_transition_full(dt, f_b, w_b, R_nb, abc, gbc)

    dfdx = _state_matrix_full(f_b, w_b, R_nb, abc, gbc)
    phi = np.eye(15) + dt * dfdx  # first order approximation

    np.testing.assert_allclose(phi_out, phi)


def test_update_state_transition_full(gyro_noise_params, acc_noise_params):
    *_, abc = acc_noise_params
    *_, gbc = gyro_noise_params

    dt = 0.1

    f_b = np.array([0.1, 0.2, 9.7])
    w_b = np.array([0.01, 0.02, 0.03])
    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    phi = _state_transition_full(dt, f_b, w_b, R_nb, abc, gbc)

    f_b_corr = np.array([0.15, 0.25, 9.6])
    w_b_corr = np.array([0.015, 0.025, 0.035])
    R_nb = ap.Attitude.from_euler([0.15, 0.25, 0.35]).as_matrix()
    _update_state_transition_full(phi, dt, f_b_corr, w_b_corr, R_nb)

    phi_expected = _state_transition_full(dt, f_b_corr, w_b_corr, R_nb, abc, gbc)

    np.testing.assert_allclose(phi, phi_expected)


def test_process_noise_cov_full(gyro_noise_params, acc_noise_params):
    dt = 0.1
    vrw, abs, abc = acc_noise_params
    arw, gbs, gbc = gyro_noise_params

    Q_out = _process_noise_cov_full(dt, vrw, arw, abs, abc, gbs, gbc)

    R_nb = ap.Attitude.from_euler([0.1, 0.2, 0.3]).as_matrix()
    W = _process_noise_psd_full(vrw, arw, abs, abc, gbs, gbc)
    dfdw = _wn_input_matrix_full(R_nb)

    Q = dt * dfdw @ W @ dfdw.T

    np.testing.assert_allclose(Q_out, Q, atol=1e-12)


def test_state_matrix(gyro_noise_params):
    *_, gbc = gyro_noise_params

    w_b = np.array([0.01, 0.02, 0.03])

    dfdx_out = _state_matrix(w_b, gbc)

    # Linearized state matrix
    dfdx = np.zeros((6, 6))
    dfdx[0:3, 0:3] = -_skew_symmetric(w_b)
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


def test_process_noise_psd(gyro_noise_params):
    arw, gbs, gbc = gyro_noise_params

    W_out = _process_noise_psd(arw, gbs, gbc)

    # White noise power spectral density matrix
    W = np.eye(6)
    W[0:3, 0:3] *= arw**2
    W[3:6, 3:6] *= 2.0 * gbs**2 / gbc

    np.testing.assert_allclose(W_out, W)


def test_state_transition(gyro_noise_params):
    *_, gbc = gyro_noise_params

    dt = 0.1
    w_b = np.array([0.01, 0.02, 0.03])

    phi_out = _state_transition(dt, dt * w_b, gbc)

    dfdx = _state_matrix(w_b, gbc)
    phi = np.eye(6) + dt * dfdx  # first order approximation

    np.testing.assert_allclose(phi_out, phi)


def test_process_noise_cov(gyro_noise_params):
    dt = 0.1
    arw, gbs, gbc = gyro_noise_params

    Q_out = _process_noise_cov(dt, arw, gbs, gbc)

    W = _process_noise_psd(arw, gbs, gbc)
    dfdw = _wn_input_matrix()
    Q = dt * dfdw @ W @ dfdw.T

    np.testing.assert_allclose(Q_out, Q, atol=1e-12)
