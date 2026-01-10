import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._vectorops import _skew_symmetric


def _state_matrix(f_b_corr, w_b_corr, R_nb, gbc) -> NDArray[np.float64]:
    """Setup linearized state matrix, dfdx."""

    beta_gyro = 1.0 / gbc

    S = _skew_symmetric  # alias skew symmetric matrix

    # State transition matrix
    dfdx = np.zeros((9, 9))
    dfdx[0:3, 0:3] = -S(w_b_corr)  # NB! update each time step
    dfdx[0:3, 3:6] = -np.eye(3)
    dfdx[3:6, 3:6] = -beta_gyro * np.eye(3)
    dfdx[6:9, 0:3] = -R_nb @ S(f_b_corr)  # NB! update each time step

    return dfdx


def _state_transition_matrix(dt, f_b_corr, w_b_corr, R_nb, gbc) -> NDArray[np.float64]:
    """
    Setup state transition matrix, phi.

    First order approximation:
        phi = I + dt * dfdx
    """
    S = _skew_symmetric  # alias skew symmetric matrix

    beta_gyro = 1.0 / gbc

    phi = np.eye(9)
    phi[0:3, 0:3] += -dt * S(w_b_corr)
    phi[0:3, 3:6] += -dt * np.eye(3)
    phi[3:6, 3:6] += -dt * beta_gyro * np.eye(3)  # => (1 - dt*beta_gyro) * I
    phi[6:9, 0:3] += -dt * (R_nb @ S(f_b_corr))
    return phi


@njit  # type: ignore[misc]
def _update_state_transition_matrix(phi, dt, I3x3, f_b, w_b, R_nb):
    """Update state transition matrix, phi."""
    S = _skew_symmetric
    phi[0:3, 0:3] = I3x3 - dt * S(w_b)
    phi[6:9, 0:3] = -dt * R_nb @ S(f_b)


def _wn_input_matrix(R_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Setup linearized (white noise) input matrix, dfdw."""

    # Input (white noise) matrix
    dfdw = np.zeros((9, 9))
    dfdw[0:3, 0:3] = -np.eye(3)
    dfdw[3:6, 3:6] = np.eye(3)
    dfdw[6:9, 6:9] = -R_nb  # NB! update each time step (for non-isotropic case)

    return dfdw


def _wn_psd_matrix(
    vrw: float, arw: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """Setup white noise (process noise) power spectral density matrix, W."""

    # White noise power spectral density matrix
    W = np.eye(9)
    W[0:3, 0:3] *= arw**2
    W[3:6, 3:6] *= 2.0 * gbs**2 / gbc
    W[6:9, 6:9] *= vrw**2

    return W


def _wn_cov_matrix(dt, vrw: float, arw: float, gbs: float, gbc: float):
    """
    Setup process noise covariance matrix, Q.

    First order approximation:
        Q = dt @ dfdw @ W @ dfdw.T
    """
    Q = np.zeros((9, 9))
    Q[0:3, 0:3] = dt * arw**2 * np.eye(3)
    Q[3:6, 3:6] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    Q[6:9, 6:9] = dt * vrw**2 * np.eye(3)
    return Q
