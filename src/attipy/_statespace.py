import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._vectorops import _skew_symmetric as S


def _state_transition(dt, f_b_corr, w_b_corr, R_nb, gbc) -> NDArray[np.float64]:
    """
    Setup state transition matrix, phi.

    First order approximation:
        phi = I + dt * dfdx
    """
    phi = np.eye(9)
    phi[0:3, 0:3] += -dt * S(w_b_corr)
    phi[0:3, 3:6] += -dt * np.eye(3)
    phi[3:6, 3:6] += -dt * np.eye(3) / gbc
    phi[6:9, 0:3] += -dt * (R_nb @ S(f_b_corr))
    return phi


@njit  # type: ignore[misc]
def _update_state_transition(phi, dt, f_b, w_b, R_nb, I3x3):
    """Update state transition matrix, phi."""
    phi[0:3, 0:3] = I3x3 - dt * S(w_b)
    phi[6:9, 0:3] = -dt * R_nb @ S(f_b)


def _process_noise_cov(dt, vrw: float, arw: float, gbs: float, gbc: float):
    """
    Setup process noise covariance matrix, Q.

    First order approximation:
        Q = dt @ dfdw @ W @ dfdw.T

    Notes
    -----
    In general, Q[6:9, 6:9] should be updated each time step if R_nb changes:

        Q[6:9, 6:9] = dt * (R_nb @ Wv @ R_nb.T)

    However, if the acceleration noise (velocity random walk) is isotropic (same
    in all axes), the rotation is not needed, and we can compute Q only once.
    """
    Q = np.eye(9)
    Q[0:3, 0:3] *= dt * arw**2
    Q[3:6, 3:6] *= dt * (2.0 * gbs**2 / gbc)
    Q[6:9, 6:9] *= dt * vrw**2
    return Q


def _state_matrix(f_b_corr, w_b_corr, R_nb, gbc) -> NDArray[np.float64]:
    """Setup linearized state matrix, dfdx."""

    dfdx = np.zeros((9, 9))
    dfdx[0:3, 0:3] = -S(w_b_corr)  # NB! update each time step
    dfdx[0:3, 3:6] = -np.eye(3)
    dfdx[3:6, 3:6] = -np.eye(3) / gbc
    dfdx[6:9, 0:3] = -R_nb @ S(f_b_corr)  # NB! update each time step

    return dfdx


def _wn_input_matrix(R_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Setup linearized (white noise) input matrix, dfdw.

    Notes
    -----
    In general, dfdw[6:9, 6:9] should be updated each time step if R_nb changes:

        dfdw[6:9, 6:9] = -R_nb

    However, if the acceleration noise (velocity random walk) is isotropic (same
    in all axes), the rotation is not needed, and we can compute dfdw only once.
    """

    dfdw = np.zeros((9, 9))
    dfdw[0:3, 0:3] = -np.eye(3)
    dfdw[3:6, 3:6] = np.eye(3)
    dfdw[6:9, 6:9] = -R_nb  # NB! update each time step

    return dfdw


def _process_noise_psd(
    vrw: float, arw: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """Setup white noise (process noise) power spectral density matrix, W."""

    W = np.eye(9)
    W[0:3, 0:3] *= arw**2
    W[3:6, 3:6] *= 2.0 * gbs**2 / gbc
    W[6:9, 6:9] *= vrw**2

    return W


@njit  # type: ignore[misc]
def _dyawda(q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute yaw angle gradient wrt to the scaled Gibbs vector.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (3,)
        Unit quaternion.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Yaw angle gradient vector.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.254, John Wiley & Sons, 2021.
    """
    qw, qx, qy, qz = q_nb
    u_y = 2.0 * (qx * qy + qz * qw)
    u_x = 1.0 - 2.0 * (qy**2 + qz**2)
    u = u_y / u_x

    duda_scale = 1.0 / u_x**2
    duda_x = -(qw * qy) * (1.0 - 2.0 * qw**2) - (2.0 * qw**2 * qx * qz)
    duda_y = (qw * qx) * (1.0 - 2.0 * qz**2) + (2.0 * qw**2 * qy * qz)
    duda_z = qw**2 * (1.0 - 2.0 * qy**2) + (2.0 * qw * qx * qy * qz)
    duda = duda_scale * np.array([duda_x, duda_y, duda_z])

    dyawda = 1.0 / (1.0 + u**2) * duda

    return dyawda  # type: ignore[no-any-return]


def _measurement_matrix(q_nb) -> None:
    """Setup linearized measurement matrix, dhdx."""
    dhdx = np.zeros((4, 9))
    dhdx[0:3, 6:9] = np.eye(3)  # velocity
    dhdx[3:4, 0:3] = _dyawda(q_nb)  # heading (yaw angle) NB! update each time step
    return dhdx
