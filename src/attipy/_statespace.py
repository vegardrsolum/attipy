import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._vectorops import _skew_symmetric as S


def _state_transition_matrix(dt, w_b, gbc):
    """
    Setup state transition matrix, phi, using the first-order approximation:

        phi = I + dt * dfdx

    where dfdx denotes the linearized state matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    phi : ndarray, shape (6, 6)
        State transition matrix.
    """
    phi = np.eye(6)
    phi[0:3, 0:3] -= dt * S(w_b)  # NB! update each time step
    phi[0:3, 3:6] -= dt * np.eye(3)
    phi[3:6, 3:6] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _update_state_transition_matrix(
    phi: NDArray[np.float64],
    dt: float,
    w_b: NDArray[np.float64],
):
    """
    Update the state transition matrix, phi, in place.

    Parameters
    ----------
    phi : ndarray, shape (6, 6)
        State transition matrix to be updated in place.
    dt : float
        Time step.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    """
    wx, wy, wz = w_b
    phi[0, 1] = dt * wz
    phi[0, 2] = -dt * wy
    phi[1, 0] = -dt * wz
    phi[1, 2] = dt * wx
    phi[2, 0] = dt * wy
    phi[2, 1] = -dt * wx


def _process_noise_cov_matrix(dt, arw, gbs, gbc):
    """
    Setup process noise covariance matrix, Q, using the first-order approximation:

        Q = dt @ dfdw @ W @ dfdw.T

    Parameters
    ----------
    dt : float
        Time step in seconds.
    arw : float
        Angular random walk (gyroscope noise density) in rad/√Hz.
    gbs : float
        Gyro bias stability (bias instability) in rad/s.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    Q : ndarray, shape (15, 15)
        Process noise covariance matrix.
    """
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = dt * arw**2 * np.eye(3)
    Q[3:6, 3:6] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _state_matrix(
    w_b: NDArray[np.float64],
    gbc: float,
) -> NDArray[np.float64]:
    """
    Setup linearized state matrix, dfdx.

    Parameters
    ----------
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    dfdx : ndarray, shape (6, 6)
        Linearized state matrix.
    """
    dfdx = np.zeros((6, 6))
    dfdx[0:3, 0:3] = -S(w_b)  # NB! update each time step
    dfdx[0:3, 3:6] = -np.eye(3)
    dfdx[3:6, 3:6] = -np.eye(3) / gbc
    return dfdx


def _wn_input_matrix() -> NDArray[np.float64]:
    """
    Setup linearized (white noise) input matrix, dfdw.

    Returns
    -------
    dfdw : ndarray, shape (6, 6)
        Linearized (white noise) input matrix.
    """
    dfdw = np.zeros((6, 6))
    dfdw[0:3, 0:3] = -np.eye(3)
    dfdw[3:6, 3:6] = np.eye(3)
    return dfdw


def _process_noise_psd(arw: float, gbs: float, gbc: float) -> NDArray[np.float64]:
    """
    Setup white noise (process noise) power spectral density matrix, W.

    Parameters
    ----------
    arw : float
        Angular random walk (gyroscope noise density) in rad/√Hz.
    gbs : float
        Gyro bias stability (bias instability) in rad/s.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    W : ndarray, shape (6, 6)
        Process noise power spectral density matrix.
    """
    W = np.eye(6)
    W[0:3, 0:3] *= arw**2
    W[3:6, 3:6] *= 2.0 * gbs**2 / gbc
    return W


@njit  # type: ignore[misc]
def _dyawda(q_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute yaw angle gradient wrt to the scaled Gibbs vector.

    Defined in terms of scaled Gibbs vector in ref [1]_, but implemented in terms of
    unit quaternion here to avoid singularities.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
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


def _measurement_matrix(q_nb, vg_b):
    """
    Setup linearized measurement matrix, dhdx.

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Unit quaternion.
    vg_b : ndarray, shape (3,)
        Gravity reference (unit) vector expressed in the body frame.

    Returns
    -------
    dhdx : ndarray, shape (4, 6)
        Linearized measurement matrix.
    """
    dhdx = np.zeros((4, 6))
    dhdx[0:3, 0:3] = S(vg_b)  # NB! update each time step
    dhdx[3:4, 0:3] = _dyawda(q_nb)  # NB! update each time step
    return dhdx


@njit  # type: ignore[misc]
def _update_measurement_matrix_yaw(dhdx, q_nb):
    """
    Heading (yaw angle) part of the measurement matrix, shape (6,).
    """
    dhdx[3:4, 0:3] = _dyawda(q_nb)
    return dhdx[3]


@njit  # type: ignore[misc]
def _update_measurement_matrix_gref(dhdx, vg_b):
    """
    Gravity reference vector part of the measurement matrix, shape (3, 6).
    """
    dhdx[0:3, 0:3] = S(vg_b)
    return dhdx[0:3]
