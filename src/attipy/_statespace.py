import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._vectorops import _skew_symmetric as S

# Error-state order
ATT_IDX = slice(0, 3)  # attitude error (2x Gibbs vector)
BG_IDX = slice(3, 6)  # gyroscope bias error


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


def _state_transition(
    dt: float, dtheta: NDArray[np.float64], gbc: float
) -> NDArray[np.float64]:
    """
    Set up the state transition matrix, phi, using the first-order approximation:

        phi = I + dt * dfdx

    where dfdx denotes the linearized state matrix.

    Assumes the following 6 states in order:
    - Attitude (3)
    - Gyro bias (3)

    Parameters
    ----------
    dt : float
        Time step in seconds.
    dtheta : ndarray, shape (3,)
        Attitude increment (coning integral) in radians.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    phi : ndarray, shape (6, 6)
        State transition matrix.
    """
    phi = np.eye(6)
    phi[ATT_IDX, ATT_IDX] -= S(dtheta)  # NB! update each time step
    phi[ATT_IDX, BG_IDX] -= dt * np.eye(3)
    phi[BG_IDX, BG_IDX] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _state_transition_update(
    phi: NDArray[np.float64],
    dtheta: NDArray[np.float64],
) -> None:
    """
    Update the state transition matrix, phi, in place:

        phi[0:3, 0:3] = I - S(dtheta)

    Assumes the following 6 states in order:
    - Attitude (3)
    - Gyro bias (3)

    Parameters
    ----------
    phi : ndarray, shape (6, 6)
        State transition matrix to be updated in place.
    dtheta : ndarray, shape (3,)
        Attitude increment (coning integral) in radians.

    Notes
    -----
    Assuming the first order approximation:

        phi = I + dt * dfdx

    where dfdx denotes the linearized state matrix.
    """
    dtx, dty, dtz = dtheta
    phi[0, 1] = dtz
    phi[0, 2] = -dty
    phi[1, 0] = -dtz
    phi[1, 2] = dtx
    phi[2, 0] = dty
    phi[2, 1] = -dtx


def _process_noise_cov(
    dt: float, arw: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """
    Set up the process noise covariance matrix, Q, using the first-order approximation:

        Q = dt @ dfdw @ W @ dfdw.T

    Assumes the following 6 states in order:
    - Attitude (3)
    - Gyro bias (3)

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
    Q : ndarray, shape (6, 6)
        Process noise covariance matrix.
    """
    Q = np.zeros((6, 6))
    Q[ATT_IDX, ATT_IDX] = dt * arw**2 * np.eye(3)
    Q[BG_IDX, BG_IDX] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _measurement_matrix(
    q_nb: NDArray[np.float64], vg_b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Set up the linearized measurement matrix, dhdx.

    Assumes the following 6 states in order:
    - Attitude (3)
    - Gyro bias (3)

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Unit quaternion.
    vg_b : ndarray, shape (3,)
        Gravity reference unit vector expressed in the body frame.

    Returns
    -------
    dhdx : ndarray, shape (4, 6)
        Linearized measurement matrix.
    """
    dhdx = np.zeros((4, 6))
    dhdx[0:3, ATT_IDX] = S(vg_b)  # gravity ref vector (NB! update)
    dhdx[3:4, ATT_IDX] = _dyawda(q_nb)  # heading (yaw angle) (NB! update)
    return dhdx


def _state_matrix(
    w_b: NDArray[np.float64],
    gbc: float,
) -> NDArray[np.float64]:
    """
    Set up the linearized state matrix, dfdx.

    Assumes the following 6 states in order:
    - Attitude (3)
    - Gyro bias (3)

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
    dfdx[ATT_IDX, ATT_IDX] = -S(w_b)  # NB! update each time step
    dfdx[ATT_IDX, BG_IDX] = -np.eye(3)
    dfdx[BG_IDX, BG_IDX] = -np.eye(3) / gbc
    return dfdx


def _wn_input_matrix() -> NDArray[np.float64]:
    """
    Set up the linearized (white noise) input matrix, dfdw.

    Assumes the following 6 states in order:
    - Attitude (3)
    - Gyro bias (3)

    and the following 6 white noise inputs in order:
    - Gyroscope white noise (3)
    - Gyroscope bias white noise (3)

    Returns
    -------
    dfdw : ndarray, shape (6, 6)
        Linearized (white noise) input matrix.
    """
    dfdw = np.zeros((6, 6))
    dfdw[ATT_IDX, 0:3] = -np.eye(3)
    dfdw[BG_IDX, 3:6] = np.eye(3)
    return dfdw


def _process_noise_psd(arw: float, gbs: float, gbc: float) -> NDArray[np.float64]:
    """
    Set up the white noise (process noise) power spectral density matrix, W.

    Assumes the following 6 white noise inputs in order:
    - Gyroscope white noise (3)
    - Gyroscope bias white noise (3)

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
