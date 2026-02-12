import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._transforms import _matrix_from_quat
from ._vectorops import _skew_symmetric as S

POS_IDX = slice(0, 3)
VEL_IDX = slice(3, 6)
ATT_IDX = slice(6, 9)
BA_IDX = slice(9, 12)
BG_IDX = slice(12, 15)


def _state_transition(
    dt: float,
    f_b: NDArray[np.float64],
    w_b: NDArray[np.float64],
    R_nb: NDArray[np.float64],
    abc: float,
    gbc: float,
) -> NDArray[np.float64]:
    """
    Setup state transition matrix, phi, using the first-order approximation:

        phi = I + dt * dfdx

    where dfdx denotes the linearized state matrix.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    f_b : ndarray, shape (3,)
        Specific force measurement (bias corrected) in body frame.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    R_nb : ndarray, shape (3, 3)
        Rotation matrix (from body to navigation frame).
    abc : float
        Accelerometer bias correlation time in seconds.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    phi : ndarray, shape (15, 15)
        State transition matrix.
    """
    phi = np.eye(15)
    phi[POS_IDX, VEL_IDX] += dt * np.eye(3)
    phi[VEL_IDX, ATT_IDX] -= dt * R_nb @ S(f_b)  # NB! update each time step
    phi[VEL_IDX, BA_IDX] -= dt * R_nb  # NB! update each time step
    phi[ATT_IDX, ATT_IDX] -= dt * S(w_b)  # NB! update each time step
    phi[ATT_IDX, BG_IDX] -= dt * np.eye(3)
    phi[BA_IDX, BA_IDX] -= dt * np.eye(3) / abc
    phi[BG_IDX, BG_IDX] -= dt * np.eye(3) / gbc
    return phi


@njit  # type: ignore[misc]
def _update_state_transition(
    phi: NDArray[np.float64],
    dt: float,
    f_b: NDArray[np.float64],
    w_b: NDArray[np.float64],
    R_nb: NDArray[np.float64],
):
    """
    Update the state transition matrix, phi, in place:

        phi[3:6, 6:9] = -dt * R_nb @ S(f_b)
        phi[3:9, 9:12] = -dt * R_nb
        phi[6:9, 6:9] = I - dt * S(w_b)

    Parameters
    ----------
    phi : ndarray, shape (15, 15)
        State transition matrix to be updated in place.
    dt : float
        Time step.
    f_b : ndarray, shape (3,)
        Specific force measurement (bias corrected) in body frame.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    R_nb : ndarray, shape (3, 3)
        Rotation matrix (from body to navigation frame).

    Notes
    -----
    Assuming the first order approximation:

        phi = I + dt * dfdx

    where dfdx denotes the linearized state matrix.
    """
    wx, wy, wz = w_b
    fx, fy, fz = f_b

    r00, r01, r02 = R_nb[0]
    r10, r11, r12 = R_nb[1]
    r20, r21, r22 = R_nb[2]

    # phi[6:9, 6:9] = np.eye(3) - dt * S(w_b)
    phi[6, 7] = dt * wz
    phi[6, 8] = -dt * wy
    phi[7, 6] = -dt * wz
    phi[7, 8] = dt * wx
    phi[8, 6] = dt * wy
    phi[8, 7] = -dt * wx

    # phi[3:9, 9:12] = -dt * R_nb
    phi[3, 9] = -dt * r00
    phi[3, 10] = -dt * r01
    phi[3, 11] = -dt * r02
    phi[4, 9] = -dt * r10
    phi[4, 10] = -dt * r11
    phi[4, 11] = -dt * r12
    phi[5, 9] = -dt * r20
    phi[5, 10] = -dt * r21
    phi[5, 11] = -dt * r22

    # phi[3:6, 6:9] = -dt * R_nb @ S(f_b)
    phi[3, 6] = -dt * (fz * r01 - fy * r02)
    phi[4, 6] = -dt * (fz * r11 - fy * r12)
    phi[5, 6] = -dt * (fz * r21 - fy * r22)
    phi[3, 7] = -dt * (-fz * r00 + fx * r02)
    phi[4, 7] = -dt * (-fz * r10 + fx * r12)
    phi[5, 7] = -dt * (-fz * r20 + fx * r22)
    phi[3, 8] = -dt * (fy * r00 - fx * r01)
    phi[4, 8] = -dt * (fy * r10 - fx * r11)
    phi[5, 8] = -dt * (fy * r20 - fx * r21)


def _process_noise_cov(
    dt: float, vrw: float, arw: float, abs: float, abc: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """
    Setup process noise covariance matrix, Q, using the first-order approximation:

        Q = dt @ dfdw @ W @ dfdw.T

    Parameters
    ----------
    dt : float
        Time step in seconds.
    vrw : float
        Velocity random walk (accelerometer noise density) in (m/s)/√Hz.
    arw : float
        Angular random walk (gyroscope noise density) in rad/√Hz.
    abs : float
        Accelerometer bias stability (bias instability) in m/s^2.
    abc : float
        Accelerometer bias correlation time in seconds.
    gbs : float
        Gyro bias stability (bias instability) in rad/s.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    Q : ndarray, shape (15, 15)
        Process noise covariance matrix.

    Notes
    -----
    In general, Q[VEL_IDX, VEL_IDX] should be updated each time step if R_nb changes:

        Q[VEL_IDX, VEL_IDX] = dt * (R_nb @ Wv @ R_nb.T)

    However, if the acceleration noise (velocity random walk) is isotropic (same
    in all axes), the rotation is not needed, and we can compute Q only once.
    """
    Q = np.zeros((15, 15))
    Q[VEL_IDX, VEL_IDX] = dt * vrw**2 * np.eye(3)
    Q[ATT_IDX, ATT_IDX] = dt * arw**2 * np.eye(3)
    Q[BA_IDX, BA_IDX] = dt * (2.0 * abs**2 / abc) * np.eye(3)
    Q[BG_IDX, BG_IDX] = dt * (2.0 * gbs**2 / gbc) * np.eye(3)
    return Q


def _state_matrix(
    f_b: NDArray[np.float64],
    w_b: NDArray[np.float64],
    R_nb: NDArray[np.float64],
    abc: float,
    gbc: float,
) -> NDArray[np.float64]:
    """
    Setup linearized state matrix, dfdx.

    Parameters
    ----------
    f_b : ndarray, shape (3,)
        Specific force measurement (bias corrected) in body frame.
    w_b : ndarray, shape (3,)
        Angular rate measurement (bias corrected) in body frame.
    R_nb : ndarray, shape (3, 3)
        Rotation matrix (from body to navigation frame).
    abc : float
        Accelerometer bias correlation time in seconds.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    dfdx : ndarray, shape (15, 15)
        Linearized state matrix.
    """
    dfdx = np.zeros((15, 15))
    dfdx[POS_IDX, VEL_IDX] = np.eye(3)
    dfdx[VEL_IDX, ATT_IDX] = -R_nb @ S(f_b)  # NB! update each time step
    dfdx[VEL_IDX, BA_IDX] = -R_nb  # NB! update each time step
    dfdx[ATT_IDX, ATT_IDX] = -S(w_b)  # NB! update each time step
    dfdx[ATT_IDX, BG_IDX] = -np.eye(3)
    dfdx[BA_IDX, BA_IDX] = -np.eye(3) / abc
    dfdx[BG_IDX, BG_IDX] = -np.eye(3) / gbc
    return dfdx


def _wn_input_matrix(R_nb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Setup linearized (white noise) input matrix, dfdw.

    Parameters
    ----------
    R_nb : ndarray, shape (3, 3)
        Rotation matrix (from body to navigation frame).

    Returns
    -------
    dfdw : ndarray, shape (15, 12)
        Linearized (white noise) input matrix.
    """
    dfdw = np.zeros((15, 12))
    dfdw[VEL_IDX, 0:3] = -R_nb  # NB! update each time step
    dfdw[ATT_IDX, 3:6] = -np.eye(3)
    dfdw[BA_IDX, 6:9] = np.eye(3)
    dfdw[BG_IDX, 9:12] = np.eye(3)
    return dfdw


def _process_noise_psd(
    vrw: float, arw: float, abs: float, abc: float, gbs: float, gbc: float
) -> NDArray[np.float64]:
    """
    Setup white noise (process noise) power spectral density matrix, W.

    Parameters
    ----------
    vrw : float
        Velocity random walk (accelerometer noise density) in (m/s)/√Hz.
    arw : float
        Angular random walk (gyroscope noise density) in rad/√Hz.
    abs : float
        Accelerometer bias stability (bias instability) in m/s^2.
    abc : float
        Accelerometer bias correlation time in seconds.
    gbs : float
        Gyro bias stability (bias instability) in rad/s.
    gbc : float
        Gyro bias correlation time in seconds.

    Returns
    -------
    W : ndarray, shape (12, 12)
        Process noise power spectral density matrix.
    """
    W = np.eye(12)
    W[0:3, 0:3] *= vrw**2
    W[3:6, 3:6] *= arw**2
    W[6:9, 6:9] *= 2.0 * abs**2 / abc
    W[9:12, 9:12] *= 2.0 * gbs**2 / gbc
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


def _measurement_matrix(
    q_nb: NDArray[np.float64], gref_n: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Setup linearized measurement matrix, dhdx.

    Parameters
    ----------
    q_nb : ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    dhdx : ndarray, shape (10, 15)
        Linearized measurement matrix.
    """
    R_nb = _matrix_from_quat(q_nb)

    dhdx = np.zeros((10, 15))
    dhdx[0:3, POS_IDX] = np.eye(3)  # position
    dhdx[3:6, VEL_IDX] = np.eye(3)  # velocity
    dhdx[6:7, ATT_IDX] = _dyawda(q_nb)  # heading (yaw angle) NB! update each time step
    dhdx[7:10, ATT_IDX] = S(R_nb.T @ gref_n)
    return dhdx
