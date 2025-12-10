import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def _quaternion_from_matrix(A: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a unit quaternion.
    """
    # TODO: remove scipy dependency
    return Rotation.from_matrix(A).as_quat(scalar_first=True)


@njit  # type: ignore[misc]
def _rot_matrix_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the rotation matrix from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    rot : numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    q0, q1, q2, q3 = q

    _2q1 = q1 + q1
    _2q2 = q2 + q2
    _2q3 = q3 + q3

    _2q1q1 = q1 * _2q1
    _2q1q2 = q1 * _2q2
    _2q1q3 = q1 * _2q3
    _2q2q2 = q2 * _2q2
    _2q2q3 = q2 * _2q3
    _2q3q3 = q3 * _2q3
    _2q0q1 = q0 * _2q1
    _2q0q2 = q0 * _2q2
    _2q0q3 = q0 * _2q3

    rot_00 = 1.0 - (_2q2q2 + _2q3q3)
    rot_01 = _2q1q2 - _2q0q3
    rot_02 = _2q1q3 + _2q0q2

    rot_10 = _2q1q2 + _2q0q3
    rot_11 = 1.0 - (_2q1q1 + _2q3q3)
    rot_12 = _2q2q3 - _2q0q1

    rot_20 = _2q1q3 - _2q0q2
    rot_21 = _2q2q3 + _2q0q1
    rot_22 = 1.0 - (_2q1q1 + _2q2q2)

    rot = np.array(
        [
            [rot_00, rot_01, rot_02],
            [rot_10, rot_11, rot_12],
            [rot_20, rot_21, rot_22],
        ]
    )
    return rot


@njit  # type: ignore[misc]
def _euler_zyx_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the Euler angles (ZYX convention) from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion (representing transformation from-body-to-origin).

    Returns
    -------
    numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.
    """
    q_w, q_x, q_y, q_z = q

    alpha = np.arctan2(2.0 * (q_y * q_z + q_x * q_w), 1.0 - 2.0 * (q_x**2 + q_y**2))
    beta = -np.arcsin(2.0 * (q_x * q_z - q_y * q_w))
    gamma = np.arctan2(2.0 * (q_x * q_y + q_z * q_w), 1.0 - 2.0 * (q_y**2 + q_z**2))

    return np.array([alpha, beta, gamma])


@njit  # type: ignore[misc]
def _rot_matrix_from_euler_zyx(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the rotation matrix (from-body-to-origin) from Euler angles.

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (alpha): Rotation about the x-axis.
            - Pitch (beta): Rotation about the y-axis.
            - Yaw (gamma): Rotation about the z-axis.

    Notes
    -----
    The Euler angles describe how to transition from the 'origin' frame to the 'body'
    frame through three consecutive (passive, intrinsic) rotations in the ZYX order.
    However, the returned rotation matrix represents the transformation of a vector
    from the 'body' frame to the 'origin' frame.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    alpha, beta, gamma = euler
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    rot_00 = cos_gamma * cos_beta
    rot_01 = -sin_gamma * cos_alpha + cos_gamma * sin_beta * sin_alpha
    rot_02 = sin_gamma * sin_alpha + cos_gamma * sin_beta * cos_alpha

    rot_10 = sin_gamma * cos_beta
    rot_11 = cos_gamma * cos_alpha + sin_gamma * sin_beta * sin_alpha
    rot_12 = -cos_gamma * sin_alpha + sin_gamma * sin_beta * cos_alpha

    rot_20 = -sin_beta
    rot_21 = cos_beta * sin_alpha
    rot_22 = cos_beta * cos_alpha

    rot = np.array(
        [[rot_00, rot_01, rot_02], [rot_10, rot_11, rot_12], [rot_20, rot_21, rot_22]]
    )
    return rot


@njit  # type: ignore[misc]
def _quaternion_from_euler_zyx(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion from Euler angles.
    """

    # TODO: Verify thath the equation are correct

    alpha_half, beta_half, gamma_half = euler / 2.0
    cos_alpha = np.cos(alpha_half)
    sin_alpha = np.sin(alpha_half)
    cos_beta = np.cos(beta_half)
    sin_beta = np.sin(beta_half)
    cos_gamma = np.cos(gamma_half)
    sin_gamma = np.sin(gamma_half)

    q_w = cos_alpha * cos_beta * cos_gamma + sin_alpha * sin_beta * sin_gamma
    q_x = sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_beta * sin_gamma
    q_y = cos_alpha * sin_beta * cos_gamma + sin_alpha * cos_beta * sin_gamma
    q_z = cos_alpha * cos_beta * sin_gamma - sin_alpha * sin_beta * cos_gamma

    return np.array([q_w, q_x, q_y, q_z])
