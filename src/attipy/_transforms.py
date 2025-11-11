import numpy as np
from numba import njit
from numpy.typing import NDArray


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
def _euler_from_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
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
