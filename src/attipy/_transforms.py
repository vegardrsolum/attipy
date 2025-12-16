import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._quatops import _canonical, _normalize


@njit  # type: ignore[misc]
def _quat_from_matrix(A: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a unit quaternion (see ref [1]_).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    """

    m00, m01, m02 = A[0]
    m10, m11, m12 = A[1]
    m20, m21, m22 = A[2]

    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return _normalize(q)


@njit  # type: ignore[misc]
def _rot_matrix_from_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
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
def _euler_zyx_from_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the Euler angles (ZYX convention) from a unit quaternion (see ref [1]_).

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

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
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
def _quat_from_euler_zyx(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion from Euler angles (see ref [1]_).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """

    alpha_half, beta_half, gamma_half = euler / 2.0
    ca_half = np.cos(alpha_half)
    sa_half = np.sin(alpha_half)
    cb_half = np.cos(beta_half)
    sb_half = np.sin(beta_half)
    cg_half = np.cos(gamma_half)
    sg_half = np.sin(gamma_half)

    q_w = ca_half * cb_half * cg_half + sa_half * sb_half * sg_half
    q_x = sa_half * cb_half * cg_half - ca_half * sb_half * sg_half
    q_y = ca_half * sb_half * cg_half + sa_half * cb_half * sg_half
    q_z = ca_half * cb_half * sg_half - sa_half * sb_half * cg_half

    return np.array([q_w, q_x, q_y, q_z])


@njit  # type: ignore[misc]
def _quat_from_rotvec(theta: NDArray[np.float64]) -> NDArray[np.float64]:

    theta_x, theta_y, theta_z = theta
    angle2 = theta_x**2 + theta_y**2 + theta_z**2

    if angle2 < 1e-6:  # 2nd order approximation (avoids division by zero)
        a = 0.25 * angle2
        c = 1.0 - a / 2.0
        s = 0.5 * (1.0 - a / 6.0)
    else:
        angle = np.sqrt(angle2)
        half_angle = 0.5 * angle
        c = np.cos(half_angle)
        s = np.sin(half_angle) / angle

    q = np.array([c, s * theta_x, s * theta_y, s * theta_z])

    return _normalize(q)


@njit  # type: ignore[misc]
def _rotvec_from_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the rotation vector from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Rotation vector.
    """
    q = _canonical(q)
    q_w, q_x, q_y, q_z = q

    q_xyz_norm = np.sqrt(q_x**2 + q_y**2 + q_z**2)
    angle = 2.0 * np.arctan2(q_xyz_norm, q_w)

    if angle <= 1e-3:  # 4th order approximation (avoids division by zero)
        angle2 = angle**2
        scale = 2.0 + (angle2) / 12.0 + 7.0 * angle2**2 / 2880.0
    else:
        scale = angle / np.sin(angle / 2.0)

    return np.array([scale * q_x, scale * q_y, scale * q_z])
