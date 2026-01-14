import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._quatops import _canonical
from ._vectorops import _normalize


@njit  # type: ignore[misc]
def _quat_from_matrix(dcm: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a unit quaternion (see ref [1]_).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    """

    r00, r01, r02 = dcm[0]
    r10, r11, r12 = dcm[1]
    r20, r21, r22 = dcm[2]

    trace = r00 + r11 + r22

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif (r00 > r11) and (r00 > r22):
        s = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz])
    return _normalize(q)


@njit  # type: ignore[misc]
def _matrix_from_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
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

    r00 = 1.0 - (_2q2q2 + _2q3q3)
    r01 = _2q1q2 - _2q0q3
    r02 = _2q1q3 + _2q0q2

    r10 = _2q1q2 + _2q0q3
    r11 = 1.0 - (_2q1q1 + _2q3q3)
    r12 = _2q2q3 - _2q0q1

    r20 = _2q1q3 - _2q0q2
    r21 = _2q2q3 + _2q0q1
    r22 = 1.0 - (_2q1q1 + _2q2q2)

    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return R


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
            - Roll (roll): Rotation about the x-axis.
            - Pitch (pitch): Rotation about the y-axis.
            - Yaw (yaw): Rotation about the z-axis.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    qw, qx, qy, qz = q

    roll = np.arctan2(2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx**2 + qy**2))
    pitch = -np.arcsin(2.0 * (qx * qz - qy * qw))
    yaw = np.arctan2(2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qy**2 + qz**2))

    return np.array([roll, pitch, yaw])


@njit  # type: ignore[misc]
def _matrix_from_euler_zyx(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the rotation matrix (from-body-to-origin) from Euler angles.

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (roll): Rotation about the x-axis.
            - Pitch (pitch): Rotation about the y-axis.
            - Yaw (yaw): Rotation about the z-axis.

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
    roll, pitch, yaw = euler
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)

    r00 = cos_yaw * cos_pitch
    r01 = -sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll
    r02 = sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll

    r10 = sin_yaw * cos_pitch
    r11 = cos_yaw * cos_roll + sin_yaw * sin_pitch * sin_roll
    r12 = -cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll

    r20 = -sin_pitch
    r21 = cos_pitch * sin_roll
    r22 = cos_pitch * cos_roll

    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return R


@njit  # type: ignore[misc]
def _quat_from_euler_zyx(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion from Euler angles (see ref [1]_).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """

    roll_half, pitch_half, yaw_half = euler / 2.0
    ca_half = np.cos(roll_half)
    sa_half = np.sin(roll_half)
    cb_half = np.cos(pitch_half)
    sb_half = np.sin(pitch_half)
    cg_half = np.cos(yaw_half)
    sg_half = np.sin(yaw_half)

    qw = ca_half * cb_half * cg_half + sa_half * sb_half * sg_half
    qx = sa_half * cb_half * cg_half - ca_half * sb_half * sg_half
    qy = ca_half * sb_half * cg_half + sa_half * cb_half * sg_half
    qz = ca_half * cb_half * sg_half - sa_half * sb_half * cg_half

    return np.array([qw, qx, qy, qz])


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
    qw, qx, qy, qz = q

    qxyz_norm = np.sqrt(qx**2 + qy**2 + qz**2)
    angle = 2.0 * np.arctan2(qxyz_norm, qw)

    if angle <= 1e-3:  # 4th order approximation (avoids division by zero)
        angle2 = angle**2
        scale = 2.0 + (angle2) / 12.0 + 7.0 * angle2**2 / 2880.0
    else:
        scale = angle / np.sin(angle / 2.0)

    return np.array([scale * qx, scale * qy, scale * qz])


@njit  # type: ignore[misc]
def _matrix_from_euler(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the rotation matrix (from-body-to-origin) from Euler angles.


    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (roll): Rotation about the x-axis.
            - Pitch (pitch): Rotation about the y-axis.
            - Yaw (yaw): Rotation about the z-axis.

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
    roll, pitch, yaw = euler
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)

    rot_00 = cos_yaw * cos_pitch
    rot_01 = -sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll
    rot_02 = sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll

    rot_10 = sin_yaw * cos_pitch
    rot_11 = cos_yaw * cos_roll + sin_yaw * sin_pitch * sin_roll
    rot_12 = -cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll

    rot_20 = -sin_pitch
    rot_21 = cos_pitch * sin_roll
    rot_22 = cos_pitch * cos_roll

    rot = np.array(
        [[rot_00, rot_01, rot_02], [rot_10, rot_11, rot_12], [rot_20, rot_21, rot_22]]
    )
    return rot


@njit  # type: ignore[misc]
def _yaw_from_quat(q_nb: NDArray[np.float64]) -> float:
    """
    Compute yaw angle from unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    float
        Yaw angle in radians.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.251, John Wiley & Sons, 2021.
    """
    qw, qx, qy, qz = q_nb
    u_y = 2.0 * (qx * qy + qz * qw)
    u_x = 1.0 - 2.0 * (qy**2 + qz**2)
    return np.arctan2(u_y, u_x)  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _quat_from_gibbs2(g2):
    """
    Compute unit quaternion, q, from 2 x Gibbs vector (scaled Gibbs vector), g2:

        qw = 2 / sqrt(4 + g2.T @ g2)
        qv = g2 / sqrt(4 + g2.T @ g2)

    where,

        q = (qw, qv[0], qv[1], qv[2])

    where,
    """
    gx, gy, gz = g2

    scale = 1.0 / np.sqrt(4.0 + gx**2 + gy**2 + gz**2)

    q = scale * np.array([2.0, gx, gy, gz])
    return q
