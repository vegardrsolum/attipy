import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._quatops import _canonical, _normalize


@njit  # type: ignore[misc]
def _quat_from_matrix(dcm: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion from a rotation matrix (see ref [1]_).

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
    Compute the direction cosine matrix (rotation matrix) from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    rot : numpy.ndarray, shape (3, 3)
        Rotation matrix.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 2.72, John Wiley & Sons, 2021.
    """

    qw, qx, qy, qz = q

    _2qx = qx + qx
    _2qy = qy + qy
    _2qz = qz + qz

    _2qxqx = qx * _2qx
    _2qxqy = qx * _2qy
    _2qxqz = qx * _2qz
    _2qyqy = qy * _2qy
    _2qyqz = qy * _2qz
    _2qzqz = qz * _2qz
    _2qwqx = qw * _2qx
    _2qwqy = qw * _2qy
    _2qwqz = qw * _2qz

    r00 = 1.0 - (_2qyqy + _2qzqz)
    r01 = _2qxqy - _2qwqz
    r02 = _2qxqz + _2qwqy

    r10 = _2qxqy + _2qwqz
    r11 = 1.0 - (_2qxqx + _2qzqz)
    r12 = _2qyqz - _2qwqx

    r20 = _2qxqz - _2qwqy
    r21 = _2qyqz + _2qwqx
    r22 = 1.0 - (_2qxqx + _2qyqy)

    dcm = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return dcm


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
    Compute the direction cosine matrix (rotation matrix) from Euler angles.

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (roll): Rotation about the x-axis.
            - Pitch (pitch): Rotation about the y-axis.
            - Yaw (yaw): Rotation about the z-axis.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    # TODO: add reference

    roll, pitch, yaw = euler

    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    r00 = cy * cp
    r01 = -sy * cr + cy * sp * sr
    r02 = sy * sr + cy * sp * cr

    r10 = sy * cp
    r11 = cy * cr + sy * sp * sr
    r12 = -cy * sr + sy * sp * cr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    dcm = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return dcm


@njit  # type: ignore[misc]
def _quat_from_euler_zyx(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion from Euler angles (see ref [1]_).

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (roll): Rotation about the x-axis.
            - Pitch (pitch): Rotation about the y-axis.
            - Yaw (yaw): Rotation about the z-axis.

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternion.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """

    roll_half, pitch_half, yaw_half = euler / 2.0

    cr_half = np.cos(roll_half)
    sr_half = np.sin(roll_half)
    cp_half = np.cos(pitch_half)
    sp_half = np.sin(pitch_half)
    cy_half = np.cos(yaw_half)
    sy_half = np.sin(yaw_half)

    qw = cr_half * cp_half * cy_half + sr_half * sp_half * sy_half
    qx = sr_half * cp_half * cy_half - cr_half * sp_half * sy_half
    qy = cr_half * sp_half * cy_half + sr_half * cp_half * sy_half
    qz = cr_half * cp_half * sy_half - sr_half * sp_half * cy_half

    return np.array([qw, qx, qy, qz])


@njit  # type: ignore[misc]
def _quat_from_rotvec(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the unit quaternion from a rotation vector.

    Parameters
    ----------
    theta : numpy.ndarray, shape (3,)
        Rotation vector (thetax, thetay, thetaz).

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz).
    """
    # TODO: add reference

    rx, ry, rz = theta

    angle2 = rx**2 + ry**2 + rz**2

    if angle2 < 1e-6:  # 2nd order approximation (avoids division by zero)
        a = 0.25 * angle2
        c = 1.0 - a / 2.0
        s = 0.5 * (1.0 - a / 6.0)
    else:
        angle = np.sqrt(angle2)
        half_angle = 0.5 * angle
        c = np.cos(half_angle)
        s = np.sin(half_angle) / angle

    q = np.array([c, s * rx, s * ry, s * rz])

    return _normalize(q)


@njit  # type: ignore[misc]
def _rotvec_from_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the rotation vector from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz).

    Returns
    -------
    numpy.ndarray, shape (3,)
        Rotation vector (thetax, thetay, thetaz).
    """
    # TODO: add reference

    qw, qx, qy, qz = _canonical(q)

    qxyz_norm = np.sqrt(qx**2 + qy**2 + qz**2)
    angle = 2.0 * np.arctan2(qxyz_norm, qw)

    if angle <= 1e-3:  # 4th order approximation (avoids division by zero)
        angle2 = angle**2
        scale = 2.0 + (angle2) / 12.0 + 7.0 * angle2**2 / 2880.0
    else:
        scale = angle / np.sin(angle / 2.0)

    rx = scale * qx
    ry = scale * qy
    rz = scale * qz

    return np.array([rx, ry, rz])


@njit  # type: ignore[misc]
def _matrix_from_euler(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the direction cosine matrix (rotation matrix) from Euler angles.

    Parameters
    ----------
    euler : numpy.ndarray, shape (3,)
        Vector of Euler angles in radians (ZYX convention). Contains the following
        three Euler angles in order:
            - Roll (roll): Rotation about the x-axis.
            - Pitch (pitch): Rotation about the y-axis.
            - Yaw (yaw): Rotation about the z-axis.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Direction cosine matrix (rotation matrix).
    """
    # TODO: add reference

    roll, pitch, yaw = euler

    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    r00 = cy * cp
    r01 = -sy * cr + cy * sp * sr
    r02 = sy * sr + cy * sp * cr

    r10 = sy * cp
    r11 = cy * cr + sy * sp * sr
    r12 = -cy * sr + sy * sp * cr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    dcm = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return dcm


@njit  # type: ignore[misc]
def _yaw_from_quat(q_nb: NDArray[np.float64]) -> float:
    """
    Compute yaw angle from a unit quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz).

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
    uy = 2.0 * (qx * qy + qz * qw)
    ux = 1.0 - 2.0 * (qy**2 + qz**2)
    return np.arctan2(uy, ux)  # type: ignore[no-any-return]


@njit  # type: ignore[misc]
def _quat_from_gibbs2(g2: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute unit quaternion from 2 x Gibbs vector (scaled Gibbs vector).

    The quaternion is computed as:

        q = 1 / sqrt(4 + ||a||^2) * [2, dax, day, daz]

    where a = [ax, ay, az] is the scaled (2x) Gibbs vector.

    Parameters
    ----------
    g2 : numpy.ndarray, shape (3,)
        2 x Gibbs vector (g2x, g2y, g2z).

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz).

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 14.229, John Wiley & Sons, 2021.
    """
    gx, gy, gz = g2

    scale = 1.0 / np.sqrt(4.0 + gx**2 + gy**2 + gz**2)

    q = scale * np.array([2.0, gx, gy, gz])
    return q
