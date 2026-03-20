from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._quatops import (
    _canonical,
    _correct_quat_with_gibbs2,
    _correct_quat_with_rotvec,
)
from ._transforms import (
    _euler_zyx_from_quat,
    _matrix_from_quat,
    _quat_from_euler_zyx,
    _quat_from_matrix,
    _quat_from_rotvec,
    _rotvec_from_quat,
)


def _asarray_check_quat(q: ArrayLike) -> NDArray[np.float64]:
    """
    Convert the input to a numpy array and check if it is a valid unit quaternion.
    """
    q = np.asarray_chkfinite(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Unit quaternion must be a 4-element array.")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0):
        raise ValueError("Unit quaternion must have a norm of 1.")
    return q


def _asarray_check_matrix(dcm: ArrayLike) -> NDArray[np.float64]:
    """
    Convert the input to a numpy array and check if it is a valid rotation matrix
    (element of SO(3)).
    """
    R = np.asarray_chkfinite(dcm, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("SO(3) matrix must be a 3x3 array.")
    I3x3 = np.eye(3)
    if not np.allclose(R.T @ R, I3x3):
        raise ValueError("SO(3) matrix must be orthogonal.")
    if not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("SO(3) matrix must have a determinant of 1.")
    return R


def _asarray_check_euler(euler: ArrayLike) -> NDArray[np.float64]:
    """
    Convert the input to a numpy array and check if it is a valid set of Euler angles.
    """
    euler = np.asarray_chkfinite(euler, dtype=float)
    if euler.shape != (3,):
        raise ValueError("Euler angles must be a 3-element array.")
    return euler


def _asarray_check_rotvec(r: ArrayLike) -> NDArray[np.float64]:
    """
    Convert the input to a numpy array and check if it is a valid rotation vector.
    """
    r = np.asarray_chkfinite(r, dtype=float)
    if r.shape != (3,):
        raise ValueError("Rotation vector must be a 3-element array.")
    return r


class Attitude:
    """
    This class encapsulates the attitude (rotation) of one orthonormal reference
    frame {b} relative to another {n}.

    Although the {n} and {b} frames can be defined arbitrarily, the main use case
    is for representing the attitude of a vehicle or sensor (body frame) relative
    to a local-level global inertial reference frame (navigation frame). The most
    common navigation frames are the North-East-Down (NED) frame and the East-North-Up
    (ENU) frame.

    Internally, the attitude is represented using a unit quaternion, q, defined
    such that it transforms a vector from the body frame, {b}, to the navigation
    frame, {n}, using:

        [0, v_n] = q ⊗ [0, v_b] ⊗ q*

    where,

    - q is the unit quaternion.
    - q* is the conjugate of the unit quaternion q.
    - v_b is a vector expressed in the body frame.
    - v_n is the same vector expressed in the navigation frame.

    and ⊗ denotes quaternion multiplication (Hamilton product).

    The class provides methods for transforming to/from a variety of attitude representations,
    including:

    - Direction cosine matrix (rotation matrix) (9 parameters).
    - Unit quaternion (4 parameters).
    - Euler angles (ZYX convention) (3 parameters).
    - Rotation vector (3 parameters).

    Parameters
    ----------
    q : ArrayLike
        The 4-element unit quaternion (qw, qx, qy, qz), where qw is the scalar
        part and (qx, qy, qz) is the vector part.
    """

    def __init__(self, q: ArrayLike) -> None:
        self._q = _canonical(_asarray_check_quat(q))

    def __repr__(self) -> str:
        qw, qx, qy, qz = self._q
        return f"Attitude(q=[{qw:.3g} + {qx:.3g}i + {qy:.3g}j + {qz:.3g}k])"

    @classmethod
    def from_quaternion(cls, q: ArrayLike) -> Self:
        """
        Initialize from a unit quaternion, q, defined such that it transforms a
        vector from the body frame, {b}, to the navigation frame, {n}, using:

            [0, v_n] = q ⊗ [0, v_b] ⊗ q*

        where,

        - q is the unit quaternion.
        - q* is the conjugate of the unit quaternion q.
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        and ⊗ denotes quaternion multiplication (Hamilton product).

        Parameters
        ----------
        q : ArrayLike
            The 4-element unit quaternion (qw, qx, qy, qz), where qw is the scalar
            part and (qx, qy, qz) is the vector part.

        Returns
        -------
        Attitude
            Attitude instance.
        """
        return cls(q)

    def as_quaternion(self) -> NDArray[np.float64]:
        """
        Represent the attitude as a unit quaternion, q, defined such that it transforms
        a vector from the body frame, {b}, to the navigation frame, {n}, using:

            [0, v_n] = q ⊗ [0, v_b] ⊗ q*

        where,

        - q is the unit quaternion.
        - q* is the conjugate of the unit quaternion q.
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        and ⊗ denotes quaternion multiplication (Hamilton product).

        Returns
        -------
        ndarray, shape (4,)
            The 4-element unit quaternion (qw, qx, qy, qz), where qw is the scalar
            part and (qx, qy, qz) is the vector part.
        """
        return self._q.copy()

    @classmethod
    def from_matrix(cls, dcm: ArrayLike) -> Self:
        """
        Initialize from a direction cosine matrix (DCM), R, defined such that it
        transforms vectors from the body frame, {b}, to the navigation frame, {n},
        using:

            v_n = R @ v_b

        where,
        - R is the 3x3 direction cosine matrix (rotation matrix).
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        Parameters
        ----------
        dcm : ArrayLike
            Direction cosine matrix (rotation matrix), R. Element of SO(3).

        Returns
        -------
        Attitude
            Attitude instance.
        """
        R = _asarray_check_matrix(dcm)
        q = _quat_from_matrix(R)
        return cls(q)

    def as_matrix(self) -> NDArray[np.float64]:
        """
        Represent the attitude as a direction cosine matrix (DCM), R, defined such
        that it transforms vectors from the body frame, {b}, to the navigation frame,
        {n}, using:

            v_n = R @ v_b

        where,

        - R is the 3x3 direction cosine matrix (rotation matrix).
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        Returns
        -------
        ndarray, shape (3, 3)
            Direction cosine matrix (rotation matrix), R.

        Notes
        -----
        The direction cosine matrix, R, is computed from the unit quaternion, q,
        using the formula:

            R = I + 2 * qw * S(qxyz) + 2 * S(qxyz)^2

        where,

        - I is the 3x3 identity matrix.
        - qw is the scalar part of the unit quaternion, q.
        - qxyz is the vector part, (qx, qy, qz), of the unit quaternion, q.
        - S(qxyz) is the skew-symmetric matrix of qxyz.
        """
        return _matrix_from_quat(self._q)

    @classmethod
    def from_euler(cls, theta: ArrayLike, degrees: bool = False) -> Self:
        """
        Initialize from a set of Euler angles (ZYX convention) (see Notes).

        Parameters
        ----------
        theta : ArrayLike
            Set of three Euler angles (ZYX convention) (roll, pitch, yaw), representing
            rotations about the X, Y, and Z axes, respectively.
        degrees : bool, optional
            Specifies whether the Euler angles are given in degrees or radians (default).

        Returns
        -------
        Attitude
            Attitude instance.

        Notes
        -----
        The ZYX Euler angles describe how to transition from the 'navigation' frame
        to the 'body' frame through three consecutive intrinsic and passive rotations
        in the ZYX order.

        Defined as:

            R = R_z(yaw) @ R_y(pitch) @ R_x(roll)

        where,

        - yaw is a first rotation about the navigation frame's Z-axis.
        - pitch is a second rotation about the intermediate Y-axis.
        - roll is a final rotation about the second intermediate X-axis to arrive
          at the body frame.

        and R is the direction cosine matrix (transforming vectors from the body frame to
        the navigation frame):

            v_n = R @ v_b
        """
        theta = _asarray_check_euler(theta)
        if degrees:
            theta = np.radians(theta)
        q = _quat_from_euler_zyx(theta)
        return cls(q)

    def as_euler(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Represent the attitude as a set of Euler angles (ZYX convention) (see Notes).

        Parameters
        ----------
        degrees : bool, optional
            Specifies whether the output Euler angles should be given in degrees
            or radians (default).

        Returns
        -------
        ndarray, shape (3,)
            The 3-element Euler (ZYX) angles (roll, pitch, yaw), representing
            rotations about the X, Y, and Z axes, respectively.

        Notes
        -----
        The ZYX Euler angles describe how to transition from the 'navigation' frame
        to the 'body' frame through three consecutive intrinsic and passive rotations
        in the ZYX order.

        Defined as:

            R = R_z(yaw) @ R_y(pitch) @ R_x(roll)

        where,

        - yaw is a first rotation about the navigation frame's Z-axis.
        - pitch is a second rotation about the intermediate Y-axis.
        - roll is a final rotation about the second intermediate X-axis to arrive
          at the body frame.

        and R is the direction cosine matrix (transforming vectors from the body frame to
        the navigation frame):

            v_n = R @ v_b
        """
        theta = _euler_zyx_from_quat(self._q)
        if degrees:
            theta = np.degrees(theta)
        return theta

    @classmethod
    def from_rotvec(cls, r: ArrayLike, degrees: bool = False) -> Self:
        """
        Initialize from a rotation vector, r, defined such that it is co-directional
        to the axis of rotation and has a norm equal to the angle of rotation [1]_.
        The rotation is assumed to be passive and from {n} to {b}.

        Parameters
        ----------
        r : ArrayLike
            Rotation vector (rx, ry, rz).
        degrees : bool, optional
            Specifies whether the input rotation vector, r, is given in degrees
            or radians (default).

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        r = _asarray_check_rotvec(r)
        if degrees:
            r = np.radians(r)
        q = _quat_from_rotvec(r)
        return cls(q)

    def as_rotvec(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Represent the attitude as a 3-element rotation vector, defined such that
        it is co-directional to the axis of rotation and has a norm equal to the
        angle of rotation [1]_. The rotation is assumed to be passive and from {n}
        to {b}.

        Parameters
        ----------
        degrees : bool, optional
            Specifies whether the output rotation vector should be given in degrees
            or radians (default).

        Returns
        -------
        ndarray, shape (3,)
            Rotation vector (rx, ry, rz).

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """

        r = _rotvec_from_quat(self._q)
        if degrees:
            r = np.degrees(r)
        return r
