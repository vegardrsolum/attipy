from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._quatops import _canonical, _normalize, _quatprod
from ._transforms import (
    _euler_zyx_from_quat,
    _matrix_from_quat,
    _quat_from_euler_zyx,
    _quat_from_matrix,
    _quat_from_rotvec,
    _rotvec_from_quat,
)


def _asarray_check_quaternion(q: ArrayLike) -> NDArray[np.float64]:
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


def _asarray_check_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
    """
    Convert the input to a numpy array and check if it is a valid rotation matrix
    (element of SO(3)).
    """
    R = np.asarray_chkfinite(matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("SO(3) matrix must be a 3x3 array.")
    I3x3 = np.eye(3)
    if not np.allclose(R.T @ R, I3x3):
        raise ValueError("SO(3) matrix must be orthogonal.")
    if not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("SO(3) matrix must have a determinant of 1.")
    return R

def _asarray_check_euler(theta: ArrayLike) -> NDArray[np.float64]:
    """
    Convert the input to a numpy array and check if it is a valid set of Euler angles.
    """
    theta = np.asarray_chkfinite(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError("Euler angles must be a 3-element array.")
    return theta


def _asarray_check_rotvec(theta: ArrayLike) -> NDArray[np.float64]:
    """
    Convert the input to a numpy array and check if it is a valid rotation vector.
    """
    theta = np.asarray_chkfinite(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError("Rotation vector must be a 3-element array.")
    return theta


class Attitude:
    """
    This class encapsulates the attitude (or rotation) of a 'body frame', {b}, relative
    to a 'navigation frame', {n}. Although the {n} and {b} frames can be defined
    arbitrarily, the main use case is for representing the attitude of a vehicle
    or sensor (body frame) relative to a local-level global reference frame (navigation frame)
    (e.g., North-East-Down, NED).

    Internally, the attitude is represented using a unit quaternion, q, defined
    such that it transforms a vector from the body frame, {b}, to the navigation frame, {n},
    using:

        [0, v_n] = q ⊗ [0, v_b] ⊗ q*

    where,

    - q is the unit quaternion.
    - q* is the conjugate of the unit quaternion q.
    - v_b is a vector expressed in the body frame.
    - v_n is the same vector expressed in the navigation frame.

    and ⊗ denotes quaternion multiplication (Hamilton product).

    Parameters
    ----------
    q : ArrayLike
        The 4-element unit quaternion, [q_w, q_x, q_y, q_z], where q_w is the scalar
        part, and q_x, q_y and q_z are the vector parts, respectively.
    """

    def __init__(self, q: ArrayLike) -> None:
        self._q = _asarray_check_quaternion(q)
        self._q = _canonical(self._q)

    def __repr__(self) -> str:
        q_w, q_x, q_y, q_z = self._q
        return f"Attitude(q=[{q_w:.3g} + {q_x:.3g}i + {q_y:.3g}j + {q_z:.3g}k])"

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
            The 4-element unit quaternion, [q_w, q_x, q_y, q_z], where q_w is the scalar
            part, and q_x, q_y and q_z are the vector parts, respectively.

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
        numpy.ndarray, shape (4,)
            The 4-element unit quaternion, [q_w, q_x, q_y, q_z], where q_w is the scalar
            part, and q_x, q_y and q_z are the vector parts, respectively.
        """
        return self._q.copy()

    @classmethod
    def from_matrix(cls, dcm: ArrayLike) -> Self:
        """
        Initialize from a direction cosine matrix (DCM), C, defined such that it
        transforms vectors from the body frame, {b}, to the navigation frame, {n},
        using:

            v_n = C @ v_b

        where,
        - ``C`` is the 3x3 direction cosine matrix (or rotation matrix).
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        Parameters
        ----------
        dcm : ArrayLike
            Direction cosine matrix, C. Should be element of SO(3).

        Returns
        -------
        Attitude
            Attitude instance.
        """
        C = _asarray_check_matrix(dcm)
        q = _quat_from_matrix(C)
        return cls(q)

    def as_matrix(self) -> NDArray[np.float64]:
        """
        Represent the attitude as a direction cosine matrix (DCM), C, defined such
        that it transforms vectors from the body frame, {b}, to the navigation frame,
        {n}, using:

            v_n = C @ v_b

        where,

        - ``C`` is the 3x3 direction cosine matrix (or rotation matrix).
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        The direction cosine matrix is computed from the unit quaternion, q, using
        the formula:

            C = I + 2 * q_w * S(q_xyz) + 2 * S(q_xyz)^2

        where,

        - I is the 3x3 identity matrix.
        - q_w is the scalar part of the unit quaternion.
        - q_xyz is the vector part, [q_x, q_y, q_z], of the unit quaternion.
        - S(q_xyz) is the skew-symmetric matrix of q_xyz.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
            Direction cosine matrix, C.
        """
        return _matrix_from_quat(self._q)

    @classmethod
    def from_euler(cls, theta: ArrayLike, degrees: bool = False) -> Self:
        """
        Initialize from a set of Euler angles (ZYX convention) (see Notes).

        Parameters
        ----------
        theta : ArrayLike
            Set of three Euler angles (ZYX convention), [alpha, beta, gamma], representing
            rotations about the X, Y, and Z axes, respectively.
        degrees : bool, default False
            If True, the input angles are interpreted as degrees. Otherwise, they are
            interpreted as radians. Internally, angles are stored as radians.

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

            C = C_z(gamma) @ C_y(beta) @ C_x(alpha)

        where,

        - gamma is a first rotation about the navigation frame's Z-axis.
        - beta is a second rotation about the intermediate Y-axis.
        - alpha is a final rotation about the second intermediate X-axis to arrive
          at the body frame.

        and C is the direction cosine matrix (transforming vectors from the body frame to
        the navigation frame):

            v_n = C @ v_b
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
        degrees : bool, default False
            If True, the output angles are in degrees. Otherwise, they are in radians.

        Returns
        -------
        numpy.ndarray, shape (3,)
            The 3-element Euler (ZYX) angles, [alpha, beta, gamma], representing
            rotations about the X, Y, and Z axes, respectively.

        Notes
        -----
        The ZYX Euler angles describe how to transition from the 'navigation' frame
        to the 'body' frame through three consecutive intrinsic and passive rotations
        in the ZYX order.

        Defined as:

            C = C_z(gamma) @ C_y(beta) @ C_x(alpha)

        where,

        - gamma is a first rotation about the navigation frame's Z-axis.
        - beta is a second rotation about the intermediate Y-axis.
        - alpha is a final rotation about the second intermediate X-axis to arrive
          at the body frame.

        and C is the direction cosine matrix (transforming vectors from the body frame to
        the navigation frame):

            v_n = C @ v_b
        """
        theta = _euler_zyx_from_quat(self._q)
        if degrees:
            theta = np.degrees(theta)
        return theta

    @classmethod
    def from_rotvec(cls, theta: ArrayLike, degrees: bool = False) -> Self:
        """
        Initialize from a rotation vector, theta, defined such that it is co-directional
        to the axis of rotation and has a norm equal to the angle of rotation [1]_.
        The rotation is assumed to be passive and from {n} to {b}.

        Parameters
        ----------
        theta : ArrayLike
            Rotation vector, [theta_x, theta_y, theta_z].
        degrees : bool, default False
            Specifies whether the input rotation vector is given in degrees or radians
            (default).

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        theta = _asarray_check_rotvec(theta)
        if degrees:
            theta = np.radians(theta)
        q = _quat_from_rotvec(theta)
        return cls(q)

    def as_rotvec(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Represent the attitude as a 3-element rotation vector, defined such that
        it is co-directional to the axis of rotation and has a norm equal to the
        angle of rotation [1]_. The rotation is assumed to be passive and from {n}
        to {b}.

        Parameters
        ----------
        degrees : bool, default False
            Specifies whether the output rotation vector should be given in degrees
            or radians (default).

        Returns
        -------
        numpy.ndarray, shape (3,)
            Rotation vector, [theta_x, theta_y, theta_z].

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """

        theta = _rotvec_from_quat(self._q)
        if degrees:
            theta = np.degrees(theta)
        return theta

    def update(self, dtheta, degrees=False):
        """
        Update the attitude with an incremental rotation given by a rotation vector.

        The attitude is updated according to:

            q[k+1] = q[k] ⊗ h(dtheta[k])

        where,

        - q[k] is the current (time step k) attitude (as unit quaternion).
        - q[k+1] is the updated (time step k+1) attitude (as unit quaternion).
        - dtheta[k] is the attitude increment from time step k to k+1, expressed
          as a rotation vector.
        - h(dtheta[k]) is the unit quaternion corresponding to the attitude increment.

        and ⊗ denotes quaternion multiplication (Hamilton product).

        Parameters
        ----------
        dtheta : ArrayLike
            Rotation vector representing the incremental rotation to be applied.
            The direction of the vector indicates the axis of rotation, and the
            magnitude (norm) of the vector indicates the angle of rotation.
        degrees : bool, default False
            Specifies whether the input rotation vector is given in degrees or radians
            (default).
        """
        dtheta = _asarray_check_rotvec(dtheta)

        if degrees:
            dtheta = np.radians(dtheta)

        dq = _quat_from_rotvec(dtheta)
        self._q = _normalize(_quatprod(self._q, dq))
