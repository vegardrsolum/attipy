import numpy as np
from numpy.typing import ArrayLike

from ._transforms import (
    _euler_zyx_from_quaternion,
    _rot_matrix_from_quaternion,
    _quaternion_from_euler_zyx,
    _quaternion_from_matrix,
)


def _asarray_check_unit_quaternion(q: ArrayLike) -> np.ndarray:
    """
    Convert the input to a numpy array and check if it is a unit quaternion.
    """
    q = np.asarray_chkfinite(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Unit quaternion must be a 4-element array.")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0):
        raise ValueError("Unit quaternion must have a norm of 1.")
    return q


def _asarray_check_matrix_so3(A: ArrayLike) -> np.ndarray:
    """
    Convert the input to a numpy array and check if it is a valid rotation matrix
    (element of SO(3)).
    """
    A = np.asarray_chkfinite(A, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("SO(3) matrix must be a 3x3 array.")
    I3x3 = np.eye(3)
    if not np.allclose(A.T @ A, I3x3):
        raise ValueError("SO(3) matrix must be orthogonal.")
    if not np.isclose(np.linalg.det(A), 1.0):
        raise ValueError("SO(3) matrix must have a determinant of 1.")
    return A


class Attitude:
    """
    This class encapsulates the attitude (or rotation) of a 'body frame', {b}, relative
    to a 'navigation frame', {n}. Although the {n} and {b} frames can be defined
    arbitrarily, the main use case is for representing the attitude of a vehicle
    or sensor (body frame) relative to a local-level inertial or global reference
    frame (navigation frame) (e.g., North-East-Down, NED).

    Internally, the attitude is represented using a unit quaternion, q, defined
    such that it transforms a vector from the body frame to the navigation frame
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

    def __init__(self, q: ArrayLike):
        self._q = _asarray_check_unit_quaternion(q)

    def __repr__(self):
        array_str = np.array2string(self._q)
        return f"Attitude(q={array_str})"

    @classmethod
    def from_quaternion(cls, q: ArrayLike):
        """
        Create an Attitude instance from a unit quaternion, q, defined such that
        it transforms a vector from the body frame, {b}, to the navigation frame,
        {n}, using:

            [0, v_n] = q ⊗ [0, v_b] ⊗ q*

        where,

        - q is the unit quaternion.
        - q* is the conjugate of the unit quaternion q.
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        and ⊗ denotes quaternion multiplication (Hamilton product).
        """
        return cls(q)

    def as_quaternion(self) -> np.ndarray:
        """
        Return the attitude as a unit quaternion, q, defined such that it transforms
        a vector from the body frame, {b}, to the navigation frame, {n}, using:

            [0, v_n] = q ⊗ [0, v_b] ⊗ q*

        where,

        - q is the unit quaternion.
        - q* is the conjugate of the unit quaternion q.
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        and ⊗ denotes quaternion multiplication (Hamilton product).
        """
        return self._q.copy()

    @classmethod
    def from_matrix(cls, A: ArrayLike):
        """
        Create an Attitude instance from a rotation matrix (or direction cosine
        matrix), defined such that it transforms vectors from the body frame, {b},
        to the navigation frame, {n}, using:

            v_n = A @ v_b

        where,
        - ``A`` is the 3x3 rotation matrix (or direction cosine matrix).
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        Parameters
        ----------
        A : ArrayLike
            Rotation matrix (element of SO(3)).
        """
        A = _asarray_check_matrix_so3(A)
        q = _quaternion_from_matrix(A)
        return cls(q)

    def as_matrix(self) -> np.ndarray:
        """
        Return the attitude as a rotation matrix (or direction cosine matrix), A,
        defined such that it transforms vectors from the body frame, {b}, to the
        navigation frame, {n}, using:

            v_n = A @ v_b

        where,

        - ``A`` is the 3x3 attitude matrix.
        - v_b is a vector expressed in the body frame, {b}.
        - v_n is the same vector expressed in the navigation frame, {n}.

        The rotation matrix is computed from the unit quaternion, q, using the formula:

            A = I + 2 * q_w * S(q_xyz) + 2 * S(q_xyz)^2

        where,

        - I is the 3x3 identity matrix.
        - q_w is the scalar part of the unit quaternion.
        - q_xyz is the vector part, [q_x, q_y, q_z], of the unit quaternion.
        - S(q_xyz) is the skew-symmetric matrix of q_xyz.
        """
        return _rot_matrix_from_quaternion(self._q)

    @classmethod
    def from_euler(cls, theta: ArrayLike, degrees=False) -> "AttitudeMatrix":
        """
        Create an Attitude instance from a set of Euler angles (ZYX convention)
        (see Notes).

        Parameters
        ----------
        theta : ArrayLike
            Set of three Euler angles (ZYX convention), [alpha, beta, gamma], representing
            rotations about the X, Y, and Z axes, respectively.
        degrees : bool, default False
            If True, the input angles are interpreted as degrees. Otherwise, they are
            interpreted as radians. Internally, angles are stored as radians.

        Notes
        -----
        The ZYX Euler angles describe how to transition from the 'navigation' frame
        to the 'body' frame through three consecutive intrinsic and passive rotations
        in the ZYX order.

        Defined as:

            A = R_z(gamma) @ R_y(beta) @ R_x(alpha)

        where,

        - gamma is a first rotation about the navigation frame's Z-axis.
        - beta is a second rotation about the intermediate Y-axis.
        - alpha is a final rotation about the second intermediate X-axis to arrive
          at the body frame.

        and A is the attitude matrix (transforming vectors from the body frame to
        the navigation frame):

            v_n = A @ v_b
        """
        theta = np.asarray_chkfinite(theta, dtype=float).reshape(3).copy()
        if degrees:
            theta *= np.pi / 180.0
        q = _quaternion_from_euler_zyx(theta)
        return cls(q)

    def as_euler(self, degrees: bool = False) -> np.ndarray:
        """
        Convert the attitude to (ZYX) Euler angles (see Notes).

        Parameters
        ----------
        degrees : bool, default False
            If True, the output angles are in degrees. Otherwise, they are in radians.

        Returns
        -------
        np.ndarray
            The 3-element Euler (ZYX) angles, [alpha, beta, gamma], representing
            rotations about the X, Y, and Z axes, respectively.

        Notes
        -----
        The ZYX Euler angles describe how to transition from the 'navigation' frame
        to the 'body' frame through three consecutive intrinsic and passive rotations
        in the ZYX order.

        Defined as:

            A = R_z(gamma) @ R_y(beta) @ R_x(alpha)

        where,

        - gamma is a first rotation about the navigation frame's Z-axis.
        - beta is a second rotation about the intermediate Y-axis.
        - alpha is a final rotation about the second intermediate X-axis to arrive
          at the body frame.

        and A is the attitude matrix (transforming vectors from the body frame to
        the navigation frame):

            v_n = A @ v_b
        """
        theta = _euler_zyx_from_quaternion(self._q)
        if degrees:
            theta *= 180.0 / np.pi
        return theta
