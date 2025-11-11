from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ._transforms import (
    _euler_from_quaternion,
    _rot_matrix_from_euler,
    _rot_matrix_from_quaternion,
    _quaternion_from_euler,
)
from ._vectorops import _normalize


class AttitudeBase(ABC):
    @abstractmethod
    def _toarray(self) -> np.ndarray:
        """
        Return the attitude representation as a ``numpy.ndarray``.
        """
        raise NotImplementedError("Not implemented.")

    def toarray(self) -> np.ndarray:
        """
        Return the attitude representation as a ``numpy.ndarray``.
        """
        return self._toarray().copy()

    def __repr__(self):
        class_name = self.__class__.__name__
        array_str = np.array2string(self._toarray())
        return f"{class_name}({array_str})"


class AttitudeMatrix(AttitudeBase):
    """
    Rotation matrix (or direction cosine matrix) representation of a rotation in
    3D space.

    Defined as:

        v_n = A @ v_b

    where,

    - A is the 3x3 attitude matrix.
    - v_b is a vector expressed in the body frame.
    - v_n is the same vector expressed in the navigation frame.
    """

    def __init__(self, A: ArrayLike) -> None:
        self._A = np.asarray_chkfinite(A, dtype=float).reshape(3, 3)
        if self._A.shape != (3, 3):
            raise ValueError("Attitude matrix must be a 3x3 matrix.")

    def _toarray(self) -> np.ndarray:
        return self._A

    def __repr__(self):
        class_name = self.__class__.__name__
        array_str = np.array2string(self._toarray())
        array_str = ("\n " + len(class_name) * " ").join(array_str.split("\n"))
        return f"{class_name}({array_str})"

    @classmethod
    def from_quaternion(cls, q: ArrayLike | "UnitQuaternion") -> "AttitudeMatrix":
        """
        Create an AttitudeMatrix from a unit quaternion.

        Parameters
        ----------
        q : ArrayLike or UnitQuaternion
            The unit quaternion, [q_w, q_x, q_y, q_z], representing the 3D rotation.

        Returns
        -------
        AttitudeMatrix
            The corresponding attitude matrix, A.
        """
        if isinstance(q, UnitQuaternion):
            q = q.toarray()
        q = np.asarray_chkfinite(q, dtype=float).reshape(4)
        A = _rot_matrix_from_quaternion(q)
        return cls(A)

    @classmethod
    def from_euler_zyx(cls, euler: ArrayLike | "EulerZYX") -> "AttitudeMatrix":
        """
        Create an AttitudeMatrix from (ZYX) Euler angles.

        Parameters
        ----------
        euler : ArrayLike or EulerZYX
            The (ZYX) Euler angles, [alpha, beta, gamma], representing the 3D rotation.

        Returns
        -------
        AttitudeMatrix
            The corresponding attitude matrix, A.
        """
        if isinstance(euler, EulerZYX):
            euler = euler.toarray()
        euler = np.asarray_chkfinite(euler, dtype=float).reshape(3)
        A = _rot_matrix_from_euler(euler)
        return cls(A)


class UnitQuaternion(AttitudeBase):
    """
    Unit quaternion representation, [q_w, q_x, q_y, q_z], of a rotation in 3D space.

    Defined as:

        [0, v_n] = q* ⊗ [0, v_b] ⊗ q

    where,

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
        self._q = _normalize(np.asarray_chkfinite(q, dtype=float).reshape(4))

    def _toarray(self) -> np.ndarray:
        return self._q

    @classmethod
    def from_euler_zyx(cls, euler: ArrayLike | "EulerZYX") -> "UnitQuaternion":
        """
        Create a UnitQuaternion from (ZYX) Euler angles.

        Parameters
        ----------
        euler : ArrayLike or EulerZYX
            The (ZYX) Euler angles, [alpha, beta, gamma], representing the 3D rotation.

        Returns
        -------
        UnitQuaternion
            The corresponding unit quaternion, q.
        """
        if isinstance(euler, EulerZYX):
            euler = euler.toarray()
        euler = np.asarray_chkfinite(euler, dtype=float).reshape(3)
        q = _quaternion_from_euler(euler)
        return cls(q)


class EulerZYX(AttitudeBase):
    """
    (ZYX) Euler angle representation of a rotation in 3D space.

    The ZYX Euler angles describe how to transition from the 'navigation' frame to
    the 'body' frame through three consecutive intrinsic and passive rotations in
    the ZYX order.

    Defined as:

        A = R_z(gamma) @ R_y(beta) @ R_x(alpha)

    where,

    - gamma is a first rotation about the navigation frame's Z-axis.
    - beta is a second rotation about the intermediate Y-axis.
    - alpha is a final rotation about the second intermediate X-axis to arrive
      at the body frame.

    and A is the attitude matrix (transforming vectors from the body frame to
    the navigation frame).

    Parameters
    ----------
    theta : ArrayLike
        The 3-element Euler (ZYX) angles, [alpha, beta, gamma], representing
        rotations about the X, Y, and Z axes, respectively.
    degrees : bool, default False
        If True, the input angles are interpreted as degrees. Otherwise, they are
        interpreted as radians. Internally, angles are stored as radians.
    """

    def __init__(self, theta: ArrayLike, degrees: bool = False) -> None:
        self._theta = np.asarray_chkfinite(theta, dtype=float).reshape(3)
        self._degrees = degrees
        if self._degrees:
            self._theta *= np.pi / 180

    def _toarray(self) -> np.ndarray:
        return self._theta

    @classmethod
    def from_quaternion(cls, q: ArrayLike | "UnitQuaternion") -> "EulerZYX":
        if isinstance(q, UnitQuaternion):
            q = q.toarray()
        theta = _euler_from_quaternion(q)
        return cls(theta)
