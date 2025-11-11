from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ._transforms import _rot_matrix_from_quaternion
from ._vectorops import _normalize


class AttitudeBase(ABC):
    @abstractmethod
    def _toarray(self):
        """
        Return the attitude representation as a ``numpy.ndarray``.
        """
        raise NotImplementedError("Not implemented.")

    def toarray(self):
        """
        Return the attitude representation as a ``numpy.ndarray``.
        """
        return self._toarray().copy()


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
        self._A = np.asarray_chkfinite(A).reshape(3, 3)
        if self._A.shape != (3, 3):
            raise ValueError("Attitude matrix must be a 3x3 matrix.")

    def _toarray(self) -> np.ndarray:
        return self._A

    @classmethod
    def from_quaternion(cls, q: ArrayLike | "UnitQuaternion") -> "AttitudeMatrix":
        if isinstance(q, UnitQuaternion):
            q = q.toarray()
        A = _rot_matrix_from_quaternion(q)
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
        self._q = _normalize(np.asarray_chkfinite(q).reshape(4))

    def _toarray(self) -> np.ndarray:
        return self._q


class EulerZYX(AttitudeBase):
    """
    Euler (ZYX) angle representation of a rotation in 3D space.

    Defined as:

        A = R_z(theta_z) @ R_y(theta_y) @ R_x(theta_x)

    where,

    - theta_z is a first rotation about the navigation frame's Z-axis.
    - theta_y is a subsequent rotation about the intermediate Y-axis.
    - theta_x is a final rotation about the second intermediate X-axis to arrive
      at the body frame.

    and A is the attitude matrix (transforming vectors from the body frame to
    the navigation frame).

    Parameters
    ----------
    theta : ArrayLike
        The 3-element Euler (ZYX) angles, [theta_x, theta_y, theta_z], representing
        rotations about the X, Y, and Z axes, respectively.
    degrees : bool, default False
        If True, the input angles are interpreted as degrees. Otherwise, they are
        interpreted as radians. Internally, angles are stored as radians.
    """

    def __init__(self, theta: ArrayLike, degrees: bool = False) -> None:
        self._theta = np.asarray_chkfinite(theta).reshape(3)
        self._degrees = degrees
        if self._degrees:
            self._theta *= np.pi / 180

    def _toarray(self) -> np.ndarray:
        return self._theta
