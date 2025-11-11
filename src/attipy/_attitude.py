from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ._transforms import _rot_matrix_from_quaternion
from ._vectorops import _normalize


class AttitudeBase(ABC):
    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError("Not implemented.")


class AttitudeMatrix(AttitudeBase):
    """
    Rotation matrix (or direction cosine matrix) representation of a rotation in
    3D space.

    Defined as:

        v_n = A @ v_b

    where
    - A is the 3x3 attitude matrix.
    - v_b is a vector expressed in the body frame.
    - v_n is the same vector expressed in the navigation frame.
    """

    def __init__(self, A: ArrayLike) -> None:
        self._A = np.asarray_chkfinite(A).reshape(3, 3)
        if self._A.shape != (3, 3):
            raise ValueError("Attitude matrix must be a 3x3 matrix.")

    @property
    def value(self) -> np.ndarray:
        return self._A.copy()

    @classmethod
    def from_quaternion(cls, q: ArrayLike | "UnitQuaternion") -> "AttitudeMatrix":
        if isinstance(q, UnitQuaternion):
            q = q.value
        A = _rot_matrix_from_quaternion(q)
        return cls(A)


class UnitQuaternion(AttitudeBase):
    """
    Unit quaternion representation, (q_w, q_x, q_y, q_z), of a rotation in 3D space.

    Parameters
    ----------
    q : ArrayLike
        The 4-element unit quaternion, (q_w, q_x, q_y, q_z), where q_w is the scalar
        part, and q_x, q_y and q_z are the vector parts.
    """

    def __init__(self, q: ArrayLike) -> None:
        self._q = _normalize(np.asarray_chkfinite(q).reshape(4))

    @property
    def value(self) -> np.ndarray:
        return self._q.copy()


class EulerZYX(AttitudeBase):
    """
    Euler (ZYX) angles representation of a rotation in 3D space.

    Parameters
    ----------
    theta : ArrayLike
        The 3-element Euler (ZYX) angles, (theta_z, theta_y, theta_x), representing
        rotations about the Z, Y, and X axes, respectively.
    """

    def __init__(self, theta: ArrayLike, degrees: bool = False) -> None:
        self._theta = np.asarray_chkfinite(theta).reshape(3)
        self._degrees = degrees
        if self._degrees:
            self._theta *= np.pi / 180

    @property
    def value(self) -> np.ndarray:
        return self._theta.copy()
