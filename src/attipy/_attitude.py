import numpy as np
from numpy.typing import ArrayLike

from ._vectorops import _normalize


class UnitQuaternion:
    """
    Unit quaternion representation, (q_w, q_x, q_y, q_z), of a rotation in 3D space.

    Parameters
    ----------
    q : ArrayLike
        The 4-element unit quaternion, (q_w, q_x, q_y, q_z), where q_w is the scalar
        part, and q_x, q_y and q_z are the vector parts.
    """

    def __init__(self, q: ArrayLike) -> None:
        self._q = _normalize(np.asarray_chkfinite(q))

    @property
    def value(self) -> np.ndarray:
        return self._q.copy()


class AttitudeMatrix:
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
        self._A = np.asarray_chkfinite(A)
        if self._A.shape != (3, 3):
            raise ValueError("Attitude matrix must be a 3x3 matrix.")

    @property
    def value(self) -> np.ndarray:
        return self._A.copy()