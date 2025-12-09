import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as _R


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


class UnitQuaternion:
    """
    Encapsulates the attitude (or rotation) in 3D space of a 'body frame', `{b}`,
    relative to a 'navigation frame', `{n}`.

    Internally, the attitude is represented using a unit quaternion, q, defined
    such that it transforms vectors from the body frame to the navigation frame
    using:

        [0, v_n] = q ⊗ [0, v_b] ⊗ q*

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

    def __init__(self, q):
        self._q = _asarray_check_unit_quaternion(q)

    def __repr__(self):
        array_str = np.array2string(self._q)
        return f"UnitQuaternion({array_str})"

    @classmethod
    def from_quaternion(cls, q) -> np.ndarray:
        """
        Create object from a unit quaternion.
        """
        q = _asarray_check_unit_quaternion(q)
        return cls(q)

    @classmethod
    def from_matrix(cls, A) -> np.ndarray:
        """
        Create object from a rotation matrix.
        """
        A = _asarray_check_matrix_so3(A)
        q = _R.from_matrix(A).as_quat(scalar_first=True)
        q = _asarray_check_unit_quaternion(q)
        return cls(q)
    
    def as_quaternion(self) -> np.ndarray:
        """
        Return the attitude representation as a unit quaternion.
        """
        return self._q.copy()

    def as_matrix(self) -> np.ndarray:
        """
        Return the attitude representation as a rotation matrix.
        """
        A = _R.from_quat(self._q).as_matrix()
        return A
