import numpy as np
from numpy.typing import ArrayLike

from ._transforms import _matrix_from_quaternion, _quaternion_from_matrix


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
    Encapsulates the attitude (or rotation) of a 'body frame', `{b}`, relative to
    a 'navigation frame', `{n}`.

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
    def from_matrix(cls, A: ArrayLike):
        """
        Create an Attitude instance from a rotation matrix (or direction cosine matrix),
        defined such that it transforms vectors from the body frame to the navigation
        frame using:

            v_n = A @ v_b

        where,
        - ``A`` is the 3x3 rotation matrix (or direction cosine matrix).
        - v_b is a vector expressed in the body frame.
        - v_n is the same vector expressed in the navigation frame.

        Parameters
        ----------
        A : ArrayLike
            Rotation matrix (element of SO(3)).
        """
        A = _asarray_check_matrix_so3(A)
        q = _quaternion_from_matrix(A)
        return cls(q)
    
    def as_quaternion(self) -> np.ndarray:
        """
        Return the attitude as a unit quaternion, q, defined such that it transforms
        a vector from the body frame to the navigation frame using:

            [0, v_n] = q ⊗ [0, v_b] ⊗ q*

        where,

        - q is the unit quaternion.
        - q* is the conjugate of the unit quaternion q.
        - v_b is a vector expressed in the body frame.
        - v_n is the same vector expressed in the navigation frame.

        and ⊗ denotes quaternion multiplication (Hamilton product).
        """
        return self._q.copy()

    def as_matrix(self) -> np.ndarray:
        """
        Return the attitude as a rotation matrix (or direction cosine matrix), A,
        defined such that it transforms vectors from the body frame to the navigation
        frame using:

            v_n = A @ v_b

        where,

        - ``A`` is the 3x3 attitude matrix.
        - v_b is a vector expressed in the body frame.
        - v_n is the same vector expressed in the navigation frame.

        The rotation matrix is computed from the unit quaternion using the formula:

            A = I + 2 * q_w * S(q_xyz) + 2 * S(q_xyz)^2

        where,

        - I is the 3x3 identity matrix.
        - q_w is the scalar part of the unit quaternion.
        - q_xyz is the vector part, [q_x, q_y, q_z], of the unit quaternion.
        - S(q_xyz) is the skew-symmetric matrix of q_xyz.
        """
        return _matrix_from_quaternion(self._q)
