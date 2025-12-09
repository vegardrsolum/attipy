from abc import ABC, abstractmethod

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


class _AttitudeBackend(ABC):
    @abstractmethod
    def _from_quaternion(self) -> np.ndarray:
        """
        Transform the attitude representation from a unit quaternion.
        """
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _to_quaternion(self) -> np.ndarray:
        """
        Transform the attitude representation to a unit quaternion.
        """
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _from_matrix(self) -> np.ndarray:
        """
        Transform the attitude representation from a rotation matrix.
        """
        raise NotImplementedError("Not implemented.")

    def _to_matrix(self) -> np.ndarray:
        """
        Transform the attitude representation to a rotation matrix.
        """
        raise NotImplementedError("Not implemented.")
    

class _QuaternionBackend(_AttitudeBackend):

    def _from_quaternion(self, q) -> np.ndarray:
        return _asarray_check_unit_quaternion(q)

    def _from_matrix(self, A) -> np.ndarray:
        A = _asarray_check_matrix_so3(A)
        q = _R.from_matrix(A).as_quat()
        return _asarray_check_unit_quaternion(q)
    
    def _to_quaternion(self, att) -> np.ndarray:
        return att
    
    def _to_matrix(self, att) -> np.ndarray:
        A = _R.from_quat(att).as_matrix()
        return A


class AttitudeBase(ABC):
    """
    Base class for attitude representation, i.e., the encapsulation of a 3D rotation
    of a 'body' relative to a reference frame (the 'navigation frame').

    The unit quaternion is used to represent the attitude internally. Thus, other
    representations should be convertable to and from the unit quaternion representation.
    The unit quaternion, q, is defined such that it transforms vectors from the body
    frame to the navigation frame using:

        [0, v_n] = q ⊗ [0, v_b] ⊗ q*

    where,

    - q is the unit quaternion.
    - q* is the conjugate of the unit quaternion q.
    - v_b is a vector expressed in the body frame.
    - v_n is the same vector expressed in the navigation frame.

    and ⊗ denotes quaternion multiplication (Hamilton product).

    Inheriting classes should define the following methods:
    - ``_asarray()`` which returns the attitude representation as a ``numpy.ndarray``.
    - ``_to_quaternion()`` which transforms the attitude representation to the unit
        quaternion representation.
    """

    _backend: _AttitudeBackend = _QuaternionBackend()

    def __init__(self, q):
        self._att = self._backend._from_quaternion(q)

    @classmethod
    def from_quaternion(cls, q) -> np.ndarray:
        """
        Create object from a unit quaternion.
        """
        return cls(q)
    
    def to_quaternion(self) -> np.ndarray:
        """
        Return the attitude representation as a unit quaternion.
        """
        return self._backend._to_quaternion(self._att)

    @classmethod
    def from_matrix(cls, A) -> np.ndarray:
        """
        Create object from a rotation matrix.
        """
        att = cls._backend._from_matrix(A)
        return cls(att)

    def to_matrix(self) -> np.ndarray:
        """
        Return the attitude representation as a rotation matrix.
        """
        return self._backend._to_matrix(self._att)

    @abstractmethod
    def _asarray(self) -> np.ndarray:
        """
        Transform the attitude representation from a unit quaternion.
        """
        raise NotImplementedError("Not implemented.")

    def asarray(self) -> np.ndarray:
        """
        Return the attitude representation as a ``numpy.ndarray``.
        """
        return self._asarray()

    def __repr__(self):
        class_name = self.__class__.__name__
        array = self._asarray()
        array_str = np.array2string(array)
        if array.ndim == 2:
            array_str = ("\n " + len(class_name) * " ").join(array_str.split("\n"))
        return f"{class_name}({array_str})"


class AttitudeMatrix(AttitudeBase):
    """
    Rotation matrix (or direction cosine matrix) representation of an attitude
    (or rotation) in 3D space, encapsulating the orientation of a 'body frame',
    `{b}`, relative to a 'navigation frame', `{n}`.

    The matrix is defined such that it transforms vectors from the body frame to
    the navigation frame:

        v_n = A @ v_b

    where,

    - ``A`` is the 3x3 attitude matrix.
    - ``v_b`` is a vector expressed in the body frame.
    - ``v_n`` is the same vector expressed in the navigation frame.

    The matrix must be orthonormal with determinant +1 (i.e., valid member of SO(3)).

    Parameters
    ----------
    A : ArrayLike
        3x3 rotation matrix mapping body-frame vectors to navigation-frame vectors.
    """

    def __init__(self, A: ArrayLike) -> None:
        super().__init__(self._backend._from_matrix(A))

    def _asarray(self) -> np.ndarray:
        return self.to_matrix()

    # def _from_quaternion(self):
    #     return _R.from_quat(self._q).as_matrix()

    # @classmethod
    # def from_quaternion(cls, q: ArrayLike | "UnitQuaternion") -> "AttitudeMatrix":
    #     """
    #     Create an attitude matrix from a unit quaternion, using the relation:

    #         A = I + 2 * q_w * S(q_xyz) + 2 * S(q_xyz)^2

    #     where,

    #     - I is the 3x3 identity matrix.
    #     - q_w is the scalar part of the unit quaternion.
    #     - q_xyz is the vector part, [q_x, q_y, q_z], of the unit quaternion.
    #     - S(q_xyz) is the skew-symmetric matrix of q_xyz.

    #     Parameters
    #     ----------
    #     q : ArrayLike or UnitQuaternion
    #         The unit quaternion, [q_w, q_x, q_y, q_z], representing the 3D rotation.

    #     Returns
    #     -------
    #     AttitudeMatrix
    #         The corresponding attitude matrix, A.
    #     """
    #     if isinstance(q, UnitQuaternion):
    #         q = q.asarray()
    #     q = _asarray_check_unit_quaternion(q).copy()
    #     A = _rot_matrix_from_quaternion(q)
    #     return cls(A)

    # @classmethod
    # def from_euler(cls, theta: ArrayLike, degrees=False) -> "AttitudeMatrix":
    #     """
    #     Create an AttitudeMatrix from (ZYX) Euler angles (see Notes).

    #     Parameters
    #     ----------
    #     theta : ArrayLike
    #         The 3-element Euler (ZYX) angles, [alpha, beta, gamma], representing
    #         rotations about the X, Y, and Z axes, respectively.
    #     degrees : bool, default False
    #         If True, the input angles are interpreted as degrees. Otherwise, they are
    #         interpreted as radians. Internally, angles are stored as radians.

    #     Returns
    #     -------
    #     AttitudeMatrix
    #         The corresponding attitude matrix, A.

    #     Notes
    #     -----
    #     The ZYX Euler angles describe how to transition from the 'navigation' frame
    #     to the 'body' frame through three consecutive intrinsic and passive rotations
    #     in the ZYX order.

    #     Defined as:

    #         A = R_z(gamma) @ R_y(beta) @ R_x(alpha)

    #     where,

    #     - gamma is a first rotation about the navigation frame's Z-axis.
    #     - beta is a second rotation about the intermediate Y-axis.
    #     - alpha is a final rotation about the second intermediate X-axis to arrive
    #       at the body frame.

    #     and A is the attitude matrix (transforming vectors from the body frame to
    #     the navigation frame):

    #         v_n = A @ v_b
    #     """
    #     theta = np.asarray_chkfinite(theta, dtype=float).reshape(3).copy()
    #     if degrees:
    #         theta *= np.pi / 180.0
    #     A = _rot_matrix_from_euler_zyx(theta)
    #     return cls(A)


class UnitQuaternion(AttitudeBase):
    """
    Unit quaternion representation, [q_w, q_x, q_y, q_z], of an attitude (or rotation)
    in 3D space, encapsulating the orientation of a 'body frame', `{b}`, relative
    to a 'navigation frame', `{n}`.

    The unit quaternion is defined such that it transforms vectors from the body
    frame to the navigation frame:

        [0, v_n] = q ⊗ [0, v_b] ⊗ q*

    where,

    - q* is the conjugate of the unit quaternion q.
    - v_b is a vector expressed in the body frame.
    - v_n is the same vector expressed in the navigation frame.

    and ⊗ denotes quaternion multiplication (Hamilton product).

    The unit quaternion is related to the attitude matrix according to:

        A = I + 2 * q_w * S(q_xyz) + 2 * S(q_xyz)^2

    where,

    - I is the 3x3 identity matrix.
    - q_w is the scalar part of the unit quaternion.
    - q_xyz is the vector part, [q_x, q_y, q_z], of the unit quaternion.
    - S(q_xyz) is the skew-symmetric matrix of q_xyz.

    Parameters
    ----------
    q : ArrayLike
        The 4-element unit quaternion, [q_w, q_x, q_y, q_z], where q_w is the scalar
        part, and q_x, q_y and q_z are the vector parts, respectively.
    """

    def __init__(self, q: ArrayLike) -> None:
        super().__init__(q)

    def _asarray(self):
        return self.to_quaternion()
