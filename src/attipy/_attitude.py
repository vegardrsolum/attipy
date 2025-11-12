from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ._transforms import (
    _euler_zyx_from_quaternion,
    _quaternion_from_euler_zyx,
    _rot_matrix_from_euler_zyx,
    _rot_matrix_from_quaternion,
)
from ._vectorops import _normalize


def _asarray_check_quaternion(q: ArrayLike) -> np.ndarray:
    q = np.asarray_chkfinite(q, dtype=float).reshape(4)
    if q.shape != (4,):
        raise ValueError("Quaternion must be a 4-element array.")
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0):
        raise ValueError("Quaternion must be a unit quaternion (norm = 1).")
    return q


def _asarray_check_matrix(A: ArrayLike) -> np.ndarray:
    A = np.asarray_chkfinite(A, dtype=float).reshape(3, 3)
    if A.shape != (3, 3):
        raise ValueError("Matrix must be a 3x3 array.")
    return A


class AttitudeBase(ABC):
    @abstractmethod
    def _asarray(self) -> np.ndarray:
        """
        Return the attitude representation as a ``numpy.ndarray``.
        """
        raise NotImplementedError("Not implemented.")

    def asarray(self) -> np.ndarray:
        """
        Return the attitude representation as a ``numpy.ndarray``.
        """
        return self._asarray().copy()

    def __repr__(self):
        class_name = self.__class__.__name__
        array = self._asarray()
        array_str = np.array2string(array)
        if array.ndim == 2:
            array_str = ("\n " + len(class_name) * " ").join(array_str.split("\n"))
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
        self._A = _asarray_check_matrix(A)

    def _asarray(self) -> np.ndarray:
        return self._A

    @classmethod
    def from_quaternion(cls, q: ArrayLike | "UnitQuaternion") -> "AttitudeMatrix":
        """
        Create an AttitudeMatrix from a unit quaternion.

        The attitude matrix can be derived from the unit quaternion, q, according to:

            A = I + 2 * q_w * S(q_xyz) + 2 * S(q_xyz)^2

        where,

        - I is the 3x3 identity matrix.
        - q_w is the scalar part of the unit quaternion.
        - q_xyz is the vector part of the unit quaternion.
        - S(q_xyz) is the skew-symmetric matrix of q_xyz.

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
            q = q.asarray()
        q = _asarray_check_quaternion(q).copy()
        A = _rot_matrix_from_quaternion(q)
        return cls(A)

    @classmethod
    def from_euler(cls, theta: ArrayLike, degrees=False) -> "AttitudeMatrix":
        """
        Create an AttitudeMatrix from (ZYX) Euler angles (see Notes).

        Parameters
        ----------
        theta : ArrayLike
            The 3-element Euler (ZYX) angles, [alpha, beta, gamma], representing
            rotations about the X, Y, and Z axes, respectively.
        degrees : bool, default False
            If True, the input angles are interpreted as degrees. Otherwise, they are
            interpreted as radians. Internally, angles are stored as radians.

        Returns
        -------
        AttitudeMatrix
            The corresponding attitude matrix, A.

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
            theta *= (np.pi / 180.0)
        A = _rot_matrix_from_euler_zyx(theta)
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
        q = _asarray_check_quaternion(q).copy()
        self._q = _normalize(q)

    def _asarray(self) -> np.ndarray:
        return self._q

    @classmethod
    def from_euler(cls, theta: ArrayLike, degrees: bool = False) -> "UnitQuaternion":
        """
        Create a unit quaternion from (ZYX) Euler angles (see Notes).

        Parameters
        ----------
        theta : ArrayLike
            The 3-element Euler (ZYX) angles, [alpha, beta, gamma], representing
            rotations about the X, Y, and Z axes, respectively.
        degrees : bool, default False
            If True, the input angles are interpreted as degrees. Otherwise, they are
            interpreted as radians. Internally, angles are stored as radians.

        Returns
        -------
        UnitQuaternion
            The corresponding unit quaternion, q.

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
            theta *= (np.pi / 180.0)
        q = _quaternion_from_euler_zyx(theta)
        return cls(q)
    
    def to_euler(self, degrees: bool = False) -> np.ndarray:
        """
        Convert the unit quaternion to (ZYX) Euler angles (see Notes).

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
        q = self._q.copy()
        theta = _euler_zyx_from_quaternion(q)
        if degrees:
            theta *= (180.0 / np.pi)
        return theta