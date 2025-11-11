import numpy as np
from numpy.typing import ArrayLike

from ._vectorops import _normalize


class UnitQuaternion:
    """
    Unit quaternion representation of a rotation in 3D space.

    Parameters
    ----------
    q_w : float
        The scalar part of the unit quaternion.
    q_x : float
        The x component of the vector part of the unit quaternion.
    q_y : float
        The y component of the vector part of the unit quaternion.
    q_z : float
        The z component of the vector part of the unit quaternion.
    """

    def __init__(self, q: ArrayLike) -> None:
        self._q = _normalize(np.asarray_chkfinite(q))
