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
