import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _canonical(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return the quaternion in canonical (standardized) form.

    Ensures a unique sign by enforcing: w > 0, or if w == 0 then x > 0, etc.
    """
    w, x, y, z = q

    flip = (
        w < 0
        or (w == 0 and x < 0)
        or (w == 0 and x == 0 and y < 0)
        or (w == 0 and x == 0 and y == 0 and z < 0)
    )

    if flip:
        return -q

    return q


@njit  # type: ignore[misc]
def _quatprod(qa: NDArray[np.float64], qb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Unit quaternion (Hamilton) product: q_a ⊗ q_b:

        qw = qw_a * qw_b - np.dot(qxyz_a, qxyz_b)
        qxyz = qw_a * qxyz_b + qw_b * qxyz_a + np.cross(qxyz_a, qxyz_b)

    Parameters
    ----------
    qa, qb : numpy.ndarray, shape (4,)
        Unit quaternions.

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternion product.
    """
    qw_a, qx_a, qy_a, qz_a = qa
    qw_b, qx_b, qy_b, qz_b = qb

    qw = qw_a * qw_b - qx_a * qx_b - qy_a * qy_b - qz_a * qz_b
    qx = qw_a * qx_b + qw_b * qx_a + qy_a * qz_b - qz_a * qy_b
    qy = qw_a * qy_b + qw_b * qy_a + qz_a * qx_b - qx_a * qz_b
    qz = qw_a * qz_b + qw_b * qz_a + qx_a * qy_b - qy_a * qx_b

    return np.array([qw, qx, qy, qz])
