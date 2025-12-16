import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _canonical_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
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
def _normalize_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    L2-normalize a vector.

    Parameters
    ----------
    q : numpy.ndarray
        Vector to be normalized

    Returns
    -------
    numpy.ndarray
        Normalized copy of `q`.
    """
    return q / np.sqrt((q * q).sum())  # type: ignore[no-any-return]  # numpy funcs declare Any as return when given scalar-like


@njit  # type: ignore[misc]
def _quatprod(qa: NDArray[np.float64], qb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Unit quaternion (Hamilton) product: q_a ⊗ q_b.

    Parameters
    ----------
    qa, qb : numpy.ndarray, shape (4,)
        Unit quaternions.

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternions result of the product.
    """
    qa_w, qa_xyz = np.split(qa, [1])
    qb_w, qb_xyz = np.split(qb, [1])
    return np.concatenate(
        (
            qa_w * qb_w - qa_xyz.T @ qb_xyz,
            qa_w * qb_xyz + qb_w * qa_xyz + np.cross(qa_xyz, qb_xyz),
        ),
        axis=0,
    )
