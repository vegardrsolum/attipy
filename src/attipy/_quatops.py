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


# @njit  # type: ignore[misc]
# def _quatprod(qa: NDArray[np.float64], qb: NDArray[np.float64]) -> NDArray[np.float64]:
#     """
#     Unit quaternion (Hamilton) product: q_a ⊗ q_b.

#     Parameters
#     ----------
#     qa, qb : numpy.ndarray, shape (4,)
#         Unit quaternions.

#     Returns
#     -------
#     numpy.ndarray, shape (4,)
#         Unit quaternions result of the product.
#     """
#     qa_w, qa_xyz = np.split(qa, [1])
#     qb_w, qb_xyz = np.split(qb, [1])
#     return np.concatenate(
#         (
#             qa_w * qb_w - qa_xyz.T @ qb_xyz,
#             qa_w * qb_xyz + qb_w * qa_xyz + np.cross(qa_xyz, qb_xyz),
#         ),
#         axis=0,
#     )


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
    qw_a, qx_a, qy_a, qz_a = qa
    qw_b, qx_b, qy_b, qz_b = qb

    qw = qw_a * qw_b - qx_a * qx_b - qy_a * qy_b - qz_a * qz_b
    qx = qw_a * qx_b + qx_a * qw_b + qy_a * qz_b - qz_a * qy_b
    qy = qw_a * qy_b - qx_a * qz_b + qy_a * qw_b + qz_a * qx_b
    qz = qw_a * qz_b - qx_a * qy_b + qy_a * qx_b + qz_a * qw_b

    return np.array([qw, qx, qy, qz])
