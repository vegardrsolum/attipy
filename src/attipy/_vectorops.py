import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
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
def _quaternion_product(
    qa: NDArray[np.float64], qb: NDArray[np.float64]
) -> NDArray[np.float64]:
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


@njit  # type: ignore[misc]
def _quaternion_product(
    qa: NDArray[np.float64], qb: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Unit quaternion (Schur) product: ``qa * qb``.

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
