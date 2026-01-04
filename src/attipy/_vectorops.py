import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _skew_symmetric(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the cross product equivalent skew symmetric matrix.

    Parameters
    ----------
    a : numpy.ndarray, shape (3,)
        Vector in which the skew symmetric matrix is based on, such that
        ``a x b = S(a) b``.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Skew symmetric matrix.
    """
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


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
