import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit  # type: ignore[misc]
def _canonical(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return the quaternion in canonical (standardized) form.

    Ensures a unique sign by enforcing: w > 0, or if w == 0 then x > 0, etc.
    """
    qw, qx, qy, qz = q

    flip = (
        qw < 0
        or (qw == 0 and qx < 0)
        or (qw == 0 and qx == 0 and qy < 0)
        or (qw == 0 and qx == 0 and qy == 0 and qz < 0)
    )

    if flip:
        return -q

    return q


@njit  # type: ignore[misc]
def _quatprod(qa: NDArray[np.float64], qb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes the product of two unit quaternions (Hamilton product):

        q = q_a ⊗ q_b

    Defined as:

        qw = qw_a * qw_b - np.dot(qxyz_a, qxyz_b)
        qxyz = qw_a * qxyz_b + qw_b * qxyz_a + np.cross(qxyz_a, qxyz_b)

    Parameters
    ----------
    qa, qb : numpy.ndarray, shape (4,)
        Unit quaternions (qw, qx, qy, qz).

    Returns
    -------
    numpy.ndarray, shape (4,)
        Unit quaternion product.

    References
    ----------
    .. [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
    2nd Edition, equation 2.69, John Wiley & Sons, 2021.
    """
    qw_a, qx_a, qy_a, qz_a = qa
    qw_b, qx_b, qy_b, qz_b = qb

    qw = qw_a * qw_b - qx_a * qx_b - qy_a * qy_b - qz_a * qz_b
    qx = qw_a * qx_b + qw_b * qx_a + qy_a * qz_b - qz_a * qy_b
    qy = qw_a * qy_b + qw_b * qy_a + qz_a * qx_b - qx_a * qz_b
    qz = qw_a * qz_b + qw_b * qz_a + qx_a * qy_b - qy_a * qx_b

    return np.array([qw, qx, qy, qz])


@njit  # type: ignore[misc]
def _normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    L2-normalize a quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Quaternion to be normalized

    Returns
    -------
    numpy.ndarray, shape (4,)
        Normalized (unit) quaternion.
    """
    qw, qx, qy, qz = q
    norm_inv = 1.0 / np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    return q * norm_inv


@njit  # type: ignore[misc]
def _correct_quat_with_gibbs2(q, da):
    """
    Corrects a unit quaternion, q, with a small attitude error, da, parameterized
    as a scaled (2x) Gibbs vector.

    The correction is applied as:

        q = q ⊗ dq

    where ⊗ denotes the quaternion product (Hamilton product), and dq is the unit
    quaternion corresponding to the scaled (2x) Gibbs vector da:

        dq = 1 / sqrt(4 + ||da||^2) * [2, dax, day, daz]

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion [qw, qx, qy, qz] (modified in place).
    da : ndarray, shape (3,)
        Small attitude error parameterized as a scaled (2x) Gibbs vector.

    Returns
    -------
    ndarray, shape (4,)
        Corrected (renormalized) unit quaternion.

    Notes
    -----
    As described in ref [1]_, this correction can be simplified by doing it in two
    steps: first a first-order correction, followed by renormalization. The scaling
    factor becomes obsolete due to the renormalization step.

    References
    ----------
    Markley & Crassidis (2014), Fundamentals of Spacecraft Attitude Determination
    and Control, Eq. (6.27)-(6.28).
    """

    qw, qx, qy, qz = q
    dax, day, daz = da

    q[0] -= 0.5 * (qx * dax + qy * day + qz * daz)
    q[1] += 0.5 * (qw * dax + qy * daz - qz * day)
    q[2] += 0.5 * (qw * day - qx * daz + qz * dax)
    q[3] += 0.5 * (qw * daz + qx * day - qy * dax)
    q[:] = _normalize(q)
