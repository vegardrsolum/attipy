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
def _correct_quat_with_gibbs2(q: NDArray[np.float64], da: NDArray[np.float64]) -> None:
    """
    Update/correct a unit quaternion, q, with a small attitude error, da, parameterized
    as a scaled (2x) Gibbs vector.

    The correction is applied as:

        q = q ⊗ dq(da)

    where ⊗ denotes the quaternion product (Hamilton product), and dq(da) is the
    unit quaternion corresponding to the scaled (2x) Gibbs vector da:

        dq(da) = 1 / sqrt(4 + ||da||^2) * [2, dax, day, daz]

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion [qw, qx, qy, qz] (modified in place).
    da : ndarray, shape (3,)
        Small attitude error (dax, day, daz) parameterized as a scaled (2x) Gibbs vector.

    Notes
    -----
    As described in ref [1]_, this correction can be simplified by doing it in two
    steps: first a correction, followed by renormalization. The scaling factor becomes
    obsolete due to the renormalization step.

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


@njit  # type: ignore[misc]
def _correct_quat_with_rotvec(
    q: NDArray[np.float64], dtheta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Update/correct a unit quaternion, q, with a small attitude increment, dtheta,
    parameterized as a rotation vector.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Unit quaternion (qw, qx, qy, qz) to be updated (in place).
    dtheta : ndarray, shape (3,)
        Attitude increment (rotation vector).

    References
    ----------
    .. [1] https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-coning (Eq. 2.5.1)
    """

    qw, qx, qy, qz = q
    rx, ry, rz = dtheta

    gamma = 0.5 * np.sqrt(rx**2 + ry**2 + rz**2)
    cos_gamma = np.cos(gamma)

    if gamma >= 1e-5:
        scale = np.sin(gamma) / (2.0 * gamma)
    else:
        scale = 0.5

    # Psi
    px = scale * rx
    py = scale * ry
    pz = scale * rz

    q[0] = cos_gamma * qw - px * qx - py * qy - pz * qz
    q[1] = px * qw + cos_gamma * qx + pz * qy - py * qz
    q[2] = py * qw - pz * qx + cos_gamma * qy + px * qz
    q[3] = pz * qw + py * qx - px * qy + cos_gamma * qz
    q[:] = _normalize(q)
