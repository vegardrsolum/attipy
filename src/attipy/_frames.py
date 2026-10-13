import numpy as np
from numpy.typing import NDArray


def _gravity_nav(g: float, nav_frame: str) -> NDArray[np.float64]:
    """
    Gravity vector expressed in the navigation frame ('NED' or 'ENU').

    Parameters
    ----------
    g : float
        Gravitational acceleration in m/s^2.
    nav_frame : {'NED', 'ENU'}
        Navigation frame in which the gravity vector is expressed.

    Returns
    -------
    ndarray, shape (3,)
        Gravity vector expressed in the navigation frame.
    """
    if nav_frame.lower() == "ned":
        g_n = np.array([0.0, 0.0, g])
    elif nav_frame.lower() == "enu":
        g_n = np.array([0.0, 0.0, -g])
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
    return g_n


def _nz2vg(nav_frame: str) -> float:
    """
    Gravity direction along the navigation frame's z-axis. Transforms the z-axis
    of the navigation frame to a gravity reference vector (unit vector).

    Parameters
    ----------
    nav_frame : {'NED', 'ENU'}
        Navigation frame.

    Returns
    -------
    float
        Gravity direction along the navigation frame's z-axis. +1.0 for 'NED' and
        -1.0 for 'ENU'.
    """
    if nav_frame.lower() == "ned":
        return 1.0
    elif nav_frame.lower() == "enu":
        return -1.0
    else:
        raise ValueError(f"Unknown navigation frame: {nav_frame}.")
