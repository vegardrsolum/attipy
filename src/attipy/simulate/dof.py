"""
Degree of freedom (DOF) signal generators for building rigid body motions.
"""

from ._dof import (
    DOF,
    Beat,
    Constant,
    Sine,
    SmootherStep,
    add,
    from_csd,
    from_psd,
    multiply,
    ramp_up,
)

__all__ = [
    "DOF",
    "Beat",
    "Constant",
    "Sine",
    "SmootherStep",
    "add",
    "from_csd",
    "from_psd",
    "multiply",
    "ramp_up",
]
