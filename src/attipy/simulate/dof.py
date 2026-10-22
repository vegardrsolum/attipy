"""
Degree of freedom (DOF) signal generators for building rigid body motions.
"""

from ._dof import (
    DOF,
    Beat,
    Constant,
    Sine,
    add,
    dofs_from_csd,
    multiply,
    ramp_up,
)

__all__ = [
    "DOF",
    "Beat",
    "Constant",
    "Sine",
    "add",
    "dofs_from_csd",
    "multiply",
    "ramp_up",
]
