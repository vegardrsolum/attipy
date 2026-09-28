"""
Degree of freedom (DOF) signal generators for building rigid body motions.
"""

from ._dof import DOF, Beat, Constant, RampUp, Sine, add, multiply

__all__ = [
    "DOF",
    "Beat",
    "Constant",
    "RampUp",
    "Sine",
    "add",
    "multiply",
]
