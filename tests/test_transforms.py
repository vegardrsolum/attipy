import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attipy._transforms import (
    _euler_zyx_from_quat,
    _rot_matrix_from_euler_zyx,
    _rot_matrix_from_quat,
)


@pytest.mark.parametrize(
    "q",
    [
        np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float),  # about x-axis
        np.array([0.96591925, 0.0, -0.25882081, 0.0], dtype=float),  # about y-axis
        np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float),  # about z-axis
    ],
)
def test_rot_matrix_from_quat(q):
    rot_matrix = _rot_matrix_from_quat(q)
    rot_matrix_expect = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_array_almost_equal(rot_matrix, rot_matrix_expect, decimal=3)


@pytest.mark.parametrize(
    "angle, axis, euler",
    [
        (
            np.radians(10.0),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, np.radians(10.0)]),
        ),  # pure yaw
        (
            np.radians(10.0),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, np.radians(10.0), 0.0]),
        ),  # pure pitch
        (
            np.radians(10.0),
            np.array([1.0, 0.0, 0.0]),
            np.array([np.radians(10.0), 0.0, 0.0]),
        ),  # pure roll
        (
            np.radians(10.0),
            np.array([1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
            np.array([0.1059987325729154, 0.0953360919950474, 0.1059987325729154]),
        ),  # mixed
    ],
)
def test__euler_zyx_from_quat(angle, axis, euler):
    q = np.array(
        [
            np.cos(angle / 2),
            np.sin(angle / 2) * axis[0],
            np.sin(angle / 2) * axis[1],
            np.sin(angle / 2) * axis[2],
        ]
    )

    alpha_beta_gamma = _euler_zyx_from_quat(q)
    np.testing.assert_array_almost_equal(alpha_beta_gamma, euler, decimal=16)


@pytest.mark.parametrize(
    "euler",
    [
        np.array([10.0, 0.0, 0.0]),  # pure roll
        np.array([0.0, 10.0, 0.0]),  # pure pitch
        np.array([0.0, 0.0, 10.0]),  # pure yaw
        np.array([10.0, -10.0, 10.0]),  # mixed
    ],
)
def test__rot_matrix_from_euler_zyx(euler):
    """
    The Numba optimized implementaiton uses from-origin-to-body (zyx) convention,
    where also the resulting rotation matrix is from-origin-to-body.
    """
    out = _rot_matrix_from_euler_zyx(euler)
    expected = Rotation.from_euler("ZYX", euler[::-1]).as_matrix()
    np.testing.assert_array_almost_equal(out, expected)
