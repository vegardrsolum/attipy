import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attipy._transforms import _rot_matrix_from_quaternion


@pytest.mark.parametrize(
    "q",
    [
        np.array([0.96591925, -0.25882081, 0.0, 0.0], dtype=float),  # about x-axis
        np.array([0.96591925, 0.0, -0.25882081, 0.0], dtype=float),  # about y-axis
        np.array([0.96591925, 0.0, 0.0, -0.25882081], dtype=float),  # about z-axis
    ],
)
def test_rot_matrix_from_quaternion(q):
    rot_matrix = _rot_matrix_from_quaternion(q)
    rot_matrix_expect = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_array_almost_equal(rot_matrix, rot_matrix_expect, decimal=3)
