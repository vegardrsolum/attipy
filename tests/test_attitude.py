import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attipy import Attitude


def _assert_quat_allclose(q_actual: np.ndarray, q_desired: np.ndarray, *args, **kwargs):
    """
    Assert that two unit quaternions are equal, considering the double-cover property.
    """
    try:
        np.testing.assert_allclose(q_actual, q_desired, *args, **kwargs)
    except AssertionError:
        np.testing.assert_allclose(q_actual, -q_desired, *args, **kwargs)


class Test_Attitude:
    euler_deg_data = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (-10.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, -10.0, 0.0),
        (0.0, 0.0, 10.0),
        (0.0, 0.0, -10.0),
        (1.0, 2.0, 3.0),
        (-1.0, -2.0, -3.0),
        (90.0, 25.0, -30.0),
        (-90.0, -25.0, 30.0),
    ]

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test__init__(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        att = Attitude(q)

        _assert_quat_allclose(att._q, q)

    def test__repr__(self):
        q = [0.52005444, -0.51089824, 0.64045922, 0.24153336]
        att = Attitude(q)
        repr_str = repr(att)
        expected_str = "Attitude(q=[0.52 + -0.511i + 0.64j + 0.242k])"
        assert repr_str == expected_str

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_quaternion(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        att = Attitude.from_quaternion(q)

        _assert_quat_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_matrix(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        matrix = R.as_matrix()

        att = Attitude.from_matrix(matrix)

        _assert_quat_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_as_matrix(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        matrix = R.as_matrix()

        att = Attitude(q)
        matrix_out = att.as_matrix()

        np.testing.assert_allclose(matrix_out, matrix)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_euler_deg(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        att = Attitude.from_euler(euler_deg, degrees=True)

        _assert_quat_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_euler_rad(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        euler_rad = np.radians(euler_deg)
        att = Attitude.from_euler(euler_rad, degrees=False)

        _assert_quat_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_as_euler_deg(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        att = Attitude(q)
        euler_out = att.as_euler(degrees=True)

        np.testing.assert_allclose(euler_out, euler_deg)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_as_euler_rad(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        att = Attitude(q)
        euler_out = att.as_euler(degrees=False)

        np.testing.assert_allclose(euler_out, np.radians(euler_deg))
