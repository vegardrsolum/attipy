import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attipy import Attitude
from attipy._quatops import _quatprod


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

        np.testing.assert_allclose(att._q, q)

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

        np.testing.assert_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_matrix(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        dcm = R.as_matrix()

        att = Attitude.from_matrix(dcm)

        np.testing.assert_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_as_matrix(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        dcm = R.as_matrix()

        att = Attitude(q)
        matrix_out = att.as_matrix()

        np.testing.assert_allclose(matrix_out, dcm)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_euler_deg(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        att = Attitude.from_euler(euler_deg, degrees=True)

        np.testing.assert_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_euler_rad(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)

        euler_rad = np.radians(euler_deg)
        att = Attitude.from_euler(euler_rad, degrees=False)

        np.testing.assert_allclose(att._q, q)

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

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_rotvec_deg(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        rotvec = R.as_rotvec(degrees=True)

        att = Attitude.from_rotvec(rotvec, degrees=True)

        np.testing.assert_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_from_rotvec_rad(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        rotvec = R.as_rotvec(degrees=False)
        att = Attitude.from_rotvec(rotvec, degrees=False)

        np.testing.assert_allclose(att._q, q)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_as_rotvec_deg(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        rotvec = R.as_rotvec(degrees=True)

        att = Attitude(q)
        rotvec_out = att.as_rotvec(degrees=True)

        np.testing.assert_allclose(rotvec_out, rotvec)

    @pytest.mark.parametrize("euler_deg", euler_deg_data)
    def test_as_rotvec_rad(self, euler_deg):
        # Use scipy as reference
        R = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True)
        q = R.as_quat(scalar_first=True)
        rotvec = R.as_rotvec(degrees=False)
        att = Attitude(q)
        rotvec_out = att.as_rotvec(degrees=False)

        np.testing.assert_allclose(rotvec_out, rotvec)
