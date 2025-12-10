import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attipy import Attitude


class Test_Attitude:
    def test__init__(self):
        att = Attitude([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(att._q, [1.0, 0.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        "euler_deg",
        [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
            (1.0, 2.0, 3.0),
            (-1.0, -2.0, -3.0),
            (90.0, 25.0, -130.0),
        ],
    )
    def test_from_euler_deg(self, euler_deg):
        att = Attitude.from_euler(euler_deg, degrees=True)
        q_out = att._q
        q_expect = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True).as_quat(
            scalar_first=True
        )
        np.testing.assert_allclose(q_out, q_expect)

    @pytest.mark.parametrize(
        "euler_deg",
        [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
            (1.0, 2.0, 3.0),
            (-1.0, -2.0, -3.0),
            (90.0, 25.0, -130.0),
        ],
    )
    def test_from_euler_rad(self, euler_deg):
        euler_rad = np.radians(euler_deg)
        att = Attitude.from_euler(euler_rad, degrees=False)
        q_out = att._q
        q_expect = Rotation.from_euler("ZYX", euler_rad[::-1], degrees=False).as_quat(
            scalar_first=True
        )
        np.testing.assert_allclose(q_out, q_expect)

    @pytest.mark.parametrize(
        "euler_deg",
        [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
            (1.0, 2.0, 3.0),
            (-1.0, -2.0, -3.0),
            (90.0, 25.0, -130.0),
        ],
    )
    def test_as_euler_deg(self, euler_deg):
        att = Attitude.from_euler(euler_deg, degrees=True)
        euler_out = att.as_euler(degrees=True)
        np.testing.assert_allclose(euler_out, euler_deg)

    @pytest.mark.parametrize(
        "euler_deg",
        [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
            (1.0, 2.0, 3.0),
            (-1.0, -2.0, -3.0),
            (90.0, 25.0, -130.0),
        ],
    )
    def test_as_euler_rad(self, euler_deg):
        euler_rad = np.radians(euler_deg)
        att = Attitude.from_euler(euler_deg, degrees=True)
        euler_out = att.as_euler(degrees=False)
        np.testing.assert_allclose(euler_out, euler_rad)
