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
        ],
    )
    def test_from_euler_deg(self, euler_deg):
        att = Attitude.from_euler(euler_deg, degrees=True)
        q_out = att._q
        q_expect = Rotation.from_euler("ZYX", euler_deg[::-1], degrees=True).as_quat(
            scalar_first=True
        )
        np.testing.assert_allclose(q_out, q_expect)

    def test_as_euler_deg(self):
        euler = np.array([10.0, 20.0, -30.0])
        att = Attitude.from_euler(euler, degrees=True)
        euler_out = att.as_euler(degrees=True)
        np.testing.assert_allclose(euler_out, euler)

    def test_as_euler_rad(self):
        euler = np.radians(np.array([-10.0, -20.0, 30.0]))
        att = Attitude.from_euler(euler, degrees=False)
        euler_out = att.as_euler(degrees=False)
        np.testing.assert_allclose(euler_out, euler)
