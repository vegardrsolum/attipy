import numpy as np

from attipy import Attitude


class Test_Attitude:
    def test__init__(self):
        att = Attitude([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(att._q, [1.0, 0.0, 0.0, 0.0])

    def test_from_euler(self):
        euler = np.array([0.0, 0.0, 0.0])
        att = Attitude.from_euler(euler)
        np.testing.assert_allclose(att._q, [1.0, 0.0, 0.0, 0.0])

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
