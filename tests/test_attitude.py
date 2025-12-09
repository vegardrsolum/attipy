import numpy as np

from attipy import Attitude


class Test_Attitude:
    def test__init__(self):
        A = Attitude([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(A._q, [1.0, 0.0, 0.0, 0.0])

    def test_from_euler(self):
        euler = np.array([0.0, 0.0, 0.0])
        A = Attitude.from_euler(euler)
        np.testing.assert_allclose(A._q, [1.0, 0.0, 0.0, 0.0])

    def test_to_euler_deg(self):
        euler = np.array([10.0, 20.0, -30.0])
        A = Attitude.from_euler(euler, degrees=True)
        euler_out = A.to_euler(degrees=True)
        np.testing.assert_allclose(euler_out, euler)

    def test_to_euler_rad(self):
        euler = np.radians(np.array([-10.0, -20.0, 30.0]))
        A = Attitude.from_euler(euler, degrees=False)
        euler_out = A.to_euler(degrees=False)
        np.testing.assert_allclose(euler_out, euler)
