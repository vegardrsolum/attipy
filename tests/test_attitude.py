import numpy as np

from attipy import UnitQuaternion, AttitudeMatrix


class Test_UnitQuaternion:
    def test__init__(self):
        q = UnitQuaternion([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q.value, [1.0, 0.0, 0.0, 0.0])


class Test_AttitudeMatrix:
    def test__init__(self):
        I_ = np.eye(3)
        A = AttitudeMatrix(I_)
        np.testing.assert_allclose(A.value, I_)

    def test_from_quaternion(self):
        A = AttitudeMatrix.from_quaternion([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(A.value, np.eye(3))
