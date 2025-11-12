import numpy as np

from attipy import AttitudeMatrix, UnitQuaternion


class Test_AttitudeMatrix:
    def test__init__(self):
        I_ = np.eye(3)
        A = AttitudeMatrix(I_)
        np.testing.assert_allclose(A.toarray(), I_)

    def test_from_quaternion(self):
        A = AttitudeMatrix.from_quaternion([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(A.toarray(), np.eye(3))

    def test_from_euler(self):
        A = AttitudeMatrix.from_euler([0.0, 0.0, 0.0])
        np.testing.assert_allclose(A.toarray(), np.eye(3))


class Test_UnitQuaternion:
    def test__init__(self):
        q = UnitQuaternion([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q.toarray(), [1.0, 0.0, 0.0, 0.0])

    def test_from_euler(self):
        euler = np.array([0.0, 0.0, 0.0])
        q = UnitQuaternion.from_euler(euler)
        np.testing.assert_allclose(q.toarray(), [1.0, 0.0, 0.0, 0.0])
