import numpy as np

from attipy import AttitudeMatrix, EulerZYX, UnitQuaternion


class Test_AttitudeMatrix:
    def test__init__(self):
        I_ = np.eye(3)
        A = AttitudeMatrix(I_)
        np.testing.assert_allclose(A.toarray(), I_)

    def test_from_quaternion(self):
        A = AttitudeMatrix.from_quaternion([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(A.toarray(), np.eye(3))

    def test_from_euler_zyx(self):
        A = AttitudeMatrix.from_euler_zyx([0.0, 0.0, 0.0])
        np.testing.assert_allclose(A.toarray(), np.eye(3))


class Test_UnitQuaternion:
    def test__init__(self):
        q = UnitQuaternion([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q.toarray(), [1.0, 0.0, 0.0, 0.0])


class Test_EulerZYX:
    def test__init__(self):
        theta = np.array([np.pi / 8, np.pi / 4, np.pi / 2])
        euler = EulerZYX(theta)
        np.testing.assert_allclose(euler.toarray(), np.array(theta))

    def test__init__radians(self):
        theta = np.array([np.pi / 8, np.pi / 4, np.pi / 2])
        euler = EulerZYX(theta, degrees=False)
        np.testing.assert_allclose(euler.toarray(), np.array(theta))

    def test__init__degrees(self):
        theta = [10.0, 20.0, 30.0]
        euler = EulerZYX(theta, degrees=True)
        np.testing.assert_allclose(euler.toarray(), (np.pi / 180.0) * np.array(theta))

    def test_from_quaternion(self):
        q = [1.0, 0.0, 0.0, 0.0]
        euler = EulerZYX.from_quaternion(q)
        np.testing.assert_allclose(euler.toarray(), np.array([0.0, 0.0, 0.0]))
