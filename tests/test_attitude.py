import pytest
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from attipy import Attitude


TEST_PATH = Path(__file__).parent


@pytest.fixture
def ahrs_data():
    import csv

    path = TEST_PATH / r"testdata/benchmark_pure_attitude_beat_202311A.csv"
    with open(path, mode="r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.asarray_chkfinite(list(reader))

    col_map = {name: i for i, name in enumerate(header)}

    t = data[col_map["Time_s"]].astype(float)
    roll = data[col_map["Roll_rad"]].astype(float)
    pitch = data[col_map["Pitch_rad"]].astype(float)
    yaw = data[col_map["Yaw_rad"]].astype(float)
    gx = data[col_map["GyroX_rads"]].astype(float)
    gy = data[col_map["GyroY_rads"]].astype(float)
    gz = data[col_map["GyroZ_rads"]].astype(float)
    ax = data[col_map["AccX_ms2"]].astype(float)
    ay = data[col_map["AccY_ms2"]].astype(float)
    az = data[col_map["AccZ_ms2"]].astype(float)

    euler = np.column_stack([roll, pitch, yaw])
    w = np.column_stack([gx, gy, gz])
    f = np.column_stack([ax, ay, az])

    return t, euler, f, w


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
        matrix = R.as_matrix()

        att = Attitude.from_matrix(matrix)

        np.testing.assert_allclose(att._q, q)

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

    def test_update(self, ahrs_data):
        _, euler, _, w = ahrs_data

        dt = 1.0 / 10.24
        att = Attitude.from_euler(euler[0])

        euler_out = []
        for w_i in w:
            att.update(w_i * dt, degrees=False)
            euler_out.append(att.as_euler(degrees=False))
        
        euler_out = np.asarray(euler_out)

        print(w[0])

        np.testing.assert_allclose(euler_out, euler, atol=0.01)
