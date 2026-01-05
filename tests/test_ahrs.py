import numpy as np
from pytest import fixture

from attipy import AHRS, Attitude
from attipy._transforms import _quat_from_euler_zyx


class Test_AHRS:

    @fixture
    def ahrs(self):
        return AHRS(10.0)

    def test__init__(self):
        fs = 10.0
        q0 = _quat_from_euler_zyx(np.radians([10.0, 20.0, 30.0]))
        bg0 = np.array([0.01, -0.02, 0.03])
        ahrs = AHRS(fs, q=q0, bg=bg0, nav_frame="NED")

        assert ahrs._fs == fs
        assert ahrs._dt == 1.0 / fs
        assert ahrs._nav_frame == "ned"
        np.testing.assert_allclose(ahrs._g_n, np.array([0.0, 0.0, 9.80665]))
        np.testing.assert_allclose(ahrs.attitude.as_quaternion(), q0)
        np.testing.assert_allclose(ahrs._bg, bg0)
        np.testing.assert_allclose(ahrs._P, 1e-6 * np.eye(9))
        np.testing.assert_allclose(ahrs._f, np.array([0.0, 0.0, -9.80665]))
        np.testing.assert_allclose(ahrs._w, np.zeros(3))

    def test__init__nav_frame(self):
        ahrs_ned = AHRS(10.0, nav_frame="NED")
        np.testing.assert_allclose(ahrs_ned._g_n, np.array([0.0, 0.0, 9.80665]))

        ahrs_enu = AHRS(10.0, nav_frame="ENU")
        np.testing.assert_allclose(ahrs_enu._g_n, np.array([0.0, 0.0, -9.80665]))

    def test_attitude(self, ahrs):
        q_expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(ahrs.attitude, Attitude)
        np.testing.assert_allclose(ahrs.attitude.as_quaternion(), q_expected)

    def test_bias_gyro_rad(self):
        ahrs = AHRS(10.0, bg=np.array([0.01, -0.02, 0.03]))
        bg_expected = np.array([0.01, -0.02, 0.03])
        np.testing.assert_allclose(ahrs.bias_gyro(degrees=False), bg_expected)

    def test_bias_gyro_deg(self):
        ahrs = AHRS(10.0, bg=np.radians([1.0, -2.0, 3.0]))
        bg_expected = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(ahrs.bias_gyro(degrees=True), bg_expected)

    def test_P(self, ahrs):
        P_expected = 1e-6 * np.eye(9)
        np.testing.assert_allclose(ahrs.P, P_expected)

    def test_update(self, pva_data):
        _, _, _, euler, f, w = pva_data
        head = euler[:, 2]
        fs = 10.24

        acc_noise_density = 0.001
        gyro_noise_density = 0.0001
        acc_noise_std = acc_noise_density * np.sqrt(fs)
        gyro_noise_std = gyro_noise_density * np.sqrt(fs)

        rng = np.random.default_rng(seed=42)
        f_imu = f + acc_noise_std * rng.standard_normal(f.shape)
        w_imu = w + gyro_noise_std * rng.standard_normal(w.shape)
        head_aid = head + np.radians(1.0) * rng.standard_normal(head.shape)

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(
            fs,
            q0,
            acc_noise_density=acc_noise_density,
            gyro_noise_density=gyro_noise_density,
        )

        euler_out = []
        for f_i, w_i, h_i in zip(f_imu, w_imu, head_aid):
            ahrs.update(f_i, w_i, degrees=False)
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_out[warmup:, :2], euler[warmup:, :2], atol=0.002
        )
