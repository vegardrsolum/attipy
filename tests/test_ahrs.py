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
        fs = 10.24

        acc_noise_density = 0.001
        gyro_noise_density = 0.0001
        acc_noise_std = acc_noise_density * np.sqrt(fs)
        gyro_noise_std = gyro_noise_density * np.sqrt(fs)

        rng = np.random.default_rng(seed=42)
        bg = np.radians([0.1, -0.2, 0.3])
        f_imu = f + acc_noise_std * rng.standard_normal(f.shape)
        w_imu = w + gyro_noise_std * rng.standard_normal(w.shape) + bg

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i in zip(f_imu, w_imu):
            ahrs.update(f_i, w_i, degrees=False)
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_out[warmup:, :2], euler[warmup:, :2], atol=0.005
        )

    def test_update_vel_aiding(self, pva_data):
        _, _, vel, euler, f, w = pva_data
        fs = 10.24

        acc_noise_density = 0.001
        gyro_noise_density = 0.0001
        acc_noise_std = acc_noise_density * np.sqrt(fs)
        gyro_noise_std = gyro_noise_density * np.sqrt(fs)

        rng = np.random.default_rng(seed=42)
        bg = np.radians([0.1, -0.2, 0.3])
        f_imu = f + acc_noise_std * rng.standard_normal(f.shape)
        w_imu = w + gyro_noise_std * rng.standard_normal(w.shape) + bg
        vel_meas = vel + 0.1 * rng.standard_normal(vel.shape)

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i, v_i in zip(f_imu, w_imu, vel_meas):
            ahrs.update(f_i, w_i, degrees=False, vel=v_i, vel_var=0.1**2 * np.ones(3))
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_out[warmup:, :2], euler[warmup:, :2], atol=0.005
        )

    def test_update_hdg_aiding(self, pva_data):
        _, _, _, euler, f, w = pva_data
        hdg = euler[:, 2]
        fs = 10.24

        acc_noise_density = 0.001
        gyro_noise_density = 0.0001
        acc_noise_std = acc_noise_density * np.sqrt(fs)
        gyro_noise_std = gyro_noise_density * np.sqrt(fs)

        rng = np.random.default_rng(seed=42)
        bg = np.radians([0.1, -0.2, 0.3])
        f_imu = f + acc_noise_std * rng.standard_normal(f.shape)
        w_imu = w + gyro_noise_std * rng.standard_normal(w.shape) + bg
        hdg_meas = hdg + np.radians(1.0) * rng.standard_normal(hdg.shape)

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i, h_i in zip(f_imu, w_imu, hdg_meas):
            ahrs.update(
                f_i,
                w_i,
                degrees=False,
                hdg=h_i,
                hdg_var=np.radians(1.0) ** 2,
                hdg_degrees=False,
            )
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(euler_out[warmup:], euler[warmup:], atol=0.005)

    def test_update_full_aiding(self, pva_data):
        _, _, vel, euler, f, w = pva_data
        hdg = euler[:, 2]
        fs = 10.24

        acc_noise_density = 0.001
        gyro_noise_density = 0.0001
        acc_noise_std = acc_noise_density * np.sqrt(fs)
        gyro_noise_std = gyro_noise_density * np.sqrt(fs)

        rng = np.random.default_rng(seed=42)
        bg = np.radians([0.1, -0.2, 0.3])
        f_imu = f + acc_noise_std * rng.standard_normal(f.shape)
        w_imu = w + gyro_noise_std * rng.standard_normal(w.shape) + bg
        hdg_meas = hdg + np.radians(1.0) * rng.standard_normal(hdg.shape)
        vel_meas = vel + 0.1 * rng.standard_normal(vel.shape)

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i, h_i, v_i in zip(f_imu, w_imu, hdg_meas, vel_meas):
            ahrs.update(
                f_i,
                w_i,
                degrees=False,
                vel=v_i,
                vel_var=0.1**2 * np.ones(3),
                hdg=h_i,
                hdg_var=np.radians(1.0) ** 2,
                hdg_degrees=False,
            )
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(euler_out[warmup:], euler[warmup:], atol=0.005)
