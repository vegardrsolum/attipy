import numpy as np
import pytest
from pytest import fixture

from attipy import AHRS, Attitude
from attipy._transforms import _quat_from_euler_zyx


class Test_AHRS:

    @fixture
    def ahrs(self):
        return AHRS(10.0)

    def test__init__(self):
        fs = 1024.0
        q = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        bg = (0.1, -0.2, 0.3)
        v = (1.0, -2.0, 3.0)
        a = (1.0, 2.0, 3.0)
        w = (0.01, -0.02, 0.03)
        P = 42.0 * np.eye(9)
        g = 9.83
        nav_frame = "enu"
        acc_noise_density = 0.00123
        gyro_noise_density = 0.000456
        gyro_bias_stability = 0.0000789
        bias_corr_time = 123.0

        ahrs = AHRS(
            fs,
            q=q,
            bg=bg,
            v=v,
            w=w,
            a=a,
            P=P,
            g=g,
            nav_frame=nav_frame,
            acc_noise_density=acc_noise_density,
            gyro_noise_density=gyro_noise_density,
            gyro_bias_stability=gyro_bias_stability,
            bias_corr_time=bias_corr_time,
        )

        assert ahrs._fs == fs
        assert ahrs._dt == 1.0 / fs
        assert ahrs._nav_frame == "enu"
        assert ahrs._g == g
        np.testing.assert_allclose(ahrs._g_n, np.array([0.0, 0.0, -g]))

        assert ahrs._vrw == acc_noise_density
        assert ahrs._arw == gyro_noise_density
        assert ahrs._gbs == gyro_bias_stability
        assert ahrs._gbc == bias_corr_time

        np.testing.assert_allclose(ahrs._att_nb._q, q)
        np.testing.assert_allclose(ahrs._bg_b, bg)
        np.testing.assert_allclose(ahrs._v_n, v)
        np.testing.assert_allclose(ahrs._w_b, w)
        np.testing.assert_allclose(ahrs._a_n, a)
        np.testing.assert_allclose(ahrs._f_b, ahrs._R_nb.T @ (ahrs._a_n - ahrs._g_n))
        np.testing.assert_allclose(ahrs._P, P)

    def test__init__default(self):
        fs = 10.0
        ahrs = AHRS(fs)

        assert ahrs._fs == fs
        assert ahrs._dt == 1.0 / fs
        assert ahrs._nav_frame == "ned"
        assert ahrs._g == 9.80665
        np.testing.assert_allclose(ahrs._g_n, np.array([0.0, 0.0, 9.80665]))

        assert ahrs._vrw == 0.001
        assert ahrs._arw == 0.0001
        assert ahrs._gbs == 0.00005
        assert ahrs._gbc == 50.0

        np.testing.assert_allclose(ahrs._att_nb._q, np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(ahrs._bg_b, np.zeros(3))
        np.testing.assert_allclose(ahrs._v_n, np.zeros(3))
        np.testing.assert_allclose(ahrs._P, 1e-6 * np.eye(9))

        np.testing.assert_allclose(ahrs._f_b, np.array([0.0, 0.0, -9.80665]))
        np.testing.assert_allclose(ahrs._w_b, np.zeros(3))

    def test__init__nav_frame(self):
        ahrs_ned = AHRS(10.0, nav_frame="NED")
        np.testing.assert_allclose(ahrs_ned._g_n, np.array([0.0, 0.0, 9.80665]))

        ahrs_enu = AHRS(10.0, nav_frame="ENU")
        np.testing.assert_allclose(ahrs_enu._g_n, np.array([0.0, 0.0, -9.80665]))

        with pytest.raises(ValueError):
            AHRS(10.0, nav_frame="invalid")

    def test_attitude(self, ahrs):
        q_expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(ahrs.attitude, Attitude)
        np.testing.assert_allclose(ahrs.attitude.as_quaternion(), q_expected)

    def test_q(self):
        q = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        ahrs = AHRS(10.0, q=q)
        np.testing.assert_allclose(ahrs.q, q)

    def test_v(self):
        v = np.array([1.0, 2.0, 3.0])
        ahrs = AHRS(10.0, v=v)
        np.testing.assert_allclose(ahrs.v, v)

    def test_bg(self):
        ahrs = AHRS(10.0, bg=np.array([0.01, -0.02, 0.03]))
        bg_expected = np.array([0.01, -0.02, 0.03])
        np.testing.assert_allclose(ahrs.bg, bg_expected)

    def test_w(self):
        w = np.array([0.1, -0.2, 0.3])
        ahrs = AHRS(10.0, w=w)
        np.testing.assert_allclose(ahrs.w, w)

    def test_a(self):
        a = np.array([1.0, 2.0, 3.0])
        ahrs = AHRS(10.0, a=a)
        np.testing.assert_allclose(ahrs.a, a)

    def test_P(self, ahrs):
        ahrs = AHRS(10.0, P=np.eye(9))
        np.testing.assert_allclose(ahrs.P, np.eye(9))

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
        vel_meas = vel + 1.0 * rng.standard_normal(vel.shape)

        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i, v_i in zip(f_imu, w_imu, vel_meas):
            ahrs.update(f_i, w_i, degrees=False, v=v_i, v_var=1.0**2 * np.ones(3))
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_out[warmup:, :2], euler[warmup:, :2], atol=0.005
        )

    def test_update_yaw_aiding(self, pva_data):
        _, _, _, euler, f, w = pva_data
        yaw = euler[:, 2]
        fs = 10.24

        acc_noise_density = 0.001
        gyro_noise_density = 0.0001
        acc_noise_std = acc_noise_density * np.sqrt(fs)
        gyro_noise_std = gyro_noise_density * np.sqrt(fs)

        rng = np.random.default_rng(seed=42)
        bg = np.radians([0.1, -0.2, 0.3])
        f_imu = f + acc_noise_std * rng.standard_normal(f.shape)
        w_imu = w + gyro_noise_std * rng.standard_normal(w.shape) + bg
        yaw_meas = yaw + np.radians(1.0) * rng.standard_normal(yaw.shape)

        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i, yaw_i in zip(f_imu, w_imu, yaw_meas):
            ahrs.update(
                f_i,
                w_i,
                degrees=False,
                yaw=yaw_i,
                yaw_var=np.radians(1.0) ** 2,
                yaw_degrees=False,
            )
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(euler_out[warmup:], euler[warmup:], atol=0.005)

    def test_update_full_aiding(self, pva_data):
        _, _, vel, euler, f, w = pva_data
        yaw = euler[:, 2]
        fs = 10.24

        acc_noise_density = 0.001
        gyro_noise_density = 0.0001
        acc_noise_std = acc_noise_density * np.sqrt(fs)
        gyro_noise_std = gyro_noise_density * np.sqrt(fs)

        rng = np.random.default_rng(seed=42)
        bg = np.radians([0.1, -0.2, 0.3])
        f_imu = f + acc_noise_std * rng.standard_normal(f.shape)
        w_imu = w + gyro_noise_std * rng.standard_normal(w.shape) + bg
        yaw_meas = yaw + np.radians(1.0) * rng.standard_normal(yaw.shape)
        vel_meas = vel + 1.0 * rng.standard_normal(vel.shape)

        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i, yaw_i, v_i in zip(f_imu, w_imu, yaw_meas, vel_meas):
            ahrs.update(
                f_i,
                w_i,
                degrees=False,
                v=v_i,
                v_var=1.0**2 * np.ones(3),
                yaw=yaw_i,
                yaw_var=np.radians(1.0) ** 2,
                yaw_degrees=False,
            )
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(euler_out[warmup:], euler[warmup:], atol=0.005)

    def test_recover_state(self, pva_data):
        _, _, _, euler, f, w = pva_data
        f, w = f[:10], w[:10]

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs_a = AHRS(fs, q0)

        q_a, q_b = [], []
        bg_a, bg_b = [], []
        v_a, v_b = [], []
        w_a, w_b = [], []
        a_a, a_b = [], []
        P_a, P_b = [], []
        for f_i, w_i in zip(f, w):
            ahrs_b = AHRS(
                fs,
                q=ahrs_a.q,
                bg=ahrs_a.bg,
                v=ahrs_a.v,
                w=ahrs_a.w,
                a=ahrs_a.a,
                P=ahrs_a.P,
                g=ahrs_a._g,
                nav_frame=ahrs_a._nav_frame,
                acc_noise_density=ahrs_a._vrw,
                gyro_noise_density=ahrs_a._arw,
                gyro_bias_stability=ahrs_a._gbs,
                bias_corr_time=ahrs_a._gbc,
            )

            ahrs_a.update(f_i, w_i, degrees=False)
            ahrs_b.update(f_i, w_i, degrees=False)

            q_a.append(ahrs_a.q)
            q_b.append(ahrs_b.q)
            bg_a.append(ahrs_a.bg)
            bg_b.append(ahrs_b.bg)
            v_a.append(ahrs_a.v)
            v_b.append(ahrs_b.v)
            w_a.append(ahrs_a.w)
            w_b.append(ahrs_b.w)
            a_a.append(ahrs_a.a)
            a_b.append(ahrs_b.a)
            P_a.append(ahrs_a.P)
            P_b.append(ahrs_b.P)

        np.testing.assert_allclose(q_a, q_b)
        np.testing.assert_allclose(bg_a, bg_b)
        np.testing.assert_allclose(v_a, v_b)
        np.testing.assert_allclose(w_a, w_b)
        np.testing.assert_allclose(a_a, a_b)
        np.testing.assert_allclose(P_a, P_b)
