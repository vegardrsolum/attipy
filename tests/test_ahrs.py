import numpy as np
import pytest
from pytest import fixture

import attipy as ap
from attipy._transforms import _quat_from_euler_zyx


class Test_AHRS:

    @fixture
    def ahrs(self):
        return ap.AHRS(10.0)

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

        ahrs = ap.AHRS(
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
        ahrs = ap.AHRS(fs)

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
        ahrs_ned = ap.AHRS(10.0, nav_frame="NED")
        np.testing.assert_allclose(ahrs_ned._g_n, np.array([0.0, 0.0, 9.80665]))

        ahrs_enu = ap.AHRS(10.0, nav_frame="ENU")
        np.testing.assert_allclose(ahrs_enu._g_n, np.array([0.0, 0.0, -9.80665]))

        with pytest.raises(ValueError):
            ap.AHRS(10.0, nav_frame="invalid")

    def test_attitude(self, ahrs):
        q_expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(ahrs.attitude, ap.Attitude)
        np.testing.assert_allclose(ahrs.attitude.as_quaternion(), q_expected)

    def test_q(self):
        q = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        ahrs = ap.AHRS(10.0, q=q)
        np.testing.assert_allclose(ahrs.q, q)

    def test_v(self):
        v = np.array([1.0, 2.0, 3.0])
        ahrs = ap.AHRS(10.0, v=v)
        np.testing.assert_allclose(ahrs.v, v)

    def test_bg(self):
        ahrs = ap.AHRS(10.0, bg=np.array([0.01, -0.02, 0.03]))
        bg_expected = np.array([0.01, -0.02, 0.03])
        np.testing.assert_allclose(ahrs.bg, bg_expected)

    def test_w(self):
        w = np.array([0.1, -0.2, 0.3])
        ahrs = ap.AHRS(10.0, w=w)
        np.testing.assert_allclose(ahrs.w, w)

    def test_a(self):
        a = np.array([1.0, 2.0, 3.0])
        ahrs = ap.AHRS(10.0, a=a)
        np.testing.assert_allclose(ahrs.a, a)

    def test_P(self, ahrs):
        ahrs = ap.AHRS(10.0, P=np.eye(9))
        np.testing.assert_allclose(ahrs.P, np.eye(9))

    def test_update(self, pva_data):
        _, _, _, euler, f, w = pva_data
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f.shape)
        w_meas = (
            w + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w.shape) + bg
        )

        # Estimate attitude using AHRS
        ahrs = ap.AHRS(fs)
        euler_est, bg_est = [], []
        for f_i, w_i in zip(f_meas, w_meas):
            ahrs.update(f_i, w_i)
            euler_est.append(ahrs.attitude.as_euler())
            bg_est.append(ahrs.bg)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        euler_expect = euler
        bg_expect = np.full(bg_est.shape, bg)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_est[warmup:, :2], euler_expect[warmup:, :2], atol=0.006
        )
        np.testing.assert_allclose(
            bg_est[warmup:, :2], bg_expect[warmup:, :2], atol=0.005
        )

    def test_update_vel_aiding(self, pva_data):
        _, _, vel, euler, f, w = pva_data
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f.shape)
        w_meas = (
            w + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w.shape) + bg
        )

        # Add velocity and heading measurement noise
        vel_var = 0.01  # (m/s)^2
        rng = np.random.default_rng(42)
        vel_meas = vel + np.sqrt(vel_var) * rng.standard_normal(vel.shape)

        # Estimate attitude using AHRS
        ahrs = ap.AHRS(fs)
        euler_est, bg_est = [], []
        for f_i, w_i, v_i in zip(f_meas, w_meas, vel_meas):
            ahrs.update(f_i, w_i, v=v_i, v_var=vel_var * np.ones(3))
            euler_est.append(ahrs.attitude.as_euler())
            bg_est.append(ahrs.bg)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        euler_expect = euler
        bg_expect = np.full(bg_est.shape, bg)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_est[warmup:, :2], euler_expect[warmup:, :2], atol=0.006
        )
        np.testing.assert_allclose(
            bg_est[warmup:, :2], bg_expect[warmup:, :2], atol=0.005
        )

    def test_update_yaw_aiding(self, pva_data):
        _, _, _, euler, f, w = pva_data
        yaw = euler[:, 2]
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f.shape)
        w_meas = (
            w + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w.shape) + bg
        )

        # Add velocity and heading measurement noise
        yaw_var = 0.0003  # rad^2
        rng = np.random.default_rng(42)
        yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

        # Estimate attitude using AHRS
        ahrs = ap.AHRS(fs)
        euler_est, bg_est = [], []
        for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
            ahrs.update(f_i, w_i, yaw=y_i, yaw_var=yaw_var)
            euler_est.append(ahrs.attitude.as_euler())
            bg_est.append(ahrs.bg)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        euler_expect = euler
        bg_expect = np.full(bg_est.shape, bg)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_est[warmup:], euler_expect[warmup:], atol=0.006
        )
        np.testing.assert_allclose(bg_est[warmup:], bg_expect[warmup:], atol=0.005)

    def test_update_full_aiding(self, pva_data):
        _, _, vel, euler, f, w = pva_data
        yaw = euler[:, 2]
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f.shape)
        w_meas = (
            w + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w.shape) + bg
        )

        # Add velocity and heading measurement noise
        vel_var = 0.01  # (m/s)^2
        yaw_var = 0.0003  # rad^2
        rng = np.random.default_rng(42)
        vel_meas = vel + np.sqrt(vel_var) * rng.standard_normal(vel.shape)
        yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

        # Estimate attitude using AHRS
        ahrs = ap.AHRS(fs)
        euler_est, bg_est = [], []
        for f_i, w_i, v_i, y_i in zip(f_meas, w_meas, vel_meas, yaw_meas):
            ahrs.update(
                f_i, w_i, v=v_i, v_var=vel_var * np.ones(3), yaw=y_i, yaw_var=yaw_var
            )
            euler_est.append(ahrs.attitude.as_euler())
            bg_est.append(ahrs.bg)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        euler_expect = euler
        bg_expect = np.full(bg_est.shape, bg)

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        np.testing.assert_allclose(
            euler_est[warmup:], euler_expect[warmup:], atol=0.006
        )
        np.testing.assert_allclose(bg_est[warmup:], bg_expect[warmup:], atol=0.005)

    def test_recover_state(self, pva_data):
        _, _, _, euler, f, w = pva_data
        f, w = f[:10], w[:10]

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs_a = ap.AHRS(fs, q0)

        q_a, q_b = [], []
        bg_a, bg_b = [], []
        v_a, v_b = [], []
        w_a, w_b = [], []
        a_a, a_b = [], []
        P_a, P_b = [], []
        for f_i, w_i in zip(f, w):
            ahrs_b = ap.AHRS(
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
