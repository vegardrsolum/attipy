import numpy as np
import pytest
from pytest import fixture

import attipy as ap
from attipy._mekf import _dyawda
from attipy._transforms import _quat_from_euler_zyx


class Test_MEKF:

    @fixture
    def mekf(self):
        return ap.MEKF(10.0)

    def test__init__(self):
        fs = 1024.0
        q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        bg_b = (0.1, -0.2, 0.3)
        v_n = (1.0, -2.0, 3.0)
        a_n = (1.0, 2.0, 3.0)
        w_b = (0.01, -0.02, 0.03)
        P = 42.0 * np.eye(9)
        g = 9.83
        nav_frame = "enu"
        acc_noise_density = 0.00123
        gyro_noise_density = 0.000456
        gyro_bias_stability = 0.0000789
        bias_corr_time = 123.0

        mekf = ap.MEKF(
            fs,
            q_nb=q_nb,
            bg_b=bg_b,
            v_n=v_n,
            w_b=w_b,
            a_n=a_n,
            P=P,
            g=g,
            nav_frame=nav_frame,
            acc_noise_density=acc_noise_density,
            gyro_noise_density=gyro_noise_density,
            gyro_bias_stability=gyro_bias_stability,
            bias_corr_time=bias_corr_time,
        )

        assert mekf._fs == fs
        assert mekf._dt == 1.0 / fs
        assert mekf._nav_frame == "enu"
        assert mekf._g == g
        np.testing.assert_allclose(mekf._g_n, np.array([0.0, 0.0, -g]))

        assert mekf._vrw == acc_noise_density
        assert mekf._arw == gyro_noise_density
        assert mekf._gbs == gyro_bias_stability
        assert mekf._gbc == bias_corr_time

        np.testing.assert_allclose(mekf._att_nb._q, q_nb)
        np.testing.assert_allclose(mekf._bg_b, bg_b)
        np.testing.assert_allclose(mekf._v_n, v_n)
        np.testing.assert_allclose(mekf._w_b, w_b)
        np.testing.assert_allclose(mekf._a_n, a_n)
        np.testing.assert_allclose(mekf._f_b, mekf._R_nb.T @ (mekf._a_n - mekf._g_n))
        np.testing.assert_allclose(mekf._P, P)

        # Check C contiguity
        assert mekf._dhdx.flags.c_contiguous
        assert mekf._phi.flags.c_contiguous
        assert mekf._Q.flags.c_contiguous

    def test__init__default(self):
        fs = 10.0
        mekf = ap.MEKF(fs)

        assert mekf._fs == fs
        assert mekf._dt == 1.0 / fs
        assert mekf._nav_frame == "ned"
        assert mekf._g == 9.80665
        np.testing.assert_allclose(mekf._g_n, np.array([0.0, 0.0, 9.80665]))

        assert mekf._vrw == 0.001
        assert mekf._arw == 0.0001
        assert mekf._gbs == 0.00005
        assert mekf._gbc == 50.0

        np.testing.assert_allclose(mekf._att_nb._q, np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(mekf._bg_b, np.zeros(3))
        np.testing.assert_allclose(mekf._v_n, np.zeros(3))
        np.testing.assert_allclose(mekf._P, 1e-6 * np.eye(9))

        np.testing.assert_allclose(mekf._f_b, np.array([0.0, 0.0, -9.80665]))
        np.testing.assert_allclose(mekf._w_b, np.zeros(3))

    def test_dhdx_vel(self, mekf):
        dhdx_vel = mekf._dhdx_vel()
        dhdx_vel_expected = np.zeros((3, 9))
        dhdx_vel_expected[:, 3:6] = np.eye(3)
        np.testing.assert_allclose(dhdx_vel, dhdx_vel_expected)
        assert dhdx_vel.flags.c_contiguous

    def test_dhdx_yaw(self, mekf):
        q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        dhdx_yaw = mekf._dhdx_yaw(q_nb)
        dhdx_yaw_expected = np.zeros((9,))
        dhdx_yaw_expected[0:3] = _dyawda(q_nb)
        np.testing.assert_allclose(dhdx_yaw, dhdx_yaw_expected)
        assert dhdx_yaw.flags.c_contiguous

    def test__init__nav_frame(self):
        mekf_ned = ap.MEKF(10.0, nav_frame="NED")
        np.testing.assert_allclose(mekf_ned._g_n, np.array([0.0, 0.0, 9.80665]))

        mekf_enu = ap.MEKF(10.0, nav_frame="ENU")
        np.testing.assert_allclose(mekf_enu._g_n, np.array([0.0, 0.0, -9.80665]))

        with pytest.raises(ValueError):
            ap.MEKF(10.0, nav_frame="invalid")

    def test_attitude(self, mekf):
        q_expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(mekf.attitude, ap.Attitude)
        np.testing.assert_allclose(mekf.attitude.as_quaternion(), q_expected)

    def test_q_nb(self):
        q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        mekf = ap.MEKF(10.0, q_nb=q_nb)
        np.testing.assert_allclose(mekf.q_nb, q_nb)
        assert mekf.q_nb is not mekf._att_nb._q  # ensure it is a copy

    def test_v_n(self):
        v_n = np.array([1.0, 2.0, 3.0])
        mekf = ap.MEKF(10.0, v_n=v_n)
        np.testing.assert_allclose(mekf.v_n, v_n)
        assert mekf.v_n is not mekf._v_n  # ensure it is a copy

    def test_bg_b(self):
        mekf = ap.MEKF(10.0, bg_b=np.array([0.01, -0.02, 0.03]))
        bg_b_expected = np.array([0.01, -0.02, 0.03])
        np.testing.assert_allclose(mekf.bg_b, bg_b_expected)
        assert mekf.bg_b is not mekf._bg_b  # ensure it is a copy

    def test_ba_b(self):
        mekf = ap.MEKF(10.0, ba_b=np.array([1.0, -2.3, 3.4]))
        ba_b_expected = np.array([1.0, -2.3, 3.4])
        np.testing.assert_allclose(mekf.ba_b, ba_b_expected)
        assert mekf.ba_b is not mekf._ba_b  # ensure it is a copy

    def test_w_b(self):
        w_b = np.array([0.1, -0.2, 0.3])
        mekf = ap.MEKF(10.0, w_b=w_b)
        np.testing.assert_allclose(mekf.w_b, w_b)
        assert mekf.w_b is not mekf._w_b  # ensure it is a copy

    def test_f_b(self):
        q_nb = (1.0, 0.0, 0.0, 0.0)  # no rotation
        mekf = ap.MEKF(10.0, q_nb=q_nb, a_n=np.zeros(3), g=9.80665, nav_frame="ned")
        np.testing.assert_allclose(mekf.f_b, np.array([0.0, 0.0, -9.80665]))
        assert mekf.f_b is not mekf._f_b  # ensure it is a copy

    def test_a_n(self):
        a_n = np.array([1.0, 2.0, 3.0])
        mekf = ap.MEKF(10.0, a_n=a_n)
        np.testing.assert_allclose(mekf.a_n, a_n)
        assert mekf.a_n is not mekf._a_n  # ensure it is a copy

    def test_P(self, mekf):
        mekf = ap.MEKF(10.0, P=np.eye(9))
        np.testing.assert_allclose(mekf.P, np.eye(9))
        assert mekf.P is not mekf._P  # ensure it is a copy

    def test_update(self, pva_sim):
        _, _, _, euler_nb, f_b, w_b = pva_sim
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg_b = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
        w_meas = (
            w_b
            + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
            + bg_b
        )

        # Estimate attitude using MEKF
        mekf = ap.MEKF(fs)
        euler_est, bg_est = [], []
        for f_i, w_i in zip(f_meas, w_meas):
            mekf.update(f_i, w_i)
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bg_b)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Truncate 600 seconds from the beginning (so that filter has converged)
        # Check roll and pitch only
        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        euler_expect = euler_nb[warmup:, :2]
        bg_expect = np.full(bg_est.shape, bg_b)[warmup:, :2]
        euler_out = euler_est[warmup:, :2]
        bg_out = bg_est[warmup:, :2]

        np.testing.assert_allclose(euler_out, euler_expect, atol=0.006)
        np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)

    def test_update_full_aiding(self, pva_sim):
        _, _, v_n, euler_nb, f_b, w_b = pva_sim
        yaw = euler_nb[:, 2]
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg_b = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
        w_meas = (
            w_b
            + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
            + bg_b
        )

        # Add velocity and heading measurement noise
        v_var = 0.01  # (m/s)^2
        yaw_var = 0.0001  # rad^2
        rng = np.random.default_rng(42)
        v_meas = v_n + np.sqrt(v_var) * rng.standard_normal(v_n.shape)
        yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

        # Estimate attitude using MEKF
        mekf = ap.MEKF(fs)
        euler_est, bg_est, v_est = [], [], []
        for f_i, w_i, v_i, y_i in zip(f_meas, w_meas, v_meas, yaw_meas):
            mekf.update(
                f_i, w_i, v_n=v_i, v_var=v_var * np.ones(3), yaw=y_i, yaw_var=yaw_var
            )
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bg_b)
            v_est.append(mekf.v_n)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)
        v_est = np.asarray(v_est)

        # Truncate 600 seconds from the beginning (so that filter has converged)
        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        euler_expect = euler_nb[warmup:]
        bg_expect = np.full(bg_est.shape, bg_b)[warmup:]
        v_expect = v_n[warmup:]
        euler_out = euler_est[warmup:]
        bg_out = bg_est[warmup:]
        v_out = v_est[warmup:]

        np.testing.assert_allclose(euler_out, euler_expect, atol=0.006)
        np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)
        np.testing.assert_allclose(v_out, v_expect, atol=2.0 * np.sqrt(v_var))

    def test_update_vel_aiding(self, pva_sim):
        _, _, v_n, euler_nb, f_b, w_b = pva_sim
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg_b = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
        w_meas = (
            w_b
            + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
            + bg_b
        )

        # Add velocity and heading measurement noise
        v_var = 0.01  # (m/s)^2
        rng = np.random.default_rng(42)
        v_meas = v_n + np.sqrt(v_var) * rng.standard_normal(v_n.shape)

        # Estimate attitude using MEKF
        mekf = ap.MEKF(fs)
        euler_est, bg_est = [], []
        for f_i, w_i, v_i in zip(f_meas, w_meas, v_meas):
            mekf.update(f_i, w_i, v_n=v_i, v_var=v_var * np.ones(3))
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bg_b)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Truncate 600 seconds from the beginning (so that filter has converged)
        # Check roll and pitch only
        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        euler_expect = euler_nb[warmup:, :2]
        bg_expect = np.full(bg_est.shape, bg_b)[warmup:, :2]
        euler_out = euler_est[warmup:, :2]
        bg_out = bg_est[warmup:, :2]

        np.testing.assert_allclose(euler_out, euler_expect, atol=0.006)
        np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)

    def test_update_yaw_aiding(self, pva_sim):
        _, _, _, euler_nb, f_b, w_b = pva_sim
        yaw = euler_nb[:, 2]
        fs = 10.24

        # Add IMU measurement noise
        acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
        gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
        bg_b = (0.001, 0.002, 0.003)  # rad/s
        rng = np.random.default_rng(42)
        f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
        w_meas = (
            w_b
            + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
            + bg_b
        )

        # Add velocity and heading measurement noise
        yaw_var = 0.0001  # rad^2
        rng = np.random.default_rng(42)
        yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

        # Estimate attitude using MEKF
        mekf = ap.MEKF(fs)
        euler_est, bg_est = [], []
        for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
            mekf.update(f_i, w_i, yaw=y_i, yaw_var=yaw_var)
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bg_b)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Truncate 600 seconds from the beginning (so that filter has converged)
        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        euler_expect = euler_nb[warmup:]
        bg_expect = np.full(bg_est.shape, bg_b)[warmup:]
        euler_out = euler_est[warmup:]
        bg_out = bg_est[warmup:]

        np.testing.assert_allclose(euler_out, euler_expect, atol=0.007)
        np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)

    def test_recover_state(self, pva_sim):
        _, _, _, euler_nb, f_b, w_b = pva_sim
        f_b, w_b = f_b[:10], w_b[:10]

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler_nb[0])
        mekf_a = ap.MEKF(fs, q0)

        q_a, q_b = [], []
        bg_a, bg_b = [], []
        v_a, v_b = [], []
        w_a, w_b = [], []
        a_a, a_b = [], []
        P_a, P_b = [], []
        for f_i, w_i in zip(f_b, w_b):
            mekf_b = ap.MEKF(
                fs,
                q_nb=mekf_a.q_nb,
                bg_b=mekf_a.bg_b,
                v_n=mekf_a.v_n,
                w_b=mekf_a.w_b,
                a_n=mekf_a.a_n,
                P=mekf_a.P,
                g=mekf_a._g,
                nav_frame=mekf_a._nav_frame,
                acc_noise_density=mekf_a._vrw,
                gyro_noise_density=mekf_a._arw,
                gyro_bias_stability=mekf_a._gbs,
                bias_corr_time=mekf_a._gbc,
            )

            mekf_a.update(f_i, w_i, degrees=False)
            mekf_b.update(f_i, w_i, degrees=False)

            q_a.append(mekf_a.q_nb)
            q_b.append(mekf_b.q_nb)
            bg_a.append(mekf_a.bg_b)
            bg_b.append(mekf_b.bg_b)
            v_a.append(mekf_a.v_n)
            v_b.append(mekf_b.v_n)
            w_a.append(mekf_a.w_b)
            w_b.append(mekf_b.w_b)
            a_a.append(mekf_a.a_n)
            a_b.append(mekf_b.a_n)
            P_a.append(mekf_a.P)
            P_b.append(mekf_b.P)

        np.testing.assert_allclose(q_a, q_b)
        np.testing.assert_allclose(bg_a, bg_b)
        np.testing.assert_allclose(v_a, v_b)
        np.testing.assert_allclose(w_a, w_b)
        np.testing.assert_allclose(a_a, a_b)
        np.testing.assert_allclose(P_a, P_b)
