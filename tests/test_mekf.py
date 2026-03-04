import numpy as np
import pytest
from pytest import fixture

import attipy as ap
from attipy import _statespace
from attipy._mekf import _dyawda
from attipy._transforms import _quat_from_euler_zyx
from attipy._vectorops import _skew_symmetric

# class Test_MEKF:

#     @fixture
#     def att(self):
#         return ap.Attitude((1.0, 0.0, 0.0, 0.0))

#     @fixture
#     def mekf(self):
#         return ap.MEKF(10.0, (1.0, 0.0, 0.0, 0.0))

#     def test__init__(self):
#         fs = 1024.0
#         q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
#         bg_b = (0.1, -0.2, 0.3)
#         v_n = (1.0, -2.0, 3.0)
#         a_n = (1.0, 2.0, 3.0)
#         w_b = (0.01, -0.02, 0.03)
#         P = 42.0 * np.eye(15)
#         g = 9.83
#         nav_frame = "enu"
#         acc_noise_density = 0.00123
#         gyro_noise_density = 0.000456
#         gyro_bias_stability = 0.0000789
#         gyro_bias_corr_time = 123.0

#         mekf = ap.MEKF(
#             fs,
#             q_nb,
#             bg=bg_b,
#             vel=v_n,
#             w=w_b,
#             acc=a_n,
#             P=P,
#             g=g,
#             nav_frame=nav_frame,
#             acc_noise_density=acc_noise_density,
#             gyro_noise_density=gyro_noise_density,
#             gyro_bias_stability=gyro_bias_stability,
#             gyro_bias_corr_time=gyro_bias_corr_time,
#         )

#         assert mekf._fs == fs
#         assert mekf._dt == 1.0 / fs
#         assert mekf._nav_frame == "enu"
#         assert mekf._g == g
#         np.testing.assert_allclose(mekf._g_n, np.array([0.0, 0.0, -g]))

#         assert mekf._vrw == acc_noise_density
#         assert mekf._arw == gyro_noise_density
#         assert mekf._gbs == gyro_bias_stability
#         assert mekf._gbc == gyro_bias_corr_time

#         np.testing.assert_allclose(mekf._att_nb._q, q_nb)
#         np.testing.assert_allclose(mekf._bg_b, bg_b)
#         np.testing.assert_allclose(mekf._v_n, v_n)
#         np.testing.assert_allclose(mekf._w_b, w_b)
#         np.testing.assert_allclose(mekf._a_n, a_n)
#         np.testing.assert_allclose(mekf._f_b, mekf._R_nb.T @ (mekf._a_n - mekf._g_n))
#         np.testing.assert_allclose(mekf._P, P)

#         # Check C contiguity
#         assert mekf._dhdx.flags.c_contiguous
#         assert mekf._phi.flags.c_contiguous
#         assert mekf._Q.flags.c_contiguous

#     def test__init__default(self):
#         fs = 10.0
#         mekf = ap.MEKF(fs, (1.0, 0.0, 0.0, 0.0))

#         assert mekf._fs == fs
#         assert mekf._dt == 1.0 / fs
#         assert mekf._nav_frame == "ned"
#         assert mekf._g == 9.80665
#         np.testing.assert_allclose(mekf._g_n, np.array([0.0, 0.0, 9.80665]))

#         assert mekf._vrw == 0.001
#         assert mekf._arw == 0.0001
#         assert mekf._gbs == 0.00005
#         assert mekf._gbc == 50.0

#         np.testing.assert_allclose(mekf._att_nb._q, np.array([1.0, 0.0, 0.0, 0.0]))
#         np.testing.assert_allclose(mekf._bg_b, np.zeros(3))
#         np.testing.assert_allclose(mekf._v_n, np.zeros(3))
#         np.testing.assert_allclose(mekf._P, 1e-6 * np.eye(15))

#         np.testing.assert_allclose(mekf._f_b, np.array([0.0, 0.0, -9.80665]))
#         np.testing.assert_allclose(mekf._w_b, np.zeros(3))

#     def test_dhdx_vel(self, mekf):
#         dhdx_vel = mekf._dhdx_vel()
#         dhdx_vel_expected = np.zeros((3, 15))
#         dhdx_vel_expected[:, 3:6] = np.eye(3)
#         np.testing.assert_allclose(dhdx_vel, dhdx_vel_expected)
#         assert dhdx_vel.flags.c_contiguous

#     def test_dhdx_yaw(self, mekf):
#         q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
#         dhdx_yaw = mekf._dhdx_yaw(q_nb)
#         dhdx_yaw_expected = np.zeros((15,))
#         dhdx_yaw_expected[6:9] = _dyawda(q_nb)
#         np.testing.assert_allclose(dhdx_yaw, dhdx_yaw_expected)
#         assert dhdx_yaw.flags.c_contiguous

#     def test__init__nav_frame(self):
#         att = ap.Attitude((1.0, 0.0, 0.0, 0.0))

#         mekf_ned = ap.MEKF(10.0, att, nav_frame="NED")
#         np.testing.assert_allclose(mekf_ned._g_n, np.array([0.0, 0.0, 9.80665]))

#         mekf_enu = ap.MEKF(10.0, att, nav_frame="ENU")
#         np.testing.assert_allclose(mekf_enu._g_n, np.array([0.0, 0.0, -9.80665]))

#         with pytest.raises(ValueError):
#             ap.MEKF(10.0, att, nav_frame="invalid")

#     def test_attitude(self, mekf):
#         q_expected = np.array([1.0, 0.0, 0.0, 0.0])
#         assert isinstance(mekf.attitude, ap.Attitude)
#         np.testing.assert_allclose(mekf.attitude.as_quaternion(), q_expected)

#     def test_position(self, att):
#         p_n = np.array([1.0, 2.0, 3.0])
#         mekf = ap.MEKF(10.0, att, pos=p_n)
#         np.testing.assert_allclose(mekf.position, p_n)
#         assert mekf.position is not mekf._p_n  # ensure it is a copy

#     def test_velocity(self, att):
#         v_n = np.array([1.0, 2.0, 3.0])
#         mekf = ap.MEKF(10.0, att, vel=v_n)
#         np.testing.assert_allclose(mekf.velocity, v_n)
#         assert mekf.velocity is not mekf._v_n  # ensure it is a copy

#     def test_acceleration(self, att):
#         a_n = np.array([1.0, 2.0, 3.0])
#         mekf = ap.MEKF(10.0, att, acc=a_n)
#         np.testing.assert_allclose(mekf.acceleration, a_n)
#         assert mekf.acceleration is not mekf._a_n  # ensure it is a copy

#     def test_bias_gyro(self, att):
#         mekf = ap.MEKF(10.0, att, bg=np.array([0.01, -0.02, 0.03]))
#         bg_expected = np.array([0.01, -0.02, 0.03])
#         np.testing.assert_allclose(mekf.bias_gyro, bg_expected)
#         assert mekf.bias_gyro is not mekf._bg_b  # ensure it is a copy

#     def test_bias_acc(self, att):
#         mekf = ap.MEKF(10.0, att, ba=np.array([1.0, -2.3, 3.4]))
#         ba_expected = np.array([1.0, -2.3, 3.4])
#         np.testing.assert_allclose(mekf.bias_acc, ba_expected)
#         assert mekf.bias_acc is not mekf._ba_b  # ensure it is a copy

#     def test_angular_rate(self, att):
#         w = np.array([0.1, -0.2, 0.3])
#         mekf = ap.MEKF(10.0, att, w=w)
#         np.testing.assert_allclose(mekf.angular_rate, w)
#         assert mekf.angular_rate is not mekf._w_b  # ensure it is a copy

#     def test_P(self, mekf, att):
#         mekf = ap.MEKF(10.0, att, P=np.eye(15))
#         np.testing.assert_allclose(mekf.P, np.eye(15))
#         assert mekf.P is not mekf._P  # ensure it is a copy

#     def test_update(self, pva_sim):
#         _, _, _, euler_nb, f_b, w_b = pva_sim
#         fs = 10.24

#         # Add IMU measurement noise
#         acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
#         gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
#         bg_b = (0.001, 0.002, 0.003)  # rad/s
#         rng = np.random.default_rng(42)
#         f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
#         w_meas = (
#             w_b
#             + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
#             + bg_b
#         )

#         # Estimate attitude using MEKF
#         att = ap.Attitude.from_euler(euler_nb[0], degrees=False)
#         mekf = ap.MEKF(fs, att)
#         euler_est, bg_est = [], []
#         for f_i, w_i in zip(f_meas, w_meas):
#             mekf.update(f_i, w_i)
#             euler_est.append(mekf.attitude.as_euler())
#             bg_est.append(mekf.bias_gyro)
#         euler_est = np.asarray(euler_est)
#         bg_est = np.asarray(bg_est)

#         # Truncate 600 seconds from the beginning (so that filter has converged)
#         # Check roll and pitch only
#         warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
#         euler_expect = euler_nb[warmup:, :2]
#         bg_expect = np.full(bg_est.shape, bg_b)[warmup:, :2]
#         euler_out = euler_est[warmup:, :2]
#         bg_out = bg_est[warmup:, :2]

#         np.testing.assert_allclose(euler_out, euler_expect, atol=0.006)
#         np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)

#     def test_update_full_aiding(self, pva_sim):
#         _, _, v_n, euler_nb, f_b, w_b = pva_sim
#         yaw = euler_nb[:, 2]
#         fs = 10.24

#         # Add IMU measurement noise
#         acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
#         gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
#         bg_b = (0.001, 0.002, 0.003)  # rad/s
#         rng = np.random.default_rng(42)
#         f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
#         w_meas = (
#             w_b
#             + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
#             + bg_b
#         )

#         # Add velocity and heading measurement noise
#         v_var = 0.01  # (m/s)^2
#         yaw_var = 0.0001  # rad^2
#         rng = np.random.default_rng(42)
#         v_meas = v_n + np.sqrt(v_var) * rng.standard_normal(v_n.shape)
#         yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

#         # Estimate attitude using MEKF
#         att = ap.Attitude.from_euler(euler_nb[0], degrees=False)
#         mekf = ap.MEKF(fs, att)
#         euler_est, bg_est, v_est = [], [], []
#         for f_i, w_i, v_i, y_i in zip(f_meas, w_meas, v_meas, yaw_meas):
#             mekf.update(
#                 f_i, w_i, vel=v_i, vel_var=v_var * np.ones(3), yaw=y_i, yaw_var=yaw_var
#             )
#             euler_est.append(mekf.attitude.as_euler())
#             bg_est.append(mekf.bias_gyro)
#             v_est.append(mekf.velocity)
#         euler_est = np.asarray(euler_est)
#         bg_est = np.asarray(bg_est)
#         v_est = np.asarray(v_est)

#         # Truncate 600 seconds from the beginning (so that filter has converged)
#         warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
#         euler_expect = euler_nb[warmup:]
#         bg_expect = np.full(bg_est.shape, bg_b)[warmup:]
#         v_expect = v_n[warmup:]
#         euler_out = euler_est[warmup:]
#         bg_out = bg_est[warmup:]
#         v_out = v_est[warmup:]

#         np.testing.assert_allclose(euler_out, euler_expect, atol=0.006)
#         np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)
#         np.testing.assert_allclose(v_out, v_expect, atol=2.0 * np.sqrt(v_var))

#     def test_update_vel_aiding(self, pva_sim):
#         _, _, v_n, euler_nb, f_b, w_b = pva_sim
#         fs = 10.24

#         # Add IMU measurement noise
#         acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
#         gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
#         bg_b = (0.001, 0.002, 0.003)  # rad/s
#         rng = np.random.default_rng(42)
#         f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
#         w_meas = (
#             w_b
#             + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
#             + bg_b
#         )

#         # Add velocity and heading measurement noise
#         v_var = 0.01  # (m/s)^2
#         rng = np.random.default_rng(42)
#         v_meas = v_n + np.sqrt(v_var) * rng.standard_normal(v_n.shape)

#         # Estimate attitude using MEKF
#         att = ap.Attitude.from_euler(euler_nb[0], degrees=False)
#         mekf = ap.MEKF(fs, att)
#         euler_est, bg_est = [], []
#         for f_i, w_i, v_i in zip(f_meas, w_meas, v_meas):
#             mekf.update(f_i, w_i, vel=v_i, vel_var=v_var * np.ones(3))
#             euler_est.append(mekf.attitude.as_euler())
#             bg_est.append(mekf.bias_gyro)
#         euler_est = np.asarray(euler_est)
#         bg_est = np.asarray(bg_est)

#         # Truncate 600 seconds from the beginning (so that filter has converged)
#         # Check roll and pitch only
#         warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
#         euler_expect = euler_nb[warmup:, :2]
#         bg_expect = np.full(bg_est.shape, bg_b)[warmup:, :2]
#         euler_out = euler_est[warmup:, :2]
#         bg_out = bg_est[warmup:, :2]

#         np.testing.assert_allclose(euler_out, euler_expect, atol=0.006)
#         np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)

#     def test_update_yaw_aiding(self, pva_sim):
#         _, _, _, euler_nb, f_b, w_b = pva_sim
#         yaw = euler_nb[:, 2]
#         fs = 10.24

#         # Add IMU measurement noise
#         acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
#         gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
#         bg_b = (0.001, 0.002, 0.003)  # rad/s
#         rng = np.random.default_rng(42)
#         f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
#         w_meas = (
#             w_b
#             + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)
#             + bg_b
#         )

#         # Add velocity and heading measurement noise
#         yaw_var = 0.0001  # rad^2
#         rng = np.random.default_rng(42)
#         yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

#         # Estimate attitude using MEKF
#         att = ap.Attitude.from_euler(euler_nb[0], degrees=False)
#         mekf = ap.MEKF(fs, att)
#         euler_est, bg_est = [], []
#         for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
#             mekf.update(f_i, w_i, yaw=y_i, yaw_var=yaw_var)
#             euler_est.append(mekf.attitude.as_euler())
#             bg_est.append(mekf.bias_gyro)
#         euler_est = np.asarray(euler_est)
#         bg_est = np.asarray(bg_est)

#         # Truncate 600 seconds from the beginning (so that filter has converged)
#         warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
#         euler_expect = euler_nb[warmup:]
#         bg_expect = np.full(bg_est.shape, bg_b)[warmup:]
#         euler_out = euler_est[warmup:]
#         bg_out = bg_est[warmup:]

#         np.testing.assert_allclose(euler_out, euler_expect, atol=0.007)
#         np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)

#     def test_recover_state(self, pva_sim):
#         _, _, _, euler_nb, f_b, w_b = pva_sim
#         f_b, w_b = f_b[:10], w_b[:10]

#         fs = 10.24
#         q0 = _quat_from_euler_zyx(euler_nb[0])
#         mekf_a = ap.MEKF(fs, ap.Attitude(q0))

#         q_a, q_b = [], []
#         bg_a, bg_b = [], []
#         v_a, v_b = [], []
#         w_a, w_b = [], []
#         a_a, a_b = [], []
#         P_a, P_b = [], []
#         for f_i, w_i in zip(f_b, w_b):
#             mekf_b = ap.MEKF(
#                 fs,
#                 att=mekf_a.attitude,
#                 bg=mekf_a.bias_gyro,
#                 ba=mekf_a.bias_acc,
#                 vel=mekf_a.velocity,
#                 w=mekf_a.angular_rate,
#                 acc=mekf_a.acceleration,
#                 P=mekf_a.P,
#                 g=mekf_a._g,
#                 nav_frame=mekf_a._nav_frame,
#                 acc_noise_density=mekf_a._vrw,
#                 gyro_noise_density=mekf_a._arw,
#                 gyro_bias_stability=mekf_a._gbs,
#                 bias_corr_time=mekf_a._gbc,
#             )

#             mekf_a.update(f_i, w_i, degrees=False)
#             mekf_b.update(f_i, w_i, degrees=False)

#             q_a.append(mekf_a.attitude.as_quaternion())
#             q_b.append(mekf_b.attitude.as_quaternion())
#             bg_a.append(mekf_a.bias_gyro)
#             bg_b.append(mekf_b.bias_gyro)
#             v_a.append(mekf_a.velocity)
#             v_b.append(mekf_b.velocity)
#             w_a.append(mekf_a.angular_rate)
#             w_b.append(mekf_b.angular_rate)
#             a_a.append(mekf_a.acceleration)
#             a_b.append(mekf_b.acceleration)
#             P_a.append(mekf_a.P)
#             P_b.append(mekf_b.P)

#         np.testing.assert_allclose(q_a, q_b)
#         np.testing.assert_allclose(bg_a, bg_b)
#         np.testing.assert_allclose(v_a, v_b)
#         np.testing.assert_allclose(w_a, w_b)
#         np.testing.assert_allclose(a_a, a_b)
#         np.testing.assert_allclose(P_a, P_b)


class Test_MEKF:

    @fixture
    def att(self):
        return ap.Attitude((1.0, 0.0, 0.0, 0.0))

    @fixture
    def mekf(self):
        return ap.MEKF(10.0, (1.0, 0.0, 0.0, 0.0))

    def test__init__(self):
        fs = 1024.0
        q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        bg_b = (0.1, -0.2, 0.3)
        w_b = (0.01, -0.02, 0.03)
        P = 42.0 * np.eye(6)
        nav_frame = "enu"
        gyro_noise_density = 0.000456
        gyro_bias_stability = 0.0000789
        gyro_bias_corr_time = 123.0

        mekf = ap.MEKF(
            fs,
            q_nb,
            bg=bg_b,
            w=w_b,
            P=P,
            nav_frame=nav_frame,
            gyro_noise_density=gyro_noise_density,
            gyro_bias_stability=gyro_bias_stability,
            gyro_bias_corr_time=gyro_bias_corr_time,
        )

        assert mekf._fs == fs
        assert mekf._dt == 1.0 / fs
        assert mekf._nav_frame == "enu"
        assert mekf._nz2vg == -1.0

        assert mekf._arw == gyro_noise_density
        assert mekf._gbs == gyro_bias_stability
        assert mekf._gbc == gyro_bias_corr_time

        np.testing.assert_allclose(mekf._att_nb._q, q_nb)
        np.testing.assert_allclose(mekf._bg_b, bg_b)
        np.testing.assert_allclose(mekf._w_b, w_b)
        np.testing.assert_allclose(mekf._P, P)

        # Check C contiguity
        assert mekf._dhdx.flags.c_contiguous
        assert mekf._phi.flags.c_contiguous
        assert mekf._Q.flags.c_contiguous

    def test__init__default(self):
        fs = 10.0
        mekf = ap.MEKF(fs, (1.0, 0.0, 0.0, 0.0))

        assert mekf._fs == fs
        assert mekf._dt == 1.0 / fs
        assert mekf._nav_frame == "ned"
        assert mekf._nz2vg == 1.0

        assert mekf._arw == 0.0001
        assert mekf._gbs == 0.00005
        assert mekf._gbc == 50.0

        np.testing.assert_allclose(mekf._att_nb._q, np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(mekf._bg_b, np.zeros(3))
        np.testing.assert_allclose(mekf._P, 1e-6 * np.eye(6))

        np.testing.assert_allclose(mekf._w_b, np.zeros(3))

    def test__init__nav_frame(self):
        att = ap.Attitude((1.0, 0.0, 0.0, 0.0))

        mekf_ned = ap.MEKF(10.0, att, nav_frame="NED")
        assert mekf_ned._nz2vg == 1.0

        mekf_enu = ap.MEKF(10.0, att, nav_frame="ENU")
        assert mekf_enu._nz2vg == -1.0

        with pytest.raises(ValueError):
            ap.MEKF(10.0, att, nav_frame="invalid")

    def test_dhdx_gref(self, mekf):
        vg_b = np.random.random(3)
        vg_b /= np.linalg.norm(vg_b)
        dhdx = mekf._dhdx_gref(vg_b)
        dhdx_expected = np.zeros((3, 6))
        dhdx_expected[:, 0:3] = _skew_symmetric(vg_b)
        np.testing.assert_allclose(dhdx, dhdx_expected)
        assert dhdx.flags.c_contiguous

    def test_dhdx_yaw(self, mekf):
        q_nb = _quat_from_euler_zyx(np.radians([10.0, -20.0, 45.0]))
        dhdx_yaw = mekf._dhdx_yaw(q_nb)
        dhdx_yaw_expected = np.zeros((6,))
        dhdx_yaw_expected[0:3] = _dyawda(q_nb)
        np.testing.assert_allclose(dhdx_yaw, dhdx_yaw_expected)
        assert dhdx_yaw.flags.c_contiguous

    def test_attitude(self, mekf):
        q_expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(mekf.attitude, ap.Attitude)
        np.testing.assert_allclose(mekf.attitude.as_quaternion(), q_expected)

    def test_bias_gyro(self, att):
        mekf = ap.MEKF(10.0, att, bg=np.array([0.01, -0.02, 0.03]))
        bg_expected = np.array([0.01, -0.02, 0.03])
        np.testing.assert_allclose(mekf.bias_gyro, bg_expected)
        assert mekf.bias_gyro is not mekf._bg_b  # ensure it is a copy

    def test_angular_rate(self, att):
        w = np.array([0.1, -0.2, 0.3])
        mekf = ap.MEKF(10.0, att, w=w)
        np.testing.assert_allclose(mekf.angular_rate, w)
        assert mekf.angular_rate is not mekf._w_b  # ensure it is a copy

    def test_P(self, mekf, att):
        mekf = ap.MEKF(10.0, att, P=np.eye(6))
        np.testing.assert_allclose(mekf.P, np.eye(6))
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
        att = ap.Attitude.from_euler(euler_nb[0], degrees=False)
        mekf = ap.MEKF(fs, att)
        euler_est, bg_est = [], []
        for f_i, w_i in zip(f_meas, w_meas):
            mekf.update(f_i, w_i)
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bias_gyro)
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
        *_, euler_nb, f_b, w_b = pva_sim
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
        att = ap.Attitude.from_euler(euler_nb[0], degrees=False)
        mekf = ap.MEKF(fs, att)
        (
            euler_est,
            bg_est,
        ) = (
            [],
            [],
        )
        for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
            mekf.update(
                f_i,
                w_i,
                yaw=y_i,
                yaw_var=yaw_var,
                gref=True,
                gref_var=0.001 * np.ones(3),
            )
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bias_gyro)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Truncate 600 seconds from the beginning (so that filter has converged)
        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning
        euler_expect = euler_nb[warmup:]
        bg_expect = np.full(bg_est.shape, bg_b)[warmup:]
        euler_out = euler_est[warmup:]
        bg_out = bg_est[warmup:]

        np.testing.assert_allclose(euler_out, euler_expect, atol=0.006)
        np.testing.assert_allclose(bg_out, bg_expect, atol=0.005)

    # def test_prep_state_transition(self, mekf):
    #     phi_out = mekf._prep_state_transition_matrix()

    #     dt = mekf._dt
    #     f_b = np.zeros(3)
    #     w_b = mekf._w_b
    #     R_nb = mekf._att_nb.as_matrix()
    #     abc = 1.0
    #     gbc = mekf._gbc
    #     phi_expect = _statespace._state_transition(dt, f_b, w_b, R_nb, abc, gbc)
    #     state_idx = np.r_[_statespace.ATT_IDX, _statespace.BG_IDX]
    #     phi_expect = phi_expect[np.ix_(state_idx, state_idx)]

    #     np.testing.assert_allclose(phi_out, phi_expect)

    # def test_prep_process_noise_cov_matrix(self, mekf):
    #     Q_out = mekf._prep_process_noise_cov_matrix()

    #     dt = mekf._dt
    #     vrw = 1.0
    #     abs = 1.0
    #     abc = 1.0
    #     arw = mekf._arw
    #     gbs = mekf._gbs
    #     gbc = mekf._gbc
    #     Q_expect = _statespace._process_noise_cov(dt, vrw, arw, abs, abc, gbs, gbc)
    #     state_idx = np.r_[_statespace.ATT_IDX, _statespace.BG_IDX]
    #     Q_expect = Q_expect[np.ix_(state_idx, state_idx)]

    #     np.testing.assert_allclose(Q_out, Q_expect)
