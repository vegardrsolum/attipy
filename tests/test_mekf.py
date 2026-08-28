import numpy as np
import pytest
from pytest import fixture
from scipy.signal import resample_poly

import attipy as ap
from attipy._mekf import _dyawda
from attipy._transforms import _quat_from_euler_zyx
from attipy._vectorops import _skew_symmetric


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
        P = 42.0 * np.eye(6)
        nav_frame = "enu"
        gyro_noise_density = 0.000456
        gyro_bias_stability = 0.0000789
        gyro_bias_corr_time = 123.0

        mekf = ap.MEKF(
            fs,
            q_nb,
            b0=bg_b,
            P0=P,
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
        np.testing.assert_allclose(mekf._P, P)

        # Check C contiguity
        assert mekf._dhdx_gref.flags.c_contiguous
        assert mekf._dhdx_yaw.flags.c_contiguous
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

    def test__init__nav_frame(self):
        mekf_ned = ap.MEKF(10.0, nav_frame="NED")
        assert mekf_ned._nz2vg == 1.0

        mekf_enu = ap.MEKF(10.0, nav_frame="ENU")
        assert mekf_enu._nz2vg == -1.0

        with pytest.raises(ValueError):
            ap.MEKF(10.0, nav_frame="invalid")

    def test_attitude(self, mekf):
        q_expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(mekf.attitude, ap.Attitude)
        np.testing.assert_allclose(mekf.attitude.as_quaternion(), q_expected)

    def test_bias(self):
        mekf = ap.MEKF(10.0, b0=np.array([0.01, -0.02, 0.03]))
        bg_expected = np.array([0.01, -0.02, 0.03])
        np.testing.assert_allclose(mekf.bias, bg_expected)
        assert mekf.bias is not mekf._bg_b  # ensure it is a copy

    def test_P(self):
        mekf = ap.MEKF(10.0, P0=np.eye(6))
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
        q0 = ap.Attitude.from_euler(euler_nb[0], degrees=False).as_quaternion()
        mekf = ap.MEKF(fs, q0)
        euler_est, bg_est = [], []
        for f_i, w_i in zip(f_meas, w_meas):
            mekf.update(f_i, w_i)
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bias)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        euler_est = resample_poly(euler_est, 2, 1)[1:-1:2]
        bg_est = resample_poly(bg_est, 2, 1)[1:-1:2]
        euler_nb = euler_nb[1:, :]
        bg_b = np.tile(bg_b, (len(bg_est), 1))

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        roll_rmse, pitch_rmse, _ = rmse(euler_nb[warmup:], euler_est[warmup:])
        bgx_rmse, bgy_rmse, _ = rmse(bg_b[warmup:], bg_est[warmup:])

        assert np.degrees(roll_rmse) <= 0.1
        assert np.degrees(pitch_rmse) <= 0.1
        assert np.degrees(bgx_rmse) <= 0.01
        assert np.degrees(bgy_rmse) <= 0.01

        np.testing.assert_allclose(
            euler_est[warmup:, :2], euler_nb[warmup:, :2], atol=0.005
        )
        np.testing.assert_allclose(bg_est[warmup:, :2], bg_b[warmup:, :2], atol=0.005)

    def test_update_with_increments(self, pva_sim):
        _, _, _, euler_nb, f_b, w_b = pva_sim
        fs = 10.24
        dt = 1.0 / fs

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
        q0 = ap.Attitude.from_euler(euler_nb[0], degrees=False).as_quaternion()
        mekf = ap.MEKF(fs, q0)
        euler_est, bg_est = [], []
        for f_i, w_i in zip(f_meas, w_meas):
            mekf.update(f_i * dt, w_i * dt, increments=True)
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bias)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        euler_est = resample_poly(euler_est, 2, 1)[1:-1:2]
        bg_est = resample_poly(bg_est, 2, 1)[1:-1:2]
        euler_nb = euler_nb[1:, :]
        bg_b = np.tile(bg_b, (len(bg_est), 1))

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        roll_rmse, pitch_rmse, _ = rmse(euler_nb[warmup:], euler_est[warmup:])
        bgx_rmse, bgy_rmse, _ = rmse(bg_b[warmup:], bg_est[warmup:])

        assert np.degrees(roll_rmse) <= 0.1
        assert np.degrees(pitch_rmse) <= 0.1
        assert np.degrees(bgx_rmse) <= 0.01
        assert np.degrees(bgy_rmse) <= 0.01

        np.testing.assert_allclose(
            euler_est[warmup:, :2], euler_nb[warmup:, :2], atol=0.005
        )
        np.testing.assert_allclose(bg_est[warmup:, :2], bg_b[warmup:, :2], atol=0.005)

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
        q0 = ap.Attitude.from_euler(euler_nb[0], degrees=False).as_quaternion()
        mekf = ap.MEKF(fs, q0)
        euler_est, bg_est = [], []
        for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
            mekf.update(
                f_i,
                w_i,
                increments=False,
                gyro_degrees=False,
                yaw=y_i,
                yaw_var=yaw_var,
                gref=True,
                gref_var=0.001 * np.ones(3),
            )
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bias)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        euler_est = resample_poly(euler_est, 2, 1)[1:-1:2]
        bg_est = resample_poly(bg_est, 2, 1)[1:-1:2]
        euler_nb = euler_nb[1:, :]
        bg_b = np.tile(bg_b, (len(bg_est), 1))

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        roll_rmse, pitch_rmse, yaw_rmse = rmse(euler_nb[warmup:], euler_est[warmup:])
        bgx_rmse, bgy_rmse, bgz_rmse = rmse(bg_b[warmup:], bg_est[warmup:])

        assert np.degrees(roll_rmse) <= 0.1
        assert np.degrees(pitch_rmse) <= 0.1
        assert np.degrees(yaw_rmse) <= 0.1
        assert np.degrees(bgx_rmse) <= 0.01
        assert np.degrees(bgy_rmse) <= 0.01
        assert np.degrees(bgz_rmse) <= 0.01

        np.testing.assert_allclose(
            euler_est[warmup:, :], euler_nb[warmup:, :], atol=0.005
        )
        np.testing.assert_allclose(bg_est[warmup:, :], bg_b[warmup:, :], atol=0.005)

    def test_update_full_aiding_increments(self, pva_sim):
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
        q0 = ap.Attitude.from_euler(euler_nb[0], degrees=False).as_quaternion()
        mekf = ap.MEKF(fs, q0)
        euler_est, bg_est = [], []
        for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
            mekf.update(
                f_i / fs,
                w_i / fs,
                increments=True,
                gyro_degrees=False,
                yaw=y_i,
                yaw_var=yaw_var,
                gref=True,
                gref_var=0.001 * np.ones(3),
            )
            euler_est.append(mekf.attitude.as_euler())
            bg_est.append(mekf.bias)
        euler_est = np.asarray(euler_est)
        bg_est = np.asarray(bg_est)

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        euler_est = resample_poly(euler_est, 2, 1)[1:-1:2]
        bg_est = resample_poly(bg_est, 2, 1)[1:-1:2]
        euler_nb = euler_nb[1:, :]
        bg_b = np.tile(bg_b, (len(bg_est), 1))

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        roll_rmse, pitch_rmse, yaw_rmse = rmse(euler_nb[warmup:], euler_est[warmup:])
        bgx_rmse, bgy_rmse, bgz_rmse = rmse(bg_b[warmup:], bg_est[warmup:])

        assert np.degrees(roll_rmse) <= 0.1
        assert np.degrees(pitch_rmse) <= 0.1
        assert np.degrees(yaw_rmse) <= 0.1
        assert np.degrees(bgx_rmse) <= 0.01
        assert np.degrees(bgy_rmse) <= 0.01
        assert np.degrees(bgz_rmse) <= 0.01

        np.testing.assert_allclose(
            euler_est[warmup:, :], euler_nb[warmup:, :], atol=0.005
        )
        np.testing.assert_allclose(bg_est[warmup:, :], bg_b[warmup:, :], atol=0.005)
