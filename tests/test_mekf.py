import numpy as np
import pytest
from pytest import fixture

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
