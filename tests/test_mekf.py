import numpy as np

import attipy as ap


class Test_MEKF:

    def test_update_full_aiding(self, pva_sim):
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

        # Add heading measurement noise
        yaw_var = 0.0001  # rad^2
        rng = np.random.default_rng(42)
        yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

        # Estimate attitude using MEKF
        att = ap.Attitude.from_euler(euler_nb[0], degrees=False)
        mekf = ap.MEKF(fs, att)
        euler_est, bg_est = [], []
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
