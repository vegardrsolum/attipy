import numpy as np
from scipy.signal import resample_poly

import attipy as ap


class Test_RTSSmoother:

    def test_update(self, pva_sim):
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

        # Estimate attitude using MEKF (forward filter), and smooth with the RTS smoother
        q0 = ap.Attitude.from_euler(euler_nb[0], degrees=False).as_quaternion()
        mekf = ap.MEKF(fs, q0)
        smoother = ap.RTSSmoother(mekf)

        euler_fwd = []
        for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
            smoother.update(  # full aiding
                f_i,
                w_i,
                yaw=y_i,
                yaw_var=yaw_var,
                gref=True,
                gref_var=0.001 * np.ones(3),
            )
            euler_fwd.append(smoother._mekf.attitude.as_euler())
        euler_fwd = np.asarray(euler_fwd)
        euler_smth = smoother.euler()

        # Half-sample shift (compensates for the time shift introduced by Euler integration)
        euler_fwd = resample_poly(euler_fwd, 2, 1)[1:-1:2]
        euler_smth = resample_poly(euler_smth, 2, 1)[1:-1:2]
        euler_ref = euler_nb[1:, :]

        # Truncate 600 seconds from the beginning (so that the forward filter has converged)
        warmup = int(fs * 600.0)

        def rmse(ref, est):
            return np.sqrt(np.mean((ref - est) ** 2, axis=0))

        rmse_fwd = rmse(euler_ref[warmup:], euler_fwd[warmup:])
        rmse_smth = rmse(euler_ref[warmup:], euler_smth[warmup:])

        # Smoothing should reduce the estimation error compared to the forward filter alone
        assert np.all(rmse_smth < rmse_fwd)
