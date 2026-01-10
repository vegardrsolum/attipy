import numpy as np

import attipy as ap


class Test_FixedIntervalSmoother:

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

        ahrs_ref = ap.AHRS(fs)  # w/o smoothing
        ahrs_smth = ap.FixedIntervalSmoother(ap.AHRS(fs), cov_smoothing=True)
        euler_ref, bg_ref, v_ref = [], [], []
        for f_i, w_i, v_i, y_i in zip(f_meas, w_meas, v_meas, yaw_meas):
            aid_kwargs = {
                "v_n": v_i,
                "v_var": v_var * np.ones(3),
                "yaw": y_i,
                "yaw_var": yaw_var,
            }
            ahrs_ref.update(f_i, w_i, **aid_kwargs)
            ahrs_smth.update(f_i, w_i, **aid_kwargs)
            euler_ref.append(ahrs_ref.attitude.as_euler())
            bg_ref.append(ahrs_ref.bg_b)
            v_ref.append(ahrs_ref.v_n)
        euler_ref = np.asarray(euler_ref)
        bg_ref = np.asarray(bg_ref)
        v_ref = np.asarray(v_ref)

        bg_smth = ahrs_smth.bg_b
        v_smth = ahrs_smth.v_n
        q_smth = ahrs_smth.q_nb
        euler_smth = np.empty_like(euler_ref)
        for i in range(len(euler_ref)):
            euler_smth[i] = ap._transforms._euler_zyx_from_quat(q_smth[i])

        # Truncate 600 seconds from the beginning (so that filter has converged)
        warmup = int(fs * 600.0)  # truncate 600 seconds from the beginning

        # Attitude
        err_euler_ref = (euler_nb - euler_ref)[warmup:]
        err_smth_euler = (euler_nb - euler_smth)[warmup:]
        assert err_smth_euler.std() < err_euler_ref.std()
        assert err_smth_euler.mean() < err_euler_ref.mean()

        # Gyro bias
        err_bg_ref = (np.full(bg_ref.shape, bg_b) - bg_ref)[warmup:]
        err_bg_smth = (np.full(bg_ref.shape, bg_b) - bg_smth)[warmup:]
        assert err_bg_smth.std() < err_bg_ref.std()
        assert err_bg_smth.mean() < err_bg_ref.mean() + 1e-6  # allow small diff

        # Velocity
        err_v_ref = (v_n - v_ref)[warmup:]
        err_v_smth = (v_n - v_smth)[warmup:]
        assert err_v_smth.std() < err_v_ref.std()
        assert err_v_smth.mean() < err_v_ref.mean()
