import numpy as np

from attipy import AHRS, Attitude
from attipy._transforms import _quat_from_euler_zyx


class Test_AHRS:

    def test__init__(self):
        fs = 10.0
        q0 = _quat_from_euler_zyx(np.radians([10.0, 20.0, 30.0]))
        bg0 = np.array([0.01, -0.02, 0.03])
        err_gyro= {"N": 0.0002, "B": 0.00003, "tau_cb": 45.0}
        ahrs = AHRS(fs, q0=q0, bg0=bg0, err_gyro=err_gyro, nav_frame="NED")

        assert ahrs._fs == fs
        assert ahrs._dt == 1.0 / fs
        assert ahrs._nav_frame == "ned"
        assert ahrs._err_gyro == err_gyro
        np.testing.assert_allclose(ahrs._vg_ref_n, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(ahrs.attitude.as_quaternion(), q0)
        np.testing.assert_allclose(ahrs._bg, bg0)
        np.testing.assert_allclose(ahrs._P, 1e-6 * np.eye(6))
        np.testing.assert_allclose(ahrs._w_corr, np.zeros(3))

    def test__init__nav_frame(self):
        ahrs_ned = AHRS(10.0, nav_frame="NED")
        np.testing.assert_allclose(ahrs_ned._vg_ref_n, np.array([0.0, 0.0, 1.0]))

        ahrs_enu = AHRS(10.0, nav_frame="ENU")
        np.testing.assert_allclose(ahrs_enu._vg_ref_n, np.array([0.0, 0.0, -1.0]))

    def test_update(self, pva_data):
        _, _, _, euler, f, w = pva_data
        head = euler[:, 2]

        rng = np.random.default_rng(seed=42)
        f_imu = f + 0.001 * rng.standard_normal(f.shape)
        w_imu = w + 0.0001 * rng.standard_normal(w.shape)
        head_aid = head + np.radians(1.0) * rng.standard_normal(head.shape)

        fs = 10.24
        q0 = _quat_from_euler_zyx(euler[0])
        ahrs = AHRS(fs, q0)

        euler_out = []
        for f_i, w_i, h_i in zip(f_imu, w_imu, head_aid):
            ahrs.update(
                f_i,
                w_i,
                degrees=False,
                head=h_i,
                head_var=np.radians(1.0),
                head_degrees=False,
                g_ref=True,
                g_var=0.1 * np.ones(3),
            )
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        np.testing.assert_allclose(euler_out, euler, atol=0.01)
