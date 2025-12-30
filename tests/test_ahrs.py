import numpy as np

from attipy import AHRS, Attitude
from attipy._transforms import _quat_from_euler_zyx


class Test_AHRS:
    def test_update(self, pva_data):
        _, _, _, euler, f, w = pva_data
        head = euler[:, 2]

        rng = np.random.default_rng()
        f_imu = f + rng.normal(0.0, 0.001, f.shape)
        w_imu = w + rng.normal(0.0, 0.0001, w.shape)
        head_aid = head + rng.normal(0.0, np.radians(1.0), head.shape)

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

        np.testing.assert_allclose(euler_out, euler, atol=0.005)
