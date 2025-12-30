import numpy as np

from attipy import AHRS, Attitude


class Test_AHRS:
    def test_update(self, pva_data):
        _, _, _, euler, f, w = pva_data

        fs = 10.24
        att = Attitude.from_euler(euler[0], degrees=False)
        ahrs = AHRS(fs, att)

        euler_out = []
        for f_i, w_i, h_i in zip(f, w, euler[:, 2]):
            ahrs.update(
                f_i,
                w_i,
                degrees=False,
                head=h_i,
                head_var=1.0,
                head_degrees=False,
                g_ref=True,
                g_var=0.1 * np.ones(3),
            )
            euler_out.append(ahrs.attitude.as_euler(degrees=False))

        euler_out = np.asarray(euler_out)

        np.testing.assert_allclose(euler_out, euler, atol=0.01)
