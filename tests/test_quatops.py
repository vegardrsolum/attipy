import numpy as np

import attipy as ap
from attipy._quatops import _correct_quat_with_gibbs2, _quatprod


def test_correct_quat_with_gibbs2():

    q = ap.Attitude.from_euler([10.0, 20.0, 30.0], degrees=True).as_quaternion()
    da = np.array([0.01, -0.02, 0.03])  # 2x Gibbs vector attitude correction

    dq = (1.0 / np.sqrt(4.0 + np.dot(da, da))) * np.array([2.0, *da])
    q_corr_expect = _quatprod(q, dq)
    q_corr_expect = q_corr_expect

    _correct_quat_with_gibbs2(q, da)

    np.testing.assert_allclose(q, q_corr_expect)
