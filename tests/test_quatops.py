import numpy as np
import pytest

import attipy as ap
from attipy._quatops import (
    _correct_quat_with_gibbs2,
    _correct_quat_with_rotvec,
    _quatprod,
)

euler_deg_data = [
    (0.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
    (-10.0, 0.0, 0.0),
    (0.0, 10.0, 0.0),
    (0.0, -10.0, 0.0),
    (0.0, 0.0, 10.0),
    (0.0, 0.0, -10.0),
    (1.0, 2.0, 3.0),
    (-1.0, -2.0, -3.0),
    (90.0, 25.0, -30.0),
    (-90.0, -25.0, 30.0),
]


@pytest.mark.parametrize("euler_deg", euler_deg_data)
def test_correct_quat_with_gibbs2(euler_deg):

    q = ap.Attitude.from_euler(euler_deg, degrees=True).as_quaternion()
    da = np.array([0.01, -0.02, 0.03])  # 2x Gibbs vector attitude correction

    dq = (1.0 / np.sqrt(4.0 + np.dot(da, da))) * np.array([2.0, *da])
    q_corr_expect = _quatprod(q, dq)
    q_corr_expect = q_corr_expect

    _correct_quat_with_gibbs2(q, da)

    np.testing.assert_allclose(q, q_corr_expect)


@pytest.mark.parametrize("euler_deg", euler_deg_data)
def test_correct_quat_with_rotvec(euler_deg):

    q = ap.Attitude.from_euler(euler_deg, degrees=True).as_quaternion()
    dtheta = np.random.default_rng(42).random(3) * 0.01

    dtheta_norm = np.linalg.norm(dtheta)
    dq_w = np.cos(0.5 * dtheta_norm)
    dq_xyz = np.sin(0.5 * dtheta_norm) * dtheta / dtheta_norm
    dq = np.array((dq_w, *dq_xyz))
    q_corr_expect = _quatprod(q, dq)

    _correct_quat_with_rotvec(q, dtheta)

    np.testing.assert_allclose(q, q_corr_expect)
