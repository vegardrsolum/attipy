import json
from pathlib import Path

import numpy as np
import pytest

from attipy._transforms import (
    _dyawda,
    _euler_zyx_from_quat,
    _matrix_from_euler_zyx,
    _matrix_from_quat,
    _nz_b_from_quat,
    _quat_from_euler_zyx,
    _quat_from_gibbs2,
    _quat_from_matrix,
    _quat_from_rotvec,
    _rotvec_from_quat,
    _yaw_from_quat,
)

_ATTITUDES = json.loads(
    (Path(__file__).parent / "testdata" / "attitudes.json").read_text()
)


@pytest.mark.parametrize("att", _ATTITUDES)
def test_matrix_from_quat(att):
    result = _matrix_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["matrix"])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_quat_from_matrix(att):
    result = _quat_from_matrix(np.array(att["matrix"]))
    np.testing.assert_allclose(result, att["quaternion"])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_euler_zyx_from_quat(att):
    result = _euler_zyx_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["euler_rad"])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_quat_from_euler_zyx(att):
    result = _quat_from_euler_zyx(np.array(att["euler_rad"]))
    np.testing.assert_allclose(result, att["quaternion"], atol=1e-14)


@pytest.mark.parametrize("att", _ATTITUDES)
def test_matrix_from_euler_zyx(att):
    result = _matrix_from_euler_zyx(np.array(att["euler_rad"]))
    np.testing.assert_allclose(result, att["matrix"], atol=1e-14)


@pytest.mark.parametrize("att", _ATTITUDES)
def test_quat_from_rotvec(att):
    result = _quat_from_rotvec(np.array(att["rotvec"]))
    np.testing.assert_allclose(result, att["quaternion"])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_rotvec_from_quat(att):
    result = _rotvec_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["rotvec"])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_yaw_from_quat(att):
    result = _yaw_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["euler_rad"][2])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_nz_b_from_quat(att):
    result = _nz_b_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["matrix"][2])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_quat_from_gibbs2(att):
    result = _quat_from_gibbs2(np.array(att["gibbs2"]))
    np.testing.assert_allclose(result, att["quaternion"])


@pytest.mark.parametrize("att", _ATTITUDES)
def test_dyawda(att):
    q = np.array(att["quaternion"])
    a = np.array(att["gibbs2"])

    # Skip cases where yaw is close to ±180 degrees, where the gradient is ill-defined
    yaw = att["euler_rad"][2]
    if abs(abs(yaw) - np.pi) < 0.01:
        return

    result = _dyawda(q)

    def yaw_from_gibbs2(a):
        ax, ay, az = a
        return np.arctan2(2.0 * (ax * ay + 2.0 * az), 4.0 + ax**2 - ay**2 - az**2)

    # Numerical gradient via centred finite differences on the scaled Gibbs vector
    eps = 1e-6
    numerical = np.empty(3)
    for i in range(3):
        a_fwd = a.copy()
        a_bwd = a.copy()
        a_fwd[i] += eps
        a_bwd[i] -= eps
        numerical[i] = (yaw_from_gibbs2(a_fwd) - yaw_from_gibbs2(a_bwd)) / (2.0 * eps)

    np.testing.assert_allclose(result, numerical, atol=1e-8)
