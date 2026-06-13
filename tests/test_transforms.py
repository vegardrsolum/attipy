import json
from pathlib import Path

import numpy as np
import pytest

from attipy._transforms import (
    _euler_zyx_from_quat,
    _matrix_from_euler_zyx,
    _matrix_from_quat,
    _nz_b_from_quat,
)

_FIXTURES = json.loads((Path(__file__).parent / "testdata" / "attitudes.json").read_text())


@pytest.mark.parametrize("att", _FIXTURES)
def test_matrix_from_quat(att):
    result = _matrix_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["matrix"])


@pytest.mark.parametrize("att", _FIXTURES)
def test_euler_zyx_from_quat(att):
    result = _euler_zyx_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["euler_rad"])


@pytest.mark.parametrize("att", _FIXTURES)
def test_matrix_from_euler_zyx(att):
    result = _matrix_from_euler_zyx(np.array(att["euler_rad"]))
    np.testing.assert_allclose(result, att["matrix"], atol=1e-14)


@pytest.mark.parametrize("att", _FIXTURES)
def test_nz_b_from_quat(att):
    result = _nz_b_from_quat(np.array(att["quaternion"]))
    np.testing.assert_allclose(result, att["matrix"][2])
