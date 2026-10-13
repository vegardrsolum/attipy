import numpy as np
import pytest

from attipy._frames import _gravity_nav, _nz2vg


@pytest.mark.parametrize(
    "nav_frame, g_n_expected",
    [
        ("NED", [0.0, 0.0, 9.81]),
        ("ned", [0.0, 0.0, 9.81]),
        ("ENU", [0.0, 0.0, -9.81]),
        ("enu", [0.0, 0.0, -9.81]),
    ],
)
def test_gravity_nav(nav_frame, g_n_expected):
    np.testing.assert_allclose(_gravity_nav(9.81, nav_frame), g_n_expected)


def test_gravity_nav_invalid():
    with pytest.raises(ValueError):
        _gravity_nav(9.81, "invalid")


@pytest.mark.parametrize(
    "nav_frame, nz2vg_expected",
    [("NED", 1.0), ("ned", 1.0), ("ENU", -1.0), ("enu", -1.0)],
)
def test_nz2vg(nav_frame, nz2vg_expected):
    assert _nz2vg(nav_frame) == nz2vg_expected


def test_nz2vg_invalid():
    with pytest.raises(ValueError):
        _nz2vg("invalid")


@pytest.mark.parametrize("nav_frame", ["NED", "ENU"])
def test_nz2vg_consistent_with_gravity_nav(nav_frame):
    vg_n = _nz2vg(nav_frame) * np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(vg_n, _gravity_nav(1.0, nav_frame))
