from pathlib import Path

import pytest

import attipy as ap

TEST_PATH = Path(__file__).parent


@pytest.fixture
def trajectory():
    fs = 10.24
    n = 1800.0 * fs  # 30 minutes
    return ap.simulate.trajectory(fs, n, g=9.80665, nav_frame="ned")
