from pathlib import Path

import pytest
import numpy as np

TEST_PATH = Path(__file__).parent


@pytest.fixture
def ahrs_data():
    data = np.genfromtxt(
        TEST_PATH / "testdata/benchmark_pure_attitude_beat_202311A.csv",
        delimiter=",",
        names=True,
        dtype=float,
    )

    t = data["Time_s"]
    euler = np.column_stack((
        data["Roll_rad"],
        data["Pitch_rad"],
        data["Yaw_rad"],
    ))
    w = np.column_stack((
        data["GyroX_rads"],
        data["GyroY_rads"],
        data["GyroZ_rads"],
    ))
    f = np.column_stack((
        data["AccX_ms2"],
        data["AccY_ms2"],
        data["AccZ_ms2"],
    ))

    return t, euler, f, w
