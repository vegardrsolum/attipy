from pathlib import Path

import pytest

TEST_PATH = Path(__file__).parent


@pytest.fixture
def ahrs_data():
    import pandas as pd

    df = pd.read_csv(TEST_PATH / r"testdata/benchmark_pure_attitude_beat_202311A.csv")

    t = df["Time_s"].values.astype(float)
    euler = df[["Roll_rad", "Pitch_rad", "Yaw_rad"]].values.astype(float)
    w = df[["GyroX_rads", "GyroY_rads", "GyroZ_rads"]].values.astype(float)
    f = df[["AccX_ms2", "AccY_ms2", "AccZ_ms2"]].values.astype(float)

    return t, euler, f, w
