from pathlib import Path

import numpy as np
import pytest

TEST_PATH = Path(__file__).parent


@pytest.fixture
def ahrs_data():
    data = np.genfromtxt(
        TEST_PATH / "testdata/benchmark_full_pva_beat_202311A.csv",
        delimiter=",",
        names=True,
        dtype=float,
    )

    t = data["Time_s"]
    pos_x = data["PosX_m"]
    pos_y = data["PosY_m"]
    pos_z = data["PosZ_m"]
    vel_x = data["VelX_ms"]
    vel_y = data["VelY_ms"]
    vel_z = data["VelZ_ms"]
    roll = data["Roll_rad"]
    pitch = data["Pitch_rad"]
    yaw = data["Yaw_rad"]
    gx = data["GyroX_rads"]
    gy = data["GyroY_rads"]
    gz = data["GyroZ_rads"]
    ax = data["AccX_ms2"]
    ay = data["AccY_ms2"]
    az = data["AccZ_ms2"]

    pos = np.column_stack((pos_x, pos_y, pos_z))
    vel = np.column_stack((vel_x, vel_y, vel_z))
    euler = np.column_stack((roll, pitch, yaw))
    w = np.column_stack((gx, gy, gz))
    f = np.column_stack((ax, ay, az))

    return t, pos, vel, euler, f, w
