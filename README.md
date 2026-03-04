# AttiPy
AttiPy is a lightweight Python library for representing and estimating the attitude
(orientation) of a body using IMU measurements and optional external aiding. It
provides an attitude abstraction with clearly defined reference frames and rotation
conventions, and a multiplicative extended Kalman filter (MEKF) for attitude estimation.

## Installation
```bash
pip install attipy
```

## Quick start

Convert to/from a variety of attitude representations using the ``Attitude`` class:

```python
import attipy as ap
import numpy as np


# From Euler angles to unit quaternion
att = ap.Attitude.from_euler([0.0, 0.0, 0.0])
q = att.as_quaternion()
```


Estimate attitude from IMU (accelerometer and gyroscope) and heading measurements
using the ``MEKF`` class:

```python
import attipy as ap
import numpy as np


# Position, velocity, attitude and IMU reference signals
fs = 10.0  # Hz
t, pos, vel, euler, f, w = ap.pva_sim(fs)
yaw = euler[:, 2]

# Add IMU measurement noise
acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
bg = (0.001, 0.002, 0.003)  # rad/s
rng = np.random.default_rng(42)
f_meas = f + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f.shape)
w_meas = w + bg + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w.shape)

# Add heading measurement noise
yaw_std = 0.01  # rad
yaw_meas = yaw + yaw_std * rng.standard_normal(yaw.shape)

# Estimate attitude using MEKF
att0 = ap.Attitude.from_euler(euler[0])
mekf = ap.MEKF(fs, att0)
euler_est = []
for f_i, w_i, y_i in zip(f_meas, w_meas, yaw_meas):
    mekf.update(f_i, w_i, yaw=y_i, yaw_var=yaw_std**2)
    euler_est.append(mekf.attitude.as_euler())
euler_est = np.asarray(euler_est)
```

## Limitations and assumptions

- Intended for small-area, low-velocity applications; Earth rotation is neglected,
and gravitational acceleration is assumed constant.
