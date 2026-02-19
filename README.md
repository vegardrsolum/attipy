# AttiPy
AttiPy is a lightweight Python library for representing and estimating the attitude
(orientation) and linear motion of a body using IMU measurements and optional external
aiding. It provides a multiplicative extended Kalman filter (MEKF) for position,
velocity and attitude (PVA) estimation, and an attitude abstraction with clearly defined
reference frames and rotation conventions.

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


Estimate roll and pitch from IMU measurements (accelerometer and gyroscope) using
the ``MEKF`` class:

```python
import attipy as ap
import numpy as np


# Position, velocity, attitude and IMU reference signals
fs = 10.0  # Hz
t, pos, vel, euler, f, w = ap.pva_sim(fs)

# Add IMU measurement noise
acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
bg = (0.001, 0.002, 0.003)  # rad/s
rng = np.random.default_rng(42)
f_meas = f + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f.shape)
w_meas = w + bg + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w.shape)

# Estimate attitude using MEKF
att = ap.Attitude.from_euler(euler[0])
mekf = ap.MEKF(fs, att)
euler_est = []
for f_i, w_i in zip(f_meas, w_meas):
    mekf.update(f_i, w_i)
    euler_est.append(mekf.attitude.as_euler())
euler_est = np.asarray(euler_est)
```

To limit integration drift, the MEKF corrects its state estimates using long-term
stable aiding measurements. When no aiding measurements are available (as in the
example above), stationarity is assumed to ensure convergence. By default, zero-velocity
aiding with a 10 m/s standard deviation is applied; this constrains roll and pitch only,
as these are the only degrees of freedom observable from specific force measurements
and the known direction of gravity. Under sustained linear acceleration, velocity
and/or position aiding is recommended to maintain accurate attitude estimates.

The following example demonstrates how to estimate position, velocity and attitude
(including yaw) from IMU and aiding measurements.

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

# Add velocity and heading measurement noise
pos_var = 0.1  # m
vel_var = 0.01  # (m/s)^2
yaw_var = 0.0001  # rad^2
rng = np.random.default_rng(42)
pos_meas = pos + np.sqrt(pos_var) * rng.standard_normal(pos.shape)
vel_meas = vel + np.sqrt(vel_var) * rng.standard_normal(vel.shape)
yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

# Estimate position, velocity and attitude using MEKF
att = ap.Attitude.from_euler(euler[0])
mekf = ap.MEKF(fs, att)
pos_est, vel_est, euler_est = [], [], []
for f_i, w_i, p_i, v_i, y_i in zip(f_meas, w_meas, pos_meas, vel_meas, yaw_meas):
    mekf.update(
        f_i,
        w_i,
        pos=p_i,
        pos_var=pos_var*np.ones(3),
        vel=v_i,
        vel_var=vel_var*np.ones(3),
        yaw=y_i,
        yaw_var=yaw_var
    )
    pos_est.append(mekf.position)
    vel_est.append(mekf.velocity)
    euler_est.append(mekf.attitude.as_euler())
pos_est = np.asarray(pos_est)
vel_est = np.asarray(vel_est)
euler_est = np.asarray(euler_est)
```

## Limitations and assumptions

- Intended for small-area, low-velocity applications; Earth rotation is neglected,
and gravitational acceleration is assumed constant.
