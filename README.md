# AttiPy
AttiPy is a lightweight Python package for representing and estimating the attitude (orientation) of a moving body using IMU measurements and optional external aiding. It provides a practical Attitude and Heading Reference System (AHRS) implementation based on a multiplicative extended Kalman filter (MEKF), along with a clean, explicit abstraction for attitude representation, with clearly defined reference frames and rotation conventions.

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


Estimate attitude using IMU (accelerometer and gyroscope) measurements with the
``AHRS`` class:

```python
import attipy as ap
import numpy as np


# Position, velocity, attitude and IMU reference signals
fs = 10.0  # Hz
t, pos, vel, euler, f_b, w_b = ap.pva_sim(fs)

# Add IMU measurement noise
acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
bg_b = (0.001, 0.002, 0.003)  # rad/s
rng = np.random.default_rng(42)
f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
w_meas = w_b + bg_b + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)

# Estimate attitude using AHRS
att0 = ap.Attitude.from_euler(euler[0])
ahrs = ap.AHRS(fs, att0)
euler_est = []
for f_i, w_i in zip(f_meas, w_meas):
    ahrs.update(f_i, w_i)
    euler_est.append(ahrs.attitude.as_euler())
euler_est = np.asarray(euler_est)
```

To limit integration drift, the AHRS state estimates must be corrected using long-term
stable aiding measurements. When no aiding is available (as in the example above),
stationarity is assumed to ensure convergence. By default, zero-velocity aiding
with a 10 m/s standard deviation is used; this constrains roll and pitch only, as
these are the only degrees of freedom observable from specific force and the known
direction of gravity.

Under sustained linear acceleration, velocity aiding is recommended to maintain
accurate attitude estimates. Heading (yaw) aiding should also be applied to correct
yaw drift. The following example shows how to incorporate both.

```python
import attipy as ap
import numpy as np


# Position, velocity, attitude and IMU reference signals
fs = 10.0  # Hz
t, pos, vel, euler, f_b, w_b = ap.pva_sim(fs)
yaw = euler[:, 2]

# Add IMU measurement noise
acc_noise_density = 0.001  # (m/s^2) / sqrt(Hz)
gyro_noise_density = 0.0001  # (rad/s) / sqrt(Hz)
bg_b = (0.001, 0.002, 0.003)  # rad/s
rng = np.random.default_rng(42)
f_meas = f_b + acc_noise_density * np.sqrt(fs) * rng.standard_normal(f_b.shape)
w_meas = w_b + bg_b + gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w_b.shape)

# Add velocity and heading measurement noise
vel_var = 0.01  # (m/s)^2
yaw_var = 0.0001  # rad^2
rng = np.random.default_rng(42)
vel_meas = vel + np.sqrt(vel_var) * rng.standard_normal(vel.shape)
yaw_meas = yaw + np.sqrt(yaw_var) * rng.standard_normal(yaw.shape)

# Estimate attitude using AHRS
att0 = ap.Attitude.from_euler(euler[0])
ahrs = ap.AHRS(fs, att0)
euler_est = []
for f_i, w_i, v_i, y_i in zip(f_meas, w_meas, vel_meas, yaw_meas):
    ahrs.update(f_i, w_i, v_n=v_i, v_var=vel_var*np.ones(3), yaw=y_i, yaw_var=yaw_var)
    euler_est.append(ahrs.attitude.as_euler())
euler_est = np.asarray(euler_est)
```

## Limitations and assumptions

- Intended for small-area, low-velocity applications; Earth rotation is neglected.
- Accelerometer bias is not estimated; a calibrated accelerometer is assumed.
