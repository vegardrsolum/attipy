# AttiPy
AttiPy is a lightweight Python package for representing and estimating the attitude
(orientation) of a moving body using IMU measurements, with optional external aiding.
It provides a practical Attitude and Heading Reference System (AHRS) implementation
based on a multiplicative extended Kalman filter (MEKF), as well as a clean abstraction
for attitude representation with clearly defined reference frames and rotation conventions.

## How to install
```
pip install attipy
```

## Quick start

Convert to/from a variaty of attitude representations using the Attitude class:

```python
import attipy as ap
import numpy as np


# From Euler angles to unit quaternion
att = ap.Attitude.from_euler([0.0, 0.0, 0.0])
q = att.as_quaternion()
```


Estimate attitude from IMU measurements (accelerometer and gyroscope) using the AHRS class:

```python
import attipy as ap
import numpy as np


# PVA/IMU reference signals
fs = 10.0  # sampling rate in Hz
*_, f, w = ap.pva_data()

# Add IMU measurement noise
acc_noise_density = 0.001
gyro_noise_density = 0.0001
rng = np.random.default_rng(42)
f += acc_noise_density * np.sqrt(fs) * rng.standard_normal(f.shape)
w += gyro_noise_density * np.sqrt(fs) * rng.standard_normal(w.shape)

# Estimate attitude using AHRS
ahrs = ap.AHRS(fs)
euler_est = []
for f_i, w_i in zip(f, w):
    ahrs.update(f_i, w_i)
    euler_est.append(ahrs.attitude.as_euler())
euler_est = np.asarray(euler_est)
```

Under sustained linear acceleration, attitude estimates can be improved using velocity
measurements as aiding, and heading aiding can be used to correct the yaw angle drift:

```python
import attipy as ap
import numpy as np


fs = 10.0  # sampling rate in Hz
ahrs = ap.AHRS(fs)

*_, vel, euler, f, w = ap.pva_data()
yaw = euler[:, 0]

euler_est = []
for f_i, w_i, v_i, yaw_i in zip(f, w, vel, yaw_i):
    ahrs.update(f_i, w_i, v=v_i, v_var=(0.1, 0.1, 0.1), yaw=yaw_i, yaw_var=1.0)
    euler_est.append(ahrs.attitude.as_euler())
euler_est = np.asarray(euler_est)
```

