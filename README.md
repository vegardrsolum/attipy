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


# Quaternion to Euler angles
q = [1.0, 0.0, 0.0, 0.0]
euler = ap.Attitude.from_quaternion(q).as_euler()

# Euler angles to quaternion
euler = [np.pi/8, np.pi/4, np.pi/2]
q = ap.Attitude.from_euler(euler).as_quaternion()

# Quaternion to rotation matrix
q = [1.0, 0.0, 0.0, 0.0]
R = ap.Attitude.from_quaternion(q).as_matrix()

# Rotation matrix to Euler angles
R = np.eye(3)
euler = ap.Attitude.from_quaternion(q).as_matrix()

# etc.
```


Estimate attitude from IMU measurements (accelerometer and gyroscope) using the AHRS class:

```python
import attipy as ap
import numpy as np


fs = 10.0  # sampling rate in Hz
ahrs = ap.AHRS(fs)

euler = []
for f_i, w_i in zip(acc, gyro):
    ahrs.update(f_i, w_i)
    euler.append(ahrs.attitude.as_euler())
euler = np.asarray(euler)
```

