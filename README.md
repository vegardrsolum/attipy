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


fs = 10.0  # sampling rate in Hz
ahrs = ap.AHRS(fs)

*_, f, w = ap.pva_data()

euler_est = []
for f_i, w_i in zip(f, w):
    ahrs.update(f_i, w_i)
    euler_est.append(ahrs.attitude.as_euler())
euler_est = np.asarray(euler_est)
```

