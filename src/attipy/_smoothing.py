import numpy as np


class RTSSmoother:
    """
    Fixed-interval smoothing for MEKF based on the Rauch-Tung-Striebel (RTS) algorithm.
    """

    def __init__(self, mekf):
        self._mekf = mekf

        # Forward sweep buffers
        self._q_buf = []
        self._b_buf = []
        self._P_buf = []
        self._dtheta_buf = []

        # Smoothed state and covariance estimates
        self._q_nb = np.empty((0, 4), dtype="float64")
        self._bg_b = np.empty((0, 3), dtype="float64")
        self._P = np.empty((0, 6, 6), dtype="float64")

    def update(self, *args, **kwargs):
        """
        Update with IMU and aiding measurements.
        """
        self._mekf.update(*args, **kwargs)
        self._q_buf.append(self._mekf.attitude.as_quaternion())
        self._b_buf.append(self._mekf.bias)
        self._P_buf.append(self._mekf.P)
        self._dtheta_buf.append(self._mekf._dtheta.copy())
        return self

    def _smooth(self):
        n_samples = len(self._q_buf)

        if n_samples == 0:
            pass
        elif n_samples == 1:
            self._q_nb = np.array(self._q_buf)
            self._bg_b = np.array(self._b_buf)
            self._P = np.array(self._P_buf)
        elif n_samples != len(self._q_nb):
            self._q_nb, self._bg_b, self._P = _rts_backward_sweep(
                self._q_buf,
                self._b_buf,
                self._P_buf,
                self._dtheta_buf,
                self._mekf._dx,
                self._mekf._phi,
                self._mekf._Q,
                self._mekf._dt,
            )