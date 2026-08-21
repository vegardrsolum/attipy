class RTSSmoother:
    """
    Fixed-interval smoothing for MEKF based on the Rauch-Tung-Striebel (RTS) algorithm.
    """

    def __init__(self, mekf):
        self._mekf = mekf

        self._q_buf = []
        self._b_buf = []
        self._P_buf = []
        self._dtheta_buf = []

    def update(self, *args, **kwargs):
        """
        Update with IMU and aiding measurements.
        """
        self._mekf.update(*args, **kwargs)
        self._q_buf.append(self._mekf.attitude.as_quaternion())
        self._b_buf.append(self._mekf.bias)
        self._P_buf.append(self._mekf.P)
        self._dtheta_buf.append(self._mekf._dtheta.copy())
