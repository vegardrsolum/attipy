class RTSSmoother:
    """
    Fixed-interval smoothing for MEKF based on the Rauch-Tung-Striebel (RTS) algorithm.
    """

    def __init__(self, mekf):
        self._mekf = mekf

    def update(self, *args, **kwargs):
        """
        Update with IMU and aiding measurements.
        """
        self._mekf.update(*args, **kwargs)