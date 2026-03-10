from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._mekf import _gravity_nav
from ._transforms import _matrix_from_euler


class DOF(ABC):
    """
    Abstract base class for degree of freedom (DOF) signal generators.
    """

    @abstractmethod
    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented.")

    @abstractmethod
    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented.")

    def y(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates y(t) signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.
        """
        t = np.asarray_chkfinite(t)
        return self._y(t)

    def dydt(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates dy(t)/dt signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.
        """
        t = np.asarray_chkfinite(t)
        return self._dydt(t)

    def d2ydt2(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates d2y(t)/dt2 signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.
        """
        t = np.asarray_chkfinite(t)
        return self._d2ydt2(t)

    def __call__(
        self, t: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Generates y(t), dy(t)/dt, and d2y(t)/dt2 signals.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.

        Returns
        -------
        y : ndarray, shape (n,)
            DOF signal y(t).
        dydt : ndarray, shape (n,)
            Time derivative, dy(t)/dt, of DOF signal.
        d2ydt2 : ndarray, shape (n,)
            Second time derivative, d2y(t)/dt2, of DOF signal.
        """
        t = np.asarray_chkfinite(t)
        y = self._y(t)
        dydt = self._dydt(t)
        d2ydt2 = self._d2ydt2(t)

        return y, dydt, d2ydt2


class BeatDOF(DOF):
    """
    1D beating sinusoidal DOF signal generator.

    Defined as:

        y = A * sin(f_beat / 2.0 * t) * cos(f_main * t + phi) + B

    where,

    - A      : Amplitude of the sine waves.
    - w_main : Angular frequency of the main sine wave.
    - w_beat : Angular frequency of the beat sine wave.
    - phi    : Phase offset of the main sine wave.
    - B      : Constant offset of the beat signal.

    Parameters
    ----------
    f_main : float
        The main frequency of the sinusoidal signal, y(t).
    f_beat : float
        The beating frequency, which controls the variation in amplitude.
    freq_hz : bool, default True.
        Whether the frequencies, ``f_main`` and ``f_beat``, are in Hz or rad/s (default).
    phase : float, default 0.0
        Phase offset of the beat signal. Default is 0.0.
    phase_degrees : bool, optional
        If True, interpret `phase` in degrees. If False, interpret in radians.
        Default is False.
    offset : float, default 0.0
        Offset of the beat signal. Default is 0.0.
    """

    def __init__(
        self,
        amp: float = 1.0,
        freq_main: float = 0.1,
        freq_beat: float = 0.01,
        freq_hz: bool = False,
        phase: float = 0.0,
        phase_degrees: bool = False,
        offset: float = 0.0,
    ) -> None:
        self._amp = amp
        self._w_main = 2.0 * np.pi * freq_main if freq_hz else freq_main
        self._w_beat = 2.0 * np.pi * freq_beat if freq_hz else freq_beat
        self._phase = np.deg2rad(phase) if phase_degrees else phase
        self._offset = offset

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        amp = self._amp
        w_main = self._w_main
        w_beat = self._w_beat
        phase = self._phase
        offset = self._offset

        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)
        y = amp * beat * main + offset
        return y  # type: ignore[no-any-return]

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        amp = self._amp
        w_main = self._w_main
        w_beat = self._w_beat
        phase = self._phase

        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = w_beat / 2.0 * np.cos(w_beat / 2.0 * t)

        dydt = amp * (dbeat * main + beat * dmain)
        return dydt  # type: ignore[no-any-return]

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:

        amp = self._amp
        w_main = self._w_main
        w_beat = self._w_beat
        phase = self._phase

        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = w_beat / 2.0 * np.cos(w_beat / 2.0 * t)
        d2main = -(w_main**2) * np.cos(w_main * t + phase)
        d2beat = -((w_beat / 2.0) ** 2) * np.sin(w_beat / 2.0 * t)
        d2ydt2 = amp * (dbeat * dmain + d2beat * main + beat * d2main + dbeat * dmain)

        return d2ydt2  # type: ignore[no-any-return]


def _specific_force_body(
    acc: NDArray[np.float64],
    euler: NDArray[np.float64],
    g_n: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Specific force in the body frame.

    Parameters
    ----------
    acc : ndarray, shape (n, 3)
        Acceleration (x_ddot, y_ddot, z_ddot) in meters per second squared.
    euler : ndarray, shape (n, 3)
        Euler angles (roll, pitch, yaw) in radians.
    g_n : ndarray, shape (3,)
        Gravity vector expressed in the navigation frame.
    """
    n = acc.shape[0]
    f_b = np.zeros((n, 3))

    for i in range(n):
        R_i = _matrix_from_euler(euler[i])
        f_b[i] = R_i.T.dot(acc[i] - g_n)

    return f_b


def _angular_velocity_body(
    euler: NDArray[np.float64], euler_dot: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Angular velocity in the body frame.

    Parameters
    ----------
    euler : ndarray, shape (n, 3)
        Euler angles (roll, pitch, yaw) in radians.
    euler_dot : ndarray, shape (n, 3)
        Time derivatives of Euler angles (roll_dot, pitch_dot, yaw_dot)
        in radians per second.
    """
    roll, pitch, _ = euler.T
    roll_dot, pitch_dot, yaw_dot = euler_dot.T

    w_x = roll_dot - np.sin(pitch) * yaw_dot
    w_y = np.cos(roll) * pitch_dot + np.sin(roll) * np.cos(pitch) * yaw_dot
    w_z = -np.sin(roll) * pitch_dot + np.cos(roll) * np.cos(pitch) * yaw_dot

    w_b = np.column_stack([w_x, w_y, w_z])

    return w_b


def pva_sim(
    fs: float = 10.0,
    n: int = 10_000,
    degrees: bool = False,
    g: float = 9.80665,
    nav_frame: str = "NED",
):
    """
    Generate position, velocity and attitude (PVA) signals, and corresponding IMU
    signals (specific force and angular rate).

    The PVA signals are characterized as:
    - Beating sinusoidal motion (0.1 Hz main frequency and 0.01 Hz beat frequency).
    - Position amplitude is +/- 1 meter.
    - Attitude (Euler angle) amplitude is +/- 0.1 radians.
    - Phases are assigned to provide variation across all axes.

    Parameters
    ----------
    fs : float, optional
        Sampling frequency in Hz. Defaults to 10.0 Hz.
    n : int, optional
        Number of samples to generate. Defaults to 10 000.
    degrees : bool, optional
        Specifies whether to return the Euler angles and the angular velocities
        in degrees and degrees per second or radians and radians per second (default).
    g : float, optional
        The gravitational acceleration in m/s^2. Defaults to the 'standard gravity'
        of 9.80665 m/s^2.
    nav_frame : {'NED', 'ENU'}, optional
        Specifies the navigation frame. Either 'NED' (North-East-Down) or 'ENU'
        (East-North-Up). Defaults to 'NED'.

    Returns
    -------
    t : ndarray, shape (n,)
        Time in seconds.
    p_n : ndarray, shape (n, 3)
        Position timeseries in m.
    v_n : ndarray, shape (n, 3)
        Velocity timeseries in m/s.
    euler_nb : ndarray, shape (n, 3)
        Euler angle timeseries in radians (default) or degrees.
    f_b : ndarray, shape (n, 3)
        Specific force timeseries in m/s^2.
    w_b : ndarray, shape (n, 3)
        Angular rate timeseries in rad/s (default) or deg/s.
    """

    f_main, f_beat = 0.1, 0.01

    # DOF signals
    phases = np.linspace(0, 2.0 * np.pi, 6, endpoint=False)
    px_sig = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[0])
    py_sig = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[1])
    pz_sig = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[2])
    roll_sig = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[3])
    pitch_sig = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[4])
    yaw_sig = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[5])

    # Time
    dt = 1.0 / fs
    t = dt * np.arange(n)

    # DOF timeseries and corresponding accelerations and rotation rates
    px, px_dot, px_ddot = px_sig(t)
    py, py_dot, py_ddot = py_sig(t)
    pz, pz_dot, pz_ddot = pz_sig(t)
    roll, roll_dot, _ = roll_sig(t)
    pitch, pitch_dot, _ = pitch_sig(t)
    yaw, yaw_dot, _ = yaw_sig(t)

    pos = np.column_stack([px, py, pz])
    vel = np.column_stack([px_dot, py_dot, pz_dot])
    acc = np.column_stack([px_ddot, py_ddot, pz_ddot])
    euler = np.column_stack([roll, pitch, yaw])
    euler_dot = np.column_stack([roll_dot, pitch_dot, yaw_dot])

    # IMU measurements (i.e., specific force and angular velocity in body frame)
    g_n = _gravity_nav(g, nav_frame.lower())
    f_b = _specific_force_body(acc, euler, g_n)
    w_b = _angular_velocity_body(euler, euler_dot)

    if degrees:
        euler = np.degrees(euler)
        w_b = np.degrees(w_b)

    return t, pos, vel, euler, f_b, w_b
