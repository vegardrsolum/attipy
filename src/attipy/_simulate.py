from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._ahrs import _gravity_nav
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

    def d2ydt2(self, t):
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
        y = self._y(t)
        dydt = self._dydt(t)
        d2ydt2 = self._d2ydt2(t)

        return y, dydt, d2ydt2


class SineDOF(DOF):
    """
    1D sine wave DOF signal generator.

    Defined as:

        y(t) = A * sin(w * t + phi) + B
        dy(t)/dt = A * w * cos(w * t + phi)
        d2y(t)/dt2 = -A * w^2 * sin(w * t + phi)

    where,

    - A  : Amplitude of the sine wave.
    - w  : Angular frequency of the sine wave.
    - phi: Phase offset of the sine wave.
    - B  : Constant offset of the sine wave.

    Parameters
    ----------
    amp : float, default 1.0
        Amplitude of the sine wave. Default is 1.0.
    freq : float, default 1.0
        Frequency of the sine wave in rad/s. Default is 1.0 rad/s.
    freq_hz : bool, optional
        If True, interpret `omega` as frequency in Hz. If False, interpret as angular
        frequency in radians per second. Default is False.
    phase : float, default 0.0
        Phase offset of the sine wave. Default is 0.0.
    phase_degrees : bool, optional
        If True, interpret `phase` in degrees. If False, interpret in radians.
        Default is False.
    offset : float, default 0.0
        Offset of the sine wave. Default is 0.0.
    """

    def __init__(
        self,
        amp: float = 1.0,
        freq: float = 1.0,
        freq_hz: bool = False,
        phase: float = 0.0,
        phase_degrees: bool = False,
        offset: float = 0.0,
    ) -> None:
        self._amp = amp
        self._w = 2.0 * np.pi * freq if freq_hz else freq
        self._phase = np.deg2rad(phase) if phase_degrees else phase
        self._offset = offset

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        y = self._amp * np.sin(self._w * t + self._phase) + self._offset
        return y

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        dydt = self._amp * self._w * np.cos(self._w * t + self._phase)
        return dydt

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        d2ydt2 = -self._amp * self._w**2 * np.sin(self._w * t + self._phase)
        return d2ydt2


class ConstantDOF(DOF):
    """
    1D constant DOF signal generator.

    Defined as:

        y(t) = C
        dy(t)/dt = 0
        d2y(t)/dt2 = 0

    where,

    - C : Constant value of the signal.

    Parameters
    ----------
    value : float, default 0.0
        Constant value of the signal. Default is 0.0.
    """

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full_like(t, self._value)

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(t)

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(t)


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


class ChirpDOF(DOF):
    """
    1D chirp sinusoidal DOF signal generator.

    This class creates a signal with a frequency that appears to vary in time.
    The frequency oscillates between 0. and a maximum frequency at a specific
    rate.

    Defined as:

        phi = 2 * f_max / f_os * sin(f_os * t)
        y = sin(phi + phase)

    where,

    - f_max : Maximum frequency the signal ramps up to, before ramping down to zero.
    - f_os  : Frequency oscillation rate.
    - phase : Phase offset of the chirp signal.

    Parameters
    ----------
    f_max : float
        The maximum frequency of the chirp signal, y(t). Default is 0.25 Hz.
    f_os : float
        The frequency oscillation rate. Default is 0.01 Hz.
    freq_hz : bool, default False.
        Whether the frequencies, ``f_max`` and ``f_os``, are in Hz or rad/s (default).
    phase : float, default 0.0
        Phase offset of the chirp signal. Default is 0.0.
    phase_degrees : bool, optional
        If True, interpret `phase` in degrees. If False, interpret in radians.
        Default is False.
    offset : float, default 0.0
        Offset of the chirp signal. Default is 0.0.
    """

    def __init__(
        self,
        amp: float = 1.0,
        f_max: float = 0.5 * np.pi,
        f_os: float = 0.02 * np.pi,
        freq_hz: bool = False,
        phase: float = 0.0,
        phase_degrees: bool = False,
        offset: float = 0.0,
    ) -> None:
        self._amp = amp
        self._w_max = 2.0 * np.pi * f_max if freq_hz else f_max
        self._w_os = 2.0 * np.pi * f_os if freq_hz else f_os
        self._phase = np.deg2rad(phase) if phase_degrees else phase
        self._offset = offset

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        amp = self._amp
        w_max = self._w_max
        w_os = self._w_os
        phase = self._phase
        offset = self._offset

        phi = 2.0 * w_max / w_os * np.sin(w_os / 2.0 * t)
        y = amp * np.sin(phi + phase) + offset
        return y  # type: ignore[no-any-return]

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        amp = self._amp
        w_max = self._w_max
        w_os = self._w_os
        phase = self._phase

        phi = 2.0 * w_max / w_os * np.sin(w_os / 2.0 * t)
        dphi = w_max * np.cos(w_os / 2.0 * t)
        dydt = amp * dphi * np.cos(phi + phase)
        return dydt  # type: ignore[no-any-return]

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        amp = self._amp
        w_max = self._w_max
        w_os = self._w_os
        phase = self._phase

        phi = 2.0 * w_max / w_os * np.sin(w_os / 2.0 * t)
        dphi = w_max * np.cos(w_os / 2.0 * t)
        d2phi = -w_max * w_os / 2.0 * np.sin(w_os / 2.0 * t)
        d2ydt2 = -amp * (dphi**2) * np.sin(phi + phase) + amp * d2phi * np.cos(
            phi + phase
        )
        return d2ydt2  # type: ignore[no-any-return]


def _specific_force_body(
    pos: NDArray[np.float64],
    acc: NDArray[np.float64],
    euler: NDArray[np.float64],
    g_n: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Specific force in the body frame.

    Parameters
    ----------
    pos : ndarray, shape (n, 3)
        Position (x, y, z) in meters.
    vel : ndarray, shape (n, 3)
        Velocity (x_dot, y_dot, z_dot) in meters per second.
    acc : ndarray, shape (n, 3)
        Acceleration (x_ddot, y_ddot, z_ddot) in meters per second squared.
    euler : ndarray, shape (n, 3)
        Euler angles (roll, pitch, yaw) in radians.
    """
    n = pos.shape[0]
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
    Generate position, velocity and attitude (PVA) data, and corresponding IMU data
    (specific force and angular rate).

    Parameters
    ----------
    fs : float, default 10.0
        Sampling frequency in Hz.
    n : int, default 10_000
        Number of samples to generate.
    degrees : bool, optional
        Specifies whether to return Euler angles and angular velocities in degrees
        and degrees per second or radians and radians per second (default).
    g : float, default 9.80665
        The gravitational acceleration. Default is 'standard gravity' of 9.80665.
    nav_frame : str, default "NED"
        Navigation frame. Either 'NED' (North-East-Down) (default) or 'ENU' (East-North-Up).
    type_ : {'standstill', 'beat', 'chirp'}, default 'beat'
        Type of motion to simulate:
        - 'standstill': no motion (stationary).
        - 'beat': beating motion (0.1 Hz main frequency and 0.01 Hz beat frequency).
        - 'chirp': chirp motion (oscillates between 0 and 0.25 Hz at a rate of 0.01 Hz).
        Attitude is +/- 5 degrees and position is +/- 1 meters. Phases are assigned
        to provide variation across all axes.

    Returns
    -------
    t : ndarray
        Time array of shape (n,).
    p_n : ndarray
        Position array of shape (n, 3).
    v_n : ndarray
        Velocity array of shape (n, 3).
    euler_nb : ndarray
        Euler angles array of shape (n, 3).
    f_b : ndarray
        Specific force array of shape (n, 3).
    w_b : ndarray
        Angular rate array of shape (n, 3).
    """

    f_main, f_beat = 0.1, 0.01

    # DOF signals
    pos_amp = 1.0
    att_amp = 0.1
    px_sig = BeatDOF(pos_amp, f_main, f_beat, freq_hz=True, phase=(0 / 3) * np.pi)
    py_sig = BeatDOF(pos_amp, f_main, f_beat, freq_hz=True, phase=(1 / 3) * np.pi)
    pz_sig = BeatDOF(pos_amp, f_main, f_beat, freq_hz=True, phase=(2 / 3) * np.pi)
    roll_sig = BeatDOF(att_amp, f_main, f_beat, freq_hz=True, phase=(3 / 3) * np.pi)
    pitch_sig = BeatDOF(att_amp, f_main, f_beat, freq_hz=True, phase=(4 / 3) * np.pi)
    yaw_sig = BeatDOF(att_amp, f_main, f_beat, freq_hz=True, phase=(5 / 3) * np.pi)

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
    f_b = _specific_force_body(pos, acc, euler, g_n)
    w_b = _angular_velocity_body(euler, euler_dot)

    if degrees:
        euler = np.degrees(euler)
        w_b = np.degrees(w_b)

    return t, pos, vel, euler, f_b, w_b
