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


class PVASimulator:
    """
    Position, velocity and attitude (PVA) and IMU simulator.

    Parameters
    ----------
    px : float or DOF, default 0.0
        X position signal.
    py : float or DOF, default 0.0
        Y position signal.
    pz : float or DOF, default 0.0
        Z position signal.
    roll : float or DOF, default 0.0
        Roll signal.
    pitch : float or DOF, default 0.0
        Pitch signal
    yaw : float or DOF, default 0.0
        Yaw signal
    degrees: bool, default False
        Whether to interpret the Euler angle signals as degrees (True) or radians (False).
        Default is False.
    g : float, default 9.80665
        The gravitational acceleration. Default is 'standard gravity' of 9.80665.
    nav_frame : str, default "NED"
        Navigation frame. Either "NED" (North-East-Down) or "ENU" (East-North-Up).
        Default is "NED".
    """

    def __init__(
        self,
        px: float | DOF = 0.0,
        py: float | DOF = 0.0,
        pz: float | DOF = 0.0,
        roll: float | DOF = 0.0,
        pitch: float | DOF = 0.0,
        yaw: float | DOF = 0.0,
        degrees: bool = False,
        g: float = 9.80665,
        nav_frame: str = "NED",
    ) -> None:
        self._px = px if isinstance(px, DOF) else ConstantDOF(px)
        self._py = py if isinstance(py, DOF) else ConstantDOF(py)
        self._pz = pz if isinstance(pz, DOF) else ConstantDOF(pz)
        self._roll = roll if isinstance(roll, DOF) else ConstantDOF(roll)
        self._pitch = pitch if isinstance(pitch, DOF) else ConstantDOF(pitch)
        self._yaw = yaw if isinstance(yaw, DOF) else ConstantDOF(yaw)
        self._degrees = degrees
        self._nav_frame = nav_frame.lower()
        self._g_n = _gravity_nav(g, self._nav_frame)

    def __call__(self, fs: float, n: int, degrees: bool | None = None):
        """
        Generate a length-n gyroscope signal and corresponding Euler angles.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        n : int
            Number of samples to generate.
        degrees : bool, optional
            Whether to return Euler angles and angular velocities in degrees and
            degrees per second (True) or radians and radians per second (False).
            Defaults to the value specified at initialization.

        Returns
        -------
        t : ndarray, shape (n,)
            Time vector in seconds.
        euler : ndarray, shape (n, 3)
            Simulated (ZYX) Euler angles (roll, pitch, yaw).
        w_b : ndarray, shape (n, 3)
            Simulated angular velocities, (w_x, w_y, w_z), in the body frame.
        """
        if degrees is None:
            degrees = self._degrees

        # Time
        dt = 1.0 / fs
        t = dt * np.arange(n)

        # DOFs and corresponding rates and accelerations
        px, px_dot, px_ddot = self._px(t)
        py, py_dot, py_ddot = self._py(t)
        pz, pz_dot, pz_ddot = self._pz(t)
        roll, roll_dot, _ = self._roll(t)
        pitch, pitch_dot, _ = self._pitch(t)
        yaw, yaw_dot, _ = self._yaw(t)

        pos = np.column_stack([px, py, pz])
        vel = np.column_stack([px_dot, py_dot, pz_dot])
        acc = np.column_stack([px_ddot, py_ddot, pz_ddot])
        euler = np.column_stack([roll, pitch, yaw])
        euler_dot = np.column_stack([roll_dot, pitch_dot, yaw_dot])

        if self._degrees:
            euler = np.deg2rad(euler)
            euler_dot = np.deg2rad(euler_dot)

        # IMU measurements (i.e., specific force and angular velocity in body frame)
        f_b = _specific_force_body(pos, acc, euler, self._g_n)
        w_b = _angular_velocity_body(euler, euler_dot)

        if degrees:
            euler = np.rad2deg(euler)
            w_b = np.rad2deg(w_b)

        return t, pos, vel, euler, f_b, w_b


def _beat_sim(g, nav_frame):
    """Create a beating PVA simulator."""
    f_main, f_beat = 0.1, 0.01

    amp_att = np.radians(5.0)
    amp_pos = 1.0
    phases_att = (0.0, 1 * np.pi / 3, 2 * np.pi / 3)
    phases_pos = (3 * np.pi / 3, 4 * np.pi / 3, 5 * np.pi / 3)

    sim = PVASimulator(
        px=BeatDOF(amp_pos, f_main, f_beat, freq_hz=True, phase=phases_pos[0]),
        py=BeatDOF(amp_pos, f_main, f_beat, freq_hz=True, phase=phases_pos[1]),
        pz=BeatDOF(amp_pos, f_main, f_beat, freq_hz=True, phase=phases_pos[2]),
        roll=BeatDOF(amp_att, f_main, f_beat, freq_hz=True, phase=phases_att[0]),
        pitch=BeatDOF(amp_att, f_main, f_beat, freq_hz=True, phase=phases_att[1]),
        yaw=BeatDOF(amp_att, f_main, f_beat, freq_hz=True, phase=phases_att[2]),
        degrees=False,
        g=g,
        nav_frame=nav_frame,
    )

    return sim


def _chirp_sim(g, nav_frame):
    """Create a chirp PVA simulator."""
    f_max, f_os = 0.25, 0.01

    amp_att = np.radians(5.0)
    amp_pos = 1.0
    phases_att = (0.0, 1 * np.pi / 3, 2 * np.pi / 3)
    phases_pos = (3 * np.pi / 3, 4 * np.pi / 3, 5 * np.pi / 3)

    sim = PVASimulator(
        px=ChirpDOF(amp_pos, f_max, f_os, freq_hz=True, phase=phases_pos[0]),
        py=ChirpDOF(amp_pos, f_max, f_os, freq_hz=True, phase=phases_pos[1]),
        pz=ChirpDOF(amp_pos, f_max, f_os, freq_hz=True, phase=phases_pos[2]),
        roll=ChirpDOF(amp_att, f_max, f_os, freq_hz=True, phase=phases_att[0]),
        pitch=ChirpDOF(amp_att, f_max, f_os, freq_hz=True, phase=phases_att[1]),
        yaw=ChirpDOF(amp_att, f_max, f_os, freq_hz=True, phase=phases_att[2]),
        degrees=False,
        g=g,
        nav_frame=nav_frame,
    )

    return sim


def pva_sim(
    fs: float = 10.0,
    n: int = 10_000,
    degrees: bool = False,
    g: float = 9.80665,
    nav_frame: str = "NED",
    type_: str = "beat",
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
        Navigation frame. Either "NED" (North-East-Down) or "ENU" (East-North-Up).
        Default is "NED".
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
    if type_.lower() == "standstill":
        sim = PVASimulator(g=g, nav_frame=nav_frame)
    elif type_.lower() == "beat":
        sim = _beat_sim(g, nav_frame)
    elif type_.lower() == "chirp":
        sim = _chirp_sim(g, nav_frame)
    else:
        raise ValueError(f"Unknown simulation type: {type_}")
    return sim(fs, n, degrees=degrees)
