from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._mekf import _gravity_nav
from .._transforms import _matrix_from_euler_zyx


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
    Beating signal generator.

    Defined as:

        y = amp * sin(w_beat / 2.0 * t) * cos(w_main * t + phase) + offset

    Parameters
    ----------
    amp : float, optional
        Amplitude of the beat signal. Default is 1.0.
    freq_main : float, optional
        Main frequency of the sinusoidal signal, y(t). Defaults to 0.1 rad/s.
    freq_beat : float, optional
        Beating frequency, controlling the variation in amplitude. Defaults to
        0.01 rad/s.
    freq_hz : bool, optional
        Whether the frequencies, ``freq_main`` and ``freq_beat``, are given in Hz
        or rad/s (default).
    phase : float, optional
        Phase offset of the beat signal. Default is 0.0.
    phase_degrees : bool, optional
        If True, interpret `phase` in degrees. If False, interpret in radians.
        Default is False.
    offset : float, optional
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
        d2ydt2 = amp * (d2beat * main + 2 * dbeat * dmain + beat * d2main)

        return d2ydt2  # type: ignore[no-any-return]


class RampUp(DOF):
    """
    Ramp-up wrapper for DOF signals.

    Scales an underlying DOF signal, y(t), by a smooth ramp-up window, w(t):

        y_rampup(t) = w(t) * y(t)

    The window is zero before the ramp-up starts, increases smoothly from 0 to 1
    during the ramp-up period, and stays at 1 afterwards:

        w(t) = 6 * x**5 - 15 * x**4 + 10 * x**3

    where ``x = (t - start) / duration`` clipped to [0, 1]. The window has
    vanishing first and second derivatives at both ends of the ramp-up period,
    so that the ramped signal and its two first time derivatives are continuous.

    Parameters
    ----------
    dof : DOF
        Underlying DOF signal generator to ramp up.
    duration : float
        Duration of the ramp-up period in seconds. Must be positive.
    start : float, optional
        Time in seconds at which the ramp-up starts. The signal, and its two
        first time derivatives, are zero before this time. Default is 0.0.
    """

    def __init__(self, dof: DOF, duration: float, start: float = 0.0) -> None:
        if duration <= 0.0:
            raise ValueError("'duration' must be positive.")
        if start < 0.0:
            raise ValueError("'start' must be non-negative.")

        self._dof = dof
        self._duration = float(duration)
        self._start = float(start)

    def _window(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Ramp-up window, w(t), and its two first time derivatives.
        """
        duration = self._duration
        x = np.clip((t - self._start) / duration, 0.0, 1.0)

        w = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
        dw = 30.0 * x**2 * (1.0 - x) ** 2 / duration
        d2w = 60.0 * x * (1.0 - 3.0 * x + 2.0 * x**2) / duration**2

        return w, dw, d2w

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        w, _, _ = self._window(t)
        y = self._dof._y(t)
        return w * y

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        w, dw, _ = self._window(t)
        y = self._dof._y(t)
        dydt = self._dof._dydt(t)
        return dw * y + w * dydt

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        w, dw, d2w = self._window(t)
        y = self._dof._y(t)
        dydt = self._dof._dydt(t)
        d2ydt2 = self._dof._d2ydt2(t)
        return d2w * y + 2.0 * dw * dydt + w * d2ydt2


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
        R_i = _matrix_from_euler_zyx(euler[i])
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


def trajectory(
    fs: float = 10.0,
    n: int = 10_000,
    degrees: bool = False,
    g: float = 9.80665,
    nav_frame: str = "NED",
    rampup: float | None = None,
    rampup_start: float = 0.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Generate synthetic, noise-free position, velocity and attitude (PVA) data,
    and corresponding IMU (specific force and angular rate) data.

    The PVA signals are characterized as:
    - Beating sinusoidal motion (0.1 Hz main frequency and 0.01 Hz beat frequency).
    - Position amplitude is +/- 1 meter.
    - Attitude (Euler angle) amplitude is +/- 0.1 radians.
    - Phases are assigned to provide variation across all axes.

    Optionally, a ramp-up period can be applied to gradually increase the amplitude
    of the signals from zero to their full values.

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
    rampup : float or None, optional
        Ramp-up duration in seconds. If ``None`` (default), no ramp-up is applied.
    rampup_start : float, optional
        Start time of the ramp-up period in seconds, i.e., the duration of the initial
        stationary period. Defaults to 0.0 seconds.

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
    px_sig: DOF = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[0])
    py_sig: DOF = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[1])
    pz_sig: DOF = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[2])
    roll_sig: DOF = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[3])
    pitch_sig: DOF = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[4])
    yaw_sig: DOF = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[5])

    if rampup is not None:
        px_sig = RampUp(px_sig, rampup, start=rampup_start)
        py_sig = RampUp(py_sig, rampup, start=rampup_start)
        pz_sig = RampUp(pz_sig, rampup, start=rampup_start)
        roll_sig = RampUp(roll_sig, rampup, start=rampup_start)
        pitch_sig = RampUp(pitch_sig, rampup, start=rampup_start)
        yaw_sig = RampUp(yaw_sig, rampup, start=rampup_start)

    # Time
    dt = 1.0 / fs
    t: NDArray[np.float64] = dt * np.arange(n, dtype=np.float64)

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
