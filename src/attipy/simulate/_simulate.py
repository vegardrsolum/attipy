from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._mekf import _gravity_nav
from .._transforms import _matrix_from_euler_zyx_batch


class DOF(ABC):
    """
    Abstract base class for degree of freedom (DOF) signal generators.
    """

    @abstractmethod
    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Signal, y(t), and its two first time derivatives, dy(t)/dt and d2y(t)/dt2.
        """
        ...

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        y, _, _ = self._evaluate(t)
        return y

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        _, dydt, _ = self._evaluate(t)
        return dydt

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        _, _, d2ydt2 = self._evaluate(t)
        return d2ydt2

    def y(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates y(t) signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.

        Returns
        -------
        ndarray, shape (n,)
            DOF signal y(t).
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

        Returns
        -------
        ndarray, shape (n,)
            Time derivative, dy(t)/dt, of DOF signal.
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

        Returns
        -------
        ndarray, shape (n,)
            Second time derivative, d2y(t)/dt2, of DOF signal.
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
        return self._evaluate(t)


class BeatDOF(DOF):
    """
    Beating signal generator.

    Defined as:

        y = amp * sin(w_beat / 2.0 * t) * cos(w_main * t + phase)

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
    """

    def __init__(
        self,
        amp: float = 1.0,
        freq_main: float = 0.1,
        freq_beat: float = 0.01,
        freq_hz: bool = False,
        phase: float = 0.0,
        phase_degrees: bool = False,
    ) -> None:
        self._amp = amp
        self._w_main = 2.0 * np.pi * freq_main if freq_hz else freq_main
        self._w_beat = 2.0 * np.pi * freq_beat if freq_hz else freq_beat
        self._phase = np.deg2rad(phase) if phase_degrees else phase

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        amp = self._amp
        w_main = self._w_main
        w_beat = self._w_beat
        phase = self._phase

        arg_main = w_main * t + phase
        arg_beat = w_beat / 2.0 * t

        main = np.cos(arg_main)
        dmain = -w_main * np.sin(arg_main)
        d2main = -(w_main**2) * main

        beat = np.sin(arg_beat)
        dbeat = w_beat / 2.0 * np.cos(arg_beat)
        d2beat = -((w_beat / 2.0) ** 2) * beat

        y = amp * beat * main
        dydt = amp * (dbeat * main + beat * dmain)
        d2ydt2 = amp * (d2beat * main + 2.0 * dbeat * dmain + beat * d2main)

        return y, dydt, d2ydt2


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

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        w, dw, d2w = self._window(t)
        y, dydt, d2ydt2 = self._dof._evaluate(t)

        y_ramped = w * y
        dydt_ramped = dw * y + w * dydt
        d2ydt2_ramped = d2w * y + 2.0 * dw * dydt + w * d2ydt2

        return y_ramped, dydt_ramped, d2ydt2_ramped


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
        Acceleration (ax, ay, az) in meters per second squared.
    euler : ndarray, shape (n, 3)
        Euler angles (roll, pitch, yaw) in radians.
    g_n : ndarray, shape (3,)
        Gravity vector expressed in the navigation frame.

    Returns
    -------
    ndarray, shape (n, 3)
        Specific force in meters per second squared, expressed in the body frame.
    """
    R_nb = _matrix_from_euler_zyx_batch(euler)
    f_b: NDArray[np.float64] = np.einsum("nji,nj->ni", R_nb, acc - g_n)

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

    Returns
    -------
    ndarray, shape (n, 3)
        Angular velocity in radians per second, expressed in the body frame.
    """
    roll, pitch, _ = euler.T
    roll_dot, pitch_dot, yaw_dot = euler_dot.T

    w_x = roll_dot - np.sin(pitch) * yaw_dot
    w_y = np.cos(roll) * pitch_dot + np.sin(roll) * np.cos(pitch) * yaw_dot
    w_z = -np.sin(roll) * pitch_dot + np.cos(roll) * np.cos(pitch) * yaw_dot

    w_b = np.column_stack([w_x, w_y, w_z])

    return w_b


def _motion_from_dofs(dofs: Sequence[DOF], t: NDArray[np.float64]) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Sample six DOF signal generators into rigid body motion timeseries.

    Parameters
    ----------
    dofs : sequence of DOF, length 6
        Signal generators for the six degrees of freedom, in the following order:
        x, y, z, roll, pitch and yaw.
    t : ndarray, shape (n,)
        Time in seconds.

    Returns
    -------
    pos : ndarray, shape (n, 3)
        Position timeseries in m.
    vel : ndarray, shape (n, 3)
        Velocity timeseries in m/s.
    acc : ndarray, shape (n, 3)
        Acceleration timeseries in m/s^2.
    euler : ndarray, shape (n, 3)
        Euler angle (roll, pitch, yaw) timeseries in radians.
    euler_dot : ndarray, shape (n, 3)
        Euler angle rate timeseries in radians per second.
    """
    if len(dofs) != 6:
        raise ValueError("'dofs' must contain exactly six DOF signal generators.")

    pos_sig = [dof(t) for dof in dofs[:3]]
    att_sig = [dof(t) for dof in dofs[3:]]

    pos = np.column_stack([y for y, _, _ in pos_sig])
    vel = np.column_stack([dydt for _, dydt, _ in pos_sig])
    acc = np.column_stack([d2ydt2 for _, _, d2ydt2 in pos_sig])
    euler = np.column_stack([y for y, _, _ in att_sig])
    euler_dot = np.column_stack([dydt for _, dydt, _ in att_sig])

    return pos, vel, acc, euler, euler_dot


def _imu_from_motion(
    acc: NDArray[np.float64],
    euler: NDArray[np.float64],
    euler_dot: NDArray[np.float64],
    g: float,
    nav_frame: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Noise-free IMU measurements corresponding to a given rigid body motion.

    Parameters
    ----------
    acc : ndarray, shape (n, 3)
        Acceleration timeseries in m/s^2, expressed in the navigation frame.
    euler : ndarray, shape (n, 3)
        Euler angle (roll, pitch, yaw) timeseries in radians.
    euler_dot : ndarray, shape (n, 3)
        Euler angle rate timeseries in radians per second.
    g : float
        The gravitational acceleration in m/s^2.
    nav_frame : {'NED', 'ENU'}
        Specifies the navigation frame. Either 'NED' (North-East-Down) or 'ENU'
        (East-North-Up).

    Returns
    -------
    f_b : ndarray, shape (n, 3)
        Specific force timeseries in m/s^2, expressed in the body frame.
    w_b : ndarray, shape (n, 3)
        Angular rate timeseries in rad/s, expressed in the body frame.
    """
    g_n = _gravity_nav(g, nav_frame.lower())

    f_b = _specific_force_body(acc, euler, g_n)
    w_b = _angular_velocity_body(euler, euler_dot)

    return f_b, w_b


def _beating_dofs():
    """
    Beating DOF signals.
    """
    f_main, f_beat = 0.1, 0.01
    phases = np.linspace(0, 2.0 * np.pi, 6, endpoint=False)
    px = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[0])
    py = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[1])
    pz = BeatDOF(1.0, f_main, f_beat, freq_hz=True, phase=phases[2])
    r = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[3])
    p = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[4])
    y = BeatDOF(0.1, f_main, f_beat, freq_hz=True, phase=phases[5])
    return px, py, pz, r, p, y


def trajectory(
    fs: float = 10.0,
    n: int = 10_000,
    degrees: bool = False,
    g: float = 9.80665,
    nav_frame: str = "NED",
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

    Parameters
    ----------
    fs : float, optional
        Sampling frequency in Hz. Must be positive. Defaults to 10.0 Hz.
    n : int, optional
        Number of samples to generate. Must be a positive whole number.
        Defaults to 10 000.
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

    if fs <= 0.0:
        raise ValueError("'fs' must be positive.")
    if n != int(n):
        raise ValueError("'n' must be a whole number of samples.")
    if n <= 0:
        raise ValueError("'n' must be positive.")

    # Time
    dt = 1.0 / fs
    t: NDArray[np.float64] = dt * np.arange(n, dtype=np.float64)

    # PVA and IMU signals
    dofs = _beating_dofs()
    pos, vel, acc, euler, euler_dot = _motion_from_dofs(dofs, t)
    f_b, w_b = _imu_from_motion(acc, euler, euler_dot, g, nav_frame)

    if degrees:
        euler = np.degrees(euler)
        w_b = np.degrees(w_b)

    return t, pos, vel, euler, f_b, w_b
