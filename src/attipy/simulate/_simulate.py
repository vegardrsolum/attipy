from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from .._mekf import _gravity_nav
from .._transforms import _matrix_from_euler_zyx_batch
from ._dof import DOF, BeatDOF, ConstantDOF


@dataclass(frozen=True, kw_only=True, slots=True)
class Motion:
    """
    Rigid body motion defined by six independent DOF signal generators.

    All parameters are keyword-only. Instances are immutable.

    Parameters
    ----------
    x : DOF, optional
        Position along the x-axis in meters. Defaults to ``ConstantDOF(0.0)``.
    y : DOF, optional
        Position along the y-axis in meters. Defaults to ``ConstantDOF(0.0)``.
    z : DOF, optional
        Position along the z-axis in meters. Defaults to ``ConstantDOF(0.0)``.
    roll : DOF, optional
        Roll angle in radians (default) or degrees. Defaults to ``ConstantDOF(0.0)``.
    pitch : DOF, optional
        Pitch angle in radians (default) or degrees. Defaults to ``ConstantDOF(0.0)``.
    yaw : DOF, optional
        Yaw angle in radians (default) or degrees. Defaults to ``ConstantDOF(0.0)``.
    degrees : bool, optional
        Specifies whether the angular DOF signals, ``roll``, ``pitch`` and ``yaw``,
        are given in degrees or radians (default).
    """

    x: DOF = field(default_factory=ConstantDOF)
    y: DOF = field(default_factory=ConstantDOF)
    z: DOF = field(default_factory=ConstantDOF)
    roll: DOF = field(default_factory=ConstantDOF)
    pitch: DOF = field(default_factory=ConstantDOF)
    yaw: DOF = field(default_factory=ConstantDOF)
    degrees: bool = False

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "roll", "pitch", "yaw"):
            if not isinstance(getattr(self, name), DOF):
                raise TypeError(f"'{name}' must be a DOF instance.")


_BEAT6DOF = Motion(
    x=BeatDOF(1.0, 0.1, 0.01, freq_hz=True, phase=0.0),
    y=BeatDOF(1.0, 0.1, 0.01, freq_hz=True, phase=np.pi / 3),
    z=BeatDOF(1.0, 0.1, 0.01, freq_hz=True, phase=2 * np.pi / 3),
    roll=BeatDOF(0.1, 0.1, 0.01, freq_hz=True, phase=np.pi),
    pitch=BeatDOF(0.1, 0.1, 0.01, freq_hz=True, phase=4 * np.pi / 3),
    yaw=BeatDOF(0.1, 0.1, 0.01, freq_hz=True, phase=5 * np.pi / 3),
    degrees=False,
)


_BEAT3DOF = replace(
    _BEAT6DOF,
    x=ConstantDOF(0.0),
    y=ConstantDOF(0.0),
    z=ConstantDOF(0.0),
)


_STATIONARY = Motion()


_MOTION_PRESETS = {
    "beat-6dof": _BEAT6DOF,
    "beat-3dof": _BEAT3DOF,
    "stationary": _STATIONARY,
}


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


def _sample_motion(motion: Motion, t: NDArray[np.float64]) -> tuple[
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
    motion : Motion
        Rigid body motion to sample.
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
    pos_sig = [dof(t) for dof in (motion.x, motion.y, motion.z)]
    att_sig = [dof(t) for dof in (motion.roll, motion.pitch, motion.yaw)]

    pos = np.column_stack([y for y, _, _ in pos_sig])
    vel = np.column_stack([dydt for _, dydt, _ in pos_sig])
    acc = np.column_stack([d2ydt2 for _, _, d2ydt2 in pos_sig])
    euler = np.column_stack([y for y, _, _ in att_sig])
    euler_dot = np.column_stack([dydt for _, dydt, _ in att_sig])

    if motion.degrees:
        euler = np.radians(euler)
        euler_dot = np.radians(euler_dot)

    return pos, vel, acc, euler, euler_dot


def _imu_from_kinematics(
    acc: NDArray[np.float64],
    euler: NDArray[np.float64],
    euler_dot: NDArray[np.float64],
    g: float,
    nav_frame: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Noise-free IMU measurements corresponding to given rigid body kinematics.

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


def trajectory(
    fs: float = 10.0,
    n: int = 10_000,
    degrees: bool = False,
    g: float = 9.80665,
    nav_frame: str = "NED",
    motion: str | Motion = "beat-6dof",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Generate synthetic, noise-free position, velocity and attitude (PVA) signals,
    and corresponding IMU (specific force and angular rate) signals.

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
    motion : str or Motion, optional
        Specifies the type of motion to generate. Either a string specifying one
        of the predefined types of motion:

        - 'stationary': No motion; all degrees of freedom remain constant at the origin.
        - 'beat-6dof': Beating sinusoidal motion in all six degrees of freedom.
        - 'beat-3dof': Beating sinusoidal motion in roll, pitch and yaw only.

        or a custom ``Motion`` instance. Defaults to 'beat-6dof'.

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

    if not isinstance(motion, Motion):
        try:
            motion = _MOTION_PRESETS[motion.lower()]
        except (KeyError, AttributeError):
            raise ValueError(f"Unknown motion type: {motion!r}")

    # Time
    dt = 1.0 / fs
    t: NDArray[np.float64] = dt * np.arange(n, dtype=np.float64)

    # PVA and IMU signals
    pos, vel, acc, euler, euler_dot = _sample_motion(motion, t)
    f_b, w_b = _imu_from_kinematics(acc, euler, euler_dot, g, nav_frame)

    if degrees:
        euler = np.degrees(euler)
        w_b = np.degrees(w_b)

    return t, pos, vel, euler, f_b, w_b
