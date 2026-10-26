from dataclasses import dataclass, field, fields
from importlib import resources

import numpy as np
from numpy.typing import NDArray

from .._frames import _gravity_nav
from .._transforms import _matrix_from_euler_zyx_batch
from ._dof import DOF, Beat, Constant, dofs_from_csd, multiply

_PACKAGE_PATH = resources.files(__package__)


def _load_csd(path: str) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Load a frequency array, shape (m,), and a cross-spectral density (CSD)
    matrix, shape (m, n, n), from an .npz file.
    """
    with open(path, "rb") as fh:
        data = np.load(fh)
        f = np.asarray(data["f"], dtype=np.float64)
        csd = np.asarray(data["csd"], dtype=np.complex128)
    return f, csd


@dataclass(frozen=True, kw_only=True, slots=True)
class Motion:
    """
    Rigid body motion defined by six DOF signal generators.

    Defaults to a stationary/standstill motion with all DOFs set to zero.

    Parameters
    ----------
    x : dof.DOF, optional
        Position along the x-axis in meters. Defaults to ``dof.Constant(0.0)``.
    y : dof.DOF, optional
        Position along the y-axis in meters. Defaults to ``dof.Constant(0.0)``.
    z : dof.DOF, optional
        Position along the z-axis in meters. Defaults to ``dof.Constant(0.0)``.
    roll : dof.DOF, optional
        Roll Euler angle in radians. Defaults to ``dof.Constant(0.0)``.
    pitch : dof.DOF, optional
        Pitch Euler angle in radians. Defaults to ``dof.Constant(0.0)``.
    yaw : dof.DOF, optional
        Yaw Euler angle in radians. Defaults to ``dof.Constant(0.0)``.
    """

    x: DOF = field(default_factory=Constant)
    y: DOF = field(default_factory=Constant)
    z: DOF = field(default_factory=Constant)
    roll: DOF = field(default_factory=Constant)
    pitch: DOF = field(default_factory=Constant)
    yaw: DOF = field(default_factory=Constant)

    def __post_init__(self) -> None:
        for f in fields(self):
            if not isinstance(getattr(self, f.name), DOF):
                raise TypeError(f"'{f.name}' must be a DOF instance.")

    @classmethod
    def from_preset(cls, preset: str, nav_frame: str = "NED") -> "Motion":
        """
        Create a Motion instance from a preset.

        Parameters
        ----------
        preset : str
            Preset name (case-insensitive). One of:

            - 'stationary': No motion; all degrees of freedom remain constant at
              the origin.
            - 'beat-6dof': Beating sinusoidal motion in all six degrees of freedom.
            - 'beat-3dof': Beating sinusoidal motion in roll, pitch and yaw only.
            - 'vessel-6dof': Wave-induced vessel response in all six degrees of
              freedom.
            - 'vessel-3dof': Wave-induced vessel response in roll, pitch and yaw
              only.
        nav_frame : {'NED', 'ENU'}, optional
            Navigation frame in which the motion is defined relative to. Either
            'NED' (North-East-Down) or 'ENU' (East-North-Up). Defaults to 'NED'.

        Returns
        -------
        Motion
            Motion instance corresponding to the specified preset and navigation frame.
        """

        if preset.lower() == "stationary":
            x, y, z, roll, pitch, yaw = _stationary()
        elif preset.lower() == "beat-6dof":
            x, y, z, roll, pitch, yaw = _beat_6dof()
        elif preset.lower() == "beat-3dof":
            x, y, z, roll, pitch, yaw = _beat_3dof()
        elif preset.lower() == "vessel-6dof":
            x, y, z, roll, pitch, yaw = _vessel_6dof()
        elif preset.lower() == "vessel-3dof":
            x, y, z, roll, pitch, yaw = _vessel_3dof()
        else:
            raise ValueError(f"Unknown motion preset '{preset}'.")

        if nav_frame.lower() == "enu":
            y = multiply(-1, y)
            z = multiply(-1, z)
            pitch = multiply(-1, pitch)
            yaw = multiply(-1, yaw)
        elif nav_frame.lower() != "ned":
            raise ValueError(f"Unknown navigation frame '{nav_frame}'.")

        return cls(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)


def _stationary() -> tuple[DOF, DOF, DOF, DOF, DOF, DOF]:
    x = Constant(0.0)
    y = Constant(0.0)
    z = Constant(0.0)
    roll = Constant(0.0)
    pitch = Constant(0.0)
    yaw = Constant(0.0)
    return x, y, z, roll, pitch, yaw


def _beat_6dof() -> tuple[DOF, DOF, DOF, DOF, DOF, DOF]:
    w_main = 2.0 * np.pi * 0.1
    w_beat = 2.0 * np.pi * 0.01
    phases = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    x = Beat(amp=1.0, omega=w_main, omega_beat=w_beat, phase=phases[0])
    y = Beat(amp=1.0, omega=w_main, omega_beat=w_beat, phase=phases[1])
    z = Beat(amp=1.0, omega=w_main, omega_beat=w_beat, phase=phases[2])
    roll = Beat(amp=0.1, omega=w_main, omega_beat=w_beat, phase=phases[3])
    pitch = Beat(amp=0.1, omega=w_main, omega_beat=w_beat, phase=phases[4])
    yaw = Beat(amp=0.1, omega=w_main, omega_beat=w_beat, phase=phases[5])
    return x, y, z, roll, pitch, yaw


def _beat_3dof() -> tuple[DOF, DOF, DOF, DOF, DOF, DOF]:
    x = Constant(0.0)
    y = Constant(0.0)
    z = Constant(0.0)
    roll, pitch, yaw = _beat_6dof()[3:]
    return x, y, z, roll, pitch, yaw


def _vessel_6dof() -> tuple[DOF, DOF, DOF, DOF, DOF, DOF]:
    csd_path = _PACKAGE_PATH.joinpath("_data", "supply_vessel_csd.npz")
    f, csd = _load_csd(str(csd_path))
    x, y, z, roll, pitch, yaw = dofs_from_csd(f, csd, nbins=50, jitter=1.0, seed=1)
    return x, y, z, roll, pitch, yaw


def _vessel_3dof() -> tuple[DOF, DOF, DOF, DOF, DOF, DOF]:
    x = Constant(0.0)
    y = Constant(0.0)
    z = Constant(0.0)
    roll, pitch, yaw = _vessel_6dof()[3:]
    return x, y, z, roll, pitch, yaw


def _specific_force_body(
    a_n: NDArray[np.float64],
    euler_nb: NDArray[np.float64],
    g_n: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Specific force in the body frame.

    Parameters
    ----------
    a_n : ndarray, shape (n, 3)
        Acceleration (ax, ay, az) in meters per second squared, expressed in the
        navigation frame.
    euler_nb : ndarray, shape (n, 3)
        Euler angles (roll, pitch, yaw) in radians.
    g_n : ndarray, shape (3,)
        Gravity vector expressed in the navigation frame.

    Returns
    -------
    ndarray, shape (n, 3)
        Specific force in meters per second squared, expressed in the body frame.
    """
    R_nb = _matrix_from_euler_zyx_batch(euler_nb)
    f_b: NDArray[np.float64] = np.einsum("nji,nj->ni", R_nb, a_n - g_n)

    return f_b


def _angular_velocity_body(
    euler_nb: NDArray[np.float64], euler_nb_dot: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Angular velocity in the body frame.

    Parameters
    ----------
    euler_nb : ndarray, shape (n, 3)
        Euler angles (roll, pitch, yaw) in radians.
    euler_nb_dot : ndarray, shape (n, 3)
        Time derivatives of Euler angles (roll_dot, pitch_dot, yaw_dot)
        in radians per second.

    Returns
    -------
    ndarray, shape (n, 3)
        Angular velocity in radians per second, expressed in the body frame.
    """
    roll, pitch, _ = euler_nb.T
    roll_dot, pitch_dot, yaw_dot = euler_nb_dot.T

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
    p_n : ndarray, shape (n, 3)
        Position timeseries in m, expressed in the navigation frame.
    v_n : ndarray, shape (n, 3)
        Velocity timeseries in m/s, expressed in the navigation frame.
    a_n : ndarray, shape (n, 3)
        Acceleration timeseries in m/s^2, expressed in the navigation frame.
    euler_nb : ndarray, shape (n, 3)
        Euler angle (roll, pitch, yaw) timeseries in radians.
    euler_nb_dot : ndarray, shape (n, 3)
        Euler angle rate timeseries in radians per second.
    """
    pos_sig = [dof(t) for dof in (motion.x, motion.y, motion.z)]
    att_sig = [dof(t) for dof in (motion.roll, motion.pitch, motion.yaw)]

    p_n = np.column_stack([y for y, _, _ in pos_sig])
    v_n = np.column_stack([dydt for _, dydt, _ in pos_sig])
    a_n = np.column_stack([d2ydt2 for _, _, d2ydt2 in pos_sig])
    euler_nb = np.column_stack([y for y, _, _ in att_sig])
    euler_nb_dot = np.column_stack([dydt for _, dydt, _ in att_sig])

    return p_n, v_n, a_n, euler_nb, euler_nb_dot


def trajectory(
    fs: float = 10.0,
    n: int = 10_000,
    *,
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
        Sampling frequency in Hz. Defaults to 10.0 Hz.
    n : int, optional
        Number of samples to generate. Defaults to 10_000.
    g : float, optional
        The gravitational acceleration in m/s^2. Defaults to the 'standard gravity'
        of 9.80665 m/s^2.
    nav_frame : {'NED', 'ENU'}, optional
        Specifies the navigation frame. Either 'NED' (North-East-Down) or 'ENU'
        (East-North-Up). Defaults to 'NED'.
    motion : str or Motion, optional
        Specifies the type of motion to generate. Either a custom ``Motion``
        instance, or a string naming one of the predefined types of motion (see
        ``Motion.from_preset``). Presets are adapted to ``nav_frame``, while
        ``Motion`` instances are used as-is and must therefore be defined relative
        to ``nav_frame``. Defaults to 'beat-6dof'.

    Returns
    -------
    t : ndarray, shape (n,)
        Time in seconds.
    p_n : ndarray, shape (n, 3)
        Position timeseries in m, expressed in the navigation frame.
    v_n : ndarray, shape (n, 3)
        Velocity timeseries in m/s, expressed in the navigation frame.
    euler_nb : ndarray, shape (n, 3)
        Euler angle timeseries in radians.
    f_b : ndarray, shape (n, 3)
        Specific force timeseries in m/s^2, expressed in the body frame.
    w_b : ndarray, shape (n, 3)
        Angular rate timeseries in rad/s, expressed in the body frame.
    """

    if fs <= 0.0:
        raise ValueError("'fs' must be positive.")
    if n != int(n):
        raise ValueError("'n' must be a whole number of samples.")
    if n <= 0:
        raise ValueError("'n' must be positive.")

    g_n = _gravity_nav(g, nav_frame)

    if not isinstance(motion, Motion):
        motion = Motion.from_preset(motion, nav_frame=nav_frame)

    # Time
    dt = 1.0 / fs
    t: NDArray[np.float64] = dt * np.arange(n, dtype=np.float64)

    # Kinematics
    p_n, v_n, a_n, euler_nb, euler_nb_dot = _sample_motion(motion, t)

    # IMU signals from kinematics
    f_b = _specific_force_body(a_n, euler_nb, g_n)
    w_b = _angular_velocity_body(euler_nb, euler_nb_dot)

    return t, p_n, v_n, euler_nb, f_b, w_b
