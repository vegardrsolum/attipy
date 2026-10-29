import numpy as np
import pytest

import attipy as ap
from attipy._transforms import _matrix_from_euler_zyx, _matrix_from_euler_zyx_batch
from attipy.simulate._dof import DOF, Beat, Constant
from attipy.simulate._simulate import (
    _PACKAGE_PATH,
    Motion,
    _angular_velocity_body,
    _load_csd,
    _sample_motion,
    _specific_force_body,
)

VESSEL_CSD_PATH = str(_PACKAGE_PATH.joinpath("_data", "supply_vessel_csd.npz"))
PRESETS = ("stationary", "beat-6dof", "beat-3dof", "vessel-6dof", "vessel-3dof")
NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


def _eval_motion(motion, t):
    """
    Evaluate all DOFs of a motion. Returns an ndarray of shape (6, 3, n), with
    the DOFs ordered as in ``NAMES``, and (y, dydt, d2ydt2) along the second axis.
    """
    return np.array([getattr(motion, name)(t) for name in NAMES])


def _strapdown_euler(fs, euler0, f_b, w_b):
    """
    Integrate attitude from IMU signals using the MEKF without aiding. Returns
    the Euler angles, shape (n, 3).
    """
    q0 = ap.Attitude.from_euler(euler0, degrees=False).as_quaternion()
    mekf = ap.MEKF(fs, q0=q0)
    euler_est = [euler0]
    for f_i, w_i in zip(f_b[1:], w_b[1:]):
        mekf.update(f_i / fs, w_i / fs, gref=False)
        euler_est.append(mekf.attitude.as_euler())
    return np.array(euler_est)


class Test_load_csd:
    def test_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        f = np.linspace(0.0, 1.0, 10)
        csd = rng.standard_normal((10, 2, 2)) + 1j * rng.standard_normal((10, 2, 2))
        path = tmp_path / "csd.npz"
        np.savez(path, f=f, csd=csd)

        f_out, csd_out = _load_csd(str(path))

        np.testing.assert_array_equal(f_out, f)
        np.testing.assert_array_equal(csd_out, csd)

    def test_casts_dtypes(self, tmp_path):
        path = tmp_path / "csd.npz"
        np.savez(path, f=np.arange(3), csd=np.ones((3, 1, 1)))

        f, csd = _load_csd(str(path))

        assert f.dtype == np.float64
        assert csd.dtype == np.complex128

    def test_vessel_data(self):
        f, csd = _load_csd(VESSEL_CSD_PATH)

        assert f.ndim == 1
        assert csd.shape == (f.size, 6, 6)
        assert np.all(np.diff(f) > 0.0)
        np.testing.assert_allclose(csd, np.conj(np.swapaxes(csd, 1, 2)))
        assert np.all(np.linalg.eigvalsh(csd) > -1e-12)


class Test_Motion:
    def test__init__(self):
        dofs = {name: Constant(float(i)) for i, name in enumerate(NAMES)}
        motion = Motion(**dofs)

        for name in NAMES:
            assert getattr(motion, name) is dofs[name]

    def test__init__default(self):
        t = np.linspace(0.0, 10.0, 100)
        motion = Motion()

        for name in NAMES:
            assert isinstance(getattr(motion, name), Constant)
        np.testing.assert_allclose(_eval_motion(motion, t), 0.0)

    def test__init__defaults_not_shared(self):
        assert Motion().x is not Motion().x

    def test__init__keyword_only(self):
        with pytest.raises(TypeError):
            Motion(Constant())

    @pytest.mark.parametrize("name", NAMES)
    def test__init__non_dof_raises(self, name):
        with pytest.raises(TypeError, match=f"'{name}'"):
            Motion(**{name: 1.0})

    @pytest.mark.parametrize("name", NAMES)
    def test_frozen(self, name):
        motion = Motion()

        with pytest.raises(AttributeError):
            setattr(motion, name, Constant())

    def test_not_iterable(self):
        with pytest.raises(TypeError):
            iter(Motion())

    def test_from_preset_stationary(self):
        t = np.linspace(0.0, 100.0, 1001)
        out = _eval_motion(Motion.from_preset("stationary"), t)

        np.testing.assert_array_equal(out, 0.0)

    def test_from_preset_beat_6dof(self):
        t = np.linspace(0.0, 100.0, 1001)
        amps = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)
        phases = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        beat_kwargs = {"omega": 2 * np.pi * 0.1, "omega_beat": 2 * np.pi * 0.01}
        motion_expect = Motion(
            **{
                name: Beat(amp=amp, phase=phase, **beat_kwargs)
                for name, amp, phase in zip(NAMES, amps, phases)
            }
        )

        out = _eval_motion(Motion.from_preset("beat-6dof"), t)

        np.testing.assert_allclose(out, _eval_motion(motion_expect, t))

    def test_from_preset_vessel_6dof_all_dofs_excited(self):
        t = np.linspace(0.0, 100.0, 1001)
        out = _eval_motion(Motion.from_preset("vessel-6dof"), t)

        assert np.all(np.std(out, axis=-1) > 0.0)

    @pytest.mark.parametrize("preset", ["beat-3dof", "vessel-3dof"])
    def test_from_preset_3dof_matches_6dof_rotations(self, preset):
        t = np.linspace(0.0, 100.0, 1001)
        out_3dof = _eval_motion(Motion.from_preset(preset), t)
        out_6dof = _eval_motion(Motion.from_preset(preset.replace("3dof", "6dof")), t)

        np.testing.assert_allclose(out_3dof[:3], 0.0)
        np.testing.assert_allclose(out_3dof[3:], out_6dof[3:])

    def test_from_preset_vessel_6dof_deterministic(self):
        t = np.linspace(0.0, 100.0, 1001)
        out_a = _eval_motion(Motion.from_preset("vessel-6dof"), t)
        out_b = _eval_motion(Motion.from_preset("vessel-6dof"), t)

        np.testing.assert_array_equal(out_a, out_b)

    def test_from_preset_vessel_6dof_matches_csd(self):
        f, csd = _load_csd(VESSEL_CSD_PATH)
        motion = Motion.from_preset("vessel-6dof")
        dofs = [getattr(motion, name) for name in NAMES]

        omega = sorted({s._w for d in dofs for s in d._dofs})
        z = np.zeros((len(dofs), len(omega)), dtype=np.complex128)
        for i, d in enumerate(dofs):
            for s in d._dofs:
                z[i, omega.index(s._w)] = s._amp * np.exp(1j * s._phase)

        np.testing.assert_allclose(
            z.conj() @ z.T / 2.0, np.trapezoid(csd, f, axis=0), atol=1e-12
        )

    @pytest.mark.parametrize("preset", PRESETS)
    def test_from_preset_nav_frame_enu(self, preset):
        # ENU is a 180 degree rotation about x, which negates y, z, pitch and yaw
        t = np.linspace(0.0, 10.0, 101)
        signs = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])[:, None, None]
        out_ned = _eval_motion(Motion.from_preset(preset, nav_frame="NED"), t)
        out_enu = _eval_motion(Motion.from_preset(preset, nav_frame="ENU"), t)

        np.testing.assert_allclose(out_enu, signs * out_ned)

    def test_from_preset_nav_frame_default_ned(self):
        t = np.linspace(0.0, 10.0, 101)
        out = _eval_motion(Motion.from_preset("beat-6dof"), t)
        out_expect = _eval_motion(Motion.from_preset("beat-6dof", nav_frame="NED"), t)

        np.testing.assert_array_equal(out, out_expect)

    @pytest.mark.parametrize(
        "preset, nav_frame", [("BEAT-6DOF", "ned"), ("Vessel-3dof", "Enu")]
    )
    def test_from_preset_case_insensitive(self, preset, nav_frame):
        t = np.linspace(0.0, 10.0, 101)
        out = _eval_motion(Motion.from_preset(preset, nav_frame=nav_frame), t)
        out_expect = _eval_motion(
            Motion.from_preset(preset.lower(), nav_frame=nav_frame.upper()), t
        )

        np.testing.assert_array_equal(out, out_expect)

    def test_from_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown motion preset"):
            Motion.from_preset("invalid")

    @pytest.mark.parametrize("preset", [None, 1.0, np.array([1, 2])])
    def test_from_preset_non_str_raises(self, preset):
        with pytest.raises(TypeError, match="'preset' must be a string"):
            Motion.from_preset(preset)

    @pytest.mark.parametrize("nav_frame", ["invalid", "NEU"])
    def test_from_preset_nav_frame_raises(self, nav_frame):
        with pytest.raises(ValueError, match="Unknown navigation frame"):
            Motion.from_preset("beat-6dof", nav_frame=nav_frame)


class Test_specific_force_body:
    def test_at_rest_and_level(self):
        g = 9.81
        acc = np.zeros((5, 3))
        euler = np.zeros((5, 3))
        g_n = np.array([0.0, 0.0, g])

        f_b = _specific_force_body(acc, euler, g_n)

        np.testing.assert_allclose(f_b, np.tile([0.0, 0.0, -g], (5, 1)))

    def test_matches_per_sample_rotation(self):
        rng = np.random.default_rng(0)
        acc = rng.standard_normal((100, 3))
        euler = rng.uniform(-np.pi, np.pi, size=(100, 3))
        g_n = np.array([0.0, 0.0, 9.81])

        f_b = _specific_force_body(acc, euler, g_n)

        expected = np.array(
            [
                _matrix_from_euler_zyx(euler_i).T.dot(acc_i - g_n)
                for acc_i, euler_i in zip(acc, euler)
            ]
        )
        np.testing.assert_allclose(f_b, expected)


class Test_angular_velocity_body:
    @pytest.mark.parametrize(
        "euler, euler_dot, w_expect",
        [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),  # yaw-invariant
            ([np.pi / 2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
            ([np.pi / 2, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, np.pi / 2, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        ],
    )
    def test_single_axis(self, euler, euler_dot, w_expect):
        w_b = _angular_velocity_body(np.array([euler]), np.array([euler_dot]))

        np.testing.assert_allclose(w_b, [w_expect], atol=1e-12)

    def test_matches_rotation_matrix_derivative(self):
        # Poisson's kinematic equation: dR_nb/dt = R_nb @ S(w_b)
        rng = np.random.default_rng(0)
        euler = rng.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, size=(100, 3))
        euler_dot = rng.standard_normal((100, 3))
        h = 1e-6

        w_b = _angular_velocity_body(euler, euler_dot)

        R = _matrix_from_euler_zyx_batch(euler)
        R_dot = (
            _matrix_from_euler_zyx_batch(euler + h * euler_dot)
            - _matrix_from_euler_zyx_batch(euler - h * euler_dot)
        ) / (2.0 * h)
        S = np.einsum("nji,njk->nik", R, R_dot)
        expected = np.column_stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]])
        np.testing.assert_allclose(w_b, expected, atol=1e-8)


class Test_sample_motion:
    def test_dof_order_maps_to_columns(self):
        class SomeDOF(DOF):
            def __init__(self, value):
                self._value = value

            def _evaluate(self, t):
                ones = np.ones_like(t)
                return (
                    self._value * ones,
                    10.0 * self._value * ones,
                    100.0 * self._value * ones,
                )

        t = np.zeros(4)
        values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        dofs = Motion(**{name: SomeDOF(v) for name, v in zip(NAMES, values)})

        pos, vel, acc, euler, euler_dot = _sample_motion(dofs, t)

        np.testing.assert_allclose(pos, np.tile([1.0, 2.0, 3.0], (4, 1)))
        np.testing.assert_allclose(vel, np.tile([10.0, 20.0, 30.0], (4, 1)))
        np.testing.assert_allclose(acc, np.tile([100.0, 200.0, 300.0], (4, 1)))
        np.testing.assert_allclose(euler, np.tile([4.0, 5.0, 6.0], (4, 1)))
        np.testing.assert_allclose(euler_dot, np.tile([40.0, 50.0, 60.0], (4, 1)))


class Test_trajectory:
    def test_default(self):
        t, *out = ap.simulate.trajectory()
        t_expect, *out_expect = ap.simulate.trajectory(
            fs=10.0, n=10_000, g=9.80665, nav_frame="NED", motion="beat-6dof"
        )

        assert t.shape == (10_000,)
        assert t[0] == 0.0
        np.testing.assert_allclose(t[1:] - t[:-1], 0.1)
        np.testing.assert_array_equal(t, t_expect)
        for arr, arr_expect in zip(out, out_expect):
            assert arr.shape == (10_000, 3)
            np.testing.assert_array_equal(arr, arr_expect)

    @pytest.mark.parametrize("preset", PRESETS)
    def test_motion_preset(self, preset):
        fs = 10.0
        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(
            fs=fs, n=100, motion=preset
        )
        expect = _eval_motion(Motion.from_preset(preset), t)

        # Kinematics sampled from the preset
        np.testing.assert_allclose(p_n, expect[:3, 0].T)
        np.testing.assert_allclose(v_n, expect[:3, 1].T)
        np.testing.assert_allclose(euler_nb, expect[3:, 0].T)

        # Specific force rotated to the navigation frame, plus gravity, recovers
        # the acceleration
        R_nb = _matrix_from_euler_zyx_batch(euler_nb)
        a_n = np.einsum("nij,nj->ni", R_nb, f_b) + [0.0, 0.0, 9.80665]
        np.testing.assert_allclose(a_n, expect[:3, 2].T, atol=1e-12)

        # Angular rate validated by strapdown integration of attitude
        euler_est = _strapdown_euler(fs, euler_nb[0], f_b, w_b)
        np.testing.assert_allclose(euler_est, euler_nb, atol=3e-3)

    def test_fs_n(self):
        fs = 20.0
        n = 5000
        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(fs=fs, n=n)

        assert t.shape == (n,)
        assert p_n.shape == (n, 3)
        assert v_n.shape == (n, 3)
        assert euler_nb.shape == (n, 3)
        assert f_b.shape == (n, 3)
        assert w_b.shape == (n, 3)
        np.testing.assert_allclose(t[1:] - t[:-1], 1 / fs)

    def test_nav_frame_raise(self):
        with pytest.raises(ValueError, match="Unknown navigation frame"):
            ap.simulate.trajectory(nav_frame="invalid")

    def test_g(self):
        n = 100
        *_, f_b, _ = ap.simulate.trajectory(n=n, g=5.0, motion="stationary")

        np.testing.assert_allclose(f_b, np.tile([0.0, 0.0, -5.0], (n, 1)))

    def test_motion_custom(self):
        n = 100
        x = Beat(amp=2.0, omega=2 * np.pi * 0.2, omega_beat=2 * np.pi * 0.02)
        motion = Motion(x=x, yaw=Constant(0.5))

        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(n=n, motion=motion)

        px, vx, ax = x(t)

        # Position and velocity along x only
        np.testing.assert_allclose(p_n[:, 0], px)
        np.testing.assert_allclose(v_n[:, 0], vx)
        np.testing.assert_allclose(p_n[:, 1:], np.zeros((n, 2)))
        np.testing.assert_allclose(v_n[:, 1:], np.zeros((n, 2)))

        # Constant yaw, level attitude
        np.testing.assert_allclose(euler_nb[:, :2], np.zeros((n, 2)))
        np.testing.assert_allclose(euler_nb[:, 2], 0.5)
        np.testing.assert_allclose(w_b, np.zeros((n, 3)))

        # Acceleration along x is rotated by yaw in the horizontal plane
        np.testing.assert_allclose(f_b[:, 0], np.cos(0.5) * ax, atol=1e-12)
        np.testing.assert_allclose(f_b[:, 1], -np.sin(0.5) * ax, atol=1e-12)
        np.testing.assert_allclose(f_b[:, 2], -9.80665)

    @pytest.mark.parametrize("motion", PRESETS)
    def test_motion_preset_enu(self, motion):
        kwargs = {"n": 100, "motion": motion}
        t_ned, *out_ned = ap.simulate.trajectory(nav_frame="NED", **kwargs)
        t_enu, *out_enu = ap.simulate.trajectory(nav_frame="ENU", **kwargs)

        # Presets are rotated 180 degrees about x, so y and z change sign
        np.testing.assert_allclose(t_enu, t_ned)
        for arr_enu, arr_ned in zip(out_enu, out_ned):
            np.testing.assert_allclose(arr_enu[:, 0], arr_ned[:, 0])
            np.testing.assert_allclose(arr_enu[:, 1:], -arr_ned[:, 1:], atol=1e-12)

    def test_motion_custom_enu(self):
        # Custom motions are used as given, also in ENU
        n = 100
        x = Beat(amp=2.0, omega=2 * np.pi * 0.2, omega_beat=2 * np.pi * 0.02)
        motion = Motion(x=x, y=x, pitch=Constant(0.1), yaw=Constant(0.5))

        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(
            n=n, motion=motion, nav_frame="ENU"
        )

        px, vx, _ = x(t)
        np.testing.assert_allclose(p_n, np.column_stack([px, px, np.zeros(n)]))
        np.testing.assert_allclose(v_n, np.column_stack([vx, vx, np.zeros(n)]))
        np.testing.assert_allclose(euler_nb, np.tile([0.0, 0.1, 0.5], (n, 1)))
        np.testing.assert_allclose(w_b, np.zeros((n, 3)))

        # Same kinematics as in NED, only gravity changes direction
        *_, f_ned, _ = ap.simulate.trajectory(n=n, motion=motion, nav_frame="NED")
        R_nb = _matrix_from_euler_zyx(np.array([0.0, 0.1, 0.5]))
        g_b = R_nb.T @ np.array([0.0, 0.0, 9.80665])
        np.testing.assert_allclose(f_b, f_ned + 2.0 * g_b, atol=1e-12)

    def test_motion_raises(self):
        with pytest.raises(ValueError, match="Unknown motion preset"):
            ap.simulate.trajectory(motion="invalid")

    @pytest.mark.parametrize("motion", [None, 1.0, np.array([1, 2])])
    def test_motion_non_str_raises(self, motion):
        with pytest.raises(TypeError, match="'preset' must be a string"):
            ap.simulate.trajectory(motion=motion)

    @pytest.mark.parametrize("fs", [0.0, -10.0])
    def test_fs_raises(self, fs):
        with pytest.raises(ValueError, match="'fs' must be positive"):
            ap.simulate.trajectory(fs=fs)

    @pytest.mark.parametrize(
        "n, match",
        [
            (0, "'n' must be positive"),
            (-5, "'n' must be positive"),
            (10.7, "'n' must be a whole number"),
        ],
    )
    def test_n_raises(self, n, match):
        with pytest.raises(ValueError, match=match):
            ap.simulate.trajectory(n=n)

    def test_n_accepts_whole_valued_float(self):
        t, *_ = ap.simulate.trajectory(n=100.0)

        assert len(t) == 100
