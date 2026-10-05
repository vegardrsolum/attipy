import numpy as np
import pytest

import attipy as ap
from attipy._transforms import _matrix_from_euler_zyx
from attipy.simulate._dof import DOF, Beat, Constant
from attipy.simulate._simulate import (
    _MOTION_PRESETS,
    _PACKAGE_PATH,
    Motion,
    _angular_velocity_body,
    _imu_from_kinematics,
    _load_csd,
    _sample_motion,
    _specific_force_body,
    _z_down_to_z_up,
)

VESSEL_CSD_PATH = str(_PACKAGE_PATH.joinpath("_data", "supply_vessel_csd.npz"))


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


class Test_motion_presets:
    NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

    def test_keys(self):
        assert set(_MOTION_PRESETS) == {
            "stationary",
            "beat-6dof",
            "beat-3dof",
            "vessel-6dof",
            "vessel-3dof",
        }

    @pytest.mark.parametrize("name", list(_MOTION_PRESETS))
    def test_returns_new_motion(self, name):
        make_motion = _MOTION_PRESETS[name]
        motion = make_motion()

        assert isinstance(motion, Motion)
        assert make_motion() is not motion

    @pytest.mark.parametrize("name", ["beat-3dof", "vessel-3dof"])
    def test_3dof_matches_6dof_rotations(self, name):
        t = np.linspace(0.0, 100.0, 1001)
        motion_3dof = _MOTION_PRESETS[name]()
        motion_6dof = _MOTION_PRESETS[name.replace("3dof", "6dof")]()

        for dof_name in ("x", "y", "z"):
            np.testing.assert_allclose(getattr(motion_3dof, dof_name).y(t), 0.0)

        for dof_name in ("roll", "pitch", "yaw"):
            for out, out_expect in zip(
                getattr(motion_3dof, dof_name)(t), getattr(motion_6dof, dof_name)(t)
            ):
                np.testing.assert_allclose(out, out_expect)

    def test_vessel_6dof_deterministic(self):
        t = np.linspace(0.0, 100.0, 1001)
        motion_a = _MOTION_PRESETS["vessel-6dof"]()
        motion_b = _MOTION_PRESETS["vessel-6dof"]()

        for name in self.NAMES:
            for out_a, out_b in zip(
                getattr(motion_a, name)(t), getattr(motion_b, name)(t)
            ):
                np.testing.assert_array_equal(out_a, out_b)

    def test_vessel_6dof_matches_csd(self):
        f, csd = _load_csd(VESSEL_CSD_PATH)
        motion = _MOTION_PRESETS["vessel-6dof"]()
        dofs = [getattr(motion, name) for name in self.NAMES]

        omega = sorted({s._w for d in dofs for s in d._dofs})
        z = np.zeros((len(dofs), len(omega)), dtype=np.complex128)
        for i, d in enumerate(dofs):
            for s in d._dofs:
                z[i, omega.index(s._w)] = s._amp * np.exp(1j * s._phase)

        np.testing.assert_allclose(
            z.conj() @ z.T / 2.0, np.trapezoid(csd, f, axis=0), atol=1e-12
        )


class Test_z_down_to_z_up:
    def test_signs(self):
        t = np.linspace(0.0, 10.0, 101)
        names = ("x", "y", "z", "roll", "pitch", "yaw")
        signs = (1.0, -1.0, -1.0, 1.0, -1.0, -1.0)
        motion = Motion(
            **{
                name: Beat(amp=i + 1.0, omega=0.5 + 0.1 * i, omega_beat=0.05, phase=i)
                for i, name in enumerate(names)
            }
        )

        motion_z_up = _z_down_to_z_up(motion)

        assert isinstance(motion_z_up, Motion)
        for name, sign in zip(names, signs):
            for out, out_expect in zip(
                getattr(motion_z_up, name)(t), getattr(motion, name)(t)
            ):
                np.testing.assert_allclose(out, sign * out_expect)


class Test_Motion:
    NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

    def test__init__(self):
        dofs = {name: Constant(float(i)) for i, name in enumerate(self.NAMES)}
        motion = Motion(**dofs)

        for name in self.NAMES:
            assert getattr(motion, name) is dofs[name]

    def test__init__default(self):
        t = np.linspace(0.0, 10.0, 100)
        motion = Motion()

        for name in self.NAMES:
            dof = getattr(motion, name)
            assert isinstance(dof, Constant)
            np.testing.assert_allclose(dof.y(t), np.zeros_like(t))

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
        names = ("x", "y", "z", "roll", "pitch", "yaw")
        values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        dofs = Motion(**{name: SomeDOF(v) for name, v in zip(names, values)})

        pos, vel, acc, euler, euler_dot = _sample_motion(dofs, t)

        np.testing.assert_allclose(pos, np.tile([1.0, 2.0, 3.0], (4, 1)))
        np.testing.assert_allclose(vel, np.tile([10.0, 20.0, 30.0], (4, 1)))
        np.testing.assert_allclose(acc, np.tile([100.0, 200.0, 300.0], (4, 1)))
        np.testing.assert_allclose(euler, np.tile([4.0, 5.0, 6.0], (4, 1)))
        np.testing.assert_allclose(euler_dot, np.tile([40.0, 50.0, 60.0], (4, 1)))


class Test_imu_from_kinematics:
    @pytest.fixture
    def kinematics(self):
        rng = np.random.default_rng(0)
        acc = rng.standard_normal((20, 3))
        euler = 0.3 * rng.standard_normal((20, 3))
        euler_dot = 0.1 * rng.standard_normal((20, 3))
        return acc, euler, euler_dot

    def test_matches_underlying_conversions(self, kinematics):
        acc, euler, euler_dot = kinematics

        f_b, w_b = _imu_from_kinematics(acc, euler, euler_dot, g=9.81, nav_frame="ENU")

        g_n = np.array([0.0, 0.0, -9.81])
        np.testing.assert_allclose(f_b, _specific_force_body(acc, euler, g_n))
        np.testing.assert_allclose(w_b, _angular_velocity_body(euler, euler_dot))

    def test_nav_frame_raises(self, kinematics):
        with pytest.raises(ValueError):
            _imu_from_kinematics(*kinematics, g=9.80665, nav_frame="invalid")


class Test_trajectory:
    def test_default(self):
        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory()

        # Expected DOF signals
        phases = np.linspace(0, 2.0 * np.pi, 6, endpoint=False)
        beat_kwargs = {"omega": 2 * np.pi * 0.1, "omega_beat": 2 * np.pi * 0.01}
        px, vx, _ = Beat(amp=1.0, phase=phases[0], **beat_kwargs)(t)
        py, vy, _ = Beat(amp=1.0, phase=phases[1], **beat_kwargs)(t)
        pz, vz, _ = Beat(amp=1.0, phase=phases[2], **beat_kwargs)(t)
        r, *_ = Beat(amp=0.1, phase=phases[3], **beat_kwargs)(t)
        p, *_ = Beat(amp=0.1, phase=phases[4], **beat_kwargs)(t)
        y, *_ = Beat(amp=0.1, phase=phases[5], **beat_kwargs)(t)

        # Time
        fs_expect = 10.0
        assert t.shape == (10_000,)
        assert t[0] == 0.0
        np.testing.assert_allclose(t[1:] - t[:-1], 1 / fs_expect)

        # Position
        assert p_n.shape == (10_000, 3)
        np.testing.assert_allclose(p_n[:, 0], px)
        np.testing.assert_allclose(p_n[:, 1], py)
        np.testing.assert_allclose(p_n[:, 2], pz)

        # Velocity
        assert v_n.shape == (10_000, 3)
        np.testing.assert_allclose(v_n[:, 0], vx)
        np.testing.assert_allclose(v_n[:, 1], vy)
        np.testing.assert_allclose(v_n[:, 2], vz)

        # Euler angles
        assert euler_nb.shape == (10_000, 3)
        np.testing.assert_allclose(euler_nb[:, 0], r)
        np.testing.assert_allclose(euler_nb[:, 1], p)
        np.testing.assert_allclose(euler_nb[:, 2], y)

        # Specific force
        assert f_b.shape == (10_000, 3)

        # Angular rate
        assert w_b.shape == (10_000, 3)

        # Validate f and w by strapdown integration using MEKF (no aiding)
        q0 = ap.Attitude.from_euler(euler_nb[0], degrees=False).as_quaternion()
        mekf = ap.MEKF(fs_expect, q0)
        euler_est = [euler_nb[0]]
        for f_i, w_i in zip(f_b[1:], w_b[1:]):
            mekf.update(f_i, w_i, gref=False)
            euler_est.append(mekf.attitude.as_euler(degrees=False))
        euler_est = np.array(euler_est)

        # TODO: check also pos and vel when strapdown estimator is available
        np.testing.assert_allclose(euler_est[:100], euler_nb[:100], atol=2e-3)

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

    def test_degrees(self):
        *_, euler_deg, _, _ = ap.simulate.trajectory(degrees=True)
        *_, euler_rad, _, _ = ap.simulate.trajectory(degrees=False)

        np.testing.assert_allclose(euler_deg, np.degrees(euler_rad))

    def test_nav_frame(self):

        # NED
        *_, f_ned, _ = ap.simulate.trajectory(nav_frame="NED")
        assert -10.0 < f_ned.mean(axis=0)[2] < -9.5

        # ENU
        *_, f_enu, _ = ap.simulate.trajectory(nav_frame="ENU")
        assert 9.5 < f_enu.mean(axis=0)[2] < 10.0

        with pytest.raises(ValueError):
            ap.simulate.trajectory(nav_frame="invalid")

    def test_g(self):
        g = 5.0
        *_, f, _ = ap.simulate.trajectory(g=g)
        assert -6.0 < f.mean(axis=0)[2] < -4

    def test_motion_default_is_beating(self):
        beating = ap.simulate.trajectory(n=100, motion="beat-6dof")
        default = ap.simulate.trajectory(n=100)

        for out, out_expect in zip(default, beating):
            np.testing.assert_allclose(out, out_expect)

    def test_motion_stationary(self):
        n = 100
        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(
            n=n, motion="stationary"
        )

        assert t.shape == (n,)
        np.testing.assert_allclose(p_n, np.zeros((n, 3)))
        np.testing.assert_allclose(v_n, np.zeros((n, 3)))
        np.testing.assert_allclose(euler_nb, np.zeros((n, 3)))
        np.testing.assert_allclose(w_b, np.zeros((n, 3)))

        # Level and at rest -> specific force balances gravity
        np.testing.assert_allclose(f_b, np.tile([0.0, 0.0, -9.80665], (n, 1)))

    def test_motion_stationary_nav_frame(self):
        n = 100
        *_, f_b, _ = ap.simulate.trajectory(n=n, motion="stationary", nav_frame="ENU")

        np.testing.assert_allclose(f_b, np.tile([0.0, 0.0, 9.80665], (n, 1)))

    def test_motion_beat_3dof(self):
        n = 100
        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(
            n=n, motion="beat-3dof"
        )

        # Expected attitude DOF signals
        beat_kwargs = {"omega": 2 * np.pi * 0.1, "omega_beat": 2 * np.pi * 0.01}
        r, *_ = Beat(amp=0.1, phase=np.pi, **beat_kwargs)(t)
        p, *_ = Beat(amp=0.1, phase=4 * np.pi / 3, **beat_kwargs)(t)
        y, *_ = Beat(amp=0.1, phase=5 * np.pi / 3, **beat_kwargs)(t)

        # No translation
        np.testing.assert_allclose(p_n, np.zeros((n, 3)))
        np.testing.assert_allclose(v_n, np.zeros((n, 3)))

        # Beating attitude
        np.testing.assert_allclose(euler_nb[:, 0], r)
        np.testing.assert_allclose(euler_nb[:, 1], p)
        np.testing.assert_allclose(euler_nb[:, 2], y)
        assert np.any(np.abs(w_b) > 0.0)

        # No translation -> specific force is gravity only
        np.testing.assert_allclose(np.linalg.norm(f_b, axis=1), 9.80665)

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

    def test_motion_vessel_6dof(self):
        fs = 10.0
        n = 1000
        _, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(
            fs=fs, n=n, motion="vessel-6dof"
        )

        for arr in (p_n, v_n, euler_nb, f_b, w_b):
            assert arr.shape == (n, 3)
            assert np.all(np.std(arr, axis=0) > 0.0)

        # Validate f and w by strapdown integration using MEKF (no aiding)
        q0 = ap.Attitude.from_euler(euler_nb[0], degrees=False).as_quaternion()
        mekf = ap.MEKF(fs, q0)
        euler_est = [euler_nb[0]]
        for f_i, w_i in zip(f_b[1:], w_b[1:]):
            mekf.update(f_i, w_i, gref=False)
            euler_est.append(mekf.attitude.as_euler(degrees=False))
        euler_est = np.array(euler_est)

        np.testing.assert_allclose(euler_est[:100], euler_nb[:100], atol=2e-3)

    def test_motion_vessel_3dof(self):
        n = 1000
        _, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(
            n=n, motion="vessel-3dof"
        )
        *_, euler_6dof, _, w_6dof = ap.simulate.trajectory(n=n, motion="vessel-6dof")

        # No translation
        np.testing.assert_allclose(p_n, np.zeros((n, 3)))
        np.testing.assert_allclose(v_n, np.zeros((n, 3)))

        # Same rotations as 6DOF
        np.testing.assert_allclose(euler_nb, euler_6dof)
        np.testing.assert_allclose(w_b, w_6dof)

        # No translation -> specific force is gravity only
        np.testing.assert_allclose(np.linalg.norm(f_b, axis=1), 9.80665)

    @pytest.mark.parametrize("motion", list(_MOTION_PRESETS))
    @pytest.mark.parametrize("degrees", [False, True])
    def test_motion_preset_enu(self, motion, degrees):
        kwargs = {"n": 100, "motion": motion, "degrees": degrees}
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

    @pytest.mark.parametrize(
        "motion", ["BEAT-6DOF", "Beat-3dof", "Stationary", "Vessel-6DOF", "VESSEL-3dof"]
    )
    def test_motion_case_insensitive(self, motion):
        out = ap.simulate.trajectory(n=10, motion=motion)
        out_expect = ap.simulate.trajectory(n=10, motion=motion.lower())

        for arr, arr_expect in zip(out, out_expect):
            np.testing.assert_allclose(arr, arr_expect)

    @pytest.mark.parametrize("motion", ["invalid", None, 1.0, np.array([1, 2])])
    def test_motion_raises(self, motion):
        with pytest.raises(ValueError, match="Unknown motion type"):
            ap.simulate.trajectory(motion=motion)

    def test_fs_raises(self):
        with pytest.raises(ValueError):
            ap.simulate.trajectory(fs=0.0)

        with pytest.raises(ValueError):
            ap.simulate.trajectory(fs=-10.0)

    def test_n_raises(self):
        with pytest.raises(ValueError):
            ap.simulate.trajectory(n=0)

        with pytest.raises(ValueError):
            ap.simulate.trajectory(n=-5)

        with pytest.raises(ValueError):
            ap.simulate.trajectory(n=10.7)

    def test_n_accepts_whole_valued_float(self):
        t, *_ = ap.simulate.trajectory(n=100.0)

        assert len(t) == 100
