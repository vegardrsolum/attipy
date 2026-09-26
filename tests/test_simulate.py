import numpy as np
import pytest

import attipy as ap
from attipy._transforms import _matrix_from_euler_zyx
from attipy.simulate._dof import DOF, BeatDOF, ConstantDOF
from attipy.simulate._simulate import (
    Motion,
    _angular_velocity_body,
    _imu_from_kinematics,
    _sample_motion,
    _specific_force_body,
)


class Test_Motion:
    NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

    def test__init__(self):
        dofs = {name: ConstantDOF(float(i)) for i, name in enumerate(self.NAMES)}
        motion = Motion(**dofs, degrees=True)

        for name in self.NAMES:
            assert getattr(motion, name) is dofs[name]
        assert motion.degrees is True

    def test__init__default(self):
        t = np.linspace(0.0, 10.0, 100)
        motion = Motion()

        for name in self.NAMES:
            dof = getattr(motion, name)
            assert isinstance(dof, ConstantDOF)
            np.testing.assert_allclose(dof.y(t), np.zeros_like(t))
        assert motion.degrees is False

    def test__init__defaults_not_shared(self):
        assert Motion().x is not Motion().x

    def test__init__keyword_only(self):
        with pytest.raises(TypeError):
            Motion(ConstantDOF())

    @pytest.mark.parametrize("name", NAMES)
    def test__init__non_dof_raises(self, name):
        with pytest.raises(TypeError, match=f"'{name}'"):
            Motion(**{name: 1.0})

    @pytest.mark.parametrize("name", NAMES + ("degrees",))
    def test_frozen(self, name):
        motion = Motion()

        with pytest.raises(AttributeError):
            setattr(motion, name, ConstantDOF())

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

    def test_degrees_converts_angular_dofs(self):
        t = np.linspace(0.0, 10.0, 100)
        dofs = {
            "x": BeatDOF(omega=0.1, omega_beat=0.01),
            "roll": ConstantDOF(30.0),
            "pitch": 5.0 * BeatDOF(omega=0.1, omega_beat=0.01),
            "yaw": 10.0 * BeatDOF(omega=0.2, omega_beat=0.02),
        }

        out_deg = _sample_motion(Motion(**dofs, degrees=True), t)
        out_rad = _sample_motion(Motion(**dofs), t)

        pos_deg, vel_deg, acc_deg, euler_deg, euler_dot_deg = out_deg
        pos_rad, vel_rad, acc_rad, euler_rad, euler_dot_rad = out_rad

        np.testing.assert_allclose(pos_deg, pos_rad)
        np.testing.assert_allclose(vel_deg, vel_rad)
        np.testing.assert_allclose(acc_deg, acc_rad)
        np.testing.assert_allclose(euler_deg, np.radians(euler_rad))
        np.testing.assert_allclose(euler_dot_deg, np.radians(euler_dot_rad))


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
        pos_amp = 1.0
        att_amp = 0.1
        phases = np.linspace(0, 2.0 * np.pi, 6, endpoint=False)
        beats = [
            BeatDOF(omega=2 * np.pi * 0.1, omega_beat=2 * np.pi * 0.01, phase=phase)
            for phase in phases
        ]
        px, vx, _ = (pos_amp * beats[0])(t)
        py, vy, _ = (pos_amp * beats[1])(t)
        pz, vz, _ = (pos_amp * beats[2])(t)
        r, *_ = (att_amp * beats[3])(t)
        p, *_ = (att_amp * beats[4])(t)
        y, *_ = (att_amp * beats[5])(t)

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
        beats = [
            BeatDOF(omega=2 * np.pi * 0.1, omega_beat=2 * np.pi * 0.01, phase=phase)
            for phase in (np.pi, 4 * np.pi / 3, 5 * np.pi / 3)
        ]
        r, *_ = (0.1 * beats[0])(t)
        p, *_ = (0.1 * beats[1])(t)
        y, *_ = (0.1 * beats[2])(t)

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
        x = 2.0 * BeatDOF(omega=2 * np.pi * 0.2, omega_beat=2 * np.pi * 0.02)
        motion = Motion(x=x, yaw=ConstantDOF(0.5))

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

    def test_motion_custom_degrees(self):
        n = 100
        amp_deg = 5.0

        roll = BeatDOF(omega=2 * np.pi * 0.1, omega_beat=2 * np.pi * 0.01)
        pitch = BeatDOF(omega=2 * np.pi * 0.2, omega_beat=2 * np.pi * 0.01)

        motion_deg = Motion(
            roll=amp_deg * roll,
            pitch=amp_deg * pitch,
            degrees=True,
        )
        motion_rad = Motion(
            roll=np.radians(amp_deg) * roll,
            pitch=np.radians(amp_deg) * pitch,
            degrees=False,
        )

        out_deg = ap.simulate.trajectory(n=n, motion=motion_deg, degrees=True)
        out_rad = ap.simulate.trajectory(n=n, motion=motion_rad, degrees=False)

        t, p_deg, v_deg, euler_deg, f_deg, w_deg = out_deg
        _, p_rad, v_rad, euler_rad, f_rad, w_rad = out_rad

        # Degrees in -> degrees out reproduces the input signals
        roll, *_ = motion_deg.roll(t)
        pitch, *_ = motion_deg.pitch(t)
        np.testing.assert_allclose(euler_deg[:, 0], roll)
        np.testing.assert_allclose(euler_deg[:, 1], pitch)

        # Equivalent to the same motion specified in radians
        np.testing.assert_allclose(p_deg, p_rad)
        np.testing.assert_allclose(v_deg, v_rad)
        np.testing.assert_allclose(f_deg, f_rad)
        np.testing.assert_allclose(np.radians(euler_deg), euler_rad)
        np.testing.assert_allclose(np.radians(w_deg), w_rad)

    @pytest.mark.parametrize("motion", ["BEAT-6DOF", "Beat-3dof", "Stationary"])
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
