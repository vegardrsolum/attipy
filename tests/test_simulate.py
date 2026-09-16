import numpy as np
import pytest

import attipy as ap
from attipy._transforms import _matrix_from_euler_zyx
from attipy.simulate._simulate import (
    DOF,
    BeatDOF,
    RampUp,
    _angular_velocity_body,
    _imu_from_motion,
    _motion_from_dofs,
    _specific_force_body,
)


@pytest.fixture
def t():
    return np.linspace(0, 10, 100)


class Test_DOF:
    @pytest.fixture
    def some_dof(self):
        class SomeDOF(DOF):

            def _evaluate(self, t):
                return np.ones_like(t), 2 * np.ones_like(t), 3 * np.ones_like(t)

        return SomeDOF()

    def test_y(self, some_dof, t):
        y = some_dof.y(t)
        np.testing.assert_allclose(y, np.ones(100))

    def test_dydt(self, some_dof, t):
        dydt = some_dof.dydt(t)
        np.testing.assert_allclose(dydt, 2 * np.ones(100))

    def test_d2ydt2(self, some_dof, t):
        d2ydt2 = some_dof.d2ydt2(t)
        np.testing.assert_allclose(d2ydt2, 3 * np.ones(100))

    def test__call__(self, some_dof, t):
        y, dydt, dy2dt2 = some_dof(t)
        np.testing.assert_allclose(y, np.ones(100))
        np.testing.assert_allclose(dydt, 2 * np.ones(100))
        np.testing.assert_allclose(dy2dt2, 3 * np.ones(100))

    def test_evaluate_is_the_only_required_method(self, some_dof, t):
        # _y, _dydt and _d2ydt2 are provided by the base class
        np.testing.assert_allclose(some_dof.y(t), some_dof(t)[0])
        np.testing.assert_allclose(some_dof.dydt(t), some_dof(t)[1])
        np.testing.assert_allclose(some_dof.d2ydt2(t), some_dof(t)[2])


class Test_BeatDOF:
    @pytest.fixture
    def beat(self):
        dof = BeatDOF(amp=2.0, freq_main=1.0, freq_beat=0.1, freq_hz=False, offset=1.0)
        return dof

    def test__init__(self):
        beat = BeatDOF(
            amp=3.0,
            freq_main=2.0,
            freq_beat=0.2,
            freq_hz=True,
            phase=4.0,
            phase_degrees=True,
            offset=5.0,
        )

        assert isinstance(beat, DOF)
        assert beat._amp == 3.0
        assert beat._w_main == pytest.approx(2.0 * np.pi * 2.0)
        assert beat._w_beat == pytest.approx(2.0 * np.pi * 0.2)
        assert beat._phase == pytest.approx((np.pi / 180.0) * 4.0)
        assert beat._offset == 5.0

    def test__init__default(self):
        beat_dof = BeatDOF()

        assert isinstance(beat_dof, DOF)
        assert beat_dof._amp == 1.0
        assert beat_dof._w_main == pytest.approx(0.1)
        assert beat_dof._w_beat == pytest.approx(0.01)
        assert beat_dof._phase == pytest.approx(0.0)
        assert beat_dof._offset == 0.0

    def test_y(self, beat, t):
        y = beat.y(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase
        offset = beat._offset

        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)

        y_expect = amp * beat * main + offset

        np.testing.assert_allclose(y, y_expect)

    def test_dydt(self, beat, t):
        dydt = beat.dydt(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase

        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = (w_beat / 2.0) * np.cos(w_beat / 2.0 * t)

        dydt_expect = amp * (dbeat * main + beat * dmain)

        np.testing.assert_allclose(dydt, dydt_expect)

    def test_d2ydt2(self, beat, t):
        d2ydt2 = beat.d2ydt2(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase

        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = (w_beat / 2.0) * np.cos(w_beat / 2.0 * t)
        d2main = -(w_main**2) * np.cos(w_main * t + phase)
        d2beat = -(w_beat**2 / 4.0) * np.sin(w_beat / 2.0 * t)

        d2ydt2_expect = amp * (d2beat * main + 2.0 * dbeat * dmain + beat * d2main)

        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test__call__(self, beat, t):
        y, dydt, d2ydt2 = beat(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase
        offset = beat._offset

        main = np.cos(w_main * t + phase)
        beat = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = (w_beat / 2.0) * np.cos(w_beat / 2.0 * t)
        d2main = -(w_main**2) * np.cos(w_main * t + phase)
        d2beat = -(w_beat**2 / 4.0) * np.sin(w_beat / 2.0 * t)

        y_expect = amp * beat * main + offset
        dydt_expect = amp * (dbeat * main + beat * dmain)
        d2ydt2_expect = amp * (d2beat * main + 2.0 * dbeat * dmain + beat * d2main)

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)


class Test_RampUp:
    @pytest.fixture
    def beat(self):
        return BeatDOF(amp=2.0, freq_main=1.0, freq_beat=0.1, offset=1.0)

    @pytest.fixture
    def rampup(self, beat):
        return RampUp(beat, 4.0, start=2.0)

    def test__init__(self, beat):
        rampup = RampUp(beat, 4.0, start=2.0)

        assert isinstance(rampup, DOF)
        assert rampup._dof is beat
        assert rampup._duration == 4.0
        assert rampup._start == 2.0

    def test__init__default(self, beat):
        rampup = RampUp(beat, 4.0)

        assert rampup._start == 0.0

    def test__init__raises(self, beat):
        with pytest.raises(ValueError):
            RampUp(beat, 0.0)

        with pytest.raises(ValueError):
            RampUp(beat, -1.0)

    def test_window(self, rampup):
        t = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # before, start, mid, end, after
        w, dw, d2w = rampup._window(t)

        np.testing.assert_allclose(w, [0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(dw, [0.0, 0.0, 30.0 * 0.5**4 / 4.0, 0.0, 0.0])
        np.testing.assert_allclose(d2w, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_y(self, rampup, beat, t):
        y = rampup.y(t)

        x = np.clip((t - 2.0) / 4.0, 0.0, 1.0)
        w = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3

        np.testing.assert_allclose(y, w * beat.y(t))

    def test_dydt(self, rampup, beat, t):
        dydt = rampup.dydt(t)

        x = np.clip((t - 2.0) / 4.0, 0.0, 1.0)
        w = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
        dw = (30.0 * x**4 - 60.0 * x**3 + 30.0 * x**2) / 4.0

        np.testing.assert_allclose(dydt, dw * beat.y(t) + w * beat.dydt(t))

    def test_d2ydt2(self, rampup, beat, t):
        d2ydt2 = rampup.d2ydt2(t)

        x = np.clip((t - 2.0) / 4.0, 0.0, 1.0)
        w = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
        dw = (30.0 * x**4 - 60.0 * x**3 + 30.0 * x**2) / 4.0
        d2w = (120.0 * x**3 - 180.0 * x**2 + 60.0 * x) / 4.0**2

        d2ydt2_expect = d2w * beat.y(t) + 2.0 * dw * beat.dydt(t) + w * beat.d2ydt2(t)

        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test_at_rest_before_start(self, rampup):
        t = np.linspace(0.0, 2.0, 100)
        y, dydt, d2ydt2 = rampup(t)

        np.testing.assert_allclose(y, 0.0)
        np.testing.assert_allclose(dydt, 0.0)
        np.testing.assert_allclose(d2ydt2, 0.0)

    def test_unaffected_after_rampup(self, rampup, beat):
        t = np.linspace(6.0, 20.0, 100)
        y, dydt, d2ydt2 = rampup(t)
        y_expect, dydt_expect, d2ydt2_expect = beat(t)

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test_derivatives_by_finite_difference(self, rampup):
        t = np.linspace(0.0, 20.0, 400_001)
        y, dydt, d2ydt2 = rampup(t)
        dt = t[1] - t[0]

        dydt_fd = (y[2:] - y[:-2]) / (2.0 * dt)
        d2ydt2_fd = (dydt[2:] - dydt[:-2]) / (2.0 * dt)

        # Central differences are inaccurate where the jerk is discontinuous,
        # i.e., at the two ends of the ramp-up period.
        t_mid = t[1:-1]
        valid = (np.abs(t_mid - 2.0) > dt) & (np.abs(t_mid - 6.0) > dt)

        np.testing.assert_allclose(dydt[1:-1][valid], dydt_fd[valid], atol=1e-6)
        np.testing.assert_allclose(d2ydt2[1:-1][valid], d2ydt2_fd[valid], atol=1e-6)


class Test_specific_force_body:
    def test_at_rest_and_level(self):
        g = 9.80665
        acc = np.zeros((5, 3))
        euler = np.zeros((5, 3))
        g_n = np.array([0.0, 0.0, g])

        f_b = _specific_force_body(acc, euler, g_n)

        np.testing.assert_allclose(f_b, np.tile([0.0, 0.0, -g], (5, 1)))

    def test_matches_per_sample_rotation(self):
        rng = np.random.default_rng(0)
        acc = rng.standard_normal((100, 3))
        euler = rng.uniform(-np.pi, np.pi, size=(100, 3))
        g_n = np.array([0.0, 0.0, 9.80665])

        f_b = _specific_force_body(acc, euler, g_n)

        expected = np.array(
            [
                _matrix_from_euler_zyx(euler_i).T.dot(acc_i - g_n)
                for acc_i, euler_i in zip(acc, euler)
            ]
        )
        np.testing.assert_allclose(f_b, expected)


class Test_motion_from_dofs:
    def test_dof_order_maps_to_columns(self):
        class ConstDOF(DOF):
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
        dofs = [ConstDOF(value) for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]

        pos, vel, acc, euler, euler_dot = _motion_from_dofs(dofs, t)

        np.testing.assert_allclose(pos, np.tile([1.0, 2.0, 3.0], (4, 1)))
        np.testing.assert_allclose(vel, np.tile([10.0, 20.0, 30.0], (4, 1)))
        np.testing.assert_allclose(acc, np.tile([100.0, 200.0, 300.0], (4, 1)))
        np.testing.assert_allclose(euler, np.tile([4.0, 5.0, 6.0], (4, 1)))
        np.testing.assert_allclose(euler_dot, np.tile([40.0, 50.0, 60.0], (4, 1)))

    @pytest.mark.parametrize("num_dofs", [0, 1, 3, 5, 7, 12])
    def test_wrong_number_of_dofs_raises(self, num_dofs):
        dofs = [BeatDOF() for _ in range(num_dofs)]

        with pytest.raises(ValueError):
            _motion_from_dofs(dofs, np.zeros(4))


class Test_imu_from_motion:
    @pytest.fixture
    def motion(self):
        rng = np.random.default_rng(0)
        acc = rng.standard_normal((20, 3))
        euler = 0.3 * rng.standard_normal((20, 3))
        euler_dot = 0.1 * rng.standard_normal((20, 3))
        return acc, euler, euler_dot

    def test_matches_underlying_conversions(self, motion):
        acc, euler, euler_dot = motion

        f_b, w_b = _imu_from_motion(acc, euler, euler_dot, g=9.81, nav_frame="ENU")

        g_n = np.array([0.0, 0.0, -9.81])
        np.testing.assert_allclose(f_b, _specific_force_body(acc, euler, g_n))
        np.testing.assert_allclose(w_b, _angular_velocity_body(euler, euler_dot))

    def test_nav_frame_raises(self, motion):
        with pytest.raises(ValueError):
            _imu_from_motion(*motion, g=9.80665, nav_frame="invalid")


class Test_trajectory:
    def test_default(self):
        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory()

        # Expected DOF signals
        pos_amp = 1.0
        att_amp = 0.1
        phases = np.linspace(0, 2.0 * np.pi, 6, endpoint=False)
        px, vx, _ = BeatDOF(pos_amp, 0.1, 0.01, freq_hz=True, phase=phases[0])(t)
        py, vy, _ = BeatDOF(pos_amp, 0.1, 0.01, freq_hz=True, phase=phases[1])(t)
        pz, vz, _ = BeatDOF(pos_amp, 0.1, 0.01, freq_hz=True, phase=phases[2])(t)
        r, *_ = BeatDOF(att_amp, 0.1, 0.01, freq_hz=True, phase=phases[3])(t)
        p, *_ = BeatDOF(att_amp, 0.1, 0.01, freq_hz=True, phase=phases[4])(t)
        y, *_ = BeatDOF(att_amp, 0.1, 0.01, freq_hz=True, phase=phases[5])(t)

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

    def test_rampup(self):
        fs, n = 10.0, 5000
        rampup, rampup_start = 120.0, 60.0
        g = 9.80665

        t, p_n, v_n, euler_nb, f_b, w_b = ap.simulate.trajectory(
            fs=fs, n=n, g=g, rampup=rampup, rampup_start=rampup_start
        )

        # Stationary (i.e., at rest at the origin) before the ramp-up starts
        before = t < rampup_start
        assert before.sum() > 0
        np.testing.assert_allclose(p_n[before], 0.0)
        np.testing.assert_allclose(v_n[before], 0.0)
        np.testing.assert_allclose(euler_nb[before], 0.0)
        np.testing.assert_allclose(w_b[before], 0.0)
        np.testing.assert_allclose(
            f_b[before], np.tile([0.0, 0.0, -g], (before.sum(), 1))
        )

        # Unaffected by the ramp-up once it is completed
        after = t >= rampup_start + rampup
        after_expect = ap.simulate.trajectory(fs=fs, n=n, g=g)
        assert after.sum() > 0
        np.testing.assert_allclose(p_n[after], after_expect[1][after])
        np.testing.assert_allclose(v_n[after], after_expect[2][after])
        np.testing.assert_allclose(euler_nb[after], after_expect[3][after])
        np.testing.assert_allclose(f_b[after], after_expect[4][after])
        np.testing.assert_allclose(w_b[after], after_expect[5][after])

    def test_rampup_raises(self):
        with pytest.raises(ValueError):
            ap.simulate.trajectory(rampup=0.0)

        with pytest.raises(ValueError):
            ap.simulate.trajectory(rampup=-1.0)

        with pytest.raises(ValueError):
            ap.simulate.trajectory(rampup=120.0, rampup_start=-1.0)
