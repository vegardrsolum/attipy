import numpy as np
import pytest

from attipy.simulate._dof import DOF, BeatDOF, ConstantDOF, RampUp


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
        dof = BeatDOF(amp=2.0, freq_main=1.0, freq_beat=0.1, freq_hz=False)
        return dof

    def test__init__(self):
        beat = BeatDOF(
            amp=3.0,
            freq_main=2.0,
            freq_beat=0.2,
            freq_hz=True,
            phase=4.0,
            phase_degrees=True,
        )

        assert isinstance(beat, DOF)
        assert beat._amp == 3.0
        assert beat._w_main == pytest.approx(2.0 * np.pi * 2.0)
        assert beat._w_beat == pytest.approx(2.0 * np.pi * 0.2)
        assert beat._phase == pytest.approx((np.pi / 180.0) * 4.0)

    def test__init__default(self):
        beat_dof = BeatDOF()

        assert isinstance(beat_dof, DOF)
        assert beat_dof._amp == 1.0
        assert beat_dof._w_main == pytest.approx(0.1)
        assert beat_dof._w_beat == pytest.approx(0.01)
        assert beat_dof._phase == pytest.approx(0.0)

    def test_y(self, beat, t):
        y = beat.y(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase

        main = np.cos(w_main * t + phase)
        beat_ = np.sin(w_beat / 2.0 * t)

        y_expect = amp * beat_ * main

        np.testing.assert_allclose(y, y_expect)

    def test_dydt(self, beat, t):
        dydt = beat.dydt(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase

        main = np.cos(w_main * t + phase)
        beat_ = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = (w_beat / 2.0) * np.cos(w_beat / 2.0 * t)

        dydt_expect = amp * (dbeat * main + beat_ * dmain)

        np.testing.assert_allclose(dydt, dydt_expect)

    def test_d2ydt2(self, beat, t):
        d2ydt2 = beat.d2ydt2(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase

        main = np.cos(w_main * t + phase)
        beat_ = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = (w_beat / 2.0) * np.cos(w_beat / 2.0 * t)
        d2main = -(w_main**2) * np.cos(w_main * t + phase)
        d2beat = -(w_beat**2 / 4.0) * np.sin(w_beat / 2.0 * t)

        d2ydt2_expect = amp * (d2beat * main + 2.0 * dbeat * dmain + beat_ * d2main)

        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test__call__(self, beat, t):
        y, dydt, d2ydt2 = beat(t)

        amp = beat._amp
        w_main = beat._w_main
        w_beat = beat._w_beat
        phase = beat._phase

        main = np.cos(w_main * t + phase)
        beat_ = np.sin(w_beat / 2.0 * t)
        dmain = -w_main * np.sin(w_main * t + phase)
        dbeat = (w_beat / 2.0) * np.cos(w_beat / 2.0 * t)
        d2main = -(w_main**2) * np.cos(w_main * t + phase)
        d2beat = -(w_beat**2 / 4.0) * np.sin(w_beat / 2.0 * t)

        y_expect = amp * beat_ * main
        dydt_expect = amp * (dbeat * main + beat_ * dmain)
        d2ydt2_expect = amp * (d2beat * main + 2.0 * dbeat * dmain + beat_ * d2main)

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)


class Test_ConstantDOF:
    @pytest.fixture
    def constant(self):
        dof = ConstantDOF(value=2.0)
        return dof

    def test__init__(self):
        constant = ConstantDOF(value=3.0)

        assert isinstance(constant, DOF)
        assert constant._value == 3.0

    def test__init__default(self):
        constant = ConstantDOF()

        assert isinstance(constant, DOF)
        assert constant._value == 0.0

    def test_y(self, constant, t):
        y = constant.y(t)

        y_expect = 2.0 * np.ones_like(t)

        np.testing.assert_allclose(y, y_expect)

    def test_dydt(self, constant, t):
        dydt = constant.dydt(t)

        dydt_expect = np.zeros_like(t)

        np.testing.assert_allclose(dydt, dydt_expect)

    def test_d2ydt2(self, constant, t):
        d2ydt2 = constant.d2ydt2(t)

        d2ydt2_expect = np.zeros_like(t)

        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test__call__(self, constant, t):
        y, dydt, d2ydt2 = constant(t)

        y_expect = 2.0 * np.ones_like(t)
        dydt_expect = np.zeros_like(t)
        d2ydt2_expect = np.zeros_like(t)

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)


class Test_RampUp:
    @pytest.fixture
    def beat(self):
        return BeatDOF(amp=2.0, freq_main=1.0, freq_beat=0.1)

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
