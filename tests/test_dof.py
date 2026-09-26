import numpy as np
import pytest

from attipy.simulate._dof import (
    DOF,
    BeatDOF,
    ConstantDOF,
    RampUp,
    SineDOF,
    _as_dof,
    _Product,
    _Sum,
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


class Test_SineDOF:
    @pytest.fixture
    def sine(self):
        dof = SineDOF(amp=2.0, freq=3.0, freq_hz=False, phase=0.5)
        return dof

    def test__init__(self):
        sine = SineDOF(
            amp=3.0,
            freq=2.0,
            freq_hz=True,
            phase=4.0,
            phase_degrees=True,
        )

        assert isinstance(sine, DOF)
        assert sine._amp == 3.0
        assert sine._w == pytest.approx(2.0 * np.pi * 2.0)
        assert sine._phase == pytest.approx((np.pi / 180.0) * 4.0)

    def test__init__default(self):
        sine = SineDOF()

        assert isinstance(sine, DOF)
        assert sine._amp == 1.0
        assert sine._w == pytest.approx(0.1)
        assert sine._phase == pytest.approx(0.0)

    def test_y(self, sine, t):
        y = sine.y(t)

        y_expect = 2.0 * np.sin(3.0 * t + 0.5)

        np.testing.assert_allclose(y, y_expect)

    def test_dydt(self, sine, t):
        dydt = sine.dydt(t)

        dydt_expect = 2.0 * 3.0 * np.cos(3.0 * t + 0.5)

        np.testing.assert_allclose(dydt, dydt_expect)

    def test_d2ydt2(self, sine, t):
        d2ydt2 = sine.d2ydt2(t)

        d2ydt2_expect = -2.0 * 3.0**2 * np.sin(3.0 * t + 0.5)

        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test__call__(self, sine, t):
        y, dydt, d2ydt2 = sine(t)

        y_expect = 2.0 * np.sin(3.0 * t + 0.5)
        dydt_expect = 2.0 * 3.0 * np.cos(3.0 * t + 0.5)
        d2ydt2_expect = -2.0 * 3.0**2 * np.sin(3.0 * t + 0.5)

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
        rampup = RampUp(beat)

        assert rampup._duration == 100.0
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


class Test__as_dof:
    def test_dof(self):
        dof = BeatDOF()
        assert _as_dof(dof) is dof

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0), np.int64(2)])
    def test_real(self, value, t):
        dof = _as_dof(value)

        assert isinstance(dof, ConstantDOF)
        assert dof._value == 2.0
        assert type(dof._value) is float
        np.testing.assert_allclose(dof.y(t), 2.0)

    @pytest.mark.parametrize("other", ["a", None, [1.0], np.array([1.0]), 1j])
    def test_unsupported(self, other):
        assert _as_dof(other) is None


class Test__Sum:
    @pytest.fixture
    def beat(self):
        return BeatDOF(amp=2.0, freq_main=1.0, freq_beat=0.1)

    @pytest.fixture
    def constant(self):
        return ConstantDOF(value=3.0)

    @pytest.fixture
    def sine(self):
        return SineDOF(amp=0.5, freq=1.5, phase=0.3)

    @pytest.fixture
    def dof_sum(self, beat, constant, sine):
        return _Sum(beat, constant, sine)

    def test__init__(self, beat, constant, sine):
        dof_sum = _Sum(beat, constant, sine)

        assert isinstance(dof_sum, DOF)
        assert dof_sum._dofs == (beat, constant, sine)

    def test__init__raises_empty(self):
        with pytest.raises(ValueError):
            _Sum()

    def test__init__raises_type(self, beat):
        with pytest.raises(TypeError):
            _Sum(beat, 1.0)

    def test_single(self, beat, t):
        dof_sum = _Sum(beat)
        y, dydt, d2ydt2 = dof_sum(t)
        y_expect, dydt_expect, d2ydt2_expect = beat(t)

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test_y(self, dof_sum, beat, constant, sine, t):
        y = dof_sum.y(t)

        y_expect = beat.y(t) + constant.y(t) + sine.y(t)

        np.testing.assert_allclose(y, y_expect)

    def test_dydt(self, dof_sum, beat, constant, sine, t):
        dydt = dof_sum.dydt(t)

        dydt_expect = beat.dydt(t) + constant.dydt(t) + sine.dydt(t)

        np.testing.assert_allclose(dydt, dydt_expect)

    def test_d2ydt2(self, dof_sum, beat, constant, sine, t):
        d2ydt2 = dof_sum.d2ydt2(t)

        d2ydt2_expect = beat.d2ydt2(t) + constant.d2ydt2(t) + sine.d2ydt2(t)

        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test__call__(self, dof_sum, beat, constant, sine, t):
        y, dydt, d2ydt2 = dof_sum(t)

        y_expect = beat.y(t) + constant.y(t) + sine.y(t)
        dydt_expect = beat.dydt(t) + constant.dydt(t) + sine.dydt(t)
        d2ydt2_expect = beat.d2ydt2(t) + constant.d2ydt2(t) + sine.d2ydt2(t)

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test_same_dof_twice(self, beat, t):
        dof_sum = _Sum(beat, beat)
        y, dydt, d2ydt2 = dof_sum(t)

        np.testing.assert_allclose(y, 2.0 * beat.y(t))
        np.testing.assert_allclose(dydt, 2.0 * beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, 2.0 * beat.d2ydt2(t))

    def test_does_not_modify_underlying(self, t):
        # Underlying DOF returns the same arrays on every call, so in-place
        # accumulation into them would be detected.
        y0, dydt0, d2ydt20 = np.ones_like(t), 2 * np.ones_like(t), 3 * np.ones_like(t)

        class SomeDOF(DOF):
            def _evaluate(self, t):
                return y0, dydt0, d2ydt20

        dof_sum = _Sum(SomeDOF(), SomeDOF())
        dof_sum(t)
        dof_sum(t)

        np.testing.assert_allclose(y0, 1.0)
        np.testing.assert_allclose(dydt0, 2.0)
        np.testing.assert_allclose(d2ydt20, 3.0)

    def test_nested(self, beat, constant, sine, t):
        dof_sum = _Sum(_Sum(beat, constant), sine)
        y, dydt, d2ydt2 = dof_sum(t)

        np.testing.assert_allclose(y, beat.y(t) + constant.y(t) + sine.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t) + constant.dydt(t) + sine.dydt(t))
        np.testing.assert_allclose(
            d2ydt2, beat.d2ydt2(t) + constant.d2ydt2(t) + sine.d2ydt2(t)
        )

    def test__add__operator(self, beat, constant, sine, t):
        dof_sum = beat + constant + sine
        y, dydt, d2ydt2 = dof_sum(t)

        assert isinstance(dof_sum, _Sum)
        np.testing.assert_allclose(y, beat.y(t) + constant.y(t) + sine.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t) + constant.dydt(t) + sine.dydt(t))
        np.testing.assert_allclose(
            d2ydt2, beat.d2ydt2(t) + constant.d2ydt2(t) + sine.d2ydt2(t)
        )

    def test__radd__operator(self, beat, t):
        dof_sum = 1.0 + beat
        y, dydt, d2ydt2 = dof_sum(t)

        assert isinstance(dof_sum, _Sum)
        assert isinstance(dof_sum._dofs[0], ConstantDOF)
        assert dof_sum._dofs[1] is beat
        np.testing.assert_allclose(y, 1.0 + beat.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t))

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0)])
    def test__add__operator_scalar(self, beat, value, t):
        dof_sum = beat + value
        y, dydt, d2ydt2 = dof_sum(t)

        assert isinstance(dof_sum, _Sum)
        assert dof_sum._dofs[0] is beat
        assert isinstance(dof_sum._dofs[1], ConstantDOF)
        np.testing.assert_allclose(y, beat.y(t) + 2.0)
        np.testing.assert_allclose(dydt, beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t))

    def test_builtin_sum(self, beat, constant, sine, t):
        dof_sum = sum([beat, constant, sine])
        y, dydt, d2ydt2 = dof_sum(t)

        np.testing.assert_allclose(y, beat.y(t) + constant.y(t) + sine.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t) + constant.dydt(t) + sine.dydt(t))
        np.testing.assert_allclose(
            d2ydt2, beat.d2ydt2(t) + constant.d2ydt2(t) + sine.d2ydt2(t)
        )

    @pytest.mark.parametrize("other", ["a", None, [1.0], 1j])
    def test__add__operator_raises(self, beat, other):
        with pytest.raises(TypeError):
            beat + other

        with pytest.raises(TypeError):
            other + beat

    def test__sub__operator(self, beat, sine, t):
        dof_sum = beat - sine
        y, dydt, d2ydt2 = dof_sum(t)

        assert isinstance(dof_sum, _Sum)
        assert dof_sum._dofs[0] is beat
        np.testing.assert_allclose(y, beat.y(t) - sine.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t) - sine.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t) - sine.d2ydt2(t))

    def test__sub__self(self, beat, t):
        y, dydt, d2ydt2 = (beat - beat)(t)

        np.testing.assert_allclose(y, 0.0)
        np.testing.assert_allclose(dydt, 0.0)
        np.testing.assert_allclose(d2ydt2, 0.0)

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0)])
    def test__sub__operator_scalar(self, beat, value, t):
        dof_sum = beat - value
        y, dydt, d2ydt2 = dof_sum(t)

        assert isinstance(dof_sum, _Sum)
        np.testing.assert_allclose(y, beat.y(t) - 2.0)
        np.testing.assert_allclose(dydt, beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t))

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0)])
    def test__rsub__operator(self, beat, value, t):
        dof_sum = value - beat
        y, dydt, d2ydt2 = dof_sum(t)

        assert isinstance(dof_sum, _Sum)
        np.testing.assert_allclose(y, 2.0 - beat.y(t))
        np.testing.assert_allclose(dydt, -beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, -beat.d2ydt2(t))

    @pytest.mark.parametrize("other", ["a", None, [1.0], 1j])
    def test__sub__operator_raises(self, beat, other):
        with pytest.raises(TypeError):
            beat - other

        with pytest.raises(TypeError):
            other - beat


class Test__Product:
    @pytest.fixture
    def beat(self):
        return BeatDOF(amp=2.0, freq_main=1.0, freq_beat=0.1)

    @pytest.fixture
    def constant(self):
        return ConstantDOF(value=3.0)

    @pytest.fixture
    def sine(self):
        return SineDOF(amp=0.5, freq=1.5, phase=0.3)

    @pytest.fixture
    def dof_product(self, beat, constant, sine):
        return _Product(beat, constant, sine)

    @pytest.fixture
    def expect(self, beat, constant, sine, t):
        a, da, d2a = beat(t)
        b, db, d2b = constant(t)
        c, dc, d2c = sine(t)

        y = a * b * c
        dydt = da * b * c + a * db * c + a * b * dc
        d2ydt2 = (
            d2a * b * c
            + a * d2b * c
            + a * b * d2c
            + 2.0 * (da * db * c + da * b * dc + a * db * dc)
        )

        return y, dydt, d2ydt2

    def test__init__(self, beat, constant, sine):
        dof_product = _Product(beat, constant, sine)

        assert isinstance(dof_product, DOF)
        assert dof_product._dofs == (beat, constant, sine)

    def test__init__raises_empty(self):
        with pytest.raises(ValueError):
            _Product()

    def test__init__raises_type(self, beat):
        with pytest.raises(TypeError):
            _Product(beat, 1.0)

    def test_single(self, beat, t):
        dof_product = _Product(beat)
        y, dydt, d2ydt2 = dof_product(t)
        y_expect, dydt_expect, d2ydt2_expect = beat(t)

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test_y(self, dof_product, expect, t):
        y = dof_product.y(t)
        np.testing.assert_allclose(y, expect[0])

    def test_dydt(self, dof_product, expect, t):
        dydt = dof_product.dydt(t)
        np.testing.assert_allclose(dydt, expect[1])

    def test_d2ydt2(self, dof_product, expect, t):
        d2ydt2 = dof_product.d2ydt2(t)
        np.testing.assert_allclose(d2ydt2, expect[2])

    def test__call__(self, dof_product, expect, t):
        y, dydt, d2ydt2 = dof_product(t)

        np.testing.assert_allclose(y, expect[0])
        np.testing.assert_allclose(dydt, expect[1])
        np.testing.assert_allclose(d2ydt2, expect[2])

    def test_scale_by_constant(self, beat, t):
        dof_product = _Product(ConstantDOF(2.0), beat)
        y, dydt, d2ydt2 = dof_product(t)

        np.testing.assert_allclose(y, 2.0 * beat.y(t))
        np.testing.assert_allclose(dydt, 2.0 * beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, 2.0 * beat.d2ydt2(t))

    def test_multiply_by_zero(self, beat, t):
        dof_product = _Product(beat, ConstantDOF(0.0))
        y, dydt, d2ydt2 = dof_product(t)

        np.testing.assert_allclose(y, 0.0)
        np.testing.assert_allclose(dydt, 0.0)
        np.testing.assert_allclose(d2ydt2, 0.0)

    def test_same_dof_twice(self, beat, t):
        dof_product = _Product(beat, beat)
        y, dydt, d2ydt2 = dof_product(t)
        a, da, d2a = beat(t)

        np.testing.assert_allclose(y, a**2)
        np.testing.assert_allclose(dydt, 2.0 * a * da)
        np.testing.assert_allclose(d2ydt2, 2.0 * da**2 + 2.0 * a * d2a)

    def test_does_not_modify_underlying(self, t):
        # Underlying DOF returns the same arrays on every call, so in-place
        # modification of them would be detected.
        y0, dydt0, d2ydt20 = (
            2 * np.ones_like(t),
            3 * np.ones_like(t),
            4 * np.ones_like(t),
        )

        class SomeDOF(DOF):
            def _evaluate(self, t):
                return y0, dydt0, d2ydt20

        dof_product = _Product(SomeDOF(), SomeDOF())
        dof_product(t)
        dof_product(t)

        np.testing.assert_allclose(y0, 2.0)
        np.testing.assert_allclose(dydt0, 3.0)
        np.testing.assert_allclose(d2ydt20, 4.0)

    def test_nested(self, beat, constant, sine, expect, t):
        dof_product = _Product(_Product(beat, constant), sine)
        y, dydt, d2ydt2 = dof_product(t)

        np.testing.assert_allclose(y, expect[0])
        np.testing.assert_allclose(dydt, expect[1])
        np.testing.assert_allclose(d2ydt2, expect[2])

    def test__mul__operator(self, beat, constant, sine, expect, t):
        dof_product = beat * constant * sine
        y, dydt, d2ydt2 = dof_product(t)

        assert isinstance(dof_product, _Product)
        np.testing.assert_allclose(y, expect[0])
        np.testing.assert_allclose(dydt, expect[1])
        np.testing.assert_allclose(d2ydt2, expect[2])

    def test__rmul__operator(self, beat, t):
        dof_product = 2.0 * beat
        y, dydt, d2ydt2 = dof_product(t)

        assert isinstance(dof_product, _Product)
        assert isinstance(dof_product._dofs[0], ConstantDOF)
        assert dof_product._dofs[1] is beat
        np.testing.assert_allclose(y, 2.0 * beat.y(t))
        np.testing.assert_allclose(dydt, 2.0 * beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, 2.0 * beat.d2ydt2(t))

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0)])
    def test__mul__operator_scalar(self, beat, value, t):
        dof_product = beat * value
        y, dydt, d2ydt2 = dof_product(t)

        assert isinstance(dof_product, _Product)
        assert dof_product._dofs[0] is beat
        assert isinstance(dof_product._dofs[1], ConstantDOF)
        np.testing.assert_allclose(y, 2.0 * beat.y(t))
        np.testing.assert_allclose(dydt, 2.0 * beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, 2.0 * beat.d2ydt2(t))

    @pytest.mark.parametrize("other", ["a", None, [1.0], 1j])
    def test__mul__operator_raises(self, beat, other):
        with pytest.raises(TypeError):
            beat * other

        with pytest.raises(TypeError):
            other * beat

    def test__neg__operator(self, beat, t):
        dof_product = -beat
        y, dydt, d2ydt2 = dof_product(t)

        assert isinstance(dof_product, _Product)
        np.testing.assert_allclose(y, -beat.y(t))
        np.testing.assert_allclose(dydt, -beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, -beat.d2ydt2(t))

    def test__neg__twice(self, beat, t):
        y, dydt, d2ydt2 = (-(-beat))(t)

        np.testing.assert_allclose(y, beat.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t))

    def test_derivatives_by_finite_difference(self, dof_product):
        t = np.linspace(0.0, 20.0, 400_001)
        y, dydt, d2ydt2 = dof_product(t)
        dt = t[1] - t[0]

        dydt_fd = (y[2:] - y[:-2]) / (2.0 * dt)
        d2ydt2_fd = (dydt[2:] - dydt[:-2]) / (2.0 * dt)

        np.testing.assert_allclose(dydt[1:-1], dydt_fd, atol=1e-6)
        np.testing.assert_allclose(d2ydt2[1:-1], d2ydt2_fd, atol=1e-6)
