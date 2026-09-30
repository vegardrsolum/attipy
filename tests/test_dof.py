import numpy as np
import pytest

import attipy as ap
from attipy.simulate._dof import (
    DOF,
    Beat,
    Constant,
    Sine,
    SmootherStep,
    _as_dof,
    _Product,
    _Sum,
    add,
    from_csd,
    from_psd,
    multiply,
    ramp_up,
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


class Test_Beat:
    @pytest.fixture
    def beat(self):
        dof = Beat(amp=2.0, omega=1.0, omega_beat=0.1)
        return dof

    def test__init__(self):
        beat = Beat(
            amp=3.0,
            omega=2.0,
            omega_beat=0.2,
            phase=4.0,
        )

        assert isinstance(beat, DOF)
        assert beat._amp == 3.0
        assert beat._w_main == pytest.approx(2.0)
        assert beat._w_beat == pytest.approx(0.2)
        assert beat._phase == pytest.approx(4.0)

    def test__init__default(self):
        beat_dof = Beat()

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


class Test_Constant:
    @pytest.fixture
    def constant(self):
        dof = Constant(value=2.0)
        return dof

    def test__init__(self):
        constant = Constant(value=3.0)

        assert isinstance(constant, DOF)
        assert constant._value == 3.0

    def test__init__default(self):
        constant = Constant()

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


class Test_Sine:
    @pytest.fixture
    def sine(self):
        dof = Sine(amp=2.0, omega=3.0, phase=0.5)
        return dof

    def test__init__(self):
        sine = Sine(
            amp=3.0,
            omega=2.0,
            phase=4.0,
        )

        assert isinstance(sine, DOF)
        assert sine._amp == 3.0
        assert sine._w == pytest.approx(2.0)
        assert sine._phase == pytest.approx(4.0)

    def test__init__default(self):
        sine = Sine()

        assert isinstance(sine, DOF)
        assert sine._amp == 1.0
        assert sine._w == pytest.approx(1.0)
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


class Test_SmootherStep:
    @pytest.fixture
    def step(self):
        return SmootherStep(duration=4.0, start=2.0)

    @pytest.fixture
    def expect(self, t):
        x = np.clip((t - 2.0) / 4.0, 0.0, 1.0)
        w = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
        dw = (30.0 * x**4 - 60.0 * x**3 + 30.0 * x**2) / 4.0
        d2w = (120.0 * x**3 - 180.0 * x**2 + 60.0 * x) / 4.0**2
        return w, dw, d2w

    def test__init__(self):
        step = SmootherStep(duration=4.0, start=2.0)

        assert isinstance(step, DOF)
        assert step._duration == 4.0
        assert step._start == 2.0

    def test__init__default(self):
        step = SmootherStep()

        assert step._duration == 100.0
        assert step._start == 0.0

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test__init__raises_duration(self, duration):
        with pytest.raises(ValueError):
            SmootherStep(duration=duration)

    def test__init__raises_start(self):
        with pytest.raises(ValueError):
            SmootherStep(duration=4.0, start=-1.0)

    def test_values(self, step):
        t = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # before, start, mid, end, after
        w, dw, d2w = step(t)

        np.testing.assert_allclose(w, [0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(dw, [0.0, 0.0, 30.0 * 0.5**4 / 4.0, 0.0, 0.0])
        np.testing.assert_allclose(d2w, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_y(self, step, expect, t):
        np.testing.assert_allclose(step.y(t), expect[0])

    def test_dydt(self, step, expect, t):
        np.testing.assert_allclose(step.dydt(t), expect[1])

    def test_d2ydt2(self, step, expect, t):
        np.testing.assert_allclose(step.d2ydt2(t), expect[2], atol=1e-12)

    def test_derivatives_by_finite_difference(self, step):
        t = np.linspace(0.0, 20.0, 400_001)
        w, dw, d2w = step(t)
        dt = t[1] - t[0]

        dw_fd = (w[2:] - w[:-2]) / (2.0 * dt)
        d2w_fd = (dw[2:] - dw[:-2]) / (2.0 * dt)

        # Central differences are inaccurate where the jerk is discontinuous,
        # i.e., at the two ends of the step period.
        t_mid = t[1:-1]
        valid = (np.abs(t_mid - 2.0) > dt) & (np.abs(t_mid - 6.0) > dt)

        np.testing.assert_allclose(dw[1:-1][valid], dw_fd[valid], atol=1e-6)
        np.testing.assert_allclose(d2w[1:-1][valid], d2w_fd[valid], atol=1e-6)


class Test_ramp_up:
    @pytest.fixture
    def beat(self):
        return Beat(omega=1.0, omega_beat=0.1)

    @pytest.fixture
    def rampup(self, beat):
        return ramp_up(beat, duration=4.0, start=2.0)

    @pytest.fixture
    def window(self, t):
        x = np.clip((t - 2.0) / 4.0, 0.0, 1.0)
        w = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
        dw = (30.0 * x**4 - 60.0 * x**3 + 30.0 * x**2) / 4.0
        d2w = (120.0 * x**3 - 180.0 * x**2 + 60.0 * x) / 4.0**2
        return w, dw, d2w

    def test_ramp_times_dof(self, rampup, beat):
        assert isinstance(rampup, DOF)
        assert isinstance(rampup, _Product)
        assert len(rampup._dofs) == 2
        assert isinstance(rampup._dofs[0], SmootherStep)
        assert rampup._dofs[1] is beat

    def test_keywords(self, beat):
        ramp = ramp_up(beat, duration=4.0, start=2.0)._dofs[0]

        assert ramp._duration == 4.0
        assert ramp._start == 2.0

    def test_default(self, beat):
        ramp = ramp_up(beat)._dofs[0]

        assert ramp._duration == 100.0
        assert ramp._start == 0.0

    def test_y(self, rampup, beat, window, t):
        w, _, _ = window

        np.testing.assert_allclose(rampup.y(t), w * beat.y(t))

    def test_dydt(self, rampup, beat, window, t):
        w, dw, _ = window

        np.testing.assert_allclose(rampup.dydt(t), dw * beat.y(t) + w * beat.dydt(t))

    def test_d2ydt2(self, rampup, beat, window, t):
        w, dw, d2w = window
        d2ydt2_expect = d2w * beat.y(t) + 2.0 * dw * beat.dydt(t) + w * beat.d2ydt2(t)

        np.testing.assert_allclose(rampup.d2ydt2(t), d2ydt2_expect, atol=1e-12)

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

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0), np.int64(2)])
    def test_real(self, value, window, t):
        w, dw, d2w = window
        y, dydt, d2ydt2 = ramp_up(value, duration=4.0, start=2.0)(t)

        np.testing.assert_allclose(y, 2.0 * w)
        np.testing.assert_allclose(dydt, 2.0 * dw)
        np.testing.assert_allclose(d2ydt2, 2.0 * d2w, atol=1e-12)

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_raises_duration(self, beat, duration):
        with pytest.raises(ValueError):
            ramp_up(beat, duration=duration)

    def test_raises_start(self, beat):
        with pytest.raises(ValueError):
            ramp_up(beat, duration=4.0, start=-1.0)

    @pytest.mark.parametrize("other", ["a", None, [1.0], np.array([1.0]), 1j])
    def test_raises_type(self, other):
        with pytest.raises(TypeError):
            ramp_up(other)


class Test__as_dof:
    def test_dof(self):
        dof = Beat()
        assert _as_dof(dof) is dof

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0), np.int64(2)])
    def test_real(self, value, t):
        dof = _as_dof(value)

        assert isinstance(dof, Constant)
        assert dof._value == 2.0
        assert type(dof._value) is float
        np.testing.assert_allclose(dof.y(t), 2.0)

    @pytest.mark.parametrize("other", ["a", None, [1.0], np.array([1.0]), 1j])
    def test_unsupported_raises(self, other):
        with pytest.raises(TypeError):
            _as_dof(other)


class Test__Sum:
    @pytest.fixture
    def beat(self):
        return Beat(omega=1.0, omega_beat=0.1)

    @pytest.fixture
    def constant(self):
        return Constant(value=3.0)

    @pytest.fixture
    def sine(self):
        return Sine(omega=1.5, phase=0.3)

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

    def test_flattens_nested(self, beat, constant, sine):
        assert _Sum(_Sum(beat, constant), sine)._dofs == (beat, constant, sine)
        assert _Sum(beat, _Sum(constant, sine))._dofs == (beat, constant, sine)

    def test_does_not_flatten_product(self, beat, constant, sine):
        product = _Product(constant, sine)
        assert _Sum(beat, product)._dofs == (beat, product)


class Test__Product:
    @pytest.fixture
    def beat(self):
        return Beat(omega=1.0, omega_beat=0.1)

    @pytest.fixture
    def constant(self):
        return Constant(value=3.0)

    @pytest.fixture
    def sine(self):
        return Sine(omega=1.5, phase=0.3)

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
        dof_product = _Product(Constant(2.0), beat)
        y, dydt, d2ydt2 = dof_product(t)

        np.testing.assert_allclose(y, 2.0 * beat.y(t))
        np.testing.assert_allclose(dydt, 2.0 * beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, 2.0 * beat.d2ydt2(t))

    def test_multiply_by_zero(self, beat, t):
        dof_product = _Product(beat, Constant(0.0))
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

    def test_flattens_nested(self, beat, constant, sine):
        assert _Product(_Product(beat, constant), sine)._dofs == (beat, constant, sine)
        assert _Product(beat, _Product(constant, sine))._dofs == (beat, constant, sine)

    def test_does_not_flatten_sum(self, beat, constant, sine):
        dof_sum = _Sum(constant, sine)
        assert _Product(beat, dof_sum)._dofs == (beat, dof_sum)

    def test_derivatives_by_finite_difference(self, dof_product):
        t = np.linspace(0.0, 20.0, 400_001)
        y, dydt, d2ydt2 = dof_product(t)
        dt = t[1] - t[0]

        dydt_fd = (y[2:] - y[:-2]) / (2.0 * dt)
        d2ydt2_fd = (dydt[2:] - dydt[:-2]) / (2.0 * dt)

        np.testing.assert_allclose(dydt[1:-1], dydt_fd, atol=1e-6)
        np.testing.assert_allclose(d2ydt2[1:-1], d2ydt2_fd, atol=1e-6)


class Test_add:
    @pytest.fixture
    def beat(self):
        return Beat(omega=1.0, omega_beat=0.1)

    @pytest.fixture
    def sine(self):
        return Sine(omega=1.5, phase=0.3)

    def test_dofs(self, beat, sine, t):
        dof_sum = add(beat, sine)
        y, dydt, d2ydt2 = dof_sum(t)

        assert isinstance(dof_sum, DOF)
        np.testing.assert_allclose(y, beat.y(t) + sine.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t) + sine.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t) + sine.d2ydt2(t))

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0), np.int64(2)])
    def test_real(self, beat, value, t):
        y, dydt, d2ydt2 = add(value, beat)(t)

        np.testing.assert_allclose(y, 2.0 + beat.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t))

    def test_reals_only(self, t):
        y, dydt, d2ydt2 = add(1.0, 2.0)(t)

        np.testing.assert_allclose(y, 3.0)
        np.testing.assert_allclose(dydt, 0.0)
        np.testing.assert_allclose(d2ydt2, 0.0)

    def test_nested(self, beat, sine, t):
        dof_sum = add(add(beat, 1.0), sine)
        y, dydt, d2ydt2 = dof_sum(t)

        assert len(dof_sum._dofs) == 3
        np.testing.assert_allclose(y, beat.y(t) + 1.0 + sine.y(t))
        np.testing.assert_allclose(dydt, beat.dydt(t) + sine.dydt(t))
        np.testing.assert_allclose(d2ydt2, beat.d2ydt2(t) + sine.d2ydt2(t))

    def test_multiply_nested(self, beat, sine, t):
        y, dydt, d2ydt2 = add(multiply(2.0, beat), sine)(t)

        np.testing.assert_allclose(y, 2.0 * beat.y(t) + sine.y(t))
        np.testing.assert_allclose(dydt, 2.0 * beat.dydt(t) + sine.dydt(t))
        np.testing.assert_allclose(d2ydt2, 2.0 * beat.d2ydt2(t) + sine.d2ydt2(t))

    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_raises_wrong_number_of_args(self, beat, n):
        with pytest.raises(TypeError):
            add(*[beat] * n)

    def test_raises_keyword_args(self, beat, sine):
        with pytest.raises(TypeError):
            add(dof1=beat, dof2=sine)

    @pytest.mark.parametrize("other", ["a", None, [1.0], np.array([1.0]), 1j])
    def test_raises_type(self, beat, other):
        with pytest.raises(TypeError):
            add(beat, other)


class Test_multiply:
    @pytest.fixture
    def beat(self):
        return Beat(omega=1.0, omega_beat=0.1)

    @pytest.fixture
    def sine(self):
        return Sine(omega=1.5, phase=0.3)

    def test_dofs(self, beat, sine, t):
        dof_product = multiply(beat, sine)
        y, dydt, d2ydt2 = dof_product(t)
        a, da, d2a = beat(t)
        b, db, d2b = sine(t)

        assert isinstance(dof_product, DOF)
        np.testing.assert_allclose(y, a * b)
        np.testing.assert_allclose(dydt, da * b + a * db)
        np.testing.assert_allclose(d2ydt2, d2a * b + 2.0 * da * db + a * d2b)

    @pytest.mark.parametrize("value", [2, 2.0, np.float64(2.0), np.int64(2)])
    def test_real(self, beat, value, t):
        y, dydt, d2ydt2 = multiply(value, beat)(t)

        np.testing.assert_allclose(y, 2.0 * beat.y(t))
        np.testing.assert_allclose(dydt, 2.0 * beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, 2.0 * beat.d2ydt2(t))

    def test_negate(self, beat, t):
        y, dydt, d2ydt2 = multiply(-1.0, beat)(t)

        np.testing.assert_allclose(y, -beat.y(t))
        np.testing.assert_allclose(dydt, -beat.dydt(t))
        np.testing.assert_allclose(d2ydt2, -beat.d2ydt2(t))

    def test_reals_only(self, t):
        y, dydt, d2ydt2 = multiply(2.0, 3.0)(t)

        np.testing.assert_allclose(y, 6.0)
        np.testing.assert_allclose(dydt, 0.0)
        np.testing.assert_allclose(d2ydt2, 0.0)

    def test_nested(self, beat, sine, t):
        dof_product = multiply(multiply(2.0, beat), sine)
        y, dydt, d2ydt2 = dof_product(t)
        a, da, d2a = beat(t)
        b, db, d2b = sine(t)

        assert len(dof_product._dofs) == 3
        np.testing.assert_allclose(y, 2.0 * a * b)
        np.testing.assert_allclose(dydt, 2.0 * (da * b + a * db))
        np.testing.assert_allclose(d2ydt2, 2.0 * (d2a * b + 2.0 * da * db + a * d2b))

    def test_add_nested(self, beat, sine, t):
        y, dydt, d2ydt2 = multiply(add(beat, 1.0), sine)(t)
        a, da, d2a = beat(t)
        b, db, d2b = sine(t)

        np.testing.assert_allclose(y, (a + 1.0) * b)
        np.testing.assert_allclose(dydt, da * b + (a + 1.0) * db)
        np.testing.assert_allclose(d2ydt2, d2a * b + 2.0 * da * db + (a + 1.0) * d2b)

    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_raises_wrong_number_of_args(self, beat, n):
        with pytest.raises(TypeError):
            multiply(*[beat] * n)

    def test_raises_keyword_args(self, beat, sine):
        with pytest.raises(TypeError):
            multiply(dof1=beat, dof2=sine)

    @pytest.mark.parametrize("other", ["a", None, [1.0], np.array([1.0]), 1j])
    def test_raises_type(self, beat, other):
        with pytest.raises(TypeError):
            multiply(beat, other)


class Test_public_api:
    def test_dof_submodule(self):
        dof = ap.simulate.dof

        assert dof.DOF is DOF
        assert dof.Beat is Beat
        assert dof.Constant is Constant
        assert dof.Sine is Sine
        assert dof.SmootherStep is SmootherStep
        assert dof.add is add
        assert dof.from_csd is from_csd
        assert dof.from_psd is from_psd
        assert dof.multiply is multiply
        assert dof.ramp_up is ramp_up
        assert set(dof.__all__) == {
            "DOF",
            "Beat",
            "Constant",
            "Sine",
            "SmootherStep",
            "add",
            "from_csd",
            "from_psd",
            "multiply",
            "ramp_up",
        }

    def test_usage(self):
        t = np.linspace(0.0, 10.0, 100)
        motion = ap.simulate.Motion(
            x=ap.simulate.dof.add(ap.simulate.dof.Beat(amp=2.0), 0.5),
            yaw=ap.simulate.dof.ramp_up(ap.simulate.dof.Sine(amp=0.1), duration=5.0),
        )

        np.testing.assert_allclose(motion.x.y(t), 2.0 * Beat().y(t) + 0.5)
        np.testing.assert_allclose(
            motion.yaw.y(t), ramp_up(Sine(amp=0.1), duration=5.0).y(t)
        )


class Test_from_psd:
    @pytest.fixture
    def freq(self):
        return np.linspace(0.0, 1.0, 101)

    @pytest.fixture
    def psd(self, freq):
        return np.exp(-(((freq - 0.2) / 0.05) ** 2))

    def test_components(self):
        # Triangular PSD with peak 2.0 at 1 Hz, and area 2.0
        freq = [0.0, 1.0, 2.0]
        psd = [0.0, 2.0, 0.0]

        dof = from_psd(freq, psd, 4, seed=1)

        # Bins [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 2] with areas 0.25, 0.75, 0.75, 0.25
        freq_expect = np.array([0.25, 0.75, 1.25, 1.75])
        amp_expect = np.sqrt(2.0 * np.array([0.25, 0.75, 0.75, 0.25]))

        assert isinstance(dof, _Sum)
        assert all(isinstance(s, Sine) for s in dof._dofs)
        np.testing.assert_allclose([s._w for s in dof._dofs], 2.0 * np.pi * freq_expect)
        np.testing.assert_allclose([s._amp for s in dof._dofs], amp_expect)
        assert all(0.0 <= s._phase < 2.0 * np.pi for s in dof._dofs)

    def test_single_component(self):
        dof = from_psd([0.0, 2.0], [0.5, 0.5], 1, seed=1)

        # Component at 1 Hz with variance equal to the area under the PSD
        (sine,) = dof._dofs
        assert sine._w == pytest.approx(2.0 * np.pi)
        assert sine._amp == pytest.approx(np.sqrt(2.0))

    @pytest.mark.parametrize("n_components", [1, 7, 33])
    @pytest.mark.parametrize("jitter", [0.0, 1.0])
    def test_variance_equals_psd_area(self, freq, psd, n_components, jitter):
        dof = from_psd(freq, psd, n_components, jitter=jitter, seed=1)

        var = sum(s._amp**2 / 2.0 for s in dof._dofs)
        assert var == pytest.approx(np.trapezoid(psd, freq))

    def test_narrow_peak_between_bin_centers(self):
        # Narrow peak at 0.25 Hz, exactly between the bin centers 0.125 and 0.375 Hz
        freq = np.linspace(0.0, 1.0, 1001)
        psd = np.where(np.abs(freq - 0.25) <= 0.005, 1.0, 0.0)

        dof = from_psd(freq, psd, 4, seed=1)

        # The peak is split equally between the two neighbouring bins
        var_k = np.array([s._amp**2 / 2.0 for s in dof._dofs])
        area = np.trapezoid(psd, freq)
        np.testing.assert_allclose(
            var_k, [area / 2.0, area / 2.0, 0.0, 0.0], atol=1e-15
        )

    @pytest.mark.parametrize("jitter", [0.5, 1.0])
    def test_jitter(self, freq, psd, jitter):
        dof = from_psd(freq, psd, 4, jitter=jitter, seed=1)
        dof_ref = from_psd(freq, psd, 4, jitter=0.0, seed=1)

        df = 0.25
        edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        freq_center = np.array([0.125, 0.375, 0.625, 0.875])
        freq_jitter = np.array([s._w for s in dof._dofs]) / (2.0 * np.pi)
        freq_ref = np.array([s._w for s in dof_ref._dofs]) / (2.0 * np.pi)

        # Frequencies are moved within their respective bins.
        assert not np.allclose(freq_jitter, freq_ref)
        assert np.all(np.abs(freq_jitter - freq_center) <= 0.5 * jitter * df)
        assert np.all(freq_jitter >= edges[:-1])
        assert np.all(freq_jitter <= edges[1:])

        # Amplitudes and phases are not affected by jitter
        np.testing.assert_allclose(
            [s._amp for s in dof._dofs], [s._amp for s in dof_ref._dofs]
        )
        np.testing.assert_allclose(
            [s._phase for s in dof._dofs], [s._phase for s in dof_ref._dofs]
        )

    def test_seed(self, freq, psd):
        phase1 = [s._phase for s in from_psd(freq, psd, 20, seed=1)._dofs]
        phase2 = [s._phase for s in from_psd(freq, psd, 20, seed=1)._dofs]
        phase3 = [s._phase for s in from_psd(freq, psd, 20, seed=2)._dofs]

        np.testing.assert_allclose(phase1, phase2)
        assert not np.allclose(phase1, phase3)

    def test_seed_generator(self, freq, psd):
        rng = np.random.default_rng(1)
        phase1 = [s._phase for s in from_psd(freq, psd, 20, seed=rng)._dofs]
        phase2 = [s._phase for s in from_psd(freq, psd, 20, seed=1)._dofs]

        np.testing.assert_allclose(phase1, phase2)

    def test_welch(self, freq, psd):
        from scipy.signal import welch

        dof = from_psd(freq, psd, 500, seed=1)

        fs = 2.0
        t = np.arange(0.0, 20_000.0, 1.0 / fs)
        freq_out, psd_out = welch(dof.y(t), fs=fs, nperseg=1024)

        np.testing.assert_allclose(
            psd_out, np.interp(freq_out, freq, psd), atol=0.05 * psd.max()
        )

    def test_zero_psd(self):
        freq = np.array([0.0, 1.0, 2.0])
        psd = np.array([0.0, 0.0, 0.0])
        dof = from_psd(freq, psd, 3)
        t = np.linspace(0.0, 1.0, 100)
        y = dof.y(t)
        np.testing.assert_allclose(y, 0.0)

    @pytest.mark.parametrize(
        "freq, psd",
        [
            ([0.0], [1.0]),  # too few values
            ([[0.0, 1.0]], [[1.0, 1.0]]),  # not 1D
            ([0.0, 1.0], [1.0, 1.0, 1.0]),  # shape mismatch
            ([-1.0, 1.0], [1.0, 1.0]),  # negative frequency
            ([0.0, 1.0, 1.0], [1.0, 1.0, 1.0]),  # not strictly increasing
            ([1.0, 0.0], [1.0, 1.0]),  # decreasing
            ([0.0, 1.0], [1.0, -1.0]),  # negative psd
        ],
    )
    def test_raises_input(self, freq, psd):
        with pytest.raises(ValueError):
            from_psd(freq, psd, 10)

    @pytest.mark.parametrize("n_components", [0, -1, 1.5])
    def test_raises_n_components(self, freq, psd, n_components):
        with pytest.raises(ValueError):
            from_psd(freq, psd, n_components)

    @pytest.mark.parametrize("jitter", [-0.1, 1.1])
    def test_raises_jitter(self, freq, psd, jitter):
        with pytest.raises(ValueError):
            from_psd(freq, psd, 10, jitter=jitter)


class Test_from_csd:
    @pytest.fixture
    def freq(self):
        return np.linspace(0.0, 1.0, 101)

    @pytest.fixture
    def psd(self, freq):
        return np.exp(-(((freq - 0.2) / 0.05) ** 2))

    @staticmethod
    def csd_2x2(psd, h, coherence):
        # Signals y_0 and y_1 = h * y_0, with the given coherence
        csd = np.zeros((psd.size, 2, 2), dtype=np.complex128)
        csd[:, 0, 0] = psd
        csd[:, 1, 1] = np.abs(h) ** 2 * psd
        csd[:, 0, 1] = np.sqrt(coherence) * h * psd
        csd[:, 1, 0] = np.conj(csd[:, 0, 1])
        return csd

    def test_returns_tuple_of_sums_of_sines(self, freq, psd):
        dofs = from_csd(freq, self.csd_2x2(psd, 2.0, 0.5), 10, seed=1)

        assert isinstance(dofs, tuple)
        assert len(dofs) == 2
        for dof in dofs:
            assert isinstance(dof, _Sum)
            assert len(dof._dofs) == 10
            assert all(isinstance(s, Sine) for s in dof._dofs)

    def test_frequencies(self, freq, psd):
        dofs = from_csd(freq, self.csd_2x2(psd, 2.0, 0.5), 4, seed=1)

        # Bins [0, 0.5] and [0.5, 1], each with two sub-bins
        freq_expect = np.array([0.125, 0.375, 0.625, 0.875])
        for dof in dofs:
            np.testing.assert_allclose(
                [s._w for s in dof._dofs], 2.0 * np.pi * freq_expect
            )

    def test_uncorrelated(self):
        # Triangular PSD with peak 2.0 at 1 Hz, and area 2.0
        freq = [0.0, 1.0, 2.0]
        psd = np.array([0.0, 2.0, 0.0])
        csd = np.zeros((3, 2, 2))
        csd[:, 0, 0] = psd
        csd[:, 1, 1] = 4.0 * psd

        y0, y1 = from_csd(freq, csd, 4, seed=1)

        # Bins [0, 1] and [1, 2], with areas 1.0 and 4.0 for y_0 and y_1
        var0_k = [s._amp**2 / 2.0 for s in y0._dofs]
        var1_k = [s._amp**2 / 2.0 for s in y1._dofs]
        np.testing.assert_allclose(
            np.sum(np.reshape(var0_k, (2, 2)), axis=1), [1.0, 1.0]
        )
        np.testing.assert_allclose(
            np.sum(np.reshape(var1_k, (2, 2)), axis=1), [4.0, 4.0]
        )

    def test_fully_coherent(self, freq, psd):
        # y_1 is y_0 scaled by 2 and phase shifted by +90 degrees
        y0, y1 = from_csd(freq, self.csd_2x2(psd, 2.0j, 1.0), 10, seed=1)

        amp0 = np.array([s._amp for s in y0._dofs])
        amp1 = np.array([s._amp for s in y1._dofs])
        phase0 = np.array([s._phase for s in y0._dofs])
        phase1 = np.array([s._phase for s in y1._dofs])

        # Only one source is active, and its components are shared by y_0 and y_1
        active = amp0 > 1e-6 * amp0.max()
        np.testing.assert_allclose(amp1, 2.0 * amp0, atol=1e-12)
        np.testing.assert_allclose(
            np.mod(phase1[active] - phase0[active], 2.0 * np.pi), np.pi / 2.0
        )

    @pytest.mark.parametrize("coherence", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("jitter", [0.0, 1.0])
    def test_variance_equals_psd_area(self, freq, psd, coherence, jitter):
        dofs = from_csd(
            freq, self.csd_2x2(psd, 2.0j, coherence), 14, jitter=jitter, seed=1
        )

        var = [sum(s._amp**2 / 2.0 for s in dof._dofs) for dof in dofs]
        area = np.trapezoid(psd, freq)
        np.testing.assert_allclose(var, [area, 4.0 * area])

    @pytest.mark.parametrize("jitter", [0.5, 1.0])
    def test_jitter(self, freq, psd, jitter):
        csd = self.csd_2x2(psd, 2.0, 0.5)
        y0, y1 = from_csd(freq, csd, 4, jitter=jitter, seed=1)
        y0_ref, _ = from_csd(freq, csd, 4, seed=1)

        df_sub = 0.25
        freq_center = np.array([0.125, 0.375, 0.625, 0.875])
        freq0 = np.array([s._w for s in y0._dofs]) / (2.0 * np.pi)
        freq1 = np.array([s._w for s in y1._dofs]) / (2.0 * np.pi)

        # Frequencies are moved within their sub-bins, equally for all signals
        np.testing.assert_allclose(freq0, freq1)
        assert not np.allclose(freq0, freq_center)
        assert np.all(np.abs(freq0 - freq_center) <= 0.5 * jitter * df_sub)

        # Amplitudes and phases are not affected by jitter
        np.testing.assert_allclose(
            [s._amp for s in y0._dofs], [s._amp for s in y0_ref._dofs]
        )
        np.testing.assert_allclose(
            [s._phase for s in y0._dofs], [s._phase for s in y0_ref._dofs]
        )

    def test_seed(self, freq, psd):
        csd = self.csd_2x2(psd, 2.0, 0.5)
        phase1 = [s._phase for s in from_csd(freq, csd, 20, seed=1)[0]._dofs]
        phase2 = [s._phase for s in from_csd(freq, csd, 20, seed=1)[0]._dofs]
        phase3 = [s._phase for s in from_csd(freq, csd, 20, seed=2)[0]._dofs]

        np.testing.assert_allclose(phase1, phase2)
        assert not np.allclose(phase1, phase3)

    def test_scipy_csd(self, freq, psd):
        from scipy.signal import csd as scipy_csd

        csd = self.csd_2x2(psd, 2.0j, 0.5)
        y0, y1 = from_csd(freq, csd, 500, jitter=1.0, seed=1)

        fs = 2.0
        t = np.arange(0.0, 20_000.0, 1.0 / fs)
        y = [y0.y(t), y1.y(t)]
        for i in range(2):
            for j in range(2):
                freq_out, csd_out = scipy_csd(y[i], y[j], fs=fs, nperseg=256)
                csd_expect = np.interp(
                    freq_out, freq, csd[:, i, j].real
                ) + 1j * np.interp(freq_out, freq, csd[:, i, j].imag)
                np.testing.assert_allclose(
                    csd_out, csd_expect, atol=0.1 * np.abs(csd[:, i, j]).max()
                )

    def test_psd_equivalence(self, freq, psd):
        (y,) = from_csd(freq, psd.reshape(-1, 1, 1), 10, jitter=1.0, seed=1)
        y_psd = from_psd(freq, psd, 10, jitter=1.0, seed=1)

        t = np.linspace(0.0, 100.0, 1001)
        np.testing.assert_allclose(y.y(t), y_psd.y(t))

    @pytest.mark.parametrize(
        "csd",
        [
            np.ones((3, 2)),  # not 3D
            np.ones((2, 2, 2)),  # wrong number of frequencies
            np.ones((3, 2, 3)),  # not square
            np.tile([[1.0, 1.0j], [1.0j, 1.0]], (3, 1, 1)),  # not Hermitian
            np.tile([[1.0, 2.0], [2.0, 1.0]], (3, 1, 1)),  # not positive semidefinite
        ],
    )
    def test_raises_csd(self, csd):
        with pytest.raises(ValueError):
            from_csd([0.0, 1.0, 2.0], csd, 10)

    @pytest.mark.parametrize(
        "freq",
        [
            [0.0],  # too few values
            [[0.0, 1.0]],  # not 1D
            [-1.0, 1.0],  # negative frequency
            [1.0, 0.0],  # decreasing
        ],
    )
    def test_raises_freq(self, freq):
        with pytest.raises(ValueError):
            from_csd(freq, np.ones((2, 1, 1)), 10)

    @pytest.mark.parametrize("n_components", [0, -1, 1.5])
    def test_raises_n_components(self, freq, psd, n_components):
        with pytest.raises(ValueError):
            from_csd(freq, psd.reshape(-1, 1, 1), n_components)

    def test_raises_n_components_not_multiple(self, freq, psd):
        with pytest.raises(ValueError):
            from_csd(freq, self.csd_2x2(psd, 2.0, 0.5), 5)

    @pytest.mark.parametrize("jitter", [-0.1, 1.1])
    def test_raises_jitter(self, freq, psd, jitter):
        with pytest.raises(ValueError):
            from_csd(freq, psd.reshape(-1, 1, 1), 10, jitter=jitter)
