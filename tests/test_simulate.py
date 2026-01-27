import numpy as np
import pytest

import attipy as ap
from attipy._simulate import DOF, BeatDOF, ChirpDOF, ConstantDOF, SineDOF


@pytest.fixture
def t():
    return np.linspace(0, 10, 100)


class Test_DOF:
    @pytest.fixture
    def some_dof(self):
        class SomeDOF(DOF):

            def _y(self, t):
                return np.ones_like(t)

            def _dydt(self, t):
                return 2 * np.ones_like(t)

            def _d2ydt2(self, t):
                return 3 * np.ones_like(t)

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


class Test_ConstantDOF:
    @pytest.fixture
    def constant_dof(self):
        return ConstantDOF(value=5.0)

    def test__init__(self):
        constant_dof = ConstantDOF(value=123.0)
        assert isinstance(constant_dof, DOF)
        assert constant_dof._value == 123.0

    def test_y(self, constant_dof, t):
        y = constant_dof.y(t)
        np.testing.assert_allclose(y, 5.0 * np.ones(100))

    def test_dydt(self, constant_dof, t):
        dydt = constant_dof.dydt(t)
        np.testing.assert_allclose(dydt, np.zeros(100))

    def test_d2ydt2(self, constant_dof, t):
        d2ydt2 = constant_dof.d2ydt2(t)
        np.testing.assert_allclose(d2ydt2, np.zeros(100))

    def test__call__(self, constant_dof, t):
        y, dydt, dy2dt2 = constant_dof(t)
        np.testing.assert_allclose(y, 5.0 * np.ones(100))
        np.testing.assert_allclose(dydt, np.zeros(100))
        np.testing.assert_allclose(dy2dt2, np.zeros(100))


class Test_SineDOF:
    @pytest.fixture
    def sine_dof(self):
        return SineDOF(2.0, 1.0)

    def test__init__(self):
        sine_dof = SineDOF(
            amp=2.0, freq=3.0, freq_hz=True, phase=4.0, phase_degrees=True, offset=5.0
        )

        assert isinstance(sine_dof, DOF)
        assert sine_dof._amp == 2.0
        assert sine_dof._w == pytest.approx(2.0 * np.pi * 3.0)
        assert sine_dof._phase == pytest.approx((np.pi / 180.0) * 4.0)
        assert sine_dof._offset == 5.0

    def test_y(self, sine_dof, t):
        y = sine_dof.y(t)
        expected_y = 2.0 * np.sin(1.0 * t + 0.0)
        np.testing.assert_allclose(y, expected_y)

    def test_dydt(self, sine_dof, t):
        dydt = sine_dof.dydt(t)
        expected_dydt = 2.0 * 1.0 * np.cos(1.0 * t + 0.0)
        np.testing.assert_allclose(dydt, expected_dydt)

    def test_d2ydt2(self, sine_dof, t):
        d2ydt2 = sine_dof.d2ydt2(t)
        expected_d2ydt2 = -2.0 * (1.0**2) * np.sin(1.0 * t + 0.0)
        np.testing.assert_allclose(d2ydt2, expected_d2ydt2)

    def test__call__(self, sine_dof, t):
        y, dydt, dy2dt2 = sine_dof(t)
        expected_y = 2.0 * np.sin(1.0 * t + 0.0)
        expected_dydt = 2.0 * 1.0 * np.cos(1.0 * t + 0.0)
        expected_d2ydt2 = -2.0 * (1.0**2) * np.sin(1.0 * t + 0.0)
        np.testing.assert_allclose(y, expected_y)
        np.testing.assert_allclose(dydt, expected_dydt)
        np.testing.assert_allclose(dy2dt2, expected_d2ydt2)

    def test_amp(self):
        sine_dof = SineDOF(amp=3.0)
        t = np.linspace(0, 2 * np.pi, 100)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = 3.0 * np.sin(t)
        dydt_expect = 3.0 * np.cos(t)
        dy2dt2_expect = -3.0 * np.sin(t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_freq_hz(self, t):
        sine_dof = SineDOF(freq=0.5, freq_hz=True)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(np.pi * t)
        dydt_expect = np.pi * np.cos(np.pi * t)
        dy2dt2_expect = -np.pi**2 * np.sin(np.pi * t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_freq_rads(self, t):
        sine_dof = SineDOF(freq=np.pi, freq_hz=False)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(np.pi * t)
        dydt_expect = np.pi * np.cos(np.pi * t)
        dy2dt2_expect = -np.pi**2 * np.sin(np.pi * t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_phase_degrees(self, t):
        sine_dof = SineDOF(phase=90.0, phase_degrees=True)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(t + np.pi / 2)
        dydt_expect = np.cos(t + np.pi / 2)
        dy2dt2_expect = -np.sin(t + np.pi / 2)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_phase_radians(self, t):
        sine_dof = SineDOF(phase=np.pi / 2, phase_degrees=False)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = np.sin(t + np.pi / 2)
        dydt_expect = np.cos(t + np.pi / 2)
        dy2dt2_expect = -np.sin(t + np.pi / 2)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)

    def test_offset(self, t):
        sine_dof = SineDOF(offset=2.0)
        y, dydt, dy2dt2 = sine_dof(t)
        y_expect = 2.0 + np.sin(t)
        dydt_expect = np.cos(t)
        dy2dt2_expect = -np.sin(t)
        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(dy2dt2, dy2dt2_expect)


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


class Test_ChirpDOF:
    @pytest.fixture
    def chirp(self):
        chirp = ChirpDOF(amp=2.0, f_max=0.25, f_os=0.01, freq_hz=True, offset=1.0)
        return chirp

    def test__init__(self):
        chirp_dof = ChirpDOF(
            3.0, 2.0, 1.0, freq_hz=True, phase=4.0, phase_degrees=True, offset=5.0
        )

        assert isinstance(chirp_dof, DOF)
        assert chirp_dof._amp == 3.0
        assert chirp_dof._w_max == pytest.approx(2.0 * np.pi * 2.0)
        assert chirp_dof._w_os == pytest.approx(2.0 * np.pi * 1.0)
        assert chirp_dof._phase == pytest.approx((np.pi / 180.0) * 4.0)
        assert chirp_dof._offset == 5.0

    def test__init__default(self):
        chirp_dof = ChirpDOF()

        assert isinstance(chirp_dof, DOF)
        assert chirp_dof._amp == 1.0
        assert chirp_dof._w_max == pytest.approx(np.pi / 2.0)
        assert chirp_dof._w_os == pytest.approx(2.0 * np.pi * 0.01)
        assert chirp_dof._phase == pytest.approx(0.0)
        assert chirp_dof._offset == 0.0

    def test_y(self, chirp, t):
        y = chirp.y(t)

        amp = chirp._amp
        w_max = chirp._w_max
        w_os = chirp._w_os
        phase = chirp._phase
        offset = chirp._offset

        phi = 2.0 * w_max / w_os * np.sin(w_os / 2.0 * t)
        y_expect = amp * np.sin(phi + phase) + offset

        np.testing.assert_allclose(y, y_expect)

    def test_dydt(self, chirp, t):
        dydt = chirp.dydt(t)

        amp = chirp._amp
        w_max = chirp._w_max
        w_os = chirp._w_os
        phase = chirp._phase

        phi = 2.0 * w_max / w_os * np.sin(w_os / 2.0 * t)
        dphi_dt = w_max * np.cos(w_os / 2.0 * t)

        dydt_expect = amp * dphi_dt * np.cos(phi + phase)

        np.testing.assert_allclose(dydt, dydt_expect)

    def test_d2ydt2(self, chirp, t):
        d2ydt2 = chirp.d2ydt2(t)

        amp = chirp._amp
        w_max = chirp._w_max
        w_os = chirp._w_os
        phase = chirp._phase

        phi = 2.0 * w_max / w_os * np.sin(w_os / 2.0 * t)
        dphi = w_max * np.cos(w_os / 2.0 * t)
        d2phi = -w_max * w_os / 2.0 * np.sin(w_os / 2.0 * t)
        d2ydt2_expect = -amp * (dphi**2) * np.sin(phi + phase) + amp * d2phi * np.cos(
            phi + phase
        )

        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)

    def test__call__(self, chirp, t):
        y, dydt, d2ydt2 = chirp(t)

        amp = chirp._amp
        w_max = chirp._w_max
        w_os = chirp._w_os
        phase = chirp._phase
        offset = chirp._offset

        phi = 2.0 * w_max / w_os * np.sin(w_os / 2.0 * t)
        dphi = w_max * np.cos(w_os / 2.0 * t)
        d2phi = -w_max * w_os / 2.0 * np.sin(w_os / 2.0 * t)

        y_expect = amp * np.sin(phi + phase) + offset
        dydt_expect = amp * dphi * np.cos(phi + phase)
        d2ydt2_expect = -amp * (dphi**2) * np.sin(phi + phase) + amp * d2phi * np.cos(
            phi + phase
        )

        np.testing.assert_allclose(y, y_expect)
        np.testing.assert_allclose(dydt, dydt_expect)
        np.testing.assert_allclose(d2ydt2, d2ydt2_expect)


class Test_pva_sim:
    def test_default(self):
        t, p_n, v_n, euler_nb, f_b, w_b = ap.pva_sim()

        # Expected DOF signals
        amp_att = np.radians(5.0)
        amp_pos = 1.0
        phases_att = (0.0, 1 * np.pi / 3, 2 * np.pi / 3)
        phases_pos = (3 * np.pi / 3, 4 * np.pi / 3, 5 * np.pi / 3)
        px, vx, _ = BeatDOF(amp_pos, 0.1, 0.01, freq_hz=True, phase=phases_pos[0])(t)
        py, vy, _ = BeatDOF(amp_pos, 0.1, 0.01, freq_hz=True, phase=phases_pos[1])(t)
        pz, vz, _ = BeatDOF(amp_pos, 0.1, 0.01, freq_hz=True, phase=phases_pos[2])(t)
        r, *_ = BeatDOF(amp_att, 0.1, 0.01, freq_hz=True, phase=phases_att[0])(t)
        p, *_ = BeatDOF(amp_att, 0.1, 0.01, freq_hz=True, phase=phases_att[1])(t)
        y, *_ = BeatDOF(amp_att, 0.1, 0.01, freq_hz=True, phase=phases_att[2])(t)

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

        # Validate f and w by strapdown integration using AHRS (no aiding)
        att0 = ap.Attitude.from_euler(euler_nb[0], degrees=False)
        ahrs = ap.AHRS(fs_expect, q_nb=att0, v_n=v_n[0])
        vel_est, euler_est = [v_n[0]], [euler_nb[0]]
        for f_i, w_i in zip(f_b[1:], w_b[1:]):
            ahrs.update(f_i, w_i, v_n=None)
            vel_est.append(ahrs.v_n)
            euler_est.append(ahrs.attitude.as_euler(degrees=False))
        vel_est = np.array(vel_est)
        euler_est = np.array(euler_est)
        pos_est = np.cumsum(vel_est, axis=0) / fs_expect

        np.testing.assert_allclose(pos_est[:100], p_n[:100], atol=1e-1)
        np.testing.assert_allclose(vel_est[:100], v_n[:100], atol=1e-1)
        np.testing.assert_allclose(euler_est[:100], euler_nb[:100], atol=1e-3)

    def test_fs_n(self):
        fs = 20.0
        n = 5000
        t, p_n, v_n, euler_nb, f_b, w_b = ap.pva_sim(fs=fs, n=n)

        assert t.shape == (n,)
        assert p_n.shape == (n, 3)
        assert v_n.shape == (n, 3)
        assert euler_nb.shape == (n, 3)
        assert f_b.shape == (n, 3)
        assert w_b.shape == (n, 3)
        np.testing.assert_allclose(t[1:] - t[:-1], 1 / fs)

    def test_degrees(self):
        *_, euler_deg, _, _ = ap.pva_sim(degrees=True)
        *_, euler_rad, _, _ = ap.pva_sim(degrees=False)

        np.testing.assert_allclose(euler_deg, np.degrees(euler_rad))

    # def test_nav_frame(self):

    #     # NED
    #     *_, f_ned, _ = ap.pva_sim(nav_frame="NED", type_="standstill")
    #     f_expect = np.full(f_ned.shape, np.array([0.0, 0.0, -9.80665]))
    #     np.testing.assert_allclose(f_ned, f_expect)

    #     # ENU
    #     *_, f_enu, _ = ap.pva_sim(nav_frame="ENU", type_="standstill")
    #     f_expect = np.full(f_enu.shape, np.array([0.0, 0.0, 9.80665]))
    #     np.testing.assert_allclose(f_enu, f_expect)

    #     with pytest.raises(ValueError):
    #         ap.pva_sim(type_="invalid")

    # def test_g(self):
    #     g = 9.81
    #     *_, f, _ = ap.pva_sim(g=g, type_="standstill")
    #     f_expect = np.full(f.shape, np.array([0.0, 0.0, -g]))
    #     np.testing.assert_allclose(f, f_expect)
