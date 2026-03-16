import numpy as np
import pytest

import attipy as ap
from attipy._simulate import DOF, BeatDOF


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


class Test_pva_sim:
    def test_default(self):
        t, p_n, v_n, euler_nb, f_b, w_b = ap.pva_sim()

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
        att0 = ap.Attitude.from_euler(euler_nb[0], degrees=False)
        dt = 1.0 / fs_expect
        mekf = ap.MEKF(fs_expect, att0)
        euler_est = [euler_nb[0]]
        for f_i, w_i in zip(f_b[1:], w_b[1:]):
            mekf.update(f_i * dt, w_i * dt, gref=False)
            euler_est.append(mekf.attitude.as_euler(degrees=False))
        euler_est = np.array(euler_est)

        # TODO: check also pos and vel when strapdown estimator is available
        np.testing.assert_allclose(euler_est[:100], euler_nb[:100], atol=2e-3)

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

    def test_nav_frame(self):

        # NED
        *_, f_ned, _ = ap.pva_sim(nav_frame="NED")
        assert -10.0 < f_ned.mean(axis=0)[2] < -9.5

        # ENU
        *_, f_enu, _ = ap.pva_sim(nav_frame="ENU")
        assert 9.5 < f_enu.mean(axis=0)[2] < 10.0

        with pytest.raises(ValueError):
            ap.pva_sim(nav_frame="invalid")

    def test_g(self):
        g = 5.0
        *_, f, _ = ap.pva_sim(g=g)
        assert -6.0 < f.mean(axis=0)[2] < -4
