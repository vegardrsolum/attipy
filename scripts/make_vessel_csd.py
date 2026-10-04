"""
Build the CSD matrix of the MSS supply vessel (Fossen, MIT licence) in a JONSWAP
sea state with cos-2s spreading, and save it as an .npz file for attipy.
"""
import io
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import waveresponse as wr

MSS_FILE = Path(r"C:/Users/vrs/Documents/WorkroomC/GitHub/MSS/HYDRO/vessels_shipx/supply/supply.mat")
DOF_NAMES = ["surge", "sway", "heave", "roll", "pitch", "yaw"]

# Sea state, relative to the vessel (waves coming from, clockwise from the bow)
HS, TP, GAMMA = 3.5, 9.5, 2.0
SPREAD_S = 2
DIRP_DEG = 45.0  # bow-quartering seas excite all six DOFs

def mss_raos(vessel):
    rao = vessel.motionRAO
    freq = np.atleast_1d(rao.w).astype(float)
    dirs = np.atleast_1d(vessel.headings).astype(float)
    order = np.argsort(freq)
    out = []
    for i in range(6):
        amp, phase = np.asarray(rao.amp[i]), np.asarray(rao.phase[i])
        if amp.ndim == 3:
            amp, phase = amp[:, :, 0], phase[:, :, 0]
        out.append(wr.RAO.from_amp_phase(
            freq[order], dirs, amp[order], phase[order],
            phase_degrees=False, phase_leading=True, freq_hz=False, degrees=False,
            clockwise=True, waves_coming_from=False,
        ))
    return out

def wave_spectrum(f_hz):
    dirs = np.arange(0.0, 360.0, 5.0)
    _, s1d = wr.JONSWAP(f_hz, freq_hz=True)(HS, TP, gamma=GAMMA)
    return wr.WaveSpectrum.from_spectrum1d(
        f_hz, dirs, s1d, wr.CosineHalfSpreading(s=SPREAD_S, degrees=True), DIRP_DEG,
        freq_hz=True, degrees=True, clockwise=True, waves_coming_from=True,
    )

def csd_matrix(raos, wave):
    wave = wave.copy() if hasattr(wave, "copy") else wave
    wave.set_wave_convention(**raos[0].wave_convention)
    freq, dirs = wave._freq, wave._dirs
    raos = [r.reshape(freq, dirs, freq_hz=False, degrees=False) for r in raos]
    S = None
    for i in range(6):
        for j in range(6):
            f, s = wr.multiply(wr.multiply(raos[i].conjugate(), raos[j]), wave,
                               output_type="DirectionalSpectrum").spectrum1d(freq_hz=True)
            if S is None:
                S = np.empty((f.size, 6, 6), complex)
            S[:, i, j] = s.real if i == j else s
    return f, S

if __name__ == "__main__":
    vessel = sio.loadmat(MSS_FILE, squeeze_me=True, struct_as_record=False)["vessel"]
    raos = mss_raos(vessel)
    f_fine = np.arange(0.02, 0.50, 0.001)
    f, S = csd_matrix(raos, wave_spectrum(f_fine))
    np.savez(Path(__file__).with_name("mss_csd_fine.npz"), f=f, S=S)
    var = np.trapezoid(np.diagonal(S, axis1=1, axis2=2).real, f, axis=0)
    print("std:", dict(zip(DOF_NAMES, np.sqrt(var).round(4))), "roll/pitch/yaw deg:", np.degrees(np.sqrt(var[3:])).round(3))
    for k, nm in enumerate(DOF_NAMES):
        p = S[:, k, k].real
        c = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(f))]); c /= c[-1]
        print(f"{nm}: 0.1% {np.interp(0.001, c, f):.3f}  99.9% {np.interp(0.999, c, f):.3f} Hz, peak {f[p.argmax()]:.3f} Hz")
    print("max |imag diag|:", np.abs(np.diagonal(S, axis1=1, axis2=2).imag).max(), "min eig rel:", (np.linalg.eigvalsh(S).min() / np.abs(S).max()))


def save_package_csd(out_path, fmin=0.05, fmax=0.30, df=0.0025):
    """Compute the CSD on a coarse uniform grid and save it for attipy."""
    vessel = sio.loadmat(MSS_FILE, squeeze_me=True, struct_as_record=False)["vessel"]
    f_grid = np.round(np.arange(fmin, fmax + df / 2, df), 6)
    f, S = csd_matrix(mss_raos(vessel), wave_spectrum(f_grid))
    np.savez_compressed(out_path, f=f.astype(np.float64), csd=S.astype(np.complex128))
    return f, S
