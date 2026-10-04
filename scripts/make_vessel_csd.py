"""
Build the cross-spectral density (CSD) matrix used by the 'vessel-6dof' and
'vessel-3dof' motion presets, and save it as an .npz file in the attipy package.

The vessel is the supply vessel from the Marine Systems Simulator (MSS) by
T. I. Fossen (MIT licence, https://github.com/cybergalactic/MSS). The sea state is
multimodal: a wind sea and a swell from different directions, each described by
a JONSWAP spectrum with cos-2s directional spreading.

Usage:

    python scripts/make_vessel_csd.py <path to MSS repository>

Requires ``scipy`` and ``waveresponse``.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import waveresponse as wr

MSS_VESSEL = Path("HYDRO/vessels_shipx/supply/supply.mat")
OUT_FILE = Path(__file__).parents[1] / "src/attipy/simulate/_data/vessel_csd.npz"

# Sea state components, with directions relative to the vessel (waves coming
# from, clockwise from the bow). Total Hs = sqrt(3.0**2 + 1.8**2) = 3.5 m.
SEA_STATE = [
    # Wind sea from 45 degrees off the starboard bow; short-crested
    {"hs": 3.0, "tp": 8.0, "gamma": 3.3, "s": 2, "dirp": 45.0},
    # Swell from the port beam; long-crested
    {"hs": 1.8, "tp": 13.0, "gamma": 5.0, "s": 10, "dirp": 270.0},
]

# Frequency grid of the stored CSD, in Hz
FMIN, FMAX, DF = 0.04, 0.30, 0.002


def mss_raos(vessel):
    """
    Motion RAOs (surge, sway, heave, roll, pitch, yaw) of an MSS vessel at zero
    speed. MSS uses x forward, y starboard and z down, and rotations in rad/m.
    """
    rao = vessel.motionRAO
    freq = np.atleast_1d(rao.w).astype(float)  # rad/s
    dirs = np.atleast_1d(vessel.headings).astype(float)  # rad, 0 = following sea
    order = np.argsort(freq)

    raos = []
    for i in range(6):
        amp, phase = np.asarray(rao.amp[i]), np.asarray(rao.phase[i])
        if amp.ndim == 3:  # (freq, heading, speed)
            amp, phase = amp[:, :, 0], phase[:, :, 0]
        raos.append(
            wr.RAO.from_amp_phase(
                freq[order],
                dirs,
                amp[order],
                phase[order],
                phase_degrees=False,
                phase_leading=True,  # x(t) = |H| * a * cos(w * t + phase)
                freq_hz=False,
                degrees=False,
                clockwise=True,
                waves_coming_from=False,
            )
        )
    return raos


def wave_spectrum(f, hs, tp, gamma, s, dirp):
    """
    Directional wave spectrum from a JONSWAP spectrum with cos-2s spreading.
    """
    dirs = np.arange(0.0, 360.0, 5.0)
    _, s1d = wr.JONSWAP(f, freq_hz=True)(hs, tp, gamma=gamma)
    return wr.WaveSpectrum.from_spectrum1d(
        f,
        dirs,
        s1d,
        wr.CosineHalfSpreading(s=s, degrees=True),
        dirp,
        freq_hz=True,
        degrees=True,
        clockwise=True,
        waves_coming_from=True,
    )


def csd_matrix(raos, wave):
    """
    One-sided CSD matrix, shape (m, n, n), of the responses of n RAOs to a wave
    spectrum, with ``csd[:, i, j]`` matching ``scipy.signal.csd(y_i, y_j)``.
    """
    wave.set_wave_convention(**raos[0].wave_convention)
    freq, dirs = wave._freq, wave._dirs
    raos = [rao.reshape(freq, dirs, freq_hz=False, degrees=False) for rao in raos]

    n = len(raos)
    csd = None
    for i in range(n):
        for j in range(n):
            f, csd_ij = wr.multiply(
                wr.multiply(raos[i].conjugate(), raos[j]),
                wave,
                output_type="DirectionalSpectrum",
            ).spectrum1d(freq_hz=True)
            if csd is None:
                csd = np.empty((f.size, n, n), dtype=np.complex128)
            csd[:, i, j] = csd_ij.real if i == j else csd_ij
    return f, csd


def sea_state_csd(raos, f):
    """
    CSD matrix of the responses to the multimodal sea state. The response CSD
    is linear in the wave spectrum, so the CSDs of the components are summed.
    """
    csd = 0.0
    for component in SEA_STATE:
        f_out, csd_k = csd_matrix(raos, wave_spectrum(f, **component))
        csd = csd + csd_k
    return f_out, csd


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mss", type=Path, help="Path to the MSS repository.")
    args = parser.parse_args()

    vessel = sio.loadmat(
        args.mss / MSS_VESSEL, squeeze_me=True, struct_as_record=False
    )["vessel"]
    raos = mss_raos(vessel)

    f = np.round(np.arange(FMIN, FMAX + DF / 2, DF), 6)
    f, csd = sea_state_csd(raos, f)
    np.savez_compressed(OUT_FILE, f=f, csd=csd)

    std = np.sqrt(np.trapezoid(np.diagonal(csd, axis1=1, axis2=2).real, f, axis=0))
    print(f"Saved {OUT_FILE} ({f.size} frequencies, {f[0]:.3f}-{f[-1]:.3f} Hz)")
    print(f"Std surge, sway, heave [m]: {np.round(std[:3], 3)}")
    print(f"Std roll, pitch, yaw [deg]: {np.round(np.degrees(std[3:]), 3)}")


if __name__ == "__main__":
    main()
