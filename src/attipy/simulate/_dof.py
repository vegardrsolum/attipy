import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

class DOF(ABC):
    """
    Abstract base class for degree of freedom (DOF) signal generators.

    A DOF signal generator produces a signal, y(t), and its two first time
    derivatives, dy(t)/dt and d2y(t)/dt2.

    Subclasses must implement ``_evaluate(t)``, which receives the time vector,
    ``t``, as an ndarray of shape (n,), and returns the tuple ``(y, dydt, d2ydt2)``,
    with each element an ndarray of shape (n,).
    """

    @abstractmethod
    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Signal, y(t), and its two first time derivatives, dy(t)/dt and d2y(t)/dt2.
        """
        ...

    def _y(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        y, _, _ = self._evaluate(t)
        return y

    def _dydt(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        _, dydt, _ = self._evaluate(t)
        return dydt

    def _d2ydt2(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        _, _, d2ydt2 = self._evaluate(t)
        return d2ydt2

    def y(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates y(t) signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.

        Returns
        -------
        ndarray, shape (n,)
            DOF signal y(t).
        """
        t = np.asarray_chkfinite(t)
        return self._y(t)

    def dydt(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates dy(t)/dt signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.

        Returns
        -------
        ndarray, shape (n,)
            Time derivative, dy(t)/dt, of DOF signal.
        """
        t = np.asarray_chkfinite(t)
        return self._dydt(t)

    def d2ydt2(self, t: ArrayLike) -> NDArray[np.float64]:
        """
        Generates d2y(t)/dt2 signal.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.

        Returns
        -------
        ndarray, shape (n,)
            Second time derivative, d2y(t)/dt2, of DOF signal.
        """
        t = np.asarray_chkfinite(t)
        return self._d2ydt2(t)

    def __call__(
        self, t: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Generates y(t), dy(t)/dt, and d2y(t)/dt2 signals.

        Parameters
        ----------
        t : array_like, shape (n,)
            Time vector in seconds.

        Returns
        -------
        y : ndarray, shape (n,)
            DOF signal y(t).
        dydt : ndarray, shape (n,)
            Time derivative, dy(t)/dt, of DOF signal.
        d2ydt2 : ndarray, shape (n,)
            Second time derivative, d2y(t)/dt2, of DOF signal.
        """
        t = np.asarray_chkfinite(t)
        return self._evaluate(t)


class Beat(DOF):
    """
    Beating DOF signal generator.

    Defined as:

        y = amp * sin(w_beat / 2.0 * t) * cos(w * t + phase)

    Parameters
    ----------
    amp : float, optional
        Amplitude of the beat signal. Defaults to 1.0.
    omega : float, optional
        Main angular frequency, w, of the sinusoidal signal, y(t), in rad/s.
        Defaults to 0.1 rad/s.
    omega_beat : float, optional
        Beating angular frequency, w_beat, controlling the variation in amplitude,
        in rad/s. Defaults to 0.01 rad/s.
    phase : float, optional
        Phase offset of the beat signal in radians. Defaults to 0.0 radians.
    """

    def __init__(
        self,
        *,
        amp: float = 1.0,
        omega: float = 0.1,
        omega_beat: float = 0.01,
        phase: float = 0.0,
    ) -> None:
        self._amp = amp
        self._w_main = omega
        self._w_beat = omega_beat
        self._phase = phase

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        amp = self._amp
        w_main = self._w_main
        w_beat = self._w_beat
        phase = self._phase

        arg_main = w_main * t + phase
        arg_beat = w_beat / 2.0 * t

        main = np.cos(arg_main)
        dmain = -w_main * np.sin(arg_main)
        d2main = -(w_main**2) * main

        beat = np.sin(arg_beat)
        dbeat = w_beat / 2.0 * np.cos(arg_beat)
        d2beat = -((w_beat / 2.0) ** 2) * beat

        y = amp * beat * main
        dydt = amp * (dbeat * main + beat * dmain)
        d2ydt2 = amp * (d2beat * main + 2.0 * dbeat * dmain + beat * d2main)

        return y, dydt, d2ydt2


class Constant(DOF):
    """
    Constant DOF signal generator.

    Defined as:

        y = value

    Parameters
    ----------
    value : float, optional
        Constant value of the signal, y(t). Defaults to 0.0.
    """

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        y = np.full_like(t, self._value, dtype=np.float64)
        dydt = np.zeros_like(y)
        d2ydt2 = np.zeros_like(y)

        return y, dydt, d2ydt2


class Sine(DOF):
    """
    Sinusoidal DOF signal generator.

    Defined as:

        y = amp * sin(w * t + phase)

    Parameters
    ----------
    amp : float, optional
        Amplitude of the sinusoidal signal. Defaults to 1.0.
    omega : float, optional
        Angular frequency, w, of the sinusoidal signal, y(t), in rad/s. Defaults
        to 1.0 rad/s.
    phase : float, optional
        Phase offset of the sinusoidal signal in radians. Defaults to 0.0 radians.
    """

    def __init__(
        self,
        *,
        amp: float = 1.0,
        omega: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        self._amp = amp
        self._w = omega
        self._phase = phase

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        amp = self._amp
        w = self._w

        arg = w * t + self._phase

        y = amp * np.sin(arg)
        dydt = amp * w * np.cos(arg)
        d2ydt2 = -(w**2) * y

        return y, dydt, d2ydt2


class SmootherStep(DOF):
    """
    Smootherstep DOF signal generator.

    Smooth step from 0 to 1, defined as:

        y = 6 * x**5 - 15 * x**4 + 10 * x**3

    where ``x = (t - start) / duration`` clipped to [0, 1]. The signal is zero
    before ``start``, increases smoothly from 0 to 1 during the step period, and
    stays at 1 afterwards. The first and second time derivatives vanish at both
    ends of the step period, so that the signal and its two first time
    derivatives are continuous.

    Parameters
    ----------
    duration : float, optional
        Duration of the step period in seconds. Must be positive. Defaults to
        100.0 seconds.
    start : float, optional
        Time in seconds at which the step starts. Defaults to 0.0 seconds.
    """

    def __init__(self, *, duration: float = 100.0, start: float = 0.0) -> None:
        if duration <= 0.0:
            raise ValueError("'duration' must be positive.")

        self._duration = float(duration)
        self._start = float(start)

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        duration = self._duration
        x = np.clip((t - self._start) / duration, 0.0, 1.0)

        w = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
        dw = 30.0 * x**2 * (1.0 - x) ** 2 / duration
        d2w = 60.0 * x * (1.0 - 3.0 * x + 2.0 * x**2) / duration**2

        return w, dw, d2w


def _as_dof(other: object, /) -> DOF:
    """
    Convert an argument to a DOF signal generator.

    DOF instances are returned as they are, and real numbers are wrapped in a
    ``Constant``. Any other argument raises a ``TypeError``.
    """
    if isinstance(other, DOF):
        return other
    if isinstance(other, numbers.Real):
        return Constant(float(other))
    raise TypeError(
        f"Arguments must be DOF instances or real numbers, got {type(other).__name__}."
    )


class _Composite(DOF):
    """
    Abstract base class for composite DOFs.

    Parameters
    ----------
    *dofs : DOF
        DOF signal generators to combine.
    """

    def __init__(self, *dofs: DOF) -> None:
        if not dofs:
            raise ValueError("At least one DOF must be given.")
        self._dofs = self._flatten(dofs)

    @classmethod
    def _flatten(cls, dofs: Iterable[DOF]) -> tuple[DOF, ...]:
        dofs_flat: list[DOF] = []
        for dof in dofs:
            if isinstance(dof, cls):
                dofs_flat.extend(dof._dofs)
            else:
                dofs_flat.append(dof)
        return tuple(dofs_flat)


class _Sum(_Composite):
    """
    Sum of DOF signals.

    Defined as:

        y(t) = y_1(t) + y_2(t) + ... + y_n(t)

    Parameters
    ----------
    *dofs : DOF
        DOFs to be added.
    """

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        y, dydt, d2ydt2 = self._dofs[0]._evaluate(t)

        y = y.copy()
        dydt = dydt.copy()
        d2ydt2 = d2ydt2.copy()

        for dof in self._dofs[1:]:
            y_i, dydt_i, d2ydt2_i = dof._evaluate(t)
            y += y_i
            dydt += dydt_i
            d2ydt2 += d2ydt2_i

        return y, dydt, d2ydt2


class _Product(_Composite):
    """
    Product of DOF signals.

    Defined as:

        y(t) = y_1(t) * y_2(t) * ... * y_n(t)

    The time derivatives are found by repeated use of the product rule.

    Parameters
    ----------
    *dofs : DOF
        DOFs to be multiplied.
    """

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        y, dydt, d2ydt2 = self._dofs[0]._evaluate(t)

        for dof in self._dofs[1:]:
            y_i, dydt_i, d2ydt2_i = dof._evaluate(t)
            d2ydt2 = d2ydt2 * y_i + 2.0 * dydt * dydt_i + y * d2ydt2_i
            dydt = dydt * y_i + y * dydt_i
            y = y * y_i

        return y, dydt, d2ydt2


def add(dof1: DOF | float, dof2: DOF | float, /) -> DOF:
    """
    Add two DOF signals together.

    Defined as:

        y(t) = y_1(t) + y_2(t)

    Parameters
    ----------
    dof1, dof2 : DOF or float
        DOFs to be added. Real numbers are treated as constant DOF signals.

    Returns
    -------
    DOF
        DOF signal generator for the sum.
    """
    return _Sum(_as_dof(dof1), _as_dof(dof2))


def multiply(dof1: DOF | float, dof2: DOF | float, /) -> DOF:
    """
    Multiply two DOF signals together.

    Defined as:

        y(t) = y_1(t) * y_2(t)

    The time derivatives are found by the product rule.

    Parameters
    ----------
    dof1, dof2 : DOF or float
        DOFs to be multiplied. Real numbers are treated as constant DOF signals.

    Returns
    -------
    DOF
        DOF signal generator for the product.
    """
    return _Product(_as_dof(dof1), _as_dof(dof2))


def ramp_up(dof: DOF | float, /, *, duration: float = 100.0, start: float = 0.0) -> DOF:
    """
    Ramp up a DOF signal.

    Scales an underlying DOF signal, y(t), by a smooth ramp-up window, w(t):

        y_rampup(t) = w(t) * y(t)

    The window is zero before the ramp-up starts, increases smoothly from 0 to 1
    during the ramp-up period, and stays at 1 afterwards:

        w(t) = 6 * x**5 - 15 * x**4 + 10 * x**3

    where ``x = (t - start) / duration`` clipped to [0, 1]. The window has
    vanishing first and second derivatives at both ends of the ramp-up period,
    so that the ramped signal and its two first time derivatives are continuous.

    This is equivalent to ``multiply(SmootherStep(duration=duration, start=start), dof)``.

    Parameters
    ----------
    dof : DOF or float
        Underlying DOF signal generator to ramp up. A real number is treated as
        a constant DOF signal.
    duration : float, optional
        Duration of the ramp-up period in seconds. Must be positive. Defaults to
        100.0 seconds.
    start : float, optional
        Time in seconds at which the ramp-up starts. The signal, and its two
        first time derivatives, are zero before this time. Defaults to 0.0.

    Returns
    -------
    DOF
        DOF signal generator for the ramped-up signal.
    """
    return multiply(SmootherStep(duration=duration, start=start), dof)


def from_psd(
    f: ArrayLike,
    psd: ArrayLike,
    /,
    nbins: int,
    *,
    jitter: float = 0.0,
    seed: int | None = None,
) -> DOF:
    """
    DOF signal realization of a one-sided power spectral density (PSD).

    Parameters
    ----------
    f : array_like, shape (m,)
        Frequencies in Hz. Must be non-negative and strictly increasing, with at
        least two values.
    psd : array_like, shape (m,)
        One-sided power spectral density in y**2 / Hz. Must be non-negative.
    nbins : int
        Number of frequency bins, and thereby sinusoidal components. Must be
        positive.
    jitter : float, optional
        Random offset of each component frequency from its bin center, as a
        fraction of the bin width. Must be in the range [0, 1], where 0.0 places
        the component at the bin center and 1.0 anywhere within the bin. Jitter
        breaks up the periodicity of evenly spaced components. Defaults to 0.0.
    seed : int, optional
        Seed used to generate random phases and jitter. Defaults to None; fresh
        unpredictable entropy will be pulled from the OS.

    Returns
    -------
    DOF
        DOF signal generator for the realization of the PSD.
    """
    f = np.asarray_chkfinite(f, dtype=np.float64)
    psd = np.asarray_chkfinite(psd, dtype=np.float64)

    if f.ndim != 1 or f.size < 2:
        raise ValueError("'f' must be a 1D array with at least two values.")
    if psd.shape != f.shape:
        raise ValueError("'psd' must have the same shape as 'f'.")
    if f[0] < 0.0 or np.any(np.diff(f) <= 0.0):
        raise ValueError("'f' must be non-negative and strictly increasing.")
    if np.any(psd < 0.0):
        raise ValueError("'psd' must be non-negative.")
    if not isinstance(nbins, numbers.Integral) or nbins < 1:
        raise ValueError("'nbins' must be a positive integer.")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError("'jitter' must be in [0, 1].")

    rng = np.random.default_rng(seed)
    phase_k = rng.uniform(0.0, 2.0 * np.pi, nbins)
    offset_k = rng.uniform(0.5 - 0.5 * jitter, 0.5 + 0.5 * jitter, nbins)

    edges = np.linspace(f[0], f[-1], nbins + 1)
    df = edges[1] - edges[0]
    freq_k = edges[:-1] + df * offset_k

    area_k = _bin_integrals(f, psd, edges)
    amp_k = np.sqrt(2.0 * area_k)

    return _sum_of_sines(amp_k, freq_k, phase_k)


def from_csd(
    f: ArrayLike,
    csd: ArrayLike,
    /,
    nbins: int,
    *,
    jitter: float = 0.0,
    seed: int | None = None,
) -> tuple[DOF, ...]:
    """
    Correlated DOF signal realizations of a cross-spectral density (CSD) matrix.

    Adapted from the spectral representation method of Deodatis (1996).

    Parameters
    ----------
    f : array_like, shape (m,)
        Frequencies in Hz, non-negative and strictly increasing, with m >= 2.
    csd : array_like, shape (m, n, n)
        One-sided cross-spectral density (CSD) matrix in y_i * y_j / Hz, where
        ``csd[:, i, j]`` matches ``scipy.signal.csd(y_i, y_j)``. Must be
        Hermitian and positive semidefinite at each frequency.
    nbins : int
        Number of bins to divide the frequency range into. The auto- and cross-
        spectra of the output DOF signals will match ``csd`` at a resolution of
        one bin width. Each bin is split into n sub-bins with one sinusoid each,
        so each output signal is a sum of ``nbins * n`` sinusoids.
    jitter : float, optional
        Random frequency offset of each component within its sub-bin. Must be in
        the range [0, 1], where 0.0 places components at sub-bin centers and 1.0
        anywhere within their sub-bins. Jitter breaks up periodicity. Defaults to
        0.0.
    seed : int, optional
        Seed used to generate random phases and jitter. Defaults to None; fresh
        unpredictable entropy will be pulled from the OS.

    Returns
    -------
    tuple of DOF, length n
        DOF signal generators.

    References
    ----------
    Deodatis, G. (1996). Simulation of ergodic multivariate stochastic processes.
    Journal of Engineering Mechanics.
    """
    f = np.asarray_chkfinite(f, dtype=np.float64)
    csd = np.asarray_chkfinite(csd, dtype=np.complex128)

    if f.ndim != 1 or f.size < 2:
        raise ValueError("'f' must be a 1D array with at least two values.")
    if f[0] < 0.0 or np.any(np.diff(f) <= 0.0):
        raise ValueError("'f' must be non-negative and strictly increasing.")
    if csd.ndim != 3 or csd.shape[0] != f.size or csd.shape[1] != csd.shape[2]:
        raise ValueError("'csd' must have shape (m, n, n), where m is the size of 'f'.")
    tol = 1e-10 * np.abs(csd).max()
    if not np.allclose(csd, csd.conj().swapaxes(1, 2), rtol=0.0, atol=tol):
        raise ValueError("'csd' must be Hermitian.")
    if np.any(np.linalg.eigvalsh(csd) < -tol):
        raise ValueError("'csd' must be positive semidefinite.")
    if not isinstance(nbins, numbers.Integral) or nbins < 1:
        raise ValueError("'nbins' must be a positive integer.")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError("'jitter' must be in [0, 1].")

    n = csd.shape[1]
    rng = np.random.default_rng(seed)
    theta_kj = rng.uniform(0.0, 2.0 * np.pi, (nbins, n))
    offset_kj = rng.uniform(0.5 - 0.5 * jitter, 0.5 + 0.5 * jitter, (nbins, n))

    # Sub-bin j of bin k holds the component of source j
    sub_edges = np.linspace(f[0], f[-1], nbins * n + 1)
    d_sub = sub_edges[1] - sub_edges[0]
    freq_kj = sub_edges[:-1].reshape(nbins, n) + d_sub * offset_kj

    # S_k[i, j] weights the sinusoid of source j in signal i
    s_kij = _hermitian_sqrt(_bin_integrals(f, csd, sub_edges[::n]))
    amp_kij = np.sqrt(2.0) * np.abs(s_kij)
    phase_kij = theta_kj[:, np.newaxis, :] - np.angle(s_kij)

    # Signal i is the sum of sinusoids over sub-bins j and frequency bins k
    dofs = []
    freq_kj_flat = freq_kj.ravel()
    for i in range(n):
        amp_kj_flat = amp_kij[:, i, :].ravel()
        phase_kj_flat = phase_kij[:, i, :].ravel()
        dofs.append(_sum_of_sines(amp_kj_flat, freq_kj_flat, phase_kj_flat))

    return tuple(dofs)


def _sum_of_sines(
    amps: NDArray[np.float64],
    freqs: NDArray[np.float64],
    phases: NDArray[np.float64],
) -> DOF:
    """
    Build a DOF signal as a sum of sinusoids:

        sum_k amps[k] * sin(2 * pi * freqs[k] * t + phases[k])

    Parameters
    ----------
    amps : NDArray[np.float64]
        Amplitudes, shape (k,). Components with zero amplitude are dropped.
    freqs : NDArray[np.float64]
        Frequencies in Hz, shape (k,).
    phases : NDArray[np.float64]
        Phases in radians, shape (k,).

    Returns
    -------
    DOF
        Sum of the sinusoids, or a zero ``Constant`` if no components remain.
    """
    sines = [
        Sine(amp=amp, omega=2.0 * np.pi * freq, phase=phase)
        for amp, freq, phase in zip(amps, freqs, phases)
        if amp > 0.0
    ]
    if not sines:
        return Constant(0.0)
    return _Sum(*sines)


def _hermitian_sqrt(a: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Compute the Hermitian square roots of Hermitian positive semidefinite
    matrices.

    Parameters
    ----------
    a : NDArray[np.complex128]
        Hermitian positive semidefinite matrices, shape (..., n, n). Negative
        eigenvalues from round-off are clipped to zero.

    Returns
    -------
    NDArray[np.complex128]
        Hermitian square roots, s, such that ``s @ s = a``, shape (..., n, n).
    """
    w, v = np.linalg.eigh(a)
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    s = (v * sqrt_w[..., np.newaxis, :]) @ v.conj().swapaxes(-1, -2)
    return s  # type: ignore[no-any-return]


def _bin_integrals(
    freq: NDArray[np.float64],
    spectrum: NDArray[np.complex128],
    edges: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Compute the exact integrals of a linearly interpolated spectrum over specified
    frequency bins.

    Parameters
    ----------
    freq : NDArray[np.float64]
        Frequencies at which the spectrum is defined, shape (m,).
    spectrum : NDArray[np.complex128]
        Spectrum values at the given frequencies, shape (m, ...).
    edges : NDArray[np.float64]
        Frequency bin edges, shape (nbins + 1,). Must be strictly increasing,
        with ``edges[0] >= freq[0]`` and ``edges[-1] == freq[-1]``.

    Returns
    -------
    NDArray[np.complex128]
        Exact integrals of the spectrum over the specified frequency bins, shape
        (nbins, ...).
    """
    grid = np.union1d(freq, edges)
    shape = (-1,) + (1,) * (spectrum.ndim - 1)

    # Linear interpolation onto the grid, which contains every knot and edge
    j = np.clip(np.searchsorted(freq, grid, side="right") - 1, 0, freq.size - 2)
    w = ((grid - freq[j]) / (freq[j + 1] - freq[j])).reshape(shape)
    s = (1.0 - w) * spectrum[j] + w * spectrum[j + 1]

    # Trapezoids are exact for a piecewise-linear function; sum them per bin
    trap = 0.5 * (s[:-1] + s[1:]) * np.diff(grid).reshape(shape)
    return np.add.reduceat(trap, np.searchsorted(grid, edges[:-1]), axis=0)
