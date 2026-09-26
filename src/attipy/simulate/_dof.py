import numbers
from abc import ABC, abstractmethod

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

    __array_ufunc__ = None

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

    def __add__(self, other: "DOF | float") -> "DOF":
        other_dof = _as_dof(other)
        if other_dof is None:
            return NotImplemented
        return _Sum(self, other_dof)

    def __radd__(self, other: "DOF | float") -> "DOF":
        other_dof = _as_dof(other)
        if other_dof is None:
            return NotImplemented
        return _Sum(other_dof, self)

    def __mul__(self, other: "DOF | float") -> "DOF":
        other_dof = _as_dof(other)
        if other_dof is None:
            return NotImplemented
        return _Product(self, other_dof)

    def __rmul__(self, other: "DOF | float") -> "DOF":
        other_dof = _as_dof(other)
        if other_dof is None:
            return NotImplemented
        return _Product(other_dof, self)

    def __neg__(self) -> "DOF":
        return _Product(ConstantDOF(-1.0), self)

    def __sub__(self, other: "DOF | float") -> "DOF":
        other_dof = _as_dof(other)
        if other_dof is None:
            return NotImplemented
        return _Sum(self, -other_dof)

    def __rsub__(self, other: "DOF | float") -> "DOF":
        other_dof = _as_dof(other)
        if other_dof is None:
            return NotImplemented
        return _Sum(other_dof, -self)


class BeatDOF(DOF):
    """
    Beating DOF signal generator.

    Defined as:

        y = sin(w_beat / 2.0 * t) * cos(w * t + phase)

    Parameters
    ----------
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
        omega: float = 0.1,
        omega_beat: float = 0.01,
        phase: float = 0.0,
    ) -> None:
        self._w_main = omega
        self._w_beat = omega_beat
        self._phase = phase

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

        y = beat * main
        dydt = dbeat * main + beat * dmain
        d2ydt2 = d2beat * main + 2.0 * dbeat * dmain + beat * d2main

        return y, dydt, d2ydt2


class ConstantDOF(DOF):
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


class SineDOF(DOF):
    """
    Sinusoidal DOF signal generator.

    Defined as:

        y = sin(w * t + phase)

    The signal has unit amplitude. Scale it by multiplying with a constant, e.g.,
    ``2.0 * SineDOF(...)``.

    Parameters
    ----------
    omega : float, optional
        Angular frequency, w, of the sinusoidal signal, y(t), in rad/s. Defaults
        to 1.0 rad/s.
    phase : float, optional
        Phase offset of the sinusoidal signal in radians. Defaults to 0.0 radians.
    """

    def __init__(
        self,
        omega: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        self._w = omega
        self._phase = phase

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        w = self._w

        arg = w * t + self._phase

        y = np.sin(arg)
        dydt = w * np.cos(arg)
        d2ydt2 = -(w**2) * y

        return y, dydt, d2ydt2


class RampUp(DOF):
    """
    Ramp-up wrapper for DOF signals.

    Scales an underlying DOF signal, y(t), by a smooth ramp-up window, w(t):

        y_rampup(t) = w(t) * y(t)

    The window is zero before the ramp-up starts, increases smoothly from 0 to 1
    during the ramp-up period, and stays at 1 afterwards:

        w(t) = 6 * x**5 - 15 * x**4 + 10 * x**3

    where ``x = (t - start) / duration`` clipped to [0, 1]. The window has
    vanishing first and second derivatives at both ends of the ramp-up period,
    so that the ramped signal and its two first time derivatives are continuous.

    Parameters
    ----------
    dof : DOF
        Underlying DOF signal generator to ramp up.
    duration : float, optional
        Duration of the ramp-up period in seconds. Must be positive. Defaults to
        100.0 seconds.
    start : float, optional
        Time in seconds at which the ramp-up starts. The signal, and its two
        first time derivatives, are zero before this time. Default is 0.0.
    """

    def __init__(self, dof: DOF, duration: float = 100.0, start: float = 0.0) -> None:
        if duration <= 0.0:
            raise ValueError("'duration' must be positive.")
        if start < 0.0:
            raise ValueError("'start' must be non-negative.")

        self._dof = dof
        self._duration = float(duration)
        self._start = float(start)

    def _window(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Ramp-up window, w(t), and its two first time derivatives.
        """
        duration = self._duration
        x = np.clip((t - self._start) / duration, 0.0, 1.0)

        w = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
        dw = 30.0 * x**2 * (1.0 - x) ** 2 / duration
        d2w = 60.0 * x * (1.0 - 3.0 * x + 2.0 * x**2) / duration**2

        return w, dw, d2w

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        w, dw, d2w = self._window(t)
        y, dydt, d2ydt2 = self._dof._evaluate(t)

        y_ramped = w * y
        dydt_ramped = dw * y + w * dydt
        d2ydt2_ramped = d2w * y + 2.0 * dw * dydt + w * d2ydt2

        return y_ramped, dydt_ramped, d2ydt2_ramped


def _as_dof(other: object) -> "DOF | None":
    """
    Convert an operand to a DOF signal generator.

    DOF instances are returned as they are, and real numbers are wrapped in a
    ``ConstantDOF``. Any other operand gives ``None``, so that the calling
    operator can return ``NotImplemented`` for unsupported operand types.
    """
    if isinstance(other, DOF):
        return other
    if isinstance(other, numbers.Real):
        return ConstantDOF(float(other))
    return None


class _Sum(DOF):
    """
    Sum of DOF signals.

    Defined as:

        y(t) = y_1(t) + y_2(t) + ... + y_n(t)

    Parameters
    ----------
    *dofs : DOF
        DOF signal generators to add together.
    """

    def __init__(self, *dofs: DOF) -> None:
        if not dofs:
            raise ValueError("At least one DOF must be given.")

        dofs_flat: list[DOF] = []
        for dof in dofs:
            if isinstance(dof, _Sum):
                dofs_flat.extend(dof._dofs)
            elif isinstance(dof, DOF):
                dofs_flat.append(dof)
            else:
                raise TypeError(
                    f"All arguments must be DOF instances, got {type(dof).__name__}."
                )
        self._dofs = tuple(dofs_flat)

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        y = np.zeros_like(t, dtype=np.float64)
        dydt = np.zeros_like(y, dtype=np.float64)
        d2ydt2 = np.zeros_like(y, dtype=np.float64)

        for dof in self._dofs:
            y_i, dydt_i, d2ydt2_i = dof._evaluate(t)
            y += y_i
            dydt += dydt_i
            d2ydt2 += d2ydt2_i

        return y, dydt, d2ydt2


class _Product(DOF):
    """
    Product of DOF signals.

    Defined as:

        y(t) = y_1(t) * y_2(t) * ... * y_n(t)

    The time derivatives are found by repeated use of the product rule.

    Parameters
    ----------
    *dofs : DOF
        DOF signal generators to multiply together.
    """

    def __init__(self, *dofs: DOF) -> None:
        if not dofs:
            raise ValueError("At least one DOF must be given.")

        dofs_flat: list[DOF] = []
        for dof in dofs:
            if isinstance(dof, _Product):
                dofs_flat.extend(dof._dofs)
            elif isinstance(dof, DOF):
                dofs_flat.append(dof)
            else:
                raise TypeError(
                    f"All arguments must be DOF instances, got {type(dof).__name__}."
                )
        self._dofs = tuple(dofs_flat)

    def _evaluate(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        y = np.ones_like(t, dtype=np.float64)
        dydt = np.zeros_like(y, dtype=np.float64)
        d2ydt2 = np.zeros_like(y, dtype=np.float64)

        for dof in self._dofs:
            y_i, dydt_i, d2ydt2_i = dof._evaluate(t)
            d2ydt2 = d2ydt2 * y_i + 2.0 * dydt * dydt_i + y * d2ydt2_i
            dydt = dydt * y_i + y * dydt_i
            y = y * y_i

        return y, dydt, d2ydt2
