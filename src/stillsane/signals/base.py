"""Signal interfaces.

A signal reduces samples to something comparable. The three kinds exist because
each needs different maths downstream, and keeping them apart is what stops the
variance engine turning into a pile of special cases:

* pairwise    -> a distance between two samples; compared as distributions
* pointwise   -> one number per sample; compared as value distributions
* categorical -> one string per sample; no band, any change is an event
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..models import Direction, Level, Sample, SignalKind


class Signal(ABC):
    name: str
    kind: SignalKind
    #: Ceiling on how severe this signal is allowed to make a verdict. Signals that
    #: depend on where the check runs from -- latency above all -- cap at WARN, so
    #: baselining on a laptop and checking from CI cannot fail a build on its own.
    max_level: Level = Level.DRIFT
    #: Explicit upper bound from config, replacing the learned band. Escape hatch
    #: for someone who knows their probe better than the statistics do; `auto`
    #: (the default) leaves this None and lets the band be learned.
    band_override: float | None = None

    def prepare(self, samples: Sequence[Sample]) -> None:  # noqa: B027 - opt-in hook
        """Optional batch warm-up before scoring.

        The embedding signal uses this to encode every text in one pass instead of
        one call per pair, which is the difference between O(n) and O(n^2) work.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}>"


class PairwiseSignal(Signal):
    kind = SignalKind.PAIRWISE
    #: Absolute floor on the band's half-width. Without this a probe whose baseline
    #: samples are byte-identical gets a zero-width band and alerts on whitespace.
    floor: float = 0.0

    @abstractmethod
    def distance(self, a: Sample, b: Sample) -> float | None:
        """Distance in [0, 1]. None means 'not applicable to these samples'."""


class PointwiseSignal(Signal):
    kind = SignalKind.POINTWISE
    direction: Direction = Direction.BOTH
    #: Absolute floor on the band's half-width, in the signal's own units.
    floor: float = 0.0
    #: Relative floor: moves smaller than this fraction of the baseline centre are
    #: never reported, however tight the band. Stops a 2ms latency wobble on a
    #: rock-steady endpoint reading as a regression.
    rel_floor: float = 0.0
    #: When True, a baseline that passed every sample is treated as a contract: any
    #: current failure is drift, no band consulted. You cannot learn a variance from
    #: a constant, and "it always parsed and now sometimes does not" is not a
    #: statistical question. Set on the structural checks.
    strict_when_perfect: bool = False

    @abstractmethod
    def value(self, sample: Sample) -> float | None:
        """One number per sample. None means 'not applicable to this sample'."""

    def format(self, value: float) -> str:
        return f"{value:.4g}"


class CategoricalSignal(Signal):
    kind = SignalKind.CATEGORICAL

    @abstractmethod
    def value(self, sample: Sample) -> str | None:
        """One string per sample. None means 'this target does not report it'."""
