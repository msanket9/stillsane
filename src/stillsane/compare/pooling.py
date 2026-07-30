"""Growing the variance estimate from clean runs, without going blind.

Every check run produces fresh evidence about how much a probe naturally varies.
Folding that evidence back into the band makes it tighter over time, so the tool
gets *more* sensitive the longer it runs -- at no extra cost, since the samples
were already paid for.

The obvious way to do that is also a trap. If every run's samples widen the band,
then drift that arrives gradually gets absorbed: each day looks normal relative to
yesterday, the band creeps outward, and after a month the probe has drifted badly
while never once alerting. Boiling frog. That is precisely the failure mode the
tool exists to prevent, so the pooling rules are asymmetric on purpose:

1. Only runs that were clean -- passed, and comfortably inside the band -- pool at
   all. A run that merely scraped past the threshold contributes nothing.
2. The band may tighten without limit, but may neither widen nor slide past a cap
   measured against the *original* baseline, never against the previous run.
3. The pool is a bounded rolling window, so it tracks the recent past rather than
   averaging over all history.
4. Only pairwise signals pool, and only from distances measured *within* a single
   run. See `within_run_evidence` -- this is the subtlest rule of the four and the
   one that decides whether any of the others hold.

Reference outputs are never touched by any of this. Only the variance estimate
grows; the baseline the user captured stays frozen until they explicitly replace
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Level, ProbeVerdict, Sample
from ..signals.base import PairwiseSignal, Signal
from .variance import EPS, pairwise_within, robust_centre_scale


@dataclass(frozen=True)
class PoolConfig:
    #: A run only contributes if every signal sat below this many normal-variance
    #: units. Deliberately well under `warn_k`.
    clean_z: float = 1.0
    #: Hard cap on pool size, oldest evicted first.
    max_values: int = 400
    #: The band may never grow beyond this multiple of its original width.
    max_widen: float = 1.5
    #: Nor may its centre slide more than this many original scales from where it
    #: started. Without this the scale cap alone leaves a ratchet.
    max_centre_shift: float = 1.0


@dataclass(frozen=True)
class Anchor:
    """What the probe looked like on day one. Pooling is always measured against
    this rather than against the previous run, so small steps cannot compound."""

    center: float
    scale: float


def anchor_of(values: Sequence[float]) -> Anchor:
    """The anchor to persist alongside a freshly captured baseline."""
    centre, scale = robust_centre_scale(values)
    return Anchor(center=centre, scale=scale)


def is_clean(verdict: ProbeVerdict, cfg: PoolConfig | None = None) -> bool:
    """Did this run pass cleanly enough to be evidence of normal behaviour?"""
    cfg = cfg or PoolConfig()
    if verdict.level is not Level.PASS:
        return False
    return all(sv.z is None or abs(sv.z) <= cfg.clean_z for sv in verdict.signals)


def merge_pool(
    existing: Sequence[float],
    new_values: Sequence[float],
    anchor: Anchor,
    cfg: PoolConfig | None = None,
) -> list[float]:
    """Fold new observations into the pool, refusing changes that drift the band.

    `anchor` describes the very first baseline. Both of its numbers matter, and a
    cap on only one of them leaves a hole:

    * The **scale** cap stops the band widening until everything fits inside it.
    * The **centre** cap stops the band sliding. Without it there is a ratchet: each
      run is judged against the current centre, so a centre that creeps a little
      every run keeps every run looking clean while the band walks away from where
      it started. Thirty small legitimate-looking steps and the probe has drifted
      badly without one alert.
    """
    cfg = cfg or PoolConfig()
    if not new_values:
        return list(existing)

    candidate = list(existing) + list(new_values)
    if len(candidate) > cfg.max_values:
        candidate = candidate[-cfg.max_values :]

    new_centre, new_scale = robust_centre_scale(candidate)

    # Refusing an update keeps the tighter, older estimate. That biases towards
    # false positives, which get noticed and fixed, over false negatives, which do
    # not get noticed at all -- and a drift monitor going quietly blind is the one
    # failure this whole tool exists to prevent.
    if anchor.scale > 0 and new_scale > anchor.scale * cfg.max_widen:
        return list(existing)
    if new_centre > anchor.center + cfg.max_centre_shift * max(anchor.scale, EPS):
        return list(existing)
    return candidate


def within_run_evidence(
    signals: Sequence[Signal], current: Sequence[Sample]
) -> dict[str, list[float]]:
    """Fresh evidence of intrinsic variance, taken from within a single run.

    Which distances get pooled decides whether pooling is safe at all, and the
    obvious choice is the wrong one.

    The tempting option is the *cross* distances -- baseline against now -- because
    that is the quantity the band is actually compared against. But cross distance
    is the drift signal itself. Feeding it back into the reference distribution
    means a drifting probe teaches the band to expect its own drift, which is
    exactly backwards.

    Distances *within* the current run measure something different and much safer:
    how much this probe varies against itself, right now. That is invariant to the
    model shifting wholesale -- a model that has drifted somewhere else is still
    just as self-consistent once it gets there, so within-run distance stays put
    while cross distance climbs. It is the one quantity that is genuinely fresh
    evidence about variance while carrying no information about drift.

    Pointwise signals are excluded entirely and deliberately. Their band sits on
    the value itself (token count, latency), so pooling current values would be
    absorbing the very movement the signal exists to detect. Their bands stay
    anchored to the baseline for good.
    """
    out: dict[str, list[float]] = {}
    if len(current) < 2:
        return out  # Need at least one pair for a within-run distance.
    for signal in signals:
        if not isinstance(signal, PairwiseSignal):
            continue
        distances = pairwise_within(current, signal)
        if distances:
            out[signal.name] = distances
    return out


def pool_from_run(
    signals_to_values: dict[str, list[float]],
    existing: dict[str, list[float]],
    anchors: dict[str, Anchor],
    cfg: PoolConfig | None = None,
) -> dict[str, list[float]]:
    """Apply `merge_pool` across every signal of a probe."""
    out = dict(existing)
    for name, values in signals_to_values.items():
        anchor = anchors.get(name) or anchor_of(existing.get(name, []) or values)
        out[name] = merge_pool(existing.get(name, []), values, anchor, cfg)
    return out
