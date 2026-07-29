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
2. The band may tighten without limit, but may only widen by a capped fraction of
   its original width.
3. The pool is a bounded rolling window, so it tracks the recent past rather than
   averaging over all history.

Reference outputs are never touched by any of this. Only the variance estimate
grows; the baseline the user captured stays frozen until they explicitly replace
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Level, ProbeVerdict
from .variance import robust_centre_scale


@dataclass(frozen=True)
class PoolConfig:
    #: A run only contributes if every signal sat below this many normal-variance
    #: units. Deliberately well under `warn_k`.
    clean_z: float = 1.0
    #: Hard cap on pool size, oldest evicted first.
    max_values: int = 400
    #: The band may never grow beyond this multiple of its original width.
    max_widen: float = 1.5


def is_clean(verdict: ProbeVerdict, cfg: PoolConfig | None = None) -> bool:
    """Did this run pass cleanly enough to be evidence of normal behaviour?"""
    cfg = cfg or PoolConfig()
    if verdict.level is not Level.PASS:
        return False
    return all(sv.z is None or abs(sv.z) <= cfg.clean_z for sv in verdict.signals)


def merge_pool(
    existing: Sequence[float],
    new_values: Sequence[float],
    original_scale: float,
    cfg: PoolConfig | None = None,
) -> list[float]:
    """Fold new observations into the pool, refusing changes that widen it too far.

    `original_scale` is the MAD-derived scale of the very first baseline. It is the
    anchor that stops slow widening: no amount of pooling can loosen the band past
    `max_widen` times what the probe looked like on day one.
    """
    cfg = cfg or PoolConfig()
    if not new_values:
        return list(existing)

    candidate = list(existing) + list(new_values)
    if len(candidate) > cfg.max_values:
        candidate = candidate[-cfg.max_values :]

    if original_scale > 0:
        _, new_scale = robust_centre_scale(candidate)
        if new_scale > original_scale * cfg.max_widen:
            # Widening past the cap. Keep the tighter existing estimate: a band that
            # refuses to loosen produces false positives, which are noticed and
            # fixed. One that loosens silently produces false negatives, which are
            # not noticed at all -- and that is the whole failure this tool exists
            # to prevent.
            return list(existing)
    return candidate


def original_scale_of(values: Sequence[float]) -> float:
    """The anchor scale to persist alongside a freshly captured baseline."""
    _, scale = robust_centre_scale(values)
    return scale


def pool_from_run(
    signals_to_values: dict[str, list[float]],
    existing: dict[str, list[float]],
    anchors: dict[str, float],
    cfg: PoolConfig | None = None,
) -> dict[str, list[float]]:
    """Apply `merge_pool` across every signal of a probe."""
    out = dict(existing)
    for name, values in signals_to_values.items():
        out[name] = merge_pool(existing.get(name, []), values, anchors.get(name, 0.0), cfg)
    return out
