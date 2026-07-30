"""Variance bands: deciding what counts as normal for a probe.

This is the load-bearing idea in stillsane. A naive drift tool compares the
current output to a single stored output and alerts past a fixed threshold. That
fails immediately in practice, because probes do not share a variance: a
temperature-0 JSON extraction is near-deterministic while a summarisation probe
legitimately rewords itself every single call. One fixed threshold either misses
real drift on the first or fires constantly on the second, and a tool that fires
constantly gets uninstalled within the week.

So the band is learned per probe, from the probe's own behaviour:

* At baseline, capture N samples and measure the distances *among them*. That
  distribution -- call it W -- is the probe's intrinsic variance.
* At check time, capture M samples and measure the distances from each baseline
  sample to each current one. Call that C.
* Under no drift, W and C are two samples from the same distribution, because
  both are "distance between two independent draws". Under drift, C shifts up.

The headline number is not a p-value. It is `z`: how far C's centre moved in
units of the probe's own normal variation. "Moved 6.2x further than this probe
normally varies" is a sentence a developer can act on at 3am.

Robust statistics throughout -- median and MAD rather than mean and stdev --
because at the sample counts anyone will pay for (N=5), one weird sample
dominates a stdev and does not budge a MAD.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..models import Band, Direction, Level, Sample, SignalKind, SignalVerdict
from ..signals.base import CategoricalSignal, PairwiseSignal, PointwiseSignal, Signal

EPS = 1e-12

#: Scales a MAD to be comparable to a standard deviation for normally distributed
#: data, so `k` keeps its familiar "number of sigmas" meaning.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class BandConfig:
    """Thresholds, in units of the probe's own learned variance."""

    warn_k: float = 3.0
    drift_k: float = 6.0
    #: Minimum baseline samples before a band is trusted enough to fail a build.
    #: Below this the verdict is capped at WARN and says so.
    min_confident_n: int = 4
    #: Corroboration path. A MAD-derived band is deliberately conservative, and at
    #: the sample counts involved it is conservative enough to miss a genuine shift
    #: that the rank test sees clearly. When the effect size is already elevated
    #: *and* the two distributions separate this cleanly, say so -- but only ever as
    #: a WARN, never a build failure, since the whole point of the band is that it,
    #: not a p-value, decides what breaks CI.
    corroborating_p: float = 0.01
    #: Fraction of `warn_k` above which the effect size counts as "elevated" for the
    #: corroboration path.
    grey_zone: float = 0.6


def robust_centre_scale(values: Sequence[float]) -> tuple[float, float]:
    """Median and MAD-derived scale. Returns (0, 0) for an empty input."""
    if not len(values):
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    centre = float(np.median(arr))
    mad = float(np.median(np.abs(arr - centre)))
    return centre, MAD_TO_SIGMA * mad


def robust_band(
    values: Sequence[float],
    *,
    direction: Direction,
    cfg: BandConfig,
    floor: float = 0.0,
    rel_floor: float = 0.0,
    override: float | None = None,
) -> Band:
    """Build the normal range for a signal from its baseline values.

    `Band.scale` is the *effective* scale: band half-width divided by `warn_k`.
    Storing it that way keeps `z` and the band exactly consistent -- `z > warn_k`
    if and only if the value sits outside the band -- even when a floor has
    widened the band beyond what the raw MAD would give.
    """
    centre, raw_scale = robust_centre_scale(values)
    n = len(values)

    if override is not None:
        # Explicit bound from config. Centre stays as measured so `z` still reads
        # as "distance from normal", but the gate is the user's number.
        eff_scale = max(abs(override - centre), EPS) / cfg.warn_k
        return Band(center=centre, scale=eff_scale, lower=None, upper=override, n=n, floored=False)

    raw_half = cfg.warn_k * raw_scale
    half = max(raw_half, floor, rel_floor * abs(centre))
    floored = half > raw_half + EPS

    upper = centre + half if direction in (Direction.UP_IS_BAD, Direction.BOTH) else None
    lower = centre - half if direction in (Direction.DOWN_IS_BAD, Direction.BOTH) else None
    return Band(
        center=centre,
        scale=max(half / cfg.warn_k, EPS),
        lower=lower,
        upper=upper,
        n=n,
        floored=floored,
    )


def display_z(z: float, band: Band) -> float | None:
    """Suppress the effect size when the band has no meaningful scale.

    A perfectly constant baseline with no floor -- `valid_json` passing 5/5, say --
    gives a scale of EPS, and dividing by it yields numbers like -1e12. The verdict
    derived from it is still correct (anything at all is outside a zero-width
    band), but printing that figure next to a real one would rightly destroy the
    reader's trust in both. Better to show no effect size than a fake one.
    """
    return None if band.scale <= EPS * 10 else z


def z_score(observed: float, band: Band, direction: Direction) -> float:
    """Signed for two-sided signals, one-sided otherwise. Compare `abs(z)` to `k`."""
    raw = (observed - band.center) / max(band.scale, EPS)
    if direction is Direction.UP_IS_BAD:
        return max(0.0, raw)
    if direction is Direction.DOWN_IS_BAD:
        return min(0.0, raw)
    return raw


def _rankdata(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Average ranks with tie-group sizes, so the U test can correct for ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ties: list[int] = []
    ordered = values[order]
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def mann_whitney_p(within: Sequence[float], cross: Sequence[float]) -> float | None:
    """One-sided Mann-Whitney U: is `cross` stochastically larger than `within`?

    Normal approximation with tie and continuity corrections -- no scipy, which
    would drag the install weight back up for a number that is deliberately not
    the gate. Returns None when either group is too small to say anything.

    This is reported as supporting evidence beside `z`, never as the decision.
    At the sample sizes involved the test has little power, and treating a weak
    p-value as authoritative is exactly how a monitor starts lying to you.
    """
    n1, n2 = len(within), len(cross)
    if n1 < 3 or n2 < 3:
        return None
    combined = np.concatenate([np.asarray(within, float), np.asarray(cross, float)])
    ranks, ties = _rankdata(combined)
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    tie_term = sum(t**3 - t for t in ties)
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return None
    # Small U1 means group 1 (within) ranks low, i.e. cross is larger: drift.
    z = (u1 + 0.5 - mu) / math.sqrt(var)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pairwise_within(samples: Sequence[Sample], signal: PairwiseSignal) -> list[float]:
    """Distances among baseline samples: the probe's intrinsic variance."""
    out = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            d = signal.distance(samples[i], samples[j])
            if d is not None:
                out.append(d)
    return out


def pairwise_cross(
    baseline: Sequence[Sample], current: Sequence[Sample], signal: PairwiseSignal
) -> list[float]:
    """Distances from each baseline sample to each current one."""
    out = []
    for b in baseline:
        for c in current:
            d = signal.distance(b, c)
            if d is not None:
                out.append(d)
    return out


def _levels(
    z: float,
    cfg: BandConfig,
    n: int,
    max_level: Level,
    p_value: float | None = None,
    floored: bool = False,
) -> tuple[Level, str]:
    """Map an effect size to a severity, capped by confidence and by the signal."""
    magnitude = abs(z)
    if magnitude > cfg.drift_k:
        level, note = Level.DRIFT, ""
    elif magnitude > cfg.warn_k:
        level, note = Level.WARN, ""
    elif (
        p_value is not None
        and not floored
        and p_value < cfg.corroborating_p
        and magnitude > cfg.warn_k * cfg.grey_zone
    ):
        # Corroboration is only meaningful against a band the probe actually
        # earned. On a floored band the observed variance was too small to measure,
        # so the rank test is busy proving that a difference we already declared to
        # be noise is nonetheless consistent -- which it always is. A
        # byte-identical baseline plus one trailing newline separates at p=4e-08.
        capped = Level.WARN if max_level.rank >= Level.WARN.rank else max_level
        return capped, f" (inside band, but distributions separate at p={p_value:.2g})"
    else:
        return Level.PASS, ""

    if level is Level.DRIFT and n < cfg.min_confident_n:
        # Not enough baseline samples for the band to be trustworthy. Say so rather
        # than failing someone's build on two data points.
        return Level.WARN, f" (band from only {n} samples, capped to warn)"
    if level.rank > max_level.rank:
        return max_level, ""
    return level, note


def evaluate_pairwise(
    signal: PairwiseSignal,
    baseline: Sequence[Sample],
    current: Sequence[Sample],
    cfg: BandConfig,
    pooled_within: Sequence[float] | None = None,
) -> SignalVerdict | None:
    """Compare the cross-distance distribution against the within-distance one."""
    within = list(pooled_within) if pooled_within else pairwise_within(baseline, signal)
    cross = pairwise_cross(baseline, current, signal)
    if not cross:
        return None  # Signal does not apply to this probe.
    if len(within) < 1:
        return SignalVerdict(
            signal=signal.name,
            kind=SignalKind.PAIRWISE,
            level=Level.PASS,
            detail="skipped: baseline has too few samples to learn a variance band",
        )

    band = robust_band(
        within,
        direction=Direction.UP_IS_BAD,
        cfg=cfg,
        floor=signal.floor,
        override=signal.band_override,
    )

    # Rescue an under-sampled baseline.
    #
    # A probe that varies only a little can easily return N identical samples at
    # baseline -- five draws from three near-identical phrasings hit this about one
    # time in eighty. The band then has zero width, and the very next check reports
    # ordinary sampling variance as drift. That is the false positive that gets a
    # monitor uninstalled in week one, and no amount of tuning the floor fixes it,
    # because the floor cannot know how much this particular probe varies.
    #
    # The current run does know. Distances *within* the current samples measure
    # intrinsic variance and are blind to wholesale drift -- a model that moved
    # somewhere else is still just as self-consistent once it gets there. So if the
    # baseline learned nothing and the current run demonstrates real internal
    # variance, the current run is better evidence and the band widens to admit it.
    #
    # This cannot mask a genuine shift: a drifted probe whose new behaviour is
    # internally consistent still yields near-zero within-run distances, leaves the
    # band floored, and fails. And where the new behaviour genuinely is that noisy,
    # a cross distance of the same order honestly is indistinguishable from noise.
    # Two conditions gate this, and both are load-bearing.
    #
    # The baseline must look *genuinely* deterministic -- every distance inside the
    # floor -- across enough pairs to mean it. A two-sample baseline yields a single
    # distance whose MAD is zero by arithmetic rather than by evidence, and treating
    # that as "deterministic" would hand a free pass to any probe with a thin
    # baseline, which is exactly backwards.
    rescue_note = ""
    deterministic_baseline = (
        len(within) >= 3 and max(within) <= max(signal.floor, EPS)
    )
    if band.floored and deterministic_baseline and signal.band_override is None:
        # Built from the current run's spread *alone*. Pooling it with the baseline
        # would not work: a dozen zeros swamp three real values and the median stays
        # at zero, so the band never actually widens. The premise of the rescue is
        # that the baseline's estimate is the unreliable one.
        within_now = pairwise_within(current, signal) if len(current) >= 2 else []
        if within_now:
            widened = robust_band(
                within_now, direction=Direction.UP_IS_BAD, cfg=cfg, floor=signal.floor
            )
            if widened.upper > band.upper:
                band = widened
                rescue_note = " [baseline under-sampled; widened from this run's own spread]"

    observed = float(np.median(cross))
    z = z_score(observed, band, Direction.UP_IS_BAD)
    p = mann_whitney_p(within, cross)
    level, note = _levels(
        z, cfg, len(within), signal.max_level, p_value=p, floored=band.floored
    )

    detail = f"{observed:.4g} vs normal {band.center:.4g} (band {band.describe()}){note}"
    if band.floored:
        detail += " [floored]"
    detail += rescue_note
    return SignalVerdict(
        signal=signal.name,
        kind=SignalKind.PAIRWISE,
        level=level,
        detail=detail,
        observed=observed,
        baseline=band.center,
        band=band,
        z=display_z(z, band),
        p_value=p,
    )


def evaluate_pointwise(
    signal: PointwiseSignal,
    baseline: Sequence[Sample],
    current: Sequence[Sample],
    cfg: BandConfig,
    pooled_values: Sequence[float] | None = None,
) -> SignalVerdict | None:
    """Compare current values against the baseline value distribution."""
    base_vals = (
        list(pooled_values)
        if pooled_values
        else [v for v in (signal.value(s) for s in baseline) if v is not None]
    )
    cur_vals = [v for v in (signal.value(s) for s in current) if v is not None]
    if not cur_vals or not base_vals:
        return None

    band = robust_band(
        base_vals,
        direction=signal.direction,
        cfg=cfg,
        floor=signal.floor,
        rel_floor=signal.rel_floor,
        override=signal.band_override,
    )
    observed = float(np.median(cur_vals))
    z = z_score(observed, band, signal.direction)
    level, note = _levels(
        z, cfg, len(base_vals), signal.max_level, floored=band.floored
    )

    # Hard contract: a structural check that was perfect at baseline and is no
    # longer perfect is drift regardless of what the band says. A band cannot
    # sensibly be learned from a constant, and "it used to always parse and now
    # sometimes does not" is not a statistical question.
    if (
        signal.strict_when_perfect
        and band.center >= 1.0 - EPS
        and min(cur_vals) < 1.0 - EPS
    ):
        failed = sum(1 for v in cur_vals if v < 1.0 - EPS)
        return SignalVerdict(
            signal=signal.name,
            kind=SignalKind.POINTWISE,
            level=Level.DRIFT,
            detail=f"was passing every baseline sample, now failing {failed}/{len(cur_vals)}",
            observed=observed,
            baseline=band.center,
            band=band,
            z=None,
            observed_label=signal.format(observed),
            baseline_label=signal.format(band.center),
        )

    detail = f"{signal.format(observed)} vs normal {signal.format(band.center)} (band {band.describe()}){note}"
    if band.floored:
        detail += " [floored]"
    return SignalVerdict(
        signal=signal.name,
        kind=SignalKind.POINTWISE,
        level=level,
        detail=detail,
        observed=observed,
        baseline=band.center,
        band=band,
        z=display_z(z, band),
        observed_label=signal.format(observed),
        baseline_label=signal.format(band.center),
    )


def evaluate_categorical(
    signal: CategoricalSignal,
    baseline: Sequence[Sample],
    current: Sequence[Sample],
    escalate: bool = False,
) -> SignalVerdict | None:
    """Exact comparison. No band -- a fingerprint either changed or it did not.

    This is the provider-drift canary: the backend model moving underneath a
    version string that did not move. Reported at WARN by default because a
    changed fingerprint is information rather than a fault; `escalate` turns it
    into a build failure for people who want to gate on it.
    """
    base_vals = [v for v in (signal.value(s) for s in baseline) if v is not None]
    cur_vals = [v for v in (signal.value(s) for s in current) if v is not None]
    if not base_vals or not cur_vals:
        return None  # Provider does not expose it.

    base_set, cur_set = set(base_vals), set(cur_vals)
    if base_set == cur_set:
        return SignalVerdict(
            signal=signal.name,
            kind=SignalKind.CATEGORICAL,
            level=Level.PASS,
            detail=", ".join(sorted(cur_set)),
            observed_label=", ".join(sorted(cur_set)),
            baseline_label=", ".join(sorted(base_set)),
        )

    level = Level.DRIFT if escalate else signal.max_level
    return SignalVerdict(
        signal=signal.name,
        kind=SignalKind.CATEGORICAL,
        level=level,
        detail=f"{', '.join(sorted(base_set))} -> {', '.join(sorted(cur_set))}",
        observed_label=", ".join(sorted(cur_set)),
        baseline_label=", ".join(sorted(base_set)),
    )


def evaluate(
    signal: Signal,
    baseline: Sequence[Sample],
    current: Sequence[Sample],
    cfg: BandConfig,
    pooled: Sequence[float] | None = None,
    escalate_categorical: bool = False,
) -> SignalVerdict | None:
    """Dispatch one signal to the right comparison. None means 'does not apply'."""
    if isinstance(signal, PairwiseSignal):
        return evaluate_pairwise(signal, baseline, current, cfg, pooled)
    if isinstance(signal, PointwiseSignal):
        return evaluate_pointwise(signal, baseline, current, cfg, pooled)
    if isinstance(signal, CategoricalSignal):
        return evaluate_categorical(signal, baseline, current, escalate_categorical)
    raise TypeError(f"Unknown signal type: {type(signal).__name__}")
