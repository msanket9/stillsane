"""What the bands actually look like, and which of them cannot be trusted.

`check` answers "did this probe move?". This answers the question that comes
before it: is the band it would be judged against a measurement at all?

The distinction is not academic. A band can be perfectly well formed and still be
wrong. A probe whose baseline samples fall into two groups -- byte identical most
of the time, a different formatting the rest -- has a median pair distance of zero
and a MAD of zero, so the robust scale collapses to nothing and the band drops to
the signal's floor. What comes out looks exactly like every other band. It is not
one. Its own baseline already sits outside it, and the first check that happens to
draw the minority formatting reports drift against an endpoint that never changed.

That failure is invisible from `check`, which sees a band and trusts it, and it
was invisible from `baseline`, which only ever reported the pairwise signals. The
cheapest fix is to let people look.

Nothing here touches the network or the embedder. Every number is recomputed from
what `stillsane baseline` already wrote to disk, which is what makes this safe to
run at any time, free, and testable with no API key.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .compare.variance import EPS, BandConfig, robust_band, robust_centre_scale
from .models import Band, Direction
from .signals.base import PairwiseSignal, PointwiseSignal, Signal
from .store.baseline import Baseline


class Finding(str, Enum):
    """What is wrong with a band, if anything.

    Deliberately at most one per signal. A collapsed scale almost always drags its
    own baseline outside the band too, and reporting both would bury the useful
    line under a restatement of it.
    """

    #: Scale came out zero while the samples visibly vary. The band is a floor, not
    #: a measurement, and the probe's own baseline escapes it.
    COLLAPSED = "collapsed"
    #: The band has a real scale, but some of the baseline it was built from still
    #: sits outside it. Usually a heavy tail rather than a defect.
    SELF_OUTSIDE = "self-outside"
    #: Floored, but the samples really are that stable. Worth knowing, not a fault.
    FLOORED_STABLE = "floored-stable"


#: Findings that mean the band will misreport. `FLOORED_STABLE` is informational:
#: a genuinely deterministic probe with a floored band is working as intended.
SUSPECT = frozenset({Finding.COLLAPSED, Finding.SELF_OUTSIDE})


@dataclass(frozen=True)
class SignalBand:
    """One signal's band, the data behind it, and the verdict on whether it holds."""

    signal: str
    band: Band
    #: "pairs" for a pairwise signal, "values" for a pointwise one. The counts mean
    #: different things and the unit is the only thing that says so.
    unit: str
    n: int
    observed_min: float
    observed_max: float
    #: Raw MAD-derived scale, before any floor was applied. Zero here is the tell.
    raw_scale: float
    #: How much of the baseline the band already excludes.
    outside: int
    finding: Finding | None = None
    #: Estimated share of clean runs this band would report as drift, in percent.
    #: None when there is not enough baseline to simulate against.
    false_alarm_pct: float | None = None
    #: How many values a single check reduces to its median. Recorded because the
    #: estimate means nothing without it: the median of three values scatters far
    #: more than the median of twenty.
    draw: int = 0

    @property
    def outside_pct(self) -> float:
        return 100.0 * self.outside / self.n if self.n else 0.0


@dataclass(frozen=True)
class ProbeBands:
    probe_id: str
    target_name: str
    version: int
    created: str
    n_samples: int
    signals: list[SignalBand] = field(default_factory=list)

    @property
    def suspect(self) -> list[SignalBand]:
        return [s for s in self.signals if s.finding in SUSPECT]


class _NoEmbedder:
    """Stands in for the real embedder, which this module must never need.

    Every pairwise number reported here was computed once by `stillsane baseline`
    and written to `variance.json`. If a future change makes this path try to embed
    text, failing loudly beats silently downloading a model in a command whose
    whole appeal is that it costs nothing.
    """

    def encode(self, texts: Sequence[str]):  # pragma: no cover - guard, never called
        raise AssertionError(
            "stillsane bands must not embed text; it reads stored distances"
        )


def no_embedder() -> _NoEmbedder:
    """Embedder placeholder for building signal metadata without loading a model."""
    return _NoEmbedder()


#: Simulation draws. Enough that the estimate is stable to about a tenth of a
#: percent, cheap enough that inspecting a config full of probes stays instant.
TRIALS = 4000

#: Fixed seed. An inspection command that returns a slightly different number every
#: time it runs is one nobody can quote in a bug report or diff between releases.
SEED = 20260807


def false_alarm_rate(
    values: Sequence[float], band: Band, draw: int, trials: int = TRIALS
) -> float | None:
    """Share of clean runs this band would call drift, as a fraction.

    `bands` used to report how many individual baseline values fell outside the
    band, which is a different and more alarming question than the one that
    matters. A check never compares a single value: it reduces the run to a median
    (`evaluate_pairwise` and `evaluate_pointwise` both do) and compares that. A
    heavy tail can put 18% of individual pairs outside a band while almost never
    moving the median far enough to fire.

    So simulate the thing that actually happens. Resample from the baseline's own
    distribution, take the median of a check-sized draw, and count how often it
    lands outside. That is the false alarm rate under the null hypothesis that
    nothing has drifted.

    The assumption is that when nothing has changed, a check's distances look like
    the baseline's own. That is precisely the assumption the band already encodes,
    so this adds no new leap -- but it is an estimate from one baseline, not a
    measured rate, and small baselines will estimate it coarsely.
    """
    if draw < 1 or len(values) < 2:
        return None
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(np.asarray(values, dtype=float), size=(trials, draw), replace=True)
    medians = np.median(sampled, axis=1)
    lower = -np.inf if band.lower is None else band.lower
    upper = np.inf if band.upper is None else band.upper
    outside = np.count_nonzero((medians < lower) | (medians > upper))
    return float(outside) / trials


#: Estimated false alarm rate, in percent, below which a heavy tail is not worth
#: reporting. A band that fires on well under one clean run in a hundred is doing
#: its job, however ragged the distribution behind it looks.
FALSE_ALARM_FLOOR = 1.0


def _classify(
    band: Band,
    values: Sequence[float],
    raw_scale: float,
    false_alarm_pct: float | None,
) -> tuple[int, Finding | None]:
    """How much of the baseline the band excludes, and whether that will bite.

    The count and the consequence are different questions, and keying the finding
    off the count alone cried wolf. Measured against real baselines: an essay probe
    had 15% of its pairs outside the band and an estimated false alarm rate of 0%,
    because a pairwise check takes the median of two dozen distances and the tail
    never moves it far enough. A latency signal had 12% of its values outside and a
    4.1% false alarm rate, because its check median is over three values and
    scatters. The second is worth an alert; the first is a distribution shape.

    A collapse still reports regardless of the estimate. That one is structural: the
    band was never measured, so the rate it implies is not evidence of anything.
    """
    outside = sum(1 for v in values if not band.contains(v))

    if outside and raw_scale <= EPS:
        return outside, Finding.COLLAPSED
    if outside and (false_alarm_pct or 0.0) >= FALSE_ALARM_FLOOR:
        return outside, Finding.SELF_OUTSIDE
    if band.floored and not outside:
        return outside, Finding.FLOORED_STABLE
    return outside, None


def _describe(
    signal: Signal, values: Sequence[float], unit: str, cfg: BandConfig, draw: int = 0
) -> SignalBand | None:
    """Rebuild one signal's band from stored numbers and judge it."""
    if not values:
        return None

    direction = getattr(signal, "direction", Direction.UP_IS_BAD)
    band = robust_band(
        values,
        direction=direction,
        cfg=cfg,
        floor=getattr(signal, "floor", 0.0),
        rel_floor=getattr(signal, "rel_floor", 0.0),
        override=signal.band_override,
    )
    _, raw_scale = robust_centre_scale(values)
    rate = false_alarm_rate(values, band, draw)
    pct = None if rate is None else 100.0 * rate
    outside, finding = _classify(band, values, raw_scale, pct)

    return SignalBand(
        signal=signal.name,
        band=band,
        unit=unit,
        n=len(values),
        observed_min=min(values),
        observed_max=max(values),
        raw_scale=raw_scale,
        outside=outside,
        finding=finding,
        false_alarm_pct=None if rate is None else 100.0 * rate,
        draw=draw,
    )


def inspect(
    baseline: Baseline,
    signals: Sequence[Signal],
    cfg: BandConfig,
    check_samples: int = 3,
) -> ProbeBands:
    """Recompute every band this baseline would be judged against.

    Pairwise signals read their distances straight from the pooled record, which is
    the same list `check` uses. Pointwise signals are recomputed from the stored
    samples, because their values were never persisted separately -- and because
    doing it here is what finally puts them in front of the user. The capture-time
    warning could only ever see pairwise signals, so a floored `length_chars` had
    no way to be mentioned at all.
    """
    out: list[SignalBand] = []
    # What one check reduces to a median. A pairwise signal compares every current
    # sample against every baseline one, so its draw is the product; a pointwise
    # signal only has the current run's own values. The difference is large and it
    # is why the two get very different false alarm rates off the same spread.
    n_baseline = max(1, len(baseline.usable))
    pairwise_draw = check_samples * n_baseline

    for signal in signals:
        if isinstance(signal, PairwiseSignal):
            described = _describe(
                signal, baseline.pooled.get(signal.name) or [], "pairs", cfg, pairwise_draw
            )
        elif isinstance(signal, PointwiseSignal):
            # A None means the signal does not apply to these samples, which is a
            # normal condition rather than an error: Anthropic reports no
            # `system_fingerprint` and names its token counts differently, so those
            # signals stay quiet instead of erroring. Nothing to show either.
            values = [v for v in (signal.value(s) for s in baseline.usable) if v is not None]
            described = _describe(signal, values, "values", cfg, check_samples)
        else:
            # Categorical signals have no band. Any change is an event, so there is
            # no normal range to be right or wrong about.
            described = None

        if described is not None:
            out.append(described)

    return ProbeBands(
        probe_id=baseline.probe_id,
        target_name=baseline.target_name,
        version=baseline.version,
        created=baseline.created,
        n_samples=len(baseline.usable),
        signals=out,
    )


#: Phrased in terms of the observed range rather than its maximum. A collapsed band
#: can be escaped from below as easily as from above -- a length band of 60..76 built
#: from samples spanning 56..68 is breached by the short ones -- and naming only the
#: max there points at a value that is comfortably inside the band.
_EXPLAIN = {
    Finding.COLLAPSED: (
        "the median and MAD are both zero, so the scale could not be measured and the "
        "band fell to its floor. The baseline itself spans {min:.4g}..{max:.4g}, and "
        "{outside} of {n} {unit} ({pct:.0f}%) fall outside the band that was built "
        "from them. The width is a built-in default rather than anything this probe "
        "demonstrated, so it is arbitrary in both directions: see the rate above for "
        "how often it actually fires. Typically the output is bimodal, identical on "
        "most runs and formatted differently on the rest. More samples will not help "
        "while one form dominates, because the median stays put and the MAD stays zero."
    ),
    Finding.SELF_OUTSIDE: (
        "{outside} of {n} {unit} ({pct:.0f}%) fall outside the band built from them, "
        "though the scale is real. A heavy tail rather than a collapse, but the band is "
        "tighter than the probe's own behaviour warrants."
    ),
    Finding.FLOORED_STABLE: (
        "defaulted to a floor rather than measured. The baseline spans "
        "{min:.4g}..{max:.4g}, which fits inside it, so this is only worth knowing "
        "about: fine if the probe really is this steady."
    ),
}


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=78)


def render(probes: Sequence[ProbeBands], verbose: bool = False) -> str:
    """Human-readable inspection report.

    Healthy signals are summarised as a count unless asked for: the point of this
    command is the ones that are wrong, and printing nine clean rows above them
    buries the finding.

    Floored-but-stable bands get named on one line rather than explained on four.
    They are worth listing, since which bands were defaulted is exactly what the
    capture-time warning could never tell you in full, but a paragraph each turns
    a clean report into a wall of text that says nothing is wrong.
    """
    lines: list[str] = []

    for probe in probes:
        lines.append(
            f"{probe.probe_id} @ {probe.target_name}"
            f"   (v{probe.version}, {probe.n_samples} sample(s), captured {probe.created[:10]})"
        )
        if not probe.signals:
            lines.append("  no bands: baseline has no usable samples")
            lines.append("")
            continue

        quiet = 0
        stable: list[str] = []
        for sb in probe.signals:
            if sb.finding is None and not verbose:
                quiet += 1
                continue
            if sb.finding is Finding.FLOORED_STABLE and not verbose:
                stable.append(sb.signal)
                continue

            band = sb.band.describe()
            if sb.band.floored:
                band += " (floor)"
            spread = f"{sb.observed_min:.4g}..{sb.observed_max:.4g}"
            lines.append(
                f"  {sb.signal:<22} band {band:<22} {sb.n:>3} {sb.unit:<7} spread {spread}"
            )
            # Only worth printing when it is not zero. A band that never fires on a
            # clean run is the normal case and saying so on every row would bury the
            # rows where it does.
            if sb.false_alarm_pct:
                lines.append(
                    f"    would report drift on ~{sb.false_alarm_pct:.1f}% of clean runs "
                    f"(median of {sb.draw} {sb.unit})"
                )

            if sb.finding is not None:
                detail = _EXPLAIN[sb.finding].format(
                    min=sb.observed_min,
                    max=sb.observed_max,
                    outside=sb.outside,
                    n=sb.n,
                    unit=sb.unit,
                    pct=sb.outside_pct,
                )
                lines.extend(
                    textwrap.wrap(
                        f"{sb.finding.value.upper()}: {detail}",
                        width=78,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                )

        if stable:
            lines.append(f"  defaulted to a floor, samples agree: {', '.join(stable)}")
        if quiet:
            lines.append(f"  {quiet} other band(s) look sound")
        lines.append("")

    suspect = [s for p in probes for s in p.suspect]
    if suspect:
        names = ", ".join(sorted({s.signal for s in suspect}))
        lines.append(f"{len(suspect)} band(s) will misreport: {names}")
        # The two findings want opposite advice, so giving both at once made half
        # of it wrong on every report.
        if any(s.finding is Finding.COLLAPSED for s in suspect):
            lines += _wrap(
                "A collapsed band is not fixed by recapturing: while one output form "
                "dominates, the median stays put and the scale stays zero. Constrain "
                "the prompt so the probe has one output regime, or pin the band "
                "explicitly in config."
            )
        else:
            lines += _wrap(
                "These have a real scale and a tail the band does not cover. Raise "
                "`baseline_samples` so the tail is better measured, or accept the "
                "rate above as the cost of catching smaller moves."
            )
    else:
        lines.append("All bands look sound.")

    return "\n".join(lines)


def payload(probes: Sequence[ProbeBands]) -> dict:
    """Machine-readable inspection result.

    Reports `outside` and `n` as exact integers and `outside_pct` alongside them,
    so a receiver can either use the ratio directly or recompute it. `finding` is
    null for a sound band rather than absent, which keeps the key set stable across
    every row and saves the consumer a membership test.
    """
    return {
        "tool": "stillsane",
        "command": "bands",
        "suspect": sum(len(p.suspect) for p in probes),
        "probes": [
            {
                "probe": p.probe_id,
                "target": p.target_name,
                "version": p.version,
                "created": p.created,
                "samples": p.n_samples,
                "bands": [
                    {
                        "signal": sb.signal,
                        "finding": sb.finding.value if sb.finding else None,
                        "suspect": sb.finding in SUSPECT,
                        "unit": sb.unit,
                        "n": sb.n,
                        "observed_min": sb.observed_min,
                        "observed_max": sb.observed_max,
                        "raw_scale": sb.raw_scale,
                        "outside": sb.outside,
                        "outside_pct": round(sb.outside_pct, 2),
                        "false_alarm_pct": (
                            None if sb.false_alarm_pct is None else round(sb.false_alarm_pct, 2)
                        ),
                        "check_draw": sb.draw,
                        "band": {
                            "center": sb.band.center,
                            "scale": sb.band.scale,
                            "lower": sb.band.lower,
                            "upper": sb.band.upper,
                            "n": sb.band.n,
                            "floored": sb.band.floored,
                        },
                    }
                    for sb in p.signals
                ],
            }
            for p in probes
        ],
    }


def as_json(probes: Sequence[ProbeBands]) -> str:
    return json.dumps(payload(probes), indent=2)


def build_probe_signals(checks, embedder=None) -> list[Signal]:
    """Signal list for inspection, with no model loaded by default."""
    from .signals import build_signals

    return build_signals(checks, embedder or no_embedder())
