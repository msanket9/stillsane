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


def _classify(band: Band, values: Sequence[float], raw_scale: float) -> tuple[int, Finding | None]:
    """How much of the baseline the band excludes, and what that means."""
    outside = sum(1 for v in values if not band.contains(v))

    if outside and raw_scale <= EPS:
        return outside, Finding.COLLAPSED
    if outside:
        return outside, Finding.SELF_OUTSIDE
    if band.floored:
        return outside, Finding.FLOORED_STABLE
    return outside, None


def _describe(signal: Signal, values: Sequence[float], unit: str, cfg: BandConfig) -> SignalBand | None:
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
    outside, finding = _classify(band, values, raw_scale)

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
    )


def inspect(baseline: Baseline, signals: Sequence[Signal], cfg: BandConfig) -> ProbeBands:
    """Recompute every band this baseline would be judged against.

    Pairwise signals read their distances straight from the pooled record, which is
    the same list `check` uses. Pointwise signals are recomputed from the stored
    samples, because their values were never persisted separately -- and because
    doing it here is what finally puts them in front of the user. The capture-time
    warning could only ever see pairwise signals, so a floored `length_chars` had
    no way to be mentioned at all.
    """
    out: list[SignalBand] = []

    for signal in signals:
        if isinstance(signal, PairwiseSignal):
            described = _describe(signal, baseline.pooled.get(signal.name) or [], "pairs", cfg)
        elif isinstance(signal, PointwiseSignal):
            # A None means the signal does not apply to these samples, which is a
            # normal condition rather than an error: Anthropic reports no
            # `system_fingerprint` and names its token counts differently, so those
            # signals stay quiet instead of erroring. Nothing to show either.
            values = [v for v in (signal.value(s) for s in baseline.usable) if v is not None]
            described = _describe(signal, values, "values", cfg)
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
        "{outside} of {n} {unit} ({pct:.0f}%) fall outside the band. A check drawing "
        "those reports drift against an endpoint that has not changed. Typically the "
        "output is bimodal: identical on most runs, formatted differently on the rest. "
        "More samples will not help while one form dominates, because the median stays "
        "put and the MAD stays zero."
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
        lines.append("Recapture will not help a collapsed band. Consider a probe whose output")
        lines.append("varies less arbitrarily, or pin the band explicitly in config.")
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
