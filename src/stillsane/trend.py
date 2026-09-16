"""The one form of "quiet" drift `check` is structurally blind to.

`check` judges each run alone. A provider that moves `semantic_distance` from
z~0.3 to a steady z~2.4 never crosses `warn_k=3`, so every run says PASS --
forever, if nothing else ever moves it further. The evidence is there the whole
time, in every run's own `observed` value, but nothing ever looks at more than
one run at once to notice the floor itself has risen.

This reads the history database only: no network, no API key, no new sampling,
no new state. Per probe/signal, it compares the median of the earliest runs
recorded against the *current* baseline version to the median of the most
recent ones, both expressed in the same band's units, and says so when the
recent group has drifted somewhere the early group had not.

Two traps shape this on purpose, both from the roadmap item this implements:

* **The reference band is fixed, computed once, not read back from history.**
  Pooling tightens a signal's band over time by design (`compare/pooling.py`),
  so the `z` a run recorded when it happened is not comparable to the `z` a
  later run recorded against a since-tightened band -- trending the *stored*
  `z` values would be comparing numbers computed on different scales and
  calling the difference a shift. `observed` is stored on a fixed scale (the
  signal's own units); converting it to `z` here, once, against today's band
  keeps both ends of the comparison on the same ruler.
* **Scoped to one baseline version.** A sustained shift is exactly what a
  re-baseline should absorb. Averaging in runs from a since-replaced baseline
  would make the first runs after a recapture look like a shift that is really
  just the old baseline's tail, so `History.trend_window` only ever returns
  rows recorded against the exact version currently on disk.

What this is not: a `warn_k`-scaled band, the same one `check` uses, would
never fire here either -- a value that never crosses `warn_k` on any single
run will not cross it as a multi-run median. The threshold below is
`warn_k * grey_zone`, `BandConfig`'s existing name for "elevated enough to be
worth a second look, short of a WARN on its own" (the same fraction the
corroboration path in `compare/variance.py` already uses for a single run).
Reusing it keeps this from inventing a second, uncalibrated number that means
almost the same thing.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field

import numpy as np

from .compare.variance import BandConfig, z_score
from .models import Band, Direction

#: Fewer than this many runs recorded against the current baseline version is
#: not a trend, it is two or three points -- reported anyway (there is no
#: statistical reason four is special), but never with a confident verdict.
MIN_RUNS = 4


@dataclass(frozen=True)
class SignalTrend:
    """Early vs. recent, for one probe/signal, against today's band."""

    probe_id: str
    target: str
    signal: str
    n: int
    window: int
    early_median: float
    recent_median: float
    early_z: float
    recent_z: float
    threshold: float
    #: Fewer than `2 * window` runs recorded, so the early/recent windows
    #: overlap or are built from a handful of points. Reported, not withheld --
    #: `calibrate` reports thin evidence rather than hiding it, and hiding this
    #: would recreate exactly the "silence looks like success" trap `status`
    #: exists to name.
    thin: bool

    @property
    def shifted(self) -> bool:
        """Was the early group inside the elevated-effect threshold, and the
        recent group outside it?

        Both conditions matter. A signal that was *already* elevated at the
        start of this baseline's life is not shifting, it has been like this
        the whole time the baseline has been in force -- worth seeing in the
        raw numbers, but not the finding this command exists to surface, and
        `bands`/`calibrate` are where a baseline that started elevated gets
        diagnosed.
        """
        return abs(self.early_z) <= self.threshold < abs(self.recent_z)


@dataclass(frozen=True)
class MissingTrend:
    """A signal with a band but not enough recorded history yet to compare."""

    probe_id: str
    target: str
    signal: str
    n: int


@dataclass(frozen=True)
class Trend:
    signals: list[SignalTrend] = field(default_factory=list)
    missing: list[MissingTrend] = field(default_factory=list)
    window: int = 5

    @property
    def shifted(self) -> list[SignalTrend]:
        return [s for s in self.signals if s.shifted]


def evaluate_signal(
    probe_id: str,
    target: str,
    signal: str,
    *,
    total: int,
    early_values: list[float],
    recent_values: list[float],
    band: Band,
    direction: Direction,
    cfg: BandConfig,
    window: int,
) -> SignalTrend | MissingTrend:
    """Compare the early and recent windows already fetched for one signal.

    Takes the two windows and the total count rather than a `History`, for the
    same reason `compare/` takes samples rather than a target and `calibrate`
    takes rows rather than a database: the arithmetic is what is worth testing,
    and it should not need SQLite on disk to exercise.
    """
    if total < MIN_RUNS or not early_values or not recent_values:
        return MissingTrend(probe_id=probe_id, target=target, signal=signal, n=total)

    early_median = float(np.median(early_values))
    recent_median = float(np.median(recent_values))
    return SignalTrend(
        probe_id=probe_id,
        target=target,
        signal=signal,
        n=total,
        window=len(recent_values),
        early_median=early_median,
        recent_median=recent_median,
        early_z=z_score(early_median, band, direction),
        recent_z=z_score(recent_median, band, direction),
        threshold=cfg.warn_k * cfg.grey_zone,
        thin=total < 2 * window,
    )


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=78)


def render(trend: Trend) -> str:
    if not trend.signals and not trend.missing:
        return (
            "No baselines with recorded history yet.\n"
            "Run `stillsane baseline` and then `stillsane check` on a schedule; "
            "come back once a few runs have landed."
        )

    lines: list[str] = []
    by_label: dict[str, list[SignalTrend]] = {}
    for s in trend.signals:
        by_label.setdefault(f"{s.probe_id} @ {s.target}", []).append(s)
    missing_by_label: dict[str, list[MissingTrend]] = {}
    for m in trend.missing:
        missing_by_label.setdefault(f"{m.probe_id} @ {m.target}", []).append(m)

    for label in sorted(set(by_label) | set(missing_by_label)):
        lines.append(label)
        for s in sorted(by_label.get(label, []), key=lambda s: abs(s.recent_z), reverse=True):
            flag = " SUSTAINED SHIFT" if s.shifted else ""
            note = "  (thin evidence)" if s.thin else ""
            lines.append(
                f"  {s.signal:<24} early z={s.early_z:+.2f}  recent z={s.recent_z:+.2f}"
                f"  ({s.n} run(s), window {s.window}){note}{flag}"
            )
            if s.shifted:
                lines += textwrap.wrap(
                    f"    moved from {s.early_median:.4g} (early, inside the normal "
                    f"range) to {s.recent_median:.4g} (recent, past "
                    f"{s.threshold:.2f}x normal variance) without any single run "
                    "crossing warn_k -- each run alone still reads as PASS.",
                    width=78, subsequent_indent="    ",
                )
        for m in missing_by_label.get(label, []):
            lines.append(
                f"  {m.signal:<24} {m.n} run(s) recorded under the current baseline "
                f"-- need at least {MIN_RUNS}"
            )
        lines.append("")

    if trend.shifted:
        names = ", ".join(f"{s.signal} on {s.probe_id} @ {s.target}" for s in trend.shifted)
        lines += _wrap(
            f"{len(trend.shifted)} signal(s) show a sustained shift: {names}. Each of "
            "these has been PASSing every run -- no single check ever crossed "
            "warn_k -- while the recent median moved somewhere the early median "
            "had not. That is what this command exists to catch: `check` judges "
            "one run at a time and cannot see it."
        )
    else:
        lines.append("No sustained shift in any signal with enough recorded history.")

    thin = [s for s in trend.signals if s.thin]
    if thin:
        lines.append("")
        lines += _wrap(
            f"{len(thin)} signal(s) above are marked thin evidence: fewer than "
            f"{2 * trend.window} runs recorded under the current baseline, so the "
            "early/recent windows overlap. A trend over a handful of runs is "
            "anecdote, the same caveat `calibrate` gives thin evidence -- treat it "
            "as a direction worth watching, not a finding."
        )

    lines.append("")
    lines += _wrap(
        "Reads history only: no probe was sampled to produce this. A sustained "
        "shift is exactly what `stillsane baseline` should absorb once you have "
        "looked at it -- recapturing resets the comparison, since it is scoped "
        "to the baseline version currently on disk."
    )
    return "\n".join(lines)


def payload(trend: Trend) -> dict:
    return {
        "tool": "stillsane",
        "command": "trend",
        "window": trend.window,
        "shifted": len(trend.shifted),
        "signals": [
            {
                "probe": s.probe_id,
                "target": s.target,
                "signal": s.signal,
                "n": s.n,
                "window": s.window,
                "early_median": s.early_median,
                "recent_median": s.recent_median,
                "early_z": round(s.early_z, 4),
                "recent_z": round(s.recent_z, 4),
                "threshold": round(s.threshold, 4),
                "shifted": s.shifted,
                "thin": s.thin,
            }
            for s in trend.signals
        ],
        "missing": [
            {"probe": m.probe_id, "target": m.target, "signal": m.signal, "n": m.n}
            for m in trend.missing
        ],
    }


def as_json(trend: Trend) -> str:
    return json.dumps(payload(trend), indent=2)
