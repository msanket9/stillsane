"""What your own runs say about `warn_k` and `drift_k`.

These two numbers are the least defensible thing in the tool. Everything else is
measured from the probe: the band comes from the probe's own variance, the centre
from its own median. The multipliers on top were picked against constructed drift
scenarios, and the README has said so plainly since the first release because
there was nothing better to say.

There is now. Every clean run records a `z` for every signal, and a clean run is
by definition one where nothing drifted, so those numbers are what normal looks
like measured on the endpoint people actually care about. How close they get to
`warn_k` is how close the tool came to crying wolf.

The one thing this cannot tell you is whether the thresholds are *sensitive
enough*. Clean runs contain no drift, so they carry no information about what
would have been caught. This measures headroom against false alarms and nothing
else, and saying otherwise would be the most tempting misreading of the output.

Reads the history database. No network, no API key, no spend.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: Below this many clean runs the spread is not a distribution, it is anecdote.
#: Reported anyway, with the caveat made loud rather than dropped in a footnote.
THIN_EVIDENCE_RUNS = 10


@dataclass(frozen=True)
class SignalCalibration:
    """One signal's worst behaviour across clean runs, for one probe.

    Scoped to a single probe deliberately. Two probes can share a signal name --
    "length_chars" says nothing about which probe it came from -- and their
    variance is not comparable: that is the entire premise the per-probe band
    exists to encode. A version of this that pooled "length_chars" across every
    probe once reported 2.0x headroom for a name that was, in the real data behind
    it, one probe sitting dead at z=0 the whole time and diluting another probe
    that was genuinely closer to firing. Aggregating hid the probe that mattered.
    """

    probe_id: str
    target: str
    signal: str
    n: int
    z_max_abs: float
    z_p95_abs: float
    #: Did the raw observed value ever differ from the band's centre on a clean
    #: run, even though `z_max_abs` is 0? `z_score` clamps a one-sided signal to
    #: 0 whenever it moves in the safe direction (a latency that got faster),
    #: so `z_max_abs == 0` alone cannot tell "this never varies" apart from
    #: "this varies plenty, always safely" -- and reporting the second as
    #: "never moved" tells the reader the wrong one of those two facts.
    moved_only_safely: bool = False

    @property
    def label(self) -> str:
        return f"{self.probe_id} @ {self.target}"

    def headroom(self, warn_k: float) -> float | None:
        """How many times further the worst clean run would have had to go to fire.

        None when the observed maximum is zero. That is not headroom of
        infinity -- there is no number of "times further" a signal that cannot
        fire would need to go -- whether because it never moved, or because
        every move it made was in the direction nobody alerts on. `render`
        distinguishes those two cases via `moved_only_safely` rather than here,
        since this method answers "how much fire-risk headroom is there",
        and the answer is the same (none) either way.
        """
        return None if self.z_max_abs <= 0 else warn_k / self.z_max_abs

    def fires(self, warn_k: float) -> bool:
        """Did a clean run already cross the warn threshold? A false alarm."""
        return self.z_max_abs > warn_k


@dataclass(frozen=True)
class Calibration:
    #: Worst-first overall, but each row still belongs to exactly one probe. Never
    #: collapse two probes' rows into one just because the signal name matches.
    signals: list[SignalCalibration]
    clean_runs: int
    warn_k: float
    drift_k: float
    first: str | None = None
    last: str | None = None
    probes: list[str] = field(default_factory=list)

    @property
    def thin(self) -> bool:
        return self.clean_runs < THIN_EVIDENCE_RUNS

    @property
    def worst(self) -> SignalCalibration | None:
        return max(self.signals, key=lambda s: s.z_max_abs) if self.signals else None

    @property
    def false_alarms(self) -> list[SignalCalibration]:
        return [s for s in self.signals if s.fires(self.warn_k)]

    @property
    def tightest_k(self) -> float | None:
        """The smallest `warn_k` that would still not have fired on this evidence.

        A floor rather than a recommendation. Setting `warn_k` exactly here would
        make the next slightly-unluckier clean run an alert, which is why the report
        says so instead of printing it as an answer.
        """
        w = self.worst
        return None if w is None or w.z_max_abs <= 0 else w.z_max_abs

    def by_probe(self) -> list[tuple[str, list[SignalCalibration]]]:
        """Rows grouped by probe, each group worst-first, groups worst-first.

        The grouping a reader actually wants: which probe should I look at first,
        and within it, which signal.
        """
        groups: dict[str, list[SignalCalibration]] = {}
        for s in self.signals:
            groups.setdefault(s.label, []).append(s)
        ordered = sorted(
            groups.items(), key=lambda kv: max(s.z_max_abs for s in kv[1]), reverse=True
        )
        return [(label, sorted(rows, key=lambda s: s.z_max_abs, reverse=True))
                for label, rows in ordered]


#: Below this, a difference between `observed` and `baseline` is float noise
#: from serialisation/reconstruction rather than a real move worth reporting.
_MOVED_EPS = 1e-9


def assess(
    rows: Sequence[tuple[str, str, str, float, float | None, float | None]],
    *,
    clean_runs: int,
    warn_k: float,
    drift_k: float,
    first: str | None = None,
    last: str | None = None,
) -> Calibration:
    """Summarise `z` on clean runs, per probe, per signal.

    Takes rows rather than a `History` for the same reason `compare/` takes samples
    rather than a target: the arithmetic is the part worth testing, and it should
    not need a database on disk to exercise.
    """
    by_key: dict[tuple[str, str, str], list[float]] = {}
    moved_raw: dict[tuple[str, str, str], bool] = {}
    probes: set[str] = set()
    for probe_id, target, signal, z, observed, baseline in rows:
        key = (probe_id, target, signal)
        by_key.setdefault(key, []).append(abs(z))
        if observed is not None and baseline is not None:
            moved_raw[key] = moved_raw.get(key, False) or abs(observed - baseline) > _MOVED_EPS
        probes.add(f"{probe_id} @ {target}")

    signals = []
    for (probe_id, target, signal), values in sorted(by_key.items()):
        arr = np.asarray(values, dtype=float)
        z_max_abs = float(arr.max())
        signals.append(
            SignalCalibration(
                probe_id=probe_id,
                target=target,
                signal=signal,
                n=len(arr),
                z_max_abs=z_max_abs,
                z_p95_abs=float(np.percentile(arr, 95)),
                moved_only_safely=(
                    z_max_abs <= 0 and moved_raw.get((probe_id, target, signal), False)
                ),
            )
        )
    signals.sort(key=lambda s: s.z_max_abs, reverse=True)

    return Calibration(
        signals=signals,
        clean_runs=clean_runs,
        warn_k=warn_k,
        drift_k=drift_k,
        first=first,
        last=last,
        probes=sorted(probes),
    )


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=78)


def render(cal: Calibration) -> str:
    if not cal.signals:
        return (
            "No clean runs recorded yet, so there is nothing to calibrate against.\n"
            "Thresholds are measured from runs where nothing drifted; run "
            "`stillsane check` on a schedule and come back."
        )

    lines = [
        f"Calibration from {cal.clean_runs} clean run(s)"
        + (f", {cal.first[:10]} to {cal.last[:10]}" if cal.first and cal.last else ""),
        f"{len(cal.probes)} probe(s): {', '.join(cal.probes)}",
        "",
    ]

    for label, rows in cal.by_probe():
        lines.append(f"{label}")
        lines.append(f"  {'signal':<24}{'n':>5}{'|z| p95':>10}{'|z| max':>10}{'headroom':>14}")
        for s in rows:
            head = s.headroom(cal.warn_k)
            if head is not None:
                head_s = f"{head:.1f}x"
            elif s.moved_only_safely:
                head_s = "safe only"
            else:
                head_s = "never moved"
            lines.append(
                f"  {s.signal:<24}{s.n:>5}{s.z_p95_abs:>10.2f}{s.z_max_abs:>10.2f}{head_s:>14}"
            )
        lines.append("")

    if cal.false_alarms:
        names = ", ".join(f"{s.signal} on {s.label}" for s in cal.false_alarms)
        lines += _wrap(
            f"warn_k={cal.warn_k:g} already fires on clean runs: {names} exceeded it "
            "while nothing had drifted. These are false alarms being recorded as "
            "warnings, and the threshold is too tight for these probes."
        )
    elif cal.tightest_k is None:
        safe_movers = [s for s in cal.signals if s.moved_only_safely]
        if safe_movers:
            # These did move -- `z_max_abs` is 0 only because every move was in
            # the direction nobody alerts on (a latency that got faster). That
            # is a real fact about the probe, not an absence of one, and it is
            # a different finding from a signal that is truly constant.
            names = ", ".join(f"{s.signal} on {s.label}" for s in safe_movers)
            lines += _wrap(
                f"No signal crossed centre on any clean run, but {names} did move -- "
                "always in the direction nobody alerts on (e.g. a latency that got "
                "faster). There is still no evidence here about how close the "
                "thresholds are to firing; check `stillsane history` for what these "
                "signals actually did."
            )
        else:
            # Every signal sat exactly on its centre on every clean run. That is not
            # enormous headroom, it is an absence of measurement: nothing has yet
            # demonstrated that these probes vary at all, so there is no evidence
            # about where a threshold should sit.
            lines += _wrap(
                "No signal moved at all on any clean run, so there is nothing to measure "
                "headroom against. That is not a verdict that the thresholds are safe, it "
                "is an absence of evidence either way. Check `stillsane bands` for whether "
                "these probes have measurable variance in the first place."
            )
    else:
        worst = cal.worst
        lines += _wrap(
            f"No clean run came closer than {cal.warn_k / worst.z_max_abs:.1f}x to the "
            f"warn threshold. The worst was {worst.signal} on {worst.label} at "
            f"|z|={worst.z_max_abs:.2f} against warn_k={cal.warn_k:g}, so on this "
            "evidence the threshold is conservative rather than trigger-happy."
        )

    if cal.tightest_k:
        lines.append("")
        lines += _wrap(
            f"The tightest warn_k that would not have fired on this data is "
            f"{cal.tightest_k:.2f}. That is a floor, not a recommendation: setting it "
            "there makes the next slightly unluckier clean run an alert."
        )

    lines.append("")
    if cal.thin:
        lines += _wrap(
            f"Based on {cal.clean_runs} clean run(s), which is thin. A handful of runs "
            "cannot show you the tail, and the tail is what a threshold is for. Treat "
            "this as a direction rather than a number."
        )
        lines.append("")

    # The misreading this output invites, said plainly rather than left implicit.
    lines += _wrap(
        "This measures headroom against false alarms only. Clean runs contain no "
        "drift, so nothing here says whether these thresholds would catch a real "
        "regression. Loosening k on the strength of this would trade a problem you "
        "can see for one you cannot."
    )
    return "\n".join(lines)


def payload(cal: Calibration) -> dict:
    return {
        "tool": "stillsane",
        "command": "calibrate",
        "clean_runs": cal.clean_runs,
        "thin_evidence": cal.thin,
        "warn_k": cal.warn_k,
        "drift_k": cal.drift_k,
        "tightest_warn_k": cal.tightest_k,
        "false_alarms": [
            {"probe": s.probe_id, "target": s.target, "signal": s.signal}
            for s in cal.false_alarms
        ],
        "first": cal.first,
        "last": cal.last,
        "signals": [
            {
                "probe": s.probe_id,
                "target": s.target,
                "signal": s.signal,
                "n": s.n,
                "z_p95_abs": round(s.z_p95_abs, 4),
                "z_max_abs": round(s.z_max_abs, 4),
                "headroom": (
                    None if s.headroom(cal.warn_k) is None else round(s.headroom(cal.warn_k), 3)
                ),
                "moved_only_safely": s.moved_only_safely,
            }
            for s in cal.signals
        ],
    }


def as_json(cal: Calibration) -> str:
    return json.dumps(payload(cal), indent=2)
