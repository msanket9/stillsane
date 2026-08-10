"""Is the canary alive?

Every other command in this tool answers a question about the model. This one
answers a question about the tool: has it actually been running, and have its runs
been telling you anything.

The gap is not hypothetical. A scheduled monitor can fail for reasons that have
nothing to do with what it watches -- a laptop asleep at the trigger, a network
that had not come up, a timeout tuned for a faster probe -- and every one of those
produces a run that completed, recorded an ERROR, and moved on. Nothing is broken
enough to notice. `stillsane check` cannot report this because each run only sees
itself; `stillsane history` shows the rows but leaves you to spot that the last
three were all errors and the one before that was four days ago.

Two distinctions carry most of the value:

* **Transport errors are not drift.** A run that could not reach the endpoint has
  measured nothing. Counting it as a healthy run overstates coverage, and counting
  it as drift is the false alarm the exit codes already exist to prevent.
* **Silence is not success.** A canary that stopped running looks exactly like a
  canary with nothing to report, unless something knows how often it should have
  run. That is what `--expect-every` is for; without it staleness is unknowable
  and this command declines to guess.

Reads the history database and nothing else. No network, no API key, no spend.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .models import Level

#: Multiple of the expected interval past which a run counts as overdue. A little
#: slack, because a scheduler that fires at 09:00 and a run that takes ten minutes
#: should not read as late by the time the next check looks.
STALE_FACTOR = 1.5

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$", re.I)
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_every(text: str) -> float:
    """`30m`, `24h`, `7d` to seconds. Raises on anything else."""
    m = _DURATION.match(text)
    if not m:
        raise ValueError(
            f"Cannot read {text!r} as an interval. Use a number and a unit, "
            "for example 30m, 24h or 7d."
        )
    return float(m.group(1)) * _UNITS[m.group(2).lower()]


def _parse_ts(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _span(seconds: float) -> str:
    """Bare duration with no "ago" on it, so it can also follow a preposition.

    Precision drops as the gap grows, because nobody needs seconds once the answer
    is "four days".
    """
    seconds = max(0.0, seconds)
    if seconds < 90:
        return "under a minute"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    hours = minutes / 60
    if hours < 36:
        return f"{hours:.0f} hours"
    return f"{hours / 24:.0f} days"


def _ago(then: datetime | None, now: datetime) -> str:
    if then is None:
        return "never"
    seconds = max(0.0, (now - then).total_seconds())
    return "just now" if seconds < 90 else f"{_span(seconds)} ago"


@dataclass(frozen=True)
class ProbeHealth:
    """One probe's record across recent runs."""

    probe_id: str
    target: str
    runs: int
    errors: int
    last_level: str
    last_seen: datetime | None
    #: Distinct transport failure reasons, most recent first. These are the actual
    #: diagnosis: "timeout after 60.0s" and "ReadError" call for different fixes.
    reasons: list[str] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.errors / self.runs if self.runs else 0.0

    @property
    def flaky(self) -> bool:
        """Erroring sometimes. Always-erroring is a different, louder problem."""
        return 0 < self.errors < self.runs

    @property
    def dead(self) -> bool:
        return self.runs > 0 and self.errors == self.runs


@dataclass(frozen=True)
class Status:
    total_runs: int
    #: (finished, level) for recent runs, oldest first, so it reads as a timeline.
    outcomes: list[tuple[datetime | None, str]]
    probes: list[ProbeHealth]
    now: datetime
    expect_every_s: float | None = None
    #: Extra calls across these runs that recovered a transport failure. A run that
    #: only passed because a dropped connection was retried is still a run against an
    #: unwell environment, so this is reported rather than absorbed.
    total_retries: int = 0

    @property
    def last_run(self) -> datetime | None:
        return self.outcomes[-1][0] if self.outcomes else None

    @property
    def last_clean(self) -> datetime | None:
        """Most recent run that actually measured something and liked it."""
        for when, level in reversed(self.outcomes):
            if level == Level.PASS.value:
                return when
        return None

    @property
    def error_runs(self) -> int:
        return sum(1 for _, level in self.outcomes if level == Level.ERROR.value)

    @property
    def silent_for(self) -> timedelta | None:
        return None if self.last_run is None else self.now - self.last_run

    @property
    def overdue(self) -> bool:
        """Has it missed a scheduled run? Unknowable without a declared cadence."""
        if self.expect_every_s is None or self.silent_for is None:
            return False
        return self.silent_for.total_seconds() > self.expect_every_s * STALE_FACTOR

    @property
    def healthy(self) -> bool:
        if not self.total_runs or self.overdue:
            return False
        last_level = self.outcomes[-1][1]
        return last_level != Level.ERROR.value and not any(p.dead for p in self.probes)


def assess(
    runs: Sequence[tuple[str, str, str, int]],
    probe_rows: Sequence[tuple[str, str, str, str, str, str | None]],
    *,
    now: datetime | None = None,
    expect_every_s: float | None = None,
) -> Status:
    """Build the health picture from what `History` returns.

    Takes rows rather than a `History` so the whole thing is testable without a
    database on disk, which is the same reason `compare/` takes samples rather than
    a target.
    """
    now = now or datetime.now(timezone.utc)

    outcomes = [(_parse_ts(finished), level) for _, finished, level, _r in runs]
    total_retries = sum(r for *_, r in runs)
    outcomes.reverse()  # `recent` is newest first; a timeline reads the other way.

    # Rows arrive newest run first, so the first sighting of a probe is its most
    # recent one. That ordering is what lets this be a single pass with no sorting:
    # latest timestamp, latest level and most recent failure reasons all fall out of
    # "have I seen this probe yet".
    seen: dict[tuple[str, str], dict] = {}
    for finished, run_id, probe_id, target, level, detail in probe_rows:
        key = (probe_id, target)
        entry = seen.setdefault(
            key,
            {"runs": set(), "errors": set(), "last": None, "level": None,
             "latest_run": None, "reasons": []},
        )
        entry["runs"].add(run_id)

        if entry["latest_run"] is None:
            entry["latest_run"] = run_id
            entry["last"] = _parse_ts(finished)
            entry["level"] = level
        elif entry["latest_run"] == run_id and level == Level.ERROR.value:
            # Still inside the newest run, one signal per row. An error anywhere in
            # it decides the probe's level for that run.
            entry["level"] = level

        if level == Level.ERROR.value:
            entry["errors"].add(run_id)
            reason = (detail or "").strip()
            if reason and reason not in entry["reasons"]:
                entry["reasons"].append(reason)

    probes = [
        ProbeHealth(
            probe_id=probe_id,
            target=target,
            runs=len(e["runs"]),
            errors=len(e["errors"]),
            last_level=e["level"] or Level.PASS.value,
            last_seen=e["last"],
            reasons=e["reasons"][:3],
        )
        for (probe_id, target), e in sorted(seen.items())
    ]

    return Status(
        total_runs=len(runs),
        total_retries=total_retries,
        outcomes=outcomes,
        probes=probes,
        now=now,
        expect_every_s=expect_every_s,
    )


#: One character per run, oldest to newest. A strip like `P P E P E` shows a
#: flapping monitor at a glance in a way a table of timestamps does not.
_MARK = {
    Level.PASS.value: "P",
    Level.WARN.value: "W",
    Level.DRIFT.value: "D",
    Level.ERROR.value: "E",
}


def render(status: Status, limit: int = 20) -> str:
    lines: list[str] = []

    if not status.total_runs:
        return (
            "No runs recorded yet.\n"
            "Run `stillsane check` once, or wait for the first scheduled run."
        )

    last_level = status.outcomes[-1][1]
    lines.append(f"last run        {_ago(status.last_run, status.now)}   {last_level}")
    lines.append(f"last clean run  {_ago(status.last_clean, status.now)}")
    lines.append(f"runs recorded   {status.total_runs}")

    if status.total_retries:
        lines.append(f"retried calls   {status.total_retries}   (transport, recovered)")

    strip = " ".join(_MARK.get(level, "?") for _, level in status.outcomes[-limit:])
    lines.append(f"recent          {strip}   (oldest to newest)")
    lines.append("")

    width = max((len(f"{p.probe_id} @ {p.target}") for p in status.probes), default=0)
    for probe in status.probes:
        label = f"{probe.probe_id} @ {probe.target}"
        if probe.dead:
            note = f"failing every run ({probe.errors}/{probe.runs})"
        elif probe.flaky:
            note = f"errored {probe.errors} of {probe.runs} run(s)"
        else:
            note = f"{probe.runs} run(s), no errors"
        lines.append(f"  {label:<{width}}  {note}")
        for reason in probe.reasons:
            lines.append(f"  {'':<{width}}    {reason}")

    lines.append("")

    def wrap(text: str) -> list[str]:
        return textwrap.wrap(text, width=78)

    if status.overdue and status.expect_every_s and status.silent_for:
        lines += wrap(
            f"OVERDUE: nothing has run for {_span(status.silent_for.total_seconds())}, "
            f"against an expected every {_fmt_seconds(status.expect_every_s)}. "
            "The schedule is not firing. Check the scheduler before trusting the "
            "quiet: a canary that stopped running looks exactly like one with "
            "nothing to report."
        )
    elif status.error_runs:
        # Worth saying plainly, because the exit code alone gets read as "the model
        # broke" by anyone who has not memorised the table.
        lines += wrap(
            f"{status.error_runs} of the last {len(status.outcomes)} run(s) ended in "
            "transport errors rather than drift. Nothing was measured on those runs. "
            "That is an environment problem, not a model one."
        )
    elif status.healthy:
        lines.append("Canary looks healthy.")

    if status.expect_every_s is None:
        lines.append("")
        lines += wrap(
            "Staleness not checked. Pass --expect-every (say 24h) to have this "
            "command tell you when a scheduled run has been missed."
        )

    return "\n".join(lines)


def _fmt_seconds(seconds: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds / size:g}{unit}"
    return f"{seconds:g}s"


def as_json(status: Status) -> str:
    return json.dumps(payload(status), indent=2)


def payload(status: Status) -> dict:
    """Machine-readable health, for a scheduler that wants to alert on the canary."""
    return {
        "tool": "stillsane",
        "command": "status",
        "healthy": status.healthy,
        "overdue": status.overdue,
        "total_runs": status.total_runs,
        "error_runs": status.error_runs,
        "total_retries": status.total_retries,
        "last_run": status.last_run.isoformat() if status.last_run else None,
        "last_clean_run": status.last_clean.isoformat() if status.last_clean else None,
        "silent_for_s": status.silent_for.total_seconds() if status.silent_for else None,
        "expect_every_s": status.expect_every_s,
        "probes": [
            {
                "probe": p.probe_id,
                "target": p.target,
                "runs": p.runs,
                "errors": p.errors,
                "error_rate": round(p.error_rate, 4),
                "flaky": p.flaky,
                "dead": p.dead,
                "reasons": p.reasons,
            }
            for p in status.probes
        ],
    }
