"""The "what changed" report.

Developers want the diff, not a score. So the report leads with which signals
moved and by how much in units of that probe's own variance, then shows the
before/after text, and stays quiet about everything that did not move.

The one formatting rule worth stating: never print a number the reader cannot act
on. An effect size of `None` prints as blank rather than as a fabricated figure,
and signals that passed are summarised as a count rather than listed.
"""

from __future__ import annotations

import os
import sys

from .models import Level, ProbeVerdict, RunResult, SignalVerdict

_COLOURS = {
    Level.PASS: "\033[32m",
    Level.WARN: "\033[33m",
    Level.DRIFT: "\033[31m",
    Level.ERROR: "\033[35m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"


def use_colour(stream=None) -> bool:
    """Respect NO_COLOR and pipes. Nobody wants escape codes in a CI log file."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


class Painter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def level(self, text: str, level: Level) -> str:
        if not self.enabled:
            return text
        return f"{_COLOURS.get(level, '')}{text}{_RESET}"

    def dim(self, text: str) -> str:
        return f"{_DIM}{text}{_RESET}" if self.enabled else text


def _signal_line(sv: SignalVerdict, paint: Painter) -> str:
    """One row: what the signal is, where it sits, and how far that is from normal."""
    observed = sv.observed_label or (f"{sv.observed:.4g}" if sv.observed is not None else "")
    band = f"band {sv.band.describe()}" if sv.band else ""
    effect = f"z={sv.z:+.1f}" if sv.z is not None else ""

    row = f"  {sv.signal:<20} {observed:>12}  {band:<20} {effect:<8}"
    if sv.level is not Level.PASS:
        row = paint.level(row, sv.level)
    return row.rstrip()


#: Signals that say nothing about what the model actually wrote. When only these
#: move there is no textual change to show, and printing a before/after anyway
#: invites the reader to hunt for a difference that is not the point -- a
#: fingerprint-only alert would display `1240.5` against `1240.50` and imply the
#: number moved.
_NON_CONTENT_SIGNALS = frozenset({"fingerprint", "model_id", "latency_ms", "transport"})


def _excerpt_block(verdict: ProbeVerdict, paint: Painter, width: int = 76) -> list[str]:
    """Before and after, stacked rather than side by side.

    Side-by-side columns look tidy in a design mock and are unreadable the moment
    the text is longer than about forty characters, which model output always is.
    """
    if not (verdict.baseline_excerpt and verdict.observed_excerpt):
        return []
    if verdict.baseline_excerpt == verdict.observed_excerpt:
        return []
    if all(sv.signal in _NON_CONTENT_SIGNALS for sv in verdict.moved):
        return []

    lines = [""]
    label = f"baseline (v{verdict.baseline_version}"
    if verdict.baseline_created:
        label += f", {verdict.baseline_created[:10]}"
    label += "):"
    lines.append(paint.dim(f"  {label}"))
    lines += [f"    {ln[:width]}" for ln in verdict.baseline_excerpt.splitlines()[:6]]
    lines.append(paint.dim("  now:"))
    lines += [f"    {ln[:width]}" for ln in verdict.observed_excerpt.splitlines()[:6]]
    return lines


def render_probe(verdict: ProbeVerdict, paint: Painter, verbose: bool = False) -> str:
    header = paint.level(f"{verdict.level.value.upper():<5}", verdict.level)
    lines = [f"{header}  {verdict.probe_id} @ {verdict.target_name}"]

    shown = verdict.signals if verbose else verdict.moved
    for sv in shown:
        lines.append(_signal_line(sv, paint))

    if not verbose:
        quiet = len(verdict.signals) - len(verdict.moved)
        if quiet and verdict.moved:
            lines.append(paint.dim(f"  {quiet} other signal(s) unchanged"))

    if verdict.level is not Level.PASS:
        lines += _excerpt_block(verdict, paint)
        if verdict.judge_note:
            lines.append("")
            lines.append(paint.dim(f"  -> {verdict.judge_note}"))

    return "\n".join(lines)


def render(result: RunResult, verbose: bool = False, colour: bool | None = None) -> str:
    paint = Painter(use_colour() if colour is None else colour)

    if not result.probes:
        return "No probes ran. Check that your config defines at least one probe and target."

    blocks = []
    for verdict in result.probes:
        # A passing probe is one line unless asked otherwise. The signal-to-noise
        # ratio of this output is what determines whether it gets read at all.
        if verdict.level is Level.PASS and not verbose:
            blocks.append(
                f"{paint.level('PASS ', Level.PASS)}  {verdict.probe_id} @ {verdict.target_name}"
            )
        else:
            blocks.append(render_probe(verdict, paint, verbose))

    counts: dict[Level, int] = {}
    for verdict in result.probes:
        counts[verdict.level] = counts.get(verdict.level, 0) + 1
    summary = "  ".join(
        f"{count} {level.value}" for level, count in sorted(counts.items(), key=lambda kv: kv[0].rank)
    )

    separator = "-" * 60
    tail = paint.level(f"{result.level.value.upper()}", result.level)
    return "\n\n".join(blocks) + f"\n\n{separator}\n{summary}   ->  {tail}"


def render_plain(result: RunResult, verbose: bool = False) -> str:
    """Colour-free rendering, for webhooks and log files."""
    return render(result, verbose=verbose, colour=False)
