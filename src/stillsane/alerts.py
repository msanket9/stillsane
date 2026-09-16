"""Alerting: webhook, Slack, exit code. Nothing else.

Alerts are best-effort by design. A webhook that is down must not turn a
successful drift check into a failed one -- the check already did its job, and
losing the result because the notification failed would be worse than the
notification failing quietly. Delivery failures are reported on stderr and
otherwise ignored.
"""

from __future__ import annotations

import json
import sys

import httpx

from .models import EXIT_CODES, Level, ProbeVerdict, RunResult
from .report import render_plain

#: Slack rejects messages over 40k; well before that they become unreadable.
SLACK_LIMIT = 2800


def payload_for(result: RunResult) -> dict:
    """Generic JSON body. Structured, so a receiver can route on it."""
    total_calls = sum(p.total_calls for p in result.probes)
    known_calls = sum(p.cost_known_calls for p in result.probes)
    total_cost = (
        sum(p.cost_usd for p in result.probes if p.cost_usd is not None)
        if known_calls
        else None
    )
    return {
        "tool": "stillsane",
        "level": result.level.value,
        "exit_code": result.exit_code,
        "finished": result.finished.isoformat() if result.finished else None,
        # "Near-zero running cost" is a claim a CI pipeline parsing this JSON
        # cannot verify from `level`/`exit_code` alone. `cost_usd` is `None`
        # (not `0.0`) when nothing reported a cost, same reason the text
        # report omits the line entirely rather than printing a fabricated
        # `$0.0000` -- most gateways never report cost at all.
        "calls": total_calls,
        "cost_usd": total_cost,
        "cost_known_calls": known_calls,
        "probes": [
            {
                "probe": p.probe_id,
                "target": p.target_name,
                "level": p.level.value,
                # A run that only passed because a dropped connection was retried is
                # still a run against an unwell environment. The text report and
                # `status` already say so; the JSON payload -- the one a CI pipeline
                # actually parses -- did not, which made it invisible to exactly the
                # consumer retries were meant to keep informed.
                "retries": p.retries,
                "stale_comparison": p.stale_comparison,
                "total_calls": p.total_calls,
                "cost_usd": p.cost_usd,
                "cost_known_calls": p.cost_known_calls,
                # "Since when" is the first question an alert provokes; these
                # answer it without sending the reader to `history`. `None`/`0`
                # for a probe that passed, same as everywhere else `None` means
                # "the question does not apply" rather than "zero".
                "first_seen": p.first_seen,
                "consecutive_runs": p.consecutive_runs,
                "moved": [
                    {
                        "signal": sv.signal,
                        "level": sv.level.value,
                        "detail": sv.detail,
                        "z": sv.z,
                        "observed": sv.observed,
                        "baseline": sv.baseline,
                    }
                    for sv in p.moved
                ],
            }
            for p in result.probes
        ],
    }


def slack_payload(result: RunResult) -> dict:
    icon = {Level.PASS: ":white_check_mark:", Level.WARN: ":warning:"}.get(
        result.level, ":rotating_light:"
    )
    moved = [p for p in result.probes if p.level is not Level.PASS]
    # The longest-running issue among the probes that moved, not an average or
    # whichever sorts first: the headline is one line, and "how long has the
    # worst of this been going" is the fact worth leading with. Omitted on a
    # fresh, first-day verdict (`streak <= 1`) -- everyone already knows today
    # is day one, and saying so on every ordinary alert would bury the runs
    # where the count is actually the news.
    streak = max((p.consecutive_runs for p in moved), default=0)
    day = f" (day {streak})" if streak > 1 else ""
    headline = (
        f"{icon} stillsane: {result.level.value.upper()}{day}"
        f" ({len(moved)}/{len(result.probes)} probes moved)"
    )
    body = render_plain(result)
    if len(body) > SLACK_LIMIT:
        body = body[:SLACK_LIMIT] + "\n... [truncated]"
    return {"text": f"{headline}\n```\n{body}\n```"}


def _post(url: str, body: dict, timeout: float = 10.0) -> bool:
    try:
        response = httpx.post(url, json=body, timeout=timeout)
        if response.status_code >= 400:
            print(
                f"stillsane: alert POST to {url.split('?')[0]} returned "
                f"HTTP {response.status_code}",
                file=sys.stderr,
            )
            return False
        return True
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        # `httpx.InvalidURL` is not an `httpx.HTTPError` subclass, so a malformed
        # `webhook`/`slack_webhook` in the config (a stray control character, an
        # unterminated IPv6-literal bracket) raised straight through `send()` and
        # crashed the whole `check` invocation on a run that had already produced
        # a valid verdict -- exactly what this module's own docstring says must
        # never happen.
        print(f"stillsane: could not deliver alert: {exc}", file=sys.stderr)
        return False


def send(result: RunResult, webhook: str | None, slack_webhook: str | None) -> None:
    """Fire configured alerts. Only ever called when something actually moved."""
    if webhook:
        _post(webhook, payload_for(result))
    if slack_webhook:
        _post(slack_webhook, slack_payload(result))


def should_alert(probe: ProbeVerdict, repeat_every: int | None) -> bool:
    """Whether this probe's verdict earns a place in this run's alert.

    `repeat_every` is `AlertConfig.repeat_every`. `None`, the default, means
    no suppression at all: every non-PASS probe always alerts, which is what
    keeps a monitor behaving like one out of the box -- this is opt-in.

    A verdict that just changed always alerts regardless of `repeat_every`
    (`consecutive_runs <= 1` covers both "first time ever" and "was passing
    last run"): suppression is about not repeating news the reader already
    has, never about missing news that is new. `repeat_every == 0` then
    suppresses every later repeat of an unbroken streak outright; a positive
    N instead re-sends every N runs into it, so a long-running drift is not
    forgotten entirely.
    """
    if probe.level is Level.PASS:
        return False
    if repeat_every is None or probe.consecutive_runs <= 1:
        return True
    if repeat_every <= 0:
        return False
    return (probe.consecutive_runs - 1) % repeat_every == 0


def exit_code_for(result: RunResult, fail_on_warn: bool = False) -> int:
    """Map a run to a process exit code.

    WARN is its own code so CI can decide. Escalating it to a failure when
    `fail_on_warn` is set keeps the meaning of "non-zero" consistent for anyone
    who just checks truthiness.
    """
    code = EXIT_CODES[result.level]
    if fail_on_warn and result.level is Level.WARN:
        return EXIT_CODES[Level.DRIFT]
    return code


def as_json(result: RunResult) -> str:
    return json.dumps(payload_for(result), indent=2)
