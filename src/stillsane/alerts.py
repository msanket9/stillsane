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

from .models import EXIT_CODES, Level, RunResult
from .report import render_plain

#: Slack rejects messages over 40k; well before that they become unreadable.
SLACK_LIMIT = 2800


def payload_for(result: RunResult) -> dict:
    """Generic JSON body. Structured, so a receiver can route on it."""
    return {
        "tool": "stillsane",
        "level": result.level.value,
        "exit_code": result.exit_code,
        "finished": result.finished.isoformat() if result.finished else None,
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
    headline = (
        f"{icon} stillsane: {result.level.value.upper()}"
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
    except httpx.HTTPError as exc:
        print(f"stillsane: could not deliver alert: {exc}", file=sys.stderr)
        return False


def send(result: RunResult, webhook: str | None, slack_webhook: str | None) -> None:
    """Fire configured alerts. Only ever called when something actually moved."""
    if webhook:
        _post(webhook, payload_for(result))
    if slack_webhook:
        _post(slack_webhook, slack_payload(result))


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
