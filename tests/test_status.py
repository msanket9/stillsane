"""Canary health.

The scenarios here are the ones that actually happened while running this tool
against a real endpoint on a schedule: a laptop asleep at the trigger, a timeout
tuned for a faster probe, and two days where every scheduled run failed without
anyone noticing. None of them were drift, and none of them were visible from
`check` or `history`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from stillsane.status import as_json, assess, parse_every, payload, render

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


def ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def run(run_id: str, hours_ago: float, level: str, retries: int = 0):
    return (run_id, ts(hours_ago), level, retries)


def result(run_id: str, hours_ago: float, probe: str, level: str, signal="semantic_distance",
           detail=None):
    return (ts(hours_ago), run_id, probe, "claude", level, detail)


def build(runs, results, **kw):
    return assess(runs, results, now=NOW, **kw)


# --- Silence is not success ------------------------------------------------


def test_no_runs_says_so_plainly():
    status = build([], [])
    assert status.total_runs == 0
    assert not status.healthy
    assert "No runs recorded yet" in render(status)


def test_overdue_when_a_scheduled_run_was_missed():
    """The failure this command exists for: it stopped running and looked fine.

    Four days of silence on a daily schedule is a dead canary, but `history` shows
    the same three green rows it showed on day one.
    """
    status = build(
        [run("r1", 96, "pass")],
        [result("r1", 96, "p", "pass")],
        expect_every_s=parse_every("24h"),
    )
    assert status.overdue
    assert not status.healthy
    assert "OVERDUE" in render(status)


def test_not_overdue_inside_the_grace_factor():
    """A run that takes ten minutes must not read as late at the next check."""
    status = build(
        [run("r1", 25, "pass")],
        [result("r1", 25, "p", "pass")],
        expect_every_s=parse_every("24h"),
    )
    assert not status.overdue
    assert status.healthy


def test_staleness_is_unknowable_without_a_declared_cadence():
    """Silence only means something if you know how often it should speak.

    Guessing the cadence from the gaps between past runs would make a monitor that
    has always been broken look correct, so this declines to guess.
    """
    status = build([run("r1", 96, "pass")], [result("r1", 96, "p", "pass")])
    assert not status.overdue
    assert "Staleness not checked" in render(status)


# --- Transport errors are not drift ----------------------------------------


def test_transport_errors_are_counted_and_named_as_environment():
    status = build(
        [run("r2", 1, "error"), run("r1", 25, "pass")],
        [
            result("r2", 1, "p", "error", signal="transport", detail="timeout after 60.0s"),
            result("r1", 25, "p", "pass"),
        ],
    )
    assert status.error_runs == 1
    assert not status.healthy  # the latest run measured nothing
    text = render(status)
    assert "transport errors rather than drift" in text
    assert "environment problem, not a model one" in text


def test_error_reasons_are_surfaced_because_they_are_the_diagnosis():
    """`timeout after 60.0s` and `ReadError` call for completely different fixes."""
    status = build(
        [run("r2", 1, "error"), run("r1", 25, "error")],
        [
            result("r2", 1, "p", "error", detail="timeout after 60.0s"),
            result("r1", 25, "p", "error", detail="ReadError: "),
        ],
    )
    probe = status.probes[0]
    assert probe.reasons == ["timeout after 60.0s", "ReadError:"]
    assert "timeout after 60.0s" in render(status)


def test_last_clean_skips_runs_that_measured_nothing():
    """An errored run is not a clean one, however recent it is."""
    status = build(
        [run("r2", 1, "error"), run("r1", 25, "pass")],
        [result("r2", 1, "p", "error"), result("r1", 25, "p", "pass")],
    )
    assert status.last_run == status.outcomes[-1][0]
    assert status.last_clean is not None
    assert status.last_clean < status.last_run


# --- Per probe -------------------------------------------------------------


def test_dead_probe_is_distinguished_from_a_flaky_one():
    """One always fails, one sometimes does. Different problems, different fixes."""
    status = build(
        [run("r2", 1, "error"), run("r1", 25, "error")],
        [
            result("r2", 1, "always", "error", detail="timeout after 60.0s"),
            result("r1", 25, "always", "error", detail="timeout after 60.0s"),
            result("r2", 1, "sometimes", "pass"),
            result("r1", 25, "sometimes", "error", detail="ReadError: "),
        ],
    )
    by_id = {p.probe_id: p for p in status.probes}
    assert by_id["always"].dead and not by_id["always"].flaky
    assert by_id["sometimes"].flaky and not by_id["sometimes"].dead
    assert by_id["sometimes"].error_rate == pytest.approx(0.5)
    assert not status.healthy  # a permanently dead probe is not health

    text = render(status)
    assert "failing every run" in text


def test_probe_level_comes_from_the_newest_run():
    """One row per signal, so an error anywhere in the latest run decides it."""
    status = build(
        [run("r2", 1, "error"), run("r1", 25, "pass")],
        [
            result("r2", 1, "p", "pass", signal="length_chars"),
            result("r2", 1, "p", "error", signal="transport", detail="timeout"),
            result("r1", 25, "p", "pass"),
        ],
    )
    assert status.probes[0].last_level == "error"
    assert status.probes[0].runs == 2
    assert status.probes[0].errors == 1


def test_a_run_counts_once_per_probe_however_many_signals_it_has():
    """Otherwise a probe with nine signals looks like nine runs."""
    status = build(
        [run("r1", 1, "pass")],
        [result("r1", 1, "p", "pass", signal=f"s{i}") for i in range(9)],
    )
    assert status.probes[0].runs == 1


# --- Healthy path and rendering --------------------------------------------


def test_healthy_canary_says_so():
    status = build(
        [run("r2", 1, "pass"), run("r1", 25, "pass")],
        [result("r2", 1, "p", "pass"), result("r1", 25, "p", "pass")],
        expect_every_s=parse_every("24h"),
    )
    assert status.healthy
    assert "Canary looks healthy." in render(status)


def test_outcome_strip_reads_oldest_to_newest():
    """A flapping monitor should be visible at a glance, not reconstructed."""
    status = build(
        [run("r3", 1, "pass"), run("r2", 25, "error"), run("r1", 49, "pass")],
        [result("r3", 1, "p", "pass")],
    )
    assert "P E P" in render(status)


def test_drift_is_not_confused_with_an_error():
    """A drifting canary is working perfectly. Only errors mean it measured nothing."""
    status = build(
        [run("r1", 1, "drift")],
        [result("r1", 1, "p", "drift")],
    )
    assert status.error_runs == 0
    assert status.healthy
    assert "transport errors" not in render(status)


# --- Interval parsing ------------------------------------------------------


@pytest.mark.parametrize(
    "text,seconds",
    [("30s", 30), ("30m", 1800), ("24h", 86400), ("7d", 604800), (" 12H ", 43200)],
)
def test_parse_every_accepts_sensible_intervals(text, seconds):
    assert parse_every(text) == seconds


@pytest.mark.parametrize("text", ["", "24", "h", "24 hours", "-5h", "abc"])
def test_parse_every_rejects_nonsense_rather_than_guessing(text):
    with pytest.raises(ValueError):
        parse_every(text)


# --- Machine readable ------------------------------------------------------


def test_payload_carries_the_health_verdict():
    status = build(
        [run("r2", 1, "error"), run("r1", 25, "pass")],
        [
            result("r2", 1, "p", "error", detail="timeout after 60.0s"),
            result("r1", 25, "p", "pass"),
        ],
        expect_every_s=parse_every("24h"),
    )
    data = payload(status)
    assert data["healthy"] is False
    assert data["error_runs"] == 1
    assert data["expect_every_s"] == 86400
    assert data["probes"][0]["reasons"] == ["timeout after 60.0s"]


def test_json_round_trips():
    status = build([run("r1", 1, "pass")], [result("r1", 1, "p", "pass")])
    data = json.loads(as_json(status))
    assert data["tool"] == "stillsane"
    assert data["command"] == "status"
    assert data["healthy"] is True


def test_recovered_transport_failures_are_reported_not_absorbed():
    """A pass that needed a retry is still a pass against an unwell environment.

    The retry exists so a dropped connection does not cost a day of data. If it
    also made the flakiness invisible it would have traded a loud problem for a
    silent one, which is the failure this whole command exists to prevent.
    """
    status = build(
        [run("r2", 1, "pass", retries=2), run("r1", 25, "pass")],
        [result("r2", 1, "p", "pass"), result("r1", 25, "p", "pass")],
        expect_every_s=parse_every("24h"),
    )
    assert status.total_retries == 2
    assert status.healthy  # recovered runs are still healthy runs
    assert "retried calls   2" in render(status)
    assert payload(status)["total_retries"] == 2


def test_no_retries_means_no_retry_line():
    """A clean week should not carry a row of zeroes."""
    status = build([run("r1", 1, "pass")], [result("r1", 1, "p", "pass")])
    assert status.total_retries == 0
    assert "retried calls" not in render(status)
