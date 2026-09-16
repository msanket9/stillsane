"""`_probe_streak`: "since when has this probe been non-PASS, uninterrupted."

Tested directly with synthetic timestamps rather than only through a live
`check()` call, because history timestamps are second-resolution and a fast
test suite can run several checks inside one wall-clock second -- exactly the
case that would make `first_seen` look right by coincidence rather than by
the walk-backward logic actually working.
"""

from __future__ import annotations

from stillsane.runner import _probe_streak


def test_a_fresh_streak_is_just_this_run():
    """No prior history at all: this run alone is the whole streak."""
    first_seen, consecutive = _probe_streak([], "2026-09-14T06:00:00+00:00")
    assert first_seen == "2026-09-14T06:00:00+00:00"
    assert consecutive == 1


def test_streak_extends_through_consecutive_non_pass_runs():
    """Monday through Thursday all drifted; Friday's run should say "day 5"
    and point back to Monday, not to whichever run happened to be most recent.
    """
    prior = [  # newest first, as `History.probe_recent_levels` returns
        ("2026-09-17T06:00:00+00:00", 2),  # Thu: drift
        ("2026-09-16T06:00:00+00:00", 1),  # Wed: warn
        ("2026-09-15T06:00:00+00:00", 2),  # Tue: drift
        ("2026-09-14T06:00:00+00:00", 2),  # Mon: drift
    ]
    first_seen, consecutive = _probe_streak(prior, "2026-09-18T06:00:00+00:00")  # Fri
    assert first_seen == "2026-09-14T06:00:00+00:00"
    assert consecutive == 5


def test_streak_stops_at_the_most_recent_pass():
    """A clean run anywhere in the lookback ends the streak there -- runs
    further back than that must not be folded into "since when".
    """
    prior = [
        ("2026-09-17T06:00:00+00:00", 2),  # drift
        ("2026-09-16T06:00:00+00:00", 0),  # pass -- the streak stops here
        ("2026-09-15T06:00:00+00:00", 2),  # older drift, must be excluded
    ]
    first_seen, consecutive = _probe_streak(prior, "2026-09-18T06:00:00+00:00")
    assert first_seen == "2026-09-17T06:00:00+00:00"
    assert consecutive == 2


def test_streak_stops_at_the_end_of_recorded_history():
    """No PASS anywhere in what was fetched: the whole lookback window counts,
    not just some default -- there is nothing further back to disqualify it.
    """
    prior = [
        ("2026-09-16T06:00:00+00:00", 1),
        ("2026-09-15T06:00:00+00:00", 1),
    ]
    first_seen, consecutive = _probe_streak(prior, "2026-09-17T06:00:00+00:00")
    assert first_seen == "2026-09-15T06:00:00+00:00"
    assert consecutive == 3


def test_error_and_warn_and_drift_all_extend_the_streak_the_same_way():
    """The streak is "non-PASS", not "the same level" -- a probe erroring one
    day and drifting the next is still one unbroken incident.
    """
    prior = [
        ("2026-09-16T06:00:00+00:00", 3),  # error
        ("2026-09-15T06:00:00+00:00", 1),  # warn
    ]
    first_seen, consecutive = _probe_streak(prior, "2026-09-17T06:00:00+00:00")  # drift
    assert first_seen == "2026-09-15T06:00:00+00:00"
    assert consecutive == 3
