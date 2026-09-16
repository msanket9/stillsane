"""Sustained sub-threshold shift -- the one drift `check` cannot see because it
judges one run at a time.

These tests cover the arithmetic (`evaluate_signal`) and the caveats that keep
the report honest, the same split `test_calibrate.py` uses for the sibling
command this one is modelled on.
"""

from __future__ import annotations

import json

import pytest

from stillsane.compare.variance import BandConfig
from stillsane.models import Band, Direction
from stillsane.trend import MIN_RUNS, MissingTrend, SignalTrend, Trend, as_json, payload, render
from stillsane.trend import evaluate_signal as _evaluate

CFG = BandConfig()  # warn_k=3.0, grey_zone=0.6 -> threshold 1.8

#: An UP_IS_BAD band: normal sits at 0.10, one unit of z is 0.02.
BAND = Band(center=0.10, scale=0.02, lower=None, upper=0.10 + CFG.warn_k * 0.02, n=8)


def evaluate(early, recent, *, total=None, window=5, band=BAND, direction=Direction.UP_IS_BAD):
    return _evaluate(
        "probe", "target", "signal",
        total=total if total is not None else max(len(early), len(recent)) * 2,
        early_values=early, recent_values=recent,
        band=band, direction=direction, cfg=CFG, window=window,
    )


def flat(text: str) -> str:
    return " ".join(text.split())


# --- The arithmetic ---------------------------------------------------------


def test_the_reports_own_motivating_example_is_flagged():
    """z~0.3 to a steady z~2.4 with warn_k=3: never crosses warn_k on any single
    run, so `check` says PASS forever. This is the exact shape §1.3 exists for.
    """
    early = [BAND.center + 0.3 * BAND.scale] * 5
    recent = [BAND.center + 2.4 * BAND.scale] * 5
    result = evaluate(early, recent)
    assert isinstance(result, SignalTrend)
    assert result.early_z == pytest.approx(0.3)
    assert result.recent_z == pytest.approx(2.4)
    assert result.shifted


def test_a_signal_elevated_since_the_start_is_not_a_shift():
    """Already outside the elevated threshold at the *early* window is not a
    shift -- it has been like this the whole time this baseline has been in
    force, which `bands`/`calibrate` diagnose, not this command.
    """
    already_high = [BAND.center + 2.4 * BAND.scale] * 5
    result = evaluate(already_high, already_high)
    assert not result.shifted


def test_a_signal_that_never_moved_is_not_a_shift():
    steady = [BAND.center] * 5
    result = evaluate(steady, steady)
    assert result.early_z == 0.0 and result.recent_z == 0.0
    assert not result.shifted


def test_a_move_that_stays_below_the_elevated_threshold_is_not_a_shift():
    """A genuine but small move -- inside grey_zone*warn_k on both ends -- is
    exactly the ordinary variance `check`'s own band already tolerates.
    """
    early = [BAND.center + 0.2 * BAND.scale] * 5
    recent = [BAND.center + 1.0 * BAND.scale] * 5
    result = evaluate(early, recent)
    assert not result.shifted


def test_direction_matters_a_latency_that_only_got_faster_is_not_a_shift():
    """`z_score` clamps a one-sided signal's safe-direction moves to 0, the
    same clamp `calibrate` already relies on -- a latency band getting faster
    must not read as an elevated effect size just because the raw numbers moved.
    """
    band = Band(center=100.0, scale=10.0, lower=None, upper=100.0 + CFG.warn_k * 10.0, n=8)
    early = [100.0] * 5
    recent = [40.0] * 5  # much faster, which is "safe" for UP_IS_BAD
    result = evaluate(early, recent, band=band, direction=Direction.UP_IS_BAD)
    assert result.recent_z == 0.0
    assert not result.shifted


# --- Insufficient evidence ---------------------------------------------------


def test_fewer_than_min_runs_is_reported_as_missing_not_as_no_shift():
    """Too little history to say anything is a different fact from "checked and
    found nothing" -- collapsing them would read as false reassurance.
    """
    result = evaluate([0.10], [0.10], total=MIN_RUNS - 1)
    assert isinstance(result, MissingTrend)
    assert result.n == MIN_RUNS - 1


def test_exactly_min_runs_is_evaluated():
    result = evaluate([0.10] * 2, [0.10] * 2, total=MIN_RUNS)
    assert isinstance(result, SignalTrend)


def test_thin_flag_when_windows_overlap():
    """Fewer than `2 * window` runs means the early/recent windows cannot be
    fully disjoint -- reported, not withheld, same as `calibrate`'s thin-run
    caveat.
    """
    result = evaluate([0.10] * 3, [0.10] * 3, total=5, window=5)
    assert result.thin


def test_not_thin_with_enough_runs():
    result = evaluate([0.10] * 5, [0.10] * 5, total=10, window=5)
    assert not result.thin


# --- Rendering ---------------------------------------------------------------


def _trend(signals=(), missing=(), window=5):
    return Trend(signals=list(signals), missing=list(missing), window=window)


def test_render_flags_a_shift_by_name():
    s = evaluate(
        [BAND.center + 0.3 * BAND.scale] * 5, [BAND.center + 2.4 * BAND.scale] * 5
    )
    text = render(_trend([s]))
    assert "SUSTAINED SHIFT" in text
    assert "signal" in text and "probe @ target" in text
    assert "1 signal(s) show a sustained shift" in flat(text)


def test_render_with_no_shift_says_so_plainly():
    s = evaluate([BAND.center] * 5, [BAND.center] * 5)
    text = render(_trend([s]))
    assert "SUSTAINED SHIFT" not in text
    assert "No sustained shift" in text


def test_render_lists_missing_signals_by_name_with_a_count():
    m = MissingTrend(probe_id="p", target="t", signal="latency_ms", n=2)
    text = render(_trend(missing=[m]))
    assert "latency_ms" in text
    assert "2 run(s) recorded" in text
    assert f"need at least {MIN_RUNS}" in text


def test_render_empty_trend_explains_itself():
    text = render(_trend())
    assert "No baselines with recorded history" in text


def test_render_marks_thin_evidence_and_gives_the_caveat():
    s = evaluate([BAND.center] * 3, [BAND.center] * 3, total=5, window=5)
    text = render(_trend([s], window=5))
    assert "thin evidence" in text
    assert "anecdote" in flat(text)


def test_render_always_states_no_new_sampling():
    """The one misreading this output invites -- that a sustained shift was
    itself measured live -- refused in every rendering, mirroring how
    `calibrate` always states its own "headroom only" caveat.
    """
    s = evaluate([BAND.center] * 5, [BAND.center] * 5)
    text = flat(render(_trend([s])))
    assert "no probe was sampled" in text


def test_render_groups_multiple_signals_under_one_probe_label():
    s1 = evaluate([BAND.center] * 5, [BAND.center] * 5)
    s2 = SignalTrend(
        probe_id="probe", target="target", signal="other", n=10, window=5,
        early_median=1.0, recent_median=1.0, early_z=0.0, recent_z=0.0,
        threshold=1.8, thin=False,
    )
    text = render(_trend([s1, s2]))
    assert text.count("probe @ target") == 1


# --- Machine readable --------------------------------------------------------


def test_payload_round_trips_shift_and_missing():
    s = evaluate(
        [BAND.center + 0.3 * BAND.scale] * 5, [BAND.center + 2.4 * BAND.scale] * 5
    )
    m = MissingTrend(probe_id="p2", target="t2", signal="latency_ms", n=1)
    data = payload(_trend([s], [m], window=5))
    assert data["command"] == "trend"
    assert data["window"] == 5
    assert data["shifted"] == 1
    assert data["signals"][0]["shifted"] is True
    assert data["missing"][0] == {"probe": "p2", "target": "t2", "signal": "latency_ms", "n": 1}


def test_json_round_trips():
    s = evaluate([BAND.center] * 5, [BAND.center] * 5)
    data = json.loads(as_json(_trend([s])))
    assert data["signals"][0]["signal"] == "signal"
    assert data["signals"][0]["shifted"] is False
