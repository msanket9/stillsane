"""Threshold calibration from recorded runs.

`warn_k` and `drift_k` were the least defensible numbers in the tool: everything
else is measured from the probe, and those two were picked against constructed
scenarios. These tests cover the arithmetic that finally answers the question from
real runs, and the caveats that stop the answer being over-read.
"""

from __future__ import annotations

import json

import pytest

from stillsane.calibrate import THIN_EVIDENCE_RUNS, as_json, assess, payload, render


def rows(*specs):
    """(signal, z) pairs into the shape History returns, all on one probe."""
    return [("probe", "target", signal, z) for signal, z in specs]


def multi_rows(*specs):
    """(probe, signal, z) triples, for tests that need more than one probe."""
    return [(probe, "target", signal, z) for probe, signal, z in specs]


def flat(text: str) -> str:
    """Rendered prose is wrapped, so assert on it with the line breaks removed."""
    return " ".join(text.split())


def build(specs, clean_runs=20, warn_k=3.0, drift_k=6.0):
    return assess(
        rows(*specs), clean_runs=clean_runs, warn_k=warn_k, drift_k=drift_k,
        first="2026-08-04T00:00:00+00:00", last="2026-08-12T00:00:00+00:00",
    )


# --- The measurement -------------------------------------------------------


def test_headroom_is_the_gap_between_worst_clean_run_and_the_threshold():
    """The number the whole exercise was for.

    A worst clean run of |z|=1.5 against warn_k=3 means the tool came half way to
    crying wolf and no further.
    """
    cal = build([("semantic_distance", 0.5), ("semantic_distance", 1.5)])
    sig = cal.signals[0]
    assert sig.z_max_abs == 1.5
    assert sig.headroom(3.0) == pytest.approx(2.0)
    assert not sig.fires(3.0)


def test_sign_is_discarded_because_a_band_has_two_sides():
    """A length that came back short is as much a false alarm as one that ran long."""
    cal = build([("length_chars", -2.4), ("length_chars", 0.3)])
    assert cal.signals[0].z_max_abs == pytest.approx(2.4)


def test_a_threshold_that_already_fires_is_called_out():
    """|z| above warn_k on a run that passed overall is a false alarm, not a warning."""
    cal = build([("latency_ms", 4.2), ("semantic_distance", 0.4)])
    assert [s.signal for s in cal.false_alarms] == ["latency_ms"]
    text = flat(render(cal))
    assert "already fires on clean runs" in text
    assert "too tight" in text


def test_a_signal_that_never_moved_reports_no_headroom_rather_than_infinity():
    """Zero movement is an absence of evidence, not infinite margin."""
    cal = build([("json_shape_distance", 0.0), ("json_shape_distance", 0.0)])
    assert cal.signals[0].headroom(3.0) is None
    assert "never moved" in flat(render(cal))
    assert "absence of evidence" in flat(render(cal))


def test_tightest_k_is_the_worst_observed_value():
    cal = build([("length_chars", 1.51), ("semantic_distance", 0.68)])
    assert cal.tightest_k == pytest.approx(1.51)


def test_signals_are_ordered_worst_first():
    """The one closest to firing is the one worth reading."""
    cal = build([("a", 0.2), ("b", 2.1), ("c", 1.0)])
    assert [s.signal for s in cal.signals] == ["b", "c", "a"]


# --- Per probe, not per bare signal name ------------------------------------


def test_two_probes_sharing_a_signal_name_are_not_merged():
    """The bug this module shipped with: found by reading real calibration data.

    `extract_invoice` and `summarise_incident` both report `length_chars`, and in
    a real run `extract_invoice`'s sat at z=0.000 across 13 clean runs while
    `summarise_incident`'s reached 1.51. Aggregating by bare signal name pooled
    them into one row and diluted the one that actually mattered -- the report
    said "length_chars: headroom 2.0x" with no way to tell which probe that
    applied to, and pulled the p95 toward zero using a probe that never moved.
    """
    cal = assess(
        multi_rows(
            ("extract_invoice", "length_chars", 0.0),
            ("extract_invoice", "length_chars", 0.0),
            ("summarise_incident", "length_chars", 1.51),
            ("summarise_incident", "length_chars", 0.2),
        ),
        clean_runs=20, warn_k=3.0, drift_k=6.0,
    )
    rows_by_probe = {(s.probe_id, s.signal): s for s in cal.signals}
    assert rows_by_probe[("extract_invoice", "length_chars")].z_max_abs == 0.0
    assert rows_by_probe[("summarise_incident", "length_chars")].z_max_abs == pytest.approx(1.51)
    # Two distinct rows, not one merged row averaging the two probes together.
    assert len(cal.signals) == 2


def test_worst_signal_names_its_own_probe():
    """The summary line must say which probe to look at, not just which signal."""
    cal = assess(
        multi_rows(
            ("quiet_probe", "length_chars", 0.1),
            ("noisy_probe", "length_chars", 2.0),
        ),
        clean_runs=20, warn_k=3.0, drift_k=6.0,
    )
    assert cal.worst.probe_id == "noisy_probe"
    text = flat(render(cal))
    assert "noisy_probe @ target" in text


def test_by_probe_groups_and_orders_correctly():
    cal = assess(
        multi_rows(
            ("a", "s1", 0.5), ("a", "s2", 2.5),
            ("b", "s1", 0.1),
        ),
        clean_runs=20, warn_k=3.0, drift_k=6.0,
    )
    groups = cal.by_probe()
    labels = [label for label, _ in groups]
    # Probe "a" has the worse signal, so it sorts first.
    assert labels[0] == "a @ target"
    a_rows = dict(groups)["a @ target"]
    # Within a probe, worst signal first.
    assert [s.signal for s in a_rows] == ["s2", "s1"]


def test_render_shows_each_probe_as_its_own_section():
    cal = assess(
        multi_rows(("p1", "length_chars", 0.5), ("p2", "length_chars", 0.3)),
        clean_runs=20, warn_k=3.0, drift_k=6.0,
    )
    text = render(cal)
    assert "p1 @ target" in text
    assert "p2 @ target" in text
    # Both probes get their own table row for length_chars, not one shared row.
    # (A third mention is legitimate: the closing summary names the worst signal.)
    assert text.count("length_chars") >= 2


def test_false_alarms_are_scoped_to_the_probe_that_fired():
    """A false alarm on one probe must not implicate a quiet one sharing the name."""
    cal = assess(
        multi_rows(("flaky", "latency_ms", 4.5), ("steady", "latency_ms", 0.3)),
        clean_runs=20, warn_k=3.0, drift_k=6.0,
    )
    assert [(s.probe_id, s.signal) for s in cal.false_alarms] == [("flaky", "latency_ms")]
    text = flat(render(cal))
    assert "latency_ms on flaky @ target" in text


def test_payload_carries_probe_and_target_per_row():
    """A JSON consumer needs to know which probe a row belongs to, same as a reader."""
    cal = assess(
        multi_rows(("p1", "length_chars", 1.2)),
        clean_runs=20, warn_k=3.0, drift_k=6.0,
    )
    row = payload(cal)["signals"][0]
    assert row["probe"] == "p1"
    assert row["target"] == "target"


def test_payload_false_alarms_are_structured_not_bare_names():
    cal = assess(
        multi_rows(("flaky", "latency_ms", 4.5)),
        clean_runs=20, warn_k=3.0, drift_k=6.0,
    )
    alarm = payload(cal)["false_alarms"][0]
    assert alarm == {"probe": "flaky", "target": "target", "signal": "latency_ms"}


# --- The caveats, which are load-bearing -----------------------------------


def test_thin_evidence_is_stated_loudly_rather_than_footnoted():
    """Nine runs cannot show you a tail, and the tail is what a threshold is for."""
    cal = build([("semantic_distance", 0.5)], clean_runs=9)
    assert cal.thin
    text = flat(render(cal))
    assert "which is thin" in text
    assert "direction rather than a number" in text


def test_ample_evidence_drops_the_thin_warning():
    cal = build([("semantic_distance", 0.5)], clean_runs=THIN_EVIDENCE_RUNS + 5)
    assert not cal.thin
    assert "which is thin" not in flat(render(cal))


def test_the_false_alarm_only_caveat_is_always_present():
    """The tempting misreading, refused in every rendering.

    Clean runs contain no drift, so this says nothing about sensitivity. Someone
    loosening `k` because "there was loads of headroom" would be trading a visible
    problem for an invisible one.
    """
    for runs in (5, 500):
        text = flat(render(build([("semantic_distance", 0.5)], clean_runs=runs)))
        assert "headroom against false alarms only" in text
        assert "would catch a real regression" in text


def test_tightest_k_is_offered_as_a_floor_not_a_setting():
    cal = build([("length_chars", 1.51)])
    text = flat(render(cal))
    assert "floor, not a recommendation" in text


def test_no_clean_runs_explains_itself():
    cal = assess([], clean_runs=0, warn_k=3.0, drift_k=6.0)
    text = flat(render(cal))
    assert "nothing to calibrate against" in text
    assert cal.worst is None


# --- Machine readable ------------------------------------------------------


def test_payload_carries_the_caveat_as_data():
    """A consumer should be able to see the evidence is thin without parsing prose."""
    cal = build([("length_chars", 1.51)], clean_runs=9)
    data = payload(cal)
    assert data["thin_evidence"] is True
    assert data["clean_runs"] == 9
    assert data["tightest_warn_k"] == pytest.approx(1.51)
    assert data["false_alarms"] == []


def test_json_round_trips():
    data = json.loads(as_json(build([("semantic_distance", 0.68)])))
    assert data["command"] == "calibrate"
    assert data["signals"][0]["signal"] == "semantic_distance"
    assert data["signals"][0]["headroom"] == pytest.approx(4.412, abs=0.01)


def test_columns_do_not_run_together():
    """`never moved` is exactly as wide as the old column, so it collided.

    Caught by a clean-install smoke test printing `0.00never moved`. Cosmetic, but
    the table is the part of this output people actually read.
    """
    cal = build([("json_shape_distance", 0.0), ("length_chars", 1.2)])
    for line in render(cal).splitlines():
        assert "0.00never" not in line
        # Every data row keeps at least one space between the last two columns.
        if line.startswith("  ") and "never moved" in line:
            assert " never moved" in line
