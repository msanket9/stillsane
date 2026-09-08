"""The five scenarios that decide whether this tool is worth installing.

If any of these regress, stillsane is either blind or a nuisance. They are the
real acceptance criteria; everything else in the suite is supporting detail.
"""

from __future__ import annotations

from conftest import (
    CHATTY_BASELINE,
    CHATTY_CURRENT,
    CHATTY_DRIFTED,
    PROSE_WRAPPED_JSON,
    STABLE_JSON,
    sample,
)

from stillsane.compare import BandConfig, compare_probe
from stillsane.models import Level, ToolCall

CHECKS = ["valid_json", {"has_keys": ["total", "due_date"]}]


def run(signals_for, baseline_texts, current_texts, checks=None, **kw):
    return compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(checks),
        baseline=[sample(t, **kw) for t in baseline_texts],
        current=[sample(t, **kw) for t in current_texts],
        cfg=BandConfig(),
    )


# --- 1. No drift ----------------------------------------------------------


def test_stable_probe_stays_quiet(signals_for):
    """Same behaviour, slightly different bytes. Must not alert."""
    verdict = run(signals_for, STABLE_JSON, STABLE_JSON[:3], CHECKS)
    assert verdict.level is Level.PASS, [
        (s.signal, s.level, s.detail) for s in verdict.moved
    ]


# --- 2. Real drift --------------------------------------------------------


def test_prose_wrapping_is_caught(signals_for):
    """Identical data, but the response stopped being parseable JSON."""
    verdict = run(signals_for, STABLE_JSON, PROSE_WRAPPED_JSON, CHECKS)
    assert verdict.level is Level.DRIFT

    moved = {s.signal: s for s in verdict.moved}
    assert "valid_json" in moved
    assert moved["valid_json"].level is Level.DRIFT

    # The data itself is still present -- the report should be able to say the
    # content survived and only the envelope broke. That distinction is the
    # difference between a useful alert and a scary one.
    has_keys = next(s for s in verdict.signals if s.signal.startswith("has_keys"))
    assert has_keys.level is Level.PASS


def test_drift_report_carries_evidence(signals_for):
    verdict = run(signals_for, STABLE_JSON, PROSE_WRAPPED_JSON, CHECKS)
    assert verdict.baseline_excerpt and verdict.observed_excerpt
    assert "Here is the extracted" in verdict.observed_excerpt or (
        "Sure!" in verdict.observed_excerpt or "Of course." in verdict.observed_excerpt
    )
    # Length moved a long way and should be reported alongside the failure.
    length = next(s for s in verdict.signals if s.signal == "length_chars")
    assert length.observed > length.baseline


# --- 3. High-variance probe stays quiet (the one that matters most) -------


def test_chatty_probe_does_not_cry_wolf(signals_for):
    """A probe that legitimately rewords itself every call must not alert.

    This is the failure mode that gets drift tools uninstalled in week one. A
    fixed similarity threshold cannot pass this test and catch the next one.
    """
    verdict = run(signals_for, CHATTY_BASELINE, CHATTY_CURRENT)
    assert verdict.level is Level.PASS, [
        (s.signal, s.level, s.detail) for s in verdict.moved
    ]


def test_chatty_probe_still_catches_real_drift(signals_for):
    """The wide band must not be so wide that it stops detecting anything.

    The semantic signal specifically must fire. Letting `length_chars` carry this
    would be luck: a topic change that happened to preserve length would slip
    through, and semantic distance is the signal that exists to catch exactly this.
    """
    verdict = run(signals_for, CHATTY_BASELINE, CHATTY_DRIFTED)
    assert verdict.level in (Level.WARN, Level.DRIFT)
    semantic = next(s for s in verdict.signals if s.signal == "semantic_distance")
    assert semantic.level is not Level.PASS
    assert semantic.p_value is not None and semantic.p_value < 0.01


def test_band_widths_reflect_probe_character(signals_for):
    """The stable probe's band must be tighter than the chatty probe's.

    This is the property that makes one tool work for both, and it is worth
    asserting directly rather than inferring it from the pass/fail results above.
    """
    stable = run(signals_for, STABLE_JSON, STABLE_JSON[:3])
    chatty = run(signals_for, CHATTY_BASELINE, CHATTY_CURRENT)

    stable_band = next(s for s in stable.signals if s.signal == "semantic_distance").band
    chatty_band = next(s for s in chatty.signals if s.signal == "semantic_distance").band
    assert stable_band.upper < chatty_band.upper


# --- 4. Zero-variance probe ----------------------------------------------


def test_identical_baseline_tolerates_whitespace(signals_for):
    """A byte-identical baseline must still absorb trivia rather than alert on it.

    The band here comes from the under-sampled rescue, which rebuilds it from the
    current run's own spread. That spread is itself tie-heavy -- two distinct texts
    repeated -- so it used to collapse the MAD a second time and land back on the
    floor. It now resolves to a measured scale via the IQR fallback, which is what
    the rescue was always trying to achieve.

    Asserted as "the band has a real scale and the whitespace passes" rather than
    "the band is floored". Floored-ness was the old mechanism, not the property
    worth protecting: a band that absorbs whitespace because it measured this
    probe's variation is strictly better than one that absorbs it by default.
    """
    identical = ['{"status": "ok", "count": 3}'] * 5
    reformatted = ['{"status": "ok", "count": 3} ', '{"status": "ok", "count": 3}\n'] * 2
    verdict = run(signals_for, identical, reformatted, ["valid_json"])
    assert verdict.level is Level.PASS, [
        (s.signal, s.level, s.detail) for s in verdict.moved
    ]

    band = next(s for s in verdict.signals if s.signal == "semantic_distance").band
    assert band.scale > 0, "the band must have some width, however it was arrived at"
    assert band.upper >= signals_for()[0].floor, "and must absorb at least the floor's worth"


def test_under_sampled_baseline_does_not_cry_wolf(signals_for):
    """A probe that varies a little can draw N identical samples at baseline.

    Found by running the real CLI against a local server: five draws from three
    near-identical phrasings came back byte-identical, the band collapsed to the
    floor, and the next ordinary check reported drift. The current run's own
    internal spread is the evidence that the baseline was simply under-sampled.
    """
    unlucky = ['{"due_date": "2026-07-01", "total": 1240.50}'] * 5
    ordinary = [
        '{"total": 1240.50, "due_date": "2026-07-01"}',
        '{"total": 1240.5, "due_date": "2026-07-01"}',
        '{"due_date": "2026-07-01", "total": 1240.50}',
    ]
    verdict = run(signals_for, unlucky, ordinary, ["valid_json"])
    assert verdict.level is Level.PASS, [
        (s.signal, s.level, s.detail) for s in verdict.moved
    ]

    semantic = next(s for s in verdict.signals if s.signal == "semantic_distance")
    assert "under-sampled" in semantic.detail


def test_the_rescue_is_not_a_blanket_amnesty(signals_for):
    """The complement, and the reason the rescue is safe.

    Same degenerate baseline, but the current run is internally consistent *and*
    says something different. Within-run spread stays near zero, so the band stays
    floored and the drift is still caught.
    """
    unlucky = ['{"due_date": "2026-07-01", "total": 1240.50}'] * 5
    moved = ['{"status": "unable to parse invoice", "total": null}'] * 3
    verdict = run(signals_for, unlucky, moved, ["valid_json"])
    assert verdict.level is Level.DRIFT


def test_identical_baseline_still_catches_content_change(signals_for):
    identical = ['{"status": "ok", "count": 3}'] * 5
    changed = ['{"status": "degraded", "count": 0, "reason": "upstream timeout"}'] * 3
    verdict = run(signals_for, identical, changed, ["valid_json"])
    assert verdict.level is Level.DRIFT


def test_a_baseline_that_was_already_flaky_is_not_held_to_perfection(signals_for):
    """`strict_when_perfect` used to key on the baseline's *median*, so a 4-of-5
    baseline -- median 1.0, but not actually perfect -- was treated as if it had
    never failed. A check with the same one-in-a-few failure rate then read as
    "was passing every baseline sample, now failing", which is false: the
    baseline was already flaky, and `signals/structural.py`'s own stated design
    is that a band should accommodate that rather than page on it.
    """
    baseline = ['{"total": 1}', '{"total": 1}', '{"total": 1}', '{"total": 1}', "not json"]
    current = ['{"total": 1}', '{"total": 1}', "not json"]
    verdict = run(signals_for, baseline, current, ["valid_json"])

    valid_json = next(s for s in verdict.signals if s.signal == "valid_json")
    assert valid_json.level is not Level.DRIFT
    assert "every baseline sample" not in valid_json.detail


def test_a_baseline_that_never_truncates_catches_one_that_does(signals_for):
    """`response_complete`, the same shape of proof as the one above for
    `valid_json`: this is not a statistical question, so a baseline of five
    complete responses must not need a variance band to catch the first
    truncated one.

    Grounded in a real miss: an essay probe run for weeks truncated on 7 of 8
    samples and nothing in the tool said so, because length and semantic
    distance alone do not reliably distinguish "a longer, complete answer" from
    "a longer answer that got cut off partway through."
    """
    baseline = [sample("A steady answer that always finishes.", finish_reason="stop")] * 5
    current = [sample("A steady answer that always fini", finish_reason="length")] * 3
    verdict = compare_probe(
        probe_id="essay",
        target_name="prod",
        signals=signals_for(),
        baseline=baseline,
        current=current,
        cfg=BandConfig(),
    )
    assert verdict.level is Level.DRIFT
    moved = {s.signal for s in verdict.moved}
    assert "response_complete" in moved


def test_a_baseline_that_truncated_once_is_not_held_to_perfection(signals_for):
    """Same shape as the `valid_json` case above: a baseline that truncated once
    in five has a `response_complete` median of 1.0 but was not perfect, and
    must not be reported as "never truncated" against a check with the same
    rate of truncation.
    """
    baseline = [sample("done.", finish_reason="stop")] * 4 + [
        sample("cut of", finish_reason="length")
    ]
    current = [sample("done.", finish_reason="stop")] * 2 + [
        sample("cut of", finish_reason="length")
    ]
    verdict = compare_probe(
        probe_id="essay",
        target_name="prod",
        signals=signals_for(),
        baseline=baseline,
        current=current,
        cfg=BandConfig(),
    )
    rc = next(s for s in verdict.signals if s.signal == "response_complete")
    assert rc.level is not Level.DRIFT
    assert "every baseline sample" not in rc.detail


# --- 5. Fingerprint change -----------------------------------------------


def test_fingerprint_change_warns_by_default(signals_for):
    """The backend model moved under a stable version string. Information, not a fault."""
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(["valid_json"]),
        baseline=[sample(t, fingerprint="fp_a4f2b1") for t in STABLE_JSON],
        current=[sample(t, fingerprint="fp_9c3e88") for t in STABLE_JSON[:3]],
    )
    assert verdict.level is Level.WARN
    fp = next(s for s in verdict.signals if s.signal == "fingerprint")
    assert fp.level is Level.WARN
    assert "fp_a4f2b1 -> fp_9c3e88" in fp.detail


def test_fingerprint_change_can_be_escalated(signals_for):
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(["valid_json"]),
        baseline=[sample(t, fingerprint="fp_a4f2b1") for t in STABLE_JSON],
        current=[sample(t, fingerprint="fp_9c3e88") for t in STABLE_JSON[:3]],
        escalate_fingerprint=True,
    )
    assert verdict.level is Level.DRIFT


def test_stable_fingerprint_is_silent(signals_for):
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(["valid_json"]),
        baseline=[sample(t, fingerprint="fp_a4f2b1") for t in STABLE_JSON],
        current=[sample(t, fingerprint="fp_a4f2b1") for t in STABLE_JSON[:3]],
    )
    assert verdict.level is Level.PASS


def test_a_subset_of_baseline_fingerprints_is_not_drift(signals_for):
    """A provider that load-balances across backend builds returns several
    `system_fingerprint` values across baseline calls. A short check run that
    happens to draw only one of them measured the same population less, not a
    change -- it must not report drift for a fingerprint the baseline already
    saw.
    """
    baseline_fps = ["fp_a4f2b1", "fp_9c3e88", "fp_a4f2b1", "fp_9c3e88", "fp_a4f2b1"]
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(["valid_json"]),
        baseline=[sample(t, fingerprint=fp) for t, fp in zip(STABLE_JSON, baseline_fps, strict=True)],
        current=[sample(t, fingerprint="fp_a4f2b1") for t in STABLE_JSON[:3]],
    )
    assert verdict.level is Level.PASS
    fp = next(s for s in verdict.signals if s.signal == "fingerprint")
    assert fp.level is Level.PASS


def test_a_genuinely_new_fingerprint_still_warns_and_names_only_itself(signals_for):
    """The counterpart to the subset case above: a value the baseline never saw
    at all is real drift, and the report should name only what is new -- not
    the baseline's own fingerprints re-echoed back as if they too were new.
    """
    baseline_fps = ["fp_a4f2b1", "fp_9c3e88", "fp_a4f2b1", "fp_9c3e88", "fp_a4f2b1"]
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(["valid_json"]),
        baseline=[sample(t, fingerprint=fp) for t, fp in zip(STABLE_JSON, baseline_fps, strict=True)],
        current=[sample(t, fingerprint="fp_c00000") for t in STABLE_JSON[:3]],
    )
    assert verdict.level is Level.WARN
    fp = next(s for s in verdict.signals if s.signal == "fingerprint")
    assert fp.level is Level.WARN
    assert "fp_c00000" in fp.detail
    assert "fp_a4f2b1" not in fp.detail.split(" -> ")[1]
    assert "fp_9c3e88" not in fp.detail.split(" -> ")[1]


# --- Agent-shaped drift ---------------------------------------------------


def test_agent_stopping_a_tool_call_is_drift(signals_for):
    """For an agent this matters more than any amount of text similarity."""
    calls = [ToolCall("lookup_invoice", ("invoice_id",)), ToolCall("fetch_total", ("id",))]
    verdict = compare_probe(
        probe_id="agent",
        target_name="prod",
        signals=signals_for(None),
        baseline=[sample("Looking that up for you.", tool_calls=list(calls)) for _ in range(5)],
        current=[
            sample("Looking that up for you.", tool_calls=[calls[0]]) for _ in range(3)
        ],
    )
    tool = next(s for s in verdict.signals if s.signal == "tool_call_distance")
    assert tool.level is Level.DRIFT
    assert verdict.level is Level.DRIFT


def test_argument_shape_change_is_drift(signals_for):
    before = [ToolCall("search", ("query", "limit"))]
    after = [ToolCall("search", ("q", "limit"))]
    verdict = compare_probe(
        probe_id="agent",
        target_name="prod",
        signals=signals_for(None),
        baseline=[sample("searching", tool_calls=list(before)) for _ in range(5)],
        current=[sample("searching", tool_calls=list(after)) for _ in range(3)],
    )
    tool = next(s for s in verdict.signals if s.signal == "tool_call_distance")
    assert tool.level is Level.DRIFT


def test_non_agent_probe_skips_tool_signal(signals_for):
    verdict = run(signals_for, STABLE_JSON, STABLE_JSON[:3])
    assert not any(s.signal == "tool_call_distance" for s in verdict.signals)


# --- Errors are not drift -------------------------------------------------


def test_dead_endpoint_is_error_not_drift(signals_for):
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(CHECKS),
        baseline=[sample(t) for t in STABLE_JSON],
        current=[sample("", error="HTTP 503 Service Unavailable") for _ in range(3)],
    )
    assert verdict.level is Level.ERROR
    assert "503" in verdict.signals[0].detail


def test_partial_failure_warns_but_still_scores(signals_for):
    current = [sample(t) for t in STABLE_JSON[:2]] + [sample("", error="timeout")]
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(CHECKS),
        baseline=[sample(t) for t in STABLE_JSON],
        current=current,
    )
    assert verdict.level is Level.WARN
    transport = next(s for s in verdict.signals if s.signal == "transport")
    assert "1/3" in transport.detail
    # The surviving samples were still compared rather than discarded.
    assert any(s.signal == "semantic_distance" for s in verdict.signals)


# --- Confidence gating ----------------------------------------------------


def test_thin_baseline_cannot_fail_a_build(signals_for):
    """Two baseline samples is not enough evidence to break someone's CI."""
    verdict = compare_probe(
        probe_id="extract_invoice",
        target_name="prod",
        signals=signals_for(None),
        baseline=[sample(t) for t in CHATTY_BASELINE[:2]],
        current=[sample(t) for t in CHATTY_DRIFTED],
        cfg=BandConfig(),
    )
    assert verdict.level is Level.WARN
    semantic = next(s for s in verdict.signals if s.signal == "semantic_distance")
    assert "capped to warn" in semantic.detail
