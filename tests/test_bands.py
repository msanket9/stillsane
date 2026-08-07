"""Band inspection.

The case that matters here is the one that is invisible everywhere else: a band
that is well formed, looks like every other band, and is wrong. These tests pin
the shape of data that produces it, because it was found in the wild against a
real provider and fixtures had never generated it.
"""

from __future__ import annotations

import json

import pytest
from conftest import sample

from stillsane.bands import (
    FALSE_ALARM_FLOOR,
    Finding,
    as_json,
    inspect,
    no_embedder,
    payload,
    render,
)
from stillsane.compare import BandConfig
from stillsane.signals import build_signals
from stillsane.store.baseline import Baseline

CFG = BandConfig()

#: Bimodal, with a minority big enough for the IQR to measure once the MAD has
#: collapsed. The estimator handles this shape, so it is no longer a finding.
MEASURABLE_BIMODAL = [0.0] * 16 + [0.128] * 12

#: Bimodal, with a minority small enough that the IQR collapses too. Roughly one
#: sample in five reformatting, which puts both quartiles inside the majority mode.
#: This is what a genuine collapse now looks like: no robust estimator can see it.
COLLAPSED_BIMODAL = [0.0] * 22 + [0.128] * 6


def baseline_with(pooled=None, samples=None) -> Baseline:
    return Baseline(
        target_name="prod",
        probe_id="probe",
        version=1,
        created="2026-08-04T00:00:00+00:00",
        config_hash="deadbeef",
        samples=samples or [],
        pooled=pooled or {},
    )


def find(report, name):
    for sb in report.signals:
        if sb.signal == name:
            return sb
    return None


# --- The collapse ---------------------------------------------------------


def test_measurable_bimodal_band_is_no_longer_collapsed(signals_for):
    """The estimator fix, asserted from the outside.

    Median 0 and MAD 0, but the minority mode is large enough that the IQR still
    measures it. This used to floor the band at 0.02 with 12 of 28 pairs outside
    it, which made a clean run report drift. It now resolves to a real scale and
    admits its own baseline.
    """
    pooled = {"semantic_distance": MEASURABLE_BIMODAL}
    sd = find(inspect(baseline_with(pooled), signals_for(), CFG), "semantic_distance")
    assert sd.raw_scale > 0
    assert not sd.band.floored
    assert sd.outside == 0
    assert sd.finding is None


def test_bimodal_pairwise_band_is_flagged_collapsed(signals_for):
    """A genuine collapse: both the MAD and the IQR report zero.

    With the minority down to roughly one sample in five, both quartiles sit inside
    the majority mode and no robust estimator can see the spread. The band lands on
    the floor while the baseline's own pairs sit outside it, which is the shape
    worth reporting now that the measurable case is handled.
    """
    report = inspect(baseline_with({"semantic_distance": COLLAPSED_BIMODAL}), signals_for(), CFG)

    sd = find(report, "semantic_distance")
    assert sd.finding is Finding.COLLAPSED
    assert sd.raw_scale == 0.0
    assert sd.outside == 6
    assert sd.observed_max == pytest.approx(0.128)
    assert report.suspect


def test_collapsed_band_sits_below_its_own_baseline(signals_for):
    """The defect in one assertion: the band excludes data it was built from."""
    pooled = {"semantic_distance": COLLAPSED_BIMODAL}
    sd = find(inspect(baseline_with(pooled), signals_for(), CFG), "semantic_distance")
    assert sd.band.upper < sd.observed_max


def test_collapse_survives_more_samples(signals_for):
    """More samples do not rescue it while one mode dominates.

    Worth pinning because "raise baseline_samples" is the advice the capture-time
    warning gives, and for this shape it is the wrong advice.
    """
    pooled = {"semantic_distance": [0.0] * 400 + [0.128] * 100}
    sd = find(inspect(baseline_with(pooled), signals_for(), CFG), "semantic_distance")
    assert sd.finding is Finding.COLLAPSED


# --- The healthy cases ----------------------------------------------------


def test_measured_band_has_no_finding(signals_for):
    """Real spread, no floor, nothing outside: the band is a measurement."""
    pooled = {"semantic_distance": [0.05, 0.07, 0.06, 0.08, 0.055, 0.075, 0.065]}
    sd = find(inspect(baseline_with(pooled), signals_for(), CFG), "semantic_distance")
    assert sd.finding is None
    assert sd.outside == 0
    assert sd.raw_scale > 0


def test_identical_samples_are_floored_but_stable(signals_for):
    """Floored with nothing outside is not a fault, and must not read as one."""
    pooled = {"semantic_distance": [0.0] * 10}
    report = inspect(baseline_with(pooled), signals_for(), CFG)
    sd = find(report, "semantic_distance")
    assert sd.finding is Finding.FLOORED_STABLE
    assert sd.band.floored
    assert not report.suspect  # informational only


def test_heavy_tail_is_self_outside_not_collapsed(signals_for):
    """A real scale with an escaping tail is a different diagnosis."""
    pooled = {"semantic_distance": [0.05, 0.06, 0.07, 0.055, 0.065, 0.06, 0.9]}
    sd = find(inspect(baseline_with(pooled), signals_for(), CFG), "semantic_distance")
    assert sd.raw_scale > 0
    assert sd.outside >= 1
    assert sd.finding is Finding.SELF_OUTSIDE


# --- False alarm rate, the number that actually matters --------------------


def test_wide_draw_absorbs_a_tail_that_the_pair_count_panics_about(signals_for):
    """The bug this fixes, from a real baseline.

    An essay probe had 15% of its baseline pairs outside the band and was reported
    as about to misreport. It was not: a pairwise check takes the median of two
    dozen cross distances, and a tail that size never moves a median of 24 far
    enough to fire. Counting individual pairs answered a question nobody asked.
    """
    pooled = {"semantic_distance": [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.30, 0.31]}
    report = inspect(baseline_with(pooled, [sample("x") for _ in range(8)]),
                     signals_for(), CFG, check_samples=3)
    sd = find(report, "semantic_distance")
    assert sd.outside > 0                             # the tail is real
    assert sd.draw == 24                              # but a check medians 24 of them
    assert sd.false_alarm_pct < FALSE_ALARM_FLOOR     # so it almost never fires
    assert sd.finding is None                         # and is therefore not a fault
    assert not report.suspect


def test_narrow_draw_does_not_absorb_the_same_tail(signals_for):
    """Same spread, three values instead of twenty-four, and now it fires.

    This is why the draw size is reported alongside the rate: they are the same
    distribution and opposite verdicts.
    """
    samples = [sample("x" * n) for n in (40, 41, 42, 43, 44, 45, 120, 130)]
    report = inspect(baseline_with(samples=samples), signals_for(), CFG, check_samples=3)
    lc = find(report, "length_chars")
    assert lc.draw == 3
    assert lc.false_alarm_pct > 0


def test_false_alarm_estimate_is_deterministic(signals_for):
    """A number that changes each run cannot be quoted in a bug report."""
    pooled = {"semantic_distance": [0.05, 0.06, 0.07, 0.055, 0.065, 0.06, 0.9]}
    first = find(inspect(baseline_with(pooled), signals_for(), CFG), "semantic_distance")
    second = find(inspect(baseline_with(pooled), signals_for(), CFG), "semantic_distance")
    assert first.false_alarm_pct == second.false_alarm_pct


def test_collapsed_band_reports_regardless_of_the_estimate(signals_for):
    """A band that was never measured makes no claim the rate could vindicate."""
    sd = find(inspect(baseline_with({"semantic_distance": COLLAPSED_BIMODAL}),
                      signals_for(), CFG), "semantic_distance")
    assert sd.finding is Finding.COLLAPSED


def test_false_alarm_needs_something_to_resample(signals_for):
    """One value is not a distribution, so decline rather than invent a rate."""
    sd = find(inspect(baseline_with({"semantic_distance": [0.05]}), signals_for(), CFG),
              "semantic_distance")
    assert sd.false_alarm_pct is None


# --- Pointwise coverage, the gap this closes ------------------------------


def test_pointwise_signals_are_inspected(signals_for):
    """`length_chars` and friends were invisible to the capture-time warning.

    They are not pooled, so the only place they could ever be reported is here,
    recomputed from the stored samples.
    """
    samples = [sample("x" * 40) for _ in range(6)]
    report = inspect(baseline_with(samples=samples), signals_for(), CFG)
    assert find(report, "length_chars") is not None


def test_bimodal_pointwise_band_is_flagged(signals_for):
    """A length band breached from below, which the estimator fix does not rescue.

    Found against a real provider: band 60..76 built from samples spanning 56..68.
    The IQR fallback gives it a real scale, so it is no longer a collapse. But
    `3 * scale` is still under the absolute floor of 8, so the floor governs, the
    band is unchanged, and two samples remain outside it. The honest diagnosis is
    therefore `SELF_OUTSIDE`: the scale is measured, the band is simply tighter
    than this probe's own behaviour warrants.

    That distinction is the point. No dispersion estimate bridges two modes 12
    apart from 8 samples, so this one is a fact about the probe rather than about
    the estimator. The direction also matters, which is why the diagnosis reports
    the observed range rather than its maximum.
    """
    samples = [sample("x" * 68) for _ in range(6)] + [sample("x" * 56) for _ in range(2)]
    lc = find(inspect(baseline_with(samples=samples), signals_for(), CFG), "length_chars")
    assert lc.finding is Finding.SELF_OUTSIDE
    assert lc.raw_scale > 0
    assert lc.observed_min < lc.band.lower
    assert lc.outside == 2


def test_inapplicable_signals_are_skipped(signals_for):
    """No token counts reported means nothing to show, not an error.

    Anthropic names its token fields differently, so these come back None. The
    documented behaviour is silence.
    """
    samples = [sample("hello", completion_tokens=None) for _ in range(5)]
    report = inspect(baseline_with(samples=samples), signals_for(), CFG)
    assert find(report, "completion_tokens") is None


def test_categorical_signals_have_no_band(signals_for):
    """Any change is an event, so there is no normal range to inspect."""
    samples = [sample("hi", fingerprint="fp_a") for _ in range(5)]
    report = inspect(baseline_with(samples=samples), signals_for(), CFG)
    assert find(report, "fingerprint") is None


def test_failed_samples_are_excluded(signals_for):
    """A transport error is not evidence about how much the probe varies."""
    samples = [sample("x" * 40) for _ in range(5)] + [sample("", error="boom")]
    lc = find(inspect(baseline_with(samples=samples), signals_for(), CFG), "length_chars")
    assert lc.n == 5


# --- Rendering and the offline guarantee ----------------------------------


def test_render_names_the_broken_band(signals_for):
    pooled = {"semantic_distance": COLLAPSED_BIMODAL}
    text = render([inspect(baseline_with(pooled), signals_for(), CFG)])
    assert "COLLAPSED" in text
    assert "semantic_distance" in text
    assert "will misreport" in text


def test_render_stays_quiet_when_sound(signals_for):
    pooled = {"semantic_distance": [0.05, 0.07, 0.06, 0.08, 0.055, 0.075]}
    text = render([inspect(baseline_with(pooled), signals_for(), CFG)])
    assert "All bands look sound." in text
    assert "COLLAPSED" not in text


def test_verbose_shows_sound_bands(signals_for):
    pooled = {"semantic_distance": [0.05, 0.07, 0.06, 0.08, 0.055, 0.075]}
    report = inspect(baseline_with(pooled), signals_for(), CFG)
    assert "semantic_distance" not in render([report])
    assert "semantic_distance" in render([report], verbose=True)


def test_floored_stable_is_named_but_not_explained(signals_for):
    """Which bands were defaulted is the thing the capture warning could not say.

    So it has to appear. But it is not a fault, and giving each one a paragraph
    turns a clean baseline into a wall of text that concludes nothing is wrong.
    """
    pooled = {"semantic_distance": [0.0] * 10}
    text = render([inspect(baseline_with(pooled), signals_for(), CFG)])
    assert "defaulted to a floor, samples agree: semantic_distance" in text
    assert "FLOORED-STABLE" not in text
    assert "All bands look sound." in text


def test_floored_stable_is_explained_when_verbose(signals_for):
    pooled = {"semantic_distance": [0.0] * 10}
    text = render([inspect(baseline_with(pooled), signals_for(), CFG)], verbose=True)
    assert "FLOORED-STABLE" in text


def test_json_reports_every_band_not_only_the_suspect_ones(signals_for):
    """The human report hides sound bands; the machine one must not.

    Filtering is a readability concern. A consumer diffing bands across runs needs
    the ones that did not move too, or it cannot tell "still sound" from "no longer
    reported".
    """
    pooled = {
        "semantic_distance": COLLAPSED_BIMODAL,
        "json_shape_distance": [0.0] * 28,
    }
    samples = [sample("x" * 40) for _ in range(5)]
    data = payload([inspect(baseline_with(pooled, samples), signals_for(), CFG)])

    bands = {b["signal"]: b for b in data["probes"][0]["bands"]}
    assert bands["semantic_distance"]["suspect"] is True
    assert bands["json_shape_distance"]["suspect"] is False  # sound, still present
    assert data["suspect"] == 1


def test_json_keeps_finding_key_on_sound_bands(signals_for):
    """Null rather than absent, so the key set is stable across rows."""
    pooled = {"semantic_distance": [0.05, 0.07, 0.06, 0.08, 0.055, 0.075]}
    data = payload([inspect(baseline_with(pooled), signals_for(), CFG)])
    for band in data["probes"][0]["bands"]:
        assert "finding" in band
        assert band["finding"] is None
        assert band["suspect"] is False


def test_json_round_trips(signals_for):
    pooled = {"semantic_distance": COLLAPSED_BIMODAL}
    report = inspect(baseline_with(pooled), signals_for(), CFG)
    data = json.loads(as_json([report]))
    sd = next(b for b in data["probes"][0]["bands"] if b["signal"] == "semantic_distance")
    assert sd["finding"] == "collapsed"
    assert sd["outside"] == 6
    assert sd["n"] == 28
    assert sd["outside_pct"] == pytest.approx(21.43, abs=0.01)
    assert sd["band"]["floored"] is True


def test_inspection_never_embeds():
    """The command must stay free and offline.

    Every pairwise number is read from the stored pool, so a code path that
    reaches for the embedder is a bug. The placeholder makes it fail loudly rather
    than quietly downloading a model.
    """
    signals = build_signals(None, no_embedder())
    pooled = {"semantic_distance": COLLAPSED_BIMODAL}
    samples = [sample("x" * 40) for _ in range(5)]
    report = inspect(baseline_with(pooled, samples), signals, CFG)
    assert find(report, "semantic_distance").finding is Finding.COLLAPSED


def test_embedder_placeholder_raises_if_used():
    with pytest.raises(AssertionError):
        no_embedder().encode(["anything"])


def test_empty_baseline_produces_no_bands(signals_for):
    report = inspect(baseline_with(), signals_for(), CFG)
    assert report.signals == []
    assert "no bands" in render([report])
