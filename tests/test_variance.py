"""Band arithmetic.

The invariant worth protecting above all others is that `z` and the band agree.
The report shows both, and a user who sees "z=4.1" next to "band <=0.09, observed
0.07" will stop trusting the tool immediately -- correctly, because one of the two
would be lying.
"""

from __future__ import annotations

import math

import pytest

from stillsane.compare import (
    BandConfig,
    evaluate_pairwise,
    mann_whitney_p,
    robust_band,
    robust_centre_scale,
    z_score,
)
from stillsane.models import Direction, Level, Sample
from stillsane.signals.base import PairwiseSignal

CFG = BandConfig()


def test_z_and_band_agree():
    """z > warn_k if and only if the value sits outside the band."""
    values = [0.10, 0.12, 0.09, 0.11, 0.13, 0.10]
    band = robust_band(values, direction=Direction.UP_IS_BAD, cfg=CFG)
    for probe in [0.0, 0.05, 0.1, band.upper - 1e-9, band.upper + 1e-9, 0.5, 1.0]:
        z = z_score(probe, band, Direction.UP_IS_BAD)
        assert (z > CFG.warn_k) == (probe > band.upper), (probe, z, band.upper)


def test_z_and_band_agree_when_floored():
    """The invariant must survive the floor widening the band."""
    band = robust_band([0.2] * 6, direction=Direction.UP_IS_BAD, cfg=CFG, floor=0.05)
    assert band.floored
    for probe in [0.2, 0.24, band.upper - 1e-9, band.upper + 1e-9, 0.9]:
        z = z_score(probe, band, Direction.UP_IS_BAD)
        assert (z > CFG.warn_k) == (probe > band.upper), (probe, z, band.upper)


def test_constant_baseline_falls_back_to_floor():
    band = robust_band([0.5] * 5, direction=Direction.UP_IS_BAD, cfg=CFG, floor=0.02)
    assert band.floored
    assert band.upper == pytest.approx(0.52)
    assert band.scale == pytest.approx(0.02 / CFG.warn_k)


def test_relative_floor_scales_with_magnitude():
    """A 5% relative floor on a 400-token baseline is 20 tokens, not 20 anything-else."""
    band = robust_band(
        [400, 400, 401, 399, 400], direction=Direction.BOTH, cfg=CFG, rel_floor=0.05
    )
    assert band.upper == pytest.approx(420)
    assert band.lower == pytest.approx(380)


def test_mad_ignores_a_single_outlier():
    """One weird sample must not widen the band. This is why it is not a stdev."""
    clean = [0.10, 0.11, 0.09, 0.10, 0.12]
    with_outlier = clean + [0.95]

    _, clean_scale = robust_centre_scale(clean)
    _, outlier_scale = robust_centre_scale(with_outlier)
    assert outlier_scale == pytest.approx(clean_scale, rel=0.5)

    # A stdev would roughly triple here; assert the contrast explicitly so the
    # reason for the design choice stays visible.
    import statistics

    assert statistics.pstdev(with_outlier) > 3 * statistics.pstdev(clean)


def test_direction_controls_which_bound_exists():
    values = [10, 11, 9, 10, 12]
    up = robust_band(values, direction=Direction.UP_IS_BAD, cfg=CFG)
    down = robust_band(values, direction=Direction.DOWN_IS_BAD, cfg=CFG)
    both = robust_band(values, direction=Direction.BOTH, cfg=CFG)

    assert up.lower is None and up.upper is not None
    assert down.upper is None and down.lower is not None
    assert both.lower is not None and both.upper is not None


def test_one_sided_z_ignores_the_safe_direction():
    band = robust_band([10, 11, 9, 10, 12], direction=Direction.UP_IS_BAD, cfg=CFG)
    assert z_score(2, band, Direction.UP_IS_BAD) == 0.0  # much faster: not a fault
    assert z_score(40, band, Direction.UP_IS_BAD) > 0


def test_band_override_replaces_the_learned_band():
    band = robust_band(
        [0.1, 0.11, 0.09], direction=Direction.UP_IS_BAD, cfg=CFG, override=0.25
    )
    assert band.upper == 0.25
    assert not band.floored
    assert z_score(0.26, band, Direction.UP_IS_BAD) > CFG.warn_k
    assert z_score(0.24, band, Direction.UP_IS_BAD) < CFG.warn_k


# --- Mann-Whitney ---------------------------------------------------------


def test_mann_whitney_is_neutral_on_identical_distributions():
    a = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]
    b = [0.11, 0.10, 0.12, 0.09, 0.10, 0.11]
    p = mann_whitney_p(a, b)
    assert 0.2 < p < 0.8


def test_mann_whitney_detects_a_clear_shift():
    within = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]
    cross = [0.40, 0.42, 0.38, 0.41, 0.39, 0.43]
    assert mann_whitney_p(within, cross) < 0.01


def test_mann_whitney_is_one_sided():
    """Only an upward shift counts. Outputs converging is not drift."""
    within = [0.40, 0.42, 0.38, 0.41, 0.39, 0.43]
    cross = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]
    assert mann_whitney_p(within, cross) > 0.99


def test_mann_whitney_declines_when_underpowered():
    assert mann_whitney_p([0.1, 0.2], [0.3, 0.4]) is None


def test_mann_whitney_handles_all_ties():
    """Degenerate input must return None rather than dividing by zero."""
    assert mann_whitney_p([0.5] * 6, [0.5] * 6) is None


def test_mann_whitney_p_is_a_probability():
    within = [0.10, 0.13, 0.09, 0.12, 0.10, 0.11]
    cross = [0.20, 0.22, 0.18, 0.21]
    p = mann_whitney_p(within, cross)
    assert 0.0 <= p <= 1.0 and not math.isnan(p)


def test_empty_input_is_survivable():
    centre, scale = robust_centre_scale([])
    assert centre == 0.0 and scale == 0.0


# --- The under-sampled-baseline rescue ------------------------------------
#
# Driven through a stub signal with programmed distances rather than real text, so
# these pin the decision boundary itself and cannot drift when an embedder changes.


class ProgrammedDistance(PairwiseSignal):
    """Returns a distance looked up from the pair of sample texts."""

    name = "semantic_distance"
    floor = 0.02

    def __init__(self, within_baseline: float, within_current: float, cross: float) -> None:
        self.within_baseline = within_baseline
        self.within_current = within_current
        self.cross = cross

    def distance(self, a, b):
        if a.text == b.text == "base":
            return self.within_baseline
        if a.text == b.text == "now":
            return self.within_current
        return self.cross


def _samples(text: str, n: int) -> list[Sample]:
    return [Sample(probe_id="p", target_name="t", text=text) for _ in range(n)]


def _verdict(signal: ProgrammedDistance):
    return evaluate_pairwise(signal, _samples("base", 5), _samples("now", 3), CFG)


def test_rescue_fires_when_the_baseline_saw_no_variation():
    """Five byte-identical baseline draws from a probe that does in fact vary."""
    verdict = _verdict(ProgrammedDistance(within_baseline=0.0, within_current=0.08, cross=0.08))
    assert verdict.level is Level.PASS
    assert "under-sampled" in verdict.detail


def test_rescue_stays_out_when_the_baseline_measured_a_real_spread():
    """The regression that motivated the fraction gate.

    A baseline spread of 0.017 sits inside the 0.02 floor but is a genuine
    measurement, not an absence of one. Widening off the current run's own
    chattiness here let clean JSON turning into prose-wrapped JSON pass.
    """
    verdict = _verdict(
        ProgrammedDistance(within_baseline=0.0175, within_current=0.22, cross=0.134)
    )
    assert verdict.level is Level.DRIFT
    assert "under-sampled" not in verdict.detail


def test_rescue_does_not_excuse_a_consistent_shift():
    """Degenerate baseline, but the new behaviour is internally consistent too."""
    verdict = _verdict(ProgrammedDistance(within_baseline=0.0, within_current=0.0, cross=0.6))
    assert verdict.level is Level.DRIFT


def test_rescue_needs_enough_baseline_pairs():
    """Two samples give one distance, whose spread is zero by arithmetic."""
    signal = ProgrammedDistance(within_baseline=0.0, within_current=0.08, cross=0.08)
    verdict = evaluate_pairwise(signal, _samples("base", 2), _samples("now", 3), CFG)
    assert "under-sampled" not in verdict.detail


def test_an_explicit_band_is_never_second_guessed():
    signal = ProgrammedDistance(within_baseline=0.0, within_current=0.5, cross=0.5)
    signal.band_override = 0.1
    verdict = _verdict(signal)
    assert verdict.level is Level.DRIFT
    assert "under-sampled" not in verdict.detail
