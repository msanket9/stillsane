"""Pooling, and the boiling-frog failure it has to avoid.

Folding clean runs back into the variance estimate is what makes the band tighten
over time. Done naively it is also how a drift monitor goes quietly blind, so
these tests exist mainly to prove it cannot.
"""

from __future__ import annotations

import pytest
from conftest import sample

from stillsane.compare import (
    Anchor,
    BandConfig,
    PoolConfig,
    anchor_of,
    is_clean,
    merge_pool,
    pool_from_run,
    robust_band,
    robust_centre_scale,
    within_run_evidence,
)
from stillsane.compare.verdict import compare_probe
from stillsane.models import Direction, Level, ProbeVerdict, SignalVerdict

CFG = PoolConfig()


def _verdict(level: Level, *zs: float) -> ProbeVerdict:
    return ProbeVerdict(
        probe_id="p",
        target_name="t",
        level=level,
        signals=[
            SignalVerdict(signal=f"s{i}", kind=None, level=Level.PASS, detail="", z=z)
            for i, z in enumerate(zs)
        ],
    )


# --- What counts as evidence of normality --------------------------------


def test_only_passing_runs_pool():
    assert is_clean(_verdict(Level.PASS, 0.2, 0.4))
    assert not is_clean(_verdict(Level.WARN, 0.2))
    assert not is_clean(_verdict(Level.DRIFT, 0.2))
    assert not is_clean(_verdict(Level.ERROR))


def test_a_run_that_barely_scraped_through_does_not_pool():
    """Passing is not the same as being evidence of normal behaviour.

    A run sitting at z=2.5 passed only because the threshold is 3. Treating it as
    a description of 'normal' is how the band starts drifting with the thing it is
    supposed to be measuring.
    """
    assert not is_clean(_verdict(Level.PASS, 2.5))
    assert is_clean(_verdict(Level.PASS, 0.9))


def test_signals_without_an_effect_size_do_not_block_pooling():
    """Categorical signals carry no z; they must not veto pooling by omission."""
    assert is_clean(_verdict(Level.PASS, *[]))
    v = _verdict(Level.PASS, 0.3)
    v.signals.append(SignalVerdict(signal="fingerprint", kind=None, level=Level.PASS, detail=""))
    assert is_clean(v)


# --- The asymmetry -------------------------------------------------------


def test_pool_may_tighten_without_limit():
    """Consistent behaviour should make the band sharper, not just stable."""
    original = [0.05, 0.30, 0.10, 0.28, 0.12, 0.26]
    anchor = anchor_of(original)
    tight = [0.10, 0.11, 0.10, 0.11, 0.10, 0.11] * 3

    pooled = merge_pool(original, tight, anchor)
    assert len(pooled) == len(original) + len(tight)

    _, before = robust_centre_scale(original)
    _, after = robust_centre_scale(pooled)
    assert after < before


def test_pool_refuses_to_widen_past_the_cap():
    """The boiling frog. Drift must never be absorbed into 'normal'."""
    original = [0.10, 0.11, 0.09, 0.10, 0.12, 0.10]
    anchor = anchor_of(original)
    creeping = [0.40, 0.55, 0.42, 0.61, 0.38, 0.58]

    pooled = merge_pool(original, creeping, anchor)
    assert pooled == original, "widening beyond the cap must be rejected outright"


def test_repeated_creep_never_accumulates():
    """Simulate a month of gradual drift arriving a little at a time.

    Each individual step is small enough that it might slip past `is_clean`. The
    anchor is what stops the sum of many small steps from moving the band, because
    every merge is measured against day one rather than against yesterday.
    """
    original = [0.10, 0.11, 0.09, 0.10, 0.12, 0.10]
    anchor = anchor_of(original)
    pool = list(original)

    for step in range(1, 31):
        drifting = [0.10 + step * 0.02 + jitter for jitter in (-0.01, 0.0, 0.01, 0.02)]
        pool = merge_pool(pool, drifting, anchor)

    final_centre, final_scale = robust_centre_scale(pool)
    assert final_scale <= anchor.scale * CFG.max_widen + 1e-9
    assert final_centre <= anchor.center + CFG.max_centre_shift * anchor.scale + 1e-9

    # And the band must still fire on the drifted values.
    band = robust_band(pool, direction=Direction.UP_IS_BAD, cfg=BandConfig(), floor=0.02)
    assert band.upper < 0.70, f"band crept out to {band.upper}"


def test_centre_cap_closes_the_ratchet():
    """The scale cap alone is not enough, and this is the case that proves it.

    Every batch here has the *same* tight spread as the original, so the scale cap
    never trips -- only the centre moves. Judged against the previous run each step
    looks unremarkable; judged against day one the band has walked off its subject.
    """
    original = [0.10, 0.11, 0.09, 0.10, 0.11, 0.10]
    anchor = anchor_of(original)
    pool = list(original)

    for step in range(1, 41):
        shifted = [0.10 + step * 0.01 + j for j in (-0.005, 0.0, 0.005, 0.01)]
        pool = merge_pool(pool, shifted, anchor)

    final_centre, _ = robust_centre_scale(pool)
    assert final_centre <= anchor.center + CFG.max_centre_shift * anchor.scale + 1e-9
    assert final_centre < 0.2, f"centre ratcheted to {final_centre}"


def test_pool_is_a_bounded_window():
    anchor = anchor_of([0.10, 0.11, 0.09, 0.10])
    pool: list[float] = []
    for _ in range(200):
        pool = merge_pool(pool, [0.10, 0.11, 0.09, 0.10], anchor)
    assert len(pool) <= CFG.max_values


def test_pool_keeps_the_recent_end_when_evicting():
    anchor = Anchor(center=0.5, scale=1.0)  # generous: nothing gets rejected
    pool = [0.1] * PoolConfig().max_values
    pool = merge_pool(pool, [0.9, 0.9, 0.9], anchor)
    assert len(pool) == PoolConfig().max_values
    assert pool[-3:] == [0.9, 0.9, 0.9]


def test_empty_update_is_a_no_op():
    original = [0.1, 0.2, 0.3]
    assert merge_pool(original, [], anchor_of(original)) == original


# --- Which distances get pooled ------------------------------------------


def test_evidence_comes_from_pairwise_signals_only(signals_for):
    """Pointwise bands sit on the value itself, so pooling current values would be
    absorbing exactly the movement the signal exists to detect."""
    current = [sample('{"a": 1}'), sample('{"a": 2}'), sample('{"a": 3}')]
    evidence = within_run_evidence(signals_for(["valid_json"]), current)

    assert "semantic_distance" in evidence
    for pointwise in ("length_chars", "completion_tokens", "latency_ms", "valid_json"):
        assert pointwise not in evidence


def test_evidence_is_measured_within_the_run_not_against_the_baseline(signals_for):
    """The decisive property: evidence must be blind to wholesale drift.

    Both runs here are internally identical but say completely different things.
    Pooling cross distances would report a large number and teach the band to
    expect the drift. Within-run distance correctly reports ~0 for both, because
    both runs are perfectly self-consistent.
    """
    signals = signals_for(None)
    original = [sample("the total is 1240.50") for _ in range(3)]
    drifted = [sample("I cannot help with that request") for _ in range(3)]

    before = within_run_evidence(signals, original)["semantic_distance"]
    after = within_run_evidence(signals, drifted)["semantic_distance"]

    assert max(before) < 0.01 and max(after) < 0.01


def test_evidence_needs_at_least_a_pair(signals_for):
    assert within_run_evidence(signals_for(None), [sample("only one")]) == {}


def test_pool_from_run_applies_per_signal():
    existing = {"semantic_distance": [0.1, 0.11], "length_chars": [400.0, 402.0]}
    anchors = {k: anchor_of(v) for k, v in existing.items()}
    new = {"semantic_distance": [0.10, 0.11], "length_chars": [401.0, 400.0]}

    pooled = pool_from_run(new, existing, anchors)
    assert len(pooled["semantic_distance"]) == 4
    assert len(pooled["length_chars"]) == 4


def test_pooled_variance_feeds_the_comparison(signals_for):
    """A pooled band actually reaches the verdict path.

    Without this the pooling module would be a well-tested orphan.
    """
    baseline = [sample('{"a": 1}') for _ in range(5)]
    current = [sample('{"a": 1}') for _ in range(3)]

    wide_pool = {"semantic_distance": [0.4, 0.45, 0.5, 0.42, 0.48, 0.44]}
    verdict = compare_probe(
        probe_id="p",
        target_name="t",
        signals=signals_for(None),
        baseline=baseline,
        current=current,
        pooled=wide_pool,
    )
    semantic = next(s for s in verdict.signals if s.signal == "semantic_distance")
    assert semantic.band.n == 6
    assert semantic.band.center == pytest.approx(0.445, abs=0.02)
