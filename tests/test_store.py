"""Baseline versioning and run history."""

from __future__ import annotations

from conftest import sample

from stillsane.compare import Anchor
from stillsane.models import Level, ProbeVerdict, RunResult, SignalVerdict
from stillsane.store import BaselineStore, History, slug


def test_first_save_is_v1(tmp_path):
    store = BaselineStore(tmp_path)
    baseline = store.save("prod", "p", [sample('{"a": 1}')], "hash1")
    assert baseline.version == 1
    assert store.latest_version("prod", "p") == 1


def test_saving_never_overwrites(tmp_path):
    """Baselines are history. Losing the old one loses the ability to explain a drift."""
    store = BaselineStore(tmp_path)
    store.save("prod", "p", [sample("first")], "hash1")
    store.save("prod", "p", [sample("second")], "hash2")

    assert store.versions("prod", "p") == [1, 2]
    assert store.load("prod", "p", version=1).samples[0].text == "first"
    assert store.load("prod", "p").samples[0].text == "second"  # latest by default


def test_round_trip_preserves_everything_compared(tmp_path):
    store = BaselineStore(tmp_path)
    original = [
        sample('{"a": 1}', fingerprint="fp_1", completion_tokens=40, latency_ms=120.0),
        sample('{"a": 2}', fingerprint="fp_1", completion_tokens=42, latency_ms=130.0),
    ]
    store.save("prod", "p", original, "hash1", pooled={"semantic_distance": [0.1, 0.12]},
               anchors={"semantic_distance": Anchor(center=0.11, scale=0.02)})

    loaded = store.load("prod", "p")
    assert [s.text for s in loaded.samples] == ['{"a": 1}', '{"a": 2}']
    assert loaded.samples[0].completion_tokens == 40
    assert loaded.samples[0].fingerprint == "fp_1"
    assert loaded.pooled["semantic_distance"] == [0.1, 0.12]
    assert loaded.anchors["semantic_distance"].center == 0.11
    assert loaded.config_hash == "hash1"


def test_metadata_records_the_provider_identity(tmp_path):
    store = BaselineStore(tmp_path)
    store.save("prod", "p", [sample("x", fingerprint="fp_9", model="m-1")], "h")
    loaded = store.load("prod", "p")
    assert loaded.fingerprint == "fp_9" and loaded.model_id == "m-1"


def test_missing_baseline_loads_as_none(tmp_path):
    assert BaselineStore(tmp_path).load("prod", "nope") is None


def test_update_variance_leaves_reference_outputs_frozen(tmp_path):
    """The whole design rests on this: only the variance estimate may grow."""
    store = BaselineStore(tmp_path)
    baseline = store.save("prod", "p", [sample("original text")], "h",
                          pooled={"semantic_distance": [0.1]})

    store.update_variance(
        baseline,
        {"semantic_distance": [0.1, 0.11, 0.09]},
        {"semantic_distance": Anchor(center=0.1, scale=0.01)},
    )

    reloaded = store.load("prod", "p")
    assert reloaded.samples[0].text == "original text"
    assert reloaded.version == 1, "growing the pool must not create a new version"
    assert reloaded.pooled["semantic_distance"] == [0.1, 0.11, 0.09]


def test_names_with_awkward_characters_are_survivable(tmp_path):
    store = BaselineStore(tmp_path)
    store.save("prod/eu-west", "probe: invoice", [sample("x")], "h")
    assert store.load("prod/eu-west", "probe: invoice") is not None


def test_slug_never_returns_empty():
    assert slug("///") == "unnamed"
    assert slug("a b") == "a-b"


def test_usable_excludes_errored_samples(tmp_path):
    store = BaselineStore(tmp_path)
    store.save("prod", "p", [sample("good"), sample("", error="timeout")], "h")
    loaded = store.load("prod", "p")
    assert len(loaded.samples) == 2 and len(loaded.usable) == 1


# --- History --------------------------------------------------------------


def _run(level=Level.DRIFT):
    return RunResult(
        probes=[
            ProbeVerdict(
                probe_id="p",
                target_name="prod",
                level=level,
                signals=[
                    SignalVerdict(
                        signal="semantic_distance",
                        kind=None,
                        level=level,
                        detail="moved",
                        observed=0.31,
                        baseline=0.08,
                        z=6.2,
                    )
                ],
            )
        ]
    )


def test_history_records_every_signal(tmp_path):
    history = History(tmp_path)
    run_id = history.record(_run())
    assert len(run_id) == 12
    assert history.recent()[0][2] == "drift"


def test_history_answers_since_when(tmp_path):
    """The question an alert always provokes."""
    history = History(tmp_path)
    for _ in range(3):
        history.record(_run())
    trend = history.signal_trend("p", "prod", "semantic_distance")
    assert len(trend) == 3
    assert all(row[1] == 0.31 and row[2] == 6.2 for row in trend)


def test_history_is_created_on_demand(tmp_path):
    history = History(tmp_path / "nested" / "deeper")
    history.record(_run(Level.PASS))
    assert history.path.exists()
