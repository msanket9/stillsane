"""Config validation, and the hash that keeps baselines honest."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from stillsane.config import Config, ProbeConfig, TargetConfig, config_hash, load_config

MINIMAL = {
    "targets": [
        {
            "name": "prod",
            "type": "openai_compatible",
            "base_url": "https://api.example.com/v1",
            "model": "some-model",
        }
    ],
    "probes": [
        {
            "id": "extract_invoice",
            "prompt": "Extract the total and due date as JSON from: ...",
            "samples": 5,
            "checks": ["valid_json", {"has_keys": ["total", "due_date"]}],
        }
    ],
}


def test_the_config_from_the_brief_loads():
    cfg = Config.model_validate(MINIMAL)
    assert cfg.targets[0].watch_fingerprint is True
    assert cfg.probes[0].baseline_samples == 5
    # The cheaper per-run default is what keeps ongoing spend down.
    assert cfg.probes[0].check_samples == 3


def test_samples_alias_only_sets_the_baseline_count():
    probe = ProbeConfig(id="p", prompt="x", samples=9)
    assert probe.baseline_samples == 9
    assert probe.check_samples == 3


def test_explicit_counts_win_over_the_alias():
    probe = ProbeConfig(id="p", prompt="x", samples=9, check_samples=2)
    assert probe.baseline_samples == 9 and probe.check_samples == 2


def test_a_single_baseline_sample_is_rejected():
    """One sample carries no variance information, so it cannot found a band."""
    with pytest.raises(ValidationError, match="at least 2"):
        ProbeConfig(id="p", prompt="x", baseline_samples=1)


def test_openai_target_requires_a_model():
    with pytest.raises(ValidationError, match="`model` is required"):
        TargetConfig(name="t", type="openai_compatible", base_url="https://x/v1")


def test_http_target_requires_a_body():
    with pytest.raises(ValidationError, match="`body` is required"):
        TargetConfig(name="t", type="http", base_url="https://x")


def test_duplicate_names_are_rejected():
    bad = {**MINIMAL, "targets": MINIMAL["targets"] * 2}
    with pytest.raises(ValidationError, match="duplicate target names"):
        Config.model_validate(bad)


def test_probe_pointing_at_an_unknown_target_is_rejected():
    bad = {
        **MINIMAL,
        "probes": [{**MINIMAL["probes"][0], "targets": ["staging"]}],
    }
    with pytest.raises(ValidationError, match="unknown target"):
        Config.model_validate(bad)


def test_pairs_expands_to_every_combination():
    cfg = Config.model_validate(
        {
            "targets": [
                {"name": "prod", "base_url": "https://a/v1", "model": "m"},
                {"name": "staging", "base_url": "https://b/v1", "model": "m"},
            ],
            "probes": [
                {"id": "one", "prompt": "x"},
                {"id": "two", "prompt": "y", "targets": ["prod"]},
            ],
        }
    )
    got = {(p.id, t.name) for p, t in cfg.pairs()}
    assert got == {("one", "prod"), ("one", "staging"), ("two", "prod")}


# --- The hash -------------------------------------------------------------


def _pair(**probe_kw):
    target = TargetConfig(name="prod", base_url="https://a/v1", model="m")
    probe = ProbeConfig(id="p", prompt="base prompt", **probe_kw)
    return probe, target


def test_editing_a_prompt_invalidates_the_baseline():
    """Otherwise your own edit reads as provider drift and the tool lies to you."""
    a, t = _pair()
    b = ProbeConfig(id="p", prompt="a different prompt")
    assert config_hash(a, t) != config_hash(b, t)


def test_changing_the_model_invalidates_the_baseline():
    probe, prod = _pair()
    other = TargetConfig(name="prod", base_url="https://a/v1", model="different-model")
    assert config_hash(probe, prod) != config_hash(probe, other)


def test_changing_temperature_invalidates_the_baseline():
    probe, prod = _pair()
    hotter = TargetConfig(name="prod", base_url="https://a/v1", model="m", temperature=1.2)
    assert config_hash(probe, prod) != config_hash(probe, hotter)


def test_changing_checks_invalidates_the_baseline():
    a, t = _pair(checks=["valid_json"])
    b = ProbeConfig(id="p", prompt="base prompt", checks=["valid_json", {"has_keys": ["x"]}])
    assert config_hash(a, t) != config_hash(b, t)


def test_sample_counts_do_not_invalidate_the_baseline():
    """Asking for more samples does not change what the model says."""
    a, t = _pair(baseline_samples=5)
    b = ProbeConfig(id="p", prompt="base prompt", baseline_samples=9, check_samples=7)
    assert config_hash(a, t) == config_hash(b, t)


def test_switching_embedder_invalidates_the_baseline():
    """A stored variance pool only means anything on the scale that produced it.

    Without this the pool stays put while every new distance arrives on a different
    scale, so the bands stop matching what they are compared against and the tool
    reports drift, or misses it, with full confidence.
    """
    probe, target = _pair()
    assert config_hash(probe, target, "model2vec") != config_hash(probe, target, "hashing")


def test_renaming_a_target_does_not_invalidate_the_baseline():
    probe, prod = _pair()
    renamed = TargetConfig(name="production", base_url="https://a/v1", model="m")
    assert config_hash(probe, prod) == config_hash(probe, renamed)


def test_hash_is_stable_across_processes():
    """It is persisted next to the baseline, so it cannot depend on hash seeding."""
    probe, target = _pair()
    assert config_hash(probe, target) == config_hash(probe, target)
    assert len(config_hash(probe, target)) == 16


def test_claude_code_only_fields_do_not_touch_other_targets_hash():
    """A field meaningful only to `claude_code` must not appear in another
    target's `identity()` at all -- present with a stable default value is not
    good enough. Adding any new key to the dict, for any reason, changes every
    existing target's hash the moment someone upgrades, which invalidates every
    baseline in existence overnight regardless of what that key's value is.

    Caught for real: `claude_command`/`allowed_tools` were added unconditionally
    once, and the bundled example's own committed baseline refused to compare
    against a fresh check because of it.
    """
    probe, http_target = _pair()
    before = config_hash(probe, http_target)

    # Simulates what upgrading past this feature looks like for a target that has
    # nothing to do with it: the schema now has the new fields, at their defaults,
    # on every target regardless of type.
    same_target_after_upgrade = TargetConfig(
        name="prod", base_url="https://a/v1", model="m",
        claude_command="claude", allowed_tools=None,
    )
    assert config_hash(probe, same_target_after_upgrade) == before


def test_claude_code_fields_do_affect_a_claude_code_targets_hash():
    """The other half: for the type they are meaningful to, they must count."""
    probe = ProbeConfig(id="p", prompt="base prompt")
    a = TargetConfig(name="c", type="claude_code")
    b = TargetConfig(name="c", type="claude_code", allowed_tools=["Read"])
    assert config_hash(probe, a) != config_hash(probe, b)


# --- Per-type validation ----------------------------------------------------


@pytest.mark.parametrize("target_type", ["openai_compatible", "http"])
def test_base_url_is_required_for_url_based_types(target_type):
    with pytest.raises(ValueError, match="base_url"):
        TargetConfig(name="t", type=target_type, model="m", body={"q": "{{prompt}}"})


def test_base_url_is_not_required_for_claude_code():
    """Nothing to point it at -- this target shells out to a local binary."""
    TargetConfig(name="claude", type="claude_code")  # must not raise


def test_claude_command_defaults_to_the_bare_command():
    assert TargetConfig(name="claude", type="claude_code").claude_command == "claude"


def test_allowed_tools_defaults_to_none():
    """None is the safe default: deny-everything mode, not agentic mode."""
    assert TargetConfig(name="claude", type="claude_code").allowed_tools is None


# --- Loading --------------------------------------------------------------


def test_load_from_disk(tmp_path):
    path = tmp_path / "stillsane.yaml"
    path.write_text(yaml.safe_dump(MINIMAL))
    cfg = load_config(path)
    assert cfg.probes[0].id == "extract_invoice"


def test_missing_config_points_at_init(tmp_path):
    with pytest.raises(FileNotFoundError, match="stillsane init"):
        load_config(tmp_path / "nope.yaml")


def test_api_key_is_read_from_the_environment(monkeypatch):
    target = TargetConfig(
        name="t", base_url="https://a/v1", model="m", api_key_env="MY_TEST_KEY"
    )
    monkeypatch.setenv("MY_TEST_KEY", "secret-value")
    assert target.api_key() == "secret-value"


def test_missing_api_key_fails_loudly(monkeypatch):
    target = TargetConfig(
        name="t", base_url="https://a/v1", model="m", api_key_env="MY_TEST_KEY"
    )
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MY_TEST_KEY"):
        target.api_key()


def test_thresholds_flow_into_the_engine():
    cfg = Config.model_validate({**MINIMAL, "thresholds": {"warn_k": 2.0, "drift_k": 4.0}})
    band = cfg.thresholds.to_band_config()
    assert band.warn_k == 2.0 and band.drift_k == 4.0
