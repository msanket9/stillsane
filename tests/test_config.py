"""Config validation, and the hash that keeps baselines honest."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from stillsane.config import (
    AlertConfig,
    Config,
    ProbeConfig,
    TargetConfig,
    Turn,
    config_hash,
    load_config,
)

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


def test_misspelled_target_key_is_rejected_by_name():
    bad = {
        **MINIMAL,
        "targets": [{**MINIMAL["targets"][0], "escalate_fingerpint": True}],
    }
    with pytest.raises(ValidationError, match="escalate_fingerpint"):
        Config.model_validate(bad)


def test_misspelled_alerts_key_is_rejected_by_name():
    bad = {**MINIMAL, "alrets": {"webhook": "https://example.com"}}
    with pytest.raises(ValidationError, match="alrets"):
        Config.model_validate(bad)


def test_repeat_every_defaults_to_no_suppression():
    """`None` -- every non-PASS run alerts -- is the safe default; suppression
    is opt-in."""
    assert AlertConfig().repeat_every is None


def test_repeat_every_accepts_zero_and_a_positive_count():
    assert AlertConfig(repeat_every=0).repeat_every == 0
    assert AlertConfig(repeat_every=3).repeat_every == 3


def test_repeat_every_is_read_via_config_not_just_the_model():
    cfg = Config.model_validate({**MINIMAL, "alerts": {"repeat_every": 0}})
    assert cfg.alerts.repeat_every == 0


# --- attribute_to (raw-model control) ---------------------------------------


def _with_control():
    return {
        "targets": [
            {**MINIMAL["targets"][0], "name": "prod", "attribute_to": "raw"},
            {**MINIMAL["targets"][0], "name": "raw"},
        ],
        "probes": [{**MINIMAL["probes"][0], "targets": ["prod", "raw"]}],
    }


def test_attribute_to_defaults_to_none():
    assert TargetConfig(name="t", base_url="https://x", model="m").attribute_to is None


def test_attribute_to_accepts_a_real_target():
    cfg = Config.model_validate(_with_control())
    assert cfg.target("prod").attribute_to == "raw"


def test_attribute_to_an_unknown_target_is_rejected():
    bad = _with_control()
    bad["targets"][0]["attribute_to"] = "nope"
    with pytest.raises(ValidationError, match="not a target this config defines"):
        Config.model_validate(bad)


def test_attribute_to_itself_is_rejected():
    bad = _with_control()
    bad["targets"][0]["attribute_to"] = "prod"
    with pytest.raises(ValidationError, match="cannot be its own control"):
        Config.model_validate(bad)


def test_attribute_to_does_not_affect_the_baseline_hash():
    """Pairing a control target is a report-time annotation, not a change to
    what gets sampled or compared -- pointing `attribute_to` at a different
    target, or unsetting it, must never invalidate an existing baseline."""
    target = TargetConfig(name="prod", base_url="https://a/v1", model="m")
    target_with_control = TargetConfig(
        name="prod", base_url="https://a/v1", model="m", attribute_to="raw"
    )
    probe = ProbeConfig(id="p", prompt="x")
    assert config_hash(probe, target) == config_hash(probe, target_with_control)


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


def test_changing_response_path_invalidates_an_http_targets_baseline():
    """`response_path` decides what gets extracted and compared -- editing it,
    e.g. fixing `content.0` to `content[type=text].text` for a thinking model,
    changes the comparison itself and must not be silently compared against a
    baseline captured under the old extraction.
    """
    probe = ProbeConfig(id="p", prompt="base prompt")
    a = TargetConfig(name="t", type="http", base_url="https://x", body={"q": "{{prompt}}"},
                      response_path="content.0.text")
    b = TargetConfig(name="t", type="http", base_url="https://x", body={"q": "{{prompt}}"},
                      response_path="content[type=text].text")
    assert config_hash(probe, a) != config_hash(probe, b)


def test_changing_method_invalidates_an_http_targets_baseline():
    probe = ProbeConfig(id="p", prompt="base prompt")
    a = TargetConfig(name="t", type="http", base_url="https://x", body={"q": "{{prompt}}"}, method="POST")
    b = TargetConfig(name="t", type="http", base_url="https://x", body={"q": "{{prompt}}"}, method="GET")
    assert config_hash(probe, a) != config_hash(probe, b)


def test_response_path_does_not_touch_a_non_http_targets_hash():
    """Mirrors the claude_code-only-fields test above: an http-only field must
    not be part of the hash for a type it has no meaning for.
    """
    probe, target = _pair()
    before = config_hash(probe, target)
    same_after_upgrade = TargetConfig(name="prod", base_url="https://a/v1", model="m", response_path="x")
    assert config_hash(probe, same_after_upgrade) == before


def test_changing_headers_invalidates_the_baseline():
    """A header can select which backend answers -- the README's own `type:
    http` example carries `x-tenant: acme` -- so editing one must not be
    silently compared against a baseline captured under a different value.
    """
    probe = ProbeConfig(id="p", prompt="base prompt")
    a = TargetConfig(name="t", type="http", base_url="https://x", body={"q": "{{prompt}}"},
                      headers={"x-tenant": "acme"})
    b = TargetConfig(name="t", type="http", base_url="https://x", body={"q": "{{prompt}}"},
                      headers={"x-tenant": "beta"})
    assert config_hash(probe, a) != config_hash(probe, b)


def test_headers_are_part_of_every_targets_hash_not_just_http():
    """Unlike `response_path`/`method`, `headers` applies to every target
    type, so it belongs in the always-present part of `identity()`.
    """
    probe, prod = _pair()
    other = TargetConfig(
        name="prod", base_url="https://a/v1", model="m", headers={"x-tenant": "acme"}
    )
    assert config_hash(probe, prod) != config_hash(probe, other)


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


# --- Scripted turns (F6) ----------------------------------------------------


def test_turn_rejects_an_unknown_role():
    with pytest.raises(ValidationError, match="user.*assistant|assistant.*user"):
        Turn(role="narrator", content="x")


def test_turn_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        Turn.model_validate({"role": "user", "content": "x", "name": "bob"})


def test_empty_turns_list_is_rejected():
    """`turns: []` says nothing `prompt` alone does not -- it should be omitted,
    not written as a no-op list."""
    with pytest.raises(ValidationError, match=r"turns.*\[\]|omit"):
        ProbeConfig(id="p", prompt="x", turns=[])


def test_turns_defaults_to_none():
    assert ProbeConfig(id="p", prompt="x").turns is None


def test_turns_are_parsed_in_order():
    probe = ProbeConfig(
        id="p",
        prompt="x",
        turns=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
    )
    assert probe.turns == [
        Turn(role="user", content="first"),
        Turn(role="assistant", content="second"),
    ]


def test_editing_turns_invalidates_the_baseline():
    """Scripted history is sent verbatim on every sample -- editing it changes
    what the target actually sees, exactly like editing `prompt` does."""
    target = TargetConfig(name="prod", base_url="https://a/v1", model="m")
    a = ProbeConfig(id="p", prompt="q", turns=[{"role": "user", "content": "a"}])
    b = ProbeConfig(id="p", prompt="q", turns=[{"role": "user", "content": "b"}])
    assert config_hash(a, target) != config_hash(b, target)


def test_adding_turns_invalidates_the_baseline():
    target = TargetConfig(name="prod", base_url="https://a/v1", model="m")
    without = ProbeConfig(id="p", prompt="q")
    with_turns = ProbeConfig(id="p", prompt="q", turns=[{"role": "user", "content": "a"}])
    assert config_hash(without, target) != config_hash(with_turns, target)


def test_a_probe_with_turns_cannot_target_claude_code():
    """`claude` in `-p` mode has no flag to inject prior assistant turns, so a
    probe scripted with `turns` would silently run as if it had none."""
    bad = {
        "targets": [{"name": "cc", "type": "claude_code"}],
        "probes": [
            {
                "id": "p",
                "prompt": "x",
                "turns": [{"role": "user", "content": "a"}],
            }
        ],
    }
    with pytest.raises(ValidationError, match="turns.*claude_code|claude_code.*turns"):
        Config.model_validate(bad)


def test_a_probe_with_turns_can_target_claude_code_if_scoped_away():
    """The same probe is fine once it is explicitly scoped off the target that
    cannot honour it -- the rejection is about the pairing, not the probe."""
    cfg = Config.model_validate(
        {
            "targets": [
                {"name": "cc", "type": "claude_code"},
                {"name": "api", "base_url": "https://a/v1", "model": "m"},
            ],
            "probes": [
                {
                    "id": "p",
                    "prompt": "x",
                    "turns": [{"role": "user", "content": "a"}],
                    "targets": ["api"],
                }
            ],
        }
    )
    assert [t.name for _, t in cfg.pairs()] == ["api"]


def test_a_probe_without_turns_can_target_claude_code():
    cfg = Config.model_validate(
        {
            "targets": [{"name": "cc", "type": "claude_code"}],
            "probes": [{"id": "p", "prompt": "x"}],
        }
    )
    assert cfg.pairs()
