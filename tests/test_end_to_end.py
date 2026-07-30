"""The full pipeline: config -> sample -> compare -> report -> exit code.

Everything here runs against a mock transport, so the whole tool is exercised with
no network, no API key and no spend. If this file passes, `stillsane check` works.
"""

from __future__ import annotations

import asyncio
import itertools
import json

import httpx
import pytest
import yaml

from stillsane import cli
from stillsane.alerts import exit_code_for, payload_for, slack_payload
from stillsane.config import Config
from stillsane.models import Level
from stillsane.report import render
from stillsane.runner import capture_baseline, check
from stillsane.store import BaselineStore, History

STABLE = [
    '{"total": 1240.50, "due_date": "2026-07-01"}',
    '{"total": 1240.5, "due_date": "2026-07-01"}',
    '{"due_date": "2026-07-01", "total": 1240.50}',
]

DRIFTED = [
    'Here you go!\n{"total": 1240.50, "due_date": "2026-07-01"}\nAnything else?',
    'Sure thing:\n{"total": 1240.5, "due_date": "2026-07-01"}\nHappy to help.',
]

CONFIG = {
    "embedder": "hashing",  # keeps the suite offline
    "targets": [
        {
            "name": "prod",
            "base_url": "https://api.example.com/v1",
            "model": "some-model",
        }
    ],
    "probes": [
        {
            "id": "extract_invoice",
            "prompt": "Extract the total and due date as JSON.",
            "baseline_samples": 5,
            "check_samples": 3,
            "checks": ["valid_json", {"has_keys": ["total", "due_date"]}],
        }
    ],
}


#: Captured before any monkeypatching. The CLI tests swap out `httpx.AsyncClient`
#: to inject a fake provider, and since that patch lands on the httpx module
#: itself, a `make_client` that reached for the patched name would call itself.
_RealAsyncClient = httpx.AsyncClient


def make_client(texts, fingerprint="fp_a4f2b1"):
    """A fake provider that cycles through `texts`."""
    cycle = itertools.cycle(texts)

    def handler(request: httpx.Request) -> httpx.Response:
        content = next(cycle)
        return httpx.Response(
            200,
            json={
                "model": "some-model",
                "system_fingerprint": fingerprint,
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": len(content) // 4},
            },
        )

    return _RealAsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def env(tmp_path):
    config = Config.model_validate(CONFIG)
    return config, BaselineStore(tmp_path), History(tmp_path)


def run_baseline(config, store, texts, fingerprint="fp_a4f2b1"):
    async def go():
        async with make_client(texts, fingerprint) as client:
            return await capture_baseline(config, store, client=client)

    return asyncio.run(go())


def run_check(config, store, history, texts, fingerprint="fp_a4f2b1"):
    async def go():
        async with make_client(texts, fingerprint) as client:
            return await check(config, store, history, client=client)

    return asyncio.run(go())


# --- The happy path -------------------------------------------------------


def test_baseline_then_check_passes(env):
    config, store, history = env
    written = run_baseline(config, store, STABLE)
    assert len(written) == 1 and written[0].version == 1

    result = run_check(config, store, history, STABLE)
    assert result.level is Level.PASS
    assert result.exit_code == 0


def test_baseline_records_the_variance_pool(env):
    config, store, _ = env
    run_baseline(config, store, STABLE)
    baseline = store.load("prod", "extract_invoice")
    assert baseline.pooled["semantic_distance"], "baseline must seed the variance pool"
    assert baseline.anchors["semantic_distance"].scale >= 0


def test_drift_is_caught_end_to_end(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    assert result.level is Level.DRIFT
    assert result.exit_code == 1
    moved = {s.signal for s in result.probes[0].moved}
    assert "valid_json" in moved


def test_fingerprint_change_warns_end_to_end(env):
    config, store, history = env
    run_baseline(config, store, STABLE, fingerprint="fp_old")
    result = run_check(config, store, history, STABLE, fingerprint="fp_new")

    assert result.level is Level.WARN
    assert result.exit_code == 2
    fp = next(s for s in result.probes[0].signals if s.signal == "fingerprint")
    assert "fp_old -> fp_new" in fp.detail


# --- Guard rails ----------------------------------------------------------


def test_check_without_a_baseline_is_an_error_not_a_pass(env):
    config, store, history = env
    result = run_check(config, store, history, STABLE)
    assert result.level is Level.ERROR
    assert result.exit_code == 3
    assert "stillsane baseline" in result.probes[0].signals[0].detail


def test_editing_the_prompt_refuses_to_compare(env):
    """Otherwise your own edit gets reported as provider drift."""
    config, store, history = env
    run_baseline(config, store, STABLE)

    edited = Config.model_validate(
        {**CONFIG, "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}]}
    )
    result = run_check(edited, store, history, STABLE)
    assert result.level is Level.ERROR
    assert "different prompt" in result.probes[0].signals[0].detail


def test_a_dead_endpoint_is_an_error_not_drift(env):
    config, store, history = env
    run_baseline(config, store, STABLE)

    def dead(request):
        return httpx.Response(503, text="down")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(dead)) as client:
            return await check(config, store, history, client=client)

    result = asyncio.run(go())
    assert result.level is Level.ERROR
    assert result.exit_code == 3


def test_baseline_refuses_to_freeze_a_broken_endpoint(env):
    """Capturing a baseline of 503s would poison every future comparison."""
    config, store, _ = env

    def dead(request):
        return httpx.Response(500, text="nope")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(dead)) as client:
            return await capture_baseline(config, store, client=client)

    with pytest.raises(RuntimeError, match="no usable samples"):
        asyncio.run(go())


# --- Pooling writeback ----------------------------------------------------


def test_a_clean_run_grows_the_pool(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    before = len(store.load("prod", "extract_invoice").pooled["semantic_distance"])

    result = run_check(config, store, history, STABLE)
    assert result.level is Level.PASS

    after = len(store.load("prod", "extract_invoice").pooled["semantic_distance"])
    assert after > before, "a clean run should sharpen the band"


def test_a_drifting_run_does_not_grow_the_pool(env):
    """The band must never learn from the thing it is meant to be detecting."""
    config, store, history = env
    run_baseline(config, store, STABLE)
    before = store.load("prod", "extract_invoice").pooled["semantic_distance"]

    run_check(config, store, history, DRIFTED)

    after = store.load("prod", "extract_invoice").pooled["semantic_distance"]
    assert after == before


def test_history_is_written(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    run_check(config, store, history, STABLE)
    assert len(history.recent()) == 1


# --- Report and alerts ----------------------------------------------------


def test_report_names_what_moved(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    text = render(result, colour=False)
    assert "DRIFT" in text
    assert "extract_invoice @ prod" in text
    assert "valid_json" in text
    assert "baseline (v1" in text and "now:" in text


def test_report_stays_quiet_on_a_pass(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)

    text = render(result, colour=False)
    assert "PASS" in text
    # A passing probe is one line; the detail is noise nobody reads.
    assert "semantic_distance" not in text


def test_verbose_shows_the_signals_that_passed(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)
    assert "semantic_distance" in render(result, verbose=True, colour=False)


def test_report_never_emits_escape_codes_when_colour_is_off(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)
    assert "\033[" not in render(result, colour=False)


def test_alert_payload_is_structured(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    payload = payload_for(result)
    assert payload["level"] == "drift"
    assert payload["exit_code"] == 1
    assert payload["probes"][0]["probe"] == "extract_invoice"
    assert any(m["signal"] == "valid_json" for m in payload["probes"][0]["moved"])
    json.dumps(payload)  # must be serialisable


def test_slack_payload_is_bounded(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)
    text = slack_payload(result)["text"]
    assert "stillsane: DRIFT" in text and len(text) < 4000


def test_fail_on_warn_promotes_the_exit_code(env):
    config, store, history = env
    run_baseline(config, store, STABLE, fingerprint="fp_old")
    result = run_check(config, store, history, STABLE, fingerprint="fp_new")

    assert exit_code_for(result, fail_on_warn=False) == 2
    assert exit_code_for(result, fail_on_warn=True) == 1


# --- CLI ------------------------------------------------------------------


def test_init_writes_a_config_that_actually_parses(tmp_path, capsys):
    """A starter config that fails validation would be an embarrassing first run."""
    path = tmp_path / "stillsane.yaml"
    assert cli.main(["-c", str(path), "init"]) == 0
    Config.model_validate(yaml.safe_load(path.read_text()))


def test_init_refuses_to_clobber(tmp_path):
    path = tmp_path / "stillsane.yaml"
    cli.main(["-c", str(path), "init"])
    assert cli.main(["-c", str(path), "init"]) == 1
    assert cli.main(["-c", str(path), "init", "--force"]) == 0


def test_missing_config_exits_cleanly(tmp_path, capsys):
    code = cli.main(["-c", str(tmp_path / "nope.yaml"), "check"])
    assert code == 3
    assert "stillsane init" in capsys.readouterr().err


def test_cli_check_returns_the_drift_exit_code(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))

    texts = {"value": STABLE}

    def fake_client(*args, **kwargs):
        return make_client(texts["value"])

    monkeypatch.setattr("stillsane.runner.httpx.AsyncClient", fake_client)

    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    assert cli.main(["-c", str(config_path), "check"]) == 0

    texts["value"] = DRIFTED
    assert cli.main(["-c", str(config_path), "check"]) == 1
    assert "DRIFT" in capsys.readouterr().out


def test_cli_json_output_is_machine_readable(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )

    cli.main(["-c", str(config_path), "baseline"])
    capsys.readouterr()
    cli.main(["-c", str(config_path), "check", "--json"])
    assert json.loads(capsys.readouterr().out)["level"] == "pass"


def test_cli_state_lives_next_to_the_config(tmp_path, monkeypatch):
    config_path = tmp_path / "nested" / "stillsane.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )

    cli.main(["-c", str(config_path), "baseline"])
    assert (config_path.parent / ".stillsane" / "baselines").is_dir()
