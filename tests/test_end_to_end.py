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
from stillsane.alerts import exit_code_for, payload_for, send, slack_payload
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


def test_an_actionable_error_actually_reaches_the_report(env):
    """The message is the whole payload for a pseudo-signal.

    These carry no measurement, so the numeric columns render blank and the line
    became a lonely `baseline` with the instruction dropped -- an error telling the
    user nothing at all.
    """
    config, store, history = env
    result = run_check(config, store, history, STABLE)  # no baseline captured

    text = render(result, colour=False)
    assert result.level is Level.ERROR
    assert "stillsane baseline" in text, text


def test_a_stale_baseline_says_so_in_the_report(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    edited = Config.model_validate(
        {**CONFIG, "probes": [{**CONFIG["probes"][0], "prompt": "Something else."}]}
    )
    text = render(run_check(edited, store, history, STABLE), colour=False)
    assert "recapture" in text and "different" in text


def test_fingerprint_only_alert_shows_no_text_diff(env):
    """Nothing the model wrote changed, so a before/after block would mislead.

    The excerpts differ only in incidental formatting (`1240.5` vs `1240.50`), and
    showing them next to a fingerprint alert reads as though the number moved.
    """
    config, store, history = env
    run_baseline(config, store, STABLE, fingerprint="fp_old")
    result = run_check(config, store, history, STABLE, fingerprint="fp_new")

    text = render(result, colour=False)
    assert result.level is Level.WARN
    assert "fingerprint" in text
    assert "baseline (v1" not in text and "now:" not in text


def test_content_drift_still_shows_the_text_diff(env):
    """The complement: suppressing the block must not suppress it when it matters."""
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    text = render(result, colour=False)
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


def test_alerts_are_delivered_to_both_sinks(env, monkeypatch):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    posted = []
    monkeypatch.setattr(
        "stillsane.alerts.httpx.post",
        lambda url, **kw: posted.append((url, kw.get("json"))) or _Ok(),
    )
    send(result, "https://example.com/hook", "https://hooks.slack.test/x")

    assert [url for url, _ in posted] == [
        "https://example.com/hook",
        "https://hooks.slack.test/x",
    ]
    assert posted[0][1]["level"] == "drift"
    assert "stillsane: DRIFT" in posted[1][1]["text"]


def test_a_dead_webhook_does_not_fail_the_check(env, monkeypatch, capsys):
    """Alerting is best-effort by design, and this is the promise being kept.

    The check already did its job. Losing that result because the notification
    could not be delivered would be strictly worse than the notification failing.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    def explode(url, **kw):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr("stillsane.alerts.httpx.post", explode)
    send(result, "https://unreachable.test/hook", None)  # must not raise

    assert "could not deliver alert" in capsys.readouterr().err


def test_a_webhook_returning_an_error_is_reported_not_raised(env, monkeypatch, capsys):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    monkeypatch.setattr("stillsane.alerts.httpx.post", lambda url, **kw: _Ok(500))
    send(result, "https://example.com/hook", None)

    assert "HTTP 500" in capsys.readouterr().err


class _Ok:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


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


# --- Floored bands are surfaced -------------------------------------------


def test_a_defaulted_band_is_reported_at_baseline_time(env):
    """Identical samples mean nothing was measured; the band was defaulted.

    The engine has always known this. Until now it kept it to itself, so a user
    could not tell a genuinely deterministic probe from an under-sampled one.
    """
    config, store, _ = env
    identical = ['{"total": 1, "due_date": "x"}']
    written = run_baseline(config, store, identical)
    assert "semantic_distance" in written[0].floored


def test_a_measured_band_is_not_reported_as_floored(env):
    config, store, _ = env
    written = run_baseline(config, store, STABLE)
    assert "semantic_distance" not in written[0].floored


def test_the_report_marks_a_floored_band(env):
    config, store, history = env
    identical = ['{"total": 1, "due_date": "x"}']
    run_baseline(config, store, identical)
    result = run_check(config, store, history, identical)
    assert "(floor)" in render(result, verbose=True, colour=False)


def test_the_report_does_not_mark_a_measured_band(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)
    text = render(result, verbose=True, colour=False)
    semantic = next(ln for ln in text.splitlines() if "semantic_distance" in ln)
    assert "(floor)" not in semantic


# --- History surface -------------------------------------------------------


def test_history_records_and_lists_signals(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    run_check(config, store, history, STABLE)

    recorded = history.recorded_signals()
    assert ("extract_invoice", "prod", "semantic_distance") in recorded


def test_history_answers_since_when_across_runs(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    run_check(config, store, history, STABLE)
    run_check(config, store, history, DRIFTED)

    trend = history.signal_trend("extract_invoice", "prod", "valid_json")
    assert len(trend) == 2
    # Most recent first: the drifted run, then the clean one.
    assert trend[0][1] == 0.0 and trend[1][1] == 1.0


def test_history_cli_lists_runs(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    cli.main(["-c", str(config_path), "baseline"])
    cli.main(["-c", str(config_path), "check"])
    capsys.readouterr()

    assert cli.main(["-c", str(config_path), "history"]) == 0
    assert "pass" in capsys.readouterr().out


def test_history_cli_needs_probe_and_target_with_signal(tmp_path, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    code = cli.main(["-c", str(config_path), "history", "--signal", "semantic_distance"])
    assert code == 1
    assert "--probe and --target" in capsys.readouterr().err


def test_capture_names_floored_pointwise_signals(env):
    """The capture-time warning used to see only the distance signals.

    `_floored_signals` read from the pooled record, which by design holds pairwise
    distances only, so `length_chars` and `completion_tokens` could never be named
    however floored they were. Against a real baseline that meant one signal
    reported and three floored in fact, and the gap was invisible until `bands`
    recomputed them separately.
    """
    config, store, _ = env
    run_baseline(config, store, STABLE)
    baseline = store.load("prod", "extract_invoice")

    from stillsane.compare import BandConfig
    from stillsane.runner import _floored_signals
    from stillsane.signals import HashingEmbedder, build_signals

    signals = build_signals(config.probes[0].checks, HashingEmbedder())
    named = _floored_signals(signals, baseline.pooled, BandConfig(), baseline.usable)

    assert "length_chars" in named, (
        "a floored pointwise band must be nameable at capture time, not only by `bands`"
    )


def test_capture_ignores_signals_that_do_not_apply(env):
    """No token counts reported means nothing to name, not a crash."""
    config, store, _ = env
    run_baseline(config, store, STABLE)
    baseline = store.load("prod", "extract_invoice")

    from stillsane.compare import BandConfig
    from stillsane.runner import _floored_signals
    from stillsane.signals import HashingEmbedder, build_signals

    for sample in baseline.samples:
        sample.completion_tokens = None

    signals = build_signals(config.probes[0].checks, HashingEmbedder())
    named = _floored_signals(signals, baseline.pooled, BandConfig(), baseline.usable)
    assert "completion_tokens" not in named
