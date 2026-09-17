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
from stillsane.alerts import exit_code_for, payload_for, send, should_alert, slack_payload
from stillsane.config import Config
from stillsane.models import Level
from stillsane.report import render
from stillsane.runner import capture_baseline, check
from stillsane.store import BaselineStore, History, RunSampleStore

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


def run_baseline(config, store, texts, fingerprint="fp_a4f2b1", compare_previous=False):
    async def go():
        async with make_client(texts, fingerprint) as client:
            return await capture_baseline(
                config, store, client=client, compare_previous=compare_previous
            )

    return asyncio.run(go())


def run_check(config, store, history, texts, fingerprint="fp_a4f2b1", against_stale=False):
    async def go():
        async with make_client(texts, fingerprint) as client:
            return await check(config, store, history, client=client, against_stale=against_stale)

    return asyncio.run(go())


# --- Concurrency -----------------------------------------------------------


def test_concurrency_is_shared_across_probes_against_one_target(tmp_path):
    """`collect` used to open a fresh semaphore per call, and every probe's
    `collect` runs concurrently in `_sample_all` -- so N probes against one
    target opened up to `DEFAULT_CONCURRENCY * N` requests in flight against
    that one endpoint, which is exactly the rate-limit outage the cap exists to
    prevent (see the module docstring in `targets/base.py`).
    """
    from stillsane.targets import DEFAULT_CONCURRENCY

    config = Config.model_validate(
        {
            "embedder": "hashing",
            "targets": [{"name": "prod", "base_url": "https://api.example.com/v1", "model": "m"}],
            "probes": [
                {"id": f"p{i}", "prompt": f"prompt {i}", "baseline_samples": 10, "check_samples": 3}
                for i in range(3)
            ],
        }
    )
    store = BaselineStore(tmp_path)

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    async def go():
        async with _RealAsyncClient(transport=httpx.MockTransport(handler)) as client:
            await capture_baseline(config, store, client=client)

    asyncio.run(go())
    assert max_in_flight <= DEFAULT_CONCURRENCY, (
        f"{max_in_flight} requests in flight against one target, "
        f"cap is {DEFAULT_CONCURRENCY}"
    )


# --- The happy path -------------------------------------------------------


def test_baseline_then_check_passes(env):
    config, store, history = env
    written = run_baseline(config, store, STABLE)
    assert len(written) == 1 and written[0].baseline.version == 1

    result = run_check(config, store, history, STABLE)
    assert result.level is Level.PASS
    assert result.exit_code == 0


def test_baseline_records_the_variance_pool(env):
    config, store, _ = env
    run_baseline(config, store, STABLE)
    baseline = store.load("prod", "extract_invoice")
    assert baseline.pooled["semantic_distance"], "baseline must seed the variance pool"
    assert baseline.anchors["semantic_distance"].scale >= 0


# --- Informed re-baselining: --compare-previous -----------------------------


def test_compare_previous_is_absent_on_the_first_ever_baseline(env):
    """Nothing to compare against yet -- must say so via `previous_version`,
    not silently skip in a way that looks identical to the flag doing
    nothing."""
    config, store, _ = env
    written = run_baseline(config, store, STABLE, compare_previous=True)
    assert written[0].previous_version is None
    assert written[0].previous_comparison is None


def test_compare_previous_shows_what_changed_between_versions(env):
    config, store, _ = env
    run_baseline(config, store, STABLE)
    written = run_baseline(config, store, DRIFTED, compare_previous=True)
    captured = written[0]

    assert captured.previous_version == 1
    assert captured.baseline.version == 2
    assert captured.previous_comparison is not None
    assert captured.previous_comparison.moved  # the drifted text really did move


def test_compare_previous_is_off_by_default(env):
    """The common path -- an ordinary `stillsane baseline` -- must do none of
    this extra work unless explicitly asked."""
    config, store, _ = env
    run_baseline(config, store, STABLE)
    written = run_baseline(config, store, DRIFTED)  # compare_previous defaults False
    assert written[0].previous_version is None
    assert written[0].previous_comparison is None


def test_compare_previous_flags_a_config_change(env):
    """The common reason to reach for this flag: a prompt edit. The
    comparison must say the config moved, not let a real content difference
    read as the provider changing underneath an unrelated re-baseline.
    """
    config, store, _ = env
    run_baseline(config, store, STABLE)

    edited = Config.model_validate(
        {**CONFIG, "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}]}
    )
    written = run_baseline(edited, store, STABLE, compare_previous=True)
    assert written[0].config_changed is True


def test_compare_previous_does_not_flag_an_unchanged_config(env):
    config, store, _ = env
    run_baseline(config, store, STABLE)
    written = run_baseline(config, store, STABLE, compare_previous=True)
    assert written[0].config_changed is False


def test_compare_previous_never_touches_the_variance_pool(env):
    """Purely informational -- the comparison must not feed back into the
    pool or anchors the way an ordinary clean `check` would. Recomputing the
    same thing every re-baseline, growing sensitivity from a comparison that
    was never actually a measured check, would be exactly the kind of
    self-inflicted drift the rest of this engine is built to avoid.
    """
    config, store, _ = env
    run_baseline(config, store, STABLE)
    before = store.load("prod", "extract_invoice", version=1)
    run_baseline(config, store, STABLE, compare_previous=True)
    after = store.load("prod", "extract_invoice", version=1)
    assert after.pooled == before.pooled
    assert after.anchors == before.anchors


def test_compare_previous_survives_a_previous_version_with_no_usable_samples(env):
    """Rare (it would have failed to capture at all), but `store.load` can in
    principle return a baseline whose `usable` is empty -- the comparison
    must decline gracefully rather than crash on an empty sample list.
    """
    config, store, _ = env
    run_baseline(config, store, STABLE)
    baseline_v1 = store.load("prod", "extract_invoice", version=1)
    for s in baseline_v1.samples:
        s.error = "simulated: no usable samples"
    # Overwrite v1's samples.jsonl directly -- there is no public API for
    # writing a broken baseline on purpose, and there should not be one.
    store._write(store._dir("prod", "extract_invoice") / "v1", baseline_v1)

    written = run_baseline(config, store, STABLE, compare_previous=True)
    assert written[0].previous_version == 1
    assert written[0].previous_comparison is None


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


def test_watch_fingerprint_false_ignores_a_changed_fingerprint(env):
    """`watch_fingerprint: false` is the documented escape hatch for a provider
    whose fingerprint churns for reasons that are not drift. It used to be read
    nowhere: a changed fingerprint still warned regardless of the setting.
    """
    config, store, history = env
    config = Config.model_validate(
        {**CONFIG, "targets": [{**CONFIG["targets"][0], "watch_fingerprint": False}]}
    )
    run_baseline(config, store, STABLE, fingerprint="fp_old")
    result = run_check(config, store, history, STABLE, fingerprint="fp_new")

    assert result.level is Level.PASS
    assert not any(s.signal == "fingerprint" for s in result.probes[0].signals)


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


def test_against_stale_compares_anyway_and_caps_at_warn(env):
    """The escape hatch: a PR that edits a probe changes the config hash, so an
    ordinary `check` refuses to compare -- correct for a scheduled run, but it
    means the PR-gating workflow the README's own example invites can never
    actually catch anything, since the edit always looks like "recapture
    first" rather than a result. `--against-stale` runs the comparison
    anyway, but must never come back as DRIFT or ERROR -- the whole promise is
    "indicative", not "a real verdict".
    """
    config, store, history = env
    run_baseline(config, store, STABLE)

    edited = Config.model_validate(
        {**CONFIG, "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}]}
    )
    result = run_check(edited, store, history, DRIFTED, against_stale=True)

    assert result.level is Level.WARN
    probe = result.probes[0]
    assert probe.stale_comparison is True
    assert probe.level is Level.WARN


def test_against_stale_never_updates_the_pool(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    before = store.load("prod", "extract_invoice").pooled

    edited = Config.model_validate(
        {**CONFIG, "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}]}
    )
    run_check(edited, store, history, STABLE, against_stale=True)

    after = store.load("prod", "extract_invoice").pooled
    assert after == before


def test_against_stale_notice_is_on_every_line(env):
    config, store, history = env
    run_baseline(config, store, STABLE)

    edited = Config.model_validate(
        {**CONFIG, "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}]}
    )
    result = run_check(edited, store, history, DRIFTED, against_stale=True)
    text = render(result, verbose=True, colour=False)

    content_lines = [ln for ln in text.splitlines() if ln.strip()]
    assert content_lines, "expected a non-empty report"
    # Every line belongs to the stale-compared probe except the trailing
    # separator and summary, which are global to the whole run.
    assert all("indicative" in ln for ln in content_lines[:-2]), text


def test_against_stale_caps_every_signal_not_just_the_aggregate(env):
    """`verdict.level` used to get capped to WARN while the individual
    `SignalVerdict.level` values underneath it stayed at DRIFT -- so the JSON
    payload's `moved[].level` (and a coloured terminal line, which keys off
    the signal's own level) could still say "drift" for a signal, directly
    under a probe whose own level claimed "warn". A machine reading the
    per-signal level rather than the headline would see exactly the real
    drift verdict this flag exists to rule out.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)

    edited = Config.model_validate(
        {**CONFIG, "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}]}
    )
    result = run_check(edited, store, history, DRIFTED, against_stale=True)
    probe = result.probes[0]

    assert probe.level is Level.WARN
    assert all(sv.level.rank <= Level.WARN.rank for sv in probe.signals), [
        (sv.signal, sv.level) for sv in probe.signals
    ]

    data = payload_for(result)
    assert all(m["level"] != "drift" for m in data["probes"][0]["moved"])


def test_an_ordinary_check_is_unaffected_by_the_against_stale_flag(env):
    """The flag must never change behaviour for a probe whose config hash
    still matches -- it only ever widens what a *mismatched* hash does.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED, against_stale=True)
    assert result.probes[0].stale_comparison is False
    assert result.level is Level.DRIFT


# --- Per-run samples -------------------------------------------------------


def test_check_persists_current_run_samples_under_the_history_run_id(env, tmp_path):
    """`history` records numbers; the text the model actually wrote used to
    exist only in whatever log captured that run's stdout. `check` now keeps
    it, keyed by the same `run_id` `history.record` assigns, so it can be
    found later without needing the original log.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    run_samples = RunSampleStore(tmp_path)

    async def go():
        async with make_client(DRIFTED) as client:
            return await check(config, store, history, client=client, run_samples=run_samples)

    result = asyncio.run(go())
    run_id = history.recent(limit=1)[0][0]

    loaded = run_samples.load(run_id)
    assert loaded is not None
    assert len(loaded) == config.probes[0].check_samples
    assert all(s.text in DRIFTED for s in loaded)
    assert result.probes[0].probe_id == loaded[0].probe_id == "extract_invoice"


def test_check_without_a_run_sample_store_does_not_error(env):
    """`run_samples` is optional -- a caller that does not pass one (or code
    running before this feature existed) must see no change in behaviour.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)
    assert result.level is Level.PASS


def test_history_run_prints_what_the_model_said(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    assert cli.main(["-c", str(config_path), "check"]) == 1
    capsys.readouterr()

    code = cli.main(["-c", str(config_path), "history"])
    assert code == 0
    history_out = capsys.readouterr().out
    run_id = history_out.splitlines()[1].split()[-1]

    code = cli.main(["-c", str(config_path), "history", "--run", run_id])
    assert code == 0
    out = capsys.readouterr().out
    assert "extract_invoice @ prod" in out
    assert any(text.splitlines()[0] in out for text in DRIFTED)


def test_history_run_reports_a_missing_id_clearly(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    code = cli.main(["-c", str(config_path), "history", "--run", "does-not-exist"])
    assert code == 1
    assert "does-not-exist" in capsys.readouterr().err


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


def test_a_long_signal_name_still_aligns_its_columns(env):
    """`has_keys[total,due_date]` is 24 characters -- the flagship example's
    own check -- and used to overflow the report's narrower signal-name
    column, shifting that one row's observed/band/effect values out of line
    with every other signal in the same report.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)
    text = render(result, verbose=True, colour=False)

    rows = [
        line for line in text.splitlines()
        if line.strip().startswith(("valid_json", "has_keys["))
    ]
    assert len(rows) == 2
    band_starts = {line.index("band") for line in rows}
    assert len(band_starts) == 1, f"columns do not line up: {rows}"


def test_report_never_emits_escape_codes_when_colour_is_off(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)
    assert "\033[" not in render(result, colour=False)


# --- Run cost ---------------------------------------------------------------


def _client_with_cost(cost):
    """A fake provider reporting a per-call cost, or none at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"message": {"content": "stable text"}, "finish_reason": "stop"}]}
        if cost is not None:
            body["usage"] = {"cost": cost}
        return httpx.Response(200, json=body)

    return _RealAsyncClient(transport=httpx.MockTransport(handler))


def test_run_cost_footer_never_emits_escape_codes_when_colour_is_off(env):
    """The existing escape-code test never exercises this line: nothing in
    the shared `make_client` fixture reports a cost, so `_cost_footer`
    returns `None` for every other test in this suite and the dimmed-text
    path (`paint.dim`) went untested. A plain substring check on the cost
    line would not have caught a leak here either -- `"this run: ..." in
    text` still holds even if ANSI codes wrapped around it -- so this checks
    the whole rendered report, the same way the existing test does.
    """
    config, store, history = env

    async def go_baseline():
        async with _client_with_cost(0.001) as client:
            return await capture_baseline(config, store, client=client)

    asyncio.run(go_baseline())

    async def go_check():
        async with _client_with_cost(0.0041) as client:
            return await check(config, store, history, client=client)

    result = asyncio.run(go_check())
    text = render(result, colour=False)
    assert "this run" in text  # guard against a vacuous pass
    assert "\033[" not in text


def test_run_cost_is_summed_in_the_footer(env):
    """"Near-zero running cost" is a claim the reader cannot check from PASS
    or an exit code alone. `Sample.cost_usd` is already populated wherever a
    gateway reports it; nothing summed it for the run before.
    """
    config, store, history = env

    async def go_baseline():
        async with _client_with_cost(0.001) as client:
            return await capture_baseline(config, store, client=client)

    asyncio.run(go_baseline())

    async def go_check():
        async with _client_with_cost(0.0041) as client:
            return await check(config, store, history, client=client)

    result = asyncio.run(go_check())
    text = render(result, colour=False)
    assert f"this run: {config.probes[0].check_samples} calls, $0.0123" in text

    data = payload_for(result)
    assert data["calls"] == config.probes[0].check_samples
    assert data["cost_usd"] == pytest.approx(0.0123)
    assert data["cost_known_calls"] == config.probes[0].check_samples


def test_run_cost_is_omitted_when_nothing_reports_one(env):
    """Most gateways never report cost. Printing a confident `$0.0000` would
    be exactly the fabricated number this report's own formatting rule
    exists to rule out, so the line must not appear at all.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)

    assert "this run" not in render(result, colour=False)
    assert payload_for(result)["cost_usd"] is None
    assert payload_for(result)["cost_known_calls"] == 0


def test_run_cost_says_when_it_is_a_partial_sum(env):
    """Some calls priced, some did not -- a gateway that only prices certain
    responses, say. Mixing known and unknown costs into one number would
    read as the run's total; it must say it is not.
    """
    config, store, history = env

    async def go_baseline():
        async with _client_with_cost(0.001) as client:
            return await capture_baseline(config, store, client=client)

    asyncio.run(go_baseline())

    costs = iter([0.004, 0.004, None])

    def handler(request: httpx.Request) -> httpx.Response:
        cost = next(costs, None)
        body = {"choices": [{"message": {"content": "stable text"}, "finish_reason": "stop"}]}
        if cost is not None:
            body["usage"] = {"cost": cost}
        return httpx.Response(200, json=body)

    async def go_check():
        async with _RealAsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check(config, store, history, client=client)

    result = asyncio.run(go_check())
    text = render(result, colour=False)
    assert "this run: $0.0080 across 2 of 3 calls" in text
    assert "this run: 3 calls" not in text

    data = payload_for(result)
    assert data["cost_usd"] == pytest.approx(0.008)
    assert data["cost_known_calls"] == 2
    assert data["calls"] == 3


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


def test_alert_payload_carries_retries(env):
    """The field the text report and `status` already show, now in the JSON too.

    `stillsane check --json` is what a CI pipeline actually parses. A run that only
    passed because a transport failure was retried is still evidence the
    environment is unwell, and that was invisible to anything reading `--json`
    even though a human running the same check would see it on screen.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)
    result.probes[0].retries = 2

    payload = payload_for(result)
    assert payload["probes"][0]["retries"] == 2
    json.dumps(payload)


def test_alert_payload_retries_defaults_to_zero(env):
    """A clean run with no retries should not need special-casing to read `0`."""
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)

    assert payload_for(result)["probes"][0]["retries"] == 0


def test_slack_payload_is_bounded(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)
    text = slack_payload(result)["text"]
    assert "stillsane: DRIFT" in text and len(text) < 4000


# --- "Since when": first_seen / consecutive_runs ----------------------------


def test_consecutive_runs_climbs_across_repeated_drift(env):
    """The case the whole feature is for: daily cron, drift on Monday, the
    same drift Tuesday through Friday -- each run should be able to say how
    many days this specific probe has been non-PASS, not just repeat day
    one's numbers under a new timestamp.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)

    first = run_check(config, store, history, DRIFTED)
    p1 = first.probes[0]
    assert p1.consecutive_runs == 1
    assert p1.first_seen is not None

    second = run_check(config, store, history, DRIFTED)
    p2 = second.probes[0]
    assert p2.consecutive_runs == 2
    # Pinned to when the streak *started*, not the most recent run.
    assert p2.first_seen == p1.first_seen

    third = run_check(config, store, history, DRIFTED)
    p3 = third.probes[0]
    assert p3.consecutive_runs == 3
    assert p3.first_seen == p1.first_seen


def test_a_clean_run_resets_the_streak(env):
    """A recovery must not leave the next drift reading as "day 3" when it is
    really day one of a new incident -- the streak is about an *unbroken* run
    of non-PASS results, not a lifetime total.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)

    run_check(config, store, history, DRIFTED)
    drifted_twice = run_check(config, store, history, DRIFTED)
    assert drifted_twice.probes[0].consecutive_runs == 2

    clean = run_check(config, store, history, STABLE)
    assert clean.probes[0].consecutive_runs == 0
    assert clean.probes[0].first_seen is None

    drifted_again = run_check(config, store, history, DRIFTED)
    assert drifted_again.probes[0].consecutive_runs == 1
    # Not asserting `first_seen` differs from the earlier streak's: history
    # timestamps are second-resolution, and this whole sequence can run
    # inside one second in a fast test -- see `History.record`'s own note on
    # `rowid` breaking ties it never exposes to a reader. `consecutive_runs`
    # resetting to 1 is what actually proves the streak restarted.


def test_a_passing_probe_never_gets_a_streak(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, STABLE)
    assert result.probes[0].consecutive_runs == 0
    assert result.probes[0].first_seen is None


def test_alert_payload_carries_since_when(env):
    config, store, history = env
    run_baseline(config, store, STABLE)
    run_check(config, store, history, DRIFTED)
    result = run_check(config, store, history, DRIFTED)

    row = payload_for(result)["probes"][0]
    assert row["consecutive_runs"] == 2
    assert row["first_seen"] is not None
    json.dumps(payload_for(result))


def test_slack_headline_shows_the_day_count_once_it_repeats(env):
    config, store, history = env
    run_baseline(config, store, STABLE)

    day_one = run_check(config, store, history, DRIFTED)
    # Day one is not itself news that this is "day one" -- nobody needs to be
    # told the obvious on the first alert.
    assert "(day" not in slack_payload(day_one)["text"].splitlines()[0]

    run_check(config, store, history, DRIFTED)
    day_three = run_check(config, store, history, DRIFTED)
    headline = slack_payload(day_three)["text"].splitlines()[0]
    assert "(day 3)" in headline


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


def test_a_malformed_webhook_url_does_not_crash_the_check(env, monkeypatch, capsys):
    """`httpx.InvalidURL` is not an `httpx.HTTPError` subclass, so a stray control
    character or an unterminated IPv6-literal bracket in a configured webhook URL
    raised straight through `send()` and crashed the whole `check` invocation on a
    run that had already produced a valid verdict -- exactly the outcome this
    module's own docstring says a broken sink must never cause.
    """
    config, store, history = env
    run_baseline(config, store, STABLE)
    result = run_check(config, store, history, DRIFTED)

    def explode(url, **kw):
        raise httpx.InvalidURL("no host supplied")

    monkeypatch.setattr("stillsane.alerts.httpx.post", explode)
    send(result, "https://[::1", None)  # must not raise

    assert "could not deliver alert" in capsys.readouterr().err


# --- Attribution: is it my app or the model? --------------------------------


_ATTRIBUTION_CONFIG = {
    "embedder": "hashing",
    "targets": [
        {
            "name": "prod",
            "base_url": "https://app.example.com/v1",
            "model": "some-model",
            "attribute_to": "raw",
        },
        {"name": "raw", "base_url": "https://api.example.com/v1", "model": "some-model"},
    ],
    "probes": [
        {
            "id": "extract_invoice",
            "targets": ["prod", "raw"],
            "prompt": "Extract the total and due date as JSON.",
            "baseline_samples": 5,
            "check_samples": 3,
            "checks": ["valid_json", {"has_keys": ["total", "due_date"]}],
        }
    ],
}


def _routed_client(prod_texts, raw_texts, fingerprint="fp_a4f2b1"):
    """A fake provider that answers differently depending on which target's
    `base_url` the request went to, so `prod` and `raw` can be driven
    independently within one run."""
    prod_cycle = itertools.cycle(prod_texts)
    raw_cycle = itertools.cycle(raw_texts)

    def handler(request: httpx.Request) -> httpx.Response:
        content = next(raw_cycle if "api.example.com" in str(request.url) else prod_cycle)
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


def _attribution_env(tmp_path):
    config = Config.model_validate(_ATTRIBUTION_CONFIG)
    return config, BaselineStore(tmp_path), History(tmp_path)


def _attribution_baseline(config, store):
    async def go():
        async with _routed_client(STABLE, STABLE) as client:
            return await capture_baseline(config, store, client=client)

    return asyncio.run(go())


def _attribution_check(config, store, history, prod_texts, raw_texts):
    async def go():
        async with _routed_client(prod_texts, raw_texts) as client:
            return await check(config, store, history, client=client)

    return asyncio.run(go())


def test_attribution_points_at_the_app_when_the_control_held_steady(tmp_path):
    """The report's own motivating case: the app moved, the raw model on the
    same probe text did not -- the change is local to the app."""
    config, store, history = _attribution_env(tmp_path)
    _attribution_baseline(config, store)
    result = _attribution_check(config, store, history, DRIFTED, STABLE)

    by_target = {p.target_name: p for p in result.probes}
    assert by_target["prod"].level is not Level.PASS
    assert by_target["raw"].level is Level.PASS
    assert by_target["prod"].attribution is not None
    assert "did not move" in by_target["prod"].attribution
    assert "local to 'prod'" in by_target["prod"].attribution
    # The control's own verdict never gets a note about itself.
    assert by_target["raw"].attribution is None


def test_attribution_points_at_the_provider_when_both_moved(tmp_path):
    config, store, history = _attribution_env(tmp_path)
    _attribution_baseline(config, store)
    result = _attribution_check(config, store, history, DRIFTED, DRIFTED)

    by_target = {p.target_name: p for p in result.probes}
    assert by_target["prod"].attribution is not None
    assert "moved too" in by_target["prod"].attribution
    assert "provider-side change" in by_target["prod"].attribution


def test_no_attribution_when_the_app_itself_did_not_move(tmp_path):
    config, store, history = _attribution_env(tmp_path)
    _attribution_baseline(config, store)
    result = _attribution_check(config, store, history, STABLE, STABLE)

    by_target = {p.target_name: p for p in result.probes}
    assert by_target["prod"].level is Level.PASS
    assert by_target["prod"].attribution is None


def test_no_attribution_when_the_control_has_no_baseline(tmp_path):
    """A control that never had `stillsane baseline` run against it errors on
    its own comparison -- nothing to attribute against, so the line must stay
    off rather than treating an ERROR as "did not move"."""
    config = Config.model_validate(_ATTRIBUTION_CONFIG)
    store = BaselineStore(tmp_path)
    history = History(tmp_path)

    # Only capture a baseline for `prod`, never for `raw`.
    only_prod = Config.model_validate(
        {**_ATTRIBUTION_CONFIG, "probes": [{**_ATTRIBUTION_CONFIG["probes"][0], "targets": ["prod"]}]}
    )

    async def go_baseline():
        async with _routed_client(STABLE, STABLE) as client:
            return await capture_baseline(only_prod, store, client=client)

    asyncio.run(go_baseline())
    result = _attribution_check(config, store, history, DRIFTED, STABLE)

    by_target = {p.target_name: p for p in result.probes}
    assert by_target["prod"].level is not Level.PASS
    assert by_target["raw"].level is Level.ERROR
    assert by_target["prod"].attribution is None


def test_no_attribution_when_the_probe_is_not_scoped_to_the_control(tmp_path):
    """`attribute_to` names a target, but the probe still has to actually run
    against it -- see the README: "both, or there is nothing to compare
    against". A probe scoped only to `prod` gets no control run to compare.
    """
    scoped = Config.model_validate(
        {**_ATTRIBUTION_CONFIG, "probes": [{**_ATTRIBUTION_CONFIG["probes"][0], "targets": ["prod"]}]}
    )
    store = BaselineStore(tmp_path)
    history = History(tmp_path)

    async def go_baseline():
        async with _routed_client(STABLE, STABLE) as client:
            return await capture_baseline(scoped, store, client=client)

    asyncio.run(go_baseline())

    async def go_check():
        async with _routed_client(DRIFTED, STABLE) as client:
            return await check(scoped, store, history, client=client)

    result = asyncio.run(go_check())
    assert len(result.probes) == 1
    assert result.probes[0].target_name == "prod"
    assert result.probes[0].attribution is None


def test_attribution_line_appears_in_the_text_report(tmp_path):
    config, store, history = _attribution_env(tmp_path)
    _attribution_baseline(config, store)
    result = _attribution_check(config, store, history, DRIFTED, STABLE)
    text = render(result, colour=False)
    assert "-> attribution:" in text
    assert "local to 'prod'" in text


def test_attribution_is_in_the_alert_payload(tmp_path):
    config, store, history = _attribution_env(tmp_path)
    _attribution_baseline(config, store)
    result = _attribution_check(config, store, history, DRIFTED, STABLE)
    payload = payload_for(result)
    row = next(p for p in payload["probes"] if p["target"] == "prod")
    assert row["attribution"] is not None
    assert "did not move" in row["attribution"]
    control_row = next(p for p in payload["probes"] if p["target"] == "raw")
    assert control_row["attribution"] is None
    json.dumps(payload)


class _Ok:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


# --- repeat_every: suppressing a resend of an unchanged verdict ------------


def _verdict(level=Level.DRIFT, consecutive_runs=1):
    from stillsane.models import ProbeVerdict

    return ProbeVerdict(
        probe_id="p", target_name="t", level=level, consecutive_runs=consecutive_runs
    )


def test_should_alert_a_passing_probe_never_alerts():
    assert not should_alert(_verdict(Level.PASS, consecutive_runs=0), repeat_every=None)
    assert not should_alert(_verdict(Level.PASS, consecutive_runs=0), repeat_every=0)


def test_should_alert_default_none_always_sends_regardless_of_streak_length():
    for n in (1, 2, 5, 100):
        assert should_alert(_verdict(consecutive_runs=n), repeat_every=None)


def test_should_alert_a_fresh_verdict_always_sends_even_under_suppression():
    """`consecutive_runs <= 1` is "this just happened" -- suppression must
    never delay the very first notice of a new problem."""
    assert should_alert(_verdict(consecutive_runs=1), repeat_every=0)
    assert should_alert(_verdict(consecutive_runs=1), repeat_every=5)


def test_should_alert_zero_suppresses_every_later_repeat():
    assert should_alert(_verdict(consecutive_runs=1), repeat_every=0)
    for n in (2, 3, 10):
        assert not should_alert(_verdict(consecutive_runs=n), repeat_every=0)


def test_should_alert_n_resends_every_n_runs_into_the_streak():
    # day 1 (sent), 2 (suppressed), 3 (sent -- every 2 runs), 4 (suppressed), 5 (sent)
    expected = {1: True, 2: False, 3: True, 4: False, 5: True}
    for day, want in expected.items():
        assert should_alert(_verdict(consecutive_runs=day), repeat_every=2) is want


def test_repeat_every_default_sends_on_every_run(tmp_path, monkeypatch, capsys):
    """The safe default: unset `repeat_every` must not suppress anything, or
    a config nobody has touched would start going quiet.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(
        yaml.safe_dump({**CONFIG, "alerts": {"webhook": "https://example.com/hook"}})
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    posted = []
    monkeypatch.setattr(
        "stillsane.alerts.httpx.post", lambda url, **kw: posted.append(url) or _Ok()
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    cli.main(["-c", str(config_path), "check"])
    cli.main(["-c", str(config_path), "check"])
    capsys.readouterr()
    assert len(posted) == 2


def test_repeat_every_zero_suppresses_a_repeat_but_never_the_first_alert(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {**CONFIG, "alerts": {"webhook": "https://example.com/hook", "repeat_every": 0}}
        )
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    posted = []
    monkeypatch.setattr(
        "stillsane.alerts.httpx.post", lambda url, **kw: posted.append(url) or _Ok()
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    cli.main(["-c", str(config_path), "check"])  # day 1: new verdict, must alert
    assert len(posted) == 1
    out1 = capsys.readouterr()
    assert "suppressed" not in out1.err

    cli.main(["-c", str(config_path), "check"])  # day 2: unchanged, must suppress
    assert len(posted) == 1  # still just the one from day 1
    out2 = capsys.readouterr()
    assert "alert suppressed" in out2.err
    assert "repeat_every=0" in out2.err
    assert "extract_invoice @ prod" in out2.err
    assert "day 2" in out2.err

    # The check itself is unaffected by suppression -- still reports DRIFT and
    # exits non-zero. Suppression only concerns the notification, never the
    # verdict or the exit code.
    code = cli.main(["-c", str(config_path), "check"])
    assert code == 1
    assert "DRIFT" in capsys.readouterr().out


def test_repeat_every_n_resends_periodically(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {**CONFIG, "alerts": {"webhook": "https://example.com/hook", "repeat_every": 2}}
        )
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    posted = []
    monkeypatch.setattr(
        "stillsane.alerts.httpx.post", lambda url, **kw: posted.append(url) or _Ok()
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    # day 1 (sent), day 2 (suppressed), day 3 (resent -- every 2 runs)
    for _ in range(3):
        cli.main(["-c", str(config_path), "check"])
    capsys.readouterr()
    assert len(posted) == 2


def test_a_recovery_then_new_drift_always_alerts_even_under_suppression(
    tmp_path, monkeypatch, capsys
):
    """Suppression is about not repeating news the reader already has -- a
    probe that recovered and then broke again is news, regardless of
    `repeat_every`, and must never be silently folded into the earlier streak.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {**CONFIG, "alerts": {"webhook": "https://example.com/hook", "repeat_every": 0}}
        )
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    posted = []
    monkeypatch.setattr(
        "stillsane.alerts.httpx.post", lambda url, **kw: posted.append(url) or _Ok()
    )

    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    cli.main(["-c", str(config_path), "check"])  # day 1: sent
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    cli.main(["-c", str(config_path), "check"])  # recovered: no alert to suppress
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    cli.main(["-c", str(config_path), "check"])  # new incident, day 1 again: must send
    capsys.readouterr()
    assert len(posted) == 2


def test_no_configured_destination_prints_no_suppression_noise(tmp_path, monkeypatch, capsys):
    """A config with no webhook/slack_webhook at all must behave exactly as
    it always has -- `repeat_every` is meaningless with nothing to send to,
    and must not start printing a stderr line for every ordinary user who has
    never configured alerts.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump({**CONFIG, "alerts": {"repeat_every": 0}}))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    cli.main(["-c", str(config_path), "check"])
    cli.main(["-c", str(config_path), "check"])
    err = capsys.readouterr().err
    assert "suppressed" not in err


def test_fail_on_warn_promotes_the_exit_code(env):
    config, store, history = env
    run_baseline(config, store, STABLE, fingerprint="fp_old")
    result = run_check(config, store, history, STABLE, fingerprint="fp_new")

    assert exit_code_for(result, fail_on_warn=False) == 2
    assert exit_code_for(result, fail_on_warn=True) == 1


def test_against_stale_exit_code_ignores_fail_on_warn(tmp_path, monkeypatch):
    """`--against-stale` promises exit 0 or 2 only. `fail_on_warn` escalating a
    WARN it produced to the DRIFT exit code would break that promise for
    anyone who has `fail_on_warn: true` set for unrelated reasons.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(
        yaml.safe_dump({**CONFIG, "alerts": {"fail_on_warn": True}})
    )
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0

    edited = {
        **CONFIG,
        "alerts": {"fail_on_warn": True},
        "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}],
    }
    config_path.write_text(yaml.safe_dump(edited))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    code = cli.main(["-c", str(config_path), "check", "--against-stale"])
    assert code == 2


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


def test_config_flag_also_works_after_the_subcommand(tmp_path, capsys):
    """`-c/--config` lived only on the top-level parser, so `stillsane check
    --config foo.yaml` -- the ordering almost anyone types first -- failed with
    "unrecognized arguments" rather than the config-not-found message this test
    asserts on. Both orderings must reach the same, correct config path.
    """
    path = tmp_path / "nope.yaml"
    code = cli.main(["check", "--config", str(path)])
    assert code == 3
    assert "stillsane init" in capsys.readouterr().err


def test_config_flag_before_the_subcommand_is_not_reset_by_it(tmp_path, capsys):
    """A subparser sharing the `-c` action's underlying default-handling with the
    top-level parser could silently discard a `-c` already parsed before the
    subcommand and fall back to the default config path instead -- caught by
    exactly this ordering pointing somewhere a real config does not exist.
    """
    path = tmp_path / "nope.yaml"
    code = cli.main(["-c", str(path), "check"])
    assert code == 3
    err = capsys.readouterr().err
    assert "stillsane init" in err
    # Confirms the path that was actually looked up, not just that some error
    # fired: if a subparser default silently replaced it, this would name
    # `DEFAULT_CONFIG` ("stillsane.yaml") instead of the path given here.
    assert str(path) in err


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


def test_bands_flags_a_baseline_the_config_has_moved_past(tmp_path, monkeypatch, capsys):
    """`bands` used to inspect the *latest* baseline against the *current*
    config's checks without ever comparing hashes, so it could report "all
    bands look sound" on a baseline `check` is about to refuse outright --
    the two commands disagreeing about whether there was anything to worry
    about.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    assert cli.main(["-c", str(config_path), "bands"]) == 0
    assert "stale" not in capsys.readouterr().out

    edited = {
        **CONFIG,
        "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}],
    }
    config_path.write_text(yaml.safe_dump(edited))

    assert cli.main(["-c", str(config_path), "bands"]) == 0
    out = capsys.readouterr().out
    assert "stale" in out
    assert "check` will refuse" in out
    # A label, not a verdict -- `bands` must not turn a config edit into a
    # false "band will misreport" finding.
    assert "All bands look sound." in out


def test_compare_previous_cli_shows_the_diff_and_flags_the_config_change(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    edited = {
        **CONFIG,
        "probes": [{**CONFIG["probes"][0], "prompt": "A completely different ask."}],
    }
    config_path.write_text(yaml.safe_dump(edited))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(DRIFTED)
    )
    code = cli.main(["-c", str(config_path), "baseline", "--compare-previous"])
    out = capsys.readouterr().out

    # Purely informational: even a wild move between versions must not fail
    # the build -- `baseline` always exits 0 on a successful capture.
    assert code == 0
    assert "v1 -> v2, config changed" in out
    assert "compared to the version it replaced" in out


def test_compare_previous_cli_without_a_config_change(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    code = cli.main(["-c", str(config_path), "baseline", "--compare-previous"])
    out = capsys.readouterr().out
    assert code == 0
    assert "v1 -> v2" in out
    assert "config changed" not in out


def test_compare_previous_cli_first_baseline_says_theres_nothing_yet(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    code = cli.main(["-c", str(config_path), "baseline", "--compare-previous"])
    out = capsys.readouterr().out
    assert code == 0
    assert "nothing to compare against" in out


def test_compare_previous_flag_is_off_prints_nothing_extra(tmp_path, monkeypatch, capsys):
    """The default invocation must read exactly as it always has -- no new
    lines for anyone who has never touched this flag."""
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    out = capsys.readouterr().out
    assert "compared to the version it replaced" not in out
    assert "nothing to compare against" not in out


def test_bands_orders_output_the_same_way_check_does(tmp_path, monkeypatch, capsys):
    """`bands` used to iterate targets-then-probes while `check` and `baseline`
    both go probe-then-target via `config.pairs()`, so the same multi-target,
    multi-probe config listed its rows in a different order depending on
    which command you ran.
    """
    multi = {
        **CONFIG,
        "targets": [
            {**CONFIG["targets"][0], "name": "prod_a"},
            {**CONFIG["targets"][0], "name": "prod_b"},
        ],
        "probes": [
            {**CONFIG["probes"][0], "id": "probe_x"},
            {**CONFIG["probes"][0], "id": "probe_y"},
        ],
    }
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(multi))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )

    cli.main(["-c", str(config_path), "baseline"])
    capsys.readouterr()

    cli.main(["-c", str(config_path), "bands"])
    bands_order = [
        line.split()[0] for line in capsys.readouterr().out.splitlines()
        if line.startswith("probe_")
    ]

    cli.main(["-c", str(config_path), "check"])
    check_order = [
        line.split()[1] for line in capsys.readouterr().out.splitlines()
        if line.startswith(("PASS", "WARN", "DRIFT", "ERROR"))
    ]

    expected = ["probe_x", "probe_x", "probe_y", "probe_y"]
    assert bands_order == expected
    assert check_order == expected


def test_status_shows_the_unbounded_history_span(tmp_path, monkeypatch, capsys):
    """Wiring check for `cmd_status` -> `History.run_span()` -> `assess()`:
    the arithmetic itself is covered in `test_status.py`, this only confirms
    the CLI actually passes the real span through rather than, say, swapping
    the tuple order or leaving the new kwargs unset.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()
    for _ in range(3):
        cli.main(["-c", str(config_path), "check"])
    capsys.readouterr()

    assert cli.main(["-c", str(config_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "history since" in out
    assert "3 run(s) recorded" in out

    assert cli.main(["-c", str(config_path), "status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs_recorded_ever"] == 3
    assert payload["first_recorded"] is not None


def test_trend_without_a_baseline_is_an_error_not_a_crash(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    code = cli.main(["-c", str(config_path), "trend"])
    assert code == 1
    assert "No baselines captured yet" in capsys.readouterr().err


def test_trend_reads_real_history_end_to_end(tmp_path, monkeypatch, capsys):
    """Full pipeline smoke test: baseline, several clean checks, then `trend`
    reads what `check` actually wrote to `history.sqlite` and says something
    sane about it. The precise shift-detection arithmetic is covered directly
    in `test_trend.py`; this only exercises the wiring between `bands.inspect`
    (for today's reference band), `History.trend_window` (for the recorded
    rows) and the CLI/JSON output.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    for _ in range(6):
        assert cli.main(["-c", str(config_path), "check"]) == 0
    capsys.readouterr()

    assert cli.main(["-c", str(config_path), "trend"]) == 0
    out = capsys.readouterr().out
    assert "extract_invoice @ prod" in out
    # STABLE cycles the same handful of byte-similar texts throughout, so
    # nothing should look like a sustained shift.
    assert "SUSTAINED SHIFT" not in out

    assert cli.main(["-c", str(config_path), "trend", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["command"] == "trend"
    assert data["shifted"] == 0
    assert any(row["signal"] == "length_chars" for row in data["signals"])

    # --strict must not fail a build merely because nothing shifted.
    assert cli.main(["-c", str(config_path), "trend", "--strict"]) == 0


def test_trend_is_scoped_to_the_current_baseline_version(tmp_path, monkeypatch, capsys):
    """Recapturing must reset what `trend` compares against -- runs recorded
    under the old baseline must not be visible to a signal's history once a
    new version is on disk.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()
    for _ in range(4):
        cli.main(["-c", str(config_path), "check"])
    capsys.readouterr()

    # Recapture: a new baseline version, with no check runs against it yet.
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    cli.main(["-c", str(config_path), "trend", "--json"])
    data = json.loads(capsys.readouterr().out)
    # Every signal reads as missing -- v1's 4 runs are invisible to v2.
    assert data["signals"] == []
    assert any(row["n"] == 0 for row in data["missing"])


def test_missing_api_key_is_an_error_not_a_traceback(tmp_path, monkeypatch, capsys):
    """A missing `api_key_env` used to raise `RuntimeError` from `build_request`,
    uncaught anywhere, and an uncaught exception exits 1 -- the DRIFT code. A CI
    job with a misconfigured secret would read that as "the model drifted"
    rather than "the job is broken", the exact confusion this tool exists to
    prevent.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    capsys.readouterr()

    keyed = {**CONFIG, "targets": [{**CONFIG["targets"][0], "api_key_env": "STILLSANE_TEST_MISSING_KEY"}]}
    config_path.write_text(yaml.safe_dump(keyed))
    monkeypatch.delenv("STILLSANE_TEST_MISSING_KEY", raising=False)

    code = cli.main(["-c", str(config_path), "check"])
    assert code == 3
    assert "STILLSANE_TEST_MISSING_KEY" in capsys.readouterr().out


def test_an_invalid_config_is_an_error_not_a_traceback(tmp_path, capsys):
    """A config that fails pydantic validation used to raise `ValidationError`
    from inside `_load`, uncaught in `main`, which exits 1 -- the DRIFT code --
    for a file that was never even loaded.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump({**CONFIG, "state_dir": 123}))

    code = cli.main(["-c", str(config_path), "check"])
    assert code == 3
    assert "stillsane:" in capsys.readouterr().err


def test_cli_check_rejects_an_unknown_probe(tmp_path, monkeypatch, capsys):
    """A typo'd `--probe` must not silently no-op and report success.

    Before this, an empty `only` match produced a `RunResult` with no probes,
    which rendered as "No probes ran" at exit 0 -- a config that genuinely does
    define a probe, reporting PASS on an invocation that checked nothing. A CI
    step gating on this exit code would have read the typo as a clean run.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    cli.main(["-c", str(config_path), "baseline"])
    capsys.readouterr()

    code = cli.main(["-c", str(config_path), "check", "--probe", "nope"])
    assert code == 3  # ERROR, not the exit 0 this used to silently return
    assert "No probes matched: nope" in capsys.readouterr().err


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


#: Two probes, so `calibrate --probe` has something real to filter between. Both
#: report `length_chars`, deliberately: that shared name is exactly what the
#: per-probe aggregation fix exists to keep separate.
TWO_PROBE_CONFIG = {
    "embedder": "hashing",
    "targets": [
        {"name": "prod", "base_url": "https://api.example.com/v1", "model": "some-model"}
    ],
    "probes": [
        {
            "id": "extract_invoice",
            "prompt": "Extract the total and due date as JSON.",
            "baseline_samples": 5,
            "check_samples": 3,
            "checks": ["valid_json", {"has_keys": ["total", "due_date"]}],
        },
        {
            "id": "summarise",
            "prompt": "Summarise this incident in three sentences.",
            "baseline_samples": 5,
            "check_samples": 3,
        },
    ],
}


def test_cli_calibrate_scopes_to_one_probe(tmp_path, monkeypatch, capsys):
    """`--probe` on `calibrate` filters rows the same way it does on `check`/`bands`.

    Uses a real two-probe config specifically because both probes report
    `length_chars`: the wiring being tested is that the CLI passes the filter down
    correctly, on top of the aggregation fix that keeps same-named signals apart.
    """
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(TWO_PROBE_CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )

    assert cli.main(["-c", str(config_path), "baseline"]) == 0
    assert cli.main(["-c", str(config_path), "check"]) == 0
    capsys.readouterr()

    assert cli.main(["-c", str(config_path), "calibrate", "--probe", "summarise"]) == 0
    out = capsys.readouterr().out
    assert "summarise @ prod" in out
    assert "extract_invoice" not in out


def test_cli_calibrate_rejects_an_unknown_probe(tmp_path, monkeypatch, capsys):
    """A typo must read as a typo, not as "you have no data at all"."""
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(TWO_PROBE_CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    cli.main(["-c", str(config_path), "baseline"])
    cli.main(["-c", str(config_path), "check"])
    capsys.readouterr()

    code = cli.main(["-c", str(config_path), "calibrate", "--probe", "nope"])
    assert code == 1
    assert "No probes matched: nope" in capsys.readouterr().err


def test_cli_calibrate_distinguishes_no_clean_runs_from_unknown_probe(
    tmp_path, monkeypatch, capsys
):
    """A probe that exists but has never had a clean run needs a different message
    than a typo, or the fix looks the same as the problem."""
    config_path = tmp_path / "stillsane.yaml"
    config_path.write_text(yaml.safe_dump(TWO_PROBE_CONFIG))
    monkeypatch.setattr(
        "stillsane.runner.httpx.AsyncClient", lambda *a, **k: make_client(STABLE)
    )
    # Baseline both, but only ever check one, so "summarise" has zero clean runs
    # recorded despite being a real probe in the config.
    cli.main(["-c", str(config_path), "baseline"])
    cli.main(["-c", str(config_path), "check", "--probe", "extract_invoice"])
    capsys.readouterr()

    code = cli.main(["-c", str(config_path), "calibrate", "--probe", "summarise"])
    assert code == 1
    err = capsys.readouterr().err
    assert "No clean runs recorded yet for: summarise" in err
    assert "No probes matched" not in err


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


def test_raw_response_bodies_are_not_persisted_by_default(env):
    """Nothing in stillsane reads `Sample.raw`, and the README tells people to
    commit `.stillsane/baselines/` to git -- for `type: http` against a user's
    own app, persisting it by default means whatever the API actually
    returned, tenant data included, lands in version-control history that is
    not easily purged. `store_raw` defaults to off, so a fresh capture must
    not write it to disk.
    """
    config, store, _ = env
    run_baseline(config, store, STABLE)
    reloaded = store.load("prod", "extract_invoice")
    assert all(s.raw == {} for s in reloaded.samples)


def test_store_raw_opts_a_target_into_persisting_response_bodies(env):
    config, store, _ = env
    opted_in = Config.model_validate(
        {**CONFIG, "targets": [{**CONFIG["targets"][0], "store_raw": True}]}
    )
    run_baseline(opted_in, store, STABLE)
    reloaded = store.load("prod", "extract_invoice")
    assert all(s.raw for s in reloaded.samples)
    assert reloaded.samples[0].raw.get("model") == "some-model"


def test_capture_names_floored_pointwise_signals(env):
    """The capture-time warning used to see only the distance signals.

    Before `capture_baseline` reused `bands.inspect` to fill `baseline.floored`,
    the equivalent capture-time code read from the pooled record, which by
    design holds pairwise distances only, so `length_chars` and
    `completion_tokens` could never be named however floored they were.
    Against a real baseline that meant one signal reported and three floored
    in fact, and the gap was invisible until `bands` recomputed them
    separately. `capture_baseline` now shares that exact code path, so this
    checks the outcome on the object it actually returns.
    """
    config, store, _ = env
    written = run_baseline(config, store, STABLE)
    captured = written[0]

    assert "length_chars" in captured.floored, (
        "a floored pointwise band must be nameable at capture time, not only by `bands`"
    )


def test_capture_ignores_signals_that_do_not_apply(env):
    """No token counts reported means nothing to name, not a crash."""
    config, store, _ = env
    run_baseline(config, store, STABLE)
    baseline = store.load("prod", "extract_invoice")

    from stillsane.bands import inspect as inspect_bands
    from stillsane.compare import BandConfig
    from stillsane.signals import HashingEmbedder, build_signals

    for sample in baseline.samples:
        sample.completion_tokens = None

    signals = build_signals(config.probes[0].checks, HashingEmbedder())
    report = inspect_bands(baseline, signals, BandConfig(), config.probes[0].check_samples)
    named = [sb.signal for sb in report.signals if sb.band.floored]
    assert "completion_tokens" not in named
