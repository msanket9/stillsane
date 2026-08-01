"""The LLM judge.

Two properties matter more than anything the judge actually says.

It must not run when nothing drifted, because that is what keeps a scheduled
monitor free. And it must not be able to take a run down when it fails, because
the verdict came from measurements and stands without it.
"""

from __future__ import annotations

import asyncio
import itertools

import httpx
import pytest

from stillsane.config import Config, JudgeConfig, ProbeConfig
from stillsane.judge import Judge, JudgeVerdict, apply, parse_verdict
from stillsane.models import Level, ProbeVerdict, SignalVerdict
from stillsane.runner import capture_baseline, check
from stillsane.store import BaselineStore

STABLE = [
    '{"total": 1240.50, "due_date": "2026-07-01"}',
    '{"total": 1240.5, "due_date": "2026-07-01"}',
    '{"due_date": "2026-07-01", "total": 1240.50}',
]
DRIFTED = [
    'Here you go!\n{"total": 1240.50, "due_date": "2026-07-01"}\nAnything else?',
    'Sure thing:\n{"total": 1240.5, "due_date": "2026-07-01"}\nGlad to help.',
]

BASE_CONFIG = {
    "embedder": "hashing",
    "targets": [{"name": "prod", "base_url": "https://api.example.com/v1", "model": "m"}],
    "probes": [
        {"id": "extract", "prompt": "Extract as JSON.", "checks": ["valid_json"]},
    ],
}
JUDGE_BLOCK = {"base_url": "https://judge.example.com/v1", "model": "judge-model"}

_RealAsyncClient = httpx.AsyncClient


def make_client(texts, judge_reply: str | None = None, judge_status: int = 200):
    """Routes probe traffic to a fake provider and judge traffic to a fake judge."""
    cycle = itertools.cycle(texts)
    calls = {"probe": 0, "judge": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "judge.example.com" in str(request.url):
            calls["judge"] += 1
            if judge_status != 200:
                return httpx.Response(judge_status, text="judge is down")
            return httpx.Response(
                200,
                json={
                    "model": "judge-model",
                    "choices": [
                        {"message": {"content": judge_reply or ""}, "finish_reason": "stop"}
                    ],
                },
            )
        calls["probe"] += 1
        content = next(cycle)
        return httpx.Response(
            200,
            json={
                "model": "m",
                "system_fingerprint": "fp_1",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(content) // 4},
            },
        )

    return _RealAsyncClient(transport=httpx.MockTransport(handler)), calls


def run(config, store, texts, judge_reply=None, judge_status=200, baseline=False):
    client, calls = make_client(texts, judge_reply, judge_status)

    async def go():
        async with client:
            if baseline:
                return await capture_baseline(config, store, client=client)
            return await check(config, store, client=client)

    return asyncio.run(go()), calls


@pytest.fixture
def env(tmp_path):
    store = BaselineStore(tmp_path)
    config = Config.model_validate({**BASE_CONFIG, "judge": JUDGE_BLOCK})
    run(config, store, STABLE, baseline=True)
    return config, store


GOOD_REPLY = (
    '{"changed": true, "severity": "breaking", '
    '"summary": "Output is still valid JSON but is now wrapped in prose."}'
)


# --- Cost: the judge must not run on a healthy day -------------------------


def test_the_judge_is_not_called_when_nothing_drifted(env):
    """The whole reason for tiering. A clean run must cost nothing extra."""
    config, store = env
    result, calls = run(config, store, STABLE, GOOD_REPLY)
    assert result.level is Level.PASS
    assert calls["judge"] == 0


def test_the_judge_is_called_once_per_drifting_probe(env):
    config, store = env
    result, calls = run(config, store, DRIFTED, GOOD_REPLY)
    assert result.level is Level.DRIFT
    assert calls["judge"] == 1


def test_no_judge_configured_means_no_judge_traffic(tmp_path):
    store = BaselineStore(tmp_path)
    config = Config.model_validate(BASE_CONFIG)  # no judge block
    run(config, store, STABLE, baseline=True)
    result, calls = run(config, store, DRIFTED)
    assert result.level is Level.DRIFT
    assert calls["judge"] == 0
    assert result.probes[0].judge_note is None


# --- What it contributes ---------------------------------------------------


def test_the_verdict_reaches_the_report(env):
    config, store = env
    result, _ = run(config, store, DRIFTED, GOOD_REPLY)
    note = result.probes[0].judge_note
    assert note.startswith("breaking: ")
    assert "wrapped in prose" in note


def test_the_judge_is_advisory_by_default(env):
    """It explains; it does not overrule a band learned from measurements."""
    config, store = env
    cosmetic = '{"changed": false, "severity": "cosmetic", "summary": "Only whitespace."}'
    result, _ = run(config, store, DRIFTED, cosmetic)
    assert result.level is Level.DRIFT
    assert "downgraded" not in result.probes[0].judge_note


def test_downgrading_is_opt_in(tmp_path):
    store = BaselineStore(tmp_path)
    config = Config.model_validate(
        {**BASE_CONFIG, "judge": {**JUDGE_BLOCK, "can_downgrade": True}}
    )
    run(config, store, STABLE, baseline=True)
    cosmetic = '{"changed": false, "severity": "cosmetic", "summary": "Only whitespace."}'
    result, _ = run(config, store, DRIFTED, cosmetic)

    assert result.level is Level.WARN
    assert result.exit_code == 2
    assert "downgraded from drift" in result.probes[0].judge_note


def test_downgrading_never_silences_a_breaking_change(tmp_path):
    """Even with downgrading on, only a judge that says "nothing changed" softens."""
    store = BaselineStore(tmp_path)
    config = Config.model_validate(
        {**BASE_CONFIG, "judge": {**JUDGE_BLOCK, "can_downgrade": True}}
    )
    run(config, store, STABLE, baseline=True)
    result, _ = run(config, store, DRIFTED, GOOD_REPLY)
    assert result.level is Level.DRIFT


# --- Failure is not fatal --------------------------------------------------


def test_a_dead_judge_leaves_the_verdict_intact(env):
    config, store = env
    result, calls = run(config, store, DRIFTED, judge_status=500)
    assert calls["judge"] == 1
    assert result.level is Level.DRIFT  # the measurement still stands
    assert result.probes[0].judge_note is None


def test_an_unparseable_judge_reply_is_ignored(env):
    config, store = env
    result, _ = run(config, store, DRIFTED, "I'm not sure, could you clarify?")
    assert result.level is Level.DRIFT
    assert result.probes[0].judge_note is None


# --- Parsing ---------------------------------------------------------------


def test_parses_a_clean_reply():
    v = parse_verdict(GOOD_REPLY)
    assert v.changed is True and v.severity == "breaking"


def test_parses_a_reply_wrapped_in_prose():
    """A judge that says "Sure, here you go:" is the exact failure this tool
    detects, so falling over on it would be absurd."""
    v = parse_verdict('Sure! Here is my assessment:\n' + GOOD_REPLY + '\nHope that helps.')
    assert v is not None and v.severity == "breaking"


def test_parses_a_fenced_reply():
    v = parse_verdict("```json\n" + GOOD_REPLY + "\n```")
    assert v is not None and v.severity == "breaking"


def test_accepts_changed_as_a_string():
    v = parse_verdict('{"changed": "yes", "severity": "cosmetic", "summary": "x"}')
    assert v.changed is True


def test_rejects_an_unknown_severity():
    assert parse_verdict('{"changed": true, "severity": "catastrophic", "summary": "x"}') is None


def test_rejects_a_missing_summary():
    assert parse_verdict('{"changed": true, "severity": "cosmetic"}') is None


def test_rejects_non_json():
    assert parse_verdict("no idea, sorry") is None


def test_a_long_summary_is_truncated():
    reply = '{"changed": true, "severity": "cosmetic", "summary": "%s"}' % ("x" * 500)
    assert len(parse_verdict(reply).summary) == 200


# --- Unit-level apply ------------------------------------------------------


def _verdict(level: Level) -> ProbeVerdict:
    return ProbeVerdict(
        probe_id="p",
        target_name="t",
        level=level,
        signals=[SignalVerdict(signal="semantic_distance", kind=None, level=level, detail="")],
    )


def test_apply_is_a_no_op_without_a_verdict():
    v = _verdict(Level.DRIFT)
    apply(v, None, can_downgrade=True)
    assert v.level is Level.DRIFT and v.judge_note is None


def test_apply_never_upgrades_a_warning():
    """The judge can soften, never sharpen. Escalation belongs to the band."""
    v = _verdict(Level.WARN)
    apply(v, JudgeVerdict(changed=True, severity="breaking", summary="bad"), can_downgrade=True)
    assert v.level is Level.WARN


def test_judge_builds_its_own_openai_target():
    """The judge points wherever it is configured, independent of the target it
    is judging -- otherwise a provider-side change moves both the thing being
    measured and the instrument measuring it."""
    judge = Judge(JudgeConfig(base_url="https://x/v1", model="m"))
    method, url, _, payload = judge._target.build_request(ProbeConfig(id="j", prompt="hi"))

    assert method == "POST" and url == "https://x/v1/chat/completions"
    assert payload["model"] == "m"
    assert payload["temperature"] == 0.0  # deterministic by default
