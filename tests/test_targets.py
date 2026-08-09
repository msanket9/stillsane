"""Target behaviour, against a mock transport. No network."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from stillsane.config import ProbeConfig, TargetConfig
from stillsane.targets import build_target, collect, dotted_get, render_template


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def oai_body(content="hello", **extra):
    body = {
        "model": "some-model",
        "system_fingerprint": "fp_abc123",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22},
    }
    body.update(extra)
    return body


def call(target, probe, handler):
    async def go():
        async with client_for(handler) as client:
            return await target.call(probe, client)

    return asyncio.run(go())


PROBE = ProbeConfig(id="p", prompt="say hello", system="be terse")
OAI = TargetConfig(name="prod", base_url="https://api.example.com/v1", model="some-model")

#: Retry behaviour with the wait taken out. The backoff is real and deliberate in
#: production, but a test suite that sleeps through it stops being run.
FAST_RETRY = OAI.model_copy(update={"retry_backoff_s": 0.0})
NO_RETRY = OAI.model_copy(update={"retries": 0})


# --- OpenAI-compatible ----------------------------------------------------


def test_extracts_the_fields_that_matter():
    target = build_target(OAI)
    sample = call(target, PROBE, lambda r: httpx.Response(200, json=oai_body("hi there")))

    assert sample.text == "hi there"
    assert sample.fingerprint == "fp_abc123"
    assert sample.model_id == "some-model"
    assert sample.completion_tokens == 22
    assert sample.prompt_tokens == 11
    assert sample.latency_ms is not None
    assert sample.ok


def test_request_shape_is_openai_chat_completions():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=oai_body())

    call(build_target(OAI), PROBE, handler)
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["body"]["model"] == "some-model"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "say hello"},
    ]


def test_base_url_trailing_slash_does_not_double_up():
    target = build_target(
        TargetConfig(name="t", base_url="https://api.example.com/v1/", model="m")
    )
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=oai_body())

    call(target, PROBE, handler)
    assert seen["url"] == "https://api.example.com/v1/chat/completions"


def test_api_key_becomes_a_bearer_header(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-secret")
    target = build_target(
        TargetConfig(name="t", base_url="https://x/v1", model="m", api_key_env="TEST_KEY")
    )
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=oai_body())

    call(target, PROBE, handler)
    assert seen["auth"] == "Bearer sk-secret"


def test_auth_header_is_configurable(monkeypatch):
    """Not every provider uses `Authorization: Bearer`.

    Anthropic's Messages API wants `x-api-key` with no prefix; Azure wants
    `api-key`. Hardcoding Bearer made those endpoints unreachable, and the only
    workaround would have been putting a live secret in `headers`, which the
    config is explicitly designed to avoid.
    """
    monkeypatch.setenv("TEST_KEY", "sk-secret")
    target = build_target(
        TargetConfig(
            name="anthropic",
            type="http",
            base_url="https://api.anthropic.com",
            path="/v1/messages",
            api_key_env="TEST_KEY",
            api_key_header="x-api-key",
            api_key_prefix="",
            headers={"anthropic-version": "2023-06-01"},
            body={"messages": [{"role": "user", "content": "{{prompt}}"}]},
            response_path="content.0.text",
        )
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}]})

    sample = call(target, PROBE, handler)
    assert seen["x-api-key"] == "sk-secret"
    assert "authorization" not in seen
    assert seen["anthropic-version"] == "2023-06-01"
    assert sample.text == "hi"


def test_bearer_remains_the_default(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-secret")
    target = build_target(
        TargetConfig(name="t", base_url="https://x/v1", model="m", api_key_env="TEST_KEY")
    )
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json=oai_body())

    call(target, PROBE, handler)
    assert seen["authorization"] == "Bearer sk-secret"


def test_tool_calls_are_normalised():
    body = oai_body(
        content=None,
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": '{"id": 4, "verbose": true}',
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    )
    sample = call(build_target(OAI), PROBE, lambda r: httpx.Response(200, json=body))
    assert sample.ok, sample.error
    assert [tc.signature() for tc in sample.tool_calls] == ["lookup(id,verbose)"]


def test_a_tool_only_reply_is_not_an_error():
    """For an agent probe, empty content with tool calls is the normal case."""
    body = oai_body(
        content=None,
        choices=[
            {
                "message": {"tool_calls": [{"function": {"name": "go", "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }
        ],
    )
    sample = call(build_target(OAI), PROBE, lambda r: httpx.Response(200, json=body))
    assert sample.ok and sample.text == ""


def test_a_genuinely_empty_reply_is_an_error():
    body = oai_body(content=None, choices=[{"message": {}, "finish_reason": "length"}])
    sample = call(build_target(OAI), PROBE, lambda r: httpx.Response(200, json=body))
    assert not sample.ok
    assert "length" in sample.error


def test_gateway_reported_cost_is_used_when_present():
    body = oai_body()
    body["usage"]["cost"] = 0.00042
    sample = call(build_target(OAI), PROBE, lambda r: httpx.Response(200, json=body))
    assert sample.cost_usd == pytest.approx(0.00042)


def test_cost_is_left_unknown_rather_than_guessed():
    sample = call(build_target(OAI), PROBE, lambda r: httpx.Response(200, json=oai_body()))
    assert sample.cost_usd is None


def test_alternative_fingerprint_field_names_are_read():
    body = oai_body()
    del body["system_fingerprint"]
    body["system_version"] = "build-99"
    sample = call(build_target(OAI), PROBE, lambda r: httpx.Response(200, json=body))
    assert sample.fingerprint == "build-99"


# --- Failure handling -----------------------------------------------------


def test_http_error_becomes_a_sample_not_an_exception():
    sample = call(
        build_target(OAI), PROBE, lambda r: httpx.Response(503, text="upstream unavailable")
    )
    assert not sample.ok
    assert "503" in sample.error
    assert sample.http_status == 503
    # The provider's error body is the fastest route to the cause.
    assert "upstream unavailable" in sample.text


def test_timeout_becomes_a_sample():
    def handler(request):
        raise httpx.TimeoutException("too slow", request=request)

    sample = call(build_target(OAI), PROBE, handler)
    assert not sample.ok and "timeout" in sample.error


def test_malformed_json_becomes_a_sample():
    sample = call(build_target(OAI), PROBE, lambda r: httpx.Response(200, text="not json"))
    assert not sample.ok and "not valid JSON" in sample.error


def test_a_transient_blip_is_retried_and_recovered():
    """A 500 is the endpoint asking to be asked again, so the sample survives.

    Four scheduled runs were lost to transient transport failures in a single week
    of running this against a real provider, and every manual re-run minutes later
    succeeded. One retry turns those into data instead of gaps.
    """
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500, text="blip")
        return httpx.Response(200, json=oai_body())

    async def go():
        async with client_for(handler) as client:
            return await collect(build_target(FAST_RETRY), PROBE, 5, client=client)

    samples = asyncio.run(go())
    assert len(samples) == 5
    assert all(s.ok for s in samples)
    # The recovery is recorded rather than hidden: a run that only worked on the
    # second try is still evidence the environment is unwell.
    assert sum(s.attempts for s in samples) == 6


def test_one_persistently_bad_sample_does_not_lose_the_others():
    """Losing four paid-for samples because the fifth failed would be wasteful.

    Retries do not change this: once they are exhausted the sample stays failed and
    the others still come back.
    """
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        # Fails on its first attempt and again on its retry. Sequential because
        # `collect` is otherwise concurrent, and "calls 2 and 3" only means "one
        # sample twice" when the requests are not interleaved.
        if calls["n"] in (2, 3):
            return httpx.Response(500, text="blip")
        return httpx.Response(200, json=oai_body())

    async def go():
        async with client_for(handler) as client:
            return await collect(
                build_target(FAST_RETRY), PROBE, 5, concurrency=1, client=client
            )

    samples = asyncio.run(go())
    assert len(samples) == 5
    assert sum(1 for s in samples if s.ok) == 4


def test_a_client_error_is_not_retried():
    """A 401 or a 404 comes back identical, so repeating it only spends money."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="nope")

    async def go():
        async with client_for(handler) as client:
            return await collect(build_target(FAST_RETRY), PROBE, 1, client=client)

    samples = asyncio.run(go())
    assert not samples[0].ok
    assert calls["n"] == 1
    assert samples[0].attempts == 1


def test_a_malformed_body_is_not_retried():
    """Not-JSON is a contract problem, and it will be not-JSON again."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, text="not json")

    async def go():
        async with client_for(handler) as client:
            return await collect(build_target(FAST_RETRY), PROBE, 1, client=client)

    samples = asyncio.run(go())
    assert not samples[0].ok
    assert calls["n"] == 1


def test_retries_can_be_switched_off():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    async def go():
        async with client_for(handler) as client:
            return await collect(build_target(NO_RETRY), PROBE, 1, client=client)

    samples = asyncio.run(go())
    assert not samples[0].ok
    assert calls["n"] == 1


# --- Plain HTTP target ----------------------------------------------------

APP = TargetConfig(
    name="app",
    type="http",
    base_url="https://app.example.com",
    path="/api/chat",
    body={"message": "{{prompt}}", "opts": {"system": "{{system}}", "stream": False}},
    response_path="data.reply",
)


def test_http_target_templates_the_body_and_reads_the_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"reply": "the answer"}})

    sample = call(build_target(APP), PROBE, handler)
    assert seen["url"] == "https://app.example.com/api/chat"
    assert seen["body"] == {
        "message": "say hello",
        "opts": {"system": "be terse", "stream": False},
    }
    assert sample.text == "the answer"


def test_http_target_reports_an_unmatched_response_path():
    """Silently returning empty text here would look exactly like model drift."""
    sample = call(
        build_target(APP), PROBE, lambda r: httpx.Response(200, json={"data": {"other": 1}})
    )
    assert not sample.ok
    assert "response_path" in sample.error


def test_http_target_without_a_path_compares_the_whole_body():
    target = build_target(
        TargetConfig(name="a", type="http", base_url="https://x", body={"q": "{{prompt}}"})
    )
    sample = call(target, PROBE, lambda r: httpx.Response(200, json={"b": 2, "a": 1}))
    assert sample.text == '{"a": 1, "b": 2}'  # sorted, so key order is not drift


# --- Helpers --------------------------------------------------------------


def test_dotted_get_walks_lists_and_dicts():
    data = {"choices": [{"message": {"content": "x"}}]}
    assert dotted_get(data, "choices.0.message.content") == "x"


def test_dotted_get_returns_none_for_a_miss():
    assert dotted_get({"a": 1}, "a.b.c") is None
    assert dotted_get({"xs": []}, "xs.3") is None


def test_render_template_reaches_into_nested_structures():
    out = render_template({"a": ["{{prompt}}", {"b": "{{system}}"}]}, {"prompt": "P", "system": "S"})
    assert out == {"a": ["P", {"b": "S"}]}


def test_render_template_leaves_non_strings_alone():
    out = render_template({"n": 5, "flag": True, "none": None}, {"prompt": "P"})
    assert out == {"n": 5, "flag": True, "none": None}


# --- response_path block filtering ----------------------------------------


def test_response_path_selects_a_block_by_type():
    """Index paths are unstable on providers with heterogeneous content blocks.

    With thinking enabled, Anthropic leads `content` with a thinking block, so
    `content.0.text` resolves to the wrong one and the probe compares an empty
    string forever without ever erroring.
    """
    body = {
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "the real answer"},
        ]
    }
    assert dotted_get(body, "content.0.text") is None
    assert dotted_get(body, "content[type=text].text") == "the real answer"


def test_response_path_filter_returns_none_when_nothing_matches():
    body = {"content": [{"type": "text", "text": "x"}]}
    assert dotted_get(body, "content[type=image].url") is None


def test_response_path_filter_works_on_a_bare_list():
    assert dotted_get([{"k": "a", "v": 1}, {"k": "b", "v": 2}], "[k=b].v") == 2


def test_response_path_filter_takes_the_first_match():
    body = {"c": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}
    assert dotted_get(body, "c[type=text].text") == "first"


def test_response_path_filter_on_a_non_list_is_none():
    assert dotted_get({"c": "not a list"}, "c[type=text].text") is None


def test_plain_paths_still_work():
    """The filter syntax must not disturb the common case."""
    assert dotted_get({"choices": [{"message": {"content": "x"}}]},
                      "choices.0.message.content") == "x"


def test_a_recovered_run_says_so_on_a_pass():
    """The retry note must survive the one-line PASS shortcut.

    A pass that only happened because a dropped connection was retried is the case
    the note exists for. `render` short-circuits passing probes to a single line, so
    the note was invisible in exactly the situation it was written for.
    """
    from stillsane.models import Level, ProbeVerdict, RunResult
    from stillsane.report import render

    recovered = RunResult(
        probes=[ProbeVerdict(probe_id="p", target_name="t", level=Level.PASS, retries=2)]
    )
    clean = RunResult(probes=[ProbeVerdict(probe_id="p", target_name="t", level=Level.PASS)])

    assert "recovered after 2 retried calls" in render(recovered, colour=False)
    # And a clean pass stays one line, or the report gets noisier for no reason.
    assert "recovered" not in render(clean, colour=False)
