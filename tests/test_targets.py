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


def test_one_bad_sample_does_not_lose_the_others():
    """Losing four paid-for samples because the fifth failed would be wasteful."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500, text="blip")
        return httpx.Response(200, json=oai_body())

    async def go():
        async with client_for(handler) as client:
            return await collect(build_target(OAI), PROBE, 5, client=client)

    samples = asyncio.run(go())
    assert len(samples) == 5
    assert sum(1 for s in samples if s.ok) == 4


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
