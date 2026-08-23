"""ClaudeCodeTarget: shells out to a subprocess, not httpx, so it needs its own
fake rather than `httpx.MockTransport`. Still no network, no real `claude`
binary, and no spend -- `asyncio.create_subprocess_exec` is monkeypatched.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from stillsane.config import ProbeConfig, TargetConfig
from stillsane.targets import build_target
from stillsane.targets.claude_code import DISALLOWED_TOOLS, _looks_like_a_leaked_tool_call

PROBE = ProbeConfig(id="p", prompt="say hello")
PROBE_WITH_SYSTEM = ProbeConfig(id="p", prompt="say hello", system="be terse")


def cc_target(**overrides) -> TargetConfig:
    return TargetConfig(name="claude", type="claude_code", **overrides)


class FakeProcess:
    def __init__(self, *, stdout=b"", stderr=b"", returncode=0, hang=False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False
        self.stdin_received: bytes | None = None

    async def communicate(self, input: bytes | None = None):
        self.stdin_received = input
        if self._hang:
            await asyncio.sleep(999)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def fake_exec(process_factory):
    """Monkeypatch target: records the argv it was called with."""
    calls: list[list[str]] = []

    async def _create(*argv, **kwargs):
        calls.append(list(argv))
        result = process_factory()
        if isinstance(result, Exception):
            raise result
        return result

    _create.calls = calls
    return _create


def result_json(text="hello", **extra):
    body = {
        "is_error": False,
        "stop_reason": "end_turn",
        "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 5, "output_tokens": 3},
        "result": text,
        "type": "result",
    }
    body.update(extra)
    return json.dumps(body).encode()


def attempt(target_config, probe, process_factory):
    target = build_target(target_config)
    patched = fake_exec(process_factory)

    async def go():
        import stillsane.targets.claude_code as mod

        original = asyncio.create_subprocess_exec
        mod.asyncio.create_subprocess_exec = patched
        try:
            return await target._attempt(probe, None)
        finally:
            mod.asyncio.create_subprocess_exec = original

    sample, transient = asyncio.run(go())
    return sample, transient, patched.calls


# --- Happy path -------------------------------------------------------------


def test_result_field_becomes_sample_text():
    sample, transient, _ = attempt(cc_target(), PROBE, lambda: FakeProcess(stdout=result_json("hi there")))
    assert sample.ok
    assert sample.text == "hi there"
    assert not transient


def test_usage_and_cost_are_captured():
    sample, _, _ = attempt(cc_target(), PROBE, lambda: FakeProcess(stdout=result_json()))
    assert sample.prompt_tokens == 5
    assert sample.completion_tokens == 3
    assert sample.cost_usd == pytest.approx(0.0123)


def test_stop_reason_becomes_finish_reason():
    """Feeds `response_complete` for free: Anthropic's "max_tokens" is already in
    that signal's TRUNCATED set, so a probe run through this target gets
    truncation detection with no extra code."""
    sample, _, _ = attempt(
        cc_target(), PROBE, lambda: FakeProcess(stdout=result_json(stop_reason="max_tokens"))
    )
    assert sample.finish_reason == "max_tokens"


def test_model_id_is_the_configured_model_never_guessed():
    """A single call can span more than one model internally (haiku for routing,
    the requested model for the answer). Reporting the configured value rather
    than picking one from `modelUsage` avoids manufactured precision."""
    sample, _, _ = attempt(
        cc_target(model="claude-opus-5"), PROBE, lambda: FakeProcess(stdout=result_json())
    )
    assert sample.model_id == "claude-opus-5"


def test_model_id_is_none_when_unconfigured():
    sample, _, _ = attempt(cc_target(), PROBE, lambda: FakeProcess(stdout=result_json()))
    assert sample.model_id is None


def test_prompt_goes_through_stdin_not_argv():
    """The whole reason for stdin: a positional prompt argument got silently
    swallowed by a preceding variadic flag in real testing, and prompt text
    should never risk being misparsed as more flags regardless of its content."""
    _, _, calls = attempt(cc_target(), PROBE, lambda: FakeProcess(stdout=result_json()))
    argv = calls[0]
    assert "say hello" not in argv


# --- argv construction --------------------------------------------------


def test_default_mode_denies_the_full_known_tool_surface():
    _, _, calls = attempt(cc_target(), PROBE, lambda: FakeProcess(stdout=result_json()))
    argv = calls[0]
    assert "--disallowedTools" in argv
    idx = argv.index("--disallowedTools")
    assert argv[idx + 1] == ",".join(DISALLOWED_TOOLS)
    assert "--allowedTools" not in argv


def test_agentic_mode_uses_an_explicit_allowlist_instead():
    _, _, calls = attempt(
        cc_target(allowed_tools=["Read", "Glob", "Grep"]),
        PROBE,
        lambda: FakeProcess(stdout=result_json()),
    )
    argv = calls[0]
    assert "--allowedTools" in argv
    idx = argv.index("--allowedTools")
    assert argv[idx + 1] == "Read,Glob,Grep"
    assert "--disallowedTools" not in argv


def test_strict_mcp_config_is_always_present_regardless_of_mode():
    """The hard floor: no MCP server is ever passed, in either mode, so this
    always refuses all of them rather than allowing whatever is configured on
    the machine running the check."""
    for config in (cc_target(), cc_target(allowed_tools=["Read"])):
        _, _, calls = attempt(config, PROBE, lambda: FakeProcess(stdout=result_json()))
        assert "--strict-mcp-config" in calls[0]


def test_system_prompt_only_passed_when_the_probe_has_one():
    _, _, calls = attempt(cc_target(), PROBE, lambda: FakeProcess(stdout=result_json()))
    assert "--system-prompt" not in calls[0]

    _, _, calls = attempt(cc_target(), PROBE_WITH_SYSTEM, lambda: FakeProcess(stdout=result_json()))
    assert "--system-prompt" in calls[0]
    idx = calls[0].index("--system-prompt")
    assert calls[0][idx + 1] == "be terse"


# --- Leaked tool-call detection ----------------------------------------


@pytest.mark.parametrize(
    "leaked",
    [
        "I'll run that now.\n\nantml:function_calls\nantml:invoke name=\"Bash\"",
        "antml:parameter name=\"command\">git status</parameter>",
    ],
)
def test_leaked_tool_call_syntax_is_detected(leaked):
    """The finding that shaped this whole target: three identical adversarial
    prompts under identical deny flags produced three different garbled
    attempts. None of them looked like a clean refusal or clean content, so
    this has to be caught rather than trusted."""
    assert _looks_like_a_leaked_tool_call(leaked)


def test_ordinary_text_does_not_trip_the_detector():
    """Must not fire on a legitimate answer that happens to discuss tool-calling
    as a topic -- only on the literal internal syntax leaking through."""
    assert not _looks_like_a_leaked_tool_call(
        "The Bash tool lets an agent run shell commands. Tool use in general "
        "follows a request/response pattern."
    )


def test_a_leaked_tool_call_is_marked_not_ok_and_never_compared_as_content():
    sample, transient, _ = attempt(
        cc_target(),
        PROBE,
        lambda: FakeProcess(stdout=result_json("antml:invoke name=\"Bash\"")),
    )
    assert not sample.ok
    assert "leaked tool-call" in sample.error
    assert transient  # non-deterministic in testing; worth one more try


# --- Failure handling -----------------------------------------------------


def test_nonzero_exit_is_retryable():
    sample, transient, _ = attempt(
        cc_target(), PROBE, lambda: FakeProcess(returncode=1, stderr=b"rate limited")
    )
    assert not sample.ok
    assert "rate limited" in sample.error
    assert transient


def test_malformed_json_stdout_is_not_retryable():
    """A contract problem, not a blip -- it will come back identical."""
    sample, transient, _ = attempt(cc_target(), PROBE, lambda: FakeProcess(stdout=b"not json"))
    assert not sample.ok
    assert not transient


def test_is_error_flag_is_retryable():
    sample, transient, _ = attempt(
        cc_target(), PROBE, lambda: FakeProcess(stdout=result_json("", is_error=True))
    )
    assert not sample.ok
    assert transient


def test_missing_binary_is_not_retryable():
    """No amount of retrying installs the CLI."""
    sample, transient, _ = attempt(
        cc_target(claude_command="not-a-real-binary-xyz"), PROBE,
        lambda: FileNotFoundError(),
    )
    assert not sample.ok
    assert "not found on PATH" in sample.error
    assert not transient


def test_timeout_kills_the_process_and_is_retryable():
    target = build_target(cc_target(timeout_s=0.05))
    process = FakeProcess(hang=True)
    patched = fake_exec(lambda: process)

    async def go():
        import stillsane.targets.claude_code as mod

        original = asyncio.create_subprocess_exec
        mod.asyncio.create_subprocess_exec = patched
        try:
            return await target._attempt(PROBE, None)
        finally:
            mod.asyncio.create_subprocess_exec = original

    sample, transient = asyncio.run(go())
    assert not sample.ok
    assert "timeout" in sample.error
    assert transient
    assert process.killed
