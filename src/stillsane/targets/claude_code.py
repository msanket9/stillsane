"""A probe answered by the `claude` CLI already installed on this machine.

The point of this target is billing, not capability: someone already paying for a
Claude Pro or Max subscription should not also need a separately metered API key
just to run a drift canary. Shelling out to `claude -p` draws on whatever the CLI
is already authenticated with -- a subscription login or an API key, whichever the
machine has -- rather than stillsane asking for a key of its own.

Two things were verified by hand against a real `claude` install before this was
written, both worth knowing before pointing a probe here:

1. `--bare` mode is not an option. Its own `--help` text says OAuth and keychain
   auth are never read in that mode, which would force the separately billed API
   key this target exists to avoid. So this runs in ordinary mode instead, and
   accepts the larger, harder-to-fully-restrict tool surface that comes with it.

2. Tool denial (`--disallowedTools` plus `--strict-mcp-config`, the strongest
   combination found) blocks real tool execution, but does not stop the model from
   *attempting* one. Three identical adversarial prompts against identical flags
   produced three different garbled attempts -- one leaked raw internal tool-call
   syntax into the answer text, never the same way twice. No evidence surfaced of
   a command actually running, but the leaked text looks enough like real content
   that a monitor comparing it against a baseline could not tell the difference on
   its own. `_looks_like_a_leaked_tool_call` exists because of this, and probes
   that read as an instruction to look something up, check something, or run
   something are the ones most likely to trigger it. Plain generation --
   summarise, extract, write -- has not shown this behaviour in testing.

Two modes follow from that, chosen per probe via `allowed_tools`:

* Unset (the default): every tool denied, no MCP servers, the mode verified above.
  For plain text-generation probes, which is most of them.
* An explicit list (e.g. `[Read, Glob, Grep]`): agentic mode, for a probe that is
  genuinely supposed to use tools -- testing against a real dataset, say. An
  allowlist rather than a blanket "agentic: true" switch on purpose: an unattended
  daily cron job silently granted broad tool access is a materially larger risk
  than one that can only do exactly what it was told it may do, and this mode has
  had far less real-world testing than the default.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from ..config import ProbeConfig
from ..models import Sample
from .base import Target

#: The full tool surface discovered by asking a real session to list it, plus the
#: OpenAI-style names a differently configured install might use. Static by
#: necessity -- `ToolSearch` can load more tools at runtime, and MCP servers add
#: names that vary per machine -- so this is the strongest *known* list, not a
#: guarantee, and `--strict-mcp-config` (which refuses all MCP servers outright,
#: since none are ever passed here) is what actually closes the MCP-shaped gap.
DISALLOWED_TOOLS = (
    "Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch",
    "NotebookEdit", "Task", "Artifact", "ReportFindings", "ScheduleWakeup",
    "Skill", "ToolSearch", "CronCreate", "CronDelete", "CronList", "DesignSync",
    "EnterWorktree", "ExitWorktree", "Monitor", "PushNotification",
    "RemoteTrigger", "SendMessage", "TaskCreate", "TaskGet", "TaskList",
    "TaskOutput", "TaskStop", "TaskUpdate",
)

#: Claude's own internal tool-invocation syntax. Seeing this in plain-text output
#: means a denied tool call was attempted and had nowhere to go, not that the
#: model wrote these tokens as an answer -- nobody answers a question in this
#: dialect. Chosen to be specific rather than broad: a looser pattern (matching
#: on words like "Bash" or "tool use") would flag legitimate probes that happen to
#: discuss tool-calling as a topic, and false positives here silently discard real
#: data the same way a false negative silently keeps garbage.
_LEAK_MARKERS = re.compile(r"antml:(?:function_calls|invoke|parameter)\b")


def _looks_like_a_leaked_tool_call(text: str) -> bool:
    return bool(_LEAK_MARKERS.search(text))


class ClaudeCodeTarget(Target):
    def build_request(self, probe: ProbeConfig) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        raise NotImplementedError(
            "ClaudeCodeTarget overrides _attempt() directly and never calls this; "
            "it exists only to satisfy Target's abstract interface."
        )

    def parse(self, probe: ProbeConfig, body: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "ClaudeCodeTarget overrides _attempt() directly and never calls this; "
            "it exists only to satisfy Target's abstract interface."
        )

    def _argv(self, probe: ProbeConfig) -> list[str]:
        argv = [
            self.config.claude_command,
            "-p",
            "--output-format", "json",
            # No MCP server is ever passed, so this is a hard floor regardless of
            # mode: refuses every MCP server rather than allowing the ambient set
            # configured on this machine, which varies per user and can expose
            # tools this target has no way to know about in advance.
            "--strict-mcp-config",
        ]
        if self.config.allowed_tools:
            # Agentic mode: only exactly what was named.
            argv += ["--allowedTools", ",".join(self.config.allowed_tools)]
        else:
            # Default mode: deny the full known surface. Not a guarantee -- see
            # the module docstring -- but the strongest restriction available.
            argv += ["--disallowedTools", ",".join(DISALLOWED_TOOLS)]
        if self.config.model:
            argv += ["--model", self.config.model]
        if probe.system:
            argv += ["--system-prompt", probe.system]
        return argv

    async def _attempt(self, probe: ProbeConfig, client: Any) -> tuple[Sample, bool]:
        """One invocation. `client` is unused -- this target never touches httpx --
        and stays in the signature only so `Target.call`'s retry loop can treat
        every target the same way regardless of transport.
        """
        sample = Sample(probe_id=probe.id, target_name=self.name)
        started = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._argv(probe),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            # The binary itself is missing -- a setup problem, not a blip. No
            # amount of retrying installs the CLI.
            sample.latency_ms = (time.perf_counter() - started) * 1000.0
            sample.error = f"{self.config.claude_command!r} not found on PATH"
            return sample, False

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(probe.prompt.encode()), timeout=self.config.timeout_s
            )
        # `asyncio.TimeoutError` and the builtin `TimeoutError` were only unified in
        # Python 3.11. This project supports 3.10, where `asyncio.wait_for` raises
        # the asyncio one specifically, and `except TimeoutError` alone does not
        # catch it -- the exception would propagate unhandled and abort the whole
        # run, breaking the exact "failures are captured, never raised" contract
        # this module's own timeout handling exists to uphold. Catching both is
        # redundant on 3.11+, where they are the same class, and required on 3.10.
        except (TimeoutError, asyncio.TimeoutError):
            proc.kill()
            await proc.wait()
            sample.latency_ms = (time.perf_counter() - started) * 1000.0
            sample.error = f"timeout after {self.config.timeout_s}s"
            return sample, True

        sample.latency_ms = (time.perf_counter() - started) * 1000.0

        if proc.returncode != 0:
            sample.error = f"claude exited {proc.returncode}: {stderr.decode()[:500].strip()}"
            # A nonzero exit here is closer in shape to an HTTP 5xx than a 4xx --
            # rate limiting, an overloaded backend, a session hiccup -- so it gets
            # the same benefit of the doubt.
            return sample, True

        try:
            body = json.loads(stdout)
        except ValueError as exc:
            sample.error = f"response was not valid JSON: {exc}"
            return sample, False
        if not isinstance(body, dict):
            sample.error = "response was not a JSON object"
            return sample, False

        sample.raw = body
        text = body.get("result") or ""

        if _looks_like_a_leaked_tool_call(text):
            # Kept in `.text`, truncated, for the same reason an HTTP error body is
            # kept: it is the fastest route to the cause, and it is gone once the
            # run ends. `.error` is what marks the sample as not-ok, so this never
            # gets compared against a baseline as if it were real content.
            sample.text = text[:500]
            sample.error = "response appears to contain a leaked tool-call attempt, not plain text"
            # Three identical adversarial prompts under identical flags produced
            # three different results, none clean -- this looked non-deterministic
            # enough in testing that retrying has real, if uncertain, value. Unlike
            # malformed JSON, which is a stable shape mismatch a retry cannot fix.
            return sample, True

        sample.text = text
        sample.finish_reason = body.get("stop_reason")
        sample.cost_usd = body.get("total_cost_usd")
        # Reports what was configured, never guessed from `modelUsage`'s keys: a
        # single call can span more than one model internally (haiku for routing,
        # the requested model for the answer), and picking one to call "the"
        # model would be manufactured precision this target cannot actually back.
        sample.model_id = self.config.model
        usage = body.get("usage") or {}
        if isinstance(usage, dict):
            sample.prompt_tokens = usage.get("input_tokens")
            sample.completion_tokens = usage.get("output_tokens")

        if body.get("is_error"):
            sample.error = f"claude reported an error: {text[:300] or 'no detail'}"
            return sample, True

        return sample, False
