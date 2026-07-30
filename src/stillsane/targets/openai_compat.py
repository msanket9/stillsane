"""OpenAI-compatible chat completions.

One code path covers most hosted providers plus local Ollama and vLLM, which is
the whole reason the brief picked this shape over breadth of provider support.
"""

from __future__ import annotations

from typing import Any

from ..config import ProbeConfig
from ..models import ToolCall
from .base import Target

#: Fields providers use for the backend build identifier. OpenAI settled on
#: `system_fingerprint`; others vary, and checking all of them costs nothing.
FINGERPRINT_FIELDS = ("system_fingerprint", "fingerprint", "system_version")


def read_fingerprint(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    for field in FINGERPRINT_FIELDS:
        if body.get(field):
            return str(body[field])
    return None


class OpenAICompatTarget(Target):
    def build_request(self, probe: ProbeConfig) -> tuple[str, str, dict[str, str], dict[str, Any]]:
        messages = []
        if probe.system:
            messages.append({"role": "system", "content": probe.system})
        messages.append({"role": "user", "content": probe.prompt})

        payload: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        return "POST", url, self._headers(), payload

    def parse(self, probe: ProbeConfig, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            return {"error": "response was not a JSON object"}

        choices = body.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}

        out: dict[str, Any] = {
            "text": message.get("content") or "",
            "model_id": body.get("model"),
            "tool_calls": [ToolCall.from_openai(tc) for tc in (message.get("tool_calls") or [])],
        }

        fingerprint = read_fingerprint(body)
        if fingerprint:
            out["fingerprint"] = fingerprint

        usage = body.get("usage") or {}
        if isinstance(usage, dict):
            out["prompt_tokens"] = usage.get("prompt_tokens")
            out["completion_tokens"] = usage.get("completion_tokens")
            # Some gateways price the call for you. Nothing else does, so cost stays
            # None rather than being guessed from a price table that would be wrong
            # the week after it was written.
            if usage.get("cost") is not None:
                out["cost_usd"] = float(usage["cost"])

        # A tool-only reply has no content, and that is not an error -- for an agent
        # probe it is the normal case. Only flag genuinely empty responses.
        if not out["text"] and not out["tool_calls"]:
            finish = (choices[0].get("finish_reason") if choices else None) or "no choices"
            out["error"] = f"empty response (finish_reason: {finish})"
        return out
