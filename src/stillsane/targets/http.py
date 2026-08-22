"""An arbitrary deployed endpoint: your app, not a model API.

This is the target that matters for the stated use case. Most people are not
monitoring a raw model -- they are monitoring the thing they shipped, which sits
in front of one, with its own retrieval, its own prompt assembly and its own bugs.
Probing the model directly would miss all of that.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import ProbeConfig
from .base import Target, dotted_get, render_template
from .openai_compat import read_fingerprint, read_finish_reason


class HTTPTarget(Target):
    def build_request(self, probe: ProbeConfig) -> tuple[str, str, dict[str, str], dict[str, Any]]:
        variables = {"prompt": probe.prompt, "system": probe.system or ""}
        payload = render_template(self.config.body or {}, variables)
        url = self.config.base_url.rstrip("/") + (self.config.path or "")
        return self.config.method.upper(), url, self._headers(), payload

    def parse(self, probe: ProbeConfig, body: Any) -> dict[str, Any]:
        if self.config.response_path:
            value = dotted_get(body, self.config.response_path)
            if value is None:
                return {
                    "error": (
                        f"response_path {self.config.response_path!r} matched nothing "
                        "in the response"
                    )
                }
            text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        elif isinstance(body, str):
            text = body
        else:
            # No path configured and a structured body: compare the whole thing
            # rather than guessing which field was the one that mattered.
            text = json.dumps(body, sort_keys=True)

        out: dict[str, Any] = {"text": text}
        fingerprint = read_fingerprint(body)
        if fingerprint:
            out["fingerprint"] = fingerprint
        finish = read_finish_reason(body)
        if finish:
            out["finish_reason"] = finish
        if isinstance(body, dict) and body.get("model"):
            out["model_id"] = str(body["model"])
        return out
