"""Targets: the only part of stillsane that touches the network."""

from __future__ import annotations

from ..config import TargetConfig
from .base import DEFAULT_CONCURRENCY, HTTPTarget, Target, collect, dotted_get, render_template
from .claude_code import ClaudeCodeTarget
from .http import GenericHTTPTarget
from .openai_compat import OpenAICompatTarget

__all__ = [
    "DEFAULT_CONCURRENCY",
    "ClaudeCodeTarget",
    "GenericHTTPTarget",
    "HTTPTarget",
    "OpenAICompatTarget",
    "Target",
    "build_target",
    "collect",
    "dotted_get",
    "render_template",
]


def build_target(config: TargetConfig) -> Target:
    if config.type == "http":
        return GenericHTTPTarget(config)
    if config.type == "claude_code":
        return ClaudeCodeTarget(config)
    return OpenAICompatTarget(config)
