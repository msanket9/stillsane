"""Signal registry: turns a probe's configured checks into signal objects.

Signals that do not apply to a given probe return None from their scoring method
and are skipped, so the always-on set costs nothing on probes it does not suit --
`tool_call_distance` is silent on a probe that never calls a tool, and
`json_shape_distance` is silent on one that returns prose.
"""

from __future__ import annotations

from typing import Any

from .base import CategoricalSignal, PairwiseSignal, PointwiseSignal, Signal
from .meta import CompletionTokens, CostUsd, Fingerprint, LatencyMs, ModelId, ResponseComplete
from .semantic import (
    Embedder,
    HashingEmbedder,
    Model2VecEmbedder,
    SemanticDistance,
    default_embedder,
)
from .shape import JsonShapeDistance, ToolCallDistance
from .structural import HasKeys, LengthChars, ValidJson

__all__ = [
    "ALWAYS_ON",
    "CategoricalSignal",
    "Embedder",
    "HashingEmbedder",
    "Model2VecEmbedder",
    "PairwiseSignal",
    "PointwiseSignal",
    "SemanticDistance",
    "Signal",
    "build_signals",
    "default_embedder",
]

#: Signals every probe gets, whether or not the config mentions them. These are
#: the ones that cost nothing extra to compute and that nobody thinks to ask for
#: until the day they would have caught something.
ALWAYS_ON = (
    "semantic_distance",
    "json_shape_distance",
    "tool_call_distance",
    "length_chars",
    "completion_tokens",
    "cost_usd",
    "latency_ms",
    "fingerprint",
    "model_id",
    "response_complete",
)


def _normalise(check: Any) -> tuple[str, Any]:
    """Accept both `- valid_json` and `- has_keys: [a, b]` forms."""
    if isinstance(check, str):
        return check, None
    if isinstance(check, dict) and len(check) == 1:
        (name, value), = check.items()
        return str(name), value
    raise ValueError(
        f"Malformed check {check!r}. Expected either a bare name (`- valid_json`) "
        "or a single-key mapping (`- has_keys: [total, due_date]`)."
    )


def build_signals(checks: list[Any] | None, embedder: Embedder) -> list[Signal]:
    """Build the signal list for one probe.

    `checks` comes straight from YAML. Unknown names raise rather than being
    ignored: a silently dropped check is a check the user believes is protecting
    them when it is not.
    """
    semantic = SemanticDistance(embedder)
    signals: list[Signal] = [
        semantic,
        JsonShapeDistance(),
        ToolCallDistance(),
        LengthChars(),
        CompletionTokens(),
        CostUsd(),
        LatencyMs(),
        Fingerprint(),
        ModelId(),
        ResponseComplete(),
    ]

    for check in checks or []:
        name, value = _normalise(check)
        if name == "valid_json":
            signals.append(ValidJson())
        elif name == "has_keys":
            if not isinstance(value, list) or not value:
                raise ValueError("`has_keys` needs a non-empty list of key names.")
            signals.append(HasKeys(value))
        elif name in ("semantic_similarity", "semantic_distance"):
            # `auto` is the default behaviour: learn the band. A number pins it.
            if value not in (None, "auto"):
                semantic.band_override = float(value)
        elif name == "max_length":
            length = LengthChars()
            length.band_override = float(value)
            signals.append(length)
        else:
            raise ValueError(
                f"Unknown check {name!r}. Supported: valid_json, has_keys, "
                "semantic_similarity, max_length."
            )
    return signals
