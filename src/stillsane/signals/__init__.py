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
from .structural import ConstantField, HasKeys, LengthChars, ValidJson

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


def _number(name: str, value: Any) -> float:
    # A quoted number (`"0.2"`) was accepted before the unit fix and stays accepted.
    try:
        if isinstance(value, bool):
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"`{name}` needs a number or `auto`, not {value!r}.") from None


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


def build_signals(
    checks: list[Any] | None, embedder: Embedder, watch_fingerprint: bool = True
) -> list[Signal]:
    """Build the signal list for one probe.

    `checks` comes straight from YAML. Unknown names raise rather than being
    ignored: a silently dropped check is a check the user believes is protecting
    them when it is not.

    `watch_fingerprint` is the escape hatch for a provider whose
    `system_fingerprint` churns for reasons that are not drift -- it drops the
    `Fingerprint` signal entirely rather than merely silencing it, so a target
    that opts out gets no fingerprint row in the report at all.
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
        ModelId(),
        ResponseComplete(),
    ]
    if watch_fingerprint:
        signals.append(Fingerprint())

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
            # The signal itself is a *distance* (0 = identical), so the two
            # spellings differ in unit: `semantic_similarity: 0.9` means "stay at
            # least 90% similar", which is a distance ceiling of 0.1. Applying the
            # number as-is made a similarity threshold a distance of 0.9, which no
            # response ever exceeds, so the check could never fire.
            if value not in (None, "auto"):
                number = _number(name, value)
                if name == "semantic_similarity":
                    if not 0.0 <= number <= 1.0:
                        raise ValueError(
                            "`semantic_similarity` is a similarity in [0, 1] "
                            "(1 = identical); use `semantic_distance` to give a distance."
                        )
                    semantic.band_override = 1.0 - number
                else:
                    if number < 0.0:
                        raise ValueError("`semantic_distance` cannot be negative.")
                    semantic.band_override = number
        elif name == "max_length":
            # A second, independent signal, not an override of the always-on
            # `length_chars` -- that one keeps its own learned two-sided band.
            # Sharing the name `length_chars` produced two entries under one
            # key: two report rows, two history rows per run, and `calibrate`
            # silently merging both into a single misleading line.
            length = LengthChars()
            length.name = "max_length"
            length.band_override = float(value)
            signals.append(length)
        elif name == "constant_fields":
            if not isinstance(value, list) or not value:
                raise ValueError("`constant_fields` needs a non-empty list of field names.")
            # One signal per field, not one signal covering the list: "total
            # changed" and "due_date changed" are different findings a report
            # should name separately, unlike `has_keys`, which only ever asks
            # a single yes/no question about all of them together.
            signals.extend(ConstantField(field) for field in value)
        else:
            raise ValueError(
                f"Unknown check {name!r}. Supported: valid_json, has_keys, "
                "semantic_similarity, semantic_distance, max_length, constant_fields."
            )
    return signals
