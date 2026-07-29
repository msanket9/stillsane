"""Shape signals: structure rather than content.

For an agent these matter more than text similarity. An agent that stopped
calling a tool, or started passing a different argument, has broken in a way no
amount of semantic similarity on its prose will reveal -- the prose often stays
perfectly plausible, which is the whole problem.
"""

from __future__ import annotations

from typing import Any

from ..models import Sample
from .base import PairwiseSignal
from .structural import extract_lenient


def key_paths(data: Any, prefix: str = "") -> set[str]:
    """Flatten a JSON value to a set of dotted key paths.

    Lists collapse to a single `[]` segment, so a three-element list and a
    two-element list have identical shape. Element *count* is a content property,
    not a structural one, and conflating them makes the signal noisy.
    """
    paths: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= key_paths(value, path)
    elif isinstance(data, list):
        path = f"{prefix}[]" if prefix else "[]"
        paths.add(path)
        for item in data:
            paths |= key_paths(item, path)
    return paths


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


class JsonShapeDistance(PairwiseSignal):
    """Distance between the key structures of two JSON responses."""

    name = "json_shape_distance"
    floor = 0.0  # Structure is discrete; any change is real, so no floor.

    def distance(self, a: Sample, b: Sample) -> float | None:
        if not (a.ok and b.ok):
            return None
        da, db = extract_lenient(a.text), extract_lenient(b.text)
        if da is None and db is None:
            return None  # Not a JSON probe; signal does not apply.
        if da is None or db is None:
            return 1.0  # One side stopped producing JSON entirely.
        return jaccard_distance(key_paths(da), key_paths(db))


class ToolCallDistance(PairwiseSignal):
    """Distance between the tool-call shapes of two responses.

    Combines which tools were called (and with which argument keys) with how many
    times, because "called the right tool twice" is a real and common agent
    regression that a set comparison alone would miss.
    """

    name = "tool_call_distance"
    floor = 0.0

    def distance(self, a: Sample, b: Sample) -> float | None:
        if not (a.ok and b.ok):
            return None
        if not a.tool_calls and not b.tool_calls:
            return None  # Not an agent probe.
        sig_a = {tc.signature() for tc in a.tool_calls}
        sig_b = {tc.signature() for tc in b.tool_calls}
        shape = jaccard_distance(sig_a, sig_b)
        count_a, count_b = len(a.tool_calls), len(b.tool_calls)
        count = (
            0.0 if count_a == count_b else abs(count_a - count_b) / max(count_a, count_b)
        )
        # Shape dominates; repetition is a lesser signal but not nothing.
        return float(min(1.0, 0.75 * shape + 0.25 * count))
