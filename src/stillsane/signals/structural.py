"""Structural checks: the cheap layer that runs first and catches a lot.

These are pointwise signals valued 0.0 or 1.0 per sample, so a probe's pass-rate
flows through the same variance machinery as everything else. That matters for
probes that were already flaky at baseline (4/5) -- a fixed "must be 1.0" rule
would page you every run, while a band accommodates the flakiness and still
catches a real slide.

The verdict layer adds one hard rule on top: a check that was perfect at baseline
and is no longer perfect is drift, band or no band. See `compare.verdict`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..models import Direction, Sample
from .base import PointwiseSignal

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, if present.

    Fenced JSON is ubiquitous enough that treating it as invalid would make the
    check useless, but prose *outside* a fence still counts as invalid -- that is
    the failure mode where a caller's `json.loads` starts throwing.
    """
    m = _FENCE.match(text)
    return m.group(1) if m else text


def parse_strict(text: str) -> Any | None:
    """Parse only if the entire response is JSON. Returns None otherwise."""
    try:
        return json.loads(strip_fence(text).strip())
    except (ValueError, TypeError):
        return None


def extract_lenient(text: str) -> Any | None:
    """Find the first balanced JSON object or array anywhere in the text.

    Used by content checks (`has_keys`) so they can answer "is the data still
    there" independently of "is the response still parseable". When output gets
    wrapped in prose those two answers diverge, and the report is much clearer
    when it can say which one moved.
    """
    strict = parse_strict(text)
    if strict is not None:
        return strict
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except (ValueError, TypeError):
                            break
            start = text.find(opener, start + 1)
    return None


class ValidJson(PointwiseSignal):
    """Does the whole response parse as JSON?"""

    name = "valid_json"
    direction = Direction.DOWN_IS_BAD
    strict_when_perfect = True

    def value(self, sample: Sample) -> float | None:
        if not sample.ok:
            return None
        return 1.0 if parse_strict(sample.text) is not None else 0.0

    def format(self, value: float) -> str:
        return "valid" if value >= 1.0 else f"{value:.0%} valid"


class HasKeys(PointwiseSignal):
    """Are the required keys present, wherever the JSON happens to be?"""

    direction = Direction.DOWN_IS_BAD
    strict_when_perfect = True

    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self.name = f"has_keys[{','.join(self.keys)}]"

    def value(self, sample: Sample) -> float | None:
        if not sample.ok:
            return None
        data = extract_lenient(sample.text)
        if not isinstance(data, dict):
            return 0.0
        return 1.0 if all(k in data for k in self.keys) else 0.0

    def format(self, value: float) -> str:
        return "present" if value >= 1.0 else f"{value:.0%} present"

    def missing(self, sample: Sample) -> list[str]:
        """Which keys are absent -- for the report, not the verdict."""
        data = extract_lenient(sample.text)
        if not isinstance(data, dict):
            return list(self.keys)
        return [k for k in self.keys if k not in data]


class LengthChars(PointwiseSignal):
    """Response length. Moves in either direction are interesting."""

    name = "length_chars"
    direction = Direction.BOTH
    floor = 8.0
    rel_floor = 0.05

    def value(self, sample: Sample) -> float | None:
        return float(len(sample.text)) if sample.ok else None

    def format(self, value: float) -> str:
        return f"{value:.0f}"
