"""Signals read off the response envelope rather than its content.

Provider fingerprint watching lives here. It is the cheapest thing in the whole
tool and it targets the one form of drift a user cannot otherwise see: the
backend model changing underneath a version string that did not change.
"""

from __future__ import annotations

from ..models import Direction, Level, Sample
from .base import CategoricalSignal, PointwiseSignal


class Fingerprint(CategoricalSignal):
    """Provider's `system_fingerprint`, where it exposes one.

    A change means the backend moved. That is information, not a fault -- output
    quality may be identical -- so the verdict layer treats it as WARN by default
    and lets config escalate it.
    """

    name = "fingerprint"
    max_level = Level.WARN

    def value(self, sample: Sample) -> str | None:
        return sample.fingerprint


class ModelId(CategoricalSignal):
    """Model id as echoed back by the provider, which can differ from the one asked for."""

    name = "model_id"

    def value(self, sample: Sample) -> str | None:
        return sample.model_id


class CompletionTokens(PointwiseSignal):
    """Output token count. The silent cost regression.

    Two-sided on purpose: fewer tokens can mean the model started truncating, more
    can mean it started padding. Both are worth knowing and only one shows up on
    the bill.
    """

    name = "completion_tokens"
    direction = Direction.BOTH
    floor = 4.0
    rel_floor = 0.05

    def value(self, sample: Sample) -> float | None:
        if not sample.ok or sample.completion_tokens is None:
            return None
        return float(sample.completion_tokens)

    def format(self, value: float) -> str:
        return f"{value:.0f}"


class CostUsd(PointwiseSignal):
    name = "cost_usd"
    direction = Direction.UP_IS_BAD
    rel_floor = 0.10

    def value(self, sample: Sample) -> float | None:
        if not sample.ok or sample.cost_usd is None:
            return None
        return float(sample.cost_usd)

    def format(self, value: float) -> str:
        return f"${value:.6f}"


class ResponseComplete(PointwiseSignal):
    """Did the response finish on its own, or get cut off by the token limit?

    Narrowly about truncation, not about `finish_reason` churn in general --
    "stop" vs "tool_calls" vs "end_turn" are all legitimate ways for a response to
    end, and flagging every change between them would fire on ordinary agent
    behaviour. Only the token-limit case is a content problem: part of the answer
    is missing, and neither `length_chars` nor `semantic_distance` reliably catch
    that, because a truncated response can still be longer than the baseline and
    embed close to it right up to where it stops.

    `strict_when_perfect`, matching `valid_json`: if the baseline never truncated,
    any truncation on a check is drift, not a statistical question -- you cannot
    learn a variance from a constant, and this is the same reasoning `valid_json`
    already uses for exactly the same shape of problem.
    """

    name = "response_complete"
    direction = Direction.DOWN_IS_BAD
    strict_when_perfect = True

    #: The only two vocabularies this project has concrete evidence for: OpenAI's
    #: "length" and Anthropic's "max_tokens". Everything else -- "stop", "end_turn",
    #: "tool_calls", "tool_use", "content_filter", or a provider not seen yet -- is
    #: treated as complete rather than guessed at.
    TRUNCATED = frozenset({"length", "max_tokens"})

    def value(self, sample: Sample) -> float | None:
        if not sample.ok or sample.finish_reason is None:
            return None
        return 0.0 if sample.finish_reason in self.TRUNCATED else 1.0

    def format(self, value: float) -> str:
        return "complete" if value >= 1.0 else "truncated"


class LatencyMs(PointwiseSignal):
    """Round-trip latency.

    Capped at WARN and given a wide relative floor because it is measured from
    wherever the check happens to run. A CI runner is not the laptop the baseline
    was captured on, and that difference is not drift.
    """

    name = "latency_ms"
    direction = Direction.UP_IS_BAD
    max_level = Level.WARN
    floor = 50.0
    rel_floor = 0.25

    def value(self, sample: Sample) -> float | None:
        if not sample.ok or sample.latency_ms is None:
            return None
        return float(sample.latency_ms)

    def format(self, value: float) -> str:
        return f"{value:.0f}ms"
