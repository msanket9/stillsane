"""Fixture builders.

Every test in this suite runs with no network, no API key, and no model download.
The embedder used throughout is the real `HashingEmbedder`, not a stub, so the
tests exercise the actual scoring path rather than a mock of it.
"""

from __future__ import annotations

import pytest

from stillsane.models import Sample, ToolCall
from stillsane.signals import HashingEmbedder, build_signals


def sample(
    text: str = "",
    *,
    probe: str = "probe",
    target: str = "prod",
    fingerprint: str | None = None,
    model: str | None = None,
    completion_tokens: int | None = None,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
    tool_calls: list[ToolCall] | None = None,
    error: str | None = None,
) -> Sample:
    return Sample(
        probe_id=probe,
        target_name=target,
        text=text,
        fingerprint=fingerprint,
        model_id=model,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        tool_calls=tool_calls or [],
        error=error,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def signals_for(embedder):
    def _build(checks=None):
        return build_signals(checks, embedder)

    return _build


# --- Scenario texts -------------------------------------------------------
# Kept here rather than inline so the scenarios read as data and the tests read
# as assertions.

#: A near-deterministic extraction probe. Real runs vary only in whitespace and
#: key order, which is what a temperature-0 JSON endpoint actually looks like.
STABLE_JSON = [
    '{"total": 1240.50, "due_date": "2026-07-01"}',
    '{"total": 1240.5, "due_date": "2026-07-01"}',
    '{"due_date": "2026-07-01", "total": 1240.50}',
    '{"total": 1240.50,  "due_date": "2026-07-01"}',
    '{"total": 1240.50, "due_date": "2026-07-01"}',
]

#: The same data, but the model started explaining itself. Downstream
#: `json.loads` now throws. This is the single most common real-world drift.
PROSE_WRAPPED_JSON = [
    'Here is the extracted information:\n{"total": 1240.50, "due_date": "2026-07-01"}\n'
    "Let me know if you need anything else!",
    'Sure! I found the following:\n{"total": 1240.5, "due_date": "2026-07-01"}\n'
    "Happy to help with more invoices.",
    'Of course. The details are:\n{"total": 1240.50, "due_date": "2026-07-01"}\n'
    "Anything else I can do?",
]

#: A summarisation probe. Every sample is different prose about the same subject.
#: A fixed similarity threshold cannot tell this apart from drift; a learned band
#: can, and that difference is the entire product.
CHATTY_BASELINE = [
    "The quarterly report shows revenue climbed to 4.2 million, driven mostly by "
    "the enterprise segment, while operating costs stayed flat year over year.",
    "Revenue reached 4.2M this quarter. Enterprise accounts did the heavy lifting. "
    "Costs were essentially unchanged compared to last year.",
    "This quarter brought in 4.2 million in revenue. Growth came from enterprise "
    "customers. Operating expenses held steady against the prior year.",
    "Topline hit 4.2 million for the quarter, with enterprise driving the gain and "
    "operating costs remaining level versus the same period last year.",
    "The company recorded 4.2 million in quarterly revenue, largely from enterprise "
    "deals, and kept operating costs flat relative to a year ago.",
]

CHATTY_CURRENT = [
    "Quarterly revenue came to 4.2 million, mainly thanks to enterprise sales, and "
    "operating costs were flat compared with the previous year.",
    "The quarter delivered 4.2M of revenue. Enterprise was the main contributor. "
    "Operating expenses did not move much year on year.",
    "Revenue for the quarter totalled 4.2 million, propelled by enterprise "
    "business, while costs stayed roughly where they were last year.",
]

#: Same chatty probe, but the model has genuinely wandered off. Different subject,
#: different register. A learned band must still catch this.
CHATTY_DRIFTED = [
    "I'd be happy to help you analyse financial documents! Could you share the "
    "report you'd like me to look at? I can summarise revenue, costs, and margins.",
    "Sure, I can help with that. Please paste the quarterly figures and I'll walk "
    "you through the key drivers and any notable changes.",
    "Of course! Send over the document and I'll break down the important numbers "
    "for you, including revenue trends and cost movements.",
]
