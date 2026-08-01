"""The LLM judge: the last and most expensive layer of the comparison.

The layering is what keeps this affordable. Structural checks are free, embedding
distance is free after a one-time download, and the judge only runs on a probe
that has *already* crossed its learned band. On a day when nothing drifted it is
never called, so a scheduled monitor costs nothing beyond the probe calls.

It is asked exactly one question: did the behaviour meaningfully change, and how.
Not "is this output good" -- that is an eval, and a different product. The band has
already decided that something moved; the judge only says whether a human would
care, and gives the one line that goes in the report.

It is advisory by default. A judge that can quietly overrule a band learned from
the probe's own measured behaviour is a way to lose real regressions, so
`can_downgrade` has to be turned on deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import JudgeConfig, ProbeConfig
from .models import Level, ProbeVerdict
from .signals.structural import extract_lenient
from .targets import build_target

SYSTEM = (
    "You compare two samples of output from the same automated prompt, captured at "
    "different times. The prompt did not change; only the model behind it may have. "
    "You answer with JSON and nothing else."
)

TEMPLATE = """\
The same prompt produced these two outputs at different times.

BEFORE:
{before}

AFTER:
{after}

Measurements that moved outside their normal range: {signals}

Did the behaviour meaningfully change, and how? Reply with JSON only:

{{"changed": true or false,
  "severity": "cosmetic" or "meaningful" or "breaking",
  "summary": "one sentence, at most 20 words"}}

severity means:
  cosmetic   - wording or formatting only; a program consuming this would not care
  meaningful - a person reading the output would notice a real difference
  breaking   - a program consuming this output would now fail or misread it
"""

SEVERITIES = ("cosmetic", "meaningful", "breaking")


@dataclass
class JudgeVerdict:
    changed: bool
    severity: str
    summary: str

    def line(self) -> str:
        return f"{self.severity}: {self.summary}"


class Judge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self._target = build_target(config.to_target())

    async def assess(
        self, verdict: ProbeVerdict, client: httpx.AsyncClient
    ) -> JudgeVerdict | None:
        """Ask about one probe. Returns None if the judge could not answer.

        Never raises. The check has already produced a verdict from measurements;
        letting a failed second opinion take that down would be strictly worse than
        going without the explanation.
        """
        before, after = verdict.baseline_excerpt, verdict.observed_excerpt
        if not before or not after:
            return None

        moved = ", ".join(sv.signal for sv in verdict.moved) or "none"
        prompt = TEMPLATE.format(before=before, after=after, signals=moved)
        probe = ProbeConfig(id="__judge__", prompt=prompt, system=SYSTEM)

        sample = await self._target.call(probe, client)
        if not sample.ok:
            return None
        return parse_verdict(sample.text)


def parse_verdict(text: str) -> JudgeVerdict | None:
    """Read the judge's reply, tolerating the ways models wrap JSON.

    Reuses the same lenient extraction the structural checks use, because a judge
    that prefixes "Sure, here you go:" is the exact failure this tool exists to
    detect and it would be absurd to fall over on it.
    """
    data: Any = extract_lenient(text)
    if not isinstance(data, dict):
        return None

    severity = str(data.get("severity", "")).strip().lower()
    if severity not in SEVERITIES:
        return None

    summary = str(data.get("summary", "")).strip()
    if not summary:
        return None

    changed = data.get("changed")
    if not isinstance(changed, bool):
        # Some models answer "true"/"yes" as a string. Accept it rather than
        # discarding an otherwise usable answer.
        changed = str(changed).strip().lower() in ("true", "yes", "1")

    return JudgeVerdict(changed=changed, severity=severity, summary=summary[:200])


def apply(verdict: ProbeVerdict, judged: JudgeVerdict | None, can_downgrade: bool) -> None:
    """Attach the judge's line, and downgrade only if explicitly allowed."""
    if judged is None:
        return
    verdict.judge_note = judged.line()

    if not can_downgrade:
        return
    # Only ever softens, and only from DRIFT to WARN. The finding stays in the
    # report either way; what changes is whether it breaks the build.
    if verdict.level is Level.DRIFT and judged.severity == "cosmetic" and not judged.changed:
        verdict.level = Level.WARN
        verdict.judge_note += " (downgraded from drift by the judge)"
