"""Aggregation: signals -> probe -> run.

The rule is simply "worst wins" at every level, with one deliberate exception:
transport errors are not drift. An endpoint that is down, rate-limited, or
returning 500s produces ERROR, not DRIFT. Conflating the two would mean an outage
and a quality regression look identical in CI, and they call for completely
different responses.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import Level, ProbeVerdict, RunResult, Sample, SignalVerdict
from ..signals.base import Signal
from .variance import BandConfig, evaluate

#: How much of a sample to show in the before/after block.
EXCERPT_CHARS = 400


def _excerpt(samples: Sequence[Sample]) -> str | None:
    """A representative output for the report.

    Picks the sample of median length rather than the first one: the first sample
    can happen to be the outlier, and showing the user an unrepresentative example
    of "normal" makes the diff harder to read, not easier.
    """
    ok = [s for s in samples if s.ok]
    if not ok:
        return None
    ranked = sorted(ok, key=lambda s: len(s.text))
    text = ranked[len(ranked) // 2].text
    if len(text) > EXCERPT_CHARS:
        return text[:EXCERPT_CHARS] + "\n... [truncated]"
    return text


def compare_probe(
    probe_id: str,
    target_name: str,
    signals: Sequence[Signal],
    baseline: Sequence[Sample],
    current: Sequence[Sample],
    cfg: BandConfig | None = None,
    pooled: dict[str, list[float]] | None = None,
    escalate_fingerprint: bool = False,
    baseline_version: int | None = None,
    baseline_created: str | None = None,
) -> ProbeVerdict:
    """Score one probe against its baseline."""
    cfg = cfg or BandConfig()
    pooled = pooled or {}

    verdict = ProbeVerdict(
        probe_id=probe_id,
        target_name=target_name,
        level=Level.PASS,
        baseline_version=baseline_version,
        baseline_created=baseline_created,
    )

    live = [s for s in current if s.ok]
    if not live:
        errors = {s.error for s in current if s.error} or {"no samples captured"}
        verdict.level = Level.ERROR
        verdict.signals = [
            SignalVerdict(
                signal="transport",
                kind=None,
                level=Level.ERROR,
                detail="; ".join(sorted(str(e) for e in errors)),
            )
        ]
        return verdict

    if not [s for s in baseline if s.ok]:
        verdict.level = Level.ERROR
        verdict.signals = [
            SignalVerdict(
                signal="baseline",
                kind=None,
                level=Level.ERROR,
                detail="baseline contains no usable samples; run `stillsane baseline`",
            )
        ]
        return verdict

    # One warm-up pass over every sample the signals will see, so batch-capable
    # signals (embeddings) encode once instead of once per pair.
    everything = list(baseline) + list(current)
    for signal in signals:
        signal.prepare(everything)

    results: list[SignalVerdict] = []
    for signal in signals:
        sv = evaluate(
            signal,
            baseline,
            current,
            cfg,
            pooled=pooled.get(signal.name),
            escalate_categorical=escalate_fingerprint,
        )
        if sv is not None:
            results.append(sv)

    # Partial failure: some samples came back, some did not. Worth surfacing but
    # not worth failing on -- one flaky call in five is not a quality regression.
    dropped = len(current) - len(live)
    if dropped:
        results.append(
            SignalVerdict(
                signal="transport",
                kind=None,
                level=Level.WARN,
                detail=f"{dropped}/{len(current)} samples failed to return",
            )
        )

    verdict.signals = results
    verdict.level = Level.worst([r.level for r in results])
    verdict.baseline_excerpt = _excerpt(baseline)
    verdict.observed_excerpt = _excerpt(current)
    return verdict


def build_run(probes: Sequence[ProbeVerdict]) -> RunResult:
    from datetime import datetime, timezone

    return RunResult(probes=list(probes), finished=datetime.now(timezone.utc))
