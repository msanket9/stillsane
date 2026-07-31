"""Orchestration: config in, verdicts out.

Two flows, sharing everything except where the samples end up:

* `capture_baseline` -- take N samples, freeze them, record the day-one anchors.
* `check` -- take M samples, compare against the frozen baseline, fold clean runs
  back into the variance pool.

The CLI is a thin shell over this module, which is what makes the whole pipeline
testable against a fake target with no network.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from .compare import (
    Anchor,
    anchor_of,
    build_run,
    compare_probe,
    is_clean,
    pairwise_within,
    pool_from_run,
    within_run_evidence,
)
from .config import Config, ProbeConfig, TargetConfig, config_hash
from .models import Level, ProbeVerdict, RunResult, Sample, SignalVerdict
from .signals import build_signals, default_embedder
from .signals.base import PairwiseSignal
from .store import Baseline, BaselineStore, History
from .targets import Target, build_target, collect


@dataclass
class Plan:
    """One probe against one target, with everything needed to run it."""

    probe: ProbeConfig
    target_config: TargetConfig
    target: Target
    expected_hash: str


def plans_for(config: Config) -> list[Plan]:
    return [
        Plan(
            probe=probe,
            target_config=target_config,
            target=build_target(target_config),
            expected_hash=config_hash(probe, target_config, config.embedder),
        )
        for probe, target_config in config.pairs()
    ]


def _error_verdict(plan: Plan, message: str, level: Level = Level.ERROR) -> ProbeVerdict:
    return ProbeVerdict(
        probe_id=plan.probe.id,
        target_name=plan.target_config.name,
        level=level,
        signals=[SignalVerdict(signal="baseline", kind=None, level=level, detail=message)],
    )


async def _sample_all(
    plans: list[Plan], counts: list[int], client: httpx.AsyncClient | None = None
) -> list[list[Sample]]:
    owned = client is None
    client = client or httpx.AsyncClient()
    try:
        return list(
            await asyncio.gather(
                *(
                    collect(plan.target, plan.probe, n, client=client)
                    for plan, n in zip(plans, counts, strict=True)
                )
            )
        )
    finally:
        if owned:
            await client.aclose()


async def capture_baseline(
    config: Config,
    store: BaselineStore,
    only: set[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[Baseline]:
    """Take fresh samples and write a new baseline version for each probe."""
    plans = [p for p in plans_for(config) if not only or p.probe.id in only]
    if not plans:
        return []

    embedder = default_embedder(config.embedder)
    batches = await _sample_all(plans, [p.probe.baseline_samples for p in plans], client)

    written = []
    for plan, samples in zip(plans, batches, strict=True):
        usable = [s for s in samples if s.ok]
        if not usable:
            errors = sorted({s.error for s in samples if s.error})
            raise RuntimeError(
                f"Probe {plan.probe.id!r} on target {plan.target_config.name!r} returned "
                f"no usable samples: {'; '.join(errors) or 'unknown error'}"
            )

        # Anchors and the initial pool come from the within-baseline distances --
        # the same quantity `within_run_evidence` contributes later, so the pool
        # stays internally consistent as it grows.
        signals = build_signals(plan.probe.checks, embedder)
        for signal in signals:
            signal.prepare(usable)

        pooled: dict[str, list[float]] = {}
        anchors: dict[str, Anchor] = {}
        for signal in signals:
            if not isinstance(signal, PairwiseSignal):
                continue
            distances = pairwise_within(usable, signal)
            if distances:
                pooled[signal.name] = distances
                anchors[signal.name] = anchor_of(distances)

        written.append(
            store.save(
                plan.target_config.name,
                plan.probe.id,
                samples,
                plan.expected_hash,
                pooled=pooled,
                anchors=anchors,
            )
        )
    return written


async def check(
    config: Config,
    store: BaselineStore,
    history: History | None = None,
    only: set[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> RunResult:
    """Sample every probe, compare against its baseline, and grow clean pools."""
    plans = [p for p in plans_for(config) if not only or p.probe.id in only]
    embedder = default_embedder(config.embedder)
    band_cfg = config.thresholds.to_band_config()

    # Resolve baselines before spending anything on the network. A missing or stale
    # baseline is a config problem, and paying for samples to discover it would be
    # rude.
    runnable: list[tuple[Plan, Baseline]] = []
    verdicts: list[ProbeVerdict] = []
    for plan in plans:
        baseline = store.load(plan.target_config.name, plan.probe.id)
        if baseline is None:
            verdicts.append(
                _error_verdict(
                    plan,
                    "no baseline captured yet -- run `stillsane baseline`",
                )
            )
        elif baseline.config_hash != plan.expected_hash:
            verdicts.append(
                _error_verdict(
                    plan,
                    f"baseline v{baseline.version} was captured under a different "
                    "prompt, model, check set or embedder; run `stillsane baseline` "
                    "to recapture",
                )
            )
        elif not baseline.usable:
            verdicts.append(_error_verdict(plan, "stored baseline has no usable samples"))
        else:
            runnable.append((plan, baseline))

    if runnable:
        batches = await _sample_all(
            [p for p, _ in runnable], [p.probe.check_samples for p, _ in runnable], client
        )
        for (plan, baseline), samples in zip(runnable, batches, strict=True):
            signals = build_signals(plan.probe.checks, embedder)
            verdict = compare_probe(
                probe_id=plan.probe.id,
                target_name=plan.target_config.name,
                signals=signals,
                baseline=baseline.usable,
                current=samples,
                cfg=band_cfg,
                pooled=baseline.pooled,
                escalate_fingerprint=plan.target_config.escalate_fingerprint,
                baseline_version=baseline.version,
                baseline_created=baseline.created,
            )
            verdicts.append(verdict)

            if is_clean(verdict):
                evidence = within_run_evidence(signals, [s for s in samples if s.ok])
                if evidence:
                    store.update_variance(
                        baseline,
                        pool_from_run(evidence, baseline.pooled, baseline.anchors),
                        baseline.anchors,
                    )

    # Preserve config order rather than completion order, so the report reads the
    # same way as the file the user wrote.
    order = {(p.probe.id, p.target_config.name): i for i, p in enumerate(plans)}
    verdicts.sort(key=lambda v: order.get((v.probe_id, v.target_name), 0))

    result = build_run(verdicts)
    if history:
        history.record(result)
    return result
