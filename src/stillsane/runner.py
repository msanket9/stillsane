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

from .bands import inspect as inspect_bands
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
from .judge import Judge
from .judge import apply as judge_apply
from .models import Level, ProbeVerdict, RunResult, Sample, SignalVerdict
from .signals import build_signals, default_embedder
from .signals.base import PairwiseSignal
from .store import Baseline, BaselineStore, History, RunSampleStore
from .targets import DEFAULT_CONCURRENCY, Target, build_target, collect


@dataclass
class Plan:
    """One probe against one target, with everything needed to run it."""

    probe: ProbeConfig
    target_config: TargetConfig
    target: Target
    expected_hash: str


@dataclass(frozen=True)
class Captured:
    """One `capture_baseline` result: the baseline just written, and which of
    its signals came out floored -- defaulted rather than measured.

    `floored` used to live as a field on `Baseline` itself, documented there as
    "not persisted... only interesting at the moment of capture" -- true, but
    that made it a field that is simply wrong on every `Baseline` `store.load`
    returns (always `[]`, since nothing ever populates it after the fact).
    Returning it alongside the baseline instead, only from the one call that
    actually knows it, says the same thing without the always-empty field.
    """

    baseline: Baseline
    floored: list[str]
    #: Set only when `capture_baseline(..., compare_previous=True)` found a
    #: version to compare against. `None` either because the flag was off, or
    #: because this was the first baseline ever captured for this probe/target
    #: -- `previous_version` tells the two cases apart.
    previous_comparison: ProbeVerdict | None = None
    previous_version: int | None = None
    #: Whether the version just replaced was captured under a different
    #: config hash. A re-baseline after a prompt edit is the common case this
    #: exists for, and the comparison must say so rather than let a real
    #: prompt-driven move read as the provider changing underneath it.
    config_changed: bool = False


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
    # One semaphore per target, shared across every plan that points at it.
    # `collect` used to cap concurrency per call, and every plan's `collect`
    # runs concurrently here -- so N probes against one target opened up to
    # `DEFAULT_CONCURRENCY * N` requests in flight, exactly the outage this cap
    # exists to prevent (see the module docstring in targets/base.py).
    semaphores: dict[str, asyncio.Semaphore] = {}
    for plan in plans:
        semaphores.setdefault(
            plan.target_config.name, asyncio.Semaphore(DEFAULT_CONCURRENCY)
        )
    try:
        return list(
            await asyncio.gather(
                *(
                    collect(
                        plan.target,
                        plan.probe,
                        n,
                        client=client,
                        semaphore=semaphores[plan.target_config.name],
                    )
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
    compare_previous: bool = False,
) -> list[Captured]:
    """Take fresh samples and write a new baseline version for each probe.

    `compare_previous` answers the question a re-baseline otherwise leaves
    unasked: a DRIFT fires, the user decides it is the new normal and runs
    this -- v1 is kept on disk but never looked at again, and the user has
    just accepted a shift without being told its size. With the flag, the
    version just replaced is compared against the one just written, using
    the exact same `compare_probe` a scheduled check would use, and the
    result rides along on `Captured.previous_comparison` purely for display:
    it never affects what gets written, never touches the variance pool, and
    has no exit code of its own -- `cmd_baseline` always returns 0.
    """
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
        signals = build_signals(
            plan.probe.checks, embedder, plan.target_config.watch_fingerprint
        )
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

        if not plan.target_config.store_raw:
            # `store.save` persists whatever is on `Sample.raw` into
            # `samples.jsonl`, and nothing in stillsane reads it back -- no
            # signal, no report. Default off, so the full response bodies the
            # README tells people to commit to git are the ones a target
            # opted into, not every target by default.
            for s in samples:
                s.raw = {}

        baseline = store.save(
            plan.target_config.name,
            plan.probe.id,
            samples,
            plan.expected_hash,
            pooled=pooled,
            anchors=anchors,
        )
        # Reuses `bands.inspect` rather than rebuilding each band a second way:
        # the two used to walk the signal list separately, one recomputing a
        # band just to read `.floored` off it. `inspect` already does that (and
        # more -- the collapsed/self-outside diagnosis `stillsane bands` shows
        # is now available here too, for free) from the baseline this call just
        # wrote, so there is nothing left for a capture-time version to redo.
        report = inspect_bands(
            baseline, signals, config.thresholds.to_band_config(), plan.probe.check_samples
        )
        floored = [sb.signal for sb in report.signals if sb.band.floored]

        previous_comparison = None
        previous_version = None
        config_changed = False
        if compare_previous and baseline.version > 1:
            previous_version = baseline.version - 1
            previous = store.load(plan.target_config.name, plan.probe.id, version=previous_version)
            if previous is not None and previous.usable:
                config_changed = previous.config_hash != baseline.config_hash
                # The version just replaced *is* the baseline here, and the
                # version just written is what gets judged against it -- the
                # same shape as an ordinary check, just with the new samples
                # standing in for a live run. `previous.pooled` is whatever
                # variance that old baseline had accumulated by the time it
                # was replaced, so the band is as informed as a real check
                # against it would have been on this same day.
                previous_comparison = compare_probe(
                    probe_id=plan.probe.id,
                    target_name=plan.target_config.name,
                    signals=signals,
                    baseline=previous.usable,
                    current=baseline.usable,
                    cfg=config.thresholds.to_band_config(),
                    pooled=previous.pooled,
                    escalate_fingerprint=plan.target_config.escalate_fingerprint,
                    baseline_version=previous.version,
                    baseline_created=previous.created,
                )

        written.append(
            Captured(
                baseline=baseline,
                floored=floored,
                previous_comparison=previous_comparison,
                previous_version=previous_version,
                config_changed=config_changed,
            )
        )
    return written


async def _run_judge(
    config: Config, verdicts: list[ProbeVerdict], client: httpx.AsyncClient | None
) -> None:
    """Ask the judge about probes that already failed their band, and only those.

    This is where the tiering pays off: on a run where nothing drifted the loop
    below has nothing to iterate over and not a single token is spent.
    """
    if config.judge is None:
        return
    suspects = [v for v in verdicts if v.level in (Level.WARN, Level.DRIFT)]
    if not suspects:
        return

    judge = Judge(config.judge)
    owned = client is None
    client = client or httpx.AsyncClient()
    try:
        results = await asyncio.gather(
            *(judge.assess(v, client) for v in suspects), return_exceptions=True
        )
    finally:
        if owned:
            await client.aclose()

    for verdict, result in zip(suspects, results, strict=True):
        # A judge that fell over must not take the run down with it. The verdict
        # came from measurements and stands on its own; the judge only adds prose.
        judged, sample = result if not isinstance(result, BaseException) else (None, None)
        # The judge call is a real API call and, on gateways that price it, a
        # real cost -- fold both into the same probe's totals `check`'s own
        # sampling loop already populated, so the footer's "this run: N
        # calls, $X" does not silently exclude exactly the calls that fire
        # only on the WARN/DRIFT runs a reader is most likely to be checking
        # the cost of. `sample` is `None` only when no call was attempted at
        # all (missing excerpts, or the unexpected-exception fallback above).
        if sample is not None:
            verdict.total_calls += 1
            if sample.cost_usd is not None:
                verdict.cost_usd = (verdict.cost_usd or 0.0) + sample.cost_usd
                verdict.cost_known_calls += 1
        judge_apply(verdict, judged, config.judge.can_downgrade)


async def check(
    config: Config,
    store: BaselineStore,
    history: History | None = None,
    only: set[str] | None = None,
    client: httpx.AsyncClient | None = None,
    against_stale: bool = False,
    run_samples: RunSampleStore | None = None,
) -> RunResult:
    """Sample every probe, compare against its baseline, and grow clean pools.

    `against_stale` is the escape hatch for a PR that edits a probe: normally a
    changed config hash refuses to compare at all (comparing new output against
    an old baseline is not "drift", it is the edit doing exactly what it was
    asked to do), which is correct for a scheduled run but useless for a CI
    check whose whole point was to catch the edit before it lands -- the PR
    just gets blocked on a local recapture instead. With the flag, the
    comparison runs anyway, but every affected verdict is capped at WARN
    (never DRIFT, never ERROR) and never grows the variance pool, since the
    numbers it produces describe an edit, not a measurement. See
    `ProbeVerdict.stale_comparison`, which the report keys off of to say so on
    every line -- the flag must never be mistakable for a real refusal or a
    real drift verdict, or for a comparison anyone would want as the default.

    `run_samples`, when given, records what each probe's current samples
    actually said under the run's own id -- see `RunSampleStore` -- so
    `stillsane history --run <id>` can show it later without anyone needing
    the original log this run printed to.
    """
    plans = [p for p in plans_for(config) if not only or p.probe.id in only]
    embedder = default_embedder(config.embedder)
    band_cfg = config.thresholds.to_band_config()
    sampled: list[tuple[Plan, list[Sample]]] = []

    # Resolve baselines before spending anything on the network. A missing or stale
    # baseline is a config problem, and paying for samples to discover it would be
    # rude.
    runnable: list[tuple[Plan, Baseline, bool]] = []
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
            if against_stale and baseline.usable:
                runnable.append((plan, baseline, True))
            else:
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
            runnable.append((plan, baseline, False))

    if runnable:
        batches = await _sample_all(
            [p for p, _, _ in runnable], [p.probe.check_samples for p, _, _ in runnable], client
        )
        for (plan, baseline, stale), samples in zip(runnable, batches, strict=True):
            signals = build_signals(
                plan.probe.checks, embedder, plan.target_config.watch_fingerprint
            )
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
            # Extra calls beyond one per sample. Attached here rather than inside
            # `compare_probe` because it says nothing about whether the probe moved:
            # it is a fact about reaching the endpoint, and the comparison layer
            # deliberately knows nothing about transport.
            verdict.retries = sum(max(0, s.attempts - 1) for s in samples)
            verdict.total_calls = len(samples)
            known_costs = [s.cost_usd for s in samples if s.cost_usd is not None]
            if known_costs:
                verdict.cost_usd = sum(known_costs)
                verdict.cost_known_calls = len(known_costs)
            sampled.append((plan, samples))

            if stale:
                verdict.stale_comparison = True
                # Cap every signal, not just the aggregate: `verdict.level` is
                # `Level.worst(sv.level for sv in signals)`, so leaving a
                # per-signal level at DRIFT while only capping the aggregate
                # left it sitting in `verdict.signals` (and so in the JSON
                # payload's `moved[].level`, and in the coloured terminal
                # line, which keys off the signal's own level) as an
                # uncapped "drift" underneath a probe that claims "warn" --
                # exactly the "mistaken for a real drift verdict" outcome
                # this flag exists to rule out for a machine reading the
                # payload rather than the headline.
                for sv in verdict.signals:
                    if sv.level.rank > Level.WARN.rank:
                        sv.level = Level.WARN
                if verdict.level.rank > Level.WARN.rank:
                    verdict.level = Level.WARN
            verdicts.append(verdict)

            # Pooling grows the variance estimate from a measurement; a stale
            # comparison never took one -- the baseline it ran against is not
            # the one this config now produces -- so it must never feed back
            # into a pool the *next*, properly-baselined check will trust.
            if not stale and is_clean(verdict):
                evidence = within_run_evidence(signals, [s for s in samples if s.ok])
                if evidence:
                    new_pooled, new_anchors = pool_from_run(
                        evidence, baseline.pooled, baseline.anchors
                    )
                    store.update_variance(baseline, new_pooled, new_anchors)

        await _run_judge(config, verdicts, client)

    # Preserve config order rather than completion order, so the report reads the
    # same way as the file the user wrote.
    order = {(p.probe.id, p.target_config.name): i for i, p in enumerate(plans)}
    verdicts.sort(key=lambda v: order.get((v.probe_id, v.target_name), 0))

    # Attribution: is a move here explained by the same move on a paired
    # raw-model target, or not? Needs every plan's verdict already built for
    # this run -- the whole point is comparing two targets' verdicts on the
    # *same* probe within one run -- so this runs once here rather than
    # per-plan inside the sampling loop above, and needs no history or
    # network of its own: nothing beyond what listing two targets already cost.
    targets_by_name = {t.name: t for t in config.targets}
    verdicts_by_key = {(v.probe_id, v.target_name): v for v in verdicts}
    for verdict in verdicts:
        if verdict.level is Level.PASS:
            continue
        target_config = targets_by_name.get(verdict.target_name)
        control_name = target_config.attribute_to if target_config else None
        if not control_name:
            continue
        control = verdicts_by_key.get((verdict.probe_id, control_name))
        if control is None or control.level is Level.ERROR:
            # Not scoped to run against the control in this run, or the
            # control run itself errored -- nothing to attribute against.
            continue
        verdict.attribution = _attribution_note(verdict, control, control_name)

    result = build_run(verdicts)
    if history:
        # Before this run is itself recorded: `probe_recent_levels` must see
        # only *prior* runs, or a probe would count this run in its own
        # streak twice.
        finished = result.finished.isoformat(timespec="seconds") if result.finished else ""
        for verdict in verdicts:
            if verdict.level is not Level.PASS:
                prior = history.probe_recent_levels(verdict.probe_id, verdict.target_name)
                verdict.first_seen, verdict.consecutive_runs = _probe_streak(prior, finished)
        run_id = history.record(result)
        if run_samples and sampled:
            for plan, samples in sampled:
                run_samples.append(run_id, samples, store_raw=plan.target_config.store_raw)
            run_samples.prune()
    return result


def _control_moved(verdict: ProbeVerdict) -> bool:
    """Did anything but the fingerprint move on this verdict?

    Fingerprints differ per account and per region even against the same
    unchanged model, so a fingerprint-only WARN on the control target is not
    evidence the underlying model moved -- attributing off it alone would
    make the control noisier than the thing it is meant to corroborate.
    Every other signal is fair game: a control that moved on
    `semantic_distance` or `valid_json` genuinely behaved differently.
    """
    return any(sv.level is not Level.PASS for sv in verdict.signals if sv.signal != "fingerprint")


def _attribution_note(verdict: ProbeVerdict, control: ProbeVerdict, control_name: str) -> str:
    """One line answering "is this my app or the model?" for a probe that
    moved and has a paired raw-model control.

    Never the final word, on purpose: the control probe usually skips the
    app's own system prompt and retrieval, so it is a different question
    asked of the same model, not a re-run of the app's own request. Moving
    together is consistent with a provider change; moving alone is
    consistent with the change being local. Neither is proof -- see the
    caveat baked into both branches below, which the report always shows
    alongside the conclusion rather than letting the headline stand alone.
    """
    if _control_moved(control):
        return (
            f"attribution: {control_name!r} (control) moved too -- consistent with a "
            f"provider-side change, not something local to {verdict.target_name!r}."
        )
    return (
        f"attribution: {control_name!r} (control) did not move -- looks local to "
        f"{verdict.target_name!r}, not the provider. Not proof {control_name!r} "
        f"itself is unchanged: {verdict.target_name!r} likely wraps its own system "
        "prompt and retrieval."
    )


def _probe_streak(prior_levels: list[tuple[str, int]], current_finished: str) -> tuple[str, int]:
    """(first_seen, consecutive_runs) for a probe that just moved, including
    the run currently being built.

    `prior_levels` is that exact probe/target's own history, newest run
    first, already reduced to one worst-signal rank per run --
    `History.probe_recent_levels`. Walking it backward from the most recent
    prior run, the streak extends through every consecutive non-PASS run and
    stops at the first PASS (rank 0) or the end of recorded history -- "since
    when has this probe been non-PASS, with no clean run in between".

    Only ever called for a verdict that already moved (the caller gates on
    `level is not Level.PASS`), so there is always at least this run's own
    entry in the streak; a probe that passed has no "since when" to answer.
    """
    first_seen = current_finished
    consecutive = 1
    for finished, rank in prior_levels:
        if rank == 0:
            break
        first_seen = finished
        consecutive += 1
    return first_seen, consecutive
