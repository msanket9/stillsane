# stillsane

**A drift canary for deployed LLM apps and agents.**

Your LLM app does not crash when it gets worse. It returns 200, latency looks
normal, the error rate is zero, and the output is quietly less correct than it was
last month. You find out when a user complains.

stillsane runs a small set of prompts against your live endpoint on a schedule,
compares each response to a stored baseline, and tells you when behaviour has
moved outside the range that probe normally varies by. Then it answers the
questions the alert provokes -- *since when*, *is it a slow slide or a step*,
*is it my app or the model*, *how big was the change I just accepted*, *did
the number change or just the wording* -- from files it already wrote, with no
new sampling and no service.

It measures change, not quality. It does not know whether your app is good; it
knows whether your app is still doing what it did when you last looked. It
observes from outside, over plain HTTP. There is nothing to instrument, no SDK
to import, no account, and no hosted service: state is a directory in your repo
and one SQLite file.

> **Status: early, pre-0.1.** Everything described below works. The config
> format may still change before 0.1. See [What is here](#what-is-here).

---

## What it looks like

A probe that extracts a total and a due date as JSON. The model was updated, and
it started explaining itself. The numbers are still correct and still present,
but every caller doing `json.loads(response)` is now throwing.

```
DRIFT  extract_invoice @ prod
  semantic_distance           0.133  band <=0.05626           z=+8.9
  length_chars                  106  band 36..52 (floor)      z=+23.2
  completion_tokens              26  band 7..15 (floor)       z=+11.2
  valid_json               0% valid  band >=1
  5 other signal(s) unchanged

  baseline (v1, 2026-07-31):
    {"due_date": "2026-07-01", "total": 1240.50}
  now:
    Sure! I found the following:
    {"total": 1240.5, "due_date": "2026-07-01"}
    Happy to help with more invoices.

------------------------------------------------------------
1 drift   ->  DRIFT
```

Exit code 1, so this fails a build. Nothing errored, nothing was slower, and no
conventional monitor would have noticed.

That `band <=0.05626` was not a number anyone chose. The baseline watched this
probe vary in whitespace, number formatting and key order, and learned how much
that is worth. The prose-wrapped version sits nearly nine times outside it.

The two bands still marked `(floor)` are the tool being honest about how it got
those: this example runs against a tidy mock whose lengths and token counts barely
move, so there was too little spread to measure and they fell back to a built-in
floor. stillsane says which of the two happened rather than presenting a defaulted
number as a measured one. `stillsane bands` reports it in full.

You can run exactly this in about thirty seconds, with no API key, from
[`examples/invoice-extract/`](https://github.com/msanket9/stillsane/tree/main/examples/invoice-extract).

---

## Quickstart

```bash
pip install stillsane
```

```bash
stillsane init       # write a starter config, then point it at your endpoint
```

```bash
stillsane baseline   # capture what "normal" looks like. Explicit, never automatic.
```

```bash
stillsane check      # compare against it. Non-zero exit on drift.
```

Put `stillsane check` on a schedule in CI and you are done. See
[In CI](#in-ci) for a workflow you can copy.

When it fires, the rest reads from disk and costs nothing:

```bash
stillsane history    # since when? what did the model actually say?
```

```bash
stillsane trend      # is this a slow slide no single run would catch?
```

```bash
stillsane bands      # is the band it was judged against a measurement at all?
```

---

## The problem

Three ways an LLM app degrades without ever erroring:

1. **The provider changed the model.** Hosted providers update the model behind an
   endpoint without changing the version string. The same call can return
   meaningfully different output months later, and this is entirely outside your
   control.
2. **Someone edited a prompt.** A tweak to a system prompt or a tool description
   breaks a behaviour three steps downstream that no test covers.
3. **Retrieval drifted.** RAG context shifts, retrieval quality slides, answers get
   vaguer.

In all three cases the app keeps working. Latency is fine. Error rate is zero.
Quality is worse. Nobody gets paged.

This is the same failure mode as a drifting sensor on unattended infrastructure: a
crash is loud and you find out immediately, but a plausible-looking wrong number
gets believed. The fix there was synthetic monitoring: walk the whole pipeline on
a schedule, validate what comes back, alert before a human notices. stillsane is
that, pointed at an LLM.

---

## Should you use this?

Probably not, if you already have something:

- **You want to know whether a prompt is good before you ship it.** Use a
  pre-ship eval framework. There are several good open-source ones, and stillsane
  will not help you. Pre-ship evaluation is a different problem, and tools built
  for it solve it better than a tool of this scope ever will.
- **You already run a tracing or eval platform.** You have evaluator scores on
  real production traffic. Watch those. Adding stillsane buys you
  provider-fingerprint watching and not much else.
- **You are willing to instrument your app.** Then instrument it. Tracing sees
  every real request; stillsane only ever sees the handful of probes you wrote.
  That is a genuine and permanent disadvantage.

stillsane is for the case none of those cover: you shipped an agent or an LLM
pipeline, quite possibly mostly AI-assisted, you have no evals and no
observability, you are never going to add a tracing SDK, and right now you would
find out about degradation from a user complaint.

If that is you, this is a config file and one command.

### How it compares

|                                        | stillsane | Pre-ship eval frameworks | Tracing platforms |
| -------------------------------------- | :-------: | :----------------------: | :---------------: |
| Answers "will this prompt work?"       |     no    |           yes            |      partly       |
| Answers "is what I shipped still fine?"|    yes    |            no            |        yes        |
| Answers "since when, and my app or the model?" | yes |         no            |      partly       |
| Scores real production traffic         |     no    |            no            |        yes        |
| Requires instrumenting your app        |     no    |            no            |        yes        |
| Requires an account / hosted service   |     no    |         usually not      |      usually      |
| Learns per-probe variance              |    yes    |            no            |        no         |
| Alerts on provider fingerprint change  |    yes    |            no            |      rarely       |
| Breadth of providers                   |  narrow   |          broad           |       broad       |
| Assertion / eval library               |  minimal  |          large           |       large       |

Both other columns cover a range of tools that differ considerably from each other;
they are a rough shape, not a specification.

**Where the alternatives win outright:** provider coverage, assertion breadth,
dataset-driven evaluation, red teaming, and maturity. Several eval frameworks also
document a drift workflow (save a baseline, re-run on a schedule, compare), and if
you are happy writing the comparison logic yourself, that gets you a good deal of
what stillsane does.

**What stillsane adds** is the comparison logic and the diagnosis. The comparison
logic: a variance model so the thing does not cry wolf, a baseline that refuses to
update itself, and fingerprint watching. As far as I can tell nothing in the
pre-ship category ships a command that compares a run against a stored baseline
and tells you what moved -- that gap is the original reason this exists. The
diagnosis is what "something moved" becomes once you can also answer *since when*
(`history`, `status`), *whether it is a real shift or noise accumulating below any
single run's threshold* (`trend`), *whether it is your app or the provider*
(`attribute_to`), *whether a specific extracted value changed* (`constant_fields`),
and *how big the last thing you accepted actually was* (`baseline
--compare-previous`). Put together: not "something moved" but "this moved, since
Tuesday, in your app rather than the model, by this much relative to what you last
accepted" -- the question a reader actually has at 9am after a 6am alert, not just
the fact that woke them up.

---

## The interesting part: variance

Most drift tools compare the new output to one stored output and alert past a
fixed threshold. That fails immediately, because probes do not share a variance.

A temperature-0 JSON extraction returns near-identical text every time. A
summarisation probe legitimately rewords itself on every single call. One fixed
threshold either misses real drift on the first or fires constantly on the second,
and a tool that fires constantly gets uninstalled inside a week.

So stillsane learns the band per probe, from the probe's own behaviour:

- At baseline it captures N samples and measures the distances **among them**.
  That distribution is the probe's intrinsic variance.
- At check time it captures M samples and measures the distance from each baseline
  sample to each new one.
- With no drift, those two sets of distances are drawn from the same distribution.
  Both are "how far apart are two independent draws". With drift, the second set
  shifts up.

The obvious objection is that you could just guess: tight threshold for the JSON
probe, loose one for the summariser. Here is what happened when I actually
measured it, against a live model, with two probes picked to make exactly that
point:

```
extract_invoice     JSON, ~60 chars     within-baseline distance  0.128
summarise_quarter   prose, ~230 chars   within-baseline distance  0.066
```

The "deterministic" extraction probe is **twice as variable** as the open-ended
summariser, which is the opposite of what I expected when I wrote the probes.

The reason is visible in the samples. The extraction output is about sixty
characters, and the model sometimes wraps it in a markdown fence and sometimes
does not, sometimes writing `"1240.50 USD"` and sometimes `1240.50`. Three
distinct outputs across five samples. On text that short, formatting variation
dominates. The summaries are all different sentences but all around 230
characters saying the same thing, so they stay close together.

That is the argument for learning the band rather than setting it. I guessed
confidently and was wrong by a factor of two on my own tool; the measurement was
right and both probes passed. Anyone hand-tuning a threshold from intuition would
have set it too tight on the probe that looked deterministic and too loose on the
one that looked chatty.

The headline number is **z**: how far behaviour moved in units of that probe's own
normal variation. "Moved 6.2x further than this probe usually varies" is a
sentence you can act on. A p-value is not.

**On calling it `z`.** It is computed as `(observed − median) / (1.4826 × MAD)`,
which is a *robust analogue* of a standard score, not a standard score. The
1.4826 makes a MAD comparable to a standard deviation for normally distributed
data, and a distribution of distances is emphatically not normal. It is bounded
below at zero and right-skewed. So `z` here assumes no distribution at all: it is
a scale-free measure of how far outside normal something sits, and **it does not
convert to a probability**. `z=6` is not a one-in-a-billion event. The thresholds
(`warn_k: 3`, `drift_k: 6`) are chosen empirically rather than derived from
Gaussian tails, and they are config knobs precisely because the right values are
an open question. Being straight about it: they were tuned against constructed
drift scenarios rather than derived, so treat them as sensible starting points
rather than settled numbers.

You do not have to take that on faith for your own probes. `stillsane calibrate`
reads the `z` values your clean runs already recorded and reports how close each
signal came to firing:

```
summarise_incident @ claude
  signal                      n   |z| p95   |z| max      headroom
  length_chars               13      1.32      1.51          2.0x
  latency_ms                 13      0.77      0.83          3.6x
  semantic_distance          13      0.42      0.68          4.4x

essay_maintainability @ claude
  signal                      n   |z| p95   |z| max      headroom
  length_chars               12      0.95      1.19          2.5x
  semantic_distance          12      0.50      0.59          5.1x
  latency_ms                 12      0.25      0.26         11.4x

extract_invoice @ claude
  signal                      n   |z| p95   |z| max      headroom
  latency_ms                 13      0.11      0.28         10.8x
  length_chars               13      0.00      0.00   never moved
  semantic_distance          13      0.00      0.00   never moved
```

A clean run is one where nothing drifted, so its `z` values are what normal looks
like, and the gap to `warn_k` is the margin before a false alarm. Reported per
probe rather than pooled by bare signal name: two probes both have a
`length_chars`, and their variance is not comparable, which is the entire reason
the band is learned per probe in the first place. Above, `extract_invoice` never
moved at all across thirteen clean runs while `summarise_incident` reached 1.51 --
pooling those into one "length_chars" row would have hidden the probe that
actually mattered behind one that was inert.

The reading above is from real clean runs against a live provider: nothing came
within 2x of the threshold, which says `warn_k: 3` is conservative rather than
trigger-happy on those probes.

It measures **headroom against false alarms only**. Clean runs contain no drift,
so nothing there says whether the thresholds would catch a real regression, and
loosening `k` on that basis would trade a visible problem for an invisible one.
The command says so every time it runs, and refuses to present a
smallest-that-would-not-have-fired value as a recommendation.

`--probe` scopes the report to one probe, the same as `check` and `bands`, which
is useful once a config has several: a CI step gating on a newly added probe's
headroom should not have unrelated probes affecting its exit code. `--strict`
exits 2 if any signal on any considered probe already fires on a clean run.

The Mann-Whitney p-value reported alongside is distribution-free and does carry
its usual meaning, which is exactly why it is supporting evidence and never the
gate.

Related decisions, since they are the ones that determine whether this is usable:

- **Robust statistics throughout.** Median and MAD, not mean and standard
  deviation. At the sample counts anyone will actually pay for, one weird sample
  dominates a standard deviation and barely moves a MAD. The known cost is that a
  MAD collapses to zero when over half the samples are identical, which a
  low-temperature model does often, dropping the band onto its floor where it can
  end up tighter than the probe's own baseline. When that happens the scale falls
  back to an interquartile range, which survives the ties a MAD does not. The
  fallback is conditional because an IQR breaks down at 25% against a MAD's 50%, so
  reaching for it unconditionally would loosen every band in the tool. Samples that
  really are all identical produce a zero IQR too, and then the floor is the honest
  answer. `stillsane bands` reports whichever happened.
- **Baselines never update themselves.** Only `stillsane baseline` replaces one. A
  monitor that silently re-baselines has defined drift out of existence.
- **Clean runs tighten the band, but widening is capped.** Passing runs feed back
  into the variance estimate, so the tool gets *more* sensitive over time at no
  extra cost. Widening is measured against the original baseline rather than
  against yesterday, so drift arriving a little at a time cannot slowly stretch the
  band around itself.
- **Editing a prompt invalidates its baseline.** The config hash covers the prompt,
  system message, scripted turns, target and embedder. Change any of them and
  `check` refuses to compare rather than reporting your own edit as provider
  drift.
- **A transport error is not drift.** A dead endpoint exits with a different code
  than a quality regression, because they call for different responses.
- **Transport failures retry; verdicts never do.** A timeout or a dropped
  connection means the request never landed, so asking again asks the same question.
  A verdict is the opposite: re-running a probe because the answer was DRIFT is
  rolling the dice until it comes up clean, which defines drift out of existence the
  same way a silent re-baseline does. So `retries` covers timeouts, dropped
  connections, 429s and 5xx, and nothing else. A 401 or a malformed body returns
  identical on the second call and only costs money. Each sample records
  `attempts`, so a flaky environment stays visible instead of being smoothed over.

---

## Commands

Python 3.10+, five direct dependencies, no torch. The embedding model is fetched
once on first use (~32MB) and cached. See
[Design constraints](#design-constraints) if you need to stay fully offline.

| Command | Reads | Costs | Answers |
| --- | --- | --- | --- |
| `stillsane init` | your logs, optionally | nothing | a starter config, or one generated from real prompts |
| `stillsane baseline` | the endpoint | N samples per probe, once | what "normal" looks like; with `--compare-previous`, how far it moved from the last version |
| `stillsane check` | the endpoint | M samples per probe, per run | did anything move; exit code for CI |
| `stillsane bands` | `.stillsane/baselines/` | nothing | is each band a measurement, and how often would it cry wolf |
| `stillsane status` | `.stillsane/history.sqlite` | nothing | is the canary itself alive and measuring |
| `stillsane history` | history + `.stillsane/runs/` | nothing | since when; what the model actually said in a given run |
| `stillsane calibrate` | history | nothing | how close clean runs came to the thresholds |
| `stillsane trend` | history + baselines | nothing | a shift too small to cross the threshold on any single run |
| `stillsane watch` | the endpoint | M samples per interval | a sleep loop, for laptops and trials only |

Every command takes `-c/--config` (default `stillsane.yaml`), and every reporting
command takes `--json` for the same result structured. The three questions an
alert provokes are below; `check` and `baseline` are covered under
[Config](#config).

### Is the canary alive?

Every other command answers a question about the model. `stillsane status` answers
one about the tool: has it actually been running, and did its runs measure
anything.

```bash
stillsane status --expect-every 24h
```

```
last run        22 minutes ago   pass
last clean run  22 minutes ago
runs recorded   6
history since   2026-07-28, 6 run(s) recorded
recent          P P E P E P   (oldest to newest)

  essay_maintainability @ claude  errored 2 of 5 run(s)
                                    timeout after 60.0s
                                    ReadError:
  extract_invoice @ claude        errored 1 of 6 run(s)
                                    ReadError:
  summarise_incident @ claude     errored 1 of 6 run(s)
                                    ReadError:

2 of the last 6 run(s) ended in transport errors rather than drift. Nothing
was measured on those runs. That is an environment problem, not a model one.
```

That is a real week of scheduled runs, and it is the failure mode worth planning
for. A monitor can fail for reasons that have nothing to do with what it watches:
a laptop asleep at the trigger, a network that had not come up, a timeout tuned
for a faster probe. Each one produces a run that completed, recorded an ERROR and
moved on. `check` cannot report it because each run only sees itself, and
`history` shows the rows but leaves you to notice the pattern.

Two distinctions do most of the work. **Transport errors are not drift**: a run
that could not reach the endpoint measured nothing, so counting it as healthy
overstates your coverage. And **silence is not success**: a canary that stopped
running looks exactly like one with nothing to report, which is what
`--expect-every` exists to disambiguate. Without it, staleness is unknowable and
the command says so rather than guessing a cadence from past gaps.

**A third kind of silence is a lost history rather than a stopped canary.** If
`.stillsane/history.sqlite` lives in a CI cache (see [In CI](#in-ci)) rather than
somewhere durable, a cache miss resets it without a single run ever failing --
the canary keeps reporting, it just forgot everything before today. `runs
recorded` alone looks the same the day after a reset as it does a month in,
because it is bounded by `--limit`. `history since` is not: it always reflects
every run this database has ever recorded, so "history since 6 hours ago, 3
run(s)" after a month of daily runs is the tell that the cache, not the canary,
is what broke.

`--strict` exits 2 when the canary is unhealthy or overdue, for a second cron job
whose only purpose is to notice that the first one stopped.

### Since when?

`status` says whether the canary is alive; `history` says what it recorded.

```bash
stillsane history
```

```
Last 3 run(s), most recent first:
  2026-08-08T04:01:56+00:00  pass    8a589970b3ad
  2026-08-08T04:01:55+00:00  pass    b91bdf3d81c2
  2026-08-08T04:01:54+00:00  pass    6a6934e6a887
```

A run can also land as `warn`, `drift` or `error`, and `error` means the endpoint
could not be reached rather than that anything moved. See [Exit codes](#exit-codes).

The question an alert always provokes is when it started, so one signal can be
followed over time:

```bash
stillsane history --probe summarise_incident --target claude --signal semantic_distance
```

```
semantic_distance  summarise_incident @ claude   (most recent first)
  2026-08-08T03:49:14+00:00      0.02034  z=+0.0
  2026-08-07T05:20:09+00:00      0.02952  z=+0.0
  2026-08-06T05:14:46+00:00       0.0341  z=+0.2
```

`--signals` lists everything that has been recorded, so you do not have to
remember signal names to look at your own data. The alert itself carries the
answer too: each probe in the JSON payload has `first_seen` and
`consecutive_runs`, and the Slack headline says `(day 4)` once a streak is more
than a passing mention. See [Alerts](#alerts).

**What did the model actually say?** `history` has numbers, not text --
investigating an alert from a few hours ago used to mean finding whatever log
captured that run's stdout, which for a scheduled job usually means digging
through CI. Every `check` run keeps its own current-run samples under
`.stillsane/runs/<run_id>/`, using the same run id `history` lists above:

```bash
stillsane history --run 8a589970b3ad
```

```
extract_invoice @ prod
  1. Sure! Here you go: {"total": 1240.50, "due_date": "2026-07-01"}. Anything else?
  2. Sure! Here you go: {"total": 1240.50, "due_date": "2026-07-01"}. Anything else?
  3. Sure! Here you go: {"total": 1240.50, "due_date": "2026-07-01"}. Anything else?
```

Only the last 50 runs are kept, oldest pruned first, so a `watch` loop cannot
grow this without bound. These files hold what your endpoint said, which for a
`type: http` target against your own app can include tenant data; they are local
state, not something to commit. Only `.stillsane/baselines/` is meant for git --
`.stillsane/runs/` and `.stillsane/history.sqlite` are not currently gitignored
by default, so add them to your own `.gitignore` if you follow the README's
advice to commit `.stillsane/baselines/`.

### Inspecting the bands

`check` tells you whether a probe moved. `stillsane bands` answers the question
underneath it: is the band it would be judged against a measurement at all?

```bash
stillsane bands
```

It reads only what is already on disk, so it costs nothing, needs no API key, and
touches no network. It reports every band, including the pointwise ones that never
appear in the capture-time warning, and names the ones that will misreport:

```
extract_invoice @ claude   (v1, 8 sample(s), captured 2026-08-04)
  semantic_distance      band <=0.02 (floor)          28 pairs   spread 0..0.1276
    would report drift on ~0.2% of clean runs (median of 24 pairs)
    COLLAPSED: the median and MAD are both zero, so the scale could not be
    measured and the band fell to its floor. The baseline itself spans
    0..0.1276, and 6 of 28 pairs (21%) fall outside the band that was built
    from them. The width is a built-in default rather than anything this probe
    demonstrated, so it is arbitrary in both directions: see the rate above
    for how often it actually fires. Typically the output is bimodal,
    identical on most runs and formatted differently on the rest. More samples
    will not help while one form dominates, because the median stays put and
    the MAD stays zero.
  length_chars           band 60..76 (floor)           8 values  spread 56..68
    would report drift on ~4.7% of clean runs (median of 3 values)
    COLLAPSED: ...

2 band(s) will misreport: length_chars, semantic_distance
A collapsed band is not fixed by recapturing: while one output form dominates,
the median stays put and the scale stays zero. Constrain the prompt so the
probe has one output regime, or pin the band explicitly in config.
```

(The second `COLLAPSED` paragraph is elided above; it repeats the first with that
signal's own numbers.)

That is a real baseline against a real provider, and it is the failure worth
knowing about. When a probe returns byte-identical output most of the time and a
different formatting the rest, the median pair distance is zero and so is the MAD.
The scale collapses, the band drops to its floor, and the result looks exactly
like every other band. It is not one: 45% of the baseline it was built from
already sits outside it.

More samples do not fix that one, which is why it gets a different message from an
ordinary floored band. While one formatting dominates, the median stays put and
the MAD stays zero however many you take.

It also estimates how often each band would cry wolf:

```
  latency_ms             band <=5174                   8 values  spread 2360..7144
    would report drift on ~4.1% of clean runs (median of 3 values)
```

That number is the one worth acting on, and it is not the same as how much of the
baseline sits outside the band. A check never compares a single value: it reduces
the run to a median and compares that. So the estimate resamples from the
baseline's own distribution, takes the median of a check-sized draw, and counts how
often it lands outside.

The difference is large. On a real baseline, an essay probe had 15% of its pairs
outside its band and an estimated false alarm rate of **0%**, because a pairwise
check medians two dozen distances and the tail never moves it far enough. A latency
signal had 12% of its values outside and a **4.1%** rate, because its median is
over three values and scatters. Same-looking numbers, opposite verdicts, which is
why the draw size is printed alongside.

It is an estimate from one baseline rather than a measured rate, and it assumes a
clean run looks like the baseline. That is the assumption the band already makes,
so it adds no new leap, but a small baseline estimates it coarsely.

It also names a baseline `check` is about to refuse. `bands` recomputes every band
from stored numbers regardless of whether the config that produced them still
matches, but a clean "all bands look sound" on a baseline whose config hash has
since moved reads as "this is fine" when it is actually "recapture before this
tells you anything about what `check` will do":

```
extract_invoice @ prod   (v1, 5 sample(s), captured 2026-08-04)
  stale: config has changed since capture; `check` will refuse this baseline until `stillsane baseline` recaptures it
```

A label, not a verdict: it never turns into a suspect finding or a nonzero exit
code on its own, even under `--strict`.

`--strict` exits 2 when any band will misreport, for a CI job that should fail on
a baseline this shape. `-v` shows every band rather than only the interesting ones.
`--json` always includes every band, sound ones included, since a consumer diffing
bands between runs needs to tell "still sound" from "no longer reported".

### A shift too small to ever cross the threshold on its own

`check` judges one run at a time, and that is a real blind spot: a provider that
quietly moves `semantic_distance` from z~0.3 to a steady z~2.4 never crosses
`warn_k: 3`, so `check` says PASS every single day. The evidence is there --
every run's own numbers say so -- but nothing ever looks across more than one
run to notice the floor itself moved. `stillsane trend` does:

```bash
stillsane trend
```

```
extract_invoice @ prod
  semantic_distance        early z=+0.30  recent z=+2.41  (12 run(s), window 5) SUSTAINED SHIFT
    moved from 0.0224 (early, inside the normal range) to 0.0631 (recent, past
    1.80x normal variance) without any single run crossing warn_k -- each run
    alone still reads as PASS.
  length_chars              early z=+0.10  recent z=+0.22  (12 run(s), window 5)

1 signal(s) show a sustained shift: semantic_distance on extract_invoice @ prod.
Each of these has been PASSing every run -- no single check ever crossed
warn_k -- while the recent median moved somewhere the early median had not.
That is what this command exists to catch: `check` judges one run at a time and
cannot see it.

Reads history only: no probe was sampled to produce this. A sustained shift is
exactly what `stillsane baseline` should absorb once you have looked at it --
recapturing resets the comparison, since it is scoped to the baseline version
currently on disk.
```

Per probe/signal, it takes the median of the earliest runs recorded against the
*current* baseline version and the median of the most recent ones (`--window`,
default 5 runs per side), and expresses both against the same, fixed band `bands`
would show today. "Fixed" is deliberate -- a signal's band tightens over time as
clean runs pool into it, so trending the `z` each run recorded *at the time* would
be comparing numbers measured on different scales and calling the difference a
shift. A sustained shift is reported when the early group sat inside `warn_k *
grey_zone` (the same "elevated but not WARN-worthy" fraction the corroboration
check already uses) and the recent group has moved past it.

Scoped to the baseline version on disk: runs recorded against a since-replaced
baseline are never averaged in with current ones -- otherwise the first runs after
a recapture would read as a shift that is really just the old baseline's tail.

`--strict` exits 2 if any signal shows a sustained shift, for a weekly job distinct
from the daily `check` -- a sustained shift is not itself an alert-worthy event on
the day it is first seen, it is a pattern worth a human noticing on a slower
cadence.

### Informed re-baselining

A DRIFT fires, you decide it is the new normal, and run `stillsane baseline`.
Ordinarily v1 is kept on disk but never looked at again -- you have just
accepted a shift without being told its size. Same question after editing a
prompt: how different is the new version's output from the old one?
`--compare-previous` answers it, using the exact comparison `check` would:

```bash
stillsane baseline --compare-previous
```

```
  extract_invoice @ prod: v2, 5 sample(s), fingerprint fp_a4f2b1
    compared to the version it replaced (v1 -> v2, config changed):
      semantic_distance          0.081  band <=0.02             z=+8.9
      length_chars                  74  band 62..70             z=+2.1

Captured 1 baseline(s). These will not change until you run this again.
```

Purely informational: whatever it shows, `baseline` still exits 0 and the new
version is already written regardless. It is not a preview you can act on before
committing to the recapture -- v2 exists either way, this is what happened, after
the fact. `config changed` is shown whenever the version just replaced was
captured under a different prompt, model or embedder, which is the common reason
to reach for this flag in the first place (a prompt edit) -- said explicitly so a
real move reads as "the edit did this" rather than as the provider changing
underneath an unrelated re-baseline.

### watch

`stillsane watch --interval 3600` is a sleep loop, and honestly so. cron or CI
does this better: they survive reboots, they log, and they can tell you when the
job itself stopped running, which a bare process cannot do for itself. `--once`
runs a single iteration, which is a convenient way to try a config.

---

## Config

Plain YAML, meant to live in git and be diffed like code.

```yaml
targets:
  - name: prod
    type: openai_compatible
    base_url: https://api.example.com/v1
    model: some-model-id
    api_key_env: PROVIDER_API_KEY   # the variable name, never the key itself
    watch_fingerprint: true
    timeout_s: 60                   # per request
    retries: 1                      # transport failures only, never a verdict
    retry_backoff_s: 2              # doubles per attempt
    temperature: 0                  # optional; sent as-is to the provider
    max_tokens: 400                 # optional

probes:
  - id: extract_invoice
    prompt: "Extract the total and due date as JSON from: ..."
    baseline_samples: 5             # paid once, this is where variance comes from
    check_samples: 3                # paid every run, only needs to find a median
    checks:
      - valid_json
      - has_keys: [total, due_date]
      - semantic_similarity: auto   # learned band, not a fixed number

alerts:
  webhook: https://hooks.example.com/...
```

`samples: 5` also works and sets the baseline count. A probe runs against every
target unless it names some with its own `targets:` list.

### Your own app rather than a model API

This is the case the tool is really for: most people are not watching a raw
model, they are watching the thing they shipped, which has its own retrieval,
prompt assembly and bugs in front of it. Use `type: http` and describe the
request:

```yaml
targets:
  - name: prod
    type: http
    base_url: https://your-app.example.com
    path: /api/extract
    method: POST                       # the default
    headers:
      x-tenant: acme
    body:
      document: "{{prompt}}"           # {{prompt}}, {{system}} and {{turns}} are substituted
    response_path: data.reply          # where the text lives in the response
```

`response_path` takes dotted paths with indexes (`choices.0.message.content`) and
a filter form (`content[type=text].text`). The filter matters on Anthropic: with
thinking enabled `content.0` is the thinking block, not the answer. Editing
`response_path`, `method`, `path`, `body` or `headers` invalidates the baseline,
because any of them can change which backend answers or which part of the answer
is compared.

**Is it my app or the model?** A `type: http` probe against `prod` above fires.
Which of the three causes was it -- the provider, a prompt edit, or retrieval? On
its own the tool cannot say. But most apps sit on a model you can also probe
directly, and running the *same probe text* against both separates the cases:
moved on the app but not on the raw model means the change is inside the app;
moved on both is consistent with the provider. List both targets on the probe
and point the app at its control:

```yaml
targets:
  - name: prod
    type: http
    base_url: https://your-app.example.com
    path: /api/extract
    body: {document: "{{prompt}}"}
    response_path: data.reply
    attribute_to: raw   # a target named below

  - name: raw
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY

probes:
  - id: extract_invoice
    targets: [prod, raw]   # both, or there is nothing to compare against
    prompt: "Extract the total and due date as JSON from: ..."
    checks: [valid_json, {has_keys: [total, due_date]}]
```

When `prod` moves, the report gains one line:

```
DRIFT  extract_invoice @ prod
  semantic_distance           0.133  band <=0.05626           z=+8.9
  ...
  -> attribution: 'raw' (control) did not move -- looks local to 'prod', not the
     provider. Not proof 'raw' itself is unchanged: 'prod' likely wraps its own
     system prompt and retrieval.
```

No new sampling: this costs exactly what listing two targets on the probe
already costs, and needs both to have actually run in the same check -- a
control missing a baseline, or simply not scoped to this probe, leaves the line
off rather than guessing. It is never proof either way, only ever "the model's
behaviour on this exact text did or did not move", which the line always says in
full rather than leaving implied. Fingerprints are deliberately excluded from
"did the control move": they vary by account and region even against an
unchanged model, so a fingerprint-only blip on the control would make attribution
noisier than the thing it is meant to corroborate. A probe that errored gets no
attribution line at all: nothing was measured to attribute.

### Providers that do not use `Authorization: Bearer`

Anthropic wants `x-api-key` with no prefix, Azure wants `api-key`. Both are
reachable without putting a live secret in `headers`:

```yaml
targets:
  - name: claude
    type: http
    base_url: https://api.anthropic.com
    path: /v1/messages
    api_key_env: ANTHROPIC_API_KEY
    api_key_header: x-api-key
    api_key_prefix: ""
    headers:
      anthropic-version: "2023-06-01"
    body:
      model: claude-opus-5
      max_tokens: 2048
      messages:
        - role: user
          content: "{{prompt}}"
    response_path: content[type=text].text
```

### Running probes through a Claude Pro or Max subscription

If you already pay for Claude Code, a drift canary should not need a second,
separately billed key just to sample a probe. `type: claude_code` shells out to
the `claude` CLI already installed and authenticated on this machine, so a probe
draws on whatever that login already covers:

```yaml
targets:
  - name: claude
    type: claude_code
    model: claude-opus-5   # optional; omit to use claude's own default

probes:
  - id: haiku
    prompt: "Write a three-line haiku about autumn leaves."
    baseline_samples: 3
    check_samples: 2
```

That is a real, complete example, verified against a live install:

```
PASS   haiku @ claude
  semantic_distance          0.4351  band <=0.5915            z=+0.0
  length_chars                   82  band 68.1..85.9          z=+1.7
  completion_tokens              34  band 23.1..40.9          z=+0.8
  cost_usd                $0.039313  band <=0.04317 (floor)   z=+0.0
  latency_ms                 4846ms  band <=7618 (floor)      z=+0.0
  model_id             claude-opus-5
  response_complete        complete  band >=1

------------------------------------------------------------
1 pass   ->  PASS
```

Unlike the mock-provider examples elsewhere in this README, these exact numbers
cannot be reproduced -- a real model genuinely varies run to run, which is the
whole reason a band exists rather than a fixed threshold. Re-running the same
probe against the same install produced a WARN a few minutes later, on the same
haiku prompt, for the same honest reason: token count drifted a little further
than usual. The shape shown -- which signals appear, what they measure -- is real
and stable; the values will differ every time you run it yourself.

Two things worth being straight about before you rely on this daily, both found
by testing against a real install rather than assumed:

- **Whether `cost_usd` is money actually charged beyond your subscription is not
  something this tool can tell you.** Claude Code reports a cost figure for its
  own usage tracking regardless of how a session is authenticated, and this target
  simply passes that number through. Check your own account before assuming it is
  free.
- **`--bare` mode was ruled out on purpose.** Its own `--help` text says OAuth and
  keychain auth are never read there, which would force the very API key this
  target exists to avoid. Running in ordinary mode instead means accepting a
  larger tool surface, and every tool is denied by default -- but tool *denial* is
  not the same as tool *use never being attempted*. Three identical adversarial
  prompts under identical deny flags produced three different garbled attempts to
  invoke one anyway, never the same way twice, though nothing observed suggested a
  command actually ran. Denied output that looks like this is detected and marked
  as an error rather than silently compared against a baseline as if it were real
  content. Probes that read as an instruction to look something up, check
  something or run something are the ones most likely to trigger it; plain
  generation -- summarise, extract, write, the haiku above -- has not shown this
  behaviour in testing.

For a probe that is genuinely supposed to use tools -- testing against a real
dataset, say -- name exactly which ones with `allowed_tools`:

```yaml
    type: claude_code
    allowed_tools: [Read, Glob, Grep]   # read-only; nothing else is available
```

Deliberately an allowlist rather than an `agentic: true` switch: an unattended
daily cron job silently granted broad tool access is a materially larger risk than
one that can only do exactly what it was told it may do. This mode has had far
less real-world testing than the default and no MCP server is ever reachable
either way, regardless of what is configured on the machine running the check.

`claude_command` overrides the binary invoked, if `claude` on `PATH` is not the
right one to use. `temperature` and `max_tokens` are accepted in config but have
no effect on this target -- the `claude` CLI's `-p` mode has no flag for either.
`turns` (below) is not supported on it, for the same reason.

### On every target

`timeout_s`, `retries` and `retry_backoff_s` as shown above. `escalate_fingerprint`
makes a changed fingerprint fail rather than warn; `watch_fingerprint: false`
drops the signal for a provider whose fingerprint churns on its own.

`store_raw: true` persists each sample's full decoded response body into
`samples.jsonl` at baseline capture time and into `.stillsane/runs/`. Off by
default: nothing in stillsane reads it, and `.stillsane/baselines/` is meant to
be committed to git, which for `type: http` against your own app means whatever
your API actually returned -- tenant data included -- landing in version-control
history that is not easily purged after the fact. Note that the *extracted text*
is always stored; `store_raw` only controls the envelope around it. Turn it on
per target only once you have an actual reader for it and have checked what that
target's responses contain.

**On cost.** Sampling is the whole mechanism, so it is worth being explicit: the
expensive part is the baseline, and you pay it once. Routine checks need only
enough samples to locate a median, because the variance estimate already lives in
the baseline. Embeddings run locally and cost nothing. The LLM judge is opt-in and
only fires when a band has already been crossed, so a normal run spends nothing
beyond the probe calls themselves.

That claim is checkable, not just asserted: `check` sums `cost_usd` across every
sample where a gateway or the `claude` CLI reported one -- the judge's own call
included, on the runs where it fires -- and prints the total in the footer and in
`--json`'s payload (`calls`, `cost_usd`, `cost_known_calls`):

```
this run: 9 calls, $0.0123
```

Omitted entirely when nothing reported a cost, which is most gateways -- a
confident `$0.0000` would be a fabricated number, not a measurement. When only
some calls priced themselves, the line says so rather than reading as a total:
`this run: $0.0123 across 6 of 9 calls`.

### Multi-turn probes

A probe is one message by default, and drift that only shows up two or three
turns into a conversation is invisible to that. `turns` scripts fixed history
in front of the probe's own live turn:

```yaml
probes:
  - id: invoice_followup
    turns:
      - role: user
        content: "Here is an invoice: $40 line one, $60 line two."
      - role: assistant
        content: "Got it -- what would you like to know?"
    prompt: "What's the total?"
```

Every entry in `turns` is fixed and sent verbatim on every sample; only the
final turn, `prompt`, is ever live. An earlier assistant turn answering for
itself on each sample would compound variance across turns until the band
stopped meaning anything, which is why history is scripted rather than
replayed from a real prior run. Editing `turns` invalidates the baseline, the
same as editing `prompt`.

For `type: http`, `{{turns}}` is `turns` alone -- the scripted history, not
including the live turn -- so it belongs alongside an explicit final message
built from `{{prompt}}`:

```yaml
body:
  messages:
    - "{{turns}}"
    - role: user
      content: "{{prompt}}"
```

As a list element, `"{{turns}}"` splices the real list of `{role, content}`
objects into `messages` in place, so the result is a single flat array:
scripted history, then the live turn, exactly what OpenAI's and Anthropic's
`messages` arrays expect. Used as an entire field's value on its own
(`messages: "{{turns}}"`), it substitutes just the scripted history with nothing
live in it at all, which is rarely what you want. Placeholders are substituted
in one pass: a prompt that itself contains the text `{{system}}` is sent with
that text intact.

Not supported on `type: claude_code`: the `claude` CLI's `-p` mode has no flag
to inject prior assistant turns, so a probe using `turns` must be scoped away
from any `claude_code` target with the probe's own `targets:` field.

### Checks

| Check | Meaning |
| --- | --- |
| `valid_json` | The whole response parses as JSON. A markdown fence is allowed; surrounding prose is not, because that is what breaks a caller's `json.loads`. |
| `has_keys: [a, b]` | Those keys are present in the JSON, found leniently. Deliberately separate from `valid_json`, so the report can say the data survived even when the envelope broke. |
| `constant_fields: [total, due_date]` | Those fields' *values*, found leniently, must stay whatever the baseline learned them to be. |
| `semantic_similarity: auto` | Learn the band. The default. |
| `semantic_similarity: 0.9` | A fixed floor on cosine *similarity* (1 = identical): a response less than 90% similar to the baseline is drift, i.e. a distance ceiling of `0.1`. `semantic_distance: 0.1` is the same rule written as a distance. The report marks a configured band `(pinned)` rather than a learned one. |
| `max_length: 2000` | Hard cap on response length, as a second signal alongside the learned `length_chars` band. |

Several signals are always on and need no configuration: semantic distance, JSON
shape, tool-call shape, length, completion tokens, cost, latency, provider
fingerprint, model id, and whether the response finished on its own rather than
getting cut off by the token limit. Signals that do not apply to a probe stay
silent, so tool-call drift says nothing about a probe that never calls a tool. A
signal that did not apply at baseline and does now -- an agent that started
calling a tool, a probe that started returning JSON -- is reported as a change,
not skipped.

The truncation one exists because of a real miss: an essay probe run for weeks
against a live provider truncated on 7 of 8 samples (`stop_reason=max_tokens`) and
nothing in the tool said so. Length and semantic distance alone do not reliably
catch this, because a truncated response can still be longer than the baseline and
stay close to it right up to where it stops -- everything up to the cutoff is
genuine content. Reads `finish_reason` (OpenAI-shaped targets) or `stop_reason`
(Anthropic, checked generically so a raw `http` target gets it too), and is silent
rather than guessing on a provider that exposes neither.

**`constant_fields` exists because the flagship example above has a gap.**
`extract_invoice` pulls `total: 1240.50` out of an invoice. If the model starts
returning `1204.50` -- a transposition, not a formatting change -- `valid_json`
still passes, `has_keys` still passes, and `semantic_distance` on a 45-character
JSON string is not built to notice a single digit swap; whether it happens to land
inside a learned band is luck. That is a content change on the exact probe this
README leads with, and nothing was aimed at it until this check:

```yaml
checks:
  - valid_json
  - has_keys: [total, due_date]
  - constant_fields: [total]
```

```
DRIFT  extract_invoice @ prod
  constant_field[total]              1204.5
  2 other signal(s) unchanged

  baseline (v1, 2026-08-04):
    {"total": 1240.50, "due_date": "2026-07-01"}
  now:
    {"total": 1204.50, "due_date": "2026-07-01"}
```

The row itself shows only the new value; the baseline value it learned (`1240.50`)
is named just below, in the same before/after block every other content signal
uses -- and in `--json`/the webhook payload, `detail` spells out the transition
directly as `"1240.5 -> 1204.5"`.

**Learned, not asserted.** The expected value is never typed into config -- it is
read off the baseline the same lenient way `has_keys` reads presence, and whatever
value (or values) it finds there becomes the known-good set. A field that took more
than one value across baseline samples stays exactly that tolerant afterwards; only
a value genuinely never seen at baseline is reported, so a field that legitimately
varies is not locked onto the first sample that happened to run. Floats are
compared numerically (`1240.5` and `1240.50` parse identically, but `1240` and
`1240.0` do not by plain string comparison, and normalising through `float()` first
is what keeps a dropped decimal point from reading as the total changing). A field
missing from the response entirely is `has_keys`'s question, not this one's --
`constant_fields` stays silent on it rather than duplicating that finding.

**Opt-in per field, not automatic across every key.** Five baseline samples cannot
prove a field never legitimately varies; locking every extracted key in by default
would page someone the first time a field that varies one time in twenty happens
to do so on baseline capture. Naming the fields worth watching is the point --
`total` and `due_date` matter for an invoice extractor; a `notes` field probably
does not.

### Thresholds

```yaml
thresholds:
  warn_k: 3          # WARN when |z| exceeds this
  drift_k: 6         # DRIFT when |z| exceeds this
  min_confident_n: 4 # below this many baseline samples, DRIFT is capped to WARN
```

The defaults are the ones every number in this README was measured against.
`calibrate` is how you find out whether they are right for your probes; change
them only on that evidence.

### Alerts

```yaml
alerts:
  webhook: https://hooks.example.com/...
  slack_webhook: https://hooks.slack.com/services/...
  fail_on_warn: false   # WARN never fails a build unless this is true
```

Fired on every non-PASS run, to `webhook` (the same JSON `--json` prints),
`slack_webhook` (a short headline plus the plain-text report), or both. Delivery
is best-effort: a webhook that is down is reported on stderr and never turns a
successful check into a failed one. Both payloads contain the before/after
excerpts -- up to 400 characters of what your endpoint said -- so pick a
destination accordingly.

**"Since when?"** is the first question any alert provokes, and the alert answers
it rather than sending the reader to `history`. Each probe in the JSON payload
carries `first_seen` (when its current run of non-PASS verdicts began) and
`consecutive_runs` (how many in a row, this one included, with no clean run in
between) -- a recovery resets both, so a probe that broke again after passing
reads as day one of a new incident, not a continuation of the old one. The Slack
headline picks up the longest-running of the probes that moved once it is more
than a passing mention: `DRIFT (day 4) (1/3 probes moved)`.

**`alerts.repeat_every`** suppresses resending an *unchanged* verdict. Unset (the
default) sends every non-PASS run, which is the safe choice -- a monitor that can
go quiet on its own judgement is one step from the exact "silence looks like
success" failure this tool exists to prevent. `repeat_every: 0` sends only when a
probe's verdict first changes and suppresses every later repeat of the same
streak; a positive number instead resends every that-many runs, so a long-running
issue is not forgotten entirely. A verdict that just changed always alerts
regardless of this setting. When a run is suppressed, `check` still prints why on
stderr and the verdict and exit code are completely unaffected -- only the
notification is skipped.

### The judge (optional)

Add a `judge` block and a probe that crosses its band gets one extra call, to
explain in a sentence what changed:

```yaml
judge:
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
```

```
DRIFT  extract_invoice @ prod
  semantic_distance           0.133  band <=0.05626           z=+8.9
  valid_json               0% valid  band >=1
  ...
  -> breaking: Still valid JSON, but now wrapped in conversational prose.
```

**It only runs on probes that already failed their band**, so on a day when
nothing drifted it is never called and costs nothing. That tiering is the point:
structural checks are free, embeddings are free after the one-time download, and
the only paid layer fires when something is already known to be wrong.

Two deliberate limits:

- **It is advisory.** By default it explains and nothing else. The verdict came
  from a band learned out of the probe's own measured behaviour, and a model that
  saw two samples does not get to overrule that. The judge also reads your
  endpoint's output verbatim, so a compromised or adversarially prompted upstream
  can talk to it directly. Set `can_downgrade: true` to let it soften a drift it
  considers purely cosmetic, once you trust both.
- **It gets its own endpoint.** Not a flag on a target, because judging with the
  same deployment you are watching means a provider-side change moves both the
  thing being measured and the instrument measuring it.

If the judge is unreachable or answers with something unparseable, the run is
unaffected: the verdict stands and the explanation is simply absent.

### Generating probes from logs

Writing twenty probes by hand is the reason most people never start, and the ones
you would write are the ones you already think about. The prompts actually hitting
your endpoint are a better sample, and you already have them:

```bash
stillsane init --from-logs requests.jsonl
```

```
Read requests.jsonl
  61 distinct prompt(s), clustered into 3
```

That number is the point. Real logs are enormously repetitive: a thousand requests
are usually a handful of shapes with different payloads stuffed into them. Near
duplicates are clustered by meaning using the embedder that already ships for
drift detection, and the most frequent variant of each cluster becomes the probe,
annotated with how often it appeared.

Reads JSONL, a JSON array, or a directory of `.json` files. Each record can be an
OpenAI-style request body, an Anthropic-style one (top-level `system`, plain or
multipart), a bare `{"prompt": ...}`, or any of those wrapped under `request`,
`body` or `payload`. Malformed lines are skipped, because refusing a 10,000-line
log over one line truncated mid-write would make the feature useless on exactly
the files it exists for.

**Checks are emitted commented out.** Guessing that a probe returns JSON and being
wrong would fail your first baseline and teach you the tool is broken. You get the
prompts and a suggestion; you decide what holds.

| Flag | |
| --- | --- |
| `--limit N` | Most probes to emit, most frequent first. Default 20. |
| `--merge-distance D` | How aggressively to cluster. Higher merges more. Default 0.12. |
| `--probes-only` | Emit just the `probes:` block, for pasting into a config you already have. |
| `--embedder hashing` | Cluster without the embedding model, fully offline. |

### From Python

Everything the CLI does goes through `stillsane.runner.check` and
`stillsane.runner.capture_baseline`, which take a `Config`, a `BaselineStore`
and an optional `httpx.AsyncClient` and return dataclasses -- that is how the
entire test suite runs with no network. They are usable from a pytest today. They
are not yet a stable API: the signatures may move before 0.1, after which they
will not.

---

## Exit codes

| Code | Meaning |
| :--: | --- |
| `0` | No drift. |
| `1` | Drift. |
| `2` | Warning only. Does not fail a build unless `fail_on_warn: true`. |
| `3` | Error. The endpoint failed, the config is invalid, or there is no usable baseline. Nothing was measured. |

---

## In CI

Copy
[`examples/invoice-extract/github-actions.yml`](https://github.com/msanket9/stillsane/blob/main/examples/invoice-extract/github-actions.yml)
into `.github/workflows/`. It runs `stillsane check` every morning, caches the
embedding model and the drift history between runs, and fails the job on drift.

Kept as one file rather than pasted here as a second copy, because two copies of a
workflow drift apart and the one in the README is the one nobody re-tests.

Three things it relies on:

- **Commit `.stillsane/baselines/`.** The workflow needs something to compare
  against. They are plain text and diff like code. `.stillsane/history.sqlite`
  and `.stillsane/runs/` change on every run and are local state, not
  reference outputs -- add them to your own `.gitignore` alongside
  `.stillsane/baselines/` staying tracked.
- **History lives in a cache, and the workflow saves it even when `check` fails.**
  Every checkout starts fresh, and without *some* history `status`, `history`,
  `calibrate` and `trend` have nothing to read. The workflow restores
  `.stillsane/history.sqlite` with `actions/cache/restore` and saves it with
  `actions/cache/save` under `if: always()`. The split matters: a DRIFT exits 1,
  a failed step fails the job, and the plain `actions/cache` action does not save
  on a failed job. Without the split, exactly the runs you most want in history
  -- the ones that fired -- are the ones that vanish, and every alert reads as
  day one. This is still best-effort, not durable: a cache eviction resets it
  silently, with no run ever failing, which is why `stillsane status` prints
  "history since <date>, N runs" from the database's true, unbounded age. Watch
  that line, not just the exit code.
- **A daily schedule is the point.** Provider-side model changes arrive without
  warning; finding out within a day is the entire product. Pin the package
  version in the workflow; the config format is not frozen yet, and a scheduled
  job that upgrades itself is how a config breaks at 6am.

**Gating a PR that edits a probe.** A prompt edit changes the config hash, so an
ordinary `check` refuses to compare against the old baseline -- correctly, since
comparing new output to an old baseline is not drift, it is the edit doing what it
was asked to do. That also means a PR check gating on plain `check` can never
actually review the edit; it just blocks on "recapture first, then commit, then
push again."

`stillsane check --against-stale` compares anyway. Every verdict it produces is
capped at WARN (never DRIFT, never the ERROR exit code) and never updates the
baseline's variance pool, and the report says so on every line: `comparison
against a baseline captured under a different config; verdicts are indicative`.
Exit code is always `0` or `2`. It exists to let a reviewer see the diff a prompt
edit produced, not to replace `stillsane baseline` -- a real baseline still needs
capturing before the next scheduled run. Do not wire it up as the only check on
a probe-editing PR without also requiring a recapture commit: the flag makes an
edit visible, it does not validate it.

---

## What is here

Release notes are in [CHANGELOG.md](https://github.com/msanket9/stillsane/blob/main/CHANGELOG.md).

Working end to end:

- The comparison engine: variance bands, effect sizes, verdict aggregation,
  variance pooling with the caps that stop gradual drift widening its own band
- Signals: structural (`valid_json`, `has_keys`, `constant_fields`,
  `max_length`), semantic, JSON shape, tool-call shape, fingerprint, model id,
  tokens, cost, latency, response truncation
- Targets: OpenAI-compatible, arbitrary HTTP (with `{{turns}}` for multi-turn
  probes), and the local `claude` CLI
- Versioned baseline store, SQLite history, per-run sample store, and the config
  hash that refuses a stale comparison
- `init` (with `--from-logs`), `baseline` (with `--compare-previous`), `check`
  (with `--against-stale`), `bands`, `status`, `history`, `calibrate`, `trend`,
  `watch`
- Webhook/Slack alerts with streak tracking and `repeat_every`; `attribute_to`
  control targets; the optional LLM judge

The test suite runs with no network, no API key and no model download. It ships
in the sdist, so you can verify the variance model yourself rather than taking
this README's word for it:

```bash
pip install -e ".[dev]" && pytest
```

There is a runnable worked example in
[`examples/invoice-extract/`](https://github.com/msanket9/stillsane/tree/main/examples/invoice-extract), with a committed baseline
and a mock provider, so you can watch a real regression get caught without an API
key. CI builds the wheel, installs it into an empty environment and runs that
example on every push, which is how a broken install gets caught before a release
rather than after one.

**Expect breakage before 0.1.** The config format is not frozen. If a field is
renamed you will get a validation error naming it, not a silent misread, but a
version pin is wise for now.

---

## Scope

stillsane detects that a deployed endpoint's behaviour changed, and helps you
work out since when, where and how much. That is the whole job, and the
following are out of it on purpose, because each one turns a finishable tool into
an unfinished platform:

- **It measures change, not quality.** Nothing in it says whether an output is
  good. `has_keys` and `constant_fields` compare against what the *baseline*
  produced, never against an answer you typed in; the judge is asked "did this
  change" and never "is this right". Tools that measure quality exist and are
  better at it; use one before you ship.
- **It observes from outside.** No tracing, no instrumentation, no SDK wrapped
  around your calls. `attribute_to` is the outside-in answer to "my app or the
  model"; it is deliberately weaker than tracing and deliberately free.
- **State is files in your repo and one SQLite file.** No hosted service, no
  accounts, no billing, no other database. Every command has `--json`; anyone
  who wants a dashboard has the data.
- **It speaks plain HTTP** (and, as the one exception that paid for itself, the
  local `claude` CLI). No agent-framework integrations: anything that exposes an
  endpoint is already covered by `type: http`.
- **No leaderboards or model benchmarking.**

**Possible, not planned: shared baselines.** A baseline is already plain text
keyed by a config hash covering prompt, model and embedder revision, which makes
it portable in principle. A public repository of baselines for public providers
("gpt-4o-mini on 2026-09-01 against these twenty probes") would let anyone check
their own account against a community reference with no hosted service, and the
fingerprint signal would become a shared early-warning system -- the one idea here
with a network effect. Not doing it yet: latency and cost are meaningless across
machines and accounts, fingerprints vary by region, a variance pool grown from
strangers' clean runs is a trust problem the anchor caps do nothing to address, and
curating a public set of baselines is a burden that looks a lot like the platform
this project refuses to be. Unlike the list above, this one is a "not yet" rather
than a "never" -- it is just not close to the top of that list.

---

## Design constraints

- Point it at an endpoint with a few prompts and get value in under five minutes.
- Near-zero running cost. Local embeddings by default, judge opt-in and only on
  suspicion.
- No internet dependency except the target endpoint, with one exception stated
  plainly: the default embedding model is a 32MB download, pinned to a fixed
  revision and fetched once. Set `HF_HUB_OFFLINE=1` once it is cached, or set
  `embedder: hashing` to never download anything, at the cost of a weaker signal
  on rewrites that preserve meaning.
- Plain text config, so it lives in git.
- Works with any OpenAI-compatible endpoint, which covers most providers plus
  local Ollama and vLLM, and with anything else over `type: http`.

---

## Licence

MIT.
