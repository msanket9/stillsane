# stillsane

**A drift canary for deployed LLM apps and agents.**

Your LLM app does not crash when it gets worse. It returns 200, latency looks
normal, the error rate is zero, and the output is quietly less correct than it was
last month. You find out when a user complains.

stillsane runs a small set of prompts against your live endpoint on a schedule,
compares each response to a stored baseline, and tells you when behaviour has
moved outside the range that probe normally varies by. It observes from outside,
over plain HTTP. There is nothing to instrument, no SDK to import, no account, and
no hosted service.

> **Status: early, v0.0.9.** Everything described below works. The config format
> may still change before 0.1. See [Status](#status).

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
[`examples/invoice-extract/`](examples/invoice-extract/).

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
  will not help you. That is not false modesty. Pre-ship evaluation is a different
  problem, and tools built for it solve it better than a tool of this scope ever
  will.
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

**What stillsane adds** is that comparison logic: a variance model so the thing
does not cry wolf, a baseline that refuses to update itself, and fingerprint
watching. As far as I can tell nothing in the pre-ship category ships a command
that compares a run against a stored baseline and tells you what moved. That gap
is the entire reason this exists.

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
  signal                        n   |z| p95   |z| max      headroom
  length_chars                 26      1.08      1.51          2.0x
  latency_ms                   26      0.67      0.83          3.6x
  semantic_distance            26      0.51      0.68          4.4x
```

A clean run is one where nothing drifted, so its `z` values are what normal looks
like, and the gap to `warn_k` is the margin before a false alarm. The reading
above is from nine clean runs against a live provider: nothing came within 2x of
the threshold, which says `warn_k: 3` is conservative rather than trigger-happy on
those probes.

It measures **headroom against false alarms only**. Clean runs contain no drift,
so nothing there says whether the thresholds would catch a real regression, and
loosening `k` on that basis would trade a visible problem for an invisible one.
The command says so every time it runs, and refuses to present a
smallest-that-would-not-have-fired value as a recommendation.

The Mann-Whitney p-value reported alongside is
distribution-free and does carry its usual meaning, which is exactly why it is
supporting evidence and never the gate.

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
  system message, checks and model. Change any of them and `check` refuses to
  compare rather than reporting your own edit as provider drift.
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

## Usage

Python 3.10+, five dependencies, no torch. Everything needed to run is installed;
the embedding model itself is fetched once on first use (~32MB) and cached. See
[Design constraints](#design-constraints) if you need to stay fully offline.

Beyond the three commands in the [quickstart](#quickstart) there is `stillsane
watch`, a sleep loop that is honest about being one. cron or CI does this better:
they survive reboots, they log, and they can tell you when the job itself stopped
running, which a bare process cannot do for itself. There is also `stillsane
bands`, below.

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

`--strict` exits 2 when the canary is unhealthy or overdue, for a second cron job
whose only purpose is to notice that the first one stopped. `--json` gives the
same verdict structured.

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
remember signal names to look at your own data. Everything lives in
`.stillsane/history.sqlite`.

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

`--strict` exits 2 when any band will misreport, for a CI job that should fail on
a baseline this shape. `-v` shows every band rather than only the interesting ones.

`--json` writes the same inspection as structured output. Unlike the human report
it always includes every band, sound ones included, since a consumer diffing bands
between runs needs to tell "still sound" from "no longer reported":

```json
{
  "signal": "semantic_distance",
  "finding": "collapsed",
  "suspect": true,
  "unit": "pairs",
  "n": 31,
  "observed_min": 0.0,
  "observed_max": 0.1276024580001831,
  "raw_scale": 0.0,
  "outside": 14,
  "outside_pct": 45.16,
  "band": {"center": 0.0, "scale": 0.006666666666666667,
           "lower": null, "upper": 0.02, "n": 31, "floored": true}
}
```

### Config

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

**Monitoring your own app rather than a model API.** This is the case the tool is
really for: most people are not watching a raw model, they are watching the thing
they shipped, which has its own retrieval, prompt assembly and bugs in front of it.
Use `type: http` and describe the request:

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
      document: "{{prompt}}"           # {{prompt}} and {{system}} are substituted
    response_path: data.reply          # where the text lives in the response
```

**Providers that do not use `Authorization: Bearer`.** Anthropic wants
`x-api-key` with no prefix, Azure wants `api-key`. Both are reachable without
putting a live secret in `headers`:

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

`response_path` takes dotted paths with indexes (`choices.0.message.content`) and
a filter form (`content[type=text].text`). The filter matters on Anthropic: with
thinking enabled `content.0` is the thinking block, not the answer.

Also on a target: `timeout_s`, `retries`, `retry_backoff_s`, `temperature`,
`max_tokens`, and `escalate_fingerprint` to make a changed fingerprint fail rather
than warn.

`samples: 5` also works and sets the baseline count.

**On cost.** Sampling is the whole mechanism, so it is worth being explicit: the
expensive part is the baseline, and you pay it once. Routine checks need only
enough samples to locate a median, because the variance estimate already lives in
the baseline. Embeddings run locally and cost nothing. The LLM judge is opt-in and
only fires when a band has already been crossed, so a normal run spends nothing
beyond the probe calls themselves.

### Checks

| Check | Meaning |
| --- | --- |
| `valid_json` | The whole response parses as JSON. A markdown fence is allowed; surrounding prose is not, because that is what breaks a caller's `json.loads`. |
| `has_keys: [a, b]` | Those keys are present in the JSON, found leniently. Deliberately separate from `valid_json`, so the report can say the data survived even when the envelope broke. |
| `semantic_similarity: auto` | Learn the band. A number instead of `auto` pins a fixed threshold. |
| `max_length: 2000` | Hard cap on response length. |

Several signals are always on and need no configuration: semantic distance, JSON
shape, tool-call shape, length, completion tokens, cost, latency, provider
fingerprint and model id. Signals that do not apply to a probe stay silent, so
tool-call drift says nothing about a probe that never calls a tool.

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
OpenAI-style request body, a bare `{"prompt": ...}`, or either of those wrapped
under `request`, `body` or `payload`. Malformed lines are skipped, because
refusing a 10,000-line log over one line truncated mid-write would make the
feature useless on exactly the files it exists for.

**Checks are emitted commented out.** Guessing that a probe returns JSON and being
wrong would fail your first baseline and teach you the tool is broken. You get the
prompts and a suggestion; you decide what holds.

| Flag | |
| --- | --- |
| `--limit N` | Most probes to emit, most frequent first. Default 20. |
| `--merge-distance D` | How aggressively to cluster. Higher merges more. Default 0.12. |
| `--probes-only` | Emit just the `probes:` block, for pasting into a config you already have. |
| `--embedder hashing` | Cluster without the embedding model, fully offline. |

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
  saw two samples does not get to overrule that. Set `can_downgrade: true` to let
  it soften a drift it considers purely cosmetic, once you trust it.
- **It gets its own endpoint.** Not a flag on a target, because judging with the
  same deployment you are watching means a provider-side change moves both the
  thing being measured and the instrument measuring it.

If the judge is unreachable or answers with something unparseable, the run is
unaffected: the verdict stands and the explanation is simply absent.

### Exit codes

| Code | Meaning |
| :--: | --- |
| `0` | No drift. |
| `1` | Drift. |
| `2` | Warning only. Does not fail a build unless `fail_on_warn: true`. |
| `3` | Error. The endpoint failed, or there is no usable baseline. |

### In CI

Copy
[`examples/invoice-extract/github-actions.yml`](examples/invoice-extract/github-actions.yml)
into `.github/workflows/`. It runs `stillsane check` every morning, caches the
embedding model between runs, and fails the job on drift.

Kept as one file rather than pasted here as a second copy, because two copies of a
workflow drift apart and the one in the README is the one nobody re-tests.

Two things it relies on:

- **Commit `.stillsane/baselines/`.** The workflow needs something to compare
  against. They are plain text and diff like code. Leave
  `.stillsane/history.sqlite` out, since it is a binary that changes every run.
- **A daily schedule is the point.** Provider-side model changes arrive without
  warning; finding out within a day is the entire product.

---

## Status

Working end to end:

- The comparison engine: variance bands, effect sizes, verdict aggregation
- Signals: structural, semantic, JSON shape, tool-call shape, fingerprint,
  tokens, cost, latency
- Variance pooling, with the caps that stop gradual drift widening its own band
- Targets: OpenAI-compatible and arbitrary HTTP
- Versioned baseline store, SQLite history, and the config hash that refuses a
  stale comparison
- `init`, `baseline`, `check`, `watch`, the report renderer, and webhook/Slack alerts
- The optional LLM judge, which only runs on probes that already failed their band
- `init --from-logs`, which clusters your logged prompts into a probe set

The test suite runs with no network, no API key and no model download. It ships
in the sdist, so you can verify the variance model yourself rather than taking
this README's word for it:

```bash
pip install -e ".[dev]" && pytest
```

There is a runnable worked example in
[`examples/invoice-extract/`](examples/invoice-extract/), with a committed baseline
and a mock provider, so you can watch a real regression get caught without an API
key. CI builds the wheel, installs it into an empty environment and runs that
example on every push, which is how a broken install gets caught before a release
rather than after one.


**Expect breakage before 0.1.** The config format is not frozen. If a field is
renamed you will get a validation error naming it, not a silent misread, but a
version pin is wise for now.

---

## Non-goals

Each of these turns a finishable project into an unfinished platform, so they are
out permanently rather than deferred:

no web dashboard · no hosted service, accounts or billing · no tracing or
instrumentation of your app · no SDK to import · no database beyond SQLite · no
agent-framework integrations, it speaks plain HTTP · no leaderboards or model
benchmarking

And the one that matters most: **stillsane is not an eval framework.** It does not
measure whether your app is good. It measures whether it *changed* from a known
baseline. Existing tools measure quality; this measures change.

---

## Design constraints

- Point it at an endpoint with a few prompts and get value in under five minutes.
- Near-zero running cost. Local embeddings by default, judge opt-in and only on
  suspicion.
- No internet dependency except the target endpoint, with one exception stated
  plainly: the default embedding model is a 32MB one-time download. Set
  `embedder: hashing` to stay fully offline, at the cost of a weaker signal on
  rewrites that preserve meaning.
- Plain text config, so it lives in git.
- Works with any OpenAI-compatible endpoint, which covers most providers plus
  local Ollama and vLLM.

---

## Licence

MIT.
