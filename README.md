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

> **Status: early — v0.0.3.** `init`, `baseline`, `check` and `watch` all work.
> The LLM judge and probe auto-generation are not built yet, and the config format
> may still change before 0.1. See [Status](#status).

---

## What it looks like

A probe that extracts a total and a due date as JSON. The model was updated, and
it started explaining itself. The numbers are still correct and still present —
but every caller doing `json.loads(response)` is now throwing.

```
DRIFT  extract_invoice @ prod
  semantic_distance           0.133  band <=0.03745       z=+17.3
  length_chars                  106  band 36..52          z=+23.2
  completion_tokens              26  band 7..15           z=+11.2
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

That `band <=0.03745` was not a number anyone chose. The baseline watched this
probe vary in whitespace, number formatting and key order, and learned how much
that is worth. The prose-wrapped version sits seventeen times further out. A
chattier probe would have learned a band twenty times wider and stayed silent
through the same rewording.

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

Put `stillsane check` on a schedule in CI and you are done — see
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
gets believed. The fix there was synthetic monitoring — walk the whole pipeline on
a schedule, validate what comes back, alert before a human notices. stillsane is
that, pointed at an LLM.

---

## Should you use this?

Probably not, if you already have something. Be honest with yourself here:

- **You want to know whether a prompt is good before you ship it.** Use a
  pre-ship eval framework. There are several good open-source ones, and stillsane
  will not help you — this is not false modesty, it is a different problem, and
  tools built for it solve it better than a tool of this scope ever will.
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
document a drift workflow — save a baseline, re-run on a schedule, compare — and if
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
threshold either misses real drift on the first or fires constantly on the second
— and a tool that fires constantly gets uninstalled inside a week.

So stillsane learns the band per probe, from the probe's own behaviour:

- At baseline it captures N samples and measures the distances **among them**.
  That distribution is the probe's intrinsic variance.
- At check time it captures M samples and measures the distance from each baseline
  sample to each new one.
- With no drift, those two sets of distances are drawn from the same distribution
  — both are "how far apart are two independent draws". With drift, the second set
  shifts up.

From this repo's own test suite, same code and no configuration:

```
stable JSON probe    band <= 0.178     # near-deterministic, tight band
chatty prose probe   band <= 0.612     # 3.4x wider, and correctly so
```

Both still catch a real regression. The stable probe flags prose creeping into its
JSON at z=9.0; the chatty probe flags a genuine topic change while staying silent
through ordinary rewording.

The headline number is **z**: how far behaviour moved in units of that probe's own
normal variation. "Moved 6.2x further than this probe usually varies" is a
sentence you can act on. A p-value is not.

**On calling it `z`.** It is computed as `(observed − median) / (1.4826 × MAD)`,
which is a *robust analogue* of a standard score, not a standard score. The
1.4826 makes a MAD comparable to a standard deviation for normally distributed
data, and a distribution of distances is emphatically not normal — it is bounded
below at zero and right-skewed. So `z` here assumes no distribution at all: it is
a scale-free measure of how far outside normal something sits, and **it does not
convert to a probability**. `z=6` is not a one-in-a-billion event. The thresholds
(`warn_k: 3`, `drift_k: 6`) are calibrated against real probe behaviour, not
derived from Gaussian tails, and they are config knobs precisely because the right
values are an empirical question. The Mann-Whitney p-value reported alongside is
distribution-free and does carry its usual meaning — which is exactly why it is
supporting evidence and never the gate.

Related decisions, since they are the ones that determine whether this is usable:

- **Robust statistics throughout.** Median and MAD, not mean and standard
  deviation. At the sample counts anyone will actually pay for, one weird sample
  dominates a standard deviation and barely moves a MAD.
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

---

## Usage

Python 3.10+, five dependencies, no torch. Everything needed to run is installed;
the embedding model itself is fetched once on first use (~32MB) and cached — see
[Design constraints](#design-constraints) if you need to stay fully offline.

Beyond the three commands in the [quickstart](#quickstart) there is `stillsane
watch`, a sleep loop that is honest about being one. cron or CI does this better:
they survive reboots, they log, and they can tell you when the job itself stopped
running, which a bare process cannot do for itself.

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
fingerprint and model id. Signals that do not apply to a probe stay silent —
tool-call drift says nothing about a probe that never calls a tool.

### Exit codes

| Code | Meaning |
| :--: | --- |
| `0` | No drift. |
| `1` | Drift. |
| `2` | Warning only. Does not fail a build unless `fail_on_warn: true`. |
| `3` | Error — the endpoint failed, or there is no usable baseline. |

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
  `.stillsane/history.sqlite` out — it is a binary that changes every run.
- **A daily schedule is the point.** Provider-side model changes arrive without
  warning; finding out within a day is the entire product.

---

## Status

Working end to end:

- The comparison engine — variance bands, effect sizes, verdict aggregation
- Signals — structural, semantic, JSON shape, tool-call shape, fingerprint,
  tokens, cost, latency
- Variance pooling, with the caps that stop gradual drift widening its own band
- Targets — OpenAI-compatible and arbitrary HTTP
- Versioned baseline store, SQLite history, and the config hash that refuses a
  stale comparison
- `init`, `baseline`, `check`, `watch`, the report renderer, and webhook/Slack alerts

The test suite runs with no network, no API key and no model download — it ships
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

Not built yet: the LLM judge (the report has a slot for its one-line verdict, but
nothing calls it) and probe auto-generation from logs.

**Expect breakage before 0.1.** The config format is not frozen. If a field is
renamed you will get a validation error naming it, not a silent misread — but a
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
- No internet dependency except the target endpoint — with one exception, noted
  honestly: the default embedding model is a 32MB one-time download. Set
  `embedder: hashing` to stay fully offline, at the cost of a weaker signal on
  rewrites that preserve meaning.
- Plain text config, so it lives in git.
- Works with any OpenAI-compatible endpoint, which covers most providers plus
  local Ollama and vLLM.

---

## Licence

MIT.
