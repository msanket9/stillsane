# Worked example: invoice extraction

A probe that asks a model to pull a total and a due date out of an invoice as
JSON, and a baseline captured against a healthy endpoint.

It ships with a mock provider so you can watch the whole thing work, including a
real regression being caught, in about thirty seconds with no API key and no
spend. The baseline in `.stillsane/baselines/` is committed, which is how you
would run this in CI too.

## Try it

Two terminals. In the first, start a healthy endpoint:

```bash
python mock_provider.py
```

In the second, check it against the committed baseline:

```bash
stillsane check
```

```
PASS   extract_invoice @ prod

------------------------------------------------------------
1 pass   ->  PASS
```

Now break it. Stop the mock provider and restart it in its degraded mode:

```bash
python mock_provider.py --mode drifted
```

The model still returns the right numbers. It has simply started explaining
itself, the way a model quietly does after a provider-side update. Nothing errors,
nothing is slower, and every caller doing `json.loads(response)` is now throwing.

```bash
stillsane check
```

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

Exit code 1, so this fails a build.

Note what the semantic band did. This mock cycles three fixed phrasings that
differ in whitespace, number formatting and key order, and the baseline learned
that a distance up to `0.056` is normal for this probe. The prose-wrapped version
sits at `0.133`, nearly nine times outside it. Nobody picked that threshold; it
came from the probe's own behaviour.

The `(floor)` markers on the other two are the tool being honest about a different
situation. Lengths and token counts barely move across those three phrasings, so
there was too little spread to measure and those bands fell back to a built-in
floor rather than presenting a defaulted number as a measured one. `valid_json`
carries no marker because its band is not a measurement at all: it passed every
baseline sample, so any failure now counts.

`stillsane bands` reports all of this from the stored baseline, without a network
call or an API key, and names any band that will misreport.

Note also what `has_keys` did: nothing. It is not in the list of signals that
moved, because `total` and `due_date` are both still there. The data survived and
the envelope broke, and the report can tell you which.

## The third case

```bash
python mock_provider.py --mode newfp
```

Identical output, but the provider reports a different `system_fingerprint`. The
backend model changed underneath a version string that did not.

```bash
stillsane check
```

```
WARN   extract_invoice @ prod
  fingerprint             fp_9c3e88
  8 other signal(s) unchanged

------------------------------------------------------------
1 warn   ->  WARN
```

Exit code 2. There is no before/after block here because nothing the model wrote
changed, and printing one would only invite you to hunt for a difference that is
not the point. This is information, not a fault: the model moved, the output has not
(yet). It is the one form of drift you cannot otherwise see, and the reason to
look at your probes before your users do. Set `escalate_fingerprint: true` on the
target to make it fail instead.

## Pointing it at something real

Replace the target in `stillsane.yaml` with your own endpoint and recapture:

```bash
stillsane baseline
```

The rest of the file needs no changes. `stillsane.yaml` has commented examples for
both an OpenAI-compatible API and a plain HTTP app of your own.

Two things worth knowing when you do:

- **Recapturing is explicit and always makes a new version.** `v1` is kept. A
  monitor that silently re-baselines has defined drift out of existence.
- **Editing the prompt invalidates the baseline.** `check` will refuse to compare
  and tell you to recapture, rather than reporting your own edit as provider drift.

## Running it on a schedule

Copy [`github-actions.yml`](github-actions.yml) into `.github/workflows/` in your
own repo. It runs `stillsane check` every morning and fails the job on drift.
Commit your `.stillsane/baselines/` directory so the workflow has something to
compare against. They are plain text and diff like code.
