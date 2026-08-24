# Changelog

Notable changes per release. Dates are the PyPI upload date.

The version in `src/stillsane/__init__.py` is the single source of truth, and a
test requires the current version to have an entry here. That is deliberate: the
version bump and the note describing it belong in the same commit as the change,
because three releases were published from a working tree whose version had
already been used by a different build.

Work that has landed but not shipped goes under **Unreleased**. That heading is
the point of the convention rather than decoration: without it, a tree ahead of
the last release looks identical to one that is level with it, which is the exact
ambiguity that produced those three.

## Unreleased

- New target, `type: claude_code`: shells out to the `claude` CLI already
  installed and authenticated on this machine, so a probe draws on a Claude Pro
  or Max subscription instead of needing a separately billed API key. Verified
  end to end against a real install, including the false-positive path
  (`stillsane bands`) and `response_complete`.

  Two things are worth knowing before pointing a probe here, both found by
  testing against a real install rather than assumed. First, `--bare` mode was
  ruled out on purpose: its own `--help` text says OAuth and keychain auth are
  never read there, which would force the very API key this target exists to
  avoid, so it runs in ordinary mode and accepts a larger tool surface instead.
  Second, every tool is denied by default, but denial is not the same as an
  attempt never being made -- three identical adversarial prompts under
  identical deny flags produced three different garbled attempts to invoke one
  anyway. Output that looks like a leaked attempt is detected and marked as an
  error rather than silently compared against a baseline as if it were real
  content, but plain generation is the recommended shape of probe for this
  target regardless. `allowed_tools` opts a probe into agentic mode with an
  explicit allowlist instead of a blanket switch, for something genuinely
  supposed to use tools; that mode has had far less real-world testing.

  Caught mid-implementation: adding `claude_command`/`allowed_tools`
  unconditionally to every target's config hash invalidated every existing
  baseline of every type, not just `claude_code` ones, the moment the schema
  gained the fields. They are now only part of the hash for the type they
  apply to.

  Also caught, on a real Python 3.10 interpreter rather than assumed: the
  timeout handler around the subprocess call caught only `TimeoutError`.
  `asyncio.TimeoutError` and the builtin `TimeoutError` are the same class
  from Python 3.11 onward, but not on 3.10, which this project still
  supports -- there, `asyncio.wait_for`'s real exception slipped past the
  `except` clause entirely and would have aborted the whole run instead of
  being captured as a retryable sample error. Confirmed by reverting the fix
  and watching the unhandled `asyncio.exceptions.TimeoutError` traceback on
  3.10, then restoring it and rerunning both the target's own suite and the
  full suite on the same interpreter.

- New always-on signal, `response_complete`: did the response finish on its own,
  or get cut off by the token limit. Reads `finish_reason` (OpenAI-shaped
  targets) or `stop_reason` (Anthropic, and any `http` target that exposes it),
  and flags the token-limit case specifically rather than any change in
  vocabulary. Found by hand in a real calibration run -- an essay probe
  truncated on 7 of 8 samples for weeks with nothing in the tool saying so,
  because a truncated response can still be longer than the baseline and stay
  close to it semantically right up to where it stops. `strict_when_perfect`,
  same as `valid_json`: if the baseline never truncated, any truncation on a
  check is drift, not a statistical question.

  Existing baselines captured before this shipped have no `finish_reason` on
  their stored samples, so the signal correctly stays silent on them rather than
  guessing -- recapture (`stillsane baseline`) to start watching for truncation
  on a probe that already has a baseline.

## 0.0.10 - 2026-08-21

- This changelog, and tests requiring the current version to have an entry, the
  entries to be unique and newest-first, and the file to be linked from the README.
- `stillsane check --json` (and the webhook/Slack payloads) now carry `retries`
  per probe. The text report and `status` already showed a recovered transport
  failure; the JSON payload, the one a CI pipeline actually parses, silently did
  not.
- Fixed `stillsane calibrate` pooling `z` values across probes that happen to
  share a signal name. `extract_invoice` and `summarise_incident` both report
  `length_chars`; in real data the first sat at z=0.000 across thirteen clean
  runs while the second reached 1.51, and the pooled report showed one row
  averaging them together with no way to tell which probe it meant. Output is
  now grouped per probe, and the JSON payload carries `probe`/`target` on every
  row and every false-alarm entry.
- `stillsane calibrate --probe` scopes the report to one probe, matching `check`
  and `bands`. Distinguishes a typo'd probe id from a real probe with no clean
  runs recorded yet, rather than giving both the same error.
- Fixed `stillsane check --probe <typo>` (and `watch`, which shares the same
  path) silently matching nothing and exiting 0 with "No probes ran" -- a
  passing exit code for an invocation that checked nothing, on a config that
  genuinely defines the probe you meant to type. `baseline` and `bands` already
  caught this; `check` did not. Now exits 3 (ERROR) and names the unmatched id.

## 0.0.9 - 2026-08-12

- `stillsane calibrate`: reads the `z` values recorded by past clean runs and
  reports how close each signal came to firing. Answers the question `warn_k` and
  `drift_k` have been an open guess about since the first release, using your own
  runs rather than constructed scenarios. Reports headroom, flags thresholds that
  already fire on clean runs, and refuses to present the tightest-that-would-not-
  have-fired value as a recommendation.
- Documented output is now tested. A `network`-marked test runs the worked example
  with the real embedder and diffs the result against both READMEs, after an
  estimator change silently invalidated the headline block twice.

## 0.0.8 - 2026-08-10

- Transport failures retry. Timeouts, dropped connections, 429s and 5xx get one
  more attempt by default; 4xx and malformed bodies do not, because they come back
  identical. A verdict is never retried, since re-running because the answer was
  DRIFT defines drift out of existence. `Sample.attempts` records the recovery, and
  `check` and `status` both report it rather than smoothing over a flaky endpoint.
- The capture-time list of defaulted bands now covers pointwise signals and honours
  `rel_floor`. It previously read only pooled distances, so a floored `length_chars`
  could never be named however wrong it was.
- Doc-consistency tests for the CLI and config surfaces: every documented command,
  flag and config key must exist, and every command and option must be documented.
- Documented `type: http` targets and providers that do not use
  `Authorization: Bearer`. Both existed and neither appeared in the README, so the
  target the tool is really for had no worked configuration.
- History gains a `retries` column, with a migration for databases written by an
  earlier version.

## 0.0.7 - 2026-08-07

- `stillsane bands` estimates how often each band would report drift on a clean
  run, by resampling the baseline and taking a check-sized median. Replaces a count
  of individual values outside the band, which asked a different and more alarming
  question than the one that matters: a probe with 15% of its pairs outside its band
  had a 0% false alarm rate, because a check compares medians.

## 0.0.6 - 2026-08-06

- `stillsane bands`: offline inspection of every learned band, naming the ones that
  will misreport. Reads only stored data, so it costs nothing and needs no key.
- `stillsane status`: whether the canary itself is alive. Distinguishes transport
  errors from drift, and with `--expect-every` reports a schedule that has stopped
  firing, which otherwise looks identical to having nothing to report.
- Fixed a band collapse on tie-heavy data. A median absolute deviation reports zero
  dispersion when more than half the samples are identical, which a low-temperature
  model does constantly, dropping the band onto its floor. The scale now falls back
  to an interquartile range in that case only.

## 0.0.5 - 2026-08-04

Two of these landed while the version file still read `0.0.4`, after `0.0.4` had
already been published. They are listed under the release that actually carried
them, which is the distinction the Unreleased heading above now makes explicit.

- Configurable auth header and prefix (`api_key_header`, `api_key_prefix`), so
  Anthropic's `x-api-key` and Azure's `api-key` are reachable without putting a live
  secret in `headers`.
- `response_path` accepts a filter form, `content[type=text].text`, needed on
  Anthropic where `content.0` is the thinking block rather than the answer.
- Floored bands are marked in reports, so a defaulted number is not presented as a
  measured one.
- `stillsane history`, and a fix for ordering ties between runs recorded in the same
  second.

## 0.0.4 - 2026-08-01

- Optional LLM judge, which runs only on probes that already crossed a band and so
  costs nothing on a clean run.

## 0.0.3 - 2026-07-31

- model2vec became a core dependency rather than an optional extra. 0.0.2 shipped
  importable only with the extra installed: the tests passed from a source checkout
  where every dependency was present, and nothing exercised a clean install. The
  `package` CI job exists because of this.
- The embedder is part of the config hash: distances only mean anything on the
  scale that produced them, so switching embedder now forces a recapture instead of
  silently comparing incomparable numbers.
- Worked example with a bundled mock provider, and CI.
- Pseudo-signal errors surface in reports. A fingerprint-only alert no longer
  prints a before/after block, which invited hunting for a difference that is not
  the point. `watch` ranks exit codes by severity.

## 0.0.2 - 2026-07-30

- Targets, baseline store, CLI, reports and alerts.
- Shipped broken: see 0.0.3.

## 0.0.1 - 2026-07-29

- First release. Comparison engine, signal suite, variance bands.
