"""Command line entry point.

Deliberately thin: parse arguments, call `runner`, render, exit. All the
interesting behaviour lives in modules that can be tested without a subprocess.

`argparse` rather than click or typer, because a monitoring tool people install to
find out their app is broken should not itself have a dependency tree.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from . import __version__
from .alerts import as_json, exit_code_for, send
from .config import Config, load_config
from .models import EXIT_CODES, Level
from .report import render
from .runner import capture_baseline, check
from .store import BaselineStore, History

DEFAULT_CONFIG = "stillsane.yaml"

#: Exit codes in increasing order of severity. The codes themselves are chosen for
#: their meaning to CI (1 = the build should fail) rather than to be ordered, so
#: anything comparing two of them has to go through this.
_SEVERITY = (
    EXIT_CODES[Level.PASS],
    EXIT_CODES[Level.WARN],
    EXIT_CODES[Level.DRIFT],
    EXIT_CODES[Level.ERROR],
)

STARTER_CONFIG = """\
# stillsane -- drift canary for a deployed LLM app.
# Capture a baseline once, then run `stillsane check` on a schedule.

targets:
  - name: prod
    type: openai_compatible
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY   # the variable name, never the key itself
    watch_fingerprint: true

    # To monitor your own deployed app instead of a model API, use:
    #
    # - name: prod
    #   type: http
    #   base_url: https://your-app.example.com
    #   path: /api/chat
    #   body: {message: "{{prompt}}"}
    #   response_path: data.reply

probes:
  - id: example
    prompt: |
      Extract the total and due date from this invoice as JSON:
      Invoice #4471, dated 2026-06-01, due 30 days later. Total: 1240.50 USD.
    baseline_samples: 5   # paid once; this is where the variance band comes from
    check_samples: 3      # paid every run; only needs to locate a median
    checks:
      - valid_json
      - has_keys: [total, due_date]

# alerts:
#   webhook: https://example.com/hook
#   slack_webhook: https://hooks.slack.com/services/...
"""


def _load(path: str) -> Config:
    return load_config(path)


def _store(config: Config, config_path: str) -> tuple[BaselineStore, History]:
    root = Path(config_path).resolve().parent / config.state_dir
    return BaselineStore(root), History(root)


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config)
    if path.exists() and not args.force:
        print(f"{path} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1
    path.write_text(STARTER_CONFIG)
    print(f"Wrote {path}")
    print("\nNext:")
    print("  1. Edit it -- point `base_url` at your endpoint and write a real probe.")
    print("  2. stillsane baseline    # capture what 'normal' looks like")
    print("  3. stillsane check       # compare against it")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    config = _load(args.config)
    store, _ = _store(config, args.config)
    only = set(args.probe) if args.probe else None

    try:
        written = asyncio.run(capture_baseline(config, store, only=only))
    except RuntimeError as exc:
        print(f"stillsane: {exc}", file=sys.stderr)
        return EXIT_CODES[Level.ERROR]

    if not written:
        print("No probes matched.", file=sys.stderr)
        return 1

    for baseline in written:
        n = len(baseline.usable)
        print(
            f"  {baseline.probe_id} @ {baseline.target_name}: "
            f"v{baseline.version}, {n} sample(s)"
            + (f", fingerprint {baseline.fingerprint}" if baseline.fingerprint else "")
        )
    print(f"\nCaptured {len(written)} baseline(s). These will not change until you run this again.")
    return 0


def _run_check(args: argparse.Namespace) -> int:
    config = _load(args.config)
    store, history = _store(config, args.config)
    only = set(args.probe) if args.probe else None

    result = asyncio.run(check(config, store, history, only=only))

    if args.json:
        print(as_json(result))
    else:
        print(render(result, verbose=args.verbose))

    if result.level is not Level.PASS:
        send(result, config.alerts.webhook, config.alerts.slack_webhook)
    return exit_code_for(result, config.alerts.fail_on_warn)


def cmd_check(args: argparse.Namespace) -> int:
    return _run_check(args)


def cmd_watch(args: argparse.Namespace) -> int:
    """Scheduled mode.

    A sleep loop, and honestly so. cron and GitHub Actions do this better -- they
    survive reboots, they log, and they alert when the job itself stops running,
    which a bare process cannot do for itself. This exists for laptops and quick
    trials; the docs point at a scheduler for anything real.
    """
    interval = args.interval
    print(f"Watching every {interval}s. Ctrl-C to stop.", file=sys.stderr)
    worst = 0
    try:
        while True:
            started = time.monotonic()
            code = _run_check(args)
            # Ranked by severity, not by numeric value. The exit codes are not
            # ordered -- DRIFT is 1 and WARN is 2 -- so `max()` on the raw numbers
            # would let a later warning mask an earlier drift in the summary.
            if _SEVERITY.index(code) > _SEVERITY.index(worst):
                worst = code
            if args.once:
                return code
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, interval - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return worst


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stillsane",
        description="Know when your LLM app quietly stops working.",
    )
    parser.add_argument("--version", action="version", version=f"stillsane {__version__}")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG, help=f"config file (default: {DEFAULT_CONFIG})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a starter config")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing config")
    p_init.set_defaults(func=cmd_init)

    p_base = sub.add_parser("baseline", help="capture a new baseline (explicit, never automatic)")
    p_base.add_argument("--probe", action="append", help="limit to this probe id (repeatable)")
    p_base.set_defaults(func=cmd_baseline)

    p_check = sub.add_parser("check", help="run once, compare, exit non-zero on drift")
    p_check.add_argument("--probe", action="append", help="limit to this probe id (repeatable)")
    p_check.add_argument("-v", "--verbose", action="store_true", help="show signals that passed")
    p_check.add_argument("--json", action="store_true", help="machine-readable output")
    p_check.set_defaults(func=cmd_check)

    p_watch = sub.add_parser("watch", help="scheduled mode (prefer cron or CI for anything real)")
    p_watch.add_argument("--probe", action="append", help="limit to this probe id (repeatable)")
    p_watch.add_argument("-v", "--verbose", action="store_true")
    p_watch.add_argument("--json", action="store_true")
    p_watch.add_argument(
        "--interval", type=float, default=3600.0, help="seconds between runs (default: 3600)"
    )
    p_watch.add_argument("--once", action="store_true", help="run a single iteration and exit")
    p_watch.set_defaults(func=cmd_watch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"stillsane: {exc}", file=sys.stderr)
        return EXIT_CODES[Level.ERROR]
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
