#!/usr/bin/env python3
"""A tiny OpenAI-compatible endpoint that can be told to degrade on command.

This exists so the example runs in thirty seconds with no API key and no spend,
and so the degradation it demonstrates is the one that actually happens in
production rather than something contrived.

    python mock_provider.py                 # behaves itself
    python mock_provider.py --mode drifted  # the regression
    python mock_provider.py --mode newfp    # same output, different backend build

Standard library only -- no dependencies beyond Python itself.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

#: Healthy output. Three phrasings of the same answer, because a real endpoint at
#: low temperature still varies a little -- in whitespace, number formatting and
#: key order. A drift tool has to tolerate exactly this and nothing more.
STABLE = [
    '{"total": 1240.50, "due_date": "2026-07-01"}',
    '{"total": 1240.5, "due_date": "2026-07-01"}',
    '{"due_date": "2026-07-01", "total": 1240.50}',
]

#: The regression. Note what has *not* changed: the numbers are still correct and
#: still present. The model simply started being helpful about it, and every caller
#: doing `json.loads(response)` is now throwing. Nothing errors, nothing is slower,
#: and no traditional monitor notices.
DRIFTED = [
    'Here is the extracted information:\n'
    '{"total": 1240.50, "due_date": "2026-07-01"}\n'
    'Let me know if you need anything else!',
    'Sure! I found the following:\n'
    '{"total": 1240.5, "due_date": "2026-07-01"}\n'
    'Happy to help with more invoices.',
    'Of course. The details are:\n'
    '{"due_date": "2026-07-01", "total": 1240.50}\n'
    'Anything else I can do for you?',
]

FINGERPRINTS = {"stable": "fp_a4f2b1", "drifted": "fp_a4f2b1", "newfp": "fp_9c3e88"}


class Handler(BaseHTTPRequestHandler):
    mode = "stable"
    #: Cycled rather than sampled at random, so the example reproduces the exact
    #: numbers in its README and CI cannot go flaky. A real endpoint is of course
    #: not deterministic -- that is the entire problem stillsane exists to handle --
    #: but a *demonstration* of it should be.
    #:
    #: Not named `responses`: BaseHTTPRequestHandler already has an attribute by
    #: that name holding a dict of status codes, and `send_response` does
    #: `if code in self.responses`. Shadowing it with an infinite iterator makes
    #: that membership test scan forever, so the server accepts connections and
    #: then hangs with no error at all.
    reply_cycle = itertools.cycle(STABLE)

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0) or 0))
        content = next(Handler.reply_cycle)

        body = {
            "model": "mock-model-v1",
            "system_fingerprint": FINGERPRINTS.get(self.mode, "fp_a4f2b1"),
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": max(1, len(content) // 4)},
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        """Silence the per-request logging; it drowns out stillsane's own output."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument(
        "--mode",
        choices=("stable", "drifted", "newfp"),
        default="stable",
        help="stable: healthy. drifted: JSON wrapped in prose. newfp: same output, new backend build.",
    )
    args = parser.parse_args()

    Handler.mode = args.mode
    Handler.reply_cycle = itertools.cycle(DRIFTED if args.mode == "drifted" else STABLE)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock provider on http://127.0.0.1:{args.port}/v1  (mode: {args.mode})")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
