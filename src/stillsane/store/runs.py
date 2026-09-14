"""What a `check` run actually saw, kept a while.

`history` records `z`, `observed`, `baseline` -- numbers, not what the model
wrote. Investigating an alert means finding out what the drifted text actually
was, and today that only ever existed in whatever log captured that run's
stdout -- for a scheduled job, usually a CI log from hours ago. This module
gives every `check` run a small, disk-capped record of its own current-run
samples, keyed by the same `run_id` `History` already assigns, so
`stillsane history --run <id>` can show what the model actually wrote without
anyone going log-hunting.

Baseline samples are not duplicated here -- they already live forever in
`.stillsane/baselines/<target>__<probe>/vN/samples.jsonl`, which is the file
the README already tells people to commit. This is the "now" side only.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ..models import Sample
from ._atomic import atomic_write

#: Oldest runs are pruned past this count. A `watch` loop firing every few
#: minutes would otherwise grow this directory without bound, and 50 runs is
#: enough for "what happened recently" without needing its own retention
#: policy to configure.
DEFAULT_KEEP = 50


class RunSampleStore:
    def __init__(self, root: Path | str, keep: int = DEFAULT_KEEP) -> None:
        self.root = Path(root) / "runs"
        self.keep = keep

    def append(self, run_id: str, samples: Sequence[Sample], store_raw: bool) -> None:
        """Add one probe's current-run samples to this run's file.

        Called once per probe within a `check` run, all under the same
        `run_id`, so a run with several probes ends up with one file holding
        all of them -- each `Sample` already carries its own
        `probe_id`/`target_name`, which is enough to group by later.

        `store_raw` mirrors the same switch `capture_baseline` honours
        (`TargetConfig.store_raw`, off by default): the full decoded response
        body carries the same tenant-data risk here that it does in a
        committed baseline, and nothing reads it back either way, so it is
        stripped unless the target has explicitly opted in.

        Rewrites the whole file through `atomic_write` rather than a plain
        append: a crash mid-`write` truncates whatever was mid-flight, and an
        `open("a")` append has no way to undo a partial last line -- `load`
        would then fail to parse that one line and lose every sample
        recorded before it in the same file, the exact failure `atomic_write`
        exists to rule out for `baseline.py`'s files.
        """
        if not samples:
            return
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        samples_file = path / "samples.jsonl"
        existing = samples_file.read_text() if samples_file.exists() else ""
        rows = [
            json.dumps((s if store_raw else replace(s, raw={})).to_dict())
            for s in samples
        ]
        atomic_write(samples_file, existing + "".join(row + "\n" for row in rows))

    def load(self, run_id: str) -> list[Sample] | None:
        """Every sample recorded for this run, across every probe in it.

        `None` distinguishes "this run kept nothing" -- pruned past `keep`,
        or predates this feature -- from "this run had no runnable probes".
        """
        path = self.root / run_id / "samples.jsonl"
        if not path.exists():
            return None
        return [
            Sample.from_dict(json.loads(line))
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    def prune(self) -> None:
        """Keep only the most recently written `keep` runs.

        Ordered by directory mtime rather than by parsing `run_id` (an opaque
        hex string with no ordering of its own) -- the store is the only
        writer, so "when was this last touched" and "how recently was this
        run's data written" are the same question.
        """
        if not self.root.is_dir():
            return
        dirs = sorted(
            (d for d in self.root.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for stale in dirs[self.keep :]:
            shutil.rmtree(stale, ignore_errors=True)
