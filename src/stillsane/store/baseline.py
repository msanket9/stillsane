"""Versioned baselines on disk.

Two rules shape this whole module:

**Baselines are never updated implicitly.** Only `stillsane baseline` writes one,
and it always writes a *new* version rather than overwriting. A monitor that
silently re-baselines has defined drift out of existence -- every run compares
against the last run, so nothing ever looks different and the tool reports success
forever while quietly measuring nothing.

**A baseline is only valid for the config that produced it.** The stored
`config_hash` covers the prompt, system message, checks and model. Edit any of
them and `check` refuses to compare rather than reporting your own edit as
provider drift.

Layout, chosen so it survives being read by a human at 3am with `cat`:

    .stillsane/baselines/<target>__<probe>/v3/
        meta.json       # created_at, config_hash, model_id, fingerprint, n
        samples.jsonl   # frozen reference outputs, one per line
        variance.json   # pooled distances + day-one anchors, per signal
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..compare.pooling import Anchor
from ..models import Sample

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(value: str) -> str:
    """Filesystem-safe directory component."""
    cleaned = _UNSAFE.sub("-", value).strip("-")
    return cleaned or "unnamed"


@dataclass
class Baseline:
    target_name: str
    probe_id: str
    version: int
    created: str
    config_hash: str
    samples: list[Sample] = field(default_factory=list)
    #: Pooled within-run distances per pairwise signal. Grows on clean runs.
    pooled: dict[str, list[float]] = field(default_factory=dict)
    #: What each signal looked like on day one. Pooling is capped against this.
    anchors: dict[str, Anchor] = field(default_factory=dict)
    model_id: str | None = None
    fingerprint: str | None = None
    #: Signals whose band came out floored at capture time, i.e. defaulted rather
    #: than measured. Not persisted -- it is a property of the numbers, recomputed
    #: whenever they are, and only interesting at the moment of capture.
    floored: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]


class BaselineStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root) / "baselines"

    def _dir(self, target_name: str, probe_id: str) -> Path:
        return self.root / f"{slug(target_name)}__{slug(probe_id)}"

    def versions(self, target_name: str, probe_id: str) -> list[int]:
        base = self._dir(target_name, probe_id)
        if not base.is_dir():
            return []
        found = []
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
                found.append(int(child.name[1:]))
        return sorted(found)

    def latest_version(self, target_name: str, probe_id: str) -> int | None:
        versions = self.versions(target_name, probe_id)
        return versions[-1] if versions else None

    def load(self, target_name: str, probe_id: str, version: int | None = None) -> Baseline | None:
        version = version or self.latest_version(target_name, probe_id)
        if version is None:
            return None
        path = self._dir(target_name, probe_id) / f"v{version}"
        meta_file = path / "meta.json"
        if not meta_file.exists():
            return None

        meta = json.loads(meta_file.read_text())
        samples = []
        samples_file = path / "samples.jsonl"
        if samples_file.exists():
            for line in samples_file.read_text().splitlines():
                if line.strip():
                    samples.append(Sample.from_dict(json.loads(line)))

        pooled: dict[str, list[float]] = {}
        anchors: dict[str, Anchor] = {}
        variance_file = path / "variance.json"
        if variance_file.exists():
            data = json.loads(variance_file.read_text())
            pooled = {k: list(v) for k, v in (data.get("pooled") or {}).items()}
            anchors = {
                k: Anchor(center=v["center"], scale=v["scale"])
                for k, v in (data.get("anchors") or {}).items()
            }

        return Baseline(
            target_name=target_name,
            probe_id=probe_id,
            version=version,
            created=meta.get("created", ""),
            config_hash=meta.get("config_hash", ""),
            samples=samples,
            pooled=pooled,
            anchors=anchors,
            model_id=meta.get("model_id"),
            fingerprint=meta.get("fingerprint"),
        )

    def save(
        self,
        target_name: str,
        probe_id: str,
        samples: list[Sample],
        config_hash: str,
        pooled: dict[str, list[float]] | None = None,
        anchors: dict[str, Anchor] | None = None,
    ) -> Baseline:
        """Write a new baseline version. Never overwrites an existing one."""
        previous = self.latest_version(target_name, probe_id) or 0
        version = previous + 1
        path = self._dir(target_name, probe_id) / f"v{version}"
        path.mkdir(parents=True, exist_ok=True)

        usable = [s for s in samples if s.ok]
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        baseline = Baseline(
            target_name=target_name,
            probe_id=probe_id,
            version=version,
            created=created,
            config_hash=config_hash,
            samples=samples,
            pooled=pooled or {},
            anchors=anchors or {},
            model_id=next((s.model_id for s in usable if s.model_id), None),
            fingerprint=next((s.fingerprint for s in usable if s.fingerprint), None),
        )
        self._write(path, baseline)
        return baseline

    def update_variance(
        self, baseline: Baseline, pooled: dict[str, list[float]], anchors: dict[str, Anchor]
    ) -> None:
        """Persist a grown variance pool in place.

        This is the only write that happens outside an explicit `baseline` command,
        and it deliberately touches `variance.json` alone -- the reference outputs
        in `samples.jsonl` stay exactly as captured.
        """
        baseline.pooled = pooled
        baseline.anchors = anchors
        path = self._dir(baseline.target_name, baseline.probe_id) / f"v{baseline.version}"
        path.mkdir(parents=True, exist_ok=True)
        self._write_variance(path, baseline)

    def _write(self, path: Path, baseline: Baseline) -> None:
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "created": baseline.created,
                    "config_hash": baseline.config_hash,
                    "target": baseline.target_name,
                    "probe": baseline.probe_id,
                    "n": len(baseline.usable),
                    "model_id": baseline.model_id,
                    "fingerprint": baseline.fingerprint,
                },
                indent=2,
            )
            + "\n"
        )
        with (path / "samples.jsonl").open("w") as fh:
            for sample in baseline.samples:
                fh.write(json.dumps(sample.to_dict()) + "\n")
        self._write_variance(path, baseline)

    def _write_variance(self, path: Path, baseline: Baseline) -> None:
        (path / "variance.json").write_text(
            json.dumps(
                {
                    "pooled": baseline.pooled,
                    "anchors": {
                        k: {"center": a.center, "scale": a.scale}
                        for k, a in baseline.anchors.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )


class BaselineMismatch(Exception):
    """The stored baseline does not match the current config.

    Raised rather than papered over: comparing against a baseline captured for a
    different prompt produces a confident, entirely wrong drift report, which is
    worse than no report at all.
    """

    def __init__(self, probe_id: str, target_name: str, stored: str, current: str) -> None:
        super().__init__(
            f"Probe {probe_id!r} on target {target_name!r} has changed since its "
            f"baseline was captured (stored {stored[:8]}, now {current[:8]}).\n"
            "The prompt, system message, checks or model differ, so the stored "
            "outputs are not a valid comparison.\n"
            "Run `stillsane baseline` to capture a new one."
        )
        self.probe_id = probe_id
        self.target_name = target_name
