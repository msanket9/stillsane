"""Core data types.

Nothing here knows about HTTP, YAML, or the filesystem. A `Sample` is the single
currency of the whole system: targets produce them, signals consume them, and the
comparison engine never sees anything else. That boundary is what lets the hard
part -- the variance maths in `stillsane.compare` -- be tested against fixtures
with no network and no API spend.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Level(str, Enum):
    """Verdict severity, ordered. `worst()` relies on the declaration order."""

    PASS = "pass"
    WARN = "warn"
    DRIFT = "drift"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return _LEVEL_RANK[self]

    @classmethod
    def worst(cls, levels: list[Level]) -> Level:
        return max(levels, key=lambda lvl: lvl.rank, default=cls.PASS)


_LEVEL_RANK = {Level.PASS: 0, Level.WARN: 1, Level.DRIFT: 2, Level.ERROR: 3}

# Exit codes, per the CLI contract. WARN is separated from DRIFT so CI can choose
# whether a warning breaks the build.
EXIT_CODES = {Level.PASS: 0, Level.DRIFT: 1, Level.WARN: 2, Level.ERROR: 3}


class SignalKind(str, Enum):
    """How a signal is compared, which decides which maths applies to it."""

    #: Needs two samples. Yields a distance. Compared as within- vs cross-distribution.
    PAIRWISE = "pairwise"
    #: One number per sample. Compared as value distributions.
    POINTWISE = "pointwise"
    #: One string per sample. No band -- any change is an event.
    CATEGORICAL = "categorical"


class Direction(str, Enum):
    """Which way a pointwise signal has to move before anyone should care."""

    UP_IS_BAD = "up"  # latency, cost, distance
    DOWN_IS_BAD = "down"  # structural check pass-rate
    BOTH = "both"  # length, token count


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation, reduced to the parts that matter for drift.

    Argument *values* are deliberately dropped from the comparison surface: they
    vary legitimately between runs. Argument *keys* are the shape, and a change of
    shape is the thing worth paging someone about.
    """

    name: str
    arg_keys: tuple[str, ...] = ()

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> ToolCall:
        fn = raw.get("function") or {}
        name = fn.get("name") or raw.get("name") or "<unnamed>"
        arg_keys: tuple[str, ...] = ()
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                arg_keys = tuple(sorted(parsed))
        elif isinstance(args, dict):
            arg_keys = tuple(sorted(args))
        return cls(name=name, arg_keys=arg_keys)

    def signature(self) -> str:
        return f"{self.name}({','.join(self.arg_keys)})"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Sample:
    """One invocation of one probe against one target."""

    probe_id: str
    target_name: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    fingerprint: str | None = None
    model_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    http_status: int | None = None
    #: Raw `finish_reason`/`stop_reason` as the provider names it, e.g. "stop",
    #: "length", "max_tokens", "tool_calls". Kept as-is rather than normalised, so a
    #: signal built later can interpret vocabulary this project has not seen yet.
    finish_reason: str | None = None
    error: str | None = None
    #: How many times the endpoint was called to produce this sample. Above one
    #: means a transient transport failure was retried. Recorded rather than
    #: swallowed: a run that only succeeded on the third try is still evidence that
    #: the environment is unwell, and hiding it turns a flaky setup into a silent one.
    attempts: int = 1
    ts: datetime = field(default_factory=_utcnow)
    #: Full decoded response body, kept for the report and for signals we have not
    #: thought of yet. Never used for comparison directly.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        d["tool_calls"] = [{"name": tc.name, "arg_keys": list(tc.arg_keys)} for tc in self.tool_calls]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Sample:
        d = dict(d)
        ts = d.get("ts")
        d["ts"] = datetime.fromisoformat(ts) if isinstance(ts, str) else _utcnow()
        d["tool_calls"] = [
            ToolCall(name=tc["name"], arg_keys=tuple(tc.get("arg_keys", ())))
            for tc in d.get("tool_calls", [])
        ]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Band:
    """The learned normal range for one signal on one probe.

    Built from a median and a MAD-derived scale rather than mean/stdev: at the
    sample counts anyone will actually pay for (N=5), stdev is dominated by
    whichever single sample happened to be weird, and MAD is not.
    """

    center: float
    scale: float
    lower: float | None
    upper: float | None
    n: int
    floored: bool = False

    def contains(self, value: float) -> bool:
        if self.lower is not None and value < self.lower:
            return False
        return not (self.upper is not None and value > self.upper)

    def describe(self) -> str:
        if self.upper is not None and self.lower is not None:
            return f"{self.lower:.4g}..{self.upper:.4g}"
        if self.upper is not None:
            return f"<={self.upper:.4g}"
        if self.lower is not None:
            return f">={self.lower:.4g}"
        return "unbounded"


@dataclass
class SignalVerdict:
    """The outcome for one signal on one probe, and the evidence behind it."""

    signal: str
    #: None for pseudo-signals that are not really measurements -- transport
    #: errors and baseline problems reuse this type so the report has one shape.
    kind: SignalKind | None
    level: Level
    detail: str
    observed: float | None = None
    baseline: float | None = None
    band: Band | None = None
    #: Effect size in units of the probe's own normal variance. The headline number.
    z: float | None = None
    #: Mann-Whitney p-value where computed. Supporting evidence, never the gate.
    p_value: float | None = None
    observed_label: str | None = None
    baseline_label: str | None = None


@dataclass
class ProbeVerdict:
    probe_id: str
    target_name: str
    level: Level
    signals: list[SignalVerdict] = field(default_factory=list)
    baseline_version: int | None = None
    baseline_created: str | None = None
    #: Populated only when the judge ran, i.e. only when a band was already crossed.
    judge_note: str | None = None
    #: Representative outputs for the report's before/after block.
    baseline_excerpt: str | None = None
    observed_excerpt: str | None = None
    #: Extra calls this probe's samples needed because a transport failure was
    #: retried. Carried up to the report on purpose: a retry that recovers a run is
    #: still evidence the environment is unwell, and a retry nobody can see turns a
    #: flaky setup into a silent one, which is the failure the retry was meant to
    #: make survivable rather than invisible.
    retries: int = 0

    @property
    def moved(self) -> list[SignalVerdict]:
        return [s for s in self.signals if s.level is not Level.PASS]


@dataclass
class RunResult:
    probes: list[ProbeVerdict] = field(default_factory=list)
    started: datetime = field(default_factory=_utcnow)
    finished: datetime | None = None

    @property
    def level(self) -> Level:
        return Level.worst([p.level for p in self.probes])

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.level]
