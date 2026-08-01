"""Config loading and, more importantly, config *hashing*.

The hash is what stops the tool lying to you. A baseline is only meaningful for
the exact prompt and target it was captured against; if you edit a probe's prompt
and stillsane silently compares the new output to the old baseline, it will report
provider drift for a change you made yourself. So the hash covers everything that
could legitimately change the output, and `check` refuses to compare across a
mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .compare.variance import BandConfig


class TargetConfig(BaseModel):
    """A live endpoint to probe. Speaks plain HTTP; nothing is instrumented."""

    name: str
    type: Literal["openai_compatible", "http"] = "openai_compatible"
    base_url: str
    model: str | None = None
    #: Name of the environment variable holding the key. Never the key itself --
    #: this file is meant to live in git.
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 60.0
    temperature: float | None = None
    max_tokens: int | None = None

    #: Watch `system_fingerprint` and friends for the backend model changing
    #: underneath a stable version string.
    watch_fingerprint: bool = True
    #: Whether a changed fingerprint fails the build or merely reports.
    escalate_fingerprint: bool = False

    # --- `type: http` only ---
    method: str = "POST"
    path: str = ""
    #: Request body template. `{{prompt}}` is substituted.
    body: dict[str, Any] | None = None
    #: Dotted path to the text in the response, e.g. `choices.0.message.content`.
    response_path: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> TargetConfig:
        if self.type == "openai_compatible" and not self.model:
            raise ValueError(f"target {self.name!r}: `model` is required for openai_compatible")
        if self.type == "http" and self.body is None:
            raise ValueError(f"target {self.name!r}: `body` is required for type: http")
        return self

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"target {self.name!r} expects the API key in ${self.api_key_env}, "
                "which is unset."
            )
        return key

    def identity(self) -> dict[str, Any]:
        """The parts of a target that change what the output looks like."""
        return {
            "type": self.type,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "body": self.body,
            "path": self.path,
        }


class ProbeConfig(BaseModel):
    id: str
    prompt: str
    system: str | None = None

    #: Paid once, at baseline. This is where the variance estimate comes from, so
    #: it is worth more samples than a routine check.
    baseline_samples: int = 5
    #: Paid on every run, forever. Only needs to be enough to estimate a median --
    #: the variance is already known from the baseline.
    check_samples: int = 3

    checks: list[Any] = Field(default_factory=list)
    #: Which targets to run against. Empty means all of them.
    targets: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_samples_alias(cls, data: Any) -> Any:
        """`samples: 5` is the obvious thing to write, so accept it.

        It sets the baseline count; the per-run count keeps its cheaper default.
        """
        if isinstance(data, dict) and "samples" in data:
            data = dict(data)
            data.setdefault("baseline_samples", data.pop("samples"))
        return data

    @field_validator("checks", mode="before")
    @classmethod
    def _empty_checks_are_no_checks(cls, v: Any) -> Any:
        """`checks:` with nothing under it parses as null, not an empty list.

        It is an easy thing to leave behind while editing, and the probe is
        perfectly valid without any checks, so treat it as none rather than
        rejecting the file over whitespace.
        """
        return [] if v is None else v

    @field_validator("baseline_samples")
    @classmethod
    def _enough_for_variance(cls, v: int) -> int:
        if v < 2:
            raise ValueError(
                "baseline_samples must be at least 2 -- a variance band cannot be "
                "learned from a single sample."
            )
        return v

    @field_validator("check_samples")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("check_samples must be at least 1")
        return v


class ThresholdConfig(BaseModel):
    warn_k: float = 3.0
    drift_k: float = 6.0
    min_confident_n: int = 4
    corroborating_p: float = 0.01
    grey_zone: float = 0.6

    def to_band_config(self) -> BandConfig:
        return BandConfig(**self.model_dump())


class JudgeConfig(BaseModel):
    """The optional LLM judge.

    Its own endpoint block rather than a flag on a target, because the model you
    want explaining a regression is rarely the model you are monitoring. Judging
    with the same deployment you are watching means a provider-side change moves
    both the thing being measured and the instrument measuring it.
    """

    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 60.0
    #: Zero by default. A judge that phrases its answer differently every run adds
    #: noise to a tool whose entire job is telling signal from noise.
    temperature: float = 0.0
    max_tokens: int = 300

    #: Whether the judge may soften a verdict it considers cosmetic. Off by default.
    #: An escalation is a second opinion agreeing with the maths; a de-escalation
    #: overrides a band that was learned from the probe's own measured behaviour, on
    #: the say-so of a model that saw two samples. Useful for cutting noise once you
    #: trust it, but it should be a decision rather than a default.
    can_downgrade: bool = False

    def to_target(self) -> TargetConfig:
        return TargetConfig(
            name="__judge__",
            type="openai_compatible",
            base_url=self.base_url,
            model=self.model,
            api_key_env=self.api_key_env,
            headers=self.headers,
            timeout_s=self.timeout_s,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class AlertConfig(BaseModel):
    webhook: str | None = None
    slack_webhook: str | None = None
    #: Whether WARN breaks the build. Off by default: warnings are for reading,
    #: drift is for stopping.
    fail_on_warn: bool = False


class Config(BaseModel):
    targets: list[TargetConfig]
    probes: list[ProbeConfig]
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    #: Opt-in. Costs money, so it only runs on probes that already failed their
    #: band, which on a healthy day is none of them.
    judge: JudgeConfig | None = None
    #: `model2vec` (default, needs a one-time 32MB download) or `hashing`
    #: (fully offline, weaker at spotting meaning-preserving rewrites).
    embedder: Literal["model2vec", "hashing"] = "model2vec"
    #: Where baselines and history live, relative to the config file.
    state_dir: str = ".stillsane"

    @field_validator("targets")
    @classmethod
    def _unique_target_names(cls, v: list[TargetConfig]) -> list[TargetConfig]:
        names = [t.name for t in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate target names: {', '.join(sorted(dupes))}")
        return v

    @field_validator("probes")
    @classmethod
    def _unique_probe_ids(cls, v: list[ProbeConfig]) -> list[ProbeConfig]:
        ids = [p.id for p in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate probe ids: {', '.join(sorted(dupes))}")
        return v

    @model_validator(mode="after")
    def _targets_exist(self) -> Config:
        known = {t.name for t in self.targets}
        for probe in self.probes:
            unknown = set(probe.targets) - known
            if unknown:
                raise ValueError(
                    f"probe {probe.id!r} references unknown target(s): "
                    f"{', '.join(sorted(unknown))}"
                )
        return self

    def target(self, name: str) -> TargetConfig:
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(name)

    def pairs(self) -> list[tuple[ProbeConfig, TargetConfig]]:
        """Every probe/target combination that should run."""
        out = []
        for probe in self.probes:
            names = probe.targets or [t.name for t in self.targets]
            for name in names:
                out.append((probe, self.target(name)))
        return out


def config_hash(probe: ProbeConfig, target: TargetConfig, embedder: str = "model2vec") -> str:
    """Fingerprint of everything that would invalidate a stored baseline.

    Sample counts are excluded deliberately: asking for more samples does not
    change what the model says, so it should not throw away a baseline.

    The embedder *is* included, even though it changes nothing about the output.
    A baseline stores a variance pool of measured distances, and those numbers only
    mean anything on the scale that produced them. Switching embedder leaves the
    stored pool intact while every new distance arrives on a different scale, so the
    bands silently stop matching the measurements they are compared against and the
    verdicts are confidently wrong. Better to force a recapture.
    """
    payload = {
        "prompt": probe.prompt,
        "system": probe.system,
        "checks": probe.checks,
        "target": target.identity(),
        "embedder": embedder,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No config at {path}. Run `stillsane init` to generate one."
        )
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} should contain a YAML mapping at the top level.")
    return Config.model_validate(raw)
