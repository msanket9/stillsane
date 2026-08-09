"""Talking to a live endpoint.

Targets are the only part of stillsane that touches the network. Everything they
produce is a `Sample`, and everything downstream consumes only `Sample`s -- which
is what keeps the comparison engine testable without a network.

Failures are captured, never raised. A probe that times out becomes a Sample with
`error` set, so one dead call cannot abort a run and lose the other four samples
you already paid for.
"""

from __future__ import annotations

import asyncio
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import ProbeConfig, TargetConfig
from ..models import Sample

#: How many requests to a single target run at once. Deliberately low: probes are
#: sampled repeatedly, and a monitoring tool that trips someone's rate limit has
#: become the outage it was supposed to detect.
DEFAULT_CONCURRENCY = 4


#: `content[type=text]` -- pick the first list element whose field equals a value.
_FILTER = re.compile(r"^([^\[\]]*)\[([^=\[\]]+)=([^\[\]]*)\]$")


def dotted_get(data: Any, path: str) -> Any:
    """Resolve a path like `choices.0.message.content` against decoded JSON.

    Three segment forms, in order of how often they are needed:

    * `key`        -- a dict lookup
    * `0`          -- a list index
    * `key[f=v]`   -- the first element of the list at `key` whose field `f`
                      equals `v`; a bare `[f=v]` filters the current list

    The filter form exists because index paths are not stable on providers that
    return heterogeneous content blocks. With thinking enabled, Anthropic's
    `content` array leads with a thinking block, so `content.0.text` silently
    resolves to the wrong block and the probe compares an empty string forever.
    `content[type=text].text` says what is actually meant.

    The general answer is to embed an expression evaluator and let people write
    arbitrary extraction code, which is a great deal of surface area for the last
    few percent of response shapes.
    """
    current = data
    for segment in path.split("."):
        if current is None:
            return None

        match = _FILTER.match(segment)
        if match:
            key, field, wanted = match.groups()
            if key:
                current = current.get(key) if isinstance(current, dict) else None
            if not isinstance(current, list):
                return None
            current = next(
                (
                    item
                    for item in current
                    if isinstance(item, dict) and str(item.get(field)) == wanted
                ),
                None,
            )
        elif segment.isdigit() and isinstance(current, list):
            index = int(segment)
            current = current[index] if index < len(current) else None
        elif isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def render_template(value: Any, variables: dict[str, str]) -> Any:
    """Substitute `{{name}}` placeholders throughout a nested structure."""
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{{" + key + "}}", replacement)
        return value
    if isinstance(value, dict):
        return {k: render_template(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, variables) for v in value]
    return value


async def _backoff(seconds: float) -> None:
    """The wait between retries, behind a seam.

    A named function rather than a bare `asyncio.sleep` so the suite can replace the
    waiting without replacing sleeping everywhere. Retry behaviour is worth testing;
    ten seconds of real backoff spread across the tests that simulate dead endpoints
    is how a fast suite quietly becomes one nobody runs.
    """
    if seconds > 0:
        await asyncio.sleep(seconds)


class Target(ABC):
    """Base for anything stillsane can point at."""

    def __init__(self, config: TargetConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def build_request(self, probe: ProbeConfig) -> tuple[str, str, dict[str, str], dict[str, Any]]:
        """Return (method, url, headers, json_body)."""

    @abstractmethod
    def parse(self, probe: ProbeConfig, body: Any) -> dict[str, Any]:
        """Pull the comparable fields out of a decoded response body."""

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.config.headers}
        key = self.config.api_key()
        if key:
            headers.setdefault(
                self.config.api_key_header.lower(), f"{self.config.api_key_prefix}{key}"
            )
        return headers

    async def call(self, probe: ProbeConfig, client: httpx.AsyncClient) -> Sample:
        """One sample, retrying only failures that measured nothing.

        The rule that matters is which failures are eligible. A timeout or a dropped
        connection means the request never landed, so trying again asks the same
        question a second time. A verdict is different: if the probe answered and the
        answer was drift, asking again until it comes up clean is the same defect as
        a monitor that silently re-baselines. So retries are gated on transport, and
        `_attempt` reports that separately from whether the sample merely failed.

        Four scheduled runs were lost to transient transport failures in one week
        while building this, and every manual re-run minutes later succeeded, which
        is what the default of one retry is calibrated against.
        """
        attempts = 1 + max(0, self.config.retries)
        for attempt in range(1, attempts + 1):
            sample, transient = await self._attempt(probe, client)
            sample.attempts = attempt
            if not transient or attempt == attempts:
                return sample
            # Backoff doubles, so a provider having a bad minute is not hammered.
            await _backoff(self.config.retry_backoff_s * (2 ** (attempt - 1)))
        return sample

    async def _attempt(
        self, probe: ProbeConfig, client: httpx.AsyncClient
    ) -> tuple[Sample, bool]:
        """One invocation. Never raises for anything the endpoint did.

        Returns the sample and whether its failure is worth retrying. That flag is
        deliberately not a field on `Sample`: it describes this attempt, not the
        observation, and persisting it would invite someone to treat a stored sample
        as retryable long after the fact.
        """
        sample = Sample(probe_id=probe.id, target_name=self.name)
        method, url, headers, payload = self.build_request(probe)

        started = time.perf_counter()
        try:
            response = await client.request(
                method, url, headers=headers, json=payload, timeout=self.config.timeout_s
            )
            sample.latency_ms = (time.perf_counter() - started) * 1000.0
            sample.http_status = response.status_code

            if response.status_code >= 400:
                sample.error = f"HTTP {response.status_code} {response.reason_phrase}".strip()
                # Keep a snippet: the provider's error body is usually the fastest
                # route to the cause, and it is gone once the run ends.
                sample.text = response.text[:500]
                # 429 and 5xx are the endpoint asking to be asked again. Every other
                # 4xx is a wrong key, a wrong path or a malformed body, and repeating
                # it just spends money to get the same answer more slowly.
                retryable = response.status_code == 429 or response.status_code >= 500
                return sample, retryable

            body = response.json()
            sample.raw = body if isinstance(body, dict) else {"response": body}
            for field, value in self.parse(probe, body).items():
                setattr(sample, field, value)

        except httpx.TimeoutException:
            sample.latency_ms = (time.perf_counter() - started) * 1000.0
            sample.error = f"timeout after {self.config.timeout_s}s"
            return sample, True
        except httpx.HTTPError as exc:
            # Connection-level: refused, reset, DNS, a read that died mid-flight.
            # The request did not land, so nothing was measured.
            sample.latency_ms = (time.perf_counter() - started) * 1000.0
            sample.error = f"{type(exc).__name__}: {exc}"
            return sample, True
        except ValueError as exc:
            # The endpoint answered with something that is not JSON. That is a
            # contract problem rather than a blip, and it comes back identical.
            sample.latency_ms = (time.perf_counter() - started) * 1000.0
            sample.error = f"response was not valid JSON: {exc}"
            return sample, False
        return sample, False


async def collect(
    target: Target,
    probe: ProbeConfig,
    n: int,
    concurrency: int = DEFAULT_CONCURRENCY,
    client: httpx.AsyncClient | None = None,
) -> list[Sample]:
    """Take `n` samples of one probe against one target."""
    owned = client is None
    client = client or httpx.AsyncClient()
    limit = asyncio.Semaphore(concurrency)

    async def one() -> Sample:
        async with limit:
            return await target.call(probe, client)

    try:
        return list(await asyncio.gather(*(one() for _ in range(n))))
    finally:
        if owned:
            await client.aclose()
