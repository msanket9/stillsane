"""Turning a log file into a probe set.

Nobody wants to hand-write twenty probes, and the ones you would write by hand are
the ones you already think about. The prompts actually hitting your endpoint are a
better sample of what matters, and you already have them.

The hard part is not reading the file, it is that real logs are enormously
repetitive: a thousand requests are usually a handful of shapes with different
payloads stuffed into them. Emitting a probe per line would be useless. So distinct
prompts get clustered by meaning and one representative is picked per cluster,
using the same embedder that already ships for drift detection -- no new dependency
and no new failure mode.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .signals.semantic import Embedder

#: Distances below this are treated as the same prompt wearing different clothes.
#: Tuned to be forgiving: over-merging costs a probe you could have had, while
#: under-merging fills the config with near-duplicates and makes the whole feature
#: not worth using.
DEFAULT_MERGE_DISTANCE = 0.12

#: Anything shorter is almost certainly a fragment rather than a real prompt.
MIN_PROMPT_CHARS = 12


@dataclass
class Candidate:
    """One distinct prompt found in the logs, and how often it appeared."""

    prompt: str
    system: str | None = None
    count: int = 1
    members: list[str] = field(default_factory=list)


def _text_of(message: Any) -> str:
    """Content can be a string or the multipart list form."""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = [p.get("text", "") for p in message if isinstance(p, dict)]
        return "\n".join(p for p in parts if p)
    return ""


def extract_from_record(record: Any) -> tuple[str, str | None] | None:
    """Pull (prompt, system) out of one logged request.

    Handles the shapes people actually have: an OpenAI-style request body, a bare
    `{"prompt": ...}`, or a log line that wraps either under `request`/`body`.
    Returns None for anything unrecognisable rather than guessing.
    """
    if not isinstance(record, dict):
        return None

    for key in ("request", "body", "payload"):
        inner = record.get(key)
        if isinstance(inner, dict):
            found = extract_from_record(inner)
            if found:
                return found

    messages = record.get("messages")
    if isinstance(messages, list):
        system = next(
            (
                _text_of(m.get("content"))
                for m in messages
                if isinstance(m, dict) and m.get("role") == "system"
            ),
            None,
        )
        user = [
            _text_of(m.get("content"))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user and user[-1].strip():
            return user[-1].strip(), (system.strip() if system else None)

    for key in ("prompt", "input", "text", "question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), None
    return None


def read_records(source: Path) -> Iterator[Any]:
    """Read a JSONL file, a JSON array, or a directory of JSON files.

    Malformed lines are skipped rather than fatal. Logs are messy by nature, and
    refusing to read a 10,000-line file because line 4,812 was truncated mid-write
    would make this useless on exactly the files it exists for.
    """
    if source.is_dir():
        for path in sorted(source.glob("*.json")):
            try:
                yield json.loads(path.read_text())
            except (ValueError, OSError):
                continue
        return

    text = source.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(text)
        except ValueError:
            data = []
        if isinstance(data, list):
            yield from data
        return

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def collect_candidates(records: Iterable[Any]) -> list[Candidate]:
    """Distinct prompts, most frequent first.

    Exact duplicates are folded here; near-duplicates are handled later by
    clustering. Frequency is kept because it is the best available proxy for which
    prompts are worth monitoring.
    """
    seen: dict[tuple[str, str | None], Candidate] = {}
    for record in records:
        found = extract_from_record(record)
        if not found:
            continue
        prompt, system = found
        if len(prompt) < MIN_PROMPT_CHARS:
            continue
        key = (prompt, system)
        if key in seen:
            seen[key].count += 1
        else:
            seen[key] = Candidate(prompt=prompt, system=system)
    return sorted(seen.values(), key=lambda c: (-c.count, c.prompt))


def cluster(
    candidates: list[Candidate],
    embedder: Embedder,
    merge_distance: float = DEFAULT_MERGE_DISTANCE,
) -> list[Candidate]:
    """Merge near-duplicate prompts, keeping the most frequent as representative.

    Greedy single-pass over candidates already sorted by frequency, so the survivor
    of each cluster is the most-seen variant. That is deliberate: it makes the
    emitted probe the one closest to what the endpoint actually handles most.

    Greedy rather than a proper clustering algorithm because the input is a few
    hundred prompts at most and the failure mode of getting it slightly wrong is a
    config the user edits anyway.
    """
    if len(candidates) < 2:
        return list(candidates)

    vectors = embedder.encode([c.prompt for c in candidates])
    kept: list[Candidate] = []
    kept_vectors: list[np.ndarray] = []

    for candidate, vector in zip(candidates, vectors, strict=True):
        merged_into = None
        for existing, existing_vector in zip(kept, kept_vectors, strict=True):
            if float(1.0 - np.dot(vector, existing_vector)) <= merge_distance:
                merged_into = existing
                break
        if merged_into is None:
            kept.append(candidate)
            kept_vectors.append(vector)
        else:
            merged_into.count += candidate.count
            merged_into.members.append(candidate.prompt)
    return kept


def _block(text: str, indent: str) -> str:
    """Render a prompt as a YAML literal block, which survives any content.

    Prompts contain quotes, colons and newlines. A literal block sidesteps every
    escaping question and, more importantly, leaves the prompt readable in the
    config file, which is where the user is going to edit it.
    """
    lines = text.strip().splitlines() or [""]
    return "\n".join(f"{indent}  {line}".rstrip() for line in lines)


def to_yaml(candidates: list[Candidate], limit: int = 20) -> str:
    """Emit the `probes:` section. The target block is written separately.

    Prompts are quoted verbatim rather than templated. A probe is only meaningful
    if it is the exact text the endpoint saw, so this deliberately does not try to
    parameterise anything out.
    """
    taken: set[str] = set()
    out = ["probes:"]
    for candidate in candidates[:limit]:
        pid = probe_id(candidate.prompt, taken)
        seen = f"  # seen {candidate.count} time{'s' if candidate.count != 1 else ''} in the logs"
        if candidate.members:
            seen += f", merged with {len(candidate.members)} near-duplicate(s)"
        out.append(seen)
        out.append(f"  - id: {pid}")
        if candidate.system:
            out.append("    system: |")
            out.append(_block(candidate.system, "    "))
        out.append("    prompt: |")
        out.append(_block(candidate.prompt, "    "))
        # The whole block stays commented, key included. A bare `checks:` with
        # nothing under it parses as null rather than an empty list, which fails
        # validation -- so a half-commented block would emit a config that does not
        # load, which is the one thing this feature cannot afford to do.
        out.append("    # checks:")
        out.append("    #   - valid_json")
        out.append("    #   - has_keys: [id, status]")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def probe_id(prompt: str, taken: set[str]) -> str:
    """A short, readable, unique id derived from the prompt itself."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in prompt.lower()).split() if w]
    skip = {"the", "a", "an", "of", "for", "from", "this", "that", "and", "to", "in", "is"}
    meaningful = [w for w in words if w not in skip][:4] or ["probe"]
    base = "_".join(meaningful)[:40]

    name = base
    n = 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name
