"""A single, shared way to write a file so a reader never sees a partial one.

Extracted from `baseline.py`, which needed it first (`variance.json` is
rewritten on every clean check), because `runs.py` needs exactly the same
guarantee for `samples.jsonl` and a second, silently-non-atomic
implementation would just be this bug waiting to be reintroduced.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` so a reader never sees a partial file.

    A crash or kill mid-write leaves whatever `write_text` had flushed so
    far -- often truncated, invalid content that the next read then raises
    on, with no way back short of deleting the file by hand. Writing to a
    sibling temp file first and `os.replace`-ing it into place is atomic on
    the same filesystem: the old content survives intact until the new
    content is fully written, so a crash mid-write loses only the write in
    progress, never the file itself.

    The temp name carries a random suffix rather than a fixed one so two
    writers racing for the same path don't stomp each other's temp file
    before either gets to `replace`.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)
