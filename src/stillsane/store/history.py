"""Run history in SQLite.

History exists so a human can answer "when did this start?" after an alert. That
question needs the per-signal numbers over time, not just a pass/fail, so every
signal of every probe gets a row.

SQLite because the brief allows it and nothing else is warranted: no server, one
file, and `sqlite3 .stillsane/history.sqlite` is a usable interface on its own.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..models import RunResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    started    TEXT NOT NULL,
    finished   TEXT,
    level      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id     TEXT NOT NULL,
    probe_id   TEXT NOT NULL,
    target     TEXT NOT NULL,
    signal     TEXT NOT NULL,
    level      TEXT NOT NULL,
    observed   REAL,
    baseline   REAL,
    z          REAL,
    p_value    REAL,
    band_upper REAL,
    band_lower REAL,
    detail     TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
CREATE INDEX IF NOT EXISTS results_probe_signal
    ON results (probe_id, target, signal);
"""


class History:
    def __init__(self, root: Path | str) -> None:
        self.path = Path(root) / "history.sqlite"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(self, result: RunResult) -> str:
        run_id = uuid.uuid4().hex[:12]
        finished = (result.finished or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started, finished, level) VALUES (?, ?, ?, ?)",
                (
                    run_id,
                    result.started.isoformat(timespec="seconds"),
                    finished,
                    result.level.value,
                ),
            )
            conn.executemany(
                "INSERT INTO results (run_id, probe_id, target, signal, level, observed, "
                "baseline, z, p_value, band_upper, band_lower, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        probe.probe_id,
                        probe.target_name,
                        sv.signal,
                        sv.level.value,
                        sv.observed,
                        sv.baseline,
                        sv.z,
                        sv.p_value,
                        sv.band.upper if sv.band else None,
                        sv.band.lower if sv.band else None,
                        sv.detail,
                    )
                    for probe in result.probes
                    for sv in probe.signals
                ],
            )
        return run_id

    def recent(self, limit: int = 20) -> list[tuple[str, str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                # rowid breaks ties. Timestamps are stored at second resolution, so
                # two runs in the same second sort arbitrarily without it -- and
                # arbitrary order is exactly wrong for a "what happened when" view.
                "SELECT run_id, finished, level FROM runs "
                "ORDER BY started DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def recorded_signals(self) -> list[tuple[str, str, str]]:
        """Every (probe, target, signal) that has history, for discovery.

        Asking someone to remember exact signal names before they can look at
        their own data is a good way to make the data unused.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT probe_id, target, signal FROM results "
                "ORDER BY probe_id, target, signal"
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def signal_trend(
        self, probe_id: str, target: str, signal: str, limit: int = 30
    ) -> list[tuple[str, float | None, float | None]]:
        """(timestamp, observed, z) over recent runs -- for answering 'since when?'."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT runs.finished, results.observed, results.z "
                "FROM results JOIN runs USING (run_id) "
                "WHERE results.probe_id = ? AND results.target = ? AND results.signal = ? "
                # See `recent` -- rowid breaks second-resolution timestamp ties.
                "ORDER BY runs.started DESC, results.rowid DESC LIMIT ?",
                (probe_id, target, signal, limit),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
