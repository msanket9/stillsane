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

#: `Level.rank`'s own ordering, inlined as SQL -- SQLite has no way to call
#: back into that Python mapping, and these are exactly `Level`'s four string
#: values, so a change to one without the other is the only way this drifts.
_LEVEL_RANK_SQL = "CASE results.level WHEN 'error' THEN 3 WHEN 'drift' THEN 2 WHEN 'warn' THEN 1 ELSE 0 END"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    started    TEXT NOT NULL,
    finished   TEXT,
    level      TEXT NOT NULL,
    retries    INTEGER NOT NULL DEFAULT 0
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


#: Columns added after the first release, as (table, column, definition).
#:
#: `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so a
#: database written by an older version keeps its old shape forever and every insert
#: naming the new column fails. Anyone who has been running this on a schedule has
#: exactly such a file, and losing their history to read one number back would be a
#: poor trade. Adding a nullable column is cheap and leaves old rows readable.
_ADDED_COLUMNS = (
    ("runs", "retries", "INTEGER NOT NULL DEFAULT 0"),
    # Which baseline version a row was compared against. `trend` needs this to
    # scope a signal's history to runs judged against the *current* baseline --
    # otherwise a re-baseline that genuinely absorbed a shift would have its old
    # numbers averaged in with the new ones, and the first runs after recapture
    # would read as a shift that is actually just the old baseline's tail. Rows
    # written before this column existed come back NULL and are excluded from
    # trend on purpose: there is no way to know which baseline they were judged
    # against, and guessing would be worse than staying quiet until enough new
    # rows accumulate.
    ("results", "baseline_version", "INTEGER"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, definition in _ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


class History:
    def __init__(self, root: Path | str) -> None:
        self.path = Path(root) / "history.sqlite"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.executescript(SCHEMA)
            _migrate(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(self, result: RunResult) -> str:
        run_id = uuid.uuid4().hex[:12]
        finished = (result.finished or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started, finished, level, retries) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    result.started.isoformat(timespec="seconds"),
                    finished,
                    result.level.value,
                    sum(p.retries for p in result.probes),
                ),
            )
            conn.executemany(
                "INSERT INTO results (run_id, probe_id, target, signal, level, observed, "
                "baseline, z, p_value, band_upper, band_lower, detail, baseline_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        probe.baseline_version,
                    )
                    for probe in result.probes
                    for sv in probe.signals
                ],
            )
        return run_id

    def recent(self, limit: int = 20) -> list[tuple[str, str, str, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                # rowid breaks ties. Timestamps are stored at second resolution, so
                # two runs in the same second sort arbitrarily without it -- and
                # arbitrary order is exactly wrong for a "what happened when" view.
                "SELECT run_id, finished, level, retries FROM runs "
                "ORDER BY started DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

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

    def probe_results(
        self, limit_runs: int = 50
    ) -> list[tuple[str, str, str, str, str, str, str | None]]:
        """(finished, run_id, probe_id, target, signal, level, detail), newest runs first.

        Per signal rather than per probe, because the caller needs the `transport`
        rows specifically: an unreachable endpoint and a drifting one both end a run
        early, and only `signal` says which happened -- `detail` alone can't be
        matched reliably, and a probe-level aggregate would throw the distinction
        away before the caller ever saw it.

        The run limit is applied to *runs* rather than to rows, since a row cap
        would silently truncate a run's probes and make a healthy run look partial.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT runs.finished, results.run_id, results.probe_id, results.target, "
                "results.signal, results.level, results.detail "
                "FROM results JOIN runs USING (run_id) "
                "WHERE results.run_id IN ("
                "  SELECT run_id FROM runs ORDER BY started DESC, rowid DESC LIMIT ?"
                ") "
                # See `recent` -- rowid breaks second-resolution timestamp ties.
                "ORDER BY runs.started DESC, results.rowid DESC",
                (limit_runs,),
            ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]

    def clean_z(
        self, limit_runs: int = 200
    ) -> list[tuple[str, str, str, float, float | None, float | None]]:
        """(probe_id, target, signal, z, observed, baseline) for every signal of
        every clean run.

        Clean runs only, and that restriction is the whole point: a run that passed
        is a run where nothing drifted, so the `z` values it recorded are what normal
        looks like. How close those get to `warn_k` is how close the tool came to
        crying wolf. A run that drifted would contaminate the answer with the very
        thing being excluded.

        `observed`/`baseline` ride along with `z` because `z` alone cannot tell
        `calibrate` whether a signal that never crossed `warn_k` genuinely never
        moved, or moved substantially but only in the direction nobody alerts on
        -- `z_score` clamps that case to exactly 0 before it is ever recorded.
        Comparing the raw values lets the report say which, without needing to
        reconstruct a band's scale (and risk doing it against the wrong `warn_k`,
        if thresholds changed since the row was written).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT results.probe_id, results.target, results.signal, results.z, "
                "results.observed, results.baseline "
                "FROM results JOIN runs USING (run_id) "
                "WHERE runs.level = 'pass' AND results.z IS NOT NULL "
                "AND results.run_id IN ("
                "  SELECT run_id FROM runs WHERE level = 'pass' "
                "  ORDER BY started DESC, rowid DESC LIMIT ?"
                ")",
                (limit_runs,),
            ).fetchall()
        return [
            (
                r[0],
                r[1],
                r[2],
                float(r[3]),
                None if r[4] is None else float(r[4]),
                None if r[5] is None else float(r[5]),
            )
            for r in rows
        ]

    def clean_run_span(self) -> tuple[int, str | None, str | None]:
        """(count, earliest, latest) over clean runs, for sizing the evidence."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT count(*), min(finished), max(finished) FROM runs WHERE level = 'pass'"
            ).fetchone()
        return (row[0] or 0, row[1], row[2])

    def run_span(self) -> tuple[int, str | None, str | None]:
        """(count, earliest, latest) over every run ever recorded, any level.

        Unlike `recent`/`probe_results`, never capped by a `--limit` -- `status`
        uses this to print "history since <date>, N runs" regardless of how many
        rows a particular render chose to show. That line is the tell for a CI
        deployment whose history lives in a best-effort cache: a cache miss resets
        the database silently, and a report that only ever showed the last 20 runs
        would look identical the day after a reset as it did a month in. Showing
        the true span makes the reset visible instead of silent.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT count(*), min(finished), max(finished) FROM runs").fetchone()
        return (row[0] or 0, row[1], row[2])

    def trend_window(
        self, probe_id: str, target: str, signal: str, baseline_version: int, window: int
    ) -> tuple[int, list[float], list[float]]:
        """(total runs, earliest `window` observed values, most recent `window`
        observed values) for one signal, scoped to one exact baseline version.

        The version scoping is what keeps a re-baseline from contaminating the
        comparison: a sustained shift is exactly what recapturing should absorb,
        so a run recorded against a since-replaced baseline must not be averaged
        in with runs judged against the current one, or the first runs after
        recapture would read as a shift that is really just the old baseline's
        tail. Rows written before this column existed (`baseline_version IS
        NULL`) never match an integer version and so are correctly excluded
        rather than guessed at.

        Two bounded queries rather than one page fetched newest-first and split
        in Python: a single `ORDER BY started DESC LIMIT n` page only ever gives
        the *recent* end, and reversing it cannot recover the *earliest* rows once
        a probe has run more than `n` times under one baseline version -- exactly
        the case (a long-lived baseline) this command exists to say something
        useful about. Fetching each end with its own bounded query costs one more
        round trip and is correct regardless of how much history there is.
        """
        with self._connect() as conn:
            total = conn.execute(
                "SELECT count(*) FROM results JOIN runs USING (run_id) "
                "WHERE results.probe_id = ? AND results.target = ? AND results.signal = ? "
                "AND results.baseline_version = ? AND results.observed IS NOT NULL",
                (probe_id, target, signal, baseline_version),
            ).fetchone()[0]
            early = conn.execute(
                "SELECT results.observed FROM results JOIN runs USING (run_id) "
                "WHERE results.probe_id = ? AND results.target = ? AND results.signal = ? "
                "AND results.baseline_version = ? AND results.observed IS NOT NULL "
                "ORDER BY runs.started ASC, results.rowid ASC LIMIT ?",
                (probe_id, target, signal, baseline_version, window),
            ).fetchall()
            recent = conn.execute(
                "SELECT results.observed FROM results JOIN runs USING (run_id) "
                "WHERE results.probe_id = ? AND results.target = ? AND results.signal = ? "
                "AND results.baseline_version = ? AND results.observed IS NOT NULL "
                "ORDER BY runs.started DESC, results.rowid DESC LIMIT ?",
                (probe_id, target, signal, baseline_version, window),
            ).fetchall()
        return (
            total or 0,
            [float(r[0]) for r in early],
            [float(r[0]) for r in reversed(recent)],
        )

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

    def probe_recent_levels(
        self, probe_id: str, target: str, limit: int = 400
    ) -> list[tuple[str, int]]:
        """(finished, worst level rank) per run this probe/target appears in,
        newest first.

        Reduced to the worst rank across that probe's own signal rows within
        each run -- "one error anywhere in the run decides the probe's level
        for that run", the same rule `status.assess` already applies when it
        builds `ProbeHealth`. `runner.check` walks this backward from the most
        recent prior run to answer "since when has this probe been non-PASS,
        with no clean run in between" -- see `first_seen`/`consecutive_runs`
        on `ProbeVerdict`.

        Ranks rather than level strings, so the caller can compare with a
        plain `== 0` / `> 0` instead of repeating the level-name mapping.
        """
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT runs.finished, MAX({_LEVEL_RANK_SQL}) "
                "FROM results JOIN runs USING (run_id) "
                "WHERE results.probe_id = ? AND results.target = ? "
                "GROUP BY runs.run_id "
                # See `recent` -- rowid breaks second-resolution timestamp ties.
                "ORDER BY runs.started DESC, runs.rowid DESC LIMIT ?",
                (probe_id, target, limit),
            ).fetchall()
        return [(r[0], int(r[1])) for r in rows]
