"""
db.py
-----
SQLite state backend for trajectory logging and observability metrics.
Uses WAL journal mode so readers never block writers.

Tables
------
trajectories
    One row per agent run. steps_json stores the full step sequence as a
    JSON array for offline replay / RL fine-tuning.

tool_events
    One row per tool dispatch. Captures name, success, latency, and the
    task it belonged to.

metrics
    One row per named metric observation (tokens_in, tokens_out, steps,
    latency_ms, etc.) tied to a task_id.

Usage
-----
    from db import Database
    db = Database(base_path / "pikaia.db")
    db.log_trajectory(task_id="t1", project="proj", steps=[...], outcome="done")
    db.log_tool_event(task_id="t1", tool_name="file_read", success=True, latency_ms=12)
    db.log_metric(task_id="t1", name="tokens_in", value=350)
    summary = db.metrics_summary(task_id="t1")

The default DB path is <base_path>/pikaia.db.
The orchestrator can override via config["db_path"].
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS trajectories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL,
    project     TEXT    NOT NULL DEFAULT '',
    agent_id    TEXT    NOT NULL DEFAULT '',
    tier        INTEGER NOT NULL DEFAULT 1,
    start_ts    TEXT    NOT NULL,
    end_ts      TEXT,
    outcome     TEXT    NOT NULL DEFAULT 'unknown',
    output      TEXT    NOT NULL DEFAULT '',
    steps_json  TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS tool_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL,
    tool_name   TEXT    NOT NULL,
    success     INTEGER NOT NULL DEFAULT 1,
    latency_ms  REAL    NOT NULL DEFAULT 0,
    error_msg   TEXT    NOT NULL DEFAULT '',
    ts          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    value       REAL    NOT NULL,
    ts          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traj_task  ON trajectories (task_id);
CREATE INDEX IF NOT EXISTS idx_tool_task  ON tool_events  (task_id);
CREATE INDEX IF NOT EXISTS idx_metric_task ON metrics     (task_id);
"""


class Database:
    """
    Thread-safe SQLite wrapper. A single connection is reused across threads
    via a lock (sqlite3 serialised mode is not reliable in all builds).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def log_trajectory(
        self,
        task_id:    str,
        project:    str,
        agent_id:   str,
        tier:       int,
        start_ts:   str,
        end_ts:     str,
        outcome:    str,
        output:     str,
        steps:      list[dict[str, Any]],
    ) -> None:
        sql = """
            INSERT INTO trajectories
                (task_id, project, agent_id, tier, start_ts, end_ts, outcome, output, steps_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(sql, (
                task_id, project, agent_id, tier,
                start_ts, end_ts, outcome, output,
                json.dumps(steps),
            ))
            self._conn.commit()

    def log_tool_event(
        self,
        task_id:    str,
        tool_name:  str,
        success:    bool,
        latency_ms: float,
        ts:         str,
        error_msg:  str = "",
    ) -> None:
        sql = """
            INSERT INTO tool_events (task_id, tool_name, success, latency_ms, error_msg, ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(sql, (task_id, tool_name, int(success), latency_ms, error_msg, ts))
            self._conn.commit()

    def log_metric(self, task_id: str, name: str, value: float, ts: str) -> None:
        sql = "INSERT INTO metrics (task_id, name, value, ts) VALUES (?, ?, ?, ?)"
        with self._lock:
            self._conn.execute(sql, (task_id, name, value, ts))
            self._conn.commit()

    def log_metrics_batch(self, rows: list[tuple[str, str, float, str]]) -> None:
        """Bulk-insert (task_id, name, value, ts) tuples."""
        sql = "INSERT INTO metrics (task_id, name, value, ts) VALUES (?, ?, ?, ?)"
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def metrics_summary(self, task_id: str) -> dict[str, float]:
        """Return {metric_name: total_value} aggregated for a task."""
        sql = "SELECT name, SUM(value) as total FROM metrics WHERE task_id=? GROUP BY name"
        with self._lock:
            rows = self._conn.execute(sql, (task_id,)).fetchall()
        return {r["name"]: r["total"] for r in rows}

    def tool_success_rate(self, task_id: str) -> dict[str, float]:
        """Return {tool_name: success_rate} for a task."""
        sql = """
            SELECT tool_name,
                   SUM(success) * 1.0 / COUNT(*) as rate
            FROM tool_events WHERE task_id=? GROUP BY tool_name
        """
        with self._lock:
            rows = self._conn.execute(sql, (task_id,)).fetchall()
        return {r["tool_name"]: round(r["rate"], 3) for r in rows}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        try:
            with self._lock:
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
        except Exception as exc:
            logger.error("db schema init failed: %s", exc)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton cache  (path → Database)
# ---------------------------------------------------------------------------

_instances: dict[str, Database] = {}
_instances_lock = threading.Lock()


def get_db(path: str | Path) -> Database:
    """Return (or create) the shared Database for *path*."""
    key = str(Path(path).resolve())
    with _instances_lock:
        if key not in _instances:
            _instances[key] = Database(key)
        return _instances[key]
