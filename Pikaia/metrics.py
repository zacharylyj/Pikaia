"""
metrics.py
----------
Per-run metrics collector. Accumulates tokens, latency, step counts, and
tool success/failure stats during an agent run, then flushes everything to
SQLite in a single batch at the end.

Usage (inside AgentRunner)
--------------------------
    mc = MetricsCollector(task_id="t1", enabled=True)
    mc.record_tokens(tokens_in=200, tokens_out=80)
    mc.record_tool_call(name="file_read", success=True, latency_ms=14.3)
    mc.record_step()
    mc.flush(db)           # writes to db.metrics / db.tool_events

The orchestrator can disable collection per-task via
task_packet["metrics_enabled"] = False (or global config["metrics_enabled"]).
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db import Database

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetricsCollector:

    def __init__(self, task_id: str, enabled: bool = True) -> None:
        self.task_id   = task_id
        self.enabled   = enabled
        self._start    = time.monotonic()

        # Aggregates
        self.tokens_in:  int = 0
        self.tokens_out: int = 0
        self.steps:      int = 0

        # Per-tool events: list of (tool_name, success, latency_ms, error_msg, ts)
        self._tool_events: list[tuple[str, bool, float, str, str]] = []

    # ------------------------------------------------------------------
    # Record helpers (no-ops when disabled)
    # ------------------------------------------------------------------

    def record_tokens(self, tokens_in: int, tokens_out: int) -> None:
        if not self.enabled:
            return
        self.tokens_in  += tokens_in
        self.tokens_out += tokens_out

    def record_tool_call(
        self,
        name:       str,
        success:    bool,
        latency_ms: float,
        error_msg:  str = "",
    ) -> None:
        if not self.enabled:
            return
        self._tool_events.append((name, success, latency_ms, error_msg, _now_iso()))

    def record_step(self) -> None:
        if self.enabled:
            self.steps += 1

    # ------------------------------------------------------------------
    # Derived stats
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000

    @property
    def tool_success_rate(self) -> float:
        if not self._tool_events:
            return 1.0
        successes = sum(1 for _, s, *_ in self._tool_events if s)
        return round(successes / len(self._tool_events), 3)

    # ------------------------------------------------------------------
    # Flush to DB
    # ------------------------------------------------------------------

    def flush(self, db: "Database") -> None:
        """Write all accumulated metrics and tool events to SQLite."""
        if not self.enabled:
            return
        ts = _now_iso()
        try:
            # Aggregate metrics batch
            metric_rows = [
                (self.task_id, "tokens_in",   float(self.tokens_in),   ts),
                (self.task_id, "tokens_out",  float(self.tokens_out),  ts),
                (self.task_id, "steps",       float(self.steps),        ts),
                (self.task_id, "latency_ms",  self.elapsed_ms,          ts),
            ]
            db.log_metrics_batch(metric_rows)

            # Per-tool events
            for tool_name, success, latency_ms, error_msg, event_ts in self._tool_events:
                db.log_tool_event(
                    task_id=self.task_id,
                    tool_name=tool_name,
                    success=success,
                    latency_ms=latency_ms,
                    ts=event_ts,
                    error_msg=error_msg,
                )
        except Exception as exc:
            logger.warning("MetricsCollector.flush failed: %s", exc)
