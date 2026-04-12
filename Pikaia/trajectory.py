"""
trajectory.py
-------------
Per-run trajectory logger. Records the complete sequence of steps (LLM
turns, tool calls, tool results) for a single agent run.

Purpose
-------
- Replay buffer for offline RL / supervised fine-tuning
- Audit trail for debugging agent behaviour
- Input to future skill-auto-improvement pipelines

Storage
-------
1. JSONL file at <base_path>/projects/<project>/trajectories/<task_id>.jsonl
   Always written (append-safe, one JSON object per line).
2. SQLite trajectories table via db.Database (if db is provided at finalise()).

Each step record has the shape:
    {
        "seq":       int,           # 0-based step number
        "type":      str,           # "llm_turn" | "tool_call" | "tool_result" | "compress"
        "ts":        str,           # ISO-8601 timestamp
        <type-specific keys>
    }

Usage
-----
    tl = TrajectoryLogger(
        task_id  = "task_abc",
        project  = "my-project",
        base_path= Path("/path/to/Pikaia"),
        enabled  = True,
    )
    tl.log_llm_turn(content="...", tokens_in=100, tokens_out=80)
    tl.log_tool_call(name="file_read", args={"path": "foo.txt"})
    tl.log_tool_result(name="file_read", result="file content...", latency_ms=12.0)
    tl.finalise(outcome="done", output="final answer", db=db_instance)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from db import Database

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrajectoryLogger:

    def __init__(
        self,
        task_id:   str,
        agent_id:  str,
        project:   str,
        tier:      int,
        base_path: Path,
        enabled:   bool = True,
    ) -> None:
        self.task_id   = task_id
        self.agent_id  = agent_id
        self.project   = project
        self.tier      = tier
        self.base_path = base_path
        self.enabled   = enabled
        self._steps:   list[dict[str, Any]] = []
        self._seq      = 0
        self._start_ts = _now_iso()

        if enabled:
            traj_dir = base_path / "projects" / project / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = traj_dir / f"{task_id}.jsonl"
        else:
            self._jsonl_path = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Step loggers
    # ------------------------------------------------------------------

    def log_llm_turn(
        self,
        content:    str,
        tokens_in:  int = 0,
        tokens_out: int = 0,
        step:       int = 0,
    ) -> None:
        self._append({
            "type":       "llm_turn",
            "step":       step,
            "content":    content[:500],   # truncate for storage
            "tokens_in":  tokens_in,
            "tokens_out": tokens_out,
        })

    def log_tool_call(self, name: str, args: dict[str, Any], step: int = 0) -> None:
        # Truncate large args (e.g. file contents passed as params)
        safe_args: dict = {}
        for k, v in args.items():
            sv = str(v)
            safe_args[k] = sv[:200] if len(sv) > 200 else v
        self._append({
            "type": "tool_call",
            "step": step,
            "name": name,
            "args": safe_args,
        })

    def log_tool_result(
        self,
        name:       str,
        result:     Any,
        latency_ms: float = 0.0,
        success:    bool  = True,
        step:       int   = 0,
    ) -> None:
        result_str = str(result)
        self._append({
            "type":       "tool_result",
            "step":       step,
            "name":       name,
            "success":    success,
            "latency_ms": round(latency_ms, 1),
            "result":     result_str[:300],
        })

    def log_compression(self, before_msgs: int, after_msgs: int, step: int = 0) -> None:
        self._append({
            "type":        "compress",
            "step":        step,
            "before_msgs": before_msgs,
            "after_msgs":  after_msgs,
        })

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalise(
        self,
        outcome: str,
        output:  str,
        db:      "Database | None" = None,
    ) -> None:
        if not self.enabled:
            return

        end_ts = _now_iso()

        # 1. Flush all buffered steps to JSONL (atomic write)
        self._flush_jsonl()

        # 2. Write summary row to SQLite if db provided
        if db is not None:
            try:
                db.log_trajectory(
                    task_id  = self.task_id,
                    project  = self.project,
                    agent_id = self.agent_id,
                    tier     = self.tier,
                    start_ts = self._start_ts,
                    end_ts   = end_ts,
                    outcome  = outcome,
                    output   = output[:1000],
                    steps    = self._steps,
                )
            except Exception as exc:
                logger.warning("TrajectoryLogger DB write failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record["seq"] = self._seq
        record["ts"]  = _now_iso()
        self._seq    += 1
        self._steps.append(record)

    def _flush_jsonl(self) -> None:
        if not self._steps or self._jsonl_path is None:
            return
        try:
            parent = self._jsonl_path.parent
            fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp_traj_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    for step in self._steps:
                        f.write(json.dumps(step) + "\n")
                os.replace(tmp, self._jsonl_path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("TrajectoryLogger JSONL flush failed: %s", exc)
