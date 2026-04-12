"""
question
--------
Ask the user a question and wait for their response.

Mechanism
---------
1. Writes the question to worker_dir/question.json
2. Updates agent state to "waiting_for_input" so the orchestrator / CLI can
   surface the question to the user
3. Polls worker_dir/answer.json for a response (written by the orchestrator
   after the user answers)
4. Falls back to blocking stdin input if running interactively (sys.stdin.isatty)
5. Returns {"answer": "", "from": "timeout"} after `timeout` seconds if nothing arrives

The orchestrator's monitor loop detects "waiting_for_input" in state.json and
should prompt the user via cli_output and then write the answer to answer.json.

params:
    question : str         - the question to ask the user
    choices  : list[str]   - optional list of valid choices to present
    timeout  : int         - seconds to wait for a response (default: 120)

returns:
    answer : str
    from   : str   - "user" | "stdin" | "timeout"

SCHEMA (self-registering)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = {
    "name": "question",
    "description": (
        "Ask the user a question and wait for their response. "
        "The orchestrator surfaces the question to the user and writes back the answer. "
        "Use this when you need clarification or a decision from the user mid-task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of valid answer choices to present",
            },
            "timeout": {
                "type": "integer",
                "description": "Seconds to wait before giving up (default: 120)",
            },
        },
        "required": ["question"],
    },
}

_POLL_INTERVAL = 0.5   # seconds between answer.json checks


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_q_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run(params: dict, context: dict) -> dict[str, Any]:
    question = params["question"]
    choices  = params.get("choices", [])
    timeout  = int(params.get("timeout", 120))

    worker_dir = context.get("worker_dir")
    if not worker_dir:
        raise RuntimeError("worker_dir not set in context — cannot use question tool")

    worker_path  = Path(worker_dir)
    question_path = worker_path / "question.json"
    answer_path   = worker_path / "answer.json"
    state_path    = worker_path / "state.json"

    # Remove any stale answer from a previous question
    if answer_path.exists():
        try:
            answer_path.unlink()
        except OSError:
            pass

    # Write question file
    _atomic_write(question_path, {
        "question":  question,
        "choices":   choices,
        "asked_at":  _now_iso(),
        "timeout":   timeout,
    })

    # Update state to signal waiting_for_input
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            state["status"]   = "waiting_for_input"
            state["question"] = question
            _atomic_write(state_path, state)
        except Exception:
            pass

    # --- Try interactive stdin first (works when running in a terminal) ---
    try:
        if sys.stdin.isatty():
            prompt = question
            if choices:
                prompt += f"\nChoices: {', '.join(choices)}"
            prompt += "\n> "
            answer = input(prompt).strip()
            _cleanup(question_path, state_path)
            return {"answer": answer, "from": "stdin"}
    except Exception:
        pass

    # --- Poll answer.json (orchestrator will write this) ---
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if answer_path.exists():
            try:
                data   = json.loads(answer_path.read_text())
                answer = str(data.get("answer", ""))
                answer_path.unlink(missing_ok=True)
                _cleanup(question_path, state_path)
                return {"answer": answer, "from": "user"}
            except Exception:
                pass
        time.sleep(_POLL_INTERVAL)

    # Timed out
    _cleanup(question_path, state_path)
    return {"answer": "", "from": "timeout"}


def _cleanup(question_path: Path, state_path: Path) -> None:
    """Remove question file and restore state status to 'running'."""
    try:
        question_path.unlink(missing_ok=True)
    except Exception:
        pass
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            if state.get("status") == "waiting_for_input":
                state["status"] = "running"
                state.pop("question", None)
                _atomic_write(state_path, state)
        except Exception:
            pass
