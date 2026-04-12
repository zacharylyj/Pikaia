"""
todo_write
----------
Manage a structured todo list for the current agent run.
Todos are persisted to worker_dir/todos.json (agent-scoped).

Mirrors the behaviour of Claude Code's /todo system so agents can track
multi-step progress the same way the assistant does.

params:
    todos : list[{content, status, activeForm?}]
            content    : str   — the task description (imperative form)
            status     : str   — "pending" | "in_progress" | "completed"
            activeForm : str   — present-continuous form shown during work (optional)

Constraints:
    - Exactly ONE todo may have status "in_progress" at a time.
    - Any number may be "pending" or "completed".

returns:
    written : bool
    count   : int
    path    : str   - worker_dir/todos.json

SCHEMA (self-registering)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = {
    "name": "todo_write",
    "description": (
        "Create or update the agent's todo list. "
        "Pass the full updated list each time. "
        "Exactly one item may be in_progress at a time. "
        "Persisted to worker_dir/todos.json."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Full todo list to write",
                "items": {
                    "type": "object",
                    "properties": {
                        "content":    {"type": "string"},
                        "status":     {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "activeForm": {"type": "string"},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["todos"],
    },
}

_VALID_STATUSES = {"pending", "in_progress", "completed"}


def run(params: dict, context: dict) -> dict[str, Any]:
    todos = params["todos"]
    if not isinstance(todos, list):
        raise ValueError("todos must be a list")

    # Validate entries
    in_progress_count = 0
    for i, item in enumerate(todos):
        if not isinstance(item, dict):
            raise ValueError(f"todos[{i}] must be a dict")
        if "content" not in item or "status" not in item:
            raise ValueError(f"todos[{i}] must have 'content' and 'status'")
        status = item["status"]
        if status not in _VALID_STATUSES:
            raise ValueError(f"todos[{i}].status must be one of {_VALID_STATUSES}, got '{status}'")
        if status == "in_progress":
            in_progress_count += 1

    if in_progress_count > 1:
        raise ValueError(
            f"Exactly one todo may be in_progress at a time; found {in_progress_count}"
        )

    # Write to worker_dir/todos.json
    worker_dir = context.get("worker_dir")
    if not worker_dir:
        raise RuntimeError("worker_dir not set in context — cannot write todos")

    todo_path = Path(worker_dir) / "todos.json"
    todo_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=todo_path.parent, prefix=".tmp_todos_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(todos, f, indent=2)
        os.replace(tmp, todo_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    return {
        "written": True,
        "count":   len(todos),
        "path":    str(todo_path),
    }
