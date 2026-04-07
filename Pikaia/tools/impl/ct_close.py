"""
ct_close
--------
Close a CT (current-state) flag. Available to both orchestrator and agents.

Agents may only close flags whose agent_id matches their own.
Orchestrator may close any flag by task_id.

params:
    task_id : str             - the task_id of the CT flag to close
    status  : "done"|"failed" - new status

returns:
    closed  : bool
    task_id : str
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    task_id   = params["task_id"]
    status    = params.get("status", "done")
    caller    = context.get("caller", "orchestrator")
    agent_id  = context.get("agent_id")
    project   = context.get("project", "")
    base_path = Path(context["base_path"])

    if status not in ("done", "failed"):
        raise ValueError(f"status must be 'done' or 'failed', got '{status}'")

    ct_path = base_path / "projects" / project / "ct.json"
    if not ct_path.exists():
        return {"closed": False, "task_id": task_id}

    entries: list[dict] = []
    try:
        entries = json.loads(ct_path.read_text())
        if isinstance(entries, dict):
            entries = [entries]
    except Exception:
        return {"closed": False, "task_id": task_id}

    closed = False
    for flag in entries:
        if flag.get("task_id") != task_id:
            continue
        # Agents may only close their own flags
        if caller == "agent" and flag.get("agent_id") != agent_id:
            raise PermissionError(
                f"Agent '{agent_id}' is not allowed to close flag owned by '{flag.get('agent_id')}'"
            )
        if flag.get("status") in ("open", "pending_approval"):
            flag["status"]    = status
            flag["closed_at"] = datetime.now(timezone.utc).isoformat()
            closed = True

    if closed:
        _save_json(ct_path, entries)

    return {"closed": closed, "task_id": task_id}


def _save_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
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
