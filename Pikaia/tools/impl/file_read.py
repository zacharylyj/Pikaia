"""
file_read
---------
Read any file as a string with scope enforcement and file_budget tracking.

Scope rules (path relative to base_path):
  orchestrator : any path under base_path
  agent        : dev/**, memory/**, worker/{agent_id}/**
  skillsmith   : skills/**, worker/skillsmith/**

params:
    path : str   - path relative to base_path (or absolute)

returns:
    content    : str
    path       : str   - normalised relative path
    size_bytes : int
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Directories each caller type may read, relative to base_path
_ALLOWED: dict[str, list[str]] = {
    "orchestrator": [],          # empty = unrestricted under base_path
    "agent":        ["dev", "memory"],
    "skillsmith":   ["skills", "worker/skillsmith"],
}


def _check_scope(rel: Path, caller: str, agent_id: str | None) -> bool:
    """Return True if rel path is within the caller's allowed scope."""
    allowed = _ALLOWED.get(caller, [])
    if not allowed:          # orchestrator — unrestricted
        return True

    # Agent can also read its own worker slot
    parts = list(rel.parts)
    if caller == "agent" and agent_id:
        if parts[:2] == ["worker", agent_id]:
            return True

    for prefix in allowed:
        prefix_parts = Path(prefix).parts
        if tuple(parts[: len(prefix_parts)]) == prefix_parts:
            return True
    return False


def run(params: dict, context: dict) -> dict[str, Any]:
    base_path = Path(context["base_path"])
    caller    = context.get("caller", "orchestrator")
    agent_id  = context.get("agent_id")

    raw_path = Path(params["path"])
    if raw_path.is_absolute():
        full_path = raw_path
        try:
            rel = full_path.relative_to(base_path)
        except ValueError:
            raise PermissionError(f"Path '{raw_path}' is outside base_path '{base_path}'")
    else:
        rel       = raw_path
        full_path = base_path / raw_path

    if not _check_scope(rel, caller, agent_id):
        raise PermissionError(
            f"Caller '{caller}' is not allowed to read '{rel}'. "
            f"Allowed: {_ALLOWED.get(caller, ['(unrestricted)'])}"
        )

    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    if not full_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {full_path}")

    # Binary detection — read first 512 bytes
    raw_bytes = full_path.read_bytes()
    if b"\x00" in raw_bytes[:512]:
        raise ValueError(f"Binary file detected; file_read only supports text: {full_path}")

    content = raw_bytes.decode("utf-8", errors="replace")

    # Decrement file_budget if set (mutable in-place via the dict reference)
    file_budget = context.get("file_budget")
    if isinstance(file_budget, dict):
        file_budget["files_fetched"] = file_budget.get("files_fetched", 0) + 1
        if file_budget["files_fetched"] > file_budget.get("max_files", 999):
            raise RuntimeError(
                f"file_budget exceeded: max {file_budget['max_files']} files per task"
            )

    return {
        "content":    content,
        "path":       str(rel),
        "size_bytes": len(raw_bytes),
    }
