"""
file_write
----------
Write or overwrite a file with atomic write + path enforcement.

Path enforcement:
  orchestrator : dev/**, memory/** under base_path
  agent        : worker/{agent_id}/** only
  skillsmith   : worker/skillsmith/** only

params:
    path    : str   - path relative to base_path (or absolute)
    content : str   - text to write

returns:
    written    : bool
    path       : str   - normalised relative path
    size_bytes : int
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def _check_write_scope(rel: Path, caller: str, agent_id: str | None) -> bool:
    parts = list(rel.parts)
    if caller == "orchestrator":
        return parts[0] in ("dev", "memory", "projects")
    if caller == "agent":
        if agent_id and parts[:2] == ["worker", agent_id]:
            return True
        return False
    if caller == "skillsmith":
        return parts[:2] == ["worker", "skillsmith"]
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

    if not _check_write_scope(rel, caller, agent_id):
        raise PermissionError(
            f"Caller '{caller}' is not allowed to write to '{rel}'"
        )

    content = params["content"]

    # Atomic write: write to a temp file in the same directory, then rename
    full_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=full_path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, full_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    size = full_path.stat().st_size
    return {
        "written":    True,
        "path":       str(rel),
        "size_bytes": size,
    }
