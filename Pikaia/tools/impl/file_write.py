"""
file_write
----------
Write or overwrite a file with atomic write + path enforcement.

Path enforcement (paths resolved relative to base_path):
  orchestrator : projects/**, memory/** under base_path
  agent        : projects/{project}/worker/{agent_id}/** only
  skillsmith   : projects/{project}/worker/skillsmith/** only

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


def _allowed_roots(base: Path, caller: str, context: dict) -> list[Path]:
    project  = context.get("project", "")
    agent_id = context.get("agent_id")

    if caller == "orchestrator":
        roots = [base / "memory"]
        if project:
            roots.append(base / "projects" / project)
        else:
            roots.append(base / "projects")
        return roots

    if caller == "agent":
        if project and agent_id:
            return [base / "projects" / project / "worker" / agent_id]
        return []

    if caller == "skillsmith":
        if project:
            return [base / "projects" / project / "worker" / "skillsmith"]
        return []

    return []


def _within(full: Path, roots: list[Path]) -> bool:
    s = str(full.resolve())
    return any(s.startswith(str(r.resolve())) for r in roots)


def run(params: dict, context: dict) -> dict[str, Any]:
    base_path = Path(context["base_path"])
    caller    = context.get("caller", "orchestrator")

    raw = Path(params["path"])
    if raw.is_absolute():
        full_path = raw.resolve()
        try:
            rel = full_path.relative_to(base_path.resolve())
        except ValueError:
            raise PermissionError(f"Path '{raw}' is outside base_path '{base_path}'")
    else:
        full_path = (base_path / raw).resolve()
        rel       = raw

    roots = _allowed_roots(base_path, caller, context)
    if not _within(full_path, roots):
        raise PermissionError(
            f"Caller '{caller}' is not allowed to write to '{rel}'. "
            f"Allowed roots: {[str(r.relative_to(base_path)) for r in roots]}"
        )

    content = params["content"]
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
