"""
file_read
---------
Read any file as a string with scope enforcement and file_budget tracking.

Scope rules (paths resolved relative to base_path):
  orchestrator : any path under base_path
  agent        : memory/**, projects/{project}/dev/**,
                 projects/{project}/worker/{agent_id}/**
  skillsmith   : skills/**, projects/{project}/worker/skillsmith/**

params:
    path   : str       - path relative to base_path (or absolute)
    offset : int|None  - first line to return, 1-based (default: 1)
    limit  : int|None  - max lines to return (default: all)

returns:
    content    : str
    path       : str   - normalised relative path
    size_bytes : int
    lines      : int   - total lines in file
    truncated  : bool  - True when offset/limit caused partial read
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _allowed_roots(base: Path, caller: str, context: dict) -> list[Path]:
    project  = context.get("project", "")
    agent_id = context.get("agent_id")

    if caller == "orchestrator":
        return [base]   # unrestricted under base_path

    if caller == "agent":
        roots = [base / "memory"]
        if project:
            roots.append(base / "projects" / project / "dev")
        if project and agent_id:
            roots.append(base / "projects" / project / "worker" / agent_id)
        return roots

    if caller == "skillsmith":
        roots = [base / "skills"]
        if project:
            roots.append(base / "projects" / project / "worker" / "skillsmith")
        return roots

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
            f"Caller '{caller}' is not allowed to read '{rel}'. "
            f"Allowed roots: {[str(r.relative_to(base_path)) for r in roots]}"
        )

    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    if not full_path.is_file():
        raise IsADirectoryError(f"Path is a directory: {full_path}")

    raw_bytes = full_path.read_bytes()
    if b"\x00" in raw_bytes[:512]:
        raise ValueError(f"Binary file detected; file_read only supports text: {full_path}")

    full_content = raw_bytes.decode("utf-8", errors="replace")

    # Decrement file_budget if set
    file_budget = context.get("file_budget")
    if isinstance(file_budget, dict):
        file_budget["files_fetched"] = file_budget.get("files_fetched", 0) + 1
        if file_budget["files_fetched"] > file_budget.get("max_files", 999):
            raise RuntimeError(
                f"file_budget exceeded: max {file_budget['max_files']} files per task"
            )

    # Offset / limit slicing (1-based line numbers, matching cat -n convention)
    offset = params.get("offset")
    limit  = params.get("limit")
    all_lines = full_content.splitlines(keepends=True)
    total_lines = len(all_lines)
    truncated   = False

    if offset is not None or limit is not None:
        start = max(0, (int(offset) - 1) if offset else 0)
        end   = (start + int(limit)) if limit else len(all_lines)
        sliced = all_lines[start:end]
        truncated = (start > 0 or end < len(all_lines))
        content = "".join(sliced)
    else:
        content = full_content

    return {
        "content":    content,
        "path":       str(rel),
        "size_bytes": len(raw_bytes),
        "lines":      total_lines,
        "truncated":  truncated,
    }
