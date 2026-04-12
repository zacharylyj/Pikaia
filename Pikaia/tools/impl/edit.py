"""
edit
----
Modify an existing file using exact string replacement.
Enforces the same path scope rules as file_read/file_write.

Exactly one occurrence of old_string must exist in the file unless
replace_all=True.  If the string does not appear, or appears more than once
when replace_all is False, the call raises with a clear message so the agent
knows to refine its approach.

params:
    path        : str   - file path relative to base_path (or absolute)
    old_string  : str   - exact text to find and replace
    new_string  : str   - replacement text (may be empty to delete)
    replace_all : bool  - replace every occurrence (default: false)

returns:
    written      : bool
    path         : str
    replacements : int   - number of replacements made

SCHEMA (self-registering)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Self-registering schema for auto-discovery by tools/schemas.py
SCHEMA = {
    "name": "edit",
    "description": (
        "Modify an existing file using exact string replacement. "
        "old_string must appear exactly once unless replace_all=True. "
        "Use this to make targeted edits without rewriting the whole file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to base_path",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace (must be unique in file unless replace_all=True)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text (may be empty to delete old_string)",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence of old_string (default: false)",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
}


# ---------------------------------------------------------------------------
# Path scope — mirrors file_write restrictions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(params: dict, context: dict) -> dict[str, Any]:
    base_path   = Path(context["base_path"])
    caller      = context.get("caller", "orchestrator")
    old_string  = params["old_string"]
    new_string  = params["new_string"]
    replace_all = bool(params.get("replace_all", False))

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
            f"Caller '{caller}' is not allowed to edit '{rel}'. "
            f"Allowed roots: {[str(r.relative_to(base_path)) for r in roots]}"
        )

    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    if not full_path.is_file():
        raise IsADirectoryError(f"Path is a directory: {full_path}")

    content = full_path.read_text(encoding="utf-8", errors="replace")

    count = content.count(old_string)
    if count == 0:
        raise ValueError(
            f"old_string not found in '{rel}'. "
            "Ensure the text matches the file exactly (including indentation and whitespace)."
        )
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string appears {count} times in '{rel}'. "
            "Provide more surrounding context to make it unique, or set replace_all=True."
        )

    if replace_all:
        new_content  = content.replace(old_string, new_string)
        replacements = count
    else:
        new_content  = content.replace(old_string, new_string, 1)
        replacements = 1

    # Atomic write
    full_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=full_path.parent, prefix=".tmp_edit_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(tmp, full_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    return {
        "written":      True,
        "path":         str(rel),
        "replacements": replacements,
    }
