"""
list
----
List files and directories at a given path.

params:
    path      : str   - directory to list (default: base_path)
    recursive : bool  - recurse into subdirectories (default: false)

returns:
    entries : list[{name, type, size_bytes, modified}]
        type: "file" | "dir" | "symlink"
        modified: ISO-8601 timestamp
    path    : str   - the resolved path that was listed
    count   : int

SCHEMA (self-registering)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = {
    "name": "list",
    "description": (
        "List files and directories at a given path. "
        "Set recursive=true to walk the full directory tree."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list (default: base_path)",
            },
            "recursive": {
                "type": "boolean",
                "description": "Recurse into subdirectories (default: false)",
            },
        },
        "required": [],
    },
}

_MAX_ENTRIES = 1000


def _entry(p: Path, base: Path) -> dict[str, Any]:
    try:
        stat = p.lstat()
        size = stat.st_size if p.is_file() else 0
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        size, mtime = 0, ""

    if p.is_symlink():
        kind = "symlink"
    elif p.is_dir():
        kind = "dir"
    else:
        kind = "file"

    try:
        rel = str(p.relative_to(base))
    except ValueError:
        rel = str(p)

    return {"name": rel, "type": kind, "size_bytes": size, "modified": mtime}


def run(params: dict, context: dict) -> dict[str, Any]:
    base_path = Path(context["base_path"])
    raw_path  = params.get("path", "")
    recursive = bool(params.get("recursive", False))

    target = (base_path / raw_path).resolve() if raw_path else base_path.resolve()

    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    if not target.is_dir():
        # Single file — return it as a one-entry list
        return {
            "entries": [_entry(target, target.parent)],
            "path":    str(target),
            "count":   1,
        }

    if recursive:
        children = sorted(target.rglob("*"), key=lambda p: str(p))
    else:
        children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))

    entries = [_entry(p, target) for p in children[:_MAX_ENTRIES]]
    return {
        "entries": entries,
        "path":    str(target),
        "count":   len(entries),
    }
