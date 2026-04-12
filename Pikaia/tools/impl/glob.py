"""
glob
----
Find files matching a glob pattern, sorted by modification time (newest first).

Uses ripgrep (rg --files --glob) when available; falls back to
pathlib.Path.rglob for full cross-platform support.

params:
    pattern : str   - glob pattern, e.g. "**/*.py" or "src/**/*.ts"
    path    : str   - base directory to search in (default: base_path)

returns:
    files     : list[str]  - matching paths relative to search base, sorted mtime desc
    count     : int
    truncated : bool       - True if results were capped at max_results
    tool_used : str        - "rg" | "python"

SCHEMA (self-registering)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

SCHEMA = {
    "name": "glob",
    "description": (
        "Find files matching a glob pattern. "
        "Returns paths sorted by modification time (newest first). "
        "Supports patterns like '**/*.py', 'src/**/*.ts', '*.json'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files against (e.g. '**/*.py')",
            },
            "path": {
                "type": "string",
                "description": "Base directory to search in (default: base_path)",
            },
        },
        "required": ["pattern"],
    },
}

_MAX_RESULTS = 500


def _rg_available() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_rg(pattern: str, search_path: Path) -> dict[str, Any]:
    cmd = ["rg", "--files", "--glob", pattern, str(search_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"files": [], "count": 0, "truncated": True, "tool_used": "rg"}

    paths = [p for p in result.stdout.splitlines() if p]
    # Sort by mtime descending (rg output order is undefined)
    try:
        paths.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    except Exception:
        pass

    truncated = len(paths) > _MAX_RESULTS
    return {
        "files":     paths[:_MAX_RESULTS],
        "count":     len(paths),
        "truncated": truncated,
        "tool_used": "rg",
    }


def _py_glob(pattern: str, search_path: Path) -> dict[str, Any]:
    try:
        matched = list(search_path.rglob(pattern) if "**" in pattern
                       else search_path.glob(pattern))
    except Exception as exc:
        raise ValueError(f"Invalid glob pattern '{pattern}': {exc}") from exc

    # Only files, sort by mtime descending
    files = [p for p in matched if p.is_file()]
    try:
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        pass

    str_paths = [str(p) for p in files]
    truncated = len(str_paths) > _MAX_RESULTS
    return {
        "files":     str_paths[:_MAX_RESULTS],
        "count":     len(str_paths),
        "truncated": truncated,
        "tool_used": "python",
    }


def run(params: dict, context: dict) -> dict[str, Any]:
    base_path   = Path(context["base_path"])
    pattern     = params["pattern"]
    raw_path    = params.get("path", "")
    search_path = (base_path / raw_path).resolve() if raw_path else base_path.resolve()

    if not search_path.exists():
        raise FileNotFoundError(f"Search path not found: {search_path}")

    if _rg_available():
        return _run_rg(pattern, search_path)
    return _py_glob(pattern, search_path)
