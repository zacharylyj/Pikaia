"""
grep
----
Search file contents using regular expressions.

Uses ripgrep (rg) when available for speed; falls back to a pure-Python
implementation using the re module + os.walk so it always works.

params:
    pattern     : str   - regex pattern to search for
    path        : str   - file or directory to search in (default: base_path)
    glob        : str   - filter files by glob pattern (e.g. "*.py")
    type        : str   - file type alias (e.g. "py", "js", "ts")
    context     : int   - lines of context around each match (default: 0)
    ignore_case : bool  - case-insensitive (default: false)
    output_mode : str   - "files_with_matches" | "content" | "count"
                          (default: "files_with_matches")
    max_results : int   - max entries to return (default: 100)

returns:
    matches   : list
        files_with_matches → list[str]  — file paths
        content            → list[{file, line, text}]
        count              → list[{file, count}]
    truncated : bool
    tool_used : str   — "rg" | "python"

SCHEMA (self-registering)
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = {
    "name": "grep",
    "description": (
        "Search file contents using regular expressions. "
        "Uses ripgrep (rg) when available, pure Python otherwise. "
        "output_mode controls return shape: files_with_matches (paths only), "
        "content (matching lines), or count (match counts per file)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern":     {"type": "string",  "description": "Regex pattern to search for"},
            "path":        {"type": "string",  "description": "File or directory to search (default: base_path)"},
            "glob":        {"type": "string",  "description": "Glob filter for files, e.g. '*.py'"},
            "type":        {"type": "string",  "description": "File type alias: py, js, ts, json, md, ..."},
            "context":     {"type": "integer", "description": "Lines of context around each match (default: 0)"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
            "output_mode": {
                "type": "string",
                "enum": ["files_with_matches", "content", "count"],
                "description": "Output shape (default: files_with_matches)",
            },
            "max_results": {"type": "integer", "description": "Max results to return (default: 100)"},
        },
        "required": ["pattern"],
    },
}

# Map type aliases → glob patterns (subset of rg's built-in types)
_TYPE_GLOBS: dict[str, list[str]] = {
    "py":   ["*.py"],
    "js":   ["*.js", "*.mjs", "*.cjs"],
    "ts":   ["*.ts", "*.tsx"],
    "json": ["*.json"],
    "md":   ["*.md", "*.markdown"],
    "rust": ["*.rs"],
    "go":   ["*.go"],
    "java": ["*.java"],
    "cpp":  ["*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp"],
    "c":    ["*.c", "*.h"],
    "html": ["*.html", "*.htm"],
    "css":  ["*.css", "*.scss", "*.sass"],
    "sh":   ["*.sh", "*.bash"],
    "yaml": ["*.yaml", "*.yml"],
    "toml": ["*.toml"],
    "xml":  ["*.xml"],
    "sql":  ["*.sql"],
}


# ---------------------------------------------------------------------------
# Ripgrep path
# ---------------------------------------------------------------------------

def _rg_available() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_rg(
    pattern:     str,
    search_path: Path,
    glob_pat:    str | None,
    type_alias:  str | None,
    context:     int,
    ignore_case: bool,
    output_mode: str,
    max_results: int,
) -> dict[str, Any]:
    cmd = ["rg", "--no-heading"]

    if ignore_case:
        cmd.append("-i")
    if context > 0:
        cmd += ["-C", str(context)]

    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")

    if glob_pat:
        cmd += ["-g", glob_pat]
    if type_alias and type_alias in _TYPE_GLOBS:
        for g in _TYPE_GLOBS[type_alias]:
            cmd += ["-g", g]

    cmd += [pattern, str(search_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"matches": [], "truncated": True, "tool_used": "rg", "error": "timeout"}

    lines = result.stdout.splitlines()
    truncated = len(lines) > max_results
    lines = lines[:max_results]

    if output_mode == "files_with_matches":
        return {"matches": lines, "truncated": truncated, "tool_used": "rg"}

    if output_mode == "count":
        matches = []
        for line in lines:
            if ":" in line:
                f, c = line.rsplit(":", 1)
                try:
                    matches.append({"file": f, "count": int(c)})
                except ValueError:
                    pass
        return {"matches": matches, "truncated": truncated, "tool_used": "rg"}

    # content mode
    matches = []
    for line in lines:
        if ":" in line:
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    matches.append({"file": parts[0], "line": int(parts[1]), "text": parts[2]})
                except ValueError:
                    matches.append({"file": parts[0], "line": 0, "text": line})
            else:
                matches.append({"file": "", "line": 0, "text": line})
    return {"matches": matches, "truncated": truncated, "tool_used": "rg"}


# ---------------------------------------------------------------------------
# Pure-Python fallback
# ---------------------------------------------------------------------------

def _py_grep(
    pattern:     str,
    search_path: Path,
    glob_pat:    str | None,
    type_alias:  str | None,
    context:     int,
    ignore_case: bool,
    output_mode: str,
    max_results: int,
) -> dict[str, Any]:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc

    # Build effective glob list
    globs: list[str] = []
    if glob_pat:
        globs.append(glob_pat)
    if type_alias and type_alias in _TYPE_GLOBS:
        globs.extend(_TYPE_GLOBS[type_alias])

    def _matches_glob(name: str) -> bool:
        if not globs:
            return True
        return any(fnmatch.fnmatch(name, g) for g in globs)

    def _iter_files() -> list[Path]:
        if search_path.is_file():
            return [search_path]
        result = []
        for root, dirs, files in os.walk(search_path):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if _matches_glob(fname):
                    result.append(Path(root) / fname)
        return result

    files_list = _iter_files()
    matches: list = []
    truncated = False

    for fp in files_list:
        if len(matches) >= max_results:
            truncated = True
            break
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines_list = text.splitlines()
        file_match_count = 0
        content_hits: list[dict] = []

        for i, line in enumerate(lines_list):
            if rx.search(line):
                file_match_count += 1
                if output_mode == "content":
                    start = max(0, i - context)
                    end   = min(len(lines_list), i + context + 1)
                    for j in range(start, end):
                        content_hits.append({
                            "file": str(fp),
                            "line": j + 1,
                            "text": lines_list[j],
                        })

        if file_match_count == 0:
            continue

        if output_mode == "files_with_matches":
            matches.append(str(fp))
        elif output_mode == "count":
            matches.append({"file": str(fp), "count": file_match_count})
        elif output_mode == "content":
            matches.extend(content_hits)

    return {"matches": matches[:max_results], "truncated": truncated, "tool_used": "python"}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(params: dict, context: dict) -> dict[str, Any]:
    base_path   = Path(context["base_path"])
    pattern     = params["pattern"]
    raw_path    = params.get("path", "")
    glob_pat    = params.get("glob")
    type_alias  = params.get("type")
    ctx_lines   = int(params.get("context", 0))
    ignore_case = bool(params.get("ignore_case", False))
    output_mode = params.get("output_mode", "files_with_matches")
    max_results = int(params.get("max_results", 100))

    search_path = (base_path / raw_path).resolve() if raw_path else base_path.resolve()

    if _rg_available():
        return _run_rg(pattern, search_path, glob_pat, type_alias,
                       ctx_lines, ignore_case, output_mode, max_results)
    return _py_grep(pattern, search_path, glob_pat, type_alias,
                    ctx_lines, ignore_case, output_mode, max_results)
