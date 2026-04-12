"""
apply_patch
-----------
Apply a unified diff patch to one or more files.

Uses the system `patch` command (available via Git on Windows / standard on
Unix) with Python difflib as a fallback for single-file patches.

The patch text should be standard unified diff format (output of
`diff -u oldfile newfile` or `git diff`).

params:
    patch   : str   - unified diff content
    path    : str   - target file (optional; overrides paths in patch headers)
    dry_run : bool  - validate without writing (default: false)
    strip   : int   - path strip level passed to patch -p (default: 1)

returns:
    applied  : bool
    patched  : list[str]   - files that were modified
    rejected : str         - rejection output if patch partially failed
    dry_run  : bool

SCHEMA (self-registering)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = {
    "name": "apply_patch",
    "description": (
        "Apply a unified diff patch to files. "
        "Accepts standard unified diff format (diff -u / git diff). "
        "Uses the system patch command; falls back to Python difflib."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "Unified diff patch content",
            },
            "path": {
                "type": "string",
                "description": "Target file path (overrides paths in patch headers)",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Validate without writing changes (default: false)",
            },
            "strip": {
                "type": "integer",
                "description": "Path strip level (-p flag for patch command, default: 1)",
            },
        },
        "required": ["patch"],
    },
}


# ---------------------------------------------------------------------------
# System patch command
# ---------------------------------------------------------------------------

def _patch_available() -> bool:
    try:
        subprocess.run(["patch", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_patch(
    patch_text: str,
    cwd:        str,
    strip:      int,
    dry_run:    bool,
    target:     str | None,
) -> dict[str, Any]:
    fd, patch_file = tempfile.mkstemp(suffix=".patch", prefix="pikaia_patch_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(patch_text)

        cmd = ["patch", f"-p{strip}", "--batch"]
        if dry_run:
            cmd.append("--dry-run")
        if target:
            cmd += [target]
        cmd += ["--input", patch_file]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, timeout=30
        )
    finally:
        try:
            os.unlink(patch_file)
        except OSError:
            pass

    applied  = result.returncode == 0
    rejected = result.stderr.strip() or ""

    # Extract patched filenames from stdout
    patched: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("patching file "):
            patched.append(line[len("patching file "):].strip())

    return {
        "applied":  applied,
        "patched":  patched,
        "rejected": rejected if not applied else "",
        "dry_run":  dry_run,
    }


# ---------------------------------------------------------------------------
# Pure-Python fallback (single-file patches only)
# ---------------------------------------------------------------------------

def _py_apply_patch(patch_text: str, base_path: Path, dry_run: bool) -> dict[str, Any]:
    """
    Minimal unified diff applier for single-file patches.
    Parses the first --- / +++ headers to find the target file, then applies.
    """
    lines = patch_text.splitlines(keepends=True)
    target_path: Path | None = None

    for line in lines:
        if line.startswith("+++ "):
            # "+++ b/some/file.py" or "+++ some/file.py"
            raw = line[4:].split("\t")[0].strip()
            # Strip leading a/ or b/ prefixes
            if raw.startswith("b/"):
                raw = raw[2:]
            elif raw.startswith("a/"):
                raw = raw[2:]
            target_path = base_path / raw
            break

    if target_path is None or not target_path.exists():
        return {
            "applied":  False,
            "patched":  [],
            "rejected": "Could not determine target file from patch headers",
            "dry_run":  dry_run,
        }

    original = target_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    # Simple hunk applier
    result_lines = list(original)
    in_hunk = False
    old_pos = 0  # 0-indexed into result_lines

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            # Parse @@ -start,count +start,count @@
            import re
            m = re.search(r"@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
            if m:
                old_start = int(m.group(1)) - 1  # 0-indexed
                old_pos   = old_start
            in_hunk = True
            i += 1
            continue

        if in_hunk:
            if line.startswith(" "):
                old_pos += 1
            elif line.startswith("-"):
                if old_pos < len(result_lines):
                    result_lines.pop(old_pos)
                # don't advance old_pos since we removed a line
            elif line.startswith("+"):
                result_lines.insert(old_pos, line[1:])
                old_pos += 1
            else:
                in_hunk = False

        i += 1

    if not dry_run:
        target_path.write_text("".join(result_lines), encoding="utf-8")

    return {
        "applied":  True,
        "patched":  [str(target_path)],
        "rejected": "",
        "dry_run":  dry_run,
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(params: dict, context: dict) -> dict[str, Any]:
    base_path  = Path(context["base_path"])
    patch_text = params["patch"]
    target     = params.get("path")
    dry_run    = bool(params.get("dry_run", False))
    strip      = int(params.get("strip", 1))

    if _patch_available():
        return _run_patch(
            patch_text = patch_text,
            cwd        = str(base_path),
            strip      = strip,
            dry_run    = dry_run,
            target     = target,
        )
    return _py_apply_patch(patch_text, base_path, dry_run)
