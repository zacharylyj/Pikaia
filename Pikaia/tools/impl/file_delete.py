"""
file_delete
-----------
Delete a file or empty directory — orchestrator only.

params:
    path : str   - path relative to base_path (or absolute)

returns:
    deleted : bool
    path    : str
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    caller = context.get("caller", "orchestrator")
    if caller != "orchestrator":
        raise PermissionError("file_delete is restricted to the orchestrator")

    base_path = Path(context["base_path"])
    raw_path  = Path(params["path"])

    if raw_path.is_absolute():
        full_path = raw_path.resolve()
        try:
            rel = full_path.relative_to(base_path.resolve())
        except ValueError:
            raise PermissionError(f"Path '{raw_path}' is outside base_path '{base_path}'")
    else:
        rel       = raw_path
        full_path = (base_path / raw_path).resolve()

    if not full_path.exists():
        return {"deleted": False, "path": str(rel)}

    if full_path.is_file():
        full_path.unlink()
    elif full_path.is_dir():
        # Only delete empty directories
        try:
            full_path.rmdir()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot delete non-empty directory '{rel}'. Remove contents first."
            ) from exc
    else:
        raise RuntimeError(f"Path is neither a file nor a directory: {rel}")

    return {"deleted": True, "path": str(rel)}
