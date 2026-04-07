"""
file_move
---------
Move or rename files — orchestrator only.
Primary use: promote worker/{agent_id}/output → dev/output/

The caller is responsible for triggering re-index after promotion
(Orchestrator._reindex_file already handles this).

params:
    src : str   - source path relative to base_path (or absolute)
    dst : str   - destination path relative to base_path (or absolute)

returns:
    moved        : bool
    src          : str
    dst          : str
    to_dev       : bool   - True if dst is under dev/ (caller should re-index)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    caller = context.get("caller", "orchestrator")
    if caller != "orchestrator":
        raise PermissionError("file_move is restricted to the orchestrator")

    base_path = Path(context["base_path"])

    def resolve(p: str) -> tuple[Path, Path]:
        raw = Path(p)
        if raw.is_absolute():
            full = raw
            try:
                rel = full.relative_to(base_path)
            except ValueError:
                raise PermissionError(f"Path '{p}' is outside base_path '{base_path}'")
        else:
            rel  = raw
            full = base_path / raw
        return full, rel

    src_full, src_rel = resolve(params["src"])
    dst_full, dst_rel = resolve(params["dst"])

    if not src_full.exists():
        raise FileNotFoundError(f"Source not found: {src_full}")

    dst_full.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_full), str(dst_full))

    to_dev = dst_rel.parts[0] == "dev" if dst_rel.parts else False

    return {
        "moved":  True,
        "src":    str(src_rel),
        "dst":    str(dst_rel),
        "to_dev": to_dev,
    }
