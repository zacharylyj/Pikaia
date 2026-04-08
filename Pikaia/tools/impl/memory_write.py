"""
memory_write
------------
Write to a memory layer — orchestrator only.
Agents use ct_close() which is a thin wrapper around this tool.

Layer routing:
  lt      → base_path/memory/lt.json            append new entry
  mt      → base_path/memory/mt.json            append; auto-embeds content if no embedding
  ct      → projects/{project}/ct.json          append or update existing by id
  st      → projects/{project}/instances/{instance_id}/st.json   full replace

params:
    layer       : "lt" | "mt" | "ct" | "st"
    entry       : dict          - the record to write
    project     : str | None    - falls back to context
    instance_id : str | None    - falls back to context

returns:
    written : bool
    layer   : str
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    caller = context.get("caller", "orchestrator")
    if caller != "orchestrator":
        raise PermissionError(
            "memory_write is restricted to the orchestrator. "
            "Agents should use ct_close() to update CT flags."
        )

    layer       = params["layer"]
    entry       = dict(params["entry"])
    project     = params.get("project") or context.get("project", "")
    instance_id = params.get("instance_id") or context.get("instance_id", "")
    base_path   = Path(context["base_path"])

    if layer == "lt":
        _write_lt(base_path, entry)
    elif layer == "mt":
        _write_mt(base_path, entry, context)
    elif layer == "ct":
        _write_ct(base_path, project, entry)
    elif layer == "st":
        _write_st(base_path, project, instance_id, entry)
    elif layer == "kg":
        import sys as _sys
        _pikaia = str(base_path)
        if _pikaia not in _sys.path:
            _sys.path.insert(0, _pikaia)
        from mt_palace import kg_write  # type: ignore[import]
        kg_result = kg_write(entry, base_path)
        return {"written": True, "layer": "kg", **kg_result}
    else:
        raise ValueError(f"Unknown memory layer: '{layer}'")

    return {"written": True, "layer": layer}


# ------------------------------------------------------------------
# Layer writers
# ------------------------------------------------------------------

def _write_lt(base_path: Path, entry: dict) -> None:
    path = base_path / "memory" / "lt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _load_list(path)
    entries.append(entry)
    _save_json(path, entries)


def _write_mt(base_path: Path, entry: dict, context: dict) -> None:
    path = base_path / "memory" / "mt.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-embed if embedding is missing
    if not entry.get("embedding") and entry.get("content"):
        embedding = _get_embedding(entry["content"], context)
        if embedding:
            entry["embedding"] = embedding

    # Palace enrichment: wing/room/entities/importance/AAAK
    try:
        import importlib.util as _ilu
        import sys as _sys
        _pikaia = str(base_path)
        if _pikaia not in _sys.path:
            _sys.path.insert(0, _pikaia)
        from mt_palace import MTWriter   # type: ignore[import]
        entry = MTWriter.enrich(entry, base_path, context)
    except Exception:
        pass  # graceful fallback — entry still gets saved without palace fields

    entries = _load_list(path)

    # Update in-place if same id exists, otherwise append
    existing_idx = next((i for i, e in enumerate(entries) if e.get("id") == entry.get("id")), None)
    if existing_idx is not None:
        entries[existing_idx] = entry
    else:
        entries.append(entry)

    _save_json(path, entries)


def _write_ct(base_path: Path, project: str, entry: dict) -> None:
    path = base_path / "projects" / project / "ct.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _load_list(path)

    existing_idx = next((i for i, e in enumerate(entries) if e.get("id") == entry.get("id")), None)
    if existing_idx is not None:
        entries[existing_idx] = entry
    else:
        entries.append(entry)

    _save_json(path, entries)


def _write_st(base_path: Path, project: str, instance_id: str, entry: dict) -> None:
    path = base_path / "projects" / project / "instances" / instance_id / "st.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_json(path, entry)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _save_json(path: Path, data: Any) -> None:
    import os, tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_embedding(text: str, context: dict) -> list[float] | None:
    try:
        import importlib.util
        base_path = Path(context["base_path"])
        impl_path = base_path / "tools" / "impl" / "embed_text.py"
        spec = importlib.util.spec_from_file_location("embed_text", str(impl_path))
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        result = mod.run({"text": text}, context)
        return result.get("embedding")
    except Exception:
        return None
