"""
memory_read
-----------
Unified read across all memory layers.

Layer routing:
  lt      → base_path/memory/lt.json           (global, all entries)
  mt      → base_path/memory/mt.json           (global, cosine search if query given)
  ct      → projects/{project}/ct.json         (per-project, all or filtered)
  st      → projects/{project}/instances/{instance_id}/st.json
  history → projects/{project}/instances/{instance_id}/history.json (cosine search)

params:
    layer       : "lt" | "mt" | "ct" | "st" | "history"
    query       : str | None    - semantic search query (used for MT and History)
    top_k       : int | None    - max results for RAG layers (default: 5)
    project     : str | None    - project name (falls back to context)
    instance_id : str | None    - instance id (falls back to context)

returns:
    list[dict]   (for lt / mt / ct / history)
    dict         (for st — the full st.json object)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> Any:
    layer       = params["layer"]
    query       = params.get("query", "")
    top_k       = params.get("top_k", 5)
    project     = params.get("project") or context.get("project", "")
    instance_id = params.get("instance_id") or context.get("instance_id", "")
    base_path   = Path(context["base_path"])

    if layer == "lt":
        return _read_lt(base_path)
    if layer == "mt":
        return _read_mt(base_path, query, top_k, context)
    if layer == "ct":
        return _read_ct(base_path, project)
    if layer == "st":
        return _read_st(base_path, project, instance_id)
    if layer == "history":
        return _read_history(base_path, project, instance_id, query, top_k, context)

    raise ValueError(f"Unknown memory layer: '{layer}'")


# ------------------------------------------------------------------
# Layer readers
# ------------------------------------------------------------------

def _read_lt(base_path: Path) -> list[dict]:
    path = base_path / "memory" / "lt.json"
    return _load_json_list(path)


def _read_mt(base_path: Path, query: str, top_k: int, context: dict) -> list[dict]:
    path    = base_path / "memory" / "mt.json"
    entries = _load_json_list(path)
    active  = [e for e in entries if e.get("status", "active") == "active"]

    if not query or not active:
        return active[:top_k]

    # Cosine search
    query_vec = _get_embedding(query, context)
    if query_vec:
        scored = [
            (e, _cosine(query_vec, e.get("embedding", [])))
            for e in active
            if e.get("embedding")
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    return active[:top_k]


def _read_ct(base_path: Path, project: str) -> list[dict]:
    path = base_path / "projects" / project / "ct.json"
    return _load_json_list(path)


def _read_st(base_path: Path, project: str, instance_id: str) -> dict:
    path = base_path / "projects" / project / "instances" / instance_id / "st.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "instance_id": instance_id,
        "project":     project,
        "summary":     "",
        "window":      [],
    }


def _read_history(
    base_path: Path,
    project: str,
    instance_id: str,
    query: str,
    top_k: int,
    context: dict,
) -> list[dict]:
    path    = base_path / "projects" / project / "instances" / instance_id / "history.json"
    entries = _load_json_list(path)

    if not query:
        return entries[-top_k:] if top_k else entries

    # Cosine search over content field
    query_vec = _get_embedding(query, context)
    if not query_vec:
        return entries[-top_k:]

    # Embed each entry lazily (expensive — use sparingly)
    scored: list[tuple[dict, float]] = []
    for e in entries:
        content = e.get("content", "")
        if not content:
            continue
        vec = _get_embedding(content, context)
        if vec:
            scored.append((e, _cosine(query_vec, vec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored[:top_k]]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass
    return []


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_embedding(text: str, context: dict) -> list[float] | None:
    """Call embed_text tool inline to get a vector. Returns None on failure."""
    try:
        base_path = Path(context["base_path"])
        import importlib.util
        impl_path = base_path / "tools" / "impl" / "embed_text.py"
        spec = importlib.util.spec_from_file_location("embed_text", str(impl_path))
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        result = mod.run({"text": text}, context)
        return result.get("embedding")
    except Exception:
        return None
