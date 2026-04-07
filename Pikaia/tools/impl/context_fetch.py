"""
context_fetch
-------------
On-demand context retrieval for agents. Called during task execution when the
agent determines it needs more information to proceed.

The agent calls this with a plain-English query. The tool:
  1. Embeds the query
  2. Cosine-searches MT (long-term knowledge base)
  3. Cosine-searches dev/index.json (project file index)
  4. Reads snippets from top-ranked files
  5. Returns a pre-formatted text block ready for direct injection into a prompt

Agents never need to know about memory layers, embedding models, or file paths.
They just describe what they need.

params:
    query               : str   — plain English description of what to retrieve
    top_k               : int   — max results per source (default: 5)
    include_files       : bool  — whether to include file snippets (default: true)
    max_chars_per_file  : int   — max characters per file snippet (default: 1500)

returns:
    mt_entries  : list[{content, score}]
    files       : list[{path, summary, score, snippet}]
    text        : str — pre-formatted context block for LLM injection
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

_MT_MIN_SCORE   = 0.45
_FILE_MIN_SCORE = 0.40


def run(params: dict, context: dict) -> dict[str, Any]:
    query              = params["query"]
    top_k              = params.get("top_k", 5)
    include_files      = params.get("include_files", True)
    max_chars_per_file = params.get("max_chars_per_file", 1500)

    base_path = Path(context["base_path"])
    project   = context.get("project", "")

    # Delegate to ContextManager.fetch — same logic, no duplication
    # Import dynamically so this tool file has no hard import deps on siblings
    try:
        cm_path = base_path / "context_manager.py"
        spec = importlib.util.spec_from_file_location("context_manager", str(cm_path))
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        return mod.ContextManager.fetch(
            query              = query,
            project            = project,
            base_path          = base_path,
            context            = context,
            top_k              = top_k,
            include_files      = include_files,
            max_chars_per_file = max_chars_per_file,
        )
    except Exception as exc:
        # Fallback: inline minimal implementation so the agent always gets something
        return _fallback_fetch(query, project, base_path, context, top_k, max_chars_per_file)


# ---------------------------------------------------------------------------
# Inline fallback — mirrors ContextManager.fetch without the import
# ---------------------------------------------------------------------------

def _fallback_fetch(
    query:     str,
    project:   str,
    base_path: Path,
    context:   dict,
    top_k:     int,
    max_chars: int,
) -> dict[str, Any]:
    query_vec = _embed(query, context)

    mt_entries: list[dict] = []
    mt_path = base_path / "memory" / "mt.json"
    if mt_path.exists():
        try:
            all_mt = json.loads(mt_path.read_text())
            if isinstance(all_mt, list):
                active = [e for e in all_mt if e.get("status", "active") == "active"]
                if query_vec:
                    scored = sorted(
                        [(e, _cosine(query_vec, e.get("embedding", [])))
                         for e in active if e.get("embedding")],
                        key=lambda x: x[1], reverse=True,
                    )
                    mt_entries = [
                        {"content": e.get("content", ""), "score": round(s, 3)}
                        for e, s in scored[:top_k] if s > _MT_MIN_SCORE
                    ]
                else:
                    mt_entries = [{"content": e.get("content", ""), "score": 0.0}
                                  for e in active[:top_k]]
        except Exception:
            pass

    files: list[dict] = []
    dev_idx = base_path / "projects" / project / "dev" / "index.json"
    if dev_idx.exists() and query_vec:
        try:
            index = json.loads(dev_idx.read_text())
            scored_f = sorted(
                [(p, e, _cosine(query_vec, e.get("embedding", [])))
                 for p, e in index.items() if e.get("embedding")],
                key=lambda x: x[2], reverse=True,
            )
            for fpath, fentry, fscore in scored_f[:top_k]:
                if fscore < _FILE_MIN_SCORE:
                    break
                snippet = ""
                try:
                    snippet = Path(fpath).read_text(encoding="utf-8", errors="replace")[:max_chars]
                except Exception:
                    pass
                files.append({
                    "path":    fpath,
                    "summary": fentry.get("summary", ""),
                    "score":   round(fscore, 3),
                    "snippet": snippet,
                })
        except Exception:
            pass

    lines = [f"## Context retrieved for: {query[:100]}"]
    if mt_entries:
        lines.append("\n### Knowledge")
        for e in mt_entries:
            lines.append(f"- (score {e['score']}) {e['content']}")
    if files:
        lines.append("\n### Relevant files")
        for f in files:
            lines.append(f"\n**{f['path']}** (score {f['score']})")
            lines.append(f"Summary: {f['summary']}")
            if f["snippet"]:
                lines.append("```")
                lines.append(f["snippet"])
                lines.append("```")
    if not mt_entries and not files:
        lines.append("_No relevant context found for this query._")

    return {"mt_entries": mt_entries, "files": files, "text": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(text: str, context: dict) -> list[float] | None:
    try:
        base_path = Path(context["base_path"])
        impl_path = base_path / "tools" / "impl" / "embed_text.py"
        spec = importlib.util.spec_from_file_location("embed_text", str(impl_path))
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        return mod.run({"text": text}, context).get("embedding")
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
