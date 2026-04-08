"""
context_manager.py
------------------
Two responsibilities:

1. Pre-dispatch assessment (called by Orchestrator before spawning any agent):
     cm.assess(task_packet, project) → enriched task_packet
   - Runs a cheap LLM gap-identification call (haiku / context_assessment pipeline)
   - For each gap, queries MT + file index with targeted sub-queries
   - Injects the new material into task_packet["context"] before the agent starts
   - Tags the packet with {pre_enriched, sufficient, gaps} so agents know the state

2. On-demand fetch (the internals behind the `context_fetch` tool):
     ContextManager.fetch(query, project, base_path, context) → {mt_entries, files, text}
   - Embeds the query
   - Cosine searches MT + dev/index.json
   - Reads snippets from top-ranked files
   - Returns a ready-to-inject text block
   This static method is called directly by the context_fetch tool impl.

Design principle: agents never know which memory layer to look in or how embeddings
work. They ask a question in plain English and get usable context back.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Score below which an MT hit is considered "weak" (worth trying to enrich)
_MT_WEAK_THRESHOLD   = 0.55
# Score below which a file hit is ignored
_FILE_MIN_THRESHOLD  = 0.40
# Max chars to read from each file for a snippet
_SNIPPET_MAX_CHARS   = 1500
# How many additional MT hits to fetch per gap
_GAP_MT_TOP_K        = 3
# How many additional files to surface per gap
_GAP_FILE_TOP_K      = 2


class ContextManager:
    """Owned by the Orchestrator. Created once at startup."""

    def __init__(self, tools: Any, base_path: Path, config: Any) -> None:
        # tools  : Orchestrator.Tools instance
        # config : OrchestratorConfig instance
        self._tools     = tools
        self._base_path = base_path
        self._config    = config

    # ------------------------------------------------------------------
    # Public: pre-dispatch assessment
    # ------------------------------------------------------------------

    def assess(self, task_packet: dict, project: str) -> dict:
        """
        Check if the task packet has sufficient context.
        Enriches it in-place if gaps are found.
        Returns the (possibly enriched) task_packet.

        Adds to task_packet["context"]:
          pre_enriched  : bool
          sufficient    : bool
          gaps          : list[str]   — what was missing
          enriched_with : list[str]   — what was added
        """
        objective = task_packet.get("objective", "")
        ctx       = task_packet.setdefault("context", {})

        sufficient    = True
        gaps:          list[str] = []
        enriched_with: list[str] = []

        # ---- Quick heuristic: are existing MT hits actually relevant? ----
        mt_scored = ctx.get("mt_retrieved", [])
        weak_hits = sum(
            1 for e in mt_scored
            if isinstance(e, dict) and e.get("score", 1.0) < _MT_WEAK_THRESHOLD
        )
        if weak_hits == len(mt_scored) and mt_scored:
            sufficient = False
            logger.debug("ContextManager: all MT hits are weak — will enrich")

        # ---- LLM gap identification ----
        llm_gaps:    list[str] = []
        llm_queries: list[str] = []
        try:
            llm_gaps, llm_queries = self._identify_gaps(objective, ctx)
            if llm_gaps:
                sufficient = False
                gaps.extend(llm_gaps)
        except Exception as e:
            logger.warning("ContextManager gap identification failed: %s", e)

        # ---- Enrich for each gap query ----
        if llm_queries:
            for q in llm_queries:
                added = self._enrich_for_query(q, project, ctx)
                if added:
                    enriched_with.append(f"[{q[:60]}] → {added} items")

        # If no explicit gaps but MT is sparse for a complex tier, try a direct re-fetch.
        # Do NOT override sufficient=True here — if the LLM identified genuine gaps
        # that we couldn't fill, the agent should still know context is thin.
        tier = task_packet.get("tier", 2)
        if tier >= 3 and len(mt_scored) < 2:
            added = self._enrich_for_query(objective, project, ctx)
            if added:
                enriched_with.append(f"direct re-fetch → {added} items")
                # Only mark sufficient if we actually recovered meaningful context
                if not gaps and added >= 2:
                    sufficient = True

        ctx["pre_enriched"]  = True
        ctx["sufficient"]    = sufficient
        ctx["gaps"]          = gaps
        ctx["enriched_with"] = enriched_with

        if not sufficient and not enriched_with:
            logger.info(
                "ContextManager: task may start with insufficient context (gaps: %s)", gaps
            )

        return task_packet

    # ------------------------------------------------------------------
    # Public static: on-demand fetch (used by context_fetch tool)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch(
        query:      str,
        project:    str,
        base_path:  Path,
        context:    dict,
        top_k:      int  = 5,
        include_files: bool = True,
        max_chars_per_file: int = _SNIPPET_MAX_CHARS,
    ) -> dict:
        """
        RAG fetch usable by the context_fetch tool.
        Returns:
          mt_entries : list[{content, score}]
          files      : list[{path, summary, score, snippet}]
          text       : pre-formatted string for LLM injection
        """
        # Embed the query
        query_vec = _embed(query, context)

        # MT search
        mt_entries: list[dict] = []
        mt_path = base_path / "memory" / "mt.json"
        if mt_path.exists():
            try:
                all_mt = json.loads(mt_path.read_text())
                if isinstance(all_mt, list):
                    active = [e for e in all_mt if e.get("status", "active") == "active"]
                    if query_vec:
                        scored = [
                            (e, _cosine(query_vec, e.get("embedding", [])))
                            for e in active if e.get("embedding")
                        ]
                        scored.sort(key=lambda x: x[1], reverse=True)
                        mt_entries = [
                            {"content": e.get("content", ""), "score": round(s, 3)}
                            for e, s in scored[:top_k]
                            if s > _MT_WEAK_THRESHOLD
                        ]
                    else:
                        mt_entries = [
                            {"content": e.get("content", ""), "score": 0.0}
                            for e in active[:top_k]
                        ]
            except Exception as e:
                logger.warning("context_fetch MT search failed: %s", e)

        # File index search
        files: list[dict] = []
        if include_files:
            dev_idx_path = base_path / "projects" / project / "dev" / "index.json"
            if dev_idx_path.exists() and query_vec:
                try:
                    dev_index = json.loads(dev_idx_path.read_text())
                    scored_files = [
                        (path, entry, _cosine(query_vec, entry.get("embedding", [])))
                        for path, entry in dev_index.items()
                        if entry.get("embedding")
                    ]
                    scored_files.sort(key=lambda x: x[2], reverse=True)
                    for fpath, fentry, fscore in scored_files[:top_k]:
                        if fscore < _FILE_MIN_THRESHOLD:
                            break
                        snippet = ""
                        try:
                            full = Path(fpath)
                            if full.exists():
                                snippet = full.read_text(encoding="utf-8", errors="replace")
                                snippet = snippet[:max_chars_per_file]
                        except Exception:
                            pass
                        files.append({
                            "path":    fpath,
                            "summary": fentry.get("summary", ""),
                            "score":   round(fscore, 3),
                            "snippet": snippet,
                        })
                except Exception as e:
                    logger.warning("context_fetch file search failed: %s", e)

        # Build pre-formatted text block
        lines: list[str] = [f"## Context retrieved for: {query[:100]}"]

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

        return {
            "mt_entries": mt_entries,
            "files":      files,
            "text":       "\n".join(lines),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_gaps(
        self, objective: str, ctx: dict
    ) -> tuple[list[str], list[str]]:
        """
        Cheap LLM call to spot what context is missing.
        Returns (gaps, search_queries).
        """
        # Summarise existing context for the LLM
        mt_preview = "; ".join(
            (e["content"] if isinstance(e, dict) else str(e))[:80]
            for e in ctx.get("mt_retrieved", [])[:3]
        )
        files_preview = "; ".join(
            (e.get("summary", e.get("path", ""))[:60] if isinstance(e, dict) else str(e))
            for e in ctx.get("relevant_files", [])[:3]
        )
        context_summary = (
            f"MT knowledge: {mt_preview or '(none)'}\n"
            f"Files available: {files_preview or '(none)'}\n"
            f"Session summary: {ctx.get('st_summary', '')[:200] or '(none)'}"
        )

        system = (
            "You are a context assessor. Given a task objective and currently available "
            "context, identify what specific knowledge or file content is MISSING that "
            "the agent would clearly need.\n"
            "Be concise. Only flag genuine gaps, not nice-to-haves.\n"
            "Output ONLY JSON: "
            '{"sufficient": true|false, "gaps": ["short description"], '
            '"queries": ["search query to fill each gap"]}'
        )
        resp = self._tools.llm_call(
            pipeline  = "context_assessment",
            system    = system,
            messages  = [{
                "role":    "user",
                "content": f"Objective: {objective}\n\nAvailable context:\n{context_summary}",
            }],
            max_tokens  = 256,
            temperature = 0.0,
        )
        raw = resp.get("content", "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data    = json.loads(raw)
        gaps    = data.get("gaps", [])
        queries = data.get("queries", [])
        return gaps, queries

    def _enrich_for_query(self, query: str, project: str, ctx: dict) -> int:
        """
        Fetch additional MT + file context for a query and merge into ctx.
        Returns number of new items added.
        """
        added = 0
        context_stub = {
            "base_path":   str(self._base_path),
            "project":     project,
            "instance_id": "",
            "config":      {},
        }
        result = ContextManager.fetch(
            query       = query,
            project     = project,
            base_path   = self._base_path,
            context     = context_stub,
            top_k       = _GAP_MT_TOP_K,
            include_files = True,
        )

        # Merge MT entries — deduplicate by content
        existing_contents = {
            (e["content"] if isinstance(e, dict) else e)
            for e in ctx.get("mt_retrieved", [])
        }
        for entry in result["mt_entries"]:
            if entry["content"] not in existing_contents:
                ctx.setdefault("mt_retrieved", []).append(entry)
                existing_contents.add(entry["content"])
                added += 1

        # Merge file entries — deduplicate by path
        existing_paths = {
            (e["path"] if isinstance(e, dict) else "")
            for e in ctx.get("relevant_files", [])
        }
        for fentry in result["files"][:_GAP_FILE_TOP_K]:
            if fentry["path"] not in existing_paths:
                ctx.setdefault("relevant_files", []).append(fentry)
                existing_paths.add(fentry["path"])
                added += 1

        return added


# ---------------------------------------------------------------------------
# Shared low-level helpers (no I/O side effects)
# ---------------------------------------------------------------------------

def _embed(text: str, context: dict) -> list[float] | None:
    """Import and call embed_text.run() to get a vector."""
    try:
        base_path = Path(context["base_path"])
        impl_path = base_path / "tools" / "impl" / "embed_text.py"
        spec = importlib.util.spec_from_file_location("embed_text", str(impl_path))
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        result = mod.run({"text": text}, context)
        return result.get("embedding")
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
