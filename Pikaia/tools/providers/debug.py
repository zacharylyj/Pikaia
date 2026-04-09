"""
Debug Provider Adapter
----------------------
A zero-network mock for offline/CI testing of the full pipeline.

Never makes HTTP calls or reads API keys. Detects which pipeline is being
served by inspecting the system prompt and returns a correctly-shaped
canned JSON or text response so every downstream parser succeeds.

Pipelines handled
-----------------
  classification       → {"type": "task"}
  skillsmith_draft     → valid skill JSON schema
  skillsmith_eval      → {"score": 0.85, "pass": true, "feedback": "..."}
  ack_validation       → valid ack object (task_id extracted from packet)
  mt_judge             → {"persist": false, "content": ""}
  context_assessment   → {"sufficient": true, "gaps": [], "queries": []}
  task_planning        → JSON array of step strings
  compression          → plain-text summary string
  file_indexing        → plain-text file summary string
  everything else      → generic completion string

Usage
-----
  python main.py --project default --debug
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class Adapter(BaseAdapter):

    def build_request(
        self,
        system:      str,
        messages:    list[dict],
        max_tokens:  int  = 1024,
        temperature: float | None = None,
        tools:       list[dict] | None = None,
    ) -> dict[str, Any]:
        # Stash system + messages so call() can inspect them
        return {"system": system, "messages": messages}

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        system   = request.get("system", "")
        messages = request.get("messages", [])
        content  = _canned_response(system, messages)
        logger.debug("[debug adapter] system snippet: %s…", system[:60])
        return {"_content": content}

    def parse_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        content = raw.get("_content", "")
        return self._standard_response(
            content        = content,
            tokens_in      = 10,
            tokens_out     = max(1, len(content.split())),
            stop_reason    = "end_turn",
            content_blocks = [{"type": "text", "text": content}],
        )

    def validate_key(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Response factory
# ---------------------------------------------------------------------------

def _last_user_text(messages: list[dict]) -> str:
    """Return the text content of the last user message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                # Anthropic-style content blocks
                return " ".join(
                    b.get("text", "") for b in c if isinstance(b, dict)
                )
            return str(c)
    return ""


def _canned_response(system: str, messages: list[dict]) -> str:
    s = system.lower()
    user_text = _last_user_text(messages)

    # ── Intent classification ──────────────────────────────────────────────
    if "intent classifier" in s or "classify the user message" in s:
        return json.dumps({"type": "task"})

    # ── SkillSmith draft / refine (must check before generic "skill") ──────
    if "skillsmith" in s and "evaluator" not in s:
        # For refine: try to preserve the existing draft's identity fields
        # by parsing them from the user message; fall back to defaults.
        base_name = "debug_skill"
        base_desc = "A debug skill created without a real LLM."
        try:
            # refine messages contain "Current draft:\n{json}"
            if "current draft" in user_text.lower():
                idx = user_text.lower().index("current draft")
                snippet = user_text[idx:].split("\n", 2)
                if len(snippet) >= 2:
                    draft_json = "\n".join(snippet[1:]).strip()
                    existing = json.loads(draft_json.split("Evaluator")[0])
                    base_name = existing.get("name", base_name)
                    base_desc = existing.get("description", base_desc)
        except Exception:
            pass
        return json.dumps({
            "name":           base_name,
            "description":    base_desc,
            "tier":           2,
            "tags":           ["debug"],
            "tools_required": ["llm_call", "file_read", "file_write"],
            "template":       "Complete the task: {{objective}}",
        })

    # ── SkillSmith evaluator ───────────────────────────────────────────────
    if "skill evaluator" in s or "score how well" in s:
        return json.dumps({
            "score":    0.85,
            "feedback": "[debug] Skill draft accepted by debug evaluator.",
            "pass":     True,
        })

    # ── Ack validation ─────────────────────────────────────────────────────
    if "agent receiving a task" in s or "json ack" in s or "planned_steps" in s:
        task_id     = "debug_task"
        restatement = "Complete the requested task."
        try:
            packet      = json.loads(user_text)
            task_id     = packet.get("task_id", task_id)
            restatement = packet.get("objective", restatement)[:120]
        except Exception:
            pass
        return json.dumps({
            "task_id":        task_id,
            "restatement":    restatement,
            "done_looks_like": ["task output returned", "no errors raised"],
            "ambiguities":    [],
            "planned_steps":  [
                "step 1: analyse the request",
                "step 2: execute the main task",
                "step 3: return result",
            ],
            "confidence":     0.90,
        })

    # ── MT judge ───────────────────────────────────────────────────────────
    if "durable knowledge" in s or ("persist" in s and "memory" in s):
        return json.dumps({"persist": False, "content": ""})

    # ── Context / gap assessment ───────────────────────────────────────────
    if "gap" in s and ("identif" in s or "assess" in s):
        return json.dumps({"sufficient": True, "gaps": [], "queries": []})

    # ── Task planner / decompose ───────────────────────────────────────────
    if "task planner" in s or "break the objective" in s:
        return json.dumps([
            "Analyse the request",
            "Execute the main task",
            "Return and verify the result",
        ])

    # ── Compression ────────────────────────────────────────────────────────
    if "compress" in s or "concise summary" in s:
        return "[debug] Prior context compressed (no real LLM used)."

    # ── File indexing ───────────────────────────────────────────────────────
    if "summarise this file" in s or ("summarise" in s and "file" in s):
        return "[debug] File summary unavailable in debug mode."

    # ── Fallback: generic completion ────────────────────────────────────────
    snippet = user_text[:80].replace("\n", " ") if user_text else "(no input)"
    return (
        f"[debug mode] Task acknowledged. "
        f"No real LLM call was made. Input: {snippet}"
    )
