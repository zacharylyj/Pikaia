"""
Orchestrator
------------
Single orchestrator that runs the full turn loop defined in the project spec.

Responsibilities (per spec Phase 5 — ORCHESTRATOR):
  1. Receive message from interface
  2. Build context  (LT + MT + CT + ST + file layers)
  3. Understand intent  (classify → clarify if needed → meta-commands)
  4. Skill pick  (embed → cosine → validate → SkillSmith on miss)
  5. Dispatch agent  (write CT flag, create worker dir, spawn)
  6. Monitor  (poll state.json, enforce budgets + timeouts)
  7. Receive result  (promote / review / retry / escalate)
  8. Post-process  (ST update + History append + MT judge + LT)

The orchestrator does NOT implement agents, memory layers, or tools directly.
It coordinates them through well-defined interfaces (llm_call, memory_read,
memory_write, embed_text, etc.) so each subsystem can be swapped later.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

IntentType   = Literal["task", "question", "meta-command", "ambiguous"]
AgentMode    = Literal["isolated", "team"]
AgentStatus  = Literal["running", "done", "failed", "killed"]
CTStatus     = Literal["open", "done", "failed", "pending_approval"]
CTType       = Literal["pending", "status", "note", "skill_approval"]

TIER_PIPELINES = {
    1: "code_generation",   # atomic
    2: "orchestration",     # composite
    3: "task_planning",     # sub-agent loop
    4: "council_agent",     # council (per agent); synthesis separate
}

TIER_TIMEOUTS = {1: 60, 2: 120, 3: 300, 4: 600}
TIER_BUDGETS  = {1: 1000, 2: 2000, 3: 6000, 4: 10000}


# ---------------------------------------------------------------------------
# Tool interface stubs
# ---------------------------------------------------------------------------
# The orchestrator calls tools through these thin wrappers.
# Replace the bodies with real ToolRegistry dispatch in phase 2.

class ToolError(Exception):
    pass


def _tool_stub(name: str, params: dict[str, Any]) -> Any:
    """Fallback dispatch — raised when no real registry is wired up."""
    raise NotImplementedError(f"Tool '{name}' not yet wired up. params={params}")


class Tools:
    """
    Thin façade the orchestrator uses to call tools.
    Each method maps 1-to-1 with a tool in tools.json.
    Pass a real ToolRegistry.dispatch as the `dispatch` argument to wire it up.
    """

    def __init__(self, dispatch: Callable[[str, dict], Any] | None = None):
        self._dispatch = dispatch or _tool_stub

    @staticmethod
    def _unwrap(result: Any) -> Any:
        """
        Extract the payload from a ToolResult envelope.
        If result is a ToolResult dict ({success, data, error}), returns data.
        Otherwise returns result unchanged (backward-compat with raw tool output).
        """
        if isinstance(result, dict) and "success" in result and "data" in result:
            return result["data"]
        return result

    def llm_call(
        self,
        pipeline: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        return self._unwrap(self._dispatch("llm_call", dict(
            pipeline=pipeline, system=system, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )))

    def embed_text(self, text: str) -> list[float]:
        result = self._unwrap(self._dispatch("embed_text", {"text": text}))
        # embed_text.run() returns {"embedding": [...], "dim": int, "model": str}
        if isinstance(result, dict):
            return result.get("embedding", [])
        return result  # already a list (unlikely but safe)

    def memory_read(
        self,
        layer: str,
        query: str = "",
        top_k: int = 5,
        project: str = "",
        instance_id: str = "",
    ) -> list[dict]:
        return self._unwrap(self._dispatch("memory_read", dict(
            layer=layer, query=query, top_k=top_k,
            project=project, instance_id=instance_id,
        )))

    def memory_write(self, layer: str, entry: dict, project: str = "", instance_id: str = "") -> None:
        self._dispatch("memory_write", dict(
            layer=layer, entry=entry, project=project, instance_id=instance_id,
        ))

    def file_read(self, path: str) -> str:
        result = self._unwrap(self._dispatch("file_read", {"path": path}))
        # file_read.run() returns {"content": str, "path": str, "size_bytes": int}
        if isinstance(result, dict):
            return result.get("content", "")
        return result  # already a string

    def file_write(self, path: str, content: str) -> None:
        self._dispatch("file_write", {"path": path, "content": content})

    def file_delete(self, path: str) -> None:
        self._dispatch("file_delete", {"path": path})

    def file_move(self, src: str, dst: str) -> None:
        self._dispatch("file_move", {"src": src, "dst": dst})

    def cli_output(self, content: str, type_: str = "response") -> None:
        self._dispatch("cli_output", {"content": content, "type": type_})

    def skill_read(self, skill_id: str) -> dict:
        return self._unwrap(self._dispatch("skill_read", {"skill_id": skill_id}))

    def skill_write(self, skill: dict, template_content: str) -> None:
        self._dispatch("skill_write", {"skill": skill, "template_content": template_content})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """
    Mirrors config.json from the spec.
    Loaded at startup; project config.json overlays on top.
    """
    default_model:          str   = "claude-sonnet-4-6"
    compression_model:      str   = "claude-haiku-4-5-20251001"
    skill_match_threshold:  float = 0.75
    promote_threshold:      float = 0.80
    ack_confidence_min:     float = 0.80
    ack_max_rounds:         int   = 2
    st_max_messages:        int   = 20
    mt_top_k:               int   = 5
    history_rag_top_k:      int   = 3
    file_summary_top_k:     int   = 3
    max_files_per_task:     int   = 5
    retry_limit:            int   = 3
    skillsmith_dry_runs:    int   = 3
    skillsmith_pass_score:  float = 0.80
    embedding_dim:          int   = 1536
    poll_interval_secs:     float = 3.0

    pipelines: dict[str, str] = field(default_factory=lambda: {
        "orchestration":      "claude-sonnet-4-6",
        "task_planning":      "claude-sonnet-4-6",
        "research":           "claude-opus-4-6",
        "council_agent":      "claude-opus-4-6",
        "council_synthesis":  "claude-opus-4-6",
        "code_generation":    "claude-sonnet-4-6",
        "compression":        "claude-haiku-4-5-20251001",
        "classification":     "claude-haiku-4-5-20251001",
        "file_indexing":      "claude-haiku-4-5-20251001",
        "mt_judge":           "claude-haiku-4-5-20251001",
        "skillsmith_draft":   "claude-sonnet-4-6",
        "skillsmith_eval":    "claude-sonnet-4-6",
        "ack_validation":     "claude-haiku-4-5-20251001",
        "context_assessment": "claude-haiku-4-5-20251001",
    })

    @classmethod
    def from_json(cls, global_path: str, project_path: str | None = None) -> "OrchestratorConfig":
        """Load global config, then overlay project-level overrides."""
        cfg = cls()
        if os.path.exists(global_path):
            with open(global_path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        if project_path and os.path.exists(project_path):
            with open(project_path) as f:
                proj = json.load(f)
            # Pipeline overrides
            if "pipelines" in proj:
                cfg.pipelines.update(proj["pipelines"])
            # Any other scalar overrides
            for k, v in proj.items():
                if k != "pipelines" and hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Context bundle
# ---------------------------------------------------------------------------

@dataclass
class TurnContext:
    """Everything assembled in step 2 and passed downstream."""
    lt_entries:       list[dict]       = field(default_factory=list)
    mt_entries:       list[dict]       = field(default_factory=list)
    ct_active:        list[dict]       = field(default_factory=list)
    st_summary:       str              = ""
    st_window:        list[dict]       = field(default_factory=list)
    project_index:    dict             = field(default_factory=dict)
    relevant_files:   list[dict]       = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Serialise the full context into a system prompt string."""
        parts: list[str] = []

        if self.lt_entries:
            lt_text = "\n".join(f"- {e['content']}" for e in self.lt_entries)
            parts.append(f"## Long-term preferences\n{lt_text}")

        if self.mt_entries:
            mt_text = "\n".join(f"- {e['content']}" for e in self.mt_entries)
            parts.append(f"## Relevant knowledge\n{mt_text}")

        if self.ct_active:
            ct_lines = []
            for f in self.ct_active:
                elapsed = ""
                if f.get("opened_at"):
                    try:
                        opened = datetime.fromisoformat(f["opened_at"])
                        secs = int((datetime.now(timezone.utc) - opened).total_seconds())
                        elapsed = f" ({secs}s elapsed)"
                    except Exception:
                        pass
                ct_lines.append(f"- [{f['status']}] {f['description']}{elapsed}")
            parts.append("## Current state\n" + "\n".join(ct_lines))

        if self.st_summary:
            parts.append(f"## Session summary\n{self.st_summary}")

        if self.project_index:
            parts.append(f"## Project file map\n{json.dumps(self.project_index, indent=2)}")

        if self.relevant_files:
            rf_lines = []
            for rf in self.relevant_files:
                rf_lines.append(f"- {rf['path']} (score {rf.get('score', '?')}): {rf.get('summary', '')}")
            parts.append("## Relevant files\n" + "\n".join(rf_lines))

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Agent dispatch record
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    agent_id:     str
    task_id:      str
    project:      str
    instance_id:  str
    skill_id:     str
    pipeline:     str
    tier:         int
    mode:         AgentMode
    team_id:      str | None
    timeout_secs: int
    token_budget: int
    spawned_at:   float = field(default_factory=time.time)
    status:       AgentStatus = "running"
    worker_dir:   str = ""

    def elapsed(self) -> float:
        return time.time() - self.spawned_at

    def meta_dict(self) -> dict:
        return {
            "agent_id":     self.agent_id,
            "task_id":      self.task_id,
            "project":      self.project,
            "instance_id":  self.instance_id,
            "skill_id":     self.skill_id,
            "pipeline":     self.pipeline,
            "tier":         self.tier,          # required by AgentRunner / BaseAgent
            "mode":         self.mode,
            "team_id":      self.team_id,
            "spawned_at":   self.spawned_at,
            "timeout_secs": self.timeout_secs,
            "token_budget": self.token_budget,
            "tokens_used":  0,
            "status":       self.status,
            "worker_dir":   self.worker_dir,    # avoids fallback path recomputation
        }


# ---------------------------------------------------------------------------
# Skill pick result
# ---------------------------------------------------------------------------

@dataclass
class SkillMatch:
    skill_id:   str
    name:       str
    tier:       int
    score:      float
    tools_ok:   bool
    pipeline:   str


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Single orchestrator. One instance per running session.

    Usage
    -----
    orch = Orchestrator(
        project="medical-notes",
        instance_id="inst_abc",
        base_path="/agent",
        tools=Tools(real_dispatch),
        config=OrchestratorConfig.from_json("config.json", "projects/medical-notes/config.json"),
    )
    response = orch.turn("summarise WHO antimicrobial guidelines")
    """

    def __init__(
        self,
        project:     str,
        instance_id: str,
        base_path:   str,
        tools:       Tools,
        config:      OrchestratorConfig | None = None,
        on_status:   Callable[[str], None] | None = None,
    ) -> None:
        self.project     = project
        self.instance_id = instance_id
        self.base_path   = Path(base_path)
        self.tools       = tools
        self.config      = config or OrchestratorConfig()
        self.on_status   = on_status or (lambda msg: logger.info("[status] %s", msg))

        # Active agent registry  agent_id → AgentRecord
        self._agents: dict[str, AgentRecord] = {}
        self._monitor_thread: threading.Thread | None = None
        self._running = False

        # Context manager — lazy import to avoid circular deps at module load
        self._ctx_manager: Any | None = None
        self._start_monitor()

    # ------------------------------------------------------------------
    # Public: main turn entry point
    # ------------------------------------------------------------------

    def turn(self, message: str) -> str:
        """
        Full turn loop. Returns the final response string.
        Called by the CLI (phase 6) for every user message.
        """
        self._status(f"Building context...")
        ctx = self._build_context(message)

        self._status("Classifying intent...")
        intent, intent_type = self._understand_intent(message, ctx)

        # Meta-commands handled immediately, no agent needed
        if intent_type == "meta-command":
            return self._handle_meta_command(intent, message, ctx)

        # Ambiguous — ask one clarifying question
        if intent_type == "ambiguous":
            return intent  # intent holds the clarifying question

        # Pick a skill
        self._status("Picking skill...")
        match = self._skill_pick(message)
        if match is None:
            self._status("No skill found — triggering SkillSmith...")
            self._trigger_skillsmith(message, ctx)
            return "I don't have a skill for that yet. SkillSmith is drafting one — I'll let you know when it's ready for approval."

        self._status(f"Skill: {match.name} (tier {match.tier}, score {match.score:.2f})")

        # Dispatch
        record = self._dispatch(message, match, ctx)

        # Monitor until done (blocking for CLI; async-friendly via on_status)
        result = self._await_agent(record)

        # Post-process
        self._post_process(message, result, ctx)

        return result.get("output", "")

    # ------------------------------------------------------------------
    # Step 2 — Build context
    # ------------------------------------------------------------------

    def _build_context(self, message: str) -> TurnContext:
        ctx = TurnContext()

        # LT — always full; overlay preferences.json on top
        try:
            ctx.lt_entries = self.tools.memory_read("lt", project=self.project)
        except Exception as e:
            logger.warning("LT read failed: %s", e)

        # preferences.json overlay: merge preference keys into LT view
        try:
            prefs_path = self._project_path("preferences.json")
            if prefs_path.exists():
                prefs: dict = json.loads(prefs_path.read_text())
                for k, v in prefs.items():
                    # Inject as synthetic LT entries so context builder sees them
                    ctx.lt_entries.append({
                        "id":       f"pref_{k}",
                        "content":  f"{k}: {v}",
                        "category": "preference",
                        "source":   "preferences.json",
                    })
        except Exception as e:
            logger.warning("Preferences overlay failed: %s", e)

        # ST — load once; used for both MT query and ST context
        st: dict = {}
        try:
            st = self._load_st()
            ctx.st_summary = st.get("summary", "")
            ctx.st_window  = st.get("window", [])
        except Exception as e:
            logger.warning("ST read failed: %s", e)

        # MT — RAG on (ST summary + message)
        try:
            mt_query = f"{st.get('summary', '')} {message}"
            ctx.mt_entries = self.tools.memory_read(
                "mt", query=mt_query, top_k=self.config.mt_top_k, project=self.project
            )
        except Exception as e:
            logger.warning("MT read failed: %s", e)

        # CT — active/open flags only
        try:
            all_ct = self.tools.memory_read("ct", project=self.project)
            ctx.ct_active = [f for f in all_ct if f.get("status") == "open"]
        except Exception as e:
            logger.warning("CT read failed: %s", e)

        # File layer 1 — always injected
        try:
            idx_path = self._project_path("file_index.json")
            if idx_path.exists():
                ctx.project_index = json.loads(idx_path.read_text())
        except Exception as e:
            logger.warning("File index read failed: %s", e)

        # File layer 2 — RAG on task
        try:
            dev_idx_path = self._project_path("dev", "index.json")
            if dev_idx_path.exists():
                dev_index = json.loads(dev_idx_path.read_text())
                query_vec = self.tools.embed_text(message)
                scored = self._cosine_top_k(query_vec, dev_index, self.config.file_summary_top_k)
                ctx.relevant_files = scored
        except Exception as e:
            logger.warning("File layer 2 failed: %s", e)

        return ctx

    # ------------------------------------------------------------------
    # Step 3 — Understand intent
    # ------------------------------------------------------------------

    def _understand_intent(self, message: str, ctx: TurnContext) -> tuple[str, IntentType]:
        """
        Returns (intent_text, intent_type).
        intent_text is the clarifying question if ambiguous,
        or the meta-command key if meta-command, else the original message.
        """
        system = (
            "You are an intent classifier. "
            "Classify the user message as one of: task, question, meta-command, ambiguous.\n"
            "Meta-commands: 'forget X', 'new goal: X', 'remember X', 'promote X'.\n"
            "If ambiguous, output ONE clarifying question.\n"
            "Respond in JSON: {\"type\": \"...\", \"clarification\": \"...\" | null, "
            "\"meta_key\": \"forget|new_goal|remember|promote\" | null}"
        )
        try:
            resp = self.tools.llm_call(
                pipeline="classification",
                system=system,
                messages=[{"role": "user", "content": message}],
                max_tokens=256,
                temperature=0.0,
            )
            data = json.loads(_strip_json_fences(resp.get("content", "")))
            intent_type: IntentType = data.get("type", "task")
            if intent_type == "ambiguous":
                return data.get("clarification", "Could you clarify?"), "ambiguous"
            if intent_type == "meta-command":
                return data.get("meta_key", ""), "meta-command"
            return message, intent_type
        except Exception as e:
            logger.warning("Intent classification failed (%s) — defaulting to task", e)
            return message, "task"

    def _handle_meta_command(self, meta_key: str, message: str, ctx: TurnContext) -> str:
        """Execute meta-commands immediately without dispatching an agent."""
        if meta_key == "remember":
            # Extract what to remember and append to LT
            entry = {
                "id":         str(uuid.uuid4()),
                "content":    message,
                "category":   "preference",
                "created_at": _now_iso(),
            }
            self.tools.memory_write("lt", entry, project=self.project)
            return "Got it — I've saved that to long-term memory."

        if meta_key == "forget":
            # Mark matching MT entries as outdated
            try:
                mt_entries = self.tools.memory_read("mt", query=message, top_k=3, project=self.project)
                for e in mt_entries:
                    e["status"] = "outdated"
                    self.tools.memory_write("mt", e, project=self.project)
            except Exception as ex:
                logger.warning("Forget failed: %s", ex)
            return "Done — I've marked that knowledge as outdated."

        if meta_key == "new_goal":
            entry = {
                "id":          str(uuid.uuid4()),
                "type":        "status",
                "description": message,
                "task_id":     None,
                "agent_id":    None,
                "instance_id": self.instance_id,
                "opened_at":   _now_iso(),
                "status":      "open",
                "closed_at":   None,
            }
            self.tools.memory_write("ct", entry, project=self.project)
            return "New goal noted in current state."

        if meta_key == "promote":
            # Extract file path from message and promote it
            # Real implementation would parse the path properly
            return "Promotion via 'promote X' — please specify the full worker path."

        return f"Unknown meta-command: {meta_key}"

    # ------------------------------------------------------------------
    # Step 4 — Skill pick
    # ------------------------------------------------------------------

    def _skill_pick(self, message: str) -> SkillMatch | None:
        """
        1. Embed intent
        2. Cosine against all active skill embeddings
        3. Validate tools_required
        4. Return best match or None (→ SkillSmith)
        """
        try:
            skills_path = self.base_path / "skills" / "skills.json"
            if not skills_path.exists():
                return None

            skills: list[dict] = json.loads(skills_path.read_text())
            active = [s for s in skills if s.get("active")]
            if not active:
                return None

            query_vec = self.tools.embed_text(message)

            # Auto-compute and cache embeddings for skills that are missing one
            skills_dirty = False
            for skill in active:
                if not skill.get("embedding"):
                    try:
                        emb = self.tools.embed_text(
                            skill.get("description", skill.get("name", ""))
                        )
                        skill["embedding"] = emb
                        skills_dirty = True
                    except Exception as e:
                        logger.warning("Skill embed failed for %s: %s", skill.get("skill_id"), e)

            if skills_dirty:
                try:
                    _atomic_write_json(skills_path, skills)
                except Exception as e:
                    logger.warning("Could not save updated skill embeddings: %s", e)

            scored: list[tuple[float, dict]] = []
            for skill in active:
                emb = skill.get("embedding")
                if not emb:
                    continue
                score = _cosine(query_vec, emb)
                scored.append((score, skill))

            if not scored:
                return None

            scored.sort(key=lambda x: (-x[0], x[1].get("tier", 99)))
            top_score, top_skill = scored[0]

            if top_score < self.config.skill_match_threshold:
                return None

            # Validate tools_required
            tools_ok = self._validate_tools_required(top_skill.get("tools_required", []))

            pipeline = TIER_PIPELINES.get(top_skill.get("tier", 2), "orchestration")

            return SkillMatch(
                skill_id  = top_skill["skill_id"],
                name      = top_skill["name"],
                tier      = top_skill.get("tier", 2),
                score     = top_score,
                tools_ok  = tools_ok,
                pipeline  = pipeline,
            )
        except Exception as e:
            logger.warning("Skill pick failed: %s", e)
            return None

    def _validate_tools_required(self, tools_required: list[str]) -> bool:
        try:
            tools_path = self.base_path / "tools" / "tools.json"
            if not tools_path.exists():
                return True  # can't validate, assume ok
            registry: list[dict] = json.loads(tools_path.read_text())
            enabled = {t["tool_id"] for t in registry if t.get("enabled")}
            return all(t in enabled for t in tools_required)
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Step 5 — Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, message: str, match: SkillMatch, ctx: TurnContext) -> AgentRecord:
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        task_id  = f"task_{uuid.uuid4().hex[:8]}"
        tier     = match.tier
        # Resolve pipeline name → model_id through config so agents can load
        # the correct adapter without needing their own copy of the pipeline map.
        # In debug mode this becomes "debug-model"; in normal mode it becomes
        # a model_id like "claude-sonnet-4-6" (still valid for _load_adapter).
        pipeline = self.config.pipelines.get(match.pipeline, match.pipeline)

        # Council mode for tier 4
        mode:    AgentMode = "team" if tier == 4 else "isolated"
        team_id: str | None = f"council_{task_id}" if tier == 4 else None

        timeout = TIER_TIMEOUTS.get(tier, 180)
        budget  = TIER_BUDGETS.get(tier, 4000)

        # Open CT pending flag
        ct_entry = {
            "id":          str(uuid.uuid4()),
            "type":        "pending",
            "description": message[:120],
            "task_id":     task_id,
            "agent_id":    agent_id,
            "instance_id": self.instance_id,
            "skill_id":    match.skill_id,
            "webui_action": None,
            "opened_at":   _now_iso(),
            "status":      "open",
            "closed_at":   None,
        }
        try:
            self.tools.memory_write("ct", ct_entry, project=self.project)
        except Exception as e:
            logger.warning("CT write failed: %s", e)

        # Create worker directory
        worker_dir = self._project_path("worker", agent_id)
        worker_dir.mkdir(parents=True, exist_ok=True)

        record = AgentRecord(
            agent_id     = agent_id,
            task_id      = task_id,
            project      = self.project,
            instance_id  = self.instance_id,
            skill_id     = match.skill_id,
            pipeline     = pipeline,
            tier         = tier,
            mode         = mode,
            team_id      = team_id,
            timeout_secs = timeout,
            token_budget = budget,
            worker_dir   = str(worker_dir),
        )

        # Write meta.json
        meta_path = worker_dir / "meta.json"
        meta_path.write_text(json.dumps(record.meta_dict(), indent=2))

        # Build task packet
        task_packet = self._build_task_packet(message, match, ctx, record)

        # Pre-dispatch context assessment — enrich if gaps found
        try:
            task_packet = self._get_ctx_manager().assess(task_packet, self.project)
            assessment  = task_packet.get("context", {})
            if not assessment.get("sufficient", True):
                self._status(
                    f"Context gaps detected: {assessment.get('gaps', [])} — "
                    f"enriched with: {assessment.get('enriched_with', [])}"
                )
        except Exception as e:
            logger.warning("Context assessment failed (proceeding): %s", e)

        # Write task packet for agent to read
        (worker_dir / "task.json").write_text(json.dumps(task_packet, indent=2))

        # Register and spawn agent thread
        self._agents[agent_id] = record
        self._spawn_agent(record, task_packet)

        self._status(f"Dispatched {agent_id} (tier {tier}, pipeline: {pipeline}, timeout: {timeout}s, budget: {budget}t)")
        return record

    def _build_task_packet(
        self,
        message: str,
        match: SkillMatch,
        ctx: TurnContext,
        record: AgentRecord,
    ) -> dict:
        """Build the Stage 1 task packet per the handshake spec."""
        try:
            skill = self.tools.skill_read(match.skill_id)
        except Exception:
            skill = {"name": match.name, "tools_required": []}

        tools_allowed = list(skill.get("tools_required", []))
        # Baseline tools always available to every agent
        for t in ("llm_call", "file_read", "file_write", "embed_text", "context_fetch"):
            if t not in tools_allowed:
                tools_allowed.append(t)

        # Preserve MT scores so ContextManager can assess quality
        mt_retrieved = [
            {"content": e.get("content", ""), "score": round(e.get("score", 0.0), 3)}
            if isinstance(e, dict) else {"content": str(e), "score": 0.0}
            for e in ctx.mt_entries
        ]

        return {
            "task_id":          record.task_id,
            "agent_id":         record.agent_id,
            "objective":        message,
            "success_criteria": [],
            "constraints":      [],
            "context": {
                "lt_summary":     " | ".join(e.get("content", "") for e in ctx.lt_entries[:3]),
                "mt_retrieved":   mt_retrieved,
                "ct_active":      ctx.ct_active,
                "st_summary":     ctx.st_summary,
                "project_index":  ctx.project_index,
                "relevant_files": ctx.relevant_files,
            },
            "skill":         match.name,
            "skill_id":      match.skill_id,
            "tier":          record.tier,
            "pipeline":      record.pipeline,
            "tools_allowed": tools_allowed,
            "token_budget":  record.token_budget,
            "timeout_secs":  record.timeout_secs,
            "file_budget": {
                "max_files":     self.config.max_files_per_task,
                "files_fetched": 0,
            },
        }

    # ------------------------------------------------------------------
    # Step 5b — Agent spawn + ack
    # ------------------------------------------------------------------

    def _spawn_agent(self, record: AgentRecord, task_packet: dict) -> None:
        """
        Spawns the agent in a daemon thread using the real AgentRunner.
        AgentRunner handles the ack/state/result lifecycle internally.
        """
        import sys
        # Ensure Pikaia package is importable from this repo layout
        _pikaia_dir = str(self.base_path)
        if _pikaia_dir not in sys.path:
            sys.path.insert(0, _pikaia_dir)

        from agent import AgentRunner
        record_dict = record.meta_dict()
        base_path   = str(self.base_path)

        def _run() -> None:
            worker = Path(record.worker_dir)
            try:
                # ----------------------------------------------------------
                # Stage 2 — ack handshake with retry (up to ack_max_rounds)
                # ----------------------------------------------------------
                ack    = {}
                ok     = False
                reason = "not started"
                attempt_packet = task_packet

                for attempt in range(self.config.ack_max_rounds + 1):
                    ack = self._generate_ack(record, attempt_packet)
                    (worker / "ack.json").write_text(json.dumps(ack, indent=2))
                    ok, reason = self._validate_ack(ack, attempt_packet, record)
                    if ok:
                        break
                    self._status(
                        f"Ack round {attempt + 1}/{self.config.ack_max_rounds + 1} "
                        f"invalid ({reason}) — retrying"
                    )
                    # Inject feedback into next attempt so the LLM can resolve it
                    feedback = attempt_packet.get("_ack_feedback", [])
                    attempt_packet = dict(task_packet)
                    attempt_packet["_ack_feedback"] = feedback + [reason]

                if not ok:
                    logger.warning(
                        "Ack failed after %d rounds for %s: %s",
                        self.config.ack_max_rounds + 1, record.agent_id, reason
                    )
                    self._mark_agent_done(record, status="failed",
                                          output=f"Ack failed: {reason}")
                    return

                self._status(
                    f"Ack received (confidence: {ack.get('confidence', '?')}, "
                    f"steps: {len(ack.get('planned_steps', []))}) — launching agent"
                )

                # ----------------------------------------------------------
                # Stage 3 — write initial state then hand off to AgentRunner
                # ----------------------------------------------------------
                state = {
                    "task_id":      record.task_id,
                    "status":       "running",
                    "step_current": 0,
                    "step_total":   len(ack.get("planned_steps", [])),
                    "steps_done":   [],
                    "step_next":    (ack["planned_steps"][0]
                                     if ack.get("planned_steps") else ""),
                    "tokens_used":  0,
                    "issues":       [],
                }
                (worker / "state.json").write_text(json.dumps(state, indent=2))

                # Give the agent the ack's planned_steps as a hint,
                # but strip orchestrator-internal retry state (_ack_feedback).
                enriched_packet = {k: v for k, v in task_packet.items()
                                   if k != "_ack_feedback"}
                enriched_packet["planned_steps"] = ack.get("planned_steps", [])
                enriched_packet["restatement"]   = ack.get("restatement", "")
                (worker / "task.json").write_text(json.dumps(enriched_packet, indent=2))

                AgentRunner.run(enriched_packet, record_dict, base_path)

            except Exception as exc:
                logger.exception("Agent %s crashed in runner: %s", record.agent_id, exc)
                self._mark_agent_done(record, status="failed", output=str(exc))

        t = threading.Thread(target=_run, daemon=True, name=f"agent-{record.agent_id}")
        t.start()

    def _generate_ack(self, record: AgentRecord, task_packet: dict) -> dict:
        """Call ack_validation pipeline to generate Stage 2 ack."""
        system = (
            "You are an agent receiving a task. "
            "Respond ONLY with a JSON ack matching this schema exactly:\n"
            '{"task_id":"...","restatement":"one sentence","done_looks_like":["..."],'
            '"ambiguities":[],"planned_steps":["tool: action"],"confidence":0.0}\n'
            "Confidence must be >= 0.80. Resolve any ambiguities before listing them."
        )
        # Inject retry feedback if present
        feedback = task_packet.get("_ack_feedback", [])
        if feedback:
            system += (
                f"\n\nPrevious attempt was rejected. Issues to fix: "
                + "; ".join(feedback)
            )
        try:
            resp = self.tools.llm_call(
                pipeline="ack_validation",
                system=system,
                messages=[{"role": "user", "content": json.dumps(task_packet)}],
                max_tokens=512,
                temperature=0.0,
            )
            return json.loads(_strip_json_fences(resp.get("content", "")))
        except Exception as e:
            logger.warning("Ack generation failed: %s — using default", e)
            return {
                "task_id":        record.task_id,
                "restatement":    task_packet.get("objective", ""),
                "done_looks_like": [],
                "ambiguities":    [],
                "planned_steps":  [],
                "confidence":     0.85,
            }

    def _validate_ack(self, ack: dict, task_packet: dict, record: AgentRecord) -> tuple[bool, str]:
        """
        Validate Stage 2 ack against spec rules.
        Returns (ok, reason).
        """
        confidence = ack.get("confidence", 0)
        if confidence < self.config.ack_confidence_min:
            return False, f"confidence {confidence} < {self.config.ack_confidence_min}"

        ambiguities = ack.get("ambiguities", [])
        if ambiguities:
            # Attempt resolution (up to ack_max_rounds — simplified here)
            return False, f"unresolved ambiguities: {ambiguities}"

        restatement = ack.get("restatement", "")
        if not restatement:
            return False, "empty restatement"

        return True, "ok"

    # ------------------------------------------------------------------
    # Step 6 — Monitor
    # ------------------------------------------------------------------

    def _start_monitor(self) -> None:
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="orchestrator-monitor"
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while self._running:
            time.sleep(self.config.poll_interval_secs)
            for agent_id, record in list(self._agents.items()):
                if record.status != "running":
                    continue
                self._check_agent(record)

    def _check_agent(self, record: AgentRecord) -> None:
        """Poll state.json and enforce budget + timeout."""
        worker = Path(record.worker_dir)
        state_path = worker / "state.json"
        if not state_path.exists():
            return

        try:
            state = json.loads(state_path.read_text())
        except Exception:
            return

        tokens_used  = state.get("tokens_used", 0)
        elapsed      = record.elapsed()
        issues       = state.get("issues", [])

        # Token budget warning at 80%
        if tokens_used > record.token_budget * 0.8:
            self._status(f"⚠ Agent {record.agent_id} at {tokens_used}/{record.token_budget} tokens")

        # Token budget exceeded — kill
        if tokens_used > record.token_budget:
            self._status(f"✗ Agent {record.agent_id} exceeded token budget — killing")
            self._kill_agent(record, reason="token_budget_exceeded")
            return

        # Timeout exceeded — kill
        if elapsed > record.timeout_secs:
            self._status(f"✗ Agent {record.agent_id} timed out ({elapsed:.0f}s) — killing")
            self._kill_agent(record, reason="timeout")
            return

        # Issues in state
        if issues:
            self._status(f"⚠ Agent {record.agent_id} issues: {issues}")

        # Done
        if state.get("status") in ("done", "failed"):
            record.status = state["status"]

    def _kill_agent(self, record: AgentRecord, reason: str) -> None:
        record.status = "killed"
        worker = Path(record.worker_dir)
        state_path = worker / "state.json"
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            state["status"] = "killed"
            state["kill_reason"] = reason
            _atomic_write_json(state_path, state)
        except Exception:
            pass
        self._close_ct_flag(record.task_id, status="failed")

    def _mark_agent_done(
        self,
        record: AgentRecord,
        status: AgentStatus,
        output: str = "",
        confidence: float = 0.0,
    ) -> None:
        record.status = status
        worker = Path(record.worker_dir)
        result = {"status": status, "output": output, "confidence": confidence}
        (worker / "result.json").write_text(json.dumps(result, indent=2))

    # ------------------------------------------------------------------
    # Step 7 — Receive result
    # ------------------------------------------------------------------

    def _await_agent(self, record: AgentRecord) -> dict:
        """Block until agent finishes or times out."""
        deadline = time.time() + record.timeout_secs + 10  # small grace period
        while time.time() < deadline:
            if record.status in ("done", "failed", "killed"):
                break
            time.sleep(0.5)

        worker = Path(record.worker_dir)
        result_path = worker / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {"status": record.status, "output": "", "confidence": 0.0}

        confidence = result.get("confidence", 0.0)
        status     = result.get("status", "failed")

        if status == "done" and confidence >= self.config.promote_threshold:
            self._auto_promote(record, result)
        elif status == "done" and confidence < self.config.promote_threshold:
            self._flag_human_review(record, result)
        elif status == "failed":
            retried = self._retry_agent(record, result)
            if retried:
                # Re-enter wait loop for the retried run; cleanup happens on that return
                return self._await_agent(record)
            self._escalate(record, result)

        self._close_ct_flag(record.task_id, status="done" if status == "done" else "failed")
        self._cleanup_worker(record)

        return result

    def _auto_promote(self, record: AgentRecord, result: dict) -> None:
        """Move deliverable files from worker slot to dev/output/."""
        worker = Path(record.worker_dir)
        dev_out = self._project_path("dev", "output")
        dev_out.mkdir(parents=True, exist_ok=True)

        for f in worker.iterdir():
            if f.suffix in (".py", ".md", ".json", ".txt", ".html") and f.name not in (
                "meta.json", "ack.json", "state.json", "task.json", "result.json"
            ):
                dst = dev_out / f.name
                try:
                    self.tools.file_move(str(f), str(dst))
                    self._status(f"Promoted {f.name} → dev/output/")
                    self._reindex_file(str(dst))
                except Exception as e:
                    logger.warning("Promote failed for %s: %s", f.name, e)

    def _reindex_file(self, path: str) -> None:
        """
        Trigger file_indexing pipeline to generate summary + embedding.
        Updates both Layer 2 (dev/index.json) and Layer 1 (file_index.json).
        """
        try:
            content = self.tools.file_read(path)
            resp = self.tools.llm_call(
                pipeline="file_indexing",
                system="Summarise this file in 2 sentences. Output only the summary.",
                messages=[{"role": "user", "content": content[:4000]}],
                max_tokens=150,
            )
            summary   = resp.get("content", "")
            embedding = self.tools.embed_text(summary)

            # Layer 2 — dev/index.json (semantic RAG index)
            dev_idx_path = self._project_path("dev", "index.json")
            dev_idx_path.parent.mkdir(parents=True, exist_ok=True)
            dev_index: dict = {}
            if dev_idx_path.exists():
                try:
                    dev_index = json.loads(dev_idx_path.read_text())
                except Exception:
                    pass
            dev_index[path] = {
                "summary":      summary,
                "tags":         [],
                "embedding":    embedding,
                "last_indexed": _now_iso()[:10],
            }
            _atomic_write_json(dev_idx_path, dev_index)

            # Layer 1 — file_index.json (structured display registry)
            # Schema: {"dev": {"output": [{"path": filename, ...}]}, "worker": {...}}
            fi_path = self._project_path("file_index.json")
            file_index: dict = {}
            if fi_path.exists():
                try:
                    file_index = json.loads(fi_path.read_text())
                except Exception:
                    pass
            file_name  = Path(path).name
            dev_output = file_index.setdefault("dev", {}).setdefault("output", [])
            new_entry  = {"path": file_name, "summary": summary[:120],
                          "last_indexed": _now_iso()[:10]}
            existing   = next((i for i, e in enumerate(dev_output)
                               if e.get("path") == file_name), None)
            if existing is not None:
                dev_output[existing] = new_entry
            else:
                dev_output.append(new_entry)
            _atomic_write_json(fi_path, file_index)

            self._status(f"Re-indexed {path}")
        except Exception as e:
            logger.warning("Re-index failed for %s: %s", path, e)

    def _flag_human_review(self, record: AgentRecord, result: dict) -> None:
        entry = {
            "id":          str(uuid.uuid4()),
            "type":        "note",
            "description": f"Low-confidence result for task {record.task_id} — needs review",
            "task_id":     record.task_id,
            "agent_id":    record.agent_id,
            "instance_id": self.instance_id,
            "webui_action": "human_review",
            "opened_at":   _now_iso(),
            "status":      "open",
            "closed_at":   None,
        }
        try:
            self.tools.memory_write("ct", entry, project=self.project)
        except Exception as e:
            logger.warning("Human review flag failed: %s", e)
        self._status(f"⚠ Low confidence — flagged for human review")

    def _retry_agent(self, record: AgentRecord, result: dict) -> bool:
        """Returns True if retry was queued, False if retry_limit exhausted."""
        meta_path = Path(record.worker_dir) / "meta.json"
        meta: dict = {}
        retries = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            retries = meta.get("retries", 0)

        if retries >= self.config.retry_limit:
            return False

        self._status(f"Retrying agent (attempt {retries + 1}/{self.config.retry_limit})...")
        meta["retries"] = retries + 1
        meta_path.write_text(json.dumps(meta, indent=2))

        # Reset agent status and respawn — simplified; real impl rebuilds task packet
        record.status    = "running"
        record.spawned_at = time.time()
        task_path = Path(record.worker_dir) / "task.json"
        if task_path.exists():
            task_packet = json.loads(task_path.read_text())
            self._spawn_agent(record, task_packet)
        return True

    def _escalate(self, record: AgentRecord, result: dict) -> None:
        self._status(f"✗ Retries exhausted for {record.task_id} — escalating to user")

    def _cleanup_worker(self, record: AgentRecord) -> None:
        # Keep worker dir for audit; just remove from active registry
        self._agents.pop(record.agent_id, None)

    # ------------------------------------------------------------------
    # Step 8 — Post-process
    # ------------------------------------------------------------------

    def _post_process(self, message: str, result: dict, ctx: TurnContext) -> None:
        output = result.get("output", "")

        # ST — append turn, compress on overflow
        self._update_st(message, output)

        # History — always append raw turn
        self._append_history("user",      message)
        self._append_history("assistant", output)

        # MT — judge whether worth persisting
        self._mt_judge(message, output)

    def _update_st(self, user_msg: str, assistant_msg: str) -> None:
        """Append turn to ST window. Compress if overflow."""
        st = self._load_st()
        window: list[dict] = st.get("window", [])
        window.append({"role": "user",      "content": user_msg,      "ts": _now_iso()})
        window.append({"role": "assistant", "content": assistant_msg, "ts": _now_iso()})

        # Overflow — compress oldest half
        max_msgs = self.config.st_max_messages * 2  # pairs → messages
        if len(window) > max_msgs:
            to_compress = window[:max_msgs // 2]
            keep        = window[max_msgs // 2:]
            try:
                turns_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in to_compress
                )
                resp = self.tools.llm_call(
                    pipeline="compression",
                    system="Compress these conversation turns into a concise summary (max 3 sentences).",
                    messages=[{"role": "user", "content": turns_text}],
                    max_tokens=300,
                )
                new_summary = resp.get("content", st.get("summary", ""))
            except Exception:
                new_summary = st.get("summary", "")
            st["summary"] = new_summary
            window = keep

        st["window"]     = window
        st["updated_at"] = _now_iso()
        self._save_st(st)

    def _append_history(self, role: str, content: str) -> None:
        history_path = self._project_path("instances", self.instance_id, "history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []
        if history_path.exists():
            try:
                entries = json.loads(history_path.read_text())
            except Exception:
                pass
        entries.append({
            "turn_id":     str(uuid.uuid4()),
            "instance_id": self.instance_id,
            "project":     self.project,
            "role":        role,
            "content":     content,
            "ts":          _now_iso(),
        })
        _atomic_write_json(history_path, entries)

    def _mt_judge(self, user_msg: str, assistant_msg: str) -> None:
        """Ask mt_judge pipeline if this exchange is worth persisting to MT."""
        try:
            resp = self.tools.llm_call(
                pipeline="mt_judge",
                system=(
                    "Does this exchange contain durable knowledge worth saving to long-term memory? "
                    "Answer JSON: {\"persist\": true|false, \"content\": \"one-sentence fact if yes\"}"
                ),
                messages=[{"role": "user", "content": f"USER: {user_msg}\nASSISTANT: {assistant_msg}"}],
                max_tokens=128,
                temperature=0.0,
            )
            data = json.loads(_strip_json_fences(resp.get("content", "")))
            if data.get("persist") and data.get("content"):
                embedding = self.tools.embed_text(data["content"])
                entry = {
                    "id":         str(uuid.uuid4()),
                    "content":    data["content"],
                    "embedding":  embedding,
                    "type":       "learned_knowledge",
                    "status":     "active",
                    "created_at": _now_iso(),
                }
                self.tools.memory_write("mt", entry, project=self.project)
        except Exception as e:
            logger.debug("MT judge skipped: %s", e)

    # ------------------------------------------------------------------
    # SkillSmith trigger
    # ------------------------------------------------------------------

    def _trigger_skillsmith(self, message: str, ctx: TurnContext) -> None:
        """
        SkillSmith pipeline:
          1. Draft skill schema (skillsmith_draft pipeline)
          2. Run up to skillsmith_dry_runs dry-run evaluations
          3. If eval passes (score >= skillsmith_pass_score), save draft
          4. Open CT pending_approval flag regardless (human must approve)
        """
        draft_id   = f"draft-{uuid.uuid4().hex[:8]}"
        draft_dir  = self._project_path("worker", "skillsmith", draft_id)
        draft_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. Draft ----
        draft: dict = {}
        try:
            resp = self.tools.llm_call(
                pipeline="skillsmith_draft",
                system=(
                    "You are SkillSmith. Draft a skill schema for this capability gap.\n"
                    "Output ONLY JSON with keys: name, description, tier (1–4), "
                    "tags (list), tools_required (list), template (prompt template string)."
                ),
                messages=[{"role": "user", "content": f"Capability needed: {message}"}],
                max_tokens=768,
            )
            draft = json.loads(_strip_json_fences(resp["content"]))
        except Exception as e:
            logger.warning("SkillSmith draft failed: %s", e)

        draft.setdefault("name",          "Draft skill")
        draft.setdefault("description",   message)
        draft.setdefault("tier",          2)
        draft.setdefault("tags",          [])
        draft.setdefault("tools_required", [])
        draft.setdefault("template",      "Complete the task: {{objective}}")
        draft["skill_id"]   = draft_id
        draft["version"]    = 1
        draft["active"]     = False
        draft["created_by"] = "auto"
        draft["created_at"] = _now_iso()

        # Save initial draft
        (draft_dir / "draft.json").write_text(json.dumps(draft, indent=2))
        self._status(f"SkillSmith drafted: {draft['name']}")

        # ---- 2. Dry runs ----
        best_score = 0.0
        for run_n in range(1, self.config.skillsmith_dry_runs + 1):
            try:
                eval_resp = self.tools.llm_call(
                    pipeline="skillsmith_eval",
                    system=(
                        "You are a skill evaluator. Score how well the given skill draft "
                        "satisfies the capability requirement. "
                        "Output JSON: {\"score\": 0.0–1.0, \"feedback\": \"...\", \"pass\": true|false}"
                    ),
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Capability needed: {message}\n\n"
                            f"Skill draft:\n{json.dumps(draft, indent=2)}"
                        ),
                    }],
                    max_tokens=256,
                    temperature=0.0,
                )
                evaluation = json.loads(_strip_json_fences(eval_resp["content"]))
                score      = float(evaluation.get("score", 0.0))
                feedback   = evaluation.get("feedback", "")
                passed     = evaluation.get("pass", False)

                self._status(
                    f"SkillSmith dry run {run_n}/{self.config.skillsmith_dry_runs}: "
                    f"score={score:.2f}, pass={passed}"
                )
                (draft_dir / f"eval_{run_n}.json").write_text(json.dumps(evaluation, indent=2))

                best_score = max(best_score, score)
                if passed or score >= self.config.skillsmith_pass_score:
                    break

                # Refine draft with feedback
                if feedback:
                    try:
                        refine_resp = self.tools.llm_call(
                            pipeline="skillsmith_draft",
                            system=(
                                "You are SkillSmith. Revise the skill draft based on feedback. "
                                "Output ONLY the updated JSON skill object."
                            ),
                            messages=[{
                                "role": "user",
                                "content": (
                                    f"Current draft:\n{json.dumps(draft, indent=2)}\n\n"
                                    f"Evaluator feedback: {feedback}"
                                ),
                            }],
                            max_tokens=768,
                        )
                        updated = json.loads(_strip_json_fences(refine_resp["content"]))
                        # Preserve identity fields
                        for key in ("skill_id", "version", "active", "created_by", "created_at"):
                            updated[key] = draft[key]
                        draft = updated
                        draft["version"] += 1
                        (draft_dir / "draft.json").write_text(json.dumps(draft, indent=2))
                    except Exception as re:
                        logger.warning("SkillSmith refine failed: %s", re)

            except Exception as e:
                logger.warning("SkillSmith eval run %d failed: %s", run_n, e)

        # ---- 3. Save final draft with embedding ----
        try:
            draft["embedding"] = self.tools.embed_text(draft["description"])
        except Exception:
            pass
        (draft_dir / "draft.json").write_text(json.dumps(draft, indent=2))

        # ---- 4. Open CT pending_approval flag ----
        ct_entry = {
            "id":           str(uuid.uuid4()),
            "type":         "skill_approval",
            "description":  (
                f"SkillSmith drafted: {draft['name']} "
                f"(best eval score: {best_score:.2f})"
            ),
            "task_id":      None,
            "agent_id":     None,
            "instance_id":  self.instance_id,
            "skill_id":     draft_id,
            "draft_path":   str(draft_dir / "draft.json"),
            "webui_action": "skill_approval_modal",
            "opened_at":    _now_iso(),
            "status":       "pending_approval",
            "closed_at":    None,
        }
        try:
            self.tools.memory_write("ct", ct_entry, project=self.project)
        except Exception as e:
            logger.warning("SkillSmith CT flag failed: %s", e)

    # ------------------------------------------------------------------
    # CT helpers
    # ------------------------------------------------------------------

    def _close_ct_flag(self, task_id: str, status: CTStatus) -> None:
        try:
            all_ct = self.tools.memory_read("ct", project=self.project)
            for flag in all_ct:
                if flag.get("task_id") == task_id and flag.get("status") == "open":
                    flag["status"]    = status
                    flag["closed_at"] = _now_iso()
                    self.tools.memory_write("ct", flag, project=self.project)
        except Exception as e:
            logger.warning("CT close failed for task %s: %s", task_id, e)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get_ctx_manager(self) -> Any:
        """Lazy-load ContextManager (avoids import at module level)."""
        if self._ctx_manager is None:
            import sys
            _pikaia_dir = str(self.base_path)
            if _pikaia_dir not in sys.path:
                sys.path.insert(0, _pikaia_dir)
            from context_manager import ContextManager
            self._ctx_manager = ContextManager(self.tools, self.base_path, self.config)
        return self._ctx_manager

    def _project_path(self, *parts: str) -> Path:
        return self.base_path / "projects" / self.project / Path(*parts)

    def _load_st(self) -> dict:
        st_path = self._project_path("instances", self.instance_id, "st.json")
        if st_path.exists():
            try:
                return json.loads(st_path.read_text())
            except Exception:
                pass
        return {"instance_id": self.instance_id, "project": self.project,
                "summary": "", "window": [], "updated_at": _now_iso()}

    def _save_st(self, st: dict) -> None:
        st_path = self._project_path("instances", self.instance_id, "st.json")
        _atomic_write_json(st_path, st)

    def _cosine_top_k(
        self,
        query: list[float],
        index: dict[str, dict],
        top_k: int,
    ) -> list[dict]:
        scored = []
        for path, entry in index.items():
            emb = entry.get("embedding")
            if not emb:
                continue
            score = _cosine(query, emb)
            scored.append({
                "path":    path,
                "summary": entry.get("summary", ""),
                "score":   round(score, 4),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _status(self, msg: str) -> None:
        self.on_status(msg)

    def shutdown(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from LLM JSON responses."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[0] is before the opening fence (empty), parts[1] is the fenced block
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    return text


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via a temp file."""
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise