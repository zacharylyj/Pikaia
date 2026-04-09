"""
agent.py
--------
Agent execution engine for all 4 tiers.

Tier 1/2  Atomic/Composite : single ReAct tool loop driven by a skill template
Tier 3    Sub-agent loop   : task_planning decompose → step loop with checkpoints
Tier 4    Council          : N parallel Tier-3-like specialists → synthesis

Entry point
-----------
    AgentRunner.run(task_packet, record, base_path)

Called by Orchestrator._spawn_agent inside a daemon thread. Writes ack.json,
state.json (checkpointed), and result.json to the agent's worker directory.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How many tool turns a single agent loop may execute before we force-stop.
_MAX_TOOL_TURNS = 20


# ---------------------------------------------------------------------------
# Adapter loader (mirrors llm_call logic — avoids circular imports)
# ---------------------------------------------------------------------------

def _load_adapter(pipeline: str, base_path: Path) -> Any:
    """
    Resolve pipeline → model_id → provider → adapter instance.
    Returns (adapter, model_id, provider_name).
    """
    config    = _load_config(base_path)
    pipelines = config.get("pipelines", {})
    model_id  = pipelines.get(pipeline, pipeline)

    models_path = base_path / "models.json"
    if not models_path.exists():
        raise FileNotFoundError(f"models.json not found at {models_path}")
    raw_models = json.loads(models_path.read_text())
    if isinstance(raw_models, dict):
        raw_models = [raw_models]
    model_entry = next(
        (m for m in raw_models if m.get("model_id") == model_id and m.get("enabled", True)),
        None,
    )
    if model_entry is None:
        raise ValueError(f"Model '{model_id}' not found or disabled in models.json")
    provider_name = model_entry["provider"]

    keys: dict = {}
    keys_path = base_path / "keys.json"
    if keys_path.exists():
        try:
            keys = json.loads(keys_path.read_text())
        except Exception:
            pass
    api_key = keys.get(provider_name)

    provider_file = base_path / "tools" / "providers" / f"{provider_name}.py"
    if not provider_file.exists():
        raise FileNotFoundError(f"Provider adapter not found: {provider_file}")
    import sys as _sys
    _pikaia = str(base_path)
    if _pikaia not in _sys.path:
        _sys.path.insert(0, _pikaia)
    _full_mod = f"tools.providers.{provider_name}"
    mod = _sys.modules.get(_full_mod) or importlib.import_module(_full_mod)
    return mod.Adapter(api_key=api_key, model_id=model_id), model_id, provider_name


def _load_config(base_path: Path) -> dict:
    cfg: dict = {}
    global_path = base_path / "config.json"
    if global_path.exists():
        try:
            cfg.update(json.loads(global_path.read_text()))
        except Exception:
            pass
    return cfg


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Shared infrastructure for all tiers.
    Subclasses implement run() with tier-specific logic.
    """

    def __init__(
        self,
        task_packet: dict,
        record:      dict,       # AgentRecord.meta_dict() serialisation
        base_path:   str,
    ) -> None:
        self.task_packet  = task_packet
        self.record       = record
        self.base_path    = Path(base_path)
        self.agent_id     = record["agent_id"]
        self.task_id      = record["task_id"]
        self.project      = record["project"]
        self.instance_id  = record["instance_id"]
        self.pipeline     = record["pipeline"]
        self.tier         = record["tier"]
        self.token_budget = record["token_budget"]
        self.worker_dir   = Path(record.get("worker_dir") or
                                 self.base_path / "projects" / self.project /
                                 "worker" / self.agent_id)
        self._tokens_used  = 0
        self._tokens_lock  = threading.Lock()   # guards _tokens_used for Tier4 parallel threads

        # Build tool registry with agent-scoped permissions
        from tools.registry import ToolRegistry
        self._registry = ToolRegistry(
            base_path    = str(self.base_path),
            project      = self.project,
            instance_id  = self.instance_id,
            caller       = "agent",
            agent_id     = self.agent_id,
            worker_dir   = str(self.worker_dir),
            token_budget = self.token_budget,
        )

        # Load provider adapter for direct multi-turn use
        try:
            self._adapter, self._model_id, self._provider = _load_adapter(
                self.pipeline, self.base_path
            )
        except Exception as exc:
            logger.warning("Adapter load failed for pipeline '%s': %s", self.pipeline, exc)
            self._adapter = self._model_id = self._provider = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Handshake helpers
    # ------------------------------------------------------------------

    def _write_ack(self, ack: dict) -> None:
        (self.worker_dir / "ack.json").write_text(json.dumps(ack, indent=2))

    def _write_state(
        self,
        step:      int,
        total:     int,
        tokens:    int,
        status:    str = "running",
        step_next: str = "",
        issues:    list[str] | None = None,
    ) -> None:
        state = {
            "task_id":      self.task_id,
            "status":       status,
            "step_current": step,
            "step_total":   total,
            "steps_done":   [],
            "step_next":    step_next,
            "tokens_used":  tokens,
            "issues":       issues or [],
        }
        state_path = self.worker_dir / "state.json"
        _atomic_write(state_path, json.dumps(state, indent=2))

    def _mark_done(self, output: str, confidence: float = 0.9) -> None:
        result = {"status": "done", "output": output, "confidence": confidence}
        _atomic_write(self.worker_dir / "result.json", json.dumps(result, indent=2))
        self._write_state(
            step=self._step_total(), total=self._step_total(),
            tokens=self._tokens_used, status="done",
        )

    def _mark_failed(self, reason: str) -> None:
        result = {"status": "failed", "output": reason, "confidence": 0.0}
        _atomic_write(self.worker_dir / "result.json", json.dumps(result, indent=2))
        self._write_state(tokens=self._tokens_used, status="failed",
                          step=0, total=0, issues=[reason])

    def _step_total(self) -> int:
        state_path = self.worker_dir / "state.json"
        if state_path.exists():
            try:
                return json.loads(state_path.read_text()).get("step_total", 0)
            except Exception:
                pass
        return 0

    # ------------------------------------------------------------------
    # ReAct tool loop
    # ------------------------------------------------------------------

    def _tool_loop(
        self,
        system:        str,
        messages:      list[dict],
        tools_allowed: list[str],
        max_turns:     int = _MAX_TOOL_TURNS,
    ) -> tuple[str, int]:
        """
        Run a ReAct (Reason+Act) multi-turn conversation.

        Returns (final_content, total_tokens_used).

        Provider-agnostic: detects provider from self._provider and formats
        tool results accordingly (Anthropic vs OpenAI vs Ollama/none).
        """
        if self._adapter is None:
            return "", 0

        from tools.schemas import get_schemas
        tool_schemas = get_schemas(tools_allowed) if tools_allowed else []

        total_tokens   = 0
        current_msgs   = list(messages)
        last_content   = ""

        for _turn in range(max_turns):
            # Build + call
            remaining_budget = max(0, self.token_budget - self._tokens_used - total_tokens)
            max_tokens_this  = min(remaining_budget or 4096, 4096)

            try:
                request = self._adapter.build_request(
                    system      = system,
                    messages    = current_msgs,
                    max_tokens  = max_tokens_this,
                    temperature = None,
                    tools       = tool_schemas if tool_schemas else None,
                )
                raw  = self._adapter.call(request)
                resp = self._adapter.parse_response(raw)
            except Exception as exc:
                logger.error("LLM call failed in tool_loop turn %d: %s", _turn, exc)
                break

            total_tokens  += resp.get("tokens_in", 0) + resp.get("tokens_out", 0)
            last_content   = resp.get("content", "")
            stop_reason    = resp.get("stop_reason", "end_turn")
            content_blocks = resp.get("content_blocks", [])
            tool_calls     = resp.get("tool_calls", [])

            # Append assistant turn (provider-appropriate)
            if self._provider == "anthropic":
                # Anthropic expects content_blocks list as assistant content
                current_msgs.append({
                    "role":    "assistant",
                    "content": content_blocks or last_content,
                })
            else:
                # OpenAI / Ollama: string content + tool_calls as sibling
                asst_msg: dict[str, Any] = {"role": "assistant", "content": last_content or None}
                if tool_calls and self._provider == "openai":
                    asst_msg["tool_calls"] = [
                        {
                            "id":       tc["id"],
                            "type":     "function",
                            "function": {
                                "name":      tc["name"],
                                "arguments": json.dumps(tc["input"]),
                            },
                        }
                        for tc in tool_calls
                    ]
                current_msgs.append(asst_msg)

            # End conditions
            if stop_reason != "tool_use" or not tool_calls:
                break

            # Execute tools
            tool_results: list[dict] = []
            for tc in tool_calls:
                try:
                    result     = self._registry.dispatch(tc["name"], tc["input"])
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                except PermissionError as exc:
                    result_str = f"PermissionError: {exc}"
                except Exception as exc:
                    result_str = f"Error executing {tc['name']}: {exc}"
                    logger.warning("Tool '%s' failed: %s", tc["name"], exc)

                if self._provider == "anthropic":
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tc["id"],
                        "content":     result_str,
                    })
                elif self._provider == "openai":
                    tool_results.append({
                        "role":         "tool",
                        "tool_call_id": tc["id"],
                        "content":      result_str,
                    })

            # Append tool results
            if self._provider == "anthropic":
                current_msgs.append({"role": "user", "content": tool_results})
            elif self._provider == "openai":
                # Each tool result is its own message
                current_msgs.extend(tool_results)
            else:
                # Ollama / unknown: can't do real tool loops — inject results as user text
                results_text = "\n".join(
                    f"Tool result: {r.get('content', '')}" for r in tool_results
                )
                current_msgs.append({"role": "user", "content": results_text})

        with self._tokens_lock:
            self._tokens_used += total_tokens
        return last_content, total_tokens

    # ------------------------------------------------------------------
    # Skill template loader
    # ------------------------------------------------------------------

    def _load_skill_template(self) -> str:
        # "skill_id" is the canonical key; fall back to "skill" (name) for older packets
        skill_id = self.task_packet.get("skill_id") or self.task_packet.get("skill", "")
        if not skill_id:
            return ""
        try:
            result = self._registry.dispatch("skill_read", {"skill_id": skill_id})
            return result.get("template", "") or ""
        except Exception:
            return ""

    def run(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tier 1 / 2  Atomic / Composite
# ---------------------------------------------------------------------------

class Tier12Agent(BaseAgent):
    """
    Single ReAct tool loop.
    Tier 1 (atomic): one tool call expected.
    Tier 2 (composite): multi-step but single continuous loop.
    """

    def run(self) -> None:
        objective = self.task_packet.get("objective", "")
        tools_allowed = self.task_packet.get("tools_allowed", [])

        # Load skill template for extra context
        template = self._load_skill_template()

        system = (
            "You are a capable agent. Complete the task given by the user using the "
            "tools available to you. Be concise and precise.\n"
        )
        if template:
            system += f"\n## Skill guidance\n{template}"

        # Context from task packet
        ctx = self.task_packet.get("context", {})
        ctx_lines: list[str] = []
        if ctx.get("lt_summary"):
            ctx_lines.append(f"Long-term context: {ctx['lt_summary']}")
        if ctx.get("st_summary"):
            ctx_lines.append(f"Session summary: {ctx['st_summary']}")
        if ctx.get("mt_retrieved"):
            ctx_lines.append("Relevant knowledge:\n" + "\n".join(
                f"- {c.get('content', str(c)) if isinstance(c, dict) else c}"
                for c in ctx["mt_retrieved"]
            ))
        if ctx_lines:
            system += "\n\n## Context\n" + "\n".join(ctx_lines)

        messages = [{"role": "user", "content": objective}]
        planned_steps = ["tool_loop: execute objective"]

        # Write initial state
        self._write_state(step=0, total=1, tokens=0, step_next=planned_steps[0])

        # Run
        content, tokens = self._tool_loop(
            system        = system,
            messages      = messages,
            tools_allowed = tools_allowed,
        )

        self._write_state(step=1, total=1, tokens=tokens, status="done")
        self._mark_done(output=content, confidence=0.85)


# ---------------------------------------------------------------------------
# Tier 3  Sub-agent loop
# ---------------------------------------------------------------------------

class Tier3Agent(BaseAgent):
    """
    Decompose objective → execute each step with tool_loop → synthesize.
    Writes state.json after every step (checkpoint pattern).
    """

    def run(self) -> None:
        objective     = self.task_packet.get("objective", "")
        tools_allowed = self.task_packet.get("tools_allowed", [])
        template      = self._load_skill_template()

        # ---- Step 0: Decompose via task_planning pipeline ----
        self._write_state(step=0, total=1, tokens=0, step_next="decompose objective")
        steps = self._decompose(objective, template)
        total = len(steps)
        self._write_state(step=0, total=total, tokens=self._tokens_used,
                          step_next=steps[0] if steps else "")

        step_outputs: list[str] = []
        for i, step_desc in enumerate(steps):
            self._write_state(
                step=i, total=total, tokens=self._tokens_used,
                step_next=step_desc,
            )

            system = (
                "You are an agent executing one step of a larger task. "
                "Use tools as needed. Be precise.\n"
                f"\n## Overall objective\n{objective}"
            )
            if template:
                system += f"\n\n## Skill guidance\n{template}"

            messages = [{"role": "user", "content": f"Complete this step: {step_desc}"}]
            content, _ = self._tool_loop(
                system        = system,
                messages      = messages,
                tools_allowed = tools_allowed,
            )
            step_outputs.append(f"Step {i + 1} ({step_desc}):\n{content}")
            self._write_state(
                step=i + 1, total=total, tokens=self._tokens_used,
            )

        # ---- Final synthesis ----
        synthesis = self._synthesize(objective, step_outputs, tools_allowed)
        self._mark_done(output=synthesis, confidence=0.88)

    def _decompose(self, objective: str, template: str) -> list[str]:
        """Ask task_planning pipeline to break objective into ordered steps."""
        system = (
            "You are a task planner. Break the objective into 3–7 concrete, executable steps. "
            "Output JSON array of strings: [\"step 1 description\", ...]"
        )
        if template:
            system += f"\n\nSkill guidance: {template}"
        messages = [{"role": "user", "content": objective}]

        # Use the adapter directly for this single call
        if self._adapter is None:
            return [objective]  # fallback: treat whole thing as one step

        try:
            request = self._adapter.build_request(
                system=system, messages=messages, max_tokens=512, temperature=0.0,
            )
            raw  = self._adapter.call(request)
            resp = self._adapter.parse_response(raw)
            with self._tokens_lock:
                self._tokens_used += resp.get("tokens_in", 0) + resp.get("tokens_out", 0)
            content = resp.get("content", "")
            steps = json.loads(_strip_json_fences(content))
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps
        except Exception as e:
            logger.warning("Decompose failed: %s — treating as single step", e)
        return [objective]

    def _synthesize(self, objective: str, step_outputs: list[str], tools_allowed: list[str]) -> str:
        """Combine step outputs into a final cohesive answer."""
        combined = "\n\n".join(step_outputs)
        system   = (
            "You are synthesising the outputs of several sub-steps into a single final answer. "
            "Be clear and concise."
        )
        messages = [
            {"role": "user", "content": f"Objective: {objective}\n\n## Steps completed\n{combined}\n\nWrite the final answer."}
        ]
        content, _ = self._tool_loop(
            system=system, messages=messages, tools_allowed=[], max_turns=2
        )
        return content


# ---------------------------------------------------------------------------
# Tier 4  Council
# ---------------------------------------------------------------------------

_COUNCIL_SPECIALISTS = [
    ("researcher",  "You are a deep researcher. Gather facts, cite evidence, be thorough."),
    ("critic",      "You are a critical analyst. Challenge assumptions, identify flaws, be rigorous."),
    ("synthesiser", "You are a synthesis expert. Integrate multiple perspectives into coherent insights."),
]


class Tier4Council(BaseAgent):
    """
    Spawn N specialist sub-agents in parallel threads, collect their outputs,
    then run a council_synthesis call to produce the final answer.
    """

    def run(self) -> None:
        objective     = self.task_packet.get("objective", "")
        tools_allowed = self.task_packet.get("tools_allowed", [])
        n_specialists = len(_COUNCIL_SPECIALISTS)

        self._write_state(step=0, total=n_specialists + 1, tokens=0,
                          step_next="spawning specialist council")

        specialist_outputs: dict[str, str] = {}
        lock = threading.Lock()

        def _run_specialist(name: str, persona: str) -> None:
            system   = f"{persona}\n\nComplete the task thoroughly."
            messages = [{"role": "user", "content": objective}]
            # Each specialist gets its own sub-loop
            try:
                content, tokens = self._tool_loop(
                    system=system, messages=messages, tools_allowed=tools_allowed
                )
            except Exception as exc:
                content = f"[{name} failed: {exc}]"
            with lock:
                specialist_outputs[name] = content

        # Each specialist gets most of the task's timeout budget;
        # leave ~60 s for synthesis afterward.
        task_timeout      = self.task_packet.get("timeout_secs", 600)
        specialist_timeout = max(60, task_timeout - 60)

        threads = []
        for name, persona in _COUNCIL_SPECIALISTS:
            t = threading.Thread(
                target=_run_specialist, args=(name, persona), daemon=True
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=specialist_timeout)

        self._write_state(
            step=n_specialists, total=n_specialists + 1,
            tokens=self._tokens_used, step_next="council synthesis",
        )

        # Council synthesis
        synthesis = self._council_synthesis(objective, specialist_outputs)
        self._mark_done(output=synthesis, confidence=0.92)

    def _council_synthesis(self, objective: str, outputs: dict[str, str]) -> str:
        parts = "\n\n".join(
            f"## {name.capitalize()}'s analysis\n{text}"
            for name, text in outputs.items()
        )
        system = (
            "You are a council chair. Synthesise the specialist analyses below into "
            "one authoritative, balanced final answer."
        )
        messages = [
            {"role": "user", "content": f"Objective: {objective}\n\n{parts}\n\nProvide the final synthesis."}
        ]
        content, _ = self._tool_loop(
            system=system, messages=messages, tools_allowed=[], max_turns=3
        )
        return content


# ---------------------------------------------------------------------------
# AgentRunner — single entry point called by Orchestrator
# ---------------------------------------------------------------------------

class AgentRunner:

    @staticmethod
    def run(task_packet: dict, record: dict, base_path: str) -> None:
        """
        Instantiate the correct tier agent and call .run().
        Exceptions are caught and written to result.json so the monitor can see them.
        """
        tier     = record.get("tier", 2)
        worker   = Path(record.get("worker_dir") or
                        Path(base_path) / "projects" / record["project"] /
                        "worker" / record["agent_id"])
        worker.mkdir(parents=True, exist_ok=True)

        cls_map = {1: Tier12Agent, 2: Tier12Agent, 3: Tier3Agent, 4: Tier4Council}
        AgentClass = cls_map.get(tier, Tier12Agent)

        agent = AgentClass(task_packet=task_packet, record=record, base_path=base_path)
        try:
            agent.run()
        except Exception as exc:
            logger.exception("AgentRunner tier-%d crashed: %s", tier, exc)
            agent._mark_failed(str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from LLM JSON responses."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    return text


def _atomic_write(path: Path, content: str) -> None:
    """Write to a temp file then atomically replace."""
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
