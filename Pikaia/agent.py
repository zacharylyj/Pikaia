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

New capabilities (compared to original)
----------------------------------------
1.  Step budget      — _tool_loop enforces max_steps (config or task_packet override).
                       On exceed, injects a budget-exhausted signal and forces a final
                       LLM turn so the agent can summarise gracefully.
2.  Error classes    — LLM and tool calls are wrapped with classify_error(); each
                       ErrorType has its own handler (retry/backoff, auth abort,
                       context-compress+retry, network retry, tool-continue).
3.  Context compress — When token usage crosses config.context_compression_threshold
                       (default 80 % of context window), older messages are summarised
                       and replaced so the loop can continue without hitting the limit.
4.  Parallel tools   — Independent tool calls (e.g. read-only ops) execute concurrently
                       via ThreadPoolExecutor.  Dependency detection classifies each
                       tool call, groups by safety class, and only parallelises the
                       safe subset.  config.parallel_tool_max_workers controls the pool.
5.  Model routing    — Short/simple tasks are routed to config.fast_model (Haiku by
                       default) rather than the main pipeline model, cutting cost.
6.  Key rotation     — keys.json may contain a list of API keys per provider.  On 429
                       or auth failure the pool advances to the next key and retries.
7.  Loop awareness   — After each tool-results turn, an awareness note is injected
                       into messages so the agent knows how many steps remain and which
                       tools it has used so far.
8.  Tool dep detect  — Covered under item 4 above.
9.  Trajectory log   — Each run writes a JSONL replay buffer + SQLite row.
10. Metrics          — Tokens, latency, steps, tool success rates written to SQLite.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default step cap — overridden by config["max_steps"] or task_packet["max_steps"].
_DEFAULT_MAX_STEPS = 15

# Tools that are safe to run in parallel (pure reads / no shared write side-effects).
_PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset({
    "file_read",
    "memory_read",
    "embed_text",
    "web_fetch",
    "http_request",
    "context_fetch",
    "skill_read",
    "code_exec",
})


# ---------------------------------------------------------------------------
# Key pool  (item 6 — API key rotation)
# ---------------------------------------------------------------------------

class _KeyPool:
    """
    Round-robin pool of API keys for a single provider.
    Advances on 429 / auth failure; blocks the failed key for `cooldown_secs`.
    """

    def __init__(self, keys: list[str], cooldown_secs: float = 60.0) -> None:
        self._keys     = deque(keys)
        self._lock     = threading.Lock()
        self._cooldown = cooldown_secs
        # {key: unblock_time}
        self._blocked: dict[str, float] = {}

    def current(self) -> str:
        with self._lock:
            return self._peek()

    def rotate(self, failed_key: str | None = None) -> str:
        """Block the failed key and advance to the next available one."""
        with self._lock:
            if failed_key and failed_key in self._keys:
                self._blocked[failed_key] = time.monotonic() + self._cooldown
            self._keys.rotate(-1)
            return self._peek()

    def _peek(self) -> str:
        now = time.monotonic()
        for _ in range(len(self._keys)):
            key = self._keys[0]
            if now >= self._blocked.get(key, 0):
                return key
            self._keys.rotate(-1)
        # All keys blocked — return the current one anyway (best effort)
        return self._keys[0]


def _build_key_pool(provider: str, base_path: Path) -> _KeyPool | None:
    """Load keys.json and return a _KeyPool if multiple keys exist."""
    keys_path = base_path / "keys.json"
    if not keys_path.exists():
        return None
    try:
        raw = json.loads(keys_path.read_text())
        entry = raw.get(provider)
        if isinstance(entry, list) and len(entry) > 1:
            return _KeyPool([k for k in entry if k])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Adapter loader
# ---------------------------------------------------------------------------

def _load_adapter(pipeline: str, base_path: Path, api_key: str | None = None) -> Any:
    """
    Resolve pipeline → model_id → provider → adapter instance.
    Returns (adapter, model_id, provider_name).
    api_key overrides whatever is in keys.json (used for key rotation).
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

    if api_key is None:
        keys: dict = {}
        keys_path = base_path / "keys.json"
        if keys_path.exists():
            try:
                raw_keys = json.loads(keys_path.read_text())
                for k, v in raw_keys.items():
                    # Support both single key (str) and list-of-keys
                    keys[k] = v[0] if isinstance(v, list) else v
            except Exception:
                pass
        api_key = keys.get(provider_name)

    provider_file = base_path / "tools" / "providers" / f"{provider_name}.py"
    if not provider_file.exists():
        raise FileNotFoundError(f"Provider adapter not found: {provider_file}")
    _pikaia = str(base_path)
    if _pikaia not in sys.path:
        sys.path.insert(0, _pikaia)
    _full_mod = f"tools.providers.{provider_name}"
    mod = sys.modules.get(_full_mod) or importlib.import_module(_full_mod)
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


def _should_use_fast_model(task_packet: dict, config: dict) -> bool:
    """
    Return True if the task is simple enough to route to config.fast_model.
    Criteria: short objective AND few tools_allowed.
    """
    fast_model = config.get("fast_model", "")
    if not fast_model:
        return False
    threshold_words = config.get("fast_model_threshold_words", 50)
    threshold_tools = config.get("fast_model_threshold_tools", 1)
    objective    = task_packet.get("objective", "")
    tools_allowed = task_packet.get("tools_allowed", [])
    return len(objective.split()) <= threshold_words and len(tools_allowed) <= threshold_tools


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
        record:      dict,
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
        self._tokens_used = 0
        self._tokens_lock = threading.Lock()

        # Load config once; make available to subclass helpers
        self._config = _load_config(self.base_path)

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

        # Load provider adapter — route to fast model for simple tasks
        try:
            if _should_use_fast_model(task_packet, self._config):
                fast_model = self._config["fast_model"]
                self._adapter, self._model_id, self._provider = _load_adapter(
                    fast_model, self.base_path
                )
                logger.debug(
                    "agent %s routed to fast_model '%s'", self.agent_id, fast_model
                )
            else:
                self._adapter, self._model_id, self._provider = _load_adapter(
                    self.pipeline, self.base_path
                )
        except Exception as exc:
            logger.warning("Adapter load failed for pipeline '%s': %s", self.pipeline, exc)
            self._adapter = self._model_id = self._provider = None  # type: ignore[assignment]

        # Key rotation pool (None if single key or rotation disabled)
        self._key_pool: _KeyPool | None = None
        if self._provider and self._config.get("key_rotation_enabled", True):
            self._key_pool = _build_key_pool(self._provider, self.base_path)

        # DeepSeek-R1 local fallback adapter (lazy-loaded on first use)
        self._deepseek_adapter: Any = None

        # Observability
        from metrics import MetricsCollector
        metrics_on = task_packet.get("metrics_enabled", self._config.get("metrics_enabled", True))
        self._metrics = MetricsCollector(task_id=self.task_id, enabled=metrics_on)

        # Trajectory logger
        from trajectory import TrajectoryLogger
        traj_on = task_packet.get("trajectory_logging", self._config.get("trajectory_logging", True))
        self._traj = TrajectoryLogger(
            task_id   = self.task_id,
            agent_id  = self.agent_id,
            project   = self.project,
            tier      = self.tier,
            base_path = self.base_path,
            enabled   = traj_on,
        )

        # Resolve DB (lazy — only used if metrics/trajectory enabled)
        self._db = None

    def _get_db(self):
        """Return the shared DB instance (lazy init)."""
        if self._db is None:
            try:
                from db import get_db
                db_path = self._config.get("db_path") or str(self.base_path / "pikaia.db")
                self._db = get_db(db_path)
            except Exception as exc:
                logger.warning("DB init failed: %s", exc)
        return self._db

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
        self._finalise_observability(outcome="done", output=output)

    def _mark_failed(self, reason: str) -> None:
        result = {"status": "failed", "output": reason, "confidence": 0.0}
        _atomic_write(self.worker_dir / "result.json", json.dumps(result, indent=2))
        self._write_state(tokens=self._tokens_used, status="failed",
                          step=0, total=0, issues=[reason])
        self._finalise_observability(outcome="failed", output=reason)

    def _step_total(self) -> int:
        state_path = self.worker_dir / "state.json"
        if state_path.exists():
            try:
                return json.loads(state_path.read_text()).get("step_total", 0)
            except Exception:
                pass
        return 0

    def _finalise_observability(self, outcome: str, output: str) -> None:
        """Flush metrics and trajectory to storage at run end."""
        db = self._get_db()
        try:
            if db:
                self._metrics.flush(db)
        except Exception as exc:
            logger.warning("Metrics flush failed: %s", exc)
        try:
            self._traj.finalise(outcome=outcome, output=output, db=db)
        except Exception as exc:
            logger.warning("Trajectory finalise failed: %s", exc)

    # ------------------------------------------------------------------
    # Context compression  (item 3)
    # ------------------------------------------------------------------

    def _context_window_size(self) -> int:
        """Return the context window for the active model (tokens)."""
        models_path = self.base_path / "models.json"
        if models_path.exists():
            try:
                raw = json.loads(models_path.read_text())
                if isinstance(raw, dict):
                    raw = [raw]
                entry = next(
                    (m for m in raw if m.get("model_id") == self._model_id), None
                )
                if entry:
                    return int(entry.get("context_window", 128000))
            except Exception:
                pass
        return 128000  # safe default

    def _compress_messages(self, messages: list[dict], step: int) -> list[dict]:
        """
        Summarise early messages to reduce context size.

        Keeps the last 6 messages verbatim (3 full turns); summarises everything
        before them into a single 'context summary' user message.
        """
        KEEP_TAIL = 6
        if len(messages) <= KEEP_TAIL:
            return messages

        to_compress = messages[:-KEEP_TAIL]
        tail        = messages[-KEEP_TAIL:]

        # Build a short text of what happened so far
        summary_parts = []
        for m in to_compress:
            role    = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                # Anthropic content blocks
                text = " ".join(
                    b.get("text", "") or b.get("content", "")
                    for b in content if isinstance(b, dict)
                )
            else:
                text = str(content)
            if text.strip():
                summary_parts.append(f"[{role}]: {text[:300]}")

        summary_text = "\n".join(summary_parts)

        # Call compression pipeline
        compressed_summary = ""
        if self._adapter is not None:
            try:
                system = (
                    "You are a conversation compressor. "
                    "Summarise the following exchange into 2–4 concise bullet points "
                    "that preserve all decisions and facts needed to continue the task. "
                    "Output plain text only."
                )
                req = self._adapter.build_request(
                    system=system,
                    messages=[{"role": "user", "content": summary_text}],
                    max_tokens=300,
                    temperature=0.0,
                )
                raw  = self._adapter.call(req)
                resp = self._adapter.parse_response(raw)
                with self._tokens_lock:
                    self._tokens_used += resp.get("tokens_in", 0) + resp.get("tokens_out", 0)
                compressed_summary = resp.get("content", "")
            except Exception as exc:
                logger.warning("Context compression call failed: %s", exc)
                compressed_summary = summary_text[:600]

        before = len(messages)
        result = [
            {"role": "user", "content": f"[Earlier context summary]\n{compressed_summary}"}
        ] + tail
        self._traj.log_compression(before_msgs=before, after_msgs=len(result), step=step)
        logger.debug("Compressed %d → %d messages at step %d", before, len(result), step)
        return result

    # ------------------------------------------------------------------
    # Parallel tool execution  (items 4 + 13)
    # ------------------------------------------------------------------

    def _partition_tool_calls(
        self, tool_calls: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Split tool_calls into (parallel_safe, sequential).

        A call is parallel-safe when:
        - Its tool name is in _PARALLEL_SAFE_TOOLS, AND
        - tool_dependency_detection is enabled in config.
        """
        if not self._config.get("tool_dependency_detection", True):
            return [], tool_calls

        safe = [tc for tc in tool_calls if tc.get("name") in _PARALLEL_SAFE_TOOLS]
        seq  = [tc for tc in tool_calls if tc.get("name") not in _PARALLEL_SAFE_TOOLS]
        return safe, seq

    def _dispatch_tool(self, tc: dict, step: int) -> tuple[dict, str]:
        """
        Execute a single tool call and return (tc, result_str).
        Records metrics + trajectory.
        """
        t0 = time.monotonic()
        try:
            result = self._registry.dispatch(tc["name"], tc["input"])
            # Unwrap ToolResult so the LLM receives actual data, not the envelope
            if isinstance(result, dict) and "success" in result and "data" in result:
                llm_payload = result["data"] if result["success"] else {"error": result.get("error", "")}
            else:
                llm_payload = result
            result_str = json.dumps(llm_payload) if not isinstance(llm_payload, str) else llm_payload
            latency_ms = (time.monotonic() - t0) * 1000
            self._metrics.record_tool_call(tc["name"], success=True, latency_ms=latency_ms)
            self._traj.log_tool_result(tc["name"], result_str, latency_ms=latency_ms,
                                       success=True, step=step)
            return tc, result_str
        except PermissionError as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            msg = f"PermissionError: {exc}"
            self._metrics.record_tool_call(tc["name"], success=False,
                                           latency_ms=latency_ms, error_msg=msg)
            self._traj.log_tool_result(tc["name"], msg, latency_ms=latency_ms,
                                       success=False, step=step)
            return tc, msg
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            msg = f"Error executing {tc['name']}: {exc}"
            logger.warning("Tool '%s' failed: %s", tc["name"], exc)
            self._metrics.record_tool_call(tc["name"], success=False,
                                           latency_ms=latency_ms, error_msg=str(exc))
            self._traj.log_tool_result(tc["name"], msg, latency_ms=latency_ms,
                                       success=False, step=step)
            return tc, msg

    def _execute_tool_calls(
        self, tool_calls: list[dict], step: int
    ) -> dict[str, str]:
        """
        Execute all tool_calls and return {tc_id: result_str}.
        Independent (parallel-safe) calls run concurrently.
        """
        results: dict[str, str] = {}

        # Log all calls to trajectory before execution
        for tc in tool_calls:
            self._traj.log_tool_call(tc["name"], tc.get("input", {}), step=step)

        safe_calls, seq_calls = self._partition_tool_calls(tool_calls)
        max_workers = self._config.get("parallel_tool_max_workers", 4)

        # Parallel execution for safe calls
        if safe_calls and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(safe_calls))) as pool:
                futures = {
                    pool.submit(self._dispatch_tool, tc, step): tc
                    for tc in safe_calls
                }
                for future in as_completed(futures):
                    tc_done, res = future.result()
                    results[tc_done["id"]] = res
        elif safe_calls:
            for tc in safe_calls:
                _, res = self._dispatch_tool(tc, step)
                results[tc["id"]] = res

        # Sequential execution for write/compute calls
        for tc in seq_calls:
            _, res = self._dispatch_tool(tc, step)
            results[tc["id"]] = res

        return results

    # ------------------------------------------------------------------
    # DeepSeek-R1 local fallback  (called when primary LLM call fails)
    # ------------------------------------------------------------------

    def _try_deepseek_fallback(
        self,
        system:       str,
        messages:     list[dict],
        max_tokens:   int,
        tool_schemas: list[dict] | None,
    ) -> "dict | None":
        """Attempt one call with deepseek-r1:1.5b as a last-resort local fallback.

        Skipped if:
          - deepseek_fallback_enabled is False in config
          - the current provider is already deepseek_local (avoid recursion)
        The adapter is cached after first load so Ollama/transformers pays the
        startup cost only once per agent run.
        """
        if not self._config.get("deepseek_fallback_enabled", True):
            return None
        if self._provider == "deepseek_local":
            return None  # already on DeepSeek — don't recurse

        try:
            if self._deepseek_adapter is None:
                from tools.providers.deepseek_local import Adapter as _DSAdapter  # noqa: F401
                self._deepseek_adapter = _DSAdapter(api_key=None, model_id="deepseek-r1:1.5b")
                logger.info("agent %s: DeepSeek fallback adapter loaded", self.agent_id)

            req  = self._deepseek_adapter.build_request(
                system      = system,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = None,
                tools       = tool_schemas if tool_schemas else None,
            )
            raw  = self._deepseek_adapter.call(req)
            resp = self._deepseek_adapter.parse_response(raw)
            logger.info(
                "agent %s: DeepSeek fallback succeeded (%d out-tokens)",
                self.agent_id, resp.get("tokens_out", 0),
            )
            return resp
        except Exception as exc:
            logger.warning("agent %s: DeepSeek fallback failed: %s", self.agent_id, exc)
            return None

    # ------------------------------------------------------------------
    # ReAct tool loop  (items 1–4, 7, 12)
    # ------------------------------------------------------------------

    def _tool_loop(
        self,
        system:        str,
        messages:      list[dict],
        tools_allowed: list[str],
        max_turns:     int | None = None,
    ) -> tuple[str, int]:
        """
        Run a ReAct (Reason+Act) multi-turn conversation.

        Returns (final_content, total_tokens_used).

        Enhanced with:
        - Step budget enforcement (config.max_steps)
        - Classify-and-route error handling
        - In-flight context compression at 80 % of context window
        - Parallel tool dispatch for independent calls
        - Loop awareness injection after each tool-results turn
        """
        if self._adapter is None:
            return "", 0

        from tools.schemas import get_schemas
        from tools.error_types import classify_error, ErrorType

        tool_schemas = get_schemas(tools_allowed) if tools_allowed else []

        # Resolve step cap: task_packet > config > module default
        max_steps = (
            self.task_packet.get("max_steps")
            or self._config.get("max_steps", _DEFAULT_MAX_STEPS)
        )
        if max_turns is not None:
            max_steps = min(max_steps, max_turns)

        context_window    = self._context_window_size()
        compress_threshold = (
            self.task_packet.get("compression_threshold")
            or self._config.get("context_compression_threshold", 0.80)
        )
        loop_awareness    = self.task_packet.get(
            "loop_awareness", self._config.get("loop_awareness_injection", True)
        )
        retry_max         = self._config.get("error_retry_max", 3)
        retry_base_delay  = self._config.get("error_retry_base_delay", 1.0)

        total_tokens  = 0
        current_msgs  = list(messages)
        last_content  = ""
        tool_use_hist: list[str] = []   # names of tools used so far

        for step in range(max_steps):
            self._metrics.record_step()

            # ── Budget injection ──────────────────────────────────────
            steps_remaining = max_steps - step
            if loop_awareness and step > 0:
                tool_summary = (
                    f"Tools used: {', '.join(tool_use_hist[-5:])}"
                    if tool_use_hist else "No tools used yet"
                )
                awareness_note = (
                    f"[System: {steps_remaining} step(s) remaining. {tool_summary}]"
                )
                # Inject as the last user message prefix
                if current_msgs and current_msgs[-1]["role"] == "user":
                    last_msg = current_msgs[-1]
                    if isinstance(last_msg.get("content"), str):
                        last_msg = dict(last_msg)
                        last_msg["content"] = awareness_note + "\n" + last_msg["content"]
                        current_msgs[-1] = last_msg

            # ── Context compression check ──────────────────────────────
            estimated_tokens = self._tokens_used + total_tokens
            if estimated_tokens > context_window * compress_threshold:
                current_msgs = self._compress_messages(current_msgs, step=step)

            # ── LLM call with error classification ────────────────────
            remaining_budget = max(0, self.token_budget - self._tokens_used - total_tokens)
            max_tokens_this  = min(remaining_budget or 4096, 4096)

            resp: dict = {}
            llm_error: Exception | None = None
            for attempt in range(retry_max + 1):
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
                    llm_error = None
                    break
                except Exception as exc:
                    llm_error  = exc
                    error_type = classify_error(exc)

                    if error_type == ErrorType.AUTH:
                        logger.error("Auth error in tool_loop: %s — aborting", exc)
                        return last_content, total_tokens

                    if error_type == ErrorType.CONTEXT_OVERFLOW:
                        logger.warning("Context overflow at step %d — compressing", step)
                        current_msgs = self._compress_messages(current_msgs, step=step)
                        continue   # retry immediately after compression

                    if error_type in (ErrorType.RATE_LIMIT, ErrorType.NETWORK):
                        # Key rotation on rate-limit
                        if error_type == ErrorType.RATE_LIMIT and self._key_pool:
                            new_key = self._key_pool.rotate(
                                failed_key=getattr(self._adapter, "api_key", None)
                            )
                            self._adapter.api_key = new_key  # type: ignore[attr-defined]
                            logger.info("Rotated API key at step %d", step)

                        if attempt < retry_max:
                            delay = retry_base_delay * (2 ** attempt)
                            logger.warning(
                                "LLM %s error (attempt %d/%d), retrying in %.1fs: %s",
                                error_type.name, attempt + 1, retry_max, delay, exc,
                            )
                            time.sleep(delay)
                            continue

                    logger.error("LLM call failed at step %d: %s", step, exc)
                    break

            if llm_error is not None or not resp:
                if llm_error is None:
                    break  # Empty response without error — give up

                # Primary failed: attempt DeepSeek-R1 1.5B local fallback
                fallback = self._try_deepseek_fallback(
                    system, current_msgs, max_tokens_this, tool_schemas
                )
                if not fallback:
                    break  # All providers exhausted — give up

                # Fallback succeeded: replace resp and continue step processing
                resp      = fallback
                llm_error = None
                logger.info("agent %s: recovered via DeepSeek fallback at step %d",
                            self.agent_id, step)
                # No break — falls through intentionally to turn_tokens processing

            turn_tokens = resp.get("tokens_in", 0) + resp.get("tokens_out", 0)
            total_tokens  += turn_tokens
            self._metrics.record_tokens(resp.get("tokens_in", 0), resp.get("tokens_out", 0))

            last_content   = resp.get("content", "")
            stop_reason    = resp.get("stop_reason", "end_turn")
            content_blocks = resp.get("content_blocks", [])
            tool_calls     = resp.get("tool_calls", [])

            # Log LLM turn to trajectory
            self._traj.log_llm_turn(
                content    = last_content,
                tokens_in  = resp.get("tokens_in", 0),
                tokens_out = resp.get("tokens_out", 0),
                step       = step,
            )

            # Append assistant turn (provider-appropriate)
            if self._provider == "anthropic":
                current_msgs.append({
                    "role":    "assistant",
                    "content": content_blocks or last_content,
                })
            else:
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

            # ── Step budget exhaustion ─────────────────────────────────
            if step >= max_steps - 1:
                # Last allowed step — force a final answer turn
                current_msgs.append({
                    "role":    "user",
                    "content": (
                        "[tool budget exhausted — return your best answer now "
                        "based on what you have gathered so far]"
                    ),
                })
                # One final LLM call (no tools)
                try:
                    req_final = self._adapter.build_request(
                        system=system, messages=current_msgs,
                        max_tokens=max_tokens_this, temperature=None, tools=None,
                    )
                    raw_final  = self._adapter.call(req_final)
                    resp_final = self._adapter.parse_response(raw_final)
                    total_tokens += resp_final.get("tokens_in", 0) + resp_final.get("tokens_out", 0)
                    self._metrics.record_tokens(
                        resp_final.get("tokens_in", 0), resp_final.get("tokens_out", 0)
                    )
                    last_content = resp_final.get("content", last_content)
                except Exception as exc:
                    logger.warning("Budget-exhausted final turn failed: %s", exc)
                break

            # ── Execute tool calls ─────────────────────────────────────
            tool_use_hist.extend(tc["name"] for tc in tool_calls)
            results_map = self._execute_tool_calls(tool_calls, step=step)

            # Collect ordered tool results (preserve original tc order)
            tool_results: list[dict] = []
            for tc in tool_calls:
                result_str = results_map.get(tc["id"], f"Error: no result for {tc['id']}")
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
                current_msgs.extend(tool_results)
            else:
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
        skill_id = self.task_packet.get("skill_id") or self.task_packet.get("skill", "")
        if not skill_id:
            return ""
        try:
            result = self._registry.dispatch("skill_read", {"skill_id": skill_id})
            # Unwrap ToolResult envelope if present
            if isinstance(result, dict) and "success" in result and "data" in result:
                result = result["data"] or {}
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
        objective     = self.task_packet.get("objective", "")
        tools_allowed = self.task_packet.get("tools_allowed", [])
        template      = self._load_skill_template()

        system = (
            "You are a capable agent. Complete the task given by the user using the "
            "tools available to you. Be concise and precise.\n"
        )
        if template:
            system += f"\n## Skill guidance\n{template}"

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
        self._write_state(step=0, total=1, tokens=0, step_next="tool_loop: execute objective")

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

        self._write_state(step=0, total=1, tokens=0, step_next="decompose objective")
        steps = self._decompose(objective, template)
        total = len(steps)
        self._write_state(step=0, total=total, tokens=self._tokens_used,
                          step_next=steps[0] if steps else "")

        step_outputs: list[str] = []
        for i, step_desc in enumerate(steps):
            self._write_state(step=i, total=total, tokens=self._tokens_used,
                              step_next=step_desc)

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
            self._write_state(step=i + 1, total=total, tokens=self._tokens_used)

        synthesis = self._synthesize(objective, step_outputs, tools_allowed)
        self._mark_done(output=synthesis, confidence=0.88)

    def _decompose(self, objective: str, template: str) -> list[str]:
        system = (
            "You are a task planner. Break the objective into 3–7 concrete, executable steps. "
            "Output JSON array of strings: [\"step 1 description\", ...]"
        )
        if template:
            system += f"\n\nSkill guidance: {template}"
        messages = [{"role": "user", "content": objective}]

        if self._adapter is None:
            return [objective]
        try:
            request = self._adapter.build_request(
                system=system, messages=messages, max_tokens=512, temperature=0.0,
            )
            try:
                raw  = self._adapter.call(request)
                resp = self._adapter.parse_response(raw)
            except Exception as exc:
                # Primary adapter failed — try DeepSeek local fallback
                resp = self._try_deepseek_fallback(system, messages, 512, None)
                if resp is None:
                    raise exc
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
            try:
                content, tokens = self._tool_loop(
                    system=system, messages=messages, tools_allowed=tools_allowed
                )
            except Exception as exc:
                content = f"[{name} failed: {exc}]"
            with lock:
                specialist_outputs[name] = content

        task_timeout       = self.task_packet.get("timeout_secs", 600)
        specialist_timeout = max(60, task_timeout - 60)

        threads = []
        for name, persona in _COUNCIL_SPECIALISTS:
            t = threading.Thread(target=_run_specialist, args=(name, persona), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=specialist_timeout)

        self._write_state(
            step=n_specialists, total=n_specialists + 1,
            tokens=self._tokens_used, step_next="council synthesis",
        )

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
        tier   = record.get("tier", 2)
        worker = Path(record.get("worker_dir") or
                      Path(base_path) / "projects" / record["project"] /
                      "worker" / record["agent_id"])
        worker.mkdir(parents=True, exist_ok=True)

        cls_map    = {1: Tier12Agent, 2: Tier12Agent, 3: Tier3Agent, 4: Tier4Council}
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
