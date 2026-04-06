"""
Anthropic Adapter
-----------------
A versatile, orchestrator-friendly adapter for the Anthropic API.

Design principles:
- Orchestrator has full external control over every parameter
- Sane defaults for agent-level usage
- Supports tools, streaming, retries, memory injection, hooks
- All state is explicit — no hidden globals
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Literal

import anthropic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------

Role = Literal["user", "assistant"]
StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds, doubles on each retry


# ---------------------------------------------------------------------------
# Config — the orchestrator's control surface
# ---------------------------------------------------------------------------

@dataclass
class AdapterConfig:
    """
    Every knob the orchestrator can turn.
    Pass a custom instance to AnthropicAdapter to override defaults.
    """

    # Model selection
    model: str = DEFAULT_MODEL

    # Generation limits
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float | None = None          # None = API default (~1.0)
    top_p: float | None = None
    top_k: int | None = None

    # Stop sequences
    stop_sequences: list[str] = field(default_factory=list)

    # System prompt — orchestrator can set a global one, agents can override
    system_prompt: str | None = None

    # Retry behaviour
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY  # base delay; doubles each attempt

    # Streaming
    stream: bool = False

    # Tools (Anthropic tool-use format)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: dict[str, Any] | None = None  # e.g. {"type": "auto"}

    # Extra kwargs forwarded verbatim to the API (escape hatch)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: Role
    content: str | list[dict[str, Any]]  # str for text, list for multi-part

    def to_api(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


def user_message(text: str) -> Message:
    return Message(role="user", content=text)


def assistant_message(text: str) -> Message:
    return Message(role="assistant", content=text)


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class AdapterResponse:
    raw: anthropic.types.Message          # full API response
    text: str                             # convenience: first text block
    stop_reason: StopReason
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[dict[str, Any]]      # populated when stop_reason == "tool_use"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        return (
            f"AdapterResponse(model={self.model!r}, "
            f"stop_reason={self.stop_reason!r}, "
            f"tokens={self.input_tokens}+{self.output_tokens}, "
            f"text={self.text[:60]!r}{'...' if len(self.text) > 60 else ''})"
        )


# ---------------------------------------------------------------------------
# Hooks — orchestrator can observe every call
# ---------------------------------------------------------------------------

@dataclass
class AdapterHooks:
    """
    Optional callbacks the orchestrator registers to observe adapter activity.
    All hooks are optional — leave None to skip.
    """
    on_request:  Callable[[dict[str, Any]], None] | None = None
    on_response: Callable[[AdapterResponse], None] | None = None
    on_retry:    Callable[[int, Exception], None] | None = None
    on_error:    Callable[[Exception], None] | None = None


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    """
    Thin, versatile adapter around the Anthropic Python SDK.

    Orchestrator usage
    ------------------
    # Create with custom config
    config = AdapterConfig(model="claude-opus-4-6", temperature=0.2)
    adapter = AnthropicAdapter(config=config)

    # Override config per-call (non-destructive)
    response = adapter.chat(
        messages=[user_message("Hello")],
        config_override=AdapterConfig(max_tokens=512),
    )

    # Inject memory / system context
    response = adapter.chat(
        messages=history,
        system="You are a search agent. Be concise.",
    )

    # Enable streaming
    for chunk in adapter.stream(messages=[user_message("Tell me a story")]):
        print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str | None = None,   # falls back to ANTHROPIC_API_KEY env var
        config: AdapterConfig | None = None,
        hooks: AdapterHooks | None = None,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self.config = config or AdapterConfig()
        self.hooks = hooks or AdapterHooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        config_override: AdapterConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> AdapterResponse:
        """
        Single-turn (or continued) chat. Returns a full AdapterResponse.

        Parameters
        ----------
        messages        : conversation history (user + assistant turns)
        system          : overrides config.system_prompt for this call only
        config_override : replace any AdapterConfig fields for this call only
        tools           : override tool list for this call only
        tool_choice     : override tool choice for this call only
        """
        cfg = self._merge_config(config_override)
        payload = self._build_payload(
            messages=messages,
            cfg=cfg,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
        )
        return self._call_with_retry(payload, cfg)

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        config_override: AdapterConfig | None = None,
    ) -> Generator[str, None, None]:
        """
        Streaming variant. Yields text chunks as they arrive.
        Usage: for chunk in adapter.stream(messages): print(chunk, end="")
        """
        cfg = self._merge_config(config_override)
        cfg.stream = True
        payload = self._build_payload(messages=messages, cfg=cfg, system=system)

        if self.hooks.on_request:
            self.hooks.on_request(payload)

        with self._client.messages.stream(**payload) as stream_ctx:
            for text in stream_ctx.text_stream:
                yield text

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        config_override: AdapterConfig | None = None,
    ) -> AdapterResponse:
        """Convenience wrapper: single user prompt → response."""
        return self.chat(
            messages=[user_message(prompt)],
            system=system,
            config_override=config_override,
        )

    # ------------------------------------------------------------------
    # Orchestrator control utilities
    # ------------------------------------------------------------------

    def update_config(self, **kwargs: Any) -> None:
        """
        Mutate the adapter's base config in-place.
        Useful for the orchestrator to switch models mid-run, adjust
        temperature globally, add tools, etc.

        adapter.update_config(model="claude-opus-4-6", temperature=0.0)
        """
        for key, value in kwargs.items():
            if not hasattr(self.config, key):
                raise ValueError(f"AdapterConfig has no field '{key}'")
            setattr(self.config, key, value)

    def with_config(self, **kwargs: Any) -> "AnthropicAdapter":
        """
        Return a new adapter sharing the same client but with an overridden
        config. Non-destructive — original adapter is unchanged.

        agent_adapter = orchestrator_adapter.with_config(
            system_prompt="You are a search agent.",
            temperature=0.3,
            max_tokens=1024,
        )
        """
        import dataclasses
        new_cfg = dataclasses.replace(self.config, **kwargs)
        return AnthropicAdapter(config=new_cfg, hooks=self.hooks)

    def register_hooks(self, hooks: AdapterHooks) -> None:
        """Replace hooks at runtime (e.g. orchestrator wires up logging)."""
        self.hooks = hooks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_config(self, override: AdapterConfig | None) -> AdapterConfig:
        """Merge a per-call override on top of the base config."""
        if override is None:
            return self.config
        import dataclasses
        base = dataclasses.asdict(self.config)
        diff = {
            k: v for k, v in dataclasses.asdict(override).items()
            if v != getattr(AdapterConfig(), k)  # only non-default fields
        }
        base.update(diff)
        return AdapterConfig(**base)

    def _build_payload(
        self,
        messages: list[Message],
        cfg: AdapterConfig,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "messages": [m.to_api() for m in messages],
        }

        # System prompt: call-level > config-level
        resolved_system = system or cfg.system_prompt
        if resolved_system:
            payload["system"] = resolved_system

        # Optional generation params
        if cfg.temperature is not None:
            payload["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            payload["top_k"] = cfg.top_k
        if cfg.stop_sequences:
            payload["stop_sequences"] = cfg.stop_sequences

        # Tools: call-level > config-level
        resolved_tools = tools if tools is not None else cfg.tools
        if resolved_tools:
            payload["tools"] = resolved_tools
            resolved_tool_choice = tool_choice or cfg.tool_choice
            if resolved_tool_choice:
                payload["tool_choice"] = resolved_tool_choice

        # Escape hatch
        payload.update(cfg.extra_kwargs)

        return payload

    def _call_with_retry(
        self,
        payload: dict[str, Any],
        cfg: AdapterConfig,
    ) -> AdapterResponse:
        if self.hooks.on_request:
            self.hooks.on_request(payload)

        last_exc: Exception | None = None
        delay = cfg.retry_delay

        for attempt in range(cfg.max_retries + 1):
            try:
                raw = self._client.messages.create(**payload)
                response = self._parse_response(raw)
                if self.hooks.on_response:
                    self.hooks.on_response(response)
                return response

            except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
                last_exc = exc
                if attempt == cfg.max_retries:
                    break
                if self.hooks.on_retry:
                    self.hooks.on_retry(attempt + 1, exc)
                logger.warning("Retry %d/%d after error: %s", attempt + 1, cfg.max_retries, exc)
                time.sleep(delay)
                delay *= 2  # exponential backoff

            except anthropic.APIConnectionError as exc:
                last_exc = exc
                if attempt == cfg.max_retries:
                    break
                if self.hooks.on_retry:
                    self.hooks.on_retry(attempt + 1, exc)
                logger.warning("Connection retry %d/%d: %s", attempt + 1, cfg.max_retries, exc)
                time.sleep(delay)
                delay *= 2

        if self.hooks.on_error and last_exc:
            self.hooks.on_error(last_exc)
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _parse_response(raw: anthropic.types.Message) -> AdapterResponse:
        text = ""
        tool_calls: list[dict[str, Any]] = []

        for block in raw.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return AdapterResponse(
            raw=raw,
            text=text,
            stop_reason=raw.stop_reason,
            model=raw.model,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# Memory helpers (short-term conversation history)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Manages a rolling message history for a single conversation thread.
    The orchestrator creates one per session/agent and passes it around.
    """

    def __init__(self, max_turns: int | None = None) -> None:
        self._messages: list[Message] = []
        self.max_turns = max_turns  # None = unlimited

    def add_user(self, text: str) -> None:
        self._messages.append(user_message(text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._messages.append(assistant_message(text))
        self._trim()

    def add_response(self, response: AdapterResponse) -> None:
        """Append an AdapterResponse as the next assistant turn."""
        self.add_assistant(response.text)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        """Serialisable snapshot — useful for persistence / long-term memory."""
        return [m.to_api() for m in self._messages]

    def load(self, snapshot: list[dict[str, Any]]) -> None:
        """Restore from a serialised snapshot."""
        self._messages = [
            Message(role=m["role"], content=m["content"]) for m in snapshot
        ]

    def _trim(self) -> None:
        if self.max_turns and len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._messages)//2}, max={self.max_turns})"


# ---------------------------------------------------------------------------
# Quick usage example (run with: python anthropic_adapter.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # --- Orchestrator sets up a base adapter ---
    base_config = AdapterConfig(
        model=DEFAULT_MODEL,
        temperature=0.7,
        max_tokens=1024,
    )

    hooks = AdapterHooks(
        on_request=lambda p: logger.info("→ request model=%s tokens=%s", p["model"], p["max_tokens"]),
        on_response=lambda r: logger.info("← response stop=%s tokens=%s+%s", r.stop_reason, r.input_tokens, r.output_tokens),
        on_retry=lambda n, e: logger.warning("retrying (%d): %s", n, e),
    )

    adapter = AnthropicAdapter(config=base_config, hooks=hooks)

    # --- Orchestrator spins up a focused agent adapter ---
    search_adapter = adapter.with_config(
        system_prompt="You are a concise search agent. Return only facts.",
        temperature=0.2,
        max_tokens=512,
    )

    # --- Short-term memory for a session ---
    memory = ConversationMemory(max_turns=10)

    # --- Simple chat loop ---
    memory.add_user("What is retrieval-augmented generation?")
    response = search_adapter.chat(messages=memory.messages)
    memory.add_response(response)
    print(response.text)