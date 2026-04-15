"""
deepseek_local.py — DeepSeek-R1 1.5B local provider
-----------------------------------------------------
Runs DeepSeek-R1:1.5B with zero API cost via two backends (tried in order):

  Strategy 1 — Ollama  (recommended, lowest RAM, fastest startup)
      ollama pull deepseek-r1:1.5b
      # Ollama must be running on localhost:11434

  Strategy 2 — HuggingFace transformers  (no Ollama required)
      pip install transformers torch accelerate
      # Downloads model on first use (~1.1 GB to ~/.cache/huggingface)

DeepSeek R1 outputs chain-of-thought reasoning wrapped in <think>…</think>
before the final answer.  This provider:
  - strips the <think> block from `content`  (clean answer only)
  - preserves the raw reasoning in response["thinking"]  (optional inspection)

No API key is required for either backend.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .base import BaseAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek model constants
# ---------------------------------------------------------------------------
OLLAMA_MODEL        = "deepseek-r1:1.5b"
HF_MODEL_ID         = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OLLAMA_BASE_URL     = "http://localhost:11434"
OLLAMA_CHAT_EP      = f"{OLLAMA_BASE_URL}/api/chat"
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_DELAY = 0.5

# Regex that captures the full <think>…</think> block (greedy off so we only
# grab the first block — R1 sometimes wraps an entire multi-step trace)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Thinking-block helpers
# ---------------------------------------------------------------------------

def _extract_thinking(text: str) -> tuple[str, str]:
    """Return (thinking_text, clean_answer).

    Strips *all* <think> blocks from the response; concatenates their inner
    text as the thinking trace.
    """
    thinking_parts: list[str] = []

    def _collect(m: re.Match) -> str:
        thinking_parts.append(m.group(1).strip())
        return ""

    clean = _THINK_RE.sub(_collect, text).strip()
    thinking = "\n\n".join(thinking_parts)
    return thinking, clean


# ---------------------------------------------------------------------------
# Transformers backend (lazy-loaded, class-level singleton)
# ---------------------------------------------------------------------------

class _TransformersBackend:
    """HuggingFace transformers inference.  Loaded once, reused across calls."""

    _pipe:  Any             = None
    _lock:  threading.Lock  = threading.Lock()  # guards one-time model load

    @classmethod
    def available(cls) -> bool:
        """Return True only if both transformers AND torch are importable."""
        try:
            import transformers  # noqa: F401
            import torch          # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def load(cls) -> Any:
        """Load the pipeline exactly once; thread-safe via double-checked locking."""
        if cls._pipe is not None:          # fast path — no lock needed
            return cls._pipe
        with cls._lock:
            if cls._pipe is not None:      # re-check inside lock
                return cls._pipe
            from transformers import pipeline as hf_pipeline  # type: ignore[import]
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1   # 0=GPU, -1=CPU
                dtype  = "auto" if torch.cuda.is_available() else None
            except ImportError:
                device = -1
                dtype  = None

            logger.info("Loading %s via transformers (device=%s)…", HF_MODEL_ID, device)
            kwargs: dict[str, Any] = dict(
                task              = "text-generation",
                model             = HF_MODEL_ID,
                device            = device,
                trust_remote_code = True,
            )
            if dtype:
                kwargs["torch_dtype"] = dtype
            cls._pipe = hf_pipeline(**kwargs)
            logger.info("DeepSeek-R1 1.5B loaded.")
            return cls._pipe

    @classmethod
    def generate(cls, messages: list[dict], max_tokens: int = 1024) -> dict[str, Any]:
        """Run inference.  Returns a minimal response dict."""
        pipe = cls.load()
        tok  = pipe.tokenizer

        # Use the model's chat template if available, otherwise build simple prompt
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            prompt = tok.apply_chat_template(
                messages,
                tokenize    = False,
                add_generation_prompt = True,
            )
        else:
            prompt = _simple_prompt(messages)

        outputs = pipe(
            prompt,
            max_new_tokens = max_tokens,
            do_sample      = True,
            temperature    = 0.6,
            pad_token_id   = tok.eos_token_id,
            return_full_text = False,   # return only the generated part
        )
        generated = outputs[0]["generated_text"]

        tokens_in  = len(tok.encode(prompt))
        tokens_out = len(tok.encode(generated))
        return {
            "_backend":    "transformers",
            "content":     generated,
            "tokens_in":   tokens_in,
            "tokens_out":  tokens_out,
        }


def _simple_prompt(messages: list[dict]) -> str:
    """Minimal chat→text conversion for models without a template."""
    parts: list[str] = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main Adapter
# ---------------------------------------------------------------------------

class Adapter(BaseAdapter):
    """Provider adapter for DeepSeek-R1 1.5B running locally."""

    def build_request(
        self,
        system:      str,
        messages:    list[dict],
        max_tokens:  int   = 1024,
        temperature: float | None = None,
        tools:       list[dict] | None = None,
        stream:      bool  = False,
    ) -> dict[str, Any]:
        """Build an Ollama-format chat request payload."""
        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Tool-use: inject tool schemas as a system instruction (text-injection mode)
        if tools:
            tool_hint = _build_tool_hint(tools)
            if full_messages and full_messages[0]["role"] == "system":
                full_messages[0]["content"] += "\n\n" + tool_hint
            else:
                full_messages.insert(0, {"role": "system", "content": tool_hint})

        payload: dict[str, Any] = {
            "model":    OLLAMA_MODEL,
            "messages": full_messages,
            "stream":   stream,
            "options":  {"num_predict": max_tokens},
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        # Stash full_messages so _call_transformers can use them
        payload["_messages_for_transformers"] = full_messages
        return payload

    def call(self, request: dict[str, Any]) -> Any:
        """Try Ollama first; fall back to transformers if Ollama is unreachable."""
        # Remove internal key before sending to Ollama
        messages_for_tf = request.pop("_messages_for_transformers", [])

        # --- Strategy 1: Ollama ---
        body    = json.dumps(request).encode()
        headers = {"Content-Type": "application/json"}
        req     = urllib.request.Request(OLLAMA_CHAT_EP, data=body,
                                         headers=headers, method="POST")
        delay   = DEFAULT_RETRY_DELAY
        for attempt in range(DEFAULT_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read().decode())
            except (urllib.error.URLError, ConnectionRefusedError) as exc:
                if attempt < DEFAULT_MAX_RETRIES:
                    logger.warning("DeepSeek/Ollama retry %d (%s)", attempt + 1, exc)
                    time.sleep(delay)
                    delay *= 2
                    continue
                logger.info("Ollama unreachable — switching to transformers backend.")
                break
            except Exception:
                raise

        # --- Strategy 2: transformers ---
        if not _TransformersBackend.available():
            raise RuntimeError(
                "DeepSeek-R1 1.5B: Ollama is not running and 'transformers' is not installed.\n"
                "Fix option A:  ollama pull deepseek-r1:1.5b  (then start Ollama)\n"
                "Fix option B:  pip install transformers torch accelerate"
            )
        max_tokens = request.get("options", {}).get("num_predict", 1024)
        return _TransformersBackend.generate(messages_for_tf, max_tokens)

    def parse_response(self, raw: Any) -> dict[str, Any]:
        """Normalise Ollama or transformers response; strip <think> blocks."""
        # Transformers backend already has a flat dict
        if isinstance(raw, dict) and raw.get("_backend") == "transformers":
            raw_content = raw.get("content", "")
            tokens_in   = raw.get("tokens_in", 0)
            tokens_out  = raw.get("tokens_out", 0)
            stop_reason = "end_turn"
        else:
            # Ollama format
            message     = raw.get("message", {})
            raw_content = message.get("content", "")
            tokens_in   = raw.get("prompt_eval_count", 0)
            tokens_out  = raw.get("eval_count", 0)
            stop_reason = "end_turn" if raw.get("done") else "max_tokens"

        # Strip <think> blocks — preserve reasoning, expose clean answer
        thinking, content = _extract_thinking(raw_content)

        content_blocks = [{"type": "text", "text": content}] if content else []

        resp = self._standard_response(
            content        = content,
            tokens_in      = tokens_in,
            tokens_out     = tokens_out,
            stop_reason    = stop_reason,
            content_blocks = content_blocks,
        )
        if thinking:
            resp["thinking"] = thinking     # available for logging / debugging
        return resp

    def validate_key(self) -> bool:
        """No API key required — always valid."""
        return True


# ---------------------------------------------------------------------------
# Tool-injection helper
# ---------------------------------------------------------------------------

def _build_tool_hint(tools: list[dict]) -> str:
    """Convert Anthropic-format tool schemas into a text instruction block."""
    lines = [
        "You have access to the following tools. To call a tool respond with a "
        "JSON block on its own line formatted exactly as shown:\n"
        '{"tool": "<name>", "input": {<params>}}\n'
    ]
    for t in tools:
        name  = t.get("name", "")
        desc  = t.get("description", "")
        props = t.get("input_schema", {}).get("properties", {})
        param_str = ", ".join(
            f"{k} ({v.get('type','any')}): {v.get('description','')}"
            for k, v in props.items()
        )
        lines.append(f"- {name}: {desc}  |  params: {param_str}")
    return "\n".join(lines)
