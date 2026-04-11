"""
OpenAI Provider Adapter
-----------------------
Implements BaseAdapter for the OpenAI Chat Completions API.

Deliverables:
- chat/completions endpoint, Bearer auth
- system prompt injected as role:system message
- standard response normalisation
- retry on rate-limit / 5xx
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from .base import BaseAdapter

logger = logging.getLogger(__name__)

ENDPOINT        = "https://api.openai.com/v1/chat/completions"
EMBED_ENDPOINT  = "https://api.openai.com/v1/embeddings"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0


def _to_openai_tool(t: dict) -> dict:
    """Translate Anthropic tool schema (input_schema) → OpenAI function format (parameters)."""
    fn: dict = {"name": t["name"], "description": t.get("description", "")}
    schema = t.get("input_schema") or t.get("parameters", {})
    if schema:
        fn["parameters"] = schema
    return {"type": "function", "function": fn}


class Adapter(BaseAdapter):

    def build_request(
        self,
        system:      str,
        messages:    list[dict],
        max_tokens:  int  = 1024,
        temperature: float | None = None,
        tools:       list[dict] | None = None,
        stream:      bool = False,
    ) -> dict[str, Any]:
        # OpenAI: system prompt as first message with role "system"
        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model":      self.model_id,
            "max_tokens": max_tokens,
            "messages":   full_messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"]       = [_to_openai_tool(t) for t in tools]
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
        return payload

    def call(
        self,
        request: dict[str, Any],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> dict[str, Any]:
        body    = json.dumps(request).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type":  "application/json",
        }
        req   = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")
        delay = DEFAULT_RETRY_DELAY
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in (429, 500, 502, 503):
                    if attempt < max_retries:
                        logger.warning("OpenAI retry %d/%d (HTTP %d)", attempt + 1, max_retries, exc.code)
                        time.sleep(delay)
                        delay *= 2
                        continue
                body_text = exc.read().decode() if exc.fp else str(exc)
                raise RuntimeError(f"OpenAI API error {exc.code}: {body_text}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning("OpenAI retry %d/%d (%s)", attempt + 1, max_retries, exc)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def parse_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        choice      = raw.get("choices", [{}])[0]
        message     = choice.get("message", {})
        content     = message.get("content") or ""
        stop_reason = choice.get("finish_reason", "stop")

        tool_calls:     list[dict] = []
        content_blocks: list[dict] = []

        if content:
            content_blocks.append({"type": "text", "text": content})

        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                input_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_args = {}
            parsed_tc = {
                "id":    tc.get("id"),
                "name":  fn.get("name"),
                "input": input_args,
            }
            tool_calls.append(parsed_tc)
            content_blocks.append({
                "type":  "tool_use",
                "id":    parsed_tc["id"],
                "name":  parsed_tc["name"],
                "input": parsed_tc["input"],
            })

        usage = raw.get("usage", {})
        return self._standard_response(
            content        = content,
            tokens_in      = usage.get("prompt_tokens", 0),
            tokens_out     = usage.get("completion_tokens", 0),
            stop_reason    = stop_reason,
            tool_calls     = tool_calls,
            content_blocks = content_blocks,
        )

    def validate_key(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("sk-"))

    # ------------------------------------------------------------------
    # Embedding helper (used by embed_text tool)
    # ------------------------------------------------------------------

    def embed(self, text: str, embed_model: str = "text-embedding-3-small") -> list[float]:
        payload = {"model": embed_model, "input": text}
        body    = json.dumps(payload).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type":  "application/json",
        }
        req = urllib.request.Request(EMBED_ENDPOINT, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data["data"][0]["embedding"]
