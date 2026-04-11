"""
Groq Provider Adapter
---------------------
Implements BaseAdapter for the Groq Cloud API (groq.com).

Groq is OpenAI-API-compatible with a different base URL and key prefix.
Free tier includes generous rate limits on Llama and Gemma models.

Free models (as of 2025):
  llama-3.1-8b-instant     — fast, cheap, good for classification/compression
  llama-3.3-70b-versatile  — strong, good for orchestration/planning/code
  gemma2-9b-it             — Google Gemma 2 9B

Get a free key at: https://console.groq.com/keys
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

BASE_URL        = "https://api.groq.com/openai/v1"
CHAT_ENDPOINT   = f"{BASE_URL}/chat/completions"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0   # Groq rate-limits are strict; back off generously


def _to_openai_tool(t: dict) -> dict:
    """Translate Anthropic tool schema (input_schema) → OpenAI/Groq function format (parameters)."""
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
        req   = urllib.request.Request(CHAT_ENDPOINT, data=body, headers=headers, method="POST")
        delay = DEFAULT_RETRY_DELAY
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                last_exc = exc
                body_text = exc.read().decode() if exc.fp else str(exc)
                if exc.code == 429:
                    # Rate limit — always retry with backoff
                    if attempt < max_retries:
                        logger.warning("Groq rate-limited, retry %d/%d (wait %.1fs)",
                                       attempt + 1, max_retries, delay)
                        time.sleep(delay)
                        delay *= 2
                        continue
                if exc.code in (500, 502, 503):
                    if attempt < max_retries:
                        logger.warning("Groq server error %d, retry %d/%d",
                                       exc.code, attempt + 1, max_retries)
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise RuntimeError(f"Groq API error {exc.code}: {body_text}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning("Groq retry %d/%d (%s)", attempt + 1, max_retries, exc)
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
        return bool(self.api_key and self.api_key.startswith("gsk_"))
