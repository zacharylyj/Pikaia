"""
Anthropic Provider Adapter
--------------------------
Implements BaseAdapter for the Anthropic Messages API.

Deliverables:
- messages endpoint, x-api-key auth
- streaming support (via stream=True in build_request)
- exponential-backoff retry on rate-limit / 5xx
- tool_use block parsing
- standard response normalisation
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Generator

from .base import BaseAdapter

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0


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
        payload: dict[str, Any] = {
            "model":      self.model_id,
            "max_tokens": max_tokens,
            "messages":   messages,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = {"type": "auto"}
        if stream:
            payload["stream"] = True
        return payload

    def call(
        self,
        request: dict[str, Any],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> dict[str, Any]:
        body = json.dumps(request).encode()
        headers = {
            "x-api-key":         self.api_key or "",
            "anthropic-version": API_VERSION,
            "content-type":      "application/json",
        }
        req = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")

        delay = DEFAULT_RETRY_DELAY
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in (429, 500, 502, 503, 529):
                    if attempt < max_retries:
                        logger.warning("Anthropic retry %d/%d (HTTP %d)", attempt + 1, max_retries, exc.code)
                        time.sleep(delay)
                        delay *= 2
                        continue
                body_text = exc.read().decode() if exc.fp else str(exc)
                raise RuntimeError(f"Anthropic API error {exc.code}: {body_text}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning("Anthropic retry %d/%d (%s)", attempt + 1, max_retries, exc)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def parse_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        content         = ""
        tool_calls:     list[dict] = []
        content_blocks: list[dict] = []

        for block in raw.get("content", []):
            btype = block.get("type")
            if btype == "text":
                content += block.get("text", "")
                content_blocks.append({"type": "text", "text": block.get("text", "")})
            elif btype == "tool_use":
                tc = {
                    "id":    block.get("id"),
                    "name":  block.get("name"),
                    "input": block.get("input", {}),
                }
                tool_calls.append(tc)
                content_blocks.append({
                    "type":  "tool_use",
                    "id":    tc["id"],
                    "name":  tc["name"],
                    "input": tc["input"],
                })

        usage       = raw.get("usage", {})
        stop_reason = raw.get("stop_reason", "end_turn")

        return self._standard_response(
            content        = content,
            tokens_in      = usage.get("input_tokens", 0),
            tokens_out     = usage.get("output_tokens", 0),
            stop_reason    = stop_reason,
            tool_calls     = tool_calls,
            content_blocks = content_blocks,
        )

    def validate_key(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("sk-ant-"))

    # ------------------------------------------------------------------
    # Streaming helper (optional — yields text chunks)
    # ------------------------------------------------------------------

    def stream(
        self,
        system:      str,
        messages:    list[dict],
        max_tokens:  int = 1024,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        request = self.build_request(
            system=system, messages=messages,
            max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        body    = json.dumps(request).encode()
        headers = {
            "x-api-key":         self.api_key or "",
            "anthropic-version": API_VERSION,
            "content-type":      "application/json",
        }
        req = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                except json.JSONDecodeError:
                    continue
