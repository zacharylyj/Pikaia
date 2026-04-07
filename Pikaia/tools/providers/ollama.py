"""
Ollama Provider Adapter
-----------------------
Implements BaseAdapter for a local Ollama instance.

Deliverables:
- localhost:11434 endpoint, no auth required
- local model routing via model_id
- standard response normalisation
- retry on transient connection errors
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

BASE_URL        = "http://localhost:11434"
CHAT_ENDPOINT   = f"{BASE_URL}/api/chat"
EMBED_ENDPOINT  = f"{BASE_URL}/api/embeddings"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 0.5


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
            "model":    self.model_id,
            "messages": full_messages,
            "stream":   stream,
            "options":  {},
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        return payload

    def call(
        self,
        request: dict[str, Any],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> dict[str, Any]:
        body    = json.dumps(request).encode()
        headers = {"Content-Type": "application/json"}
        req     = urllib.request.Request(CHAT_ENDPOINT, data=body, headers=headers, method="POST")
        delay   = DEFAULT_RETRY_DELAY
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read().decode())
            except (urllib.error.URLError, ConnectionRefusedError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning("Ollama retry %d/%d (%s)", attempt + 1, max_retries, exc)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(
                    f"Ollama not reachable at {BASE_URL}. Is it running?"
                ) from exc
            except Exception as exc:
                raise
        raise last_exc  # type: ignore[misc]

    def parse_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        message     = raw.get("message", {})
        content     = message.get("content", "")
        stop_reason = "end_turn" if raw.get("done") else "max_tokens"

        tokens_in  = raw.get("prompt_eval_count", 0)
        tokens_out = raw.get("eval_count", 0)

        content_blocks = [{"type": "text", "text": content}] if content else []

        return self._standard_response(
            content        = content,
            tokens_in      = tokens_in,
            tokens_out     = tokens_out,
            stop_reason    = stop_reason,
            content_blocks = content_blocks,
        )

    def validate_key(self) -> bool:
        # No key required for local Ollama
        return True

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model_id, "prompt": text}
        body    = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        req     = urllib.request.Request(EMBED_ENDPOINT, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("embedding", [])
