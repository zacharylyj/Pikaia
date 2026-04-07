"""
BaseAdapter
-----------
Abstract contract all provider adapters must implement.

Standard response dict (all adapters must return this from parse_response):
{
    "content":        str,           # concatenated text from all text blocks
    "content_blocks": list[dict],    # raw blocks (text + tool_use) for agent tool loop
    "tokens_in":      int,
    "tokens_out":     int,
    "model_id":       str,
    "provider":       str,
    "stop_reason":    str,
    "tool_calls":     list[dict],    # populated when stop_reason == "tool_use"
}
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """All provider adapters inherit from this."""

    def __init__(self, api_key: str | None, model_id: str) -> None:
        self.api_key  = api_key
        self.model_id = model_id

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_request(
        self,
        system:      str,
        messages:    list[dict],
        max_tokens:  int,
        temperature: float | None,
        tools:       list[dict] | None,
    ) -> dict[str, Any]:
        """Return a provider-specific payload dict ready to POST."""

    @abstractmethod
    def call(self, request: dict[str, Any]) -> Any:
        """POST the request to the provider endpoint. Return raw response."""

    @abstractmethod
    def parse_response(self, raw: Any) -> dict[str, Any]:
        """Normalise raw provider response into the standard response dict."""

    @abstractmethod
    def validate_key(self) -> bool:
        """Return True if api_key appears valid (basic format check, no API call)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self.__class__.__module__.rsplit(".", 1)[-1]

    def _standard_response(
        self,
        content:        str,
        tokens_in:      int,
        tokens_out:     int,
        stop_reason:    str,
        tool_calls:     list[dict] | None = None,
        content_blocks: list[dict] | None = None,
    ) -> dict[str, Any]:
        return {
            "content":        content,
            "content_blocks": content_blocks or [],
            "tokens_in":      tokens_in,
            "tokens_out":     tokens_out,
            "model_id":       self.model_id,
            "provider":       self.provider_name,
            "stop_reason":    stop_reason,
            "tool_calls":     tool_calls or [],
        }
