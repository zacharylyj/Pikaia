"""
error_types.py
--------------
Error classification for LLM and tool call failures.

Used by _tool_loop to route exceptions to appropriate handlers instead of
a single catch-all. Each ErrorType maps to a distinct recovery strategy:

    RATE_LIMIT       → retry with exponential backoff
    AUTH             → surface to caller (bad key, not retryable)
    CONTEXT_OVERFLOW → compress conversation history and retry
    NETWORK          → retry with backoff (transient)
    TOOL             → log and continue (tool errors should not abort the loop)
    UNKNOWN          → log and break the loop

Default retry/backoff limits are defined in config.json (error_retry_max,
error_retry_base_delay). The orchestrator can patch these per task via the
agent context.
"""

from __future__ import annotations

from enum import Enum, auto


class ErrorType(Enum):
    RATE_LIMIT       = auto()
    AUTH             = auto()
    CONTEXT_OVERFLOW = auto()
    NETWORK          = auto()
    TOOL             = auto()
    UNKNOWN          = auto()


def classify_error(exc: Exception) -> ErrorType:
    """
    Inspect exception message/type and return the matching ErrorType.
    Checks are ordered from most-specific to least-specific.
    """
    msg = str(exc).lower()

    # Rate limit: HTTP 429 or provider-specific messages
    if any(x in msg for x in ("429", "rate limit", "too many requests", "rate_limit_exceeded")):
        return ErrorType.RATE_LIMIT

    # Auth: HTTP 401/403 or key-related messages
    if any(x in msg for x in ("401", "403", "authentication", "api key", "api_key",
                               "unauthorized", "invalid_api_key", "permission denied")):
        return ErrorType.AUTH

    # Context overflow: token/context window exceeded
    if any(x in msg for x in ("context_length_exceeded", "maximum context", "context window",
                               "too long", "tokens exceed", "context_window", "prompt is too long",
                               "reduce your prompt")):
        return ErrorType.CONTEXT_OVERFLOW

    # Network: transient connectivity issues
    if any(x in msg for x in ("timeout", "timed out", "connection", "socket", "network",
                               "unreachable", "urlopen error", "connection refused",
                               "broken pipe", "remote end closed")):
        return ErrorType.NETWORK

    return ErrorType.UNKNOWN
