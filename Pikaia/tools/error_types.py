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

Classification approach
~~~~~~~~~~~~~~~~~~~~~~~
We check both the **exception type** (preferred — unambiguous) and the
**message string** (fallback — catches provider-specific error text).
Message checks use specific phrases rather than short substrings to avoid
false positives (e.g. bare ``"connection"`` also appears in unrelated
messages such as "no connection between concepts").
"""

from __future__ import annotations

import socket
import urllib.error
from enum import Enum, auto


class ErrorType(Enum):
    RATE_LIMIT       = auto()
    AUTH             = auto()
    CONTEXT_OVERFLOW = auto()
    NETWORK          = auto()
    TOOL             = auto()
    UNKNOWN          = auto()


# ---------------------------------------------------------------------------
# Message-based matchers — use explicit, specific phrases only.
# Deliberately avoided: bare "connection", "error", "failed", "key"
# ---------------------------------------------------------------------------

_RATE_LIMIT_PHRASES = frozenset([
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "rate_limit_exceeded",
    "ratelimit",
    "quota exceeded",
    "requests per minute",
])

_AUTH_PHRASES = frozenset([
    "401",
    "403",
    "authentication",
    "api_key",
    "invalid_api_key",
    "invalid api key",
    "unauthorized",
    "permission denied",
    "access denied",
    "forbidden",
    "not authorized",
    "credentials",
])

_CONTEXT_PHRASES = frozenset([
    "context_length_exceeded",
    "maximum context",
    "context window",
    "tokens exceed",
    "context_window",
    "prompt is too long",
    "reduce your prompt",
    "maximum tokens",
    "max_tokens exceeded",
    "too long",
])

# Use specific multi-word network phrases — NOT bare "connection"
_NETWORK_PHRASES = frozenset([
    "connection refused",
    "connection reset",
    "connection timed out",
    "connection aborted",
    "connection closed",
    "connection error",
    "timed out",
    "timeout",
    "read timeout",
    "write timeout",
    "socket timeout",
    "network unreachable",
    "network error",
    "urlopen error",
    "broken pipe",
    "remote end closed",
    "eof occurred",
    "ssl: eof",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "no route to host",
])


def classify_error(exc: Exception) -> ErrorType:
    """
    Inspect exception type and message and return the matching ErrorType.
    Type checks take priority over message checks.
    Checks are ordered from most-specific to least-specific.
    """
    # ------------------------------------------------------------------
    # Type-based checks (preferred — not fooled by message wording)
    # ------------------------------------------------------------------

    # Network-level errors
    if isinstance(exc, (
        TimeoutError,
        ConnectionError,          # ConnectionRefusedError, ConnectionResetError, …
        socket.timeout,
        socket.gaierror,
        socket.herror,
        urllib.error.URLError,
    )):
        # urllib.error.HTTPError is a subclass of URLError; check its status
        if isinstance(exc, urllib.error.HTTPError):
            code = exc.code or 0
            if code == 429:
                return ErrorType.RATE_LIMIT
            if code in (401, 403):
                return ErrorType.AUTH
            if code >= 500:
                return ErrorType.NETWORK
            # 4xx other than auth → UNKNOWN (don't silently retry)
            return ErrorType.UNKNOWN
        return ErrorType.NETWORK

    # ------------------------------------------------------------------
    # Message-based checks (fallback for provider SDK exceptions that
    # don't subclass a useful built-in)
    # ------------------------------------------------------------------
    msg = str(exc).lower()

    if any(phrase in msg for phrase in _RATE_LIMIT_PHRASES):
        return ErrorType.RATE_LIMIT

    if any(phrase in msg for phrase in _AUTH_PHRASES):
        return ErrorType.AUTH

    if any(phrase in msg for phrase in _CONTEXT_PHRASES):
        return ErrorType.CONTEXT_OVERFLOW

    if any(phrase in msg for phrase in _NETWORK_PHRASES):
        return ErrorType.NETWORK

    return ErrorType.UNKNOWN
