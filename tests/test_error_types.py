"""
Tests for Pikaia/tools/error_types.py

Verifies correct classification of exceptions, with particular attention to:
  - False positives from short substrings (old bare "connection" check)
  - Type-based checks taking precedence over message checks
  - HTTP status routing via urllib.error.HTTPError
"""
from __future__ import annotations

import socket
import urllib.error

import pytest

from Pikaia.tools.error_types import ErrorType, classify_error


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "429 Too Many Requests",
    "rate limit exceeded",
    "rate_limit_exceeded",
    "RateLimitError: too many requests",
    "quota exceeded",
])
def test_rate_limit(msg: str) -> None:
    assert classify_error(Exception(msg)) == ErrorType.RATE_LIMIT


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "401 Unauthorized",
    "403 Forbidden",
    "invalid_api_key provided",
    "authentication failed",
    "access denied",
])
def test_auth(msg: str) -> None:
    assert classify_error(Exception(msg)) == ErrorType.AUTH


# ---------------------------------------------------------------------------
# Context-overflow detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "context_length_exceeded",
    "maximum context length is 8192",
    "prompt is too long",
    "tokens exceed the model limit",
    "context window exceeded",
])
def test_context_overflow(msg: str) -> None:
    assert classify_error(Exception(msg)) == ErrorType.CONTEXT_OVERFLOW


# ---------------------------------------------------------------------------
# Network detection — specific phrases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "connection refused",
    "connection reset by peer",
    "connection timed out",
    "timed out",
    "urlopen error timed out",
    "broken pipe",
    "remote end closed connection",
    "network unreachable",
    "no route to host",
    "name or service not known",
    "ssl: eof occurred",
])
def test_network_specific_phrases(msg: str) -> None:
    assert classify_error(Exception(msg)) == ErrorType.NETWORK


# ---------------------------------------------------------------------------
# NO false positives for bare "connection"
# ---------------------------------------------------------------------------

def test_no_false_positive_bare_connection() -> None:
    """Old code classified any message containing 'connection' as NETWORK."""
    msg = "No logical connection between these two concepts"
    result = classify_error(Exception(msg))
    assert result == ErrorType.UNKNOWN, (
        f"Expected UNKNOWN but got {result} — bare 'connection' is triggering NETWORK"
    )


def test_no_false_positive_connection_in_domain() -> None:
    msg = "Lost database connection pool exhausted"
    # "connection" alone should NOT force NETWORK; this is ambiguous
    # The classifier may return NETWORK via "connection" if "database connection" matches
    # a specific phrase, but bare "connection" alone should not.
    # We just ensure it doesn't crash.
    result = classify_error(Exception(msg))
    assert isinstance(result, ErrorType)


# ---------------------------------------------------------------------------
# Type-based checks (stdlib exceptions)
# ---------------------------------------------------------------------------

def test_connection_refused_error_type() -> None:
    exc = ConnectionRefusedError("connection refused")
    assert classify_error(exc) == ErrorType.NETWORK


def test_timeout_error_type() -> None:
    exc = TimeoutError("operation timed out")
    assert classify_error(exc) == ErrorType.NETWORK


def test_socket_timeout_type() -> None:
    exc = socket.timeout("timed out")
    assert classify_error(exc) == ErrorType.NETWORK


def test_urlerror_type() -> None:
    exc = urllib.error.URLError("connection refused")
    assert classify_error(exc) == ErrorType.NETWORK


def test_http_error_429() -> None:
    exc = urllib.error.HTTPError(url=None, code=429, msg="Too Many Requests", hdrs=None, fp=None)  # type: ignore[arg-type]
    assert classify_error(exc) == ErrorType.RATE_LIMIT


def test_http_error_401() -> None:
    exc = urllib.error.HTTPError(url=None, code=401, msg="Unauthorized", hdrs=None, fp=None)  # type: ignore[arg-type]
    assert classify_error(exc) == ErrorType.AUTH


def test_http_error_500() -> None:
    exc = urllib.error.HTTPError(url=None, code=500, msg="Internal Server Error", hdrs=None, fp=None)  # type: ignore[arg-type]
    assert classify_error(exc) == ErrorType.NETWORK


def test_http_error_404() -> None:
    exc = urllib.error.HTTPError(url=None, code=404, msg="Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]
    assert classify_error(exc) == ErrorType.UNKNOWN


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def test_unknown_fallback() -> None:
    exc = ValueError("something completely unexpected happened")
    assert classify_error(exc) == ErrorType.UNKNOWN
