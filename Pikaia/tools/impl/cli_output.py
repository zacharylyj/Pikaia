"""
cli_output
----------
Print a formatted response to the terminal — orchestrator only.

params:
    content : str
    type    : "response" | "warning" | "error" | "prompt"  (default: "response")

returns:
    {} (side-effect only)
"""

from __future__ import annotations

import sys
from typing import Any

# ANSI colour codes (disabled on non-TTY)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"

_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOUR else text


_PREFIX = {
    "response": "",
    "warning":  _c(_YELLOW, "⚠ WARNING: "),
    "error":    _c(_RED,    "✗ ERROR: "),
    "prompt":   _c(_CYAN,   "? "),
}

_DIVIDERS = {
    "response": _c(_GREEN, "─" * 60),
    "warning":  "",
    "error":    "",
    "prompt":   "",
}


def run(params: dict, context: dict) -> dict[str, Any]:
    caller = context.get("caller", "orchestrator")
    if caller != "orchestrator":
        raise PermissionError("cli_output is restricted to the orchestrator")

    content  = params.get("content", "")
    msg_type = params.get("type", "response")

    prefix  = _PREFIX.get(msg_type, "")
    divider = _DIVIDERS.get(msg_type, "")

    if divider:
        print(divider)
    print(f"{prefix}{content}")
    if divider:
        print(divider)

    sys.stdout.flush()
    return {}
