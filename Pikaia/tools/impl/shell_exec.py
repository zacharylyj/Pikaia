"""
shell_exec
----------
Run a shell command in a subprocess.

Security model
~~~~~~~~~~~~~~
shell=True is unavoidable for multi-stage pipelines (pipes, redirection,
glob expansion).  We mitigate the risk in two layers:

1. **Blocklist** – a set of regex patterns that match obviously dangerous
   commands (rm -rf, sudo, curl|sh, env-var dumps, fork bombs, etc.).
   Any match returns an error without execution.

2. **Optional allowlist** – if ``shell_exec_allowlist`` is set in the
   project config.json, *only* commands whose stripped text starts with
   one of the listed prefixes are allowed (the blocklist is still applied
   on top).

For full isolation (multi-tenant / untrusted agents) you should run
Pikaia inside a Docker container or a seccomp-restricted sandbox.  No
in-process filter replaces OS-level isolation.

params:
    cmd     : str           - shell command to run
    cwd     : str | None    - working directory (default: base_path)
    timeout : int | None    - seconds before kill (default: 30)
    env     : dict | None   - extra environment variables

returns:
    stdout     : str
    stderr     : str
    returncode : int
    timed_out  : bool
    blocked    : bool   (present and True only when a command is blocked)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety patterns — ordered from most destructive to least.
# Each pattern is matched case-insensitively against the full command string.
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # Recursive / destructive filesystem ops
    (r"rm\s+(?:[^\s]*\s+)*-[^\s]*r",     "recursive rm"),
    (r"\bdd\b.*\bif\s*=",                 "dd block device write"),
    (r"\bmkfs\b",                          "filesystem format"),
    (r"\bshred\b",                         "secure file wipe"),
    (r"\bfdisk\b|\bparted\b",             "disk partition tool"),

    # Privilege escalation
    (r"\bsudo\b",                          "sudo privilege escalation"),
    (r"\bsu\s+",                           "su user switch"),
    (r"\bchown\s+.*\s+/",                  "chown on root path"),
    (r"chmod\s+[0-7]*[2367][0-7]*\s+/",   "chmod on root path"),

    # Remote-code execution via pipe
    (r"\bcurl\b[^|]*\|[^|]*\b(ba?sh|sh|zsh|ksh|csh|fish|python\d*|perl|ruby|node)\b",
     "curl pipe to shell/interpreter"),
    (r"\bwget\b[^|]*\|[^|]*\b(ba?sh|sh|zsh|ksh|csh|fish|python\d*|perl|ruby|node)\b",
     "wget pipe to shell/interpreter"),
    (r"\bfetch\b[^|]*\|[^|]*\b(ba?sh|sh|zsh)",
     "fetch pipe to shell"),

    # Credential / secret exfiltration
    (r"\bprintenv\b",                      "environment dump"),
    (r"\benv\b\s*$",                       "bare env dump"),
    (r"\bset\b\s*$",                       "bare set dump"),
    (r"echo\s+['\"]?\$\{?(AWS_|GITHUB_|ANTHROPIC_|OPENAI_|SECRET|TOKEN|KEY|PASS|PRIVATE|CREDENTIAL)",
     "secret variable echo"),
    (r"\bcat\b[^|]*/(etc/passwd|etc/shadow|etc/sudoers|\.ssh/id_|\.aws/credentials)",
     "sensitive file read"),

    # eval with variable expansion
    (r"\beval\b.*\$\(",                    "eval with command substitution"),

    # Fork bomb
    (r":\s*\(\s*\)\s*\{",                 "fork bomb"),
    (r"while\s+true.*fork|fork.*while\s+true",
     "fork loop"),

    # /dev/null writes that could be legitimate but /dev/sda etc. are not
    (r">\s*/dev/(?!null\b)",              "write to device file"),
]

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.IGNORECASE), label)
    for pat, label in _DANGEROUS_PATTERNS
]


def _load_config(base_path: Path) -> dict:
    cfg_path = base_path / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            pass
    return {}


def _check_safety(cmd: str, config: dict) -> tuple[bool, str]:
    """
    Returns (allowed: bool, reason: str).
    ``reason`` is non-empty only when the command is blocked.
    """
    # 1. Allowlist check (if configured)
    allowlist: list[str] | None = config.get("shell_exec_allowlist")
    if allowlist is not None:
        stripped = cmd.strip()
        if not any(stripped.startswith(prefix) for prefix in allowlist):
            return False, (
                f"Command not in shell_exec_allowlist: {cmd[:80]!r}. "
                "Set shell_exec_allowlist in config.json or remove the allowlist "
                "to allow all non-blocked commands."
            )

    # 2. Blocklist check (always applied)
    for pattern, label in _COMPILED:
        if pattern.search(cmd):
            return False, f"Command blocked by safety filter [{label}]: {cmd[:80]!r}"

    return True, ""


def run(params: dict, context: dict) -> dict[str, Any]:
    cmd     = params["cmd"]
    cwd     = params.get("cwd") or context["base_path"]
    timeout = params.get("timeout", 30)
    env_extra: dict = params.get("env") or {}
    env = {**os.environ, **env_extra} if env_extra else None

    base_path = Path(context["base_path"])
    config    = _load_config(base_path)

    # Safety gate — runs before any subprocess is created
    allowed, reason = _check_safety(cmd, config)
    if not allowed:
        logger.warning("shell_exec blocked: %s", reason)
        return {
            "stdout":     "",
            "stderr":     f"[shell_exec BLOCKED] {reason}",
            "returncode": 1,
            "timed_out":  False,
            "blocked":    True,
        }

    try:
        result = subprocess.run(
            cmd,
            shell=True,   # required for pipes / redirection / globs
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
            "timed_out":  False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "stdout":     exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr":     exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            "returncode": -1,
            "timed_out":  True,
        }
