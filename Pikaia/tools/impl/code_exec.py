"""
code_exec
---------
Run Python or JavaScript in a subprocess using the host runtime.

Security model
~~~~~~~~~~~~~~
Unlike shell_exec this tool does NOT use ``shell=True``, so the risk surface
is narrower (no shell injection from the cmd string itself).  However, the
spawned process runs with the **same user account and filesystem access** as
the Pikaia process, so:

  * It can read/write arbitrary files (project dir, home dir, /tmp, …).
  * It can import any installed package, make network calls, spawn children.
  * It inherits the parent's environment variables (API keys, etc.).

For untrusted or agent-generated code you should run Pikaia inside a
container (Docker / Podman) or use a proper sandbox (e.g. PyPy sandbox,
nsjail, gVisor).  The ``code_exec_enabled`` config key lets operators
disable this tool entirely when security policy requires it.

Config keys (config.json):
    code_exec_enabled : bool   - set to false to disable this tool globally
                                 (default: true)
    code_exec_timeout : int    - default timeout in seconds (default: 10)
    code_exec_max_output_bytes : int - truncate stdout/stderr to this many
                                 bytes (default: 65536)

params:
    code     : str                    - source code to execute
    language : "python" | "js"        - runtime (default: "python")
    timeout  : int | None             - seconds before kill (overrides config)

returns:
    stdout     : str
    stderr     : str
    returncode : int
    timed_out  : bool
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT    = 10
_DEFAULT_MAX_OUTPUT = 65_536   # 64 KiB per stream


def _load_config(base_path: Path) -> dict:
    cfg_path = base_path / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            pass
    return {}


def run(params: dict, context: dict) -> dict[str, Any]:
    base_path = Path(context["base_path"])
    config    = _load_config(base_path)

    # Operator kill-switch
    if not config.get("code_exec_enabled", True):
        return {
            "stdout":     "",
            "stderr":     "[code_exec DISABLED] code_exec_enabled is false in config.json",
            "returncode": 1,
            "timed_out":  False,
        }

    code     = params["code"]
    language = params.get("language", "python").lower()
    timeout  = params.get("timeout") or config.get("code_exec_timeout", _DEFAULT_TIMEOUT)
    max_out  = config.get("code_exec_max_output_bytes", _DEFAULT_MAX_OUTPUT)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py" if language == "python" else ".js",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        if language == "python":
            cmd = [sys.executable, tmp_path]
        elif language in ("js", "javascript", "node"):
            cmd = ["node", tmp_path]
        else:
            return {
                "stdout":     "",
                "stderr":     f"Unsupported language: {language}",
                "returncode": 1,
                "timed_out":  False,
            }

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                # No shell=True — cmd is a list, not a string
            )
            stdout = result.stdout
            stderr = result.stderr
            # Truncate runaway output so the agent context window isn't blown
            if len(stdout) > max_out:
                stdout = stdout[:max_out] + f"\n[...truncated at {max_out} bytes]"
            if len(stderr) > max_out:
                stderr = stderr[:max_out] + f"\n[...truncated at {max_out} bytes]"
            return {
                "stdout":     stdout,
                "stderr":     stderr,
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
        except FileNotFoundError:
            return {
                "stdout":     "",
                "stderr":     f"Runtime not found for language '{language}'. Is it installed?",
                "returncode": 127,
                "timed_out":  False,
            }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
