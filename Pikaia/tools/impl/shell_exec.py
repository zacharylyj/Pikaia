"""
shell_exec
----------
Run any shell command in a subprocess.

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
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    cmd     = params["cmd"]
    cwd     = params.get("cwd") or context["base_path"]
    timeout = params.get("timeout", 30)
    env_extra: dict = params.get("env") or {}
    env = {**os.environ, **env_extra} if env_extra else None

    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            shell=True,
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
