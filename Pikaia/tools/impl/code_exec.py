"""
code_exec
---------
Run Python or JavaScript in an isolated sandbox subprocess.

params:
    code     : str                    - source code to execute
    language : "python" | "js"        - runtime (default: "python")
    timeout  : int | None             - seconds before kill (default: 10)

returns:
    stdout     : str
    stderr     : str
    returncode : int
    timed_out  : bool
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import os
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    code     = params["code"]
    language = params.get("language", "python").lower()
    timeout  = params.get("timeout", 10)

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
                # No shell=True for safety in sandbox mode
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
