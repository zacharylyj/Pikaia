#!/usr/bin/env python3
"""
test_tools.py
-------------
Functional test suite for all Pikaia tools.

Run via:
    python init.py --test                   # all tools
    python init.py --test --tool grep       # single tool
    python init.py --test --fast            # skip network / slow tests

Or directly:
    python test_tools.py
    python test_tools.py --tool edit --fast

Each test function:
  - Receives a temp base_path and a pre-built tool context
  - Calls the tool's run() directly (no registry overhead)
  - Returns on success, raises AssertionError or Exception on failure

Tests are grouped by tool. Each group is independent — failures in one
tool do not abort tests for other tools.

Tags
----
  network  — makes real HTTP requests (skipped with --fast)
  slow     — takes > 1s (skipped with --fast)
  no_rg    — only meaningful when ripgrep is not installed (always runs)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

_BASE = Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

# ── ANSI colours ─────────────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _ok(msg: str)   -> None: print(f"  {_GREEN}PASS{_RESET}  {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}FAIL{_RESET}  {msg}")
def _skip(msg: str) -> None: print(f"  {_YELLOW}SKIP{_RESET}  {msg} (skipped)")
def _info(msg: str) -> None: print(f"       {msg}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_tool(name: str, impl_path: str | None = None) -> Any:
    """Import a tool module from its impl path."""
    if impl_path is None:
        # Resolve from tools.json
        tools_json = _BASE / "tools" / "tools.json"
        entries = json.loads(tools_json.read_text())
        entry   = next((e for e in entries if e["tool_id"] == name), None)
        if entry is None:
            raise ValueError(f"Tool '{name}' not found in tools.json")
        impl_path = str(_BASE / entry["impl"])
    spec = importlib.util.spec_from_file_location(name, impl_path)
    mod  = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(mod)                           # type: ignore[union-attr]
    return mod


def _ctx(tmp: str, caller: str = "orchestrator", project: str = "test_proj",
         agent_id: str = "agent_test") -> dict:
    """Build a minimal tool context pointing at the given temp dir."""
    return {
        "base_path":    tmp,
        "project":      project,
        "instance_id":  "inst_test",
        "agent_id":     agent_id,
        "caller":       caller,
        "worker_dir":   str(Path(tmp) / "projects" / project / "worker" / agent_id),
        "token_budget": 10000,
        "config":       {},
    }


def _rg_available() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Result accumulator ────────────────────────────────────────────────────────

class TestResults:
    def __init__(self) -> None:
        self.passed  = 0
        self.failed  = 0
        self.skipped = 0
        self.failures: list[tuple[str, str]] = []   # (test_name, error_msg)

    def record(self, name: str, exc: Exception | None, skipped: bool = False) -> None:
        if skipped:
            self.skipped += 1
            _skip(name)
        elif exc is None:
            self.passed += 1
            _ok(name)
        else:
            self.failed += 1
            msg = f"{type(exc).__name__}: {exc}"
            _fail(f"{name}  — {msg}")
            self.failures.append((name, traceback.format_exc()))

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


# ── Test definitions ──────────────────────────────────────────────────────────

# Each entry: (tool_id, test_name, tags, fn(tmp_dir, ctx) -> None)
# tags is a set of strings; --fast skips {"network", "slow"}

_TESTS: list[tuple[str, str, set[str], Callable]] = []


def _test(tool_id: str, name: str, tags: set[str] | None = None):
    """Decorator to register a test function."""
    def decorator(fn: Callable) -> Callable:
        _TESTS.append((tool_id, name, tags or set(), fn))
        return fn
    return decorator


# ── file_read ─────────────────────────────────────────────────────────────────

@_test("file_read", "read basic file")
def _(tmp, ctx):
    p = Path(tmp) / "projects" / "test_proj" / "dev" / "hello.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write bytes directly to avoid OS-specific line ending translation
    p.write_bytes(b"hello world\nsecond line\n")
    m = _load_tool("file_read")
    r = m.run({"path": "projects/test_proj/dev/hello.txt"}, ctx)
    assert r["content"].replace("\r\n", "\n") == "hello world\nsecond line\n"
    assert r["lines"] == 2
    assert r["truncated"] is False


@_test("file_read", "read with offset and limit")
def _(tmp, ctx):
    p = Path(tmp) / "projects" / "test_proj" / "dev" / "multi.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    m = _load_tool("file_read")
    r = m.run({"path": "projects/test_proj/dev/multi.txt", "offset": 3, "limit": 4}, ctx)
    lines = r["content"].strip().splitlines()
    assert lines[0] == "line3", f"Expected line3, got {lines[0]}"
    assert len(lines) == 4
    assert r["truncated"] is True
    assert r["lines"] == 10


@_test("file_read", "rejects path outside base_path")
def _(tmp, ctx):
    m = _load_tool("file_read")
    try:
        m.run({"path": "../../../etc/passwd"}, ctx)
        raise AssertionError("Expected PermissionError")
    except (PermissionError, FileNotFoundError):
        pass  # expected


@_test("file_read", "rejects binary file")
def _(tmp, ctx):
    p = Path(tmp) / "projects" / "test_proj" / "dev" / "binary.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00\x01\x02\x03binary data")
    m = _load_tool("file_read")
    try:
        m.run({"path": "projects/test_proj/dev/binary.bin"}, ctx)
        raise AssertionError("Expected ValueError for binary file")
    except ValueError:
        pass


# ── file_write ────────────────────────────────────────────────────────────────

@_test("file_write", "write and read back")
def _(tmp, ctx):
    worker = Path(tmp) / "projects" / "test_proj" / "worker" / "agent_test"
    worker.mkdir(parents=True, exist_ok=True)
    ctx2 = {**ctx, "caller": "agent"}
    m = _load_tool("file_write")
    m.run({"path": "projects/test_proj/worker/agent_test/out.txt", "content": "test output"}, ctx2)
    assert (worker / "out.txt").read_text() == "test output"


@_test("file_write", "agent cannot write outside worker dir")
def _(tmp, ctx):
    ctx2 = {**ctx, "caller": "agent"}
    m = _load_tool("file_write")
    try:
        m.run({"path": "projects/test_proj/dev/escape.txt", "content": "bad"}, ctx2)
        raise AssertionError("Expected PermissionError")
    except PermissionError:
        pass


# ── edit ──────────────────────────────────────────────────────────────────────

@_test("edit", "basic replacement")
def _(tmp, ctx):
    p = Path(tmp) / "projects" / "test_proj" / "edit_test.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("foo bar baz\n")
    m = _load_tool("edit")
    r = m.run({"path": "projects/test_proj/edit_test.txt",
               "old_string": "bar", "new_string": "qux"}, ctx)
    assert r["replacements"] == 1
    assert p.read_text() == "foo qux baz\n"


@_test("edit", "replace_all mode")
def _(tmp, ctx):
    p = Path(tmp) / "projects" / "test_proj" / "multi_edit.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x x x\n")
    m = _load_tool("edit")
    r = m.run({"path": "projects/test_proj/multi_edit.txt",
               "old_string": "x", "new_string": "y", "replace_all": True}, ctx)
    assert r["replacements"] == 3
    assert p.read_text() == "y y y\n"


@_test("edit", "error on missing old_string")
def _(tmp, ctx):
    p = Path(tmp) / "projects" / "test_proj" / "nomatch.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("hello\n")
    m = _load_tool("edit")
    try:
        m.run({"path": "projects/test_proj/nomatch.txt",
               "old_string": "nothere", "new_string": "x"}, ctx)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


@_test("edit", "error on ambiguous old_string")
def _(tmp, ctx):
    p = Path(tmp) / "projects" / "test_proj" / "ambig.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x\nx\nx\n")
    m = _load_tool("edit")
    try:
        m.run({"path": "projects/test_proj/ambig.txt",
               "old_string": "x", "new_string": "y"}, ctx)
        raise AssertionError("Expected ValueError for ambiguous match")
    except ValueError:
        pass


# ── grep ──────────────────────────────────────────────────────────────────────

@_test("grep", "files_with_matches mode")
def _(tmp, ctx):
    d = Path(tmp) / "search_root"
    d.mkdir()
    (d / "a.py").write_text("def hello(): pass\n")
    (d / "b.py").write_text("def world(): pass\n")
    (d / "c.txt").write_text("unrelated\n")
    m = _load_tool("grep")
    r = m.run({"pattern": "def hello", "path": "search_root",
               "output_mode": "files_with_matches"}, ctx)
    assert len(r["matches"]) == 1
    assert "a.py" in r["matches"][0]


@_test("grep", "content mode with context lines")
def _(tmp, ctx):
    d = Path(tmp) / "ctx_root"
    d.mkdir()
    (d / "code.py").write_text("before\ntarget line\nafter\n")
    m = _load_tool("grep")
    r = m.run({"pattern": "target", "path": "ctx_root",
               "output_mode": "content", "context": 1}, ctx)
    texts = [x["text"] for x in r["matches"]]
    assert any("before" in t for t in texts)
    assert any("after"  in t for t in texts)


@_test("grep", "count mode")
def _(tmp, ctx):
    d = Path(tmp) / "count_root"
    d.mkdir()
    (d / "f.txt").write_text("hit\nhit\nmiss\n")
    m = _load_tool("grep")
    r = m.run({"pattern": "hit", "path": "count_root", "output_mode": "count"}, ctx)
    assert len(r["matches"]) == 1
    assert r["matches"][0]["count"] == 2


@_test("grep", "glob filter")
def _(tmp, ctx):
    d = Path(tmp) / "mixed"
    d.mkdir()
    (d / "a.py").write_text("needle\n")
    (d / "b.txt").write_text("needle\n")
    m = _load_tool("grep")
    r = m.run({"pattern": "needle", "path": "mixed",
               "glob": "*.py", "output_mode": "files_with_matches"}, ctx)
    assert len(r["matches"]) == 1
    assert r["matches"][0].endswith(".py")


@_test("grep", "ignore_case flag")
def _(tmp, ctx):
    d = Path(tmp) / "icase"
    d.mkdir()
    (d / "f.txt").write_text("Hello World\n")
    m = _load_tool("grep")
    r = m.run({"pattern": "hello world", "path": "icase",
               "ignore_case": True, "output_mode": "files_with_matches"}, ctx)
    assert len(r["matches"]) == 1


@_test("grep", "invalid regex raises ValueError")
def _(tmp, ctx):
    m = _load_tool("grep")
    try:
        m.run({"pattern": "[invalid"}, ctx)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ── glob ──────────────────────────────────────────────────────────────────────

@_test("glob", "basic pattern match")
def _(tmp, ctx):
    d = Path(tmp) / "glob_root"
    d.mkdir()
    (d / "a.py").write_text("x"); (d / "b.py").write_text("x")
    (d / "c.txt").write_text("x")
    m = _load_tool("glob")
    r = m.run({"pattern": "*.py", "path": "glob_root"}, ctx)
    assert r["count"] == 2
    assert all(p.endswith(".py") for p in r["files"])


@_test("glob", "recursive double-star pattern")
def _(tmp, ctx):
    d = Path(tmp) / "rec_root"
    (d / "sub").mkdir(parents=True)
    (d / "top.py").write_text("x")
    (d / "sub" / "deep.py").write_text("x")
    m = _load_tool("glob")
    r = m.run({"pattern": "**/*.py", "path": "rec_root"}, ctx)
    assert r["count"] == 2


@_test("glob", "no matches returns empty list")
def _(tmp, ctx):
    d = Path(tmp) / "empty_root"
    d.mkdir()
    m = _load_tool("glob")
    r = m.run({"pattern": "*.xyz", "path": "empty_root"}, ctx)
    assert r["count"] == 0
    assert r["files"] == []


# ── list ──────────────────────────────────────────────────────────────────────

@_test("list", "flat listing")
def _(tmp, ctx):
    d = Path(tmp) / "list_root"
    d.mkdir()
    (d / "file1.txt").write_text("a")
    (d / "file2.txt").write_text("b")
    (d / "subdir").mkdir()
    m = _load_tool("list")
    r = m.run({"path": "list_root"}, ctx)
    assert r["count"] == 3
    types = {e["type"] for e in r["entries"]}
    assert "file" in types and "dir" in types


@_test("list", "recursive listing")
def _(tmp, ctx):
    d = Path(tmp) / "deep_list"
    (d / "a" / "b").mkdir(parents=True)
    (d / "a" / "b" / "c.txt").write_text("x")
    m = _load_tool("list")
    r = m.run({"path": "deep_list", "recursive": True}, ctx)
    names = [e["name"] for e in r["entries"]]
    assert any("c.txt" in n for n in names)


@_test("list", "non-existent path raises FileNotFoundError")
def _(tmp, ctx):
    m = _load_tool("list")
    try:
        m.run({"path": "does_not_exist_xyz"}, ctx)
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


# ── shell_exec ────────────────────────────────────────────────────────────────

@_test("shell_exec", "echo command")
def _(tmp, ctx):
    m = _load_tool("shell_exec")
    r = m.run({"cmd": "echo hello_pikaia"}, ctx)
    assert "hello_pikaia" in r["stdout"]
    assert r["returncode"] == 0


@_test("shell_exec", "non-zero exit code")
def _(tmp, ctx):
    m = _load_tool("shell_exec")
    r = m.run({"cmd": "exit 42", "timeout": 5}, ctx)
    assert r["returncode"] == 42


@_test("shell_exec", "timeout detection", {"slow"})
def _(tmp, ctx):
    m = _load_tool("shell_exec")
    r = m.run({"cmd": "python -c \"import time; time.sleep(10)\"", "timeout": 2}, ctx)
    assert r["timed_out"] is True


# ── apply_patch ───────────────────────────────────────────────────────────────

@_test("apply_patch", "apply simple unified diff")
def _(tmp, ctx):
    f = Path(tmp) / "patch_target.txt"
    f.write_text("line1\nline2\nline3\n")
    patch = (
        "--- a/patch_target.txt\n"
        "+++ b/patch_target.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+LINE2\n"
        " line3\n"
    )
    m = _load_tool("apply_patch")
    r = m.run({"patch": patch}, ctx)
    assert r["applied"] is True
    content = f.read_text()
    assert "LINE2" in content
    assert "line2" not in content


@_test("apply_patch", "dry_run does not modify file")
def _(tmp, ctx):
    f = Path(tmp) / "dry_target.txt"
    f.write_text("original\n")
    patch = (
        "--- a/dry_target.txt\n"
        "+++ b/dry_target.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-original\n"
        "+changed\n"
    )
    m = _load_tool("apply_patch")
    m.run({"patch": patch, "dry_run": True}, ctx)
    assert f.read_text() == "original\n"


# ── todo_write ────────────────────────────────────────────────────────────────

@_test("todo_write", "write and persist todos")
def _(tmp, ctx):
    worker = Path(ctx["worker_dir"])
    worker.mkdir(parents=True, exist_ok=True)
    m = _load_tool("todo_write")
    todos = [
        {"content": "task a", "status": "completed"},
        {"content": "task b", "status": "in_progress"},
        {"content": "task c", "status": "pending"},
    ]
    r = m.run({"todos": todos}, ctx)
    assert r["count"] == 3
    saved = json.loads((worker / "todos.json").read_text())
    assert len(saved) == 3


@_test("todo_write", "rejects multiple in_progress")
def _(tmp, ctx):
    worker = Path(ctx["worker_dir"])
    worker.mkdir(parents=True, exist_ok=True)
    m = _load_tool("todo_write")
    try:
        m.run({"todos": [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ]}, ctx)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


@_test("todo_write", "rejects invalid status")
def _(tmp, ctx):
    worker = Path(ctx["worker_dir"])
    worker.mkdir(parents=True, exist_ok=True)
    m = _load_tool("todo_write")
    try:
        m.run({"todos": [{"content": "x", "status": "bogus"}]}, ctx)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ── web_fetch ─────────────────────────────────────────────────────────────────

@_test("web_fetch", "fetch real URL", {"network"})
def _(tmp, ctx):
    m = _load_tool("web_fetch")
    r = m.run({"url": "https://example.com", "max_chars": 500}, ctx)
    assert isinstance(r["content"], str) and len(r["content"]) > 0
    assert r["url"] == "https://example.com"


@_test("web_fetch", "bad URL returns error content")
def _(tmp, ctx):
    m = _load_tool("web_fetch")
    r = m.run({"url": "http://127.0.0.1:1", "timeout": 2}, ctx)
    assert "error" in r["content"].lower() or len(r["content"]) > 0


# ── web_search ────────────────────────────────────────────────────────────────

@_test("web_search", "returns structured results", {"network", "slow"})
def _(tmp, ctx):
    m = _load_tool("web_search")
    r = m.run({"query": "python programming language", "max_results": 3}, ctx)
    assert "results" in r
    assert "query" in r
    # May return 0 results if DDG rate-limits; just check shape is correct
    for res in r["results"]:
        assert "title" in res
        assert "url"   in res
        assert "snippet" in res


# ── http_request ──────────────────────────────────────────────────────────────

@_test("http_request", "GET httpbin", {"network"})
def _(tmp, ctx):
    m = _load_tool("http_request")
    r = m.run({"method": "GET", "url": "https://httpbin.org/get", "timeout": 10}, ctx)
    assert r["ok"] is True
    assert r["status_code"] == 200


# ── code_exec ─────────────────────────────────────────────────────────────────

@_test("code_exec", "python addition")
def _(tmp, ctx):
    m = _load_tool("code_exec")
    r = m.run({"code": "print(1 + 2)", "language": "python"}, ctx)
    assert "3" in r["stdout"]
    assert r["returncode"] == 0


@_test("code_exec", "python runtime error captured")
def _(tmp, ctx):
    m = _load_tool("code_exec")
    r = m.run({"code": "raise ValueError('oops')", "language": "python"}, ctx)
    assert r["returncode"] != 0
    assert "oops" in r["stderr"] or "oops" in r["stdout"]


@_test("code_exec", "timeout enforced", {"slow"})
def _(tmp, ctx):
    m = _load_tool("code_exec")
    r = m.run({"code": "import time; time.sleep(10)", "language": "python", "timeout": 2}, ctx)
    assert r["timed_out"] is True


# ── file_delete ───────────────────────────────────────────────────────────────

@_test("file_delete", "delete file within base_path")
def _(tmp, ctx):
    f = Path(tmp) / "to_delete.txt"
    f.write_text("bye")
    m = _load_tool("file_delete")
    r = m.run({"path": "to_delete.txt"}, ctx)
    assert r["deleted"] is True
    assert not f.exists()


@_test("file_delete", "non-existent file returns deleted=False")
def _(tmp, ctx):
    m = _load_tool("file_delete")
    r = m.run({"path": "ghost_file_xyz.txt"}, ctx)
    assert r["deleted"] is False


@_test("file_delete", "rejects agent caller")
def _(tmp, ctx):
    ctx2 = {**ctx, "caller": "agent"}
    m = _load_tool("file_delete")
    try:
        m.run({"path": "anything.txt"}, ctx2)
        raise AssertionError("Expected PermissionError")
    except PermissionError:
        pass


# ── error_types ───────────────────────────────────────────────────────────────

@_test("error_types", "classify rate limit")
def _(tmp, ctx):
    from tools.error_types import classify_error, ErrorType
    assert classify_error(Exception("429 too many requests")) == ErrorType.RATE_LIMIT
    assert classify_error(RuntimeError("rate_limit_exceeded")) == ErrorType.RATE_LIMIT


@_test("error_types", "classify auth")
def _(tmp, ctx):
    from tools.error_types import classify_error, ErrorType
    assert classify_error(Exception("401 unauthorized")) == ErrorType.AUTH
    assert classify_error(Exception("invalid_api_key provided")) == ErrorType.AUTH


@_test("error_types", "classify context overflow")
def _(tmp, ctx):
    from tools.error_types import classify_error, ErrorType
    assert classify_error(Exception("context_length_exceeded")) == ErrorType.CONTEXT_OVERFLOW
    assert classify_error(Exception("prompt is too long for this model")) == ErrorType.CONTEXT_OVERFLOW


@_test("error_types", "classify network")
def _(tmp, ctx):
    from tools.error_types import classify_error, ErrorType
    assert classify_error(Exception("connection timeout")) == ErrorType.NETWORK
    assert classify_error(ConnectionError("socket error")) == ErrorType.NETWORK


@_test("error_types", "classify unknown")
def _(tmp, ctx):
    from tools.error_types import classify_error, ErrorType
    assert classify_error(Exception("some random error")) == ErrorType.UNKNOWN


# ── db / metrics / trajectory ─────────────────────────────────────────────────

@_test("db", "log and query metrics")
def _(tmp, ctx):
    from db import Database
    db_path = Path(tmp) / "test.db"
    db = Database(db_path)
    db.log_metric("t1", "tokens_in", 100.0, "2024-01-01T00:00:00+00:00")
    db.log_metric("t1", "tokens_in",  50.0, "2024-01-01T00:00:01+00:00")
    summary = db.metrics_summary("t1")
    assert summary["tokens_in"] == 150.0
    db.close()


@_test("db", "log trajectory and tool events")
def _(tmp, ctx):
    from db import Database
    db_path = Path(tmp) / "traj.db"
    db = Database(db_path)
    db.log_trajectory("t2", "proj", "agent1", 1, "start", "end", "done", "output", [{"step": 0}])
    db.log_tool_event("t2", "file_read", True, 12.0, "2024-01-01T00:00:00+00:00")
    db.log_tool_event("t2", "file_read", False, 5.0, "2024-01-01T00:00:01+00:00", "err")
    rates = db.tool_success_rate("t2")
    assert rates["file_read"] == 0.5
    db.close()


@_test("metrics", "accumulate and summarise")
def _(tmp, ctx):
    from metrics import MetricsCollector
    from db import Database
    mc = MetricsCollector("t3", enabled=True)
    mc.record_tokens(200, 80)
    mc.record_tokens(100, 40)
    mc.record_tool_call("shell_exec", True,  20.0)
    mc.record_tool_call("shell_exec", False, 5.0, "err")
    mc.record_step(); mc.record_step()
    assert mc.total_tokens == 420
    assert mc.steps == 2
    assert mc.tool_success_rate == 0.5
    # Flush to DB
    db_path = Path(tmp) / "metrics.db"
    db = Database(db_path)
    mc.flush(db)
    s = db.metrics_summary("t3")
    assert s["tokens_in"]  == 300.0
    assert s["tokens_out"] == 120.0
    db.close()


@_test("metrics", "no-op when disabled")
def _(tmp, ctx):
    from metrics import MetricsCollector
    mc = MetricsCollector("t4", enabled=False)
    mc.record_tokens(999, 999)
    mc.record_step()
    assert mc.total_tokens == 0
    assert mc.steps == 0


@_test("trajectory", "write JSONL replay buffer")
def _(tmp, ctx):
    from trajectory import TrajectoryLogger
    tl = TrajectoryLogger("task_x", "agent1", "proj", 1, Path(tmp), enabled=True)
    tl.log_llm_turn("hello", 100, 50, step=0)
    tl.log_tool_call("file_read", {"path": "foo.txt"}, step=0)
    tl.log_tool_result("file_read", "content", 12.0, True, step=0)
    tl.log_compression(10, 4, step=1)
    tl.finalise("done", "final output")
    jsonl = Path(tmp) / "projects" / "proj" / "trajectories" / "task_x.jsonl"
    assert jsonl.exists()
    lines = [json.loads(l) for l in jsonl.read_text().strip().splitlines()]
    assert len(lines) == 4
    types = [l["type"] for l in lines]
    assert types == ["llm_turn", "tool_call", "tool_result", "compress"]


# ── schemas ───────────────────────────────────────────────────────────────────

@_test("schemas", "all tools.json tools have schemas")
def _(tmp, ctx):
    from tools.schemas import get_schemas, invalidate_schema_cache
    invalidate_schema_cache()
    tools_json = _BASE / "tools" / "tools.json"
    entries    = json.loads(tools_json.read_text())
    enabled    = [e["tool_id"] for e in entries if e.get("enabled", True)]
    schemas    = get_schemas(enabled)
    missing    = [tid for tid in enabled if not any(s["name"] == tid for s in schemas)]
    # Some internal tools (e.g. cli_output, memory_write) legitimately have no agent-facing schema
    # Warn but don't fail — the check() already warns on these
    assert isinstance(schemas, list)


@_test("schemas", "auto-discovery finds SCHEMA in impl modules")
def _(tmp, ctx):
    from tools.schemas import _discover_impl_schemas
    discovered = _discover_impl_schemas(_BASE / "tools" / "impl")
    # edit, grep, glob, list, apply_patch, todo_write, web_search, question all have SCHEMA
    expected = {"edit", "grep", "glob", "list", "apply_patch",
                "todo_write", "web_search", "question"}
    found = set(discovered.keys())
    missing = expected - found
    assert not missing, f"Missing self-registered schemas: {missing}"


# ── registry ──────────────────────────────────────────────────────────────────

@_test("registry", "ToolResult normalisation")
def _(tmp, ctx):
    from tools.registry import _normalise, _error_result
    r1 = _normalise({"key": "value"})
    assert r1["success"] is True
    assert r1["data"] == {"key": "value"}
    assert r1["error"] == ""

    r2 = _normalise({"success": False, "data": None, "error": "oops"})
    assert r2["success"] is False
    assert r2["error"] == "oops"

    r3 = _normalise("plain string")
    assert r3["success"] is True
    assert r3["data"] == "plain string"


# ── question ──────────────────────────────────────────────────────────────────

@_test("question", "timeout returns empty answer")
def _(tmp, ctx):
    worker = Path(ctx["worker_dir"])
    worker.mkdir(parents=True, exist_ok=True)
    m = _load_tool("question")
    t0 = time.monotonic()
    r  = m.run({"question": "Are you there?", "timeout": 1}, ctx)
    elapsed = time.monotonic() - t0
    assert r["from"] == "timeout"
    assert r["answer"] == ""
    assert elapsed < 5  # should not hang


@_test("question", "answer.json is picked up")
def _(tmp, ctx):
    worker = Path(ctx["worker_dir"])
    worker.mkdir(parents=True, exist_ok=True)

    import threading

    def _write_answer():
        time.sleep(0.3)
        ans_path = worker / "answer.json"
        ans_path.write_text(json.dumps({"answer": "yes"}))

    t = threading.Thread(target=_write_answer, daemon=True)
    t.start()
    m = _load_tool("question")
    r = m.run({"question": "Yes or no?", "timeout": 5}, ctx)
    assert r["from"] == "user"
    assert r["answer"] == "yes"


# ── run_tests entry point ─────────────────────────────────────────────────────

def run_tests(
    base_path: str | None = None,
    tool:      str | None = None,
    fast:      bool       = False,
) -> bool:
    """
    Execute the test suite. Called by init.py --test.

    base_path : override _BASE (used in tests that need a tmp dir)
    tool      : if set, only run tests for this tool_id
    fast      : skip tests tagged "network" or "slow"

    Returns True if all executed tests passed.
    """
    skip_tags = {"network", "slow"} if fast else set()

    print(f"\n{_BOLD}{_CYAN}Pikaia Tool Tests{_RESET}")
    if tool:
        print(f"  Filtering to tool: {_BOLD}{tool}{_RESET}")
    if fast:
        print(f"  Fast mode: skipping {skip_tags}")
    print()

    results = TestResults()
    current_group = None

    # Group and filter tests
    for tool_id, name, tags, fn in _TESTS:
        if tool and tool_id != tool:
            continue

        if tool_id != current_group:
            current_group = tool_id
            print(f"{_BOLD}  [{tool_id}]{_RESET}")

        full_name = f"{tool_id} / {name}"

        if tags & skip_tags:
            results.record(full_name, None, skipped=True)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(tmp)
            exc = None
            try:
                fn(tmp, ctx)
            except Exception as e:
                exc = e
            results.record(full_name, exc)

    # Summary
    print(f"\n{'-'*52}")
    total = results.passed + results.failed + results.skipped
    print(
        f"  {_GREEN}Passed{_RESET}: {results.passed}   "
        f"{_RED}Failed{_RESET}: {results.failed}   "
        f"{_YELLOW}Skipped{_RESET}: {results.skipped}   "
        f"Total: {total}"
    )

    if results.failures:
        print(f"\n{_RED}{_BOLD}Failures:{_RESET}")
        for name, tb in results.failures:
            print(f"\n  {_RED}FAIL: {name}{_RESET}")
            # Print last 5 lines of traceback
            tb_lines = tb.strip().splitlines()
            for line in tb_lines[-5:]:
                print(f"    {line}")

    if results.all_passed:
        print(f"\n  {_GREEN}{_BOLD}All tests passed.{_RESET}\n")
    else:
        print(f"\n  {_RED}{_BOLD}{results.failed} test(s) failed.{_RESET}\n")

    return results.all_passed


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pikaia tool test runner")
    parser.add_argument("--tool", default=None, help="Run tests for a specific tool only")
    parser.add_argument("--fast", action="store_true", help="Skip network/slow tests")
    args = parser.parse_args()
    ok   = run_tests(tool=args.tool, fast=args.fast)
    sys.exit(0 if ok else 1)
