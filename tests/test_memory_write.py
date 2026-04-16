"""
Tests for Pikaia/tools/impl/memory_write.py

Covers:
  - ct.json locking under concurrent writes (regression for the race condition)
  - _write_mt no longer double-imports mt_palace on fallback
  - lt / st / ct routing basics
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Make Pikaia importable from any working directory
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Pikaia.tools.impl import memory_write  # noqa: E402
from Pikaia.tools.impl.memory_write import run  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ctx(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "projects" / "proj").mkdir(parents=True, exist_ok=True)
    return {
        "base_path":   str(tmp_path),
        "project":     "proj",
        "instance_id": "inst1",
        "caller":      "orchestrator",
    }


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------

def test_lt_write(ctx: dict, tmp_path: Path) -> None:
    entry = {"id": "e1", "content": "hello"}
    result = run({"layer": "lt", "entry": entry}, ctx)
    assert result["written"] is True
    path = tmp_path / "memory" / "lt.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert any(e.get("id") == "e1" for e in data)


def test_st_write(ctx: dict, tmp_path: Path) -> None:
    entry = {"summary": "some summary", "window": []}
    result = run({"layer": "st", "entry": entry}, ctx)
    assert result["written"] is True
    path = tmp_path / "projects" / "proj" / "instances" / "inst1" / "st.json"
    assert path.exists()


def test_ct_write_and_update(ctx: dict, tmp_path: Path) -> None:
    entry = {"id": "task1", "status": "open", "title": "Do something"}
    run({"layer": "ct", "entry": entry}, ctx)
    # Update existing entry
    updated = {"id": "task1", "status": "done", "title": "Do something"}
    run({"layer": "ct", "entry": updated}, ctx)

    path = tmp_path / "projects" / "proj" / "ct.json"
    data = json.loads(path.read_text())
    matching = [e for e in data if e.get("id") == "task1"]
    assert len(matching) == 1, "Should not duplicate entries with same id"
    assert matching[0]["status"] == "done"


def test_unknown_layer_raises(ctx: dict) -> None:
    with pytest.raises(ValueError, match="Unknown memory layer"):
        run({"layer": "zz", "entry": {}}, ctx)


def test_non_orchestrator_raises(ctx: dict) -> None:
    ctx2 = {**ctx, "caller": "agent"}
    with pytest.raises(PermissionError):
        run({"layer": "lt", "entry": {}}, ctx2)


# ---------------------------------------------------------------------------
# ct.json concurrency — write 50 distinct entries from 10 threads
# ---------------------------------------------------------------------------

def test_ct_concurrent_writes(ctx: dict, tmp_path: Path) -> None:
    """
    10 threads each write 5 distinct CT entries.  After all threads finish,
    all 50 entries must be present (no silent overwrites due to race conditions).
    """
    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        for i in range(5):
            entry = {
                "id":     f"t{thread_id}_e{i}",
                "status": "open",
                "thread": thread_id,
                "seq":    i,
            }
            try:
                run({"layer": "ct", "entry": entry}, ctx)
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent write: {errors}"

    path = tmp_path / "projects" / "proj" / "ct.json"
    data = json.loads(path.read_text())
    assert len(data) == 50, f"Expected 50 entries, got {len(data)}"


# ---------------------------------------------------------------------------
# _write_mt: no double-import when mt_palace is unavailable
# ---------------------------------------------------------------------------

def test_write_mt_fallback_no_double_import(ctx: dict, tmp_path: Path) -> None:
    """
    When mt_palace cannot be imported, _write_mt should fall back to the
    legacy JSON path WITHOUT attempting a second import.
    """
    import_call_count = 0
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None  # type: ignore[union-attr]

    # Track how many times 'mt_palace' is imported
    import_counts: dict[str, int] = {"mt_palace": 0}

    real_import = __import__

    def counting_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mt_palace":
            import_counts["mt_palace"] += 1
            raise ImportError("mt_palace not available (test stub)")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=counting_import):
        entry = {"id": "m1", "content": "test content"}
        memory_write._write_mt(tmp_path, entry, ctx)

    # Should only attempt to import mt_palace ONCE (not twice)
    assert import_counts["mt_palace"] == 1, (
        f"mt_palace was imported {import_counts['mt_palace']} times — "
        "expected exactly 1 (double-import bug re-introduced?)"
    )

    # Legacy path should have written the entry
    path = tmp_path / "memory" / "mt.json"
    assert path.exists(), "Legacy mt.json not created"
    data = json.loads(path.read_text())
    assert any(e.get("id") == "m1" for e in data)
