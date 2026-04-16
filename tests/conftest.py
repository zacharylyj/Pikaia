"""
Shared pytest fixtures for the Pikaia test suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make sure the repo root is on sys.path so ``import Pikaia`` works from any
# working directory, including CI runners that checkout to an arbitrary path.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture()
def tmp_base(tmp_path: Path) -> Path:
    """
    A temporary directory that mimics the Pikaia base_path layout:
        <tmp>/memory/
        <tmp>/projects/
        <tmp>/config.json  (empty config)
    """
    (tmp_path / "memory").mkdir()
    (tmp_path / "projects").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({}))
    return tmp_path


@pytest.fixture()
def base_context(tmp_base: Path) -> dict:
    """Minimal context dict expected by tool run() functions."""
    return {
        "base_path":   str(tmp_base),
        "project":     "test_project",
        "instance_id": "test_instance",
        "caller":      "orchestrator",
    }
