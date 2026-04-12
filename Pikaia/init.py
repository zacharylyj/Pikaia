#!/usr/bin/env python3
"""
init.py — Bootstrap and integrity checker for AGENT.

Usage:
    python init.py                      # first-time setup wizard
    python init.py --check              # validate structure + JSON + pipelines
    python init.py --fix                # auto-fix recoverable issues
    python init.py --project <name>     # scaffold a specific project (non-interactive)
    python init.py --test               # run functional tool tests (no API key needed)
    python init.py --test --tool grep   # run tests for a specific tool only
    python init.py --test --fast        # skip slow / network-dependent tests
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

_BASE_PATH = Path(__file__).resolve().parent   # Pikaia/


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "default_model":         "claude-sonnet-4-6",
    "compression_model":     "claude-haiku-4-5-20251001",
    "skill_match_threshold": 0.75,
    "promote_threshold":     0.80,
    "ack_confidence_min":    0.80,
    "ack_max_rounds":        2,
    "st_max_messages":       20,
    "mt_top_k":              5,
    "history_rag_top_k":     3,
    "file_summary_top_k":    3,
    "max_files_per_task":    5,
    "retry_limit":           3,
    "skillsmith_dry_runs":   3,
    "skillsmith_pass_score": 0.80,
    "embedding_dim":         1536,
    "interfaces":            ["cli"],

    # ── Tier 1: Core Stability ──────────────────────────────────────────
    # Step budget: hard cap on tool-loop iterations per agent run.
    # Orchestrator may override via task_packet["max_steps"].
    "max_steps":                     15,

    # Context compression: trigger at this fraction of the model context window.
    # Orchestrator may override via task_packet["compression_threshold"].
    "context_compression_threshold": 0.80,

    # Parallel tool execution: max worker threads for independent tool calls.
    # Orchestrator may override via task_packet["parallel_tool_workers"].
    "parallel_tool_max_workers":     4,

    # Error retry policy (applied per ErrorType by _tool_loop).
    # Orchestrator may patch context["error_retry_max"] per task.
    "error_retry_max":               3,
    "error_retry_base_delay":        1.0,   # seconds; doubles on each retry

    # ── Tier 2: Efficiency ──────────────────────────────────────────────
    # Model routing: simple tasks (short prompt, ≤1 tool) are routed here.
    # Set to "" or null to disable routing (always use pipeline model).
    # Orchestrator may override via task_packet["fast_model"].
    "fast_model":                    "claude-haiku-4-5-20251001",
    "fast_model_threshold_words":    50,    # route to fast_model if prompt ≤ N words
    "fast_model_threshold_tools":    1,     # route to fast_model if tools_allowed ≤ N

    # ── Tier 3: Scaling & Data Layer ────────────────────────────────────
    # Trajectory logging: store per-run replay buffer as JSONL + SQLite.
    # Orchestrator may disable for sensitive tasks via task_packet["trajectory_logging"].
    "trajectory_logging":            True,

    # Observability metrics: collect tokens/latency/tool stats per run.
    # Orchestrator may adjust granularity via task_packet["metrics_enabled"].
    "metrics_enabled":               True,

    # API key rotation: rotate provider keys on 429/auth failures.
    # Requires keys.json to list multiple keys: {"anthropic": ["key1", "key2"]}.
    "key_rotation_enabled":          True,

    # ── Bonus ────────────────────────────────────────────────────────────
    # Loop awareness: inject remaining step budget + tool summary after each turn.
    # Orchestrator may disable via task_packet["loop_awareness"].
    "loop_awareness_injection":      True,

    # Tool dependency detection: analyse tool calls for data dependencies
    # before dispatching; independent calls run in parallel.
    # Orchestrator may override strategy via task_packet["tool_exec_strategy"].
    "tool_dependency_detection":     True,

    "pipelines": {
        "orchestration":     "claude-sonnet-4-6",
        "task_planning":     "claude-sonnet-4-6",
        "research":          "claude-opus-4-6",
        "council_agent":     "claude-opus-4-6",
        "council_synthesis": "claude-opus-4-6",
        "code_generation":   "claude-sonnet-4-6",
        "compression":       "claude-haiku-4-5-20251001",
        "classification":    "claude-haiku-4-5-20251001",
        "file_indexing":     "claude-haiku-4-5-20251001",
        "mt_judge":          "claude-haiku-4-5-20251001",
        "skillsmith_draft":  "claude-sonnet-4-6",
        "skillsmith_eval":   "claude-sonnet-4-6",
        "ack_validation":    "claude-haiku-4-5-20251001",
    },
}

DEFAULT_MODELS = [
    {
        "model_id":       "claude-sonnet-4-6",
        "provider":       "anthropic",
        "call_format":    "messages",
        "strengths":      ["general tasks", "code", "orchestration"],
        "weaknesses":     ["not cheapest for simple tasks"],
        "context_window": 200000,
        "cost_tier":      "medium",
        "speed_tier":     "medium",
        "enabled":        True,
    },
    {
        "model_id":       "claude-opus-4-6",
        "provider":       "anthropic",
        "call_format":    "messages",
        "strengths":      ["complex reasoning", "research", "long context"],
        "weaknesses":     ["slower", "most expensive"],
        "context_window": 200000,
        "cost_tier":      "high",
        "speed_tier":     "slow",
        "enabled":        True,
    },
    {
        "model_id":       "claude-haiku-4-5-20251001",
        "provider":       "anthropic",
        "call_format":    "messages",
        "strengths":      ["fast", "cheap", "classification", "compression"],
        "weaknesses":     ["less capable for complex tasks"],
        "context_window": 200000,
        "cost_tier":      "low",
        "speed_tier":     "fast",
        "enabled":        True,
    },
    {
        "model_id":       "gpt-4o",
        "provider":       "openai",
        "call_format":    "messages",
        "strengths":      ["multimodal", "code", "general tasks"],
        "weaknesses":     ["not cheapest"],
        "context_window": 128000,
        "cost_tier":      "medium",
        "speed_tier":     "medium",
        "enabled":        True,
    },
    {
        "model_id":       "gpt-4o-mini",
        "provider":       "openai",
        "call_format":    "messages",
        "strengths":      ["cheap", "fast", "simple tasks"],
        "weaknesses":     ["less capable than gpt-4o"],
        "context_window": 128000,
        "cost_tier":      "low",
        "speed_tier":     "fast",
        "enabled":        True,
    },
    {
        "model_id":       "llama3.2",
        "provider":       "ollama",
        "call_format":    "messages",
        "strengths":      ["local", "private", "no API cost"],
        "weaknesses":     ["requires local GPU/CPU"],
        "context_window": 128000,
        "cost_tier":      "free",
        "speed_tier":     "variable",
        "enabled":        True,
    },
]

# Directories that must exist under _BASE_PATH
REQUIRED_DIRS = [
    "memory",
    "skills/templates",
    "tools/impl",
    "tools/providers",
    "projects",
]

# Core module files that must exist (added by recent features)
REQUIRED_MODULE_FILES = [
    "db.py",
    "metrics.py",
    "trajectory.py",
    "tools/error_types.py",
    "tools/schemas.py",
    "tools/registry.py",
    "agent.py",
    "Orchestrator.py",
    "context_manager.py",
    "mt_palace.py",
]

# Files that must exist under _BASE_PATH (path → default value)
REQUIRED_FILES: dict[str, object] = {
    "memory/lt.json":    [],
    "memory/mt.json":    [],
    "skills/skills.json": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ok(msg: str)   -> None: print(f"  \033[32m✓\033[0m  {msg}")
def _warn(msg: str) -> None: print(f"  \033[33m⚠\033[0m  {msg}")
def _err(msg: str)  -> None: print(f"  \033[31m✗\033[0m  {msg}")
def _info(msg: str) -> None: print(f"     {msg}")


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------

def scaffold(base: Path = _BASE_PATH) -> None:
    """Create all required directories and initialise missing files."""
    for d in REQUIRED_DIRS:
        (base / d).mkdir(parents=True, exist_ok=True)

    for rel, default in REQUIRED_FILES.items():
        path = base / rel
        if not path.exists():
            _save_json(path, default)

    # config.json — write defaults, preserving existing keys
    cfg_path = base / "config.json"
    existing = _load_json(cfg_path) or {}
    merged   = {**DEFAULT_CONFIG, **existing}
    _save_json(cfg_path, merged)

    # models.json — write defaults only if file is missing / single-entry
    models_path = base / "models.json"
    raw_models  = _load_json(models_path)
    if raw_models is None or (isinstance(raw_models, dict)):
        _save_json(models_path, DEFAULT_MODELS)

    # keys.json — create empty if missing
    keys_path = base / "keys.json"
    if not keys_path.exists():
        _save_json(keys_path, {"anthropic": None, "openai": None, "ollama": None})


def scaffold_project(project: str, base: Path = _BASE_PATH) -> None:
    """Create per-project directory structure."""
    proj = base / "projects" / project
    for sub in ("logs", "dev/output", "instances", "worker", "trajectories"):
        (proj / sub).mkdir(parents=True, exist_ok=True)

    for fname, default in [
        ("ct.json",         []),
        ("file_index.json", {"generated_at": _now_iso(), "dev": {}, "worker": {}}),
        ("dev/index.json",  {}),
        ("preferences.json", {}),
        ("config.json",      {}),
    ]:
        fpath = proj / fname
        if not fpath.exists():
            _save_json(fpath, default)


# ---------------------------------------------------------------------------
# Key collection
# ---------------------------------------------------------------------------

def collect_keys(base: Path = _BASE_PATH) -> None:
    keys_path = base / "keys.json"
    existing  = _load_json(keys_path) or {}

    print("\nAPI Key Setup")
    print("─────────────────────────────────────────────────")
    print("Leave blank to skip a provider.\n")

    providers = {
        "anthropic": "Anthropic API key (sk-ant-...): ",
        "openai":    "OpenAI API key (sk-...):        ",
        "ollama":    "Ollama — press Enter (no key needed): ",
    }
    updated = False
    for provider, prompt in providers.items():
        current = existing.get(provider)
        if provider == "ollama":
            ans = input(prompt).strip()
            if not current and ans == "":
                existing["ollama"] = None   # marks ollama as configured (no key)
                updated = True
        else:
            masked = (current[:8] + "...") if current else "(not set)"
            ans    = input(f"{prompt}[{masked}] ").strip()
            if ans:
                existing[provider] = ans
                updated = True

    if updated:
        _save_json(keys_path, existing)
        print("\nkeys.json updated.")
    else:
        print("\nNo changes.")


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self) -> None:
        self.errors:   list[str] = []
        self.warnings: list[str] = []
        self.ok_count: int       = 0

    def error(self, msg: str)   -> None: self.errors.append(msg);   _err(msg)
    def warning(self, msg: str) -> None: self.warnings.append(msg); _warn(msg)
    def good(self, msg: str)    -> None: self.ok_count += 1;         _ok(msg)

    @property
    def passed(self) -> bool:
        return not self.errors


def check(base: Path = _BASE_PATH) -> CheckResult:
    r = CheckResult()

    # ── 1. Directory structure ───────────────────────────────────────────────
    print("\n[1] Directory structure")
    for d in REQUIRED_DIRS:
        p = base / d
        if p.exists():
            r.good(str(d))
        else:
            r.error(f"Missing directory: {d}")

    # ── 2. JSON file validity ────────────────────────────────────────────────
    print("\n[2] JSON file validity")
    json_files = list((base / "memory").glob("*.json")) + \
                 list((base / "skills").glob("*.json")) + \
                 [base / "config.json", base / "models.json", base / "tools" / "tools.json"]
    for path in json_files:
        if not path.exists():
            r.warning(f"Missing: {path.relative_to(base)}")
            continue
        data = _load_json(path)
        if data is None:
            r.error(f"Invalid JSON: {path.relative_to(base)}")
        else:
            r.good(str(path.relative_to(base)))

    # ── 3. Tool implementations ──────────────────────────────────────────────
    print("\n[3] Tool implementations")
    tools_json_path = base / "tools" / "tools.json"
    tools_data      = _load_json(tools_json_path) or []
    if isinstance(tools_data, dict):
        tools_data = [tools_data]
    for entry in tools_data:
        if not entry.get("enabled", True):
            continue
        tool_id = entry.get("tool_id", "?")
        impl    = base / entry.get("impl", "")
        # Check impl file exists
        if not impl.exists():
            r.error(f"Missing impl: {entry.get('impl')} for tool '{tool_id}'")
            continue
        # Check required fields
        missing_fields = [f for f in ("tool_id", "impl", "permissions") if f not in entry]
        if missing_fields:
            r.error(f"{tool_id} — missing fields in tools.json: {missing_fields}")
            continue
        if not isinstance(entry.get("permissions"), list) or not entry["permissions"]:
            r.error(f"{tool_id} — 'permissions' must be a non-empty list")
            continue
        r.good(f"{tool_id}  ({', '.join(entry['permissions'])})")

    # ── 4. Tool schema coverage ──────────────────────────────────────────────
    print("\n[4] Tool schema coverage")
    try:
        import importlib.util as _ilu
        _sp = str(base)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)
        _schemas_path = base / "tools" / "schemas.py"
        _spec = _ilu.spec_from_file_location("_schemas_check", str(_schemas_path))
        _smod = _ilu.module_from_spec(_spec)       # type: ignore[arg-type]
        _spec.loader.exec_module(_smod)             # type: ignore[union-attr]
        _merged = _smod._get_merged_schemas()
        for entry in tools_data:
            if not entry.get("enabled", True):
                continue
            tid = entry.get("tool_id", "")
            if tid in _merged:
                r.good(f"{tid} — schema present")
            else:
                r.warning(f"{tid} — no schema in schemas.py or impl SCHEMA dict (agents cannot use this tool)")
    except Exception as exc:
        r.warning(f"Schema coverage check skipped: {exc}")

    # ── 5. Pipeline → model coverage ────────────────────────────────────────
    print("\n[5] Pipeline model coverage")
    cfg         = _load_json(base / "config.json") or {}
    pipelines   = cfg.get("pipelines", {})
    raw_models  = _load_json(base / "models.json") or []
    if isinstance(raw_models, dict):
        raw_models = [raw_models]
    model_ids = {m["model_id"] for m in raw_models if m.get("enabled", True)}
    for pipe, model_id in pipelines.items():
        if model_id in model_ids:
            r.good(f"{pipe} → {model_id}")
        else:
            r.error(f"Pipeline '{pipe}' references unknown/disabled model '{model_id}'")
    # Fast model routing
    fast_model = cfg.get("fast_model", "")
    if fast_model:
        if fast_model in model_ids:
            r.good(f"fast_model → {fast_model}")
        else:
            r.warning(f"fast_model '{fast_model}' not found in models.json (routing disabled)")

    # ── 6. Core module file integrity ────────────────────────────────────────
    print("\n[6] Core module files")
    for rel in REQUIRED_MODULE_FILES:
        p = base / rel
        if p.exists():
            r.good(rel)
        else:
            r.error(f"Missing core module: {rel}")

    # ── 7. Config key completeness ───────────────────────────────────────────
    print("\n[7] Config key completeness")
    cfg = _load_json(base / "config.json") or {}
    # Flatten DEFAULT_CONFIG (exclude nested dicts like 'pipelines')
    flat_defaults = {k: v for k, v in DEFAULT_CONFIG.items() if not isinstance(v, dict)}
    missing_keys  = [k for k in flat_defaults if k not in cfg]
    extra_keys    = [k for k in cfg if k not in DEFAULT_CONFIG and k not in ("pipelines",)]
    if missing_keys:
        r.warning(f"{len(missing_keys)} config key(s) missing: {missing_keys} — run --fix to add defaults")
    else:
        r.good("All default config keys present")
    if extra_keys:
        r.good(f"{len(extra_keys)} project-specific key(s): {extra_keys[:5]}")

    # ── 8. Skill embeddings ──────────────────────────────────────────────────
    print("\n[8] Skill embeddings")
    skills = _load_json(base / "skills" / "skills.json") or []
    if isinstance(skills, dict):
        skills = [skills]
    active_skills = [s for s in skills if s.get("active")]
    if not active_skills:
        r.good("No active skills (SkillSmith will create them on demand)")
    for s in active_skills:
        name = s.get("name", s.get("skill_id", "?"))
        if s.get("embedding"):
            r.good(f"{name} — has embedding")
        else:
            r.warning(f"{name} — missing embedding (re-run SkillSmith to regenerate)")
        tmpl = s.get("template", "")
        if tmpl and not (base / "skills" / tmpl).exists():
            r.warning(f"{name} — template file missing: {tmpl}")

    # ── 9. Stale CT flags ────────────────────────────────────────────────────
    print("\n[9] CT flag health (all projects)")
    projects_dir = base / "projects"
    if projects_dir.exists():
        found_any = False
        for proj_dir in sorted(projects_dir.iterdir()):
            if not proj_dir.is_dir():
                continue
            ct_path = proj_dir / "ct.json"
            ct_data = _load_json(ct_path) or []
            open_flags = [e for e in ct_data if e.get("status") == "open"]
            for flag in open_flags:
                found_any = True
                opened_at = flag.get("opened_at", "")
                try:
                    opened_dt = datetime.fromisoformat(opened_at)
                    age = datetime.now(timezone.utc) - opened_dt
                    if age > timedelta(hours=24):
                        r.warning(
                            f"[{proj_dir.name}] Stale CT flag (open {int(age.total_seconds()//3600)}h): "
                            f"{flag.get('description','')[:60]}"
                        )
                    else:
                        r.good(f"[{proj_dir.name}] CT flag open {int(age.total_seconds()//60)}m: "
                               f"{flag.get('description','')[:40]}")
                except Exception:
                    r.warning(f"[{proj_dir.name}] CT flag with unparseable opened_at: {opened_at}")
        if not found_any:
            r.good("No open CT flags")
    else:
        r.good("No projects yet")

    # ── 10. File index coverage ──────────────────────────────────────────────
    print("\n[10] File index coverage (all projects)")
    if projects_dir.exists():
        found_any = False
        for proj_dir in sorted(projects_dir.iterdir()):
            if not proj_dir.is_dir():
                continue
            dev_out = proj_dir / "dev" / "output"
            dev_idx = _load_json(proj_dir / "dev" / "index.json") or {}
            if dev_out.exists():
                found_any = True
                actual_files = set(
                    str(f.relative_to(proj_dir)) for f in dev_out.rglob("*") if f.is_file()
                )
                indexed = set(dev_idx.keys())
                missing = actual_files - indexed
                for mf in missing:
                    r.warning(f"[{proj_dir.name}] Not indexed: {mf} — run --fix to queue re-index")
                if not missing:
                    r.good(f"[{proj_dir.name}] All dev/output files indexed ({len(actual_files)})")
        if not found_any:
            r.good("No dev/output files yet")

    # ── 11. Trajectory + DB directories ─────────────────────────────────────
    print("\n[11] Observability paths")
    if projects_dir.exists():
        for proj_dir in sorted(projects_dir.iterdir()):
            if not proj_dir.is_dir():
                continue
            traj_dir = proj_dir / "trajectories"
            if traj_dir.exists():
                count = len(list(traj_dir.glob("*.jsonl")))
                r.good(f"[{proj_dir.name}] trajectories/ exists ({count} JSONL file(s))")
            else:
                r.warning(f"[{proj_dir.name}] trajectories/ missing — run --fix to create")
    db_path = base / "pikaia.db"
    if db_path.exists():
        r.good(f"pikaia.db present ({db_path.stat().st_size // 1024} KB)")
    else:
        r.good("pikaia.db not yet created (created on first agent run)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  OK: {r.ok_count}   Warnings: {len(r.warnings)}   Errors: {len(r.errors)}")
    if r.passed:
        print("  \033[32mAll checks passed.\033[0m")
    else:
        print("  \033[31mFailed — run with --fix to repair recoverable issues.\033[0m")
    print()
    return r


# ---------------------------------------------------------------------------
# --fix
# ---------------------------------------------------------------------------

def fix(base: Path = _BASE_PATH) -> None:
    """Auto-repair recoverable issues found by --check."""
    print("\nRunning --fix...")
    fixed = 0

    # Fix 1: scaffold missing directories + files
    print("\n[fix] Scaffolding missing structure...")
    scaffold(base)
    print("  ✓  Structure scaffolded")
    fixed += 1

    # Fix 2: close stale CT flags (open > 24h) → status: failed
    print("\n[fix] Closing stale CT flags (>24h open)...")
    projects_dir = base / "projects"
    if projects_dir.exists():
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            ct_path = proj_dir / "ct.json"
            ct_data = _load_json(ct_path)
            if not isinstance(ct_data, list):
                continue
            changed = False
            for flag in ct_data:
                if flag.get("status") != "open":
                    continue
                opened_at = flag.get("opened_at", "")
                try:
                    opened_dt = datetime.fromisoformat(opened_at)
                    age = datetime.now(timezone.utc) - opened_dt
                    if age > timedelta(hours=24):
                        flag["status"]    = "failed"
                        flag["closed_at"] = _now_iso()
                        _info(f"  Closed stale flag: {flag.get('description','')[:60]}")
                        changed = True
                        fixed  += 1
                except Exception:
                    pass
            if changed:
                _save_json(ct_path, ct_data)

    # Fix 3: initialise missing JSON data files
    print("\n[fix] Initialising missing data files...")
    for rel, default in REQUIRED_FILES.items():
        path = base / rel
        if not path.exists():
            _save_json(path, default)
            _info(f"  Created {rel}")
            fixed += 1

    # Fix 4: config.json — fill missing keys with defaults
    print("\n[fix] Merging missing config keys...")
    cfg_path = base / "config.json"
    existing = _load_json(cfg_path) or {}
    before   = len(existing)
    for k, v in DEFAULT_CONFIG.items():
        if k not in existing:
            existing[k] = v
    if len(existing) > before:
        _save_json(cfg_path, existing)
        _info(f"  Added {len(existing) - before} missing config key(s)")
        fixed += 1

    # Fix 5: models.json — expand single-entry to full list
    print("\n[fix] Checking models.json...")
    models_path = base / "models.json"
    raw_models  = _load_json(models_path)
    if isinstance(raw_models, dict):
        _save_json(models_path, DEFAULT_MODELS)
        _info("  Expanded single-entry models.json to full registry")
        fixed += 1
    else:
        existing_ids = {m.get("model_id") for m in (raw_models or [])}
        added = 0
        for m in DEFAULT_MODELS:
            if m["model_id"] not in existing_ids:
                (raw_models or []).append(m)
                added += 1
        if added:
            _save_json(models_path, raw_models)
            _info(f"  Added {added} missing model(s) to models.json")
            fixed += 1

    # Fix 6: create missing trajectories/ dirs for all projects
    print("\n[fix] Creating missing trajectories/ directories...")
    projects_dir = base / "projects"
    if projects_dir.exists():
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            traj_dir = proj_dir / "trajectories"
            if not traj_dir.exists():
                traj_dir.mkdir(parents=True, exist_ok=True)
                _info(f"  Created {proj_dir.name}/trajectories/")
                fixed += 1

    # Remaining issues require API access (re-indexing dev/ files)
    print("\n[fix] Note: re-indexing unindexed dev/ files requires a running API key.")
    print("      Start the agent and run tasks to trigger re-indexing automatically.")

    print(f"\n  Fixed {fixed} issue(s). Run --check to verify.\n")


# ---------------------------------------------------------------------------
# Setup wizard (first run)
# ---------------------------------------------------------------------------

def setup_wizard(first_project: str = "default") -> None:
    print("─" * 52)
    print("  AGENT — First-Time Setup")
    print("─" * 52)

    print("\nScaffolding directory structure...")
    scaffold()
    print("  ✓  Done\n")

    collect_keys()

    print(f"\nCreating project '{first_project}'...")
    scaffold_project(first_project)
    print(f"  ✓  Project '{first_project}' ready\n")

    print("Setup complete. Run:  python main.py --project " + first_project)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AGENT bootstrap + integrity checker")
    parser.add_argument("--check",   action="store_true", help="Validate structure")
    parser.add_argument("--fix",     action="store_true", help="Auto-repair recoverable issues")
    parser.add_argument("--project", default=None,        help="Create / scaffold a specific project")
    parser.add_argument("--test",    action="store_true", help="Run functional tool tests")
    parser.add_argument("--tool",    default=None,        help="Limit --test to a specific tool name")
    parser.add_argument("--fast",    action="store_true", help="Skip slow / network tests")
    args = parser.parse_args()

    if args.check:
        result = check()
        sys.exit(0 if result.passed else 1)

    elif args.fix:
        fix()
        check()

    elif args.project:
        scaffold()
        scaffold_project(args.project)
        print(f"Project '{args.project}' ready.")

    elif args.test:
        # Import and run the tool test suite
        test_path = _BASE_PATH / "test_tools.py"
        if not test_path.exists():
            print(f"\033[31mtest_tools.py not found at {test_path}\033[0m")
            sys.exit(1)
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("test_tools", str(test_path))
        mod  = _ilu.module_from_spec(spec)          # type: ignore[arg-type]
        spec.loader.exec_module(mod)                 # type: ignore[union-attr]
        passed = mod.run_tests(
            base_path = str(_BASE_PATH),
            tool      = args.tool,
            fast      = args.fast,
        )
        sys.exit(0 if passed else 1)

    else:
        # First-run wizard
        first_proj = "default"
        setup_wizard(first_proj)


if __name__ == "__main__":
    main()
