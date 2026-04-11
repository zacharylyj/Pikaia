#!/usr/bin/env python3
"""
init.py — Bootstrap and integrity checker for AGENT.

Usage:
    python init.py                      # first-time setup wizard
    python init.py --check              # validate structure + JSON + pipelines
    python init.py --fix                # auto-fix recoverable issues
    python init.py --project <name>     # scaffold a specific project (non-interactive)
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
    for sub in ("logs", "dev/output", "instances", "worker"):
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

    # ── 3. Tool impl files ───────────────────────────────────────────────────
    print("\n[3] Tool implementations")
    tools_json_path = base / "tools" / "tools.json"
    tools_data      = _load_json(tools_json_path) or []
    if isinstance(tools_data, dict):
        tools_data = [tools_data]
    for entry in tools_data:
        if not entry.get("enabled", True):
            continue
        impl = base / entry.get("impl", "")
        if impl.exists():
            r.good(entry["tool_id"])
        else:
            r.error(f"Missing impl: {entry.get('impl')} for tool '{entry.get('tool_id')}'")

    # ── 4. Pipeline → model coverage ────────────────────────────────────────
    print("\n[4] Pipeline model coverage")
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

    # ── 5. Skill embeddings ──────────────────────────────────────────────────
    print("\n[5] Skill embeddings")
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

    # ── 6. Stale CT flags ────────────────────────────────────────────────────
    print("\n[6] CT flag health (all projects)")
    projects_dir = base / "projects"
    if projects_dir.exists():
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            ct_path = proj_dir / "ct.json"
            ct_data = _load_json(ct_path) or []
            open_flags = [e for e in ct_data if e.get("status") == "open"]
            for flag in open_flags:
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
    else:
        r.good("No projects yet")

    # ── 7. File index coverage ───────────────────────────────────────────────
    print("\n[7] File index coverage (all projects)")
    if projects_dir.exists():
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            dev_out = proj_dir / "dev" / "output"
            dev_idx = _load_json(proj_dir / "dev" / "index.json") or {}
            if dev_out.exists():
                actual_files = set(
                    str(f.relative_to(proj_dir)) for f in dev_out.rglob("*") if f.is_file()
                )
                indexed = set(dev_idx.keys())
                missing = actual_files - indexed
                for mf in missing:
                    r.warning(f"[{proj_dir.name}] Not indexed: {mf} — run /check --fix to re-index")
                if not missing:
                    r.good(f"[{proj_dir.name}] All dev/output files indexed ({len(actual_files)})")

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
    args = parser.parse_args()

    if args.check:
        result = check()
        sys.exit(0 if result.passed else 1)

    elif args.fix:
        fix()
        check()   # show status after fix

    elif args.project:
        scaffold()
        scaffold_project(args.project)
        print(f"Project '{args.project}' ready.")

    else:
        # First-run wizard
        first_proj = "default"
        setup_wizard(first_proj)


if __name__ == "__main__":
    main()
