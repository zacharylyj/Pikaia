#!/usr/bin/env python3
"""
main.py — CLI entry point for AGENT.

Usage:
    python main.py --project <name>
    python main.py --project <name> --instance <id>   # resume existing session
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_BASE_PATH = Path(__file__).resolve().parent   # Pikaia/

if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

from Orchestrator import Orchestrator, OrchestratorConfig, Tools  # noqa: E402
from tools.registry import ToolRegistry                            # noqa: E402


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
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


def _project_path(project: str, *parts: str) -> Path:
    base = _BASE_PATH / "projects" / project
    return base / Path(*parts) if parts else base


def _fmt(text: str, colour: str) -> str:
    codes = {"grey": "\033[90m", "cyan": "\033[36m", "yellow": "\033[33m",
             "red": "\033[31m", "bold": "\033[1m", "green": "\033[32m"}
    return f"{codes.get(colour,'')}{text}\033[0m" if sys.stdout.isatty() else text


# ---------------------------------------------------------------------------
# Project / instance management
# ---------------------------------------------------------------------------

def _ensure_project(project: str) -> None:
    """Create project directory structure if it does not already exist."""
    proj = _project_path(project)
    for sub in ("logs", "dev/output", "instances", "worker"):
        (proj / sub).mkdir(parents=True, exist_ok=True)

    for fname, default in [
        ("ct.json",           []),
        ("file_index.json",   {"generated_at": _now_iso(), "dev": {}, "worker": {}}),
        ("dev/index.json",    {}),
        ("preferences.json",  {}),
        ("config.json",       {}),
    ]:
        fpath = proj / fname
        if not fpath.exists():
            _save_json(fpath, default)


def _create_or_resume(project: str, instance_id: str | None) -> str:
    if instance_id:
        inst_dir = _project_path(project, "instances", instance_id)
        if inst_dir.exists():
            print(_fmt(f"Resuming instance {instance_id}", "grey"))
            return instance_id
        print(_fmt(f"Instance '{instance_id}' not found — starting fresh", "yellow"))

    new_id   = f"inst_{uuid.uuid4().hex[:8]}"
    inst_dir = _project_path(project, "instances", new_id)
    inst_dir.mkdir(parents=True, exist_ok=True)
    _save_json(inst_dir / "st.json", {
        "instance_id": new_id,
        "project":     project,
        "summary":     "",
        "window":      [],
        "updated_at":  _now_iso(),
    })
    _save_json(inst_dir / "history.json", [])
    return new_id


def _make_orchestrator(project: str, instance_id: str,
                        debug: bool = False) -> tuple[Orchestrator, ToolRegistry]:
    cfg = OrchestratorConfig.from_json(
        str(_BASE_PATH / "config.json"),
        str(_project_path(project, "config.json")),
    )
    if debug:
        # Route every pipeline through the no-op debug adapter
        for key in cfg.pipelines:
            cfg.pipelines[key] = "debug-model"

    registry = ToolRegistry(
        base_path   = str(_BASE_PATH),
        project     = project,
        instance_id = instance_id,
        caller      = "orchestrator",
    )
    tools = Tools(dispatch=registry.dispatch)
    orch  = Orchestrator(
        project     = project,
        instance_id = instance_id,
        base_path   = str(_BASE_PATH),
        tools       = tools,
        config      = cfg,
        on_status   = lambda msg: print(_fmt(f"  {msg}", "grey")),
    )
    return orch, registry


# ---------------------------------------------------------------------------
# /command handlers
# ---------------------------------------------------------------------------

def _cmd_status(project: str) -> None:
    entries = _load_json(_project_path(project, "ct.json")) or []
    if not entries:
        print("CT board is empty.")
        return
    open_  = [e for e in entries if e.get("status") == "open"]
    closed = [e for e in entries if e.get("status") != "open"]
    print(f"\n{_fmt('── CT Board: ' + project, 'bold')}")
    if open_:
        print(_fmt("  OPEN", "yellow"))
        for e in open_:
            ts = e.get("opened_at", "")[:16]
            print(f"    [{e.get('type','?')}] {e.get('description','')[:60]}  {ts}")
    if closed:
        print(_fmt("  CLOSED (last 5)", "grey"))
        for e in closed[-5:]:
            ts = e.get("closed_at", "")[:16]
            print(_fmt(f"    [{e.get('status','?')}] {e.get('description','')[:60]}  {ts}", "grey"))
    print()


def _cmd_memory(args: list[str], project: str, instance_id: str) -> None:
    sub = args[0].lower() if args else "lt"

    if sub == "lt":
        data = _load_json(_BASE_PATH / "memory" / "lt.json") or []
        print(f"\n{_fmt('── Long-term memory', 'bold')} ({len(data)} entries)")
        if not data:
            print("  (empty)")
        for e in data:
            print(f"  [{e.get('category','?')}] {e.get('content','')}")

    elif sub == "mt":
        data   = _load_json(_BASE_PATH / "memory" / "mt.json") or []
        active = [e for e in data if e.get("status", "active") == "active"]
        print(f"\n{_fmt('── Mid-term memory', 'bold')} ({len(active)} active / {len(data)} total)")
        if not active:
            print("  (empty)")
        for e in active[-10:]:
            ts = e.get("created_at", "")[:16]
            print(f"  [{e.get('type','?')}] {e.get('content','')[:80]}  {_fmt(ts, 'grey')}")

    elif sub == "st":
        st = _load_json(_project_path(project, "instances", instance_id, "st.json")) or {}
        print(f"\n{_fmt('── Short-term memory', 'bold')} ({instance_id})")
        summary = st.get("summary") or "(none)"
        window  = st.get("window", [])
        print(f"  Summary : {summary[:120]}")
        print(f"  Window  : {len(window)} messages")
        for msg in window[-4:]:
            role    = _fmt(msg.get("role", "?").upper(), "cyan")
            content = msg.get("content", "")[:100]
            print(f"  {role}: {content}")

    else:
        print("Usage: /memory lt | mt | st")
    print()


def _cmd_skills() -> None:
    data   = _load_json(_BASE_PATH / "skills" / "skills.json") or []
    active = [s for s in data if s.get("active")]
    print(f"\n{_fmt('── Skills', 'bold')} ({len(active)} active / {len(data)} total)")
    if not active:
        print("  No active skills. SkillSmith will create them on demand.")
    for s in sorted(active, key=lambda x: x.get("tier", 9)):
        tier  = s.get("tier", "?")
        ver   = s.get("version", "?")
        name  = s.get("name", "?")
        tags  = ", ".join(s.get("tags", []))
        tools = ", ".join(s.get("tools_required", []))
        print(f"  T{tier} v{ver}  {_fmt(name, 'bold'):<32s}  tags: {tags}")
        if tools:
            print(_fmt(f"         tools: {tools}", "grey"))
    print()


def _cmd_models() -> None:
    raw  = _load_json(_BASE_PATH / "models.json") or []
    if isinstance(raw, dict):
        raw = [raw]
    cfg       = _load_json(_BASE_PATH / "config.json") or {}
    pipelines = cfg.get("pipelines", {})

    print(f"\n{_fmt('── Models', 'bold')} ({len(raw)})")
    for m in raw:
        enabled = "" if m.get("enabled", True) else _fmt("  [DISABLED]", "red")
        cost    = m.get("cost_tier", "?")
        speed   = m.get("speed_tier", "?")
        print(f"  {m.get('model_id','?'):<38s} {m.get('provider','?'):<12s} cost:{cost:<8s} speed:{speed}{enabled}")

    print(f"\n{_fmt('── Pipeline map', 'bold')}")
    for pipe, model in pipelines.items():
        print(f"  {pipe:<24s} → {model}")
    print()


def _cmd_approve(project: str) -> None:
    ct_path = _project_path(project, "ct.json")
    entries = _load_json(ct_path) or []
    pending = [e for e in entries if
               e.get("type") == "skill_approval" and
               e.get("status") == "pending_approval"]

    if not pending:
        print("No pending skill approvals.")
        return

    for flag in pending:
        skill_id = flag.get("skill_id", "?")
        name     = flag.get("description", "").replace("SkillSmith drafted: ", "")
        print(f"\n{_fmt('── Skill Approval', 'bold')} ─────────────────────────────")
        print(f"  skill_id  : {skill_id}")
        print(f"  name      : {name}")
        print(f"  opened    : {flag.get('opened_at','')[:16]}")

        # Try to find draft details in worker/skillsmith/
        draft_path = _project_path(project, "worker", "skillsmith", skill_id, "draft.json")
        draft      = _load_json(draft_path)
        if draft:
            print(f"  tier      : {draft.get('tier','?')}")
            print(f"  tools     : {', '.join(draft.get('tools_required', []))}")
            print(f"  desc      : {draft.get('description','')[:80]}")
        else:
            print(_fmt("  (no draft.json found in worker/skillsmith/)", "grey"))

        try:
            answer = input(f"  Approve '{name}'? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipped.")
            continue

        if answer == "y":
            flag["status"]    = "done"
            flag["closed_at"] = _now_iso()
            _save_json(ct_path, entries)

            # Write the approved skill into skills/skills.json so _skill_pick can find it
            if draft:
                draft["active"]      = True
                draft["approved_at"] = _now_iso()
                skills_path = _BASE_PATH / "skills" / "skills.json"
                skills_path.parent.mkdir(parents=True, exist_ok=True)
                existing = _load_json(skills_path) or []
                if not isinstance(existing, list):
                    existing = [existing]
                # Replace if same skill_id already present, otherwise append
                idx = next((i for i, s in enumerate(existing)
                            if s.get("skill_id") == draft.get("skill_id")), None)
                if idx is not None:
                    existing[idx] = draft
                else:
                    existing.append(draft)
                _save_json(skills_path, existing)
                print(_fmt(f"  ✓ Skill '{draft.get('name','?')}' is now active.", "green"))
            else:
                print(_fmt(f"  ✓ CT flag approved (no draft found to activate).", "green"))
        else:
            flag["status"]    = "failed"
            flag["closed_at"] = _now_iso()
            _save_json(ct_path, entries)
            print(_fmt("  ✗ Rejected.", "yellow"))
    print()


def _cmd_instances(project: str) -> None:
    inst_dir = _project_path(project, "instances")
    if not inst_dir.exists():
        print("No instances found.")
        return
    instances = sorted(d.name for d in inst_dir.iterdir() if d.is_dir())
    print(f"\n{_fmt('── Instances: ' + project, 'bold')} ({len(instances)})")
    for iid in instances:
        st      = _load_json(inst_dir / iid / "st.json") or {}
        updated = _fmt(st.get("updated_at", "")[:16], "grey")
        summary = (st.get("summary") or "(no summary)")[:60]
        print(f"  {iid}  {updated}  {summary}")
    print()


def _cmd_files(project: str) -> None:
    idx = _load_json(_project_path(project, "file_index.json")) or {}
    ts  = _fmt(idx.get("generated_at", "")[:16], "grey")
    print(f"\n{_fmt('── File index: ' + project, 'bold')}  {ts}")

    dev_files = 0
    for folder, files in idx.get("dev", {}).items():
        flist = files if isinstance(files, list) else []
        for f in flist:
            print(f"  dev/{folder}/{f.get('path','?')}  {f.get('size_kb','?')}kb")
            dev_files += 1

    worker = idx.get("worker", {})
    for agent_id, info in worker.items():
        status = _fmt(info.get("status","?"), "grey")
        nfiles = len(info.get("files", []))
        print(f"  worker/{agent_id}/  [{status}]  {nfiles} file(s)")

    if dev_files == 0 and not worker:
        print("  (index is empty — run tasks to populate)")
    print()


def _cmd_check() -> None:
    init_path = _BASE_PATH / "init.py"
    if not init_path.exists():
        print("init.py not found.")
        return
    import subprocess
    subprocess.run([sys.executable, str(init_path), "--check"], check=False)


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

_HELP = """\
Commands:
  /status             CT board for current project
  /memory lt|mt|st    Inspect memory layers
  /skills             Active skills + versions
  /models             Available models + pipeline map
  /approve            Review pending skill approvals
  /instances          Running instances for this project
  /files              Project file index (layer 1)
  /check              Run integrity check (init.py --check)
  /project <name>     Switch project
  /new project <name> Create + switch to new project
  /exit               Close this instance
"""

Session = tuple[str, str, Orchestrator, ToolRegistry]

# Set by main() before the REPL starts; read by _dispatch_command when
# spawning a new orchestrator on /project or /new project.
_DEBUG_MODE: bool = False


def _dispatch_command(line: str, project: str, instance_id: str,
                      orch: Orchestrator, reg: ToolRegistry) -> tuple[Session, bool]:
    """
    Returns ((project, instance_id, orch, reg), should_exit).
    """
    parts = line.strip().split()
    cmd   = parts[0].lower()
    args  = parts[1:]

    if cmd in ("/exit", "/quit"):
        return (project, instance_id, orch, reg), True

    if cmd in ("/help", "/?"):
        print(_HELP)
    elif cmd == "/status":
        _cmd_status(project)
    elif cmd == "/memory":
        _cmd_memory(args, project, instance_id)
    elif cmd == "/skills":
        _cmd_skills()
    elif cmd == "/models":
        _cmd_models()
    elif cmd == "/approve":
        _cmd_approve(project)
    elif cmd == "/instances":
        _cmd_instances(project)
    elif cmd == "/files":
        _cmd_files(project)
    elif cmd == "/check":
        _cmd_check()
    elif cmd == "/project":
        if not args:
            print("Usage: /project <name>")
        else:
            new_proj = args[0]
            _ensure_project(new_proj)
            new_inst = _create_or_resume(new_proj, None)
            new_orch, new_reg = _make_orchestrator(new_proj, new_inst, debug=_DEBUG_MODE)
            print(f"Switched to project '{new_proj}'  instance: {new_inst}")
            return (new_proj, new_inst, new_orch, new_reg), False
    elif cmd == "/new":
        if len(args) >= 2 and args[0].lower() == "project":
            new_proj = args[1]
            _ensure_project(new_proj)
            new_inst = _create_or_resume(new_proj, None)
            new_orch, new_reg = _make_orchestrator(new_proj, new_inst, debug=_DEBUG_MODE)
            print(f"Created project '{new_proj}'  instance: {new_inst}")
            return (new_proj, new_inst, new_orch, new_reg), False
        else:
            print("Usage: /new project <name>")
    else:
        print(f"Unknown command '{cmd}'. Type /help for the list.")

    return (project, instance_id, orch, reg), False


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_BANNER = """
  ╔══════════════════════════════════════╗
  ║              A G E N T              ║
  ╚══════════════════════════════════════╝
"""


def main() -> None:
    global _DEBUG_MODE

    parser = argparse.ArgumentParser(description="AGENT — AI agent orchestration framework")
    parser.add_argument("--project",  default="default", metavar="NAME",
                        help="Project to load (default: default)")
    parser.add_argument("--instance", default=None,      metavar="ID",
                        help="Resume an existing instance by ID")
    parser.add_argument("--debug",    action="store_true",
                        help="Debug mode: use mock LLM adapter (no API keys needed)")
    args = parser.parse_args()

    project      = args.project
    _DEBUG_MODE  = args.debug

    # First-run: auto-scaffold if projects/ doesn't exist
    if not (_BASE_PATH / "projects").exists():
        print("First run detected — scaffolding project structure...")
        init_path = _BASE_PATH / "init.py"
        if init_path.exists():
            import subprocess
            subprocess.run([sys.executable, str(init_path), "--project", project], check=False)

    _ensure_project(project)
    instance_id = _create_or_resume(project, args.instance)
    orch, reg   = _make_orchestrator(project, instance_id, debug=_DEBUG_MODE)

    print(_BANNER)
    print(f"  Project  : {_fmt(project, 'cyan')}")
    print(f"  Instance : {_fmt(instance_id, 'grey')}")
    print(f"  Base     : {_fmt(str(_BASE_PATH), 'grey')}")
    if _DEBUG_MODE:
        print(f"  Mode     : {_fmt('DEBUG  (no API keys used — mock LLM responses)', 'yellow')}")
    print(f"\n  Type {_fmt('/help', 'bold')} for commands or {_fmt('/exit', 'bold')} to quit.\n")

    # Enable readline on platforms that support it
    try:
        import readline as _rl
        _rl.parse_and_bind("tab: complete")
    except ImportError:
        pass

    while True:
        try:
            prompt = f"{_fmt('[' + project + ']', 'cyan')} {_fmt('>', 'bold')} "
            line   = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  (Use {_fmt('/exit', 'bold')} to quit.)")
            continue

        if not line:
            continue

        if line.startswith("/"):
            (project, instance_id, orch, reg), should_exit = _dispatch_command(
                line, project, instance_id, orch, reg
            )
            if should_exit:
                print("Goodbye.")
                try:
                    orch.shutdown()
                except Exception:
                    pass
                break
            continue

        # Regular message — send to orchestrator
        try:
            response = orch.turn(line)
            if response:
                print(f"\n{response}\n")
        except KeyboardInterrupt:
            print(_fmt("\n  [interrupted]", "yellow"))
        except Exception as exc:
            print(_fmt(f"\n  [error] {exc}\n", "red"))


if __name__ == "__main__":
    main()
