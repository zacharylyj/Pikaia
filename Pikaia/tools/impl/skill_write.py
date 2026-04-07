"""
skill_write
-----------
Write a new skill version to the registry — SkillSmith only, post human gate.

Checks that the CT skill_approval flag for this skill_id has been closed
(status: done) before writing. Raises if the gate has not been passed.

On write:
  - Sets previous active version to active: false
  - Increments version number
  - Writes template file to skills/templates/{skill_id}_v{version}.md
  - Embeds the skill description
  - Appends new entry to skills.json

params:
    skill            : dict   - skill schema (skill_id required; version auto-incremented)
    template_content : str    - prompt template markdown

returns:
    written  : bool
    skill_id : str
    version  : int
"""

from __future__ import annotations

import json
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    caller = context.get("caller", "")
    if caller != "skillsmith":
        raise PermissionError(
            "skill_write is restricted to SkillSmith (caller='skillsmith')"
        )

    skill            = dict(params["skill"])
    template_content = params["template_content"]
    base_path        = Path(context["base_path"])
    project          = context.get("project", "")

    skill_id = skill.get("skill_id")
    if not skill_id:
        raise ValueError("skill.skill_id is required")

    # ------------------------------------------------------------------
    # 1. Verify human gate: CT skill_approval must be closed (done)
    # ------------------------------------------------------------------
    _verify_gate(base_path, project, skill_id)

    # ------------------------------------------------------------------
    # 2. Load current skills.json
    # ------------------------------------------------------------------
    skills_path = base_path / "skills" / "skills.json"
    skills_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if skills_path.exists():
        raw = json.loads(skills_path.read_text())
        entries = raw if isinstance(raw, list) else [raw]

    # ------------------------------------------------------------------
    # 3. Determine new version; deactivate old
    # ------------------------------------------------------------------
    existing = [e for e in entries if e.get("skill_id") == skill_id]
    current_version = max((e.get("version", 0) for e in existing), default=0)
    new_version = current_version + 1

    for e in entries:
        if e.get("skill_id") == skill_id:
            e["active"] = False

    # ------------------------------------------------------------------
    # 4. Write template file
    # ------------------------------------------------------------------
    templates_dir = base_path / "skills" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    template_filename = f"{skill_id}_v{new_version}.md"
    template_path     = templates_dir / template_filename
    template_path.write_text(template_content, encoding="utf-8")

    # ------------------------------------------------------------------
    # 5. Embed description
    # ------------------------------------------------------------------
    description = skill.get("description", "")
    embedding   = _get_embedding(description, context) if description else []

    # ------------------------------------------------------------------
    # 6. Build and append new entry
    # ------------------------------------------------------------------
    new_entry: dict = {
        **skill,
        "version":    new_version,
        "previous":   current_version if current_version > 0 else None,
        "active":     True,
        "template":   f"templates/{template_filename}",
        "embedding":  embedding,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    new_entry.setdefault("created_by", "auto")
    entries.append(new_entry)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    _save_json(skills_path, entries)

    return {"written": True, "skill_id": skill_id, "version": new_version}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _verify_gate(base_path: Path, project: str, skill_id: str) -> None:
    """Raise if the CT skill_approval for skill_id is not closed (done)."""
    ct_path = base_path / "projects" / project / "ct.json"
    if not ct_path.exists():
        raise PermissionError(
            f"CT file not found for project '{project}'. "
            "skill_write requires a closed skill_approval CT flag."
        )
    entries = json.loads(ct_path.read_text())
    if isinstance(entries, dict):
        entries = [entries]

    approval = next(
        (
            e for e in entries
            if e.get("type") == "skill_approval"
            and e.get("skill_id") == skill_id
        ),
        None,
    )
    if approval is None:
        raise PermissionError(
            f"No skill_approval CT flag found for skill_id '{skill_id}'. "
            "Human approval is required before skill_write."
        )
    if approval.get("status") != "done":
        raise PermissionError(
            f"skill_approval flag for '{skill_id}' has status '{approval.get('status')}'. "
            "Must be 'done' (approved by human) before writing."
        )


def _get_embedding(text: str, context: dict) -> list[float]:
    try:
        base_path = Path(context["base_path"])
        impl_path = base_path / "tools" / "impl" / "embed_text.py"
        spec = importlib.util.spec_from_file_location("embed_text", str(impl_path))
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        result = mod.run({"text": text}, context)
        return result.get("embedding", [])
    except Exception:
        return []


def _save_json(path: Path, data: Any) -> None:
    import os, tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
