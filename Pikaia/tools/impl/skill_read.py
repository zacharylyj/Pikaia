"""
skill_read
----------
Fetch the active skill schema + template text by skill_id.

params:
    skill_id : str   - the skill_id to look up

returns:
    skill    : dict   - the full skill schema entry (active version)
    template : str    - content of the prompt template file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    skill_id  = params["skill_id"]
    base_path = Path(context["base_path"])

    skills_path = base_path / "skills" / "skills.json"
    if not skills_path.exists():
        raise FileNotFoundError(f"skills.json not found at {skills_path}")

    raw = json.loads(skills_path.read_text())
    entries: list[dict] = raw if isinstance(raw, list) else [raw]

    # Find active version of this skill
    skill = next(
        (s for s in entries if s.get("skill_id") == skill_id and s.get("active", False)),
        None,
    )
    if skill is None:
        # Fallback: any version (newest first)
        candidates = [s for s in entries if s.get("skill_id") == skill_id]
        if not candidates:
            raise ValueError(f"Skill '{skill_id}' not found in skills.json")
        skill = sorted(candidates, key=lambda s: s.get("version", 0), reverse=True)[0]

    # Load template — may be an inline string or a relative file path.
    # Heuristic: if it contains spaces or newlines, or has no file-like extension,
    # treat it as an inline template.  Otherwise try to load it as a file.
    template_val  = skill.get("template", "")
    template_text = ""
    if template_val:
        _is_inline = (
            " " in template_val          # inline text always has spaces
            or "\n" in template_val      # multi-line inline
            or "{{" in template_val      # Jinja-style placeholder → inline
            or not Path(template_val).suffix  # no file extension
        )
        if _is_inline:
            template_text = template_val
        else:
            template_path = base_path / "skills" / template_val
            if not template_path.exists():
                template_path = base_path / template_val
            if template_path.exists():
                template_text = template_path.read_text(encoding="utf-8")
            else:
                # Fall back to treating the value as inline text
                template_text = template_val

    return {"skill": skill, "template": template_text}
