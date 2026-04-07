"""
ToolRegistry
------------
Loads all enabled tools from tools.json, enforces caller permissions,
and dispatches run(params, context) for each call.

Usage (orchestrator wires this up):

    registry = ToolRegistry(
        base_path="/path/to/Pikaia",
        project="my-project",
        instance_id="inst_abc",
        caller="orchestrator",
    )
    # Pass registry.dispatch as the Tools dispatch function:
    tools = Tools(dispatch=registry.dispatch)
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(
        self,
        base_path:   str,
        project:     str,
        instance_id: str,
        caller:      str = "orchestrator",   # "orchestrator" | "agent" | "skillsmith"
        agent_id:    str | None = None,
        worker_dir:  str | None = None,
        token_budget: int | None = None,
    ) -> None:
        self.base_path   = str(base_path)
        self.caller      = caller
        self.context: dict[str, Any] = {
            "base_path":    self.base_path,
            "project":      project,
            "instance_id":  instance_id,
            "agent_id":     agent_id,
            "caller":       caller,
            "worker_dir":   worker_dir,
            "token_budget": token_budget,
            "config":       self._load_config(self.base_path, project),
        }
        self._tools: dict[str, tuple[Any, dict]] = {}
        self._load_tools()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def dispatch(self, name: str, params: dict) -> Any:
        """Main entry point. Called by Orchestrator's Tools.dispatch."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found or not enabled")

        mod, entry = self._tools[name]

        # Permission check
        permissions = entry.get("permissions", [])
        if self.caller not in permissions:
            raise PermissionError(
                f"Caller '{self.caller}' is not permitted to use tool '{name}'. "
                f"Allowed: {permissions}"
            )

        return mod.run(params, self.context)

    def update_context(self, **kwargs: Any) -> None:
        """Allow orchestrator to update context mid-session (e.g. token_budget)."""
        self.context.update(kwargs)

    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_tools(self) -> None:
        tools_json = Path(self.base_path) / "tools" / "tools.json"
        if not tools_json.exists():
            logger.warning("tools.json not found at %s", tools_json)
            return

        entries: list[dict] = json.loads(tools_json.read_text())
        for entry in entries:
            if not entry.get("enabled", True):
                continue
            tool_id   = entry["tool_id"]
            impl_rel  = entry["impl"]                         # e.g. "tools/impl/shell_exec.py"
            full_path = Path(self.base_path) / impl_rel
            if not full_path.exists():
                logger.warning("Tool impl not found: %s", full_path)
                continue
            try:
                mod = self._import_module(tool_id, str(full_path))
                self._tools[tool_id] = (mod, entry)
            except Exception as exc:
                logger.error("Failed to load tool '%s': %s", tool_id, exc)

    @staticmethod
    def _import_module(name: str, path: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
        spec.loader.exec_module(mod)                        # type: ignore[union-attr]
        return mod

    @staticmethod
    def _load_config(base_path: str, project: str) -> dict:
        cfg: dict = {}
        global_path = Path(base_path) / "config.json"
        if global_path.exists():
            try:
                cfg.update(json.loads(global_path.read_text()))
            except Exception:
                pass
        proj_path = Path(base_path) / "projects" / project / "config.json"
        if proj_path.exists():
            try:
                proj = json.loads(proj_path.read_text())
                if "pipelines" in proj:
                    cfg.setdefault("pipelines", {}).update(proj["pipelines"])
                for k, v in proj.items():
                    if k != "pipelines":
                        cfg[k] = v
            except Exception:
                pass
        return cfg
