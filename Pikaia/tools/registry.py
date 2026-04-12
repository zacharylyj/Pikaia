"""
ToolRegistry
------------
Loads all enabled tools from tools.json, enforces caller permissions,
and dispatches run(params, context) for each call.

Tool execution standardisation (item 7)
-----------------------------------------
Every tool result is normalised to a ToolResult before being returned to
the caller.  If a tool's run() already returns a dict with "success", it is
passed through unchanged.  Plain dicts and non-dict values are wrapped.

    ToolResult fields
    -----------------
    success : bool   — True if the call completed without exception
    data    : Any    — the tool's actual output (original return value)
    error   : str    — empty string on success; exception message on failure

Callers that want raw dicts (e.g. agent._tool_loop storing result_str) can
call dispatch() as before — the ToolResult is a TypedDict so it's fully
JSON-serialisable.

Usage (orchestrator wires this up):

    registry = ToolRegistry(
        base_path="/path/to/Pikaia",
        project="my-project",
        instance_id="inst_abc",
        caller="orchestrator",
    )
    tools = Tools(dispatch=registry.dispatch)
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolResult — standardised return shape
# ---------------------------------------------------------------------------

class ToolResult(TypedDict):
    success: bool
    data:    Any
    error:   str


def _normalise(raw: Any) -> ToolResult:
    """
    Wrap a raw tool return value into a ToolResult dict.

    - If raw is already a dict with a "success" key, pass it through
      (tool opted into the standard format explicitly).
    - Otherwise wrap it as {success: True, data: raw, error: ""}.
    """
    if isinstance(raw, dict) and "success" in raw:
        return ToolResult(
            success=bool(raw.get("success", True)),
            data=raw.get("data", raw),
            error=str(raw.get("error", "")),
        )
    return ToolResult(success=True, data=raw, error="")


def _error_result(exc: Exception) -> ToolResult:
    return ToolResult(success=False, data=None, error=str(exc))


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(
        self,
        base_path:    str,
        project:      str,
        instance_id:  str,
        caller:       str = "orchestrator",
        agent_id:     str | None = None,
        worker_dir:   str | None = None,
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
        """
        Main entry point.  Dispatches to the tool impl and returns a
        normalised ToolResult dict.

        Raises ValueError for unknown tools, PermissionError for caller
        violations (both propagate unchanged — callers handle them).
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found or not enabled")

        mod, entry = self._tools[name]

        permissions = entry.get("permissions", [])
        if self.caller not in permissions:
            raise PermissionError(
                f"Caller '{self.caller}' is not permitted to use tool '{name}'. "
                f"Allowed: {permissions}"
            )

        try:
            raw = mod.run(params, self.context)
            return _normalise(raw)
        except (PermissionError, ValueError):
            # Re-raise access/validation errors without wrapping
            raise
        except Exception as exc:
            logger.warning("Tool '%s' raised exception: %s", name, exc)
            raise

    def dispatch_raw(self, name: str, params: dict) -> Any:
        """
        Like dispatch() but returns the raw tool output without normalisation.
        Used internally where callers need the original dict shape.
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found or not enabled")
        mod, entry = self._tools[name]
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
            impl_rel  = entry["impl"]
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
