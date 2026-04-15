"""
llm_call
--------
Pipeline resolver + provider router for all LLM calls.

Flow:
  pipeline tag (or direct model_id)
      → resolve model_id from config.pipelines
      → look up provider in models.json
      → load API key from keys.json
      → import provider adapter from tools/providers/{provider}.py
      → adapter.build_request() → adapter.call() → adapter.parse_response()
      → enforce token_budget if set in context
      → return standard response dict

params:
    pipeline    : str              - pipeline name (e.g. "orchestration") or direct model_id
    system      : str              - system prompt
    messages    : list[dict]       - conversation messages [{"role":..,"content":..}]
    max_tokens  : int | None       - generation limit (default: 1024)
    temperature : float | None     - sampling temperature
    tools       : list[dict] | None - tool definitions (provider-specific format)

returns (standard response):
    content     : str
    tokens_in   : int
    tokens_out  : int
    model_id    : str
    provider    : str
    stop_reason : str
    tool_calls  : list[dict]
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run(params: dict, context: dict) -> dict[str, Any]:
    pipeline    = params["pipeline"]
    system      = params.get("system", "")
    messages    = params["messages"]
    max_tokens  = params.get("max_tokens", 1024)
    temperature = params.get("temperature")
    tools       = params.get("tools")

    base_path = Path(context["base_path"])
    config    = context.get("config", {})

    # ------------------------------------------------------------------
    # 1. Resolve pipeline → model_id
    # ------------------------------------------------------------------
    pipelines = config.get("pipelines", {})
    model_id  = pipelines.get(pipeline, pipeline)  # fallback: treat as direct model_id

    # ------------------------------------------------------------------
    # 2. Look up provider in models.json
    # ------------------------------------------------------------------
    models_path = base_path / "models.json"
    if not models_path.exists():
        raise FileNotFoundError(f"models.json not found at {models_path}")

    raw_models = json.loads(models_path.read_text())
    # models.json may be a list or a single object
    if isinstance(raw_models, dict):
        raw_models = [raw_models]

    model_entry = next(
        (m for m in raw_models if m.get("model_id") == model_id and m.get("enabled", True)),
        None,
    )
    if model_entry is None:
        raise ValueError(
            f"Model '{model_id}' not found or disabled in models.json"
        )

    provider_name = model_entry["provider"]

    # ------------------------------------------------------------------
    # 3. Load API key from keys.json
    # ------------------------------------------------------------------
    keys_path = base_path / "keys.json"
    keys: dict = {}
    if keys_path.exists():
        try:
            keys = json.loads(keys_path.read_text())
        except Exception:
            pass
    api_key = keys.get(provider_name)

    # ------------------------------------------------------------------
    # 4. Import provider adapter
    # ------------------------------------------------------------------
    provider_file = base_path / "tools" / "providers" / f"{provider_name}.py"
    if not provider_file.exists():
        raise FileNotFoundError(
            f"Provider adapter not found: {provider_file}. "
            f"Supported providers: anthropic, openai, ollama"
        )

    # Use importlib.import_module (not spec_from_file_location) so that relative
    # imports inside the provider files (e.g. `from .base import BaseAdapter`) work.
    import sys as _sys
    _pikaia = str(base_path)
    if _pikaia not in _sys.path:
        _sys.path.insert(0, _pikaia)
    _full_mod = f"tools.providers.{provider_name}"
    mod = _sys.modules.get(_full_mod) or importlib.import_module(_full_mod)

    adapter = mod.Adapter(api_key=api_key, model_id=model_id)

    # ------------------------------------------------------------------
    # 5. Build + call  (with DeepSeek-R1 local fallback on failure)
    # ------------------------------------------------------------------
    request = adapter.build_request(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
    )
    try:
        raw      = adapter.call(request)
        response = adapter.parse_response(raw)
    except Exception as primary_exc:
        fallback_enabled = config.get("deepseek_fallback_enabled", True)
        if fallback_enabled and provider_name != "deepseek_local":
            logger.warning(
                "llm_call: provider '%s' failed (%s) — trying DeepSeek-R1 local fallback",
                provider_name, primary_exc,
            )
            try:
                import sys as _sys2
                _pikaia2 = str(base_path)
                if _pikaia2 not in _sys2.path:
                    _sys2.path.insert(0, _pikaia2)
                from tools.providers.deepseek_local import Adapter as _DSAdapter  # type: ignore[import]
                ds_adapter  = _DSAdapter(api_key=None, model_id="deepseek-r1:1.5b")
                ds_request  = ds_adapter.build_request(
                    system=system, messages=messages,
                    max_tokens=max_tokens, temperature=temperature, tools=tools,
                )
                raw      = ds_adapter.call(ds_request)
                response = ds_adapter.parse_response(raw)
                logger.info("llm_call: DeepSeek fallback succeeded for pipeline '%s'", pipeline)
            except Exception as ds_exc:
                logger.error("llm_call: DeepSeek fallback also failed: %s", ds_exc)
                raise primary_exc from None  # raise the original error
        else:
            raise

    # ------------------------------------------------------------------
    # 6. Token budget enforcement
    # ------------------------------------------------------------------
    token_budget = context.get("token_budget")
    if token_budget is not None:
        used = response.get("tokens_in", 0) + response.get("tokens_out", 0)
        remaining = token_budget - used
        context["token_budget"] = remaining
        if remaining < 0:
            logger.warning(
                "Token budget overrun: used %d, budget was %d", used, token_budget
            )

    return response
