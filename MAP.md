# Pikaia — Architecture Map

Every `.py` file, its functions, what data flows in/out, and how they wire together.

---

## System Flow (top-level)

```
User input
    │
    ▼
main.py  ──creates──►  Orchestrator.py  ──spawns thread──►  agent.py
    │                        │                                   │
    │                        │ builds context                    │ tool loop
    │                   context_manager.py                       │
    │                        │                            tools/registry.py
    │                   tools/registry.py                        │
    │                        │                           tools/impl/*.py
    │                   tools/impl/*.py                          │
    │                        │                          tools/providers/*.py
    │                   tools/providers/*.py                     │
    │                        │                           External APIs
    │                   memory/  projects/              memory/  projects/
    │                        │                                   │
    │                        └──► db.py / metrics.py / trajectory.py
    │
    └── init.py  (first-run scaffolding, integrity check, test runner → test_tools.py)
```

---

## Module Detail

---

### `main.py`
**Role:** CLI entry point. REPL loop, `/command` dispatcher, project/instance lifecycle.

| Function | In | Out | Connects to |
|---|---|---|---|
| `main()` | `--project`, `--instance`, `--debug`, `--groq`, `--ollama` CLI args | — (starts REPL) | `Orchestrator`, `ToolRegistry`, `OrchestratorConfig` |
| `_make_orchestrator(project, instance_id, debug, groq, ollama)` | `str, str, bool, bool, bool` | `(Orchestrator, ToolRegistry)` | `Orchestrator.py`, `tools/registry.py`, `config.json` |
| `_ensure_project(project)` | `str` | — | `projects/{project}/` directory tree |
| `_create_or_resume(project, instance_id)` | `str, str\|None` | `str` instance_id | `projects/{project}/instances/` |
| `_dispatch_command(line, ...)` | `str` command line | `(Session, bool)` should_exit | all `_cmd_*` helpers |
| `_cmd_approve(project)` | `str` | — (interactive prompt) | `ct.json`, `skills/skills.json` |
| `_cmd_status/memory/skills/models/instances/files/check` | `str` project | — (prints) | `*.json` data files |

**CLI flags:**
- `--project NAME` — project workspace (default: `default`)
- `--instance ID` — resume an existing session
- `--debug` — mock LLM calls, no API key needed
- `--groq` — route all pipelines through Groq API
- `--ollama` — route all pipelines through local Ollama

**Key data types:**
- `Session = tuple[str, str, Orchestrator, ToolRegistry]`
- `_DEBUG_MODE`, `_GROQ_MODE`, `_OLLAMA_MODE` — module-level flags

---

### `Orchestrator.py`
**Role:** Full turn loop — context → intent → skill → dispatch → monitor → post-process.

#### Classes

| Class | Fields | Role |
|---|---|---|
| `OrchestratorConfig` | 15 scalar fields + `pipelines: dict[str,str]` | All tunable knobs; loaded from `config.json` |
| `TurnContext` | `lt_entries, mt_entries, ct_active, st_summary, st_window, project_index, relevant_files` | Context bundle assembled once per turn |
| `AgentRecord` | `agent_id, task_id, project, pipeline, tier, status, worker_dir, ...` | Tracks a live agent dispatch |
| `SkillMatch` | `skill_id, name, tier, score, tools_ok, pipeline` | Result of `_skill_pick` |
| `Tools` | wraps `dispatch: Callable` | Thin façade — one method per tool |

#### Key methods on `Orchestrator`

| Method | In | Out | Connects to |
|---|---|---|---|
| `turn(message)` | `str` | `str` response | all steps below |
| `_build_context(message)` | `str` | `TurnContext` | `tools.memory_read` (lt/mt/ct), `embed_text`, `preferences.json`, `file_index.json` |
| `_understand_intent(message, ctx)` | `str, TurnContext` | `(str, IntentType)` | `tools.llm_call` (classification pipeline) |
| `_skill_pick(message)` | `str` | `SkillMatch \| None` | `tools.embed_text`, `skills/skills.json` |
| `_dispatch(message, match, ctx)` | `str, SkillMatch, TurnContext` | `AgentRecord` | `tools.memory_write` (ct), `_spawn_agent`, `context_manager.assess` |
| `_build_task_packet(message, match, ctx, record)` | — | `dict` task packet | `tools.skill_read` |
| `_spawn_agent(record, task_packet)` | `AgentRecord, dict` | — (daemon thread) | `agent.AgentRunner` |
| `_generate_ack(record, task_packet)` | — | `dict` ack | `tools.llm_call` (ack_validation pipeline) |
| `_validate_ack(ack, packet, record)` | — | `(bool, str)` | — |
| `_await_agent(record)` | `AgentRecord` | `dict` result | polls `worker/{id}/result.json` |
| `_auto_promote(record, result)` | — | — | `tools.file_move`, `_reindex_file` |
| `_reindex_file(path)` | `str` | — | `tools.llm_call` (file_indexing), `tools.embed_text`, `dev/index.json`, `file_index.json` |
| `_post_process(message, result, ctx)` | — | — | `_update_st`, `_append_history`, `_mt_judge` |
| `_update_st(user_msg, assistant_msg)` | `str, str` | — | `tools.llm_call` (compression), `st.json` |
| `_append_history(role, content)` | `str, str` | — | `instances/{id}/history.json` |
| `_mt_judge(user_msg, assistant_msg)` | `str, str` | — | `tools.llm_call` (mt_judge), `tools.memory_write` (mt) |
| `_trigger_skillsmith(message, ctx)` | `str, TurnContext` | — | `tools.llm_call` (skillsmith_draft/eval), `tools.memory_write` (ct), `tools.embed_text` |
| `_monitor_loop()` | — | — (background thread) | polls `worker/{id}/state.json`, calls `_kill_agent` |

**`Tools._unwrap(result)`** — static method: extracts `.data` from a `ToolResult` dict or returns raw value as-is. Applied to all methods that consume tool return values (`llm_call`, `embed_text`, `memory_read`, `file_read`, `skill_read`).

**Task packet schema** (passed to agent):
```
{
  task_id, agent_id, objective, success_criteria, constraints,
  context: { lt_summary, mt_retrieved, ct_active, st_summary,
             project_index, relevant_files },
  skill, skill_id, tier, pipeline,
  tools_allowed: list[str],
  token_budget, timeout_secs,
  file_budget: { max_files, files_fetched },
  max_steps?,        # per-task override for agent step budget
  fast_model?,       # per-task override for fast-model routing
  planned_steps?,    # added by _spawn_agent after ack
  restatement?
}
```

---

### `agent.py`
**Role:** Agent execution engine for all 4 tiers. Called in a daemon thread by `Orchestrator._spawn_agent`.

#### Helper classes and functions

| Class / Function | In | Out | Role |
|---|---|---|---|
| `_KeyPool` | `keys: list[str]` | — | Round-robin key rotation with per-key cooldown tracking |
| `_KeyPool.get()` | — | `str \| None` | Returns next available key; skips keys on cooldown |
| `_KeyPool.mark_failed(key, cooldown_secs)` | `str, float` | — | Puts key on cooldown after 429/auth error |
| `_build_key_pool(provider, base_path)` | `str, Path` | `_KeyPool` | Reads `keys.json`, handles single string or list |
| `_should_use_fast_model(task_packet, config)` | `dict, dict` | `bool` | True if prompt ≤ N words AND tools ≤ N |
| `_load_adapter(pipeline, base_path, api_key?)` | `str, Path, str\|None` | `(Adapter, model_id, provider_name)` | Loads provider from `models.json`/`keys.json` |

#### BaseAgent

| Method | In | Out | Connects to |
|---|---|---|---|
| `__init__(task_packet, record, base_path)` | `dict, dict, Path` | — | `ToolRegistry`, `_load_adapter`, `_build_key_pool`, `MetricsCollector`, `TrajectoryLogger` |
| `_context_window_size()` | — | `int` | reads `models.json` context window |
| `_compress_messages(messages, step)` | `list[dict], int` | `list[dict]` | Keeps last 6 messages, summarises earlier; logs `compress` trajectory step |
| `_partition_tool_calls(tool_calls)` | `list[dict]` | `(parallel: list, sequential: list)` | Splits by `_PARALLEL_SAFE_TOOLS` membership |
| `_dispatch_tool(tc, step)` | `dict, int` | `str` result | Calls `ToolRegistry.dispatch`; records `tool_call`/`tool_result` in trajectory + metrics |
| `_execute_tool_calls(tool_calls, step)` | `list[dict], int` | `dict[id→str]` | Parallel via `ThreadPoolExecutor` + sequential dispatch |
| `_tool_loop(system, messages, tools_allowed)` | `str, list, list[str]` | `(str content, int tokens)` | Full ReAct loop with budget, error classification, compression, key rotation |
| `_load_skill_template()` | — | `str` template | `ToolRegistry.dispatch("skill_read")` + unwrap |
| `_write_state(step, total, tokens, status?, issues?)` | ints + optional str | — | `worker/{id}/state.json` |
| `_mark_done(output, confidence)` | `str, float` | — | `worker/{id}/result.json`, `state.json`, `_finalise_observability()` |
| `_mark_failed(reason)` | `str` | — | `worker/{id}/result.json`, `state.json`, `_finalise_observability()` |
| `_finalise_observability()` | — | — | `MetricsCollector.flush(db)`, `TrajectoryLogger.finalise(outcome, output, db)` |

**`_PARALLEL_SAFE_TOOLS`** (frozenset): `file_read`, `memory_read`, `embed_text`, `web_fetch`, `http_request`, `context_fetch`, `skill_read`, `code_exec` — dispatched concurrently via `ThreadPoolExecutor`.

**Error classification via `tools/error_types.py`:**

| ErrorType | Recovery in `_tool_loop` |
|-----------|--------------------------|
| `RATE_LIMIT` | Rotate API key → retry with exponential backoff |
| `AUTH` | Abort immediately |
| `CONTEXT_OVERFLOW` | `_compress_messages()` → retry |
| `NETWORK` | Retry with exponential backoff |
| `UNKNOWN` | Log + break loop |

#### Tier classes

| Class | Method | Connects to |
|---|---|---|
| `Tier12Agent` | `run()` | `_tool_loop` once |
| `Tier3Agent` | `run()` | `_decompose`, N × `_tool_loop`, `_synthesize` |
| `Tier3Agent` | `_decompose(objective, template)` → `list[str]` steps | `Adapter.build_request/call/parse_response`, `_strip_json_fences` |
| `Tier4Council` | `run()` | N parallel `_tool_loop` threads + `_council_synthesis` |
| `AgentRunner` | `run(task_packet, record, base_path)` | instantiates correct tier class |

**Agent tier map:**
```
tier 1 → Tier12Agent  (atomic — single tool call expected)
tier 2 → Tier12Agent  (composite — multi-step continuous loop)
tier 3 → Tier3Agent   (decompose → step loop → synthesize)
tier 4 → Tier4Council (3 parallel specialists → council synthesis)
```

**Worker directory files written:**
```
worker/{agent_id}/
  ack.json      { task_id, restatement, planned_steps, confidence, ... }
  state.json    { task_id, status, step_current, step_total, tokens_used, issues }
  result.json   { status, output, confidence }
  task.json     (copy of enriched task packet)
  meta.json     (AgentRecord fields)
  todos.json    (written by todo_write tool, optional)
  question.json (pending question to user, written by question tool, optional)
  answer.json   (user answer, written by orchestrator, optional)
```

---

### `db.py`
**Role:** Thread-safe SQLite WAL backend. Singleton per path, shared across all modules.

| Function / Method | In | Out | Role |
|---|---|---|---|
| `get_db(path?)` | `str \| Path \| None` | `PikaiaDB` | Singleton factory; creates tables on first call |
| `PikaiaDB.log_trajectory(task_id, tier, outcome, steps_json, ...)` | — | — | INSERT into `trajectories` |
| `PikaiaDB.log_tool_event(task_id, tool_name, success, latency_ms)` | — | — | INSERT into `tool_events` |
| `PikaiaDB.log_metric(task_id, name, value)` | — | — | INSERT into `metrics` |
| `PikaiaDB.log_metrics_batch(task_id, metrics_dict)` | `str, dict` | — | Batch INSERT into `metrics` |
| `PikaiaDB.metrics_summary(task_id?)` | `str \| None` | `dict` | Aggregated metrics; optional task filter |
| `PikaiaDB.tool_success_rate(tool_name?)` | `str \| None` | `dict` | Per-tool or overall success rate |

**Tables:**
```
trajectories  — task_id, tier, outcome, steps (JSON), ts
tool_events   — task_id, tool_name, success, latency_ms, ts
metrics       — task_id, name, value, ts
```

---

### `metrics.py`
**Role:** Per-run metrics accumulator. Created by `BaseAgent.__init__`, flushed at run end.

| Class / Method | In | Out | Role |
|---|---|---|---|
| `MetricsCollector(task_id, enabled)` | `str, bool` | — | Init with empty counters |
| `record_llm(tokens_in, tokens_out)` | `int, int` | — | Accumulate token counts |
| `record_step()` | — | — | Increment step counter |
| `record_tool(tool_name, success, latency_ms)` | `str, bool, float` | — | Accumulate tool event |
| `flush(db)` | `PikaiaDB` | — | `db.log_metrics_batch` + `db.log_tool_event` for all recorded tools |
| `total_tokens` (property) | — | `int` | `tokens_in + tokens_out` |
| `elapsed_ms` (property) | — | `float` | Milliseconds since `__init__` |
| `tool_success_rate` (property) | — | `float` | Fraction of tool calls that succeeded |

---

### `trajectory.py`
**Role:** Per-run step-by-step replay buffer. Created by `BaseAgent.__init__`, finalised at run end.

| Class / Method | In | Out | Role |
|---|---|---|---|
| `TrajectoryLogger(task_id, project, base_path, enabled)` | — | — | Init; sets JSONL path to `projects/<project>/trajectories/<task_id>.jsonl` |
| `log(step_type, data)` | `str, dict` | — | Append step to in-memory buffer |
| `finalise(outcome, output, db)` | `str, str, PikaiaDB` | — | Write JSONL file + `db.log_trajectory(...)` |

**Step types:** `llm_turn`, `tool_call`, `tool_result`, `compress`

---

### `tools/error_types.py`
**Role:** Centralised error classification for all LLM and tool failures.

| Class / Function | In | Out | Role |
|---|---|---|---|
| `ErrorType` (enum) | — | — | `RATE_LIMIT`, `AUTH`, `CONTEXT_OVERFLOW`, `NETWORK`, `TOOL`, `UNKNOWN` |
| `classify_error(exc)` | `Exception` | `ErrorType` | String-match on exception message; ordered most-specific first |

---

### `context_manager.py`
**Role:** Pre-dispatch context gap assessment + on-demand agent context fetch.

| Method / Function | In | Out | Connects to |
|---|---|---|---|
| `ContextManager.__init__(tools, base_path, config)` | `Tools, Path, OrchestratorConfig` | — | holds reference to `Tools` |
| `ContextManager.assess(task_packet, project)` | `dict, str` | `dict` enriched packet | `tools.llm_call` (context_assessment), `tools.memory_read` (mt), `tools.embed_text`, `dev/index.json` |
| `ContextManager.fetch(query, project, base_path, context)` | `str, str, Path, dict` | `dict {mt_entries, files, text}` | `_embed` (embed_text.py), `mt.json`, `dev/index.json`, reads file snippets |
| `_strip_json_fences(text)` | `str` | `str` | — |
| `_embed(text, context)` | `str, dict` | `list[float] \| None` | `tools/impl/embed_text.py` via spec_from_file_location |
| `_cosine(a, b)` | `list[float], list[float]` | `float` | — |

---

### `mt_palace.py`
**Role:** MemPalace MT storage/retrieval engine. Wing/Room tagging, AAAK compression, KG triple store, 4-layer retrieval.

| Class / Function | In | Out | Role |
|---|---|---|---|
| `RoomDetector.detect(text)` | `str` | `(wing, room)` | Keyword → wing/room taxonomy |
| `EntityExtractor.extract(text)` | `str` | `{persons, projects}` | Regex-based entity detection |
| `AAAKCodec.compress(entry, entities, room, existing_codes)` | `dict, dict, str, set` | `dict` compressed entry | AAAK lossy compression |
| `ImportanceScorer.score(entry)` | `dict` | `float` 0–1 | Multi-signal importance scoring |
| `KnowledgeGraph.add_triple(subject, predicate, object, ...)` | strings | `dict` | Append temporal triple to `kg.json` |
| `KnowledgeGraph.query(subject?, predicate?, object?, as_of?)` | optional filters | `list[dict]` | Temporal triple query |
| `KnowledgeGraph.subject_timeline(subject)` | `str` | `list[dict]` | Full history for a subject |
| `MTWriter.write(entry, context)` | `dict, dict` | `dict` stored entry | Full MT write pipeline: room tagging → AAAK → KG → `mt.json` |
| `MTReader.read(query, top_k, context, wing?, room?, palace_layer?)` | query+filters | `list[dict]` | L1/L2/L3 retrieval from `mt.json` |
| `kg_read(params, base_path)` | filter params | `list[dict]` | Entry point for `memory_read` tool KG layer |
| `kg_write(params, base_path)` | triple data | `dict` | Entry point for `memory_write` tool KG layer |

**Memory files:**
```
memory/mt.json   list[MTEntry]  — MemPalace entries with embedding + palace fields
memory/kg.json   list[Triple]   — {id, subject, predicate, object, valid_from, valid_to}
memory/lt.json   list[LTEntry]  — permanent preferences/facts
```

---

### `init.py`
**Role:** First-run setup wizard, project scaffolding, integrity checker (11 checks), test runner.

| Function | In | Out | Connects to |
|---|---|---|---|
| `scaffold(base)` | `Path` | — | Creates `memory/`, `skills/`, `tools/`, data JSON files |
| `scaffold_project(project, base)` | `str, Path` | — | Creates `projects/{project}/` tree + `trajectories/` dir |
| `collect_keys(base)` | `Path` | — | Interactive prompt → `keys.json` |
| `check(base)` | `Path` | `CheckResult` | Runs 11 integrity checks (see below) |
| `fix(base)` | `Path` | — | Auto-repairs recoverable issues incl. missing `trajectories/` dirs |
| `setup_wizard(first_project)` | `str` | — | Calls `scaffold`, `scaffold_project`, `collect_keys` |
| `main()` | CLI args `--check`, `--fix`, `--project`, `--test`, `--tool`, `--fast` | — | Entry point; delegates `--test` to `test_tools.py` |

**11 integrity checks:**
1. Directory structure
2. JSON file validity
3. Tool impl files + required fields
4. Tool schema coverage (all tools.json entries have a schema)
5. Pipeline model coverage + fast_model validity
6. Core module file integrity (`agent.py`, `db.py`, `metrics.py`, `trajectory.py`, `error_types.py`, etc.)
7. Config key completeness (all required keys present)
8. Skill embeddings present
9. Stale CT flags (>24h open)
10. File index coverage
11. Observability paths (`trajectories/` exists, `pikaia.db` writable)

---

### `test_tools.py`
**Role:** Functional tool test suite — 55 tests, no API key required.

| Function | In | Out | Role |
|---|---|---|---|
| `_test(tool_id, name, tags?)` | decorator args | decorated fn | Registers test; `tags={"network","slow"}` skipped with `--fast` |
| `_load_tool(name)` | `str` | module | Imports `tools/impl/{name}.py` directly from `tools.json` path |
| `_ctx(tmp, caller, project?, agent_id?)` | `Path, str, ...` | `dict` context | Builds minimal tool context pointing at `tmp` directory |
| `run_tests(base_path, tool?, fast?)` | `Path, str\|None, bool` | `int` exit code | Runs registered tests; prints PASS/FAIL/SKIP per test |
| `TestResults` | — | — | Accumulates pass/fail/skip counts |

**Test groups:** shell_exec, code_exec, file_read, file_write, file_delete, file_move, edit, grep, glob, list, apply_patch, todo_write, web_search (tagged network), question, http_request (tagged network), web_fetch (tagged network), memory (read/write), embed_text, llm_call (tagged network)

---

### `tools/registry.py`
**Role:** Tool loader + dispatcher. Reads `tools/tools.json`, enforces caller permissions, calls `mod.run(params, context)`, wraps result in `ToolResult`.

| Method | In | Out | Connects to |
|---|---|---|---|
| `__init__(base_path, project, instance_id, caller, agent_id?, worker_dir?, token_budget?)` | strings | — | `tools/tools.json`, `config.json` |
| `dispatch(name, params)` | `str, dict` | `ToolResult` | `tools/impl/{name}.py` → `mod.run(params, context)` → `_normalise(raw)` |
| `dispatch_raw(name, params)` | `str, dict` | `Any` | Raw tool output without normalisation (backward compat) |
| `update_context(**kwargs)` | key=value | — | updates `self.context` dict in place |
| `available_tools()` | — | `list[str]` | — |
| `_normalise(raw)` | `Any` | `ToolResult` | Wraps any return value into `{success, data, error}` |

**`ToolResult` (TypedDict):**
```python
{
  "success": bool,
  "data":    Any,    # raw tool return value on success
  "error":   str,    # error message on failure (empty string on success)
}
```

**`context` dict passed to every tool `run()`:**
```python
{
  "base_path":    str,
  "project":      str,
  "instance_id":  str,
  "agent_id":     str | None,
  "caller":       "orchestrator" | "agent" | "skillsmith",
  "worker_dir":   str | None,
  "token_budget": int | None,
  "config":       dict,   # merged global + project config.json
}
```

---

### `tools/schemas.py`
**Role:** Anthropic-format JSON schemas for all 26 tools. Supports self-registering `SCHEMA` dicts in impl modules.

| Function | In | Out | Role |
|---|---|---|---|
| `get_schemas(tool_names)` | `list[str]` | `list[dict]` | Returns Anthropic tool schemas; merged built-ins + discovered |
| `_discover_impl_schemas(impl_dir?)` | `Path \| None` | `dict[str, dict]` | Scans `tools/impl/*.py` for module-level `SCHEMA` dicts |
| `_get_merged_schemas()` | — | `dict[str, dict]` | Cached merge: built-in SCHEMAS + impl-discovered; impl takes precedence |
| `invalidate_schema_cache()` | — | — | Clears cache, forces re-discovery on next call |

**Self-registering pattern:**
```python
# tools/impl/my_tool.py
SCHEMA = {
    "name": "my_tool",
    "description": "...",
    "input_schema": {"type": "object", "properties": {...}, "required": [...]},
}
```

---

## Provider Adapters (`tools/providers/`)

All implement `BaseAdapter`. Loaded dynamically via `importlib.import_module("tools.providers.{name}")`.

| File | Provider | `build_request` key fields | `call` returns | `parse_response` returns |
|---|---|---|---|---|
| `base.py` | Abstract | — | — | `{content, content_blocks, tokens_in, tokens_out, model_id, provider, stop_reason, tool_calls}` |
| `anthropic.py` | Anthropic API | `model, messages, system, max_tokens, tools` | Anthropic SDK `Message` object | standard response dict |
| `openai.py` | OpenAI API | `model, messages, max_tokens, tools` | OpenAI response JSON | standard response dict |
| `groq.py` | Groq API | `model, messages, max_tokens, tools` | Groq response JSON | standard response dict |
| `ollama.py` | Local Ollama | `model, messages, stream, options` | Ollama JSON response | standard response dict (text injection fallback) |
| `debug.py` | Mock (no network) | passthrough `{system, messages}` | `{"_content": str}` | standard response dict (canned) |

`debug.py` detects pipeline from system prompt keywords → returns shaped canned JSON for classification, skillsmith, ack, mt_judge, context_assessment, task_planning, compression, file_indexing, generic.

---

## Tool Implementations (`tools/impl/`)

All expose `run(params: dict, context: dict) -> dict`. All results wrapped in `ToolResult` by `registry.dispatch`.

### File & Code Tools

| Tool | Params in | Returns | Notes |
|---|---|---|---|
| `shell_exec` | `cmd, cwd?, timeout?` | `{stdout, stderr, returncode}` | subprocess |
| `code_exec` | `code, language?, timeout?` | `{stdout, stderr, returncode}` | subprocess (python/node sandbox) |
| `file_read` | `path, offset?, limit?` | `{content, path, size_bytes, lines, truncated}` | `offset` = 1-based line number; returns line count + truncated flag |
| `file_write` | `path, content` | `{written, path}` | Agents: worker slot only |
| `file_delete` | `path` | `{deleted, path}` | Orchestrator only |
| `file_move` | `src, dst` | `{moved, src, dst}` | Orchestrator only |
| `edit` | `path, old_string, new_string, replace_all?` | `{edited, path, replacements}` | Enforces uniqueness: error if 0 matches; error if >1 and `replace_all=False` |
| `apply_patch` | `patch, path?, dry_run?, strip?` | `{applied, patched_files}` | System `patch` command; Python difflib fallback |

### Search & Navigation Tools

| Tool | Params in | Returns | Notes |
|---|---|---|---|
| `grep` | `pattern, path?, glob?, type?, context?, ignore_case?, output_mode?, max_results?` | `{matches, count}` | Tries `rg` first; Python re fallback; modes: `files_with_matches`, `content`, `count` |
| `glob` | `pattern, path?` | `{files}` sorted by mtime desc | Tries `rg --files --glob`; pathlib fallback; MAX 500 |
| `list` | `path?, recursive?` | `{entries: [{name, type, size_bytes, modified}]}` | type: file/dir/symlink; MAX 1000 entries |

### Web & HTTP Tools

| Tool | Params in | Returns | Notes |
|---|---|---|---|
| `web_fetch` | `url, max_chars?, timeout?` | `{url, content, truncated}` | HTTP + HTML strip |
| `web_search` | `query, max_results?` | `{results: [{title, url, snippet}]}` | DuckDuckGo HTML endpoint; no API key |
| `http_request` | `method, url, headers?, body?, timeout?` | `{status_code, headers, body, ok}` | Generic REST |

### Memory & Context Tools

| Tool | Params in | Returns | Notes |
|---|---|---|---|
| `memory_read` | `layer, query?, top_k?, project?, instance_id?, wing?, room?, palace_layer?, subject?, predicate?, object?, as_of?, subject_timeline?` | `list[dict]` or `dict` (ST) | MemPalace / KG layers fully supported |
| `memory_write` | `layer, entry, project?, instance_id?` | `{written, layer}` | Orchestrator only |
| `context_fetch` | `query, top_k?, include_files?, max_chars_per_file?` | `{text, mt_entries, files}` | `context_manager.ContextManager.fetch` |

### LLM & Skill Tools

| Tool | Params in | Returns | Notes |
|---|---|---|---|
| `llm_call` | `pipeline, system?, messages, max_tokens?, temperature?` | `{content, tokens_in, tokens_out, stop_reason}` | Pipeline resolver + provider router |
| `embed_text` | `text` | `{embedding: list[float], dim, model}` | OpenAI → Ollama → hash fallback |
| `skill_read` | `skill_id` | `{skill_id, name, tier, template, ...}` | `skills/skills.json` + `skills/templates/*.md` |
| `skill_write` | `skill, template_content` | `{written, skill_id, version}` | CT gate check; SkillSmith only |

### Agent Lifecycle Tools

| Tool | Params in | Returns | Notes |
|---|---|---|---|
| `ct_close` | `task_id, status` | `{closed, task_id, status}` | Agents: own flag only |
| `todo_write` | `todos: [{content, status, activeForm?}]` | `{written, count}` | Persists to `worker_dir/todos.json`; exactly one `in_progress` allowed |
| `question` | `question, choices?, timeout?` | `{answer, from}` | Writes `question.json`; polls `answer.json`; stdin fallback |
| `send_message` | `channel, message, parse_mode?` | `{sent, channel}` | Telegram / Discord / Slack |
| `cli_output` | `content, type?` | `{printed}` | stdout; Orchestrator only |

---

## Data Files

```
Pikaia/
├── config.json          OrchestratorConfig fields (global defaults + agent loop tuning)
├── models.json          list[{model_id, provider, context_window, enabled, ...}]
├── keys.json            {provider_name: api_key_string | list[str]}
├── pikaia.db            SQLite WAL — trajectories, tool_events, metrics tables
├── tools/tools.json     list[{tool_id, impl, enabled, permissions:[...]}]
├── memory/
│   ├── lt.json          list[{id, content, category, created_at}]
│   ├── mt.json          list[MTEntry]  (+ palace fields: wing, room, importance, aaak_code, embedding)
│   └── kg.json          list[{id, subject, predicate, object, valid_from, valid_to, confidence}]
├── skills/
│   ├── skills.json      list[{skill_id, name, tier, tools_required, template, embedding, active, version}]
│   └── templates/       {skill_id}_v{n}.md  — prompt template files
└── projects/{project}/
    ├── config.json      project-level config overrides
    ├── ct.json          list[CTEntry]  {id, type, description, task_id, status, opened_at, closed_at}
    ├── preferences.json dict  — user preference key/values (overlaid onto LT context)
    ├── file_index.json  {path: {summary, last_indexed}}  — Layer 1 file map
    ├── trajectories/    per-run JSONL replay buffers ({task_id}.jsonl)
    ├── dev/
    │   ├── index.json   {path: {summary, embedding, tags}}  — Layer 2 semantic RAG
    │   └── output/      promoted deliverable files
    ├── instances/{id}/
    │   ├── st.json      {instance_id, project, summary, window:[msgs], updated_at}
    │   └── history.json list[{turn_id, role, content, ts}]
    └── worker/{agent_id}/
        ├── meta.json    AgentRecord fields
        ├── task.json    enriched task packet
        ├── ack.json     {task_id, restatement, planned_steps, confidence, ambiguities}
        ├── state.json   {status, step_current, step_total, tokens_used, issues}
        ├── result.json  {status, output, confidence}
        ├── todos.json   agent todo list (written by todo_write tool, optional)
        ├── question.json pending question to user (optional)
        └── answer.json  user answer (written by orchestrator, optional)
```

---

## Connection Graph

```
main.py
 ├─► Orchestrator.py          creates Orchestrator + OrchestratorConfig
 └─► tools/registry.py        creates ToolRegistry, wraps as Tools

Orchestrator.py
 ├─► tools/registry.py        Tools.dispatch → all tool calls; _unwrap(ToolResult)
 ├─► context_manager.py       lazy: _get_ctx_manager().assess(task_packet)
 └─► agent.py                 lazy: AgentRunner.run() in daemon thread

agent.py
 ├─► tools/registry.py        own ToolRegistry for tool loop dispatch
 ├─► tools/schemas.py         get_schemas(tools_allowed)
 ├─► tools/providers/*.py     _load_adapter() for direct LLM calls (build/call/parse)
 ├─► tools/error_types.py     classify_error(exc) for all LLM/tool failures
 ├─► metrics.py               MetricsCollector per run; flush(db) at end
 ├─► trajectory.py            TrajectoryLogger per run; finalise(outcome, db) at end
 └─► db.py                    get_db() for SQLite persistence

metrics.py
 └─► db.py                    log_metrics_batch, log_tool_event

trajectory.py
 └─► db.py                    log_trajectory

tools/registry.py
 └─► tools/impl/*.py          mod.run(params, context) → _normalise() → ToolResult

tools/impl/llm_call.py
 └─► tools/providers/*.py     importlib.import_module("tools.providers.{name}")

tools/impl/embed_text.py
 └─► tools/providers/ollama.py  (strategy 2) | hash fallback (strategy 3)

tools/impl/memory_read.py
 └─► mt_palace.py             MTReader.read() for layer=mt
                               kg_read() for layer=kg

tools/impl/memory_write.py
 └─► mt_palace.py             MTWriter.write() for layer=mt
                               kg_write() for layer=kg

tools/impl/context_fetch.py
 └─► context_manager.py       ContextManager.fetch() (static method)

tools/impl/skill_write.py
 └─► tools/impl/embed_text.py spec_from_file_location (embed description)

context_manager.py
 └─► tools/impl/embed_text.py spec_from_file_location (embed queries)

mt_palace.py
 └─► tools/impl/embed_text.py spec_from_file_location (embed MT entries)

init.py
 └─► test_tools.py            run_tests(base_path, tool, fast) when --test flag given
```

---

## Caller Permission Matrix

`tools/tools.json` enforces which caller may use which tool.

| Tool | orchestrator | agent | skillsmith |
|---|---|---|---|
| `shell_exec` | ✓ | ✓ | ✓ |
| `code_exec` | ✓ | ✓ | ✓ |
| `file_read` | ✓ | ✓ | ✓ |
| `file_write` | ✓ | ✓ (worker slot only) | ✓ |
| `file_delete` | ✓ | — | — |
| `file_move` | ✓ | — | — |
| `edit` | ✓ | ✓ | ✓ |
| `apply_patch` | ✓ | ✓ | ✓ |
| `grep` | ✓ | ✓ | ✓ |
| `glob` | ✓ | ✓ | ✓ |
| `list` | ✓ | ✓ | ✓ |
| `http_request` | ✓ | ✓ | ✓ |
| `web_fetch` | ✓ | ✓ | ✓ |
| `web_search` | ✓ | ✓ | ✓ |
| `send_message` | ✓ | ✓ | — |
| `cli_output` | ✓ | — | — |
| `llm_call` | ✓ | ✓ | ✓ |
| `embed_text` | ✓ | ✓ | ✓ |
| `memory_read` | ✓ | ✓ | ✓ |
| `memory_write` | ✓ | — | — |
| `context_fetch` | — | ✓ | ✓ |
| `skill_read` | ✓ | ✓ | ✓ |
| `skill_write` | — | — | ✓ |
| `ct_close` | ✓ | ✓ (own only) | — |
| `todo_write` | ✓ | ✓ | — |
| `question` | ✓ | ✓ | — |
