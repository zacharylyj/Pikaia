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
    │
    └── init.py  (first-run scaffolding, key collection, integrity check)
```

---

## Module Detail

---

### `main.py`
**Role:** CLI entry point. REPL loop, `/command` dispatcher, project/instance lifecycle.

| Function | In | Out | Connects to |
|---|---|---|---|
| `main()` | `--project`, `--instance`, `--debug` CLI args | — (starts REPL) | `Orchestrator`, `ToolRegistry`, `OrchestratorConfig` |
| `_make_orchestrator(project, instance_id, debug)` | `str, str, bool` | `(Orchestrator, ToolRegistry)` | `Orchestrator.py`, `tools/registry.py`, `config.json` |
| `_ensure_project(project)` | `str` | — | `projects/{project}/` directory tree |
| `_create_or_resume(project, instance_id)` | `str, str\|None` | `str` instance_id | `projects/{project}/instances/` |
| `_dispatch_command(line, ...)` | `str` command line | `(Session, bool)` should_exit | all `_cmd_*` helpers |
| `_cmd_approve(project)` | `str` | — (interactive prompt) | `ct.json`, `skills/skills.json` |
| `_cmd_status/memory/skills/models/instances/files/check` | `str` project | — (prints) | `*.json` data files |

**Key data types:**
- `Session = tuple[str, str, Orchestrator, ToolRegistry]`
- `_DEBUG_MODE: bool` — module-level flag, read by `_dispatch_command` on project switch

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
  planned_steps?,   # added by _spawn_agent after ack
  restatement?
}
```

---

### `agent.py`
**Role:** Agent execution engine for all 4 tiers. Called in a daemon thread by `Orchestrator._spawn_agent`.

| Class / Function | In | Out | Connects to |
|---|---|---|---|
| `_load_adapter(pipeline, base_path)` | `str, Path` | `(Adapter, model_id, provider_name)` | `models.json`, `keys.json`, `tools/providers/{provider}.py` |
| `BaseAgent.__init__` | `task_packet, record, base_path` | — | `tools/registry.py` (builds own `ToolRegistry`), `_load_adapter` |
| `BaseAgent._tool_loop(system, messages, tools_allowed)` | `str, list[dict], list[str]` | `(str content, int tokens)` | `tools/schemas.py` (get_schemas), `Adapter.build_request/call/parse_response`, `ToolRegistry.dispatch` |
| `BaseAgent._load_skill_template()` | — | `str` template | `ToolRegistry.dispatch("skill_read", ...)` |
| `BaseAgent._write_state(step, total, tokens, ...)` | ints + str | — | `worker/{id}/state.json` |
| `BaseAgent._mark_done(output, confidence)` | `str, float` | — | `worker/{id}/result.json`, `state.json` |
| `Tier12Agent.run()` | — | — | `_tool_loop` once |
| `Tier3Agent.run()` | — | — | `_decompose`, N × `_tool_loop`, `_synthesize` |
| `Tier3Agent._decompose(objective, template)` | `str, str` | `list[str]` steps | `Adapter.build_request/call/parse_response`, `_strip_json_fences` |
| `Tier4Council.run()` | — | — | N parallel `_tool_loop` threads + `_council_synthesis` |
| `AgentRunner.run(task_packet, record, base_path)` | `dict, dict, str` | — | instantiates correct tier class |

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
  ack.json     { task_id, restatement, planned_steps, confidence, ... }
  state.json   { task_id, status, step_current, step_total, tokens_used, issues }
  result.json  { status, output, confidence }
  task.json    (copy of enriched task packet)
  meta.json    (AgentRecord fields)
```

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
**Role:** First-run setup wizard, project scaffolding, integrity checker, key collector.

| Function | In | Out | Connects to |
|---|---|---|---|
| `scaffold(base)` | `Path` | — | Creates `memory/`, `skills/`, `tools/`, data JSON files |
| `scaffold_project(project, base)` | `str, Path` | — | Creates `projects/{project}/` directory tree + default JSONs |
| `collect_keys(base)` | `Path` | — | Interactive prompt → `keys.json` |
| `check(base)` | `Path` | `CheckResult` | Validates all JSON files, key presence, model entries, tool impls |
| `fix(base)` | `Path` | — | Auto-repairs issues found by `check()` |
| `setup_wizard(first_project)` | `str` | — | Calls `scaffold`, `scaffold_project`, `collect_keys` in sequence |
| `main()` | CLI args `--check`, `--fix`, `--project` | — | Entry point for `python init.py` |

---

### `tools/registry.py`
**Role:** Tool loader + dispatcher. Reads `tools/tools.json`, enforces caller permissions, calls `mod.run(params, context)`.

| Method | In | Out | Connects to |
|---|---|---|---|
| `__init__(base_path, project, instance_id, caller, agent_id?, worker_dir?, token_budget?)` | strings | — | `tools/tools.json`, `config.json` |
| `dispatch(name, params)` | `str, dict` | `Any` | `tools/impl/{name}.py` → `mod.run(params, context)` |
| `update_context(**kwargs)` | key=value | — | updates `self.context` dict in place |
| `available_tools()` | — | `list[str]` | — |

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
  "config":       dict,  # merged global + project config.json
}
```

---

### `tools/schemas.py`
**Role:** Anthropic-format JSON schemas for all 17 tools. Used by agents to advertise tools in LLM requests.

| Function | In | Out |
|---|---|---|
| `get_schemas(tool_names)` | `list[str]` | `list[dict]` Anthropic tool schemas |

---

## Provider Adapters (`tools/providers/`)

All implement `BaseAdapter`. Loaded dynamically via `importlib.import_module("tools.providers.{name}")`.

| File | Provider | `build_request` key fields | `call` returns | `parse_response` returns |
|---|---|---|---|---|
| `base.py` | Abstract | — | — | `{content, content_blocks, tokens_in, tokens_out, model_id, provider, stop_reason, tool_calls}` |
| `anthropic.py` | Anthropic API | `model, messages, system, max_tokens, tools` | Anthropic SDK `Message` object | standard response dict |
| `openai.py` | OpenAI API | `model, messages, max_tokens, tools` | OpenAI response JSON | standard response dict |
| `ollama.py` | Local Ollama | `model, messages, stream, options` | Ollama JSON response | standard response dict |
| `debug.py` | Mock (no network) | passthrough `{system, messages}` | `{"_content": str}` | standard response dict (canned) |

`debug.py` detects pipeline from system prompt keywords → returns shaped canned JSON for classification, skillsmith, ack, mt_judge, context_assessment, task_planning, compression, file_indexing, generic.

---

## Tool Implementations (`tools/impl/`)

All expose `run(params: dict, context: dict) -> dict`.

| Tool | Params in | Returns | External / file I/O |
|---|---|---|---|
| `shell_exec` | `cmd, cwd?, timeout?` | `{stdout, stderr, returncode}` | subprocess |
| `code_exec` | `code, language?, timeout?` | `{stdout, stderr, returncode}` | subprocess (python/node) |
| `file_read` | `path` | `{content, path, size_bytes}` | filesystem (sandboxed to base_path) |
| `file_write` | `path, content` | `{written, path}` | filesystem (agent slot only) |
| `file_delete` | `path` | `{deleted, path}` | filesystem |
| `file_move` | `src, dst` | `{moved, src, dst}` | filesystem |
| `http_request` | `method, url, headers?, body?, timeout?` | `{status_code, headers, body, ok}` | HTTP |
| `web_fetch` | `url, max_chars?, timeout?` | `{url, content, truncated}` | HTTP + HTML strip |
| `send_message` | `channel, message, parse_mode?` | `{sent, channel}` | Telegram / Discord / Slack API |
| `llm_call` | `pipeline, system, messages, max_tokens?, temperature?` | `{content, tokens_in, tokens_out, stop_reason}` | `tools/providers/{provider}.py`, `models.json`, `keys.json` |
| `embed_text` | `text` | `{embedding: list[float], dim, model}` | OpenAI embed API → Ollama → hash fallback |
| `memory_read` | `layer, query?, top_k?, project?, instance_id?, ...` | `list[dict]` or `dict` (ST) | `memory/*.json`, `projects/*/ct.json`, `mt_palace.py` |
| `memory_write` | `layer, entry, project?, instance_id?` | `{written, layer}` | `memory/*.json`, `projects/*/ct.json`, `mt_palace.MTWriter` |
| `context_fetch` | `query, top_k?, include_files?, max_chars_per_file?` | `{text, mt_entries, files}` | `context_manager.ContextManager.fetch` |
| `skill_read` | `skill_id` | `{skill_id, name, tier, template, ...}` | `skills/skills.json`, `skills/templates/*.md` |
| `skill_write` | `skill, template_content` | `{written, skill_id, version}` | CT gate check, `skills/skills.json`, `embed_text` — SkillSmith only |
| `ct_close` | `task_id, status` | `{closed, task_id, status}` | `projects/*/ct.json` — agent closes own flag only |
| `cli_output` | `content, type?` | `{printed}` | stdout |

---

## Data Files

```
Pikaia/
├── config.json          OrchestratorConfig fields (global defaults)
├── models.json          list[{model_id, provider, enabled, ...}]
├── keys.json            {provider_name: api_key_string}
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
        └── result.json  {status, output, confidence}
```

---

## Connection Graph

```
main.py
 ├─► Orchestrator.py          creates Orchestrator + OrchestratorConfig
 └─► tools/registry.py        creates ToolRegistry, wraps as Tools

Orchestrator.py
 ├─► tools/registry.py        Tools.dispatch → all tool calls
 ├─► context_manager.py       lazy: _get_ctx_manager().assess(task_packet)
 └─► agent.py                 lazy: AgentRunner.run() in daemon thread

agent.py
 ├─► tools/registry.py        own ToolRegistry for tool loop dispatch
 ├─► tools/schemas.py         get_schemas(tools_allowed)
 └─► tools/providers/*.py     _load_adapter() for direct LLM calls (build/call/parse)

tools/registry.py
 └─► tools/impl/*.py          mod.run(params, context) for each dispatch call

tools/impl/llm_call.py
 └─► tools/providers/*.py     importlib.import_module("tools.providers.{name}")

tools/impl/embed_text.py
 └─► tools/providers/ollama.py  (strategy 2) | hash fallback (strategy 3)

tools/impl/memory_read.py
 └─► mt_palace.py             MTReader.read() for layer=mt with palace_layer
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
```

---

## Caller Permission Matrix

`tools/tools.json` enforces which caller may use which tool.

| Tool | orchestrator | agent | skillsmith |
|---|---|---|---|
| llm_call | ✓ | ✓ | ✓ |
| embed_text | ✓ | ✓ | ✓ |
| memory_read | ✓ | ✓ | ✓ |
| memory_write | ✓ | ✓ | ✓ |
| context_fetch | ✓ | ✓ | — |
| skill_read | ✓ | ✓ | ✓ |
| skill_write | — | — | ✓ |
| ct_close | ✓ | ✓ (own only) | — |
| file_read | ✓ | ✓ | ✓ |
| file_write | ✓ | ✓ (worker slot only) | — |
| shell_exec | ✓ | ✓ | — |
| code_exec | ✓ | ✓ | — |
| http_request | ✓ | ✓ | — |
| web_fetch | ✓ | ✓ | — |
| send_message | ✓ | ✓ | — |
| cli_output | ✓ | ✓ | — |
