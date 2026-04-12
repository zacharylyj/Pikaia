# Pikaia — Architecture Reference

Pikaia is a multi-tier agentic orchestration framework. A single **Orchestrator** receives user messages, builds context from layered memory, picks or synthesises a skill, spawns the right class of agent, monitors execution, and writes results back into memory for future sessions.

---

## Directory Layout

```
Pikaia/
├── Orchestrator.py          # Core orchestration engine
├── main.py                  # CLI entry point
├── init.py                  # Setup wizard, integrity checker & test runner
├── agent.py                 # 4-tier agent execution engine
├── context_manager.py       # Pre-dispatch + on-demand context retrieval
├── mt_palace.py             # MemPalace MT memory engine (wing/room/AAAK/KG)
├── db.py                    # SQLite WAL backend (trajectories, metrics, tool events)
├── metrics.py               # Per-run MetricsCollector (tokens, latency, tool stats)
├── trajectory.py            # Per-run TrajectoryLogger (JSONL replay buffer + SQLite)
├── test_tools.py            # Functional tool test suite (55 tests, no API key needed)
├── config.json              # Global config (models, thresholds, pipelines, budgets)
├── models.json              # Supported LLM registry
├── keys.json                # API keys (not committed)
├── skills/                  # Skill templates (versioned JSON + markdown)
├── memory/                  # Persistent memory files
│   ├── lt.json              #   Long-term memory
│   ├── mt.json              #   Medium-term memory (MemPalace format)
│   └── kg.json              #   Knowledge Graph (temporal triple store)
├── projects/                # Per-project workspaces
│   └── <project>/
│       ├── ct.json          #   Concurrent tasks / open flags
│       ├── dev/index.json   #   File embedding index for RAG
│       ├── trajectories/    #   Per-run JSONL replay buffers
│       └── worker/<id>/
│           ├── ack.json          # Agent acknowledgement
│           ├── state.json        # Checkpoint state
│           ├── task.json         # Full task packet
│           ├── result.json       # Final output
│           ├── todos.json        # Agent todo list (if used)
│           ├── question.json     # Pending user question (if used)
│           └── answer.json       # User answer (written by orchestrator)
└── tools/
    ├── registry.py          # ToolRegistry — dispatch + permission enforcement + ToolResult
    ├── schemas.py           # Tool schemas (built-in + auto-discovery from impl SCHEMA dicts)
    ├── error_types.py       # ErrorType enum + classify_error() for LLM/tool failures
    ├── tools.json           # Tool metadata, impl paths & caller permissions
    ├── providers/           # LLM provider adapters
    │   ├── anthropic.py
    │   ├── openai.py
    │   ├── groq.py
    │   └── ollama.py
    └── impl/                # Tool implementations (26 tools)
```

---

## Orchestration Pipeline

Every user message travels through these stages:

```
User message
    │
    ▼
1. Context build      ← pulls LT + MT + CT + ST + file index
    │
    ▼
2. Intent classify    ← task / question / meta-command / ambiguous
    │
    ▼
3. Clarify?           ← ask back if intent is ambiguous
    │
    ▼
4. Skill pick         ← embed query → cosine vs. skill embeddings
    │  (miss)
    ├──────────────→ SkillSmith: draft → N×(eval+refine) → CT flag for approval
    │
    ▼
5. Context assess     ← ContextManager.assess() pre-enriches task packet
    │
    ▼
6. Dispatch agent     ← write CT flag, create worker dir, ack retry loop
    │
    ▼
7. Agent executes     ← ReAct tool loop (tier-dependent, step-budgeted)
    │
    ▼
8. Post-process       ← ST update, History append, MT judge, LT promote
    │                    metrics flush, trajectory finalise
    ▼
Response to user
```

---

## Agent Tiers

Agents are dispatched at one of four tiers based on task complexity:

| Tier | Class | Strategy | Timeout |
|------|-------|----------|---------|
| **1** Atomic | `Tier12Agent` | Single ReAct tool loop, skill template | 60 s |
| **2** Composite | `Tier12Agent` | Single ReAct tool loop, richer context | 120 s |
| **3** Sub-agent loop | `Tier3Agent` | Decompose → step loops with checkpoints → synthesise | 300 s |
| **4** Council | `Tier4Council` | N parallel specialists + synthesis | 600 s |

### ReAct Tool Loop

Each tier drives one or more **ReAct loops** with these built-in capabilities:

```
System prompt + task
    │
    ▼
LLM (tool-use enabled)          ← model routed: fast_model for simple tasks
    │   ├── text reply  →  done
    │   └── tool calls  →  partition (parallel-safe vs sequential)
    │                        │
    │                        ├── parallel-safe → ThreadPoolExecutor
    │                        └── sequential   → serial dispatch
    │
    ├── inject tool results
    ├── inject loop awareness note (steps remaining, tools used)
    ▼
    loop (up to max_steps, default 15)
    │
    ├── budget exhausted? → inject wrap-up signal → final LLM turn
    ├── context overflow? → compress history → retry
    ├── rate limit?       → rotate API key → retry with backoff
    └── auth error?       → abort immediately
```

The loop is provider-aware:
- **Anthropic** — `content_blocks` list (tool_use / tool_result)
- **OpenAI / Groq** — `tool_calls` + `role:"tool"` messages
- **Ollama** — text injection fallback (no native tool-use)

---

## Agent Capabilities (New)

### Step Budget
Every `_tool_loop` run is capped at `config.max_steps` (default 15). On exhaustion, a `[tool budget exhausted]` signal is injected and the agent is given one final LLM turn to return its best answer gracefully. Override per task via `task_packet["max_steps"]`.

### Error Classification
All LLM and tool call failures pass through `tools/error_types.py`:

| ErrorType | Recovery |
|-----------|----------|
| `RATE_LIMIT` | Retry with exponential backoff + API key rotation |
| `AUTH` | Abort immediately (bad key, not retryable) |
| `CONTEXT_OVERFLOW` | Compress conversation history → retry |
| `NETWORK` | Retry with exponential backoff |
| `UNKNOWN` | Log and break the loop |

### In-flight Context Compression
When accumulated tokens exceed `config.context_compression_threshold` (default 80%) of the model's context window, older messages are summarised via the `compression` pipeline and replaced. The tail of the conversation is preserved verbatim.

### Parallel Tool Execution
Independent tool calls (reads) execute concurrently via `ThreadPoolExecutor`. Write/compute calls always run sequentially. Controlled by `config.parallel_tool_max_workers` (default 4) and `config.tool_dependency_detection`.

### Model Routing
Short/simple tasks (≤50 words, ≤1 tool) are automatically routed to `config.fast_model` (`claude-haiku` by default) to cut cost. Override per task via `task_packet["fast_model"]`.

### API Key Rotation
`keys.json` may list multiple keys per provider as an array. A `_KeyPool` round-robins keys on 429/auth failures with per-key cooldown tracking.

---

## Memory Architecture

Pikaia uses **five memory layers** that serve different timescales and purposes.

```
┌─────────────────────────────────────────────────────────────────┐
│  LT  Long-Term      memory/lt.json         Permanent facts      │
│  MT  Medium-Term    memory/mt.json         Session knowledge     │
│  CT  Concurrent     projects/<p>/ct.json   Open task flags       │
│  ST  Short-Term     instances/<id>/st.json Active conversation   │
│  KG  Knowledge Graph memory/kg.json        Temporal triple store │
└─────────────────────────────────────────────────────────────────┘
```

### MT — Medium-Term Memory (MemPalace)

Every MT entry is enriched by `mt_palace.py` before storage:

```
Plain text entry
    │
    ├── RoomDetector    → tags entry with wing + room
    ├── EntityExtractor → extracts persons, projects, code identifiers
    ├── AAAKCodec       → lossy compression (~30× smaller, human-readable)
    ├── ImportanceScorer → 0–1 importance score
    └── KnowledgeGraph  → registers entities as temporal triples
```

**Wing / Room hierarchy:**

```
Wings (domains)      Rooms (subtopics)
───────────────      ──────────────────────────────────────────
technical            auth, api, code, data, deploy, testing
decisions            decisions, planning
knowledge            knowledge
issues               issues
```

**AAAK compression format:**

```
ENTITIES|topic_keywords|"key sentence here"|emotions|FLAGS
```

Example:
```
AuthService,JWT|token refresh expiry|"Decided to use sliding expiry with 15-min access tokens"|concern|DECISION
```

**4-layer MT retrieval:**

| Layer | Scope | When used |
|-------|-------|-----------|
| **L1** | Highest-importance entries, spread across rooms | Always (essential story) |
| **L2** | Wing/room-filtered + cosine search | On-demand |
| **L3** | Full semantic cosine search across all MT | Deep research |

---

## Tools

26 tools routed through `ToolRegistry` with per-caller permissions. All tools return a normalised `ToolResult` envelope `{success, data, error}`.

### File & Code Tools

| Tool | Who can call | Purpose |
|------|-------------|---------|
| `file_read` | all | Read files with optional `offset`/`limit` for large files |
| `file_write` | all | Write files atomically (agent: worker slot only) |
| `edit` | all | Exact-string replacement — enforces uniqueness, `replace_all` opt-in |
| `file_delete` | orchestrator | Delete files or empty directories |
| `file_move` | orchestrator | Move / promote worker outputs |
| `apply_patch` | all | Apply unified diff via system `patch`; Python fallback |
| `shell_exec` | all | Run shell commands in subprocess |
| `code_exec` | all | Run Python/JS in isolated sandbox |

### Search & Navigation Tools

| Tool | Who can call | Purpose |
|------|-------------|---------|
| `grep` | all | Regex search — rg when available, Python fallback; `files_with_matches`/`content`/`count` modes |
| `glob` | all | Pattern file finder sorted by mtime; rg when available |
| `list` | all | Directory listing; `recursive` mode for full tree |

### Web & HTTP Tools

| Tool | Who can call | Purpose |
|------|-------------|---------|
| `web_fetch` | all | Fetch URL as clean stripped text |
| `web_search` | all | DuckDuckGo search — returns `{title, url, snippet}`; no API key needed |
| `http_request` | all | Generic REST calls |

### Memory & Context Tools

| Tool | Who can call | Purpose |
|------|-------------|---------|
| `memory_read` | all | Read LT/MT/CT/ST/KG layers |
| `memory_write` | orchestrator | Write to memory layers |
| `context_fetch` | agent, skillsmith | On-demand plain-English context retrieval |

### LLM & Skill Tools

| Tool | Who can call | Purpose |
|------|-------------|---------|
| `llm_call` | all | Pipeline-routed LLM calls |
| `embed_text` | all | Generate embedding vectors |
| `skill_read` | all | Fetch skill template by ID |
| `skill_write` | skillsmith | Write new skill version (post human gate) |

### Agent Lifecycle Tools

| Tool | Who can call | Purpose |
|------|-------------|---------|
| `ct_close` | orchestrator, agent | Close CT flag by task_id |
| `todo_write` | orchestrator, agent | Manage agent todo list (persisted to worker dir) |
| `question` | orchestrator, agent | Ask user a question mid-task; polls `answer.json`; stdin fallback |
| `send_message` | orchestrator, agent | Telegram / Discord / Slack |
| `cli_output` | orchestrator | Print to terminal |

### Self-registering Schemas

Tools can declare their own schema by adding a module-level `SCHEMA` dict:

```python
# tools/impl/my_tool.py
SCHEMA = {
    "name": "my_tool",
    "description": "...",
    "input_schema": { ... }
}
```

`tools/schemas.py` auto-discovers these at startup. No central file edit needed when adding new tools.

---

## Observability

### Metrics (`metrics.py`)
`MetricsCollector` accumulates per-run: `tokens_in`, `tokens_out`, `steps`, `latency_ms`, per-tool success rates. Flushed to SQLite at run end.

### Trajectory Logging (`trajectory.py`)
`TrajectoryLogger` writes a step-by-step replay buffer:
- **JSONL** at `projects/<project>/trajectories/<task_id>.jsonl` — one JSON object per step
- **SQLite** row in `pikaia.db` for structured queries

Step types: `llm_turn`, `tool_call`, `tool_result`, `compress`.

### SQLite Backend (`db.py`)
WAL-mode SQLite at `pikaia.db` with three tables:

| Table | Content |
|-------|---------|
| `trajectories` | One row per agent run (task, outcome, full steps as JSON) |
| `tool_events` | One row per tool dispatch (name, success, latency_ms) |
| `metrics` | One row per metric observation (name, value, task_id) |

Thread-safe via `get_db(path)` singleton + lock.

---

## Supported Models

| Model | Provider | Context | Speed | Use in |
|-------|----------|---------|-------|--------|
| `claude-sonnet-4-6` | Anthropic | 200k | medium | orchestration, code, default |
| `claude-opus-4-6` | Anthropic | 200k | slow | research, council |
| `claude-haiku-4-5-20251001` | Anthropic | 200k | fast | classification, compression, fast_model |
| `gpt-4o` | OpenAI | 128k | medium | general, multimodal |
| `gpt-4o-mini` | OpenAI | 128k | fast | simple tasks |
| `llama3.2` | Ollama | 128k | local | private, no-cost |

Pipeline → model mapping is in `config.json` under `"pipelines"`.

---

## Configuration Reference (`config.json`)

### Core

| Key | Default | Description |
|-----|---------|-------------|
| `default_model` | `claude-sonnet-4-6` | Fallback model |
| `compression_model` | `claude-haiku-4-5-20251001` | ST compression model |
| `skill_match_threshold` | `0.75` | Minimum cosine score to use existing skill |
| `promote_threshold` | `0.80` | Score to promote output to MT |
| `ack_confidence_min` | `0.80` | Minimum agent ack confidence |
| `retry_limit` | `3` | Agent retry attempts |

### Agent Loop

| Key | Default | Description |
|-----|---------|-------------|
| `max_steps` | `15` | Hard cap on tool-loop iterations per run |
| `context_compression_threshold` | `0.80` | Fraction of context window before compression |
| `parallel_tool_max_workers` | `4` | Max threads for parallel tool execution |
| `error_retry_max` | `3` | Max retries on rate-limit/network errors |
| `error_retry_base_delay` | `1.0` | Base backoff in seconds (doubles each retry) |
| `fast_model` | `claude-haiku-4-5-20251001` | Model for simple tasks (set to `""` to disable) |
| `fast_model_threshold_words` | `50` | Route to fast_model if prompt ≤ N words |
| `fast_model_threshold_tools` | `1` | Route to fast_model if tools ≤ N |
| `loop_awareness_injection` | `true` | Inject step budget + tool history each turn |
| `tool_dependency_detection` | `true` | Classify calls as parallel-safe or sequential |
| `key_rotation_enabled` | `true` | Rotate API keys on 429/auth failures |

### Observability

| Key | Default | Description |
|-----|---------|-------------|
| `trajectory_logging` | `true` | Write per-run JSONL + SQLite trajectory |
| `metrics_enabled` | `true` | Collect tokens/latency/tool stats |

---

## SkillSmith

When no skill matches a task (cosine score < `skill_match_threshold`), SkillSmith is triggered:

```
Draft skill (sonnet)
    │
    └── N × (eval → refine)    ← up to skillsmith_dry_runs cycles
         │
         ▼
    pass_score reached?
         ├── yes → write to skills/, set CT approval flag
         └── no  → mark as low-confidence, still write (human reviews CT)
```

---

## Integrity Checker

```bash
python init.py --check     # run all 11 checks
python init.py --fix       # auto-repair recoverable issues
python init.py --test      # run 55 functional tool tests
python init.py --test --fast   # skip network/slow tests (50 tests)
python init.py --test --tool grep  # test a single tool
```

**`--check` covers:**
1. Directory structure
2. JSON file validity
3. Tool impl files + required fields
4. Tool schema coverage
5. Pipeline model coverage + fast_model
6. Core module file integrity
7. Config key completeness
8. Skill embeddings
9. Stale CT flags (>24h open)
10. File index coverage
11. Observability paths (trajectories/, pikaia.db)

---

## Agent Worker Directory

```
projects/<project>/worker/<agent_id>/
├── meta.json       AgentRecord fields
├── task.json       full enriched task packet
├── ack.json        {task_id, restatement, planned_steps, confidence}
├── state.json      {status, step_current, step_total, tokens_used, issues}
├── result.json     {status, output, confidence}
├── todos.json      agent todo list (written by todo_write tool)
├── question.json   pending question to user (written by question tool)
└── answer.json     user response (written by orchestrator)
```
