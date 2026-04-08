# Pikaia — Architecture Reference

Pikaia is a multi-tier agentic orchestration framework. A single **Orchestrator** receives user messages, builds context from layered memory, picks or synthesises a skill, spawns the right class of agent, monitors execution, and writes results back into memory for future sessions.

---

## Directory Layout

```
Pikaia/
├── Orchestrator.py          # Core orchestration engine
├── main.py                  # CLI entry point
├── init.py                  # Setup wizard & integrity checker
├── agent.py                 # 4-tier agent execution engine
├── context_manager.py       # Pre-dispatch + on-demand context retrieval
├── mt_palace.py             # MemPalace MT memory engine (wing/room/AAAK/KG)
├── config.json              # Global config (models, thresholds, pipelines)
├── models.json              # Supported LLM registry
├── keys.json                # API keys (not committed)
├── skills/                  # Skill templates (versioned JSON)
├── memory/                  # Persistent memory files (gitignored at runtime)
│   ├── lt.json              #   Long-term memory
│   ├── mt.json              #   Medium-term memory (MemPalace format)
│   └── kg.json              #   Knowledge Graph (temporal triple store)
├── projects/                # Per-project workspaces (gitignored at runtime)
│   └── <project>/
│       ├── ct.json          #   Concurrent tasks / open flags
│       ├── dev/index.json   #   File embedding index for RAG
│       └── instances/
│           └── <id>/
│               ├── ack.json       # Agent acknowledgement
│               ├── state.json     # Checkpoint state
│               ├── task.json      # Full task packet
│               └── result.json    # Final output
└── tools/
    ├── registry.py          # ToolRegistry — dispatch + permission enforcement
    ├── schemas.py           # Anthropic tool_use input schemas
    ├── tools.json           # Tool metadata & implementation paths
    ├── providers/           # LLM provider adapters
    │   ├── anthropic.py
    │   ├── openai.py
    │   └── ollama.py
    └── impl/                # Tool implementations (18 tools)
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
7. Agent executes     ← ReAct tool loop (tier-dependent)
    │
    ▼
8. Post-process       ← ST update, History append, MT judge, LT promote
    │
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

Each tier ultimately drives one or more **ReAct loops**:

```
System prompt + task
    │
    ▼
LLM (tool-use enabled)
    │   ├── text reply  →  done
    │   └── tool call   →  execute → inject result → loop
    ▼
Result
```

The loop is provider-aware:
- **Anthropic** — `content_blocks` list (tool_use / tool_result)
- **OpenAI** — `tool_calls` + `role:"tool"` messages
- **Ollama** — Text injection fallback (no native tool-use)

Maximum 20 tool turns per loop before force-stop.

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

### LT — Long-Term Memory

Append-only. Written by the Orchestrator after a task is completed and judged significant. Also contains **user preferences** (loaded from a `preferences.json` overlay at startup and injected as synthetic LT entries into every context build).

```json
{ "id": "lt-001", "content": "User prefers TypeScript over JavaScript.", "created_at": "..." }
```

### MT — Medium-Term Memory (MemPalace)

The richest layer. Every MT entry is enriched by `mt_palace.py` before storage:

```
Plain text entry
    │
    ├── RoomDetector   → tags entry with wing + room
    ├── EntityExtractor → extracts persons, projects, code identifiers
    ├── AAAKCodec      → lossy compression (~30× smaller, human-readable)
    ├── ImportanceScorer → 0–1 importance score
    └── KnowledgeGraph → registers entities as temporal triples
```

**Wing / Room hierarchy:**

```
Wings (domains)          Rooms (subtopics)
─────────────────        ─────────────────────────────────────────
technical                auth, api, code, data, deploy, testing
decisions                decisions, planning
knowledge                knowledge
issues                   issues
```

**AAAK compression format:**

```
ENTITIES|topic_keywords|"key sentence here"|emotions|FLAGS
```

Example:
```
AuthService,JWT|token refresh expiry|"Decided to use sliding expiry with 15-min access tokens"|concern|DECISION
```

No decoder needed — the format is human-readable and injectable directly into prompts.

**4-layer MT retrieval:**

| Layer | Scope | Token budget | When used |
|-------|-------|-------------|-----------|
| **L1** | Highest-importance entries, spread across rooms | 500–800 | Always (essential story) |
| **L2** | Wing/room-filtered + cosine search | 200–500 | On-demand |
| **L3** | Full semantic cosine search across all MT | Unrestricted | Deep research |

### CT — Concurrent Tasks

Open-flag tracker at the project level. Each agent spawn writes a CT entry; the agent closes it via `ct_close`. Also used for skill approval requests from SkillSmith.

```json
{ "id": "ct-abc", "type": "pending", "status": "open", "task_id": "...", "agent_id": "..." }
```

### ST — Short-Term Memory

Full conversation transcript for the active agent instance. Stored under `instances/<id>/st.json`. Compressed by haiku when it exceeds `st_max_messages` (default: 20). Discarded after the instance closes.

### KG — Knowledge Graph

Temporal triple store (`memory/kg.json`). Written when entities are extracted during MT enrichment.

```json
{ "subject": "AuthService", "predicate": "uses", "object": "JWT",
  "valid_from": "2025-01-01", "valid_to": null, "confidence": 0.9 }
```

Supports `invalidate()` (set `valid_to`), `merge_entities()`, and `timeline(subject)`.

---

## Context System

### Pre-dispatch Assessment (`ContextManager.assess`)

Before any agent is spawned, the Orchestrator calls:

```python
task_packet = self._get_ctx_manager().assess(task_packet, project)
```

This:
1. Checks if existing MT hits are weak (score < 0.55)
2. Calls haiku (cheap) to identify genuine gaps: `{"sufficient": bool, "gaps": [...], "queries": [...]}`
3. For each gap query, fetches additional MT entries + file snippets via cosine search
4. Merges new items into `task_packet["context"]` (deduplicated)
5. Tags the packet: `{pre_enriched, sufficient, gaps, enriched_with}`

Agents start with the richest possible context without doing any memory management themselves.

### On-demand Fetch (`context_fetch` tool)

During execution, any agent can call:

```
context_fetch(query="how does the auth refresh flow work?")
```

Returns a pre-formatted text block:

```
## Context retrieved for: how does the auth refresh flow work?

### Knowledge
- (score 0.87) AuthService uses sliding JWT with 15-min access tokens

### Relevant files
**src/auth/refresh.ts** (score 0.82)
Summary: Handles token refresh and sliding expiry logic
```

Agents ask questions in plain English and receive ready-to-use context. They never interact with memory layers, embedding vectors, or cosine arithmetic directly.

### Context Build at Dispatch

`_build_context()` assembles the initial task packet context from:

| Source | What it provides |
|--------|-----------------|
| LT memory | User preferences + permanent facts |
| MT memory | Top-K relevant entries (L1 layer) |
| CT open flags | Pending tasks, approvals |
| ST summary | Compressed conversation so far |
| File index (`dev/index.json`) | Top-K relevant file summaries |

---

## Tools

18 tools are available, routed through `ToolRegistry` with per-caller permissions.

| Tool | Who can call | Purpose |
|------|-------------|---------|
| `shell_exec` | all | Run shell commands |
| `code_exec` | all | Run Python/JS in sandbox |
| `file_read` | all | Read files (scope-enforced) |
| `file_write` | all | Write files atomically |
| `file_delete` | orchestrator | Delete files/dirs |
| `file_move` | orchestrator | Move/promote worker outputs |
| `http_request` | all | Generic REST calls |
| `web_fetch` | all | Fetch URL as clean text |
| `cli_output` | orchestrator | Print to terminal |
| `send_message` | orchestrator, agent | Telegram / Discord / Slack |
| `llm_call` | all | Pipeline-routed LLM calls |
| `embed_text` | all | Generate embedding vectors |
| `memory_read` | all | Read LT/MT/CT/ST/KG layers |
| `memory_write` | orchestrator | Write to memory layers |
| `skill_read` | all | Fetch skill template |
| `skill_write` | skillsmith | Write new skill version |
| `ct_close` | orchestrator, agent | Close CT flag by task_id |
| `context_fetch` | agent, skillsmith | On-demand context retrieval |

---

## Supported Models

| Model | Provider | Context | Speed | Use in |
|-------|----------|---------|-------|--------|
| `claude-sonnet-4-6` | Anthropic | 200k | medium | orchestration, code, default |
| `claude-opus-4-6` | Anthropic | 200k | slow | research, council |
| `claude-haiku-4-5-20251001` | Anthropic | 200k | fast | classification, compression, assessment |
| `gpt-4o` | OpenAI | 128k | medium | general, multimodal |
| `gpt-4o-mini` | OpenAI | 128k | fast | simple tasks |
| `llama3.2` | Ollama | 128k | local | private, no-cost |

Pipeline → model mapping is configured in `config.json` under `"pipelines"`.

---

## Configuration Reference (`config.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `default_model` | `claude-sonnet-4-6` | Fallback model when pipeline has no override |
| `compression_model` | `claude-haiku-4-5-20251001` | Model used for ST compression |
| `skill_match_threshold` | `0.75` | Minimum cosine score to use an existing skill |
| `promote_threshold` | `0.80` | Score threshold for promoting output to MT |
| `ack_confidence_min` | `0.80` | Minimum confidence in agent acknowledgement |
| `ack_max_rounds` | `2` | Max ack retry attempts before failing |
| `st_max_messages` | `20` | ST message count before compression |
| `mt_top_k` | `5` | MT entries returned per context build |
| `history_rag_top_k` | `3` | History snippets returned per query |
| `file_summary_top_k` | `3` | File summaries returned per context build |
| `max_files_per_task` | `5` | Max files read during file index RAG |
| `retry_limit` | `3` | Agent retry attempts on failure |
| `skillsmith_dry_runs` | `3` | Draft-eval-refine cycles before approval |
| `skillsmith_pass_score` | `0.80` | Minimum eval score to approve a new skill |
| `embedding_dim` | `1536` | Expected embedding vector dimension |

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

Skills are stored as versioned JSON templates under `Pikaia/skills/`.

---

## Agent Worker Directory

Each spawned agent gets an isolated directory:

```
projects/<project>/instances/<uuid>/
├── task.json      ← full enriched task packet (written before spawn)
├── ack.json       ← agent acknowledgement (planned steps, confidence)
├── state.json     ← checkpoint state (updated during execution)
└── result.json    ← final output (written on completion)
```

The Orchestrator polls `state.json` to monitor progress and enforces tier budgets and timeouts.
