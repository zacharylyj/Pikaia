# Getting Started with Pikaia

This guide covers installation, API key setup, first run, and basic configuration.

---

## Requirements

- Python 3.10 or higher
- At least one of:
  - **Anthropic API key** (recommended — used by default pipelines)
  - **OpenAI API key**
  - **Groq API key** (free tier available)
  - **Ollama** running locally (free, no key required)
- Optional: `ripgrep` (`rg`) — used by `grep`, `glob`, and `list` tools for faster search; pure-Python fallback is available

---

## 1. Clone and Install

```bash
git clone https://github.com/zacharylyj/Pikaia.git
cd Pikaia
pip install anthropic openai requests          # minimum dependencies
```

Optional dependencies:
```bash
pip install groq                               # Groq provider support
# Install Ollama from https://ollama.com, then pull a model:
ollama pull llama3.2
# Install ripgrep for faster file search (grep/glob/list tools):
# Windows: winget install BurntSushi.ripgrep.MSVC
# macOS:   brew install ripgrep
```

---

## 2. Add API Keys

Open `Pikaia/keys.json` and fill in whichever providers you have:

```json
{
  "anthropic": "sk-ant-...",
  "openai":    "sk-...",
  "groq":      "gsk_...",
  "ollama":    ""
}
```

You only need the key(s) for the providers your configured pipelines use. The default `config.json` uses Anthropic models, so only `anthropic` is required out of the box.

### Multiple keys per provider (rotation)

You can supply a list of keys for any provider. On rate-limit (429) or auth errors, Pikaia automatically rotates to the next key with per-key cooldown tracking:

```json
{
  "anthropic": ["sk-ant-key1...", "sk-ant-key2...", "sk-ant-key3..."],
  "openai":    "sk-..."
}
```

Key rotation is enabled by default. Disable with `"key_rotation_enabled": false` in `config.json`.

---

## 3. Run the Setup Wizard

```bash
cd Pikaia
python init.py
```

The wizard will:
- Validate `config.json`, `models.json`, `keys.json`, and `tools/tools.json`
- Create the required directory structure (`memory/`, `projects/`, `skills/`, `trajectories/`)
- Initialise empty memory files (`lt.json`, `mt.json`, `kg.json`)
- Scaffold a default project workspace
- Report any missing or misconfigured items

### Validation-only mode (no writes)

```bash
python init.py --check
```

Runs 11 integrity checks:
1. Directory structure
2. JSON file validity
3. Tool impl files + required fields
4. Tool schema coverage
5. Pipeline model coverage + fast_model
6. Core module integrity (agent.py, db.py, metrics.py, trajectory.py, etc.)
7. Config key completeness (all required keys present)
8. Skill embeddings
9. Stale CT flags (>24h open)
10. File index coverage
11. Observability paths (trajectories/, pikaia.db writable)

### Auto-fix recoverable issues

```bash
python init.py --fix
```

### Run the functional tool test suite

```bash
python init.py --test              # run all 55 tests
python init.py --test --fast       # skip network/slow tests (50 tests)
python init.py --test --tool grep  # test a single tool
```

Tests run without an API key. They cover all 26 tools across file I/O, search, web, memory, shell execution, code execution, patching, and more.

### Scaffold a named project

```bash
python init.py --project myproject
```

---

## 4. Start the CLI

```bash
python main.py
```

With options:

```bash
python main.py --project myproject     # explicit project workspace
python main.py --instance <id>         # resume an existing session
python main.py --groq                  # use Groq free-tier API
python main.py --ollama                # use local Ollama models
python main.py --debug                 # mock LLM calls (no API key needed)
```

You will see the Pikaia banner and a `>` prompt. Type any task or question.

---

## 5. First Interactions

### Ask a question

```
> What is the capital of France?
```

A Tier 1 agent handles this directly — fast, single tool loop, no decomposition. Short/simple tasks (≤50 words, ≤1 tool) are automatically routed to the `fast_model` (`claude-haiku` by default) to reduce cost.

### Give a task

```
> Write a Python function that calculates compound interest and save it to finance.py
```

The Orchestrator will:
1. Classify as `task`
2. Match or create a skill
3. Run `ContextManager.assess()` to check available context
4. Spawn a Tier 1 or 2 agent with the enriched task packet
5. The agent uses `file_write` to save the result
6. Output is promoted to MT memory for future sessions

### Complex multi-step task

```
> Refactor the authentication module: extract the token refresh logic into its own service,
  add unit tests, and update the API docs.
```

This will tier-up to a **Tier 3 sub-agent loop**:
1. Task is decomposed into steps
2. Each step runs its own tool loop with checkpoints
3. Results are synthesised into a final output

---

## 6. Provider Modes

### Default (Anthropic)

```bash
python main.py
```

Uses the pipelines defined in `config.json`. Default is `claude-sonnet-4-6` for orchestration.

### Groq (free tier)

```bash
python main.py --groq
```

Requires `"groq": "gsk_..."` in `keys.json`. All pipelines route to Groq's `llama-3.1-70b-versatile`.

### Ollama (local, no cost)

```bash
python main.py --ollama
```

Requires Ollama running (`ollama serve`) with a model pulled (`ollama pull llama3.2`). Routes all pipelines to the local model. Tool-use falls back to text injection mode.

### Debug mode (no API key needed)

```bash
python main.py --debug
```

LLM calls return canned responses shaped to each pipeline's expected format. Useful for testing tool logic and orchestration flow without any API credentials.

---

## 7. Switching Models

Edit `Pikaia/config.json` to change which model handles each pipeline:

```json
{
  "pipelines": {
    "orchestration":   "claude-sonnet-4-6",
    "code_generation": "gpt-4o",
    "research":        "claude-opus-4-6",
    "compression":     "claude-haiku-4-5-20251001"
  }
}
```

To use Ollama for everything (fully local, no API cost):

```json
{
  "default_model": "llama3.2",
  "pipelines": {
    "orchestration":      "llama3.2",
    "task_planning":      "llama3.2",
    "code_generation":    "llama3.2",
    "compression":        "llama3.2",
    "classification":     "llama3.2",
    "context_assessment": "llama3.2"
  }
}
```

> **Note:** Ollama models do not support native tool-use. The ReAct loop falls back to text injection mode, which is less reliable for complex tool chains.

---

## 8. Agent Loop Configuration

These `config.json` keys control the agent's ReAct tool loop:

| Key | Default | Description |
|-----|---------|-------------|
| `max_steps` | `15` | Hard cap on tool-loop iterations per run |
| `context_compression_threshold` | `0.80` | Fraction of context window before compression |
| `parallel_tool_max_workers` | `4` | Max threads for parallel tool execution |
| `error_retry_max` | `3` | Max retries on rate-limit/network errors |
| `error_retry_base_delay` | `1.0` | Base backoff in seconds (doubles each retry) |
| `fast_model` | `claude-haiku-4-5-20251001` | Model for simple tasks |
| `fast_model_threshold_words` | `50` | Route to fast_model if prompt ≤ N words |
| `fast_model_threshold_tools` | `1` | Route to fast_model if tools ≤ N |
| `loop_awareness_injection` | `true` | Inject step budget + tool history each turn |
| `tool_dependency_detection` | `true` | Classify calls as parallel-safe or sequential |
| `key_rotation_enabled` | `true` | Rotate API keys on 429/auth failures |

Set `fast_model` to `""` to disable fast-model routing entirely.

---

## 9. User Preferences

Create `Pikaia/preferences.json` to inject persistent preferences into every context build:

```json
{
  "language":     "Python",
  "style":        "Prefer type hints and docstrings on all functions.",
  "output":       "Always include example usage in generated code.",
  "verbosity":    "Be concise in explanations.",
  "project_note": "This is a FastAPI backend. Use async/await throughout."
}
```

Each key-value pair becomes a synthetic LT memory entry. Agents see these preferences before they see the task.

---

## 10. Project Workspaces

A **project** is an isolated workspace with its own:
- `ct.json` — open task flags
- `dev/index.json` — file embedding index for RAG (auto-built when files are indexed)
- `trajectories/` — per-run JSONL replay buffers
- `worker/` — per-agent run directories

Scaffold additional projects:

```bash
python init.py --project backend
python init.py --project frontend
```

Run Pikaia against a specific project:

```bash
python main.py --project backend
```

---

## 11. Memory Inspection

Memory files are plain JSON and can be inspected at any time:

```bash
# What has been remembered long-term?
cat Pikaia/memory/lt.json

# What is in medium-term memory (MemPalace format)?
cat Pikaia/memory/mt.json

# What entities and relationships have been extracted?
cat Pikaia/memory/kg.json

# What tasks are currently open for a project?
cat Pikaia/projects/default/ct.json
```

To query memory from a running session, agents and tools use `memory_read`:

```python
# Inside a tool call or agent prompt:
memory_read(layer="mt", query="authentication token refresh", palace_layer=2)
memory_read(layer="kg", subject="AuthService")
```

---

## 12. Observability

### Metrics

Per-run token usage, latency, and tool success rates are collected automatically and flushed to `pikaia.db` (SQLite) at run end. Enabled by default; disable with `"metrics_enabled": false` in `config.json`.

### Trajectory Logging

Every agent run produces a step-by-step replay buffer:
- **JSONL** at `projects/<project>/trajectories/<task_id>.jsonl` — one JSON object per step
- **SQLite** row in `pikaia.db` for structured queries

Step types: `llm_turn`, `tool_call`, `tool_result`, `compress`.

Enabled by default; disable with `"trajectory_logging": false` in `config.json`.

### SQLite Backend

`pikaia.db` (WAL mode) in the Pikaia root:

| Table | Content |
|-------|---------|
| `trajectories` | One row per agent run (task, outcome, full steps as JSON) |
| `tool_events` | One row per tool dispatch (name, success, latency_ms) |
| `metrics` | One row per metric observation (name, value, task_id) |

---

## 13. File Indexing for RAG

Place source files in your project's worker directory and the Orchestrator will index them automatically during `_reindex_file()` calls. The index is stored in:

```
projects/<project>/dev/index.json
```

Each entry contains a summary and embedding vector. `context_fetch` and `ContextManager.assess()` both search this index when building context for agents.

---

## 14. SkillSmith (Automatic Skill Creation)

If no skill matches a task, SkillSmith automatically drafts one:

```
No skill match found (score 0.62 < threshold 0.75)
  → SkillSmith: drafting new skill...
  → Dry run 1/3: eval score 0.71
  → Dry run 2/3: eval score 0.84 — PASS
  → Skill written: skills/data_pipeline_builder_v1.json
  → CT approval flag created (review in ct.json)
```

Review pending skill approvals:

```bash
cat Pikaia/projects/default/ct.json | python -m json.tool | grep -A5 "skill_approval"
```

Approve by removing the CT flag or setting `"status": "done"`.

---

## 15. Tools Overview

26 tools are available to agents, routed through `ToolRegistry` with per-caller permissions. All tools return a normalised `{success, data, error}` envelope.

| Category | Tools |
|----------|-------|
| File & Code | `file_read` (offset/limit), `file_write`, `edit`, `file_delete`, `file_move`, `apply_patch`, `shell_exec`, `code_exec` |
| Search | `grep` (rg/Python fallback), `glob` (rg/Python fallback), `list` |
| Web & HTTP | `web_fetch`, `web_search` (DuckDuckGo, no key), `http_request` |
| Memory | `memory_read`, `memory_write`, `context_fetch` |
| LLM & Skills | `llm_call`, `embed_text`, `skill_read`, `skill_write` |
| Agent Lifecycle | `ct_close`, `todo_write`, `question`, `send_message`, `cli_output` |

---

## 16. Troubleshooting

### `KeyError: 'anthropic'` during LLM call
Your `keys.json` is missing the `anthropic` key. Add it or switch pipelines to a different provider.

### Agent always fails ack validation
Lower `ack_confidence_min` in `config.json` (try `0.65`) or increase `ack_max_rounds` to `3`.

### Skill match always misses
Lower `skill_match_threshold` (try `0.65`) or run `init.py --check` to verify embeddings are being generated (requires a working `embed_text` tool and API key).

### Ollama not found
Ensure Ollama is running: `ollama serve`. Verify the model is pulled: `ollama list`.

### `UnicodeEncodeError` on Windows terminal
Run with UTF-8 encoding: `set PYTHONIOENCODING=utf-8` before `python main.py`.

### Tool tests failing
Run `python init.py --test --tool <name>` to isolate the failing tool. Use `--fast` to skip network-dependent tests.

### Rate limits
Add multiple API keys as a JSON array in `keys.json` (see Section 2). The `_KeyPool` rotates keys automatically with per-key cooldown.

---

## 17. Configuration Quick Reference

| File | Purpose |
|------|---------|
| `Pikaia/config.json` | Models, thresholds, pipeline assignments, agent loop tuning |
| `Pikaia/models.json` | Registered LLM providers and capabilities |
| `Pikaia/keys.json` | API keys — single string or list per provider (never commit this) |
| `Pikaia/preferences.json` | User preferences injected into every context |
| `Pikaia/skills/` | Versioned skill templates |
| `Pikaia/memory/lt.json` | Long-term memory (persistent across sessions) |
| `Pikaia/memory/mt.json` | Medium-term memory (MemPalace format) |
| `Pikaia/memory/kg.json` | Knowledge graph (entity relationships) |
| `Pikaia/pikaia.db` | SQLite observability store (trajectories, metrics, tool events) |

---

## 18. Architecture in One Paragraph

When you send a message, the Orchestrator builds context from five memory layers (LT facts, MT knowledge with MemPalace wing/room tagging, open CT flags, the current ST conversation, and a file embedding index), classifies your intent, picks or creates a skill, runs `ContextManager.assess()` to pre-enrich the task packet, then spawns the appropriate agent tier (1–4). Agents execute a ReAct tool loop — calling shell, file, HTTP, memory, and LLM tools as needed — with a hard step budget, automatic context compression, parallel tool execution for independent reads, model routing to fast-model for simple tasks, and exponential-backoff retry with API key rotation on failures. Each run is fully logged to a JSONL trajectory buffer and SQLite metrics store. When the agent finishes, its output is written back into MT memory (enriched with entity extraction, AAAK compression, and KG triple updates) and may be promoted to LT. The next session starts richer.

For a full architecture breakdown, see [README.md](README.md).
