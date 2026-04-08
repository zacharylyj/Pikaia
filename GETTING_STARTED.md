# Getting Started with Pikaia

This guide covers installation, API key setup, first run, and basic configuration.

---

## Requirements

- Python 3.10 or higher
- At least one of:
  - **Anthropic API key** (recommended — used by default pipelines)
  - **OpenAI API key**
  - **Ollama** running locally (free, no key required)

---

## 1. Clone and Install

```bash
git clone https://github.com/zacharylyj/Pikaia.git
cd Pikaia
pip install anthropic openai requests          # minimum dependencies
```

Optional (for Ollama local models):
```bash
# Install Ollama from https://ollama.com, then pull a model:
ollama pull llama3.2
```

---

## 2. Add API Keys

Open `Pikaia/keys.json` and fill in whichever providers you have:

```json
{
  "anthropic": "sk-ant-...",
  "openai":    "sk-...",
  "ollama":    ""
}
```

You only need the key(s) for the providers your configured pipelines use. The default `config.json` uses Anthropic models exclusively, so only `anthropic` is required out of the box.

---

## 3. Run the Setup Wizard

```bash
cd Pikaia
python init.py
```

The wizard will:
- Validate `config.json`, `models.json`, `keys.json`, and `tools/tools.json`
- Create the required directory structure (`memory/`, `projects/`, `skills/`)
- Initialise empty memory files (`lt.json`, `mt.json`, `kg.json`)
- Scaffold a default project workspace
- Report any missing or misconfigured items

### Validation-only mode (no writes)

```bash
python init.py --check
```

### Auto-fix recoverable issues

```bash
python init.py --fix
```

### Scaffold a named project

```bash
python init.py --project myproject
```

---

## 4. Start the CLI

```bash
python main.py
```

or with an explicit project:

```bash
python main.py --project myproject
```

You will see the Pikaia banner and a `>` prompt. Type any task or question.

---

## 5. First Interactions

### Ask a question

```
> What is the capital of France?
```

A Tier 1 agent handles this directly — fast, single tool loop, no decomposition.

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

## 6. Switching Models

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

## 7. User Preferences

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

## 8. Project Workspaces

A **project** is an isolated workspace with its own:
- `ct.json` — open task flags
- `dev/index.json` — file embedding index for RAG (auto-built when files are indexed)
- `instances/` — per-agent run directories

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

## 9. Memory Inspection

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

## 10. File Indexing for RAG

Place source files in your project's worker directory and the Orchestrator will index them automatically during `_reindex_file()` calls. The index is stored in:

```
projects/<project>/dev/index.json
```

Each entry contains a summary and embedding vector. `context_fetch` and `ContextManager.assess()` both search this index when building context for agents.

To trigger manual re-indexing, the Orchestrator calls `_reindex_file(path)` whenever a file is written by an agent.

---

## 11. SkillSmith (Automatic Skill Creation)

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

## 12. Troubleshooting

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

---

## 13. Configuration Quick Reference

| File | Purpose |
|------|---------|
| `Pikaia/config.json` | Models, thresholds, pipeline assignments |
| `Pikaia/models.json` | Registered LLM providers and capabilities |
| `Pikaia/keys.json` | API keys (never commit this) |
| `Pikaia/preferences.json` | User preferences injected into every context |
| `Pikaia/skills/` | Versioned skill templates |
| `Pikaia/memory/lt.json` | Long-term memory (persistent across sessions) |
| `Pikaia/memory/mt.json` | Medium-term memory (MemPalace format) |
| `Pikaia/memory/kg.json` | Knowledge graph (entity relationships) |

---

## 14. Architecture in One Paragraph

When you send a message, the Orchestrator builds context from five memory layers (LT facts, MT knowledge with MemPalace wing/room tagging, open CT flags, the current ST conversation, and a file embedding index), classifies your intent, picks or creates a skill, runs `ContextManager.assess()` to pre-enrich the task packet, then spawns the appropriate agent tier (1–4). Agents execute a ReAct tool loop — calling shell, file, HTTP, memory, and LLM tools as needed — and can request more context at any time via `context_fetch("plain English query")`. When the agent finishes, its output is written back into MT memory (enriched with entity extraction, AAAK compression, and KG triple updates) and may be promoted to LT. The next session starts richer.

For a full architecture breakdown, see [README.md](README.md).
