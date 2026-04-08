"""
schemas.py
----------
Anthropic tool_use input schemas for all 17 tools.
Used by agents to advertise available tools in llm_call requests.

Usage:
    from tools.schemas import get_schemas
    tool_schemas = get_schemas(["web_fetch", "file_write", "llm_call"])
"""

from __future__ import annotations

SCHEMAS: dict[str, dict] = {

    "shell_exec": {
        "name": "shell_exec",
        "description": "Run a shell command in a subprocess. Returns stdout, stderr, returncode.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cmd":     {"type": "string",  "description": "Shell command to run"},
                "cwd":     {"type": "string",  "description": "Working directory (default: base_path)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
            },
            "required": ["cmd"],
        },
    },

    "code_exec": {
        "name": "code_exec",
        "description": "Run Python or JavaScript in an isolated sandbox. Returns stdout, stderr, returncode.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code":     {"type": "string", "description": "Source code to execute"},
                "language": {"type": "string", "enum": ["python", "js"], "description": "Runtime (default: python)"},
                "timeout":  {"type": "integer", "description": "Timeout in seconds (default: 10)"},
            },
            "required": ["code"],
        },
    },

    "file_read": {
        "name": "file_read",
        "description": "Read a file as text. Path is relative to base_path. Returns content, path, size_bytes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to base_path"},
            },
            "required": ["path"],
        },
    },

    "file_write": {
        "name": "file_write",
        "description": "Write text to a file (atomic). Agents may only write to their own worker slot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "File path relative to base_path"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    },

    "http_request": {
        "name": "http_request",
        "description": "Make an HTTP request. Returns status_code, headers, body, ok.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method":  {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)"},
                "url":     {"type": "string", "description": "Full URL"},
                "headers": {"type": "object", "description": "Request headers"},
                "body":    {"description": "Request body (dict→JSON, str→raw)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
            },
            "required": ["method", "url"],
        },
    },

    "web_fetch": {
        "name": "web_fetch",
        "description": "Fetch a URL and return clean readable text (HTML stripped). Returns url, content, truncated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string",  "description": "URL to fetch"},
                "max_chars": {"type": "integer", "description": "Max characters to return (default: 8000)"},
                "timeout":   {"type": "integer", "description": "Timeout in seconds (default: 15)"},
            },
            "required": ["url"],
        },
    },

    "send_message": {
        "name": "send_message",
        "description": "Send a message via Telegram, Discord, or Slack (must be in config.interfaces).",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel":    {"type": "string", "description": "Channel: telegram | discord | slack"},
                "message":    {"type": "string", "description": "Message text"},
                "parse_mode": {"type": "string", "description": "Telegram parse mode: Markdown | HTML"},
            },
            "required": ["channel", "message"],
        },
    },

    "llm_call": {
        "name": "llm_call",
        "description": "Call an LLM via a named pipeline. Returns content, tokens_in, tokens_out, stop_reason.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pipeline":    {"type": "string",  "description": "Pipeline name (e.g. research, code_generation) or direct model_id"},
                "system":      {"type": "string",  "description": "System prompt"},
                "messages":    {"type": "array",   "description": "Conversation messages [{role, content}]"},
                "max_tokens":  {"type": "integer", "description": "Max tokens to generate (default: 1024)"},
                "temperature": {"type": "number",  "description": "Sampling temperature"},
            },
            "required": ["pipeline", "messages"],
        },
    },

    "embed_text": {
        "name": "embed_text",
        "description": "Generate an embedding vector for a string. Returns embedding, dim, model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to embed"},
            },
            "required": ["text"],
        },
    },

    "memory_read": {
        "name": "memory_read",
        "description": (
            "Read from a memory layer. "
            "layer=mt supports MemPalace retrieval: use wing/room to filter by domain/subtopic, "
            "palace_layer (0-3) for depth (1=essential story, 2=filtered, 3=full semantic). "
            "layer=kg queries the knowledge graph temporal triple store. "
            "Returns list of entries (or ST dict for layer=st)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "enum": ["lt", "mt", "ct", "st", "history", "kg"],
                    "description": "Memory layer to read",
                },
                "query":       {"type": "string",  "description": "Semantic search query (MT/History)"},
                "top_k":       {"type": "integer", "description": "Max results (default: 5)"},
                "project":     {"type": "string",  "description": "Project name (default: current)"},
                "instance_id": {"type": "string",  "description": "Instance ID (default: current)"},
                # Palace-specific (layer=mt)
                "wing": {
                    "type": "string",
                    "description": "Filter MT by wing domain (technical/decisions/knowledge/issues)",
                },
                "room": {
                    "type": "string",
                    "description": (
                        "Filter MT by room subtopic "
                        "(auth/api/code/data/deploy/testing/planning/decisions/"
                        "research/issues/architecture/performance/security)"
                    ),
                },
                "palace_layer": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": (
                        "MemPalace retrieval depth: "
                        "1=essential story (top by importance, grouped by room), "
                        "2=wing/room filtered cosine search, "
                        "3=full semantic search (default when no wing/room given)"
                    ),
                },
                # KG-specific (layer=kg)
                "subject":          {"type": "string", "description": "KG triple subject filter"},
                "predicate":        {"type": "string", "description": "KG triple predicate filter"},
                "object":           {"type": "string", "description": "KG triple object filter"},
                "as_of":            {"type": "string", "description": "ISO date for temporal KG query"},
                "subject_timeline": {"type": "string", "description": "Return full timeline for this subject"},
            },
            "required": ["layer"],
        },
    },

    "skill_read": {
        "name": "skill_read",
        "description": "Fetch active skill schema + template text by skill_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_id": {"type": "string", "description": "Skill ID to look up"},
            },
            "required": ["skill_id"],
        },
    },

    "ct_close": {
        "name": "ct_close",
        "description": "Close the CT flag for this task. Agents may only close their own flag.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID of the CT flag to close"},
                "status":  {"type": "string", "enum": ["done", "failed"],
                            "description": "Final status"},
            },
            "required": ["task_id", "status"],
        },
    },

    "context_fetch": {
        "name": "context_fetch",
        "description": (
            "Retrieve relevant context on demand. Provide a plain-English description of what "
            "you need to know. Returns knowledge from the memory base and relevant file snippets. "
            "Call this whenever you realise you're missing information needed to complete the task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Plain-English description of the context or information you need",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results per source (default: 5)",
                },
                "include_files": {
                    "type": "boolean",
                    "description": "Whether to include file snippets (default: true)",
                },
                "max_chars_per_file": {
                    "type": "integer",
                    "description": "Max characters to read per file snippet (default: 1500)",
                },
            },
            "required": ["query"],
        },
    },
}


def get_schemas(tool_names: list[str]) -> list[dict]:
    """Return Anthropic-format tool schemas for the given tool names."""
    return [SCHEMAS[n] for n in tool_names if n in SCHEMAS]
