"""
embed_text
----------
Generate a fixed-dimension embedding vector for a string.

Provider resolution order:
  1. OpenAI text-embedding-3-small (1536 dim) — if openai key in keys.json
  2. Ollama — if "ollama" in keys.json (null value is fine, key just needs to exist)
  3. Deterministic fallback — hash-based 1536-dim float vector (no API call)
     Useful for development / testing. NOT semantically meaningful.

The returned dimension always matches config["embedding_dim"] (default 1536).

params:
    text : str   - string to embed

returns:
    embedding : list[float]   - embedding vector
    dim       : int           - vector dimension
    model     : str           - model/method used
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path
from typing import Any


def run(params: dict, context: dict) -> dict[str, Any]:
    text      = params["text"]
    base_path = Path(context["base_path"])
    config    = context.get("config", {})
    target_dim = config.get("embedding_dim", 1536)

    # Load keys
    keys_path = base_path / "keys.json"
    keys: dict = {}
    if keys_path.exists():
        try:
            keys = json.loads(keys_path.read_text())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Strategy 1: OpenAI
    # ------------------------------------------------------------------
    openai_key = keys.get("openai")
    if openai_key:
        try:
            vec = _openai_embed(text, openai_key, target_dim)
            return {"embedding": vec, "dim": len(vec), "model": "text-embedding-3-small"}
        except Exception:
            pass  # fall through

    # ------------------------------------------------------------------
    # Strategy 2: Ollama (local)
    # ------------------------------------------------------------------
    if "ollama" in keys:
        try:
            models_path = base_path / "models.json"
            embed_model = "nomic-embed-text"  # common ollama embedding model
            if models_path.exists():
                mlist = json.loads(models_path.read_text())
                if isinstance(mlist, dict):
                    mlist = [mlist]
                om = next((m for m in mlist if "embed" in m.get("model_id", "")), None)
                if om:
                    embed_model = om["model_id"]

            import sys as _sys
            _pikaia = str(base_path)
            if _pikaia not in _sys.path:
                _sys.path.insert(0, _pikaia)
            _mod_name = "tools.providers.ollama"
            _omod = _sys.modules.get(_mod_name) or importlib.import_module(_mod_name)
            adapter = _omod.Adapter(api_key=None, model_id=embed_model)
            vec = adapter.embed(text)
            vec = _pad_or_truncate(vec, target_dim)
            return {"embedding": vec, "dim": len(vec), "model": f"ollama/{embed_model}"}
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Strategy 3: Deterministic hash fallback
    # ------------------------------------------------------------------
    vec = _hash_embed(text, target_dim)
    return {"embedding": vec, "dim": len(vec), "model": "hash-fallback"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _openai_embed(text: str, api_key: str, target_dim: int) -> list[float]:
    import urllib.request
    import urllib.error

    payload = {"model": "text-embedding-3-small", "input": text}
    body    = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body, headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    vec = data["data"][0]["embedding"]
    return _pad_or_truncate(vec, target_dim)


def _hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding from SHA-256. NOT semantic — for dev only."""
    seed   = text.encode("utf-8")
    floats: list[float] = []
    i      = 0
    while len(floats) < dim:
        digest = hashlib.sha256(seed + struct.pack(">I", i)).digest()
        # Each digest gives 8 floats (32 bytes / 4 bytes per float)
        for j in range(0, 32, 4):
            bits, = struct.unpack(">I", digest[j:j+4])
            # Map uint32 → [-1, 1]
            floats.append((bits / 2_147_483_647.5) - 1.0)
        i += 1
    vec = floats[:dim]
    # L2-normalise
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _pad_or_truncate(vec: list[float], dim: int) -> list[float]:
    if len(vec) == dim:
        return vec
    if len(vec) > dim:
        return vec[:dim]
    # Pad with zeros
    return vec + [0.0] * (dim - len(vec))
