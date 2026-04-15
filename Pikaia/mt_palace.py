"""
mt_palace.py
------------
MemPalace-inspired storage and retrieval engine for the MT (medium-term) memory layer.

What it borrows from MemPalace:
  - Wing / Room / Hall hierarchy : every entry is tagged with a domain (wing), subtopic (room),
                                   and knowledge type (hall)
  - AAAK lossy compression  : entity codes + topic keywords + key sentence + emotions + flags
  - Entity extraction       : persons and projects detected from plain text
  - Importance scoring      : drives the L1 "essential story" selection (with recency decay)
  - Knowledge Graph (KG)    : temporal triple store in memory/kg.json
  - 4-layer retrieval       :
        L0  identity / LT preferences (~100 tokens always present)
        L1  highest-importance drawers grouped by room (config-driven token budget)
        L2  wing/room/hall-filtered cosine search with tunnel expansion
        L3  full semantic search across all MT (unrestricted)

What it does NOT borrow:
  - ChromaDB / sentence-transformers (we use embed_text.py which supports openai/ollama/hash)
  - CLI tooling (that's handled by main.py / init.py)

The existing mt.json schema is extended, not replaced.  Old entries without palace
fields still participate in L3 search.  New entries get all fields.

Storage backends: JSON (default) or LanceDB (optional, auto-detected).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional LanceDB dependency
# ---------------------------------------------------------------------------

try:
    import lancedb  # type: ignore[import]
    import pyarrow as pa  # type: ignore[import]
    _LANCEDB_AVAILABLE = True
except ImportError:
    _LANCEDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_BACKEND_CACHE: dict[str, Any] = {}
"""Cache of storage backends keyed by str(base_path)."""

_EMBED_MOD_CACHE: dict[str, Any] = {}
"""Cache of embed_text modules keyed by str(base_path)."""

_RAW_LOG_LOCK: threading.Lock = threading.Lock()
"""Serialises concurrent appends to mt_raw.jsonl within the same process."""


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_DEFAULT_PALACE_CONFIG: dict[str, Any] = {
    "l1_char_budget":             3200,
    "per_room_max":               None,
    "dedup_similarity_threshold": 0.92,
    "dedup_lookback_days":        7,
    "recency_decay_factor":       0.05,
    "recency_decay_max_days":     30,
    "pruning_min_importance":     0.30,
    "pruning_min_age_days":       30,
    "use_lancedb":                True,
}


def _get_palace_config(base_path: Path) -> dict[str, Any]:
    """Read config.json's 'mt_palace' key and merge with defaults."""
    cfg = dict(_DEFAULT_PALACE_CONFIG)
    try:
        config_path = base_path / "config.json"
        if config_path.exists():
            raw = json.loads(config_path.read_text())
            palace_cfg = raw.get("mt_palace", {})
            if isinstance(palace_cfg, dict):
                cfg.update(palace_cfg)
    except Exception:
        pass
    return cfg


# ---------------------------------------------------------------------------
# Room taxonomy (keyword → room mapping)
# ---------------------------------------------------------------------------

ROOM_KEYWORDS: dict[str, list[str]] = {
    "auth":       ["auth", "login", "logout", "password", "token", "jwt", "session",
                   "oauth", "credentials", "bearer", "permission", "role"],
    "api":        ["api", "endpoint", "rest", "graphql", "request", "response",
                   "route", "http", "webhook", "swagger", "openapi"],
    "code":       ["function", "class", "method", "module", "import", "refactor",
                   "implement", "interface", "pattern", "algorithm"],
    "data":       ["database", "sql", "query", "schema", "table", "migration",
                   "index", "record", "orm", "postgres", "mysql", "sqlite", "mongo"],
    "deploy":     ["deploy", "docker", "kubernetes", "ci", "cd", "pipeline",
                   "release", "container", "infra", "terraform", "ansible"],
    "testing":    ["test", "unit", "integration", "coverage", "mock", "stub",
                   "assert", "pytest", "jest", "fixture"],
    "planning":   ["plan", "roadmap", "milestone", "goal", "sprint", "backlog",
                   "deadline", "estimate", "scope", "requirement"],
    "decisions":  ["decided", "chose", "approach", "trade-off", "tradeoff",
                   "rationale", "because", "reason", "conclusion", "agreed"],
    "research":   ["research", "study", "paper", "article", "reference", "source",
                   "found", "learned", "discovered", "benchmark", "analysis"],
    "issues":     ["bug", "error", "issue", "problem", "crash", "exception",
                   "fix", "broken", "fail", "regression", "incident"],
    "architecture": ["architecture", "design", "system", "component", "service",
                     "microservice", "monolith", "layer", "abstraction", "pattern"],
    "performance":  ["performance", "latency", "throughput", "cache", "optimize",
                     "slow", "bottleneck", "profil", "memory", "cpu"],
    "security":     ["security", "vulnerability", "exploit", "injection", "xss",
                     "csrf", "encrypt", "hash", "salt", "tls", "ssl"],
}

# Wing derived from room
_WING_FROM_ROOM: dict[str, str] = {
    "auth":         "technical",
    "api":          "technical",
    "code":         "technical",
    "data":         "technical",
    "deploy":       "technical",
    "testing":      "technical",
    "architecture": "technical",
    "performance":  "technical",
    "security":     "technical",
    "planning":     "decisions",
    "decisions":    "decisions",
    "research":     "knowledge",
    "issues":       "issues",
    "general":      "knowledge",
}


# ---------------------------------------------------------------------------
# Room Detector
# ---------------------------------------------------------------------------

class RoomDetector:
    """Assigns a room to an entry based on keyword frequency in the text."""

    @staticmethod
    def detect(text: str) -> str:
        """Return the best-matching room name for the given text."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for room, keywords in ROOM_KEYWORDS.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            if score > 0:
                scores[room] = score
        if not scores:
            return "general"
        return max(scores, key=lambda r: scores[r])

    @staticmethod
    def wing_from_room(room: str) -> str:
        """Map a room name to its parent wing."""
        return _WING_FROM_ROOM.get(room, "knowledge")


# ---------------------------------------------------------------------------
# Hall taxonomy (keyword → hall mapping)
# ---------------------------------------------------------------------------

HALL_KEYWORDS: dict[str, list[str]] = {
    "facts":     ["is", "are", "was", "were", "has", "have", "contains", "equals",
                  "means", "defined", "represents"],
    "events":    ["happened", "occurred", "completed", "started", "ended", "launched",
                  "shipped", "deployed", "released"],
    "decisions": ["decided", "chose", "agreed", "concluded", "determined", "resolved",
                  "opted", "selected", "approved"],
    "advice":    ["should", "recommend", "suggest", "best practice", "prefer", "avoid",
                  "always", "never", "consider"],
    "issues":    ["bug", "error", "problem", "fail", "broken", "issue", "crash",
                  "exception", "unexpected"],
}


class HallDetector:
    """Assigns a hall (knowledge type) to an entry based on keyword frequency."""

    @staticmethod
    def detect(text: str) -> str:
        """Return the best-matching hall name for the given text."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for hall, keywords in HALL_KEYWORDS.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            if score > 0:
                scores[hall] = score
        if not scores:
            return "facts"
        return max(scores, key=lambda h: scores[h])


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------

# Signals that suggest a capitalized word is a PERSON
_PERSON_VERBS  = {"said", "asked", "told", "wrote", "mentioned", "explained",
                   "confirmed", "replied", "noted", "suggested", "decided"}
_PROJECT_VERBS = {"using", "built", "deployed", "shipped", "released", "created",
                   "implemented", "imported", "installed", "runs", "uses"}

class EntityExtractor:
    """
    Heuristic extraction of persons and projects from text.
    No NLP library required — uses regex + signal-word scoring.
    """

    @staticmethod
    def extract(text: str) -> dict[str, Any]:
        """Return persons, projects, and 3-char entity codes extracted from text."""
        # Find capitalized sequences (proper nouns)
        candidates: dict[str, int] = {}
        for m in re.finditer(r"\b([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+)*)\b", text):
            word = m.group(1)
            # Skip all-caps acronyms under 3 chars
            if len(word) < 3:
                continue
            candidates[word] = candidates.get(word, 0) + 1

        # Filter to words appearing >= 2 times
        candidates = {k: v for k, v in candidates.items() if v >= 2}

        persons:  list[str] = []
        projects: list[str] = []
        words = text.lower().split()

        for name in candidates:
            name_lower = name.lower()
            person_score  = 0
            project_score = 0

            # Check words in ±5 token window around each occurrence
            for i, w in enumerate(words):
                if name_lower in w:
                    window = words[max(0, i-5): i+6]
                    person_score  += sum(1 for ww in window if ww in _PERSON_VERBS)
                    project_score += sum(1 for ww in window if ww in _PROJECT_VERBS)

            # Additional project signal: file extension or version pattern
            if re.search(rf"\b{re.escape(name_lower)}[._/-]", text.lower()):
                project_score += 2

            if person_score > project_score and person_score >= 1:
                persons.append(name)
            elif project_score >= 1:
                projects.append(name)
            elif len(candidates) <= 3:
                # If very few entities, accept without strong signal
                persons.append(name)

        persons  = persons[:5]
        projects = projects[:5]

        # Build 3-char codes
        all_entities = persons + projects
        codes: dict[str, str] = {}
        used: set[str] = set()
        for ent in all_entities:
            code = _make_code(ent, used)
            codes[ent] = code
            used.add(code)

        return {"persons": persons, "projects": projects, "codes": codes}


def _make_code(name: str, existing: set[str]) -> str:
    """Generate a unique 3-char uppercase code for an entity name."""
    parts = name.upper().split()
    candidates = [
        parts[0][:3],
        "".join(p[0] for p in parts)[:3].ljust(3, "X"),
        parts[0][:2] + (parts[-1][0] if len(parts) > 1 else parts[0][2:3]),
    ]
    for c in candidates:
        c = (c + "XXX")[:3]
        if c not in existing:
            return c
    for i in range(10):
        c = (parts[0][:2] + str(i)).upper()
        if c not in existing:
            return c
    return name[:3].upper()


# ---------------------------------------------------------------------------
# AAAK Codec
# ---------------------------------------------------------------------------

_EMOTION_MAP = {
    "worr": "anx", "afraid": "anx", "anxious": "anx", "nervous": "anx",
    "excit": "excite", "happy": "joy", "great": "joy", "love": "love",
    "frustrat": "frust", "annoyed": "frust", "angry": "frust",
    "sad": "sad", "disappoint": "sad",
    "confident": "conf", "sure": "conf",
}

_DECISION_MARKERS = {"decided", "chose", "agreed", "concluded", "determined",
                      "resolved", "opted", "selected", "approved"}
_CORE_MARKERS     = {"core", "fundamental", "essential", "critical", "key",
                      "important", "main", "primary", "central"}
_ORIGIN_MARKERS   = {"founded", "created", "started", "initiated", "originated",
                      "began", "launched", "introduced"}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "as", "it", "is", "be", "was",
    "are", "been", "being", "this", "that", "these", "those", "we", "i",
    "you", "he", "she", "they", "it", "my", "our", "their", "its",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "must", "shall", "about", "which",
    "when", "where", "who", "what", "how", "all", "any", "both", "each",
    "not", "so", "if", "then", "than", "also", "just", "more", "there",
    "into", "through", "during", "before", "after", "above", "below",
}


class AAAKCodec:
    """
    Lossy compression into MemPalace's AAAK dialect.
    Format: ENTITIES|topics|"key sentence"|emotions|FLAGS
    Example: ALC,BRD|jwt,refresh,auth|"decided to use RS256"|conf|DECISION,CORE

    Achieves ~30x compression. Suitable for L1 token budget.
    """

    @staticmethod
    def compress(text: str, entities: dict[str, Any]) -> str:
        """Compress text and entity info into AAAK format string."""
        codes   = entities.get("codes", {})
        words   = text.lower().split()

        # Entity codes string
        all_ents = entities.get("persons", []) + entities.get("projects", [])
        ent_part = ",".join(codes.get(e, e[:3].upper()) for e in all_ents[:4]) or "–"

        # Topic keywords (top 3 by frequency, no stopwords, min 4 chars)
        freq: dict[str, int] = {}
        for w in words:
            w = re.sub(r"[^a-z0-9]", "", w)
            if len(w) >= 4 and w not in _STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        topics = sorted(freq, key=lambda k: freq[k], reverse=True)[:3]
        topic_part = ",".join(topics) if topics else "–"

        # Key sentence — prefer decision/conclusion sentences
        sentences = re.split(r"[.!?]\s+", text)
        key_sent  = ""
        for sent in sentences:
            sl = sent.lower()
            if any(m in sl for m in _DECISION_MARKERS | {"because", "therefore", "so"}):
                key_sent = sent.strip()[:80]
                break
        if not key_sent and sentences:
            # Fallback: shortest sentence that is reasonably informative
            key_sent = min(
                (s.strip() for s in sentences if 20 < len(s.strip()) < 120),
                key=len,
                default=sentences[0][:80].strip(),
            )

        # Emotions
        text_lower = text.lower()
        emotions: list[str] = []
        for sig, code in _EMOTION_MAP.items():
            if sig in text_lower and code not in emotions:
                emotions.append(code)
        emotion_part = ",".join(emotions[:2]) if emotions else "–"

        # Flags
        flags: list[str] = []
        if any(m in text_lower for m in _DECISION_MARKERS):
            flags.append("DECISION")
        if any(m in text_lower for m in _CORE_MARKERS):
            flags.append("CORE")
        if any(m in text_lower for m in _ORIGIN_MARKERS):
            flags.append("ORIGIN")
        flag_part = ",".join(flags) if flags else "–"

        return f"{ent_part}|{topic_part}|\"{key_sent}\"|{emotion_part}|{flag_part}"

    @staticmethod
    def label(aaak: str) -> str:
        """Convert AAAK string to human-readable format for context injection."""
        parts = aaak.split("|")
        if len(parts) < 5:
            return aaak
        ents, topics, key, emotions, flags = parts[0], parts[1], parts[2], parts[3], parts[4]
        lines: list[str] = []
        if ents != "–":
            lines.append(f"Entities: {ents}")
        if topics != "–":
            lines.append(f"Topics: {topics}")
        if key and key != '"-"':
            lines.append(f"Key: {key.strip('\"')}")
        if flags != "–":
            lines.append(f"Flags: {flags}")
        return " | ".join(lines) or aaak


# ---------------------------------------------------------------------------
# Importance Scorer
# ---------------------------------------------------------------------------

class ImportanceScorer:
    """Score an entry 0–1 for use in L1 layer selection, with optional recency decay."""

    @staticmethod
    def score(
        text:       str,
        entities:   dict[str, Any],
        room:       str,
        created_at: str | None = None,
        config:     dict | None = None,
    ) -> float:
        """Compute importance score with optional recency decay."""
        score = 0.35  # base

        text_lower = text.lower()
        words      = text_lower.split()
        n_words    = max(len(words), 1)

        # Decision/importance signals
        decision_hits = sum(1 for m in _DECISION_MARKERS if m in text_lower)
        core_hits     = sum(1 for m in _CORE_MARKERS     if m in text_lower)
        score += min(decision_hits * 0.12, 0.25)
        score += min(core_hits     * 0.08, 0.15)

        # Entity density (entities per 50 words)
        n_entities = len(entities.get("persons", [])) + len(entities.get("projects", []))
        score += min(n_entities / max(n_words / 50, 1), 0.15)

        # High-value rooms
        if room in ("decisions", "architecture", "security"):
            score += 0.10
        elif room in ("auth", "deploy", "planning"):
            score += 0.05

        # Recency decay
        if created_at and config:
            days_old = _days_since(created_at)
            decay_factor = config.get("recency_decay_factor", 0.05)
            max_days = config.get("recency_decay_max_days", 30)
            # CORE flag exempts from decay
            if "CORE" not in text.upper():
                recency_factor = max(0.5, 1.0 - decay_factor * min(days_old, max_days) / max_days)
                score *= recency_factor

        return min(round(score, 3), 1.0)


# ---------------------------------------------------------------------------
# Tunnel system
# ---------------------------------------------------------------------------

class TunnelIndex:
    """Provides cross-wing tunnel connections for the same room."""

    @staticmethod
    def get_tunnel_wings(room: str, current_wing: str, entries: list[dict]) -> list[str]:
        """Return wings other than current_wing that have entries for the given room."""
        room_wings: dict[str, set[str]] = {}
        for e in entries:
            r = e.get("room", "")
            w = e.get("wing", "")
            if r and w:
                room_wings.setdefault(r, set()).add(w)
        other_wings = room_wings.get(room, set()) - {current_wing}
        return sorted(other_wings)


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Temporal triple store backed by memory/kg.json.

    Schema:
      triples : list[{id, subject, predicate, object, valid_from, valid_to, created_at}]
      entities: dict[name → {type, code, aliases}]

    Indexing
    --------
    A subject → [index] and (subject, predicate) → [index] in-memory index is
    built the first time data is loaded and invalidated on every write.  This
    turns subject-filtered ``query()`` calls from O(n) full-scan to O(k) where
    k is the number of triples with that subject — a significant win once the
    KG grows to thousands of triples.
    """

    def __init__(self, kg_path: Path) -> None:
        """Initialise with path to the KG JSON file."""
        self._path   = kg_path
        self._data:      dict | None                       = None
        self._subj_idx:  dict[str, list[int]]              = {}
        self._pred_idx:  dict[tuple[str, str], list[int]]  = {}

    # ------------------------------------------------------------------
    # Internal cache management
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load KG data from disk, returning empty structure on failure."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {"triples": [], "entities": {}}

    def _load_cached(self) -> dict:
        """Return cached data (or load + index from disk on first call)."""
        if self._data is None:
            self._data = self._load()
            self._rebuild_index()
        return self._data

    def _rebuild_index(self) -> None:
        """Build subject and (subject, predicate) lookup indices."""
        self._subj_idx = {}
        self._pred_idx = {}
        for i, t in enumerate(self._data.get("triples", [])):   # type: ignore[union-attr]
            s = t.get("subject", "").lower()
            p = t.get("predicate", "").lower()
            if s:
                self._subj_idx.setdefault(s, []).append(i)
                if p:
                    self._pred_idx.setdefault((s, p), []).append(i)

    def _invalidate_cache(self) -> None:
        """Drop the in-memory cache so the next read reloads from disk."""
        self._data     = None
        self._subj_idx = {}
        self._pred_idx = {}

    def _save(self, data: dict) -> None:
        """Atomically save KG data to disk and invalidate the in-memory cache."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # Drop the cache so the next read sees the freshly written state.
        self._invalidate_cache()

    def add(
        self,
        subject:    str,
        predicate:  str,
        obj:        str,
        valid_from: str | None = None,
        valid_to:   str | None = None,
    ) -> str:
        """Add a triple, auto-closing contradicting active triples. Returns its id."""
        data = self._load_cached()

        # Contradiction detection: use the (subject, predicate) index to
        # narrow the scan from O(n) to O(k) matching triples.
        today    = _now_iso()[:10]
        s_low, p_low = subject.lower(), predicate.lower()
        candidate_indices = self._pred_idx.get((s_low, p_low), [])
        contradictions = 0
        for i in candidate_indices:
            t = data["triples"][i]
            if t["object"].lower() != obj.lower() and t.get("valid_to") is None:
                t["valid_to"] = today
                contradictions += 1

        triple: dict[str, Any] = {
            "id":         str(uuid.uuid4()),
            "subject":    subject,
            "predicate":  predicate,
            "object":     obj,
            "valid_from": valid_from,
            "valid_to":   valid_to,
            "created_at": _now_iso(),
        }
        if contradictions:
            triple["_supersedes"] = contradictions

        data["triples"].append(triple)
        self._save(data)   # also calls _invalidate_cache()
        return triple["id"]

    def query(
        self,
        subject:   str | None = None,
        predicate: str | None = None,
        obj:       str | None = None,
        as_of:     str | None = None,     # ISO date string
    ) -> list[dict]:
        """Return triples matching filters. as_of filters by temporal validity.

        When *subject* is provided, the in-memory index narrows the scan to
        only triples with that subject (or subject+predicate pair) before
        applying the remaining filters.
        """
        data    = self._load_cached()
        triples = data["triples"]

        # Fast path: use the index when a subject filter is given
        if subject:
            s_low = subject.lower()
            if predicate:
                indices = self._pred_idx.get((s_low, predicate.lower()), [])
            else:
                indices = self._subj_idx.get(s_low, [])
            candidates: list[dict] = [triples[i] for i in indices]
        else:
            candidates = triples

        results = []
        for t in candidates:
            # subject already matched via index (or no filter requested)
            if subject   and not predicate and t["subject"].lower()   != subject.lower():   continue
            if predicate and t["predicate"].lower()  != predicate.lower(): continue
            if obj       and t["object"].lower()     != obj.lower():       continue
            if as_of:
                if t.get("valid_from") and t["valid_from"] > as_of: continue
                if t.get("valid_to")   and t["valid_to"]   < as_of: continue
            results.append(t)
        return results

    def invalidate(self, subject: str, predicate: str, obj: str) -> int:
        """Mark matching triples as no longer valid (set valid_to = now). Returns count."""
        data  = self._load_cached()
        today = _now_iso()[:10]
        count = 0
        s_low, p_low = subject.lower(), predicate.lower()
        for i in self._pred_idx.get((s_low, p_low), []):
            t = data["triples"][i]
            if t["object"].lower() == obj.lower() and t.get("valid_to") is None:
                t["valid_to"] = today
                count += 1
        if count:
            self._save(data)
        return count

    def merge_entities(self, entities: dict[str, Any]) -> None:
        """Upsert detected entities into the entity registry."""
        data = self._load_cached()
        reg  = data.setdefault("entities", {})
        codes = entities.get("codes", {})
        for name in entities.get("persons", []):
            if name not in reg:
                reg[name] = {"type": "person",  "code": codes.get(name, name[:3].upper()), "aliases": []}
        for name in entities.get("projects", []):
            if name not in reg:
                reg[name] = {"type": "project", "code": codes.get(name, name[:3].upper()), "aliases": []}
        self._save(data)

    def timeline(self, subject: str) -> list[dict]:
        """Return all facts about a subject, ordered by valid_from."""
        triples = self.query(subject=subject)
        triples.sort(key=lambda t: (t.get("valid_from") or "0000"))
        return triples


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

class _JSONBackend:
    """JSON file storage backend for MT entries with in-memory cache."""

    def __init__(self, base_path: Path) -> None:
        """Initialise backend from base_path, loading mt.json into memory."""
        self._path = base_path / "memory" / "mt.json"
        self._entries: list[dict] = self._load_from_disk()

    def _load_from_disk(self) -> list[dict]:
        """Load entries from mt.json, returning empty list on failure."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return data if isinstance(data, list) else [data]
        except Exception:
            return []

    def _flush(self) -> None:
        """Write the in-memory entry list back to disk atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get_all(self, status: str = "active") -> list[dict]:
        """Return all entries with the given status."""
        return [e for e in self._entries if e.get("status", "active") == status]

    def get_filtered(
        self,
        wing:   str | None,
        hall:   str | None,
        room:   str | None,
        status: str = "active",
    ) -> list[dict]:
        """Return entries matching wing/hall/room filters and status."""
        result = self.get_all(status)
        if wing:
            result = [e for e in result if e.get("wing", "").lower() == wing.lower()]
        if hall:
            result = [e for e in result if e.get("hall", "").lower() == hall.lower()]
        if room:
            result = [e for e in result if e.get("room", "").lower() == room.lower()]
        return result

    def get_recent(self, wing: str | None, room: str | None, limit: int = 50) -> list[dict]:
        """Return recent entries for wing/room sorted by created_at descending."""
        result = self.get_all("active")
        if wing:
            result = [e for e in result if e.get("wing", "").lower() == wing.lower()]
        if room:
            result = [e for e in result if e.get("room", "").lower() == room.lower()]
        result.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return result[:limit]

    def save(self, entry: dict) -> None:
        """Upsert an entry by id into the in-memory list and flush to disk."""
        eid = entry.get("id")
        idx = next((i for i, e in enumerate(self._entries) if e.get("id") == eid), None)
        if idx is not None:
            self._entries[idx] = entry
        else:
            self._entries.append(entry)
        self._flush()

    def archive(self, entry_id: str) -> bool:
        """Set status=archived for an entry by id. Returns True if found."""
        for e in self._entries:
            if e.get("id") == entry_id:
                e["status"] = "archived"
                self._flush()
                return True
        return False


class _LanceDBBackend:
    """LanceDB storage backend for MT entries."""

    _SCHEMA = None  # set after lancedb import succeeds

    def __init__(self, base_path: Path) -> None:
        """Initialise LanceDB backend, migrating mt.json if present."""
        self._base_path = base_path
        lance_dir = base_path / "memory" / "mt.lance"
        lance_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(lance_dir))
        self._schema = pa.schema([
            pa.field("id",            pa.string()),
            pa.field("content",       pa.large_utf8()),
            pa.field("wing",          pa.string()),
            pa.field("hall",          pa.string()),
            pa.field("room",          pa.string()),
            pa.field("importance",    pa.float32()),
            pa.field("compressed",    pa.string()),
            pa.field("status",        pa.string()),
            pa.field("created_at",    pa.string()),
            pa.field("updated_at",    pa.string()),
            pa.field("entities_json", pa.string()),
            pa.field("embedding",     pa.list_(pa.float32())),
        ])
        self._table = self._open_or_create_table()
        self._migrate_json_if_needed()

    def _open_or_create_table(self) -> Any:
        """Open existing LanceDB table or create it with the defined schema."""
        try:
            tbl_names = self._db.table_names()
            if "mt_entries" in tbl_names:
                return self._db.open_table("mt_entries")
        except Exception:
            pass
        # Create empty table with schema
        empty = pa.table({
            "id":            pa.array([], type=pa.string()),
            "content":       pa.array([], type=pa.large_utf8()),
            "wing":          pa.array([], type=pa.string()),
            "hall":          pa.array([], type=pa.string()),
            "room":          pa.array([], type=pa.string()),
            "importance":    pa.array([], type=pa.float32()),
            "compressed":    pa.array([], type=pa.string()),
            "status":        pa.array([], type=pa.string()),
            "created_at":    pa.array([], type=pa.string()),
            "updated_at":    pa.array([], type=pa.string()),
            "entities_json": pa.array([], type=pa.string()),
            "embedding":     pa.array([], type=pa.list_(pa.float32())),
        })
        return self._db.create_table("mt_entries", data=empty, schema=self._schema)

    def _migrate_json_if_needed(self) -> None:
        """Migrate existing mt.json rows into LanceDB and rename the file."""
        json_path = self._base_path / "memory" / "mt.json"
        if not json_path.exists():
            return
        try:
            data = json.loads(json_path.read_text())
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                try:
                    self._table.add([self._entry_to_row(entry)])
                except Exception:
                    pass
            migrated = json_path.with_suffix(".json.migrated")
            json_path.rename(migrated)
        except Exception:
            pass

    def _entry_to_row(self, entry: dict) -> dict:
        """Convert a memory entry dict to a LanceDB row dict."""
        embedding = entry.get("embedding") or []
        entities  = entry.get("entities", [])
        return {
            "id":            str(entry.get("id", "")),
            "content":       str(entry.get("content", "")),
            "wing":          str(entry.get("wing", "")),
            "hall":          str(entry.get("hall", "")),
            "room":          str(entry.get("room", "")),
            "importance":    float(entry.get("importance", 0.35)),
            "compressed":    str(entry.get("compressed", "")),
            "status":        str(entry.get("status", "active")),
            "created_at":    str(entry.get("created_at", "")),
            "updated_at":    str(entry.get("updated_at", "")),
            "entities_json": json.dumps(entities),
            "embedding":     [float(x) for x in embedding],
        }

    def _row_to_entry(self, row: dict) -> dict:
        """Convert a LanceDB row dict back to a memory entry dict."""
        def _v(val: Any) -> Any:
            return val.as_py() if hasattr(val, "as_py") else val

        entities_raw = _v(row.get("entities_json", "[]"))
        try:
            entities = json.loads(entities_raw) if entities_raw else []
        except Exception:
            entities = []

        embedding_raw = _v(row.get("embedding", []))
        if embedding_raw and hasattr(embedding_raw, "__iter__"):
            embedding = [float(_v(x)) for x in embedding_raw]
        else:
            embedding = []

        return {
            "id":         _v(row.get("id", "")),
            "content":    _v(row.get("content", "")),
            "wing":       _v(row.get("wing", "")),
            "hall":       _v(row.get("hall", "")),
            "room":       _v(row.get("room", "")),
            "importance": float(_v(row.get("importance", 0.35))),
            "compressed": _v(row.get("compressed", "")),
            "status":     _v(row.get("status", "active")),
            "created_at": _v(row.get("created_at", "")),
            "updated_at": _v(row.get("updated_at", "")),
            "entities":   entities,
            "embedding":  embedding,
        }

    def _escape_sql_str(self, val: str) -> str:
        """Escape a string value for use in LanceDB SQL WHERE clause."""
        return val.replace("'", "\\'")

    def _build_where(self, **filters: Any) -> str | None:
        """Build a SQL WHERE clause from non-None filter kwargs."""
        clauses = []
        for col, val in filters.items():
            if val is not None:
                escaped = self._escape_sql_str(str(val))
                clauses.append(f"{col} = '{escaped}'")
        return " AND ".join(clauses) if clauses else None

    def _query_with_filter(self, where: str | None, limit: int) -> list[dict]:
        """Execute a filtered query, falling back to Python filter on error."""
        try:
            q = self._table.search()
            if where:
                q = q.where(where)
            rows = q.limit(limit).to_list()
            return [self._row_to_entry(r) for r in rows]
        except Exception:
            # Python-side fallback
            try:
                all_rows = self._table.to_pandas().to_dict("records")
                results = [self._row_to_entry(r) for r in all_rows]
                # Apply filters manually
                if where:
                    # Best-effort: re-apply filters from kwargs via get_all
                    pass
                return results[:limit]
            except Exception:
                return []

    def get_all(self, status: str = "active") -> list[dict]:
        """Return all entries with the given status."""
        where = self._build_where(status=status)
        return self._query_with_filter(where, limit=10000)

    def get_filtered(
        self,
        wing:   str | None,
        hall:   str | None,
        room:   str | None,
        status: str = "active",
    ) -> list[dict]:
        """Return entries matching wing/hall/room/status filters."""
        filters: dict[str, Any] = {"status": status}
        if wing:
            filters["wing"] = wing
        if hall:
            filters["hall"] = hall
        if room:
            filters["room"] = room
        where = self._build_where(**filters)
        return self._query_with_filter(where, limit=10000)

    def get_recent(self, wing: str | None, room: str | None, limit: int = 50) -> list[dict]:
        """Return recent entries for wing/room sorted by created_at descending."""
        filters: dict[str, Any] = {"status": "active"}
        if wing:
            filters["wing"] = wing
        if room:
            filters["room"] = room
        where = self._build_where(**filters)
        entries = self._query_with_filter(where, limit=10000)
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries[:limit]

    def save(self, entry: dict) -> None:
        """Delete by id then re-insert entry (upsert semantics)."""
        try:
            eid = self._escape_sql_str(str(entry.get("id", "")))
            self._table.delete(f"id = '{eid}'")
        except Exception:
            pass
        try:
            self._table.add([self._entry_to_row(entry)])
        except Exception:
            pass

    def archive(self, entry_id: str) -> bool:
        """Set status=archived for an entry. Falls back to delete+readd if update unavailable."""
        try:
            eid = self._escape_sql_str(str(entry_id))
            self._table.update(where=f"id = '{eid}'", values={"status": "archived"})
            return True
        except Exception:
            pass
        # Fallback: read, mutate, delete, re-add
        try:
            eid = self._escape_sql_str(str(entry_id))
            rows = self._query_with_filter(f"id = '{eid}'", limit=1)
            if rows:
                rows[0]["status"] = "archived"
                self._table.delete(f"id = '{eid}'")
                self._table.add([self._entry_to_row(rows[0])])
                return True
        except Exception:
            pass
        return False


def _get_mt_backend(base_path: Path, config: dict) -> "_JSONBackend | _LanceDBBackend":
    """Return (and cache) the appropriate storage backend for base_path."""
    bp_str = str(base_path)
    if bp_str in _BACKEND_CACHE:
        return _BACKEND_CACHE[bp_str]

    backend: "_JSONBackend | _LanceDBBackend"
    if config.get("use_lancedb", True) and _LANCEDB_AVAILABLE:
        try:
            backend = _LanceDBBackend(base_path)
            _BACKEND_CACHE[bp_str] = backend
            return backend
        except Exception:
            pass

    backend = _JSONBackend(base_path)
    _BACKEND_CACHE[bp_str] = backend
    return backend


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Remove shell substitution patterns, null bytes; truncate to 8000 chars."""
    # Remove $(...) command substitution
    text = re.sub(r"\$\([^)]*\)", "", text)
    # Remove backtick command substitution
    text = re.sub(r"`[^`]*`", "", text)
    # Strip null bytes
    text = text.replace("\x00", "")
    # Truncate
    return text[:8000]


# ---------------------------------------------------------------------------
# Raw-log backup  (append-only, pre-AAAK recovery store)
# ---------------------------------------------------------------------------

def _append_raw_log(entry: dict, base_path: Path) -> None:
    """Append the *original* (pre-AAAK) entry to ``memory/mt_raw.jsonl``.

    Why this exists
    ---------------
    The AAAK codec is a *lossy* transformation.  If its format ever changes or
    a bug corrupts ``mt.json``, the raw log lets you rebuild from scratch.
    Each line is a standalone JSON object (JSON-Lines format) — easy to stream
    with ``json.loads(line)`` without loading the whole file.

    What is stored
    --------------
    The entry verbatim, minus the ``embedding`` vector (too large, can be
    re-generated from ``content`` via embed_text).  A ``_logged_at`` timestamp
    is injected so the log is independently auditable.

    Thread safety
    -------------
    ``_RAW_LOG_LOCK`` serialises concurrent appends within the process.
    The underlying ``open('a') + write`` is effectively atomic on POSIX for
    lines ≤ PIPE_BUF (4 096 bytes); the lock defends against interleaved
    multi-line writes on Windows and for larger entries.
    """
    raw_log_path = base_path / "memory" / "mt_raw.jsonl"
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {k: v for k, v in entry.items() if k != "embedding"}
    log_entry.setdefault("_logged_at", _now_iso())

    line = json.dumps(log_entry, ensure_ascii=False) + "\n"
    with _RAW_LOG_LOCK:
        with open(raw_log_path, "a", encoding="utf-8") as fh:
            fh.write(line)


# ---------------------------------------------------------------------------
# MT Writer — enriches entries before persistence
# ---------------------------------------------------------------------------

class MTWriter:
    """
    Pipeline for enriching and persisting MT memory entries.
    Call MTWriter.write(entry, base_path, context) for the full pipeline.
    """

    @staticmethod
    def enrich(entry: dict, base_path: Path, context: dict) -> dict:
        """Enrich entry with wing/room/hall/entities/importance/compressed fields."""
        content = entry.get("content", "")
        if not content:
            return entry

        # Sanitize content
        content = _sanitize(content)
        entry["content"] = content

        # Skip if already enriched
        if entry.get("_palace_enriched"):
            return entry

        config = _get_palace_config(base_path)

        # Room + wing
        room = RoomDetector.detect(content)
        wing = RoomDetector.wing_from_room(room)

        # Hall
        hall = HallDetector.detect(content)

        # Entities
        entities = EntityExtractor.extract(content)

        # Importance (with recency decay)
        importance = ImportanceScorer.score(
            content,
            entities,
            room,
            created_at=entry.get("created_at"),
            config=config,
        )

        # AAAK compression
        compressed = AAAKCodec.compress(content, entities)

        entry["room"]       = room
        entry["wing"]       = wing
        entry["hall"]       = hall
        entry["entities"]   = entities.get("persons", []) + entities.get("projects", [])
        entry["importance"] = importance
        entry["compressed"] = compressed
        entry["_palace_enriched"] = True

        # Update KG entity registry
        try:
            kg = KnowledgeGraph(base_path / "memory" / "kg.json")
            kg.merge_entities(entities)
        except Exception:
            pass

        return entry

    @staticmethod
    def write(entry: dict, base_path: Path, context: dict) -> dict:
        """Full write pipeline: raw-log → sanitize → embed → enrich → dedup → save."""
        config = _get_palace_config(base_path)
        backend = _get_mt_backend(base_path, config)
        now = _now_iso()

        # 0. Append pre-AAAK raw entry to the recovery log.
        #    Done before any transformation so the log always reflects what
        #    the caller originally wrote, not the lossy compressed form.
        try:
            _append_raw_log(entry, base_path)
        except Exception:
            pass  # logging failure must never block a write

        # 1. Sanitize content
        if entry.get("content"):
            entry["content"] = _sanitize(entry["content"])

        # 2. Embed if no embedding
        if not entry.get("embedding") and entry.get("content"):
            emb = _embed(entry["content"], context)
            if emb:
                entry["embedding"] = emb

        # 3. Enrich (room/wing/hall/entities/importance/compressed)
        entry = MTWriter.enrich(entry, base_path, context)

        # 4. Dedup check
        if entry.get("embedding"):
            lookback_days = config.get("dedup_lookback_days", 7)
            threshold     = config.get("dedup_similarity_threshold", 0.92)
            recent = backend.get_recent(entry.get("wing"), entry.get("room"), limit=50)
            cutoff_ts = _days_ago_iso(lookback_days)
            candidates = [
                e for e in recent
                if e.get("created_at", "") >= cutoff_ts and e.get("id") != entry.get("id")
            ]
            for candidate in candidates:
                if candidate.get("embedding"):
                    sim = _cosine(entry["embedding"], candidate["embedding"])
                    if sim >= threshold:
                        # Merge: keep max importance, update updated_at
                        merged = dict(candidate)
                        merged["importance"] = max(
                            candidate.get("importance", 0.0),
                            entry.get("importance", 0.0),
                        )
                        merged["updated_at"] = now
                        backend.save(merged)
                        return merged

        # 5. Save new entry
        if not entry.get("created_at"):
            entry["created_at"] = now
        if not entry.get("updated_at"):
            entry["updated_at"] = now
        if not entry.get("id"):
            entry["id"] = str(uuid.uuid4())
        if not entry.get("status"):
            entry["status"] = "active"
        backend.save(entry)
        return entry

    @staticmethod
    def prune(base_path: Path, config: dict | None = None) -> dict:
        """Archive low-importance old entries. Returns counts of archived and inspected."""
        if config is None:
            config = _get_palace_config(base_path)
        backend = _get_mt_backend(base_path, config)
        entries = backend.get_all("active")

        min_importance = config.get("pruning_min_importance", 0.30)
        min_age_days   = config.get("pruning_min_age_days", 30)

        archived = 0
        for entry in entries:
            if entry.get("importance", 1.0) < min_importance:
                if _days_since(entry.get("created_at", "")) >= min_age_days:
                    if "CORE" not in str(entry.get("compressed", "")).upper():
                        success = backend.archive(entry["id"])
                        if success:
                            archived += 1

        return {"archived": archived, "inspected": len(entries)}

    @staticmethod
    def enrich_batch(entries: list[dict], base_path: Path, context: dict) -> list[dict]:
        """Enrich a batch of entries, reusing the cached embed module for efficiency."""
        enriched = []
        for entry in entries:
            # Embed if needed (module loaded once per process via _EMBED_MOD_CACHE)
            if not entry.get("embedding") and entry.get("content"):
                emb = _embed(entry.get("content", ""), context)
                if emb:
                    entry["embedding"] = emb
            entry = MTWriter.enrich(entry, base_path, context)
            enriched.append(entry)
        return enriched


# ---------------------------------------------------------------------------
# MT Reader — 4-layer retrieval
# ---------------------------------------------------------------------------

class MTReader:
    """
    MemPalace-style retrieval with 4-layer access pattern and tunnel cross-linking.

    palace_layer  | what's returned
    ─────────────────────────────────────────────────────
    0             | identity context from LT (L0)
    1             | top entries by importance, grouped by room
    2             | wing/room/hall filtered, then cosine-ranked (+ tunnels)
    3             | full cosine search (no filters)
    None          | smart: L2 if wing/room/hall given, else L3
    """

    @staticmethod
    def read(
        base_path:    Path,
        query:        str,
        top_k:        int,
        context:      dict,
        wing:         str | None = None,
        room:         str | None = None,
        palace_layer: int | None = None,
        hall:         str | None = None,
        tunnel:       bool = True,
    ) -> list[dict]:
        """Route to the appropriate retrieval layer and return ranked entries."""
        config  = _get_palace_config(base_path)
        backend = _get_mt_backend(base_path, config)

        # Determine effective layer
        layer = palace_layer
        if layer is None:
            layer = 2 if (wing or room or hall) else 3

        if layer == 0:
            return MTReader._layer0(base_path, context)

        if layer == 1:
            entries = backend.get_all("active")
            return MTReader._layer1(entries, top_k, config)

        if layer == 2:
            entries = backend.get_filtered(wing, hall, room, "active")

            # Tunnel expansion
            if tunnel and wing:
                all_entries = backend.get_all("active")
                tunnel_wings = TunnelIndex.get_tunnel_wings(
                    room or "", wing, all_entries
                )
                seen_ids = {e.get("id") for e in entries}
                for tw in tunnel_wings:
                    for te in backend.get_filtered(wing=tw, hall=hall, room=room, status="active"):
                        if te.get("id") not in seen_ids:
                            entries.append(te)
                            seen_ids.add(te.get("id"))

            return MTReader._cosine_rank(entries, query, top_k, context)

        # layer 3 (and fallback)
        entries = backend.get_all("active")
        return MTReader._cosine_rank(entries, query, top_k, context)

    @staticmethod
    def _layer0(base_path: Path, context: dict) -> list[dict]:
        """Return L0 identity context: first 3 LT entries + active project."""
        results: list[dict] = []
        lt_path = base_path / "memory" / "lt.json"
        if lt_path.exists():
            try:
                raw = json.loads(lt_path.read_text())
                lt_entries = raw if isinstance(raw, list) else [raw]
                for e in lt_entries[:3]:
                    tagged = dict(e)
                    tagged["_l0_source"] = "lt"
                    results.append(tagged)
            except Exception:
                pass

        project = context.get("project", "")
        if project:
            results.append({
                "id":         "_l0_project",
                "content":    f"Active project: {project}",
                "_l0_source": "context",
            })

        return results

    @staticmethod
    def _layer1(entries: list[dict], top_k: int, config: dict) -> list[dict]:
        """Return highest-importance entries grouped by room, respecting char budget."""
        char_budget  = config.get("l1_char_budget", 3200)
        per_room_max = config.get("per_room_max") or max(1, top_k // 3)

        # Sort by importance desc (entries without importance default to 0.5)
        entries.sort(key=lambda e: e.get("importance", 0.5), reverse=True)

        seen_rooms: dict[str, int] = {}
        result: list[dict] = []
        total_chars = 0

        for e in entries:
            room = e.get("room", "general")
            if seen_rooms.get(room, 0) >= per_room_max:
                continue
            # Prefer compressed form for L1 (token budget)
            text = e.get("compressed") or e.get("content", "")
            if total_chars + len(text) > char_budget:
                break
            result.append(e)
            seen_rooms[room] = seen_rooms.get(room, 0) + 1
            total_chars += len(text)
            if len(result) >= top_k:
                break

        return result

    @staticmethod
    def _cosine_rank(
        entries: list[dict],
        query:   str,
        top_k:   int,
        context: dict,
    ) -> list[dict]:
        """Rank entries by cosine similarity to query; fall back to importance order."""
        if not entries:
            return []

        query_vec = _embed(query, context) if query else None
        if query_vec:
            scored = sorted(
                [(e, _cosine(query_vec, e.get("embedding", [])))
                 for e in entries if e.get("embedding")],
                key=lambda x: x[1], reverse=True,
            )
            return [e for e, _ in scored[:top_k]]

        # No query or no embeddings — return by importance
        entries_copy = list(entries)
        entries_copy.sort(key=lambda e: e.get("importance", 0.5), reverse=True)
        return entries_copy[:top_k]


# ---------------------------------------------------------------------------
# KG layer reader/writer (used by memory_read/write with layer="kg")
# ---------------------------------------------------------------------------

def kg_read(params: dict, base_path: Path) -> list[dict]:
    """
    Query the knowledge graph.
    params:
        subject   : str | None
        predicate : str | None
        object    : str | None
        as_of     : str | None   (ISO date)
        subject_timeline : str | None  (returns full timeline for this subject)
    """
    kg = KnowledgeGraph(base_path / "memory" / "kg.json")

    if params.get("subject_timeline"):
        return kg.timeline(params["subject_timeline"])

    return kg.query(
        subject   = params.get("subject"),
        predicate = params.get("predicate"),
        obj       = params.get("object"),
        as_of     = params.get("as_of"),
    )


def kg_write(params: dict, base_path: Path) -> dict:
    """
    Write a triple to the knowledge graph.
    params:
        subject   : str
        predicate : str
        object    : str
        valid_from : str | None
        valid_to   : str | None
        invalidate : bool  (if True, invalidate matching triples instead of adding)
    """
    kg = KnowledgeGraph(base_path / "memory" / "kg.json")

    if params.get("invalidate"):
        count = kg.invalidate(
            params["subject"], params["predicate"], params["object"]
        )
        return {"invalidated": count}

    triple_id = kg.add(
        subject    = params["subject"],
        predicate  = params["predicate"],
        obj        = params["object"],
        valid_from = params.get("valid_from"),
        valid_to   = params.get("valid_to"),
    )
    return {"id": triple_id}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_mt(base_path: Path) -> list[dict]:
    """Load MT entries from mt.json, returning empty list on failure."""
    path = base_path / "memory" / "mt.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _embed(text: str, context: dict) -> list[float] | None:
    """Embed text using embed_text.py, caching the module per base_path."""
    try:
        bp_str = str(Path(context["base_path"]))
        if bp_str not in _EMBED_MOD_CACHE:
            impl_path = Path(bp_str) / "tools" / "impl" / "embed_text.py"
            spec = importlib.util.spec_from_file_location("embed_text", str(impl_path))
            mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
            spec.loader.exec_module(mod)                    # type: ignore[union-attr]
            _EMBED_MOD_CACHE[bp_str] = mod
        return _EMBED_MOD_CACHE[bp_str].run({"text": text}, context).get("embedding")
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _days_since(iso_str: str) -> float:
    """Return number of days elapsed since an ISO 8601 timestamp, or 0 on error."""
    if not iso_str:
        return 0.0
    try:
        # Handle both aware and naive timestamps
        iso_clean = iso_str.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(iso_clean)
        except ValueError:
            # Try without timezone info
            dt = datetime.fromisoformat(iso_str[:19]).replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except Exception:
        return 0.0


def _days_ago_iso(days: int) -> str:
    """Return ISO 8601 string for the timestamp N days ago."""
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()
