"""
mt_palace.py
------------
MemPalace-inspired storage and retrieval engine for the MT (medium-term) memory layer.

What it borrows from MemPalace:
  - Wing / Room hierarchy   : every entry is tagged with a domain (wing) and subtopic (room)
  - AAAK lossy compression  : entity codes + topic keywords + key sentence + emotions + flags
  - Entity extraction       : persons and projects detected from plain text
  - Importance scoring      : drives the L1 "essential story" selection
  - Knowledge Graph (KG)    : temporal triple store in memory/kg.json
  - 4-layer retrieval       :
        L0  identity / LT preferences (~100 tokens always present)
        L1  highest-importance drawers grouped by room (500–800 token budget)
        L2  wing/room-filtered cosine search (on-demand, ~200–500 tokens)
        L3  full semantic search across all MT (unrestricted)

What it does NOT borrow:
  - ChromaDB / sentence-transformers (we use embed_text.py which supports openai/ollama/hash)
  - SQLite (KG stored as JSON with atomic writes)
  - CLI tooling (that's handled by main.py / init.py)

The existing mt.json schema is extended, not replaced.  Old entries without palace
fields still participate in L3 search.  New entries get all fields.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    "decisions":  ["decided", "chose", "chose", "approach", "trade-off", "tradeoff",
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
        return _WING_FROM_ROOM.get(room, "knowledge")


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
        """
        Returns:
          persons  : list[str]
          projects : list[str]
          codes    : dict[str → 3-char-code]
        """
        # Find capitalized sequences (proper nouns)
        candidates: dict[str, int] = {}
        for m in re.finditer(r"\b([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+)*)\b", text):
            word = m.group(1)
            # Skip all-caps acronyms under 3 chars
            if len(word) < 3:
                continue
            candidates[word] = candidates.get(word, 0) + 1

        # Filter to words appearing ≥ 2 times
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
    """Score an entry 0–1 for use in L1 layer selection."""

    @staticmethod
    def score(text: str, entities: dict[str, Any], room: str) -> float:
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

        return min(round(score, 3), 1.0)


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Temporal triple store backed by memory/kg.json.

    Schema:
      triples : list[{id, subject, predicate, object, valid_from, valid_to, created_at}]
      entities: dict[name → {type, code, aliases}]
    """

    def __init__(self, kg_path: Path) -> None:
        self._path = kg_path

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {"triples": [], "entities": {}}

    def _save(self, data: dict) -> None:
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

    def add(
        self,
        subject:    str,
        predicate:  str,
        obj:        str,
        valid_from: str | None = None,
        valid_to:   str | None = None,
    ) -> str:
        """Add a triple. Returns its id."""
        data = self._load()
        triple = {
            "id":         str(uuid.uuid4()),
            "subject":    subject,
            "predicate":  predicate,
            "object":     obj,
            "valid_from": valid_from,
            "valid_to":   valid_to,
            "created_at": _now_iso(),
        }
        data["triples"].append(triple)
        self._save(data)
        return triple["id"]

    def query(
        self,
        subject:   str | None = None,
        predicate: str | None = None,
        obj:       str | None = None,
        as_of:     str | None = None,     # ISO date string
    ) -> list[dict]:
        """Return triples matching filters. as_of filters by temporal validity."""
        data    = self._load()
        results = []
        for t in data["triples"]:
            if subject   and t["subject"].lower()   != subject.lower():   continue
            if predicate and t["predicate"].lower()  != predicate.lower(): continue
            if obj       and t["object"].lower()     != obj.lower():       continue
            if as_of:
                if t.get("valid_from") and t["valid_from"] > as_of:       continue
                if t.get("valid_to")   and t["valid_to"]   < as_of:       continue
            results.append(t)
        return results

    def invalidate(self, subject: str, predicate: str, obj: str) -> int:
        """Mark matching triples as no longer valid (set valid_to = now). Returns count."""
        data  = self._load()
        today = _now_iso()[:10]
        count = 0
        for t in data["triples"]:
            if (t["subject"].lower()   == subject.lower()   and
                    t["predicate"].lower() == predicate.lower() and
                    t["object"].lower()    == obj.lower()        and
                    t.get("valid_to") is None):
                t["valid_to"] = today
                count += 1
        if count:
            self._save(data)
        return count

    def merge_entities(self, entities: dict[str, Any]) -> None:
        """Upsert detected entities into the entity registry."""
        data = self._load()
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
# MT Writer — enriches entries before persistence
# ---------------------------------------------------------------------------

class MTWriter:
    """
    Call MTWriter.enrich(entry, base_path, context) before saving to mt.json.
    Adds: wing, room, entities, importance, compressed.
    """

    @staticmethod
    def enrich(entry: dict, base_path: Path, context: dict) -> dict:
        content = entry.get("content", "")
        if not content:
            return entry

        # Skip if already enriched
        if entry.get("_palace_enriched"):
            return entry

        # Room + wing
        room = RoomDetector.detect(content)
        wing = RoomDetector.wing_from_room(room)

        # Entities
        entities = EntityExtractor.extract(content)

        # Importance
        importance = ImportanceScorer.score(content, entities, room)

        # AAAK compression
        compressed = AAAKCodec.compress(content, entities)

        entry["room"]       = room
        entry["wing"]       = wing
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


# ---------------------------------------------------------------------------
# MT Reader — 4-layer retrieval
# ---------------------------------------------------------------------------

_L1_CHAR_BUDGET = 3200   # max chars across all L1 results


class MTReader:
    """
    MemPalace-style retrieval from mt.json.

    palace_layer  │ what's returned
    ─────────────────────────────────────────────────────
    0             │ identity context from LT (caller handles this)
    1             │ top entries by importance, grouped by room
    2             │ wing/room filtered, then cosine-ranked
    3             │ full cosine search (no filters)
    None          │ smart: L2 if wing/room given, else L3
    """

    @staticmethod
    def read(
        base_path:     Path,
        query:         str,
        top_k:         int,
        context:       dict,
        wing:          str | None = None,
        room:          str | None = None,
        palace_layer:  int | None = None,
    ) -> list[dict]:
        entries = _load_mt(base_path)

        # Determine effective layer
        layer = palace_layer
        if layer is None:
            layer = 2 if (wing or room) else 3

        if layer == 1:
            return MTReader._layer1(entries, top_k)
        if layer == 2:
            return MTReader._layer2(entries, query, top_k, context, wing, room)
        # layer 3 (and fallback)
        return MTReader._layer3(entries, query, top_k, context)

    @staticmethod
    def _layer1(entries: list[dict], top_k: int) -> list[dict]:
        """
        Essential story: highest-importance entries, deduplicated by room,
        respecting the char budget.
        """
        active = [e for e in entries if e.get("status", "active") == "active"]
        # Sort by importance desc (entries without importance default to 0.5)
        active.sort(key=lambda e: e.get("importance", 0.5), reverse=True)

        seen_rooms: dict[str, int] = {}
        result: list[dict] = []
        total_chars = 0
        per_room_max = max(1, top_k // 3)  # spread across rooms

        for e in active:
            room = e.get("room", "general")
            if seen_rooms.get(room, 0) >= per_room_max:
                continue
            # Prefer compressed form for L1 (token budget)
            text = e.get("compressed") or e.get("content", "")
            if total_chars + len(text) > _L1_CHAR_BUDGET:
                break
            result.append(e)
            seen_rooms[room] = seen_rooms.get(room, 0) + 1
            total_chars += len(text)
            if len(result) >= top_k:
                break

        return result

    @staticmethod
    def _layer2(
        entries:  list[dict],
        query:    str,
        top_k:    int,
        context:  dict,
        wing:     str | None,
        room:     str | None,
    ) -> list[dict]:
        """Wing/room filtered, then cosine-ranked within the subset."""
        active = [e for e in entries if e.get("status", "active") == "active"]

        # Filter
        if wing:
            active = [e for e in active if e.get("wing", "").lower() == wing.lower()]
        if room:
            active = [e for e in active if e.get("room", "").lower() == room.lower()]

        if not active:
            return []

        # Cosine rank
        query_vec = _embed(query, context) if query else None
        if query_vec:
            scored = sorted(
                [(e, _cosine(query_vec, e.get("embedding", [])))
                 for e in active if e.get("embedding")],
                key=lambda x: x[1], reverse=True,
            )
            return [e for e, _ in scored[:top_k]]

        # No query — return by importance
        active.sort(key=lambda e: e.get("importance", 0.5), reverse=True)
        return active[:top_k]

    @staticmethod
    def _layer3(
        entries:  list[dict],
        query:    str,
        top_k:    int,
        context:  dict,
    ) -> list[dict]:
        """Full semantic search across all active MT entries."""
        active = [e for e in entries if e.get("status", "active") == "active"]
        if not active:
            return []

        query_vec = _embed(query, context) if query else None
        if query_vec:
            scored = sorted(
                [(e, _cosine(query_vec, e.get("embedding", [])))
                 for e in active if e.get("embedding")],
                key=lambda x: x[1], reverse=True,
            )
            return [e for e, _ in scored[:top_k]]

        # No query + no embeddings — return by importance
        active.sort(key=lambda e: e.get("importance", 0.5), reverse=True)
        return active[:top_k]


# ---------------------------------------------------------------------------
# KG layer reader (used by memory_read with layer="kg")
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
    path = base_path / "memory" / "mt.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _embed(text: str, context: dict) -> list[float] | None:
    try:
        base_path = Path(context["base_path"])
        impl_path = base_path / "tools" / "impl" / "embed_text.py"
        spec = importlib.util.spec_from_file_location("embed_text", str(impl_path))
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        return mod.run({"text": text}, context).get("embedding")
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
