"""
core/long_term_memory.py — Tiered long-term memory for JARVIS.

Replaces the flat-list lookup of bobert_memory.json["facts"] (which gets stuffed
into every prompt and grows without bound) with a Mem0 / Letta-style three-tier
memory system:

  (a) working   — last N conversational turns, held in process. Cheap.
  (b) semantic  — atomic facts about the user / environment, embedded in
                  ChromaDB and additionally indexed by rank_bm25 so retrieval
                  is a hybrid of dense semantic + sparse lexical. Top-k facts
                  are retrieved per turn instead of dumping every fact.
  (c) episodic  — full per-turn conversation log with timestamps. Searchable
                  by time window and free-text topic. Lets JARVIS answer
                  'what did we talk about last Tuesday' without keeping the
                  raw transcript in the prompt.

A self-editing reflector (`reflect_and_consolidate`) runs every N turns or on
shutdown: it scans semantic facts for near-duplicates and obvious staleness
(superseded contradicting facts) and overwrites or removes the loser. This is
the Mem0-style 'update memories with new info instead of appending forever'
property — without it, the store eventually contradicts itself.

First-boot migration
────────────────────
On first call to `ensure_loaded()`, if no semantic collection exists yet but
the legacy bobert_memory.json["facts"] list is present, every fact is imported
as a semantic memory tagged `source="bobert_memory_migration"` and a marker
file at data/long_term_memory/migrated.flag is written so we never re-import.

Graceful degradation
────────────────────
chromadb / sentence-transformers / rank_bm25 are all LAZILY imported. If any
is missing, the corresponding feature degrades:
  - chromadb absent           → semantic dense retrieval disabled,
                                 BM25-only search still works on facts kept
                                 in the JSON sidecar.
  - sentence-transformers     → ditto (no embeddings available)
  - rank_bm25 absent          → hybrid falls back to dense-only when chroma is
                                 there, else returns the most-recent-N facts.

Importing this module NEVER crashes the companion even with no extras.

Public API
──────────
  ensure_loaded()                              -> None    # idempotent boot
  add_fact(text, *, source='', tags=None)      -> str id
  update_fact(fact_id, text)                   -> bool
  delete_fact(fact_id)                         -> bool
  list_facts(limit=None)                       -> list[dict]
  retrieve_facts(query, k=8)                   -> list[dict]    # hybrid
  record_turn(role, text, *, ts=None)          -> None
  get_working_window(n=12)                     -> list[dict]
  search_episodes(query='', start=None,
                  end=None, limit=20)          -> list[dict]
  reflect_and_consolidate(llm_call=None)       -> dict
  is_available()                               -> dict   # per-feature flags
  status()                                     -> dict
  config_summary()                             -> dict
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import threading
import time
import uuid
from typing import Callable, Iterable, Optional

from core.atomic_io import _atomic_write_json


# ──────────────────────────────────────────────────────────────────────────
#  PATHS / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR     = os.path.join(_PROJECT_DIR, "data", "long_term_memory")
_CHROMA_DIR   = os.path.join(_DATA_DIR, "chroma")
_FACTS_JSON   = os.path.join(_DATA_DIR, "facts.json")        # mirror + BM25 source
_EPISODE_LOG  = os.path.join(_DATA_DIR, "episodes.jsonl")    # per-turn log
_MIGRATE_FLAG = os.path.join(_DATA_DIR, "migrated.flag")

_LEGACY_BOBERT_MEMORY = os.path.join(_PROJECT_DIR, "bobert_memory.json")

LTM_COLLECTION    = "jarvis_semantic_facts"
LTM_EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
WORKING_WINDOW    = 24      # turns kept in working memory
EPISODE_MAX_LINES = 50000   # rotate the jsonl when it exceeds this
RETRIEVE_K        = 8
HYBRID_DENSE_W    = 0.65    # dense vs. BM25 score blend (0..1)
HYBRID_SPARSE_W   = 1.0 - HYBRID_DENSE_W

# Reflector tunables.
REFLECTOR_DUP_SIM = 0.92    # cosine sim above which two facts are duplicates
REFLECTOR_RUN_EVERY_TURNS = 50
# Cap the O(n^2) pairwise dedupe so a large fact store doesn't stall the
# reflector for seconds on every run. Above this many facts we only consider
# the most-recently-updated REFLECTOR_MAX_PAIRWISE for the pairwise pass.
REFLECTOR_MAX_PAIRWISE = 400

_lock = threading.RLock()

# Dedicated locks guarding lazy construction of heavyweight singletons.
# Without these, two concurrent retrieve_facts() callers can each build a
# SentenceTransformer (~6 GB transient RAM) and race on _embedder=.
_embedder_lock = threading.Lock()
_chroma_lock   = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────────────────────────────────

# Cached lazy-loaded handles. Each goes through its own try/except so a
# missing dep degrades the feature instead of taking the whole module out.
_chroma_client = None
_collection    = None
_embedder      = None
# After a FAILED embedder load, stand down until this wall-clock time before
# retrying. Without it, a persistent load failure (e.g. the 2026-07-07
# stdout-isatty regression) re-attempted the full ~200-weight model load on EVERY
# embed call — 174 loads in one session, hammering CPU/disk. 0.0 = no cooldown.
_embedder_failed_until = 0.0
_EMBEDDER_RETRY_COOLDOWN_S = 300.0
_bm25_index    = None
_bm25_corpus_ids: list[str] = []
_bm25_corpus:    list[list[str]] = []

# In-memory mirror of the semantic facts list. Keyed by id. Persisted to
# _FACTS_JSON so the BM25 path works even without ChromaDB.
_facts: dict[str, dict] = {}

# Working memory ring buffer. Tuples (role, text, ts).
_working: list[dict] = []
_loaded = False

# Reflector counter — incremented each record_turn. Triggers consolidation
# at REFLECTOR_RUN_EVERY_TURNS.
_turns_since_reflect = 0

# Episodic-rotation counter — incremented on every episode append. Mirrors the
# _writes_since_rotate pattern in skills/pattern_learning.py: count appends and
# only do the (relatively costly) line-count + trim every Nth write, instead of
# gating on a timestamp modulo that may never become true. Guarded by _lock
# (always held by the caller, _append_episode_locked).
EPISODE_ROTATE_CHECK_EVERY = 500
_writes_since_rotate = 0


# ──────────────────────────────────────────────────────────────────────────
#  LAZY DEP PROBES
# ──────────────────────────────────────────────────────────────────────────

def _try_import_chroma():
    global _chroma_client, _collection
    # Fast path: already initialised. Reading a single Python attribute is
    # atomic so the unlocked check is safe.
    if _collection is not None:
        return _collection
    with _chroma_lock:
        # Re-check under the lock: another thread may have built it while we
        # were blocked.
        if _collection is not None:
            return _collection
        try:
            import chromadb
        except Exception:
            return None
        try:
            os.makedirs(_CHROMA_DIR, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)
            _collection = _chroma_client.get_or_create_collection(
                name=LTM_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            return _collection
        except Exception as e:
            print(f"  [ltm] chroma init failed: {e}")
            return None


def _try_import_embedder():
    global _embedder, _embedder_failed_until
    # Fast path: already loaded. Avoids serialising every embed call behind
    # the construction lock.
    if _embedder is not None:
        return _embedder
    import time as _t
    # Backoff: a recent load failure stands down instead of hot-retrying the full
    # model load on every embed call (the churn that stressed the box on
    # 2026-07-07). Cleared implicitly when the cooldown elapses.
    if _embedder_failed_until and _t.time() < _embedder_failed_until:
        return None
    with _embedder_lock:
        # Re-check inside the lock — another thread may have just built the
        # model while we were waiting. Without this guard two concurrent
        # cold callers each instantiate SentenceTransformer (~6 GB transient
        # RAM each) and race on the assignment below.
        if _embedder is not None:
            return _embedder
        if _embedder_failed_until and _t.time() < _embedder_failed_until:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            _embedder_failed_until = _t.time() + _EMBEDDER_RETRY_COOLDOWN_S
            return None
        try:
            dev = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    dev = "cuda"
            except Exception:
                pass
            print(f"  [ltm] loading embedder {LTM_EMBED_MODEL} on {dev}")
            _embedder = SentenceTransformer(LTM_EMBED_MODEL, device=dev)
            return _embedder
        except Exception as e:
            # GPU-first, but degrade to CPU on a cuda OOM / driver hiccup
            # rather than disabling semantic recall entirely. The 3090 can
            # hit its 24 GB ceiling when image-gen (SD ~6 GB) loads while
            # qwen2.5:14b (~10 GB) is resident — don't let that kill recall.
            if dev == "cuda":
                print(f"  [ltm] cuda embedder load failed ({e}); retrying on CPU")
                try:
                    _embedder = SentenceTransformer(LTM_EMBED_MODEL, device="cpu")
                    return _embedder
                except Exception as e2:
                    print(f"  [ltm] CPU embedder load also failed: {e2}")
                    _embedder_failed_until = _t.time() + _EMBEDDER_RETRY_COOLDOWN_S
                    return None
            print(f"  [ltm] embedder load failed: {e}")
            _embedder_failed_until = _t.time() + _EMBEDDER_RETRY_COOLDOWN_S
            return None


def _try_import_bm25():
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
        return True
    except Exception:
        return False


def is_available() -> dict:
    """Per-feature availability flags. Useful for diagnostics and for the
    skill layer to print a single 'pip install …' hint listing only the
    missing pieces."""
    try:
        import chromadb  # noqa: F401
        chroma_ok = True
    except Exception:
        chroma_ok = False
    try:
        import sentence_transformers  # noqa: F401
        embed_ok = True
    except Exception:
        embed_ok = False
    try:
        import rank_bm25  # noqa: F401
        bm25_ok = True
    except Exception:
        bm25_ok = False
    return {
        "chromadb":              chroma_ok,
        "sentence_transformers": embed_ok,
        "rank_bm25":             bm25_ok,
        "fully_available":       chroma_ok and embed_ok and bm25_ok,
    }


# ──────────────────────────────────────────────────────────────────────────
#  PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
    except Exception:
        pass


def _save_facts_locked() -> None:
    """Atomically persist _facts to disk. Caller must hold _lock.

    Uses the shared mkstemp-based helper so concurrent writers can't collide
    on a fixed ``.tmp`` filename — each call gets a unique sibling tempfile.
    """
    _ensure_dirs()
    try:
        _atomic_write_json(_FACTS_JSON, list(_facts.values()))
    except Exception as e:
        print(f"  [ltm] facts save failed: {e}")


def _load_facts_locked() -> None:
    """Repopulate _facts from disk. Caller must hold _lock."""
    _facts.clear()
    if not os.path.exists(_FACTS_JSON):
        return
    try:
        with open(_FACTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ltm] facts load failed: {e}")
        return
    if not isinstance(data, list):
        return
    for entry in data:
        if not isinstance(entry, dict):
            continue
        fid = str(entry.get("id") or "")
        if not fid:
            continue
        entry.setdefault("text", "")
        entry.setdefault("source", "")
        entry.setdefault("tags", [])
        entry.setdefault("created_at", time.time())
        entry.setdefault("updated_at", entry["created_at"])
        _facts[fid] = entry


# ──────────────────────────────────────────────────────────────────────────
#  BM25 INDEX
# ──────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]


def _rebuild_bm25_locked() -> None:
    """Recompute the BM25 corpus index from _facts. Caller must hold _lock."""
    global _bm25_index, _bm25_corpus, _bm25_corpus_ids
    _bm25_corpus_ids = []
    _bm25_corpus = []
    _bm25_index = None
    if not _facts:
        return
    if not _try_import_bm25():
        return
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return
    for fid, entry in _facts.items():
        toks = _tokenize(entry.get("text", ""))
        if not toks:
            continue
        _bm25_corpus_ids.append(fid)
        _bm25_corpus.append(toks)
    if not _bm25_corpus:
        return
    try:
        _bm25_index = BM25Okapi(_bm25_corpus)
    except Exception as e:
        print(f"  [ltm] bm25 build failed: {e}")
        _bm25_index = None


# ──────────────────────────────────────────────────────────────────────────
#  EMBEDDING + CHROMA WRITES
# ──────────────────────────────────────────────────────────────────────────

def _embed(texts: list[str]):
    """Encode `texts`; returns numpy array or None if no embedder."""
    emb = _try_import_embedder()
    if emb is None:
        return None
    try:
        return emb.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as e:
        print(f"  [ltm] embed failed: {e}")
        return None


def _chroma_upsert(fid: str, text: str, meta: dict) -> bool:
    coll = _try_import_chroma()
    if coll is None:
        return False
    vec = _embed([text])
    if vec is None:
        return False
    try:
        # Chroma metadata can't hold nested lists — flatten tags to a CSV.
        safe_meta = dict(meta)
        if isinstance(safe_meta.get("tags"), list):
            safe_meta["tags"] = ",".join(str(t) for t in safe_meta["tags"])
        # upsert: replace any prior chunk for this id.
        try:
            coll.delete(ids=[fid])
        except Exception:
            pass
        coll.add(
            ids=[fid],
            embeddings=vec.tolist(),
            documents=[text],
            metadatas=[safe_meta],
        )
        return True
    except Exception as e:
        print(f"  [ltm] chroma upsert failed: {e}")
        return False


def _chroma_delete(fid: str) -> None:
    coll = _try_import_chroma()
    if coll is None:
        return
    try:
        coll.delete(ids=[fid])
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
#  MIGRATION
# ──────────────────────────────────────────────────────────────────────────

def _migrate_legacy_locked() -> int:
    """Pull facts from bobert_memory.json["facts"] into the new store the
    first time we boot. Returns number of facts migrated."""
    if os.path.exists(_MIGRATE_FLAG):
        return 0
    if not os.path.exists(_LEGACY_BOBERT_MEMORY):
        # No legacy file → still drop the flag so we don't keep looking.
        try:
            _ensure_dirs()
            with open(_MIGRATE_FLAG, "w", encoding="utf-8") as f:
                f.write("no-legacy\n")
        except Exception:
            pass
        return 0
    try:
        with open(_LEGACY_BOBERT_MEMORY, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ltm] legacy load failed: {e}")
        return 0
    migrated = 0
    for raw in data.get("facts", []):
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        # Skip if a fact with identical text is already present (e.g. a
        # partial run got interrupted before the flag was written).
        if any(f.get("text") == text for f in _facts.values()):
            continue
        fid = _new_fact_id(text)
        entry = {
            "id":         fid,
            "text":       text,
            "source":     "bobert_memory_migration",
            "tags":       ["legacy"],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        _facts[fid] = entry
        _chroma_upsert(fid, text, entry)
        migrated += 1
    if migrated:
        _save_facts_locked()
        _rebuild_bm25_locked()
    try:
        with open(_MIGRATE_FLAG, "w", encoding="utf-8") as f:
            f.write(f"migrated={migrated} ts={int(time.time())}\n")
    except Exception:
        pass
    print(f"  [ltm] migrated {migrated} legacy fact(s) from bobert_memory.json")
    return migrated


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API — boot
# ──────────────────────────────────────────────────────────────────────────

def ensure_loaded() -> None:
    """Idempotent boot. Loads JSON mirror, runs first-boot migration, builds
    the BM25 index. Safe to call from anywhere — cheap after first call."""
    global _loaded
    with _lock:
        if _loaded:
            return
        _ensure_dirs()
        _load_facts_locked()
        # Migration runs even when chromadb isn't installed — the JSON
        # mirror + BM25 path still benefits from the imported facts.
        _migrate_legacy_locked()
        _rebuild_bm25_locked()
        _loaded = True


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API — semantic facts
# ──────────────────────────────────────────────────────────────────────────

def _new_fact_id(text: str) -> str:
    """Deterministic-prefix id so two add_fact() calls in the same ms for
    different texts don't collide; the random suffix prevents collisions
    on intentional re-adds."""
    h = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:10]
    return f"fact_{h}_{uuid.uuid4().hex[:6]}"


def add_fact(text: str,
             *,
             source: str = "",
             tags: Optional[Iterable[str]] = None) -> str:
    """Insert a new semantic fact. Returns the new id."""
    if not text or not text.strip():
        raise ValueError("add_fact: empty text")
    text = text.strip()
    ensure_loaded()
    with _lock:
        # De-dupe by exact text — common when a fact extractor re-emits an
        # already-known fact on a later turn.
        for fid, entry in _facts.items():
            if entry.get("text") == text:
                return fid
        fid = _new_fact_id(text)
        now = time.time()
        entry = {
            "id":         fid,
            "text":       text,
            "source":     source,
            "tags":       list(tags or []),
            "created_at": now,
            "updated_at": now,
        }
        _facts[fid] = entry
        _chroma_upsert(fid, text, entry)
        _save_facts_locked()
        _rebuild_bm25_locked()
        return fid


def update_fact(fact_id: str, text: str) -> bool:
    """Overwrite an existing fact's text (used by the reflector when it
    finds a more correct or newer version of an existing memory)."""
    if not text or not text.strip():
        return False
    text = text.strip()
    ensure_loaded()
    with _lock:
        entry = _facts.get(fact_id)
        if entry is None:
            return False
        entry["text"] = text
        entry["updated_at"] = time.time()
        _chroma_upsert(fact_id, text, entry)
        _save_facts_locked()
        _rebuild_bm25_locked()
        return True


def delete_fact(fact_id: str) -> bool:
    ensure_loaded()
    with _lock:
        if fact_id not in _facts:
            return False
        del _facts[fact_id]
        _chroma_delete(fact_id)
        _save_facts_locked()
        _rebuild_bm25_locked()
        return True


def list_facts(limit: Optional[int] = None) -> list[dict]:
    ensure_loaded()
    with _lock:
        items = list(_facts.values())
    items.sort(key=lambda e: e.get("updated_at", 0.0), reverse=True)
    if limit is not None:
        items = items[:limit]
    return [dict(e) for e in items]


def retrieve_facts(query: str, k: int = RETRIEVE_K) -> list[dict]:
    """Hybrid retrieval: dense (chroma) + sparse (bm25). Each path
    contributes a normalised score in [0,1]; final rank is a weighted
    blend. If chroma is unavailable, falls back to BM25-only. If both are
    unavailable, returns the most recently updated facts (still useful —
    gives the LLM *something* relevant-ish in the prompt)."""
    ensure_loaded()
    if not query or not query.strip():
        return list_facts(limit=k)
    q = query.strip()
    with _lock:
        if not _facts:
            return []

        # ── dense
        dense_scores: dict[str, float] = {}
        coll = _try_import_chroma()
        if coll is not None:
            vec = _embed([q])
            if vec is not None:
                try:
                    res = coll.query(
                        query_embeddings=[vec[0].tolist()],
                        n_results=min(max(k * 3, 10), max(1, len(_facts))),
                        include=["distances", "metadatas"],
                    )
                    ids = (res.get("ids") or [[]])[0]
                    dists = (res.get("distances") or [[]])[0]
                    for fid, dist in zip(ids, dists):
                        # cosine distance → similarity, clipped to [0,1]
                        dense_scores[str(fid)] = max(0.0, 1.0 - float(dist))
                except Exception as e:
                    print(f"  [ltm] chroma query failed: {e}")

        # ── sparse (bm25)
        sparse_scores: dict[str, float] = {}
        if _bm25_index is not None and _bm25_corpus_ids:
            try:
                toks = _tokenize(q)
                raw = _bm25_index.get_scores(toks)
                # Normalise to [0,1] by the max so blending is meaningful.
                m = max(raw) if len(raw) else 0.0
                if m > 0:
                    for fid, score in zip(_bm25_corpus_ids, raw):
                        sparse_scores[fid] = float(score) / float(m)
            except Exception as e:
                print(f"  [ltm] bm25 score failed: {e}")

        # ── blend
        if not dense_scores and not sparse_scores:
            return list_facts(limit=k)
        # Use whichever weights still apply if one side is missing.
        if not dense_scores:
            blended = {fid: s for fid, s in sparse_scores.items()}
        elif not sparse_scores:
            blended = {fid: s for fid, s in dense_scores.items()}
        else:
            blended = {}
            for fid in set(dense_scores) | set(sparse_scores):
                d = dense_scores.get(fid, 0.0)
                s = sparse_scores.get(fid, 0.0)
                blended[fid] = HYBRID_DENSE_W * d + HYBRID_SPARSE_W * s

        ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
        out: list[dict] = []
        for fid, score in ranked[:k]:
            entry = _facts.get(fid)
            if entry is None:
                continue
            row = dict(entry)
            row["score"] = round(float(score), 4)
            out.append(row)
        return out


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API — working memory (last N turns, in-process)
# ──────────────────────────────────────────────────────────────────────────

def get_working_window(n: int = WORKING_WINDOW) -> list[dict]:
    """Last N turns from working memory. Cheap; in-process only."""
    ensure_loaded()
    with _lock:
        if n <= 0:
            return []
        return [dict(t) for t in _working[-n:]]


def _append_episode_locked(entry: dict) -> None:
    """Persist one turn to the episodic jsonl. Caller must hold _lock."""
    global _writes_since_rotate
    _ensure_dirs()
    line = json.dumps(entry, ensure_ascii=False)
    try:
        with open(_EPISODE_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"  [ltm] episode write failed: {e}")
        return
    # Light rotation — count appends and only every Nth write do we pay for the
    # line-count + trim. (The old gate, int(ts) % 97 == 0, could stay false
    # forever, letting episodes.jsonl grow without bound.) Respects the
    # EPISODE_MAX_LINES bound: keep only the most recent N lines.
    _writes_since_rotate += 1
    if _writes_since_rotate >= EPISODE_ROTATE_CHECK_EVERY:
        _writes_since_rotate = 0
        try:
            with open(_EPISODE_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > EPISODE_MAX_LINES:
                keep = lines[-EPISODE_MAX_LINES:]
                tmp = _EPISODE_LOG + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(keep)
                os.replace(tmp, _EPISODE_LOG)
        except Exception:
            pass


def record_turn(role: str, text: str, *, ts: Optional[float] = None) -> None:
    """Record one conversational turn. Pushes into both working memory and
    the episodic log. Empty texts and known wake-only utterances are
    dropped at the call site (mirroring memory.record_voice_command)."""
    global _turns_since_reflect
    if not text or not text.strip():
        return
    role = (role or "user").strip().lower() or "user"
    ts = ts or time.time()
    lt = time.localtime(ts)
    entry = {
        "ts":   ts,
        "iso":  time.strftime("%Y-%m-%dT%H:%M:%S", lt),
        "date": time.strftime("%Y-%m-%d", lt),
        "day":  time.strftime("%A", lt),
        "hour": lt.tm_hour,
        "role": role,
        "text": text.strip()[:2000],
    }
    ensure_loaded()
    with _lock:
        _working.append(entry)
        if len(_working) > WORKING_WINDOW * 4:
            # Trim well beyond the read window so callers asking for larger
            # windows still find a few extra turns of head-room.
            del _working[: len(_working) - WORKING_WINDOW * 4]
        _append_episode_locked(entry)
        _turns_since_reflect += 1
        should_reflect = _turns_since_reflect >= REFLECTOR_RUN_EVERY_TURNS
    if should_reflect:
        # Run reflector outside the lock; it acquires its own.
        try:
            reflect_and_consolidate()
        except Exception as e:
            print(f"  [ltm] reflector raised: {e}")
        with _lock:
            _turns_since_reflect = 0


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API — episodic search
# ──────────────────────────────────────────────────────────────────────────

def _iter_episodes() -> Iterable[dict]:
    if not os.path.exists(_EPISODE_LOG):
        return
    try:
        with open(_EPISODE_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return


def search_episodes(query: str = "",
                    start: Optional[_dt.date] = None,
                    end:   Optional[_dt.date] = None,
                    limit: int = 20) -> list[dict]:
    """Return matching episodic turns, newest-first. `query` does a
    case-insensitive substring match on the turn text; pass '' for
    pure time-window search."""
    ensure_loaded()
    q = (query or "").strip().lower()
    out: list[dict] = []
    for entry in _iter_episodes():
        date_str = entry.get("date", "")
        try:
            d = _dt.date.fromisoformat(date_str) if date_str else None
        except Exception:
            d = None
        if start is not None and (d is None or d < start):
            continue
        if end is not None and (d is None or d > end):
            continue
        if q and q not in (entry.get("text", "") or "").lower():
            continue
        out.append(entry)
    out.sort(key=lambda e: e.get("ts", 0.0), reverse=True)
    return out[:limit]


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API — self-editing reflector
# ──────────────────────────────────────────────────────────────────────────

def _cosine_sim(a, b) -> float:
    """Cosine similarity on two pre-normalised numpy vectors. Returns 0
    if either is None."""
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np
        return float(np.dot(a, b))
    except Exception:
        return 0.0


def reflect_and_consolidate(
    llm_call: Optional[Callable[[str, list[dict]], Optional[str]]] = None,
) -> dict:
    """Scan semantic facts for near-duplicates and obvious contradictions.

    Near-duplicate pass: for each unordered pair (a, b) whose embeddings
    cosine-similarity exceeds REFLECTOR_DUP_SIM, the older fact is
    deleted (its information is presumed captured by the newer one).

    Contradiction pass: if an `llm_call` callable is provided, every
    pair with cosine-sim in [0.6, REFLECTOR_DUP_SIM) is passed to it
    along with the small context window. The llm_call should return:
      - ''      → both facts stay
      - 'A'     → keep A, delete B
      - 'B'     → keep B, delete A
      - 'MERGE: <new text>' → replace BOTH with one new fact
    A None llm_call simply skips the contradiction pass.

    Returns a small summary dict counting actions taken.
    """
    ensure_loaded()
    summary = {"checked_pairs": 0, "duplicates_removed": 0,
               "contradictions_resolved": 0, "merged": 0}

    with _lock:
        # Stable snapshot: capture (id, text) pairs together so a concurrent
        # add_fact/update_fact/delete_fact can't shift what an id maps to
        # mid-run. We delete by stable key (id) and re-verify the text is
        # unchanged before deleting, so a concurrent edit can't cause the
        # wrong fact to be removed.
        items = sorted(
            _facts.values(),
            key=lambda e: e.get("updated_at", 0.0),
            reverse=True,
        )
        # Cap the pairwise work: above the threshold, only the most-recently-
        # updated REFLECTOR_MAX_PAIRWISE facts participate, bounding the
        # O(n^2) scan instead of letting it grow with the whole store.
        if len(items) > REFLECTOR_MAX_PAIRWISE:
            items = items[:REFLECTOR_MAX_PAIRWISE]
        ids = [str(e.get("id") or "") for e in items]
        texts = [e.get("text", "") for e in items]
        snapshot_text = dict(zip(ids, texts))
        if len(ids) < 2:
            return summary

    vecs = _embed(texts)
    if vecs is None:
        # No embedder → skip semantic dedupe but still do exact-text dedupe.
        with _lock:
            seen: dict[str, str] = {}
            for fid, entry in list(_facts.items()):
                t = entry.get("text", "")
                if t in seen:
                    older, newer = (
                        (seen[t], fid) if _facts[seen[t]]["created_at"] <=
                                          entry["created_at"]
                        else (fid, seen[t])
                    )
                    # Delete the older duplicate.
                    if older in _facts:
                        del _facts[older]
                        _chroma_delete(older)
                        summary["duplicates_removed"] += 1
                    seen[t] = newer
                else:
                    seen[t] = fid
            if summary["duplicates_removed"]:
                _save_facts_locked()
                _rebuild_bm25_locked()
        return summary

    # ── pairwise scan
    to_delete: set[str] = set()
    n = len(ids)
    for i in range(n):
        if ids[i] in to_delete:
            continue
        for j in range(i + 1, n):
            # 2026-07-07 bug-hunt (LOW-MED): once ids[i] itself has been marked
            # for deletion (by an earlier j in this same inner loop), STOP — a
            # doomed fact must not keep matching and dragging OTHER facts into
            # to_delete just because they're similar to it (cascade over-delete
            # of a fact that's only a near-dup of the already-condemned one).
            if ids[i] in to_delete:
                break
            if ids[j] in to_delete:
                continue
            sim = _cosine_sim(vecs[i], vecs[j])
            summary["checked_pairs"] += 1
            if sim >= REFLECTOR_DUP_SIM:
                # near-dup: drop the older one
                a, b = ids[i], ids[j]
                _before = len(to_delete)
                with _lock:
                    if (_facts.get(a, {}).get("created_at", 0) <=
                            _facts.get(b, {}).get("created_at", 0)):
                        to_delete.add(a)
                    else:
                        to_delete.add(b)
                # Count ACTUAL new deletions, not qualifying pairs — a fact
                # already condemned must not inflate the tally.
                if len(to_delete) > _before:
                    summary["duplicates_removed"] += 1
                continue
            if 0.6 <= sim < REFLECTOR_DUP_SIM and llm_call is not None:
                a_text = texts[i]
                b_text = texts[j]
                try:
                    verdict = (llm_call(
                        "Two facts about the user. Are they contradictory? "
                        "Reply 'A' to keep the first, 'B' to keep the second, "
                        "'MERGE: <new text>' to fuse them, or '' if both stay.",
                        [{"role": "fact_a", "text": a_text},
                         {"role": "fact_b", "text": b_text}],
                    ) or "").strip()
                except Exception as e:
                    print(f"  [ltm] reflector llm raised: {e}")
                    verdict = ""
                if verdict.upper().startswith("A"):
                    to_delete.add(ids[j])
                    summary["contradictions_resolved"] += 1
                elif verdict.upper().startswith("B"):
                    to_delete.add(ids[i])
                    summary["contradictions_resolved"] += 1
                elif verdict.upper().startswith("MERGE"):
                    merged = verdict.split(":", 1)[-1].strip()
                    if merged:
                        with _lock:
                            entry = _facts.get(ids[i])
                            # Only merge if the survivor still holds the text we
                            # reasoned about — a concurrent update_fact could
                            # have changed it out from under us.
                            if (entry is not None and
                                    entry.get("text", "") ==
                                    snapshot_text.get(ids[i])):
                                entry["text"] = merged
                                entry["updated_at"] = time.time()
                                _chroma_upsert(ids[i], merged, entry)
                                texts[i] = merged
                                snapshot_text[ids[i]] = merged
                                to_delete.add(ids[j])
                                summary["merged"] += 1

    if to_delete:
        with _lock:
            for fid in to_delete:
                entry = _facts.get(fid)
                # Stable-key delete with a content guard: only remove the fact
                # if it still matches what the pairwise pass compared. This
                # prevents a concurrent add_fact/update_fact (which can change
                # what an id maps to) from causing the wrong fact to be deleted.
                if entry is not None and \
                        entry.get("text", "") == snapshot_text.get(fid):
                    del _facts[fid]
                    _chroma_delete(fid)
            _save_facts_locked()
            _rebuild_bm25_locked()
    return summary


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API — diagnostics
# ──────────────────────────────────────────────────────────────────────────

def status() -> dict:
    avail = is_available()
    with _lock:
        episodes = 0
        try:
            if os.path.exists(_EPISODE_LOG):
                with open(_EPISODE_LOG, "r", encoding="utf-8") as f:
                    for _ in f:
                        episodes += 1
        except Exception:
            pass
        return {
            "available":   avail,
            "loaded":      _loaded,
            "facts":       len(_facts),
            "working":     len(_working),
            "episodes":    episodes,
            "chroma_dir":  _CHROMA_DIR,
            "facts_path":  _FACTS_JSON,
            "episode_log": _EPISODE_LOG,
            "migrated":    os.path.exists(_MIGRATE_FLAG),
        }


def config_summary() -> dict:
    return {
        "LTM_COLLECTION":             LTM_COLLECTION,
        "LTM_EMBED_MODEL":            LTM_EMBED_MODEL,
        "WORKING_WINDOW":             WORKING_WINDOW,
        "EPISODE_MAX_LINES":          EPISODE_MAX_LINES,
        "RETRIEVE_K":                 RETRIEVE_K,
        "HYBRID_DENSE_W":             HYBRID_DENSE_W,
        "HYBRID_SPARSE_W":            HYBRID_SPARSE_W,
        "REFLECTOR_DUP_SIM":          REFLECTOR_DUP_SIM,
        "REFLECTOR_RUN_EVERY_TURNS":  REFLECTOR_RUN_EVERY_TURNS,
    }


# ──────────────────────────────────────────────────────────────────────────
#  Offline smoke test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    ensure_loaded()
    print("availability:", is_available())
    print("status:", json.dumps(status(), indent=2, default=str))
    fid = add_fact("User likes Michael Jackson", source="smoke_test")
    print("added:", fid)
    print("retrieve('what music does the user like'):",
          retrieve_facts("what music does the user like", k=3))
    record_turn("user", "play michael jackson")
    record_turn("assistant", "Queueing Michael Jackson, sir.")
    print("working:", get_working_window(4))
    print("reflect:", reflect_and_consolidate())
