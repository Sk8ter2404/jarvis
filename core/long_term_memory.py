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
  set_reflector_llm(fn)                        -> None   # contradiction pass
  reset_all()                                  -> int    # full wipe (+backup)
  forget_since(cutoff_ts)                      -> dict   # time-window purge
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
import shutil
import threading
import time
import uuid
from typing import Callable, Iterable, Optional

from core.atomic_io import _atomic_write_json
from core import paths as _paths


# ──────────────────────────────────────────────────────────────────────────
#  PATHS / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = _paths.PROJECT_DIR

# Staging-aware root via the canonical chooser (core/paths — the 2026-07-21
# fix for the private-_DATA_DIR bug class): a JARVIS_STAGING process keeps its
# store under data_staging/ and can never touch the live one. That matters
# here specifically because the DESTRUCTIVE maintenance APIs below
# (reset_all / forget_since) are reachable from core.actions directly — the
# monolith's _ltm_enabled staging gate does not cover them. Bound at import
# like the rest of these constants; tests repoint them directly.
_DATA_DIR     = os.path.join(_paths.data_dir(create=False),
                             "long_term_memory")
# The legacy import source lives at the project root on a live box; a staging
# process sees the staged copy instead, mirroring memory.py's redirect.
_LEGACY_BOBERT_MEMORY = (
    os.path.join(_paths.data_dir(create=False), "bobert_memory.json")
    if _paths.is_staging()
    else os.path.join(_PROJECT_DIR, "bobert_memory.json"))

_CHROMA_DIR   = os.path.join(_DATA_DIR, "chroma")
_FACTS_JSON   = os.path.join(_DATA_DIR, "facts.json")        # mirror + BM25 source
_EPISODE_LOG  = os.path.join(_DATA_DIR, "episodes.jsonl")    # per-turn log
_MIGRATE_FLAG = os.path.join(_DATA_DIR, "migrated.flag")

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
# Cap the CONTRADICTION pass's LLM adjudications per reflector run — a large
# 0.6..REFLECTOR_DUP_SIM cohort must not stall the serial ltm-queue worker
# for minutes calling the LLM on every mid-band pair. (2026-07-21 audit #39)
REFLECTOR_MAX_LLM_PAIRS = 20
# Fact sources the contradiction pass treats as ground truth: a fact from one
# of these is never condemned in favour of a survivor from an untrusted
# (ambient-extraction) source — a mis-heard Whisper variant must not delete a
# migrated/backfilled fact. (2026-07-21 audit #39)
_TRUSTED_FACT_SOURCES = {"bobert_memory_migration", "bobert_memory_backfill"}

_lock = threading.RLock()

# Dedicated locks guarding lazy construction of heavyweight singletons.
# Without these, two concurrent retrieve_facts() callers can each build a
# SentenceTransformer (~6 GB transient RAM) and race on _embedder=.
_embedder_lock = threading.Lock()
_chroma_lock   = threading.Lock()
# Serialises every emb.encode() forward pass. Retrieval, upsert and the
# reflector all embed through _embed(); without this two concurrent callers
# ran overlapping torch forward passes on the shared model (extra transient
# VRAM + contention). Held only around the encode itself, never a lazy load.
# (2026-07-08 #15/#28)
_encode_lock   = threading.Lock()


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
            # Honour the LTM_EMBED_DEVICE knob (config/user_settings); "" keeps
            # the historical auto-pick (cuda if present). "cpu" frees ~0.4GB of
            # VRAM for the local LLM at a negligible latency cost. 2026-07-10.
            dev = ""
            try:
                from core import config as _cfg
                dev = (getattr(_cfg, "LTM_EMBED_DEVICE", "") or "").strip().lower()
            except Exception:
                dev = ""
            if not dev:
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
    global _embedder, _embedder_failed_until
    emb = _try_import_embedder()
    if emb is None:
        return None
    # 2026-07-08 (#28): serialise the forward pass so retrieval / upsert / the
    # reflector never run two concurrent encodes on the shared model. Doing this
    # inside _embed makes the discipline automatic — every encode goes through
    # here. The lazy load already happened above, so only the encode is held.
    with _encode_lock:
        try:
            return emb.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            print(f"  [ltm] embed failed: {e}")
            # 2026-07-08 (#27): a transient CUDA/OOM during encode used to be
            # swallowed while leaving the (possibly wedged) model resident, so
            # every later encode failed the same way — recall stayed dead for the
            # whole session. Drop the handle, free VRAM best-effort and arm the
            # reload cooldown so a subsequent call rebuilds a fresh embedder.
            msg = str(e).lower()
            # Match CUDA/OOM signals precisely — a bare "oom" substring would
            # also fire on unrelated words like "boom", so require it as a token.
            if "cuda" in msg or "out of memory" in msg or re.search(r"\boom\b", msg):
                with _embedder_lock:
                    _embedder = None
                    _embedder_failed_until = time.time() + _EMBEDDER_RETRY_COOLDOWN_S
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
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
    upsert_failures = 0
    # Only Chroma-available runs can (or need) confirm dense indexing. When
    # Chroma is absent the JSON mirror + BM25 are authoritative, so a failed
    # upsert is expected and must NOT block the flag.
    chroma_avail = _try_import_chroma() is not None
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
        if not _chroma_upsert(fid, text, entry) and chroma_avail:
            upsert_failures += 1
        migrated += 1
    if migrated:
        _save_facts_locked()
        _rebuild_bm25_locked()
    # 2026-07-08 (#16): only claim migration complete once every fact is
    # confirmed into Chroma. If Chroma is up but some upserts failed (transient
    # embedder/CUDA hiccup), DON'T drop the flag — a later boot re-runs (the
    # exact-text skip above makes re-import idempotent) and _reconcile_chroma_locked
    # back-fills the missing dense vectors. Previously the flag was written
    # unconditionally, permanently dropping those facts from the dense index.
    if upsert_failures:
        print(f"  [ltm] migration deferred: {upsert_failures} chroma upsert(s) "
              f"failed; will retry on next boot")
        return migrated
    try:
        with open(_MIGRATE_FLAG, "w", encoding="utf-8") as f:
            f.write(f"migrated={migrated} ts={int(time.time())}\n")
    except Exception:
        pass
    print(f"  [ltm] migrated {migrated} legacy fact(s) from bobert_memory.json")
    return migrated


def _reconcile_chroma_locked() -> int:
    """Re-upsert any _facts missing from the Chroma collection. Caller holds
    _lock. Runs at boot so facts that only reached the JSON mirror (e.g. a
    migration where Chroma/embedder was transiently down) get their dense
    vectors back-filled once Chroma is available again. Cheap no-op on a healthy
    store (every id already present) and a no-op without Chroma. (2026-07-08 #16)"""
    coll = _try_import_chroma()
    if coll is None or not _facts:
        return 0
    try:
        existing = coll.get(include=[])
        have = set((existing or {}).get("ids") or [])
    except Exception as e:
        print(f"  [ltm] chroma reconcile skipped: {e}")
        return 0
    backfilled = 0
    for fid, entry in list(_facts.items()):
        if fid in have:
            continue
        if _chroma_upsert(fid, entry.get("text", ""), entry):
            backfilled += 1
    if backfilled:
        print(f"  [ltm] chroma back-filled {backfilled} missing fact(s)")
    return backfilled


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
        # 2026-07-08 (#16): back-fill any facts present in the JSON mirror but
        # missing from Chroma (e.g. a prior boot's migration/upsert failed).
        _reconcile_chroma_locked()
        # 2026-07-08 (#30): rotate episodes.jsonl at boot based on ACTUAL line
        # count. The per-process _writes_since_rotate counter resets every boot,
        # so on a frequently-restarted box the in-append check may never reach its
        # threshold and the log would grow unbounded across restarts. A boot-time
        # trim bounds the file regardless of how often the process restarts.
        _rotate_episodes_locked()
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

    # Cheap emptiness gate + a fact-count snapshot, then release the lock.
    with _lock:
        if not _facts:
            return []
        n_facts = len(_facts)

    # 2026-07-08 (#15/#29): prime the heavyweight chroma/embedder singletons and
    # embed the query OUTSIDE _lock. A cold SentenceTransformer load + encode can
    # take several seconds; doing it under _lock stalled record_turn and every
    # other caller behind the whole cold load. The dense chroma query hits its own
    # store and needs no _lock either. Only the brief bm25 / _facts reads + the
    # score blend below run under _lock.
    dense_scores: dict[str, float] = {}
    coll = _try_import_chroma()
    if coll is not None:
        vec = _embed([q])
        if vec is not None:
            try:
                res = coll.query(
                    query_embeddings=[vec[0].tolist()],
                    n_results=min(max(k * 3, 10), max(1, n_facts)),
                    include=["distances", "metadatas"],
                )
                ids = (res.get("ids") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for fid, dist in zip(ids, dists):
                    # cosine distance → similarity, clipped to [0,1]
                    dense_scores[str(fid)] = max(0.0, 1.0 - float(dist))
            except Exception as e:
                print(f"  [ltm] chroma query failed: {e}")

    with _lock:
        if not _facts:
            return []

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


def _rotate_episodes_locked() -> None:
    """Trim episodes.jsonl to the most-recent EPISODE_MAX_LINES lines when it
    exceeds that bound. Caller must hold _lock. Cheap no-op when the file is
    absent or under the bound. Shared by the per-append check AND the boot-time
    trim in ensure_loaded so the log stays bounded even on a box that restarts
    before the in-process append counter ever trips. (2026-07-08 #30)"""
    try:
        if not os.path.exists(_EPISODE_LOG):
            return
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
    # forever, letting episodes.jsonl grow without bound.) The same trim also
    # runs once at boot (#30) so a frequently-restarted box can't outrun this
    # per-process counter. Respects EPISODE_MAX_LINES: keep only the most recent.
    _writes_since_rotate += 1
    if _writes_since_rotate >= EPISODE_ROTATE_CHECK_EVERY:
        _writes_since_rotate = 0
        _rotate_episodes_locked()


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
        # Run reflector outside the lock; it acquires its own. Pass the
        # injected adjudicator so the contradiction pass actually runs in
        # production (2026-07-21 audit #39) — with nothing injected this is
        # llm_call=None and the pass is skipped, exactly the old behavior.
        try:
            reflect_and_consolidate(llm_call=_reflector_llm)
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

# Injected adjudicator for the contradiction pass. record_turn's periodic
# trigger hands this to reflect_and_consolidate(); production wires it from
# the monolith's LTM bridge via set_reflector_llm() (this core module must
# not import bobert_companion). None — the default — skips the contradiction
# pass, which was the only production behavior before the 2026-07-21 audit
# (#39: every caller passed llm_call=None, so the pass was dead code).
_reflector_llm: Optional[Callable[[str, list], Optional[str]]] = None


def set_reflector_llm(fn: Optional[Callable[[str, list], Optional[str]]]) -> None:
    """Install the LLM used by reflect_and_consolidate's contradiction pass.

    ``fn`` has the same contract as reflect_and_consolidate's ``llm_call``
    parameter: fn(prompt, context_msgs) -> ''/'A'/'B'/'MERGE: <text>' (a
    None/''/raising call means "both facts stay"). Pass None to disable."""
    global _reflector_llm
    _reflector_llm = fn


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
        # Source snapshot for the contradiction pass's trusted-source guard —
        # captured with the text snapshot so a concurrent edit can't shift it.
        snapshot_source = dict(zip(ids, ((e.get("source") or "")
                                         for e in items)))
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
    llm_pairs = 0   # contradiction-pass adjudications this run (bounded)
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
                # Bound the per-run LLM work — beyond the cap, remaining
                # mid-band pairs simply wait for a later run instead of
                # stalling the serial ltm-queue worker. (2026-07-21 #39)
                if llm_pairs >= REFLECTOR_MAX_LLM_PAIRS:
                    continue
                llm_pairs += 1
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
                if (verdict.upper().startswith("A")
                        or verdict.upper().startswith("B")):
                    # 'A' keeps the first presented (ids[i]); 'B' the second.
                    if verdict.upper().startswith("A"):
                        survivor, condemned = ids[i], ids[j]
                    else:
                        survivor, condemned = ids[j], ids[i]
                    # Trusted-source guard: never delete a migrated/backfilled
                    # fact in favour of a survivor from an untrusted (ambient-
                    # extraction) source — both stay. (2026-07-21 audit #39)
                    if (snapshot_source.get(condemned, "")
                            in _TRUSTED_FACT_SOURCES
                            and snapshot_source.get(survivor, "")
                            not in _TRUSTED_FACT_SOURCES):
                        continue
                    to_delete.add(condemned)
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
#  PUBLIC API — destructive maintenance (full wipe / time-window purge)
# ──────────────────────────────────────────────────────────────────────────

def reset_all() -> int:
    """Wipe the ENTIRE long-term store: semantic facts (JSON mirror + chroma
    + BM25 index), the episodic turn log, and the in-process working window —
    after snapshotting facts.json / episodes.jsonl into
    _DATA_DIR/backups/pre_reset_<ts>/.

    Mirrors _act_reset_memory's contract: if the backup copy fails, the wipe
    is REFUSED (the exception propagates so the caller can disclose it).
    migrated.flag is deliberately LEFT IN PLACE so _migrate_legacy_locked
    cannot resurrect the wiped facts from bobert_memory.json on the next
    boot. Degrades gracefully without chromadb — the JSON + BM25 + episode
    wipe still succeeds. Returns the number of semantic facts cleared.
    (2026-07-21 audit #17: reset_memory wiped only bobert_memory.json while
    _ltm_context kept injecting the surviving facts every turn.)"""
    global _collection
    ensure_loaded()
    with _lock:
        # Backup FIRST — refuse to wipe anything if the copy fails.
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(_DATA_DIR, "backups", f"pre_reset_{ts}")
        to_copy = [p for p in (_FACTS_JSON, _EPISODE_LOG) if os.path.exists(p)]
        if to_copy:
            os.makedirs(backup_dir, exist_ok=True)
            for src in to_copy:
                shutil.copy2(src, os.path.join(backup_dir,
                                               os.path.basename(src)))

        count = len(_facts)
        fact_ids = list(_facts.keys())
        _facts.clear()
        _save_facts_locked()

        # Episodic log + working window: a wipe that leaves the verbatim turn
        # log (or the turns already in get_working_window() prompt context)
        # is not a wipe.
        try:
            if os.path.exists(_EPISODE_LOG):
                os.remove(_EPISODE_LOG)
        except Exception:
            # Locked by a concurrent reader — truncate in place instead.
            with open(_EPISODE_LOG, "w", encoding="utf-8"):
                pass
        _working.clear()

        # Chroma: drop the whole collection (also clears any orphan vectors)
        # and recreate it fresh. With no client (an injected bare collection
        # handle, e.g. in tests) fall back to per-id deletes. Chroma absent →
        # nothing to do; the JSON/BM25 wipe above is authoritative.
        coll = _try_import_chroma()
        if coll is not None:
            try:
                with _chroma_lock:
                    if _chroma_client is not None:
                        _chroma_client.delete_collection(LTM_COLLECTION)
                        _collection = _chroma_client.get_or_create_collection(
                            name=LTM_COLLECTION,
                            metadata={"hnsw:space": "cosine"},
                        )
                    elif fact_ids:
                        coll.delete(ids=fact_ids)
            except Exception as e:
                print(f"  [ltm] chroma reset failed: {e}")

        _rebuild_bm25_locked()
        return count


def forget_since(cutoff_ts: float) -> dict:
    """Purge every stored trace of conversation recorded at or after
    ``cutoff_ts`` (epoch seconds): in-process working turns, episodic-log
    lines, and semantic facts created inside the window.

    The episode rewrite uses the tmp + os.replace pattern (mirroring
    _rotate_episodes_locked) so a crash can't half-truncate the log.
    Unparseable / ts-less lines and facts are KEPT — legacy entries are
    treated as old, matching _act_forget_last_hour's convention for
    bobert_memory entries. Exceptions propagate: the caller must DISCLOSE a
    failed purge rather than claim success (the silent-survival gap is the
    bug). Returns counts {"episodes": n, "facts": n, "working": n}.
    (2026-07-21 audit #51: forget_last_hour left the hour's verbatim turns
    in episodes.jsonl and its facts in the semantic store.)"""
    counts = {"episodes": 0, "facts": 0, "working": 0}
    ensure_loaded()
    with _lock:
        # (a) In-process working window — purging only the file would leave
        # the turns in get_working_window() prompt context all session.
        kept_working = []
        for entry in _working:
            try:
                ts = float(entry.get("ts", 0.0))
            except (TypeError, ValueError):
                ts = 0.0
            if ts >= cutoff_ts:
                counts["working"] += 1
            else:
                kept_working.append(entry)
        _working[:] = kept_working

        # (b) Episodic log — atomic rewrite keeping only pre-cutoff lines.
        if os.path.exists(_EPISODE_LOG):
            with open(_EPISODE_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
            keep_lines = []
            for line in lines:
                ts = None
                try:
                    ts = float(json.loads(line).get("ts"))
                except Exception:
                    ts = None       # unparseable / ts-less → treated as old
                if ts is not None and ts >= cutoff_ts:
                    counts["episodes"] += 1
                    continue
                keep_lines.append(line)
            if counts["episodes"]:
                tmp = _EPISODE_LOG + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(keep_lines)
                os.replace(tmp, _EPISODE_LOG)

        # (c) Semantic facts created inside the window — drop from the JSON
        # mirror + chroma per id, then one save + BM25 rebuild at the end.
        doomed = []
        for fid, entry in _facts.items():
            try:
                created = float(entry.get("created_at", 0.0))
            except (TypeError, ValueError):
                created = 0.0
            if created >= cutoff_ts:
                doomed.append(fid)
        for fid in doomed:
            del _facts[fid]
            _chroma_delete(fid)
        if doomed:
            _save_facts_locked()
            _rebuild_bm25_locked()
        counts["facts"] = len(doomed)
    return counts


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
