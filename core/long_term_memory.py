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

Embedder switch (MEMORY_EMBED_MODEL, 2026-10-02)
────────────────────────────────────────────────
core/config.MEMORY_EMBED_MODEL picks the embedder ("BAAI/bge-small-en-v1.5"
= the default and the index exactly as before; "voyage-4-nano" = opt-in).
The live index is bound to the model that built it and is never searched or
written with another; a different setting is rebuilt from the stored fact
texts on a background thread and swapped in only when complete, with the old
index kept as a .bak. A model that cannot load falls back to bge-small with
one log line. See the SAFE RE-INDEX section.

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
# The reflector's MERGE text is a new fact written by a model: it gets the
# same write-time guards merge_memory applies to every learned fact.
from core.memory_guards import (
    _is_secret_fact, _is_internal_noise_fact, MAX_FACT_LEN,
)


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

# ──────────────────────────────────────────────────────────────────────────
#  EMBEDDER PROFILES — the MEMORY_EMBED_MODEL switch (2026-10-02)
# ──────────────────────────────────────────────────────────────────────────
# One entry per model the semantic index may be built with. The default,
# bge-small, is loaded and called EXACTLY as before the switch existed: no
# prompts, no extra load kwargs, the legacy collection, no manifest file.
#
# voyage-4-nano won the 2026-10-02 memory A/B on the owner's own 236 facts
# (the right fact ranked first 49/50 vs 39/50 for bge-small; +33 ms per
# recall on the CPU; rerankers only made it slower and worse). Its entry is
# that A/B's loading recipe, unchanged: sentence-transformers with
# trust_remote_code (one ~100-line bidirectional-Qwen3 file, read and benign
# -- pinned to the revision that was read, so an upstream push cannot swap
# the code), fp32 on the CPU (the config asks for bf16, 2-7x slower on this
# AVX2-only CPU), MRL-truncated to 512-d (scored the same as 2048-d), and the
# model card's query / document prompts prepended as plain text, then
# L2-normalised by encode() after the truncation.
#
# Vectors from two models are not comparable, so each index is BOUND to the
# profile that built it (the SAFE RE-INDEX section below): a vector is only
# ever written to, or searched against, the collection of its own profile.
_DEFAULT_EMBED_PROFILE = "bge-small"
_EMBED_PROFILES: dict[str, dict] = {
    "bge-small": {
        "model":        LTM_EMBED_MODEL,
        "dim":          384,
        "collection":   LTM_COLLECTION,
        "query_prefix": "",
        "doc_prefix":   "",
        "load_kwargs":  {},
        "fp32":         False,
        # "" = the LTM_EMBED_DEVICE knob, else auto (cuda if present), as ever.
        "device":       "",
        # REFLECTOR_DUP_SIM (0.92) was tuned on this model's cosine range.
        "reflector_calibrated": True,
    },
    "voyage-4-nano": {
        "model":        "voyageai/voyage-4-nano",
        "dim":          512,
        "collection":   LTM_COLLECTION + "__voyage-4-nano-512",
        "query_prefix": ("Represent the query for retrieving supporting "
                         "documents: "),
        "doc_prefix":   "Represent the document for retrieval: ",
        "load_kwargs":  {
            "revision":          "67fabc9bef010dabc5f6024aa1b1b6b93410426f",
            "trust_remote_code": True,
            "truncate_dim":      512,
        },
        "fp32":         True,
        # The A/B ran it on the CPU: no VRAM taken from the local brain. An
        # explicit LTM_EMBED_DEVICE still wins.
        "device":       "cpu",
        # Its cosine range was never measured against REFLECTOR_DUP_SIM, and
        # that threshold DELETES facts: until it is calibrated, the reflector
        # removes exact duplicates only on this model.
        "reflector_calibrated": False,
    },
}
# Accepted spellings of MEMORY_EMBED_MODEL (compared lower-cased).
_EMBED_PROFILE_ALIASES = {
    "":                        "bge-small",
    "bge-small":               "bge-small",
    "bge-small-en-v1.5":       "bge-small",
    "baai/bge-small-en-v1.5":  "bge-small",
    "voyage":                  "voyage-4-nano",
    "voyage-4-nano":           "voyage-4-nano",
    "voyageai/voyage-4-nano":  "voyage-4-nano",
}
# Index binding of a collection whose stored stamp names another model than
# the manifest says: nothing embeds for it, so it is never read or written.
_UNBOUND_PROFILE = "unbound"

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
# Leaf lock (2026-10-02 review): pairs installing a freshly loaded embedder
# (_install_embedder) with an index swap's rebinding, so a load that was in
# flight when the swap landed can never install the stale model. Nothing
# else is ever acquired while it is held.
_binding_lock  = threading.Lock()


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

# The live index binding (2026-10-02, MEMORY_EMBED_MODEL): which profile
# built the collection being served, its Chroma name, its dimension (None =
# unknown: the legacy, manifest-less index, checked by nothing new), and the
# profile _embedder was loaded for. ensure_loaded() reads them from
# embed_index.json; without that file they stay these defaults, i.e. today's
# bge-small index. Only _swap_in_index_locked() changes them at runtime.
_index_profile_key = _DEFAULT_EMBED_PROFILE
_collection_name   = LTM_COLLECTION
_index_dim: Optional[int] = None
_embedder_key      = _DEFAULT_EMBED_PROFILE
# Bumped by every index swap. retrieve_facts() reads it before it takes the
# collection and embeds the query, and drops the dense half if it moved: a
# query vector from one model can then never search the other's index.
_binding_gen = 0
# Set (to the profile key) once that profile's model failed to load in this
# process: MEMORY_EMBED_MODEL then resolves to the default for the session.
_embed_fallback_from: Optional[str] = None
_embed_notes_printed: set = set()   # one-time log lines already printed

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
    global _chroma_client, _collection, _index_profile_key
    # Fast path: already initialised. Reading a single Python attribute is
    # atomic so the unlocked check is safe.
    if _collection is not None:
        return _collection
    if _index_profile_key == _UNBOUND_PROFILE:
        return None     # refused below; a background rebuild replaces it
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
            coll = _chroma_client.get_or_create_collection(
                name=_collection_name,
                metadata=_collection_metadata(_index_profile_key),
            )
            # 2026-10-02: a collection stamped by another model than the one
            # the manifest binds it to is never read or written with this
            # one (only a hand-edited store gets here).
            stamped = _stamped_model(coll)
            want = (_EMBED_PROFILES.get(_index_profile_key) or {}).get("model")
            if stamped and want and stamped != want:
                print(f"  [ltm] index {_collection_name} holds {stamped} "
                      f"vectors, not {want}; semantic recall is off until "
                      f"it is rebuilt")
                _index_profile_key = _UNBOUND_PROFILE
                return None
            _collection = coll
            return _collection
        except Exception as e:
            print(f"  [ltm] chroma init failed: {e}")
            return None


def _try_import_embedder():
    global _embedder_failed_until
    # Fast path: already loaded. Avoids serialising every embed call behind
    # the construction lock.
    if _embedder is not None:
        # 2026-10-02: the loaded model must be the one that built the live
        # index; anything else would mix two models' vectors in one index.
        return _embedder if _embedder_key == _index_profile_key else None
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
            return _embedder if _embedder_key == _index_profile_key else None
        if _embedder_failed_until and _t.time() < _embedder_failed_until:
            return None
        # The model pull draws huggingface_hub's "You are sending
        # unauthenticated requests to the HF Hub" server warning on every
        # load; drop ONLY that message (core/log_filters.py).
        try:
            from core.log_filters import install_hf_unauthenticated_filter
            install_hf_unauthenticated_filter()
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            _embedder_failed_until = _t.time() + _EMBEDDER_RETRY_COOLDOWN_S
            return None
        if _index_profile_key != _DEFAULT_EMBED_PROFILE:
            # An index built by another MEMORY_EMBED_MODEL profile (2026-10-02).
            # The default profile keeps the untouched path below.
            return _load_bound_profile_embedder_locked()
        try:
            # Honour the LTM_EMBED_DEVICE knob (config/user_settings); "" keeps
            # the historical auto-pick (cuda if present). "cpu" frees ~0.4GB of
            # VRAM for the local LLM at a negligible latency cost. 2026-07-10.
            # One resolver for every profile since 2026-10-02 (_embed_device;
            # it never raises).
            dev = _embed_device(_DEFAULT_EMBED_PROFILE)
            print(f"  [ltm] loading embedder {LTM_EMBED_MODEL} on {dev}")
            # Installed only while the live index is still this profile's
            # (always, with default settings): _install_embedder.
            return _install_embedder(
                SentenceTransformer(LTM_EMBED_MODEL, device=dev),
                _DEFAULT_EMBED_PROFILE)
        except Exception as e:
            # GPU-first, but degrade to CPU on a cuda OOM / driver hiccup
            # rather than disabling semantic recall entirely. The 3090 can
            # hit its 24 GB ceiling when image-gen (SD ~6 GB) loads while
            # qwen2.5:14b (~10 GB) is resident — don't let that kill recall.
            if dev == "cuda":
                print(f"  [ltm] cuda embedder load failed ({e}); retrying on CPU")
                try:
                    return _install_embedder(
                        SentenceTransformer(LTM_EMBED_MODEL, device="cpu"),
                        _DEFAULT_EMBED_PROFILE)
                except Exception as e2:
                    print(f"  [ltm] CPU embedder load also failed: {e2}")
                    _embedder_failed_until = _t.time() + _EMBEDDER_RETRY_COOLDOWN_S
                    return None
            print(f"  [ltm] embedder load failed: {e}")
            _embedder_failed_until = _t.time() + _EMBEDDER_RETRY_COOLDOWN_S
            return None


def _try_import_bm25():
    try:
        from rank_bm25 import BM25Okapi
        # Availability probe only. pyflakes does not read `# noqa`, so drop the
        # name instead — the import itself (and any error it raises) is
        # unchanged (2026-10-02, CI lints core/).
        del BM25Okapi
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
#  EMBEDDER PROFILE HELPERS (MEMORY_EMBED_MODEL, 2026-10-02)
# ──────────────────────────────────────────────────────────────────────────

def _note_embed_once(tag: str, line: str) -> None:
    """Print `line` the first time `tag` is seen in this process."""
    if tag in _embed_notes_printed:
        return
    _embed_notes_printed.add(tag)
    print(line)


def _configured_embed_model() -> str:
    """MEMORY_EMBED_MODEL from core.config (user_settings.json overridable),
    read at call time; '' when unset or unreadable."""
    try:
        from core import config as _cfg
        return str(getattr(_cfg, "MEMORY_EMBED_MODEL", "") or "")
    except Exception:
        return ""


def _desired_profile_key() -> str:
    """The profile the index SHOULD be built with: MEMORY_EMBED_MODEL, or
    the default when it is unknown or its model already failed to load in
    this process (each case logs one line, once)."""
    raw = _configured_embed_model().strip()
    key = _EMBED_PROFILE_ALIASES.get(raw.lower())
    if key is None:
        _note_embed_once(
            "unknown:" + raw,
            f"  [ltm] MEMORY_EMBED_MODEL {raw!r} is not a known memory "
            f"embedder ({', '.join(sorted(_EMBED_PROFILES))}); using "
            f"{LTM_EMBED_MODEL}")
        key = _DEFAULT_EMBED_PROFILE
    if key == _embed_fallback_from:
        key = _DEFAULT_EMBED_PROFILE
    return key


def _note_embed_fallback(key: str, why: str) -> None:
    """`key`'s model cannot load: use the default profile for the rest of
    the session, and say so in ONE clear line."""
    global _embed_fallback_from
    if key == _DEFAULT_EMBED_PROFILE:
        return              # nothing to fall back to; the caller logs it
    _embed_fallback_from = key
    model = (_EMBED_PROFILES.get(key) or {}).get("model") or key
    _note_embed_once(
        "fallback:" + key,
        f"  [ltm] memory embedder {model} could not load ({why[:200]}); "
        f"falling back to {LTM_EMBED_MODEL} for this session")


def _embed_device(key: str) -> str:
    """Torch device for `key`: the LTM_EMBED_DEVICE knob when set, else the
    profile's own pick, else the historical auto-pick (cuda if present)."""
    dev = ""
    try:
        from core import config as _cfg
        dev = (getattr(_cfg, "LTM_EMBED_DEVICE", "") or "").strip().lower()
    except Exception:
        dev = ""
    if not dev:
        dev = (_EMBED_PROFILES.get(key) or {}).get("device") or ""
    if not dev:
        dev = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                dev = "cuda"
        except Exception:
            pass
    return dev


def _load_profile_model(key: str, dev: str):
    """Construct one profile's SentenceTransformer (raises on failure). The
    default profile is SentenceTransformer(model, device=dev), as ever."""
    from sentence_transformers import SentenceTransformer
    prof = _EMBED_PROFILES[key]
    kwargs = dict(prof.get("load_kwargs") or {})
    if prof.get("fp32"):
        import torch
        kwargs["model_kwargs"] = {"dtype": torch.float32}
    return SentenceTransformer(prof["model"], device=dev, **kwargs)


def _load_bound_profile_embedder_locked():
    """Load the embedder of a NON-default index binding. Caller holds
    _embedder_lock. A model that cannot load never gets a stand-in -- that
    would mix two models' vectors in one index: the session falls back to
    the default profile (one log line) and a background rebuild moves the
    index onto it; dense recall is off until that completes."""
    global _embedder_failed_until
    key = _index_profile_key
    prof = _EMBED_PROFILES.get(key)
    if prof is None:
        if key != _UNBOUND_PROFILE:
            _note_embed_fallback(key, "not a known memory embedder")
            _maybe_start_reindex()
        return None
    dev = _embed_device(key)
    try:
        print(f"  [ltm] loading embedder {prof['model']} on {dev}")
        model = _load_profile_model(key, dev)
    except Exception as e:
        _embedder_failed_until = time.time() + _EMBEDDER_RETRY_COOLDOWN_S
        _note_embed_fallback(key, f"{type(e).__name__}: {e}")
        _maybe_start_reindex()
        return None
    return _install_embedder(model, key)


def _install_embedder(model, key: str):
    """Make `model`, just loaded for profile `key`, the live embedder -- unless
    a background index swap moved the live index to another profile while it
    loaded (2026-10-02 review). Installing it anyway bound the stale model
    under its own key: the key check then refused it on every later call and
    dense recall stayed off for the rest of the session. The swap rebinds
    under the same lock, so whichever lands second sees the other. Returns
    the live embedder, or None when none fits the live index."""
    global _embedder, _embedder_key
    with _binding_lock:
        if key == _index_profile_key:
            _embedder = model
            _embedder_key = key
            return model
        live, live_key = _embedder, _embedder_key
        now_key = _index_profile_key
    print(f"  [ltm] embedder for {key} discarded: the index moved to "
          f"{now_key} while it loaded")
    return live if (live is not None and live_key == now_key) else None


def _collection_metadata(key: str, *, stamp: bool = False) -> dict:
    """Chroma metadata for a new collection of profile `key`. The legacy
    default index keeps its exact historical metadata unless `stamp` (the
    re-indexer stamps everything it builds); every other profile is always
    stamped with its model and dimension."""
    meta = {"hnsw:space": "cosine"}
    prof = _EMBED_PROFILES.get(key)
    if prof is not None and (stamp or key != _DEFAULT_EMBED_PROFILE):
        meta["embed_model"] = prof["model"]
        meta["embed_dim"] = int(prof["dim"])
    return meta


def _stamped_model(coll) -> str:
    """The embed_model stamp on a Chroma collection ('' = unstamped)."""
    try:
        meta = getattr(coll, "metadata", None) or {}
        return str(meta.get("embed_model") or "") if isinstance(meta, dict) else ""
    except Exception:
        return ""


def _query_text(q: str) -> str:
    """A retrieval query as the live index's model expects it."""
    return (_EMBED_PROFILES.get(_index_profile_key) or {}).get(
        "query_prefix", "") + q


def _doc_text(t: str) -> str:
    """A stored fact as the live index's model expects it (the Chroma
    document itself always stays the raw fact text)."""
    return (_EMBED_PROFILES.get(_index_profile_key) or {}).get(
        "doc_prefix", "") + t


def _dims_ok(vec) -> bool:
    """False when the live index's dimension is known and `vec` (one row)
    does not have it. The legacy manifest-less index has no recorded
    dimension, so nothing about it changes."""
    if not _index_dim:
        return True
    try:
        return len(vec) == _index_dim
    except Exception:
        return False


def _reflector_semantic_ok() -> bool:
    """Whether the reflector's similarity passes may run on the live index's
    model -- REFLECTOR_DUP_SIM DELETES facts, so only on a calibrated one."""
    prof = _EMBED_PROFILES.get(_index_profile_key) or {}
    if prof.get("reflector_calibrated"):
        return True
    _note_embed_once(
        "reflector:" + str(_index_profile_key),
        f"  [ltm] reflector: similarity passes are off on "
        f"{prof.get('model') or _index_profile_key} until REFLECTOR_DUP_SIM "
        f"is calibrated for it; exact duplicates are still removed")
    return False


def is_available() -> dict:
    """Per-feature availability flags. Useful for diagnostics and for the
    skill layer to print a single 'pip install …' hint listing only the
    missing pieces."""
    # Import probes: each name is dropped right after its import so pyflakes
    # (which ignores `# noqa`) sees no unused import; the probe is unchanged.
    try:
        import chromadb
        del chromadb
        chroma_ok = True
    except Exception:
        chroma_ok = False
    try:
        import sentence_transformers
        del sentence_transformers
        embed_ok = True
    except Exception:
        embed_ok = False
    try:
        import rank_bm25
        del rank_bm25
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


def _chroma_meta(meta: dict) -> dict:
    """Chroma metadata can't hold nested lists — flatten tags to a CSV."""
    safe_meta = dict(meta)
    if isinstance(safe_meta.get("tags"), list):
        safe_meta["tags"] = ",".join(str(t) for t in safe_meta["tags"])
    return safe_meta


def _chroma_upsert(fid: str, text: str, meta: dict) -> bool:
    coll = _try_import_chroma()
    if coll is None:
        return False
    vec = _embed([_doc_text(text)])
    if vec is None:
        return False
    if _index_dim and not _dims_ok(vec[0]):
        print(f"  [ltm] chroma upsert refused: the vector does not fit the "
              f"{_index_dim}-d index")
        return False
    try:
        safe_meta = _chroma_meta(meta)
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
#  SAFE RE-INDEX — switching MEMORY_EMBED_MODEL (2026-10-02)
# ──────────────────────────────────────────────────────────────────────────
# The rules:
#   * ONE binding is live: _collection + _index_profile_key + _embedder (for
#     that profile). Every live write and every query goes through it, so a
#     query is always embedded by the model that built the index it searches.
#   * A different MEMORY_EMBED_MODEL never touches the live binding. A daemon
#     thread builds the new profile's OWN collection from the stored fact
#     texts (facts.json is the source of truth), with its own model instance,
#     in small batches that each wait for a gap in the conversation
#     (core/local_traffic) -- never holding _lock while it embeds, so no voice
#     turn waits on it. Wall-clock budget _REINDEX_BUDGET_S; on any failure
#     the old index simply stays live and the next start tries again.
#   * Facts added / edited / removed meanwhile land in the old index as
#     usual; the builder re-diffs the store against what it has built until
#     a check under _lock finds nothing left, and swaps IN THAT SAME HOLD.
#   * The swap writes embed_index.json first (the commit point), then labels
#     the retired collection "<name>.bak". Nothing is ever deleted: older
#     .bak collections, and anything that had to be moved out of the way,
#     stay in the store and in the manifest's "backups" list.
#   * A model that cannot load falls back to the default profile with one log
#     line (_note_embed_fallback) -- it is never replaced by another model on
#     the same index.
_EMBED_INDEX_FILE = "embed_index.json"
_REINDEX_BATCH = 8            # facts per encode
_REINDEX_PAUSE_S = 0.05       # breather between batches
_REINDEX_BUDGET_S = 1800.0    # the build once the model is loaded, wall clock
_REINDEX_MAX_ROUNDS = 20      # re-diff passes before giving up
_reindex_guard = threading.Lock()
_reindex_thread: Optional[threading.Thread] = None
_reindex_state: dict = {"state": "idle"}


def _embed_index_path() -> str:
    # Derived from _DATA_DIR at call time, so a repointed store (staging,
    # tests) carries its manifest with it.
    return os.path.join(_DATA_DIR, _EMBED_INDEX_FILE)


def _read_embed_index() -> Optional[dict]:
    """The index manifest, or None when absent (the legacy default index)
    or unreadable. Never raises."""
    path = _embed_index_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        act = data.get("active") if isinstance(data, dict) else None
        if (isinstance(act, dict) and isinstance(act.get("profile"), str)
                and isinstance(act.get("collection"), str)
                and act["collection"]):
            return data
        raise ValueError("no active index recorded")
    except Exception as e:
        _note_embed_once("manifest", f"  [ltm] {_EMBED_INDEX_FILE} unreadable "
                         f"({type(e).__name__}); using the default index")
        return None


def _resolve_index_binding_locked() -> None:
    """Bind the live index to what embed_index.json records. Caller holds
    _lock; runs first in ensure_loaded(), before anything opens Chroma.
    Without the file this leaves the defaults: today's bge-small index."""
    global _index_profile_key, _collection_name, _index_dim
    man = _read_embed_index()
    if man is None:
        return
    act = man["active"]
    _index_profile_key = act["profile"]
    _collection_name = act["collection"]
    dim = act.get("dim")
    _index_dim = dim if isinstance(dim, int) and dim > 0 else None


def _set_reindex_state(state: str, target: str = "", note: str = "") -> None:
    global _reindex_state
    _reindex_state = {"state": state, "target": target, "note": note,
                      "ts": time.time()}


def _maybe_start_reindex() -> bool:
    """Start the background rebuild when the wanted profile differs from the
    one that built the live index. Cheap, never blocks, never raises; a
    no-op with default settings, without a Chroma client, before the store
    is loaded, or while a rebuild is already running."""
    global _reindex_thread
    try:
        if not _loaded:
            return False        # ensure_loaded() calls back once it is
        if _desired_profile_key() == _index_profile_key:
            return False
        if _chroma_client is None:
            _try_import_chroma()
        if _chroma_client is None:
            return False        # no real store: nothing to rebuild
        with _reindex_guard:
            if _reindex_thread is not None and _reindex_thread.is_alive():
                return False
            t = threading.Thread(target=_reindex_worker, name="ltm-reindex",
                                 daemon=True)
            _reindex_thread = t
            t.start()
        return True
    except Exception as e:
        print(f"  [ltm] memory index rebuild not started: {e}")
        return False


def _reindex_worker() -> None:
    """Body of the ltm-reindex thread: converge the live index onto the
    wanted profile. A target whose model fails to load becomes the default
    (fallback) and the loop runs once more for it."""
    try:
        for _ in range(3):
            target = _desired_profile_key()
            if target == _index_profile_key:
                return
            if _run_reindex(target):
                return
            if _desired_profile_key() == target:
                return          # not a load failure: the next start retries
    except Exception as e:
        _set_reindex_state("failed", note=type(e).__name__)
        print(f"  [ltm] memory index rebuild stopped ({type(e).__name__}: "
              f"{e}); the current index stays live")


def _chroma_collection_names(client) -> set:
    names = set()
    for c in (client.list_collections() or []):
        names.add(str(getattr(c, "name", c)))
    return names


def _free_bak_name(base: str, taken: set) -> str:
    """'<base>.bak', or a timestamped variant when that is taken (an older
    backup is never overwritten)."""
    cand = base + ".bak"
    if cand not in taken:
        return cand
    stem = base + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
    cand, n = stem, 2
    while cand in taken:
        cand, n = f"{stem}-{n}", n + 1
    return cand


def _open_reindex_target(client, key: str):
    """(collection, {fid: text already in it}, [moved-aside entries]) for
    profile `key`'s collection. A partial build of the same model (an
    interrupted earlier run) is completed rather than redone; a collection
    by that name holding anything else is renamed aside, never emptied."""
    prof = _EMBED_PROFILES[key]
    name = prof["collection"]
    moved = []
    coll = None
    names = _chroma_collection_names(client)
    if name in names:
        coll = client.get_collection(name)
        meta = getattr(coll, "metadata", None) or {}
        stamped = _stamped_model(coll)
        try:
            stamped_dim = int(meta.get("embed_dim") or 0)
        except Exception:
            stamped_dim = 0
        same = (stamped == prof["model"] and stamped_dim == prof["dim"])
        legacy_default = (not stamped and key == _DEFAULT_EMBED_PROFILE
                          and name == LTM_COLLECTION)
        if not (same or legacy_default):
            bak = _free_bak_name(name, names)
            coll.modify(name=bak)
            moved.append({"profile": "", "model": stamped or "unknown",
                          "collection": bak, "retired_at": time.time()})
            coll = None
    if coll is None:
        coll = client.get_or_create_collection(
            name=name, metadata=_collection_metadata(key, stamp=True))
    got = coll.get(include=["documents"]) or {}
    have = {str(i): (d or "") for i, d in
            zip(got.get("ids") or [], got.get("documents") or [])}
    return coll, have, moved


def _reindex_encode(model, prof: dict, texts: list):
    """Embed one batch of fact texts with the builder's OWN model, waiting
    first for a gap in the conversation (bounded by the gate's own caps).
    Raises if a vector does not have the profile's dimension."""
    def _go():
        return model.encode(
            [prof["doc_prefix"] + t for t in texts],
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False)
    try:
        from core import local_traffic as _lt
    except Exception:
        _lt = None
    if _lt is None:
        vecs = _go()
    else:
        with _lt.background_work("ltm-reindex"):
            with _lt.slot():
                vecs = _go()
    if len(vecs) != len(texts) or any(len(v) != prof["dim"] for v in vecs):
        raise ValueError(f"{prof['model']} returned vectors that are not "
                         f"{prof['dim']}-d")
    return vecs


def _run_reindex(key: str) -> bool:
    """Build profile `key`'s index from the stored fact texts and swap it in.
    True once swapped; False (old index still live) on any failure."""
    prof = _EMBED_PROFILES.get(key)
    client = _chroma_client
    if prof is None or client is None:
        return False
    cur = (_EMBED_PROFILES.get(_index_profile_key) or {}).get(
        "model") or _index_profile_key
    t0 = time.monotonic()
    _set_reindex_state("loading", key)
    try:
        from core.log_filters import install_hf_unauthenticated_filter
        install_hf_unauthenticated_filter()
    except Exception:
        pass
    try:
        dev = _embed_device(key)
        print(f"  [ltm] memory index: building a {prof['model']} index on "
              f"{dev} in the background ({cur} stays live until it is done)")
        model = _load_profile_model(key, dev)
    except Exception as e:
        _set_reindex_state("failed", key, "model load")
        if key == _DEFAULT_EMBED_PROFILE:
            print(f"  [ltm] memory index rebuild: {prof['model']} could not "
                  f"load ({type(e).__name__}: {e}); the current index stays")
        else:
            _note_embed_fallback(key, f"{type(e).__name__}: {e}")
        return False
    # The build budget starts once the model is loaded (2026-10-02 review):
    # the first load downloads ~0.7 GB, and a budget started before it was
    # spent on the download, so a slow link threw away a finished download
    # the moment the first batch checked the clock.
    deadline = time.monotonic() + _REINDEX_BUDGET_S
    try:
        coll, have, moved = _open_reindex_target(client, key)
    except Exception as e:
        _set_reindex_state("failed", key, "open collection")
        print(f"  [ltm] memory index rebuild: could not open "
              f"{prof['collection']} ({type(e).__name__}: {e}); {cur} stays")
        return False
    _set_reindex_state("building", key)
    for _round in range(_REINDEX_MAX_ROUNDS):
        with _lock:
            want = {fid: (e.get("text") or "") for fid, e in _facts.items()}
            put = [fid for fid, t in want.items() if have.get(fid) != t]
            drop = [fid for fid in have if fid not in want]
            if not put and not drop:
                return _swap_in_index_locked(key, coll, model, len(want),
                                             t0, moved)
            metas = {fid: _chroma_meta(_facts[fid]) for fid in put}
        try:
            if drop:
                coll.delete(ids=drop)
                for fid in drop:
                    have.pop(fid, None)
            for i in range(0, len(put), _REINDEX_BATCH):
                if time.monotonic() > deadline:
                    _set_reindex_state("failed", key, "time budget")
                    print(f"  [ltm] memory index rebuild ran past "
                          f"{int(_REINDEX_BUDGET_S)} s; {cur} stays live, "
                          f"the next start resumes it")
                    return False
                batch = put[i:i + _REINDEX_BATCH]
                texts = [want[fid] for fid in batch]
                vecs = _reindex_encode(model, prof, texts)
                coll.delete(ids=batch)
                coll.add(ids=batch, embeddings=vecs.tolist(),
                         documents=texts,
                         metadatas=[metas[fid] for fid in batch])
                for fid, t in zip(batch, texts):
                    have[fid] = t
                time.sleep(_REINDEX_PAUSE_S)
        except Exception as e:
            _set_reindex_state("failed", key, type(e).__name__)
            print(f"  [ltm] memory index rebuild failed ({type(e).__name__}: "
                  f"{e}); {cur} stays live")
            return False
    _set_reindex_state("failed", key, "did not settle")
    print(f"  [ltm] memory index rebuild did not settle in "
          f"{_REINDEX_MAX_ROUNDS} passes; {cur} stays live")
    return False


def _swap_in_index_locked(key: str, coll, model, n_facts: int, t0: float,
                          moved: list) -> bool:
    """Make the finished `key` index the live one. Caller holds _lock, so no
    write can land between the builder's last check and this swap."""
    global _collection, _collection_name, _index_profile_key, _index_dim
    global _embedder, _embedder_key, _embedder_failed_until, _binding_gen
    prof = _EMBED_PROFILES[key]
    old_coll, old_name, old_key = _collection, _collection_name, _index_profile_key
    old_model = (_EMBED_PROFILES.get(old_key) or {}).get("model") or old_key
    man = _read_embed_index() or {}
    # The collection going live is no longer a backup, whatever it was.
    backups = [b for b in (man.get("backups") or []) if isinstance(b, dict)
               and b.get("collection") != prof["collection"]]
    backups += moved
    now = time.time()
    active = {"profile": key, "model": prof["model"], "dim": prof["dim"],
              "collection": prof["collection"], "facts": n_facts,
              "built_at": now}
    retire = old_coll is not None and old_name != prof["collection"]
    if retire:
        backups.append({"profile": old_key, "model": old_model,
                        "collection": old_name, "retired_at": now})
    try:
        _ensure_dirs()
        _atomic_write_json(_embed_index_path(),
                           {"active": active, "backups": backups})
    except Exception as e:
        _set_reindex_state("failed", key, "manifest write")
        print(f"  [ltm] memory index rebuild: could not record it ({e}); "
              f"{old_model} stays live")
        return False
    # Under _binding_lock: an embedder load in flight for the old binding
    # then cannot install itself over this one (_install_embedder).
    with _binding_lock:
        _collection = coll
        _collection_name = prof["collection"]
        _index_profile_key = key
        _index_dim = int(prof["dim"])
        _embedder = model
        _embedder_key = key
        _embedder_failed_until = 0.0
        _binding_gen += 1
    note = ""
    if retire:
        # Cosmetic, after the commit point: label the retired index. If the
        # rename fails it is still kept, under its old name.
        try:
            bak = _free_bak_name(old_name,
                                 _chroma_collection_names(_chroma_client))
            old_coll.modify(name=bak)
            backups[-1]["collection"] = bak
            _atomic_write_json(_embed_index_path(),
                               {"active": active, "backups": backups})
            note = f"; the previous index is kept as {bak}"
        except Exception as e:
            note = (f"; the previous index is kept as {old_name} (not "
                    f"renamed: {type(e).__name__})")
    _set_reindex_state("done", key)
    print(f"  [ltm] memory index rebuilt with {prof['model']} ({n_facts} "
          f"facts, {prof['dim']}-d) in {time.monotonic() - t0:.1f} s{note}")
    return True


def _purge_from_backups(ids: list) -> None:
    """Remove `ids` from every backup collection the manifest lists. Used by
    forget_since(): a fact the owner asked to forget must not live on in a
    retired index. Best effort, never raises."""
    if not ids or _chroma_client is None:
        return
    man = _read_embed_index() or {}
    for b in (man.get("backups") or []):
        name = b.get("collection") if isinstance(b, dict) else None
        if not name:
            continue
        try:
            _chroma_client.get_collection(name).delete(ids=list(ids))
        except Exception as e:
            print(f"  [ltm] backup index {name}: purge failed "
                  f"({type(e).__name__})")


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
        # 2026-10-02: bind the live index (embed_index.json) BEFORE anything
        # below opens Chroma or embeds; without the file nothing changes.
        _resolve_index_binding_locked()
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
        # 2026-10-02: a MEMORY_EMBED_MODEL other than the index's own starts
        # the background rebuild (a no-op with default settings).
        _maybe_start_reindex()


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
    # Read BEFORE the collection and the query vector are taken (2026-10-02):
    # if a background rebuild swaps the index in between, the two may come
    # from different models, so the dense half sits this query out.
    gen = _binding_gen
    coll = _try_import_chroma()
    if coll is not None:
        vec = _embed([_query_text(q)])
        if vec is not None and (gen != _binding_gen
                                or (_index_dim and not _dims_ok(vec[0]))):
            vec = None
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
    parameter: fn(prompt, context_msgs) -> 'A'/'B'/'KEEP'/'MERGE: <text>'
    (anything else, a None/'' or a raising call means "both facts stay").
    Pass None to disable."""
    global _reflector_llm
    _reflector_llm = fn


# Injected sink for the contradiction pass's decisions (2026-10-01). The
# reflector settles contradictions in THIS store only, while the monolith's
# bobert_memory.json facts -- rendered in full into every system prompt --
# kept both sides of a contradiction it had "resolved". The monolith wires a
# sink that applies each decision there too. Contract:
#   fn([(kind, removed_texts, replacement_text_or_None), ...])
# in the order applied: ("contradiction", (condemned,), final survivor text
# or None) and ("merge", (rewritten, merged_away), merged text). The KIND
# lets the sink tell a model's A/B verdict, which may only REMOVE a text
# whose survivor it already holds, from a merge rewrite (2026-10-01). Called
# once per run, outside _lock, with only the decisions actually applied
# here. Near-duplicate removals are not reported: the legacy store dedupes
# those itself. A raising sink never breaks the run.
_reflector_sink: Optional[Callable[[list], None]] = None


def set_reflector_sink(fn: Optional[Callable[[list], None]]) -> None:
    """Install (or, with None, remove) the reflector decision sink."""
    global _reflector_sink
    _reflector_sink = fn


def _may_condemn(condemned_source: str, survivor_source: str) -> bool:
    """THE trusted-source rule, shared by every reflector deletion path: a
    migrated / backfilled fact is never deleted in favour of a fact from an
    untrusted (ambient-extraction) source -- a mis-heard Whisper variant must
    not delete it (2026-07-21 audit #39). One helper so the near-duplicate
    (with or without an embedder), contradiction and MERGE paths cannot
    drift apart again: until 2026-10-01 only the contradiction path applied
    it, and the others deleted 5 of the 23 migration-day facts."""
    return not (condemned_source in _TRUSTED_FACT_SOURCES
                and survivor_source not in _TRUSTED_FACT_SOURCES)


# A local model answering through the JARVIS persona tacks ", sir." onto a
# bare verdict; strip that before reading the verdict or the merged text.
# Comma-led only, so a fact that really ends in the word ("...named Sir")
# keeps it.
_SIR_TAIL_RE = re.compile(r"\s*[,;]\s*sir\b[\s.!]*$", re.IGNORECASE)
_MERGE_RE = re.compile(r"^\W*MERGE\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)


def _parse_reflector_verdict(reply) -> tuple[str, str]:
    """('A' | 'B' | 'MERGE' | 'KEEP', merged_text) for one adjudicator reply.

    STRICT (2026-10-01): the old prefix test read 'Both stay' (the prompt's
    own words), 'Because...' or 'Based on...' as 'B' and deleted fact A, and
    'As far as I can tell...' or 'Agreed' as 'A' and deleted fact B. Only a
    bare A / B (quotes, markdown, a trailing period or ', sir' allowed) or
    'MERGE: <text>' acts; everything else -- KEEP, '', chatter -- keeps
    both facts."""
    raw = (reply or "").strip() if isinstance(reply, str) else ""
    m = _MERGE_RE.match(raw)
    if m:
        merged = _SIR_TAIL_RE.sub("", m.group(1)).strip().strip("'\"`*").strip()
        return ("MERGE", merged) if merged else ("KEEP", "")
    core = _SIR_TAIL_RE.sub("", raw).strip().strip("'\"`*. ").upper()
    if core in ("A", "B"):
        return core, ""
    return "KEEP", ""


def _merged_text_ok(merged: str, a_text: str, b_text: str) -> bool:
    """A MERGE reply becomes a stored fact, so it gets merge_memory's write
    guards (no credential, no internal noise, no runaway length), and it may
    not grow past its two inputs: survivors absorbing fact after fact in one
    run grew into blobs until the token cap cut them mid-word."""
    if not merged or len(merged) > MAX_FACT_LEN:
        return False
    if _is_secret_fact(merged) or _is_internal_noise_fact(merged):
        return False
    return len(merged) <= len(a_text or "") + len(b_text or "") + 40


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
      - 'KEEP' / '' / anything unparsed → both facts stay
      - 'A'     → keep A, delete B
      - 'B'     → keep B, delete A
      - 'MERGE: <new text>' → replace BOTH with one new fact
    (see _parse_reflector_verdict). A None llm_call simply skips the
    contradiction pass. No path deletes or rewrites a trusted fact in
    favour of an untrusted one (_may_condemn).

    Applied contradiction / MERGE decisions are reported to the installed
    sink (set_reflector_sink). Each deletion logs its id, source and reason,
    never the fact text.

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

    # 2026-10-02: the similarity passes run only on a model REFLECTOR_DUP_SIM
    # was calibrated for (_reflector_semantic_ok); otherwise, exactly as with
    # no embedder, only exact duplicates go.
    vecs = (_embed([_doc_text(t) for t in texts])
            if _reflector_semantic_ok() else None)
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
                    # Delete the older duplicate -- unless that would drop a
                    # trusted (migrated/backfilled) entry for an untrusted
                    # copy: the same _may_condemn rule as the embedder path
                    # (2026-10-01). Otherwise the trusted label is lost and
                    # the contradiction pass may later condemn the survivor.
                    if not _may_condemn(_facts[older].get("source") or "",
                                        _facts[newer].get("source") or ""):
                        older, newer = newer, older
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
    reasons: dict[str, str] = {}    # fid -> why it is deleted (for the log)
    # Decisions for the sink: a condemned / merged-away fid -> the fid that
    # survives it (a contradiction is reported only if the delete goes
    # through, with the survivor's FINAL text, following merges), plus MERGE
    # rewrites -- both inputs and the result -- as applied in place, in order.
    sink_on_delete: dict[str, str] = {}
    sink_applied: list = []
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
                # near-dup: drop the older one -- unless the older one is a
                # trusted fact and the newer is not (2026-10-01): migrated and
                # backfilled facts are ALWAYS the older side, and a newer
                # near-identical variant is exactly what a mis-heard name
                # looks like. Then the untrusted variant goes instead.
                a, b = ids[i], ids[j]
                _before = len(to_delete)
                with _lock:
                    if (_facts.get(a, {}).get("created_at", 0) <=
                            _facts.get(b, {}).get("created_at", 0)):
                        older, newer = a, b
                    else:
                        older, newer = b, a
                if not _may_condemn(snapshot_source.get(older, ""),
                                    snapshot_source.get(newer, "")):
                    older = newer
                to_delete.add(older)
                reasons.setdefault(older, "duplicate")
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
                        "Reply with EXACTLY one of: A (keep only the first), "
                        "B (keep only the second), KEEP (both stay -- the "
                        "answer whenever they do not contradict), or "
                        "MERGE: <one fused fact>. No other words.",
                        [{"role": "fact_a", "text": a_text},
                         {"role": "fact_b", "text": b_text}],
                    ) or "").strip()
                except Exception as e:
                    print(f"  [ltm] reflector llm raised: {e}")
                    verdict = ""
                kind, merged = _parse_reflector_verdict(verdict)
                if kind in ("A", "B"):
                    # 'A' keeps the first presented (ids[i]); 'B' the second.
                    if kind == "A":
                        survivor, condemned = ids[i], ids[j]
                    else:
                        survivor, condemned = ids[j], ids[i]
                    # Trusted-source guard: never delete a migrated/backfilled
                    # fact in favour of a survivor from an untrusted (ambient-
                    # extraction) source — both stay. (2026-07-21 audit #39)
                    if not _may_condemn(snapshot_source.get(condemned, ""),
                                        snapshot_source.get(survivor, "")):
                        continue
                    to_delete.add(condemned)
                    reasons[condemned] = "contradiction"
                    sink_on_delete[condemned] = survivor
                    summary["contradictions_resolved"] += 1
                elif kind == "MERGE":
                    # A MERGE rewrites ids[i] and deletes ids[j], so a trusted
                    # fact on EITHER side would lose its text or its label to
                    # model output (2026-10-01): both stay instead.
                    if not (_may_condemn(snapshot_source.get(ids[i], ""), "")
                            and _may_condemn(snapshot_source.get(ids[j], ""),
                                             "")):
                        continue
                    if not _merged_text_ok(merged, a_text, b_text):
                        continue
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
                            sink_applied.append(
                                ("merge", (a_text, b_text), merged))
                            texts[i] = merged
                            snapshot_text[ids[i]] = merged
                            to_delete.add(ids[j])
                            reasons[ids[j]] = f"merged into {ids[i]}"
                            sink_on_delete[ids[j]] = ids[i]
                            summary["merged"] += 1

    def _final_survivor_text(fid: str) -> Optional[str]:
        # Follow survivor -> survivor while the survivor was itself condemned
        # (a chain within one run); the text of the last one still stored,
        # or None when it is gone too (the condemned text is then dropped).
        sv, seen = sink_on_delete.get(fid), {fid}
        while sv in to_delete and sv in sink_on_delete and sv not in seen:
            seen.add(sv)
            sv = sink_on_delete[sv]
        entry = _facts.get(sv) if sv is not None else None
        if entry is None or sv in to_delete:
            return None
        return entry.get("text") or None

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
                    # Auditable (2026-10-01): which fact went and why -- id,
                    # source and reason only, never the fact's text.
                    print(f"  [ltm] reflector removed {fid} "
                          f"({snapshot_source.get(fid) or 'unknown source'}, "
                          f"{reasons.get(fid, 'unspecified')})")
                    # A merged-away fact was reported with its merge.
                    if reasons.get(fid) == "contradiction":
                        sink_applied.append(
                            ("contradiction",
                             (snapshot_text.get(fid, ""),),
                             _final_survivor_text(fid)))
            _save_facts_locked()
            _rebuild_bm25_locked()
    if sink_applied and _reflector_sink is not None:
        try:
            _reflector_sink(list(sink_applied))
        except Exception as e:
            print(f"  [ltm] reflector sink failed: {type(e).__name__}")
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
                        # The LIVE index (2026-10-02: per MEMORY_EMBED_MODEL
                        # profile; LTM_COLLECTION with default settings).
                        _chroma_client.delete_collection(_collection_name)
                        _collection = _chroma_client.get_or_create_collection(
                            name=_collection_name,
                            metadata=_collection_metadata(_index_profile_key),
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
    bug). Returns counts {"episodes": n, "facts": n, "working": n} plus
    "fact_texts", the texts of the dropped facts that were LEARNED in the
    window (source "merge_memory": mirrored from bobert_memory.json at learn
    time); the caller must drop them there too or the "forgotten" fact stays
    in every system prompt (2026-10-01).

    Migrated / backfilled facts (_TRUSTED_FACT_SOURCES) are KEPT whatever
    their created_at (2026-10-01): it records when an OLD bobert_memory fact
    was copied in (a backfill run, or a migration re-run at boot after failed
    chroma upserts), not when the owner taught it, so a forget within the
    hour after a backfill dropped those facts here and -- through
    fact_texts -- from bobert_memory.json too, with no backup.
    (2026-07-21 audit #51: forget_last_hour left the hour's verbatim turns
    in episodes.jsonl and its facts in the semantic store.)"""
    counts = {"episodes": 0, "facts": 0, "working": 0, "fact_texts": []}
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
            if (entry.get("source") or "") in _TRUSTED_FACT_SOURCES:
                continue        # an old fact copied in, not learned (above)
            try:
                created = float(entry.get("created_at", 0.0))
            except (TypeError, ValueError):
                created = 0.0
            if created >= cutoff_ts:
                doomed.append(fid)
        for fid in doomed:
            _t = _facts[fid].get("text", "")
            # Only a fact merge_memory mirrored at learn time names a
            # bobert_memory.json entry learned in the window.
            if (isinstance(_t, str) and _t.strip()
                    and _facts[fid].get("source") == "merge_memory"):
                counts["fact_texts"].append(_t)
            del _facts[fid]
            _chroma_delete(fid)
        if doomed:
            _save_facts_locked()
            _rebuild_bm25_locked()
            # ...and from any index retired by an embedder switch
            # (2026-10-02): a forgotten fact must not survive in a .bak.
            _purge_from_backups(doomed)
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
            # 2026-10-02: which model built the live index, which one is
            # wanted, and how a switch between them is going.
            "embedder": {
                "index_profile": _index_profile_key,
                "index_collection": _collection_name,
                "wanted_profile": _desired_profile_key(),
                "fallback_from": _embed_fallback_from,
                "rebuild": dict(_reindex_state),
            },
        }


def config_summary() -> dict:
    return {
        "LTM_COLLECTION":             LTM_COLLECTION,
        "LTM_EMBED_MODEL":            LTM_EMBED_MODEL,
        "MEMORY_EMBED_MODEL":         _configured_embed_model(),
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
