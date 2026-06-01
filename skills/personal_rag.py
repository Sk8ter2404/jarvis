"""
Personal-files RAG search tool — JARVIS "second brain".

Thin skill layer over `core.rag_indexer`. Boots the indexing daemon on
load (best-effort; non-fatal if optional deps are missing), exposes
voice-friendly actions ('find that doc about X', 'search my files for
…') that talk to Claude with the top-k chunks pulled from ChromaDB,
and registers a JARVIS tool `search_my_files(query, k=5)` for the
LLM to call mid-reply.

Registered actions
------------------
    rag_search        — speak top hits for a query
    rag_search_quiet  — search but return a structured string Claude can quote
    rag_reindex       — kick a one-shot full-scan in the background
    rag_status        — short human status line
    rag_configure     — set RAG_* knobs, e.g. 'paths=C:/Users/YourName/Documents'
    rag_open_top      — open the top-ranked file from the most recent search

Voice-trigger hints (intentionally not a pre-router — the main
chain-resolver still owns dispatch). Patterns Claude / the LLM
should recognise:

    "find that doc about X"
    "search my files for X"
    "what was that note on X"

Optional dependencies (lazy-imported by core.rag_indexer):
    chromadb, sentence-transformers   ← REQUIRED for any RAG
    watchdog                          ← live re-index (recommended)
    pypdf                             ← .pdf extraction
    python-docx                       ← .docx extraction
    torch (cuda)                      ← GPU embeddings (optional)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_core_on_path() -> None:
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)


# ── runtime knobs (overridable via rag_configure) ────────────────────
RAG_AUTOSTART: bool = True
RAG_DEFAULT_K: int = 5
RAG_SNIPPET_CHARS: int = 280       # snippet length when speaking results
RAG_VOICE_MAX_HITS: int = 3        # cap spoken hits to keep replies tight
# ─────────────────────────────────────────────────────────────────────


_last_query: str = ""
_last_hits: list[dict] = []
_lock = threading.Lock()


def _rag():
    """Lazy import. Returns the module, or None if its mandatory deps
    aren't installed."""
    _ensure_core_on_path()
    try:
        from core import rag_indexer  # type: ignore
        return rag_indexer
    except Exception as e:
        print(f"  [personal-rag] core.rag_indexer import failed: {e}")
        return None


def _short_snippet(text: str, n: int = RAG_SNIPPET_CHARS) -> str:
    if not text:
        return ""
    t = " ".join(text.split())
    if len(t) <= n:
        return t
    return t[:n].rstrip() + "…"


def _format_hits_for_voice(hits: list[dict]) -> str:
    if not hits:
        return "I didn't find anything in your files matching that, sir."
    out: list[str] = []
    for i, h in enumerate(hits[:RAG_VOICE_MAX_HITS], 1):
        name = h.get("filename") or os.path.basename(h.get("path", "")) or "?"
        snip = _short_snippet(h.get("snippet", ""))
        out.append(f"{i}. {name} — {snip}")
    n = len(hits)
    summary = ("Top match" if n == 1
               else f"Top {min(n, RAG_VOICE_MAX_HITS)} matches")
    return f"{summary}, sir:\n" + "\n".join(out)


def _format_hits_for_llm(hits: list[dict]) -> str:
    """Compact, LLM-readable block. Used both by `rag_search_quiet`
    and by the `search_my_files` tool Claude can call."""
    if not hits:
        return "[no matches]"
    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        path = h.get("path", "")
        snip = h.get("snippet", "").strip()
        score = h.get("score", 0.0)
        lines.append(
            f"[{i}] path={path} score={score:.3f}\n{snip}"
        )
    return "\n\n".join(lines)


# ── action implementations ───────────────────────────────────────────
def rag_search(arg: str = "") -> str:
    """Voice-friendly search. Speak the top hits."""
    q = (arg or "").strip()
    if not q:
        return "What should I search your files for, sir?"
    rag = _rag()
    if rag is None or not rag.is_available():
        return ("Personal RAG is offline, sir — install chromadb and "
                "sentence-transformers to enable it.")
    hits = rag.search(q, k=RAG_DEFAULT_K)
    with _lock:
        global _last_query, _last_hits
        _last_query = q
        _last_hits = hits
    return _format_hits_for_voice(hits)


def rag_search_quiet(arg: str = "") -> str:
    """Same search but emits a machine-readable block for Claude's
    context window."""
    q = (arg or "").strip()
    if not q:
        return "[error: empty query]"
    rag = _rag()
    if rag is None or not rag.is_available():
        return "[error: personal RAG unavailable]"
    hits = rag.search(q, k=RAG_DEFAULT_K)
    with _lock:
        global _last_query, _last_hits
        _last_query = q
        _last_hits = hits
    return _format_hits_for_llm(hits)


def search_my_files(query: str, k: int = RAG_DEFAULT_K) -> str:
    """Tool entry point Claude can call mid-reply.

    Returns a compact text block of the top-k chunks. The wrapper
    exists so Claude's tool schema can present a plain Python-callable
    signature (`search_my_files(query, k=5)`), distinct from the
    voice-action signature `(arg: str)`.
    """
    rag = _rag()
    if rag is None or not rag.is_available():
        return "[error: personal RAG unavailable — install chromadb + sentence-transformers]"
    try:
        k = int(k)
    except Exception:
        k = RAG_DEFAULT_K
    hits = rag.search(query, k=max(1, min(k, 20)))
    with _lock:
        global _last_query, _last_hits
        _last_query = str(query)
        _last_hits = hits
    return _format_hits_for_llm(hits)


def rag_reindex(_: str = "") -> str:
    """Trigger a one-shot full scan on a worker thread."""
    rag = _rag()
    if rag is None or not rag.is_available():
        return "Personal RAG is offline, sir."

    def _bg():
        try:
            summary = rag.index_once()
            print(f"  [personal-rag] reindex summary: {summary}")
        except Exception as e:
            print(f"  [personal-rag] reindex failed: {e}")

    threading.Thread(target=_bg, name="rag-manual-reindex",
                     daemon=True).start()
    return "Reindexing your files in the background, sir."


def rag_status(_: str = "") -> str:
    rag = _rag()
    if rag is None:
        return "Personal RAG module not loaded, sir."
    if not rag.is_available():
        return ("Personal RAG is offline, sir — install chromadb and "
                "sentence-transformers.")
    s = rag.status()
    chunks = rag.collection_size()
    last = s.get("last_full_scan_ts") or 0
    last_str = (time.strftime("%H:%M:%S", time.localtime(last))
                if last else "never")
    state = "running" if s.get("running") else "idle"
    wd = "on" if s.get("watchdog_active") else "off"
    return (f"Personal RAG {state} — {chunks} chunks indexed, "
            f"watchdog {wd}, last full scan {last_str}, "
            f"{s.get('errors', 0)} errors.")


def rag_configure(arg: str = "") -> str:
    """Set RAG_* config at runtime. Format: 'key=value' (key is the
    lowercased suffix after RAG_, e.g. 'embed_model', 'index_paths').
    `index_paths` and `exclude_globs` accept comma-separated values."""
    rag = _rag()
    if rag is None:
        return "Personal RAG module not loaded, sir."
    if "=" not in (arg or ""):
        return ("Usage: rag_configure key=value "
                "(index_paths|exclude_globs|embed_model|reranker_model|"
                "device|chunk_chars|chunk_overlap|max_file_bytes).")
    key, _, value = arg.partition("=")
    key = key.strip().lower()
    value = value.strip()

    multi = {"index_paths", "exclude_globs"}
    int_keys = {"chunk_chars", "chunk_overlap", "max_file_bytes"}

    kwargs: dict = {}
    if key in multi:
        kwargs[f"rag_{key}"] = [v.strip() for v in value.split(",") if v.strip()]
    elif key in int_keys:
        try:
            kwargs[f"rag_{key}"] = int(value)
        except ValueError:
            return f"Value for {key} must be an integer, sir."
    elif key in {"embed_model", "reranker_model", "device", "collection"}:
        kwargs[f"rag_{key}"] = value
    else:
        return f"Unknown RAG key '{key}', sir."

    new_cfg = rag.configure(**kwargs)
    return f"RAG {key} set to {new_cfg.get(f'RAG_{key.upper()}')!r}, sir."


def rag_open_top(_: str = "") -> str:
    """Open the top hit from the most recent search in the OS default app."""
    with _lock:
        hits = list(_last_hits)
        q = _last_query
    if not hits:
        return "No recent search to open, sir."
    top = hits[0]
    path = top.get("path", "")
    if not path or not os.path.exists(path):
        return f"Top result from '{q}' no longer exists on disk, sir."
    try:
        os.startfile(path)  # Windows-only; matches the rest of JARVIS
        return f"Opening {os.path.basename(path)}, sir."
    except Exception as e:
        return f"Couldn't open {os.path.basename(path)}: {e}"


# ── registration / autostart ─────────────────────────────────────────
def register(actions: dict) -> None:
    actions["rag_search"]        = rag_search
    actions["rag_search_quiet"]  = rag_search_quiet
    actions["search_my_files"]   = rag_search_quiet  # alias usable as a tool name
    actions["rag_reindex"]       = rag_reindex
    actions["rag_status"]        = rag_status
    actions["rag_configure"]     = rag_configure
    actions["rag_open_top"]      = rag_open_top

    rag = _rag()
    if rag is None:
        return  # core module not importable — actions still registered for status

    if RAG_AUTOSTART and rag.is_available():
        def _bg():
            try:
                time.sleep(3.0)  # let skill loader + whisper settle
                rag.start(initial_scan=True)
            except Exception as e:
                print(f"  [personal-rag] autostart failed: {e}")
        t = threading.Thread(target=_bg, name="personal-rag-autostart",
                             daemon=True)
        t.start()
    elif RAG_AUTOSTART and not rag.is_available():
        print("  [personal-rag] autostart skipped — install "
              "chromadb + sentence-transformers to enable RAG.")
