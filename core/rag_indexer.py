"""
Personal-files RAG indexer for JARVIS.

A long-running daemon that watches a configurable list of user folders
(default: ~/Documents, ~/Desktop, ~/OneDrive) via `watchdog`, extracts
text from each supported file, chunks it, embeds it via Ollama
(default model: nomic-embed-text — GPU-accelerated on the 3090, no
Python tokenizer dependency), and stores the chunks in a local
ChromaDB persistent collection at C:/JARVIS/data/rag_chroma/.

Search side
-----------
A `search(query, k=5)` helper does dense semantic search through Chroma
and, if available, reranks the top-N (default 25) candidates with a
cross-encoder reranker (BAAI/bge-reranker-base). The skill layer
(skills/personal_rag.py) wraps this into a voice-friendly action and a
JARVIS tool exposed to Claude as `search_my_files`.

Supported file types
--------------------
- .txt, .md, .rst, .log
- source code: .py .js .ts .tsx .jsx .go .rs .java .cpp .c .h .hpp .cs
  .rb .php .sh .ps1 .sql .yaml .yml .toml .json .xml .css .html .htm
- .pdf  (via `pypdf`)
- .docx (via `python-docx`)

All optional dependencies are LAZILY imported. Importing this module
never crashes the companion. The first time `index_once()` /
`start()` / `search()` is called, the deps are loaded and a friendly
install hint is printed if anything is missing.

Public API
----------
    from core import rag_indexer as rag
    rag.start()                  # spawn watchdog + initial scan thread
    rag.stop()
    rag.index_once()             # blocking single pass over RAG_INDEX_PATHS
    rag.search("query", k=5)     # → list[dict] of {path, snippet, score, ...}
    rag.status()                 # dict
    rag.is_available()           # True iff chromadb + sentence-transformers present

Configuration (read at start time; override via configure()):
    RAG_INDEX_PATHS       — list of folders to index
    RAG_EXCLUDE_GLOBS     — fnmatch patterns to skip (node_modules, .git,
                            secret-shaped names, .csv/.tsv; core/config.py)
    RAG_EMBED_MODEL       — Ollama embedding model name (default: nomic-embed-text)
    RAG_OLLAMA_ENDPOINT   — Ollama embeddings HTTP endpoint
    RAG_EMBED_BATCH       — chunks per HTTP batch (parallel POSTs)
    RAG_RERANKER_MODEL    — cross-encoder reranker id; "" disables rerank
    RAG_MAX_FILE_BYTES    — skip files larger than this (default 25 MB)
    RAG_CHUNK_CHARS       — chunk size in characters (default 1200)
    RAG_CHUNK_OVERLAP     — chunk overlap in characters (default 200)
    RAG_DEVICE            — "auto" | "cpu" | "cuda" | "cuda:N" for the
                            reranker only (embeddings run via Ollama
                            regardless). "auto" = the CPU: see _device()
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_CHROMA_DIR = os.path.join(_DATA_DIR, "rag_chroma")
_STATE_PATH = os.path.join(_DATA_DIR, "rag_state.json")


def _user_home() -> str:
    return os.path.expanduser("~")


def _default_index_paths() -> list[str]:
    home = _user_home()
    candidates = [
        os.path.join(home, "Documents"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive"),
    ]
    return [p for p in candidates if os.path.isdir(p)]


# ── tunables (overridable via configure()) ───────────────────────────
RAG_INDEX_PATHS: list[str] = _default_index_paths()
# ONE copy of the exclude list: core/config.py (structural skips + secret-
# shaped names + .csv/.tsv exports; matching rules documented there). It used
# to be a second, private list here that knew no secret names. No fallback on
# purpose: if core.config cannot import, this module does not import either,
# and personal_rag reports RAG offline instead of indexing with no excludes.
from core.config import RAG_EXCLUDE_GLOBS as _CONFIG_EXCLUDE_GLOBS  # noqa: E402
RAG_EXCLUDE_GLOBS: list[str] = list(_CONFIG_EXCLUDE_GLOBS)
RAG_EMBED_MODEL: str = "nomic-embed-text"
RAG_OLLAMA_ENDPOINT: str = "http://127.0.0.1:11434/api/embeddings"
RAG_EMBED_BATCH: int = 16  # parallel POSTs per encode() call
RAG_EMBED_TIMEOUT: float = 30.0
RAG_RERANKER_MODEL: str = "BAAI/bge-reranker-base"
RAG_MAX_FILE_BYTES: int = 25 * 1024 * 1024
RAG_CHUNK_CHARS: int = 1200
RAG_CHUNK_OVERLAP: int = 200
RAG_DEVICE: str = "auto"
RAG_COLLECTION: str = "personal_files"
# When the persisted collection was built with a DIFFERENT embed model the
# vectors are dimensionally incompatible. Dropping is destructive (forces a
# multi-hour re-embed), so it is OPT-IN: a model-name mismatch logs a clear
# warning and KEEPS the existing data. Flip this True (or pass force=True to
# index_once) only when an explicit, intentional re-index is wanted.
RAG_REINDEX_ON_MODEL_CHANGE: bool = False

# File extensions we extract text from. Keep small and explicit — the
# scanner skips everything else (images, video, archives, binaries).
_TEXT_EXTS: set[str] = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
    ".java", ".kt", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".cs", ".rb", ".php", ".sh", ".ps1", ".bat", ".sql",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".json", ".xml", ".css", ".scss", ".html", ".htm",
}
_PDF_EXTS: set[str] = {".pdf"}
_DOCX_EXTS: set[str] = {".docx"}

_SUPPORTED_EXTS: set[str] = _TEXT_EXTS | _PDF_EXTS | _DOCX_EXTS


# ── lazy-imported globals ────────────────────────────────────────────
_chroma_client = None
_collection = None
_embed_model = None
_reranker = None
_observer = None
_indexer_thread: Optional[threading.Thread] = None
_stop_flag = threading.Event()
# Short-lived lock guarding ONLY _stats / _last_full_scan_ts / _last_error
# mutations. Must never wrap I/O — readers like status() take it briefly to
# snapshot, so anything slow under this lock blocks them.
_lock = threading.RLock()
# Per-resource first-call guards: a long index_once() must not block a
# concurrent search()'s lazy init of the same singleton.
_collection_init_lock = threading.Lock()
_embedder_init_lock = threading.Lock()
_reranker_init_lock = threading.Lock()
_event_q: "queue.Queue[str]" = queue.Queue()
_last_full_scan_ts: float = 0.0
_last_error: str = ""
_stats = {
    "files_indexed": 0,
    "files_skipped": 0,
    "chunks_written": 0,
    "errors": 0,
}


# ── helpers ──────────────────────────────────────────────────────────
def _norm_path(path) -> str:
    """The form every exclude comparison uses: forward slashes, lower case.
    fnmatch.fnmatch() is case-insensitive on Windows but case-SENSITIVE on
    Linux, so the matcher lowers both sides and compiles the patterns with
    fnmatch.translate (a case-sensitive regex) — the same on every OS."""
    return str(path or "").replace("\\", "/").lower()


@functools.lru_cache(maxsize=8)
def _compiled_excludes(globs: tuple) -> tuple:
    """(whole-path regex, name regex) for one exclude list; None when that
    kind has no patterns. A pattern with a slash is a whole-path pattern; one
    without is a name pattern (see _is_excluded)."""
    path_parts: list[str] = []
    name_parts: list[str] = []
    for g in globs:
        pat = _norm_path(g).strip()
        if not pat:
            continue
        (path_parts if "/" in pat else name_parts).append(fnmatch.translate(pat))
    path_rx = re.compile("|".join(path_parts)) if path_parts else None
    name_rx = re.compile("|".join(name_parts)) if name_parts else None
    return path_rx, name_rx


def _names_below_root(norm: str) -> list[str]:
    """The file's name plus the name of every folder between the watched root
    holding `norm` (the longest matching RAG_INDEX_PATHS entry) and the file.
    Folders at or above the root are left out, so a user folder that happens
    to contain "pass" or "token" can't hide everything under it. A path under
    no root yields its file name only."""
    best = ""
    for root in RAG_INDEX_PATHS or ():
        r = _norm_path(root).rstrip("/")
        if r and len(r) > len(best) and norm.startswith(r + "/"):
            best = r
    rel = norm[len(best) + 1:] if best else norm.rsplit("/", 1)[-1]
    return [part for part in rel.split("/") if part]


def _is_excluded(path: str, is_dir: bool = False) -> bool:
    """True when RAG_EXCLUDE_GLOBS says never to read `path`.

    Case-insensitive, / and \\ alike, on both sides. A pattern containing a
    slash ("*/node_modules/*") is matched against the whole path (and, for a
    folder, the path plus a trailing slash). A pattern without one
    ("*password*", "*.csv") is matched against the file name and every folder
    name below the watched root — so the walk, the watcher and the index's
    garbage-collect pass all agree on what is excluded."""
    globs = RAG_EXCLUDE_GLOBS
    if isinstance(globs, str):          # one pattern, not its letters
        globs = [globs]
    path_rx, name_rx = _compiled_excludes(
        tuple(g for g in (globs or ()) if isinstance(g, str)))
    norm = _norm_path(path).rstrip("/")
    if path_rx is not None and (path_rx.match(norm) or (
            is_dir and path_rx.match(norm + "/"))):
        return True
    if name_rx is not None:
        return any(name_rx.match(n) for n in _names_below_root(norm))
    return False


def _supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _SUPPORTED_EXTS


def _file_id(path: str) -> str:
    """Stable per-file id. We use a SHA1 of the abspath so paths with
    odd characters don't break Chroma's id constraints."""
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()


def _read_text(path: str) -> str:
    """Pull text out of one file. Returns empty string on any failure
    so the caller can simply skip it. Lazy-imports optional deps."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in _TEXT_EXTS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext in _PDF_EXTS:
            try:
                from pypdf import PdfReader
            except ImportError:
                return ""
            try:
                reader = PdfReader(path)
                pages = []
                for page in reader.pages:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        continue
                return "\n".join(pages)
            except Exception:
                return ""
        if ext in _DOCX_EXTS:
            try:
                import docx  # python-docx
            except ImportError:
                return ""
            try:
                d = docx.Document(path)
                return "\n".join(p.text for p in d.paragraphs)
            except Exception:
                return ""
    except Exception:
        return ""
    return ""


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Simple character-based chunker. Splits on paragraph boundaries
    where possible so chunks don't slice mid-sentence."""
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        # Try to back off to the nearest paragraph break inside the
        # tail half of the window so chunks split on \n\n where possible.
        if end < n:
            cut = text.rfind("\n\n", i + size // 2, end)
            if cut == -1:
                cut = text.rfind("\n", i + size // 2, end)
            if cut != -1 and cut > i + size // 4:
                end = cut
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


# ── lazy initialisation ──────────────────────────────────────────────
def is_available() -> bool:
    """Probe whether the indexing + search path has its mandatory deps.
    Returns True iff `chromadb` is importable. Ollama is reached over
    HTTP and probed lazily on first embed call — its absence is logged
    but does not flip is_available() so the rest of RAG (status,
    config) stays responsive. PDF / docx / watchdog / reranker are
    *optional* — their absence just disables the feature they back."""
    try:
        import chromadb
        del chromadb  # probe only; pyflakes ignores noqa (2026-10-02)
        return True
    except Exception:
        return False


def _device() -> str:
    """The reranker's torch device. "auto" is the CPU (2026-10-04): it used
    to be "cuda" whenever torch had CUDA, i.e. cuda:0 — the RTX 3090 that
    holds the local brain with ~1.1 GB free — where bge-reranker-base (~1.1
    GB fp32) plus a CUDA context would have landed on the first file search.
    A card only when RAG_DEVICE names one ("cuda" / "cuda:N")."""
    dev = str(RAG_DEVICE or "auto").strip().lower()
    if dev == "cpu" or dev == "cuda" or dev.startswith("cuda:"):
        return dev
    return "cpu"


# ── brain-eviction guard (2026-10-04) ──────────────────────────────────
# With OLLAMA_MAX_LOADED_MODELS=1 every embedding request while the voice
# brain is loaded UNLOADS it: the boot scans of 10-02 15:05:40 (657 chunks)
# and 10-03 17:35:41 (3,699 chunks) did, and the next brain loads took 54 s
# and 9 s (Ollama server.log; the brain was gone 9 minutes the second time).
# Indexing can wait, so it does: the boot scan, the watcher and the rescan
# of a deferred scan embed only while core.ollama_opts.eviction_risk says a
# request for the embed model unloads nothing (it is loaded already, nothing
# else is, or the server may hold more than one model), and try again
# RAG_DEFER_RETRY_S later otherwise. A search is the owner's own request and
# still runs (one line in the log says what it unloads); so does his
# reindex (below).
#
# Review 2026-10-04:
#  * "reindex my files" is the owner's request too. With the brain kept
#    loaded for 24 h, a reindex that waited for it never ran while JARVIS
#    said it was reindexing. An owner reindex (index_once(owner=True))
#    therefore embeds like his search does, unloading the brain if it must
#    (his reply says so) - but it pauses while he is mid-sentence or
#    mid-turn (OWNER_SCAN_MAX_WAIT_S at most), so none of its requests is in
#    flight when his turn's brain request lands. status() reports any wait.
#  * While the server holds one model, a background embedding also waits
#    while the owner is talking to JARVIS (core.local_traffic's background
#    gate): one sent just before his turn's brain request would reload the
#    embed model right after the brain answered, unloading it again.
RAG_DEFER_RETRY_S: float = 600.0
OWNER_SCAN_MAX_WAIT_S: float = 120.0


class EmbedDeferred(RuntimeError):
    """An embedding was not sent: it would have unloaded another Ollama
    model (the voice brain). The message says what and why."""


def _ollama_base(endpoint: str) -> str:
    return str(endpoint or "").split("/api/", 1)[0].rstrip("/")


def _owner_talking_reason(hard_only: bool = False) -> str:
    """'' unless the owner is talking to JARVIS right now: core.local_traffic's
    background-gate reason ('utterance', 'turn', or 'conversation' - the
    quiet window after a turn; ``hard_only`` ignores that one). Unconfigured
    gate = ''. Never raises."""
    try:
        from core import local_traffic as _lt
        why = _lt.GATE.defer_reason()
        if not why or (hard_only and why not in _lt.HARD_REASONS):
            return ""
        return f"the owner is talking to JARVIS ({why})"
    except Exception:
        return ""


def _embed_eviction_reason(model: str, endpoint: str,
                           resident: "list | None" = None) -> str:
    """Why a BACKGROUND embedding request for ``model`` must wait right now
    ('' = it may go): core.ollama_opts.eviction_risk against the server
    ``endpoint`` points at (``resident``: its /api/ps list, already read),
    and - while that server holds one model - the owner talking to JARVIS.
    Fails closed. Never raises."""
    try:
        from core import ollama_opts as _oo
        base = _ollama_base(endpoint)
        why = _oo.eviction_risk(model, base, resident=resident)
        if why:
            return why
        if _oo.effective_max_loaded(base) < 2:
            return _owner_talking_reason()
        return ""
    except Exception as e:  # pragma: no cover - in-tree import
        return f"eviction guard unavailable ({type(e).__name__})"


def _wait_owner_quiet(max_s: "float | None" = None) -> None:
    """An owner scan's request waits here while he is mid-sentence or
    mid-turn (at most ``max_s``, default OWNER_SCAN_MAX_WAIT_S; the scan he
    asked for starts inside his own turn). Never raises."""
    try:
        cap = OWNER_SCAN_MAX_WAIT_S if max_s is None else float(max_s)
        end = time.monotonic() + max(0.0, cap)
        while (_owner_talking_reason(hard_only=True)
               and not _stop_flag.is_set() and time.monotonic() < end):
            time.sleep(0.25)
    except Exception:
        pass


def embed_would_unload() -> str:
    """'' when embedding now unloads no other Ollama model, else why it would
    (the owner's "reindex my files" reply says so). Never raises."""
    try:
        from core.ollama_opts import eviction_risk
        return eviction_risk(RAG_EMBED_MODEL, _ollama_base(RAG_OLLAMA_ENDPOINT))
    except Exception:
        return ""


# index_once(owner=True) marks its own thread; _OllamaEmbedder.encode reads
# the mark there (its pool workers are other threads) and hands it on.
_owner_scan = threading.local()


# The deferral reason last logged ('' = none since the last embedding that
# went through), so a deferred scan logs once, not per file.
_deferral_logged = [""]


def _log_deferral(where: str, reason: str) -> None:
    if _deferral_logged[0] != reason:
        _deferral_logged[0] = reason
        print(f"  [rag] {where} deferred: {reason}; retrying in "
              f"{RAG_DEFER_RETRY_S / 60:.0f} min")


class _OllamaEmbedder:
    """Thin wrapper that mimics SentenceTransformer's .encode() interface
    but POSTs to Ollama's /api/embeddings. One request per chunk, but
    fanned out across a thread pool so a batch finishes in roughly
    `len(chunks) / batch_size` round-trip times instead of serial.

    Returns numpy float32 arrays so the rest of the indexer (which
    calls .tolist() before handing embeddings to Chroma) is unchanged.

    Every request first asks _embed_eviction_reason (per request, so a brain
    loaded mid-file stops the rest of the file) unless ``allow_evict`` - the
    owner's search - and raises EmbedDeferred instead of sending. A request
    of an owner scan (``owner``) is sent like a search, after
    _wait_owner_quiet. A request that loads the embed model NEXT TO other
    models re-reads /api/ps afterwards (core.ollama_opts.note_coload).
    """

    def __init__(self, model: str, endpoint: str,
                 batch_size: int = 16, timeout: float = 30.0):
        self.model = model
        self.endpoint = endpoint
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)

    def _embed_one(self, text: str, allow_evict: bool = False,
                   owner: bool = False) -> list[float]:
        coload = None
        if owner:
            _wait_owner_quiet()
            allow_evict = True
        if not allow_evict:
            from core import ollama_opts as _oo
            base = _ollama_base(self.endpoint)
            before = _oo.resident_models(base, _oo.probe_timeout(base))
            reason = _embed_eviction_reason(self.model, self.endpoint,
                                            resident=before)
            if reason:
                raise EmbedDeferred(reason)
            if before and not any(_oo.same_tag(n, self.model)
                                  for n in before):
                coload = before     # loads next to them: check afterwards
        # One-shot GPU snapshot on first embedding call so VRAM
        # allocation for nomic-embed-text is captured in the log.
        # Safe to call on every request — dedup is inside log_gpu_state.
        try:
            from core import gpu_state as _gpu_state
            _gpu_state.log_gpu_state(self.model)
        except Exception:
            pass
        body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if coload is not None:
            try:
                from core import ollama_opts as _oo
                base = _ollama_base(self.endpoint)
                _oo.note_coload(coload,
                                _oo.resident_models(base,
                                                    _oo.probe_timeout(base)),
                                self.model)
            except Exception:
                pass
        emb = payload.get("embedding") or []
        if not emb:
            raise RuntimeError(f"Ollama returned empty embedding (model={self.model})")
        return [float(x) for x in emb]

    def _normalise(self, vec: "list[float]"):
        # Cosine search via Chroma assumes normalised vectors when we
        # want to compare dot-products; replicate SentenceTransformer's
        # normalize_embeddings=True behaviour locally.
        s = 0.0
        for x in vec:
            s += x * x
        if s <= 0.0:
            return vec
        inv = s ** -0.5
        return [x * inv for x in vec]

    def encode(self, texts, batch_size: Optional[int] = None,
               convert_to_numpy: bool = True,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = False,
               allow_evict: bool = False,
               **_ignored):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype="float32") if convert_to_numpy else []

        n_workers = batch_size if batch_size else self.batch_size
        n_workers = max(1, min(n_workers, len(texts)))
        one = functools.partial(
            self._embed_one, allow_evict=allow_evict,
            owner=bool(getattr(_owner_scan, "active", False)))

        if n_workers == 1 or len(texts) == 1:
            vecs = [one(t) for t in texts]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                vecs = list(pool.map(one, texts))

        if normalize_embeddings:
            vecs = [self._normalise(v) for v in vecs]

        if convert_to_numpy:
            return np.asarray(vecs, dtype="float32")
        return vecs


def _ollama_reachable() -> tuple[bool, str]:
    """Reachability check for the Ollama embedding endpoint, as
    (ok, reason). `reason` is '' when ok.

    DOES NOT EMBED. This used to do a real one-shot embedding round-trip with
    a 5 s timeout, which was wrong twice over (both observed live 2026-07-21):

      1. FALSE NEGATIVE. When the 16 GB voice brain is resident — i.e. always,
         at boot — Ollama must EVICT it and cold-load nomic-embed-text to serve
         the ping. That takes far longer than 5 s, so the probe timed out and
         the boot log announced "endpoint unreachable — indexing will fail"
         about an endpoint that was working perfectly; minutes later the same
         endpoint served real embeddings.
      2. SELF-INFLICTED HARM. The probe existed only to print a log line, but
         issuing it EVICTED the primary brain at boot. With
         OLLAMA_MAX_LOADED_MODELS=1 an embedding request is never free.

    A GET of /api/tags answers both real questions — is the daemon up, and is
    the embed model pulled — without loading anything.
    """
    base = RAG_OLLAMA_ENDPOINT.split("/api/", 1)[0].rstrip("/")
    try:
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    names = set()
    for m in (payload.get("models") or []):
        name = (m or {}).get("name") or (m or {}).get("model") or ""
        if name:
            names.add(name)
            # Ollama reports fully-qualified tags ("nomic-embed-text:latest");
            # accept a bare-name configuration too.
            names.add(name.split(":", 1)[0])
    if RAG_EMBED_MODEL not in names and RAG_EMBED_MODEL.split(":", 1)[0] not in names:
        return False, (f"daemon is up but model {RAG_EMBED_MODEL!r} is not "
                       f"pulled (ollama pull {RAG_EMBED_MODEL})")
    return True, ""


def _get_embedder():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    with _embedder_init_lock:
        if _embed_model is not None:
            return _embed_model
        print(f"  [rag] using Ollama embedder model={RAG_EMBED_MODEL} "
              f"endpoint={RAG_OLLAMA_ENDPOINT}")
        _embed_model = _OllamaEmbedder(
            model=RAG_EMBED_MODEL,
            endpoint=RAG_OLLAMA_ENDPOINT,
            batch_size=RAG_EMBED_BATCH,
            timeout=RAG_EMBED_TIMEOUT,
        )
        return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is not None or not RAG_RERANKER_MODEL:
        return _reranker
    with _reranker_init_lock:
        if _reranker is not None or not RAG_RERANKER_MODEL:
            return _reranker
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return None
        try:
            dev = _device()
            print(f"  [rag] loading reranker {RAG_RERANKER_MODEL} on {dev}")
            _reranker = CrossEncoder(RAG_RERANKER_MODEL, device=dev)
        except Exception as e:
            # GPU-first, but fall back to CPU on a cuda OOM rather than
            # losing rerank entirely (the 3090 can hit 24 GB when image-gen
            # loads alongside a large Ollama model).
            if str(dev).startswith("cuda"):
                print(f"  [rag] cuda reranker load failed ({e}); retrying on CPU")
                try:
                    _reranker = CrossEncoder(RAG_RERANKER_MODEL, device="cpu")
                except Exception as e2:
                    print(f"  [rag] reranker unavailable ({e2}); skipping rerank")
                    _reranker = None
            else:
                print(f"  [rag] reranker unavailable ({e}); skipping rerank")
                _reranker = None
        return _reranker


def _get_collection(force_reindex: bool = False):
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    with _collection_init_lock:
        if _collection is not None:
            return _collection
        import chromadb
        os.makedirs(_CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=RAG_COLLECTION,
            metadata={
                "hnsw:space": "cosine",
                "embed_model": RAG_EMBED_MODEL,
            },
        )
        # Stamp / migration check. We do NOT delete on mismatch by default:
        # a config typo or a missing stamp would otherwise silently wipe the
        # whole index and force a multi-hour re-embed. Two cases:
        #   (a) MISSING/unstamped stamp → treat as current. Stamp it in place
        #       and KEEP the data (no embed model actually changed; the stamp
        #       just pre-dates this code path).
        #   (b) genuine model-name mismatch → WARN loudly and KEEP the data,
        #       rebuilding only when an explicit re-index is requested
        #       (RAG_REINDEX_ON_MODEL_CHANGE).
        try:
            existing_model = ""
            meta = getattr(_collection, "metadata", None) or {}
            if isinstance(meta, dict):
                existing_model = str(meta.get("embed_model") or "")
            if not existing_model:
                # Unstamped (or freshly created) — adopt the current model as
                # the stamp without touching any stored vectors.
                try:
                    new_meta = dict(meta) if isinstance(meta, dict) else {}
                    new_meta["hnsw:space"] = "cosine"
                    new_meta["embed_model"] = RAG_EMBED_MODEL
                    _collection.modify(metadata=new_meta)
                except Exception as e:
                    print(f"  [rag] could not stamp collection embed_model "
                          f"({e}); continuing with existing data")
            elif existing_model != RAG_EMBED_MODEL:
                if force_reindex or RAG_REINDEX_ON_MODEL_CHANGE:
                    try:
                        count = int(_collection.count())
                    except Exception:
                        count = 0
                    print(f"  [rag] embed model changed "
                          f"({existing_model} → {RAG_EMBED_MODEL}) and "
                          f"explicit reindex requested; dropping collection "
                          f"for re-index ({count} chunks discarded)")
                    _chroma_client.delete_collection(name=RAG_COLLECTION)
                    _collection = _chroma_client.get_or_create_collection(
                        name=RAG_COLLECTION,
                        metadata={
                            "hnsw:space": "cosine",
                            "embed_model": RAG_EMBED_MODEL,
                        },
                    )
                else:
                    print(f"  [rag] WARNING: embed model differs from the "
                          f"stored stamp ({existing_model} → "
                          f"{RAG_EMBED_MODEL}). Keeping existing collection "
                          f"to preserve your index; queries may be degraded "
                          f"if this is a real model change. Set "
                          f"RAG_REINDEX_ON_MODEL_CHANGE=True (or pass "
                          f"force=True to index_once) to rebuild deliberately.")
        except Exception as e:
            print(f"  [rag] migration check failed ({e}); continuing")
        return _collection


# ── indexing ─────────────────────────────────────────────────────────
def _iter_files(root: str) -> Iterable[str]:
    """Recursively walk `root`, yielding absolute paths of supported,
    non-excluded files within the size budget."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories cheaply by filtering dirnames — through
        # the same _is_excluded the watcher and the GC pass use.
        keep = []
        for d in dirnames:
            if _is_excluded(os.path.join(dirpath, d), is_dir=True):
                continue
            keep.append(d)
        dirnames[:] = keep
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not _supported(path) or _is_excluded(path):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size <= 0 or size > RAG_MAX_FILE_BYTES:
                continue
            yield path


def _file_signature(path: str) -> str:
    """Cheap content-change fingerprint: mtime+size. Avoids hashing
    every file on every scan. Stored as Chroma metadata so we can
    skip unchanged files."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    return f"{int(st.st_mtime)}:{st.st_size}"


def _existing_signature(file_id: str) -> str:
    coll = _get_collection()
    try:
        res = coll.get(where={"file_id": file_id}, include=["metadatas"], limit=1)
    except Exception:
        return ""
    metas = res.get("metadatas") if isinstance(res, dict) else None
    if metas:
        m = metas[0]
        return str(m.get("sig", "")) if isinstance(m, dict) else ""
    return ""


def _delete_file(file_id: str) -> None:
    coll = _get_collection()
    try:
        coll.delete(where={"file_id": file_id})
    except Exception:
        pass


def _index_file(path: str) -> int:
    """Embed and write one file's chunks. Returns count of chunks written,
    or 0 if skipped/unchanged."""
    global _last_error
    if not _supported(path) or _is_excluded(path):
        return 0
    fid = _file_id(path)
    sig = _file_signature(path)
    if not sig:
        return 0
    prev = _existing_signature(fid)
    if prev and prev == sig:
        return 0  # unchanged — skip

    text = _read_text(path)
    if not text or not text.strip():
        _delete_file(fid)  # was indexed before, now empty / unreadable
        return 0

    chunks = _chunk(text, RAG_CHUNK_CHARS, RAG_CHUNK_OVERLAP)
    if not chunks:
        return 0

    embedder = _get_embedder()
    coll = _get_collection()

    try:
        embeddings = embedder.encode(
            chunks, batch_size=RAG_EMBED_BATCH, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True,
        )
    except EmbedDeferred:
        # Not an error: the caller re-tries the file later (nothing written,
        # the old chunks stay searchable).
        raise
    except (urllib.error.URLError, urllib.error.HTTPError,
            ConnectionError, TimeoutError) as e:
        with _lock:
            _stats["errors"] += 1
            _last_error = f"ollama embed({path}): {e}"
        return 0
    except Exception as e:
        with _lock:
            _stats["errors"] += 1
            _last_error = f"embed({path}): {e}"
        return 0

    # Replace any prior chunks for this file in one go (only after a
    # successful embed — otherwise an Ollama outage would wipe data).
    _delete_file(fid)
    ids = [f"{fid}:{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "file_id": fid,
            "path": path,
            "filename": os.path.basename(path),
            "chunk_index": i,
            "sig": sig,
            "ext": os.path.splitext(path)[1].lower(),
        }
        for i in range(len(chunks))
    ]
    try:
        coll.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception as e:
        with _lock:
            _stats["errors"] += 1
            _last_error = f"add({path}): {e}"
        return 0
    with _lock:
        _stats["chunks_written"] += len(chunks)
        _stats["files_indexed"] += 1
    _deferral_logged[0] = ""
    return len(chunks)


def index_once(progress: Optional[Callable[[str, int], None]] = None,
               force: bool = False, owner: bool = False) -> dict:
    """Walk every RAG_INDEX_PATHS root once and index unchanged-skipping
    everything. Blocking; returns a small summary dict (see _index_once).
    ``owner``: the owner asked for this scan ("reindex my files") - its
    embeddings go out like his search's, after waiting out his own
    utterance / turn, instead of waiting for the voice brain to unload
    (review 2026-10-04, see EmbedDeferred)."""
    prev = getattr(_owner_scan, "active", False)
    _owner_scan.active = bool(owner)
    try:
        return _index_once(progress, force)
    finally:
        _owner_scan.active = prev


def _index_once(progress: Optional[Callable[[str, int], None]] = None,
                force: bool = False) -> dict:
    """Walk every RAG_INDEX_PATHS root once and index unchanged-skipping
    everything. Blocking; returns a small summary dict.

    `force=True` requests a deliberate rebuild: if the persisted collection
    was stamped with a different embed model it is dropped and re-embedded.
    By default (force=False) a model-name mismatch is NON-destructive — the
    existing index is preserved and only a warning is logged.

    Concurrency: this does NOT hold _lock for the duration of the scan —
    on a 10k-file tree that would block status() and any other reader for
    minutes. Per-resource init locks make concurrent first-call safe; the
    short-lived _lock is taken only around _stats / _last_error mutations.
    """
    if not is_available():
        return {"ok": False, "error": "chromadb not installed"}
    global _last_full_scan_ts, _last_error, _collection
    with _lock:
        _last_error = ""

    # An explicit force overrides the non-destructive default for this run
    # only. Drop the cached singleton and rebuild with the destructive
    # reindex enabled FOR THIS CALL ONLY — passed as a parameter, NOT by
    # mutating the module-global RAG_REINDEX_ON_MODEL_CHANGE. The old code
    # flipped that global True across the rebuild window, so a concurrent
    # search()/index_once() calling _get_collection() during that window
    # could observe it True on an embed-model stamp mismatch and DROP the
    # entire collection (multi-hour re-embed / data loss). 2026-05-30 audit.
    if force:
        with _collection_init_lock:
            _collection = None
        _get_collection(force_reindex=True)

    # Warm singletons outside the global lock — their own init locks
    # serialize the first-call race without blocking readers.
    _get_collection()
    _get_embedder()

    seen_files: set[str] = set()
    files_seen_count = 0
    deferred = ""
    for root in RAG_INDEX_PATHS:
        if not os.path.isdir(root):
            continue
        for path in _iter_files(root):
            if _stop_flag.is_set():
                break
            seen_files.add(_file_id(path))
            try:
                _index_file(path)
            except EmbedDeferred as e:
                # Embedding now would unload the voice brain: stop the walk
                # (the next file would wait for the same reason) and leave
                # the rest for the retry. Unchanged files never reach the
                # embedder, so the walk only stops at real work.
                deferred = str(e)
                break
            except Exception as e:
                with _lock:
                    _stats["errors"] += 1
                    _last_error = f"{path}: {e}"
                continue
            files_seen_count += 1
            if progress and (files_seen_count % 25 == 0):
                try:
                    progress(path, files_seen_count)
                except Exception:
                    pass
        if _stop_flag.is_set() or deferred:
            break

    if deferred:
        # An unfinished walk must not garbage-collect: every file it never
        # reached would look deleted. The daemon's drain loop runs the scan
        # again later (an owner scan's rest included, as a background one).
        _scan_retry_at[0] = time.time() + RAG_DEFER_RETRY_S
        _log_deferral("index scan", deferred)
        with _lock:
            return {
                "ok": True,
                "deferred": deferred,
                "files_seen": files_seen_count,
                "files_indexed_total": _stats["files_indexed"],
                "chunks_written_total": _stats["chunks_written"],
                "errors": _stats["errors"],
                "excluded_dropped": 0,
                "ts": _last_full_scan_ts,
            }

    # Garbage-collect: drop chunks whose file no longer exists on disk, AND
    # chunks of a file that now matches RAG_EXCLUDE_GLOBS. The second half is
    # new: a file indexed before its pattern was added (a passwords file, a
    # .csv device export) used to stay searchable forever, because the walk
    # merely stopped visiting it while the file itself stayed on disk.
    excluded_dropped = 0
    try:
        coll = _get_collection()
        existing = coll.get(include=["metadatas"])
        metas = existing.get("metadatas") or []
        ids = existing.get("ids") or []
        excluded_memo: dict[str, bool] = {}
        stale_ids: list = []
        for cid, m in zip(ids, metas):
            if not isinstance(m, dict) or not m.get("path"):
                continue
            p = str(m.get("path", ""))
            if p not in excluded_memo:
                excluded_memo[p] = _is_excluded(p)
            if excluded_memo[p]:
                stale_ids.append(cid)
            elif m.get("file_id") not in seen_files and not os.path.isfile(p):
                stale_ids.append(cid)
        if stale_ids:
            coll.delete(ids=stale_ids)
        excluded_dropped = sum(1 for hit in excluded_memo.values() if hit)
        if excluded_dropped:
            print(f"  [rag] dropped {excluded_dropped} file(s) from the index "
                  f"that now match RAG_EXCLUDE_GLOBS")
    except Exception:
        excluded_dropped = 0

    _scan_retry_at[0] = 0.0
    _deferral_logged[0] = ""
    with _lock:
        _last_full_scan_ts = time.time()
        return {
            "ok": True,
            "files_seen": files_seen_count,
            "files_indexed_total": _stats["files_indexed"],
            "chunks_written_total": _stats["chunks_written"],
            "errors": _stats["errors"],
            "excluded_dropped": excluded_dropped,
            "ts": _last_full_scan_ts,
        }


# ── watchdog daemon ──────────────────────────────────────────────────
# time.time() when a deferred full scan is due again (0.0 = none pending).
_scan_retry_at = [0.0]


def _run_scan(label: str) -> dict:
    """One index_once() for the daemon. A deferred scan (see EmbedDeferred)
    has already scheduled its retry (_scan_retry_at, run by the drain loop)
    and logged why; a finished one prints its summary."""
    summary = index_once()
    if not (isinstance(summary, dict) and summary.get("deferred")):
        print(f"  [rag] {label}: {summary}")
    return summary


def _drain_event_queue() -> None:
    """Single-threaded reindex worker — drains _event_q. Coalesces
    multiple events for the same path within a short window into one
    re-index call (debounce ~2 s). A path whose embedding was deferred
    (EmbedDeferred) waits RAG_DEFER_RETRY_S and is tried again; a deferred
    full scan is re-run here when it is due."""
    pending: dict[str, float] = {}
    while not _stop_flag.is_set():
        try:
            path = _event_q.get(timeout=0.1)
            pending[path] = max(pending.get(path, 0.0), time.time() + 2.0)
        except queue.Empty:
            pass

        now = time.time()
        ready = [p for p, due in pending.items() if due <= now]
        for p in ready:
            pending.pop(p, None)
            if not os.path.exists(p):
                try:
                    _delete_file(_file_id(p))
                except Exception:
                    pass
                continue
            try:
                _index_file(p)
            except EmbedDeferred as e:
                pending[p] = now + RAG_DEFER_RETRY_S
                _log_deferral("re-index", str(e))
            except Exception as e:
                global _last_error
                with _lock:
                    _stats["errors"] += 1
                    _last_error = f"reindex({p}): {e}"
        if _scan_retry_at[0] and now >= _scan_retry_at[0] and not ready:
            try:
                _run_scan("rescan after deferral")
            except Exception as e:
                _scan_retry_at[0] = now + RAG_DEFER_RETRY_S
                print(f"  [rag] rescan failed: {e}")


def _start_watchdog() -> bool:
    """Boot the watchdog Observer over every RAG_INDEX_PATHS root."""
    global _observer
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("  [rag] watchdog not installed; live re-index disabled "
              "(run `pip install watchdog` to enable)")
        return False

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event):  # noqa: ARG002
            if event.is_directory:
                return
            # A rename (dest_path set) queues the NEW name to index AND the
            # OLD one, which no longer exists, so the drain deletes its
            # chunks. Without the old one, a note renamed to an excluded
            # name ("passwords.txt") stayed searchable under its old name
            # until the next full scan.
            dest = getattr(event, "dest_path", "") or ""
            src = getattr(event, "src_path", "") or ""
            for path in ((dest, src) if dest else (src,)):
                if not path or not _supported(path) or _is_excluded(path):
                    continue
                try:
                    _event_q.put_nowait(path)
                except Exception:
                    pass

    obs = Observer()
    handler = _Handler()
    watched = 0
    for root in RAG_INDEX_PATHS:
        if not os.path.isdir(root):
            continue
        try:
            obs.schedule(handler, root, recursive=True)
            watched += 1
        except Exception as e:
            print(f"  [rag] watchdog schedule({root}) failed: {e}")
    if watched == 0:
        return False
    obs.start()
    _observer = obs
    print(f"  [rag] watchdog active on {watched} root(s)")
    return True


def start(initial_scan: bool = True) -> bool:
    """Boot the indexer daemon. Returns True iff at least the chromadb
    + embedder backing is available. Watchdog is best-effort — the
    daemon still does the initial scan even when watchdog is missing,
    so manual reindex_path() calls (and per-restart scans) keep the
    collection fresh."""
    if not is_available():
        print("  [rag] chromadb not installed; RAG disabled "
              "(pip install chromadb)")
        return False
    # One-shot probe so the boot log surfaces an unreachable Ollama
    # before the first index attempt silently piles up errors.
    _reach_ok, _reach_why = _ollama_reachable()
    if not _reach_ok:
        print(f"  [rag] WARNING: Ollama embeddings endpoint "
              f"{RAG_OLLAMA_ENDPOINT} unusable — indexing will fail "
              f"until `ollama serve` is running with model "
              f"'{RAG_EMBED_MODEL}' pulled ({_reach_why})")
    global _indexer_thread
    if _indexer_thread is not None and _indexer_thread.is_alive():
        return True
    _stop_flag.clear()

    def _bg():
        if initial_scan:
            try:
                _run_scan("initial scan")
            except Exception as e:
                global _last_error
                _last_error = f"initial scan: {e}"
                print(f"  [rag] initial scan failed: {e}")
        # Live re-index loop. Stays alive even if watchdog isn't.
        _drain_event_queue()

    _start_watchdog()  # may quietly disable itself
    _indexer_thread = threading.Thread(
        target=_bg, name="rag-indexer", daemon=True,
    )
    _indexer_thread.start()
    return True


def stop() -> None:
    _stop_flag.set()
    obs = _observer
    if obs is not None:
        try:
            obs.stop()
            obs.join(timeout=2.0)
        except Exception:
            pass
    globals()["_observer"] = None


# ── search ───────────────────────────────────────────────────────────
def search(query: str, k: int = 5, candidates: int = 25,
           paths: Optional[Iterable[str]] = None) -> list[dict]:
    """Semantic search over the personal file collection.

    Returns up to `k` hits, each a dict:
        {path, filename, snippet, chunk_index, score, ext}
    `paths` (optional) filters to chunks whose path starts with any of
    the supplied prefixes — useful for scoping a search to one folder.
    """
    if not query or not query.strip():
        return []
    if not is_available():
        return []
    try:
        coll = _get_collection()
        embedder = _get_embedder()
    except Exception as e:
        print(f"  [rag] search init failed: {e}")
        return []
    # The owner asked for this search: it runs even when the embedding
    # unloads the voice brain (the background indexer waits instead); the
    # next brain call reloads it (its load_ms shows on [turn-timing]).
    try:
        qvec = embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False, allow_evict=True,
        )[0]
    except Exception as e:
        print(f"  [rag] embed-query failed ({e}); Ollama unreachable?")
        return []
    try:
        res = coll.query(
            query_embeddings=[qvec.tolist()],
            n_results=max(candidates, k),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"  [rag] chroma query failed: {e}")
        return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            continue
        path = str(meta.get("path", ""))
        if paths and not any(path.startswith(p) for p in paths):
            continue
        # Never hand back a file RAG_EXCLUDE_GLOBS now covers. index_once()
        # drops such chunks only in its clean-up pass AFTER the whole walk
        # (an hour or more when a big folder was just added), and not at all
        # when that pass fails; until then a passwords file indexed under an
        # older list would still be read out. Same check as the walk.
        if path and _is_excluded(path):
            continue
        # cosine distance → similarity score in (-1, 1]; clip to [0, 1]
        sim = max(0.0, 1.0 - float(dist))
        hits.append({
            "path": path,
            "filename": str(meta.get("filename", "")),
            "snippet": doc or "",
            "chunk_index": int(meta.get("chunk_index", 0)),
            "score": sim,
            "ext": str(meta.get("ext", "")),
        })

    # Optional cross-encoder rerank for higher precision.
    rer = _get_reranker()
    if rer is not None and hits:
        try:
            pairs = [(query, h["snippet"]) for h in hits]
            scores = rer.predict(pairs)
            for h, s in zip(hits, scores):
                h["score"] = float(s)
            hits.sort(key=lambda h: h["score"], reverse=True)
        except Exception as e:
            print(f"  [rag] rerank failed ({e}); using raw cosine ranking")

    return hits[:k]


# ── config helpers ───────────────────────────────────────────────────
# Keys whose cached singleton must be rebuilt when the value changes.
_EMBEDDER_KEYS = {"RAG_EMBED_MODEL", "RAG_OLLAMA_ENDPOINT",
                  "RAG_EMBED_BATCH", "RAG_EMBED_TIMEOUT"}
_RERANKER_KEYS = {"RAG_RERANKER_MODEL", "RAG_DEVICE"}
_COLLECTION_KEYS = {"RAG_COLLECTION"}


def configure(**kwargs) -> dict:
    """Update tunables at runtime. Returns the new effective config."""
    global _embed_model, _reranker, _collection
    g = globals()
    for key, value in kwargs.items():
        upper = key.upper()
        if upper in {
            "RAG_INDEX_PATHS", "RAG_EXCLUDE_GLOBS", "RAG_EMBED_MODEL",
            "RAG_OLLAMA_ENDPOINT", "RAG_EMBED_BATCH", "RAG_EMBED_TIMEOUT",
            "RAG_RERANKER_MODEL", "RAG_MAX_FILE_BYTES", "RAG_CHUNK_CHARS",
            "RAG_CHUNK_OVERLAP", "RAG_DEVICE", "RAG_COLLECTION",
        }:
            changed = g[upper] != value
            g[upper] = value
            # Invalidate cached singletons so the new value actually takes
            # effect — otherwise _get_embedder()/_get_reranker()/
            # _get_collection() keep returning objects built with the old
            # config and the confirmation to the user is a lie.
            if changed and upper in _EMBEDDER_KEYS:
                with _embedder_init_lock:
                    _embed_model = None
            if changed and upper in _RERANKER_KEYS:
                with _reranker_init_lock:
                    _reranker = None
            if changed and upper in _COLLECTION_KEYS:
                with _collection_init_lock:
                    _collection = None
    return current_config()


def current_config() -> dict:
    return {
        "RAG_INDEX_PATHS": list(RAG_INDEX_PATHS),
        "RAG_EXCLUDE_GLOBS": list(RAG_EXCLUDE_GLOBS),
        "RAG_EMBED_MODEL": RAG_EMBED_MODEL,
        "RAG_OLLAMA_ENDPOINT": RAG_OLLAMA_ENDPOINT,
        "RAG_EMBED_BATCH": RAG_EMBED_BATCH,
        "RAG_EMBED_TIMEOUT": RAG_EMBED_TIMEOUT,
        "RAG_RERANKER_MODEL": RAG_RERANKER_MODEL,
        "RAG_MAX_FILE_BYTES": RAG_MAX_FILE_BYTES,
        "RAG_CHUNK_CHARS": RAG_CHUNK_CHARS,
        "RAG_CHUNK_OVERLAP": RAG_CHUNK_OVERLAP,
        "RAG_DEVICE": RAG_DEVICE,
        "RAG_COLLECTION": RAG_COLLECTION,
        "chroma_path": _CHROMA_DIR,
    }


def status() -> dict:
    # Snapshot under the short-lived lock so concurrent mutations don't
    # tear the dict. Must stay fast — index_once() no longer holds _lock.
    with _lock:
        stats_snapshot = dict(_stats)
        last_full_scan_ts = _last_full_scan_ts
        last_error = _last_error
    return {
        "available": is_available(),
        "running": _indexer_thread is not None and _indexer_thread.is_alive(),
        "watchdog_active": _observer is not None,
        "last_full_scan_ts": last_full_scan_ts,
        "last_error": last_error,
        # Review 2026-10-04: indexing that is WAITING (EmbedDeferred) is not
        # "idle" - why it waits ('' = it is not) and when a deferred full
        # scan runs again (0.0 = none pending).
        "deferred": _deferral_logged[0],
        "retry_at": _scan_retry_at[0],
        **stats_snapshot,
        "config": current_config(),
    }


def collection_size() -> int:
    """Number of chunks currently stored. Best-effort; returns 0 on
    error so callers don't have to try/except themselves."""
    try:
        coll = _get_collection()
        return int(coll.count())
    except Exception:
        return 0


if __name__ == "__main__":  # pragma: no cover — smoke test
    print("rag_indexer available:", is_available())
    print("config:", current_config())
    if is_available():
        print("collection size:", collection_size())
