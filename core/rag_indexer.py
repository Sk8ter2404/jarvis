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
    RAG_EXCLUDE_GLOBS     — fnmatch patterns to skip (node_modules, .git, …)
    RAG_EMBED_MODEL       — Ollama embedding model name (default: nomic-embed-text)
    RAG_OLLAMA_ENDPOINT   — Ollama embeddings HTTP endpoint
    RAG_EMBED_BATCH       — chunks per HTTP batch (parallel POSTs)
    RAG_RERANKER_MODEL    — cross-encoder reranker id; "" disables rerank
    RAG_MAX_FILE_BYTES    — skip files larger than this (default 25 MB)
    RAG_CHUNK_CHARS       — chunk size in characters (default 1200)
    RAG_CHUNK_OVERLAP     — chunk overlap in characters (default 200)
    RAG_DEVICE            — "auto" | "cuda" | "cpu" for the reranker only
                            (embeddings run via Ollama regardless)
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
RAG_EXCLUDE_GLOBS: list[str] = [
    "*/.git/*", "*/node_modules/*", "*/__pycache__/*", "*/.venv/*",
    "*/venv/*", "*/dist/*", "*/build/*", "*/.cache/*", "*/.next/*",
    "*/Library/Caches/*", "*/AppData/Local/*", "*/AppData/Roaming/*",
    "*.tmp", "*.lock", "*.cache",
]
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
def _is_excluded(path: str) -> bool:
    norm = path.replace("\\", "/")
    for pat in RAG_EXCLUDE_GLOBS:
        if fnmatch.fnmatch(norm, pat):
            return True
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
        import chromadb  # noqa: F401
        return True
    except Exception:
        return False


def _device() -> str:
    if RAG_DEVICE in ("cpu", "cuda"):
        return RAG_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class _OllamaEmbedder:
    """Thin wrapper that mimics SentenceTransformer's .encode() interface
    but POSTs to Ollama's /api/embeddings. One request per chunk, but
    fanned out across a thread pool so a batch finishes in roughly
    `len(chunks) / batch_size` round-trip times instead of serial.

    Returns numpy float32 arrays so the rest of the indexer (which
    calls .tolist() before handing embeddings to Chroma) is unchanged.
    """

    def __init__(self, model: str, endpoint: str,
                 batch_size: int = 16, timeout: float = 30.0):
        self.model = model
        self.endpoint = endpoint
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)

    def _embed_one(self, text: str) -> list[float]:
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
               **_ignored):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype="float32") if convert_to_numpy else []

        n_workers = batch_size if batch_size else self.batch_size
        n_workers = max(1, min(n_workers, len(texts)))

        if n_workers == 1 or len(texts) == 1:
            vecs = [self._embed_one(t) for t in texts]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                vecs = list(pool.map(self._embed_one, texts))

        if normalize_embeddings:
            vecs = [self._normalise(v) for v in vecs]

        if convert_to_numpy:
            return np.asarray(vecs, dtype="float32")
        return vecs


def _ollama_reachable() -> bool:
    """Cheap one-shot reachability check for the Ollama endpoint.
    True iff a single embedding round-trip succeeds. Used only for
    logging / diagnostics; callers can still try and fail loudly."""
    try:
        _OllamaEmbedder(
            RAG_EMBED_MODEL, RAG_OLLAMA_ENDPOINT,
            batch_size=1, timeout=5.0,
        )._embed_one("ping")
        return True
    except Exception:
        return False


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
            if str(dev) == "cuda":
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
        # Prune excluded directories cheaply by filtering dirnames.
        keep = []
        for d in dirnames:
            full = os.path.join(dirpath, d).replace("\\", "/")
            if any(fnmatch.fnmatch(full + "/", p) or fnmatch.fnmatch(full, p)
                   for p in RAG_EXCLUDE_GLOBS):
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
    return len(chunks)


def index_once(progress: Optional[Callable[[str, int], None]] = None,
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
    for root in RAG_INDEX_PATHS:
        if not os.path.isdir(root):
            continue
        for path in _iter_files(root):
            if _stop_flag.is_set():
                break
            seen_files.add(_file_id(path))
            try:
                _index_file(path)
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
        if _stop_flag.is_set():
            break

    # Garbage-collect: drop chunks whose file_id no longer appears on disk.
    try:
        coll = _get_collection()
        existing = coll.get(include=["metadatas"])
        metas = existing.get("metadatas") or []
        ids = existing.get("ids") or []
        stale_ids = [
            cid for cid, m in zip(ids, metas)
            if isinstance(m, dict)
            and m.get("file_id") not in seen_files
            and m.get("path")
            and not os.path.isfile(str(m.get("path", "")))
        ]
        if stale_ids:
            coll.delete(ids=stale_ids)
    except Exception:
        pass

    with _lock:
        _last_full_scan_ts = time.time()
        return {
            "ok": True,
            "files_seen": files_seen_count,
            "files_indexed_total": _stats["files_indexed"],
            "chunks_written_total": _stats["chunks_written"],
            "errors": _stats["errors"],
            "ts": _last_full_scan_ts,
        }


# ── watchdog daemon ──────────────────────────────────────────────────
def _drain_event_queue() -> None:
    """Single-threaded reindex worker — drains _event_q. Coalesces
    multiple events for the same path within a short window into one
    re-index call (debounce ~2 s)."""
    pending: dict[str, float] = {}
    while not _stop_flag.is_set():
        try:
            path = _event_q.get(timeout=0.1)
            pending[path] = time.time() + 2.0
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
            except Exception as e:
                global _last_error
                with _lock:
                    _stats["errors"] += 1
                    _last_error = f"reindex({p}): {e}"


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
            path = getattr(event, "dest_path", "") or event.src_path
            if not path or not _supported(path) or _is_excluded(path):
                return
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
    if not _ollama_reachable():
        print(f"  [rag] WARNING: Ollama embeddings endpoint "
              f"{RAG_OLLAMA_ENDPOINT} unreachable — indexing will fail "
              f"until `ollama serve` is running with model "
              f"'{RAG_EMBED_MODEL}' pulled")
    global _indexer_thread
    if _indexer_thread is not None and _indexer_thread.is_alive():
        return True
    _stop_flag.clear()

    def _bg():
        if initial_scan:
            try:
                summary = index_once()
                print(f"  [rag] initial scan: {summary}")
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
    try:
        qvec = embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
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
