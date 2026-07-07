"""Thorough unit tests for core.rag_indexer — the personal-files RAG indexer.

The real module lazily imports heavy, optional deps (chromadb, numpy, torch,
sentence_transformers, pypdf, python-docx, watchdog) and reaches Ollama over
HTTP for embeddings. To stay deterministic and fully offline we inject light
fakes into ``sys.modules`` for every optional dep BEFORE importing the target,
so the suite behaves identically whether or not those packages happen to be
installed on the host and NEVER loads a real model or opens a socket. All disk
writes are redirected at a per-test tempdir; threading/sleep is avoided by
driving the worker functions directly rather than spawning the daemon.

Two paths are exercised:
  * the with-deps path — chromadb + a fake embedder/reranker are present and
    indexing/querying/ranking/persistence-stamp logic runs end to end;
  * the degraded path — ``is_available()`` is forced False (chromadb import
    fails) and every public entrypoint returns its safe empty/early value.

stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest import mock


class _LockSetsGlobal:
    """A context-manager stand-in for one of the module's init locks that
    populates a module global on __enter__. Used to exercise the second read of
    each double-checked-lock singleton getter: the outer check sees None, then
    holding the (fake) lock the global is already set — emulating another thread
    that won the init race — so the inner re-check returns it without rebuilding.
    """

    def __init__(self, attr, value):
        self._attr = attr
        self._value = value

    def __enter__(self):
        import core.rag_indexer as _rag
        setattr(_rag, self._attr, self._value)
        return self

    def __exit__(self, *exc):
        return False


# ───────────────────────── fake optional deps ──────────────────────────
# These are installed into sys.modules before the target module is imported
# so its lazy `import chromadb` / `import numpy` / ... resolve to our fakes.

class _FakeNdArray:
    """Minimal stand-in for a 2-D numpy array as used by the indexer:
    supports .tolist(), len(), and integer indexing (qvec = encode(...)[0])."""

    def __init__(self, rows):
        self._rows = [list(r) for r in rows]

    def tolist(self):
        return [list(r) for r in self._rows]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return _FakeVec(self._rows[i])

    def __eq__(self, other):  # only used in assertions/debugging
        return isinstance(other, _FakeNdArray) and other._rows == self._rows


class _FakeVec:
    """A single embedding row — needs .tolist() for the search query path."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _make_fake_numpy():
    np = types.ModuleType("numpy")

    def asarray(rows, dtype=None):
        rows = list(rows)
        return _FakeNdArray(rows)

    def zeros(shape, dtype=None):
        # Only ever called as zeros((0, 0)) → an empty 2-D array.
        n = shape[0] if isinstance(shape, (tuple, list)) else shape
        return _FakeNdArray([[] for _ in range(n)])

    np.asarray = asarray
    np.zeros = zeros
    np.float32 = "float32"
    return np


class _FakeCudaUnavailable:
    @staticmethod
    def is_available():
        return False


class _FakeCudaAvailable:
    @staticmethod
    def is_available():
        return True


def _make_fake_torch(cuda_available=False):
    torch = types.ModuleType("torch")
    torch.cuda = _FakeCudaAvailable() if cuda_available else _FakeCudaUnavailable()
    return torch


def _make_fake_chromadb():
    """A chromadb fake whose PersistentClient/collection can be configured per
    test. By default it has no constructor side effects; tests reach in through
    the module-level singletons or patch _get_collection directly. This exists
    mostly so the bare `import chromadb` in is_available()/_get_collection
    succeeds; the behaviour-bearing fakes live in _FakeCollection below."""
    chromadb = types.ModuleType("chromadb")

    class _PersistentClient:
        last_instance = None

        def __init__(self, path=None):
            self.path = path
            self._collections = {}
            _PersistentClient.last_instance = self

        def get_or_create_collection(self, name=None, metadata=None):
            coll = self._collections.get(name)
            if coll is None:
                coll = _FakeCollection(name=name, metadata=dict(metadata or {}))
                self._collections[name] = coll
            return coll

        def delete_collection(self, name=None):
            self._collections.pop(name, None)

    chromadb.PersistentClient = _PersistentClient
    return chromadb


class _FakeCollection:
    """In-memory Chroma collection good enough for the indexer's calls:
    add / get / delete (by ids or where=file_id) / query / count / modify."""

    def __init__(self, name="c", metadata=None):
        self.name = name
        self.metadata = dict(metadata or {})
        # id -> (document, metadata)
        self._store = {}
        self.added_calls = []
        self.query_return = None  # tests may pre-seed a query result
        self.raise_on_add = False
        self.raise_on_query = False
        self.raise_on_get = False

    def modify(self, metadata=None):
        if metadata is not None:
            self.metadata = dict(metadata)

    def add(self, ids=None, embeddings=None, documents=None, metadatas=None):
        if self.raise_on_add:
            raise RuntimeError("add boom")
        self.added_calls.append(
            {"ids": list(ids or []), "embeddings": embeddings,
             "documents": list(documents or []), "metadatas": list(metadatas or [])}
        )
        for i, _id in enumerate(ids or []):
            self._store[_id] = (documents[i], metadatas[i])

    def get(self, where=None, include=None, limit=None, ids=None):
        if self.raise_on_get:
            raise RuntimeError("get boom")
        items = list(self._store.items())
        if where and "file_id" in where:
            fid = where["file_id"]
            items = [(i, v) for i, v in items if v[1].get("file_id") == fid]
        if limit is not None:
            items = items[:limit]
        out_ids = [i for i, _ in items]
        out_docs = [v[0] for _, v in items]
        out_metas = [v[1] for _, v in items]
        return {"ids": out_ids, "documents": out_docs, "metadatas": out_metas}

    def delete(self, where=None, ids=None):
        if ids is not None:
            for i in ids:
                self._store.pop(i, None)
            return
        if where and "file_id" in where:
            fid = where["file_id"]
            doomed = [i for i, v in self._store.items() if v[1].get("file_id") == fid]
            for i in doomed:
                self._store.pop(i, None)

    def query(self, query_embeddings=None, n_results=10, include=None):
        if self.raise_on_query:
            raise RuntimeError("query boom")
        if self.query_return is not None:
            return self.query_return
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def count(self):
        return len(self._store)


class _FakeEmbedder:
    """Stands in for _OllamaEmbedder / SentenceTransformer. Returns a fixed
    small vector per text and records the texts it was asked to encode."""

    def __init__(self, dim=3, fail=False):
        self.dim = dim
        self.fail = fail
        self.calls = []

    def encode(self, texts, batch_size=None, convert_to_numpy=True,
               show_progress_bar=False, normalize_embeddings=False, **_):
        if self.fail:
            raise urllib.error.URLError("ollama down")
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        self.calls.append(texts)
        rows = [[float(len(t) % 7) + 0.1 * j for j in range(self.dim)] for t in texts]
        return _FakeNdArray(rows)


class _FakeReranker:
    def __init__(self, scores=None, fail=False):
        self._scores = scores
        self.fail = fail
        self.predicted = []

    def predict(self, pairs):
        if self.fail:
            raise RuntimeError("rerank boom")
        self.predicted.append(list(pairs))
        if self._scores is not None:
            return list(self._scores)[:len(pairs)]
        # Default: longer snippet → higher score (deterministic ordering).
        return [float(len(p[1])) for p in pairs]


# ── fakes for the optional deps ──────────────────────────────────────────
# core.rag_indexer imports every heavy dep LAZILY (inside functions:
# `import chromadb` in is_available()/_get_collection, `import numpy` inside
# encode(), `import torch` in _device(), `from sentence_transformers import
# CrossEncoder` in _get_reranker()). So the target can be imported with the
# real environment present, and the fakes only need to exist WHILE a test
# runs. We therefore build the fake-module map once but DO NOT write it into
# sys.modules at module level — that previously leaked a fake numpy (which has
# no `.array`) process-wide during test discovery and broke every later test
# using real numpy. Instead _RagBase.setUp installs them via
# mock.patch.dict(sys.modules, _FAKE_MODULES) so they are scoped to each test
# and restored on teardown.


def _build_fake_modules():
    st = types.ModuleType("sentence_transformers")
    st.CrossEncoder = lambda *a, **k: _FakeReranker()
    return {
        "numpy": _make_fake_numpy(),
        "torch": _make_fake_torch(cuda_available=False),
        "chromadb": _make_fake_chromadb(),
        "sentence_transformers": st,
    }


_FAKE_MODULES = _build_fake_modules()

# Imported with the real environment; lazy deps are injected per-test below.
from core import rag_indexer as rag  # noqa: E402


# ───────────────────────────── base fixture ────────────────────────────
class _RagBase(unittest.TestCase):
    """Redirects all on-disk paths at a tempdir and resets module singletons /
    tunables / stats between tests so each runs in isolation."""

    # Tunables we mutate and must restore so cross-test leakage can't happen.
    _SAVED_TUNABLES = (
        "RAG_INDEX_PATHS", "RAG_EXCLUDE_GLOBS", "RAG_EMBED_MODEL",
        "RAG_OLLAMA_ENDPOINT", "RAG_EMBED_BATCH", "RAG_EMBED_TIMEOUT",
        "RAG_RERANKER_MODEL", "RAG_MAX_FILE_BYTES", "RAG_CHUNK_CHARS",
        "RAG_CHUNK_OVERLAP", "RAG_DEVICE", "RAG_COLLECTION",
        "RAG_REINDEX_ON_MODEL_CHANGE",
    )

    def setUp(self):
        # Install fake optional deps (numpy/torch/chromadb/sentence_transformers)
        # into sys.modules ONLY for the duration of this test, then restore.
        # Scoped here (not at module level) so a fake numpy can never leak into
        # the wider suite and shadow the real one. A fresh map per test keeps
        # the per-test stateful fakes (PersistentClient.last_instance, etc.)
        # from bleeding across tests.
        fake_modules = _build_fake_modules()
        mods_patch = mock.patch.dict(sys.modules, fake_modules)
        mods_patch.start()
        self.addCleanup(mods_patch.stop)

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

        # Redirect the data/chroma/state paths.
        for attr, val in (
            ("_DATA_DIR", os.path.join(self.tmp, "data")),
            ("_CHROMA_DIR", os.path.join(self.tmp, "data", "rag_chroma")),
            ("_STATE_PATH", os.path.join(self.tmp, "data", "rag_state.json")),
        ):
            p = mock.patch.object(rag, attr, val)
            p.start()
            self.addCleanup(p.stop)

        # Snapshot + restore tunables.
        self._saved = {k: getattr(rag, k) for k in self._SAVED_TUNABLES}
        self.addCleanup(self._restore_tunables)

        # Silence the module's diagnostic prints. Besides cutting noise this
        # avoids a Windows-only UnicodeEncodeError: several log lines contain a
        # '→' arrow which the legacy cp1252 console codec can't encode, and the
        # module's broad except-clauses would otherwise mask the real branch
        # under test. Tests never assert on stdout.
        pp = mock.patch("builtins.print", lambda *a, **k: None)
        pp.start()
        self.addCleanup(pp.stop)

        # Reset singletons + shared state.
        self._reset_module_state()
        self.addCleanup(self._reset_module_state)

        # Default: a fresh empty index folder, no real folders walked.
        rag.RAG_INDEX_PATHS = []
        # The system tempdir lives under .../AppData/Local/Temp on Windows,
        # which the production RAG_EXCLUDE_GLOBS deliberately skips. For the
        # filesystem-walk tests we want our tmp files to be visible, so trim
        # the globs to the structural ones the tests actually assert on (the
        # AppData/cache patterns are an environmental artifact here).
        rag.RAG_EXCLUDE_GLOBS = [
            "*/.git/*", "*/node_modules/*", "*/__pycache__/*",
            "*/.venv/*", "*/venv/*", "*/dist/*", "*/build/*",
            "*.tmp", "*.lock", "*.cache",
        ]

    def _restore_tunables(self):
        for k, v in self._saved.items():
            setattr(rag, k, v)

    def _reset_module_state(self):
        rag._chroma_client = None
        rag._collection = None
        rag._embed_model = None
        rag._reranker = None
        rag._observer = None
        rag._indexer_thread = None
        rag._stop_flag.clear()
        rag._last_full_scan_ts = 0.0
        rag._last_error = ""
        for k in rag._stats:
            rag._stats[k] = 0
        # Drain the event queue.
        try:
            while True:
                rag._event_q.get_nowait()
        except Exception:
            pass

    # helpers -----------------------------------------------------------
    def _write(self, relpath, text):
        full = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        return full

    def _install_collection(self, coll=None):
        coll = coll or _FakeCollection(metadata={"embed_model": rag.RAG_EMBED_MODEL})
        rag._collection = coll
        return coll

    def _install_embedder(self, emb=None):
        emb = emb or _FakeEmbedder()
        rag._embed_model = emb
        return emb


# ─────────────────────────── pure helpers ──────────────────────────────
class ExclusionAndSupportTests(_RagBase):
    def test_is_excluded_matches_glob(self):
        self.assertTrue(rag._is_excluded(r"C:\proj\node_modules\x.js"))
        self.assertTrue(rag._is_excluded("/home/u/.git/config"))
        self.assertTrue(rag._is_excluded("/tmp/foo.tmp"))

    def test_is_excluded_false_for_plain_file(self):
        self.assertFalse(rag._is_excluded("/home/u/Documents/notes.md"))

    def test_supported_extensions(self):
        self.assertTrue(rag._supported("a.txt"))
        self.assertTrue(rag._supported("A.MD"))
        self.assertTrue(rag._supported("x.pdf"))
        self.assertTrue(rag._supported("y.docx"))
        self.assertTrue(rag._supported("code.py"))

    def test_unsupported_extensions(self):
        self.assertFalse(rag._supported("image.png"))
        self.assertFalse(rag._supported("movie.mp4"))
        self.assertFalse(rag._supported("archive.zip"))
        self.assertFalse(rag._supported("noext"))

    def test_file_id_is_stable_sha1(self):
        a = rag._file_id("foo.txt")
        b = rag._file_id("foo.txt")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 40)  # sha1 hex
        self.assertNotEqual(a, rag._file_id("bar.txt"))


class ChunkTests(_RagBase):
    def test_empty_text_no_chunks(self):
        self.assertEqual(rag._chunk("", 100, 10), [])

    def test_short_text_one_chunk(self):
        self.assertEqual(rag._chunk("hello world", 100, 10), ["hello world"])

    def test_long_text_multiple_chunks_with_overlap(self):
        text = "x" * 500
        chunks = rag._chunk(text, 100, 20)
        self.assertGreater(len(chunks), 1)
        # Every chunk is within the size budget after stripping.
        for c in chunks:
            self.assertLessEqual(len(c), 100)

    def test_prefers_paragraph_break(self):
        # A double-newline in the tail half of the window should be the cut.
        first = "a" * 60
        second = "b" * 60
        text = first + "\n\n" + second
        chunks = rag._chunk(text, 100, 10)
        # The first chunk should end at the paragraph break, i.e. be just "a"*60.
        self.assertEqual(chunks[0], first)

    def test_single_newline_fallback_cut(self):
        # No blank line, but a single newline in the tail half is used.
        text = ("a" * 60) + "\n" + ("b" * 60)
        chunks = rag._chunk(text, 100, 10)
        self.assertEqual(chunks[0], "a" * 60)

    def test_normalises_crlf(self):
        chunks = rag._chunk("line1\r\nline2", 100, 10)
        self.assertIn("line1\nline2", chunks[0])

    def test_whitespace_only_chunk_dropped(self):
        # Pure whitespace strips to empty and contributes no chunk.
        self.assertEqual(rag._chunk("   \n  \n   ", 100, 10), [])


# ───────────────────────────── _read_text ──────────────────────────────
class ReadTextTests(_RagBase):
    def test_reads_plain_text(self):
        p = self._write("notes.txt", "hello sir")
        self.assertEqual(rag._read_text(p), "hello sir")

    def test_unknown_extension_returns_empty(self):
        p = self._write("thing.bin", "data")
        # .bin isn't in any ext set → falls through to the trailing return "".
        self.assertEqual(rag._read_text(p), "")

    def test_text_read_error_returns_empty(self):
        # open() raising is swallowed → "".
        with mock.patch("builtins.open", side_effect=OSError("nope")):
            self.assertEqual(rag._read_text(os.path.join(self.tmp, "x.txt")), "")

    def test_pdf_missing_dep_returns_empty(self):
        p = self._write("doc.pdf", "%PDF-stub")
        with mock.patch.dict(sys.modules, {"pypdf": None}):
            self.assertEqual(rag._read_text(p), "")

    def test_pdf_extracts_pages(self):
        p = self._write("doc.pdf", "stub")

        class _Page:
            def __init__(self, t):
                self._t = t

            def extract_text(self):
                return self._t

        class _Reader:
            def __init__(self, path):
                self.pages = [_Page("page one"), _Page("page two")]

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _Reader
        with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
            out = rag._read_text(p)
        self.assertEqual(out, "page one\npage two")

    def test_pdf_page_extract_error_skips_page(self):
        p = self._write("doc.pdf", "stub")

        class _GoodPage:
            def extract_text(self):
                return "good"

        class _BadPage:
            def extract_text(self):
                raise ValueError("bad page")

        class _Reader:
            def __init__(self, path):
                self.pages = [_GoodPage(), _BadPage()]

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _Reader
        with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
            out = rag._read_text(p)
        self.assertEqual(out, "good")

    def test_pdf_reader_construct_error_returns_empty(self):
        p = self._write("doc.pdf", "stub")

        class _Reader:
            def __init__(self, path):
                raise RuntimeError("corrupt pdf")

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _Reader
        with mock.patch.dict(sys.modules, {"pypdf": fake_pypdf}):
            self.assertEqual(rag._read_text(p), "")

    def test_docx_missing_dep_returns_empty(self):
        p = self._write("doc.docx", "stub")
        with mock.patch.dict(sys.modules, {"docx": None}):
            self.assertEqual(rag._read_text(p), "")

    def test_docx_extracts_paragraphs(self):
        p = self._write("doc.docx", "stub")

        class _Para:
            def __init__(self, t):
                self.text = t

        class _Doc:
            def __init__(self):
                self.paragraphs = [_Para("para a"), _Para("para b")]

        fake_docx = types.ModuleType("docx")
        fake_docx.Document = lambda path: _Doc()
        with mock.patch.dict(sys.modules, {"docx": fake_docx}):
            out = rag._read_text(p)
        self.assertEqual(out, "para a\npara b")

    def test_docx_document_error_returns_empty(self):
        p = self._write("doc.docx", "stub")

        def _boom(path):
            raise RuntimeError("bad docx")

        fake_docx = types.ModuleType("docx")
        fake_docx.Document = _boom
        with mock.patch.dict(sys.modules, {"docx": fake_docx}):
            self.assertEqual(rag._read_text(p), "")


# ─────────────────────── is_available / _device ────────────────────────
class AvailabilityTests(_RagBase):
    def test_is_available_true_when_chromadb_imports(self):
        # The fake chromadb is registered → importable → True.
        self.assertTrue(rag.is_available())

    def test_is_available_false_when_chromadb_missing(self):
        with mock.patch.dict(sys.modules, {"chromadb": None}):
            self.assertFalse(rag.is_available())

    def test_device_explicit_cpu(self):
        rag.RAG_DEVICE = "cpu"
        self.assertEqual(rag._device(), "cpu")

    def test_device_explicit_cuda(self):
        rag.RAG_DEVICE = "cuda"
        self.assertEqual(rag._device(), "cuda")

    def test_device_auto_prefers_cuda_when_available(self):
        rag.RAG_DEVICE = "auto"
        with mock.patch.dict(sys.modules, {"torch": _make_fake_torch(cuda_available=True)}):
            self.assertEqual(rag._device(), "cuda")

    def test_device_auto_falls_back_to_cpu(self):
        rag.RAG_DEVICE = "auto"
        with mock.patch.dict(sys.modules, {"torch": _make_fake_torch(cuda_available=False)}):
            self.assertEqual(rag._device(), "cpu")

    def test_device_auto_cpu_when_torch_missing(self):
        rag.RAG_DEVICE = "auto"
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(rag._device(), "cpu")


# ─────────────────────────── _OllamaEmbedder ───────────────────────────
class OllamaEmbedderTests(_RagBase):
    def setUp(self):
        super().setUp()
        # _embed_one best-effort calls core.gpu_state.log_gpu_state, which on a
        # real host shells out to nvidia-smi. Swap in a no-op so the embedder
        # unit tests stay offline and never spawn a subprocess.
        fake_gpu = types.ModuleType("core.gpu_state")
        fake_gpu.log_gpu_state = lambda *a, **k: None
        p = mock.patch.dict(sys.modules, {"core.gpu_state": fake_gpu})
        p.start()
        self.addCleanup(p.stop)

    def _patch_urlopen(self, payload=None, error=None):
        """Patch urllib so no real socket is opened."""
        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                import json as _json
                return _json.dumps(self._body).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _fake_urlopen(req, timeout=None):
            if error is not None:
                raise error
            return _Resp(payload if payload is not None else {"embedding": [1.0, 2.0, 2.0]})

        return mock.patch.object(rag.urllib.request, "urlopen", _fake_urlopen)

    def test_embed_one_returns_floats(self):
        emb = rag._OllamaEmbedder("m", "http://x", batch_size=2, timeout=1.0)
        with self._patch_urlopen(payload={"embedding": [1, 2, 3]}):
            out = emb._embed_one("hello")
        self.assertEqual(out, [1.0, 2.0, 3.0])

    def test_embed_one_empty_embedding_raises(self):
        emb = rag._OllamaEmbedder("m", "http://x")
        with self._patch_urlopen(payload={"embedding": []}):
            with self.assertRaises(RuntimeError):
                emb._embed_one("hello")

    def test_batch_size_floored_to_one(self):
        emb = rag._OllamaEmbedder("m", "http://x", batch_size=0)
        self.assertEqual(emb.batch_size, 1)

    def test_normalise_unit_length(self):
        emb = rag._OllamaEmbedder("m", "http://x")
        out = emb._normalise([3.0, 4.0])  # length 5 → (0.6, 0.8)
        self.assertAlmostEqual(out[0], 0.6, places=6)
        self.assertAlmostEqual(out[1], 0.8, places=6)

    def test_normalise_zero_vector_returned_asis(self):
        emb = rag._OllamaEmbedder("m", "http://x")
        self.assertEqual(emb._normalise([0.0, 0.0]), [0.0, 0.0])

    def test_encode_str_input_single_path(self):
        emb = rag._OllamaEmbedder("m", "http://x", batch_size=4)
        with self._patch_urlopen(payload={"embedding": [1.0, 0.0, 0.0]}):
            arr = emb.encode("hello", convert_to_numpy=True)
        self.assertEqual(arr.tolist(), [[1.0, 0.0, 0.0]])

    def test_encode_empty_list_numpy(self):
        emb = rag._OllamaEmbedder("m", "http://x")
        arr = emb.encode([], convert_to_numpy=True)
        self.assertEqual(arr.tolist(), [])

    def test_encode_empty_list_no_numpy(self):
        emb = rag._OllamaEmbedder("m", "http://x")
        self.assertEqual(emb.encode([], convert_to_numpy=False), [])

    def test_encode_multi_uses_thread_pool(self):
        emb = rag._OllamaEmbedder("m", "http://x", batch_size=4)
        with self._patch_urlopen(payload={"embedding": [0.5, 0.5, 0.5]}):
            arr = emb.encode(["a", "b", "c"], convert_to_numpy=True)
        self.assertEqual(len(arr.tolist()), 3)

    def test_encode_normalize_applied(self):
        emb = rag._OllamaEmbedder("m", "http://x", batch_size=1)
        with self._patch_urlopen(payload={"embedding": [3.0, 4.0]}):
            arr = emb.encode(["a"], convert_to_numpy=True, normalize_embeddings=True)
        row = arr.tolist()[0]
        self.assertAlmostEqual(row[0], 0.6, places=6)
        self.assertAlmostEqual(row[1], 0.8, places=6)

    def test_encode_no_numpy_returns_list(self):
        emb = rag._OllamaEmbedder("m", "http://x", batch_size=1)
        with self._patch_urlopen(payload={"embedding": [1.0, 1.0]}):
            out = emb.encode(["a"], convert_to_numpy=False)
        self.assertEqual(out, [[1.0, 1.0]])

    def test_embed_one_gpu_snapshot_failure_swallowed(self):
        # The gpu_state import/log is best-effort; make it raise and confirm
        # the embedding still succeeds.
        emb = rag._OllamaEmbedder("m", "http://x")
        fake_gpu = types.ModuleType("core.gpu_state")

        def _boom(model):
            raise RuntimeError("no smi")

        fake_gpu.log_gpu_state = _boom
        with mock.patch.dict(sys.modules, {"core.gpu_state": fake_gpu}):
            with self._patch_urlopen(payload={"embedding": [1.0]}):
                self.assertEqual(emb._embed_one("x"), [1.0])

    def test_ollama_reachable_true(self):
        with self._patch_urlopen(payload={"embedding": [1.0]}):
            self.assertTrue(rag._ollama_reachable())

    def test_ollama_reachable_false_on_error(self):
        with self._patch_urlopen(error=urllib.error.URLError("down")):
            self.assertFalse(rag._ollama_reachable())


# ─────────────────────── lazy init singletons ──────────────────────────
class GetEmbedderTests(_RagBase):
    def test_creates_and_caches(self):
        e1 = rag._get_embedder()
        e2 = rag._get_embedder()
        self.assertIs(e1, e2)
        self.assertIsInstance(e1, rag._OllamaEmbedder)

    def test_returns_existing_without_lock(self):
        sentinel = object()
        rag._embed_model = sentinel
        self.assertIs(rag._get_embedder(), sentinel)

    def test_double_checked_lock_returns_racer_value(self):
        # Simulate another thread winning the init race: the outer check sees
        # None, but by the time we hold the lock the global is populated, so the
        # inner re-check returns that value without building a second embedder.
        sentinel = object()
        self.assertIsNone(rag._embed_model)
        with mock.patch.object(rag, "_embedder_init_lock",
                               _LockSetsGlobal("_embed_model", sentinel)):
            self.assertIs(rag._get_embedder(), sentinel)


class GetRerankerTests(_RagBase):
    def test_disabled_when_model_blank(self):
        rag.RAG_RERANKER_MODEL = ""
        self.assertIsNone(rag._get_reranker())

    def test_returns_cached(self):
        sentinel = object()
        rag._reranker = sentinel
        self.assertIs(rag._get_reranker(), sentinel)

    def test_double_checked_lock_returns_racer_value(self):
        # Model configured (so the outer guard doesn't short-circuit on a blank
        # name) and _reranker None at the outer check, but populated by the time
        # we hold the lock → the inner re-check returns it without loading a
        # CrossEncoder.
        rag.RAG_RERANKER_MODEL = "some/model"
        sentinel = object()
        self.assertIsNone(rag._reranker)
        with mock.patch.object(rag, "_reranker_init_lock",
                               _LockSetsGlobal("_reranker", sentinel)):
            self.assertIs(rag._get_reranker(), sentinel)

    def test_missing_sentence_transformers_returns_none(self):
        rag.RAG_RERANKER_MODEL = "some/model"
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            self.assertIsNone(rag._get_reranker())

    def test_loads_cross_encoder(self):
        rag.RAG_RERANKER_MODEL = "some/model"
        rag.RAG_DEVICE = "cpu"
        marker = _FakeReranker()
        st = types.ModuleType("sentence_transformers")
        st.CrossEncoder = lambda model, device=None: marker
        with mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            self.assertIs(rag._get_reranker(), marker)

    def test_cuda_oom_falls_back_to_cpu(self):
        rag.RAG_RERANKER_MODEL = "some/model"
        rag.RAG_DEVICE = "cuda"
        cpu_marker = _FakeReranker()
        attempts = {"n": 0}

        def _ctor(model, device=None):
            attempts["n"] += 1
            if device == "cuda":
                raise RuntimeError("CUDA OOM")
            return cpu_marker

        st = types.ModuleType("sentence_transformers")
        st.CrossEncoder = _ctor
        with mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            self.assertIs(rag._get_reranker(), cpu_marker)
        self.assertEqual(attempts["n"], 2)

    def test_cuda_then_cpu_both_fail_returns_none(self):
        rag.RAG_RERANKER_MODEL = "some/model"
        rag.RAG_DEVICE = "cuda"

        def _ctor(model, device=None):
            raise RuntimeError("boom-%s" % device)

        st = types.ModuleType("sentence_transformers")
        st.CrossEncoder = _ctor
        with mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            self.assertIsNone(rag._get_reranker())

    def test_cpu_device_load_failure_returns_none(self):
        # dev != cuda branch: a CPU load failure just disables rerank.
        rag.RAG_RERANKER_MODEL = "some/model"
        rag.RAG_DEVICE = "cpu"

        def _ctor(model, device=None):
            raise RuntimeError("no weights")

        st = types.ModuleType("sentence_transformers")
        st.CrossEncoder = _ctor
        with mock.patch.dict(sys.modules, {"sentence_transformers": st}):
            self.assertIsNone(rag._get_reranker())


# ─────────────────────────── _get_collection ───────────────────────────
class GetCollectionTests(_RagBase):
    def test_returns_cached(self):
        sentinel = object()
        rag._collection = sentinel
        self.assertIs(rag._get_collection(), sentinel)

    def test_double_checked_lock_returns_racer_value(self):
        # _collection None at the outer check but set by the time the lock is
        # held → the inner re-check returns it without importing chromadb or
        # touching disk (the race-loser fast path).
        sentinel = object()
        self.assertIsNone(rag._collection)
        with mock.patch.object(rag, "_collection_init_lock",
                               _LockSetsGlobal("_collection", sentinel)):
            self.assertIs(rag._get_collection(), sentinel)

    def test_creates_with_metadata_and_stamps_when_unstamped(self):
        # Fresh collection comes back with no embed_model → it gets stamped.
        coll = rag._get_collection()
        self.assertIsNotNone(coll)
        self.assertEqual(coll.metadata.get("embed_model"), rag.RAG_EMBED_MODEL)
        # Chroma dir created on disk.
        self.assertTrue(os.path.isdir(rag._CHROMA_DIR))

    def test_stamp_modify_failure_is_swallowed(self):
        # If modify() raises, _get_collection still returns the collection.
        class _Coll(_FakeCollection):
            def modify(self, metadata=None):
                raise RuntimeError("modify denied")

        # Pre-seed the client to hand back our raising collection.
        chromadb = sys.modules["chromadb"]
        orig = chromadb.PersistentClient.get_or_create_collection

        def _get_or_create(self, name=None, metadata=None):
            return _Coll(name=name, metadata={})  # unstamped → triggers modify

        chromadb.PersistentClient.get_or_create_collection = _get_or_create
        try:
            coll = rag._get_collection()
            self.assertIsInstance(coll, _Coll)
        finally:
            chromadb.PersistentClient.get_or_create_collection = orig

    def test_model_mismatch_keeps_data_by_default(self):
        # Stored stamp differs and RAG_REINDEX_ON_MODEL_CHANGE is False / no
        # force → the collection is preserved (not dropped).
        rag.RAG_EMBED_MODEL = "new-model"
        rag.RAG_REINDEX_ON_MODEL_CHANGE = False
        chromadb = sys.modules["chromadb"]
        orig = chromadb.PersistentClient.get_or_create_collection
        existing = _FakeCollection(metadata={"embed_model": "old-model"})

        def _get_or_create(self, name=None, metadata=None):
            return existing

        chromadb.PersistentClient.get_or_create_collection = _get_or_create
        try:
            coll = rag._get_collection()
            self.assertIs(coll, existing)  # same object → not rebuilt
        finally:
            chromadb.PersistentClient.get_or_create_collection = orig

    def test_model_mismatch_with_force_drops_and_rebuilds(self):
        rag.RAG_EMBED_MODEL = "new-model"
        chromadb = sys.modules["chromadb"]
        orig_get = chromadb.PersistentClient.get_or_create_collection
        orig_del = chromadb.PersistentClient.delete_collection
        old = _FakeCollection(metadata={"embed_model": "old-model"})
        old._store["x:0"] = ("doc", {"file_id": "x"})
        rebuilt = _FakeCollection(metadata={"embed_model": "new-model"})
        seq = [old, rebuilt]
        deleted = {"n": 0}

        def _get_or_create(self, name=None, metadata=None):
            return seq.pop(0)

        def _delete(self, name=None):
            deleted["n"] += 1

        chromadb.PersistentClient.get_or_create_collection = _get_or_create
        chromadb.PersistentClient.delete_collection = _delete
        try:
            coll = rag._get_collection(force_reindex=True)
            self.assertIs(coll, rebuilt)
            self.assertEqual(deleted["n"], 1)
        finally:
            chromadb.PersistentClient.get_or_create_collection = orig_get
            chromadb.PersistentClient.delete_collection = orig_del

    def test_force_reindex_count_error_defaults_zero(self):
        # count() raising during the force-drop path is tolerated.
        rag.RAG_EMBED_MODEL = "new-model"
        chromadb = sys.modules["chromadb"]
        orig_get = chromadb.PersistentClient.get_or_create_collection
        orig_del = chromadb.PersistentClient.delete_collection

        class _BadCount(_FakeCollection):
            def count(self):
                raise RuntimeError("count boom")

        old = _BadCount(metadata={"embed_model": "old-model"})
        rebuilt = _FakeCollection(metadata={"embed_model": "new-model"})
        seq = [old, rebuilt]
        chromadb.PersistentClient.get_or_create_collection = (
            lambda self, name=None, metadata=None: seq.pop(0))
        chromadb.PersistentClient.delete_collection = (
            lambda self, name=None: None)
        try:
            coll = rag._get_collection(force_reindex=True)
            self.assertIs(coll, rebuilt)
        finally:
            chromadb.PersistentClient.get_or_create_collection = orig_get
            chromadb.PersistentClient.delete_collection = orig_del

    def test_migration_check_outer_exception_swallowed(self):
        # If reading .metadata blows up entirely, the except at the bottom of
        # _get_collection catches it and returns the collection anyway.
        class _Weird(_FakeCollection):
            @property
            def metadata(self):
                raise RuntimeError("meta explode")

            @metadata.setter
            def metadata(self, v):
                pass

        chromadb = sys.modules["chromadb"]
        orig = chromadb.PersistentClient.get_or_create_collection
        weird = _Weird(metadata={})
        chromadb.PersistentClient.get_or_create_collection = (
            lambda self, name=None, metadata=None: weird)
        try:
            coll = rag._get_collection()
            self.assertIs(coll, weird)
        finally:
            chromadb.PersistentClient.get_or_create_collection = orig


# ───────────────────────── filesystem walking ──────────────────────────
class IterFilesTests(_RagBase):
    def test_yields_supported_files(self):
        self._write("root/a.txt", "hi")
        self._write("root/b.md", "yo")
        self._write("root/pic.png", "binary")  # unsupported → skipped
        root = os.path.join(self.tmp, "root")
        got = sorted(os.path.basename(p) for p in rag._iter_files(root))
        self.assertEqual(got, ["a.txt", "b.md"])

    def test_prunes_excluded_dirs(self):
        self._write("root/keep.txt", "hi")
        self._write("root/node_modules/dep.js", "x")
        self._write("root/.git/config", "x")
        root = os.path.join(self.tmp, "root")
        got = [os.path.basename(p) for p in rag._iter_files(root)]
        self.assertEqual(got, ["keep.txt"])

    def test_descends_into_kept_subdir(self):
        # A non-excluded subdirectory survives the dir filter (the keep.append
        # branch) and its supported files are yielded.
        self._write("root/sub/nested.txt", "deep")
        self._write("root/node_modules/skip.js", "x")  # excluded sibling
        root = os.path.join(self.tmp, "root")
        got = sorted(os.path.basename(p) for p in rag._iter_files(root))
        self.assertEqual(got, ["nested.txt"])

    def test_skips_zero_byte_files(self):
        self._write("root/empty.txt", "")
        self._write("root/full.txt", "data")
        root = os.path.join(self.tmp, "root")
        got = [os.path.basename(p) for p in rag._iter_files(root)]
        self.assertEqual(got, ["full.txt"])

    def test_skips_oversize_files(self):
        self._write("root/big.txt", "x" * 100)
        rag.RAG_MAX_FILE_BYTES = 10
        root = os.path.join(self.tmp, "root")
        self.assertEqual(list(rag._iter_files(root)), [])

    def test_getsize_oserror_skipped(self):
        self._write("root/a.txt", "hi")
        root = os.path.join(self.tmp, "root")
        with mock.patch.object(rag.os.path, "getsize", side_effect=OSError("gone")):
            self.assertEqual(list(rag._iter_files(root)), [])


class FileSignatureTests(_RagBase):
    def test_signature_is_mtime_size(self):
        p = self._write("a.txt", "hello")
        sig = rag._file_signature(p)
        self.assertIn(":", sig)
        mtime, size = sig.split(":")
        self.assertEqual(int(size), 5)
        self.assertTrue(mtime.isdigit())

    def test_signature_empty_on_missing(self):
        self.assertEqual(rag._file_signature(os.path.join(self.tmp, "ghost.txt")), "")

    def test_existing_signature_reads_metadata(self):
        coll = self._install_collection()
        coll._store["fid:0"] = ("doc", {"file_id": "fid", "sig": "123:45"})
        self.assertEqual(rag._existing_signature("fid"), "123:45")

    def test_existing_signature_absent_returns_blank(self):
        self._install_collection()
        self.assertEqual(rag._existing_signature("nope"), "")

    def test_existing_signature_get_error_blank(self):
        coll = self._install_collection()
        coll.raise_on_get = True
        self.assertEqual(rag._existing_signature("fid"), "")

    def test_delete_file_calls_collection(self):
        coll = self._install_collection()
        coll._store["fid:0"] = ("d", {"file_id": "fid"})
        rag._delete_file("fid")
        self.assertEqual(coll.count(), 0)

    def test_delete_file_error_swallowed(self):
        coll = self._install_collection()

        def _boom(*a, **k):
            raise RuntimeError("delete boom")

        coll.delete = _boom
        # Must not raise.
        rag._delete_file("fid")


# ───────────────────────────── _index_file ─────────────────────────────
class IndexFileTests(_RagBase):
    def test_unsupported_skipped(self):
        p = self._write("a.png", "x")
        self.assertEqual(rag._index_file(p), 0)

    def test_missing_signature_skipped(self):
        # Supported ext but the file doesn't exist → sig == "" → 0.
        self.assertEqual(rag._index_file(os.path.join(self.tmp, "ghost.txt")), 0)

    def test_unchanged_file_skipped(self):
        p = self._write("a.txt", "hello world")
        coll = self._install_collection()
        self._install_embedder()
        sig = rag._file_signature(p)
        fid = rag._file_id(p)
        coll._store[fid + ":0"] = ("old", {"file_id": fid, "sig": sig})
        self.assertEqual(rag._index_file(p), 0)

    def test_empty_text_deletes_prior(self):
        p = self._write("a.txt", "   ")  # whitespace → empty after strip
        coll = self._install_collection()
        self._install_embedder()
        fid = rag._file_id(p)
        coll._store[fid + ":0"] = ("old", {"file_id": fid, "sig": "stale"})
        self.assertEqual(rag._index_file(p), 0)
        self.assertEqual(coll.count(), 0)  # prior chunk removed

    def test_no_chunks_returns_zero(self):
        p = self._write("a.txt", "content")
        self._install_collection()
        self._install_embedder()
        with mock.patch.object(rag, "_chunk", return_value=[]):
            self.assertEqual(rag._index_file(p), 0)

    def test_happy_path_writes_chunks(self):
        p = self._write("a.txt", "hello world this is content")
        coll = self._install_collection()
        emb = self._install_embedder()
        with mock.patch.object(rag, "_chunk", return_value=["c1", "c2"]):
            n = rag._index_file(p)
        self.assertEqual(n, 2)
        self.assertEqual(coll.count(), 2)
        self.assertEqual(rag._stats["chunks_written"], 2)
        self.assertEqual(rag._stats["files_indexed"], 1)
        # The embedder saw the chunk texts.
        self.assertIn(["c1", "c2"], emb.calls)
        # Metadata carries path/filename/sig/ext.
        meta = coll.added_calls[0]["metadatas"][0]
        self.assertEqual(meta["filename"], "a.txt")
        self.assertEqual(meta["ext"], ".txt")
        self.assertIn("sig", meta)

    def test_replaces_prior_chunks(self):
        p = self._write("a.txt", "fresh content here")
        coll = self._install_collection()
        self._install_embedder()
        fid = rag._file_id(p)
        # Prior version had 3 chunks under a stale signature.
        for i in range(3):
            coll._store[f"{fid}:{i}"] = ("old%d" % i, {"file_id": fid, "sig": "stale"})
        with mock.patch.object(rag, "_chunk", return_value=["new"]):
            rag._index_file(p)
        # Only the single new chunk remains.
        self.assertEqual(coll.count(), 1)

    def test_embed_url_error_records_stat(self):
        p = self._write("a.txt", "content")
        self._install_collection()
        self._install_embedder(_FakeEmbedder(fail=True))  # raises URLError
        with mock.patch.object(rag, "_chunk", return_value=["c1"]):
            self.assertEqual(rag._index_file(p), 0)
        self.assertEqual(rag._stats["errors"], 1)
        self.assertIn("ollama embed", rag._last_error)

    def test_embed_generic_error_records_stat(self):
        p = self._write("a.txt", "content")
        self._install_collection()

        class _Boom(_FakeEmbedder):
            def encode(self, *a, **k):
                raise ValueError("weird embed failure")

        self._install_embedder(_Boom())
        with mock.patch.object(rag, "_chunk", return_value=["c1"]):
            self.assertEqual(rag._index_file(p), 0)
        self.assertEqual(rag._stats["errors"], 1)
        self.assertIn("embed(", rag._last_error)

    def test_add_error_records_stat(self):
        p = self._write("a.txt", "content")
        coll = self._install_collection()
        coll.raise_on_add = True
        self._install_embedder()
        with mock.patch.object(rag, "_chunk", return_value=["c1"]):
            self.assertEqual(rag._index_file(p), 0)
        self.assertEqual(rag._stats["errors"], 1)
        self.assertIn("add(", rag._last_error)


# ───────────────────────────── index_once ──────────────────────────────
class IndexOnceTests(_RagBase):
    def test_degraded_when_unavailable(self):
        with mock.patch.object(rag, "is_available", return_value=False):
            out = rag.index_once()
        self.assertFalse(out["ok"])
        self.assertIn("chromadb", out["error"])

    def test_indexes_a_tree(self):
        self._write("docs/a.txt", "alpha content")
        self._write("docs/b.md", "beta content")
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "docs")]
        coll = self._install_collection()
        self._install_embedder()
        with mock.patch.object(rag, "_chunk", return_value=["chunk"]):
            out = rag.index_once()
        self.assertTrue(out["ok"])
        self.assertEqual(out["files_seen"], 2)
        self.assertEqual(coll.count(), 2)
        self.assertGreater(out["ts"], 0)

    def test_skips_missing_roots(self):
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "does_not_exist")]
        self._install_collection()
        self._install_embedder()
        out = rag.index_once()
        self.assertTrue(out["ok"])
        self.assertEqual(out["files_seen"], 0)

    def test_stop_flag_halts_walk(self):
        for i in range(5):
            self._write(f"docs/f{i}.txt", "content")
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "docs")]
        self._install_collection()
        self._install_embedder()
        rag._stop_flag.set()  # already set → inner loop breaks immediately
        out = rag.index_once()
        self.assertTrue(out["ok"])
        self.assertEqual(out["files_seen"], 0)

    def test_progress_callback_invoked(self):
        # 25-file cadence → make 25 files so progress fires once.
        for i in range(25):
            self._write(f"docs/f{i:02d}.txt", "content")
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "docs")]
        self._install_collection()
        self._install_embedder()
        seen = []
        with mock.patch.object(rag, "_chunk", return_value=["c"]):
            rag.index_once(progress=lambda path, n: seen.append((path, n)))
        self.assertTrue(seen)
        self.assertEqual(seen[0][1], 25)

    def test_progress_callback_error_swallowed(self):
        for i in range(25):
            self._write(f"docs/f{i:02d}.txt", "content")
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "docs")]
        self._install_collection()
        self._install_embedder()

        def _boom(path, n):
            raise RuntimeError("progress boom")

        with mock.patch.object(rag, "_chunk", return_value=["c"]):
            out = rag.index_once(progress=_boom)  # must not raise
        self.assertTrue(out["ok"])

    def test_index_file_exception_counted_and_continues(self):
        self._write("docs/a.txt", "content")
        self._write("docs/b.txt", "content")
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "docs")]
        self._install_collection()
        self._install_embedder()
        calls = {"n": 0}

        def _boom(path):
            calls["n"] += 1
            raise RuntimeError("index failure")

        with mock.patch.object(rag, "_index_file", _boom):
            out = rag.index_once()
        self.assertTrue(out["ok"])
        self.assertEqual(calls["n"], 2)            # both files attempted
        self.assertGreaterEqual(rag._stats["errors"], 2)

    def test_force_drops_collection_first(self):
        rag.RAG_INDEX_PATHS = []
        self._install_embedder()
        with mock.patch.object(rag, "_get_collection") as gc:
            gc.return_value = _FakeCollection()
            rag.index_once(force=True)
        # force path calls _get_collection(force_reindex=True) at least once.
        self.assertTrue(
            any(c.kwargs.get("force_reindex") for c in gc.call_args_list)
        )

    def test_garbage_collects_stale_chunks(self):
        # A chunk whose path no longer exists and isn't in seen_files is purged.
        rag.RAG_INDEX_PATHS = []  # nothing on disk → seen_files empty
        coll = self._install_collection()
        coll._store["ghost:0"] = (
            "old", {"file_id": "ghost", "path": os.path.join(self.tmp, "gone.txt")})
        self._install_embedder()
        out = rag.index_once()
        self.assertTrue(out["ok"])
        self.assertEqual(coll.count(), 0)  # stale chunk removed

    def test_gc_keeps_chunks_for_existing_files(self):
        live = self._write("docs/live.txt", "still here")
        rag.RAG_INDEX_PATHS = []  # not walked, but file exists on disk
        coll = self._install_collection()
        fid = rag._file_id(live)
        coll._store[fid + ":0"] = ("doc", {"file_id": fid, "path": live})
        self._install_embedder()
        rag.index_once()
        self.assertEqual(coll.count(), 1)  # kept because os.path.isfile(path)

    def test_gc_exception_swallowed(self):
        rag.RAG_INDEX_PATHS = []
        coll = self._install_collection()
        coll.raise_on_get = True  # the GC get() blows up
        self._install_embedder()
        out = rag.index_once()  # must still succeed
        self.assertTrue(out["ok"])


# ─────────────────────── _drain_event_queue ────────────────────────────
class DrainEventQueueTests(_RagBase):
    def test_reindexes_pending_path(self):
        p = self._write("a.txt", "content")
        indexed = []
        with mock.patch.object(rag, "_index_file",
                               side_effect=lambda path: indexed.append(path)):
            # Seed a due item, then let the loop run exactly one pass.
            rag._event_q.put(p)
            self._run_one_drain_pass(due_now=True)
        self.assertEqual(indexed, [p])

    def test_deletes_when_path_gone(self):
        gone = os.path.join(self.tmp, "gone.txt")
        deleted = []
        with mock.patch.object(rag, "_delete_file",
                               side_effect=lambda fid: deleted.append(fid)):
            rag._event_q.put(gone)
            self._run_one_drain_pass(due_now=True)
        self.assertEqual(deleted, [rag._file_id(gone)])

    def test_delete_error_swallowed(self):
        gone = os.path.join(self.tmp, "gone.txt")
        with mock.patch.object(rag, "_delete_file", side_effect=RuntimeError("x")):
            rag._event_q.put(gone)
            self._run_one_drain_pass(due_now=True)  # must not raise

    def test_index_error_records_stat(self):
        p = self._write("a.txt", "content")
        with mock.patch.object(rag, "_index_file", side_effect=RuntimeError("boom")):
            rag._event_q.put(p)
            self._run_one_drain_pass(due_now=True)
        self.assertEqual(rag._stats["errors"], 1)
        self.assertIn("reindex(", rag._last_error)

    def test_empty_queue_poll_is_tolerated(self):
        # One loop pass with nothing queued exercises the queue.Empty branch.
        # Patch get() to raise immediately so we don't actually block 0.1s.
        flags = iter([False, True])

        def _is_set():
            try:
                return next(flags)
            except StopIteration:
                return True

        with mock.patch.object(rag._event_q, "get",
                               side_effect=rag.queue.Empty), \
                mock.patch.object(rag._stop_flag, "is_set", _is_set):
            rag._drain_event_queue()  # must not raise; pending stays empty

    def _run_one_drain_pass(self, due_now):
        """Drive _drain_event_queue for a single iteration deterministically:
        stop the loop after the first body run, and force pending items 'due'
        by patching time.time so due <= now is immediately true."""
        original_time = rag.time.time
        # First call (inside try) sets due = now()+2; subsequent now() reads
        # must be >= that so the item is processed in the same pass.
        base = original_time()
        seq = iter([base, base + 100, base + 100, base + 100, base + 100])

        def _fake_time():
            try:
                return next(seq)
            except StopIteration:
                return base + 100

        # is_set: False for the first while-check, True afterwards so we run
        # exactly one iteration of the loop body.
        flags = iter([False, True, True, True])

        def _is_set():
            try:
                return next(flags)
            except StopIteration:
                return True

        with mock.patch.object(rag.time, "time", _fake_time), \
                mock.patch.object(rag._stop_flag, "is_set", _is_set):
            rag._drain_event_queue()


# ─────────────────────────── _start_watchdog ───────────────────────────
class StartWatchdogTests(_RagBase):
    def test_missing_watchdog_returns_false(self):
        with mock.patch.dict(sys.modules, {"watchdog": None,
                                           "watchdog.observers": None,
                                           "watchdog.events": None}):
            self.assertFalse(rag._start_watchdog())

    def test_no_valid_roots_returns_false(self):
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "nope")]
        fake_wd = self._install_fake_watchdog()
        with fake_wd:
            self.assertFalse(rag._start_watchdog())

    def test_schedules_and_starts(self):
        os.makedirs(os.path.join(self.tmp, "watched"), exist_ok=True)
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "watched")]
        observer = self._FakeObserver()
        with self._install_fake_watchdog(observer):
            ok = rag._start_watchdog()
        self.assertTrue(ok)
        self.assertTrue(observer.started)
        self.assertEqual(observer.scheduled, 1)
        self.assertIs(rag._observer, observer)

    def test_schedule_exception_is_tolerated(self):
        os.makedirs(os.path.join(self.tmp, "a"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp, "b"), exist_ok=True)
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")]
        observer = self._FakeObserver(fail_first_schedule=True)
        with self._install_fake_watchdog(observer):
            ok = rag._start_watchdog()
        # One schedule raised, the other succeeded → overall True.
        self.assertTrue(ok)
        self.assertEqual(observer.scheduled, 1)

    def test_handler_enqueues_supported_file_events(self):
        os.makedirs(os.path.join(self.tmp, "watched"), exist_ok=True)
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "watched")]
        observer = self._FakeObserver()
        with self._install_fake_watchdog(observer):
            rag._start_watchdog()
        handler = observer.handler

        # Supported file event → enqueued.
        ev = types.SimpleNamespace(is_directory=False, src_path="x.txt", dest_path="")
        handler.on_any_event(ev)
        self.assertEqual(rag._event_q.get_nowait(), "x.txt")

        # Directory event → ignored.
        handler.on_any_event(types.SimpleNamespace(is_directory=True, src_path="d"))
        # Unsupported extension → ignored.
        handler.on_any_event(types.SimpleNamespace(
            is_directory=False, src_path="x.png", dest_path=""))
        self.assertTrue(rag._event_q.empty())

        # dest_path (a move) is preferred when present.
        handler.on_any_event(types.SimpleNamespace(
            is_directory=False, src_path="old.txt", dest_path="new.txt"))
        self.assertEqual(rag._event_q.get_nowait(), "new.txt")

    def test_handler_enqueue_failure_swallowed(self):
        os.makedirs(os.path.join(self.tmp, "watched"), exist_ok=True)
        rag.RAG_INDEX_PATHS = [os.path.join(self.tmp, "watched")]
        observer = self._FakeObserver()
        with self._install_fake_watchdog(observer):
            rag._start_watchdog()
        handler = observer.handler
        with mock.patch.object(rag._event_q, "put_nowait",
                               side_effect=RuntimeError("full")):
            # Should swallow the queue error.
            handler.on_any_event(types.SimpleNamespace(
                is_directory=False, src_path="x.txt", dest_path=""))

    # -- watchdog fakes -------------------------------------------------
    class _FakeObserver:
        def __init__(self, fail_first_schedule=False):
            self.scheduled = 0
            self.started = False
            self.stopped = False
            self.joined = False
            self.handler = None
            self._fail_first = fail_first_schedule

        def schedule(self, handler, root, recursive=True):
            self.handler = handler
            if self._fail_first:
                self._fail_first = False
                raise RuntimeError("schedule failed")
            self.scheduled += 1

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self, timeout=None):
            self.joined = True

    def _install_fake_watchdog(self, observer=None):
        observer = observer or self._FakeObserver()
        observers_mod = types.ModuleType("watchdog.observers")
        observers_mod.Observer = lambda: observer
        events_mod = types.ModuleType("watchdog.events")

        class _FileSystemEventHandler:
            pass

        events_mod.FileSystemEventHandler = _FileSystemEventHandler
        wd = types.ModuleType("watchdog")
        return mock.patch.dict(sys.modules, {
            "watchdog": wd,
            "watchdog.observers": observers_mod,
            "watchdog.events": events_mod,
        })


# ───────────────────────────── start / stop ────────────────────────────
class StartStopTests(_RagBase):
    def test_start_returns_false_when_unavailable(self):
        with mock.patch.object(rag, "is_available", return_value=False):
            self.assertFalse(rag.start())

    def test_start_spawns_thread_and_scans(self):
        # Avoid real threads/watchdog/ollama: stub everything the boot touches.
        events = {"scan": 0, "drain": 0, "watchdog": 0}

        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target
                self._alive = False

            def start(self):
                self._alive = True
                self._target()  # run inline, deterministically

            def is_alive(self):
                return self._alive

        with mock.patch.object(rag, "is_available", return_value=True), \
                mock.patch.object(rag, "_ollama_reachable", return_value=True), \
                mock.patch.object(rag, "index_once",
                                  side_effect=lambda: events.__setitem__("scan", 1) or {"ok": True}), \
                mock.patch.object(rag, "_drain_event_queue",
                                  side_effect=lambda: events.__setitem__("drain", 1)), \
                mock.patch.object(rag, "_start_watchdog",
                                  side_effect=lambda: events.__setitem__("watchdog", 1) or True), \
                mock.patch.object(rag.threading, "Thread", _ImmediateThread):
            ok = rag.start(initial_scan=True)
        self.assertTrue(ok)
        self.assertEqual(events["scan"], 1)
        self.assertEqual(events["drain"], 1)
        self.assertEqual(events["watchdog"], 1)

    def test_start_warns_when_ollama_unreachable(self):
        class _NoopThread:
            def __init__(self, target=None, name=None, daemon=None):
                pass

            def start(self):
                pass

            def is_alive(self):
                return True

        with mock.patch.object(rag, "is_available", return_value=True), \
                mock.patch.object(rag, "_ollama_reachable", return_value=False), \
                mock.patch.object(rag, "_start_watchdog", return_value=False), \
                mock.patch.object(rag.threading, "Thread", _NoopThread):
            # Just exercising the unreachable-warning branch; no assertion on
            # the print, only that start() still returns True.
            self.assertTrue(rag.start())

    def test_start_skips_initial_scan(self):
        events = {"scan": 0, "drain": 0}

        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

            def is_alive(self):
                return True

        with mock.patch.object(rag, "is_available", return_value=True), \
                mock.patch.object(rag, "_ollama_reachable", return_value=True), \
                mock.patch.object(rag, "index_once",
                                  side_effect=lambda: events.__setitem__("scan", 1)), \
                mock.patch.object(rag, "_drain_event_queue",
                                  side_effect=lambda: events.__setitem__("drain", 1)), \
                mock.patch.object(rag, "_start_watchdog", return_value=True), \
                mock.patch.object(rag.threading, "Thread", _ImmediateThread):
            rag.start(initial_scan=False)
        self.assertEqual(events["scan"], 0)   # skipped
        self.assertEqual(events["drain"], 1)  # loop still runs

    def test_start_initial_scan_exception_recorded(self):
        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

            def is_alive(self):
                return True

        with mock.patch.object(rag, "is_available", return_value=True), \
                mock.patch.object(rag, "_ollama_reachable", return_value=True), \
                mock.patch.object(rag, "index_once", side_effect=RuntimeError("scan boom")), \
                mock.patch.object(rag, "_drain_event_queue", return_value=None), \
                mock.patch.object(rag, "_start_watchdog", return_value=True), \
                mock.patch.object(rag.threading, "Thread", _ImmediateThread):
            rag.start()
        self.assertIn("initial scan", rag._last_error)

    def test_start_returns_true_if_already_running(self):
        class _AliveThread:
            def is_alive(self):
                return True

        rag._indexer_thread = _AliveThread()
        with mock.patch.object(rag, "is_available", return_value=True), \
                mock.patch.object(rag, "_ollama_reachable", return_value=True):
            self.assertTrue(rag.start())

    def test_stop_stops_observer(self):
        observer = StartWatchdogTests._FakeObserver()
        rag._observer = observer
        rag.stop()
        self.assertTrue(rag._stop_flag.is_set())
        self.assertTrue(observer.stopped)
        self.assertIsNone(rag._observer)

    def test_stop_no_observer_is_noop(self):
        rag._observer = None
        rag.stop()  # must not raise
        self.assertTrue(rag._stop_flag.is_set())

    def test_stop_observer_error_swallowed(self):
        class _BadObserver:
            def stop(self):
                raise RuntimeError("stop boom")

            def join(self, timeout=None):
                pass

        rag._observer = _BadObserver()
        rag.stop()  # swallow → no raise
        self.assertIsNone(rag._observer)


# ─────────────────────────────── search ────────────────────────────────
class SearchTests(_RagBase):
    def _seed_query(self, coll, docs, metas, dists):
        coll.query_return = {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    def test_blank_query_returns_empty(self):
        self.assertEqual(rag.search("   "), [])
        self.assertEqual(rag.search(""), [])

    def test_unavailable_returns_empty(self):
        with mock.patch.object(rag, "is_available", return_value=False):
            self.assertEqual(rag.search("hello"), [])

    def test_init_failure_returns_empty(self):
        with mock.patch.object(rag, "_get_collection", side_effect=RuntimeError("no chroma")):
            self.assertEqual(rag.search("hello"), [])

    def test_embed_query_failure_returns_empty(self):
        self._install_collection()
        self._install_embedder(_FakeEmbedder(fail=True))
        rag.RAG_RERANKER_MODEL = ""
        self.assertEqual(rag.search("hello"), [])

    def test_query_failure_returns_empty(self):
        coll = self._install_collection()
        coll.raise_on_query = True
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = ""
        self.assertEqual(rag.search("hello"), [])

    def test_basic_ranking_by_cosine(self):
        coll = self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = ""  # no rerank → cosine score order
        self._seed_query(
            coll,
            docs=["doc near", "doc far"],
            metas=[
                {"path": "/a.txt", "filename": "a.txt", "chunk_index": 0, "ext": ".txt"},
                {"path": "/b.txt", "filename": "b.txt", "chunk_index": 1, "ext": ".txt"},
            ],
            dists=[0.1, 0.9],
        )
        hits = rag.search("query", k=5)
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(hits[0]["score"], 0.9, places=6)  # 1 - 0.1
        self.assertAlmostEqual(hits[1]["score"], max(0.0, 1.0 - 0.9), places=6)
        self.assertEqual(hits[0]["path"], "/a.txt")
        self.assertEqual(hits[0]["snippet"], "doc near")
        self.assertEqual(hits[0]["chunk_index"], 0)

    def test_distance_over_one_clipped_to_zero(self):
        coll = self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = ""
        self._seed_query(
            coll, docs=["d"],
            metas=[{"path": "/a.txt", "filename": "a.txt", "chunk_index": 0, "ext": ".txt"}],
            dists=[1.5])
        hits = rag.search("query")
        self.assertEqual(hits[0]["score"], 0.0)

    def test_k_limit_applied(self):
        coll = self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = ""
        self._seed_query(
            coll,
            docs=[f"doc{i}" for i in range(5)],
            metas=[{"path": f"/{i}.txt", "filename": f"{i}.txt",
                    "chunk_index": i, "ext": ".txt"} for i in range(5)],
            dists=[0.1 * i for i in range(5)])
        hits = rag.search("query", k=2)
        self.assertEqual(len(hits), 2)

    def test_non_dict_meta_skipped(self):
        coll = self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = ""
        self._seed_query(
            coll,
            docs=["good", "bad"],
            metas=[{"path": "/a.txt", "filename": "a", "chunk_index": 0, "ext": ".txt"},
                   "not-a-dict"],
            dists=[0.2, 0.3])
        hits = rag.search("query")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["path"], "/a.txt")

    def test_paths_filter(self):
        coll = self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = ""
        self._seed_query(
            coll,
            docs=["a", "b"],
            metas=[{"path": "/keep/a.txt", "filename": "a", "chunk_index": 0, "ext": ".txt"},
                   {"path": "/skip/b.txt", "filename": "b", "chunk_index": 0, "ext": ".txt"}],
            dists=[0.2, 0.1])
        hits = rag.search("query", paths=["/keep"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["path"], "/keep/a.txt")

    def test_rerank_reorders_hits(self):
        coll = self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = "some/model"
        # Cosine would rank A first (smaller distance), but rerank scores B higher.
        self._seed_query(
            coll,
            docs=["short", "a much longer snippet"],
            metas=[{"path": "/a.txt", "filename": "a", "chunk_index": 0, "ext": ".txt"},
                   {"path": "/b.txt", "filename": "b", "chunk_index": 0, "ext": ".txt"}],
            dists=[0.1, 0.5])
        # Reranker: explicit scores making B win.
        rag._reranker = _FakeReranker(scores=[0.2, 0.95])
        hits = rag.search("query")
        self.assertEqual(hits[0]["path"], "/b.txt")  # reranked to top
        self.assertAlmostEqual(hits[0]["score"], 0.95, places=6)

    def test_rerank_failure_falls_back_to_cosine(self):
        coll = self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = "some/model"
        self._seed_query(
            coll,
            docs=["a", "b"],
            metas=[{"path": "/a.txt", "filename": "a", "chunk_index": 0, "ext": ".txt"},
                   {"path": "/b.txt", "filename": "b", "chunk_index": 0, "ext": ".txt"}],
            dists=[0.1, 0.5])
        rag._reranker = _FakeReranker(fail=True)
        hits = rag.search("query")
        # Falls back to cosine order (A first, smaller distance).
        self.assertEqual(hits[0]["path"], "/a.txt")

    def test_no_results_empty_list(self):
        self._install_collection()
        self._install_embedder()
        rag.RAG_RERANKER_MODEL = ""
        # Default query_return → empty docs/metas/dists.
        self.assertEqual(rag.search("query"), [])


# ──────────────────────── configure / config / status ──────────────────
class ConfigTests(_RagBase):
    def test_configure_updates_known_keys(self):
        cfg = rag.configure(rag_embed_model="custom-model", rag_chunk_chars=999)
        self.assertEqual(rag.RAG_EMBED_MODEL, "custom-model")
        self.assertEqual(rag.RAG_CHUNK_CHARS, 999)
        self.assertEqual(cfg["RAG_EMBED_MODEL"], "custom-model")

    def test_configure_ignores_unknown_keys(self):
        before = rag.current_config()
        rag.configure(totally_unknown_key="zzz")
        after = rag.current_config()
        self.assertEqual(before, after)
        self.assertFalse(hasattr(rag, "TOTALLY_UNKNOWN_KEY"))

    def test_configure_accepts_uppercase(self):
        rag.configure(RAG_DEVICE="cuda")
        self.assertEqual(rag.RAG_DEVICE, "cuda")

    def test_configure_invalidates_cached_embedder(self):
        # A cached embedder built with the OLD model must be dropped when the
        # model changes, so the next _get_embedder() rebuilds with the new one.
        rag._embed_model = rag._OllamaEmbedder(model="old", endpoint="e")
        rag.configure(rag_embed_model="new-model")
        self.assertIsNone(rag._embed_model)
        emb = rag._get_embedder()
        self.assertEqual(emb.model, "new-model")

    def test_configure_invalidates_cached_reranker_and_collection(self):
        rag._reranker = object()
        rag._collection = object()
        rag.configure(rag_reranker_model="other-reranker",
                      rag_collection="other_collection")
        self.assertIsNone(rag._reranker)
        self.assertIsNone(rag._collection)

    def test_configure_same_value_keeps_cached_singletons(self):
        # Re-asserting the current value must NOT throw away warm singletons.
        sentinel = object()
        rag._embed_model = sentinel
        rag.configure(rag_embed_model=rag.RAG_EMBED_MODEL)
        self.assertIs(rag._embed_model, sentinel)

    def test_current_config_shape(self):
        cfg = rag.current_config()
        for key in ("RAG_INDEX_PATHS", "RAG_EMBED_MODEL", "RAG_CHUNK_CHARS",
                    "RAG_OLLAMA_ENDPOINT", "chroma_path"):
            self.assertIn(key, cfg)
        # Lists are copies, not the live module list.
        self.assertIsNot(cfg["RAG_INDEX_PATHS"], rag.RAG_INDEX_PATHS)

    def test_status_reports_state(self):
        rag._stats["files_indexed"] = 7
        rag._last_error = "boom"
        st = rag.status()
        self.assertTrue(st["available"])
        self.assertFalse(st["running"])
        self.assertFalse(st["watchdog_active"])
        self.assertEqual(st["files_indexed"], 7)
        self.assertEqual(st["last_error"], "boom")
        self.assertIn("config", st)

    def test_status_watchdog_active_when_observer_set(self):
        rag._observer = object()
        self.assertTrue(rag.status()["watchdog_active"])

    def test_status_running_true_with_alive_thread(self):
        class _Alive:
            def is_alive(self):
                return True

        rag._indexer_thread = _Alive()
        self.assertTrue(rag.status()["running"])


# ──────────────────────────── collection_size ──────────────────────────
class CollectionSizeTests(_RagBase):
    def test_returns_count(self):
        coll = self._install_collection()
        coll._store["a"] = ("d", {})
        coll._store["b"] = ("d", {})
        self.assertEqual(rag.collection_size(), 2)

    def test_zero_on_error(self):
        with mock.patch.object(rag, "_get_collection", side_effect=RuntimeError("x")):
            self.assertEqual(rag.collection_size(), 0)


# ───────────────────────── module-level helpers ────────────────────────
class ModuleHelperTests(_RagBase):
    def test_user_home_expands(self):
        self.assertEqual(rag._user_home(), os.path.expanduser("~"))

    def test_default_index_paths_filters_to_existing_dirs(self):
        real = os.path.join(self.tmp, "Documents")
        os.makedirs(real, exist_ok=True)

        def _fake_home():
            return self.tmp

        with mock.patch.object(rag, "_user_home", _fake_home):
            paths = rag._default_index_paths()
        # Only Documents exists in the tmp home; Desktop/OneDrive don't.
        self.assertEqual(paths, [real])


if __name__ == "__main__":
    unittest.main()
