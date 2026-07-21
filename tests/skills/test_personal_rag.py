"""Logic tests for skills/personal_rag.py.

personal_rag is a thin voice layer over core.rag_indexer. Tests cover the
pure formatting helpers (snippet truncation, voice vs LLM hit rendering),
the graceful-degradation paths when the RAG backend is offline, and the
registered actions with core.rag_indexer mocked so no ChromaDB / embeddings /
filesystem are touched.

The autostart thread in register() is neutered by the harness (Thread.start
no-ops), so loading the skill never kicks off a real index scan.
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_rag(available=True, hits=None):
    """Build a stand-in for core.rag_indexer with controllable behaviour."""
    rag = mock.MagicMock()
    rag.is_available.return_value = available
    rag.search.return_value = hits if hits is not None else []
    return rag


class _RagStubPatch:
    """Inject a stub ``core.rag_indexer`` so the skill's ``_rag()`` (which does
    ``from core import rag_indexer``) resolves to the stub and NEVER imports the
    real chromadb/OTEL stack.

    Patching only ``sys.modules['core.rag_indexer']`` is NOT enough once another
    test (e.g. tests/test_rag_indexer.py) has already imported the real module:
    Python then satisfies ``from core import rag_indexer`` via the ``rag_indexer``
    ATTRIBUTE on the already-loaded ``core`` package (``getattr``), bypassing the
    sys.modules entry entirely. The real module gets imported, dragging in
    chromadb — and doing so ~52× across this file's tests eventually segfaults
    chromadb's native OpenTelemetry init. So we also override (and restore) the
    ``core.rag_indexer`` package attribute. Behaves like a context manager AND a
    started patcher (exposes ``.stop()``), matching the previous call site."""

    def __init__(self, stub):
        self.stub = stub
        self._dict_patch = mock.patch.dict(sys.modules, {"core.rag_indexer": stub})
        self._saved_attr = None      # (had_attr, prev_value)
        self._core = None

    def start(self):
        self._dict_patch.start()
        try:
            import core as _core
            self._core = _core
            had = hasattr(_core, "rag_indexer")
            self._saved_attr = (had, getattr(_core, "rag_indexer", None))
            _core.rag_indexer = self.stub
        except Exception:
            self._saved_attr = None
        return self

    def stop(self):
        if self._core is not None and self._saved_attr is not None:
            had, prev = self._saved_attr
            if had:
                self._core.rag_indexer = prev
            else:
                try:
                    delattr(self._core, "rag_indexer")
                except AttributeError:
                    pass
        self._dict_patch.stop()


def _load_rag_skill():
    """Load personal_rag with a fake core.rag_indexer injected so register()'s
    _rag() resolves to a stub — never importing the real chromadb/OTEL stack
    (which stalls ~10s/test on cloud-metadata detection in this environment and,
    re-imported across the suite, can segfault chromadb's native init)."""
    stub = _fake_rag(available=False)
    patcher = _RagStubPatch(stub)
    patcher.start()
    mod, actions = load_skill_isolated("personal_rag")
    return mod, actions, patcher


class PersonalRagFormattingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    # ── _short_snippet ───────────────────────────────────────────────────
    def test_short_snippet_collapses_whitespace(self):
        self.assertEqual(self.mod._short_snippet("a   b\n\tc"), "a b c")

    def test_short_snippet_truncates_with_ellipsis(self):
        out = self.mod._short_snippet("x" * 500, n=50)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 51)

    def test_short_snippet_empty(self):
        self.assertEqual(self.mod._short_snippet(""), "")

    # ── _format_hits_for_voice ───────────────────────────────────────────
    def test_format_voice_no_hits(self):
        out = self.mod._format_hits_for_voice([])
        self.assertIn("didn't find anything", out)

    def test_format_voice_caps_at_max_hits_and_numbers(self):
        hits = [{"filename": f"f{i}.txt", "snippet": f"snip{i}"} for i in range(5)]
        out = self.mod._format_hits_for_voice(hits)
        # RAG_VOICE_MAX_HITS caps spoken hits at 3.
        self.assertIn("Top 3 matches", out)
        self.assertIn("f0.txt", out)
        self.assertIn("f2.txt", out)
        self.assertNotIn("f3.txt", out)

    def test_format_voice_single_hit_phrasing(self):
        out = self.mod._format_hits_for_voice([{"filename": "notes.md", "snippet": "hi"}])
        self.assertIn("Top match", out)
        self.assertIn("notes.md", out)

    def test_format_voice_falls_back_to_basename(self):
        out = self.mod._format_hits_for_voice(
            [{"path": "docs/report.pdf", "snippet": "q3"}])
        self.assertIn("report.pdf", out)

    # ── _format_hits_for_llm ─────────────────────────────────────────────
    def test_format_llm_no_hits(self):
        self.assertEqual(self.mod._format_hits_for_llm([]), "[no matches]")

    def test_format_llm_includes_path_and_score(self):
        out = self.mod._format_hits_for_llm(
            [{"path": "a.txt", "snippet": "body", "score": 0.4242}])
        self.assertIn("path=a.txt", out)
        self.assertIn("0.424", out)
        self.assertIn("body", out)


class PersonalRagActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    # ── rag_search ───────────────────────────────────────────────────────
    def test_rag_search_empty_query_prompts(self):
        self.assertIn("What should I search", self.actions["rag_search"](""))

    def test_rag_search_offline_message(self):
        with mock.patch.object(self.mod, "_rag", return_value=_fake_rag(available=False)):
            out = self.actions["rag_search"]("budget doc")
        self.assertIn("offline", out.lower())
        self.assertIn("chromadb", out.lower())

    def test_rag_search_renders_hits_and_caches(self):
        rag = _fake_rag(hits=[{"filename": "plan.md", "snippet": "the plan",
                               "path": r"C:\plan.md"}])
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.actions["rag_search"]("plan")
        self.assertIn("plan.md", out)
        rag.search.assert_called_once()
        # The query + hits are cached for rag_open_top.
        self.assertEqual(self.mod._last_query, "plan")
        self.assertEqual(len(self.mod._last_hits), 1)

    # ── rag_search_quiet (machine-readable) ──────────────────────────────
    def test_rag_search_quiet_empty_query(self):
        self.assertIn("empty query", self.actions["rag_search_quiet"](""))

    def test_rag_search_quiet_offline_marker(self):
        with mock.patch.object(self.mod, "_rag", return_value=_fake_rag(available=False)):
            self.assertIn("unavailable", self.actions["rag_search_quiet"]("x"))

    # ── search_my_files tool wrapper ─────────────────────────────────────
    def test_search_my_files_clamps_k(self):
        rag = _fake_rag(hits=[])
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            self.mod.search_my_files("q", k=999)
        # k is clamped into [1, 20].
        self.assertEqual(rag.search.call_args.kwargs.get("k"), 20)

    def test_search_my_files_bad_k_uses_default(self):
        rag = _fake_rag(hits=[])
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            self.mod.search_my_files("q", k="not-an-int")
        self.assertEqual(rag.search.call_args.kwargs.get("k"), self.mod.RAG_DEFAULT_K)

    # ── rag_reindex ──────────────────────────────────────────────────────
    def test_rag_reindex_offline(self):
        with mock.patch.object(self.mod, "_rag", return_value=_fake_rag(available=False)):
            self.assertIn("offline", self.actions["rag_reindex"]("").lower())

    def test_rag_reindex_kicks_background(self):
        # no_background_threads keeps the worker from actually running
        # index_once(); the action should still report it queued.
        from tests._skill_harness import no_background_threads
        with mock.patch.object(self.mod, "_rag", return_value=_fake_rag()), \
                no_background_threads():
            out = self.actions["rag_reindex"]("")
        self.assertIn("background", out.lower())

    # ── rag_status ───────────────────────────────────────────────────────
    def test_rag_status_module_missing(self):
        with mock.patch.object(self.mod, "_rag", return_value=None):
            self.assertIn("not loaded", self.actions["rag_status"]("").lower())

    def test_rag_status_running_summary(self):
        rag = _fake_rag()
        rag.status.return_value = {"running": True, "watchdog_active": True,
                                   "last_full_scan_ts": 0, "errors": 2}
        rag.collection_size.return_value = 1234
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.actions["rag_status"]("")
        self.assertIn("running", out)
        self.assertIn("1234 chunks", out)
        self.assertIn("watchdog on", out)
        self.assertIn("2 errors", out)

    # ── rag_configure ────────────────────────────────────────────────────
    def test_rag_configure_usage_when_no_equals(self):
        with mock.patch.object(self.mod, "_rag", return_value=_fake_rag()):
            self.assertIn("Usage", self.actions["rag_configure"]("index_paths"))

    def test_rag_configure_unknown_key(self):
        with mock.patch.object(self.mod, "_rag", return_value=_fake_rag()):
            self.assertIn("Unknown RAG key", self.actions["rag_configure"]("bogus=1"))

    def test_rag_configure_int_key_validation(self):
        with mock.patch.object(self.mod, "_rag", return_value=_fake_rag()):
            out = self.actions["rag_configure"]("chunk_chars=notnum")
        self.assertIn("must be an integer", out)

    def test_rag_configure_multi_value_splits_on_comma(self):
        rag = _fake_rag()
        rag.configure.return_value = {"RAG_INDEX_PATHS": ["a", "b"]}
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.actions["rag_configure"]("index_paths=a, b")
        self.assertEqual(rag.configure.call_args.kwargs.get("rag_index_paths"), ["a", "b"])
        self.assertIn("index_paths set to", out)

    # ── rag_open_top ─────────────────────────────────────────────────────
    def test_rag_open_top_no_recent_search(self):
        with self.mod._lock:
            self.mod._last_hits = []
        self.assertIn("No recent search", self.actions["rag_open_top"](""))

    def test_rag_open_top_missing_file(self):
        with self.mod._lock:
            self.mod._last_hits = [{"path": r"C:\definitely\not\here_xyz.txt"}]
            self.mod._last_query = "ghost"
        out = self.actions["rag_open_top"]("")
        self.assertIn("no longer exists", out)


class RagModuleHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    def test_ensure_core_on_path_inserts(self):
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path if p != self.mod._PROJECT_DIR]
            self.mod._ensure_core_on_path()
            self.assertIn(self.mod._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved

    def test_rag_import_failure_returns_none(self):
        # Force `from core import rag_indexer` inside _rag() to raise so the
        # except branch (print + return None) runs. Patch the core package's
        # attribute to a property-less stub and drop the sys.modules entry, then
        # make importlib raise via a builtins.__import__ shim.
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "core" and a and a[2] and "rag_indexer" in a[2]:
                raise ImportError("chromadb missing")
            return real_import(name, *a, **k)
        with mock.patch.dict(sys.modules):
            sys.modules.pop("core.rag_indexer", None)
            try:
                import core as _core
                if hasattr(_core, "rag_indexer"):
                    delattr(_core, "rag_indexer")
            except Exception:
                pass
            with mock.patch("builtins.__import__", _imp):
                self.assertIsNone(self.mod._rag())


class RagSearchQuietSuccessTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    def test_quiet_renders_llm_block_and_caches(self):
        rag = _fake_rag(hits=[{"path": "a.txt", "snippet": "body",
                               "score": 0.5}])
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.actions["rag_search_quiet"]("plan")
        self.assertIn("path=a.txt", out)
        self.assertEqual(self.mod._last_query, "plan")
        self.assertEqual(len(self.mod._last_hits), 1)

    def test_search_my_files_unavailable_marker(self):
        with mock.patch.object(self.mod, "_rag",
                               return_value=_fake_rag(available=False)):
            out = self.mod.search_my_files("q")
        self.assertIn("unavailable", out)

    def test_search_my_files_success_caches_query(self):
        rag = _fake_rag(hits=[{"path": "b.txt", "snippet": "x", "score": 0.1}])
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.mod.search_my_files("topic", k=3)
        self.assertIn("path=b.txt", out)
        self.assertEqual(self.mod._last_query, "topic")


class RagReindexBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    def test_reindex_bg_closure_runs_index_once(self):
        import threading as _thr
        rag = _fake_rag()
        rag.index_once.return_value = {"files": 3}
        captured = {}
        with mock.patch.object(self.mod, "_rag", return_value=rag), \
             mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)):
            out = self.actions["rag_reindex"]("")
        self.assertIn("background", out.lower())
        # Drive the worker body → exercises the index_once + print branch.
        captured["t"]()
        rag.index_once.assert_called_once()

    def test_reindex_bg_closure_swallows_failure(self):
        import threading as _thr
        rag = _fake_rag()
        rag.index_once.side_effect = RuntimeError("scan boom")
        captured = {}
        with mock.patch.object(self.mod, "_rag", return_value=rag), \
             mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)):
            self.actions["rag_reindex"]("")
        captured["t"]()   # must not raise


class RagStatusAndConfigureBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    def test_status_offline_message(self):
        with mock.patch.object(self.mod, "_rag",
                               return_value=_fake_rag(available=False)):
            out = self.actions["rag_status"]("")
        self.assertIn("offline", out.lower())

    def test_status_idle_with_never_scan(self):
        rag = _fake_rag()
        rag.status.return_value = {"running": False, "watchdog_active": False,
                                   "last_full_scan_ts": 0, "errors": 0}
        rag.collection_size.return_value = 0
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.actions["rag_status"]("")
        self.assertIn("idle", out)
        self.assertIn("watchdog off", out)
        self.assertIn("never", out)

    def test_configure_module_not_loaded(self):
        with mock.patch.object(self.mod, "_rag", return_value=None):
            self.assertIn("not loaded",
                          self.actions["rag_configure"]("embed_model=foo").lower())

    def test_configure_int_key_success(self):
        rag = _fake_rag()
        rag.configure.return_value = {"RAG_CHUNK_CHARS": 800}
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.actions["rag_configure"]("chunk_chars=800")
        self.assertEqual(rag.configure.call_args.kwargs.get("rag_chunk_chars"), 800)
        self.assertIn("chunk_chars set to", out)

    def test_configure_string_key_success(self):
        rag = _fake_rag()
        rag.configure.return_value = {"RAG_EMBED_MODEL": "bge-small"}
        with mock.patch.object(self.mod, "_rag", return_value=rag):
            out = self.actions["rag_configure"]("embed_model=bge-small")
        self.assertEqual(rag.configure.call_args.kwargs.get("rag_embed_model"),
                         "bge-small")
        self.assertIn("embed_model set to", out)


class RagOpenTopSuccessTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    def test_open_top_launches_existing_file(self):
        with self.mod._lock:
            self.mod._last_hits = [{"path": "docs/plan.md"}]
            self.mod._last_query = "plan"
        # os.path.exists True + os.startfile mocked (startfile is Windows-only,
        # so patch it on by attribute even where absent).
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os, "startfile", create=True) as sf:
            out = self.actions["rag_open_top"]("")
        sf.assert_called_once_with("docs/plan.md")
        self.assertIn("Opening plan.md", out)

    def test_open_top_startfile_failure_reported(self):
        with self.mod._lock:
            self.mod._last_hits = [{"path": "docs/plan.md"}]
            self.mod._last_query = "plan"
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os, "startfile", create=True,
                               side_effect=OSError("no handler")):
            out = self.actions["rag_open_top"]("")
        self.assertIn("Couldn't open", out)


class RagRegisterTests(unittest.TestCase):
    def test_register_core_missing_still_registers_actions(self):
        # _rag() returns None → register wires actions but skips autostart.
        mod, _, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)
        actions = {}
        with mock.patch.object(mod, "_rag", return_value=None):
            mod.register(actions)
        for name in ("rag_search", "rag_search_quiet", "search_my_files",
                     "rag_reindex", "rag_status", "rag_configure", "rag_open_top"):
            self.assertIn(name, actions)

    def test_register_autostart_disabled_when_unavailable(self):
        mod, _, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)
        actions = {}
        with mock.patch.object(mod, "_rag",
                               return_value=_fake_rag(available=False)), \
             mock.patch.object(mod, "RAG_AUTOSTART", True):
            mod.register(actions)   # prints the skip notice, no thread
        self.assertIn("rag_search", actions)

    def test_register_autostart_spawns_and_runs_initial_scan(self):
        import threading as _thr
        mod, _, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)
        rag = _fake_rag(available=True)
        actions = {}
        captured = {}
        with mock.patch.object(mod, "_rag", return_value=rag), \
             mock.patch.object(mod, "RAG_AUTOSTART", True), \
             mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)):
            mod.register(actions)
        # Drive the autostart closure: sleep patched, start() called.
        with mock.patch.object(mod.time, "sleep", return_value=None):
            captured["t"]()
        rag.start.assert_called_once_with(initial_scan=True)

    def test_register_autostart_closure_swallows_failure(self):
        import threading as _thr
        mod, _, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)
        rag = _fake_rag(available=True)
        rag.start.side_effect = RuntimeError("index boom")
        actions = {}
        captured = {}
        with mock.patch.object(mod, "_rag", return_value=rag), \
             mock.patch.object(mod, "RAG_AUTOSTART", True), \
             mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)):
            mod.register(actions)
        with mock.patch.object(mod.time, "sleep", return_value=None):
            captured["t"]()   # must not raise


class RagRegisterConfigPushTests(unittest.TestCase):
    """register() must forward core.config's RAG_* knobs into the indexer.

    Regression for the dead-config defect (AUDIT_2026_07_21: 'nothing ever
    applies them to the indexer'): core/config.py defined RAG_INDEX_PATHS /
    RAG_EMBED_MODEL / RAG_OLLAMA_ENDPOINT / RAG_RERANKER_MODEL, but nothing
    in production called rag_indexer.configure(), so editing the constants
    (or overriding them via data/user_settings.json) was a silent no-op."""

    def setUp(self):
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    def test_register_pushes_config_into_indexer(self):
        import threading as _thr
        from core import config as real_cfg
        rag = _fake_rag(available=True)
        actions = {}
        captured = {}
        with mock.patch.object(real_cfg, "RAG_ENABLED", True, create=True), \
             mock.patch.object(real_cfg, "RAG_INDEX_PATHS",
                               ["X:/docs"], create=True), \
             mock.patch.object(real_cfg, "RAG_EMBED_MODEL",
                               "cfg-embed", create=True), \
             mock.patch.object(real_cfg, "RAG_OLLAMA_ENDPOINT",
                               "http://cfg:1/api/embeddings", create=True), \
             mock.patch.object(real_cfg, "RAG_RERANKER_MODEL",
                               "cfg-rerank", create=True), \
             mock.patch.object(self.mod, "_rag", return_value=rag), \
             mock.patch.object(self.mod, "RAG_AUTOSTART", True), \
             mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)):
            self.mod.register(actions)
        rag.configure.assert_called_once_with(
            rag_index_paths=["X:/docs"],
            rag_embed_model="cfg-embed",
            rag_ollama_endpoint="http://cfg:1/api/embeddings",
            rag_reranker_model="cfg-rerank",
        )
        # Drive the autostart closure and verify configure() preceded start()
        # — the indexer must scan the CONFIGURED paths, not its defaults.
        with mock.patch.object(self.mod.time, "sleep", return_value=None):
            captured["t"]()
        rag.start.assert_called_once_with(initial_scan=True)
        names = [c[0] for c in rag.mock_calls]
        self.assertLess(names.index("configure"), names.index("start"))

    def test_register_config_push_survives_missing_keys(self):
        # Config without the four knobs → getattr falls back to the indexer's
        # own module defaults; register() must not raise and still registers.
        from core import config as real_cfg
        rag = _fake_rag(available=False)  # unavailable → no autostart thread
        rag.RAG_INDEX_PATHS = ["D:/fallback-docs"]
        rag.RAG_EMBED_MODEL = "fallback-embed"
        rag.RAG_OLLAMA_ENDPOINT = "http://fallback:11434/api/embeddings"
        rag.RAG_RERANKER_MODEL = "fallback-rerank"
        keys = ("RAG_INDEX_PATHS", "RAG_EMBED_MODEL",
                "RAG_OLLAMA_ENDPOINT", "RAG_RERANKER_MODEL")
        saved = {k: getattr(real_cfg, k) for k in keys if hasattr(real_cfg, k)}
        actions = {}
        try:
            for k in saved:
                delattr(real_cfg, k)
            with mock.patch.object(real_cfg, "RAG_ENABLED", True, create=True), \
                 mock.patch.object(self.mod, "_rag", return_value=rag):
                self.mod.register(actions)  # must not raise
        finally:
            for k, v in saved.items():
                setattr(real_cfg, k, v)
        rag.configure.assert_called_once_with(
            rag_index_paths=["D:/fallback-docs"],
            rag_embed_model="fallback-embed",
            rag_ollama_endpoint="http://fallback:11434/api/embeddings",
            rag_reranker_model="fallback-rerank",
        )
        self.assertIn("rag_search", actions)


class RagConfigDeadKnobInvariantTests(unittest.TestCase):
    """Source-scanning guard against the dead-config bug class recurring:
    every top-level RAG_* constant in core/config.py (except RAG_ENABLED,
    the master switch personal_rag reads directly) must appear as a
    forwarded kwarg inside register() in skills/personal_rag.py. Adding a
    fifth RAG_* knob to core/config.py fails here until it is actually
    pushed into rag_indexer.configure()."""

    def test_every_config_rag_knob_is_forwarded_in_register(self):
        import ast
        import os
        import re
        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "core", "config.py"),
                  encoding="utf-8") as f:
            cfg_src = f.read()
        knobs = sorted(set(re.findall(r"(?m)^(RAG_[A-Z0-9_]+)\s*=", cfg_src)))
        knobs = [k for k in knobs if k != "RAG_ENABLED"]
        # Sanity: the four knobs this guard was written for must be seen.
        self.assertGreaterEqual(len(knobs), 4, knobs)
        with open(os.path.join(root, "skills", "personal_rag.py"),
                  encoding="utf-8") as f:
            skill_src = f.read()
        register_src = None
        for node in ast.parse(skill_src).body:
            if isinstance(node, ast.FunctionDef) and node.name == "register":
                register_src = ast.get_source_segment(skill_src, node)
                break
        self.assertIsNotNone(
            register_src, "skills/personal_rag.py must define register()")
        for k in knobs:
            kwarg = k.lower() + "="
            self.assertIn(
                kwarg, register_src,
                f"core/config.py defines {k} but personal_rag.register() never "
                f"forwards it via rag.configure({kwarg}...) — dead config "
                f"(AUDIT_2026_07_21: 'nothing ever applies them to the "
                f"indexer'). Wire it into the config push in register().")


if __name__ == "__main__":
    unittest.main()
