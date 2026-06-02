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
            [{"path": r"C:\docs\report.pdf", "snippet": "q3"}])
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


if __name__ == "__main__":
    unittest.main()
