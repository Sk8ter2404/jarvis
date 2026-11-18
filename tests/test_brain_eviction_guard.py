"""Nothing in the background may unload the voice brain (2026-10-04).

WHY
===
Ollama runs with OLLAMA_MAX_LOADED_MODELS=1: a request naming a model that
is not loaded unloads whatever is. On 10-02 15:05:40 and 10-03 17:35:41 the
RAG indexer's boot scan (657 and 3,699 nomic-embed-text chunks) unloaded the
16.8 GB brain; the next brain loads took 54 s and 9 s and the brain was gone
for 9 minutes the second time (Ollama server.log). What these pin:

  * core.ollama_opts.eviction_risk says when a request would unload a
    loaded model (and treats "could not tell" as a reason to wait);
  * the RAG embedder sends NO request while it would, and the boot scan and
    the folder watcher wait and retry instead (the owner's own search still
    runs, and - review 2026-10-04, tests/test_brain_prefix_review.py - so
    does his "reindex my files");
  * core/orchestrator's Ollama call keeps the brain's keep_alive instead of
    resetting its residency to Ollama's 5-minute default.

Every class fails on origin/main d5931da (the guard does not exist there).
No network: urllib is faked at the boundary.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
import unittest
import urllib.request
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ollama_opts as oo       # noqa: E402
from core import rag_indexer as rag      # noqa: E402
from tests.test_rag_indexer import _RagBase  # noqa: E402


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ollama(loaded, record=None, embedding=(1.0, 0.0)):
    """A fake urlopen: GET /api/ps lists ``loaded`` (None = unreachable);
    an embeddings POST returns ``embedding``. ``record`` collects
    (method, url)."""
    def _urlopen(req, timeout=None):
        url = getattr(req, "full_url", "") or ""
        method = getattr(req, "method", None) or (
            "POST" if getattr(req, "data", None) else "GET")
        if record is not None:
            record.append((method, url))
        if url.endswith("/api/ps"):
            if loaded is None:
                raise OSError("connection refused")
            return _Resp({"models": [{"name": n} for n in loaded]})
        return _Resp({"embedding": list(embedding)})
    return _urlopen


class MaxLoadedModelsTests(unittest.TestCase):
    def test_reads_the_env_jarvis_persists(self):
        self.assertEqual(oo.max_loaded_models({}), 1)
        self.assertEqual(oo.max_loaded_models({oo.MAX_LOADED_ENV: "2"}), 2)
        self.assertEqual(oo.max_loaded_models({oo.MAX_LOADED_ENV: " 3 "}), 3)
        for junk in ("", "0", "-1", "two", "1.5"):
            self.assertEqual(
                oo.max_loaded_models({oo.MAX_LOADED_ENV: junk}), 1, junk)


class EvictionRiskTests(unittest.TestCase):
    def _risk(self, model, loaded, cap=1):
        with mock.patch.object(urllib.request, "urlopen", _ollama(loaded)):
            return oo.eviction_risk(model, "http://127.0.0.1:11434",
                                    timeout_s=1.0, max_loaded=cap)

    def test_the_boot_scan_case_is_refused(self):
        why = self._risk("nomic-embed-text", ["gemma4:26b-a4b-it-qat"])
        self.assertIn("would unload gemma4:26b-a4b-it-qat", why)
        self.assertIn("Ollama holds 1 model", why)

    def test_an_already_loaded_model_unloads_nothing(self):
        self.assertEqual(self._risk("nomic-embed-text",
                                    ["nomic-embed-text:latest"]), "")
        self.assertEqual(self._risk("gemma4:26b-a4b-it-qat",
                                    ["gemma4:26b-a4b-it-qat"]), "")

    def test_another_tag_of_the_same_family_is_a_load(self):
        self.assertTrue(self._risk("gemma4:12b", ["gemma4:26b-a4b-it-qat"]))

    def test_nothing_loaded_or_room_for_another(self):
        self.assertEqual(self._risk("nomic-embed-text", []), "")
        self.assertEqual(self._risk("nomic-embed-text",
                                    ["gemma4:26b-a4b-it-qat"], cap=2), "")
        self.assertTrue(self._risk("nomic-embed-text", ["a:1", "b:2"], cap=2))

    def test_unknown_is_a_reason_to_wait(self):
        self.assertIn("could not read", self._risk("nomic-embed-text", None))
        self.assertEqual(oo.eviction_risk("", "http://x"), "no model named")

    def test_the_cap_defaults_to_the_process_env_and_the_server(self):
        # Review 2026-10-04: the server's own cap counts too (its log's
        # "server config" line) - 2 here and 2 there co-loads; 2 here with
        # the server's unknown fails closed (see test_brain_prefix_review).
        with mock.patch.dict(os.environ, {oo.MAX_LOADED_ENV: "2"}), \
                mock.patch.object(oo, "server_max_loaded_models",
                                  return_value=2), \
                mock.patch.object(urllib.request, "urlopen",
                                  _ollama(["gemma4:26b-a4b-it-qat"])):
            self.assertEqual(oo.eviction_risk("nomic-embed-text",
                                              "http://h", timeout_s=1), "")
        with mock.patch.dict(os.environ, {oo.MAX_LOADED_ENV: "2"}), \
                mock.patch.object(oo, "server_max_loaded_models",
                                  return_value=None), \
                mock.patch.object(urllib.request, "urlopen",
                                  _ollama(["gemma4:26b-a4b-it-qat"])):
            self.assertIn("would unload", oo.eviction_risk(
                "nomic-embed-text", "http://h", timeout_s=1))


class EmbedderWaitsTests(_RagBase):
    def setUp(self):
        super().setUp()
        import core
        fake_gpu = types.ModuleType("core.gpu_state")
        fake_gpu.log_gpu_state = lambda model: None
        for p in (mock.patch.dict(sys.modules, {"core.gpu_state": fake_gpu}),
                  mock.patch.object(core, "gpu_state", fake_gpu, create=True),
                  mock.patch.dict(os.environ, {oo.MAX_LOADED_ENV: "1"})):
            p.start()
            self.addCleanup(p.stop)
        self.calls = []
        self.emb = rag._OllamaEmbedder(
            "nomic-embed-text", "http://127.0.0.1:11434/api/embeddings",
            batch_size=1, timeout=1.0)

    def _with(self, loaded):
        return mock.patch.object(rag.urllib.request, "urlopen",
                                 _ollama(loaded, record=self.calls))

    def test_no_embedding_is_sent_while_the_brain_is_loaded(self):
        with self._with(["gemma4:26b-a4b-it-qat"]):
            with self.assertRaises(rag.EmbedDeferred) as cm:
                self.emb._embed_one("hello")
        self.assertIn("gemma4:26b-a4b-it-qat", str(cm.exception))
        self.assertEqual([m for m, _ in self.calls], ["GET"])
        self.assertTrue(self.calls[0][1].endswith("/api/ps"))

    def test_it_embeds_when_that_unloads_nothing(self):
        with self._with(["nomic-embed-text:latest", "gemma4:26b-a4b-it-qat"]):
            self.assertEqual(self.emb._embed_one("hello"), [1.0, 0.0])
        self.assertEqual([m for m, _ in self.calls], ["GET", "POST"])

    def test_the_owners_search_may_unload_it(self):
        with self._with(["gemma4:26b-a4b-it-qat"]):
            out = self.emb.encode(["q"], convert_to_numpy=False,
                                  allow_evict=True)
        self.assertEqual(out, [[1.0, 0.0]])
        self.assertEqual([m for m, _ in self.calls], ["POST"])

    def test_every_request_asks_so_a_brain_loaded_mid_file_stops_it(self):
        loaded = [["nomic-embed-text:latest"], ["nomic-embed-text:latest"],
                  ["gemma4:26b-a4b-it-qat"]]

        def _urlopen(req, timeout=None):
            url = req.full_url
            self.calls.append(url)
            if url.endswith("/api/ps"):
                return _Resp({"models": [{"name": n}
                                         for n in loaded.pop(0)]})
            return _Resp({"embedding": [1.0]})
        with mock.patch.object(rag.urllib.request, "urlopen", _urlopen):
            with self.assertRaises(rag.EmbedDeferred):
                self.emb.encode(["a", "b", "c"], batch_size=1,
                                convert_to_numpy=False)
        self.assertEqual(sum(u.endswith("/api/embeddings")
                             for u in self.calls), 2)

    def test_search_passes_allow_evict(self):
        seen = {}

        class _Emb:
            def encode(self, texts, **kw):
                seen.update(kw)
                raise RuntimeError("stop here")
        self._install_collection()
        rag._embed_model = _Emb()
        self.assertEqual(rag.search("my tax forms"), [])
        self.assertIs(seen.get("allow_evict"), True)


class _DeferringEmbedder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kw):
        self.calls += 1
        raise rag.EmbedDeferred("loading nomic-embed-text would unload "
                                "gemma4:26b-a4b-it-qat (Ollama holds 1 model)")


class IndexerWaitsTests(_RagBase):
    def setUp(self):
        super().setUp()
        self.coll = self._install_collection()
        self.emb = _DeferringEmbedder()
        rag._embed_model = self.emb
        root = os.path.join(self.tmp, "docs")
        os.makedirs(root)
        for i in range(3):
            self._write(os.path.join("docs", f"n{i}.txt"), f"note {i} " * 50)
        rag.RAG_INDEX_PATHS = [root]
        rag._scan_retry_at[0] = 0.0
        rag._deferral_logged[0] = ""
        self.addCleanup(lambda: rag._scan_retry_at.__setitem__(0, 0.0))

    def test_the_scan_stops_and_reports_the_deferral(self):
        summary = rag.index_once()
        self.assertTrue(summary["ok"])
        self.assertIn("would unload gemma4", summary["deferred"])
        self.assertEqual(self.emb.calls, 1)          # stopped at the 1st file
        self.assertEqual(rag._stats["errors"], 0)    # not an error

    def test_an_unfinished_walk_never_garbage_collects(self):
        # A chunk of a file the walk never reached must survive. v3.0.26:
        # a deferred EMBEDDING no longer stops the walk (it always finishes,
        # see WalkThenQueueTests), so the unfinished walk here is a stop.
        self.coll.add(ids=["old:0"], embeddings=[[0.1, 0.2, 0.3]],
                      documents=["kept"],
                      metadatas=[{"file_id": "old", "path": "/gone/old.txt",
                                  "chunk_index": 0}])
        rag._stop_flag.set()
        self.addCleanup(rag._stop_flag.clear)
        rag.index_once()
        self.assertIn("old:0", self.coll.get()["ids"])

    def test_a_finished_walk_collects_a_deleted_file_while_embedding_waits(
            self):
        self.coll.add(ids=["old:0"], embeddings=[[0.1, 0.2, 0.3]],
                      documents=["gone"],
                      metadatas=[{"file_id": "old", "path": "/gone/old.txt",
                                  "chunk_index": 0}])
        summary = rag.index_once()
        self.assertTrue(summary["deferred"])
        self.assertNotIn("old:0", self.coll.get()["ids"])

    def test_the_daemon_schedules_a_retry(self):
        t0 = time.time()
        rag._run_scan("initial scan")
        self.assertGreaterEqual(rag._scan_retry_at[0],
                                t0 + rag.RAG_DEFER_RETRY_S - 1)

    def test_a_completed_scan_clears_the_retry(self):
        rag._scan_retry_at[0] = 123.0
        rag._embed_model = None
        rag.RAG_INDEX_PATHS = []
        self._install_embedder()
        rag._run_scan("rescan after deferral")
        self.assertEqual(rag._scan_retry_at[0], 0.0)

    def test_the_watcher_retries_a_deferred_file_later(self):
        path = os.path.join(self.tmp, "docs", "n0.txt")
        retried = []

        def _index(p):
            retried.append(p)
            rag._stop_flag.set()            # one pass is enough
            raise rag.EmbedDeferred("busy")
        stamps = iter([1000.0, 1003.0])
        rag._event_q.put(path)
        fake_time = types.SimpleNamespace(time=lambda: next(stamps, 1003.0))
        logged = []
        with mock.patch.object(rag, "_index_file", _index), \
                mock.patch.object(rag, "time", fake_time), \
                mock.patch.object(rag, "_log_deferral",
                                  lambda where, why: logged.append(where)):
            rag._drain_event_queue()
        self.assertEqual(retried, [path])
        self.assertEqual(logged, ["re-index"])
        self.assertEqual(rag._stats["errors"], 0)


class BackgroundNeverLoadsTests(_RagBase):
    """v3.0.26 (reliability-06): with the brain out, the guard let the
    background scan LOAD nomic-embed-text into Ollama's only slot (11-16
    22:26:26), so the next brain call had to load the brain back. A
    background request now goes only while the embed model is loaded
    already; the owner's search and reindex still load it."""

    MODEL = "nomic-embed-text"
    URL = "http://127.0.0.1:11434/api/embeddings"

    def setUp(self):
        super().setUp()
        import core
        fake_gpu = types.ModuleType("core.gpu_state")
        fake_gpu.log_gpu_state = lambda model: None
        for p in (mock.patch.dict(sys.modules, {"core.gpu_state": fake_gpu}),
                  mock.patch.object(core, "gpu_state", fake_gpu, create=True),
                  mock.patch.dict(os.environ, {oo.MAX_LOADED_ENV: "1"}),
                  mock.patch("core.local_traffic.GATE.defer_reason",
                             return_value=None)):
            p.start()
            self.addCleanup(p.stop)
        self.calls = []
        self.emb = rag._OllamaEmbedder(self.MODEL, self.URL, batch_size=1,
                                       timeout=1.0)

    def _with(self, loaded):
        return mock.patch.object(rag.urllib.request, "urlopen",
                                 _ollama(loaded, record=self.calls))

    def test_the_live_default_is_off(self):
        self.assertIs(rag.RAG_BACKGROUND_MAY_LOAD_EMBED, False)

    def test_an_empty_one_slot_server_is_a_wait(self):
        with self._with([]):
            why = rag._embed_eviction_reason(self.MODEL, self.URL)
            self.assertIn("only model slot", why)
            self.assertIn("only model slot", self.emb.defer_reason())
            with self.assertRaises(rag.EmbedDeferred):
                self.emb._embed_one("hello")
        self.assertNotIn("POST", [m for m, _ in self.calls])

    def test_a_loaded_embed_model_goes(self):
        with self._with(["nomic-embed-text:latest"]):
            self.assertEqual(self.emb.defer_reason(), "")
            self.assertEqual(self.emb._embed_one("hello"), [1.0, 0.0])

    def test_the_owners_search_still_loads_it(self):
        with self._with([]):
            out = self.emb.encode(["q"], convert_to_numpy=False,
                                  allow_evict=True)
        self.assertEqual(out, [[1.0, 0.0]])

    def test_a_two_model_server_may_still_co_load(self):
        with self._with([]), mock.patch.object(oo, "effective_max_loaded",
                                               return_value=2):
            self.assertEqual(self.emb.defer_reason(), "")

    def test_the_flag_restores_the_old_load(self):
        with mock.patch.object(rag, "RAG_BACKGROUND_MAY_LOAD_EMBED", True), \
                self._with([]):
            self.assertEqual(self.emb.defer_reason(), "")

    def test_an_unreadable_server_is_a_wait(self):
        with self._with(None):
            self.assertIn("could not read", self.emb.defer_reason())

    def test_index_file_asks_before_reading_the_file(self):
        """A deferral used to parse the file (PDF / DOCX) first and throw
        the text away, every 10 minutes."""
        self._install_collection()
        rag._embed_model = self.emb
        path = self._write("docs/report.txt", "quarterly numbers " * 40)
        read = []
        real_read = rag._read_text
        with self._with([]), mock.patch.object(
                rag, "_read_text",
                lambda p: read.append(p) or real_read(p)):
            with self.assertRaises(rag.EmbedDeferred):
                rag._index_file(path)
        self.assertEqual(read, [], "the file was read before the check")
        self.assertEqual([m for m, _ in self.calls], ["GET"])

    def test_an_owner_scan_does_not_ask(self):
        self._install_collection()
        rag._embed_model = self.emb
        path = self._write("docs/report.txt", "quarterly numbers " * 40)
        rag._owner_scan.active = True
        self.addCleanup(setattr, rag._owner_scan, "active", False)
        with self._with([]), mock.patch.object(rag, "_wait_owner_quiet"):
            self.assertGreater(rag._index_file(path), 0)
        self.assertIn("POST", [m for m, _ in self.calls])


class _GatedEmbedder:
    """An embedder whose background gate the test opens and closes."""

    def __init__(self):
        self.reason = "loading nomic-embed-text would take Ollama's only " \
                      "model slot"
        self.encoded = 0

    def defer_reason(self):
        return self.reason

    def encode(self, texts, **kw):
        import numpy as np
        self.encoded += len(texts)
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype="float32")


class WalkThenQueueTests(_RagBase):
    """v3.0.26 (reliability-06): no scan ever finished (no '[rag] initial
    scan:' in 47 sessions) and every 10-minute retry walked the roots
    again. The walk now always finishes and queues the changed files; a
    deferral stops only the embedding, and the retry resumes the queue."""

    def setUp(self):
        super().setUp()
        self.coll = self._install_collection()
        self.emb = _GatedEmbedder()
        rag._embed_model = self.emb
        self.root = os.path.join(self.tmp, "docs")
        self.paths = [self._write(os.path.join("docs", f"n{i}.txt"),
                                  f"note {i} " * 50) for i in range(3)]
        # one file indexed already, with its current signature
        self.done = self._write(os.path.join("docs", "done.txt"), "old " * 50)
        fid = rag._file_id(self.done)
        self.coll.add(ids=[fid + ":0"], embeddings=[[0.0, 1.0, 0.0]],
                      documents=["old"],
                      metadatas=[{"file_id": fid, "path": self.done,
                                  "sig": rag._file_signature(self.done),
                                  "chunk_index": 0}])
        rag.RAG_INDEX_PATHS = [self.root]
        rag._scan_retry_at[0] = 0.0
        rag._deferral_logged[0] = ""
        self.addCleanup(lambda: rag._scan_retry_at.__setitem__(0, 0.0))
        self.addCleanup(lambda: rag._deferral_logged.__setitem__(0, ""))

    def test_a_deferred_scan_walks_everything_and_reads_nothing(self):
        read, lookups = [], []
        real_lookup = rag._existing_signature
        with mock.patch.object(rag, "_read_text",
                               lambda p: read.append(p) or "x"), \
                mock.patch.object(rag, "_existing_signature",
                                  lambda fid: lookups.append(fid)
                                  or real_lookup(fid)):
            summary = rag.index_once()
        self.assertIn("only model slot", summary.get("deferred", ""))
        self.assertEqual(summary["files_seen"], 4)
        self.assertEqual(summary["pending"], 3)
        self.assertEqual(read, [])
        # The walk reads every signature in ONE bulk get; only _index_file's
        # own re-check of the first queued file looks one up.
        self.assertLessEqual(len(lookups), 1)
        self.assertEqual(sorted(rag._pending_paths), sorted(self.paths))
        self.assertNotIn(self.done, rag._pending_paths)
        self.assertGreater(rag._last_walk[0], 0.0)
        self.assertEqual(rag.status()["pending"], 3)

    def test_the_retry_resumes_the_queue_and_never_walks(self):
        rag.index_once()
        self.emb.reason = ""                 # the embed model is loaded now
        out = []
        with mock.patch.object(rag, "_iter_files",
                               side_effect=AssertionError("walked again")), \
                mock.patch("builtins.print", lambda *a, **k: out.append(
                    " ".join(str(x) for x in a))):
            res = rag._resume_pending("resume after deferral")
        self.assertEqual(res["embedded_now"], 3)
        self.assertEqual(res["pending"], 0)
        self.assertEqual(rag._pending_paths, [])
        self.assertEqual(rag._scan_retry_at[0], 0.0)
        self.assertGreater(rag._last_full_scan_ts, 0.0)
        self.assertTrue(any("index up to date" in line for line in out), out)
        self.assertEqual(self.coll.count(), 4)

    def test_a_retry_that_is_still_deferred_keeps_the_queue(self):
        rag.index_once()
        t0 = time.time()
        res = rag._resume_pending("resume after deferral")
        self.assertTrue(res["deferred"])
        self.assertEqual(len(rag._pending_paths), 3)
        self.assertGreaterEqual(rag._scan_retry_at[0],
                                t0 + rag.RAG_DEFER_RETRY_S - 1)

    def test_the_drain_loop_resumes_the_queue_instead_of_walking(self):
        rag._scan_retry_at[0] = 1.0                 # long due
        resumed, walked = [], []

        def _resume(label):
            resumed.append(label)
            rag._stop_flag.set()
            return {}

        def _walk(label):
            walked.append(label)
            rag._stop_flag.set()
            return {}
        with mock.patch.object(rag, "_resume_pending", _resume), \
                mock.patch.object(rag, "_run_scan", _walk):
            rag._drain_event_queue()
        self.assertEqual(walked, [], "the retry walked the roots again")
        self.assertEqual(resumed, ["resume after deferral"])

    def test_the_daemon_logs_that_the_walk_finished(self):
        out = []
        with mock.patch("builtins.print", lambda *a, **k: out.append(
                " ".join(str(x) for x in a))):
            rag._run_scan("initial scan")
        line = [x for x in out if "[rag] initial scan: walk done" in x]
        self.assertTrue(line, out)
        self.assertIn("4 file(s), 3 to embed", line[0])


class OrchestratorKeepAliveTests(unittest.TestCase):
    def _post(self, monolith):
        from core import orchestrator as orch
        sent = {}

        def _urlopen(req, timeout=None):
            sent.update(json.loads(req.data.decode("utf-8")))
            return _Resp({"message": {"content": "done"}})
        mods = {} if monolith is None else {"bobert_companion": monolith}
        with mock.patch.dict(sys.modules, mods), \
                mock.patch.object(urllib.request, "urlopen", _urlopen):
            if monolith is None:
                sys.modules.pop("bobert_companion", None)
            self.assertEqual(orch._ollama_call("gemma4:26b-a4b-it-qat",
                                               "sys", "hi"), "done")
        return sent

    def test_it_sends_the_running_brains_keep_alive(self):
        bc = types.ModuleType("bobert_companion")
        bc._local_keep_alive = lambda: "24h"
        sent = self._post(bc)
        self.assertEqual(sent["keep_alive"], "24h")
        self.assertIn("num_ctx", sent["options"])

    def test_standalone_it_sends_none_as_before(self):
        saved = sys.modules.get("bobert_companion")
        try:
            sent = self._post(None)
        finally:
            if saved is not None:
                sys.modules["bobert_companion"] = saved
        self.assertNotIn("keep_alive", sent)


if __name__ == "__main__":
    unittest.main()
