"""Review fixes to the brain-prefix branch (2026-10-04) - the light tier.

WHY
===
An adversarial review of claude/brain-prefix-stable found, and these pin:

1. DENSE TEXT. The character estimate (3.7 characters per token) counted a
   sensor table, JSON, URLs and file paths at 0.29-0.68 of their real size:
   the brain's tokenizer spends a token on every digit. A 6,000-character
   JSON action result reached Ollama at ~18.6k tokens of a 16,384 window
   while the budget said it fit. prompt_budget.estimate_tokens is now the
   larger of the character estimate and dense_estimate. The pinned counts
   below were measured offline with the brain's own vocabulary (the GGUF's
   tokens + merges).
2. HALVED WINDOW. Ollama cuts a prompt longer than its runner's window to
   about HALF of it (server.log: limit=8195 keep=5 or limit=8194 keep=4 on
   all 30 cuts of 10-01..10-04, all at num_ctx 16,384). 8,195 is therefore
   never a smaller window, and ObservedWindow must not learn it.
3. THE SERVER'S CAP. The eviction guard trusted OLLAMA_MAX_LOADED_MODELS from
   JARVIS's own environment; the server reads it only at its own start. It
   now trusts min(own, the running server's logged value), 1 when unknown,
   and 1 for good once a co-load unloaded something anyway.
4. OWNER REINDEX. "Reindex my files" never ran while the brain was loaded
   (with keep_alive 24h: always) although JARVIS said it was reindexing. An
   owner scan now embeds like his search does, after waiting out his own
   utterance / turn; the reply says when the brain will reload; rag_status
   says when indexing is waiting.
5. While the server holds one model, background embeddings also wait while
   the owner is talking to JARVIS (an embedding queued behind his turn's
   brain request would unload the brain right after it answered).

Every class fails on 5e4d596 (the branch before the review). No network,
no real Ollama log: urllib and the log path are faked.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
import urllib.request
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import local_traffic as lt     # noqa: E402
from core import ollama_opts as oo       # noqa: E402
from core import prompt_budget as pb     # noqa: E402
from core import rag_indexer as rag      # noqa: E402
from tests.test_rag_indexer import _FakeEmbedder, _RagBase  # noqa: E402

NL = "\n"


# ════════════════════════════════════════════════════════════════════════════
#  1. Dense text is not under-counted
# ════════════════════════════════════════════════════════════════════════════
# (text, tokens the brain's tokenizer really uses - measured offline)
_PINNED = [
    ("Sensor 12: 45.3 C | 67% | 3021 RPM", 23),
    ('{"id": 12345, "name": "item_3", "v": 3.25}', 27),
    ("https://www.example3.com/news/2026/10/04/article-12345"
     "?utm_source=x&id=123", 41),
    ("C:/Projects/Reports/Q3/Lab_4/report_v2.docx  123 KB  2026-09-14 08:05",
     41),
    ("GPU Hot Spot: 71.4 C | 88% | 2140 RPM\nCPU Package: 64.0 C | 31% | "
     "1210 RPM", 44),
]


def _sensors(n):
    return NL.join(f"Sensor {i}: {20 + (i * 37) % 75}.{i % 10} C | "
                   f"{(i * 13) % 100}% | {300 + (i * 977) % 4900} RPM"
                   for i in range(n))


def _json(n):
    return json.dumps([{"id": 1000 + (i * 7919) % 98999, "name": f"item_{i}",
                        "v": round(((i * 31) % 1000) / 10.0, 1),
                        "ts": "2026-10-04T15:%02d:%02dZ" % (i % 60,
                                                             (i * 7) % 60)}
                       for i in range(n)])


def _urls(n):
    return NL.join(f"https://www.example{i}.com/news/2026/10/04/article-"
                   f"{10000 + (i * 7907) % 89999}?id={100 + i % 900}"
                   for i in range(n))


# (6,000-character result, real tokens measured offline)
_RESULTS = [(_sensors(200)[:6000], 4123), (_json(120)[:6000], 4228),
            (_urls(100)[:6000], 3514)]


class DenseEstimateTests(unittest.TestCase):
    def test_pinned_dense_samples_are_never_under_counted(self):
        for text, real in _PINNED:
            self.assertGreaterEqual(pb.estimate_tokens(text), real, text)
            # ...and not wildly over (a prompt estimated 1.33x its real size
            # would read as truncated).
            self.assertLessEqual(pb.estimate_tokens(text), real * 1.3, text)

    def test_a_6000_character_result_is_counted_near_its_real_size(self):
        for text, real in _RESULTS:
            est = pb.estimate_tokens(text)
            self.assertGreaterEqual(est, real, text[:60])
            self.assertLessEqual(est, real * 1.2, text[:60])

    def test_every_digit_and_line_break_counts(self):
        # Each ASCII digit and each line break is at least one token of the
        # brain's tokenizer: a lower bound no estimate may go under.
        for text, _real in _RESULTS:
            floor = sum(c.isdigit() for c in text) + text.count(NL)
            self.assertGreaterEqual(pb.estimate_tokens(text), floor)

    def test_prose_and_long_words_keep_the_character_estimate(self):
        prose = ("The telescope captured new images of the pillars, revealing "
                 "stars forming within the clouds of gas and dust. ") * 40
        self.assertEqual(pb.estimate_tokens(prose),
                         -(-len(prose) // pb.CHARS_PER_TOKEN))
        self.assertEqual(pb.estimate_tokens("S" * 50000),
                         -(-50000 // pb.CHARS_PER_TOKEN))

    def test_chat_estimates_and_measures_use_it(self):
        text = _RESULTS[1][0]
        msgs = [{"role": "user", "content": text}]
        self.assertGreaterEqual(pb.estimate_chat_tokens("sys", msgs),
                                _RESULTS[1][1])
        ex = pb.ExactCounts()
        self.assertGreaterEqual(
            pb.measure_chat_tokens("sys", msgs, model="m", exact=ex),
            _RESULTS[1][1])
        # after an exact prefix the estimated tail is dense-aware too
        ex.note("m", "sys", [], 9)
        self.assertGreaterEqual(
            pb.measure_chat_tokens("sys", msgs, model="m", exact=ex),
            9 + _RESULTS[1][1])

    def test_junk_never_raises(self):
        for junk in (None, 12, b"bytes", ["x"]):
            self.assertEqual(pb.estimate_tokens(junk), 0)


# ════════════════════════════════════════════════════════════════════════════
#  2. Ollama's halving is not a window
# ════════════════════════════════════════════════════════════════════════════
class HalvedWindowTests(unittest.TestCase):
    def test_the_logged_cuts_are_the_halving_signature(self):
        self.assertTrue(pb.halved_window(8195, 16384))
        self.assertTrue(pb.halved_window(8194, 16384))
        self.assertTrue(pb.halved_window(6146, 12288))
        self.assertFalse(pb.halved_window(4098, 16384))
        self.assertFalse(pb.halved_window(15200, 16384))
        for junk in (None, "x"):
            self.assertFalse(pb.halved_window(junk, 16384))
            self.assertFalse(pb.halved_window(8195, junk))

    def test_a_prompt_that_should_have_fit_and_was_halved_teaches_nothing(self):
        # The review's probe: ~16.2k estimated (inside num_ctx), cut to 8,195
        # - the estimate was low, the runner's window was 16,384.
        w = pb.ObservedWindow(clock=lambda: 0.0)
        self.assertFalse(w.note(16200, 8195, num_ctx=16384))
        self.assertEqual(w.limit, 0)
        self.assertEqual(w.effective(16384), 16384)

    def test_a_cut_at_another_count_is_still_learned(self):
        w = pb.ObservedWindow(clock=lambda: 0.0)
        self.assertTrue(w.note(15800, 4098, num_ctx=16384))
        self.assertEqual(w.effective(16384), 4098)


# ════════════════════════════════════════════════════════════════════════════
#  3. The guard trusts the running server's cap, failing closed
# ════════════════════════════════════════════════════════════════════════════
def _config_line(cap):
    v = "" if cap is None else str(cap)
    return ('time=2026-10-04T13:00:00.000-05:00 level=INFO source=routes.go:2099'
            ' msg="server config" env="map[OLLAMA_CONTEXT_LENGTH:262144 '
            f'OLLAMA_KEEP_ALIVE:5m0s OLLAMA_MAX_LOADED_MODELS:{v} '
            'OLLAMA_NUM_PARALLEL:1]"\n')


class _LogBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = os.path.join(tmp.name, "server.log")
        oo._server_log_scan.clear()
        self.addCleanup(oo._server_log_scan.clear)
        oo._coload_evicted[0] = False
        self.addCleanup(oo._coload_evicted.__setitem__, 0, False)

    def _write(self, *lines, mode="w"):
        with open(self.log, mode, encoding="utf-8", newline="") as fh:
            fh.write("".join(lines))


class ServerCapTests(_LogBase):
    def test_the_last_config_line_wins_and_appends_are_read(self):
        self._write(_config_line(1), "a log line\n", _config_line(2),
                    "level=INFO msg=\"llama runner started\"\n")
        self.assertEqual(oo.server_max_loaded_models(log_path=self.log), 2)
        self._write("x\n", _config_line(1), mode="a")
        self.assertEqual(oo.server_max_loaded_models(log_path=self.log), 1)

    def test_a_new_shorter_log_is_read_from_the_start(self):
        self._write(_config_line(1), "y" * 4000 + "\n")
        self.assertEqual(oo.server_max_loaded_models(log_path=self.log), 1)
        self._write(_config_line(3))
        self.assertEqual(oo.server_max_loaded_models(log_path=self.log), 3)

    def test_unknown_is_none(self):
        self.assertIsNone(oo.server_max_loaded_models(
            log_path=os.path.join(os.path.dirname(self.log), "missing.log")))
        self._write(_config_line(None))           # the server's own default
        self.assertIsNone(oo.server_max_loaded_models(log_path=self.log))
        self._write(_config_line(2))
        self.assertIsNone(oo.server_max_loaded_models(
            "http://192.0.2.5:11434", log_path=self.log))   # remote server

    def test_effective_cap_is_the_smaller_and_fails_closed(self):
        two = {oo.MAX_LOADED_ENV: "2"}
        self._write(_config_line(1))
        self.assertEqual(oo.effective_max_loaded(environ=two,
                                                 log_path=self.log), 1)
        self._write(_config_line(2))
        oo._server_log_scan.clear()
        self.assertEqual(oo.effective_max_loaded(environ=two,
                                                 log_path=self.log), 2)
        self.assertEqual(oo.effective_max_loaded(
            environ={oo.MAX_LOADED_ENV: "4"}, log_path=self.log), 2)
        with mock.patch.object(oo, "ollama_server_log", return_value=None):
            self.assertEqual(oo.effective_max_loaded(environ=two), 1)
        # this process at 1: the log is not even read
        with mock.patch.object(oo, "server_max_loaded_models") as srv:
            self.assertEqual(oo.effective_max_loaded(environ={}), 1)
        srv.assert_not_called()

    def test_eviction_risk_defaults_to_the_effective_cap(self):
        # The fail-open case: 2 in JARVIS's environment, the server started
        # before the change still at 1 -> the boot scan must wait.
        self._write(_config_line(1))

        def _urlopen(req, timeout=None):
            return _Resp({"models": [{"name": "gemma4:26b-a4b-it-qat"}]})
        with mock.patch.dict(os.environ, {oo.MAX_LOADED_ENV: "2"}), \
                mock.patch.object(oo, "ollama_server_log",
                                  return_value=self.log), \
                mock.patch.object(urllib.request, "urlopen", _urlopen):
            why = oo.eviction_risk("nomic-embed-text", "http://127.0.0.1:11434",
                                   timeout_s=1)
            self.assertIn("would unload gemma4", why)
            self._write(_config_line(2))
            oo._server_log_scan.clear()
            self.assertEqual(oo.eviction_risk(
                "nomic-embed-text", "http://127.0.0.1:11434", timeout_s=1), "")

    def test_a_co_load_that_unloaded_something_pins_the_cap_to_one(self):
        self._write(_config_line(2))
        two = {oo.MAX_LOADED_ENV: "2"}
        self.assertFalse(oo.note_coload(["gemma4:26b"], None))
        self.assertFalse(oo.note_coload(["gemma4:26b"],
                                        ["gemma4:26b", "nomic:latest"]))
        self.assertEqual(oo.effective_max_loaded(environ=two,
                                                 log_path=self.log), 2)
        with mock.patch("builtins.print"):
            self.assertTrue(oo.note_coload(["gemma4:26b"], ["nomic:latest"],
                                           "nomic"))
        self.assertEqual(oo.effective_max_loaded(environ=two,
                                                 log_path=self.log), 1)


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class GatePublicReasonTests(unittest.TestCase):
    def test_defer_reason_is_the_predicates_and_never_raises(self):
        g = lt.BackgroundGate(defer_reason=lambda: "turn")
        self.assertEqual(g.defer_reason(), "turn")
        self.assertIsNone(lt.BackgroundGate().defer_reason())

        def _boom():
            raise RuntimeError("x")
        self.assertIsNone(lt.BackgroundGate(defer_reason=_boom).defer_reason())


# ════════════════════════════════════════════════════════════════════════════
#  4/5. The embedder: owner scans, the owner-talking gate, co-load checks
# ════════════════════════════════════════════════════════════════════════════
class _EmbBase(_RagBase):
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
        oo._coload_evicted[0] = False
        self.addCleanup(oo._coload_evicted.__setitem__, 0, False)
        self.gate = mock.patch.object(lt.GATE, "defer_reason",
                                      return_value=None)
        self.gate_reason = self.gate.start()
        self.addCleanup(self.gate.stop)
        self.calls = []
        self.loaded = [["gemma4:26b-a4b-it-qat"]]
        self.emb = rag._OllamaEmbedder(
            "nomic-embed-text", "http://127.0.0.1:11434/api/embeddings",
            batch_size=1, timeout=1.0)

    def _urlopen(self, req, timeout=None):
        url = req.full_url
        if url.endswith("/api/ps"):
            self.calls.append("GET")
            now = self.loaded[0] if len(self.loaded) == 1 else self.loaded.pop(0)
            return _Resp({"models": [{"name": n} for n in now]})
        self.calls.append("POST")
        return _Resp({"embedding": [1.0, 0.0]})

    def _patched(self):
        return mock.patch.object(rag.urllib.request, "urlopen", self._urlopen)


class OwnerTalkingGateTests(_EmbBase):
    def test_a_background_embedding_waits_while_the_owner_talks(self):
        self.loaded = [[]]                       # nothing loaded: no risk
        self.gate_reason.return_value = "conversation"
        with self._patched(), self.assertRaises(rag.EmbedDeferred) as cm:
            self.emb._embed_one("hello")
        self.assertIn("talking to JARVIS (conversation)", str(cm.exception))
        self.assertNotIn("POST", self.calls)
        self.gate_reason.return_value = None
        with self._patched():
            self.assertEqual(self.emb._embed_one("hello"), [1.0, 0.0])

    def test_not_when_the_server_holds_two(self):
        self.loaded = [[]]
        self.gate_reason.return_value = "conversation"
        with self._patched(), \
                mock.patch.object(oo, "effective_max_loaded", return_value=2):
            self.assertEqual(self.emb._embed_one("hello"), [1.0, 0.0])

    def test_a_co_load_that_unloads_the_brain_latches_the_guard(self):
        # The server may hold 2 by its log, but the embed model did not fit
        # next to the brain: Ollama unloaded it anyway.
        self.loaded = [["gemma4:26b-a4b-it-qat"], ["nomic-embed-text:latest"]]
        with self._patched(), \
                mock.patch.object(oo, "effective_max_loaded",
                                  side_effect=lambda *a, **k:
                                  1 if oo._coload_evicted[0] else 2), \
                mock.patch("builtins.print"):
            self.assertEqual(self.emb._embed_one("a"), [1.0, 0.0])
        self.assertEqual(self.calls, ["GET", "POST", "GET"])
        self.assertTrue(oo._coload_evicted[0])


class OwnerScanTests(_EmbBase):
    def test_an_owner_scan_embeds_while_the_brain_is_loaded(self):
        rag._owner_scan.active = True
        self.addCleanup(setattr, rag._owner_scan, "active", False)
        with self._patched():
            out = self.emb.encode(["a", "b"], convert_to_numpy=False)
        self.assertEqual(out, [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(self.calls, ["POST", "POST"])

    def test_it_waits_out_the_owners_own_utterance_and_turn(self):
        rag._owner_scan.active = True
        self.addCleanup(setattr, rag._owner_scan, "active", False)
        seq = ["turn", "utterance", "conversation", None]
        self.gate_reason.side_effect = lambda: seq.pop(0) if len(seq) > 1 else seq[0]
        sleeps = []
        with self._patched(), \
                mock.patch.object(rag.time, "sleep", sleeps.append):
            self.emb.encode(["a"], convert_to_numpy=False)
        # turn and utterance wait; the soft 'conversation' window does not
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(self.calls, ["POST"])

    def test_index_once_marks_only_its_own_thread_and_restores(self):
        seen = []
        self._install_collection()

        class _Emb(_FakeEmbedder):
            def encode(self, texts, **kw):
                seen.append(getattr(rag._owner_scan, "active", False))
                return super().encode(texts, **kw)
        rag._embed_model = _Emb()
        root = os.path.join(self.tmp, "docs")
        os.makedirs(root)
        self._write(os.path.join("docs", "n.txt"), "note " * 50)
        rag.RAG_INDEX_PATHS = [root]
        rag.index_once(owner=True)
        self.assertEqual(seen, [True])
        self.assertFalse(getattr(rag._owner_scan, "active", False))
        self._write(os.path.join("docs", "n.txt"), "changed " * 50)
        rag.index_once()
        self.assertEqual(seen, [True, False])

    def test_a_background_scan_still_waits(self):
        with self._patched(), self.assertRaises(rag.EmbedDeferred):
            self.emb.encode(["a"], convert_to_numpy=False)
        self.assertNotIn("POST", self.calls)


class DeferralStatusTests(_RagBase):
    def test_status_reports_a_waiting_index(self):
        rag._deferral_logged[0] = ""
        rag._scan_retry_at[0] = 0.0
        self.addCleanup(rag._deferral_logged.__setitem__, 0, "")
        self.addCleanup(rag._scan_retry_at.__setitem__, 0, 0.0)
        s = rag.status()
        self.assertEqual((s["deferred"], s["retry_at"]), ("", 0.0))
        rag._log_deferral("index scan", "loading nomic-embed-text would "
                          "unload gemma4 (Ollama holds 1 model)")
        rag._scan_retry_at[0] = 1234.0
        s = rag.status()
        self.assertIn("would unload gemma4", s["deferred"])
        self.assertEqual(s["retry_at"], 1234.0)

    def test_a_finished_scan_clears_it(self):
        rag._deferral_logged[0] = "old reason"
        self.addCleanup(rag._deferral_logged.__setitem__, 0, "")
        self._install_collection()
        self._install_embedder()
        rag.RAG_INDEX_PATHS = []
        rag.index_once()
        self.assertEqual(rag.status()["deferred"], "")


# ════════════════════════════════════════════════════════════════════════════
#  4. What JARVIS says: "reindex my files" and "RAG status"
# ════════════════════════════════════════════════════════════════════════════
class ReindexReplyTests(unittest.TestCase):
    def setUp(self):
        from tests.skills.test_personal_rag import _load_rag_skill
        self.mod, self.actions, patcher = _load_rag_skill()
        self.addCleanup(patcher.stop)

    def _reindex(self, unloads):
        import threading as _thr
        from tests.skills.test_personal_rag import _fake_rag
        rag_ = _fake_rag()
        rag_.embed_would_unload.return_value = unloads
        rag_.index_once.return_value = {"ok": True}
        captured = {}
        with mock.patch.object(self.mod, "_rag", return_value=rag_), \
                mock.patch.object(_thr.Thread, "start",
                                  lambda t: captured.__setitem__("t", t._target)):
            out = self.actions["rag_reindex"]("")
        with mock.patch("builtins.print"):
            captured["t"]()
        return out, rag_

    def test_the_reindex_is_an_owner_scan(self):
        _out, rag_ = self._reindex("")
        rag_.index_once.assert_called_once_with(owner=True)

    def test_the_reply_says_when_the_brain_must_reload(self):
        out, _ = self._reindex("loading nomic-embed-text would unload gemma4")
        self.assertIn("in the background", out)
        self.assertIn("my next answer may take a few seconds longer", out)
        out, _ = self._reindex("")
        self.assertEqual(out, "Reindexing your files in the background, sir.")

    def test_status_says_when_indexing_waits(self):
        from tests.skills.test_personal_rag import _fake_rag
        rag_ = _fake_rag()
        rag_.collection_size.return_value = 10
        rag_.status.return_value = {
            "running": True, "watchdog_active": True, "last_full_scan_ts": 0,
            "errors": 0, "deferred": "loading nomic-embed-text would unload "
            "gemma4 (Ollama holds 1 model)", "retry_at": 1_760_000_000.0}
        with mock.patch.object(self.mod, "_rag", return_value=rag_):
            out = self.actions["rag_status"]("")
        self.assertIn("waiting (next try ", out)
        self.assertIn("won't unload my local model", out)
        self.assertIn('Say "reindex my files"', out)
        rag_.status.return_value = {"running": True, "watchdog_active": True,
                                    "last_full_scan_ts": 0, "errors": 0}
        with mock.patch.object(self.mod, "_rag", return_value=rag_):
            self.assertNotIn("waiting", self.actions["rag_status"](""))


if __name__ == "__main__":
    unittest.main()
