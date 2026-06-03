"""Logic tests for skills/ambient_multimodal_extract.py.

This skill is a background daemon that folds mic + system-audio + screen
observations into long-term memory. Tests cover the deterministic core:
  • _format_window rendering of the merged multimodal stream (mic /
    system_audio / screen), with sensitive screen lines skipped,
  • _llm_extract JSON parsing from a noisy reply (and the empty-dict failure
    modes),
  • _merge_into_memory delegating to bobert_companion.merge_memory and
    counting only what was actually added,
  • _run_once: windowing/cutoff, the no-data short-circuit, and the
    full mic→llm→merge path with counters updated,
  • the start/stop/status/now actions (threads neutered so no real daemon).

The extract-log path is redirected to a temp file; bobert_companion is a
controllable stub injected via _get_bobert. No LLM/network/real writes.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _bc_with(llm_return="", merge_return=([], [])):
    bc = mock.MagicMock()
    bc._llm_quick.return_value = llm_return
    bc.merge_memory.return_value = merge_return
    # _get_config reads attributes off bobert; give sane numeric defaults.
    bc.AMBIENT_EXTRACT_BATCH = 50
    bc.AMBIENT_EXTRACT_INTERVAL_S = 300.0
    bc.AMBIENT_EXTRACT_ENABLED = False
    return bc


class AmbientExtractFormatTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_multimodal_extract")

    # ── _format_window ───────────────────────────────────────────────────
    def test_format_window_renders_all_sources(self):
        entries = [
            {"ts": 100.0, "source": "mic", "text": "let's ship it", "window": "Slack"},
            {"ts": 101.0, "source": "system_audio", "text": "podcast talk",
             "window": "Chrome"},
            {"ts": 102.0, "source": "screen", "summary": "a kanban board",
             "window": "Jira", "entities": ["Acme", "Q3"]},
        ]
        out = self.mod._format_window(entries)
        self.assertIn("mic", out)
        self.assertIn("let's ship it", out)
        self.assertIn("system_audio", out)
        self.assertIn("podcast talk", out)
        self.assertIn("screen", out)
        self.assertIn("a kanban board", out)
        self.assertIn("Acme", out)

    def test_format_window_skips_sensitive_screen_lines(self):
        entries = [
            {"ts": 1.0, "source": "screen", "summary": "1Password vault",
             "window": "1Password", "sensitive": True},
            {"ts": 2.0, "source": "mic", "text": "hello world", "window": ""},
        ]
        out = self.mod._format_window(entries)
        self.assertNotIn("1Password vault", out)
        self.assertIn("hello world", out)

    # ── _llm_extract ─────────────────────────────────────────────────────
    def test_llm_extract_parses_json(self):
        reply = ('Here you go: {"new_facts": ["likes tea"], "new_projects": [],'
                 ' "mentions": [{"text": "Acme", "source": "screen", '
                 '"attribution": "visible"}]} done')
        bc = _bc_with(llm_return=reply)
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.mod._llm_extract("some window text")
        self.assertEqual(out["new_facts"], ["likes tea"])
        self.assertEqual(out["mentions"][0]["text"], "Acme")

    def test_llm_extract_empty_on_no_json(self):
        bc = _bc_with(llm_return="no json here at all")
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._llm_extract("x"), {})

    def test_llm_extract_empty_when_no_quick_helper(self):
        bc = mock.MagicMock()
        bc._llm_quick = "not callable"
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._llm_extract("x"), {})

    # ── _merge_into_memory ───────────────────────────────────────────────
    def test_merge_counts_only_added(self):
        bc = _bc_with(merge_return=(["fact1"], ["proj1", "proj2"]))
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            facts, projs = self.mod._merge_into_memory(
                {"new_facts": ["fact1"], "new_projects": ["proj1", "proj2"]})
        self.assertEqual((facts, projs), (1, 2))
        bc.merge_memory.assert_called_once()

    def test_merge_noop_when_empty(self):
        bc = _bc_with()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._merge_into_memory({"new_facts": [], "new_projects": []}),
                             (0, 0))
        bc.merge_memory.assert_not_called()

    def test_merge_handles_merge_failure(self):
        bc = _bc_with()
        bc.merge_memory.side_effect = RuntimeError("disk full")
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._merge_into_memory({"new_facts": ["x"]}), (0, 0))


class AmbientExtractRunTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_multimodal_extract")
        # Reset counters + redirect the extract log to a temp file.
        self.mod._runs_total = 0
        self.mod._facts_added_total = 0
        self.mod._projects_added_total = 0
        self.tmp = tempfile.mkdtemp(prefix="ambextract_test_")
        self.addCleanup(self._cleanup)
        self.mod._DATA_DIR = self.tmp
        self.mod._EXTRACT_JSONL = os.path.join(self.tmp, "extracts.jsonl")

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_run_once_no_data_short_circuits(self):
        bc = _bc_with()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_tail_jsonl", return_value=[]):
            summary = self.mod._run_once()
        self.assertEqual(summary["facts_added"], 0)
        self.assertEqual(self.mod._runs_total, 1)
        # Even an empty pass appends a log line.
        self.assertTrue(os.path.exists(self.mod._EXTRACT_JSONL))

    def test_run_once_filters_old_entries_by_cutoff(self):
        bc = _bc_with()
        now = time.time()
        # One fresh mic entry, one ancient one (beyond interval*2.5 cutoff).
        audio = [{"ts": now, "source": "mic", "text": "fresh talk", "window": ""},
                 {"ts": now - 100000, "source": "mic", "text": "ancient", "window": ""}]
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_tail_jsonl", side_effect=[audio, []]), \
             mock.patch.object(self.mod, "_llm_extract", return_value={}) as llm:
            summary = self.mod._run_once()
        # Only the fresh mic entry survived the window.
        self.assertEqual(summary["mic_entries"], 1)
        # window text was non-empty so the LLM was consulted.
        llm.assert_called_once()

    def test_run_once_merges_and_updates_counters(self):
        bc = _bc_with()
        now = time.time()
        audio = [{"ts": now, "source": "mic", "text": "I started project Atlas",
                  "window": "Notes"}]
        extracted = {"new_facts": ["uses Notes"], "new_projects": ["Atlas"],
                     "mentions": [{"text": "Atlas", "source": "mic",
                                   "attribution": "speaker"}]}
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_tail_jsonl", side_effect=[audio, []]), \
             mock.patch.object(self.mod, "_llm_extract", return_value=extracted), \
             mock.patch.object(self.mod, "_merge_into_memory", return_value=(1, 1)):
            summary = self.mod._run_once()
        self.assertEqual(summary["facts_added"], 1)
        self.assertEqual(summary["projects_added"], 1)
        self.assertEqual(self.mod._facts_added_total, 1)
        self.assertEqual(summary["mentions"][0]["text"], "Atlas")

    # ── actions ──────────────────────────────────────────────────────────
    def test_status_off_by_default(self):
        out = self.actions["ambient_extract_status"]("")
        self.assertIn("OFF", out)
        self.assertIn("0 passes", out)

    def test_stop_when_not_running(self):
        self.assertIn("not running", self.actions["ambient_extract_stop"](""))

    def test_now_runs_one_pass(self):
        bc = _bc_with()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_tail_jsonl", return_value=[]):
            out = self.actions["ambient_extract_now"]("")
        self.assertIn("Ambient extraction complete", out)
        self.assertIn("0 new facts", out)

    def test_start_reports_engaged(self):
        # Thread.start is neutered → no real daemon; the thread object is
        # still constructed so is_alive() is False and the action reports
        # the engaged message (it doesn't depend on the loop actually running).
        bc = _bc_with()
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            out = self.actions["ambient_extract_start"]("")
        self.assertIn("Ambient extractor engaged", out)

    def test_run_once_empty_window_text_short_circuits(self):
        # A non-empty entry list whose entries all render to nothing (e.g. a
        # sensitive screen line that _format_window drops) → window_text is
        # blank → the second short-circuit returns the empty summary.
        bc = _bc_with()
        now = time.time()
        screen = [{"ts": now, "source": "screen", "summary": "vault",
                   "window": "1Password", "sensitive": True}]
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_tail_jsonl", side_effect=[[], screen]), \
             mock.patch.object(self.mod, "_llm_extract") as llm:
            summary = self.mod._run_once()
        self.assertEqual(summary["facts_added"], 0)
        llm.assert_not_called()        # never reached the LLM
        self.assertEqual(self.mod._runs_total, 1)


class AmbientExtractHelperTests(unittest.TestCase):
    """Pure helpers + the small config/path utilities."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_multimodal_extract")

    def test_ensure_project_on_path_inserts_once(self):
        import sys
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path if p != self.mod._PROJECT_DIR]
            self.mod._ensure_project_on_path()
            self.assertIn(self.mod._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved

    def test_get_config_default_when_no_bobert(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.assertEqual(self.mod._get_config("ANYTHING", 7), 7)

    def test_get_config_reads_attr_from_bobert(self):
        bc = mock.MagicMock()
        bc.SOME_KNOB = 123
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._get_config("SOME_KNOB", 0), 123)

    def test_get_bobert_prefers_bobert_companion(self):
        import sys
        sentinel = object()
        with mock.patch.dict(sys.modules, {"bobert_companion": sentinel}):
            self.assertIs(self.mod._get_bobert(), sentinel)

    # ── _tail_jsonl ──────────────────────────────────────────────────────
    def test_tail_jsonl_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._tail_jsonl("nope.jsonl", 10), [])

    def test_tail_jsonl_nonpositive_n(self):
        self.assertEqual(self.mod._tail_jsonl("x.jsonl", 0), [])

    def test_tail_jsonl_reads_last_n_and_skips_junk(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "log.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"n": 1}) + "\n")
                f.write("\n")                     # blank → skipped
                f.write("{ not json\n")           # corrupt → skipped
                f.write(json.dumps({"n": 2}) + "\n")
            out = self.mod._tail_jsonl(p, 10)
        self.assertEqual([e["n"] for e in out], [1, 2])

    def test_tail_jsonl_caps_to_n(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "log.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for i in range(20):
                    f.write(json.dumps({"n": i}) + "\n")
            out = self.mod._tail_jsonl(p, 3)
        self.assertEqual([e["n"] for e in out], [17, 18, 19])

    def test_tail_jsonl_read_error_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._tail_jsonl("x.jsonl", 5), [])

    def test_format_window_system_audio_only(self):
        # Covers the system_audio rendering branch in isolation.
        out = self.mod._format_window(
            [{"ts": 5.0, "source": "system_audio", "text": "a call",
              "window": "Zoom"}])
        self.assertIn("system_audio (Zoom)", out)
        self.assertIn("a call", out)

    def test_format_window_infers_mic_when_source_missing(self):
        # No "source" key, has "text" and no "summary" → inferred as mic.
        out = self.mod._format_window([{"ts": 0, "text": "hi", "window": "X"}])
        self.assertIn("mic", out)
        self.assertIn("hi", out)


class AmbientExtractLlmAndMergeEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_multimodal_extract")

    def test_llm_extract_no_bobert(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.assertEqual(self.mod._llm_extract("x"), {})

    def test_llm_extract_call_raises_returns_empty(self):
        bc = mock.MagicMock()
        bc._llm_quick.side_effect = RuntimeError("api down")
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._llm_extract("x"), {})

    def test_llm_extract_empty_raw_returns_empty(self):
        bc = _bc_with(llm_return="")
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._llm_extract("x"), {})

    def test_llm_extract_malformed_json_in_braces_returns_empty(self):
        bc = _bc_with(llm_return="prefix {bad: json, no quotes} suffix")
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._llm_extract("x"), {})

    def test_merge_no_bobert(self):
        with mock.patch.object(self.mod, "_get_bobert", return_value=None):
            self.assertEqual(self.mod._merge_into_memory({"new_facts": ["x"]}),
                             (0, 0))

    def test_merge_when_merge_not_callable(self):
        bc = mock.MagicMock()
        bc.merge_memory = "not callable"
        with mock.patch.object(self.mod, "_get_bobert", return_value=bc):
            self.assertEqual(self.mod._merge_into_memory({"new_facts": ["x"]}),
                             (0, 0))


class AmbientExtractLogRotateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_multimodal_extract")
        self.tmp = tempfile.mkdtemp(prefix="ambext_log_")
        self.addCleanup(self._cleanup)
        self.mod._DATA_DIR = self.tmp
        self.mod._EXTRACT_JSONL = os.path.join(self.tmp, "extracts.jsonl")

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_append_extract_log_writes_line(self):
        self.mod._append_extract_log({"ts": 1.0, "facts_added": 2})
        with open(self.mod._EXTRACT_JSONL, encoding="utf-8") as f:
            line = json.loads(f.readline())
        self.assertEqual(line["facts_added"], 2)

    def test_append_extract_log_write_failure_swallowed(self):
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            self.mod._append_extract_log({"ts": 1.0})   # no raise

    def test_append_extract_log_rotates_when_oversized(self):
        # Pin a tiny hard cap so the rotation branch (keep last cap lines) runs.
        self.mod._EXTRACT_HARD_CAP = 4
        with open(self.mod._EXTRACT_JSONL, "w", encoding="utf-8") as f:
            for i in range(20):                 # 20 >= 4*1.5 → triggers rotate
                f.write(json.dumps({"n": i}) + "\n")
        self.mod._append_extract_log({"n": 999})
        with open(self.mod._EXTRACT_JSONL, encoding="utf-8") as f:
            kept = [json.loads(x) for x in f if x.strip()]
        # Rotation keeps the last HARD_CAP lines (the append happens first).
        self.assertEqual(len(kept), 4)
        self.assertEqual(kept[-1]["n"], 999)

    def test_append_extract_log_rotate_under_threshold_noop(self):
        self.mod._EXTRACT_HARD_CAP = 2000
        self.mod._append_extract_log({"n": 1})
        with open(self.mod._EXTRACT_JSONL, encoding="utf-8") as f:
            kept = [x for x in f if x.strip()]
        self.assertEqual(len(kept), 1)

    def test_append_extract_log_rotate_failure_swallowed(self):
        # Oversized file pushes execution into the rotate block; mkstemp raising
        # there is caught and printed, not propagated (the append already
        # succeeded).
        self.mod._EXTRACT_HARD_CAP = 4
        with open(self.mod._EXTRACT_JSONL, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({"n": i}) + "\n")
        with mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("no temp space")):
            self.mod._append_extract_log({"n": 999})   # no raise


class AmbientExtractActionEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_multimodal_extract")
        self.addCleanup(self._reset)

    def _reset(self):
        # Ensure no live thread leaks between tests.
        self.mod._thread = None
        self.mod._started_at = None

    def test_start_when_already_running_is_idempotent(self):
        fake = mock.MagicMock()
        fake.is_alive.return_value = True
        self.mod._thread = fake
        out = self.actions["ambient_extract_start"]("")
        self.assertIn("already running", out)

    def test_stop_running_thread_reports_summary(self):
        # A fake thread that reports alive, then dead after join → clean stop.
        fake = mock.MagicMock()
        alive = {"v": True}
        fake.is_alive.side_effect = lambda: alive["v"]

        def _join(timeout=None):
            alive["v"] = False
        fake.join.side_effect = _join
        self.mod._thread = fake
        self.mod._started_at = time.time() - 10
        self.mod._runs_total = 3
        self.mod._facts_added_total = 5
        self.mod._projects_added_total = 1
        out = self.actions["ambient_extract_stop"]("")
        self.assertIn("disengaged", out)
        self.assertIn("3 passes", out)
        self.assertIn("5 facts", out)

    def test_stop_thread_not_dying_reports_unclean(self):
        fake = mock.MagicMock()
        fake.is_alive.return_value = True          # stays alive after join
        fake.join.return_value = None
        self.mod._thread = fake
        self.mod._started_at = time.time()
        out = self.actions["ambient_extract_stop"]("")
        self.assertIn("did not stop cleanly", out)

    def test_now_handles_run_once_exception(self):
        with mock.patch.object(self.mod, "_run_once",
                               side_effect=RuntimeError("boom")):
            out = self.actions["ambient_extract_now"]("")
        self.assertIn("failed", out.lower())


class AmbientExtractLoopTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_multimodal_extract")

    def test_loop_runs_once_then_exits_on_stop(self):
        # wait() returns True on the first check → loop breaks after one pass.
        self.mod._stop_evt = self.mod.threading.Event()
        with mock.patch.object(self.mod, "_get_config", return_value=300.0), \
             mock.patch.object(self.mod, "_run_once") as run, \
             mock.patch.object(self.mod._stop_evt, "wait", return_value=True):
            self.mod._loop()
        run.assert_called_once()

    def test_loop_swallows_run_once_exception_then_exits(self):
        self.mod._stop_evt = self.mod.threading.Event()
        with mock.patch.object(self.mod, "_get_config", return_value=300.0), \
             mock.patch.object(self.mod, "_run_once",
                               side_effect=RuntimeError("extract boom")), \
             mock.patch.object(self.mod._stop_evt, "wait", return_value=True):
            self.mod._loop()
        self.assertIn("extraction failed", self.mod._last_error)

    def test_loop_skips_body_when_already_stopped(self):
        # _stop_evt already set → while-condition false → body never runs.
        self.mod._stop_evt = self.mod.threading.Event()
        self.mod._stop_evt.set()
        with mock.patch.object(self.mod, "_get_config", return_value=300.0), \
             mock.patch.object(self.mod, "_run_once") as run:
            self.mod._loop()
        run.assert_not_called()


class AmbientExtractRegisterTests(unittest.TestCase):
    def test_register_adds_four_actions_no_autostart(self):
        mod, _ = load_skill_isolated("ambient_multimodal_extract", register=False)
        actions = {}
        with mock.patch.object(mod, "_get_config", return_value=False):
            mod.register(actions)
        for name in ("ambient_extract_start", "ambient_extract_stop",
                     "ambient_extract_status", "ambient_extract_now"):
            self.assertIn(name, actions)

    def test_register_autostart_spawns_background_starter(self):
        mod, _ = load_skill_isolated("ambient_multimodal_extract", register=False)
        actions = {}
        import threading as _thr
        captured = {}

        def _fake_start(self):
            # Run the target inline (it's the _bg closure) with sleep + start
            # patched so nothing real happens.
            captured["target"] = self._target
        with mock.patch.object(mod, "_get_config", return_value=True), \
             mock.patch.object(_thr.Thread, "start", _fake_start):
            mod.register(actions)
        # The autostart thread was constructed around the _bg closure.
        self.assertIn("target", captured)
        # Drive the closure directly: it sleeps then calls ambient_extract_start.
        with mock.patch.object(mod.time, "sleep", return_value=None), \
             mock.patch.object(mod, "ambient_extract_start") as start:
            captured["target"]()
        start.assert_called_once()

    def test_register_autostart_bg_swallows_exception(self):
        mod, _ = load_skill_isolated("ambient_multimodal_extract", register=False)
        actions = {}
        import threading as _thr
        captured = {}
        with mock.patch.object(_thr.Thread, "start",
                               lambda self: captured.__setitem__("t", self._target)), \
             mock.patch.object(mod, "_get_config", return_value=True):
            mod.register(actions)
        # _bg's body raising (sleep blows up) is caught and printed, not raised.
        with mock.patch.object(mod.time, "sleep",
                               side_effect=RuntimeError("sleep boom")):
            captured["t"]()   # must not raise


if __name__ == "__main__":
    unittest.main()
