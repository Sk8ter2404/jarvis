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


if __name__ == "__main__":
    unittest.main()
