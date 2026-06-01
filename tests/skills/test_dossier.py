"""Logic tests for skills/dossier.py.

Targets the aggregation + summary logic that makes the "pull up the file on X"
moment: topic normalization, the memory/task/log gatherers (driven against
temp files), sentence shortening, the deterministic two-sentence spoken
summary, and the compile_dossier() / _act_dossier() orchestration.

External I/O is fully mocked: the DuckDuckGo fetch (_gather_web) is patched so
no network call happens, and the tkinter card renderer subprocess is stubbed so
_act_dossier never spawns a process.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class DossierHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")

    # ── _normalize_topic ─────────────────────────────────────────────────
    def test_normalize_strips_lead_ins(self):
        n = self.mod._normalize_topic
        self.assertEqual(n("the file on Sam"), "Sam")
        self.assertEqual(n("dossier on Bambu"), "Bambu")
        self.assertEqual(n("about Apple Music"), "Apple Music")
        # Only the leading "the " is stripped here (one lead-in per call).
        self.assertEqual(n("the printer"), "printer")

    def test_normalize_strips_punctuation(self):
        self.assertEqual(self.mod._normalize_topic('  "Apple Music".  '),
                         "Apple Music")

    def test_normalize_empty(self):
        self.assertEqual(self.mod._normalize_topic("   "), "")

    # ── _match / _shorten_sentence ───────────────────────────────────────
    def test_match_case_insensitive(self):
        self.assertTrue(self.mod._match("sam", "Talked to SAM earlier"))
        self.assertFalse(self.mod._match("sam", "nothing relevant"))
        self.assertFalse(self.mod._match("", "anything"))

    def test_shorten_sentence_under_limit_untouched(self):
        self.assertEqual(self.mod._shorten_sentence("short text", 100),
                         "short text")

    def test_shorten_sentence_truncates_on_word_boundary(self):
        out = self.mod._shorten_sentence("alpha beta gamma delta", 12)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 13)
        self.assertNotIn("delta", out)

    # ── _compact_task_line ───────────────────────────────────────────────
    def test_compact_task_line_strips_checkbox_and_done(self):
        out = self.mod._compact_task_line("- [x] ship the thing ✓ DONE — 2026")
        self.assertNotIn("[x]", out)
        self.assertNotIn("DONE", out)
        self.assertIn("ship the thing", out)


class DossierGatherTests(unittest.TestCase):
    """Gatherers driven against temp memory/todo files."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")
        self.tmpdir = tempfile.mkdtemp()
        self.mem = os.path.join(self.tmpdir, "bobert_memory.json")
        self.todo = os.path.join(self.tmpdir, "jarvis_todo.md")
        self._patches = [
            mock.patch.object(self.mod, "_MEMORY_FILE", self.mem),
            mock.patch.object(self.mod, "_TODO_FILE", self.todo),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self.mem, self.todo):
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def test_gather_memory_matches_facts_and_projects(self):
        with open(self.mem, "w", encoding="utf-8") as f:
            json.dump({
                "facts": ["Sam is the manager", "unrelated note"],
                "projects": ["Sam onboarding flow"],
            }, f)
        out = self.mod._gather_memory("sam")
        self.assertEqual(len(out), 2)
        self.assertTrue(any("manager" in x for x in out))
        self.assertTrue(any(x.startswith("(project)") for x in out))

    def test_gather_memory_missing_file(self):
        self.assertEqual(self.mod._gather_memory("sam"), [])

    def test_gather_tasks_open_before_done(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [x] email Sam the report\n")
            f.write("- [ ] call Sam back\n")
            f.write("- [ ] unrelated task\n")
        out = self.mod._gather_tasks("sam")
        self.assertEqual(len(out), 2)
        # Open task first.
        self.assertIn("call Sam back", out[0])
        self.assertIn("email Sam", out[1])

    def test_gather_tasks_missing_file(self):
        self.assertEqual(self.mod._gather_tasks("sam"), [])


class DossierSummaryTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")

    def test_summary_no_data_is_apologetic(self):
        out = self.mod._build_spoken_summary("Sam", [], [], [], "")
        self.assertIn("very little on Sam", out)
        self.assertIn("nothing in memory", out)

    def test_summary_counts_pluralise(self):
        out = self.mod._build_spoken_summary(
            "Sam", ["f1", "f2"], ["t1"], ["l1", "l2", "l3"], "")
        self.assertIn("2 memory facts", out)
        self.assertIn("1 task entry", out)
        self.assertIn("3 log mentions", out)
        self.assertIn("Sam", out)

    def test_summary_prefers_web_for_second_sentence(self):
        out = self.mod._build_spoken_summary(
            "Bambu", ["a fact"], [], [], "Bambu Lab makes 3D printers")
        self.assertIn("Bambu Lab makes 3D printers", out)

    def test_summary_falls_back_to_fact_when_no_web(self):
        out = self.mod._build_spoken_summary("X", ["the latest note"], [], [], "")
        self.assertIn("Most recent note", out)
        self.assertIn("the latest note", out)


class DossierCompileTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")

    def test_compile_empty_topic_asks_for_subject(self):
        data = self.mod.compile_dossier("   ")
        self.assertEqual(data["topic"], "")
        self.assertIn("need a subject", data["summary"].lower())

    def test_compile_aggregates_all_sources(self):
        with mock.patch.object(self.mod, "_gather_memory",
                               return_value=["fact about sam"]), \
             mock.patch.object(self.mod, "_gather_tasks",
                               return_value=["call sam"]), \
             mock.patch.object(self.mod, "_gather_logs",
                               return_value=["[10:00] sam pinged"]), \
             mock.patch.object(self.mod, "_gather_web",
                               return_value="Sam is a name"):
            data = self.mod.compile_dossier("the file on Sam")
        self.assertEqual(data["topic"], "Sam")
        self.assertEqual(data["facts"], ["fact about sam"])
        self.assertEqual(data["tasks"], ["call sam"])
        self.assertIn("sam pinged", data["logs"][0])
        self.assertIn("Sam", data["summary"])

    def test_act_dossier_no_topic_returns_summary(self):
        # Empty subject short-circuits before any card render.
        out = self.actions["dossier"]("")
        self.assertIn("need a subject", out.lower())

    def test_act_dossier_builds_report_without_rendering(self):
        # Stub the card pipeline so no state file is written and no subprocess
        # spawns; assert the structured report the follow-up LLM phrases aloud.
        with mock.patch.object(self.mod, "_gather_memory",
                               return_value=["Sam manages the team"]), \
             mock.patch.object(self.mod, "_gather_tasks", return_value=[]), \
             mock.patch.object(self.mod, "_gather_logs", return_value=[]), \
             mock.patch.object(self.mod, "_gather_web", return_value=""), \
             mock.patch.object(self.mod, "_write_state"), \
             mock.patch.object(self.mod, "_ensure_renderer_running"), \
             mock.patch.object(self.mod, "_top_monitor_geometry",
                               return_value=(0, 0, 1920, 1080)):
            out = self.actions["pull_up_file"]("Sam")
        self.assertIn("DOSSIER on 'Sam'", out)
        self.assertIn("Sam manages the team", out)
        self.assertIn("tasks: (none)", out)
        self.assertIn("HUD card displayed", out)


if __name__ == "__main__":
    unittest.main()
