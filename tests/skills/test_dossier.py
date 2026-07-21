"""Logic tests for skills/dossier.py.

Targets the aggregation + summary logic that makes the "pull up the file on X"
moment: topic normalization, the memory/task/log gatherers (driven against
temp files), sentence shortening, the deterministic two-sentence spoken
summary, and the compile_dossier() / _act_dossier() orchestration. Also covers
the card-state I/O, the PID/renderer-liveness helpers, the subprocess spawn
guard, the tkinter renderer (tkinter fully faked), register()'s host-module
patching, and the __main__ CLI dispatch.

External I/O is fully mocked: the DuckDuckGo fetch (_gather_web) is patched so
no network call happens, urllib is stubbed where the real fetch is exercised,
the tkinter card renderer subprocess is stubbed so _act_dossier never spawns a
process, and tkinter itself is faked so _renderer_main draws no real window. No
network, hardware, threads, or sleeps. All fixtures use generic names only.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── shared fakes / helpers ───────────────────────────────────────────────

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring the prior
    state (including absence) on exit so tests stay isolated. Mirrors the
    approved pattern in test_self_diagnostic.py. Pass ``name=None`` to force a
    module to look un-importable inside the block."""
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


@contextlib.contextmanager
def block_import(*names):
    """Force ``import <name>`` to raise ImportError inside the block, so a
    deferred-import fallback is exercised even when the real dep is installed on
    the dev/CI box.

    Implemented by inserting a ``None`` sentinel into ``sys.modules`` for each
    name: CPython's import machinery treats ``sys.modules[name] is None`` as
    "known-absent" and raises ImportError without consulting finders. This is
    deliberately chosen over patching ``builtins.__import__`` because a global
    ``__import__`` patch disrupts coverage.py's tracer for the rest of the
    process. Restores the prior state (value or absence) on exit. Top-level
    names only — dossier's lazy imports (psutil) are top-level."""
    saved: dict[str, object] = {}
    for name in names:
        saved[name] = sys.modules.get(name, _SENTINEL)
        sys.modules[name] = None        # sentinel -> import raises ImportError
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _fake_response(payload):
    """Context-manager stand-in for urllib.request.urlopen()."""
    body = (json.dumps(payload).encode("utf-8")
            if not isinstance(payload, (bytes, bytearray)) else payload)
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


class _FakeWidget:
    """Stands in for any tkinter widget (Tk, Frame, Label). Every method is a
    no-op that returns another fake so chained calls (pack().config()) work."""
    def __init__(self, *a, **k):
        self.after_calls = []

    def __getattr__(self, _name):
        return self._noop

    def _noop(self, *a, **k):
        return self


class _FakeTk(_FakeWidget):
    """Root window fake that records ``after`` callbacks so the scheduled
    closures (_slide / _tick) can be invoked deterministically by the test."""
    def after(self, _delay, func=None, *a, **k):
        if func is not None:
            self.after_calls.append(func)
        return "after-id"

    def mainloop(self):
        return None

    def destroy(self):
        self.destroyed = True
        return None


def make_fake_tkinter(root):
    """A fake ``tkinter`` module whose Tk() returns the supplied root and whose
    Frame/Label return inert fakes. Lets _renderer_main run windowless."""
    tk = types.ModuleType("tkinter")
    tk.Tk = lambda *a, **k: root
    tk.Frame = lambda *a, **k: _FakeWidget()
    tk.Label = lambda *a, **k: _FakeWidget()
    return tk


# ──────────────────────────────────────────────────────────────────────────
# EXISTING TESTS (kept verbatim — wave-1 baseline)
# ──────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────
# NEW: gatherer edge / corrupt-file paths
# ──────────────────────────────────────────────────────────────────────────
class DossierGatherEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")
        self.tmpdir = tempfile.mkdtemp(prefix="dossier_edge_")
        self.addCleanup(self._cleanup)
        self.mem = os.path.join(self.tmpdir, "bobert_memory.json")
        self.todo = os.path.join(self.tmpdir, "jarvis_todo.md")
        mock.patch.object(self.mod, "_MEMORY_FILE", self.mem).start()
        mock.patch.object(self.mod, "_TODO_FILE", self.todo).start()
        self.addCleanup(mock.patch.stopall)

    def _cleanup(self):
        for fn in os.listdir(self.tmpdir):
            try:
                os.unlink(os.path.join(self.tmpdir, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def test_gather_memory_corrupt_json_returns_empty(self):
        with open(self.mem, "w", encoding="utf-8") as f:
            f.write("{ not valid json :::")
        self.assertEqual(self.mod._gather_memory("sam"), [])

    def test_gather_memory_non_string_items_skipped(self):
        # Non-str entries (ints/dicts) must be ignored, not crash.
        with open(self.mem, "w", encoding="utf-8") as f:
            json.dump({"facts": [123, {"x": 1}, "sam note"],
                       "projects": [None, "sam project"]}, f)
        out = self.mod._gather_memory("sam")
        self.assertIn("sam note", out)
        self.assertTrue(any(x.startswith("(project)") for x in out))
        self.assertEqual(len(out), 2)

    def test_gather_memory_open_raises_returns_empty(self):
        with open(self.mem, "w", encoding="utf-8") as f:
            json.dump({"facts": ["sam"]}, f)
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._gather_memory("sam"), [])

    def test_gather_tasks_open_raises_returns_empty(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] call sam\n")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._gather_tasks("sam"), [])


# ──────────────────────────────────────────────────────────────────────────
# NEW: _gather_logs (temp logs dir)
# ──────────────────────────────────────────────────────────────────────────
class DossierLogGatherTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")
        self.logs = tempfile.mkdtemp(prefix="dossier_logs_")
        self.addCleanup(self._cleanup)
        mock.patch.object(self.mod, "_LOGS_DIR", self.logs).start()
        self.addCleanup(mock.patch.stopall)

    def _cleanup(self):
        for fn in os.listdir(self.logs):
            try:
                os.unlink(os.path.join(self.logs, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.logs)
        except OSError:
            pass

    def _write(self, name, text):
        with open(os.path.join(self.logs, name), "w", encoding="utf-8") as f:
            f.write(text)

    def test_missing_logs_dir(self):
        with mock.patch.object(self.mod, "_LOGS_DIR",
                               os.path.join(self.logs, "nope")):
            self.assertEqual(self.mod._gather_logs("sam"), [])

    def test_listdir_raises_returns_empty(self):
        with mock.patch("os.listdir", side_effect=OSError("denied")):
            self.assertEqual(self.mod._gather_logs("sam"), [])

    def test_matches_and_trims_line(self):
        self._write("session_1.log", "[10:00] talked to sam\nunrelated\n")
        out = self.mod._gather_logs("sam")
        self.assertEqual(len(out), 1)
        self.assertIn("sam", out[0])

    def test_long_line_truncated_with_ellipsis(self):
        long = "x sam " + ("y" * 300)
        self._write("session_1.log", long + "\n")
        out = self.mod._gather_logs("sam")
        self.assertTrue(out[0].endswith("…"))
        self.assertLessEqual(len(out[0]), self.mod.LOG_LINE_MAX_LEN)

    def test_newest_log_scanned_first(self):
        # Filenames sort reverse=True, so session_2 (newer) is scanned before
        # session_1 — its mention should appear first in the output.
        self._write("session_1.log", "[09:00] sam older\n")
        self._write("session_2.log", "[11:00] sam newer\n")
        out = self.mod._gather_logs("sam")
        self.assertIn("newer", out[0])

    def test_caps_at_max_log_lines(self):
        # Far more matches than MAX_LOG_LINES_SHOWN -> result is capped.
        body = "".join(f"[{i:02d}:00] sam ping {i}\n" for i in range(30))
        self._write("session_1.log", body)
        out = self.mod._gather_logs("sam")
        self.assertEqual(len(out), self.mod.MAX_LOG_LINES_SHOWN)

    def test_tail_seek_drops_partial_first_line(self):
        # Force a tiny tail window so the file is bigger than LOG_TAIL_BYTES and
        # the reader seeks into the middle, dropping the partial first line.
        with mock.patch.object(self.mod, "LOG_TAIL_BYTES", 40):
            # First (long) line is the partial that gets dropped; the matching
            # 'sam' mention is on a later, fully-read line.
            self._write("session_1.log",
                        ("HEADER " + "z" * 80 + "\n") + "[12:00] sam tail hit\n")
            out = self.mod._gather_logs("sam")
        self.assertEqual(len(out), 1)
        self.assertIn("tail hit", out[0])

    def test_per_file_read_error_skipped(self):
        self._write("session_1.log", "[10:00] sam here\n")

        real_open = open

        def _boom(path, *a, **k):
            if str(path).endswith("session_1.log"):
                raise OSError("read fail")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", side_effect=_boom):
            # The only log errors on read -> gracefully skipped -> empty.
            self.assertEqual(self.mod._gather_logs("sam"), [])

    def test_only_dot_log_files_considered(self):
        self._write("notes.txt", "[10:00] sam in a txt file\n")
        self.assertEqual(self.mod._gather_logs("sam"), [])


# ──────────────────────────────────────────────────────────────────────────
# NEW: _gather_web (urllib mocked)
# ──────────────────────────────────────────────────────────────────────────
class DossierWebTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")

    def test_empty_topic_returns_empty_no_fetch(self):
        with mock.patch.object(self.mod.urllib.request, "urlopen") as uo:
            self.assertEqual(self.mod._gather_web(""), "")
        uo.assert_not_called()

    def test_abstract_text_used(self):
        payload = {"AbstractText": "Acme builds widgets.", "RelatedTopics": []}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_fake_response(payload)):
            out = self.mod._gather_web("acme")
        self.assertEqual(out, "Acme builds widgets.")

    def test_abstract_is_shortened(self):
        long_abstract = " ".join(["word"] * 200)
        payload = {"AbstractText": long_abstract, "RelatedTopics": []}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_fake_response(payload)):
            out = self.mod._gather_web("acme")
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 321)

    def test_related_topics_fallback(self):
        payload = {"AbstractText": "",
                   "RelatedTopics": [{"Text": "Acme is a fictional company."}]}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_fake_response(payload)):
            out = self.mod._gather_web("acme")
        self.assertEqual(out, "Acme is a fictional company.")

    def test_related_topics_non_dict_first_entry(self):
        # First related entry is not a dict (DDG nests groups) -> '' returned.
        payload = {"AbstractText": "", "RelatedTopics": [["not", "a", "dict"]]}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_fake_response(payload)):
            self.assertEqual(self.mod._gather_web("acme"), "")

    def test_empty_abstract_and_empty_related(self):
        payload = {"AbstractText": "  ", "RelatedTopics": []}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_fake_response(payload)):
            self.assertEqual(self.mod._gather_web("acme"), "")

    def test_related_topics_blank_text(self):
        payload = {"AbstractText": "", "RelatedTopics": [{"Text": "   "}]}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_fake_response(payload)):
            self.assertEqual(self.mod._gather_web("acme"), "")

    def test_network_error_returns_empty(self):
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=OSError("offline")):
            self.assertEqual(self.mod._gather_web("acme"), "")

    def test_malformed_json_returns_empty(self):
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_fake_response(b"<<not json>>")):
            self.assertEqual(self.mod._gather_web("acme"), "")


# ──────────────────────────────────────────────────────────────────────────
# NEW: remaining _build_spoken_summary branches
# ──────────────────────────────────────────────────────────────────────────
class DossierSummaryBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")

    def test_three_part_oxford_join(self):
        # facts + tasks + logs all present -> ", and " oxford join.
        out = self.mod._build_spoken_summary(
            "Sam", ["f1"], ["t1"], ["l1"], "")
        self.assertIn("memory fact", out)
        self.assertIn(", and ", out)

    def test_two_part_and_join(self):
        # Exactly two source kinds (facts + tasks, no logs) -> " and " join.
        out = self.mod._build_spoken_summary("Sam", ["f1"], ["t1"], [], "")
        self.assertIn("1 memory fact and 1 task entry", out)
        self.assertNotIn(", and ", out)

    def test_single_part_blob(self):
        # Exactly one source kind -> the blob is that single phrase, no join.
        out = self.mod._build_spoken_summary("Sam", ["f1", "f2"], [], [], "x")
        self.assertIn("I've got 2 memory facts.", out)

    def test_second_sentence_task_fallback(self):
        # No web, no facts, but tasks present -> "Top of the queue:".
        out = self.mod._build_spoken_summary("Sam", [], ["call sam back"], [], "")
        self.assertIn("Top of the queue", out)
        self.assertIn("call sam back", out)

    def test_second_sentence_card_fallback(self):
        # Counts exist (logs) but no web/facts/tasks -> dry card fallback.
        out = self.mod._build_spoken_summary("Sam", [], [], ["[10:00] sam"], "")
        self.assertIn("Card displayed on the top monitor", out)

    def test_web_without_trailing_period_gets_one(self):
        out = self.mod._build_spoken_summary(
            "Sam", [], [], [], "Sam is a manager")
        self.assertIn("Sam is a manager.", out)


# ──────────────────────────────────────────────────────────────────────────
# NEW: _top_monitor_geometry
# ──────────────────────────────────────────────────────────────────────────
class DossierGeometryTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")

    def test_reads_top_monitor(self):
        bc = types.ModuleType("bobert_companion")
        bc.MONITORS = {"top": (100, 200, 1920, 1080)}
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._top_monitor_geometry(),
                             (100, 200, 1920, 1080))

    def test_falls_back_to_first_monitor_value(self):
        bc = types.ModuleType("bobert_companion")
        bc.MONITORS = {"left": (5, 6, 800, 600)}   # no 'top' key
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._top_monitor_geometry(), (5, 6, 800, 600))

    def test_default_when_no_monitors_attr(self):
        bc = types.ModuleType("bobert_companion")
        # No MONITORS attribute at all.
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._top_monitor_geometry(), (0, 0, 1920, 1080))

    def test_default_when_module_absent(self):
        with inject_modules(bobert_companion=None):
            # __main__ also has no MONITORS in the test process -> default.
            self.assertEqual(self.mod._top_monitor_geometry(), (0, 0, 1920, 1080))

    def test_bad_monitor_values_fall_through_to_default(self):
        bc = types.ModuleType("bobert_companion")
        bc.MONITORS = {"top": ("a", "b", "c", "d")}   # int() raises
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._top_monitor_geometry(), (0, 0, 1920, 1080))


# ──────────────────────────────────────────────────────────────────────────
# NEW: _write_state / _load_state_safe
# ──────────────────────────────────────────────────────────────────────────
class DossierStateIoTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")
        self.tmpdir = tempfile.mkdtemp(prefix="dossier_state_")
        self.addCleanup(self._cleanup)
        self.state = os.path.join(self.tmpdir, "dossier_state.json")
        mock.patch.object(self.mod, "_STATE_FILE", self.state).start()
        self.addCleanup(mock.patch.stopall)

    def _cleanup(self):
        for fn in os.listdir(self.tmpdir):
            try:
                os.unlink(os.path.join(self.tmpdir, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def test_write_then_load_roundtrip(self):
        self.mod._write_state({"topic": "sam", "n": 1})
        self.assertTrue(os.path.exists(self.state))
        self.assertEqual(self.mod._load_state_safe(), {"topic": "sam", "n": 1})

    def test_load_missing_returns_none(self):
        self.assertIsNone(self.mod._load_state_safe())

    def test_load_corrupt_returns_none(self):
        with open(self.state, "w", encoding="utf-8") as f:
            f.write("{ broken json")
        self.assertIsNone(self.mod._load_state_safe())

    def test_write_failure_cleans_tmp_and_raises(self):
        # json.dump raises mid-write -> the .tmp file is unlinked and the error
        # propagates. Assert no stray .tmp remains in the dir.
        with mock.patch.object(self.mod.json, "dump",
                               side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                self.mod._write_state({"x": object()})
        leftovers = [f for f in os.listdir(self.tmpdir) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_write_failure_unlink_also_raises_still_propagates(self):
        # json.dump raises AND the tmp-cleanup os.unlink raises too -> the inner
        # except swallows the unlink error and the ORIGINAL error propagates.
        with mock.patch.object(self.mod.json, "dump",
                               side_effect=ValueError("boom")), \
             mock.patch("os.unlink", side_effect=OSError("cannot unlink")):
            with self.assertRaises(ValueError):
                self.mod._write_state({"x": 1})


# ──────────────────────────────────────────────────────────────────────────
# NEW: speak-time expiry refresh (2026-07-06 audit finding [31])
# ──────────────────────────────────────────────────────────────────────────
class DossierExpiryRefreshTests(unittest.TestCase):
    """The parent-side watcher must hold the card's countdown while TTS is
    speaking, fall back to the old fixed countdown when the host flag is
    absent, and stop cleanly on dismissal / replacement."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")
        self.tmpdir = tempfile.mkdtemp(prefix="dossier_expiry_")
        self.addCleanup(self._cleanup)
        self.state = os.path.join(self.tmpdir, "dossier_state.json")
        mock.patch.object(self.mod, "_STATE_FILE", self.state).start()
        self.addCleanup(mock.patch.stopall)

    def _cleanup(self):
        for fn in os.listdir(self.tmpdir):
            try:
                os.unlink(os.path.join(self.tmpdir, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def _seed_state(self, shown_at, expiry_ts, dismissed=False):
        self.mod._write_state({
            "topic": "sam", "shown_at": shown_at,
            "expiry_ts": expiry_ts, "dismissed": dismissed,
        })

    # ── _tts_playback_active_flag ────────────────────────────────────────
    def test_flag_true_when_host_speaking(self):
        bc = types.ModuleType("bobert_companion")
        bc._tts_playback_active = [True]
        with inject_modules(bobert_companion=bc):
            self.assertTrue(self.mod._tts_playback_active_flag())

    def test_flag_false_when_host_idle(self):
        bc = types.ModuleType("bobert_companion")
        bc._tts_playback_active = [False]
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._tts_playback_active_flag())

    def test_flag_missing_attribute_falls_back_false(self):
        # Older host without the v1.96.0 flag -> old behavior exactly.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._tts_playback_active_flag())

    def test_flag_missing_module_falls_back_false(self):
        with inject_modules(bobert_companion=None):
            self.assertFalse(self.mod._tts_playback_active_flag())

    def test_flag_wrong_shape_falls_back_false(self):
        # Not a non-empty list/tuple -> defensively False.
        bc = types.ModuleType("bobert_companion")
        bc._tts_playback_active = True
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._tts_playback_active_flag())

    # ── _refresh_expiry_if_speaking ──────────────────────────────────────
    def test_refresh_pushes_expiry_while_tts_active(self):
        shown = 1000.0
        self._seed_state(shown, expiry_ts=1010.0)
        with mock.patch.object(self.mod, "_tts_playback_active_flag",
                               return_value=True), \
             mock.patch.object(self.mod.time, "time", return_value=1009.0):
            self.assertTrue(self.mod._refresh_expiry_if_speaking(shown))
        cur = self.mod._load_state_safe()
        self.assertEqual(cur["expiry_ts"],
                         1009.0 + self.mod.CARD_DURATION_SECONDS)

    def test_no_refresh_when_idle_and_keeps_polling_until_expiry(self):
        shown = 1000.0
        self._seed_state(shown, expiry_ts=1010.0)
        with mock.patch.object(self.mod, "_tts_playback_active_flag",
                               return_value=False), \
             mock.patch.object(self.mod.time, "time", return_value=1005.0):
            self.assertTrue(self.mod._refresh_expiry_if_speaking(shown))
        # expiry_ts untouched -> normal countdown when idle.
        self.assertEqual(self.mod._load_state_safe()["expiry_ts"], 1010.0)

    def test_stops_after_expiry_when_idle(self):
        shown = 1000.0
        self._seed_state(shown, expiry_ts=1010.0)
        with mock.patch.object(self.mod, "_tts_playback_active_flag",
                               return_value=False), \
             mock.patch.object(self.mod.time, "time", return_value=1011.0):
            self.assertFalse(self.mod._refresh_expiry_if_speaking(shown))

    def test_stops_when_dismissed(self):
        shown = 1000.0
        self._seed_state(shown, expiry_ts=9e9, dismissed=True)
        with mock.patch.object(self.mod, "_tts_playback_active_flag",
                               return_value=True):
            self.assertFalse(self.mod._refresh_expiry_if_speaking(shown))

    def test_stops_when_state_missing(self):
        self.assertFalse(self.mod._refresh_expiry_if_speaking(1000.0))

    def test_stops_when_newer_card_owns_state(self):
        # A fresh dossier replaced ours -> our watcher must not touch it.
        self._seed_state(shown_at=2000.0, expiry_ts=9e9)
        with mock.patch.object(self.mod, "_tts_playback_active_flag",
                               return_value=True):
            self.assertFalse(self.mod._refresh_expiry_if_speaking(1000.0))
        self.assertEqual(self.mod._load_state_safe()["expiry_ts"], 9e9)

    def test_refresh_write_failure_swallowed_keeps_polling(self):
        shown = 1000.0
        self._seed_state(shown, expiry_ts=1010.0)
        with mock.patch.object(self.mod, "_tts_playback_active_flag",
                               return_value=True), \
             mock.patch.object(self.mod, "_write_state",
                               side_effect=OSError("disk full")):
            self.assertTrue(self.mod._refresh_expiry_if_speaking(shown))

    # ── watcher thread + wiring ──────────────────────────────────────────
    def test_watcher_thread_loops_until_step_false(self):
        calls = []

        def _step(shown_at):
            calls.append(shown_at)
            return len(calls) < 3

        with mock.patch.object(self.mod, "_refresh_expiry_if_speaking",
                               side_effect=_step), \
             mock.patch.object(self.mod, "EXPIRY_REFRESH_POLL_SECONDS", 0.0):
            t = self.mod._start_expiry_refresh_watcher(1000.0)
            t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(calls, [1000.0, 1000.0, 1000.0])
        self.assertTrue(t.daemon)

    def test_act_dossier_starts_watcher_with_shown_at(self):
        with mock.patch.object(self.mod, "_gather_memory", return_value=[]), \
             mock.patch.object(self.mod, "_gather_tasks", return_value=[]), \
             mock.patch.object(self.mod, "_gather_logs", return_value=[]), \
             mock.patch.object(self.mod, "_gather_web", return_value=""), \
             mock.patch.object(self.mod, "_write_state"), \
             mock.patch.object(self.mod, "_ensure_renderer_running"), \
             mock.patch.object(self.mod, "_top_monitor_geometry",
                               return_value=(0, 0, 1920, 1080)), \
             mock.patch.object(self.mod, "_start_expiry_refresh_watcher") as w:
            self.actions["dossier"]("Sam")
        w.assert_called_once()
        self.assertIsInstance(w.call_args[0][0], float)


# ──────────────────────────────────────────────────────────────────────────
# NEW: PID / renderer liveness + spawn guard
# ──────────────────────────────────────────────────────────────────────────
class DossierRendererProcTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")
        self.tmpdir = tempfile.mkdtemp(prefix="dossier_pid_")
        self.addCleanup(self._cleanup)
        self.pid = os.path.join(self.tmpdir, "dossier_card.pid")
        mock.patch.object(self.mod, "_PID_FILE", self.pid).start()
        self.addCleanup(mock.patch.stopall)

    def _cleanup(self):
        for fn in os.listdir(self.tmpdir):
            try:
                os.unlink(os.path.join(self.tmpdir, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    # ── _pid_alive ───────────────────────────────────────────────────────
    def test_pid_alive_zero_is_false(self):
        self.assertFalse(self.mod._pid_alive(0))
        self.assertFalse(self.mod._pid_alive(-3))

    def test_pid_alive_uses_psutil_when_available(self):
        psutil = types.ModuleType("psutil")
        psutil.pid_exists = lambda pid: pid == 4321
        with inject_modules(psutil=psutil):
            self.assertTrue(self.mod._pid_alive(4321))
            self.assertFalse(self.mod._pid_alive(9999))

    def test_pid_alive_oskill_fallback_alive(self):
        # psutil un-importable -> os.kill(pid, 0) path. Patch os.kill so it
        # doesn't raise, meaning the process is considered alive.
        with block_import("psutil"), \
             mock.patch("os.kill", return_value=None) as k:
            self.assertTrue(self.mod._pid_alive(1234))
        k.assert_called_once_with(1234, 0)

    def test_pid_alive_oskill_fallback_dead(self):
        with block_import("psutil"), \
             mock.patch("os.kill", side_effect=ProcessLookupError):
            self.assertFalse(self.mod._pid_alive(1234))

    # ── _renderer_alive ──────────────────────────────────────────────────
    def test_renderer_alive_no_pid_file(self):
        self.assertFalse(self.mod._renderer_alive())

    def test_renderer_alive_bad_pid_file(self):
        with open(self.pid, "w", encoding="utf-8") as f:
            f.write("not-an-int")
        self.assertFalse(self.mod._renderer_alive())

    def test_renderer_alive_true_when_pid_live(self):
        with open(self.pid, "w", encoding="utf-8") as f:
            f.write("777")
        with mock.patch.object(self.mod, "_pid_alive", return_value=True):
            self.assertTrue(self.mod._renderer_alive())

    def test_renderer_alive_empty_pid_file_is_zero(self):
        with open(self.pid, "w", encoding="utf-8") as f:
            f.write("")
        # Parses to 0 -> _pid_alive(0) -> False, no exception.
        self.assertFalse(self.mod._renderer_alive())

    def test_renderer_alive_open_raises_returns_false(self):
        with open(self.pid, "w", encoding="utf-8") as f:
            f.write("123")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertFalse(self.mod._renderer_alive())

    # ── _ensure_renderer_running ─────────────────────────────────────────
    def test_ensure_kills_stale_renderer_and_respawns(self):
        # Regression: a live renderer built its labels once from the OLD state,
        # so 'dossier on A' then 'dossier on B' left A's content on screen. The
        # fix terminates the stale renderer and spawns a fresh one that rebuilds
        # from the just-written state.
        with open(self.pid, "w", encoding="utf-8") as f:
            f.write("777")
        with mock.patch.object(self.mod, "_pid_alive", return_value=True), \
             mock.patch.object(self.mod.os, "kill") as kill, \
             mock.patch.object(self.mod.subprocess, "Popen") as popen:
            self.mod._ensure_renderer_running()
        kill.assert_called_once_with(777, self.mod.signal.SIGTERM)
        popen.assert_called_once()
        # Stale PID file is cleared so the new renderer's write wins.
        self.assertFalse(os.path.exists(self.pid))

    def test_ensure_kill_failure_swallowed_still_spawns(self):
        with open(self.pid, "w", encoding="utf-8") as f:
            f.write("777")
        with mock.patch.object(self.mod, "_pid_alive", return_value=True), \
             mock.patch.object(self.mod.os, "kill",
                               side_effect=OSError("gone")), \
             mock.patch.object(self.mod.subprocess, "Popen") as popen:
            # Must not raise, and the fresh renderer still spawns.
            self.mod._ensure_renderer_running()
        popen.assert_called_once()

    def test_ensure_spawns_when_not_alive(self):
        with mock.patch.object(self.mod, "_renderer_alive", return_value=False), \
             mock.patch.object(self.mod.subprocess, "Popen") as popen:
            self.mod._ensure_renderer_running()
        popen.assert_called_once()
        argv = popen.call_args[0][0]
        self.assertIn("--render", argv)
        self.assertIn("--parent-pid", argv)

    def test_ensure_spawn_failure_swallowed(self):
        with mock.patch.object(self.mod, "_renderer_alive", return_value=False), \
             mock.patch.object(self.mod.subprocess, "Popen",
                               side_effect=OSError("no exec")):
            # Must not raise.
            self.mod._ensure_renderer_running()

    def test_ensure_spawn_windows_creationflags(self):
        # Force the win32 branch so CREATE_NO_WINDOW is read.
        with mock.patch.object(self.mod, "_renderer_alive", return_value=False), \
             mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "CREATE_NO_WINDOW",
                               0x08000000, create=True), \
             mock.patch.object(self.mod.subprocess, "Popen") as popen:
            self.mod._ensure_renderer_running()
        self.assertEqual(popen.call_args.kwargs.get("creationflags"), 0x08000000)


# ──────────────────────────────────────────────────────────────────────────
# NEW: tkinter renderer (_renderer_main) — tkinter fully faked
# ──────────────────────────────────────────────────────────────────────────
class DossierRendererMainTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")
        self.tmpdir = tempfile.mkdtemp(prefix="dossier_render_")
        self.addCleanup(self._cleanup)
        self.state = os.path.join(self.tmpdir, "dossier_state.json")
        self.pid = os.path.join(self.tmpdir, "dossier_card.pid")
        mock.patch.object(self.mod, "_STATE_FILE", self.state).start()
        mock.patch.object(self.mod, "_PID_FILE", self.pid).start()
        self.addCleanup(mock.patch.stopall)

    def _cleanup(self):
        for fn in os.listdir(self.tmpdir):
            try:
                os.unlink(os.path.join(self.tmpdir, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def _write_state(self, **over):
        st = {
            "topic": "sam",
            "facts": ["sam fact one", "sam fact two"],
            "tasks": ["call sam"],
            "logs": ["[10:00] sam pinged"],
            "web": "Sam is a manager.",
            "geometry": [0, 0, 1920, 1080],
            "expiry_ts": 10_000_000_000.0,
            "dismissed": False,
        }
        st.update(over)
        with open(self.state, "w", encoding="utf-8") as f:
            json.dump(st, f)

    def test_no_state_returns_zero_early(self):
        # No state file -> render returns 0 before building any window.
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.assertEqual(self.mod._renderer_main(0), 0)

    def test_full_render_runs_animation_and_tick(self):
        self._write_state()
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            rc = self.mod._renderer_main(parent_pid=0)
            self.assertEqual(rc, 0)
            # Drive every scheduled callback (slide + tick) at least once.
            self.assertTrue(root.after_calls)
            for cb in list(root.after_calls):
                cb()
        # PID file is removed in the finally block.
        self.assertFalse(os.path.exists(self.pid))

    def test_render_with_empty_sections_uses_placeholders(self):
        # Empty facts/tasks/logs/web -> the "empty_text" branches render.
        self._write_state(facts=[], tasks=[], logs=[], web="")
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.assertEqual(self.mod._renderer_main(0), 0)
            for cb in list(root.after_calls):
                cb()

    def test_render_bad_geometry_uses_default(self):
        self._write_state(geometry=["x", "y"])   # unpack/int -> ValueError
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.assertEqual(self.mod._renderer_main(0), 0)

    def test_slide_runs_to_completion(self):
        # Invoke the slide callback repeatedly; since root.after is faked it
        # re-appends itself, so draining the queue exhausts slide_steps and
        # hits the early-return guard (step >= slide_steps).
        self._write_state()
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.mod._renderer_main(0)
            # The first scheduled callback is _slide. Run it many times.
            slide = root.after_calls[0]
            for _ in range(100):
                slide()
        # No assertion needed beyond "did not raise"; coverage records the
        # loop body + the guard return.

    def test_tick_dismissed_destroys(self):
        self._write_state(dismissed=True)
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.mod._renderer_main(0)
            # Run the tick callback (last scheduled) — dismissed -> destroy().
            for cb in list(root.after_calls):
                cb()
        self.assertTrue(getattr(root, "destroyed", False))

    def test_tick_expired_destroys(self):
        self._write_state(expiry_ts=1.0)   # already in the past
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.mod._renderer_main(0)
            for cb in list(root.after_calls):
                cb()
        self.assertTrue(getattr(root, "destroyed", False))

    def test_tick_parent_dead_destroys(self):
        # 2026-07-21 audit: the watchdog now consults the corpse-aware
        # core.parent_watch.parent_is_alive (as hud_card does), not the
        # psutil-backed _pid_alive that reports True for a kernel-stuck
        # corpse. Stub the authoritative layer to say "dead".
        self._write_state()
        root = _FakeTk()
        import core.parent_watch as _pw
        with inject_modules(tkinter=make_fake_tkinter(root)), \
             mock.patch.object(_pw, "parent_is_alive", return_value=False):
            self.mod._renderer_main(parent_pid=4242)
            for cb in list(root.after_calls):
                cb()
        # NB: getattr(root, "destroyed", False) is vacuously truthy on the
        # fakes (__getattr__ returns _noop), so check the instance dict.
        self.assertIs(root.__dict__.get("destroyed"), True)

    def test_tick_parent_dead_fallback_without_parent_watch(self):
        # If core.parent_watch is un-importable the tick falls back to the
        # local _pid_alive — a dead parent must still tear the card down.
        self._write_state()
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)), \
             block_import("core.parent_watch"), \
             mock.patch.object(self.mod, "_pid_alive", return_value=False):
            self.mod._renderer_main(parent_pid=4242)
            for cb in list(root.after_calls):
                cb()
        self.assertIs(root.__dict__.get("destroyed"), True)

    def test_tick_fallback_no_parent_to_watch_stays_up(self):
        # Fallback path, pid <= 0 ("no parent supplied"): must read alive —
        # parent_is_alive's pid<=0 convention is preserved by the lambda.
        self._write_state()
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)), \
             block_import("core.parent_watch"), \
             mock.patch.object(self.mod, "_pid_alive",
                               return_value=False) as pa:
            self.mod._renderer_main(parent_pid=0)
            for cb in list(root.after_calls):
                cb()
        # _pid_alive(0) would say False; the fallback must not even consult
        # it for pid<=0, and the card must survive the tick.
        pa.assert_not_called()
        self.assertNotIn("destroyed", root.__dict__)

    def test_tick_state_vanished_destroys(self):
        self._write_state()
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.mod._renderer_main(0)
            # Remove the state file so the tick's _load_state_safe returns None.
            os.unlink(self.state)
            for cb in list(root.after_calls):
                cb()
        self.assertTrue(getattr(root, "destroyed", False))

    def test_pid_write_failure_is_swallowed(self):
        # open() for the PID write raises, but render still proceeds (no state
        # afterward -> returns 0). Exercises the try/except around the PID write.
        real_open = open

        def _boom(path, *a, **k):
            if str(path).endswith("dossier_card.pid") and "w" in (a[0] if a else k.get("mode", "")):
                raise OSError("pid locked")
            return real_open(path, *a, **k)

        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)), \
             mock.patch("builtins.open", side_effect=_boom):
            self.assertEqual(self.mod._renderer_main(0), 0)

    def test_alpha_attribute_failure_swallowed(self):
        # root.attributes('-alpha', ...) raises (some WMs reject alpha) but
        # '-topmost' must still succeed -> only the alpha call raises.
        self._write_state()

        class _AlphaPickyTk(_FakeTk):
            def attributes(self, *a, **k):
                if a and a[0] == "-alpha":
                    raise RuntimeError("alpha unsupported")
                return self

        root = _AlphaPickyTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.assertEqual(self.mod._renderer_main(0), 0)

    def test_mainloop_keyboardinterrupt_swallowed(self):
        # Ctrl-C during mainloop is caught; the finally still removes the PID
        # file and the renderer returns 0.
        self._write_state()

        class _CtrlCTk(_FakeTk):
            def mainloop(self):
                raise KeyboardInterrupt

        root = _CtrlCTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.assertEqual(self.mod._renderer_main(0), 0)
        self.assertFalse(os.path.exists(self.pid))

    def test_tick_inner_exception_destroys(self):
        # _load_state_safe raising inside _tick hits the tick's outer except,
        # which destroys the window defensively.
        self._write_state()
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.mod._renderer_main(0)
            tick = root.after_calls[-1]   # last scheduled callback is _tick
            with mock.patch.object(self.mod, "_load_state_safe",
                                   side_effect=RuntimeError("state read blew up")):
                tick()
        self.assertTrue(getattr(root, "destroyed", False))

    def test_finally_pid_remove_failure_swallowed(self):
        # The finally's os.remove(_PID_FILE) failing must not propagate.
        self._write_state()
        root = _FakeTk()
        with inject_modules(tkinter=make_fake_tkinter(root)), \
             mock.patch("os.remove", side_effect=OSError("cannot remove")):
            self.assertEqual(self.mod._renderer_main(0), 0)

    def test_tick_except_destroy_also_raises_swallowed(self):
        # Both the tick body AND root.destroy() raise -> the doubly-nested
        # except still swallows it (the watchdog tick never propagates).
        self._write_state()

        class _DestroyBoomTk(_FakeTk):
            def destroy(self):
                raise RuntimeError("destroy failed")

        root = _DestroyBoomTk()
        with inject_modules(tkinter=make_fake_tkinter(root)):
            self.mod._renderer_main(0)
            tick = root.after_calls[-1]
            with mock.patch.object(self.mod, "_load_state_safe",
                                   side_effect=RuntimeError("read blew up")):
                # Must not raise despite destroy() also failing.
                tick()


# ──────────────────────────────────────────────────────────────────────────
# NEW (2026-07-21 audit): parent-watchdog stale-duplicate sweep
# ──────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


class ParentWatchdogStaleDuplicateSweepTests(unittest.TestCase):
    """2026-07-21 audit: dossier's card renderer was the LAST --parent-pid
    watchdog still gating on the psutil-backed _pid_alive, which reports True
    for both Windows dead states (terminated-but-unreaped and kernel-stuck
    corpse), so the card outlived a dead JARVIS. hud_card was migrated to
    core.parent_watch on 2026-07-14 (bug-hunt #24) while dossier's copy
    rotted — the classic stale-duplicate class — so this guard sweeps EVERY
    subprocess card renderer rather than pinning one file: a future renderer
    copied from the stale pattern fails the sweep, not just dossier.
    """

    _EXCLUDED_DIRS = ("tests", "__pycache__", ".git", ".claude",
                      "backups", "_backups", "dist", "models",
                      "logs", "logs_staging", "data", "data_staging",
                      "node_modules", "venv", ".venv")

    def _card_renderer_sources(self):
        """Map relpath -> source for every production file that is a
        subprocess card renderer with a local psutil pid helper, i.e. it
        both carries a --parent-pid CLI contract and defines _pid_alive.
        (Parent-side managers like blue_green_manager keep _pid_alive for
        CHILD liveness — the safe direction — and don't match.)"""
        found = {}
        for base, dirs, files in os.walk(_PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in self._EXCLUDED_DIRS]
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(base, fn)
                try:
                    with open(path, "r", encoding="utf-8",
                              errors="replace") as fh:
                        src = fh.read()
                except OSError:
                    continue
                if "--parent-pid" in src and "def _pid_alive(" in src:
                    rel = os.path.relpath(path, _PROJECT_ROOT)
                    found[rel.replace("\\", "/")] = src
        return found

    def test_sweep_discovers_the_known_renderers(self):
        # If the discovery predicate ever rots, the sweep would pass
        # vacuously — pin the two renderers it must always see.
        rels = set(self._card_renderer_sources())
        self.assertIn("hud_card.py", rels)
        self.assertIn("skills/dossier.py", rels)

    def test_every_card_renderer_watchdog_uses_parent_watch(self):
        offenders = []
        for rel, src in sorted(self._card_renderer_sources().items()):
            if "from core.parent_watch import parent_is_alive" not in src:
                offenders.append(
                    f"{rel}: parent watchdog missing the corpse-aware "
                    f"core.parent_watch.parent_is_alive import")
            if "not _pid_alive(parent_pid)" in src:
                offenders.append(
                    f"{rel}: pre-migration psutil gate "
                    f"'not _pid_alive(parent_pid)' still present")
        self.assertEqual([], offenders,
                         "card renderers must watch their parent via "
                         "core.parent_watch (psutil.pid_exists reads a "
                         "corpsed parent as alive):\n" + "\n".join(offenders))


# ──────────────────────────────────────────────────────────────────────────
# NEW: _act_dossier report-formatting + render-error branches
# ──────────────────────────────────────────────────────────────────────────
class DossierActReportTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dossier")

    def _stub_render(self):
        return (
            mock.patch.object(self.mod, "_write_state"),
            mock.patch.object(self.mod, "_ensure_renderer_running"),
            mock.patch.object(self.mod, "_top_monitor_geometry",
                              return_value=(0, 0, 1920, 1080)),
        )

    def test_report_includes_all_sections_populated(self):
        a, b, c = self._stub_render()
        with a, b, c, \
             mock.patch.object(self.mod, "_gather_memory",
                               return_value=["sam fact"]), \
             mock.patch.object(self.mod, "_gather_tasks",
                               return_value=["call sam"]), \
             mock.patch.object(self.mod, "_gather_logs",
                               return_value=["[10:00] sam pinged"]), \
             mock.patch.object(self.mod, "_gather_web",
                               return_value="Sam is a manager."):
            out = self.actions["dossier"]("Sam")
        self.assertIn("facts (newest first)", out)
        self.assertIn("tasks (open first)", out)
        self.assertIn("recent log mentions:", out)
        self.assertIn("web abstract: Sam is a manager.", out)
        self.assertIn("sam pinged", out)

    def test_report_all_sections_empty(self):
        a, b, c = self._stub_render()
        with a, b, c, \
             mock.patch.object(self.mod, "_gather_memory", return_value=[]), \
             mock.patch.object(self.mod, "_gather_tasks", return_value=[]), \
             mock.patch.object(self.mod, "_gather_logs", return_value=[]), \
             mock.patch.object(self.mod, "_gather_web", return_value=""):
            out = self.actions["dossier"]("Sam")
        self.assertIn("facts: (none)", out)
        self.assertIn("tasks: (none)", out)
        self.assertIn("recent log mentions: (none)", out)
        self.assertIn("web abstract: (unavailable)", out)

    def test_card_render_exception_is_swallowed(self):
        # _write_state raising must not break the report — the structured text
        # is still returned (card-render failure is logged, not fatal).
        with mock.patch.object(self.mod, "_gather_memory",
                               return_value=["sam fact"]), \
             mock.patch.object(self.mod, "_gather_tasks", return_value=[]), \
             mock.patch.object(self.mod, "_gather_logs", return_value=[]), \
             mock.patch.object(self.mod, "_gather_web", return_value=""), \
             mock.patch.object(self.mod, "_top_monitor_geometry",
                               return_value=(0, 0, 1920, 1080)), \
             mock.patch.object(self.mod, "_write_state",
                               side_effect=OSError("disk full")):
            out = self.actions["dossier"]("Sam")
        self.assertIn("DOSSIER on 'Sam'", out)
        self.assertIn("sam fact", out)

    def test_all_aliases_dispatch_to_handler(self):
        for alias in ("dossier", "pull_up_file", "pull_up_dossier", "file_on",
                      "dossier_on", "what_do_you_have_on", "whats_on_file"):
            self.assertIn(alias, self.actions)
            # Empty arg short-circuits without touching the card pipeline.
            self.assertIn("need a subject", self.actions[alias]("").lower())


# ──────────────────────────────────────────────────────────────────────────
# NEW: register() host-module patching
# ──────────────────────────────────────────────────────────────────────────
class DossierRegisterTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dossier")
        self._names = {
            "dossier", "pull_up_file", "pull_up_dossier", "file_on",
            "dossier_on", "what_do_you_have_on", "whats_on_file",
        }

    def test_register_wires_all_aliases(self):
        acts = {}
        # No bobert_companion present -> the INFORMATIVE_ACTIONS patch is a
        # no-op (import raises, swallowed) but every alias is still wired.
        with inject_modules(bobert_companion=None):
            self.mod.register(acts)
        for name in self._names:
            self.assertIn(name, acts)
            self.assertIs(acts[name], self.mod._act_dossier)

    def test_register_extends_host_sets(self):
        bc = types.ModuleType("bobert_companion")
        bc.INFORMATIVE_ACTIONS = set()
        bc.LONG_RUNNING_ACTIONS = set()
        bc._MID_TASK_STATUS_BUCKET = {}
        with inject_modules(bobert_companion=bc):
            self.mod.register({})
        self.assertTrue(self._names <= bc.INFORMATIVE_ACTIONS)
        self.assertTrue(self._names <= bc.LONG_RUNNING_ACTIONS)
        for alias in self._names:
            self.assertEqual(bc._MID_TASK_STATUS_BUCKET[alias], "dossier")

    def test_register_tolerates_missing_host_attrs(self):
        # bobert_companion present but without the expected sets/dict -> the
        # isinstance guards skip them; register() must not raise.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            acts = {}
            self.mod.register(acts)   # must not raise
        self.assertIn("dossier", acts)

    def test_register_swallows_host_patch_exception(self):
        # Accessing INFORMATIVE_ACTIONS blows up -> the except branch logs and
        # register() still completes with aliases wired.
        class _Boom(types.ModuleType):
            @property
            def INFORMATIVE_ACTIONS(self):
                raise RuntimeError("kaboom")

        bc = _Boom("bobert_companion")
        with inject_modules(bobert_companion=bc):
            acts = {}
            self.mod.register(acts)
        self.assertIn("dossier", acts)


# ──────────────────────────────────────────────────────────────────────────
# NEW: __main__ CLI dispatch
# ──────────────────────────────────────────────────────────────────────────
class DossierCliTests(unittest.TestCase):
    """Drive the ``if __name__ == '__main__'`` block.

    The source is compiled once and exec'd into a *controlled* globals dict that
    starts as a shallow copy of the already-loaded module's namespace with
    ``__name__`` forced to ``"__main__"``. Because the block calls
    ``_act_dossier`` / ``_renderer_alive`` / ``_renderer_main`` etc. as globals,
    seeding stubs into that dict gives full control with no real window, process
    spawn, network, file write, or sleep — and no second import side effects.
    """

    @classmethod
    def setUpClass(cls):
        mod, _ = load_skill_isolated("dossier")
        with open(mod.__file__, "r", encoding="utf-8") as f:
            src = f.read()
        # Slice out just the body of ``if __name__ == "__main__":`` and dedent
        # it. We exec ONLY this block against the already-loaded module's
        # namespace, so the real top-level ``def``s are NOT re-run (which would
        # clobber our seeded stubs). The block references argparse/sys/time and
        # the action/renderer helpers as globals — all resolvable from the
        # module dict, with our stubs layered on top.
        import textwrap
        marker = 'if __name__ == "__main__":'
        idx = src.index(marker)
        body = src[idx + len(marker):]
        # Drop the leading newline, keep the indented block, then dedent.
        body = body.lstrip("\n")
        cls.block_src = textwrap.dedent(body)
        # Compile under a SYNTHETIC filename (not the real module path) so
        # coverage.py attributes these exec'd lines to "<dossier-main-block>"
        # rather than overwriting skills/dossier.py's coverage record. Exec'ing
        # under the real filename would key the data to dossier.py and clobber
        # the genuine module coverage with just this ~20-line block.
        cls.code = compile(cls.block_src, "<dossier-main-block>", "exec")

    def setUp(self):
        self.mod, _ = load_skill_isolated("dossier")

    def _run_main(self, argv, **stub_globals):
        g = dict(self.mod.__dict__)
        g["__name__"] = "__main__"
        g.update(stub_globals)
        saved_argv = sys.argv
        sys.argv = ["dossier.py"] + argv
        try:
            exec(self.code, g)
        finally:
            sys.argv = saved_argv
        return g

    def test_no_args_prints_help(self):
        # No flags -> argparse prints help; nothing raises, no SystemExit.
        printed = {}
        # Seed a fake parser so we can assert help is printed without touching
        # the real argparse output stream.
        with mock.patch.object(self.mod.argparse.ArgumentParser, "print_help",
                               lambda self: printed.setdefault("help", True)):
            self._run_main([])
        self.assertTrue(printed.get("help"))

    def test_demo_compiles_and_exits_zero(self):
        # --demo path: stub _act_dossier (so no card/network) and force the
        # keep-alive loop to exit immediately via _renderer_alive=False.
        act = mock.MagicMock(return_value="DOSSIER on 'acme': ...")
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(
                ["--demo", "acme"],
                _act_dossier=act,
                _renderer_alive=lambda: False,        # loop exits on first check
                time=types.SimpleNamespace(time=lambda: 0.0, sleep=lambda *_: None),
            )
        self.assertEqual(ctx.exception.code, 0)
        act.assert_called_once_with("acme")

    def test_render_dispatch_invokes_renderer_main(self):
        # --render path: _renderer_main is stubbed to return 7 -> sys.exit(7).
        rm = mock.MagicMock(return_value=7)
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(["--render", "--parent-pid", "5"], _renderer_main=rm)
        self.assertEqual(ctx.exception.code, 7)
        rm.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
