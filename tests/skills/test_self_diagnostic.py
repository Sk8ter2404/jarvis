"""Logic tests for skills/self_diagnostic.py.

self_diagnostic probes every JARVIS subsystem and auto-queues repair tasks.
The probes themselves touch hardware/network, so tests target the
deterministic, mockable LOGIC around them:
  • the canonical _result shape,
  • the PnP camera diagnosis state-machine (absent / problem / ok / unknown),
  • the recent-problem voice-mood flag window,
  • _suggested_files_for / _suggested_files_for_action source-file hints,
  • _traceback_excerpt tail extraction,
  • _last_successful_ts history walk,
  • _collect_action_error_groups grouping + threshold,
  • _summarise / diagnostic_status / diagnostic_history / whats_broken
    rendering (history + todo paths redirected to a temp dir).

A fake bobert_companion is injected where helpers consult it. The scheduler /
boot-sweep thread in register() is neutered by the harness. No probe ever runs
against real hardware.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class SelfDiagPureTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("self_diagnostic")

    # ── _result ──────────────────────────────────────────────────────────
    def test_result_ok_clears_error(self):
        r = self.mod._result(True, 12.34)
        self.assertTrue(r["ok"])
        self.assertEqual(r["latency_ms"], 12.3)   # rounded to 1 dp
        self.assertIsNone(r["error"])

    def test_result_failure_defaults_error(self):
        r = self.mod._result(False, 5.0)
        self.assertEqual(r["error"], "unknown error")
        self.assertFalse(r["ok"])

    # ── _camera_pnp_diagnosis ────────────────────────────────────────────
    def test_pnp_diagnosis_none_is_unknown(self):
        d = self.mod._camera_pnp_diagnosis(None)
        self.assertEqual(d["failure_mode"], "unknown")
        self.assertFalse(d["hardware_present"])

    def test_pnp_diagnosis_absent(self):
        d = self.mod._camera_pnp_diagnosis([])
        self.assertEqual(d["failure_mode"], "absent")
        self.assertFalse(d["hardware_present"])

    def test_pnp_diagnosis_absent_when_not_present(self):
        devices = [{"name": "Cam", "status": "OK", "problem": 0, "present": False}]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "absent")

    def test_pnp_diagnosis_problem_device(self):
        devices = [{"name": "Logi Cam", "status": "Error", "problem": 43,
                    "present": True}]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "problem")
        self.assertTrue(d["has_problem_device"])
        self.assertIn("Logi Cam", d["summary"])
        self.assertIn("43", d["summary"])

    def test_pnp_diagnosis_healthy(self):
        devices = [{"name": "Cam", "status": "OK", "problem": 0, "present": True}]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "ok")
        self.assertEqual(d["healthy_devices"], 1)

    def test_pnp_diagnosis_healthy_wins_over_problem_when_mixed(self):
        devices = [
            {"name": "Good", "status": "OK", "problem": 0, "present": True},
            {"name": "Bad", "status": "Error", "problem": 22, "present": True},
        ]
        d = self.mod._camera_pnp_diagnosis(devices)
        self.assertEqual(d["failure_mode"], "ok")     # at least one healthy
        self.assertTrue(d["has_problem_device"])
        self.assertEqual(d["healthy_devices"], 1)

    # ── recent-problem flag ──────────────────────────────────────────────
    def test_recent_problem_flag_window(self):
        self.mod._recent_problem_at[0] = 0.0
        self.assertFalse(self.mod.get_recent_problem_flag(now=1000.0))
        self.mod._mark_recent_problem(now=1000.0)
        self.assertTrue(self.mod.get_recent_problem_flag(now=1000.0 + 60))
        # Beyond the window → flag clears.
        self.assertFalse(self.mod.get_recent_problem_flag(
            now=1000.0 + self.mod._RECENT_PROBLEM_WINDOW_SEC + 1))

    # ── _suggested_files_for ─────────────────────────────────────────────
    def test_suggested_files_known_and_unknown(self):
        self.assertIn("face_tracker", self.mod._suggested_files_for("webcam"))
        self.assertIn("bambu_monitor", self.mod._suggested_files_for("bambu"))
        self.assertEqual(self.mod._suggested_files_for("nonexistent"), "(no suggestion)")

    # ── _suggested_files_for_action ──────────────────────────────────────
    def test_suggested_files_for_action_maps_skill_module(self):
        bc = mock.MagicMock()
        fn = mock.MagicMock()
        fn.__module__ = "skill_timer"
        bc.ACTIONS = {"set_timer": fn}
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            out = self.mod._suggested_files_for_action("set_timer")
        self.assertEqual(out, "skills/timer.py")

    def test_suggested_files_for_action_core_module(self):
        bc = mock.MagicMock()
        fn = mock.MagicMock()
        fn.__module__ = "core.tts"
        bc.ACTIONS = {"speak": fn}
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._suggested_files_for_action("speak"), "core/tts.py")

    def test_suggested_files_for_action_no_bc(self):
        with mock.patch.object(self.mod, "_bc", return_value=None):
            self.assertIn("bobert_companion.py",
                          self.mod._suggested_files_for_action("anything"))

    # ── _traceback_excerpt ───────────────────────────────────────────────
    def test_traceback_excerpt_keeps_tail(self):
        tb = "line1\n\nline2\nline3\nline4\nline5\nline6"
        out = self.mod._traceback_excerpt(tb, max_lines=3)
        self.assertEqual(out.splitlines(), ["line4", "line5", "line6"])

    def test_traceback_excerpt_empty(self):
        self.assertEqual(self.mod._traceback_excerpt(""), "")

    # ── _last_successful_ts ──────────────────────────────────────────────
    def test_last_successful_ts_finds_most_recent_ok(self):
        history = [
            {"iso": "2026-05-01T00:00:00", "probes": {"webcam": {"ok": True}}},
            {"iso": "2026-05-02T00:00:00", "probes": {"webcam": {"ok": False}}},
        ]
        # Walks backward; the most recent OK is the 05-01 run.
        self.assertEqual(self.mod._last_successful_ts(history, "webcam"),
                         "2026-05-01T00:00:00")

    def test_last_successful_ts_none_when_never_ok(self):
        history = [{"iso": "2026-05-02T00:00:00", "probes": {"stt": {"ok": False}}}]
        self.assertIsNone(self.mod._last_successful_ts(history, "stt"))

    # ── _collect_action_error_groups ─────────────────────────────────────
    def test_collect_action_errors_groups_and_thresholds(self):
        # 3 same-class errors for one action (>= group count 3) and 1 for
        # another (below threshold).
        errors = (
            [{"action": "play_music", "exc_class": "KeyError", "exc_msg": "no key",
              "traceback": "tb", "ts": float(i)} for i in range(3)]
            + [{"action": "see_screen", "exc_class": "TimeoutError",
                "exc_msg": "slow", "traceback": "tb2", "ts": 9.0}]
        )
        bc = mock.MagicMock()
        bc.get_recent_action_errors.return_value = errors
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            groups = self.mod._collect_action_error_groups()
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["action"], "play_music")
        self.assertEqual(g["exc_class"], "KeyError")
        self.assertEqual(g["count"], 3)

    def test_collect_action_errors_empty_when_no_getter(self):
        bc = mock.MagicMock(spec=[])   # no get_recent_action_errors attr
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.assertEqual(self.mod._collect_action_error_groups(), [])


class SelfDiagSummaryTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("self_diagnostic")

    def _run(self, failed, sev=None):
        sev = sev or {}
        return {
            "ts": 0.0, "iso": "2026-05-30T00:00:00", "duration_ms": 1234.0,
            "probes": {c: {"severity": sev.get(c, self.mod.SEVERITY_MED),
                           "error": f"{c} down"} for c in failed},
            "failed": failed,
            "severity_failed": {c: sev.get(c, self.mod.SEVERITY_MED) for c in failed},
        }

    def test_summarise_all_nominal(self):
        run = self._run([])
        out = self.mod._summarise(run, [])
        self.assertIn("All systems nominal", out)

    def test_summarise_single_issue(self):
        run = self._run(["microphone"], {"microphone": self.mod.SEVERITY_HIGH})
        out = self.mod._summarise(run, ["microphone"])
        self.assertIn("one issue — microphone", out)
        self.assertIn("1 high", out.lower())
        self.assertIn("1 repair task", out)

    def test_summarise_many_issues_truncates(self):
        comps = ["webcam", "stt", "tts", "gpu", "ram", "disk"]
        run = self._run(comps)
        out = self.mod._summarise(run, [])
        self.assertIn("6 issues", out)
        self.assertIn("and 3 more", out)   # lists first 3 + "and N more"

    def test_diagnostic_status_no_run(self):
        self.mod._state["last_run"] = None
        self.assertIn("No diagnostic has run yet", self.actions["diagnostic_status"](""))

    def test_diagnostic_status_reports_failures(self):
        run = self._run(["microphone"])
        run["ts"] = self.mod._now() - 30
        self.mod._state["last_run"] = run
        out = self.actions["diagnostic_status"]("")
        self.assertIn("microphone", out)
        self.assertIn("seconds ago", out)

    def test_diagnostic_status_all_nominal_age(self):
        run = self._run([])
        run["ts"] = self.mod._now() - 5
        self.mod._state["last_run"] = run
        out = self.actions["diagnostic_status"]("")
        self.assertIn("All systems nominal", out)


class SelfDiagFileTests(unittest.TestCase):
    """Tests touching jarvis_todo.md / history — redirect those paths to a
    temp dir so the real project files aren't read or written."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("self_diagnostic")
        self.tmp = tempfile.mkdtemp(prefix="selfdiag_test_")
        self.addCleanup(self._cleanup)
        self.todo = os.path.join(self.tmp, "jarvis_todo.md")
        self.history = os.path.join(self.tmp, "self_diagnostic.json")
        self.mod._TODO_PATH = self.todo
        self.mod._HISTORY_PATH = self.history

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

    # ── whats_broken ─────────────────────────────────────────────────────
    def test_whats_broken_no_todo_file(self):
        # _TODO_PATH points at a not-yet-created temp file.
        self.assertIn("can't find jarvis_todo.md", self.actions["whats_broken"](""))

    def test_whats_broken_clean_queue(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("# todo\n- [ ] **2026-05-30** something unrelated\n")
        self.assertIn("queue is clean", self.actions["whats_broken"](""))

    def test_whats_broken_lists_open_tasks(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: microphone reports x.\n")
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: stt reports y.\n")
        out = self.actions["whats_broken"]("")
        self.assertIn("2 open repair tasks", out)
        self.assertIn("microphone", out)
        self.assertIn("stt", out)

    def test_whats_broken_single_task_with_date(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-29** [self-diag] - Fix: webcam reports z.\n")
        out = self.actions["whats_broken"]("")
        self.assertIn("One open repair task", out)
        self.assertIn("webcam", out)
        self.assertIn("2026-05-29", out)

    # ── _open_selfdiag_components dedupe ──────────────────────────────────
    def test_open_components_parses_todo(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: disk reports full.\n")
            f.write("- [x] **2026-05-29** [self-diag] - Fix: ram reports done.\n")  # closed
        comps = self.mod._open_selfdiag_components()
        self.assertIn("disk", comps)
        self.assertNotIn("ram", comps)   # checked box → not open

    # ── diagnostic_history ───────────────────────────────────────────────
    def test_diagnostic_history_empty(self):
        self.assertIn("No diagnostic history", self.actions["diagnostic_history"](""))

    def test_diagnostic_history_lists_runs(self):
        import json
        runs = [
            {"iso": "2026-05-30T01:00:00", "failed": []},
            {"iso": "2026-05-30T02:00:00", "failed": ["stt", "tts"]},
        ]
        with open(self.history, "w", encoding="utf-8") as f:
            json.dump(runs, f)
        out = self.actions["diagnostic_history"]("5")
        self.assertIn("all nominal", out)
        self.assertIn("2 issue(s)", out)
        self.assertIn("stt", out)

    # ── _queue_repair_task ───────────────────────────────────────────────
    def test_queue_repair_task_skips_low_severity(self):
        run = {"probes": {"claude_api": {"severity": self.mod.SEVERITY_LOW,
                                         "error": "capped"}}}
        self.assertFalse(self.mod._queue_repair_task("claude_api", run, []))
        # No file written for a LOW failure.
        self.assertFalse(os.path.exists(self.todo))

    def test_queue_repair_task_appends_for_med(self):
        run = {"probes": {"webcam": {"severity": self.mod.SEVERITY_MED,
                                     "error": "no frame", "latency_ms": 40,
                                     "details": {}}}}
        ok = self.mod._queue_repair_task("webcam", run, [])
        self.assertTrue(ok)
        with open(self.todo, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("[self-diag] - Fix: webcam", body)
        self.assertIn("face_tracker", body)   # suggested file hint

    def test_queue_repair_task_dedupes_open_component(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30** [self-diag] - Fix: webcam reports x.\n")
        run = {"probes": {"webcam": {"severity": self.mod.SEVERITY_MED,
                                     "error": "again", "latency_ms": 1, "details": {}}}}
        # Already open → not queued again.
        self.assertFalse(self.mod._queue_repair_task("webcam", run, []))


if __name__ == "__main__":
    unittest.main()
