"""Logic tests for skills/bambu_monitor.py.

bambu_monitor owns the MQTT connection + shared _state dict and turns Bambu
print reports into JARVIS announcements. We exercise:

  • pure formatters: _format_minutes, _format_temp, _strip_filename
  • _on_message — JSON report → _state extraction (incl. AMS / filename key
    variants and chamber-history sampling)
  • _compute_risk_level — nominal / chamber-swing-amber / hard-fault-red
  • _handle_state_change — 0% start, milestone, layer-1, FINISH, in-flight
    error, FAILED announcements + the new-print / cold-start suppression logic
  • check_print / how_is_the_print actions across idle / running / pause /
    finish / no-data states
  • is_printer_offline and get_last_print_completion_summary

register() is never allowed to touch the network or real files: we load with
bobert_companion mocked out of sys.modules and the persistence helpers + flag /
overlay writers patched, and threads are neutered by the harness. Every test
that could announce patches the skill's _enqueue_speech so nothing reaches the
real pending_speech.json, and the durable reminder-state load/save are stubbed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated, make_fake_skill_utils


def _load_bambu():
    """Load bambu_monitor with every external side-effect neutralised.

    register() calls _maybe_prompt_for_credentials() (writes a flag file +
    announces) and start_monitor() (no-op without config / paho). We mock
    bobert_companion so the announce path can't write pending_speech.json, and
    stub the flag / overlay / reminder persistence so registration is inert.
    """
    fake_bc = mock.MagicMock()
    # No config attributes → _read_config() returns ("","","") → start_monitor
    # bails before constructing any MQTT client.
    with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
        with mock.patch("os.path.exists", return_value=True):
            # exists()==True short-circuits _maybe_prompt_for_credentials (it
            # thinks the flag is already written) so register() stays silent.
            mod, actions = load_skill_isolated(
                "bambu_monitor", utils=make_fake_skill_utils())
    return mod, actions


class BambuFormatTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()

    def test_format_minutes_units(self):
        f = self.mod._format_minutes
        self.assertEqual(f(0), "less than a minute")
        self.assertEqual(f(1), "1 minute")
        self.assertEqual(f(5), "5 minutes")
        self.assertEqual(f(60), "1 hour")
        self.assertEqual(f(90), "1 hour and 30 minutes")
        self.assertEqual(f(125), "2 hours and 5 minutes")

    def test_format_minutes_invalid(self):
        self.assertEqual(self.mod._format_minutes(None), "")
        self.assertEqual(self.mod._format_minutes("soon"), "")

    def test_format_temp_rounds_and_labels(self):
        self.assertEqual(self.mod._format_temp(219.8), "220 degrees")
        self.assertEqual(self.mod._format_temp(60), "60 degrees")

    def test_format_temp_zero_or_missing_is_blank(self):
        # 0 °C means "sensor not active"; non-numeric is also dropped.
        self.assertEqual(self.mod._format_temp(0), "")
        self.assertEqual(self.mod._format_temp(None), "")
        self.assertEqual(self.mod._format_temp("hot"), "")

    def test_strip_filename(self):
        self.assertEqual(
            self.mod._strip_filename("/cache/Bracket_v2.gcode.3mf"),
            "Bracket v2.gcode")  # only the final extension is stripped
        self.assertEqual(self.mod._strip_filename("Test_Part.3mf"), "Test Part")
        self.assertEqual(self.mod._strip_filename(""), "")

    def test_strip_filename_caps_length(self):
        out = self.mod._strip_filename("x" * 200 + ".3mf")
        self.assertLessEqual(len(out), 60)


class BambuOnMessageTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
            self.mod._chamber_history.clear()

    def _msg(self, payload: dict):
        m = mock.MagicMock()
        m.payload = json.dumps(payload).encode("utf-8")
        return m

    def test_on_message_extracts_print_fields(self):
        # Patch the downstream side-effects so we isolate the parse step.
        with mock.patch.object(self.mod, "_handle_state_change"), \
             mock.patch.object(self.mod, "_write_overlay_state"):
            self.mod._on_message(None, None, self._msg({"print": {
                "gcode_state": "RUNNING", "layer_num": 47, "total_layer": 312,
                "mc_percent": 15, "mc_remaining": 88, "nozzle_temper": 220.0,
                "bed_temper": 60.0, "subtask_name": "widget.3mf"}}))
        with self.mod._state_lock:
            self.assertEqual(self.mod._state["gcode_state"], "RUNNING")
            self.assertEqual(self.mod._state["layer_num"], 47)
            self.assertEqual(self.mod._state["filename"], "widget.3mf")
            self.assertGreater(self.mod._state["last_update"], 0.0)

    def test_on_message_bad_payload_is_ignored(self):
        m = mock.MagicMock()
        m.payload = b"\xff\xfe not json"
        # Must not raise and must not advance last_update.
        with mock.patch.object(self.mod, "_handle_state_change"), \
             mock.patch.object(self.mod, "_write_overlay_state"):
            self.mod._on_message(None, None, m)
        with self.mod._state_lock:
            self.assertEqual(self.mod._state["last_update"], 0.0)

    def test_on_message_samples_chamber_history(self):
        with mock.patch.object(self.mod, "_handle_state_change"), \
             mock.patch.object(self.mod, "_write_overlay_state"):
            for temp in (30.0, 31.0, 0.0, 32.0):  # the 0.0 must be ignored
                self.mod._on_message(None, None,
                                     self._msg({"print": {"chamber_temper": temp}}))
        self.assertEqual(list(self.mod._chamber_history), [30.0, 31.0, 32.0])

    def test_on_message_accepts_ams_object(self):
        with mock.patch.object(self.mod, "_handle_state_change"), \
             mock.patch.object(self.mod, "_write_overlay_state"):
            self.mod._on_message(None, None,
                                 self._msg({"print": {"ams": {"humidity": 4}}}))
        with self.mod._state_lock:
            self.assertEqual(self.mod._state["ams_status"], {"humidity": 4})


class BambuRiskLevelTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
            self.mod._chamber_history.clear()

    def _set(self, **kw):
        with self.mod._state_lock:
            self.mod._state.update(kw)

    def test_nominal_is_level_zero(self):
        self._set(gcode_state="RUNNING", print_error=0)
        level, note = self.mod._compute_risk_level()
        self.assertEqual(level, 0)
        self.assertEqual(note, "")

    def test_failed_state_is_red(self):
        self._set(gcode_state="FAILED")
        level, note = self.mod._compute_risk_level()
        self.assertEqual(level, 2)
        self.assertIn("FAILED", note)

    def test_nonzero_error_is_red(self):
        self._set(gcode_state="RUNNING", print_error=8451)
        level, note = self.mod._compute_risk_level()
        self.assertEqual(level, 2)
        self.assertIn("8451", note)

    def test_ams_fault_keyword_is_red(self):
        self._set(gcode_state="RUNNING", print_error=0,
                  ams_status="tray 1 runout detected")
        level, note = self.mod._compute_risk_level()
        self.assertEqual(level, 2)
        self.assertIn("AMS", note)

    def test_chamber_swing_is_amber(self):
        self._set(gcode_state="RUNNING", print_error=0)
        with self.mod._state_lock:
            self.mod._chamber_history.extend([30.0, 33.0, 37.0])  # swing 7 ≥ 6
        level, note = self.mod._compute_risk_level()
        self.assertEqual(level, 1)
        self.assertIn("swing", note.lower())

    def test_small_chamber_swing_stays_nominal(self):
        self._set(gcode_state="RUNNING", print_error=0)
        with self.mod._state_lock:
            self.mod._chamber_history.extend([30.0, 31.0, 32.0])  # swing 2 < 6
        level, _ = self.mod._compute_risk_level()
        self.assertEqual(level, 0)


class BambuStateChangeTests(unittest.TestCase):
    """_handle_state_change announcement logic. All speech + durable
    persistence is stubbed so nothing real is written."""

    def setUp(self):
        self.mod, self.actions = _load_bambu()
        self._reset_state()
        # Fresh per-print bookkeeping.
        self.mod._last_gcode_state[0] = None
        self.mod._current_print_filename[0] = None
        self.mod._announced_milestones.clear()
        self.mod._announced_error_codes.clear()
        self.mod._announced_start[0] = False
        self.mod._announced_layer1[0] = False
        self.mod._last_finish_announced_at[0] = 0.0
        self.mod._post_finish_bed_watch[0] = False
        self.mod._bed_cool_announced[0] = False
        # No durable reminder file interaction during these tests.
        self._rload = mock.patch.object(self.mod, "_load_reminder_persistence",
                                        return_value={})
        self._rsave = mock.patch.object(self.mod, "_save_reminder_persistence")
        self._rload.start(); self._rsave.start()
        self.addCleanup(self._rload.stop)
        self.addCleanup(self._rsave.stop)

    def _reset_state(self):
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0

    def _set(self, **kw):
        with self.mod._state_lock:
            self.mod._state.update(kw)

    def _run(self):
        """Drive one _handle_state_change with speech captured."""
        with mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._handle_state_change()
        return [c.args[0] for c in enq.call_args_list]

    def test_fresh_print_announces_start_with_eta(self):
        # prev gcode is a terminal state so this counts as a witnessed start.
        self.mod._last_gcode_state[0] = "IDLE"
        self._set(gcode_state="RUNNING", filename="cube.3mf", mc_percent=0,
                  mc_remaining=252)
        msgs = self._run()
        self.assertTrue(any("Print started" in m for m in msgs))
        self.assertTrue(any("4 hours" in m for m in msgs))  # 252 min
        self.assertTrue(self.mod._announced_start[0])

    def test_cold_start_midflight_suppresses_start_and_passed_milestones(self):
        # prev_gcode None + RUNNING + already at 60% → no "Print started",
        # and 25/50 are pre-marked so they never blurt.
        self.mod._last_gcode_state[0] = None
        self._set(gcode_state="RUNNING", filename="big.3mf", mc_percent=60,
                  layer_num=200, mc_remaining=120)
        msgs = self._run()
        self.assertFalse(any("Print started" in m for m in msgs))
        self.assertIn(25, self.mod._announced_milestones)
        self.assertIn(50, self.mod._announced_milestones)

    def test_milestone_25_announced_once(self):
        # Already past the start; cross 25%.
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self._set(gcode_state="RUNNING", filename="cube.3mf", mc_percent=25,
                  mc_remaining=180)
        first = self._run()
        self.assertTrue(any("25%" in m for m in first))
        # Second pass at the same percent must not re-announce.
        second = self._run()
        self.assertFalse(any("25%" in m for m in second))

    def test_layer_one_adhesion_announced(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self._set(gcode_state="RUNNING", filename="cube.3mf", layer_num=2,
                  mc_percent=1)
        msgs = self._run()
        self.assertTrue(any("Layer 1 adhesion" in m for m in msgs))
        self.assertTrue(self.mod._announced_layer1[0])

    def test_inflight_error_announced_with_layer(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self._set(gcode_state="RUNNING", filename="cube.3mf",
                  print_error=8451, layer_num=142)
        msgs = self._run()
        hit = [m for m in msgs if "Error code 8451" in m]
        self.assertTrue(hit)
        self.assertIn("layer 142", hit[0])

    def test_inflight_error_deduped(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self._set(gcode_state="RUNNING", filename="cube.3mf",
                  print_error=777, layer_num=10)
        self._run()
        again = self._run()
        self.assertFalse(any("Error code 777" in m for m in again))

    def test_failed_state_announced(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self._set(gcode_state="FAILED", filename="cube.3mf", layer_num=88)
        msgs = self._run()
        hit = [m for m in msgs if "failed" in m.lower()]
        self.assertTrue(hit)
        self.assertIn("layer 88", hit[0])

    def test_finish_announces_and_arms_bed_watch(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self._set(gcode_state="FINISH", filename="cube.3mf")
        msgs = self._run()
        self.assertTrue(any("Print complete" in m for m in msgs))
        self.assertTrue(self.mod._post_finish_bed_watch[0])

    def test_bed_cool_followup_after_finish(self):
        # Arm the bed watch (as if FINISH already announced), then drop bed temp.
        self.mod._post_finish_bed_watch[0] = True
        self.mod._bed_cool_announced[0] = False
        self.mod._last_gcode_state[0] = "FINISH"
        self.mod._current_print_filename[0] = "cube"
        self._set(gcode_state="FINISH", filename="cube.3mf", bed_temper=35.0)
        msgs = self._run()
        self.assertTrue(any("bed has cooled" in m.lower() for m in msgs))
        self.assertTrue(self.mod._bed_cool_announced[0])

    def test_state_change_hook_is_fired(self):
        seen = []
        self.mod.register_state_change_hook(
            lambda snap, prev, cur: seen.append((prev, cur)))
        self.mod._last_gcode_state[0] = "IDLE"
        self._set(gcode_state="RUNNING", filename="cube.3mf", mc_percent=0)
        self._run()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], "RUNNING")
        # Clean up the module-global hook list so we don't leak into siblings.
        with self.mod._state_change_hooks_lock:
            self.mod._state_change_hooks.clear()

    def test_buggy_hook_does_not_break_announcements(self):
        def boom(*_a):
            raise RuntimeError("hook exploded")
        self.mod.register_state_change_hook(boom)
        self.mod._last_gcode_state[0] = "IDLE"
        self._set(gcode_state="RUNNING", filename="cube.3mf", mc_percent=0,
                  mc_remaining=60)
        msgs = self._run()  # must still announce despite the hook raising
        self.assertTrue(any("Print started" in m for m in msgs))
        with self.mod._state_change_hooks_lock:
            self.mod._state_change_hooks.clear()


class BambuActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0

    def _set(self, **kw):
        with self.mod._state_lock:
            self.mod._state.update(kw)

    def test_check_print_no_fresh_status(self):
        out = self.actions["check_print"]("")
        self.assertIn("fresh status", out.lower())

    def test_check_print_idle(self):
        self._set(gcode_state="IDLE", last_update=time.time())
        self.assertIn("No active print", self.actions["check_print"](""))

    def test_check_print_running_reports_layer_and_eta(self):
        self._set(gcode_state="RUNNING", layer_num=47, total_layer=312,
                  mc_remaining=18, filename="widget.3mf", last_update=time.time())
        out = self.actions["check_print"]("")
        self.assertIn("widget", out)
        self.assertIn("layer 47 of 312", out)
        self.assertIn("18 minutes", out)

    def test_check_print_finish_and_pause(self):
        self._set(gcode_state="FINISH", filename="widget.3mf",
                  last_update=time.time())
        self.assertIn("finished", self.actions["check_print"]("").lower())
        self._set(gcode_state="PAUSE")
        self.assertIn("paused", self.actions["check_print"]("").lower())

    def test_how_is_the_print_includes_temps(self):
        self._set(gcode_state="RUNNING", layer_num=10, total_layer=100,
                  mc_remaining=30, filename="part.3mf", nozzle_temper=220.0,
                  bed_temper=60.0, last_update=time.time())
        out = self.actions["how_is_the_print"]("")
        self.assertIn("nozzle at 220 degrees", out)
        self.assertIn("bed at 60 degrees", out)
        self.assertIn("layer 10 of 100", out)

    def test_print_details_is_alias_of_how_is_the_print(self):
        self.assertIs(self.actions["print_details"],
                      self.actions["how_is_the_print"])


class BambuOfflineAndSummaryTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
        self.mod._mqtt_connected_ok[0] = False

    def test_offline_false_when_unconfigured(self):
        # _read_config returns ("","","") with bobert_companion mocked → not
        # "offline", just inactive.
        self.assertFalse(self.mod.is_printer_offline())

    def test_offline_true_when_configured_but_silent(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=("1.2.3.4", "code", "serial")):
            with self.mod._state_lock:
                self.mod._state["last_update"] = 0.0
            self.mod._mqtt_connected_ok[0] = False
            self.assertTrue(self.mod.is_printer_offline())

    def test_online_when_recent_push(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=("1.2.3.4", "code", "serial")):
            with self.mod._state_lock:
                self.mod._state["last_update"] = time.time()
            self.assertFalse(self.mod.is_printer_offline())

    def test_online_when_paho_session_up(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=("1.2.3.4", "code", "serial")):
            with self.mod._state_lock:
                self.mod._state["last_update"] = 0.0
            self.mod._mqtt_connected_ok[0] = True
            self.assertFalse(self.mod.is_printer_offline())

    def test_completion_summary_none_when_no_finish(self):
        self.mod._last_finish_announced_at[0] = 0.0
        self.assertIsNone(self.mod.get_last_print_completion_summary())

    def test_completion_summary_computes_delta(self):
        now = time.time()
        self.mod._last_finish_announced_at[0] = now
        self.mod._print_start_ts[0] = now - 3600           # ran 60 min
        self.mod._print_initial_estimate_min[0] = 90       # estimated 90 min
        with self.mod._state_lock:
            self.mod._state["gcode_state"] = "FINISH"
            self.mod._state["filename"] = "cube.3mf"
        summary = self.mod.get_last_print_completion_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["elapsed_minutes"], 60)
        self.assertEqual(summary["estimated_minutes"], 90)
        self.assertEqual(summary["delta_minutes"], 30)  # finished 30 min under
        self.assertEqual(summary["filename"], "cube")

    def test_completion_summary_drops_filename_when_state_moved_on(self):
        # finish_ts is fresh but gcode_state is no longer FINISH (a new print
        # or idle) → the filename can't be trusted, so it's blanked.
        now = time.time()
        self.mod._last_finish_announced_at[0] = now
        self.mod._print_start_ts[0] = 0.0
        self.mod._print_initial_estimate_min[0] = None
        with self.mod._state_lock:
            self.mod._state["gcode_state"] = "RUNNING"
            self.mod._state["filename"] = "next.3mf"
        summary = self.mod.get_last_print_completion_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["filename"], "")
        self.assertIsNone(summary["elapsed_minutes"])
        self.assertIsNone(summary["delta_minutes"])


class BambuReminderPersistenceTests(unittest.TestCase):
    """_load_reminder_persistence / _save_reminder_persistence round-trip
    against a tempfile (never the real data/bambu_reminder_state.json)."""

    def setUp(self):
        self.mod, self.actions = _load_bambu()
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "reminder.json")
        self._p = mock.patch.object(self.mod, "_REMINDER_STATE_FILE", self.path)
        self._p.start()
        self.addCleanup(self._p.stop)

        def _cleanup():
            import shutil
            shutil.rmtree(self.d, ignore_errors=True)
        self.addCleanup(_cleanup)

    def test_load_missing_returns_empty(self):
        self.assertEqual(self.mod._load_reminder_persistence(), {})

    def test_save_then_load_roundtrip(self):
        self.mod._save_reminder_persistence({"finish:cube": {"ts": 1.0}})
        self.assertEqual(self.mod._load_reminder_persistence(),
                         {"finish:cube": {"ts": 1.0}})

    def test_load_corrupt_returns_empty(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{ not json")
        self.assertEqual(self.mod._load_reminder_persistence(), {})

    def test_load_non_dict_returns_empty(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.mod._load_reminder_persistence(), {})

    def test_save_failure_is_logged(self):
        # os.replace raising inside the save → logged, not raised.
        with mock.patch("os.replace", side_effect=OSError("locked")):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._save_reminder_persistence({"k": {"ts": 1.0}})
        self.assertIn("reminder-state save failed", buf.getvalue())

    def test_save_failure_tmp_cleanup_also_fails(self):
        # os.replace raises AND the tmp-file cleanup os.remove also raises →
        # the inner `except: pass` swallows the cleanup error, the original
        # error re-raises, and the outer handler logs it.
        with mock.patch("os.replace", side_effect=OSError("locked")), \
             mock.patch("os.remove", side_effect=OSError("cleanup denied")):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._save_reminder_persistence({"k": {"ts": 1.0}})
        self.assertIn("reminder-state save failed", buf.getvalue())


class BambuEnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()

    def test_routes_through_proactive_announce(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(return_value=None)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.mod._enqueue_speech("hi sir")
        bc.proactive_announce.assert_called_once()
        self.assertEqual(bc.proactive_announce.call_args.kwargs.get("source"),
                         "bambu")

    def test_announce_raises_falls_through_to_file(self):
        # proactive_announce IS present but raises → the except swallows it and
        # we fall through to the atomic file write.
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(side_effect=RuntimeError("boom"))
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(p)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", p):
            self.mod._enqueue_speech("after announce error")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "after announce error")

    def test_fallback_appends_to_queue(self):
        # bobert_companion present but WITHOUT proactive_announce → atomic write
        # to the speech queue (redirected to a tempfile).
        bc = types.ModuleType("bobert_companion")  # no proactive_announce
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with open(p, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", p):
            self.mod._enqueue_speech("new sir")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new sir"])

    def test_fallback_corrupt_queue_resets(self):
        bc = types.ModuleType("bobert_companion")
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with open(p, "w", encoding="utf-8") as f:
            f.write("{ corrupt")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", p):
            self.mod._enqueue_speech("after corrupt")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["after corrupt"])

    def test_fallback_atomic_write_failure_is_logged(self):
        bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")), \
             mock.patch("os.path.exists", return_value=False):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._enqueue_speech("doomed")
        self.assertIn("speech-queue write failed", buf.getvalue())


class BambuCredentialsPromptTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()

    def test_prompt_skipped_when_flag_exists(self):
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._maybe_prompt_for_credentials()
        enq.assert_not_called()

    def test_prompt_fires_and_writes_flag(self):
        fd, p = tempfile.mkstemp(suffix=".flag")
        os.close(fd)
        os.remove(p)  # absent → prompt fires + flag is written
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with mock.patch.object(self.mod, "_CREDS_PROMPT_FLAG", p), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._maybe_prompt_for_credentials()
        enq.assert_called_once()
        self.assertIn("credentials aren't configured", enq.call_args.args[0])
        self.assertTrue(os.path.exists(p))

    def test_prompt_flag_write_failure_is_logged(self):
        with mock.patch.object(self.mod, "_CREDS_PROMPT_FLAG",
                               os.path.join("nonexistent_dir_zzz", "f.flag")), \
             mock.patch("os.path.exists", return_value=False), \
             mock.patch.object(self.mod, "_enqueue_speech"):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._maybe_prompt_for_credentials()
        self.assertIn("could not write creds-prompt flag", buf.getvalue())


class BambuReadConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()

    def test_read_config_strips_and_defaults(self):
        bc = types.ModuleType("bobert_companion")
        bc.BAMBU_PRINTER_IP = "  1.2.3.4  "
        bc.BAMBU_ACCESS_CODE = "code "
        bc.BAMBU_SERIAL = " SERIAL "
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertEqual(self.mod._read_config(),
                             ("1.2.3.4", "code", "SERIAL"))

    def test_read_config_exception_returns_blanks(self):
        # import_module raising → ("","","").
        with mock.patch("importlib.import_module",
                        side_effect=RuntimeError("no bc")):
            self.assertEqual(self.mod._read_config(), ("", "", ""))


class BambuRiskAmsExceptionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
            self.mod._chamber_history.clear()

    def test_unserializable_ams_is_swallowed(self):
        class _Unserializable:
            pass
        with self.mod._state_lock:
            self.mod._state.update(gcode_state="RUNNING", print_error=0,
                                   ams_status=_Unserializable())
        # json.dumps on the AMS object raises inside the except → ams_str "" →
        # no AMS fault classification, falls through to nominal.
        level, note = self.mod._compute_risk_level()
        self.assertEqual(level, 0)


class BambuWriteOverlayStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
            self.mod._chamber_history.clear()

    def test_writes_snapshot_json(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with self.mod._state_lock:
            self.mod._state.update(gcode_state="RUNNING", layer_num=10,
                                   total_layer=100, filename="part.3mf",
                                   last_update=time.time())
        with mock.patch.object(self.mod, "_OVERLAY_STATE_FILE", p):
            self.mod._write_overlay_state()
        with open(p, encoding="utf-8") as f:
            snap = json.load(f)
        self.assertEqual(snap["gcode_state"], "RUNNING")
        self.assertEqual(snap["layer_num"], 10)
        self.assertIn("risk_level", snap)

    def test_write_failure_is_logged(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk")):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._write_overlay_state()
        self.assertIn("overlay state write failed", buf.getvalue())


class BambuStartStopMonitorTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        self.mod._mqtt_client[0] = None
        self.mod._poll_thread[0] = None
        self.mod._mqtt_connected_ok[0] = False

    def test_start_monitor_unconfigured_returns_false(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=("", "", "")):
            self.assertFalse(self.mod.start_monitor())

    def test_start_monitor_no_mqtt_returns_false(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=("1.2.3.4", "code", "serial")), \
             mock.patch.object(self.mod, "_HAS_MQTT", False):
            self.assertFalse(self.mod.start_monitor())

    def test_start_monitor_start_mqtt_fails_returns_false(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=("1.2.3.4", "code", "serial")), \
             mock.patch.object(self.mod, "_HAS_MQTT", True), \
             mock.patch.object(self.mod, "_start_mqtt", return_value=None):
            self.assertFalse(self.mod.start_monitor())

    def test_start_monitor_success_spins_thread(self):
        fake_client = mock.MagicMock()
        with mock.patch.object(self.mod, "_read_config",
                               return_value=("1.2.3.4", "code", "serial")), \
             mock.patch.object(self.mod, "_HAS_MQTT", True), \
             mock.patch.object(self.mod, "_start_mqtt",
                               return_value=fake_client):
            # Thread.start is neutered by the harness, so the poll loop never
            # actually runs.
            ok = self.mod.start_monitor()
        self.assertTrue(ok)
        self.assertIs(self.mod._mqtt_client[0], fake_client)
        self.assertIsNotNone(self.mod._poll_thread[0])
        # Clean up module-global handles so we don't leak into siblings.
        self.mod._mqtt_client[0] = None
        self.mod._poll_thread[0] = None

    def test_start_monitor_tears_down_existing(self):
        # A live client/thread → start_monitor calls stop_monitor first.
        self.mod._mqtt_client[0] = mock.MagicMock()
        self.mod._poll_thread[0] = mock.MagicMock()
        with mock.patch.object(self.mod, "stop_monitor") as stop, \
             mock.patch.object(self.mod, "_read_config",
                               return_value=("", "", "")):
            self.mod.start_monitor()
        stop.assert_called_once()

    def test_stop_monitor_tears_down_client(self):
        client = mock.MagicMock()
        self.mod._mqtt_client[0] = client
        self.mod._poll_thread[0] = mock.MagicMock()
        with self.mod._state_lock:
            self.mod._state["gcode_state"] = "RUNNING"
        with mock.patch.object(self.mod, "_write_overlay_state"):
            self.mod.stop_monitor()
        client.loop_stop.assert_called_once()
        client.disconnect.assert_called_once()
        self.assertIsNone(self.mod._mqtt_client[0])
        # State reset to None / 0.0.
        with self.mod._state_lock:
            self.assertIsNone(self.mod._state["gcode_state"])
            self.assertEqual(self.mod._state["last_update"], 0.0)

    def test_stop_monitor_disconnect_failure_is_logged(self):
        client = mock.MagicMock()
        client.loop_stop.side_effect = RuntimeError("paho boom")
        self.mod._mqtt_client[0] = client
        with mock.patch.object(self.mod, "_write_overlay_state"):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod.stop_monitor()
        self.assertIn("disconnect on stop failed", buf.getvalue())


class BambuStartMqttTests(unittest.TestCase):
    """_start_mqtt with the paho `mqtt` module fully mocked — exercises client
    construction, the CallbackAPIVersion branch, the on_connect/on_disconnect
    callbacks, and the connect_async failure path. No real socket is opened."""

    def setUp(self):
        self.mod, self.actions = _load_bambu()
        self.mod._mqtt_connected_ok[0] = False

    def _fake_mqtt(self, *, with_cb_version=True):
        fake = types.SimpleNamespace()
        fake.MQTTv311 = 4
        fake.ssl = types.SimpleNamespace(CERT_NONE=0)
        client = mock.MagicMock()
        fake.Client = mock.MagicMock(return_value=client)
        if with_cb_version:
            fake.CallbackAPIVersion = types.SimpleNamespace(VERSION2="v2")
        return fake, client

    def test_no_mqtt_returns_none(self):
        with mock.patch.object(self.mod, "_HAS_MQTT", False):
            self.assertIsNone(self.mod._start_mqtt("1.2.3.4", "code", "serial"))

    def test_builds_client_and_starts_loop(self):
        fake, client = self._fake_mqtt(with_cb_version=True)
        with mock.patch.object(self.mod, "mqtt", fake):
            out = self.mod._start_mqtt("1.2.3.4", "code", "serial")
        self.assertIs(out, client)
        client.username_pw_set.assert_called_once_with("bblp", "code")
        client.connect_async.assert_called_once()
        client.loop_start.assert_called_once()
        # callback_api_version kwarg was passed because CallbackAPIVersion exists
        self.assertIn("callback_api_version", fake.Client.call_args.kwargs)

    def test_builds_client_without_callback_api_version(self):
        fake, client = self._fake_mqtt(with_cb_version=False)
        with mock.patch.object(self.mod, "mqtt", fake):
            self.mod._start_mqtt("1.2.3.4", "code", "serial")
        self.assertNotIn("callback_api_version", fake.Client.call_args.kwargs)

    def test_reconnect_delay_set_failure_is_swallowed(self):
        fake, client = self._fake_mqtt()
        client.reconnect_delay_set.side_effect = RuntimeError("old paho")
        with mock.patch.object(self.mod, "mqtt", fake):
            out = self.mod._start_mqtt("1.2.3.4", "code", "serial")
        # The except around reconnect_delay_set keeps the connect path alive.
        self.assertIs(out, client)

    def test_connect_async_failure_returns_none(self):
        fake, client = self._fake_mqtt()
        client.connect_async.side_effect = RuntimeError("bad host")
        with mock.patch.object(self.mod, "mqtt", fake):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                out = self.mod._start_mqtt("1.2.3.4", "code", "serial")
        self.assertIsNone(out)
        self.assertIn("could not schedule connect", buf.getvalue())

    def test_on_connect_subscribes_and_sets_flag(self):
        fake, client = self._fake_mqtt()
        with mock.patch.object(self.mod, "mqtt", fake):
            self.mod._start_mqtt("1.2.3.4", "code", "serial")
        on_connect = client.on_connect
        on_disconnect = client.on_disconnect
        # rc == 0 → connected flag set + subscribe.
        self.mod._mqtt_connected_ok[0] = False
        on_connect(client, None, None, 0)
        self.assertTrue(self.mod._mqtt_connected_ok[0])
        client.subscribe.assert_called_once_with("device/serial/report")
        # rc != 0 → flag stays False.
        on_connect(client, None, None, 5)
        self.assertFalse(self.mod._mqtt_connected_ok[0])
        # on_disconnect flips the flag back off.
        self.mod._mqtt_connected_ok[0] = True
        on_disconnect(client, None, 0)
        self.assertFalse(self.mod._mqtt_connected_ok[0])


class BambuHandleStateChangeBranchTests(BambuStateChangeTests):
    """Extra _handle_state_change branches not covered by the sibling suite:
    coercion excepts, RUNNING-from-terminal new-print without filename, FINISH
    without filename, the already-announced FINISH bed-watch arm, and the
    bed-cool already-recorded path. Inherits the sibling setUp (state reset +
    reminder persistence stubbed)."""

    def test_new_print_via_running_from_terminal_without_filename(self):
        # No filename, prev terminal state, now RUNNING → new_print via the
        # gcode transition branch (covers the elif), announces start.
        self.mod._last_gcode_state[0] = "FINISH"
        self.mod._current_print_filename[0] = None
        self._set(gcode_state="RUNNING", mc_percent=0, mc_remaining=60)
        msgs = self._run()
        self.assertTrue(any("Print started" in m for m in msgs))

    def test_bad_percent_is_swallowed(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self._set(gcode_state="RUNNING", filename="cube.3mf",
                  mc_percent="bad", mc_remaining=60)
        # Non-numeric percent → pct None → no milestone crash.
        self._run()  # must not raise

    def test_cold_start_bad_layer_is_swallowed(self):
        # prev None + RUNNING + non-numeric layer → the cold-start layer
        # coercion except is hit.
        self.mod._last_gcode_state[0] = None
        self._set(gcode_state="RUNNING", filename="big.3mf", mc_percent=60,
                  layer_num="??")
        self._run()  # must not raise
        self.assertTrue(self.mod._announced_start[0])

    def test_start_estimate_bad_remaining_is_swallowed(self):
        # Witnessed start with a non-numeric remaining → the estimate coercion
        # except is hit; start still announced (without ETA).
        self.mod._last_gcode_state[0] = "IDLE"
        self._set(gcode_state="RUNNING", filename="cube.3mf", mc_percent=0,
                  mc_remaining="soon")
        msgs = self._run()
        self.assertTrue(any(m == "Print started, sir." for m in msgs))

    def test_layer1_bad_layer_is_swallowed(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = "cube"
        self.mod._announced_start[0] = True
        self.mod._announced_layer1[0] = False
        self._set(gcode_state="RUNNING", filename="cube.3mf",
                  layer_num="five", mc_percent=1)
        self._run()  # must not raise
        self.assertFalse(self.mod._announced_layer1[0])

    def test_finish_without_filename_uses_generic_copy(self):
        self.mod._last_gcode_state[0] = "RUNNING"
        self.mod._current_print_filename[0] = None
        self.mod._announced_start[0] = True
        self._set(gcode_state="FINISH")  # no filename
        msgs = self._run()
        hit = [m for m in msgs if "Print complete" in m]
        self.assertTrue(hit)
        self.assertNotIn("'", hit[0])  # generic — no quoted filename

    def test_finish_already_announced_arms_bed_watch_quietly(self):
        # Durable state says this print's FINISH was already announced → no
        # re-announce, but the bed-watch is armed so the cool-down can still
        # fire (since bedcool wasn't recorded yet).
        self.mod._last_gcode_state[0] = "FINISH"
        self.mod._current_print_filename[0] = "cube"
        self._set(gcode_state="FINISH", filename="cube.3mf")
        with mock.patch.object(self.mod, "_load_reminder_persistence",
                               return_value={"finish:cube": {"ts": 1.0}}), \
             mock.patch.object(self.mod, "_save_reminder_persistence"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._handle_state_change()
        self.assertFalse(any("Print complete" in c.args[0]
                             for c in enq.call_args_list))
        self.assertTrue(self.mod._post_finish_bed_watch[0])

    def test_bed_cool_already_recorded_clears_flags(self):
        # Bed watch armed + bed cool, but BOTH the finish and bedcool keys
        # already exist in durable state → the FINISH block stays quiet and the
        # bed-cool block hits its already-recorded else branch (clears flags,
        # no re-announce).
        self.mod._post_finish_bed_watch[0] = True
        self.mod._bed_cool_announced[0] = False
        self.mod._last_gcode_state[0] = "FINISH"
        self.mod._current_print_filename[0] = "cube"
        self.mod._last_finish_announced_at[0] = time.time()  # block re-announce
        self._set(gcode_state="FINISH", filename="cube.3mf", bed_temper=30.0)
        durable = {"finish:cube": {"ts": 1.0}, "bedcool:cube": {"ts": 1.0}}
        with mock.patch.object(self.mod, "_load_reminder_persistence",
                               return_value=durable), \
             mock.patch.object(self.mod, "_save_reminder_persistence"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._handle_state_change()
        self.assertFalse(any("bed has cooled" in c.args[0].lower()
                             for c in enq.call_args_list))
        self.assertTrue(self.mod._bed_cool_announced[0])
        self.assertFalse(self.mod._post_finish_bed_watch[0])

    def test_bed_cool_bad_temp_is_swallowed(self):
        # Bed watch armed but bed_temper is non-numeric → the float() except
        # is hit without crashing.
        self.mod._post_finish_bed_watch[0] = True
        self.mod._bed_cool_announced[0] = False
        self.mod._last_gcode_state[0] = "FINISH"
        self.mod._current_print_filename[0] = "cube"
        self._set(gcode_state="FINISH", filename="cube.3mf", bed_temper="warm")
        self._run()  # must not raise
        # Still armed (no valid temp to clear it).
        self.assertTrue(self.mod._post_finish_bed_watch[0])


class BambuOnMessageChamberExceptionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
            self.mod._chamber_history.clear()

    def test_nonnumeric_chamber_temp_is_swallowed(self):
        m = mock.MagicMock()
        m.payload = json.dumps({"print": {"chamber_temper": "hot"}}).encode()
        with mock.patch.object(self.mod, "_handle_state_change"), \
             mock.patch.object(self.mod, "_write_overlay_state"):
            self.mod._on_message(None, None, m)  # float("hot") → except, no crash
        self.assertEqual(list(self.mod._chamber_history), [])


class BambuActionBranchTests(unittest.TestCase):
    """check_print / how_is_the_print branches the sibling suite skips:
    PAUSE/in-progress without filename, how_is_the_print no-fresh / idle /
    finish / pause / running-without-temps."""

    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0

    def _set(self, **kw):
        with self.mod._state_lock:
            self.mod._state.update(kw)

    def test_check_print_in_progress_without_filename(self):
        self._set(gcode_state="RUNNING", layer_num=5, total_layer=50,
                  mc_remaining=20, last_update=time.time())
        out = self.actions["check_print"]("")
        self.assertIn("Print in progress", out)
        self.assertIn("layer 5 of 50", out)

    def test_how_is_the_print_no_fresh_status(self):
        out = self.actions["how_is_the_print"]("")
        self.assertIn("fresh status", out.lower())

    def test_how_is_the_print_idle(self):
        self._set(gcode_state="IDLE", last_update=time.time())
        self.assertIn("No active print", self.actions["how_is_the_print"](""))

    def test_how_is_the_print_finish(self):
        self._set(gcode_state="FINISH", filename="cube.3mf",
                  last_update=time.time())
        self.assertIn("finished", self.actions["how_is_the_print"]("").lower())

    def test_how_is_the_print_pause_with_temps(self):
        self._set(gcode_state="PAUSE", nozzle_temper=210.0, bed_temper=55.0,
                  last_update=time.time())
        out = self.actions["how_is_the_print"]("")
        self.assertIn("paused", out.lower())
        self.assertIn("nozzle at 210 degrees", out)

    def test_how_is_the_print_running_without_temps(self):
        # No nozzle/bed → the temp tail is empty; still reports layer/eta.
        self._set(gcode_state="RUNNING", layer_num=10, total_layer=100,
                  mc_remaining=30, last_update=time.time())
        out = self.actions["how_is_the_print"]("")
        self.assertIn("Print in progress", out)
        self.assertIn("layer 10 of 100", out)
        self.assertNotIn("nozzle", out)


class BambuRegisterPromptTests(unittest.TestCase):
    def test_register_calls_prompt_when_ip_missing(self):
        # With no BAMBU_PRINTER_IP, register() takes the `if not ip:` branch and
        # calls _maybe_prompt_for_credentials(). os.path.exists→True makes the
        # prompt's own flag check short-circuit, so no real flag file is written
        # while still covering register()'s call site + start_monitor's
        # unconfigured early-out.
        fake_bc = mock.MagicMock()
        fake_bc.BAMBU_PRINTER_IP = ""
        fake_bc.BAMBU_ACCESS_CODE = ""
        fake_bc.BAMBU_SERIAL = ""
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}), \
             mock.patch("os.path.exists", return_value=True):
            mod, actions = load_skill_isolated(
                "bambu_monitor", utils=make_fake_skill_utils())
        self.assertIn("check_print", actions)
        self.assertIn("how_is_the_print", actions)


if __name__ == "__main__":
    unittest.main()
