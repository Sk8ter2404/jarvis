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
import sys
import time
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


if __name__ == "__main__":
    unittest.main()
