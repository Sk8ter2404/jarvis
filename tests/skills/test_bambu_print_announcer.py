"""Logic tests for skills/bambu_print_announcer.py.

This skill rides on bambu_monitor: it adds the 10%/95% milestones, early-layer
adhesion checkpoints, the gated+rate-limited proactive_print_announcer layer
(runout / AMS-fault / celebratory completion), and pause/resume MQTT commands.

We drive it with a *fake* skill_bambu_monitor injected into sys.modules so we
never touch a real printer or its state, and patch _enqueue_speech so nothing
reaches pending_speech.json. _proactive_announce's focus-mode gate and 10-min
rate-limit are exercised directly.
"""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_bambu(state=None, *, client=None, connected=True):
    """A stand-in skill_bambu_monitor exposing the attributes the announcer
    reaches for: _state, _state_lock, _mqtt_client, _mqtt_connected_ok,
    _read_config, _format_minutes, _enqueue_speech."""
    m = types.ModuleType("skill_bambu_monitor")
    m._state_lock = threading.Lock()
    m._state = state or {"last_update": 0.0}
    m._mqtt_client = [client]
    m._mqtt_connected_ok = [connected]
    m._read_config = lambda: ("1.2.3.4", "code", "SERIAL123")
    m._enqueue_speech = mock.MagicMock()

    def _fmt(minutes):
        try:
            mi = int(minutes)
        except (TypeError, ValueError):
            return ""
        if mi <= 0:
            return ""
        if mi < 60:
            return f"{mi} minutes"
        h, r = divmod(mi, 60)
        return f"{h} hours and {r} minutes" if r else f"{h} hours"
    m._format_minutes = _fmt
    return m


class AnnouncerLoadMixin:
    def _load(self, **bambu_kw):
        fake = _fake_bambu(**bambu_kw)
        self._patch = mock.patch.dict(sys.modules,
                                      {"skill_bambu_monitor": fake})
        self._patch.start()
        self.addCleanup(self._patch.stop)
        mod, actions = load_skill_isolated("bambu_print_announcer")
        # Reset per-print + rate-limit bookkeeping for determinism.
        mod._announced_pct.clear()
        mod._announced_layers.clear()
        mod._current_filename[0] = None
        mod._armed_for_new_print[0] = False
        mod._saw_running_this_print[0] = False
        mod._announced_runout[0] = False
        mod._announced_ams_fault[0] = False
        mod._announced_completion[0] = False
        mod._last_announcement_at[0] = 0.0
        mod._last_suppressed_reason[0] = ""
        return mod, actions, fake


class AnnouncerProactiveGateTests(AnnouncerLoadMixin, unittest.TestCase):
    def test_proactive_announce_fires_when_clear(self):
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_is_focus_active", return_value=False), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            ok = mod._proactive_announce("hello sir")
        self.assertTrue(ok)
        enq.assert_called_once_with("hello sir")

    def test_proactive_announce_suppressed_by_focus_mode(self):
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_is_focus_active", return_value=True), \
             mock.patch.object(mod, "_enqueue_speech") as enq:
            ok = mod._proactive_announce("hello sir")
        self.assertFalse(ok)
        enq.assert_not_called()
        self.assertEqual(mod._last_suppressed_reason[0], "focus mode")

    def test_proactive_announce_rate_limited(self):
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_is_focus_active", return_value=False), \
             mock.patch.object(mod, "_enqueue_speech"):
            self.assertTrue(mod._proactive_announce("first"))
            # Immediately after, the 10-min throttle blocks the second.
            self.assertFalse(mod._proactive_announce("second"))
        self.assertIn("rate-limited", mod._last_suppressed_reason[0])


class AnnouncerMilestoneTests(AnnouncerLoadMixin, unittest.TestCase):
    def test_extra_milestone_10pct_fires_once(self):
        mod, _a, _f = self._load()
        # Prime as a known print already armed (so no mid-flight suppression).
        mod._armed_for_new_print[0] = True
        mod._current_filename[0] = "cube.3mf"
        state = {"last_update": time.time(), "gcode_state": "RUNNING",
                 "filename": "cube.3mf", "mc_percent": 12, "layer_num": 30,
                 "total_layer": 300, "mc_remaining": 100}
        with mock.patch.object(mod, "_read_state", return_value=state), \
             mock.patch.object(mod, "_proactive_announce",
                               return_value=True) as ann:
            mod._check_milestones()
        msgs = [c.args[0] for c in ann.call_args_list]
        self.assertTrue(any("10%" in m for m in msgs))
        self.assertIn(10, mod._announced_pct)

    def test_midflight_discovery_suppresses_past_milestones(self):
        mod, _a, _f = self._load()
        # Cold-boot path: the filename is already known (no filename-change
        # reset) but we were never armed, and the printer is RUNNING past both
        # thresholds. The priming poll marks 10 + 95 as announced and returns
        # WITHOUT speaking, so JARVIS doesn't blurt 10% on an 95%-done print.
        mod._current_filename[0] = "big.3mf"
        mod._armed_for_new_print[0] = False
        state = {"last_update": time.time(), "gcode_state": "RUNNING",
                 "filename": "big.3mf", "mc_percent": 95, "layer_num": 290,
                 "total_layer": 300, "mc_remaining": 5}
        with mock.patch.object(mod, "_read_state", return_value=state), \
             mock.patch.object(mod, "_proactive_announce") as ann:
            mod._check_milestones()
        ann.assert_not_called()
        self.assertIn(10, mod._announced_pct)
        self.assertIn(95, mod._announced_pct)
        self.assertTrue(mod._armed_for_new_print[0])

    def test_early_layer_checkpoint_fires(self):
        mod, _a, _f = self._load()
        mod._armed_for_new_print[0] = True
        mod._current_filename[0] = "cube.3mf"
        mod._announced_pct.add(10)  # don't let the % milestone steal the turn
        state = {"last_update": time.time(), "gcode_state": "RUNNING",
                 "filename": "cube.3mf", "mc_percent": 12, "layer_num": 5,
                 "total_layer": 200, "mc_remaining": 100}
        with mock.patch.object(mod, "_read_state", return_value=state), \
             mock.patch.object(mod, "_proactive_announce",
                               return_value=True) as ann:
            mod._check_milestones()
        msgs = [c.args[0] for c in ann.call_args_list]
        self.assertTrue(any("first layer adhesion" in m.lower() for m in msgs))


class AnnouncerRunoutAmsTests(AnnouncerLoadMixin, unittest.TestCase):
    def test_runout_by_error_prefix(self):
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_proactive_announce",
                               return_value=True) as ann:
            mod._check_runout_and_ams("0300_0300_0001_0001", None, "PAUSE")
        msgs = [c.args[0] for c in ann.call_args_list]
        self.assertTrue(any("run out" in m.lower() for m in msgs))
        self.assertTrue(mod._announced_runout[0])
        # Runout also covers the AMS-fault announcement to avoid double-speak.
        self.assertTrue(mod._announced_ams_fault[0])

    def test_runout_by_ams_keyword(self):
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_proactive_announce",
                               return_value=True) as ann:
            mod._check_runout_and_ams(0, "tray empty, no filament", "RUNNING")
        self.assertTrue(any("run out" in c.args[0].lower()
                            for c in ann.call_args_list))

    def test_ams_fault_without_runout(self):
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_proactive_announce",
                               return_value=True) as ann:
            mod._check_runout_and_ams(0, "ams tray jam detected", "RUNNING")
        msgs = [c.args[0] for c in ann.call_args_list]
        self.assertTrue(any("AMS is reporting a fault" in m for m in msgs))
        self.assertTrue(mod._announced_ams_fault[0])
        self.assertFalse(mod._announced_runout[0])

    def test_no_announcement_when_clean(self):
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_proactive_announce") as ann:
            mod._check_runout_and_ams(0, "tray 1 ready", "RUNNING")
        ann.assert_not_called()


class AnnouncerCompletionTests(AnnouncerLoadMixin, unittest.TestCase):
    def test_celebratory_completion_only_after_running(self):
        mod, _a, _f = self._load()
        mod._saw_running_this_print[0] = True
        with mock.patch.object(mod, "_proactive_announce",
                               return_value=True) as ann:
            mod._check_celebratory_completion("FINISH")
        self.assertTrue(any("part is ready" in c.args[0].lower()
                            for c in ann.call_args_list))
        self.assertTrue(mod._announced_completion[0])

    def test_no_celebration_for_unwitnessed_finish(self):
        mod, _a, _f = self._load()
        mod._saw_running_this_print[0] = False  # never saw it run
        with mock.patch.object(mod, "_proactive_announce") as ann:
            mod._check_celebratory_completion("FINISH")
        ann.assert_not_called()


class AnnouncerCommandTests(AnnouncerLoadMixin, unittest.TestCase):
    def test_send_command_no_bambu_module(self):
        # Force the announcer to see no bambu monitor at all.
        mod, _a, _f = self._load()
        with mock.patch.object(mod, "_get_bambu_module", return_value=None):
            ok, err = mod._send_print_command("pause")
        self.assertFalse(ok)
        self.assertIn("not loaded", err)

    def test_send_command_no_client(self):
        mod, _a, _f = self._load(client=None)
        ok, err = mod._send_print_command("pause")
        self.assertFalse(ok)
        self.assertIn("no MQTT client", err)

    def test_send_command_not_connected(self):
        client = mock.MagicMock()
        mod, _a, _f = self._load(client=client, connected=False)
        ok, err = mod._send_print_command("pause")
        self.assertFalse(ok)
        self.assertIn("not connected", err)

    def test_send_command_success_publishes(self):
        client = mock.MagicMock()
        client.publish.return_value = mock.MagicMock(rc=0)
        mod, _a, _f = self._load(client=client, connected=True)
        ok, err = mod._send_print_command("resume")
        self.assertTrue(ok)
        self.assertEqual(err, "")
        # The published topic must target the configured serial.
        topic = client.publish.call_args.args[0]
        self.assertEqual(topic, "device/SERIAL123/request")

    def test_send_command_publish_rc_nonzero_is_failure(self):
        client = mock.MagicMock()
        client.publish.return_value = mock.MagicMock(rc=4)  # MQTT_ERR_NO_CONN
        mod, _a, _f = self._load(client=client, connected=True)
        ok, err = mod._send_print_command("pause")
        self.assertFalse(ok)
        self.assertIn("not connected", err)


class AnnouncerActionTests(AnnouncerLoadMixin, unittest.TestCase):
    def _running_state(self):
        return {"last_update": time.time(), "gcode_state": "RUNNING"}

    def test_pause_when_running(self):
        client = mock.MagicMock()
        client.publish.return_value = mock.MagicMock(rc=0)
        mod, actions, _f = self._load(client=client, connected=True)
        with mock.patch.object(mod, "_read_state",
                               return_value=self._running_state()):
            out = actions["pause_print"]("")
        self.assertIn("Pausing", out)

    def test_pause_when_already_paused(self):
        mod, actions, _f = self._load()
        with mock.patch.object(mod, "_read_state",
                               return_value={"gcode_state": "PAUSE"}):
            out = actions["pause_print"]("")
        self.assertIn("already paused", out.lower())

    def test_pause_when_no_active_print(self):
        mod, actions, _f = self._load()
        with mock.patch.object(mod, "_read_state",
                               return_value={"gcode_state": "IDLE"}):
            out = actions["pause_print"]("")
        self.assertIn("no active print", out.lower())

    def test_resume_when_paused(self):
        client = mock.MagicMock()
        client.publish.return_value = mock.MagicMock(rc=0)
        mod, actions, _f = self._load(client=client, connected=True)
        with mock.patch.object(mod, "_read_state",
                               return_value={"gcode_state": "PAUSE"}):
            out = actions["resume_print"]("")
        self.assertIn("Resuming", out)

    def test_resume_when_not_paused(self):
        mod, actions, _f = self._load()
        with mock.patch.object(mod, "_read_state",
                               return_value={"gcode_state": "RUNNING"}):
            out = actions["resume_print"]("")
        self.assertIn("already running", out.lower())

    def test_pause_reports_unreachable_printer(self):
        # Running state but the command can't be sent (no client) → graceful.
        mod, actions, _f = self._load(client=None)
        with mock.patch.object(mod, "_read_state",
                               return_value=self._running_state()):
            out = actions["pause_print"]("")
        self.assertIn("couldn't reach the printer", out.lower())

    def test_announcer_status_focus_mode(self):
        mod, actions, _f = self._load()
        with mock.patch.object(mod, "_is_focus_active", return_value=True):
            out = actions["proactive_announcer_status"]("")
        self.assertIn("focus mode", out.lower())

    def test_announcer_status_rate_limited(self):
        mod, actions, _f = self._load()
        mod._last_announcement_at[0] = time.time()  # just spoke
        with mock.patch.object(mod, "_is_focus_active", return_value=False):
            out = actions["proactive_announcer_status"]("")
        self.assertIn("rate-limited", out.lower())

    def test_announcer_status_armed(self):
        mod, actions, _f = self._load()
        mod._last_announcement_at[0] = 0.0
        with mock.patch.object(mod, "_is_focus_active", return_value=False):
            out = actions["proactive_announcer_status"]("")
        self.assertIn("armed", out.lower())


if __name__ == "__main__":
    unittest.main()
