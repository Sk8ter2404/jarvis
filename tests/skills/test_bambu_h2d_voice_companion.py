"""Logic tests for skills/bambu_h2d_voice_companion.py.

In-character H2D announcements layered on bambu_monitor's state-change hook.
We test:

  • formatter fallbacks (_format_minutes / _format_temp / _strip_filename) that
    work even when bambu_monitor isn't loaded
  • _scan_for_layer_shift / _scan_for_ams_issue keyword detectors
  • _process_snapshot — milestone (gated) vs layer-shift / AMS / FAILED
    (direct, rate-limit-bypassing) announcements + per-print dedup
  • the print_status action: offline / no-fresh-state / idle / finish / running
  • _register_bambu_hook graceful degradation

All speech routing is patched (_gated_announce / _direct_enqueue) so nothing
hits pending_speech.json, and bambu_monitor is represented by a fake module so
no real printer state is read.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_bambu(state=None):
    m = types.ModuleType("skill_bambu_monitor")
    m._state_lock = threading.Lock()
    m._state = state if state is not None else {"last_update": 0.0}
    m.register_state_change_hook = mock.MagicMock()
    return m


class VoiceCompanionMixin:
    def _load(self, bambu_state="__absent__", bambu_module=None):
        patches = []
        if bambu_module is not None:
            # Caller supplied a fully-built fake bambu module (used when the
            # test needs custom helper functions on it).
            p = mock.patch.dict(sys.modules,
                                {"skill_bambu_monitor": bambu_module})
            p.start()
            patches.append(p)
            self._fake = bambu_module
        elif bambu_state != "__absent__":
            fake = _fake_bambu(bambu_state)
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake})
            p.start()
            patches.append(p)
            self._fake = fake
        else:
            # Ensure no stale bambu module is visible.
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": None})
            p.start()
            patches.append(p)
            self._fake = None
        for p in patches:
            self.addCleanup(p.stop)
        mod, actions = load_skill_isolated("bambu_h2d_voice_companion")
        # Reset per-print bookkeeping.
        mod._current_filename[0] = None
        mod._announced_milestones.clear()
        mod._announced_error_codes.clear()
        mod._announced_layer_shift[0] = False
        mod._announced_ams_error[0] = False
        mod._announced_failed[0] = False
        return mod, actions


class VoiceCompanionFormatTests(VoiceCompanionMixin, unittest.TestCase):
    def test_format_minutes_fallback_without_bambu(self):
        mod, _a = self._load()  # bambu absent → local fallback path
        self.assertEqual(mod._format_minutes(5), "5 minutes")
        self.assertEqual(mod._format_minutes(125), "2 hours and 5 minutes")
        self.assertEqual(mod._format_minutes(0), "")
        self.assertEqual(mod._format_minutes(None), "")

    def test_format_temp_fallback(self):
        mod, _a = self._load()
        self.assertEqual(mod._format_temp(219.6), "220 degrees")
        self.assertEqual(mod._format_temp(0), "")

    def test_strip_filename_fallback(self):
        mod, _a = self._load()
        self.assertEqual(mod._strip_filename("My_Part.gcode"), "My Part")
        self.assertEqual(mod._strip_filename(""), "")


class VoiceCompanionScanTests(VoiceCompanionMixin, unittest.TestCase):
    def test_layer_shift_detected_in_error(self):
        mod, _a = self._load()
        self.assertTrue(mod._scan_for_layer_shift("layer shift on plate 1", None))
        self.assertTrue(mod._scan_for_layer_shift(None, "axis shifted"))

    def test_layer_shift_absent(self):
        mod, _a = self._load()
        self.assertFalse(mod._scan_for_layer_shift(0, "all nominal"))

    def test_ams_issue_requires_fault_signature(self):
        mod, _a = self._load()
        # The literal word "ams" in a healthy block must NOT trip.
        self.assertFalse(mod._scan_for_ams_issue(0, "ams tray 1 ready"))
        # A fault keyword alongside an AMS keyword does.
        self.assertTrue(mod._scan_for_ams_issue(0, "ams spool jam"))

    def test_ams_issue_none_payload(self):
        mod, _a = self._load()
        self.assertFalse(mod._scan_for_ams_issue("0", None))


class VoiceCompanionSnapshotTests(VoiceCompanionMixin, unittest.TestCase):
    def test_milestone_routed_through_gated_announce(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        # 25 already spoken (one milestone per snapshot); crossing 50 announces
        # the 50 line and routes it through the gated path, not direct.
        mod._announced_milestones.add(25)
        snap = {"filename": "cube.3mf", "mc_percent": 50, "mc_remaining": 120}
        with mock.patch.object(mod, "_gated_announce") as gated, \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        msgs = [c.args[0] for c in gated.call_args_list]
        self.assertTrue(any("Print at 50%" in m for m in msgs))
        self.assertIn("2 hours", msgs[0])
        direct.assert_not_called()
        self.assertIn(50, mod._announced_milestones)

    def test_milestone_deduped(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "mc_percent": 25, "mc_remaining": 60}
        with mock.patch.object(mod, "_gated_announce") as gated:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
                mod._process_snapshot(snap, "RUNNING")
        self.assertEqual(sum("25%" in c.args[0] for c in gated.call_args_list), 1)

    def test_layer_shift_uses_direct_enqueue(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "print_error": "layer shift detected",
                "layer_num": 88}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        msgs = [c.args[0] for c in direct.call_args_list]
        self.assertTrue(any("Layer shift detected" in m for m in msgs))
        self.assertTrue(any("layer 88" in m for m in msgs))
        self.assertTrue(mod._announced_layer_shift[0])

    def test_ams_unwell_direct_enqueue(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "ams_status": "ams spool jam"}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        self.assertTrue(any("AMS appears to be unwell" in c.args[0]
                            for c in direct.call_args_list))

    def test_failed_state_direct_enqueue(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "layer_num": 142}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "FAILED")
        hit = [c.args[0] for c in direct.call_args_list
               if "failed" in c.args[0].lower()]
        self.assertTrue(hit)
        self.assertIn("layer 142", hit[0])
        self.assertTrue(mod._announced_failed[0])

    def test_new_filename_resets_bookkeeping(self):
        mod, _a = self._load()
        mod._current_filename[0] = "old"
        mod._announced_milestones.add(25)
        mod._announced_layer_shift[0] = True
        snap = {"filename": "new_part.3mf", "mc_percent": 5}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue"):
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        # Reset on the filename change.
        self.assertNotIn(25, mod._announced_milestones)
        self.assertFalse(mod._announced_layer_shift[0])


class VoiceCompanionStatusActionTests(VoiceCompanionMixin, unittest.TestCase):
    def test_status_offline_when_bambu_absent(self):
        mod, actions = self._load()  # no bambu module
        out = actions["print_status"]("")
        self.assertIn("monitor isn't running", out.lower())

    def test_status_no_fresh_state(self):
        mod, actions = self._load(bambu_state={"last_update": 0.0})
        out = actions["print_status"]("")
        self.assertIn("fresh status", out.lower())

    def test_status_idle(self):
        mod, actions = self._load(
            bambu_state={"last_update": time.time(), "gcode_state": "IDLE"})
        self.assertIn("No active print", actions["print_status"](""))

    def test_status_finish(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "FINISH",
            "filename": "cube.3mf"})
        self.assertIn("finished", actions["print_status"]("").lower())

    def test_status_running_with_temps(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "RUNNING",
            "filename": "widget.3mf", "layer_num": 10, "total_layer": 100,
            "mc_remaining": 30, "nozzle_temper": 220.0, "bed_temper": 60.0})
        out = actions["print_status"]("")
        self.assertIn("widget", out)
        self.assertIn("layer 10 of 100", out)
        self.assertIn("nozzle at 220 degrees", out)
        self.assertIn("bed at 60 degrees", out)


class VoiceCompanionHookTests(VoiceCompanionMixin, unittest.TestCase):
    def test_register_hook_when_bambu_present(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        # The fake exposes register_state_change_hook as a MagicMock.
        ok = mod._register_bambu_hook()
        self.assertTrue(ok)
        self.assertTrue(mod._hook_registered[0])
        self._fake.register_state_change_hook.assert_called()

    def test_register_hook_when_bambu_absent(self):
        mod, _a = self._load()  # no bambu
        self.assertFalse(mod._register_bambu_hook())

    def test_hook_callback_swallows_errors(self):
        mod, _a = self._load(bambu_state={"last_update": time.time()})
        # If _process_snapshot raises, the hook must not propagate it.
        with mock.patch.object(mod, "_process_snapshot",
                               side_effect=RuntimeError("boom")):
            # Should not raise.
            mod._on_bambu_state_change({"filename": "x"}, "IDLE", "RUNNING")

    def test_register_hook_when_not_callable(self):
        # bambu present but register_state_change_hook is missing/non-callable.
        fake = types.ModuleType("skill_bambu_monitor")
        fake.register_state_change_hook = "not callable"
        mod, _a = self._load(bambu_module=fake)
        self.assertFalse(mod._register_bambu_hook())

    def test_register_hook_raises_is_caught(self):
        fake = _fake_bambu({"last_update": 0.0})
        fake.register_state_change_hook = mock.MagicMock(
            side_effect=RuntimeError("registry full"))
        mod, _a = self._load(bambu_module=fake)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertFalse(mod._register_bambu_hook())
        self.assertIn("hook registration failed", buf.getvalue())


class VoiceCompanionDelegationTests(VoiceCompanionMixin, unittest.TestCase):
    """The formatter helpers prefer bambu_monitor's implementation when it's
    loaded; the sibling tests only cover the bambu-absent local fallback. Here
    we exercise the delegation path (success + the delegate-raises fallback)."""

    def _bambu_with(self, **fns):
        m = types.ModuleType("skill_bambu_monitor")
        m._state_lock = threading.Lock()
        m._state = {"last_update": 0.0}
        for name, fn in fns.items():
            setattr(m, name, fn)
        return m

    def test_format_minutes_delegates_to_bambu(self):
        bm = self._bambu_with(_format_minutes=lambda mins: f"DELEGATED {mins}")
        mod, _a = self._load(bambu_module=bm)
        self.assertEqual(mod._format_minutes(7), "DELEGATED 7")

    def test_format_minutes_delegate_raises_falls_back(self):
        def _boom(_m):
            raise RuntimeError("bm broken")
        bm = self._bambu_with(_format_minutes=_boom)
        mod, _a = self._load(bambu_module=bm)
        # Falls through to the local formatter.
        self.assertEqual(mod._format_minutes(5), "5 minutes")

    def test_format_temp_delegates_and_falls_back(self):
        bm = self._bambu_with(_format_temp=lambda t: f"T{t}")
        mod, _a = self._load(bambu_module=bm)
        self.assertEqual(mod._format_temp(60), "T60")

        def _boom(_t):
            raise RuntimeError("x")
        bm2 = self._bambu_with(_format_temp=_boom)
        mod2, _a2 = self._load(bambu_module=bm2)
        self.assertEqual(mod2._format_temp(60), "60 degrees")

    def test_format_temp_local_nonnumeric_returns_blank(self):
        mod, _a = self._load()  # bambu absent → local path
        self.assertEqual(mod._format_temp("hot"), "")

    def test_strip_filename_delegates_and_falls_back(self):
        bm = self._bambu_with(_strip_filename=lambda n: f"S:{n}")
        mod, _a = self._load(bambu_module=bm)
        self.assertEqual(mod._strip_filename("part.3mf"), "S:part.3mf")

        def _boom(_n):
            raise RuntimeError("x")
        bm2 = self._bambu_with(_strip_filename=_boom)
        mod2, _a2 = self._load(bambu_module=bm2)
        self.assertEqual(mod2._strip_filename("My_Part.gcode"), "My Part")

    def test_get_announcer_module_resolves(self):
        mod, _a = self._load()
        ann = types.ModuleType("skill_bambu_print_announcer")
        with mock.patch.dict(sys.modules,
                             {"skill_bambu_print_announcer": ann}):
            self.assertIs(mod._get_announcer_module(), ann)
        with mock.patch.dict(sys.modules,
                             {"skill_bambu_print_announcer": None}):
            self.assertIsNone(mod._get_announcer_module())

    def test_read_state_exception_returns_none(self):
        # A _state_lock that isn't a context manager makes the `with` raise.
        fake = types.ModuleType("skill_bambu_monitor")
        fake._state_lock = object()
        fake._state = {"last_update": 1.0}
        mod, _a = self._load(bambu_module=fake)
        self.assertIsNone(mod._read_state())


class VoiceCompanionAnnounceRoutingTests(VoiceCompanionMixin, unittest.TestCase):
    def test_gated_announce_routes_through_announcer(self):
        mod, _a = self._load()
        ann = types.ModuleType("skill_bambu_print_announcer")
        ann._proactive_announce = mock.MagicMock(return_value=True)
        with mock.patch.dict(sys.modules,
                             {"skill_bambu_print_announcer": ann}):
            mod._gated_announce("milestone sir")
        ann._proactive_announce.assert_called_once_with("milestone sir")

    def test_gated_announce_falls_back_when_announcer_raises(self):
        mod, _a = self._load()
        ann = types.ModuleType("skill_bambu_print_announcer")
        ann._proactive_announce = mock.MagicMock(side_effect=RuntimeError("x"))
        with mock.patch.dict(sys.modules,
                             {"skill_bambu_print_announcer": ann}), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            mod._gated_announce("milestone sir")
        direct.assert_called_once_with("milestone sir")

    def test_gated_announce_falls_back_when_announcer_absent(self):
        mod, _a = self._load()
        with mock.patch.dict(sys.modules,
                             {"skill_bambu_print_announcer": None}), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            mod._gated_announce("hello")
        direct.assert_called_once_with("hello")

    def test_direct_enqueue_prefers_bobert_companion(self):
        # Inject a fake bobert_companion into sys.modules so import_module
        # returns it (never the real monolith file at the project root).
        mod, _a = self._load()
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(return_value=True)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            mod._direct_enqueue("alert sir")
        bc.proactive_announce.assert_called_once()
        self.assertEqual(bc.proactive_announce.call_args.kwargs.get("source"),
                         "bambu_voice_companion")

    def test_direct_enqueue_bobert_announce_raises_falls_through(self):
        # bobert_companion present and proactive_announce IS callable but
        # raises → the except swallows it and we fall through to the bambu
        # enqueue path. (Bare fake module → never imports the real monolith.)
        bm = _fake_bambu({"last_update": 0.0})
        bm._enqueue_speech = mock.MagicMock()
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(
            side_effect=RuntimeError("announce boom"))
        mod, _a = self._load(bambu_module=bm)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            mod._direct_enqueue("alert sir")
        bm._enqueue_speech.assert_called_once_with("alert sir")

    def test_direct_enqueue_bambu_enqueue_raises_falls_to_file(self):
        # bobert absent (bare module) AND bambu's _enqueue_speech raises →
        # the last-resort file write runs. Queue file pre-exists with valid
        # JSON so the existing-file read branch is exercised too.
        import json
        import tempfile
        bm = _fake_bambu({"last_update": 0.0})
        bm._enqueue_speech = mock.MagicMock(side_effect=RuntimeError("nope"))
        bc = types.ModuleType("bobert_companion")  # no proactive_announce
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        with open(p, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        mod, _a = self._load(bambu_module=bm)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(mod, "_SPEECH_QUEUE", p):
            mod._direct_enqueue("appended sir")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "appended sir"])

    def test_direct_enqueue_corrupt_and_nonlist_queue(self):
        # Existing queue file that's non-JSON → read except (data=[]), then a
        # second call against a queue holding a non-list JSON → the
        # isinstance guard resets it. Both land the message.
        import json
        import tempfile
        mod, _a = self._load()  # bambu absent
        bc = types.ModuleType("bobert_companion")  # no proactive_announce
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        # (a) corrupt content → read except.
        with open(p, "w", encoding="utf-8") as f:
            f.write("{ not json")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(mod, "_SPEECH_QUEUE", p):
            mod._direct_enqueue("after corrupt")
        # (b) non-list JSON → isinstance guard resets to [].
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(mod, "_SPEECH_QUEUE", p):
            mod._direct_enqueue("after nonlist")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["after nonlist"])

    def test_direct_enqueue_file_write_failure_is_logged(self):
        mod, _a = self._load()
        bc = types.ModuleType("bobert_companion")  # no proactive_announce
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(mod, "_atomic_write_json",
                               side_effect=OSError("disk full")), \
             mock.patch.object(mod.os.path, "exists", return_value=False):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                mod._direct_enqueue("doomed")
        self.assertIn("speech-queue write failed", buf.getvalue())


class VoiceCompanionScanExceptionTests(VoiceCompanionMixin, unittest.TestCase):
    def test_layer_shift_json_dumps_failure_is_swallowed(self):
        mod, _a = self._load()

        class _Unserializable:
            pass
        # json.dumps on a non-str, non-serializable ams → except → no crash.
        self.assertFalse(
            mod._scan_for_layer_shift(0, _Unserializable()))

    def test_ams_issue_json_dumps_failure_returns_false(self):
        mod, _a = self._load()

        class _Unserializable:
            pass
        self.assertFalse(mod._scan_for_ams_issue(0, _Unserializable()))

    def test_ams_issue_no_ams_keyword_returns_false(self):
        mod, _a = self._load()
        # Blob has a fault word but no AMS/spool/filament keyword → early False.
        self.assertFalse(mod._scan_for_ams_issue(0, "nozzle jam"))


class VoiceCompanionSnapshotBranchTests(VoiceCompanionMixin, unittest.TestCase):
    def test_milestone_without_eta(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        # No mc_remaining → the "Print at X%, sir." (no ETA tail) branch.
        snap = {"filename": "cube.3mf", "mc_percent": 25}
        with mock.patch.object(mod, "_gated_announce") as gated:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        msgs = [c.args[0] for c in gated.call_args_list]
        self.assertTrue(any(m == "Print at 25%, sir." for m in msgs))

    def test_pct_coercion_failure_skips_milestones(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        # Non-numeric mc_percent → pct_f stays None → no milestone announced,
        # but the rest of the snapshot pass still runs without crashing.
        snap = {"filename": "cube.3mf", "mc_percent": "almost"}
        with mock.patch.object(mod, "_gated_announce") as gated, \
             mock.patch.object(mod, "_direct_enqueue"):
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        gated.assert_not_called()


class VoiceCompanionStatusBranchTests(VoiceCompanionMixin, unittest.TestCase):
    def test_status_pause_with_temps(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "PAUSE",
            "nozzle_temper": 215.0, "bed_temper": 55.0})
        out = actions["print_status"]("")
        self.assertIn("paused", out.lower())
        self.assertIn("nozzle at 215 degrees", out)

    def test_status_running_without_filename(self):
        # No filename → the "Print in progress" else branch.
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "RUNNING",
            "layer_num": 5, "total_layer": 50, "mc_remaining": 20})
        out = actions["print_status"]("")
        self.assertIn("Print in progress", out)
        self.assertIn("layer 5 of 50", out)


if __name__ == "__main__":
    unittest.main()
