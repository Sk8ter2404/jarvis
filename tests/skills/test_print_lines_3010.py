"""v3.0.10 lane F: print lines (skills/bambu_monitor.py,
skills/proactive_print_companion.py, skills/bambu_h2d_voice_companion.py).

Live 2026-10-28 (paraphrased):
  * 22:26:44 the finish line was dropped as "stale" and the companion's
    "Shall I queue a cooldown timer for you, sir?" was spoken on its own.
    The drain asked a second, never-started copy of bambu_monitor (empty
    state), so every queued finish line failed the check (FINISH_LINE_KEEP;
    the drain half is pinned in tests/monolith/test_monolith_finish_line_keep).
  * 21:58 the owner-away hold released 25 % beside 50 %, each with a hedged
    companion line naming faults nobody saw ("a hiccup in the cooling fans")
    (PRINT_MILESTONE_SUPERSEDE, PRINT_MILESTONE_ONE_LINE).

Each flag is pinned on and off (off = the old path exactly). No printer, no
network; the speech queue is a temp file or a mock.

    python tools/run_tests.py print_lines_3010
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from core import config
from tests._skill_harness import load_skill_isolated
from tests.skills.test_bambu_monitor import _load_bambu


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


class _MonitorBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mod, _ = _load_bambu()
        self.mod._REMINDER_STATE_FILE = os.path.join(self._tmp.name, "rs.json")
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
        self.mod._finish_line_passed_at[0] = 0.0

    def flag(self, name, value):
        p = mock.patch.object(config, name, value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def state(self, **kw):
        with self.mod._state_lock:
            self.mod._state.update(kw)


# ── FINISH_LINE_KEEP: the drain-time check ────────────────────────────────
class FinishLineKeepTests(_MonitorBase):
    def sv(self, tag):
        return _quiet(self.mod.speech_still_valid, tag)

    def test_kept_unless_a_new_job_is_under_way(self):
        self.flag("FINISH_LINE_KEEP", True)
        for gs in ("FINISH", "IDLE", "FAILED", "", None):
            with self.subTest(state=gs):
                self.state(gcode_state=gs, filename="cube.3mf")
                self.assertTrue(self.sv("bambu_finish:cube")[0])
        # A file-name jitter (gcode_file in place of subtask_name) keeps it.
        self.state(gcode_state="FINISH", filename="plate_1.gcode")
        self.assertTrue(self.sv("bambu_finish:cube")[0])
        for gs in ("RUNNING", "PREPARE", "PAUSE"):
            with self.subTest(state=gs):
                self.state(gcode_state=gs, filename="next.3mf")
                ok, log = self.sv("bambu_finish:cube")
                self.assertFalse(ok)
                self.assertIn("finish line stale (a new job is under way)", log)
                self.assertIn("state=" + gs, log)
                self.assertIn("want=cube", log)
        self.assertTrue(self.sv("something:else")[0])

    def test_a_changed_job_id_is_a_new_job(self):
        self.flag("FINISH_LINE_KEEP", True)
        self.state(gcode_state="FINISH", filename="cube.3mf", task_id="111")
        tag = self.mod._finish_tag("cube")
        self.assertEqual(tag, "bambu_finish:cube|job=111")
        self.assertTrue(self.sv(tag)[0])
        self.state(filename="other.3mf")            # same job: still kept
        self.assertTrue(self.sv(tag)[0])
        self.state(task_id="222")
        ok, log = self.sv(tag)
        self.assertFalse(ok)
        self.assertIn("job id changed", log)

    def test_flag_off_is_the_old_check_exactly(self):
        self.flag("FINISH_LINE_KEEP", False)
        self.state(gcode_state="FINISH", filename="cube.3mf", task_id="111")
        self.assertEqual(self.mod._finish_tag("cube"), "bambu_finish:cube")
        self.assertTrue(self.sv("bambu_finish:cube")[0])
        self.assertFalse(self.sv("bambu_finish:other")[0])
        ok, log = self.sv("bambu_finish:cube|job=111")
        self.assertFalse(ok)
        self.assertEqual(log, "")                   # the old drop is unlogged
        self.state(gcode_state="")
        self.assertFalse(self.sv("bambu_finish:cube")[0])
        # The offer tag is not a finish tag on the old path.
        self.assertTrue(self.sv("bambu_finish_offer:")[0])

    def test_the_offer_needs_its_finish_line_just_before(self):
        self.flag("FINISH_LINE_KEEP", True)
        ok, log = self.sv("bambu_finish_offer:")
        self.assertFalse(ok)                        # no finish line spoken
        self.assertIn("finish offer dropped", log)
        self.state(gcode_state="FINISH", filename="cube.3mf")
        self.assertTrue(self.sv("bambu_finish:cube")[0])
        self.assertTrue(self.sv("bambu_finish_offer:")[0])
        # The head judged stale: the tail goes with it.
        self.mod._finish_line_passed_at[0] = 0.0
        self.state(gcode_state="RUNNING")
        self.assertFalse(self.sv("bambu_finish:cube")[0])
        self.assertFalse(self.sv("bambu_finish_offer:")[0])
        # Too long after the head: dropped.
        self.mod._finish_line_passed_at[0] = 1000.0
        with mock.patch.object(self.mod.time, "time",
                               return_value=1000.0
                               + self.mod.FINISH_OFFER_WINDOW_S + 1):
            self.assertFalse(self.sv("bambu_finish_offer:")[0])

    def test_the_witnessed_finish_queues_the_job_tag(self):
        self.flag("FINISH_LINE_KEEP", True)
        calls = []
        with mock.patch.object(self.mod, "_enqueue_speech",
                               side_effect=lambda m, **k: calls.append(k)):
            self.state(gcode_state="RUNNING", filename="cube.3mf",
                       mc_percent=10, task_id="9")
            self.mod._handle_state_change()
            self.state(gcode_state="FINISH", mc_percent=100)
            self.mod._handle_state_change()
        tags = [k.get("still_valid") for k in calls if k.get("still_valid")]
        self.assertEqual(tags, ["bambu_finish:cube|job=9"])


# ── PRINT_MILESTONE_SUPERSEDE: bambu_monitor's plain lines ────────────────
class MilestoneSupersedeTests(_MonitorBase):
    def setUp(self):
        super().setUp()
        self.queue = os.path.join(self._tmp.name, "pending_speech.json")
        self.mod._SPEECH_QUEUE = self.queue
        # A parent without proactive_announce: the skill's own queue write.
        p = mock.patch.dict(sys.modules,
                            {"bobert_companion": types.ModuleType("bc")})
        p.start()
        self.addCleanup(p.stop)

    def drive(self, *pcts, remaining=None):
        self.state(gcode_state="RUNNING", filename="cube.3mf",
                   mc_percent=0, mc_remaining=remaining)
        self.mod._last_gcode_state[0] = None       # joined: no start line
        self.mod._handle_state_change()
        for pct in pcts:
            self.state(mc_percent=pct)
            self.mod._handle_state_change()
        if not os.path.exists(self.queue):
            return []
        with open(self.queue, encoding="utf-8") as f:
            return [e["message"] for e in json.load(f)]

    def test_a_newer_milestone_replaces_an_unspoken_one(self):
        self.flag("PRINT_MILESTONE_SUPERSEDE", True)
        msgs = self.drive(26, 51)
        self.assertEqual(msgs, ["Print is 50% complete, sir."])
        self.assertEqual(self.mod._milestone_had_eta, {25: False, 50: False})

    def test_the_eta_tail_is_recorded_for_the_companion(self):
        self.flag("PRINT_MILESTONE_SUPERSEDE", True)
        msgs = self.drive(26, remaining=90)
        self.assertEqual(len(msgs), 1)
        self.assertIn("Estimated 1 hour and 30 minutes remaining", msgs[0])
        self.assertEqual(self.mod._milestone_had_eta, {25: True})

    def test_flag_off_queues_both(self):
        self.flag("PRINT_MILESTONE_SUPERSEDE", False)
        self.assertEqual(self.drive(26, 51),
                         ["Print is 25% complete, sir.",
                          "Print is 50% complete, sir."])

    def test_the_announcer_gets_the_family_tag(self):
        self.flag("PRINT_MILESTONE_SUPERSEDE", True)
        bc = types.ModuleType("bc")
        bc.proactive_announce = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.mod._enqueue_speech("x", supersede="bambu_milestone")
        self.assertEqual(bc.proactive_announce.call_args.kwargs,
                         {"source": "bambu", "supersede": "bambu_milestone"})


# ── the companion: one line per milestone, the offer tied to its head ─────
def _fake_bambu(state, announced, had_eta=None):
    m = types.ModuleType("skill_bambu_monitor")
    m._state_lock = threading.Lock()
    m._state = state
    m._announced_milestones = announced
    if had_eta is not None:
        m._milestone_had_eta = had_eta
    m.register_state_change_hook = mock.MagicMock()
    return m


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def flag(self, name, value):
        p = mock.patch.object(config, name, value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def load(self, remaining=60, had_eta=None, bambu=True):
        st = {"last_update": 1.0, "gcode_state": "RUNNING",
              "filename": "Part_PLA.3mf", "mc_percent": 30.0,
              "mc_remaining": remaining, "total_layer": 120}
        fake = _fake_bambu(st, {25}, had_eta) if bambu else None
        p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake,
                                          "skill_dnd_focus_mode": None})
        p.start()
        self.addCleanup(p.stop)
        mod, _a = load_skill_isolated("proactive_print_companion")
        mod._PATTERNS_FILE = os.path.join(self.tmpdir.name, "p.json")
        mod._current_filename[0] = "Part_PLA.3mf"
        mod._last_gcode_state[0] = "RUNNING"
        mod._print_midflight[0] = False
        mod._milestone_detected_at[25] = 5000.0
        return mod

    def poll(self, mod):
        with mock.patch.object(mod.time, "time",
                               return_value=5000.0
                               + mod.MILESTONE_OFFSET_SECONDS + 1), \
             mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_sample_vision_async"), \
             contextlib.redirect_stdout(io.StringIO()):
            mod._poll_once()
        self.assertIn(25, mod._milestone_announced)
        return enq.call_args_list

    def test_no_eta_no_line(self):
        self.flag("PRINT_MILESTONE_ONE_LINE", True)
        self.assertEqual(self.poll(self.load(remaining=None)), [])

    def test_the_plain_line_gave_the_time_so_no_second_line(self):
        self.flag("PRINT_MILESTONE_ONE_LINE", True)
        self.assertEqual(self.poll(self.load(had_eta={25: True})), [])

    def test_a_clock_eta_line_without_a_hedge(self):
        self.flag("PRINT_MILESTONE_ONE_LINE", True)
        self.flag("PRINT_MILESTONE_SUPERSEDE", True)
        mod = self.load(remaining=90, had_eta={25: False})
        calls = self.poll(mod)
        self.assertEqual(len(calls), 1)
        line = calls[0].args[0]
        self.assertIn("Quarter of the way along", line)
        self.assertIn("completion at", line)
        for hedge in mod._MCU_HEDGE_SUFFIXES:
            self.assertNotIn(hedge, line)
        self.assertEqual(calls[0].kwargs, {"supersede": "bambu_milestone"})

    def test_quiet_in_focus_mode(self):
        self.flag("PRINT_MILESTONE_ONE_LINE", True)
        mod = self.load(remaining=90, had_eta={25: False})
        dnd = types.ModuleType("skill_dnd_focus_mode")
        dnd.is_focus_mode_active = lambda: True
        with mock.patch.dict(sys.modules, {"skill_dnd_focus_mode": dnd}):
            self.assertEqual(self.poll(mod), [])

    def test_flag_off_is_the_old_hedged_line(self):
        self.flag("PRINT_MILESTONE_ONE_LINE", False)
        self.flag("PRINT_MILESTONE_SUPERSEDE", False)
        mod = self.load(remaining=None, had_eta={25: True})
        calls = self.poll(mod)
        self.assertEqual(len(calls), 1)
        self.assertIn("trajectory still settling", calls[0].args[0])
        self.assertTrue(any(h in calls[0].args[0]
                            for h in mod._MCU_HEDGE_SUFFIXES))
        self.assertEqual(calls[0].kwargs, {})

    def offer_call(self):
        mod = self.load()
        mod._saw_running_this_print[0] = True
        with mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_completion_offer_line",
                               return_value="Shall I queue a cooldown timer "
                                            "for you, sir?"), \
             mock.patch.object(mod, "_record_print_outcome"):
            mod._maybe_announce_completion_offer({"filename": "Part_PLA.3mf"})
        enq.assert_called_once()
        return enq.call_args

    def test_the_offer_is_tied_to_its_finish_line(self):
        self.flag("FINISH_LINE_KEEP", True)
        self.assertEqual(self.offer_call().kwargs,
                         {"still_valid": "bambu_finish_offer:"})

    def test_flag_off_the_offer_queues_on_its_own(self):
        self.flag("FINISH_LINE_KEEP", False)
        self.assertEqual(self.offer_call().kwargs, {})

    def test_enqueue_passes_the_tags_and_survives_an_old_announcer(self):
        mod = self.load()
        bc = types.ModuleType("bobert_companion")
        seen = []

        def old_announce(message, source="skill"):
            seen.append((message, source))
        bc.proactive_announce = old_announce
        with mock.patch.object(mod.importlib, "import_module", return_value=bc):
            mod._enqueue_speech("hi", still_valid="bambu_finish_offer:")
        self.assertEqual(seen, [("hi", "print_companion")])


# ── the H2D voice companion leaves the milestones to bambu_monitor ───────
class VoiceCompanionTests(unittest.TestCase):
    def load(self):
        fake = _fake_bambu({"last_update": 0.0}, set())
        p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake})
        p.start()
        self.addCleanup(p.stop)
        mod, _a = load_skill_isolated("bambu_h2d_voice_companion")
        self.addCleanup(mod._stop_hook_retry)
        mod._current_filename[0] = "cube"
        mod._announced_milestones.clear()
        return mod

    def snap(self, mod, value):
        p = mock.patch.object(config, "PRINT_MILESTONE_ONE_LINE", value,
                              create=True)
        p.start()
        self.addCleanup(p.stop)
        with mock.patch.object(mod, "_gated_announce") as gated, \
             mock.patch.object(mod, "_direct_enqueue"):
            with mod._state_lock:
                mod._process_snapshot({"filename": "cube.3mf",
                                       "mc_percent": 25,
                                       "mc_remaining": 60}, "RUNNING")
        return [c.args[0] for c in gated.call_args_list]

    def test_quiet_when_the_monitor_owns_the_line(self):
        mod = self.load()
        self.assertEqual(self.snap(mod, True), [])
        self.assertIn(25, mod._announced_milestones)

    def test_flag_off_speaks_its_variant(self):
        mod = self.load()
        msgs = self.snap(mod, False)
        self.assertTrue(any("Print at 25%" in m for m in msgs), msgs)


if __name__ == "__main__":
    unittest.main()
