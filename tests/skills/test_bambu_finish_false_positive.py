"""Live false 'Print complete' announcements (2026-10-09 / 2026-10-12).

A FINISH is spoken only when THIS session watched the same file RUNNING/PAUSE;
a FINISH found at boot, or a new file name arriving while the printer still
says FINISH, is never a completion; a restart never replays a finish; a queued
finish line is dropped if the printer is no longer FINISH for that file.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tests.skills.test_bambu_monitor import _load_bambu


class FinishFalsePositiveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.statefile = os.path.join(self._tmp.name, "rs.json")
        self.boot()

    def boot(self):
        """A fresh process: new module state, same durable file."""
        self.mod, _ = _load_bambu()
        self.mod._REMINDER_STATE_FILE = self.statefile
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0

    def push(self, **kw):
        with self.mod._state_lock:
            self.mod._state.update(kw)
        with mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._handle_state_change()
        return [c.args[0] for c in enq.call_args_list
                if "Print complete" in c.args[0]]

    def test_boot_in_finish_is_silent(self):
        # 10-09 18:32:48 / 10-12 15:26:42: first push after boot is FINISH.
        self.assertEqual(self.push(gcode_state="FINISH",
                                   filename="all servo arms.3mf"), [])
        self.assertEqual(self.push(gcode_state="FINISH",
                                   filename="all servo arms.3mf"), [])

    def test_new_name_during_finish_is_a_new_job_not_a_completion(self):
        # 10-12 15:31:35: '3D PCB' started while gcode_state was still FINISH.
        self.push(gcode_state="RUNNING", filename="all servo arms.3mf",
                  mc_percent=50)
        self.assertEqual(len(self.push(gcode_state="FINISH",
                                       filename="all servo arms.3mf")), 1)
        self.assertEqual(self.push(gcode_state="FINISH",
                                   filename="3D PCB.3mf"), [])
        self.assertEqual(self.push(gcode_state="RUNNING",
                                   filename="3D PCB.3mf", mc_percent=1), [])
        out = self.push(gcode_state="FINISH", filename="3D PCB.3mf")
        self.assertEqual(len(out), 1)
        self.assertIn("3D PCB", out[0])

    def test_witnessed_finish_not_replayed_after_restart(self):
        self.push(gcode_state="RUNNING", filename="cube.3mf", mc_percent=10)
        self.assertEqual(len(self.push(gcode_state="FINISH",
                                       filename="cube.3mf")), 1)
        self.boot()
        self.assertEqual(self.push(gcode_state="FINISH",
                                   filename="cube.3mf"), [])

    def test_reprint_of_same_file_still_announces(self):
        for _ in range(2):
            self.push(gcode_state="RUNNING", filename="cube.3mf",
                      mc_percent=10)
            self.assertEqual(len(self.push(gcode_state="FINISH",
                                           filename="cube.3mf")), 1)

    def test_job_id_keys_the_finish(self):
        self.push(gcode_state="RUNNING", filename="cube.3mf", task_id="111")
        self.push(gcode_state="FINISH", filename="cube.3mf", task_id="111")
        self.boot()
        self.push(gcode_state="RUNNING", filename="cube.3mf", task_id="222")
        self.assertEqual(len(self.push(gcode_state="FINISH",
                                       filename="cube.3mf",
                                       task_id="222")), 1)

    def test_queued_finish_dropped_when_printer_moved_on(self):
        sv = self.mod.speech_still_valid
        with self.mod._state_lock:
            self.mod._state.update(gcode_state="FINISH", filename="cube.3mf")
        self.assertTrue(sv("bambu_finish:cube"))
        self.assertFalse(sv("bambu_finish:other"))
        with self.mod._state_lock:
            self.mod._state.update(gcode_state="RUNNING")
        self.assertFalse(sv("bambu_finish:cube"))
        self.assertTrue(sv("something:else"))


if __name__ == "__main__":
    unittest.main()
