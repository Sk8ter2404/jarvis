"""'Clear the RAM' has an action and an honest answer (2026-11-17).

Live 2026-11-16 12:11:36 "Jarvis, try to clear the ram." -> the model's
[ACTION: clear_ram], "[router-miss] clear_ram unknown", the autocorrector
found nothing (0.74), and the reply claimed JARVIS lacked "the permissions to
reach into the system core". There was simply no handler. Now three routes
reach skills/system_pulse.free_memory_answer (read-only: the reading, the
biggest users, an offer to close one by name):

  * the whole-utterance fast path (core/fast_paths.is_free_memory_request),
    through the REAL entry point a live turn uses (_run_voice_shortcuts),
    the LLM booby-trapped;
  * the model naming free_memory (documented in the SYSTEM HEALTH section);
  * the model inventing clear_ram again (core/action_aliases row), through
    the real parse_and_run_actions._runner.

The real skills/system_pulse.py is loaded as skill_system_pulse with its
process table, GPU reading and command lines injected: no real process is
read. Neutral fixture names.

    python tools/run_tests.py monolith.test_monolith_free_memory
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import unittest
from unittest import mock

from core import failure_markers
from tests._monolith_harness import requires_monolith
from tests._skill_harness import load_skill_isolated
from tests.monolith.test_monolith_diag_fixes import _ShortcutBase

_GB = 1024 ** 3


class _PulseFixture:
    PULSE = {"cpu_pct": 30.0, "cpu_temp_c": 55.0, "ram_pct": 97.0,
             "gpu_util_pct": 20.0, "disk_free_gb": 500.0, "active_apps": 8}

    def load_pulse(self):
        prev = sys.modules.get("skill_system_pulse")

        def _restore():
            if prev is None:
                sys.modules.pop("skill_system_pulse", None)
            else:
                sys.modules["skill_system_pulse"] = prev
        self.addCleanup(_restore)
        with contextlib.redirect_stdout(io.StringIO()):
            self.pulse, self.pulse_actions = load_skill_isolated("system_pulse")
        with self.pulse._pulse_cache_lock:
            self.pulse._pulse_cache = {"pulse": dict(self.PULSE),
                                       "ts": time.time()}
        helper = os.path.join(self.pulse._PROJECT_DIR, "tools",
                              "demo_helper.py")
        rows = [(9001, "chrome.exe", 7.0 * _GB), (9002, "chrome.exe", 6.0 * _GB),
                (9003, "llama-server.exe", 8.5 * _GB),
                (9004, "claude.exe", 8.4 * _GB),
                (9005, "python.exe", 2.0 * _GB),
                (os.getpid(), "pythonw.exe", 1.5 * _GB)]
        cmdlines = {9005: ["python.exe", helper]}
        for name, kw in (
                ("_read_process_memory", {"side_effect": lambda: list(rows)}),
                ("_python_cmdline",
                 {"side_effect": lambda pid: cmdlines.get(pid, [])}),
                ("_gpu_busy_processes", {"side_effect": lambda: []}),
                ("_gather_pulse",
                 {"side_effect": AssertionError("cache is fresh")})):
            p = mock.patch.object(self.pulse, name, **kw)
            p.start()
            self.addCleanup(p.stop)


EXPECTED = ("Memory at 97 percent, sir — getting tight. The biggest memory "
            "users are Chrome at 13 gigabytes, my local brain model at 8.5 "
            "and Claude at 8.5. Memory another program holds only comes back "
            "when that program closes, and closing one is your call. Say "
            "'close Chrome' and I'll close it. Unloading my brain model would "
            "only make my next answer slower.")


@requires_monolith
class ClearTheRamFastPath(_PulseFixture, _ShortcutBase):
    def setUp(self):
        super().setUp()
        self.load_pulse()

    def test_the_live_line_is_answered_honestly_before_the_llm(self):
        handled, log = self._turn("Jarvis, try to clear the ram.")
        self.assertTrue(handled, "the ask reached the LLM")
        self.assertIn("[fast-path] memory-free", log)
        self.assertEqual(self.spoken, [EXPECTED])
        self.assertNotIn("permission", self.spoken[0].lower())
        self.llm.assert_not_called()

    def test_other_phrasings(self):
        for text in ("free up some memory", "Jarvis, can you free up some RAM?",
                     "clear the RAM please", "clean up the ram"):
            with self.subTest(text=text):
                self.spoken.clear()
                handled, log = self._turn(text)
                self.assertTrue(handled, text)
                self.assertIn("[fast-path] memory-free", log)
                self.assertEqual(self.spoken, [EXPECTED])
        self.llm.assert_not_called()

    def test_his_own_memory_is_not_the_pcs(self):
        for text in ("clear your memory", "clear the memory", "clear memory"):
            with self.subTest(text=text):
                handled, log = self._turn(text)
                self.assertNotIn("memory-free", log)
                self.assertEqual(self.spoken, [])

    def test_no_pulse_skill_goes_to_the_model(self):
        sys.modules.pop("skill_system_pulse", None)
        handled, log = self._turn("Jarvis, try to clear the ram.")
        self.assertFalse(handled)
        self.assertNotIn("memory-free", log)


@requires_monolith
class ClearTheRamAction(_PulseFixture, _ShortcutBase):
    """The model's own token: free_memory by name, or the invented clear_ram
    rewritten by the alias table, both through the real _runner."""

    def setUp(self):
        super().setUp()
        self.load_pulse()
        bc = self.bc
        for name in ("_write_hud_state", "record_session_action",
                     "record_action_history", "record_action_error"):
            self._p(bc, name, lambda *a, **k: None)
        self._p(bc, "PC_CONTROL_ENABLED", True)
        self._p(bc, "ACTION_ALIASES", True)
        self._p(bc._cmd_autocorrect, "_embed_disabled", True)
        acts = dict(bc.ACTIONS)
        acts["free_memory"] = self.pulse_actions["free_memory"]
        self._p(bc, "ACTIONS", acts)

    def _dispatch(self, reply):
        bc = self.bc
        prev = bc._begin_turn_grounding("Jarvis, try to clear the ram.")
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                res = bc.parse_and_run_actions(reply)
        finally:
            bc._end_turn_grounding(prev)
        return res, out.getvalue()

    def test_the_invented_clear_ram_runs_free_memory(self):
        (_clean, ran), log = self._dispatch("[ACTION: clear_ram]")
        self.assertIn("[alias] clear_ram -> free_memory", log)
        self.assertNotIn("clear_ram unknown", log)
        self.assertEqual([(n, r) for n, r, _i in ran],
                         [("free_memory", EXPECTED)])

    def test_free_memory_by_name_is_spoken_as_written(self):
        (_clean, ran), _log = self._dispatch("[ACTION: free_memory]")
        self.assertEqual([(n, r) for n, r, _i in ran],
                         [("free_memory", EXPECTED)])
        self.assertIn("free_memory", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        # No failure marker: a result with one goes to the failure
        # follow-up, where the model wrote the 11-16 "permissions" line.
        low = EXPECTED.lower()
        self.assertFalse([m for m in failure_markers.FAILURE_MARKERS
                          if m in low])

    def test_nothing_is_closed(self):
        closed = mock.Mock(return_value="closed")
        self.bc.ACTIONS["close_window"] = closed
        self._dispatch("[ACTION: free_memory]")
        closed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
