"""v3.0.7 "why is the RAM so high", through the REAL entry point a live turn
uses (bobert_companion._run_voice_shortcuts), with the LLM booby-trapped and
the real skills/system_pulse.py loaded as skill_system_pulse.

Live 2026-10-28 18:53 (v3.0.5) "Why is the RAM so high?" at 82 % memory went
through the model to system_pulse (which gets no argument from it) and came
back opening with the all-nominal line - the number buried, the question
unanswered at the front, and JARVIS's own inference load on the GPU left
unexplained. It is now a fast path: the percent, the biggest
memory users, and "that's me thinking" for his own GPU load.

    python -m unittest tests.monolith.test_monolith_v307_ram_why
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests._skill_harness import load_skill_isolated
from tests.monolith.test_monolith_diag_fixes import _ShortcutBase

_GB = 1024 ** 3


@requires_monolith
class WhyIsTheRamSoHigh(_ShortcutBase):
    PULSE = {"cpu_pct": 51.0, "cpu_temp_c": 61.0, "ram_pct": 82.0,
             "gpu_util_pct": 100.0, "disk_free_gb": 500.0, "active_apps": 8}
    USERS = ("The biggest memory users are my local brain model at 8 "
             "gigabytes, a code editor at 4.5 and Chrome at 3.")

    def setUp(self):
        super().setUp()
        prev = sys.modules.get("skill_system_pulse")

        def _restore():
            if prev is None:
                sys.modules.pop("skill_system_pulse", None)
            else:
                sys.modules["skill_system_pulse"] = prev
        self.addCleanup(_restore)
        self.pulse, self.pulse_actions = load_skill_isolated("system_pulse")
        with self.pulse._pulse_cache_lock:
            self.pulse._pulse_cache = {"pulse": dict(self.PULSE),
                                       "ts": time.time()}
        own = os.getpid()
        rows = [(9001, "llama-server.exe", 8.0 * _GB),
                (9002, "Code.exe", 2.5 * _GB), (9003, "Code.exe", 2.0 * _GB),
                (9004, "chrome.exe", 1.5 * _GB), (9005, "chrome.exe", 1.5 * _GB),
                (own, "pythonw.exe", 2.0 * _GB),
                (9006, "notepad.exe", 0.1 * _GB)]
        self.gpu_rows = [("llama-server.exe", 9001, 90, 0)]
        self.recent = True
        for name, kw in (
                ("_read_process_memory", {"side_effect": lambda: list(rows)}),
                ("_gpu_busy_processes",
                 {"side_effect": lambda: self.gpu_rows}),
                ("_own_inference_recent",
                 {"side_effect": lambda *a: self.recent}),
                ("_gather_pulse",
                 {"side_effect": AssertionError("cache is fresh")})):
            p = mock.patch.object(self.pulse, name, **kw)
            p.start()
            self.addCleanup(p.stop)
        # The real code-editor name is not in the pulse's app table; give it
        # a neutral spoken name for the fixture.
        p = mock.patch.dict(self.pulse._MEM_APP_NAMES, {"code": "a code editor"})
        p.start()
        self.addCleanup(p.stop)

    def test_live_question_is_answered_before_the_llm(self):
        handled, log = self._turn("Why is the RAM so high?")
        self.assertTrue(handled, "the memory question reached the LLM")
        self.assertIn("[fast-path] memory-users", log)
        self.assertEqual(self.spoken, [
            "Memory at 82 percent, sir. " + self.USERS + " The GPU's busy at "
            "100 percent - that's me thinking."])
        self.assertNotIn("nominal", self.spoken[0])
        self.llm.assert_not_called()

    def test_other_phrasings(self):
        for text in ("Jarvis, what's using all my memory?",
                     "why is memory so high"):
            with self.subTest(text=text):
                self.spoken.clear()
                handled, _log = self._turn(text)
                self.assertTrue(handled)
                self.assertTrue(self.spoken[0].startswith(
                    "Memory at 82 percent, sir. " + self.USERS),
                    self.spoken[0])
        self.llm.assert_not_called()

    def test_a_game_on_the_gpu_is_not_me_thinking(self):
        self.gpu_rows = [("fortniteclient-win64-shipping.exe", 9100, 95, 0)]
        handled, _log = self._turn("Why is the RAM so high?")
        self.assertTrue(handled)
        self.assertEqual(self.spoken, ["Memory at 82 percent, sir. " +
                                       self.USERS])

    def test_hows_the_computer_doing_still_goes_to_the_model(self):
        handled, log = self._turn("How's the computer doing?")
        self.assertFalse(handled)
        self.assertNotIn("memory-users", log)
        self.assertEqual(self.spoken, [])

    def test_setting_off_goes_to_the_model_as_before(self):
        with mock.patch("core.config.SYSTEM_PULSE_MEMORY_ANSWER", False,
                        create=True):
            handled, _log = self._turn("Why is the RAM so high?")
        self.assertFalse(handled)
        self.assertEqual(self.spoken, [])

    def test_no_pulse_skill_goes_to_the_model(self):
        sys.modules.pop("skill_system_pulse", None)
        handled, _log = self._turn("Why is the RAM so high?")
        self.assertFalse(handled)


if __name__ == "__main__":
    unittest.main()
