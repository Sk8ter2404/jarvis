"""July 4 audit leftovers, rechecked 2026-10-02 against v2.0.173 (cluster C1,
the three that could change JARVIS's state on a plain sentence):

  A28  "shut down the computer" / "power off the tv" armed the overnight
       shutdown prompt instead of running, and a plain "No." then powered
       JARVIS off. The pre-router now asks the dispatcher's self-termination
       test (core.action_risk.asked_for_self_termination) first.
  A37  A reply saying "I'm always listening, sir." with no token injected
       wake_word_mode_off, which persists - the owner's wake-word mode
       silently stayed off. Only a switch BACK to normal listening injects.
  A29  In standby, "come back here", "I need you to pass the salt" and "wake
       me at seven" woke JARVIS and got a greeting. The name still wakes from
       anywhere; the soft phrases only as the whole utterance.

    python -m unittest tests.monolith.test_monolith_audit_c1
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m


@requires_monolith
class ShutdownPreRouterTests(_Base):
    """A28."""

    def setUp(self):
        bc = self.bc
        orig = dict(bc._shutdown_prompt_pending)
        self.addCleanup(lambda: (bc._shutdown_prompt_pending.clear(),
                                 bc._shutdown_prompt_pending.update(orig)))
        bc._shutdown_prompt_pending["armed"] = False
        self.speak = self._p(bc, "_speak")
        self.shutdown = self._p(bc, "_act_shutdown_jarvis")
        self.overnight = self._p(bc, "_act_start_overnight_upgrade")

    def test_device_commands_never_arm_the_prompt(self):
        bc = self.bc
        for text in ("shut down the computer", "power off the tv",
                     "shut down chrome", "Jarvis, shut down the PC"):
            with self.subTest(text=text):
                bc._shutdown_prompt_pending["armed"] = False
                self.assertFalse(bc._check_and_arm_shutdown_prompt(text))
                self.assertFalse(bc._shutdown_prompt_pending.get("armed"))
        self.speak.assert_not_called()

    def test_jarvis_itself_still_arms_it(self):
        bc = self.bc
        for text in ("JARVIS, shut down", "shut down", "power off",
                     "shut down jarvis", "go offline now",
                     "turn yourself off"):
            with self.subTest(text=text):
                bc._shutdown_prompt_pending["armed"] = False
                self.assertTrue(bc._check_and_arm_shutdown_prompt(text))
                self.assertTrue(bc._shutdown_prompt_pending["armed"])

    def test_no_after_a_device_command_never_powers_jarvis_off(self):
        # The live chain: "power off the tv", then a plain "No." - the
        # prompt is never armed, so the "No." has nothing to confirm.
        bc = self.bc
        bc._check_and_arm_shutdown_prompt("power off the tv")
        self.assertFalse(bc._handle_shutdown_prompt("No."))
        self.shutdown.assert_not_called()

    def test_a_device_command_while_armed_is_not_a_reinforced_shutdown(self):
        bc = self.bc
        self.assertTrue(bc._check_and_arm_shutdown_prompt("JARVIS, shut down"))
        bc._handle_shutdown_prompt("shut down the computer")
        self.shutdown.assert_not_called()

    def test_a_repeated_shutdown_still_reinforces(self):
        bc = self.bc
        self.assertTrue(bc._check_and_arm_shutdown_prompt("JARVIS, shut down"))
        self.assertTrue(bc._handle_shutdown_prompt("shut down"))
        self.shutdown.assert_called_once()


@requires_monolith
class AlwaysListeningInjectTests(_Base):
    """A37."""

    def _action(self, reply):
        out = self.bc._detect_preemptive_hallucination(reply)
        return out[1] if out else None

    def test_describing_always_listening_never_injects_wake_word_off(self):
        for reply in ("I'm always listening, sir.",
                      "Always listening, sir - just say the word.",
                      "Don't worry, I'm on normal listening duty."):
            with self.subTest(reply=reply):
                self.assertNotEqual(self._action(reply), "wake_word_mode_off")

    def test_a_switch_back_to_normal_listening_still_injects(self):
        for reply in ("Switching back to normal listening, sir.",
                      "Going back to normal listening.",
                      "Resuming always-on listening, sir.",
                      "Disabling wake word mode, sir."):
            with self.subTest(reply=reply):
                self.assertEqual(self._action(reply), "wake_word_mode_off")


@requires_monolith
class StandbySoftWakeTests(_Base):
    """A29."""

    def setUp(self):
        bc = self.bc
        self._speak = self._p(bc, "_speak")
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_audio_music_should_refuse_wake", return_value=False)
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_ambient_learning_feed")
        self._p(bc, "context_aware_greeting",
                return_value=("Back online, sir.", 1.0))
        self._p(bc, "OVERNIGHT_FLAG_FILE",
                os.path.join(tempfile.gettempdir(), "no_such_overnight_flag.json"))
        for name in ("_sleep_mode", "_standby_mode"):
            cell = getattr(bc, name)
            self.addCleanup(cell.__setitem__, 0, cell[0])

    def _asleep(self):
        self.bc._sleep_mode[0] = True
        self.bc._standby_mode[0] = True

    def test_soft_phrases_inside_a_sentence_never_wake(self):
        for text in ("come back here", "I need you to pass the salt",
                     "wake me at seven", "can you wake up the kids"):
            with self.subTest(text=text):
                self._asleep()
                self.bc._handle_sleep_standby(text)
                self.assertTrue(self.bc._sleep_mode[0])
        self._speak.assert_not_called()

    def test_a_soft_phrase_on_its_own_still_wakes(self):
        for text in ("wake up", "Wake up, please.", "come back",
                     "I need you", "start listening", "okay wake up sir"):
            with self.subTest(text=text):
                self._asleep()
                self.bc._handle_sleep_standby(text)
                self.assertFalse(self.bc._sleep_mode[0])

    def test_the_name_wakes_from_anywhere(self):
        for text in ("hey JARVIS wake up please",
                     "so jarvis are you there", "Jarvis."):
            with self.subTest(text=text):
                self._asleep()
                self.bc._handle_sleep_standby(text)
                self.assertFalse(self.bc._sleep_mode[0])

    def test_the_shared_wake_regex_is_unchanged(self):
        # The dialogue hold and the R6 rescue still match anywhere.
        self.assertTrue(self.bc._wake_word_heard("come back here"))


if __name__ == "__main__":
    unittest.main()
