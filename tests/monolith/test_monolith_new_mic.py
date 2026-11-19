"""A newly plugged microphone is asked about, not followed (2026-11-17,
overhaul finding speech-mic-announce).

THE LIVE LOG. 11-17 17:09:57 a "USB microphone" became Active and the Windows
default; 17:10:23 JARVIS moved onto it and said "Switched to USB microphone,
sir."; he was heard (or not) on it until he typed the switch back at 17:12:19.
Following the default is the design (the Stream Deck moves it), so only a
STRANGER is held: AUDIO_FOLLOW_NEW_MIC (core/config, default True = follow; the hold
is an opt-in, so these tests patch it False).

Driven through the REAL _refresh_devices with the audio-flap world (a frozen
PortAudio table, Windows' endpoint truth, a hand clock). Device names are
SYNTHETIC. Mutation-checked by hand: `return (now_eps[0], prev_ep)` in
_apply_new_mic_hold -> test_a_stranger_mic_is_not_followed_and_is_asked_about
goes red; the `_audio_endpoint_gone` leave-a-gone-mic branch ->
test_a_stranger_is_followed_when_the_mic_in_use_is_gone goes red.

    python tools/run_tests.py  (selector monolith.test_monolith_new_mic)
"""
from __future__ import annotations

import unittest
import unittest.mock

from tests._monolith_harness import requires_monolith
from tests.monolith import test_monolith_audio_flap as F

STRANGER = "{0.0.1.00000000}.{synthetic-usb-mic}"
F.NAMES[STRANGER] = "Microphone (Synthetic USB Mic)"


@requires_monolith
class NewMicHoldTests(F._Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        bc._NEW_MIC_STATE.update(known=None, held=None, accepted="",
                                 notified={})
        for name, val in (("AUDIO_AUTOSWITCH_HEADSET", ""),
                          ("AUDIO_AUTOSWITCH_MIC_FALLBACK", "")):
            self._p(bc, name, val, create=True)
        # The live default follows the Windows default as always (the hold is
        # an opt-in); every test here opts in explicitly.
        from core import config
        self.assertTrue(config.AUDIO_FOLLOW_NEW_MIC)
        self.assertTrue(bc.AUDIO_FOLLOW_NEW_MIC)
        self._p(bc, "AUDIO_FOLLOW_NEW_MIC", False)
        self._settle()

    def _plug_stranger(self):
        w = self.world
        w.active.add(STRANGER)
        F.ORDER.insert(0, STRANGER)
        self.addCleanup(F.ORDER.remove, STRANGER)
        w.default_in = STRANGER

    def _plug_stranger_once(self):
        self.world.active.add(STRANGER)
        if STRANGER not in F.ORDER:
            F.ORDER.insert(0, STRANGER)
            self.addCleanup(F.ORDER.remove, STRANGER)
        self.world.default_in = STRANGER

    def _passes(self, n=4):
        for _ in range(n):               # past the 8 s hysteresis, twice over
            self._pass()

    def test_a_stranger_mic_is_not_followed_and_is_asked_about(self):
        self._plug_stranger()
        self._passes(6)
        names = self.world.frozen_names()
        self.assertEqual(names[self.bc._device_cache["in"]], F.NAMES[F.DESK])
        asked = [m for m in self.spoken if "appeared" in m]
        self.assertEqual(len(asked), 1, self.spoken)      # once, not per pass
        self.assertIn("use the new microphone", asked[0])
        self.assertFalse([m for m in self.spoken if m.startswith("Switched")])
        self.assertTrue(any("[audio] new microphone" in ln
                            for ln in self.printed), self.printed)

    def test_a_flapping_stranger_is_announced_once_per_flap_window(self):
        for _ in range(3):
            self._plug_stranger_once()
            self._passes(3)
            self.world.active.discard(STRANGER)       # unplugged ...
            self.world.default_in = F.DESK            # ... default back
            self._passes(3)
        asked = [m for m in self.spoken if "appeared" in m]
        self.assertEqual(len(asked), 1, self.spoken)
        self.assertTrue(any("already told sir" in ln for ln in self.printed))

    def test_saying_yes_follows_it_without_a_second_announcement(self):
        self._plug_stranger()
        self._passes(4)
        self.spoken.clear()
        name = self.bc._accept_held_new_mic()
        self.assertIn("USB Mic", name)
        self._passes(4)
        names = self.world.frozen_names()
        self.assertEqual(names[self.bc._device_cache["in"]],
                         F.NAMES[STRANGER])
        self.assertEqual(self.spoken, [])          # he was told in words
        self.assertEqual(self.bc._accept_held_new_mic(), "")   # used up

    def test_a_stranger_is_followed_when_the_mic_in_use_is_gone(self):
        self._plug_stranger()
        self._passes(2)
        self.world.set_desk(False)                 # the desk mic leaves ...
        self.world.default_in = STRANGER           # ... and the default stays
        self._passes(4)
        names = self.world.frozen_names()
        self.assertEqual(names[self.bc._device_cache["in"]],
                         F.NAMES[STRANGER])

    def test_a_mic_that_was_there_at_boot_is_followed_as_always(self):
        self.world.default_in = F.HS_MIC           # present since boot
        self._passes(4)
        names = self.world.frozen_names()
        self.assertEqual(names[self.bc._device_cache["in"]], F.NAMES[F.HS_MIC])
        self.assertFalse([m for m in self.spoken if "appeared" in m])

    def test_a_configured_name_is_followed_even_when_new(self):
        self._p(self.bc, "AUDIO_AUTOSWITCH_MIC_FALLBACK", "Synthetic USB",
                create=True)
        self._plug_stranger()
        self._passes(4)
        names = self.world.frozen_names()
        self.assertEqual(names[self.bc._device_cache["in"]],
                         F.NAMES[STRANGER])

    def test_a_preferred_mic_in_use_is_respected_and_nothing_is_asked(self):
        self._p(self.bc, "PREFERRED_INPUT_DEVICES", ["Desk Mic"])
        self._plug_stranger()
        self._passes(5)
        self.assertEqual(self.spoken, [])
        names = self.world.frozen_names()
        self.assertEqual(names[self.bc._device_cache["in"]], F.NAMES[F.DESK])

    def test_the_live_default_follows_any_new_default(self):
        self._p(self.bc, "AUDIO_FOLLOW_NEW_MIC", True)
        self._plug_stranger()
        self._passes(4)
        names = self.world.frozen_names()
        self.assertEqual(names[self.bc._device_cache["in"]],
                         F.NAMES[STRANGER])

    def test_the_default_moving_back_ends_the_hold(self):
        self._plug_stranger()
        self._passes(3)
        self.world.default_in = F.DESK
        self._passes(3)
        self.assertIsNone(self.bc._NEW_MIC_STATE["held"])


class NewMicActionTests(unittest.TestCase):
    def test_use_new_mic_and_its_alias_ask_the_monolith(self):
        import sys
        import types
        from skills import audio_autoswitch as S
        stub = types.SimpleNamespace(_accept_held_new_mic=lambda: "the USB Mic")
        actions = {}
        S.register(actions)
        with unittest.mock.patch.dict(sys.modules, {"bobert_companion": stub}):
            for name in ("use_new_mic", "switch_to_new_mic"):
                self.assertIn("the USB Mic", actions[name](""))
        stub2 = types.SimpleNamespace(_accept_held_new_mic=lambda: "")
        with unittest.mock.patch.dict(sys.modules, {"bobert_companion": stub2}):
            self.assertIn("no new microphone", actions["use_new_mic"](""))
        self.assertIn("use_new_mic", S.SPEAK_VERBATIM_ACTIONS)
        self.assertIn("switch_to_new_mic", S.SPEAK_VERBATIM_ACTIONS)


if __name__ == "__main__":
    unittest.main()
