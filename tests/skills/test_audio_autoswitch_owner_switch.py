"""skills/audio_autoswitch: the device voice actions stamp the switch they
just confirmed in words (2026-11-17, overhaul actions-09).

Live 11-17 17:12:22 "switch the microphone to the ..." answered "Listening on
... now, sir.", and 48 s later the monolith's own "Switched to ..., sir." for
the same move was spoken again. The action now tells the monolith what it
moved (bobert_companion._note_owner_audio_switch); the monolith side is
tests/monolith/test_monolith_audio_flap.OwnerConfirmedSwitchTests.

Nothing real: the audio_switch calls are mocked, the monolith is a stub in
sys.modules, device names are synthetic.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from audio import audio_switch as A
from skills import audio_autoswitch as S

MIC = ("{0.0.1.00000000}.{synthetic-desk}", "Microphone (Synthetic Desk Mic)")
HS_MIC = ("{0.0.1.00000000}.{synthetic-hs}",
          "Headset Microphone (Synthetic Headset)")
HS_OUT = ("{0.0.0.00000000}.{synthetic-hs}", "Headphones (Synthetic Headset)")
SPK = ("{0.0.0.00000000}.{synthetic-spk}", "Speakers (Synthetic Speakers)")
CFG = {"AUDIO_AUTOSWITCH_HEADSET": "Synthetic Headset",
       "AUDIO_AUTOSWITCH_FALLBACK": "Synthetic Speakers",
       "AUDIO_AUTOSWITCH_MIC_FALLBACK": "Synthetic Desk Mic",
       "AUDIO_AUTOSWITCH_ENABLED": False,
       "PREFERRED_INPUT_DEVICES": [], "MICROPHONE_INDEX": None}


class OwnerSwitchStampTests(unittest.TestCase):
    def _run(self, action, *, ok=True, monolith=True):
        stamps = []
        stub = types.SimpleNamespace(
            _note_owner_audio_switch=lambda d, ep, name: stamps.append(
                (d, ep, name)),
            PREFERRED_INPUT_DEVICES=[], MICROPHONE_INDEX=None)
        actions = {}
        mods = {"bobert_companion": stub} if monolith else {}
        with mock.patch.object(S, "_cfg",
                               side_effect=lambda n, d=None: CFG.get(n, d)), \
                mock.patch.dict(sys.modules, mods):
            if not monolith:
                sys.modules.pop("bobert_companion", None)
            S.register(actions)
            with mock.patch.object(A, "headset_powered", return_value=True), \
                    mock.patch.object(A, "find_active_capture",
                                      return_value=HS_MIC), \
                    mock.patch.object(A, "resolve_mic_fallback",
                                      return_value=MIC), \
                    mock.patch.object(A, "find_active", return_value=HS_OUT), \
                    mock.patch.object(A, "resolve_fallback", return_value=SPK), \
                    mock.patch.object(A, "capture_override", return_value=None), \
                    mock.patch.object(A, "set_default_capture",
                                      return_value=ok), \
                    mock.patch.object(A, "set_default_render",
                                      return_value=ok), \
                    mock.patch("builtins.print"):
                said = actions[action]()
            self.loaded_after = "bobert_companion" in sys.modules
        return said, stamps

    def test_each_action_stamps_what_it_moved(self):
        for action, want in (("use_desk_mic", ("in",) + MIC),
                             ("use_headset_mic", ("in",) + HS_MIC),
                             ("use_headset", ("out",) + HS_OUT),
                             ("use_speakers", ("out",) + SPK)):
            said, stamps = self._run(action)
            self.assertEqual(stamps, [want], (action, said))

    def test_a_failed_switch_stamps_nothing(self):
        for action in ("use_desk_mic", "use_headset_mic", "use_headset",
                       "use_speakers"):
            said, stamps = self._run(action, ok=False)
            self.assertEqual(stamps, [], action)
            self.assertIn("couldn't", said)

    def test_no_monolith_loaded_is_fine_and_imports_nothing(self):
        said, stamps = self._run("use_desk_mic", monolith=False)
        self.assertEqual(stamps, [])
        self.assertIn("Synthetic Desk Mic", said)
        self.assertFalse(self.loaded_after,
                         "the stamp must never import the monolith")


if __name__ == "__main__":
    unittest.main()
