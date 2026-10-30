"""Action-fit rule 7 sees play_music's transport arguments (cross-lane review
2026-10-29), and an accepted song offer is still a song.

Rule 7 (core.action_fit.check_rule) blocks the same (name, arg) again for a
bare acknowledgement or decline. Lane C (PLAY_MUSIC_TRANSPORT_WORDS) made
[ACTION: play_music, pause] run the pause_music handler, so after a routed
"Jarvis Pause movie." a decline answered with [ACTION: play_music, pause]
was a DIFFERENT pair, ran, and paused again - the live 18:53:47 double
pause, and without winrt the media-key fallback is a toggle that flips the
movie back. _action_fit_refusal now hands the check canonical pairs
(_action_fit_canonical_pair): play_music with a transport argument is the
transport action its handler runs. What runs is unchanged.

Second, core.actions._play_music_transport_op read only sir's own words for
"play <title>": "Yes." to "Shall I play Carry On for you, sir?" became
play_music, Carry On, and "Carry On" read as a resume. The offer the turn
took now counts as his words too.

These drive the REAL _run_llm_dispatch with the offer-yes harness (every
action stubbed or its media handlers patched; nothing is played or paused).
Paraphrased fixtures; no LLM, no audio.

    python tools/run_tests.py monolith.test_monolith_action_fit_transport_1029
"""
from __future__ import annotations

import collections
import contextlib
import io
import time
import unittest

import core.config as cfg
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_offer_yes import _OfferBase

NOTE_OFFER = "Shall I note that for Claude to add, sir?"
SONG_OFFER = "Shall I play Carry On for you, sir?"


@requires_monolith
class _TransportFitBase(_OfferBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.hist = collections.deque(maxlen=5)
        self._p(bc, "_action_history", self.hist)
        self._p(bc, "record_action_history",
                side_effect=lambda n, a, r: self.hist.append(
                    {"action": n, "arg": a, "result": str(r),
                     "at": time.time()}))
        self._p(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", True, create=True)
        self._p(cfg, "ACTION_FIT_MODE", "shadow", create=True)
        self._p(bc, "ACTION_FIT_MODE", "shadow", create=True)
        self._p(bc, "ACTION_FIT_ACK_REPEAT_ENFORCE", True, create=True)
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._stub("pause_music", "Paused Chrome, sir.")
        self._stub("resume_music", "Resumed Chrome, sir.")
        self._stub("note_for_claude", "Noted for Claude, sir.")
        self.script: list[str] = []
        self.brain = (lambda _prompt: self.script.pop(0) if self.script
                      else "Standing by, sir.")

    def turn(self, text, replies=()):
        self.script = list(replies)
        self.followups = []
        return self._turn(text)


class PlayMusicTransportRepeatTests(_TransportFitBase):
    def setUp(self):
        super().setUp()
        self._stub("play_music", "Paused Chrome, sir.")

    def _pause_then_decline(self):
        self.turn("Jarvis Pause movie.")
        self.assertEqual(self.calls["pause_music"], [""])
        self.turn("Tell me a fun fact.", [NOTE_OFFER])
        return self.turn("No, not right now.",
                         ["Very well, sir. [ACTION: play_music, pause]"])

    def test_a_decline_after_a_routed_pause_does_not_pause_again(self):
        printed = self._pause_then_decline()
        self.assertEqual(self.calls["play_music"], [])
        self.assertEqual(self.calls["pause_music"], [""])
        self.assertIn("[action-fit] blocked play_music(pause)", printed)

    def test_an_all_right_after_a_routed_resume_does_not_resume_again(self):
        self.turn("Jarvis Resume movie.")
        self.assertEqual(self.calls["resume_music"], [""])
        printed = self.turn("All right.",
                            ["Very good, sir. [ACTION: play_music, resume]"])
        self.assertEqual(self.calls["play_music"], [])
        self.assertEqual(self.calls["resume_music"], [""])
        self.assertIn("[action-fit] blocked play_music(resume)", printed)

    def test_without_the_canonical_pairs_the_repeat_runs(self):
        # The defect itself: the raw pairs differ, so rule 7 lets it run.
        self._p(self.bc, "_action_fit_canonical_pair",
                side_effect=lambda n, a: (n, a))
        self._pause_then_decline()
        self.assertEqual(self.calls["play_music"], ["pause"])

    def test_a_raising_canonical_pair_falls_back_to_the_raw_pairs(self):
        self._p(self.bc, "_action_fit_canonical_pair",
                side_effect=RuntimeError("boom"))
        printed = self._pause_then_decline()
        self.assertEqual(self.calls["play_music"], ["pause"])
        self.assertIn("canonical pair failed", printed)

    def test_a_real_resume_after_a_pause_still_runs(self):
        self.turn("Jarvis Pause movie.")
        self.turn("Resume.", ["Resuming, sir. [ACTION: play_music, resume]"])
        self.assertEqual(self.calls["play_music"], ["resume"])

    def test_the_direct_repeat_is_still_blocked(self):
        self.turn("Jarvis Pause movie.")
        self.turn("Tell me a fun fact.", [NOTE_OFFER])
        self.turn("No, not right now.",
                  ["Very well, sir. [ACTION: pause_music]"])
        self.assertEqual(self.calls["pause_music"], [""])

    def test_transport_words_off_is_the_old_check(self):
        self._p(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", False)
        self._p(self.bc, "PLAY_MUSIC_TRANSPORT_WORDS", False, create=True)
        self.turn("Jarvis Pause movie.", ["Pausing. [ACTION: pause_music]"])
        self.turn("Tell me a fun fact.", [NOTE_OFFER])
        self.turn("No, not right now.",
                  ["Very well, sir. [ACTION: play_music, pause]"])
        self.assertEqual(self.calls["play_music"], ["pause"])

    def test_rule_seven_off_is_the_old_check(self):
        self._p(self.bc, "ACTION_FIT_ACK_REPEAT_ENFORCE", False)
        self._pause_then_decline()
        self.assertEqual(self.calls["play_music"], ["pause"])


class CanonicalPairTests(_TransportFitBase):
    def _canon(self, name, arg, said=""):
        bc = self.bc
        prev = bc._begin_turn_grounding(said)
        try:
            return bc._action_fit_canonical_pair(name, arg)
        finally:
            bc._end_turn_grounding(prev)

    def test_transport_arguments(self):
        for arg, want in (("pause", "pause_music"), ("resume", "resume_music"),
                          ("resume the movie", "resume_music"),
                          ("next song", "next_song"), ("it", "resume_music")):
            with self.subTest(arg=arg):
                self.assertEqual(self._canon("play_music", arg), (want, ""))

    def test_songs_and_other_actions_are_unchanged(self):
        for name, arg, said in (("play_music", "Earth Song", ""),
                                ("play_music", "Carry On", "play Carry On"),
                                ("pause_music", "", ""),
                                ("open_app", "pause", "")):
            with self.subTest(name=name, arg=arg):
                self.assertEqual(self._canon(name, arg, said), (name, arg))

    def test_a_name_bound_to_the_play_music_handler(self):
        self._actions["play_song_alias_test"] = self._actions["play_music"]
        self.assertEqual(self._canon("play_song_alias_test", "pause"),
                         ("pause_music", ""))


class AcceptedSongOfferTests(_TransportFitBase):
    def setUp(self):
        super().setUp()
        import core.actions as A
        self._actions["play_music"] = A._act_play_music
        self.search = self._p(A, "_act_apple_music",
                              return_value="streamed via apple music")
        self.resume = self._p(A, "_act_resume_music",
                              return_value="Resumed Chrome, sir.")

    def test_yes_to_a_song_offer_plays_the_song(self):
        self.turn("Jarvis, any music suggestions?", [SONG_OFFER])
        printed = self.turn("Yes.",
                            ["Right away. [ACTION: play_music, Carry On]"])
        self.search.assert_called_once_with("Carry On")
        self.resume.assert_not_called()
        self.assertIn("the offer sir took named this title", printed)

    def test_the_taken_offer_reaches_the_classifier(self):
        import core.actions as A
        bc = self.bc
        for user, offer, arg, want in (
                ("Yes.", SONG_OFFER, "Carry On", None),
                ("Yes, go ahead.", "Shall I play Resume, sir?", "Resume",
                 None),
                ("Yes.", "Shall I resume the movie?", "resume", "resume")):
            with self.subTest(user=user, offer=offer):
                prev = bc._begin_turn_grounding(user)
                try:
                    bc._note_turn_accepted_offer(offer)
                    with contextlib.redirect_stdout(io.StringIO()):
                        got = A._play_music_transport_op(arg)
                finally:
                    bc._end_turn_grounding(prev)
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
