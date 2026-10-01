"""Monolith wiring for owner-only learning (core/learn_gate.py).

THE LIVE FAILURE (2026-09-30): with no wake word required, JARVIS answered a
phone call in the room, and because learn_from_turn treated every ANSWERED
turn as the owner's, the other person's speech became ten "facts about the
user" in one afternoon.

Pinned here, with LEARN_ONLY_FROM_OWNER on:
  * learn_from_turn sends every turn through the learn gate; a turn that was
    not typed, not led by the wake word, not in the owner's voice and not a
    follow-up never reaches the extractor; a confidently-other voice never
    does, even after "JARVIS";
  * the voiceprint check maps core.voice_id's answer (and the memory_write
    permission) to a verdict, and is skipped for typed turns;
  * a standby wake opens the follow-up window;
  * merge_memory keeps an automated learner's facts only when its provenance
    vouches for the owner (owner-directed or owner voice);
  * the ambient learner skips speech the voiceprint cannot attribute, without
    spending the content judge's local-LLM call;
  * the main loop passes typed / wake / the raw capture.
And with it off (the shipped default) nothing changes.

GENERIC fixtures only; no real memory file, voiceprint or audio is touched.

    python -m unittest tests.monolith.test_monolith_learn_gate
"""
from __future__ import annotations

import copy
import inspect
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_CLEAR = {"no_speech_prob": 0.05, "avg_logprob": -0.2}


@requires_monolith
class _GateBase(MonolithGlobalsTestCase):
    """Gate on, classification inline (no thread), extraction captured."""

    def setUp(self):
        bc = self.bc
        self.lg = bc._learn_gate_mod
        self.enqueued = []
        self.verdicts = []          # what _learn_voice_verdict returns, in order
        self.voice_calls = 0

        def _verdict(audio, sr):
            self.voice_calls += 1
            return self.verdicts.pop(0) if self.verdicts else (
                self.lg.UNAVAILABLE, 0.0)

        for name, value in (
                ("LEARN_ONLY_FROM_OWNER", True),
                ("LEARN_EVERY_TURN", True),
                ("_dialogue_gate_active", lambda: False),
                ("_learn_gate_submit", bc._learn_gate_classify),
                ("_learn_enqueue", self.enqueued.append),
                ("_learn_voice_verdict", _verdict)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        bc._learn_gate_state[0] = bc._learn_gate_mod.LearnGate(90)

    def turn(self, text="we fly out Tuesday", **kw):
        kw.setdefault("audio", object())
        kw.setdefault("sample_rate", 16000)
        self.bc.learn_from_turn(text, "Noted.", {}, conf=_CLEAR, **kw)


class LearnFromTurnGateTests(_GateBase):
    def test_the_live_failure_someone_else_answered_does_not_teach(self):
        self.verdicts = [(self.lg.NOT_OWNER, 0.41)]
        self.turn()
        self.assertEqual(self.enqueued, [])

    def test_unaddressed_speech_with_no_voice_evidence_does_not_teach(self):
        for verdict in ((self.lg.UNAVAILABLE, 0.0), (self.lg.UNSURE, 0.66)):
            with self.subTest(verdict=verdict[0]):
                self.verdicts = [verdict]
                self.turn()
        self.assertEqual(self.enqueued, [])

    def test_someone_else_saying_jarvis_still_does_not_teach(self):
        self.verdicts = [(self.lg.NOT_OWNER, 0.30)]
        self.turn("jarvis remember I have a dentist appointment", wake=True)
        self.assertEqual(self.enqueued, [])

    def test_typed_turns_teach_without_a_voice_check(self):
        self.turn(injected=True, audio=None, sample_rate=0)
        self.assertEqual(len(self.enqueued), 1)
        self.assertEqual(self.voice_calls, 0)

    def test_wake_word_turns_teach(self):
        self.turn("jarvis I moved to a standing desk", wake=True)
        self.assertEqual(len(self.enqueued), 1)

    def test_owner_voice_teaches_and_carries_the_verdict(self):
        self.verdicts = [(self.lg.OWNER, 0.81)]
        self.turn()
        u, a, owner, conf, voice = self.enqueued[0]
        self.assertTrue(owner)
        self.assertEqual(conf, _CLEAR)
        self.assertEqual(voice, self.lg.OWNER)

    def test_a_follow_up_in_the_owners_conversation_teaches(self):
        self.turn("jarvis what's on my calendar", wake=True)
        self.turn("and move the dentist to Friday")
        self.assertEqual(len(self.enqueued), 2)

    def test_a_standby_wake_opens_the_window(self):
        self.bc._learn_gate_note_wake()
        self.turn("I switched to oat milk")
        self.assertEqual(len(self.enqueued), 1)

    def test_overheard_speech_needs_the_owner_voice(self):
        self.bc.learn_from_turn("we should repaint", "", {},
                                owner_directed=False, voice=self.lg.UNAVAILABLE)
        self.assertEqual(self.enqueued, [])
        self.bc.learn_from_turn("we should repaint", "", {},
                                owner_directed=False, voice=self.lg.OWNER)
        self.assertEqual(len(self.enqueued), 1)
        self.assertFalse(self.enqueued[0][2])
        self.assertEqual(self.voice_calls, 0)  # the caller's verdict is used

    def test_the_log_line_never_carries_the_turn_text(self):
        self.verdicts = [(self.lg.NOT_OWNER, 0.41)]
        with mock.patch("builtins.print") as p:
            self.turn("my secret plan is zorblat")
        logged = " ".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("[learn-gate]", logged)
        self.assertNotIn("zorblat", logged)


@requires_monolith
class GateThreadTests(MonolithGlobalsTestCase):
    def test_the_real_classifier_thread_keeps_turn_order(self):
        # No inline shortcut: learn_from_turn -> queue -> learn-gate thread.
        import threading
        import time as _time
        bc = self.bc
        lg = bc._learn_gate_mod
        got, done = [], threading.Event()

        def _enqueue(turn):
            got.append(turn[0])
            if len(got) == 2:
                done.set()

        bc._learn_gate_state[0] = lg.LearnGate(90)
        with mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", True), \
             mock.patch.object(bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(bc, "_dialogue_gate_active", lambda: False), \
             mock.patch.object(bc, "_learn_enqueue", _enqueue), \
             mock.patch.object(bc, "_learn_voice_verdict",
                               return_value=(lg.UNAVAILABLE, 0.0)):
            bc.learn_from_turn("follow-up before any wake", "", {})
            bc.learn_from_turn("jarvis first real turn", "", {}, wake=True)
            bc.learn_from_turn("a follow-up", "", {})
            self.assertTrue(done.wait(5.0), "the learn-gate thread never ran")
            _time.sleep(0.05)
        # The pre-wake turn stayed out even though it was classified by a
        # thread that saw the wake right after it.
        self.assertEqual(got, ["jarvis first real turn", "a follow-up"])


@requires_monolith
class GateOffTests(MonolithGlobalsTestCase):
    def test_off_queues_every_turn_as_before_and_checks_no_voice(self):
        bc = self.bc
        got = []
        with mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", False), \
             mock.patch.object(bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(bc, "_dialogue_gate_active", lambda: False), \
             mock.patch.object(bc, "_learn_enqueue", got.append), \
             mock.patch.object(bc, "_learn_voice_verdict") as vv, \
             mock.patch.object(bc, "_learn_gate_submit") as sub:
            bc.learn_from_turn("anything at all", "ok", {}, conf=_CLEAR,
                               audio=object(), sample_rate=16000)
        self.assertEqual(got, [("anything at all", "ok", True, _CLEAR)])
        vv.assert_not_called()
        sub.assert_not_called()

    def test_off_a_standby_wake_queues_nothing(self):
        with mock.patch.object(self.bc, "LEARN_ONLY_FROM_OWNER", False), \
             mock.patch.object(self.bc, "_learn_gate_submit") as sub:
            self.bc._learn_gate_note_wake()
        sub.assert_not_called()

    def test_the_shipped_default_is_off(self):
        import core.config as cfg
        src = inspect.getsource(cfg)
        self.assertIn("LEARN_ONLY_FROM_OWNER = False", src)


@requires_monolith
class VoiceVerdictWiringTests(MonolithGlobalsTestCase):
    def _verdict(self, *, enrolled=("owner",), available=True,
                 ident=("owner", 0.8), may_write=True, audio="pcm", sr=16000):
        import core.voice_id as vid
        with mock.patch.object(vid, "list_enrolled",
                               return_value=list(enrolled)), \
             mock.patch.object(vid, "is_available", return_value=available), \
             mock.patch.object(vid, "identify_speaker",
                               return_value=ident) as ident_mock, \
             mock.patch.object(vid, "can", return_value=may_write):
            out = self.bc._learn_voice_verdict(audio, sr)
        return out, ident_mock

    def test_matched_owner(self):
        (v, s), _ = self._verdict()
        self.assertEqual((v, s), (self.bc._learn_gate_mod.OWNER, 0.8))

    def test_matched_guest_without_memory_write_is_not_the_owner(self):
        (v, _s), _ = self._verdict(may_write=False)
        self.assertEqual(v, self.bc._learn_gate_mod.NOT_OWNER)

    def test_low_score_is_someone_else(self):
        (v, _s), _ = self._verdict(ident=(None, 0.35))
        self.assertEqual(v, self.bc._learn_gate_mod.NOT_OWNER)

    def test_nobody_enrolled_or_no_audio_is_unavailable_and_embeds_nothing(self):
        for kw in ({"enrolled": ()}, {"audio": None}, {"sr": 0},
                   {"available": False}):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                (v, _s), ident = self._verdict(**kw)
                self.assertEqual(v, self.bc._learn_gate_mod.UNAVAILABLE)
                ident.assert_not_called()

    def test_a_voice_id_crash_is_unavailable_not_a_raise(self):
        import core.voice_id as vid
        with mock.patch.object(vid, "list_enrolled", side_effect=OSError):
            v, _s = self.bc._learn_voice_verdict("pcm", 16000)
        self.assertEqual(v, self.bc._learn_gate_mod.UNAVAILABLE)


@requires_monolith
class MergeMemoryFactGateTests(MonolithGlobalsTestCase):
    def setUp(self):
        self.store = {"facts": [], "projects": [], "topics": [], "sessions": []}

        def _load():
            return copy.deepcopy(self.store)

        def _save(m):
            self.store = m

        for name, value in (("load_memory", _load), ("save_memory", _save),
                            ("_owner_vocab", lambda: frozenset()),
                            ("_ltm_learn_facts", mock.MagicMock())):
            p = mock.patch.object(self.bc, name, value)
            p.start()
            self.addCleanup(p.stop)

    def merge(self, prov, on=True):
        with mock.patch.object(self.bc, "LEARN_ONLY_FROM_OWNER", on):
            return self.bc.merge_memory(new_facts=["User likes tea"],
                                        provenance=prov)[0]

    def test_the_ambient_extractor_teaches_no_facts(self):
        self.assertEqual(self.merge({"owner_directed": False,
                                     "source": "ambient extractor"}), [])
        self.assertEqual(self.store["facts"], [])

    def test_overheard_owner_voice_keeps_its_facts(self):
        self.assertEqual(self.merge({"owner_directed": False,
                                     "owner_voice": True}), ["User likes tea"])

    def test_owner_directed_keeps_its_facts(self):
        self.assertEqual(self.merge({"owner_directed": True,
                                     "turn_text": "I like tea",
                                     "conf": _CLEAR}), ["User likes tea"])

    def test_a_deliberate_write_without_provenance_is_unaffected(self):
        self.assertEqual(self.merge(None), ["User likes tea"])

    def test_off_overheard_facts_land_as_before(self):
        self.assertEqual(self.merge({"owner_directed": False}, on=False),
                         ["User likes tea"])


@requires_monolith
class ProvenanceTests(MonolithGlobalsTestCase):
    def test_owner_voice_reaches_the_provenance(self):
        lg = self.bc._learn_gate_mod
        prov = self.bc._learn_provenance([("x", "", False, None, lg.OWNER)])
        self.assertTrue(prov["owner_voice"])
        self.assertFalse(prov["owner_directed"])

    def test_older_queue_entries_have_no_owner_voice(self):
        prov = self.bc._learn_provenance([("x", "", False, None)])
        self.assertFalse(prov["owner_voice"])


@requires_monolith
class AmbientPathTests(MonolithGlobalsTestCase):
    def _gated(self, vid, on=True):
        bc = self.bc
        with mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", on), \
             mock.patch.object(bc, "AMBIENT_LISTEN_ENABLED", True), \
             mock.patch.object(bc, "_dialogue_gate_active", lambda: False), \
             mock.patch.object(bc, "_ambient_media_is_playing",
                               return_value=False), \
             mock.patch.object(bc, "_ambient_owner_voice", return_value=vid), \
             mock.patch.object(bc, "_call_local_llm",
                               return_value="PERSON") as judge, \
             mock.patch.object(bc, "learn_from_turn") as lft:
            bc._ambient_learn_from_gated("we should repaint the kitchen soon",
                                         {}, conf=_CLEAR)
        return lft, judge

    def test_unattributable_speech_is_skipped_without_the_llm_judge(self):
        lft, judge = self._gated((False, "unavailable", 0.0))
        lft.assert_not_called()
        judge.assert_not_called()

    def test_owner_voice_is_passed_on_with_its_verdict(self):
        lft, _ = self._gated((True, "owner", 0.9))
        lft.assert_called_once()
        self.assertEqual(lft.call_args.kwargs["voice"],
                         self.bc._learn_gate_mod.OWNER)
        self.assertIs(lft.call_args.kwargs["owner_directed"], False)

    def test_off_the_content_heuristic_path_still_learns(self):
        lft, _ = self._gated((False, "unavailable", 0.0), on=False)
        lft.assert_called_once()


@requires_monolith
class MainLoopWiringTests(MonolithGlobalsTestCase):
    def test_the_answered_turn_passes_typed_wake_and_its_capture(self):
        src = inspect.getsource(self.bc.main)
        self.assertIn("_typed = _injected_text is not None", src)
        for piece in ("injected=_typed",
                      "wake=_text_has_wake_prefix(text)",
                      "audio=None if _typed else _last_capture_audio",
                      "sample_rate=0 if _typed else _last_capture_sr"):
            self.assertIn(piece, src)

    def test_a_standby_wake_notes_the_gate(self):
        src = inspect.getsource(self.bc._handle_sleep_standby)
        wake_at = src.index('print("  [wake] Waking up")')
        self.assertIn("_learn_gate_note_wake()", src[wake_at:wake_at + 400])


if __name__ == "__main__":
    unittest.main()
