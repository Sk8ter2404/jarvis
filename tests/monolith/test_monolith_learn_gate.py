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
import os
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

    def test_a_standby_wake_in_someone_elses_voice_opens_nothing(self):
        # B081 (2026-10-01): the standby wake opened the window with no voice
        # check, so after a guest's "JARVIS" the guest's UNSURE follow-ups
        # taught as "follow-up in the owner's conversation".
        self.verdicts = [(self.lg.NOT_OWNER, 0.30), (self.lg.UNSURE, 0.66)]
        with mock.patch("builtins.print") as p:
            self.bc._learn_gate_note_wake(object(), 16000)
        self.turn("I switched to oat milk")
        self.assertEqual(self.enqueued, [])
        self.assertEqual(self.voice_calls, 2)   # the wake was checked
        logged = " ".join(str(c.args[0]) for c in p.call_args_list if c.args)
        self.assertIn("standby wake did not open the window", logged)

    def test_a_standby_wake_in_the_owners_voice_still_opens_it(self):
        self.verdicts = [(self.lg.OWNER, 0.83), (self.lg.UNSURE, 0.66)]
        self.bc._learn_gate_note_wake(object(), 16000)
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
class AmbientOwnerVoiceVerdictTests(MonolithGlobalsTestCase):
    """B080 (2026-10-01): the overheard-speech learner had its own copy of
    "is this the owner" that accepted ANY enrolled name, skipping the
    memory_write permission and the reject floor the answered-turn path
    applies. Enrolling a family member made her room talk teach."""

    def _owner_voice(self, ident, may_write, floor=0.60):
        import core.voice_id as vid
        bc = self.bc
        with mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "list_enrolled",
                               return_value=["owner", "guest"]), \
             mock.patch.object(vid, "identify_speaker", return_value=ident), \
             mock.patch.object(vid, "can", return_value=may_write), \
             mock.patch.object(bc, "LEARN_VOICE_REJECT_BELOW", floor):
            return bc._ambient_owner_voice(object(), 16000)

    def test_an_enrolled_guest_without_memory_write_is_not_the_owner(self):
        self.assertEqual(self._owner_voice(("guest", 0.85), False),
                         (True, "unknown", 0.85))

    def test_a_match_under_the_owners_raised_floor_is_not_the_owner(self):
        self.assertEqual(self._owner_voice(("owner", 0.78), True, floor=0.85),
                         (True, "unknown", 0.78))

    def test_the_owner_with_memory_write_is_still_the_owner(self):
        self.assertEqual(self._owner_voice(("owner", 0.91), True),
                         (True, "owner", 0.91))

    def test_the_guests_overheard_speech_never_reaches_the_learner(self):
        import core.voice_id as vid
        bc = self.bc
        with mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", True), \
             mock.patch.object(bc, "AMBIENT_LISTEN_ENABLED", True), \
             mock.patch.object(bc, "_dialogue_gate_active", lambda: False), \
             mock.patch.object(bc, "_ambient_media_is_playing",
                               return_value=False), \
             mock.patch.object(vid, "is_available", return_value=True), \
             mock.patch.object(vid, "list_enrolled",
                               return_value=["owner", "guest"]), \
             mock.patch.object(vid, "identify_speaker",
                               return_value=("guest", 0.88)), \
             mock.patch.object(vid, "can", return_value=False), \
             mock.patch.object(bc, "_call_local_llm", return_value="PERSON"), \
             mock.patch.object(bc, "learn_from_turn") as lft:
            bc._ambient_learn_from_gated("my sister moved to Denver", {},
                                         object(), 16000, conf=_CLEAR)
        lft.assert_not_called()


@requires_monolith
class LearningAcrossAMemoryWipeTests(MonolithGlobalsTestCase):
    """B018 (2026-10-01): the learner waits up to two minutes for the owner
    to go quiet, so turns spoken just before "reset your memory" / "forget
    the last hour" + "yes" were extracted AFTER the wipe and written back
    into the memory just wiped."""

    def setUp(self):
        bc = self.bc
        self.store = {"facts": ["User likes tea"], "projects": [],
                      "topics": [], "sessions": []}

        def _load():
            return copy.deepcopy(self.store)

        def _save(m):
            self.store = copy.deepcopy(m)

        import contextlib
        for name, value in (
                ("load_memory", _load), ("save_memory", _save),
                ("_owner_vocab", lambda: frozenset()),
                ("_ltm_learn_facts", mock.MagicMock()),
                ("_rebuild_after_learning", mock.MagicMock()),
                ("_bg_local_slot",
                 lambda *_a, **_k: contextlib.nullcontext())):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(bc._lt, "background_work",
                              lambda *_a, **_k: contextlib.nullcontext())
        p.start()
        self.addCleanup(p.stop)

    def _reset_memory_via_the_action(self):
        """Run the REAL core.actions reset against this monolith, with every
        file / store side effect redirected (nothing real is touched)."""
        import tempfile
        import core.actions as A
        import core.long_term_memory as LTM
        bc = self.bc
        with tempfile.TemporaryDirectory() as td:
            mem_file = os.path.join(td, "bobert_memory.json")
            with open(mem_file, "w", encoding="utf-8") as f:
                f.write("{}")
            with mock.patch.object(A, "_bc", return_value=bc), \
                 mock.patch.object(bc, "MEMORY_FILE", mem_file), \
                 mock.patch.object(LTM, "reset_all", return_value=0), \
                 mock.patch.object(bc.pattern_memory,
                                   "reset_conversation_logs", create=True), \
                 mock.patch.object(bc, "_rebuild_prompt_now", create=True), \
                 mock.patch("builtins.print"):
                out = A._act_reset_memory()
        self.assertIn("memory reset", out)

    def test_a_turn_extracted_while_the_reset_ran_is_not_learned(self):
        bc = self.bc
        bc._learn_pending[:] = [("I have a cat", "Noted.", True, None)]
        bc._learn_worker_live[0] = True

        def _extraction(*_a, **_k):
            # The owner says "reset your memory" -> "yes" while the local
            # model is still extracting the turn above.
            self._reset_memory_via_the_action()
            return ('{"new_facts": ["User has a cat"], "new_projects": [], '
                    '"topic": ""}')

        with mock.patch.object(bc, "_llm_quick", side_effect=_extraction):
            bc._learn_worker()
        self.assertEqual(self.store["facts"], [])

    def test_a_queued_turn_is_dropped_by_the_wipe(self):
        bc = self.bc
        bc._learn_pending[:] = [("I have a cat", "Noted.", True, None)]
        bc._learn_invalidate(None)
        self.assertEqual(bc._learn_pending, [])

    def test_a_turn_still_on_the_learn_gate_queue_is_not_learned(self):
        bc = self.bc
        lg = bc._learn_gate_mod
        held, enqueued = [], []
        bc._learn_gate_state[0] = lg.LearnGate(90)
        with mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", True), \
             mock.patch.object(bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(bc, "_dialogue_gate_active", lambda: False), \
             mock.patch.object(bc, "_learn_gate_submit", held.append), \
             mock.patch.object(bc, "_learn_enqueue", enqueued.append), \
             mock.patch("builtins.print"):
            bc.learn_from_turn("jarvis I have a cat", "Noted.", {}, wake=True)
            bc._learn_invalidate(None)           # the wipe
            for item in held:
                bc._learn_gate_classify(item)
            bc.learn_from_turn("jarvis I have a dog", "Noted.", {}, wake=True)
            bc._learn_gate_classify(held[-1])
        self.assertEqual([t[0] for t in enqueued], ["jarvis I have a dog"])

    def test_overheard_speech_judged_across_the_wipe_is_not_learned(self):
        bc = self.bc
        enqueued = []

        def _judge_meanwhile(*_a, **_k):
            # The content judge waits for the background slot; the forget
            # lands in between.
            inv = getattr(bc, "_learn_invalidate", None)
            if inv is not None:
                inv(None)
            return True

        with mock.patch.object(bc, "LEARN_ONLY_FROM_OWNER", True), \
             mock.patch.object(bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(bc, "AMBIENT_LISTEN_ENABLED", True), \
             mock.patch.object(bc, "_dialogue_gate_active", lambda: False), \
             mock.patch.object(bc, "_ambient_media_is_playing",
                               return_value=False), \
             mock.patch.object(bc, "_ambient_owner_voice",
                               return_value=(True, "owner", 0.9)), \
             mock.patch.object(bc, "_ambient_should_learn_text",
                               side_effect=_judge_meanwhile), \
             mock.patch.object(bc, "_learn_gate_submit",
                               bc._learn_gate_classify), \
             mock.patch.object(bc, "_learn_enqueue", enqueued.append), \
             mock.patch("builtins.print"):
            bc._learn_gate_state[0] = bc._learn_gate_mod.LearnGate(90)
            bc._ambient_learn_from_gated("my sister moved to Denver", {},
                                         object(), 16000, conf=_CLEAR)
        self.assertEqual(enqueued, [])


@requires_monolith
class TestInjectsDoNotTeachTests(MonolithGlobalsTestCase):
    """B023 (2026-10-01): every inject counted as the owner typing, and a
    typed turn always teaches -- so Claude Code's live-verification lines
    (driver.py, say_to_jarvis) were learned as facts about the owner."""

    def _drain(self, items):
        import json
        import tempfile
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f)
            with mock.patch.object(bc, "INJECTED_COMMANDS_PATH", path):
                text = bc._drain_injected_command()
        return text, getattr(bc, "_last_inject_source", [None])[0]

    def test_a_test_inject_is_marked(self):
        self.assertEqual(
            self._drain([{"text": "what time is it", "ts": 1.0,
                          "source": "test"}]),
            ("what time is it", "test"))

    def test_the_owners_typed_injects_are_not(self):
        for item in ({"text": "hello", "ts": 1.0}, "hello"):
            with self.subTest(item=item):
                text, src = self._drain([item])
                self.assertEqual(text, "hello")
                self.assertNotEqual(src, "test")

    def test_the_main_loop_never_learns_a_test_inject(self):
        src = inspect.getsource(self.bc.main)
        guard = src.index('_last_inject_source[0] == "test"')
        learn = src.index("learn_from_turn(text, reply, memory")
        self.assertLess(guard, learn)
        self.assertIn("not learning from this turn: ", src[guard:learn])
        self.assertIn("test inject", src[guard:learn])


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
        greet_at = src.index("context_aware_greeting(", wake_at)
        # 2026-10-01 (B081): a mic wake hands its capture over for the voice
        # check; a typed wake has no capture -- and a test harness's typed
        # wake opens nothing (behaviour pinned in test_monolith_sec7). The
        # capture is the RAW (pre auto-gain) buffer the standby phase
        # publishes as _last_capture_audio (the voice-loop batch), not the
        # boosted local ``audio``: voice-ID wants natural audio.
        self.assertIn("_learn_gate_note_wake(_last_capture_audio, SAMPLE_RATE)",
                      src[wake_at:greet_at])
        self.assertIn("_learn_gate_note_wake()", src[wake_at:greet_at])


if __name__ == "__main__":
    unittest.main()
