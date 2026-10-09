"""The clone voice's render cache wired into the monolith (2026-10-05):
voice architecture C4 (persistent cache, cache-aware first line), C6 (the
quiet gate the seeding keeper reads) and C8 ([turn-timing] clone_cache /
t3_ms_tok).

The REAL client against a FAKE loopback server (tests/_clone_voice_fake.py),
the real _speak / _speak_sentences / synthesise; only Kokoro, the speaker
and the HUD are faked (the clone server tests' harness). Nothing is played,
no GPU, and the harness never sets the cache up on disk (the kick is gated
off in test mode) unless a test points it at a temporary folder.

    python -m unittest tests.monolith.test_monolith_clone_cache
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from tests.monolith.test_monolith_clone_server import _Base, _SpeakBase

OPENER = "Very good, sir."
SHORT_REPLY = "Very good, sir. The lamp is off now."
# A first line the clone plan splits anyway (over CLAUSE_SPLIT_MIN_CHARS).
SPLIT_REPLY = ("Very good, sir. The lamp in the study is off now and the "
               "heating is down to sixty eight.")
SPLIT_REST = ("The lamp in the study is off now and the heating is down to "
              "sixty eight.")
LONG_S = ("The forecast for tomorrow looks mild, with a gentle breeze from "
          "the west and grey skies all afternoon.")


class CacheAwarePlanTests(_Base):
    def setUp(self):
        super().setUp()
        self.clone_setup()
        from core import config
        self._p(config, "VOICE_CLONE_CACHE", "on")

    def test_a_cached_opener_starts_the_plan(self):
        bc = self.bc
        plain = bc._speech_chunks(SPLIT_REPLY, "clone")
        self.assertGreater(len(plain), 1)               # split anyway
        self.quiet(bc.synthesise, OPENER)             # now cached (memory)
        self.assertEqual(bc._speech_chunks(SPLIT_REPLY, "clone"),
                         [OPENER, SPLIT_REST])
        # A short reply voiced in one piece is never split for the cache
        # (a new stream per piece: ~0.4-0.7 s of silence until C5).
        self.assertEqual(bc._speech_chunks(SHORT_REPLY, "clone"),
                         [SHORT_REPLY])
        # Kokoro's plan never looks at the clone cache.
        self.assertEqual(bc._speech_chunks(SPLIT_REPLY, "kokoro"),
                         [SPLIT_REPLY])

    def test_a_struggling_clone_plans_as_if_nothing_were_cached(self):
        bc = self.bc
        self.quiet(bc.synthesise, OPENER)
        self.client._fails = 1                 # a miss since its last line
        self.assertEqual(bc._speech_chunks(SPLIT_REPLY, "clone"),
                         bc._sentence_tts.plan_clone_chunks(SPLIT_REPLY))

    def test_a_cached_long_first_sentence_is_not_split(self):
        bc = self.bc
        plan = bc._speech_chunks(LONG_S, "clone")
        self.assertEqual(getattr(plan[0], "clause", ""), "head")
        self.quiet(bc.synthesise, LONG_S)
        self.assertEqual(bc._speech_chunks(LONG_S, "clone"), [LONG_S])

    def test_shadow_and_off_keep_todays_plan(self):
        bc = self.bc
        from core import config
        self.quiet(bc.synthesise, OPENER)
        plain = bc._sentence_tts.plan_clone_chunks(SPLIT_REPLY)
        for m in ("shadow", "off"):
            with mock.patch.object(config, "VOICE_CLONE_CACHE", m):
                self.assertEqual(bc._speech_chunks(SPLIT_REPLY, "clone"),
                                 plain, m)

    def test_turn_timing_notes_where_the_line_came_from(self):
        bc = self.bc
        self.quiet(bc.synthesise, OPENER)
        names = dict(self.stats)
        self.assertEqual(names.get("clone_cache"), "miss")
        self.assertAlmostEqual(names.get("t3_ms_tok"), 4.25, places=1)
        self.stats.clear()
        n = len(self.srv.tts_texts())
        self.quiet(bc.synthesise, OPENER)
        names = dict(self.stats)
        self.assertEqual(names.get("clone_cache"), "mem")
        # Noted as None ('-'): a cached line has no T3 speed of its own.
        self.assertIn("t3_ms_tok", names)
        self.assertIsNone(names["t3_ms_tok"])
        # Where it came from is the structure, not a wall-clock bound (a
        # cached line's time is the server check, 0-2 ms, more on a busy
        # box at Idle priority).
        self.assertIsInstance(names.get("clone_ms"), int)
        self.assertEqual(len(self.srv.tts_texts()), n)     # no render
        self.assertIn("(first line, cached)", self.out)

    def test_t3_speed_is_the_first_lines_or_nothing(self):
        # A cached first line, then a later line of the same reply rendered
        # on the same thread: first value wins, so the later line's speed
        # must never stand in for the first one's.
        bc = self.bc
        self.quiet(bc.synthesise, OPENER)
        self.stats.clear()
        self.quiet(bc.synthesise, OPENER)                     # cached
        self.quiet(bc.synthesise, "The lamp is off now.")     # rendered
        first = {}
        for name, value in self.stats:
            first.setdefault(name, value)
        self.assertEqual(first["clone_cache"], "mem")
        self.assertIsNone(first["t3_ms_tok"])

    def test_a_refused_cached_take_is_tagged_refused(self):
        bc = self.bc
        out = self.cvc.Outcome(reason="error (x)", counted=True,
                               cache="refused")
        self.assertEqual(bc._clone_cache_tag(out), "refused")
        self.assertIn("cached take refused", bc._clone_line_tag(out))
        self.assertEqual(bc._clone_cache_tag(self.cvc.Outcome(reason="x")),
                         "miss")


class SpeakWithCacheTests(_SpeakBase):
    def setUp(self):
        super().setUp()
        from core import config
        self._p(config, "VOICE_CLONE_CACHE", "on")

    def test_a_reply_opening_with_a_cached_line_renders_only_the_rest(self):
        self.assertTrue(self.speak(OPENER))
        self.assertEqual(self.srv.tts_texts(), [OPENER])
        self.played.clear()
        self.assertTrue(self.speak(SPLIT_REPLY))
        self.join_workers()
        self.assertEqual(self.srv.tts_texts(), [OPENER, SPLIT_REST])
        self.assertEqual(len(self.played), 2)
        self.assertTrue(all(self.voice_of(a) == "clone" for a in self.played))
        self.assertEqual(self.kokoro_texts, [])

    def test_a_short_reply_stays_one_render(self):
        self.assertTrue(self.speak(OPENER))
        self.played.clear()
        self.assertTrue(self.speak(SHORT_REPLY))
        self.join_workers()
        self.assertEqual(self.srv.tts_texts(), [OPENER, SHORT_REPLY])
        self.assertEqual(len(self.played), 1)


class SetupAndGateTests(_Base):
    def test_the_kick_never_sets_the_cache_up(self):
        # Only main() does (at boot): the server tests drive the kick with
        # may_start forced on, and must not get a disk folder and a daemon.
        self.clone_setup(ready=False)
        bc = self.bc
        self._p(bc, "_clone_cache_setup_done", [False])
        self._p(bc, "_clone_server_may_start", return_value=True)
        setup = self._p(bc, "_clone_cache_setup")
        start = self._p(self.client, "start_async", return_value=True)
        self.assertTrue(bc._clone_server_kick())
        start.assert_called_once()
        setup.assert_not_called()
        self.assertIsNone(self.client.store.disk_dir)
        # ...while boot does set it up (a global main() references).
        self.assertIn("_clone_cache_setup", bc.main.__code__.co_names)

    def test_setup_refuses_the_harness_and_staging(self):
        self.clone_setup(ready=False)
        bc = self.bc
        self._p(bc, "_clone_cache_setup_done", [False])
        attach = self._p(self.client, "attach_cache")
        self.quiet(bc._clone_cache_setup)              # may_start: False
        attach.assert_not_called()

    def test_setup_attaches_the_folder_and_starts_the_keeper_once(self):
        self.clone_setup()
        bc = self.bc
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self._p(bc, "_clone_cache_setup_done", [False])
        self._p(bc, "_clone_server_may_start", return_value=True)
        from core import paths
        self._p(paths, "data_dir", return_value=tmp.name)
        start = self._p(self.client, "start_keeper", return_value=True)
        self.quiet(bc._clone_cache_setup)
        self.assertIn("[clone-cache] mode", self.out)
        self.quiet(bc._clone_cache_setup)
        self.assertEqual(self.out, "")                 # once
        self.assertEqual(self.client.store.disk_dir,
                         os.path.join(tmp.name, "clone_cache"))
        self.assertEqual(start.call_count, 1)
        kw = start.call_args.kwargs
        self.assertIs(kw["gate_fn"], bc._clone_seed_gate)
        self.assertIs(kw["abort_fn"], bc._clone_seed_abort)

    def _quiet_room(self):
        bc = self.bc
        now = time.monotonic()
        for name in ("_last_convo_activity", "_last_owner_turn_at",
                     "_last_owner_voice_at"):
            self._p(bc, name, [now - 600.0])
        self._p(bc, "_main_loop_started_at", [now - 900.0])
        self._p(bc, "_turn_in_progress", [False])
        self._p(bc, "_utterance_in_progress", [False])
        self._p(bc, "_filler_on_device", [False])
        self._p(bc, "last_speech_time", time.time() - 600.0)
        bc._tts_playback_active[0] = False

    def test_the_seed_gate(self):
        self.clone_setup()
        bc = self.bc
        self._quiet_room()
        self.assertIsNone(bc._clone_seed_gate())
        self.assertFalse(bc._clone_seed_abort())
        bc._turn_in_progress[0] = True
        self.assertEqual(bc._clone_seed_gate(), "mid-turn")
        self.assertTrue(bc._clone_seed_abort())
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = True           # he started talking
        self.assertEqual(bc._clone_seed_gate(), "mid-turn")
        self.assertTrue(bc._clone_seed_abort())
        bc._utterance_in_progress[0] = False
        with bc._SPEAK_LOCK:
            self.assertEqual(bc._clone_seed_gate(), "speaking")
            self.assertTrue(bc._clone_seed_abort())
        bc._last_owner_turn_at[0] = time.monotonic() - 10.0
        self.assertEqual(bc._clone_seed_gate(), "owner active")
        bc._last_owner_turn_at[0] = time.monotonic() - 600.0
        self._p(bc, "last_speech_time", time.time() - 2.0)
        self.assertEqual(bc._clone_seed_gate(), "just spoke")
        self._p(bc, "last_speech_time", time.time() - 600.0)
        bc._main_loop_started_at[0] = 0.0
        self.assertEqual(bc._clone_seed_gate(), "booting")
        bc._main_loop_started_at[0] = time.monotonic() - 900.0
        self.rest_clone()
        self.assertEqual(bc._clone_seed_gate(), "clone not speaking")

    def test_every_input_of_the_seed_gate_closes_it(self):
        # Each input alone (the others quiet) closes the gate: a seed render
        # must never queue on the GPU ahead of a line he is about to hear.
        self.clone_setup()
        bc = self.bc
        now = time.monotonic
        cases = (
            ("_filler_on_device", [True], "speaking"),
            ("_last_owner_voice_at", [now() - 10.0], "owner active"),
            ("_last_convo_activity", [now() - 10.0], "owner active"),
            ("_last_owner_turn_at", [now() - 10.0], "owner active"),
        )
        for name, value, why in cases:
            self._quiet_room()
            self.assertIsNone(bc._clone_seed_gate(), name)
            with mock.patch.object(bc, name, value):
                self.assertEqual(bc._clone_seed_gate(), why, name)
        self._quiet_room()
        bc._tts_playback_active[0] = True
        try:
            self.assertEqual(bc._clone_seed_gate(), "speaking")
            self.assertTrue(bc._clone_seed_abort())
        finally:
            bc._tts_playback_active[0] = False
        self.assertIsNone(bc._clone_seed_gate())
        game = types.SimpleNamespace(_st=types.SimpleNamespace(active=True))
        with mock.patch.dict(sys.modules, {"skill_game_mode": game}):
            self.assertEqual(bc._clone_seed_gate(), "game")
            game._st.active = False
            self.assertIsNone(bc._clone_seed_gate())

    def test_history_lines_are_cleaned_like_speech(self):
        bc = self.bc
        out = bc._clone_seed_clean(
            "[intent:amused] Certainly, sir. [ACTION: lights_off] **Done**.")
        self.assertIn("Certainly, sir.", out)
        self.assertNotIn("[", out)
        self.assertNotIn("*", out)
        self.assertNotIn("lights_off", out)
        self.assertEqual(bc._clone_seed_clean(None), "")

    def test_history_reads_the_logs_and_the_episode_store(self):
        bc = self.bc
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        logs = os.path.join(tmp.name, "logs")
        os.makedirs(logs)
        with open(os.path.join(logs, "session_2026-01-01_00-00-00.log"), "w",
                  encoding="utf-8") as f:
            f.write("[00:00:01]   JARVIS: Right away, sir.\n")
        os.makedirs(os.path.join(tmp.name, "data", "long_term_memory"))
        with open(os.path.join(tmp.name, "data", "long_term_memory",
                               "episodes.jsonl"), "w", encoding="utf-8") as f:
            f.write('{"role": "assistant", "text": "Of course, sir."}\n')
        self._p(bc, "LOGS_DIR", logs)
        from core import paths
        self._p(paths, "data_dir", return_value=os.path.join(tmp.name,
                                                             "data"))
        self.assertEqual(bc._clone_seed_history(),
                         [["Right away, sir."], ["Of course, sir."]])

    def test_forget_voice_line_is_spoken_verbatim(self):
        self.assertIn("forget_voice_line", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)


class KeeperThreadHygieneTests(_Base):
    def test_one_keeper_thread_however_often_it_is_asked(self):
        self.clone_setup()
        bc = self.bc
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.assertTrue(self.client.attach_cache(tmp.name))
        from core import clone_seed
        ticks = []
        gate = threading.Event()

        def fake_loop(self_k):
            ticks.append(1)
            gate.wait(5.0)
        self.addCleanup(gate.set)
        with mock.patch.object(clone_seed.CacheKeeper, "_loop", fake_loop):
            for _ in range(5):
                self.client.start_keeper(gate_fn=bc._clone_seed_gate)
            time.sleep(0.1)
        self.assertEqual(len(ticks), 1)
        alive = [t for t in threading.enumerate()
                 if t.name == "clone-cache-keeper" and t.is_alive()]
        self.assertGreaterEqual(len(alive), 1)
        gate.set()


if __name__ == "__main__":
    unittest.main()
