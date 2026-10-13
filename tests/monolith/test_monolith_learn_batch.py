"""Batched ambient learning wired into the monolith (2026-10-12).

With LEARN_BATCH_ENABLED the three background learners stop making their own
local-model requests:

* learn_from_turn's memory extraction (one call per owner turn);
* the ambient content judge (one call per overheard line);
* the multimodal extractor's periodic pass (skills/ambient_multimodal_extract)

and queue instead. ONE request goes out when the owner has been quiet for
LEARN_BATCH_QUIET_S, never while an utterance or turn is in progress, and
never ahead of an owner turn. Off, every caller behaves exactly as before
(the whole older monolith suite runs with it off: see _monolith_harness).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)


class _Fake:
    """A fake model behind the batcher: records requests, answers from a
    script."""

    def __init__(self):
        self.calls = []
        self.reply = lambda system, user, mt: ""

    def __call__(self, system, user, max_tokens=200):
        self.calls.append((system, user, max_tokens))
        return self.reply(system, user, max_tokens)


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "LEARN_BATCH_ENABLED", True)
        self._p(bc, "LEARN_BATCH_INTERVAL_S", 0.0)
        self._p(bc, "LEARN_BATCH_QUIET_S", 90.0)
        self._p(bc, "LEARN_BATCH_MAX_JOBS", 20)
        self._p(bc, "LEARN_EVERY_TURN", True)
        self._p(bc, "LEARN_ONLY_FROM_OWNER", False)
        self._p(bc, "_dialogue_gate_active", return_value=False)
        self._p(bc, "_chat_takes_local_branch", return_value=False)
        self.fake = _Fake()
        self._p(bc, "_llm_quick", self.fake)
        self.logs = []
        batcher = bc._learn_batch_mod.LearnBatcher(
            bc._learn_batch_llm, bc._learn_batch_quiet,
            enabled=bc._learn_batch_on,
            interval_s=lambda: bc.LEARN_BATCH_INTERVAL_S,
            max_jobs=lambda: bc.LEARN_BATCH_MAX_JOBS,
            log=self.logs.append, poll_s=3600.0)
        batcher._ensure_thread = lambda: None       # driven by hand
        self.batcher = batcher
        saved = bc._learn_batcher_state[0]
        bc._learn_batcher_state[0] = batcher
        self.addCleanup(bc._learn_batcher_state.__setitem__, 0, saved)
        saved_q = dict(bc._learn_batch_queued)
        self.addCleanup(lambda: (bc._learn_batch_queued.clear(),
                                 bc._learn_batch_queued.update(saved_q)))
        bc._learn_batch_queued.update({True: False, False: False})
        saved_p = list(bc._learn_pending)
        bc._learn_pending.clear()
        self.addCleanup(lambda: (bc._learn_pending.clear(),
                                 bc._learn_pending.extend(saved_p)))
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc._last_convo_activity[0] = 0.0
        bc._last_owner_turn_at[0] = 0.0
        self.applied = []
        self._p(bc, "_learn_apply",
                side_effect=lambda text, batch=None, epoch=None:
                self.applied.append((text, list(batch or []), epoch)))

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _flush(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.batcher.flush_once(force=True)


class QuietPredicateTests(_Base):
    def test_quiet_when_nothing_has_happened(self):
        self.assertTrue(self.bc._learn_batch_quiet())

    def test_not_quiet_during_an_utterance_or_a_turn(self):
        bc = self.bc
        bc._utterance_in_progress[0] = True
        self.assertFalse(bc._learn_batch_quiet())
        bc._utterance_in_progress[0] = False
        bc._turn_in_progress[0] = True
        self.assertFalse(bc._learn_batch_quiet())

    def test_not_quiet_right_after_a_reply_then_quiet_after_the_window(self):
        bc = self.bc
        bc._last_convo_activity[0] = time.monotonic() - 10.0
        self.assertFalse(bc._learn_batch_quiet())
        bc._last_convo_activity[0] = time.monotonic() - 120.0
        self.assertTrue(bc._learn_batch_quiet())
        bc._last_owner_turn_at[0] = time.monotonic() - 5.0
        self.assertFalse(bc._learn_batch_quiet())

    def test_default_off_switch_restores_the_old_path(self):
        bc = self.bc
        bc.LEARN_BATCH_ENABLED = False
        self.assertFalse(bc._learn_batch_on())
        self.assertFalse(bc.learn_batch_submit_text(
            "t", "s", "u", 4, lambda text: None))
        self.assertEqual(self.batcher.pending(), 0)


class LearnExtractionTests(_Base):
    def test_turns_queue_without_a_worker_and_go_out_as_one_request(self):
        bc = self.bc
        mem = bc._empty_memory()
        started = []
        self._p(bc.threading, "Thread",
                side_effect=lambda *a, **k: started.append(k) or mock.Mock())
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(4):
                bc.learn_from_turn(f"owner says {i}", f"reply {i}", mem)
        self.assertEqual(started, [], "no per-turn worker thread")
        self.assertEqual(len(bc._learn_pending), 4)
        self.assertEqual(self.batcher.pending(), 1, "one job for the kind")
        self.assertEqual(self.fake.calls, [])
        self.fake.reply = lambda s, u, m: json.dumps(
            {"new_facts": ["User has a cat"], "new_projects": [],
             "topic": "pets"})
        res = self._flush()
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(res["jobs"], 1)
        self.assertEqual(len(self.applied), 1)
        text, batch, _epoch = self.applied[0]
        self.assertEqual([t[0] for t in batch],
                         [f"owner says {i}" for i in range(4)])
        self.assertEqual(json.loads(text)["new_facts"], ["User has a cat"])
        self.assertEqual(bc._learn_pending, [])
        # the plain single/multi-turn extraction prompt is unchanged
        system, user, mt = self.fake.calls[0]
        self.assertIn("Turn 1:\nUser said: owner says 0", user)
        self.assertEqual(mt, 400)

    def test_overheard_and_owner_turns_stay_in_separate_jobs(self):
        bc = self.bc
        mem = bc._empty_memory()
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I love tea", "ok", mem)
            bc.learn_from_turn("the neighbour likes jazz", "", mem,
                               owner_directed=False)
        self.assertEqual(self.batcher.pending(), 2)

        def reply(system, user, mt):
            rows = []
            for line in user.splitlines():
                if line.startswith("=== JOB "):
                    rows.append({"id": int(line.split()[2]),
                                 "output": {"new_facts": [], "new_projects": [],
                                            "topic": ""}})
            return json.dumps({"results": rows})
        self.fake.reply = reply
        self._flush()
        self.assertEqual(len(self.fake.calls), 1, "both kinds, ONE request")
        self.assertEqual(len(self.applied), 2)
        kinds = sorted(bool(b[0][2]) for _t, b, _e in self.applied)
        self.assertEqual(kinds, [False, True])

    def test_owner_active_keeps_the_turns_queued_for_later(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I have a dog", "Noted.", bc._empty_memory())
        bc._turn_in_progress[0] = True          # he is mid-turn
        self.assertIsNone(self._flush())
        self.assertEqual(self.fake.calls, [], "never in front of his turn")
        self.assertEqual(len(bc._learn_pending), 1)
        bc._turn_in_progress[0] = False
        self.fake.reply = lambda s, u, m: '{"new_facts": [], "topic": ""}'
        self._flush()
        self.assertEqual(len(self.applied), 1)

    def test_a_deferred_send_hands_the_taken_turns_back(self):
        """He starts talking after the batch was assembled but before the
        send: the turns return to the queue and the job stays queued."""
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I have a dog", "Noted.", bc._empty_memory())
        n = {"calls": 0}

        def quiet():
            n["calls"] += 1
            return n["calls"] == 1
        self.batcher._quiet = quiet
        self.assertIsNone(self._flush())
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(len(bc._learn_pending), 1)
        self.assertTrue(bc._learn_batch_queued[True])
        self.assertEqual(self.batcher.pending(), 1)

    def test_a_memory_wipe_while_queued_drops_the_turns(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I have a dog", "Noted.", bc._empty_memory())
            bc._learn_invalidate(None)
        self.assertEqual(bc._learn_pending, [])
        self.assertIsNone(self._flush())
        self.assertEqual(self.fake.calls, [])

    def test_more_turns_than_one_extraction_holds_ride_the_next_flush(self):
        bc = self.bc
        cap = bc._LEARN_BATCH_TURNS_MAX
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(cap + 3):
                bc.learn_from_turn(f"fact number {i}", "ok",
                                   bc._empty_memory())
        self.fake.reply = lambda s, u, m: '{"new_facts": [], "topic": ""}'
        self._flush()
        self.assertEqual(len(self.applied[0][1]), cap)
        self.assertEqual(len(bc._learn_pending), 3)
        self.assertEqual(self.batcher.pending(), 1, "a job for the leftovers")
        self._flush()
        self.assertEqual(len(self.applied[1][1]), 3)

    def test_off_runs_the_old_worker(self):
        bc = self.bc
        bc.LEARN_BATCH_ENABLED = False
        started = []

        class _T:
            def __init__(self, *a, **k):
                started.append(k.get("name"))

            def start(self):
                pass
        self._p(bc.threading, "Thread", _T)
        bc._learn_worker_live[0] = False
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I have a dog", "Noted.", bc._empty_memory())
        self.assertEqual(started, ["learn-from-turn"])
        self.assertEqual(self.batcher.pending(), 0)
        bc._learn_worker_live[0] = False


class AmbientJudgeTests(_Base):
    def _ambient(self, text="my sister moved to Denver last spring"):
        bc = self.bc
        self._p(bc, "AMBIENT_LISTEN_ENABLED", True)
        self._p(bc, "_ambient_media_is_playing", return_value=False)
        self._p(bc, "_ambient_owner_voice", return_value=(True, "owner", 0.9))
        self.ingested = []
        self._p(bc, "learn_from_turn",
                side_effect=lambda *a, **k: self.ingested.append((a, k)))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            bc._ambient_learn_from_gated(text, {}, object(), 16000)
        return out.getvalue()

    def _answer(self, word):
        self.fake.reply = lambda s, u, m: word

    def test_the_judge_waits_for_the_batch_then_ingests_a_person(self):
        self._ambient()
        self.assertEqual(self.ingested, [], "not ingested before the verdict")
        self.assertEqual(self.fake.calls, [], "and no judge call of its own")
        self.assertEqual(self.batcher.pending(), 1)
        self._answer("PERSON")
        self._flush()
        self.assertEqual(len(self.ingested), 1)
        self.assertEqual(self.ingested[0][0][0],
                         "my sister moved to Denver last spring")
        self.assertIs(self.ingested[0][1]["owner_directed"], False)

    def test_a_media_verdict_skips_the_line(self):
        self._ambient()
        self._answer("MEDIA")
        self._flush()
        self.assertEqual(self.ingested, [])

    def test_judge_fails_open_when_the_model_gives_nothing(self):
        self._ambient()
        self.fake.reply = lambda s, u, m: ""
        for _ in range(3):          # MAX_ATTEMPTS: retried, then answered ""
            self._flush()
        self.assertEqual(len(self.ingested), 1, "fail open like the single "
                         "judge")

    def test_a_judge_and_an_extraction_share_one_request(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            bc.learn_from_turn("I have a cat", "Noted.", bc._empty_memory())
        self._ambient()

        def reply(system, user, mt):
            rows = []
            for part in user.split("=== JOB ")[1:]:
                jid = int(part.split()[0])
                rows.append({"id": jid, "output": "PERSON"
                             if "Transcript:" in part
                             else {"new_facts": []}})
            return json.dumps({"results": rows})
        self.fake.reply = reply
        self._flush()
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(len(self.ingested), 1)
        self.assertEqual(len(self.applied), 1)

    def test_low_confidence_is_dropped_before_any_judge_job(self):
        bc = self.bc
        self._p(bc, "is_valid_speech", return_value=(False, "low confidence"))
        self._p(bc, "AMBIENT_LISTEN_ENABLED", True)
        self._p(bc, "_ambient_media_is_playing", return_value=False)
        self._p(bc, "_ambient_owner_voice", return_value=(True, "owner", 0.9))
        ing = self._p(bc, "learn_from_turn")
        with contextlib.redirect_stdout(io.StringIO()):
            bc._ambient_learn_from_gated("a hallucinated line of text", {},
                                         object(), 16000, conf={"x": 1})
        self.assertEqual(self.batcher.pending(), 0)
        ing.assert_not_called()

    def test_off_judges_inline_as_before(self):
        bc = self.bc
        bc.LEARN_BATCH_ENABLED = False
        self._p(bc, "_ambient_content_is_media", return_value=False)
        self._ambient()
        self.assertEqual(len(self.ingested), 1)
        self.assertEqual(self.batcher.pending(), 0)

    def test_the_judge_prompt_and_verdict_are_shared_with_the_inline_path(self):
        bc = self.bc
        self.assertIn("MEDIA or PERSON", bc._AMBIENT_JUDGE_SYSTEM)
        self.assertTrue(bc._ambient_judge_says_media("Answer: MEDIA."))
        self.assertFalse(bc._ambient_judge_says_media("PERSON"))
        self.assertFalse(bc._ambient_judge_says_media(""))
        self.assertFalse(bc._ambient_judge_says_media("MEDIA or PERSON"))


class ExtractorSkillTests(_Base):
    def setUp(self):
        super().setUp()
        import importlib.util
        path = os.path.join(_PROJECT, "skills", "ambient_multimodal_extract.py")
        spec = importlib.util.spec_from_file_location(
            "ame_under_test", path)
        self.ame = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ame)
        self._p(self.ame, "_get_bobert", return_value=self.bc)
        self._p(self.ame, "_append_extract_log")
        self.merged = []
        self._p(self.ame, "_merge_into_memory",
                side_effect=lambda ex: self.merged.append(ex) or (1, 0))
        entry = {"ts": time.time(), "source": "mic",
                 "text": "I am building a rover for school"}
        self._p(self.ame, "_tail_jsonl",
                side_effect=lambda path, n: [entry]
                if path == self.ame._AUDIO_JSONL else [])

    def test_the_periodic_pass_joins_the_batch(self):
        ame = self.ame
        self.assertTrue(ame._run_once_batched())
        self.assertEqual(self.fake.calls, [], "no extractor call of its own")
        self.assertEqual(self.batcher.pending(), 1)
        # a second tick while one waits does not stack another window
        self.assertTrue(ame._run_once_batched())
        self.assertEqual(self.batcher.pending(), 1)
        self.fake.reply = lambda s, u, m: json.dumps(
            {"new_facts": ["User is building a rover"], "new_projects": []})
        self._flush()
        self.assertEqual(self.merged,
                         [{"new_facts": ["User is building a rover"],
                           "new_projects": []}])
        self.assertFalse(ame._batch_job_waiting[0])
        self.assertEqual(ame._facts_added_total >= 1, True)

    def test_off_returns_false_so_the_old_path_runs(self):
        self.bc.LEARN_BATCH_ENABLED = False
        self.assertFalse(self.ame._run_once_batched())
        self.assertEqual(self.batcher.pending(), 0)

    def test_an_empty_window_needs_no_call(self):
        self._p(self.ame, "_tail_jsonl", return_value=[])
        self.assertTrue(self.ame._run_once_batched())
        self.assertEqual(self.batcher.pending(), 0)

    def test_parse_extract_reads_a_noisy_reply(self):
        self.assertEqual(self.ame._parse_extract('ok {"new_facts": ["a"]} x'),
                         {"new_facts": ["a"]})
        self.assertEqual(self.ame._parse_extract(""), {})
        self.assertEqual(self.ame._parse_extract(None), {})


if __name__ == "__main__":
    unittest.main()
