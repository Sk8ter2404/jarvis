"""Tests for core/topic_hygiene.py — the write gate for auto-learned topics
and projects, plus the read-time grounding that routes "what am I working on"
to a real source.

THE LIVE FAILURE (2026-09-29): asked "what am I working on lately", JARVIS
named a "mystery" and a "project" that were Whisper mis-hearings of TV / room
audio, learned as a topic/project after ONE overheard line and then rendered
into every system prompt. These pin the four rules that stop it (see the
module docstring) and the prompt routing for the grounded answer.

GENERIC fixtures only: made-up non-words ("zorblat", "flemwick") stand in for
the mis-heard phrases; nothing here reads or writes a real data file (temp
dirs only).

    python -m unittest tests.test_topic_hygiene
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from core import topic_hygiene as th

_CLEAR = {"no_speech_prob": 0.05, "avg_logprob": -0.2}
# The synthetic values the monolith gives a typed / injected turn.
_TYPED = {"no_speech_prob": 0.0, "avg_logprob": -0.1}


class NonWordRuleTests(unittest.TestCase):
    """Rule 4: a label made MOSTLY of non-words is not a real topic."""

    def test_non_word_topic_is_flagged(self):
        self.assertIn("zorblat", th.garbled_reason("Zorblat mystery"))
        self.assertTrue(th.garbled_reason("Flemwick grundlefax"))

    def test_real_phrases_pass(self):
        for label in ("garden shed", "Plans for the weekend",
                      "Building a robot arm", "espresso machine",
                      "Home automation with smart lights",
                      "Sourdough baking", "Neighborhood cleanup",
                      "carburettor rebuild on the mower",
                      "Microcontroller firmware flashing",
                      "Vintage car restoration", "Debt payoff planning"):
            with self.subTest(label=label):
                self.assertEqual(th.garbled_reason(label), "")

    def test_one_rare_word_in_a_real_phrase_passes(self):
        # "mostly": one unknown of three content words is not garbled.
        self.assertEqual(th.garbled_reason("zorblat garden shed"), "")

    def test_owner_vocabulary_exempts_his_own_proper_nouns(self):
        vocab = th.owner_vocab_from_texts(
            ["how is the zorblat build going", "order more parts for zorblat"])
        self.assertIn("zorblat", vocab)
        self.assertEqual(th.garbled_reason("Zorblat mystery", vocab), "")

    def test_one_mention_is_not_owner_vocabulary(self):
        # A single (possibly mis-heard) turn cannot vouch for its own words,
        # and an identical repeat (a hallucination loop) counts once.
        vocab = th.owner_vocab_from_texts(
            ["the zorblat mystery", "the zorblat mystery"])
        self.assertNotIn("zorblat", vocab)

    def test_inflections_affixes_and_compounds_are_words(self):
        for w in ("rebuilding", "supplementation", "carefully", "workbench",
                  "cleanup", "unplugged", "deployment", "dramatic"):
            with self.subTest(word=w):
                self.assertTrue(th.is_known_word(w))
        self.assertFalse(th.is_known_word("zorblat"))


class TranscriptQualityTests(unittest.TestCase):
    """Rule 2: a shaky transcript never teaches a topic."""

    def test_clear_and_typed_turns_pass(self):
        t = "I'm building a garden shed this weekend"
        self.assertEqual(th.transcript_quality_reason(t, _CLEAR), "")
        self.assertEqual(th.transcript_quality_reason(t, _TYPED), "")
        self.assertEqual(th.transcript_quality_reason(t, None), "")

    def test_low_confidence_rejected_even_when_answerable(self):
        # -1.2 passes the RESPONSE gate (-1.5) but not the learn gate (-1.0).
        why = th.transcript_quality_reason(
            "the mystery continues tonight",
            {"no_speech_prob": 0.1, "avg_logprob": -1.2})
        self.assertIn("avg_logprob", why)

    def test_high_no_speech_rejected(self):
        why = th.transcript_quality_reason(
            "the mystery continues tonight",
            {"no_speech_prob": 0.7, "avg_logprob": -0.2})
        self.assertIn("no_speech_prob", why)

    def test_repetitive_transcript_rejected(self):
        loop = "thank you so much " * 8
        self.assertIn("compression_ratio",
                      th.transcript_quality_reason(loop, _CLEAR))
        self.assertIn("compression_ratio", th.transcript_quality_reason(
            "fine", {"compression_ratio": 3.1}))

    def test_garbled_transcript_rejected(self):
        self.assertIn("garbled", th.transcript_quality_reason(
            "zorblat flemwick grundlefax", _CLEAR))

    def test_malformed_conf_values_are_ignored_not_fatal(self):
        self.assertEqual(th.transcript_quality_reason(
            "garden shed", {"no_speech_prob": "n/a", "avg_logprob": True}), "")

    def test_ambient_speech_is_never_a_topic_source(self):
        self.assertIn("not owner-directed", th.screen_reason(
            owner_directed=False, turn_text="garden shed", conf=_CLEAR))
        self.assertEqual(th.screen_reason(
            owner_directed=True, turn_text="garden shed", conf=_CLEAR), "")


class SightingTests(unittest.TestCase):
    """Rule 3: seen in >= 2 SEPARATE owner turns before it is surfaced."""

    def setUp(self):
        self.mem = {"topics": [], "projects": []}

    def _see(self, label, text, kind="topic", vocab=frozenset(), now=1000.0):
        return th.observe(self.mem, kind=kind, label=label,
                          turn=th.turn_id(text), vocab=vocab, now=now)

    def test_first_sighting_is_held(self):
        ok, why = self._see("garden shed", "I'm building a garden shed")
        self.assertFalse(ok)
        self.assertIn("1 of 2", why)
        self.assertEqual(len(self.mem[th.CANDIDATES_KEY]), 1)

    def test_second_separate_turn_surfaces_even_when_paraphrased(self):
        self._see("weekend plans", "any plans this weekend, I might garden")
        ok, why = self._see("Plans for the weekend",
                            "what should I do this weekend", now=2000.0)
        self.assertTrue(ok, why)
        self.assertEqual(len(self.mem[th.CANDIDATES_KEY]), 1)

    def test_the_same_utterance_twice_is_one_turn(self):
        self._see("garden shed", "tell me about the garden shed")
        ok, _ = self._see("garden shed", "Tell me about the garden shed!",
                          now=2000.0)
        self.assertFalse(ok)

    def test_topics_and_projects_are_counted_separately(self):
        self._see("garden shed", "turn one about the shed", kind="project")
        ok, _ = self._see("garden shed", "turn two about the shed",
                          kind="topic", now=2000.0)
        self.assertFalse(ok)

    def test_unrelated_labels_do_not_pool_sightings(self):
        self._see("Zorblat mystery", "turn one")
        ok, _ = self._see("mystery novel", "turn two", now=2000.0)
        self.assertFalse(ok)

    def test_non_word_label_stays_held_after_two_turns(self):
        self._see("Zorblat mystery", "something about zorblat")
        ok, why = self._see("Zorblat mystery", "zorblat again", now=2000.0)
        self.assertFalse(ok)
        self.assertIn("unrecognised", why)

    def test_owner_vocabulary_releases_a_real_proper_noun(self):
        vocab = frozenset({"zorblat"})
        self._see("Zorblat build", "how is the zorblat build", vocab=vocab)
        ok, _ = self._see("Zorblat build", "order zorblat parts",
                          vocab=vocab, now=2000.0)
        self.assertTrue(ok)

    def test_candidate_store_is_bounded(self):
        def word(i):   # a distinct letters-only token per i
            return "".join(chr(97 + int(d)) for d in str(i)) + "qz"
        for i in range(th.MAX_CANDIDATES + 15):
            th.observe(self.mem, kind="topic", label=word(i),
                       turn=f"t{i}", now=float(i))
        cands = self.mem[th.CANDIDATES_KEY]
        self.assertEqual(len(cands), th.MAX_CANDIDATES)
        # The OLDEST are evicted, the newest kept.
        self.assertIn(word(th.MAX_CANDIDATES + 14), [c["label"] for c in cands])
        self.assertNotIn(word(0), [c["label"] for c in cands])

    def test_matches_any_catches_a_paraphrased_project(self):
        self.assertTrue(th.matches_any("Garden shed build",
                                       ["Building a garden shed"]))
        self.assertFalse(th.matches_any("Bird feeder",
                                        ["Building a garden shed"]))

    def test_forget_sightings_since_drops_the_window_only(self):
        self._see("garden shed", "turn one", now=100.0)
        self._see("garden shed", "turn two", now=5000.0)
        self._see("bird feeder", "turn three", now=6000.0)
        removed = th.forget_sightings_since(self.mem, 4000.0)
        self.assertEqual(removed, 2)
        cands = self.mem[th.CANDIDATES_KEY]
        self.assertEqual(len(cands), 1)
        self.assertEqual([t["ts"] for t in cands[0]["turns"]], [100.0])


class OwnerLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="topic_hygiene_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.log = os.path.join(self.tmp, "voice_commands.jsonl")

    def _write(self, texts):
        with open(self.log, "w", encoding="utf-8") as fh:
            for i, t in enumerate(texts):
                fh.write(json.dumps({"ts": 1000.0 + i, "text": t}) + "\n")
            fh.write("not json\n\n")

    def test_missing_log_is_none_not_empty(self):
        self.assertIsNone(th.read_owner_texts(os.path.join(self.tmp, "nope")))
        self.assertEqual(th.load_owner_vocab(os.path.join(self.tmp, "nope")),
                         frozenset())

    def test_reads_texts_and_builds_vocab(self):
        self._write(["open the zorblat panel", "zorblat status please"])
        self.assertEqual(th.read_owner_texts(self.log),
                         ["open the zorblat panel", "zorblat status please"])
        self.assertIn("zorblat", th.load_owner_vocab(self.log))

    def test_vocab_cache_refreshes_when_the_log_changes(self):
        self._write(["alpha one"])
        self.assertNotIn("flemwick", th.load_owner_vocab(self.log))
        self._write(["flemwick one", "flemwick two", "and a longer third line"])
        self.assertIn("flemwick", th.load_owner_vocab(self.log))


class FindSuspectsTests(unittest.TestCase):
    """The audit's view of what is ALREADY stored — same rules as the gate."""

    def _mem(self):
        return {
            "topics": [
                {"date": "2026-09-01", "ts": 1.0, "topic": "garden shed roof"},
                {"date": "2026-09-02", "ts": 2.0, "topic": "Flemwick mystery"},
                {"date": "2026-09-03", "ts": 3.0, "topic": "garden shed build"},
                {"date": "2026-09-04", "ts": 4.0, "topic": "weather check"},
                {"date": "2026-09-05", "ts": 5.0, "topic": "restaurant venture"},
            ],
            "projects": ["Building a garden shed", "Zorblat supplement research",
                         "Bird feeder"],
        }

    _LOG = ["what's the weather like", "check the weather for tomorrow",
            "how is the bird feeder print", "remind me about the bird feeder"]

    def test_flags_non_words_and_uncorroborated_items(self):
        got = {(s["kind"], s["text"]) for s in
               th.find_suspects(self._mem(), self._LOG)}
        self.assertEqual(got, {("topic", "Flemwick mystery"),
                               ("topic", "restaurant venture"),
                               ("project", "Zorblat supplement research")})

    def test_every_suspect_carries_index_and_reason(self):
        for s in th.find_suspects(self._mem(), self._LOG):
            self.assertTrue(s["reason"])
            self.assertIsInstance(s["index"], int)

    def test_without_the_owner_log_only_the_non_word_rule_runs(self):
        got = {(s["kind"], s["text"]) for s in th.find_suspects(self._mem(), None)}
        self.assertEqual(got, {("topic", "Flemwick mystery")})

    def test_tolerates_a_malformed_store(self):
        self.assertEqual(th.find_suspects({"topics": "x", "projects": None}), [])
        self.assertEqual(th.find_suspects(
            {"topics": [None, {"topic": 3}], "projects": [5, ""]}, []), [])


class ForgetLastHourReachesSightingsTests(unittest.TestCase):
    """forget_last_hour promises "that hour never happened": the new hidden
    sighting store is a conversation trace too, so it must be pruned, or a
    forgotten hour could still promote a topic on its next mention."""

    def test_recent_sightings_are_forgotten(self):
        from core import actions as A
        now = 1_000_000.0
        mem = {"topics": [], "sessions": [], th.CANDIDATES_KEY: [
            {"kind": "topic", "label": "garden shed", "key": ["garden", "shed"],
             "turns": [{"id": "a", "ts": now - 600}]},
            {"kind": "topic", "label": "bird feeder", "key": ["bird", "feeder"],
             "turns": [{"id": "b", "ts": now - 86400}]},
        ]}
        bc = mock.MagicMock()
        bc.load_memory.return_value = mem
        bc.pattern_memory.forget_voice_commands_since.return_value = 0
        with mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch.object(A.time, "time", return_value=now), \
                mock.patch("core.long_term_memory.forget_since",
                           return_value={"episodes": 0, "facts": 0}):
            out = A._act_forget_last_hour()
        self.assertEqual(out, "forgot 1 item(s) from the last hour")
        saved = bc.save_memory.call_args[0][0]
        self.assertEqual([c["label"] for c in saved[th.CANDIDATES_KEY]],
                         ["bird feeder"])


class LexiconTests(unittest.TestCase):
    """2026-09-29 follow-up: the first ~3.6k-word list flagged ordinary words
    and real work in the live store (the dry run's false positives were
    everyday vocabulary, a place name, a tech acronym and a client name). The
    lexicon is now a ~23k-word bundled list plus place names and acronyms,
    loaded lazily; the owner's stored facts count as known words."""

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_bundled_lexicon_is_large_permissive_and_clean(self):
        from core import common_words as cw
        self.assertGreaterEqual(len(cw.lexicon()), 20000)
        path = os.path.join(self._ROOT, "core", "english_words.txt")
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        header = "\n".join(ln for ln in lines if ln.startswith("#"))
        self.assertIn("MIT", header)
        self.assertIn("Copyright (c) 2022 OpenAI", header)
        words = [ln for ln in lines if ln and not ln.startswith("#")]
        self.assertGreaterEqual(len(words), 20000)
        self.assertEqual(words, sorted(set(words)))
        self.assertTrue(all(re.fullmatch(r"[a-z]{3,}", w) for w in words))

    def test_lexicon_loads_lazily(self):
        code = ("import core.topic_hygiene, core.common_words as cw\n"
                "print(cw._LEXICON is None)\n"
                "core.topic_hygiene.is_known_word('garden')\n"
                "print(cw._LEXICON is None)\n")
        out = subprocess.run([sys.executable, "-c", code], cwd=self._ROOT,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.split(), ["True", "False"], out.stderr)

    def test_everyday_words_places_and_acronyms_are_known(self):
        for w in ("capabilities", "conversion", "explanation", "gratitude",
                  "mulching", "australia", "tokyo", "pid", "vpn", "hdmi"):
            with self.subTest(word=w):
                self.assertTrue(th.is_known_word(w))

    def test_ordinary_labels_are_not_flagged(self):
        for label in ("overview of capabilities", "unit conversion",
                      "the capital of Australia", "PID loop explanation",
                      "time in Tokyo", "a note of gratitude",
                      "mulching flower beds", "VPN setup for the office"):
            with self.subTest(label=label):
                self.assertEqual(th.garbled_reason(label), "")

    def test_non_words_are_still_unknown(self):
        for w in ("zorblat", "flemwick", "grundlefax", "bofferton"):
            with self.subTest(word=w):
                self.assertFalse(th.is_known_word(w))
        self.assertTrue(th.garbled_reason("Zorblat mystery"))


class NonceShapeTests(unittest.TestCase):
    """A label is flagged only when at least half its content words are
    unknown AND one of those is nonce-shaped (4+ letters, not an acronym)."""

    def test_acronyms_written_in_capitals_are_known(self):
        self.assertEqual(th.acronyms("QZX work on the PCBs"),
                         frozenset({"qzx", "pcb"}))
        self.assertEqual(th.garbled_reason("QZXV rollout"), "")
        self.assertTrue(th.garbled_reason("qzxv rollout"))

    def test_short_unknown_tokens_never_flag_on_their_own(self):
        # A lower-cased client initialism is an abbreviation, not a mis-hearing.
        self.assertEqual(th.garbled_reason("qzx work"), "")
        self.assertEqual(th.garbled_reason("qzx"), "")

    def test_stored_facts_make_a_client_name_known(self):
        mem = {"facts": ["User does contract work for Flemwick Ltd",
                         "Flemwick is the user's biggest client"],
               "projects": ["Flemwick rollout"], "topics": []}
        self.assertEqual(th.find_suspects(mem, None), [])
        mem["facts"] = []
        self.assertEqual([s["text"] for s in th.find_suspects(mem, None)],
                         ["Flemwick rollout"])
        self.assertEqual(th.facts_vocab({"facts": "not a list"}), frozenset())

    def test_one_fact_cannot_vouch_for_a_non_word(self):
        # Live 2026-09-29: the only fact holding the mis-heard name was learned
        # from the same overheard line as the topic. One fact is not enough...
        junk = {"facts": ["The user is working on a Flemwick case"]}
        self.assertNotIn("flemwick", th.facts_vocab(junk))
        # ...unless the fact writes it as an acronym in capitals.
        self.assertIn("qzx", th.facts_vocab({"facts": ["Works with the QZX team"]}))

    def test_reason_says_whether_the_owner_ever_said_it(self):
        mem = {"topics": [{"topic": "restaurant venture", "ts": 1.0},
                          {"topic": "bird feeder", "ts": 2.0}],
               "projects": []}
        got = {s["text"]: s["reason"] for s in th.find_suspects(
            mem, ["how is the bird feeder print going", "open the browser"])}
        self.assertIn("no logged owner turn uses its words",
                      got["restaurant venture"])
        self.assertIn("only 1 logged owner turn", got["bird feeder"])

    def test_fact_mentions_are_reported_but_never_corroborate(self):
        mem = {"topics": [{"topic": "restaurant venture", "ts": 1.0}],
               "projects": [],
               "facts": ["The user mentioned a restaurant venture"]}
        got = th.find_suspects(mem, ["open the browser"])
        self.assertEqual(len(got), 1)
        self.assertIn("appear in 1 stored fact(s)", got[0]["reason"])


class GroundedPromptTests(unittest.TestCase):
    """Read-time grounding in the tracked prompt: "what am I working on" has a
    real action, it survives the local router's slimming, and the persona no
    longer tells the model to volunteer learned projects."""

    @classmethod
    def setUpClass(cls):
        from core import prompt_router, prompts
        cls.router = prompt_router
        cls.prompts = prompts

    def test_working_on_questions_reach_project_status_on_the_local_route(self):
        for q in ("JARVIS, what am I working on?",
                  "what am I working on lately",
                  "what have I been doing lately",
                  "what are my projects"):
            with self.subTest(q=q):
                slim = self.router.slim_pc_control(q, self.prompts.PC_CONTROL_PROMPT)
                self.assertIn("[ACTION: project_status]", slim)

    def test_project_status_is_its_own_routed_section(self):
        names = [h for h, _ in self.router.split_pc_control(
            self.prompts.PC_CONTROL_PROMPT)[1]]
        self.assertIn("PROJECT STATUS", names)
        self.assertTrue(self.router._keywords_for("PROJECT STATUS"))

    def test_persona_does_not_volunteer_learned_projects(self):
        base = self.prompts.BASE_SYSTEM_PROMPT
        self.assertNotIn("Reference past projects and conversations", base)
        self.assertIn("never bring up an auto-learned topic or project", base)
        self.assertIn("never from the auto-learned topics", base)


if __name__ == "__main__":
    unittest.main()
