"""Fact provenance in core/long_term_memory (2026-10-02).

Every NEW fact written through add_fact(..., provenance=...) remembers who
said it (the voice-ID verdict: "owner" / "unknown"), how it reached JARVIS
("voice" / "typed" / "web" / "ambient"), the sentence it came from (clipped to
about 200 chars) and when that was heard. Facts stored before this have no
provenance and must still load and be retrieved exactly as before.

Privacy: the sentence lives in facts.json only -- never in Chroma metadata,
never in a log line, never in the spoken description.

Also here: guest mode's half of the store (core/guest_mode.py) -- while it is
on, add_fact and record_turn write nothing.

Every fact and sentence below is made up. Nothing outside a per-test temp dir
is written (tests.test_long_term_memory._LtmBase repoints every path).

    python tools/run_tests.py fact_provenance
"""
from __future__ import annotations

import ast
import json
import os
import time
import unittest
from unittest import mock

import core.long_term_memory as ltm
from core import guest_mode
from tests.test_long_term_memory import _LtmBase

_SENTENCE = "my cousin Quillon keeps three ferrets named after planets"


def _prov(**kw):
    base = {"speaker": "owner", "source": "voice", "utterance": _SENTENCE,
            "ts": 1_790_000_000.0}
    base.update(kw)
    return base


def _printed(p) -> str:
    return " ".join(" ".join(str(a) for a in c.args) for c in p.call_args_list)


class ProvenanceRecordTests(unittest.TestCase):
    def test_a_record_keeps_speaker_source_sentence_and_time(self):
        rec = ltm.make_provenance(speaker="owner", source="typed",
                                  utterance="  I  switched\nto oat milk ",
                                  ts=1_790_000_000.0)
        self.assertEqual(rec, {"speaker": "owner", "source": "typed",
                               "utterance": "I switched to oat milk",
                               "ts": 1_790_000_000.0})

    def test_every_documented_source_is_kept(self):
        for src in ("voice", "typed", "web", "ambient"):
            with self.subTest(src=src):
                self.assertEqual(ltm.make_provenance(
                    speaker="unknown", source=src)["source"], src)

    def test_anything_else_is_unknown(self):
        rec = ltm.make_provenance(speaker="Quillon", source="telepathy")
        self.assertEqual((rec["speaker"], rec["source"]),
                         ("unknown", "unknown"))
        rec = ltm.make_provenance(speaker=None, source=None, utterance=42)
        self.assertEqual((rec["speaker"], rec["source"], rec["utterance"]),
                         ("unknown", "unknown", ""))

    def test_the_sentence_is_cut_to_about_200_chars(self):
        long = "word " * 120
        rec = ltm.make_provenance(speaker="owner", source="voice",
                                  utterance=long)
        self.assertLessEqual(len(rec["utterance"]), ltm.UTTERANCE_MAX_CHARS)
        self.assertGreater(len(rec["utterance"]), 180)
        self.assertTrue(rec["utterance"].endswith("…"))
        short = ltm.make_provenance(speaker="owner", source="voice",
                                    utterance="short one")
        self.assertEqual(short["utterance"], "short one")

    def test_a_missing_or_bad_time_is_now(self):
        for bad in (None, "noon", -5, True, float("nan"), float("inf"),
                    1_790_000_000_000):
            with self.subTest(bad=bad):
                before = time.time()
                rec = ltm.make_provenance(speaker="owner", source="voice",
                                          ts=bad)
                self.assertGreaterEqual(rec["ts"], before)
                self.assertLessEqual(rec["ts"], time.time())

    def test_reading_back_a_legacy_or_malformed_entry_is_none(self):
        for entry in ({"text": "a fact"}, {"text": "x", "provenance": None},
                      {"text": "x", "provenance": "owner said so"},
                      {"text": "x", "provenance": ["owner"]}, None, "fact"):
            with self.subTest(entry=entry):
                self.assertIsNone(ltm.fact_provenance(entry))

    def test_reading_back_normalises_what_is_on_disk(self):
        got = ltm.fact_provenance({"provenance": {
            "speaker": "admin", "source": 7, "utterance": None, "ts": "x"}})
        self.assertEqual(got, {"speaker": "unknown", "source": "unknown",
                               "utterance": "", "ts": None})


class ProvenanceStoreTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()

    def test_a_new_fact_stores_its_provenance(self):
        fid = ltm.add_fact("User's cousin keeps three ferrets",
                           source="merge_memory", provenance=_prov())
        entry = ltm._facts[fid]
        self.assertEqual(entry["provenance"], {
            "speaker": "owner", "source": "voice", "utterance": _SENTENCE,
            "ts": 1_790_000_000.0})
        # The writer label the reflector's trust rule reads is untouched.
        self.assertEqual(entry["source"], "merge_memory")
        listed = ltm.list_facts()[0]
        self.assertEqual(ltm.fact_provenance(listed)["speaker"], "owner")

    def test_provenance_survives_a_restart(self):
        fid = ltm.add_fact("User's cousin keeps three ferrets",
                           provenance=_prov(source="web", speaker="unknown"))
        with open(ltm._FACTS_JSON, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk[0]["provenance"]["source"], "web")
        ltm._facts = {}
        ltm._loaded = False
        ltm.ensure_loaded()
        self.assertEqual(ltm.fact_provenance(ltm._facts[fid]),
                         {"speaker": "unknown", "source": "web",
                          "utterance": _SENTENCE, "ts": 1_790_000_000.0})

    def test_no_provenance_given_stores_none_as_before(self):
        fid = ltm.add_fact("User likes tea")
        self.assertNotIn("provenance", ltm._facts[fid])
        self.assertIsNone(ltm.fact_provenance(ltm._facts[fid]))

    def test_a_repeated_fact_keeps_the_first_provenance(self):
        a = ltm.add_fact("User likes tea", provenance=_prov(source="voice"))
        b = ltm.add_fact("User likes tea", provenance=_prov(source="typed"))
        self.assertEqual(a, b)
        self.assertEqual(ltm._facts[a]["provenance"]["source"], "voice")

    def test_the_sentence_is_never_printed(self):
        with mock.patch("builtins.print") as p:
            ltm.add_fact("User's cousin keeps three ferrets",
                         provenance=_prov())
            with mock.patch.object(guest_mode, "_on", [True]):
                ltm.add_fact("User's aunt sails", provenance=_prov())
        self.assertNotIn("Quillon", _printed(p))
        self.assertNotIn("ferrets", _printed(p))


class LegacyFactsTests(_LtmBase):
    """Old facts without provenance still load and are retrievable."""

    def test_old_facts_load_and_are_retrieved(self):
        self._install_fake_bm25()
        with mock.patch.object(ltm, "_try_import_chroma", lambda: None), \
             mock.patch.object(ltm, "_try_import_embedder", lambda: None):
            os.makedirs(ltm._DATA_DIR, exist_ok=True)
            legacy = [
                {"id": "fact_old1", "text": "User drinks green tea",
                 "source": "bobert_memory_migration", "tags": ["legacy"],
                 "created_at": 1_700_000_000.0,
                 "updated_at": 1_700_000_000.0},
                {"id": "fact_old2", "text": "User rides a blue bicycle",
                 "source": "merge_memory"},
            ]
            with open(ltm._FACTS_JSON, "w", encoding="utf-8") as f:
                json.dump(legacy, f)
            with open(ltm._MIGRATE_FLAG, "w", encoding="utf-8") as f:
                f.write("done\n")
            ltm.ensure_loaded()
            new = ltm.add_fact("User keeps bees", provenance=_prov())
            got = ltm.retrieve_facts("green tea", k=3)
        self.assertEqual(got[0]["id"], "fact_old1")
        self.assertIsNone(ltm.fact_provenance(got[0]))
        self.assertEqual(set(ltm._facts), {"fact_old1", "fact_old2", new})
        self.assertIsNone(ltm.fact_provenance(ltm._facts["fact_old2"]))


class ProvenanceChromaTests(_LtmBase):
    def test_chroma_metadata_never_carries_the_provenance(self):
        self._install_fake_embedder()
        coll = self._install_fake_chroma()
        ltm.ensure_loaded()
        fid = ltm.add_fact("User's cousin keeps three ferrets", tags=["a"],
                           provenance=_prov())
        # The dense write happened (a nested dict used to fail it) ...
        self.assertIn(fid, coll.store)
        meta = coll.store[fid]["metadata"]
        # ... and the sentence stayed in facts.json.
        self.assertNotIn("provenance", meta)
        self.assertNotIn("Quillon", json.dumps(meta))
        self.assertEqual(ltm._facts[fid]["provenance"]["utterance"], _SENTENCE)


class PickOriginTests(unittest.TestCase):
    ORIGINS = [
        {"utterance": "my sister Wenna moved to a lighthouse", "n": 1},
        {"utterance": "and the dog is called Biscuit", "n": 2},
        {"utterance": "what time is it", "n": 3},
    ]

    def test_the_turn_sharing_the_most_words_wins(self):
        self.assertEqual(ltm.pick_origin(
            "User's sister Wenna lives in a lighthouse", self.ORIGINS)["n"], 1)
        self.assertEqual(ltm.pick_origin(
            "User's dog is named Biscuit", self.ORIGINS)["n"], 2)

    def test_a_tie_goes_to_the_most_recent_turn(self):
        self.assertEqual(ltm.pick_origin("User likes jazz",
                                         self.ORIGINS)["n"], 3)

    def test_nothing_to_pick_from(self):
        self.assertIsNone(ltm.pick_origin("x", []))
        self.assertIsNone(ltm.pick_origin("x", None))
        self.assertIsNone(ltm.pick_origin("x", ["junk", 4]))

    def test_word_overlap_folds_case_plural_and_possessive(self):
        self.assertEqual(ltm.word_overlap("User's Ferrets", "my ferret"), 1)
        self.assertEqual(ltm.word_overlap(None, 5), 0)


class DescribeProvenanceTests(unittest.TestCase):
    NOW = time.mktime((2026, 10, 2, 15, 30, 0, 0, 0, -1))

    def _say(self, **kw):
        return ltm.describe_provenance(_prov(**kw), now=self.NOW)

    def test_owner_by_voice_today(self):
        said = self._say(ts=self.NOW - 600)
        self.assertTrue(said.startswith("you told me by voice, today at "),
                        said)
        self.assertIn("3:20 PM", said)

    def test_each_source_reads_differently(self):
        cases = {("voice", "unknown"): "couldn't confirm as yours",
                 ("typed", "owner"): "typed to me",
                 ("web", "unknown"): "my web page",
                 ("ambient", "owner"): "overheard you",
                 ("ambient", "unknown"): "overheard it in the room"}
        for (src, spk), needle in cases.items():
            with self.subTest(src=src, spk=spk):
                self.assertIn(needle, self._say(source=src, speaker=spk,
                                                ts=self.NOW - 60))

    def test_older_dates(self):
        self.assertIn("yesterday at", self._say(ts=self.NOW - 86400))
        older = self._say(ts=time.mktime((2026, 9, 24, 9, 5, 0, 0, 0, -1)))
        self.assertIn("on Thursday 24 September at 9:05 AM", older)
        last_year = self._say(ts=time.mktime((2025, 3, 1, 9, 0, 0, 0, 0, -1)))
        self.assertIn("2025", last_year)

    def test_the_sentence_is_never_spoken(self):
        for src in ("voice", "typed", "web", "ambient", "bogus"):
            with self.subTest(src=src):
                self.assertNotIn("Quillon", self._say(source=src))

    def test_no_record_is_empty(self):
        self.assertEqual(ltm.describe_provenance(None), "")
        self.assertEqual(ltm.describe_provenance("owner"), "")


class GuestModeFlagTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(guest_mode.set_on, False)

    def test_on_and_off(self):
        guest_mode.set_on(True)
        self.assertTrue(guest_mode.is_on())
        guest_mode.set_on(0)
        self.assertFalse(guest_mode.is_on())

    def test_visitors_voices_are_a_live_session_switch(self):
        """The voice-ID bypass (visitors' voices pass the gates) needs a live
        "guest mode on" this run, never just the saved flag, and closes with
        guest mode (review 2026-10-02: it used to reset on every restart)."""
        self.assertFalse(guest_mode.voices_open())
        guest_mode.set_on(True)                # the boot seed: memory only
        self.assertFalse(guest_mode.voices_open())
        guest_mode.set_voices_open(True)       # the live voice / web flip
        self.assertTrue(guest_mode.voices_open())
        guest_mode.set_on(False)               # off closes the gates too
        self.assertFalse(guest_mode.voices_open())
        guest_mode.set_on(True)
        self.assertFalse(guest_mode.voices_open())
        # Open without guest mode on is not open.
        guest_mode.set_on(False)
        guest_mode.set_voices_open(True)
        self.assertFalse(guest_mode.voices_open())

    def test_never_raises_on_a_broken_slot(self):
        with mock.patch.object(guest_mode, "_on", []):
            self.assertFalse(guest_mode.is_on())
            guest_mode.set_on(True)            # no raise
            self.assertFalse(guest_mode.voices_open())
        with mock.patch.object(guest_mode, "_voices", []):
            self.assertFalse(guest_mode.voices_open())
            guest_mode.set_voices_open(True)   # no raise

    def test_the_module_never_reads_the_owners_settings(self):
        """Importing it must not seed from core.config: a test or tool that
        only imports a module always starts with guest mode OFF (the boot
        seed is bobert_companion._guest_mode_boot)."""
        with open(guest_mode.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported = [n for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual([getattr(n, "module", None) for n in imported],
                         ["__future__"])


class GuestModeConfigTests(unittest.TestCase):
    """The saved flip comes back at boot: GUEST_MODE is a core.config
    constant (default off) that _apply_user_settings reads from
    user_settings.json, like REQUIRE_WAKE_MODE."""

    def test_the_default_is_off(self):
        import core.config as cfg
        with open(cfg.__file__, encoding="utf-8") as f:
            self.assertIn("\nGUEST_MODE = False\n", f.read())

    def test_a_saved_true_is_applied(self):
        import core.config as cfg
        saved = (cfg.GUEST_MODE, list(cfg._SAFETY_SETTINGS_WARNINGS),
                 cfg._USER_SETTINGS_ERROR)

        def restore():
            cfg.GUEST_MODE = saved[0]
            cfg._SAFETY_SETTINGS_WARNINGS[:] = saved[1]
            cfg._USER_SETTINGS_ERROR = saved[2]
        self.addCleanup(restore)
        cfg.GUEST_MODE = False
        with mock.patch.object(cfg.os.path, "exists", return_value=True), \
             mock.patch("core.config.open", mock.mock_open(
                 read_data='{"GUEST_MODE": true}'), create=True):
            cfg._apply_user_settings()
        self.assertIs(cfg.GUEST_MODE, True)


class GuestModeStoreTests(_LtmBase):
    def setUp(self):
        super().setUp()
        self._force_no_deps()
        ltm.ensure_loaded()
        self.addCleanup(guest_mode.set_on, False)

    def test_no_fact_is_stored_while_guests_are_here(self):
        guest_mode.set_on(True)
        self.assertEqual(ltm.add_fact("User's guest likes rum",
                                      provenance=_prov()), "")
        self.assertEqual(ltm._facts, {})
        self.assertFalse(os.path.exists(ltm._FACTS_JSON))

    def test_no_turn_is_recorded_while_guests_are_here(self):
        guest_mode.set_on(True)
        ltm.record_turn("user", "my name is Orlo and I am visiting")
        self.assertEqual(ltm._working, [])
        self.assertFalse(os.path.exists(ltm._EPISODE_LOG))

    def test_deleting_still_works_in_guest_mode(self):
        fid = ltm.add_fact("User likes tea")
        guest_mode.set_on(True)
        self.assertTrue(ltm.delete_fact(fid))

    def test_writes_resume_when_it_is_turned_off(self):
        guest_mode.set_on(True)
        ltm.add_fact("dropped while on")
        guest_mode.set_on(False)
        fid = ltm.add_fact("kept after", provenance=_prov())
        ltm.record_turn("user", "back to normal")
        self.assertEqual([e["text"] for e in ltm._facts.values()],
                         ["kept after"])
        self.assertTrue(fid)
        self.assertEqual(len(ltm._working), 1)


class GuestModeVoiceCommandLogTests(unittest.TestCase):
    """memory.record_voice_command: a visitor's words must not become the
    owner's habits or vocabulary."""

    def test_nothing_is_logged_in_guest_mode(self):
        import memory as pattern_memory
        self.addCleanup(guest_mode.set_on, False)
        with mock.patch.object(pattern_memory, "_ensure_dir") as ensure, \
             mock.patch("builtins.open", side_effect=AssertionError(
                 "opened a file")):
            guest_mode.set_on(True)
            pattern_memory.record_voice_command("play the lighthouse song",
                                                active_app="")
        ensure.assert_not_called()


class ProvenanceRoutingTests(unittest.TestCase):
    """The local route ships only the prompt sections a turn implicates
    (core/prompt_router.slim_pc_control): every common way of asking where a
    fact came from must carry where_learned to the model."""

    def test_every_way_of_asking_reaches_the_action(self):
        from core.prompt_router import slim_pc_control
        from core.prompts import PC_CONTROL_PROMPT
        for said in ("where did you learn that", "who told you that",
                     "how do you know that", "how did you know that",
                     "Jarvis, how did you know that?",
                     "where did you get that from"):
            self.assertIn("where_learned",
                          slim_pc_control(said, PC_CONTROL_PROMPT), said)


if __name__ == "__main__":
    unittest.main()
