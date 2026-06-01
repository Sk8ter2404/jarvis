"""Logic tests for skills/chappie_consciousness.py.

Chappie is JARVIS's silent continuous self-learner. The tests cover the
deterministic core:
  • the Layer-2 transcript quality filter (length, noise-prob, logprob, RMS,
    blocklist phrases),
  • episode grouping by time-gap + same-window (Layer 3),
  • bucket→episode assembly incl. in-call detection,
  • JSON extraction from a noisy LLM reply,
  • the UTC daily budget gate + reset,
  • fact accumulation from episodes with the LLM mocked (dedupe, open-question
    capture, observation counting),
  • the silent recall actions reading from a redirected episodes/facts file.

Episode/fact/cursor files are redirected to a temp dir so nothing in data/ is
touched. _llm is patched so no Ollama/Claude call happens. The background
consciousness thread is neutered by the harness (Thread.start no-op).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class ChappieFilterTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")

    def _good(self, **over):
        e = {"text": "we shipped the new release today",
             "no_speech_prob": 0.1, "avg_logprob": -0.3, "rms": 0.02}
        e.update(over)
        return e

    def test_filter_accepts_clean_entry(self):
        self.assertTrue(self.mod._passes_filter(self._good()))

    def test_filter_rejects_short_text(self):
        self.assertFalse(self.mod._passes_filter(self._good(text="hi")))

    def test_filter_rejects_high_no_speech_prob(self):
        self.assertFalse(self.mod._passes_filter(self._good(no_speech_prob=0.9)))

    def test_filter_rejects_low_logprob(self):
        self.assertFalse(self.mod._passes_filter(self._good(avg_logprob=-2.0)))

    def test_filter_rejects_low_rms(self):
        self.assertFalse(self.mod._passes_filter(self._good(rms=0.001)))

    def test_filter_rejects_noise_phrase_with_trailing_punct(self):
        # "Thank you." normalises to "thank you" which is on the blocklist.
        self.assertFalse(self.mod._passes_filter(self._good(text="Thank you.")))

    def test_filter_rejects_non_numeric_fields(self):
        self.assertFalse(self.mod._passes_filter(self._good(rms="loud")))

    # ── _extract_json ────────────────────────────────────────────────────
    def test_extract_json_from_noisy_reply(self):
        raw = 'Sure! Here it is:\n{"summary": "ok", "topics": ["a"]} -- done'
        out = self.mod._extract_json(raw)
        self.assertEqual(out["summary"], "ok")
        self.assertEqual(out["topics"], ["a"])

    def test_extract_json_returns_none_on_no_object(self):
        self.assertIsNone(self.mod._extract_json("no braces here"))
        self.assertIsNone(self.mod._extract_json(""))


class ChappieEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")

    def test_group_episodes_splits_on_time_gap(self):
        base = 1000.0
        filtered = [
            {"ts": base,       "window": "Code", "text": "one"},
            {"ts": base + 10,  "window": "Code", "text": "two"},      # within 30s
            {"ts": base + 200, "window": "Code", "text": "three"},    # >30s → new ep
        ]
        eps = self.mod._group_episodes(filtered)
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[0]["utterance_count"], 2)
        self.assertEqual(eps[1]["utterance_count"], 1)

    def test_group_episodes_splits_on_window_change(self):
        base = 2000.0
        filtered = [
            {"ts": base,      "window": "Teams", "text": "call talk"},
            {"ts": base + 5,  "window": "Chrome", "text": "browsing"},  # window changed
        ]
        eps = self.mod._group_episodes(filtered)
        self.assertEqual(len(eps), 2)

    def test_group_episodes_empty(self):
        self.assertEqual(self.mod._group_episodes([]), [])

    def test_bucket_to_episode_detects_in_call(self):
        ep = self.mod._bucket_to_episode([
            {"ts": 5.0, "window": "Zoom Meeting", "text": "hi team"},
            {"ts": 9.0, "window": "Zoom Meeting", "text": "agenda"},
        ])
        self.assertTrue(ep["in_call"])
        self.assertEqual(ep["duration_s"], 4.0)
        self.assertEqual(ep["utterances"], ["hi team", "agenda"])
        self.assertEqual(ep["summary"], "")  # not yet enriched

    def test_bucket_to_episode_not_in_call_for_plain_window(self):
        ep = self.mod._bucket_to_episode([{"ts": 1.0, "window": "Notepad", "text": "x y z"}])
        self.assertFalse(ep["in_call"])

    # ── budget gate ──────────────────────────────────────────────────────
    def test_budget_resets_on_date_roll(self):
        cur = {"budget_date": "1999-01-01", "budget_used_usd": 5.0}
        self.assertTrue(self.mod._check_budget_and_reset(cur))  # rolled → reset → has budget
        self.assertEqual(cur["budget_used_usd"], 0.0)

    def test_budget_blocks_when_spent(self):
        today = self.mod._today_utc()
        cur = {"budget_date": today, "budget_used_usd": self.mod.DAILY_BUDGET_USD + 1}
        self.assertFalse(self.mod._check_budget_and_reset(cur))

    def test_charge_budget_accumulates(self):
        cur = {"budget_used_usd": 0.0}
        self.mod._charge_budget(cur, 0.01)
        self.mod._charge_budget(cur, 0.02)
        self.assertAlmostEqual(cur["budget_used_usd"], 0.03)


class ChappieFactsTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")

    def test_update_facts_creates_record_and_dedupes(self):
        eps = [{
            "id": "ep_1", "start_ts": 100.0, "summary": "Sam joined the IT team",
            "topics": ["work"], "mood": "casual", "new_entities": ["Sam"],
        }]
        facts: dict = {}
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        llm_out = json.dumps({"updates": [
            {"entity": "Sam", "type": "person",
             "new_fact": "Sam works in IT sales",
             "open_question": "which company?"}]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            touched = self.mod._update_facts_from_episodes(eps, facts, cursors)
        self.assertEqual(touched, 1)
        self.assertIn("Sam", facts)
        self.assertEqual(facts["Sam"]["type"], "person")
        self.assertEqual(facts["Sam"]["observation_count"], 1)
        self.assertEqual(facts["Sam"]["facts"][0]["text"], "Sam works in IT sales")
        self.assertIn("which company?", facts["Sam"]["open_questions"])

    def test_update_facts_skips_episodes_without_entities(self):
        eps = [{"id": "ep_2", "start_ts": 1.0, "summary": "ambient noise",
                "topics": [], "mood": "", "new_entities": []}]
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        with mock.patch.object(self.mod, "_llm") as llm:
            touched = self.mod._update_facts_from_episodes(eps, {}, cursors)
        self.assertEqual(touched, 0)
        llm.assert_not_called()  # no entity-bearing episodes → no LLM spend

    def test_update_facts_no_double_count_same_fact(self):
        eps = [{"id": "e", "start_ts": 1.0, "summary": "s", "topics": [],
                "mood": "", "new_entities": ["Acme"]}]
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        existing = {"Acme": {"first_observed": 1.0, "type": "company",
                             "observation_count": 1,
                             "facts": [{"text": "Acme makes widgets", "ts": 1.0}],
                             "open_questions": []}}
        llm_out = json.dumps({"updates": [
            {"entity": "Acme", "type": "company",
             "new_fact": "acme MAKES widgets",   # same fact, different case
             "open_question": ""}]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            self.mod._update_facts_from_episodes(eps, existing, cursors)
        # Substring dedupe (case-insensitive) keeps a single fact.
        self.assertEqual(len(existing["Acme"]["facts"]), 1)

    def test_update_facts_respects_budget(self):
        eps = [{"id": "e", "start_ts": 1.0, "summary": "s", "topics": [],
                "mood": "", "new_entities": ["X"]}]
        cursors = {"budget_date": self.mod._today_utc(),
                   "budget_used_usd": self.mod.DAILY_BUDGET_USD + 1}
        with mock.patch.object(self.mod, "_llm") as llm:
            touched = self.mod._update_facts_from_episodes(eps, {}, cursors)
        self.assertEqual(touched, 0)
        llm.assert_not_called()


class ChappieRecallActionTests(unittest.TestCase):
    """Recall actions read the episodes / facts files. Redirect them to a
    temp dir so we control content and touch nothing in data/."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")
        self.tmp = tempfile.mkdtemp(prefix="chappie_test_")
        self.addCleanup(self._cleanup)
        self.episodes = os.path.join(self.tmp, "episodes.jsonl")
        self.facts = os.path.join(self.tmp, "facts.json")
        self.mod._EPISODES_FILE = self.episodes
        self.mod._FACTS_FILE = self.facts

    def _cleanup(self):
        for f in (self.episodes, self.facts):
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _write_facts(self, data):
        with open(self.facts, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _write_episodes(self, eps):
        with open(self.episodes, "w", encoding="utf-8") as f:
            for ep in eps:
                f.write(json.dumps(ep) + "\n")

    # ── chappie_recall_entity ────────────────────────────────────────────
    def test_recall_entity_needs_a_name(self):
        out = self.actions["chappie_recall_entity"]("")
        self.assertIn("need a name", out.lower())

    def test_recall_entity_unknown(self):
        self._write_facts({})
        out = self.actions["chappie_recall_entity"]("Bob")
        self.assertIn("Nothing on file for Bob", out)

    def test_recall_entity_returns_facts_and_open_questions(self):
        self._write_facts({"Sam": {
            "observation_count": 4,
            "facts": [{"text": "works in IT sales"}],
            "open_questions": ["which company"]}})
        out = self.actions["chappie_recall_entity"]("Sam")
        self.assertIn("works in IT sales", out)
        self.assertIn("4 observation", out)
        self.assertIn("which company", out)

    def test_recall_entity_case_insensitive_substring(self):
        self._write_facts({"Sam Smith": {
            "observation_count": 1, "facts": [{"text": "drives a Tesla"}],
            "open_questions": []}})
        out = self.actions["chappie_recall_entity"]("sam")
        self.assertIn("drives a Tesla", out)

    # ── chappie_recall_today ─────────────────────────────────────────────
    def test_recall_today_nothing_when_no_file(self):
        # File doesn't exist yet.
        out = self.actions["chappie_recall_today"]("")
        self.assertIn("Nothing on the record", out)

    def test_recall_today_lists_today_summaries(self):
        now = time.time()
        self._write_episodes([
            {"start_ts": now - 60, "summary": "discussed the print job",
             "topics": ["3d printing"], "new_entities": []},
        ])
        out = self.actions["chappie_recall_today"]("")
        self.assertIn("discussed the print job", out)

    def test_recall_today_keyword_filter_miss(self):
        now = time.time()
        self._write_episodes([
            {"start_ts": now - 60, "summary": "talked about lunch",
             "topics": [], "new_entities": []},
        ])
        out = self.actions["chappie_recall_today"]("kubernetes")
        self.assertIn("Nothing on 'kubernetes'", out)

    def test_recall_today_excludes_yesterday(self):
        yesterday = time.time() - 26 * 3600
        self._write_episodes([
            {"start_ts": yesterday, "summary": "old stuff",
             "topics": [], "new_entities": []},
        ])
        out = self.actions["chappie_recall_today"]("")
        self.assertIn("Nothing on the record", out)

    # ── chappie_status ───────────────────────────────────────────────────
    def test_status_counts_episodes_and_budget(self):
        self._write_episodes([{"start_ts": 1.0, "summary": "a"},
                               {"start_ts": 2.0, "summary": "b"}])
        self._write_facts({"X": {}, "Y": {}})
        with mock.patch.object(self.mod, "_load_cursors",
                               return_value={"budget_used_usd": 0.25}):
            out = self.actions["chappie_status"]("")
        self.assertIn("2 episode(s)", out)
        self.assertIn("2 entity record(s)", out)
        self.assertIn("$0.250", out)


if __name__ == "__main__":
    unittest.main()
