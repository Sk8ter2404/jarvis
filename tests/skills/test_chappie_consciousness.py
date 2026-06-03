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

import contextlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring the prior
    state (including absence) on exit. For dotted names the leaf is also set as
    an attribute on its already-imported parent package. Mirrors the isolation
    contract proven in tests/skills/test_self_diagnostic.py."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                with contextlib.suppress(AttributeError):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


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


class ChappieIOHelperTests(unittest.TestCase):
    """_atomic_write_json (real path + local fallback + error), _load_json,
    cursor load/save, _extract_json error branch."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")
        self.tmp = tempfile.mkdtemp(prefix="chappie_io_")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(self.tmp, fn))
        with contextlib.suppress(OSError):
            os.rmdir(self.tmp)

    # ── _atomic_write_json ────────────────────────────────────────────────
    def test_atomic_write_delegates_to_core_when_importable(self):
        # core.atomic_io._atomic_write_json is real — exercise the happy
        # delegation path (the function the module prefers).
        path = os.path.join(self.tmp, "c.json")
        self.mod._atomic_write_json(path, {"a": 1})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1})

    def test_atomic_write_local_fallback_when_core_unavailable(self):
        # Force the core import to fail so the tempfile+os.replace fallback runs.
        path = os.path.join(self.tmp, "fallback.json")
        real_import = self.mod.importlib.import_module

        def _imp(name, *a, **k):
            if name == "core.atomic_io":
                raise ImportError("no core")
            return real_import(name, *a, **k)
        with mock.patch.object(self.mod.importlib, "import_module", side_effect=_imp):
            self.mod._atomic_write_json(path, {"b": 2})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"b": 2})

    def test_atomic_write_fallback_cleans_tmp_on_error(self):
        # core import fails (→ fallback), then json.dump raises mid-write: the
        # tempfile must be removed and the error re-raised.
        path = os.path.join(self.tmp, "boom.json")
        real_import = self.mod.importlib.import_module

        def _imp(name, *a, **k):
            if name == "core.atomic_io":
                raise ImportError("no core")
            return real_import(name, *a, **k)
        with mock.patch.object(self.mod.importlib, "import_module", side_effect=_imp), \
             mock.patch.object(self.mod.json, "dump", side_effect=ValueError("bad")):
            with self.assertRaises(ValueError):
                self.mod._atomic_write_json(path, object())
        # No stray .chappie_*.tmp left behind.
        leftovers = [n for n in os.listdir(self.tmp) if n.startswith(".chappie_")]
        self.assertEqual(leftovers, [])

    def test_atomic_write_fallback_success_runs_os_replace(self):
        # Block the `from core.atomic_io import ...` at the __import__ level (a
        # mock on importlib.import_module does NOT intercept a `from` import) so
        # the local tempfile->os.replace fallback fully runs and writes the file.
        path = os.path.join(self.tmp, "viafallback.json")
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "core.atomic_io":
                raise ImportError("no core")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=_imp):
            self.mod._atomic_write_json(path, {"ok": True})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"ok": True})

    def test_atomic_write_fallback_replace_and_cleanup_both_fail(self):
        # Fallback runs (core blocked), the write succeeds but os.replace raises,
        # AND the temp-file cleanup os.remove ALSO raises -> the inner
        # `except Exception: pass` swallows the cleanup error and the original
        # os.replace error propagates.
        path = os.path.join(self.tmp, "doomed.json")
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "core.atomic_io":
                raise ImportError("no core")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=_imp), \
             mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("replace denied")), \
             mock.patch.object(self.mod.os, "remove",
                               side_effect=OSError("remove denied")):
            with self.assertRaises(OSError):
                self.mod._atomic_write_json(path, {"x": 1})

    # ── _load_json ────────────────────────────────────────────────────────
    def test_load_json_missing_returns_default(self):
        sentinel = {"default": True}
        self.assertIs(self.mod._load_json(os.path.join(self.tmp, "nope.json"),
                                          sentinel), sentinel)

    def test_load_json_reads_existing(self):
        path = os.path.join(self.tmp, "ok.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.mod._load_json(path, None), [1, 2, 3])

    def test_load_json_corrupt_returns_default(self):
        path = os.path.join(self.tmp, "corrupt.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.mod._load_json(path, "DEF"), "DEF")

    # ── cursors ───────────────────────────────────────────────────────────
    def test_load_cursors_defaults_when_absent(self):
        self.mod._CURSORS_FILE = os.path.join(self.tmp, "cursors.json")
        c = self.mod._load_cursors()
        self.assertEqual(c["transcripts_offset"], 0)
        self.assertEqual(c["budget_used_usd"], 0.0)
        self.assertEqual(c["version"], 1)

    def test_save_then_load_cursors_roundtrip(self):
        self.mod._CURSORS_FILE = os.path.join(self.tmp, "cursors.json")
        c = self.mod._default_cursors()
        c["transcripts_offset"] = 4096
        c["budget_used_usd"] = 0.12
        self.mod._save_cursors(c)
        again = self.mod._load_cursors()
        self.assertEqual(again["transcripts_offset"], 4096)
        self.assertAlmostEqual(again["budget_used_usd"], 0.12)

    def test_save_cursors_swallows_write_error(self):
        self.mod._CURSORS_FILE = os.path.join(self.tmp, "cursors.json")
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only")):
            # Must not raise — logged to console instead.
            self.mod._save_cursors(self.mod._default_cursors())

    # ── _extract_json error branch ────────────────────────────────────────
    def test_extract_json_malformed_object_returns_none(self):
        # A '{' is present but the body never closes → raw_decode raises → None.
        self.assertIsNone(self.mod._extract_json('prefix {"a": '))

    def test_extract_json_non_dict_top_level_returns_none(self):
        # First '{' belongs to nothing dict-like once decoded (a list here).
        self.assertIsNone(self.mod._extract_json('[1, 2, 3]'))


class ChappieLLMTests(unittest.TestCase):
    """_llm routes background passes to bobert_companion._call_local_llm only
    (never Claude). bobert_companion is faked so the monolith never boots."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")

    def test_llm_calls_local_model(self):
        bc = types.ModuleType("bobert_companion")
        seen = {}

        def _local(system, messages, max_tokens=600):
            seen["system"] = system
            seen["messages"] = messages
            seen["max_tokens"] = max_tokens
            return "  local answer  "
        bc._call_local_llm = _local
        with inject_modules(bobert_companion=bc):
            out = self.mod._llm("SYS", "USER", max_tokens=222)
        self.assertEqual(out, "local answer")   # stripped
        self.assertEqual(seen["system"], "SYS")
        self.assertEqual(seen["messages"], [{"role": "user", "content": "USER"}])
        self.assertEqual(seen["max_tokens"], 222)

    def test_llm_returns_empty_on_failure(self):
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(side_effect=RuntimeError("ollama down"))
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._llm("s", "u"), "")

    def test_llm_none_result_becomes_empty_string(self):
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(return_value=None)
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._llm("s", "u"), "")


class ChappieTranscriptIOTests(unittest.TestCase):
    """_read_new_transcripts / _append_episodes / _read_episodes_since /
    _load_facts / _save_facts against a redirected temp dir."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")
        self.tmp = tempfile.mkdtemp(prefix="chappie_tx_")
        self.addCleanup(self._cleanup)
        self.transcripts = os.path.join(self.tmp, "transcripts.jsonl")
        self.episodes = os.path.join(self.tmp, "episodes.jsonl")
        self.facts = os.path.join(self.tmp, "facts.json")
        self.mod._TRANSCRIPTS_FILE = self.transcripts
        self.mod._EPISODES_FILE = self.episodes
        self.mod._FACTS_FILE = self.facts

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(self.tmp, fn))
        with contextlib.suppress(OSError):
            os.rmdir(self.tmp)

    # ── _read_new_transcripts ─────────────────────────────────────────────
    def test_read_new_transcripts_missing_file(self):
        entries, off = self.mod._read_new_transcripts(0)
        self.assertEqual(entries, [])
        self.assertEqual(off, 0)

    def test_read_new_transcripts_from_offset(self):
        line1 = json.dumps({"text": "first", "ts": 1.0}) + "\n"
        line2 = json.dumps({"text": "second", "ts": 2.0}) + "\n"
        with open(self.transcripts, "w", encoding="utf-8") as f:
            f.write(line1)
        offset_after_first = len(line1.encode("utf-8"))
        with open(self.transcripts, "a", encoding="utf-8") as f:
            f.write(line2)
        entries, new_off = self.mod._read_new_transcripts(offset_after_first)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "second")
        self.assertGreater(new_off, offset_after_first)

    def test_read_new_transcripts_skips_blank_and_bad_lines(self):
        with open(self.transcripts, "w", encoding="utf-8") as f:
            f.write("\n")                                   # blank
            f.write("{not json}\n")                          # malformed
            f.write(json.dumps({"text": "good", "ts": 3.0}) + "\n")
        entries, _ = self.mod._read_new_transcripts(0)
        self.assertEqual([e["text"] for e in entries], ["good"])

    def test_read_new_transcripts_open_error_returns_offset(self):
        with open(self.transcripts, "w", encoding="utf-8") as f:
            f.write("{}\n")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            entries, off = self.mod._read_new_transcripts(7)
        self.assertEqual(entries, [])
        self.assertEqual(off, 7)

    # ── _append_episodes ──────────────────────────────────────────────────
    def test_append_episodes_noop_on_empty(self):
        self.mod._append_episodes([])
        self.assertFalse(os.path.exists(self.episodes))

    def test_append_episodes_writes_jsonl(self):
        self.mod._append_episodes([{"id": "a"}, {"id": "b"}])
        with open(self.episodes, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual([e["id"] for e in lines], ["a", "b"])

    def test_append_episodes_swallows_write_error(self):
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            self.mod._append_episodes([{"id": "x"}])   # must not raise

    # ── _read_episodes_since ──────────────────────────────────────────────
    def test_read_episodes_since_missing_file(self):
        eps, cur = self.mod._read_episodes_since(0)
        self.assertEqual(eps, [])
        self.assertEqual(cur, 0)

    def test_read_episodes_since_skips_consumed_and_bad(self):
        with open(self.episodes, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "old"}) + "\n")     # index 0 (consumed)
            f.write("garbage\n")                           # index 1 (bad → skip)
            f.write(json.dumps({"id": "new"}) + "\n")      # index 2 (kept)
        eps, cur = self.mod._read_episodes_since(1)
        self.assertEqual([e["id"] for e in eps], ["new"])
        self.assertEqual(cur, 3)

    def test_read_episodes_since_skips_blank_lines(self):
        with open(self.episodes, "w", encoding="utf-8") as f:
            f.write("\n")                                    # blank → skipped
            f.write(json.dumps({"id": "real"}) + "\n")
        eps, cur = self.mod._read_episodes_since(0)
        self.assertEqual([e["id"] for e in eps], ["real"])
        self.assertEqual(cur, 2)

    def test_read_episodes_since_open_error(self):
        with open(self.episodes, "w", encoding="utf-8") as f:
            f.write("{}\n")
        with mock.patch("builtins.open", side_effect=OSError("x")):
            eps, cur = self.mod._read_episodes_since(2)
        self.assertEqual(eps, [])
        self.assertEqual(cur, 2)

    # ── _load_facts / _save_facts ─────────────────────────────────────────
    def test_save_then_load_facts(self):
        self.mod._save_facts({"Acme": {"type": "company"}})
        self.assertEqual(self.mod._load_facts(), {"Acme": {"type": "company"}})

    def test_save_facts_swallows_error(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("nope")):
            self.mod._save_facts({"X": {}})   # must not raise


class ChappieEnrichTests(unittest.TestCase):
    """_enrich_episode + _process_episodes_once with _llm mocked."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")
        self.tmp = tempfile.mkdtemp(prefix="chappie_en_")
        self.addCleanup(self._cleanup)
        self.transcripts = os.path.join(self.tmp, "transcripts.jsonl")
        self.episodes = os.path.join(self.tmp, "episodes.jsonl")
        self.mod._TRANSCRIPTS_FILE = self.transcripts
        self.mod._EPISODES_FILE = self.episodes

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(self.tmp, fn))
        with contextlib.suppress(OSError):
            os.rmdir(self.tmp)

    def _raw_ep(self, **over):
        ep = {"id": "ep_1", "start_ts": 1.0, "end_ts": 2.0, "duration_s": 1.0,
              "window": "Code", "in_call": False, "utterance_count": 1,
              "utterances": ["we ship friday"], "summary": "", "topics": [],
              "mood": "", "new_entities": []}
        ep.update(over)
        return ep

    def test_enrich_fills_fields_from_llm(self):
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        llm_out = json.dumps({"summary": "User plans a Friday release",
                              "topics": ["release", "work", None],
                              "mood": "focused",
                              "new_entities": ["Friday", ""]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            ep = self.mod._enrich_episode(self._raw_ep(), cursors)
        self.assertEqual(ep["summary"], "User plans a Friday release")
        # Falsy items (None / "") are dropped by the `if t` guard; truthy
        # strings are kept after stripping.
        self.assertEqual(ep["topics"], ["release", "work"])
        self.assertEqual(ep["mood"], "focused")
        self.assertEqual(ep["new_entities"], ["Friday"])      # "" dropped
        # Budget was charged once.
        self.assertGreater(cursors["budget_used_usd"], 0.0)

    def test_enrich_skips_llm_when_out_of_budget(self):
        cursors = {"budget_date": self.mod._today_utc(),
                   "budget_used_usd": self.mod.DAILY_BUDGET_USD + 1}
        with mock.patch.object(self.mod, "_llm") as llm:
            ep = self.mod._enrich_episode(self._raw_ep(), cursors)
        llm.assert_not_called()
        self.assertEqual(ep["summary"], "")    # left empty, stored anyway

    def test_enrich_handles_unparseable_llm_reply(self):
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        with mock.patch.object(self.mod, "_llm", return_value="not json at all"):
            ep = self.mod._enrich_episode(self._raw_ep(), cursors)
        self.assertEqual(ep["summary"], "")
        self.assertEqual(ep["topics"], [])

    def test_enrich_uses_user_name_env(self):
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        captured = {}

        def _llm(system, user, max_tokens=350):
            captured["system"] = system
            return "{}"
        with mock.patch.dict(os.environ, {"JARVIS_USER_NAME": "Alice"}), \
             mock.patch.object(self.mod, "_llm", side_effect=_llm):
            self.mod._enrich_episode(self._raw_ep(), cursors)
        self.assertIn("Alice", captured["system"])

    # ── _process_episodes_once ────────────────────────────────────────────
    def test_process_episodes_once_no_transcripts(self):
        cursors = self.mod._default_cursors()
        produced = self.mod._process_episodes_once(cursors)
        self.assertEqual(produced, 0)

    def test_process_episodes_once_full_pipeline(self):
        # Two close utterances (one episode), enriched with a non-empty summary.
        rows = [
            {"text": "we are shipping the release", "ts": 100.0, "window": "Code",
             "no_speech_prob": 0.1, "avg_logprob": -0.2, "rms": 0.02},
            {"text": "the deploy looks clean now", "ts": 110.0, "window": "Code",
             "no_speech_prob": 0.1, "avg_logprob": -0.2, "rms": 0.02},
        ]
        with open(self.transcripts, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cursors = self.mod._default_cursors()
        with mock.patch.object(self.mod, "_llm",
                               return_value='{"summary": "User shipped a release"}'):
            produced = self.mod._process_episodes_once(cursors)
        self.assertEqual(produced, 1)
        self.assertGreater(cursors["transcripts_offset"], 0)
        with open(self.episodes, encoding="utf-8") as f:
            stored = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(stored[0]["summary"], "User shipped a release")

    def test_process_episodes_once_drops_empty_summaries(self):
        rows = [{"text": "mumble mumble background", "ts": 5.0, "window": "X",
                 "no_speech_prob": 0.1, "avg_logprob": -0.2, "rms": 0.02}]
        with open(self.transcripts, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cursors = self.mod._default_cursors()
        # LLM returns empty summary → episode is dropped (not appended).
        with mock.patch.object(self.mod, "_llm", return_value='{"summary": ""}'):
            produced = self.mod._process_episodes_once(cursors)
        self.assertEqual(produced, 0)
        self.assertFalse(os.path.exists(self.episodes))
        # Offset still advances so the noise isn't re-read.
        self.assertGreater(cursors["transcripts_offset"], 0)

    def test_process_episodes_once_all_filtered_out(self):
        # Every row fails the quality gate → no episodes, offset advances.
        rows = [{"text": "hi", "ts": 1.0, "window": "X"}]   # too short
        with open(self.transcripts, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cursors = self.mod._default_cursors()
        with mock.patch.object(self.mod, "_llm") as llm:
            produced = self.mod._process_episodes_once(cursors)
        self.assertEqual(produced, 0)
        llm.assert_not_called()


class ChappieFactsBranchTests(unittest.TestCase):
    """Edge branches of _update_facts_from_episodes + _process_facts_once."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")
        self.tmp = tempfile.mkdtemp(prefix="chappie_fb_")
        self.addCleanup(self._cleanup)
        self.episodes = os.path.join(self.tmp, "episodes.jsonl")
        self.facts = os.path.join(self.tmp, "facts.json")
        self.mod._EPISODES_FILE = self.episodes
        self.mod._FACTS_FILE = self.facts

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(self.tmp, fn))
        with contextlib.suppress(OSError):
            os.rmdir(self.tmp)

    def _ep(self, **over):
        ep = {"id": "e", "start_ts": 1.0, "summary": "s", "topics": [],
              "mood": "", "new_entities": ["Acme"]}
        ep.update(over)
        return ep

    def test_update_facts_skips_nameless_update(self):
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        llm_out = json.dumps({"updates": [{"entity": "  ", "new_fact": "ignored"}]})
        facts: dict = {}
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            touched = self.mod._update_facts_from_episodes([self._ep()], facts, cursors)
        self.assertEqual(touched, 0)
        self.assertEqual(facts, {})

    def test_update_facts_promotes_other_type(self):
        # Existing record typed "other"; a concrete type in the update upgrades it.
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        facts = {"Acme": {"first_observed": 1.0, "type": "other",
                          "observation_count": 1, "facts": [], "open_questions": []}}
        llm_out = json.dumps({"updates": [
            {"entity": "Acme", "type": "company", "new_fact": "", "open_question": ""}]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            self.mod._update_facts_from_episodes([self._ep()], facts, cursors)
        self.assertEqual(facts["Acme"]["type"], "company")

    def test_update_facts_caps_open_questions_at_five(self):
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        facts = {"Acme": {"first_observed": 1.0, "type": "company",
                          "observation_count": 1, "facts": [],
                          "open_questions": ["q1", "q2", "q3", "q4", "q5"]}}
        llm_out = json.dumps({"updates": [
            {"entity": "Acme", "type": "company", "new_fact": "",
             "open_question": "q6"}]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            self.mod._update_facts_from_episodes([self._ep()], facts, cursors)
        oq = facts["Acme"]["open_questions"]
        self.assertEqual(len(oq), 5)
        self.assertIn("q6", oq)
        self.assertNotIn("q1", oq)   # oldest dropped by [-5:]

    def test_update_facts_caps_facts_history_at_twenty(self):
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        existing_facts = [{"text": f"fact number {i}", "ts": 1.0} for i in range(20)]
        facts = {"Acme": {"first_observed": 1.0, "type": "company",
                          "observation_count": 1, "facts": existing_facts,
                          "open_questions": []}}
        llm_out = json.dumps({"updates": [
            {"entity": "Acme", "type": "company",
             "new_fact": "a brand new distinct fact", "open_question": ""}]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            self.mod._update_facts_from_episodes([self._ep()], facts, cursors)
        kept = facts["Acme"]["facts"]
        self.assertEqual(len(kept), 20)                      # capped
        self.assertEqual(kept[-1]["text"], "a brand new distinct fact")
        self.assertNotIn("fact number 0",
                         [f["text"] for f in kept])           # oldest dropped

    def test_update_facts_empty_llm_updates(self):
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        facts: dict = {}
        with mock.patch.object(self.mod, "_llm", return_value='{"updates": []}'):
            touched = self.mod._update_facts_from_episodes([self._ep()], facts, cursors)
        self.assertEqual(touched, 0)

    def test_update_facts_empty_episode_list_returns_zero(self):
        # Guard clause: no episodes at all → 0 without touching the LLM.
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        with mock.patch.object(self.mod, "_llm") as llm:
            self.assertEqual(self.mod._update_facts_from_episodes([], {}, cursors), 0)
        llm.assert_not_called()

    def test_update_facts_blank_type_left_as_other(self):
        # new record defaults type to "other"; update offers a blank/"other"
        # type → the promotion branch is skipped and "other" stays.
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        facts: dict = {}
        llm_out = json.dumps({"updates": [
            {"entity": "Acme", "type": "", "new_fact": "x", "open_question": ""}]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            self.mod._update_facts_from_episodes([self._ep()], facts, cursors)
        self.assertEqual(facts["Acme"]["type"], "other")

    def test_update_facts_no_entity_episodes_returns_zero(self):
        # Episode has a summary but NO new_entities → summaries_block empty.
        cursors = {"budget_date": self.mod._today_utc(), "budget_used_usd": 0.0}
        with mock.patch.object(self.mod, "_llm") as llm:
            touched = self.mod._update_facts_from_episodes(
                [self._ep(new_entities=[])], {}, cursors)
        self.assertEqual(touched, 0)
        llm.assert_not_called()

    # ── _process_facts_once ───────────────────────────────────────────────
    def test_process_facts_once_no_new_episodes(self):
        cursors = self.mod._default_cursors()
        self.assertEqual(self.mod._process_facts_once(cursors), 0)

    def test_process_facts_once_persists_when_touched(self):
        with open(self.episodes, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "e1", "start_ts": 1.0, "summary": "Sam joined",
                                "topics": [], "mood": "", "new_entities": ["Sam"]}) + "\n")
        cursors = self.mod._default_cursors()
        llm_out = json.dumps({"updates": [
            {"entity": "Sam", "type": "person", "new_fact": "Sam is on the team",
             "open_question": ""}]})
        with mock.patch.object(self.mod, "_llm", return_value=llm_out):
            touched = self.mod._process_facts_once(cursors)
        self.assertEqual(touched, 1)
        self.assertEqual(cursors["episodes_count"], 1)
        self.assertIn("Sam", self.mod._load_facts())

    def test_process_facts_once_no_touch_skips_save(self):
        with open(self.episodes, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "e1", "start_ts": 1.0, "summary": "noise",
                                "topics": [], "mood": "", "new_entities": []}) + "\n")
        cursors = self.mod._default_cursors()
        with mock.patch.object(self.mod, "_save_facts") as save, \
             mock.patch.object(self.mod, "_llm"):
            touched = self.mod._process_facts_once(cursors)
        self.assertEqual(touched, 0)
        save.assert_not_called()
        self.assertEqual(cursors["episodes_count"], 1)   # cursor still advances


class ChappieLoopTests(unittest.TestCase):
    """Drive _chappie_loop through controlled iterations. The infinite loop is
    broken via a sentinel raised from time.sleep (BaseException so the loop's
    own `except Exception` doesn't swallow it)."""
    class _LoopBreak(BaseException):
        pass

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")

    def test_loop_runs_both_layers_and_saves(self):
        # Force both interval gates open by starting the run-clocks far in the
        # past, then break out on the first sleep of the SECOND iteration.
        self.mod._last_episode_run[0] = 0.0
        self.mod._last_fact_run[0] = 0.0
        sleeps = {"n": 0}

        def _sleep(_secs):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise self._LoopBreak
        cursors = self.mod._default_cursors()
        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.time, "time", return_value=1_000_000.0), \
             mock.patch.object(self.mod, "_load_cursors", return_value=cursors), \
             mock.patch.object(self.mod, "_process_episodes_once", return_value=2) as ep, \
             mock.patch.object(self.mod, "_process_facts_once", return_value=3) as fc, \
             mock.patch.object(self.mod, "_save_cursors") as save:
            with self.assertRaises(self._LoopBreak):
                self.mod._chappie_loop()
        ep.assert_called_once()
        fc.assert_called_once()
        save.assert_called_once_with(cursors)
        # Clocks advanced to "now".
        self.assertEqual(self.mod._last_episode_run[0], 1_000_000.0)
        self.assertEqual(self.mod._last_fact_run[0], 1_000_000.0)

    def test_loop_runs_layers_but_no_output_still_saves(self):
        # Intervals elapsed (work runs) but both processors return 0 → the
        # "+N" prints are skipped, yet cursors are still saved.
        self.mod._last_episode_run[0] = 0.0
        self.mod._last_fact_run[0] = 0.0
        sleeps = {"n": 0}

        def _sleep(_secs):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise self._LoopBreak
        cursors = self.mod._default_cursors()
        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.time, "time", return_value=1_000_000.0), \
             mock.patch.object(self.mod, "_load_cursors", return_value=cursors), \
             mock.patch.object(self.mod, "_process_episodes_once", return_value=0), \
             mock.patch.object(self.mod, "_process_facts_once", return_value=0), \
             mock.patch.object(self.mod, "_save_cursors") as save:
            with self.assertRaises(self._LoopBreak):
                self.mod._chappie_loop()
        save.assert_called_once_with(cursors)

    def test_loop_skips_layers_before_intervals_elapse(self):
        # Run-clocks set to "now" → neither interval has elapsed → no work, no
        # cursor save.
        now = 2_000_000.0
        self.mod._last_episode_run[0] = now
        self.mod._last_fact_run[0] = now
        sleeps = {"n": 0}

        def _sleep(_secs):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise self._LoopBreak
        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.time, "time", return_value=now), \
             mock.patch.object(self.mod, "_load_cursors",
                               return_value=self.mod._default_cursors()), \
             mock.patch.object(self.mod, "_process_episodes_once") as ep, \
             mock.patch.object(self.mod, "_process_facts_once") as fc, \
             mock.patch.object(self.mod, "_save_cursors") as save:
            with self.assertRaises(self._LoopBreak):
                self.mod._chappie_loop()
        ep.assert_not_called()
        fc.assert_not_called()
        save.assert_not_called()

    def test_loop_tick_error_is_swallowed(self):
        # A failing tick must not kill the thread; the loop continues to its next
        # sleep (which raises the sentinel on the second pass).
        self.mod._last_episode_run[0] = 0.0
        self.mod._last_fact_run[0] = 0.0
        sleeps = {"n": 0}

        def _sleep(_secs):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise self._LoopBreak
        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.time, "time", return_value=1.0), \
             mock.patch.object(self.mod, "_load_cursors",
                               side_effect=RuntimeError("transient")):
            with self.assertRaises(self._LoopBreak):
                self.mod._chappie_loop()


class ChappieThreadStartTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")

    def test_ensure_thread_started_idempotent(self):
        # Already started by import (module-load fallback). A second call must
        # not spawn another thread.
        self.assertTrue(self.mod._thread_started[0])
        with mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._ensure_thread_started()
        Thread.assert_not_called()

    def test_ensure_thread_started_spawns_when_unstarted(self):
        self.mod._thread_started[0] = False
        with mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._ensure_thread_started()
        Thread.assert_called_once()
        self.assertTrue(Thread.call_args.kwargs.get("daemon"))
        Thread.return_value.start.assert_called_once()
        self.assertTrue(self.mod._thread_started[0])

    def test_register_wires_actions_and_starts_thread(self):
        mod, _ = load_skill_isolated("chappie_consciousness", register=False)
        actions: dict = {}
        with mock.patch.object(mod, "_ensure_thread_started") as ens:
            mod.register(actions)
        self.assertIn("chappie_recall_entity", actions)
        self.assertIn("chappie_recall_today", actions)
        self.assertIn("chappie_status", actions)
        ens.assert_called_once()

    def test_register_actions_alias_is_register(self):
        self.assertIs(self.mod.register_actions, self.mod.register)


class ChappieRecallBranchTests(unittest.TestCase):
    """Remaining recall branches: exact match, bits-less record, recall_today
    keyword hit + read exception + missing-summary, status with no episode file."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("chappie_consciousness")
        self.tmp = tempfile.mkdtemp(prefix="chappie_rb_")
        self.addCleanup(self._cleanup)
        self.episodes = os.path.join(self.tmp, "episodes.jsonl")
        self.facts = os.path.join(self.tmp, "facts.json")
        self.mod._EPISODES_FILE = self.episodes
        self.mod._FACTS_FILE = self.facts

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(self.tmp, fn))
        with contextlib.suppress(OSError):
            os.rmdir(self.tmp)

    def _write_facts(self, data):
        with open(self.facts, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _write_episodes(self, eps):
        with open(self.episodes, "w", encoding="utf-8") as f:
            for ep in eps:
                f.write(json.dumps(ep) + "\n")

    def test_recall_entity_strips_trailing_punctuation(self):
        self._write_facts({"Sam": {"observation_count": 1,
                                   "facts": [{"text": "in IT"}], "open_questions": []}})
        out = self.actions["chappie_recall_entity"]("Sam?")
        self.assertIn("in IT", out)

    def test_recall_entity_substring_scans_past_non_match(self):
        # First key doesn't match; the loop advances and matches the second,
        # exercising the substring-scan continuation.
        self._write_facts({
            "Zebra Corp": {"observation_count": 1, "facts": [{"text": "unrelated"}],
                           "open_questions": []},
            "Acme Industries": {"observation_count": 2,
                                "facts": [{"text": "makes anvils"}],
                                "open_questions": []},
        })
        out = self.actions["chappie_recall_entity"]("acme")
        self.assertIn("makes anvils", out)

    def test_recall_entity_record_without_specifics(self):
        # Record is truthy (so it's "found") but has no facts/observation_count/
        # open_questions → the "on file but no specifics" branch. An empty {}
        # would read as falsy and fall into the not-found path instead.
        self._write_facts({"Ghost": {"type": "person"}})
        out = self.actions["chappie_recall_entity"]("Ghost")
        self.assertIn("on file but I don't have specifics", out)

    def test_recall_today_keyword_hit(self):
        now = time.time()
        self._write_episodes([
            {"start_ts": now - 120, "summary": "talked about the Kubernetes upgrade",
             "topics": ["devops"], "new_entities": ["Kubernetes"]},
            {"start_ts": now - 60, "summary": "lunch plans", "topics": [],
             "new_entities": []},
        ])
        out = self.actions["chappie_recall_today"]("kubernetes")
        self.assertIn("Kubernetes upgrade", out)
        self.assertNotIn("lunch", out)

    def test_recall_today_read_exception(self):
        # File exists but reading it explodes → the "tripped reading" branch.
        self._write_episodes([{"start_ts": time.time(), "summary": "x"}])
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            out = self.actions["chappie_recall_today"]("")
        self.assertIn("tripped reading", out)

    def test_recall_today_skips_blank_and_bad_lines(self):
        now = time.time()
        with open(self.episodes, "w", encoding="utf-8") as f:
            f.write("\n")                                    # blank → skipped
            f.write("{not json}\n")                           # malformed → skipped
            f.write(json.dumps({"start_ts": now - 30,
                                "summary": "real moment"}) + "\n")
        out = self.actions["chappie_recall_today"]("")
        self.assertIn("real moment", out)

    def test_recall_today_missing_summary_placeholder(self):
        now = time.time()
        self._write_episodes([{"start_ts": now - 30, "topics": [], "new_entities": []}])
        out = self.actions["chappie_recall_today"]("")
        self.assertIn("(no summary)", out)

    def test_status_handles_missing_episode_file(self):
        # No episodes file → ep_count 0; facts file absent → 0 records.
        with mock.patch.object(self.mod, "_load_cursors",
                               return_value={"budget_used_usd": 0.0}):
            out = self.actions["chappie_status"]("")
        self.assertIn("0 episode(s)", out)
        self.assertIn("0 entity record(s)", out)

    def test_status_episode_count_read_error(self):
        # Episode file exists but the count read raises → swallowed, count 0.
        self._write_episodes([{"start_ts": 1.0, "summary": "a"}])
        with mock.patch.object(self.mod, "_load_cursors",
                               return_value={"budget_used_usd": 0.0}), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            out = self.actions["chappie_status"]("")
        self.assertIn("0 episode(s)", out)


if __name__ == "__main__":
    unittest.main()
