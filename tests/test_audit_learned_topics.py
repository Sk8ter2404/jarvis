"""Tests for tools/audit_learned_topics.py — the audit of topics/projects
learned BEFORE the 2026-09-29 write gate existed.

The owner's standing rule is "clean up = tidy, never delete", so the contract
pinned here is mostly about what the tool must NOT do:
  * the default run is a DRY RUN — the store's bytes are untouched;
  * --quarantine MOVES suspects into the store's own "quarantined" section in
    one atomic write — every original item is still in the file afterwards;
  * --restore puts them back exactly where they were;
  * the semantic long-term store is REPORTED, never modified.

Temp-dir fixtures only (generic made-up non-words); never the live store.

    python -m unittest tests.test_audit_learned_topics
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest

from core import topic_hygiene as th
from tools import audit_learned_topics as tool

_MEM = {
    "first_meeting": "2026-01-01",
    "facts": ["User likes tea"],
    "projects": ["Building a garden shed", "Zorblat supplement research",
                 "restaurant venture", "Bird feeder"],
    "topics": [
        {"date": "2026-09-01", "ts": 1000.0, "location": "desk",
         "topic": "garden shed roof"},
        {"date": "2026-09-02", "ts": 2000.0, "location": "desk",
         "topic": "Flemwick mystery"},
        {"date": "2026-09-03", "ts": 3000.0, "location": "desk",
         "topic": "garden shed build"},
        {"date": "2026-09-04", "ts": 4000.0, "location": "desk",
         "topic": "weather check"},
        {"date": "2026-09-05", "ts": 5000.0, "location": "desk",
         "topic": "restaurant venture"},
    ],
    "sessions": [],
}
_LOG = ["what's the weather like", "check the weather for tomorrow",
        "how is the bird feeder print", "remind me about the bird feeder",
        "open the browser"]
_SUSPECTS = {("topic", "Flemwick mystery"), ("topic", "restaurant venture"),
             ("project", "Zorblat supplement research"),
             ("project", "restaurant venture")}


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit_topics_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = os.path.join(self.tmp, "bobert_memory.json")
        with open(self.store, "w", encoding="utf-8") as fh:
            json.dump(_MEM, fh, indent=2)
        os.makedirs(os.path.join(self.tmp, "memory"))
        with open(os.path.join(self.tmp, "memory", "voice_commands.jsonl"),
                  "w", encoding="utf-8") as fh:
            for i, t in enumerate(_LOG):
                fh.write(json.dumps({"ts": 9000.0 + i, "text": t}) + "\n")

    def run_tool(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = tool.main(["--store", self.store, *args])
        return code, buf.getvalue()

    def digest(self):
        with open(self.store, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def load(self):
        with open(self.store, encoding="utf-8") as fh:
            return json.load(fh)


class DryRunTests(_Base):
    def test_default_run_lists_suspects_and_writes_nothing(self):
        before = self.digest()
        code, out = self.run_tool()
        self.assertEqual(code, 0)
        self.assertEqual(self.digest(), before)
        self.assertIn("DRY RUN - nothing written", out)
        for _kind, text in _SUSPECTS:
            self.assertIn(repr(text), out)
        self.assertIn("reason:", out)
        # Real, corroborated entries are not suspects.
        self.assertNotIn("'garden shed roof'", out)
        self.assertNotIn("'Bird feeder'", out)
        # The exact next command is printed for the operator.
        self.assertIn("--quarantine", out)

    def test_missing_owner_log_runs_the_non_word_rule_only(self):
        os.remove(os.path.join(self.tmp, "memory", "voice_commands.jsonl"))
        code, out = self.run_tool()
        self.assertEqual(code, 0)
        self.assertIn("NOT FOUND", out)
        self.assertIn("'Flemwick mystery'", out)
        self.assertNotIn("'restaurant venture'", out)

    def test_missing_store_is_an_error_not_a_crash(self):
        os.remove(self.store)
        code, out = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("no memory store", out)


class QuarantineTests(_Base):
    def test_moves_never_deletes(self):
        code, out = self.run_tool("--quarantine")
        self.assertEqual(code, 0, out)
        mem = self.load()
        live = {("topic", t["topic"]) for t in mem["topics"]} | \
               {("project", p) for p in mem["projects"]}
        held = {(r["kind"], r["item"]["topic"] if r["kind"] == "topic"
                 else r["item"]) for r in mem[th.QUARANTINE_KEY]}
        self.assertEqual(held, _SUSPECTS)
        self.assertFalse(live & held)
        # Every original item is still in the file, whole.
        original = {("topic", t["topic"]) for t in _MEM["topics"]} | \
                   {("project", p) for p in _MEM["projects"]}
        self.assertEqual(live | held, original)
        for r in mem[th.QUARANTINE_KEY]:
            self.assertTrue(r["reason"])
            self.assertTrue(r["quarantined_at"])
        # Unrelated sections untouched.
        self.assertEqual(mem["facts"], _MEM["facts"])
        self.assertEqual(mem["first_meeting"], _MEM["first_meeting"])

    def test_second_quarantine_is_a_no_op(self):
        self.run_tool("--quarantine")
        after_first = self.digest()
        code, out = self.run_tool("--quarantine")
        self.assertEqual(code, 0)
        self.assertEqual(self.digest(), after_first)
        self.assertIn("no suspects", out)

    def test_restore_round_trips_exactly(self):
        self.run_tool("--quarantine")
        code, out = self.run_tool("--restore")
        self.assertEqual(code, 0, out)
        mem = self.load()
        self.assertEqual(mem.pop(th.QUARANTINE_KEY), [])
        self.assertEqual(mem, _MEM)

    def test_restore_with_nothing_quarantined(self):
        code, out = self.run_tool("--restore")
        self.assertEqual(code, 0)
        self.assertIn("nothing to restore", out)

    def test_restore_keeps_malformed_records(self):
        mem = dict(_MEM)
        mem[th.QUARANTINE_KEY] = [{"kind": "project", "item": 42},
                                  {"kind": "mystery", "item": "x"}, "junk"]
        new, n = tool.restore(mem)
        self.assertEqual(n, 0)
        self.assertEqual(len(new[th.QUARANTINE_KEY]), 3)

    def test_quarantine_does_not_mutate_its_input(self):
        mem = json.loads(json.dumps(_MEM))
        suspects = th.find_suspects(mem, _LOG)
        tool.quarantine(mem, suspects)
        self.assertEqual(mem, _MEM)


class PerItemTests(_Base):
    """--only / --pick / --list-quarantined: the owner chooses item by item.
    A selection without --quarantine is still a dry run; an unknown label or
    index refuses the whole run; nothing is ever deleted."""

    def test_selection_without_quarantine_is_a_dry_run(self):
        before = self.digest()
        code, out = self.run_tool("--pick", "topic:1")
        self.assertEqual(code, 0)
        self.assertEqual(self.digest(), before)
        self.assertIn("selected 1 item(s)", out)
        self.assertIn("'Flemwick mystery'", out)
        self.assertIn("--quarantine --pick topic:1", out)

    def test_pick_moves_only_the_picked_items(self):
        code, out = self.run_tool("--quarantine", "--pick", "topic:1,project:2")
        self.assertEqual(code, 0, out)
        mem = self.load()
        held = [(r["kind"], r["index"]) for r in mem[th.QUARANTINE_KEY]]
        self.assertEqual(sorted(held), [("project", 2), ("topic", 1)])
        self.assertNotIn("Flemwick mystery",
                         [t["topic"] for t in mem["topics"]])
        # The other suspects are untouched.
        self.assertIn("restaurant venture", [t["topic"] for t in mem["topics"]])
        self.assertIn("Zorblat supplement research", mem["projects"])
        self.assertNotIn("restaurant venture", mem["projects"])

    def test_only_selects_by_exact_label_even_if_not_flagged(self):
        code, out = self.run_tool("--quarantine", "--only", "Bird feeder",
                                  "--only", "weather check")
        self.assertEqual(code, 0, out)
        mem = self.load()
        recs = {(r["kind"], _label(r)): r["reason"]
                for r in mem[th.QUARANTINE_KEY]}
        self.assertEqual(set(recs), {("project", "Bird feeder"),
                                     ("topic", "weather check")})
        self.assertTrue(all(r == tool.OWNER_SELECTED for r in recs.values()))

    def test_unknown_label_or_index_refuses_and_writes_nothing(self):
        before = self.digest()
        for args in (("--quarantine", "--only", "no such label"),
                     ("--quarantine", "--pick", "topic:99"),
                     ("--quarantine", "--pick", "topic:1,bogus"),
                     ("--quarantine", "--pick", "topic:1", "--only", "nope")):
            with self.subTest(args=args):
                code, out = self.run_tool(*args)
                self.assertEqual(code, 2)
                self.assertIn("refused, nothing written", out)
                self.assertEqual(self.digest(), before)

    def test_list_quarantined_is_read_only(self):
        self.run_tool("--quarantine")
        before = self.digest()
        code, out = self.run_tool("--list-quarantined")
        self.assertEqual(code, 0)
        self.assertEqual(self.digest(), before)
        self.assertIn("4 quarantined entries", out)
        self.assertIn("[quarantined #  0]", out)
        self.assertIn("reason:", out)

    def test_restore_one_entry_leaves_the_rest_quarantined(self):
        self.run_tool("--quarantine")
        mem = self.load()
        first = _label(mem[th.QUARANTINE_KEY][0])
        code, out = self.run_tool("--restore", "--pick", "quarantined:0")
        self.assertEqual(code, 0, out)
        mem = self.load()
        self.assertEqual(len(mem[th.QUARANTINE_KEY]), 3)
        self.assertIn(first, [t["topic"] for t in mem["topics"]]
                      + mem["projects"])
        code, out = self.run_tool("--restore", "--only",
                                  "Zorblat supplement research")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.load()[th.QUARANTINE_KEY]), 2)

    def test_restore_selection_errors_refuse(self):
        self.run_tool("--quarantine")
        before = self.digest()
        for args in (("--restore", "--pick", "topic:0"),
                     ("--restore", "--pick", "quarantined:9"),
                     ("--restore", "--only", "never quarantined")):
            with self.subTest(args=args):
                code, out = self.run_tool(*args)
                self.assertEqual(code, 2)
                self.assertEqual(self.digest(), before)

    def test_facts_vocabulary_reaches_the_audit(self):
        mem = self.load()
        mem["facts"] += ["User runs support for the Flemwick office",
                         "Flemwick pays on the first of the month"]
        with open(self.store, "w", encoding="utf-8") as fh:
            json.dump(mem, fh)
        _code, out = self.run_tool()
        self.assertNotIn("mostly unrecognised words (flemwick)", out)


def _label(rec):
    item = rec["item"]
    return item["topic"] if rec["kind"] == "topic" else item


class SemanticStoreReportTests(_Base):
    def test_ltm_mirrors_are_reported_and_left_alone(self):
        ltm_dir = os.path.join(self.tmp, "data", "long_term_memory")
        os.makedirs(ltm_dir)
        facts = os.path.join(ltm_dir, "facts.json")
        rows = [{"id": "fact_1", "text": "Zorblat supplement research",
                 "tags": ["project"]},
                {"id": "fact_2", "text": "Bird feeder", "tags": ["project"]},
                {"id": "fact_3", "text": "User likes tea", "tags": ["learned"]},
                # Trimmed out of bobert_memory.json long ago, still recalled.
                {"id": "fact_4", "text": "Flemwick grundlefax build",
                 "tags": ["project"]}]
        with open(facts, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        with open(facts, "rb") as fh:
            before = fh.read()
        _code, out = self.run_tool("--quarantine")
        self.assertIn("report only", out)
        self.assertIn("fact_1", out)
        self.assertIn("fact_4", out)
        self.assertNotIn("fact_2", out)
        self.assertNotIn("fact_3", out)
        with open(facts, "rb") as fh:
            self.assertEqual(fh.read(), before)


if __name__ == "__main__":
    unittest.main()
