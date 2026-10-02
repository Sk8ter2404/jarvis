"""docs/ACTION_INDEX.md is a machine-checked, never-stale inventory.

THE GAP (2026-10-02). tools/gen_action_index.py produced an accurate index,
but nothing made anyone run it: the committed file had drifted (registered
actions were missing from it) and no test noticed. Its coverage columns were
also per HANDLER GROUP, so an untested alias hid behind a tested sibling, and
the speak column ignored a skill's own module-level speak-set declarations, so
every action a skill routes itself read as "neither".

The generator now emits one row per action with two coverage columns -
``spoken note`` and ``tested`` (see the index's header for both conventions) -
and this file holds the guards:

  * DRIFT - the SET of action names in the committed index must equal the set
    the generator finds now. Names only: file:line locations move on every
    commit and are deliberately never compared.
  * RATCHET - the number of untested actions (no git-tracked tests/**/*.py
    file names the action as a string literal) must not grow, and an action
    that is untested now but is not listed as untested in the committed index
    fails BY NAME. A short, reasoned allow-list covers the genuinely untestable.
  * The two coverage columns themselves, on synthetic trees.

Static reads only: the generator parses with ast and never imports the
monolith, so all of this runs on the Linux CI runner. Every fixture name here
is synthetic, and no real action name may be written as a string literal in
this file - it would count as that action's test reference.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TOOLS = os.path.join(_PROJECT, "tools")
_INDEX = os.path.join(_PROJECT, "docs", "ACTION_INDEX.md")
_REGENERATE = "python tools/gen_action_index.py"


def _load_generator():
    """Import tools/gen_action_index.py by path. It only writes under
    ``__main__``, so importing it writes nothing."""
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    spec = importlib.util.spec_from_file_location(
        "jarvis_test_gen_action_index_coverage",
        os.path.join(_TOOLS, "gen_action_index.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_generator()


# The ceiling for UNTESTED actions: registered actions that no git-tracked
# tests/**/*.py file names as a string literal (the index's `tested` column),
# not counting _UNTESTABLE below. It may only go DOWN. When you add tests that
# name previously untested actions, LOWER this to the new count (the index's
# Summary table prints it) so the ratchet keeps what you won. Never raise it
# to get green - write the test instead.
MAX_UNTESTED = 69

# Genuinely untestable actions, exempt from the ratchet. One per line:
#     <action_name> :: <one-line reason>
# Deliberately ONE string literal, not a set of names: the `tested` column
# counts any string literal under tests/ that equals an action name, so a set
# of names here would mark every entry "tested" and hide it. Keep it short;
# every entry must be registered and still untested (a stale one fails).
_UNTESTABLE = """
"""
_MAX_UNTESTABLE = 10


def _untestable():
    """``({name: reason}, [malformed lines])`` parsed from _UNTESTABLE."""
    out, bad = {}, []
    for line in _UNTESTABLE.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, reason = (p.strip() for p in line.partition("::"))
        if sep and re.fullmatch(r"[A-Za-z0-9_]+", name) and reason:
            out[name] = reason
        else:
            bad.append(line)
    return out, bad


def _parse_full_index(text):
    """``{action: {column header: cell}}`` from an index's "Full index" table.
    Read WITHOUT the generator, so the committed document is checked on its
    own terms. A first cell naming several actions gives each the same row."""
    rows, header, in_table = {}, None, False
    for line in text.splitlines():
        if line.startswith("## "):
            in_table = line.startswith("## Full index")
            continue
        if not in_table or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if "`" not in cells[0]:
            continue                       # the |---| separator
        row = dict(zip(header, cells))
        for name in re.findall(r"`([^`]+)`", cells[0]):
            rows[name] = row
    return rows


def _committed_index(testcase):
    if not os.path.exists(_INDEX):
        testcase.skipTest("ACTION_INDEX.md not present in this checkout")
    with open(_INDEX, encoding="utf-8") as f:
        return _parse_full_index(f.read())


_LIVE = {}


def _live_rows(testcase):
    """The generator's rows for THIS checkout, computed once per run. Skips
    when git cannot list tracked files: the .gitignore fallback could index
    untracked work in progress and report drift that is not there."""
    if "rows" not in _LIVE:
        tracked = GEN.git_tracked_files(_PROJECT)
        _LIVE["rows"] = (None if tracked is None
                         else GEN.collect_rows(_PROJECT, tracked=tracked))
    if _LIVE["rows"] is None:
        testcase.skipTest("git cannot list tracked files in this checkout")
    return _LIVE["rows"]


class CommittedIndexDriftTests(unittest.TestCase):

    def test_committed_names_equal_generated_names(self):
        committed = set(_committed_index(self))
        generated = {r["action"] for r in _live_rows(self)}
        self.assertGreater(len(committed), 100, "index table parse rotted")
        added = sorted(generated - committed)
        removed = sorted(committed - generated)
        if added or removed:
            self.fail(
                "docs/ACTION_INDEX.md is stale: its action names differ from "
                "what the generator finds now.\n"
                "  registered, missing from the index (%d): %s\n"
                "  in the index, no longer registered (%d): %s\n"
                "Regenerate it and commit the result:  %s"
                % (len(added), ", ".join(added) or "-",
                   len(removed), ", ".join(removed) or "-", _REGENERATE))


class UntestedActionRatchetTests(unittest.TestCase):

    def setUp(self):
        rows = _live_rows(self)
        self.registered = {r["action"] for r in rows}
        self.untested_all = {r["action"] for r in rows if not r["tested"]}
        self.untestable, self.malformed = _untestable()
        self.untested = self.untested_all - set(self.untestable)

    def _newcomers(self):
        """Untested now, but NOT listed as untested in the committed index:
        an action new since the index was generated, or one whose only test
        reference was removed."""
        committed = _committed_index(self)
        return sorted(n for n in self.untested
                      if committed.get(n, {}).get("tested") != "no")

    _HOW = ("Add a test that names the action as a string literal (a dispatch "
            "or registration test). Only if it is genuinely untestable, add "
            "`name :: one-line reason` to _UNTESTABLE in "
            "tests/test_action_index_coverage.py.")

    def test_no_action_is_newly_untested(self):
        newcomers = self._newcomers()
        if newcomers:
            self.fail("action(s) with no test reference that "
                      "docs/ACTION_INDEX.md does not list as untested "
                      "(%d): %s\n%s"
                      % (len(newcomers), ", ".join(newcomers), self._HOW))

    def test_untested_count_does_not_grow(self):
        n = len(self.untested)
        if n <= MAX_UNTESTED:
            return
        newcomers = self._newcomers()
        if newcomers:
            named = "New since the committed index: " + ", ".join(newcomers)
        else:
            named = ("The committed index was regenerated after the newcomer "
                     "was added, so it is one of these: "
                     + ", ".join(sorted(self.untested)))
        self.fail("%d registered actions have no test reference; the ceiling "
                  "is MAX_UNTESTED = %d and it may only go down.\n%s\n%s"
                  % (n, MAX_UNTESTED, named, self._HOW))

    def test_untestable_allowlist_is_small_and_current(self):
        self.assertEqual(self.malformed, [],
                         "every _UNTESTABLE line is `name :: one-line reason`")
        self.assertLessEqual(len(self.untestable), _MAX_UNTESTABLE,
                             "_UNTESTABLE is for the genuinely untestable few")
        unknown = sorted(set(self.untestable) - self.registered)
        self.assertEqual(unknown, [],
                         "_UNTESTABLE names action(s) nothing registers "
                         "any more - delete those lines")
        tested = sorted(set(self.untestable) - self.untested_all)
        self.assertEqual(tested, [],
                         "_UNTESTABLE action(s) now have a test reference - "
                         "delete those lines")


def _write(root, rel, text, bom=False):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8-sig" if bom else "utf-8",
              newline="\n") as f:
        f.write(text)


class CoverageColumnsTests(unittest.TestCase):
    """The two coverage columns, on a synthetic tree whose tracked set is
    passed explicitly (no git needed). One handler carries four aliases so a
    per-handler column could not tell them apart."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jarvis_idx_cov_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        files = {
            "bobert_companion.py":
                'ACTIONS = {"mono_said": _act_mono, "mono_quiet": _act_mono}\n'
                'INFORMATIVE_ACTIONS = {"mono_said"}\n'
                'SPEAK_RESULT_VERBATIM_ACTIONS = {"mono_absent"}\n\n\n'
                'def _act_mono(_=""):\n    return ""\n',
            "core/prompts.py": 'P = "[ACTION: fx_status]"\n',
            "skills/fx_skill.py":
                'SPEAK_VERBATIM_ACTIONS = ("fx_status",)\n'
                'INFORMATIVE_ACTIONS: tuple = ("fx_report",)\n'
                'SELF_VOICED_ACTIONS = frozenset({"FX_Chat"})\n\n\n'
                'def register(actions):\n'
                '    for alias in ("fx_status", "fx_report", "fx_chat",\n'
                '                  "fx_plain"):\n'
                '        actions[alias] = _fx\n\n\n'
                'def _fx(_=""):\n    return ""\n',
            "tests/test_fx.py":
                'NAMED = "fx_status"\n'
                '# "fx_report" in a comment is not a reference\n'
                'PROSE = "then fx_chat runs"\n',
        }
        for rel, text in files.items():
            _write(self.tmp, rel, text)
        # A BOM-prefixed test file in a sub-package still counts.
        _write(self.tmp, "tests/sub/test_fx_bom.py", 'X = ["fx_plain"]\n',
               bom=True)
        self.tracked = set(files) | {"tests/sub/test_fx_bom.py"}

    def _index(self):
        text, counts, _n = GEN.build_index(self.tmp, tracked=self.tracked)
        return text, counts, _parse_full_index(text)

    def test_one_row_per_action_with_both_columns(self):
        _text, _c, rows = self._index()
        self.assertEqual(
            set(rows), {"mono_said", "mono_quiet", "fx_status", "fx_report",
                        "fx_chat", "fx_plain"})
        for name, row in rows.items():
            self.assertEqual(re.findall(r"`([^`]+)`", row["action"]), [name],
                             "every action gets its own row")
            self.assertIn("spoken note", row)
            self.assertIn("tested", row)

    def test_spoken_note_reads_skill_declarations(self):
        _text, c, rows = self._index()
        spoken = {n: r["spoken note"].strip("*") for n, r in rows.items()}
        self.assertEqual(spoken, {
            "fx_status": "VERBATIM",        # skill SPEAK_VERBATIM_ACTIONS
            "fx_report": "INFORMATIVE",     # skill INFORMATIVE_ACTIONS
            "fx_chat": "SELF-VOICED",       # skill SELF_VOICED_ACTIONS
            "fx_plain": "neither",
            "mono_said": "INFORMATIVE",     # monolith INFORMATIVE_ACTIONS
            "mono_quiet": "neither",
        })
        self.assertEqual((c["verbatim"], c["informative"], c["self_voiced"],
                          c["neither"]), (1, 2, 1, 2))

    def test_tested_is_an_exact_string_literal_per_action(self):
        _text, c, rows = self._index()
        tested = {n: r["tested"] for n, r in rows.items()}
        self.assertEqual(tested, {
            "fx_status": "yes",       # NAMED = "fx_status"
            "fx_plain": "yes",        # BOM file, tests/ sub-directory
            "fx_report": "no",        # only inside a comment
            "fx_chat": "no",          # only inside a longer string
            "mono_said": "no",
            "mono_quiet": "no",
        })
        self.assertEqual((c["tested"], c["no_tests"]), (2, 4))

    def test_summary_counts_are_at_the_top(self):
        text, _c, _rows = self._index()
        summary = text.index("## Summary")
        self.assertLess(summary, text.index("## Full index"))
        head = text[summary:text.index("## Full index")]
        self.assertIn("| tested | 2 |", head)
        self.assertIn("| **untested** (no test names it) | 4 |", head)
        self.assertIn("| **no spoken note** (neither) | 2 |", head)

    def test_untracked_test_file_is_not_a_reference(self):
        _write(self.tmp, "tests/test_local_wip.py", 'N = "mono_quiet"\n')
        _text, _c, rows = self._index()
        self.assertEqual(rows["mono_quiet"]["tested"], "no")


if __name__ == "__main__":
    unittest.main()
