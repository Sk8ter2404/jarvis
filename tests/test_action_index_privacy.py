"""docs/ACTION_INDEX.md is PUBLIC: it may only ever name tracked sources.

THE LEAK (2026-09-30). tools/gen_action_index.py globbed every skills/*.py ON
DISK. The owner's machine carries gitignored personal skills - .gitignore keeps
those FILES out of the public repo precisely because their action names embed a
specific person or a one-off personal event - so regenerating the index on the
live tree copied those private action names, with their file:line locations,
into a tracked document. The generator now reads only files git TRACKS (with a
.gitignore fallback when git cannot answer).

Every fixture here is SYNTHETIC ("privateword_*", "wip_*"): a test about a
private-name leak must not spell a private name.
"""
from __future__ import annotations

import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TOOLS = os.path.join(_PROJECT, "tools")


def _load_tool(name):
    """Import tools/<name>.py by path. The generator only runs under
    ``__main__``, so importing it writes nothing."""
    if _TOOLS not in sys.path:
        sys.path.insert(0, _TOOLS)
    spec = importlib.util.spec_from_file_location(
        "jarvis_test_" + name, os.path.join(_TOOLS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_tracked(root):
    """``git ls-files`` as a set of POSIX paths, or None without git."""
    try:
        r = subprocess.run(["git", "-C", root, "ls-files", "-z"],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return {p.decode("utf-8", "replace") for p in r.stdout.split(b"\0") if p}


def _git_ok():
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _write(root, rel, text):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


_SKILL = ('def register(actions):\n'
          '    actions["{name}"] = _handler_{name}\n\n\n'
          'def _handler_{name}(_=""):\n'
          '    return ""\n')


def _make_tree(root):
    """A miniature JARVIS tree: one tracked skill, one GITIGNORED personal
    skill, one UNTRACKED work-in-progress skill, and a tracked monolith."""
    _write(root, "bobert_companion.py",
           'ACTIONS = {"mono_public": _act_mono}\n'
           'INFORMATIVE_ACTIONS = {"mono_public"}\n'
           'SPEAK_RESULT_VERBATIM_ACTIONS = {"public_thing"}\n\n\n'
           'def _act_mono(_=""):\n    return ""\n')
    _write(root, "core/prompts.py", 'P = "[ACTION: mono_public]"\n')
    _write(root, "skills/public_skill.py", _SKILL.format(name="public_thing"))
    _write(root, "skills/privateword_skill.py",
           _SKILL.format(name="privateword_secret_action"))
    _write(root, "skills/wip_skill.py", _SKILL.format(name="wip_untracked_action"))
    _write(root, "tests/test_privateword.py",
           'NAME = "privateword_secret_action"\n')
    _write(root, ".gitignore",
           "skills/privateword_skill.py\ntests/test_privateword.py\n")


class GeneratorIndexesOnlyTrackedFilesTests(unittest.TestCase):

    def setUp(self):
        self.gen = _load_tool("gen_action_index")
        self.tmp = tempfile.mkdtemp(prefix="jarvis_idx_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        _make_tree(self.tmp)

    def _git_add_tracked(self):
        if not _git_ok():
            self.skipTest("git unavailable")
        for args in (["init", "-q"],
                     # The monolith is added by a one-char pathspec GLOB: the
                     # suite's live-data guard (tests/live_data_guard.py)
                     # refuses any subprocess whose argv names the boot
                     # script, and git expands the glob itself.
                     ["add", "bobert_companio?.py", "core/prompts.py",
                      "skills/public_skill.py", ".gitignore"]):
            r = subprocess.run(["git", "-C", self.tmp] + args,
                               capture_output=True, timeout=60)
            if r.returncode != 0:
                self.skipTest("git %s failed: %r" % (args[0], r.stderr[:200]))

    def test_private_and_untracked_skills_never_reach_the_index(self):
        self._git_add_tracked()
        out = self.gen.main(root=self.tmp)
        with open(out, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("`public_thing`", text)
        self.assertIn("`mono_public`", text)
        for leaked in ("privateword", "privateword_skill.py",
                       "wip_untracked_action", "wip_skill.py"):
            self.assertNotIn(leaked, text,
                             "an ignored/untracked source reached the PUBLIC "
                             "index: %r" % leaked)

    def test_a_private_test_file_does_not_feed_the_test_counts(self):
        self._git_add_tracked()
        text, _counts, _groups = self.gen.build_index(self.tmp)
        self.assertNotIn("test_privateword", text)

    def test_gitignored_skill_on_disk_changes_nothing_in_the_output(self):
        """The coverage columns (2026-10-02) read more of each source: a
        skill's speak-set declarations and every test file's string literals.
        A gitignored skills/*.py ON DISK that declares speak routing for a
        PUBLIC action, plus a gitignored test naming that action, must leave
        the output exactly as if neither file existed."""
        _write(self.tmp, "skills/public_quiet.py",
               _SKILL.format(name="public_quiet"))
        _write(self.tmp, "skills/privateword_speaker.py",
               'SPEAK_VERBATIM_ACTIONS = ("public_quiet",)\n\n\n'
               + _SKILL.format(name="privateword_spoken_action"))
        _write(self.tmp, "tests/test_privateword_speaker.py",
               'NAME = "public_quiet"\n')
        with open(os.path.join(self.tmp, ".gitignore"), "a",
                  encoding="utf-8", newline="\n") as f:
            f.write("skills/privateword_speaker.py\n"
                    "tests/test_privateword_speaker.py\n")
        self._git_add_tracked()
        subprocess.run(["git", "-C", self.tmp, "add", "skills/public_quiet.py"],
                       capture_output=True, timeout=60, check=True)
        ignored = subprocess.run(
            ["git", "-C", self.tmp, "check-ignore",
             "skills/privateword_speaker.py",
             "tests/test_privateword_speaker.py"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(ignored.stdout.split(),
                         ["skills/privateword_speaker.py",
                          "tests/test_privateword_speaker.py"],
                         "fixture is vacuous: git does not ignore the files")

        with open(self.gen.main(root=self.tmp), encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("privateword", text)
        rows = [ln for ln in text.splitlines()
                if ln.startswith("| `public_quiet` |")]
        self.assertEqual(len(rows), 1, text)
        cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        self.assertEqual(cells[2], "neither",
                         "a gitignored skill's speak declaration reached the "
                         "public index's spoken note")
        self.assertEqual(cells[4], "no",
                         "a gitignored test file counted as a test reference")

    def test_no_git_fallback_still_drops_gitignored_files(self):
        """When git cannot answer, the .gitignore patterns decide - and they
        can only ever EXCLUDE. The fallback also says so, loudly."""
        err = io.StringIO()
        with mock.patch.object(self.gen, "git_tracked_files", return_value=None), \
                redirect_stderr(err):
            text, _c, _g = self.gen.build_index(self.tmp)
        self.assertIn("`public_thing`", text)
        self.assertNotIn("privateword", text)
        self.assertIn("WARNING", err.getvalue())

    def test_publishable_sources_with_a_known_tracked_set(self):
        paths = [os.path.join(self.tmp, "skills", n)
                 for n in ("public_skill.py", "privateword_skill.py",
                           "wip_skill.py")]
        kept = self.gen.publishable_sources(
            paths, self.tmp, tracked={"skills/public_skill.py"})
        self.assertEqual([os.path.basename(p) for p in kept],
                         ["public_skill.py"])

    def test_fallback_pattern_matcher(self):
        pats = ["skills/privateword_skill.py", "data/*", "*.log", "logs/"]
        m = self.gen._ignored_by_patterns
        self.assertTrue(m("skills/privateword_skill.py", pats))
        self.assertTrue(m("data/x.json", pats))
        self.assertTrue(m("a/b/c.log", pats))
        self.assertTrue(m("logs/session.txt", pats))
        self.assertFalse(m("skills/public_skill.py", pats))


class CommittedIndexIsTrackedOnlyTests(unittest.TestCase):
    """The committed index, checked against THIS checkout's git index."""

    def setUp(self):
        # Deliberately NOT through the generator module: this checks the
        # COMMITTED document, and must not depend on (or be rewritten by) the
        # code that produces it.
        self.tracked = _git_tracked(_PROJECT)
        if self.tracked is None:
            self.skipTest("not a git checkout / git unavailable")
        p = os.path.join(_PROJECT, "docs", "ACTION_INDEX.md")
        if not os.path.exists(p):
            self.skipTest("ACTION_INDEX.md not present in this checkout")
        with open(p, encoding="utf-8") as f:
            self.index = f.read()

    def test_every_location_names_a_tracked_file(self):
        locs = set(re.findall(r"`([A-Za-z0-9_./-]+\.py):\d+`", self.index))
        self.assertTrue(locs, "no file:line locations parsed - parser rotted")
        bad = sorted(p for p in locs if p not in self.tracked)
        self.assertEqual(bad, [],
                         "the PUBLIC index cites files git does not track "
                         "(regenerate with tools/gen_action_index.py): %s" % bad)

    def test_every_indexed_action_is_registered_by_a_tracked_file(self):
        rs = _load_tool("registration_scan")
        registered = set(rs.scan_file(
            os.path.join(_PROJECT, "bobert_companion.py"),
            targets=("ACTIONS",), filename="bobert_companion.py"))
        for rel in sorted(self.tracked):
            if not rel.endswith(".py"):
                continue
            if not (rel.startswith("core/") or rel.startswith("skills/")):
                continue
            if rel.count("/") > 2:
                continue
            try:
                registered |= set(rs.scan_file(os.path.join(_PROJECT, rel),
                                               filename=rel))
            except (SyntaxError, OSError):
                continue
        names = set()
        in_table = False
        for line in self.index.splitlines():
            if line.startswith("## Full index"):
                in_table = True
                continue
            if in_table and line.startswith("| `"):
                first = line.split("|")[1]
                names |= set(re.findall(r"`([^`]+)`", first))
        self.assertGreater(len(names), 100, "index table parse rotted")
        stray = sorted(names - registered)
        self.assertEqual(stray, [],
                         "the PUBLIC index lists %d action(s) no tracked file "
                         "registers (a private/untracked skill leaked in, or "
                         "the index is stale)" % len(stray))


if __name__ == "__main__":
    unittest.main()
