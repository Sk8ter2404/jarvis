"""Tests for core.version — the single source of truth for the release string.

The module reads the top-level VERSION file once at import and exposes it as
``__version__`` / ``VERSION`` / ``version_string()``. Pins: the import-time read
returns the live VERSION-file contents, ``version_string()`` echoes
``__version__``, and the ``_read_version`` helper degrades to the ``0.0.0-dev``
fallback when the file is missing/unreadable or empty (so a packaging slip never
crashes the import-light tier). The real VERSION file is never modified; the
helper is exercised against a temp path or a patched ``open``.

stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from core import version as ver


class VersionConstantsTests(unittest.TestCase):
    def test_module_attrs_present_and_consistent(self):
        # __version__, VERSION and version_string() all agree.
        self.assertIsInstance(ver.__version__, str)
        self.assertEqual(ver.VERSION, ver.__version__)
        self.assertEqual(ver.version_string(), ver.__version__)

    def test_version_is_nonempty(self):
        # Whatever the VERSION file holds (or the fallback), it's never blank.
        self.assertTrue(ver.version_string().strip())


class ReadVersionTests(unittest.TestCase):
    def test_reads_file_contents_stripped(self):
        # A VERSION file with surrounding whitespace reads back trimmed.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "VERSION")
            with open(path, "w", encoding="utf-8") as f:
                f.write("  1.2.3-test\n")
            with mock.patch.object(ver, "_VERSION_FILE", path):
                self.assertEqual(ver._read_version(), "1.2.3-test")

    def test_empty_file_falls_back(self):
        # A present-but-empty VERSION file → the _FALLBACK sentinel.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "VERSION")
            with open(path, "w", encoding="utf-8") as f:
                f.write("   \n")
            with mock.patch.object(ver, "_VERSION_FILE", path):
                self.assertEqual(ver._read_version(), ver._FALLBACK)

    def test_missing_file_falls_back(self):
        # No VERSION file at the configured path → fallback, no raise.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "does-not-exist", "VERSION")
            with mock.patch.object(ver, "_VERSION_FILE", path):
                self.assertEqual(ver._read_version(), ver._FALLBACK)

    def test_oserror_on_open_falls_back(self):
        # Any OSError opening the file (permissions, etc.) degrades to fallback.
        with mock.patch.object(ver, "open", create=True,
                               side_effect=OSError("nope")):
            self.assertEqual(ver._read_version(), ver._FALLBACK)


def _git(root, *args, when=None):
    """Run git in a throwaway repo; `when` pins the commit's date (epoch s)."""
    env = dict(os.environ)
    if when is not None:
        env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = f"{int(when)} +0000"
    subprocess.run(["git", "-C", root, *args], check=True, capture_output=True,
                   text=True, env=env)


def _repo(root, version="1.2.3", when=1790600000, tag=True):
    """A throwaway git checkout whose VERSION was committed at `when`."""
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.com")
    _git(root, "config", "user.name", "t")
    with open(os.path.join(root, "VERSION"), "w", encoding="utf-8") as f:
        f.write(version + "\n")
    _git(root, "add", "VERSION")
    _git(root, "commit", "-qm", "release", when=when)
    if tag:
        _git(root, "tag", f"v{version}")


@unittest.skipIf(shutil.which("git") is None, "git not on PATH")
class ReleaseTimestampTests(unittest.TestCase):
    """release_timestamp: the date of the release on disk comes from git (a
    git release never writes data/version.json, so that file's date went four
    months stale - 2026-10-02), the VERSION mtime only outside a checkout."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="relts_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_tag_date_wins_over_a_newer_version_mtime(self):
        _repo(self.root, when=1790600000)
        os.utime(os.path.join(self.root, "VERSION"), (1795000000, 1795000000))
        self.assertEqual(ver.release_timestamp(self.root), 1790600000)

    def test_untagged_release_uses_the_commit_that_set_version(self):
        _repo(self.root, when=1790600000, tag=False)
        with open(os.path.join(self.root, "other.py"), "w") as f:
            f.write("x = 1\n")
        _git(self.root, "add", "other.py")
        _git(self.root, "commit", "-qm", "later work", when=1790700000)
        self.assertEqual(ver.release_timestamp(self.root), 1790600000)

    def test_a_copy_that_is_not_a_checkout_uses_the_version_mtime(self):
        path = os.path.join(self.root, "VERSION")
        with open(path, "w", encoding="utf-8") as f:
            f.write("1.2.3\n")
        os.utime(path, (1700000000, 1700000000))
        self.assertEqual(ver.release_timestamp(self.root), 1700000000)

    def test_a_dir_inside_another_repo_never_borrows_its_dates(self):
        _repo(self.root, when=1790600000)
        sub = os.path.join(self.root, "copy")
        os.makedirs(sub)
        path = os.path.join(sub, "VERSION")
        with open(path, "w", encoding="utf-8") as f:
            f.write("1.2.3\n")
        os.utime(path, (1700000000, 1700000000))
        self.assertEqual(ver.release_timestamp(sub), 1700000000)

    def test_nothing_known_is_none(self):
        self.assertIsNone(ver.release_timestamp(self.root))

    def test_asking_again_runs_no_git(self):
        """"What version are you" runs on the action path, and each uncached
        answer was 2-3 git subprocesses at up to 3 s apiece (2026-10-02
        review). The release on disk has not changed, so neither has its date."""
        _repo(self.root, when=1790600000)
        self.assertEqual(ver.release_timestamp(self.root), 1790600000)
        with mock.patch("subprocess.run", wraps=subprocess.run) as run:
            self.assertEqual(ver.release_timestamp(self.root), 1790600000)
        run.assert_not_called()

    def test_a_new_release_on_disk_is_read_again(self):
        _repo(self.root, when=1790600000)
        self.assertEqual(ver.release_timestamp(self.root), 1790600000)
        path = os.path.join(self.root, "VERSION")
        with open(path, "w", encoding="utf-8") as f:
            f.write("1.2.4\n")
        _git(self.root, "add", "VERSION")
        _git(self.root, "commit", "-qm", "release 1.2.4", when=1790700000)
        _git(self.root, "tag", "v1.2.4")
        os.utime(path, (1795000000, 1795000000))
        self.assertEqual(ver.release_timestamp(self.root), 1790700000)

    def test_an_answer_git_did_not_give_is_not_kept(self):
        # git timed out: the VERSION mtime stands in this once, and the next
        # ask goes back to git rather than keeping the stand-in.
        _repo(self.root, when=1790600000)
        os.utime(os.path.join(self.root, "VERSION"), (1700000000, 1700000000))
        with mock.patch("subprocess.run",
                        side_effect=subprocess.TimeoutExpired("git", 3)):
            self.assertEqual(ver.release_timestamp(self.root), 1700000000)
        self.assertEqual(ver.release_timestamp(self.root), 1790600000)


if __name__ == "__main__":
    unittest.main()
