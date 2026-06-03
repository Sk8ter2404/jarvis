"""Tests for tools/install_git_hooks.py — the pre-commit PII-guard installer.

CI-safe: stdlib only; git is mocked, the hook is written to a temp dir.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import tools.install_git_hooks as igh


class HooksDirTests(unittest.TestCase):
    def test_absolute_path_returned(self):
        absp = os.path.abspath("somehooks")
        fake = mock.Mock(returncode=0, stdout=absp + "\n")
        with mock.patch.object(igh.subprocess, "run", return_value=fake):
            self.assertEqual(igh.hooks_dir(), absp)

    def test_relative_path_joined_to_root(self):
        fake = mock.Mock(returncode=0, stdout="weird/hooks\n")
        with mock.patch.object(igh.subprocess, "run", return_value=fake):
            self.assertEqual(igh.hooks_dir(),
                             os.path.join(igh._PROJECT_ROOT, "weird/hooks"))

    def test_git_nonzero_falls_back(self):
        fake = mock.Mock(returncode=128, stdout="")
        with mock.patch.object(igh.subprocess, "run", return_value=fake):
            self.assertEqual(igh.hooks_dir(),
                             os.path.join(igh._PROJECT_ROOT, ".git", "hooks"))

    def test_git_exception_falls_back(self):
        with mock.patch.object(igh.subprocess, "run", side_effect=OSError("no git")):
            self.assertEqual(igh.hooks_dir(),
                             os.path.join(igh._PROJECT_ROOT, ".git", "hooks"))


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.dir, ignore_errors=True))

    def test_writes_hook_with_lf_endings(self):
        dest = igh.install(target_dir=self.dir)
        self.assertEqual(dest, os.path.join(self.dir, "pre-commit"))
        with open(dest, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)            # POSIX sh needs LF
        self.assertIn(b"check_no_pii.py", raw)
        self.assertTrue(raw.startswith(b"#!/bin/sh"))

    def test_makedirs_failure_returns_none(self):
        with mock.patch.object(igh.os, "makedirs", side_effect=OSError("ro")):
            self.assertIsNone(igh.install(target_dir=self.dir))

    def test_write_failure_returns_none(self):
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            self.assertIsNone(igh.install(target_dir=self.dir))

    def test_chmod_failure_still_succeeds(self):
        with mock.patch.object(igh.os, "chmod", side_effect=OSError("no chmod")):
            dest = igh.install(target_dir=self.dir)
        self.assertIsNotNone(dest)

    def test_default_target_uses_hooks_dir(self):
        with mock.patch.object(igh, "hooks_dir", return_value=self.dir):
            dest = igh.install()
        self.assertEqual(dest, os.path.join(self.dir, "pre-commit"))


class MainTests(unittest.TestCase):
    def test_success(self):
        with mock.patch.object(igh, "install", return_value="/x/pre-commit"):
            self.assertEqual(igh.main(), 0)

    def test_failure(self):
        with mock.patch.object(igh, "install", return_value=None):
            self.assertEqual(igh.main(), 1)


if __name__ == "__main__":
    unittest.main()
