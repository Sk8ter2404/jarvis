#!/usr/bin/env python3
"""Unit tests for tools/auto_publish.py — the pipeline -> reviewable-PR bridge.

Every git/GitHub call is injected (a fake runner + a fake poster), so the
orchestration is verified with no real git, network, or working tree touched.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import auto_publish  # noqa: E402


class FakeRunner:
    """Stand-in for subprocess.run keyed on the git sub-command.

    `results` maps a git sub-command (e.g. "status", "commit") to a
    (returncode, stdout, stderr) tuple; anything unlisted succeeds empty.
    """

    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        sub = cmd[1] if len(cmd) > 1 else ""
        rc, out, err = self.results.get(sub, (0, "", ""))
        return subprocess.CompletedProcess(cmd, rc, out, err)

    def ran(self, sub):
        return any(len(c) > 1 and c[1] == sub for c in self.calls)


class RaisingRunner:
    def __call__(self, cmd, **kw):
        raise OSError("git missing")


class WorkingChangesTests(unittest.TestCase):
    def test_clean_tree_is_empty(self):
        self.assertEqual(auto_publish.working_changes(FakeRunner()), [])

    def test_parses_porcelain_paths(self):
        r = FakeRunner({"status": (0, " M core/x.py\n?? new.py\n", "")})
        self.assertEqual(auto_publish.working_changes(r), ["core/x.py", "new.py"])

    def test_nonzero_returncode_is_empty(self):
        r = FakeRunner({"status": (1, "", "fatal: not a repo")})
        self.assertEqual(auto_publish.working_changes(r), [])

    def test_exception_is_empty(self):
        self.assertEqual(auto_publish.working_changes(RaisingRunner()), [])


class BranchNameTests(unittest.TestCase):
    def test_plain_stamp(self):
        self.assertEqual(auto_publish.make_branch_name("2026-06-03"),
                         "auto/overnight-2026-06-03")

    def test_unsafe_chars_sanitized(self):
        self.assertEqual(auto_publish.make_branch_name("06/03 01:22"),
                         "auto/overnight-06-03-01-22")


class CommitTests(unittest.TestCase):
    def test_branch_create_failure(self):
        r = FakeRunner({"checkout": (1, "", "already exists")})
        ok, detail = auto_publish.commit_to_branch("b", "msg", r)
        self.assertFalse(ok)
        self.assertIn("branch create failed", detail)

    def test_stage_failure(self):
        r = FakeRunner({"add": (1, "", "permission denied")})
        ok, detail = auto_publish.commit_to_branch("b", "msg", r)
        self.assertFalse(ok)
        self.assertIn("stage failed", detail)

    def test_commit_blocked_by_pii_guard(self):
        # the pre-commit guard rejects -> non-zero commit with a message
        r = FakeRunner({"commit": (1, "BLOCKED: possible secret", "")})
        ok, detail = auto_publish.commit_to_branch("b", "msg", r)
        self.assertFalse(ok)
        self.assertIn("commit blocked/failed", detail)
        self.assertIn("BLOCKED", detail)

    def test_success(self):
        r = FakeRunner({"commit": (0, "1 file changed", "")})
        ok, detail = auto_publish.commit_to_branch("b", "msg", r)
        self.assertTrue(ok)
        self.assertTrue(r.ran("checkout") and r.ran("add") and r.ran("commit"))


class PushTests(unittest.TestCase):
    def test_push_success(self):
        self.assertTrue(auto_publish.push_branch("b", FakeRunner()))

    def test_push_failure(self):
        r = FakeRunner({"push": (1, "", "rejected")})
        self.assertFalse(auto_publish.push_branch("b", r))


class TokenTests(unittest.TestCase):
    def test_prefers_jarvis_token(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_GITHUB_TOKEN": "tok-j",
                              "GITHUB_TOKEN": "tok-g"}):
            self.assertEqual(auto_publish._token(), "tok-j")

    def test_falls_back_to_github_token(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("JARVIS_GITHUB_TOKEN", "GITHUB_TOKEN")}
        env["GITHUB_TOKEN"] = "tok-g"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(auto_publish._token(), "tok-g")

    def test_none_when_absent(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("JARVIS_GITHUB_TOKEN", "GITHUB_TOKEN")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(auto_publish._token())


class RunOrchestrationTests(unittest.TestCase):
    def _run(self, argv, runner, poster=None):
        lines = []
        code = auto_publish.run(argv, runner=runner, poster=poster,
                                stamp="t", out=lines.append)
        return code, "\n".join(lines)

    def test_no_changes_short_circuits(self):
        code, text = self._run([], FakeRunner())
        self.assertEqual(code, 0)
        self.assertIn("No working-tree changes", text)

    def test_commit_failure_returns_1_and_never_pushes(self):
        r = FakeRunner({"status": (0, " M x.py\n", ""),
                        "commit": (1, "BLOCKED", "")})
        code, text = self._run([], r)
        self.assertEqual(code, 1)
        self.assertIn("Aborted", text)
        self.assertFalse(r.ran("push"))

    def test_push_failure_returns_1(self):
        r = FakeRunner({"status": (0, " M x.py\n", ""),
                        "push": (1, "", "rejected")})
        code, text = self._run([], r)
        self.assertEqual(code, 1)
        self.assertIn("Push failed", text)

    def test_success_opens_pr(self):
        r = FakeRunner({"status": (0, " M x.py\n", "")})
        seen = {}

        def poster(branch, title, body):
            seen["branch"] = branch
            return "https://github.com/x/y/pull/1"

        code, text = self._run(["--summary", "did things"], r, poster=poster)
        self.assertEqual(code, 0)
        self.assertIn("Pull request opened", text)
        self.assertEqual(seen["branch"], "auto/overnight-t")

    def test_success_pr_open_failure_is_graceful(self):
        r = FakeRunner({"status": (0, " M x.py\n", "")})
        code, text = self._run([], r, poster=lambda *a: None)
        self.assertEqual(code, 0)
        self.assertIn("Couldn't open the PR", text)

    def test_no_pr_flag_skips_pr(self):
        r = FakeRunner({"status": (0, " M x.py\n", "")})
        poster_called = []
        code, text = self._run(["--no-pr"], r,
                               poster=lambda *a: poster_called.append(1))
        self.assertEqual(code, 0)
        self.assertIn("Skipped PR", text)
        self.assertFalse(poster_called)


class PrHeadTests(unittest.TestCase):
    def test_same_repo_when_no_head_owner(self):
        env = {k: v for k, v in os.environ.items()
               if k != "JARVIS_GITHUB_HEAD_OWNER"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(auto_publish._pr_head("feat/x"), "feat/x")

    def test_cross_fork_when_head_owner_set(self):
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_HEAD_OWNER": "alice"}):
            self.assertEqual(auto_publish._pr_head("feat/x"), "alice:feat/x")


if __name__ == "__main__":
    unittest.main()
