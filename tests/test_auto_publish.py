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
    (returncode, stdout, stderr) tuple; anything unlisted succeeds empty —
    EXCEPT `check-ignore`, which defaults to rc 1 ("not ignored"), matching
    real git's behaviour for an arbitrary path (rc 0 means the path IS ignored).
    """

    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        sub = cmd[1] if len(cmd) > 1 else ""
        default = (1, "", "") if sub == "check-ignore" else (0, "", "")
        rc, out, err = self.results.get(sub, default)
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
        # `git add -u` (tracked-only stage) erroring aborts before commit.
        r = FakeRunner({"add": (1, "", "permission denied")})
        ok, detail = auto_publish.commit_to_branch("b", "msg", r)
        self.assertFalse(ok)
        self.assertIn("stage", detail)
        self.assertFalse(r.ran("commit"))

    def test_commit_blocked_by_pii_guard(self):
        # the pre-commit guard rejects -> non-zero commit with a message. A
        # non-empty staged set is faked so staging passes and we reach commit.
        r = FakeRunner({"diff": (0, "core/x.py\n", ""),
                        "commit": (1, "BLOCKED: possible secret", "")})
        ok, detail = auto_publish.commit_to_branch("b", "msg", r)
        self.assertFalse(ok)
        self.assertIn("commit blocked/failed", detail)
        self.assertIn("BLOCKED", detail)

    def test_success(self):
        # diff --cached reports one benign staged file -> guard passes, commit runs.
        r = FakeRunner({"diff": (0, "core/x.py\n", ""),
                        "commit": (0, "1 file changed", "")})
        ok, detail = auto_publish.commit_to_branch("b", "msg", r)
        self.assertTrue(ok)
        self.assertTrue(r.ran("checkout") and r.ran("add") and r.ran("commit"))
        # never blanket-stages: no `git add -A` is ever issued
        self.assertFalse(any(c[:3] == ["git", "add", "-A"] for c in r.calls))
        # tracked stage uses `git add -u`
        self.assertTrue(any(c[:3] == ["git", "add", "-u"] for c in r.calls))


class ScopedStageTests(unittest.TestCase):
    """The P0-4 fix: staging is scoped + guarded so owner transcripts (logs/),
    runtime data, gitignored paths, and embedded-repo clones can NEVER be staged
    into a public PR. Every git call is injected — no real repo touched."""

    @staticmethod
    def _added_paths(runner):
        """Paths explicitly handed to `git add -- <path>` (excludes `add -u`)."""
        out = []
        for c in runner.calls:
            if c[:3] == ["git", "add", "--"]:
                out.append(c[3])
        return out

    def test_uses_add_u_not_add_all(self):
        r = FakeRunner({"diff": (0, "src.py\n", "")})
        ok, _ = auto_publish._scoped_stage(r)
        self.assertTrue(ok)
        self.assertTrue(any(c[:3] == ["git", "add", "-u"] for c in r.calls))
        self.assertFalse(any(c[:3] == ["git", "add", "-A"] for c in r.calls))

    def test_untracked_logs_transcripts_data_never_added(self):
        porcelain = (
            "?? logs/session_2026-06-06.log\n"
            "?? logs/owner_transcript.txt\n"
            "?? data/long_term_memory/facts.json\n"
            "?? memory/notes.txt\n"
            "?? conversation_transcript_2026.txt\n"
            "?? debug.log\n"
            "?? core/new_feature.py\n"      # the one legitimate new source file
        )
        r = FakeRunner({"status": (0, porcelain, ""),
                        "diff": (0, "core/new_feature.py\n", "")})
        ok, why = auto_publish._scoped_stage(r)
        self.assertTrue(ok, why)
        added = self._added_paths(r)
        self.assertEqual(added, ["core/new_feature.py"])
        for leak in ("logs/session_2026-06-06.log", "logs/owner_transcript.txt",
                     "data/long_term_memory/facts.json", "memory/notes.txt",
                     "conversation_transcript_2026.txt", "debug.log"):
            self.assertNotIn(leak, added)

    def test_embedded_repo_dir_is_skipped(self):
        # an embedded clone surfaces as an untracked DIR (trailing slash); adding
        # it would create a gitlink, so it must never be passed to `git add`.
        porcelain = "?? vendor_clone/\n?? logs/\n?? skills/real.py\n"
        r = FakeRunner({"status": (0, porcelain, ""),
                        "diff": (0, "skills/real.py\n", "")})
        ok, _ = auto_publish._scoped_stage(r)
        self.assertTrue(ok)
        self.assertEqual(self._added_paths(r), ["skills/real.py"])

    def test_guard_aborts_if_logfile_lands_in_index(self):
        # Defense in depth: even if a log somehow got staged, the post-stage
        # `git diff --cached` guard refuses to commit.
        r = FakeRunner({"diff": (0, "core/x.py\nlogs/session.log\n", "")})
        ok, why = auto_publish._scoped_stage(r)
        self.assertFalse(ok)
        self.assertIn("refusing to commit", why)
        self.assertIn("logs/session.log", why)

    def test_guard_aborts_if_gitignored_path_in_index(self):
        # A staged path git itself reports as ignored -> refuse (check-ignore rc 0).
        r = FakeRunner({"diff": (0, "core/x.py\nsecret.env\n", ""),
                        "check-ignore": (0, "secret.env\n", "")})
        ok, why = auto_publish._scoped_stage(r)
        self.assertFalse(ok)
        self.assertIn("refusing to commit", why)

    def test_guard_aborts_on_embedded_git_path(self):
        r = FakeRunner({"diff": (0, "vendor/.git/config\n", "")})
        ok, why = auto_publish._scoped_stage(r)
        self.assertFalse(ok)
        self.assertIn("refusing to commit", why)

    def test_empty_staged_set_is_rejected(self):
        r = FakeRunner({"diff": (0, "", "")})
        ok, why = auto_publish._scoped_stage(r)
        self.assertFalse(ok)
        self.assertIn("nothing to stage", why)

    def test_is_never_stage_classifier(self):
        for p in ("logs/x.log", "logs/owner_transcript.txt", "data/mem.json",
                  "memory/n.txt", "x.LOG", "a/b/transcript_2026.txt",
                  "vendor/.git/config", "backups/old.zip"):
            self.assertTrue(auto_publish._is_never_stage(p), p)
        for p in ("core/x.py", "tools/auto_publish.py", "README.md",
                  "audio/itunes_bridge.py", "skills/face_id.py"):
            self.assertFalse(auto_publish._is_never_stage(p), p)

    def test_is_never_stage_covers_owner_prose(self):
        # P0-4 follow-up: owner-authored prose/scratch that the dir/suffix/
        # substring classes miss (root-level .md/.txt/.ps1) must still be
        # backstopped so it can't reach a public PR even if a tree forgets to
        # gitignore it. Matched on the BASENAME (the `_mine_` prefix, or the
        # exact VOICE_COMMAND_TESTS.md) at any depth.
        for p in ("_mine_assistant.txt", "_mine_signal.ps1", "_mine_turns.ps1",
                  "VOICE_COMMAND_TESTS.md", "sub/dir/_mine_notes.txt",
                  "nested/VOICE_COMMAND_TESTS.md"):
            self.assertTrue(auto_publish._is_never_stage(p), p)
        # benign look-alikes — a "mine" substring (not the `_mine_` basename
        # prefix) or a differently-named doc — must NOT be excluded.
        for p in ("core/mine.py", "skills/determine_intent.py",
                  "tools/examine_logs.py", "docs/VOICE_COMMANDS.md"):
            self.assertFalse(auto_publish._is_never_stage(p), p)

    def test_owner_prose_untracked_never_added(self):
        # The exact gap P0-4 flagged: these untracked owner files sit at the
        # repo root, are NOT under a never-stage dir, do not end in .log, and
        # hold no "transcript" — yet must never be staged. check-ignore here
        # defaults to "not ignored" (FakeRunner), so this proves the
        # _NEVER_STAGE_* backstop excludes them independently of .gitignore.
        porcelain = (
            "?? _mine_assistant.txt\n"
            "?? _mine_signal.ps1\n"
            "?? _mine_turns.ps1\n"
            "?? VOICE_COMMAND_TESTS.md\n"
            "?? core/new_feature.py\n"      # the one legitimate new source file
        )
        r = FakeRunner({"status": (0, porcelain, ""),
                        "diff": (0, "core/new_feature.py\n", "")})
        ok, why = auto_publish._scoped_stage(r)
        self.assertTrue(ok, why)
        added = self._added_paths(r)
        self.assertEqual(added, ["core/new_feature.py"])
        for leak in ("_mine_assistant.txt", "_mine_signal.ps1",
                     "_mine_turns.ps1", "VOICE_COMMAND_TESTS.md"):
            self.assertNotIn(leak, added)


class ScopedStageRealGitTests(unittest.TestCase):
    """End-to-end against REAL git in a throwaway repo: proves the actual fix
    keeps logs/transcripts/embedded clones out of the index (not just the fake).
    Skipped if git is unavailable."""

    def setUp(self):
        import shutil
        import tempfile
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        self.tmp = tempfile.mkdtemp(prefix="auto_pub_git_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        def g(*args):
            subprocess.run(["git", *args], cwd=self.tmp, check=True,
                           capture_output=True, text=True)

        g("init", "-q")
        g("config", "user.email", "t@t.com")
        g("config", "user.name", "t")
        # .gitignore mirroring the repo's never-publish classes
        with open(os.path.join(self.tmp, ".gitignore"), "w") as f:
            f.write("logs/\n*.log\ndata/\n")
        with open(os.path.join(self.tmp, "main.py"), "w") as f:
            f.write("print('hi')\n")
        g("add", ".gitignore", "main.py")
        g("commit", "-qm", "init")
        # now create exactly what the overnight run might leave behind:
        os.makedirs(os.path.join(self.tmp, "logs"))
        with open(os.path.join(self.tmp, "logs", "owner_transcript.txt"), "w") as f:
            f.write("PRIVATE owner conversation\n")
        with open(os.path.join(self.tmp, "session.log"), "w") as f:
            f.write("log line\n")
        os.makedirs(os.path.join(self.tmp, "data"))
        with open(os.path.join(self.tmp, "data", "memory.json"), "w") as f:
            f.write('{"pii": "x"}\n')
        # an embedded clone (nested git repo) with a commit -> would become a gitlink
        emb = os.path.join(self.tmp, "embedded_clone")
        os.makedirs(emb)
        subprocess.run(["git", "init", "-q"], cwd=emb, check=True,
                       capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=emb,
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=emb, check=True,
                       capture_output=True, text=True)
        with open(os.path.join(emb, "inner.py"), "w") as f:
            f.write("x = 1\n")
        subprocess.run(["git", "add", "inner.py"], cwd=emb, check=True,
                       capture_output=True, text=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=emb, check=True,
                       capture_output=True, text=True)
        # a LEGITIMATE new source file that SHOULD be published
        with open(os.path.join(self.tmp, "feature.py"), "w") as f:
            f.write("VALUE = 42\n")
        # and a tracked-file edit that SHOULD be published
        with open(os.path.join(self.tmp, "main.py"), "w") as f:
            f.write("print('updated')\n")

    def _runner(self):
        root = self.tmp

        def runner(cmd, **kw):
            kw.pop("cwd", None)  # force our temp repo regardless of module _ROOT
            return subprocess.run(cmd, cwd=root, **kw)
        return runner

    def _staged(self):
        r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                           cwd=self.tmp, capture_output=True, text=True)
        return set(l.strip() for l in r.stdout.splitlines() if l.strip())

    def test_scoped_stage_excludes_logs_data_embedded_keeps_source(self):
        ok, why = auto_publish._scoped_stage(self._runner())
        self.assertTrue(ok, why)
        staged = self._staged()
        # the intended source changes ARE staged
        self.assertIn("feature.py", staged)
        self.assertIn("main.py", staged)
        # the never-publish classes are NOT staged
        self.assertNotIn("logs/owner_transcript.txt", staged)
        self.assertNotIn("session.log", staged)
        self.assertNotIn("data/memory.json", staged)
        self.assertNotIn("embedded_clone", staged)
        # no path under logs/ or data/, nothing ending in .log, no transcript
        for s in staged:
            self.assertFalse(s.startswith("logs/"), s)
            self.assertFalse(s.startswith("data/"), s)
            self.assertFalse(s.endswith(".log"), s)
            self.assertNotIn("transcript", s.lower())


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
                        "diff": (0, "x.py\n", ""),
                        "commit": (1, "BLOCKED", "")})
        code, text = self._run([], r)
        self.assertEqual(code, 1)
        self.assertIn("Aborted", text)
        self.assertFalse(r.ran("push"))

    def test_push_failure_returns_1(self):
        r = FakeRunner({"status": (0, " M x.py\n", ""),
                        "diff": (0, "x.py\n", ""),
                        "push": (1, "", "rejected")})
        code, text = self._run([], r)
        self.assertEqual(code, 1)
        self.assertIn("Push failed", text)

    def test_success_opens_pr(self):
        r = FakeRunner({"status": (0, " M x.py\n", ""),
                        "diff": (0, "x.py\n", "")})
        seen = {}

        def poster(branch, title, body):
            seen["branch"] = branch
            return "https://github.com/x/y/pull/1"

        code, text = self._run(["--summary", "did things"], r, poster=poster)
        self.assertEqual(code, 0)
        self.assertIn("Pull request opened", text)
        self.assertEqual(seen["branch"], "auto/overnight-t")

    def test_success_pr_open_failure_is_graceful(self):
        r = FakeRunner({"status": (0, " M x.py\n", ""),
                        "diff": (0, "x.py\n", "")})
        code, text = self._run([], r, poster=lambda *a: None)
        self.assertEqual(code, 0)
        self.assertIn("Couldn't open the PR", text)

    def test_no_pr_flag_skips_pr(self):
        r = FakeRunner({"status": (0, " M x.py\n", ""),
                        "diff": (0, "x.py\n", "")})
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
