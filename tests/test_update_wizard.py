"""Tests for tools/update_wizard.py — the check -> confirm -> pull -> verify flow.

CI-safe: stdlib only. No real git/tests run — the subprocess runner, the update
checker, stdin, and stdout are all injected/mocked.
"""
from __future__ import annotations

import unittest
from unittest import mock

import tools.update_wizard as uw


def _dispatch(mapping, default=(0, "", "")):
    """Fake runner: pick a (rc, stdout, stderr) by matching a substring of the
    joined command; falls back to `default`."""
    def run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        for sub, val in mapping.items():
            if sub in joined:
                rc, so, se = val
                return mock.Mock(returncode=rc, stdout=so, stderr=se)
        rc, so, se = default
        return mock.Mock(returncode=rc, stdout=so, stderr=se)
    return run


def _raises(exc):
    def run(cmd, **kw):
        raise exc
    return run


class CurrentCommitTests(unittest.TestCase):
    def test_success(self):
        self.assertEqual(uw.current_commit(_dispatch({"rev-parse": (0, "abc123\n", "")})),
                         "abc123")

    def test_nonzero_returns_question(self):
        self.assertEqual(uw.current_commit(_dispatch({"rev-parse": (1, "", "")})), "?")

    def test_exception_returns_question(self):
        self.assertEqual(uw.current_commit(_raises(OSError("no git"))), "?")


class TrackedTreeDirtyTests(unittest.TestCase):
    def test_dirty(self):
        self.assertTrue(uw.tracked_tree_dirty(_dispatch({"status": (0, " M f.py\n", "")})))

    def test_clean(self):
        self.assertFalse(uw.tracked_tree_dirty(_dispatch({"status": (0, "", "")})))

    def test_nonzero_is_not_dirty(self):
        self.assertFalse(uw.tracked_tree_dirty(_dispatch({"status": (1, "", "")})))

    def test_exception_is_not_dirty(self):
        self.assertFalse(uw.tracked_tree_dirty(_raises(OSError("boom"))))


class ApplyUpdateTests(unittest.TestCase):
    def test_success(self):
        ok, detail = uw.apply_update(_dispatch({"fetch": (0, "", ""),
                                                "merge": (0, "Updating a..b\n", "")}))
        self.assertTrue(ok)
        self.assertIn("Updating", detail)

    def test_fetch_failure(self):
        ok, detail = uw.apply_update(_dispatch({"fetch": (1, "", "network down")}))
        self.assertFalse(ok)
        self.assertIn("fetch failed", detail)

    def test_ff_failure(self):
        ok, detail = uw.apply_update(_dispatch({"fetch": (0, "", ""),
                                                "merge": (1, "", "Not possible to fast-forward")}))
        self.assertFalse(ok)
        self.assertIn("diverged", detail)

    def test_exception(self):
        ok, detail = uw.apply_update(_raises(OSError("kaboom")))
        self.assertFalse(ok)
        self.assertIn("update error", detail)


class VerifyTests(unittest.TestCase):
    def test_pass(self):
        ok, detail = uw.verify(_dispatch({"run_tests": (0, "OK", "")}))
        self.assertTrue(ok)

    def test_fail_with_tail(self):
        ok, detail = uw.verify(_dispatch({"run_tests": (1, "a\nb\nc\nd", "")}))
        self.assertFalse(ok)
        self.assertIn("FAILED", detail)
        self.assertIn("d", detail)        # last lines included

    def test_fail_empty_body(self):
        ok, detail = uw.verify(_dispatch({"run_tests": (1, "", "")}))
        self.assertFalse(ok)
        self.assertIn("see output", detail)

    def test_exception(self):
        ok, detail = uw.verify(_raises(OSError("nope")))
        self.assertFalse(ok)
        self.assertIn("verify error", detail)


_UPD = {"checked": True, "update_available": True, "current": "1.2.1",
        "latest": "v1.3.0", "release_url": "https://gh/r/v1.3.0"}


class MainTests(unittest.TestCase):
    def _main(self, result, argv=None, *, dirty=False, apply=(True, "ff"),
              ver=(True, "ok"), ans="y"):
        out = []
        with mock.patch("core.update_checker.check_for_update", return_value=result), \
             mock.patch.object(uw, "tracked_tree_dirty", return_value=dirty), \
             mock.patch.object(uw, "current_commit", return_value="abc123"), \
             mock.patch.object(uw, "apply_update", return_value=apply), \
             mock.patch.object(uw, "verify", return_value=ver) as v:
            code = uw.main(argv or [], runner=lambda *a, **k: None,
                           input_fn=lambda _p: ans, out=out.append)
        return code, "\n".join(out), v

    def test_not_checked(self):
        code, out, _ = self._main({"checked": False, "detail": "no GitHub token"})
        self.assertEqual(code, 2)
        self.assertIn("Couldn't check", out)
        self.assertIn("JARVIS_GITHUB_TOKEN", out)

    def test_up_to_date(self):
        code, out, _ = self._main({"checked": True, "update_available": False,
                                   "current": "1.2.1"})
        self.assertEqual(code, 0)
        self.assertIn("up to date", out)

    def test_check_only(self):
        code, out, _ = self._main(_UPD, argv=["--check"])
        self.assertEqual(code, 0)
        self.assertIn("Update available", out)
        self.assertIn("Release notes", out)

    def test_no_release_url_skips_notes(self):
        r = dict(_UPD); r["release_url"] = None
        code, out, _ = self._main(r, argv=["--check"])
        self.assertNotIn("Release notes", out)

    def test_confirm_declined(self):
        code, out, _ = self._main(_UPD, ans="n")
        self.assertEqual(code, 0)
        self.assertIn("Skipped", out)

    def test_dirty_tree_aborts(self):
        code, out, _ = self._main(_UPD, argv=["--yes"], dirty=True)
        self.assertEqual(code, 1)
        self.assertIn("Aborting", out)

    def test_apply_failure(self):
        code, out, _ = self._main(_UPD, argv=["--yes"], apply=(False, "diverged"))
        self.assertEqual(code, 1)
        self.assertIn("Update failed", out)

    def test_verify_failure_shows_rollback(self):
        code, out, _ = self._main(_UPD, argv=["--yes"], ver=(False, "tests FAILED"))
        self.assertEqual(code, 1)
        self.assertIn("Roll back with:  git reset --hard abc123", out)

    def test_success_full(self):
        code, out, v = self._main(_UPD, argv=["--yes"])
        self.assertEqual(code, 0)
        self.assertIn("Done. Restart JARVIS", out)
        v.assert_called_once()        # verify ran

    def test_skip_verify(self):
        code, out, v = self._main(_UPD, argv=["--yes", "--skip-verify"])
        self.assertEqual(code, 0)
        self.assertIn("Done.", out)
        v.assert_not_called()          # verify skipped

    def test_interactive_yes(self):
        code, out, _ = self._main(_UPD, ans="yes")
        self.assertEqual(code, 0)
        self.assertIn("Done.", out)


if __name__ == "__main__":
    unittest.main()
