#!/usr/bin/env python3
"""Unit tests for core/bug_reporter.py — capture, scrub, format, outbox.

The scrubber tests are the important ones: they prove personal data and secret
shapes never survive into a report (and thus never leave a user's machine).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bug_reporter  # noqa: E402


class ScrubTests(unittest.TestCase):
    def test_empty_and_none(self):
        self.assertEqual(bug_reporter.scrub(""), "")
        self.assertEqual(bug_reporter.scrub(None), "")

    def test_plain_text_passes_through(self):
        self.assertEqual(bug_reporter.scrub("the timer never fired"),
                         "the timer never fired")

    def test_email_redacted(self):
        out = bug_reporter.scrub("contact me at jane.doe@example.com please")
        self.assertNotIn("jane.doe@example.com", out)
        self.assertIn("<EMAIL>", out)

    def test_windows_user_path_redacted(self):
        out = bug_reporter.scrub(r"opened C:\Users\someuser\secret\notes.txt ok")
        self.assertNotIn("someuser", out)
        self.assertIn(r"C:\Users\<USER>", out)

    def test_unix_home_paths_redacted(self):
        self.assertNotIn("alice", bug_reporter.scrub("/home/alice/data"))
        self.assertNotIn("bob", bug_reporter.scrub("/Users/bob/data"))

    def test_token_shapes_redacted(self):
        for tok in ("ghp_" + "a" * 30, "github_pat_" + "b" * 25,
                    "sk-ant-" + "c" * 20, "gho_" + "d" * 25,
                    "AKIA" + "ABCDEFGHIJKLMNOP", "xoxb-12345678abcd"):
            out = bug_reporter.scrub(f"key is {tok} here")
            self.assertNotIn(tok, out, tok)
            self.assertIn("<KEY>", out)

    def test_bearer_redacted(self):
        out = bug_reporter.scrub("Authorization: Bearer abcdef1234567890zz")
        self.assertIn("Bearer <KEY>", out)
        self.assertNotIn("abcdef1234567890zz", out)

    def test_env_style_secret_value_redacted(self):
        out = bug_reporter.scrub("API_KEY = supersecretvalue123")
        self.assertNotIn("supersecretvalue123", out)
        self.assertIn("<REDACTED>", out)
        out2 = bug_reporter.scrub("password: hunter2hunter2")
        self.assertNotIn("hunter2hunter2", out2)

    def test_ipv4_redacted(self):
        out = bug_reporter.scrub("connected to 203.0.113.42 then 192.168.1.5")
        self.assertNotIn("203.0.113.42", out)
        self.assertNotIn("192.168.1.5", out)
        self.assertEqual(out.count("<IP>"), 2)

    def test_long_hex_redacted(self):
        out = bug_reporter.scrub("digest " + "a" * 40)
        self.assertIn("<HEX>", out)
        self.assertNotIn("a" * 40, out)


class MakeReportTests(unittest.TestCase):
    def test_kind_normalises(self):
        self.assertEqual(bug_reporter.make_report("auto", "x")["kind"], "auto")
        self.assertEqual(bug_reporter.make_report("user", "x")["kind"], "user")
        self.assertEqual(bug_reporter.make_report("garbage", "x")["kind"], "user")

    def test_fields_scrubbed(self):
        rep = bug_reporter.make_report(
            "user", "fail at foo@bar.com",
            detail=r"path C:\Users\someuser\x",
            context={"host": "10.0.0.9", "note": "ghp_" + "z" * 30})
        self.assertIn("<EMAIL>", rep["summary"])
        self.assertIn("<USER>", rep["detail"])
        self.assertEqual(rep["context"]["host"], "<IP>")
        self.assertNotIn("someuser", json.dumps(rep))
        self.assertNotIn("bar.com", json.dumps(rep))
        self.assertIn("<KEY>", rep["context"]["note"])

    def test_explicit_version_and_ts(self):
        rep = bug_reporter.make_report("user", "x", version="9.9.9", ts=123.0)
        self.assertEqual(rep["version"], "9.9.9")
        self.assertEqual(rep["ts"], 123.0)

    def test_default_version_and_ts(self):
        rep = bug_reporter.make_report("user", "x")
        self.assertIsInstance(rep["version"], str)
        self.assertGreater(rep["ts"], 0)

    def test_truncation(self):
        rep = bug_reporter.make_report("user", "s" * 999, detail="d" * 9000,
                                       tb="t" * 9000)
        self.assertLessEqual(len(rep["summary"]), 300)
        self.assertLessEqual(len(rep["detail"]), 4000)
        self.assertLessEqual(len(rep["traceback"]), 6000)


class CaptureExceptionTests(unittest.TestCase):
    def test_builds_auto_report_with_traceback(self):
        try:
            raise ValueError("boom at /home/alice/x")
        except ValueError as e:
            rep = bug_reporter.capture_exception(e, where="timer", context={"a": 1})
        self.assertEqual(rep["kind"], "auto")
        self.assertIn("[timer]", rep["summary"])
        self.assertIn("ValueError", rep["summary"])
        self.assertIn("Traceback", rep["traceback"])
        self.assertNotIn("alice", json.dumps(rep))

    def test_no_where_prefix(self):
        try:
            raise RuntimeError("x")
        except RuntimeError as e:
            rep = bug_reporter.capture_exception(e)
        self.assertTrue(rep["summary"].startswith("RuntimeError"))


class OutboxTests(unittest.TestCase):
    def test_append_and_record(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "sub", "bugs.jsonl")
        rep = bug_reporter.record_bug("user", "thing broke", outbox=path)
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as fh:
            line = json.loads(fh.readline())
        self.assertEqual(line["summary"], "thing broke")
        self.assertEqual(rep["summary"], "thing broke")
        # second record appends, doesn't overwrite
        bug_reporter.record_bug("auto", "second", outbox=path)
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(len(fh.readlines()), 2)

    def test_append_oserror_returns_false(self):
        with mock.patch.object(bug_reporter.os, "makedirs",
                               side_effect=OSError("readonly")):
            self.assertFalse(bug_reporter.append_outbox({"x": 1}, "/nope/a.jsonl"))


class FormatAndUrlTests(unittest.TestCase):
    def test_auto_issue_has_traceback_section(self):
        rep = bug_reporter.make_report("auto", "Crash", tb="Traceback line")
        title, body = bug_reporter.format_issue(rep)
        self.assertTrue(title.startswith("[auto-detected]"))
        self.assertIn("**Traceback**", body)
        self.assertIn("```", body)

    def test_user_issue_has_detail_and_context(self):
        rep = bug_reporter.make_report("user", "Bad", detail="it broke",
                                       context={"skill": "timer"})
        title, body = bug_reporter.format_issue(rep)
        self.assertTrue(title.startswith("[user-reported]"))
        self.assertIn("**Details**", body)
        self.assertIn("**Context**", body)
        self.assertIn("- skill: timer", body)

    def test_minimal_issue_still_valid(self):
        rep = bug_reporter.make_report("user", "Tiny")
        title, body = bug_reporter.format_issue(rep)
        self.assertIn("Tiny", title)
        self.assertIn("**Source:** user-reported", body)
        self.assertIn("scrubbed", body)

    def test_browser_url_structure(self):
        rep = bug_reporter.make_report("user", "Hello world")
        url = bug_reporter.browser_submit_url(rep, owner="o", repo="r")
        self.assertTrue(url.startswith("https://github.com/o/r/issues/new?"))
        self.assertIn("title=", url)
        self.assertIn("body=", url)
        self.assertIn("labels=bug", url)
        self.assertIn("Hello+world", url)


class AutoCaptureTests(unittest.TestCase):
    """The rate-limited self-detect path."""

    def setUp(self):
        bug_reporter._recent_auto.clear()
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "bugs.jsonl")

    def _exc(self, msg="boom"):
        try:
            raise ValueError(msg)
        except ValueError as e:
            return e

    def test_first_capture_records(self):
        rep = bug_reporter.auto_capture(self._exc(), where="dispatch",
                                        now=100.0, outbox=self.path)
        self.assertIsNotNone(rep)
        self.assertEqual(rep["kind"], "auto")
        self.assertTrue(os.path.exists(self.path))

    def test_duplicate_within_window_suppressed(self):
        bug_reporter.auto_capture(self._exc(), where="dispatch", now=100.0,
                                  outbox=self.path)
        dup = bug_reporter.auto_capture(self._exc(), where="dispatch", now=200.0,
                                        outbox=self.path)
        self.assertIsNone(dup)

    def test_after_window_records_again(self):
        bug_reporter.auto_capture(self._exc(), where="dispatch", now=100.0,
                                  outbox=self.path)
        again = bug_reporter.auto_capture(self._exc(), where="dispatch",
                                          now=500.0, outbox=self.path)
        self.assertIsNotNone(again)

    def test_different_where_not_suppressed(self):
        a = bug_reporter.auto_capture(self._exc(), where="a", now=100.0,
                                      outbox=self.path)
        b = bug_reporter.auto_capture(self._exc(), where="b", now=100.0,
                                      outbox=self.path)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)

    def test_report_is_scrubbed(self):
        rep = bug_reporter.auto_capture(self._exc("fail at a@b.com"),
                                        where="x", now=100.0, outbox=self.path)
        self.assertIn("<EMAIL>", rep["summary"])

    def test_default_now_uses_clock(self):
        rep = bug_reporter.auto_capture(self._exc(), where="clock",
                                        outbox=self.path)
        self.assertIsNotNone(rep)


class ApiSubmitTests(unittest.TestCase):
    def test_auto_submit_flag(self):
        with mock.patch.dict(os.environ, {"JARVIS_BUG_AUTO_SUBMIT": "1"}):
            self.assertTrue(bug_reporter.auto_submit_enabled())
        with mock.patch.dict(os.environ, {"JARVIS_BUG_AUTO_SUBMIT": "0"}):
            self.assertFalse(bug_reporter.auto_submit_enabled())

    def test_issue_token_lookup(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("JARVIS_GITHUB_TOKEN", "GITHUB_TOKEN")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(bug_reporter._issue_token())
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "g"}):
            self.assertEqual(bug_reporter._issue_token(), "g")

    def test_api_submit_no_token_returns_none(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("JARVIS_GITHUB_TOKEN", "GITHUB_TOKEN")}
        with mock.patch.dict(os.environ, env, clear=True):
            rep = bug_reporter.make_report("user", "x")
            self.assertIsNone(bug_reporter.api_submit_issue(rep))

    def test_api_submit_posts_and_returns_url(self):
        rep = bug_reporter.make_report("auto", "Crash", tb="TB")
        seen = {}

        def opener(url, payload, token):
            seen.update(url=url, payload=payload, token=token)
            return "https://github.com/o/r/issues/9"

        out = bug_reporter.api_submit_issue(rep, owner="o", repo="r",
                                            token="tok", opener=opener)
        self.assertEqual(out, "https://github.com/o/r/issues/9")
        self.assertEqual(seen["url"], "https://api.github.com/repos/o/r/issues")
        self.assertEqual(seen["token"], "tok")
        self.assertIn(b"Crash", seen["payload"])

    def test_api_submit_opener_none_result(self):
        rep = bug_reporter.make_report("user", "x")
        self.assertIsNone(bug_reporter.api_submit_issue(
            rep, token="t", opener=lambda *a: None))

    def test_api_submit_opener_raises_is_swallowed(self):
        rep = bug_reporter.make_report("user", "x")

        def boom(*a):
            raise RuntimeError("net down")

        self.assertIsNone(bug_reporter.api_submit_issue(rep, token="t",
                                                        opener=boom))


class ReportBugActionTests(unittest.TestCase):
    """The core.actions surface (_act_report_bug) — light tier, no monolith boot."""

    def setUp(self):
        from core import actions
        self.A = actions

    @mock.patch("webbrowser.open", return_value=True)
    @mock.patch("core.bug_reporter.append_outbox", return_value=True)
    def test_logs_and_opens_issue(self, m_append, m_web):
        out = self.A._act_report_bug("the timer never fired")
        self.assertIn("Logged it", out)
        m_append.assert_called_once()
        m_web.assert_called_once()
        rep = m_append.call_args[0][0]
        self.assertEqual(rep["kind"], "user")
        self.assertEqual(rep["summary"], "the timer never fired")

    def test_empty_description_prompts(self):
        out = self.A._act_report_bug("   ")
        self.assertIn("Tell me what went wrong", out)

    @mock.patch("webbrowser.open", side_effect=Exception("no browser"))
    @mock.patch("core.bug_reporter.append_outbox", return_value=True)
    def test_browser_failure_still_logs(self, m_append, m_web):
        out = self.A._act_report_bug("something broke")
        self.assertIn("Logged it locally", out)
        m_append.assert_called_once()

    @mock.patch("core.bug_reporter.api_submit_issue",
                return_value="https://github.com/x/y/issues/3")
    @mock.patch("core.bug_reporter.append_outbox", return_value=True)
    def test_api_submit_when_enabled(self, m_append, m_api):
        with mock.patch.dict(os.environ, {"JARVIS_BUG_AUTO_SUBMIT": "1"}):
            out = self.A._act_report_bug("broke")
        self.assertIn("filed a GitHub issue", out)
        m_api.assert_called_once()

    @mock.patch("core.bug_reporter.api_submit_issue", return_value=None)
    @mock.patch("core.bug_reporter.append_outbox", return_value=True)
    def test_api_submit_failure_when_enabled(self, m_append, m_api):
        with mock.patch.dict(os.environ, {"JARVIS_BUG_AUTO_SUBMIT": "1"}):
            out = self.A._act_report_bug("broke")
        self.assertIn("didn't go through", out)


if __name__ == "__main__":
    unittest.main()
