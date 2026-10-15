"""core/claude_accounts.py + tools/claude_account_run.py: JARVIS's Claude Code
jobs run on the owner's second (Pro) profile first through its last day, on
Max whenever Pro is at its 5-hour / weekly limit (back to Pro when it resets),
and never on Pro afterwards. The sign-in file is checked for existence only;
no token is ever read, logged or printed. Everything runs on temp folders and
fake runners - no real `claude` call, no live data/ write."""
import builtins
import datetime as dt
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import claude_accounts as ca  # noqa: E402
from core import config  # noqa: E402

TODAY = dt.date(2026, 10, 14)
# A token look-alike built at RUNTIME so the literal never sits in the source
# (the repo's PII gate greps tests/); it must never appear in any output.
FAKE_TOKEN = "-".join(("sk", "ant", "oat01", "Z" * 24))


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pro = os.path.join(self.tmp.name, "claude-pro")
        os.makedirs(self.pro)
        self.state = os.path.join(self.tmp.name, "claude_account_state.json")
        p = mock.patch.object(ca, "STATE_PATH", self.state)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.multiple(config, CLAUDE_PRO_FIRST_UNTIL="2026-11-12",
                                CLAUDE_PRO_CONFIG_DIR=self.pro)
        q.start()
        self.addCleanup(q.stop)

    def sign_in(self):
        with open(os.path.join(self.pro, ca.SIGN_IN_FILE), "w") as f:
            f.write("{}")


class ChooseTests(_Base):
    def test_pro_first_while_signed_in_and_in_date(self):
        self.sign_in()
        self.assertEqual(ca.choose(today=TODAY)[0], ca.PRO)
        self.assertEqual(ca.choose(today=dt.date(2026, 11, 12))[0], ca.PRO)  # last day counts

    def test_never_pro_after_the_last_day(self):
        self.sign_in()
        profile, why = ca.choose(today=dt.date(2026, 11, 13))
        self.assertEqual(profile, ca.MAX)
        self.assertIn("2026-11-12", why)

    def test_off_by_default_and_on_a_bad_date(self):
        self.sign_in()
        for value in ("", "next month", None):
            with mock.patch.object(config, "CLAUDE_PRO_FIRST_UNTIL", value):
                self.assertEqual(ca.choose(today=TODAY)[0], ca.MAX, value)

    def test_not_signed_in_means_max(self):
        profile, why = ca.choose(today=TODAY)
        self.assertEqual(profile, ca.MAX)
        self.assertIn("isn't signed in", why)

    def test_the_sign_in_file_is_never_opened(self):
        self.sign_in()
        real_open = builtins.open

        def guarded(path, *a, **k):
            if str(path).endswith(ca.SIGN_IN_FILE):
                raise AssertionError("the sign-in file was opened")
            return real_open(path, *a, **k)
        with mock.patch("builtins.open", guarded):
            self.assertEqual(ca.choose(today=TODAY)[0], ca.PRO)

    def test_default_folder_is_in_the_home_folder(self):
        with mock.patch.object(config, "CLAUDE_PRO_CONFIG_DIR", ""):
            self.assertEqual(ca.pro_dir(),
                             os.path.join(os.path.expanduser("~"), ".claude-pro"))


class EnvTests(_Base):
    BASE = {"PATH": "x", "ANTHROPIC_API_KEY": FAKE_TOKEN,
            "CLAUDE_CODE_OAUTH_TOKEN": FAKE_TOKEN, "ANTHROPIC_AUTH_TOKEN": FAKE_TOKEN,
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:1", "CLAUDECODE": "1",
            "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH": "1",
            "CLAUDE_CONFIG_DIR": "C:/elsewhere"}

    def test_pro_gets_its_folder_and_no_auth_overrides(self):
        env = ca.env_for(ca.PRO, self.BASE)
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], self.pro)
        for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN",
                  "ANTHROPIC_BASE_URL", "CLAUDECODE", "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH"):
            self.assertNotIn(k, env, k)
        self.assertEqual(env["PATH"], "x")

    def test_max_uses_the_default_profile(self):
        env = ca.env_for(ca.MAX, self.BASE)
        self.assertNotIn("CLAUDE_CONFIG_DIR", env)
        self.assertNotIn("ANTHROPIC_API_KEY", env)

    def test_the_base_is_not_modified(self):
        base = dict(self.BASE)
        ca.env_for(ca.PRO, base)
        self.assertEqual(base, self.BASE)


class LimitParsingTests(unittest.TestCase):
    def ev(self, status, kind="five_hour", resets=1792050000):
        return {"type": "rate_limit_event",
                "rate_limit_info": {"status": status, "rateLimitType": kind,
                                    "resetsAt": resets}}

    def test_rejected_event_gives_kind_and_reset(self):
        lim = ca.limit_from_event(self.ev("rejected"))
        self.assertEqual((lim.kind, lim.resets_at), ("five_hour", 1792050000.0))
        self.assertEqual(ca.limit_from_event(self.ev("rejected", "seven_day",
                                                     1792050000000)).resets_at,
                         1792050000.0)                       # milliseconds too

    def test_allowed_and_warning_events_are_not_limits(self):
        for status in ("allowed", "allowed_warning"):
            self.assertIsNone(ca.limit_from_event(self.ev(status)))
        self.assertIsNone(ca.limit_from_event({"type": "assistant"}))
        self.assertIsNone(ca.limit_from_event("nope"))

    def test_the_cli_wording(self):
        cases = {
            "You've hit your session limit \u00b7 resets 9pm": "five_hour",
            "You\u2019ve hit your weekly limit \u00b7 resets Oct 20, 9am": "seven_day",
            "You've hit your Opus limit": "seven_day_opus",
            "You've hit your usage limit \u00b7 resets 3pm": "usage",
        }
        for text, kind in cases.items():
            with self.subTest(text=text):
                lim = ca.limit_from_text(text)
                self.assertEqual(lim.kind, kind)
                self.assertIsNone(lim.resets_at)

    def test_old_wording_carries_the_reset(self):
        lim = ca.limit_from_text("Claude AI usage limit reached|1792050000")
        self.assertEqual(lim.resets_at, 1792050000.0)

    def test_a_long_reply_that_mentions_the_words_is_not_a_limit(self):
        reply = ("Implemented the rate-limit banner. When the CLI says \"You've hit "
                 "your session limit\" the HUD now shows it. ") * 10
        self.assertIsNone(ca.limit_from_text(reply))

    def test_a_limit_event_on_a_job_that_finished_is_not_a_refusal(self):
        ok = "\n".join(json.dumps(x) for x in (
            self.ev("rejected", "seven_day_opus"),
            {"type": "result", "is_error": False, "result": "done"}))
        self.assertIsNone(ca.scan_output(ok))
        failed = "\n".join(json.dumps(x) for x in (
            self.ev("rejected"),
            {"type": "result", "is_error": True,
             "result": "You've hit your session limit \u00b7 resets 9pm"}))
        self.assertEqual(ca.scan_output(failed).kind, "five_hour")

    def test_signed_out_wording(self):
        self.assertTrue(ca.signed_out_text("Not logged in. Run claude auth login to authenticate."))
        self.assertTrue(ca.signed_out_text("Invalid API key \u00b7 Please run /login"))
        self.assertFalse(ca.signed_out_text("All tests pass."))


class RefusalStateTests(_Base):
    def test_a_known_reset_parks_pro_until_then_and_it_comes_back(self):
        self.sign_in()
        now = time.time()
        until, msg = ca.note_pro_refusal(ca.Limit("five_hour", now + 3600), now=now)
        self.assertAlmostEqual(until, now + 3600 + ca.RESET_MARGIN_S, delta=1)
        self.assertIn("5-hour limit", msg)
        self.assertEqual(ca.choose(now=now + 60, today=TODAY)[0], ca.MAX)
        self.assertEqual(ca.choose(now=until + 1, today=TODAY)[0], ca.PRO)   # reset: back to Pro

    def test_text_limits_get_a_re_probe_time(self):
        now = 1_000_000.0
        until, _ = ca.note_pro_refusal(ca.Limit("seven_day", None), now=now)
        self.assertEqual(until, now + ca.PROBE_S["seven_day"])
        until, msg = ca.note_pro_refusal(signed_out=True, now=now)
        self.assertEqual(until, now + ca.PROBE_S["signed_out"])
        self.assertIn("isn't signed in", msg)

    def test_state_holds_no_secrets(self):
        ca.note_pro_refusal(ca.Limit("five_hour", None))
        with open(self.state, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(set(data), {"pro_blocked_until", "pro_blocked_kind", "pro_blocked_at"})


class RunTests(_Base):
    def runner(self, outputs):
        calls = []

        def _run(cmd, env=None, **kw):
            calls.append(env)
            out = outputs[min(len(calls) - 1, len(outputs) - 1)]
            return SimpleNamespace(returncode=out[0], stdout=out[1], stderr="")
        return _run, calls

    def test_pro_limit_reruns_the_same_job_on_max(self):
        self.sign_in()
        run, calls = self.runner([(0, "You've hit your session limit \u00b7 resets 9pm"),
                                  (0, "real answer")])
        logs = []
        res = ca.run(["claude", "--print", "x"], env={"ANTHROPIC_API_KEY": FAKE_TOKEN},
                     runner=run, log=logs.append)
        self.assertEqual(res.stdout, "real answer")
        self.assertEqual(res.claude_profile, ca.MAX)
        self.assertEqual(calls[0]["CLAUDE_CONFIG_DIR"], self.pro)
        self.assertNotIn("CLAUDE_CONFIG_DIR", calls[1])
        self.assertTrue(all("ANTHROPIC_API_KEY" not in c for c in calls))
        self.assertEqual(len(logs), 1)
        self.assertIn("5-hour limit", logs[0])
        self.assertNotIn(FAKE_TOKEN, logs[0])
        self.assertEqual(ca.choose(today=TODAY)[0], ca.MAX)             # parked

    def test_pro_success_is_one_call(self):
        self.sign_in()
        run, calls = self.runner([(0, "done")])
        res = ca.run(["claude"], env={}, runner=run)
        self.assertEqual((len(calls), res.claude_profile), (1, ca.PRO))

    def test_signed_out_pro_falls_back_too(self):
        self.sign_in()
        run, calls = self.runner([(1, "Not logged in. Run claude auth login to authenticate."),
                                  (0, "ok")])
        res = ca.run(["claude"], env={}, runner=run)
        self.assertEqual((len(calls), res.claude_profile), (2, ca.MAX))

    def test_max_jobs_are_never_retried(self):
        run, calls = self.runner([(0, "You've hit your session limit")])
        res = ca.run(["claude"], env={}, runner=run)                  # not signed in -> Max
        self.assertEqual((len(calls), res.claude_profile), (1, ca.MAX))


class StatusToolTests(_Base):
    def setUp(self):
        super().setUp()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "claude_account_run_under_test",
            os.path.join(_ROOT, "tools", "claude_account_run.py"))
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)

    def test_check_prints_no_token(self):
        fake = json.dumps({"loggedIn": True, "authMethod": "claude.ai",
                           "email": "owner@example.com", "subscriptionType": "pro",
                           "accessToken": FAKE_TOKEN, "refreshToken": FAKE_TOKEN})
        with mock.patch.object(self.tool.subprocess, "run",
                               return_value=SimpleNamespace(stdout=fake, returncode=0)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.tool.check()
        out = buf.getvalue()
        self.assertNotIn(FAKE_TOKEN, out)
        self.assertIn("owner@example.com", out)
        self.assertIn("not shown", out)

    def test_status_names_the_choice(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.tool.status()
        self.assertIn("Max", buf.getvalue())
        self.assertIn("2026-11-12", buf.getvalue())


class SpawnSiteTests(unittest.TestCase):
    """Every Claude Code spawn goes through the profile choice."""

    def read(self, rel):
        with open(os.path.join(_ROOT, rel), encoding="utf-8") as f:
            return f.read()

    def test_the_retired_loop_drivers_are_gone(self):
        # The old driver templates spawned `claude` with the blanket permission
        # bypass against the live tree; the safe runner replaced them.
        src = self.read("upgrade_jarvis.py")
        self.assertNotIn("_LOOP_DRIVER_TEMPLATE", src)
        self.assertNotIn("def spawn_claude_code", src)

    def test_pipeline_overnight_and_dry_audit_use_it(self):
        self.assertIn("acct.run(cmd, env=env, runner=subprocess.run",
                      self.read("tools/multi_agent_pipeline.py"))
        self.assertIn("_acct.run(_cli_cmd, env=_env, runner=subprocess.run",
                      self.read("overnight_upgrade.py"))
        self.assertIn("claude_account_run.py", self.read("upgrade_jarvis.py"))

    def test_pipeline_recognises_the_current_limit_wording(self):
        from tools import multi_agent_pipeline as P
        self.assertTrue(P._looks_rate_limited(
            "You've hit your session limit \u00b7 resets 9pm", ""))


if __name__ == "__main__":
    unittest.main()
