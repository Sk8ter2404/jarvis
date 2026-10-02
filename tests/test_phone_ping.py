"""Tests for core/phone_ping.py — the policy that decides when JARVIS texts
the owner's phone unprompted.

Every PhonePinger here is built with a FAKE bridge (a recorder standing in for
skills/phone_bridge.push_to_phone), a fake config dict, frozen wall / monotonic
clocks, and an inline spawn, so a send happens synchronously inside the call
and nothing touches the network, a thread, the real data/ dir or the system
clock (CI runs in UTC; every time below is an explicit naive datetime).

What is pinned:
  * no bridge = a no-op with ONE log line; master switch; per-category
    switches; never from a staging / test process;
  * the gates for an ordinary ping: dedupe, focus mode, he is here (an owner
    turn within PHONE_PING_AWAY_MIN), quiet hours (held, then ONE message when
    they end — or folded into a summary due soon), the hourly cap;
  * a critical (security) ping skips focus / presence / quiet hours / the
    hourly cap but has its own ceiling;
  * the scrubber: secret env values, bot tokens, sk- keys, long random tokens,
    "password is …" never leave the box;
  * the event sources: the Bambu state-change hook (transitions only; the
    first observation after a restart never pings), unanswered confirmations
    (names only, never arguments; once per prompt; retried while he is here),
    the daily summary (once a day, skipped when missed, deferred by focus);
  * persistence of the held backlog / journal across a restart;
  * the module singleton that skills/phone_bridge attaches to.

stdlib unittest + mock only (CI light tier).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from core import phone_ping as pp


def fake_secret(prefix: str, n: int) -> str:
    """A credential-SHAPED fixture built at run time, so no secret-looking
    literal sits in the source for tools/check_no_pii.py to (rightly) flag."""
    body = ("q7Xk2Lm9Pz4Rt8Vw3Yb6Nc1Hd5Jf0Gs" * 4)[:n]
    return prefix + body


class FakeBridge:
    """Stands in for phone_bridge.push_to_phone via the sender signature."""

    def __init__(self, ok=True, configured=True):
        self.ok = ok
        self.is_configured = configured
        self.sent: list[dict] = []

    def send(self, text, *, priority, title, category):
        self.sent.append({"text": text, "priority": priority, "title": title,
                          "category": category})
        return {"telegram": bool(self.ok)}

    def configured(self):
        return self.is_configured


class Clock:
    def __init__(self, wall=dt.datetime(2026, 10, 2, 14, 0), mono=10_000.0):
        self.wall = wall
        self.mono = mono

    def advance(self, seconds):
        self.wall += dt.timedelta(seconds=seconds)
        self.mono += seconds


def make(bridge=None, *, cfg=None, clock=None, idle=3600.0, focus=False,
         blocked="", pending=None, state_path=None, env=None, spawn=None,
         asleep=False):
    """A PhonePinger on fakes. Returns (pinger, bridge, clock, cfg, logs)."""
    bridge = bridge if bridge is not None else FakeBridge()
    conf = dict(pp.DEFAULTS)
    conf.update(cfg or {})
    clock = clock or Clock()
    logs: list[str] = []
    state = {"idle": idle, "focus": focus, "blocked": blocked,
             "pending": list(pending or []), "asleep": asleep}
    p = pp.PhonePinger(
        send=bridge.send, configured=bridge.configured,
        cfg=lambda name: conf.get(name),
        wall_now=lambda: clock.wall, mono_now=lambda: clock.mono,
        focus_active=lambda: state["focus"],
        owner_idle_s=lambda: state["idle"],
        asleep=lambda: state["asleep"],
        blocked=lambda: state["blocked"],
        pending=lambda: state["pending"],
        printer_now=lambda: "",
        env=lambda: (env if env is not None else {}),
        state_path=state_path,
        spawn=spawn or (lambda job: (job(), True)[1]),
        log=logs.append)
    p._test_state = state
    return p, bridge, clock, conf, logs


# ─── no bridge / switches / staging ──────────────────────────────────────

class GateTests(unittest.TestCase):
    def test_unconfigured_is_a_noop_with_exactly_one_log_line(self):
        p, bridge, _c, _cfg, logs = make(FakeBridge(configured=False))
        for _ in range(5):
            self.assertEqual(p.ping("print", "Print finished, sir."),
                             pp.UNCONFIGURED)
        p.tick()
        self.assertEqual(bridge.sent, [])
        lines = [l for l in logs if "not configured" in l]
        self.assertEqual(len(lines), 1, logs)
        self.assertIn("how do I connect my phone", lines[0])
        self.assertFalse(p.active())

    def test_no_sender_attached_counts_as_unconfigured(self):
        p = pp.PhonePinger(cfg=lambda n: pp.DEFAULTS.get(n), log=lambda s: None,
                           state_path=None)
        self.assertEqual(p.ping("print", "x"), pp.UNCONFIGURED)

    def test_master_switch_off(self):
        p, bridge, *_ = make(cfg={"PHONE_PING_ENABLED": False})
        for cat in ("print", "confirm", "robot"):
            with self.subTest(cat=cat):
                self.assertEqual(p.ping(cat, "Print finished."), pp.DISABLED)
        self.assertEqual(bridge.sent, [])

    def test_master_switch_off_never_silences_guard_alerts(self):
        """2026-10-02 review: "don't ping me" / "turn off phone pings" saved
        the master switch off and silently ended guard-mode pushes. Security
        answers to its own switch only."""
        p, bridge, *_ = make(cfg={"PHONE_PING_ENABLED": False})
        self.assertEqual(p.ping("security", "Intruder.", critical=True),
                         pp.QUEUED)
        self.assertEqual(len(bridge.sent), 1)
        p, bridge, *_ = make(cfg={"PHONE_PING_ENABLED": False,
                                  "PHONE_PING_SECURITY": False})
        self.assertEqual(p.ping("security", "Intruder.", critical=True),
                         pp.CATEGORY_OFF)
        self.assertEqual(bridge.sent, [])
        # ...and never from staging, nor without a bridge.
        p, bridge, *_ = make(cfg={"PHONE_PING_ENABLED": False},
                             blocked="staging")
        self.assertEqual(p.ping("security", "Intruder.", critical=True),
                         pp.BLOCKED)
        p, bridge, *_ = make(FakeBridge(configured=False),
                             cfg={"PHONE_PING_ENABLED": False})
        self.assertEqual(p.ping("security", "Intruder.", critical=True),
                         pp.UNCONFIGURED)

    def test_each_category_has_its_own_switch(self):
        for cat, flag in (("print", "PHONE_PING_PRINT"),
                          ("confirm", "PHONE_PING_CONFIRM"),
                          ("security", "PHONE_PING_SECURITY"),
                          ("robot", "PHONE_PING_ROBOT")):
            with self.subTest(cat=cat):
                p, bridge, *_ = make(cfg={flag: False})
                self.assertEqual(p.ping(cat, "something"), pp.CATEGORY_OFF)
                other = "robot" if cat != "robot" else "print"
                self.assertEqual(p.ping(other, "something else"), pp.QUEUED)
                self.assertEqual(len(bridge.sent), 1)

    def test_summary_is_off_by_default(self):
        self.assertIs(pp.DEFAULTS["PHONE_PING_SUMMARY"], False)
        for flag in ("PHONE_PING_ENABLED", "PHONE_PING_PRINT",
                     "PHONE_PING_SECURITY", "PHONE_PING_ROBOT"):
            self.assertIs(pp.DEFAULTS[flag], True, flag)

    def test_confirm_pings_are_off_by_default(self):
        """2026-10-02 review: a confirmation lapses after 45 s, but the ping
        needs it 2+ minutes old and him 10 minutes away - so every one would
        be the lapsed "nothing ran" kind, needing nothing from him."""
        self.assertIs(pp.DEFAULTS["PHONE_PING_CONFIRM"], False)
        p, bridge, *_ = make(pending=[{"key": "queue:1.0", "age_s": 900.0,
                                       "text": "waiting", "owner": True}])
        p.tick()
        self.assertEqual(bridge.sent, [])

    def test_never_from_staging(self):
        p, bridge, *_ = make(blocked="staging")
        self.assertEqual(p.ping("security", "Intruder.", critical=True),
                         pp.BLOCKED)
        self.assertEqual(bridge.sent, [])

    def test_live_blocked_reads_staging_and_test_mode(self):
        with mock.patch.dict(os.environ, {"JARVIS_STAGING": "1"}):
            self.assertEqual(pp._live_blocked(), "staging")
        with mock.patch.dict(os.environ, {"JARVIS_STAGING": "",
                                          "JARVIS_TEST_MODE": "1"}):
            self.assertEqual(pp._live_blocked(), "test mode")

    def test_empty_and_unknown_category(self):
        p, bridge, *_ = make()
        self.assertEqual(p.ping("print", "   "), pp.EMPTY)
        self.assertEqual(p.ping("weather", "Rain."), pp.UNKNOWN)
        # The summary only goes out through the scheduled path.
        self.assertEqual(p.ping("summary", "hi"), pp.UNKNOWN)
        self.assertEqual(bridge.sent, [])


# ─── delivery ────────────────────────────────────────────────────────────

class DeliveryTests(unittest.TestCase):
    def test_happy_path_sends_and_journals(self):
        p, bridge, *_ = make()
        out = p.ping("print", "Print finished, sir: 'benchy' is done.")
        self.assertEqual(out, pp.QUEUED)
        self.assertEqual(len(bridge.sent), 1)
        sent = bridge.sent[0]
        self.assertEqual(sent["text"], "Print finished, sir: 'benchy' is done.")
        self.assertEqual(sent["priority"], "normal")
        self.assertEqual(sent["category"], "print")
        self.assertEqual(sent["title"], "JARVIS")
        self.assertEqual(p._journal[-1]["out"], pp.SENT)

    def test_bridge_failure_is_journalled_as_failed(self):
        p, bridge, *_ = make(FakeBridge(ok=False))
        self.assertEqual(p.ping("print", "Print failed."), pp.QUEUED)
        self.assertEqual(p._journal[-1]["out"], pp.FAILED)

    def test_full_sender_queue_marks_failed(self):
        p, bridge, *_ = make(spawn=lambda job: False)
        self.assertEqual(p.ping("print", "Print finished."), pp.QUEUED)
        self.assertEqual(bridge.sent, [])
        self.assertEqual(p._journal[-1]["out"], pp.FAILED)

    def test_send_is_off_the_callers_thread_by_default(self):
        """The real spawn queues the job for the sender thread: the caller
        (the Bambu MQTT callback, the guard monitor) never waits on HTTP."""
        bridge = FakeBridge()
        p = pp.PhonePinger(send=bridge.send, configured=bridge.configured,
                           cfg=lambda n: pp.DEFAULTS.get(n),
                           owner_idle_s=lambda: 9999.0,
                           focus_active=lambda: False, blocked=lambda: "",
                           wall_now=lambda: dt.datetime(2026, 10, 2, 14, 0),
                           state_path=None, log=lambda s: None)
        started = []
        with mock.patch.object(pp.threading.Thread, "start",
                               lambda self: started.append(self.name)):
            self.assertEqual(p.ping("print", "Print finished."), pp.QUEUED)
        self.assertEqual(started, ["phone-ping-sender"])
        self.assertEqual(bridge.sent, [])          # not sent inline
        self.assertEqual(p._q.qsize(), 1)
        p._q.get_nowait()()                        # the sender's job
        self.assertEqual(len(bridge.sent), 1)

    def test_text_is_clipped(self):
        p, bridge, *_ = make()
        p.ping("robot", "word " * 400)
        self.assertLessEqual(len(bridge.sent[0]["text"]), pp.MESSAGE_MAX)

    def test_a_raising_policy_never_reaches_the_caller(self):
        p, bridge, *_ = make()
        p.focus_active = mock.Mock(side_effect=RuntimeError("boom"))
        p.in_quiet_hours = mock.Mock(side_effect=RuntimeError("boom"))
        self.assertEqual(p.ping("print", "x"), pp.FAILED)


# ─── presence / focus / quiet hours / rate limit / dedupe ───────────────

class PolicyTests(unittest.TestCase):
    def test_goodnight_holds_pings_like_quiet_hours(self):
        """2026-10-02 review: "goodnight" at 22:00 (overnight mode) and a
        print finishing at 22:30 must not buzz the phone at his bedside: the
        ping waits for the morning digest, and goes out once he wakes."""
        clock = Clock(wall=dt.datetime(2026, 10, 2, 22, 30))
        p, bridge, clock, *_ = make(clock=clock, asleep=True)
        self.assertEqual(p.ping("print", "Print finished, sir."), pp.QUIET)
        self.assertEqual(bridge.sent, [])
        p.tick()                                  # still asleep: held
        self.assertEqual(bridge.sent, [])
        # Guard alerts are never held.
        self.assertEqual(p.ping("security", "Intruder.", critical=True),
                         pp.QUEUED)
        p._test_state["asleep"] = False
        clock.wall = dt.datetime(2026, 10, 3, 7, 5)
        p.tick()
        self.assertEqual(len(bridge.sent), 2)
        self.assertIn("Print finished", bridge.sent[1]["text"])

    def test_present_owner_is_not_pinged(self):
        p, bridge, *_ = make(idle=120.0)            # spoke 2 minutes ago
        self.assertEqual(p.ping("print", "Print finished."), pp.PRESENT)
        self.assertEqual(bridge.sent, [])
        p._test_state["idle"] = 11 * 60.0
        self.assertEqual(p.ping("print", "Print finished again."), pp.QUEUED)

    def test_away_minutes_zero_pings_even_while_present(self):
        p, bridge, *_ = make(idle=5.0, cfg={"PHONE_PING_AWAY_MIN": 0})
        self.assertEqual(p.ping("print", "Print finished."), pp.QUEUED)

    def test_unknown_presence_counts_as_away(self):
        p, bridge, *_ = make(idle=None)
        self.assertEqual(p.ping("print", "Print finished."), pp.QUEUED)

    def test_focus_mode_drops_ordinary_but_not_critical(self):
        p, bridge, *_ = make(focus=True)
        self.assertEqual(p.ping("print", "Print finished."), pp.FOCUS)
        self.assertEqual(p.ping("security", "Someone is at the desk.",
                                critical=True), pp.QUEUED)
        self.assertEqual([s["category"] for s in bridge.sent], ["security"])
        self.assertEqual(bridge.sent[0]["priority"], "high")

    def test_quiet_hours_hold_ordinary_pings(self):
        for hh, mm in ((23, 0), (23, 30), (2, 0), (6, 59)):
            with self.subTest(time=f"{hh}:{mm}"):
                p, bridge, *_ = make(clock=Clock(dt.datetime(2026, 10, 2, hh, mm)))
                self.assertEqual(p.ping("print", "Print finished."), pp.QUIET)
                self.assertEqual(bridge.sent, [])
                self.assertEqual(len(p._held), 1)
        for hh, mm in ((7, 0), (12, 0), (22, 59)):
            with self.subTest(time=f"{hh}:{mm}"):
                p, bridge, *_ = make(clock=Clock(dt.datetime(2026, 10, 2, hh, mm)))
                self.assertEqual(p.ping("print", "Print finished."), pp.QUEUED)

    def test_quiet_hours_do_not_hold_critical(self):
        p, bridge, *_ = make(clock=Clock(dt.datetime(2026, 10, 2, 3, 0)))
        self.assertEqual(p.ping("security", "Motion on the desk camera.",
                                critical=True, priority="urgent"), pp.QUEUED)
        self.assertEqual(bridge.sent[0]["priority"], "urgent")

    def test_quiet_window_variants(self):
        p, *_ = make(cfg={"PHONE_PING_QUIET_START": "13:00",
                          "PHONE_PING_QUIET_END": "14:30"})
        self.assertTrue(p.in_quiet_hours(dt.datetime(2026, 1, 1, 13, 0)))
        self.assertTrue(p.in_quiet_hours(dt.datetime(2026, 1, 1, 14, 29)))
        self.assertFalse(p.in_quiet_hours(dt.datetime(2026, 1, 1, 14, 30)))
        self.assertFalse(p.in_quiet_hours(dt.datetime(2026, 1, 1, 12, 59)))
        p2, *_ = make(cfg={"PHONE_PING_QUIET_START": "07:00",
                           "PHONE_PING_QUIET_END": "07:00"})
        self.assertFalse(p2.in_quiet_hours(dt.datetime(2026, 1, 1, 3, 0)))
        # A garbage setting falls back to the 23:00-07:00 default.
        p3, *_ = make(cfg={"PHONE_PING_QUIET_START": "late",
                           "PHONE_PING_QUIET_END": None})
        self.assertTrue(p3.in_quiet_hours(dt.datetime(2026, 1, 1, 23, 30)))
        self.assertFalse(p3.in_quiet_hours(dt.datetime(2026, 1, 1, 8, 0)))

    def test_hourly_cap(self):
        p, bridge, clock, *_ = make()
        outs = [p.ping("robot", f"event {i}") for i in range(8)]
        self.assertEqual(outs[:6], [pp.QUEUED] * 6)
        self.assertEqual(outs[6:], [pp.RATE_LIMITED] * 2)
        self.assertEqual(len(bridge.sent), 6)
        clock.advance(3601)
        self.assertEqual(p.ping("robot", "event later"), pp.QUEUED)

    def test_hourly_cap_is_a_setting(self):
        p, bridge, *_ = make(cfg={"PHONE_PING_MAX_PER_HOUR": 2})
        outs = [p.ping("robot", f"event {i}") for i in range(3)]
        self.assertEqual(outs, [pp.QUEUED, pp.QUEUED, pp.RATE_LIMITED])

    def test_critical_has_its_own_ceiling_and_does_not_use_the_cap(self):
        p, bridge, *_ = make(cfg={"PHONE_PING_MAX_PER_HOUR": 1})
        self.assertEqual(p.ping("robot", "ordinary"), pp.QUEUED)
        outs = [p.ping("security", f"alert {i}", critical=True)
                for i in range(pp.CRITICAL_MAX_PER_HOUR + 1)]
        self.assertEqual(outs.count(pp.QUEUED), pp.CRITICAL_MAX_PER_HOUR)
        self.assertEqual(outs[-1], pp.RATE_LIMITED)

    def test_dedupe_key(self):
        p, bridge, clock, *_ = make()
        self.assertEqual(p.ping("print", "done", dedupe_key="k"), pp.QUEUED)
        self.assertEqual(p.ping("print", "done", dedupe_key="k"), pp.DEDUPED)
        clock.advance(pp.DEDUPE_S + 1)
        self.assertEqual(p.ping("print", "done", dedupe_key="k"), pp.QUEUED)

    def test_a_present_outcome_does_not_burn_the_dedupe_key(self):
        p, bridge, *_ = make(idle=30.0, cfg={"PHONE_PING_CONFIRM": True})
        self.assertEqual(p.ping("confirm", "waiting", dedupe_key="c"),
                         pp.PRESENT)
        # The retry while he is still here is not journalled twice.
        self.assertEqual(p.ping("confirm", "waiting", dedupe_key="c"),
                         pp.PRESENT)
        self.assertEqual(len([e for e in p._journal if e["out"] == pp.PRESENT]),
                         1)
        p._test_state["idle"] = 3600.0
        self.assertEqual(p.ping("confirm", "waiting", dedupe_key="c"),
                         pp.QUEUED)
        self.assertEqual(len(bridge.sent), 1)


# ─── scrubbing ───────────────────────────────────────────────────────────

class ScrubTests(unittest.TestCase):
    BOT_TOKEN = fake_secret("123456789:", 35)

    def test_secret_env_values_are_redacted(self):
        env = {"TELEGRAM_BOT_TOKEN": self.BOT_TOKEN,
               "NTFY_TOPIC": "jarvis-7f3k2q9x",
               "BAMBU_ACCESS_CODE": "48213377",
               "USERNAME": "someone"}
        text = (f"token {self.BOT_TOKEN} topic jarvis-7f3k2q9x code 48213377 "
                f"user someone")
        clean, _withheld = pp.scrub(text, env=env)
        for secret in (self.BOT_TOKEN, "jarvis-7f3k2q9x", "48213377"):
            self.assertNotIn(secret, clean)
        self.assertIn("someone", clean)   # not a secret-named variable

    def test_credential_shapes_are_redacted(self):
        samples = [
            self.BOT_TOKEN,
            fake_secret("sk" + "-ant-api03-", 30),
            fake_secret("gh" + "p_", 36),
            "AK" + "IA" + "Q7XK2LM9PZ4RT8VW",
            fake_secret("", 40),
        ]
        for s in samples:
            with self.subTest(s=s[:12]):
                clean, _w = pp.scrub(f"Robot says {s} ok", env={})
                self.assertNotIn(s, clean)
                self.assertIn("[redacted]", clean)

    def test_a_credential_sentence_is_withheld_entirely(self):
        clean, withheld = pp.scrub("the wifi password is hunter2", env={})
        self.assertNotIn("hunter2", clean)
        self.assertTrue(withheld)

    def test_withheld_text_is_replaced_by_a_generic_line(self):
        p, bridge, *_ = make()
        self.assertEqual(p.ping("robot", "My API key is "
                                + fake_secret("sk" + "-ant-", 24)),
                         pp.QUEUED)
        sent = bridge.sent[0]["text"]
        self.assertNotIn("sk-ant", sent)
        self.assertEqual(sent, pp._WITHHELD["robot"])

    def test_ordinary_text_passes_untouched(self):
        for text in ("Print finished, sir: 'benchy v2 0 2mm' is done.",
                     "The print is paused with error 50348044 at layer 40 "
                     "of 212, sir: 'tiger pendant' needs you."):
            clean, withheld = pp.scrub(text, env={"X_TOKEN": "abcdef123"})
            self.assertEqual(clean, text)
            self.assertFalse(withheld)

    def test_the_live_env_is_scrubbed_by_default(self):
        bridge = FakeBridge()
        value = fake_secret("", 30)
        p, bridge, *_ = make(bridge, env={"PUSHOVER_TOKEN": value})
        p.ping("robot", f"stuck, see {value}")
        self.assertNotIn(value, bridge.sent[0]["text"])


# ─── held backlog + summary ──────────────────────────────────────────────

class HeldAndSummaryTests(unittest.TestCase):
    def test_held_pings_go_out_as_one_message_when_quiet_hours_end(self):
        clock = Clock(dt.datetime(2026, 10, 2, 2, 14))
        p, bridge, *_ = make(clock=clock)
        self.assertEqual(p.ping("print", "Print finished, sir: 'benchy' is done."),
                         pp.QUIET)
        clock.advance(3600)
        self.assertEqual(p.ping("print", "Print failed, sir: 'cube'."), pp.QUIET)
        clock.wall = dt.datetime(2026, 10, 2, 6, 50)
        p.tick()
        self.assertEqual(bridge.sent, [])          # still quiet
        clock.wall = dt.datetime(2026, 10, 2, 7, 1)
        p.tick()
        self.assertEqual(len(bridge.sent), 1)
        digest = bridge.sent[0]["text"]
        self.assertIn("While you were away overnight", digest)
        self.assertIn("02:14 Print finished, sir: 'benchy' is done.", digest)
        self.assertIn("03:14 Print failed, sir: 'cube'.", digest)
        self.assertEqual(p._held, [])
        p.tick()
        self.assertEqual(len(bridge.sent), 1)      # once

    def test_a_summary_due_soon_carries_the_held_pings_instead(self):
        clock = Clock(dt.datetime(2026, 10, 2, 2, 14))
        p, bridge, *_ = make(clock=clock, cfg={"PHONE_PING_SUMMARY": True,
                                               "PHONE_PING_SUMMARY_TIME": "07:30"})
        p.ping("print", "Print finished, sir: 'benchy' is done.")
        clock.wall = dt.datetime(2026, 10, 2, 7, 1)
        p.tick()
        self.assertEqual(bridge.sent, [])
        clock.wall = dt.datetime(2026, 10, 2, 7, 30)
        p.tick()
        self.assertEqual(len(bridge.sent), 1)
        text = bridge.sent[0]["text"]
        self.assertTrue(text.startswith("Morning summary, sir."), text)
        self.assertIn("02:14 Print finished", text)
        self.assertEqual(bridge.sent[0]["category"], "summary")
        self.assertEqual(p._held, [])

    def test_summary_off_sends_nothing(self):
        clock = Clock(dt.datetime(2026, 10, 2, 7, 30))
        p, bridge, *_ = make(clock=clock)
        p.tick()
        self.assertEqual(bridge.sent, [])

    def test_summary_once_a_day_and_all_quiet(self):
        clock = Clock(dt.datetime(2026, 10, 2, 7, 29))
        p, bridge, *_ = make(clock=clock, cfg={"PHONE_PING_SUMMARY": True})
        p.tick()
        self.assertEqual(bridge.sent, [])
        clock.wall = dt.datetime(2026, 10, 2, 7, 31)
        p.tick()
        p.tick()
        self.assertEqual(len(bridge.sent), 1)
        self.assertIn("all quiet since yesterday 07:31", bridge.sent[0]["text"])
        clock.wall = dt.datetime(2026, 10, 3, 7, 31)
        p.tick()
        self.assertEqual(len(bridge.sent), 2)
        self.assertIn("since yesterday 07:31", bridge.sent[1]["text"])

    def test_summary_lists_what_happened(self):
        clock = Clock(dt.datetime(2026, 10, 2, 13, 5))
        p, bridge, *_ = make(clock=clock, cfg={"PHONE_PING_SUMMARY": True,
                                               "PHONE_PING_SUMMARY_TIME": "22:00"})
        p.ping("print", "Print finished, sir: 'benchy' is done.")
        p._test_state["idle"] = 5.0
        clock.advance(600)
        p.ping("robot", "The robot is stuck.")       # present: he heard it
        clock.wall = dt.datetime(2026, 10, 2, 22, 0)
        p.tick()
        text = bridge.sent[-1]["text"]
        self.assertTrue(text.startswith("Evening summary, sir. 2 things since"),
                        text)
        self.assertIn("13:05 Print finished", text)
        self.assertIn("13:15 The robot is stuck.", text)

    def test_a_missed_summary_is_skipped_not_sent_late(self):
        clock = Clock(dt.datetime(2026, 10, 2, 15, 0))
        p, bridge, _c, _cfg, logs = make(clock=clock,
                                         cfg={"PHONE_PING_SUMMARY": True})
        p.tick()
        self.assertEqual(bridge.sent, [])
        self.assertEqual(p._summary_skipped_on, "2026-10-02")
        self.assertTrue(any("missed" in l for l in logs))

    def test_focus_defers_the_summary(self):
        clock = Clock(dt.datetime(2026, 10, 2, 7, 30))
        p, bridge, *_ = make(clock=clock, focus=True,
                             cfg={"PHONE_PING_SUMMARY": True})
        p.tick()
        self.assertEqual(bridge.sent, [])
        p._test_state["focus"] = False
        clock.advance(600)
        p.tick()
        self.assertEqual(len(bridge.sent), 1)

    def test_summary_adds_the_printer_line(self):
        clock = Clock(dt.datetime(2026, 10, 2, 7, 30))
        p, bridge, *_ = make(clock=clock, cfg={"PHONE_PING_SUMMARY": True})
        p.printer_now = lambda: "Printer: printing 67% (benchy)."
        p.tick()
        self.assertIn("Printer: printing 67% (benchy).", bridge.sent[0]["text"])


# ─── persistence ─────────────────────────────────────────────────────────

class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="phone_ping_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, "phone_ping_state.json")

    def test_held_backlog_survives_a_restart(self):
        clock = Clock(dt.datetime(2026, 10, 2, 2, 14))
        p, *_ = make(clock=clock, state_path=self.path)
        p.ping("print", "Print finished, sir: 'benchy' is done.")
        self.assertTrue(os.path.exists(self.path))
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data["held"]), 1)
        # A new process (the overnight upgrade restarts JARVIS).
        clock.wall = dt.datetime(2026, 10, 2, 7, 5)
        p2, bridge2, *_ = make(clock=clock, state_path=self.path)
        p2.tick()
        self.assertEqual(len(bridge2.sent), 1)
        self.assertIn("Print finished", bridge2.sent[0]["text"])

    def test_a_corrupt_state_file_is_ignored(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("[1, 2")
        p, bridge, *_ = make(state_path=self.path)
        self.assertEqual(p.ping("print", "Print finished."), pp.QUEUED)

    def test_malformed_rows_are_dropped_on_load(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"held": [{"t": "x"}, {"t": 1.0, "cat": "evil",
                                             "text": "x"},
                                {"t": 1.0, "cat": "print", "text": "ok"}],
                       "journal": "nope", "summary_sent_on": 5}, f)
        p, *_ = make(state_path=self.path)
        p._load()
        self.assertEqual([h["text"] for h in p._held], ["ok"])
        self.assertEqual(p._journal, [])
        self.assertEqual(p._summary_sent_on, "")


# ─── event sources ───────────────────────────────────────────────────────

class BambuHookTests(unittest.TestCase):
    SNAP = {"gcode_state": "FINISH", "filename": "/sdcard/Tiger_pendant-FAST.gcode.3mf",
            "layer_num": 212, "total_layer": 212, "print_error": 0}

    def test_finish_after_running_pings_once(self):
        p, bridge, *_ = make()
        p.on_bambu_state(dict(self.SNAP), "RUNNING", "FINISH")
        p.on_bambu_state(dict(self.SNAP), "RUNNING", "FINISH")
        p.on_bambu_state(dict(self.SNAP), "FINISH", "FINISH")
        self.assertEqual(len(bridge.sent), 1)
        self.assertEqual(bridge.sent[0]["text"],
                         "Print finished, sir: 'Tiger pendant FAST' is done.")

    def test_first_observation_after_a_restart_never_pings(self):
        p, bridge, *_ = make()
        p.on_bambu_state(dict(self.SNAP), None, "FINISH")
        p.on_bambu_state(dict(self.SNAP, gcode_state="FAILED"), "IDLE", "FAILED")
        self.assertEqual(bridge.sent, [])

    def test_failed_names_the_layer(self):
        p, bridge, *_ = make()
        snap = dict(self.SNAP, gcode_state="FAILED", layer_num=47)
        p.on_bambu_state(snap, "RUNNING", "FAILED")
        self.assertEqual(len(bridge.sent), 1)
        self.assertIn("Print failed at layer 47 of 212", bridge.sent[0]["text"])
        self.assertEqual(bridge.sent[0]["priority"], "high")

    def test_pause_with_an_error_pings_but_a_plain_pause_does_not(self):
        p, bridge, *_ = make()
        p.on_bambu_state(dict(self.SNAP, gcode_state="PAUSE"), "RUNNING", "PAUSE")
        self.assertEqual(bridge.sent, [])
        snap = dict(self.SNAP, gcode_state="PAUSE", print_error=117473282,
                    layer_num=40)
        p.on_bambu_state(snap, "PAUSE", "PAUSE")
        p.on_bambu_state(snap, "PAUSE", "PAUSE")
        self.assertEqual(len(bridge.sent), 1)
        self.assertIn("paused with error 117473282 at layer 40",
                      bridge.sent[0]["text"])

    # -- 2026-10-02 review: the hook runs on EVERY MQTT push --
    PAUSED = {"gcode_state": "PAUSE", "filename": "cube.3mf",
              "print_error": 117473282, "layer_num": 9}

    def test_a_pause_seen_after_a_restart_never_pings(self):
        """The first observation after a restart (prev None), and every push
        after it while the printer stays paused (prev PAUSE), is not a pause
        starting: no ping per JARVIS restart (the nightly upgrade too)."""
        p, bridge, *_ = make()
        p.on_bambu_state(dict(self.PAUSED), None, "PAUSE")
        for _ in range(5):
            p.on_bambu_state(dict(self.PAUSED), "PAUSE", "PAUSE")
        self.assertEqual(bridge.sent, [])

    def test_a_long_pause_is_one_ping_not_one_an_hour(self):
        p, bridge, clock, *_ = make()
        p.on_bambu_state(dict(self.PAUSED), "RUNNING", "PAUSE")
        for _ in range(6):
            clock.advance(3600.0)
            p.on_bambu_state(dict(self.PAUSED), "PAUSE", "PAUSE")
        self.assertEqual(len(bridge.sent), 1)

    def test_a_new_error_in_the_same_pause_pings_once_more(self):
        p, bridge, *_ = make()
        p.on_bambu_state(dict(self.PAUSED), "RUNNING", "PAUSE")
        p.on_bambu_state(dict(self.PAUSED, print_error=50364437), "PAUSE",
                         "PAUSE")
        p.on_bambu_state(dict(self.PAUSED, print_error=50364437), "PAUSE",
                         "PAUSE")
        self.assertEqual(len(bridge.sent), 2)

    def test_a_held_pause_is_one_morning_line(self):
        """In quiet hours a paused printer re-reported for hours is ONE line
        in the 07:00 digest, not one per hour."""
        clock = Clock(wall=dt.datetime(2026, 10, 2, 1, 0))
        p, bridge, clock, *_ = make(clock=clock)
        p.on_bambu_state(dict(self.PAUSED), "RUNNING", "PAUSE")
        for _ in range(4):
            clock.advance(3600.0)
            p.on_bambu_state(dict(self.PAUSED), "PAUSE", "PAUSE")
        # The same event re-reported by another source (same key) after the
        # dedupe window is still one held line.
        p.ping("print", "The print is paused, sir.",
               dedupe_key="print:pause:cube:117473282")
        self.assertEqual(len(p._held), 1)
        clock.wall = dt.datetime(2026, 10, 2, 7, 1)
        p.tick()
        self.assertEqual(len(bridge.sent), 1)
        self.assertEqual(bridge.sent[0]["text"].count("paused"), 1)

    def test_a_cancelled_print_is_not_a_failure(self):
        """Cancelling from Bambu Handy or the printer goes FAILED with the
        cancel code (HMS 0300-400C): he did it himself."""
        p, bridge, *_ = make()
        for code in (0x0300400C, "50348044", "0x0300400C"):
            with self.subTest(code=code):
                p.on_bambu_state(dict(self.SNAP, gcode_state="FAILED",
                                      print_error=code), "RUNNING", "FAILED")
        self.assertEqual(bridge.sent, [])
        p.on_bambu_state(dict(self.SNAP, gcode_state="FAILED",
                              print_error=50364437), "RUNNING", "FAILED")
        self.assertEqual(len(bridge.sent), 1)

    def test_a_reprint_of_the_same_file_pings_again(self):
        """A second copy of the same file within the hour gets its own finish
        ping: a new print starting drops the old run's keys."""
        p, bridge, clock, *_ = make()
        p.on_bambu_state(dict(self.SNAP), "RUNNING", "FINISH")
        clock.advance(600.0)
        p.on_bambu_state(dict(self.SNAP, gcode_state="PREPARE"), "FINISH",
                         "PREPARE")
        p.on_bambu_state(dict(self.SNAP, gcode_state="RUNNING"), "PREPARE",
                         "RUNNING")
        clock.advance(1200.0)
        p.on_bambu_state(dict(self.SNAP), "RUNNING", "FINISH")
        self.assertEqual(len(bridge.sent), 2)
        # A retry after a failure, too.
        p.on_bambu_state(dict(self.SNAP, gcode_state="FAILED", print_error=5),
                         "RUNNING", "FAILED")
        p.on_bambu_state(dict(self.SNAP, gcode_state="RUNNING"), "FAILED",
                         "RUNNING")
        p.on_bambu_state(dict(self.SNAP, gcode_state="FAILED", print_error=5),
                         "RUNNING", "FAILED")
        self.assertEqual(len(bridge.sent), 4)

    def test_a_resume_does_not_reset_the_pause_report(self):
        """Pause / resume / pause with the same error inside the hour: the
        error was already reported this print."""
        p, bridge, *_ = make()
        p.on_bambu_state(dict(self.PAUSED), "RUNNING", "PAUSE")
        p.on_bambu_state(dict(self.PAUSED, gcode_state="RUNNING"), "PAUSE",
                         "RUNNING")
        p.on_bambu_state(dict(self.PAUSED), "RUNNING", "PAUSE")
        self.assertEqual(len(bridge.sent), 1)

    def test_print_switch_off_silences_the_hook(self):
        p, bridge, *_ = make(cfg={"PHONE_PING_PRINT": False})
        p.on_bambu_state(dict(self.SNAP), "RUNNING", "FINISH")
        self.assertEqual(bridge.sent, [])

    def test_a_garbage_snapshot_never_raises(self):
        p, bridge, *_ = make()
        p.on_bambu_state(None, "RUNNING", "FINISH")
        p.on_bambu_state({"layer_num": "x"}, object(), None)
        self.assertEqual(bridge.sent[0]["text"],
                         "Print finished, sir: your print is done.")

    def test_tick_registers_the_hook_once(self):
        p, *_ = make()
        mod = types.ModuleType("skill_bambu_monitor")
        mod.hooks = []
        mod.register_state_change_hook = mod.hooks.append
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": mod}):
            p.tick()
            p.tick()
        self.assertEqual(len(mod.hooks), 1)
        self.assertEqual(mod.hooks[0], p.on_bambu_state)

    def test_hook_against_the_real_bambu_monitor_fanout(self):
        """skills/bambu_monitor's own register/fire pair calls the hook with
        (snapshot, prev, current)."""
        from tests._skill_harness import load_skill_isolated
        try:
            bm, _a = load_skill_isolated("bambu_monitor", register=False)
        except unittest.SkipTest:
            raise
        self.addCleanup(sys.modules.pop, "skill_bambu_monitor", None)
        p, bridge, *_ = make()
        bm.register_state_change_hook(p.on_bambu_state)
        with bm._state_lock:
            bm._state.update({"gcode_state": "FINISH", "filename": "cube.3mf"})
        bm._fire_state_change_hooks("RUNNING", "FINISH")
        self.assertEqual(bridge.sent[0]["text"],
                         "Print finished, sir: 'cube' is done.")


_CONFIRM_ON = {"PHONE_PING_CONFIRM": True}


class ConfirmTests(unittest.TestCase):
    def test_pings_once_after_n_minutes_while_away(self):
        item = {"key": "queue:1.000", "age_s": 60.0,
                "text": pp.confirm_text(["reset_memory"], lapsed=True)}
        p, bridge, *_ = make(pending=[item], cfg=_CONFIRM_ON)
        p.tick()
        self.assertEqual(bridge.sent, [])           # 1 min < 2 min
        item["age_s"] = 11 * 60.0
        p.tick()
        p.tick()
        self.assertEqual(len(bridge.sent), 1)
        self.assertEqual(bridge.sent[0]["category"], "confirm")
        self.assertIn("'reset memory'", bridge.sent[0]["text"])
        self.assertIn("Nothing ran", bridge.sent[0]["text"])

    def test_a_prompt_left_for_hours_still_pings_only_once(self):
        """Past the one-hour dedupe window the per-prompt memory still holds:
        a lapsed prompt nobody answered all afternoon is ONE ping, not one an
        hour."""
        item = {"key": "queue:9.000", "age_s": 700.0, "text": "waiting"}
        p, bridge, clock, *_ = make(pending=[item], cfg=_CONFIRM_ON)
        p.tick()
        for _ in range(4):
            clock.advance(3 * 3600)
            item["age_s"] += 3 * 3600
            p.tick()
        self.assertEqual(len(bridge.sent), 1)

    def test_waits_while_he_is_here_then_pings(self):
        item = {"key": "queue:2.000", "age_s": 300.0, "text": "waiting"}
        p, bridge, *_ = make(pending=[item], idle=60.0, cfg=_CONFIRM_ON)
        p.tick()
        self.assertEqual(bridge.sent, [])
        p._test_state["idle"] = 700.0
        p.tick()
        self.assertEqual(len(bridge.sent), 1)

    def test_confirm_after_minutes_is_a_setting(self):
        item = {"key": "queue:3.000", "age_s": 300.0, "text": "waiting"}
        p, bridge, *_ = make(pending=[item],
                             cfg={"PHONE_PING_CONFIRM_AFTER_MIN": 10,
                                  "PHONE_PING_CONFIRM": True})
        p.tick()
        self.assertEqual(bridge.sent, [])

    def test_confirm_text_never_carries_an_argument_or_odd_name(self):
        t = pp.confirm_text(["run_shell", "Rm -rf /", "send_email", "x"],
                            lapsed=True)
        self.assertIn("'run shell' and 'an action' and 2 more", t)
        self.assertNotIn("rf", t)
        self.assertIn("lapsed", t)
        self.assertIn("is waiting on your yes",
                      pp.confirm_text(["shutdown_pc"], lapsed=False))

    def test_a_prompt_he_did_not_raise_never_pings(self):
        """A skill's own offer, or a prompt no turn of his raised (the TV, a
        guest), is not texted - and is not re-checked every tick."""
        item = {"key": "queue:4.000", "age_s": 900.0, "text": "waiting",
                "owner": False}
        p, bridge, *_ = make(pending=[item], cfg=_CONFIRM_ON)
        p.tick()
        p.tick()
        self.assertEqual(bridge.sent, [])
        self.assertIn("queue:4.000", p._pending_done)

    def test_live_probe_marks_whose_prompt_it_is(self):
        now = pp.time.monotonic()
        bc = types.ModuleType("bobert_companion")
        bc._pending_confirmation = [("reset_memory", "")]
        bc.CONFIRMATION_TTL_S = 45.0
        bc._shutdown_prompt_pending = {}
        cases = (
            (now - 700.0, now - 705.0, True),    # his turn 5 s before
            (now - 700.0, now - 1000.0, False),  # a skill's offer, long after
            (now - 700.0, 0.0, False),           # no turn of his at all
            (now - 700.0, now - 600.0, False),   # he spoke after it was raised
        )
        for at, owner_at, want in cases:
            with self.subTest(owner_at=owner_at):
                bc._pending_confirmation_at = [at]
                bc._last_owner_turn_at = [owner_at]
                with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
                    items = pp._live_pending()
                self.assertEqual(items[0]["owner"], want)
        # The shutdown prompt: armed (wall clock) 600 s ago, his turn 2 s
        # before it.
        bc._pending_confirmation = []
        bc._last_owner_turn_at = [now - 602.0]
        bc.SHUTDOWN_PROMPT_TIMEOUT_S = 30.0
        bc._shutdown_prompt_pending = {"armed": True,
                                       "expires_at": pp.time.time() - 570.0}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            items = pp._live_pending()
        self.assertTrue(items[0]["owner"])
        bc._last_owner_turn_at = [now - 3000.0]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertFalse(pp._live_pending()[0]["owner"])

    def test_live_probe_reads_names_only(self):
        bc = types.ModuleType("bobert_companion")
        bc._pending_confirmation = [("run_shell", "echo hunter2 > secret.txt"),
                                    ("send_email", "to boss: I quit")]
        bc._pending_confirmation_at = [pp.time.monotonic() - 700.0]
        bc.CONFIRMATION_TTL_S = 45.0
        bc._shutdown_prompt_pending = {"armed": True,
                                       "expires_at": pp.time.time() - 600.0}
        bc.SHUTDOWN_PROMPT_TIMEOUT_S = 30.0
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            items = pp._live_pending()
        self.assertEqual(len(items), 2)
        q = items[0]
        self.assertTrue(q["key"].startswith("queue:"))
        self.assertGreater(q["age_s"], 600)
        self.assertIn("'run shell' and 'send email'", q["text"])
        for secret in ("hunter2", "boss", "quit", "echo"):
            self.assertNotIn(secret, q["text"])
        self.assertEqual(items[1]["text"], pp.SHUTDOWN_PROMPT_TEXT)
        self.assertGreater(items[1]["age_s"], 600)

    def test_live_probe_empty_queue_and_no_monolith(self):
        bc = types.ModuleType("bobert_companion")
        bc._pending_confirmation = []
        bc._pending_confirmation_at = [0.0]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertEqual(pp._live_pending(), [])
        with mock.patch.dict(sys.modules):
            sys.modules.pop("bobert_companion", None)
            self.assertEqual(pp._live_pending(), [])


class LiveSeamTests(unittest.TestCase):
    def test_owner_idle_from_the_monolith_stamp(self):
        bc = types.ModuleType("bobert_companion")
        bc._last_owner_turn_at = [pp.time.monotonic() - 125.0]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(pp, "_live_input_idle_s", return_value=None):
            idle = pp._live_owner_idle_s()
        self.assertTrue(124.0 <= idle < 200.0, idle)

    def _awake_bc(self, *, sleep=False, standby=False, muted=False):
        bc = types.ModuleType("bobert_companion")
        bc._last_owner_turn_at = [pp.time.monotonic() - 1800.0]
        bc._sleep_mode = [sleep]
        bc._standby_mode = [standby]
        bc._tts_muted = [muted]
        return bc

    def test_keyboard_and_mouse_input_count_as_present(self):
        """2026-10-02 review: in wake-word mode he often works at the desk in
        silence; a print callout said out loud there must not also buzz his
        phone. Typing within PHONE_PING_AWAY_MIN = here."""
        bc = self._awake_bc()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(pp, "_live_input_idle_s", return_value=20.0):
            idle = pp._live_owner_idle_s()
        self.assertAlmostEqual(idle, 20.0, delta=1.0)
        # The turn clock still wins when it is the more recent one.
        bc._last_owner_turn_at = [pp.time.monotonic() - 5.0]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(pp, "_live_input_idle_s", return_value=900.0):
            self.assertLess(pp._live_owner_idle_s(), 60.0)

    def test_input_only_counts_while_jarvis_says_it_out_loud(self):
        """Asleep, in standby (the standby loop voices only timers and
        promises) or with his voice muted, JARVIS did NOT say it at the desk:
        typing there is no reason to hold the ping."""
        for kw in ({"sleep": True}, {"standby": True}, {"muted": True}):
            with self.subTest(**kw):
                bc = self._awake_bc(**kw)
                with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
                     mock.patch.object(pp, "_live_input_idle_s",
                                       return_value=20.0):
                    idle = pp._live_owner_idle_s()
                self.assertGreater(idle, 1700.0)

    def test_input_idle_probe_off_windows(self):
        with mock.patch.object(pp.sys, "platform", "linux"):
            self.assertIsNone(pp._live_input_idle_s())

    def test_overnight_flag_is_asleep(self):
        d = tempfile.mkdtemp(prefix="jarvis_pp_flag_")
        self.addCleanup(shutil.rmtree, d, True)
        flag = os.path.join(d, ".overnight_active")
        bc = types.ModuleType("bobert_companion")
        bc.OVERNIGHT_FLAG_FILE = flag
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertFalse(pp._live_asleep())              # no flag
            with open(flag, "w", encoding="utf-8") as f:
                f.write(str(pp.time.time() + 3600))
            self.assertTrue(pp._live_asleep())               # goodnight
            with open(flag, "w", encoding="utf-8") as f:
                f.write(str(pp.time.time() - 60))
            self.assertFalse(pp._live_asleep())              # expired
            with open(flag, "w", encoding="utf-8") as f:
                f.write("garbage")
            self.assertFalse(pp._live_asleep())
        with mock.patch.dict(sys.modules):
            sys.modules.pop("bobert_companion", None)
            self.assertFalse(pp._live_asleep())

    def test_focus_reads_either_focus_mode(self):
        bc = types.ModuleType("bobert_companion")
        bc.focus_mode_active = lambda: True
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertTrue(pp._live_focus_active())
        dnd = types.ModuleType("skill_dnd_focus_mode")
        dnd.is_focus_mode_active = lambda: True
        with mock.patch.dict(sys.modules, {"skill_dnd_focus_mode": dnd}):
            sys.modules.pop("bobert_companion", None)
            self.assertTrue(pp._live_focus_active())

    def test_printer_now_line(self):
        import threading
        mod = types.ModuleType("skill_bambu_monitor")
        mod._state_lock = threading.Lock()
        mod._state = {"gcode_state": "RUNNING", "mc_percent": 67,
                      "filename": "benchy.3mf"}
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": mod}):
            self.assertEqual(pp._live_printer_now(),
                             "Printer: printing 67% (benchy).")
            mod._state["gcode_state"] = "IDLE"
            self.assertEqual(pp._live_printer_now(), "")


class ParserTests(unittest.TestCase):
    def test_parse_hhmm(self):
        self.assertEqual(pp.parse_hhmm("23:00", "07:00"), 23 * 60)
        self.assertEqual(pp.parse_hhmm("7", "00:00"), 7 * 60)
        self.assertEqual(pp.parse_hhmm("07.30", "00:00"), 7 * 60 + 30)
        self.assertEqual(pp.parse_hhmm(22, "00:00"), 22 * 60)
        self.assertEqual(pp.parse_hhmm("25:00", "07:00"), 7 * 60)
        self.assertEqual(pp.parse_hhmm(True, "06:15"), 6 * 60 + 15)


class SingletonTests(unittest.TestCase):
    def setUp(self):
        pp._reset_for_tests()
        self.addCleanup(pp._reset_for_tests)

    def test_attach_bridge_and_module_ping(self):
        bridge = FakeBridge()
        p = pp.attach_bridge(bridge.send, bridge.configured)
        self.assertIs(p, pp.get_pinger())
        p.cfg = lambda n: pp.DEFAULTS.get(n)
        p.blocked = lambda: ""
        p.focus_active = lambda: False
        p.owner_idle_s = lambda: 9999.0
        p.wall_now = lambda: dt.datetime(2026, 10, 2, 12, 0)
        p._state_path = None
        p.spawn = lambda job: (job(), True)[1]
        p.log = lambda s: None
        self.assertEqual(pp.ping("robot", "Stuck under the desk."), pp.QUEUED)
        self.assertEqual(bridge.sent[0]["text"], "Stuck under the desk.")
        st = pp.status()
        self.assertTrue(st["configured"])
        self.assertEqual(st["sent_last_hour"], 1)
        self.assertEqual(st["last"]["cat"], "robot")

    def test_start_watcher_needs_an_active_pinger(self):
        p = pp.get_pinger()
        p.log = lambda s: None
        self.assertFalse(pp.start_watcher())       # no bridge attached
        bridge = FakeBridge()
        pp.attach_bridge(bridge.send, bridge.configured)
        p.cfg = lambda n: pp.DEFAULTS.get(n)
        p.blocked = lambda: ""
        started = []
        with mock.patch.object(pp.threading.Thread, "start",
                               lambda self: started.append(self.name)):
            self.assertTrue(pp.start_watcher())
        self.assertEqual(started, ["phone-ping-watcher"])


if __name__ == "__main__":
    unittest.main()
