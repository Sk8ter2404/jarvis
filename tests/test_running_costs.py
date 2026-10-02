"""Tests for core.running_costs + the running_costs action.

"How much does it cost to run you" gets a real breakdown: live GPU power from
nvidia-smi (summed over every card) plus a CPU estimate, times the hours JARVIS
has run today / this month, times ELECTRICITY_RATE_PER_KWH; this session's
Claude spend from core.llm_client's tally priced by core.model_catalog; and a
one-line verdict. Fakes only: subprocess.run, psutil, memory.py's session index
and the usage tally are all stubbed, so nothing spawns and nothing touches the
network or the live data files.
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
import time
import types
import unittest
from unittest import mock

import core.config as cfg
import core.model_catalog as mc
import core.running_costs as rc
from core.failure_markers import FAILURE_MARKERS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _local(y, mo, d, h=0, mi=0):
    return time.mktime((y, mo, d, h, mi, 0, 0, 0, -1))


class _Run:
    """A fake subprocess.run: records its call, then returns or raises."""

    def __init__(self, stdout="", returncode=0, exc=None):
        self.stdout, self.returncode, self.exc = stdout, returncode, exc
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return types.SimpleNamespace(stdout=self.stdout,
                                     returncode=self.returncode)


class ParsePowerDrawTests(unittest.TestCase):
    def test_single_gpu(self):
        self.assertAlmostEqual(rc.parse_power_draw("85.32\n"), 85.32)

    def test_multi_gpu_lines_are_summed(self):
        self.assertAlmostEqual(rc.parse_power_draw("120.50\n 30.25 \n9.25\n"),
                               160.0)

    def test_unreadable_cards_are_skipped(self):
        self.assertAlmostEqual(
            rc.parse_power_draw("[N/A]\n70.0\n[Not Supported]\n"), 70.0)

    def test_no_reading_is_none(self):
        for text in (None, "", "\n", "[N/A]\n[Not Supported]",
                     "nan\n-5\ninf"):
            self.assertIsNone(rc.parse_power_draw(text), repr(text))


class PowerQueryTests(unittest.TestCase):
    def test_multi_gpu_query_sums_every_card(self):
        fake = _Run(stdout="210.40\n95.60\n")
        with mock.patch.object(rc.subprocess, "run", fake):
            self.assertAlmostEqual(rc.gpu_watts(), 306.0)
        args, kwargs = fake.calls[0]
        self.assertEqual(args, ["nvidia-smi", "--query-gpu=power.draw",
                                "--format=csv,noheader,nounits"])
        self.assertLessEqual(kwargs["timeout"], 5)
        self.assertEqual(kwargs["creationflags"], rc._NO_WINDOW)

    def test_missing_binary_is_none(self):
        fake = _Run(exc=FileNotFoundError("nvidia-smi"))
        with mock.patch.object(rc.subprocess, "run", fake):
            self.assertIsNone(rc.gpu_watts())

    def test_timeout_is_none(self):
        fake = _Run(exc=subprocess.TimeoutExpired("nvidia-smi", 2.0))
        with mock.patch.object(rc.subprocess, "run", fake):
            self.assertIsNone(rc.gpu_watts())

    def test_error_exit_is_none(self):
        fake = _Run(stdout="NVIDIA-SMI has failed", returncode=9)
        with mock.patch.object(rc.subprocess, "run", fake):
            self.assertIsNone(rc.gpu_watts())

    def test_windows_spawn_hides_the_console(self):
        def _reload():
            importlib.reload(rc)
        self.addCleanup(_reload)
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(subprocess, "CREATE_NO_WINDOW", 0x08000000,
                               create=True):
            importlib.reload(rc)
            self.assertEqual(rc._NO_WINDOW, 0x08000000)


class CpuEstimateTests(unittest.TestCase):
    def test_unknown_utilisation_is_the_idle_floor(self):
        self.assertEqual(rc.cpu_watts(None), rc.CPU_IDLE_WATTS)

    def test_linear_between_idle_and_full_and_clamped(self):
        self.assertEqual(rc.cpu_watts(0), rc.CPU_IDLE_WATTS)
        self.assertEqual(rc.cpu_watts(100), rc.CPU_FULL_WATTS)
        self.assertEqual(rc.cpu_watts(250), rc.CPU_FULL_WATTS)
        self.assertAlmostEqual(
            rc.cpu_watts(50), (rc.CPU_IDLE_WATTS + rc.CPU_FULL_WATTS) / 2)


class RateSettingTests(unittest.TestCase):
    def test_config_default_is_a_float_014(self):
        self.assertEqual(cfg.ELECTRICITY_RATE_PER_KWH, 0.14)
        # float, so _apply_user_settings keeps a saved 0.3 as 0.3
        self.assertIsInstance(cfg.ELECTRICITY_RATE_PER_KWH, float)

    def test_user_settings_override_reaches_the_rate(self):
        import json
        orig = cfg.ELECTRICITY_RATE_PER_KWH
        self.addCleanup(setattr, cfg, "ELECTRICITY_RATE_PER_KWH", orig)
        m = mock.mock_open(read_data=json.dumps(
            {"ELECTRICITY_RATE_PER_KWH": "0.31"}))
        with mock.patch("core.config.os.path.exists", return_value=True), \
             mock.patch("core.config.open", m, create=True):
            cfg._apply_user_settings()
        self.assertEqual(rc.electricity_rate(), 0.31)

    def test_rate_is_read_live_and_bad_values_fall_back(self):
        for val, want in ((0.28, 0.28), (0.0, 0.0), (-1.0, 0.14),
                          ("cheap", 0.14), (float("nan"), 0.14)):
            with mock.patch.object(cfg, "ELECTRICITY_RATE_PER_KWH", val):
                self.assertEqual(rc.electricity_rate(), want, repr(val))

    def test_rate_scales_the_spoken_electricity_cost(self):
        base = dict(gpu_w=500.0, cpu_w=500.0, session_h=1.0, today_h=10.0,
                    month_h=100.0, cloud_usd=0.0, cloud_calls=0)
        # 1 kW x 10 h = 10 kWh today, 100 kWh this month
        low = rc.compose(rate=0.14, **base)
        high = rc.compose(rate=0.28, **base)
        self.assertIn("14 cents a kilowatt-hour", low)
        self.assertIn("about $1.40 for 10 hours today", low)
        self.assertIn("about $14.00 for 100 hours this month", low)
        self.assertIn("28 cents a kilowatt-hour", high)
        self.assertIn("about $2.80 for 10 hours today", high)


class RunningHoursTests(unittest.TestCase):
    def test_today_and_month_from_live_session_plus_persisted_spans(self):
        now = _local(2026, 10, 15, 12)
        start = now - 2 * 3600
        spans = [
            (_local(2026, 10, 14, 10), _local(2026, 10, 14, 13)),   # 3 h
            # crosses into the month at midnight: only 2 h count ...
            (_local(2026, 9, 30, 22), _local(2026, 10, 1, 2)),
            # ... and overlaps this one, which must merge (03:00-05:00 new)
            (_local(2026, 10, 1, 1), _local(2026, 10, 1, 5)),
            # the live session's own checkpoint row: not counted twice
            (start, now - 600),
            (_local(2026, 8, 3, 9), _local(2026, 8, 3, 17)),        # old
        ]
        session_h, today_h, month_h = rc.running_hours(now, start, spans)
        self.assertAlmostEqual(session_h, 2.0)
        self.assertAlmostEqual(today_h, 2.0)
        self.assertAlmostEqual(month_h, 2.0 + 3.0 + 5.0)

    def test_session_running_over_midnight_splits_at_the_day(self):
        now = _local(2026, 10, 15, 1)
        session_h, today_h, month_h = rc.running_hours(
            now, now - 3 * 3600, [])
        self.assertAlmostEqual(session_h, 3.0)
        self.assertAlmostEqual(today_h, 1.0)
        self.assertAlmostEqual(month_h, 3.0)

    def test_no_start_and_no_history_is_zero(self):
        self.assertEqual(rc.running_hours(time.time(), None, []),
                         (0.0, 0.0, 0.0))

    def test_persisted_spans_come_from_the_session_summary_index(self):
        entries = [
            {"iso_start": "2026-10-14T10:00:00", "ts": _local(2026, 10, 14, 11)},
            {"iso_start": "2026-10-13T08:00:00",
             "iso_end": "2026-10-13T08:30:00"},                  # no ts
            {"date": "2026-10-01", "summary": "legacy, no times"},
            {"iso_start": "2026-10-12T09:00:00", "ts": "garbage",
             "iso_end": "not a time"},
            {"iso_start": "2026-10-12T09:00:00",
             "ts": _local(2026, 10, 12, 8)},                     # end < start
            "not a dict",
        ]
        fake = types.SimpleNamespace(_load_sessions_file=lambda: entries)
        with mock.patch.dict(sys.modules, {"memory": fake}):
            spans = rc.persisted_session_spans()
        self.assertEqual(spans, [
            (_local(2026, 10, 14, 10), _local(2026, 10, 14, 11)),
            (_local(2026, 10, 13, 8), _local(2026, 10, 13, 8, 30)),
        ])

    def test_unreadable_index_is_no_history(self):
        def _boom():
            raise OSError("disk")
        for fake in (types.SimpleNamespace(),
                     types.SimpleNamespace(_load_sessions_file=_boom)):
            with mock.patch.dict(sys.modules, {"memory": fake}):
                self.assertEqual(rc.persisted_session_spans(), [])

    def test_session_start_is_the_monoliths_when_loaded(self):
        fake = types.SimpleNamespace(_session_start_time=1234.5)
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            self.assertEqual(rc._session_start(), 1234.5)


class CloudCostTests(unittest.TestCase):
    def test_sums_every_model_with_cache_pricing(self):
        usage = {
            "claude-sonnet-5-5": {"calls": 4, "input": 100_000,
                                  "output": 10_000, "cache_read": 1_000_000,
                                  "cache_write": 0},
            "claude-haiku-4-5": {"calls": 2, "input": 0, "output": 0,
                                 "cache_read": 0, "cache_write": 1_000_000},
            "claude-mystery-9": {"calls": 3, "input": 5_000, "output": 50,
                                 "cache_read": 0, "cache_write": 0},
        }
        son, hai = mc.by_id("claude-sonnet-5-5"), mc.by_id("claude-haiku-4-5")
        want = ((100_000 + 1_000_000 * 0.10) * son.in_price
                + 10_000 * son.out_price
                + 1_000_000 * 1.25 * hai.in_price) / 1_000_000
        usd, priced, unpriced = rc.cloud_cost(usage)
        self.assertAlmostEqual(usd, want)
        self.assertEqual((priced, unpriced), (6, 3))

    def test_empty_tally_is_free(self):
        self.assertEqual(rc.cloud_cost({}), (0.0, 0, 0))

    def test_month_source_is_the_persisted_tally(self):
        import core.llm_usage as lu
        with mock.patch.object(lu, "month_usage", return_value={"x": 1}):
            self.assertEqual(rc._month_usage(), {"x": 1})
        with mock.patch.object(lu, "month_usage", side_effect=OSError):
            self.assertIsNone(rc._month_usage())

    def test_reads_the_llm_client_tally(self):
        import core.llm_client as llm
        with mock.patch.object(llm, "session_usage", {
                "claude-haiku-4-5": {"calls": 1, "input": 1_000_000,
                                     "output": 0, "cache_read": 0,
                                     "cache_write": 0}}):
            usd, priced, _ = rc.cloud_cost(rc._session_usage())
        self.assertAlmostEqual(usd, mc.by_id("claude-haiku-4-5").in_price)
        self.assertEqual(priced, 1)


class SpokenTextTests(unittest.TestCase):
    CASES = (
        dict(gpu_w=212.4, cpu_w=48.3, rate=0.14, session_h=2.5, today_h=3.2,
             month_h=74.6, cloud_usd=0.234, cloud_calls=17),
        dict(gpu_w=None, cpu_w=20.0, rate=0.14, session_h=0.01, today_h=0.01,
             month_h=0.01, cloud_usd=0.0, cloud_calls=0),
        dict(gpu_w=None, cpu_w=62.0, rate=0.3, session_h=5.0, today_h=5.0,
             month_h=300.0, cloud_usd=0.0, cloud_calls=0),
        dict(gpu_w=350.0, cpu_w=125.0, rate=1.25, session_h=30.0,
             today_h=12.0, month_h=600.0, cloud_usd=123.456, cloud_calls=900,
             unpriced_calls=4),
        dict(gpu_w=0.0, cpu_w=20.0, rate=0.0, session_h=1.0, today_h=1.0,
             month_h=1.0, cloud_usd=0.0001, cloud_calls=1),
        # with a persisted month tally
        dict(gpu_w=212.4, cpu_w=48.3, rate=0.14, session_h=2.5, today_h=3.2,
             month_h=74.6, cloud_usd=0.234, cloud_calls=17,
             month_cloud_usd=4.1049, month_calls=312),
        dict(gpu_w=None, cpu_w=20.0, rate=0.14, session_h=0.01, today_h=0.01,
             month_h=0.5, cloud_usd=0.0, cloud_calls=0,
             month_cloud_usd=1234.567, month_calls=1, month_unpriced=2),
        dict(gpu_w=None, cpu_w=20.0, rate=0.0, session_h=0.01, today_h=0.01,
             month_h=0.01, cloud_usd=0.0, cloud_calls=0,
             month_cloud_usd=0.0, month_calls=0),
    )

    def test_two_or_three_plain_sentences_without_markdown(self):
        for case in self.CASES:
            text = rc.compose(**case)
            self.assertNotRegex(text, r"[*_#`|\[\]~>\n]", case)
            self.assertNotRegex(text, r"(^|\s)-\s", case)
            ends = re.findall(r"[.!?](?=\s|$)", text)
            self.assertIn(len(ends), (2, 3), f"{case}: {text}")

    def test_numbers_are_rounded(self):
        for case in self.CASES:
            text = rc.compose(**case)
            self.assertNotRegex(text, r"\d\.\d{3}", text)
            self.assertNotRegex(text, r"\$\d+\.\d(?!\d)", text)

    def test_never_reads_as_a_failure(self):
        # running_costs is spoken verbatim; a failure-marker word would route
        # the answer to the failure follow-up instead of the speaker.
        for case in self.CASES:
            low = rc.compose(**case).lower()
            hits = [m for m in FAILURE_MARKERS if m in low]
            self.assertEqual(hits, [], low)

    def test_full_breakdown(self):
        text = rc.compose(**self.CASES[0])
        self.assertIn("about 212 watts", text)
        self.assertIn("CPU roughly 48", text)
        self.assertIn("estimate", text)
        self.assertIn("for 3.2 hours today", text)
        self.assertIn("for 75 hours this month", text)
        self.assertIn("about 23 cents across 17 calls", text)
        self.assertIn("check your credits", text)
        self.assertTrue(text.endswith("mostly the cloud, sir."), text)

    def test_no_gpu_reading_is_said_plainly(self):
        text = rc.compose(**self.CASES[2])
        self.assertIn("no GPU power reading", text)
        self.assertIn("CPU alone is roughly 62 watts", text)
        self.assertIn("no Claude calls yet", text)
        self.assertIn("all of it electricity", text)

    def test_unpriced_calls_are_counted_and_named(self):
        text = rc.compose(**self.CASES[3])
        self.assertIn("across 904 calls, 4 of them on a model I have no price",
                      text)
        self.assertIn("about $123", text)

    def test_without_a_month_tally_the_verdict_covers_the_session(self):
        text = rc.compose(**self.CASES[0])
        self.assertIn("no month-to-date cloud tally yet", text)
        self.assertNotIn("at list prices", text)
        self.assertIn("Verdict: about 33 cents this session so far", text)

    def test_month_tally_is_reported_and_drives_the_verdict(self):
        text = rc.compose(**self.CASES[5])
        self.assertIn("this session come to about 23 cents across 17 calls, "
                      "and about $4.10 across 312 calls this month at list "
                      "prices.", text)
        self.assertNotIn("no month-to-date", text)
        # 0.2607 kW x 74.6 h x $0.14 = $2.72 of power + $4.10 of cloud
        self.assertIn("Verdict: about $6.83 this month so far, mostly the "
                      "cloud, sir.", text)

    def test_month_tally_with_no_calls_this_session(self):
        text = rc.compose(**self.CASES[6])
        self.assertIn("No Claude calls yet this session, and about $1,235 "
                      "across 3 calls this month at list prices, 2 of them "
                      "on a model I have no price for.", text)
        self.assertTrue(text.endswith("this month so far, mostly the cloud, "
                                      "sir."), text)
        text = rc.compose(**self.CASES[7])
        self.assertIn("No Claude calls yet this session, and none so far this "
                      "month.", text)
        self.assertTrue(text.endswith(
            "Verdict: next to nothing so far this month, sir."), text)


class ReportTests(unittest.TestCase):
    def test_report_wires_the_live_readings(self):
        now = _local(2026, 10, 15, 12)
        usage = {"claude-sonnet-5-5": {"calls": 3, "input": 30_000,
                                       "output": 3_000, "cache_read": 0,
                                       "cache_write": 0}}
        with mock.patch.object(rc.subprocess, "run",
                               _Run(stdout="100.0\n50.0\n")), \
             mock.patch.object(rc, "_read_cpu_percent", return_value=40.0), \
             mock.patch.object(rc, "_session_start",
                               return_value=now - 2 * 3600), \
             mock.patch.object(rc, "persisted_session_spans",
                               return_value=[]), \
             mock.patch.object(rc, "_session_usage", return_value=usage), \
             mock.patch.object(rc, "_month_usage", return_value=None), \
             mock.patch.object(cfg, "ELECTRICITY_RATE_PER_KWH", 0.14):
            text = rc.report(now=now)
        self.assertIn("GPU is drawing about 150 watts", text)
        self.assertIn("CPU roughly 62", text)
        self.assertIn("for 2 hours today", text)
        self.assertIn("across 3 calls", text)
        self.assertIn("no month-to-date cloud tally yet", text)

    def test_report_prices_the_persisted_month_tally(self):
        now = _local(2026, 10, 15, 12)
        session = {"claude-haiku-4-5": {"calls": 2, "input": 1_000_000,
                                        "output": 0, "cache_read": 0,
                                        "cache_write": 0}}
        month = {"claude-haiku-4-5": {"calls": 40, "input": 3_000_000,
                                      "output": 0, "cache_read": 0,
                                      "cache_write": 0},
                 "claude-mystery-9": {"calls": 1, "input": 5, "output": 5,
                                      "cache_read": 0, "cache_write": 0}}
        with mock.patch.object(rc.subprocess, "run", _Run(stdout="0\n")), \
             mock.patch.object(rc, "_read_cpu_percent", return_value=0.0), \
             mock.patch.object(rc, "_session_start",
                               return_value=now - 3600), \
             mock.patch.object(rc, "persisted_session_spans",
                               return_value=[]), \
             mock.patch.object(rc, "_session_usage", return_value=session), \
             mock.patch.object(rc, "_month_usage", return_value=month), \
             mock.patch.object(cfg, "ELECTRICITY_RATE_PER_KWH", 0.14):
            text = rc.report(now=now)
        price = mc.by_id("claude-haiku-4-5").in_price
        self.assertIn(f"this session come to about ${price:.2f} "
                      f"across 2 calls", text)
        self.assertIn(f"about ${3 * price:.2f} across 41 calls this month at "
                      f"list prices, 1 of them on a model I have no price "
                      f"for.", text)
        self.assertIn("this month so far, mostly the cloud, sir.", text)

    def test_missing_nvidia_smi_still_answers(self):
        with mock.patch.object(rc.subprocess, "run",
                               _Run(exc=FileNotFoundError("nvidia-smi"))), \
             mock.patch.object(rc, "_read_cpu_percent", return_value=None), \
             mock.patch.object(rc, "_session_start", return_value=None), \
             mock.patch.object(rc, "persisted_session_spans",
                               return_value=[]), \
             mock.patch.object(rc, "_session_usage", return_value={}), \
             mock.patch.object(rc, "_month_usage", return_value=None):
            text = rc.report()
        self.assertIn("no GPU power reading", text)
        self.assertTrue(text.endswith("sir."), text)

    def test_action_returns_the_report(self):
        import core.actions as A
        with mock.patch.object(rc, "report", return_value="sentinel") as r:
            self.assertEqual(A._act_running_costs(""), "sentinel")
        r.assert_called_once_with()
        self.assertIn("_act_running_costs", A.__all__)


class MonolithRegistrationTests(unittest.TestCase):
    """Static read of bobert_companion.py (the monolith is not imported)."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(_ROOT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            cls.tree = ast.parse(fh.read())

    def _top_level(self, name):
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets
                           if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name):
                targets = [node.target.id]
            else:
                continue
            if name in targets:
                return node.value
        self.fail(f"{name} not found in bobert_companion.py")

    def test_registered_in_actions(self):
        d = self._top_level("ACTIONS")
        handlers = {k.value: getattr(v, "id", None)
                    for k, v in zip(d.keys, d.values)
                    if isinstance(k, ast.Constant)}
        self.assertEqual(handlers.get("running_costs"), "_act_running_costs")

    def test_spoken_verbatim_not_informative(self):
        def names(node):
            return {e.value for e in node.elts if isinstance(e, ast.Constant)}
        self.assertIn("running_costs",
                      names(self._top_level("SPEAK_RESULT_VERBATIM_ACTIONS")))
        self.assertNotIn("running_costs",
                         names(self._top_level("INFORMATIVE_ACTIONS")))


if __name__ == "__main__":
    unittest.main()
