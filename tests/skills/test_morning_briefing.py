"""Logic tests for skills/morning_briefing.py.

Covers the pure helpers (ordinal, pending-task count, weather phrase, bed-time
remark, Outlook summary timeout wrapper), the full _build_briefing assembly
with every external source mocked, the same-day fired flag, the chain entry
point's TOCTOU/suppression behaviour, the manual action, the speech-queue
enqueue (proactive_announce + atomic-write fallback), the HUD card pop, the
lazy-imported source helpers (news / umbrella / robot / outlook summary), and
the _fire_briefing flag-then-build path. No network, no COM, no real
pending_speech.json writes — every external module is faked in setUp and
removed on exit.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Install fake modules into sys.modules for the duration of the block,
    restoring prior state (including absence) on exit — the module-isolation
    contract from tests/skills/test_self_diagnostic.py. morning_briefing's
    helpers do lazy ``from . import X`` then ``import X``; the skill is loaded
    under the name ``skill_morning_briefing`` (NOT a package), so the relative
    import fails and the absolute ``import X`` is what resolves. We therefore
    register both the bare name and the dotted package-attribute name so either
    resolution path finds the fake."""
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _fake_source(**attrs):
    """A throwaway module object with the given callables/attributes set."""
    m = types.ModuleType("fake_source")
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class MorningBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_briefing")

    # ── _ordinal (pure) ──────────────────────────────────────────────────
    def test_ordinal(self):
        o = self.mod._ordinal
        self.assertEqual(o(1), "1st")
        self.assertEqual(o(2), "2nd")
        self.assertEqual(o(3), "3rd")
        self.assertEqual(o(4), "4th")
        self.assertEqual(o(11), "11th")   # teens are all "th"
        self.assertEqual(o(12), "12th")
        self.assertEqual(o(21), "21st")
        self.assertEqual(o(22), "22nd")

    # ── _count_pending_tasks ─────────────────────────────────────────────
    def test_count_pending_tasks(self):
        todo = "- [ ] one\n- [x] done\n- [ ] two\nnot a task\n- [ ] three\n"
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=todo)):
            self.assertEqual(self.mod._count_pending_tasks(), 3)

    def test_count_pending_tasks_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._count_pending_tasks(), 0)

    # ── _fetch_weather (briefing_sources) ────────────────────────────────
    def test_fetch_weather_phrase(self):
        fake_bs = mock.MagicMock()
        # 18 C is stored; the briefing must SPEAK Fahrenheit (18 C → 64 F),
        # matching weather_briefing's standalone report — not the raw Celsius.
        fake_bs.get_weather_data.return_value = {"temp_c": 18, "desc": "Overcast", "source": "wttr"}
        import sys
        # _fetch_weather does `from . import briefing_sources` first; inject a
        # package-style module name so that import resolves to our fake.
        with mock.patch.dict(sys.modules, {"skill_morning_briefing.briefing_sources": fake_bs,
                                           "briefing_sources": fake_bs}):
            out = self.mod._fetch_weather()
        self.assertIn("64 degrees and overcast", out)
        # Guard against regression: the raw Celsius value must never be spoken.
        self.assertNotIn("18 degrees", out)

    def test_fetch_weather_degraded(self):
        fake_bs = mock.MagicMock()
        fake_bs.get_weather_data.return_value = None
        import sys
        with mock.patch.dict(sys.modules, {"skill_morning_briefing.briefing_sources": fake_bs,
                                           "briefing_sources": fake_bs}):
            self.assertEqual(self.mod._fetch_weather(), "")

    def test_fetch_weather_speaks_fahrenheit_not_celsius(self):
        # Regression guard for the morning_briefing/weather_briefing unit
        # disagreement: briefing_sources stores Celsius, but the briefing must
        # SPEAK Fahrenheit (store-Celsius / speak-Fahrenheit). A known 24 C
        # reading — the exact value that once leaked through unconverted —
        # must voice as 75 F (24*9/5+32 = 75.2 → 75), never the raw "24".
        fake_bs = mock.MagicMock()
        fake_bs.get_weather_data.return_value = {"temp_c": 24, "desc": "clear", "source": "wttr"}
        import sys
        with mock.patch.dict(sys.modules, {"skill_morning_briefing.briefing_sources": fake_bs,
                                           "briefing_sources": fake_bs}):
            out = self.mod._fetch_weather()
        self.assertIn("75 degrees", out)
        self.assertNotIn("24 degrees", out)

    # ── _bed_remark ──────────────────────────────────────────────────────
    def test_bed_remark_late_night(self):
        # mtime at 02:30 local → triggers the dry "pace yourself" remark.
        late = time.mktime(time.struct_time((2026, 6, 1, 2, 30, 0, 0, 152, -1)))
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir", return_value=["s.log"]), \
             mock.patch.object(self.mod.os.path, "getmtime", return_value=late):
            out = self.mod._bed_remark()
        self.assertIn("2:30 AM", out)
        self.assertIn("pace yourself", out)

    def test_bed_remark_daytime_silent(self):
        noon = time.mktime(time.struct_time((2026, 6, 1, 12, 0, 0, 0, 152, -1)))
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir", return_value=["s.log"]), \
             mock.patch.object(self.mod.os.path, "getmtime", return_value=noon):
            self.assertEqual(self.mod._bed_remark(), "")

    # ── _outlook_summary timeout wrapper ─────────────────────────────────
    def test_outlook_summary_returns_blocking_result(self):
        with mock.patch.object(self.mod, "_outlook_summary_blocking",
                               return_value="one unread email"):
            self.assertEqual(self.mod._outlook_summary(), "one unread email")

    def test_outlook_summary_swallows_blocking_error(self):
        with mock.patch.object(self.mod, "_outlook_summary_blocking",
                               side_effect=RuntimeError("graph down")):
            self.assertEqual(self.mod._outlook_summary(), "")

    # ── _build_briefing assembly ─────────────────────────────────────────
    def test_build_briefing_full(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value="18 degrees and clear in your area"), \
             mock.patch.object(self.mod, "_count_pending_tasks", return_value=3), \
             mock.patch.object(self.mod, "_bed_remark", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value="Today's headlines, sir. X."), \
             mock.patch.object(self.mod, "_fetch_umbrella_alert", return_value="bring an umbrella"), \
             mock.patch.object(self.mod, "_outlook_summary", return_value="one unread email"), \
             mock.patch.object(self.mod, "_fetch_robot_volunteer", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("Good morning, sir", out)
        self.assertIn("18 degrees and clear", out)
        self.assertIn("3 tasks queued", out)
        self.assertIn("From Outlook: one unread email", out)
        self.assertIn("bring an umbrella", out)
        # News present → leading briefing-intent tag.
        self.assertTrue(out.startswith("[intent:briefing]"))

    def test_build_briefing_empty_queue_phrase(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value=""), \
             mock.patch.object(self.mod, "_count_pending_tasks", return_value=0), \
             mock.patch.object(self.mod, "_bed_remark", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value=""), \
             mock.patch.object(self.mod, "_fetch_umbrella_alert", return_value=""), \
             mock.patch.object(self.mod, "_outlook_summary", return_value=""), \
             mock.patch.object(self.mod, "_fetch_robot_volunteer", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("mercifully empty", out)
        # No news → no intent tag.
        self.assertFalse(out.startswith("[intent:briefing]"))

    def test_build_briefing_single_task(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value=""), \
             mock.patch.object(self.mod, "_count_pending_tasks", return_value=1), \
             mock.patch.object(self.mod, "_bed_remark", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value=""), \
             mock.patch.object(self.mod, "_fetch_umbrella_alert", return_value=""), \
             mock.patch.object(self.mod, "_outlook_summary", return_value=""), \
             mock.patch.object(self.mod, "_fetch_robot_volunteer", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("one task queued", out)

    # ── same-day fired flag ──────────────────────────────────────────────
    def test_already_fired_today_true(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=today)):
            self.assertTrue(self.mod._briefing_already_fired_today())

    def test_already_fired_today_stale(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="1999-01-01")):
            self.assertFalse(self.mod._briefing_already_fired_today())

    def test_already_fired_today_no_flag(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertFalse(self.mod._briefing_already_fired_today())

    # ── _fire_from_chain suppression ─────────────────────────────────────
    def test_fire_from_chain_suppressed_when_already_fired(self):
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_fire_briefing") as fire:
            self.mod._fire_from_chain("test")
        fire.assert_not_called()

    def test_fire_from_chain_fires_after_delay(self):
        # First check (pre) and second check (post-delay) both False → fires.
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=False), \
             mock.patch.object(self.mod.time, "sleep") as slp, \
             mock.patch.object(self.mod, "_fire_briefing") as fire:
            self.mod._fire_from_chain("chain pick")
        slp.assert_called_once()  # the pre-fire delay
        fire.assert_called_once()

    # ── morning_briefing action ──────────────────────────────────────────
    def test_action_builds_and_marks(self):
        mod, actions = load_skill_isolated("morning_briefing")
        with mock.patch.object(mod, "_build_briefing", return_value="Good morning, sir."), \
             mock.patch.object(mod, "_show_card_safe"), \
             mock.patch.object(mod, "_mark_briefing_fired_today") as mark:
            out = actions["morning_briefing"]("")
        self.assertEqual(out, "Good morning, sir.")
        mark.assert_called_once()

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("morning_briefing")
        with mock.patch.object(mod, "_build_briefing", side_effect=RuntimeError("boom")):
            out = actions["morning_briefing"]("")
        self.assertIn("failed", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# _fetch_weather edge branches (briefing_sources faked)
# ─────────────────────────────────────────────────────────────────────────
class FetchWeatherTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_briefing")

    def _with_bs(self, bs):
        return inject_modules(**{"skill_morning_briefing.briefing_sources": bs,
                                 "briefing_sources": bs})

    def test_weather_no_desc_uses_outside_phrase(self):
        # 7 C stored → 45 F spoken (7*9/5+32 = 44.6 → 45).
        bs = _fake_source(get_weather_data=lambda: {"temp_c": 7, "desc": "", "source": "wttr"})
        with self._with_bs(bs):
            self.assertEqual(self.mod._fetch_weather(), "45 degrees outside")

    def test_weather_stale_cache_adds_suffix(self):
        # 4 C stored → 39 F spoken (4*9/5+32 = 39.2 → 39); cached suffix retained.
        bs = _fake_source(get_weather_data=lambda: {
            "temp_c": 4, "desc": "fog", "source": "cache", "stale": True})
        with self._with_bs(bs):
            out = self.mod._fetch_weather()
        self.assertIn("39 degrees and fog in your area", out)
        self.assertIn("(cached)", out)

    def test_weather_bad_temp_returns_blank(self):
        bs = _fake_source(get_weather_data=lambda: {"temp_c": "not-a-number"})
        with self._with_bs(bs):
            self.assertEqual(self.mod._fetch_weather(), "")

    def test_weather_source_import_unavailable_returns_blank(self):
        # Both the package-relative and absolute import fail → the helper logs
        # and returns "" without raising (and crucially never reaches the real
        # briefing_sources, which would hit the network).
        with inject_modules(**{"skill_morning_briefing.briefing_sources": None,
                               "briefing_sources": None}), \
             mock.patch("builtins.__import__",
                        side_effect=_block_import("briefing_sources")):
            self.assertEqual(self.mod._fetch_weather(), "")

    def test_count_pending_tasks_read_error_returns_zero(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._count_pending_tasks(), 0)


# ─────────────────────────────────────────────────────────────────────────
# _enqueue_speech: proactive_announce path + atomic-write fallback
# ─────────────────────────────────────────────────────────────────────────
class EnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_briefing")
        self.tmp = tempfile.mkdtemp(prefix="mbrief_speech_")
        self.addCleanup(self._cleanup)
        self.queue = os.path.join(self.tmp, "pending_speech.json")
        self.mod._SPEECH_QUEUE = self.queue

    def _cleanup(self):
        for f in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, f))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_routes_through_proactive_announce(self):
        calls = []
        bc = _fake_source(proactive_announce=lambda msg, source=None: calls.append((msg, source)))
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("hello sir")
        self.assertEqual(calls, [("hello sir", "morning")])
        # Announcer handled it → no local queue file written.
        self.assertFalse(os.path.exists(self.queue))

    def test_falls_back_to_atomic_write_when_no_announcer(self):
        # bobert_companion present but WITHOUT proactive_announce → local write.
        bc = types.ModuleType("bobert_companion")
        import json
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("queued line")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "queued line")

    def test_fallback_appends_to_existing_queue(self):
        import json
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.mod._enqueue_speech("new")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_fallback_corrupt_queue_is_replaced(self):
        import json
        with open(self.queue, "w", encoding="utf-8") as f:
            f.write("{garbage")
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.mod._enqueue_speech("fresh")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "fresh")

    def test_fallback_write_failure_is_swallowed(self):
        # Atomic write raises → console fallback, never propagates.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only share")):
            self.mod._enqueue_speech("doomed")   # no raise


# ─────────────────────────────────────────────────────────────────────────
# _show_card_safe
# ─────────────────────────────────────────────────────────────────────────
class ShowCardTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_briefing")

    def test_show_card_calls_hud(self):
        seen = []
        hud = _fake_source(show_card=lambda which: seen.append(which))
        with inject_modules(hud_card=hud):
            self.mod._show_card_safe()
        self.assertEqual(seen, ["morning"])

    def test_show_card_swallows_import_failure(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no hud_card")):
            self.mod._show_card_safe()   # no raise

    def test_show_card_bootstraps_sys_path_when_root_missing(self):
        # With _PROJECT_DIR absent from sys.path, _show_card_safe re-inserts it
        # before importing hud_card (covers the call-time path-bootstrap guard).
        seen = []
        hud = _fake_source(show_card=lambda which: seen.append(which))
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.abspath(p) != os.path.abspath(self.mod._PROJECT_DIR)]
            with inject_modules(hud_card=hud):
                self.mod._show_card_safe()
            self.assertIn(self.mod._PROJECT_DIR, sys.path)   # re-inserted
        finally:
            sys.path[:] = saved
        self.assertEqual(seen, ["morning"])


# ─────────────────────────────────────────────────────────────────────────
# _outlook_summary_blocking (ms_graph faked)
# ─────────────────────────────────────────────────────────────────────────
class OutlookSummaryBlockingTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_briefing")

    def _with_ms(self, ms):
        return inject_modules(**{"skill_morning_briefing.ms_graph": ms,
                                 "ms_graph": ms})

    def test_unread_plural_and_meeting_with_organizer(self):
        ms = _fake_source(
            get_unread_mail_count=lambda: 4,
            get_first_meeting=lambda when: {
                "start": __import__("datetime").datetime(2026, 6, 1, 9, 5),
                "subject": "Budget", "organizer": "Dana Scully <dana@x.invalid>"})
        with self._with_ms(ms):
            out = self.mod._outlook_summary_blocking()
        self.assertIn("4 unread emails", out)
        self.assertIn("9:05 AM", out)
        self.assertIn("with Dana", out)
        self.assertIn("Budget", out)

    def test_single_unread_phrasing(self):
        ms = _fake_source(get_unread_mail_count=lambda: 1,
                          get_first_meeting=lambda when: None)
        with self._with_ms(ms):
            out = self.mod._outlook_summary_blocking()
        self.assertEqual(out, "one unread email")

    def test_meeting_pm_without_organizer(self):
        ms = _fake_source(
            get_unread_mail_count=lambda: 0,
            get_first_meeting=lambda when: {
                "start": __import__("datetime").datetime(2026, 6, 1, 14, 0),
                "subject": "", "organizer": "me"})
        with self._with_ms(ms):
            out = self.mod._outlook_summary_blocking()
        self.assertIn("first meeting is at 2:00 PM", out)
        self.assertNotIn("with", out)   # organizer is "me" → no who-clause

    def test_unread_getter_raises_is_tolerated(self):
        def _boom():
            raise RuntimeError("graph 500")
        ms = _fake_source(get_unread_mail_count=_boom,
                          get_first_meeting=lambda when: None)
        with self._with_ms(ms):
            self.assertEqual(self.mod._outlook_summary_blocking(), "")

    def test_meeting_getter_raises_is_tolerated(self):
        def _boom(when):
            raise RuntimeError("graph 500")
        ms = _fake_source(get_unread_mail_count=lambda: 0, get_first_meeting=_boom)
        with self._with_ms(ms):
            self.assertEqual(self.mod._outlook_summary_blocking(), "")

    def test_ms_graph_unavailable_returns_blank(self):
        # _outlook_summary_blocking imports ms_graph via the import statement
        # (relative then absolute) — block both at the __import__ level so the
        # missing-dependency branch returns "".
        with inject_modules(**{"skill_morning_briefing.ms_graph": None, "ms_graph": None}), \
             mock.patch("builtins.__import__", side_effect=_block_import("ms_graph")):
            self.assertEqual(self.mod._outlook_summary_blocking(), "")

    def test_outlook_summary_timeout_returns_blank(self):
        # Force the future to time out → wrapper returns "".
        from concurrent.futures import TimeoutError as FutureTimeoutError

        class _Future:
            def result(self, timeout=None):
                raise FutureTimeoutError()

        class _Exec:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn):
                return _Future()

        with mock.patch.object(self.mod, "ThreadPoolExecutor", lambda max_workers=1: _Exec()):
            self.assertEqual(self.mod._outlook_summary(), "")

    def test_outlook_summary_executor_unavailable_returns_blank(self):
        with mock.patch.object(self.mod, "ThreadPoolExecutor",
                               side_effect=RuntimeError("no threads")):
            self.assertEqual(self.mod._outlook_summary(), "")


# ─────────────────────────────────────────────────────────────────────────
# _bed_remark file/dir edge paths
# ─────────────────────────────────────────────────────────────────────────
class BedRemarkTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_briefing")

    def test_remark_without_memory_file(self):
        # Regression: bobert_memory.json is NOT required — the signal is the
        # newest log mtime, so a fresh install without a memory file still
        # gets the late-night remark.
        late = time.mktime(time.struct_time((2026, 6, 1, 3, 15, 0, 0, 152, -1)))
        with mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir", return_value=["s.log"]), \
             mock.patch.object(self.mod.os.path, "getmtime", return_value=late):
            out = self.mod._bed_remark()
        self.assertIn("3:15 AM", out)
        self.assertIn("pace yourself", out)

    def test_no_logs_dir_blank(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=False):
            self.assertEqual(self.mod._bed_remark(), "")

    def test_listdir_error_blank(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir", side_effect=OSError("io")):
            self.assertEqual(self.mod._bed_remark(), "")

    def test_no_log_files_blank(self):
        # logs dir exists but has no .log files → latest_mtime stays 0.0.
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir", return_value=["notes.txt"]):
            self.assertEqual(self.mod._bed_remark(), "")


# ─────────────────────────────────────────────────────────────────────────
# Lazy-imported source helpers: news / umbrella / robot
# ─────────────────────────────────────────────────────────────────────────
class SourceHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_briefing")

    def _inject(self, modname, mod):
        return inject_modules(**{f"skill_morning_briefing.{modname}": mod, modname: mod})

    # ── _fetch_news ──────────────────────────────────────────────────────
    def test_fetch_news_returns_text(self):
        nb = _fake_source(get_news_text=lambda: "Today's headlines, sir. Rates up.")
        with self._inject("news_briefing", nb):
            self.assertEqual(self.mod._fetch_news(), "Today's headlines, sir. Rates up.")

    def test_fetch_news_getter_raises_blank(self):
        def _boom():
            raise RuntimeError("feed dead")
        nb = _fake_source(get_news_text=_boom)
        with self._inject("news_briefing", nb):
            self.assertEqual(self.mod._fetch_news(), "")

    def test_fetch_news_module_unavailable_blank(self):
        with inject_modules(**{"skill_morning_briefing.news_briefing": None,
                               "news_briefing": None}), \
             mock.patch("builtins.__import__",
                        side_effect=_block_import("news_briefing")):
            self.assertEqual(self.mod._fetch_news(), "")

    # ── _fetch_umbrella_alert ────────────────────────────────────────────
    def test_fetch_umbrella_returns_text(self):
        wb = _fake_source(get_umbrella_alert=lambda when: "Bring an umbrella, sir.")
        with self._inject("weather_briefing", wb):
            self.assertEqual(self.mod._fetch_umbrella_alert(), "Bring an umbrella, sir.")

    def test_fetch_umbrella_getter_raises_blank(self):
        def _boom(when):
            raise RuntimeError("open-meteo down")
        wb = _fake_source(get_umbrella_alert=_boom)
        with self._inject("weather_briefing", wb):
            self.assertEqual(self.mod._fetch_umbrella_alert(), "")

    def test_fetch_umbrella_module_unavailable_blank(self):
        with inject_modules(**{"skill_morning_briefing.weather_briefing": None,
                               "weather_briefing": None}), \
             mock.patch("builtins.__import__",
                        side_effect=_block_import("weather_briefing")):
            self.assertEqual(self.mod._fetch_umbrella_alert(), "")

    # ── _fetch_robot_volunteer ───────────────────────────────────────────
    def test_fetch_robot_returns_text(self):
        rr = _fake_source(get_morning_volunteer_text=lambda: "The bearing arrived.")
        with self._inject("repo_robot", rr):
            self.assertEqual(self.mod._fetch_robot_volunteer(), "The bearing arrived.")

    def test_fetch_robot_none_coerced_to_blank(self):
        rr = _fake_source(get_morning_volunteer_text=lambda: None)
        with self._inject("repo_robot", rr):
            self.assertEqual(self.mod._fetch_robot_volunteer(), "")

    def test_fetch_robot_getter_raises_blank(self):
        def _boom():
            raise RuntimeError("repo robot offline")
        rr = _fake_source(get_morning_volunteer_text=_boom)
        with self._inject("repo_robot", rr):
            self.assertEqual(self.mod._fetch_robot_volunteer(), "")

    def test_fetch_robot_module_unavailable_blank(self):
        with inject_modules(**{"skill_morning_briefing.repo_robot": None,
                               "repo_robot": None}), \
             mock.patch("builtins.__import__",
                        side_effect=_block_import("repo_robot")):
            self.assertEqual(self.mod._fetch_robot_volunteer(), "")


# ─────────────────────────────────────────────────────────────────────────
# _build_briefing extra branches + _fire_briefing + mark/flag errors
# ─────────────────────────────────────────────────────────────────────────
class BuildAndFireTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_briefing")

    def _all_sources(self, **over):
        base = dict(_fetch_weather="", _count_pending_tasks=0, _bed_remark="",
                    _fetch_news="", _fetch_umbrella_alert="", _outlook_summary="",
                    _fetch_robot_volunteer="")
        base.update(over)
        ctx = contextlib.ExitStack()
        for name, val in base.items():
            ctx.enter_context(mock.patch.object(self.mod, name, return_value=val))
        return ctx

    def test_build_umbrella_without_news_no_tag(self):
        # Umbrella + robot appended, but no news → plain head, no intent tag.
        with self._all_sources(_fetch_umbrella_alert="Bring an umbrella, sir.",
                               _fetch_robot_volunteer="The part shipped."):
            out = self.mod._build_briefing()
        self.assertIn("Bring an umbrella, sir.", out)
        self.assertIn("The part shipped.", out)
        self.assertFalse(out.startswith("[intent:briefing]"))

    def test_build_bed_remark_appended_to_empty_queue(self):
        # Regression: the remark must survive even when there are no tasks —
        # it used to be dropped outside the pending>1 branch.
        with self._all_sources(_count_pending_tasks=0, _bed_remark=" — pace yourself"):
            out = self.mod._build_briefing()
        self.assertIn("mercifully empty — pace yourself.", out)

    def test_build_bed_remark_appended_to_single_task(self):
        with self._all_sources(_count_pending_tasks=1, _bed_remark=" — pace yourself"):
            out = self.mod._build_briefing()
        self.assertIn("one task queued — pace yourself.", out)

    def test_build_bed_remark_appended_to_multi_task(self):
        with self._all_sources(_count_pending_tasks=5, _bed_remark=" — pace yourself"):
            out = self.mod._build_briefing()
        self.assertIn("5 tasks queued — pace yourself.", out)

    def test_fire_briefing_marks_then_builds_and_enqueues(self):
        order = []
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_mark_briefing_fired_today",
                               side_effect=lambda: order.append("mark")), \
             mock.patch.object(self.mod, "_build_briefing",
                               side_effect=lambda: order.append("build") or "TEXT"), \
             mock.patch.object(self.mod, "_enqueue_speech",
                               side_effect=lambda t: order.append(("enqueue", t))), \
             mock.patch.object(self.mod, "_show_card_safe",
                               side_effect=lambda: order.append("card")):
            self.mod._fire_briefing("reason")
        # Flag is marked BEFORE the build so a crash can't re-trigger.
        self.assertEqual(order[0], "mark")
        self.assertIn(("enqueue", "TEXT"), order)
        self.assertIn("card", order)

    def test_fire_briefing_suppressed_when_already_fired(self):
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_mark_briefing_fired_today") as mark, \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._fire_briefing("auto", force=False)
        mark.assert_not_called()
        enq.assert_not_called()

    def test_fire_briefing_force_bypasses_suppression(self):
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_mark_briefing_fired_today"), \
             mock.patch.object(self.mod, "_build_briefing", return_value="T"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_show_card_safe"):
            self.mod._fire_briefing("manual", force=True)
        enq.assert_called_once_with("T")

    def test_fire_briefing_build_failure_aborts_quietly(self):
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_mark_briefing_fired_today"), \
             mock.patch.object(self.mod, "_build_briefing", side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._fire_briefing("reason")   # no raise
        enq.assert_not_called()

    def test_fire_from_chain_aborts_on_post_delay_recheck(self):
        # Pre-check False, post-delay re-check True → fire suppressed.
        checks = iter([False, True])
        with mock.patch.object(self.mod, "_briefing_already_fired_today",
                               side_effect=lambda: next(checks)), \
             mock.patch.object(self.mod.time, "sleep"), \
             mock.patch.object(self.mod, "_fire_briefing") as fire:
            self.mod._fire_from_chain("chain")
        fire.assert_not_called()

    def test_mark_briefing_fired_writes_today(self):
        tmp = tempfile.mkdtemp(prefix="mbrief_flag_")
        self.addCleanup(lambda: _rmtree(tmp))
        flag = os.path.join(tmp, ".morning_briefing_last")
        self.mod._BRIEFING_FLAG_FILE = flag
        self.mod._mark_briefing_fired_today()
        with open(flag, encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), time.strftime("%Y-%m-%d"))

    def test_mark_briefing_fired_write_error_swallowed(self):
        with mock.patch("builtins.open", side_effect=OSError("ro")):
            self.mod._mark_briefing_fired_today()   # no raise

    def test_already_fired_read_error_returns_false(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertFalse(self.mod._briefing_already_fired_today())


def _block_import(*blocked):
    real = __import__

    def _imp(name, *a, **k):
        if name.split(".")[0] in blocked:
            raise ImportError(f"blocked {name}")
        return real(name, *a, **k)
    return _imp


def _rmtree(path):
    try:
        for f in os.listdir(path):
            try:
                os.unlink(os.path.join(path, f))
            except OSError:
                pass
        os.rmdir(path)
    except OSError:
        pass


if __name__ == "__main__":
    unittest.main()
