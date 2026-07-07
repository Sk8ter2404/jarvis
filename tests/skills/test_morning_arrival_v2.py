"""Logic tests for skills/morning_arrival_v2.py.

v2 is presence-gated. Tests cover the morning-window check, the six data-source
section formatters (weather / teams / news intro-strip / print / deliveries /
calendar) with their sibling skills mocked, the JARVIS-cadence composition with
the 60-second TTS budget + drop order, same-day + chain-already-fired
suppression, the chain entry, and the manual action — plus the sibling-skill
importer, the speech-queue enqueue (proactive + atomic-write fallback), the
JSON state load/save, the face_tracker presence snapshot + sustained-presence
rising-edge math, the full _section_print state machine, _gather_sections
parallel collection, and the fire path. The presence-watcher thread is neutered
by the harness; nothing real (network/face/printer/COM) runs. Module-level
mutable state (_presence_first_seen_at) is reset in tearDown so tests don't
leak the rising-edge anchor into one another.
"""
from __future__ import annotations

import contextlib
import datetime
import importlib.util
import json
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
    """Install fake modules into sys.modules for the block, restoring prior
    state (including absence) on exit. Mirrors the isolation contract from
    tests/skills/test_self_diagnostic.py."""
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


def _fake_mod(name="fake", **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class MorningArrivalV2Tests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_arrival_v2")
        self.addCleanup(lambda: self.mod._presence_first_seen_at.__setitem__(0, 0.0))

    # ── _within_morning_window (pure-ish) ────────────────────────────────
    def test_within_morning_window(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=time.struct_time((2026, 6, 1, 8, 0, 0, 0, 152, -1))):
            self.assertTrue(self.mod._within_morning_window())
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=time.struct_time((2026, 6, 1, 15, 0, 0, 0, 152, -1))):
            self.assertFalse(self.mod._within_morning_window())

    # ── _section_teams ───────────────────────────────────────────────────
    def test_section_teams_plural_with_sender(self):
        ms = mock.MagicMock()
        ms.get_teams_unread_count.return_value = {"count": 3, "top_sender": "Sam"}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_teams()
        self.assertEqual(out, "3 new Teams messages, one from Sam")

    def test_section_teams_single(self):
        ms = mock.MagicMock()
        ms.get_teams_unread_count.return_value = {"count": 1, "top_sender": "Alex"}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "one new Teams message from Alex")

    def test_section_teams_zero(self):
        ms = mock.MagicMock()
        ms.get_teams_unread_count.return_value = {"count": 0}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "")

    def test_section_teams_skill_absent(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_teams(), "")

    # ── _section_news strips the intro greeting ──────────────────────────
    def test_section_news_strips_intro(self):
        nb = mock.MagicMock()
        nb.get_news_text.return_value = "Today's headlines, sir. Rates rise. Storm clears."
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            out = self.mod._section_news()
        self.assertEqual(out, "Rates rise. Storm clears.")

    def test_section_news_empty(self):
        nb = mock.MagicMock()
        nb.get_news_text.return_value = ""
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.assertEqual(self.mod._section_news(), "")

    def test_section_news_skill_absent(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_news(), "")

    def test_section_deliveries_blank_after_strip(self):
        # action_check_orders returns whitespace-only → stripped to "" → "".
        aot = _fake_mod(action_check_orders=lambda _: "   ")
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "")

    # ── _section_deliveries drops sentinel responses ─────────────────────
    def test_section_deliveries_real_result(self):
        aot = mock.MagicMock()
        aot.action_check_orders.return_value = "Your headphones arrive tomorrow."
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "Your headphones arrive tomorrow.")

    def test_section_deliveries_sentinel_dropped(self):
        aot = mock.MagicMock()
        aot.action_check_orders.return_value = "No active Amazon orders right now."
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "")

    # ── _section_calendar ────────────────────────────────────────────────
    def test_section_calendar_subject(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = {
            "start": datetime.datetime(2026, 6, 1, 10, 30),
            "subject": "Standup", "organizer": "me"}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_calendar()
        self.assertEqual(out, "Standup at 10:30 AM")

    def test_section_calendar_none(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = None
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "")

    # ── _compose_briefing JARVIS cadence ─────────────────────────────────
    def test_compose_briefing_full(self):
        parts = {"weather": "Bring an umbrella, sir", "teams": "one new Teams message from Sam",
                 "calendar": "a sync at 10", "print": "the H2D finished overnight",
                 "deliveries": "Headphones arrive today.", "news": "Markets up."}
        out = self.mod._compose_briefing(parts)
        self.assertTrue(out.startswith("[intent:briefing] Good morning, sir."))
        self.assertIn("Bring an umbrella, sir.", out)
        self.assertIn("One new Teams message from Sam.", out)   # capitalised
        self.assertIn("You have a sync at 10.", out)
        self.assertIn("The H2D finished overnight.", out)

    def test_compose_briefing_nothing_overnight(self):
        parts = {k: "" for k in ("weather", "teams", "calendar", "print", "deliveries", "news")}
        out = self.mod._compose_briefing(parts)
        self.assertIn("Nothing of note overnight.", out)

    # ── TTS budget ───────────────────────────────────────────────────────
    def test_estimate_tts_strips_tag(self):
        secs = self.mod._estimate_tts_seconds("[intent:briefing] " + ("y" * 60))
        self.assertAlmostEqual(secs, 4.0, places=3)

    def test_compose_within_budget_drops_news_first(self):
        news_blob = "z" * 400
        parts = {"weather": "Dry today.", "teams": "teams ping", "calendar": "a sync at 10",
                 "print": "print done", "deliveries": "pkg today", "news": news_blob}
        with mock.patch.object(self.mod, "TTS_BUDGET_SECONDS", 10.0):
            out = self.mod._compose_within_budget(parts)
        # news is first in TTS_DROP_ORDER → dropped; calendar (last) survives.
        self.assertNotIn(news_blob, out)
        self.assertIn("You have a sync at 10.", out)

    # ── suppression ──────────────────────────────────────────────────────
    def test_already_fired_today(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod, "_load_state", return_value={"last_fired_date": today}):
            self.assertTrue(self.mod._already_fired_today())
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            self.assertFalse(self.mod._already_fired_today())

    def test_fire_arrival_suppressed_when_already_fired(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_build_briefing") as build:
            out = self.mod._fire_arrival("auto", force=False)
        self.assertEqual(out, "")
        build.assert_not_called()

    def test_fire_arrival_suppressed_when_chain_already_briefed(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_chain_morning_briefing_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_build_briefing") as build:
            out = self.mod._fire_arrival("auto", force=False)
        self.assertEqual(out, "")
        build.assert_not_called()

    def test_chain_suppression_marks_fired_so_watcher_stops(self):
        """Regression: when the chain already briefed, the suppressed path
        must still mark v2's own same-day state, otherwise the presence
        watcher re-invokes _fire_arrival every poll for the rest of the
        morning window."""
        with mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_chain_morning_briefing_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.mod._fire_arrival("auto", force=False)
        self.assertEqual(out, "")
        enq.assert_not_called()
        mark.assert_called_once()
        self.assertIn("morning_chain", mark.call_args[0][0])

    def test_fire_arrival_force_bypasses_suppression(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_build_briefing", return_value="[intent:briefing] hi"), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_mark_fired"):
            out = self.mod._fire_arrival("manual", force=True)
        self.assertEqual(out, "[intent:briefing] hi")

    # ── manual action ────────────────────────────────────────────────────
    def test_action_returns_text(self):
        mod, actions = load_skill_isolated("morning_arrival_v2")
        with mock.patch.object(mod, "_fire_arrival", return_value="[intent:briefing] Good morning."):
            out = actions["morning_arrival_v2"]("")
        self.assertIn("Good morning", out)

    def test_action_no_content(self):
        mod, actions = load_skill_isolated("morning_arrival_v2")
        with mock.patch.object(mod, "_fire_arrival", return_value=""):
            out = actions["morning_arrival_v2"]("")
        self.assertIn("no content", out.lower())

    # ── extra section branches ───────────────────────────────────────────
    def test_section_teams_plural_no_sender(self):
        ms = _fake_mod(get_teams_unread_count=lambda: {"count": 2, "top_sender": ""})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "2 new Teams messages")

    def test_section_teams_single_no_sender(self):
        ms = _fake_mod(get_teams_unread_count=lambda: {"count": 1, "top_sender": ""})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "one new Teams message")

    def test_section_teams_non_dict_result(self):
        ms = _fake_mod(get_teams_unread_count=lambda: None)
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "")

    def test_section_teams_bad_count_coerces_zero(self):
        ms = _fake_mod(get_teams_unread_count=lambda: {"count": "lots"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "")

    def test_section_teams_getter_raises(self):
        def _boom():
            raise RuntimeError("chat.read 403")
        ms = _fake_mod(get_teams_unread_count=_boom)
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "")

    def test_section_teams_no_getter_attr(self):
        ms = _fake_mod()   # no get_teams_unread_count
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "")

    def test_section_news_no_getter_attr(self):
        nb = _fake_mod()   # no get_news_text
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.assertEqual(self.mod._section_news(), "")

    def test_section_news_getter_raises(self):
        def _boom():
            raise RuntimeError("feeds down")
        nb = _fake_mod(get_news_text=_boom)
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.assertEqual(self.mod._section_news(), "")

    def test_section_weather_returns_alert(self):
        wb = _fake_mod(get_umbrella_alert=lambda when: "  Bring an umbrella.  ")
        with mock.patch.object(self.mod, "_import_skill", return_value=wb):
            self.assertEqual(self.mod._section_weather(), "Bring an umbrella.")

    def test_section_weather_skill_absent(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_weather(), "")

    def test_section_weather_no_getter_attr(self):
        wb = _fake_mod()
        with mock.patch.object(self.mod, "_import_skill", return_value=wb):
            self.assertEqual(self.mod._section_weather(), "")

    def test_section_weather_getter_raises(self):
        def _boom(when):
            raise RuntimeError("meteo down")
        wb = _fake_mod(get_umbrella_alert=_boom)
        with mock.patch.object(self.mod, "_import_skill", return_value=wb):
            self.assertEqual(self.mod._section_weather(), "")

    def test_section_deliveries_skill_absent(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_deliveries(), "")

    def test_section_deliveries_alt_function_name(self):
        # Falls back to check_orders when action_check_orders is absent.
        aot = _fake_mod(check_orders=lambda _: "Package arriving today.")
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "Package arriving today.")

    def test_section_deliveries_no_callable(self):
        aot = _fake_mod()   # neither action_check_orders nor check_orders
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "")

    def test_section_deliveries_getter_raises(self):
        def _boom(_):
            raise RuntimeError("amazon down")
        aot = _fake_mod(action_check_orders=_boom)
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "")

    def test_section_deliveries_nothing_currently_sentinel(self):
        aot = _fake_mod(action_check_orders=lambda _: "Nothing currently in transit.")
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "")

    def test_section_calendar_skill_absent(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_calendar(), "")

    def test_section_calendar_no_getter_attr(self):
        ms = _fake_mod()
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "")

    def test_section_calendar_getter_raises(self):
        def _boom(when):
            raise RuntimeError("graph down")
        ms = _fake_mod(get_first_meeting=_boom)
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "")

    def test_section_calendar_meeting_without_start_hour(self):
        ms = _fake_mod(get_first_meeting=lambda when: {"subject": "x", "start": "not-a-dt"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "")

    def test_section_calendar_long_subject_with_organizer_becomes_sync(self):
        ms = _fake_mod(get_first_meeting=lambda when: {
            "start": datetime.datetime(2026, 6, 1, 9, 15),
            "subject": "x" * 60,   # > 40 chars → not used verbatim
            "organizer": "Dana Scully"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_calendar()
        self.assertEqual(out, "a sync with Dana at 9:15 AM")

    def test_section_calendar_long_subject_no_organizer_keeps_subject(self):
        ms = _fake_mod(get_first_meeting=lambda when: {
            "start": datetime.datetime(2026, 6, 1, 9, 15),
            "subject": "y" * 60, "organizer": ""})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_calendar()
        self.assertIn("y" * 60, out)

    def test_section_calendar_no_subject_no_organizer_generic(self):
        ms = _fake_mod(get_first_meeting=lambda when: {
            "start": datetime.datetime(2026, 6, 1, 13, 0),
            "subject": "", "organizer": ""})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "a meeting at 1 PM")

    def test_section_calendar_on_the_hour_morning_drops_suffix(self):
        # Whole-hour morning meeting between 8 and 11 → bare hour, no AM/PM.
        ms = _fake_mod(get_first_meeting=lambda when: {
            "start": datetime.datetime(2026, 6, 1, 9, 0),
            "subject": "Standup", "organizer": "me"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "Standup at 9")

    def test_section_calendar_organizer_email_only_no_who(self):
        # Organizer's first token contains '@' → no who-label derived.
        ms = _fake_mod(get_first_meeting=lambda when: {
            "start": datetime.datetime(2026, 6, 1, 9, 15),
            "subject": "z" * 60, "organizer": "dana@x.invalid"})
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_calendar()
        # No who → long subject retained as-is.
        self.assertIn("z" * 60, out)


# ─────────────────────────────────────────────────────────────────────────
# _import_skill resolution order
# ─────────────────────────────────────────────────────────────────────────
class ImportSkillTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival_v2")

    def test_prefers_live_registered_module(self):
        live = _fake_mod("skill_face_tracker", marker="LIVE")
        with inject_modules(skill_face_tracker=live):
            out = self.mod._import_skill("face_tracker")
        self.assertIs(out, live)

    def test_falls_back_to_skills_package_import(self):
        target = _fake_mod("skills.news_briefing", marker="PKG")
        with inject_modules(skill_news_briefing=None):
            with mock.patch.object(self.mod.importlib, "import_module",
                                   return_value=target) as imp:
                out = self.mod._import_skill("news_briefing")
        self.assertIs(out, target)
        imp.assert_called_once_with("skills.news_briefing")

    def test_absolute_fallback_when_package_import_fails(self):
        target = _fake_mod("bambu_monitor", marker="ABS")

        def _imp(name, *a, **k):
            if name == "skills.bambu_monitor":
                raise ImportError("no skills pkg")
            if name == "bambu_monitor":
                return target
            raise ImportError(name)

        with inject_modules(skill_bambu_monitor=None):
            with mock.patch.object(self.mod.importlib, "import_module", side_effect=_imp):
                out = self.mod._import_skill("bambu_monitor")
        self.assertIs(out, target)

    def test_returns_none_when_all_imports_fail(self):
        with inject_modules(skill_ghost=None):
            with mock.patch.object(self.mod.importlib, "import_module",
                                   side_effect=ImportError("nope")):
                self.assertIsNone(self.mod._import_skill("ghost"))


# ─────────────────────────────────────────────────────────────────────────
# _enqueue_speech: proactive_announce + atomic-write fallback
# ─────────────────────────────────────────────────────────────────────────
class EnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival_v2")
        self.tmp = tempfile.mkdtemp(prefix="arrv2_speech_")
        self.addCleanup(lambda: _rmtree(self.tmp))
        # _enqueue_speech derives the queue path from _PROJECT_DIR; redirect it.
        self.mod._PROJECT_DIR = self.tmp
        self.queue = os.path.join(self.tmp, "pending_speech.json")

    def test_routes_through_proactive_announce(self):
        calls = []
        bc = _fake_mod("bobert_companion",
                       proactive_announce=lambda msg, source=None: calls.append((msg, source)))
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("morning sir")
        self.assertEqual(calls, [("morning sir", "arrival_v2")])
        self.assertFalse(os.path.exists(self.queue))

    def test_fallback_atomic_write_when_no_announcer(self):
        bc = _fake_mod("bobert_companion")   # no proactive_announce
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("queued")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "queued")

    def test_fallback_appends_and_tolerates_corrupt(self):
        with open(self.queue, "w", encoding="utf-8") as f:
            f.write("{broken")
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.mod._enqueue_speech("after-corrupt")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data[-1]["message"], "after-corrupt")

    def test_fallback_write_failure_swallowed(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("ro")):
            self.mod._enqueue_speech("doomed")   # no raise


# ─────────────────────────────────────────────────────────────────────────
# state load / save / mark / already-fired
# ─────────────────────────────────────────────────────────────────────────
class StateIOTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival_v2")
        self.tmp = tempfile.mkdtemp(prefix="arrv2_state_")
        self.addCleanup(lambda: _rmtree(self.tmp))
        self.state = os.path.join(self.tmp, "state.json")
        self.mod._STATE_FILE = self.state

    def test_load_state_missing_is_empty(self):
        self.assertEqual(self.mod._load_state(), {})

    def test_save_then_load_round_trip(self):
        self.mod._save_state({"last_fired_date": "2026-06-01", "n": 3})
        self.assertEqual(self.mod._load_state(),
                         {"last_fired_date": "2026-06-01", "n": 3})

    def test_load_state_corrupt_is_empty(self):
        with open(self.state, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.mod._load_state(), {})

    def test_load_state_json_null_is_empty(self):
        with open(self.state, "w", encoding="utf-8") as f:
            f.write("null")
        self.assertEqual(self.mod._load_state(), {})

    def test_save_state_write_error_swallowed(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("ro")):
            self.mod._save_state({"x": 1})   # no raise

    def test_mark_fired_persists_date_reason_ts(self):
        with mock.patch.object(self.mod.time, "time", return_value=42.0):
            self.mod._mark_fired("presence watcher")
        st = self.mod._load_state()
        self.assertEqual(st["last_fired_date"], time.strftime("%Y-%m-%d"))
        self.assertEqual(st["last_reason"], "presence watcher")
        self.assertEqual(st["last_fired_ts"], 42.0)

    def test_already_fired_today_true_false(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod, "_load_state", return_value={"last_fired_date": today}):
            self.assertTrue(self.mod._already_fired_today())
        with mock.patch.object(self.mod, "_load_state",
                               return_value={"last_fired_date": "1999-01-01"}):
            self.assertFalse(self.mod._already_fired_today())


# ─────────────────────────────────────────────────────────────────────────
# _chain_morning_briefing_fired_today (skill_morning_chain faked)
# ─────────────────────────────────────────────────────────────────────────
class ChainFiredTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival_v2")

    def test_returns_true_when_any_chain_skill_fired(self):
        mc = _fake_mod("skill_morning_chain",
                       SKILL_NAMES=("arrival", "handoff", "briefing"),
                       _skill_already_fired_today=lambda n: n == "briefing")
        with inject_modules(skill_morning_chain=mc):
            self.assertTrue(self.mod._chain_morning_briefing_fired_today())

    def test_returns_false_when_none_fired(self):
        mc = _fake_mod("skill_morning_chain",
                       SKILL_NAMES=("arrival", "handoff", "briefing"),
                       _skill_already_fired_today=lambda n: False)
        with inject_modules(skill_morning_chain=mc):
            self.assertFalse(self.mod._chain_morning_briefing_fired_today())

    def test_returns_false_when_chain_not_importable(self):
        def _imp(name, *a, **k):
            raise ImportError("no chain")
        with inject_modules(skill_morning_chain=None):
            with mock.patch.object(self.mod.importlib, "import_module", side_effect=_imp):
                self.assertFalse(self.mod._chain_morning_briefing_fired_today())

    def test_returns_false_when_checker_missing(self):
        mc = _fake_mod("skill_morning_chain", SKILL_NAMES=("arrival",))
        # no _skill_already_fired_today attr
        with inject_modules(skill_morning_chain=mc):
            self.assertFalse(self.mod._chain_morning_briefing_fired_today())

    def test_returns_false_when_checker_raises(self):
        def _boom(_):
            raise RuntimeError("flag read error")
        mc = _fake_mod("skill_morning_chain", SKILL_NAMES=("arrival",),
                       _skill_already_fired_today=_boom)
        with inject_modules(skill_morning_chain=mc):
            self.assertFalse(self.mod._chain_morning_briefing_fired_today())


# ─────────────────────────────────────────────────────────────────────────
# presence detection
# ─────────────────────────────────────────────────────────────────────────
class PresenceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival_v2")
        self.addCleanup(lambda: self.mod._presence_first_seen_at.__setitem__(0, 0.0))

    def test_face_state_uses_snapshot_fn(self):
        ft = _fake_mod("face_tracker",
                       _snapshot_state=lambda: {"face_visible": True})
        with mock.patch.object(self.mod, "_import_skill", return_value=ft):
            self.assertEqual(self.mod._face_tracker_state(), {"face_visible": True})

    def test_face_state_snapshot_non_dict_returns_none(self):
        ft = _fake_mod("face_tracker", _snapshot_state=lambda: "weird")
        with mock.patch.object(self.mod, "_import_skill", return_value=ft):
            self.assertIsNone(self.mod._face_tracker_state())

    def test_face_state_snapshot_raises_returns_none(self):
        def _boom():
            raise RuntimeError("lock held")
        ft = _fake_mod("face_tracker", _snapshot_state=_boom)
        with mock.patch.object(self.mod, "_import_skill", return_value=ft):
            self.assertIsNone(self.mod._face_tracker_state())

    def test_face_state_falls_back_to_state_dict_with_lock(self):
        import threading
        lock = threading.Lock()
        ft = _fake_mod("face_tracker")
        ft._state = {"face_visible": False, "x": 1}
        ft._state_lock = lock
        # no _snapshot_state attr → uses _state under _state_lock
        with mock.patch.object(self.mod, "_import_skill", return_value=ft):
            out = self.mod._face_tracker_state()
        self.assertEqual(out, {"face_visible": False, "x": 1})
        self.assertFalse(lock.locked())   # released

    def test_face_state_state_not_dict_returns_none(self):
        ft = _fake_mod("face_tracker")
        ft._state = "not-a-dict"
        with mock.patch.object(self.mod, "_import_skill", return_value=ft):
            self.assertIsNone(self.mod._face_tracker_state())

    def test_face_state_no_lock_copies_state(self):
        ft = _fake_mod("face_tracker")
        ft._state = {"face_visible": True}
        ft._state_lock = None
        with mock.patch.object(self.mod, "_import_skill", return_value=ft):
            self.assertEqual(self.mod._face_tracker_state(), {"face_visible": True})

    def test_face_state_lock_raises_returns_none(self):
        ft = _fake_mod("face_tracker")
        ft._state = {"face_visible": True}
        ft._state_lock = _RaisingLock()
        with mock.patch.object(self.mod, "_import_skill", return_value=ft):
            self.assertIsNone(self.mod._face_tracker_state())

    def test_face_state_tracker_absent_returns_none(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertIsNone(self.mod._face_tracker_state())

    def test_sustained_zero_when_state_unknown(self):
        with mock.patch.object(self.mod, "_face_tracker_state", return_value=None):
            self.assertEqual(self.mod._sustained_presence_seconds(), 0.0)

    def test_sustained_zero_when_not_visible_resets_anchor(self):
        self.mod._presence_first_seen_at[0] = 12345.0
        with mock.patch.object(self.mod, "_face_tracker_state",
                               return_value={"face_visible": False}):
            self.assertEqual(self.mod._sustained_presence_seconds(), 0.0)
        self.assertEqual(self.mod._presence_first_seen_at[0], 0.0)

    def test_sustained_zero_when_no_sample_yet(self):
        with mock.patch.object(self.mod, "_face_tracker_state",
                               return_value={"face_visible": True, "last_sample_at": 0.0}):
            self.assertEqual(self.mod._sustained_presence_seconds(), 0.0)

    def test_sustained_anchors_to_last_face_at(self):
        snap = {"face_visible": True, "last_sample_at": 1000.0,
                "last_face_at": 990.0, "first_face_at": 980.0}
        with mock.patch.object(self.mod, "_face_tracker_state", return_value=snap), \
             mock.patch.object(self.mod.time, "time", return_value=1000.0):
            secs = self.mod._sustained_presence_seconds()
        self.assertAlmostEqual(secs, 10.0)   # now(1000) - last_face_at(990)

    def test_sustained_anchors_to_first_face_when_no_last(self):
        snap = {"face_visible": True, "last_sample_at": 1000.0,
                "last_face_at": 0.0, "first_face_at": 985.0}
        with mock.patch.object(self.mod, "_face_tracker_state", return_value=snap), \
             mock.patch.object(self.mod.time, "time", return_value=1000.0):
            secs = self.mod._sustained_presence_seconds()
        self.assertAlmostEqual(secs, 15.0)

    def test_sustained_anchors_to_now_when_no_face_timestamps(self):
        snap = {"face_visible": True, "last_sample_at": 1000.0}
        with mock.patch.object(self.mod, "_face_tracker_state", return_value=snap), \
             mock.patch.object(self.mod.time, "time", return_value=1000.0):
            secs = self.mod._sustained_presence_seconds()
        self.assertAlmostEqual(secs, 0.0)   # anchored to now → ~0 elapsed


# ─────────────────────────────────────────────────────────────────────────
# _section_print state machine
# ─────────────────────────────────────────────────────────────────────────
class SectionPrintTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival_v2")
        # Point the on-disk bambu state at a temp path so the file fallback
        # never reads the real overlay file.
        self.tmp = tempfile.mkdtemp(prefix="arrv2_bambu_")
        self.addCleanup(lambda: _rmtree(self.tmp))
        self.bambu = os.path.join(self.tmp, "bambu_overlay_state.json")
        self.mod._BAMBU_STATE = self.bambu

    def test_overnight_summary_with_filename_and_phrase(self):
        bm = _fake_mod("bambu_monitor",
                       get_last_print_completion_summary=lambda within_seconds=None: {
                           "filename": "gear bracket", "finish_phrase": "3 AM"})
        with mock.patch.object(self.mod, "_import_skill", return_value=bm):
            out = self.mod._section_print()
        self.assertEqual(out, "the H2D finished of 'gear bracket' at 3 AM overnight")

    def test_overnight_summary_without_phrase(self):
        bm = _fake_mod("bambu_monitor",
                       get_last_print_completion_summary=lambda within_seconds=None: {
                           "filename": "", "finish_phrase": ""})
        with mock.patch.object(self.mod, "_import_skill", return_value=bm):
            self.assertEqual(self.mod._section_print(), "the H2D finished overnight")

    def test_summary_raises_falls_through_to_state(self):
        def _boom(within_seconds=None):
            raise RuntimeError("mqtt gone")
        bm = _fake_mod("bambu_monitor", get_last_print_completion_summary=_boom)
        bm._state = {}            # empty live state
        bm._state_lock = None
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._section_print(), "")

    def test_live_state_running_with_percent(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 5000.0, "gcode_state": "RUNNING",
                     "filename": "part.3mf", "mc_percent": 42}
        bm._strip_filename = lambda n: "part"
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=5001.0):
            self.assertEqual(self.mod._section_print(), "the H2D is mid-print at 42 percent")

    def test_live_state_running_without_percent(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 5000.0, "gcode_state": "PRINTING",
                     "filename": "", "mc_percent": None}
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=5001.0):
            self.assertEqual(self.mod._section_print(), "the H2D is mid-print")

    def test_live_state_failed(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 5000.0, "gcode_state": "FAILED"}
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=5001.0):
            self.assertEqual(self.mod._section_print(), "the H2D's overnight print failed")

    def test_live_state_paused(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 5000.0, "gcode_state": "PAUSE"}
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=5001.0):
            self.assertEqual(self.mod._section_print(), "the H2D is paused mid-print")

    def test_live_state_idle_returns_blank(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 5000.0, "gcode_state": "FINISH"}
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=5001.0):
            self.assertEqual(self.mod._section_print(), "")

    def test_state_too_old_returns_blank(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 1000.0, "gcode_state": "RUNNING"}
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=1000.0 + 40 * 3600):
            self.assertEqual(self.mod._section_print(), "")

    def test_no_last_update_returns_blank(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"gcode_state": "RUNNING"}   # no last_update
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._section_print(), "")

    def test_file_fallback_used_when_no_live_state(self):
        # bm has no usable live state → read the on-disk overlay file.
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        # no _state/_state_lock → live state stays {}
        with open(self.bambu, "w", encoding="utf-8") as f:
            json.dump({"last_update": 7000.0, "gcode_state": "RUNNING",
                       "filename": "C:/prints/cool_thing.gcode", "mc_percent": 88}, f)
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=7001.0):
            self.assertEqual(self.mod._section_print(), "the H2D is mid-print at 88 percent")

    def test_file_fallback_corrupt_is_blank(self):
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        with open(self.bambu, "w", encoding="utf-8") as f:
            f.write("{broken json")
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=7001.0):
            self.assertEqual(self.mod._section_print(), "")

    def test_bambu_skill_absent_uses_file(self):
        # _import_skill returns None → straight to the file fallback.
        with open(self.bambu, "w", encoding="utf-8") as f:
            json.dump({"last_update": 8000.0, "gcode_state": "FAILED"}, f)
        with mock.patch.object(self.mod, "_import_skill", return_value=None), \
             mock.patch.object(self.mod.time, "time", return_value=8001.0):
            self.assertEqual(self.mod._section_print(), "the H2D's overnight print failed")

    def test_live_state_lock_raises_falls_back_to_file(self):
        # The `with lock` around the live state raises → state stays {} and the
        # on-disk overlay file is consulted instead.
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = _RaisingLock()
        bm._state = {"last_update": 9000.0, "gcode_state": "RUNNING"}
        with open(self.bambu, "w", encoding="utf-8") as f:
            json.dump({"last_update": 9000.0, "gcode_state": "PAUSE"}, f)
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=9001.0):
            self.assertEqual(self.mod._section_print(), "the H2D is paused mid-print")

    def test_strip_filename_fallback_regex_when_helper_raises(self):
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 5000.0, "gcode_state": "PAUSE",
                     "filename": "my_cool-part.3mf"}

        def _boom(_):
            raise RuntimeError("strip fail")
        bm._strip_filename = _boom
        # Even though paused doesn't echo fname, the regex path still executes.
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=5001.0):
            self.assertEqual(self.mod._section_print(), "the H2D is paused mid-print")

    def test_regex_filename_exception_is_swallowed(self):
        # Both _strip_filename absent AND the regex basename cleanup raises →
        # fname stays "" and the state-machine result still returns.
        import threading
        bm = _fake_mod("bambu_monitor")
        bm.get_last_print_completion_summary = lambda within_seconds=None: None
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 5000.0, "gcode_state": "FAILED",
                     "filename": "weird.3mf"}
        # no _strip_filename → regex fallback runs; force it to raise.
        with mock.patch.object(self.mod, "_import_skill", return_value=bm), \
             mock.patch.object(self.mod.time, "time", return_value=5001.0), \
             mock.patch.object(self.mod.re, "sub", side_effect=RuntimeError("re boom")):
            self.assertEqual(self.mod._section_print(), "the H2D's overnight print failed")


# ─────────────────────────────────────────────────────────────────────────
# _gather_sections + _build_briefing + fire path + watcher
# ─────────────────────────────────────────────────────────────────────────
class GatherAndFireTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_arrival_v2")
        self.addCleanup(lambda: self.mod._presence_first_seen_at.__setitem__(0, 0.0))

    def test_gather_sections_collects_all_six(self):
        with mock.patch.object(self.mod, "_section_weather", return_value="W"), \
             mock.patch.object(self.mod, "_section_teams", return_value="T"), \
             mock.patch.object(self.mod, "_section_print", return_value="P"), \
             mock.patch.object(self.mod, "_section_deliveries", return_value="D"), \
             mock.patch.object(self.mod, "_section_calendar", return_value="C"), \
             mock.patch.object(self.mod, "_section_news", return_value="N"):
            out = self.mod._gather_sections()
        self.assertEqual(out, {"weather": "W", "teams": "T", "print": "P",
                               "deliveries": "D", "calendar": "C", "news": "N"})

    def test_gather_sections_crashed_section_maps_blank(self):
        with mock.patch.object(self.mod, "_section_weather",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod, "_section_teams", return_value="T"), \
             mock.patch.object(self.mod, "_section_print", return_value=""), \
             mock.patch.object(self.mod, "_section_deliveries", return_value=""), \
             mock.patch.object(self.mod, "_section_calendar", return_value=""), \
             mock.patch.object(self.mod, "_section_news", return_value=""):
            out = self.mod._gather_sections()
        self.assertEqual(out["weather"], "")
        self.assertEqual(out["teams"], "T")

    def test_gather_sections_timeout_maps_blank(self):
        # Force every future to raise FutureTimeoutError so the timeout branch
        # runs deterministically (no real slow section / sleep).
        from concurrent.futures import TimeoutError as FutureTimeoutError

        class _Fut:
            def __init__(self, fn):
                self._fn = fn

            def result(self, timeout=None):
                raise FutureTimeoutError()

        class _Exec:
            def __init__(self, max_workers=None):
                pass

            def submit(self, fn):
                return _Fut(fn)

            def shutdown(self, wait=True, cancel_futures=False):
                pass

        with mock.patch.object(self.mod, "ThreadPoolExecutor", _Exec):
            out = self.mod._gather_sections()
        self.assertEqual(set(out.values()), {""})   # all sections dropped

    def test_compose_within_budget_break_after_single_drop(self):
        # news is empty (the first TTS_DROP_ORDER key → `continue`), weather is
        # the oversized payload; dropping weather brings it under → loop breaks.
        # This exercises BOTH the skip-empty `continue` and the `break`.
        parts = {"weather": "z" * 2000, "teams": "", "calendar": "", "print": "",
                 "deliveries": "", "news": ""}
        with mock.patch.object(self.mod, "TTS_BUDGET_SECONDS", 5.0):
            out = self.mod._compose_within_budget(parts)
        self.assertNotIn("z" * 2000, out)
        # Everything dropped → the "nothing overnight" filler stands in.
        self.assertIn("Nothing of note overnight.", out)

    def test_build_briefing_pipes_gather_to_compose(self):
        with mock.patch.object(self.mod, "_gather_sections",
                               return_value={"weather": "Dry.", "teams": "", "calendar": "",
                                             "print": "", "deliveries": "", "news": ""}):
            out = self.mod._build_briefing()
        self.assertIn("Dry.", out)
        self.assertTrue(out.startswith("[intent:briefing]"))

    def test_compose_within_budget_returns_immediately_when_short(self):
        parts = {"weather": "Dry.", "teams": "", "calendar": "", "print": "",
                 "deliveries": "", "news": ""}
        out = self.mod._compose_within_budget(parts)
        self.assertIn("Dry.", out)

    def test_estimate_tts_empty_body_is_zero(self):
        self.assertEqual(self.mod._estimate_tts_seconds("[intent:briefing] "), 0.0)
        self.assertEqual(self.mod._estimate_tts_seconds(""), 0.0)

    def test_fire_arrival_happy_path_enqueues_and_marks(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_chain_morning_briefing_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_build_briefing", return_value="[intent:briefing] hi"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.mod._fire_arrival("presence")
        self.assertEqual(out, "[intent:briefing] hi")
        enq.assert_called_once_with("[intent:briefing] hi")
        mark.assert_called_once_with("presence")

    def test_fire_arrival_empty_text_not_enqueued(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_chain_morning_briefing_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_build_briefing", return_value=""), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.mod._fire_arrival("presence")
        self.assertEqual(out, "")
        enq.assert_not_called()
        mark.assert_not_called()

    def test_fire_arrival_build_failure_returns_blank(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_chain_morning_briefing_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_build_briefing", side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            out = self.mod._fire_arrival("presence")
        self.assertEqual(out, "")
        enq.assert_not_called()

    def test_fire_from_chain_suppressed_when_fired(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            self.assertEqual(self.mod._fire_from_chain("chain"), "")
        fire.assert_not_called()

    def test_fire_from_chain_delegates_when_not_fired(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_fire_arrival", return_value="TEXT") as fire:
            self.assertEqual(self.mod._fire_from_chain("chain"), "TEXT")
        fire.assert_called_once_with("chain")

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("morning_arrival_v2")
        with mock.patch.object(mod, "_fire_arrival", side_effect=RuntimeError("kaboom")):
            out = actions["morning_arrival_v2"]("")
        self.assertIn("failed", out.lower())

    def test_register_binds_both_action_aliases(self):
        actions = {}
        # Thread.start is neutered by the harness, so register() returns cleanly.
        mod, actions = load_skill_isolated("morning_arrival_v2", actions=actions)
        self.assertIn("morning_arrival_v2", actions)
        self.assertIn("arrival_briefing_v2", actions)
        self.assertIs(actions["morning_arrival_v2"], actions["arrival_briefing_v2"])


# ─────────────────────────────────────────────────────────────────────────
# _watch_for_arrival single-iteration drive
# ─────────────────────────────────────────────────────────────────────────
class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("morning_arrival_v2")
        self.addCleanup(lambda: self.mod._presence_first_seen_at.__setitem__(0, 0.0))

    def _run_one_loop(self):
        """Drive _watch_for_arrival through exactly one while-iteration by
        making the second time.sleep raise to break the infinite loop."""
        calls = {"sleep": 0}

        def _sleep(_):
            calls["sleep"] += 1
            if calls["sleep"] >= 2:   # 1st = head-start, 2nd = loop tail
                raise KeyboardInterrupt
        return _sleep, calls

    def _drive(self):
        """Run the watcher once with stdout swallowed — the watcher's startup
        banner contains non-cp1252 glyphs (≥, –) that would crash the real
        Windows console encoder; the live loader captures stdout but our
        direct call here does not."""
        import io
        return contextlib.redirect_stdout(io.StringIO())

    def test_watcher_fires_on_sustained_presence(self):
        sleep, _ = self._run_one_loop()
        with self._drive(), \
             mock.patch.object(self.mod.time, "sleep", side_effect=sleep), \
             mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_within_morning_window", return_value=True), \
             mock.patch.object(self.mod, "_sustained_presence_seconds", return_value=5.0), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_for_arrival()
        fire.assert_called_once()

    def test_watcher_skips_when_already_fired(self):
        # Let the head-start sleep (1) and the skip-branch sleep (2) pass so the
        # `continue` after the early-out actually executes, then break on the
        # next loop's skip-branch sleep (3).
        calls = {"n": 0}

        def _sleep(_):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise KeyboardInterrupt
        with self._drive(), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_within_morning_window", return_value=True), \
             mock.patch.object(self.mod, "_sustained_presence_seconds", return_value=5.0), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_for_arrival()
        fire.assert_not_called()

    def test_watcher_skips_outside_window(self):
        sleep, _ = self._run_one_loop()
        with self._drive(), \
             mock.patch.object(self.mod.time, "sleep", side_effect=sleep), \
             mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_within_morning_window", return_value=False), \
             mock.patch.object(self.mod, "_sustained_presence_seconds", return_value=5.0), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_for_arrival()
        fire.assert_not_called()

    def test_watcher_below_threshold_does_not_fire(self):
        sleep, _ = self._run_one_loop()
        with self._drive(), \
             mock.patch.object(self.mod.time, "sleep", side_effect=sleep), \
             mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_within_morning_window", return_value=True), \
             mock.patch.object(self.mod, "_sustained_presence_seconds", return_value=1.0), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_for_arrival()
        fire.assert_not_called()

    def test_watcher_tick_error_is_swallowed(self):
        # An exception inside the try block is caught and logged; the loop
        # then sleeps and (here) the 2nd sleep raises KeyboardInterrupt.
        sleep, _ = self._run_one_loop()
        with self._drive(), \
             mock.patch.object(self.mod.time, "sleep", side_effect=sleep), \
             mock.patch.object(self.mod, "_already_fired_today",
                               side_effect=RuntimeError("state read boom")), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_for_arrival()
        fire.assert_not_called()


class _RaisingLock:
    """A context-manager lock whose ``__enter__`` raises — used to exercise the
    defensive ``try/except`` around ``with lock:`` snapshot blocks."""
    def __enter__(self):
        raise RuntimeError("lock acquisition failed")

    def __exit__(self, *a):
        return False


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


class MorningArrivalV2ImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        # Re-exec the source with the project root removed from sys.path so the
        # `if _PROJECT_DIR not in sys.path: sys.path.insert(...)` guard runs.
        mod, _ = load_skill_isolated("morning_arrival_v2")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("ma_v2_reexec", path)
        m = importlib.util.module_from_spec(spec)
        m.skill_utils = {}
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.abspath(p) != os.path.abspath(proj)]
            spec.loader.exec_module(m)
            self.assertIn(m._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved


if __name__ == "__main__":
    unittest.main()
