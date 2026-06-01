"""Logic tests for skills/morning_arrival.py.

Covers the data-section formatters (time, weather C→F, first-meeting phrasing),
the "Three things require your attention" clause builder, full briefing
composition, the TTS-budget estimator + progressive section dropping, the
overnight-merge log parser, overnight-anomaly extraction, the silence-gate
helper, same-day suppression, the chain entry's silence gate, and the manual
action. ThreadPoolExecutor sections are mocked away; no network/hardware.
"""
from __future__ import annotations

import datetime
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class MorningArrivalTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_arrival")

    # ── _section_time_phrase (pure) ──────────────────────────────────────
    def test_section_time_phrase_format(self):
        out = self.mod._section_time_phrase()
        self.assertRegex(out, r"^\d{1,2}:\d{2} (AM|PM)$")

    # ── _section_weather_phrase (C→F via briefing_sources) ───────────────
    def test_weather_phrase_converts_to_fahrenheit(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": 18, "desc": "Clear"}  # 18C → 64F
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            out = self.mod._section_weather_phrase()
        self.assertEqual(out, "64 degrees and clear")

    def test_weather_phrase_degraded(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_weather_phrase(), "")

    # ── _section_first_meeting_phrase ────────────────────────────────────
    def test_first_meeting_on_the_hour(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = {
            "start": datetime.datetime(2026, 6, 1, 10, 0),
            "subject": "Sam sync", "organizer": "Alex Morgan",
        }
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_first_meeting_phrase()
        # On-the-hour morning meeting drops ":00" → "Sam sync at 10".
        self.assertEqual(out, "Sam sync at 10")

    def test_first_meeting_none(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = None
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_first_meeting_phrase(), "")

    # ── _attention_clause (pure) ─────────────────────────────────────────
    def test_attention_clause_counts(self):
        a = self.mod._attention_clause
        self.assertEqual(a([]), "")
        self.assertIn("One item requires your attention: x.", a(["x"]))
        self.assertIn("Two things require your attention: x, and y.", a(["x", "y"]))
        three = a(["x", "y", "z"])
        self.assertIn("Three things require your attention", three)
        self.assertIn("x, y, and z.", three)

    # ── _compose_briefing (pure) ─────────────────────────────────────────
    def test_compose_briefing_groups_attention_items(self):
        parts = {"time": "7:42 AM", "weather": "64 degrees and clear",
                 "teams": "one new Teams chat from Sam",
                 "print": "the H2D finished your bracket print",
                 "anomalies": "", "meeting": "a sync at 10",
                 "claude": "Claude Code merged 3 improvements", "headline": "Markets up"}
        out = self.mod._compose_briefing(parts)
        self.assertTrue(out.startswith("[intent:briefing] Good morning, sir."))
        self.assertIn("It's 7:42 AM, 64 degrees and clear.", out)
        self.assertIn("Two things require your attention", out)
        self.assertIn("You have a sync at 10.", out)
        self.assertIn("Overnight, Claude Code merged 3 improvements.", out)
        self.assertIn("In the news, Markets up.", out)

    def test_compose_briefing_minimal(self):
        parts = {k: "" for k in ("time", "weather", "meeting", "claude",
                                 "print", "teams", "anomalies", "headline")}
        out = self.mod._compose_briefing(parts)
        self.assertEqual(out, "[intent:briefing] Good morning, sir.")

    # ── _estimate_tts_seconds + _compose_within_budget ───────────────────
    def test_estimate_tts_strips_intent_tag(self):
        # 30 chars of body at 15 chars/s → 2.0 s. Tag not counted.
        secs = self.mod._estimate_tts_seconds("[intent:briefing] " + ("x" * 30))
        self.assertAlmostEqual(secs, 2.0, places=3)

    def test_compose_within_budget_drops_low_priority(self):
        # Force a tiny budget so everything droppable gets dropped, but the
        # three attention items (teams/print/anomalies) must survive.
        long = "x" * 200
        parts = {"time": "7 AM", "weather": long, "meeting": long,
                 "claude": long, "print": "print done", "teams": "teams ping",
                 "anomalies": "gpu anomaly", "headline": long}
        with mock.patch.object(self.mod, "TTS_BUDGET_SECONDS", 8.0):
            out = self.mod._compose_within_budget(parts)
        # Attention spine retained.
        self.assertIn("Three things require your attention", out)
        self.assertIn("teams ping", out)
        self.assertIn("print done", out)
        # Droppable long sections gone.
        self.assertNotIn("In the news", out)

    # ── _count_overnight_merges (log parsing) ────────────────────────────
    def test_count_overnight_merges_counts_decrements(self):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")  # in-window (now)
        log = (
            f"=== upgrade loop started {ts} ===\n"
            "[loop] iteration 1 of 5 — 5 task(s) remain\n"
            "[loop] iteration 2 of 5 — 3 task(s) remain\n"   # -2
            "[loop] iteration 3 of 5 — 2 task(s) remain\n"   # -1
        )
        m = mock.mock_open(read_data=log.encode())
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os.path, "getsize", return_value=len(log)), \
             mock.patch("builtins.open", m):
            n = self.mod._count_overnight_merges()
        self.assertEqual(n, 3)   # (5→3) + (3→2)

    def test_count_overnight_merges_missing_log(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._count_overnight_merges(), 0)

    def test_section_claude_code_phrase(self):
        with mock.patch.object(self.mod, "_count_overnight_merges", return_value=3):
            self.assertEqual(self.mod._section_claude_code_phrase(),
                             "Claude Code merged 3 improvements")
        with mock.patch.object(self.mod, "_count_overnight_merges", return_value=1):
            self.assertEqual(self.mod._section_claude_code_phrase(),
                             "Claude Code merged one improvement")
        with mock.patch.object(self.mod, "_count_overnight_merges", return_value=0):
            self.assertEqual(self.mod._section_claude_code_phrase(), "")

    # ── _section_overnight_anomalies ─────────────────────────────────────
    def test_overnight_anomalies_gpu_and_disk(self):
        history = [{"ts": time.time(),
                    "probes": {"gpu": {"ok": False, "severity": "HIGH", "error": "ECC"},
                               "disk": {"ok": False, "severity": "MEDIUM", "error": "SMART"}}}]
        import json
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(history))):
            out = self.mod._section_overnight_anomalies()
        self.assertEqual(out, "GPU and disk anomalies overnight")

    def test_overnight_anomalies_skips_low_severity(self):
        history = [{"ts": time.time(),
                    "probes": {"gpu": {"ok": False, "severity": "LOW", "error": "blip"}}}]
        import json
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(history))):
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    def test_overnight_anomalies_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._section_overnight_anomalies(), "")

    # ── _silence_hours_since_last_speech + suppression ───────────────────
    def test_silence_hours_prefers_pre_wake(self):
        fake_bc = mock.MagicMock()
        fake_bc._pre_wake_silence_seconds = [8 * 3600]  # 8h
        import sys
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertAlmostEqual(self.mod._silence_hours_since_last_speech(), 8.0, places=2)

    def test_already_fired_today(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod, "_load_state", return_value={"last_fired_date": today}):
            self.assertTrue(self.mod._arrival_already_fired_today())
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            self.assertFalse(self.mod._arrival_already_fired_today())

    # ── _fire_from_chain silence gate ────────────────────────────────────
    def test_fire_from_chain_blocked_by_short_silence(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech", return_value=2.0), \
             mock.patch.object(self.mod, "_fire_arrival") as fire:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "")
        fire.assert_not_called()

    def test_fire_from_chain_fires_when_silent_enough(self):
        with mock.patch.object(self.mod, "_arrival_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech", return_value=9.0), \
             mock.patch.object(self.mod.time, "sleep"), \
             mock.patch.object(self.mod, "_fire_arrival", return_value="briefing!") as fire:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "briefing!")
        fire.assert_called_once()

    def test_fire_from_chain_degrades_open_when_silence_unknown(self):
        # silence_h None (bobert unreachable) → fire anyway.
        with mock.patch.object(self.mod, "_arrival_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_silence_hours_since_last_speech", return_value=None), \
             mock.patch.object(self.mod.time, "sleep"), \
             mock.patch.object(self.mod, "_fire_arrival", return_value="briefing!") as fire:
            out = self.mod._fire_from_chain("chain")
        self.assertEqual(out, "briefing!")
        fire.assert_called_once()

    # ── morning_arrival action ───────────────────────────────────────────
    def test_action_returns_built_text(self):
        mod, actions = load_skill_isolated("morning_arrival")
        with mock.patch.object(mod, "_build_briefing", return_value="[intent:briefing] Good morning."), \
             mock.patch.object(mod, "_enqueue_speech"), \
             mock.patch.object(mod, "_mark_fired"):
            out = actions["morning_arrival"]("")
        self.assertIn("Good morning", out)

    def test_action_no_content_message(self):
        mod, actions = load_skill_isolated("morning_arrival")
        with mock.patch.object(mod, "_build_briefing", return_value=""), \
             mock.patch.object(mod, "_enqueue_speech"), \
             mock.patch.object(mod, "_mark_fired"):
            out = actions["morning_arrival"]("")
        self.assertIn("no content", out.lower())


if __name__ == "__main__":
    unittest.main()
