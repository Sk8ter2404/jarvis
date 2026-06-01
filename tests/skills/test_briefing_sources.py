"""Logic tests for skills/briefing_sources.py.

briefing_sources is a HELPER module (no register()), shared by the morning /
daily briefing skills. It provides a fallback chain for weather (wttr →
open-meteo → cache) and the first meeting (outlook → graph → google-ics).

These tests target the pure / mockable logic:
  • Open-Meteo WMO weather-code → description map,
  • the today/tomorrow meeting window arithmetic,
  • ICS DTSTART parsing (all-day, UTC-Z, naive-local),
  • the open-meteo weather builder with urlopen mocked,
  • the weather chain's fallback ordering and cache stamping,
  • the meeting chain returning the first non-None source.

All network I/O (urllib.request.urlopen) and disk caches are mocked /
redirected — nothing hits wttr.in, ipapi, open-meteo, or the real cache
files.
"""
from __future__ import annotations

import datetime
import json
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_response(payload):
    """Build a context-manager stand-in for urllib.request.urlopen()."""
    body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__ = mock.Mock(return_value=resp)
    resp.__exit__ = mock.Mock(return_value=False)
    return resp


class BriefingSourcesTests(unittest.TestCase):
    def setUp(self):
        # briefing_sources has no register(); the harness still execs + returns it.
        self.mod, _ = load_skill_isolated("briefing_sources")

    # ── Open-Meteo code map ──────────────────────────────────────────────
    def test_open_meteo_descriptions_cover_key_codes(self):
        m = self.mod._OPEN_METEO_DESCRIPTIONS
        self.assertEqual(m[0], "clear")
        self.assertEqual(m[3], "overcast")
        self.assertEqual(m[95], "thunderstorms")
        self.assertEqual(m[75], "heavy snow")

    # ── _meeting_window ──────────────────────────────────────────────────
    def test_meeting_window_today_starts_now_ends_eod(self):
        start, end = self.mod._meeting_window("today")
        self.assertEqual((end.hour, end.minute, end.second), (23, 59, 59))
        self.assertLessEqual(start, end)
        self.assertEqual(start.date(), datetime.datetime.now().date())

    def test_meeting_window_tomorrow_is_full_next_day(self):
        start, end = self.mod._meeting_window("tomorrow")
        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).date()
        self.assertEqual(start.date(), tomorrow)
        self.assertEqual((start.hour, start.minute), (0, 0))
        self.assertEqual((end.hour, end.minute, end.second), (23, 59, 59))

    # ── _parse_ics_dtstart ───────────────────────────────────────────────
    def test_parse_ics_all_day(self):
        dt = self.mod._parse_ics_dtstart("20260615")
        self.assertEqual((dt.year, dt.month, dt.day), (2026, 6, 15))
        self.assertEqual((dt.hour, dt.minute), (0, 0))

    def test_parse_ics_naive_local(self):
        dt = self.mod._parse_ics_dtstart("20260615T143000")
        self.assertEqual((dt.hour, dt.minute, dt.second), (14, 30, 0))

    def test_parse_ics_utc_z_converts_to_local(self):
        # A Z-suffixed UTC value parses to a naive local datetime (offset-aware
        # → astimezone(None) → tz-stripped). We assert it round-trips back to
        # the same instant rather than a fixed wall-clock (test-host TZ-agnostic).
        dt = self.mod._parse_ics_dtstart("20260615T120000Z")
        self.assertIsNotNone(dt)
        back = dt.astimezone().astimezone(datetime.timezone.utc)
        self.assertEqual((back.hour, back.minute), (12, 0))

    def test_parse_ics_garbage_returns_none(self):
        self.assertIsNone(self.mod._parse_ics_dtstart("not-a-date"))

    # ── _weather_from_wttr (urlopen mocked) ──────────────────────────────
    def test_weather_from_wttr_parses_payload(self):
        payload = {"current_condition": [
            {"temp_C": "18", "weatherDesc": [{"value": "Overcast"}]}]}
        with mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            out = self.mod._weather_from_wttr()
        self.assertEqual(out["temp_c"], 18)
        self.assertEqual(out["desc"], "overcast")

    # ── _weather_from_open_meteo ─────────────────────────────────────────
    def test_weather_from_open_meteo_uses_code_map(self):
        payload = {"current": {"temperature_2m": 14.6, "weather_code": 3}}
        with mock.patch.object(self.mod, "_resolve_location", return_value=(51.5, -0.1)), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            out = self.mod._weather_from_open_meteo()
        self.assertEqual(out["temp_c"], 15)        # rounded
        self.assertEqual(out["desc"], "overcast")  # code 3

    def test_weather_from_open_meteo_none_without_location(self):
        with mock.patch.object(self.mod, "_resolve_location", return_value=None):
            self.assertIsNone(self.mod._weather_from_open_meteo())

    def test_weather_from_open_meteo_none_when_temp_missing(self):
        payload = {"current": {"weather_code": 0}}
        with mock.patch.object(self.mod, "_resolve_location", return_value=(1.0, 2.0)), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            self.assertIsNone(self.mod._weather_from_open_meteo())

    # ── get_weather_data chain ───────────────────────────────────────────
    def test_weather_chain_returns_first_success_and_stamps_source(self):
        with mock.patch.object(self.mod, "_weather_from_wttr",
                               return_value={"temp_c": 20, "desc": "clear"}), \
             mock.patch.object(self.mod, "_save_weather_cache") as save:
            out = self.mod.get_weather_data()
        self.assertEqual(out["source"], "wttr")
        save.assert_called_once()   # success writes the cache

    def test_weather_chain_falls_through_to_open_meteo(self):
        with mock.patch.object(self.mod, "_weather_from_wttr",
                               side_effect=RuntimeError("wttr down")), \
             mock.patch.object(self.mod, "_weather_from_open_meteo",
                               return_value={"temp_c": 9, "desc": "rain"}), \
             mock.patch.object(self.mod, "_save_weather_cache"):
            out = self.mod.get_weather_data()
        self.assertEqual(out["source"], "open-meteo")
        self.assertEqual(out["temp_c"], 9)

    def test_weather_chain_uses_cache_when_all_live_fail(self):
        with mock.patch.object(self.mod, "_weather_from_wttr", return_value=None), \
             mock.patch.object(self.mod, "_weather_from_open_meteo", return_value=None), \
             mock.patch.object(self.mod, "_weather_from_cache",
                               return_value={"temp_c": 5, "desc": "fog", "stale": True}):
            out = self.mod.get_weather_data()
        self.assertEqual(out["source"], "cache")
        self.assertTrue(out["stale"])

    def test_weather_chain_returns_none_when_everything_dead(self):
        with mock.patch.object(self.mod, "_weather_from_wttr", return_value=None), \
             mock.patch.object(self.mod, "_weather_from_open_meteo", return_value=None), \
             mock.patch.object(self.mod, "_weather_from_cache", return_value=None):
            self.assertIsNone(self.mod.get_weather_data())

    # ── _weather_from_cache staleness ────────────────────────────────────
    def test_weather_cache_marks_stale_beyond_window(self):
        old_ts = 1000.0  # ancient
        with mock.patch.object(self.mod, "_safe_load_json",
                               return_value={"temp_c": 12, "desc": "x", "ts": old_ts}):
            out = self.mod._weather_from_cache()
        self.assertTrue(out["stale"])
        self.assertEqual(out["temp_c"], 12)

    # ── get_first_meeting_data chain ─────────────────────────────────────
    def test_meeting_chain_returns_first_source(self):
        meeting = {"start": datetime.datetime.now(), "subject": "Standup",
                   "organizer": "Sam"}
        with mock.patch.object(self.mod, "_meeting_from_outlook", return_value=None), \
             mock.patch.object(self.mod, "_meeting_from_graph", return_value=meeting), \
             mock.patch.object(self.mod, "_meeting_from_google_ics", return_value=None):
            out = self.mod.get_first_meeting_data("today")
        self.assertEqual(out["source"], "graph")
        self.assertEqual(out["subject"], "Standup")

    def test_meeting_chain_none_when_no_source_has_events(self):
        with mock.patch.object(self.mod, "_meeting_from_outlook", return_value=None), \
             mock.patch.object(self.mod, "_meeting_from_graph", return_value=None), \
             mock.patch.object(self.mod, "_meeting_from_google_ics", return_value=None):
            self.assertIsNone(self.mod.get_first_meeting_data("today"))

    def test_meeting_chain_skips_raising_source(self):
        meeting = {"start": datetime.datetime.now(), "subject": "Sync", "organizer": ""}
        with mock.patch.object(self.mod, "_meeting_from_outlook",
                               side_effect=RuntimeError("COM blew up")), \
             mock.patch.object(self.mod, "_meeting_from_graph", return_value=None), \
             mock.patch.object(self.mod, "_meeting_from_google_ics", return_value=meeting):
            out = self.mod.get_first_meeting_data("today")
        self.assertEqual(out["source"], "google-ics")

    # ── _meeting_from_graph token gating ─────────────────────────────────
    def test_graph_skips_without_token(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value=None):
            self.assertIsNone(self.mod._meeting_from_graph("today"))

    def test_graph_skips_expired_token(self):
        with mock.patch.object(self.mod, "_safe_load_json",
                               return_value={"access_token": "x", "expires_at": 1.0}):
            self.assertIsNone(self.mod._meeting_from_graph("today"))


if __name__ == "__main__":
    unittest.main()
