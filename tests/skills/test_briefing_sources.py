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
  • the meeting chain returning the first non-None source,
  • config resolution (bobert_companion attr lookup, import failure),
  • atomic-write / safe-load JSON round-trips,
  • _resolve_location (config / cached-geo / ipapi / stale-fallback),
  • the full Outlook-COM path with fake pythoncom + win32com injected,
  • the Microsoft Graph calendarView body with urlopen mocked,
  • the Google-Calendar ICS fetch + VEVENT parsing.

All network I/O (urllib.request.urlopen) and disk caches are mocked /
redirected — nothing hits wttr.in, ipapi, open-meteo, or the real cache
files. Datetime-sensitive tests freeze the module clock so the today /
tomorrow window slicing is host-TZ-agnostic.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules and restore prior
    state — including absence — on exit. For dotted names (``win32com.client``)
    the leaf is ALSO set as an attribute on its already-imported parent package,
    because ``import a.b.c`` resolves the leaf via ``getattr(parent, leaf)``
    when the parent is a real package. Mirrors the isolation contract used by
    tests/skills/test_self_diagnostic.py."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


# A fixed midday reference so window-sensitive meeting tests never straddle
# midnight on a late-running CI host (the failure mode the weather_briefing
# suite's _frozen_now guards against).
_FROZEN_NOW = datetime.datetime(2026, 6, 1, 12, 0, 0)


@contextlib.contextmanager
def _frozen_now(mod, when=_FROZEN_NOW):
    """Freeze the SKILL module's ``datetime.datetime.now()`` to ``when`` so the
    today/tomorrow meeting-window arithmetic is deterministic and host-clock /
    timezone agnostic. A datetime subclass keeps .date()/.replace()/+timedelta
    working for the rest of the code path."""
    class _Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return when

    patched = types.SimpleNamespace(datetime=_Frozen, timedelta=datetime.timedelta,
                                    time=datetime.time, timezone=datetime.timezone)
    with mock.patch.object(mod, "datetime", patched):
        yield when


def _fake_response(payload):
    """Build a context-manager stand-in for urllib.request.urlopen()."""
    body = json.dumps(payload).encode("utf-8") if not isinstance(payload, (bytes, str)) \
        else (payload.encode("utf-8") if isinstance(payload, str) else payload)
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

    def test_parse_ics_utc_z_malformed_returns_none(self):
        # Ends with Z but isn't a valid YYYYMMDDTHHMMSSZ → ValueError → None.
        self.assertIsNone(self.mod._parse_ics_dtstart("2026Z"))

    # ── _weather_from_wttr (urlopen mocked) ──────────────────────────────
    def test_weather_from_wttr_parses_payload(self):
        payload = {"current_condition": [
            {"temp_C": "18", "weatherDesc": [{"value": "Overcast"}]}]}
        with mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            out = self.mod._weather_from_wttr()
        self.assertEqual(out["temp_c"], 18)
        self.assertEqual(out["desc"], "overcast")

    def test_weather_from_wttr_missing_desc_is_blank(self):
        # weatherDesc omitted → desc falls back to "" without raising.
        payload = {"current_condition": [{"temp_C": "7"}]}
        with mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            out = self.mod._weather_from_wttr()
        self.assertEqual(out["temp_c"], 7)
        self.assertEqual(out["desc"], "")

    # ── _weather_from_open_meteo ─────────────────────────────────────────
    def test_weather_from_open_meteo_uses_code_map(self):
        payload = {"current": {"temperature_2m": 14.6, "weather_code": 3}}
        with mock.patch.object(self.mod, "_resolve_location", return_value=(51.5, -0.1)), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            out = self.mod._weather_from_open_meteo()
        self.assertEqual(out["temp_c"], 15)        # rounded
        self.assertEqual(out["desc"], "overcast")  # code 3

    def test_weather_from_open_meteo_unknown_code_blank_desc(self):
        payload = {"current": {"temperature_2m": 10.0, "weather_code": 999}}
        with mock.patch.object(self.mod, "_resolve_location", return_value=(1.0, 2.0)), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            out = self.mod._weather_from_open_meteo()
        self.assertEqual(out["temp_c"], 10)
        self.assertEqual(out["desc"], "")          # code not in map

    def test_weather_from_open_meteo_none_without_location(self):
        with mock.patch.object(self.mod, "_resolve_location", return_value=None):
            self.assertIsNone(self.mod._weather_from_open_meteo())

    def test_weather_from_open_meteo_none_when_temp_missing(self):
        payload = {"current": {"weather_code": 0}}
        with mock.patch.object(self.mod, "_resolve_location", return_value=(1.0, 2.0)), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
            self.assertIsNone(self.mod._weather_from_open_meteo())

    def test_weather_from_open_meteo_none_when_current_absent(self):
        payload = {}      # no "current" key at all → {} → temp missing → None
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

    def test_weather_cache_fresh_not_stale(self):
        # ts == now → age ~0 → not stale; desc None coalesces to "".
        with mock.patch.object(self.mod.time, "time", return_value=5000.0), \
             mock.patch.object(self.mod, "_safe_load_json",
                               return_value={"temp_c": 8, "desc": None, "ts": 5000.0}):
            out = self.mod._weather_from_cache()
        self.assertFalse(out["stale"])
        self.assertEqual(out["desc"], "")

    def test_weather_cache_none_when_no_payload(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value=None):
            self.assertIsNone(self.mod._weather_from_cache())

    def test_weather_cache_none_when_temp_missing(self):
        with mock.patch.object(self.mod, "_safe_load_json",
                               return_value={"desc": "x", "ts": 1.0}):
            self.assertIsNone(self.mod._weather_from_cache())

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

    def test_meeting_chain_outlook_wins_when_first(self):
        meeting = {"start": datetime.datetime.now(), "subject": "Early", "organizer": ""}
        with mock.patch.object(self.mod, "_meeting_from_outlook", return_value=meeting), \
             mock.patch.object(self.mod, "_meeting_from_graph") as graph, \
             mock.patch.object(self.mod, "_meeting_from_google_ics") as ics:
            out = self.mod.get_first_meeting_data("today")
        self.assertEqual(out["source"], "outlook")
        graph.assert_not_called()       # short-circuits on first success
        ics.assert_not_called()

    # ── _meeting_from_graph token gating ─────────────────────────────────
    def test_graph_skips_without_token(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value=None):
            self.assertIsNone(self.mod._meeting_from_graph("today"))

    def test_graph_skips_expired_token(self):
        with mock.patch.object(self.mod, "_safe_load_json",
                               return_value={"access_token": "x", "expires_at": 1.0}):
            self.assertIsNone(self.mod._meeting_from_graph("today"))


# ─────────────────────────────────────────────────────────────────────────
# Config + small-IO helpers
# ─────────────────────────────────────────────────────────────────────────
class ConfigAndIOTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("briefing_sources")

    def test_config_reads_attr_from_fake_bobert(self):
        fake_bc = types.ModuleType("bobert_companion")
        fake_bc.GOOGLE_CALENDAR_ICS_URL = "https://example.invalid/cal.ics"
        with inject_modules(bobert_companion=fake_bc):
            self.assertEqual(
                self.mod._config("GOOGLE_CALENDAR_ICS_URL", "fallback"),
                "https://example.invalid/cal.ics")

    def test_config_returns_default_when_attr_absent(self):
        fake_bc = types.ModuleType("bobert_companion")  # no attr set
        with inject_modules(bobert_companion=fake_bc):
            self.assertEqual(self.mod._config("MISSING_KEY", 42), 42)

    def test_config_returns_default_when_import_fails(self):
        # Force importlib.import_module to raise so the except branch runs.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no monolith")):
            self.assertEqual(self.mod._config("ANYTHING", "dflt"), "dflt")

    def test_atomic_write_and_safe_load_round_trip(self):
        tmp = tempfile.mkdtemp(prefix="briefsrc_io_")
        self.addCleanup(lambda: _rmtree(tmp))
        path = os.path.join(tmp, "payload.json")
        self.mod._atomic_write_json(path, {"a": 1, "b": "two"})
        self.assertEqual(self.mod._safe_load_json(path), {"a": 1, "b": "two"})

    def test_atomic_write_cleans_up_temp_on_failure(self):
        tmp = tempfile.mkdtemp(prefix="briefsrc_io_")
        self.addCleanup(lambda: _rmtree(tmp))
        path = os.path.join(tmp, "x.json")
        # json.dump raises on a non-serialisable object → unlink temp + re-raise.
        with self.assertRaises(TypeError):
            self.mod._atomic_write_json(path, {"bad": object()})
        # No stray .tmp left behind.
        leftovers = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_safe_load_json_missing_file_is_none(self):
        self.assertIsNone(self.mod._safe_load_json(
            os.path.join(tempfile.gettempdir(), "definitely_not_here_briefsrc.json")))

    def test_safe_load_json_bad_json_is_none(self):
        tmp = tempfile.mkdtemp(prefix="briefsrc_io_")
        self.addCleanup(lambda: _rmtree(tmp))
        path = os.path.join(tmp, "broken.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertIsNone(self.mod._safe_load_json(path))

    def test_save_weather_cache_writes_expected_shape(self):
        captured = {}
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=lambda p, payload: captured.update(payload)), \
             mock.patch.object(self.mod.time, "time", return_value=123.0):
            self.mod._save_weather_cache({"temp_c": 11, "desc": "rain", "source": "wttr"})
        self.assertEqual(captured["temp_c"], 11)
        self.assertEqual(captured["desc"], "rain")
        self.assertEqual(captured["source"], "wttr")
        self.assertEqual(captured["ts"], 123.0)

    def test_save_weather_cache_swallows_write_error(self):
        # A failing atomic write must not propagate (logged + swallowed).
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._save_weather_cache({"temp_c": 1, "desc": ""})  # no raise


# ─────────────────────────────────────────────────────────────────────────
# _resolve_location fallback ladder
# ─────────────────────────────────────────────────────────────────────────
class ResolveLocationTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("briefing_sources")

    def test_uses_configured_lat_lon(self):
        def _cfg(name, default):
            return {"OPEN_METEO_LAT": "40.0", "OPEN_METEO_LON": "-73.5"}.get(name, default)
        with mock.patch.object(self.mod, "_config", side_effect=_cfg):
            self.assertEqual(self.mod._resolve_location(), (40.0, -73.5))

    def test_config_lat_lon_non_numeric_falls_through_to_cache(self):
        def _cfg(name, default):
            return {"OPEN_METEO_LAT": "bogus", "OPEN_METEO_LON": "also"}.get(name, default)
        cached = {"lat": 1.5, "lon": 2.5, "ts": 9000.0}
        with mock.patch.object(self.mod, "_config", side_effect=_cfg), \
             mock.patch.object(self.mod, "_safe_load_json", return_value=cached), \
             mock.patch.object(self.mod.time, "time", return_value=9000.0):
            self.assertEqual(self.mod._resolve_location(), (1.5, 2.5))

    def test_uses_fresh_cached_geo(self):
        cached = {"lat": 12.0, "lon": 34.0, "ts": 1000.0}
        with mock.patch.object(self.mod, "_config", return_value=None), \
             mock.patch.object(self.mod, "_safe_load_json", return_value=cached), \
             mock.patch.object(self.mod.time, "time", return_value=1000.0):
            self.assertEqual(self.mod._resolve_location(), (12.0, 34.0))

    def test_falls_back_to_ipapi_when_no_cache(self):
        payload = {"latitude": 51.51, "longitude": -0.13}
        with mock.patch.object(self.mod, "_config", return_value=None), \
             mock.patch.object(self.mod, "_safe_load_json", return_value=None), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)), \
             mock.patch.object(self.mod, "_atomic_write_json") as wrote, \
             mock.patch.object(self.mod.time, "time", return_value=2000.0):
            lat, lon = self.mod._resolve_location()
        self.assertAlmostEqual(lat, 51.51)
        self.assertAlmostEqual(lon, -0.13)
        wrote.assert_called_once()   # fresh lookup is cached for 30 days

    def test_ipapi_write_failure_is_swallowed(self):
        payload = {"latitude": 1.0, "longitude": 2.0}
        with mock.patch.object(self.mod, "_config", return_value=None), \
             mock.patch.object(self.mod, "_safe_load_json", return_value=None), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("ro fs")), \
             mock.patch.object(self.mod.time, "time", return_value=2000.0):
            # Write failure must not lose the freshly-resolved coordinates.
            self.assertEqual(self.mod._resolve_location(), (1.0, 2.0))

    def test_stale_cache_used_when_ipapi_fails(self):
        # Cache too old for the fresh-window guard, but ipapi also down →
        # the stale cached value is the last resort.
        stale = {"lat": 7.0, "lon": 8.0, "ts": 0.0}
        with mock.patch.object(self.mod, "_config", return_value=None), \
             mock.patch.object(self.mod, "_safe_load_json", return_value=stale), \
             mock.patch("urllib.request.urlopen", side_effect=OSError("offline")), \
             mock.patch.object(self.mod.time, "time",
                               return_value=self.mod._GEO_CACHE_MAX_AGE_SECONDS + 10):
            self.assertEqual(self.mod._resolve_location(), (7.0, 8.0))

    def test_none_when_ipapi_fails_and_no_cache(self):
        with mock.patch.object(self.mod, "_config", return_value=None), \
             mock.patch.object(self.mod, "_safe_load_json", return_value=None), \
             mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertIsNone(self.mod._resolve_location())

    def test_corrupt_stale_cache_returns_none_on_ipapi_failure(self):
        # Cache present but missing lat/lon keys → KeyError swallowed → None.
        with mock.patch.object(self.mod, "_config", return_value=None), \
             mock.patch.object(self.mod, "_safe_load_json", return_value={"ts": 0.0}), \
             mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertIsNone(self.mod._resolve_location())

    def test_fresh_cache_corrupt_falls_through_to_ipapi(self):
        # Cache is FRESH (within the 30-day window) but lacks lat/lon → the
        # inner KeyError is swallowed and we proceed to the ipapi lookup.
        fresh_corrupt = {"ts": 5000.0}   # fresh but no lat/lon
        payload = {"latitude": 3.3, "longitude": 4.4}
        with mock.patch.object(self.mod, "_config", return_value=None), \
             mock.patch.object(self.mod, "_safe_load_json", return_value=fresh_corrupt), \
             mock.patch.object(self.mod.time, "time", return_value=5000.0), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)), \
             mock.patch.object(self.mod, "_atomic_write_json"):
            self.assertEqual(self.mod._resolve_location(), (3.3, 4.4))


# ─────────────────────────────────────────────────────────────────────────
# Outlook COM path (fake pythoncom + win32com.client injected)
# ─────────────────────────────────────────────────────────────────────────
class _FakeStart:
    """An appointment .Start with a .Format attr → the COM path rebuilds a
    naive datetime from its y/m/d/h/m fields."""
    def __init__(self, dt):
        self.year, self.month, self.day = dt.year, dt.month, dt.day
        self.hour, self.minute = dt.hour, dt.minute

    def Format(self, *_a):  # noqa: N802 (COM-style name)
        return "formatted"


class _FakeAppt:
    def __init__(self, start, subject="", organizer="", raises=False):
        self.Start = start
        self.Subject = subject
        self.Organizer = organizer
        self._raises = raises

    def __getattr__(self, name):
        # Only used if code reaches for something unexpected.
        raise AttributeError(name)


def _make_outlook(appts, restrict_raises=False):
    """Build fake pythoncom + win32com.client modules whose Dispatch chain
    yields ``appts`` from the calendar's restricted Items collection."""
    pythoncom = types.ModuleType("pythoncom")
    pythoncom.CoInitialize = mock.MagicMock()
    pythoncom.CoUninitialize = mock.MagicMock()

    items = mock.MagicMock()
    items.IncludeRecurrences = False
    if restrict_raises:
        items.Restrict.side_effect = RuntimeError("bad restriction")
    else:
        items.Restrict.return_value = appts
    # When Restrict raises, the code iterates `items` directly.
    items.__iter__ = mock.Mock(return_value=iter(appts))

    calendar = mock.MagicMock()
    calendar.Items = items
    namespace = mock.MagicMock()
    namespace.GetDefaultFolder.return_value = calendar
    outlook = mock.MagicMock()
    outlook.GetNamespace.return_value = namespace

    win32com = types.ModuleType("win32com")
    client = types.ModuleType("win32com.client")
    client.Dispatch = mock.MagicMock(return_value=outlook)
    win32com.client = client
    return pythoncom, win32com, client


class OutlookComTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("briefing_sources")

    def test_outlook_missing_pywin32_returns_none(self):
        # No pythoncom / win32com installed (the CI case) → the lazy import
        # raises → None. On a Windows dev box pythoncom IS installed, so we
        # force the import to fail via a raising __import__ rather than relying
        # on the dep's absence (which would let the real COM stack load).
        real_import = __import__

        def _fake_import(name, *a, **k):
            if name.split(".")[0] in ("pythoncom", "win32com"):
                raise ImportError(f"blocked: {name}")
            return real_import(name, *a, **k)

        saved = {n: sys.modules.pop(n, _SENTINEL)
                 for n in ("pythoncom", "win32com", "win32com.client")}
        try:
            with mock.patch("builtins.__import__", side_effect=_fake_import):
                self.assertIsNone(self.mod._meeting_from_outlook("today"))
        finally:
            for n, v in saved.items():
                if v is not _SENTINEL:
                    sys.modules[n] = v

    def test_outlook_returns_first_in_window(self):
        soon = _FROZEN_NOW + datetime.timedelta(hours=1)
        appt = _FakeAppt(_FakeStart(soon), subject="Kickoff", organizer="Dana")
        pythoncom, win32com, client = _make_outlook([appt])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            out = self.mod._meeting_from_outlook("today")
        self.assertEqual(out["subject"], "Kickoff")
        self.assertEqual(out["organizer"], "Dana")
        self.assertEqual((out["start"].hour, out["start"].minute),
                         (soon.hour, soon.minute))
        pythoncom.CoInitialize.assert_called_once()
        pythoncom.CoUninitialize.assert_called_once()

    def test_outlook_start_without_format_used_directly(self):
        # .Start lacking a .Format attr is taken as a real datetime as-is.
        soon = _FROZEN_NOW + datetime.timedelta(hours=2)
        appt = _FakeAppt(soon, subject="Direct", organizer="")
        pythoncom, win32com, client = _make_outlook([appt])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            out = self.mod._meeting_from_outlook("today")
        self.assertEqual(out["subject"], "Direct")
        self.assertEqual(out["start"], soon)

    def test_outlook_skips_events_outside_window(self):
        past = _FROZEN_NOW - datetime.timedelta(hours=3)
        future = _FROZEN_NOW + datetime.timedelta(hours=1)
        a_past = _FakeAppt(_FakeStart(past), subject="Past")
        a_future = _FakeAppt(_FakeStart(future), subject="Future")
        pythoncom, win32com, client = _make_outlook([a_past, a_future])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            out = self.mod._meeting_from_outlook("today")
        self.assertEqual(out["subject"], "Future")   # past one skipped

    def test_outlook_no_events_returns_none(self):
        pythoncom, win32com, client = _make_outlook([])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            self.assertIsNone(self.mod._meeting_from_outlook("today"))

    def test_outlook_restrict_failure_falls_back_to_full_iteration(self):
        soon = _FROZEN_NOW + datetime.timedelta(hours=1)
        appt = _FakeAppt(_FakeStart(soon), subject="Fallback")
        pythoncom, win32com, client = _make_outlook([appt], restrict_raises=True)
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            out = self.mod._meeting_from_outlook("today")
        self.assertEqual(out["subject"], "Fallback")

    def test_outlook_dispatch_failure_returns_none(self):
        pythoncom, win32com, client = _make_outlook([])
        client.Dispatch.side_effect = RuntimeError("Outlook not running")
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            self.assertIsNone(self.mod._meeting_from_outlook("today"))
        # CoUninitialize still runs in the finally block.
        pythoncom.CoUninitialize.assert_called_once()

    def test_outlook_per_appt_error_is_skipped(self):
        # First appt blows up on attribute access; second is clean.
        soon = _FROZEN_NOW + datetime.timedelta(hours=1)
        bad = mock.MagicMock()
        type(bad).Start = mock.PropertyMock(side_effect=RuntimeError("COM glitch"))
        good = _FakeAppt(_FakeStart(soon), subject="Recovered")
        pythoncom, win32com, client = _make_outlook([bad, good])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            out = self.mod._meeting_from_outlook("today")
        self.assertEqual(out["subject"], "Recovered")

    def test_outlook_coinitialize_failure_is_tolerated(self):
        soon = _FROZEN_NOW + datetime.timedelta(hours=1)
        appt = _FakeAppt(_FakeStart(soon), subject="Still works")
        pythoncom, win32com, client = _make_outlook([appt])
        pythoncom.CoInitialize.side_effect = RuntimeError("already init")
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            out = self.mod._meeting_from_outlook("today")
        self.assertEqual(out["subject"], "Still works")

    def test_outlook_couninitialize_failure_is_tolerated(self):
        # CoUninitialize raising inside the finally block must be swallowed so
        # a clean result still returns.
        soon = _FROZEN_NOW + datetime.timedelta(hours=1)
        appt = _FakeAppt(_FakeStart(soon), subject="Clean")
        pythoncom, win32com, client = _make_outlook([appt])
        pythoncom.CoUninitialize.side_effect = RuntimeError("uninit boom")
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}), _frozen_now(self.mod):
            out = self.mod._meeting_from_outlook("today")
        self.assertEqual(out["subject"], "Clean")


# ─────────────────────────────────────────────────────────────────────────
# Microsoft Graph calendarView body (urlopen mocked)
# ─────────────────────────────────────────────────────────────────────────
class GraphBodyTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("briefing_sources")

    def _token(self):
        # Far-future expiry so the token-gate passes.
        return {"access_token": "tok", "expires_at": 9_999_999_999.0}

    def test_graph_parses_first_event(self):
        body = {"value": [{
            "subject": "Roadmap",
            "start": {"dateTime": "2026-06-15T09:30:00.0000000"},
            "organizer": {"emailAddress": {"name": "Pat"}},
        }]}
        with mock.patch.object(self.mod, "_safe_load_json", return_value=self._token()), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(body)):
            out = self.mod._meeting_from_graph("today")
        self.assertEqual(out["subject"], "Roadmap")
        self.assertEqual(out["organizer"], "Pat")
        self.assertTrue(hasattr(out["start"], "hour"))

    def test_graph_empty_value_returns_none(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value=self._token()), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response({"value": []})):
            self.assertIsNone(self.mod._meeting_from_graph("today"))

    def test_graph_missing_value_key_returns_none(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value=self._token()), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response({})):
            self.assertIsNone(self.mod._meeting_from_graph("today"))

    def test_graph_http_error_returns_none(self):
        import urllib.error
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with mock.patch.object(self.mod, "_safe_load_json", return_value=self._token()), \
             mock.patch("urllib.request.urlopen", side_effect=err):
            self.assertIsNone(self.mod._meeting_from_graph("today"))

    def test_graph_generic_network_error_returns_none(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value=self._token()), \
             mock.patch("urllib.request.urlopen", side_effect=OSError("conn reset")):
            self.assertIsNone(self.mod._meeting_from_graph("today"))

    def test_graph_malformed_start_returns_none(self):
        # Event present but start.dateTime missing → inner parse fails → None.
        body = {"value": [{"subject": "NoStart", "organizer": {}}]}
        with mock.patch.object(self.mod, "_safe_load_json", return_value=self._token()), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(body)):
            self.assertIsNone(self.mod._meeting_from_graph("today"))

    def test_graph_organizer_missing_is_blank(self):
        body = {"value": [{
            "subject": "Solo",
            "start": {"dateTime": "2026-06-15T08:00:00"},
            # organizer omitted entirely → organizer "" via swallowed except
        }]}
        with mock.patch.object(self.mod, "_safe_load_json", return_value=self._token()), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(body)):
            out = self.mod._meeting_from_graph("today")
        self.assertEqual(out["subject"], "Solo")
        self.assertEqual(out["organizer"], "")

    def test_graph_token_without_expiry_still_used(self):
        # expires_at falsy/absent → expiry check skipped, request proceeds.
        body = {"value": [{
            "subject": "NoExpiry",
            "start": {"dateTime": "2026-06-15T07:00:00"},
            "organizer": {"emailAddress": {"name": "Lee"}},
        }]}
        with mock.patch.object(self.mod, "_safe_load_json",
                               return_value={"access_token": "tok"}), \
             mock.patch("urllib.request.urlopen", return_value=_fake_response(body)):
            out = self.mod._meeting_from_graph("today")
        self.assertEqual(out["subject"], "NoExpiry")


# ─────────────────────────────────────────────────────────────────────────
# Google Calendar ICS fetch + VEVENT parsing
# ─────────────────────────────────────────────────────────────────────────
class GoogleIcsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("briefing_sources")

    def _ics(self, *vevents):
        return ("BEGIN:VCALENDAR\r\n" + "".join(vevents) + "END:VCALENDAR\r\n")

    def _vevent(self, dtstart, summary=None, organizer_cn=None):
        lines = ["BEGIN:VEVENT\r\n", f"DTSTART:{dtstart}\r\n"]
        if summary is not None:
            lines.append(f"SUMMARY:{summary}\r\n")
        if organizer_cn is not None:
            lines.append(f"ORGANIZER;CN={organizer_cn}:mailto:x@y.invalid\r\n")
        lines.append("END:VEVENT\r\n")
        return "".join(lines)

    def test_ics_no_url_configured_returns_none(self):
        with mock.patch.object(self.mod, "_config", return_value=""):
            self.assertIsNone(self.mod._meeting_from_google_ics("today"))

    def test_ics_fetch_failure_returns_none(self):
        with mock.patch.object(self.mod, "_config", return_value="https://x.invalid/c.ics"), \
             mock.patch("urllib.request.urlopen", side_effect=OSError("dns")):
            self.assertIsNone(self.mod._meeting_from_google_ics("today"))

    @staticmethod
    def _resp(text):
        resp = mock.MagicMock()
        resp.read.return_value = text.encode("utf-8")
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        return resp

    def test_ics_picks_earliest_in_window(self):
        later = (_FROZEN_NOW + datetime.timedelta(hours=4)).strftime("%Y%m%dT%H%M%S")
        sooner = (_FROZEN_NOW + datetime.timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
        text = self._ics(
            self._vevent(later, summary="Later", organizer_cn="Bob"),
            self._vevent(sooner, summary="Sooner", organizer_cn="Alice"),
        )
        with mock.patch.object(self.mod, "_config", return_value="https://x.invalid/c.ics"), \
             mock.patch("urllib.request.urlopen", return_value=self._resp(text)), \
             _frozen_now(self.mod):
            out = self.mod._meeting_from_google_ics("today")
        self.assertEqual(out["subject"], "Sooner")
        self.assertEqual(out["organizer"], "Alice")

    def test_ics_skips_events_outside_window(self):
        yesterday = (_FROZEN_NOW - datetime.timedelta(days=1)).strftime("%Y%m%dT%H%M%S")
        text = self._ics(self._vevent(yesterday, summary="Old"))
        with mock.patch.object(self.mod, "_config", return_value="https://x.invalid/c.ics"), \
             mock.patch("urllib.request.urlopen", return_value=self._resp(text)), \
             _frozen_now(self.mod):
            self.assertIsNone(self.mod._meeting_from_google_ics("today"))

    def test_ics_event_without_dtstart_is_skipped(self):
        good = (_FROZEN_NOW + datetime.timedelta(hours=2)).strftime("%Y%m%dT%H%M%S")
        text = ("BEGIN:VCALENDAR\r\n"
                "BEGIN:VEVENT\r\nSUMMARY:NoStart\r\nEND:VEVENT\r\n"
                + self._vevent(good, summary="HasStart")
                + "END:VCALENDAR\r\n")
        with mock.patch.object(self.mod, "_config", return_value="https://x.invalid/c.ics"), \
             mock.patch("urllib.request.urlopen", return_value=self._resp(text)), \
             _frozen_now(self.mod):
            out = self.mod._meeting_from_google_ics("today")
        self.assertEqual(out["subject"], "HasStart")

    def test_ics_event_with_no_summary_is_blank(self):
        # DTSTART present, SUMMARY absent → subject "" (the regex misses).
        good = (_FROZEN_NOW + datetime.timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
        text = self._ics(self._vevent(good))   # no summary/organizer
        with mock.patch.object(self.mod, "_config", return_value="https://x.invalid/c.ics"), \
             mock.patch("urllib.request.urlopen", return_value=self._resp(text)), \
             _frozen_now(self.mod):
            out = self.mod._meeting_from_google_ics("today")
        self.assertEqual(out["subject"], "")
        self.assertEqual(out["organizer"], "")

    def test_ics_unfolds_continuation_lines(self):
        # A folded SUMMARY (continuation line begins with a space) must be
        # stitched back together before the SUMMARY regex runs.
        when = (_FROZEN_NOW + datetime.timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
        text = ("BEGIN:VCALENDAR\r\n"
                "BEGIN:VEVENT\r\n"
                f"DTSTART:{when}\r\n"
                "SUMMARY:Quarterly plan\r\n review\r\n"
                "END:VEVENT\r\n"
                "END:VCALENDAR\r\n")
        with mock.patch.object(self.mod, "_config", return_value="https://x.invalid/c.ics"), \
             mock.patch("urllib.request.urlopen", return_value=self._resp(text)), \
             _frozen_now(self.mod):
            out = self.mod._meeting_from_google_ics("today")
        self.assertEqual(out["subject"], "Quarterly planreview")

    def test_ics_no_vevents_returns_none(self):
        text = "BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n"
        with mock.patch.object(self.mod, "_config", return_value="https://x.invalid/c.ics"), \
             mock.patch("urllib.request.urlopen", return_value=self._resp(text)), \
             _frozen_now(self.mod):
            self.assertIsNone(self.mod._meeting_from_google_ics("today"))


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
