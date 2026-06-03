"""Logic tests for skills/weather_briefing.py.

Covers the pure forecast logic — WMO category mapping, hour formatting, the
"hours falling on this day" slice, umbrella-alert phrasing, the two-hour
change detector's priority ladder (thunderstorm > precip-jump > temp-drop >
category-change), and the cooldown state read. It also drives the I/O-adjacent
layers with everything mocked: the Open-Meteo hourly fetch+parse (against a
fake urlopen returning canned JSON — no network), location resolution (against
a fake briefing_sources), the proactive speech-queue funnel, the cooldown
record + watcher-iteration logic, and register()'s action wiring + watcher-arm
branches. No test hits the network; the watcher thread is never actually run.

ISOLATION: the few module fakes (briefing_sources, a fake bobert_companion for
the speech funnel) live only inside an `inject_modules` with-block that
saves+restores sys.modules on exit — there are NO module-level sys.modules
writes. The weather skill's only persistent global is the on-disk state file,
which tests redirect to a temp path or mock _safe_load_json/_atomic_write_json
around, so the real weather_briefing_state.json is never touched.

CRITICAL: weather_briefing._config() does `importlib.import_module(
"bobert_companion")`, which — if it ran for real — would execute the ~14K-line
monolith and call sys.exit(0) at its early-boot singleton lock. Every test here
therefore either mocks _read_config / _config or injects a lightweight fake
bobert_companion; the real monolith is NEVER imported.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
import unittest
from datetime import datetime, timedelta
from unittest import mock

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring the prior
    state (including absence) on exit. weather_briefing imports briefing_sources
    and bobert_companion by bare name (after putting the skills dir / project
    root on sys.path), so a plain sys.modules entry is enough to intercept them
    — there is no real parent package to patch for these names."""
    saved: dict[str, object] = {}
    missing: set[str] = set()
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if saved[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name in mods:
            if name in missing:
                sys.modules.pop(name, None)
            elif saved[name] is not _SENTINEL:
                sys.modules[name] = saved[name]


def _fake_urlopen_returning(payload, *, raises=None):
    """Return a stand-in for urllib.request.urlopen yielding a context manager
    whose .read() returns `payload` (str) UTF-8 encoded. If `raises` is set, the
    opener raises it instead (network-failure path)."""
    class _Resp:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _opener(req, timeout=None):
        if raises is not None:
            raise raises
        data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        return _Resp(data)

    return _opener


def _hour(dt, *, prob=0, mm=0.0, code=0, temp=20.0):
    """Build one hourly-forecast bucket the way _fetch_hourly_forecast does."""
    return {
        "dt": dt, "hour_local": dt.hour, "temp_c": temp,
        "precip_prob": prob, "precip_mm": mm, "weather_code": code,
        "desc": "", "category": None,  # category filled by caller when needed
    }


@contextlib.contextmanager
def _frozen_now(mod, when):
    """Freeze the SKILL's datetime.now() to `when` (a real datetime) so the
    today/tomorrow slice logic is deterministic regardless of the test
    machine's wall clock or timezone — otherwise a now+2h fixture crosses
    midnight near end-of-day and the 'today' slice legitimately drops it (this
    failed CI at 22:45 UTC). A datetime subclass keeps .date()/.replace()/
    +timedelta working."""
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return when
    with mock.patch.object(mod, "datetime", _Frozen):
        yield


class WeatherBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    # ── _weather_category (pure) ─────────────────────────────────────────
    def test_weather_category_mapping(self):
        c = self.mod._weather_category
        self.assertEqual(c(0), "clear")
        self.assertEqual(c(2), "cloudy")
        self.assertEqual(c(3), "overcast")
        self.assertEqual(c(61), "rain")
        self.assertEqual(c(75), "snow")
        self.assertEqual(c(95), "thunderstorm")
        self.assertEqual(c(48), "fog")
        self.assertEqual(c(-1), "unknown")

    # ── _format_hour (pure) ──────────────────────────────────────────────
    def test_format_hour(self):
        f = self.mod._format_hour
        self.assertEqual(f(0), "12 AM")
        self.assertEqual(f(9), "9 AM")
        self.assertEqual(f(12), "12 PM")
        self.assertEqual(f(15), "3 PM")

    # ── _slice_for_day (pure) ────────────────────────────────────────────
    def test_slice_for_today_drops_past_hours(self):
        now = datetime(2026, 6, 1, 12, 0, 0)   # fixed midday → deterministic
        past = _hour(now - timedelta(hours=3))
        future = _hour(now + timedelta(hours=2))
        tomorrow = _hour(now + timedelta(days=1))
        with _frozen_now(self.mod, now):
            window = self.mod._slice_for_day([past, future, tomorrow], "today")
        self.assertIn(future, window)
        self.assertNotIn(past, window)
        self.assertNotIn(tomorrow, window)

    def test_slice_for_tomorrow(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        today_h = _hour(now + timedelta(hours=1))
        tmrw_h = _hour(now + timedelta(days=1))
        with _frozen_now(self.mod, now):
            window = self.mod._slice_for_day([today_h, tmrw_h], "tomorrow")
        self.assertEqual(window, [tmrw_h])

    # ── get_umbrella_alert ───────────────────────────────────────────────
    def _cfg(self, **over):
        base = {"enabled": True, "umbrella_prob": 50, "lookahead_h": 2,
                "sig_temp_drop_c": 5, "cooldown_h": 4, "poll_minutes": 30,
                "proactive": True}
        base.update(over)
        return base

    def test_umbrella_alert_rain(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        h = _hour(now + timedelta(hours=2), prob=80, code=61)
        h["category"] = "rain"
        with _frozen_now(self.mod, now), \
             mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[h]):
            out = self.mod.get_umbrella_alert("today")
        self.assertIn("umbrella", out.lower())
        self.assertIn("80%", out)
        self.assertIn("rain", out)

    def test_umbrella_alert_snow_phrasing(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        h = _hour(now + timedelta(hours=1), prob=70, code=73)
        h["category"] = "snow"
        with _frozen_now(self.mod, now), \
             mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[h]):
            out = self.mod.get_umbrella_alert("today")
        self.assertIn("snow", out.lower())
        self.assertIn("layering up", out.lower())

    def test_umbrella_alert_none_below_threshold(self):
        now = datetime.now()
        h = _hour(now + timedelta(hours=1), prob=10, code=61)
        h["category"] = "rain"
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[h]):
            self.assertEqual(self.mod.get_umbrella_alert("today"), "")

    def test_umbrella_alert_disabled(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg(enabled=False)):
            self.assertEqual(self.mod.get_umbrella_alert("today"), "")

    def test_umbrella_alert_no_forecast(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[]):
            self.assertEqual(self.mod.get_umbrella_alert("today"), "")

    # ── _detect_two_hour_change priority ladder ──────────────────────────
    # The detector picks `current` as the bucket straddling now (dt <= now <
    # dt+1h) and `window` as buckets in [now, now+lookahead]. We anchor the
    # baseline 30 min in the past (so it straddles now regardless of clock
    # drift) and put the change buckets comfortably inside the window.
    def _baseline(self, **over):
        cur = _hour(datetime.now() - timedelta(minutes=30), **over)
        cur["category"] = "clear"
        return cur

    def test_detect_thunderstorm_takes_priority(self):
        now = datetime.now()
        cur = self._baseline(prob=0, code=0)
        filler = _hour(now + timedelta(minutes=30), prob=0, code=0)
        filler["category"] = "clear"
        storm = _hour(now + timedelta(minutes=90), prob=60, code=95)
        storm["category"] = "thunderstorm"
        klass, msg = self.mod._detect_two_hour_change([cur, filler, storm], self._cfg())
        self.assertEqual(klass, "thunderstorm_incoming")
        self.assertIn("thunderstorms", msg.lower())

    def test_detect_precip_jump(self):
        now = datetime.now()
        cur = self._baseline(prob=5, code=0)
        filler = _hour(now + timedelta(minutes=20), prob=5, code=0)
        filler["category"] = "clear"
        wet = _hour(now + timedelta(minutes=90), prob=70, code=61)
        wet["category"] = "rain"
        klass, msg = self.mod._detect_two_hour_change([cur, filler, wet], self._cfg())
        self.assertEqual(klass, "precip_jump")
        self.assertIn("wetter", msg.lower())

    def test_detect_temp_drop_reported_in_fahrenheit(self):
        now = datetime.now()
        cur = self._baseline(prob=0, code=0, temp=20.0)
        filler = _hour(now + timedelta(minutes=20), prob=0, code=0, temp=20.0)
        filler["category"] = "clear"
        cold = _hour(now + timedelta(minutes=90), prob=0, code=0, temp=12.0)  # 8C drop
        cold["category"] = "clear"
        klass, msg = self.mod._detect_two_hour_change([cur, filler, cold], self._cfg())
        self.assertEqual(klass, "temp_drop")
        # 8 C drop → ~14 F. The message speaks Fahrenheit + a jacket nudge.
        self.assertIn("14 degrees", msg)
        self.assertIn("jacket", msg.lower())

    def test_detect_nothing_when_window_too_short(self):
        now = datetime.now()
        klass, msg = self.mod._detect_two_hour_change([_hour(now)], self._cfg())
        self.assertIsNone(klass)
        self.assertEqual(msg, "")

    # ── cooldown state read ──────────────────────────────────────────────
    def test_cooldown_active_within_window(self):
        import time
        recent = {"alerts": {"precip_jump": time.time() - 100}}
        with mock.patch.object(self.mod, "_safe_load_json", return_value=recent):
            self.assertTrue(self.mod._alert_cooldown_active("precip_jump", 3600))

    def test_cooldown_inactive_when_expired(self):
        import time
        old = {"alerts": {"precip_jump": time.time() - 99999}}
        with mock.patch.object(self.mod, "_safe_load_json", return_value=old):
            self.assertFalse(self.mod._alert_cooldown_active("precip_jump", 3600))

    def test_cooldown_inactive_when_never_fired(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value={}):
            self.assertFalse(self.mod._alert_cooldown_active("temp_drop", 3600))

    # ── weather_briefing action (closure binds to self.mod's globals) ────
    def test_action_prefers_umbrella(self):
        _, actions = load_skill_isolated("weather_briefing", utils=None)
        # The action reads module globals via name lookup, so patching the
        # SAME module object the action came from is what matters. Reload to
        # get a paired (mod, actions).
        mod, actions = load_skill_isolated("weather_briefing")
        with mock.patch.object(mod, "get_umbrella_alert", return_value="grab brolly"), \
             mock.patch.object(mod, "get_two_hour_alert", return_value="ignored"):
            out = actions["weather_briefing"]("")
        self.assertEqual(out, "grab brolly")

    def test_action_falls_back_to_two_hour_alert(self):
        mod, actions = load_skill_isolated("weather_briefing")
        with mock.patch.object(mod, "get_umbrella_alert", return_value=""), \
             mock.patch.object(mod, "get_two_hour_alert", return_value="storm soon"):
            out = actions["weather_briefing"]("")
        self.assertEqual(out, "storm soon")

    def test_action_unremarkable_when_quiet(self):
        mod, actions = load_skill_isolated("weather_briefing")
        with mock.patch.object(mod, "get_umbrella_alert", return_value=""), \
             mock.patch.object(mod, "get_two_hour_alert", return_value=""):
            out = actions["weather_briefing"]("")
        self.assertIn("unremarkable", out.lower())

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("weather_briefing")
        with mock.patch.object(mod, "get_umbrella_alert", side_effect=RuntimeError("boom")):
            out = actions["weather_briefing"]("")
        self.assertIn("failed", out.lower())


# ─────────────────────────────────────────────────────────────────────────
#  _config / _read_config (with a FAKE bobert_companion — never the monolith)
# ─────────────────────────────────────────────────────────────────────────
class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def test_config_reads_attr_off_fake_bobert(self):
        bc = types.ModuleType("bobert_companion")
        bc.WEATHER_POLL_MINUTES = 17
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._config("WEATHER_POLL_MINUTES", 30), 17)

    def test_config_returns_default_when_attr_absent(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._config("WEATHER_POLL_MINUTES", 30), 30)

    def test_config_returns_default_when_import_fails(self):
        # bobert_companion absent -> import_module raises -> default. The
        # None-sentinel makes the import raise ImportError deterministically
        # (and crucially avoids executing the real monolith).
        with inject_modules(bobert_companion=None):
            sys.modules["bobert_companion"] = None  # type: ignore[assignment]
            try:
                self.assertEqual(self.mod._config("ANYTHING", "fallback"), "fallback")
            finally:
                sys.modules.pop("bobert_companion", None)

    def test_read_config_coerces_types_and_defaults(self):
        bc = types.ModuleType("bobert_companion")
        bc.WEATHER_BRIEFING_ENABLED = 1
        bc.WEATHER_UMBRELLA_PROB_THRESHOLD = "65"   # int() coercion
        with inject_modules(bobert_companion=bc):
            cfg = self.mod._read_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["umbrella_prob"], 65)
        self.assertEqual(cfg["poll_minutes"], 30)   # untouched default
        self.assertEqual(cfg["lookahead_h"], 2)


# ─────────────────────────────────────────────────────────────────────────
#  _safe_load_json
# ─────────────────────────────────────────────────────────────────────────
class SafeLoadJsonTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def test_missing_file_returns_none(self):
        self.assertIsNone(self.mod._safe_load_json(r"C:\nope\missing.json"))

    def test_reads_valid_json(self, ):
        import json
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"alerts": {"temp_drop": 123.0}}, f)
            out = self.mod._safe_load_json(path)
        finally:
            os.unlink(path)
        self.assertEqual(out["alerts"]["temp_drop"], 123.0)

    def test_corrupt_json_returns_none(self):
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            self.assertIsNone(self.mod._safe_load_json(path))
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────
#  _enqueue_speech — funnel through bobert_companion.proactive_announce
# ─────────────────────────────────────────────────────────────────────────
class EnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def test_announces_via_bobert(self):
        calls = {}
        bc = types.ModuleType("bobert_companion")

        def _announce(msg, source=None):
            calls["msg"] = msg
            calls["source"] = source
            return True

        bc.proactive_announce = _announce
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("rain incoming")
        self.assertEqual(calls["msg"], "rain incoming")
        self.assertEqual(calls["source"], "weather")

    def test_falls_through_when_announce_returns_falsey(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = lambda msg, source=None: False
        # announce returns False -> the "speech-queue unavailable" print path.
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("noop")   # no raise == covered

    def test_no_announcer_attribute(self):
        bc = types.ModuleType("bobert_companion")   # no proactive_announce
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("still fine")

    def test_announce_raises_is_swallowed(self):
        bc = types.ModuleType("bobert_companion")

        def _boom(msg, source=None):
            raise RuntimeError("queue write failed")

        bc.proactive_announce = _boom
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("resilient")   # exception swallowed


# ─────────────────────────────────────────────────────────────────────────
#  _resolve_location — defers to briefing_sources._resolve_location
# ─────────────────────────────────────────────────────────────────────────
class ResolveLocationTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def test_returns_location_from_briefing_sources(self):
        bs = types.ModuleType("briefing_sources")
        bs._resolve_location = lambda: (51.5, -0.12)
        with inject_modules(briefing_sources=bs):
            self.assertEqual(self.mod._resolve_location(), (51.5, -0.12))

    def test_returns_none_when_resolver_raises(self):
        bs = types.ModuleType("briefing_sources")

        def _boom():
            raise RuntimeError("geo down")

        bs._resolve_location = _boom
        with inject_modules(briefing_sources=bs):
            self.assertIsNone(self.mod._resolve_location())

    def test_returns_none_when_briefing_sources_unimportable(self):
        # Both `from . import briefing_sources` and `import briefing_sources`
        # fail -> None. The None-sentinel forces the bare import to raise.
        with inject_modules(briefing_sources=None):
            sys.modules["briefing_sources"] = None  # type: ignore[assignment]
            try:
                self.assertIsNone(self.mod._resolve_location())
            finally:
                sys.modules.pop("briefing_sources", None)


# ─────────────────────────────────────────────────────────────────────────
#  _fetch_hourly_forecast — URL build + JSON parse (fake urlopen, no network)
# ─────────────────────────────────────────────────────────────────────────
class FetchHourlyForecastTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def test_returns_empty_when_no_location(self):
        with mock.patch.object(self.mod, "_resolve_location", return_value=None):
            self.assertEqual(self.mod._fetch_hourly_forecast(), [])

    def test_parses_valid_payload(self):
        import json
        payload = json.dumps({"hourly": {
            "time": ["2026-06-01T12:00", "2026-06-01T13:00"],
            "temperature_2m": [20.0, 18.5],
            "precipitation_probability": [10, 80],
            "precipitation": [0.0, 2.4],
            "weather_code": [0, 61],
        }})
        with mock.patch.object(self.mod, "_resolve_location", return_value=(40.0, -75.0)), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               _fake_urlopen_returning(payload)):
            out = self.mod._fetch_hourly_forecast()
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["category"], "clear")
        self.assertEqual(out[1]["category"], "rain")
        self.assertEqual(out[1]["precip_prob"], 80)
        self.assertEqual(out[1]["desc"], "light rain")
        self.assertAlmostEqual(out[1]["precip_mm"], 2.4)

    def test_handles_none_and_malformed_fields(self):
        import json
        # Row 0: a malformed timestamp -> skipped (continue). Row 1: None temp/
        # prob/mm/code -> coerced to None/0/0.0/-1. Row 2: non-numeric strings
        # -> caught by the per-field try/except.
        payload = json.dumps({"hourly": {
            "time": ["not-a-timestamp", "2026-06-01T13:00", "2026-06-01T14:00"],
            "temperature_2m": [10.0, None, "warm"],
            "precipitation_probability": [0, None, "lots"],
            "precipitation": [0.0, None, "wet"],
            "weather_code": [0, None, "bad"],
        }})
        with mock.patch.object(self.mod, "_resolve_location", return_value=(1.0, 2.0)), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               _fake_urlopen_returning(payload)):
            out = self.mod._fetch_hourly_forecast()
        # The malformed-timestamp row is dropped; two rows survive.
        self.assertEqual(len(out), 2)
        # Row with None values:
        self.assertIsNone(out[0]["temp_c"])
        self.assertEqual(out[0]["precip_prob"], 0)
        self.assertEqual(out[0]["precip_mm"], 0.0)
        self.assertEqual(out[0]["weather_code"], -1)
        # Row with unparseable strings falls back to the same safe defaults:
        self.assertIsNone(out[1]["temp_c"])
        self.assertEqual(out[1]["precip_prob"], 0)
        self.assertEqual(out[1]["weather_code"], -1)

    def test_network_failure_returns_empty(self):
        with mock.patch.object(self.mod, "_resolve_location", return_value=(1.0, 2.0)), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               _fake_urlopen_returning("", raises=OSError("offline"))):
            self.assertEqual(self.mod._fetch_hourly_forecast(), [])

    def test_empty_hourly_block_yields_empty_list(self):
        import json
        with mock.patch.object(self.mod, "_resolve_location", return_value=(1.0, 2.0)), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               _fake_urlopen_returning(json.dumps({}))):
            self.assertEqual(self.mod._fetch_hourly_forecast(), [])


# ─────────────────────────────────────────────────────────────────────────
#  Extra umbrella branches: measurable-precip fallback + thunderstorm phrasing
# ─────────────────────────────────────────────────────────────────────────
class UmbrellaExtraBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "umbrella_prob": 50, "lookahead_h": 2,
                "sig_temp_drop_c": 5, "cooldown_h": 4, "poll_minutes": 30,
                "proactive": True}
        base.update(over)
        return base

    def test_umbrella_measurable_precip_when_prob_below_threshold(self):
        # prob 0 (below threshold) but >=1mm measurable precip -> still flagged.
        now = datetime(2026, 6, 1, 12, 0, 0)
        h = _hour(now + timedelta(hours=1), prob=0, mm=1.5, code=61)
        h["category"] = "rain"
        with _frozen_now(self.mod, now), \
             mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[h]):
            out = self.mod.get_umbrella_alert("today")
        self.assertIn("umbrella", out.lower())

    def test_umbrella_thunderstorm_phrasing(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        h = _hour(now + timedelta(hours=1), prob=75, code=95)
        h["category"] = "thunderstorm"
        with _frozen_now(self.mod, now), \
             mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[h]):
            out = self.mod.get_umbrella_alert("today")
        self.assertIn("thunderstorms", out.lower())
        self.assertIn("stay indoors", out.lower())

    def test_umbrella_empty_window_returns_blank(self):
        # Forecast present but the day-slice is empty (all hours already past).
        now = datetime(2026, 6, 1, 12, 0, 0)
        past = _hour(now - timedelta(hours=3), prob=90, code=61)
        past["category"] = "rain"
        with _frozen_now(self.mod, now), \
             mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[past]):
            self.assertEqual(self.mod.get_umbrella_alert("today"), "")

    def test_umbrella_rain_hour_below_prob_and_mm_returns_blank(self):
        # A rain-category hour in-window, but prob < threshold AND precip < 1mm:
        # both the prob-filter and the measurable-precip fallback yield nothing
        # -> "" (covers the second `if not rainy: return ''` at line 305).
        now = datetime(2026, 6, 1, 12, 0, 0)
        h = _hour(now + timedelta(hours=1), prob=20, mm=0.2, code=61)
        h["category"] = "rain"
        with _frozen_now(self.mod, now), \
             mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[h]):
            self.assertEqual(self.mod.get_umbrella_alert("today"), "")


# ─────────────────────────────────────────────────────────────────────────
#  Detector: category-change branch + current-from-window fallback
# ─────────────────────────────────────────────────────────────────────────
class DetectorExtraBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "umbrella_prob": 50, "lookahead_h": 2,
                "sig_temp_drop_c": 5, "cooldown_h": 4, "poll_minutes": 30,
                "proactive": True}
        base.update(over)
        return base

    def test_category_change_into_fog(self):
        now = datetime.now()
        cur = _hour(now - timedelta(minutes=30), prob=0, code=0)
        cur["category"] = "clear"
        filler = _hour(now + timedelta(minutes=20), prob=0, code=0)
        filler["category"] = "clear"
        foggy = _hour(now + timedelta(minutes=90), prob=0, code=45)
        foggy["category"] = "fog"
        foggy["desc"] = "foggy"
        klass, msg = self.mod._detect_two_hour_change([cur, filler, foggy], self._cfg())
        self.assertEqual(klass, "category_change")
        self.assertIn("foggy", msg)

    def test_current_temp_none_skips_temp_block_reaches_category(self):
        # current bucket has temp_c=None, so the temp-drop block's guard
        # (`current["temp_c"] is not None`) is False and control flows past it
        # (378->396) into the category section, which then reports the shift.
        now = datetime.now()
        cur = _hour(now - timedelta(minutes=30), prob=0, code=0, temp=None)
        cur["category"] = "clear"
        filler = _hour(now + timedelta(minutes=20), prob=0, code=0, temp=None)
        filler["category"] = "clear"
        snowy = _hour(now + timedelta(minutes=90), prob=0, code=73, temp=None)
        snowy["category"] = "snow"
        snowy["desc"] = "snow"
        klass, msg = self.mod._detect_two_hour_change([cur, filler, snowy], self._cfg())
        self.assertEqual(klass, "category_change")
        self.assertIn("snow", msg)

    def test_current_falls_back_to_window_first_when_no_straddle(self):
        # No bucket straddles `now` (all are in the future), so `current`
        # becomes window[0]. A precip jump from that baseline still fires.
        now = datetime.now()
        first = _hour(now + timedelta(minutes=10), prob=5, code=0)
        first["category"] = "clear"
        wet = _hour(now + timedelta(minutes=80), prob=70, code=61)
        wet["category"] = "rain"
        klass, msg = self.mod._detect_two_hour_change([first, wet], self._cfg())
        self.assertEqual(klass, "precip_jump")

    def test_no_change_when_all_quiet(self):
        # Two-hour window (>=2 buckets), current temp present, no precip jump,
        # no significant temp drop, and the only category change is into a
        # NON-precip state (clear -> cloudy). The category loop's `continue`
        # arms fire and the function falls through to the final (None, "")
        # return — covering the category-skip + terminal-return branches.
        now = datetime.now()
        cur = _hour(now - timedelta(minutes=30), prob=0, code=0, temp=20.0)
        cur["category"] = "clear"
        filler = _hour(now + timedelta(minutes=20), prob=0, code=0, temp=20.0)
        filler["category"] = "clear"            # same cat -> continue
        cloudy = _hour(now + timedelta(minutes=80), prob=0, code=2, temp=19.0)
        cloudy["category"] = "cloudy"           # changed, but non-precip -> skip
        klass, msg = self.mod._detect_two_hour_change(
            [cur, filler, cloudy], self._cfg())
        self.assertIsNone(klass)
        self.assertEqual(msg, "")

    def test_temp_drop_skips_none_temp_hours(self):
        # current temp present; one window hour has temp_c=None (skipped via the
        # `if h["temp_c"] is None: continue` at line 382), and no hour produces
        # a >= threshold drop -> no temp_drop alert. With all categories clear
        # and probs flat, the detector returns (None, "").
        now = datetime.now()
        cur = _hour(now - timedelta(minutes=30), prob=0, code=0, temp=20.0)
        cur["category"] = "clear"
        gap = _hour(now + timedelta(minutes=30), prob=0, code=0, temp=None)
        gap["category"] = "clear"
        warm = _hour(now + timedelta(minutes=90), prob=0, code=0, temp=19.5)
        warm["category"] = "clear"
        klass, msg = self.mod._detect_two_hour_change([cur, gap, warm], self._cfg())
        self.assertIsNone(klass)


# ─────────────────────────────────────────────────────────────────────────
#  get_two_hour_alert wrapper
# ─────────────────────────────────────────────────────────────────────────
class GetTwoHourAlertTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "umbrella_prob": 50, "lookahead_h": 2,
                "sig_temp_drop_c": 5, "cooldown_h": 4, "poll_minutes": 30,
                "proactive": True}
        base.update(over)
        return base

    def test_disabled_returns_blank(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(enabled=False)):
            self.assertEqual(self.mod.get_two_hour_alert(), "")

    def test_no_forecast_returns_blank(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[]):
            self.assertEqual(self.mod.get_two_hour_alert(), "")

    def test_returns_detector_message(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_fetch_hourly_forecast",
                               return_value=[object()]), \
             mock.patch.object(self.mod, "_detect_two_hour_change",
                               return_value=("precip_jump", "wetter at 3 PM")):
            self.assertEqual(self.mod.get_two_hour_alert(), "wetter at 3 PM")


# ─────────────────────────────────────────────────────────────────────────
#  Cooldown record + _alert_cooldown_active edge paths
# ─────────────────────────────────────────────────────────────────────────
class CooldownRecordTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def test_cooldown_handles_bad_timestamp(self):
        state = {"alerts": {"precip_jump": "not-a-number"}}
        with mock.patch.object(self.mod, "_safe_load_json", return_value=state):
            # float() of the bad value raises -> the except returns False.
            self.assertFalse(self.mod._alert_cooldown_active("precip_jump", 3600))

    def test_record_alert_writes_state(self):
        captured = {}

        def _fake_write(path, data):
            captured["path"] = path
            captured["data"] = data

        with mock.patch.object(self.mod, "_safe_load_json", return_value={}), \
             mock.patch.object(self.mod, "_atomic_write_json", _fake_write):
            self.mod._record_alert("temp_drop")
        self.assertIn("temp_drop", captured["data"]["alerts"])
        self.assertIsInstance(captured["data"]["alerts"]["temp_drop"], float)

    def test_record_alert_merges_existing_alerts(self):
        captured = {}
        with mock.patch.object(self.mod, "_safe_load_json",
                               return_value={"alerts": {"precip_jump": 1.0}}), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d: captured.update(data=d)):
            self.mod._record_alert("temp_drop")
        self.assertEqual(captured["data"]["alerts"]["precip_jump"], 1.0)
        self.assertIn("temp_drop", captured["data"]["alerts"])

    def test_record_alert_swallows_write_error(self):
        with mock.patch.object(self.mod, "_safe_load_json", return_value={}), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._record_alert("temp_drop")   # must not raise


# ─────────────────────────────────────────────────────────────────────────
#  _watch_loop — exercise ONE iteration without looping forever
# ─────────────────────────────────────────────────────────────────────────
class WatchLoopTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "umbrella_prob": 50, "lookahead_h": 2,
                "sig_temp_drop_c": 5, "cooldown_h": 4, "poll_minutes": 30,
                "proactive": True}
        base.update(over)
        return base

    @contextlib.contextmanager
    def _break_after_sleeps(self, n_break=2):
        """Let _watch_loop run a bounded number of iterations: each time.sleep
        is a no-op until the `n_break`-th call, which raises a sentinel that
        breaks out so the `while True` doesn't spin forever. With the default
        n_break=2 the loop runs exactly one full iteration (1st sleep = startup
        delay, 2nd = end-of-iteration / recovery guard). Pass n_break=3 to allow
        a `continue`-then-loop-again path to execute its `continue` statement.

        The sentinel subclasses BaseException — NOT Exception — on purpose: the
        loop body's `except Exception` recovery guard must let it propagate, or
        it would be caught, sleep again, re-raise, and livelock. Tests catch the
        sentinel themselves around the _watch_loop() call."""
        class _Break(BaseException):
            pass

        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] >= n_break:
                raise _Break()

        with mock.patch.object(self.mod.time, "sleep", _sleep):
            yield _Break

    def test_watch_loop_fires_alert_when_change_and_no_cooldown(self):
        fired = {}
        with self._break_after_sleeps() as Break:
            with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
                 mock.patch.object(self.mod, "_fetch_hourly_forecast",
                                   return_value=[object()]), \
                 mock.patch.object(self.mod, "_detect_two_hour_change",
                                   return_value=("precip_jump", "rain soon")), \
                 mock.patch.object(self.mod, "_alert_cooldown_active", return_value=False), \
                 mock.patch.object(self.mod, "_enqueue_speech",
                                   side_effect=lambda m: fired.update(msg=m)), \
                 mock.patch.object(self.mod, "_record_alert",
                                   side_effect=lambda k: fired.update(klass=k)):
                try:
                    self.mod._watch_loop()
                except Break:
                    pass
        self.assertEqual(fired.get("msg"), "rain soon")
        self.assertEqual(fired.get("klass"), "precip_jump")

    def test_watch_loop_suppresses_alert_during_cooldown(self):
        with self._break_after_sleeps() as Break:
            with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
                 mock.patch.object(self.mod, "_fetch_hourly_forecast",
                                   return_value=[object()]), \
                 mock.patch.object(self.mod, "_detect_two_hour_change",
                                   return_value=("precip_jump", "rain soon")), \
                 mock.patch.object(self.mod, "_alert_cooldown_active", return_value=True), \
                 mock.patch.object(self.mod, "_enqueue_speech") as enq, \
                 mock.patch.object(self.mod, "_record_alert") as rec:
                try:
                    self.mod._watch_loop()
                except Break:
                    pass
        enq.assert_not_called()
        rec.assert_not_called()

    def test_watch_loop_empty_forecast_skips_detection(self):
        # fetch returns [] -> the `if hourly:` guard is False, detection is
        # skipped (462 branch), and the loop sleeps to the end-of-iteration.
        with self._break_after_sleeps() as Break:
            with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
                 mock.patch.object(self.mod, "_fetch_hourly_forecast", return_value=[]), \
                 mock.patch.object(self.mod, "_detect_two_hour_change") as detect, \
                 mock.patch.object(self.mod, "_enqueue_speech") as enq:
                try:
                    self.mod._watch_loop()
                except Break:
                    pass
        detect.assert_not_called()
        enq.assert_not_called()

    def test_watch_loop_no_alert_class_does_not_enqueue(self):
        # fetch yields data but the detector reports nothing notable
        # (klass None) -> the `if klass and msg:` guard is False (464 branch).
        with self._break_after_sleeps() as Break:
            with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
                 mock.patch.object(self.mod, "_fetch_hourly_forecast",
                                   return_value=[object()]), \
                 mock.patch.object(self.mod, "_detect_two_hour_change",
                                   return_value=(None, "")), \
                 mock.patch.object(self.mod, "_enqueue_speech") as enq:
                try:
                    self.mod._watch_loop()
                except Break:
                    pass
        enq.assert_not_called()

    def test_watch_loop_skips_when_proactive_disabled(self):
        # enabled but proactive False -> the `if not (enabled and proactive)`
        # guard sleeps the poll interval and `continue`s WITHOUT fetching. Use
        # n_break=3 so the in-branch sleep (call #2) is a no-op and the
        # `continue` statement actually executes before the next loop's sleep
        # (call #3) breaks out.
        with self._break_after_sleeps(n_break=3) as Break:
            with mock.patch.object(self.mod, "_read_config",
                                   return_value=self._cfg(proactive=False)), \
                 mock.patch.object(self.mod, "_fetch_hourly_forecast") as fetch:
                try:
                    self.mod._watch_loop()
                except Break:
                    pass
        fetch.assert_not_called()

    def test_watch_loop_recovers_from_iteration_exception(self):
        # _read_config raising inside the outer try is caught by the
        # `except Exception: logging.exception(...); time.sleep(60)` guard.
        # The startup sleep is call #1; the recovery sleep is call #2 -> break.
        with self._break_after_sleeps() as Break:
            with mock.patch.object(self.mod, "_read_config",
                                   side_effect=RuntimeError("cfg boom")):
                try:
                    self.mod._watch_loop()
                except Break:
                    pass
        # Reaching here without propagating the RuntimeError == recovery covered.

    def test_watch_loop_inner_except_catches_fetch_error(self):
        # _fetch_hourly_forecast raising is caught by the INNER
        # `except Exception` (470-471) — the loop logs and continues to its
        # end-of-iteration sleep rather than crashing the outer guard.
        with self._break_after_sleeps() as Break:
            with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
                 mock.patch.object(self.mod, "_fetch_hourly_forecast",
                                   side_effect=RuntimeError("fetch boom")), \
                 mock.patch.object(self.mod, "_enqueue_speech") as enq:
                try:
                    self.mod._watch_loop()
                except Break:
                    pass
        enq.assert_not_called()   # error path skipped straight to the sleep


# ─────────────────────────────────────────────────────────────────────────
#  register() — action wiring + watcher-arm branches
# ─────────────────────────────────────────────────────────────────────────
class WeatherImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        # Re-exec the source with the project root removed from sys.path so the
        # `if _PROJECT_DIR not in sys.path: sys.path.insert(...)` guard runs.
        mod, _ = load_skill_isolated("weather_briefing")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("weather_reexec", path)
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


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("weather_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "umbrella_prob": 50, "lookahead_h": 2,
                "sig_temp_drop_c": 5, "cooldown_h": 4, "poll_minutes": 30,
                "proactive": True}
        base.update(over)
        return base

    def test_register_wires_both_actions(self):
        actions: dict = {}
        with mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(enabled=False)):
            self.mod.register(actions)
        self.assertIn("weather_briefing", actions)
        self.assertIs(actions["weather_briefing"], actions["weather_forecast"])

    def test_register_disabled_does_not_start_thread(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(enabled=False)), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register({})
        Thread.assert_not_called()

    def test_register_proactive_off_does_not_start_thread(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(proactive=False)), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register({})
        Thread.assert_not_called()

    def test_register_arms_watcher_when_enabled(self):
        started = {"n": 0}

        class _FakeThread:
            def __init__(self, target=None, daemon=None, **k):
                self._target = target

            def start(self):
                started["n"] += 1   # never actually run the loop

        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod.threading, "Thread", _FakeThread):
            self.mod.register({})
        self.assertEqual(started["n"], 1)

    def test_action_smoke_via_register(self):
        # End-to-end through the registered closure: umbrella wins.
        actions: dict = {}
        with mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(enabled=False)):
            self.mod.register(actions)
        with mock.patch.object(self.mod, "get_umbrella_alert", return_value="brolly time"):
            self.assertEqual(actions["weather_briefing"](""), "brolly time")


if __name__ == "__main__":
    unittest.main()
