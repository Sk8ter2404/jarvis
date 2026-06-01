"""Logic tests for skills/weather_briefing.py.

Covers the pure forecast logic — WMO category mapping, hour formatting, the
"hours falling on this day" slice, umbrella-alert phrasing, the two-hour
change detector's priority ladder (thunderstorm > precip-jump > temp-drop >
category-change), and the cooldown state read. The Open-Meteo fetch is mocked
throughout so no test hits the network. The proactive watcher thread is
neutered by the harness.
"""
from __future__ import annotations

import contextlib
import unittest
from datetime import datetime, timedelta
from unittest import mock

from tests._skill_harness import load_skill_isolated


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


if __name__ == "__main__":
    unittest.main()
