"""Logic tests for skills/evening_briefing.py.

Covers the pure text helpers (tomorrow-weather phrasing, count humaniser,
wttr/Open-Meteo parsers, the session-log pattern scanner + dry observation),
the cross-skill Bambu/face-tracker reads, today's-tasks/interaction counters,
the section builders that sit over briefing_sources / news_briefing /
weather_briefing / Outlook COM, the full _build_briefing assembly with every
external source mocked, the speech-queue + persistence helpers, the scheduler
loop's gating arithmetic, and register()'s action + mid-task-status bridge.

ISOLATION CONTRACT (wave-1/2 lessons):
  * Every fake module is installed only for the duration of a `with` block via
    ``mock.patch.dict(sys.modules, {...})`` (auto-restored) or the local
    ``inject_modules`` save/restore ctx manager — never a bare module-level
    ``sys.modules`` write. The real ``bobert_companion`` / cross-skill modules
    are left exactly as they were after each test.
  * No real network, LLM, hardware, Outlook COM, threads, or sleeps: the
    scheduler thread is neutered by the harness, ``time.sleep`` is patched, and
    every fetch is mocked. ``_atomic_write_json`` is patched so neither
    pending_speech.json nor evening_briefing_state.json is ever written.
  * Fixtures use generic names/places only (no personal data).
"""
from __future__ import annotations

import contextlib
import datetime
import json
import sys
import time
import types
import unittest
from collections import Counter
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


def _fake_bc():
    """A minimal ``bobert_companion`` stand-in carrying just the config knobs +
    registries ``register()`` consults. Injected during every load so the skill
    never imports the real 14K-line monolith — which, with a live JARVIS
    instance holding the singleton lock, ``sys.exit(0)``s mid-import, and which
    CI does not ship anyway."""
    bc = types.ModuleType("bobert_companion")
    bc.EVENING_BRIEFING_ENABLED = True
    bc.EVENING_BRIEFING_HOUR = 22
    bc.EVENING_BRIEFING_MINUTE = 0
    bc.EVENING_BRIEFING_WAIT_MINUTES = 30
    bc.LONG_RUNNING_ACTIONS = set()
    bc._MID_TASK_STATUS_BUCKET = {}
    bc.proactive_announce = lambda *a, **k: None
    return bc


def _load(register=True):
    """Load skills/evening_briefing in isolation with a fake bobert_companion
    present for the duration of ``register()`` (the harness neuters the
    scheduler thread). Returns ``(module, actions)``. The fake bc is removed
    again on return, so each test installs its own bc where it needs one."""
    with mock.patch.dict(sys.modules, {"bobert_companion": _fake_bc()}):
        return load_skill_isolated("evening_briefing", register=register)


@contextlib.contextmanager
def block_import(*names):
    """Force ``import <name>`` (and ``from pkg import <name>``) to raise
    ImportError inside the block, AND detach any already-imported target from
    sys.modules so the import machinery can't satisfy it from cache. Restores
    everything on exit. Used to exercise a section builder's missing-dependency
    branch deterministically even though the real module exists on the dev box
    (importing it would hit the network)."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        leaf = name.rsplit(".", 1)[-1]
        # `from pkg import leaf` calls __import__(pkg, fromlist=[leaf]); cover
        # that by also inspecting the fromlist for a blocked leaf.
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist") or ()
        if (name in blocked or leaf in blocked or name.split(".")[0] in blocked
                or any(f in blocked for f in fromlist)):
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    # Detach the cached module AND the parent-package attribute (e.g.
    # ``skills.briefing_sources`` + ``skills.briefing_sources`` attr on the
    # ``skills`` package) so neither ``import x`` nor ``from skills import x``
    # can be satisfied from cache, forcing the blocked __import__ to run.
    saved_mod: dict[str, object] = {}
    saved_attr: list = []
    for name in list(blocked):
        for key in (name, f"skills.{name}"):
            if key in sys.modules:
                saved_mod[key] = sys.modules.pop(key)
        skills_pkg = sys.modules.get("skills")
        if skills_pkg is not None and hasattr(skills_pkg, name):
            saved_attr.append((skills_pkg, name, getattr(skills_pkg, name)))
            with contextlib.suppress(AttributeError):
                delattr(skills_pkg, name)
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            setattr(parent, leaf, prev)
        for key, mod in saved_mod.items():
            sys.modules[key] = mod


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install / remove fake modules in ``sys.modules`` for the
    duration of the block, restoring the previous state — including absence —
    on exit. ``name=None`` forces the module ABSENT inside the block. Dotted
    names are supported (the leaf is also set on its parent package so
    ``from skills import briefing_sources`` resolves the fake)."""
    saved: dict[str, object] = {}
    saved_attr: list = []
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append((parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                with contextlib.suppress(AttributeError):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved.get(name, _SENTINEL)
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _resp(payload):
    """A urlopen() context-manager stand-in whose .read() yields JSON bytes."""
    r = mock.MagicMock()
    r.read.return_value = json.dumps(payload).encode("utf-8")
    r.__enter__ = lambda s: r
    r.__exit__ = lambda *a: False
    return r


class EveningBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load()

    # ── _phrase_tomorrow (pure) ──────────────────────────────────────────
    def test_phrase_tomorrow_with_desc(self):
        self.assertEqual(
            self.mod._phrase_tomorrow(18, 9, "Partly Cloudy"),
            "tomorrow looks like a high of 18, low of 9, and partly cloudy",
        )

    def test_phrase_tomorrow_without_desc(self):
        self.assertEqual(
            self.mod._phrase_tomorrow(18, 9, ""),
            "tomorrow looks like a high of 18 and a low of 9",
        )

    # ── _humanize_count (pure) ───────────────────────────────────────────
    def test_humanize_count(self):
        h = self.mod._humanize_count
        self.assertEqual(h(2), "twice")
        self.assertEqual(h(3), "three times")
        self.assertEqual(h(4), "four times")
        self.assertEqual(h(11), "11 times")

    # ── _tomorrow_weather_from_wttr (parses mocked JSON) ─────────────────
    def test_tomorrow_weather_from_wttr_parses(self):
        payload = {
            "weather": [
                {"maxtempC": "20", "mintempC": "10", "maxtempF": "20", "mintempF": "10", "hourly": []},   # today
                {"maxtempC": "18", "mintempC": "9", "maxtempF": "18", "mintempF": "9",
                 "hourly": [{"time": "1200",
                             "weatherDesc": [{"value": "Sunny"}]}]},  # tomorrow
            ]
        }
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = __import__("json").dumps(payload).encode()
        fake_resp.__enter__ = lambda s: fake_resp
        fake_resp.__exit__ = lambda *a: False
        with mock.patch.object(self.mod.urllib.request, "urlopen", return_value=fake_resp):
            out = self.mod._tomorrow_weather_from_wttr()
        self.assertIn("high of 18", out)
        self.assertIn("low of 9", out)
        self.assertIn("sunny", out)

    def test_tomorrow_weather_from_wttr_network_fail(self):
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=OSError("no net")):
            self.assertEqual(self.mod._tomorrow_weather_from_wttr(), "")

    def test_fetch_tomorrow_weather_falls_back_to_open_meteo(self):
        with mock.patch.object(self.mod, "_tomorrow_weather_from_wttr", return_value=""), \
             mock.patch.object(self.mod, "_tomorrow_weather_from_open_meteo",
                               return_value="tomorrow looks like a high of 5 and a low of 1"):
            out = self.mod._fetch_tomorrow_weather()
        self.assertIn("high of 5", out)

    # ── _scan_today_for_patterns + _dry_observation ──────────────────────
    def test_scan_today_for_patterns(self):
        log = (
            "[10:00:00] [action] play_music: 'Michael Jackson'\n"
            "  You:    play Michael Jackson\n"
            "  You:    play Michael Jackson please\n"
            "[10:05:00] [action] see_screen:\n"
        )
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["x.log"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=log)):
            actions, plays, you = self.mod._scan_today_for_patterns()
        self.assertEqual(you, 2)
        self.assertEqual(actions["play_music"], 1)
        self.assertEqual(plays["michael jackson"], 2)  # filler "please" stripped

    def test_dry_observation_play_pattern(self):
        plays = Counter({"michael jackson": 4})
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(Counter(), plays, 10)):
            out = self.mod._dry_observation()
        self.assertIn("'play michael jackson'", out)
        self.assertIn("four times", out)
        self.assertIn("pattern emerges", out.lower())

    def test_dry_observation_repeated_action(self):
        actions = Counter({"check_weather": 5, "see_screen": 99})  # boring excluded
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(actions, Counter(), 10)):
            out = self.mod._dry_observation()
        self.assertIn("check weather", out)
        self.assertIn("five times", out)

    def test_dry_observation_nothing(self):
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(Counter(), Counter(), 0)):
            self.assertEqual(self.mod._dry_observation(), "")

    def test_dry_observation_below_threshold(self):
        # 2 plays is under DRY_OBS_MIN_COUNT (3) → no remark.
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(Counter(), Counter({"jazz": 2}), 5)):
            self.assertEqual(self.mod._dry_observation(), "")

    # ── _count_tasks_completed_today ─────────────────────────────────────
    def test_count_tasks_completed_today(self):
        today = time.strftime("%Y-%m-%d")
        todo = (f"- [x] done one {today}\n"
                f"- [x] old task 1999-01-01\n"
                f"- [ ] pending {today}\n"
                f"- [x] done two {today}\n")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=todo)):
            self.assertEqual(self.mod._count_tasks_completed_today(), 2)

    # ── _bambu_status (cross-skill read) ─────────────────────────────────
    def test_bambu_status_running(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time(), "gcode_state": "RUNNING",
                       "filename": "p.3mf", "layer_num": 5, "total_layer": 50,
                       "mc_remaining": 45}
        fake._strip_filename = lambda s: "p"
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("still printing", out)
        self.assertIn("layer 5 of 50", out)

    def test_bambu_status_failed(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time(), "gcode_state": "FAILED", "filename": ""}
        fake._strip_filename = lambda s: ""
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertIn("failure", self.mod._bambu_status().lower())

    def test_bambu_status_absent(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_bambu_monitor", None)
            self.assertEqual(self.mod._bambu_status(), "")

    # ── _build_briefing assembly ─────────────────────────────────────────
    def test_build_briefing_full(self):
        with mock.patch.object(self.mod, "_count_voice_interactions_today", return_value=5), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=2), \
             mock.patch.object(self.mod, "_bambu_status", return_value="the H2D is still printing"), \
             mock.patch.object(self.mod, "_fetch_tomorrow_weather",
                               return_value="tomorrow looks like a high of 18"), \
             mock.patch.object(self.mod, "_first_meeting_tomorrow",
                               return_value="your first meeting tomorrow is at 9 AM"), \
             mock.patch.object(self.mod, "_dry_observation",
                               return_value="you said 'play X' four times today, sir."), \
             mock.patch.object(self.mod, "_fetch_news", return_value="Today's headlines, sir. Y."), \
             mock.patch.object(self.mod, "_fetch_tomorrow_umbrella", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("Good evening, sir. 5 voice interactions", out)
        self.assertIn("2 tasks cleared", out)
        self.assertIn("Currently, the H2D is still printing", out)
        self.assertIn("For tomorrow,", out)
        self.assertTrue(out.startswith("[intent:briefing]"))  # news included

    def test_build_briefing_quiet_day(self):
        with mock.patch.object(self.mod, "_count_voice_interactions_today", return_value=0), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=0), \
             mock.patch.object(self.mod, "_bambu_status", return_value=""), \
             mock.patch.object(self.mod, "_fetch_tomorrow_weather", return_value=""), \
             mock.patch.object(self.mod, "_first_meeting_tomorrow", return_value=""), \
             mock.patch.object(self.mod, "_dry_observation", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value=""), \
             mock.patch.object(self.mod, "_fetch_tomorrow_umbrella", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("A quiet day on the voice channel", out)
        self.assertFalse(out.startswith("[intent:briefing]"))

    # ── evening_briefing action ──────────────────────────────────────────
    def test_action_returns_and_enqueues(self):
        mod, actions = _load()
        with mock.patch.object(mod, "_build_briefing", return_value="Good evening, sir."), \
             mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_show_card_safe"), \
             mock.patch.object(mod, "_save_last_fired_date"):
            out = actions["evening_briefing"]("")
        self.assertEqual(out, "Good evening, sir.")
        enq.assert_called_once()

    def test_action_handles_exception(self):
        mod, actions = _load()
        with mock.patch.object(mod, "_build_briefing", side_effect=RuntimeError("boom")):
            out = actions["evening_briefing"]("")
        self.assertIn("failed", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# Weather parsers — wttr edge cases + the Open-Meteo fallback path.
# ─────────────────────────────────────────────────────────────────────────
class WeatherParserTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    def test_wttr_no_noon_bucket_uses_middle(self):
        # No 11/12/13:00 entry → falls back to the middle hourly bucket.
        payload = {"weather": [
            {"maxtempC": "20", "mintempC": "10", "maxtempF": "20", "mintempF": "10", "hourly": []},
            {"maxtempC": "16", "mintempC": "7", "maxtempF": "16", "mintempF": "7", "hourly": [
                {"time": "0300", "weatherDesc": [{"value": "Foggy"}]},
                {"time": "0900", "weatherDesc": [{"value": "Cloudy"}]},
                {"time": "1800", "weatherDesc": [{"value": "Clear"}]},
            ]},
        ]}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_resp(payload)):
            out = self.mod._tomorrow_weather_from_wttr()
        self.assertIn("high of 16", out)
        self.assertIn("cloudy", out)   # middle bucket (index 1 of 3)

    def test_wttr_empty_weatherdesc_yields_no_desc(self):
        # weatherDesc is an empty list → noon[...][0] raises IndexError, which
        # the inner handler swallows → desc "" → phrase has no description.
        # (The string-shaped variant — weatherDesc="Sunny" — raises
        # AttributeError instead; see test_wttr_string_weatherdesc_yields_no_desc
        # for that companion case, now also covered by the inner handler.)
        payload = {"weather": [
            {"maxtempC": "20", "mintempC": "10", "maxtempF": "20", "mintempF": "10", "hourly": []},
            {"maxtempC": "14", "mintempC": "6", "maxtempF": "14", "mintempF": "6",
             "hourly": [{"time": "1200", "weatherDesc": []}]},
        ]}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_resp(payload)):
            out = self.mod._tomorrow_weather_from_wttr()
        self.assertEqual(out, "tomorrow looks like a high of 14 and a low of 6")

    def test_wttr_string_weatherdesc_yields_no_desc(self):
        # Malformed payload: weatherDesc is a bare *string* ("Sunny") rather
        # than the expected list-of-dicts. `noon.get("weatherDesc", [{}])[0]`
        # then yields the char 'S', whose .get() raises AttributeError. The
        # inner handler now includes AttributeError, so it degrades to desc ""
        # → no-desc phrase, honouring the "'' on failure" contract instead of
        # propagating out of the weather builder.
        payload = {"weather": [
            {"maxtempC": "20", "mintempC": "10", "maxtempF": "20", "mintempF": "10", "hourly": []},
            {"maxtempC": "14", "mintempC": "6", "maxtempF": "14", "mintempF": "6",
             "hourly": [{"time": "1200", "weatherDesc": "Sunny"}]},
        ]}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_resp(payload)):
            out = self.mod._tomorrow_weather_from_wttr()
        self.assertEqual(out, "tomorrow looks like a high of 14 and a low of 6")

    def test_wttr_missing_tomorrow_block_returns_empty(self):
        # Only today's block → weather[1] raises IndexError → "".
        payload = {"weather": [{"maxtempC": "20", "mintempC": "10", "maxtempF": "20", "mintempF": "10", "hourly": []}]}
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_resp(payload)):
            self.assertEqual(self.mod._tomorrow_weather_from_wttr(), "")

    def test_wttr_retry_then_success(self):
        # First urlopen attempt raises, the retry succeeds → phrase returned.
        payload = {"weather": [
            {"maxtempC": "20", "mintempC": "10", "maxtempF": "20", "mintempF": "10", "hourly": []},
            {"maxtempC": "12", "mintempC": "4", "maxtempF": "12", "mintempF": "4", "hourly": []},
        ]}
        seq = [OSError("transient 5xx"), _resp(payload)]

        def _urlopen(*a, **k):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        with mock.patch.object(self.mod.urllib.request, "urlopen", side_effect=_urlopen):
            out = self.mod._tomorrow_weather_from_wttr()
        self.assertIn("high of 12", out)
        self.assertEqual(seq, [])   # both the failed + retry attempts consumed

    def test_fetch_tomorrow_weather_returns_wttr_when_present(self):
        # wttr succeeds → Open-Meteo fallback is never consulted (line 300).
        with mock.patch.object(self.mod, "_tomorrow_weather_from_wttr",
                               return_value="tomorrow looks like a high of 9"), \
             mock.patch.object(self.mod, "_tomorrow_weather_from_open_meteo") as om:
            out = self.mod._fetch_tomorrow_weather()
        self.assertIn("high of 9", out)
        om.assert_not_called()

    # ── Open-Meteo fallback ──────────────────────────────────────────────
    def _fake_sources(self, loc=(40.0, -75.0), raises=False):
        bs = types.ModuleType("briefing_sources")
        if raises:
            def _boom():
                raise RuntimeError("geo down")
            bs._resolve_location = _boom
        else:
            bs._resolve_location = lambda: loc
        return bs

    def test_open_meteo_parses_daily(self):
        bs = self._fake_sources()
        payload = {"daily": {
            "temperature_2m_max": [20.4, 17.8],
            "temperature_2m_min": [11.1, 8.6],
            "weather_code": [1, 61],
        }}
        with inject_modules(**{"skills.briefing_sources": bs, "briefing_sources": bs}), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_resp(payload)):
            out = self.mod._tomorrow_weather_from_open_meteo()
        self.assertIn("high of 18", out)   # 17.8 rounds to 18
        self.assertIn("low of 9", out)     # 8.6 rounds to 9
        self.assertIn("light rain", out)   # WMO code 61

    def test_open_meteo_unknown_code_drops_desc(self):
        bs = self._fake_sources()
        payload = {"daily": {
            "temperature_2m_max": [20.0, 15.0],
            "temperature_2m_min": [10.0, 5.0],
            "weather_code": [0, 999],   # 999 not in _WMO_DESCRIPTIONS
        }}
        with inject_modules(**{"skills.briefing_sources": bs, "briefing_sources": bs}), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_resp(payload)):
            out = self.mod._tomorrow_weather_from_open_meteo()
        self.assertEqual(out, "tomorrow looks like a high of 15 and a low of 5")

    def test_open_meteo_no_location_returns_empty(self):
        bs = self._fake_sources(loc=None)
        with inject_modules(**{"skills.briefing_sources": bs, "briefing_sources": bs}):
            self.assertEqual(self.mod._tomorrow_weather_from_open_meteo(), "")

    def test_open_meteo_resolve_raises_returns_empty(self):
        bs = self._fake_sources(raises=True)
        with inject_modules(**{"skills.briefing_sources": bs, "briefing_sources": bs}):
            self.assertEqual(self.mod._tomorrow_weather_from_open_meteo(), "")

    def test_open_meteo_fetch_raises_returns_empty(self):
        bs = self._fake_sources()
        with inject_modules(**{"skills.briefing_sources": bs, "briefing_sources": bs}), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=OSError("offline")):
            self.assertEqual(self.mod._tomorrow_weather_from_open_meteo(), "")

    def test_open_meteo_bad_payload_returns_empty(self):
        # tomorrow's max temp is non-numeric → float() raises ValueError INSIDE
        # the parse try (so the `(KeyError, IndexError, ValueError, TypeError)`
        # handler is exercised) → graceful "". Also forces _PROJECT_DIR onto
        # sys.path via a sentinel so that prepend branch is covered.
        bs = self._fake_sources()
        payload = {"daily": {
            "temperature_2m_max": [20.0, "n/a"],   # [1] not float-able
            "temperature_2m_min": [10.0, 5.0],
            "weather_code": [0, 1],
        }}
        sentinel = "C:/eb-fake-project-dir"
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        with inject_modules(**{"skills.briefing_sources": bs, "briefing_sources": bs}), \
             mock.patch.object(self.mod, "_PROJECT_DIR", sentinel), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=_resp(payload)):
            self.assertEqual(self.mod._tomorrow_weather_from_open_meteo(), "")
        self.assertIn(sentinel, sys.path)

    def test_open_meteo_sources_unavailable_returns_empty(self):
        # Both `from skills import briefing_sources` and bare `import
        # briefing_sources` fail → graceful "" (no _resolve_location runs).
        with block_import("briefing_sources"):
            self.assertEqual(self.mod._tomorrow_weather_from_open_meteo(), "")


# ─────────────────────────────────────────────────────────────────────────
# Outlook calendar (tomorrow's first meeting) — fake pythoncom + win32com COM.
# ─────────────────────────────────────────────────────────────────────────
class _FakeStart:
    """A win32com COM 'Start' value: has .Format plus Y/M/D/H/M attributes."""
    def __init__(self, dt):
        self.year, self.month, self.day = dt.year, dt.month, dt.day
        self.hour, self.minute = dt.hour, dt.minute

    def Format(self, *_a):   # noqa: N802 (COM casing)
        return "fmt"


class _FakeAppt:
    def __init__(self, start, subject="", organizer=""):
        self.Start = start
        self.Subject = subject
        self.Organizer = organizer


class _FakeItems(list):
    def __init__(self, appts):
        super().__init__(appts)
        self.IncludeRecurrences = False

    def Sort(self, *_a):       # noqa: N802
        pass

    def Restrict(self, *_a):   # noqa: N802
        return self


def _make_outlook(appts, restrict_raises=False, com_init_raises=False):
    """Build fake pythoncom + win32com.client modules wired so
    win32com.client.Dispatch('Outlook.Application') returns a calendar whose
    Items yields ``appts``. ``com_init_raises`` makes both CoInitialize and
    CoUninitialize throw (their failures are designed to be swallowed)."""
    pythoncom = types.ModuleType("pythoncom")
    if com_init_raises:
        def _co_boom():
            raise RuntimeError("COM apartment busy")
        pythoncom.CoInitialize = _co_boom
        pythoncom.CoUninitialize = _co_boom
    else:
        pythoncom.CoInitialize = lambda: None
        pythoncom.CoUninitialize = lambda: None

    items = _FakeItems(appts)
    if restrict_raises:
        def _restrict(*_a):
            raise RuntimeError("restrict not supported")
        items.Restrict = _restrict

    calendar = types.SimpleNamespace(Items=items)
    namespace = types.SimpleNamespace(GetDefaultFolder=lambda n: calendar)
    app = types.SimpleNamespace(GetNamespace=lambda mapi: namespace)

    win32com = types.ModuleType("win32com")
    client = types.ModuleType("win32com.client")
    client.Dispatch = lambda progid: app
    win32com.client = client
    return pythoncom, win32com, client


class FirstMeetingTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()
        self.tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    def _at(self, hour, minute=0):
        return _FakeStart(datetime.datetime.combine(
            self.tomorrow, datetime.time(hour, minute)))

    def test_no_win32com_returns_empty(self):
        # pythoncom unimportable (as on a CI runner / non-Windows host) →
        # best-effort "". block_import forces the ImportError branch even though
        # pywin32 is installed on the Windows dev box.
        with block_import("pythoncom", "win32com"):
            self.assertEqual(self.mod._first_meeting_tomorrow(), "")

    def test_meeting_with_organizer_and_subject(self):
        appt = _FakeAppt(self._at(9, 30), subject="Design review",
                         organizer="Sam Contoso <sam@example.com>")
        pythoncom, win32com, client = _make_outlook([appt])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            out = self.mod._first_meeting_tomorrow()
        self.assertIn("9:30 AM", out)
        self.assertIn("with Sam Contoso", out)
        self.assertIn("Design review", out)

    def test_meeting_pm_no_subject(self):
        appt = _FakeAppt(self._at(14, 0), subject="", organizer="")
        pythoncom, win32com, client = _make_outlook([appt])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            out = self.mod._first_meeting_tomorrow()
        self.assertIn("2:00 PM", out)
        self.assertNotIn("--", out)        # no subject suffix
        self.assertNotIn("with", out)      # no organizer

    def test_meeting_organizer_email_only_is_dropped(self):
        # Organizer that is just an email address → no "with X" clause.
        appt = _FakeAppt(self._at(8, 5), subject="Sync",
                         organizer="someone@example.com")
        pythoncom, win32com, client = _make_outlook([appt])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            out = self.mod._first_meeting_tomorrow()
        self.assertIn("8:05 AM", out)
        self.assertNotIn("with", out)

    def test_appt_outside_window_skipped(self):
        # An appt on the day AFTER tomorrow is filtered out → "".
        far = _FakeStart(datetime.datetime.combine(
            self.tomorrow + datetime.timedelta(days=2), datetime.time(9, 0)))
        appt = _FakeAppt(far, subject="Too far", organizer="")
        pythoncom, win32com, client = _make_outlook([appt])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            self.assertEqual(self.mod._first_meeting_tomorrow(), "")

    def test_restrict_unsupported_falls_back_to_all_items(self):
        # items.Restrict() raising is swallowed; iteration still finds the appt.
        appt = _FakeAppt(self._at(10, 15), subject="Standup", organizer="")
        pythoncom, win32com, client = _make_outlook([appt], restrict_raises=True)
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            out = self.mod._first_meeting_tomorrow()
        self.assertIn("10:15 AM", out)

    def test_dispatch_failure_returns_empty(self):
        pythoncom, win32com, client = _make_outlook([])
        client.Dispatch = mock.MagicMock(side_effect=RuntimeError("no Outlook"))
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            self.assertEqual(self.mod._first_meeting_tomorrow(), "")

    def test_bad_appt_is_skipped_then_good_one_used(self):
        # First appt raises while reading .Start (caught per-appt, continue),
        # second is valid.
        class _Boom:
            @property
            def Start(self):
                raise RuntimeError("corrupt item")
        good = _FakeAppt(self._at(11, 45), subject="Review", organizer="")
        pythoncom, win32com, client = _make_outlook([_Boom(), good])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            out = self.mod._first_meeting_tomorrow()
        self.assertIn("11:45 AM", out)

    def test_start_already_datetime_no_format(self):
        # When appt.Start is a plain datetime (no COM .Format attr), it's used
        # as-is (the `start_dt = start` branch) instead of being rebuilt.
        plain = datetime.datetime.combine(self.tomorrow, datetime.time(7, 0))
        appt = _FakeAppt(plain, subject="Early sync", organizer="")
        pythoncom, win32com, client = _make_outlook([appt])
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            out = self.mod._first_meeting_tomorrow()
        self.assertIn("7:00 AM", out)
        self.assertIn("Early sync", out)

    def test_com_init_failures_are_swallowed(self):
        # CoInitialize / CoUninitialize raising (already-initialised apartment,
        # teardown hiccup) must not break the query nor propagate.
        appt = _FakeAppt(self._at(9, 0), subject="Kickoff", organizer="")
        pythoncom, win32com, client = _make_outlook([appt], com_init_raises=True)
        with inject_modules(pythoncom=pythoncom, win32com=win32com,
                            **{"win32com.client": client}):
            out = self.mod._first_meeting_tomorrow()
        self.assertIn("9:00 AM", out)
        self.assertIn("Kickoff", out)


# ─────────────────────────────────────────────────────────────────────────
# Bambu cross-skill read — remaining branches.
# ─────────────────────────────────────────────────────────────────────────
class BambuStatusTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    def _monitor(self, state, strip=lambda s: s, lock=None):
        fake = mock.MagicMock()
        fake._state_lock = lock
        fake._state = state
        fake._strip_filename = strip
        return fake

    def test_finish_recent_with_filename(self):
        fake = self._monitor(
            {"last_update": time.time() - 600, "gcode_state": "FINISH",
             "filename": "bracket.3mf"}, strip=lambda s: "bracket")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("finished 'bracket' earlier today", out)

    def test_finish_recent_without_filename(self):
        fake = self._monitor(
            {"last_update": time.time() - 600, "gcode_state": "FINISH",
             "filename": ""}, strip=lambda s: "")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertIn("finished its print earlier today", self.mod._bambu_status())

    def test_finish_stale_is_silent(self):
        # Finished > 12h ago → not worth mentioning at 22:00.
        fake = self._monitor(
            {"last_update": time.time() - 13 * 3600, "gcode_state": "FINISH",
             "filename": "x.3mf"}, strip=lambda s: "x")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_status(), "")

    def test_running_hours_only_remaining(self):
        # mc_remaining of exactly 120 → "2 hours remaining" (rm == 0 branch).
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "", "layer_num": 0, "total_layer": 0,
             "mc_remaining": 120}, strip=lambda s: "")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("print in progress", out)
        self.assertIn("about 2 hours remaining", out)

    def test_running_single_hour_and_minutes(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "PREPARE",
             "filename": "part.gcode", "layer_num": 3, "total_layer": 40,
             "mc_remaining": 75}, strip=lambda s: "part")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("about 1 hour and 15 minutes remaining", out)

    def test_running_uses_state_lock(self):
        # A real lock is acquired/released around the dict() copy.
        import threading
        lock = threading.Lock()
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "", "layer_num": 1, "total_layer": 2,
             "mc_remaining": 10}, strip=lambda s: "", lock=lock)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("10 minutes remaining", out)
        self.assertFalse(lock.locked())

    def test_state_none_returns_empty(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = None
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_status(), "")

    def test_unknown_gcode_state_returns_empty(self):
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "IDLE", "filename": ""},
            strip=lambda s: "")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_status(), "")

    def test_read_exception_returns_empty(self):
        # Accessing _state raises → swallowed → "".
        fake = mock.MagicMock()
        type(fake)._state_lock = mock.PropertyMock(side_effect=RuntimeError("boom"))
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_status(), "")

    def test_last_update_zero_returns_empty(self):
        # A populated state that never received its first poll (last_update 0.0)
        # → no print info to report.
        fake = self._monitor(
            {"last_update": 0.0, "gcode_state": "RUNNING", "filename": "x.3mf"},
            strip=lambda s: "x")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_status(), "")

    def test_strip_filename_raises_is_swallowed(self):
        # _strip_filename blowing up must not abort the status line.
        def _boom(_s):
            raise RuntimeError("bad name")
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "x.3mf", "layer_num": 2, "total_layer": 9,
             "mc_remaining": 5}, strip=_boom)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("print in progress", out)   # fname stayed ""
        self.assertIn("layer 2 of 9", out)

    def test_non_numeric_remaining_is_swallowed(self):
        # mc_remaining that can't be int()'d → the remaining clause is skipped,
        # not raised (int("soon") → ValueError → pass).
        fake = self._monitor(
            {"last_update": time.time(), "gcode_state": "RUNNING",
             "filename": "", "layer_num": 4, "total_layer": 8,
             "mc_remaining": "soon"}, strip=lambda s: "")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("layer 4 of 8", out)
        self.assertNotIn("remaining", out)


# ─────────────────────────────────────────────────────────────────────────
# face_tracker presence read (_user_at_desk).
# ─────────────────────────────────────────────────────────────────────────
class UserAtDeskTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    def _ft(self, snap):
        fake = mock.MagicMock()
        fake._snapshot_state.return_value = snap
        return fake

    def test_present(self):
        fake = self._ft({"last_sample_at": time.time(), "current_monitor": "left"})
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIs(self.mod._user_at_desk(), True)

    def test_away(self):
        fake = self._ft({"last_sample_at": time.time(), "current_monitor": "away"})
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIs(self.mod._user_at_desk(), False)

    def test_unknown_monitor_value(self):
        fake = self._ft({"last_sample_at": time.time(), "current_monitor": "elsewhere"})
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIsNone(self.mod._user_at_desk())

    def test_no_sample_yet(self):
        fake = self._ft({"last_sample_at": 0, "current_monitor": "left"})
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIsNone(self.mod._user_at_desk())

    def test_not_loaded(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_face_tracker", None)
            self.assertIsNone(self.mod._user_at_desk())

    def test_no_snapshot_func(self):
        fake = mock.MagicMock(spec=[])   # no _snapshot_state attribute
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIsNone(self.mod._user_at_desk())

    def test_snapshot_raises(self):
        fake = mock.MagicMock()
        fake._snapshot_state.side_effect = RuntimeError("boom")
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIsNone(self.mod._user_at_desk())


# ─────────────────────────────────────────────────────────────────────────
# Session-log scraping + task counting (filesystem-backed counters).
# ─────────────────────────────────────────────────────────────────────────
class LogScrapingTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    def test_todays_log_paths_globs_pattern(self):
        with mock.patch.object(self.mod.glob, "glob",
                               return_value=["b.log", "a.log"]) as g:
            out = self.mod._todays_log_paths()
        self.assertEqual(out, ["a.log", "b.log"])   # sorted
        # Pattern embeds today's ISO date.
        today_iso = datetime.date.today().isoformat()
        self.assertIn(today_iso, g.call_args[0][0])

    def test_count_voice_interactions_counts_needle(self):
        log = ("  You:    hello there\n"
               "[action] foo:\n"
               "  You:    play jazz\n"
               "  JARVIS: sure\n")
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["s.log"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=log)):
            self.assertEqual(self.mod._count_voice_interactions_today(), 2)

    def test_count_voice_interactions_open_error_skips(self):
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["x.log"]), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._count_voice_interactions_today(), 0)

    def test_count_tasks_no_todo_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._count_tasks_completed_today(), 0)

    def test_count_tasks_open_error_returns_zero(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._count_tasks_completed_today(), 0)

    def test_scan_open_error_skips_path(self):
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["x.log"]), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            actions, plays, you = self.mod._scan_today_for_patterns()
        self.assertEqual((len(actions), len(plays), you), (0, 0, 0))

    def test_scan_play_stopword_filtered(self):
        # "play music" → stopword, not counted as a play target.
        log = "  You:    play music\n  You:    play music\n"
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["x.log"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=log)):
            _actions, plays, you = self.mod._scan_today_for_patterns()
        self.assertEqual(you, 2)
        self.assertEqual(len(plays), 0)


# ─────────────────────────────────────────────────────────────────────────
# News / umbrella section builders (over news_briefing / weather_briefing).
# ─────────────────────────────────────────────────────────────────────────
class NewsAndUmbrellaTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    def test_fetch_news_returns_text(self):
        nb = types.ModuleType("news_briefing")
        nb.get_news_text = lambda: "Today's headlines, sir. Markets up."
        with inject_modules(**{"skills.news_briefing": nb, "news_briefing": nb}):
            out = self.mod._fetch_news()
        self.assertIn("headlines", out)

    def test_fetch_news_module_unavailable(self):
        # Both `from . import news_briefing` and the bare-import fallback fail →
        # graceful "" (real module is blocked so no feed/LLM network is hit).
        with block_import("news_briefing"):
            self.assertEqual(self.mod._fetch_news(), "")

    def test_fetch_news_getter_raises(self):
        nb = types.ModuleType("news_briefing")
        def _boom():
            raise RuntimeError("all feeds down")
        nb.get_news_text = _boom
        with inject_modules(**{"skills.news_briefing": nb, "news_briefing": nb}):
            self.assertEqual(self.mod._fetch_news(), "")

    def test_fetch_umbrella_returns_alert(self):
        wb = types.ModuleType("weather_briefing")
        wb.get_umbrella_alert = lambda when: f"Bring an umbrella {when}, sir."
        with inject_modules(**{"skills.weather_briefing": wb, "weather_briefing": wb}):
            out = self.mod._fetch_tomorrow_umbrella()
        self.assertIn("umbrella tomorrow", out)

    def test_fetch_umbrella_module_unavailable(self):
        # weather_briefing import blocked on both paths → graceful "".
        with block_import("weather_briefing"):
            self.assertEqual(self.mod._fetch_tomorrow_umbrella(), "")

    def test_fetch_umbrella_getter_raises(self):
        wb = types.ModuleType("weather_briefing")
        def _boom(_when):
            raise RuntimeError("open-meteo down")
        wb.get_umbrella_alert = _boom
        with inject_modules(**{"skills.weather_briefing": wb, "weather_briefing": wb}):
            self.assertEqual(self.mod._fetch_tomorrow_umbrella(), "")


# ─────────────────────────────────────────────────────────────────────────
# _build_briefing — remaining branches (singular interaction, umbrella line).
# ─────────────────────────────────────────────────────────────────────────
class BuildBriefingBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    def _build(self, **over):
        defaults = dict(
            _count_voice_interactions_today=0, _count_tasks_completed_today=0,
            _bambu_status="", _fetch_tomorrow_weather="", _first_meeting_tomorrow="",
            _dry_observation="", _fetch_news="", _fetch_tomorrow_umbrella="",
        )
        defaults.update(over)
        cms = [mock.patch.object(self.mod, name,
                                 return_value=val if not callable(val) else val)
               for name, val in defaults.items()]
        with contextlib.ExitStack() as stack:
            for cm in cms:
                stack.enter_context(cm)
            return self.mod._build_briefing()

    def test_single_interaction_singular_phrasing(self):
        out = self._build(_count_voice_interactions_today=1)
        self.assertIn("One voice interaction logged today", out)

    def test_single_task_singular_plural(self):
        out = self._build(_count_voice_interactions_today=3,
                          _count_tasks_completed_today=1)
        self.assertIn("1 task cleared", out)   # "task" singular

    def test_umbrella_line_included(self):
        out = self._build(_count_voice_interactions_today=2,
                          _fetch_tomorrow_umbrella="An umbrella may serve you well, sir.")
        self.assertIn("An umbrella may serve you well", out)

    def test_observation_capitalised_no_news(self):
        out = self._build(_count_voice_interactions_today=2,
                          _dry_observation="you said 'play X' three times today, sir.")
        self.assertIn("You said 'play X' three times today, sir.", out)
        self.assertFalse(out.startswith("[intent:briefing]"))   # no news → no tag

    def test_only_weather_tomorrow_segment(self):
        out = self._build(_count_voice_interactions_today=2,
                          _fetch_tomorrow_weather="tomorrow looks like a high of 18")
        self.assertIn("For tomorrow, tomorrow looks like a high of 18.", out)


# ─────────────────────────────────────────────────────────────────────────
# Speech queue + config + persistence helpers.
# ─────────────────────────────────────────────────────────────────────────
class SpeechQueueAndStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    # ── _enqueue_speech ──────────────────────────────────────────────────
    def test_enqueue_uses_bc_announcer(self):
        bc = types.ModuleType("bobert_companion")
        calls = []
        bc.proactive_announce = lambda msg, source=None: calls.append((msg, source))
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.mod._enqueue_speech("Good evening, sir.")
        self.assertEqual(calls, [("Good evening, sir.", "evening")])

    def test_enqueue_falls_back_to_atomic_write(self):
        # bc present but WITHOUT proactive_announce → direct atomic write path.
        bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("hello sir")
        wr.assert_called_once()
        path, data = wr.call_args[0][0], wr.call_args[0][1]
        self.assertEqual(path, self.mod._SPEECH_QUEUE)
        self.assertEqual(data[-1]["message"], "hello sir")

    def test_enqueue_appends_to_existing_queue(self):
        bc = types.ModuleType("bobert_companion")
        existing = json.dumps([{"ts": 1.0, "message": "old"}])
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=existing)), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("new")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_enqueue_corrupt_queue_resets_to_list(self):
        bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{not json")), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("fresh")
        data = wr.call_args[0][1]
        self.assertEqual([d["message"] for d in data], ["fresh"])

    def test_enqueue_atomic_write_failure_is_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only share")):
            # Must not raise — falls back to a console print.
            self.mod._enqueue_speech("resilient")

    def test_enqueue_announcer_raises_falls_through(self):
        # A broken proactive_announce must not silence the briefing; it falls
        # through to the local atomic write.
        bc = types.ModuleType("bobert_companion")
        def _boom(*_a, **_k):
            raise RuntimeError("announcer broke")
        bc.proactive_announce = _boom
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._enqueue_speech("still spoken")
        wr.assert_called_once()

    # ── _read_config ─────────────────────────────────────────────────────
    def test_read_config_from_bc(self):
        bc = types.ModuleType("bobert_companion")
        bc.EVENING_BRIEFING_ENABLED = False
        bc.EVENING_BRIEFING_HOUR = 21
        bc.EVENING_BRIEFING_MINUTE = 15
        bc.EVENING_BRIEFING_WAIT_MINUTES = 5
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": False, "hour": 21, "minute": 15, "wait_min": 5})

    def test_read_config_defaults_when_bc_import_fails(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": True, "hour": 22, "minute": 0, "wait_min": 30})

    # ── last-fired persistence ───────────────────────────────────────────
    def test_load_last_fired_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_load_last_fired_reads_value(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data='{"last_fired_date": "2026-06-01"}')):
            self.assertEqual(self.mod._load_last_fired_date(), "2026-06-01")

    def test_load_last_fired_corrupt_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{bad")):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_save_last_fired_writes_atomic(self):
        with mock.patch.object(self.mod, "_atomic_write_json") as wr:
            self.mod._save_last_fired_date("2026-06-01")
        wr.assert_called_once_with(self.mod._STATE_FILE, {"last_fired_date": "2026-06-01"})

    def test_save_last_fired_write_failure_swallowed(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._save_last_fired_date("2026-06-01")   # must not raise


# ─────────────────────────────────────────────────────────────────────────
# HUD card helper.
# ─────────────────────────────────────────────────────────────────────────
class ShowCardTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    def test_show_card_calls_hud(self):
        hud = types.ModuleType("hud_card")
        calls = []
        hud.show_card = lambda which: calls.append(which)
        with inject_modules(hud_card=hud):
            self.mod._show_card_safe()
        self.assertEqual(calls, ["evening"])

    def test_show_card_failure_is_swallowed(self):
        hud = types.ModuleType("hud_card")
        def _boom(_which):
            raise RuntimeError("no display")
        hud.show_card = _boom
        with inject_modules(hud_card=hud):
            self.mod._show_card_safe()   # must not raise

    def test_show_card_import_failure_is_swallowed(self):
        with inject_modules(hud_card=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no hud_card")):
            self.mod._show_card_safe()   # must not raise

    def test_show_card_inserts_project_dir_on_path(self):
        # When _PROJECT_DIR isn't yet on sys.path, the helper prepends it before
        # importing hud_card. Use a sentinel dir and restore sys.path after.
        sentinel = "C:/eb-fake-project-dir"
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        hud = types.ModuleType("hud_card")
        hud.show_card = lambda which: None
        with mock.patch.object(self.mod, "_PROJECT_DIR", sentinel), \
             inject_modules(hud_card=hud):
            self.mod._show_card_safe()
        self.assertIn(sentinel, sys.path)


# ─────────────────────────────────────────────────────────────────────────
# Scheduler-loop gating arithmetic + _wait_for_presence + _fire_briefing.
# Each scheduler test runs exactly ONE full loop iteration, broken out of the
# `while True` at the top of the second pass (see _run_one_iteration). The
# sentinel derives from BaseException so the loop's own `except Exception`
# handler does NOT swallow it.
# ─────────────────────────────────────────────────────────────────────────
class _LoopBreak(BaseException):
    pass


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load()

    # ── _wait_for_presence ───────────────────────────────────────────────
    def test_wait_for_presence_none_returns_false_fast(self):
        # face_tracker has no data (None) → bail immediately, no polling.
        with mock.patch.object(self.mod, "_user_at_desk", return_value=None), \
             mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertFalse(self.mod._wait_for_presence(120))
        slp.assert_not_called()

    def test_wait_for_presence_detects_user(self):
        with mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertTrue(self.mod._wait_for_presence(120))
        slp.assert_not_called()   # present on first poll → no sleep

    def test_wait_for_presence_times_out(self):
        # First read non-None (False) so we enter the loop; clock advances past
        # the deadline on the second time() read so the loop exits → False.
        reads = iter([False, False, False])
        with mock.patch.object(self.mod, "_user_at_desk",
                               side_effect=lambda: next(reads, False)), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=[1000.0, 1000.0, 2000.0]), \
             mock.patch.object(self.mod.time, "sleep"):
            self.assertFalse(self.mod._wait_for_presence(30))

    def test_wait_for_presence_appears_after_one_poll(self):
        # Guard read False (enters loop); poll#1 False → one sleep; poll#2 True
        # → returns True. time() stays under the deadline (start+30) throughout.
        reads = iter([False, False, True])
        with mock.patch.object(self.mod, "_user_at_desk",
                               side_effect=lambda: next(reads)), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=[1000.0, 1001.0, 1002.0]), \
             mock.patch.object(self.mod.time, "sleep") as slp:
            self.assertTrue(self.mod._wait_for_presence(30))
        slp.assert_called_once()

    # ── _fire_briefing ───────────────────────────────────────────────────
    def test_fire_briefing_pipeline(self):
        with mock.patch.object(self.mod, "_build_briefing", return_value="Good evening, sir."), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_show_card_safe") as card, \
             mock.patch.object(self.mod, "_save_last_fired_date") as save:
            out = self.mod._fire_briefing("user-present")
        self.assertEqual(out, "Good evening, sir.")
        enq.assert_called_once_with("Good evening, sir.")
        card.assert_called_once()
        save.assert_called_once()

    # ── _scheduler_loop single-iteration gating ──────────────────────────
    def _run_one_iteration(self, **patches):
        """Drive _scheduler_loop through exactly one full pass of the body.

        ``time.sleep`` is a harmless no-op; the loop is instead broken at the
        TOP of the second pass by having ``_read_config`` raise ``_LoopBreak``
        on its second call. This lets the entire first-iteration body run —
        including the trailing ``time.sleep(POLL); continue`` statements that a
        break-at-sleep strategy would skip — before the loop unwinds."""
        provided_cfg = patches.pop("_read_config", None)
        if provided_cfg is None:
            provided_cfg = mock.MagicMock(return_value={
                "enabled": True, "hour": 22, "minute": 0, "wait_min": 30})
        calls = {"n": 0}

        def _cfg_then_break(*a, **k):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise _LoopBreak
            return provided_cfg(*a, **k)

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(self.mod.time, "sleep", lambda *_a: None))
        stack.enter_context(mock.patch.object(self.mod, "_read_config", _cfg_then_break))
        for name, val in patches.items():
            if name == "now":
                continue
            stack.enter_context(mock.patch.object(self.mod, name, val))
        if "now" in patches:
            fake_dt = mock.MagicMock()
            fake_dt.now.return_value = patches["now"]
            stack.enter_context(mock.patch.object(self.mod.datetime, "datetime", fake_dt))
        with stack:
            with self.assertRaises(_LoopBreak):
                self.mod._scheduler_loop()

    def test_loop_disabled_skips(self):
        fire = mock.MagicMock()
        self._run_one_iteration(
            _read_config=mock.MagicMock(return_value={"enabled": False, "hour": 22,
                                                      "minute": 0, "wait_min": 30}),
            _fire_briefing=fire)
        fire.assert_not_called()

    def test_loop_already_fired_today_skips(self):
        today = datetime.date.today().isoformat()
        now = datetime.datetime.now().replace(hour=23, minute=0, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 22,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=today),
            _fire_briefing=fire)
        fire.assert_not_called()

    def test_loop_before_scheduled_time_skips(self):
        now = datetime.datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 22,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _fire_briefing=fire)
        fire.assert_not_called()

    def test_loop_past_catchup_window_marks_done(self):
        # scheduled 02:00, now 05:00 same day → 180 min late (> CATCHUP 120) →
        # mark today done WITHOUT firing, so a stale slot can't drop a briefing
        # into the small hours.
        now = datetime.datetime.now().replace(hour=5, minute=0, second=0, microsecond=0)
        fire = mock.MagicMock()
        save = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 2,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _save_last_fired_date=save,
            _fire_briefing=fire)
        fire.assert_not_called()
        save.assert_called_once_with(now.date().isoformat())

    def test_loop_fires_when_user_present(self):
        now = datetime.datetime.now().replace(hour=22, minute=5, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 22,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _wait_for_presence=mock.MagicMock(return_value=True),
            _fire_briefing=fire)
        fire.assert_called_once_with("user-present")

    def test_loop_recheck_after_wait_suppresses_double_fire(self):
        # TOCTOU regression: the manual action (or a second instance) fires the
        # briefing DURING the presence wait. The post-wait re-check of the
        # same-day flag must suppress the scheduler's fire.
        now = datetime.datetime.now().replace(hour=22, minute=5, second=0, microsecond=0)
        today = now.date().isoformat()
        reads = iter(["", today])  # pre-check clean, post-wait already fired
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 22,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(side_effect=lambda: next(reads, today)),
            _wait_for_presence=mock.MagicMock(return_value=True),
            _fire_briefing=fire)
        fire.assert_not_called()

    def test_loop_fires_timed_out_when_no_presence(self):
        now = datetime.datetime.now().replace(hour=22, minute=5, second=0, microsecond=0)
        fire = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 22,
                                                      "minute": 0, "wait_min": 30}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _wait_for_presence=mock.MagicMock(return_value=False),
            _fire_briefing=fire)
        fire.assert_called_once_with("timed-out")

    def test_loop_fires_immediately_when_wait_zero(self):
        # wait_min == 0 → presence not polled; fires "timed-out".
        now = datetime.datetime.now().replace(hour=22, minute=5, second=0, microsecond=0)
        fire = mock.MagicMock()
        wait = mock.MagicMock()
        self._run_one_iteration(
            now=now,
            _read_config=mock.MagicMock(return_value={"enabled": True, "hour": 22,
                                                      "minute": 0, "wait_min": 0}),
            _load_last_fired_date=mock.MagicMock(return_value=""),
            _wait_for_presence=wait,
            _fire_briefing=fire)
        wait.assert_not_called()
        fire.assert_called_once_with("timed-out")

    def test_loop_swallows_body_exception(self):
        # An exception inside the body is logged, not propagated; the loop then
        # hits the trailing sleep (which raises our sentinel).
        self._run_one_iteration(
            _read_config=mock.MagicMock(side_effect=RuntimeError("config boom")))


# ─────────────────────────────────────────────────────────────────────────
# register() — action wiring + long-running / mid-task-status bridge.
# ─────────────────────────────────────────────────────────────────────────
class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = _load(register=False)

    def test_register_adds_action_and_starts_thread_when_enabled(self):
        bc = types.ModuleType("bobert_companion")
        bc.EVENING_BRIEFING_ENABLED = True
        bc.EVENING_BRIEFING_HOUR = 22
        bc.EVENING_BRIEFING_MINUTE = 0
        bc.EVENING_BRIEFING_WAIT_MINUTES = 30
        bc.LONG_RUNNING_ACTIONS = set()
        bc._MID_TASK_STATUS_BUCKET = {}
        actions = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        self.assertIn("evening_briefing", actions)
        # Mid-task-status bridge wired up.
        self.assertIn("evening_briefing", bc.LONG_RUNNING_ACTIONS)
        self.assertEqual(bc._MID_TASK_STATUS_BUCKET["evening_briefing"], "_generic")
        # Scheduler thread constructed daemon=True and started.
        Thread.assert_called_once()
        self.assertTrue(Thread.call_args.kwargs.get("daemon"))
        Thread.return_value.start.assert_called_once()

    def test_register_disabled_skips_thread(self):
        bc = types.ModuleType("bobert_companion")
        bc.EVENING_BRIEFING_ENABLED = False
        bc.EVENING_BRIEFING_HOUR = 22
        bc.EVENING_BRIEFING_MINUTE = 0
        bc.EVENING_BRIEFING_WAIT_MINUTES = 30
        bc.LONG_RUNNING_ACTIONS = set()
        bc._MID_TASK_STATUS_BUCKET = {}
        actions = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        self.assertIn("evening_briefing", actions)   # action still registered
        Thread.assert_not_called()                   # but no scheduler thread

    def test_register_mid_task_bridge_failure_is_swallowed(self):
        # bc import inside the bridge raises → caught; action still registered,
        # and (enabled default True) the thread is still constructed.
        bc = types.ModuleType("bobert_companion")
        bc.EVENING_BRIEFING_ENABLED = True
        bc.EVENING_BRIEFING_HOUR = 22
        bc.EVENING_BRIEFING_MINUTE = 0
        bc.EVENING_BRIEFING_WAIT_MINUTES = 30
        actions = {}
        real_import = self.mod.importlib.import_module

        def _imp(name, *a, **k):
            # Let _read_config's import_module work; break only the bridge's
            # `import bobert_companion as _bc` (a builtins.__import__ call).
            return real_import(name, *a, **k)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod.threading, "Thread"), \
             mock.patch("builtins.__import__", side_effect=ImportError("blocked")):
            with mock.patch.object(self.mod, "_read_config",
                                   return_value={"enabled": True, "hour": 22,
                                                 "minute": 0, "wait_min": 30}):
                self.mod.register(actions)
        self.assertIn("evening_briefing", actions)


if __name__ == "__main__":
    unittest.main()
