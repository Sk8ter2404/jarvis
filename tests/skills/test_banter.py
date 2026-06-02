"""Logic tests for skills/banter.py.

Banter watches the command history + ambient signals for behavioural "tells"
and, gated by cadence/quiet-hours/call-suppression + a probability roll, drops
ONE dry zinger into the pending_speech queue. This suite covers:

  • text normalization, the four tell detectors (repeat_question / repeat_open /
    tab_clutter[+windows] / music_while_music) across hit, miss, and edge paths,
  • zinger selection + placeholder formatting (every variant renders; the
    format-failure fallback),
  • environment probes (_all_window_titles / _is_in_call / _is_sleep_or_standby /
    _chrome_process_count / _visible_window_count) via injected fake modules,
  • pattern_memory access bridges (_load_voice_commands / _extract_target_safe),
  • config read + persistent state load/save (round-trip, corruption,
    save-failure paths),
  • the speech-queue writer (fresh / append / corrupt-reset / write-failure),
  • the whole _scheduler_loop across every gate + trigger + persistence (time,
    random and sleep all frozen; a sentinel raised from sleep stops the
    otherwise-infinite loop after one pass),
  • _format_status branches, the banter_status action, and register()
    (enabled / disabled / status-callable).

ISOLATION CONTRACT (critical): banter's register() and _read_config call
``importlib.import_module("bobert_companion")``. With the real ~14K-line
monolith absent from sys.modules that import EXECUTES it for real — and its
module-level boot runs a singleton lock that can ``sys.exit``, aborting the run.
So every test class seeds a *minimal fake* ``bobert_companion`` into sys.modules
BEFORE the skill is loaded and restores the prior state (including absence) in
tearDown — and asserts the real monolith never got imported. Mirrors the
injection contract of tests/skills/test_self_diagnostic.py /
tests/skills/test_anticipation_engine.py. No real LLM/network/thread/sleep; real
numpy is left untouched; no personal data in any fixture.
"""
from __future__ import annotations

import contextlib
import datetime
import io
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


# ─── fake-module injection (mirrors test_self_diagnostic.inject_modules) ────
@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules and, for dotted names,
    set the leaf as an attribute on its already-imported parent package. Restores
    the previous state — INCLUDING ABSENCE — on exit so the skill sees exactly
    the fake we provide and tests stay isolated. Pass ``name=None`` to force a
    module to look absent for the duration of the block."""
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


@contextlib.contextmanager
def block_import(*names):
    """Force a bare ``import <name>`` to raise ImportError inside the block, even
    when the real dependency is installed on the dev box — banter's probes use a
    bare ``import pygetwindow`` / ``import psutil`` (not importlib), so a None
    sys.modules entry alone is fragile across re-imports. We detach any
    already-imported target AND patch ``__import__`` to raise, restoring both."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved_mod: dict[str, object] = {}
    for name in blocked:
        if name in sys.modules:
            saved_mod[name] = sys.modules.pop(name)
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for name, mod in saved_mod.items():
            sys.modules[name] = mod


def _benign_bc():
    """Minimal fake ``bobert_companion`` exposing only what banter reads: the
    three config knobs (defaults keep the engine ENABLED) plus the sleep/standby
    and music slots the gates consult via ``sys.modules.get``. A plain
    ModuleType with just these attrs — the real monolith is never touched."""
    bc = types.ModuleType("bobert_companion")
    bc.BANTER_ENABLED = True
    bc.BANTER_COOLDOWN_MINUTES = 30
    bc.BANTER_PER_TELL_COOLDOWN_MINUTES = 180
    # Gate slots default to "not sleeping / no music" so an unconfigured fake is
    # permissive; individual tests override as needed.
    bc._sleep_mode = [False]
    bc._standby_mode = [False]
    bc._jarvis_played_music_at = [0.0]
    return bc


class _StopLoop(Exception):
    """Sentinel raised from a patched time.sleep to break _scheduler_loop after
    a fixed number of poll iterations."""


def _sleep_after(n):
    """Return a fake time.sleep that no-ops the first ``n`` calls then raises
    _StopLoop — lets a bounded number of scheduler passes complete then exits."""
    box = {"n": n}

    def _fake(_seconds):
        if box["n"] <= 0:
            raise _StopLoop()
        box["n"] -= 1
    return _fake


def _entry(text, ago_seconds, iso_date=None):
    ts = time.time() - ago_seconds
    iso = (iso_date or datetime.date.today().isoformat()) + "T12:00:00"
    return {"ts": ts, "iso": iso, "text": text}


class _BanterTestBase(unittest.TestCase):
    """Loads the skill in isolation WITH a benign fake ``bobert_companion``
    pre-seeded, so register()/_read_config's ``import_module('bobert_companion')``
    resolves the fake instead of executing (and possibly ``sys.exit``-ing in) the
    real monolith. Restores sys.modules — including prior absence — in tearDown,
    and asserts the real monolith was never imported as a side effect of the test.
    """
    def setUp(self):
        self._seeded_prev = sys.modules.get("bobert_companion", _SENTINEL)
        sys.modules["bobert_companion"] = _benign_bc()
        self.mod, self.actions = load_skill_isolated("banter")
        # A few paths print on failure (speech-queue write, state save, scheduler
        # firing). Tests assert on return values / side effects, so swallow
        # direct-call stdout to keep the runner output clean.
        _out = contextlib.redirect_stdout(io.StringIO())
        _out.__enter__()
        self.addCleanup(_out.__exit__, None, None, None)

    def tearDown(self):
        cur = sys.modules.get("bobert_companion")
        # The fake we seeded (or a fake a test swapped in) must still be a plain
        # ModuleType — i.e. the real ~14K-line monolith never got imported over
        # the top of it during the test.
        if cur is not None:
            self.assertIsInstance(
                cur, types.ModuleType,
                "real bobert_companion monolith leaked into sys.modules")
            self.assertFalse(
                hasattr(cur, "__file__") and
                os.path.basename(getattr(cur, "__file__", "") or "")
                == "bobert_companion.py",
                "real bobert_companion.py was imported — isolation breached")
        # Restore prior state (including absence).
        if self._seeded_prev is _SENTINEL:
            sys.modules.pop("bobert_companion", None)
        else:
            sys.modules["bobert_companion"] = self._seeded_prev


# ─── text normalization ────────────────────────────────────────────────────
class BanterNormalizeTests(_BanterTestBase):
    def test_normalize_strips_punct_and_lowercases(self):
        self.assertEqual(self.mod._normalize_text("What's the WEATHER today!?"),
                         "whats the weather today")

    def test_normalize_collapses_whitespace(self):
        self.assertEqual(self.mod._normalize_text("  open    chrome  "),
                         "open chrome")

    def test_normalize_none_is_empty(self):
        self.assertEqual(self.mod._normalize_text(None), "")

    def test_normalize_empty(self):
        self.assertEqual(self.mod._normalize_text(""), "")


# ─── repeat_question detector ──────────────────────────────────────────────
class BanterRepeatQuestionTests(_BanterTestBase):
    def test_detects_near_duplicate_within_window(self):
        entries = [
            _entry("What's the weather today", 60),
            _entry("Whats the weather today!", 30),
        ]
        tell = self.mod._detect_repeat_question(entries)
        self.assertIsNotNone(tell)
        self.assertEqual(tell["tell"], "repeat_question")
        self.assertEqual(tell["n"], 2)
        self.assertEqual(tell["text"], "whats the weather today")
        self.assertEqual(tell["key"], "repeat_question:whats the weather today")
        self.assertGreaterEqual(tell["minutes"], 1)

    def test_ignores_short_commands(self):
        # < 3 words are excluded so "stop"/"next" don't trigger.
        entries = [_entry("stop", 60), _entry("stop", 30)]
        self.assertIsNone(self.mod._detect_repeat_question(entries))

    def test_ignores_old_entries(self):
        # Both outside the 10-min window.
        entries = [
            _entry("what is the weather today", 60 * 60),
            _entry("what is the weather today", 50 * 60),
        ]
        self.assertIsNone(self.mod._detect_repeat_question(entries))

    def test_single_occurrence_no_tell(self):
        self.assertIsNone(self.mod._detect_repeat_question(
            [_entry("what is the weather today", 60)]))

    def test_fewer_than_two_recent_short_circuits(self):
        # Only one entry inside the window → the len(recent) < 2 guard returns.
        entries = [
            _entry("what is the weather today", 30),
            _entry("what is the weather today", 60 * 60),   # old
        ]
        self.assertIsNone(self.mod._detect_repeat_question(entries))

    def test_entries_without_numeric_ts_ignored(self):
        # A non-numeric ts is filtered out of the recent window.
        bad = {"ts": "not-a-number", "iso": "x", "text": "what is the weather today"}
        entries = [bad, bad]
        self.assertIsNone(self.mod._detect_repeat_question(entries))

    def test_picks_most_repeated_bucket(self):
        # Two distinct repeated utterances; the more frequent one wins.
        entries = [
            _entry("please open the garage door", 50),
            _entry("please open the garage door", 40),
            _entry("please open the garage door", 30),
            _entry("what is the weather today", 20),
            _entry("what is the weather today", 10),
        ]
        tell = self.mod._detect_repeat_question(entries)
        self.assertEqual(tell["n"], 3)
        self.assertEqual(tell["text"], "please open the garage door")


# ─── repeat_open detector ──────────────────────────────────────────────────
class BanterRepeatOpenTests(_BanterTestBase):
    def test_detects_repeat_open_over_threshold(self):
        entries = [_entry("open chrome", 100 + i) for i in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "chrome")):
            tell = self.mod._detect_repeat_open(entries)
        self.assertIsNotNone(tell)
        self.assertEqual(tell["tell"], "repeat_open")
        self.assertEqual(tell["target"], "chrome")
        self.assertGreaterEqual(tell["n"], self.mod.REPEAT_OPEN_THRESHOLD)
        self.assertIn("repeat_open:chrome:", tell["key"])

    def test_under_threshold_no_tell(self):
        entries = [_entry("open chrome", 100 + i) for i in range(3)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "chrome")):
            self.assertIsNone(self.mod._detect_repeat_open(entries))

    def test_non_open_category_ignored(self):
        entries = [_entry("play jazz", 100 + i) for i in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("play", "jazz")):
            self.assertIsNone(self.mod._detect_repeat_open(entries))

    def test_empty_when_no_entries_today(self):
        # ISO date is yesterday → every entry skipped → counter empty → None.
        yesterday = (datetime.date.today() -
                     datetime.timedelta(days=1)).isoformat()
        entries = [_entry("open chrome", 100 + i, iso_date=yesterday)
                   for i in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "chrome")):
            self.assertIsNone(self.mod._detect_repeat_open(entries))

    def test_extract_returns_none_skips_entry(self):
        # _extract_target_safe → None exercises the `extracted is None: continue`.
        entries = [_entry("mystery phrase", 100 + i) for i in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=None):
            self.assertIsNone(self.mod._detect_repeat_open(entries))

    def test_non_string_iso_skipped(self):
        entries = [{"ts": time.time(), "iso": 12345, "text": "open chrome"}
                   for _ in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "chrome")):
            self.assertIsNone(self.mod._detect_repeat_open(entries))

    def test_empty_target_ignored(self):
        # cat == "open" but a falsy target hits the `or not target` branch.
        entries = [_entry("open", 100 + i) for i in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "")):
            self.assertIsNone(self.mod._detect_repeat_open(entries))


# ─── tab_clutter detector ──────────────────────────────────────────────────
class BanterTabClutterTests(_BanterTestBase):
    def test_chrome_clutter_detected(self):
        with mock.patch.object(self.mod, "_chrome_process_count", return_value=47), \
             mock.patch.object(self.mod, "_visible_window_count", return_value=3):
            tell = self.mod._detect_tab_clutter()
        self.assertEqual(tell["tell"], "tab_clutter")
        self.assertEqual(tell["n"], 47)
        self.assertEqual(tell["key"], "tab_clutter:chrome")

    def test_window_clutter_fallback(self):
        with mock.patch.object(self.mod, "_chrome_process_count", return_value=2), \
             mock.patch.object(self.mod, "_visible_window_count", return_value=55):
            tell = self.mod._detect_tab_clutter()
        self.assertEqual(tell["tell"], "tab_clutter_windows")
        self.assertEqual(tell["n"], 55)
        self.assertEqual(tell["key"], "tab_clutter:windows")

    def test_no_clutter(self):
        with mock.patch.object(self.mod, "_chrome_process_count", return_value=5), \
             mock.patch.object(self.mod, "_visible_window_count", return_value=10):
            self.assertIsNone(self.mod._detect_tab_clutter())

    def test_threshold_is_strict_greater_than(self):
        # Exactly at the threshold is NOT clutter (`> THRESHOLD`).
        thr = self.mod.TAB_CLUTTER_THRESHOLD
        with mock.patch.object(self.mod, "_chrome_process_count", return_value=thr), \
             mock.patch.object(self.mod, "_visible_window_count", return_value=thr):
            self.assertIsNone(self.mod._detect_tab_clutter())


# ─── music_while_music detector ────────────────────────────────────────────
class BanterMusicWhileMusicTests(_BanterTestBase):
    def test_no_bc_returns_none(self):
        with inject_modules(bobert_companion=None):
            self.assertIsNone(self.mod._detect_music_while_music([]))

    def test_detects_play_after_jarvis_started_music(self):
        fake_bc = _benign_bc()
        fake_bc._jarvis_played_music_at = [time.time() - 60]   # 1 min ago
        play_entry = _entry("play some jazz", 30)              # after that ts
        with inject_modules(bobert_companion=fake_bc), \
             mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("play", "jazz")):
            tell = self.mod._detect_music_while_music([play_entry])
        self.assertIsNotNone(tell)
        self.assertEqual(tell["tell"], "music_while_music")
        self.assertEqual(tell["target"], "jazz")
        self.assertTrue(tell["key"].startswith("music_while_music:"))

    def test_no_recent_jarvis_music(self):
        fake_bc = _benign_bc()
        fake_bc._jarvis_played_music_at = [time.time() - 99999]  # too old
        with inject_modules(bobert_companion=fake_bc):
            self.assertIsNone(self.mod._detect_music_while_music([]))

    def test_zero_timestamp_means_never_played(self):
        fake_bc = _benign_bc()
        fake_bc._jarvis_played_music_at = [0.0]
        with inject_modules(bobert_companion=fake_bc):
            self.assertIsNone(self.mod._detect_music_while_music([]))

    def test_missing_slot_returns_none(self):
        # No _jarvis_played_music_at attribute → stays 0.0 → None.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            self.assertIsNone(self.mod._detect_music_while_music([]))

    def test_bad_slot_type_raises_handled(self):
        # A list whose first element can't be coerced to float makes
        # float(ts_slot[0]) raise → the except returns None.
        bc = types.ModuleType("bobert_companion")
        bc._jarvis_played_music_at = ["not-a-float"]
        with inject_modules(bobert_companion=bc):
            self.assertIsNone(self.mod._detect_music_while_music([]))

    def test_non_list_slot_stays_zero(self):
        # A non-list slot fails the isinstance guard → last_jarvis_music stays
        # 0.0 → the <= 0 branch returns None (no exception raised).
        bc = types.ModuleType("bobert_companion")
        bc._jarvis_played_music_at = object()
        with inject_modules(bobert_companion=bc):
            self.assertIsNone(self.mod._detect_music_while_music([]))

    def test_recent_music_but_no_play_request(self):
        # JARVIS played recently, but the only entry is a non-play command.
        fake_bc = _benign_bc()
        fake_bc._jarvis_played_music_at = [time.time() - 60]
        entry = _entry("open chrome", 30)
        with inject_modules(bobert_companion=fake_bc), \
             mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "chrome")):
            self.assertIsNone(self.mod._detect_music_while_music([entry]))

    def test_play_request_before_jarvis_music_ignored(self):
        # The play utterance predates JARVIS's playback → not a layering tell.
        now = time.time()
        fake_bc = _benign_bc()
        fake_bc._jarvis_played_music_at = [now - 60]
        old_play = {"ts": now - 120, "iso": "x", "text": "play jazz"}  # before
        with inject_modules(bobert_companion=fake_bc), \
             mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("play", "jazz")):
            self.assertIsNone(self.mod._detect_music_while_music([old_play]))

    def test_extract_none_skips_candidate(self):
        fake_bc = _benign_bc()
        fake_bc._jarvis_played_music_at = [time.time() - 60]
        entry = _entry("garbled audio", 30)
        with inject_modules(bobert_companion=fake_bc), \
             mock.patch.object(self.mod, "_extract_target_safe", return_value=None):
            self.assertIsNone(self.mod._detect_music_while_music([entry]))


# ─── zinger selection / formatting ─────────────────────────────────────────
class BanterZingerTests(_BanterTestBase):
    def test_pick_zinger_formats_placeholders(self):
        line = self.mod._pick_zinger({"tell": "tab_clutter", "n": 47})
        self.assertIn("47", line)

    def test_pick_zinger_repeat_question_interpolates_count(self):
        nth_variant = "That's the {n}th time you've asked me that today, sir."
        with mock.patch.object(self.mod.random, "choice", return_value=nth_variant):
            line = self.mod._pick_zinger(
                {"tell": "repeat_question", "n": 3, "minutes": 5})
        self.assertEqual(line, "That's the 3th time you've asked me that today, sir.")

    def test_pick_zinger_every_variant_renders_for_every_tell(self):
        # No variant may crash on .format(**tell); each yields non-empty text
        # with all placeholders resolved. Build a tell dict carrying every key
        # any bank line references.
        sample = {
            "repeat_question": {"tell": "repeat_question", "n": 3, "minutes": 5},
            "repeat_open": {"tell": "repeat_open", "target": "Chrome", "n": 7},
            "tab_clutter": {"tell": "tab_clutter", "n": 47},
            "tab_clutter_windows": {"tell": "tab_clutter_windows", "n": 52},
            "music_while_music": {"tell": "music_while_music", "target": "jazz"},
        }
        for tell_name, variants in self.mod._ZINGER_BANK.items():
            tell = sample[tell_name]
            for variant in variants:
                with mock.patch.object(self.mod.random, "choice",
                                       return_value=variant):
                    line = self.mod._pick_zinger(tell)
                self.assertTrue(line, f"{tell_name} variant rendered empty")
                self.assertNotIn("{", line)   # all placeholders resolved

    def test_pick_zinger_unknown_tell_empty(self):
        self.assertEqual(self.mod._pick_zinger({"tell": "no_such_tell"}), "")

    def test_pick_zinger_format_failure_returns_raw_line(self):
        # A line with a placeholder the tell dict lacks → .format raises →
        # the except returns the unformatted line verbatim.
        bad_line = "Missing {nonexistent_key} here, sir."
        with mock.patch.object(self.mod.random, "choice", return_value=bad_line):
            out = self.mod._pick_zinger({"tell": "tab_clutter", "n": 1})
        self.assertEqual(out, bad_line)

    def test_zinger_bank_each_tell_has_variants(self):
        for tell, variants in self.mod._ZINGER_BANK.items():
            self.assertGreaterEqual(len(variants), 2,
                                    f"{tell} should have >=2 variants")


# ─── environment probes ────────────────────────────────────────────────────
def _fake_gw(all_titles=None, raise_all=False, raise_iter=False):
    """Fake ``pygetwindow``: getAllWindows() yields windows whose .title is each
    of ``all_titles``. ``raise_all`` makes getAllWindows itself raise;
    ``raise_iter`` makes a window's .title access raise (per-window guard)."""
    gw = types.ModuleType("pygetwindow")

    class _Win:
        def __init__(self, title, boom=False):
            self._boom = boom
            self._title = title

        @property
        def title(self):
            if self._boom:
                raise RuntimeError("title boom")
            return self._title

    def _get_all():
        if raise_all:
            raise RuntimeError("enum boom")
        wins = [_Win(t) for t in (all_titles or [])]
        if raise_iter:
            wins.append(_Win("x", boom=True))
        return wins

    gw.getAllWindows = _get_all
    return gw


class BanterEnvironmentTests(_BanterTestBase):
    # ── _all_window_titles ───────────────────────────────────────────────
    def test_all_window_titles_collects_nonblank(self):
        gw = _fake_gw(all_titles=["  ", "Notepad", "", "Outlook"])
        with inject_modules(pygetwindow=gw):
            out = self.mod._all_window_titles()
        self.assertEqual(out, ["Notepad", "Outlook"])

    def test_all_window_titles_no_pygetwindow(self):
        with block_import("pygetwindow"):
            self.assertEqual(self.mod._all_window_titles(), [])

    def test_all_window_titles_enum_raises(self):
        gw = _fake_gw(raise_all=True)
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._all_window_titles(), [])

    def test_all_window_titles_per_window_title_raises(self):
        # The inner getattr(w, "title", "") swallows a window whose title access
        # raises, keeping the good ones.
        gw = _fake_gw(all_titles=["Good Window"], raise_iter=True)
        with inject_modules(pygetwindow=gw):
            out = self.mod._all_window_titles()
        self.assertIn("Good Window", out)

    # ── _is_in_call ──────────────────────────────────────────────────────
    def test_is_in_call_matches_title(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Weekly sync | Microsoft Teams Meeting"]):
            self.assertTrue(self.mod._is_in_call())

    def test_is_in_call_zoom(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Zoom Meeting"]):
            self.assertTrue(self.mod._is_in_call())

    def test_is_in_call_false_when_no_meeting(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Inbox - Outlook", "Notepad"]):
            self.assertFalse(self.mod._is_in_call())

    def test_is_in_call_false_when_no_windows(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=[]):
            self.assertFalse(self.mod._is_in_call())

    def test_is_in_call_real_window_hit_via_fake_gw(self):
        # Drive the real loop body (not the _all_window_titles mock) end to end.
        gw = _fake_gw(all_titles=["Standup | Microsoft Teams Meeting", "Notepad"])
        with inject_modules(pygetwindow=gw):
            self.assertTrue(self.mod._is_in_call())

    # ── _is_sleep_or_standby ─────────────────────────────────────────────
    def test_sleep_mode_true(self):
        bc = _benign_bc()
        bc._sleep_mode = [True]
        with inject_modules(bobert_companion=bc):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_standby_mode_true(self):
        bc = _benign_bc()
        bc._standby_mode = [True]
        with inject_modules(bobert_companion=bc):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_sleep_standby_both_false(self):
        with inject_modules(bobert_companion=_benign_bc()):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_sleep_standby_no_bc(self):
        with inject_modules(bobert_companion=None):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_sleep_standby_attr_missing_returns_false(self):
        # bc present but without the mode lists → getattr raises → except False.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._is_sleep_or_standby())

    # ── _chrome_process_count ────────────────────────────────────────────
    def _fake_psutil(self, names, iter_raises=False, per_proc_raises=False):
        ps = types.ModuleType("psutil")

        class _Proc:
            def __init__(self, name, boom=False):
                self._boom = boom
                self.info = {"name": name}

            # When per_proc_raises, accessing .info["name"] is fine but the
            # skill's `p.info.get("name")` is wrapped in try/except; simulate a
            # raising .info via a property.
        procs = []
        for n in names:
            procs.append(_Proc(n))
        if per_proc_raises:
            class _BoomProc:
                @property
                def info(self):
                    raise RuntimeError("proc boom")
            procs.append(_BoomProc())

        def _iter(attrs=None):
            if iter_raises:
                raise RuntimeError("iter boom")
            return procs

        ps.process_iter = _iter
        return ps

    def test_chrome_process_count_counts_chrome(self):
        ps = self._fake_psutil(["chrome.exe", "chrome.exe", "explorer.exe",
                                "chrome"])
        with inject_modules(psutil=ps):
            self.assertEqual(self.mod._chrome_process_count(), 3)

    def test_chrome_process_count_no_psutil(self):
        with block_import("psutil"):
            self.assertEqual(self.mod._chrome_process_count(), 0)

    def test_chrome_process_count_iter_raises(self):
        ps = self._fake_psutil([], iter_raises=True)
        with inject_modules(psutil=ps):
            self.assertEqual(self.mod._chrome_process_count(), 0)

    def test_chrome_process_count_per_proc_error_skipped(self):
        # A process whose .info access raises is skipped, not fatal.
        ps = self._fake_psutil(["chrome.exe"], per_proc_raises=True)
        with inject_modules(psutil=ps):
            self.assertEqual(self.mod._chrome_process_count(), 1)

    # ── _visible_window_count ────────────────────────────────────────────
    def test_visible_window_count_filters_system_surfaces(self):
        titles = ["Program Manager", "jarvis_hud", "Real Work - VS Code",
                  "Inbox - Chrome"]
        with mock.patch.object(self.mod, "_all_window_titles", return_value=titles):
            self.assertEqual(self.mod._visible_window_count(), 2)

    def test_visible_window_count_all_filtered(self):
        titles = ["Program Manager", "Default IME", "jarvis_reticle", "Settings"]
        with mock.patch.object(self.mod, "_all_window_titles", return_value=titles):
            self.assertEqual(self.mod._visible_window_count(), 0)


# ─── pattern_memory access bridges ─────────────────────────────────────────
class BanterPatternMemoryTests(_BanterTestBase):
    def test_load_voice_commands_uses_loader(self):
        pm = types.ModuleType("pattern_memory")
        pm._load_entries = lambda: [{"text": "hi"}]
        with inject_modules(pattern_memory=pm):
            self.assertEqual(self.mod._load_voice_commands(), [{"text": "hi"}])

    def test_load_voice_commands_loader_returns_none(self):
        pm = types.ModuleType("pattern_memory")
        pm._load_entries = lambda: None       # `or []` normalises to []
        with inject_modules(pattern_memory=pm):
            self.assertEqual(self.mod._load_voice_commands(), [])

    def test_load_voice_commands_loader_raises(self):
        pm = types.ModuleType("pattern_memory")
        def _boom():
            raise RuntimeError("read fail")
        pm._load_entries = _boom
        with inject_modules(pattern_memory=pm):
            self.assertEqual(self.mod._load_voice_commands(), [])

    def test_load_voice_commands_no_loader_attr(self):
        pm = types.ModuleType("pattern_memory")   # no _load_entries
        # memory fallback also lacks the attr.
        mem = types.ModuleType("memory")
        with inject_modules(pattern_memory=pm, memory=mem):
            self.assertEqual(self.mod._load_voice_commands(), [])

    def test_load_voice_commands_falls_back_to_memory_module(self):
        # pattern_memory import fails → falls back to ``memory``.
        mem = types.ModuleType("memory")
        mem._load_entries = lambda: [{"text": "fallback"}]
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=lambda name: (
                                   mem if name == "memory"
                                   else (_ for _ in ()).throw(ImportError(name)))):
            self.assertEqual(self.mod._load_voice_commands(), [{"text": "fallback"}])

    def test_load_voice_commands_both_imports_fail(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no module")):
            self.assertEqual(self.mod._load_voice_commands(), [])

    def test_extract_target_safe_delegates(self):
        pm = types.ModuleType("pattern_memory")
        pm._extract_target = lambda text: ("open", "chrome")
        with inject_modules(pattern_memory=pm):
            self.assertEqual(self.mod._extract_target_safe("open chrome"),
                             ("open", "chrome"))

    def test_extract_target_safe_no_fn(self):
        pm = types.ModuleType("pattern_memory")   # no _extract_target
        mem = types.ModuleType("memory")
        with inject_modules(pattern_memory=pm, memory=mem):
            self.assertIsNone(self.mod._extract_target_safe("open chrome"))

    def test_extract_target_safe_fn_raises(self):
        pm = types.ModuleType("pattern_memory")
        def _boom(_text):
            raise RuntimeError("extract fail")
        pm._extract_target = _boom
        with inject_modules(pattern_memory=pm):
            self.assertIsNone(self.mod._extract_target_safe("open chrome"))

    def test_extract_target_safe_import_fails(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no pm")):
            self.assertIsNone(self.mod._extract_target_safe("open chrome"))

    def test_extract_target_safe_falls_back_to_memory(self):
        mem = types.ModuleType("memory")
        mem._extract_target = lambda text: ("play", "jazz")
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=lambda name: (
                                   mem if name == "memory"
                                   else (_ for _ in ()).throw(ImportError(name)))):
            self.assertEqual(self.mod._extract_target_safe("play jazz"),
                             ("play", "jazz"))


# ─── config read + persistent state ────────────────────────────────────────
class BanterConfigStateTests(_BanterTestBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="banter_test_")
        self.addCleanup(self._cleanup)
        self.state_file = os.path.join(self.tmp, "banter_state.json")
        # Redirect persistence so the real project file is never touched.
        self.mod._STATE_FILE = self.state_file
        self.mod._PROJECT_DIR = self.tmp

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    # ── _read_config ─────────────────────────────────────────────────────
    def test_read_config_reads_bc_knobs(self):
        bc = types.ModuleType("bobert_companion")
        bc.BANTER_ENABLED = False
        bc.BANTER_COOLDOWN_MINUTES = 45
        bc.BANTER_PER_TELL_COOLDOWN_MINUTES = 240
        with inject_modules(bobert_companion=bc):
            cfg = self.mod._read_config()
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["cooldown"], 45)
        self.assertEqual(cfg["per_tell_cd"], 240)

    def test_read_config_defaults_when_knobs_absent(self):
        # bc present but without the knobs → getattr defaults apply.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            cfg = self.mod._read_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["cooldown"], 30)
        self.assertEqual(cfg["per_tell_cd"], 180)

    def test_read_config_import_failure_uses_defaults(self):
        # importlib.import_module raising is the bc=None branch.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": True, "cooldown": 30, "per_tell_cd": 180})

    # ── _load_state / _save_state ────────────────────────────────────────
    def test_load_state_missing_file(self):
        self.assertEqual(self.mod._load_state(), {})

    def test_save_then_load_round_trip(self):
        self.mod._save_state({"last_fire_at": 123.0, "last_tell": "tab_clutter"})
        self.assertTrue(os.path.exists(self.state_file))
        loaded = self.mod._load_state()
        self.assertEqual(loaded["last_tell"], "tab_clutter")
        self.assertEqual(loaded["last_fire_at"], 123.0)

    def test_load_state_corrupt_json(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        self.assertEqual(self.mod._load_state(), {})

    def test_load_state_non_dict_payload(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.mod._load_state(), {})

    def test_save_state_swallows_mkstemp_failure(self):
        with mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("disk full")):
            self.mod._save_state({"x": 1})   # must not raise
        self.assertFalse(os.path.exists(self.state_file))

    def test_save_state_replace_failure_unlinks_tmp(self):
        before = set(os.listdir(self.tmp))
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")):
            self.mod._save_state({"y": 2})   # must not raise
        after = set(os.listdir(self.tmp))
        self.assertEqual(before, after)       # tmp cleaned up
        self.assertFalse(os.path.exists(self.state_file))

    def test_save_state_replace_and_unlink_both_fail(self):
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("rename denied")), \
             mock.patch.object(self.mod.os, "unlink",
                               side_effect=OSError("unlink denied")):
            self.mod._save_state({"z": 3})   # must not raise
        self.assertFalse(os.path.exists(self.state_file))


# ─── speech queue ──────────────────────────────────────────────────────────
class BanterSpeechQueueTests(_BanterTestBase):
    def test_enqueue_speech_writes_to_temp_queue(self):
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(qp)   # ensure the "file does not exist yet" branch
        try:
            with mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("That's the 3rd time, sir.")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertIn("3rd time", data[0]["message"])
            self.assertIn("ts", data[0])
        finally:
            if os.path.exists(qp):
                os.unlink(qp)

    def test_enqueue_speech_appends_to_existing(self):
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(qp, "w", encoding="utf-8") as f:
                json.dump([{"ts": 1.0, "message": "earlier"}], f)
            with mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("later line")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["message"], "earlier")
            self.assertEqual(data[1]["message"], "later line")
        finally:
            os.unlink(qp)

    def test_enqueue_speech_corrupt_existing_resets(self):
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(qp, "w", encoding="utf-8") as f:
                f.write("{ corrupt")
            with mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("fresh start")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["message"], "fresh start")
        finally:
            os.unlink(qp)

    def test_enqueue_speech_write_failure_is_swallowed(self):
        # _atomic_write_json raising must be caught, no raise (prints a warning).
        with mock.patch.object(self.mod, "_SPEECH_QUEUE", "X:/nope/queue.json"), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only fs")):
            self.mod._enqueue_speech("dropped line")   # must not raise


# ─── _format_status / banter_status action ─────────────────────────────────
class BanterStatusTests(_BanterTestBase):
    def test_status_no_fires(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = self.actions["banter_status"]("")
        self.assertIn("no zingers yet", out.lower())
        self.assertTrue(out.startswith("Banter engine"))

    def test_status_reports_last_zinger_minutes(self):
        state = {"last_fire_at": time.time() - 600, "last_tell": "tab_clutter"}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = self.actions["banter_status"]("")
        self.assertIn("last zinger", out.lower())
        self.assertIn("tab_clutter", out)
        self.assertIn("minute", out)
        self.assertIn("until next eligible", out)

    def test_status_recent_fire_seconds_phrasing(self):
        # < 60s ago → "N seconds ago".
        state = {"last_fire_at": time.time() - 20, "last_tell": "repeat_open"}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = self.actions["banter_status"]("")
        self.assertIn("seconds ago", out)

    def test_status_hours_phrasing_singular(self):
        # ~1h ago drives the hour branch AND the singular "1 hour" path; a fire
        # that old is past the 30-min cooldown so no "until next eligible".
        state = {"last_fire_at": time.time() - 3600, "last_tell": "music_while_music"}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = self.actions["banter_status"]("")
        self.assertIn("1 hour ago", out)
        self.assertNotIn("until next eligible", out)

    def test_status_disabled_in_config(self):
        bc = types.ModuleType("bobert_companion")
        bc.BANTER_ENABLED = False
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = self.actions["banter_status"]("")
        self.assertIn("engine disabled in config", out)

    def test_status_surfaces_call_and_sleep(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True):
            out = self.actions["banter_status"]("")
        self.assertIn("in a call", out.lower())
        self.assertIn("sleep/standby active", out)

    def test_status_action_swallows_exception(self):
        with mock.patch.object(self.mod, "_format_status",
                               side_effect=RuntimeError("kaboom")):
            out = self.actions["banter_status"]("")
        self.assertIn("banter status failed", out)
        self.assertIn("kaboom", out)


# ─── _scheduler_loop ───────────────────────────────────────────────────────
class BanterSchedulerLoopTests(_BanterTestBase):
    """Drive _scheduler_loop for a bounded number of poll iterations. time.sleep
    is patched to no-op the INITIAL_DELAY + first poll-end sleep(s) then raise
    _StopLoop, so the otherwise-infinite while exits deterministically. random
    and every gate are stubbed; nothing real runs."""

    def setUp(self):
        super().setUp()
        # The inner-exception path logs via logging.exception — swallow it so the
        # deliberate-failure test doesn't spam the runner.
        mock.patch.object(self.mod.logging, "exception",
                          lambda *a, **k: None).start()
        self.addCleanup(mock.patch.stopall)

    @contextlib.contextmanager
    def _loop_env(self, *, enabled=True, sleep=False, in_call=False,
                  state=None, entries=None, sleeps_before_stop=2):
        cfg = {"enabled": enabled, "cooldown": 30, "per_tell_cd": 180}
        state = {} if state is None else state
        entries = [] if entries is None else entries
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=_sleep_after(sleeps_before_stop)), \
             mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=sleep), \
             mock.patch.object(self.mod, "_is_in_call", return_value=in_call), \
             mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_load_voice_commands", return_value=entries):
            yield

    def test_loop_disabled_skips_everything(self):
        enqueue = mock.MagicMock()
        with self._loop_env(enabled=False), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_gate_sleep(self):
        enqueue = mock.MagicMock()
        with self._loop_env(sleep=True), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_gate_in_call(self):
        enqueue = mock.MagicMock()
        with self._loop_env(in_call=True), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_cooldown_blocks_fire(self):
        enqueue = mock.MagicMock()
        recent = {"last_fire_at": time.time() - 60}   # within 30-min cooldown
        with self._loop_env(state=recent), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_no_tell_no_fire(self):
        enqueue = mock.MagicMock()
        with self._loop_env(), \
             mock.patch.object(self.mod, "_detect_repeat_question", return_value=None), \
             mock.patch.object(self.mod, "_detect_music_while_music", return_value=None), \
             mock.patch.object(self.mod, "_detect_repeat_open", return_value=None), \
             mock.patch.object(self.mod, "_detect_tab_clutter", return_value=None), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_fires_and_persists_when_probability_passes(self):
        enqueue = mock.MagicMock()
        saved = {}
        tell = {"tell": "tab_clutter", "key": "tab_clutter:chrome", "n": 47}
        with self._loop_env(), \
             mock.patch.object(self.mod, "_detect_repeat_question", return_value=tell), \
             mock.patch.object(self.mod.random, "random", return_value=0.0), \
             mock.patch.object(self.mod, "_pick_zinger", return_value="47 tabs, sir."), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue), \
             mock.patch.object(self.mod, "_save_state",
                               side_effect=lambda s: saved.update(s)):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args[0][0], "47 tabs, sir.")
        self.assertEqual(saved["last_tell"], "tab_clutter")
        self.assertEqual(saved["last_line"], "47 tabs, sir.")
        self.assertIn("last_fire_at", saved)
        self.assertIn("tab_clutter:chrome", saved["last_per_tell_at"])

    def test_loop_probability_suppresses_fire(self):
        enqueue = mock.MagicMock()
        tell = {"tell": "tab_clutter", "key": "tab_clutter:chrome", "n": 47}
        with self._loop_env(), \
             mock.patch.object(self.mod, "_detect_repeat_question", return_value=tell), \
             mock.patch.object(self.mod.random, "random", return_value=0.99), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_per_tell_cooldown_suppresses(self):
        # The detected tell fired within its per-tell cooldown → skipped, no fire.
        enqueue = mock.MagicMock()
        tell = {"tell": "tab_clutter", "key": "tab_clutter:chrome", "n": 47}
        state = {"last_per_tell_at": {"tab_clutter:chrome": time.time() - 60}}
        with self._loop_env(state=state), \
             mock.patch.object(self.mod, "_detect_repeat_question", return_value=tell), \
             mock.patch.object(self.mod, "_detect_music_while_music", return_value=None), \
             mock.patch.object(self.mod, "_detect_repeat_open", return_value=None), \
             mock.patch.object(self.mod, "_detect_tab_clutter", return_value=None), \
             mock.patch.object(self.mod.random, "random", return_value=0.0), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_detector_exception_is_caught(self):
        # A detector raising is caught inside the detector for-loop (prints,
        # sets tell=None) and the tick proceeds to no-fire without crashing.
        enqueue = mock.MagicMock()
        with self._loop_env(), \
             mock.patch.object(self.mod, "_detect_repeat_question",
                               side_effect=RuntimeError("detector boom")), \
             mock.patch.object(self.mod, "_detect_music_while_music", return_value=None), \
             mock.patch.object(self.mod, "_detect_repeat_open", return_value=None), \
             mock.patch.object(self.mod, "_detect_tab_clutter", return_value=None), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()

    def test_loop_empty_zinger_skips_fire(self):
        # A matched tell whose _pick_zinger yields "" → no enqueue/persist.
        enqueue = mock.MagicMock()
        save = mock.MagicMock()
        tell = {"tell": "tab_clutter", "key": "tab_clutter:chrome", "n": 47}
        with self._loop_env(), \
             mock.patch.object(self.mod, "_detect_repeat_question", return_value=tell), \
             mock.patch.object(self.mod.random, "random", return_value=0.0), \
             mock.patch.object(self.mod, "_pick_zinger", return_value=""), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue), \
             mock.patch.object(self.mod, "_save_state", save):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        enqueue.assert_not_called()
        save.assert_not_called()

    def test_loop_prunes_ancient_per_tell_entries(self):
        # An 8-day-old per-tell timestamp is pruned when state is persisted.
        enqueue = mock.MagicMock()
        saved = {}
        tell = {"tell": "tab_clutter", "key": "tab_clutter:chrome", "n": 47}
        old = time.time() - 8 * 86400
        state = {"last_per_tell_at": {"stale:key": old}}
        with self._loop_env(state=state), \
             mock.patch.object(self.mod, "_detect_repeat_question", return_value=tell), \
             mock.patch.object(self.mod.random, "random", return_value=0.0), \
             mock.patch.object(self.mod, "_pick_zinger", return_value="line"), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue), \
             mock.patch.object(self.mod, "_save_state",
                               side_effect=lambda s: saved.update(s)):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        self.assertNotIn("stale:key", saved["last_per_tell_at"])
        self.assertIn("tab_clutter:chrome", saved["last_per_tell_at"])

    def test_loop_inner_exception_is_logged_and_continues(self):
        # An exception mid-poll (here from _read_config) is caught by the loop's
        # outer try/except, which logs, sleeps, then loops again. Allow the
        # INITIAL_DELAY sleep + one in-except sleep, then the second failing
        # pass's sleep raises _StopLoop.
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=_sleep_after(2)), \
             mock.patch.object(self.mod, "_read_config",
                               side_effect=RuntimeError("cfg blew up")):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()


# ─── register() ────────────────────────────────────────────────────────────
class BanterRegisterTests(_BanterTestBase):
    def test_register_adds_action_and_starts_thread_when_enabled(self):
        actions = {}
        bc = _benign_bc()
        bc.BANTER_ENABLED = True
        made = mock.MagicMock()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "Thread", return_value=made) as Thread:
            self.mod.register(actions)
        self.assertIn("banter_status", actions)
        Thread.assert_called_once()
        made.start.assert_called_once()

    def test_register_disabled_no_thread(self):
        actions = {}
        bc = _benign_bc()
        bc.BANTER_ENABLED = False
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        self.assertIn("banter_status", actions)   # action still registered
        Thread.assert_not_called()

    def test_registered_action_is_callable(self):
        actions = {}
        bc = _benign_bc()
        bc.BANTER_ENABLED = False
        with inject_modules(bobert_companion=bc):
            self.mod.register(actions)
        with mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = actions["banter_status"]("ignored-arg")
        self.assertTrue(out.startswith("Banter engine"))


if __name__ == "__main__":
    unittest.main()
