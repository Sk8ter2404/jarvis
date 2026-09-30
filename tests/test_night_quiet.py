"""NIGHT_QUIET_ENABLED (core/night_quiet.py): ONE master switch over every
clock-driven night quieting (owner request 2026-09-29: "disable night owl", then
"disable them all for now"). ON, the shipped default, keeps the old behaviour;
OFF, JARVIS sounds at night exactly as he does in the daytime. Each gated
behaviour has a test with the knob on and a test with it off; what the owner
asks for himself (saying he is tired) keeps working either way.

Rules these tests keep:
  * the knob is patched explicitly in every behaviour test, so the gitignored
    data/user_settings.json can never decide a result;
  * the shipped default is read from the core/config.py SOURCE, not from the
    runtime value that user_settings.json may have overridden;
  * the clock is always pinned (datetime arguments or a patched clock
    helper), never the real wall clock (CI runs in UTC).

Skill-side gates live in tests/skills/test_night_quiet_skills.py and
tests/skills/test_night_owl_auto.py; monolith ones (wake greeting, late-night
remark) in tests/monolith/test_monolith_night_quiet.py.

    python -m unittest tests.test_night_quiet
"""
from __future__ import annotations

import ast
import datetime
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from core import config as cfg
from core import emotion_tracker as et
from core import tone_detector as td
from core import tts
from core import voice_emotion as ve
from core.emotion_tracker import ProsodyHints

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LATE = datetime.datetime(2026, 9, 29, 23, 30)         # 23:30
SMALL_HOURS = datetime.datetime(2026, 9, 30, 2, 0)    # 02:00
EARLY = datetime.datetime(2026, 9, 30, 6, 30)         # 06:30, past the clock bands
DAY = datetime.datetime(2026, 9, 29, 14, 0)           # 14:00


def _quiet(on):
    """Pin NIGHT_QUIET_ENABLED for the duration of a `with` block."""
    return mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", on, create=True)


def _clock_says_late_night(value=True):
    """Pin core.tone_detector's clock test (the 22:00-04:59 band)."""
    return mock.patch.object(td, "_is_late_night_hour", return_value=value)


def _config_source():
    with open(os.path.join(_ROOT, "core", "config.py"), encoding="utf-8") as fh:
        return fh.read()


def _config_literals() -> dict:
    lits = {}
    for node in ast.parse(_config_source()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    lits[tgt.id] = node.value.value
    return lits


class _NoStateFile(unittest.TestCase):
    """A temp dir whose anticipation_state.json exists only when written."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jv_night_quiet_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = os.path.join(self.tmp, "anticipation_state.json")

    def _write_late_hour_state(self, age_s=60.0):
        with open(self.state, "w", encoding="utf-8") as fh:
            json.dump({"last_trigger": "late_hour",
                       "last_proactive_at": time.time() - age_s}, fh)


# ─── the setting itself ──────────────────────────────────────────────────
class ShippedSettingTests(unittest.TestCase):

    def test_config_source_ships_it_on_with_a_comment(self):
        self.assertIs(_config_literals().get("NIGHT_QUIET_ENABLED"), True)
        src = _config_source()
        head = src[:src.index("\nNIGHT_QUIET_ENABLED = True")]
        block = head[head.rindex("\n\n"):].strip().splitlines()
        self.assertTrue(block and all(ln.startswith("#") for ln in block))
        text = " ".join(ln.lstrip("# ") for ln in block)
        self.assertIn("NIGHT_QUIET_ENABLED", text)
        self.assertIn("on the next start", text)

    def test_settings_window_row_and_example_template(self):
        spec = importlib.util.spec_from_file_location(
            "sw_night_quiet", os.path.join(_ROOT, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        row = sw.SCHEMA["NIGHT_QUIET_ENABLED"]
        self.assertEqual((row["tab"], row["type"], row["default"]),
                         ("voice", "bool", True))
        with open(os.path.join(_ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            example = json.load(fh)
        self.assertIs(example["NIGHT_QUIET_ENABLED"], True)

    def test_reader_follows_the_knob_at_call_time(self):
        nq = importlib.import_module("core.night_quiet")
        with _quiet(True):
            self.assertTrue(nq.night_quiet_enabled())
        with _quiet(False):
            self.assertFalse(nq.night_quiet_enabled())

    def test_unreadable_knob_keeps_the_old_behaviour(self):
        nq = importlib.import_module("core.night_quiet")

        class _Unreadable:
            def __bool__(self):
                raise ValueError("not a bool")
        with _quiet(_Unreadable()):
            self.assertTrue(nq.night_quiet_enabled())

    def test_night_owl_auto_needs_both_knobs(self):
        nq = importlib.import_module("core.night_quiet")
        for quiet, auto, want in ((True, True, True), (True, False, False),
                                  (False, True, False), (False, False, False)):
            with _quiet(quiet), \
                 mock.patch.object(cfg, "NIGHT_OWL_AUTO", auto, create=True):
                self.assertIs(nq.night_owl_auto_enabled(), want, (quiet, auto))


def _comment_above(assignment: str) -> str:
    """The '#' comment block directly above `assignment` in core/config.py,
    joined into one line."""
    src = _config_source()
    head = src[:src.index("\n" + assignment)]
    block = head[head.rindex("\n\n"):].strip().splitlines()
    assert block and all(ln.startswith("#") for ln in block), block
    return " ".join(ln.lstrip("# ") for ln in block)


# ─── the night-owl help / comment describe what the mode really does ─────
class NightOwlHelpIsAccurateTests(unittest.TestCase):
    """Review 2026-09-30: the help and config comment said night-owl mode is
    only 'a voice about 15% quieter', held announcements and a dimmed overlay.
    It also slows the voice (NIGHT_OWL_RATE_DELTA_PP; Kokoro honours the rate
    as speed), adds a one-short-sentence prompt addendum and turns off the
    'thinking' filler clip. Each effect is pinned in the code AND the text."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "sw_night_owl_help",
            os.path.join(_ROOT, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        cls.help = sw.SCHEMA["NIGHT_OWL_AUTO"]["help"]
        cls.comment = _comment_above("NIGHT_OWL_AUTO = True")

    def test_the_mode_really_slows_and_quietens_the_voice(self):
        from tests._skill_harness import load_skill_isolated
        mod, _actions = load_skill_isolated("night_owl_mode")
        fake_bc = types.SimpleNamespace(
            _resolve_tts_preset=lambda text, tone: (
                "neutral", {"rate": "+0%", "gain": 1.0}))
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            mod._install_tts_modifier()
        name, preset = fake_bc._resolve_tts_preset("Hello there.", None)
        self.assertEqual(name, "neutral_nightowl")
        self.assertEqual(preset["rate"], "-5%")
        self.assertAlmostEqual(preset["gain"], 0.85)
        self.assertIn("ONE short sentence", mod.NIGHT_OWL_PROMPT_ADDENDUM)

    def test_the_mode_really_turns_off_the_filler_clip(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        start = src.index("def _filler_suppressed(")
        body = src[start:src.index("\ndef ", start + 1)]
        self.assertIn("is_night_owl_active", body)
        self.assertIn('"night-owl"', body)

    def test_help_and_comment_name_every_effect(self):
        for where, text in (("help", self.help), ("comment", self.comment)):
            low = text.lower()
            for word in ("quieter", "slower", "sentence", "filler",
                         "held", "overlay"):
                self.assertIn(word, low, f"{where} omits {word!r}: {text}")


# ─── the master switch's own help / comment match what it now gates ──────
class MasterSwitchTextTests(unittest.TestCase):

    def test_help_and_comment_name_the_late_fixes(self):
        spec = importlib.util.spec_from_file_location(
            "sw_night_quiet_help",
            os.path.join(_ROOT, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        help_ = sw.SCHEMA["NIGHT_QUIET_ENABLED"]["help"].lower()
        comment = _comment_above("NIGHT_QUIET_ENABLED = True").lower()
        for text in (help_, comment):
            self.assertIn("proactive", text)
            self.assertIn("same as in the daytime", text)
        self.assertIn("still up, sir?", comment)
        self.assertIn("late hour", comment)


# ─── core/prompts.py: the two clock-only rules in the system prompt ──────
class PromptLateHourRuleTests(unittest.TestCase):

    def test_on_the_prompt_is_unchanged(self):
        from core import prompts
        with _quiet(True):
            p = prompts.base_system_prompt()
        self.assertIs(p, prompts.BASE_SYSTEM_PROMPT)
        self.assertIn("LATE HOUR — local time after 22:00", p)
        self.assertIn("no venting keywords, daylight hours.", p)

    def test_off_the_hour_alone_is_no_stress_signal(self):
        from core import prompts
        with _quiet(False):
            p = prompts.base_system_prompt()
        self.assertNotIn("LATE HOUR", p)
        self.assertNotIn("daylight hours", p)
        # His own stress signals still put JARVIS in the calm register.
        self.assertIn("VENTING KEYWORDS", p)
        self.assertIn("RAPID OR FRAGMENTED SPEECH", p)
        self.assertIn("no venting keywords. Do not announce the shift", p)
        self.assertEqual(
            len(prompts.BASE_SYSTEM_PROMPT) - len(p),
            len(prompts._LATE_HOUR_STRESS_SIGNAL)
            + len(prompts._DAYLIGHT_RETURN_CLAUSE))

    def test_each_clock_rule_is_in_the_prompt_exactly_once(self):
        from core import prompts
        base = prompts.BASE_SYSTEM_PROMPT
        self.assertEqual(base.count(prompts._LATE_HOUR_STRESS_SIGNAL), 1)
        self.assertEqual(base.count(prompts._DAYLIGHT_RETURN_CLAUSE), 1)


# ─── the core modules still run as scripts (lazy knob import) ────────────
class RunsAsAScriptTests(unittest.TestCase):
    """`python core/tts.py` runs its __main__ self-test with no package on
    sys.path. A module-level `from core.night_quiet import ...` broke it
    (ModuleNotFoundError: No module named 'core'); the reader is imported
    lazily in core/tts.py and core/tone_detector.py."""

    def _run(self, *rel):
        tmp = tempfile.mkdtemp(prefix="jv_script_")
        self.addCleanup(shutil.rmtree, tmp, True)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, os.path.join(_ROOT, *rel)], cwd=tmp, env=env,
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=120)

    def test_tts_self_test_runs_and_passes(self):
        r = self._run("core", "tts.py")
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("OK", r.stdout)
        self.assertNotIn("BAD", r.stdout)

    def test_tone_detector_runs_as_a_script(self):
        r = self._run("core", "tone_detector.py")
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])

    def test_as_a_script_the_reader_keeps_the_old_behaviour(self):
        # No package to import: both modules fall back to ON, never crash.
        with mock.patch.dict(sys.modules, {"core.night_quiet": None}):
            self.assertTrue(tts.night_quiet_enabled())
            self.assertTrue(td.night_quiet_enabled())


# ─── 2. core/tts.py: the 'hushed_late' preset (gain 0.55) ────────────────
class HushedLatePresetTests(_NoStateFile):

    def test_on_hushes_from_23_00(self):
        with _quiet(True):
            self.assertTrue(tts.detect_late_hour(state_path=self.state, now=LATE))
            name, preset = tts.resolve_tts_preset(
                "Hello there.", None, now=LATE, state_path=self.state)
        self.assertEqual(name, "hushed_late")
        self.assertEqual(preset["gain"], 0.55)

    def test_off_sounds_like_daytime_at_23_00(self):
        with _quiet(False):
            self.assertFalse(tts.detect_late_hour(state_path=self.state, now=LATE))
            self.assertIsNone(tts.detect_context_preset(
                "calm text", peak_rms=0.0, now=LATE, state_path=self.state))
            name, preset = tts.resolve_tts_preset(
                "Hello there.", None, now=LATE, state_path=self.state)
        self.assertEqual(name, "neutral")
        self.assertEqual(preset["gain"], 1.0)

    def test_on_late_hour_state_file_extends_it_past_the_clock(self):
        self._write_late_hour_state()
        with _quiet(True):
            self.assertTrue(tts.detect_late_hour(state_path=self.state, now=EARLY))

    def test_off_ignores_a_fresh_late_hour_state_file(self):
        self._write_late_hour_state()
        with _quiet(False):
            self.assertFalse(tts.detect_late_hour(state_path=self.state, now=EARLY))
            name, preset = tts.resolve_tts_preset(
                "Hello there.", None, now=EARLY, state_path=self.state)
        self.assertEqual((name, preset["gain"]), ("neutral", 1.0))

    def test_off_still_lets_an_emergency_word_through(self):
        # Not a night rule: brisk_alert for 'help' is clock-independent.
        with _quiet(False):
            self.assertEqual(tts.detect_context_preset(
                "help", peak_rms=0.0, now=LATE, state_path=self.state),
                "brisk_alert")


# ─── 3a. core/tone_detector.py: the time-of-day 'late_night' tone ────────
class LateNightToneTests(unittest.TestCase):

    def test_on_neutral_words_at_night_get_the_late_night_tone(self):
        with _quiet(True), _clock_says_late_night():
            tone = td.detect_tone("open the notes")
        self.assertEqual(tone, "late_night")
        self.assertIn("USER_TONE: late-night", td._tone_system_addendum(tone))

    def test_off_neutral_words_at_night_get_no_tone(self):
        with _quiet(False), _clock_says_late_night():
            tone = td.detect_tone("open the notes")
        self.assertIsNone(tone)
        self.assertEqual(td._tone_system_addendum(tone), "")

    def test_off_his_own_words_still_count(self):
        with _quiet(False), _clock_says_late_night():
            self.assertEqual(td.detect_tone("i'm exhausted"), "tired")


# ─── 3b. core/voice_emotion.py: late_night mood, addendum, gain 0.65 ─────
class LateNightMoodTests(unittest.TestCase):

    def _route_and_voice(self, text, when):
        route = ve.route_voice_emotion(text, now=when.timestamp())
        tone = route["mood"] if route["mood"] != "casual" else None
        # A daytime clock for the preset itself isolates the mood's preset
        # from the separate hushed_late rule tested above.
        name, preset = tts.resolve_tts_preset(
            "Hello there.", tone, now=DAY,
            state_path=os.path.join(_ROOT, "__no_such_state__.json"))
        return route, name, preset

    def test_on_small_hours_route_to_the_late_night_mood(self):
        with _quiet(True), _clock_says_late_night():
            route, name, preset = self._route_and_voice("open the notes",
                                                        SMALL_HOURS)
        self.assertEqual(route["mood"], "late_night")
        self.assertIn("USER_TONE: late_night", route["addendum"])
        self.assertEqual((name, preset["gain"]), ("late_night", 0.65))

    def test_off_small_hours_sound_like_daytime(self):
        with _quiet(False), _clock_says_late_night():
            route, name, preset = self._route_and_voice("open the notes",
                                                        SMALL_HOURS)
        self.assertEqual(route, {"mood": "casual", "addendum": ""})
        self.assertEqual((name, preset["gain"]), ("neutral", 1.0))

    def test_off_saying_he_is_tired_still_softens_the_voice(self):
        with _quiet(False), _clock_says_late_night(False):
            route, name, preset = self._route_and_voice("i'm exhausted", DAY)
        self.assertEqual(route["mood"], "late_night")
        self.assertEqual(name, "late_night")


# ─── 3c. core/emotion_tracker.py: the time-only 'tired' label ────────────
class TimeOnlyTiredTests(unittest.TestCase):

    def test_on_a_short_line_at_02_00_reads_as_tired(self):
        with _quiet(True):
            r = et.classify_emotion("what's the time", ProsodyHints(hour=2))
        self.assertEqual((r.label, r.tts_preset), ("tired", "concerned"))
        self.assertEqual(tts._TTS_EMOTION_PRESETS[r.tts_preset]["gain"], 0.92)

    def test_off_a_short_line_at_02_00_reads_as_nothing(self):
        with _quiet(False):
            r = et.classify_emotion("what's the time", ProsodyHints(hour=2))
        self.assertIsNone(r.label)
        self.assertIsNone(r.tts_preset)
        self.assertEqual(r.addendum, "")

    def test_off_his_own_words_still_read_as_tired(self):
        with _quiet(False):
            r = et.classify_emotion("I'm exhausted, going to bed",
                                    ProsodyHints(hour=2))
        self.assertEqual(r.label, "tired")


# ─── core/prompts.py: the night-owl entry stays true with the knobs ──────
class NightOwlPromptTextTests(unittest.TestCase):

    def test_prompt_no_longer_promises_an_unconditional_23_00_engage(self):
        from core import prompts
        section = prompts.PC_CONTROL_PROMPT
        start = section.index("NIGHT-OWL MODE")
        section = section[start:section.index("SCREEN VISION", start)]
        self.assertNotIn("Auto-engages at 23:00", section)
        self.assertIn("a setting the user may have", section)


if __name__ == "__main__":
    unittest.main()
