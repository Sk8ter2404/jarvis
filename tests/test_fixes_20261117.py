"""The 2026-11-17 review fixes (JARVIS-Research/review_20261117_logs_hands_brain.json,
logs finding 1 and 3; hands finding 1), light tier.

  * game mode ON needs a game / mode / power word, enforced in shadow too
    (Dell 11-17 12:46 "Jarvis is running on Quinn36B" -> game_mode_on);
  * launch_app starts Fortnite through the Epic launcher URI (16:03:23
    "could not launch Fortnite ... [WinError 2]"), and the prompt teaches
    "time to play Fortnite" -> game_mode_on + launch_app;
  * a YouTube search whose query is only "YouTube" opens the home page
    (16:12:48 "open YouTube." played a random Short);
  * "did you log anything while I was at school" is the away recap
    (16:02:13), core/owner_presence;
  * built-in STT name fixes: Yetsi -> Etsy (private names stay in user settings);
  * the face loop skips its per-pass Kinect feed step while the hand-tracker
    pump thread runs (core.handtrack_depth.pump_running, getattr-guarded).

CI SAFETY: monolith helpers are lifted out of bobert_companion.py's SOURCE by
AST and exec'd with stub globals; os.startfile is always mocked (a test must
never launch a real program).
"""
from __future__ import annotations

import ast
import os
import re
import sys
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import action_fit  # noqa: E402
from core import actions as A  # noqa: E402
from core import config  # noqa: E402
from core import owner_presence as op  # noqa: E402
from core import prompt_router, prompts  # noqa: E402
from core import stt_vocab  # noqa: E402
from core.failure_markers import TERMINAL_FAILURE_PREFIX  # noqa: E402

_MONOLITH = os.path.join(_PROJECT, "bobert_companion.py")
_CACHE: dict = {}


def _source() -> str:
    if "src" not in _CACHE:
        with open(_MONOLITH, encoding="utf-8") as fh:
            _CACHE["src"] = fh.read()
    return _CACHE["src"]


def _tree():
    if "tree" not in _CACHE:
        _CACHE["tree"] = ast.parse(_source())
    return _CACHE["tree"]


def _lift(names, ns):
    """Exec the top-level defs AND simple assignments named in ``names``."""
    wanted = []
    for n in _tree().body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            wanted.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in tgts):
                wanted.append(n)
    assert len(wanted) == len(names), [getattr(n, "name", "assign") for n in wanted]
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _MONOLITH, "exec"), ns)
    return ns


def _fn_source(name: str) -> str:
    fn = next(n for n in _tree().body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(_source(), fn)


# ── 1. game mode by mishearing ──────────────────────────────────────────────
class GameModeGuardTests(unittest.TestCase):
    def setUp(self):
        self.flags = {"ACTION_FIT_GAME_MODE_ENFORCE": True}
        self.ns = _lift(
            ["_GAME_MODE_ON_GUARDED", "_GAME_MODE_EXTRA_WORD_RE",
             "_GAME_MODE_SKIPPED_LINE", "_game_mode_fit_refusal"],
            {"re": re, "_action_fit": action_fit,
             "_TERMINAL_FAILURE_PREFIX": TERMINAL_FAILURE_PREFIX,
             "_action_fit_flag": lambda k: self.flags.get(k, True)})
        self.fn = self.ns["_game_mode_fit_refusal"]

    def _check(self, text, name="game_mode_on", reply=("game_mode_on",)):
        rule, reason = action_fit.check_rule(text, name, "")
        return self.fn(rule, name, text, reason, reply)

    def test_the_live_dell_line_is_skipped_and_he_says_why(self):
        out = self._check("Jarvis is running on Quinn36B")
        self.assertTrue(out.startswith(TERMINAL_FAILURE_PREFIX), out)
        line = out[len(TERMINAL_FAILURE_PREFIX):]
        self.assertIn("skipped game mode", line)
        self.assertIn("said nothing about a game", line)

    def test_the_alias_low_power_mode_is_guarded_too(self):
        out = self._check("Jarvis is running on Quinn36B", "low_power_mode",
                          ("low_power_mode",))
        self.assertTrue(out and out.startswith(TERMINAL_FAILURE_PREFIX))

    def test_game_words_fit(self):
        for text in ("time to play Fortnite", "game mode please",
                     "get out of the way, I'm gaming", "divert power",
                     "free up some memory", "Let's play"):
            rule, reason = action_fit.check_rule(text, "game_mode_on", "")
            self.assertIsNone(reason, text)
            self.assertIsNone(self.fn(rule, "game_mode_on", text, reason), text)

    def test_performance_fits_here(self):
        self.assertEqual(self._check("boost performance for a bit"), "fits")

    def test_beside_other_actions_it_is_a_plain_refusal(self):
        out = self._check("Jarvis is running on Quinn36B",
                          reply=("game_mode_on", "system_pulse"))
        self.assertTrue(out.startswith("refused (action-fit): "), out)
        self.assertFalse(out.startswith(TERMINAL_FAILURE_PREFIX))

    def test_game_mode_off_is_not_this_guard(self):
        self.assertIsNone(self._check("turn it off", "game_mode_off",
                                      ("game_mode_off",)))

    def test_flag_off_leaves_it_to_the_generic_rule(self):
        self.flags["ACTION_FIT_GAME_MODE_ENFORCE"] = False
        self.assertIsNone(self._check("Jarvis is running on Quinn36B"))

    def test_other_rules_are_not_this_guard(self):
        self.assertIsNone(self.fn(action_fit.RULE_FACE_LOOK, "game_mode_on",
                                  "x", "reason"))

    def test_config_default_on(self):
        self.assertIs(config.ACTION_FIT_GAME_MODE_ENFORCE, True)

    def test_refusal_calls_the_guard_before_the_shadow_return(self):
        body = _fn_source("_action_fit_refusal")
        guard = body.index("_game_mode_fit_refusal(")
        self.assertLess(guard, body.index("if mode != \"on\" and not enforced"))


# ── 3. Fortnite ─────────────────────────────────────────────────────────────
class FortniteLaunchTests(unittest.TestCase):
    URI = ("com.epicgames.launcher://apps/fn%3A4fe75bbc5a674f4f9b356b5c90567da5"
           "%3AFortnite?action=launch&silent=true")

    def test_the_uri_is_the_epic_one(self):
        self.assertEqual(A.FORTNITE_EPIC_URI, self.URI)

    def test_names(self):
        for n in ("Fortnite", "fortnite.", "the Fortnite game", "Fort Nite",
                  "fortnight", "  FORTNITE  "):
            self.assertTrue(A._is_fortnite_name(n), n)
        for n in ("Discord", "fortnite tracker", "", None):
            self.assertFalse(A._is_fortnite_name(n), n)

    def test_launch_app_starts_it_through_epic(self):
        bc = mock.Mock()
        with mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch.object(A, "_unresolved_open_line", return_value=None), \
                mock.patch.object(A, "_site_shortcut_url", return_value=None), \
                mock.patch.object(A.sys, "platform", "win32"), \
                mock.patch.object(A.subprocess, "Popen") as popen, \
                mock.patch.object(A.os, "startfile", create=True) as startfile:
            out = A._act_launch_app("Fortnite")
        startfile.assert_called_once_with(self.URI)
        popen.assert_not_called()
        bc._resolve_known_app.assert_not_called()
        self.assertIn("launched Fortnite", out)

    def test_a_startfile_error_is_a_launch_failure(self):
        with mock.patch.object(A.sys, "platform", "win32"), \
                mock.patch.object(A.os, "startfile", create=True,
                                  side_effect=OSError("no handler")):
            out = A._launch_fortnite()
        self.assertTrue(out.startswith("could not launch Fortnite"), out)
        self.assertIn("no handler", out)

    def test_off_windows_never_calls_startfile(self):
        with mock.patch.object(A.sys, "platform", "linux"), \
                mock.patch.object(A.os, "startfile", create=True) as startfile:
            out = A._launch_fortnite()
        startfile.assert_not_called()
        self.assertTrue(out.startswith("could not launch Fortnite"))

    def test_the_prompt_teaches_game_mode_then_the_launch(self):
        core, _secs = prompt_router.split_pc_control(prompts.PC_CONTROL_PROMPT)
        flat = " ".join(core.split())
        self.assertIn("Time to play Fortnite", flat)
        self.assertIn("[ACTION: game_mode_on] [ACTION: launch_app, Fortnite]",
                      flat)
        # In the always-on core: every turn has it, whatever the router keeps.
        slim = " ".join(prompt_router.slim_pc_control(
            "time to play Fortnite", prompts.PC_CONTROL_PROMPT).split())
        self.assertIn("[ACTION: game_mode_on] [ACTION: launch_app, Fortnite]",
                      slim)


# ── 4. "open YouTube" ───────────────────────────────────────────────────────
class YouTubeHomeTests(unittest.TestCase):
    def test_bare_queries(self):
        for q in ("YouTube", "youtube.", "You Tube", "the YouTube homepage",
                  "youtube.com", "YouTube home page", "open YouTube", "'YouTube'"):
            self.assertTrue(A.is_bare_youtube_query(q), q)
        for q in ("YouTube shorts", "funny youtube videos", "lofi", "", None,
                  "youtube rewind 2019"):
            self.assertFalse(A.is_bare_youtube_query(q), q)

    def _alias(self, actions):
        ns = _lift(["_YOUTUBE_QUERY_ACTIONS", "_youtube_home_alias"],
                   {"ACTIONS": actions, "print": lambda *a, **k: None})
        return ns["_youtube_home_alias"]

    def test_the_live_turn_opens_the_home_page(self):
        fn = self._alias({"open_url": object(), "youtube_search_direct": object()})
        self.assertEqual(fn("youtube_search_direct", "YouTube"),
                         ("open_url", "https://www.youtube.com"))
        self.assertEqual(fn("youtube", "you tube"),
                         ("open_url", "https://www.youtube.com"))

    def test_a_real_query_is_left_alone(self):
        fn = self._alias({"open_url": object()})
        self.assertEqual(fn("youtube_search_direct", "lofi beats"),
                         ("youtube_search_direct", "lofi beats"))
        self.assertEqual(fn("open_url", "YouTube"), ("open_url", "YouTube"))

    def test_no_open_url_no_rewrite(self):
        fn = self._alias({})
        self.assertEqual(fn("youtube_search_direct", "YouTube"),
                         ("youtube_search_direct", "YouTube"))

    def test_the_runner_applies_it_before_the_play_alias(self):
        src = _source()
        at = src.index("name, arg = _youtube_home_alias(name, arg)")
        self.assertLess(at, src.index("name = _youtube_play_alias(name, arg)", at))


# ── 5. the away question ────────────────────────────────────────────────────
class AwayRecapAskTests(unittest.TestCase):
    def test_live_and_paraphrased_asks(self):
        for t in ("did you log anything while I was at school today?",
                  "Jarvis, did anything happen while I was away?",
                  "what happened while I was gone",
                  "anything come up while I was out?",
                  "Did anything happen while I was at work, Jarvis?",
                  "what happened while I’ve been away"):
            self.assertTrue(op.is_away_recap_ask(t), t)

    def test_not_an_away_ask(self):
        for t in ("any messages I missed on Teams while I was gone?",
                  "did my print finish while I was away",
                  "what did I miss", "did you log anything",
                  "what happened in the game while I was away",
                  "I was away at school", "", None):
            self.assertFalse(op.is_away_recap_ask(t), t)

    def test_answer_line(self):
        self.assertEqual(op.away_answer_line(spoke=True), "")
        recap = "While you were away, sir: the print finished."
        self.assertEqual(op.away_answer_line(
            spoke=False, last_recap=recap, last_recap_age_s=60), recap)
        self.assertEqual(op.away_answer_line(
            spoke=False, last_recap=recap, last_recap_age_s=4 * 3600),
            op.AWAY_NOTHING_WAITING_LINE)
        self.assertEqual(op.away_answer_line(spoke=False, still_queued=True),
                         op.AWAY_STILL_HELD_LINE)
        self.assertEqual(op.away_answer_line(spoke=False),
                         op.AWAY_NOTHING_WAITING_LINE)
        self.assertEqual(op.away_answer_line(spoke=False, last_recap_age_s="x"),
                         op.AWAY_NOTHING_WAITING_LINE)

    def test_the_shortcut_runs_before_the_brain_shortcuts(self):
        body = _fn_source("_run_voice_shortcuts")
        at = body.index("_away_recap_shortcut(text)")
        self.assertLess(body.index("_repeat_last_line_shortcut(text)"), at)
        self.assertLess(at, body.index("_run_alarm_set_shortcut(text)"))

    def test_the_drain_notes_each_spoken_line(self):
        body = _fn_source("_speak_pending")
        at = body.index("spoke_any = True")
        self.assertLess(body.index("_note_pending_spoken(item, msg)", at) - at, 300)

    def test_config_default_on(self):
        self.assertIs(config.AWAY_RECAP_ROUTE, True)


# ── 6. STT name fixes ───────────────────────────────────────────────────────
class SttReplacementTests(unittest.TestCase):
    def fix(self, text):
        return stt_vocab.apply_replacements(text, config.STT_REPLACEMENTS)

    def test_etsy(self):
        self.assertEqual(self.fix("how's the Yetsi pace doing?"),
                         "how's the Etsy page doing?")
        self.assertEqual(self.fix("open yetsy"), "open Etsy")
        self.assertEqual(self.fix("any etsey orders"), "any Etsy orders")

    def test_whole_words_only(self):
        self.assertEqual(self.fix("my mother bought it"), "my mother bought it")
        self.assertEqual(self.fix("motherboard temps"), "motherboard temps")

    def test_a_user_map_merges_with_the_built_in_one(self):
        # core.config._apply_user_settings merges a dict override
        # ({**cur, **val}), so his own entries never drop these.
        merged = {**config.STT_REPLACEMENTS, "a cello": "Zorblat"}
        self.assertEqual(merged["yetsi"], "Etsy")
        self.assertEqual(merged["a cello"], "Zorblat")


# ── 7. the hand-tracker pump hook ───────────────────────────────────────────
class HandtrackPumpHookTests(unittest.TestCase):
    def _fn(self, mod):
        return _lift(["_handtrack_pump_running"],
                     {"_handtrack_depth": mod})["_handtrack_pump_running"]

    def test_absent_function_is_not_running(self):
        self.assertFalse(self._fn(types.SimpleNamespace())())

    def test_running(self):
        self.assertTrue(self._fn(types.SimpleNamespace(pump_running=lambda: True))())

    def test_only_true_counts(self):
        self.assertFalse(self._fn(types.SimpleNamespace(pump_running=lambda: 1))())
        self.assertFalse(self._fn(types.SimpleNamespace(pump_running=lambda: None))())

    def test_a_raising_pump_check_is_not_running(self):
        def boom():
            raise RuntimeError("x")
        self.assertFalse(self._fn(types.SimpleNamespace(pump_running=boom))())

    def test_the_face_loop_skips_its_step_while_the_pump_runs(self):
        src = _source()
        at = src.index("_handtrack_depth.publish_kinect()")
        guard = src.rindex("if not camera_off and not _handtrack_pump_running():", 0, at)
        self.assertLess(at - guard, 120)
        self.assertEqual(src.count("_handtrack_depth.publish_kinect()"), 1)


if __name__ == "__main__":
    unittest.main()
