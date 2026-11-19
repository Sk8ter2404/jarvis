"""Misheard names (v3.0.26, ovh-b): "holo HUD" heard as "hollow hood / hud /
hug" and "hologram hot hood" failed every time on 2026-11-17 (the day he shut
JARVIS down), and the launch_app follow-up round invented "Hollow Knight".

  * core/dispatcher.holo_route: "the holo HUD" / "the hologram HUD" and their
    misheard forms open / close the hologram window without the brain; a bare
    "hollow" / "holo", "Hollow Knight" and the overlay's words do not.
  * core/config.STT_REPLACEMENTS: whole-phrase rewrites only.
  * core/prompt_router: the misheard phrases load HOLO MODE; the prompt calls
    the hologram "the holo HUD" and the overlay no longer claims that name.
  * core/misheard_names + core/actions._act_launch_app: a launch name nothing
    could start is checked against JARVIS's own windows (offered as a terminal
    line, never run) and the installed apps (named in the failure line), and
    the follow-up round gets a rule against guessing app names.

Synthetic names and the short live command phrases only. Light tier (no
monolith import); os.startfile / subprocess are always mocked, and the
Start-menu list is always a fixture - a test never launches or reads a real
program.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import actions as A  # noqa: E402
from core import config  # noqa: E402
from core import misheard_names as mn  # noqa: E402
from core import offer_reply  # noqa: E402
from core import prompt_router as pr  # noqa: E402
from core import prompts  # noqa: E402
from core import stt_vocab  # noqa: E402
from core.dispatcher import holo_route  # noqa: E402
from core.failure_markers import TERMINAL_FAILURE_PREFIX, terminal_failure_text  # noqa: E402

# The 11-17 transcripts as the owner's turns logged them (command phrases only).
LIVE_OPEN = (
    "Jarvis open a hollow hood.",          # 16:53:44 - declined
    "Jarvis open hologram hot hood.",      # 16:53:58 - holo_show, 'hot hood'
    "Jarvis open hollow hood.",            # 17:12:41 - launch_app 'Hollow Hood'
    "Jarvis open hollow hud.",             # 17:12:54 - launch_app, 'Hollow Knight'
    "Jarvis is open the hologram HUD.",    # 16:03:52 - brain, holo_mode_on
    "Jarvis open hologram HUD.",           # 16:54:41 - brain, holo_mode_on
    "Jarvis open the hologram HUD.",       # 17:00:24 - brain, holo_mode_on
)
LIVE_CLOSE = ("Jarvis close holographic hug.",)   # 16:11:01 - brain, holo_mode_off

REGISTERED = {"holo_mode_on", "holo_mode_off", "hand_menu",
              "show_holographic_overlay"}


def _heard(text: str) -> str:
    """What the turn reads after the built-in STT_REPLACEMENTS (both engines)."""
    return stt_vocab.apply_replacements(text, config.STT_REPLACEMENTS)


# ── 1. the voice route ──────────────────────────────────────────────────────
class HoloHudRouteTests(unittest.TestCase):
    def test_every_live_open_routes_to_the_hologram(self):
        for t in LIVE_OPEN:
            with self.subTest(t=t):
                self.assertEqual(holo_route(t), "[ACTION: holo_mode_on]")
                # ... and after the STT rewrite too (what the turn really sees).
                self.assertEqual(holo_route(_heard(t)), "[ACTION: holo_mode_on]")

    def test_the_live_close_routes_off(self):
        for t in LIVE_CLOSE:
            self.assertEqual(holo_route(t), "[ACTION: holo_mode_off]", t)
            self.assertEqual(holo_route(_heard(t)), "[ACTION: holo_mode_off]", t)

    def test_the_holo_hud_by_name(self):
        for t in ("open the holo HUD", "fire up the holo HUD",
                  "bring up the hologram HUD", "turn on the holo hud please",
                  "Jarvis, show me the holographic HUD", "holo HUD on",
                  "open my hollow hud", "start the holo hut"):
            self.assertEqual(holo_route(t), "[ACTION: holo_mode_on]", t)
        for t in ("close the holo HUD", "kill the holo hud", "holo HUD off",
                  "turn off the hologram HUD", "shut down the holo HUD",
                  "Jarvis is close the hollow hood"):
            self.assertEqual(holo_route(t), "[ACTION: holo_mode_off]", t)
        self.assertEqual(holo_route("open the holo HUD with the Kinect"),
                         "[ACTION: holo_mode_on, kinect]")
        self.assertEqual(holo_route("open the holo HUD on the Quest"),
                         "[ACTION: holo_quest]")

    def test_near_misses_stay_with_the_brain(self):
        for t in ("open Hollow Knight", "open hollow", "open holo",
                  "bring up the holo", "open holographic",
                  "show the holographic overlay",
                  "engage the holographic display",
                  "turn off the holographic display",
                  "open the holo HUD v2", "open Hulu", "open Halo",
                  "open hot hood", "open the hood", "open the HUD",
                  "is the hologram on", "is the holo HUD on",
                  "open hollow hood and play music",
                  "what is a holo HUD", "a hollow log"):
            self.assertIsNone(holo_route(t, holo_up=True), t)

    def test_the_is_lead_is_dropped_only_before_a_verb(self):
        # "is the hologram on" is a status question: never "the hologram on".
        self.assertIsNone(holo_route("Jarvis is the hologram on"))
        self.assertIsNone(holo_route("is hologram on"))
        self.assertEqual(holo_route("Jarvis is open the hologram"),
                         "[ACTION: holo_mode_on]")


# ── 2. the STT rewrite ──────────────────────────────────────────────────────
class HoloSttReplacementTests(unittest.TestCase):
    def test_the_live_lines_are_rewritten(self):
        self.assertEqual(_heard("Jarvis open a hollow hood."),
                         "Jarvis open a holo HUD.")
        self.assertEqual(_heard("Jarvis open hollow hud."), "Jarvis open holo HUD.")
        self.assertEqual(_heard("open Hollow Hug"), "open holo HUD")
        self.assertEqual(_heard("Jarvis open hologram hot hood."),
                         "Jarvis open hologram HUD.")
        self.assertEqual(_heard("Jarvis close holographic hug."),
                         "Jarvis close holographic HUD.")
        self.assertEqual(_heard("open the hollow h u d"), "open the holo HUD")

    def test_single_words_and_other_phrases_are_never_touched(self):
        for t in ("open Hollow Knight", "a hollow log", "the hollow of a tree",
                  "is the hood hot", "a hot hood", "hug me", "my hood is up",
                  "Hollow", "holo", "the holo deck", "holographic overlay"):
            self.assertEqual(_heard(t), t, t)

    def test_every_built_in_holo_entry_is_a_phrase(self):
        for src in config.STT_REPLACEMENTS:
            if any(w in src for w in ("holo", "hollow")):
                self.assertGreaterEqual(len(src.split()), 2, src)

    def test_parakeet_uses_the_same_map(self):
        # _parakeet_post_text applies STT_REPLACEMENTS before the Parakeet map,
        # so the built-ins reach the live engine (STT_ENGINE parakeet).
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        at = src.index("def _parakeet_post_text")
        body = src[at:at + 600]
        self.assertIn('globals().get("STT_REPLACEMENTS")', body)


# ── 3. routing and the prompt ───────────────────────────────────────────────
class HoloPromptRoutingTests(unittest.TestCase):
    FULL = prompts.PC_CONTROL_PROMPT

    def setUp(self):
        _core, self.sections = pr.split_pc_control(self.FULL)

    def test_the_misheard_phrases_load_holo_mode(self):
        # Raw (a typed turn, or a variant the STT map does not hold).
        for q in ("open hollow hud", "open a hollow hood", "close hollow hug",
                  "open the holo HUD"):
            with self.subTest(q=q):
                self.assertIn("HOLO MODE", pr.select_sections(q, self.sections)[0])
                self.assertIn("holo_mode_on", pr.turn_pc_block(q, self.FULL))

    def test_a_bare_hollow_does_not(self):
        for q in ("open Hollow Knight", "that sounds hollow"):
            self.assertNotIn("HOLO MODE",
                             pr.select_sections(q, self.sections)[0], q)

    def test_the_prompt_names_the_hologram_the_holo_hud(self):
        flat = " ".join(self.FULL.split())
        self.assertIn("'open the holo HUD' / 'the hologram HUD' (his name for "
                      "it) → [ACTION: holo_mode_on]", flat)
        self.assertIn("'close the holo HUD' → [ACTION: holo_mode_off]", flat)
        # The overlay's own lines no longer claim "the holo HUD" (a stale
        # duplicate the brain would have to choose between).
        start = flat.index("show_holographic_overlay —")
        end = flat.index("workshop_hud —", start)
        overlay = flat[start:end]
        self.assertNotIn("fire up the holo HUD", overlay)
        self.assertNotIn("kill the holo HUD", overlay)
        self.assertIn("'The holo HUD' is the hologram, holo_mode_on.", overlay)


# ── 4. JARVIS's own features ────────────────────────────────────────────────
class FeatureForTests(unittest.TestCase):
    def label(self, name, registered=REGISTERED):
        f = mn.feature_for(name, registered=registered)
        return f.label if f else None

    def test_the_live_names_are_the_holo_hud(self):
        for n in ("Hollow Hood", "hollow hud", "Hollow Hug", "holo HUD",
                  "the hologram HUD", "Hologram", "hologram hot hood",
                  "holographic hug", "the Hollow Hood app"):
            self.assertEqual(self.label(n), "the holo HUD", n)

    def test_sound_alike_names_beyond_the_route(self):
        for n in ("hollow head", "hollo hood", "holagram", "hollow h u d",
                  "holo hut"):
            self.assertEqual(self.label(n), "the holo HUD", n)
        self.assertEqual(self.label("hand menu"), "the hand menu")
        self.assertEqual(self.label("hand meny"), "the hand menu")
        self.assertEqual(self.label("the holographic display"),
                         "the holographic overlay")

    def test_near_misses_are_no_feature(self):
        for n in ("Hollow Knight", "Hollow", "holo", "Hulu", "Halo",
                  "Halo Infinite", "HoloCure", "Holo Cure", "HoloLens", "hud",
                  "hood", "hot hood", "hot tub", "home hub", "Hand Brake",
                  "HandBrake", "Hulu Plus", "Halo Hub", "hologram knight",
                  "holo hud v2", "holo deck", "Hello Kitty", "Arc Raiders",
                  "Notepad", "Steam", "Spotify", "Discord", "Outlook",
                  "Microsoft Teams", "Chrome", "Hollow Hood Simulator 2",
                  "", "a", "the app", None, 42, "x" * 80):
            self.assertIsNone(self.label(n), n)

    def test_an_extra_word_never_vanishes_into_the_sound_key(self):
        # h / w / y count as vowels in the key, so a run-together name could
        # swallow a whole short word: "hughologram" keys like "hologram".
        for n in ("hug hologram", "hug hologram hud", "holy hologram",
                  "holographic display holy", "hello hologram",
                  "hologram hud way", "hand menu hi", "hand menu a",
                  "hologram a", "holo gram x"):
            self.assertIsNone(self.label(n), n)
        # ... while letters merely run together still match.
        self.assertEqual(self.label("hollow h u d"), "the holo HUD")
        self.assertEqual(self.label("handmenu"), "the hand menu")
        self.assertTrue(mn.names_alike("work shop", "workshop"))
        self.assertFalse(mn.names_alike("hug hologram", "hologram"))

    def test_a_feature_that_is_not_loaded_is_never_offered(self):
        self.assertIsNone(self.label("Hollow Hood", registered={"hand_menu"}))
        self.assertIsNone(self.label("Hollow Hood", registered=set()))
        self.assertEqual(self.label("hand menu", registered={"hand_menu"}),
                         "the hand menu")

    def test_two_features_at_once_is_none(self):
        twin = mn.Feature("the twin", "twin_on", "[ACTION: twin_on]",
                          ("hollow hood",))
        with mock.patch.object(mn, "FEATURES", mn.FEATURES + (twin,)):
            # The exact route still names the holo HUD ...
            self.assertEqual(self.label("Hollow Hood", REGISTERED | {"twin_on"}),
                             "the holo HUD")
            # ... but a sound-alike that fits both is no guess.
            self.assertIsNone(self.label("hollow head", REGISTERED | {"twin_on"}))

    def test_every_feature_opens_and_is_named_safely(self):
        for f in mn.FEATURES:
            self.assertTrue(f.token.startswith(f"[ACTION: {f.action}"), f)
            self.assertNotIn("toggle", f.action)
            self.assertNotIn("off", f.action)
            for n in f.names:
                words = n.split()
                self.assertTrue(len(words) >= 2 or len(n) >= 6, n)

    def test_phonetic_key(self):
        self.assertEqual(mn.phonetic_key("hollow"), mn.phonetic_key("holo"))
        self.assertEqual(mn.phonetic_key("hood"), mn.phonetic_key("hud"))
        self.assertNotEqual(mn.phonetic_key("knight"), mn.phonetic_key("hud"))
        self.assertEqual(mn.phonetic_key(""), "")
        self.assertEqual(mn.phonetic_key(None), "")


class OfferRoundTripTests(unittest.TestCase):
    def test_the_offer_is_kept_as_an_open_offer_and_reads_back(self):
        f = mn.feature_for("Hollow Hood", registered=REGISTERED)
        line = mn.offer_line("Hollow Hood.", f)
        self.assertEqual(line, "I've no app called Hollow Hood, sir. "
                               "Shall I open the holo HUD instead?")
        offer = offer_reply.offer_text(line)
        self.assertEqual(offer, "Shall I open the holo HUD instead?")
        self.assertEqual(mn.offer_token(offer), "[ACTION: holo_mode_on]")

    def test_every_feature_round_trips(self):
        for f in mn.FEATURES:
            offer = offer_reply.offer_text(mn.offer_line("Some Name", f))
            self.assertEqual(mn.offer_token(offer), f.token, f.label)

    def test_other_offers_have_no_token(self):
        for o in ("Shall I open Notepad instead?",
                  "Shall I search for it on the web instead?",
                  "I'll assume you meant Hollow Knight; shall I launch that "
                  "instead?", "", None):
            self.assertEqual(mn.offer_token(o), "", o)


# ── 5. the installed apps ───────────────────────────────────────────────────
INSTALLED = ("Steam", "Deadlock", "Wallpaper Engine", "Discord",
             "Microsoft Teams", "Outlook", "Notepad", "Bambu Studio",
             "Example Racer", "Steam Support Center")


class ClosestAppsTests(unittest.TestCase):
    def test_close_names(self):
        self.assertEqual(mn.closest_apps("Discrod", INSTALLED), ["Discord"])
        self.assertEqual(mn.closest_apps("Bamboo Studio", INSTALLED),
                         ["Bambu Studio"])
        self.assertEqual(mn.closest_apps("dead lock", INSTALLED), ["Deadlock"])
        self.assertEqual(mn.closest_apps("Steem", INSTALLED), ["Steam"])
        self.assertEqual(mn.closest_apps("Example Raser", INSTALLED),
                         ["Example Racer"])

    def test_nothing_close(self):
        for n in ("Hollow Hood", "Hollow Knight", "Teams Classic Player",
                  "Teams", "Spotify", "", None):
            self.assertEqual(mn.closest_apps(n, INSTALLED), [], n)
        self.assertEqual(mn.closest_apps("Discrod", ()), [])
        self.assertEqual(mn.closest_apps("Discrod", None), [])

    def test_at_most_two_closest_first(self):
        apps = ("Example One", "Example Ane", "Example Onx", "Other")
        got = mn.closest_apps("Example On", apps)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0], "Example One")

    def test_saved_sites(self):
        sites = {"example forge": "https://example.test/forge",
                 "zorblat tickets": "https://example.test/z",
                 "not a page": "file:///c:/x", 7: "https://example.test/7"}
        self.assertEqual(mn.closest_sites("Example Forj", sites),
                         ["example forge"])
        self.assertEqual(mn.closest_sites("zorblat tikets", sites),
                         ["zorblat tickets"])
        for n in ("example forge",       # exact: it already opened its page
                  "Hollow Hood", "not a page", "Steam", "", None):
            self.assertEqual(mn.closest_sites(n, sites), [], n)
        self.assertEqual(mn.closest_sites("Example Forj", None), [])
        self.assertEqual(mn.closest_sites("Example Forj", ["x"]), [])
        self.assertEqual(mn.closest_sites_note([]), "")
        self.assertEqual(mn.closest_sites_note(["example forge"]),
                         "; a saved site with a similar name is example forge")
        self.assertEqual(mn.closest_sites_note(["a1", "b2"]),
                         "; saved sites with similar names are a1 and b2")

    def test_note(self):
        self.assertEqual(mn.closest_apps_note([]),
                         "; no installed app has a similar name")
        self.assertEqual(mn.closest_apps_note(["Discord"]),
                         "; the closest installed app is Discord")
        self.assertEqual(mn.closest_apps_note(["A1", "B2"]),
                         "; the closest installed apps are A1 and B2")


class FollowupRuleTests(unittest.TestCase):
    NOT_FOUND = ("could not launch Hollow Hood: not found as an app on this PC "
                 "([WinError 2] x); no installed app has a similar name")

    def test_only_for_a_not_found_launch(self):
        self.assertEqual(mn.followup_rule([("launch_app", self.NOT_FOUND)]),
                         mn.FOLLOWUP_RULE)
        self.assertEqual(mn.followup_rule([("open_on_monitor", self.NOT_FOUND)]),
                         mn.FOLLOWUP_RULE)
        for res in ([("launch_app", "launched Notepad")],
                    [("launch_app", "could not launch X: access denied")],
                    [], None, [("x",)], [("x", None)]):
            self.assertEqual(mn.followup_rule(res), "", res)

    def test_the_setting_switches_it_off(self):
        with mock.patch.object(config, "LAUNCH_NOT_FOUND_SUGGEST", False):
            self.assertEqual(mn.followup_rule([("launch_app", self.NOT_FOUND)]), "")
        self.assertEqual(
            mn.followup_rule([("launch_app", self.NOT_FOUND)], enabled=False), "")

    def test_the_rule_forbids_guessing(self):
        self.assertIn("Never guess another app or game name", mn.FOLLOWUP_RULE)

    def test_config_default_on(self):
        self.assertIs(config.LAUNCH_NOT_FOUND_SUGGEST, True)


# ── 6. launch_app itself ────────────────────────────────────────────────────
class _FakeBc:
    def __init__(self, actions):
        self.ACTIONS = actions
        self.ran = []

    def _resolve_known_app(self, _name):
        return None


class LaunchNotFoundTests(unittest.TestCase):
    """The REAL _act_launch_app; only the OS edges are faked."""

    def setUp(self):
        self.ran = []
        self.bc = _FakeBc({a: (lambda arg="", a=a: self.ran.append(a) or "ok")
                           for a in REGISTERED | {"launch_app"}})
        self.apps = ()
        for target, kw in (
                (A, dict(attribute="_bc", return_value=self.bc)),
                (A, dict(attribute="_unresolved_open_line", return_value="")),
                (A, dict(attribute="_site_shortcut_url", return_value=None)),
                (A, dict(attribute="_website_name_url", return_value=None)),
                (A, dict(attribute="_start_apps", side_effect=lambda: self.apps)),
                (config, dict(attribute="SITE_SHORTCUTS", new={})),
                (A.shutil, dict(attribute="which", return_value=None)),
                (A.sys, dict(attribute="platform", new="win32"))):
            self._start(mock.patch.object(target, **kw))
        self.popen = self._start(mock.patch.object(A.subprocess, "Popen"))
        self.startfile = self._start(mock.patch.object(
            A.os, "startfile", create=True,
            side_effect=FileNotFoundError(2, "The system cannot find the file "
                                             "specified")))

    def _start(self, patcher):
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def launch(self, name):
        out = io.StringIO()
        with redirect_stdout(out):
            res = A._act_launch_app(name)
        return res, out.getvalue()

    def test_the_live_name_is_offered_not_run(self):
        res, log = self.launch("Hollow Hood")
        self.assertTrue(res.startswith(TERMINAL_FAILURE_PREFIX), res)
        self.assertEqual(terminal_failure_text(res),
                         "I've no app called Hollow Hood, sir. "
                         "Shall I open the holo HUD instead?")
        self.assertEqual(self.ran, [])            # nothing opened on a guess
        self.popen.assert_not_called()            # not even explorer.exe
        self.assertIn("[launch-feature] no app called 'Hollow Hood'", log)
        # still a failure for every FAILURE_MARKERS reader (open_on_monitor)
        self.assertTrue(A._launch_failed(res))

    def test_the_offer_needs_the_feature_registered(self):
        self.bc.ACTIONS = {"launch_app": object()}
        res, _log = self.launch("Hollow Hood")
        self.assertFalse(res.startswith(TERMINAL_FAILURE_PREFIX), res)
        self.assertIn("not found as an app on this PC", res)

    def test_an_unknown_game_names_what_is_installed(self):
        self.apps = (("Example Racer", "id.racer"), ("Notepad", "id.notepad"))
        res, _log = self.launch("Hollow Knight")
        self.assertTrue(res.startswith("could not launch Hollow Knight: not "
                                       "found as an app on this PC ("), res)
        self.assertTrue(res.endswith("; no installed app has a similar name"), res)
        self.assertEqual(mn.followup_rule([("launch_app", res)]),
                         mn.FOLLOWUP_RULE)
        res, _log = self.launch("Example Raser")
        self.assertTrue(res.endswith("; the closest installed app is "
                                     "Example Racer"), res)
        self.popen.assert_not_called()

    def test_an_unreadable_start_menu_claims_nothing(self):
        self.apps = ()
        res, _log = self.launch("Hollow Knight")
        self.assertTrue(res.endswith(")"), res)
        self.assertNotIn("installed app", res)

    def test_a_saved_site_that_sounds_like_it_is_named(self):
        # Live config default shape: SITE_SHORTCUTS is the owner's dict, read
        # live from core.config (here a synthetic one).
        self.apps = (("Example Racer", "id.racer"),)
        with mock.patch.object(config, "SITE_SHORTCUTS",
                               {"example forge": "https://example.test/f"}):
            res, _log = self.launch("Example Forj")
            self.assertTrue(res.endswith(
                "; no installed app has a similar name; a saved site with a "
                "similar name is example forge"), res)
            self.apps = ()             # Start menu unreadable: the site only
            res, _log = self.launch("Example Forj")
            self.assertTrue(res.endswith(
                "); a saved site with a similar name is example forge"), res)
        self.popen.assert_not_called()

    def test_the_setting_off_is_the_3025_line(self):
        self.apps = (("Example Racer", "id.racer"),)
        with mock.patch.object(config, "LAUNCH_NOT_FOUND_SUGGEST", False):
            res, log = self.launch("Hollow Hood")
        self.assertEqual(res, "could not launch Hollow Hood: not found as an app "
                              "on this PC ([Errno 2] The system cannot find the "
                              "file specified)")
        self.assertNotIn("[launch-feature]", log)

    def test_an_installed_app_still_launches(self):
        self.apps = (("Hollow Hood", "id.hh"),)   # a real app of that name wins
        res, _log = self.launch("Hollow Hood")
        self.assertEqual(res, "launched Hollow Hood")
        self.popen.assert_called_once_with(
            ["explorer.exe", "shell:AppsFolder\\id.hh"], close_fds=True)
        self.assertEqual(self.ran, [])

    def test_a_non_not_found_error_is_never_an_offer(self):
        self.startfile.side_effect = PermissionError(5, "Access is denied")
        res, _log = self.launch("Hollow Hood")
        self.assertEqual(res, "could not launch Hollow Hood: [Errno 5] Access "
                              "is denied")


if __name__ == "__main__":
    unittest.main()
