"""v3.0.9 lane B1: a site said by name opens as a site (core/actions.py).

Live 2026-10-28 21:12 "Jarvis Open Makers World on main monitor." ran
[ACTION: open_on_monitor, main | Makers World]; nothing knew the name was a
website, so it was launched as a program ("could not launch Makers World:
[WinError 2]") and JARVIS offered a web search instead.

Pinned: the maker / 3D-printing sites are in the site table; with
SITE_NAME_SPOKEN_FORMS on, the plural / possessive spoken forms ("makers
world", "maker's world") find the same row, launch_app opens a known site in
the browser once no program or Start-menu app has the name, and a name still
not found says "not found as an app on this PC"; an unknown name is never made into a URL. With
the flag off only the exact table spelling matches and the raw launch error
comes back (the old path). No real launches: os.startfile / Popen and the
browser open are patched.

    python -m unittest tests.test_actions_site_names_309
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import actions as act
import core.config as cfg

MAKERWORLD = "https://makerworld.com/"


class _FlagBase(unittest.TestCase):
    def _flag(self, value):
        p = mock.patch.object(cfg, "SITE_NAME_SPOKEN_FORMS", value,
                              create=True)
        p.start()
        self.addCleanup(p.stop)


class SiteTableTests(_FlagBase):
    def test_maker_sites_are_in_the_table(self):
        u = act._website_name_url
        self.assertEqual(u("makerworld"), MAKERWORLD)
        self.assertEqual(u("maker world"), MAKERWORLD)
        self.assertEqual(u("printables"), "https://www.printables.com/")
        self.assertEqual(u("Printables.com"), "https://www.printables.com/")
        self.assertEqual(u("thingiverse"), "https://www.thingiverse.com/")
        for n in ("cults", "cults3d", "cults 3d", "the cults3d website"):
            self.assertEqual(u(n), "https://cults3d.com/", n)

    def test_spoken_plural_and_possessive_forms(self):
        self._flag(True)
        u = act._website_name_url
        for n in ("Makers World", "maker's world", "Maker’s World",
                  "the makers world website", "MakerWorld.com"):
            self.assertEqual(u(n), MAKERWORLD, n)

    def test_unknown_names_are_never_made_into_urls(self):
        self._flag(True)
        u = act._website_name_url
        for n in ("Bambu Studio", "spotify", "notepad", "claude", "twitch",
                  "news", "makers", "world", "", None):
            self.assertIsNone(u(n), n)

    def test_flag_off_only_the_exact_spelling(self):
        self._flag(False)
        u = act._website_name_url
        self.assertIsNone(u("Makers World"))
        self.assertIsNone(u("maker's world"))
        self.assertEqual(u("maker world"), MAKERWORLD)
        self.assertEqual(u("youtube"), "https://www.youtube.com")


class LaunchAppTests(_FlagBase):
    def setUp(self):
        for target, kw in (("_unresolved_open_line", {"return_value": ""}),
                           ("_site_shortcut_url", {"return_value": None})):
            p = mock.patch.object(act, target, **kw)
            p.start()
            self.addCleanup(p.stop)
        self.opened = []
        p = mock.patch.object(act, "_act_open_url",
                              side_effect=lambda u: (self.opened.append(u)
                                                     or f"opened {u}"))
        p.start()
        self.addCleanup(p.stop)

    def _fake_bc(self):
        return mock.MagicMock(_resolve_known_app=lambda n: None)

    def test_a_known_site_opens_when_no_program_has_the_name(self):
        self._flag(True)
        out = self._launch_missing("Makers World")
        self.assertEqual(self.opened, [MAKERWORLD])
        self.assertIn(MAKERWORLD, out)

    def test_an_installed_app_with_a_sites_name_still_opens_as_the_app(self):
        # Review 10-29: a Start-menu "YouTube" app launched on the old path;
        # the site table must not take it over (it is checked last).
        self._flag(True)
        missing = FileNotFoundError(2, "The system cannot find the file "
                                       "specified")
        with mock.patch.object(act, "_bc", return_value=self._fake_bc()), \
             mock.patch.object(act.shutil, "which", return_value=None), \
             mock.patch.object(act.sys, "platform", "win32"), \
             mock.patch.object(act.os, "startfile", side_effect=missing,
                               create=True), \
             mock.patch.object(act, "_start_app_id",
                               return_value="Example.YouTube!App"), \
             mock.patch.object(act.subprocess, "Popen") as popen:
            out = act._act_launch_app("YouTube")
        self.assertEqual(out, "launched YouTube")
        self.assertEqual(self.opened, [])
        popen.assert_called_once()

    def _launch_missing(self, name):
        missing = FileNotFoundError(2, "The system cannot find the file "
                                       "specified")
        with mock.patch.object(act, "_bc", return_value=self._fake_bc()), \
             mock.patch.object(act.shutil, "which", return_value=None), \
             mock.patch.object(act.sys, "platform", "linux"), \
             mock.patch.object(act.subprocess, "Popen", side_effect=missing):
            return act._act_launch_app(name)

    def test_a_name_still_not_found_says_so(self):
        self._flag(True)
        out = self._launch_missing("Frobnicator Pro")
        self.assertTrue(out.startswith("could not launch Frobnicator Pro: "
                                       "not found as an app on this PC"), out)
        self.assertTrue(act._launch_failed(out))
        self.assertEqual(self.opened, [])

    def test_flag_off_is_the_old_path(self):
        self._flag(False)
        out = self._launch_missing("Makers World")
        self.assertEqual(self.opened, [])
        self.assertNotIn("not found as an app", out)
        self.assertTrue(out.startswith("could not launch Makers World: "), out)

    def test_other_launch_errors_keep_their_own_words(self):
        self._flag(True)
        line = act._launch_failure_line("Thing", PermissionError("denied"))
        self.assertEqual(line, "could not launch Thing: denied")


if __name__ == "__main__":
    unittest.main()
