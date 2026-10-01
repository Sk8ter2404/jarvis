"""The dashboard's pinned wake-word switch drives REQUIRE_WAKE_MODE, live
(2026-10-01).

The owner's "wake-word mode" is REQUIRE_WAKE_MODE — respond only when
addressed by name — flipped live by the "wake word mode on/off" voice command
(bobert_companion._act_wake_word_mode_set: the runtime mirror, core.config and
the settings file). The web dashboard's pinned banner switch wrote
WAKE_WORD_AUTOSTART at first and, after the 157 batch, START_IN_STANDBY —
"start in standby" — by saving the settings file, so flipping it never
changed what JARVIS answers to and showed the wrong knob's state.

The switch now sends wake_word_mode_on / wake_word_mode_off through the tray
control plane (POST /api/control -> tray_commands.json -> the monolith's
drainer -> _act_wake_word_mode_set), so it applies live exactly as the voice
command does, and it shows the LIVE state the loop publishes in hud_state.json
(``require_wake_mode``). START_IN_STANDBY and WAKE_WORD_AUTOSTART stay
ordinary settings rows. The monolith half is
tests/monolith/test_monolith_diag_fixes.py::WakeWordModeTrayTests.

Headless-CI safe: a 127.0.0.1:0 server in a temp dir (tests.test_web_interface).

    python -m unittest tests.test_web_wake_mode
"""
from __future__ import annotations

import json
import os
import unittest

from tools import web_interface as wi
from tests.test_web_interface import _ServerBase, _get, _get_raw, _js_fn, _post


def _banner(page):
    banner = page[page.index('<div class="wakebanner">'):]
    return banner[:banner.index("</div>")]


class PinnedSwitchIsRequireWakeModeTests(unittest.TestCase):
    PAGE = wi._DASHBOARD_PAGE

    def test_the_switch_drives_require_wake_mode(self):
        self.assertIn("const WAKE_KEY = 'REQUIRE_WAKE_MODE';", self.PAGE)
        banner = _banner(self.PAGE)
        self.assertIn("REQUIRE_WAKE_MODE", banner)
        self.assertNotIn("(START_IN_STANDBY)", banner)
        label = banner[banner.index('class="lbl">') + len('class="lbl">'):]
        label = label[:label.index("</span>")].lower()
        self.assertIn("wake-word mode", label)
        self.assertNotIn("standby", label)

    def test_saving_it_goes_through_the_live_control_plane(self):
        page = self.PAGE
        save = page[page.index("wakeSave.addEventListener"):]
        save = save[:save.index("});") + 3]
        self.assertIn("sendControl(", save)
        self.assertIn("wake_word_mode_on", save)
        self.assertIn("wake_word_mode_off", save)
        self.assertNotIn("saveSetting(", save,
                         "the banner wrote the settings file again (not live)")

    def test_it_shows_the_live_state(self):
        status = _js_fn(self.PAGE, "refreshStatus")
        self.assertIn("require_wake_mode", status)

    def test_the_other_wake_knobs_are_still_ordinary_rows(self):
        from tools import settings_window as sw
        for key in ("START_IN_STANDBY", "WAKE_WORD_AUTOSTART",
                    "REQUIRE_WAKE_MODE"):
            self.assertIn(key, sw.SCHEMA)

    def test_both_commands_are_web_controls(self):
        self.assertIn("wake_word_mode_on", wi.TRAY_WEB_COMMANDS)
        self.assertIn("wake_word_mode_off", wi.TRAY_WEB_COMMANDS)
        self.assertNotIn("wake_word_mode_on", wi._TRAY_CONFIRM)


class StatusCarriesTheLiveModeTests(unittest.TestCase):
    def test_published_value_is_reported(self):
        self.assertIs(wi._status_flags({"require_wake_mode": True})
                      ["require_wake_mode"], True)
        self.assertIs(wi._status_flags({"require_wake_mode": False})
                      ["require_wake_mode"], False)

    def test_unpublished_is_unknown_not_off(self):
        self.assertIsNone(wi._status_flags({})["require_wake_mode"])


class WakeControlRouteTests(_ServerBase):
    def test_the_switch_queues_the_live_command(self):
        for cmd in ("wake_word_mode_on", "wake_word_mode_off"):
            code, d = _post(self.base + "/api/control", {"cmd": cmd})
            self.assertEqual(code, 200, d)
        with open(self.tray_path, encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual([i["cmd"] for i in items],
                         ["wake_word_mode_on", "wake_word_mode_off"])
        self.assertTrue(all(i.get("cid") for i in items))

    def test_status_route_reports_it(self):
        with open(self.hud_path, "w", encoding="utf-8") as f:
            json.dump({"state": "Idle", "require_wake_mode": True}, f)
        code, s = _get(self.base + "/api/status")
        self.assertEqual(code, 200)
        self.assertIs(s["require_wake_mode"], True)

    def test_the_page_carries_the_switch(self):
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn('id="wakeToggle"', body)
        self.assertIn("REQUIRE_WAKE_MODE", body)


if __name__ == "__main__":
    unittest.main()
