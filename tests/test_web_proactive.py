"""The dashboard's Proactive page (GET / POST /api/proactive, 2026-10-02).

ONE page lists every behaviour JARVIS starts on his own - its settings key,
current value, a one-line description, and whether a change applies live or on
the next start - with a switch that saves through the Settings tab's own write
helper (tools/web_interface._write_settings). The list is DATA
(web_interface.PROACTIVE_FEATURES), so these tests can hold it to the code:

  * completeness: core/config.py and bobert_companion.py are grepped for
    on/off flags with a proactive / background name; each must be on the page,
    or listed in NOT_PROACTIVE below with the reason it is not one;
  * truthfulness: a "restart" row's key is a core/config.py constant, a "code"
    row's key lives only in bobert_companion.py, and a switchable row is a
    bool row of the Settings schema (the only keys the save path accepts);
  * the switch goes through _write_settings (faked here), and nothing else;
  * every value the page shows lands as text, never markup.

Headless-CI safe: a 127.0.0.1:0 server in a temp dir (tests.test_web_interface).

    python tools/run_tests.py web_proactive
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unittest
from unittest import mock

from tools import settings_window as sw
from tools import web_interface as wi
from tests.test_web_interface import _ServerBase, _get, _get_raw, _js_fn, _post
from tests.test_web_timeline import DOM_SHIM

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The proactive / background flags found in the code (2026-10-02 audit). Each
# must be a row of the page.
KNOWN_CONFIG_KEYS = (
    "AMBIENT_LISTEN_ENABLED", "AMBIENT_SCREEN_ENABLED", "TEAMS_NUDGE_ENABLED",
    "PROCESSING_FILLER_ENABLED", "LOCAL_PREFIX_REPRIME",
    "MISSION_NARRATION_ENABLED", "NIGHT_QUIET_ENABLED", "NIGHT_OWL_AUTO",
    "STANDBY_LOOP_ENABLED", "KINECT_PRESENCE_STANDBY", "KINECT_PRESENCE_WAKE",
    "KINECT_GREET_ON_ENTRY", "KINECT_POSTURE_NUDGE",
    "GREET_NEW_PEOPLE_ENABLED", "OVERNIGHT_UPGRADE_ENABLED",
    "GAME_MODE_ENABLED", "AUDIO_AUTOSWITCH_ENABLED", "CHAPPIE_ENABLED",
    "MID_TASK_STATUS_ENABLED", "UPDATE_CHECK_ENABLED", "ROBOT_ENABLED",
    "APPLE_MUSIC_AUTOSTART", "APPLE_MUSIC_KEEP_OPEN",
)
KNOWN_MONOLITH_KEYS = (
    "PROACTIVE_ENABLED", "ANTICIPATION_ENABLED",
    "ANTICIPATION_BRIEFING_ENABLED", "WEEKLY_DIGEST_ENABLED",
    "DAILY_BRIEFING_ENABLED", "EVENING_BRIEFING_ENABLED",
    "DAILY_RECAP_ENABLED", "WEATHER_BRIEFING_PROACTIVE",
    "AMAZON_TRACKING_ENABLED", "LEARN_EVERY_TURN", "AMBIENT_AUDIO_ENABLED",
)

# Names that look proactive / background. A module-level `NAME = True|False`
# in core/config.py or bobert_companion.py matching this must be on the page
# or in NOT_PROACTIVE.
_PROACTIVE_NAME = re.compile(
    r"PROACTIVE|BANTER|ANTICIPATION|NUDGE|REPRIME|FILLER|NARRATION|MID_TASK|"
    r"GREET|BRIEFING|RECAP|DIGEST|AUTOSTART|KEEP_OPEN|CHAPPIE|OVERNIGHT|"
    r"UPDATE_CHECK|AMBIENT_|LEARN_EVERY|PRESENCE_|NIGHT_|STANDBY_LOOP|"
    r"AUTO_LAUNCH|AUTO_ON|AUTO_WHILE|TRACKING|ROBOT_|GAME_MODE_ENABLED|"
    r"AUTOSWITCH")
_BOOL_FLAG = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*(?:True|False)\b", re.M)

# Matches of the grep that are NOT a behaviour of their own, and why.
_RETIRED = "a retired overlay, superseded by the unified HUD (forced off)"
NOT_PROACTIVE = {
    "AMBIENT_LEARNING_FORCE_LOCAL": "picks the model ambient learning uses",
    "AMBIENT_MUSIC_REFUSE_WAKE": "a wake-word filter",
    "AMBIENT_STT_YIELD": "scheduling inside ambient listening",
    "PROCESSING_FILLER_SKIP_PLEASANTRIES": "a condition on the processing "
                                           "filler (its row)",
    "FILLER_DUCK_HOLD": "how long the music stays ducked around the filler "
                        "and the answer",
    "PROCESSING_FILLER_PRERENDER": "renders the answer during the filler; "
                                   "says nothing of its own",
    "AMBIENT_EXTRACT_ENABLED": "starts nothing; the extractor follows the "
                               "capture sources",
    "PROACTIVE_REQUIRE_FACE": "a condition on proactive comments",
    "PROACTIVE_REQUIRE_OWNER_VOICE": "a condition on proactive comments",
    "PRESENCE_HOLD_ENABLED": "a condition on queued proactive lines: they "
                             "wait while he is away or the room is talking",
    "NEWS_BRIEFING_ENABLED": "headlines inside the morning / evening "
                             "briefings, no thread of its own",
    "NEWS_BRIEFING_SUMMARIZE": "how the briefing headlines are worded",
    "WEATHER_BRIEFING_ENABLED": "the weather skill's master switch; its "
                                "watcher is the WEATHER_BRIEFING_PROACTIVE "
                                "row",
    "KINECT_PRESENCE_ENABLED": "room-presence sensing; its behaviours are "
                               "the PRESENCE_STANDBY / _WAKE rows",
    "WAKE_WORD_AUTOSTART": "starts the wake-word detector (input)",
    "ITUNES_AUTO_LAUNCH": "opens iTunes only when a music command needs it",
    "HOLOGRAPHIC_OVERLAY_AUTO_LAUNCH": _RETIRED,
    "HOLO_WORKSHOP_AUTO_ON_THINK": _RETIRED,
    "WORKSHOP_HUD_AUTO_LAUNCH": _RETIRED,
    "WORKSHOP_PRINT_MONITOR_AUTO_LAUNCH": _RETIRED,
    "BAMBU_OVERLAY_AUTO_WHILE_PRINTING": _RETIRED,
    "BAMBU_CAMERA_AUTO_WHILE_PRINTING": _RETIRED,
}


def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _defined(src, key):
    return re.search(r"^%s\s*=" % re.escape(key), src, re.M) is not None


def _page_keys():
    return {f["key"] for f in wi.PROACTIVE_FEATURES if f["key"]}


class CompletenessTests(unittest.TestCase):
    def setUp(self):
        self.config_src = _read("core/config.py")
        self.bc_src = _read("bobert_companion.py")

    def test_every_known_flag_is_on_the_page(self):
        missing = [k for k in KNOWN_CONFIG_KEYS + KNOWN_MONOLITH_KEYS
                   if k not in _page_keys()]
        self.assertEqual(missing, [], "proactive flags missing from the page")

    def test_the_known_flags_are_where_the_list_says(self):
        for k in KNOWN_CONFIG_KEYS:
            self.assertTrue(_defined(self.config_src, k),
                            f"{k} is not a core/config.py constant")
        for k in KNOWN_MONOLITH_KEYS:
            self.assertTrue(_defined(self.bc_src, k),
                            f"{k} is not a bobert_companion.py constant")
            self.assertFalse(_defined(self.config_src, k),
                             f"{k} moved to core/config.py: make its row "
                             f"'restart'")

    def test_the_grep_finds_nothing_unlisted(self):
        """A new proactive-looking on/off flag fails here until it is added
        to PROACTIVE_FEATURES (or to NOT_PROACTIVE, with the reason)."""
        found = set()
        for src in (self.config_src, self.bc_src):
            found |= {m.group(1) for m in _BOOL_FLAG.finditer(src)
                      if _PROACTIVE_NAME.search(m.group(1))}
        self.assertGreater(len(found), 30)              # the grep still works
        unlisted = sorted(found - _page_keys() - set(NOT_PROACTIVE))
        self.assertEqual(unlisted, [],
                         "add these to web_interface.PROACTIVE_FEATURES, or "
                         "to NOT_PROACTIVE with the reason they are not one")
        stale = sorted(set(NOT_PROACTIVE) - found)
        self.assertEqual(stale, [], "NOT_PROACTIVE names flags that are gone")

    def test_banter_has_no_switch_anywhere(self):
        self.assertIn('getattr(bc, "BANTER_ENABLED"', _read("skills/banter.py"))
        self.assertFalse(_defined(self.config_src, "BANTER_ENABLED"))
        self.assertFalse(_defined(self.bc_src, "BANTER_ENABLED"))


class RowTruthTests(unittest.TestCase):
    def setUp(self):
        self.config_src = _read("core/config.py")
        self.bc_src = _read("bobert_companion.py")

    def test_every_row_is_complete(self):
        keys = [f["key"] for f in wi.PROACTIVE_FEATURES if f["key"]]
        self.assertEqual(len(keys), len(set(keys)), "a key is listed twice")
        for f in wi.PROACTIVE_FEATURES:
            self.assertIn(f["applies"], ("restart", "live", "code", "always"))
            for field in ("name", "what", "how"):
                self.assertTrue(f[field].strip(), (f["name"], field))

    def test_applies_matches_where_the_key_lives(self):
        for f in wi.PROACTIVE_FEATURES:
            key, applies = f["key"], f["applies"]
            if applies in ("restart", "live"):
                self.assertTrue(_defined(self.config_src, key), key)
            elif applies == "code":
                self.assertTrue(_defined(self.bc_src, key), key)
                self.assertFalse(_defined(self.config_src, key), key)
            else:
                self.assertFalse(_defined(self.config_src, key), key)

    def test_no_row_claims_a_live_save(self):
        """A save writes user_settings.json only, and nothing re-reads it
        while JARVIS runs (_write_settings' RESTART CAVEAT)."""
        self.assertEqual(
            [f["key"] for f in wi.PROACTIVE_FEATURES if f["applies"] == "live"],
            [])

    def test_switchable_rows_are_bool_schema_rows(self):
        settable = [f["key"] for f in wi.PROACTIVE_FEATURES
                    if wi._proactive_settable(f, sw.SCHEMA)]
        self.assertEqual(len(settable), 17)
        for key in settable:
            self.assertEqual(sw.SCHEMA[key]["type"], "bool", key)
            self.assertIn(key, sw.persisted_keys())
        for key in ("CHAPPIE_ENABLED", "ROBOT_ENABLED", "PROACTIVE_ENABLED",
                    "BANTER_ENABLED"):
            self.assertNotIn(key, settable)


class _FakeRuntime(wi.NoRuntime):
    live = True

    def __init__(self, flags):
        self._flags = flags

    def flag(self, name):
        return self._flags.get(name)


class ProactiveRouteTests(_ServerBase):
    def server_extra(self):
        return {"runtime": _FakeRuntime({"PROACTIVE_ENABLED": True,
                                         "TEAMS_NUDGE_ENABLED": False})}

    def test_the_page_lists_every_row(self):
        code, d = _get(self.base + "/api/proactive")
        self.assertEqual(code, 200)
        self.assertEqual(d["count"], len(wi.PROACTIVE_FEATURES))
        rows = {r["key"]: r for r in d["features"] if r["key"]}
        self.assertEqual(rows["PROACTIVE_ENABLED"]["value"], True)
        self.assertEqual(rows["PROACTIVE_ENABLED"]["applies"], "code")
        self.assertFalse(rows["PROACTIVE_ENABLED"]["settable"])
        self.assertEqual(rows["TEAMS_NUDGE_ENABLED"]["value"], False)
        self.assertTrue(rows["TEAMS_NUDGE_ENABLED"]["settable"])
        self.assertEqual(rows["TEAMS_NUDGE_ENABLED"]["applies"], "restart")
        self.assertEqual(rows["BANTER_ENABLED"]["value"], True)
        # A bare web process cannot know a monolith constant: unknown.
        self.assertIsNone(rows["ANTICIPATION_ENABLED"]["value"])

    def test_the_switch_saves_through_the_settings_helper(self):
        calls = []

        def fake_write(updates, path):
            calls.append((dict(updates), path))
            return dict(updates)

        with mock.patch.object(wi, "_write_settings", side_effect=fake_write):
            code, d = _post(self.base + "/api/proactive",
                            {"key": "TEAMS_NUDGE_ENABLED", "on": True})
        self.assertEqual(code, 200, d)
        self.assertEqual(calls, [({"TEAMS_NUDGE_ENABLED": True},
                                  self.user_settings_path)])
        self.assertEqual(d["applied"], {"TEAMS_NUDGE_ENABLED": True})
        self.assertEqual(d["applies"], "restart")
        self.assertFalse(os.path.exists(self.user_settings_path))

    def test_refusals_write_nothing(self):
        cases = (
            ({"key": "NOT_A_FEATURE", "on": True}, 404),
            ({"key": "WEB_INTERFACE_TOKEN", "on": True}, 404),
            ({"key": "TEAMS_NUDGE_ENABLED", "on": "yes"}, 400),
            ({"key": "TEAMS_NUDGE_ENABLED"}, 400),
            ({"key": "PROACTIVE_ENABLED", "on": False}, 400),   # code row
            ({"key": "CHAPPIE_ENABLED", "on": True}, 400),      # not in schema
            ({"key": "BANTER_ENABLED", "on": False}, 400),      # no switch
            ({"key": "", "on": False}, 404),
        )
        with mock.patch.object(wi, "_write_settings") as write:
            for body, want in cases:
                code, d = _post(self.base + "/api/proactive", body)
                self.assertEqual(code, want, (body, d))
            write.assert_not_called()
        self.assertFalse(os.path.exists(self.user_settings_path))

    def test_a_form_post_is_refused(self):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            self.base + "/api/proactive", method="POST",
            data=b"key=TEAMS_NUDGE_ENABLED&on=true",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 415)

    def test_a_real_save_lands_in_the_settings_file(self):
        code, d = _post(self.base + "/api/proactive",
                        {"key": "TEAMS_NUDGE_ENABLED", "on": True})
        self.assertEqual(code, 200, d)
        with open(self.user_settings_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"TEAMS_NUDGE_ENABLED": True})
        code, d = _get(self.base + "/api/proactive")
        row = next(r for r in d["features"]
                   if r["key"] == "TEAMS_NUDGE_ENABLED")
        self.assertEqual((row["value"], row["saved"], row["pending_restart"]),
                         (False, True, True))


class ProactiveTokenTests(_ServerBase):
    token = "s3cr3t"

    def test_the_token_is_required(self):
        code, _ = _get_raw(self.base + "/api/proactive")
        self.assertEqual(code, 401)
        code, _ = _post(self.base + "/api/proactive",
                        {"key": "TEAMS_NUDGE_ENABLED", "on": True})
        self.assertEqual(code, 401)
        self.assertFalse(os.path.exists(self.user_settings_path))
        code, _ = _get_raw(self.base + "/api/proactive",
                           headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)


class ProactivePageTests(unittest.TestCase):
    def setUp(self):
        self.html = wi._dashboard_html("")

    def test_the_view_is_in_the_navigation(self):
        self.assertIn('id="navProactive"', self.html)
        self.assertIn('id="viewProactive"', self.html)
        self.assertIn("proactive:{nav:navProactive, view:viewProactive}",
                      self.html)
        self.assertIn("postJSON('/api/proactive'", self.html)

    def test_rendered_values_never_reach_innerHTML(self):
        for fn in ("renderProactive", "loadProactive", "saveProactive"):
            body = _js_fn(self.html, fn)
            for m in re.finditer(r"innerHTML\s*=\s*([^;]+);", body):
                self.assertRegex(m.group(1).strip(), r"^'[^'+]*'$",
                                 f"{fn} builds innerHTML from data")


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class ProactiveRenderXssTests(unittest.TestCase):
    HOSTILE = "<script>alert(1)</script>"

    def test_hostile_text_is_text(self):
        html = wi._dashboard_html("")
        js = "\n".join(_js_fn(html, f) + "\n}" for f in
                       ("onOff", "renderProactive"))
        consts = re.search(r"^const PRO_APPLIES = [^;]+;", html,
                           re.M | re.S).group(0)
        h = json.dumps(self.HOSTILE)
        script = DOM_SHIM + consts + "\n" + js + """
const proList = document.createElement('div');
renderProactive({features: [{key: %(h)s, name: %(h)s, what: %(h)s,
  how: %(h)s, applies: %(h)s, settable: true, value: true}]});
console.log(JSON.stringify({html: HTML_SETS, text: textOf(proList)}));
""" % {"h": h}
        out = subprocess.run([shutil.which("node"), "-e", script],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(set(res["html"]) - {""}, set())
        self.assertIn(self.HOSTILE, res["text"])


if __name__ == "__main__":
    unittest.main()
