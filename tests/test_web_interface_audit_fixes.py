"""Regression tests for the 2026-09-30 web-dashboard audit (tools/web_interface).

One class per finding, in the audit's order:
  1  the reply capture returned the LEAD-IN ("[ACTION: get_time] One moment,
     sir.") instead of the answer;
  2  CPU read 0% forever (psutil's per-thread cpu_percent baseline vs a new
     thread per request);
  3  the phone layout (390 px) was 583 px wide with the tabs off-screen;
  4  the Camera tab was hard-coded left/right/kinect, blind to the camera gate,
     hid its reason detail behind hover, and re-armed dead tiles 4x a second;
  5  the Actions tab read a stale doc and "Send" typed the bare name into the
     LLM's command channel;
  6  no standby/mute state or controls, and a typed command was silently
     dropped in standby;
  7  the "model" chip showed the last [intent:] tag and VRAM summed both GPUs;
  8  the log view rebuilt itself every second (killing any selection) with no
     noise filter or search;
  9  the Voice tab named the edge voice while Kokoro spoke, and offered voice
     cloning with no VRAM warning;
 10  security headers, a constant-time token compare, and a lock around the
     command-queue read-modify-write;
 11  accessibility / remembered tab / hidden-tab polling.

Headless-CI safe: every server binds 127.0.0.1:0 in a temp dir, every JARVIS
input (camera roster, camera gate, live ACTIONS) comes from a FAKE runtime, and
nothing here writes to the live tray or inject files.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from collections import namedtuple
from unittest import mock

from tools import web_interface as wi
from core import camera_tiles as ct
from tests.test_web_interface import (_ServerBase, _get, _get_raw, _js_fn,
                                      _no_live_gpu, _post, _urlopen_retry)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)


def _page(base):
    code, html = _get_raw(base + "/")
    assert code == 200
    return html


def _css_rule(html, selector):
    """The body of the FIRST CSS rule whose selector is exactly ``selector``."""
    m = re.search(r"(?m)^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", html)
    return m.group(1) if m else ""


# getattr: this module must IMPORT against the pre-fix server too, so the
# fail-on-old-code proof reports each test instead of one import error.
class FakeRuntime(getattr(wi, "NoRuntime", object)):
    """A running JARVIS, as far as the web server can tell."""

    live = True

    def __init__(self, cameras=None, kinect=True, snapshot=None, actions=None,
                 speak=None):
        self._cameras = cameras
        self._kinect = kinect
        self._snapshot = snapshot
        self._actions = actions
        self._speak = speak

    def cameras(self):
        return self._cameras

    def kinect_enabled(self):
        return self._kinect

    def gate_key(self, cam):
        return "name:" + str(cam.get("name", "")).lower()

    def gate_snapshot(self):
        return self._snapshot

    def actions(self):
        return self._actions

    def speak_sets(self):
        return self._speak


# ═══════════════════════════════════════════════════════════════════════════
# 1  the reply capture
# ═══════════════════════════════════════════════════════════════════════════
ACTION_TURN = [
    "[22:07:44]   You:    what time is it",
    "[22:07:44]   Thinking…",
    "[22:07:45]   [local-llm] served via some-model pe=13860/368 ev=13",
    "[22:07:45]   JARVIS: [ACTION: get_time] One moment, sir.",
    "[22:07:45]   [action] get_time: current time is 10:07 PM",
    "[22:07:50]   Reading results (depth 1)…",
    "[22:07:51]   JARVIS: It is 10:07 PM, sir.",
    "[22:07:55]   [turn-timing] kind=inject outcome=ok you=1 end=10748",
    "[22:07:55] Listening…",
]


class ReplyCaptureTests(unittest.TestCase):

    def test_multi_line_action_turn_returns_the_answer_not_the_lead_in(self):
        res = wi.parse_turn_lines(ACTION_TURN)
        self.assertEqual(res["reply"], "It is 10:07 PM, sir.")
        self.assertTrue(res["ended"])
        self.assertNotIn("[ACTION", res["reply"])
        self.assertEqual(res["actions"][0]["name"], "get_time")
        # the transcript keeps every line, cleaned of stamps and tags
        self.assertIn("JARVIS: One moment, sir.", res["lines"])
        self.assertFalse(any(re.match(r"\[\d\d:", ln) for ln in res["lines"]))

    def test_a_verbatim_action_turn_answers_with_the_action_result(self):
        res = wi.parse_turn_lines([
            "[22:08:34]   JARVIS: [ACTION: device_report] One moment, sir.",
            "[22:08:35]   [action] device_report: The desk device is charging, sir.",
            "[22:08:35]   [answer-first] dropped lead-in (3 words)",
            "[22:08:44]   [turn-timing] kind=inject outcome=ok",
        ])
        self.assertEqual(res["reply"], "The desk device is charging, sir.")

    def test_a_fallback_line_replaces_the_models_line(self):
        res = wi.parse_turn_lines([
            "[10:00:01]   JARVIS: I'm afraid I've run out of material, sir.",
            "[10:00:01]   [joke-fallback] reply refused a joke request",
            "[10:00:01]   JARVIS (spoken): Why did the robot cross the road, sir?",
            "[10:00:04]   [turn-timing] kind=inject outcome=ok",
        ])
        self.assertEqual(res["reply"], "Why did the robot cross the road, sir?")
        self.assertTrue(res["spoken"])
        self.assertNotIn("run out of material", res["reply"])

    def test_intent_tags_and_stamps_are_stripped(self):
        self.assertEqual(wi.clean_reply_text(
            "[intent:chat] Good evening, sir. [ACTION: lights_on]"),
            "Good evening, sir.")
        res = wi.parse_turn_lines(["[01:02:03]   JARVIS: [intent:greet] Hello, sir.",
                                   "Listening…"])
        self.assertEqual(res["reply"], "Hello, sir.")

    def test_standby_drop_is_reported(self):
        res = wi.parse_turn_lines(["[10:00:00]   [standby] ignored: 'what time is it'"])
        self.assertTrue(res["standby"])
        self.assertTrue(res["ended"])

    def _log(self, d):
        log_dir = os.path.join(d, "logs")
        os.makedirs(log_dir)
        lg = os.path.join(log_dir, "session_2026-09-30_10-00-00.log")
        with open(lg, "w", encoding="utf-8") as f:
            f.write("[10:00:00] boot\n")
        return log_dir, lg

    def _append_later(self, lg, chunks):
        def run():
            for delay, text in chunks:
                time.sleep(delay)
                with open(lg, "a", encoding="utf-8") as f:
                    f.write(text)
        threading.Thread(target=run, daemon=True).start()

    def test_wait_for_reply_waits_for_the_turns_own_end_marker(self):
        """THE P0. The answer lands 1.6 s after the lead-in; the old capture
        returned ~1 s after the lead-in and never saw it."""
        with tempfile.TemporaryDirectory() as d:
            log_dir, lg = self._log(d)
            self._append_later(lg, [
                (0.3, "[10:00:01]   [inject] what time is it\n"
                      "[10:00:01]   JARVIS: [ACTION: get_time] One moment, sir.\n"
                      "[10:00:01]   [action] get_time: current time is 10:07 PM\n"),
                (1.6, "[10:00:03]   JARVIS: It is 10:07 PM, sir.\n"),
                (0.4, "[10:00:04]   [turn-timing] kind=inject outcome=ok\n"),
            ])
            t0 = time.monotonic()
            res = wi.wait_for_reply("what time is it", log_dir, timeout=15.0)
            took = time.monotonic() - t0
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["reply"], "It is 10:07 PM, sir.")
        self.assertLess(took, 8.0, "the end marker should end the wait, not the timeout")

    def test_wait_for_reply_anchors_on_a_web_tagged_inject_line(self):
        """2026-10-02: the loop logs a dashboard command as
        "[inject] (web) <text>"; the reply capture still finds its turn."""
        with tempfile.TemporaryDirectory() as d:
            log_dir, lg = self._log(d)
            self._append_later(lg, [
                (0.3, "[10:00:01]   [inject] (web) what time is it\n"
                      "[10:00:01]   You:    what time is it\n"
                      "[10:00:02]   JARVIS: It is ten, sir.\n"
                      "[10:00:03]   [turn-timing] kind=inject outcome=ok\n"),
            ])
            res = wi.wait_for_reply("what time is it", log_dir, timeout=15.0)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["reply"], "It is ten, sir.")

    def test_wait_for_reply_ignores_another_commands_output(self):
        with tempfile.TemporaryDirectory() as d:
            log_dir, lg = self._log(d)
            self._append_later(lg, [
                (0.3, "[10:00:01]   [inject] something else\n"
                      "[10:00:01]   JARVIS: Not this one, sir.\n"
                      "[10:00:02]   [turn-timing] kind=inject outcome=ok\n"
                      "[10:00:03]   [inject] what day is it\n"
                      "[10:00:03]   JARVIS: It is Wednesday, sir.\n"
                      "[10:00:03]   [inject] a third command\n"
                      "[10:00:03]   JARVIS: Nor this one.\n"),
            ])
            res = wi.wait_for_reply("what day is it", log_dir, timeout=6.0)
        self.assertEqual(res["reply"], "It is Wednesday, sir.")

    def test_wait_for_reply_reports_a_standby_drop(self):
        with tempfile.TemporaryDirectory() as d:
            log_dir, lg = self._log(d)
            self._append_later(lg, [
                (0.3, "[10:00:01]   [inject] (standby) what time is it\n"
                      "[10:00:01]   [standby] ignored: 'what time is it'\n"),
            ])
            res = wi.wait_for_reply("what time is it", log_dir, timeout=6.0)
        self.assertEqual(res["status"], "standby")


class SayUsesTheCapturedAnswerTests(_ServerBase):
    reply_reader = staticmethod(lambda text, log_dir, timeout: {
        "status": "ok", "lines": ["JARVIS: One moment, sir.", "JARVIS: Noon, sir."],
        "reply": "Noon, sir.", "actions": [{"name": "get_time", "result": "12:00"}]})

    def test_say_returns_the_reply_field(self):
        code, data = _post(self.base + "/api/say", {"text": "what time is it"})
        self.assertEqual(code, 200)
        self.assertEqual(data["reply"], "Noon, sir.")
        self.assertEqual(data["actions"][0]["name"], "get_time")


class SayStandbyStatusTests(_ServerBase):
    reply_reader = staticmethod(lambda text, log_dir, timeout: {
        "status": "standby", "lines": [], "reply": ""})

    def test_standby_status_reaches_the_page(self):
        code, data = _post(self.base + "/api/say", {"text": "hello"})
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "standby")
        self.assertIn("d.status==='standby'", _page(self.base))


# ═══════════════════════════════════════════════════════════════════════════
# 2  CPU %
# ═══════════════════════════════════════════════════════════════════════════
_CT = namedtuple("scputimes", "user system idle interrupt dpc")


def _fake_psutil():
    """cpu_times() climbs 1 s busy + 1 s idle per call (a steady 50 %), and
    cpu_percent() answers 0.0 - what the real psutil returns on the FIRST call
    of every thread, i.e. on every request of a ThreadingHTTPServer."""
    m = types.ModuleType("psutil")
    n = {"i": 0}

    def cpu_times():
        n["i"] += 1
        i = n["i"]
        return _CT(float(i), 0.0, float(i), 0.0, 0.0)

    m.cpu_times = cpu_times
    m.cpu_percent = lambda interval=None: 0.0
    vm = types.SimpleNamespace(total=16e9, available=8e9)
    m.virtual_memory = lambda: vm
    m.disk_partitions = lambda all=False: []
    return m


class CpuPercentTests(unittest.TestCase):

    def setUp(self):
        _no_live_gpu(self)       # _system_info's live GPU probes (hermetic guard)
        self._saved = dict(wi._cpu_state)
        wi._cpu_state.update({"times": None, "at": 0.0, "pct": None})
        self.addCleanup(lambda: (wi._cpu_state.clear(), wi._cpu_state.update(self._saved)))

    def _system_from_a_new_thread(self, d):
        box = {}

        def run():
            box["s"] = wi._system_info(os.path.join(d, "nope.json"),
                                       os.path.join(d, "logs"))
        t = threading.Thread(target=run)
        t.start()
        t.join(10)
        return box["s"]

    def test_cpu_is_measured_across_request_threads(self):
        fake = _fake_psutil()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(sys.modules, {"psutil": fake}), \
                mock.patch.object(wi, "_nvidia_smi_gpus", return_value=[]):
            first = self._system_from_a_new_thread(d)
            wi._cpu_state["at"] = 0.0          # let the next poll measure again
            second = self._system_from_a_new_thread(d)
        self.assertEqual(first["cpu_pct"], 50.0)
        self.assertEqual(second["cpu_pct"], 50.0)

    def test_close_polls_reuse_the_last_figure(self):
        fake = _fake_psutil()
        self.assertEqual(wi._cpu_percent(fake, now=100.0), 50.0)
        calls = fake.cpu_times
        fake.cpu_times = lambda: (_ for _ in ()).throw(AssertionError("measured again"))
        self.assertEqual(wi._cpu_percent(fake, now=100.2), 50.0)
        fake.cpu_times = calls

    def test_busy_total_ignores_idle_and_iowait(self):
        t = namedtuple("t", "user system idle iowait guest")(3.0, 1.0, 5.0, 1.0, 2.0)
        self.assertEqual(wi._cpu_busy_total(t), (4.0, 10.0))


# ═══════════════════════════════════════════════════════════════════════════
# 3  the phone layout
# ═══════════════════════════════════════════════════════════════════════════
class PhoneLayoutTests(_ServerBase):

    def test_the_nav_scrolls_inside_its_own_row(self):
        html = _page(self.base)
        nav = _css_rule(html, "nav.views")
        for decl in ("overflow-x:auto", "min-width:0", "flex-wrap:nowrap",
                     "max-width:100%"):
            self.assertIn(decl, nav, "nav.views lost %r - the tabs run off a "
                                     "phone screen again" % decl)
        self.assertIn("flex:0 0 auto", _css_rule(html, "nav.views button"))

    def test_the_phone_breakpoint_gives_the_nav_a_full_row(self):
        html = _page(self.base)
        m = re.search(r"@media \(max-width: 640px\) \{(.*?)\n  \}", html, re.S)
        self.assertIsNotNone(m, "no phone breakpoint")
        self.assertRegex(m.group(1), r"nav\.views \{[^}]*flex:1 1 100%")

    def test_nothing_forces_a_wide_minimum(self):
        html = _page(self.base)
        css = html.split("<style>")[1].split("</style>")[0]
        for px in re.findall(r"min-width:(\d+)px", css):
            self.assertLessEqual(int(px), 180, "a %spx min-width cannot fit "
                                               "a 390px phone with gutters" % px)


# ═══════════════════════════════════════════════════════════════════════════
# 4  the Camera tab
# ═══════════════════════════════════════════════════════════════════════════
_RIGHT_ONLY = [{"index": 0, "label": "Right webcam (top of right monitor)",
                "name": "cam b", "look_x": 0.85}]
_BOTH = [{"index": 2, "label": "Left webcam", "name": "cam a", "look_x": 0.5},
         {"index": 0, "label": "Right webcam", "name": "cam b", "look_x": 0.85}]


class CameraTilesHelperTests(unittest.TestCase):

    def test_tiles_follow_the_roster(self):
        tiles = ct.tiles_from_config(_RIGHT_ONLY, True, gate_key=lambda c: "g")
        self.assertEqual([t["cam"] for t in tiles], ["right", "kinect"])
        self.assertEqual(tiles[0]["gate_key"], "g")
        self.assertEqual(tiles[1]["gate_key"], "kinect")

    def test_kinect_off_means_no_kinect_tile(self):
        self.assertEqual([t["cam"] for t in ct.tiles_from_config(_BOTH, False)],
                         ["left", "right"])

    def test_a_kinect_camera_entry_is_the_kinect_tile(self):
        tiles = ct.tiles_from_config([{"type": "kinect", "index": 1}], False)
        self.assertEqual([t["cam"] for t in tiles], ["kinect"])

    def test_first_entry_wins_a_side_and_junk_is_skipped(self):
        cams = [None, "x", {"label": "Left one"}, {"label": "left two"}]
        self.assertEqual([t["cam"] for t in ct.tiles_from_config(cams, False)],
                         ["left"])

    def test_side_rule(self):
        self.assertEqual(ct.percam_side({"label": "Right cam"}), "right")
        self.assertEqual(ct.percam_side({"look_x": 0.5}), "left")
        self.assertEqual(ct.percam_side({"look_x": 0.9}), "right")
        self.assertEqual(ct.percam_side({"look_x": "bogus"}), "left")

    def test_gate_summary_states(self):
        base = {"devices": {"k": {}}, "quarantined": {}}
        self.assertIsNone(ct.gate_summary(base, "k"))
        self.assertIsNone(ct.gate_summary(None, "k"))
        cases = {
            "quarantined": {"quarantined": {"k": {"label": "x", "why": "y"}}},
            "usb_storm": {"storm_active": True, "storm_remaining_s": 300.0},
            "slow_retry": {"devices": {"k": {"hold_s": 1500.0, "slow_retry_s": 1800.0}}},
            "absent": {"devices": {"k": {"absent": True}}},
            "locked": {"devices": {"k": {"locked_by": ["MeetingApp"]}}},
            "backoff": {"devices": {"k": {"hold_s": 45.0}}},
            "opening": {"devices": {"k": {"in_flight": "producer"}}},
        }
        for state, extra in cases.items():
            snap = dict(base)
            snap.update(extra)
            g = ct.gate_summary(snap, "k")
            self.assertEqual(g["state"], state)
            self.assertTrue(g["message"].endswith("."), state)
        slow = ct.gate_summary(cases["slow_retry"], "k")
        self.assertEqual(slow["retry_in_s"], 1500.0)
        self.assertIn("25 min", slow["message"])
        self.assertIn("30 min", slow["message"])

    def test_fmt_wait(self):
        self.assertEqual(ct.fmt_wait(45), "45 s")
        self.assertEqual(ct.fmt_wait(300), "5 min")
        self.assertEqual(ct.fmt_wait(3900), "1 h 5 min")

    def test_one_key_vocabulary_for_writer_and_dashboard(self):
        self.assertIs(wi._CAMERA_PREVIEW_CAMS, ct.PREVIEW_KEYS)
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8", errors="replace") as f:
            mono = f.read()
        self.assertIn("_HUD_PERCAM_PREVIEW_KEYS = _camera_tiles.PREVIEW_KEYS", mono)
        self.assertIn("return _camera_tiles.percam_side(cam)", mono)
        for src in (mono, open(wi.__file__, encoding="utf-8").read()):
            self.assertNotIn('("left", "right", "kinect")', src,
                             "a second copy of the preview-key tuple is back")


class _CamRuntimeBase(_ServerBase):
    runtime = None

    def server_extra(self):
        return {"runtime": self.runtime} if self.runtime is not None else {}

    def setUp(self):
        super().setUp()
        # No powershell, ever (the Kinect rung's enumeration probe).
        saved = (wi._probe_kinect_devices, dict(wi._kinect_enum_cache), wi._kinect_health)
        wi._probe_kinect_devices = lambda: (True, ("sensor",), "stub")
        wi._kinect_enum_cache.update({"ts": 0.0, "present": None, "names": (),
                                      "how": "never run"})
        wi._kinect_health = lambda: None

        def restore():
            wi._probe_kinect_devices, cache, wi._kinect_health = saved
            wi._kinect_enum_cache.clear()
            wi._kinect_enum_cache.update(cache)
        self.addCleanup(restore)


class CameraTilesRouteTests(_CamRuntimeBase):
    runtime = FakeRuntime(
        cameras=_RIGHT_ONLY, kinect=True,
        snapshot={"storm_active": False, "quarantined": {},
                  "devices": {"kinect": {"hold_s": 1500.0, "slow_retry_s": 1800.0},
                              "name:cam b": {"hold_s": 0.0}}})

    def test_tiles_come_from_the_live_roster_with_gate_verdicts(self):
        code, d = _get(self.base + "/api/camera-tiles")
        self.assertEqual(code, 200)
        self.assertEqual(d["source"], "live")
        self.assertEqual([t["cam"] for t in d["tiles"]], ["right", "kinect"])
        self.assertIsNone(d["tiles"][0]["gate"])
        self.assertEqual(d["tiles"][1]["gate"]["state"], "slow_retry")
        self.assertEqual(d["tiles"][1]["gate"]["retry_in_s"], 1500.0)

    def test_an_unlisted_camera_says_not_configured(self):
        code, r = _get(self.base + "/api/camera-reason?cam=left")
        self.assertEqual(code, 200)
        self.assertEqual(r["state"], "not_configured")
        self.assertIn("camera list", r["message"])

    def test_the_gate_explains_a_held_kinect_with_a_countdown(self):
        code, r = _get(self.base + "/api/camera-reason?cam=kinect")
        self.assertEqual(r["state"], "held_by_gate")
        self.assertEqual(r["gate_state"], "slow_retry")
        self.assertEqual(r["retry_in_s"], 1500.0)
        self.assertIn("next try in 25 min", r["detail"])


class CameraGateWebcamTests(_CamRuntimeBase):
    runtime = FakeRuntime(cameras=_BOTH, kinect=False,
                          snapshot={"devices": {"name:cam a": {"hold_s": 120.0}}})

    def test_a_webcam_in_backoff_is_held_by_gate(self):
        code, r = _get(self.base + "/api/camera-reason?cam=left")
        self.assertEqual(r["state"], "held_by_gate")
        self.assertEqual(r["gate_state"], "backoff")
        self.assertIn("retrying in 2 min", r["detail"])

    def test_a_webcam_the_gate_is_not_holding_is_just_off(self):
        code, r = _get(self.base + "/api/camera-reason?cam=right")
        self.assertEqual(r["state"], "webcam_off")

    def test_kinect_switched_off_is_disabled_not_a_fault(self):
        code, r = _get(self.base + "/api/camera-reason?cam=kinect")
        self.assertEqual(r["state"], "disabled")


class CameraTilesDefaultTests(_CamRuntimeBase):
    runtime = FakeRuntime(cameras=None)      # nothing known about a roster

    def test_no_jarvis_to_ask_means_the_default_three(self):
        code, d = _get(self.base + "/api/camera-tiles")
        self.assertEqual(d["source"], "default")
        self.assertEqual([t["cam"] for t in d["tiles"]], ["left", "right", "kinect"])

    def test_the_live_runtime_is_inert_outside_the_booted_process(self):
        # tests run under a unittest __main__, never the aliased monolith
        self.assertFalse(wi.LiveRuntime().live)
        self.assertIsNone(wi.LiveRuntime().cameras())


class CameraPageContractTests(_ServerBase):

    def test_dead_tiles_back_off_exponentially(self):
        html = _page(self.base)
        self.assertIn("const CAM_BACKOFF_MAX_MS", html)
        back = _js_fn(html, "camBackoffMs")
        self.assertIn("Math.pow(2", back)
        self.assertIn("CAM_BACKOFF_MAX_MS", back)
        self.assertIn("camMayTry(img)", _js_fn(html, "startCameraStreams"))
        self.assertIn("camMayTry(img)", _js_fn(html, "pollTile"))
        self.assertIn("nextTry", _js_fn(html, "camMayTry"))
        wire = _js_fn(html, "wireCameraTile")
        self.assertIn("camBackoffMs(errs)", wire)

    def test_the_reason_detail_is_visible_not_only_a_hover_title(self):
        html = _page(self.base)
        self.assertIn(".camtile .camdetail", html)
        self.assertIn("camDetail(off, detail)", _js_fn(html, "camSay"))
        explain = _js_fn(html, "explainTile")
        self.assertIn("camSay(off, j.message, j.detail", explain)
        self.assertIn("'not_configured'", explain)
        self.assertIn("j.retry_in_s", explain)


# ═══════════════════════════════════════════════════════════════════════════
# 5  the Actions tab
# ═══════════════════════════════════════════════════════════════════════════
class _Recorder:
    def __init__(self, result="ok"):
        self.calls = []
        self.result = result

    def __call__(self, arg=""):
        self.calls.append(arg)
        return self.result


class ActionConfirmRulesTests(unittest.TestCase):

    def test_side_effect_families_need_confirmation(self):
        for name in ("shutdown", "restart", "restart_jarvis", "send_vip_reply",
                     "send_email", "archive_email", "forget_last_hour",
                     "start_overnight_upgrade", "text_mom", "reset_memory",
                     "delete_timer", "run_shell", "type", "hotkey", "click",
                     "decline_call", "answer_call", "web_interface_off"):
            self.assertTrue(wi.action_confirm_reason(name), name)

    def test_reads_run_on_one_click(self):
        for name in ("get_time", "system_status", "camera_status",
                     "version_info", "weather", "show_recent_facts"):
            self.assertEqual(wi.action_confirm_reason(name), "", name)


class LiveActionsRouteTests(_ServerBase):

    def server_extra(self):
        self.get_time = _Recorder("It is noon, sir.")
        self.send_x = _Recorder("sent")
        self.boom = lambda arg="": 1 / 0
        acts = {"get_time": self.get_time, "send_x": self.send_x,
                "boom": self.boom, "restart": _Recorder()}
        return {"runtime": FakeRuntime(actions=acts,
                                       speak=({"get_time"}, set(), set()))}

    def _act(self, body, ctype="application/json"):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + "/api/action", data=data,
                                     headers={"Content-Type": ctype}, method="POST")
        try:
            with _urlopen_retry(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def test_the_list_is_the_live_registry(self):
        code, d = _get(self.base + "/api/actions")
        self.assertEqual(code, 200)
        self.assertEqual(d["source"], "live")
        rows = {a["name"]: a for a in d["actions"]}
        self.assertEqual(set(rows), {"get_time", "send_x", "boom", "restart"})
        self.assertEqual(rows["get_time"]["spoken"], "VERBATIM")
        self.assertFalse(rows["get_time"]["confirm"])
        self.assertTrue(rows["send_x"]["confirm"])
        self.assertTrue(d["confirm_rules"])

    def test_run_by_name_calls_the_handler_directly(self):
        code, d = self._act({"name": "get_time", "arg": ""})
        self.assertEqual(code, 200)
        self.assertEqual(d["status"], "done")
        self.assertEqual(d["result"], "It is noon, sir.")
        self.assertEqual(self.get_time.calls, [""])
        # never through the command channel
        self.assertFalse(os.path.exists(self.inject_path))

    def test_a_side_effect_action_needs_confirm(self):
        code, d = self._act({"name": "send_x"})
        self.assertEqual(code, 409)
        self.assertTrue(d["confirm_required"])
        self.assertEqual(self.send_x.calls, [])
        code, d = self._act({"name": "send_x", "confirm": True, "arg": "hi"})
        self.assertEqual(code, 200)
        self.assertEqual(self.send_x.calls, ["hi"])

    def test_unknown_name_and_bad_input(self):
        self.assertEqual(self._act({"name": "nope"})[0], 404)
        self.assertEqual(self._act({"name": ""})[0], 400)
        self.assertEqual(self._act({"name": "get_time", "arg": 5})[0], 400)
        self.assertEqual(self._act({"name": "get_time"}, ctype="text/plain")[0], 415)

    def test_a_double_click_is_refused(self):
        self.assertEqual(self._act({"name": "get_time"})[0], 200)
        code, d = self._act({"name": "get_time"})
        self.assertEqual(code, 429)

    def test_a_raising_action_reports_the_error(self):
        code, d = self._act({"name": "boom"})
        self.assertEqual(code, 200)
        self.assertEqual(d["status"], "error")
        self.assertIn("ZeroDivisionError", d["error"])

    def test_restart_goes_through_the_tray_control_plane(self):
        code, d = self._act({"name": "restart", "confirm": True})
        self.assertEqual(code, 200)
        self.assertEqual(d["via"], "tray")
        with open(self.tray_path, encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual(items[-1]["cmd"], "restart")

    def setUp(self):
        wi._action_last_call.clear()
        super().setUp()


class NoRegistryActionTests(_ServerBase):

    def server_extra(self):
        return {"runtime": wi.NoRuntime()}

    def test_no_live_registry_is_503_and_the_list_falls_back_to_the_index(self):
        req = urllib.request.Request(
            self.base + "/api/action", data=b'{"name": "get_time"}',
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("expected 503")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 503)
        code, d = _get(self.base + "/api/actions")
        self.assertEqual(d["source"], "index")

    def test_the_page_runs_actions_by_name_not_through_the_command_box(self):
        html = _page(self.base)
        render = _js_fn(html, "renderActions")
        self.assertNotIn("sendCommand(a.name", render,
                         "Send types the bare name into the LLM's channel again")
        self.assertIn("runAction(a, arg.value, send)", render)
        run = _js_fn(html, "runAction")
        self.assertIn("'/api/action'", run)
        self.assertIn("window.confirm", run)


# ═══════════════════════════════════════════════════════════════════════════
# 6  standby + controls   /   7  the status strip
# ═══════════════════════════════════════════════════════════════════════════
class StatusFlagsTests(_ServerBase):

    def _hud(self, **kw):
        with open(self.hud_path, "w", encoding="utf-8") as f:
            json.dump(kw, f)

    def test_standby_and_mutes_reach_the_status(self):
        self._hud(state="Standby", sleep_mode=True, standby_mode=True,
                  mic_muted=True, tts_muted=True, daemons_paused=True,
                  now_doing="STANDBY", active_action="", last_transcript="hi",
                  llm_backend="some-local:7b", last_intent_tag="chat")
        code, s = _get(self.base + "/api/status")
        self.assertTrue(s["standby"])
        self.assertFalse(s["awake"])
        for k in ("mic_muted", "tts_muted", "daemons_paused"):
            self.assertTrue(s[k], k)
        self.assertEqual(s["now_doing"], "STANDBY")
        self.assertEqual(s["last_transcript"], "hi")
        # the model chip is the REAL backend, not the last intent tag
        self.assertEqual(s["model"], "some-local:7b")
        self.assertEqual(s["last_intent_tag"], "chat")
        self.assertIsInstance(s["gpus"], list)

    def test_claude_backend_label(self):
        self._hud(state="Idle", llm_backend="anthropic")
        self.assertEqual(_get(self.base + "/api/status")[1]["model"], "Claude (cloud)")

    def test_awake_by_default(self):
        self._hud(state="Idle")
        s = _get(self.base + "/api/status")[1]
        self.assertTrue(s["awake"])
        self.assertFalse(s["standby"])

    def test_the_strip_shows_the_new_fields(self):
        html = _page(self.base)
        status = _js_fn(html, "refreshStatus")
        for needle in ("chip('brain'", "s.model", "s.gpus", "chip('mic'",
                       "chip('voice out'", "s.standby", "s.active_action",
                       "chip('last heard'"):
            self.assertIn(needle, status)
        self.assertNotIn("s.routing || s.model", status)


class ControlRouteTests(_ServerBase):

    def _ctl(self, body):
        return _post(self.base + "/api/control", body)

    def test_force_wake_is_appended_to_the_tray_inbox(self):
        code, d = self._ctl({"cmd": "force_wake"})
        self.assertEqual(code, 200)
        code, d = self._ctl({"cmd": "mic_mute_toggle"})
        with open(self.tray_path, encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual([i["cmd"] for i in items], ["force_wake", "mic_mute_toggle"])
        self.assertTrue(all("ts" in i for i in items))

    def test_only_the_allowlist(self):
        self.assertEqual(self._ctl({"cmd": "trigger_overnight"})[0], 400)
        self.assertEqual(self._ctl({"cmd": "shutdown"})[0], 400)
        self.assertFalse(os.path.exists(self.tray_path))

    def test_restart_needs_confirm(self):
        code, d = self._ctl({"cmd": "restart"})
        self.assertEqual(code, 409)
        self.assertFalse(os.path.exists(self.tray_path))
        self.assertEqual(self._ctl({"cmd": "restart", "confirm": True})[0], 200)

    def test_every_web_command_is_one_the_tray_dispatcher_handles(self):
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8", errors="replace") as f:
            src = f.read()
        body = src[src.index("def _dispatch_tray_command("):]
        body = body[:body.index("\ndef ", 10)]
        for cmd in wi.TRAY_WEB_COMMANDS:
            self.assertTrue(('cmd == "%s"' % cmd) in body or ('"%s"' % cmd) in body,
                            "%s is not a branch of _dispatch_tray_command" % cmd)

    def test_the_page_guards_standby_and_confirms_restart(self):
        html = _page(self.base)
        send = _js_fn(html, "sendCommand")
        self.assertIn("LAST_STATUS.standby", send)
        self.assertIn("WAKE_WORD_RE.test(text)", send)
        self.assertIn("sendControl('force_wake')", send)
        self.assertRegex(html, r"cmd:'restart',[^\n]*danger:true")
        self.assertIn("window.confirm(confirmText)", _js_fn(html, "sendControl"))


# ═══════════════════════════════════════════════════════════════════════════
# 8  the log view
# ═══════════════════════════════════════════════════════════════════════════
class LogIncrementalTests(unittest.TestCase):

    def test_since_returns_only_new_complete_lines(self):
        with tempfile.TemporaryDirectory() as d:
            lg = os.path.join(d, "session_2026-09-30_10-00-00.log")
            with open(lg, "w", encoding="utf-8", newline="\n") as f:
                f.write("a\nb\n")
            first = wi.tail_log(d, 50)
            self.assertEqual(first["lines"], ["a", "b"])
            self.assertFalse(first["append"])
            with open(lg, "a", encoding="utf-8", newline="\n") as f:
                f.write("c\nhalf")
            nxt = wi.tail_log(d, 50, since=first["offset"], log_name=first["log"])
            self.assertTrue(nxt["append"])
            self.assertEqual(nxt["lines"], ["c"])      # "half" is not complete
            with open(lg, "a", encoding="utf-8", newline="\n") as f:
                f.write(" done\n")
            again = wi.tail_log(d, 50, since=nxt["offset"], log_name=nxt["log"])
            self.assertEqual(again["lines"], ["half done"])
            idle = wi.tail_log(d, 50, since=again["offset"], log_name=again["log"])
            self.assertEqual(idle["lines"], [])

    def test_a_rotated_log_falls_back_to_a_full_tail(self):
        with tempfile.TemporaryDirectory() as d:
            lg = os.path.join(d, "session_2026-09-30_10-00-00.log")
            with open(lg, "w", encoding="utf-8") as f:
                f.write("x\n")
            out = wi.tail_log(d, 50, since=999999, log_name=os.path.basename(lg))
            self.assertFalse(out["append"])
            out = wi.tail_log(d, 50, since=0, log_name="session_old.log")
            self.assertFalse(out["append"])
            self.assertEqual(out["lines"], ["x"])


class LogPageContractTests(_ServerBase):

    def test_route_passes_since_through(self):
        lg = os.path.join(self.log_dir, "session_2026-09-30_10-00-00.log")
        with open(lg, "w", encoding="utf-8", newline="\n") as f:
            f.write("one\n")
        code, d = _get(self.base + "/api/log/tail?lines=10")
        with open(lg, "a", encoding="utf-8", newline="\n") as f:
            f.write("two\n")
        code, d2 = _get(self.base + "/api/log/tail?lines=10&since=%d&log=%s"
                        % (d["offset"], d["log"]))
        self.assertTrue(d2["append"])
        self.assertEqual(d2["lines"], ["two"])

    def test_the_log_appends_filters_and_searches(self):
        html = _page(self.base)
        log = _js_fn(html, "refreshLog")
        self.assertIn("&since=", log)
        self.assertIn("logEl.appendChild(frag)", log)
        self.assertNotIn("logEl.innerHTML = frag", log,
                         "the whole log is rebuilt every poll again")
        self.assertIn("selectionInLog()", log)
        m = re.search(r"const NOISE_RE = (/.*?/i);", html)
        self.assertIsNotNone(m)
        for noisy in ("kinect-preview", "vad", "air-mouse"):
            self.assertIn(noisy, m.group(1))
        self.assertIn('id="noiseToggle" type="checkbox" checked', html)
        self.assertIn('id="logSearch"', html)
        self.assertIn("#log.hide-noise .noise { display:none; }", html)


# ═══════════════════════════════════════════════════════════════════════════
# 9  the Voice tab
# ═══════════════════════════════════════════════════════════════════════════
class VoiceEngineTests(unittest.TestCase):

    def test_kokoro_reports_its_own_voice(self):
        kt = types.ModuleType("core.kokoro_tts")
        kt._VOICE = "bm_somebody"
        with mock.patch.dict(sys.modules, {"core.kokoro_tts": kt}):
            self.assertEqual(wi._base_voice("kokoro", "en-GB-RyanNeural"),
                             ("kokoro", "bm_somebody"))

    def test_edge_reports_tts_voice(self):
        self.assertEqual(wi._base_voice("edge", "en-GB-X"), ("edge", "en-GB-X"))

    def test_the_payload_names_engine_and_voice(self):
        cfg = types.SimpleNamespace(VOICE_CLONE_PROFILE="p", VOICE_CLONE_ENABLED=False,
                                    TTS_BACKEND="kokoro", TTS_VOICE="en-GB-RyanNeural",
                                    VOICE_CLONE_MODEL="chatterbox", VOICE_CLONE_DEVICE="",
                                    AI_BACKEND="ollama", LOCAL_LLM_MODEL="big:26b")
        d = wi._voices_info(config_mod=cfg)
        self.assertEqual(d["engine"], "kokoro")
        self.assertNotEqual(d["voice"], "en-GB-RyanNeural")
        self.assertTrue(d["summary"].startswith("kokoro"))
        self.assertTrue(d["llm_local"])
        self.assertEqual(d["local_model"], "big:26b")


class VoicePageContractTests(_ServerBase):

    def test_clone_buttons_warn_about_vram_first(self):
        html = _page(self.base)
        render = _js_fn(html, "renderVoices")
        self.assertIn("window.confirm(cloneWarning(d, p.name))", render)
        self.assertNotIn("'normal (' + (d.tts_voice", render)
        warn = _js_fn(html, "cloneWarning")
        self.assertIn("VRAM", warn)


# ═══════════════════════════════════════════════════════════════════════════
# 10  security
# ═══════════════════════════════════════════════════════════════════════════
class SecurityHeaderTests(_ServerBase):

    def _headers(self, path):
        req = urllib.request.Request(self.base + path)
        try:
            with _urlopen_retry(req, timeout=5) as r:
                return r.headers
        except urllib.error.HTTPError as e:
            return e.headers

    def test_every_response_refuses_framing_and_caching(self):
        for path in ("/", "/api/status", "/api/nope", "/api/camera-preview"):
            h = self._headers(path)
            self.assertEqual(h.get("X-Frame-Options"), "DENY", path)
            self.assertIn("frame-ancestors 'none'", h.get("Content-Security-Policy", ""), path)
            self.assertEqual(h.get("Cache-Control"), "no-store", path)
            self.assertEqual(h.get("X-Content-Type-Options"), "nosniff", path)
            self.assertEqual(len(h.get_all("Cache-Control")), 1, path)

    def test_token_compare_is_constant_time(self):
        with open(wi.__file__, encoding="utf-8") as f:
            src = f.read()
        body = src[src.index("def _authorized("):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("hmac.compare_digest(", body)
        self.assertNotIn("== token", body)


class TokenStillEnforcedTests(_ServerBase):
    token = "s3cr3t"

    def test_wrong_and_right_token(self):
        code, _ = _get_raw(self.base + "/api/status", headers={"X-Auth-Token": "s3cr3u"})
        self.assertEqual(code, 401)
        code, _ = _get_raw(self.base + "/api/status", headers={"X-Auth-Token": "s3cr3t"})
        self.assertEqual(code, 200)
        for path in ("/api/control", "/api/action", "/api/panel/x/action"):
            code, _ = _post(self.base + path, {"cmd": "force_wake"})
            self.assertEqual(code, 401, path)
        self.assertFalse(os.path.exists(self.tray_path))


class QueueLockTests(unittest.TestCase):

    def test_concurrent_injects_neither_lose_nor_duplicate(self):
        """Widen the read->replace window with a slow temp file: without the
        lock every thread reads the same list and all but one command is lost."""
        real = tempfile.mkstemp

        def slow(*a, **k):
            time.sleep(0.02)
            return real(*a, **k)

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "injected_commands.json")
            with mock.patch.object(wi.tempfile, "mkstemp", slow):
                ts = [threading.Thread(target=wi.inject_command, args=("cmd %d" % i, p))
                      for i in range(16)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join(20)
            with open(p, encoding="utf-8") as f:
                items = json.load(f)
        self.assertEqual(sorted(i["text"] for i in items),
                         sorted("cmd %d" % i for i in range(16)))

    def test_concurrent_tray_commands_all_land(self):
        real = tempfile.mkstemp

        def slow(*a, **k):
            time.sleep(0.02)
            return real(*a, **k)

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tray_commands.json")
            with mock.patch.object(wi.tempfile, "mkstemp", slow):
                ts = [threading.Thread(target=wi.send_tray_command, args=("force_wake", p))
                      for _ in range(10)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join(20)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), 10)


# ═══════════════════════════════════════════════════════════════════════════
# 11  accessibility, remembered tab, hidden-tab polling
# ═══════════════════════════════════════════════════════════════════════════
class PageHygieneTests(_ServerBase):

    def test_a11y_basics(self):
        html = _page(self.base)
        self.assertIn('<label for="text" class="sr-only">', html)
        self.assertIn('id="reply" aria-live="polite"', html)
        self.assertIn('role="log"', html)
        self.assertIn('aria-current="page"', html)
        self.assertIn("setAttribute('aria-current', 'page')", _js_fn(html, "showView"))
        self.assertIn("min-height:100vh", _css_rule(html, "body"))
        self.assertIn("color-scheme:dark", html)
        self.assertIn("setAttribute('aria-label', label)", _js_fn(html, "buildControl"))

    def test_storage_is_wrapped_and_the_last_tab_is_remembered(self):
        html = _page(self.base)
        self.assertIn("try { const v = window.localStorage.getItem(k)", _js_fn(html, "lsGet"))
        self.assertIn("try { window.localStorage.setItem(k, v); } catch", _js_fn(html, "lsSet"))
        self.assertEqual(html.count("localStorage"), 2,
                         "every storage access goes through lsGet/lsSet")
        self.assertIn("lsSet('jarvis.view', which)", _js_fn(html, "showView"))

    def test_hidden_tabs_do_not_poll(self):
        html = _page(self.base)
        self.assertIn("!document.hidden", _js_fn(html, "pollsWanted"))
        self.assertIn("if (pollsWanted()) refreshStatus();", html)
        self.assertIn("if (pollsWanted()) refreshLog();", html)
        self.assertIn("document.addEventListener('visibilitychange'", html)



class ClientGoneQuietTests(unittest.TestCase):
    """A client that hung up mid-reply gets one line, not a traceback; any
    other request error keeps the stdlib's full report (2026-10-02 census)."""

    def _server(self):
        srv = wi._WebServer(("127.0.0.1", 0), wi._Handler,
                            bind_and_activate=False)
        self.addCleanup(srv.server_close)
        return srv

    def _handle(self, srv, exc):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                raise exc
            except Exception:
                srv.handle_error(None, ("127.0.0.1", 48254))
        return out.getvalue(), err.getvalue()

    def test_each_client_gone_error_is_one_line(self):
        srv = self._server()
        for exc in (ConnectionResetError(10054, "forcibly closed"),
                    ConnectionAbortedError(10053, "aborted"),
                    BrokenPipeError(32, "broken pipe")):
            with self.subTest(type(exc).__name__):
                out, err = self._handle(srv, exc)
                self.assertIn("closed the connection", out)
                self.assertEqual(out.count("\n"), 1)
                self.assertNotIn("Traceback", out + err)

    def test_any_other_error_keeps_the_full_report(self):
        out, err = self._handle(self._server(), ValueError("real bug"))
        self.assertIn("Traceback", out + err)
        self.assertIn("ValueError: real bug", out + err)

    def test_create_server_uses_the_quiet_server(self):
        httpd = wi.create_server(bind="127.0.0.1", port=0, token="",
                                 runtime=wi.NoRuntime())
        self.addCleanup(httpd.server_close)
        self.assertIsInstance(httpd, wi._WebServer)


if __name__ == "__main__":
    unittest.main()
