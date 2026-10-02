"""The dashboard's "What JARVIS did" timeline (GET /api/timeline, 2026-10-02).

The last 50 turns as a timeline: time, source (voice / typed / web), each
action run with ok / failed, and the latency fields of the turn's
[turn-timing] line. Nothing in the running process keeps that per-turn record,
so the server reads the session logs READ-ONLY through the latency report's own
parser (tools/turn_latency_report.parse_log) - no second [turn-timing] parser.

What was SAID (the owner's words, JARVIS's reply) is personal. It is in the
payload only when DASHBOARD_SHOW_TRANSCRIPTS (core/config.py, default False) is
on AND the request's PEER address is loopback: a LAN client never sees it, with
the setting on, with the token, and whatever headers it sends.

Every log line here is SYNTHETIC (made-up commands, no real transcript).
Headless-CI safe: a 127.0.0.1:0 server in a temp dir (tests.test_web_interface).

    python tools/run_tests.py web_timeline
"""
from __future__ import annotations

import builtins
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import core.config as core_config
from tools import settings_window as sw
from tools import web_interface as wi
from tests.test_web_interface import _ServerBase, _get, _get_raw, _js_fn

_TIMING_VOICE = (
    "[turn-timing] kind=voice outcome=ok vad_break=0 stt_start=3 stt_end=400 "
    "you=410 llm_post=450 llm_done=1200 actions_done=1210 synth_start=1220 "
    "first_play=1500 end=3000 prompt_eval_count=900 prompt_eval_ms=200 "
    "eval_count=20 eval_ms=300 llm_calls=1 turn_ctx_chars=10 sys_chars=20 "
    "followup_rounds=0 filler=0 filler_ms=- lead_dropped=0")

# Three turns: a spoken one, one typed on the dashboard (the loop logs
# "[inject] (web) ..."), and one typed through another inject writer.
LOG = (
    "[09:00:01] Listening…\n"
    "[09:00:02]   You:    zebra quartz lantern\n"
    "[09:00:03]   JARVIS: [ACTION: get_time] One moment, sir.\n"
    "[09:00:03]   [action] get_time: current time is nine\n"
    "[09:00:04]   JARVIS: It is nine, sir.\n"
    "[09:00:05]   " + _TIMING_VOICE + "\n"
    "[09:00:30] Listening…\n"
    "[09:01:00]   [inject] (web) open the marble door\n"
    "[09:01:00]   You:    open the marble door\n"
    "[09:01:01]   [action] launch_app failed [os]: no such file\n"
    "[09:01:01]   [action] ⚠  REQUIRES CONFIRMATION: send_email(x) — say "
    "'yes' to proceed\n"
    "[09:01:02]   [turn-timing] kind=inject outcome=ok vad_break=- "
    "stt_start=- stt_end=- you=5 llm_post=- llm_done=- actions_done=- "
    "synth_start=- first_play=- end=900\n"
    "[09:02:00]   [inject] copper kettle words\n"
    "[09:02:00]   You:    copper kettle words\n"
    "[09:02:01]   [action] get_weather: could not reach the service\n"
    "[09:02:02]   [turn-timing] kind=inject outcome=error end=800\n"
)
SAID = ("zebra quartz lantern", "open the marble door", "copper kettle words",
        "It is nine")
LOG_NAME = "session_2026-10-02_09-00-00.log"


def _write_log(log_dir, text=LOG, name=LOG_NAME):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, name), "w", encoding="utf-8") as f:
        f.write(text)


def _said_anywhere(payload) -> list:
    blob = json.dumps(payload)
    return [w for w in SAID if w in blob]


class _TempLogs(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jarvis_timeline_")
        self.addCleanup(shutil.rmtree, self.dir, True)


class BuildTimelineTests(_TempLogs):
    def test_sources_actions_and_latency_newest_first(self):
        _write_log(self.dir)
        d = wi.build_timeline(self.dir)
        self.assertEqual(d["count"], 3)
        self.assertEqual(d["logs"], [LOG_NAME])
        typed, web, voice = d["turns"]
        self.assertEqual([t["source"] for t in d["turns"]],
                         ["typed", "web", "voice"])
        self.assertEqual((voice["time"], voice["date"]),
                         ("09:00:05", "2026-10-02"))
        self.assertEqual(voice["actions"], [{"name": "get_time",
                                             "status": "ok"}])
        self.assertEqual(web["actions"], [
            {"name": "launch_app", "status": "failed"},
            {"name": "send_email", "status": "pending"}])
        # A soft failure is judged by the follow-up loop's own markers.
        self.assertEqual(typed["actions"], [{"name": "get_weather",
                                             "status": "failed"}])
        self.assertEqual(typed["outcome"], "error")
        self.assertEqual(voice["latency"], {
            "answer_ms": 1500, "end_ms": 3000, "stt_ms": 397, "llm_ms": 750,
            "filler_ms": None})
        self.assertEqual(voice["marks"]["vad_break"], 0)
        self.assertEqual(voice["marks"]["first_play"], 1500)
        self.assertIsNone(web["latency"]["answer_ms"])
        self.assertEqual(web["latency"]["end_ms"], 900)

    def test_no_words_unless_asked(self):
        _write_log(self.dir)
        d = wi.build_timeline(self.dir)
        self.assertFalse(d["show_text"])
        for t in d["turns"]:
            self.assertNotIn("you", t)
            self.assertNotIn("reply", t)
        self.assertEqual(_said_anywhere(d), [])

    def test_words_when_asked(self):
        _write_log(self.dir)
        d = wi.build_timeline(self.dir, show_text=True)
        typed, web, voice = d["turns"]
        self.assertEqual(voice["you"], "zebra quartz lantern")
        self.assertEqual(voice["reply"], "It is nine, sir.")
        self.assertEqual(web["you"], "open the marble door")
        self.assertEqual(typed["you"], "copper kettle words")

    def test_a_standby_web_inject_is_still_web(self):
        _write_log(self.dir, "[09:01:00]   [inject] (standby) (web) Jarvis hi\n"
                             "[09:01:00]   You:    Jarvis hi\n"
                             "[09:01:02]   [turn-timing] kind=inject "
                             "outcome=ok end=900\n")
        self.assertEqual(wi.build_timeline(self.dir)["turns"][0]["source"],
                         "web")

    def test_only_the_last_50_turns_across_logs(self):
        def turns(n, tag):
            return "".join(
                f"[10:{i // 60:02d}:{i % 60:02d}]   You:    {tag} {i}\n"
                f"[10:{i // 60:02d}:{i % 60:02d}]   [turn-timing] "
                f"kind=voice outcome=ok end={i}\n" for i in range(n))
        _write_log(self.dir, turns(40, "old"),
                   "session_2026-10-01_10-00-00.log")
        _write_log(self.dir, turns(20, "new"),
                   "session_2026-10-02_10-00-00.log")
        d = wi.build_timeline(self.dir, show_text=True)
        self.assertEqual(d["count"], 50)
        self.assertEqual(d["turns"][0]["you"], "new 19")    # newest first
        self.assertEqual(d["turns"][19]["you"], "new 0")
        self.assertEqual(d["turns"][20]["you"], "old 39")
        self.assertEqual(d["turns"][-1]["you"], "old 10")   # 10 oldest gone
        self.assertEqual(d["logs"], ["session_2026-10-02_10-00-00.log",
                                     "session_2026-10-01_10-00-00.log"])

    def test_an_action_carries_its_name_only(self):
        _write_log(self.dir,
                   "[09:00:02]   You:    hello\n"
                   "[09:00:03]   [action] <img src=x onerror=alert(1)>: ok\n"
                   "[09:00:03]   [action] read_mail: secret subject line\n"
                   "[09:00:05]   [turn-timing] kind=voice outcome=ok end=1\n")
        d = wi.build_timeline(self.dir)
        self.assertEqual(d["turns"][0]["actions"],
                         [{"name": "read_mail", "status": "ok"}])
        blob = json.dumps(d)
        self.assertNotIn("onerror", blob)
        self.assertNotIn("secret subject", blob)

    def test_no_logs_is_an_empty_timeline(self):
        self.assertEqual(wi.build_timeline(self.dir)["turns"], [])
        self.assertEqual(
            wi.build_timeline(os.path.join(self.dir, "missing"))["count"], 0)

    def test_the_logs_are_only_read(self):
        _write_log(self.dir)
        real_open = builtins.open
        modes = []

        def guarded(file, mode="r", *a, **k):
            modes.append(mode)
            if any(c in mode for c in "wax+"):
                raise AssertionError(f"timeline opened {file!r} for {mode!r}")
            return real_open(file, mode, *a, **k)

        with mock.patch("builtins.open", guarded):
            wi.build_timeline(self.dir, show_text=True)
        self.assertEqual(modes, ["rb"])

    def test_it_reuses_the_latency_reports_parser(self):
        from tools import turn_latency_report as rep
        _write_log(self.dir)
        with mock.patch.object(rep, "parse_log",
                               wraps=rep.parse_log) as parse:
            wi.build_timeline(self.dir)
        parse.assert_called_once()
        self.assertEqual(parse.call_args.kwargs, {"keep_lines": 50})


class LocalClientTests(unittest.TestCase):
    def test_loopback_peers_are_local(self):
        for host in ("127.0.0.1", "127.8.9.10", "::1", "::ffff:127.0.0.1"):
            self.assertTrue(wi.is_local_client(host), host)

    def test_everything_else_is_not(self):
        # Documentation addresses (RFC 5737 / 3849) stand in for LAN peers.
        for host in ("192.0.2.20", "198.51.100.7", "203.0.113.9", "0.0.0.0",
                     "2001:db8::5", "fe80::1%eth0", "::ffff:192.0.2.20",
                     "localhost", "", None, "not an address"):
            self.assertFalse(wi.is_local_client(host), host)


class SettingTests(unittest.TestCase):
    def test_off_by_default_and_in_the_settings_schema(self):
        self.assertIs(core_config.DASHBOARD_SHOW_TRANSCRIPTS, False)
        row = sw.SCHEMA["DASHBOARD_SHOW_TRANSCRIPTS"]
        self.assertEqual((row["type"], row["default"]), ("bool", False))

    def test_only_a_real_true_turns_it_on(self):
        for value, want in ((True, True), (False, False), ("true", False),
                            (1, False), (None, False)):
            with mock.patch.object(core_config, "DASHBOARD_SHOW_TRANSCRIPTS",
                                   value):
                self.assertIs(wi._transcripts_setting(), want, value)


class TimelineRouteTests(_ServerBase):
    def setUp(self):
        super().setUp()
        _write_log(self.log_dir)

    def _setting(self, on):
        p = mock.patch.object(core_config, "DASHBOARD_SHOW_TRANSCRIPTS", on)
        p.start()
        self.addCleanup(p.stop)

    def test_the_route_answers_json(self):
        code, d = _get(self.base + "/api/timeline")
        self.assertEqual(code, 200)
        self.assertEqual(d["count"], 3)
        self.assertEqual([t["source"] for t in d["turns"]],
                         ["typed", "web", "voice"])

    def test_words_hidden_by_default_even_on_this_pc(self):
        self._setting(False)
        code, d = _get(self.base + "/api/timeline")
        self.assertEqual(code, 200)
        self.assertEqual(d["transcripts"], "off")
        self.assertEqual(_said_anywhere(d), [])

    def test_this_pc_sees_the_words_when_the_setting_is_on(self):
        self._setting(True)
        code, d = _get(self.base + "/api/timeline")
        self.assertEqual(d["transcripts"], "shown")
        self.assertEqual(d["turns"][2]["you"], "zebra quartz lantern")

    def test_a_lan_client_never_sees_the_words(self):
        """The setting is on, but the peer is a LAN address: no words, and
        a header claiming loopback changes nothing - only the socket's peer
        address is checked."""
        self._setting(True)
        seen = []

        def lan_peer(host):
            seen.append(host)
            return False

        with mock.patch.object(wi, "is_local_client", side_effect=lan_peer):
            code, body = _get_raw(self.base + "/api/timeline", headers={
                "X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"})
        self.assertEqual(code, 200)
        self.assertEqual(seen, ["127.0.0.1"])        # the socket's peer
        d = json.loads(body)
        self.assertEqual(d["transcripts"], "local_only")
        self.assertFalse(d["show_text"])
        for t in d["turns"]:
            self.assertNotIn("you", t)
            self.assertNotIn("reply", t)
        for words in SAID:
            self.assertNotIn(words, body)

    def test_it_is_served_as_json_not_html(self):
        import urllib.request
        with urllib.request.urlopen(self.base + "/api/timeline",
                                    timeout=5) as r:
            self.assertTrue(r.headers["Content-Type"].startswith(
                "application/json"))
            self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")


class TimelineTokenTests(_ServerBase):
    token = "s3cr3t"

    def test_the_token_is_required(self):
        code, _ = _get_raw(self.base + "/api/timeline")
        self.assertEqual(code, 401)
        code, _ = _get_raw(self.base + "/api/timeline",
                           headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)


class TimelinePageTests(unittest.TestCase):
    """The view is in the navigation, and nothing it renders goes through
    innerHTML: words, action names and errors land via textContent."""

    def setUp(self):
        self.html = wi._dashboard_html("")

    def test_the_view_is_in_the_navigation(self):
        self.assertIn('id="navTimeline"', self.html)
        self.assertIn('id="viewTimeline"', self.html)
        self.assertIn("timeline: {nav:navTimeline, view:viewTimeline}",
                      self.html)
        self.assertIn("fetch(q('/api/timeline')", self.html)

    def test_rendered_values_never_reach_innerHTML(self):
        for fn in ("renderTimeline", "loadTimeline", "tlChip"):
            body = _js_fn(self.html, fn)
            for m in re.finditer(r"innerHTML\s*=\s*([^;]+);", body):
                self.assertRegex(m.group(1).strip(), r"^'[^'+]*'$",
                                 f"{fn} builds innerHTML from data")


def _node():
    return shutil.which("node")


@unittest.skipUnless(_node(), "node is not installed")
class TimelineRenderXssTests(unittest.TestCase):
    """Run the page's own renderTimeline under node against a minimal DOM with
    hostile values in every field: they must arrive as text, never markup."""

    HOSTILE = "<img src=x onerror=alert(1)>"

    def test_hostile_text_is_text(self):
        html = wi._dashboard_html("")
        js = "\n".join(_js_fn(html, f) + "\n}" for f in
                       ("fmtMs", "tlChip", "renderTimeline"))
        consts = "\n".join(re.search(r"^const %s = [^;]+;" % n, html,
                                     re.M | re.S).group(0)
                           for n in ("TL_STATUS",))
        h = json.dumps(self.HOSTILE)
        script = DOM_SHIM + consts + "\n" + js + """
const tlList = document.createElement('div');
renderTimeline({turns: [{time: %(h)s, date: %(h)s, source: %(h)s,
  outcome: %(h)s, actions: [{name: %(h)s, status: %(h)s}],
  latency: {}, you: %(h)s, reply: %(h)s}]});
console.log(JSON.stringify({html: HTML_SETS, text: textOf(tlList)}));
""" % {"h": h}
        out = subprocess.run([_node(), "-e", script], capture_output=True,
                             text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        # Only the page's own constant strings ever go through innerHTML.
        self.assertEqual(set(res["html"]) - {""}, set())
        self.assertIn(self.HOSTILE, res["text"])


# A minimal DOM: every innerHTML assignment is recorded (and ignored); text
# goes through textContent. Enough for the render functions under test.
DOM_SHIM = r"""
const HTML_SETS = [];
function El(tag) { this.tag = tag; this.children = []; this.textContent = '';
  this.className = ''; this.style = {}; this.title = ''; this.attrs = {};
  this.classList = {add: () => {}, toggle: () => {}};
  this.listeners = {}; }
El.prototype.appendChild = function (c) {
  if (c && c.frag) this.children.push(...c.children); else this.children.push(c);
  return c; };
El.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };
El.prototype.addEventListener = function (k, f) { this.listeners[k] = f; };
Object.defineProperty(El.prototype, 'innerHTML', {
  set(v) { HTML_SETS.push(String(v)); this.children = []; },
  get() { return ''; }});
const document = {
  createElement: (t) => new El(t),
  createDocumentFragment: () => { const f = new El('#frag'); f.frag = true; return f; },
  getElementById: () => new El('div') };
function textOf(el) { return [el.textContent, el.title,
  ...el.children.map(textOf)].join('|'); }
"""


if __name__ == "__main__":
    unittest.main()
