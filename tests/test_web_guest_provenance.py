"""The web dashboard's half of fact provenance and guest mode (2026-10-02).

  * GET /api/memory carries each fact's provenance -- who said it (the
    voice-ID verdict), how it reached JARVIS, when -- and None for a fact
    stored before provenance existed. The SENTENCE it came from is personal,
    like a transcript: it is in the payload only for a loopback peer with
    DASHBOARD_SHOW_TRANSCRIPTS on (the timeline's rule), never for a LAN
    client, with the token or without.
  * GET /api/status reports ``guest_mode`` from hud_state.json, the strip
    shows a chip while it is on, and the controls offer it on / off through
    the tray control plane (POST /api/control).

Every fact and sentence is made up. Headless-CI safe: a 127.0.0.1:0 server in a
temp dir (tests.test_web_interface._ServerBase); the long-term store's paths
are repointed at temp files.

    python tools/run_tests.py web_guest_provenance
"""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import core.config as core_config
import core.long_term_memory as ltm
from tools import web_interface as wi
from tests.test_web_interface import _ServerBase, _get, _get_raw
from tests.test_web_interface_audit_fixes import _page

_SENTENCE = "my cousin Quillon keeps three ferrets"
_FACTS = [
    {"id": "fact_new", "text": "User's cousin keeps three ferrets",
     "source": "merge_memory", "tags": ["learned"], "updated_at": 2.0,
     "provenance": {"speaker": "owner", "source": "voice",
                    "utterance": _SENTENCE, "ts": 1_790_000_000.0}},
    {"id": "fact_old", "text": "User drinks green tea",
     "source": "bobert_memory_migration", "tags": ["legacy"],
     "updated_at": 1.0},
]


class _MemoryServer(_ServerBase):
    def setUp(self):
        super().setUp()
        facts = os.path.join(self.d, "facts.json")
        with open(facts, "w", encoding="utf-8") as f:
            json.dump(_FACTS, f)
        for name, value in (("_FACTS_JSON", facts),
                            ("_EPISODE_LOG", os.path.join(self.d, "ep.jsonl")),
                            ("_loaded", False)):
            p = mock.patch.object(ltm, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _setting(self, on):
        p = mock.patch.object(core_config, "DASHBOARD_SHOW_TRANSCRIPTS", on)
        p.start()
        self.addCleanup(p.stop)

    def _facts(self, body):
        return {f["text"]: f for f in json.loads(body)["facts"]}


class MemoryProvenanceRouteTests(_MemoryServer):
    def test_each_fact_says_who_how_and_when(self):
        self._setting(False)
        code, body = _get_raw(self.base + "/api/memory")
        self.assertEqual(code, 200)
        facts = self._facts(body)
        self.assertEqual(facts["User's cousin keeps three ferrets"]
                         ["provenance"],
                         {"speaker": "owner", "source": "voice",
                          "ts": 1_790_000_000.0})
        # A fact from before provenance: listed as always, no record.
        self.assertIsNone(facts["User drinks green tea"]["provenance"])
        self.assertEqual(json.loads(body)["utterances"], "off")
        self.assertNotIn("Quillon", body)

    def test_this_pc_sees_the_sentence_when_the_setting_is_on(self):
        self._setting(True)
        code, d = _get(self.base + "/api/memory")
        self.assertEqual(d["utterances"], "shown")
        new = [f for f in d["facts"] if f["provenance"]][0]
        self.assertEqual(new["provenance"]["utterance"], _SENTENCE)

    def test_a_lan_client_never_sees_the_sentence(self):
        self._setting(True)
        with mock.patch.object(wi, "is_local_client", return_value=False):
            code, body = _get_raw(self.base + "/api/memory", headers={
                "X-Forwarded-For": "127.0.0.1"})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["utterances"], "local_only")
        self.assertNotIn("Quillon", body)
        self.assertEqual(self._facts(body)["User's cousin keeps three ferrets"]
                         ["provenance"]["source"], "voice")

    def test_read_memory_is_hidden_by_default(self):
        out = wi._read_memory()
        self.assertNotIn("Quillon", json.dumps(out))
        self.assertIn("Quillon", json.dumps(wi._read_memory(
            show_utterances=True)))


class MemoryProvenancePageTests(_ServerBase):
    def test_the_memory_list_renders_provenance_as_text(self):
        html = _page(self.base)
        render = html[html.index("function renderFacts("):]
        render = render[:render.index("\n}", 0)]
        self.assertIn("provLabel(fact.provenance)", render)
        self.assertIn("fact.provenance.utterance", render)
        self.assertNotIn("innerHTML=fact", render.replace(" ", ""))
        label = html[html.index("function provLabel("):]
        label = label[:label.index("\n}")]
        for word in ("'you'", "'unconfirmed voice'", "'overheard'",
                     "'web page'", "'source not recorded'"):
            self.assertIn(word, html[:html.index("function renderFacts(")])
        self.assertIn("toLocaleString", label)


class GuestModeStatusTests(_ServerBase):
    def _status(self, hud):
        with open(self.hud_path, "w", encoding="utf-8") as f:
            json.dump(hud, f)
        return _get(self.base + "/api/status")[1]

    def test_status_reports_guest_mode(self):
        self.assertIs(self._status({"state": "Idle", "guest_mode": True})
                      ["guest_mode"], True)
        self.assertIs(self._status({"state": "Idle", "guest_mode": False})
                      ["guest_mode"], False)
        # Not published (an older JARVIS): unknown, never "off".
        self.assertIsNone(self._status({"state": "Idle"})["guest_mode"])

    def test_the_flags_helper(self):
        self.assertIs(wi._status_flags({"guest_mode": 1})["guest_mode"], True)
        self.assertIsNone(wi._status_flags({})["guest_mode"])

    def test_the_page_shows_a_chip_and_offers_the_switch(self):
        html = _page(self.base)
        self.assertIn("if (s.guest_mode) strip.appendChild(chip('guest mode'",
                      html)
        self.assertRegex(html, r"\{cmd:'guest_mode_on',[^\n]*show:")
        self.assertRegex(html, r"\{cmd:'guest_mode_off',[^\n]*show:")

    def test_the_switch_goes_through_the_tray_control_plane(self):
        for cmd in ("guest_mode_on", "guest_mode_off"):
            self.assertIn(cmd, wi.TRAY_WEB_COMMANDS)
        import urllib.request
        req = urllib.request.Request(
            self.base + "/api/control", method="POST",
            data=json.dumps({"cmd": "guest_mode_on"}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
        with open(self.tray_path, encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual([i["cmd"] for i in items], ["guest_mode_on"])


if __name__ == "__main__":
    unittest.main()
