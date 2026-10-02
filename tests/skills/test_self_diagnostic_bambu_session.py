"""The self-check FAILs when the printer had no session this boot (NEW #3, 2026-10-02).

Live 2026-10-01 the Bambu monitor never connected after 19:44 (TCP 8883
closed) and every self-check sweep still reported the printer PASS -
"skipped: printer offline (monitor backed off)". Two faults:

  * _probe_bambu did `from skills import bambu_monitor`, which imports a
    SECOND, never-started copy of the skill (the live one is
    sys.modules["skill_bambu_monitor"]). That copy has no session and no
    status, so its is_printer_offline() was ALWAYS True and the probe
    ALWAYS skipped - a stale duplicate.
  * even on the live module, "offline" was a pass whether or not the
    monitor had connected once since boot.

Now the probe reads the LIVE module only. No session since boot, with a
failed attempt recorded (or past the first-connect grace) -> FAIL with the
monitor's reason and hint. A printer that had a session and is now asleep
is still a skip; a monitor still making its first attempt is not judged.

    python -m unittest tests.skills.test_self_diagnostic_bambu_session
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

from tests.skills.test_self_diagnostic import _ProbeTestBase, inject_modules

BC = types.SimpleNamespace(BAMBU_PRINTER_IP="192.0.2.10",
                           BAMBU_ACCESS_CODE="code", BAMBU_SERIAL="ser")


def _monitor(*, offline=True, ever=False, failures=0, age=600.0,
             state="failed", kind="refused",
             reason="192.0.2.10 answers, but it refused the connection on "
                    "port 8883",
             hint="Check that LAN-only or developer mode is still on."):
    bm = types.ModuleType("skill_bambu_monitor")
    bm.is_printer_offline = lambda: offline
    bm.connection_status = lambda: {
        "state": state, "kind": kind, "reason": reason, "hint": hint,
        "ip": "192.0.2.10", "failures": failures, "ever_connected": ever,
        "never_connected_since_boot": not ever, "monitor_running": True,
        "age_s": age}
    return bm


class BambuSessionProbeTests(_ProbeTestBase):

    def _probe(self, live, duplicate=None):
        mods = {"skill_bambu_monitor": live}
        # What `from skills import bambu_monitor` used to get: a fresh copy
        # that never started, so it always says "offline".
        mods["skills.bambu_monitor"] = duplicate or _monitor(offline=True)
        with mock.patch.object(self.mod, "_bc", return_value=BC), \
             inject_modules(**mods):
            return self.mod._probe_bambu()

    def test_no_session_since_boot_fails_with_the_reason(self):
        r = self._probe(_monitor(offline=True, ever=False, failures=12))
        self.assertFalse(r["ok"],
                         "no MQTT session all boot was reported as a PASS")
        self.assertTrue(r["tested"])
        self.assertIn("since JARVIS started", r["error"])
        self.assertIn("refused", r["error"])
        self.assertIn("developer mode", r["error"])

    def test_the_live_module_is_read_not_a_fresh_duplicate(self):
        # Live monitor: connected and healthy. The duplicate says offline.
        live = _monitor(offline=False, ever=True, failures=0,
                        state="connected", kind="", reason="up")
        dup = _monitor(offline=True, ever=False, failures=0, age=0.0,
                       state="not started", kind="", reason="")
        with mock.patch.object(self.mod, "_bc", return_value=BC), \
             inject_modules(**{"skill_bambu_monitor": live,
                               "skills.bambu_monitor": dup}), \
             mock.patch.dict(self.mod.sys.modules, {"paho": None}):
            r = self.mod._probe_bambu()
        # With the live module online the probe goes on to its own connect
        # (paho blocked here -> the paho error), never the duplicate's skip.
        self.assertNotIn("skipped", r.get("details") or {})

    def test_a_printer_asleep_after_a_session_is_still_a_skip(self):
        r = self._probe(_monitor(offline=True, ever=True, failures=3,
                                 state="disconnected", kind="dropped"))
        self.assertTrue(r["ok"])
        self.assertIn("offline", r["details"]["skipped"])

    def test_the_first_attempt_is_not_judged(self):
        r = self._probe(_monitor(offline=True, ever=False, failures=0,
                                 age=10.0, state="connecting", kind="",
                                 reason="connecting"))
        self.assertTrue(r["ok"])
        self.assertIn("connecting", r["details"]["skipped"])

    def test_no_failure_yet_but_past_the_grace_fails(self):
        r = self._probe(_monitor(offline=True, ever=False, failures=0,
                                 age=600.0, state="connecting", kind="",
                                 reason="connecting to 192.0.2.10", hint=""))
        self.assertFalse(r["ok"])
        self.assertIn("since JARVIS started", r["error"])

    def test_a_monitor_without_connection_status_keeps_the_old_skip(self):
        bm = types.ModuleType("skill_bambu_monitor")
        bm.is_printer_offline = lambda: True
        r = self._probe(bm)
        self.assertTrue(r["ok"])
        self.assertIn("offline", r["details"]["skipped"])


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
