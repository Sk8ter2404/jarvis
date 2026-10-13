"""Hermetic tests for skills/sh_broadlink.py (BroadLink RM Pro local control).

python-broadlink is replaced by a fake module: no LAN discovery, no hub, no
network. The code store is redirected to a per-test temp dir.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class FakeHub:
    def __init__(self, learn_data=None, rf=False, auth_raises=False):
        self.sent = []
        self.learning = False
        self._learn_data = learn_data
        self._auth_raises = auth_raises
        self.sweeps = 0
        # python-broadlink 0.19 shape: check_frequency() -> (found, MHz),
        # find_rf_packet(frequency=None).
        self.freq_reads = list(rf) if isinstance(rf, (list, tuple)) else [(True, 433.92)]
        self.rf_packet_freq = "unset"
        self.cancelled = False
        if rf:
            self.sweep_frequency = self._sweep
            self.check_frequency = self._check_frequency
            self.find_rf_packet = self._find_rf_packet
            self.cancel_sweep_frequency = self._cancel

    def _check_frequency(self):
        return self.freq_reads.pop(0) if len(self.freq_reads) > 1 else self.freq_reads[0]

    def _find_rf_packet(self, frequency=None):
        self.rf_packet_freq = frequency

    def _cancel(self):
        self.cancelled = True

    def _sweep(self):
        self.sweeps += 1

    def auth(self):
        if self._auth_raises:
            raise OSError("no auth")
        return True

    def send_data(self, raw):
        self.sent.append(bytes(raw))

    def enter_learning(self):
        self.learning = True

    def check_data(self):
        if self._learn_data is None:
            raise Exception("storage error: nothing yet")
        return self._learn_data


class FakePlug:
    """A BroadLink device that is not a remote hub."""
    def auth(self):
        return True


def fake_module(devices, raises=False):
    m = mock.Mock()
    if raises:
        m.discover.side_effect = OSError("network down")
    else:
        m.discover.return_value = devices
    return m


class BroadlinkBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_broadlink")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "broadlink_codes.json")
        self.mod._CODES_PATH = self.path
        self.mod._hub["dev"] = None
        self.mod._bl = None

    def use(self, devices, raises=False):
        self.mod._bl = fake_module(devices, raises)

    def write_codes(self, codes):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(codes, f)

    @staticmethod
    def entry(raw=b"\x01\x02", kind="ir"):
        return {"code": base64.b64encode(raw).decode(), "type": kind,
                "learned_at": "2026-10-13T00:00:00+00:00"}


class DiscoveryTests(BroadlinkBase):
    def test_finds_hub_skips_non_remote(self):
        hub = FakeHub()
        self.use([FakePlug(), hub])
        self.assertIs(self.mod.discover(), hub)
        self.mod._bl.discover.assert_called_once()
        self.assertIn("timeout", self.mod._bl.discover.call_args.kwargs)

    def test_never_raises(self):
        self.use([], raises=True)
        self.assertIsNone(self.mod.discover())
        self.use([FakeHub(auth_raises=True)])
        self.assertIsNone(self.mod.discover())

    def test_library_missing(self):
        self.mod._bl = False
        self.assertIsNone(self.mod.discover())
        self.assertIn("isn't installed", self.mod.learn("stereo"))

    def test_hub_cached(self):
        self.use([FakeHub()])
        self.mod.discover()
        self.mod.discover()
        self.assertEqual(self.mod._bl.discover.call_count, 1)


class SendTests(BroadlinkBase):
    def test_no_learned_code_returns_error_for_alexa_fallback(self):
        self.use([FakeHub()])
        r = self.mod.set_state({"name": "Stereo"}, on=True)
        self.assertIn("error", r)
        self.assertNotIn("ok", r)
        self.assertIn("learn the stereo remote code", r["error"])

    def test_corrupt_store_is_tolerated(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.mod._load_codes(), {})
        self.assertIn("error", self.mod.set_state({"name": "Stereo"}, on=True))
        self.assertIn("No remote codes", self.mod.list_codes())
        self.write_codes([1, 2])
        self.assertEqual(self.mod._load_codes(), {})
        self.write_codes({"bad": "x", "good": self.entry()})
        self.assertEqual(list(self.mod._load_codes()), ["good"])

    def test_explicit_on_off_codes(self):
        hub = FakeHub()
        self.use([hub])
        self.write_codes({"stereo on": self.entry(b"ON"),
                          "stereo off": self.entry(b"OFF")})
        r = self.mod.set_state({"name": "Stereo"}, on=False)
        self.assertTrue(r["ok"])
        self.assertEqual(r["applied"], {"on": False})
        self.assertEqual(hub.sent, [b"OFF"])

    def test_toggle_code_is_honest(self):
        hub = FakeHub()
        self.use([hub])
        self.write_codes({"stereo": self.entry(b"TOG")})
        for on in (True, False):
            r = self.mod.set_state({"name": "Stereo"}, on=on)
            self.assertTrue(r["ok"])
            self.assertTrue(r["toggle"])
            self.assertTrue(r["state_unknown"])
            self.assertEqual(r["applied"], {})
            self.assertIn("sent the toggle code", r["note"])
        self.assertEqual(hub.sent, [b"TOG", b"TOG"])

    def test_hub_missing_and_send_failure_are_errors(self):
        self.write_codes({"stereo": self.entry()})
        self.use([])
        self.assertIn("error", self.mod.set_state({"name": "stereo"}, on=True))
        hub = FakeHub()
        hub.send_data = mock.Mock(side_effect=OSError("boom"))
        self.use([hub])
        r = self.mod.set_state({"name": "stereo"}, on=True)
        self.assertIn("send failed", r["error"])
        self.assertIsNone(self.mod._hub["dev"])

    def test_corrupt_code_and_unsupported_kwargs(self):
        self.use([FakeHub()])
        self.write_codes({"stereo": {"code": "!!notb64!!", "type": "ir"}})
        self.assertIn("corrupt", self.mod.set_state({"name": "stereo"}, on=True)["error"])
        self.assertIn("error", self.mod.set_state({"name": "stereo"}, brightness=5))
        self.assertIn("error", self.mod.set_state({}, on=True))
        self.assertIn("error", self.mod.get_state({"name": "stereo"}))
        self.assertEqual(self.mod.list_devices(), [])


class LearnTests(BroadlinkBase):
    def test_learn_success_persists(self):
        hub = FakeHub(learn_data=b"\xaa\xbb")
        self.use([hub])
        said = []
        out = self.mod.learn("the Stereo remote code", say=said.append)
        self.assertIn("Learned the stereo IR code", out)
        self.assertTrue(hub.learning)
        self.assertIn("press the button now", said[0])
        data = self.mod._load_codes()
        self.assertEqual(base64.b64decode(data["stereo"]["code"]), b"\xaa\xbb")
        self.assertEqual(data["stereo"]["type"], "ir")
        self.assertTrue(data["stereo"]["learned_at"])
        self.assertEqual([f for f in os.listdir(self.tmp.name) if f.endswith(".tmp")], [])

    def test_learn_timeout_stores_nothing(self):
        self.use([FakeHub(learn_data=None)])
        out = self.mod.learn("stereo", timeout=0.05, _sleep=lambda s: None)
        self.assertIn("not learned", out)
        self.assertFalse(os.path.exists(self.path))

    def test_learn_no_hub(self):
        self.use([])
        self.assertIn("couldn't find", self.mod.learn("stereo"))

    def test_learn_needs_name(self):
        self.use([FakeHub()])
        self.assertIn("Which remote code", self.mod.learn("   "))

    def test_learn_rf(self):
        hub = FakeHub(learn_data=b"RF", rf=True)
        self.use([hub])
        out = self.mod.learn("garage", kind="rf")
        self.assertIn("RF code", out)
        self.assertEqual(hub.sweeps, 1)
        self.assertEqual(self.mod._load_codes()["garage"]["type"], "rf")

    def test_learn_rf_waits_for_a_real_lock_and_passes_the_frequency(self):
        # (False, 0.0) is a truthy tuple: it must NOT count as a lock.
        hub = FakeHub(learn_data=b"RF",
                      rf=[(False, 0.0), (False, 0.0), (True, 315.0)])
        self.use([hub])
        out = self.mod.learn("gate", kind="rf", _sleep=lambda s: None)
        self.assertIn("RF code", out)
        self.assertEqual(hub.rf_packet_freq, 315.0)

    def test_learn_rf_with_no_lock_learns_nothing(self):
        hub = FakeHub(learn_data=b"RF", rf=[(False, 0.0)])
        self.use([hub])
        out = self.mod.learn("gate", kind="rf", timeout=0.05,
                             _sleep=lambda s: None)
        self.assertIn("No RF signal", out)
        self.assertTrue(hub.cancelled)
        self.assertNotIn("gate", self.mod._load_codes())

    def test_learn_rf_unsupported_hub(self):
        self.use([FakeHub()])
        self.assertIn("doesn't do RF", self.mod.learn("garage", kind="rf"))


class ActionTests(BroadlinkBase):
    def test_registered(self):
        for a in ("broadlink_learn", "broadlink_list", "broadlink_forget",
                  "learn_remote_code", "list_remote_codes", "forget_remote_code"):
            self.assertIn(a, self.actions)
            self.assertIn(a, self.mod.SPEAK_VERBATIM_ACTIONS)

    def test_learn_list_forget_flow(self):
        self.use([FakeHub(learn_data=b"X")])
        self.assertIn("No remote codes", self.actions["broadlink_list"](""))
        self.assertIn("Learned the stereo IR code",
                      self.actions["broadlink_learn"]("the stereo remote code"))
        self.assertIn("stereo (ir)", self.actions["broadlink_list"](""))
        self.assertIn("Forgot the stereo", self.actions["broadlink_forget"]("forget the stereo code"))
        self.assertIn("no learned code", self.actions["broadlink_forget"]("stereo"))
        self.assertEqual(self.mod._load_codes(), {})

    def test_learn_is_never_automatic(self):
        # set_state / discover must never enter learning mode on their own.
        hub = FakeHub(learn_data=b"X")
        self.use([hub])
        self.mod.set_state({"name": "stereo"}, on=True)
        self.mod.discover()
        self.assertFalse(hub.learning)


class RouterWiringTests(unittest.TestCase):
    def test_brand_maps_to_skill(self):
        from core import smart_home_router as r
        self.assertEqual(r._controller_for("BroadLink"), "sh_broadlink")
        self.assertEqual(r._controller_for("broadlink"), "sh_broadlink")

    def test_unlearned_device_errors_through_call_skill(self):
        from core import smart_home_router as r
        mod, _ = load_skill_isolated("sh_broadlink")
        mod._CODES_PATH = os.path.join(tempfile.gettempdir(), "no_such_dir_bl", "x.json")
        with mock.patch.object(r, "_import_skill", return_value=mod):
            res = r._call_skill("sh_broadlink", {"name": "Stereo"}, on=True)
        self.assertIn("error", res)     # _dispatch_one then falls back to Alexa


if __name__ == "__main__":
    unittest.main()
