"""The headset daemon's DEVICE sentences reach the monolith's audio flap
governor WITH their kind (2026-09-29).

Live 2026-09-29 15:20-15:29: a desk mic's endpoint flapped, the default
recording device bounced, and the daemon's "I may not be able to hear you"
alert and its "... off the powered-off headset now" recovery line went out on
every bounce -- alongside the monolith's own "Switched to ..." line. The two
writers only share ONE rate limit and ONE flap detector if the daemon hands
over what KIND of sentence each one is, so:

  * AudioAutoSwitch(announce_kind=...) sends device sentences there with
    "switch" / "deaf" / "deaf-clear"; the battery warning stays on announce();
  * without announce_kind nothing changes (every existing test constructs it
    that way);
  * skills/audio_autoswitch builds its daemon with announce_kind wired to
    _announce_kind, which calls bobert_companion._audio_device_announce and
    fails OPEN to the old path when that is missing or raises.

No real COM / HID / device: everything is mocked. Device names are SYNTHETIC.
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from audio import audio_switch as A

HS_MIC = "{0.0.1.00000000}.{headset-mic}"
HS_OUT = "{0.0.0.00000000}.{headset-out}"
DESK = "{0.0.1.00000000}.{desk-mic}"
SPK = "{0.0.0.00000000}.{speakers}"
HS_MIC_NAME = "Headset Microphone (Wireless Headset)"
DESK_NAME = "Microphone (Desk Mic)"


def _rows(desk_state="Active"):
    return [
        (HS_OUT, "Headphones (Wireless Headset)", "Active"),
        (SPK, "Speakers (USB Speakers)", "Active"),
        (HS_MIC, HS_MIC_NAME, "Active"),
        (DESK, DESK_NAME, desk_state),
    ]


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self, _self=None):
        return self.t


class _Stub:
    """A stand-in bobert_companion: no monolith import, no real queue."""
    PREFERRED_INPUT_DEVICES: list = []
    MICROPHONE_INDEX = None

    def __init__(self, with_gate=True, gate_raises=False):
        self.gated: list = []
        self.plain: list = []
        if with_gate:
            def _gate(message, kind="switch"):
                if gate_raises:
                    raise RuntimeError("governor blew up")
                self.gated.append((message, kind))
                return True
            self._audio_device_announce = _gate

    def proactive_announce(self, message, source="skill", **_kw):
        self.plain.append((message, source))
        return True


class _monolith:
    def __init__(self, stub):
        self.stub = stub

    def __enter__(self):
        self._had = "bobert_companion" in sys.modules
        self._prev = sys.modules.get("bobert_companion")
        sys.modules["bobert_companion"] = self.stub
        return self.stub

    def __exit__(self, *exc):
        if self._had:
            sys.modules["bobert_companion"] = self._prev
        else:
            sys.modules.pop("bobert_companion", None)
        return False


class DaemonHandsOverTheKindTests(unittest.TestCase):
    def _sw(self, sink, plain):
        sw = A.AudioAutoSwitch("Wireless Headset", "USB Speakers", poll_s=3.0,
                               announce=plain.append, mic_fallback="Desk Mic",
                               follow_mic=True,
                               announce_kind=lambda m, k: sink.append((m, k)))
        sw._believed_on = True
        sw._prior_default = SPK
        return sw

    def _tick(self, sw, clock, rows, cur_id, cur_name, *, write_ok=True):
        with _monolith(_Stub()), \
                mock.patch.object(A, "headset_powered", return_value=False), \
                mock.patch.object(A, "list_render", return_value=rows), \
                mock.patch.object(A, "default_capture",
                                  return_value=(cur_id, cur_name)), \
                mock.patch.object(A, "set_default_render", return_value=True), \
                mock.patch.object(A, "set_default_capture",
                                  return_value=write_ok), \
                mock.patch.object(A.AudioAutoSwitch, "_now", clock), \
                mock.patch.object(A, "_log"):
            try:
                sw.tick()
            except Exception:
                pass
        clock.t += 3.0

    def test_deaf_alert_and_its_recovery_carry_their_kinds(self):
        sink, plain, clock = [], [], _Clock()
        sw = self._sw(sink, plain)
        # Desk mic NotPresent, default on the powered-off headset's mic, and
        # no other Active recording endpoint: the "nowhere to go" alert.
        self._tick(sw, clock, _rows("NotPresent"), HS_MIC, HS_MIC_NAME)
        kinds = [k for _m, k in sink]
        self.assertIn("deaf", kinds, sink)
        deaf = [m for m, k in sink if k == "deaf"]
        self.assertIn("hear you", deaf[0])
        # The desk mic comes back and Windows makes it the default again.
        sink.clear()
        self._tick(sw, clock, _rows("Active"), DESK, DESK_NAME)
        self.assertEqual([k for _m, k in sink], ["deaf-clear"], sink)
        self.assertIn("off the powered-off headset now", sink[0][0])
        # Nothing about the microphone leaked onto the un-governed path.
        self.assertFalse([m for m in plain if "microphone" in m.lower()],
                         plain)

    def test_a_rescue_move_is_a_switch(self):
        sink, plain, clock = [], [], _Clock()
        sw = self._sw(sink, plain)
        # Default on the dead headset's mic, desk mic Active -> the fallback
        # rescue moves the Windows default and says so.
        self._tick(sw, clock, _rows("Active"), HS_MIC, HS_MIC_NAME)
        self.assertTrue(sink, "the rescue said nothing")
        self.assertEqual({k for _m, k in sink}, {"switch"}, sink)

    def test_the_render_switch_is_a_switch(self):
        sink, plain = [], []
        sw = A.AudioAutoSwitch("Wireless Headset", "USB Speakers", poll_s=3.0,
                               announce=plain.append,
                               announce_kind=lambda m, k: sink.append((m, k)))
        with mock.patch.object(A, "find_active",
                               return_value=(HS_OUT, "Headphones (Wireless "
                                                     "Headset)")), \
                mock.patch.object(A, "default_render_id", return_value=SPK), \
                mock.patch.object(A, "set_default_render", return_value=True):
            self.assertEqual(sw._switch_to_headset(), "to_headset")
        self.assertEqual([k for _m, k in sink], ["switch"])
        self.assertEqual(plain, [])

    def test_the_battery_warning_stays_on_the_plain_path(self):
        sink, plain = [], []
        sw = A.AudioAutoSwitch("Wireless Headset", announce=plain.append,
                               announce_kind=lambda m, k: sink.append((m, k)))
        with mock.patch.object(sw, "battery_pct", return_value=5.0):
            sw._check_low_battery()
        self.assertEqual(len(plain), 1)
        self.assertEqual(sink, [])

    def test_without_announce_kind_everything_goes_through_announce(self):
        plain, clock = [], _Clock()
        sw = A.AudioAutoSwitch("Wireless Headset", "USB Speakers", poll_s=3.0,
                               announce=plain.append, mic_fallback="Desk Mic",
                               follow_mic=True)
        sw._believed_on = True
        sw._prior_default = SPK
        self._tick(sw, clock, _rows("NotPresent"), HS_MIC, HS_MIC_NAME)
        self.assertTrue(any("hear you" in m for m in plain), plain)


class SkillWiringTests(unittest.TestCase):
    def _skill(self):
        from skills import audio_autoswitch as S
        return S

    def test_the_daemon_is_built_with_the_governed_sink(self):
        S = self._skill()
        cfg = {"AUDIO_AUTOSWITCH_HEADSET": "Wireless Headset",
               "AUDIO_AUTOSWITCH_MIC": True,
               "AUDIO_AUTOSWITCH_MIC_FALLBACK": "Desk Mic"}
        with mock.patch.object(S, "_cfg",
                               side_effect=lambda n, d=None: cfg.get(n, d)):
            d = S._make_daemon()
        self.assertIs(d.announce_kind, S._announce_kind)

    def test_announce_kind_routes_to_the_monolith_governor(self):
        S = self._skill()
        with _monolith(_Stub()) as stub:
            S._announce_kind("Switched to the Desk Mic, sir.", "switch")
            S._announce_kind("Sir, I may not be able to hear you.", "deaf")
        self.assertEqual(stub.gated,
                         [("Switched to the Desk Mic, sir.", "switch"),
                          ("Sir, I may not be able to hear you.", "deaf")])
        self.assertEqual(stub.plain, [])

    def test_no_governor_falls_back_to_the_plain_announcement(self):
        S = self._skill()
        with _monolith(_Stub(with_gate=False)) as stub:
            S._announce_kind("Sir, I may not be able to hear you.", "deaf")
        self.assertEqual(stub.plain,
                         [("Sir, I may not be able to hear you.", "audio")])

    def test_a_raising_governor_still_gets_the_alert_out(self):
        S = self._skill()
        with _monolith(_Stub(gate_raises=True)) as stub, \
                mock.patch("builtins.print"):
            S._announce_kind("Sir, I may not be able to hear you.", "deaf")
        self.assertEqual(stub.plain,
                         [("Sir, I may not be able to hear you.", "audio")])


if __name__ == "__main__":
    unittest.main()
