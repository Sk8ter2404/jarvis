"""Monolith half of the what_microphone fix (2026-09-29).

"what microphone are you using" named a configured mic Windows reported
NotPresent while the capture ran on another device: the only thing the skill
could read was get_current_mic_name() — the SELECTED device, what the next open
would ask for, out of PortAudio's frozen list. These tests pin the monolith
pieces the skill now reads instead:

  * record_speech publishes what its InputStream REALLY opened on
    (_note_live_capture / get_live_capture_device), including after the
    stale-index retry that silently swaps in the system default;
  * get_capture_endpoints() returns Windows' recording endpoints WITH their
    state (and None — never [] — when Windows cannot be asked);
  * get_default_capture_name() names Windows' live default;
  * the skill, loaded against the REAL monolith, finds every one of those names
    (a rename on either side would otherwise degrade silently).

Synthetic device names only; sounddevice's InputStream / query_devices, the
MMDevice enumerator and audio_switch are all faked — no device is opened.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests._skill_harness import load_skill_isolated

_DESK = "Microphone (Desk Mic)"
_HEADSET = "Headset Microphone (Headset Mic)"


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _drive_record_speech(self, stream_factory, in_dev, names):
        """Run record_speech just far enough to open + start its stream, then
        let the watchdog bail it out (the pattern test_monolith_runtime_bugfixes
        uses). Returns what record_speech returned."""
        bc = self.bc
        bc._watchdog_reset_signal.set()
        self.addCleanup(bc._watchdog_reset_signal.clear)
        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "get_input_device", return_value=in_dev)
        self._p(bc, "_safe_close_stream", lambda s: None)
        self._p(bc.sd, "InputStream", stream_factory)
        self._p(bc.sd, "query_devices",
                side_effect=lambda i=None: {"name": names[i]})
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        with mock.patch("builtins.print"):
            return bc.record_speech(timeout=0.0)


@requires_monolith
class LiveCaptureRecordTests(_Base):
    def test_nothing_published_before_the_first_open(self):
        self.assertIsNone(self.bc.get_live_capture_device())

    def test_record_speech_publishes_the_device_it_opened(self):
        class FakeStream:
            device = 7                     # sounddevice's RESOLVED index

            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

        self._drive_record_speech(FakeStream, in_dev=7, names={7: _HEADSET})
        rec = self.bc.get_live_capture_device()
        self.assertIsNotNone(rec, "record_speech opened a stream but published "
                                  "nothing — what_microphone would still read "
                                  "the SELECTED device")
        self.assertEqual((rec["index"], rec["name"], rec["requested"],
                          rec["via_default"]), (7, _HEADSET, 7, False))
        self.assertIn("live", rec)

    def test_stale_index_retry_publishes_the_default_it_fell_back_to(self):
        """The retry path swaps the cached mic for device=None. The published
        record must be the device the RETRY opened, not the one requested."""
        bc = self.bc
        opened = []

        class FakeStream:
            def __init__(self, *a, device=None, **k):
                if device == 3:
                    raise bc.sd.PortAudioError("Error querying device 3")
                self.device = 5            # what device=None resolved to
                opened.append(device)

            def start(self):
                pass

        self._drive_record_speech(FakeStream, in_dev=3,
                                  names={3: _DESK, 5: _HEADSET})
        rec = bc.get_live_capture_device()
        self.assertEqual(opened, [None])
        self.assertEqual((rec["index"], rec["name"], rec["via_default"]),
                         (5, _HEADSET, True))

    def test_returned_record_is_a_copy(self):
        self.bc._live_capture_device[0] = {"index": 1, "name": _DESK}
        rec = self.bc.get_live_capture_device()
        rec["name"] = "tampered"
        self.assertEqual(self.bc._live_capture_device[0]["name"], _DESK)

    def test_note_live_capture_never_raises(self):
        class Weird:
            @property
            def device(self):
                raise RuntimeError("boom")

        self._p(self.bc.sd, "query_devices", side_effect=RuntimeError("pa"))
        self.bc._note_live_capture(Weird(), 2)   # must not raise
        self.bc._note_live_capture(object(), None)


@requires_monolith
class WindowsEndpointTests(_Base):
    def _fake_switch(self, rows=None, raises=None):
        mod = types.ModuleType("audio.audio_switch")
        mod.CAPTURE_PREFIX = "{0.0.1."

        def list_render():
            if raises:
                raise raises
            return list(rows or [])

        mod.list_render = list_render
        patcher = mock.patch.dict(sys.modules, {"audio.audio_switch": mod})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_capture_rows_only_with_their_state(self):
        self._p(self.bc, "_win_endpoint_enumerator", return_value=object())
        self._fake_switch([
            ("{0.0.0.00000000}.{a}", "Speakers (Desk Speakers)", "Active"),
            ("{0.0.1.00000000}.{b}", _DESK, "NotPresent"),
            ("{0.0.1.00000000}.{c}", _HEADSET, "Active"),
        ])
        self.assertEqual(self.bc.get_capture_endpoints(), [
            ("{0.0.1.00000000}.{b}", _DESK, "NotPresent"),
            ("{0.0.1.00000000}.{c}", _HEADSET, "Active"),
        ])

    def test_unavailable_is_none_never_an_empty_list(self):
        self._p(self.bc, "_win_endpoint_enumerator", return_value=None)
        self._fake_switch([("{0.0.1.00000000}.{b}", _DESK, "Active")])
        self.assertIsNone(self.bc.get_capture_endpoints())

    def test_enumeration_error_or_empty_is_none(self):
        self._p(self.bc, "_win_endpoint_enumerator", return_value=object())
        self._fake_switch(raises=OSError("COM"))
        self.assertIsNone(self.bc.get_capture_endpoints())
        self._fake_switch([])
        self.assertIsNone(self.bc.get_capture_endpoints())

    def test_default_capture_name(self):
        self._p(self.bc, "_win_default_endpoints",
                return_value=("{0.0.0.00000000}.{a}", "{0.0.1.00000000}.{c}"))
        self._p(self.bc, "_win_endpoint_friendly_name",
                side_effect=lambda i: _HEADSET if i.endswith("{c}") else None)
        self.assertEqual(self.bc.get_default_capture_name(), _HEADSET)
        self._p(self.bc, "_win_default_endpoints", return_value=(None, None))
        self.assertIsNone(self.bc.get_default_capture_name())


@requires_monolith
class SkillAgainstRealMonolithTests(_Base):
    """The skill resolves every helper by NAME on the loaded monolith; a fake
    can only prove the skill's logic. This proves the names line up."""

    def _mic(self):
        patcher = mock.patch.dict(sys.modules, {"bobert_companion": self.bc})
        patcher.start()
        self.addCleanup(patcher.stop)
        _mod, actions = load_skill_isolated("audio_devices")
        return actions["what_microphone"]("")

    def test_notpresent_live_device_is_not_claimed(self):
        bc = self.bc
        bc._live_capture_device[0] = {"index": 3, "name": _DESK,
                                      "requested": 3, "via_default": False,
                                      "at": 0.0}
        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "MICROPHONE_INDEX", None)
        self._p(bc, "PREFERRED_INPUT_DEVICES", ["Desk Mic"])
        self._p(bc, "get_capture_endpoints", return_value=[
            ("{0.0.1.00000000}.{b}", _DESK, "NotPresent"),
            ("{0.0.1.00000000}.{c}", _HEADSET, "Active")])
        self._p(bc, "get_default_capture_name", return_value=_HEADSET)
        # The SELECTED device is still the NotPresent one — the old answer.
        self._p(bc, "get_current_mic_name", return_value=f"[3] {_DESK}")
        out = self._mic()
        self.assertNotIn("listening on Desk Mic", out)
        self.assertIn("isn't connected", out)
        self.assertIn("Windows' default microphone is Headset Mic", out)

    def test_live_device_wins_over_the_selected_one(self):
        bc = self.bc
        bc._live_capture_device[0] = {"index": 5, "name": _HEADSET,
                                      "requested": None, "via_default": True,
                                      "at": 0.0}
        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "MICROPHONE_INDEX", None)
        self._p(bc, "PREFERRED_INPUT_DEVICES", [])
        self._p(bc, "get_capture_endpoints", return_value=[
            ("{0.0.1.00000000}.{c}", _HEADSET, "Active")])
        self._p(bc, "get_default_capture_name", return_value=_HEADSET)
        self._p(bc, "get_current_mic_name", return_value=f"[3] {_DESK}")
        self.assertEqual(self._mic(), "I'm listening on Headset Mic, sir.")


if __name__ == "__main__":
    unittest.main()
