"""what_microphone must name the device the LIVE capture stream is reading.

Live, v2.0.115 (2026-09-29, typed turn): "what microphone are you using" ->
[ACTION: what_microphone] -> "I'm listening on <a configured mic>, sir." At
that moment Windows reported that microphone NotPresent, and the capture was
running on a different device. The log said why:
    [audio] the Windows default audio endpoint moved to a device that is not
            in PortAudio's current enumeration …
    [audio] device drift detected but a mic/audio stream is live … deferring
            PortAudio reinit
i.e. PortAudio's device list was frozen (the re-enumeration is deferred while
a stream is live) and the skill read the SELECTED device's name out of it —
what the next open would ask for — never the live stream's device, and never
whether Windows still had it.

The fix answers from the monolith's get_live_capture_device() (what the stream
really opened on), checks it against Windows' own endpoint state
(get_capture_endpoints), reports a missing preferred mic, and names Windows'
live default when the stream cannot be vouched for. These tests pin that with
SYNTHETIC device tables only — no real device is read.
"""
from __future__ import annotations

import re
import sys
import types
import unittest
from unittest import mock

from core.failure_markers import FAILURE_MARKERS
from tests._skill_harness import load_skill_isolated

_DESK = "Microphone (Desk Mic)"
_HEADSET = "Headset Microphone (Headset Mic)"
_SPEAKERS = "Speakers (Desk Speakers)"

_CAP = "{0.0.1.00000000}"


def _ep(name, state):
    return (f"{_CAP}.{{{name}}}", name, state)


def _friendly(raw):
    m = re.search(r"\(([^)]*)\)", raw or "")
    return m.group(1).strip() if m else (raw or "")


def _fake_bc(*, live=None, endpoints=None, default=None, selected=None,
             preferred=None, mic_index=None, pa_names=None, disabled=False,
             speaker=_SPEAKERS):
    """A stand-in monolith exposing exactly what the skill reads. Anything left
    as None is simply absent, which is how an older monolith looks."""
    bc = types.ModuleType("bobert_companion")
    bc._friendly_device_name = _friendly
    bc._mic_input_disabled = lambda: disabled
    bc.get_current_speaker_name = lambda: speaker
    if selected is not None:
        bc.get_current_mic_name = lambda: selected
    if live is not None:
        rec = live if isinstance(live, dict) else {"index": 3, "name": live}
        bc.get_live_capture_device = lambda: dict(rec, live=True)
    if endpoints is not None:
        bc.get_capture_endpoints = lambda: list(endpoints)
    if default is not None:
        bc.get_default_capture_name = lambda: default
    bc.PREFERRED_INPUT_DEVICES = list(preferred or [])
    bc.MICROPHONE_INDEX = mic_index
    names = pa_names or {}
    bc.sd = types.SimpleNamespace(
        query_devices=lambda i: {"name": names[i]})
    return bc


class _Base(unittest.TestCase):
    def _mic(self, bc, action="what_microphone"):
        patcher = mock.patch.dict(sys.modules, {"bobert_companion": bc})
        patcher.start()
        self.addCleanup(patcher.stop)
        _mod, actions = load_skill_isolated("audio_devices")
        return actions[action]("")

    def assertSpokenVerbatim(self, out):
        """An HONEST answer must not carry a failure marker, or the monolith
        drops it from the verbatim speak path and re-prompts the model."""
        low = out.lower()
        hits = [m for m in FAILURE_MARKERS if m in low]
        self.assertEqual(hits, [], f"{out!r} would be treated as a failure")


class TheLiveIncidentTests(_Base):
    def test_notpresent_preferred_mic_is_never_claimed(self):
        """The 2026-09-29 shape: the preferred mic is NotPresent, the capture
        stream is still opened on its old PortAudio slot, and Windows' default
        has moved to another device."""
        bc = _fake_bc(
            live=_DESK, selected=f"[3] {_DESK}", preferred=["Desk Mic"],
            endpoints=[_ep(_DESK, "NotPresent"), _ep(_HEADSET, "Active")],
            default=_HEADSET)
        out = self._mic(bc)
        self.assertNotIn("listening on Desk Mic", out)
        self.assertEqual(
            out,
            "Your preferred microphone, Desk Mic, isn't connected, sir, yet my "
            "capture stream is still opened on it, so I'm not certain which "
            "microphone is actually hearing you. Windows' default microphone "
            "is Headset Mic.")
        self.assertSpokenVerbatim(out)

    def test_names_the_live_device_not_the_selected_one(self):
        """get_current_mic_name() (the SELECTED device) says Desk Mic; the
        stream actually opened on the headset. The answer is the headset."""
        bc = _fake_bc(
            live=_HEADSET, selected=f"[3] {_DESK}",
            endpoints=[_ep(_DESK, "Active"), _ep(_HEADSET, "Active")])
        self.assertEqual(self._mic(bc), "I'm listening on Headset Mic, sir.")

    def test_missing_preferred_mic_is_reported_with_the_live_device(self):
        bc = _fake_bc(
            live=_HEADSET, selected=f"[3] {_DESK}", preferred=["Desk Mic"],
            endpoints=[_ep(_DESK, "NotPresent"), _ep(_HEADSET, "Active")])
        out = self._mic(bc)
        self.assertEqual(
            out, "Your preferred microphone, Desk Mic, isn't connected, sir — "
                 "I'm listening on Headset Mic instead.")
        self.assertSpokenVerbatim(out)


class LiveDeviceTests(_Base):
    def test_stale_stream_without_a_preference(self):
        bc = _fake_bc(live=_DESK,
                      endpoints=[_ep(_DESK, "Unplugged"), _ep(_HEADSET, "Active")],
                      default=_HEADSET)
        out = self._mic(bc)
        self.assertEqual(
            out, "My capture stream is still opened on Desk Mic, sir, but "
                 "Windows reports it isn't connected, so I'm not certain which "
                 "microphone is actually hearing you. Windows' default "
                 "microphone is Headset Mic.")
        self.assertSpokenVerbatim(out)

    def test_device_windows_does_not_list_at_all_is_not_claimed(self):
        bc = _fake_bc(live=_DESK, endpoints=[_ep(_HEADSET, "Active")])
        out = self._mic(bc)
        self.assertNotIn("listening on Desk Mic", out)
        self.assertIn("not certain", out)

    def test_disabled_device_says_disabled(self):
        bc = _fake_bc(live=_DESK, endpoints=[_ep(_DESK, "Disabled")])
        self.assertIn("Windows reports it is disabled", self._mic(bc))

    def test_windows_unavailable_reports_the_stream_device(self):
        # No endpoint states (non-Windows host, COM failure): the stream's own
        # device is still the best evidence, and "unknown" is never "absent".
        bc = _fake_bc(live=_HEADSET)
        self.assertEqual(self._mic(bc), "I'm listening on Headset Mic, sir.")

    def test_mme_truncated_name_still_matches_its_endpoint(self):
        full = "Headset Microphone (Studio Headset Mic)"
        truncated = full[:31]
        bc = _fake_bc(live=truncated, endpoints=[_ep(full, "Active")])
        self.assertTrue(self._mic(bc).startswith("I'm listening on "))

    def test_sound_mapper_reads_the_windows_default(self):
        bc = _fake_bc(live="Microsoft Sound Mapper - Input",
                      endpoints=[_ep(_HEADSET, "Active")], default=_HEADSET)
        self.assertEqual(self._mic(bc), "I'm listening on Headset Mic, sir.")

    def test_unreadable_live_name_is_unknown_not_guessed(self):
        bc = _fake_bc(live={"index": 3, "name": ""}, selected=f"[3] {_DESK}",
                      endpoints=[_ep(_DESK, "Active")])
        out = self._mic(bc)
        self.assertIn("couldn't determine", out)
        self.assertNotIn("Desk Mic", out)

    def test_mic_input_switched_off(self):
        bc = _fake_bc(live=_HEADSET, disabled=True)
        self.assertEqual(self._mic(bc), "My microphone input is switched off, sir.")

    def test_pinned_index_preference_is_checked_by_name(self):
        bc = _fake_bc(live=_HEADSET, mic_index=4, pa_names={4: _DESK},
                      endpoints=[_ep(_DESK, "NotPresent"), _ep(_HEADSET, "Active")])
        self.assertTrue(self._mic(bc).startswith(
            "Your preferred microphone, Desk Mic, isn't connected, sir"))

    def test_present_preference_is_not_mentioned(self):
        bc = _fake_bc(live=_DESK, preferred=["Desk Mic"],
                      endpoints=[_ep(_DESK, "Active")])
        self.assertEqual(self._mic(bc), "I'm listening on Desk Mic, sir.")


class NoLiveStreamYetTests(_Base):
    """Before the first capture opens (or on an older monolith), the SELECTED
    device is all there is — worded as what it is, and presence-checked."""

    def test_selected_device_is_worded_as_selected(self):
        bc = _fake_bc(selected=f"[1] {_HEADSET}",
                      endpoints=[_ep(_HEADSET, "Active")])
        self.assertEqual(self._mic(bc), "I'm set to listen on Headset Mic, sir.")

    def test_selected_notpresent_device_is_flagged(self):
        bc = _fake_bc(selected=f"[3] {_DESK}",
                      endpoints=[_ep(_DESK, "NotPresent"), _ep(_HEADSET, "Active")],
                      default=_HEADSET)
        out = self._mic(bc)
        self.assertEqual(
            out, "I'm set to listen on Desk Mic, sir, but Windows reports it "
                 "isn't connected. Windows' default microphone is Headset Mic.")
        self.assertSpokenVerbatim(out)

    def test_system_default_names_the_windows_default(self):
        bc = _fake_bc(selected="(system default)", default=_HEADSET)
        self.assertEqual(
            self._mic(bc),
            "I'm set to listen on Windows' default microphone, Headset Mic, sir.")


class CombinedAnswerTests(_Base):
    def test_combined_answer_uses_the_live_microphone(self):
        bc = _fake_bc(live=_HEADSET, selected=f"[3] {_DESK}",
                      endpoints=[_ep(_HEADSET, "Active")])
        self.assertEqual(
            self._mic(bc, "audio_devices"),
            "I'm listening on Headset Mic and speaking through Desk Speakers, sir.")

    def test_combined_answer_keeps_the_honest_mic_sentence(self):
        bc = _fake_bc(live=_DESK, endpoints=[_ep(_DESK, "NotPresent")])
        out = self._mic(bc, "audio_devices")
        self.assertIn("not certain which microphone", out)
        self.assertTrue(out.endswith("I'm speaking through Desk Speakers, sir."),
                        out)


if __name__ == "__main__":
    unittest.main()
