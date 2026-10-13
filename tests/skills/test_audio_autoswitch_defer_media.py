"""Output switches wait while media plays (2026-10-12). A video pauses when its
output device changes; the 10-09 census saw ~16 such switches in 10 days.
No real device/COM/SMTC: everything is mocked, names are synthetic."""
from __future__ import annotations

import unittest
from unittest import mock

from audio import audio_switch as A


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _daemon(playing, defer=True, **kw):
    d = A.AudioAutoSwitch("Wireless Headset", fallback="Speakers",
                          defer_during_media=defer,
                          media_playing=lambda: playing[0], **kw)
    clk = _Clock()
    d._now = clk
    return d, clk


class DeferDuringMediaTests(unittest.TestCase):
    def _run(self, fn):
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="SPK"), \
             mock.patch.object(A, "set_default_render", return_value=True) as sdr, \
             mock.patch.object(A, "_log") as log:
            fn(sdr, log)

    def test_deferred_while_playing_then_switches_after_quiet(self):
        playing = [True]
        d, clk = _daemon(playing)

        def body(sdr, log):
            self.assertIsNone(d.tick(was_on=False))
            sdr.assert_not_called()
            self.assertEqual(sum("deferring" in str(c) for c in log.call_args_list), 1)
            clk.t += 30
            d.tick()                       # still playing: still waiting, no 2nd log
            sdr.assert_not_called()
            self.assertEqual(sum("deferring" in str(c) for c in log.call_args_list), 1)
            playing[0] = False
            clk.t += 1
            d.tick()                       # paused only 0s: quiet window not met
            sdr.assert_not_called()
            clk.t += 6
            self.assertEqual(d.tick(), "to_headset")
            sdr.assert_called_once_with("HS")
            self.assertIsNone(d._pending_out)
        self._run(body)

    def test_not_playing_switches_immediately(self):
        d, clk = _daemon([False], defer_quiet_s=0.0)
        self._run(lambda sdr, log: (self.assertEqual(d.tick(was_on=False), "to_headset"),
                                    sdr.assert_called_once()))

    def test_setting_off_is_old_behaviour(self):
        d, clk = _daemon([True], defer=False)
        self._run(lambda sdr, log: (self.assertEqual(d.tick(was_on=False), "to_headset"),
                                    sdr.assert_called_once()))

    def test_bounded_wait_switches_anyway(self):
        d, clk = _daemon([True], defer_max_s=60.0)

        def body(sdr, log):
            d.tick(was_on=False)
            sdr.assert_not_called()
            clk.t += 61
            self.assertEqual(d.tick(), "to_headset")
            sdr.assert_called_once()
        self._run(body)

    def test_flip_back_cancels_pending(self):
        playing = [True]
        d, clk = _daemon(playing)
        d.tick  # noqa
        with mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="SPK"), \
             mock.patch.object(A, "set_default_render", return_value=True) as sdr, \
             mock.patch.object(A, "_log"):
            with mock.patch.object(A, "headset_powered", return_value=True):
                d.tick(was_on=False)
            with mock.patch.object(A, "headset_powered", return_value=False):
                d.tick()
            self.assertIsNone(d._pending_out)
            playing[0] = False
            clk.t += 60
            with mock.patch.object(A, "headset_powered", return_value=False):
                d.tick()
            sdr.assert_not_called()

    def test_mic_path_unaffected(self):
        d, clk = _daemon([True], follow_mic=True)
        d.mic_fallback = "Desk"
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="SPK"), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "_log"), \
             mock.patch.object(d, "_follow_mic_state", return_value="mic") as fm:
            d.tick(was_on=False)
            fm.assert_called_once()
            self.assertEqual(d.last_mic_label, "mic")

    def test_probe_failure_never_blocks(self):
        def boom():
            raise RuntimeError("smtc")
        d = A.AudioAutoSwitch("H", defer_during_media=True, media_playing=boom, defer_quiet_s=0.0)
        d._now = _Clock()
        self._run(lambda sdr, log: (d.tick(was_on=False), sdr.assert_called_once()))
        # probe failure counts as not playing; quiet window still applies
