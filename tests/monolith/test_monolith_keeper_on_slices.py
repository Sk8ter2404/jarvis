"""The audio-path monolith tests again, at the SHIPPED default
PLAYBACK_KEEPER='on' (review fix, 2026-10-05).

tests/_monolith_harness.py runs every monolith test with the keeper OFF (its
thread opens asynchronously and could outlive a test's patch of ``bc.sd``),
so without this file the live default would be covered only by the keeper's
own tests - the green-by-knob shape. Each class below re-runs an existing
class unchanged, with:

  * PLAYBACK_KEEPER = 'on', a REAL PlaybackKeeper wired exactly like the
    monolith's (open_stream / claim / release through the monolith's own
    helpers and owner cell), fresh per test so no back-off or device state
    leaks between tests;
  * the native edge faked: ``_keeper_open_stream`` returns a zero stream that
    touches no device (the real one refuses under a test run anyway);
  * ``_keeper_audible`` = the real check minus the harness's JARVIS_STAGING=1
    (tray mute and MUTE_TTS still hold nothing), so the keeper really engages:
    every reply, line and playback holds the speaker, the reaper polls at
    10 ms, each play waits for / marks its open against the keeper, and the
    refresh deny chain sees the keeper's cell.

The keeper is shut down - and its close waited for - before the harness
restores the globals, so nothing carries into the next test.

Three re-run tests have a different outcome with the keeper on BY DESIGN,
and override just that test with the keeper-on expectation (review
2026-10-09: every monolith class was re-run with the keeper on; these were
the only differences): the two sec3 headset barge-in tests (a line cut
before its open is never opened) and R1PlaybackOpenTests' slow-claim timing
(the keeper's own claim must not tick the test's shared fake clock).

    python -m unittest tests.monolith.test_monolith_keeper_on_slices
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:      # light runner: every class below is skipped
    np = None

# Modules, not classes: a TestCase class imported into this namespace would
# be collected (and run) a second time with the keeper OFF.
from tests.monolith import test_monolith_after_reply as _after_reply
from tests.monolith import test_monolith_runtime_bugfixes as _bugfixes
from tests.monolith import test_monolith_sec1 as _sec1
from tests.monolith import test_monolith_sec3 as _sec3
from tests.monolith import test_monolith_self_echo as _self_echo
from tests.monolith import test_monolith_sentence_tts as _sentence
from tests.monolith import test_monolith_turn_timing as _turn_timing
from tests._monolith_harness import MonolithGlobalsTestCase


class _ZeroStream:
    """The keeper's stream at the native edge: plays nothing, opens nothing."""

    def __init__(self, device):
        self.device = device
        self.closed = False

    def abort(self, ignore_errors=True):
        pass

    def close(self, ignore_errors=True):
        self.closed = True


class _KeeperOn:
    """Mixin: see the module docstring. Must come FIRST in the bases."""

    KEEPER_OPEN_S = 0.02     # the keeper's open, off the critical path

    def setUp(self):
        bc = self.bc
        from core import playback_keeper as pk
        bc.PLAYBACK_KEEPER = "on"          # the harness restores it after
        opened = self._keeper_opened = []
        lock = threading.Lock()

        def _open(device):
            time.sleep(self.KEEPER_OPEN_S)
            st = _ZeroStream(device)
            with lock:
                opened.append(st)
            return st

        def _audible():
            # bc._keeper_audible without the staging check (the harness sets
            # JARVIS_STAGING=1 for every test): tray mute and MUTE_TTS still
            # hold nothing.
            try:
                if bc._tts_muted[0]:
                    return False
                return bc._self_echo_audible()
            except Exception:
                return False

        keeper = pk.PlaybackKeeper(
            open_stream=lambda device: bc._keeper_open_stream(device),
            claim=lambda: bc._keeper_claim(),
            release=lambda: bc._keeper_release(),
            log=lambda line: None)
        self._keeper = keeper
        # The media gate's playback-meter probe (pycaw) runs on a thread
        # started at each capture end; the keeper's few ms per play give it
        # time to reach the real audio session API inside the self-echo
        # tests, which the hermetic guard refuses. Unknown (None) is its
        # documented "could not read" answer.
        for name, val in (("_keeper_open_stream", _open),
                          ("_keeper_audible", _audible),
                          ("_playback_keeper", keeper),
                          ("_media_probe_read", lambda: None)):
            p = mock.patch.object(bc, name, val)
            p.start()
            self.addCleanup(p.stop)
        # Registered last, so it runs FIRST: the keeper's close returns while
        # the fakes above are still in place.
        self.addCleanup(self._quiesce_keeper)
        super().setUp()

    def _quiesce_keeper(self):
        quiet = self._keeper.shutdown(wait_s=3.0)
        if not quiet:
            self.fail("the keeper's close did not return within 3 s")


class PlaybackReaperKeeperOnTests(_KeeperOn, _sec3.PlaybackReaperTests):
    pass


class BargeInKeeperOnTests(_KeeperOn, _sec1.BargeInTests):
    pass


class RefreshDevicesReinitGuardKeeperOnTests(
        _KeeperOn, _bugfixes.RefreshDevicesReinitGuardTests):
    pass


class PlayWithLipsyncEndpointSwapKeeperOnTests(
        _KeeperOn, _bugfixes.PlayWithLipsyncEndpointSwapTests):
    pass


class PaTeardownGateKeeperOnTests(_KeeperOn, _bugfixes.PaTeardownGateTests):
    pass


class PaAbandonedCloseGateKeeperOnTests(
        _KeeperOn, _bugfixes.PaAbandonedCloseGateTests):
    """One difference, by design: the play's keeper is still lingering when
    the abandoned close completes, so the next pass defers for the KEEPER
    (the only owner left in the way), asks it to step aside, and the reinit
    runs on the pass after it (live: one DEVICE_CHECK_INTERVAL, 4 s, later).
    _run_refresh models that following pass - and only when the keeper
    really yielded, so every other deferral is asserted exactly as before."""

    def _run_refresh(self, picks=None):
        bc = self.bc
        yields = self._keeper.yields
        terminated, printed, pick = super()._run_refresh(picks)
        if terminated or self._keeper.yields == yields:
            return terminated, printed, pick
        end = time.monotonic() + 2.0
        while bc._tts_keeper_active[0] and time.monotonic() < end:
            time.sleep(0.005)
        self.assertFalse(bc._tts_keeper_active[0],
                         "the keeper did not step aside for the reinit")
        terminated, more, pick = super()._run_refresh(picks)
        return terminated, printed + more, pick


class PlayStreamFenceKeeperOnTests(_KeeperOn, _bugfixes.PlayStreamFenceTests):
    pass


class PipelinedSpeakKeeperOnTests(_KeeperOn, _sentence.PipelinedSpeakTests):
    pass


class RealPlaybackKeeperOnTests(_KeeperOn, _sentence.RealPlaybackTests):
    pass


class R1ReaperMarkKeeperOnTests(_KeeperOn, _turn_timing.R1ReaperMarkTests):
    pass


class AfterReplyBargeKeeperOnTests(_KeeperOn, _after_reply.BargeTests):
    pass


class SelfEchoLiveSequenceKeeperOnTests(_KeeperOn,
                                        _self_echo.LiveSequenceTests):
    pass


class _CutBeforeOpenMixin:
    """The two sec3 headset barge-in tests run the barge watch synchronously
    (_ImmediateThread), so the interrupt is already set BEFORE the line's
    open. With the keeper on, a line cut before its open is never opened
    (CutBeforeOpenTests in test_monolith_playback_keeper) - so the expected
    outcome differs BY DESIGN: no sd.play(), no stream for the reaper to
    abort or close, sd.stop() still never called, the listener still opened
    and closed, the playback cell still dropped. (Review 2026-10-09: these
    were the keeper-on differences found by re-running every monolith class
    with the keeper on; recorded here instead of left unrun.)"""

    def _headset_cut(self, fake_sd, barge_stream, extra):
        bc = self.bc
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        bc._barge_in_interrupted = True
        patches = [
            mock.patch.object(bc, "sd", fake_sd),
            mock.patch.object(bc, "_tts_layer", fake_layer),
            mock.patch.object(bc, "_audio_ducker", mock.Mock()),
            mock.patch.object(bc, "get_output_device", return_value=1),
            mock.patch.object(bc, "_write_hud_state"),
            mock.patch.object(bc, "_feed_playback_reference"),
            mock.patch.object(bc.threading, "Thread", _sec3._ImmediateThread),
            mock.patch.object(bc.time, "sleep"),
            mock.patch.object(bc, "BARGE_IN_ENABLED", True),
            mock.patch.object(bc, "ROBOT_ENABLED", False),
            mock.patch.object(bc, "is_using_headset", return_value=True),
            mock.patch.object(bc, "_start_barge_in_listener",
                              return_value=barge_stream),
        ] + extra
        started = [p.start() for p in patches]
        try:
            with mock.patch("builtins.print"):
                bc.play_with_lipsync(np.zeros(48, dtype=np.float32), 24000)
        finally:
            for p in patches:
                p.stop()
        return started


class PlayWithLipsyncExtraPathsKeeperOnTests(
        _KeeperOn, _CutBeforeOpenMixin,
        _sec3.PlayWithLipsyncExtraPathsTests):

    def test_barge_in_headset_path_opens_and_closes_listener(self):
        fake_sd = mock.Mock()
        stream = _sec3._wire_fake_stream(fake_sd, active=True)
        barge_stream = mock.Mock()
        scs_patch = mock.patch.object(self.bc, "_safe_close_stream")
        started = self._headset_cut(fake_sd, barge_stream, [scs_patch])
        fake_sd.play.assert_not_called()          # cut before the open
        fake_sd.get_stream.assert_not_called()
        stream.abort.assert_not_called()
        stream.close.assert_not_called()
        fake_sd.stop.assert_not_called()
        started[-1].assert_called_once_with(barge_stream)
        self.assertFalse(self.bc._tts_playback_active[0])


class PlayWithLipsyncDefensiveHandlerKeeperOnTests(
        _KeeperOn, _CutBeforeOpenMixin,
        _sec3.PlayWithLipsyncDefensiveHandlerTests):

    def test_barge_watch_routed_abort_exception_swallowed(self):
        fake_sd = mock.Mock()
        stream = _sec3._wire_fake_stream(fake_sd, active=True)
        stream.abort.side_effect = RuntimeError("abort boom")
        self._headset_cut(fake_sd, mock.Mock(), [
            mock.patch.object(self.bc, "_safe_close_stream")])
        fake_sd.play.assert_not_called()          # nothing left to abort
        stream.abort.assert_not_called()
        stream.close.assert_not_called()
        fake_sd.stop.assert_not_called()
        self.assertFalse(self.bc._tts_playback_active[0])


class R1PlaybackOpenKeeperOnTests(_KeeperOn, _turn_timing.R1PlaybackOpenTests):

    def test_the_setup_before_the_duck_is_timed_too(self):
        # The parent's slow claim advances the shared fake clock on EVERY
        # _pa_claim_owner - with the keeper on that includes the keeper's
        # own claim on its thread (during the play's wait for its open), one
        # extra 50 ms tick (610 vs 560). Advance it for the play's cell only:
        # everything the PLAY does before its open is still timed the same.
        bc = self.bc
        claim = bc._pa_claim_owner

        def slow_claim(cell, *a, **k):
            if cell is bc._tts_playback_active:
                self.clock.advance(0.050)
            return claim(cell, *a, **k)

        def slow_device():
            self.clock.advance(0.300)        # a device refresh / reinit
            return 1

        self._p(bc, "_pa_claim_owner", side_effect=slow_claim)
        self._p(bc, "get_output_device", side_effect=slow_device)
        self._turn()
        self._play()
        self.assertEqual(self._emit()["play_open_ms"], "560")
        self.assertEqual(len(self._keeper_opened), 1)   # it really held


class KeeperEngagedTests(_KeeperOn, MonolithGlobalsTestCase):
    """Guard against this file going green for the wrong reason: the keeper
    must really hold the speaker under the mixin."""

    def test_the_keeper_really_engages_in_a_play(self):
        bc = self.bc
        import numpy as np
        stream = _sec3._ReaperFakeStream(active=False)
        fake_sd = _sec3._ReaperFakeSd(stream)
        layer = mock.Mock()
        layer.is_muted.return_value = False
        for name, val in (("sd", fake_sd), ("_tts_layer", layer),
                          ("_audio_ducker", mock.Mock()),
                          ("BARGE_IN_ENABLED", False),
                          ("ROBOT_ENABLED", False),
                          ("get_output_device", lambda: 1),
                          ("_write_hud_state", lambda **k: None),
                          ("_feed_playback_reference", lambda *a: None)):
            p = mock.patch.object(bc, name, val)
            p.start()
            self.addCleanup(p.stop)
        bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        self.assertEqual(len(self._keeper_opened), 1)
        self.assertTrue(bc._tts_keeper_active[0], "lingering after the play")
        self.assertEqual(bc._reap_poll_s(), 0.01)


if __name__ == "__main__":
    unittest.main()
