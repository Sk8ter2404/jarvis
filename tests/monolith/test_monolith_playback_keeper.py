"""Monolith wiring of the playback keeper (PLAYBACK_KEEPER) and the primed
play stream (PLAYBACK_PRIMED_STREAM), 2026-10-05.

The keeper itself (core/playback_keeper.py) is pinned CI-light in
tests/test_playback_keeper.py. This file pins what the monolith promises
around it:

  * the keeper's owner cell is an owner like any other: _pa_streams_live
    counts it, _refresh_devices defers its destructive reinit for it (and
    asks it to step aside when it is the only owner in the way), and the
    self-diagnostic's mirror of the owner list names it;
  * 'off' is the old path: no holder, the reaper polls at 50 ms;
  * 'on': a reply, a line and a playback each hold the speaker, the play
    waits (bounded) for a keeper open in flight, the turn line notes
    keeper=1/0, the reaper polls at 10 ms and a barge-in still cuts within
    one 50 ms slice, on ONE reaper thread;
  * the keeper never opens a real device under a test run, and its stream
    is a plain zero-filled OutputStream;
  * the primed stream plays every frame, ends with CallbackStop (never during
    priming) and goes to the same reaper; its -9999 fallback is sd.play's.

No real audio device is touched: sd is a fake everywhere, and
_keeper_open_stream refuses under a test run.

    python -m unittest tests.monolith.test_monolith_playback_keeper
"""
from __future__ import annotations

import inspect
import re
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

try:
    import numpy as np
except ImportError:      # light runner: every class below is skipped
    np = None


class _FakeKeeper:
    """Stands in for bc._playback_keeper: records calls, never threads."""

    def __init__(self, live=True):
        self.enabled = True
        self.live = live
        self.calls = []
        self._tok = 0

    def set_enabled(self, on):
        self.calls.append(("set_enabled", bool(on)))
        self.enabled = bool(on)

    def begin(self, device=None):
        self._tok += 1
        self.calls.append(("begin", device, self._tok))
        return self._tok

    def end(self, tok):
        self.calls.append(("end", tok))

    def wait_settled(self, timeout):
        self.calls.append(("wait_settled", timeout))
        return True

    def is_live(self):
        return self.live

    def note_play(self):
        self.calls.append(("note_play",))

    def request_yield(self):
        self.calls.append(("request_yield",))

    def reinit_done(self):
        self.calls.append(("reinit_done",))

    def names(self):
        return [c[0] for c in self.calls]


class _PlayStream:
    """A fake play stream: records (call, thread) and goes inactive on
    abort/close, or after ``natural_s`` (a clip that ends by itself)."""

    def __init__(self, natural_s=None):
        self.calls = []
        self._active = True
        self._t0 = time.monotonic()
        self.natural_s = natural_s
        self.latency = 0.182

    @property
    def active(self):
        if (self.natural_s is not None
                and time.monotonic() - self._t0 >= self.natural_s):
            self._active = False
        return self._active

    def abort(self, ignore_errors=True):
        self.calls.append(("abort", threading.get_ident(), time.monotonic()))
        self._active = False

    def stop(self, ignore_errors=True):
        self.calls.append(("stop", threading.get_ident(), time.monotonic()))

    def close(self, ignore_errors=True):
        self.calls.append(("close", threading.get_ident(), time.monotonic()))
        self._active = False


class _FakeSd:
    """sounddevice stand-in for the playback body."""

    class PortAudioError(Exception):
        pass

    class CallbackStop(Exception):
        pass

    def __init__(self, stream=None):
        self.stream = stream or _PlayStream(natural_s=0.0)
        self.play_calls = []
        self.stop_calls = 0
        self.output_streams = []
        self._last_callback = None

    def play(self, audio, sr, device=None):
        self.play_calls.append((len(audio), sr, device))

    def get_stream(self):
        return self.stream

    def stop(self, *a, **k):
        self.stop_calls += 1


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, target, name, *args, **kwargs):
        p = mock.patch.object(target, name, *args, **kwargs)
        val = p.start()
        self.addCleanup(p.stop)
        return val


class OwnerCellTests(_Base):
    def _run_refresh(self):
        """One forced _refresh_devices pass with a fake PortAudio; returns
        (terminate_called, printed)."""
        bc = self.bc
        printed = []
        terminated = {"called": False}
        with mock.patch.object(bc.sd, "_terminate",
                               side_effect=lambda: terminated.__setitem__(
                                   "called", True)), \
                mock.patch.object(bc.sd, "_initialize"), \
                mock.patch.object(bc.sd, "query_devices",
                                  return_value={"name": "FakeDev"}), \
                mock.patch.object(bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(bc, "SPEAKER_INDEX", None), \
                mock.patch.object(bc, "_pick_device",
                                  return_value=(0, "FakeDev")), \
                mock.patch("builtins.print",
                           side_effect=lambda *a, **k: printed.append(
                               " ".join(str(x) for x in a))):
            bc._device_cache["checked_at"] = 0.0
            bc._refresh_devices(force=True)
        return terminated["called"], printed

    def test_pa_streams_live_counts_the_keeper_cell(self):
        bc = self.bc
        with bc._mic_lock:
            self.assertFalse(bc._pa_streams_live())
            bc._tts_keeper_active[0] = True
            self.assertTrue(bc._pa_streams_live())

    def test_refresh_defers_for_the_keeper_and_asks_it_to_step_aside(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        with bc._mic_lock:
            bc._tts_keeper_active[0] = True
        terminated, printed = self._run_refresh()
        self.assertFalse(terminated, "sd._terminate under the keeper's live "
                                     "stream is the 0xc0000374")
        self.assertFalse(bc._pa_reinit_active[0])
        self.assertIn("playback keeper holds the speaker", "\n".join(printed))
        self.assertEqual(fake.names(), ["request_yield"])

    def test_another_owner_keeps_its_own_reason_and_no_yield(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        with bc._mic_lock:
            bc._tts_keeper_active[0] = True
            bc._tts_playback_active[0] = True
        terminated, printed = self._run_refresh()
        self.assertFalse(terminated)
        self.assertIn("TTS playback is live", "\n".join(printed))
        self.assertNotIn("request_yield", fake.names(),
                         "the keeper is not the only owner in the way")

    def test_a_reinit_that_ran_tells_the_keeper(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        terminated, _ = self._run_refresh()
        self.assertTrue(terminated)
        self.assertEqual(fake.names(), ["reinit_done"])

    def test_self_diagnostic_mirror_names_every_owner_cell(self):
        # Stale-duplicate guard: skills/self_diagnostic._mic_owned mirrors
        # _pa_streams_live minus the probe's own _diag_capture_active cell.
        bc = self.bc
        src = (inspect.getsource(bc._pa_streams_live)
               + inspect.getsource(bc._pa_mic_capture_live))
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.strip().startswith(("#", '"""')))
        owners = set(re.findall(r"\b(_[a-z_]+)\[0\]", code))
        owners.discard("_diag_capture_active")
        import skills.self_diagnostic as sdiag
        mirror = set(re.findall(r'_owner_flag\("(_[a-z_]+)"\)',
                                inspect.getsource(sdiag)))
        self.assertIn("_tts_keeper_active", owners)
        self.assertTrue(owners <= mirror, owners - mirror)


class KeeperModeTests(_Base):
    def test_off_holds_nothing_and_polls_at_50ms(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        self._p(bc, "PLAYBACK_KEEPER", "off")
        self.assertEqual(bc._keeper_begin(6), 0)
        self.assertNotIn("begin", fake.names())
        self.assertEqual(bc._reap_poll_s(), 0.05)
        fake.enabled = False
        bc._keeper_before_open()
        self.assertEqual(fake.names(), ["set_enabled"])

    def test_on_holds_the_speaker_and_polls_at_10ms(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        self._p(bc, "PLAYBACK_KEEPER", "on")
        self._p(bc, "_is_staging", lambda: False)
        self._p(bc, "_tts_layer", None)
        bc._tts_muted[0] = False
        self.assertEqual(bc._keeper_begin(6), 1)
        self.assertIn(("begin", 6, 1), fake.calls)
        self.assertEqual(bc._reap_poll_s(), 0.01)
        bc._device_cache["out"] = 4
        bc._keeper_begin()                       # the last picked speaker
        self.assertIn(("begin", 4, 2), fake.calls)

    def test_nothing_is_held_while_muted_or_staging(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        self._p(bc, "PLAYBACK_KEEPER", "on")
        layer = mock.Mock()
        layer.is_muted.return_value = False
        self._p(bc, "_tts_layer", layer)
        with mock.patch.object(bc, "_is_staging", lambda: True):
            self.assertEqual(bc._keeper_begin(6), 0)
        self._p(bc, "_is_staging", lambda: False)
        bc._tts_muted[0] = True
        self.assertEqual(bc._keeper_begin(6), 0)
        bc._tts_muted[0] = False
        layer.is_muted.return_value = True        # MUTE_TTS
        self.assertEqual(bc._keeper_begin(6), 0)
        self.assertNotIn("begin", fake.names())

    def test_mode_is_case_and_space_tolerant(self):
        bc = self.bc
        self._p(bc, "_playback_keeper", _FakeKeeper())
        for v, on in ((" ON ", True), ("On", True), ("off", False),
                      ("", False), (None, False), ("shadow", False)):
            with mock.patch.object(bc, "PLAYBACK_KEEPER", v):
                self.assertEqual(bc._keeper_mode_on(), on, v)


class KeeperStreamTests(_Base):
    def test_never_opens_a_real_device_under_a_test_run(self):
        bc = self.bc
        fake_sd = mock.Mock()
        self._p(bc, "sd", fake_sd)
        with self.assertRaises(RuntimeError):
            bc._keeper_open_stream(6)
        fake_sd.OutputStream.assert_not_called()

    def test_stream_is_a_started_zero_filled_output_stream(self):
        bc = self.bc
        fake_sd = mock.Mock()
        self._p(bc, "sd", fake_sd)
        self._p(bc, "_keeper_test_run", lambda: False)
        st = bc._keeper_open_stream(6)
        kw = fake_sd.OutputStream.call_args.kwargs
        self.assertEqual((kw["device"], kw["channels"], kw["dtype"],
                          kw["samplerate"]), (6, 1, "float32", 24000))
        self.assertIs(kw["callback"], bc._keeper_zeros)
        st.start.assert_called_once()
        fake_sd.play.assert_not_called()          # never sd.play's slot

    def test_failed_start_closes_and_reraises(self):
        bc = self.bc
        fake_sd = mock.Mock()
        fake_sd.OutputStream.return_value.start.side_effect = OSError("-9999")
        self._p(bc, "sd", fake_sd)
        self._p(bc, "_keeper_test_run", lambda: False)
        with self.assertRaises(OSError):
            bc._keeper_open_stream(6)
        fake_sd.OutputStream.return_value.close.assert_called_once()

    def test_callback_writes_silence(self):
        out = np.ones((624, 1), dtype=np.float32)
        self.bc._keeper_zeros(out, 624, None, None)
        self.assertFalse(out.any())

    def test_claim_and_release_use_the_owner_gate(self):
        bc = self.bc
        self.assertTrue(bc._keeper_claim())
        self.assertTrue(bc._tts_keeper_active[0])
        bc._keeper_release()
        self.assertFalse(bc._tts_keeper_active[0])
        # A reinit in flight refuses the claim (the gate's bounded 1 s wait).
        with bc._mic_lock:
            bc._pa_reinit_active[0] = True
        try:
            t0 = time.monotonic()
            self.assertFalse(bc._keeper_claim())
            self.assertLess(time.monotonic() - t0, 3.0)
        finally:
            with bc._mic_lock:
                bc._pa_reinit_active[0] = False
        self.assertFalse(bc._tts_keeper_active[0])

    def test_real_keeper_end_to_end_with_fakes(self):
        # The REAL keeper object wired the monolith's way, with a fake
        # stream factory: a reply holds the speaker and the owner cell, the
        # cell drops after the close.
        bc = self.bc
        from core import playback_keeper as pk
        opened = []

        class _St:
            def abort(self, ignore_errors=True):
                opened.append("abort")

            def close(self, ignore_errors=True):
                opened.append("close")

        def _open(dev):
            opened.append(("open", dev))
            return _St()

        k = pk.PlaybackKeeper(open_stream=_open,
                              claim=lambda: bc._keeper_claim(),
                              release=lambda: bc._keeper_release(),
                              log=lambda line: None, linger_s=0.05)
        self.addCleanup(k.shutdown)
        k.set_enabled(True)
        tok = k.begin(6)
        end = time.monotonic() + 2.0
        while not k.is_live() and time.monotonic() < end:
            time.sleep(0.005)
        self.assertTrue(k.is_live())
        self.assertTrue(bc._tts_keeper_active[0])
        with bc._mic_lock:
            self.assertTrue(bc._pa_streams_live())
        k.end(tok)
        end = time.monotonic() + 2.0
        while bc._tts_keeper_active[0] and time.monotonic() < end:
            time.sleep(0.005)
        self.assertFalse(bc._tts_keeper_active[0])
        self.assertEqual(opened, [("open", 6), "abort", "close"])


class _PlayBase(_Base):
    def setUp(self):
        bc = self.bc
        layer = mock.Mock()
        layer.is_muted.return_value = False
        self._p(bc, "_tts_layer", layer)
        self._p(bc, "_audio_ducker", mock.Mock())
        self._p(bc, "BARGE_IN_ENABLED", False)
        self._p(bc, "ROBOT_ENABLED", False)
        self._p(bc, "get_output_device", return_value=6)
        self._p(bc, "_write_hud_state")
        self._p(bc, "_feed_playback_reference")
        self._p(bc, "_is_staging", lambda: False)
        bc._tts_muted[0] = False
        bc._tts_current_text[0] = ""


class PlaybackBodyTests(_PlayBase):
    def test_on_the_play_holds_the_speaker_and_waits_for_an_open(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper(live=True))
        self._p(bc, "PLAYBACK_KEEPER", "on")
        order = []
        fake_sd = _FakeSd()
        orig_play = fake_sd.play
        fake_sd.play = lambda a, sr, device=None: (order.append("sd.play"),
                                                  orig_play(a, sr, device))
        self._p(bc, "sd", fake_sd)
        stats = []
        self._p(bc, "_tt_note_stat",
                side_effect=lambda n, v: stats.append((n, v)))
        bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        names = fake.names()
        self.assertEqual(fake_sd.play_calls, [(240, 24000, 6)])
        self.assertIn(("begin", 6, 1), fake.calls)
        self.assertLess(names.index("begin"), names.index("wait_settled"))
        self.assertIn(("end", 1), fake.calls)
        self.assertIn(("keeper", 1), stats)
        self.assertIn("note_play", names)
        self.assertFalse(bc._tts_playback_active[0])

    def test_keeper_not_live_is_noted_zero(self):
        bc = self.bc
        self._p(bc, "_playback_keeper", _FakeKeeper(live=False))
        self._p(bc, "PLAYBACK_KEEPER", "on")
        self._p(bc, "sd", _FakeSd())
        stats = []
        self._p(bc, "_tt_note_stat",
                side_effect=lambda n, v: stats.append((n, v)))
        bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        self.assertIn(("keeper", 0), stats)

    def test_off_is_the_old_path(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        fake.enabled = False
        self._p(bc, "PLAYBACK_KEEPER", "off")
        fake_sd = _FakeSd()
        self._p(bc, "sd", fake_sd)
        stats = []
        self._p(bc, "_tt_note_stat",
                side_effect=lambda n, v: stats.append((n, v)))
        bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        self.assertEqual(fake_sd.play_calls, [(240, 24000, 6)])
        self.assertNotIn("begin", fake.names())
        self.assertNotIn("wait_settled", fake.names())
        self.assertFalse([s for s in stats if s[0] == "keeper"])

    def test_a_muted_play_holds_nothing(self):
        bc = self.bc
        fake = self._p(bc, "_playback_keeper", _FakeKeeper())
        self._p(bc, "PLAYBACK_KEEPER", "on")
        bc._tts_layer.is_muted.return_value = True
        self._p(bc, "sd", _FakeSd())
        bc.play_with_lipsync(np.zeros(24, dtype=np.float32), 24000)
        self.assertNotIn("begin", fake.names())

    def _barge(self, keeper_mode):
        """play_with_lipsync on a worker against a long active clip; an
        accepted barge-in ~60 ms in. Returns (stream, caller ident, ms from
        the interrupt to the reaper's abort)."""
        bc = self.bc
        self._p(bc, "_playback_keeper", _FakeKeeper())
        self._p(bc, "PLAYBACK_KEEPER", keeper_mode)
        stream = _PlayStream(natural_s=None)
        fake_sd = _FakeSd(stream)
        self._p(bc, "sd", fake_sd)
        self._p(bc, "_barge_in_wake_enabled", return_value=True)
        caller = []

        def _worker():
            caller.append(threading.get_ident())
            bc.play_with_lipsync(np.zeros(24000 * 2, dtype=np.float32),
                                 24000)

        w = threading.Thread(target=_worker, daemon=True)
        w.start()
        end = time.monotonic() + 2.0
        while not bc._tts_playback_active[0] and time.monotonic() < end:
            time.sleep(0.005)
        time.sleep(0.06)
        t_cut = time.monotonic()
        self.assertTrue(bc.request_tts_interrupt(source="test"))
        w.join(5.0)
        self.assertFalse(w.is_alive())
        aborts = [c for c in stream.calls if c[0] == "abort"]
        self.assertEqual(len(aborts), 1, stream.calls)
        self.assertEqual(fake_sd.stop_calls, 0)
        return stream, caller[0], (aborts[0][2] - t_cut) * 1000.0

    def test_barge_in_cuts_within_one_slice_on_one_reaper_thread(self):
        stream, caller, ms = self._barge("on")
        idents = {c[1] for c in stream.calls}
        self.assertEqual(len(idents), 1, stream.calls)
        self.assertNotIn(caller, idents)
        # One 10 ms poll plus scheduling; the deterministic poll interval is
        # pinned below, so this bound only has to catch a lost cut.
        self.assertLess(ms, 250.0)

    def _poll_intervals(self, mode):
        """The reaper's sleeps for a clip that ends after three polls."""
        bc = self.bc
        self._p(bc, "PLAYBACK_KEEPER", mode)
        stream = _PlayStream(natural_s=None)
        polls = []

        def _sleep(s):
            polls.append(s)
            if len(polls) >= 3:
                stream._active = False
        # bc.time only (a wrapper around the real module), so no other
        # thread's sleep is touched.
        fake_time = mock.Mock(wraps=time)
        fake_time.sleep.side_effect = _sleep
        with mock.patch.object(bc, "time", fake_time):
            done = threading.Event()
            bc._reap_playback(stream, done, 1.0)
        self.assertTrue(done.is_set())
        self.assertEqual([c[0] for c in stream.calls], ["stop", "close"])
        return polls

    def test_reaper_polls_at_10ms_with_the_keeper_on(self):
        self.assertEqual(self._poll_intervals("on"), [0.01, 0.01, 0.01])

    def test_reaper_polls_at_50ms_with_the_keeper_off(self):
        self.assertEqual(self._poll_intervals("off"), [0.05, 0.05, 0.05])


class HolderSiteTests(_Base):
    def test_dispatch_holds_the_speaker_for_the_whole_turn(self):
        bc = self.bc
        events = []
        self._p(bc, "_keeper_begin",
                side_effect=lambda *a: (events.append("begin"), 7)[1])
        self._p(bc, "_keeper_end", side_effect=lambda t: events.append(
            ("end", t)))
        self._p(bc, "_run_llm_dispatch_body",
                side_effect=lambda text: (events.append("body"), "ok")[1])
        self.assertEqual(bc._run_llm_dispatch("what time is it"), "ok")
        self.assertEqual(events, ["begin", "body", ("end", 7)])

    def test_dispatch_releases_the_speaker_when_the_body_raises(self):
        bc = self.bc
        ended = []
        self._p(bc, "_keeper_begin", return_value=9)
        self._p(bc, "_keeper_end", side_effect=ended.append)
        self._p(bc, "_run_llm_dispatch_body",
                side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            bc._run_llm_dispatch("hello")
        self.assertEqual(ended, [9])

    def test_speak_holds_from_before_the_render_to_after_the_last_line(self):
        bc = self.bc
        events = []
        self._p(bc, "_keeper_begin",
                side_effect=lambda *a: (events.append("begin"), 5)[1])
        self._p(bc, "_keeper_end",
                side_effect=lambda t: events.append(("end", t)))
        self._p(bc, "synthesise", side_effect=lambda t: (
            events.append("render"), (np.zeros(10, dtype=np.float32),
                                      24000))[1])
        self._p(bc, "play_with_lipsync",
                side_effect=lambda a, sr: events.append("play"))
        self._p(bc, "_sentence_tts_plan", return_value=None)
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_is_staging", lambda: False)
        bc._tts_muted[0] = False
        self.assertTrue(bc._speak("The weather is fine, sir."))
        self.assertEqual(events, ["begin", "render", "play", ("end", 5)])

    def test_speak_releases_the_speaker_when_playback_fails(self):
        bc = self.bc
        ended = []
        self._p(bc, "_keeper_begin", return_value=3)
        self._p(bc, "_keeper_end", side_effect=ended.append)
        self._p(bc, "synthesise", side_effect=lambda t: (
            np.zeros(10, dtype=np.float32), 24000))
        self._p(bc, "play_with_lipsync",
                side_effect=RuntimeError("PortAudio reinit hung"))
        self._p(bc, "_sentence_tts_plan", return_value=None)
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_session_start_time", time.time() - 3600)
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_is_staging", lambda: False)
        bc._tts_muted[0] = False
        with mock.patch("builtins.print"):
            self.assertFalse(bc._speak("The weather is fine, sir."))
        self.assertEqual(ended, [3])

    def test_teardown_latches_the_keeper_off(self):
        from core import actions as A
        bc = mock.Mock()
        bc._mic_lock = threading.Lock()
        bc._pa_streams_live.return_value = False
        with mock.patch("builtins.print"):
            A._release_audio_streams(bc, budget_s=0.5)
        bc._playback_keeper_shutdown.assert_called_once_with()

    def test_turn_flags_name_both_settings(self):
        self.assertIn("PLAYBACK_KEEPER", self.bc._TURN_FLAG_KEYS)
        self.assertIn("PLAYBACK_PRIMED_STREAM", self.bc._TURN_FLAG_KEYS)


class PrimedStreamTests(_Base):
    def _status(self, priming):
        return mock.Mock(priming_output=priming)

    def test_callback_plays_every_frame_then_stops_after_priming(self):
        bc = self.bc
        self._p(bc, "sd", _FakeSd())
        data = np.arange(1, 1001, dtype=np.float32).reshape(-1, 1)
        cb = bc._primed_play_callback(data)
        got = []
        # Priming: 8 x 100 frames inside start() - never a stop, even when
        # the data would run out (it does not here).
        for _ in range(8):
            out = np.full((100, 1), -1.0, dtype=np.float32)
            cb(out, 100, None, self._status(True))
            got.append(out.copy())
        # Real callbacks: 2 more blocks of data, the last ends the stream.
        out = np.full((100, 1), -1.0, dtype=np.float32)
        cb(out, 100, None, self._status(False))
        got.append(out.copy())
        out = np.full((150, 1), -1.0, dtype=np.float32)
        with self.assertRaises(bc.sd.CallbackStop):
            cb(out, 150, None, self._status(False))
        got.append(out.copy())
        flat = np.concatenate(got).reshape(-1)
        np.testing.assert_array_equal(flat[:1000], np.arange(1, 1001))
        self.assertFalse(flat[1000:].any(), "past the end must be silence")

    def test_a_line_shorter_than_the_priming_never_stops_while_priming(self):
        bc = self.bc
        self._p(bc, "sd", _FakeSd())
        data = np.ones((150, 1), dtype=np.float32)
        cb = bc._primed_play_callback(data)
        for _ in range(8):                         # priming: no raise
            cb(np.empty((100, 1), dtype=np.float32), 100, None,
               self._status(True))
        with self.assertRaises(bc.sd.CallbackStop):
            cb(np.empty((100, 1), dtype=np.float32), 100, None,
               self._status(False))

    def test_open_primes_with_the_callback_and_keeps_the_dtype(self):
        bc = self.bc
        fake_sd = mock.Mock()
        self._p(bc, "sd", fake_sd)
        st = bc._open_primed_stream(np.zeros(480, dtype=np.float64), 24000, 6)
        kw = fake_sd.OutputStream.call_args.kwargs
        self.assertTrue(kw["prime_output_buffers_using_stream_callback"])
        self.assertEqual((kw["device"], kw["channels"], kw["dtype"],
                          kw["samplerate"]), (6, 1, "float32", 24000))
        st.start.assert_called_once()
        fake_sd.play.assert_not_called()
        bc._open_primed_stream(np.zeros(480, dtype=np.int16), 24000, None)
        self.assertEqual(fake_sd.OutputStream.call_args.kwargs["dtype"],
                         "int16")

    def test_failed_start_closes_and_reraises(self):
        bc = self.bc
        fake_sd = mock.Mock()
        fake_sd.OutputStream.return_value.start.side_effect = OSError("x")
        self._p(bc, "sd", fake_sd)
        with self.assertRaises(OSError):
            bc._open_primed_stream(np.zeros(48, dtype=np.float32), 24000, 6)
        fake_sd.OutputStream.return_value.close.assert_called_once()


class PrimedPlaybackTests(_PlayBase):
    def test_primed_play_goes_to_the_same_reaper_not_sd_play(self):
        bc = self.bc
        self._p(bc, "PLAYBACK_KEEPER", "off")
        self._p(bc, "PLAYBACK_PRIMED_STREAM", True)
        own = _PlayStream(natural_s=0.0)
        opened = []
        self._p(bc, "_open_primed_stream",
                side_effect=lambda a, sr, dev: (opened.append(dev), own)[1])
        fake_sd = _FakeSd()
        fake_sd.get_stream = mock.Mock(
            side_effect=AssertionError("sd.get_stream must not be read"))
        self._p(bc, "sd", fake_sd)
        bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        self.assertEqual(opened, [6])
        self.assertEqual(fake_sd.play_calls, [])
        self.assertEqual([c[0] for c in own.calls], ["stop", "close"])
        self.assertFalse(bc._tts_playback_active[0])

    def test_primed_open_failure_falls_back_to_the_default_device(self):
        bc = self.bc
        self._p(bc, "PLAYBACK_PRIMED_STREAM", True)
        own = _PlayStream(natural_s=0.0)
        fake_sd = _FakeSd()
        self._p(bc, "sd", fake_sd)
        opened = []

        def _open(a, sr, dev):
            opened.append(dev)
            if dev == 6:
                raise fake_sd.PortAudioError("-9999")
            return own

        self._p(bc, "_open_primed_stream", side_effect=_open)
        self._p(bc, "_usb_storm_note_audio_drop")
        with mock.patch("builtins.print"):
            bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        self.assertEqual(opened, [6, None])
        self.assertEqual(fake_sd.play_calls, [])
        self.assertEqual([c[0] for c in own.calls], ["stop", "close"])


if __name__ == "__main__":
    unittest.main()
