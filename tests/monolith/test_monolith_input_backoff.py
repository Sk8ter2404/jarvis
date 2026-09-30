"""Monolith wiring for the capture-open backoff (R10, 2026-09-29).

THE LIVE INCIDENT (session_2026-09-29_18-32-31.log). A USB hub reset took the
desk mic's endpoint away and the re-enumeration that followed left PortAudio
with NO input device. Every record_speech() call then failed at once — the
cached index (-9999 'A device ID has been used that is out of range'), then its
retry with the system default ('Error querying device -1') — returned None,
and the main loop called it again immediately: ~200 failures a second, each
logged by logging.exception as a two-part chained traceback. 5,859 failures and
11,718 tracebacks in two 20-second bursts, a 20.7 MB log.

These tests drive the REAL record_speech (and the real _capture_utterance /
_drain_injected_command where the main loop's drain is the point) against:

  * a fake PortAudio (the monolith's ``sd``) whose device TABLE is frozen
    until _initialize(), like the real one, and whose InputStream fails the
    way the live one did;
  * a fake Windows (is the desk mic present? the default endpoints);
  * ONE frozen clock for everything: _refresh_devices' time.time(), the
    audio-flap governor and the backoff's own clock — advanced only by the
    backoff's sleep hook, so every attempt time is exact.

The loop drivers are bounded by a call count, so a tree WITHOUT the backoff —
where the clock never moves because nothing ever waits — fails its assertions
instead of hanging.

No real audio device, MMDevice API or PortAudio. Device names are SYNTHETIC.

    python -m unittest tests.monolith.test_monolith_input_backoff
"""
from __future__ import annotations

import builtins
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

MIC = "{0.0.1.00000000}.{r10-desk-mic}"
SPK = "{0.0.0.00000000}.{r10-speakers}"
NAMES = {MIC: "Microphone (R10 Desk Mic)", SPK: "Speakers (R10 USB Speakers)"}
T0 = 50000.0
MME_2 = ("Error opening InputStream: Unanticipated host error [PaErrorCode "
         "-9999]: 'A device ID has been used that is out of range for your "
         "system.' [MME error 2]")


class _Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def rel(self):
        return round(self.t - T0, 3)


class _Default:
    """sd.default: .device resolves from the FROZEN table (-1 = none)."""

    def __init__(self, world):
        self._w = world
        self.hostapi = 0

    @property
    def device(self):
        rows = self._w.table
        ins = [i for i, r in enumerate(rows) if r["max_input_channels"] > 0]
        outs = [i for i, r in enumerate(rows) if r["max_output_channels"] > 0]
        return (ins[0] if ins else -1, outs[0] if outs else -1)


class _World:
    """Windows' truth (is the desk mic there?) plus a PortAudio stand-in
    whose device TABLE is frozen until _initialize()."""

    class PortAudioError(Exception):
        pass

    def __init__(self, clock, mic_present=False):
        self.clock = clock
        self.mic_present = mic_present
        self.table = self._rows()
        self.inits_at: list = []        # clock.rel() of every _initialize()
        self.opens: list = []           # (clock.rel(), requested device, ok)
        self.fail_left = None           # int: "-1" that many times, then open
        self.error = None               # callable: raise THIS on every open
        self.owner_flag = None          # record_speech's cell, set by the test
        self.claimed_at_teardown: list = []
        self.default = _Default(self)
        world = self

        class _Stream:
            def __init__(s, *a, device=None, **k):
                world._open(s, device)

            def start(s):
                pass

            def stop(s):
                pass

            def close(s):
                pass
        self.InputStream = _Stream

    # ── Windows ─────────────────────────────────────────────────────────
    def eps(self):
        return (SPK, MIC if self.mic_present else None)

    def active(self):
        return frozenset({SPK, MIC} if self.mic_present else {SPK})

    # ── PortAudio ───────────────────────────────────────────────────────
    def _rows(self):
        rows = []
        if self.mic_present:
            rows.append({"name": NAMES[MIC], "hostapi": 0,
                         "max_input_channels": 1, "max_output_channels": 0,
                         "default_samplerate": 48000})
        rows.append({"name": NAMES[SPK], "hostapi": 0,
                     "max_input_channels": 0, "max_output_channels": 2,
                     "default_samplerate": 48000})
        return rows

    def query_devices(self, idx=None, kind=None, **_kw):
        if idx is None:
            return [dict(r) for r in self.table]
        if isinstance(idx, int) and 0 <= idx < len(self.table):
            return dict(self.table[idx])
        raise self.PortAudioError(f"Error querying device {idx}")

    def check_input_settings(self, **_kw):
        return None

    def _terminate(self):
        if self.owner_flag is not None:
            self.claimed_at_teardown.append(bool(self.owner_flag[0]))

    def _initialize(self):
        self.inits_at.append(self.clock.rel())
        self.table = self._rows()

    def _open(self, stream, device):
        requested = device
        err = None
        if self.error is not None:
            err = self.error()
        elif self.fail_left is not None:
            if self.fail_left > 0:
                self.fail_left -= 1
                err = self.PortAudioError("Error querying device -1")
        else:
            if device is None:
                device = self.default.device[0]
                if device < 0:
                    err = self.PortAudioError("Error querying device -1")
            if err is None:
                if (not 0 <= device < len(self.table)
                        or self.table[device]["max_input_channels"] <= 0):
                    err = self.PortAudioError(
                        f"Error querying device {device}")
                elif not self.mic_present:
                    err = self.PortAudioError(MME_2)
        self.opens.append((self.clock.rel(), requested, err is None))
        if err is not None:
            raise err
        stream.device = device


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    MIC_PRESENT_AT_START = False

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.clock = _Clock()
        self.world = _World(self.clock, self.MIC_PRESENT_AT_START)
        self.printed: list = []
        self.tracebacks: list = []
        self.events: list = []          # [(rel time, callable)], fired by sleep
        c, w = self.clock, self.world
        self._p(builtins, "print",
                side_effect=lambda *a, **k: self.printed.append(
                    " ".join(str(x) for x in a)))

        def _log(*a, **k):
            self.tracebacks.append((a, k))
        self._p(bc.logging, "exception", side_effect=_log)
        self._p(bc.logging, "error",
                side_effect=lambda *a, **k: (_log(*a, **k)
                                             if k.get("exc_info") else None))
        self._p(bc, "sd", w)
        self._p(bc, "_win_default_endpoints", side_effect=lambda: w.eps())
        self._p(bc, "_win_active_endpoint_ids", side_effect=lambda: w.active())
        self._p(bc, "_win_endpoint_friendly_name",
                side_effect=lambda eid: NAMES.get(eid))
        self.rec = [False]
        w.owner_flag = self.rec
        self._p(bc, "_record_speech_active", self.rec)
        self._p(bc, "_pathb_mic_active", [False])
        self._p(bc, "_ambient_stream_active", [0])
        self._p(bc, "_diag_capture_active", [0])
        self._p(bc, "_enroll_capture_active", [0])
        self.tts = [False]
        self._p(bc, "_tts_playback_active", self.tts)
        self._p(bc, "_pa_close_pending", [0])
        self._p(bc, "MICROPHONE_INDEX", None)
        self._p(bc, "SPEAKER_INDEX", None)
        self._p(bc, "PREFERRED_INPUT_DEVICES", [])
        self._p(bc, "PREFERRED_OUTPUT_DEVICES", [])
        self._p(bc, "proactive_announce", return_value=True)
        # ONE clock: _refresh_devices' wall clock, the flap governor and the
        # backoff all read it; only the backoff's sleep moves it.
        self._p(bc.time, "time", side_effect=c)
        self._p(bc._audio_flap, "_clock", c)
        self._p(bc, "_input_backoff_clock", c, create=True)
        self._p(bc, "_input_backoff_sleep", side_effect=self._sleep,
                create=True)
        # record_speech reaches its open; nothing real underneath.
        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "_safe_close_stream", lambda s: None)
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        self._p(bc, "_usb_storm_note_audio_drop", lambda *a, **k: None)
        self._p(bc, "_heartbeat")
        # A successful open bails out of the capture loop at once (the
        # pattern the other record_speech tests use); _heartbeat is patched,
        # so nothing clears it.
        bc._watchdog_reset_signal.set()
        self.addCleanup(bc._watchdog_reset_signal.clear)
        self.tmp = tempfile.mkdtemp(prefix="r10_backoff_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.inject = os.path.join(self.tmp, "injected_commands.json")
        self._p(bc, "INJECTED_COMMANDS_PATH", self.inject)
        self._p(bc, "PENDING_SPEECH_PATH",
                os.path.join(self.tmp, "pending_speech.json"))
        mods = mock.patch.dict(bc.sys.modules)
        mods.start()
        self.addCleanup(mods.stop)
        bc.sys.modules.pop("skill_wake_listener", None)
        backoff = getattr(bc, "_input_open_backoff", None)
        if backoff is not None:
            backoff.reset()
        bc._pa_defer_logged[0] = None
        bc._device_cache.update({
            "in": None, "out": None, "checked_at": 0.0,
            "last_in_name": None, "last_out_name": None,
            "last_in_index": None, "last_out_index": None,
            "last_in_endpoint": None, "last_out_endpoint": None,
            "last_default_endpoints": None,
            "last_devices_signature": None, "last_reenum_at": c.t,
            "last_active_endpoints": None, "pa_active_endpoints": None,
            "pa_stale_logged": False, "last_seen_default_endpoints": None,
            "default_candidate": None, "reenum_requested": False,
        })
        self._settle()

    def _settle(self):
        """Two quiet refresh passes (baselines), then: the last
        re-enumeration was long ago, and nothing has been logged."""
        bc = self.bc
        for _ in range(2):
            bc._device_cache["checked_at"] = 0.0
            bc._refresh_devices()
        bc._device_cache["last_reenum_at"] = self.clock.t - 100.0
        bc._device_cache["reenum_requested"] = False
        bc._pa_defer_logged[0] = None
        self.world.inits_at.clear()
        self.world.claimed_at_teardown.clear()
        self.printed.clear()

    def _sleep(self, dt):
        self.clock.t = round(self.clock.t + float(dt), 6)
        for ev in sorted(self.events, key=lambda e: e[0]):
            if self.clock.rel() >= ev[0]:
                self.events.remove(ev)
                ev[1]()

    def _listen(self, until, max_calls=600, stop_on_open=True):
        """The main loop listening over and over: record_speech(timeout=20)
        until the clock passes T0 + ``until`` (or an open succeeds). Bounded
        by ``max_calls`` — without a backoff the clock never moves."""
        calls = 0
        while self.clock.rel() < until and calls < max_calls:
            calls += 1
            self.bc.record_speech(timeout=20)
            if stop_on_open and self.world.opens and self.world.opens[-1][2]:
                break
        return calls

    def _times(self, ok=None):
        return [t for t, _d, good in self.world.opens
                if ok is None or good is ok]

    def _lines(self, needle):
        return [ln for ln in self.printed if needle in ln]


class NoInputDeviceOutageTests(_Base):
    """'Error querying device -1' twenty times, then the device opens."""

    def test_attempts_follow_the_backoff_and_the_open_resets_it(self):
        self.world.fail_left = 20
        self._listen(until=300)
        times = self._times()
        self.assertEqual(len(times), 21, times[:40])
        self.assertTrue(self.world.opens[-1][2], "never recovered")
        gaps = [round(b - a, 3) for a, b in zip(times, times[1:])]
        # 0.5 -> 1 -> 2 -> 5 s, capped; one attempt is pulled in to the
        # moment the spaced re-enumeration becomes permissible (10.0).
        self.assertEqual(times[:8], [0.0, 0.5, 1.5, 3.5, 8.5, 10.0, 15.0, 20.0])
        self.assertTrue(all(0.5 <= g <= 5.0 for g in gaps), gaps)
        self.assertFalse(self.bc._input_open_backoff.active,
                         "a successful open must end the episode")
        self.assertEqual(self.bc._input_open_backoff.remaining(self.clock.t),
                         0.0)

    def test_one_line_per_episode_a_summary_and_no_tracebacks(self):
        self.world.fail_left = 20
        self._listen(until=300)
        self.assertEqual(self.tracebacks, [],
                         "a known failure class must never log a traceback")
        first = self._lines("no input device —")
        self.assertEqual(len(first), 1, self.printed)
        self.assertIn("Error querying device -1", first[0])
        summary = self._lines("still no input device;")
        self.assertEqual(len(summary), 1, self.printed)   # at t = 60 s
        self.assertIn("16 attempts in 60 s", summary[0])
        back = self._lines("opened again after")
        self.assertEqual(len(back), 1, self.printed)
        self.assertIn("20 failed attempt(s) over 85 s", back[0])
        # record_speech's own per-(device, error) throttle stays quiet.
        self.assertEqual(self._lines("InputStream open failed on device"),
                         [l for l in first])
        # ...and no status line per attempt either.
        self.assertEqual(self._lines("identical failure(s)"), [])

    def test_re_enumeration_runs_in_the_first_failing_attempt(self):
        # No stream owner holds PortAudio: the no-device failure itself
        # performs the re-enumeration (the same safe path), not "whichever
        # refresh pass comes next".
        self.world.fail_left = 1000
        self.bc.record_speech(timeout=20)
        self.assertEqual(self.world.inits_at, [0.0],
                         "the first no-device failure did not re-enumerate")
        self.assertEqual(self.bc._device_cache["last_reenum_at"], self.clock.t)
        self.assertEqual(self._times(), [0.0])

    def test_re_enumerations_stay_spaced_and_never_under_its_own_claim(self):
        self.world.fail_left = 1000
        self._listen(until=61)
        at = self.world.inits_at
        self.assertEqual(at, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(self.world.claimed_at_teardown, [False] * len(at),
                         "PortAudio torn down while record_speech held its "
                         "claim")


class OwnerHoldsPortAudioTests(_Base):
    def test_re_enumeration_waits_for_the_owner_then_runs(self):
        # TTS is playing (the barge-in stream is live): tearing PortAudio down
        # now is the 0xc0000374 heap corruption. Deferred — then performed on
        # the first attempt after the owner releases.
        self.tts[0] = True
        self.world.fail_left = 1000
        self.events.append((2.0, lambda: self.tts.__setitem__(0, False)))
        self._listen(until=4.0)
        self.assertEqual(self._times()[:4], [0.0, 0.5, 1.5, 3.5])
        self.assertEqual(self.world.inits_at, [3.5])
        self.assertEqual(len(self._lines("TTS playback is live")), 1,
                         self.printed)
        self.assertEqual(self.world.claimed_at_teardown, [False])


class RecoveryResetTests(_Base):
    def test_a_new_outage_starts_again_at_half_a_second(self):
        w = self.world
        w.fail_left = 3
        self._listen(until=30)
        self.assertEqual(self._times(), [0.0, 0.5, 1.5, 3.5])
        w.fail_left = 2
        self._listen(until=30)
        self.assertEqual(self._times(), [0.0, 0.5, 1.5, 3.5, 3.5, 4.0, 5.0])
        self.assertEqual(self._times(ok=True), [3.5, 5.0])
        self.assertEqual(len(self._lines("no input device —")), 2)
        self.assertEqual(len(self._lines("opened again after")), 2)


class DeviceReturnsTests(_Base):
    def test_a_returning_device_is_tried_at_once_not_at_the_next_step(self):
        # TTS holds PortAudio until 8.8 s (so no re-enumeration has run), the
        # desk mic comes back at 8.9 s — inside the 5 s wait from 8.5 to 13.5.
        w = self.world
        self.tts[0] = True
        self.events.append((8.8, lambda: self.tts.__setitem__(0, False)))

        def _back():
            w.mic_present = True
        self.events.append((8.9, _back))
        self._listen(until=30)
        self.assertEqual(self._times(ok=False), [0.0, 0.5, 1.5, 3.5, 8.5])
        self.assertEqual(self._times(ok=True), [9.0],
                         "the device came back at 8.9 s; the next poll must "
                         "try it, not the 13.5 s step")
        self.assertEqual(self.world.opens[-1][1], 0)
        self.assertFalse(self.bc._input_open_backoff.active)


class StaysResponsiveTests(_Base):
    def _loop(self, until, max_calls=600):
        """The main loop's own shape: drain an injected command, then
        capture (inject or mic)."""
        bc = self.bc
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "resume_face_tracking")
        self._p(bc, "should_be_proactive", return_value=False)
        self._p(bc, "set_state")
        handled, calls = [], 0
        while self.clock.rel() < until and calls < max_calls:
            calls += 1
            inj = bc._drain_injected_command()
            cap = bc._capture_utterance(inj, {})
            if inj is not None:
                handled.append((self.clock.rel(), cap[0] if cap else None))
        return handled

    def _write_inject(self, text):
        with open(self.inject, "w", encoding="utf-8") as f:
            json.dump([text], f)

    def test_injected_command_is_drained_during_the_outage(self):
        self.world.fail_left = 10000
        self.events.append((9.6, lambda: self._write_inject("what time is it")))
        handled = self._loop(until=16)
        self.assertEqual(handled, [(9.6, "what time is it")],
                         "an injected command waited out the backoff (or was "
                         "never drained)")
        # The outage schedule carries on where it was.
        self.assertEqual(self._times()[:7],
                         [0.0, 0.5, 1.5, 3.5, 8.5, 10.0, 15.0])
        self.assertEqual(self.printed.count("Listening…"), 1,
                         "one status line per attempt is the spam this "
                         "fix removes")

    def test_work_already_pending_does_not_turn_into_a_hot_loop(self):
        # An inject file that is present when the wait begins (the loop top
        # just had its chance at it) must not make every call return at once.
        self.world.fail_left = 10000
        self.bc.record_speech(timeout=20)            # attempt at 0.0
        self._write_inject("stuck")
        self.bc.record_speech(timeout=20)
        self.assertEqual(self._times(), [0.0, 0.5])
        self.assertEqual(self.clock.rel(), 0.5)

    def test_the_callers_listen_budget_is_never_exceeded(self):
        self.world.fail_left = 10000
        self._listen(until=3.0)                      # 0, .5, 1.5, 3.5 -> 8.5
        self.assertEqual(self._times(), [0.0, 0.5, 1.5, 3.5])
        self.assertIsNone(self.bc.record_speech(timeout=2.0))
        self.assertEqual(self.clock.rel(), 5.5)
        self.assertEqual(len(self._times()), 4, "no attempt before it was due")


class UnexpectedErrorTests(_Base):
    def test_an_unexpected_type_keeps_its_traceback_once_and_backs_off(self):
        self.world.error = lambda: RuntimeError("synthetic driver fault")
        self._listen(until=30)
        self.assertEqual(self._times()[:9],
                         [0.0, 0.5, 1.5, 3.5, 8.5, 13.5, 18.5, 23.5, 28.5])
        self.assertEqual(len(self.tracebacks), 1, self.tracebacks)
        args, kwargs = self.tracebacks[0]
        exc_info = kwargs.get("exc_info")
        self.assertTrue(exc_info, "the unexpected error lost its traceback")
        self.assertIsInstance(exc_info[1], RuntimeError)
        self.assertEqual(self._lines("no input device"), [])


class NoIdenticalRetryTests(_Base):
    def test_the_system_default_retry_is_never_the_same_open(self):
        # The cached index fails (-9999), the system default is tried ONCE;
        # when the cached device already was the default, it is not repeated.
        w = self.world
        w.mic_present = True
        w.table = w._rows()
        self.bc._device_cache.update({"in": 0, "checked_at": self.clock.t})
        w.mic_present = False                  # Windows lost it; table stale
        self._p(self.bc, "_refresh_devices")   # keep the stale cached index
        self.bc.record_speech(timeout=20)
        self.assertEqual([d for _t, d, _ok in w.opens], [0, None],
                         "the cached index, then ONE system-default retry")
        # The stale index is still named once (the retry is news)...
        self.assertEqual(len(self._lines("retrying with the system default")),
                         1, self.printed)
        w.opens.clear()
        self.bc._device_cache["in"] = None
        w.table = [r for r in w.table if r["max_input_channels"] == 0]
        self.bc._input_open_backoff.reset()
        self.bc.record_speech(timeout=20)
        self.assertEqual(len(w.opens), 1, "device=None retried as device=None")


if __name__ == "__main__":
    unittest.main()
