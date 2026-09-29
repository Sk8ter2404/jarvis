"""Monolith wiring for audio-device flap damping + stable capture (2026-09-29).

THE LIVE INCIDENT (2026-09-29 15:20-15:29, owner away). The owner's USB desk
mic's Windows ENDPOINT went Active <-> NotPresent every ~20-40 s while its USB
device stayed connected; his wireless headset was powered off but its dongle
kept its endpoints Active. Windows bounced the default recording device
between the two and JARVIS:

  1. SPOKE 17 audio-device sentences in ~10 minutes to an empty room;
  2. re-picked its capture device on every bounce;
  3. opened a STALE PortAudio index right after the endpoint list changed:
     get_mic_buffer -> PaErrorCode -9999 'A device ID has been used that is
     out of range for your system.' [MME error 2]. PortAudio's MME host API
     stores each device's waveIn id at Pa_Initialize and Windows renumbers
     those ids when an endpoint comes or goes, so the frozen row for the
     headset mic named an id one past the end of the shrunken list.

These tests drive the REAL _refresh_devices against a fake PortAudio whose
device TABLE is frozen at _initialize() (as PortAudio's is) and a fake
Windows (the live endpoint set + default), on a hand-driven clock:

  * a flapping endpoint -> ONE plain sentence, `[audio-flap]` log lines, and
    nothing else spoken for the rest of the storm;
  * the index JARVIS would open always names the same device in Windows'
    live list as in PortAudio's table (never a stale index); a stale
    direction is re-enumerated first, and while that is deferred by a live
    stream the capture device is NOT re-picked;
  * hysteresis: a moved default is followed only after AUDIO_REPICK_STABLE_S,
    unless the device JARVIS is using is gone;
  * one audio-device sentence per AUDIO_ANNOUNCE_MIN_GAP_S, released by
    _speak_pending; a newer one replaces an unspoken queued one;
  * get_mic_buffer Path B resolves its mic BEFORE claiming, and a failed open
    drops the cached index, asks for a re-enumeration and logs once;
  * each knob (AUDIO_FLAP_WINDOW_S, AUDIO_FLAP_THRESHOLD,
    AUDIO_ANNOUNCE_MIN_GAP_S, AUDIO_REPICK_STABLE_S) is consumed.

No real audio device, MMDevice API or PortAudio. Device names are SYNTHETIC.

    python -m unittest tests.monolith.test_monolith_audio_flap
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

DESK = "{0.0.1.00000000}.{desk-mic}"
HS_MIC = "{0.0.1.00000000}.{headset-mic}"
HS_OUT = "{0.0.0.00000000}.{headset-out}"
SPK = "{0.0.0.00000000}.{speakers}"
ORDER = [DESK, HS_MIC, HS_OUT, SPK]          # inputs enumerate first
NAMES = {DESK: "Microphone (Desk Mic)",
         HS_MIC: "Headset Microphone (Wireless Headset)",
         HS_OUT: "Headphones (Wireless Headset)",
         SPK: "Speakers (USB Speakers)"}
DEAF = ("Sir, I may not be able to hear you. The 'Wireless Headset' headset "
        "measures powered off and there is no other recording device.")
CLEAR = ("Windows' default microphone is off the powered-off headset now, "
         "sir. I still cannot prove it is picking up sound until I hear you.")


class _World:
    """Windows' truth (Active endpoints, the default pair) plus a PortAudio
    stand-in whose TABLE is frozen until _initialize(), exactly like the real
    one. Used as the monolith's `sd`."""

    class PortAudioError(Exception):
        pass

    def __init__(self):
        self.active = set(ORDER)
        self.default_in = DESK
        self.default_out = SPK
        self.inits = 0
        self.default = mock.Mock()
        self.default.hostapi = 0
        self.default.device = (0, 3)
        self.table = self._rows()

    # ── Windows ─────────────────────────────────────────────────────────
    def _rows(self):
        rows = []
        for ep in ORDER:
            if ep not in self.active:
                continue
            is_in = ep.startswith("{0.0.1.")
            rows.append({"name": NAMES[ep], "hostapi": 0,
                         "max_input_channels": 1 if is_in else 0,
                         "max_output_channels": 0 if is_in else 2,
                         "default_samplerate": 48000})
        return rows

    def live_names(self):
        return [r["name"] for r in self._rows()]

    def frozen_names(self):
        return [r["name"] for r in self.table]

    def eps(self):
        return (self.default_out if self.default_out in self.active else None,
                self.default_in if self.default_in in self.active else None)

    def set_desk(self, present: bool):
        if present:
            self.active.add(DESK)
            self.default_in = DESK
        else:
            self.active.discard(DESK)
            self.default_in = HS_MIC

    # ── PortAudio (frozen at _initialize) ───────────────────────────────
    def query_devices(self, idx=None, **_kw):
        if idx is None:
            return [dict(r) for r in self.table]
        if isinstance(idx, int) and 0 <= idx < len(self.table):
            return dict(self.table[idx])
        raise self.PortAudioError(f"Error querying device {idx}")

    def check_input_settings(self, **_kw):
        return None

    def _terminate(self):
        pass

    def _initialize(self):
        self.inits += 1
        self.table = self._rows()


class _Clock:
    def __init__(self, t=100000.0):
        self.t = t

    def __call__(self):
        return self.t


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.world = _World()
        self.clock = _Clock()
        self.spoken: list[str] = []
        self.printed: list[str] = []
        self.mic_live = [False]
        w, c = self.world, self.clock
        # Every console line of the test (the governor's [audio-flap] lines
        # included) lands in self.printed, never on the runner's stdout.
        import builtins
        self._p(builtins, "print",
                side_effect=lambda *a, **k: self.printed.append(
                    " ".join(str(x) for x in a)))
        self._p(bc, "sd", w)
        self._p(bc, "_win_default_endpoints", side_effect=lambda: w.eps())
        # create=True / the getattr below: the SAME fixture drives an
        # unmodified tree too, so the behavioural tests fail there on their
        # assertions (a stale index adopted, a bounce followed, 17 sentences)
        # rather than on a missing name.
        self._p(bc, "_win_active_endpoint_ids",
                side_effect=lambda: frozenset(w.active), create=True)
        self._p(bc, "_win_endpoint_friendly_name",
                side_effect=lambda eid: NAMES.get(eid))
        self._p(bc, "_record_speech_active", self.mic_live)
        self._p(bc, "_pathb_mic_active", [False])
        self._p(bc, "_ambient_stream_active", [0])
        self._p(bc, "_diag_capture_active", [0])
        self._p(bc, "_enroll_capture_active", [0])
        self._p(bc, "_tts_playback_active", [False])
        self._p(bc, "_pa_close_pending", [0])
        self._p(bc, "MICROPHONE_INDEX", None)
        self._p(bc, "SPEAKER_INDEX", None)
        self._p(bc, "PREFERRED_INPUT_DEVICES", [])
        self._p(bc, "PREFERRED_OUTPUT_DEVICES", [])
        self._p(bc, "proactive_announce",
                side_effect=lambda m, *a, **k: self.spoken.append(m) or True)
        # The wall clock _refresh_devices reads and the governor's clock move
        # together, by hand.
        self._p(bc.time, "time", side_effect=c)
        if getattr(bc, "_audio_flap", None) is not None:
            self._p(bc._audio_flap, "_clock", c)
        bc.sys.modules.pop("skill_wake_listener", None)
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

    def _pass(self, dt: float = 4.0):
        """Advance the clock and run ONE refresh pass, then the drain's
        flush — what the main loop does between turns."""
        self.clock.t += dt
        self.bc._device_cache["checked_at"] = 0.0
        with mock.patch("builtins.print",
                        side_effect=lambda *a, **k: self.printed.append(
                            " ".join(str(x) for x in a))):
            self.bc._refresh_devices()
            flush = getattr(self.bc, "_audio_flap_flush", None)
            if flush is not None:
                flush()

    def _announce(self, msg, kind):
        """Submit a daemon sentence the way skills/audio_autoswitch does; on
        a tree without the governor that is a plain proactive_announce."""
        gate = getattr(self.bc, "_audio_device_announce", None)
        with mock.patch("builtins.print",
                        side_effect=lambda *a, **k: self.printed.append(
                            " ".join(str(x) for x in a))):
            if gate is None:
                self.bc.proactive_announce(msg, source="audio")
            else:
                gate(msg, kind)

    def _settle(self):
        """Boot + a few quiet passes on the desk mic."""
        for _ in range(3):
            self._pass()
        self.assertEqual(self.bc._device_cache["in"], 0)
        self.spoken.clear()
        self.printed.clear()

    def _index_is_live(self, idx) -> bool:
        """Does the index JARVIS would open name the SAME device in Windows'
        live list as in PortAudio's frozen table? False = a stale index."""
        live, frozen = self.world.live_names(), self.world.frozen_names()
        return (0 <= idx < len(live) and idx < len(frozen)
                and live[idx] == frozen[idx])


class FlapStormReplayTests(_Base):
    """Ten minutes of the 2026-09-29 flap: the desk mic's endpoint toggles
    every 28 s, the default follows it, the daemon adds its hearing alert
    and recovery line -- and no stream is live, so re-enumeration can run."""

    def _replay(self, *, minutes=10.0, period=28.0, check=None):
        self._settle()
        t_end = self.clock.t + minutes * 60.0
        next_flip = self.clock.t + period
        present = True
        while self.clock.t < t_end:
            self._pass()
            if self.clock.t >= next_flip:
                present = not present
                self.world.set_desk(present)
                next_flip += period
                self._pass(1.0)
                self._announce(CLEAR if present else DEAF,
                               "deaf-clear" if present else "deaf")
            if check is not None:
                check()

    def test_a_flap_storm_is_one_plain_sentence_not_seventeen(self):
        self._replay()
        flap = [m for m in self.spoken if "keeps dropping in and out" in m]
        self.assertLessEqual(len(self.spoken), 3, self.spoken)
        self.assertEqual(len(flap), 1, self.spoken)
        self.assertIn("the Desk Mic", flap[0])
        self.assertIn("stop announcing", flap[0])
        self.assertTrue(any(ln.lstrip().startswith("[audio-flap]")
                            for ln in self.printed), self.printed[:20])
        self.assertTrue(any("FLAPPING" in ln for ln in self.printed))

    def test_it_says_once_when_the_device_has_settled(self):
        self._replay(minutes=4.0)
        self.spoken.clear()
        self.world.set_desk(True)
        for _ in range(int(11 * 60 / 4)):        # 11 quiet minutes
            self._pass()
        settled = [m for m in self.spoken if "steady for 10 minutes" in m]
        self.assertEqual(len(settled), 1, self.spoken)
        self.assertIn("the Desk Mic", settled[0])

    def test_the_index_jarvis_would_open_is_never_stale(self):
        stale: list = []

        def check():
            for kind in ("in", "out"):
                idx = self.bc._device_cache.get(kind)
                if idx is not None and not self._index_is_live(idx):
                    stale.append((round(self.clock.t), kind, idx,
                                  self.world.frozen_names()[idx]
                                  if idx < len(self.world.table) else "?"))
        self._replay(minutes=3.0, check=check)
        self.assertEqual(stale, [], "an index from PortAudio's frozen table "
                                    "that Windows has since renumbered")
        self.assertGreater(self.world.inits, 3,
                           "each endpoint change must re-enumerate")

    def test_the_capture_device_follows_what_is_actually_there(self):
        self._settle()
        self.world.set_desk(False)
        self._pass()
        idx = self.bc._device_cache["in"]
        self.assertIsNotNone(idx)
        self.assertEqual(self.world.frozen_names()[idx], NAMES[HS_MIC],
                         "the desk mic is gone: follow at once, hysteresis "
                         "or not")
        self.assertTrue(self._index_is_live(idx))


class StaleEnumerationTests(_Base):
    def test_a_stale_direction_is_not_re_picked_while_a_stream_is_live(self):
        self._settle()
        inits = self.world.inits
        before = self.bc._device_cache["in"]
        self.mic_live[0] = True               # record_speech holds the mic
        self.world.set_desk(False)
        self._pass()
        self.assertEqual(self.world.inits, inits, "no teardown under a stream")
        self.assertEqual(self.bc._device_cache["in"], before,
                         "no index from the stale table may be adopted")
        self.assertFalse(any("mic →" in ln for ln in self.printed),
                         self.printed)
        self.assertTrue(any("indices are stale" in ln
                            for ln in self.printed), self.printed)
        stale_lines = sum("device indices are stale" in ln
                          for ln in self.printed)
        self._pass()
        self.assertEqual(sum("device indices are stale" in ln
                             for ln in self.printed), stale_lines,
                         "the stale state is logged once, not per pass")
        # The stream ends -> the very next pass re-enumerates and picks the
        # headset mic at its NEW index.
        self.mic_live[0] = False
        self._pass()
        self.assertEqual(self.world.inits, inits + 1)
        idx = self.bc._device_cache["in"]
        self.assertEqual(self.world.frozen_names()[idx], NAMES[HS_MIC])
        self.assertTrue(self._index_is_live(idx))

    def test_an_unchanged_endpoint_list_never_re_enumerates(self):
        # NEGATIVE CONTROL: the staleness check must not turn into a reinit
        # every 4 s.
        self._settle()
        inits = self.world.inits
        for _ in range(10):
            self._pass()
        self.assertEqual(self.world.inits, inits)

    def test_inconsistent_snapshots_are_treated_as_unknown(self):
        # The default endpoint is Active by definition; an Active set that
        # lacks it was read across a change and must not be believed.
        self._settle()
        inits = self.world.inits
        self._p(self.bc, "_win_active_endpoint_ids",
                return_value=frozenset({HS_OUT}))
        self._pass()
        self.assertEqual(self.world.inits, inits)
        self.assertFalse(any("[audio-flap]" in ln for ln in self.printed))


class RepickHysteresisTests(_Base):
    def _press(self):
        """A Stream-Deck-style move: the default mic goes to the headset
        while the desk mic stays Active."""
        self.world.default_in = HS_MIC

    def test_a_moved_default_is_followed_only_once_it_has_held(self):
        self._settle()
        self._press()
        self._pass()
        self.assertEqual(self.bc._device_cache["in"], 0,
                         "followed on the first pass: every bounce drags the "
                         "capture device along")
        self.assertTrue(any("waiting until it has held" in ln
                            for ln in self.printed), self.printed)
        self._pass()
        self.assertEqual(self.bc._device_cache["in"], 0)
        self._pass()                      # 8 s after it was first seen
        self.assertEqual(self.bc._device_cache["in"], 1)
        self.assertEqual(self.spoken, ["Switched to Wireless Headset, sir."])
        self.assertEqual(sum("waiting until it has held" in ln
                             for ln in self.printed), 1)

    def test_a_default_that_bounces_back_never_moves_jarvis(self):
        self._settle()
        self._press()
        self._pass()
        self.world.default_in = DESK
        self._pass()
        for _ in range(3):
            self._pass()
        self.assertEqual(self.bc._device_cache["in"], 0)
        self.assertEqual(self.spoken, [])
        self.assertTrue(any("moved back before it had held" in ln
                            for ln in self.printed), self.printed)

    def test_knob_zero_follows_on_the_first_pass(self):
        self._p(self.bc, "AUDIO_REPICK_STABLE_S", 0.0)
        self._settle()
        self._press()
        self._pass()
        self.assertEqual(self.bc._device_cache["in"], 1)

    def test_knob_is_the_hold_time(self):
        self._p(self.bc, "AUDIO_REPICK_STABLE_S", 30.0)
        self._settle()
        self._press()
        for _ in range(4):                # 12 s after first seen
            self._pass()
        self.assertEqual(self.bc._device_cache["in"], 0)
        for _ in range(5):                # 32 s
            self._pass()
        self.assertEqual(self.bc._device_cache["in"], 1)


class GovernorWiringTests(_Base):
    def test_one_sentence_per_gap_and_the_drain_releases_the_held_one(self):
        bc = self.bc
        self.assertTrue(bc._audio_device_announce("Switched to A, sir."))
        self.clock.t += 20
        bc._audio_device_announce("Switched to B, sir.")
        self.assertEqual(self.spoken, ["Switched to A, sir."])
        self.clock.t += 41
        tmp = tempfile.mkdtemp(prefix="jarvis_flap_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self._p(bc, "PENDING_SPEECH_PATH", os.path.join(tmp, "q.json"))
        self._p(bc, "_speak")
        with mock.patch("builtins.print"):
            bc._speak_pending()
        self.assertEqual(self.spoken[-1], "Switched to B, sir.",
                         "_speak_pending must release the held sentence")

    def test_governed_sentences_carry_the_supersede_tag(self):
        calls = []
        self._p(self.bc, "proactive_announce",
                side_effect=lambda *a, **k: calls.append((a, k)) or True)
        self.bc._audio_device_announce("Switched to A, sir.", "switch")
        self.bc._enqueue_device_announcement("wake word down", kind="other")
        self.assertEqual(calls[0][1].get("supersede"), "audio-device")
        self.assertEqual(calls[0][1].get("source"), "audio")
        self.assertNotIn("supersede", calls[1][1])

    def test_other_kind_is_never_swallowed_by_a_storm(self):
        for _ in range(3):
            self.bc._audio_flap_note("endpoint:x", "the Desk Mic")
        self.spoken.clear()
        self.bc._enqueue_device_announcement(
            "The wake-word detector dropped during a device change, sir.",
            kind="other")
        self.assertEqual(len(self.spoken), 1)

    def test_the_silent_mic_alert_is_quiet_inside_a_storm(self):
        bc = self.bc
        for _ in range(3):
            bc._audio_flap_note("endpoint:x", "the Desk Mic")
        self.spoken.clear()
        bc._silent_mic_warned[0] = False
        with mock.patch("builtins.print"):
            self.assertTrue(bc._report_silent_mic(45.0, 1000.0))
        self.assertEqual(self.spoken, [])

    def test_a_governor_error_fails_open(self):
        self._p(self.bc._audio_flap, "submit",
                side_effect=RuntimeError("boom"))
        with mock.patch("builtins.print"):
            self.bc._audio_device_announce(DEAF, "deaf")
        self.assertEqual(self.spoken, [DEAF])


class KnobConsumerTests(_Base):
    def test_threshold(self):
        self._p(self.bc, "AUDIO_FLAP_THRESHOLD", 5)
        for _ in range(3):
            self.bc._audio_flap_note("endpoint:x", "the Desk Mic")
        self.assertFalse(self.bc._audio_flap.storm_active())
        self.bc._audio_flap.reset()
        self.bc.AUDIO_FLAP_THRESHOLD = 3
        for _ in range(3):
            self.bc._audio_flap_note("endpoint:x", "the Desk Mic")
        self.assertTrue(self.bc._audio_flap.storm_active())

    def test_window(self):
        self._p(self.bc, "AUDIO_FLAP_WINDOW_S", 150.0)
        for _ in range(3):
            self.bc._audio_flap_note("endpoint:x", "the Desk Mic")
            self.clock.t += 100
        self.assertFalse(self.bc._audio_flap.storm_active())
        self.bc._audio_flap.reset()
        self.bc.AUDIO_FLAP_WINDOW_S = 300.0
        for _ in range(3):
            self.bc._audio_flap_note("endpoint:x", "the Desk Mic")
            self.clock.t += 100
        self.assertTrue(self.bc._audio_flap.storm_active())

    def test_min_gap(self):
        self._p(self.bc, "AUDIO_ANNOUNCE_MIN_GAP_S", 10.0)
        self.bc._audio_device_announce("Switched to A, sir.")
        self.clock.t += 30
        self.bc._audio_device_announce("Switched to B, sir.")
        self.assertEqual(len(self.spoken), 2)
        self.bc._audio_flap.reset()
        self.spoken.clear()
        self.bc.AUDIO_ANNOUNCE_MIN_GAP_S = 60.0
        self.bc._audio_device_announce("Switched to A, sir.")
        self.clock.t += 30
        self.bc._audio_device_announce("Switched to B, sir.")
        self.assertEqual(len(self.spoken), 1)


class ProactiveAnnounceSupersedeTests(MonolithGlobalsTestCase):
    """A newer audio-device sentence REPLACES an unspoken queued one."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jarvis_flap_q_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        p = mock.patch.object(self.bc, "__file__",
                              os.path.join(self.tmp, "bobert_companion.py"))
        p.start()
        self.addCleanup(p.stop)
        self.queue = os.path.join(self.tmp, "pending_speech.json")

    def _read(self):
        with open(self.queue, encoding="utf-8") as f:
            return json.load(f)

    def test_same_tag_replaces_untagged_is_untouched(self):
        bc = self.bc
        with mock.patch("builtins.print"):
            bc.proactive_announce("print is done")
            bc.proactive_announce("Switched to your headset, sir.",
                                  source="audio", supersede="audio-device")
            bc.proactive_announce("Switched to the Desk Mic, sir.",
                                  source="audio", supersede="audio-device")
        msgs = [e["message"] for e in self._read()]
        self.assertEqual(msgs, ["print is done",
                                "Switched to the Desk Mic, sir."])


class PathBOpenFailureTests(_Base):
    def _fail_open(self, **kw):
        raise self.world.PortAudioError(
            "Error opening InputStream: Unanticipated host error "
            "[PaErrorCode -9999]: 'A device ID has been used that is out of "
            "range for your system.' [MME error 2]")

    def test_resolves_before_claiming_and_fails_over_quietly(self):
        bc = self.bc
        seen_flag = []

        def _resolve():
            seen_flag.append(bc._pathb_mic_active[0])
            return 1
        self._p(bc, "get_input_device", side_effect=_resolve)
        self._p(bc, "_mic_input_disabled", return_value=False)
        self.world.InputStream = self._fail_open
        bc._device_cache["in"] = 1
        with mock.patch("builtins.print",
                        side_effect=lambda *a, **k: self.printed.append(
                            " ".join(str(x) for x in a))):
            self.assertIsNone(bc.get_mic_buffer(0.1, sample_rate=16000))
            bc._device_cache["in"] = 1
            self.assertIsNone(bc.get_mic_buffer(0.1, sample_rate=16000))
        self.assertEqual(seen_flag, [False, False],
                         "resolved while Path B's own flag was set: the "
                         "re-enumeration it needs could never run")
        self.assertIsNone(bc._device_cache["in"], "stale index kept cached")
        self.assertTrue(bc._device_cache["reenum_requested"])
        lines = [ln for ln in self.printed if "InputStream open failed" in ln]
        self.assertEqual(len(lines), 1, lines)
        self.assertFalse(bc._pathb_mic_active[0])

    def test_a_requested_re_enumeration_runs_but_not_in_a_loop(self):
        bc = self.bc
        self._settle()
        inits = self.world.inits
        with mock.patch("builtins.print"):
            bc._note_input_open_failure("get_mic_buffer", 0,
                                        RuntimeError("-9999"))
        bc._device_cache["last_reenum_at"] = self.clock.t - 3.0
        self._pass(1.0)
        self.assertEqual(self.world.inits, inits,
                         "no more than one re-enumeration per 10 s")
        self._pass(10.0)
        self.assertEqual(self.world.inits, inits + 1)
        self.assertFalse(bc._device_cache["reenum_requested"])


if __name__ == "__main__":
    unittest.main()
