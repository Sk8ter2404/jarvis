"""The boot order: the first camera open waits for the boot to settle
(2026-11-16, CAMERA_FIRST_OPEN_WAIT_S).

WHY. Both first v3 boots on the Dell (11-06, 11-16) froze ~3.5-3.75 min: the
webcam's first open sat inside the camera driver that long, and the main
thread could not get through the boot line until it came back (11-16 stack:
the boot line's TTS preset -> core.audio_processor -> noisereduce -> scipy,
inside an extension module's DLL load). No voice, no mic, no hand tracking.
Now main() loads the audio DSP before any camera open, and the boot's face
tracker waits until the main loop is listening with the mic open (bounded).

CI SAFETY: the helpers are lifted out of the monolith's SOURCE by AST and
exec'd with stub globals, so this runs on the light-deps runner.
"""
from __future__ import annotations

import ast
import os
import sys
import threading
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

_MONOLITH = os.path.join(_PROJECT, "bobert_companion.py")
_CACHE: dict = {}


def _source() -> str:
    if "src" not in _CACHE:
        with open(_MONOLITH, encoding="utf-8") as fh:
            _CACHE["src"] = fh.read()
    return _CACHE["src"]


def _lift(names, ns):
    tree = ast.parse(_source())
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(wanted) == len(names), [n.name for n in wanted]
    exec(compile(ast.Module(body=wanted, type_ignores=[]), _MONOLITH, "exec"), ns)
    return ns


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeStop:
    """Its wait() advances the fake clock and runs scheduled events."""

    def __init__(self, clock, events=None):
        self.clock = clock
        self.events = sorted(events or [], key=lambda e: e[0])
        self._set = False
        self.waits = 0

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, s):
        self.waits += 1
        self.clock.t += s
        while self.events and self.events[0][0] <= self.clock.t:
            self.events.pop(0)[1]()
        if self.waits > 100000:
            raise AssertionError("boot wait never returned")


class BootWaitTests(unittest.TestCase):
    def setUp(self):
        self.ns = _lift(["_face_track_boot_wait"], {
            "time": time, "CAMERA_FIRST_OPEN_WAIT_S": 45.0,
            "_BOOT_MIC_GRACE_S": 5.0, "_boot_listening": threading.Event(),
            "_mic_bus_open": lambda: False, "_face_track_stop": threading.Event()})
        self.wait = self.ns["_face_track_boot_wait"]
        self.clock = FakeClock()
        self.listening = threading.Event()
        self.mic = {"open": False}

    def run_wait(self, cap=None, events=None, mic_open=None):
        stop = FakeStop(self.clock, events)
        t0 = self.clock.t
        why = self.wait(cap, listening=self.listening,
                        mic_open=mic_open or (lambda: self.mic["open"]),
                        stop=stop, clock=self.clock)
        return why, self.clock.t - t0, stop

    def test_off_when_the_cap_is_zero(self):
        self.assertEqual(self.run_wait(0.0)[0], "off")
        self.assertEqual(self.run_wait("x")[0], "off")

    def test_settled_at_once_when_listening_with_the_mic_open(self):
        self.listening.set()
        self.mic["open"] = True
        why, waited, _ = self.run_wait()
        self.assertEqual((why, waited), ("settled", 0.0))

    def test_waits_for_listening_then_the_mic(self):
        events = [(1012.0, self.listening.set),
                  (1013.5, lambda: self.mic.__setitem__("open", True))]
        why, waited, _ = self.run_wait(events=events)
        self.assertEqual(why, "settled")
        self.assertAlmostEqual(waited, 13.6, delta=0.25)

    def test_mic_grace_when_the_mic_never_opens(self):
        events = [(1010.0, self.listening.set)]
        why, waited, _ = self.run_wait(events=events)
        self.assertEqual(why, "mic grace")
        self.assertAlmostEqual(waited, 15.0, delta=0.25)

    def test_a_raising_mic_check_is_not_open(self):
        self.listening.set()

        def boom():
            raise RuntimeError("bus")
        why, waited, _ = self.run_wait(mic_open=boom)
        self.assertEqual(why, "mic grace")
        self.assertAlmostEqual(waited, 5.0, delta=0.25)

    def test_cap_when_it_never_listens(self):
        why, waited, _ = self.run_wait()
        self.assertEqual(why, "cap")
        self.assertAlmostEqual(waited, 45.0, delta=0.25)

    def test_cap_bounds_the_mic_grace_too(self):
        events = [(1043.0, self.listening.set)]
        why, waited, _ = self.run_wait(events=events)
        self.assertEqual(why, "cap")
        self.assertAlmostEqual(waited, 45.0, delta=0.25)

    def test_stop_ends_it(self):
        stop_events = []
        stop = FakeStop(self.clock)
        stop_events.append((1002.0, stop.set))
        stop.events = stop_events
        why = self.wait(None, listening=self.listening, mic_open=lambda: False,
                        stop=stop, clock=self.clock)
        self.assertEqual(why, "stop")

    def test_real_defaults_return_quickly_with_real_events(self):
        # The module-level defaults (no injected events/clock) on a settled
        # boot: returns 'settled' without sleeping.
        self.ns["_boot_listening"].set()
        self.ns["_mic_bus_open"] = lambda: True
        t0 = time.monotonic()
        self.assertEqual(self.wait(), "settled")
        self.assertLess(time.monotonic() - t0, 1.0)


class AfterBootTargetTests(unittest.TestCase):
    def _ns(self, why):
        calls = []
        ns = _lift(["_face_tracking_thread_after_boot"], {
            "time": time, "_face_track_boot_wait": lambda: why,
            "_face_tracking_thread": lambda: calls.append("ran"),
            "print": lambda *a, **k: calls.append(("print",) + a)})
        return ns["_face_tracking_thread_after_boot"], calls

    def test_runs_the_supervisor_after_the_wait(self):
        for why in ("settled", "mic grace", "cap", "off"):
            fn, calls = self._ns(why)
            fn()
            self.assertIn("ran", calls, why)

    def test_off_prints_nothing(self):
        fn, calls = self._ns("off")
        fn()
        self.assertEqual(calls, ["ran"])

    def test_stop_skips_the_supervisor(self):
        fn, calls = self._ns("stop")
        fn()
        self.assertNotIn("ran", calls)


class SourceTests(unittest.TestCase):
    def test_main_starts_the_after_boot_target(self):
        src = _source()
        self.assertIn("threading.Thread(target=_face_tracking_thread_after_boot,", src)
        self.assertNotIn("threading.Thread(target=_face_tracking_thread, daemon=True)", src)

    def test_the_audio_dsp_loads_before_the_face_thread_starts(self):
        # reliability-11 (2026-11-21, BOOT_AUDIO_PROC_PREWARM): main() calls
        # _audio_proc_boot_ready() right before the face thread starts; that
        # helper reports the build done on a thread during load_skills (joined
        # before the camera probe) or, off / failed, builds it synchronously.
        src = _source()
        main_at = src.index("\ndef main(")
        warm = src.index("_audio_proc_boot_ready()", main_at)
        start = src.index("threading.Thread(target=_face_tracking_thread_after_boot,",
                          main_at)
        self.assertLess(warm, start)
        self.assertLess(start - warm, 800)
        helper_at = src.index("\ndef _audio_proc_boot_ready(")
        helper = src[helper_at:src.index("\ndef ", helper_at + 1)]
        self.assertIn("_ap_boot.get_processor()", helper)
        # The background build is joined before the camera probe can open
        # a camera.
        join = src.index("_audio_proc_prewarm_join()", main_at)
        probe = src.index("probe_cameras_and_update_config()", main_at)
        self.assertLess(join, probe)

    def test_both_listening_lines_set_the_event(self):
        src = _source()
        for line in ('print("Listening… (realtime)")', 'print("Listening…")'):
            at = src.index(line)
            nxt = src.index("_boot_listening.set()", at)
            self.assertLess(nxt - at, 160, line)

    def test_the_sleeping_standby_listen_sets_the_event_too(self):
        # 2026-11-17 (Dell 12:32): wake-word mode boots asleep, so only the
        # "Sleeping… (say 'JARVIS' to wake)" listen runs at first; without
        # the event there the first camera open waited the whole cap.
        src = _source()
        start = src.index("\ndef _handle_sleep_standby(")
        body = src[start:src.index("\ndef ", start + 1)]
        at = body.index("(say 'JARVIS' to wake)")
        nxt = body.index("_boot_listening.set()", at)
        self.assertLess(nxt - at, 600)
        # It must be on the mic-listen path, before the capture starts.
        self.assertLess(nxt, body.index("_tt_rec_since = _tt(\"now\")", at))

    def test_config_default_is_bounded_and_machine_local(self):
        from core import config, peer_settings
        self.assertGreater(config.CAMERA_FIRST_OPEN_WAIT_S, 0)
        self.assertLessEqual(config.CAMERA_FIRST_OPEN_WAIT_S, 120)
        self.assertFalse(peer_settings.is_shared("CAMERA_FIRST_OPEN_WAIT_S"))


if __name__ == "__main__":
    unittest.main()
