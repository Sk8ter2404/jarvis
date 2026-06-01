#!/usr/bin/env python3
"""test_watchdog.py — verifies the main-loop watchdog (bug-5 fix).

Simulates the record_speech() hang scenario by mocking sd.InputStream so
audio_q.get(timeout=0.1) always raises queue.Empty (no audio ever arrives).
Then:
  * Scenario 1 — a stale heartbeat triggers the watchdog's recovery log +
    reset signal within 90s (well within, because the test lowers the
    timing constants so the proof completes in ~1s).
  * Scenario 2 — once the reset signal is set, record_speech() bails out
    of the audio-wait loop and returns None instead of blocking forever.

Run directly:  python test_watchdog.py
Exit code 0 on success, non-zero on any failure.
"""

import os
import sys
import time
import threading
import types

# Route the early-boot singleton lock to jarvis_staging.lock so we don't
# clobber a live prod instance's jarvis.lock during the test.
os.environ["JARVIS_STAGING"] = "1"

# Stub out sounddevice BEFORE bobert_companion imports it. The fake
# InputStream successfully enters its context manager but never delivers
# audio — exactly the failure mode the watchdog is meant to recover from.
class _FakeStream:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

class _PortAudioError(Exception):
    pass

_sd = types.ModuleType("sounddevice")
_sd.InputStream  = lambda *a, **kw: _FakeStream()
_sd.OutputStream = lambda *a, **kw: _FakeStream()
_sd.PortAudioError = _PortAudioError
_sd.query_devices  = lambda *a, **kw: []
_sd.query_hostapis = lambda *a, **kw: []
_sd.default = types.SimpleNamespace(device=(None, None), samplerate=16000)
_sd.rec  = lambda *a, **kw: None
_sd.play = lambda *a, **kw: None
_sd.wait = lambda *a, **kw: None
_sd.stop = lambda *a, **kw: None
sys.modules["sounddevice"] = _sd

print("[test] importing bobert_companion (this can take a few seconds)…")
import bobert_companion as bc  # noqa: E402

# bobert_companion's record_speech short-circuits when BLUE_GREEN_ROLE
# is "staging" (no mic). We set JARVIS_STAGING=1 only to redirect the
# lock file; flip the role back so record_speech actually exercises
# the (mocked) InputStream path.
bc.BLUE_GREEN_ROLE = "prod"

# Speed up the watchdog so the test completes in seconds, not minutes.
# The spec demands "within 90s"; we just compress the same mechanism.
bc._MAIN_LOOP_HEARTBEAT_TIMEOUT = 3.0
bc._MAIN_LOOP_WATCHDOG_INTERVAL = 0.5

# Fresh state.
bc._watchdog_reset_signal.clear()
bc._watchdog_stop_event.clear()
bc._main_loop_heartbeat[0] = time.time()

# Wire the watchdog the same way main() does.
threading.Thread(target=bc._main_loop_watchdog_thread, daemon=True).start()

# ── Scenario 1: stale heartbeat → recovery log + reset signal ──────────
print("[test] scenario 1: stale heartbeat must trigger watchdog within 90s")
# Backdate the heartbeat so the next watchdog tick decides we're stalled.
bc._main_loop_heartbeat[0] = time.time() - 65.0

deadline = time.time() + 90.0
fired = False
while time.time() < deadline:
    if bc._watchdog_reset_signal.is_set():
        fired = True
        break
    time.sleep(0.1)

if not fired:
    print("[test] FAIL: watchdog never set the reset signal within 90s")
    sys.exit(1)
print(f"[test] scenario 1 PASS: watchdog fired after "
      f"{time.time() - (deadline - 90.0):.2f}s")

# ── Scenario 2: reset signal causes record_speech to return None ───────
print("[test] scenario 2: record_speech returns None when reset signal set")
# Make sure the heartbeat is fresh so the watchdog doesn't keep stomping
# on us; we want exactly one stable reset-signal state for this scenario.
bc._main_loop_heartbeat[0] = time.time()
bc._watchdog_reset_signal.set()

result_box: dict = {"audio": "PENDING", "elapsed": None}
t0 = time.time()

def _runner():
    try:
        result_box["audio"] = bc.record_speech(timeout=20)
    except Exception as e:
        result_box["audio"] = f"EXCEPTION: {e!r}"
    result_box["elapsed"] = time.time() - t0

t = threading.Thread(target=_runner, daemon=True)
t.start()
t.join(timeout=10.0)

if t.is_alive():
    print(f"[test] FAIL: record_speech still running after 10s "
          f"(reset signal set={bc._watchdog_reset_signal.is_set()})")
    sys.exit(1)

if result_box["audio"] is not None:
    print(f"[test] FAIL: record_speech returned {result_box['audio']!r}, "
          f"expected None")
    sys.exit(1)

print(f"[test] scenario 2 PASS: record_speech returned None in "
      f"{result_box['elapsed']:.2f}s")

# Quiet the watchdog so the interpreter exits cleanly.
bc._watchdog_stop_event.set()
print("[test] all scenarios PASSED")
sys.exit(0)
