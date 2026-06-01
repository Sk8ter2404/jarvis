#!/usr/bin/env python3
"""test_bambu_persistence.py — task #71 re-validation.

Verifies the print-reminder persistence mechanism in skills/bambu_monitor.py:
the state file `data/bambu_reminder_state.json` is written on FINISH and
prevents a JARVIS bounce from replaying the same print's announcement.

Scenarios:
  1. Inject a simulated MQTT FINISH message and verify _enqueue_speech is
     called once + the state file is written with a finish:<filename> key.
  2. Clear in-memory flags to simulate a bounce (but leave the state file
     intact), re-inject the identical message, and verify the FINISH
     announcement does NOT replay.
  3. Watchdog: scan the pending_speech.json the fallback path wrote and
     confirm only one 'Print complete' entry exists.
  4. Sanity check: a DIFFERENT print's FINISH (post-bounce) still announces.

Run directly:  python test_bambu_persistence.py
Exit code 0 on success, non-zero on any failure.
"""

import json
import os
import sys
import tempfile

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Use a temp directory for state files so we don't clobber production state.
_tmpdir = tempfile.mkdtemp(prefix="bambu_test_")
_test_state_file = os.path.join(_tmpdir, "data", "bambu_reminder_state.json")
_test_speech_queue = os.path.join(_tmpdir, "pending_speech.json")

import skills.bambu_monitor as bm  # noqa: E402

bm._REMINDER_STATE_FILE = _test_state_file
bm._SPEECH_QUEUE = _test_speech_queue
os.makedirs(os.path.dirname(_test_state_file), exist_ok=True)

# Capture _enqueue_speech calls. Keep the real write semantics so the
# pending_speech.json scan at the end also works.
_speech_calls: list = []
_orig_enqueue = bm._enqueue_speech


def _capture_enqueue(msg):
    _speech_calls.append(msg)
    _orig_enqueue(msg)


bm._enqueue_speech = _capture_enqueue

# Block the bobert_companion import inside _enqueue_speech so the fallback
# atomic write into _test_speech_queue runs deterministically.
import importlib  # noqa: E402

_orig_import = importlib.import_module


def _fake_import(name, *a, **kw):
    if name == "bobert_companion":
        raise ImportError("blocked for test isolation")
    return _orig_import(name, *a, **kw)


importlib.import_module = _fake_import


def _reset_in_memory_only():
    """Simulate a JARVIS bounce: clear runtime flags, leave state file alone."""
    bm._announced_milestones.clear()
    bm._announced_error_codes.clear()
    bm._current_print_filename[0] = None
    bm._last_gcode_state[0] = None
    bm._announced_start[0] = False
    bm._announced_layer1[0] = False
    bm._post_finish_bed_watch[0] = False
    bm._bed_cool_announced[0] = False
    bm._last_finish_announced_at[0] = 0.0
    bm._print_start_ts[0] = 0.0
    bm._print_initial_estimate_min[0] = None
    with bm._state_lock:
        for k in list(bm._state.keys()):
            bm._state[k] = None if k != "last_update" else 0.0
        bm._chamber_history.clear()


def _make_msg(payload_dict):
    """Build a fake paho-mqtt msg with a JSON payload."""

    class FakeMsg:
        pass

    m = FakeMsg()
    m.payload = json.dumps(payload_dict).encode("utf-8")
    return m


def _count_finish_in_speech_queue() -> int:
    """Watchdog: count 'Print complete' entries in pending_speech.json."""
    if not os.path.exists(_test_speech_queue):
        return 0
    with open(_test_speech_queue, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sum(1 for e in data if "Print complete" in (e.get("message") or ""))


FINISH_PAYLOAD = {
    "print": {
        "gcode_state": "FINISH",
        "subtask_name": "test_persistence_print.3mf",
        "mc_percent": 100,
        # Bed is still hot so the cool-down callout doesn't fire during the
        # test — we're only validating the FINISH replay guard here.
        "bed_temper": 55.0,
    }
}

EXPECTED_FNAME = "test persistence print"  # _strip_filename normalises this
EXPECTED_KEY = f"finish:{EXPECTED_FNAME}"

# ── Scenario 1: first FINISH message announces + persists ─────────────────
print("[test] scenario 1: first FINISH message announces and persists state")
_reset_in_memory_only()
_speech_calls.clear()
if os.path.exists(_test_state_file):
    os.remove(_test_state_file)
if os.path.exists(_test_speech_queue):
    os.remove(_test_speech_queue)

bm._on_message(None, None, _make_msg(FINISH_PAYLOAD))

finish_speeches = [s for s in _speech_calls if "Print complete" in s]
if len(finish_speeches) != 1:
    print(f"[test] FAIL scenario 1: expected 1 'Print complete' enqueue, "
          f"got {len(finish_speeches)}: {_speech_calls}")
    sys.exit(1)

if not os.path.exists(_test_state_file):
    print(f"[test] FAIL scenario 1: state file {_test_state_file} was not written")
    sys.exit(1)

with open(_test_state_file, "r", encoding="utf-8") as f:
    persisted = json.load(f)

if EXPECTED_KEY not in persisted:
    print(f"[test] FAIL scenario 1: expected key '{EXPECTED_KEY}' missing from "
          f"persisted state: {persisted}")
    sys.exit(1)

print(f"[test] scenario 1 PASS: enqueued FINISH once, persisted key '{EXPECTED_KEY}'")

# ── Scenario 2: bounce + same FINISH message must NOT re-announce ─────────
print("[test] scenario 2: simulated bounce — replaying same FINISH must be silent")
_reset_in_memory_only()
_speech_calls.clear()

if not os.path.exists(_test_state_file):
    print("[test] FAIL scenario 2: state file vanished during bounce simulation")
    sys.exit(1)

# Sanity: re-load via the module's own helper to catch encoding / parse issues
# the planner flagged as a regression risk.
reload_check = bm._load_reminder_persistence()
if EXPECTED_KEY not in reload_check:
    print(f"[test] FAIL scenario 2: _load_reminder_persistence() lost the key "
          f"after bounce simulation: {reload_check}")
    sys.exit(1)

bm._on_message(None, None, _make_msg(FINISH_PAYLOAD))

finish_speeches = [s for s in _speech_calls if "Print complete" in s]
if len(finish_speeches) != 0:
    print(f"[test] FAIL scenario 2: FINISH replayed after bounce: {finish_speeches}")
    sys.exit(1)

print("[test] scenario 2 PASS: no duplicate FINISH announcement after bounce")

# ── Scenario 3: watchdog scan of pending_speech.json ──────────────────────
print("[test] scenario 3: watchdog — pending_speech.json has no duplicate FINISH")
finish_count = _count_finish_in_speech_queue()
if finish_count != 1:
    print(f"[test] FAIL scenario 3: pending_speech.json contains {finish_count} "
          f"'Print complete' entries, expected exactly 1")
    sys.exit(1)
print(f"[test] scenario 3 PASS: pending_speech.json has exactly 1 'Print complete' entry")

# ── Scenario 4: a different print should still announce normally ─────────
print("[test] scenario 4: a DIFFERENT print's FINISH should still announce")
_reset_in_memory_only()
_speech_calls.clear()

NEW_PAYLOAD = {
    "print": {
        "gcode_state": "FINISH",
        "subtask_name": "different_print.3mf",
        "mc_percent": 100,
        "bed_temper": 55.0,
    }
}
bm._on_message(None, None, _make_msg(NEW_PAYLOAD))

finish_speeches = [s for s in _speech_calls if "Print complete" in s]
if len(finish_speeches) != 1:
    print(f"[test] FAIL scenario 4: new print did not announce — got "
          f"{finish_speeches}")
    sys.exit(1)

with open(_test_state_file, "r", encoding="utf-8") as f:
    persisted = json.load(f)

NEW_KEY = "finish:different print"
if NEW_KEY not in persisted or EXPECTED_KEY not in persisted:
    print(f"[test] FAIL scenario 4: state file missing one of the two finish "
          f"keys: {persisted}")
    sys.exit(1)
print("[test] scenario 4 PASS: distinct prints announce independently; both persisted")

print("[test] all scenarios PASSED")
sys.exit(0)
