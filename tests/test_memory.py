"""Logic tests for core.memory — the contextual-promise engine.

Drives the public lifecycle (make_promise → register_condition → _tick fires →
fulfil/cancel), the persistence round-trip, the built-in time conditions, the
pending cap + retention prune, and the watcher start/stop idempotence.

Every test redirects the module's on-disk paths at a per-test tempdir and
resets the in-memory registry, so nothing touches the real
memory/pending_promises.json. The announcer is replaced with a recording stub
so no lazy import of bobert_companion and no real speech queue is involved.
stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from core import memory


class _MemoryTestBase(unittest.TestCase):
    """Isolate every test: fresh tempdir for the promises file, a clean
    in-memory registry, and a stopped watcher. The condition registry is
    snapshotted and restored so register_condition() tests can't leak."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        mem_dir = self._tmp.name
        promises_file = os.path.join(mem_dir, "pending_promises.json")
        self._patchers = [
            mock.patch.object(memory, "_MEM_DIR", mem_dir),
            mock.patch.object(memory, "_PROMISES_FILE", promises_file),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)

        # Reset the lazily-loaded in-memory registry so each test starts blank.
        with memory._lock:
            memory._promises[:] = []
            memory._next_id[0] = 1
            memory._loaded[0] = False
        self.addCleanup(self._reset_state)

        # Snapshot + restore the condition registry (some tests register one).
        self._saved_conditions = dict(memory._conditions)
        self.addCleanup(lambda: memory._conditions.clear()
                        or memory._conditions.update(self._saved_conditions))

        # Make sure no watcher thread survives a test.
        self.addCleanup(memory.stop_watcher)

        self.mem_dir = mem_dir
        self.promises_file = promises_file

    def _reset_state(self):
        with memory._lock:
            memory._promises[:] = []
            memory._next_id[0] = 1
            memory._loaded[0] = False
        memory._announce_fn[0] = None

    def _read_file(self):
        with open(self.promises_file, "r", encoding="utf-8") as f:
            return json.load(f)


class MakePromiseTests(_MemoryTestBase):
    def test_make_promise_returns_incrementing_ids(self):
        a = memory.make_promise("ping", "manual", source="t")
        b = memory.make_promise("pong", "manual", source="t")
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)

    def test_make_promise_persists_to_disk(self):
        memory.make_promise("the print is done", "manual", source="bambu")
        data = self._read_file()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "the print is done")
        self.assertEqual(data[0]["condition"], "manual")
        self.assertEqual(data[0]["source"], "bambu")
        self.assertEqual(data[0]["status"], "pending")

    def test_make_promise_records_deadline(self):
        pid = memory.make_promise("x", "manual", deadline_s=100.0)
        p = next(p for p in memory.list_promises() if p["id"] == pid)
        self.assertIsNotNone(p["deadline"])
        self.assertGreater(p["deadline"], time.time())

    def test_empty_message_rejected(self):
        with self.assertRaises(ValueError):
            memory.make_promise("", "manual")

    def test_empty_condition_rejected(self):
        with self.assertRaises(ValueError):
            memory.make_promise("hi", "")

    def test_oversized_params_rejected(self):
        big = {"k": "x" * (memory._MAX_PARAMS_BYTES + 10)}
        with self.assertRaises(ValueError):
            memory.make_promise("hi", "manual", params=big)

    def test_params_are_copied_not_aliased(self):
        src = {"threshold_c": 40.0}
        memory.make_promise("cool", "bambu_bed_cool", params=src)
        src["threshold_c"] = 999  # mutate caller's dict after the call
        stored = memory.list_promises()[0]["params"]
        self.assertEqual(stored["threshold_c"], 40.0)


class ListAndSnapshotTests(_MemoryTestBase):
    def test_list_excludes_delivered_by_default(self):
        pid = memory.make_promise("done", "manual")
        memory.make_promise("pending one", "manual")
        memory._announce_fn[0] = mock.MagicMock()
        memory.fulfil_promise(pid)
        pending = memory.list_promises()
        self.assertEqual([p["message"] for p in pending], ["pending one"])

    def test_list_include_delivered_shows_all(self):
        pid = memory.make_promise("done", "manual")
        memory._announce_fn[0] = mock.MagicMock()
        memory.fulfil_promise(pid)
        allp = memory.list_promises(include_delivered=True)
        self.assertEqual(len(allp), 1)
        self.assertEqual(allp[0]["status"], "delivered")

    def test_list_returns_deep_copies(self):
        memory.make_promise("x", "manual", params={"n": [1, 2]})
        snap = memory.list_promises()
        snap[0]["params"]["n"].append(99)        # mutate the snapshot
        self.assertEqual(memory.list_promises()[0]["params"]["n"], [1, 2])


class CancelTests(_MemoryTestBase):
    def test_cancel_marks_cancelled(self):
        pid = memory.make_promise("x", "manual")
        self.assertTrue(memory.cancel_promise(pid))
        allp = memory.list_promises(include_delivered=True)
        self.assertEqual(allp[0]["status"], "cancelled")
        self.assertIsNotNone(allp[0]["fired_at"])

    def test_cancel_unknown_id_returns_false(self):
        self.assertFalse(memory.cancel_promise(424242))

    def test_cancel_twice_second_is_false(self):
        pid = memory.make_promise("x", "manual")
        self.assertTrue(memory.cancel_promise(pid))
        self.assertFalse(memory.cancel_promise(pid))   # no longer pending

    def test_cancelled_promise_never_fires_in_tick(self):
        pid = memory.make_promise("x", "time_after", params={"delay_s": 0.0})
        memory.cancel_promise(pid)
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        # delay_s=0 would normally fire immediately, but it's cancelled.
        memory._tick()
        announcer.assert_not_called()


class FulfilTests(_MemoryTestBase):
    def test_fulfil_fires_announcer_and_marks_delivered(self):
        pid = memory.make_promise("bed is cool", "manual", source="bambu")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        self.assertTrue(memory.fulfil_promise(pid))
        announcer.assert_called_once()
        msg, src = announcer.call_args[0]
        self.assertEqual(msg, "bed is cool")
        self.assertEqual(src, "promise:bambu")
        allp = memory.list_promises(include_delivered=True)
        self.assertEqual(allp[0]["status"], "delivered")

    def test_fulfil_unknown_returns_false(self):
        self.assertFalse(memory.fulfil_promise(999))

    def test_fulfil_already_delivered_returns_false(self):
        pid = memory.make_promise("x", "manual")
        memory._announce_fn[0] = mock.MagicMock()
        self.assertTrue(memory.fulfil_promise(pid))
        self.assertFalse(memory.fulfil_promise(pid))

    def test_announcer_exception_still_marks_delivered(self):
        # A raising announcer must not leave the promise stuck pending.
        pid = memory.make_promise("x", "manual")
        boom = mock.MagicMock(side_effect=RuntimeError("speech down"))
        memory._announce_fn[0] = boom
        self.assertTrue(memory.fulfil_promise(pid))
        self.assertEqual(memory.list_promises(include_delivered=True)[0]["status"],
                         "delivered")


class RegisterConditionAndTickTests(_MemoryTestBase):
    def test_custom_condition_fires_when_true(self):
        flag = {"ready": False}
        memory.register_condition("flag_set", lambda p: flag["ready"])
        memory.make_promise("the flag flipped", "flag_set")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer

        memory._tick()
        announcer.assert_not_called()       # predicate still False

        flag["ready"] = True
        memory._tick()
        announcer.assert_called_once()
        self.assertEqual(announcer.call_args[0][0], "the flag flipped")

    def test_register_condition_overrides_existing(self):
        memory.register_condition("manual", lambda p: True)   # override the stub
        memory.make_promise("now fires", "manual")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_called_once()

    def test_unknown_condition_never_fires_but_is_stored(self):
        memory.make_promise("orphan", "no_such_condition")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()
        self.assertEqual(memory.list_promises()[0]["status"], "pending")

    def test_condition_that_raises_is_swallowed(self):
        memory.register_condition("boom", lambda p: (_ for _ in ()).throw(ValueError("x")))
        memory.make_promise("never", "boom")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()                       # must not propagate
        announcer.assert_not_called()
        self.assertEqual(memory.list_promises()[0]["status"], "pending")

    def test_deadline_expiry_takes_precedence(self):
        # Deadline already in the past AND a condition that would fire: expiry wins.
        memory.register_condition("always", lambda p: True)
        pid = memory.make_promise("x", "always", deadline_s=10.0)
        # Force the deadline into the past.
        with memory._lock:
            for p in memory._promises:
                if p["id"] == pid:
                    p["deadline"] = time.time() - 1
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()
        self.assertEqual(memory.list_promises(include_delivered=True)[0]["status"],
                         "expired")


class BuiltinTimeConditionTests(_MemoryTestBase):
    def test_time_after_fires_once_delay_elapsed(self):
        memory.make_promise("ten seconds passed", "time_after",
                            params={"delay_s": 5.0})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer

        # Just created → not yet.
        memory._tick()
        announcer.assert_not_called()

        # Jump the clock forward past the delay.
        future = time.time() + 6.0
        with mock.patch.object(memory.time, "time", return_value=future):
            memory._tick()
        announcer.assert_called_once()

    def test_time_after_zero_delay_never_fires(self):
        # delay must be > 0 to arm — guards against an instantly-true promise.
        memory.make_promise("x", "time_after", params={"delay_s": 0.0})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()

    def test_time_at_fires_after_epoch(self):
        target = time.time() - 1      # already in the past
        memory.make_promise("the moment arrived", "time_at",
                            params={"epoch": target})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_called_once()

    def test_time_at_future_does_not_fire(self):
        memory.make_promise("later", "time_at",
                            params={"epoch": time.time() + 9999})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()

    def test_manual_condition_never_auto_fires(self):
        memory.make_promise("x", "manual")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()


class PersistenceRoundTripTests(_MemoryTestBase):
    def test_reload_recovers_promises_and_next_id(self):
        memory.make_promise("first", "manual")
        memory.make_promise("second", "manual")
        # Simulate a process restart: drop in-memory state, force a reload.
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
            memory._next_id[0] = 1
        reloaded = memory.list_promises()
        self.assertEqual({p["message"] for p in reloaded}, {"first", "second"})
        # next_id continues past the highest seen id, so no id collision.
        new_id = memory.make_promise("third", "manual")
        self.assertEqual(new_id, 3)

    def test_corrupt_file_is_tolerated(self):
        with open(self.promises_file, "w", encoding="utf-8") as f:
            f.write("{ this is not valid json ][")
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
        # Should not raise; just yields an empty registry.
        self.assertEqual(memory.list_promises(), [])

    def test_non_list_json_is_ignored(self):
        with open(self.promises_file, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
        self.assertEqual(memory.list_promises(), [])

    def test_hand_edited_entry_gets_defensive_defaults(self):
        # An entry missing most keys must not crash load; defaults fill in.
        with open(self.promises_file, "w", encoding="utf-8") as f:
            json.dump([{"id": "7", "message": "partial"}], f)
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
        loaded = memory.list_promises(include_delivered=True)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["id"], 7)            # coerced to int
        self.assertEqual(loaded[0]["condition"], "manual")  # default
        self.assertEqual(loaded[0]["status"], "pending")    # default


class PruneAndCapTests(_MemoryTestBase):
    def test_old_delivered_promise_pruned_on_save(self):
        pid = memory.make_promise("old", "manual")
        memory._announce_fn[0] = mock.MagicMock()
        memory.fulfil_promise(pid)
        # Backdate its fired_at beyond the retention window, then trigger a save.
        with memory._lock:
            for p in memory._promises:
                p["fired_at"] = time.time() - (memory._RETENTION_S + 10)
            memory._save_locked()
        self.assertEqual(self._read_file(), [])

    def test_pending_promises_capped_to_max(self):
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        # Create more than the cap; oldest pending should be dropped on save.
        total = memory._MAX_PENDING_PROMISES + 5
        base = time.time()
        with memory._lock:
            for i in range(total):
                memory._promises.append({
                    "id": i + 1, "created_at": base + i, "deadline": None,
                    "message": f"p{i}", "condition": "manual", "params": {},
                    "source": "t", "status": "pending", "fired_at": None,
                })
            memory._next_id[0] = total + 1
            memory._save_locked()
        kept = self._read_file()
        self.assertEqual(len(kept), memory._MAX_PENDING_PROMISES)
        # The newest survive; the oldest (p0) is dropped.
        messages = {p["message"] for p in kept}
        self.assertIn(f"p{total - 1}", messages)
        self.assertNotIn("p0", messages)


class WatcherTests(_MemoryTestBase):
    def test_start_watcher_is_idempotent(self):
        memory.start_watcher(mock.MagicMock(), interval_s=1.0)
        first = memory._watcher_thread[0]
        memory.start_watcher(mock.MagicMock(), interval_s=1.0)
        second = memory._watcher_thread[0]
        self.assertIs(first, second)         # no second thread spawned
        self.assertTrue(first.is_alive())

    def test_stop_watcher_joins_thread(self):
        memory.start_watcher(mock.MagicMock(), interval_s=1.0)
        t = memory._watcher_thread[0]
        memory.stop_watcher()
        self.assertFalse(t.is_alive())
        self.assertIsNone(memory._watcher_thread[0])

    def test_watcher_fires_a_ready_promise(self):
        # A real (short-interval) watcher should pick up a past-epoch promise.
        memory.make_promise("watcher saw it", "time_at",
                            params={"epoch": time.time() - 1})
        announcer = mock.MagicMock()
        memory.start_watcher(announcer, interval_s=1.0)
        deadline = time.time() + 5.0
        while time.time() < deadline and not announcer.called:
            time.sleep(0.02)
        memory.stop_watcher()
        announcer.assert_called_once()
        self.assertEqual(announcer.call_args[0][0], "watcher saw it")


class ActionHelperTests(_MemoryTestBase):
    def test_action_list_empty(self):
        self.assertIn("No outstanding promises", memory.action_list_promises(""))

    def test_action_list_summarises_pending(self):
        memory.make_promise("check the oven", "manual")
        out = memory.action_list_promises("")
        self.assertIn("check the oven", out)
        self.assertIn("#1", out)
        self.assertIn("1 outstanding promise", out)

    def test_action_cancel_happy(self):
        pid = memory.make_promise("x", "manual")
        out = memory.action_cancel_promise(str(pid))
        self.assertIn("cancelled", out.lower())
        self.assertEqual(memory.list_promises(), [])

    def test_action_cancel_bad_id(self):
        self.assertIn("could not parse", memory.action_cancel_promise("abc").lower())

    def test_action_cancel_missing_id(self):
        self.assertIn("don't have", memory.action_cancel_promise("999").lower())

    def test_action_cancel_blank_shows_format(self):
        self.assertIn("format", memory.action_cancel_promise("").lower())

    def test_register_actions_wires_both(self):
        actions = {}
        memory.register_actions(actions)
        self.assertIn("list_promises", actions)
        self.assertIn("cancel_promise", actions)


# ─────────────────────────────────────────────────────────────────────────
# Coverage-completion tests: persistence edge cases, the bambu conditions,
# the time-condition error branches, the announce-fn resolver, the watcher
# loop crash path, and action_list_promises age formatting.
# ─────────────────────────────────────────────────────────────────────────

class _FakeBambuModule:
    """Stand-in for skill_bambu_monitor: exposes a _state dict and a
    _state_lock context manager, matching the two attributes _bambu_state()
    reaches for (mod._state_lock / mod._state)."""

    def __init__(self, state):
        self._state = state
        self._state_lock = threading.RLock()


class BambuStateTests(_MemoryTestBase):
    """_bambu_state() reads sys.modules['skill_bambu_monitor']. We inject a
    fake module so the bambu conditions are exercised with no real printer."""

    def _install_bambu(self, state):
        import sys
        fake = _FakeBambuModule(state)
        self.addCleanup(lambda: sys.modules.pop("skill_bambu_monitor", None))
        sys.modules["skill_bambu_monitor"] = fake
        return fake

    def test_bambu_state_absent_module_returns_empty(self):
        import sys
        sys.modules.pop("skill_bambu_monitor", None)
        self.assertEqual(memory._bambu_state(), {})

    def test_bambu_state_lock_raise_returns_empty(self):
        import sys
        fake = _FakeBambuModule({"gcode_state": "RUNNING"})
        # Replace the lock with one whose __enter__ raises so the except path runs.
        bad = mock.MagicMock()
        bad.__enter__.side_effect = RuntimeError("lock down")
        fake._state_lock = bad
        self.addCleanup(lambda: sys.modules.pop("skill_bambu_monitor", None))
        sys.modules["skill_bambu_monitor"] = fake
        self.assertEqual(memory._bambu_state(), {})

    def test_bambu_state_returns_copy(self):
        self._install_bambu({"gcode_state": "FINISH"})
        snap = memory._bambu_state()
        snap["gcode_state"] = "MUTATED"
        self.assertEqual(memory._bambu_state()["gcode_state"], "FINISH")

    # ── _cond_bambu_print_finish ────────────────────────────────────────
    def test_print_finish_no_state(self):
        import sys
        sys.modules.pop("skill_bambu_monitor", None)
        memory.make_promise("done printing", "bambu_print_finish")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()

    def test_print_finish_stale_update_does_not_fire(self):
        # last_update is BEFORE the promise was created → not trusted yet.
        self._install_bambu({"gcode_state": "FINISH", "last_update": 1.0})
        memory.make_promise("done printing", "bambu_print_finish")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()

    def test_print_finish_fires_on_fresh_finish(self):
        fake = self._install_bambu({"gcode_state": "running", "last_update": 0.0})
        memory.make_promise("done printing", "bambu_print_finish")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        # Now report a FINISH with a fresh timestamp (after created_at).
        fake._state["gcode_state"] = "finish"   # lower-case → upper() in code
        fake._state["last_update"] = time.time() + 5
        memory._tick()
        announcer.assert_called_once()

    # ── _cond_bambu_bed_cool ────────────────────────────────────────────
    def test_bed_cool_no_state(self):
        import sys
        sys.modules.pop("skill_bambu_monitor", None)
        memory.make_promise("bed cooled", "bambu_bed_cool")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()

    def test_bed_cool_waits_for_finish_then_cools(self):
        fake = self._install_bambu({"gcode_state": "running", "last_update": 0.0,
                                    "bed_temper": 60.0})
        memory.make_promise("bed cooled", "bambu_bed_cool",
                            params={"threshold_c": 40.0})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        # Bed already below threshold but no FINISH seen yet → must NOT fire.
        fake._state["bed_temper"] = 30.0
        memory._tick()
        announcer.assert_not_called()
        # Report a fresh FINISH → arms _finish_seen, bed still 30 < 40 → fires.
        fake._state["gcode_state"] = "FINISH"
        fake._state["last_update"] = time.time() + 5
        memory._tick()
        announcer.assert_called_once()

    def test_bed_cool_persists_finish_seen_latch(self):
        # 2026-07-07 bug-hunt (LOW): arming _finish_seen (FINISH observed but bed
        # still hot → the promise does NOT fire) must be WRITTEN TO DISK, else a
        # restart in the FINISH→cool window reloads a promise with no latch and
        # waits forever for a second FINISH that never comes (print's done).
        self._install_bambu({"gcode_state": "FINISH",
                             "last_update": time.time() + 5,
                             "bed_temper": 60.0})    # hot → arms but won't fire
        memory.make_promise("bed cooled", "bambu_bed_cool",
                            params={"threshold_c": 40.0})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()                    # arms _finish_seen; bed hot → no fire
        announcer.assert_not_called()
        # The latch must be on DISK (survives the next tick's _load_locked reload).
        data = self._read_file()
        self.assertTrue(data[0].get("params", {}).get("_finish_seen"),
                        "_finish_seen must be persisted after the arming tick")

    def test_bed_cool_finish_seen_but_bed_none(self):
        self._install_bambu({"gcode_state": "FINISH",
                             "last_update": time.time() + 5,
                             "bed_temper": None})
        memory.make_promise("bed cooled", "bambu_bed_cool")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()       # _finish_seen set, but bed_temper None → no fire
        announcer.assert_not_called()

    def test_bed_cool_bed_unparseable_is_false(self):
        self._install_bambu({"gcode_state": "FINISH",
                             "last_update": time.time() + 5,
                             "bed_temper": "not-a-number"})
        memory.make_promise("bed cooled", "bambu_bed_cool")
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()

    def test_bed_cool_above_threshold_does_not_fire(self):
        self._install_bambu({"gcode_state": "FINISH",
                             "last_update": time.time() + 5,
                             "bed_temper": 55.0})
        memory.make_promise("bed cooled", "bambu_bed_cool",
                            params={"threshold_c": 40.0})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()


class TimeConditionErrorBranchTests(_MemoryTestBase):
    def test_time_at_bad_epoch_value_is_false(self):
        # params.epoch un-floatable → except branch → False (no fire).
        memory.make_promise("x", "time_at", params={"epoch": "soon"})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()

    def test_time_after_bad_delay_value_is_false(self):
        memory.make_promise("x", "time_after", params={"delay_s": "later"})
        announcer = mock.MagicMock()
        memory._announce_fn[0] = announcer
        memory._tick()
        announcer.assert_not_called()


class PersistenceEdgeCaseTests(_MemoryTestBase):
    def test_empty_file_load_is_noop(self):
        # A whitespace-only file → raw is empty → early return, empty registry.
        with open(self.promises_file, "w", encoding="utf-8") as f:
            f.write("   \n")
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
        self.assertEqual(memory.list_promises(include_delivered=True), [])

    def test_non_dict_entries_skipped(self):
        with open(self.promises_file, "w", encoding="utf-8") as f:
            json.dump([{"id": 1, "message": "ok", "condition": "manual"},
                       "garbage", 42, None], f)
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
        loaded = memory.list_promises(include_delivered=True)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["message"], "ok")

    def test_non_dict_params_reset_to_empty(self):
        with open(self.promises_file, "w", encoding="utf-8") as f:
            json.dump([{"id": 2, "message": "p", "params": "not-a-dict"}], f)
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
        loaded = memory.list_promises(include_delivered=True)
        self.assertEqual(loaded[0]["params"], {})

    def test_uncoercible_id_defaults_to_zero(self):
        # id is a list → int() raises TypeError → defaulted to 0.
        with open(self.promises_file, "w", encoding="utf-8") as f:
            json.dump([{"id": [1, 2], "message": "weird"}], f)
        with memory._lock:
            memory._promises[:] = []
            memory._loaded[0] = False
        loaded = memory.list_promises(include_delivered=True)
        self.assertEqual(loaded[0]["id"], 0)

    def test_ensure_dir_failure_swallowed(self):
        # _ensure_dir's makedirs raises → except: pass (no crash on save).
        memory.make_promise("x", "manual")
        with mock.patch.object(memory.os, "makedirs",
                               side_effect=OSError("denied")):
            with memory._lock:
                memory._save_locked()   # must not raise

    def test_save_outer_exception_swallowed(self):
        # mkstemp raises → outer try/except prints + swallows. No crash.
        memory.make_promise("x", "manual")
        with mock.patch.object(memory.tempfile, "mkstemp",
                               side_effect=OSError("no temp")):
            with memory._lock:
                memory._save_locked()   # must not raise

    def test_save_replace_failure_unlinks_temp(self):
        # os.replace fails → inner except unlinks the temp then re-raises;
        # the outer except swallows. The temp must not be left behind.
        memory.make_promise("x", "manual")
        seen = {}
        orig = memory.tempfile.mkstemp

        def spy(*a, **k):
            fd, path = orig(*a, **k)
            seen["tmp"] = path
            return fd, path

        with mock.patch.object(memory.tempfile, "mkstemp", side_effect=spy), \
                mock.patch.object(memory.os, "replace",
                                  side_effect=OSError("replace boom")):
            with memory._lock:
                memory._save_locked()
        self.assertFalse(os.path.exists(seen["tmp"]))

    def test_save_replace_and_unlink_both_fail_swallowed(self):
        # os.replace fails AND the temp os.unlink in the inner except ALSO
        # fails → innermost `except Exception: pass` swallows it, the re-raise
        # bubbles to the outer try and is swallowed. No crash; temp may linger.
        memory.make_promise("x", "manual")
        with mock.patch.object(memory.os, "replace",
                               side_effect=OSError("replace boom")), \
                mock.patch.object(memory.os, "unlink",
                                  side_effect=OSError("unlink boom")):
            with memory._lock:
                memory._save_locked()   # must not raise


class ResolveAnnounceFnTests(_MemoryTestBase):
    def test_explicit_announce_fn_used(self):
        sentinel = mock.MagicMock()
        memory._announce_fn[0] = sentinel
        self.assertIs(memory._resolve_announce_fn(), sentinel)

    def test_lazy_import_bobert_proactive_announce(self):
        import sys
        memory._announce_fn[0] = None
        fake_bc = mock.MagicMock()
        fake_bc.proactive_announce = mock.MagicMock()
        self.addCleanup(lambda: sys.modules.pop("bobert_companion", None))
        sys.modules["bobert_companion"] = fake_bc
        self.assertIs(memory._resolve_announce_fn(), fake_bc.proactive_announce)

    def test_fallback_print_announcer_when_no_companion(self):
        import sys
        memory._announce_fn[0] = None
        # Ensure the import path fails: install a companion WITHOUT a callable.
        broken = mock.MagicMock()
        broken.proactive_announce = "not callable"
        self.addCleanup(lambda: sys.modules.pop("bobert_companion", None))
        sys.modules["bobert_companion"] = broken
        fn = memory._resolve_announce_fn()
        # The fallback is a lambda that prints; calling it must not raise.
        with mock.patch("builtins.print"):
            fn("hello", "src")

    def test_fallback_when_import_raises(self):
        import sys
        import builtins
        memory._announce_fn[0] = None
        sys.modules.pop("bobert_companion", None)
        real_import = builtins.__import__

        def boom(name, *a, **k):
            if name == "bobert_companion":
                raise ImportError("no companion in test")
            return real_import(name, *a, **k)

        with mock.patch.object(builtins, "__import__", side_effect=boom):
            fn = memory._resolve_announce_fn()
        with mock.patch("builtins.print"):
            fn("hello")   # default source kwarg path


class WatcherLoopCrashTests(_MemoryTestBase):
    def test_tick_exception_inside_loop_is_logged_not_fatal(self):
        # Force _tick to raise on the first call, then stop the loop. The
        # loop's inner try/except must catch it and the loop must exit
        # cleanly when the stop event is set.
        calls = {"n": 0}

        def flaky_tick():
            calls["n"] += 1
            raise RuntimeError("tick boom")

        with mock.patch.object(memory, "_tick", side_effect=flaky_tick):
            memory._watcher_stop.clear()
            t = threading.Thread(
                target=memory._watcher_loop, args=(0.05,), daemon=True)
            t.start()
            # Let it iterate at least once, then ask it to stop.
            deadline = time.time() + 3.0
            while calls["n"] < 1 and time.time() < deadline:
                time.sleep(0.01)
            memory._watcher_stop.set()
            t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertGreaterEqual(calls["n"], 1)

    def test_outer_except_catches_wait_failure(self):
        # The OUTER try/except in _watcher_loop guards against the loop's own
        # control flow raising (e.g. _watcher_stop.wait throwing). We make
        # wait() raise once, then on the recovery wait() return True to break.
        real_stop = memory._watcher_stop
        fake_stop = mock.MagicMock()
        fake_stop.is_set.return_value = False
        # 1st wait (inner, after _tick) raises → caught by OUTER except;
        # 2nd wait (in the outer except recovery) returns True → loop breaks.
        fake_stop.wait.side_effect = [RuntimeError("wait boom"), True]
        with mock.patch.object(memory, "_watcher_stop", fake_stop), \
                mock.patch.object(memory, "_tick", return_value=None):
            # Run synchronously — it should return promptly via the break.
            memory._watcher_loop(0.01)
        # Restore is automatic via patch context; sanity check the real one is back.
        self.assertIs(memory._watcher_stop, real_stop)
        # Both waits were consumed (inner-raise then outer-recovery-break).
        self.assertEqual(fake_stop.wait.call_count, 2)


class ActionListAgeFormattingTests(_MemoryTestBase):
    def _make_with_age(self, seconds_ago):
        pid = memory.make_promise("task", "manual")
        with memory._lock:
            for p in memory._promises:
                if p["id"] == pid:
                    p["created_at"] = time.time() - seconds_ago
        return pid

    def test_age_seconds(self):
        self._make_with_age(5)
        self.assertIn("s ago", memory.action_list_promises(""))

    def test_age_minutes(self):
        self._make_with_age(120)        # 2 minutes
        out = memory.action_list_promises("")
        self.assertIn("m ago", out)

    def test_age_hours(self):
        self._make_with_age(3 * 3600 + 5 * 60)   # 3h05m
        out = memory.action_list_promises("")
        self.assertRegex(out, r"\d+h\d{2}m ago")


if __name__ == "__main__":
    unittest.main()
