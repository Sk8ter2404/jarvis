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


if __name__ == "__main__":
    unittest.main()
