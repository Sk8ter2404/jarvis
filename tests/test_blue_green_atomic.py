"""Regression: blue_green_manager._atomic_write_json must write the cross-process
instances.json via a UNIQUE per-call temp (mkstemp), NOT a fixed `path + ".tmp"`.

During a blue/green upgrade the prod (blue) and staging (green) JARVIS instances
BOTH heartbeat-write data/instances.json concurrently (plus the smoke driver), so
a shared temp name lets one writer truncate another's half-written temp -> a
0-byte / garbled instances.json that breaks role resolution and can derail the
handoff. Found by the full-codebase audit.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import blue_green_manager as bgm


class BlueGreenAtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.p = os.path.join(self.d, "instances.json")

    def test_roundtrip(self):
        self.assertTrue(bgm._atomic_write_json(self.p, {"a": 1, "b": 2}))
        with open(self.p, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1, "b": 2})
        self.assertFalse(os.path.exists(self.p + ".tmp"))

    def test_uses_unique_mkstemp_not_fixed_tmp(self):
        real = tempfile.mkstemp
        seen = []

        def spy(*a, **k):
            fd, name = real(*a, **k)
            seen.append(name)
            return fd, name

        with mock.patch("tempfile.mkstemp", side_effect=spy):
            self.assertTrue(bgm._atomic_write_json(self.p, {"x": 1}))
        self.assertEqual(len(seen), 1, "must use exactly one mkstemp temp")
        self.assertNotEqual(seen[0], self.p + ".tmp",
                            "must NOT use the fixed shared temp name")
        self.assertTrue(os.path.basename(seen[0]).startswith("instances.json."))

    def test_two_writers_get_distinct_temps(self):
        real = tempfile.mkstemp
        seen = []

        def spy(*a, **k):
            fd, name = real(*a, **k)
            seen.append(name)
            return fd, name

        with mock.patch("tempfile.mkstemp", side_effect=spy):
            bgm._atomic_write_json(self.p, {"w": 1})
            bgm._atomic_write_json(self.p, {"w": 2})
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1],
                            "concurrent writers must get distinct temps")


class HandoffSignalOrphanTests(unittest.TestCase):
    """2026-08-20 LOW finding: consume_handoff_signal() had NO age check, and
    the pipeline has a path that orphans data/handoff.signal.

    run_blue_green_handoff() writes the signal unconditionally. If prod is
    already down (or dies inside the ~3-21 s window before its next tick), the
    grace loop breaks on its FIRST poll, so signal_handoff_failure() is never
    written and nothing deletes the file. The freshly promoted prod then
    consumed a takeover addressed to its PREDECESSOR: it announced "Switching
    to the new version, sir" and exited ~grace seconds into its first main
    loop, leaving the box mute for the 5-10 minutes the watchdog takes to
    notice — while the pipeline printed "handoff complete" and returned ok:True.

    Its sibling consume_handoff_state() has had HANDOFF_STATE_TTL_SECONDS since
    it was written, with a docstring naming this exact orphan class.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.sig = os.path.join(self.d, "handoff.signal")
        p = mock.patch.object(bgm, "HANDOFF_SIGNAL_FILE", self.sig)
        p.start()
        self.addCleanup(p.stop)

    def _write(self, signaled_at, **extra):
        payload = {"signaled_at": signaled_at, "reason": "upgrade",
                   "target_version": "9.9.9", "grace_seconds": 10}
        payload.update(extra)
        self.assertTrue(bgm._atomic_write_json(self.sig, payload))

    def test_signal_written_after_boot_is_consumed(self):
        # The legitimate case: the ceremony signalled a prod that was already
        # running. Must still work, and must still be one-shot.
        self._write(bgm._PROCESS_START + 5.0)
        got = bgm.consume_handoff_signal()
        self.assertIsInstance(got, dict)
        self.assertEqual(got["target_version"], "9.9.9")
        self.assertFalse(os.path.exists(self.sig))
        self.assertIsNone(bgm.consume_handoff_signal())

    def test_signal_written_before_boot_is_refused_and_cleared(self):
        # THE regression. A signal older than this interpreter was addressed to
        # our predecessor.
        self._write(bgm._PROCESS_START - 1.0)
        with mock.patch("builtins.print"):
            self.assertIsNone(bgm.consume_handoff_signal())
        self.assertFalse(os.path.exists(self.sig),
                         "an orphan must not survive to poison the NEXT boot")

    def test_guard_is_boot_relative_not_a_wall_clock_ttl(self):
        """A flat TTL was the obvious fix and would have let the orphan
        through: the legitimate consume happens after a 3 s choreography sleep,
        a 0-18 s grace poll, a staging teardown and a full cold boot. Prove the
        rule is 'before our start', not 'older than N seconds' — a signal from
        two hours in the past but still after our start is honoured."""
        self._write(bgm._PROCESS_START + 0.001)
        with mock.patch.object(bgm.time, "time",
                               return_value=bgm._PROCESS_START + 7200.0):
            self.assertIsNotNone(bgm.consume_handoff_signal())

    def test_missing_signaled_at_is_still_consumed(self):
        # Older/hand-written payloads have no timestamp; refusing them would
        # break the ceremony rather than fix the orphan.
        self.assertTrue(bgm._atomic_write_json(self.sig, {"reason": "upgrade"}))
        self.assertIsNotNone(bgm.consume_handoff_signal())

    def test_non_numeric_signaled_at_is_still_consumed(self):
        self._write("not-a-timestamp")
        self.assertIsNotNone(bgm.consume_handoff_signal())

    def test_non_dict_payload_is_none(self):
        with open(self.sig, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")
        self.assertIsNone(bgm.consume_handoff_signal())
        self.assertFalse(os.path.exists(self.sig))

    def test_clear_unconsumed_reports_whether_it_removed_anything(self):
        self.assertFalse(bgm.clear_unconsumed_handoff_signal(),
                         "nothing on disk -> nothing cleared, and the caller "
                         "must be able to tell (it logs on True)")
        self._write(bgm._PROCESS_START + 1.0)
        self.assertTrue(bgm.clear_unconsumed_handoff_signal())
        self.assertFalse(os.path.exists(self.sig))
        self.assertFalse(bgm.clear_unconsumed_handoff_signal())

    def test_clear_unconsumed_never_raises(self):
        self._write(bgm._PROCESS_START + 1.0)
        with mock.patch.object(bgm.os, "remove",
                               side_effect=OSError("locked")):
            self.assertFalse(bgm.clear_unconsumed_handoff_signal())


if __name__ == "__main__":
    unittest.main()
