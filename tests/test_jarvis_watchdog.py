"""tools/jarvis_watchdog.py — the resurrection net's liveness check.

CORPSE BLINDNESS (2026-07-14, live): _jarvis_running() COUNTED the CIM rows
whose command line matched bobert_companion and called any count > 0 "alive".
A kernel-stuck 'terminating forever' process (thread parked in a CUDA/audio
driver at exit) keeps its row — command line intact — until Windows reboots.
So one corpse permanently convinced the watchdog that JARVIS was running: the
real JARVIS died at 10:49 and every 5-minute tick no-opped against two
day-old corpses. The check must ask whether each PID is GENUINELY EXECUTING.

H-7 (2026-08-20): the resurrection net also had to learn that "dead + no
clean-shutdown flag" is the NORMAL state for hours during an upgrade run, not
a crash — while never becoming parkable-forever, which is the mirror-image bug
that cost six days of downtime in July. See the banner above _WdBase.
"""
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

_spec = importlib.util.spec_from_file_location(
    "jarvis_watchdog_under_test",
    os.path.join(_PROJECT, "tools", "jarvis_watchdog.py"))
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)


def _cim(pids):
    """Fake the PowerShell CIM query: it prints one PID per line."""
    out = mock.Mock()
    out.stdout = "\n".join(str(p) for p in pids)
    return out


class JarvisRunningTests(unittest.TestCase):
    def test_no_rows_means_dead(self):
        with mock.patch.object(wd.subprocess, "run", return_value=_cim([])):
            self.assertFalse(wd._jarvis_running())

    def test_live_pid_means_running(self):
        import core.parent_watch as pw
        with mock.patch.object(wd.subprocess, "run", return_value=_cim([4242])), \
             mock.patch.object(pw, "parent_is_alive", return_value=True):
            self.assertTrue(wd._jarvis_running())

    def test_corpse_only_means_DEAD(self):
        # THE regression: rows exist, but every one of them is a kernel-stuck
        # corpse. The watchdog must resurrect, not sit on its hands.
        import core.parent_watch as pw
        with mock.patch.object(wd.subprocess, "run",
                               return_value=_cim([50916, 53452])), \
             mock.patch.object(pw, "parent_is_alive", return_value=False), \
             mock.patch.object(wd, "_note") as note:
            self.assertFalse(wd._jarvis_running())
        # and it says so in the log, so the next human knows why it booted
        self.assertTrue(any("CORPSE" in str(c) for c in note.call_args_list))

    def test_live_pid_beside_corpses_means_running(self):
        # A healthy instance next to yesterday's corpses must NOT be double-booted.
        import core.parent_watch as pw
        with mock.patch.object(wd.subprocess, "run",
                               return_value=_cim([50916, 7777])), \
             mock.patch.object(pw, "parent_is_alive",
                               side_effect=lambda p: p == 7777):
            self.assertTrue(wd._jarvis_running())

    def test_query_failure_fails_safe(self):
        # An unreadable process table must never cause a double boot.
        with mock.patch.object(wd.subprocess, "run",
                               side_effect=OSError("wmi down")):
            self.assertTrue(wd._jarvis_running())



# ══════════════════════════════════════════════════════════════════════════
# H-7 (2026-08-20) — the watchdog must not resurrect JARVIS into a tree that
# an upgrade run is mid-rewrite on, AND must not be parkable forever by a
# marker nobody cleaned up.
#
# The bug: the overnight path DOES write data/clean_shutdown.flag before
# spawning the pipeline, so the watchdog correctly stands down at the start.
# But the pipeline's tester stage boots a real prod JARVIS for its smoke test
# — and bobert_companion's boot path deletes clean_shutdown.flag
# unconditionally — then kills it with Stop-Process -Force, which never
# reaches atexit. No killer writes the flag back (`clean_shutdown` has zero
# hits in upgrade_jarvis.py / multi_agent_pipeline.py / stability_smoke_test.py
# / _boot_jarvis.ps1). From the first tester stage onward the watchdog saw
# "dead + no flag" for hours and booted a live-mic JARVIS every 5 minutes out
# of half-written source.
#
# The mirror-image bug this must never become: 2026-07-15 → 07-21, a crash
# left clean_shutdown.flag behind and the watchdog resurrected nothing for six
# days. Hence every "must not block forever" test below.
# ══════════════════════════════════════════════════════════════════════════


class _WdBase(unittest.TestCase):
    """Redirect wd.PROJ into a tempdir so no test can read or write the real
    C:\\JARVIS\\data flags (writing a real upgrade_in_progress.flag would park
    the live scheduled task for hours)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        os.makedirs(os.path.join(self.tmp, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp, "logs"), exist_ok=True)
        self._proj = wd.PROJ
        wd.PROJ = self.tmp
        self.addCleanup(setattr, wd, "PROJ", self._proj)

    # -- helpers --
    def marker_path(self):
        return os.path.join(self.tmp, "data", "upgrade_in_progress.flag")

    def write_marker(self, **over):
        now = time.time()
        payload = {"pid": os.getpid(), "started_at": now,
                   "heartbeat_at": now, "expires_at": now + 600,
                   "hard_deadline": now + 21600, "argv": "upgrade_jarvis.py"}
        payload.update(over)
        with open(self.marker_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return payload

    def disable_path(self):
        return os.path.join(self.tmp, "data", "watchdog_disabled.flag")


class UpgradeMarkerGateTests(_WdBase):
    def test_absent_marker_never_blocks(self):
        self.assertFalse(wd._upgrade_in_progress())

    def test_live_marker_blocks(self):
        # THE regression: an upgrade is rewriting the tree right now.
        self.write_marker()
        with mock.patch.object(wd, "_note"):
            self.assertTrue(wd._upgrade_in_progress())
        self.assertTrue(os.path.exists(self.marker_path()))

    def test_lapsed_lease_does_not_block_and_is_discarded(self):
        # The crashed-upgrade case: the heartbeat thread died with its process,
        # so the lease ran out. The net MUST re-arm, on this very tick.
        self.write_marker(expires_at=time.time() - 1)
        with mock.patch.object(wd, "_note") as note:
            self.assertFalse(wd._upgrade_in_progress())
        self.assertFalse(os.path.exists(self.marker_path()))
        self.assertTrue(any("ABANDONED" in str(c) for c in note.call_args_list))

    def test_hard_ceiling_overrides_a_still_fresh_lease(self):
        # A heartbeat thread that keeps running after the upgrade has wedged
        # would refresh expires_at forever. The absolute ceiling is the bound
        # that makes "parked forever" impossible even then.
        now = time.time()
        self.write_marker(started_at=now - 30000, expires_at=now + 600,
                          hard_deadline=now - 1)
        with mock.patch.object(wd, "_note") as note:
            self.assertFalse(wd._upgrade_in_progress())
        self.assertFalse(os.path.exists(self.marker_path()))
        self.assertTrue(any("ceiling" in str(c) for c in note.call_args_list))

    def test_dead_owner_abandons_marker_immediately(self):
        self.write_marker(pid=999999)
        with mock.patch.object(wd, "_pid_alive", return_value=False), \
             mock.patch.object(wd, "_note") as note:
            self.assertFalse(wd._upgrade_in_progress())
        self.assertFalse(os.path.exists(self.marker_path()))
        self.assertTrue(any("is gone" in str(c) for c in note.call_args_list))

    def test_marker_pid_check_assumes_alive_when_unresolvable(self):
        # The time bounds already cap the marker, so an unresolvable pid must
        # not be allowed to blow the gate open early.
        self.write_marker(pid=999999)
        with mock.patch.object(wd, "_pid_alive",
                               side_effect=lambda p, unknown: unknown), \
             mock.patch.object(wd, "_note"):
            self.assertTrue(wd._upgrade_in_progress())

    def test_missing_deadlines_are_treated_as_abandoned(self):
        # A marker with no expiry is precisely the park-forever shape.
        with open(self.marker_path(), "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid()}, f)
        with mock.patch.object(wd, "_note"):
            self.assertFalse(wd._upgrade_in_progress())
        self.assertFalse(os.path.exists(self.marker_path()))

    def test_unreadable_marker_blocks_only_briefly(self):
        with open(self.marker_path(), "w", encoding="utf-8") as f:
            f.write("not json at all")
        with mock.patch.object(wd, "_note"):
            self.assertTrue(wd._upgrade_in_progress())      # fresh -> grace
        old = time.time() - wd._MARKER_MALFORMED_GRACE_S - 5
        os.utime(self.marker_path(), (old, old))
        with mock.patch.object(wd, "_note"):
            self.assertFalse(wd._upgrade_in_progress())     # stale -> re-arm
        self.assertFalse(os.path.exists(self.marker_path()))

    def test_non_object_json_is_abandoned(self):
        with open(self.marker_path(), "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]")
        old = time.time() - wd._MARKER_MALFORMED_GRACE_S - 5
        os.utime(self.marker_path(), (old, old))
        with mock.patch.object(wd, "_note"):
            self.assertFalse(wd._upgrade_in_progress())


class DisableFlagExpiryTests(_WdBase):
    def test_absent_flag_does_not_hold(self):
        self.assertFalse(wd._disable_flag_holds())

    def test_fresh_flag_holds(self):
        open(self.disable_path(), "w").close()
        self.assertTrue(wd._disable_flag_holds())

    def test_expired_flag_is_ignored_but_NOT_deleted(self):
        # Bounded off-switch: authority lapses, the human's file stays put.
        open(self.disable_path(), "w").close()
        old = time.time() - wd._DISABLE_FLAG_MAX_AGE_S - 60
        os.utime(self.disable_path(), (old, old))
        with mock.patch.object(wd, "_note") as note:
            self.assertFalse(wd._disable_flag_holds())
        self.assertTrue(os.path.exists(self.disable_path()))
        self.assertTrue(any("EXPIRED" in str(c) for c in note.call_args_list))

    def test_permanent_body_never_expires(self):
        with open(self.disable_path(), "w", encoding="utf-8") as f:
            f.write("permanent - owner disabled this on purpose\n")
        old = time.time() - wd._DISABLE_FLAG_MAX_AGE_S * 10
        os.utime(self.disable_path(), (old, old))
        self.assertTrue(wd._disable_flag_holds())

    def test_max_age_env_override(self):
        open(self.disable_path(), "w").close()
        old = time.time() - 120
        os.utime(self.disable_path(), (old, old))
        with mock.patch.dict(os.environ,
                             {"JARVIS_WATCHDOG_DISABLE_MAX_AGE_S": "60"}), \
             mock.patch.object(wd, "_note"):
            self.assertFalse(wd._disable_flag_holds())


class PipelineRunningTests(_WdBase):
    def test_no_rows_means_no_pipeline(self):
        with mock.patch.object(wd.subprocess, "run", return_value=_cim([])):
            self.assertFalse(wd._pipeline_running())

    def test_live_driver_pid_blocks(self):
        with mock.patch.object(wd.subprocess, "run", return_value=_cim([31337])), \
             mock.patch.object(wd, "_pid_alive", return_value=True), \
             mock.patch.object(wd, "_note"):
            self.assertTrue(wd._pipeline_running())

    def test_corpse_only_does_not_block(self):
        # No expiry lives on this predicate, so a kernel-stuck upgrade_jarvis
        # row must never be able to park the net until the next reboot.
        with mock.patch.object(wd.subprocess, "run", return_value=_cim([31337])), \
             mock.patch.object(wd, "_pid_alive", return_value=False), \
             mock.patch.object(wd, "_note") as note:
            self.assertFalse(wd._pipeline_running())
        self.assertTrue(any("CORPSE" in str(c) for c in note.call_args_list))

    def test_query_failure_fails_OPEN(self):
        # Opposite of _jarvis_running() on purpose: that one fails safe because
        # a double boot is the risk; this one fails open because a permanent
        # block is the risk.
        with mock.patch.object(wd.subprocess, "run",
                               side_effect=OSError("wmi down")):
            self.assertFalse(wd._pipeline_running())

    def test_own_pid_is_ignored(self):
        with mock.patch.object(wd.subprocess, "run",
                               return_value=_cim([os.getpid()])):
            self.assertFalse(wd._pipeline_running())

    def test_cmdline_pattern_matches_every_pipeline_shape(self):
        for cmd in (
            r'"C:\Python\python.exe" -u "C:\JARVIS\upgrade_jarvis.py" --relaunch',
            r"python -u C:\Users\B\AppData\Local\Temp\tmpq7_jarvis_pipeline_loop.py",
            r"python -u /tmp/tmpq7_jarvis_upgrade_loop.py",
            r"python -m tools.multi_agent_pipeline",
        ):
            self.assertTrue(re.search(wd._PIPELINE_CMDLINE_RE, cmd, re.I), cmd)

    def test_cmdline_pattern_does_not_match_the_unit_suite(self):
        # An unanchored 'upgrade_jarvis\.py' also matches
        # tests/test_upgrade_jarvis.py — running the test suite would then have
        # parked the resurrection net for the length of the run.
        for cmd in (
            r"python C:\JARVIS\tests\test_upgrade_jarvis.py",
            r"python -m unittest tests.test_upgrade_jarvis",
            r"pythonw C:\JARVIS\tools\jarvis_watchdog.py",
        ):
            self.assertIsNone(re.search(wd._PIPELINE_CMDLINE_RE, cmd, re.I), cmd)


class MainGateTests(_WdBase):
    """End-to-end over main(): who gets resurrected and who does not."""

    def _run_main(self, jarvis_alive=False, pipeline=False):
        boots = []

        def _fake_run(cmd, *a, **k):
            boots.append(cmd)
            return mock.Mock(stdout="", returncode=0)

        with mock.patch.object(wd, "_jarvis_running", return_value=jarvis_alive), \
             mock.patch.object(wd, "_pipeline_running", return_value=pipeline), \
             mock.patch.object(wd.subprocess, "run", side_effect=_fake_run), \
             mock.patch.object(wd, "_note"):
            rc = wd.main()
        return rc, boots

    def test_normal_crash_still_resurrects(self):
        # No flags at all, JARVIS dead: the original contract must survive.
        rc, boots = self._run_main()
        self.assertEqual(rc, 0)
        self.assertTrue(any("_boot_jarvis.ps1" in str(c) for c in boots))

    def test_mid_upgrade_does_NOT_resurrect(self):
        self.write_marker()
        rc, boots = self._run_main()
        self.assertEqual(rc, 0)
        self.assertEqual(boots, [])

    def test_stale_marker_does_NOT_block_resurrection(self):
        # A crashed upgrade left its marker behind. The net must re-arm, and
        # the abandoned marker must be cleaned up so it cannot block again.
        self.write_marker(expires_at=time.time() - 1)
        rc, boots = self._run_main()
        self.assertTrue(any("_boot_jarvis.ps1" in str(c) for c in boots))
        self.assertFalse(os.path.exists(self.marker_path()))

    def test_marker_past_hard_ceiling_does_NOT_block_resurrection(self):
        now = time.time()
        self.write_marker(started_at=now - 40000, expires_at=now + 600,
                          hard_deadline=now - 1)
        rc, boots = self._run_main()
        self.assertTrue(any("_boot_jarvis.ps1" in str(c) for c in boots))

    def test_running_pipeline_without_a_marker_does_NOT_resurrect(self):
        rc, boots = self._run_main(pipeline=True)
        self.assertEqual(boots, [])

    def test_expired_disable_flag_no_longer_parks_the_net(self):
        open(self.disable_path(), "w").close()
        old = time.time() - wd._DISABLE_FLAG_MAX_AGE_S - 60
        os.utime(self.disable_path(), (old, old))
        rc, boots = self._run_main()
        self.assertTrue(any("_boot_jarvis.ps1" in str(c) for c in boots))

    def test_clean_shutdown_flag_still_wins(self):
        open(os.path.join(self.tmp, "data", "clean_shutdown.flag"), "w").close()
        rc, boots = self._run_main()
        self.assertEqual(boots, [])

    def test_live_jarvis_is_never_double_booted(self):
        rc, boots = self._run_main(jarvis_alive=True)
        self.assertEqual(boots, [])



class WriterReaderContractTests(_WdBase):
    """upgrade_jarvis.py writes the marker; this module reads it. Neither
    file's own unit tests can catch a drift between the two, and the whole
    design leans on the reader holding NO copy of the writer's timeout policy
    — it only compares the absolute deadlines in the file against now(). So
    drive the real writer and the real reader against one shared tree."""

    def setUp(self):
        super().setUp()
        import upgrade_jarvis as U
        self.U = U
        self._saved_proj = U.PROJECT_DIR
        self._saved_state = dict(U._marker_state)
        U.PROJECT_DIR = self.tmp          # same tree the watchdog is reading
        self.addCleanup(self._restore_writer)
        U._marker_state.clear()
        U._marker_state.update({"depth": 0, "owned": False, "stop": None,
                                "thread": None, "started_at": 0.0,
                                "hard_deadline": 0.0})

    def _restore_writer(self):
        U = self.U
        for _ in range(10):
            if not (U._marker_state.get("depth") or U._marker_state.get("owned")):
                break
            U.release_upgrade_marker()
        U._marker_state.clear()
        U._marker_state.update(self._saved_state)
        U.PROJECT_DIR = self._saved_proj

    def _boots(self):
        """Run watchdog main() with JARVIS dead and return the boot argvs."""
        seen = []

        def _fake_run(cmd, *a, **k):
            seen.append(cmd)
            return mock.Mock(stdout="", returncode=0)

        with mock.patch.object(wd, "_jarvis_running", return_value=False), \
             mock.patch.object(wd, "_pipeline_running", return_value=False), \
             mock.patch.object(wd.subprocess, "run", side_effect=_fake_run), \
             mock.patch.object(wd, "_note"), \
             mock.patch("sys.stdout", io.StringIO()):
            wd.main()
        return seen

    def test_a_real_acquire_stops_the_real_watchdog(self):
        # THE H-7 regression, end to end across both files.
        with mock.patch("sys.stdout", io.StringIO()):
            self.assertTrue(self.U.acquire_upgrade_marker())
        self.assertEqual(self.U._upgrade_marker_path(), self.marker_path())
        with mock.patch.object(wd, "_note"):
            self.assertTrue(wd._upgrade_in_progress())
        self.assertEqual(self._boots(), [])

    def test_a_real_release_re_arms_the_real_watchdog(self):
        with mock.patch("sys.stdout", io.StringIO()):
            self.U.acquire_upgrade_marker()
            self.U.release_upgrade_marker()
        self.assertFalse(wd._upgrade_in_progress())
        self.assertTrue(any("_boot_jarvis.ps1" in str(c) for c in self._boots()))

    def test_a_crashed_upgrade_cannot_park_the_net_forever(self):
        # Acquire, then simulate the process vanishing without releasing: the
        # heartbeat stops and the lease runs out. This is the mirror-image bug
        # (July's six-day outage) and the marker must NOT survive it.
        with mock.patch("sys.stdout", io.StringIO()):
            self.U.acquire_upgrade_marker()
        self.U._marker_state["stop"].set()          # heartbeat is gone
        data = json.load(open(self.marker_path(), encoding="utf-8"))
        data["expires_at"] = time.time() - 1        # lease lapsed
        with open(self.marker_path(), "w", encoding="utf-8") as f:
            json.dump(data, f)

        self.assertTrue(any("_boot_jarvis.ps1" in str(c) for c in self._boots()))
        self.assertFalse(os.path.exists(self.marker_path()))

    def test_writer_deadlines_are_what_the_reader_expects(self):
        # Schema guard: the reader keys off exactly these three fields.
        with mock.patch("sys.stdout", io.StringIO()):
            self.U.acquire_upgrade_marker()
        data = json.load(open(self.marker_path(), encoding="utf-8"))
        for key in ("pid", "expires_at", "hard_deadline"):
            self.assertIn(key, data)
        self.assertIsInstance(data["pid"], int)
        self.assertGreater(data["hard_deadline"], data["expires_at"])

    def test_retuning_the_writer_timeout_needs_no_reader_edit(self):
        # The point of putting ABSOLUTE deadlines in the file: the reader has
        # no copy of the policy to keep in sync (this project's #1 bug class).
        with mock.patch.dict(os.environ,
                             {"JARVIS_UPGRADE_MARKER_TTL_S": "120",
                              "JARVIS_UPGRADE_MARKER_MAX_AGE_S": "180"}), \
             mock.patch("sys.stdout", io.StringIO()):
            self.U.acquire_upgrade_marker()
        data = json.load(open(self.marker_path(), encoding="utf-8"))
        self.assertAlmostEqual(data["hard_deadline"] - data["started_at"],
                               180, delta=2)
        with mock.patch.object(wd, "_note"):
            self.assertTrue(wd._upgrade_in_progress())
        # ...and once that shorter ceiling passes, the reader re-arms with no
        # knowledge of where 180 came from.
        data["hard_deadline"] = time.time() - 1
        with open(self.marker_path(), "w", encoding="utf-8") as f:
            json.dump(data, f)
        with mock.patch.object(wd, "_note"):
            self.assertFalse(wd._upgrade_in_progress())


if __name__ == "__main__":
    unittest.main()
