"""Logic tests for skills/system_monitor.py.

The check_system action builds a JARVIS-cadence CPU/RAM/disk/network report
from psutil. We pin every psutil source (cpu_percent, virtual_memory,
disk_usage) and stub the two helper collectors (_top_processes, _network_rates)
so the rendered sentence is deterministic and no live machine probe runs.

The background _monitor_loop is never started (harness neuters threads); we
test _build_report and the graceful psutil-missing degradation instead.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _LoopBreak(BaseException):
    """Raised from a stubbed time.sleep to break a `while True` background loop
    deterministically. A BaseException (not Exception) so the loop's own
    `except Exception` handlers don't swallow it."""


class _VM:
    """Minimal stand-in for psutil.virtual_memory()."""
    def __init__(self, percent, used_gb, total_gb=32.0):
        self.percent = percent
        self.used = int(used_gb * 1024 ** 3)
        self.total = int(total_gb * 1024 ** 3)


class _Disk:
    def __init__(self, free_gb, total_gb):
        self.free = int(free_gb * 1024 ** 3)
        self.total = int(total_gb * 1024 ** 3)


class _Net:
    def __init__(self, recv, sent):
        self.bytes_recv = recv
        self.bytes_sent = sent


class SystemMonitorReportTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")

    def _run_report(self, *, cpu, ram_pct, ram_used=8.0, top=None,
                    net=(0.0, 0.0), disk=(500.0, 1000.0)):
        top = top if top is not None else []
        fake_psutil = mock.MagicMock()
        fake_psutil.cpu_percent.return_value = cpu
        fake_psutil.virtual_memory.return_value = _VM(ram_pct, ram_used)
        fake_psutil.disk_usage.return_value = _Disk(*disk)
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake_psutil), \
             mock.patch.object(self.mod, "_top_processes", return_value=top), \
             mock.patch.object(self.mod, "_network_rates", return_value=net):
            return self.mod._build_report()

    def test_nominal_opener(self):
        out = self._run_report(cpu=10, ram_pct=40)
        self.assertTrue(out.startswith("Systems nominal, sir."))
        self.assertIn("CPU at 10 percent", out)

    def test_holding_opener(self):
        out = self._run_report(cpu=70, ram_pct=85)
        self.assertTrue(out.startswith("Systems holding up, sir."))

    def test_working_hard_opener(self):
        out = self._run_report(cpu=95, ram_pct=92)
        self.assertIn("working rather hard", out)

    def test_chrome_offender_phrasing(self):
        out = self._run_report(cpu=50, ram_pct=50,
                               top=[("chrome.exe", 88.0)])
        self.assertIn("Chrome is, as usual, the primary offender", out)

    def test_generic_offender_phrasing(self):
        # Spoken the way he would say it (2026-10-02): no ".exe", and one
        # period - it used to read "blender.exe is the primary offender..".
        out = self._run_report(cpu=50, ram_pct=50,
                               top=[("blender.exe", 70.0)])
        self.assertIn("Blender is the primary offender. C drive", out)
        self.assertNotIn(".exe", out)
        self.assertNotIn("..", out)

    def test_defender_named_as_defender_in_the_report(self):
        out = self._run_report(cpu=95, ram_pct=50, top=[("MsMpEng.exe", 80.0)])
        self.assertIn("Windows Defender is the primary offender", out)
        self.assertNotIn("MsMpEng", out)

    def test_own_process_in_the_report(self):
        out = self._run_report(cpu=95, ram_pct=50,
                               top=[(self.mod._SELF_KEY, 80.0)])
        self.assertIn("Most of that is my own speech recognition", out)

    def test_disk_line_included(self):
        out = self._run_report(cpu=10, ram_pct=30, disk=(123.0, 1000.0))
        self.assertIn("C drive has 123 gigs free of 1000", out)

    def test_network_line_included_when_active(self):
        out = self._run_report(cpu=10, ram_pct=30, net=(2048.0, 512.0))
        self.assertIn("network at 2048 down, 512 up", out)

    def test_network_line_omitted_when_quiet(self):
        out = self._run_report(cpu=10, ram_pct=30, net=(1.0, 1.0))  # < 5 kbps
        self.assertNotIn("network at", out)

    def test_ram_committed_reported(self):
        out = self._run_report(cpu=10, ram_pct=30, ram_used=12.0)
        self.assertIn("12 of 32 gigs committed", out)

    def test_disk_failure_omits_disk_line_and_no_extras(self):
        # disk_usage raises → c_total_gb=0.0 → disk line dropped (173->175);
        # with net quiet there are no extras at all → the no-extras return (181).
        fake = mock.MagicMock()
        fake.cpu_percent.return_value = 10
        fake.virtual_memory.return_value = _VM(30, 8.0)
        fake.disk_usage.side_effect = OSError("no C: drive")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod, "_top_processes", return_value=[]), \
             mock.patch.object(self.mod, "_network_rates", return_value=(0.0, 0.0)):
            out = self.mod._build_report()
        self.assertNotIn("C drive", out)
        self.assertNotIn("network at", out)
        self.assertTrue(out.endswith("committed."))

    def test_network_line_present_but_no_disk(self):
        # disk fails (no disk line) yet network is active → extras_str is just
        # the network clause, exercising the extras-present return (180).
        fake = mock.MagicMock()
        fake.cpu_percent.return_value = 10
        fake.virtual_memory.return_value = _VM(30, 8.0)
        fake.disk_usage.side_effect = OSError("no C: drive")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod, "_top_processes", return_value=[]), \
             mock.patch.object(self.mod, "_network_rates", return_value=(99.0, 9.0)):
            out = self.mod._build_report()
        self.assertNotIn("C drive", out)
        self.assertIn("network at 99 down, 9 up", out)


class SystemMonitorDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")

    def test_build_report_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            out = self.mod._build_report()
        self.assertIn("requires the psutil package", out)

    def test_check_system_action_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            out = self.actions["check_system"]("")
        self.assertIn("psutil", out.lower())

    def test_check_system_wraps_exceptions(self):
        with mock.patch.object(self.mod, "_build_report",
                               side_effect=RuntimeError("boom")):
            out = self.actions["check_system"]("")
        self.assertIn("system check failed", out.lower())
        self.assertIn("boom", out)

    def test_top_processes_empty_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._top_processes(3), [])

    def test_network_rates_zero_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._network_rates(), (0.0, 0.0))


class SystemMonitorTopProcessTests(unittest.TestCase):
    """_top_processes aggregates per-name CPU and drops the idle process."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")

    def test_aggregates_and_sorts_and_drops_idle(self):
        # Two chrome children + one idle process. cpu_percent is called twice
        # per proc (throwaway then real); we return the "real" value on the
        # second read for each.
        def _proc(name, cpu):
            p = mock.MagicMock()
            p.info = {"name": name}
            p.cpu_percent.side_effect = [0.0, cpu]
            return p

        procs = [_proc("chrome.exe", 30.0), _proc("chrome.exe", 25.0),
                 _proc("System Idle Process", 99.0), _proc("python.exe", 10.0)]
        fake_psutil = mock.MagicMock()
        fake_psutil.process_iter.return_value = procs
        # psutil.NoSuchProcess / AccessDenied must be real exception classes.
        fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake_psutil), \
             mock.patch.object(self.mod.time, "sleep"):
            top = self.mod._top_processes(3)
        names = [n for n, _ in top]
        self.assertNotIn("System Idle Process", names)
        # chrome aggregated to 55.0, ahead of python at 10.0
        self.assertEqual(top[0][0], "chrome.exe")
        self.assertAlmostEqual(top[0][1], 55.0)

    def test_skips_procs_that_vanish_on_either_read(self):
        # A process that raises NoSuchProcess on the FIRST cpu_percent (warmup)
        # is dropped before the list; one that raises on the SECOND (rated) read
        # is skipped too. Only the survivor is returned, and a proc whose
        # info has no name falls back to "pid <n>".
        NoSuch = type("NoSuchProcess", (Exception,), {})
        Denied = type("AccessDenied", (Exception,), {})

        def _proc(name, warmup_exc=None, rated_exc=None, cpu=0.0, pid=7):
            p = mock.MagicMock()
            p.pid = pid
            p.info = {"name": name}
            seq = []
            seq.append(warmup_exc if warmup_exc else 0.0)
            seq.append(rated_exc if rated_exc else cpu)
            p.cpu_percent.side_effect = seq
            return p

        gone_warmup = _proc("ghost.exe", warmup_exc=NoSuch())
        gone_rated = _proc("dying.exe", rated_exc=Denied())
        survivor = _proc("ok.exe", cpu=42.0)
        noname = _proc(None, cpu=5.0, pid=999)   # name None → "pid 999"
        fake = mock.MagicMock()
        fake.process_iter.return_value = [gone_warmup, gone_rated, survivor, noname]
        fake.NoSuchProcess = NoSuch
        fake.AccessDenied = Denied
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "sleep"):
            top = self.mod._top_processes(5)
        names = [n for n, _ in top]
        self.assertIn("ok.exe", names)
        self.assertIn("pid 999", names)
        self.assertNotIn("ghost.exe", names)
        self.assertNotIn("dying.exe", names)


class SystemMonitorNetworkRatesTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")

    def test_network_rates_computes_kbps_over_window(self):
        # +1,048,576 bytes recv and +524,288 sent over a 1s window =
        # 1024 kB/s down, 512 kB/s up.
        fake = mock.MagicMock()
        fake.net_io_counters.side_effect = [_Net(0, 0),
                                            _Net(1024 * 1024, 512 * 1024)]
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "sleep"):
            down, up = self.mod._network_rates(window_seconds=1.0)
        self.assertAlmostEqual(down, 1024.0)
        self.assertAlmostEqual(up, 512.0)


class SystemMonitorEnqueueSpeechTests(unittest.TestCase):
    """_enqueue_speech prefers bobert_companion.proactive_announce(), else
    falls back to an atomic append against pending_speech.json."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")
        self.tmp = tempfile.mkdtemp(prefix="sysmon_speech_")
        self.queue = os.path.join(self.tmp, "pending_speech.json")
        self.mod._SPEECH_QUEUE = self.queue
        self._saved_bc = sys.modules.get("bobert_companion")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self._saved_bc is not None:
            sys.modules["bobert_companion"] = self._saved_bc
        else:
            sys.modules.pop("bobert_companion", None)
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_routes_through_proactive_announce_when_available(self):
        bc = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.mod._enqueue_speech("hello sir")
        bc.proactive_announce.assert_called_once()
        # message passed positionally, source kwarg identifies the skill
        self.assertEqual(bc.proactive_announce.call_args[0][0], "hello sir")
        self.assertEqual(bc.proactive_announce.call_args[1]["source"],
                         "system_monitor")
        # Announce handled it → no fallback file written.
        self.assertFalse(os.path.exists(self.queue))

    def test_falls_back_to_atomic_write_when_no_parent(self):
        # bobert_companion absent → direct append to the speech queue file.
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("bobert_companion", None)
            with mock.patch("importlib.import_module",
                            side_effect=ImportError("no bobert yet")):
                self.mod._enqueue_speech("fallback line")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "fallback line")

    def test_fallback_appends_to_existing_and_tolerates_corrupt_queue(self):
        # Pre-existing corrupt JSON is treated as empty, then the new entry is
        # appended (so a garbled file can't drop the alert).
        with open(self.queue, "w", encoding="utf-8") as f:
            f.write("{ this is not valid json")
        with mock.patch("importlib.import_module",
                        side_effect=ImportError("no bobert")):
            self.mod._enqueue_speech("after corruption")
        with open(self.queue, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["after corruption"])

    def test_announce_attr_missing_falls_through_to_file(self):
        # Parent imports but exposes no proactive_announce → fallback path.
        bc = mock.MagicMock(spec=[])   # no attributes at all
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch("importlib.import_module", return_value=bc):
            self.mod._enqueue_speech("attr missing")
        with open(self.queue, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["message"], "attr missing")

    def test_write_failure_is_caught_and_logged(self):
        # The atomic writer raising must not propagate — the alert degrades to a
        # console print instead of crashing the monitor thread.
        with mock.patch("importlib.import_module",
                        side_effect=ImportError("no bobert")), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            # Should not raise.
            self.mod._enqueue_speech("doomed write")


class SystemMonitorLoopTests(unittest.TestCase):
    """Drive _monitor_loop deterministically: time is controlled, the sliding
    window is advanced by hand, and a stubbed time.sleep raises _LoopBreak to
    exit after the iteration(s) under test."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")
        self.mod._last_cpu_alert_at[0] = 0.0
        self.mod._last_ram_alert_at[0] = 0.0

    def _make_psutil(self, cpu_seq, ram_seq):
        fake = mock.MagicMock()
        fake.cpu_percent.side_effect = list(cpu_seq)
        fake.virtual_memory.side_effect = [_VM(p, 8.0) for p in ram_seq]
        return fake

    def test_loop_returns_immediately_without_psutil(self):
        # No psutil → loop never enters; initial sleep also never reached.
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False), \
             mock.patch.object(self.mod.time, "sleep") as slept:
            self.mod._monitor_loop()
        slept.assert_not_called()

    def test_sustained_high_cpu_queues_alert_with_culprit(self):
        # 13 samples all >=90% spanning >0.9*SUSTAIN seconds → ratio 1.0 fires
        # the CPU alert exactly once, naming the culprit from _top_processes(1).
        n = 13
        cpu_seq = [95.0] * n
        ram_seq = [40.0] * n
        fake = self._make_psutil(cpu_seq, ram_seq)
        # Advance wall clock 6s per sample so the window span exceeds 0.9*60s.
        base = 1_700_000_000.0
        ticks = iter(base + 6.0 * i for i in range(n + 5))

        def _now():
            return next(ticks)

        def _sleep(_):
            # Break once the alert has been queued (cooldown stamp set).
            if self.mod._last_cpu_alert_at[0] > 0.0:
                raise _LoopBreak
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", _now), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_top_processes",
                               return_value=[("blender.exe", 97.0)]), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        enq.assert_called_once()
        msg = enq.call_args[0][0]
        self.assertIn("pinned above 90 percent", msg)
        self.assertIn("Blender appears to be the culprit", msg)
        self.assertNotIn(".exe", msg)
        self.assertGreater(self.mod._last_cpu_alert_at[0], 0.0)

    def test_high_ram_single_sample_queues_alert(self):
        # One sample with RAM >= 90% fires the RAM alert immediately; CPU low so
        # only the memory message is queued.
        fake = self._make_psutil([10.0], [93.0])
        base = 1_700_000_000.0

        def _sleep(_):
            if self.mod._last_ram_alert_at[0] > 0.0:
                raise _LoopBreak
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", return_value=base), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        enq.assert_called_once()
        self.assertIn("memory usage is at 93 percent", enq.call_args[0][0])

    def test_ram_alert_respects_cooldown(self):
        # RAM already alerted "just now" → within ALERT_COOLDOWN_SECONDS a high
        # sample does NOT re-queue. Two iterations so the 229->245 fall-through
        # (RAM high, cooldown active) is exercised as a completed arc, not only
        # an arc terminated by the break.
        base = 1_700_000_000.0
        self.mod._last_ram_alert_at[0] = base - 10   # 10s ago, < 600s cooldown
        fake = self._make_psutil([10.0, 10.0], [95.0, 95.0])
        calls = {"n": 0}

        def _sleep(_):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise _LoopBreak   # let the first iteration complete cleanly
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", return_value=base), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        enq.assert_not_called()

    def _run_one_high_ram_sample(self, pulse_mod):
        fake = self._make_psutil([10.0, 10.0], [93.0, 93.0])
        base = 1_700_000_000.0
        calls = {"n": 0}

        def _sleep(_):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise _LoopBreak
        mods = {} if pulse_mod is None else {"skill_system_pulse": pulse_mod}
        with mock.patch.dict(sys.modules, mods), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", return_value=base), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            if pulse_mod is None:
                sys.modules.pop("skill_system_pulse", None)
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        return enq, base

    def test_ram_alert_yields_to_recent_system_pulse_ram_alert(self):
        # 2026-10-01 regression: the pulse had just spoken about RAM, yet this
        # monitor spoke too (2026-09-30, three RAM alerts in 3.5 min). A fresh
        # pulse "ram" stamp now holds the slot.
        pulse = types.SimpleNamespace(
            _alert_lock=threading.Lock(),
            _last_abnormal_alert={"ram": 1_700_000_000.0 - 120.0},
            PROACTIVE_COOLDOWN_SECONDS=3600)
        enq, _ = self._run_one_high_ram_sample(pulse)
        enq.assert_not_called()

    def test_ram_alert_claims_the_shared_pulse_slot(self):
        # ...and when this monitor speaks first it stamps the pulse's "ram"
        # key, so the pulse's next 15-min check stays quiet for the hour.
        pulse = types.SimpleNamespace(
            _alert_lock=threading.Lock(), _last_abnormal_alert={},
            PROACTIVE_COOLDOWN_SECONDS=3600)
        enq, base = self._run_one_high_ram_sample(pulse)
        enq.assert_called_once()
        self.assertEqual(pulse._last_abnormal_alert["ram"], base)

    def test_ram_alert_without_pulse_uses_own_cooldown(self):
        enq, _ = self._run_one_high_ram_sample(None)
        enq.assert_called_once()

    def test_single_low_sample_does_not_evaluate_window(self):
        # First iteration: one sample, window span 0.0 < threshold, RAM low →
        # neither alert path runs; loop falls straight to the cadence sleep.
        fake = self._make_psutil([20.0], [50.0])
        base = 1_700_000_000.0

        def _sleep(_):
            raise _LoopBreak   # break after the very first iteration
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", return_value=base), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        enq.assert_not_called()

    def test_full_window_below_ratio_does_not_alert(self):
        # 13 samples, only a few high → high_count/len < 0.8 even once the
        # window is full, so the CPU alert never fires (ratio branch 212->228).
        n = 13
        cpu_seq = ([95.0] * 3) + ([10.0] * (n - 3))   # 3/13 high ≈ 0.23
        ram_seq = [40.0] * n
        fake = self._make_psutil(cpu_seq, ram_seq)
        base = 1_700_000_000.0
        ticks = iter(base + 6.0 * i for i in range(n + 5))
        calls = {"n": 0}

        def _sleep(_):
            calls["n"] += 1
            if calls["n"] >= n:
                raise _LoopBreak
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", side_effect=lambda: next(ticks)), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        enq.assert_not_called()

    def test_full_high_window_blocked_by_cpu_cooldown(self):
        # Window full of highs (ratio met) BUT a CPU alert fired recently →
        # cooldown branch (213->228) suppresses the repeat.
        n = 13
        base = 1_700_000_000.0
        self.mod._last_cpu_alert_at[0] = base + 6.0 * n   # "now"-ish, within cooldown
        fake = self._make_psutil([95.0] * n, [40.0] * n)
        ticks = iter(base + 6.0 * i for i in range(n + 5))
        calls = {"n": 0}

        def _sleep(_):
            calls["n"] += 1
            if calls["n"] >= n:
                raise _LoopBreak
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", side_effect=lambda: next(ticks)), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        enq.assert_not_called()

    def test_loop_iteration_exception_is_caught_and_loop_continues(self):
        # cpu_percent raising Exception is swallowed by the loop's try/except,
        # which then sleeps POLL_INTERVAL; our stub raises _LoopBreak there to
        # confirm the except branch (and its recovery sleep) executed.
        fake = mock.MagicMock()
        fake.cpu_percent.side_effect = RuntimeError("psutil hiccup")
        seen = {"poll_sleep": False}

        def _sleep(secs):
            # The except-branch recovery sleep uses POLL_INTERVAL_SECONDS.
            if secs == self.mod.POLL_INTERVAL_SECONDS:
                seen["poll_sleep"] = True
                raise _LoopBreak
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.logging, "exception") as logexc:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        self.assertTrue(seen["poll_sleep"])
        logexc.assert_called()


class SystemMonitorImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        # Re-exec the source with the project root removed from sys.path so the
        # `if _PROJECT_DIR not in sys.path: sys.path.insert(...)` guard runs.
        # core.atomic_io is already cached, so the from-import still resolves.
        mod, _ = load_skill_isolated("system_monitor")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("system_monitor_reexec", path)
        m = importlib.util.module_from_spec(spec)
        m.skill_utils = {}
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.abspath(p) != os.path.abspath(proj)]
            spec.loader.exec_module(m)
            self.assertIn(m._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved


class SystemMonitorRegisterTests(unittest.TestCase):
    def test_register_with_psutil_starts_monitor_thread(self):
        # Harness neuters Thread.start, but register() should still construct the
        # daemon thread and register the action.
        mod, actions = load_skill_isolated("system_monitor")
        self.assertIn("check_system", actions)

    def test_register_without_psutil_skips_thread(self):
        # With psutil absent at register time, the action is still registered and
        # no thread machinery runs. Re-register on a freshly loaded module.
        mod, _ = load_skill_isolated("system_monitor")
        actions = {}
        with mock.patch.object(mod, "_HAS_PSUTIL", False), \
             mock.patch("threading.Thread") as Thread:
            mod.register(actions)
        self.assertIn("check_system", actions)
        Thread.assert_not_called()



class _CT:
    """psutil.Process.cpu_times() stand-in."""
    def __init__(self, user, system=0.0):
        self.user = user
        self.system = system


class SystemMonitorCulpritTests(unittest.TestCase):
    """NEW #14 (live 2026-10-01). 18:14:56, over a show: "MsMpEng.exe appears
    to be the culprit. You may want to investigate" - a routine Defender scan.
    22:06:12: "python.exe ..." - JARVIS's own process. Both culprits came from
    ONE 0.5 s _top_processes(1) sample taken after the window, spoken as the
    raw image name. Now: averaged over the window, named the way he would say
    it, maintenance log-only, JARVIS itself = "my own speech recognition"."""

    N = 13          # samples 6 s apart: the window fills and the alert fires

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_monitor")
        self.mod._last_cpu_alert_at[0] = 0.0
        self.mod._last_ram_alert_at[0] = 0.0

    def _psutil(self, rates):
        """A psutil whose process_iter reports, on its k-th call, each pid's
        CPU time as rate * 6 s * k (rates: {pid: (name, cpu_s_per_s)})."""
        fake = mock.MagicMock()
        fake.cpu_percent.side_effect = [96.0] * self.N
        fake.virtual_memory.side_effect = [_VM(40.0, 8.0)] * self.N
        fake.cpu_count.return_value = 8
        calls = {"k": 0}

        def _iter(_attrs=None):
            calls["k"] += 1
            out = []
            for pid, (name, rate) in rates.items():
                p = mock.MagicMock()
                p.pid = pid
                p.info = {"name": name,
                          "cpu_times": _CT(rate * 6.0 * calls["k"])}
                out.append(p)
            return out
        fake.process_iter.side_effect = _iter
        return fake

    def _run(self, rates, *, instant_top):
        """Drive _monitor_loop through one full high window. `instant_top` is
        what the old one-instant _top_processes(1) sample would have said."""
        base = 1_700_000_000.0
        ticks = iter(base + 6.0 * i for i in range(self.N + 5))

        def _sleep(_):
            if self.mod._last_cpu_alert_at[0] > 0.0:
                raise _LoopBreak
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", self._psutil(rates)), \
             mock.patch.object(self.mod.time, "time", lambda: next(ticks)), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_top_processes",
                               return_value=instant_top), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch("builtins.print") as pr:
            with self.assertRaises(_LoopBreak):
                self.mod._monitor_loop()
        logged = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        spoken = [c.args[0] for c in enq.call_args_list]
        return spoken, logged

    def test_live_18_14_defender_scan_is_logged_not_spoken(self):
        spoken, logged = self._run(
            {101: ("MsMpEng.exe", 6.5), 102: ("chrome.exe", 0.4),
             103: ("System Idle Process", 1.0)},
            instant_top=[("MsMpEng.exe", 41.0)])
        self.assertEqual(spoken, [], "a routine Defender scan was announced")
        self.assertIn("Windows Defender (msmpeng.exe", logged)
        self.assertIn("routine maintenance, not announced", logged)
        # The window still counts: no re-log every 5 s.
        self.assertGreater(self.mod._last_cpu_alert_at[0], 0.0)

    def test_live_22_06_own_process_is_my_own_speech_recognition(self):
        own = os.getpid()
        spoken, _ = self._run({own: ("python.exe", 5.0),
                               202: ("chrome.exe", 0.5)},
                              instant_top=[("python.exe", 60.0)])
        self.assertEqual(len(spoken), 1)
        self.assertIn("mostly my own speech recognition", spoken[0])
        self.assertNotIn("python", spoken[0].lower())

    def test_the_culprit_is_averaged_over_the_window(self):
        # Chrome burned the minute; Blender only tops the final instant.
        spoken, _ = self._run({301: ("chrome.exe", 5.0),
                               302: ("blender.exe", 0.2)},
                              instant_top=[("blender.exe", 90.0)])
        self.assertEqual(len(spoken), 1)
        self.assertIn("Chrome appears to be the culprit", spoken[0])
        self.assertNotIn("lender", spoken[0])

    def test_no_baseline_falls_back_to_the_instant_sample(self):
        spoken, _ = self._run({}, instant_top=[("blender.exe", 90.0)])
        self.assertEqual(len(spoken), 1)
        self.assertIn("Blender appears to be the culprit", spoken[0])

    def test_no_culprit_at_all_is_still_reported(self):
        spoken, _ = self._run({}, instant_top=[])
        self.assertEqual(spoken, ["Sir, CPU usage has been pinned above 90 "
                                  "percent for most of the past minute. You "
                                  "may want to investigate."])

    # ── the pure helpers ─────────────────────────────────────────────────
    def test_window_culprit_math(self):
        base = {1: ("chrome.exe", 100.0), 2: ("MsMpEng.exe", 50.0),
                0: ("System Idle Process", 1000.0)}
        now = {1: ("chrome.exe", 130.0), 2: ("MsMpEng.exe", 60.0),
               0: ("System Idle Process", 1400.0),
               3: ("new.exe", 20.0)}          # started inside the window
        key, pct = self.mod._window_culprit(base, now, 60.0, own_pid=999,
                                            ncpu=4)
        self.assertEqual(key, "chrome.exe")
        self.assertAlmostEqual(pct, 30.0 / 60.0 / 4 * 100.0)
        self.assertIsNone(self.mod._window_culprit({}, now, 60.0))
        self.assertIsNone(self.mod._window_culprit(base, now, 0.0))

    def test_window_culprit_reused_pid_counts_from_zero(self):
        base = {7: ("old.exe", 500.0)}
        now = {7: ("other.exe", 12.0)}
        key, _ = self.mod._window_culprit(base, now, 60.0, own_pid=1, ncpu=1)
        self.assertEqual(key, "other.exe")

    def test_spoken_names(self):
        f = self.mod._spoken_process_name
        self.assertEqual(f("MsMpEng.exe"), "Windows Defender")
        self.assertEqual(f("TiWorker.exe"), "Windows Update")
        self.assertEqual(f("chrome.exe"), "Chrome")
        self.assertEqual(f("FortniteClient-Win64-Shipping.exe"), "Fortnite")
        self.assertEqual(f("blender.exe"), "Blender")
        self.assertEqual(f("SomeTool.EXE"), "SomeTool")
        self.assertEqual(f(self.mod._SELF_KEY), "my own speech recognition")
        self.assertEqual(f(""), "an unnamed process")

    def test_alert_lines(self):
        line = self.mod._cpu_alert_line
        self.assertIsNone(line("msmpeng.exe"))
        self.assertIsNone(line("searchindexer.exe"))
        self.assertNotIn("?", line("chrome.exe"))
        self.assertIn("Chrome appears to be the culprit", line("chrome.exe"))

    def test_top_processes_marks_jarvis_itself(self):
        own = os.getpid()

        def _proc(name, cpu, pid):
            p = mock.MagicMock()
            p.pid = pid
            p.info = {"name": name}
            p.cpu_percent.side_effect = [0.0, cpu]
            return p
        fake = mock.MagicMock()
        fake.process_iter.return_value = [_proc("python.exe", 70.0, own),
                                          _proc("python.exe", 5.0, own + 1)]
        fake.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake.AccessDenied = type("AccessDenied", (Exception,), {})
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "sleep"):
            top = self.mod._top_processes(2)
        self.assertEqual(top[0], (self.mod._SELF_KEY, 70.0))
        self.assertEqual(top[1], ("python.exe", 5.0))


if __name__ == "__main__":
    unittest.main()
