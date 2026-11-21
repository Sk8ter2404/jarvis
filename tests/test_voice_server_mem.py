"""core/voice_server_mem.py -- the clone voice server's memory, read from
JARVIS's side (v3.0.27, overhaul finding reliability-08).

Live 2026-11-20 20:51: the RAM alert's "[mem-users]" line named the voice
server (pid 24620, 3.0 GB) only because RAM was at 95 % and it was in the top
three. Pinned here:
* a reading of the pid the server reported (psutil RSS) is logged the first
  time and again after a LOG_STEP_MB move, not every minute;
* passing VOICE_SERVER_RSS_WARN_MB logs ONE warning per server pid and runs
  ``on_warn`` once; 0 turns the warning off;
* no pid / a gone process -> nothing, never an exception;
* REPORT ONLY: the module never calls anything on the process beyond
  memory_info() (no kill / terminate / suspend / restart).

Stdlib only (psutil faked): CI-light tier.
"""
from __future__ import annotations

import os
import re
import sys
import types
import unittest
from unittest import mock

from core import voice_server_mem as vsm

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MB = 1024 * 1024


class _Proc:
    calls: list = []

    def __init__(self, table, pid):
        if pid not in table:
            raise LookupError("no such process")
        self._table = table
        self._pid = pid

    def memory_info(self):
        _Proc.calls.append(("memory_info", self._pid))
        return types.SimpleNamespace(rss=int(self._table[self._pid] * _MB))

    def __getattr__(self, name):        # kill / terminate / suspend / ...
        _Proc.calls.append((name, self._pid))
        raise AssertionError(f"voice_server_mem touched the process: {name}")


class _FakePsutil:
    def __init__(self, table):
        self.table = table

    def Process(self, pid):             # noqa: N802 - psutil's name
        return _Proc(self.table, pid)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _mon(table, pid=24620, warn=6000.0, clock=None):
    logs = []
    m = vsm.VoiceServerMem(pid_fn=lambda: pid, warn_mb=lambda: warn,
                           log=logs.append, clock=clock or _Clock(),
                           psutil_mod=_FakePsutil(table))
    return m, logs


class ReadingTests(unittest.TestCase):
    def setUp(self):
        _Proc.calls = []

    def test_first_reading_is_logged_with_pid_and_gb(self):
        m, logs = _mon({24620: 3072.0})
        self.assertEqual(m.sample(), (24620, 3072.0))
        self.assertEqual(logs, ["  [voice-mem] voice server pid 24620 rss "
                                "3.0 GB"])
        self.assertEqual(m.describe(), "voice server pid 24620 rss 3.0 GB")

    def test_small_moves_are_not_logged_again(self):
        table = {24620: 3072.0}
        m, logs = _mon(table)
        m.sample()
        table[24620] = 3072.0 + vsm.LOG_STEP_MB - 1
        m.sample()
        self.assertEqual(len(logs), 1)
        table[24620] = 3072.0 + vsm.LOG_STEP_MB + 1
        m.sample()
        self.assertEqual(len(logs), 2)
        self.assertIn("rss 4.0 GB", logs[-1])

    def test_no_server_or_gone_process_is_silent(self):
        for pid in (None, 0, 555):
            m, logs = _mon({24620: 3072.0}, pid=pid)
            self.assertIsNone(m.sample())
            self.assertEqual(logs, [])
            self.assertEqual(m.describe(), "")

    def test_tick_samples_at_most_once_a_minute(self):
        clock = _Clock()
        m, _ = _mon({24620: 3072.0}, clock=clock)
        self.assertIsNotNone(m.tick())
        clock.t += vsm.SAMPLE_EVERY_S - 1
        self.assertIsNone(m.tick())
        clock.t += 2
        self.assertIsNotNone(m.tick())

    def test_only_memory_info_is_ever_called(self):
        m, _ = _mon({24620: 9000.0})
        for _ in range(3):
            m.sample(on_warn=lambda _l: None)
        self.assertTrue(_Proc.calls)
        self.assertEqual({c[0] for c in _Proc.calls}, {"memory_info"})


class WarningTests(unittest.TestCase):
    def test_one_warning_per_pid_over_the_limit(self):
        table = {24620: 6100.0}
        m, logs = _mon(table)
        fired = []
        m.sample(on_warn=fired.append)
        table[24620] = 8000.0
        m.sample(on_warn=fired.append)
        m.sample(on_warn=fired.append)
        warns = [l for l in logs if "WARNING" in l]
        self.assertEqual(len(warns), 1)
        self.assertEqual(fired, warns)
        self.assertIn("pid 24620 rss 6.0 GB is over the 6000 MB limit "
                      "(VOICE_SERVER_RSS_WARN_MB) - report only, it is not "
                      "restarted", warns[0])

    def test_a_new_server_pid_can_warn_again(self):
        table = {1: 7000.0, 2: 7000.0}
        pid = [1]
        logs = []
        m = vsm.VoiceServerMem(pid_fn=lambda: pid[0], warn_mb=lambda: 6000.0,
                               log=logs.append, clock=_Clock(),
                               psutil_mod=_FakePsutil(table))
        m.sample()
        pid[0] = 2
        m.sample()
        self.assertEqual(len([l for l in logs if "WARNING" in l]), 2)

    def test_under_the_limit_never_warns(self):
        m, logs = _mon({24620: 5999.0})
        fired = []
        m.sample(on_warn=fired.append)
        self.assertEqual(fired, [])
        self.assertFalse(any("WARNING" in l for l in logs))

    def test_zero_turns_the_warning_off(self):
        m, logs = _mon({24620: 50000.0}, warn=0.0)
        fired = []
        m.sample(on_warn=fired.append)
        self.assertEqual(fired, [])
        self.assertFalse(any("WARNING" in l for l in logs))

    def test_a_raising_hook_or_log_never_escapes(self):
        def boom(_l):
            raise RuntimeError("x")
        m = vsm.VoiceServerMem(pid_fn=lambda: 1, warn_mb=lambda: 1.0,
                               log=boom, clock=_Clock(),
                               psutil_mod=_FakePsutil({1: 10.0}))
        self.assertEqual(m.sample(on_warn=boom), (1, 10.0))


class SettingTests(unittest.TestCase):
    def test_config_default_is_6000(self):
        import core.config as cfg
        self.assertEqual(cfg.VOICE_SERVER_RSS_WARN_MB, 6000)
        self.assertEqual(vsm.warn_mb_setting(), 6000.0)

    def test_setting_is_read_live_and_never_negative(self):
        with mock.patch("core.config.VOICE_SERVER_RSS_WARN_MB", -5,
                        create=True):
            self.assertEqual(vsm.warn_mb_setting(), 0.0)
        with mock.patch("core.config.VOICE_SERVER_RSS_WARN_MB", 4000,
                        create=True):
            self.assertEqual(vsm.warn_mb_setting(), 4000.0)

    def test_server_pid_comes_from_the_loaded_client_only(self):
        fake = types.SimpleNamespace(
            CLIENT=types.SimpleNamespace(server_pid=lambda: 24620))
        with mock.patch.dict(sys.modules,
                             {"core.clone_voice_client": fake}):
            self.assertEqual(vsm.clone_server_pid(), 24620)
        with mock.patch.dict(sys.modules,
                             {"core.clone_voice_client": None}):
            self.assertIsNone(vsm.clone_server_pid())


class SourceTests(unittest.TestCase):
    def test_module_has_no_restart_or_kill_path(self):
        with open(os.path.join(_PROJECT, "core", "voice_server_mem.py"),
                  encoding="utf-8") as f:
            src = f.read()
        code = re.sub(r'"""[\s\S]*?"""', "", src)
        for word in (".kill(", ".terminate(", ".suspend(", "stop_server",
                     "subprocess", "Popen", "taskkill", "os.kill"):
            self.assertNotIn(word, code, word)


if __name__ == "__main__":
    unittest.main()
