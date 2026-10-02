"""tools/turn_latency_report.py — p50/p90 per stage from the [turn-timing]
lines (speed plan R1, 2026-10-01).

Every log here is SYNTHETIC (made-up offsets, no real transcript). The golden
test pins the whole report for an 8-line session log; the rest pin the
parsing, windowing, splits and the two hard rules: the tool never writes
anything and never prints log text.

Run: python tools/run_tests.py test_turn_latency_report
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import datetime
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tools import turn_latency_report as rep

_R1 = ("tail_ms={tail} clip_ms={clip} stt_wait_ms={wait} stt_engine=- "
       "load_ms={load} total_ms={total} play_open_ms={po} "
       "filler_clip_ms={fc} eot={eot} st_p=- st_n=- pre=- cut={cut} "
       "amb_deferred=- cache=- ")

# The synthetic 8-line session log. Line 3 is a stand-in transcript that
# must never reach the report; line 6 is a safety line it must count.
SYNTH = (
    "[08:59:59]   [turn-flags] TURN_TAIL_PROBE=True PROCESSING_FILLER_DELAY=0.5\n"
    "[09:00:05]   [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=60 "
    "stt_end=1460 you=1500 llm_post=1560 llm_done=2600 actions_done=2630 "
    "synth_start=2640 first_play=3600 end=6000 prompt_eval_count=14000 "
    "prompt_eval_ms=400 eval_count=20 eval_ms=200 llm_calls=1 "
    "turn_ctx_chars=1200 sys_chars=49000 followup_rounds=0 filler=0 "
    "filler_ms=- tail_ms=1400 clip_ms=3200 stt_wait_ms=0 stt_engine=- "
    "load_ms=10 total_ms=650 play_open_ms=40 filler_clip_ms=- eot=- st_p=- "
    "st_n=- pre=- cut=- amb_deferred=- cache=- lead_dropped=0\n"
    "[09:00:20]   You:    zebra quartz lantern\n"
    "[09:00:31]   [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=70 "
    "stt_end=1870 you=1900 llm_post=1960 llm_done=3100 actions_done=3140 "
    "synth_start=4600 first_play=5500 end=8000 prompt_eval_count=14100 "
    "prompt_eval_ms=500 eval_count=25 eval_ms=250 llm_calls=1 "
    "turn_ctx_chars=1300 sys_chars=49000 followup_rounds=0 filler=1 "
    "filler_ms=2100 tail_ms=1700 clip_ms=4100 stt_wait_ms=300 stt_engine=- "
    "load_ms=12 total_ms=800 play_open_ms=60 filler_clip_ms=2200 eot=- "
    "st_p=- st_n=- pre=- cut=- amb_deferred=- cache=- lead_dropped=0\n"
    "[09:01:10]   [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=65 "
    "stt_end=1665 you=1700 llm_post=1750 llm_done=2850 actions_done=2880 "
    "synth_start=4300 first_play=5300 end=7000 prompt_eval_count=13900 "
    "prompt_eval_ms=450 eval_count=22 eval_ms=220 llm_calls=1 "
    "turn_ctx_chars=1250 sys_chars=49000 followup_rounds=0 filler=1 "
    "filler_ms=2000 lead_dropped=0\n"
    "[09:02:00]   [speak] playback failed: PortAudioError: synthetic\n"
    "[09:03:00]   [turn-timing] kind=inject outcome=ok vad_break=- "
    "stt_start=- stt_end=- you=2 llm_post=60 llm_done=1260 actions_done=1290 "
    "synth_start=1295 first_play=2300 end=4000 prompt_eval_count=14000 "
    "prompt_eval_ms=420 eval_count=18 eval_ms=180 llm_calls=1 "
    "turn_ctx_chars=1100 sys_chars=49000 followup_rounds=0 filler=0 "
    "filler_ms=- tail_ms=- clip_ms=- stt_wait_ms=- stt_engine=- load_ms=9 "
    "total_ms=640 play_open_ms=35 filler_clip_ms=- eot=- st_p=- st_n=- "
    "pre=- cut=- amb_deferred=- cache=- lead_dropped=0\n"
    "[09:04:00]   [turn-timing] kind=inject outcome=shortcut vad_break=- "
    "stt_start=- stt_end=- you=1 llm_post=- llm_done=- actions_done=- "
    "synth_start=- first_play=- end=900 prompt_eval_count=- "
    "prompt_eval_ms=- eval_count=- eval_ms=- llm_calls=0 turn_ctx_chars=- "
    "sys_chars=- followup_rounds=0 filler=0 filler_ms=- lead_dropped=0\n"
)

GOLDEN = """\
JARVIS turn latency report (ms; read-only, numbers only)
logs: <LOGS>  (1 session files)
window: start .. end   outcome: ok   excluded: 0 turn(s)
turns: typed/ok=1, typed/shortcut=1, mic/ok=3

== mic turns (kind=voice, outcome=ok)  n=3
                                                    n     p50     p90
  end to end (ms)
  EOS->answer       tail_ms+first_play              2    6100    6980
  EOS->answer       1344+first_play                 3    6644    6804
  EOS->audible      +play_open_ms                   2    6150    7038
  EOS->first sound  tail_ms+min(filler,first)       2    4400    4880
  stages (ms)
  tail_ms          end of speech->VAD break         2    1550    1670
  clip_ms          captured clip                    2    3650    4010
  pre_stt          vad_break->stt_start             3      65      69
  stt_wait_ms      wait for _stt_lock               2     150     270
  stt              stt_start->stt_end               3    1600    1760
  gate             stt_end->you                     3      35      39
  prep             you->llm_post                    3      60      60
  llm              llm_post->llm_done               3    1100    1132
  prompt_eval_ms                                    3     450     490
  eval_ms                                           3     220     244
  load_ms                                           2      11      12
  total_ms         Ollama total                     2     725     785
  llm_overhead     llm-prompt_eval-eval             3     430     438
  llm_outside      llm-total_ms                     2     365     385
  post_llm         llm_done->actions_done           3      30      38
  speak_wait       actions_done->synth_start        3    1420    1452
  synth            synth_start->first_play          3     960     992
  play_open_ms     duck->stream open                2      50      58
  filler_ms        t0->first filler clip            2    2050    2090
  filler_clip_ms   first filler clip                1    2200    2200
  you->first_play                                   3    3600    3600
  first_play       t0->first answer audio           3    5300    5460
  end              t0->end                          3    7000    7800

-- mic split filler: filler  n=2
                                                    n     p50     p90
  end to end (ms)
  EOS->answer       tail_ms+first_play              1    7200    7200
  EOS->answer       1344+first_play                 2    6744    6824
  EOS->audible      +play_open_ms                   1    7260    7260
  EOS->first sound  tail_ms+min(filler,first)       1    3800    3800
  stages (ms)
  tail_ms          end of speech->VAD break         1    1700    1700
  clip_ms          captured clip                    1    4100    4100
  pre_stt          vad_break->stt_start             2      68      70
  stt_wait_ms      wait for _stt_lock               1     300     300
  stt              stt_start->stt_end               2    1700    1780
  gate             stt_end->you                     2      32      34
  prep             you->llm_post                    2      55      59
  llm              llm_post->llm_done               2    1120    1136
  prompt_eval_ms                                    2     475     495
  eval_ms                                           2     235     247
  load_ms                                           1      12      12
  total_ms         Ollama total                     1     800     800
  llm_overhead     llm-prompt_eval-eval             2     410     426
  llm_outside      llm-total_ms                     1     340     340
  post_llm         llm_done->actions_done           2      35      39
  speak_wait       actions_done->synth_start        2    1440    1456
  synth            synth_start->first_play          2     950     990
  play_open_ms     duck->stream open                1      60      60
  filler_ms        t0->first filler clip            2    2050    2090
  filler_clip_ms   first filler clip                1    2200    2200
  you->first_play                                   2    3600    3600
  first_play       t0->first answer audio           2    5400    5480
  end              t0->end                          2    7500    7900

-- mic split filler: no-filler  n=1
                                                    n     p50     p90
  end to end (ms)
  EOS->answer       tail_ms+first_play              1    5000    5000
  EOS->answer       1344+first_play                 1    4944    4944
  EOS->audible      +play_open_ms                   1    5040    5040
  EOS->first sound  tail_ms+min(filler,first)       1    5000    5000
  stages (ms)
  tail_ms          end of speech->VAD break         1    1400    1400
  clip_ms          captured clip                    1    3200    3200
  pre_stt          vad_break->stt_start             1      60      60
  stt_wait_ms      wait for _stt_lock               1       0       0
  stt              stt_start->stt_end               1    1400    1400
  gate             stt_end->you                     1      40      40
  prep             you->llm_post                    1      60      60
  llm              llm_post->llm_done               1    1040    1040
  prompt_eval_ms                                    1     400     400
  eval_ms                                           1     200     200
  load_ms                                           1      10      10
  total_ms         Ollama total                     1     650     650
  llm_overhead     llm-prompt_eval-eval             1     440     440
  llm_outside      llm-total_ms                     1     390     390
  post_llm         llm_done->actions_done           1      30      30
  speak_wait       actions_done->synth_start        1      10      10
  synth            synth_start->first_play          1     960     960
  play_open_ms     duck->stream open                1      40      40
  you->first_play                                   1    2100    2100
  first_play       t0->first answer audio           1    3600    3600
  end              t0->end                          1    6000    6000

== typed turns (kind=inject, outcome=ok)  n=1
                                                    n     p50     p90
  end to end (ms)
  drain->answer     first_play                      1    2300    2300
  drain->audible    first_play+play_open_ms         1    2335    2335
  stages (ms)
  prep             you->llm_post                    1      58      58
  llm              llm_post->llm_done               1    1200    1200
  prompt_eval_ms                                    1     420     420
  eval_ms                                           1     180     180
  load_ms                                           1       9       9
  total_ms         Ollama total                     1     640     640
  llm_overhead     llm-prompt_eval-eval             1     600     600
  llm_outside      llm-total_ms                     1     560     560
  post_llm         llm_done->actions_done           1      30      30
  speak_wait       actions_done->synth_start        1       5       5
  synth            synth_start->first_play          1    1005    1005
  play_open_ms     duck->stream open                1      35      35
  you->first_play                                   1    2298    2298
  first_play       t0->first answer audio           1    2300    2300
  end              t0->end                          1    4000    4000

== safety counters (lines in the window)
  [speak] playback failed                            1
  tts-reaper wedged                                  0
  [filler] clip still playing                        0
  wake-word mode refusals                            0   (0.00 per mic turn)
  [eot-shadow] resumed=1                             0
  [stt-rescue]                                       0
  kokoro render failed                               0
"""


def _voice(ts, first_play=3000, filler=0, tail="-", cut="-", eot="-",
           outcome="ok"):
    return (f"[{ts}]   [turn-timing] kind=voice outcome={outcome} "
            f"vad_break=0 stt_start=60 stt_end=1460 you=1500 llm_post=1560 "
            f"llm_done=2600 actions_done=2630 synth_start=2640 "
            f"first_play={first_play} end=6000 prompt_eval_count=1 "
            f"prompt_eval_ms=400 eval_count=2 eval_ms=200 llm_calls=1 "
            f"turn_ctx_chars=1 sys_chars=1 followup_rounds=0 filler={filler} "
            f"filler_ms={'2000' if filler else '-'} "
            + _R1.format(tail=tail, clip="-", wait="-", load="-", total="-",
                         po="-", fc="-", eot=eot, cut=cut)
            + "lead_dropped=0\n")


class _LogDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jarvis_tlr_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, text):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)

    def report(self, **kw):
        return rep.build_report(self.dir, **kw)

    def run_main(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = rep.main(["--logs", self.dir, *args])
        return code, buf.getvalue()


class GoldenTests(_LogDir):
    def test_golden_report_for_the_synthetic_8_line_log(self):
        self.assertEqual(len(SYNTH.splitlines()), 8)
        self.write("session_2026-10-02_08-59-58.log", SYNTH)
        code, out = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(out.replace(self.dir, "<LOGS>"), GOLDEN)

    def test_the_report_never_prints_log_text(self):
        self.write("session_2026-10-02_08-59-58.log", SYNTH)
        out = self.report()
        for words in ("zebra", "quartz", "lantern", "PortAudioError",
                      "synthetic", "You:"):
            self.assertNotIn(words, out)

    def test_the_tool_never_writes(self):
        self.write("session_2026-10-02_08-59-58.log", SYNTH)
        real_open = builtins.open
        modes = []

        def guarded(file, mode="r", *a, **k):
            modes.append(mode)
            if any(c in mode for c in "wax+"):
                raise AssertionError(f"report opened {file!r} for {mode!r}")
            return real_open(file, mode, *a, **k)

        before = sorted(os.listdir(self.dir))
        with mock.patch("builtins.open", guarded), \
                mock.patch("os.makedirs", side_effect=AssertionError("mkdir")):
            code, _ = self.run_main("--split", "day")
        self.assertEqual(code, 0)
        self.assertEqual(modes, ["rb"])
        self.assertEqual(sorted(os.listdir(self.dir)), before)


class ParseTests(_LogDir):
    def test_midnight_rollover(self):
        self.write("session_2026-10-02_23-50-00.log",
                   _voice("23:58:00") + "[00:00:01]   noise\n"
                   + _voice("00:03:00"))
        turns, _ = rep.parse_log(os.path.join(
            self.dir, "session_2026-10-02_23-50-00.log"))
        self.assertEqual([t["ts"] for t in turns], [
            datetime.datetime(2026, 10, 2, 23, 58, 0),
            datetime.datetime(2026, 10, 3, 0, 3, 0)])

    def test_flags_are_per_session_and_seen_by_later_turns(self):
        self.write("session_2026-10-02_09-00-00.log",
                   _voice("09:00:05")
                   + "[09:00:06]   [turn-flags] AMBIENT_STT_YIELD=True\n"
                   + _voice("09:00:09"))
        self.write("session_2026-10-02_10-00-00.log", _voice("10:00:05"))
        _, turns, _ = rep.load(self.dir)
        self.assertEqual([t["flags"].get("AMBIENT_STT_YIELD", "?")
                          for t in turns], ["?", "True", "?"])

    def test_files_that_are_not_session_logs_are_ignored(self):
        self.write("tray.log", _voice("09:00:05"))
        self.write("session_bad-name.log", _voice("09:00:05"))
        paths, turns, _ = rep.load(self.dir)
        self.assertEqual(turns, [])

    def test_lines_without_a_timestamp_or_kind_are_skipped(self):
        self.write("session_2026-10-02_09-00-00.log",
                   _voice("09:00:05").split("]", 1)[1]
                   + "[09:00:06]   [turn-timing] outcome=ok\n")
        _, turns, _ = rep.load(self.dir)
        self.assertEqual(turns, [])

    def test_pct_is_the_baselines_linear_interpolation(self):
        self.assertEqual(rep.pct([1, 2, 3, 4], 50), 2)     # 2.5 rounds even
        self.assertEqual(rep.pct([10, 20], 90), 19)
        self.assertEqual(rep.pct([7], 90), 7)
        self.assertIsNone(rep.pct([None, None], 50))


class SelectionTests(_LogDir):
    def setUp(self):
        super().setUp()
        self.write("session_2026-10-01_15-00-00.log",
                   _voice("15:16:46") + _voice("15:16:47", first_play=33673)
                   + _voice("15:20:00"))
        self.write("session_2026-10-02_09-00-00.log", _voice("09:00:05"))

    def _n(self, out):
        line = [ln for ln in out.splitlines() if ln.startswith("== mic")][0]
        return int(line.rsplit("n=", 1)[1])

    def test_the_benchmark_turn_is_excluded_by_default(self):
        out = self.report()
        self.assertIn("excluded: 1 turn(s)", out)
        self.assertEqual(self._n(out), 3)
        code, out = self.run_main("--no-default-exclude")
        self.assertEqual(self._n(out), 4)
        code, out = self.run_main("--exclude", "2026-10-02 09:00:05")
        self.assertEqual(self._n(out), 2)

    def test_since_and_until(self):
        _, out = self.run_main("--since", "2026-10-02")
        self.assertEqual(self._n(out), 1)
        _, out = self.run_main("--until", "2026-10-01")   # the whole day
        self.assertEqual(self._n(out), 2)
        _, out = self.run_main("--since", "2026-10-01 15:17",
                               "--until", "2026-10-01T15:21:00")
        self.assertEqual(self._n(out), 1)

    def test_kind_restricts_the_sections(self):
        _, out = self.run_main("--kind", "voice")
        self.assertIn("== mic turns", out)
        self.assertNotIn("== typed turns", out)
        _, out = self.run_main("--kind", "inject")
        self.assertNotIn("== mic turns", out)

    def test_outcome_all_keeps_errors(self):
        self.write("session_2026-10-03_09-00-00.log",
                   _voice("09:00:05", outcome="error"))
        self.assertEqual(self._n(self.report()), 3)
        self.assertEqual(self._n(self.report(outcome="all")), 4)

    def test_bad_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                rep.main(["--logs", self.dir, "--split", "bogus"])
            with self.assertRaises(SystemExit):
                rep.main(["--logs", self.dir, "--since", "yesterday"])
            self.assertEqual(rep.main(["--logs", os.path.join(
                self.dir, "missing")]), 2)


class SplitTests(_LogDir):
    def _groups(self, out, label):
        return [ln.split(f"split {label}: ", 1)[1]
                for ln in out.splitlines() if f"split {label}: " in ln]

    def test_flag_split_reads_the_boot_line(self):
        self.write("session_2026-10-02_09-00-00.log",
                   "[09:00:00]   [turn-flags] PROCESSING_FILLER_DELAY=0.5\n"
                   + _voice("09:00:05") + _voice("09:01:05"))
        self.write("session_2026-10-02_10-00-00.log",
                   "[10:00:00]   [turn-flags] PROCESSING_FILLER_DELAY=1.0\n"
                   + _voice("10:00:05"))
        self.write("session_2026-09-30_10-00-00.log", _voice("10:00:05"))
        out = self.report(kinds=("voice",),
                          split="flag=PROCESSING_FILLER_DELAY")
        self.assertEqual(self._groups(out, "flag=PROCESSING_FILLER_DELAY"),
                         ["0.5  n=2", "1.0  n=1", "?  n=1"])

    def test_cut_splits_by_presence_and_eot_by_value(self):
        self.write("session_2026-10-02_09-00-00.log",
                   _voice("09:00:05", cut="420", eot="st")
                   + _voice("09:01:05", cut="380", eot="rms")
                   + _voice("09:02:05"))
        out = self.report(kinds=("voice",), split="cut")
        self.assertEqual(self._groups(out, "cut"), ["cut  n=2", "no-cut  n=1"])
        out = self.report(kinds=("voice",), split="eot")
        self.assertEqual(self._groups(out, "eot"),
                         ["rms  n=1", "st  n=1", "-  n=1"])

    def test_day_split(self):
        self.write("session_2026-10-02_23-50-00.log",
                   _voice("23:58:00") + _voice("00:03:00"))
        out = self.report(kinds=("voice",), split="day")
        self.assertEqual(self._groups(out, "day"),
                         ["2026-10-02  n=1", "2026-10-03  n=1"])

    def test_old_lines_without_r1_fields_still_report(self):
        self.write("session_2026-09-29_09-00-00.log",
                   "[09:00:05]   [turn-timing] kind=voice outcome=ok "
                   "vad_break=0 stt_start=60 stt_end=1460 you=1500 "
                   "llm_post=1560 llm_done=2600 actions_done=2630 "
                   "synth_start=2640 first_play=5000 end=6000 "
                   "prompt_eval_count=1 prompt_eval_ms=400 eval_count=2 "
                   "eval_ms=200 llm_calls=1 turn_ctx_chars=1 sys_chars=1 "
                   "followup_rounds=0 filler=0 filler_ms=- lead_dropped=0\n")
        out = self.report(kinds=("voice",))
        self.assertIn("EOS->answer       1344+first_play                 1"
                      "    6344    6344", out)
        self.assertIn("EOS->answer       tail_ms+first_play              0"
                      "       -       -", out)

    def test_safety_counters_are_windowed(self):
        self.write("session_2026-10-02_09-00-00.log",
                   _voice("09:00:05")
                   + "[09:00:06]   [bg-audio] wake-word mode — ignoring "
                     "non-wake utterance: 'synthetic'\n"
                   + "[09:00:07]   [audio] tts-reaper wedged in native code\n"
                   + "[09:00:08]   [tts] kokoro render failed (X: y)\n"
                   + "[09:00:09]   [kokoro] render failed (X: y)\n"
                   + "[09:00:10]   [eot-shadow] fire_ms=300 resumed=1\n"
                   + "[09:00:11]   [filler] clip still playing after 3s\n")
        out = self.report()
        self.assertRegex(out, r"wake-word mode refusals\s+1   \(1\.00 per mic")
        self.assertRegex(out, r"tts-reaper wedged\s+1\n")
        self.assertRegex(out, r"kokoro render failed\s+2\n")
        self.assertRegex(out, r"\[eot-shadow\] resumed=1\s+1\n")
        self.assertRegex(out, r"\[filler\] clip still playing\s+1\n")
        self.assertNotIn("synthetic", out)
        out = self.report(since=datetime.datetime(2026, 10, 2, 9, 0, 8))
        self.assertRegex(out, r"wake-word mode refusals\s+0")
        self.assertRegex(out, r"kokoro render failed\s+2\n")


class ModuleHygieneTests(unittest.TestCase):
    def test_stdlib_only(self):
        with open(rep.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        self.assertLessEqual(mods, {"__future__", "argparse", "datetime",
                                    "glob", "os", "re", "sys"})

    def test_default_log_folder_and_exclusion(self):
        self.assertEqual(rep.DEFAULT_LOG_DIR, r"C:\JARVIS\logs")
        self.assertEqual(rep.DEFAULT_EXCLUDE, ("2026-10-01 15:16:47",))
        self.assertEqual(rep.ASSUMED_TAIL_MS, 21 * 64)


if __name__ == "__main__":
    unittest.main()
