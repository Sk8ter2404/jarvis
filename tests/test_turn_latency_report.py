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

_R1 = ("tail_ms={tail} cap_lag_ms={lag} clip_ms={clip} stt_wait_ms={wait} "
       "stt_engine=- load_ms={load} total_ms={total} play_open_ms={po} "
       "out_lat_ms={ol} filler_clip_ms={fc} eot={eot} st_p=- st_n=- pre=- "
       "cut={cut} amb_deferred=- cache=- ")

# The synthetic 8-line session log. Line 3 is a stand-in transcript that
# must never reach the report; line 6 is a safety line it must count.
SYNTH = (
    "[08:59:59]   [turn-flags] TURN_TAIL_PROBE=True PROCESSING_FILLER_DELAY=0.5\n"
    "[09:00:05]   [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=60 "
    "stt_end=1460 you=1500 llm_post=1560 llm_done=2600 actions_done=2630 "
    "synth_start=2640 first_play=3600 end=6000 prompt_eval_count=14000 "
    "prompt_eval_ms=400 eval_count=20 eval_ms=200 llm_calls=1 "
    "turn_ctx_chars=1200 sys_chars=49000 followup_rounds=0 filler=0 "
    "filler_ms=- tail_ms=1400 cap_lag_ms=120 clip_ms=3200 stt_wait_ms=0 "
    "stt_engine=- load_ms=10 total_ms=650 play_open_ms=40 out_lat_ms=30 "
    "filler_clip_ms=- eot=- st_p=- st_n=- pre=- cut=- amb_deferred=- "
    "cache=- lead_dropped=0\n"
    "[09:00:20]   You:    zebra quartz lantern\n"
    "[09:00:31]   [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=70 "
    "stt_end=1870 you=1900 llm_post=1960 llm_done=3100 actions_done=3140 "
    "synth_start=4600 first_play=5500 end=8000 prompt_eval_count=14100 "
    "prompt_eval_ms=500 eval_count=25 eval_ms=250 llm_calls=1 "
    "turn_ctx_chars=1300 sys_chars=49000 followup_rounds=0 filler=1 "
    "filler_ms=2100 tail_ms=1700 cap_lag_ms=200 clip_ms=4100 "
    "stt_wait_ms=300 stt_engine=- load_ms=12 total_ms=800 play_open_ms=60 "
    "out_lat_ms=50 filler_clip_ms=2200 eot=- st_p=- st_n=- pre=- cut=- "
    "amb_deferred=- cache=- lead_dropped=0\n"
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
    "filler_ms=- tail_ms=- cap_lag_ms=- clip_ms=- stt_wait_ms=- "
    "stt_engine=- load_ms=9 total_ms=640 play_open_ms=35 out_lat_ms=25 "
    "filler_clip_ms=- eot=- st_p=- st_n=- pre=- cut=- amb_deferred=- "
    "cache=- lead_dropped=0\n"
    "[09:04:00]   [turn-timing] kind=inject outcome=shortcut vad_break=- "
    "stt_start=- stt_end=- you=1 llm_post=- llm_done=- actions_done=- "
    "synth_start=- first_play=- end=900 prompt_eval_count=- "
    "prompt_eval_ms=- eval_count=- eval_ms=- llm_calls=0 turn_ctx_chars=- "
    "sys_chars=- followup_rounds=0 filler=0 filler_ms=- lead_dropped=0\n"
)

GOLDEN = """\
JARVIS turn latency report (ms; read-only, numbers only)
logs: <LOGS>  (1 session files, oldest log starts 2026-10-02 08:59:58)
window: start .. end   outcome: ok   excluded: 0 turn(s)
covered: 2026-10-02 09:00:05 .. 2026-10-02 09:04:00  (first .. last turn)
turns: typed/ok=1, typed/shortcut=1, mic/ok=3

== mic turns (kind=voice, outcome=ok)  n=3
                                                    n     p50     p90
  end to end (ms)
  EOS->answer       tail+lag+first_play             2    6260    7172
  EOS->answer       tail_ms+first_play              2    6100    6980
  EOS->answer       1344+first_play                 3    6644    6804
  EOS->stream open  +play_open_ms                   2    6310    7230
  EOS->audible      +play_open_ms+out_lat_ms        2    6350    7278
  EOS->first sound  tail+lag+min(filler,first)      2    4560    5008
  stages (ms)
  tail_ms          end of speech->clip end          2    1550    1670
  cap_lag_ms       clip end->VAD break              2     160     192
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
  play_open_ms     play entry->stream started       2      50      58
  out_lat_ms       reported output latency          2      40      48
  filler_ms        t0->first filler clip            2    2050    2090
  filler_clip_ms   first filler clip                1    2200    2200
  you->first_play                                   3    3600    3600
  first_play       t0->first answer audio           3    5300    5460
  end              t0->end                          3    7000    7800

-- mic split filler: filler  n=2
                                                    n     p50     p90
  end to end (ms)
  EOS->answer       tail+lag+first_play             1    7400    7400
  EOS->answer       tail_ms+first_play              1    7200    7200
  EOS->answer       1344+first_play                 2    6744    6824
  EOS->stream open  +play_open_ms                   1    7460    7460
  EOS->audible      +play_open_ms+out_lat_ms        1    7510    7510
  EOS->first sound  tail+lag+min(filler,first)      1    4000    4000
  stages (ms)
  tail_ms          end of speech->clip end          1    1700    1700
  cap_lag_ms       clip end->VAD break              1     200     200
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
  play_open_ms     play entry->stream started       1      60      60
  out_lat_ms       reported output latency          1      50      50
  filler_ms        t0->first filler clip            2    2050    2090
  filler_clip_ms   first filler clip                1    2200    2200
  you->first_play                                   2    3600    3600
  first_play       t0->first answer audio           2    5400    5480
  end              t0->end                          2    7500    7900

-- mic split filler: no-filler  n=1
                                                    n     p50     p90
  end to end (ms)
  EOS->answer       tail+lag+first_play             1    5120    5120
  EOS->answer       tail_ms+first_play              1    5000    5000
  EOS->answer       1344+first_play                 1    4944    4944
  EOS->stream open  +play_open_ms                   1    5160    5160
  EOS->audible      +play_open_ms+out_lat_ms        1    5190    5190
  EOS->first sound  tail+lag+min(filler,first)      1    5120    5120
  stages (ms)
  tail_ms          end of speech->clip end          1    1400    1400
  cap_lag_ms       clip end->VAD break              1     120     120
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
  play_open_ms     play entry->stream started       1      40      40
  out_lat_ms       reported output latency          1      30      30
  you->first_play                                   1    2100    2100
  first_play       t0->first answer audio           1    3600    3600
  end              t0->end                          1    6000    6000

== typed turns (kind=inject, outcome=ok)  n=1
                                                    n     p50     p90
  end to end (ms)
  drain->answer     first_play                      1    2300    2300
  drain->stream open first_play+play_open_ms        1    2335    2335
  drain->audible    +out_lat_ms                     1    2360    2360
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
  play_open_ms     play entry->stream started       1      35      35
  out_lat_ms       reported output latency          1      25      25
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
           outcome="ok", vad="0", lag="-", po="-", ol="-"):
    return (f"[{ts}]   [turn-timing] kind=voice outcome={outcome} "
            f"vad_break={vad} stt_start=60 stt_end=1460 you=1500 "
            f"llm_post=1560 llm_done=2600 actions_done=2630 synth_start=2640 "
            f"first_play={first_play} end=6000 prompt_eval_count=1 "
            f"prompt_eval_ms=400 eval_count=2 eval_ms=200 llm_calls=1 "
            f"turn_ctx_chars=1 sys_chars=1 followup_rounds=0 filler={filler} "
            f"filler_ms={'2000' if filler else '-'} "
            + _R1.format(tail=tail, lag=lag, clip="-", wait="-", load="-",
                         total="-", po=po, ol=ol, fc="-", eot=eot, cut=cut)
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

    def test_keep_lines_gives_the_last_turns_their_own_lines(self):
        """The dashboard timeline's view (2026-10-02): each of the last N
        turns carries the stamped lines since the previous [turn-timing]
        line; older turns carry none, and the report never asks for them."""
        self.write("session_2026-10-02_09-00-00.log",
                   "[09:00:01]   You:    zebra one\n" + _voice("09:00:05")
                   + "[09:00:06]   You:    zebra two\n"
                   + "unstamped continuation\n"
                   + "[09:00:07]   [action] get_time: nine\n"
                   + _voice("09:00:09")
                   + "[09:00:10]   [turn-flags] X=1\n"
                   + "[09:00:11]   You:    zebra three\n"
                   + _voice("09:00:15"))
        path = os.path.join(self.dir, "session_2026-10-02_09-00-00.log")
        turns, _ = rep.parse_log(path, keep_lines=2)
        self.assertNotIn("lines", turns[0])
        self.assertEqual(turns[1]["lines"], [
            "[09:00:06]   You:    zebra two",
            "[09:00:07]   [action] get_time: nine"])
        self.assertEqual(turns[2]["lines"], [
            "[09:00:10]   [turn-flags] X=1",
            "[09:00:11]   You:    zebra three"])
        self.assertEqual(turns[2]["flags"], {"X": "1"})
        plain, _ = rep.parse_log(path)
        self.assertFalse(any("lines" in t for t in plain))
        self.assertEqual([t["kv"] for t in plain], [t["kv"] for t in turns])

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


def _row(out, label):
    """(n, p50, p90) of the first report row starting with `label`; the
    dashes of an empty row come back as None."""
    for ln in out.splitlines():
        if ln.strip().startswith(label):
            n, p50, p90 = ln.split()[-3:]
            return (int(n), None if p50 == "-" else int(p50),
                    None if p90 == "-" else int(p90))
    raise AssertionError(f"no row {label!r} in:\n{out}")


class R1ReviewTests(_LogDir):
    """The R1 review's report fixes (2026-10-01)."""

    def test_eos_rows_take_only_turns_that_ended_on_a_vad_break(self):
        # A capture cut at MAX_RECORDING_SECS has no end of speech: its t0
        # is the stream close (vad_break=-) and its tail ~0 (still talking).
        self.write("session_2026-10-02_09-00-00.log",
                   _voice("09:00:05", first_play=3000, tail="1400",
                          lag="100", po="40", ol="30")
                   + _voice("09:01:05", first_play=2000, tail="0", lag="-",
                            po="40", ol="30", vad="-"))
        out = self.report(kinds=("voice",))
        self.assertEqual(_row(out, "EOS->answer       tail_ms+first_play"),
                         (1, 4400, 4400))
        self.assertEqual(_row(out, "EOS->answer       1344+first_play"),
                         (1, 4344, 4344))
        self.assertEqual(_row(out, "tail_ms"), (1, 1400, 1400))
        self.assertEqual(_row(out, "EOS->answer       tail+lag+first_play"),
                         (1, 4500, 4500))
        self.assertEqual(_row(out, "EOS->stream open"), (1, 4540, 4540))
        self.assertEqual(_row(out, "EOS->audible"), (1, 4570, 4570))
        self.assertEqual(_row(out, "EOS->first sound"), (1, 4500, 4500))
        # ...while the turn itself still counts everywhere else.
        self.assertEqual(_row(out, "first_play")[0], 2)
        self.assertIn("n=2", out)

    def test_audible_adds_the_reported_output_latency(self):
        self.write("session_2026-10-02_09-00-00.log",
                   _voice("09:00:05", first_play=3000, tail="1400",
                          lag="100", po="40", ol="30")
                   + _voice("09:01:05", first_play=3000, tail="1400",
                            lag="100", po="40"))
        out = self.report(kinds=("voice",))
        self.assertEqual(_row(out, "EOS->stream open"), (2, 4540, 4540))
        self.assertEqual(_row(out, "EOS->audible"), (1, 4570, 4570))
        self.assertEqual(_row(out, "out_lat_ms"), (1, 30, 30))
        self.assertEqual(_row(out, "cap_lag_ms"), (2, 100, 100))

    def test_the_report_prints_the_span_it_covers(self):
        self.write("session_2026-10-02_09-00-00.log",
                   _voice("09:00:05") + _voice("09:30:00"))
        self.write("session_2026-10-03_08-00-00.log", _voice("08:10:00"))
        out = self.report()
        self.assertIn("covered: 2026-10-02 09:00:05 .. 2026-10-03 08:10:00",
                      out)
        self.assertIn("oldest log starts 2026-10-02 09:00:00", out)
        self.assertNotIn("WARNING", out)
        out = self.report(since=datetime.datetime(2026, 10, 2, 9, 10))
        self.assertIn("covered: 2026-10-02 09:30:00 .. 2026-10-03 08:10:00",
                      out)
        self.assertNotIn("WARNING", out)

    def test_a_window_older_than_the_logs_is_flagged(self):
        # JARVIS keeps only its newest LOG_KEEP_COUNT logs: a week-long
        # --since against ~3 days of files must say so, not report 3 days
        # as a week.
        self.write("session_2026-10-02_09-00-00.log", _voice("09:00:05"))
        _, out = self.run_main("--since", "2026-09-25")
        warn = [ln for ln in out.splitlines() if ln.startswith("WARNING")]
        self.assertEqual(len(warn), 1, out)
        self.assertIn("2026-10-02 09:00:00", warn[0])
        self.assertIn("2026-09-25 00:00:00", warn[0])
        self.assertIn(f"newest {rep.LOG_KEEP_COUNT}", warn[0])
        self.assertIn("--logs", warn[0])

    def test_no_turns_in_the_window(self):
        self.write("session_2026-10-02_09-00-00.log", _voice("09:00:05"))
        out = self.report(since=datetime.datetime(2026, 10, 5))
        self.assertIn("covered: no turns", out)

    def test_log_keep_count_mirrors_the_monolith(self):
        import re
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "bobert_companion.py")
        if not os.path.isfile(path):
            self.skipTest("monolith not in this checkout")
        with open(path, encoding="utf-8") as fh:
            m = re.search(r"^LOG_KEEP_COUNT\s*=\s*(\d+)", fh.read(), re.M)
        self.assertIsNotNone(m)
        self.assertEqual(rep.LOG_KEEP_COUNT, int(m.group(1)))


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
