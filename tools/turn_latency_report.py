#!/usr/bin/env python3
"""Turn latency report from the [turn-timing] lines in JARVIS session logs.

Speed plan R1 (2026-10-01). Every voice / typed turn prints ONE
``[turn-timing]`` line (core/turn_timing.py); this tool reads them back and
prints p50 / p90 per stage plus the end-to-end metrics the speed plan judges
every later change by:

  EOS -> answer       tail_ms + first_play   (mic turns; measured end of speech)
  EOS -> answer       1344 + first_play      (the pre-R1 assumption, for the
                                              old baseline and for lines that
                                              carry no tail_ms)
  EOS -> audible      EOS -> answer + play_open_ms
  EOS -> first sound  tail_ms + min(filler_ms, first_play)

Mic turns (kind=voice) and typed / injected turns (kind=inject) are reported
separately, mic turns also split filler / no filler (or by --split).

    python tools/turn_latency_report.py
    python tools/turn_latency_report.py --since 2026-10-02 --kind voice
    python tools/turn_latency_report.py --split flag=PROCESSING_FILLER_DELAY
    python tools/turn_latency_report.py --split day --outcome all

READ-ONLY, NUMBERS ONLY. The logs are opened for reading and nothing is ever
written: the report goes to stdout. It parses only the [turn-timing] and
[turn-flags] lines and COUNTS the safety lines, so no transcript, reply or
other log text is ever printed (the 2026-09-06 PII near-miss: benchmark output
never goes into the repo). Stdlib only.

--split flag=KEY groups sessions by the value on their boot
"[turn-flags] KEY=value ..." line (the monolith's _log_turn_flags); a session
without one (older logs, or a key it did not print) groups as "?".

The 2026-10-01 15:16:47 turn is excluded by default (attributed to Plan A's
own live benchmark, unverified); --no-default-exclude keeps it.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import re
import sys

DEFAULT_LOG_DIR = r"C:\JARVIS\logs"
ASSUMED_TAIL_MS = 1344          # 21 chunks x 64 ms: the pre-R1 silence wait
DEFAULT_EXCLUDE = ("2026-10-01 15:16:47",)
SPLITS = ("filler", "eot", "stt_engine", "pre", "cut", "cache", "day",
          "flag=<KEY>")
KIND_LABEL = {"voice": "mic", "inject": "typed", "realtime": "realtime"}

_TS = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]")
_NAME = re.compile(r"session_(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)-(\d\d)\.log$")

# (label, test) — counted, never printed. "[stt-rescue]" is the tag R6's
# Whisper rescue prints; "[eot-shadow] ... resumed=1" is R7's would-be cut.
SAFETY = (
    ("[speak] playback failed", lambda s: "[speak] playback failed" in s),
    ("tts-reaper wedged", lambda s: "tts-reaper wedged" in s),
    ("[filler] clip still playing", lambda s: "[filler] clip still playing" in s),
    ("wake-word mode refusals",
     lambda s: "wake-word mode" in s and "ignoring" in s),
    ("[eot-shadow] resumed=1",
     lambda s: "[eot-shadow]" in s and "resumed=1" in s),
    ("[stt-rescue]", lambda s: "[stt-rescue]" in s),
    ("kokoro render failed",
     lambda s: "render failed" in s and "kokoro" in s),
)


# ── parsing ───────────────────────────────────────────────────────────────
def parse_kv(text: str) -> dict:
    """key=value tokens after a [tag] -> {key: str}."""
    out = {}
    for tok in text.split():
        k, sep, v = tok.partition("=")
        if sep and k:
            out[k] = v
    return out


def session_start(path: str) -> "datetime.datetime | None":
    m = _NAME.search(os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.datetime.strptime(
            f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}",
            "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_log(path: str):
    """One session log -> (turns, events). A turn is {"ts", "session",
    "kv", "flags"}; an event is (ts, safety label). The wall clock comes from
    the "[HH:MM:SS]" prefix plus the file name's date, rolling over midnight
    (a jump back of more than 6 h is the next day)."""
    start = session_start(path)
    if start is None:
        return [], []
    with open(path, "rb") as fh:
        raw = fh.read()
    day = start.date()
    last = None
    flags: dict = {}
    turns, events = [], []
    base = os.path.basename(path)
    for line in raw.decode("utf-8", errors="replace").splitlines():
        m = _TS.match(line)
        if not m:
            continue
        try:
            t = datetime.time(int(m.group(1)), int(m.group(2)),
                              int(m.group(3)))
        except ValueError:
            continue
        cur = datetime.datetime.combine(day, t)
        if last is not None and cur < last - datetime.timedelta(hours=6):
            day += datetime.timedelta(days=1)
            cur = datetime.datetime.combine(day, t)
        last = cur
        i = line.find("[turn-timing]")
        if i >= 0:
            kv = parse_kv(line[i + len("[turn-timing]"):])
            if "kind" in kv:
                turns.append({"ts": cur, "session": base, "kv": kv,
                              "flags": dict(flags)})
            continue
        i = line.find("[turn-flags]")
        if i >= 0:
            flags = parse_kv(line[i + len("[turn-flags]"):])
            continue
        for label, test in SAFETY:
            if test(line):
                events.append((cur, label))
                break
    return turns, events


def load(log_dir: str):
    paths = sorted(glob.glob(os.path.join(log_dir, "session_*.log")))
    turns, events = [], []
    for p in paths:
        try:
            t, e = parse_log(p)
        except OSError as exc:
            print(f"  (skipped {os.path.basename(p)}: {exc})", file=sys.stderr)
            continue
        turns.extend(t)
        events.extend(e)
    return paths, turns, events


# ── metrics ───────────────────────────────────────────────────────────────
def _i(kv: dict, key: str) -> "int | None":
    try:
        return int(kv.get(key, "-"))
    except (TypeError, ValueError):
        return None


def _d(kv, a, b):
    x, y = _i(kv, a), _i(kv, b)
    return None if x is None or y is None else y - x


def _sub(x, *ys):
    if x is None or any(y is None for y in ys):
        return None
    return x - sum(ys)


def _nonneg(v):
    return v if v is not None and v >= 0 else None


def _eos_answer(kv):
    tail, fp = _i(kv, "tail_ms"), _i(kv, "first_play")
    return None if tail is None or fp is None else tail + fp


def _eos_first_sound(kv):
    tail, fp, fm = _i(kv, "tail_ms"), _i(kv, "first_play"), _i(kv, "filler_ms")
    if tail is None:
        return None
    firsts = [x for x in (fp, fm) if x is not None]
    return tail + min(firsts) if firsts else None


# (label, fn(kv) -> int | None). Mic turns.
E2E_MIC = (
    ("EOS->answer       tail_ms+first_play", _eos_answer),
    ("EOS->answer       1344+first_play",
     lambda kv: (None if _i(kv, "first_play") is None
                 else ASSUMED_TAIL_MS + _i(kv, "first_play"))),
    ("EOS->audible      +play_open_ms",
     lambda kv: (None if _eos_answer(kv) is None
                 or _i(kv, "play_open_ms") is None
                 else _eos_answer(kv) + _i(kv, "play_open_ms"))),
    ("EOS->first sound  tail_ms+min(filler,first)", _eos_first_sound),
)
# Typed turns have no capture: their clock starts at the inject drain.
E2E_TYPED = (
    ("drain->answer     first_play", lambda kv: _i(kv, "first_play")),
    ("drain->audible    first_play+play_open_ms",
     lambda kv: (None if _i(kv, "first_play") is None
                 or _i(kv, "play_open_ms") is None
                 else _i(kv, "first_play") + _i(kv, "play_open_ms"))),
)

STAGES = (
    ("tail_ms          end of speech->VAD break", lambda kv: _i(kv, "tail_ms")),
    ("clip_ms          captured clip", lambda kv: _i(kv, "clip_ms")),
    ("pre_stt          vad_break->stt_start",
     lambda kv: _d(kv, "vad_break", "stt_start")),
    ("stt_wait_ms      wait for _stt_lock", lambda kv: _i(kv, "stt_wait_ms")),
    ("stt              stt_start->stt_end", lambda kv: _d(kv, "stt_start",
                                                          "stt_end")),
    ("gate             stt_end->you", lambda kv: _d(kv, "stt_end", "you")),
    ("prep             you->llm_post", lambda kv: _d(kv, "you", "llm_post")),
    ("llm              llm_post->llm_done",
     lambda kv: _d(kv, "llm_post", "llm_done")),
    ("prompt_eval_ms", lambda kv: _i(kv, "prompt_eval_ms")),
    ("eval_ms", lambda kv: _i(kv, "eval_ms")),
    ("load_ms", lambda kv: _i(kv, "load_ms")),
    ("total_ms         Ollama total", lambda kv: _i(kv, "total_ms")),
    ("llm_overhead     llm-prompt_eval-eval",
     lambda kv: _sub(_d(kv, "llm_post", "llm_done"),
                     _i(kv, "prompt_eval_ms"), _i(kv, "eval_ms"))),
    ("llm_outside      llm-total_ms",
     lambda kv: _sub(_d(kv, "llm_post", "llm_done"), _i(kv, "total_ms"))),
    ("post_llm         llm_done->actions_done",
     lambda kv: _d(kv, "llm_done", "actions_done")),
    ("speak_wait       actions_done->synth_start",
     lambda kv: _nonneg(_d(kv, "actions_done", "synth_start"))),
    ("synth            synth_start->first_play",
     lambda kv: _d(kv, "synth_start", "first_play")),
    ("play_open_ms     duck->stream open",
     lambda kv: _i(kv, "play_open_ms")),
    ("filler_ms        t0->first filler clip", lambda kv: _i(kv, "filler_ms")),
    ("filler_clip_ms   first filler clip", lambda kv: _i(kv, "filler_clip_ms")),
    ("you->first_play", lambda kv: _d(kv, "you", "first_play")),
    ("first_play       t0->first answer audio",
     lambda kv: _i(kv, "first_play")),
    ("end              t0->end", lambda kv: _i(kv, "end")),
)


def pct(xs, p):
    """Linear-interpolated percentile of the non-None values, rounded to an
    int (the method the speed plan's baseline used); None when empty."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return int(round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo)))


def _row(label, vals):
    xs = [v for v in vals if v is not None]
    if not xs:
        return f"  {label:<46s} {0:>4d} {'-':>7s} {'-':>7s}"
    return (f"  {label:<46s} {len(xs):>4d} {pct(xs, 50):>7d} "
            f"{pct(xs, 90):>7d}")


def table(turns, kind: str) -> list:
    kvs = [t["kv"] for t in turns]
    out = [f"  {'':<46s} {'n':>4s} {'p50':>7s} {'p90':>7s}"]
    e2e = E2E_MIC if kind == "voice" else E2E_TYPED
    out.append("  end to end (ms)")
    for label, fn in e2e:
        out.append(_row(label, [fn(kv) for kv in kvs]))
    out.append("  stages (ms)")
    for label, fn in STAGES:
        vals = [fn(kv) for kv in kvs]
        if any(v is not None for v in vals):
            out.append(_row(label, vals))
    return out


# ── splitting ─────────────────────────────────────────────────────────────
def split_key(spec: str):
    """--split value -> fn(turn) -> group label. Raises ValueError."""
    if spec == "filler":
        return lambda t: ("filler" if (_i(t["kv"], "filler") or 0) > 0
                          else "no-filler")
    if spec == "day":
        return lambda t: t["ts"].date().isoformat()
    if spec == "cut":
        return lambda t: "cut" if _i(t["kv"], "cut") is not None else "no-cut"
    if spec in ("eot", "stt_engine", "pre", "cache"):
        return lambda t: t["kv"].get(spec, "-") or "-"
    if spec.startswith("flag=") and len(spec) > 5:
        key = spec[5:]
        return lambda t: t["flags"].get(key, "?")
    raise ValueError(f"unknown --split {spec!r} (one of: {', '.join(SPLITS)})")


def _group_order(label: str):
    return (label in ("-", "?"), label)


# ── selection ─────────────────────────────────────────────────────────────
def parse_when(text: str, end_of_day: bool = False) -> datetime.datetime:
    """'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM[:SS]' (a 'T' works too). A bare
    date as --until means the end of that day."""
    s = text.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    d = datetime.datetime.strptime(s, "%Y-%m-%d")
    return d + datetime.timedelta(days=1) if end_of_day else d


def _in_window(ts, since, until):
    if since is not None and ts < since:
        return False
    if until is not None and ts >= until:
        return False
    return True


# ── report ────────────────────────────────────────────────────────────────
def build_report(log_dir, since=None, until=None, kinds=("voice", "inject"),
                 split=None, outcome="ok", exclude=DEFAULT_EXCLUDE) -> str:
    paths, turns, events = load(log_dir)
    excluded = set(exclude or ())
    sel, dropped = [], 0
    for t in turns:
        if not _in_window(t["ts"], since, until):
            continue
        if t["ts"].strftime("%Y-%m-%d %H:%M:%S") in excluded:
            dropped += 1
            continue
        sel.append(t)
    lines = ["JARVIS turn latency report (ms; read-only, numbers only)"]
    lines.append(f"logs: {log_dir}  ({len(paths)} session files)")
    win = (f"{since:%Y-%m-%d %H:%M:%S}" if since else "start") + " .. " + \
        (f"{until:%Y-%m-%d %H:%M:%S}" if until else "end")
    lines.append(f"window: {win}   outcome: {outcome}   "
                 f"excluded: {dropped} turn(s)")
    counts = {}
    for t in sel:
        key = (t["kv"].get("kind", "?"), t["kv"].get("outcome", "?"))
        counts[key] = counts.get(key, 0) + 1
    lines.append("turns: " + (", ".join(
        f"{KIND_LABEL.get(k, k)}/{o}={n}"
        for (k, o), n in sorted(counts.items())) or "none"))
    grouper = split_key(split) if split else None
    for kind in kinds:
        mine = [t for t in sel if t["kv"].get("kind") == kind
                and (outcome == "all" or t["kv"].get("outcome") == outcome)]
        lines.append("")
        lines.append(f"== {KIND_LABEL.get(kind, kind)} turns (kind={kind}, "
                     f"outcome={outcome})  n={len(mine)}")
        if not mine:
            continue
        lines.extend(table(mine, kind))
        g = grouper
        label = split
        if g is None and kind == "voice":
            g, label = split_key("filler"), "filler"
        if g is None:
            continue
        groups = {}
        for t in mine:
            groups.setdefault(g(t), []).append(t)
        for name in sorted(groups, key=_group_order):
            lines.append("")
            lines.append(f"-- {KIND_LABEL.get(kind, kind)} split {label}: "
                         f"{name}  n={len(groups[name])}")
            lines.extend(table(groups[name], kind))
    # Safety counters over the same window (any rise blocks a promotion).
    lines.append("")
    lines.append("== safety counters (lines in the window)")
    n_mic = sum(1 for t in sel if t["kv"].get("kind") == "voice")
    tally = {label: 0 for label, _ in SAFETY}
    for ts, label in events:
        if _in_window(ts, since, until):
            tally[label] += 1
    for label, _ in SAFETY:
        extra = ""
        if label == "wake-word mode refusals":
            per = (f"{tally[label] / n_mic:.2f}" if n_mic else "-")
            extra = f"   ({per} per mic turn)"
        lines.append(f"  {label:<46s} {tally[label]:>5d}{extra}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="p50/p90 turn latency from [turn-timing] log lines "
                    "(read-only; prints to stdout).")
    ap.add_argument("--logs", default=DEFAULT_LOG_DIR,
                    help=f"session log folder (default {DEFAULT_LOG_DIR})")
    ap.add_argument("--since", help="YYYY-MM-DD[ HH:MM[:SS]] (inclusive)")
    ap.add_argument("--until", help="YYYY-MM-DD[ HH:MM[:SS]] (a bare date "
                                    "includes that whole day)")
    ap.add_argument("--kind", choices=("voice", "inject", "realtime", "all"),
                    default=None, help="default: voice and inject")
    ap.add_argument("--split", help="filler | eot | stt_engine | pre | cut | "
                                    "cache | day | flag=<settings key>")
    ap.add_argument("--outcome", choices=("ok", "all"), default="ok")
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="'YYYY-MM-DD HH:MM:SS'",
                    help="drop the turn logged at this second (repeatable)")
    ap.add_argument("--no-default-exclude", action="store_true",
                    help=f"keep {', '.join(DEFAULT_EXCLUDE)}")
    a = ap.parse_args(argv)
    try:
        since = parse_when(a.since) if a.since else None
        until = parse_when(a.until, end_of_day=True) if a.until else None
        if a.split:
            split_key(a.split)
    except ValueError as exc:
        ap.error(str(exc))
    if not os.path.isdir(a.logs):
        print(f"no such log folder: {a.logs}", file=sys.stderr)
        return 2
    if a.kind is None:
        kinds = ("voice", "inject")
    elif a.kind == "all":
        kinds = ("voice", "inject", "realtime")
    else:
        kinds = (a.kind,)
    exclude = list(a.exclude)
    if not a.no_default_exclude:
        exclude.extend(DEFAULT_EXCLUDE)
    sys.stdout.write(build_report(a.logs, since=since, until=until,
                                  kinds=kinds, split=a.split,
                                  outcome=a.outcome, exclude=exclude))
    return 0


if __name__ == "__main__":
    sys.exit(main())
