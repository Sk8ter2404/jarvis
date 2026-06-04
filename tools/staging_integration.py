#!/usr/bin/env python3
"""Heavy/LOCAL integration gate: boot a STAGING JARVIS ONCE (no mic, muted TTS)
and run a battery of read-only utterances through the REAL pipeline
(capture-inject -> LLM -> action -> reply), asserting NON-FABRICATED replies.

This is the one tier the unit suite can't reach: the ~14K-line monolith can't
import on a bare runner, so the end-to-end main loop is validated behaviourally
here instead. It is NOT part of `tools/run_tests.py` (which must stay fast +
hermetic for CI) — run it explicitly on a machine with ANTHROPIC_API_KEY set:

    python tools/staging_integration.py            # default battery
    python tools/staging_integration.py -v         # echo the captured replies

Exit 0 = booted + every battery item answered with its expected marker; else 1.

Non-fabrication is proven by markers the LLM cannot invent:
  * version  -> "1.20.5" (only readable from the VERSION file via the
                 version_info action),
  * system   -> "cpu"        (system_pulse reads live psutil stats),
  * time     -> "it is"      (get_time formats the real clock).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INJECT = os.path.join(PROJECT, "injected_commands_staging.json")
LOG = os.path.join(PROJECT, "_staging_integration.log")

# (utterance, [expected lowercase substrings — ALL must appear in the reply window])
DEFAULT_BATTERY = [
    ("what time is it", ["current time is"]),       # get_time real-clock output
    ("what version are you on", ["1.20.5"]),          # version_info reads VERSION file
    ("give me a system status report", ["cpu"]),     # system_pulse live psutil stats
]

_lines: list[str] = []
_lock = threading.Lock()


def _pump(stream, fh) -> None:
    for raw in iter(stream.readline, ""):
        with _lock:
            _lines.append(raw.rstrip("\n"))
        try:
            fh.write(raw)
            fh.flush()
        except Exception:
            pass


def _wait(substr: str, timeout: float, since: int = 0):
    """Return the 1-based line index where `substr` first appears at/after
    `since`, or None on timeout."""
    deadline = time.time() + timeout
    sub = substr.lower()
    while time.time() < deadline:
        with _lock:
            for i in range(since, len(_lines)):
                if sub in _lines[i].lower():
                    return i + 1
        time.sleep(0.4)
    return None


def _inject(text: str) -> None:
    tmp = INJECT + ".tmp"
    # NO BOM — JARVIS rejects a BOM'd inject file as corrupt JSON.
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([{"text": text}], f)
    os.replace(tmp, INJECT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the captured JARVIS: reply for each item")
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    ap.add_argument("--reply-timeout", type=float, default=90.0)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[integration] ANTHROPIC_API_KEY not set — the LLM turn will fail.\n"
              "  Inject it first, e.g. (PowerShell):\n"
              "  $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')")
        return 2

    for p in (INJECT, INJECT + ".consuming", LOG):
        try:
            os.remove(p)
        except OSError:
            pass

    env = dict(os.environ)
    env.update({"JARVIS_STAGING": "1", "MUTE_TTS": "1", "JARVIS_TEST_MODE": "1",
                "PYTHONUNBUFFERED": "1"})

    print("[integration] booting staging JARVIS (no mic, muted)...", flush=True)
    fh = open(LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", "bobert_companion.py", "--staging"],
        cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, close_fds=True,
    )
    threading.Thread(target=_pump, args=(proc.stdout, fh), daemon=True).start()

    results: list[tuple[str, bool, str]] = []
    try:
        booted = _wait("listening", args.boot_timeout) or _wait("standby", 5) \
            or _wait("[vad]", 5)
        if booted is None:
            print("[integration] FAIL — never reached standby/Listening", flush=True)
            return _finish(proc, fh, results, ok=False)
        print(f"[integration] booted (marker @ line {booted}); running battery of "
              f"{len(DEFAULT_BATTERY)}...", flush=True)

        for utterance, expected in DEFAULT_BATTERY:
            mark = len(_lines)
            _inject(utterance)
            reply_idx = _wait("jarvis:", args.reply_timeout, since=mark)
            reply_line = ""
            if reply_idx is not None:
                with _lock:
                    reply_line = _lines[reply_idx - 1]
            missing = [s for s in expected if _wait(s, 6, since=mark) is None]
            ok = reply_idx is not None and not missing
            note = ("no reply" if reply_idx is None
                    else f"missing {missing}" if missing else "ok")
            results.append((utterance, ok, note))
            mark_char = "PASS" if ok else "FAIL"
            print(f"  [{mark_char}] {utterance!r} -> {note}", flush=True)
            if args.verbose and reply_line:
                print(f"         reply: {reply_line.strip()[:160]}", flush=True)
            time.sleep(1.0)  # let the loop settle before the next inject
    finally:
        pass

    return _finish(proc, fh, results, ok=all(r[1] for r in results) and bool(results))


def _finish(proc, fh, results, *, ok: bool) -> int:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass
    try:
        fh.flush()
        fh.close()
    except Exception:
        pass
    try:
        os.remove(INJECT)
    except OSError:
        pass
    passed = sum(1 for _, k, _ in results if k)
    print(f"\n[integration] {passed}/{len(results)} passed — "
          f"VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
