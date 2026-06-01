#!/usr/bin/env python3
"""Boot a STAGING JARVIS (no mic, muted TTS) and verify it answers an injected
command end-to-end — exercises the real LLM + TTS render path without touching
prod or the speakers. Used to validate hot-path refactors (LLM client, TTS
cache, prompt assembly) with a real round-trip.

    python tools/staging_inject_smoke.py "what time is it" --expect "it is"

Exit 0 = booted + answered + (optional) expected substring seen; else 1.
Requires ANTHROPIC_API_KEY in the environment for the LLM turn.
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
LOG = os.path.join(PROJECT, "_staging_smoke.log")

_lines: list[str] = []
_lock = threading.Lock()


def _pump(stream, fh):
    for raw in iter(stream.readline, ""):
        with _lock:
            _lines.append(raw.rstrip("\n"))
        try:
            fh.write(raw); fh.flush()
        except Exception:
            pass


def _wait(substr: str, timeout: float, since: int = 0):
    deadline = time.time() + timeout
    sub = substr.lower()
    while time.time() < deadline:
        with _lock:
            for i in range(since, len(_lines)):
                if sub in _lines[i].lower():
                    return i + 1
        time.sleep(0.5)
    return None


def _inject(text: str):
    tmp = INJECT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([{"text": text}], f)
    os.replace(tmp, INJECT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", help="utterance to inject once booted")
    ap.add_argument("--expect", default="", help="substring expected in the reply")
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    ap.add_argument("--reply-timeout", type=float, default=90.0)
    args = ap.parse_args()

    for p in (INJECT, INJECT + ".consuming", LOG):
        try: os.remove(p)
        except OSError: pass

    env = dict(os.environ)
    env.update({"JARVIS_STAGING": "1", "MUTE_TTS": "1", "JARVIS_TEST_MODE": "1",
                "PYTHONUNBUFFERED": "1"})

    print(f"[smoke] booting staging JARVIS, will inject: {args.command!r}", flush=True)
    fh = open(LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", "bobert_companion.py", "--staging"],
        cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        close_fds=True,
    )
    threading.Thread(target=_pump, args=(proc.stdout, fh), daemon=True).start()

    ok = True
    try:
        booted = _wait("Listening", args.boot_timeout) or _wait("skills loaded", 5)
        print(f"[smoke] boot marker @ {booted}", flush=True)
        if booted is None:
            ok = False
        else:
            mark = len(_lines)
            _inject(args.command)
            you = _wait("You:", 40, since=mark) or _wait("[inject]", 5, since=mark)
            reply = _wait("JARVIS:", args.reply_timeout, since=mark)
            print(f"[smoke] echo @ {you}  reply @ {reply}", flush=True)
            ok = reply is not None
            if ok and args.expect:
                hit = _wait(args.expect, 5, since=mark)
                print(f"[smoke] expected {args.expect!r} @ {hit}", flush=True)
                ok = hit is not None
    finally:
        try:
            proc.terminate()
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired: proc.kill()
        except Exception:
            pass
        try: fh.flush(); fh.close()
        except Exception: pass

    print(f"\n[smoke] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
