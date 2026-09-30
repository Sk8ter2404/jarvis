#!/usr/bin/env python3
"""run-jarvis driver — boot, wake, and DRIVE JARVIS (Bobert) headlessly.

JARVIS is a voice assistant, but it exposes a file-based INJECT CHANNEL so you
can drive the LIVE app WITHOUT a microphone: the main loop drains
``injected_commands.json`` at the top of every iteration (this is the PROD
path, not a test stub — the action fires for real). This driver wraps the exact
flow the project's developers use to exercise the running app:

    1. ensure JARVIS is up   (boot via _boot_jarvis.ps1 if it isn't)
    2. force-wake it         (tray command channel — mic-independent; standby
                              DROPS injected commands, so this is required)
    3. inject an utterance   (append to injected_commands.json, atomically)
    4. tail the session log  (return JARVIS's reply + the [action] result)

Run from anywhere — the project root is found by walking up to the dir that
holds ``bobert_companion.py``.

Usage:
    python .claude/skills/run-jarvis/driver.py "what time is it"
    python .claude/skills/run-jarvis/driver.py --boot "play billie jean"
    python .claude/skills/run-jarvis/driver.py --status
    python .claude/skills/run-jarvis/driver.py --wake

Exit code 0 = a reply/action was captured (or --status found it running),
1 = not running / timed out / standby dropped it.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

ALIVE_WINDOW_S = 15          # JARVIS logs whisper/vad constantly; a log write
                             # within this many seconds means it's alive.


def project_dir() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "bobert_companion.py")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    # Fallback: .claude/skills/run-jarvis/driver.py -> project is 3 levels up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


PROJ = project_dir()
INJECT = os.path.join(PROJ, "injected_commands.json")
TRAY = os.path.join(PROJ, "tray_commands.json")
LOGS = os.path.join(PROJ, "logs")
BOOT = os.path.join(PROJ, "_boot_jarvis.ps1")


def _write_atomic(path: str, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=PROJ, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def latest_log() -> str | None:
    files = glob.glob(os.path.join(LOGS, "session_*.log"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _tail(path: str, n: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


def _prod_jarvis_pids() -> list[int] | None:
    """PIDs of running PROD JARVIS processes (bobert_companion.py without
    --staging), or None when processes cannot be listed here (no psutil)."""
    try:
        import psutil
    except Exception:
        return None
    pids = []
    try:
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = (p.info.get("name") or "").lower()
                cmd = " ".join(p.info.get("cmdline") or [])
            except Exception:
                continue
            if name.startswith("python") and "bobert_companion" in cmd and "--staging" not in cmd:
                pids.append(int(p.info["pid"]))
    except Exception:
        return None
    return pids


def is_running() -> bool:
    """True when a PROD JARVIS process exists.

    The old test -- "the session log was written in the last 15 s" -- misfires on
    an idle JARVIS, which only logs 'Listening…' about every 20 s: the driver then
    ran _boot_jarvis.ps1, whose first step KILLS the running instance (seen live
    2026-09-29 14:57, mid test-sweep). Log freshness is now only the fallback when
    processes cannot be listed."""
    pids = _prod_jarvis_pids()
    if pids is not None:
        return bool(pids)
    lg = latest_log()
    return bool(lg) and (time.time() - os.path.getmtime(lg)) < ALIVE_WINDOW_S


def boot(timeout: float = 45.0) -> bool:
    print("[driver] booting JARVIS via _boot_jarvis.ps1 ...", file=sys.stderr)
    env = dict(os.environ)
    # 2026-07-15 (P2 phase B): default brain is gemma4:26b-a4b-it-qat now that TTS
    # moved OFF the 3090 (Kokoro-on-CPU) freed ~13GB. Measured on-box: 0 empties
    # over 5 real turns, 94-110 tok/s, 18.8GB resident → ~5.8GB free with the brain
    # alone (the old "26b returns EMPTY" claim was a since-fixed ollama bug). 26B MoE,
    # multimodal so vision shares it. Falls back to gemma4:12b via _LOCAL_LLM_PREFERENCE
    # + the empty-response failover. Only set as a DEFAULT (setdefault) so a per-boot
    # JARVIS_LOCAL_LLM_MODEL override (A/B, or a lower-VRAM box) still wins.
    env.setdefault("JARVIS_LOCAL_LLM_MODEL", "gemma4:26b-a4b-it-qat")
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", BOOT],
            cwd=PROJ, env=env, capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        print(f"[driver] boot launch failed: {e}", file=sys.stderr)
        return False
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(1.5)
        lg = latest_log()
        if lg and any(k in _tail(lg, 30).lower()
                      for k in ("listening on", "standby", "sleeping", "ready")):
            print("[driver] JARVIS is up.", file=sys.stderr)
            return True
    print("[driver] boot timed out waiting for the ready marker.", file=sys.stderr)
    return False


def force_wake() -> None:
    # The inject channel is mic-independent, so the wake WORD can't reach it;
    # the tray command channel is drained regardless of sleep/standby.
    # Append, never overwrite: a command the app hasn't drained yet (a queued
    # "restart" from a redeploy, say) must not be silently replaced.
    items = []
    if os.path.exists(TRAY):
        try:
            items = json.loads(open(TRAY, encoding="utf-8").read().strip() or "[]")
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []
    items.append({"cmd": "force_wake"})
    _write_atomic(TRAY, json.dumps(items))
    time.sleep(1.5)


def inject(text: str) -> None:
    items = []
    if os.path.exists(INJECT):
        try:
            raw = open(INJECT, encoding="utf-8").read().strip() or "[]"
            items = json.loads(raw)
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []
    items.append({"text": text, "ts": time.time()})
    _write_atomic(INJECT, json.dumps(items, indent=2))


QUIET_S = 3.0          # a plain reply is complete after this much log silence
ACTION_QUIET_S = 6.0   # after an [action] line: the informative follow-up round
                       # prints "Reading results" up to ~4 s later (seen live)


def _reply_complete(lines: list[str], pending_followup: bool, quiet_for: float) -> bool:
    """Has this turn finished talking? Never while a follow-up round is running
    or an [ACTION: …] tag still waits for its [action] result; otherwise after
    QUIET_S of log silence (ACTION_QUIET_S when the last line was a result)."""
    if not lines or pending_followup:
        return False
    last_reply = next((l for l in reversed(lines) if "jarvis:" in l.lower()), "")
    if "[action:" in last_reply.lower():
        after = lines[lines.index(last_reply) + 1:]
        if not any("[action]" in l.lower() for l in after):
            return False
    need = ACTION_QUIET_S if "[action]" in lines[-1].lower() else QUIET_S
    return quiet_for >= need


def wait_for_reply(text: str, timeout: float = 75.0) -> dict:
    """Tail the live log from the current end, return the JARVIS reply + the
    [action] result line(s) for this utterance -- including the answer of an
    informative action's follow-up round, which lands several seconds after
    the [action] line. (It used to return 2 s after the first [action] line,
    before that answer, and to sit out the whole timeout on replies with no
    action at all.)"""
    lg = latest_log()
    if not lg:
        return {"status": "no_log", "lines": []}
    pos = os.path.getsize(lg)
    snippet = text[:30].lower()
    saw_inject = False
    lines: list[str] = []
    pending_followup = False
    last_new = time.time()
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            with open(lg, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except Exception:
            continue
        for line in chunk.splitlines():
            low = line.lower()
            if "[inject]" in low and snippet in low:
                saw_inject = True
                last_new = time.time()
                if "(standby)" in low:
                    # standby will drop it — caller should --wake and retry.
                    return {"status": "standby_ignored", "lines": []}
                continue
            if not saw_inject:
                continue
            if "reading results" in low:
                pending_followup = True
                last_new = time.time()
            if "jarvis:" in low or "[action]" in low:
                lines.append(line.rstrip())
                last_new = time.time()
                if "jarvis:" in low:
                    pending_followup = False
        if _reply_complete(lines, pending_followup, time.time() - last_new):
            return {"status": "ok", "lines": lines}
    return {"status": "timeout" if not lines else "partial", "lines": lines}


def main() -> int:
    ap = argparse.ArgumentParser(description="Drive JARVIS headlessly via the inject channel.")
    ap.add_argument("utterance", nargs="?", help="What to say to JARVIS.")
    ap.add_argument("--boot", action="store_true", help="(Re)boot before driving.")
    ap.add_argument("--status", action="store_true", help="Just report whether JARVIS is running.")
    ap.add_argument("--wake", action="store_true", help="Just force-wake (clear sleep/standby).")
    ap.add_argument("--timeout", type=float, default=75.0)
    args = ap.parse_args()

    print(f"[driver] project: {PROJ}", file=sys.stderr)

    if args.status:
        up = is_running()
        print(f"JARVIS running: {up}  (log: {latest_log()})")
        return 0 if up else 1

    if args.boot or not is_running():
        if not is_running():
            print("[driver] JARVIS not running.", file=sys.stderr)
        if not boot():
            return 1

    # Wake only when asked (--wake) or when standby actually dropped the
    # command (handled below). An unconditional force_wake made JARVIS say
    # "At your service, sir." before EVERY driven turn, and his desk mic heard
    # it and answered it as the owner (live 2026-09-29 quality sweep).
    if args.wake:
        force_wake()

    if args.wake and not args.utterance:
        print("[driver] force-woke JARVIS.")
        return 0

    if not args.utterance:
        ap.error("give an utterance, or use --status / --wake")

    inject(args.utterance)
    print(f"[driver] injected: {args.utterance!r}", file=sys.stderr)
    res = wait_for_reply(args.utterance, timeout=args.timeout)

    if res["status"] == "standby_ignored":
        print("[driver] standby dropped the command — retrying after a wake ...", file=sys.stderr)
        force_wake()
        inject(args.utterance)
        res = wait_for_reply(args.utterance, timeout=args.timeout)

    print(f"\n=== JARVIS ({res['status']}) ===")
    for line in res["lines"]:
        print(line)
    return 0 if res["status"] in ("ok", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
