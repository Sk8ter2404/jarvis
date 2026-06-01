"""External CLI helper that injects a voice-equivalent command into a
running JARVIS instance.

JARVIS's main loop polls `injected_commands.json` at the top of each
iteration (see bobert_companion._drain_injected_command). This script
appends one entry to that queue file atomically — so an external
tester (or a model running in Claude Code) can drive JARVIS without
going through the mic + Whisper path.

Usage:
    python tools/say_to_jarvis.py "what time is it"
    python tools/say_to_jarvis.py --wait "play Should I Stay or Should I Go"

With --wait, the script tails the most recent session log for ~30s
after the inject and prints any matching response block.
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE_PATH  = os.path.join(PROJECT_DIR, "injected_commands.json")
LOGS_DIR    = os.path.join(PROJECT_DIR, "logs")


def _read_existing_queue() -> list:
    if not os.path.exists(QUEUE_PATH):
        return []
    try:
        with open(QUEUE_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return []
        decoded, _ = json.JSONDecoder().raw_decode(raw)
        return decoded if isinstance(decoded, list) else []
    except Exception:
        return []


def enqueue(text: str) -> None:
    items = _read_existing_queue()
    items.append({"text": text, "ts": time.time()})
    fd, tmp = tempfile.mkstemp(dir=PROJECT_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, QUEUE_PATH)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise


def _latest_session_log() -> str | None:
    if not os.path.isdir(LOGS_DIR):
        return None
    candidates = sorted(
        glob.glob(os.path.join(LOGS_DIR, "*.log")) +
        glob.glob(os.path.join(LOGS_DIR, "*.txt")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return candidates[0] if candidates else None


def tail_for_response(text: str, timeout_s: float = 30.0) -> None:
    log_path = _latest_session_log()
    if log_path is None:
        print("[say_to_jarvis] no session log found in logs/ — skipping --wait tail")
        return
    print(f"[say_to_jarvis] tailing {log_path} for up to {timeout_s:.0f}s …")
    try:
        f = open(log_path, "r", encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[say_to_jarvis] can't open log: {e}")
        return
    try:
        f.seek(0, os.SEEK_END)
        start = time.time()
        saw_inject = False
        marker_inject = f"[inject]"
        marker_reply  = "JARVIS:"
        while time.time() - start < timeout_s:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            if marker_inject in line and text[:40] in line:
                saw_inject = True
                print(f"  {line.rstrip()}")
                continue
            if saw_inject:
                print(f"  {line.rstrip()}")
                if marker_reply in line:
                    return
    finally:
        f.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject a voice-equivalent command into a running JARVIS."
    )
    parser.add_argument("text", help="Utterance text to inject as a user turn.")
    parser.add_argument(
        "--wait", action="store_true",
        help="Tail the latest session log for ~30s and print the response.",
    )
    args = parser.parse_args()
    text = args.text.strip()
    if not text:
        print("[say_to_jarvis] refused: empty text", file=sys.stderr)
        return 2
    enqueue(text)
    print(f"[say_to_jarvis] queued: {text!r} → {QUEUE_PATH}")
    if args.wait:
        tail_for_response(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
