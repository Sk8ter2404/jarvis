"""Unload Ollama's resident model while a game is running. Out-of-process, and
it only ever UNLOADS.

WHY THIS EXISTS, and why it cannot live inside JARVIS
-----------------------------------------------------
Measured on the owner's machine 2026-09-04. He shut JARVIS down from the tray at
20:47:08 to free VRAM for Fortnite. At 20:55 — eight minutes after the JARVIS
process was gone — `llama-server` was STILL holding 21.3 GB of commit and 15 GB
of the 3090's 24 GB.

That is not a bug in JARVIS's teardown. `keep_alive` is OLLAMA's state, not
JARVIS's: JARVIS asks for a generation, Ollama keeps the runner warm for the
keep_alive window, and the runner outlives whoever asked. So no amount of
in-process JARVIS logic can hold that memory down — by construction. The lever
has to sit outside JARVIS, and it has to keep pulling, because anything that
touches Ollama re-warms the model:

    bobert_companion.py  _call_local_llm            (chat)
    bobert_companion.py  _call_local_vision         (vision)
    bobert_companion.py  _warm_up_local_llm_async   (BOOT, posts keep_alive 20m)
    core/orchestrator.py                            (its own one-shot /api/chat)
    core/rag_indexer.py                             (/api/embeddings)

Five request sites, not one. An earlier design gated two of them and called it
coverage — this repo's #1 bug class, partial coverage presented as coverage.
A latch that re-unloads every tick needs no chokepoint: it does not care WHO
reloaded the model or why, only that it is resident while a game is running.

The boot path matters most. `_warm_up_local_llm_async()` fires at
bobert_companion.py:24903 and deliberately loads the model resident with
keep_alive 20m. If JARVIS restarts mid-game — watchdog, crash, or the owner —
boot allocates ~15 GB into a card that was measured with 766 MiB free. This
guard is what makes that survivable.

WHY UNLOAD-ONLY
---------------
It restores nothing, re-enables nothing, and remembers nothing. There is no
prior value to capture and therefore no state it can strand: if this process is
killed at any instant, the worst outcome is a model that stays unloaded and
reloads on the owner's next question. Compare a suspend/restore design, which
must correctly capture and replay a dozen prior values and can leave JARVIS
wedged if it dies mid-transition. The asymmetry is the whole argument.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not fix a leaking game. Fortnite's lobby leak was measured tonight at a
sustained 0.5-0.7 GB/min with no plateau, taking the game process from 10 GB to
67 GB in under two hours; even with JARVIS entirely off, that alone exhausts
48 GB. This guard reclaims the ~15 GB JARVIS's brain holds. That is a real and
worthwhile chunk, and it is not a cure for the game.

It also does not touch physical RAM. Trimming idle processes' working sets (the
thing that actually kept the owner playing tonight, +4.53 GB on the first pass)
is a separate concern with a separate lifetime, and is not folded in here.

stdlib only — urllib, no requests, no psutil.

Usage:
    python tools/vram_yield_guard.py                 # run until the game exits
    python tools/vram_yield_guard.py --dry-run       # report, unload nothing
    python tools/vram_yield_guard.py --games a.exe,b.exe
    python tools/vram_yield_guard.py --once          # single pass, for tests
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA

# Defaulted IN CODE, not in a data file. data/ belongs to the running JARVIS and
# a guard that needs a seed file to work is a guard that silently does nothing
# the first time it is needed.
DEFAULT_GAMES = (
    "FortniteClient-Win64-Shipping.exe",
    "RocketLeague.exe",
    "cs2.exe",
    "VALORANT-Win64-Shipping.exe",
    "eldenring.exe",
    "Cyberpunk2077.exe",
    "helldivers2.exe",
    "gta5.exe",
    "Overwatch.exe",
)

# The owner's explicit override. `full_power` (or any caller) touches this file;
# while it is fresher than the TTL the guard keeps its hands off, so it can
# never unload a brain he just asked for — possibly mid-generation. Lives beside
# this file's project root but NOT under data/.
INHIBIT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".vram_guard_inhibit")
INHIBIT_TTL_S = 900.0            # 15 min, matching the 'full power' window


def _get(path: str, timeout: float = 5.0):
    """GET a small JSON document from Ollama, or None. Never raises."""
    try:
        with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def loaded_models() -> list[dict]:
    """Currently resident runners, newest API shape first. Never raises."""
    doc = _get("/api/ps")
    if not isinstance(doc, dict):
        return []
    out = []
    for m in doc.get("models") or []:
        if isinstance(m, dict) and m.get("name"):
            out.append({"name": m["name"],
                        "size_vram": int(m.get("size_vram") or 0),
                        "size": int(m.get("size") or 0)})
    return out


def _num_ctx(model: str) -> int:
    """The context this model's warm runner was loaded with.

    MUST match, and must never be omitted. Ollama keys a runner by
    (model, options), so a /api/generate post without num_ctx does not address
    the resident runner at all — it asks for a DIFFERENT one at the model's own
    default window (262144 for gemma4:26b-a4b). Measured live 2026-07-21: that
    put `ollama ps` at CONTEXT 262144 with the 3090 pinned at 24147/24576 MiB
    and the model spilled to CPU, and the next voice turn died on the 50 s read
    timeout. A VRAM guard that did that would cause the exact failure it exists
    to prevent — which is why tests/test_ollama_runner_reuse.py fails any
    generation post that omits it (it caught this file on 2026-09-04)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.ollama_opts import local_num_ctx
        return int(local_num_ctx(model))
    except Exception:
        # Never fall back to "no options" — that IS the bug. 12288 is this
        # repo's measured big-model window and is safe for any tag: worst case
        # it addresses a runner that is not resident, which is a no-op.
        return 12288


def unload(model: str, timeout: float = 30.0) -> bool:
    """Ask Ollama to drop this runner now (keep_alive 0). True if accepted.

    Uses the HTTP API rather than shelling `ollama stop` so there is no console
    window, no PATH dependency, and a real timeout."""
    body = json.dumps({"model": model, "prompt": "", "keep_alive": 0,
                       "options": {"num_ctx": _num_ctx(model)}}).encode()
    req = urllib.request.Request(
        OLLAMA + "/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True
    except urllib.error.HTTPError:
        # A 4xx here means Ollama understood and refused (e.g. model gone).
        # Treat as done rather than retrying forever.
        return True
    except Exception:
        return False


def running_games(names: tuple[str, ...]) -> list[str]:
    """Which allowlisted game executables are running. Never raises.

    tasklist is used rather than WMI/CIM because it is ~20x cheaper and this
    runs on a poll loop while the owner is gaming — the probe must not be the
    thing that costs him frames."""
    try:
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    live = out.stdout.lower()
    return [n for n in names if n.lower() in live]


def inhibited(now: float | None = None) -> bool:
    """True while the owner has explicitly asked for full power."""
    try:
        age = (now or time.time()) - os.path.getmtime(INHIBIT_FILE)
        return age < INHIBIT_TTL_S
    except Exception:
        return False


def hold_full_power() -> None:
    """Public helper for a 'full power' voice action: suppress the guard."""
    try:
        with open(INHIBIT_FILE, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def _mib(n: int) -> str:
    return f"{n / (1 << 20):,.0f} MiB"


def tick(names: tuple[str, ...], dry: bool) -> tuple[int, int]:
    """One pass. Returns (games_seen, models_unloaded)."""
    games = running_games(names)
    if not games:
        return 0, 0
    if inhibited():
        print(f"  inhibited (full power) — leaving {len(loaded_models())} "
              f"model(s) resident", flush=True)
        return len(games), 0
    freed = 0
    for m in loaded_models():
        tag = f"{m['name']} ({_mib(m['size_vram'])} VRAM)"
        if dry:
            print(f"  DRY-RUN would unload {tag}", flush=True)
            continue
        if unload(m["name"]):
            freed += 1
            print(f"  unloaded {tag} — game running: {', '.join(games)}",
                  flush=True)
        else:
            print(f"  FAILED to unload {tag}", flush=True)
    return len(games), freed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", default="",
                    help="comma-separated exe names (adds to the defaults)")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--max-minutes", type=float, default=480.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--idle-exits", type=int, default=6,
                    help="stand down after this many consecutive ticks with no "
                         "game running (0 = never exit)")
    a = ap.parse_args(argv)

    names = tuple(DEFAULT_GAMES) + tuple(
        s.strip() for s in a.games.split(",") if s.strip())

    print(f"[vram-guard] watching {len(names)} game(s) against {OLLAMA}; "
          f"unload-only, interval {a.interval:g}s"
          f"{' (DRY RUN)' if a.dry_run else ''}", flush=True)

    if a.once:
        g, f = tick(names, a.dry_run)
        print(f"[vram-guard] once: {g} game(s), {f} unloaded", flush=True)
        return 0

    deadline = time.time() + a.max_minutes * 60
    idle = 0
    total = 0
    while time.time() < deadline:
        games, freed = tick(names, a.dry_run)
        total += freed
        if games:
            idle = 0
        else:
            idle += 1
            if a.idle_exits and idle >= a.idle_exits:
                print(f"[vram-guard] no game for {idle} ticks — standing down "
                      f"({total} unload(s) this run)", flush=True)
                return 0
        time.sleep(a.interval)
    print(f"[vram-guard] time limit reached ({total} unload(s))", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
