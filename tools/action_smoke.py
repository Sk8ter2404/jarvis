"""action_smoke.py — execute EVERY registered action handler once and report.

The unit suite tests handlers it knows about; live batteries test what gets
spoken. This sweep closes the gap the owner kept hitting ("this is all
supposed to be tested"): it enumerates the COMPLETE ACTIONS dict off the
monolith harness (real handler code, stubbed hardware) and calls each one
with a benign argument, cataloguing crashes, empty returns, and honest
failures. It is a SMOKE layer — "the handler runs and returns a string" —
not a behaviour oracle; pair it with the live voice battery.

Skipped by design (would kill the harness process / wipe state / hang on
real I/O): see _DENYLIST. Everything else runs, including hardware-touching
handlers — on the harness their probes fail HONESTLY, and an exception
(rather than an error string) is exactly the bug class this exists to catch.

Usage:  python tools/action_smoke.py [--json out.json]
Exit 0 when nothing crashed; 1 when any handler raised.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")

# STATE ISOLATION — set BEFORE anything imports the monolith. The first full
# sweep ran against the LIVE state: an action persisted KINECT_GAZE_ENABLED
# into data/user_settings.json, which then leaked into unrelated face-tracker
# unit tests and turned two ci_sim gates red (2026-07-10). JARVIS_STAGING=1
# flips the blue/green isolation: every module-level state path (locks,
# settings, data files, logs) is rerouted to the *_staging equivalents, so a
# sweep can execute settings-toggling actions without mutating the real box.
os.environ.setdefault("JARVIS_STAGING", "1")

# Handlers that must NOT be invoked from a sweep: process control, state
# wipes, spawning long-lived subprocesses/threads that outlive the harness,
# or blocking interactive flows. Names, not handlers — aliases resolve to the
# same fn and get skipped via the resolved id too.
_DENYLIST_NAMES = {
    # process / power control
    "restart", "exit_jarvis", "quit_jarvis", "shutdown_jarvis", "shut_down",
    "power_off_jarvis", "turn_off_jarvis", "reboot",
    # destructive state
    "reset_memory", "forget_last_hour", "export_memory", "clear_tasks",
    "smart_home_purge_cookie", "forget_alexa_login",
    # long-lived side effects / upgrade pipeline
    "upgrade", "start_overnight_upgrade", "check_for_updates", "check_updates",
    "is_there_an_update", "stop_pipeline", "queue_task", "create_skill",
    "reload_skills", "run_smoke_test", "run_diagnostic",
    # opens real windows / apps / long media flows on the dev box
    "open_url", "youtube", "youtube_play", "netflix", "prime_video",
    "disney_plus", "hulu", "max", "spotify", "play_streaming", "apple_music",
    "open_apple_music", "play_music", "play_vibe", "play_unheard",
    "web_search", "search",
    # input injection on the live desktop
    "click", "press", "hotkey", "type", "scroll", "screenshot",
    "run_shell", "launch_app",
}

def main() -> int:
    from tests._monolith_harness import load_monolith
    bc = load_monolith()
    # Register the SKILL actions too — the core dict alone is ~135 of the
    # ~529 total. Skill register() functions may start daemon pollers; this
    # is a one-shot process, so they die with it.
    if "--no-skills" not in sys.argv:
        try:
            bc.load_skills()
            print(f"[smoke] skills loaded — {len(bc.ACTIONS)} total actions")
        except Exception as e:
            print(f"[smoke] load_skills failed ({type(e).__name__}: {e}) — "
                  f"sweeping core actions only")
    actions: dict = dict(bc.ACTIONS)
    deny_fns = {id(fn) for name, fn in actions.items()
                if name in _DENYLIST_NAMES}

    results = {"ok": [], "honest_fail": [], "empty": [], "crash": [],
               "skipped": []}
    t_start = time.time()
    for name in sorted(actions):
        fn = actions[name]
        if name in _DENYLIST_NAMES or id(fn) in deny_fns:
            results["skipped"].append(name)
            continue
        try:
            t0 = time.time()
            out = fn("test")
            dt = time.time() - t0
            if out is None or (isinstance(out, str) and not out.strip()):
                results["empty"].append(name)
            elif isinstance(out, str) and any(
                    m in out.lower() for m in
                    ("failed", "couldn't", "can't", "unavailable", "error",
                     "not configured", "not installed", "no ", "unable")):
                results["honest_fail"].append(f"{name}: {out[:90]}")
            else:
                results["ok"].append(name)
            if dt > 20:
                print(f"  SLOW {name}: {dt:.0f}s", flush=True)
        except Exception as e:
            results["crash"].append(f"{name}: {type(e).__name__}: {e}")

    print(f"\n=== ACTION SMOKE: {len(actions)} actions in "
          f"{time.time()-t_start:.0f}s ===")
    print(f"  OK:           {len(results['ok'])}")
    print(f"  honest-fail:  {len(results['honest_fail'])} "
          f"(hardware/creds absent on harness — returned an error STRING)")
    print(f"  empty:        {len(results['empty'])}")
    print(f"  skipped:      {len(results['skipped'])} (destructive/denylist)")
    print(f"  CRASH:        {len(results['crash'])}")
    for c in results["crash"]:
        print(f"    !! {c}")
    for e in results["empty"][:15]:
        print(f"    (empty) {e}")

    if "--json" in sys.argv:
        out_path = sys.argv[sys.argv.index("--json") + 1]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out_path}")
    return 1 if results["crash"] else 0


if __name__ == "__main__":
    sys.exit(main())
