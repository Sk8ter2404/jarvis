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

Usage:  python tools/action_smoke.py [--json out.json] [--only a,b,c]
                                     [--no-skills] [--settings FILE]
                                     [--preflight-only] [--keep-sandbox]
Exit 0 when nothing crashed; 1 when any handler raised; 2 when the sandbox
is not hermetic (the sweep refuses to start); 3 when the sweep tried to
touch the real tree (each attempt was blocked).

HERMETIC SANDBOX (2026-10-01)
=============================
The 09-05 live diagnostic: "running action_smoke makes the LIVE JARVIS
speak fake alerts". The sweep set JARVIS_STAGING=1 and a redirected
JARVIS_SETTINGS_PATH, but dozens of state paths are bound to their module's
__file__ and honour no redirect: the pending-speech queue the live loop
speaks from (bobert_companion.proactive_announce and ~25 skills), the
inject and tray inboxes, jarvis_todo.md (the overnight pipeline's work
list), every root *_state.json, data/clean_shutdown.flag. Run from the live
install, every announcing action ("Reminder, sir — test") landed in the
live queue. Since 2026-09-30 the test package's live-data guard refuses
those writes — but only because this tool happens to import tests/, and a
refusal is not a sandbox: every state-writing action then crashed instead
of being swept, and JARVIS_ALLOW_LIVE_DATA=1 (or inheriting
JARVIS_STAGING=0, which `setdefault` kept) reopened the hole.

So the sweep no longer runs in the tree it lives in. The parent copies the
CODE (git-tracked files, plus untracked skills/*.py) into a fresh temp dir —
no data/, no queues, no root state — and re-runs itself there with every
redirect forced (not defaulted) into that copy: JARVIS_DATA_DIR,
JARVIS_SETTINGS_PATH, JARVIS_LOCK_DIR, JARVIS_STAGING=1, MUTE_TTS=1, and
JARVIS_GUARD_LIVE_ROOT pointing the live-data guard at the REAL tree, so the
copy is free to write while any write that reaches the real tree is refused
and counted. Every __file__-bound path now resolves inside the copy by
construction. Before a single action runs the child checks (hermetic_problems)
that the monolith, the data dir, the settings file, the lock dir and all three
queues are inside the sandbox and outside the real tree, then proves it by
queueing a probe announcement and finding it in the SANDBOX queue. Anything
off -> exit 2, nothing swept. The copy is deleted afterwards unless
--keep-sandbox.

What this does NOT isolate: the network, the GPU and the shared Ollama
daemon (see the hermetic guard the test package arms; never judge live
stability while a sweep runs), and the desktop itself (see _DENYLIST_*).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

# The tree this copy of the tool lives in. In the parent that is the real
# install / worktree; in the child it is the sandbox copy.
_HERE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Env the parent hands the child. _SANDBOX_FLAG marks a child run.
_SANDBOX_FLAG = "--in-sandbox"
_REAL_ROOT_ENV = "JARVIS_SMOKE_REAL_ROOT"
_GUARD_ROOT_ENV = "JARVIS_GUARD_LIVE_ROOT"   # read by tests/live_data_guard.py
# Inherited escapes that would re-open the hole; never passed to the child.
_ESCAPE_ENV = ("JARVIS_ALLOW_LIVE_DATA",)

# Top-level directories never copied into the sandbox (runtime state, bulk,
# VCS) when the tracked-file list is unavailable.
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "data", "data_staging", "logs", "logs_staging",
    "backups", "_backups", "models", "dist", "node_modules", "screenshots",
    "camera_previews", "memory", "tts", "Robot Project", ".claude",
})

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
    # persists its raw sweep arg ("test") as the Hue bridge IP and reports a
    # false OK; staging redirection is belt, this is suspenders.
    "hue_set_bridge_ip",
    # long-lived side effects / upgrade pipeline
    "upgrade", "start_overnight_upgrade", "check_for_updates", "check_updates",
    "is_there_an_update", "stop_pipeline", "queue_task", "create_skill",
    "reload_skills", "run_smoke_test", "run_diagnostic",
    # opens real windows / apps / long media flows on the dev box
    "open_url", "youtube", "youtube_play", "netflix", "prime_video",
    "disney_plus", "hulu", "max", "spotify", "play_streaming", "apple_music",
    "open_apple_music", "play_music", "play_vibe", "play_unheard",
    "web_search", "search",
    # 2026-07-11 sweep leaks: the youtube_search skill's aliases each opened
    # a REAL browser tab (3 tabs of the yt-dlp resolution of "test"), and
    # keep_music_open launches/pins the Apple Music app + a keep-alive
    # daemon.
    "youtube_direct", "youtube_search_direct", "yt_direct",
    "keep_music_open",
    # skills/site_builder.py: a paid Opus call for a site about "test",
    # then a real browser tab on the saved page.
    "build_website",
    # input injection on the live desktop
    "click", "press", "hotkey", "type", "scroll", "screenshot",
    "run_shell", "launch_app",
}

# Skills whose SPAWNING actions pop real overlay windows on the live desktop —
# JARVIS_STAGING isolates state paths, but the SCREEN is not staged: the
# 2026-07-11 sweep walked these alphabetically and stacked NINE HUDs on the
# owner's monitors (holo canvas, hud_v2, print monitor, bambu camera/overlay,
# workshop, arc reactor, full-screen jarvis_holo via suit_up; dossier's card
# left hud_card.pid in the repo root). Module-based so all ~100 aliases are
# covered without a name list; hide_*/dismiss_*/*_off/*_status stay swept —
# they only close/query windows.
_DENYLIST_MODULES = ("holographic_overlay", "dossier", "suit_up",
                     "night_owl_mode",
                     # morning_handoff's workspace actions ARRANGE REAL
                     # WINDOWS via win32 (launch apps, move/minimize,
                     # volume) — the 2026-07-11 sweep minimized the live
                     # JARVIS HUD through this path.
                     "morning_handoff",
                     # every youtube_search action opens a real browser tab
                     "youtube_search",
                     # show_globe / globe_pin open the holographic globe
                     "globe")


def _spawns_desktop_windows(name: str, fn) -> bool:
    mod = getattr(fn, "__module__", "") or ""
    if not any(m in mod for m in _DENYLIST_MODULES):
        return False
    safe = (name.endswith("_off") or name.endswith("_status")
            or name.startswith(("hide_", "dismiss_")))
    return not safe


# ── sandbox ────────────────────────────────────────────────────────────────

def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _inside(path: str, root: str) -> bool:
    p, r = _norm(path), _norm(root)
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:          # different drives
        return False


def hermetic_problems(real_root: str, sandbox_root: str, paths: dict) -> list:
    """Why the named runtime ``paths`` are NOT hermetic, or [] when every one
    lies inside ``sandbox_root`` and outside ``real_root``. Also refuses a
    sandbox that IS, or sits inside, the real tree (copying into a subdir of
    the live tree would make "inside the sandbox" mean "inside the live
    tree"). Pure; never raises."""
    problems = []
    try:
        if _inside(sandbox_root, real_root) or _inside(real_root, sandbox_root):
            problems.append(f"sandbox {sandbox_root!r} overlaps the real tree "
                            f"{real_root!r}")
        for label, path in sorted((paths or {}).items()):
            if not path:
                problems.append(f"{label}: unresolved")
                continue
            if _inside(path, real_root):
                problems.append(f"{label} -> {path} is inside the real tree "
                                f"{real_root}")
            elif not _inside(path, sandbox_root):
                problems.append(f"{label} -> {path} is outside the sandbox "
                                f"{sandbox_root}")
    except Exception as e:      # pragma: no cover - defensive
        problems.append(f"could not check the sandbox: {e}")
    return problems


def _code_files(real_root: str) -> list:
    """Relative paths of the code to copy: every git-tracked file, plus any
    UNTRACKED skills/*.py (gitignored personal skills are code, and the sweep
    should exercise them). Falls back to a walk that skips the runtime-state
    and bulk directories and every root-level file but code."""
    rels = []
    try:
        out = subprocess.run(["git", "-C", real_root, "ls-files", "-z"],
                             capture_output=True, timeout=60, check=True)
        rels = [r for r in out.stdout.decode("utf-8", "replace").split("\0")
                if r]
    except Exception:
        rels = []
    if not rels:
        for base, dirs, files in os.walk(real_root):
            rel_base = os.path.relpath(base, real_root)
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fn in files:
                if rel_base == "." and not fn.endswith((".py",)) \
                        and fn != "VERSION":
                    continue    # root runtime state (*.json, queues, logs)
                rels.append(os.path.normpath(os.path.join(rel_base, fn)))
    skills = os.path.join(real_root, "skills")
    if os.path.isdir(skills):
        for base, dirs, files in os.walk(skills):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    rels.append(os.path.relpath(os.path.join(base, fn),
                                                real_root))
    seen, out_rels = set(), []
    for r in rels:
        k = os.path.normcase(os.path.normpath(r))
        if k not in seen:
            seen.add(k)
            out_rels.append(os.path.normpath(r))
    return out_rels


def build_sandbox(real_root: str, base_dir: str | None = None,
                  settings: str | None = None) -> str:
    """Copy the code of ``real_root`` into a fresh temp dir and return the
    copy's root. It gets an EMPTY data/ (plus, with ``settings``, a copy of
    that settings file) and no queues or state files at all."""
    sandbox = tempfile.mkdtemp(prefix="jarvis_action_smoke_", dir=base_dir)
    tree = os.path.join(sandbox, "tree")
    for rel in _code_files(real_root):
        src = os.path.join(real_root, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(tree, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    os.makedirs(os.path.join(tree, "data"), exist_ok=True)
    os.makedirs(os.path.join(tree, "locks"), exist_ok=True)
    if settings:
        shutil.copy2(settings, os.path.join(tree, "data", "user_settings.json"))
    return tree


def sandbox_env(real_root: str, tree: str, base_env=None) -> dict:
    """The child's environment: every runtime redirect FORCED into the
    sandbox copy (never setdefault — an inherited JARVIS_STAGING=0 or a live
    JARVIS_DATA_DIR must not survive), the live-data guard pointed at the
    real tree, and the escape hatches stripped."""
    env = dict(os.environ if base_env is None else base_env)
    for k in _ESCAPE_ENV:
        env.pop(k, None)
    data = os.path.join(tree, "data")
    env.update({
        "JARVIS_STAGING": "1",
        "MUTE_TTS": "1",
        "JARVIS_TEST_MODE": "1",
        "JARVIS_BUG_AUTO_CAPTURE": "0",
        "JARVIS_DATA_DIR": data,
        "JARVIS_SETTINGS_PATH": os.path.join(data, "user_settings.json"),
        "JARVIS_LOCK_DIR": os.path.join(tree, "locks"),
        _GUARD_ROOT_ENV: real_root,
        _REAL_ROOT_ENV: real_root,
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def _arg_value(argv: list, flag: str):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _parent(argv: list) -> int:
    real_root = _HERE_ROOT
    settings = _arg_value(argv, "--settings")
    if settings and not os.path.isfile(settings):
        print(f"[smoke] --settings {settings!r}: no such file")
        return 2
    json_out = _arg_value(argv, "--json")
    tree = build_sandbox(real_root, settings=settings)
    sandbox = os.path.dirname(tree)
    try:
        env = sandbox_env(real_root, tree)
        problems = hermetic_problems(real_root, tree, {
            "data dir": env["JARVIS_DATA_DIR"],
            "settings": env["JARVIS_SETTINGS_PATH"],
            "lock dir": env["JARVIS_LOCK_DIR"],
            "code copy": tree,
        })
        if problems:
            print("[smoke] REFUSED — the sandbox is not hermetic:")
            for p in problems:
                print(f"    !! {p}")
            return 2
        child_argv = [a for a in argv if a not in ("--keep-sandbox",)]
        if json_out:
            i = child_argv.index("--json")
            child_argv[i + 1] = os.path.join(sandbox, "results.json")
        if settings:
            i = child_argv.index("--settings")
            del child_argv[i:i + 2]
        print(f"[smoke] sandbox: {tree}", flush=True)
        cmd = [sys.executable, os.path.join(tree, "tools", "action_smoke.py"),
               _SANDBOX_FLAG] + child_argv
        rc = subprocess.call(cmd, cwd=tree, env=env)
        if json_out and os.path.isfile(os.path.join(sandbox, "results.json")):
            shutil.copy2(os.path.join(sandbox, "results.json"), json_out)
            print(f"saved {json_out}")
        return rc
    finally:
        if "--keep-sandbox" in argv:
            print(f"[smoke] sandbox kept: {sandbox}")
        else:
            shutil.rmtree(sandbox, ignore_errors=True)


def _child_preflight(bc, real_root: str, tree: str) -> list:
    """Every runtime path the sweep can write, checked against the real tree
    BEFORE any action runs, then the speech queue proven by a probe."""
    from core import paths as _paths
    try:
        from tools.settings_window import settings_path as _settings_path
        settings = _settings_path()
    except Exception:
        settings = os.environ.get("JARVIS_SETTINGS_PATH", "")
    problems = hermetic_problems(real_root, tree, {
        "monolith": getattr(bc, "__file__", ""),
        "data dir": _paths.data_dir(create=False),
        "settings": settings,
        "lock dir": bc._singleton_lock_dir(),
        "speech queue": getattr(bc, "PENDING_SPEECH_PATH", ""),
        "inject queue": getattr(bc, "INJECTED_COMMANDS_PATH", ""),
        "tray inbox": getattr(bc, "TRAY_COMMANDS_FILE", ""),
    })
    if problems:
        return problems
    # Prove it: an announcement must land in THIS copy's queue.
    probe = f"action-smoke sandbox probe {os.getpid()}"
    queue = bc.PENDING_SPEECH_PATH
    try:
        if not bc.proactive_announce(probe, source="action-smoke"):
            return ["the probe announcement was not queued"]
        with open(queue, encoding="utf-8") as f:
            if probe not in f.read():
                return [f"the probe announcement is not in {queue}"]
    except Exception as e:
        return [f"the speech-queue probe failed: {e}"]
    finally:
        try:
            if os.path.exists(queue):
                os.remove(queue)
        except Exception:
            pass
    return []


def _real_tree_escapes(real_root: str) -> list:
    """Writes the live-data guard refused because they reached the REAL tree."""
    try:
        from tests import live_data_guard as g
        return [v for v in g.violations()
                if real_root and _norm(real_root) in os.path.normcase(
                    str(v.get("detail", "")))]
    except Exception:
        return []


def _child(argv: list) -> int:
    tree = _HERE_ROOT
    real_root = os.environ.get(_REAL_ROOT_ENV, "")
    if not real_root or _inside(tree, real_root):
        print("[smoke] REFUSED — the child must run inside a sandbox copy "
              "made by the parent (run tools/action_smoke.py without "
              f"{_SANDBOX_FLAG}).")
        return 2
    sys.path.insert(0, tree)
    os.chdir(tree)
    from tests._monolith_harness import load_monolith
    bc = load_monolith()
    problems = _child_preflight(bc, real_root, tree)
    if problems:
        print("[smoke] REFUSED — the sweep is not hermetic:")
        for p in problems:
            print(f"    !! {p}")
        return 2
    print("[smoke] preflight OK — every runtime path is inside the sandbox; "
          "the probe announcement landed in the sandbox queue")
    if "--preflight-only" in argv:
        return 0
    # MIRROR THE BOOT ALIAS (2026-07-14 audit, finding #13). At boot the
    # monolith does `sys.modules["bobert_companion"] = sys.modules["__main__"]`,
    # so ~18 skills bridge back to it with
    #     sys.modules.get("__main__") or sys.modules.get("bobert_companion")
    # which is CORRECT in production. But `__main__` ALWAYS exists, so that `or`
    # never falls through: in THIS process it resolves to the sweep runner, and
    # every one of those skills silently takes its dead "monolith unreachable"
    # branch — the sweep reports them "OK" while exercising nothing (and
    # local_vision, whose call sites were unguarded, actually CRASHED). Alias
    # __main__ to the monolith so the sweep drives the REAL code paths, exactly
    # as the live app does.
    sys.modules["_smoke_runner_main"] = sys.modules["__main__"]
    sys.modules["__main__"] = bc
    # Register the SKILL actions too — the core dict alone is ~135 of the
    # ~529 total. Skill register() functions may start daemon pollers; this
    # is a one-shot process, so they die with it.
    if "--no-skills" not in argv:
        try:
            bc.load_skills()
            print(f"[smoke] skills loaded — {len(bc.ACTIONS)} total actions")
        except Exception as e:
            print(f"[smoke] load_skills failed ({type(e).__name__}: {e}) — "
                  f"sweeping core actions only")
    actions: dict = dict(bc.ACTIONS)
    only = _arg_value(argv, "--only")
    if only:
        wanted = {n.strip().lower() for n in only.split(",") if n.strip()}
        actions = {k: v for k, v in actions.items() if k in wanted}
    deny_fns = {id(fn) for name, fn in bc.ACTIONS.items()
                if name in _DENYLIST_NAMES}

    results = {"ok": [], "honest_fail": [], "empty": [], "crash": [],
               "skipped": []}
    t_start = time.time()
    for name in sorted(actions):
        fn = actions[name]
        if (name in _DENYLIST_NAMES or id(fn) in deny_fns
                or _spawns_desktop_windows(name, fn)):
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

    escapes = _real_tree_escapes(real_root)
    results["blocked_real_tree_writes"] = [
        f"{v.get('kind')}: {v.get('detail')}" for v in escapes]

    print(f"\n=== ACTION SMOKE: {len(actions)} actions in "
          f"{time.time()-t_start:.0f}s ===")
    print(f"  OK:           {len(results['ok'])}")
    print(f"  honest-fail:  {len(results['honest_fail'])} "
          f"(hardware/creds absent on harness — returned an error STRING)")
    print(f"  empty:        {len(results['empty'])}")
    print(f"  skipped:      {len(results['skipped'])} (destructive/denylist)")
    print(f"  CRASH:        {len(results['crash'])}")
    print(f"  real-tree writes blocked: {len(escapes)}")
    for c in results["crash"]:
        print(f"    !! {c}")
    for e in results["empty"][:15]:
        print(f"    (empty) {e}")
    for e in results["blocked_real_tree_writes"]:
        print(f"    !! ESCAPE BLOCKED {e}")

    json_out = _arg_value(argv, "--json")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    if escapes:
        return 3
    return 1 if results["crash"] else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if _SANDBOX_FLAG in argv:
        argv.remove(_SANDBOX_FLAG)
        return _child(argv)
    return _parent(argv)


if __name__ == "__main__":
    sys.exit(main())
