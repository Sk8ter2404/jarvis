# JARVIS test suite (stdlib unittest — no external deps, runs headless).
# Run all:  python tools/run_tests.py    (or)   python -m unittest discover -s tests
#
# THIS FILE IS THE CHOKEPOINT for all three "a test run must not damage the box
# it runs on" guards. Importing ANY test module imports this package first, so
# arming them here covers every entry path there is — `python -m unittest
# tests.foo`, `unittest discover`, the capped scratchpad harness, an IDE's test
# runner, and all three tools/ runners (which also call them explicitly; every
# one is idempotent).
#
# Do NOT delete these blocks when refactoring: tests/test_mem_guard.py,
# tests/test_browser_guard.py and tests/test_live_data_guard.py each read this
# file's SOURCE and fail if their install call disappears, moves inside a
# function, loses its try/except, or changes order (this repo's #1 bug class is
# the rule that quietly stops being applied in one of its copies).

# ── 1. MEMORY CEILING ───────────────────────────────────────────────────────
# Incident 2026-08-20 05:04: an uncapped run committed ~144 GB on a 48 GB box
# and BUGCHECKED the machine. The three tools/ runners applied the ceiling; this
# file did not — so `python -m unittest tests.<suite>`, which is exactly the
# BISECT path that produced the bugcheck, ran unbounded while still printing the
# browser guard's armed banner. It goes FIRST so it also bounds the guards' own
# imports. Escape hatch: JARVIS_TEST_MEM_CAP_GB=0.
try:  # never let a guard break test COLLECTION — an unguarded run beats no run
    import os as _os
    import sys as _sys

    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from tools.mem_guard import apply_memory_ceiling as _apply_memory_ceiling

    _apply_memory_ceiling()
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[mem-guard] WARNING: no ceiling from tests/__init__.py "
          f"({type(_exc).__name__}: {_exc}) — a runaway allocation in this run "
          f"can exhaust this machine", flush=True)

# ── 2. LIVE-DATA GUARD ──────────────────────────────────────────────────────
# Incident 2026-08-20: test runs deleted the owner's hand-written
# data/clean_shutdown.flag four times in one day and JARVIS resurrected against
# his explicit instruction. It belongs HERE, and NOT in the tools/ runners — the
# escape route is a DETACHED bobert_companion.py child spawned from a daemon
# thread 1.5s after the triggering test returned, and it resolves the flag path
# off its own __file__, so neither JARVIS_DATA_DIR nor JARVIS_STAGING nor any
# runner-scoped redirect can reach it. See tests/live_data_guard.py.
#
# IT MUST BE ARMED BEFORE THE BROWSER GUARD. Both wrap os.startfile, and the
# browser guard's stub BLOCKS — it never calls what it wrapped — so the guard
# installed LAST is the only one that ever runs, and the guard installed FIRST
# is the one the other's _reset_for_tests() restores instead of discarding.
# Arming this one first gives both properties: while the browser guard is on,
# nothing shell-launches at all; when it is disabled via
# JARVIS_ALLOW_REAL_BROWSER, this hook is what still refuses a boot script.
# Escape hatch for a human deliberately driving live state: JARVIS_ALLOW_LIVE_DATA=1.
try:  # never let a guard break test COLLECTION — an unguarded run beats no run
    from tests import live_data_guard as _live_data_guard

    _live_data_guard.install()
    print(_live_data_guard.banner(), flush=True)
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[live-data-guard] WARNING: not armed from tests/__init__.py "
          f"({type(_exc).__name__}: {_exc}) — this run CAN damage the owner's "
          f"live data/", flush=True)

# ── 3. REAL-BROWSER GUARD ───────────────────────────────────────────────────
# Incident 2026-08-20 11:38-11:41: full-CI runs spammed dozens of live tabs into
# the owner's DEFAULT Chrome profile via production code calling
# webbrowser.open().
# Escape hatch for a human debugging a real browser flow: JARVIS_ALLOW_REAL_BROWSER=1.
try:  # never let a guard break test COLLECTION — an unguarded run beats no run
    from tools import browser_guard as _browser_guard

    _browser_guard.install()
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[browser-guard] WARNING: not armed from tests/__init__.py "
          f"({type(_exc).__name__}: {_exc}) — this run CAN open REAL browsers",
          flush=True)

# ── 4. HERMETIC GUARD (network / real input / hardware probes / screen) ────
# Found 2026-09-30 with a live JARVIS up: suites reached the live Ollama
# (/api/ps, /api/tags), itunes.apple.com, the real nvidia-smi and `ollama ps`,
# a real SetForegroundWindow, and a HUD overlay window - all through
# production code, all green for the wrong reason. One sys.addaudithook hook
# refuses and records each (a hook cannot be displaced by a test's
# mock.patch); the atexit summary names the offending tests. 2026-10-02: a
# tripwire run caught a monolith test photographing the owner's whole desktop
# (mss, then PIL.ImageGrab), so it refuses real screen captures too. Armed
# LAST: it wraps nothing the other guards wrap. Escape hatches:
# JARVIS_ALLOW_REAL_NETWORK, JARVIS_ALLOW_REAL_INPUT,
# JARVIS_ALLOW_HARDWARE_PROBES, JARVIS_ALLOW_SCREEN_CAPTURE (=1). See
# tools/hermetic_guard.py.
try:  # never let a guard break test COLLECTION — an unguarded run beats no run
    from tools import hermetic_guard as _hermetic_guard

    _hermetic_guard.install()
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[hermetic-guard] WARNING: not armed from tests/__init__.py "
          f"({type(_exc).__name__}: {_exc}) — this run CAN reach the network, "
          f"real input, live hardware and the screen", flush=True)

# ── 5. NO CUDA DRIVER (v2.0.180) ────────────────────────────────────────────
# core/cuda_preinit starts the NVIDIA driver (cuInit, no context) before
# ctranslate2 loads - lazily, from the Whisper / standby-detector / Smart Turn
# paths that unit tests drive with fakes. A test must never load the real
# driver, whatever CUDA_VISIBLE_DEVICES says; core/cuda_preinit skips (loads
# nothing) while this is "1". tests/test_cuda_preinit.py pins it. Escape hatch:
# JARVIS_NO_CUDA_DRIVER=0 set before the run.
try:  # never let a guard break test COLLECTION — an unguarded run beats no run
    import os as _os_cuda

    _os_cuda.environ.setdefault("JARVIS_NO_CUDA_DRIVER", "1")
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[cuda-guard] WARNING: JARVIS_NO_CUDA_DRIVER not set "
          f"({type(_exc).__name__}: {_exc})", flush=True)

# ── 6. NO SCREEN READS (2026-10-05) ────────────────────────────────────────
# core.screen_scope (EnumWindows titles), core.uia_host (UI Automation),
# core.screen_ocr (the OCR worker) and the click / screen-memory captures read
# the owner's REAL windows. A test must never see them - not even a window
# title in its output: every one of those readers asks
# core.screen_privacy.reads_blocked(), which is true while this is "1", and a
# test injects fakes instead (tests/_screen_fakes.py). Escape hatch:
# JARVIS_NO_SCREEN_READ=0 set before the run.
try:  # never let a guard break test COLLECTION — an unguarded run beats no run
    import os as _os_screen

    _os_screen.environ.setdefault("JARVIS_NO_SCREEN_READ", "1")
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[screen-read-guard] WARNING: JARVIS_NO_SCREEN_READ not set "
          f"({type(_exc).__name__}: {_exc})", flush=True)
