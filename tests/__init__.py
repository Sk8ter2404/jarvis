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
