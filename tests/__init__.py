# JARVIS test suite (stdlib unittest — no external deps, runs headless).
# Run all:  python tools/run_tests.py    (or)   python -m unittest discover -s tests
#
# THE CHOKEPOINT for the REAL-BROWSER guard (incident 2026-08-20, 11:38-11:41:
# full-CI runs spammed dozens of live tabs into the owner's DEFAULT Chrome
# profile via production code calling webbrowser.open()). Importing ANY test
# module imports this package first, so arming the guard here covers every entry
# path there is — `python -m unittest tests.foo`, `unittest discover`, the capped
# scratchpad harness, an IDE's test runner, and all three tools/ runners (which
# also call it explicitly next to the memory ceiling; install() is idempotent).
#
# Do NOT delete this block when refactoring: tests/test_browser_guard.py reads
# this file's SOURCE and fails if the install call disappears (this repo's #1 bug
# class is the rule that quietly stops being applied in one of its copies).
# Escape hatch for a human debugging a real browser flow: JARVIS_ALLOW_REAL_BROWSER=1.
try:  # never let the guard break test COLLECTION — an unguarded run beats no run
    import os as _os
    import sys as _sys

    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from tools import browser_guard as _browser_guard

    _browser_guard.install()
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[browser-guard] WARNING: not armed from tests/__init__.py "
          f"({type(_exc).__name__}: {_exc}) — this run CAN open REAL browsers",
          flush=True)

# THE CHOKEPOINT for the LIVE-DATA guard (incident 2026-08-20: test runs deleted
# the owner's hand-written data/clean_shutdown.flag four times in one day and
# JARVIS resurrected against his explicit instruction). It belongs HERE, beside
# the browser guard, and NOT in the tools/ runners — the escape route is a
# DETACHED bobert_companion.py child spawned from a daemon thread 1.5s after the
# triggering test returned, and it resolves the flag path off its own __file__,
# so neither JARVIS_DATA_DIR nor JARVIS_STAGING nor any runner-scoped redirect
# can reach it. See tests/live_data_guard.py for the captured stack.
#
# Do NOT delete this block when refactoring: tests/test_live_data_guard.py reads
# this file's SOURCE and fails if the install call disappears (this repo's #1 bug
# class is the rule that quietly stops being applied in one of its copies).
# Escape hatch for a human deliberately driving live state: JARVIS_ALLOW_LIVE_DATA=1.
try:  # never let the guard break test COLLECTION — an unguarded run beats no run
    from tests import live_data_guard as _live_data_guard

    _live_data_guard.install()
except Exception as _exc:  # noqa: BLE001 - collection must survive anything
    print(f"[live-data-guard] WARNING: not armed from tests/__init__.py "
          f"({type(_exc).__name__}: {_exc}) — this run CAN damage the owner's "
          f"live data/", flush=True)
