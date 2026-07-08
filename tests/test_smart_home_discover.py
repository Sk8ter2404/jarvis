"""Focused unit tests for ``skills.smart_home_discover``.

These target the 2026-07-08 fix (finding #18): ``_run_async`` must bound a
coroutine with an overall timeout so a stalled alexapy / Playwright endpoint
can't wedge the voice dispatch thread. Stdlib ``unittest`` only — no network,
no alexapy, no event loop leakage.
"""
from __future__ import annotations

import asyncio
import unittest

from skills import smart_home_discover as d


class RunAsyncTimeoutTests(unittest.TestCase):
    def test_fast_coro_returns_value_with_timeout_set(self):
        async def _quick():
            await asyncio.sleep(0)
            return "ok"

        self.assertEqual(d._run_async(_quick(), timeout=5.0), "ok")

    def test_no_timeout_preserves_legacy_behaviour(self):
        async def _quick():
            return 42

        self.assertEqual(d._run_async(_quick()), 42)

    def test_slow_coro_raises_timeout(self):
        async def _slow():
            await asyncio.sleep(5.0)
            return "never"

        with self.assertRaises(TimeoutError):
            d._run_async(_slow(), timeout=0.1)

    def test_timeout_constant_is_bounded(self):
        # A sane ceiling — generous for a warm cookie fetch, small enough to
        # fail fast rather than hang the assistant loop.
        self.assertTrue(0 < d._DISCOVERY_TIMEOUT_SEC <= 120)


if __name__ == "__main__":
    unittest.main()
