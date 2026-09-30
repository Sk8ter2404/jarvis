"""Monolith-tier test for the boot-time update-check thread.

`_update_check_thread` lives at module level in the monolith (so it's coverage-
measured), but it only runs inside a daemon Thread at real boot. This drives it
directly with a no-op sleep + a mocked update_checker so both branches (update /
no update) are exercised without touching the network or sleeping 45s.

Monolith-tier: needs the monolith's heavy deps, so it runs locally and skips on
the light CI tier.
    python -m unittest tests.monolith.test_monolith_update_check
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class UpdateCheckThreadTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        # boot_nudge stamps ``nudged_for`` into the update cache after an
        # announcement, and the cache path is bound to core/update_checker's
        # __file__ (no env redirect reaches it): without this the second test
        # REWROTE the live data/update_check.json on every run (found
        # 2026-09-30 by a write audit).
        tmp = tempfile.mkdtemp(prefix="jarvis_update_check_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.cache = os.path.join(tmp, "update_check.json")
        p = mock.patch("core.update_checker.default_cache_path",
                       return_value=self.cache)
        p.start()
        self.addCleanup(p.stop)

    def test_no_update_is_silent(self):
        with mock.patch.object(self.bc.time, "sleep", lambda *_a, **_k: None), \
             mock.patch("core.update_checker.cached_check",
                        return_value={"update_available": False, "current": "1.2.0"}), \
             mock.patch.object(self.bc, "proactive_announce") as pa:
            self.bc._update_check_thread()
        pa.assert_not_called()

    def test_update_available_announces_once(self):
        with mock.patch.object(self.bc.time, "sleep", lambda *_a, **_k: None), \
             mock.patch("core.update_checker.cached_check",
                        return_value={"update_available": True, "latest": "v1.3.0",
                                      "current": "1.2.0"}), \
             mock.patch.object(self.bc, "proactive_announce") as pa:
            self.bc._update_check_thread()
        pa.assert_called_once()
        self.assertIn("v1.3.0", pa.call_args.args[0])
        self.assertEqual(pa.call_args.kwargs.get("source"), "update_check")
        # The stamp landed in the temp cache - proof the redirect is real.
        from core import update_checker
        self.assertEqual(
            (update_checker.read_cache(self.cache) or {}).get("nudged_for"),
            "v1.3.0")


if __name__ == "__main__":
    unittest.main()
