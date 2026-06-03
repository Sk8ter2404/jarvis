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

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class UpdateCheckThreadTests(MonolithGlobalsTestCase):
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


if __name__ == "__main__":
    unittest.main()
