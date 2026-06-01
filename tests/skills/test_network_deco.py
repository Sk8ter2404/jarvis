"""Logic tests for skills/network_deco.py (the genericised TP-Link Deco skill).

No real router / ARP / network: `is_available` and `_current_snapshot` are
mocked so the action degradation paths run deterministically without reaching a
router or reading the owner's real `data/deco_network.json`. Also pins the
env-driven genericisation (no hardcoded owner subnet).
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class NetworkDecoTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("network_deco")

    # ── env-driven genericisation ────────────────────────────────────────
    def test_subnet_prefix_is_derived_from_host(self):
        m = self.mod
        self.assertEqual(m.DECO_SUBNET_PREFIX,
                         m.DECO_HOST_DEFAULT.rsplit(".", 1)[0] + ".")
        self.assertTrue(m.DECO_SUBNET_PREFIX.endswith("."))

    def test_printer_ips_is_a_set(self):
        # Derived from the BAMBU_PRINTER_IP env var (empty set when unset).
        self.assertIsInstance(self.mod._PRINTER_IPS, set)

    # ── action surface ───────────────────────────────────────────────────
    def test_registers_core_actions(self):
        for a in ("who_is_on_wifi", "is_printer_online", "deco_status"):
            self.assertIn(a, self.actions)
        self.assertGreater(len(self.actions), 8)

    # ── graceful degradation (the skill's core contract) ─────────────────
    def test_actions_degrade_gracefully_without_router(self):
        # No router library + no cached snapshot -> informative string, no crash.
        with mock.patch.object(self.mod, "is_available", return_value=False), \
             mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
             mock.patch.object(self.mod, "_refresh_snapshot", return_value=None), \
             mock.patch.object(self.mod, "_host", return_value="192.168.1.1"), \
             mock.patch.object(self.mod, "_arp_table", return_value=[]):
            for a in ("who_is_on_wifi", "is_printer_online", "deco_status",
                      "network_usage", "deco_topology"):
                out = self.actions[a]("")
                self.assertIsInstance(out, str, a)
                self.assertTrue(out.strip(), f"{a} returned empty")


if __name__ == "__main__":
    unittest.main()
