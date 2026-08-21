"""Focused unit tests for ``skills.smart_home_discover``.

These target the 2026-07-08 fix (finding #18): ``_run_async`` must bound a
coroutine with an overall timeout so a stalled alexapy / Playwright endpoint
can't wedge the voice dispatch thread — plus the smart-home catalog-from-LAN
rework: ``_lan_discover_all`` / ``_build_catalog_v2`` / the device_id-aware
merge / the non-blocking voice action. Stdlib ``unittest`` only — no network,
no alexapy, no real brand-skill imports (LAN sources are injected as fake
modules under the monolith loader names), no event loop leakage.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

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


# ════════════════════════════════════════════════════════════════════════════
# Catalog-from-LAN rework (smarthome cluster). Fake brand skills are injected
# under the monolith loader names (``skill_sh_kasa`` …) so the router's
# _skill_module lookup finds them FIRST and no real brand skill (with its UDP
# broadcast / multicast / bridge probes) ever loads.
# ════════════════════════════════════════════════════════════════════════════
_SENTINEL = object()


@contextlib.contextmanager
def _inject_modules(**mods):
    """Temporarily install (or force-absent via ``name=None``) fake modules in
    ``sys.modules`` for the with-block; restores prior state on exit."""
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _fake_source(name, devices=None, *, raises=None, sleep_s=0.0):
    """A fake brand skill module exposing ``list_devices()``."""
    mod = types.ModuleType(name)

    def list_devices():
        if sleep_s:
            time.sleep(sleep_s)
        if raises is not None:
            raise raises
        return [dict(x) for x in (devices or [])]

    mod.list_devices = list_devices
    return mod


def _kasa_five():
    # Addresses here are RFC 5737 TEST-NET-1 (192.0.2.0/24) on purpose: this
    # repo is public and tools/check_no_pii.py HARD-fails on the owner's real
    # LAN subnet. Never paste a real 192.168.x address into these fixtures.
    return [{"name": f"entry light {i}", "brand": "TP-Link", "model": "HS103",
             "type": "plug", "capabilities": ["on_off"],
             "lan_ip": f"192.0.2.{60 + i}"} for i in range(5)]


def _tuya_one_with_secret():
    return [{"name": "garage plug", "brand": "Tuya", "model": "gw",
             "type": "plug", "capabilities": ["on_off"],
             "lan_ip": "192.0.2.90",
             "_tuya": {"id": "dev1", "key": "SECRET-LOCAL-KEY"}}]


def _all_sources(kasa=None, tuya=None, govee=None, hue=None):
    """Injectable fakes for ALL four LAN sources (monolith loader names)."""
    return {
        "skill_sh_kasa": kasa if kasa is not None
        else _fake_source("skill_sh_kasa", _kasa_five()),
        "skill_sh_tuya": tuya if tuya is not None
        else _fake_source("skill_sh_tuya", []),
        "skill_sh_govee": govee if govee is not None
        else _fake_source("skill_sh_govee", []),
        "skill_sh_hue": hue if hue is not None
        else _fake_source("skill_sh_hue", []),
    }


class _DiscoverTestBase(unittest.TestCase):
    """Redirect every module path constant at a throwaway temp dir (mirrors
    _RouterTestBase in tests/test_smart_home_router.py) so the real catalog,
    cookies and todo file are never read or written."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="shdiscover_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        data_dir = os.path.join(self.tmp, "data")
        os.makedirs(data_dir, exist_ok=True)
        self._saved = {}
        for k, v in {
            "_DATA_DIR": data_dir,
            "_COOKIE_JSON_PATH": os.path.join(data_dir, "alexa_cookie.json"),
            "_COOKIE_PICKLE_PATH": os.path.join(data_dir,
                                                "alexa_cookie.pickle"),
            "_CATALOG_PATH": os.path.join(data_dir,
                                          "smart_home_devices.json"),
            "_TODO_PATH": os.path.join(self.tmp, "jarvis_todo.md"),
        }.items():
            self._saved[k] = getattr(d, k)
            setattr(d, k, v)
        self.addCleanup(self._restore_paths)

    def _restore_paths(self):
        for k, v in self._saved.items():
            setattr(d, k, v)

    def _write_catalog(self, obj):
        with open(d._CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def _read_catalog(self):
        with open(d._CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)


# ── _lan_discover_all ───────────────────────────────────────────────────────
class LanDiscoverAllTests(_DiscoverTestBase):
    def test_collects_all_sources_and_strips_secrets(self):
        arp = [{"ip": "192.0.2.60", "mac": "AA:BB:CC:DD:EE:60",
                "oui": "AABBCC", "brand_oui_hint": "TP-Link"}]
        mods = _all_sources(tuya=_fake_source("skill_sh_tuya",
                                              _tuya_one_with_secret()))
        with _inject_modules(**mods):
            records, status = d._lan_discover_all(arp)
        self.assertEqual(len(records), 6)   # 5 kasa + 1 tuya
        for r in records:
            self.assertIn(r["controller_skill"], ("sh_kasa", "sh_tuya"))
            self.assertTrue(r["source"].startswith("lan:"))
            self.assertTrue(r["device_id"])
            self.assertEqual(r["alexa_entity_id"], "")
        # HARD GATE: the Tuya local key must never reach the catalog.
        # (Quoted-key form — "sh_tuya" itself contains the _tuya substring.)
        self.assertNotIn("SECRET-LOCAL-KEY", json.dumps(records))
        self.assertNotIn('"_tuya"', json.dumps(records))
        # ARP-matched record gets a stable mac device_id; others a name slug.
        by_name = {r["name"]: r for r in records}
        self.assertEqual(by_name["entry light 0"]["device_id"],
                         "mac:AA:BB:CC:DD:EE:60")
        self.assertEqual(by_name["garage plug"]["device_id"],
                         "lan:sh_tuya:garage_plug")
        self.assertTrue(status["sh_kasa"]["ok"])
        self.assertEqual(status["sh_kasa"]["count"], 5)
        self.assertTrue(status["sh_tuya"]["ok"])
        self.assertEqual(status["sh_govee"]["count"], 0)

    def test_one_source_raising_does_not_hurt_others(self):
        mods = _all_sources(
            kasa=_fake_source("skill_sh_kasa",
                              raises=RuntimeError("udp dead")),
            tuya=_fake_source("skill_sh_tuya", _tuya_one_with_secret()))
        with _inject_modules(**mods):
            records, status = d._lan_discover_all([])
        self.assertEqual([r["controller_skill"] for r in records],
                         ["sh_tuya"])
        self.assertFalse(status["sh_kasa"]["ok"])
        self.assertIn("udp dead", status["sh_kasa"]["error"])
        self.assertTrue(status["sh_tuya"]["ok"])

    def test_straggler_abandoned_under_deadline(self):
        # House-rule pin: a stalled source must NOT wedge the caller — the
        # deadline-bounded join abandons the daemon thread.
        mods = _all_sources(hue=_fake_source("skill_sh_hue", [{"name": "x"}],
                                             sleep_s=5.0))
        t0 = time.monotonic()
        with mock.patch.object(d, "_LAN_SOURCE_TIMEOUT_SEC", 0.5), \
             mock.patch.object(d, "_LAN_TOTAL_TIMEOUT_SEC", 0.5), \
             _inject_modules(**mods):
            records, status = d._lan_discover_all([])
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 3.0,
                        f"LAN discovery ran {elapsed:.1f}s — join not bounded")
        self.assertFalse(status["sh_hue"]["ok"])
        self.assertEqual(status["sh_hue"]["error"], "timed out")
        # The fast sources still delivered.
        self.assertEqual(status["sh_kasa"]["count"], 5)

    def test_unavailable_skill_reported_not_fatal(self):
        mods = _all_sources()
        mods["skill_sh_govee"] = types.ModuleType("skill_sh_govee")  # no API
        with _inject_modules(**mods):
            records, status = d._lan_discover_all([])
        self.assertEqual(status["sh_govee"],
                         {"ok": False, "count": 0,
                          "error": "skill unavailable"})
        self.assertEqual(status["sh_kasa"]["count"], 5)


# ── _build_catalog_v2 ───────────────────────────────────────────────────────
def _lan_status_ok(kasa=5, tuya=0, govee=0, hue=0):
    return {"sh_kasa": {"ok": True, "count": kasa},
            "sh_tuya": {"ok": True, "count": tuya},
            "sh_govee": {"ok": True, "count": govee},
            "sh_hue": {"ok": True, "count": hue}}


class BuildCatalogV2Tests(_DiscoverTestBase):
    def _lan_records(self):
        return [d._lan_record("sh_kasa", r, []) for r in _kasa_five()]

    def test_lan_only_build(self):
        alexa_status = {"ok": False, "count": 0,
                        "error": "no cached Amazon sign-in", "console": True}
        cat = d._build_catalog_v2(None, alexa_status, self._lan_records(),
                                  _lan_status_ok(), [], force=False)
        self.assertEqual(cat["version"], 2)
        self.assertEqual(cat["device_count"], 5)
        self.assertFalse(cat["sources"]["alexa"]["ok"])
        self.assertTrue(cat["sources"]["sh_kasa"]["ok"])
        for rec in cat["devices"]:
            self.assertEqual(rec["controller_skill"], "sh_kasa")
            self.assertEqual(rec["source"], "lan:sh_kasa")

    def test_alexa_and_lan_twin_deduped_and_enriched(self):
        devices = {"echo": [], "groups": [], "smarthome": [
            {"friendlyName": "entry light 0", "manufacturerName": "TP-Link",
             "entityId": "ent-1",
             "capabilities": [{"interface": "Alexa.PowerController"}]},
        ]}
        cat = d._build_catalog_v2(devices, {"ok": True, "count": 1},
                                  self._lan_records(), _lan_status_ok(),
                                  [], force=False)
        # 1 Alexa record + 5 LAN records sharing one identity → 5 devices.
        self.assertEqual(cat["device_count"], 5)
        twin = next(r for r in cat["devices"] if r["name"] == "entry light 0")
        self.assertEqual(twin["alexa_entity_id"], "ent-1")   # Alexa kept
        self.assertEqual(twin["lan_ip"], "192.0.2.60")    # LAN filled in
        self.assertEqual(twin["source"], "alexa")

    def test_union_preserves_missed_lan_device_unless_forced(self):
        kept = d._lan_record("sh_kasa",
                             {"name": "porch plug", "brand": "TP-Link",
                              "type": "plug", "capabilities": ["on_off"],
                              "lan_ip": "192.0.2.99"}, [])
        self._write_catalog({"device_count": 1, "devices": [kept]})
        fresh_lan = self._lan_records()   # porch plug NOT re-found
        cat = d._build_catalog_v2(None, {"ok": False, "count": 0},
                                  fresh_lan, _lan_status_ok(), [],
                                  force=False)
        self.assertEqual(cat["device_count"], 6)
        self.assertIn("porch plug", [r["name"] for r in cat["devices"]])
        forced = d._build_catalog_v2(None, {"ok": False, "count": 0},
                                     fresh_lan, _lan_status_ok(), [],
                                     force=True)
        self.assertEqual(forced["device_count"], 5)
        self.assertNotIn("porch plug", [r["name"] for r in forced["devices"]])


# ── _merge_with_existing_catalog by device_id ───────────────────────────────
class MergeByDeviceIdTests(_DiscoverTestBase):
    def test_user_rename_survives_but_fresh_lan_ip_wins(self):
        stored = d._lan_record("sh_kasa",
                               {"name": "entry light 0", "brand": "TP-Link",
                                "type": "plug", "capabilities": ["on_off"],
                                "lan_ip": "192.0.2.60"},
                               [{"ip": "192.0.2.60",
                                 "mac": "AA:BB:CC:DD:EE:60"}])
        stored["name"] = "porch light"          # user's manual rename
        self._write_catalog({"device_count": 1, "devices": [stored]})
        fresh_rec = d._lan_record("sh_kasa",
                                  {"name": "entry light 0",
                                   "brand": "TP-Link", "type": "plug",
                                   "capabilities": ["on_off"],
                                   "lan_ip": "192.0.2.77"},   # DHCP moved
                                  [{"ip": "192.0.2.77",
                                    "mac": "AA:BB:CC:DD:EE:60"}])
        merged = d._merge_with_existing_catalog(
            {"device_count": 1, "devices": [fresh_rec]})
        dev = merged["devices"][0]
        self.assertEqual(dev["name"], "porch light")        # rename preserved
        self.assertEqual(dev["lan_ip"], "192.0.2.77")    # fresh scan wins

    def test_alexa_record_manual_lan_ip_still_preserved(self):
        # Old behaviour pinned: for a non-LAN (Alexa) record the user's manual
        # lan_ip survives a re-run.
        self._write_catalog({"device_count": 1, "devices": [
            {"alexa_entity_id": "e1", "name": "Office Lamp",
             "lan_ip": "10.0.0.9", "lan_mac": "AA:AA:AA:AA:AA:AA"}]})
        fresh = {"device_count": 1, "devices": [
            {"alexa_entity_id": "e1", "name": "Office Lamp",
             "source": "alexa", "lan_ip": "", "lan_mac": ""}]}
        merged = d._merge_with_existing_catalog(fresh)
        self.assertEqual(merged["devices"][0]["lan_ip"], "10.0.0.9")


# ── voice action end-to-end (no cookie / cookie timeout) ────────────────────
class VoiceActionEndToEndTests(_DiscoverTestBase):
    def _run(self, arg="", *, tuya_secret=False, run_async=None):
        mods = _all_sources(
            tuya=_fake_source("skill_sh_tuya", _tuya_one_with_secret())
            if tuya_secret else None)
        patches = [
            mock.patch.object(d, "_alexapy",
                              return_value=types.ModuleType("alexapy")),
            mock.patch.object(d, "_scan_lan_arp", return_value=[]),
            mock.patch.object(d, "_say"),
        ]
        if run_async is not None:
            patches.append(mock.patch.object(d, "_run_async",
                                             side_effect=run_async))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(_inject_modules(**mods))
            return d.smart_home_discover(arg)

    def test_no_cookie_scans_lan_and_reports_alexa_unavailable(self):
        # THE core regression: without a cookie the old action returned the
        # CLI hint and never touched the LAN.
        out = self._run(tuya_secret=True)
        self.assertNotEqual(out, d._CLI_HINT)
        self.assertIn("5 Kasa", out)
        self.assertIn("Alexa discovery unavailable", out)
        cat = self._read_catalog()
        self.assertEqual(cat["device_count"], 6)
        self.assertFalse(cat["sources"]["alexa"]["ok"])
        # HARD GATE: no Tuya secret material in the saved catalog.
        # (Quoted-key form — "sh_tuya" itself contains the _tuya substring.)
        raw = json.dumps(cat)
        self.assertNotIn("SECRET-LOCAL-KEY", raw)
        self.assertNotIn('"_tuya"', raw)

    def test_cookie_timeout_still_lands_lan_catalog(self):
        # Old regression: a TimeoutError aborted the action with "catalog is
        # unchanged" — now the LAN catalog still lands and Alexa is reported.
        with open(d._COOKIE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"email": "e", "saved_at": time.time()}, f)
        with open(d._COOKIE_PICKLE_PATH, "wb") as f:
            f.write(b"x")

        def _raise_timeout(coro, timeout=None):
            coro.close()
            raise TimeoutError("discovery timed out")

        out = self._run(run_async=_raise_timeout)
        self.assertIn("timed out", out)
        self.assertEqual(self._read_catalog()["device_count"], 5)

    def test_force_skips_merge_and_rebuilds_clean(self):
        stale = d._lan_record("sh_kasa",
                              {"name": "gone plug", "brand": "TP-Link",
                               "type": "plug", "capabilities": ["on_off"],
                               "lan_ip": "192.0.2.99"}, [])
        self._write_catalog({"device_count": 1, "devices": [stale]})
        out = self._run("force")
        # The old action refused outright with the CLI hint on force; the new
        # one completes the rebuild (the hint survives only as the
        # Alexa-unavailable suffix inside a successful summary).
        self.assertIn("Catalog updated", out)
        cat = self._read_catalog()
        self.assertEqual(cat["device_count"], 5)
        self.assertNotIn("gone plug", [r["name"] for r in cat["devices"]])


# ── wipe guard end-to-end ───────────────────────────────────────────────────
class WipeGuardVoiceTests(_DiscoverTestBase):
    def test_all_sources_empty_keeps_existing_catalog(self):
        self._write_catalog({"device_count": 2, "devices": [
            {"alexa_entity_id": "e1", "name": "Office Lamp"},
            {"alexa_entity_id": "e2", "name": "Hall Light"}]})
        before = open(d._CATALOG_PATH, "rb").read()
        mods = {name: _fake_source(name, []) for name in
                ("skill_sh_kasa", "skill_sh_tuya", "skill_sh_govee",
                 "skill_sh_hue")}
        with mock.patch.object(d, "_alexapy", return_value=None), \
             mock.patch.object(d, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(d, "_say"), \
             _inject_modules(**mods):
            out = d.smart_home_discover("")
        self.assertIn("kept your existing catalog", out)
        self.assertEqual(open(d._CATALOG_PATH, "rb").read(), before)


if __name__ == "__main__":
    unittest.main()
