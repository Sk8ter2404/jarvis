"""Coverage-focused unit tests for ``core.smart_home_router``.

The router reads a device catalog (``data/smart_home_devices.json``), maps each
device's brand to a ``skills/sh_<brand>.py`` controller skill, parses a freeform
utterance into a structured action, resolves the target device(s), and
dispatches via the brand skill — with an Alexa-cookie fallback on any failure.

``tests/skills/test_smart_home_router_skill.py`` already covers the pure
utterance-parsing / value-extraction / device-matching surface plus a few
end-to-end paths with ``_dispatch_one`` stubbed. THIS file deliberately targets
the parts that file leaves cold so the module clears the coverage floor:

  * the catalog loader + TTL cache (``_load_catalog_raw`` / ``_ensure_catalog``),
  * brand→skill discovery (``_controller_for`` / ``_present_brands`` /
    ``_present_skills``) and the lazy, cached ``_import_skill``,
  * the missing-brand TODO logger (``_log_missing_brand``) writing to a temp
    ``jarvis_todo.md`` (idempotent within session AND against file contents),
  * per-device dispatch (``_call_skill`` / ``_dispatch_one``) incl. direct OK,
    direct-error→Alexa, both-fail, and the no-skill→Alexa branch,
  * the whole Alexa fallback (``_alexa_login`` / ``_alexa_set_state``) with the
    ``smart_home_discover`` helper + ``alexapy`` faked entirely,
  * fan-out to a whole room (plural ``_resolve_devices``) and the multi-device
    summary,
  * caching/refresh + the speakable status/list actions + ``warm_up`` +
    ``register`` (incl. its swallow-on-failure path).

ISOLATION CONTRACT (CI breaks otherwise):
  * No real network / devices / threads / sleep, no personal data in fixtures.
  * ``_CATALOG_PATH`` / ``_TODO_PATH`` are redirected to a per-test temp dir so
    the real catalog and todo file are never read or written.
  * Every fake brand skill / ``alexapy`` / ``smart_home_discover`` is installed
    into ``sys.modules`` ONLY for the duration of a test via a save/restore
    context manager (``_inject_modules``) that also patches the parent-package
    attribute (so ``from skills import smart_home_discover`` can't resolve the
    leaf off the real ``skills`` package and leak), then restores absence.
  * Module ``_state`` (catalog/cache/modules/missing-set/login) is reset in
    ``setUp`` AND ``tearDown`` so no run pollutes the next or the wider suite.
  * The ~14K-line ``bobert_companion`` monolith is never imported.

Stdlib ``unittest`` + ``unittest.mock`` only.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from core import smart_home_router as router


# ─── fake-module injection (save/restore, parent-attr aware) ────────────────
_SENTINEL = object()


@contextlib.contextmanager
def _inject_modules(**mods):
    """Temporarily install fake modules into ``sys.modules`` for the with-block.

    For dotted names (``skills.smart_home_discover``) the leaf is ALSO set as an
    attribute on its already-imported parent package, because
    ``from skills import smart_home_discover`` / ``import skills.x`` resolve the
    leaf via ``getattr(parent, leaf)`` once the parent package is real and
    loaded — a bare ``sys.modules`` write would be bypassed and leak. Passing
    ``name=None`` forces the name ABSENT for the block. Everything (including
    prior absence) is restored on exit so tests stay isolated.
    """
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            if name in missing:
                sys.modules.pop(name, None)
            elif saved_mod[name] is not _SENTINEL:
                sys.modules[name] = saved_mod[name]


@contextlib.contextmanager
def _block_import(*names):
    """Force ``import <name>`` / ``from <pkg> import <name>`` to raise
    ImportError inside the with-block, even for an already-imported dotted
    target. Patching ``__import__`` alone is not enough: a ``from skills import
    smart_home_discover`` is satisfied straight from ``sys.modules`` / as an
    attribute of the already-loaded ``skills`` namespace package, so the blocked
    ``__import__`` is never consulted. So we ALSO detach the name from
    ``sys.modules`` and drop the leaf attribute from its parent package for the
    duration, then restore both. This is what stops the REAL
    ``smart_home_discover`` (which constructs an Alexa login with personal data
    on call) from loading when we mean to exercise the import-failure branch.
    """
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Direct `import a.b.c` arrives as name="a.b.c"; a `from a.b import c`
        # arrives as name="a.b", fromlist=("c",) — so the fully-qualified target
        # is name + "." + each fromlist item. Block on either form, plus the top
        # package, so a namespace-subpackage `from`-import can't slip through.
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        for item in (fromlist or ()):
            if f"{name}.{item}" in blocked:
                raise ImportError(f"blocked: {name}.{item}")
        return real_import(name, globals, locals, fromlist, level)

    saved_mod: dict[str, object] = {}
    saved_attr: list = []
    for name in blocked:
        if name in sys.modules:
            saved_mod[name] = sys.modules.pop(name)
        if "." in name:
            parent_name, _, leaf = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, leaf):
                saved_attr.append((parent, leaf, getattr(parent, leaf)))
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            setattr(parent, leaf, prev)
        for name, mod in saved_mod.items():
            sys.modules[name] = mod


def _fake_brand_skill(name="sh_fake", *, set_state=None, is_available=None,
                      has_set_state=True):
    """Build a fake ``skills.sh_<brand>`` module exposing the uniform brand
    interface the router calls (``set_state`` + optional ``is_available``)."""
    mod = types.ModuleType(name)
    if has_set_state:
        mod.set_state = set_state if set_state is not None else (
            lambda device, **kw: {"ok": True, "applied": kw})
    if is_available is not None:
        mod.is_available = is_available
    return mod


# A synthetic catalog. Generic device names / rooms only — no personal data.
def _catalog():
    return {
        "device_count": 4,
        "echo_count": 1,
        "group_count": 2,
        "devices": [
            {"name": "Office Lamp", "brand": "LIFX", "type": "light",
             "alexa_room": "Office", "alexa_groups": ["Work"],
             "controller_skill": "sh_lifx",
             "alexa_entity_id": "ent-office",
             "capabilities": ["on_off", "dim", "color"]},
            {"name": "Kitchen Ceiling", "brand": "Philips Hue", "type": "light",
             "alexa_room": "Kitchen", "alexa_groups": [],
             "alexa_entity_id": "ent-kitchen-1",
             "capabilities": ["on_off", "dim"]},
            {"name": "Kitchen Counter", "brand": "Philips Hue", "type": "light",
             "alexa_room": "Kitchen", "alexa_groups": [],
             "alexa_entity_id": "ent-kitchen-2",
             "capabilities": ["on_off", "dim"]},
            {"name": "Hall Thermostat", "brand": "Nest", "type": "thermostat",
             "alexa_room": "Hall", "alexa_groups": [],
             "controller_skill": "sh_nest",
             "alexa_entity_id": "ent-hall",
             "capabilities": ["thermostat"]},
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# Base: redirect catalog/todo paths to a temp dir and reset module state.
# ════════════════════════════════════════════════════════════════════════════
class _RouterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="shrouter_test_")
        self.addCleanup(self._cleanup_tmp)
        self.catalog_path = os.path.join(self.tmp, "smart_home_devices.json")
        self.todo_path = os.path.join(self.tmp, "jarvis_todo.md")

        # Redirect the module-level paths so nothing touches the real files.
        self._saved_catalog_path = router._CATALOG_PATH
        self._saved_todo_path = router._TODO_PATH
        router._CATALOG_PATH = self.catalog_path
        router._TODO_PATH = self.todo_path

        self._reset_state()
        self.addCleanup(self._restore)

    # ----- helpers -----
    def _reset_state(self):
        with router._state_lock:
            router._state["catalog"] = None
            router._state["loaded_at"] = 0.0
            router._state["modules"] = {}
            router._state["missing_logged"] = set()
            router._state["alexapy_login"] = None

    def _restore(self):
        router._CATALOG_PATH = self._saved_catalog_path
        router._TODO_PATH = self._saved_todo_path
        self._reset_state()

    def _cleanup_tmp(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _write_catalog(self, obj):
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(obj, f)


# ════════════════════════════════════════════════════════════════════════════
# Catalog loader + TTL cache
# ════════════════════════════════════════════════════════════════════════════
class CatalogLoaderTests(_RouterTestBase):
    def test_load_raw_missing_file_returns_none(self):
        # Path points at a not-yet-created temp file.
        self.assertIsNone(router._load_catalog_raw())

    def test_load_raw_parses_json(self):
        self._write_catalog({"device_count": 2, "devices": []})
        got = router._load_catalog_raw()
        self.assertEqual(got["device_count"], 2)

    def test_load_raw_corrupt_json_returns_none(self):
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertIsNone(router._load_catalog_raw())

    def test_ensure_loads_then_caches_within_ttl(self):
        self._write_catalog(_catalog())
        first = router._ensure_catalog()
        self.assertEqual(first["device_count"], 4)
        # Mutate the file; a second call inside the TTL must serve the cache,
        # i.e. NOT re-read disk.
        self._write_catalog({"device_count": 99, "devices": []})
        second = router._ensure_catalog()
        self.assertEqual(second["device_count"], 4)
        self.assertIs(second, first)

    def test_ensure_reloads_when_stale(self):
        self._write_catalog(_catalog())
        router._ensure_catalog()
        # Age the cache past the TTL → next call re-reads.
        with router._state_lock:
            router._state["loaded_at"] = time.monotonic() - (
                router._CATALOG_TTL_SECS + 10)
        self._write_catalog({"device_count": 7, "devices": []})
        got = router._ensure_catalog()
        self.assertEqual(got["device_count"], 7)

    def test_ensure_force_bypasses_cache(self):
        self._write_catalog(_catalog())
        router._ensure_catalog()
        self._write_catalog({"device_count": 5, "devices": []})
        got = router._ensure_catalog(force=True)
        self.assertEqual(got["device_count"], 5)

    def test_ensure_missing_file_returns_none_and_clears(self):
        # Prime a cached catalog, then make the file vanish + force a reload.
        self._write_catalog(_catalog())
        router._ensure_catalog()
        os.unlink(self.catalog_path)
        self.assertIsNone(router._ensure_catalog(force=True))
        with router._state_lock:
            self.assertIsNone(router._state["catalog"])


# ════════════════════════════════════════════════════════════════════════════
# Brand → skill discovery + lazy import cache
# ════════════════════════════════════════════════════════════════════════════
class BrandDiscoveryTests(_RouterTestBase):
    def test_controller_for_aliases_and_substring(self):
        self.assertEqual(router._controller_for("Philips Hue"), "sh_hue")
        self.assertEqual(router._controller_for("signify"), "sh_hue")
        self.assertEqual(router._controller_for("TP-Link Kasa"), "sh_kasa")
        self.assertEqual(router._controller_for("tapo"), "sh_kasa")
        self.assertEqual(router._controller_for("Govee Strip"), "sh_govee")
        self.assertEqual(router._controller_for("google nest"), "sh_nest")

    def test_controller_for_empty_and_unknown(self):
        self.assertIsNone(router._controller_for(""))
        self.assertIsNone(router._controller_for("Wyze"))

    def test_present_brands_dedupes_and_trims(self):
        cat = {"devices": [
            {"brand": "LIFX"}, {"brand": "LIFX"},
            {"brand": "  Philips Hue  "}, {"brand": ""}, {"no_brand": 1},
        ]}
        self.assertEqual(router._present_brands(cat), {"LIFX", "Philips Hue"})

    def test_present_brands_handles_none_devices(self):
        self.assertEqual(router._present_brands({"devices": None}), set())

    def test_present_skills_groups_by_controller(self):
        grouped = router._present_skills(_catalog())
        # explicit controller_skill on LIFX/Nest, inferred sh_hue for the 2 Hue.
        self.assertEqual(len(grouped["sh_hue"]), 2)
        self.assertEqual(len(grouped["sh_lifx"]), 1)
        self.assertEqual(len(grouped["sh_nest"]), 1)

    def test_present_skills_skips_unmappable_brand(self):
        cat = {"devices": [{"name": "X", "brand": "Wyze"}]}
        self.assertEqual(router._present_skills(cat), {})

    def test_import_skill_caches_module(self):
        fake = _fake_brand_skill("skills.sh_lifx")
        with mock.patch.object(router.importlib, "import_module",
                               return_value=fake) as imp:
            first = router._import_skill("sh_lifx")
            second = router._import_skill("sh_lifx")
        self.assertIs(first, fake)
        self.assertIs(second, fake)
        imp.assert_called_once_with("skills.sh_lifx")  # cached → one import

    def test_import_skill_failure_returns_none(self):
        with mock.patch.object(router.importlib, "import_module",
                               side_effect=ImportError("missing optional dep")):
            self.assertIsNone(router._import_skill("sh_govee"))
        # A failed import is NOT cached (so a later install can succeed).
        with router._state_lock:
            self.assertNotIn("sh_govee", router._state["modules"])


# ════════════════════════════════════════════════════════════════════════════
# Missing-brand TODO logger
# ════════════════════════════════════════════════════════════════════════════
class LogMissingBrandTests(_RouterTestBase):
    def test_empty_brand_noop(self):
        with open(self.todo_path, "w", encoding="utf-8") as f:
            f.write("# todo\n")
        router._log_missing_brand("")
        with open(self.todo_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "# todo\n")  # untouched

    def test_appends_task_for_unknown_brand(self):
        with open(self.todo_path, "w", encoding="utf-8") as f:
            f.write("# todo\n")
        router._log_missing_brand("Wyze")
        with open(self.todo_path, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("Build controller skill for brand 'Wyze'", body)
        self.assertIn("skills/sh_wyze.py", body)   # slugified brand
        # Recorded in the session set so it won't re-log.
        with router._state_lock:
            self.assertIn("wyze", router._state["missing_logged"])

    def test_idempotent_within_session(self):
        with open(self.todo_path, "w", encoding="utf-8") as f:
            f.write("# todo\n")
        router._log_missing_brand("Wyze")
        # Second call (same session) returns before re-reading/writing.
        with mock.patch("builtins.open",
                        side_effect=AssertionError("should not reopen")):
            router._log_missing_brand("Wyze")  # must not touch the file

    def test_idempotent_against_existing_file_marker(self):
        # Marker already present on disk but NOT in the session set → must not
        # append a duplicate.
        marker = "[sh-router] Build controller skill for brand 'Wyze'"
        with open(self.todo_path, "w", encoding="utf-8") as f:
            f.write(f"# todo\n- [ ] {marker}. blah\n")
        router._log_missing_brand("Wyze")
        with open(self.todo_path, encoding="utf-8") as f:
            self.assertEqual(f.read().count(marker), 1)

    def test_no_todo_file_skips_write(self):
        # _TODO_PATH points at a non-existent file → after the print it returns
        # without creating it.
        self.assertFalse(os.path.exists(self.todo_path))
        router._log_missing_brand("Govee")
        self.assertFalse(os.path.exists(self.todo_path))
        # Still recorded in-session so the print fires only once.
        with router._state_lock:
            self.assertIn("govee", router._state["missing_logged"])

    def test_blank_slug_falls_back_to_unknown(self):
        # A brand of only punctuation slugs to "" → code substitutes "unknown".
        with open(self.todo_path, "w", encoding="utf-8") as f:
            f.write("# todo\n")
        router._log_missing_brand("!!!")
        with open(self.todo_path, encoding="utf-8") as f:
            self.assertIn("skills/sh_unknown.py", f.read())


# ════════════════════════════════════════════════════════════════════════════
# Extra utterance-classification branches not covered by the skill test
# ════════════════════════════════════════════════════════════════════════════
class ClassifyExtraTests(_RouterTestBase):
    def test_blank_utterance_returns_no_verb(self):
        a = router._classify_action("")
        self.assertIsNone(a["verb"])
        self.assertEqual(a["descriptor"], "")

    def test_unlock_verb(self):
        a = router._classify_action("unlock the back door")
        self.assertEqual(a["verb"], "unlock")
        self.assertEqual(a["descriptor"], "the back door")

    def test_scene_verb(self):
        a = router._classify_action("activate movie scene")
        self.assertEqual(a["verb"], "scene")
        self.assertEqual(a["descriptor"], "movie")

    def test_switch_on_form(self):
        a = router._classify_action("switch on the office lamp")
        self.assertEqual(a["verb"], "on")
        self.assertEqual(a["descriptor"], "the office lamp")

    def test_compound_spoken_number_sets_full_value(self):
        # 2026-07-07 MED bug: 'sixty five' must resolve to 65, not 60.
        a = router._classify_action("set the bedroom to sixty five")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["temperature"], 65)   # 40..110 → temperature
        b = router._classify_action("set the lamp to seventy two")
        self.assertEqual(b["temperature"], 72)
        # brightness range (<40) compound
        c = router._classify_action("set the lamp to thirty five")
        self.assertEqual(c["brightness"], 35)

    def test_spoken_number_helper(self):
        self.assertEqual(router._parse_spoken_number("sixty five"), 65)
        self.assertEqual(router._parse_spoken_number("one hundred"), 100)
        self.assertEqual(router._parse_spoken_number("hundred"), 100)
        self.assertEqual(router._parse_spoken_number("seventy"), 70)
        self.assertEqual(router._parse_spoken_number("72"), 72)
        self.assertIsNone(router._parse_spoken_number(""))
        self.assertIsNone(router._parse_spoken_number("warm"))

    def test_status_question_is_a_query_not_a_command(self):
        # 2026-07-07 HIGH bug: "are the lights on" must NOT turn the lights on.
        for utt, dev in (("are the lights on", "lights"),
                         ("is the office light on", "office light"),
                         ("are the lights on?", "lights"),
                         ("is the front door locked", "front door"),
                         ("what's the bedroom light", "bedroom light")):
            a = router._classify_action(utt)
            self.assertEqual(a["verb"], "query", utt)
            self.assertNotIn(a["verb"], ("on", "off", "lock"), utt)
            self.assertEqual(a["descriptor"], dev, utt)

    def test_plain_command_still_parses_after_query_guard(self):
        # The guard must not swallow real commands that happen to contain 'on'.
        a = router._classify_action("turn on the office lamp")
        self.assertEqual(a["verb"], "on")
        b = router._classify_action("office lamp on")
        self.assertEqual(b["verb"], "on")

    def test_dim_carries_color_temperature(self):
        a = router._classify_action("dim the office to 40% warm 2700K")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["brightness"], 40)
        self.assertEqual(a["color_temperature"], 2700)
        self.assertEqual(a["descriptor"], "the office")

    def test_dim_carries_named_color(self):
        a = router._classify_action("dim the office to 40% red")
        self.assertEqual(a["color"][0], "red")
        self.assertEqual(a["brightness"], 40)

    def test_brighten_defaults_to_full(self):
        a = router._classify_action("brighten the kitchen")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["brightness"], 100)

    def test_set_to_bare_brightness_number(self):
        # Bare number <= 100 that's below the thermostat band → brightness.
        a = router._classify_action("set the office to 20")
        self.assertEqual(a["brightness"], 20)

    def test_set_to_bare_temperature_number(self):
        a = router._classify_action("set the hall to 70")
        self.assertEqual(a["temperature"], 70)

    def test_set_unrecognised_value_keeps_verb_only(self):
        # 'set X to <gibberish>' parses the verb but no value sticks.
        a = router._classify_action("set the office to sparkle")
        self.assertEqual(a["verb"], "set")
        self.assertNotIn("temperature", a)
        self.assertNotIn("brightness", a)
        self.assertNotIn("color", a)

    def test_make_color(self):
        a = router._classify_action("make the office blue")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["color"][0], "blue")
        # Only the color word is stripped from the descriptor (the leading
        # article is not), so 'the office blue' -> 'the office'.
        self.assertEqual(a["descriptor"], "the office")

    def test_make_percent(self):
        a = router._classify_action("make the office 25%")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["brightness"], 25)

    def test_make_without_value_has_no_verb(self):
        # 'make the bed' isn't a smart-home command (no color / percent).
        a = router._classify_action("make the bed")
        self.assertIsNone(a["verb"])

    def test_trailing_off_form(self):
        a = router._classify_action("kitchen light off")
        self.assertEqual(a["verb"], "off")
        self.assertEqual(a["descriptor"], "kitchen light")

    def test_set_to_percent_symbol(self):
        a = router._classify_action("set the office to 55%")
        self.assertEqual(a["verb"], "set")
        self.assertEqual(a["brightness"], 55)
        self.assertEqual(a["descriptor"], "the office")

    def test_set_to_number_word_temperature(self):
        # 'seventy' isn't caught by the digit-only _extract_temperature regex,
        # so it falls through to the bare number-word branch → temperature.
        a = router._classify_action("set the hall to seventy")
        self.assertEqual(a["temperature"], 70)


class ExtractTemperatureExtraTests(_RouterTestBase):
    def test_degrees_without_to_prefix(self):
        self.assertEqual(router._extract_temperature("it is 72 degrees"), 72)

    def test_out_of_band_degrees_ignored(self):
        self.assertIsNone(router._extract_temperature("120 degrees"))

    def test_no_temperature(self):
        self.assertIsNone(router._extract_temperature("hello there"))


# ════════════════════════════════════════════════════════════════════════════
# Device matching: plural / whole-room fan-out
# ════════════════════════════════════════════════════════════════════════════
class ResolveDevicesTests(_RouterTestBase):
    def test_no_devices_in_catalog(self):
        self.assertEqual(router._resolve_devices("office", {"devices": []}), [])

    def test_no_match_above_threshold(self):
        self.assertEqual(router._resolve_devices("garage door", _catalog()), [])

    def test_singular_returns_one(self):
        got = router._resolve_devices("office lamp", _catalog())
        self.assertEqual([d["name"] for d in got], ["Office Lamp"])

    def test_plural_fans_out_whole_room(self):
        # 'the kitchen lights' → both kitchen lights (same room + type).
        got = router._resolve_devices("the kitchen lights", _catalog())
        self.assertEqual({d["name"] for d in got},
                         {"Kitchen Ceiling", "Kitchen Counter"})

    def test_want_type_filters_to_thermostat(self):
        got = router._resolve_devices("hall", _catalog(), want_type="thermostat")
        self.assertEqual([d["name"] for d in got], ["Hall Thermostat"])

    def test_want_type_with_no_typed_match_falls_back(self):
        # No light called 'hall' → the want_type filter finds nothing and the
        # code keeps the original (thermostat) best match rather than erroring.
        got = router._resolve_devices("hall", _catalog(), want_type="light")
        self.assertEqual([d["name"] for d in got], ["Hall Thermostat"])


# ════════════════════════════════════════════════════════════════════════════
# _call_skill — brand-skill invocation contract
# ════════════════════════════════════════════════════════════════════════════
class CallSkillTests(_RouterTestBase):
    def test_skill_not_loadable(self):
        with mock.patch.object(router, "_import_skill", return_value=None):
            out = router._call_skill("sh_lifx", {"name": "X"}, on=True)
        self.assertIn("not loadable", out["error"])

    def test_skill_missing_set_state(self):
        mod = _fake_brand_skill("skills.sh_lifx", has_set_state=False)
        with mock.patch.object(router, "_import_skill", return_value=mod):
            out = router._call_skill("sh_lifx", {"name": "X"}, on=True)
        self.assertIn("no set_state", out["error"])

    def test_skill_set_state_raises(self):
        def boom(device, **kw):
            raise RuntimeError("bridge offline")
        mod = _fake_brand_skill("skills.sh_lifx", set_state=boom)
        with mock.patch.object(router, "_import_skill", return_value=mod):
            out = router._call_skill("sh_lifx", {"name": "X"}, on=True)
        self.assertIn("bridge offline", out["error"])
        self.assertIn("sh_lifx.set_state raised", out["error"])

    def test_skill_returns_non_dict_wrapped(self):
        mod = _fake_brand_skill("skills.sh_lifx",
                                set_state=lambda device, **kw: "done")
        with mock.patch.object(router, "_import_skill", return_value=mod):
            out = router._call_skill("sh_lifx", {"name": "X"}, on=True)
        self.assertTrue(out["ok"])
        self.assertEqual(out["raw"], "done")

    def test_skill_returns_dict_passthrough(self):
        mod = _fake_brand_skill(
            "skills.sh_lifx",
            set_state=lambda device, **kw: {"ok": True, "echo": kw})
        with mock.patch.object(router, "_import_skill", return_value=mod):
            out = router._call_skill("sh_lifx", {"name": "X"}, on=False, brightness=10)
        self.assertEqual(out["echo"], {"on": False, "brightness": 10})


# ════════════════════════════════════════════════════════════════════════════
# Alexa fallback: _alexa_login
# ════════════════════════════════════════════════════════════════════════════
class AlexaLoginTests(_RouterTestBase):
    def test_cached_login_returned(self):
        sentinel = object()
        with router._state_lock:
            router._state["alexapy_login"] = sentinel
        self.assertIs(router._alexa_login(), sentinel)

    def test_cached_failure_returns_none(self):
        with router._state_lock:
            router._state["alexapy_login"] = "_failed"
        self.assertIsNone(router._alexa_login())

    def test_discover_import_unavailable_caches_failed(self):
        # Force `from skills import smart_home_discover` to raise so the
        # fallback-unavailable branch runs. _block_import detaches the real
        # module too, so the personal-data-bearing real discover never loads.
        with _block_import("skills.smart_home_discover"):
            out = router._alexa_login()
        self.assertIsNone(out)
        with router._state_lock:
            self.assertEqual(router._state["alexapy_login"], "_failed")

    def test_restore_login_success_is_cached(self):
        login_obj = object()
        disc = types.ModuleType("skills.smart_home_discover")
        disc._restore_login_from_cookie = lambda: login_obj
        with _inject_modules(**{"skills.smart_home_discover": disc}):
            out = router._alexa_login()
        self.assertIs(out, login_obj)
        with router._state_lock:
            self.assertIs(router._state["alexapy_login"], login_obj)

    def test_restore_login_raises_caches_failed(self):
        def boom():
            raise RuntimeError("cookie expired")
        disc = types.ModuleType("skills.smart_home_discover")
        disc._restore_login_from_cookie = boom
        with _inject_modules(**{"skills.smart_home_discover": disc}):
            out = router._alexa_login()
        self.assertIsNone(out)
        with router._state_lock:
            self.assertEqual(router._state["alexapy_login"], "_failed")

    def test_restore_login_returns_none_caches_failed(self):
        disc = types.ModuleType("skills.smart_home_discover")
        disc._restore_login_from_cookie = lambda: None
        with _inject_modules(**{"skills.smart_home_discover": disc}):
            self.assertIsNone(router._alexa_login())
        with router._state_lock:
            self.assertEqual(router._state["alexapy_login"], "_failed")


# ════════════════════════════════════════════════════════════════════════════
# Alexa fallback: _alexa_set_state
# ════════════════════════════════════════════════════════════════════════════
class AlexaSetStateTests(_RouterTestBase):
    def _device(self, entity="ent-1"):
        d = {"name": "Office Lamp"}
        if entity is not None:
            d["alexa_entity_id"] = entity
        return d

    def test_no_entity_id(self):
        out = router._alexa_set_state({"name": "X"}, {"on": True})
        self.assertIn("no alexa entity id", out["error"])

    def test_login_unavailable(self):
        with mock.patch.object(router, "_alexa_login", return_value=None):
            out = router._alexa_set_state(self._device(), {"on": True})
        self.assertIn("alexa fallback unavailable", out["error"])

    def test_alexapy_not_importable(self):
        # Source does a plain `import alexapy` (not importlib), so block it at
        # the __import__ level to force the not-importable branch regardless of
        # whether alexapy happens to be installed.
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _block_import("alexapy"):
            out = router._alexa_set_state(self._device(), {"on": True})
        self.assertIn("alexapy not importable", out["error"])

    def test_alexaapi_missing_attr(self):
        alexapy = types.ModuleType("alexapy")  # no AlexaAPI attribute
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy):
            out = router._alexa_set_state(self._device(), {"on": True})
        self.assertIn("AlexaAPI missing", out["error"])

    def test_runner_unavailable(self):
        # alexapy + AlexaAPI present, but importing the async runner fails.
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(
            set_appliance_state=lambda *a, **k: None)
        disc = types.ModuleType("skills.smart_home_discover")
        # Deliberately omit `_run_async` so `from ... import _run_async` raises.
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device(), {"on": True})
        self.assertIn("asyncio runner unavailable", out["error"])

    def test_happy_on(self):
        calls = {}

        async def _set_appliance_state(login, entity, target):
            calls["args"] = (login, entity, target)

        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(
            set_appliance_state=_set_appliance_state)
        disc = types.ModuleType("skills.smart_home_discover")
        # A faithful, synchronous stand-in for the discover skill's asyncio
        # runner: run the coroutine to completion and return its result.
        disc._run_async = lambda coro: _drain(coro)
        login = object()
        with mock.patch.object(router, "_alexa_login", return_value=login), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device("ent-1"),
                                          {"on": True})
        self.assertTrue(out["ok"])
        self.assertEqual(out["set"], "ON")
        self.assertEqual(out["path"], "alexa")
        self.assertEqual(calls["args"], (login, "ent-1", "ON"))

    def test_brightness_only_maps_to_on(self):
        async def _set_appliance_state(login, entity, target):
            return None
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(
            set_appliance_state=_set_appliance_state)
        disc = types.ModuleType("skills.smart_home_discover")
        disc._run_async = lambda coro: _drain(coro)
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device(), {"brightness": 50})
        self.assertEqual(out["set"], "ON")

    def test_off_target(self):
        async def _set_appliance_state(login, entity, target):
            return None
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(
            set_appliance_state=_set_appliance_state)
        disc = types.ModuleType("skills.smart_home_discover")
        disc._run_async = lambda coro: _drain(coro)
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device(), {"on": False})
        self.assertEqual(out["set"], "OFF")

    def test_no_compatible_call(self):
        # color-only request → target stays None → "no compatible call".
        async def _set_appliance_state(login, entity, target):
            return None
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(
            set_appliance_state=_set_appliance_state)
        disc = types.ModuleType("skills.smart_home_discover")
        disc._run_async = lambda coro: _drain(coro)
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device(),
                                          {"color": (1, 2, 3)})
        self.assertIn("no compatible alexapy call", out["error"])

    def test_no_set_appliance_state_method(self):
        # AlexaAPI present but lacks set_appliance_state → inner branch refuses.
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace()  # no set_appliance_state
        disc = types.ModuleType("skills.smart_home_discover")
        disc._run_async = lambda coro: _drain(coro)
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device(), {"on": True})
        self.assertIn("no compatible alexapy call", out["error"])

    def test_alexa_call_raises_inside_coro(self):
        async def _set_appliance_state(login, entity, target):
            raise RuntimeError("graph 500")
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(
            set_appliance_state=_set_appliance_state)
        disc = types.ModuleType("skills.smart_home_discover")
        disc._run_async = lambda coro: _drain(coro)
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device(), {"on": True})
        self.assertIn("alexa call failed", out["error"])

    def test_runner_itself_raises(self):
        async def _set_appliance_state(login, entity, target):
            return None
        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(
            set_appliance_state=_set_appliance_state)

        def _boom_runner(coro):
            coro.close()   # avoid 'coroutine never awaited' warning
            raise RuntimeError("loop already running")
        disc = types.ModuleType("skills.smart_home_discover")
        disc._run_async = _boom_runner
        with mock.patch.object(router, "_alexa_login", return_value=object()), \
             _inject_modules(alexapy=alexapy,
                             **{"skills.smart_home_discover": disc}):
            out = router._alexa_set_state(self._device(), {"on": True})
        self.assertIn("alexa fallback runner failed", out["error"])


def _drain(coro):
    """Synchronously run a coroutine to completion and return its value.

    Deterministic stand-in for the discover skill's real ``_run_async`` event-
    loop runner — drives the coroutine with no real loop, threads, or I/O.
    """
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("coroutine did not complete synchronously")


# ════════════════════════════════════════════════════════════════════════════
# _dispatch_one — direct / fallback / failure matrix
# ════════════════════════════════════════════════════════════════════════════
class DispatchOneTests(_RouterTestBase):
    def _dev(self, **over):
        d = {"name": "Office Lamp", "brand": "LIFX",
             "controller_skill": "sh_lifx", "alexa_entity_id": "ent-1"}
        d.update(over)
        return d

    def test_no_recognised_params(self):
        # An action with no verb/values → kwargs empty → "nothing to do".
        out = router._dispatch_one(self._dev(), {"verb": None})
        self.assertIn("nothing to do", out["error"])

    def test_direct_success(self):
        with mock.patch.object(router, "_call_skill",
                               return_value={"ok": True}) as cs:
            out = router._dispatch_one(self._dev(), {"verb": "on"})
        self.assertEqual(out["path"], "direct/sh_lifx")
        self.assertEqual(out["device"], "Office Lamp")
        cs.assert_called_once()

    def test_direct_fail_then_alexa_recovers(self):
        with mock.patch.object(router, "_call_skill",
                               return_value={"error": "lan timeout"}), \
             mock.patch.object(router, "_alexa_set_state",
                               return_value={"ok": True, "set": "ON"}):
            out = router._dispatch_one(self._dev(), {"verb": "on"})
        self.assertEqual(out["path"], "alexa-after-sh_lifx-fail")
        self.assertEqual(out["direct_error"], "lan timeout")
        self.assertEqual(out["device"], "Office Lamp")

    def test_direct_fail_and_alexa_fail(self):
        with mock.patch.object(router, "_call_skill",
                               return_value={"error": "lan timeout"}), \
             mock.patch.object(router, "_alexa_set_state",
                               return_value={"error": "no cookie"}):
            out = router._dispatch_one(self._dev(), {"verb": "on"})
        self.assertEqual(out["path"], "failed/sh_lifx")
        self.assertIn("lan timeout", out["error"])
        self.assertIn("no cookie", out["error"])

    def test_no_skill_uses_alexa_and_logs_missing(self):
        dev = self._dev(brand="Wyze", controller_skill=None)
        with mock.patch.object(router, "_alexa_set_state",
                               return_value={"ok": True}) as ax, \
             mock.patch.object(router, "_log_missing_brand") as logm:
            out = router._dispatch_one(dev, {"verb": "on"})
        logm.assert_called_once_with("Wyze")
        ax.assert_called_once()
        self.assertEqual(out["device"], "Office Lamp")
        self.assertEqual(out["path"], "alexa")

    def test_no_skill_alexa_fails_path_label(self):
        dev = self._dev(brand="Wyze", controller_skill=None)
        with mock.patch.object(router, "_alexa_set_state",
                               return_value={"error": "no cookie"}), \
             mock.patch.object(router, "_log_missing_brand"):
            out = router._dispatch_one(dev, {"verb": "on"})
        self.assertEqual(out["path"], "alexa-failed")
        self.assertEqual(out["device"], "Office Lamp")


# ════════════════════════════════════════════════════════════════════════════
# _summarize_results — every branch of the spoken summary
# ════════════════════════════════════════════════════════════════════════════
class SummarizeResultsTests(_RouterTestBase):
    def test_all_failed_surfaces_first_error(self):
        out = router._summarize_results(
            {"verb": "on", "descriptor": "office"},
            [{"error": "boom", "device": "Office Lamp"}])
        self.assertIn("didn't work", out.lower())
        self.assertIn("boom", out)

    def test_on_single(self):
        out = router._summarize_results(
            {"verb": "on", "descriptor": "office lamp"},
            [{"ok": True, "device": "Office Lamp"}])
        self.assertIn("On, sir", out)
        self.assertIn("Office Lamp", out)

    def test_on_multi(self):
        out = router._summarize_results(
            {"verb": "on", "descriptor": "kitchen"},
            [{"ok": True, "device": "A"}, {"ok": True, "device": "B"}])
        self.assertIn("2 kitchen devices", out)

    def test_off_single_and_multi(self):
        one = router._summarize_results(
            {"verb": "off", "descriptor": "office lamp"},
            [{"ok": True, "device": "Office Lamp"}])
        self.assertIn("Off, sir", one)
        multi = router._summarize_results(
            {"verb": "off", "descriptor": "kitchen"},
            [{"ok": True, "device": "A"}, {"ok": True, "device": "B"}])
        self.assertIn("2 kitchen devices", multi)

    def test_brightness_summary(self):
        out = router._summarize_results(
            {"verb": "set", "brightness": 40, "descriptor": "office"},
            [{"ok": True, "device": "Office Lamp"}])
        self.assertIn("Set to 40%", out)

    def test_temperature_summary(self):
        out = router._summarize_results(
            {"verb": "set", "temperature": 68, "descriptor": "hall"},
            [{"ok": True, "device": "Hall Thermostat"}])
        self.assertIn("Setpoint at 68", out)

    def test_color_summary(self):
        out = router._summarize_results(
            {"verb": "set", "color": ("blue", (0, 60, 255)), "descriptor": "office"},
            [{"ok": True, "device": "Office Lamp"}])
        self.assertIn("Color set to blue", out)

    def test_lock_and_unlock_summary(self):
        locked = router._summarize_results(
            {"verb": "lock", "descriptor": "front door"},
            [{"ok": True, "device": "Front Door"}])
        self.assertIn("Locked, sir", locked)
        unlocked = router._summarize_results(
            {"verb": "unlock", "descriptor": "front door"},
            [{"ok": True, "device": "Front Door"}])
        self.assertIn("Unlocked, sir", unlocked)

    def test_scene_summary(self):
        out = router._summarize_results(
            {"verb": "scene", "descriptor": "movie"},
            [{"ok": True, "device": "Movie"}])
        self.assertIn("Scene running, sir", out)

    def test_default_done_summary(self):
        # A verb the summary has no special-case for falls to the generic line.
        out = router._summarize_results(
            {"verb": "mystery", "descriptor": "thing"},
            [{"ok": True, "device": "Thing"}])
        self.assertIn("Done, sir", out)

    def test_on_with_brightness_is_not_plain_on_branch(self):
        # verb 'on' but brightness present → falls through to the brightness
        # summary, not the "On, sir" branch.
        out = router._summarize_results(
            {"verb": "on", "brightness": 60, "descriptor": "office"},
            [{"ok": True, "device": "Office Lamp"}])
        self.assertIn("Set to 60%", out)


# ════════════════════════════════════════════════════════════════════════════
# smart_home_control — fan-out + want_type selection (real internals)
# ════════════════════════════════════════════════════════════════════════════
class SmartHomeControlIntegrationTests(_RouterTestBase):
    def setUp(self):
        super().setUp()
        self._write_catalog(_catalog())

    def test_fan_out_to_whole_room(self):
        # 'turn off the kitchen lights' → both kitchen lights dispatched.
        seen = []

        def _disp(device, action):
            seen.append(device["name"])
            return {"ok": True, "device": device["name"]}
        with mock.patch.object(router, "_dispatch_one", side_effect=_disp):
            out = router.smart_home_control("turn off the kitchen lights")
        self.assertEqual(set(seen), {"Kitchen Ceiling", "Kitchen Counter"})
        self.assertIn("2 the kitchen lights devices", out)

    def test_color_request_targets_light_type(self):
        captured = {}

        def _disp(device, action):
            captured["dev"] = device["name"]
            return {"ok": True, "device": device["name"]}
        with mock.patch.object(router, "_dispatch_one", side_effect=_disp):
            out = router.smart_home_control("set the office lamp to blue")
        self.assertEqual(captured["dev"], "Office Lamp")
        self.assertIn("Color set to blue", out)

    def test_lock_want_type_prefers_lock_device(self):
        # A lock verb sets want_type='lock'. Catalog has a light AND a lock that
        # both token-match 'front'; the want_type filter must pick the lock.
        self._write_catalog({
            "device_count": 2,
            "devices": [
                {"name": "Front Light", "brand": "LIFX", "type": "light",
                 "alexa_room": "Entry", "controller_skill": "sh_lifx"},
                {"name": "Front Lock", "brand": "Nest", "type": "lock",
                 "alexa_room": "Entry", "controller_skill": "sh_nest"},
            ],
        })
        captured = {}

        def _disp(device, action):
            captured["dev"] = device["name"]
            return {"ok": True, "device": device["name"]}
        with mock.patch.object(router, "_dispatch_one", side_effect=_disp):
            out = router.smart_home_control("lock the front")
        self.assertEqual(captured["dev"], "Front Lock")
        self.assertIn("Locked, sir", out)

    def test_empty_catalog_devices_message(self):
        self._write_catalog({"device_count": 0, "devices": []})
        out = router.smart_home_control("turn off the office lamp")
        self.assertIn("No smart-home catalog", out)

    def test_empty_utterance_prompts(self):
        self.assertIn("something to do",
                      router.smart_home_control("   ").lower())

    def test_unparseable_utterance(self):
        self.assertIn("couldn't parse",
                      router.smart_home_control("what time is it").lower())

    def test_no_device_match_message(self):
        self.assertIn("don't see anything",
                      router.smart_home_control("turn off the garage").lower())

    def test_temperature_routes_to_thermostat(self):
        # want_type='thermostat' selects Hall Thermostat over any lamp.
        captured = {}

        def _disp(device, action):
            captured["dev"] = device["name"]
            return {"ok": True, "device": device["name"]}
        with mock.patch.object(router, "_dispatch_one", side_effect=_disp):
            out = router.smart_home_control("set the hall to 68")
        self.assertEqual(captured["dev"], "Hall Thermostat")
        self.assertIn("Setpoint at 68", out)

    def test_real_dispatch_with_faked_brand_skill(self):
        # End-to-end through the REAL _dispatch_one/_call_skill, with only the
        # brand skill import faked: 'turn off the office lamp' → sh_lifx.set_state.
        calls = []

        def _set_state(device, **kw):
            calls.append((device["name"], kw))
            return {"ok": True, "device": device["name"]}
        fake = _fake_brand_skill("skills.sh_lifx", set_state=_set_state)
        with mock.patch.object(router.importlib, "import_module",
                               return_value=fake):
            out = router.smart_home_control("turn off the office lamp")
        self.assertEqual(calls, [("Office Lamp", {"on": False})])
        self.assertIn("Off, sir", out)


# ════════════════════════════════════════════════════════════════════════════
# Speakable actions: smart_home_devices / status / refresh
# ════════════════════════════════════════════════════════════════════════════
class SpeakableActionTests(_RouterTestBase):
    def test_devices_no_catalog(self):
        # No file on disk → _ensure_catalog returns None.
        self.assertIn("No smart-home catalog",
                      router.smart_home_devices(""))

    def test_devices_empty_catalog(self):
        self._write_catalog({"device_count": 0, "devices": []})
        self.assertIn("catalog is empty", router.smart_home_devices(""))

    def test_devices_grouped_by_room(self):
        self._write_catalog(_catalog())
        out = router.smart_home_devices("")
        self.assertIn("4 devices", out)
        self.assertIn("2 in Kitchen", out)
        self.assertIn("1 in Office", out)

    def test_devices_unassigned_room_bucket(self):
        self._write_catalog({
            "device_count": 1,
            "devices": [{"name": "Floating Bulb", "type": "light"}],  # no room
        })
        self.assertIn("(unassigned)", router.smart_home_devices(""))

    def test_status_not_loaded(self):
        self.assertIn("Catalog not loaded", router.smart_home_router_status(""))

    def test_status_active_dep_missing_and_no_skill(self):
        # Catalog with: an available skill (sh_lifx), a dep-missing skill
        # (sh_hue.is_available -> False), and an unmappable brand (Wyze).
        self._write_catalog({
            "device_count": 4,
            "devices": [
                {"name": "L", "brand": "LIFX", "type": "light",
                 "controller_skill": "sh_lifx"},
                {"name": "H1", "brand": "Philips Hue", "type": "light"},
                {"name": "H2", "brand": "Philips Hue", "type": "light"},
                {"name": "W", "brand": "Wyze", "type": "plug"},
            ],
        })

        def _imp(skill_name):
            if skill_name == "sh_lifx":
                return _fake_brand_skill("skills.sh_lifx",
                                         is_available=lambda: True)
            if skill_name == "sh_hue":
                return _fake_brand_skill("skills.sh_hue",
                                         is_available=lambda: False)
            return None
        with mock.patch.object(router, "_import_skill", side_effect=_imp):
            out = router.smart_home_router_status("")
        self.assertIn("active: sh_lifx(1)", out)
        self.assertIn("dep-missing: sh_hue(2)", out)
        self.assertIn("no-skill: Wyze", out)

    def test_status_skill_without_is_available_counts_active(self):
        # A loaded module lacking is_available() is treated as available.
        self._write_catalog({
            "device_count": 1,
            "devices": [{"name": "L", "brand": "LIFX", "type": "light",
                         "controller_skill": "sh_lifx"}],
        })
        mod = _fake_brand_skill("skills.sh_lifx")  # no is_available
        with mock.patch.object(router, "_import_skill", return_value=mod):
            out = router.smart_home_router_status("")
        self.assertIn("active: sh_lifx(1)", out)

    def test_status_is_available_raises_is_dep_missing(self):
        self._write_catalog({
            "device_count": 1,
            "devices": [{"name": "L", "brand": "LIFX", "type": "light",
                         "controller_skill": "sh_lifx"}],
        })

        def _raise():
            raise RuntimeError("probe blew up")
        mod = _fake_brand_skill("skills.sh_lifx", is_available=_raise)
        with mock.patch.object(router, "_import_skill", return_value=mod):
            out = router.smart_home_router_status("")
        self.assertIn("dep-missing: sh_lifx(1)", out)

    def test_status_unloadable_skill_is_dep_missing(self):
        self._write_catalog({
            "device_count": 1,
            "devices": [{"name": "L", "brand": "LIFX", "type": "light",
                         "controller_skill": "sh_lifx"}],
        })
        with mock.patch.object(router, "_import_skill", return_value=None):
            out = router.smart_home_router_status("")
        self.assertIn("dep-missing: sh_lifx(1)", out)

    def test_status_no_devices_message(self):
        self._write_catalog({"device_count": 0, "devices": []})
        out = router.smart_home_router_status("")
        self.assertIn("no devices, sir", out)

    def test_refresh_no_catalog(self):
        # Nothing on disk → refresh reports no catalog and clears caches.
        out = router.refresh_smart_home_router("")
        self.assertIn("No catalog on disk", out)

    def test_refresh_reports_counts(self):
        self._write_catalog(_catalog())
        # Prime caches with stale junk to prove refresh wipes them.
        with router._state_lock:
            router._state["modules"] = {"stale": object()}
            router._state["catalog"] = {"device_count": 999, "devices": []}
        with mock.patch.object(router, "_import_skill", return_value=None):
            out = router.refresh_smart_home_router("")
        self.assertIn("4 devices", out)
        self.assertIn("3 brand controller", out)  # sh_lifx, sh_hue, sh_nest
        with router._state_lock:
            self.assertEqual(router._state["modules"], {})


# ════════════════════════════════════════════════════════════════════════════
# warm_up + register
# ════════════════════════════════════════════════════════════════════════════
class WarmUpAndRegisterTests(_RouterTestBase):
    def test_warm_up_no_catalog_is_noop(self):
        # No file → _ensure_catalog None → returns without importing anything.
        with mock.patch.object(router, "_import_skill") as imp:
            router.warm_up()
        imp.assert_not_called()

    def test_warm_up_imports_present_skills_and_logs_missing(self):
        self._write_catalog({
            "device_count": 2,
            "devices": [
                {"name": "L", "brand": "LIFX", "type": "light",
                 "controller_skill": "sh_lifx"},
                {"name": "W", "brand": "Wyze", "type": "plug"},   # no skill
            ],
        })
        with mock.patch.object(router, "_import_skill") as imp, \
             mock.patch.object(router, "_log_missing_brand") as logm:
            router.warm_up()
        imp.assert_called_once_with("sh_lifx")
        logm.assert_called_once_with("Wyze")

    def test_register_wires_actions_and_calls_warm_up(self):
        actions: dict = {}
        with mock.patch.object(router, "warm_up") as wu:
            router.register(actions)
        wu.assert_called_once()
        for name in ("smart_home_control", "control_device",
                     "control_smart_home", "smart_home_devices",
                     "smart_home_list", "smart_home_router_status",
                     "refresh_smart_home_router"):
            self.assertIn(name, actions)
        self.assertIs(actions["control_device"], router.smart_home_control)

    def test_register_swallows_warm_up_failure(self):
        # register() wraps warm_up() in try/except so a bad catalog at boot
        # can't break skill registration.
        actions: dict = {}
        with mock.patch.object(router, "warm_up",
                               side_effect=RuntimeError("catalog blew up")):
            router.register(actions)   # must not raise
        self.assertIn("smart_home_control", actions)


# ════════════════════════════════════════════════════════════════════════════
# Remaining pure-branch coverage (extractors, kwargs, match-score guards,
# todo-logger IO error paths)
# ════════════════════════════════════════════════════════════════════════════
class ActionToKwargsExtraTests(_RouterTestBase):
    def test_lock_and_unlock(self):
        self.assertEqual(router._action_to_kwargs({"verb": "lock"}),
                         {"locked": True})
        self.assertEqual(router._action_to_kwargs({"verb": "unlock"}),
                         {"locked": False})

    def test_temperature_and_color_temperature(self):
        kw = router._action_to_kwargs(
            {"verb": "set", "temperature": 68, "color_temperature": 3000})
        self.assertEqual(kw["temperature"], 68)
        self.assertEqual(kw["color_temperature"], 3000)

    def test_color_carries_rgb_and_name(self):
        kw = router._action_to_kwargs(
            {"verb": "set", "color": ("red", (255, 0, 0))})
        self.assertEqual(kw["color"], (255, 0, 0))
        self.assertEqual(kw["color_name"], "red")

    def test_brightness_zero_forces_off(self):
        kw = router._action_to_kwargs({"verb": "set", "brightness": 0})
        self.assertEqual(kw["brightness"], 0)
        self.assertFalse(kw["on"])

    def test_none_brightness_ignored(self):
        kw = router._action_to_kwargs({"verb": "on", "brightness": None})
        self.assertNotIn("brightness", kw)
        self.assertTrue(kw["on"])


class ExtractorEdgeTests(_RouterTestBase):
    def test_strip_filler_recurses_multiple_prefixes(self):
        # 'could you please ' + 'the ' all peel off.
        self.assertEqual(router._strip_filler("Could you please the office"),
                         "office")

    def test_parse_number_strips_units_and_handles_empty(self):
        self.assertEqual(router._parse_number("65°"), 65)
        self.assertEqual(router._parse_number("100%"), 100)
        self.assertIsNone(router._parse_number(""))
        self.assertIsNone(router._parse_number("   "))

    def test_parse_number_superscript_digit_is_none(self):
        # A Unicode superscript digit passes str.isdigit() but int() rejects it
        # (category No, not Nd) → the ValueError branch returns None rather than
        # raising. '²' is superscript two.
        self.assertTrue("²".isdigit())
        self.assertIsNone(router._parse_number("²"))

    def test_extract_percent_word_only_form(self):
        # No digits before 'percent', spelled-out number → word branch.
        self.assertEqual(router._extract_percent("forty percent"), 40)

    def test_extract_percent_word_unparseable(self):
        self.assertIsNone(router._extract_percent("lots percent"))

    def test_extract_percent_none(self):
        self.assertIsNone(router._extract_percent("no number here"))

    def test_extract_temperature_to_prefix_out_of_band(self):
        # 'to 39' is below the 40 floor → ignored even with the to-prefix.
        self.assertIsNone(router._extract_temperature("set to 39"))

    def test_extract_color_temperature_empty_and_nomatch(self):
        self.assertIsNone(router._extract_color_temperature(""))
        self.assertIsNone(router._extract_color_temperature("2700"))  # no K/kelvin

    def test_match_score_empty_descriptor(self):
        self.assertEqual(router._match_score("", {"name": "Office Lamp"}), 0.0)

    def test_match_score_device_with_no_text(self):
        # Descriptor has real tokens but the device has no matchable text.
        self.assertEqual(router._match_score("office lamp", {}), 0.0)


class LogMissingBrandIOErrorTests(_RouterTestBase):
    def test_read_failure_returns_quietly(self):
        # File exists, but open-for-read raises → function returns without
        # appending (and without propagating).
        with open(self.todo_path, "w", encoding="utf-8") as f:
            f.write("# todo\n")
        real_open = open

        def _open(path, *a, **k):
            if os.path.abspath(path) == os.path.abspath(self.todo_path) \
                    and ("r" in (a[0] if a else k.get("mode", "r"))):
                raise OSError("read denied")
            return real_open(path, *a, **k)
        with mock.patch("builtins.open", side_effect=_open):
            router._log_missing_brand("Wyze")   # must not raise
        # Nothing appended (marker absent).
        with open(self.todo_path, encoding="utf-8") as f:
            self.assertNotIn("Build controller skill", f.read())

    def test_append_failure_is_swallowed(self):
        # Read succeeds (marker absent) but the append-open raises → swallowed.
        with open(self.todo_path, "w", encoding="utf-8") as f:
            f.write("# todo\n")
        real_open = open

        def _open(path, *a, **k):
            mode = a[0] if a else k.get("mode", "r")
            if os.path.abspath(path) == os.path.abspath(self.todo_path) \
                    and "a" in mode:
                raise OSError("disk full")
            return real_open(path, *a, **k)
        with mock.patch("builtins.open", side_effect=_open):
            router._log_missing_brand("Wyze")   # must not raise


class QueryReplyNoFailureMarkerTests(unittest.TestCase):
    """2026-07-08: the status-query SUCCESS reply ("are the lights on?") must not
    contain any core.failure_markers.FAILURE_MARKER substring. It is spoken via
    the verbatim path and classified by _is_failure on those substrings, so a
    marker (the old "can't read its live state") suppressed the honest answer AND
    misclassified the query as a failed action (extra LLM round-trip)."""

    def test_query_reply_contains_no_failure_marker(self):
        from core import failure_markers
        cat = {"devices": [{"name": "office lights", "brand": "hue"}]}
        with mock.patch.object(router, "_try_pointing_resolution",
                               return_value=None), \
                mock.patch.object(router, "_ensure_catalog", return_value=cat), \
                mock.patch.object(router, "_classify_action",
                                  return_value={"verb": "query",
                                                "descriptor": "office lights"}), \
                mock.patch.object(router, "_resolve_devices",
                                  return_value=[{"name": "office lights"}]):
            reply = router.smart_home_control("are the office lights on")
        low = reply.lower()
        hits = [m for m in failure_markers.FAILURE_MARKERS if m in low]
        self.assertEqual(hits, [],
                         f"query reply matched failure markers {hits}: {reply!r}")
        self.assertIn("office lights", reply)   # it IS the honest status answer


if __name__ == "__main__":
    unittest.main()
