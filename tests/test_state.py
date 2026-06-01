"""Contract tests for core.state — the runtime single-element-list slots.

These slots are INTENTIONALLY single-element lists (``_x = [v]``, read via
``_x[0]``, written via ``_x[0] = new``). Under CPython's GIL, list
__setitem__ is atomic, so reader threads in consumer skills observe writes
without a lock — and the list IDENTITY must never change so the shared
mutable reference stays shared. These tests pin that contract: the seed
values, the by-index mutation behaviour, the stable identity, and the
``__all__`` wildcard-re-export wiring that bobert_companion relies on.

This is a contract/shape suite by design — the slots have no logic of their
own, but the single-element-list idiom and the seeds-from-config wiring are
load-bearing and worth guarding against an accidental refactor to plain vars.

stdlib unittest only. No mutation of real state survives a test (each slot is
restored in tearDown).
"""
from __future__ import annotations

import threading
import unittest

from core import state
from core import config


# Every list-wrapped slot and its documented seed value.
_BOOL_LIST_SLOTS = {
    "_sleep_mode": False,
    "_standby_mode": False,
    "_tts_muted": False,
    "_ambient_mode_active": False,
    "_daemons_paused": False,
}
_FLOAT_LIST_SLOTS = {
    "_jarvis_played_music_at": 0.0,
    "_ambient_music_last_hit": 0.0,
}
_INT_LIST_SLOTS = {
    "_ambient_music_hits": 0,
}


class SeedValueTests(unittest.TestCase):
    def test_bool_slots_seed_false(self):
        for name, seed in _BOOL_LIST_SLOTS.items():
            slot = getattr(state, name)
            self.assertIsInstance(slot, list)
            self.assertEqual(len(slot), 1)
            self.assertEqual(slot[0], seed, name)

    def test_numeric_slots_seed_zero(self):
        for name, seed in {**_FLOAT_LIST_SLOTS, **_INT_LIST_SLOTS}.items():
            slot = getattr(state, name)
            self.assertEqual(len(slot), 1)
            self.assertEqual(slot[0], seed, name)

    def test_last_wake_date_seeds_none(self):
        self.assertEqual(state._last_wake_date, [None])

    def test_wake_history_is_empty_list(self):
        # Not a single-element slot — a growing history list.
        self.assertEqual(state._wake_history, [])

    def test_shutdown_prompt_pending_shape(self):
        d = state._shutdown_prompt_pending
        self.assertIsInstance(d, dict)
        self.assertEqual(d.get("armed"), False)
        self.assertIn("expires_at", d)

    def test_overnight_run_now_is_event(self):
        self.assertIsInstance(state._overnight_run_now, threading.Event)
        self.assertFalse(state._overnight_run_now.is_set())


class ConfigSeededSlotTests(unittest.TestCase):
    """The audio-master and debug slots seed from core.config so flipping the
    config knob is the canonical default; tray toggles are runtime overrides."""

    def test_audio_master_seeds_from_config(self):
        self.assertEqual(state._audio_master_enabled[0],
                         config.AUDIO_PROCESSING_ENABLED)

    def test_debug_mode_seeds_from_config(self):
        self.assertEqual(state._debug_mode[0], config.VAD_DEBUG)

    def test_audio_sub_toggles_default_true(self):
        self.assertEqual(state._audio_aec_enabled[0], True)
        self.assertEqual(state._audio_ns_enabled[0], True)
        self.assertEqual(state._audio_agc_enabled[0], True)


class MutationContractTests(unittest.TestCase):
    """A by-index write must change the element WITHOUT replacing the list
    object — that stable identity is what lets consumer threads share the ref."""

    def test_index_write_visible_through_same_object(self):
        slot = state._sleep_mode
        original_id = id(slot)
        self.addCleanup(lambda: slot.__setitem__(0, False))
        slot[0] = True
        # A separate alias (as a consumer skill would hold) sees the new value.
        alias = state._sleep_mode
        self.assertIs(alias, slot)
        self.assertEqual(alias[0], True)
        self.assertEqual(id(state._sleep_mode), original_id)

    def test_identity_stable_across_writes(self):
        before = id(state._tts_muted)
        self.addCleanup(lambda: state._tts_muted.__setitem__(0, False))
        state._tts_muted[0] = True
        state._tts_muted[0] = False
        self.assertEqual(id(state._tts_muted), before)


class WildcardExportTests(unittest.TestCase):
    """bobert_companion does ``from core.state import *`` to rebind these
    underscore-prefixed names; __all__ must therefore opt every slot in."""

    def test_all_lists_every_underscore_slot(self):
        # Each slot referenced by the suite must be exported.
        expected = (set(_BOOL_LIST_SLOTS) | set(_FLOAT_LIST_SLOTS)
                    | set(_INT_LIST_SLOTS)
                    | {"_last_wake_date", "_wake_history",
                       "_shutdown_prompt_pending", "_overnight_run_now",
                       "_audio_master_enabled", "_audio_aec_enabled",
                       "_audio_ns_enabled", "_audio_agc_enabled", "_debug_mode"})
        self.assertTrue(expected.issubset(set(state.__all__)),
                        expected - set(state.__all__))

    def test_every_all_name_exists_as_attribute(self):
        for name in state.__all__:
            self.assertTrue(hasattr(state, name), name)

    def test_all_names_are_underscore_prefixed(self):
        # The whole reason __all__ exists is to re-export _-prefixed names.
        for name in state.__all__:
            self.assertTrue(name.startswith("_"), name)

    def test_wildcard_import_rebinds_slots(self):
        # Simulate the parent module's wildcard import into a fresh namespace
        # and confirm the slot objects are the very same list instances.
        ns: dict = {}
        exec("from core.state import *", ns)
        self.assertIs(ns["_sleep_mode"], state._sleep_mode)
        self.assertIs(ns["_debug_mode"], state._debug_mode)


if __name__ == "__main__":
    unittest.main()
