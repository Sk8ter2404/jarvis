"""Monolith side of the 2026-10-01 web / tray / Settings fix batch.

Each class is one finding, driven through the REAL bobert_companion code:

  * B013  Settings values for the audio stages and VAD debug never reached the
          runtime after the first boot - the persisted tray value always won
          (one mechanism with config-wiring B051: toggle_cfg_seed).
  * B014  START_IN_STANDBY was dead from the second boot on (it only applied
          when hud_state.json had no sleep_mode key), and on a fresh install.
  * B016  a tray "Ambient Mode" OFF did not stop the fact-extractor and did not
          survive a restart; the voice toggle could only ever turn it ON; the
          gated-turn learner ignored an explicit OFF (one mechanism with
          config-wiring B052 / B053: the saved AMBIENT_LISTEN_ENABLED).
  * B075  the tray-command drainer (the ONLY reader of tray_commands.json,
          which the dashboard and the Settings window also write) and the hud
          writes were tied to the tray icon being on.
  * B077  a web-queued command re-appended by the drain race ran twice.

No device, window, tray or network is touched: every file lives in a temp dir,
_write_hud_state is captured or pointed at a temp file, and the ambient skills
are stand-in modules.

    python -B -m unittest tests.monolith.test_monolith_web_tray_settings
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
import types
import unittest
from collections import OrderedDict
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.tmp = tempfile.mkdtemp(prefix="web_tray_settings_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.hud_file = os.path.join(self.tmp, "hud_state.json")
        self._p(bc, "HUD_STATE_FILE", self.hud_file)
        self._p(bc, "HUD_ENABLED", True)
        cache = bc._hud_state_cache
        snap = dict(cache)
        self.addCleanup(lambda: (cache.clear(), cache.update(snap)))
        cache.clear()
        # A restore test must not reach the real ambient skill or daemons.
        self._p(bc, "ACTIONS", {})
        os.environ.pop("JARVIS_START_IN_STANDBY", None)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _cfg(self, **values):
        import core.config as cfg
        for name, value in values.items():
            self._p(cfg, name, value)

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()

    def _persist(self, data):
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _saved(self):
        with open(self.hud_file, encoding="utf-8") as f:
            return json.load(f)

    def _restore(self):
        return self._quiet(self.bc._restore_tray_toggle_state)


# ── B013 ───────────────────────────────────────────────────────────────────
# Merged with the config-wiring batch's B051, which fixed the same bug first:
# the record of the Settings value each tray toggle was saved against is
# toggle_cfg_seed, and a key missing from it (a hud_state.json from before the
# fix) keeps the tray's value for that one boot. These tests drive that one
# mechanism.
class SettingsBackedToggleRestoreTests(_Base):
    """A Settings value saved in the window or on the web panel must reach the
    runtime after the restart, not lose to the persisted tray value."""

    CELL_CFG = (("_debug_mode", "VAD_DEBUG"),
                ("_audio_master_enabled", "AUDIO_PROCESSING_ENABLED"),
                ("_audio_aec_enabled", "AUDIO_ECHO_CANCEL"),
                ("_audio_ns_enabled", "AUDIO_NOISE_SUPPRESS"),
                ("_audio_agc_enabled", "AUDIO_AGC"))

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.cells = {}
        for name, _cfg_name in self.CELL_CFG:
            self.cells[name] = [None]
            self._p(bc, name, self.cells[name])
        self._cfg(START_IN_STANDBY=False)

    def _restore(self):
        # core/state.py seeds each cell from its config value at import; the
        # restore only overrides a cell when the tray's value wins.
        import core.config as cfg
        for name, cfg_name in self.CELL_CFG:
            self.cells[name][0] = bool(getattr(cfg, cfg_name))
        return super()._restore()

    def test_a_settings_change_wins_over_the_persisted_tray_value(self):
        # The tray left AEC + VAD debug on while Settings also said on
        # (recorded). The owner then unticked both in Settings and restarted.
        self._persist({"echo_cancel_enabled": True, "debug_mode": True,
                       "toggle_cfg_seed": {"echo_cancel_enabled": True,
                                           "debug_mode": True}})
        self._cfg(AUDIO_ECHO_CANCEL=False, VAD_DEBUG=False)
        self._restore()
        self.assertIs(self.cells["_audio_aec_enabled"][0], False,
                      "AEC unticked in Settings stayed ON after the restart")
        self.assertIs(self.cells["_debug_mode"][0], False,
                      "VAD debug unticked in Settings stayed ON")

    def test_without_a_record_the_tray_value_is_kept_once_and_recorded(self):
        # Every hud_state.json written before the fix: the keys are there,
        # the record is not. The tray value holds for this boot, and the
        # record is written so the next Settings change wins.
        self._persist({"agc_enabled": False, "noise_suppress_enabled": False,
                       "audio_processing_enabled": False})
        self._cfg(AUDIO_AGC=True, AUDIO_NOISE_SUPPRESS=True,
                  AUDIO_PROCESSING_ENABLED=True)
        self._restore()
        self.assertIs(self.cells["_audio_agc_enabled"][0], False)
        seed = self._saved()["toggle_cfg_seed"]
        self.assertIs(seed["agc_enabled"], True)
        # The owner then unticks NS in Settings (the tray had it on) and
        # restarts: from now on the Settings value applies.
        self._cfg(AUDIO_NOISE_SUPPRESS=False)
        self._persist(dict(self._saved(), noise_suppress_enabled=True))
        self._restore()
        self.assertIs(self.cells["_audio_ns_enabled"][0], False)

    def test_a_tray_choice_survives_while_settings_are_unchanged(self):
        self._persist({"noise_suppress_enabled": False, "debug_mode": True,
                       "toggle_cfg_seed": {"noise_suppress_enabled": True,
                                           "debug_mode": False}})
        self._cfg(AUDIO_NOISE_SUPPRESS=True, VAD_DEBUG=False)
        self._restore()
        self.assertIs(self.cells["_audio_ns_enabled"][0], False)
        self.assertIs(self.cells["_debug_mode"][0], True)

    def test_the_restore_records_what_it_booted_against(self):
        self._persist({"agc_enabled": True,
                       "toggle_cfg_seed": {"agc_enabled": True}})
        self._cfg(VAD_DEBUG=True, AUDIO_PROCESSING_ENABLED=True,
                  AUDIO_ECHO_CANCEL=False, AUDIO_NOISE_SUPPRESS=True,
                  AUDIO_AGC=False)
        self._restore()
        seed = self._saved()["toggle_cfg_seed"]
        self.assertEqual(seed, {"debug_mode": True,
                                "audio_processing_enabled": True,
                                "echo_cancel_enabled": False,
                                "noise_suppress_enabled": True,
                                "agc_enabled": False})
        self.assertIs(self._saved()["agc_enabled"], False)

    def test_the_record_survives_an_early_hud_write(self):
        # A HUD write before the restore rewrites the WHOLE file from the cache
        # - the import seed must carry the record, or the tray choice is lost.
        src = inspect.getsource(self.bc)
        seed = src[src.index("Seed the persisted tray-toggle cells"):]
        seed = seed[:seed.index("_last_mic_hud_write")]
        self.assertIn('"toggle_cfg_seed"', seed)


# ── B014 ───────────────────────────────────────────────────────────────────
class StartInStandbyTests(_Base):
    def setUp(self):
        super().setUp()
        self.sleep = [False]
        self.standby = [False]
        self._p(self.bc, "_sleep_mode", self.sleep)
        self._p(self.bc, "_standby_mode", self.standby)

    def test_applies_although_hud_state_already_has_sleep_mode(self):
        # The state of every box after its first boot: sleep_mode is written
        # back by the restore and by the 1 Hz tray publisher.
        self._persist({"sleep_mode": False, "standby_mode": False})
        self._cfg(START_IN_STANDBY=True)
        self._restore()
        self.assertTrue(self.sleep[0], "START_IN_STANDBY ignored - booted "
                                       "always-listening")
        self.assertTrue(self.standby[0])
        self.assertTrue(self._saved()["sleep_mode"])

    def test_applies_on_a_fresh_install(self):
        self.assertFalse(os.path.exists(self.hud_file))
        self._cfg(START_IN_STANDBY=True)
        self._restore()
        self.assertTrue(self.sleep[0])
        self.assertTrue(self.standby[0])
        self.assertTrue(self._saved()["standby_mode"],
                        "the tray must see the standby it booted into")

    def test_applies_after_a_corrupt_state_file(self):
        with open(self.hud_file, "w", encoding="utf-8") as f:
            f.write("{ not json")
        self._cfg(START_IN_STANDBY=True)
        self._restore()
        self.assertTrue(self.sleep[0])

    def test_off_keeps_an_awake_boot_awake(self):
        self._persist({"sleep_mode": False, "standby_mode": False})
        self._cfg(START_IN_STANDBY=False)
        self._restore()
        self.assertFalse(self.sleep[0])
        self.assertFalse(self.standby[0])

    def test_off_still_lets_a_crash_survival_sleep_win(self):
        self._persist({"sleep_mode": True, "standby_mode": True})
        self._cfg(START_IN_STANDBY=False)
        self._restore()
        self.assertTrue(self.sleep[0])

    def test_fresh_install_with_it_off_writes_nothing(self):
        self._cfg(START_IN_STANDBY=False)
        self._restore()
        self.assertFalse(self.sleep[0])
        self.assertFalse(os.path.exists(self.hud_file))


# ── B016 ───────────────────────────────────────────────────────────────────
# Merged with the config-wiring batch's B052 / B053, which fixed the same bug
# first: the tray toggle goes through the voice setter
# (core.actions._act_ambient_mode_set), both flip from what is really running
# (_ambient_effective_on), and an OFF is saved as AMBIENT_LISTEN_ENABLED - live
# (the gated-turn learner's gate) and in user_settings.json (the only key the
# boot autostarts read). These tests drive that one mechanism.
class AmbientOffTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.active = [False]
        self._p(bc, "_ambient_mode_active", self.active)
        self.writes = []
        self._p(bc, "_write_hud_state",
                side_effect=lambda **k: self.writes.append(k))
        self.listen_start = mock.Mock(return_value="mic on")
        self.listen_stop = mock.Mock(return_value="mic off")
        self._p(bc, "ACTIONS", {"ambient_listen_start": self.listen_start,
                                "ambient_listen_stop": self.listen_stop})
        # The daemon AMBIENT_LISTEN_ENABLED auto-started at boot.
        thread = mock.Mock()
        thread.is_alive.return_value = True
        self.listen_mod = mock.Mock(_thread=thread)
        self.ext = types.ModuleType("skill_ambient_multimodal_extract")
        self.ext.ambient_extract_start = mock.Mock(return_value="")
        self.ext.ambient_extract_stop = mock.Mock(return_value="")
        mods = mock.patch.dict(sys.modules, {
            "skill_ambient_listen": self.listen_mod,
            "skill_ambient_multimodal_extract": self.ext})
        mods.start()
        self.addCleanup(mods.stop)
        self._p(bc, "_is_staging", return_value=False)
        # The setter writes the flag live and saves it: point the save at a
        # temp file, and put both live copies back afterwards.
        self._p(bc, "AMBIENT_LISTEN_ENABLED", True)
        self._cfg(AMBIENT_LISTEN_ENABLED=True)
        self.settings = os.path.join(self.tmp, "user_settings.json")
        self._settings_doc(True)
        env = mock.patch.dict(os.environ,
                              {"JARVIS_SETTINGS_PATH": self.settings})
        env.start()
        self.addCleanup(env.stop)

    def _settings_doc(self, on):
        with open(self.settings, "w", encoding="utf-8") as f:
            json.dump({"AMBIENT_LISTEN_ENABLED": on}, f)

    def _saved_setting(self):
        with open(self.settings, encoding="utf-8") as f:
            return json.load(f)["AMBIENT_LISTEN_ENABLED"]

    def test_tray_off_also_stops_the_fact_extractor(self):
        self._quiet(self.bc._dispatch_tray_command, "ambient_mode_toggle", {})
        self.listen_stop.assert_called_once_with("")
        self.ext.ambient_extract_stop.assert_called_once_with("")
        self.listen_start.assert_not_called()

    def test_tray_off_is_saved_for_the_next_boot(self):
        import core.config as cfg
        self._quiet(self.bc._dispatch_tray_command, "ambient_mode_toggle", {})
        self.assertIs(self._saved_setting(), False)
        self.assertIs(self.bc.AMBIENT_LISTEN_ENABLED, False)
        self.assertIs(cfg.AMBIENT_LISTEN_ENABLED, False)

    def test_an_explicit_on_saves_it_back_on(self):
        self.listen_mod._thread.is_alive.return_value = False
        self._settings_doc(False)
        self._p(self.bc, "AMBIENT_LISTEN_ENABLED", False)
        self._quiet(self.bc._dispatch_tray_command, "ambient_mode_toggle", {})
        self.listen_start.assert_called_once_with("")
        self.assertIs(self._saved_setting(), True)
        self.assertIs(self.bc.AMBIENT_LISTEN_ENABLED, True)

    def test_voice_toggle_flips_what_is_really_running(self):
        # Running (autostarted) while the cell says False: "ambient mode"
        # must turn it OFF, not "start" the running daemon.
        self._quiet(self.bc._act_ambient_mode_toggle, "")
        self.listen_stop.assert_called_once_with("")
        self.listen_start.assert_not_called()
        self.assertIs(self._saved_setting(), False)

    def test_the_restore_never_reopens_the_mic_after_an_off(self):
        # hud_state still says "on" from before the OFF; the saved setting
        # (what the OFF wrote) wins.
        self._persist({"ambient_mode_active": True})
        self._cfg(START_IN_STANDBY=False, AMBIENT_LISTEN_ENABLED=False)
        self._restore()
        self.listen_start.assert_not_called()
        self.assertIs(self.active[0], False)

    def test_gated_turn_learner_stops_after_an_explicit_off(self):
        bc = self.bc
        mem = {"facts": [], "projects": []}
        with mock.patch.object(bc, "_ambient_media_is_playing",
                               return_value=False), \
             mock.patch.object(bc, "_call_local_llm", return_value="PERSON"), \
             mock.patch.object(bc, "learn_from_turn") as lft:
            self._quiet(bc._dispatch_tray_command, "ambient_mode_toggle", {})
            self._quiet(bc._ambient_learn_from_gated,
                        "the wifi password is hunter2", mem)
            lft.assert_not_called()
            self.listen_mod._thread.is_alive.return_value = False
            self._quiet(bc._dispatch_tray_command, "ambient_mode_toggle", {})
            self._quiet(bc._ambient_learn_from_gated,
                        "the wifi password is hunter2", mem)
            lft.assert_called_once()


# ── B075 ───────────────────────────────────────────────────────────────────
class DrainerWithoutTrayTests(_Base):
    def _enclosing_ifs(self, target_name):
        """The `if` tests enclosing main()'s Thread(target=<target_name>)."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.bc.main)))
        found = []

        def walk(node, stack):
            for child in ast.iter_child_nodes(node):
                if (isinstance(child, ast.Call)
                        and any(kw.arg == "target"
                                and isinstance(kw.value, ast.Name)
                                and kw.value.id == target_name
                                for kw in child.keywords)):
                    found.append(list(stack))
                walk(child, stack + ([child] if isinstance(child, ast.If)
                                     else []))
        walk(tree, [])
        self.assertTrue(found, f"no Thread(target={target_name}) in main()")
        return found[0]

    def test_the_drainer_runs_with_the_tray_icon_off(self):
        for target in ("_tray_command_drainer", "_tray_state_publisher"):
            ifs = self._enclosing_ifs(target)
            tests = " | ".join(ast.unparse(i.test) for i in ifs)
            self.assertNotIn("TRAY_ENABLED", tests,
                             f"{target} only runs with the tray icon on - the "
                             f"dashboard and Settings controls die with it")
            self.assertIn("_is_staging()", tests,
                          f"{target} must never run in staging (it shares the "
                          f"live tray_commands.json)")

    def test_tray_off_boot_prunes_a_leftover_restart_first(self):
        src = textwrap.dedent(inspect.getsource(self.bc.main))
        block = src[src.index("if not _is_staging():\n        if not TRAY_ENABLED:"):]
        block = block[:block.index("target=_tray_command_drainer")]
        self.assertIn("_prune_stale_tray_commands(", block)

    def test_hud_state_keeps_feeding_the_web_dashboard(self):
        with mock.patch.object(self.bc, "HUD_ENABLED", False), \
             mock.patch.object(self.bc, "TRAY_ENABLED", False), \
             mock.patch.object(self.bc, "WEB_INTERFACE_ENABLED", True):
            self.bc._write_hud_state(mic_muted=True)
        self.assertTrue(self._saved()["mic_muted"],
                        "with the HUD and tray off the dashboard froze")


# ── B077 ───────────────────────────────────────────────────────────────────
class WebCommandDedupeTests(_Base):
    def test_a_web_command_reappended_by_the_drain_race_runs_once(self):
        from tools import web_interface as wi
        bc = self.bc
        self._p(bc, "_tray_seen_cids", OrderedDict(), create=True)
        muted = [False]
        self._p(bc, "_mic_muted", muted)
        self._p(bc, "_write_hud_state")
        inbox = os.path.join(self.tmp, "tray_commands.json")
        wi.send_tray_command("mic_mute_toggle", inbox)
        with open(inbox, encoding="utf-8") as f:
            entries = json.load(f)
        inflight = inbox + ".inflight"

        def _claim():
            with open(inflight, "w", encoding="utf-8") as f:
                json.dump(entries, f)
            return self._quiet(bc._process_inflight, inflight)

        _claim()
        self.assertTrue(muted[0])
        # The web writer read the inbox just before the drainer's claim and
        # wrote the SAME entry back:
        _, log = _claim()
        self.assertTrue(muted[0], "the duplicate un-muted the mic again")
        self.assertIn("skipped duplicate", log)


if __name__ == "__main__":
    unittest.main()
