"""core.peer_settings - one set of settings across the owner's machines
(2026-10-20: "the hand over should be like JARVIS jumping from one machine to
another like an identical copy, except for obvious hardware differences").

Pure: temp files only, no live settings, no network.

    python -m unittest tests.test_peer_settings
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from core import peer_settings as S

NOW = 1_800_000_000.0


class SharedKeyTests(unittest.TestCase):
    def test_behaviour_settings_are_shared(self):
        for k in ("HUMOR_MODE", "TURN_CHECK_MODE", "CLOUD_BOOST_UNTIL",
                  "MEMORY_EMBED_MODEL", "SITE_SHORTCUTS", "MEDIA_AEC_MODE",
                  "FOLLOWUP_WINDOW_S", "MIC_BUS_MODE", "LEARN_ONLY_FROM_OWNER"):
            with self.subTest(k=k):
                self.assertTrue(S.is_shared(k))

    def test_hardware_endpoints_and_secrets_stay_on_their_machine(self):
        for k in ("CAMERAS", "MONITORS", "HUD_MONITOR", "VAD_THRESHOLD",
                  "KINECT_ENABLED", "HAND_MOUSE_ENABLED", "AIR_MOUSE_ENGINE",
                  "AUDIO_AUTOSWITCH_HEADSET", "WHISPER_DEVICE", "RAG_INDEX_PATHS",
                  "LOCAL_LLM_BASE_URL", "RAG_OLLAMA_ENDPOINT",
                  "VOICE_CLONE_SERVER_CMD", "PEER_HANDOFF_SSH",
                  "HANDOFF_RESUME_WHEN_PEER_STOPS", "SERVER_WATCH_HOST",
                  "WEB_INTERFACE_TOKEN", "MICROPHONE_INDEX", "AMBIENT_SCREEN_ENABLED",
                  "SOME_API_KEY", "_private", "lowercase_key", ""):
            with self.subTest(k=k):
                self.assertFalse(S.is_shared(k))


class LedgerTests(unittest.TestCase):
    def test_first_sight_is_age_unknown_and_a_change_is_stamped(self):
        ledger = {}
        self.assertEqual(S.refresh_ledger({"HUMOR_MODE": "dry", "CAMERAS": []},
                                          ledger, now=NOW), [])
        self.assertEqual(ledger, {"HUMOR_MODE": {"v": "dry", "t": 0.0}})
        self.assertEqual(S.refresh_ledger({"HUMOR_MODE": "sarcastic"}, ledger,
                                          now=NOW + 5), ["HUMOR_MODE"])
        self.assertEqual(ledger["HUMOR_MODE"], {"v": "sarcastic", "t": NOW + 5})

    def test_a_removed_key_is_recorded_as_unset(self):
        ledger = {"HUMOR_MODE": {"v": "dry", "t": 0.0}}
        S.refresh_ledger({}, ledger, now=NOW)
        self.assertEqual(ledger["HUMOR_MODE"], {"v": S.UNSET, "t": NOW})


class MergeTests(unittest.TestCase):
    def test_the_newer_change_wins_per_key(self):
        settings = {"HUMOR_MODE": "dry", "FOLLOWUP_WINDOW_S": 45}
        ledger = {"HUMOR_MODE": {"v": "dry", "t": NOW},
                  "FOLLOWUP_WINDOW_S": {"v": 45, "t": NOW + 100}}
        peer = {"HUMOR_MODE": {"v": "sarcastic", "t": NOW + 50},
                "FOLLOWUP_WINDOW_S": {"v": 20, "t": NOW + 10}}
        self.assertEqual(S.merge_peer(settings, ledger, peer), ["HUMOR_MODE"])
        self.assertEqual(settings, {"HUMOR_MODE": "sarcastic", "FOLLOWUP_WINDOW_S": 45})

    def test_age_unknown_on_both_sides_changes_nothing(self):
        settings = {"FOLLOWUP_WINDOW_S": 45}
        ledger = {"FOLLOWUP_WINDOW_S": {"v": 45, "t": 0.0}}
        self.assertEqual(S.merge_peer(settings, ledger,
                                      {"FOLLOWUP_WINDOW_S": {"v": 20, "t": 0.0}}), [])
        self.assertEqual(settings["FOLLOWUP_WINDOW_S"], 45)

    def test_a_taken_value_never_echoes_back(self):
        a_set, a_led = {"HUMOR_MODE": "dry"}, {"HUMOR_MODE": {"v": "dry", "t": 0.0}}
        b_set, b_led = {"HUMOR_MODE": "sarcastic"}, {"HUMOR_MODE": {"v": "sarcastic", "t": NOW}}
        self.assertEqual(S.merge_peer(a_set, a_led, b_led), ["HUMOR_MODE"])
        # A's ledger now carries B's own stamp: B takes nothing back.
        self.assertEqual(S.merge_peer(b_set, b_led, a_led), [])

    def test_hardware_keys_from_the_peer_are_ignored(self):
        settings = {"CAMERAS": ["mine"]}
        peer = {"CAMERAS": {"v": ["theirs"], "t": NOW}}
        self.assertEqual(S.merge_peer(settings, {}, peer), [])
        self.assertEqual(settings["CAMERAS"], ["mine"])

    def test_an_unset_on_the_peer_removes_the_key(self):
        settings = {"HUMOR_MODE": "dry"}
        ledger = {"HUMOR_MODE": {"v": "dry", "t": 0.0}}
        S.merge_peer(settings, ledger, {"HUMOR_MODE": {"v": S.UNSET, "t": NOW}})
        self.assertNotIn("HUMOR_MODE", settings)

    def test_junk_never_raises(self):
        for junk in (None, [], "x", {"HUMOR_MODE": "notadict"},
                     {"HUMOR_MODE": {"t": "soon"}}, {"HUMOR_MODE": {"v": 1, "t": "x"}}):
            with self.subTest(junk=junk):
                self.assertEqual(S.merge_peer({}, {}, junk), [])


class FileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.settings = os.path.join(self.dir, "user_settings.json")
        self.ledger = os.path.join(self.dir, "settings_ledger.json")

    def _write(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _read(self, path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_apply_writes_a_backup_and_the_newer_values(self):
        self._write(self.settings, {"HUMOR_MODE": "dry", "CAMERAS": ["laptop"]})
        logs = []
        taken = S.apply_peer_ledger({"HUMOR_MODE": {"v": "sarcastic", "t": NOW},
                                     "CAMERAS": {"v": ["desk"], "t": NOW}},
                                    self.settings, self.ledger, log=logs.append)
        self.assertEqual(taken, ["HUMOR_MODE"])
        self.assertEqual(self._read(self.settings),
                         {"HUMOR_MODE": "sarcastic", "CAMERAS": ["laptop"]})
        self.assertEqual(self._read(self.settings + S.BACKUP_SUFFIX)["HUMOR_MODE"], "dry")
        self.assertIn("HUMOR_MODE", logs[0])

    def test_boot_merge_is_off_without_a_handoff_partner(self):
        self._write(self.settings, {"HUMOR_MODE": "dry"})
        self._write(os.path.join(self.dir, "peer_memory_from_other.json"),
                    {"host": "other", "settings": {"HUMOR_MODE": {"v": "sarcastic", "t": NOW}}})
        self.assertEqual(S.apply_pending(self.dir, self.settings, self.ledger), [])
        self.assertEqual(self._read(self.settings)["HUMOR_MODE"], "dry")

    def test_boot_merge_takes_the_peer_bundle_on_disk_but_never_its_own(self):
        self._write(self.settings, {"HUMOR_MODE": "dry", "PEER_HANDOFF_SSH": "me@peer"})
        self._write(os.path.join(self.dir, "peer_memory_from_other.json"),
                    {"host": "other", "settings": {"HUMOR_MODE": {"v": "sarcastic", "t": NOW}}})
        self._write(os.path.join(self.dir, "peer_memory_from_self.json"),
                    {"host": "selfhost", "settings": {"HUMOR_MODE": {"v": "loud", "t": NOW + 9}}})
        with mock.patch("socket.gethostname", return_value="SELFHOST"):
            taken = S.apply_pending(self.dir, self.settings, self.ledger)
        self.assertEqual(taken, ["HUMOR_MODE"])
        self.assertEqual(self._read(self.settings)["HUMOR_MODE"], "sarcastic")

    def test_load_ledger_saves_only_on_change(self):
        self._write(self.settings, {"HUMOR_MODE": "dry"})
        self.assertEqual(S.load_ledger(self.settings, self.ledger, now=NOW),
                         {"HUMOR_MODE": {"v": "dry", "t": 0.0}})
        mtime = os.path.getmtime(self.ledger)
        S.load_ledger(self.settings, self.ledger, now=NOW + 1)
        self.assertEqual(os.path.getmtime(self.ledger), mtime)

    def test_a_missing_or_broken_settings_file_is_quiet(self):
        self.assertEqual(S.apply_peer_ledger({"HUMOR_MODE": {"v": 1, "t": NOW}},
                                             self.settings, self.ledger), [])
        with open(self.settings, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(S.apply_peer_ledger({"HUMOR_MODE": {"v": 1, "t": NOW}},
                                             self.settings, self.ledger), [])
        self.assertEqual(S.apply_pending(self.dir, self.settings, self.ledger), [])


if __name__ == "__main__":
    unittest.main()
