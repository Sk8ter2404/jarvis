"""v3 W01 speak-contract ratchet: an action that returns an answer is voiced.

2026-10-18 12:06:45: smart_home_discover ran its scan and the reply still said
"I'll need to run a scan" - the action sat in no speak set, so its result was
logged and dropped. This pins the set of registered actions that are in no
speak set (NEVER_VOICED, today's list); a NEW one fails the test.

    python -m unittest tests.monolith.test_monolith_speak_contract_ratchet
"""
from __future__ import annotations

import os
import unittest

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# -- speak contract ----------------------------------------------------------
# Every action a skill registers is voiced one of three ways (a verbatim
# result, an informative follow-up, or it speaks for itself) - or it is
# "never voiced": it runs, and whatever it returns is logged and dropped.
# That is right for an action that only DOES something and returns a bare
# "ok"; it is the 2026-10-18 smart_home_discover bug for one that returns an
# answer (the scan ran, the reply said "I'll need to run a scan").
#
# This is a ratchet. NEVER_VOICED is today's list; a NEW registration that is
# in no speak set fails the test. Route it (SPEAK_RESULT_VERBATIM_ACTIONS or
# INFORMATIVE_ACTIONS, or a module-level declaration in the skill) if it
# returns an answer; add it here, on purpose, only if it is a pure side effect.
NEVER_VOICED = frozenset((
    'ambient_audio_start', 'ambient_audio_stop', 'ambient_extract_now',
    'ambient_extract_start', 'ambient_extract_stop', 'ambient_full_start',
    'ambient_full_stop', 'ambient_learning_mode',
    'ambient_learning_mode_off', 'ambient_learning_mode_on',
    'ambient_listen_start', 'ambient_listen_stop', 'ambient_listening',
    'ambient_mic_only', 'ambient_mode', 'ambient_mode_off',
    'ambient_mode_on', 'ambient_screen_start', 'ambient_screen_stop',
    'anticipation_briefing_now', 'apple_music', 'bambu_setup',
    'browser_open', 'browser_reset_profile', 'browser_screenshot',
    'browser_status', 'browser_stop', 'chappie_mode', 'clear_llm_cache',
    'clear_tasks', 'click', 'close_last_opened', 'close_window',
    'configure_printer', 'control_light', 'control_plug', 'create_skill',
    'disable_guest_network', 'disable_night_owl', 'disney_plus',
    'dossier', 'dossier_on', 'enable_guest_network', 'enable_night_owl',
    'end_night_owl', 'enroll_xtts_sample', 'enter_ambient_learning',
    'exit_ambient_learning', 'exit_jarvis', 'export_memory', 'file_on',
    'first_time_printer_setup', 'focus_window', 'force_backup',
    'forget_alexa_login', 'forget_last_hour', 'globe_clear', 'globe_pin',
    'greet_new_people_off', 'greet_new_people_on', 'guest_mode_off',
    'guest_mode_on', 'hide_globe', 'hide_hud', 'hotkey',
    'hours_reminder_fire', 'hue_set_bridge_ip', 'hulu', 'kasa_control',
    'kasa_smart_home_control', 'kick_guest_network', 'latency_benchmark',
    'launch_app', 'magichome_list_devices', 'max', 'media_next',
    'media_playpause', 'media_prev', 'minimize_window', 'morning_tabs',
    'move_window_to_monitor', 'netflix', 'night_owl_mode',
    'night_owl_off', 'night_owl_on', 'notice_new_people',
    'open_apple_music', 'open_on_monitor', 'pause_diagnostics',
    'pause_phone_bridge', 'play_streaming', 'power_off_jarvis', 'press',
    'prime_video', 'print_companion_history', 'pull_up_dossier',
    'pull_up_file', 'quit_jarvis', 'rag_search_quiet',
    'refresh_smart_home_router', 'reload_skills', 'replay_last_action',
    'reset_kernel', 'reset_llm_cache', 'reset_memory', 'restart',
    'resume_diagnostics', 'resume_phone_bridge', 'run_shell',
    'run_smoke_test', 'say_aloud', 'scroll', 'search', 'session_resume',
    'set_timer', 'set_tts_backend', 'set_volume', 'setup_bambu',
    'setup_printer', 'show_globe', 'show_hud', 'shut_down',
    'shutdown_jarvis', 'silent_learning', 'silent_learning_off',
    'silent_learning_on', 'smart_home_purge_cookie', 'spotify',
    'start_eavesdropping', 'start_overnight_upgrade',
    'stop_eavesdropping', 'stop_greeting_people', 'stop_pipeline',
    'suit_up', 'suit_up_sequence', 'switch_llm', 'switch_llm_picker',
    'test_each_skill', 'test_mic', 'test_tts', 'test_vision',
    'toggle_hud', 'turn_off_jarvis', 'type', 'upgrade',
    'voice_gating_off', 'voice_gating_on', 'volume_down', 'volume_mute',
    'volume_unmute', 'volume_up', 'vscode_command',
    'wake_listener_configure', 'wake_listener_start',
    'wake_listener_stop', 'wake_resume_answer_then_quiet',
    'wake_resume_stay_talkative', 'wake_word_mode_off',
    'wake_word_mode_on', 'weekly_digest_now', 'what_do_you_have_on',
    'whats_on_file', 'youtube', 'youtube_play'
))


def _load_registration_scan():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "registration_scan", os.path.join(_ROOT, "tools", "registration_scan.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires_monolith
class SpeakContractRatchetTests(_Base):
    _memo = None

    def _never_voiced(self):
        # the source scan is read-only and slow: do it once per run
        if SpeakContractRatchetTests._memo is None:
            SpeakContractRatchetTests._memo = self._scan_never_voiced()
        return SpeakContractRatchetTests._memo

    def _scan_never_voiced(self):
        import tests.monolith.test_monolith_runtime_bugfixes as T
        reader = [c for c in vars(T).values()
                  if isinstance(c, type)
                  and hasattr(c, "_declared_speak_names")][0]._declared_speak_names
        bc = self.bc
        rs = _load_registration_scan()
        routed = (set(bc.SPEAK_RESULT_VERBATIM_ACTIONS)
                  | set(bc.INFORMATIVE_ACTIONS) | set(bc.SELF_VOICED_ACTIONS))
        paths = [os.path.join(_ROOT, "bobert_companion.py")]
        for sub in ("skills", "core"):
            d = os.path.join(_ROOT, sub)
            paths += [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.endswith(".py")]
        for p in paths:
            routed |= reader(p)
        found = set()
        for p in paths:
            try:
                regs = rs.scan_file(p)
            except SyntaxError:
                continue
            found |= {n for n in regs if n not in routed}
        return found

    def test_smart_home_discover_is_informative_after_the_skill_folds_in(self):
        import importlib
        mod = importlib.import_module("skills.smart_home_discover")
        bc = self.bc
        saved = {n: set(getattr(bc, n)) for n in (
            "INFORMATIVE_ACTIONS", "SPEAK_RESULT_VERBATIM_ACTIONS",
            "SELF_VOICED_ACTIONS")}
        try:
            bc._collect_skill_speak_sets(mod, "smart_home_discover")
            self.assertIn("smart_home_discover", bc.INFORMATIVE_ACTIONS)
            for alias in ("discover_smart_home", "smart_home_setup",
                          "refresh_smart_home"):
                self.assertIn(alias, bc.INFORMATIVE_ACTIONS)
            self.assertFalse(
                bc.INFORMATIVE_ACTIONS & bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        finally:
            for n, v in saved.items():
                getattr(bc, n).clear()
                getattr(bc, n).update(v)

    def test_the_scan_sees_the_tree(self):
        found = self._never_voiced()
        self.assertGreater(len(found), 100, len(found))

    def test_no_new_never_voiced_action(self):
        new = sorted(self._never_voiced() - NEVER_VOICED)
        self.assertEqual(
            new, [],
            "these registered actions are in no speak set, so whatever they "
            "return is logged and never voiced (the smart_home_discover "
            "shape). Route them (INFORMATIVE_ACTIONS / "
            "SPEAK_RESULT_VERBATIM_ACTIONS, or a module-level declaration in "
            "the skill); add to NEVER_VOICED only for a pure side effect: "
            f"{new}")

    def test_the_ledger_has_no_stale_rows(self):
        stale = sorted(NEVER_VOICED - self._never_voiced())
        self.assertEqual(
            stale, [],
            f"NEVER_VOICED names now routed or gone - delete them: {stale}")


if __name__ == "__main__":
    unittest.main()
