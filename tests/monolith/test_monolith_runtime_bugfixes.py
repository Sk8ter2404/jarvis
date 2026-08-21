"""Regression tests for runtime bugs JARVIS itself queued from live sessions.

2026-06-02:

  1. Printer-status actions (check_print / how_is_the_print / print_details)
     returned a result that was logged but never SPOKEN, because they weren't in
     INFORMATIVE_ACTIONS — so the result->speech follow-up loop never fired
     (unlike check_credits, which is listed and speaks correctly).

  2. Vision-click targeting overshot on a >100%-scaled multi-monitor rig:
     find_click_target added a NATIVE-pixel offset to a LOGICAL monitor origin
     without scaling, so clicks landed too far right/down.

2026-06-03 ("you didn't speak it"):

  3. version_info (and the system_pulse status family) returned a finished,
     user-facing answer ("I'm on version 1.20.4, last updated …, sir.") that was
     logged but never SPOKEN: the action wasn't in INFORMATIVE_ACTIONS (so no
     follow-up LLM turn) and its result isn't a failure (so the failure
     follow-up didn't fire either) — the user only heard the "One moment, sir."
     preamble. Unlike the printer fix, these results are already perfect
     sentences, so they are now spoken VERBATIM via SPEAK_RESULT_VERBATIM_ACTIONS
     + _speak_verbatim_results(), with dedup so an inlined answer isn't
     double-spoken and side-effect/failure results are left alone.

2026-06-07 (mic-stream TOCTOU, REVIEW_FINDINGS_2 P1-4):

  4. record_speech() opened+started its sd.InputStream BEFORE publishing mic
     ownership via _record_speech_active[0]=True. _refresh_devices() only skips
     the destructive sd._terminate()/sd._initialize() reinit while that flag is
     True, so in the window between the live stream and the (late) flag flip a
     concurrent background caller (self_diagnostic / ambient_listen ->
     get_input_device -> _refresh_devices) could tear PortAudio out from under
     the just-started callback and heap-corrupt the process (0xc0000374). The
     flag is now set BEFORE the open, and cleared again on any open/start
     failure, so the reinit guard can never observe a live-stream-but-flag-False
     state. Tests assert (a) the flag is already True at InputStream construction
     and at .start(), (b) it's cleared if the open raises, and (c) the
     _refresh_devices guard actually defers the reinit while the flag is set.

2026-06-07 (sd.play endpoint swap, REVIEW_FINDINGS_2 P1-9):

  5. play_with_lipsync() resolved out_dev = get_output_device() and then opened
     sd.play(audio, sr, device=out_dev). The autoswitch daemon
     (audio/audio_switch.py) flips the default render endpoint for all roles on a
     3 s poll, so a headset/speaker autoswitch landing between that resolve and
     the PortAudio open made the open fail with DirectSound -9999 and the
     utterance was silently dropped (no retry, unlike record_speech's input
     side). The play call now runs through a _play_audio_safe() helper that, on
     PortAudioError, invalidates the cached output index and retries once on
     device=None (the now-current system default) so the speech finishes on the
     new endpoint; if that also fails it re-raises into _speak's existing
     device-hiccup handler (fails loud, never silent). Tests assert (a) a healthy
     play is untouched (single call, original device), (b) a one-shot -9999 is
     recovered by a device=None retry AND the stale cache is invalidated, and
     (c) a persistent PortAudioError propagates rather than being swallowed.

Monolith-tier (full-deps): run locally; skip on the light-deps CI runner.
    python -m unittest tests.monolith.test_monolith_runtime_bugfixes
"""
from __future__ import annotations

import io
import os
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class PrinterStatusInformativeTests(MonolithGlobalsTestCase):
    def test_printer_status_actions_are_informative(self):
        # Without these in INFORMATIVE_ACTIONS the dispatch follow-up loop breaks
        # immediately and the printer status is logged but never voiced.
        for name in ("check_print", "how_is_the_print", "print_details"):
            self.assertIn(name, self.bc.INFORMATIVE_ACTIONS,
                          f"{name} must be informative so its result is spoken")

    def test_check_credits_still_informative(self):
        # The reference behaviour we're matching — guard against accidental removal.
        self.assertIn("check_credits", self.bc.INFORMATIVE_ACTIONS)


@requires_monolith
class FindClickTargetScalingTests(MonolithGlobalsTestCase):
    @staticmethod
    def _png(w: int, h: int) -> bytes:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (0, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()

    def test_native_pixels_scaled_to_logical_before_adding_origin(self):
        """A target at the native-pixel CENTRE of a 3840x2160 capture of a
        2560x1440-LOGICAL monitor (150% scale) whose origin is (-2560, 0) must
        click at the LOGICAL centre (-1280, 720) — not the un-scaled (-640, 1080)
        the old code produced."""
        bc = self.bc
        png_lowres = self._png(1568, 882)     # Pass-1 downscale (max_dim 1568)
        png_native = self._png(3840, 2160)    # Pass-2 full-res (native pixels)

        def fake_shot(monitor=None, max_dim=1568):
            return png_lowres if max_dim <= 1568 else png_native

        def fake_vision(_desc, _png, w, h):
            # Pass-1 (full image) -> centre; Pass-2 (the 500x500 crop) -> centre.
            return (784, 441) if (w, h) == (1568, 882) else (250, 250)

        with mock.patch.dict(bc.MONITORS, {"qa": (-2560, 0, 2560, 1440)}), \
             mock.patch.object(bc, "take_screenshot", side_effect=fake_shot), \
             mock.patch.object(bc, "_query_vision_for_coords", side_effect=fake_vision):
            pt = bc.find_click_target("a sidebar item", monitor="qa")

        self.assertIsNotNone(pt)
        # native centre (1920,1080) x (2560/3840, 1440/2160) -> (1280,720); + origin
        self.assertAlmostEqual(pt[0], -1280, delta=2)
        self.assertAlmostEqual(pt[1], 720, delta=2)
        # And definitively NOT the old buggy native-added coordinate.
        self.assertNotEqual((pt[0], pt[1]), (-640, 1080))

    def test_no_scale_when_native_equals_logical(self):
        """At 100% scale (native == logical) the scaling is a no-op, so an
        un-scaled single-monitor setup can't regress."""
        bc = self.bc
        png_lowres = self._png(1280, 720)
        png_native = self._png(2560, 1440)   # == logical below

        def fake_shot(monitor=None, max_dim=1568):
            return png_lowres if max_dim <= 1568 else png_native

        def fake_vision(_desc, _png, w, h):
            return (640, 360) if (w, h) == (1280, 720) else (250, 250)

        with mock.patch.dict(bc.MONITORS, {"qa": (0, 0, 2560, 1440)}), \
             mock.patch.object(bc, "take_screenshot", side_effect=fake_shot), \
             mock.patch.object(bc, "_query_vision_for_coords", side_effect=fake_vision):
            pt = bc.find_click_target("x", monitor="qa")

        self.assertIsNotNone(pt)
        # native == logical -> the returned point equals the native refined coord
        # plus the (0,0) origin, i.e. no scale distortion.
        self.assertTrue(0 <= pt[0] <= 2560 and 0 <= pt[1] <= 1440)


@requires_monolith
class VerbatimResultSpokenTests(MonolithGlobalsTestCase):
    """Bug 3: an informational action whose result is a finished sentence
    (version_info / system_pulse) must be SPOKEN, exactly once, without an LLM
    round-trip — and without double-speaking or speaking side-effect/failure
    results.
    """

    VER = "I'm on version 1.20.4, last updated Saturday morning at 7:03 AM, sir."

    # ── membership ──────────────────────────────────────────────────────────
    def test_version_family_in_verbatim_set(self):
        for name in ("version_info", "what_version", "when_updated"):
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} must speak its result verbatim")

    def test_status_family_in_verbatim_set(self):
        for name in ("system_pulse", "check_system", "status_report"):
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)

    def test_readout_family_in_verbatim_set(self):
        # Read-out actions whose result is a finished, user-facing sentence the
        # user explicitly asked for. Each was in NEITHER speak set, so its answer
        # was logged but NEVER voiced (only the "Of course, sir" preamble). This
        # is the same "you're not speaking for some actions still" class as the
        # version_info/system_pulse fixes above. Confirmed each returns a
        # spoken-ready sentence and does not self-speak in its handler.
        for name in ("weather_briefing", "weather_forecast",
                     "wake_word_mode_status",
                     "check_for_updates", "check_updates", "is_there_an_update",
                     "model_costs", "llm_costs", "model_prices", "compare_models",
                     "morning_briefing",
                     "smart_home_control", "control_device", "control_smart_home",
                     "smart_home_router_status"):
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} returns a finished answer that must be spoken")

    def test_readout_completeness_sweep_in_verbatim_set(self):
        # 2026-07-04 full-repo audit + live drive: every action below RETURNS a
        # finished user-facing answer, does NOT self-speak, and was in NEITHER
        # speak set — so its answer was logged but never voiced (the recurring
        # "you're not speaking for some actions still" class). Status/list
        # one-liners only; multi-item readers + side-effect actions are excluded
        # (a later INFORMATIVE pass owns those). Regression guard: keep them
        # voiced.
        # NOTE (2026-08-20 privacy audit): two names that used to be pinned here
        # belonged to GITIGNORED personal skills and embedded a real person /
        # a one-off personal event. They were spelled out in this TRACKED test
        # and in the tracked monolith, so .gitignore was not actually hiding
        # them — the repo is public. Those skills now declare their own
        # SPEAK_VERBATIM_ACTIONS and bobert_companion folds them in at load
        # time (_collect_skill_speak_sets); the mechanism is covered by
        # SkillContributedSpeakSetTests below. Do NOT re-add private names here.
        sweep = (
            "air_mouse_status", "amazon_tracking_status", "ambient_extract_status",
            "ambient_listen_status", "anticipation_briefing_status", "anticipation_status",
            "are_you_ok", "audio_music_status", "banter_status",
            "cancel_promise", "chappie_recall_today", "chappie_status", "check_budget",
            "deco_status", "diagnostic_daemon_status", "diagnostic_history",
            "diagnostic_status", "do_you_recognize_me", "draft_preview_gate_status",
            "email_triage_status", "face_track_status", "focus_mode_status",
            "gaze_calibration_status", "gaze_stats", "gaze_status", "gesture_status",
            "guard_status", "hardware_sensors", "is_printer_online", "last_diagnostic_run",
            "list_enrolled_faces", "list_enrolled_voices", "list_notification_rules",
            "list_pending_drafts", "list_phone_backends", "list_playlists",
            "list_point_targets", "list_promises", "list_smart_home_devices",
            "list_tts_backends", "look_around", "mcp_status", "music_aggregate",
            "music_history", "night_owl_status", "notification_triage_status",
            "outbound_gate_status", "pattern_aggregate", "pattern_offer_now",
            "pattern_predictions", "pattern_stats", "phone_bridge_status", "phone_status",
            "point_status", "predictive_morning_setup", "print_companion_status",
            "print_status", "rag_status", "read_changelog", "recognize_face",
            "robot_status", "run_diagnostic", "schedule_status", "screen_watch_status",
            "search_my_files", "self_diagnostic", "show_changelog", "show_last_diagnostic",
            "show_llm_stats", "show_recent_facts", "smart_home_catalog", "status_panel",
            "suit_diagnostics", "system_status", "triage_status", "tv_detect_status",
            "tv_status", "vip_intercept_status", "voice_id_status", "wake_listener_status",
            "weekly_digest", "weekly_digest_status",
            "what_changed", "what_is_broken", "whats_broken", "whats_new", "who_am_i",
            "who_is_talking", "whos_at_the_desk", "whos_talking", "workshop_status",
        )
        for name in sweep:
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} returns a finished answer that must be spoken")

    def test_readout_completeness_v180_stability_announcer_calendar(self):
        # v1.80.0 continuation of the never-voiced readout sweep. Each RETURNS a
        # finished user-facing sentence, does NOT self-speak, and was in neither
        # speak set. stability-gate + announcer are single-path direct-turn
        # readouts; the calendar aliases are ALSO orchestrator-dispatched, but
        # the worker runs actions directly (core/orchestrator.py) — never through
        # _speak_verbatim_results — so voicing them affects only the direct turn
        # the user asked from, no double-speak. Regression guard: keep voiced.
        for name in ("last_stability_gate", "last_stability_gate_result",
                     "last_gate_result", "stability_gate_status", "gate_status",
                     "proactive_announcer_status",
                     "calendar_today", "calendar_next", "ms_graph_calendar"):
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} returns a finished answer that must be spoken")

    def test_cancel_timer_in_verbatim_set(self):
        # 2026-07-21 audit: cancel_timer was registered on the line after
        # list_timers (which IS voiced) but was in NEITHER speak set, so its
        # verdict — including the honest "there are no timers running, sir."
        # when no timer existed — was dropped, and the LLM's inline "Cancelled,
        # sir." hallucination was the only thing heard. Same cancel-confirmation
        # class as cancel_schedule / remove_schedule / cancel_promise, which
        # were already verbatim.
        self.assertIn("cancel_timer", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                      "cancel_timer's verdict must be voiced verbatim")
        # Kept OUT of INFORMATIVE (the sets must stay disjoint — see
        # test_speak_sets_are_disjoint).
        self.assertNotIn("cancel_timer", self.bc.INFORMATIVE_ACTIONS)

    def test_cancel_timer_honest_verdict_passes_failure_guard(self):
        # The honest no-timers verdict must carry NO FAILURE_MARKERS substring,
        # or _speak_verbatim_results' failure guard silently re-swallows it and
        # the fix above is moot. Guards against a future FAILURE_MARKERS
        # addition (e.g. a "no timer" marker) undoing the voicing. Exercises
        # the REAL skill handler, not a canned string.
        from core.failure_markers import FAILURE_MARKERS
        from tests._skill_harness import load_skill_isolated
        modt, actions = load_skill_isolated("timer")
        modt._timers.clear()
        out = actions["cancel_timer"]("")
        low = out.lower()
        self.assertIn("no timers", low)   # the honest correction itself
        hits = [m for m in FAILURE_MARKERS if m in low]
        self.assertEqual(hits, [],
                         f"cancel_timer's honest verdict {out!r} matches "
                         f"FAILURE_MARKERS {hits} — the verbatim guard would "
                         f"swallow it and the verdict goes silent again")

    def test_verbatim_set_excludes_side_effect_actions(self):
        # TRUE side-effect actions must NEVER verbatim-speak their result (the
        # inline reply already confirms them, and the effect is the point) —
        # guards against a careless future add. NOTE: weather_briefing was
        # previously (wrongly) listed here and thereby made SILENT — it has no
        # side effect; its result IS the answer, so it moved to the verbatim set
        # (see test_readout_family_in_verbatim_set).
        for name in ("play_music", "volume_up", "set_timer", "launch_app",
                     "pause_music", "next_song"):
            self.assertNotIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)

    # ── _speak_verbatim_results() unit behaviour ────────────────────────────
    def test_helper_speaks_informational_result(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            handled = bc._speak_verbatim_results(
                [("version_info", self.VER, False)], already_spoken="One moment, sir.")
        self.assertEqual(spoken, [self.VER])
        self.assertEqual(handled, {"version_info"})

    def test_helper_dedupes_already_spoken(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            # The inline reply already contained the answer (case-insensitively).
            handled = bc._speak_verbatim_results(
                [("version_info", self.VER, False)],
                already_spoken=f"Certainly. {self.VER.upper()}")
        self.assertEqual(spoken, [], "must not re-speak an already-voiced answer")
        self.assertEqual(handled, set())

    def test_helper_skips_failures(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            handled = bc._speak_verbatim_results(
                [("version_info", "could not read version info: boom", False)])
        self.assertEqual(spoken, [], "raw failures stay with the failure follow-up")
        self.assertEqual(handled, set())

    def test_helper_ignores_non_verbatim_actions(self):
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)):
            handled = bc._speak_verbatim_results(
                [("play_music", "playing Take Five by Dave Brubeck", True)])
        self.assertEqual(spoken, [])
        self.assertEqual(handled, set())

    # ── end-to-end through _run_llm_dispatch ────────────────────────────────
    def _dispatch_capture(self, reply, actions):
        """Run _run_llm_dispatch with a canned LLM reply + stub actions, and
        return the list of strings handed to _speak."""
        bc = self.bc
        spoken = []
        with mock.patch.object(bc, "get_response_with_animation",
                               return_value=reply), \
             mock.patch.object(bc, "maybe_glance_response", return_value=None), \
             mock.patch.object(bc, "_speak",
                               side_effect=lambda t, *a, **k: spoken.append(t)), \
             mock.patch.object(bc, "_apply_quip_layer",
                               side_effect=lambda s, r: s), \
             mock.patch.object(bc, "get_followup_response",
                               side_effect=lambda info: ""), \
             mock.patch.dict(bc.ACTIONS, actions), \
             mock.patch.object(bc, "PC_CONTROL_ENABLED", True):
            bc._run_llm_dispatch("what version are you on?")
        return spoken

    def test_dispatch_speaks_version_result_exactly_once(self):
        # THE BUG: preamble was spoken, version answer was dropped.
        spoken = self._dispatch_capture(
            "One moment, sir. [ACTION: version_info]",
            {"version_info": lambda a="": self.VER})
        self.assertIn("One moment, sir.", spoken)
        self.assertEqual(sum(1 for s in spoken if self.VER in s), 1,
                         f"version answer must be spoken exactly once: {spoken}")

    def test_dispatch_does_not_double_speak_inlined_answer(self):
        # When the LLM already inlined the answer, speak it once, not twice.
        spoken = self._dispatch_capture(
            f"{self.VER} [ACTION: version_info]",
            {"version_info": lambda a="": self.VER})
        self.assertEqual(sum(1 for s in spoken if self.VER in s), 1,
                         f"inlined answer double-spoken: {spoken}")

    def test_dispatch_side_effect_result_not_verbatim_spoken(self):
        # play_music is in INFORMATIVE_ACTIONS but NOT the verbatim set — its
        # result must not be read aloud as a second confirmation.
        spoken = self._dispatch_capture(
            "Playing your jazz playlist, sir. [ACTION: play_music, jazz]",
            {"play_music": lambda a="": "playing Take Five by Dave Brubeck"})
        self.assertFalse(any("Take Five" in s for s in spoken),
                         f"side-effect music result was verbatim-spoken: {spoken}")


@requires_monolith
class SpeakContractBugHunt20260707Tests(MonolithGlobalsTestCase):
    """2026-07-07 bug-hunt: ~18 actions whose finished user-facing result was
    SILENTLY DROPPED because the registered name was in NEITHER INFORMATIVE_ACTIONS
    nor SPEAK_RESULT_VERBATIM_ACTIONS — the follow-up loop only voices results for
    actions in one of those sets. The one-line confirmations went to the verbatim
    set; the multi-line/queryable results went to INFORMATIVE. The two sets MUST
    stay DISJOINT (see test_speak_sets_are_disjoint). Regression guard: keep them
    routed and keep the sets disjoint.
    """

    # Names added to SPEAK_RESULT_VERBATIM_ACTIONS (finished one-liners). Each was
    # verified to be a real register()ed action in its named skill and to have
    # been in NEITHER speak set before this fix.
    _VERBATIM_ADDS = (
        # email_triage.py
        "confirm_pending_draft", "send_draft", "send_pending_draft",
        "archive_email", "archive_message", "scrap_pending_draft", "discard_draft",
        # phone_bridge.py
        "notify_phone", "text_my_phone", "push_to_phone",
        # face_id.py
        "enroll_face", "learn_my_face", "remember_this_person", "forget_face",
        # guard_mode.py
        "guard_on", "guard_off",
        # enroll_voice.py
        "enroll_voice", "learn_my_voice", "forget_voice", "set_active_speaker",
        # kinect_gestures.py
        "gestures_on", "gestures_off",
        # kinect_pointing.py
        "point_control_on", "point_control_off", "forget_point_target",
        # image_gen.py
        "generate_image", "make_picture",
        # obs_control.py
        "obs_toggle_mute", "obs_switch_scene", "obs_start_recording",
        "obs_stop_recording", "obs_pause_recording",
        # schedule_manager.py
        "schedule_once", "schedule_recurring", "schedule_cron", "schedule_when",
        "when_condition", "cancel_schedule", "remove_schedule", "fire_schedule",
        "run_schedule",
        # model_picker.py
        "set_model", "set_brain",
        # night_owl_mode.py
        "good_morning",
        # personal_rag.py
        "rag_reindex", "rag_configure", "rag_open_top",
        # sh_ecobee.py
        "ecobee_complete_setup",
        # notification_triage.py
        "add_notification_rule", "remove_notification_rule",
        "pause_notification_triage", "resume_notification_triage",
        # network_deco.py one-line-status aliases
        "printer_online", "device_online", "network_usage", "bandwidth_hogs",
        "whats_using_bandwidth", "deco_refresh", "refresh_network",
        # media fallbacks
        "play_unheard", "play_vibe", "skip_track",
        "play_playlist", "shuffle_library",
        "keep_music_open", "stop_keeping_music_open",
        "youtube_search_direct", "youtube_direct", "yt_direct",
    )

    # Names added to INFORMATIVE_ACTIONS (multi-line / re-summarised).
    _INFORMATIVE_ADDS = (
        # code_executor.py — output carries tracebacks / a "format:" hint the
        # verbatim guard would swallow, so INFORMATIVE (LLM re-summarises) is right.
        "run_python", "python", "eval_python", "compute",
        # network_deco.py roll-call aliases — multi-device client LIST.
        "who_is_on_the_wifi", "network_clients", "list_wifi_clients",
        "network_topology",
    )

    def test_verbatim_additions_present(self):
        for name in self._VERBATIM_ADDS:
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} must speak its finished one-liner result")

    def test_informative_additions_present(self):
        for name in self._INFORMATIVE_ADDS:
            self.assertIn(name, self.bc.INFORMATIVE_ACTIONS,
                          f"{name} must be informative so its result is re-summarised")

    def test_code_executor_not_verbatim(self):
        # run_python et al. carry tracebacks / "format:" hints — they must be
        # INFORMATIVE, never verbatim (the verbatim guard would swallow them).
        for name in ("run_python", "python", "eval_python", "compute"):
            self.assertNotIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)

    def test_speak_sets_are_disjoint(self):
        # The follow-up loop routes INFORMATIVE (re-summarise) and the main loop
        # speaks VERBATIM directly; an action in both would be double-handled.
        overlap = (set(self.bc.INFORMATIVE_ACTIONS)
                   & set(self.bc.SPEAK_RESULT_VERBATIM_ACTIONS))
        self.assertEqual(overlap, set(),
                         f"INFORMATIVE_ACTIONS and SPEAK_RESULT_VERBATIM_ACTIONS "
                         f"must stay disjoint; overlap: {sorted(overlap)}")


@requires_monolith
class FaceTrackWakeReopenTests(MonolithGlobalsTestCase):
    """2026-07-04: the face-track soft-wake reopened the camera with a raw
    cv2.VideoCapture(cam["index"], CAP_DSHOW) on the STATIC index, bypassing
    _open_capture's name-based live-index resolution and the Kinect path. A USB
    re-enumeration could then silently wake the WRONG camera. The wake reopen
    must go through the shared _open_capture(cam) opener, like the initial and
    recovery opens. Source-level guard (the reopen lives in a camera hot-path
    thread that is impractical to drive in a unit test)."""

    def _face_track_source(self):
        with open(self.bc.__file__, encoding="utf-8") as f:
            src = f.read()
        # The wake-reopen block is bounded by these two markers. The END marker
        # phrase ALSO appears earlier in a comment (the "read failure #1 → woke
        # via release+reopen" note ~line 4983), which is BEFORE `start`; searching
        # from 0 would pick that occurrence and slice an empty/backwards string
        # (the whole reason this test spuriously failed). Search for the end
        # marker starting AT `start` so we bound the real reopen block.
        start = src.index("The old handle is now released")
        end = src.index("woke via release+reopen", start)
        return src[start:end]

    def test_wake_reopen_uses_shared_opener_not_static_index(self):
        block = self._face_track_source()
        self.assertIn("_open_capture(cam)", block,
                      "soft-wake must reopen via the name-resolving _open_capture")
        self.assertNotIn('cv2.VideoCapture(cam["index"]', block,
                         "soft-wake must not reopen on the raw STATIC cam index")


class RecordSpeechOwnershipOrderingTests(MonolithGlobalsTestCase):
    """Bug 4 (REVIEW_FINDINGS_2 P1-4): the mic-ownership flag must be published
    BEFORE record_speech opens/starts its InputStream, and dropped again if the
    open fails — so _refresh_devices' reinit guard can never tear PortAudio down
    under a live-but-unflagged stream.

    These drive record_speech without a real mic: sd.InputStream and
    get_input_device are mocked, the watchdog reset signal is pre-set so the
    capture loop bails on its first iteration (still running the finally that
    closes the stream + clears the flag), and _safe_close_stream is stubbed.
    """

    def setUp(self):
        bc = self.bc
        # record_speech short-circuits to None when the mic is "disabled"
        # (staging sets that), so force it live for these tests.
        self._p_disabled = mock.patch.object(bc, "_mic_input_disabled",
                                              return_value=False)
        self._p_close = mock.patch.object(bc, "_safe_close_stream")
        self._p_getdev = mock.patch.object(bc, "get_input_device", return_value=0)
        self._p_disabled.start()
        self._p_close.start()
        self._p_getdev.start()
        # The watchdog Event is module-global and NOT in the harness restore
        # set, so clear it after each test regardless of how the body exits.
        self.addCleanup(bc._watchdog_reset_signal.clear)
        self.addCleanup(self._p_disabled.stop)
        self.addCleanup(self._p_close.stop)
        self.addCleanup(self._p_getdev.stop)

    def test_flag_is_true_when_stream_opened_and_started(self):
        bc = self.bc
        seen = {"at_open": None, "at_start": None}

        class FakeStream:
            def __init__(_self, *a, **k):
                # Ownership MUST already be published by the time PortAudio is
                # handed a live callback — this is the TOCTOU window.
                seen["at_open"] = bc._record_speech_active[0]

            def start(_self):
                seen["at_start"] = bc._record_speech_active[0]

            def stop(_self):
                pass

            def close(_self):
                pass

        # Bail out of the capture loop immediately (first watchdog check) so we
        # exercise open -> start -> flag-set -> finally without real audio.
        bc._watchdog_reset_signal.set()
        with mock.patch.object(bc.sd, "InputStream", FakeStream):
            out = bc.record_speech(timeout=0.0)

        self.assertIsNone(out)  # watchdog-bail returns None
        self.assertIs(seen["at_open"], True,
                      "flag must be True BEFORE sd.InputStream is constructed")
        self.assertIs(seen["at_start"], True,
                      "flag must be True BEFORE stream.start()")
        # And the finally restored it so the next turn starts clean.
        self.assertFalse(bc._record_speech_active[0])

    def test_sample_rate_published_before_open(self):
        bc = self.bc
        seen = {"sr": None}

        class FakeStream:
            def __init__(_self, *a, **k):
                seen["sr"] = bc._record_speech_sr[0]

            def start(_self):
                pass

            def stop(_self):
                pass

            def close(_self):
                pass

        bc._watchdog_reset_signal.set()
        with mock.patch.object(bc.sd, "InputStream", FakeStream):
            bc.record_speech(timeout=0.0)
        self.assertEqual(seen["sr"], bc.SAMPLE_RATE,
                         "stream sample rate must be published before the open")

    def test_flag_cleared_when_open_raises(self):
        """If the InputStream open (incl. the system-default retry) fails, the
        ownership flag must NOT be left stuck True — otherwise _refresh_devices
        would defer the reinit forever against a stream that doesn't exist."""
        bc = self.bc

        # Both the first open and the device=None retry raise PortAudioError.
        def boom(*a, **k):
            raise bc.sd.PortAudioError("no such device")

        with mock.patch.object(bc.sd, "InputStream", side_effect=boom), \
                mock.patch("builtins.print"):
            out = bc.record_speech(timeout=0.0)

        self.assertIsNone(out)
        self.assertFalse(bc._record_speech_active[0],
                         "flag must be cleared after an open failure")

    def test_flag_cleared_when_start_raises(self):
        bc = self.bc

        class FakeStream:
            def __init__(_self, *a, **k):
                pass

            def start(_self):
                raise RuntimeError("start boom")

            def stop(_self):
                pass

            def close(_self):
                pass

        with mock.patch.object(bc.sd, "InputStream", FakeStream), \
                mock.patch("builtins.print"):
            out = bc.record_speech(timeout=0.0)

        self.assertIsNone(out)
        self.assertFalse(bc._record_speech_active[0],
                         "flag must be cleared after a start() failure")


@requires_monolith
class RefreshDevicesReinitGuardTests(MonolithGlobalsTestCase):
    """The other half of P1-4: _refresh_devices must DEFER the destructive
    PortAudio reinit while record_speech owns the mic (flag True). Paired with
    the ordering fix above, this is what makes a mid-capture teardown
    impossible."""

    def _run_refresh(self, *, active: bool):
        bc = self.bc
        terminated = {"called": False}

        def fake_terminate():
            terminated["called"] = True

        prev = bc._record_speech_active[0]
        bc._record_speech_active[0] = active
        try:
            with mock.patch.object(bc.sd, "_terminate", side_effect=fake_terminate), \
                    mock.patch.object(bc.sd, "_initialize"), \
                    mock.patch.object(bc.sd, "query_devices",
                                      return_value={"name": "FakeMic"}), \
                    mock.patch.object(bc, "_pick_device",
                                      return_value=(0, "FakeMic")), \
                    mock.patch.object(bc, "MICROPHONE_INDEX", None), \
                    mock.patch.object(bc, "SPEAKER_INDEX", None), \
                    mock.patch("builtins.print"):
                # force=True bypasses the time/signature short-circuits so the
                # flag guard is the only thing that can stop the reinit.
                bc._refresh_devices(force=True)
        finally:
            bc._record_speech_active[0] = prev
        return terminated["called"]

    def test_reinit_deferred_while_record_speech_active(self):
        self.assertFalse(
            self._run_refresh(active=True),
            "sd._terminate() must NOT run while record_speech owns the mic")

    def test_reinit_runs_when_mic_idle(self):
        # Control: with the flag clear, force=True DOES reinit — proving the
        # deferral above is the flag's doing, not an unrelated short-circuit.
        self.assertTrue(
            self._run_refresh(active=False),
            "sd._terminate() should run when no capture owns the mic")

    def test_reinit_deferred_while_ambient_stream_active(self):
        # ambient_listen's loopback/mic daemon holds a dedicated InputStream but
        # sets NONE of the record_speech/Path-B flags; _refresh_devices must defer
        # its destructive PortAudio reinit on the ambient ownership flag too, or it
        # tears PortAudio down under the live loopback callback (0xc0000374 heap
        # corruption, HIGH 2026-07-08).
        bc = self.bc
        terminated = {"called": False}
        prev = bc._ambient_stream_active[0]
        bc._ambient_stream_active[0] = True
        try:
            with mock.patch.object(bc.sd, "_terminate",
                                   side_effect=lambda: terminated.__setitem__("called", True)), \
                    mock.patch.object(bc.sd, "_initialize"), \
                    mock.patch.object(bc.sd, "query_devices",
                                      return_value={"name": "FakeMic"}), \
                    mock.patch.object(bc, "_pick_device", return_value=(0, "FakeMic")), \
                    mock.patch.object(bc, "MICROPHONE_INDEX", None), \
                    mock.patch.object(bc, "SPEAKER_INDEX", None), \
                    mock.patch("builtins.print"):
                bc._refresh_devices(force=True)
        finally:
            bc._ambient_stream_active[0] = prev
        self.assertFalse(
            terminated["called"],
            "sd._terminate() must NOT run while an ambient stream is live")


@requires_monolith
class GetMicBufferPathBExclusionTests(MonolithGlobalsTestCase):
    """get_mic_buffer Path B must NOT open a second InputStream on a device
    record_speech already owns (HIGH double-open capture stall, 2026-07-08).
    Path A2 only taps record_speech when the sample rates match; when they differ
    the code lands in Path B, which must yield the device instead of double-open."""

    def test_pathb_aborts_when_record_speech_active_at_other_rate(self):
        bc = self.bc
        opened = {"called": False}

        def _fake_stream(*a, **k):
            opened["called"] = True
            raise AssertionError("Path B opened a second InputStream over record_speech")

        prev_active = bc._record_speech_active[0]
        prev_sr = bc._record_speech_sr[0]
        # record_speech owns the mic at 48 kHz; caller wants 16 kHz → sr mismatch
        # skips the A2 tap and lands in Path B, which must bail.
        bc._record_speech_active[0] = True
        bc._record_speech_sr[0] = 48000
        try:
            with mock.patch.object(bc, "_mic_input_disabled", return_value=False), \
                    mock.patch.dict(bc.sys.modules, {}, clear=False), \
                    mock.patch.object(bc.sd, "InputStream", side_effect=_fake_stream), \
                    mock.patch.object(bc, "get_input_device", return_value=0), \
                    mock.patch("builtins.print"):
                # Ensure no wake-word detector tap is available (Path A1 skip).
                bc.sys.modules.pop("skill_wake_listener", None)
                out = bc.get_mic_buffer(0.1, sample_rate=16000)
        finally:
            bc._record_speech_active[0] = prev_active
            bc._record_speech_sr[0] = prev_sr
        self.assertIsNone(out)
        self.assertFalse(opened["called"],
                         "Path B must not open a stream while record_speech holds the mic")


@requires_monolith
class PlayWithLipsyncEndpointSwapTests(MonolithGlobalsTestCase):
    """Bug 5 (REVIEW_FINDINGS_2 P1-9): a default-render endpoint swap landing
    mid-open (DirectSound -9999) must NOT silently drop the utterance.
    play_with_lipsync must retry the failed sd.play once on the system default,
    and fail loud (propagate) only if that also fails.

    These drive the no-robot, no-barge-in, non-muted branch of play_with_lipsync
    with sd fully faked, so no real audio device is touched. A tiny zero buffer
    keeps audio_secs ~0 so the bounded sd.wait() join returns immediately.
    """

    class _FakeSd:
        """Stand-in for the sounddevice module: records every play(device=...)
        and lets the test program a sequence of side effects (an exception type
        raises, anything else is treated as a successful open). get_stream()
        returns an already-inactive fake stream so the single-toucher
        tts-reaper (_reap_playback) exits its poll loop immediately — the
        old wait()/stop() surface is gone from the playback path by design
        (2026-08 barge-in-stall fix)."""

        class PortAudioError(Exception):
            pass

        class _FakeStream:
            active = False

            def abort(self, ignore_errors=True):
                pass

            def stop(self, ignore_errors=True):
                pass

            def close(self, ignore_errors=True):
                pass

        def __init__(self, play_effects):
            # play_effects: list, one entry consumed per play() call. An entry
            # that is an Exception instance is raised; None means success.
            self._effects = list(play_effects)
            self.play_calls = []   # list of the `device` kwarg per call
            self._stream = self._FakeStream()

        def play(self, audio, sr, device=None):
            self.play_calls.append(device)
            effect = self._effects.pop(0) if self._effects else None
            if isinstance(effect, BaseException):
                raise effect

        def get_stream(self):
            return self._stream

    def _run_play(self, play_effects, *, out_dev=7):
        """Run play_with_lipsync with sd faked + out_dev pinned. Returns the
        _FakeSd so the test can inspect play_calls. Propagated exceptions are
        left to the caller (we assert on them)."""
        import numpy as np
        bc = self.bc
        fake_sd = self._FakeSd(play_effects)
        audio = np.zeros(8, dtype=np.float32)

        with mock.patch.object(bc, "sd", fake_sd), \
                mock.patch.object(bc, "get_output_device", return_value=out_dev), \
                mock.patch.object(bc, "ROBOT_ENABLED", False), \
                mock.patch.object(bc, "BARGE_IN_ENABLED", False), \
                mock.patch.object(bc, "_tts_layer", None), \
                mock.patch.object(bc, "_feed_playback_reference"), \
                mock.patch.object(bc, "_write_hud_state"), \
                mock.patch.object(bc, "_audio_ducker"), \
                mock.patch("builtins.print"):
            bc.play_with_lipsync(audio, 16000)
        return fake_sd

    def test_healthy_play_uses_resolved_device_once(self):
        # Control: no error -> exactly one play(), on the resolved device, no
        # fallback. Proves the retry path doesn't fire on the happy path.
        fake_sd = self._run_play([None], out_dev=7)
        self.assertEqual(fake_sd.play_calls, [7],
                         "healthy playback must open the resolved device exactly once")

    def test_minus_9999_recovers_on_system_default(self):
        # THE BUG: first open -9999s because the endpoint was swapped; the
        # utterance must be retried on device=None (the new default), not dropped.
        bc = self.bc
        # Seed a stale cached output index so we can prove it gets invalidated.
        bc._device_cache["out"] = 7
        bc._device_cache["checked_at"] = 1.0e12
        err = self._FakeSd.PortAudioError("DirectSound error [PaErrorCode -9999]")
        fake_sd = self._run_play([err, None], out_dev=7)

        self.assertEqual(
            fake_sd.play_calls, [7, None],
            "a -9999 on the resolved device must retry once on the system default")
        # The stale index must be invalidated so the next turn re-resolves fresh.
        self.assertIsNone(bc._device_cache["out"],
                          "cached output index must be cleared after the -9999 fallback")
        self.assertEqual(bc._device_cache["checked_at"], 0.0)

    def test_persistent_portaudio_error_propagates(self):
        # Fails loud, never silent: if BOTH the resolved device and the
        # system-default retry raise, the error must surface to _speak's existing
        # device-hiccup handler rather than being swallowed inside play.
        err1 = self._FakeSd.PortAudioError("first -9999")
        err2 = self._FakeSd.PortAudioError("default also gone")
        with self.assertRaises(self._FakeSd.PortAudioError):
            self._run_play([err1, err2], out_dev=7)


@requires_monolith
class InjectedTurnBypassesBgGateTests(MonolithGlobalsTestCase):
    """2026-07-02 live-test: with wake-word mode latched (_require_wake_runtime,
    e.g. after standby→tray force_wake, which deliberately does NOT clear it),
    every driver-injected command without a literal "Jarvis" prefix was dropped
    by the background-audio gate — the headless test/driver path could not
    drive the app at all. Injected commands are explicit local operator input,
    not overheard room audio, so the per-turn wrapper _bg_gate_for_turn must
    bypass the gate for them and leave real mic turns fully gated."""

    def test_injected_turn_never_gated(self):
        # Even in the maximal-refusal state (gate would say True), an injected
        # turn must pass. Make the underlying gate explode-if-called to prove
        # the bypass short-circuits BEFORE any gate logic runs.
        with mock.patch.object(
                self.bc, "_should_refuse_background_audio",
                side_effect=AssertionError("gate must not run for injects")):
            self.assertEqual(self.bc._bg_gate_for_turn("set a timer", True),
                             (False, ""))

    def test_mic_turn_still_delegates_to_gate(self):
        with mock.patch.object(
                self.bc, "_should_refuse_background_audio",
                return_value=(True, "wake-word mode")) as gate:
            self.assertEqual(
                self.bc._bg_gate_for_turn("set a timer", False),
                (True, "wake-word mode"))
        gate.assert_called_once_with("set a timer")


@requires_monolith
class TranscribeCudaLivelockTests(MonolithGlobalsTestCase):
    """2026-07-08 (finding #6): transcribe()'s CUDA/OOM recovery dropped _stt and
    reloaded the SAME over-budget GPU config → load→OOM→drop→reload livelock. A
    module counter now flips the sticky _force_whisper_cpu_int8 after 2 CONSECUTIVE
    CUDA failures so the next reload lands on crash-proof CPU int8; a clean
    transcribe resets the counter."""

    def setUp(self):
        # These two globals aren't in the harness restore set — snapshot + restore
        # them ourselves so this class can't leak the sticky CPU flag.
        bc = self.bc
        self._saved_fail = bc._consecutive_whisper_cuda_failures
        self._saved_flag = bc._force_whisper_cpu_int8
        bc._consecutive_whisper_cuda_failures = 0
        bc._force_whisper_cpu_int8 = False

        def _restore():
            bc._consecutive_whisper_cuda_failures = self._saved_fail
            bc._force_whisper_cpu_int8 = self._saved_flag
        self.addCleanup(_restore)

    @staticmethod
    def _audio():
        import numpy as np
        return np.zeros(16000, dtype=np.float32)

    def test_two_consecutive_cuda_failures_force_cpu_int8(self):
        bc = self.bc

        def _reload_failing():
            # _ensure_whisper normally repopulates _stt; simulate a reload that
            # produces a model which OOMs the moment it decodes.
            m = mock.Mock()
            m.transcribe.side_effect = RuntimeError("CUDA out of memory")
            bc._stt = m

        with mock.patch.object(bc, "_stt_engine", "faster_whisper"), \
             mock.patch.object(bc, "_ensure_whisper", side_effect=_reload_failing):
            bc.transcribe(self._audio())
            self.assertEqual(bc._consecutive_whisper_cuda_failures, 1)
            self.assertFalse(bc._force_whisper_cpu_int8,
                             "one failure must NOT yet force CPU")
            bc.transcribe(self._audio())
            self.assertGreaterEqual(bc._consecutive_whisper_cuda_failures, 2)
            self.assertTrue(bc._force_whisper_cpu_int8,
                            "2nd consecutive CUDA failure must flip the sticky "
                            "CPU-int8 flag to break the reload livelock")

    def test_clean_transcribe_resets_counter(self):
        bc = self.bc
        bc._consecutive_whisper_cuda_failures = 5   # pretend we'd been failing

        seg = mock.Mock(text="hello", no_speech_prob=0.1, avg_logprob=-0.2)
        info = mock.Mock(no_speech_prob=0.1)
        good = mock.Mock()
        good.transcribe.return_value = (iter([seg]), info)

        def _reload_good():
            bc._stt = good

        with mock.patch.object(bc, "_stt_engine", "faster_whisper"), \
             mock.patch.object(bc, "_ensure_whisper", side_effect=_reload_good):
            text, _conf = bc.transcribe(self._audio())

        self.assertEqual(text, "hello")
        self.assertEqual(bc._consecutive_whisper_cuda_failures, 0,
                         "a clean decode must reset the CUDA-failure counter")


@requires_monolith
class VlmCoLoadTrueFreeVramTests(MonolithGlobalsTestCase):
    """2026-07-08 (finding #7): the VLM co-load guard counted only Ollama-resident
    models and was blind to ~6 GB of whisper+chatterbox on cuda:0, so it could
    green-light a co-load that over-commits. The guard now ALSO consults true free
    VRAM (torch.cuda.mem_get_info / nvidia-smi) and refuses when free < needed +
    headroom, regardless of framework."""

    def _patch_vision_ready(self, free_mb):
        """Common patches: local-vision enabled + reachable + model present, and
        NO big Ollama model resident, with a stubbed free-VRAM probe."""
        bc = self.bc
        return [
            mock.patch.object(bc, "LOCAL_VISION_FALLBACK", True),
            mock.patch.object(bc, "LOCAL_VISION_MODEL", "llava:7b"),
            mock.patch.object(bc, "_ollama_alive", return_value=True),
            mock.patch.object(bc, "_ollama_has_model", return_value=True),
            mock.patch.object(bc, "_ollama_big_model_resident", return_value=None),
            mock.patch.object(bc, "_cuda0_free_vram_mb", return_value=free_mb),
            mock.patch.dict(os.environ, {}, clear=False),
        ]

    def test_low_free_vram_refuses_even_without_resident_ollama_model(self):
        bc = self.bc
        os.environ.pop("JARVIS_ALLOW_VLM_COLOAD", None)
        patches = self._patch_vision_ready(free_mb=3000)  # far below need+headroom
        with mock.patch.object(bc, "requests") as req, \
             patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6]:
            out = bc._call_local_vision("what's on screen?", [b"\x89PNG..."])
        self.assertIsNone(out, "must refuse when cuda:0 lacks true free VRAM")
        req.post.assert_not_called()

    def test_ample_free_vram_allows_coload(self):
        bc = self.bc
        os.environ.pop("JARVIS_ALLOW_VLM_COLOAD", None)
        resp = mock.Mock()
        resp.ok = True
        resp.json.return_value = {"message": {"content": "a tidy desk"}}
        patches = self._patch_vision_ready(free_mb=20000)  # plenty of headroom
        with mock.patch.object(bc, "requests") as req, \
             patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6]:
            req.post.return_value = resp
            req.RequestException = Exception
            out = bc._call_local_vision("what's on screen?", [b"\x89PNG..."])
        self.assertEqual(out, "a tidy desk")
        req.post.assert_called_once()

    def test_probe_unavailable_falls_back_to_ollama_check(self):
        # None (no torch, no nvidia-smi) must NOT block — preserves prior
        # behaviour on non-NVIDIA / no-torch boxes; the Ollama check is sole guard.
        bc = self.bc
        os.environ.pop("JARVIS_ALLOW_VLM_COLOAD", None)
        resp = mock.Mock()
        resp.ok = True
        resp.json.return_value = {"message": {"content": "a plant"}}
        patches = self._patch_vision_ready(free_mb=None)
        with mock.patch.object(bc, "requests") as req, \
             patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6]:
            req.post.return_value = resp
            req.RequestException = Exception
            out = bc._call_local_vision("what's on screen?", [b"\x89PNG..."])
        self.assertEqual(out, "a plant")
        req.post.assert_called_once()

    def test_cuda0_free_vram_mb_never_raises(self):
        # Best-effort probe: torch absent AND nvidia-smi absent → returns None,
        # never raises.
        bc = self.bc
        with mock.patch.dict("sys.modules", {"torch": None}):
            val = bc._cuda0_free_vram_mb()
        self.assertTrue(val is None or isinstance(val, int))


@requires_monolith
class CallLlmTrimsLeadingAssistantTests(MonolithGlobalsTestCase):
    """2026-07-08 (finding #11): the first post-boot turn could send a history that
    LEADS with an assistant message (boot-time / follow-up appends) → Claude 400 →
    the whole first turn degrades to local. _call_llm now runs
    _trim_conversation_history() right after appending the user turn, BEFORE the
    dispatch, so the request is always well-formed."""

    def test_leading_assistant_trimmed_before_dispatch(self):
        import core.config as cfg
        bc = self.bc
        bc.conversation_history[:] = [
            {"role": "assistant", "content": "Systems online, sir."},   # boot line
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier reply"},
        ]
        captured = {}

        def _spy(_sys_prompt, hist):
            captured["first_role"] = hist[0]["role"] if hist else None
            return "acknowledged"

        with mock.patch.object(cfg, "model_route", return_value="local"), \
             mock.patch.object(bc, "_local_then_cloud_or_honest", side_effect=_spy):
            bc._call_llm("hello there")

        self.assertEqual(captured.get("first_role"), "user",
                         "history sent to the model must NOT lead with assistant")


@requires_monolith
class LlmQuickOllamaBoundedTests(MonolithGlobalsTestCase):
    """2026-07-08 (finding #12): _llm_quick's ollama branch called ollama.chat with
    no timeout/try-except, so a wedged runner blocked the background one-shot
    forever. It now routes through _ollama_chat_bounded inside try/except and
    degrades to the local fallback (then "") like the claude branch."""

    def test_ollama_branch_uses_bounded_wrapper(self):
        bc = self.bc
        with mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "_ollama_chat_bounded",
                               return_value={"message": {"content": "bounded ok"}}) as b:
            out = bc._llm_quick("sys", "user")
        self.assertEqual(out, "bounded ok")
        b.assert_called_once()

    def test_wedged_ollama_degrades_to_local_fallback(self):
        bc = self.bc
        with mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "_ollama_chat_bounded",
                               side_effect=Exception("read timed out")), \
             mock.patch.object(bc, "_call_local_llm",
                               return_value="local fallback") as loc:
            out = bc._llm_quick("sys", "user")
        self.assertEqual(out, "local fallback")
        loc.assert_called_once()

    def test_wedged_ollama_and_no_local_returns_empty(self):
        bc = self.bc
        with mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "_ollama_chat_bounded",
                               side_effect=Exception("read timed out")), \
             mock.patch.object(bc, "_call_local_llm", return_value=""):
            out = bc._llm_quick("sys", "user")
        self.assertEqual(out, "")


# ════════════════════════════════════════════════════════════════════════════
#  2026-07-21 audit: speak-set family completeness (the stale-duplicate class)
# ════════════════════════════════════════════════════════════════════════════
#
# Both classes below guard the SAME recurring failure shape: a voicing rule
# applied to PART of an action family while sibling names silently rotted in
# neither speak set. Membership assertions pin today's fix; the source-scanning
# invariants (via tools/registration_scan.py — the one shared home for the
# "what does this file register?" rule) make the NEXT sibling added to the
# skill fail the suite instead of being silently dropped.

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_registration_scan():
    """Load tools/registration_scan.py without needing tools/ on sys.path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "registration_scan", os.path.join(_ROOT, "tools", "registration_scan.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires_monolith
class AirMouseFamilyVoicedTests(MonolithGlobalsTestCase):
    """2026-07-21 audit: air_mouse_on / air_mouse_off returned the finished
    gesture-vocabulary walkthrough (incl. the graceful "Note the Kinect is
    off" sensor note, which carries no FAILURE_MARKER by design) but were in
    NEITHER speak set — while their siblings air_mouse_status / arm / disarm /
    calibrate_air_mouse WERE voiced. The owner heard only the inline preamble
    and was never told the gesture vocabulary, nor that the Kinect was off and
    the feature would not actually work."""

    def test_air_mouse_on_off_in_verbatim_set(self):
        for name in ("air_mouse_on", "air_mouse_off"):
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} returns a finished answer that must be spoken")
            # Kept DISJOINT from INFORMATIVE (see test_speak_sets_are_disjoint).
            self.assertNotIn(name, self.bc.INFORMATIVE_ACTIONS)

    def test_every_registered_air_mouse_name_is_voiced(self):
        # Source-scanning family-completeness invariant: EVERY name
        # skills/kinect_air_mouse.py registers (direct assigns AND the
        # alias-tuple loops) is a verbatim-class one-liner speaker, so every
        # one of them must be in SPEAK_RESULT_VERBATIM_ACTIONS. Catches both
        # this regression and any FUTURE air-mouse action added to register()
        # without a voicing route — the exact "fixed in one copy, missed in
        # the sibling" failure the audit documented.
        rs = _load_registration_scan()
        regs = rs.scan_file(os.path.join(_ROOT, "skills", "kinect_air_mouse.py"))
        # Sanity: the scanner parsed both direct assigns and the alias loops
        # (13 names at HEAD); an empty/partial scan must not vacuously pass.
        self.assertIn("air_mouse_on", regs)
        self.assertIn("take_the_cursor", regs)      # from an alias-tuple loop
        self.assertGreaterEqual(len(regs), 13)
        missing = sorted(n for n in regs
                         if n not in self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertEqual(
            missing, [],
            f"kinect_air_mouse registers action(s) with NO voicing route — "
            f"their finished one-line results would be logged and dropped: "
            f"{missing}")


@requires_monolith
class HolographicOverlayStatusVoicedTests(MonolithGlobalsTestCase):
    """2026-07-21 audit: the ENTIRE holographic_overlay package's status
    read-outs were missed by the 2026-07-04 read-out sweep (its registrations
    live in the package __init__, not a flat skills/*.py). Each *_status
    handler returns one finished user-facing sentence and never self-speaks,
    so "is the printer overlay up?" was answered to the log only."""

    _STATUS_READOUTS = (
        "bambu_overlay_status", "bambu_camera_status", "workshop_hud_status",
        "workshop_print_monitor_status", "holo_hud_v2_status",
        "arc_reactor_status_status", "stark_status_ring_status",
        "holographic_status",
    )

    def test_holo_status_readouts_in_verbatim_set(self):
        for name in self._STATUS_READOUTS:
            self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                          f"{name} returns a finished answer that must be spoken")
            self.assertNotIn(name, self.bc.INFORMATIVE_ACTIONS)

    def test_every_holo_status_readout_is_routed(self):
        # Source-scanning invariant: every holographic_overlay registration
        # whose HANDLER is a *_status read-out must be in one of the two speak
        # sets, so a FUTURE status action added to the package fails the suite
        # instead of being silently dropped. Matching on the handler symbol
        # (not the registered name) is deliberate: the package also registers
        # the name "arc_reactor_status" as a TOGGLE alias (_act_arc_status_
        # toggle) — a side-effect action that must NOT be verbatim-voiced, and
        # a name-suffix match would wrongly demand it.
        rs = _load_registration_scan()
        regs = rs.scan_file(
            os.path.join(_ROOT, "skills", "holographic_overlay", "__init__.py"))
        status_names = sorted(n for n, r in regs.items()
                              if r.symbol.endswith("_status"))
        # Sanity: the full 8-read-out family is visible at HEAD; a partial
        # scan must not vacuously pass.
        self.assertGreaterEqual(len(status_names), 8, status_names)
        routed = (set(self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
                  | set(self.bc.INFORMATIVE_ACTIONS))
        missing = [n for n in status_names if n not in routed]
        self.assertEqual(
            missing, [],
            f"holographic_overlay status read-out(s) in NEITHER speak set — "
            f"their answers would be logged and dropped: {missing}")

    def test_speak_sets_still_disjoint_after_additions(self):
        # Re-assert the two-set disjointness invariant explicitly here so a
        # botched future edit to THIS family (adding a name to both sets)
        # fails next to the family tests, not only in the 2026-07-07 class.
        overlap = (set(self.bc.INFORMATIVE_ACTIONS)
                   & set(self.bc.SPEAK_RESULT_VERBATIM_ACTIONS))
        self.assertEqual(overlap, set(), sorted(overlap))


@requires_monolith
class PaTeardownGateTests(MonolithGlobalsTestCase):
    """2026-08-14 portaudio-teardown-race (0xc0000374): _refresh_devices'
    owner-flag check used to be a one-shot snapshot taken WITHOUT _mic_lock,
    while sd._terminate()/sd._initialize() are long native calls — an owner
    could claim AND open a stream inside that native window. The _pa_gate
    check-and-latch makes check+latch one atomic step under _mic_lock and
    makes claims (_pa_claim_owner) wait, bounded, while the latch is up, so
    teardown under a live/opening/closing guarded stream is structurally
    impossible. These pin the gate's mechanics plus the two release-order
    fixes (flag must cover the stream's native close)."""

    _REFRESH_PATCHES = None  # built per-test; see _refresh_ctx

    def _refresh_ctx(self, fake_terminate):
        """The RefreshDevicesReinitGuardTests mock harness: real bc.sd with
        _terminate/_initialize/query_devices patched, _pick_device stubbed,
        no explicit device indices, prints silenced."""
        bc = self.bc
        return [
            mock.patch.object(bc.sd, "_terminate", side_effect=fake_terminate),
            mock.patch.object(bc.sd, "_initialize"),
            mock.patch.object(bc.sd, "query_devices",
                              return_value={"name": "FakeMic"}),
            mock.patch.object(bc, "_pick_device", return_value=(0, "FakeMic")),
            mock.patch.object(bc, "MICROPHONE_INDEX", None),
            mock.patch.object(bc, "SPEAKER_INDEX", None),
            mock.patch("builtins.print"),
        ]

    def test_claim_blocked_then_released_by_latch(self):
        # A claim during a reinit must WAIT (bounded) and return False WITHOUT
        # setting the cell; once the latch drops (with notify_all) a fresh
        # claim succeeds immediately.
        bc = self.bc
        cell = [False]
        bc._pa_reinit_active[0] = True
        try:
            t0 = time.monotonic()
            ok = bc._pa_claim_owner(cell, timeout=0.2)
            dt = time.monotonic() - t0
        finally:
            with bc._pa_gate:
                bc._pa_reinit_active[0] = False
                bc._pa_gate.notify_all()
        self.assertFalse(ok, "claim must fail while the latch is up")
        self.assertFalse(cell[0], "a timed-out claim must NOT set the cell")
        self.assertGreaterEqual(dt, 0.15, "the wait must be real, not a poll-once")
        self.assertLess(dt, 2.0, "the wait must be bounded")
        self.assertTrue(bc._pa_claim_owner(cell, timeout=0.2),
                        "claim must succeed once the latch is clear")
        self.assertTrue(cell[0])

    def test_no_claim_can_interleave_into_native_window(self):
        # THE regression test for the check→terminate TOCTOU: while
        # sd._terminate is executing, the latch must be up and a concurrent
        # _pa_claim_owner must fail without flipping the flag — deterministic
        # proof that no owner can claim (and therefore open a stream) inside
        # the native window.
        bc = self.bc
        bc._device_cache["checked_at"] = 0.0
        result = {}

        def fake_terminate():
            result["latched"] = bc._pa_reinit_active[0]
            out = {}

            def _worker():
                out["ok"] = bc._pa_claim_owner(bc._record_speech_active,
                                               timeout=0.3)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=5.0)
            result["worker_done"] = not t.is_alive()
            result["claim_ok"] = out.get("ok")
            result["flag_after"] = bc._record_speech_active[0]

        patches = self._refresh_ctx(fake_terminate)
        for p in patches:
            p.start()
        try:
            bc._refresh_devices(force=True)
        finally:
            for p in patches:
                p.stop()

        self.assertIs(result.get("latched"), True,
                      "_pa_reinit_active must be latched while sd._terminate runs")
        self.assertIs(result.get("worker_done"), True,
                      "the concurrent claim must complete (bounded), not deadlock")
        self.assertIs(result.get("claim_ok"), False,
                      "no claim may land while sd._terminate executes")
        self.assertIs(result.get("flag_after"), False,
                      "the owner flag must stay False after the refused claim")
        # After the refresh: latch dropped, claims flow again.
        self.assertFalse(bc._pa_reinit_active[0])
        self.assertTrue(bc._pa_claim_owner(bc._record_speech_active, timeout=0.5))
        bc._record_speech_active[0] = False

    def test_reinit_never_latches_while_any_owner_live(self):
        # Every owner cell — including the NEW diag/enroll cells — must defer
        # the destructive reinit, and the latch must never be taken.
        bc = self.bc
        cells = ("_record_speech_active", "_pathb_mic_active",
                 "_ambient_stream_active", "_tts_playback_active",
                 "_diag_capture_active", "_enroll_capture_active")
        refcounts = ("_ambient_stream_active", "_diag_capture_active")
        for name in cells:
            with self.subTest(owner=name):
                cell = getattr(bc, name)
                prev = cell[0]
                cell[0] = 1 if name in refcounts else True
                terminated = {"called": False}
                patches = self._refresh_ctx(
                    lambda: terminated.__setitem__("called", True))
                for p in patches:
                    p.start()
                try:
                    bc._device_cache["checked_at"] = 0.0
                    bc._refresh_devices(force=True)
                finally:
                    for p in patches:
                        p.stop()
                    cell[0] = prev
                self.assertFalse(terminated["called"],
                                 f"sd._terminate must NOT run while {name} is live")
                self.assertFalse(bc._pa_reinit_active[0],
                                 f"latch must never be taken while {name} is live")

    def test_latch_cleared_when_terminate_raises(self):
        # A raising sd._terminate must still drop the latch (finally +
        # notify_all) — a stuck latch would time out every future claim
        # (JARVIS deaf AND mute).
        bc = self.bc
        bc._device_cache["checked_at"] = 0.0

        def boom():
            raise RuntimeError("portaudio terminate boom")

        patches = self._refresh_ctx(boom)
        for p in patches:
            p.start()
        try:
            bc._refresh_devices(force=True)
        finally:
            for p in patches:
                p.stop()
        self.assertFalse(bc._pa_reinit_active[0],
                         "latch must be cleared even when sd._terminate raises")
        self.assertTrue(bc._pa_claim_owner(bc._record_speech_active, timeout=0.5),
                        "claims must flow again after a failed reinit")
        bc._record_speech_active[0] = False

    def test_repick_path_never_latches_or_waits(self):
        # The cheap cleared-cache re-pick (sec2's 2026-07-21 scenario) must
        # never touch the latch, so concurrent claimants never wait on it.
        bc = self.bc
        sig = ((0, "Mic", 1, 0),)
        bc._device_cache.update({
            "in": None, "out": None, "checked_at": 0.0,
            "last_devices_signature": sig,
            "last_reenum_at": time.time(),   # periodic re-enum NOT due
        })
        latch_seen = []

        def _pick(prefs, want_input=True):
            latch_seen.append(bc._pa_reinit_active[0])
            return (3, "Mic X") if want_input else (5, "Spk Y")

        fake_sd = mock.Mock()
        with mock.patch.object(bc, "sd", fake_sd), \
                mock.patch.object(bc, "_devices_signature", return_value=sig), \
                mock.patch.object(bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(bc, "SPEAKER_INDEX", None), \
                mock.patch.object(bc, "_pick_device", side_effect=_pick), \
                mock.patch("builtins.print"):
            bc._refresh_devices(force=False)
        fake_sd._terminate.assert_not_called()
        self.assertTrue(latch_seen, "the re-pick must actually have run")
        self.assertFalse(any(latch_seen),
                         "the repick-only path must never latch the gate")
        # A zero-timeout claim succeeds — nobody waits behind a re-pick.
        cell = [False]
        self.assertTrue(bc._pa_claim_owner(cell, timeout=0))
        self.assertTrue(cell[0])

    def test_record_speech_flag_covers_close(self):
        # Release-order fix: record_speech's finally must CLOSE the stream
        # while _record_speech_active is still True (close-then-release), so a
        # reinit can never latch mid-close.
        bc = self.bc
        seen = {}

        def _capture_close(stream):
            seen["flag_at_close"] = bc._record_speech_active[0]

        class FakeStream:
            def __init__(_self, *a, **k):
                pass

            def start(_self):
                pass

        bc._watchdog_reset_signal.set()
        self.addCleanup(bc._watchdog_reset_signal.clear)
        with mock.patch.object(bc, "_mic_input_disabled", return_value=False), \
                mock.patch.object(bc, "get_input_device", return_value=0), \
                mock.patch.object(bc, "_safe_close_stream",
                                  side_effect=_capture_close), \
                mock.patch.object(bc.sd, "InputStream", FakeStream), \
                mock.patch("builtins.print"):
            out = bc.record_speech(timeout=0.0)
        self.assertIsNone(out)   # watchdog-bail returns None
        self.assertIs(seen.get("flag_at_close"), True,
                      "the stream must be closed BEFORE ownership is released")
        self.assertFalse(bc._record_speech_active[0],
                         "ownership must be released after the close")

    def test_record_speech_claim_timeout_returns_none_without_flag(self):
        # A reinit hung past the claim bound must skip the cycle: None back to
        # the main loop (treated as no-speech), no flag set, no stream opened.
        bc = self.bc
        bc._pa_reinit_active[0] = True
        try:
            with mock.patch.object(bc, "_mic_input_disabled",
                                   return_value=False), \
                    mock.patch.object(bc, "get_input_device", return_value=0), \
                    mock.patch.object(
                        bc.sd, "InputStream",
                        side_effect=AssertionError(
                            "no stream may open on a refused claim")), \
                    mock.patch.object(bc, "_pa_claim_owner",
                                      wraps=bc._pa_claim_owner) as claim, \
                    mock.patch("builtins.print"):
                # Real claim path against the held latch: waits its bounded
                # ~1s default, then refuses.
                out = bc.record_speech(timeout=0.0)
        finally:
            with bc._pa_gate:
                bc._pa_reinit_active[0] = False
                bc._pa_gate.notify_all()
        self.assertIsNone(out)
        claim.assert_called_once()
        self.assertFalse(bc._record_speech_active[0])

    def test_play_with_lipsync_flag_covers_barge_close(self):
        # Release-order fix: _tts_playback_active must still be True when the
        # barge-in stream's close runs, and False after return. (The old code
        # cleared the flag at the top of the finally, before the close — the
        # ordering its own comment claimed not to have.)
        import numpy as np
        bc = self.bc
        seen = {}
        barge = object()

        def _capture_close(stream):
            if stream is barge:
                seen["flag_at_barge_close"] = bc._tts_playback_active[0]

        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = True   # muted path: no sd.play
        audio = np.zeros(24, dtype=np.float32)
        with mock.patch.object(bc, "sd", fake_sd), \
                mock.patch.object(bc, "_tts_layer", fake_layer), \
                mock.patch.object(bc, "_audio_ducker", mock.Mock()), \
                mock.patch.object(bc, "BARGE_IN_ENABLED", True), \
                mock.patch.object(bc, "ROBOT_ENABLED", False), \
                mock.patch.object(bc, "is_using_headset", return_value=True), \
                mock.patch.object(bc, "_start_barge_in_listener",
                                  return_value=barge), \
                mock.patch.object(bc, "get_output_device", return_value=1), \
                mock.patch.object(bc, "_write_hud_state"), \
                mock.patch.object(bc, "_feed_playback_reference"), \
                mock.patch.object(bc, "_safe_close_stream",
                                  side_effect=_capture_close), \
                mock.patch("builtins.print"):
            bc.play_with_lipsync(audio, 24000)
        self.assertIs(seen.get("flag_at_barge_close"), True,
                      "barge-in close must run while TTS ownership is held")
        self.assertFalse(bc._tts_playback_active[0],
                         "TTS ownership must be released after the close")

    def test_play_with_lipsync_claim_timeout_raises_without_flag(self):
        # A hung reinit at playback time must lose the utterance LOUDLY
        # (RuntimeError into _speak's device-hiccup handler), never open the
        # barge-in/play streams, and never leave the flag set.
        import numpy as np
        bc = self.bc
        with mock.patch.object(bc, "_pa_claim_owner", return_value=False), \
                mock.patch.object(
                    bc, "get_output_device",
                    side_effect=AssertionError(
                        "playback setup must not run on a refused claim")), \
                mock.patch("builtins.print"):
            with self.assertRaises(RuntimeError):
                bc.play_with_lipsync(np.zeros(8, dtype=np.float32), 16000)
        self.assertFalse(bc._tts_playback_active[0])

    def test_pathb_deny_still_yields_and_does_not_claim(self):
        # The deny_if conversion must preserve the yield-to-record_speech rule
        # (GetMicBufferPathBExclusionTests semantics) AND leave the Path-B
        # flag unclaimed on the veto.
        bc = self.bc
        prev_active = bc._record_speech_active[0]
        prev_sr = bc._record_speech_sr[0]
        bc._record_speech_active[0] = True
        bc._record_speech_sr[0] = 48000     # sr mismatch → skips A2, lands in B
        try:
            with mock.patch.object(bc, "_mic_input_disabled",
                                   return_value=False), \
                    mock.patch.object(
                        bc.sd, "InputStream",
                        side_effect=AssertionError(
                            "Path B must not open over record_speech")), \
                    mock.patch.object(bc, "get_input_device", return_value=0), \
                    mock.patch("builtins.print"):
                bc.sys.modules.pop("skill_wake_listener", None)
                out = bc.get_mic_buffer(0.1, sample_rate=16000)
        finally:
            bc._record_speech_active[0] = prev_active
            bc._record_speech_sr[0] = prev_sr
        self.assertIsNone(out)
        self.assertFalse(bc._pathb_mic_active[0],
                         "a deny_if veto must not leave the Path-B flag claimed")


@requires_monolith
class PaAbandonedCloseGateTests(MonolithGlobalsTestCase):
    """H-6 (2026-08-20) — an ABANDONED native Pa_CloseStream must defer the
    destructive PortAudio reinit.

    Found by an adversarial pre-ship review of the 2026-08-14
    portaudio-teardown-race wave and independently verified. Both abandonable
    closes — _safe_close_stream's daemon and play_with_lipsync's tts-reaper —
    give the caller a BOUNDED wait and then ABANDON the daemon ("it dies with
    the process"). The caller's finally then cleared its owner flag
    UNCONDITIONALLY, directly beneath a comment asserting "the flag covers its
    whole native lifetime". But a caller timeout is POSITIVE EVIDENCE that the
    daemon is still inside PortAudio's stop/close, so that flag drop handed
    _refresh_devices permission to run sd._terminate()/sd._initialize()
    straight into a live native call — re-opening the exact 0xc0000374
    heap-corruption window the gate was built to close.

    The fix must NOT freeze the owner flags: the barge-in gate, ambient
    listen, the face tracker, the dossier and the self-diagnostic probe all
    read them and would stall forever on a lie. So the in-flight native close
    gets its OWN count, _pa_close_pending — bumped by the CALLER
    (_pa_close_handoff) while its own owner flag is still up, retired by the
    DAEMON (_pa_close_done) after its native close returns.
    """

    # ---- harness ---------------------------------------------------------
    def _refresh_ctx(self, fake_terminate, printed):
        """RefreshDevicesReinitGuardTests' mock harness, with prints captured
        so the deny REASON can be asserted (each deny branch has its own
        line — a shared message would hide which rule fired)."""
        bc = self.bc
        return [
            mock.patch.object(bc.sd, "_terminate", side_effect=fake_terminate),
            mock.patch.object(bc.sd, "_initialize"),
            mock.patch.object(bc.sd, "query_devices",
                              return_value={"name": "FakeMic"}),
            mock.patch.object(bc, "MICROPHONE_INDEX", None),
            mock.patch.object(bc, "SPEAKER_INDEX", None),
            mock.patch("builtins.print",
                       side_effect=lambda *a, **k: printed.append(
                           " ".join(str(x) for x in a))),
        ]

    def _run_refresh(self, picks=None):
        """Drive one forced _refresh_devices pass. Returns
        (terminate_called, printed_lines, pick_device_mock)."""
        bc = self.bc
        printed = []
        terminated = {"called": False}
        pick = mock.Mock(return_value=(0, "FakeMic"))
        patches = self._refresh_ctx(
            lambda: terminated.__setitem__("called", True), printed)
        patches.append(mock.patch.object(bc, "_pick_device", pick))
        for p in patches:
            p.start()
        try:
            bc._device_cache["checked_at"] = 0.0
            bc._refresh_devices(force=True)
        finally:
            for p in patches:
                p.stop()
        return terminated["called"], printed, pick

    @staticmethod
    def _wait_zero(cell, deadline_s=5.0):
        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            if cell[0] == 0:
                return True
            time.sleep(0.01)
        return False

    # ---- the gate itself -------------------------------------------------
    def test_abandoned_close_defers_reinit_with_its_own_reason(self):
        # THE H-6 regression. Every owner flag is down (they must be — the
        # callers legitimately released them) yet a native close is still in
        # flight: the destructive reinit must NOT run.
        bc = self.bc
        with bc._mic_lock:
            bc._pa_close_pending[0] = 1
        terminated, printed, _ = self._run_refresh()
        self.assertFalse(terminated,
                         "sd._terminate must NOT run while an abandoned "
                         "native close is still inside PortAudio")
        self.assertFalse(bc._pa_reinit_active[0],
                         "the reinit latch must never be taken on the "
                         "deferred path")
        blob = "\n".join(printed)
        self.assertIn("an abandoned native close is still in flight", blob)
        self.assertIn("deferring PortAudio reinit", blob)

    def test_cheap_repick_still_runs_while_the_reinit_is_deferred(self):
        # ACCEPTED DEGRADATION boundary: deferring costs HOTPLUG DISCOVERY
        # only. The owner now follows the Windows default from a Stream Deck,
        # so switching between ALREADY-ENUMERATED devices happens constantly
        # and must keep working — that is the cheap _pick_device re-pick over
        # the existing enumeration, which runs after the deny chain.
        bc = self.bc
        with bc._mic_lock:
            bc._pa_close_pending[0] = 1
        bc._device_cache["last_reenum_at"] = 0.0
        terminated, _, pick = self._run_refresh()
        self.assertFalse(terminated)
        self.assertEqual(pick.call_count, 2,
                         "the input AND output re-picks must still run while "
                         "the destructive reinit is deferred")
        self.assertGreater(bc._device_cache["checked_at"], 0.0,
                           "a deferred pass must still refresh the cache")
        self.assertEqual(bc._device_cache["last_reenum_at"], 0.0,
                         "a DEFERRED pass must not re-arm the periodic "
                         "hotplug sweep — it has to retry soon, not in "
                         "another DEVICE_REENUM_INTERVAL")

    def test_reinit_runs_again_once_the_count_is_retired(self):
        # The deferral is not sticky: with the count back at 0 the very same
        # refresh performs the destructive reinit.
        bc = self.bc
        with bc._mic_lock:
            bc._pa_close_pending[0] = 0
        terminated, _, _ = self._run_refresh()
        self.assertTrue(terminated,
                        "with no close in flight the reinit must proceed")

    def test_pa_streams_live_lists_the_new_cell(self):
        # The canonical owner list must name it, so the ONE written-down
        # definition of "a stream is live" stays honest even though
        # _refresh_devices spells its checks out inline.
        bc = self.bc
        with bc._mic_lock:
            self.assertFalse(bc._pa_streams_live())
            bc._pa_close_pending[0] = 1
            self.assertTrue(bc._pa_streams_live(),
                            "_pa_streams_live must count an in-flight "
                            "abandoned close as a live stream owner")

    def test_close_pending_does_not_deny_new_claims(self):
        # DELIBERATE SCOPE: the count gates only the DESTRUCTIVE reinit. A
        # wedged close on one stream must not make JARVIS deaf by refusing
        # every new capture claim — the mic/TTS flags already permit opening
        # a new stream while another one is closing.
        bc = self.bc
        with bc._mic_lock:
            bc._pa_close_pending[0] = 2
        cell = [False]
        self.assertTrue(bc._pa_claim_owner(cell, timeout=0),
                        "an in-flight close must not block unrelated claims")
        self.assertTrue(cell[0])
        bc._pa_release_owner(cell)

    # ---- the hand-off pair ----------------------------------------------
    def test_handoff_defers_and_a_late_daemon_releases_it(self):
        # THE property that makes the degradation narrow: the count is keyed
        # on the daemon still being inside its native call. A daemon that
        # merely finishes LATE retires the count itself — no timer, no
        # liveness sweep, no window where the gate guesses.
        bc = self.bc
        release = threading.Event()
        self.addCleanup(release.set)

        def _wedged():
            release.wait(10.0)
            bc._pa_close_done()

        t = threading.Thread(target=_wedged, daemon=True)
        bc._pa_close_handoff(t)
        self.assertEqual(bc._pa_close_pending[0], 1)
        terminated, _, _ = self._run_refresh()
        self.assertFalse(terminated, "a live hand-off must defer the reinit")

        release.set()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        self.assertTrue(self._wait_zero(bc._pa_close_pending),
                        "a late-finishing daemon must retire its own count")
        terminated, _, _ = self._run_refresh()
        self.assertTrue(terminated,
                        "the reinit must resume once the native close returns")

    def test_handoff_retires_when_the_thread_cannot_start(self):
        # Thread exhaustion under load is real here (it is why _speak keeps a
        # belt-and-braces _tts_playback_active clear). A hand-off whose thread
        # never STARTS registered a close that will never happen — that would
        # defer hotplug for the life of the process.
        bc = self.bc

        class _DeadThread:
            def start(self):
                raise RuntimeError("can't start new thread")

        with self.assertRaises(RuntimeError):
            bc._pa_close_handoff(_DeadThread())
        self.assertEqual(bc._pa_close_pending[0], 0,
                         "a failed start must retire the registration")

    def test_count_never_goes_negative(self):
        # A negative count would read as falsy and turn the deferral into a
        # permanent PASS — strictly worse than the bug being fixed.
        bc = self.bc
        for _ in range(5):
            bc._pa_close_done()
        self.assertEqual(bc._pa_close_pending[0], 0)
        with bc._mic_lock:
            bc._pa_close_pending[0] = 1
        bc._pa_close_done()
        bc._pa_close_done()
        self.assertEqual(bc._pa_close_pending[0], 0)

    # ---- _safe_close_stream (record_speech + barge_stream abandons) ------
    def test_safe_close_stream_abandon_holds_the_count_until_the_daemon_returns(self):
        # _safe_close_stream is the ONE edit that covers record_speech's
        # abandon AND play_with_lipsync's barge_stream abandon.
        bc = self.bc
        release = threading.Event()
        self.addCleanup(release.set)
        stream = mock.Mock()
        stream.close.side_effect = lambda: release.wait(10.0)
        fake_sd = mock.Mock()
        with mock.patch.object(bc, "sd", fake_sd), \
                mock.patch("builtins.print"), \
                self.assertLogs("root", level="WARNING") as cm:
            bc._safe_close_stream(stream, timeout_sec=0.05)
        # It really was abandoned. H-3 (2026-08-20) removed the sd.stop() this
        # used to assert on — that call could not free the hung InputStream
        # (module-level stop() only touches sounddevice's _last_callback) and
        # cross-closed the live TTS play stream — so the abandon is proved by
        # the warning it now logs instead.
        self.assertIn("abandoning the close daemon", "\n".join(cm.output))
        fake_sd.stop.assert_not_called()
        self.assertEqual(bc._pa_close_pending[0], 1,
                         "an abandoned close must stay registered")
        terminated, _, _ = self._run_refresh()
        self.assertFalse(terminated,
                         "the reinit must be deferred while the abandoned "
                         "stream.close is still executing")
        release.set()
        self.assertTrue(self._wait_zero(bc._pa_close_pending))
        terminated, _, _ = self._run_refresh()
        self.assertTrue(terminated)

    def test_safe_close_stream_normal_path_leaves_no_phantom_count(self):
        # The retire happens BEFORE the daemon sets the caller's done-event,
        # so a caller whose bounded wait SUCCEEDS observes a clean count the
        # instant it returns — no transient deferral on every utterance.
        bc = self.bc
        stream = mock.Mock()
        with mock.patch("builtins.print"):
            bc._safe_close_stream(stream, timeout_sec=2.0)
        stream.close.assert_called_once()
        self.assertEqual(bc._pa_close_pending[0], 0,
                         "a completed close must retire before waking the "
                         "caller, or every healthy turn would defer hotplug")
        terminated, _, _ = self._run_refresh()
        self.assertTrue(terminated)

    # ---- the tts-reaper --------------------------------------------------
    def test_reap_playback_retires_before_setting_the_done_event(self):
        # Ordering proof for the reaper half: _pa_close_done() must run BEFORE
        # done_evt.set(), else the caller could wake and immediately see a
        # phantom count.
        bc = self.bc
        seen = {}

        class _RecordingEvent(threading.Event):
            def set(self):
                seen["count_at_set"] = bc._pa_close_pending[0]
                super().set()

        stream = mock.Mock()
        stream.active = False
        with bc._mic_lock:
            bc._pa_close_pending[0] = 1
        evt = _RecordingEvent()
        with mock.patch("builtins.print"):
            bc._reap_playback(stream, evt, 0.0)
        self.assertTrue(evt.is_set())
        self.assertEqual(seen.get("count_at_set"), 0,
                         "_reap_playback must retire the count before it "
                         "signals the caller")
        self.assertEqual(bc._pa_close_pending[0], 0)

    def _lipsync_ctx(self, fake_sd, robot=False):
        bc = self.bc
        layer = mock.Mock()
        layer.is_muted.return_value = False     # exercise the REAL play path
        return [
            mock.patch.object(bc, "sd", fake_sd),
            mock.patch.object(bc, "_tts_layer", layer),
            mock.patch.object(bc, "_audio_ducker", mock.Mock()),
            mock.patch.object(bc, "BARGE_IN_ENABLED", False),
            mock.patch.object(bc, "ROBOT_ENABLED", robot),
            mock.patch.object(bc, "send", mock.Mock()),
            mock.patch.object(bc, "get_output_device", return_value=1),
            mock.patch.object(bc, "_write_hud_state"),
            mock.patch.object(bc, "_feed_playback_reference"),
            mock.patch("builtins.print"),
        ]

    def _healthy_playback(self, robot):
        """Run one healthy play_with_lipsync through the REAL reaper (the fake
        stream reports inactive, so it finishes immediately). Returns the
        _pa_close_handoff spy."""
        import numpy as np
        bc = self.bc
        stream = mock.Mock()
        stream.active = False
        fake_sd = mock.Mock()
        fake_sd.get_stream.return_value = stream
        spy = mock.Mock(wraps=bc._pa_close_handoff)
        patches = self._lipsync_ctx(fake_sd, robot=robot)
        patches.append(mock.patch.object(bc, "_pa_close_handoff", spy))
        for p in patches:
            p.start()
        try:
            bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        finally:
            for p in patches:
                p.stop()
        return spy

    def test_play_with_lipsync_hands_the_reaper_through_the_gate(self):
        bc = self.bc
        spy = self._healthy_playback(robot=False)
        spy.assert_called_once()
        handed = spy.call_args[0][0]
        self.assertEqual(getattr(handed, "name", None), "tts-reaper")
        self.assertEqual(bc._pa_close_pending[0], 0)
        self.assertFalse(bc._tts_playback_active[0])

    def test_play_with_lipsync_robot_twin_hands_the_reaper_through_the_gate(self):
        # The robot branch is a deliberate SHARED body, not a divergent copy —
        # the stale-duplicate rule this codebase keeps paying for.
        bc = self.bc
        spy = self._healthy_playback(robot=True)
        spy.assert_called_once()
        self.assertEqual(getattr(spy.call_args[0][0], "name", None),
                         "tts-reaper")
        self.assertEqual(bc._pa_close_pending[0], 0)
        self.assertFalse(bc._tts_playback_active[0])

    def test_abandoned_reaper_defers_reinit_and_still_clears_the_tts_flag(self):
        # END-TO-END H-6. A wedged tts-reaper: the caller's bounded wait
        # expires, it abandons the daemon and runs its finally.
        #   * _tts_playback_active MUST still be cleared (it must not regress
        #     to lying — the barge-in gate, ambient listen and the face
        #     tracker read it and would stall forever).
        #   * _pa_close_pending MUST still be up, so the destructive reinit is
        #     deferred while PortAudio is inside the close.
        import numpy as np
        bc = self.bc
        release = threading.Event()
        self.addCleanup(release.set)
        entered = threading.Event()

        def _wedged_reaper(stream, done_evt, audio_secs):
            # Models a daemon stuck inside Pa_CloseStream: it neither retires
            # the count nor signals the caller.
            entered.set()
            release.wait(15.0)
            bc._pa_close_done()
            done_evt.set()

        # Only the caller's ~6 s abandon wait is shortened; every other wait
        # (the amp pump's 0.2 s join, the barge watcher's 0.02 s poll) is left
        # exactly as the code sets it.
        _RealEvent = threading.Event

        class _NoLongWaitEvent(_RealEvent):
            def wait(self, timeout=None):
                if timeout is not None and timeout > 1.0:
                    return _RealEvent.wait(self, 0.05)
                return _RealEvent.wait(self, timeout)

        stream = mock.Mock()
        fake_sd = mock.Mock()
        fake_sd.get_stream.return_value = stream
        patches = self._lipsync_ctx(fake_sd, robot=False)
        patches.append(mock.patch.object(bc, "_reap_playback",
                                         side_effect=_wedged_reaper))
        patches.append(mock.patch.object(bc.threading, "Event",
                                         _NoLongWaitEvent))
        for p in patches:
            p.start()
        try:
            bc.play_with_lipsync(np.zeros(240, dtype=np.float32), 24000)
        finally:
            for p in patches:
                p.stop()

        self.assertTrue(entered.wait(5.0), "the reaper must have started")
        self.assertFalse(bc._tts_playback_active[0],
                         "the TTS owner flag must STILL be cleared on abandon "
                         "— it must not regress to lying to the barge-in "
                         "gate / ambient listen / face tracker")
        self.assertEqual(bc._pa_close_pending[0], 1,
                         "the abandoned native close must still be registered")
        terminated, printed, _ = self._run_refresh()
        self.assertFalse(terminated,
                         "H-6: sd._terminate must NOT run while the abandoned "
                         "tts-reaper is inside PortAudio")
        self.assertIn("an abandoned native close is still in flight",
                      "\n".join(printed))

        release.set()
        self.assertTrue(self._wait_zero(bc._pa_close_pending))
        terminated, _, _ = self._run_refresh()
        self.assertTrue(terminated,
                        "hotplug discovery must return once the native close "
                        "actually completes")


@requires_monolith
class PaCloseGateStaleDuplicateTests(MonolithGlobalsTestCase):
    """The #1 bug class here is a rule fixed in ONE copy while the others rot.
    core/wake_word.py carries its own _safe_close_stream that abandons a
    native close the same way — and _refresh_devices calls det.pause() (which
    lands in that copy) IMMEDIATELY BEFORE its own sd._terminate(). Pin both
    the source-level mirror and the wiring."""

    def test_wake_word_close_copy_registers_with_the_host_gate(self):
        import core.wake_word as ww
        import inspect
        src = inspect.getsource(ww._safe_close_stream)
        self.assertIn("_pa_close_handoff", src,
                      "core/wake_word._safe_close_stream abandons a native "
                      "close too — it must register with the host's "
                      "_pa_close_pending gate or _refresh_devices will "
                      "sd._terminate() straight through its own det.pause()")
        self.assertIn("_pa_close_done", src)

    def test_wake_word_close_defers_the_hosts_reinit(self):
        # Functional half: with the monolith in sys.modules, a wedged
        # wake-word close must hold the host's count up.
        import core.wake_word as ww
        bc = self.bc
        release = threading.Event()
        self.addCleanup(release.set)

        class _Stream:
            def stop(self):
                pass

            def close(self):
                release.wait(10.0)

        with mock.patch.dict(ww.sys.modules,
                             {"bobert_companion": bc}, clear=False), \
                mock.patch("builtins.print"):
            ww._safe_close_stream(_Stream(), timeout_sec=0.05)
        self.assertEqual(bc._pa_close_pending[0], 1,
                         "an abandoned wake-word close must defer the host's "
                         "PortAudio reinit")
        release.set()
        end = time.monotonic() + 5.0
        while time.monotonic() < end and bc._pa_close_pending[0]:
            time.sleep(0.01)
        self.assertEqual(bc._pa_close_pending[0], 0,
                         "the wake-word daemon must retire its own count")

    def test_self_diagnostic_mirror_lists_every_canonical_cell(self):
        # skills/self_diagnostic._mic_owned mirrors _pa_streams_live minus the
        # probe's own _diag_capture_active claim. Derive the canonical list
        # from the monolith SOURCE so a new cell can never be added on one
        # side only.
        import inspect
        import re
        bc = self.bc
        canon = (inspect.getsource(bc._pa_streams_live)
                 + inspect.getsource(bc._pa_mic_capture_live))
        cells = {m for m in re.findall(r"_[a-z_]+_(?:active|pending)\b", canon)}
        cells.discard("_diag_capture_active")   # documented exception
        cells.discard("_pa_reinit_active")
        self.assertIn("_pa_close_pending", cells,
                      "the canonical owner list must name the H-6 cell")
        sd_path = os.path.join(os.path.dirname(os.path.dirname(bc.__file__)),
                               "JARVIS", "skills", "self_diagnostic.py")
        if not os.path.isfile(sd_path):
            sd_path = os.path.join(os.path.dirname(bc.__file__),
                                   "skills", "self_diagnostic.py")
        with io.open(sd_path, encoding="utf-8") as fh:
            probe_src = fh.read()
        start = probe_src.index("def _mic_owned()")
        # Slice the REAL function extent, not a fixed character window: the
        # old `start + 1500` proxy broke the moment a comment was added inside
        # _mic_owned (H-4, 2026-08-20), reporting a mirror drift that did not
        # exist. The body ends at the next line indented 4 spaces (the probe's
        # own scope) that is not blank.
        rest = probe_src[start:]
        end = len(rest)
        for m in re.finditer(r"\n {4}(?=\S)", rest):
            end = m.start()
            break
        block = rest[:end]
        missing = sorted(c for c in cells if c not in block)
        self.assertEqual(
            missing, [],
            f"skills/self_diagnostic._mic_owned no longer mirrors "
            f"bobert_companion's canonical owner list: {missing}")


# ═══════════════════════════════════════════════════════════════════════════
#  H-5 (2026-08-20) — focus mode's ALIAS names were in neither speak set
#
#  skills/focus_mode.py:397-398 binds "focus_mode" -> focus_mode_on and
#  "end_focus_mode" -> focus_mode_off, the SAME handlers as the routed
#  focus_mode_on / focus_mode_off names. Nothing on the path canonicalises an
#  alias: parse_and_run_actions records the RAW name and
#  _speak_verbatim_results gates on `name.lower() in
#  SPEAK_RESULT_VERBATIM_ACTIONS`. So "JARVIS, end focus mode" ran
#  focus_mode_off, which calls _build_recap(clear=True) — DESTROYING the held
#  items — and returned the recap into a path with no speaker. The owner heard
#  nothing and the "what you missed" list was gone; whats_missed cannot
#  recover it, because the buffer is already cleared.
#
#  WHY NOT CANONICALISE BY HANDLER IDENTITY (the tempting general fix — make
#  an alias inherit its sibling's speak-set membership)? Measured against the
#  tree at HEAD it is WRONG twice over:
#    * skills/network_deco.py:883-884 registers ONE handler,
#      _act_deco_topology, under two names that are DELIBERATELY split —
#      "deco_topology" is verbatim (a one-line status) and "network_topology"
#      is informative (a multi-device list the LLM should summarise), with the
#      split spelled out in bobert_companion's INFORMATIVE_ACTIONS comment.
#      Handler-identity canonicalisation would silently collapse that.
#    * skills/personal_rag.py registered "search_my_files" (verbatim) and
#      "rag_search_quiet" (silent) on the SAME machine-readable handler.
#      Canonicalising by handler would have PROPAGATED that pairing rather
#      than exposing it; the real defect was the BINDING, and it is now fixed
#      by binding the spoken name to the spoken formatter (2026-08-20), which
#      is a per-name decision canonicalisation could not have made.
#  The membership decision is legitimately per-NAME. What must not be silent
#  is DRIFT — so the anti-drift mechanism is a source scan, below.
# ═══════════════════════════════════════════════════════════════════════════
@requires_monolith
class FocusModeFamilyVoicedTests(MonolithGlobalsTestCase):
    def test_focus_mode_aliases_are_voiced(self):
        for name in ("focus_mode", "end_focus_mode"):
            self.assertIn(
                name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                f"{name} is a registered alias of the focus-mode handlers; "
                f"unvoiced, ending a block destroys the held-items recap and "
                f"says nothing")
            self.assertNotIn(name, self.bc.INFORMATIVE_ACTIONS)

    def test_every_registered_focus_mode_name_is_voiced(self):
        # Source-scanning family invariant, mirroring
        # test_every_registered_air_mouse_name_is_voiced: every name
        # skills/focus_mode.py registers returns a finished, user-facing
        # one-liner (the on/off confirmations, the resume RECAP, the
        # whats_missed / focus_mode_status queries), so every one of them must
        # be in SPEAK_RESULT_VERBATIM_ACTIONS. A FUTURE alias added to
        # register() now fails the suite instead of silently swallowing a
        # recap — which is exactly how "end_focus_mode" rotted.
        rs = _load_registration_scan()
        regs = rs.scan_file(os.path.join(_ROOT, "skills", "focus_mode.py"))
        # Sanity: 9 names at HEAD; a partial scan must not vacuously pass.
        self.assertIn("focus_mode_on", regs)
        self.assertIn("end_focus_mode", regs)
        self.assertGreaterEqual(len(regs), 9, sorted(regs))
        missing = sorted(n for n in regs
                         if n not in self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertEqual(
            missing, [],
            f"skills/focus_mode.py registers action(s) with NO voicing route "
            f"— their finished one-line results (and, for the disengage "
            f"aliases, the destroyed recap) would be dropped: {missing}")

    def test_disengage_aliases_share_the_recap_destroying_handler(self):
        # The premise the fix rests on, asserted from source rather than
        # assumed: end_focus_mode really is bound to focus_mode_off, which
        # really does clear the buffer. If a future edit gives it its own
        # self-speaking handler this test says so instead of silently passing.
        rs = _load_registration_scan()
        regs = rs.scan_file(os.path.join(_ROOT, "skills", "focus_mode.py"))
        self.assertEqual(regs["end_focus_mode"].symbol, "focus_mode_off")
        self.assertEqual(regs["focus_mode"].symbol, "focus_mode_on")
        with io.open(os.path.join(_ROOT, "skills", "focus_mode.py"),
                     encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def focus_mode_off"):]
        body = body[:body.index("\n    def ")] if "\n    def " in body else body
        self.assertIn("clear=True", body,
                      "focus_mode_off no longer destroys the held items — "
                      "re-check whether the unvoiced-alias harm still applies")


@requires_monolith
class ActionAliasSpeakSetDriftTests(MonolithGlobalsTestCase):
    """Repo-wide anti-drift scan for the H-5 class.

    Groups every registered action by (file, handler symbol) and flags any
    family where SOME names are routed to a speak set and others are in
    NEITHER. Membership is legitimately per-name (see the block comment
    above), so this does not demand uniformity — it demands that every
    non-uniform family be DECLARED here, with why. A new alias added to a
    routed handler fails this test instead of silently going mute.
    """

    # (relative path, handler symbol) -> why this family is legitimately
    # non-uniform, or the open finding that tracks it. Fix a finding, delete
    # its row. Every row was verified against the tree on 2026-08-20.
    #
    # A name is "routed" if it is in one of the monolith's two speak sets OR if
    # its own skill DECLARES it (module-level SPEAK_VERBATIM_ACTIONS /
    # INFORMATIVE_ACTIONS, folded in at load time by
    # bobert_companion._collect_skill_speak_sets). The declaration route exists
    # so a GITIGNORED personal skill never has to spell its private action names
    # out in tracked source; without teaching the scan about it, moving a name
    # there would look like the name went mute.
    _DECLARED_SPLITS = {
        ("bobert_companion.py", "_act_read_changelog"):
            "OPEN FINDING: 'recent_changes' is registered one line below three "
            "routed aliases of the same handler and documented alongside them, "
            "but is in neither speak set.",
        ("bobert_companion.py", "_act_web_search"):
            "'search' is a bare-verb alias of the informative 'web_search'; "
            "left unrouted deliberately so a naked 'search' does not force a "
            "follow-up round-trip.",
        ("skills/sh_kasa.py", "smart_home_control"):
            "control_light / control_plug / kasa_control are side-effect "
            "control aliases; the routed smart_home_control / control_device "
            "pair carries the spoken confirmation for the family.",
        ("skills/trip_planner.py", "_action_status"):
            # Names elided on purpose: this skill is gitignored precisely
            # because its action names embed a one-off personal event, and this
            # file is TRACKED and public. See the privacy note on `sweep`.
            "the *_brief alias of this status handler returns the long "
            "multi-paragraph briefing, not the one-line read-out, so it is "
            "deliberately not verbatim-voiced.",
    }

    @staticmethod
    def _declared_speak_names(path):
        """Names a skill routes itself, read from SOURCE (no import).

        Mirrors bobert_companion._collect_skill_speak_sets: a module-level
        SPEAK_VERBATIM_ACTIONS / INFORMATIVE_ACTIONS iterable of string
        literals. Source-read rather than imported so a gitignored personal
        skill is never executed by the suite.
        """
        import ast
        out = set()
        try:
            with io.open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError):
            return out
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if not names & {"SPEAK_VERBATIM_ACTIONS", "INFORMATIVE_ACTIONS"}:
                continue
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
                for elt in value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
        return out

    def test_no_undeclared_alias_family_drifts_out_of_the_speak_sets(self):
        rs = _load_registration_scan()
        routed = (set(self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
                  | set(self.bc.INFORMATIVE_ACTIONS))
        # Sanity: an empty/partial speak set must not make this vacuous.
        self.assertGreater(len(routed), 300, len(routed))

        paths = [os.path.join(_ROOT, "bobert_companion.py")]
        for sub in ("skills", "core"):
            d = os.path.join(_ROOT, sub)
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".py"):
                    paths.append(os.path.join(d, fname))

        # Skills that route their own names count as routed (see the header
        # note). load_skills() has not run under the import-only harness, so
        # without this the declaration route would read as a regression.
        for path in paths:
            routed |= self._declared_speak_names(path)

        found = {}
        for path in paths:
            rel = os.path.relpath(path, _ROOT).replace(os.sep, "/")
            try:
                regs = rs.scan_file(path)
            except SyntaxError:
                continue
            by_symbol = {}
            for name, r in regs.items():
                if r.symbol.startswith(rs.OPAQUE_SYMBOL_PREFIXES):
                    continue     # lambda@ / expr@ / ?alias: — no real family
                by_symbol.setdefault(r.symbol, []).append(name)
            for symbol, names in by_symbol.items():
                inset = [n for n in names if n in routed]
                out = sorted(n for n in names if n not in routed)
                if inset and out:
                    found[(rel, symbol)] = out

        # Sanity: the scan must actually see the tree. Anchored on a
        # DELIBERATE split (a bare-verb alias left unrouted on purpose) rather
        # than an open finding, so closing a finding cannot make the whole
        # scan vacuously pass.
        self.assertIn(("bobert_companion.py", "_act_web_search"), found)

        undeclared = sorted(k for k in found if k not in self._DECLARED_SPLITS)
        self.assertEqual(
            undeclared, [],
            "these action families have some aliases routed to a speak set and "
            "others in NEITHER — the H-5 shape, where the owner says the alias, "
            "the work happens, and JARVIS says nothing. Route them, or declare "
            "the split in _DECLARED_SPLITS with why: "
            + "; ".join(f"{f}:{sym} -> {found[(f, sym)]}"
                        for f, sym in undeclared))

        # A row whose FILE is absent from this checkout is not stale, it is
        # unobservable: the personal skills are gitignored, so a public clone
        # legitimately cannot see them. Only rows whose file IS present and
        # whose family no longer drifts are stale.
        stale = sorted(
            k for k in self._DECLARED_SPLITS
            if k not in found
            and os.path.exists(os.path.join(_ROOT, k[0].replace("/", os.sep))))
        self.assertEqual(
            stale, [],
            f"_DECLARED_SPLITS names families that no longer drift — delete "
            f"the stale rows so the ledger keeps meaning something: {stale}")

    def test_focus_mode_is_no_longer_a_drifting_family(self):
        # The H-5 fix itself, expressed against the scan: focus_mode must NOT
        # appear in the drift ledger, and must not need to.
        self.assertNotIn(("skills/focus_mode.py", "focus_mode_off"),
                         self._DECLARED_SPLITS)
        self.assertNotIn(("skills/focus_mode.py", "focus_mode_on"),
                         self._DECLARED_SPLITS)


# ═══════════════════════════════════════════════════════════════════════════
#  2026-08-20 audit — six defects, all the same shape: a path that reports (or
#  implies) success it did not verify, or degrades without saying so.
# ═══════════════════════════════════════════════════════════════════════════


@requires_monolith
class Pyttsx3WallClockCapTests(MonolithGlobalsTestCase):
    """(a) The pyttsx3 rung of the TTS ladder had NO wall-clock bound.

    synthesise() runs on the voice thread under _SPEAK_LOCK. Every other rung
    is capped (voice clone 20 s, edge-tts 30 s, Kokoro, SAPI5 subprocess 15 s)
    because a HANG there is unrecoverable: the main loop never reaches the
    watchdog's reset consumer, so JARVIS goes permanently deaf AND mute while
    every other _speak() caller piles up on the untimed lock. runAndWait()
    drives a synchronous native SAPI render and was the one uncapped rung.
    """

    def _wedged_pyttsx3(self, gate):
        engine = mock.Mock()
        engine.runAndWait.side_effect = lambda: gate.wait(30)
        fake = mock.Mock()
        fake.init.return_value = engine
        return fake, engine

    def _release(self, gate):
        gate.set()
        deadline = time.time() + 10
        while self.bc._pyttsx3_inflight[0] and time.time() < deadline:
            time.sleep(0.02)

    def test_a_wedged_render_is_bounded_and_falls_through_to_sapi5(self):
        import sys
        gate = threading.Event()
        self.addCleanup(self._release, gate)
        fake, engine = self._wedged_pyttsx3(gate)
        sentinel = ("sapi5-sentinel", 22050)
        with mock.patch.dict(sys.modules, {"pyttsx3": fake}), \
                mock.patch.object(self.bc, "_PYTTSX3_TIMEOUT_S", 0.4), \
                mock.patch.object(self.bc, "_try_sapi5_then_silence",
                                  return_value=sentinel) as sapi:
            t0 = time.time()
            out = self.bc._pyttsx3_tts("hello")
            elapsed = time.time() - t0
        # The whole point: the CALLER (the voice thread) came back.
        self.assertLess(elapsed, 8.0,
                        "pyttsx3 render is not wall-clock bounded — the voice "
                        "thread stayed parked in native SAPI under _SPEAK_LOCK")
        self.assertEqual(out, sentinel,
                         "a timed-out pyttsx3 render must fall through to the "
                         "subprocess-bounded SAPI5 rung, never hang or raise")
        sapi.assert_called_once()
        self.assertTrue(engine.runAndWait.called,
                        "test is vacuous if the wedged rung was never entered")

    def test_a_single_flight_refuses_a_second_engine_while_one_is_wedged(self):
        # pyttsx3.init() memoises its driver, so a second call during a wedge
        # would hand out the SAME stuck engine and start a concurrent
        # runAndWait on it. The abandoned worker holds the guard until it
        # really finishes.
        import sys
        gate = threading.Event()
        self.addCleanup(self._release, gate)
        fake, _engine = self._wedged_pyttsx3(gate)
        sentinel = ("sapi5-sentinel", 22050)
        with mock.patch.dict(sys.modules, {"pyttsx3": fake}), \
                mock.patch.object(self.bc, "_PYTTSX3_TIMEOUT_S", 0.3), \
                mock.patch.object(self.bc, "_try_sapi5_then_silence",
                                  return_value=sentinel):
            first = self.bc._pyttsx3_tts("one")
            second = self.bc._pyttsx3_tts("two")
        self.assertEqual(first, sentinel)
        self.assertEqual(second, sentinel)
        self.assertEqual(fake.init.call_count, 1,
                         "a second engine was constructed while the first was "
                         "still wedged")

    def test_a_render_failure_still_reports_and_routes_to_sapi5(self):
        # The pre-existing raise path must survive the move onto a worker: a
        # failure is REPORTED to the caller, never swallowed into silence, and
        # the single-flight guard must not leak.
        import sys
        engine = mock.Mock()
        engine.save_to_file.side_effect = AssertionError("")
        fake = mock.Mock()
        fake.init.return_value = engine
        sentinel = ("sapi5-sentinel", 22050)
        with mock.patch.dict(sys.modules, {"pyttsx3": fake}), \
                mock.patch.object(self.bc, "_try_sapi5_then_silence",
                                  return_value=sentinel) as sapi:
            out = self.bc._pyttsx3_tts("boom")
        self.assertEqual(out, sentinel)
        sapi.assert_called_once()
        self.assertFalse(self.bc._pyttsx3_inflight[0],
                         "the single-flight guard leaked after a clean failure")


@requires_monolith
class SearchMyFilesBindingTests(MonolithGlobalsTestCase):
    """(b) `search_my_files` sat in SPEAK_RESULT_VERBATIM_ACTIONS — whose whole
    contract is "this result is already a finished, user-facing sentence" —
    while being bound to rag_search_quiet, the MACHINE-READABLE handler. JARVIS
    synthesised and read aloud raw `[1] path=<absolute path> score=0.812`
    blocks: up to RAG_DEFAULT_K untruncated chunks of the owner's private
    files, absolute paths included. Nothing else ever consumed that block (no
    mid-reply tool loop; the name is not in INFORMATIVE_ACTIONS), so the
    speaker was its only reader.

    Fixed by CHANGING THE BINDING, not by adding another name to a list. The
    alternative — canonicalising aliases by handler identity — is rejected in
    the H-5 block comment above and would have been wrong here twice over: it
    would have propagated this defect onto rag_search_quiet, and it would have
    collapsed the deliberate deco_topology / network_topology split.
    """

    def test_b_spoken_name_is_bound_to_the_spoken_formatter(self):
        rs = _load_registration_scan()
        regs = rs.scan_file(os.path.join(_ROOT, "skills", "personal_rag.py"))
        self.assertIn("search_my_files", regs)
        self.assertEqual(
            regs["search_my_files"].symbol, "rag_search",
            "search_my_files is voiced VERBATIM, so it must resolve to the "
            "voice formatter; bound to rag_search_quiet it reads a "
            "machine-readable block aloud")
        # The machine-readable twin keeps its own name and its own behaviour.
        self.assertEqual(regs["rag_search_quiet"].symbol, "rag_search_quiet")

    def test_b_membership_and_disjointness_unchanged(self):
        self.assertIn("search_my_files", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertIn("rag_search", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertNotIn("search_my_files", self.bc.INFORMATIVE_ACTIONS)
        # rag_search_quiet stays deliberately silent: it is the block-producing
        # twin, and nothing should ever read a block aloud.
        self.assertNotIn("rag_search_quiet",
                         self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertNotIn("rag_search_quiet", self.bc.INFORMATIVE_ACTIONS)

    def test_b_the_failure_guard_could_never_have_caught_the_block(self):
        # Why this was silent-by-construction rather than merely unlucky:
        # _speak_verbatim_results' only escape hatch is FAILURE_MARKERS, and a
        # hits block (or either sentinel) contains none of them. Pinned so a
        # future "just add a marker" fix is understood to be insufficient.
        from core.failure_markers import FAILURE_MARKERS
        for probe in ("[1] path=C:/x/y.txt score=0.812\nsome private text",
                      "[no matches]",
                      "[error: personal RAG unavailable]"):
            low = probe.lower()
            self.assertFalse(
                any(m.lower() in low for m in FAILURE_MARKERS),
                f"{probe!r} now trips a failure marker — re-check this test's "
                f"reasoning, but do NOT rely on markers to gate "
                f"machine-readable output")


@requires_monolith
class FaceIdFamilyVoicedTests(MonolithGlobalsTestCase):
    """(c) learn_guest — the ONLY guest-enrolment name core/prompts.py
    documents, and the one skills/face_tracker.py actively solicits ("say
    'remember their face'") — was in NEITHER speak set, while its sibling alias
    remember_this_person was verbatim. Five of learn_guest's branches carry no
    FAILURE_MARKER, so they were printed and dropped: the "what's their name?"
    question, the honest "everyone I can see is already someone I recognise"
    refusal, and the "(Captured N good views.)" confirmation. The owner heard
    only the model's affirmative prose — success-shaped audio for an enrolment
    that never happened.
    """

    def test_c_guest_enrolment_names_are_voiced(self):
        for name in ("learn_guest", "remember_their_face", "learn_their_face",
                     "remember_my_face"):
            self.assertIn(
                name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                f"{name} is a registered enrolment name whose handler reply is "
                f"the only honest signal of what actually happened")
            # Sets must stay disjoint (see test_speak_sets_are_disjoint).
            self.assertNotIn(name, self.bc.INFORMATIVE_ACTIONS)

    def test_c_every_registered_face_id_name_is_routed(self):
        # Source-scanning family invariant, mirroring the air-mouse and
        # focus-mode ones: a FUTURE alias added to skills/face_id.register()
        # fails the suite instead of silently swallowing an enrolment reply.
        # "Routed", not "uniformly verbatim": the whoami query aliases are
        # legitimately INFORMATIVE.
        rs = _load_registration_scan()
        regs = rs.scan_file(os.path.join(_ROOT, "skills", "face_id.py"))
        self.assertIn("learn_guest", regs)
        self.assertIn("enroll_face", regs)
        self.assertGreaterEqual(len(regs), 15, sorted(regs))
        routed = (set(self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
                  | set(self.bc.INFORMATIVE_ACTIONS))
        missing = sorted(n for n in regs if n not in routed)
        self.assertEqual(
            missing, [],
            f"skills/face_id.py registers action(s) with NO voicing route — "
            f"their reply, refusals included, would be dropped: {missing}")

    def test_c_guest_aliases_share_the_enrolling_handler(self):
        # The premise the fix rests on, asserted from source not assumed.
        rs = _load_registration_scan()
        regs = rs.scan_file(os.path.join(_ROOT, "skills", "face_id.py"))
        for alias in ("remember_their_face", "remember_this_person",
                      "learn_their_face"):
            self.assertEqual(regs[alias].symbol, "learn_guest")
        self.assertEqual(regs["remember_my_face"].symbol, "enroll_face")


@requires_monolith
class SkillContributedSpeakSetTests(MonolithGlobalsTestCase):
    """(d) Owner-private action names from GITIGNORED personal skills were
    spelled out in TRACKED source (this module's speak sets and a pinned test
    tuple), so .gitignore was not actually hiding them — the repo is public and
    the strings are already pushed. tools/check_no_pii.py cannot catch a
    re-introduction either: its owner-name patterns are word-boundary anchored
    and there is no word boundary inside an identifier. The private skills now
    declare their own speak-set membership and the monolith folds it in.
    """

    def _fake_skill(self, **attrs):
        import types
        m = types.ModuleType("skill_fake_private")
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def _discard(self, *names):
        for n in names:
            self.bc.SPEAK_RESULT_VERBATIM_ACTIONS.discard(n)
            self.bc.INFORMATIVE_ACTIONS.discard(n)

    def test_d_declared_names_are_folded_into_the_live_sets(self):
        self.addCleanup(self._discard, "zz_probe_verbatim", "zz_probe_info")
        mod = self._fake_skill(
            SPEAK_VERBATIM_ACTIONS=("zz_probe_verbatim",),
            INFORMATIVE_ACTIONS=("zz_probe_info",))
        self.bc._collect_skill_speak_sets(mod, "fake_private")
        self.assertIn("zz_probe_verbatim",
                      self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertIn("zz_probe_info", self.bc.INFORMATIVE_ACTIONS)

    def test_d_disjointness_violation_is_refused_and_logged(self):
        # Never silently corrupt the routing: a name already in the other set
        # is refused, loudly. (version_info is verbatim at HEAD.)
        import contextlib
        import io as _io
        self.assertIn("version_info", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        mod = self._fake_skill(INFORMATIVE_ACTIONS=("version_info",))
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._collect_skill_speak_sets(mod, "fake_private")
        self.assertNotIn("version_info", self.bc.INFORMATIVE_ACTIONS,
                         "a skill was allowed to break set disjointness")
        self.assertIn("REFUSED", buf.getvalue(),
                      "the refusal was silent — a degraded path must LOG")

    def test_d_a_malformed_declaration_cannot_break_skill_loading(self):
        for bad in ("not-an-iterable-of-names", 17, object()):
            mod = self._fake_skill(SPEAK_VERBATIM_ACTIONS=bad)
            self.bc._collect_skill_speak_sets(mod, "fake_private")   # no raise
        self.assertNotIn("n", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS,
                         "a bare string was iterated character-by-character")

    def test_d_the_loader_actually_invokes_the_collector(self):
        # Otherwise the mechanism exists and routes nothing.
        with io.open(os.path.join(_ROOT, "bobert_companion.py"),
                     encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def load_skills("):]
        body = body[:body.index("\ndef ", 1)]
        self.assertIn("_collect_skill_speak_sets(mod, name)", body,
                      "load_skills does not call _collect_skill_speak_sets — a "
                      "private skill's declared names would never be routed")

    def test_d_every_declared_name_is_actually_registered_by_that_skill(self):
        # A declaration for a name the skill never registers routes NOTHING —
        # it would look like coverage while the action stayed mute. Generic
        # scan over skills/, so no private filename or action name appears
        # here. Uses the same source-reader the drift ledger uses.
        rs = _load_registration_scan()
        d = os.path.join(_ROOT, "skills")
        checked = 0
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(d, fname)
            declared = ActionAliasSpeakSetDriftTests._declared_speak_names(path)
            if not declared:
                continue
            try:
                regs = rs.scan_file(path)
            except SyntaxError:
                continue
            missing = sorted(n for n in declared if n not in regs)
            self.assertEqual(
                missing, [],
                f"skills/{fname} declares speak-set membership for name(s) it "
                f"does not register — the declaration routes nothing: "
                f"{missing}")
            checked += 1
        # Sanity: at least one skill uses the mechanism, or this is vacuous.
        self.assertGreaterEqual(
            checked, 1,
            "no skill declares SPEAK_VERBATIM_ACTIONS / INFORMATIVE_ACTIONS — "
            "the private-name routing mechanism has no user, so a private name "
            "has probably been moved back into tracked source")

    def test_d_no_owner_private_names_in_the_tracked_tree(self):
        # Uses the OWNER'S OWN pattern list (tools/pii_local.py, gitignored) so
        # this tracked test carries no private string of its own. It also
        # applies the rule the shipped gate is missing: identifiers are scanned
        # a second time with "_" treated as a separator, which is exactly why
        # <name>_boss_mode_status slipped past a word-boundary-anchored pattern.
        import importlib.util
        import re
        import subprocess
        local = os.path.join(_ROOT, "tools", "pii_local.py")
        if not os.path.exists(local):
            self.skipTest("tools/pii_local.py (gitignored) absent — the "
                          "owner's private pattern list is unavailable here")
        spec = importlib.util.spec_from_file_location("pii_local", local)
        pii = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pii)
        pats = []
        for attr in ("WARN", "HARD", "WARN_PATTERNS", "HARD_PATTERNS"):
            for row in (getattr(pii, attr, None) or ()):
                try:
                    label, rx = row[0], row[1]
                except (TypeError, IndexError, KeyError):
                    continue
                low = str(label).lower()
                if isinstance(rx, str) and any(k in low for k in
                                               ("name", "trip", "venue")):
                    pats.append((label, re.compile(rx)))
        if not pats:
            self.skipTest("tools/pii_local.py exports no personal-name pattern")

        try:
            out = subprocess.run(["git", "ls-files"], cwd=_ROOT, timeout=90,
                                 capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git unavailable")
        if out.returncode != 0:
            self.skipTest("not a git checkout")

        # Files whose leak belongs to a DIFFERENT open finding: a generated doc
        # and two VIP test fixtures. Listed by PATH only — never by content.
        # ALLOWED, not required: closing those findings keeps this green.
        elsewhere = {
            "docs/ACTION_INDEX.md",
            "tests/test_audit_2026_07_14.py",
            "tests/skills/test_vip_intercept.py",
        }
        hits = []
        for rel in out.stdout.splitlines():
            rel = rel.strip()
            if not rel or rel in elsewhere:
                continue
            if os.path.splitext(rel)[1].lower() not in (
                    ".py", ".md", ".ps1", ".json", ".yml", ".yaml", ".txt"):
                continue
            path = os.path.join(_ROOT, rel.replace("/", os.sep))
            try:
                with io.open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            tokenised = text.replace("_", " ")
            for label, rx in pats:
                if rx.search(text) or rx.search(tokenised):
                    hits.append(f"{rel} ({label})")
                    break
        self.assertEqual(
            sorted(set(hits)), [],
            "tracked file(s) carry a name the project classifies as private "
            "(paths and labels only; contents deliberately not echoed): "
            + ", ".join(sorted(set(hits))))


@requires_monolith
class BlueGreenHardenedExitTests(MonolithGlobalsTestCase):
    """(e) The blue/green handoff was the ONE intentional stop that skipped the
    hardened teardown: a bare `return` out of main() into CPython finalization
    -> ExitProcess under the loader lock — the exact route that produced
    22-hour immortal zombies twice. None of the six atexit handlers releases
    the CUDA context, the Kinect handle, the WASAPI streams or the :8766
    socket, and mark_intentional_exit() was never called either, so even a
    perfect handoff left no watchdog handshake flag.
    """

    def _run_teardown(self):
        import sys
        import core.actions as _ca
        order = []

        def rec(tag, ret=None):
            def _f(*a, **k):
                order.append(tag)
                return ret
            return _f

        wi = mock.Mock()
        wi._httpd = object()
        wi._stop.side_effect = rec("port")
        bgm = mock.Mock()
        bgm.unregister_instance.side_effect = rec("bgm")
        timer = mock.Mock()
        timer.start.side_effect = rec("failsafe")
        focus = mock.Mock()
        focus.set.side_effect = rec("focus")

        with mock.patch.dict(sys.modules, {"skill_web_interface": wi}), \
                mock.patch.object(self.bc, "mark_intentional_exit",
                                  rec("intent")), \
                mock.patch.object(self.bc, "_release_singleton",
                                  rec("singleton")), \
                mock.patch.object(self.bc.threading, "Timer",
                                  mock.Mock(return_value=timer)), \
                mock.patch.object(_ca, "_release_native_resources",
                                  rec("natives")), \
                mock.patch.object(self.bc, "_focus_tracker_stop", focus), \
                mock.patch.object(self.bc, "_shutdown_hud", rec("hud")), \
                mock.patch.object(self.bc, "_shutdown_tray", rec("tray")), \
                mock.patch.object(self.bc, "_bgm", bgm), \
                mock.patch.object(self.bc, "_restore_prior_power_plan",
                                  rec("power")), \
                mock.patch.object(self.bc, "close_log", rec("log")), \
                mock.patch.object(self.bc, "_hard_exit", rec("exit")):
            self.bc._blue_green_teardown_and_exit()
        return order

    def test_e_teardown_runs_and_ends_in_the_undeadlockable_exit(self):
        order = self._run_teardown()
        for tag in ("intent", "singleton", "port", "failsafe", "natives",
                    "focus", "hud", "tray", "bgm", "exit"):
            self.assertIn(tag, order, f"{tag} step missing: {order}")
        self.assertEqual(order[-1], "exit",
                         f"teardown did not end in _hard_exit: {order}")

    def test_e_order_is_load_bearing(self):
        order = self._run_teardown()
        i = order.index
        self.assertLess(i("intent"), i("exit"),
                        "without mark_intentional_exit the watchdog reads the "
                        "takeover as an unintended death")
        self.assertLess(i("singleton"), i("natives"),
                        "the successor's singleton lock must be freed BEFORE "
                        "anything that can wedge in native code")
        self.assertLess(i("port"), i("natives"),
                        "the dashboard socket must be freed before the natives")
        self.assertLess(i("failsafe"), i("natives"),
                        "nothing below the failsafe is time-bounded; arm it "
                        "first or a wedged release is unrecoverable")
        self.assertLess(i("natives"), i("exit"),
                        "a thread parked in a driver at exit time corpse-pins "
                        "the VRAM")

    def test_e_handoff_call_site_never_returns_bare_from_main(self):
        with io.open(os.path.join(_ROOT, "bobert_companion.py"),
                     encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("if _blue_green_loop_tick():")
        lines = src[i:i + 900].splitlines()
        self.assertTrue(
            any("_blue_green_teardown_and_exit()" in ln for ln in lines),
            "the handoff exit fell back to a bare return — that is the "
            "ExitProcess-under-the-loader-lock route")
        call_i = next(n for n, ln in enumerate(lines)
                      if "_blue_green_teardown_and_exit()" in ln)
        ret_i = next(n for n, ln in enumerate(lines) if ln.strip() == "return")
        self.assertLess(call_i, ret_i,
                        "teardown must run BEFORE the return")

    def test_e_teardown_uses_the_one_hardened_native_release(self):
        # Re-inlining the release (unload() without the in-flight CUDA wait,
        # kinect close() without final=True) is exactly the stale duplicate
        # that reopened the corpse class in the Ctrl-C path.
        with io.open(os.path.join(_ROOT, "bobert_companion.py"),
                     encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def _blue_green_teardown_and_exit"):]
        body = body[:body.index("\ndef ", 1)]
        # Scan the CODE, not the docstring (which names the hardened steps).
        _d0 = body.index('"""')
        code = body[body.index('"""', _d0 + 3) + 3:]
        self.assertIn("from core.actions import _release_native_resources",
                      code)
        for inlined in ("close(final=True)", "unload(", "sd.stop("):
            self.assertNotIn(inlined, code,
                             f"{inlined} was re-inlined here instead of "
                             f"going through _release_native_resources")
        # And it must NOT end the session: the conversation is handed over.
        self.assertNotIn("save_session_to_memory", code)


@requires_monolith
class CameraReopenBackoffTests(MonolithGlobalsTestCase):
    """(f) The P1-3 camera-contention yield was applied exactly once. THREE
    separate sites wrote entry["next_reopen_at"] and only the wedge branch
    consulted contention, so the very next failed reopen overwrote the 30 s
    yield with a flat 2 s and the loop went back to hammering DirectShow — the
    overlapping open/release churn this file names as the leading suspect for
    the silent crash cascade. Worse, in the scenario the feature exists for
    (Teams holding the device) the wedge branch is unreachable, so the yield
    never ran at all — while the log line promised a backoff the code then
    abandoned.
    """

    def setUp(self):
        super().setUp()
        self._reset_cache()
        self.addCleanup(self._reset_cache)

    def _reset_cache(self):
        self.bc._camera_locker_cache[0] = 0.0
        self.bc._camera_locker_cache[1] = []

    def test_f_every_failed_reopen_re_evaluates_contention(self):
        entry = {"next_reopen_at": 0.0, "contention_logged": False}
        wide = self.bc.CAMERA_CONTENTION_BACKOFF_SEC
        for attempt, now in enumerate((100.0, 130.0, 160.0)):
            backoff, _lockers = self.bc._schedule_camera_reopen(
                entry, "emeet", 0, now, lockers=["Teams.exe"])
            self.assertEqual(backoff, wide, f"attempt {attempt}")
            self.assertAlmostEqual(
                entry["next_reopen_at"], now + wide, places=6,
                msg=f"attempt {attempt}: the contention yield decayed back to "
                    f"the flat reopen spacing — that is the defect")

    def test_f_hint_is_logged_once_and_recovery_is_logged_too(self):
        import contextlib
        import io as _io
        entry = {"next_reopen_at": 0.0, "contention_logged": False}
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._schedule_camera_reopen(entry, "emeet", 0, 100.0,
                                            lockers=["Teams.exe"])
            self.bc._schedule_camera_reopen(entry, "emeet", 0, 130.0,
                                            lockers=["Teams.exe"])
        self.assertEqual(buf.getvalue().count("appears LOCKED"), 1,
                         "the actionable hint must be one-shot per episode")
        self.assertTrue(entry["contention_logged"])

        buf2 = _io.StringIO()
        with contextlib.redirect_stdout(buf2):
            backoff, _ = self.bc._schedule_camera_reopen(
                entry, "emeet", 0, 200.0, lockers=[])
        self.assertEqual(backoff, self.bc.CAMERA_REOPEN_BACKOFF_SEC)
        self.assertAlmostEqual(entry["next_reopen_at"],
                               200.0 + self.bc.CAMERA_REOPEN_BACKOFF_SEC,
                               places=6)
        self.assertFalse(entry["contention_logged"])
        self.assertIn("no longer appears", buf2.getvalue(),
                      "leaving the degraded state must LOG, not silently speed "
                      "back up")

    def test_f_locker_lookup_is_cached_off_the_face_track_hot_path(self):
        calls = []

        def _walk():
            calls.append(1)
            return ["OBS"]

        with mock.patch.object(self.bc, "find_camera_locking_processes",
                               side_effect=_walk):
            self.assertEqual(self.bc._camera_lockers_cached(500.0), ["OBS"])
            self.assertEqual(
                self.bc._camera_lockers_cached(
                    500.0 + self.bc._CAMERA_LOCKER_CACHE_TTL_S / 2), ["OBS"])
            self.assertEqual(len(calls), 1, "the process walk was not cached")
            self.bc._camera_lockers_cached(
                500.0 + self.bc._CAMERA_LOCKER_CACHE_TTL_S + 1.0)
            self.assertEqual(len(calls), 2, "the cache never expires")

    def test_f_next_reopen_at_has_exactly_one_scheduling_writer(self):
        # The stale duplicate is this codebase's #1 bug class, and three copies
        # of this rule are what made the yield dead. Any assignment outside
        # _schedule_camera_reopen must be a plain reset to 0.0.
        with io.open(os.path.join(_ROOT, "bobert_companion.py"),
                     encoding="utf-8") as fh:
            lines = fh.readlines()
        helper_start = next(i for i, ln in enumerate(lines)
                            if ln.startswith("def _schedule_camera_reopen("))
        helper_end = next(i for i in range(helper_start + 1, len(lines))
                          if lines[i].startswith("def "))
        offenders = []
        for i, ln in enumerate(lines):
            if helper_start <= i < helper_end:
                continue
            for marker in ('next_reopen_at"] =', "next_reopen_at'] ="):
                if marker in ln:
                    rhs = ln.split(marker, 1)[1].strip().rstrip(",")
                    if rhs not in ("0.0", "0"):
                        offenders.append(f"{i + 1}: {ln.strip()}")
        self.assertEqual(
            offenders, [],
            "a second scheduling writer of next_reopen_at re-appeared — that "
            "is exactly how the contention yield was overwritten: "
            + "; ".join(offenders))

    def test_f_wedge_branch_logs_the_backoff_it_actually_armed(self):
        with io.open(os.path.join(_ROOT, "bobert_companion.py"),
                     encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("_backoff, lockers = _schedule_camera_reopen(")
        block = src[i:i + 1600]
        self.assertIn("will reopen in {_backoff:.1f}s", block,
                      "the dead-camera line must report the backoff that was "
                      "armed, not a constant the code goes on to ignore")

if __name__ == "__main__":
    unittest.main()
