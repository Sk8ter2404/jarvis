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
        sweep = (
            "air_mouse_status", "amazon_tracking_status", "ambient_extract_status",
            "ambient_listen_status", "anticipation_briefing_status", "anticipation_status",
            "are_you_ok", "audio_music_status", "banter_status", "bonnaroo_status",
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
            "wayne_boss_mode_status", "weekly_digest", "weekly_digest_status",
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
        raises, anything else is treated as a successful open)."""

        class PortAudioError(Exception):
            pass

        def __init__(self, play_effects):
            # play_effects: list, one entry consumed per play() call. An entry
            # that is an Exception instance is raised; None means success.
            self._effects = list(play_effects)
            self.play_calls = []   # list of the `device` kwarg per call

        def play(self, audio, sr, device=None):
            self.play_calls.append(device)
            effect = self._effects.pop(0) if self._effects else None
            if isinstance(effect, BaseException):
                raise effect

        def wait(self):
            pass

        def stop(self):
            pass

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


if __name__ == "__main__":
    unittest.main()
