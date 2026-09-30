"""Safely import the ``bobert_companion`` monolith for unit tests.

The monolith is the ~14K-line entrypoint. Two things make a naive ``import
bobert_companion`` unsafe in a test:

  1. Its module-level ``_early_boot_singleton_lock()`` can ``sys.exit`` if it
     thinks another instance holds the lock. Setting the process-wide sentinel
     ``_JARVIS_SINGLETON_PID`` to our own PID *before* import short-circuits it
     (bobert_companion.py:165) — no lock, no exit. The heavy boot (threads,
     devices, the conversation loop) is gated behind ``if __name__ ==
     "__main__"`` so a plain import never runs it.
  2. It top-level-imports heavy deps (numpy/sounddevice/cv2/soundfile/requests)
     that are ABSENT on the light-deps CI runner, so the monolith can't import
     there at all. Monolith tests therefore run in the LOCAL full tier only:
     decorate them with ``@requires_monolith`` so they skip cleanly on CI.

Usage in a monolith test module::

    from tests._monolith_harness import load_monolith, requires_monolith

    @requires_monolith
    class FooTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.bc = load_monolith()
        def test_something(self):
            self.assertEqual(self.bc._humanize_seconds(90), "1 minute")
"""
from __future__ import annotations

import copy
import importlib.util
import os
import sys
import unittest
from collections import deque

# Heavy deps the monolith imports at top level; all must be present to import it.
_MONOLITH_DEPS = ("numpy", "sounddevice", "cv2", "soundfile", "requests")


def _monolith_importable() -> bool:
    for dep in _MONOLITH_DEPS:
        try:
            if importlib.util.find_spec(dep) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


MONOLITH_AVAILABLE = _monolith_importable()

# Decorator: monolith tests run only where the heavy deps exist (local full
# tier), and skip on the light-deps CI runner.
requires_monolith = unittest.skipUnless(
    MONOLITH_AVAILABLE,
    "monolith heavy deps (numpy/sounddevice/cv2/soundfile/requests) absent — "
    "local full tier only")

_bc = None  # cached module so the (one-time) import cost is paid once per process


def load_monolith():
    """Import + return the ``bobert_companion`` module, booting nothing.

    Idempotent (cached). Only call from tests guarded by ``@requires_monolith``.
    """
    global _bc
    if _bc is not None:
        return _bc
    # Short-circuit the boot singleton lock + force the quietest runtime posture.
    os.environ["_JARVIS_SINGLETON_PID"] = str(os.getpid())
    os.environ.setdefault("JARVIS_STAGING", "1")
    os.environ.setdefault("JARVIS_TEST_MODE", "1")
    os.environ.setdefault("MUTE_TTS", "1")
    # A raising action files a self-detected bug report into the outbox. That
    # outbox used to be bound to core/bug_reporter.py's __file__ - the LIVE
    # data/bug_reports.jsonl, which no env redirect reached (found 2026-09-30
    # by a write audit); it now resolves through core.paths at call time
    # (data_staging/ here, or the runner's JARVIS_DATA_DIR). Off by default
    # all the same: tests that exercise the hook set it explicitly.
    os.environ.setdefault("JARVIS_BUG_AUTO_CAPTURE", "0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import bobert_companion as bc  # noqa: E402  (env must be set first)
    # Per-install tuning must never reach a test. At import the monolith applied
    # THIS box's user_settings.json SPEECH_FILTER_OVERRIDES (mutating the shared
    # core.speech_filter module) and sized its follow-up window from
    # FOLLOWUP_WINDOW_S. Both are expected to differ per machine (the Dell edge
    # node sets them), so put the shipped defaults back: every test then sees the
    # same thresholds on every box, not "passes only on the author's machine".
    import core.speech_filter as _sf  # noqa: E402
    from core.followup_window import FollowupWindow  # noqa: E402
    _sf.reset_overrides()
    bc.WHISPER_TRUST_RMS = _sf.WHISPER_TRUST_RMS
    bc._followup_window = FollowupWindow(0)
    _bc = bc
    return bc


# ──────────────────────────────────────────────────────────────────────────
#  Shared monolith-globals isolation base
# ──────────────────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS
# ---------------
# The six tests/monolith/test_monolith_secN.py modules import the *real*
# bobert_companion monolith once (harness-cached) and exercise its functions.
# Many of those functions read/write module-level globals — a lot of them
# re-exported from core.state / core.config via ``from core.state import *``.
# A test that mutates one of these and fails to restore it (or restores it in
# the wrong order — e.g. ``mock.patch.dict(bc.__dict__, ...)`` stopped *after*
# a sibling ``mock.patch.object(bc, "_tts_layer", Mock())`` resurrects the
# Mock from the dict snapshot) leaks state into every later test in the
# process. Two such leaks were observed in the full ``tools/run_tests.py``:
#
#   * ``_last_wake_date`` (a single-element list slot in core.state, re-exported
#     on bc) left stamped with today's date by a force_wake dispatch test →
#     broke ``tests.test_state.SeedValueTests.test_last_wake_date_seeds_none``.
#   * ``_tts_layer`` (the core.tts module reference) left as a bare ``Mock`` by
#     a synthesise test whose ``patch.dict(bc.__dict__)`` teardown re-applied an
#     already-stopped ``_tts_layer`` patch → ``_speak`` could no longer strip
#     the ``[wry]`` tag, breaking sec6 ``SpeakTests``.
#
# Per-test snapshot/restore is brittle at ~7.8K tests, so instead every
# monolith test class inherits ``MonolithGlobalsTestCase``. It captures a
# *pristine* baseline of the mutated globals exactly once (the first test's
# ``run``, before any monolith test body has executed) and DEEP-RESTORES that
# baseline after EVERY test. Any leak — present or future, in-place or
# rebound — is therefore self-healing.
#
# IMPLEMENTATION NOTE — why ``run()`` and not ``tearDown()``
# ----------------------------------------------------------
# Most sec3/4/5 test classes define their own ``setUp``/``tearDown`` WITHOUT
# calling ``super()``, so a base ``tearDown`` would be silently shadowed. The
# restore is therefore wired into ``run()`` (which subclasses never override)
# in a ``try/finally`` that wraps the subclass's whole setUp→test→tearDown
# cycle. Mutable containers are restored IN PLACE (clear + refill the SAME
# object) so re-exported aliases that hold the original reference stay
# consistent; rebindable scalars / module refs are restored with ``setattr``.

# Module globals on bobert_companion that monolith tests mutate (directly or
# via the functions under test). Restoring all of these to their import-time
# values after each test makes cross-test pollution impossible regardless of
# which test leaked. Names that don't exist on the module (older/newer trees)
# are skipped gracefully at snapshot time.
_MONOLITH_RESTORE_NAMES = (
    # ── conversation + memory buffers ──────────────────────────────────────
    "conversation_history",
    # ── wake / greeting bookkeeping (re-exported from core.state) ──────────
    "_last_wake_date", "_wake_history", "_pre_wake_silence_seconds",
    # ── focus mode / do-not-disturb (skills/focus_mode.py drives these) ─────
    # Single-element-list flag cells + the bounded missed buffer. A test that
    # engages focus (or appends to the buffer) must not leak it into the next.
    "_focus_mode", "_focus_until", "_focus_missed_buffer",
    # ── single-element runtime state slots (core.state) ────────────────────
    "_sleep_mode", "_standby_mode", "_tts_muted", "_ambient_mode_active",
    "_daemons_paused", "_debug_mode", "_audio_master_enabled",
    "_audio_aec_enabled", "_audio_ns_enabled", "_audio_agc_enabled",
    "_jarvis_played_music_at", "_ambient_music_last_hit", "_ambient_music_hits",
    "_session_resume_done", "_main_loop_heartbeat",
    "_last_heartbeat_publish", "_main_loop_heartbeat_path_cache",
    "_bg_handoff_grace_s",
    "_silent_mic_warned", "_silent_mic_warned_at", "_silent_mic_warned_device",
    # ── voice fast-path latches (bobert_companion-local single-element lists) ─
    # Reset to [None]/[False] so a wake detector built (or a disable-latch
    # tripped) in one test can't bleed into the next. (tests/monolith/
    # test_monolith_voice_wiring.py also resets these locally; this makes the
    # heal global.) NOTE: a clean latch does NOT guarantee _standby_wake_detected
    # returns None — on a host with WAKE_WORD_AUTOSTART on it rebuilds a real
    # detector, so tests wanting the Whisper branch must still mock it.
    "_standby_wake_detector", "_standby_wake_disabled_for_session",
    # ── tray / action-dispatch queues + bookkeeping ────────────────────────
    "_pending_confirmation", "_pending_autocorrect_choice",
    "_action_error_log", "_action_history",
    "_session_action_counts", "_session_app_names",
    # ── audio / device / capture state ─────────────────────────────────────
    "_device_cache", "_recent_spoken_messages", "_record_speech_taps",
    "_record_speech_active", "_pathb_mic_active", "_tts_playback_active",
    "_ambient_stream_active",
    # What record_speech's stream really opened on (2026-09-29). A record left
    # by a test that drove record_speech would make a later what_microphone
    # test answer from THAT test's fake device instead of its own fixture.
    "_live_capture_device",
    # Self-echo gate (2026-09-29): the last utterance's capture timing. A
    # window left by a test that drove record_speech would make a LATER
    # test's gate judge a different turn against it.
    "_last_capture_window",
    # Capture open-failure log throttle (2026-09-29): a leaked entry would
    # silence the NEXT test's first open-failure line.
    "_open_fail_log",
    # Capture-open backoff (R10, 2026-09-29): the unexpected-error traceback
    # dedupe (a leaked entry would silence the NEXT test's traceback). The
    # backoff object itself is reset in _restore_monolith_pristine — an
    # episode left open by a test whose open failed would make the NEXT
    # test's record_speech WAIT (really sleep) before it even tries.
    "_input_open_tb_seen", "_input_backoff_clock", "_input_backoff_sleep",
    # Noise gate (R10): when JARVIS's last heard line finished. A line
    # spoken by one test would make a LATER test's "thank you" a reply.
    "_last_jarvis_line",
    # PortAudio teardown-gate cells (2026-08-14): the diag/enroll owner
    # refcounts and the reinit latch. A leaked non-zero cell would make every
    # later _refresh_devices test silently defer its reinit; a leaked latch
    # would time out every claim.
    "_diag_capture_active", "_enroll_capture_active", "_pa_reinit_active",
    # Processing filler (2026-09-29): the two module objects, rebind-restored.
    # Rebinding alone would put back the SAME (possibly latched) instance, so
    # _restore_monolith_pristine also resets their internal state — see
    # _reset_filler_state. Tests that need filler state should still patch in
    # a FRESH ProcessingFiller / ClipCache rather than mutate these.
    "_processing_filler", "_filler_clips",
    # Per-turn timing line (2026-09-29): rebind-restored, and its active turn
    # dropped in _restore_monolith_pristine — a turn left open by a test that
    # drove a capture would make a LATER test's dispatch print a stray
    # [turn-timing] line into output it asserts on.
    "_turn_timing",
    # Local prompt-prefix stability (2026-09-29): the conversation-activity
    # gates, the deferred-rebuild and re-prime single-flight cells and the
    # phrase-rotation cache. A leaked _turn_in_progress / pending rebuild /
    # running re-prime would make a LATER test's freeze or re-prime decision
    # pass or fail for the wrong reason. _system_prompt is rebind-restored so
    # a test that applied a rebuild cannot leak its prompt.
    "_system_prompt",
    "_phrase_rotation_last", "_last_convo_activity", "_turn_in_progress",
    "_utterance_in_progress", "_prompt_rebuild_pending",
    "_prompt_rebuild_waiter", "_reprime_running", "_reprime_again",
    "_reprime_prefix_hash",
    # Local background traffic control (2026-09-29, r6): the owner-turn stamp
    # that opens the re-prime-after-eviction window, the prime's age / POST
    # mark (the hit / evicted diagnostic) and learn_from_turn's queue + worker
    # flag. A leaked owner stamp would make a LATER test's background call
    # schedule a real re-prime thread; a leaked worker flag would make every
    # later learn_from_turn queue a turn nobody drains. The shared gate in
    # core.local_traffic is reset in _restore_monolith_pristine.
    "_last_owner_turn_at", "_reprime_primed_at", "_reprime_posts_mark",
    "_learn_pending", "_learn_worker_live",
    # Proactive-remark gates (2026-09-30): the owner's last MIC turn, the
    # remark history (repeat ring, spoken-remark times, last attempt, logged
    # hold) and the face-presence state (the detector's last detail, the
    # per-camera trackers and frame fingerprints). A leaked voice stamp or
    # tracker would make a LATER test's should_be_proactive pass or fail for
    # the wrong reason; a leaked remark would make a later remark a "repeat".
    "_last_owner_voice_at", "_proactive_recent", "_proactive_remarks_at",
    "_proactive_last_attempt_at", "_proactive_hold_logged",
    "_face_detect_last", "_face_presence_trackers", "_face_presence_fp",
    # "What was the first thing I asked" (v2.0.148): the session's opening
    # owner utterances. A test that recorded one would make a LATER test's
    # first-thing recall answer from THAT test's session. Its stamps (for
    # "forget the last hour") and its "start lost" latch (a leaked True would
    # silence every later test's record) are restored with it.
    "_session_opening_turns", "_session_opening_ts", "_session_opening_lost",
    # H-6 (2026-08-20): the abandoned-native-close count. A leaked non-zero
    # value would make every later _refresh_devices test silently defer its
    # reinit — the exact "green for the wrong reason" shape.
    "_pa_close_pending",
    "_camera_failure_summary",
    # Camera quarantine + producer-liveness state (2026-09-05). A test that
    # benches a camera or fakes a stalled heartbeat must not leak it: a
    # leftover quarantine would make every later camera test silently skip
    # the device, and a leftover heartbeat would make the watchdog tests
    # green for the wrong reason.
    "_camera_quarantine", "_camera_pending_releases",
    "_face_track_heartbeat", "_face_track_stall_state",
    "_face_track_caps",
    # Side-tile webcam index resolver (2026-09-05). These five cells ARE the
    # DirectShow-enumeration leak gate, and leaving them untracked makes every
    # test that touches the resolver ORDER-DEPENDENT: once one test records a
    # device fingerprint, a later test's call short-circuits on it and returns
    # whatever is in _kinect_preview_webcam_idx instead of enumerating the
    # pygrabber list the test just installed — passing or failing on the
    # machine's real camera set rather than on the code under test.
    "_kinect_preview_webcam_idx", "_kinect_preview_webcam_resolved",
    "_kinect_preview_webcam_resolved_at", "_kinect_preview_webcam_fingerprint",
    "_kinect_preview_webcam_enumerated_at",
    # ...and the four cells behind the gate's fail-open ALARM. Same hazard, one
    # step removed: the alarm is throttled and edge-triggered, so a test that
    # leaves _webcam_fingerprint_degraded True (or the throttle stamped at a
    # mocked time) makes the NEXT test's warning silently not fire — a test
    # asserting the alarm would then fail for a reason that has nothing to do
    # with the code it is testing, and one asserting silence would pass
    # vacuously.
    "_webcam_fingerprint_degraded", "_webcam_fingerprint_degraded_since",
    "_webcam_fingerprint_warned_at", "_webcam_fingerprint_leaky_enums",
    # ...and the AMPLIFIER accounting (2026-09-06). These are the counters that
    # make "the amplifier fired" a checkable statement instead of a story, so a
    # test that leaks a non-zero window start would make the NEXT test's report
    # line silently not print (throttled + edge-triggered, exactly like the
    # alarm cells above), and a leaked invalidation count would let a soak-
    # honesty test pass on somebody else's failures.
    "_side_tile_gate_counts", "_side_tile_gate_window",
    # ...and the producer-owned decline note's per-slot throttle (2026-09-06).
    # Same edge-triggered hazard: a leaked timestamp makes the NEXT test's
    # "declined N own-handle open(s)" line silently not print, so a test
    # asserting the line fails for a reason unrelated to its own code.
    "_tile_open_declined_note",
    # ...and the camera-gate refusal-note throttle + the stale-tile throttle
    # (2026-09-29). Same edge-triggered hazard as the decline note above.
    "_camera_gate_refusal_noted", "_preview_starved_log",
    "_face_track_open_verdict",
    # ...and which devices have resolved to Media Foundation this process,
    # which decides _camera_open's no-DirectShow-fallback rule, and how each
    # device's last open ended (the gate's lock classification reads it).
    "_camera_msmf_seen", "_camera_open_last_result",
    # ...and the privacy-log "who is using a webcam" cache (a leaked entry
    # would make the next test's lock decision depend on this one's).
    "_camera_users_cache",
    # ...and the camera-OPEN path's slot of the same gate, which
    # _dshow_name_to_index() owns. Same hazard as the resolver's cells above and
    # then some: this one is consulted by _open_capture and by the boot rescue,
    # so a fingerprint left behind by one test makes the NEXT test's
    # _dshow_name_to_index() answer out of a stale cache instead of the device
    # list the test just installed — i.e. green for the wrong reason.
    "_dshow_open_devices_cache",
    # ── single-flight / barge-in counter cells (2026-07-08 bug-hunt) ────────
    # New single-element-list cells: reset in place so a test that abandons a
    # clone worker (leaving _voice_clone_inflight True) or accepts a barge-in
    # (bumping _tts_interrupt_seq) can't poison the next test.
    "_voice_clone_inflight", "_tts_interrupt_seq",
    # Device dialogues (2026-09-29): the running flag / handle, the
    # post-dialogue holds and the self-voiced set. A leaked flag would make a
    # later request_tts_interrupt test accept a stop with nothing playing; a
    # leaked hold would silence a later _speak_pending test.
    "_dialogue_active", "_dialogue_current", "_speech_hold_until",
    "_turn_hold_until", "_turn_hold_reason", "SELF_VOICED_ACTIONS",
    "_hud_state_cache", "_focused_window_state", "CAMERAS",
    "last_speech_time", "last_face_seen", "_apple_music_last_seen",
    # ── STT / whisper model handles ────────────────────────────────────────
    "_stt", "_stt_device", "_stt_model_name", "_stt_engine",
    # ── local-LLM / ollama latches + caches ────────────────────────────────
    "_RESOLVED_LOCAL_LLM_MODEL", "_OLLAMA_INSTALL_TRIGGERED",
    "_OLLAMA_PULL_TRIGGERED", "_LOCAL_VISION_PULL_TRIGGERED",
    "_LOCAL_CHEATSHEET_CACHE",
    # ── TTS prosody singletons + layer/loop refs ───────────────────────────
    "_tts_layer", "_last_wry", "_last_intent_override", "_last_mood",
    "_last_user_text", "_last_emotion", "_last_voice_route", "_last_user_tone",
    "_tts_loop", "_tts_loop_thread", "_barge_in_interrupted",
    # ── wake-word barge-in (feat/barge-in) ─────────────────────────────────
    # _tts_current_text is the single-element echo-gate cell published by
    # _speak(); restored in place. (_tts_interrupt is a threading.Event —
    # identity-restored only, so tests that set() it must clear() it in
    # their own cleanup.)
    "_tts_current_text",
    # ── subprocess handles + logging ───────────────────────────────────────
    "_hud_process", "_tray_process", "_reticle_process",
    "_log_file_handle", "_log_file_path",
    # ── misc boot/runtime singletons ───────────────────────────────────────
    "_prior_power_plan_guid", "_pyautogui", "_SINGLETON_HELD_FD",
)

# Captured ONCE, lazily, before the first monolith test body runs. Maps each
# tracked name to a (kind, pristine_deepcopy) pair so we can restore in place.
_MONOLITH_PRISTINE: dict = {}
_MONOLITH_PRISTINE_READY = False

# Container kinds we restore IN PLACE to preserve object identity (so the
# re-exported aliases in core.state / consumer skills keep pointing at the
# same live object). Everything else is restored by rebinding the attribute.
_IN_PLACE_TYPES = (list, dict, set, deque)


def _capture_monolith_pristine(bc) -> None:
    """Snapshot the import-time value of every tracked global, exactly once."""
    global _MONOLITH_PRISTINE_READY
    if _MONOLITH_PRISTINE_READY:
        return
    for name in _MONOLITH_RESTORE_NAMES:
        if not hasattr(bc, name):
            continue
        val = getattr(bc, name)
        if isinstance(val, _IN_PLACE_TYPES):
            # Deep-copy so nested mutation (e.g. CAMERAS' inner dicts, the
            # device-cache values) is captured, not aliased.
            _MONOLITH_PRISTINE[name] = ("inplace", copy.deepcopy(val))
        else:
            # Scalars / module refs / handles: keep the reference itself.
            _MONOLITH_PRISTINE[name] = ("rebind", val)
    _MONOLITH_PRISTINE_READY = True


def _restore_monolith_pristine(bc) -> None:
    """Deep-restore every tracked global to its captured pristine baseline.

    Mutable containers are cleared + refilled on the SAME object so aliases
    stay valid; scalars / refs are reassigned. Never raises (a restore failure
    must not mask the test's own result)."""
    for name, (kind, pristine) in _MONOLITH_PRISTINE.items():
        try:
            if kind == "inplace":
                cur = getattr(bc, name, None)
                fresh = copy.deepcopy(pristine)
                if isinstance(cur, list):
                    cur[:] = fresh
                elif isinstance(cur, dict):
                    cur.clear()
                    cur.update(fresh)
                elif isinstance(cur, set):
                    cur.clear()
                    cur.update(fresh)
                elif isinstance(cur, deque):
                    cur.clear()
                    cur.extend(fresh)
                else:
                    # The object was rebound to a non-container by a leaky
                    # test (e.g. patch.dict resurrecting a different type) —
                    # put the pristine container back wholesale.
                    setattr(bc, name, fresh)
            else:
                setattr(bc, name, pristine)
        except Exception:
            # Best-effort: a single stubborn slot must not abort the rest.
            pass
    _reset_filler_state(bc)
    _reset_camera_gate(bc)
    try:
        bc._turn_timing.reset()
    except Exception:
        pass
    # A test that failed while a background job held the shared gate must
    # not make every later tagged call wait out the deferral cap.
    try:
        bc._lt.GATE.reset()
    except Exception:
        pass
    # Audio flap governor (2026-09-29): one process-wide instance, rebound
    # never, so its flip history / held sentence / storm must be wiped in
    # place — a storm left open by one test would keep the NEXT test's
    # device announcements quiet (green or red for the wrong reason).
    try:
        bc._audio_flap.reset()
    except Exception:
        pass
    # Self-echo registry (core/self_echo.py, 2026-09-29): every real _speak /
    # play_with_lipsync a test runs is remembered there on the REAL monotonic
    # clock, so a line spoken by one test would make a LATER test's mic
    # transcript of the same words a "self-echo" (dropped for the wrong
    # reason). Wiped in place.
    try:
        bc._self_echo._reset_for_tests()
    except Exception:
        pass
    # Capture-open backoff (R10, 2026-09-29): one process-wide instance, so
    # its episode is wiped in place (see _MONOLITH_RESTORE_NAMES).
    try:
        bc._input_open_backoff.reset()
    except Exception:
        pass


def _reset_camera_gate(bc) -> None:
    """Give the NEXT test a camera gate with no history (2026-09-29).

    The gate is one process-wide object (bc._camera_gate, also installed in
    audio.kinect_bridge), and its whole job is to REMEMBER: a backoff rung, a
    lock, a USB-storm cool-down, the last open for the min-gap and the boot
    stagger. A test that fails an open would otherwise make every later test's
    open of that device a refusal - pass or fail on execution order. The same
    object is kept (only its state is cleared) so the bridge's reference stays
    valid, and it keeps its PRODUCTION rule values: tests that need a different
    rule build their own gate. Never raises."""
    try:
        gate = getattr(bc, "_camera_gate", None)
        if gate is not None:
            gate.reset()
    except Exception:
        pass


def _reset_filler_state(bc) -> None:
    """Reset the shared processing-filler objects to their import-time state.

    The rebind restore above puts back the same ProcessingFiller / ClipCache
    INSTANCES, so a test that ran a real teardown entry (the tray restart
    branch, the blue/green teardown, _release_native_resources) would leave
    the shared filler latched off (closed) for the rest of the process — and
    every later test using it would pass or fail for the wrong reason. Never
    raises."""
    try:
        f = bc._processing_filler
        with f._lock:
            f._closed = False
            f._current = None
            f._playing = 0
            f._captures = 0
            f.last_reason = ""
        f._idle.set()
    except Exception:
        pass
    try:
        c = bc._filler_clips
        with c._mu:
            c._clips.clear()
            c._rejected = set()
        c._warming = False
    except Exception:
        pass


@requires_monolith
class MonolithGlobalsTestCase(unittest.TestCase):
    """Base for every monolith test class.

    Loads the cached monolith once per class (``setUpClass``) and guarantees
    that the tracked bobert_companion globals are deep-restored to their
    pristine import-time values after EVERY test — wrapping the subclass's
    own setUp/test/tearDown in ``run()`` so the cleanup can't be shadowed by a
    subclass that overrides setUp/tearDown without calling ``super()``."""

    bc = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bc = load_monolith()

    def run(self, result=None):
        # On the light-deps CI runner / ci-sim the monolith can't import, so
        # these classes are @requires_monolith-skipped. Don't call
        # load_monolith() (it would raise ModuleNotFoundError before the skip
        # is reported) — just let super().run() record the skip.
        if not MONOLITH_AVAILABLE:
            return super().run(result)
        bc = self.bc if self.bc is not None else load_monolith()
        _capture_monolith_pristine(bc)
        # Neutralise owner-specific LLM routing (this dev box's user_settings.json
        # sets MODEL_ROUTING / AMBIENT_LEARNING_FORCE_LOCAL to local) so EVERY
        # monolith test runs against the shipped defaults and stays deterministic
        # regardless of the box it runs on. Tests that want a route override it
        # explicitly. Restored after the test.
        import core.config as _cfg
        _saved_route = dict(_cfg.MODEL_ROUTING)
        _saved_force = _cfg.AMBIENT_LEARNING_FORCE_LOCAL
        _cfg.MODEL_ROUTING = {"chat": "auto", "vision": "auto", "ambient": "auto"}
        _cfg.AMBIENT_LEARNING_FORCE_LOCAL = False
        # Start clean too, not only end clean: anything that ran before the
        # first monolith test (an import-time Kinect pump, a light-tier test)
        # may have left history in the process-wide camera gate.
        _reset_camera_gate(bc)
        try:
            return super().run(result)
        finally:
            _restore_monolith_pristine(bc)
            _cfg.MODEL_ROUTING = _saved_route
            _cfg.AMBIENT_LEARNING_FORCE_LOCAL = _saved_force
