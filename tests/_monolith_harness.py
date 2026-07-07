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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import bobert_companion as bc  # noqa: E402  (env must be set first)
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
    "_record_speech_active", "_tts_playback_active", "_camera_failure_summary",
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
        try:
            return super().run(result)
        finally:
            _restore_monolith_pristine(bc)
            _cfg.MODEL_ROUTING = _saved_route
            _cfg.AMBIENT_LEARNING_FORCE_LOCAL = _saved_force
