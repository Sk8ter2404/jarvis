"""
Claude API credits monitor skill for JARVIS.

Actions added:
  check_credits  — open console.anthropic.com/settings/billing in a NEW Chrome
                   window parked OFF-SCREEN (outside the virtual screen bounding
                   box, never visible on any monitor), capture its pixels via
                   win32 PrintWindow(PW_RENDERFULLCONTENT), use vision to read
                   the credit balance, then close the window. Reuses the user's
                   logged-in Chrome session. Returns the dollar amount remaining.

Background monitor (started automatically when this skill loads, if
ENABLE_BACKGROUND_MONITOR is True):
  Every CHECK_INTERVAL_SECONDS, runs the same check silently. If the balance
  drops below ALERT_THRESHOLD_DOLLARS, writes a TTS alert to
  pending_speech.json so JARVIS speaks it on the next idle pass without
  interrupting whatever the user is doing.

Trigger phrases JARVIS should map to check_credits:
  "check my credits"
  "how many credits do I have"
  "what is my Claude balance"
  "am I running low on credits"
"""
import json
import logging
import os
import re
import threading
import time

from core.atomic_io import _atomic_write_json

# ---- configuration ------------------------------------------------------
BILLING_URL               = "https://console.anthropic.com/settings/billing"
CHECK_INTERVAL_SECONDS    = 3600     # 1 hour between background checks
ALERT_THRESHOLD_DOLLARS   = 5.0      # speak when balance drops below this
PAGE_LOAD_WAIT            = 6.0      # seconds to wait for the page to render
ENABLE_BACKGROUND_MONITOR = True     # set False to disable the periodic checker
INITIAL_DELAY_SECONDS     = 3600     # wait a full hour before the first background check
                                     # (same as CHECK_INTERVAL_SECONDS — avoids firing on
                                     # every restart when JARVIS is being relaunched frequently)
ALERT_COOLDOWN_SECONDS    = 4 * 3600 # don't repeat a low-balance alert more than this often
# -------------------------------------------------------------------------

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "credits_state.json")

# Mutable holders for thread state. The two alert stamps are PERSISTED in
# credits_state.json (2026-10-01): held only in memory, the documented 12 h
# login / 4 h low-balance cooldowns reset at every restart, so each session
# that ran past the first hourly check spoke the login alert again — seven
# times across 09-29/09-30, at night too.
_last_alert_at = [0.0]
_last_login_alert_at = [0.0]
_check_lock = threading.Lock()   # serialize: only one check at a time
_speech_lock = threading.Lock()  # guard pending_speech.json read-modify-write


def _enqueue_speech(message: str):
    """Route a proactive announcement through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json if the parent module hasn't loaded yet
    (e.g. unit test, import-time skill registration before bobert_companion
    finishes initialising). Matches the canonical pattern used by
    skills/bambu_monitor.py, skills/weather_briefing.py, skills/wellness.py,
    etc., so every co-writer of pending_speech.json funnels through the same
    atomic helper and there's no per-skill race drift."""
    try:
        bc = _import_companion()
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="credits")
            return
    except Exception:
        pass

    with _speech_lock:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            # Atomic write failed (e.g. read-only network share, full disk,
            # permission denied). Fall back to console so the alert isn't
            # silently lost — at minimum the user sees it in the log stream.
            print(f"  [credits] speech-queue write failed ({e}); alert: {message}")


def _save_state(balance, raw: str):
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "checked_at": time.time(),
                "balance":    balance,
                "raw":        raw,
                # The alert cooldowns, so a restart can't reset them.
                "last_alert_at":       _last_alert_at[0],
                "last_login_alert_at": _last_login_alert_at[0],
            }, f, indent=2)
    except Exception:
        pass


def _load_alert_cooldowns() -> None:
    """Restore the persisted alert stamps into the in-memory cells (keeping
    whichever is newer). A missing / corrupt file, or a stamp from the
    future, leaves the cells alone. Never raises."""
    try:
        if not os.path.exists(_STATE_FILE):
            return
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        now = time.time()
        for key, cell in (("last_alert_at", _last_alert_at),
                          ("last_login_alert_at", _last_login_alert_at)):
            v = data.get(key)
            if isinstance(v, (int, float)) and 0 < v <= now + 60:
                cell[0] = max(cell[0], float(v))
    except Exception:
        pass


def _import_companion():
    """Reach back into bobert_companion for the helpers we need.
    Importing at call time (not at module load) avoids any circular-import
    issue when the skill loader pulls us in."""
    import importlib
    return importlib.import_module("bobert_companion")


# CREDITS_CHECK_BACKEND (core.config, 2026-10-02): "auto" reads the billing
# screenshot through ask_vision (routed by MODEL_ROUTING["vision"], unchanged);
# "local" always uses the local vision model, so the billing page never goes to
# Claude. Unknown / unreadable values mean "auto" — the shipped default.
_CREDITS_BACKENDS = ("auto", "local")


def _credits_backend() -> str:
    try:
        from core import config as _cfg
        v = str(getattr(_cfg, "CREDITS_CHECK_BACKEND", "") or "").strip().lower()
    except Exception:
        v = ""
    return v if v in _CREDITS_BACKENDS else "auto"


def _ask_local_vision(bc, question: str, png: bytes) -> str:
    """The local vision model's answer, in ask_vision's own shapes: tagged
    "[local-vision] …" on success, a parenthesised failure line otherwise.
    Never calls Claude. Honours SCREEN_VISION_ENABLED like ask_vision."""
    if not getattr(bc, "SCREEN_VISION_ENABLED", True):
        return "(screen vision is disabled — set SCREEN_VISION_ENABLED = True)"
    local = bc._call_local_vision(question, [png])
    if local:
        return f"[local-vision] {local}"
    return ("(local vision unavailable — CREDITS_CHECK_BACKEND is local; turn "
            "on LOCAL_VISION_FALLBACK or route vision to local)")


def _read_credits_via_vision():
    """Open billing page in an OFF-SCREEN Chrome window, vision-read the
    balance, then close it. The window is parked outside the virtual screen
    bounding box for its entire lifetime, so it never appears on any
    monitor. Returns (dollars_or_None, raw_vision_answer_or_status)."""
    bc = _import_companion()

    # Spawn the billing page in a brand-new Chrome window kept off-screen
    # the whole time. _open_url_offscreen_capture handles spawn → park →
    # wait → PrintWindow capture → close, reusing the user's logged-in
    # session (so the page renders the actual balance, not a login wall).
    try:
        png, status = bc._open_url_offscreen_capture(BILLING_URL, PAGE_LOAD_WAIT)
    except Exception as e:
        return None, f"open_failed: {e}"
    if png is None:
        return None, f"capture_failed: {status}"

    question = (
        "This is a screenshot of the Anthropic console billing page. "
        "Find the CURRENT CREDIT BALANCE — the dollar amount of credits "
        "remaining on the account (often labelled 'Credit balance', "
        "'Remaining credits', or shown prominently near the top of the "
        "page).\n\n"
        "Reply with ONLY one line, exactly one of:\n"
        "  BALANCE: $X.XX     (substitute the real amount)\n"
        "  LOGIN_REQUIRED     (if the page is showing a sign-in screen)\n"
        "  NOT_FOUND          (if you can't see a balance for any other reason)"
    )

    try:
        if _credits_backend() == "local":
            answer = _ask_local_vision(bc, question, png).strip()
        else:
            answer = bc.ask_vision(question, png).strip()
    except Exception as e:
        answer = f"vision_failed: {e}"

    # No cleanup needed — _open_url_offscreen_capture already closed the
    # window it spawned.

    upper = answer.upper()
    if "LOGIN_REQUIRED" in upper:
        return None, "login_required"
    if "NOT_FOUND" in upper:
        return None, answer

    m = re.search(r"\$\s*([0-9][0-9,]*\.?[0-9]*)", answer)
    if m:
        try:
            dollars = float(m.group(1).replace(",", ""))
            return dollars, answer
        except ValueError:
            return None, answer
    return None, answer


def _background_work(tag: str):
    """core.local_traffic.background_work(tag, opt_in=True) -- a speed plan
    R5 tag, which waits only while BACKGROUND_TAG_STRICT is on -- or a no-op
    context when core can't be imported. Never raises."""
    try:
        from core import local_traffic as _lt
        return _lt.background_work(tag, opt_in=True)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def _wait_for_local_quiet() -> str:
    """The running JARVIS's wait_for_local_quiet("vision"): for this tagged
    thread, wait (bounded) until a background vision call may run. Never
    imports the monolith (a bare import is not a running JARVIS); 'none'
    when it isn't loaded. Never raises.

    With CREDITS_CHECK_BACKEND "local" the read goes to the local vision model
    whatever MODEL_ROUTING["vision"] says, so the monolith's routing answer
    ('cloud' when vision is not routed local) does not apply: the cycle waits
    on the shared gate like any local read, BEFORE the capture (2026-10-02
    integration audit - otherwise the wait happened inside _call_local_vision,
    after the billing page was already captured, with the owner mid-turn)."""
    try:
        import sys
        bc = sys.modules.get("bobert_companion")
        if bc is None:
            return "none"
        if _credits_backend() == "local":
            from core import local_traffic as _lt
            return _lt.wait_for_quiet()
        waiter = getattr(bc, "wait_for_local_quiet", None)
        return waiter("vision") if callable(waiter) else "none"
    except Exception:
        return "none"


def _check_and_maybe_alert():
    """One periodic-monitor cycle. Reads the balance and queues a TTS alert
    if it has dropped below the configured threshold.

    Speed plan R5 (2026-10-02): the cycle is background work for the one-slot
    local model (its vision read goes local on a local-vision install), so it
    is tagged, and with BACKGROUND_TAG_STRICT on it waits (bounded) for a
    quiet moment BEFORE it takes the check lock and opens the billing page:
    the capture is fresh when the read goes out, and the owner's own "check my
    credits" is never told a check is running while this one only waits."""
    with _background_work("credits-monitor"):
        _wait_for_local_quiet()
        _check_and_maybe_alert_now()


def _check_and_maybe_alert_now():
    """The cycle itself (see _check_and_maybe_alert)."""
    if not _check_lock.acquire(blocking=False):
        return   # another check is already running — skip this tick
    try:
        try:
            dollars, raw = _read_credits_via_vision()
        except Exception as e:
            print(f"  [credits] background check error: {e}")
            return
        now = time.time()
        if dollars is None:
            if raw == "login_required" and (now - _last_login_alert_at[0]) > 12 * 3600:
                _enqueue_speech(
                    "Sir, I tried to check your Claude credits but the "
                    "console is asking for a login."
                )
                _last_login_alert_at[0] = now
            # Saved AFTER the alert decision so the new stamp is on disk.
            _save_state(dollars, raw)
            return
        print(f"  [credits] background check: ${dollars:.2f}")
        if dollars < ALERT_THRESHOLD_DOLLARS:
            if (now - _last_alert_at[0]) > ALERT_COOLDOWN_SECONDS:
                _enqueue_speech(
                    f"Heads up, sir — your Claude credits are getting low. "
                    f"Only ${dollars:.2f} remaining."
                )
                _last_alert_at[0] = now
        _save_state(dollars, raw)
    finally:
        _check_lock.release()


def _background_monitor_loop():  # pragma: no cover - daemon while-True monitor loop; blocks on time.sleep(CHECK_INTERVAL_SECONDS) between live billing-page captures. Its one work step, _check_and_maybe_alert(), is unit-tested directly.
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        try:
            _check_and_maybe_alert()
            time.sleep(CHECK_INTERVAL_SECONDS)
        except Exception:
            logging.exception("[credits] monitor loop iteration failed")
            try:
                time.sleep(CHECK_INTERVAL_SECONDS)
            except Exception:
                logging.exception("[credits] monitor loop sleep failed")


def register(actions):
    def check_credits(_: str = "") -> str:
        if not _check_lock.acquire(blocking=False):
            return "credits check already in progress — give it a moment"
        try:
            try:
                dollars, raw = _read_credits_via_vision()
            except Exception as e:
                return f"credits check failed: {e}"
            _save_state(dollars, raw)
            if dollars is None:
                if raw == "login_required":
                    return ("the Anthropic console wants a login first — "
                            "sign in once and I'll be able to read it after")
                if raw == "screenshot_failed":
                    return "couldn't screenshot the billing page"
                return f"couldn't read balance from page (vision said: {raw[:120]})"
            return f"Claude credit balance: ${dollars:.2f} remaining"
        finally:
            _check_lock.release()

    actions["check_credits"] = check_credits

    # Restore the alert cooldowns BEFORE anything can save state or alert.
    _load_alert_cooldowns()

    if ENABLE_BACKGROUND_MONITOR:
        # Guard against duplicate loops on skill reload (load_skills re-execs
        # the module → fresh globals; only an OS-thread name check survives).
        # A duplicate here would independently re-screenshot the billing page
        # every CHECK_INTERVAL_SECONDS — wasteful and rate-limit-risky.
        if any(th.name == "credits-monitor" and th.is_alive()
               for th in threading.enumerate()):
            print("  [credits] monitor already running — skipping duplicate "
                  "(skill reload)")
        else:
            t = threading.Thread(target=_background_monitor_loop, daemon=True,
                                 name="credits-monitor")
            t.start()
            print(
                f"  [credits] background monitor active — checks every "
                f"{CHECK_INTERVAL_SECONDS}s, alert below ${ALERT_THRESHOLD_DOLLARS:.2f}"
            )
