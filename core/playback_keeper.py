"""core/playback_keeper.py -- keep the speaker endpoint awake for one reply.

PLAYBACK_KEEPER (core/config.py), 2026-10-05.

WHY
===
Every clip JARVIS says is its own ``sd.play()`` stream on MME, and opening
that stream is bimodal: the 2026-09-30 probe file split 1,071 opens into 661
under 50 ms and 405 at 300-400 ms, and the owner's turns since 10-02 show
``play_open_ms`` p50 ~377 ms. The fast opens line up with OTHER audio playing
on the same speaker (10-02/10-04 YouTube, 10-05 00:23 Media Player: 43 / 42 /
28 ms while it played, 352 ms right after it stopped).

The owner-approved SILENT check, 2026-10-05 00:35 (zeros only, his
"Speakers (Realtek USB2.0 Audio)", MME index 6, nothing else playing):
  * sd.play() with nothing else open: 10/10 opens 320-346 ms (median 321),
    and 4/4 again afterwards (320-340);
  * the same with a zero-filled OutputStream held open on that speaker:
    10/10 opens 3.6-9.7 ms (median 3.8);
  * the time is Pa_OpenStream (median 321 ms), not Pa_StartStream (0.2 ms);
  * WASAPI shared is no different: 321-333 ms without, 3.8-6.2 ms with;
  * the keeper's own open is the one slow open (322 ms), its abort + close
    10 ms.
So the fix is to have "other audio playing" ourselves -- silence -- while a
reply is being prepared and spoken: the first line opens fast, and so does
every sentence after it (each sentence is still a new stream). The keeper's
own ~0.3 s open runs on its thread while the brain thinks and the line
renders, off the critical path.

CONTRACT (every rule below is load-bearing)
===========================================
* PER TURN, NEVER PERMANENT. The stream exists only while a HOLDER is active
  (a voice turn being answered, one _speak, one playback) plus ``linger_s``
  after the last one ends. A holder counts for at most ``max_hold_s``. A
  stream that never closed would keep the monolith's _refresh_devices from
  ever running its destructive PortAudio re-enumeration (it defers while any
  owner cell is set), so a replugged mic would never be followed.
* YIELDS TO THE RE-ENUMERATION. When _refresh_devices wants that
  re-enumeration and the keeper is the only owner in the way, it calls
  ``request_yield()``: the keeper closes at once and stays closed until
  ``reinit_done()`` or ``yield_s``, whichever comes first.
* SINGLE TOUCHER. Every native call on the keeper's stream (open, start,
  abort, close) happens on ONE daemon thread ("tts-keeper"). Callers only
  change state under a plain lock and notify; nothing a caller does can block
  on PortAudio. The stream is never published into sounddevice's
  ``_last_callback`` slot (it is built with ``sd.OutputStream``), so no
  ``sd.play()``/``sd.stop()`` elsewhere can reach it.
* OWNER CELL. ``claim()`` runs BEFORE the open and ``release()`` only AFTER
  the close returned, so the teardown gate sees the stream for its whole
  native lifetime (claim-before-open, close-before-release). A close that
  wedges in native code keeps the cell up -- which keeps sd._terminate() away
  from the live native call, exactly like _pa_close_pending.
* THE CALLBACK WRITES ZEROS AND NOTHING ELSE (no locks, no I/O, no logging).
* NEVER HOT-LOOPS. A refused claim or a failed open backs off (5 s, doubling
  to 60 s) and is logged once per streak.
* OFF IS ABSENT. While disabled no thread is started and every call is a
  cheap no-op.

Stdlib only (light CI tier): the monolith injects ``open_stream`` / ``claim``
/ ``release``; tests inject fakes.
"""
from __future__ import annotations

import threading
import time

__all__ = ["PlaybackKeeper", "LINGER_S", "MAX_HOLD_S", "YIELD_S",
           "READY_WAIT_S", "OPEN_WEDGED_S", "REAP_POLL_S", "UNSET"]

# Seconds the stream stays after the last holder ended. Covers the gap
# between a reply and its follow-up round, or a filler and the answer.
LINGER_S = 2.0
# One holder never keeps the stream longer than this (a token leaked by a
# raise, a turn whose actions run for minutes).
MAX_HOLD_S = 60.0
# After request_yield(): stay closed this long unless reinit_done() comes
# first, so a refresh pass (every DEVICE_CHECK_INTERVAL) can run.
YIELD_S = 10.0
# How long a playback waits (pure Event wait) for a keeper open that is
# already in flight before it opens its own stream (the measured keeper open
# was 322 ms; play_open_ms p90 391).
READY_WAIT_S = 0.45
# A keeper open in flight longer than this is treated as wedged in native
# code: playbacks stop waiting for it at all (they would otherwise pay
# READY_WAIT_S each, for as long as it hangs).
OPEN_WEDGED_S = 2.0
# The playback reaper's poll interval while the keeper is on (it is 0.05 s
# when it is off): the gap from a clip's natural end to its close, and the
# barge-in cut latency.
REAP_POLL_S = 0.01

_BACKOFF_FIRST_S = 5.0
_BACKOFF_MAX_S = 60.0
UNSET = object()


class PlaybackKeeper:
    """See the module docstring. Thread-safe; no method ever raises."""

    def __init__(self, *, open_stream, claim, release, log=print,
                 clock=time.monotonic, linger_s: float = LINGER_S,
                 max_hold_s: float = MAX_HOLD_S, yield_s: float = YIELD_S,
                 thread_factory=threading.Thread):
        self._open_stream = open_stream
        self._claim = claim
        self._release = release
        self._log = log
        self._clock = clock
        self._linger_s = float(linger_s)
        self._max_hold_s = float(max_hold_s)
        self._yield_s = float(yield_s)
        self._thread_factory = thread_factory
        self._cv = threading.Condition(threading.Lock())
        self._enabled = False
        self._shutdown = False
        self._thread = None
        self._holders: dict = {}          # token -> begin time
        self._next_token = 0
        self._linger_until = 0.0
        self._device = None               # what the speaking path uses now
        # False after a PortAudio re-enumeration: every index read before it
        # may now name another device (MME renumbers), so nothing opens
        # until a caller hands over a device read after it.
        self._device_known = True
        self._blocked_until = 0.0         # yield / failure back-off
        self._block_is_yield = False
        self._backoff_s = _BACKOFF_FIRST_S
        self._fail_logged = False
        # Stream state: written by the keeper thread only (under the lock).
        self._stream = None
        self._stream_dev = None
        self._opened_at = 0.0
        self._open_ms = 0
        self._opening = False
        self._open_started = 0.0
        self._settled = threading.Event()  # set while no open is in flight
        self._settled.set()
        self._plays = 0
        self._pending: list = []          # log lines, printed off the lock
        # Lifetime counters (status / tests).
        self.opens = 0
        self.closes = 0
        self.failures = 0
        self.yields = 0

    # ── configuration ────────────────────────────────────────────────────
    def set_enabled(self, on: bool) -> None:
        """Turn the keeper on or off. Off closes a live stream (on the
        keeper thread) and never starts one again until it is turned on."""
        try:
            with self._cv:
                self._enabled = bool(on) and not self._shutdown
                if not self._enabled:
                    self._holders.clear()
                    self._linger_until = 0.0
                self._cv.notify_all()
        except Exception:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── holders ──────────────────────────────────────────────────────────
    def begin(self, device=UNSET) -> int:
        """A reply (or one playback) is on its way: keep the endpoint awake
        until the matching end(). Returns a token for end(); 0 = disabled
        (end(0) is a no-op). ``device``, when given, is the output device
        the speaking path will use. Never blocks on PortAudio."""
        tok, lines = 0, []
        try:
            with self._cv:
                if not self._enabled or self._shutdown:
                    return 0
                if device is not UNSET:
                    self._device = device
                    self._device_known = True
                self._next_token += 1
                tok = self._next_token
                self._holders[tok] = self._clock()
                self._ensure_thread()
                self._cv.notify_all()
                lines = self._take_pending_locked()
        except Exception:
            return 0
        for line in lines:
            self._say(line)
        return tok

    def end(self, token: int) -> None:
        """The holder ``token`` is done; the stream lingers ``linger_s``."""
        if not token:
            return
        try:
            with self._cv:
                if self._holders.pop(token, None) is not None:
                    self._linger_until = max(self._linger_until,
                                             self._clock() + self._linger_s)
                self._cv.notify_all()
        except Exception:
            pass

    def note_device(self, device) -> None:
        """The output device a playback is about to open. A keeper on
        another device is closed and reopened on this one."""
        try:
            with self._cv:
                if self._device != device or not self._device_known:
                    self._device = device
                    self._device_known = True
                    self._cv.notify_all()
        except Exception:
            pass

    def note_play(self) -> None:
        """Count one playback opened while the keeper was live (log only)."""
        try:
            with self._cv:
                if self._stream is not None:
                    self._plays += 1
        except Exception:
            pass

    # ── the re-enumeration ───────────────────────────────────────────────
    def request_yield(self) -> None:
        """_refresh_devices wants its destructive re-enumeration and the
        keeper is the only owner in the way: close now, stay closed until
        reinit_done() or yield_s. Never blocks."""
        try:
            with self._cv:
                if not self._enabled:
                    return
                until = self._clock() + self._yield_s
                if until > self._blocked_until:
                    self._blocked_until = until
                    self._block_is_yield = True
                self.yields += 1
                self._cv.notify_all()
        except Exception:
            pass

    def reinit_done(self) -> None:
        """The re-enumeration ran: a yield block ends now (a failure
        back-off does not), and the device index the keeper knew is
        forgotten until a caller hands over a fresh one."""
        try:
            with self._cv:
                self._device_known = False
                if self._block_is_yield:
                    self._blocked_until = 0.0
                    self._block_is_yield = False
                    self._cv.notify_all()
        except Exception:
            pass

    # ── readers ──────────────────────────────────────────────────────────
    def is_live(self) -> bool:
        """True while the keeper's stream is open, started and on the device
        the speaking path uses now."""
        try:
            return (self._stream is not None and self._device_known
                    and self._stream_dev == self._device)
        except Exception:
            return False

    def wait_settled(self, timeout: float = READY_WAIT_S) -> bool:
        """Wait (bounded, pure Event) while a keeper open is in flight, so a
        playback does not race it into the endpoint. True = no open in
        flight (any more). Never raises."""
        try:
            if not self._opening:
                return True
            left = OPEN_WEDGED_S - (self._clock() - self._open_started)
            if left <= 0:
                return False         # wedged: do not wait for it at all
            return bool(self._settled.wait(
                max(0.0, min(float(timeout), left))))
        except Exception:
            return False

    def shutdown(self) -> None:
        """Latch off for the life of the process and close the stream (on
        the keeper thread; the teardown gate's owner cell shows when that
        close has returned)."""
        try:
            with self._cv:
                self._shutdown = True
                self._enabled = False
                self._holders.clear()
                self._linger_until = 0.0
                self._cv.notify_all()
        except Exception:
            pass

    # ── the keeper thread ────────────────────────────────────────────────
    def _ensure_thread(self) -> None:
        # Caller holds the lock.
        t = self._thread
        if t is not None and t.is_alive():
            return
        try:
            t = self._thread_factory(target=self._run, name="tts-keeper",
                                     daemon=True)
            t.start()
            self._thread = t
        except Exception as e:      # thread exhaustion: no keeper, no harm
            self._thread = None
            self._note_failure_locked(f"could not start its thread: {e}")

    def _want_locked(self, now: float):
        """(want_open, seconds until the decision may change by itself, or
        None: nothing changes until a caller notifies)."""
        if self._shutdown or not self._enabled:
            return False, None
        for tok, t0 in list(self._holders.items()):
            if now - t0 >= self._max_hold_s:
                # A holder past its bound: dropped, so a leaked token can
                # never keep the endpoint (and the re-enumeration gate) held.
                del self._holders[tok]
        pending = bool(self._holders) or now < self._linger_until
        if not pending:
            return False, None
        if now < self._blocked_until:
            return False, max(0.01, self._blocked_until - now)
        if not self._device_known:
            return False, None
        if self._holders:
            oldest = min(self._holders.values())
            return True, max(0.01, oldest + self._max_hold_s - now)
        return True, max(0.01, self._linger_until - now)

    def _run(self) -> None:
        while True:
            action = None
            with self._cv:
                while True:
                    now = self._clock()
                    want, wait_s = self._want_locked(now)
                    if self._stream is not None:
                        if not want or self._stream_dev != self._device:
                            action = "close"
                            break
                    elif want:
                        action = "open"
                        dev = self._device
                        self._opening = True
                        self._open_started = now
                        self._settled.clear()
                        break
                    elif self._shutdown:
                        self._thread = None
                        return
                    self._cv.wait(timeout=wait_s)
            if action == "open":
                self._do_open(dev)
            else:
                self._do_close()

    def _do_open(self, dev) -> None:
        t0 = self._clock()
        claimed = False
        try:
            claimed = bool(self._claim())
            if not claimed:
                with self._cv:
                    self._note_failure_locked(
                        "the PortAudio re-enumeration held the owner gate")
                return
            st = self._open_stream(dev)
            with self._cv:
                self._stream = st
                self._stream_dev = dev
                self._opened_at = self._clock()
                self._open_ms = int(round((self._opened_at - t0) * 1000.0))
                self._plays = 0
                self.opens += 1
                self._backoff_s = _BACKOFF_FIRST_S
                if self._fail_logged:
                    self._fail_logged = False
                    self._pending.append("  [playback-keeper] holding the "
                                         "speaker again")
        except BaseException as e:
            if claimed:
                self._safe_release()
            with self._cv:
                self._note_failure_locked(
                    f"open failed on device {dev}: {type(e).__name__}: {e}")
        finally:
            with self._cv:
                self._opening = False
                self._settled.set()
                self._cv.notify_all()
                lines = self._take_pending_locked()
            for line in lines:
                self._say(line)

    def _do_close(self) -> None:
        st = self._stream
        try:
            try:
                st.abort(ignore_errors=True)
            except Exception:
                pass
            try:
                st.close(ignore_errors=True)
            except Exception:
                pass
        finally:
            # Release only AFTER the native close returned (close-then-
            # release); a close that never returns keeps the cell up.
            self._safe_release()
            with self._cv:
                held = self._clock() - self._opened_at
                dev, open_ms, plays = self._stream_dev, self._open_ms, self._plays
                self._stream = None
                self._stream_dev = None
                self.closes += 1
                self._cv.notify_all()
            self._say(f"  [playback-keeper] held the speaker (device {dev}) "
                      f"{held:.1f}s: open {open_ms} ms, {plays} "
                      f"play{'s' if plays != 1 else ''} while held")

    def _safe_release(self) -> None:
        try:
            self._release()
        except Exception:
            pass

    def _note_failure_locked(self, why: str) -> None:
        # Caller holds the lock.
        self.failures += 1
        now = self._clock()
        self._blocked_until = max(self._blocked_until, now + self._backoff_s)
        self._block_is_yield = False
        backoff = self._backoff_s
        self._backoff_s = min(_BACKOFF_MAX_S, self._backoff_s * 2.0)
        if not self._fail_logged:
            self._fail_logged = True
            self._pending.append(
                f"  [playback-keeper] {why} -- not holding the speaker for "
                f"{backoff:.0f}s (playback is unaffected; logged once until "
                f"it works again)")

    def _take_pending_locked(self) -> list:
        lines, self._pending = self._pending, []
        return lines

    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:
            pass
