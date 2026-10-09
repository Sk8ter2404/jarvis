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
  ``reinit_done()`` or ``yield_s``, whichever comes first. The refresh then
  WAITS (bounded, ``wait_settled(YIELD_WAIT_S)``) for that close to return
  before it returns itself (review fix, 2026-10-09): its caller --
  record_speech, right before its InputStream open -- must not open a
  stream while the keeper is still inside Pa_CloseStream.
* SINGLE TOUCHER. Every native call on the keeper's stream (open, start,
  abort, close) happens on ONE daemon thread ("tts-keeper"). Callers only
  change state under a plain lock and notify; nothing a caller does can block
  on PortAudio. The stream is never published into sounddevice's
  ``_last_callback`` slot (it is built with ``sd.OutputStream``), so no
  ``sd.play()``/``sd.stop()`` elsewhere can reach it.
* ONE OPEN AT A TIME ON THE SPEAKER (review fixes, 2026-10-05 and -09).
  PortAudio's open-stream list is not thread-safe (Pa_OpenStream links the
  new stream in at its END, Pa_CloseStream unlinks it at its START, with no
  lock), so a playback's own open never runs at the same moment as a keeper
  open or close, in EITHER order:
    - a playback calls ``enter_play_open()`` right before it opens its
      stream. A keeper open or close already IN FLIGHT is waited for until
      it returns (that costs no more than the cold open the playback would
      otherwise make itself); one that is only DUE -- a device change, a
      cold start, a block that just ran out -- is waited for at most
      ``READY_WAIT_S``. Then the playback marks its own open as in flight;
    - the keeper thread starts no open or close while a playback's open is
      marked, until ``exit_play_open()`` (or ``OPEN_WEDGED_S``, so a wedged
      native open on the playback's side can never stall the keeper for
      good).
  So giving up on a DUE call is safe (the mark holds it off). The one case
  that is not kept apart is a keeper call that has been in flight for
  ``OPEN_WEDGED_S`` (wedged in the driver): a playback cannot wait on it
  for ever, so it opens anyway -- and a keeper native call that ever takes
  that long QUARANTINES the keeper for the rest of the process (it closes
  what it holds and never opens again; logged once), so that can happen
  once per run at most.
  Scope: the fence keeps the keeper apart from the playbacks and from the
  re-enumeration's caller. The other streams (record_speech, the barge-in
  listener, the wake word, the tts-reaper's closes) are not fenced against
  the keeper, exactly as they are not fenced against each other; the
  keeper's own two calls per turn are timed away from them (the open at
  dispatch, the close ``LINGER_S`` after the reply).
* NO STALE INDEX. ``reinit_done()`` forgets the device; a keeper open that
  was decided before a re-enumeration and claimed after it re-checks the
  device under the lock and opens nothing (the claim waits out the reinit's
  latch, and ``reinit_done()`` runs while that latch is still held).
* OWNER CELL. ``claim()`` runs BEFORE the open and ``release()`` only AFTER
  the close returned, so the teardown gate sees the stream for its whole
  native lifetime (claim-before-open, close-before-release). A close that
  wedges in native code keeps the cell up -- which keeps sd._terminate() away
  from the live native call, exactly like _pa_close_pending.
* THE CALLBACK WRITES ZEROS AND NOTHING ELSE (no locks, no I/O, no logging).
* NEVER HOT-LOOPS. A refused claim or a failed open backs off (5 s, doubling
  to 60 s) and is logged once per streak.
* QUARANTINE. A keeper native open or close that took ``OPEN_WEDGED_S`` or
  longer switches the keeper off for the life of the process (see ONE OPEN
  AT A TIME). Playback itself is unaffected; only its fast open goes.
* OFF IS ABSENT. While disabled no thread is started and every call is a
  cheap no-op.

Stdlib only (light CI tier): the monolith injects ``open_stream`` / ``claim``
/ ``release``; tests inject fakes.
"""
from __future__ import annotations

import threading
import time

__all__ = ["PlaybackKeeper", "LINGER_S", "MAX_HOLD_S", "YIELD_S",
           "YIELD_WAIT_S", "READY_WAIT_S", "OPEN_WEDGED_S", "REAP_POLL_S",
           "UNSET"]

# Seconds the stream stays after the last holder ended. Covers the gap
# between a reply and its follow-up round, or a filler and the answer.
LINGER_S = 2.0
# One holder never keeps the stream longer than this (a token leaked by a
# raise, a turn whose actions run for minutes).
MAX_HOLD_S = 60.0
# After request_yield(): stay closed this long unless reinit_done() comes
# first, so a refresh pass (every DEVICE_CHECK_INTERVAL) can run.
YIELD_S = 10.0
# After request_yield(): how long _refresh_devices waits for the keeper's
# close to return before it returns to its caller (which opens the mic next).
# The close itself took 10 ms; the budget also covers a keeper OPEN that was
# in flight when the yield came (live play_open_ms max 699 ms, 10-01..06).
YIELD_WAIT_S = 1.0
# How long a playback waits for a keeper open or close that is DUE but not
# started yet (one already in flight is waited for until it returns, bounded
# by OPEN_WEDGED_S). The measured keeper open was 322 ms, its close 10 ms;
# play_open_ms p90 391.
READY_WAIT_S = 0.45
# A native open or close in flight longer than this is treated as wedged:
# playbacks stop waiting for a wedged keeper call at all (they would otherwise
# wait on it for as long as it hangs), the keeper stops waiting for a wedged
# playback open, and a keeper native call that took this long quarantines
# the keeper for the rest of the process.
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
        # A native call took OPEN_WEDGED_S or longer: off for good (see
        # QUARANTINE in the module docstring).
        self._quarantined = False
        # Stream state: written by the keeper thread only (under the lock).
        self._stream = None
        self._stream_dev = None
        self._opened_at = 0.0
        self._open_ms = 0
        # "open" / "close" from the moment the keeper thread commits to a
        # native call until it returned (an open's claim included), else None.
        self._native = None
        self._native_t0 = 0.0
        self._plays = 0
        # Playbacks opening their own stream right now: token -> start time
        # (enter_play_open .. exit_play_open). The keeper starts no native
        # call while one is marked (see ONE OPEN AT A TIME).
        self._play_opens: dict = {}
        self._next_play = 0
        self._pending: list = []          # log lines, printed off the lock
        # Lifetime counters (status / tests).
        self.opens = 0
        self.closes = 0
        self.failures = 0
        self.yields = 0
        self.dropped_opens = 0            # decided, then stale after the claim

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

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    # ── holders ──────────────────────────────────────────────────────────
    def begin(self, device=UNSET) -> int:
        """A reply (or one playback) is on its way: keep the endpoint awake
        until the matching end(). Returns a token for end(); 0 = disabled
        or quarantined (end(0) is a no-op). ``device``, when given, is the
        output device the speaking path will use; a different one than the
        keeper holds closes it and reopens it there. Never blocks on
        PortAudio."""
        tok, lines = 0, []
        try:
            with self._cv:
                if (not self._enabled or self._shutdown
                        or self._quarantined):
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
        """Wait (bounded) while the keeper has an open or close in flight or
        due: one in flight until it returns (at most OPEN_WEDGED_S), one only
        due for at most ``timeout``. True = none (any more); False = one is
        still due after ``timeout``, or the one in flight is wedged (no wait
        at all). _refresh_devices calls it after request_yield() so the
        keeper's close has returned before its caller opens a stream. Never
        raises."""
        try:
            with self._cv:
                return self._wait_quiet_locked(timeout)
        except Exception:
            return False

    def enter_play_open(self, timeout: float = READY_WAIT_S):
        """A playback is about to open its OWN stream on the speaker. Waits
        (bounded, see wait_settled) until the keeper has no open or close in
        flight or due, then marks the playback's open as in flight, so the
        keeper thread starts none until exit_play_open(token). Returns
        (token, settled); token 0 = the keeper never ran (nothing to keep
        apart from). The wait and the mark happen under one lock, so the
        keeper cannot slip a native call in between: settled=False after a
        call that was only DUE leaves it held off by the mark; only a keeper
        call wedged in flight (OPEN_WEDGED_S, which quarantines the keeper)
        can still be running. Never raises."""
        try:
            with self._cv:
                if self._thread is None:
                    return 0, True
                settled = self._wait_quiet_locked(timeout)
                self._next_play += 1
                tok = self._next_play
                self._play_opens[tok] = self._clock()
                return tok, settled
        except Exception:
            return 0, False

    def exit_play_open(self, token: int) -> None:
        """The playback's open returned (or raised): the keeper may make
        native calls again. Never raises."""
        if not token:
            return
        try:
            with self._cv:
                if self._play_opens.pop(token, None) is not None:
                    self._cv.notify_all()
        except Exception:
            pass

    def shutdown(self, wait_s: float = 0.0) -> bool:
        """Latch off for the life of the process and close the stream (on
        the keeper thread; the teardown gate's owner cell shows when that
        close has returned). With ``wait_s`` > 0, also wait that long (real
        time) for the close: True = no stream and no native call left (the
        interpreter-exit path, which must not let sounddevice's own exit
        handler terminate PortAudio under it). Never raises."""
        try:
            with self._cv:
                self._shutdown = True
                self._enabled = False
                self._holders.clear()
                self._linger_until = 0.0
                self._cv.notify_all()
                if wait_s <= 0:
                    return self._stream is None and self._native is None
                end = time.monotonic() + float(wait_s)
                while self._stream is not None or self._native is not None:
                    t = self._thread
                    if t is None or not t.is_alive():
                        return False
                    left = end - time.monotonic()
                    if left <= 0:
                        return False
                    self._cv.wait(left)
                return True
        except Exception:
            return False

    # ── the one-open-at-a-time rule ──────────────────────────────────────
    def _native_due_locked(self, now: float) -> bool:
        """True while the keeper thread has an open or close in flight, or
        one it will start as soon as it runs (caller holds the lock)."""
        if self._native is not None:
            return True
        t = self._thread
        if t is None or not t.is_alive():
            return False               # nobody would make that call
        want, _ = self._want_locked(now)
        if self._stream is None:
            return want
        return (not want) or self._stream_dev != self._device

    def _wait_quiet_locked(self, timeout: float) -> bool:
        """Bounded wait (caller holds the lock) until no keeper native call
        is in flight or due.

        * A call IN FLIGHT is waited for until it returns: a caller that
          went ahead would run its own open alongside it (review 2026-10-09:
          the old flat ``timeout`` let a 0.6 s keeper open overlap a play's
          open). Bounded by the wedge rule -- once the call has run
          OPEN_WEDGED_S on the keeper's clock it is not waited for at all --
          and by OPEN_WEDGED_S of real time from the start of this wait.
        * A call only DUE (not started) is waited for at most ``timeout``.

        Budgets are REAL time (the injected clock may stand still in tests);
        the wedge rule uses the keeper's clock."""
        start = time.monotonic()
        timeout = max(0.0, float(timeout))
        end = start + timeout
        hard_end = start + max(timeout, OPEN_WEDGED_S)
        while True:
            now = self._clock()
            if not self._native_due_locked(now):
                return True
            rt = time.monotonic()
            if self._native is not None:
                # In flight: until it returns, but never past the moment it
                # counts as wedged (then ``left`` <= 0 below: a wedged call
                # is not waited for at all), nor past hard_end.
                age = now - self._native_t0
                deadline = min(hard_end, rt + (OPEN_WEDGED_S - age))
            else:
                deadline = end
            left = deadline - rt
            if left <= 0:
                return False
            self._cv.wait(left)

    def _play_hold_locked(self, now: float):
        """Seconds the keeper must still hold off its native calls for a
        playback's open in flight, or None (caller holds the lock). An open
        marked longer ago than OPEN_WEDGED_S no longer counts (wedged)."""
        oldest = None
        for tok, t0 in list(self._play_opens.items()):
            age = now - t0
            if age >= self._max_hold_s:
                del self._play_opens[tok]          # bound the dict
                continue
            if age >= OPEN_WEDGED_S:
                continue
            if oldest is None or age > oldest:
                oldest = age
        if oldest is None:
            return None
        return max(0.01, OPEN_WEDGED_S - oldest)

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
        # (Quarantined: no holder or linger is left - _quarantine_locked
        # clears them and begin() adds none - so nothing is wanted.)
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
            with self._cv:
                while True:
                    now = self._clock()
                    want, wait_s = self._want_locked(now)
                    action = None
                    if self._stream is not None:
                        if not want or self._stream_dev != self._device:
                            action = "close"
                    elif want:
                        action = "open"
                    elif self._shutdown or self._quarantined:
                        self._thread = None
                        return
                    if action is not None:
                        hold = self._play_hold_locked(now)
                        if hold is None:
                            break
                        # A playback is opening its own stream on the
                        # speaker right now: no keeper call until it has
                        # returned (exit_play_open notifies), bounded by
                        # OPEN_WEDGED_S.
                        wait_s = hold if wait_s is None else min(wait_s, hold)
                    self._cv.wait(timeout=wait_s)
                dev = self._device
                self._native = action
                self._native_t0 = now
            if action == "open":
                self._do_open(dev)
            else:
                self._do_close()

    def _open_still_wanted_locked(self, dev) -> bool:
        """After the claim (which can wait out a re-enumeration's latch):
        still enabled, still wanted, not blocked, and ``dev`` still the
        device the speaking path uses, read AFTER any reinit_done()."""
        want, _ = self._want_locked(self._clock())
        return (want and self._stream is None and self._device_known
                and self._device == dev)

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
            with self._cv:
                still = self._open_still_wanted_locked(dev)
                if not still:
                    self.dropped_opens += 1
            if not still:
                # Decided before a re-enumeration (or a yield, an end, a
                # device change) that landed while the claim waited: that
                # index may name another device now. Open nothing; the loop
                # decides again from the current state.
                self._safe_release()
                claimed = False
                return
            n0 = self._clock()
            try:
                st = self._open_stream(dev)
            finally:
                took = self._clock() - n0
                if took >= OPEN_WEDGED_S:
                    with self._cv:
                        self._quarantine_locked(
                            f"its open on device {dev} took {took:.1f}s")
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
                self._native = None
                self._cv.notify_all()
                lines = self._take_pending_locked()
            for line in lines:
                self._say(line)

    def _do_close(self) -> None:
        st = self._stream
        n0 = self._clock()
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
            took = self._clock() - n0
            # Release only AFTER the native close returned (close-then-
            # release); a close that never returns keeps the cell up.
            self._safe_release()
            with self._cv:
                held = self._clock() - self._opened_at
                dev, open_ms, plays = self._stream_dev, self._open_ms, self._plays
                self._stream = None
                self._stream_dev = None
                self._native = None
                self.closes += 1
                if took >= OPEN_WEDGED_S:
                    self._quarantine_locked(
                        f"its close on device {dev} took {took:.1f}s")
                self._cv.notify_all()
                lines = self._take_pending_locked()
            self._say(f"  [playback-keeper] held the speaker (device {dev}) "
                      f"{held:.1f}s: open {open_ms} ms, {plays} "
                      f"play{'s' if plays != 1 else ''} while held")
            for line in lines:
                self._say(line)

    def _safe_release(self) -> None:
        try:
            self._release()
        except Exception:
            pass

    def _quarantine_locked(self, why: str) -> None:
        """A keeper native call took OPEN_WEDGED_S or longer (caller holds
        the lock): off for the rest of the process. A stream it holds is
        still closed by the thread (that releases the speaker and the owner
        cell); nothing is opened again. Logged once."""
        if self._quarantined:
            return
        self._quarantined = True
        self._holders.clear()
        self._linger_until = 0.0
        self._pending.append(
            f"  [playback-keeper] {why} (wedged: {OPEN_WEDGED_S:.0f}s or "
            f"more) -- the keeper is OFF for the rest of this run, so no "
            f"line opens the speaker alongside a hung keeper call again "
            f"(playback is unaffected, only slower to open; a restart "
            f"re-arms it)")
        self._cv.notify_all()

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
