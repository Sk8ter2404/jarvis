"""core/mic_bus.py - one always-open microphone stream with a ring buffer
(MIC_BUS_MODE, 2026-10-05).

WHY THIS EXISTS
---------------
record_speech opened a NEW input stream for every capture and closed it
after, so nothing was heard between the end of one capture and the next
"Recording...": 744 of 6,057 s (12.3 %) of the 10-05 00:21 session, 17-20 %
of media time across the logs, a median 4 s (max 18 s) after a 30 s capture
- and ~200 stream open / close cycles an hour, each a trip through the
0xc0000374 close path. The bus keeps ONE stream open on the selected mic and
keeps the last RING_S seconds, so

  * a capture's pre-roll, an overlapping media segment (B2) and a wake-word
    cut (D1) can reach back into audio heard BEFORE the capture started;
  * the echo canceller (core/audio_processor.MediaEchoCanceller) sees one
    continuous stream and stays converged.

DESIGN
------
  * ONE owner thread (never exits) is the only thread that opens or closes
    the stream. The monolith claims / releases the PortAudio owner cell
    around it (``claim`` / ``release``: _pa_claim_owner on _mic_bus_active),
    and closes through its bounded, abandonable close (``close_stream``:
    _safe_close_stream -> _pa_close_handoff), so the PortAudio reinit gate
    and the H-6 pending-close accounting cover the bus exactly as they cover
    a capture stream.
  * Opens happen only when a capture asks (``ensure(device)``), on the
    capture's own pacing - the R10 back-off stays the one place that paces
    a mic that will not open. Once open the stream stays open across
    captures until: the mic is muted or disabled (``should_run`` False - the
    tray's mic-pause CLOSES the stream, it does not just drop frames), a
    device refresh needs PortAudio (``suspend`` / ``resume``), the capture
    asks for another device, frames stop arriving (the stream died), or the
    bus is turned off.
  * The callback only copies, indexes and enqueues (bounded; the oldest
    frame is dropped and counted). A DSP worker thread (never exits) runs
    ``process`` (the echo canceller, when on), writes the rings and fans
    each frame out to subscribers - bounded queues that drop their oldest
    frame, so a slow consumer can never stall capture.

Frames are indexed by a monotonic sample counter (``n_end`` = the index just
past the frame's last sample); rings: raw, linear (echo removed) and
suppressed (for detection). Never raises out of its public methods.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from typing import NamedTuple

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 1024
RING_S = 30.0
CALLBACK_QUEUE_MAX = 256         # ~16 s of 1024-sample frames
SUBSCRIBER_QUEUE_MAX = 128       # ~8 s
DEAD_AFTER_S = 2.0               # no frame this long = the stream died
ENSURE_WAIT_S = 2.5
SUSPEND_WAIT_S = 2.5
POLL_S = 0.1


class BusUnavailable(RuntimeError):
    """The bus cannot open right now for a reason that is not a device
    failure (muted / off, suspended for a device refresh, the PortAudio
    claim refused): the capture skips this cycle and books nothing."""


class Frame(NamedTuple):
    n_end: int          # sample index just past the frame's last sample
    t: float            # monotonic time the frame arrived
    raw: np.ndarray     # the mic as recorded
    lin: np.ndarray     # echo removed (== raw when no canceller runs)
    sup: np.ndarray     # residual-suppressed (detection only; == lin
                        # when no canceller runs)
    aec: bool           # the canceller produced lin / sup


class Subscription:
    """A reader's bounded queue of Frames. ``dropped`` counts frames lost
    to a full queue (oldest first)."""

    def __init__(self, bus, maxsize: int = SUBSCRIBER_QUEUE_MAX):
        self._bus = bus
        self._q: "queue.Queue[Frame]" = queue.Queue(maxsize=max(1, maxsize))
        self.dropped = 0
        self.closed = False

    def _offer(self, fr: Frame) -> None:
        while True:
            try:
                self._q.put_nowait(fr)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def get(self, block: bool = True, timeout: "float | None" = None) -> Frame:
        """The next Frame - queue.Queue's own semantics (raises queue.Empty
        on a timeout), so record_speech reads a subscription exactly like
        its own capture queue."""
        return self._q.get(block, timeout)

    def qsize(self) -> int:
        return self._q.qsize()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._bus._unsubscribe(self)


class _Ring:
    def __init__(self, n: int):
        self.n = int(n)
        self.buf = np.zeros(self.n, np.float32)

    def write(self, n_end: int, x: np.ndarray) -> None:
        m = min(len(x), self.n)
        x = x[-m:]
        i = (n_end - m) % self.n
        end = i + m
        if end <= self.n:
            self.buf[i:end] = x
        else:
            cut = self.n - i
            self.buf[i:] = x[:cut]
            self.buf[:m - cut] = x[cut:]

    def read(self, start: int, stop: int) -> np.ndarray:
        i = start % self.n
        m = stop - start
        end = i + m
        if end <= self.n:
            return self.buf[i:end].copy()
        return np.concatenate([self.buf[i:], self.buf[:end - self.n]])


class MicBus:
    """The always-open microphone (see the module docstring).

    Injected (the monolith's, or a test's fakes):
      open_stream(device, callback) -> a STARTED stream (raises on failure)
      close_stream(stream)           bounded, abandonable close
      claim() -> bool / release()    the PortAudio owner cell
      should_run() -> bool           not muted / disabled / turned off
      process(mono, t) -> (lin, sup) or None   the echo canceller
      tap_fanout(frame)              the legacy record taps (ambient ...):
                                     the monolith picks the frame's raw or
                                     echo-cancelled copy
      tap_allowed() -> bool          False while a private capture runs
      on_reopen()                    a new stream: the canceller re-anchors
    """

    def __init__(self, *, open_stream, close_stream, claim=None,
                 release=None, should_run=None, process=None,
                 tap_fanout=None, tap_allowed=None, on_reopen=None,
                 sample_rate: int = SAMPLE_RATE, chunk: int = CHUNK,
                 ring_s: float = RING_S, clock=time.monotonic, log=print):
        self.sr = int(sample_rate)
        self.chunk = int(chunk)
        self._open_stream = open_stream
        self._close_stream = close_stream
        self._claim = claim or (lambda: True)
        self._release = release or (lambda: None)
        self._should_run = should_run or (lambda: True)
        self.process = process
        self._tap_fanout = tap_fanout
        self._tap_allowed = tap_allowed or (lambda: True)
        self._on_reopen = on_reopen
        self._clock = clock
        self._log = log
        self._cv = threading.Condition()
        self._cbq: "collections.deque" = collections.deque()
        self._cb_cv = threading.Condition()
        self._subs: list = []
        self._listeners: list = []
        self._subs_lock = threading.Lock()
        self._t_last_frame = None
        self._dsp_last_n = None
        n = int(ring_s * self.sr)
        self._rings = {"raw": _Ring(n), "lin": _Ring(n), "sup": _Ring(n)}
        self._ring_lock = threading.Lock()
        self._ring_n = n
        # Stream state (owner thread writes; readers read under _cv).
        self.stream = None
        self.device_open = None
        self._want = None             # (device,) a capture asked for
        self._want_seq = 0
        self._done_seq = 0
        self._last_error = None
        self._suspended = 0           # refcount of suspend() calls
        self._closing_for = ""
        self.n_cb = 0                 # samples seen by the callback
        self.n_written = 0            # samples written to the rings
        self._t_last_cb = None
        self.cb_dropped = 0
        self.opens = 0
        self.closes = 0
        self.closed_s = 0.0           # wanted open, but closed
        self._closed_since = None
        self._owner = None
        self._dsp = None
        self._threads_lock = threading.Lock()
        self._stop = False

    # ── threads ──────────────────────────────────────────────────────────
    def _start_threads(self) -> bool:
        with self._threads_lock:
            try:
                if self._owner is None or not self._owner.is_alive():
                    self._owner = threading.Thread(
                        target=self._owner_loop, name="mic-bus", daemon=True)
                    self._owner.start()
                if self._dsp is None or not self._dsp.is_alive():
                    self._dsp = threading.Thread(
                        target=self._dsp_loop, name="mic-bus-dsp", daemon=True)
                    self._dsp.start()
                return True
            except Exception as e:      # thread exhaustion: stay closed
                self._last_error = e
                return False

    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:
            pass

    # ── the callback (PortAudio thread): copy, index, enqueue ───────────
    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        try:
            mono = (indata[:, 0].copy() if getattr(indata, "ndim", 1) > 1
                    else np.array(indata, dtype=np.float32, copy=True))
            t = self._clock()
            with self._cb_cv:
                self.n_cb += len(mono)
                self._t_last_cb = t
                if len(self._cbq) >= CALLBACK_QUEUE_MAX:
                    self._cbq.popleft()
                    self.cb_dropped += 1
                self._cbq.append((self.n_cb, t, mono))
                self._cb_cv.notify()
        except Exception:
            pass

    # ── the DSP worker: canceller, rings, fan-out ────────────────────────
    def _dsp_loop(self) -> None:
        while not self._stop:
            with self._cb_cv:
                while not self._cbq:
                    if self._stop:
                        return
                    self._cb_cv.wait(1.0)
                n_end, t, mono = self._cbq.popleft()
            try:
                self._dsp_one(n_end, t, mono)
            except Exception:
                pass

    def _dsp_one(self, n_end: int, t: float, mono: np.ndarray) -> None:
        if (self._dsp_last_n is not None
                and n_end - len(mono) != self._dsp_last_n
                and self._on_reopen is not None):
            # Frames were dropped (a full callback queue) or the stream was
            # reopened: the canceller must re-anchor.
            try:
                self._on_reopen()
            except Exception:
                pass
        self._dsp_last_n = n_end
        lin = sup = mono
        aec = False
        proc = self.process
        if proc is not None:
            try:
                out = proc(mono, t)
            except Exception:
                out = None
            if out is not None:
                lin, sup = out
                aec = True
        with self._ring_lock:
            self._rings["raw"].write(n_end, mono)
            self._rings["lin"].write(n_end, lin)
            self._rings["sup"].write(n_end, sup)
            self.n_written = n_end
        fr = Frame(n_end, t, mono, lin, sup, aec)
        self._t_last_frame = t
        with self._subs_lock:
            subs = list(self._subs)
            listeners = list(self._listeners)
        for s in subs:
            s._offer(fr)
        for fn in listeners:
            try:
                fn(fr)
            except Exception:
                pass
        if self._tap_fanout is not None:
            try:
                if self._tap_allowed():
                    self._tap_fanout(fr)
            except Exception:
                pass

    # ── readers ──────────────────────────────────────────────────────────
    def subscribe(self, maxsize: int = SUBSCRIBER_QUEUE_MAX) -> Subscription:
        s = Subscription(self, maxsize)
        with self._subs_lock:
            self._subs.append(s)
        return s

    def add_listener(self, fn) -> None:
        """``fn(frame)`` on the DSP thread for every frame - it must only
        enqueue (the pre-gate's feed)."""
        with self._subs_lock:
            if fn not in self._listeners:
                self._listeners.append(fn)

    def last_frame_time(self) -> "float | None":
        """Arrival time of the newest frame in the rings (its last sample
        is index n_written - 1)."""
        return self._t_last_frame

    def _unsubscribe(self, s: Subscription) -> None:
        with self._subs_lock:
            try:
                self._subs.remove(s)
            except ValueError:
                pass

    def read(self, kind: str, start: int, stop: int) -> "np.ndarray | None":
        """Samples [start, stop) of ring ``kind`` ('raw' | 'lin' | 'sup'),
        or None when any of them is not in the ring."""
        try:
            start, stop = int(start), int(stop)
            with self._ring_lock:
                if (stop <= start or start < 0 or stop > self.n_written
                        or start < self.n_written - self._ring_n
                        or kind not in self._rings):
                    return None
                return self._rings[kind].read(start, stop)
        except Exception:
            return None

    def oldest(self) -> int:
        """The oldest sample index still in the rings."""
        with self._ring_lock:
            return max(0, self.n_written - self._ring_n)

    def last(self, kind: str, n: int) -> "np.ndarray | None":
        """The newest ``n`` samples of ring ``kind`` (None if fewer)."""
        with self._ring_lock:
            end = self.n_written
        return self.read(kind, end - int(n), end)

    # ── control (any thread) ─────────────────────────────────────────────
    def is_open(self) -> bool:
        with self._cv:
            return self.stream is not None

    def ensure(self, device, timeout: float = ENSURE_WAIT_S):
        """A capture wants the bus open on ``device``. Returns (ok, error):
        ok when the stream is open on it (opening it now if needed - on the
        owner thread, waited for, bounded), error = the exception of a
        failed open (the caller books it with the R10 back-off)."""
        if not self._start_threads():
            return False, self._last_error
        with self._cv:
            if (self.stream is not None and self.device_open == device
                    and not self._suspended):
                return True, None
            self._want = (device,)
            self._want_seq += 1
            seq = self._want_seq
            self._cv.notify_all()
            deadline = time.monotonic() + max(0.0, float(timeout))
            while self._done_seq < seq:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return False, BusUnavailable("the mic bus did not open "
                                                 "in time")
                self._cv.wait(rem)
            if self.stream is not None and self.device_open == device:
                return True, None
            return False, self._last_error

    def suspend(self, timeout: float = SUSPEND_WAIT_S) -> bool:
        """Close the stream for a PortAudio re-enumeration and keep it
        closed until resume(). True once it is closed (bounded wait). Every
        suspend() must be paired with one resume()."""
        with self._cv:
            self._suspended += 1
            self._cv.notify_all()
            deadline = time.monotonic() + max(0.0, float(timeout))
            while self.stream is not None:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return False
                self._cv.wait(rem)
            return True

    def resume(self) -> None:
        with self._cv:
            self._suspended = max(0, self._suspended - 1)
            self._cv.notify_all()

    def wake(self) -> None:
        """Re-check should_run now (a mute / mode flip)."""
        with self._cv:
            self._cv.notify_all()

    def status(self) -> dict:
        with self._cv:
            open_ = self.stream is not None
            dev = self.device_open
            sus = self._suspended
        with self._subs_lock:
            nsub = len(self._subs)
            sub_dropped = sum(s.dropped for s in self._subs)
        return {"open": open_, "device": dev, "suspended": bool(sus),
                "opens": self.opens, "closes": self.closes,
                "n": self.n_written, "callback_dropped": self.cb_dropped,
                "subscribers": nsub, "subscriber_dropped": sub_dropped,
                "closed_s": round(self.closed_s, 1)}

    def take_closed_s(self) -> float:
        """Seconds the bus was wanted but closed since the last call."""
        with self._cv:
            now = self._clock()
            v = self.closed_s
            if self._closed_since is not None:
                v += now - self._closed_since
                self._closed_since = now
            self.closed_s = 0.0
            return v

    # ── the owner thread: the ONLY one that opens / closes ───────────────
    def _close_locked_out(self, why: str) -> None:
        """Close the stream (called by the owner thread WITHOUT _cv held)."""
        with self._cv:
            st = self.stream
        if st is None:
            return
        try:
            self._close_stream(st)
        except Exception:
            pass
        finally:
            try:
                self._release()
            except Exception:
                pass
            with self._cv:
                self.stream = None
                self.device_open = None
                self.closes += 1
                if self._want is not None and why != "off":
                    self._closed_since = self._clock()
                self._cv.notify_all()

    def _open(self, device) -> None:
        err = None
        if not self._claim():
            err = BusUnavailable("the mic bus could not claim the device "
                                 "(a reinit in flight, or another capture "
                                 "holds it)")
        else:
            st = None
            try:
                st = self._open_stream(device, self._callback)
            except BaseException as e:      # noqa: BLE001 - booked, not raised
                err = e
                st = None
            if st is None:
                try:
                    self._release()
                except Exception:
                    pass
            else:
                with self._cb_cv:
                    self._t_last_cb = self._clock()
                with self._cv:
                    self.stream = st
                    self.device_open = device
                    self.opens += 1
                    if self._closed_since is not None:
                        self.closed_s += self._clock() - self._closed_since
                        self._closed_since = None
                if self._on_reopen is not None:
                    try:
                        self._on_reopen()
                    except Exception:
                        pass
        with self._cv:
            self._last_error = err
            self._done_seq = self._want_seq
            self._cv.notify_all()

    def shutdown(self) -> None:
        """Close the stream and let both threads end - for tests and process
        exit only (in a live JARVIS the bus never stops; it closes and
        reopens)."""
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        with self._cb_cv:
            self._cb_cv.notify_all()
        self._close_locked_out("off")

    def _owner_loop(self) -> None:
        while not self._stop:
            try:
                self._owner_step()
            except Exception:
                time.sleep(POLL_S)

    def _owner_step(self) -> None:
        with self._cv:
            self._cv.wait(POLL_S)
            want = self._want
            want_seq, done_seq = self._want_seq, self._done_seq
            st, dev = self.stream, self.device_open
            suspended = bool(self._suspended)
        try:
            run = bool(self._should_run())
        except Exception:
            run = False
        if st is not None:
            with self._cb_cv:
                last = self._t_last_cb
            dead = last is not None and self._clock() - last > DEAD_AFTER_S
            if not run or suspended or dead or (
                    want is not None and want[0] != dev
                    and want_seq != done_seq):
                why = ("off" if not run else "suspend" if suspended
                       else "dead" if dead else "device")
                if dead:
                    self._say("  [mic-bus] no audio from the microphone for "
                              f"{DEAD_AFTER_S:.0f} s - closing the stream; "
                              "the next capture reopens it")
                self._close_locked_out(why)
                if why == "dead":
                    with self._cv:
                        self._want = None
            elif want_seq != done_seq:
                with self._cv:          # already open on that device
                    self._done_seq = self._want_seq
                    self._cv.notify_all()
            return
        if want_seq == done_seq:
            return
        if not run or suspended:
            with self._cv:
                self._last_error = BusUnavailable(
                    "the mic bus is " + ("suspended for a device refresh"
                                         if suspended else "off (muted)"))
                self._done_seq = self._want_seq
                self._cv.notify_all()
            return
        self._open(want[0])
