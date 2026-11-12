"""alarm skill - a wake-up alarm that rings until someone answers it.

Live 2026-11-12 11:32-11:34 the owner asked four times for "an actual alarm
that will wake me up" and got set_timer each time: one spoken line at speech
volume, then silence. A timer tells; an alarm rings.

  set_alarm, <clock time | duration> [| label]   "wake me at 7", "alarm in 20
                                                 minutes"
  list_alarms                                    what is set
  cancel_alarm, [id|all]                         drop a pending alarm
  stop_alarm                                     silence the one ringing
  snooze_alarm, [minutes]                        ring again later (9 min)

When an alarm is due it:
  * moves the default output to the desk speakers when the audio autoswitch
    skill offers that (a headset on the desk wakes nobody),
  * unmutes and raises the master volume to ALARM_VOLUME (full; never lowers it),
  * loops a tone through Windows' own player (winsound), not PortAudio, so a
    device re-enumeration or a busy TTS stream cannot stop or crash it,
  * says a wake-up line over the tone every ALARM_SAY_EVERY_S ("Sir, you
    need to wake up."), naming what the alarm is for when he gave a label,
  * rings until "stop" / "I'm up" / "snooze" is heard, a key or the mouse is
    touched, or ALARM_MAX_RING_S passes, then puts the volume back.

Pending alarms are saved (data/alarms.json) and re-armed at boot; one that
came due while JARVIS was down rings at once if it is under
ALARM_LATE_RING_S late, otherwise it is reported as missed. It cannot ring
while the PC or JARVIS is off: it is not a substitute for a phone alarm on a
morning that matters.

Everything that touches the machine goes through the module-level hooks
(_sound_start, _sound_stop, _volume_read, _volume_write, _input_tick,
_to_speakers) so tests replace them and nothing real is reached.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
import threading
import time
import wave
from datetime import datetime

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:  # pragma: no cover - import-time guard
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

# Every reply is a finished sentence about the alarm store: voiced as written.
SPEAK_VERBATIM_ACTIONS = (
    "set_alarm", "list_alarms", "cancel_alarm", "stop_alarm", "snooze_alarm",
)

ALARM_VOLUME = 1.0           # master volume while ringing: he is a heavy sleeper
ALARM_MAX_RING_S = 600.0     # give up after ten minutes with no answer
ALARM_SNOOZE_MIN = 9         # default snooze
ALARM_INPUT_GRACE_S = 3.0    # input in the first seconds does not dismiss
ALARM_LATE_RING_S = 2 * 3600.0
ALARM_SAY_FIRST_S = 2.0      # first spoken line, seconds into the ring
ALARM_SAY_EVERY_S = 18.0     # then one every this many seconds
# What he says over the tone, in order, then round again from the second.
# None of them may read as an answer to the alarm (ringing_command): the mic
# hears the speakers, and a line with "stop" in it could end its own alarm.
WAKE_LINES = (
    "Sir, it is {clock}. Time to wake up.",
    "Sir, you need to wake up.",
    "I must insist, sir. You asked me to wake you.",
    "Sir. Wake up.",
    "On your feet, sir. The day is not going to wait.",
    "Still asleep, sir? This has been ringing for {minutes}.",
    "Sir, I can keep this up far longer than you can sleep through it.",
)
_ALARMS_FILE_NAME = "alarms.json"
_TONE_FILE_NAME = "alarm_tone.wav"

_lock = threading.Lock()
_persist_lock = threading.Lock()
# id -> {"at": epoch seconds, "label": str, "timer": threading.Timer}
_alarms: dict = {}
_next_id = [1]
# The alarm ringing now: {"id", "label", "stop": Event, "why": str} or None.
_ringing: list = [None]
_actions_ref: list = [None]


# ── machine hooks (tests replace these) ─────────────────────────────────
def _data_file(name: str) -> str:
    from core import paths as _paths
    return _paths.data_file(name)


def _sound_start(path: str) -> None:
    import winsound
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC
                       | winsound.SND_LOOP | winsound.SND_NODEFAULT)


def _sound_stop() -> None:
    # A None sound stops whatever this process is playing. SND_PURGE is not
    # honoured on current Windows, so it is not what this relies on.
    import winsound
    winsound.PlaySound(None, 0)


def _endpoint():
    from core import actions as _ca
    return _ca._endpoint_volume()


def _volume_read():
    """(level 0..1, muted) of the default output, or None when unreadable."""
    try:
        vol = _endpoint()
        return (float(vol.GetMasterVolumeLevelScalar()), bool(vol.GetMute()))
    except Exception:
        return None


def _volume_write(level, muted) -> bool:
    try:
        vol = _endpoint()
        vol.SetMute(1 if muted else 0, None)
        vol.SetMasterVolumeLevelScalar(max(0.0, min(1.0, float(level))), None)
        return True
    except Exception:
        return False


def _input_tick():
    """Windows' last-input tick (keyboard or mouse), or None."""
    try:
        import ctypes

        class _LII(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        lii = _LII()
        lii.cbSize = ctypes.sizeof(_LII)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return int(lii.dwTime)
    except Exception:
        pass
    return None


def _to_speakers() -> None:
    """Ask the audio autoswitch skill for the desk speakers, if it is loaded."""
    try:
        fn = (_actions_ref[0] or {}).get("switch_to_speakers")
        if callable(fn):
            fn("")
    except Exception as e:
        print(f"  [alarm] could not move audio to the speakers ({e})")


def _say(message: str, key: str | None = None) -> None:
    try:
        from skills import timer as _timer
        _timer._enqueue_speech(message, key=key)
    except Exception as e:
        print(f"  [alarm] could not queue speech ({e}): {message}")


# ── the tone ─────────────────────────────────────────────────────────────
def _tone_path() -> str:
    """The looped alarm tone, written once: four rising beeps and a gap, so
    a spoken "stop" has quiet to land in."""
    path = _data_file(_TONE_FILE_NAME)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    rate = 22050
    frames = bytearray()

    def beep(freq: float, secs: float) -> None:
        n = int(rate * secs)
        for i in range(n):
            # 8 ms fade at both ends: no click, still sharp.
            edge = min(1.0, i / (rate * 0.008), (n - i) / (rate * 0.008))
            v = 0.9 * edge * math.sin(2.0 * math.pi * freq * i / rate)
            frames.extend(struct.pack("<h", int(v * 32767)))

    def gap(secs: float) -> None:
        frames.extend(b"\x00\x00" * int(rate * secs))

    for freq in (1900.0, 2400.0, 1900.0, 2400.0):
        beep(freq, 0.16)
        gap(0.09)
    gap(1.1)
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    os.replace(tmp, path)
    return path


# ── parsing ──────────────────────────────────────────────────────────────
def _parse_when(text: str):
    """Seconds from now for 'at 7', '7:30 am', 'in 20 minutes', '6 tomorrow'.
    None when no time is named."""
    from skills import timer as _timer
    t = (text or "").strip().lower()
    if not t:
        return None
    secs = _timer._parse_clock_time(t)
    if secs is not None:
        return secs
    # "at 7" / "for 7 o'clock" with no am/pm: the next time the clock reads it.
    m = re.search(r"\b(?:at|for)\s+(\d{1,2})(?:\s*o'?clock)?\b(?!\s*(?:sec|min|hour|hr|day))", t)
    if m and 1 <= int(m.group(1)) <= 12:
        hour = int(m.group(1)) % 12
        now = datetime.fromtimestamp(time.time())
        best = None
        for h in (hour, hour + 12):
            target = now.replace(hour=h, minute=0, second=0, microsecond=0)
            delta = (target - now).total_seconds()
            if delta <= 0:
                delta += 86400.0
            best = delta if best is None else min(best, delta)
        return int(best)
    secs = _timer._parse_duration(t)
    if secs is not None and secs > 0:
        return int(secs)
    return None


def _split_args(args: str):
    raw = (args or "").strip()
    when, _, label = raw.partition("|")
    return _parse_when(when if label else raw), label.strip()


def _clock(epoch: float) -> str:
    d = datetime.fromtimestamp(epoch)
    hour = d.hour % 12 or 12
    return f"{hour}:{d.minute:02d} {'AM' if d.hour < 12 else 'PM'}"


def _span(secs: float) -> str:
    secs = int(max(0, secs))
    if secs < 90:
        return f"{secs} seconds"
    mins = round(secs / 60)
    if mins < 90:
        return f"{mins} minutes"
    hours, rem = divmod(mins, 60)
    return f"{hours} hours" + (f" {rem} minutes" if rem else "")


# ── persistence ──────────────────────────────────────────────────────────
def _snapshot() -> list:
    with _lock:
        return [{"id": aid, "at": a["at"], "label": a["label"]}
                for aid, a in sorted(_alarms.items())]


def _persist() -> None:
    try:
        with _persist_lock:
            _atomic_write_json(_data_file(_ALARMS_FILE_NAME), _snapshot())
    except Exception as e:
        print(f"  [alarm] could not save alarms ({e})")


def restore_persisted_alarms() -> int:
    """Re-arm the alarms the previous process saved. Returns how many."""
    try:
        with open(_data_file(_ALARMS_FILE_NAME), "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"  [alarm] could not read saved alarms ({e})")
        return 0
    n = 0
    now = time.time()
    for row in data if isinstance(data, list) else []:
        try:
            at = float(row["at"])
            label = str(row.get("label") or "")
        except Exception:
            continue
        late = now - at
        if late > ALARM_LATE_RING_S:
            print(f"  [alarm] missed while I was off: {_clock(at)} {label}".rstrip())
            _say(f"Sir, I was not running at {_clock(at)}, so that alarm "
                 "never rang.", key=f"alarm-missed#{int(at)}")
            continue
        _arm(max(1.0, at - now), label, at=at)
        n += 1
    _persist()
    return n


# ── arming and ringing ───────────────────────────────────────────────────
def _arm(secs: float, label: str, at: float | None = None) -> int:
    with _lock:
        aid = _next_id[0]
        _next_id[0] += 1
    fire_at = float(at) if at is not None else time.time() + float(secs)
    t = threading.Timer(max(0.0, float(secs)), _fire, args=(aid,))
    t.daemon = True
    with _lock:
        _alarms[aid] = {"at": fire_at, "label": label, "timer": t}
    t.start()
    return aid


def _fire(aid: int) -> None:
    with _lock:
        a = _alarms.pop(aid, None)
    if a is None:
        return
    _persist()
    ring(aid, a["label"])


def is_ringing() -> bool:
    return _ringing[0] is not None


def ring(aid: int, label: str = "") -> None:
    """Ring until answered. Blocks its own thread; never raises."""
    stop = threading.Event()
    me = {"id": aid, "label": label, "stop": stop, "why": ""}
    with _lock:
        if _ringing[0] is not None:
            # One alarm at a time: the second one's label joins the first.
            print(f"  [alarm] #{aid} came due while #{_ringing[0]['id']} rings")
            return
        _ringing[0] = me
    print(f"  [alarm] ⏰ ringing #{aid} {label}".rstrip())
    before = None
    try:
        _to_speakers()
        before = _volume_read()
        if before is not None:
            _volume_write(max(before[0], ALARM_VOLUME), False)
        _sound_start(_tone_path())
        started = time.monotonic()
        tick0 = None
        said = 0
        say_at = ALARM_SAY_FIRST_S
        while not stop.wait(0.25):
            age = time.monotonic() - started
            if age >= ALARM_MAX_RING_S:
                me["why"] = "timeout"
                break
            if age >= say_at:
                _say(wake_line(said, age, label), key=f"alarm-wake#{aid}.{said}")
                said += 1
                say_at = age + ALARM_SAY_EVERY_S
            tick = _input_tick()
            if age < ALARM_INPUT_GRACE_S or tick0 is None:
                tick0 = tick
            elif tick is not None and tick != tick0:
                me["why"] = "input"
                break
    except Exception as e:
        me["why"] = me["why"] or f"error: {type(e).__name__}: {e}"
        print(f"  [alarm] ring failed ({e})")
    finally:
        try:
            _sound_stop()
        except Exception as e:
            print(f"  [alarm] could not stop the tone ({e})")
        if before is not None:
            _volume_write(before[0], before[1])
        with _lock:
            _ringing[0] = None
    why = me["why"] or "voice"
    print(f"  [alarm] #{aid} stopped ({why})")
    if why == "timeout":
        _say(f"Sir, your alarm rang for {int(ALARM_MAX_RING_S // 60)} minutes "
             "with no answer, so I have stopped it.", key=f"alarm-timeout#{aid}")
    elif why == "input":
        _say("Alarm off, sir.", key=f"alarm-off#{aid}")
    elif why.startswith("error"):
        _say("Sir, your alarm is due, and I could not ring it.",
             key=f"alarm-error#{aid}")


def wake_line(n: int, age_s: float, label: str = "", now: float | None = None) -> str:
    """The n-th line (0-based) spoken over the tone, ``age_s`` into the ring."""
    if n < len(WAKE_LINES):
        line = WAKE_LINES[n]
    else:
        line = WAKE_LINES[1 + (n - 1) % (len(WAKE_LINES) - 1)]
    mins = max(1, int(round(age_s / 60.0)))
    line = line.format(clock=_clock(time.time() if now is None else now),
                       minutes="a minute" if mins == 1 else f"{mins} minutes")
    if n == 0 and label:
        line += f" You set this for {label}."
    return line


def _stop_ringing(why: str) -> bool:
    with _lock:
        me = _ringing[0]
        if me is None:
            return False
        me["why"] = why
    me["stop"].set()
    return True


# ── what he says while it rings ──────────────────────────────────────────
_SNOOZE_RE = re.compile(r"\bsnooze\b|\b(?:five|ten|a few) more minutes\b")
_STOP_RE = re.compile(
    r"\b(?:stop|off|quiet|enough|shut up|silence|cancel|dismiss)\b"
    r"|\bi(?:'m| am) (?:up|awake)\b|\bokay,? okay\b")


def ringing_command(text: str) -> str:
    """'stop' / 'snooze' / '' for a line heard while an alarm rings."""
    if not is_ringing():
        return ""
    t = re.sub(r"[^a-z' ]+", " ", (text or "").lower())
    if _SNOOZE_RE.search(t):
        return "snooze"
    if _STOP_RE.search(t):
        return "stop"
    return ""


def handle_ringing(text: str):
    """Act on a line heard while ringing. The spoken reply, or None when the
    line is not about the alarm (or nothing rings)."""
    cmd = ringing_command(text)
    if cmd == "snooze":
        return snooze("")
    if cmd == "stop":
        return stop_alarm("")
    return None


# ── actions ──────────────────────────────────────────────────────────────
def set_alarm(args: str = "") -> str:
    secs, label = _split_args(args)
    if secs is None or secs <= 0:
        return ("I couldn't tell when, sir. Try 'wake me at 7 AM' or "
                "'alarm in 20 minutes'.")
    aid = _arm(secs, label)
    _persist()
    at = time.time() + secs
    tail = f" for {label}" if label else ""
    return (f"Alarm #{aid} set for {_clock(at)}{tail}, {_span(secs)} from "
            "now, sir. It rings until you say stop or touch the keyboard.")


def list_alarms(_: str = "") -> str:
    rows = _snapshot()
    if not rows:
        return "No alarms are set, sir."
    parts = [f"#{r['id']} at {_clock(r['at'])}"
             + (f" for {r['label']}" if r["label"] else "") for r in rows]
    return ("One alarm, sir: " if len(parts) == 1
            else f"{len(parts)} alarms, sir: ") + ", ".join(parts) + "."


def cancel_alarm(args: str = "") -> str:
    low = (args or "").strip().lower()
    with _lock:
        if not _alarms:
            ids = []
        elif low in ("all", "everything", "them all", "all alarms"):
            ids = list(_alarms)
        else:
            m = re.search(r"\d+", low)
            if m and int(m.group(0)) in _alarms:
                ids = [int(m.group(0))]
            elif m:
                return f"There is no alarm #{m.group(0)}, sir."
            else:
                ids = [min(_alarms, key=lambda k: _alarms[k]["at"])]
        gone = [(i, _alarms.pop(i)) for i in ids]
    for _i, a in gone:
        a["timer"].cancel()
    if _stop_ringing("cancelled") and not gone:
        return "Alarm stopped, sir."
    if not gone:
        return "No alarms are set, sir."
    _persist()
    if len(gone) == 1:
        return f"Cancelled the {_clock(gone[0][1]['at'])} alarm, sir."
    return f"Cancelled {len(gone)} alarms, sir."


def stop_alarm(_: str = "") -> str:
    if _stop_ringing("voice"):
        return "Alarm off, sir."
    return "No alarm is ringing, sir."


def snooze(args: str = "") -> str:
    m = re.search(r"\d+", args or "")
    mins = int(m.group(0)) if m else ALARM_SNOOZE_MIN
    mins = max(1, min(60, mins))
    with _lock:
        me = _ringing[0]
        label = me["label"] if me else ""
    if not _stop_ringing("snooze"):
        return "No alarm is ringing, sir."
    _arm(mins * 60.0, label)
    _persist()
    return f"Snoozed, sir. It rings again in {mins} minutes."


def register(actions):
    _actions_ref[0] = actions
    actions["set_alarm"] = set_alarm
    actions["list_alarms"] = list_alarms
    actions["cancel_alarm"] = cancel_alarm
    actions["stop_alarm"] = stop_alarm
    actions["snooze_alarm"] = snooze
    try:
        n = restore_persisted_alarms()
        if n:
            print(f"  [alarm] re-armed {n} saved alarm(s)")
    except Exception as e:
        print(f"  [alarm] restore failed ({e})")
