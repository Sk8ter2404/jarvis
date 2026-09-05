"""Answer "is my headset on?" / "how much battery does my headset have?" aloud.

WHY THIS SKILL EXISTS
=====================
Same defect class as skills/audio_devices.py, one layer deeper. When a
capability is missing this local brain does NOT say "I can't" — it emits a
wrong-but-plausible action, which reads to the owner as the model being stupid.
The headset question had a worse version of that: an action DID exist, and it
answered CONFIDENTLY WRONG.

Before this skill, core/prompts.py routed the literal phrase "is my headset on"
to `audio_autoswitch_status`, whose answer is built on
`audio/audio_switch.py::find_active()`, which documents as fact:

    'Active' == powered on and present (a wireless headset that's OFF reads
    NotPresent/Unplugged).

That sentence is FALSE for this headset and had never been checked. With the
headset powered OFF, both CORSAIR VOID ELITE endpoints still report
Status=OK / state Active, because what Windows enumerates is the DONGLE, and
the dongle is plugged in either way. So "is my headset on" was answered by a
detector that structurally cannot tell — the exact bug class this project keeps
tripping over: something reporting success it never verified.

The real signal is the dongle's Corsair vendor HID interface, and that is what
`audio/void_link.py` reads. This skill is only the mouth: it turns void_link's
THREE-valued state into a finished spoken sentence.

WHAT I MEASURED MYSELF (2026-09-04, this agent, headset powered ON)
==================================================================
    audio.void_link.discover_device()
        -> ('\\\\?\\hid#vid_1b1c&pid_0a51&mi_03&col02#...', 20, 5)   0.003 s
    _raw_sample() -> 640058b101 -> ('on', 88)   0.002 s   (x3, identical)
    is_headset_on() -> True, battery_percent() -> 88       0.005 s

    audio_switch.list_render(), same moment, headset ON:
        Active | {0.0.0.0000 | Headset Earphone (CORSAIR VOID ELITE ...)
        Active | {0.0.1.0000 | Headset Microphone (CORSAIR VOID ELITE ...)
        find_active('VOID ELITE') -> matched the earphone endpoint

  HONEST LIMIT ON THAT: the headset was ON for my whole session, so I measured
  only the ON case. The claim that the endpoints ALSO read Active while the
  headset is OFF — the claim that condemns find_active() as a detector — is
  inherited from the 2026-09-04 capture, not re-measured by me. I did not power
  the headset off, so I cannot call that one first-hand.

  ONE NEW DATA POINT, which contradicts a note in void_link.py's history: that
  module records every observed ON-state battery byte as having bit 0x80 set
  (0xe0, 0xde, 0xdc, 0xda). Mine was 0x58 — high bit CLEAR — while the link was
  up, and 0x58 & 0x7F = 88 continues the drain curve 96 -> 94 -> 92 -> 90 -> 88
  exactly. So the high bit is NOT always set on a live reading. What it means is
  still unknown; `& 0x7F` reads correctly either way, so nothing here changes.
  Recorded so the next reader does not build on "it is always set".

THE THREE-VALUED STATE IS THE WHOLE POINT
=========================================
void_link answers on / off / UNKNOWN, and UNKNOWN is a real answer, not a
polite off. The dongle went completely silent for 105 CONSECUTIVE SECONDS
during a pairing handshake while the owner was turning the headset ON. A skill
that collapsed unknown into "off" would have told him his headset was off while
he was switching it on — which is the same lie as the old endpoint detector,
just from a better sensor. So:

    on      -> "Your headset is on, sir - battery is at 88 percent."
    off     -> "Your headset is off, sir."
    unknown -> "I can't tell whether your headset is on, sir - ..."

and "on but I couldn't read the battery" is its own sentence too, because a
powered-down headset reads 0x00 in the battery byte and speaking that as
"0 percent" would be a fabricated number.

DESIGN NOTES
============
* void_link is imported LAZILY, exactly like audio_devices.py's `_bc()`. If the
  import fails (a non-Windows CI runner, a missing module, a syntax error
  someone lands upstream), this skill still LOADS and still REGISTERS — it just
  answers "I can't tell". A skill that failed to import would deregister all
  seven action names, and the model would go straight back to emitting a
  wrong-but-plausible neighbour. Degrading to an honest "unknown" is the whole
  point; degrading to no action at all recreates the original bug.

* No fallback to the Windows endpoint state, ever. It is available (the
  monolith has helpers, audio_switch.find_active() is one import away) and it
  would make "unknown" rarer — by answering with the sensor that is already
  known to be wrong. An honest unknown beats a confident guess.

* No `probe_once()` battery fallback either. It was considered: when the link
  is up but `battery_percent()` is None, a fresh raw sample might carry a
  number. Rejected — `probe_once()` is explicitly undebounced and documented as
  able to lie mid-handshake, and "on, battery unknown" is a perfectly good
  sentence. Fewer claims.

* THE ANSWER IS MEASURED DURING THE CALL, NOT REMEMBERED (fixed 2026-09-04;
  this is a correction to the first version of this skill, which had the bug
  described below and shipped with it). The obvious API — `is_headset_on()` —
  reads void_link's PROCESS-WIDE DEBOUNCED BELIEF, and that belief HOLDS
  across an unknown sample by design: silence must not fire a spurious device
  switch in audio_switch's poller. It also never returns to UNKNOWN once it
  has settled (only `VoidLink.reset()` does that, and nothing in the tree
  calls it). So in a warm JARVIS process `is_headset_on()` can never again
  answer None, the UNKNOWN sentence below becomes unreachable, and this skill
  speaks a belief of unbounded age as present-tense fact — with a battery
  number attached.

  DEMONSTRATED, not theorised: warm the shared link on the real dongle, then
  make every sample silent (which is exactly what an unplugged dongle or an
  iCUE grab of the vendor interface does — both named in void_link's own
  docstring). Six consecutive calls answered "Your headset is on, sir -
  battery is at 92 percent." while `void_link.probe_once()` at that same
  instant returned ('unknown', None). The mirror case is worse and is IN the
  measured capture: the belief settles OFF, then the dongle goes silent for
  105 s while the owner is powering the headset ON — and the old code answered
  a flat "Your headset is off, sir." at the exact moment the sensor was saying
  nothing.

  So `_read()` now builds its OWN `VoidLink` per call. A fresh instance starts
  at UNKNOWN and has no memory to hold, so silence stays UNKNOWN instead of
  resurrecting a stale belief — while the debounce that instance carries is
  void_link's own, still requiring N consecutive agreeing samples before a
  state is believed (hard-won behaviour 2: at 20:21:34 a single sample showed
  a live battery byte with the link still down). The debounce logic is REUSED,
  not re-implemented here: a second copy of that rule is precisely this
  repo's #1 bug class, the stale duplicate. The per-call instance is also why
  this costs nothing elsewhere — it never touches void_link's shared link, so
  audio_switch's poller keeps its own smoothed view.

* The hold rule and this skill do not disagree; they answer different
  questions. audio_switch asks "should I SWITCH the default device now?", for
  which holding through silence is right — a switch is an action with a cost.
  This skill answers "is it on RIGHT NOW?", asked once, out loud, and for that
  question silence has an honest answer already written below.

All five replies are finished, user-facing sentences, so every name is declared
in SPEAK_VERBATIM_ACTIONS below. Without that declaration the answer is
computed, logged and silently dropped — the same defect wearing a different hat.
"""
from __future__ import annotations

import importlib
import sys
import time
from typing import Optional, Tuple

# Names whose return value is already a finished sentence and must be spoken
# verbatim. bobert_companion._collect_skill_speak_sets folds this into
# SPEAK_RESULT_VERBATIM_ACTIONS at load time.
SPEAK_VERBATIM_ACTIONS = (
    "headset_status", "is_headset_on", "headset_on", "is_my_headset_on",
    "headset_battery", "how_much_battery_headset", "headset_battery_level",
)

_LINK_MODULE = "audio.void_link"

# The three-valued vocabulary, mirrored locally so this module has an answer
# even when void_link could not be imported at all (there is then nothing to
# read LINK_ON/LINK_OFF/LINK_UNKNOWN off).
_ON = "on"
_OFF = "off"
_UNKNOWN = "unknown"

# Below this, the reply nudges him to charge it.
#
# JUDGEMENT CALL, NOT A MEASUREMENT. 15 is not a property of the hardware; it
# is copied from audio/audio_switch.py's AudioAutoSwitch.low_pct ("warn once
# when battery drops below this") so this skill and the auto-switch daemon
# agree on what "low" means instead of inventing a second threshold. The
# percentage itself is always spoken, so a wrong threshold only changes the
# nudge, never the number.
LOW_BATTERY_PCT = 15

# How long this skill will spend MEASURING before it gives up and says it
# cannot tell, and how many samples it will take inside that window.
#
# The fast half is measured (2026-09-04, real dongle, headset ON): one C9 64
# round trip is 2-3 ms, so settling a two-sample debounce costs ~6 ms and the
# cap below is never approached.
#
# The slow half is arithmetic on void_link's own published timeouts, NOT a
# measurement: a silent sample costs READ_TIMEOUT_MS (400) plus, when the
# cached path failed fast, one re-discovery and a second read — so roughly
# 0.85 s in the worst case. 0.75 s is deliberately BELOW that, so a slow
# silence is asked exactly once. Re-asking a device that is not answering only
# doubles the wait for the same answer; void_link makes the identical argument
# for its own RETRY_BUDGET_S. Worst case for a whole call is therefore about
# one silent sample, ~0.9 s.
#
# HONEST LIMIT: how long a barely-responsive dongle might take was never
# measured, here or in void_link. If one exists, it reads as UNKNOWN — which
# is the conservative direction, not a wrong assertion.
_FRESH_BUDGET_S = 0.75
_MAX_SAMPLES = 4


def _link():
    """The audio.void_link module, or None.

    Lazy and failure-tolerant on purpose — see the module docstring. Prefers an
    already-imported copy so a test (or another component) that installed a
    stand-in in sys.modules is honoured rather than bypassed."""
    mod = sys.modules.get(_LINK_MODULE)
    if mod is not None:
        return mod
    try:
        return importlib.import_module(_LINK_MODULE)
    except Exception:
        return None


def _clean_pct(value) -> Optional[int]:
    """A battery percentage we are willing to say out loud, or None.

    Accepts 1-100 only. Rejecting 0 is deliberate: void_link reports None for a
    zero battery byte precisely because a powered-down headset reads 0x00 there
    and "0 percent" would be a fabricated reading. Anything outside the range,
    or not a number at all, is not a percentage this skill can vouch for."""
    try:
        if isinstance(value, bool) or value is None:
            return None
        pct = int(value)
    except Exception:
        return None
    return pct if 1 <= pct <= 100 else None


def _measure(vl) -> Optional[Tuple[str, Optional[int]]]:
    """(state, battery) from samples taken DURING THIS CALL, or None if this
    void_link offers no way to measure freshly.

    Builds a PRIVATE `VoidLink`, which is the whole fix: a fresh instance
    starts at UNKNOWN and carries no belief to hold, so a silent dongle reads
    as UNKNOWN instead of resurrecting whatever the shared link last decided —
    while the debounce inside that instance is void_link's own, so a single
    mid-handshake sample still cannot flip the answer.

    None (the "cannot measure" return) is NOT the same as ('unknown', None). A
    module that exposes VoidLink but whose state() raises has been measured and
    the measurement failed — that is an unknown, and falling back to the held
    belief there would walk straight back into the defect."""
    cls = getattr(vl, "VoidLink", None)
    if not callable(cls):
        return None
    try:
        debounce = int(getattr(vl, "DEFAULT_DEBOUNCE", 2))
    except Exception:
        debounce = 2
    try:
        link = cls(debounce=debounce)
    except Exception:
        return None
    sample = getattr(link, "state", None)
    if not callable(sample):
        return None

    # Read the vocabulary off the module rather than assuming our local copy
    # matches it; the local _ON/_OFF exist for the module-missing case.
    on = getattr(vl, "LINK_ON", _ON)
    off = getattr(vl, "LINK_OFF", _OFF)

    deadline = time.monotonic() + _FRESH_BUDGET_S
    for attempt in range(_MAX_SAMPLES):
        # Always take the first sample; only the extra ones are budgeted.
        if attempt and time.monotonic() >= deadline:
            break
        try:
            state, battery = sample()
        except Exception:
            return _UNKNOWN, None
        if state == on:
            return _ON, battery
        if state == off:
            return _OFF, None
        # Anything else is the instance still unsettled: either the samples
        # disagree (a lie is in flight) or nothing is answering. Try again
        # while there is budget; if we run out, that IS the answer.
    return _UNKNOWN, None


def _read() -> Tuple[str, Optional[int]]:
    """(state, battery_pct) with state in on/off/unknown. NEVER raises.

    Every failure mode — module missing, function missing, function raising, a
    three-valued None — lands on UNKNOWN, and UNKNOWN is never rewritten as OFF
    inside this function.

    THAT SENTENCE USED TO BE A LIE AT THE SYSTEM LEVEL, which is why the
    measurement below exists. It was true of this function and false of the
    answer the owner heard: the old code asked `is_headset_on()`, whose UNKNOWN
    had ALREADY been rewritten as True or False by void_link's process-wide
    hold before it ever arrived here. See the module docstring."""
    vl = _link()
    if vl is None:
        return _UNKNOWN, None

    fresh = _measure(vl)
    if fresh is not None:
        state, battery = fresh
        if state != _ON:
            return state, None
        return _ON, _clean_pct(battery)

    # No VoidLink to measure with. Fall back to the required contract.
    #
    # SAY WHAT THIS IS: against the real module this branch is dead (it has
    # exposed VoidLink since it was written), and it carries NO freshness
    # guarantee — if some future void_link drops the class while keeping a
    # holding is_headset_on(), the staleness comes back. It is kept because a
    # skill that answers nothing is worse than one that answers late: losing
    # these seven action names sends the model back to emitting a
    # wrong-but-plausible neighbour, which is the bug this whole skill exists
    # to kill. tests/skills/test_headset_status.py pins the real module against
    # this by asserting VoidLink is still there.
    is_on = getattr(vl, "is_headset_on", None)
    if not callable(is_on):
        return _UNKNOWN, None
    try:
        state = is_on()
    except Exception:
        return _UNKNOWN, None

    # The contract is strictly True / False / None. Anything else is a broken
    # detector, and a broken detector is an unknown, never an "off".
    if state is False:
        return _OFF, None
    if state is not True:      # None (genuinely unknown), or a contract breach
        return _UNKNOWN, None

    # Link is up. Battery is a separate, optional read: `battery_percent` is a
    # convenience void_link offers on top of the required contract, so it is
    # fetched by getattr and its absence costs a number, not the answer.
    battery = None
    getter = getattr(vl, "battery_percent", None)
    if callable(getter):
        try:
            battery = getter()
        except Exception:
            battery = None
    return _ON, _clean_pct(battery)


def _battery_phrase(pct: int) -> str:
    """'battery is at 88 percent' / the low-battery variant."""
    if pct <= LOW_BATTERY_PCT:
        return f"battery is down to {pct} percent, so it's worth charging"
    return f"battery is at {pct} percent"


def headset_status(_arg: str = "") -> str:
    """'is my headset on' — the question that used to be answered by a sensor
    that cannot tell."""
    state, pct = _read()
    if state == _ON:
        if pct is None:
            return ("Your headset is on, sir, though I couldn't read its "
                    "battery level.")
        return f"Your headset is on, sir - {_battery_phrase(pct)}."
    if state == _OFF:
        return "Your headset is off, sir."
    # UNKNOWN. Not "off". Not a guess dressed up as an answer.
    #
    # Deliberately does NOT name a cause. "the dongle isn't answering" was the
    # first wording and it is a claim this function cannot support: unknown is
    # also reached when void_link could not be imported at all, when the
    # function is missing, and when it raised — in none of which has anything
    # been established about the dongle. "I'm not getting a reading" is true in
    # every one of those branches.
    return ("I can't tell whether your headset is on, sir - I'm not getting a "
            "reading from it.")


def headset_battery(_arg: str = "") -> str:
    """'how much battery does my headset have'."""
    state, pct = _read()
    if state == _ON:
        if pct is None:
            return ("Your headset is on, sir, but I couldn't read its battery "
                    "level.")
        if pct <= LOW_BATTERY_PCT:
            return (f"Your headset is down to {pct} percent, sir - worth "
                    f"charging it.")
        return f"Your headset is at {pct} percent, sir."
    if state == _OFF:
        # The battery byte reads 0x00 while the headset is powered down, which
        # is an absence of a reading and not a reading of zero.
        return ("Your headset is off, sir - I can't read its battery while "
                "it's powered down.")
    return ("I can't tell whether your headset is on, sir, so I can't give you "
            "a battery level.")


def register(actions):
    actions["headset_status"] = headset_status
    actions["is_headset_on"] = headset_status
    actions["headset_on"] = headset_status
    actions["is_my_headset_on"] = headset_status
    actions["headset_battery"] = headset_battery
    actions["how_much_battery_headset"] = headset_battery
    actions["headset_battery_level"] = headset_battery
    # ASCII only: the skill loader prints this to a cp1252 console.
    print("  [headset_status] ready - actions: headset_status, is_headset_on, "
          "headset_battery, how_much_battery_headset.")
