"""Tests for audio.void_link — "is the CORSAIR VOID ELITE headset powered ON?"

EVERY FIXTURE IN THIS FILE IS A REAL CAPTURED BYTE STRING, not an invention.
They come from the 2026-09-04 poller run (one sample/second, changes only)
recorded in the module docstring of audio/void_link.py:

    20:15:27  6400003300   headset OFF
    20:15:29  6400003400   headset OFF   (byte[3] flips 33<->34 = noise)
    20:19:41  None         NO REPLY AT ALL - sustained 105 seconds
    20:21:26  6400002600   byte[4]=00
    20:21:31  6400003300   byte[4]=00
    20:21:34  6400e0b100   battery byte populated, byte[4] STILL 00
    20:21:37  6400dea101   byte[4]=01  <-- LINK UP
    20:21:38  6400deb101   byte[4]=01  (byte[3] a1<->b1 = noise)
    20:21:40  6400dcb101   byte[4]=01
    20:21:55  6400dca101   byte[4]=01
    20:21:56  6400dcb101   byte[4]=01  (stable ON ever since)

The raw HID layer (`void_link._raw_sample`) is mocked in every test, so nothing
here touches hid.dll, the dongle, or the running JARVIS process. The module
must also IMPORT on the ubuntu CI runner where hid.dll does not exist — that is
verified by the plain import at the top of this file plus
`test_module_imports_without_hid`.

The three behaviours these tests exist to lock down, all measured rather than
assumed:

  1. NO REPLY IS NOT "OFF". 105 seconds of silence happened while the owner was
     turning the headset ON. `test_hundred_consecutive_no_replies_hold_state`
     replays that as 100 samples and asserts the state does not budge.
  2. A SINGLE SAMPLE CAN LIE (6400e0b100 at 20:21:34 — battery present, link
     still down), so changes are debounced.
  3. A powered-down headset reads 0x00 in the battery byte and must be reported
     as None, NEVER as "0%".

stdlib unittest + mock, matching the other audio/ tests in this directory.
"""
from __future__ import annotations

import unittest
from unittest import mock

from audio import void_link as vl


# ─────────────────────────────────────────────────────────────────────────
# REAL captured replies (2026-09-04). Names carry the timestamp they came
# from so a future reader can find them in the capture.
# ─────────────────────────────────────────────────────────────────────────
OFF_2015_27 = bytes.fromhex("6400003300")   # headset OFF
OFF_2015_29 = bytes.fromhex("6400003400")   # headset OFF, byte[3] flipped
NO_REPLY = None                             # 20:19:41, sustained 105 s
HS_2021_26 = bytes.fromhex("6400002600")    # mid-handshake, byte[4]=00
HS_2021_31 = bytes.fromhex("6400003300")    # mid-handshake, byte[4]=00
TRANSIENT_2021_34 = bytes.fromhex("6400e0b100")  # battery present, link DOWN
ON_2021_37 = bytes.fromhex("6400dea101")    # LINK UP, battery 94
ON_2021_38 = bytes.fromhex("6400deb101")    # LINK UP, battery 94
ON_2021_40 = bytes.fromhex("6400dcb101")    # LINK UP, battery 92
ON_2021_55 = bytes.fromhex("6400dca101")    # LINK UP, battery 92
ON_2021_56 = bytes.fromhex("6400dcb101")    # LINK UP, battery 92

# The full transition, in capture order.
CAPTURE_TIMELINE = [
    OFF_2015_27, OFF_2015_29, OFF_2015_27, OFF_2015_29,
    NO_REPLY,
    HS_2021_26, HS_2021_31, TRANSIENT_2021_34,
    ON_2021_37, ON_2021_38, ON_2021_40, ON_2021_55, ON_2021_56,
]


def _feed(samples):
    """Patch the raw HID layer to yield `samples` in order, then repeat the
    last one forever (so a test can over-poll without an IndexError)."""
    seq = list(samples)

    def _next():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return mock.patch.object(vl, "_raw_sample", side_effect=_next)


def _always(sample):
    return mock.patch.object(vl, "_raw_sample", return_value=sample)


# ─────────────────────────────────────────────────────────────────────────
# 1. Pure parsing of the real bytes
# ─────────────────────────────────────────────────────────────────────────
class ParseCapturedRepliesTests(unittest.TestCase):

    def test_off_replies_are_off_with_no_battery(self):
        """6400003300 / 6400003400 -> OFF, battery None.

        Both are real headset-OFF samples; they differ only in byte[3], which
        carries no state."""
        for reply in (OFF_2015_27, OFF_2015_29):
            with self.subTest(reply=reply.hex()):
                self.assertEqual(vl.parse_reply(reply), (vl.LINK_OFF, None))

    def test_on_replies_carry_the_measured_battery(self):
        """6400dea101 -> ON/94 and 6400dcb101 -> ON/92 (0xde & 0x7F = 94,
        0xdc & 0x7F = 92 — the draining pack seen in the capture)."""
        self.assertEqual(vl.parse_reply(ON_2021_37), (vl.LINK_ON, 94))
        self.assertEqual(vl.parse_reply(ON_2021_40), (vl.LINK_ON, 92))

    def test_battery_sequence_matches_the_capture(self):
        """The five ON-era samples decode to 94, 94, 92, 92 — and the earlier
        0xe0 would be 96, which is why the mask is & 0x7F and not the raw byte
        (0xde is 222, not a percentage)."""
        got = [vl.parse_reply(r)[1]
               for r in (ON_2021_37, ON_2021_38, ON_2021_40, ON_2021_55)]
        self.assertEqual(got, [94, 94, 92, 92])
        self.assertEqual(TRANSIENT_2021_34[vl.BATTERY_BYTE] & vl.BATTERY_MASK, 96)

    def test_transient_is_not_yet_on_despite_a_battery_byte(self):
        """6400e0b100 at 20:21:34 — the battery byte is already populated
        (0xe0 = 96%) three seconds BEFORE the link came up, while byte[4] is
        still 00. Only byte[4] decides. Reporting this as ON would be exactly
        the guess this module refuses to make."""
        state, battery = vl.parse_reply(TRANSIENT_2021_34)
        self.assertNotEqual(state, vl.LINK_ON)
        self.assertEqual(state, vl.LINK_OFF)
        self.assertIsNone(battery)

    def test_handshake_replies_are_off_not_unknown(self):
        """The two mid-handshake replies still have a valid 0x64 echo and
        byte[4]=00, so they read OFF. (The debounce is what stops them from
        mattering — see the timeline test.)"""
        self.assertEqual(vl.parse_reply(HS_2021_26), (vl.LINK_OFF, None))
        self.assertEqual(vl.parse_reply(HS_2021_31), (vl.LINK_OFF, None))

    def test_no_reply_is_unknown_never_off(self):
        """The 105-second silence. This is the single most important assertion
        in the file: mapping no-reply to OFF would fire a spurious audio-device
        switch in the middle of the owner powering the headset ON."""
        self.assertEqual(vl.parse_reply(None), (vl.LINK_UNKNOWN, None))
        self.assertEqual(vl.parse_reply(b""), (vl.LINK_UNKNOWN, None))

    def test_truncated_reply_is_unknown(self):
        self.assertEqual(vl.parse_reply(bytes.fromhex("640000")),
                         (vl.LINK_UNKNOWN, None))

    def test_bad_magic_is_unknown_not_a_misparse(self):
        """byte[0] must echo the 0x64 we asked with. A reply that does not is
        some other interface's report or a stale buffer — even when its byte[4]
        would otherwise say ON, it must NOT be parsed."""
        garbage = bytes.fromhex("00" + ON_2021_37.hex()[2:])   # 0x00dea101...
        self.assertEqual(garbage[vl.LINK_BYTE], 0x01)          # would say ON
        self.assertEqual(vl.parse_reply(garbage), (vl.LINK_UNKNOWN, None))
        self.assertEqual(vl.parse_reply(bytes.fromhex("ff00dea101")),
                         (vl.LINK_UNKNOWN, None))

    def test_unobserved_link_byte_is_unknown(self):
        """Only 0x00 and 0x01 were ever seen in byte[4]. Anything else is
        reported as unknown rather than lumped in with either."""
        for link in (0x02, 0x7F, 0xFF):
            with self.subTest(link=link):
                reply = bytes([0x64, 0x00, 0xde, 0xa1, link])
                self.assertEqual(vl.parse_reply(reply), (vl.LINK_UNKNOWN, None))

    def test_bytes_1_and_3_are_ignored(self):
        """byte[3] oscillates in BOTH states (33<->34 off, a1<->b1 on) and
        byte[1] never moved. Mutating either must not change the verdict."""
        for base in (OFF_2015_27, ON_2021_37):
            for idx in (1, 3):
                mutated = bytearray(base)
                mutated[idx] ^= 0xFF
                with self.subTest(base=base.hex(), idx=idx):
                    self.assertEqual(vl.parse_reply(bytes(mutated)),
                                     vl.parse_reply(base))

    def test_zero_battery_byte_never_reports_zero_percent(self):
        """A synthetic ON reply with a 0x00 battery byte reports None, not 0.
        (Not observed live — the byte was always populated once the link was
        up — so this pins the defensive branch, and the branch is documented
        as defensive rather than measured.)"""
        state, battery = vl.parse_reply(bytes([0x64, 0x00, 0x00, 0xa1, 0x01]))
        self.assertEqual(state, vl.LINK_ON)
        self.assertIsNone(battery)


# ─────────────────────────────────────────────────────────────────────────
# 2. probe_once — one raw sample, no memory
# ─────────────────────────────────────────────────────────────────────────
class ProbeOnceTests(unittest.TestCase):

    def test_probe_once_reports_the_sampled_reply(self):
        with _always(ON_2021_40):
            self.assertEqual(vl.probe_once(), (vl.LINK_ON, 92))
        with _always(OFF_2015_27):
            self.assertEqual(vl.probe_once(), (vl.LINK_OFF, None))
        with _always(NO_REPLY):
            self.assertEqual(vl.probe_once(), (vl.LINK_UNKNOWN, None))

    def test_probe_once_takes_exactly_one_sample(self):
        with _always(OFF_2015_27) as raw:
            vl.probe_once()
        self.assertEqual(raw.call_count, 1)

    def test_probe_once_never_raises(self):
        """Contract: NEVER raises. A blown-up raw layer degrades to UNKNOWN."""
        with mock.patch.object(vl, "_raw_sample", side_effect=OSError("boom")):
            self.assertEqual(vl.probe_once(), (vl.LINK_UNKNOWN, None))

    def test_probe_once_has_no_memory(self):
        """No debounce, no smoothing — it reports the lying transient as-is.
        The debounce lives in VoidLink, deliberately."""
        with _feed([ON_2021_37, TRANSIENT_2021_34]):
            self.assertEqual(vl.probe_once()[0], vl.LINK_ON)
            self.assertEqual(vl.probe_once()[0], vl.LINK_OFF)


# ─────────────────────────────────────────────────────────────────────────
# 3. VoidLink — debounce + hold-across-unknown
# ─────────────────────────────────────────────────────────────────────────
class DebounceTests(unittest.TestCase):

    def test_two_agreeing_samples_establish_the_state(self):
        link = vl.VoidLink(debounce=2)
        with _always(ON_2021_40):
            self.assertEqual(link.state()[0], vl.LINK_UNKNOWN)  # not yet settled
            self.assertEqual(link.state(), (vl.LINK_ON, 92))

    def test_single_off_sample_amid_on_does_not_flip(self):
        """The debounce's whole job. One OFF in a run of ONs must be absorbed."""
        link = vl.VoidLink(debounce=2)
        with _feed([ON_2021_37, ON_2021_38]):
            link.state()
            self.assertEqual(link.state()[0], vl.LINK_ON)
        with _always(OFF_2015_27):
            self.assertEqual(link.state()[0], vl.LINK_ON)       # single OFF
        with _always(ON_2021_40):
            self.assertEqual(link.state()[0], vl.LINK_ON)       # run resumes
        with _always(OFF_2015_29):
            self.assertEqual(link.state()[0], vl.LINK_ON)       # first OFF again
            self.assertEqual(link.state()[0], vl.LINK_OFF)      # second -> flip

    def test_single_on_sample_amid_off_does_not_flip(self):
        link = vl.VoidLink(debounce=2)
        with _always(OFF_2015_27):
            link.state()
            self.assertEqual(link.state()[0], vl.LINK_OFF)
        with _always(ON_2021_37):
            self.assertEqual(link.state()[0], vl.LINK_OFF)
        with _always(OFF_2015_29):
            self.assertEqual(link.state()[0], vl.LINK_OFF)

    def test_debounce_is_configurable(self):
        link = vl.VoidLink(debounce=3)
        with _always(ON_2021_40):
            self.assertEqual(link.state()[0], vl.LINK_UNKNOWN)
            self.assertEqual(link.state()[0], vl.LINK_UNKNOWN)
            self.assertEqual(link.state()[0], vl.LINK_ON)

    def test_debounce_of_one_is_immediate(self):
        link = vl.VoidLink(debounce=1)
        with _always(ON_2021_40):
            self.assertEqual(link.state(), (vl.LINK_ON, 92))

    def test_nonsense_debounce_is_clamped_not_fatal(self):
        for bad in (0, -5, None, "two"):
            with self.subTest(bad=bad):
                link = vl.VoidLink(debounce=bad)
                self.assertGreaterEqual(link.debounce, 1)

    def test_shipped_default_debounce_is_at_least_two(self):
        """A HARD FLOOR on the shipped constant.

        Every other test in this class passes an explicit debounce (1, 2 or 3),
        so none of them can notice the default changing, and the only other
        reference to DEFAULT_DEBOUNCE in this file is a bound expressed in
        terms of DEFAULT_DEBOUNCE itself — self-referential, so it adapts to
        whatever the constant becomes. Nothing pinned the value, yet the value
        is what every production caller gets: shared_link(), is_headset_on()
        and battery_percent() all build VoidLink() with no argument.

        Hard-won behaviour 2 (a single sample can lie — 6400e0b100 at 20:21:34
        carried a live battery byte with the link still down) only protects
        anybody at a debounce of 2 or more. `test_debounce_of_one_is_immediate`
        proves 1 is a SUPPORTED CONFIGURATION; this asserts it is not the
        DEFAULT one, so a one-token "make it snappier" edit cannot ship with a
        green suite."""
        self.assertGreaterEqual(
            vl.DEFAULT_DEBOUNCE, 2,
            "the shipped default must absorb one lying sample")

    def test_default_constructed_link_absorbs_a_single_disagreeing_sample(self):
        """The same floor, stated as behaviour instead of as a number.

        VoidLink() is constructed with NO argument here, exactly as production
        constructs it. Settle it OFF on real captured OFF bytes, then feed ONE
        real captured ON reply: the believed state must not move. At a default
        of 1 it moves on that single sample, and the next disagreeing sample
        moves it back — an audio-device bounce mid-session."""
        link = vl.VoidLink()
        with _always(OFF_2015_27):
            for _ in range(link.debounce):      # settle, whatever the default
                link.state()
            self.assertEqual(link.state()[0], vl.LINK_OFF)
        with _always(ON_2021_37):
            self.assertEqual(
                link.state()[0], vl.LINK_OFF,
                "the default-constructed link flipped on ONE sample")


class UnknownHoldsTests(unittest.TestCase):

    def _settled_on(self):
        """A link that has genuinely settled ON (battery 92, from 6400dcb101)."""
        link = vl.VoidLink(debounce=2)
        with _always(ON_2021_40):
            link.state()
            self.assertEqual(link.state(), (vl.LINK_ON, 92))
        return link

    def test_hundred_consecutive_no_replies_hold_state(self):
        """The measured 105-second silence, replayed. A detector that mapped
        no-reply to OFF would have switched the audio device mid-handshake."""
        link = self._settled_on()
        with _always(NO_REPLY) as raw:
            for i in range(100):
                state, battery = link.state()
                self.assertEqual(state, vl.LINK_ON, f"flipped at sample {i}")
                self.assertEqual(battery, 92)
        self.assertEqual(raw.call_count, 100)
        self.assertEqual(link.last_battery, 92)

    def test_unknown_holds_an_off_state_too(self):
        link = vl.VoidLink(debounce=2)
        with _always(OFF_2015_27):
            link.state()
            link.state()
        with _always(NO_REPLY):
            for _ in range(100):
                self.assertEqual(link.state(), (vl.LINK_OFF, None))

    def test_unknown_before_anything_is_known_stays_unknown(self):
        """Holding the last known state cannot invent one. A fresh link that
        has only ever seen silence reports UNKNOWN, not OFF."""
        link = vl.VoidLink(debounce=2)
        with _always(NO_REPLY):
            for _ in range(10):
                self.assertEqual(link.state(), (vl.LINK_UNKNOWN, None))

    def test_unknown_does_not_manufacture_a_change(self):
        """A pending flip that is interrupted by silence must still need its
        full quota of agreeing samples.

        Half the story on purpose: this asserts only that silence cannot
        INVENT a change, which is true whether UNKNOWN holds the pending run
        or clears it. `test_silence_does_not_erase_a_pending_change` below
        covers the other half, and it is the one that actually pins the
        module's documented judgement call."""
        link = vl.VoidLink(debounce=2)
        with _always(ON_2021_40):
            link.state()
            link.state()
        with _always(OFF_2015_27):
            self.assertEqual(link.state()[0], vl.LINK_ON)   # pending 1/2
        with _always(NO_REPLY):
            for _ in range(50):
                self.assertEqual(link.state()[0], vl.LINK_ON)

    def test_silence_does_not_erase_a_pending_change(self):
        """The module's ONE explicitly-flagged judgement call, pinned.

        VoidLink's docstring says an UNKNOWN sample arriving mid-flip is "no
        information" and leaves the pending run INTACT rather than resetting
        it. The test above only proves silence cannot INVENT a flip — which
        holds under BOTH choices, so on its own it gives the decision no
        signal whatsoever. This is the other half: silence must not DESTROY a
        pending run either.

        NOT A MEASUREMENT. The 2026-09-04 capture contains no OFF/silence/OFF
        run, so the hardware proves neither choice; the module says as much.
        What this test does is lock in the choice the module documents, so
        that reversing it fails loudly instead of silently.

        Under the rejected alternative (clear _pending on UNKNOWN) the link
        below stays ON forever.
        """
        link = self._settled_on()                            # ON, battery 92
        with _always(OFF_2015_27):
            self.assertEqual(link.state()[0], vl.LINK_ON)    # pending 1/2
        with _always(NO_REPLY):
            self.assertEqual(link.state()[0], vl.LINK_ON)    # contributes nothing
        with _always(OFF_2015_29):
            state, battery = link.state()                    # 2/2 -> believed
        self.assertEqual(state, vl.LINK_OFF,
                         "silence erased the pending OFF run")
        self.assertIsNone(battery)
        self.assertIsNone(link.last_battery)

    def test_off_alternating_with_silence_still_reaches_off(self):
        """The realistic failure mode, at length.

        When the owner powers the headset off, the poller sees valid OFF
        replies interleaved with dongle silence — exactly the mix the capture
        shows either side of a link transition (real OFF replies from
        20:15:27, sustained no-reply at 20:19:41). If silence reset the
        pending run, `_pending_count` could never reach the debounce and the
        detector would report ON forever, with audio_switch still routing to a
        headset that is switched off.
        """
        link = self._settled_on()
        pattern = [OFF_2015_27, NO_REPLY] * 10               # 20 samples
        seen = []
        with _feed(pattern):
            for _ in pattern:
                seen.append(link.state()[0])
        self.assertEqual(seen[0], vl.LINK_ON)                # pending 1/2
        self.assertEqual(seen[1], vl.LINK_ON)                # silence: no info
        self.assertEqual(seen[2], vl.LINK_OFF,               # 2/2 -> flip
                         "never converged: the pending run was lost to silence")
        self.assertEqual(set(seen[2:]), {vl.LINK_OFF})
        self.assertIsNone(link.last_battery)

    def test_state_never_raises(self):
        link = vl.VoidLink(debounce=2)
        with mock.patch.object(vl, "_raw_sample", side_effect=RuntimeError("x")):
            self.assertEqual(link.state(), (vl.LINK_UNKNOWN, None))


class BatteryTests(unittest.TestCase):

    def test_battery_is_none_while_the_link_is_off(self):
        """Nobody may ever report "0%" for a powered-down headset. The OFF-era
        replies carry 0x00 in the battery byte and the ON-era ones carry a real
        value; going ON->OFF must clear the last reading, not keep the stale
        one and not turn it into a zero."""
        link = vl.VoidLink(debounce=2)
        with _always(ON_2021_37):
            link.state()
            self.assertEqual(link.state(), (vl.LINK_ON, 94))
        self.assertEqual(link.last_battery, 94)
        with _always(OFF_2015_27):
            link.state()
            state, battery = link.state()
        self.assertEqual(state, vl.LINK_OFF)
        self.assertIsNone(battery)
        self.assertIsNone(link.last_battery)
        self.assertNotEqual(battery, 0)

    def test_battery_refreshes_while_the_link_stays_up(self):
        """94 -> 92 across the capture's ON samples, without a state change."""
        link = vl.VoidLink(debounce=2)
        with _feed([ON_2021_37, ON_2021_38]):
            link.state()
            self.assertEqual(link.state(), (vl.LINK_ON, 94))
        with _always(ON_2021_40):
            self.assertEqual(link.state(), (vl.LINK_ON, 92))
        self.assertEqual(link.last_battery, 92)

    def test_battery_survives_silence(self):
        link = vl.VoidLink(debounce=2)
        with _always(ON_2021_40):
            link.state()
            link.state()
        with _always(NO_REPLY):
            link.state()
        self.assertEqual(link.last_battery, 92)


# ─────────────────────────────────────────────────────────────────────────
# 4. The whole captured transition, replayed end to end
# ─────────────────────────────────────────────────────────────────────────
class CaptureTimelineTests(unittest.TestCase):

    def test_replaying_the_real_capture_never_reports_on_early(self):
        """Feed the 2026-09-04 samples in order. The debounced state must:
          * read OFF through the headset-off run,
          * HOLD (not flip) through the no-reply gap,
          * still not be ON at 20:21:34 despite the battery byte,
          * become ON only after two agreeing ON samples (20:21:38)."""
        link = vl.VoidLink(debounce=2)
        seen = []
        with _feed(CAPTURE_TIMELINE):
            for _ in CAPTURE_TIMELINE:
                seen.append(link.state()[0])

        # index 0..3 = the OFF run, 4 = the 105 s silence, 5..7 = handshake,
        # 8..12 = the ON run.
        self.assertEqual(seen[1], vl.LINK_OFF)          # settled after 2 samples
        self.assertEqual(seen[4], vl.LINK_OFF)          # silence HELD the OFF
        self.assertEqual(seen[7], vl.LINK_OFF)          # transient did NOT lift
        self.assertEqual(seen[8], vl.LINK_OFF)          # first ON is pending only
        self.assertEqual(seen[9], vl.LINK_ON)           # second ON confirms
        self.assertEqual(seen[-1], vl.LINK_ON)
        self.assertNotIn(vl.LINK_ON, seen[:9],
                         "reported ON before the link actually came up")
        self.assertEqual(link.last_battery, 92)

    def test_the_silence_alone_would_not_have_flipped_anything(self):
        """The same timeline with the gap widened to the real 105 samples."""
        timeline = (CAPTURE_TIMELINE[:4] + [NO_REPLY] * 105
                    + CAPTURE_TIMELINE[5:])
        link = vl.VoidLink(debounce=2)
        states = []
        with _feed(timeline):
            for _ in timeline:
                states.append(link.state()[0])
        self.assertEqual(set(states[4:109]), {vl.LINK_OFF})
        self.assertEqual(states[-1], vl.LINK_ON)


# ─────────────────────────────────────────────────────────────────────────
# 5. is_headset_on — the three-valued convenience
# ─────────────────────────────────────────────────────────────────────────
class IsHeadsetOnTests(unittest.TestCase):

    def setUp(self):
        vl._shared = None      # fresh process-wide link per test

    tearDown = setUp

    def test_true_when_on(self):
        with _always(ON_2021_40):
            self.assertIs(vl.is_headset_on(), True)

    def test_false_when_off(self):
        with _always(OFF_2015_27):
            self.assertIs(vl.is_headset_on(), False)

    def test_none_when_genuinely_unknown(self):
        """Collapsing this into False is the exact defect that makes the
        feature misfire — during the 105-second pairing silence it would report
        the headset OFF while the owner was turning it ON."""
        with _always(NO_REPLY):
            result = vl.is_headset_on()
        self.assertIsNone(result)
        self.assertIsNot(result, False)

    def test_none_when_the_dongle_is_absent(self):
        with mock.patch.object(vl, "_HID_READY", False):
            vl._shared = None
            self.assertIsNone(vl.is_headset_on())

    def test_bounded_number_of_samples(self):
        """It settles within `debounce` extra samples rather than spinning.

        NOTE: this bound is deliberately expressed in terms of the constant, so
        it says nothing about the constant's VALUE. The floor on the value is
        pinned by DebounceTests.test_shipped_default_debounce_is_at_least_two."""
        with _always(NO_REPLY) as raw:
            vl.is_headset_on()
        self.assertLessEqual(raw.call_count, vl.DEFAULT_DEBOUNCE + 1)

    def test_one_transient_sample_does_not_bounce_the_device(self):
        """The production entry point, and the failure this whole feature is
        supposed to avoid.

        is_headset_on() drives shared_link(), which is built with the shipped
        default — so this is the only debounce the owner's machine ever runs.
        Settle it OFF, then deliver ONE ON reply, the same one-sample class as
        the capture's 20:21:34 transient. The answer must still be False: a
        True here moves the Windows default render device and the next sample
        moves it straight back."""
        with _always(OFF_2015_27):
            self.assertIs(vl.is_headset_on(), False)
        with _always(ON_2021_37) as raw:
            self.assertIs(vl.is_headset_on(), False,
                          "one sample flipped the shared link -> device bounce")
            self.assertEqual(raw.call_count, 1)      # it really was ONE sample

    def test_shares_state_across_calls(self):
        with _always(ON_2021_40):
            vl.is_headset_on()
        with _always(NO_REPLY):
            self.assertIs(vl.is_headset_on(), True)   # held, not forgotten
        self.assertEqual(vl.battery_percent(), 92)


# ─────────────────────────────────────────────────────────────────────────
# 6. Device-path handling (hard-won behaviour 3) + graceful degradation
# ─────────────────────────────────────────────────────────────────────────
class DevicePathTests(unittest.TestCase):

    def setUp(self):
        vl.invalidate_device()

    tearDown = setUp

    def test_failed_sample_invalidates_the_cached_path(self):
        """The dongle has been replugged for months and the machine carries
        ~19 present VID_1B1C interfaces; a cached path WILL go stale. A failed
        exchange must drop the cache so the next call re-discovers."""
        vl._device = ("\\\\?\\hid#vid_1b1c&pid_0a51#stale", 2, 5)
        with mock.patch.object(vl, "_HID_READY", True), \
                mock.patch.object(vl, "_exchange", return_value=None), \
                mock.patch.object(vl, "discover_device",
                                  return_value=None) as disco:
            self.assertIsNone(vl._raw_sample())
        self.assertIsNone(vl._device, "stale path was kept after a failure")
        self.assertEqual(disco.call_count, 1, "did not re-discover")

    def test_stale_path_is_retried_once_on_a_fresh_path(self):
        """A fast failure is the stale-path signature (CreateFileW returns
        immediately), so one re-discovery + retry is worth it — and it must
        succeed on the new path."""
        vl._device = ("stale", 2, 5)
        fresh = ("fresh", 2, 5)
        with mock.patch.object(vl, "_HID_READY", True), \
                mock.patch.object(vl, "_exchange",
                                  side_effect=[None, ON_2021_40]) as ex, \
                mock.patch.object(vl, "discover_device", return_value=fresh):
            self.assertEqual(vl._raw_sample(), ON_2021_40)
        self.assertEqual(ex.call_count, 2)
        self.assertEqual(vl._device, fresh)

    def test_freshly_discovered_path_is_not_retried(self):
        """No cache to blame -> one exchange only, so a silent (headset-off)
        device is never asked twice and the call stays sub-second."""
        with mock.patch.object(vl, "_HID_READY", True), \
                mock.patch.object(vl, "_exchange", return_value=None) as ex, \
                mock.patch.object(vl, "discover_device",
                                  return_value=("p", 2, 5)) as disco:
            self.assertIsNone(vl._raw_sample())
        self.assertEqual(ex.call_count, 1)
        self.assertEqual(disco.call_count, 1)

    def test_missing_dongle_is_unknown_not_off(self):
        with mock.patch.object(vl, "_HID_READY", True), \
                mock.patch.object(vl, "discover_device", return_value=None):
            self.assertEqual(vl.probe_once(), (vl.LINK_UNKNOWN, None))

    def test_raw_sample_never_raises(self):
        with mock.patch.object(vl, "_HID_READY", True), \
                mock.patch.object(vl, "discover_device",
                                  side_effect=OSError("setupapi exploded")):
            self.assertIsNone(vl._raw_sample())


class ImportSafetyTests(unittest.TestCase):

    def test_module_imports_without_hid(self):
        """CI is ubuntu-latest: there is no hid.dll and ctypes.WinDLL does not
        exist. The module must still import and answer UNKNOWN."""
        self.assertIsInstance(vl._HID_READY, bool)
        with mock.patch.object(vl, "_HID_READY", False):
            self.assertIsNone(vl.discover_device())
            self.assertIsNone(vl._raw_sample())
            self.assertEqual(vl.probe_once(), (vl.LINK_UNKNOWN, None))

    def test_public_contract_is_present(self):
        """Other components are being written against exactly these names."""
        self.assertEqual((vl.LINK_ON, vl.LINK_OFF, vl.LINK_UNKNOWN),
                         ("on", "off", "unknown"))
        for name in ("probe_once", "VoidLink", "is_headset_on"):
            self.assertTrue(hasattr(vl, name), name)
        link = vl.VoidLink()
        self.assertIsNone(link.last_battery)
        self.assertTrue(callable(link.state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
