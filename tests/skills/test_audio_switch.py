"""Unit tests for audio/audio_switch.py — the render-device filter, the
three-valued headset POWER detector, the hold-on-unknown state machine, and the
loud fallback diagnosis. No real COM, no real HID, no device mutation: every
device function is mocked, so this runs on the CI light tier too.

THE BUG THESE TESTS EXIST FOR (measured 2026-09-04 on this machine)
==================================================================
With the CORSAIR VOID ELITE Wireless headset POWERED OFF, Windows reported
BOTH of its endpoints as Status=OK / Active, because the endpoint belongs to
the DONGLE and the dongle stays plugged in. find_active() therefore answered
"headset on" forever and the auto-switch could never fire. The rows in
`ROWS_HEADSET_OFF` are that measurement; `test_void_off_is_reported_off_*`
asserts the new detector gets it right while explicitly proving the old signal
gets it wrong on the very same rows.

Second measured fact encoded here: the dongle answers NOTHING for ~105 seconds
while the headset pairs (20:19:41 -> 20:21:26). At the default 3 s poll that is
~35 unknown samples in a row, every time the owner powers the headset on, so
`test_hundred_unknowns_hold_state_and_never_switch` replays it.

Third: the saved fallback really was "Blue Snowball" — a MICROPHONE — which
`find_active(render_only=True)` can never match, so the switch-away was a
silent no-op. Verified live 2026-09-04: find_active("Blue Snowball") -> None,
find_active("Blue Snowball", render_only=False) -> "Microphone (Blue Snowball )".
"""
import sys
import types
import unittest
from unittest import mock

from audio import audio_switch as A


# ── The real 2026-09-04 endpoint list, trimmed to the rows that matter. ──────
# Both VOID rows read Active in this capture and the headset was POWERED OFF.
ROWS_HEADSET_OFF = [
    ("{0.0.0.00000000}.{ren}",
     "Headset Earphone (CORSAIR VOID ELITE Wireless Gaming Headset)", "Active"),
    ("{0.0.1.00000000}.{cap}",
     "Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Headset)", "Active"),
    ("{0.0.0.00000000}.{spk}", "Speakers (Realtek USB2.0 Audio)", "Active"),
    ("{0.0.1.00000000}.{mic}", "Microphone (Blue Snowball )", "Active"),
    ("{0.0.0.00000000}.{mon}", "LC32G7xT (NVIDIA High Definition Audio)", "NotPresent"),
]


class _FakeVoidLink:
    """Stand-in for audio.void_link with the exact public contract it promises."""
    LINK_ON = "on"
    LINK_OFF = "off"
    LINK_UNKNOWN = "unknown"

    def __init__(self, answer=None, battery=None, raises=False, device=True):
        self.answer = answer            # True / False / None
        self._battery = battery
        self._raises = raises
        self._device = device
        self.calls = 0

    def is_headset_on(self):
        self.calls += 1
        if self._raises:
            raise RuntimeError("simulated HID failure")
        return self.answer

    def battery_percent(self):
        return self._battery

    def discover_device(self):
        return ("\\\\?\\hid#vid_1b1c&pid_0a51", 20, 5) if self._device else None


class FindActiveTests(unittest.TestCase):
    def test_render_only_skips_capture_and_matches_fragment(self):
        rows = [
            # capture (mic) endpoint — id prefix {0.0.1. — must be SKIPPED for output
            ("{0.0.1.00000000}.{cap}", "Headset Microphone (3- CORSAIR VOID ELITE)", "Active"),
            # render (earphone) endpoint — id prefix {0.0.0. — the match we want
            ("{0.0.0.00000000}.{ren}", "Headset Earphone (3- CORSAIR VOID ELITE)", "Active"),
            ("{0.0.0.00000000}.{spk}", "Speakers (Realtek USB2.0 Audio)", "Active"),
        ]
        with mock.patch.object(A, "list_render", return_value=rows):
            got = A.find_active("corsair void elite")
        self.assertEqual(got[0], "{0.0.0.00000000}.{ren}")
        self.assertIn("Earphone", got[1])

    def test_inactive_headset_not_found(self):
        rows = [("{0.0.0.00000000}.{ren}", "Headset Earphone (CORSAIR VOID ELITE)", "NotPresent")]
        with mock.patch.object(A, "list_render", return_value=rows):
            self.assertIsNone(A.find_active("corsair void elite"))

    def test_empty_fragment_is_none(self):
        self.assertIsNone(A.find_active(""))

    def test_supplied_rows_are_used_without_re_enumerating(self):
        """The `rows` passthrough keeps one poll from enumerating twice."""
        with mock.patch.object(A, "list_render") as enumerate_:
            got = A.find_active("Realtek", rows=ROWS_HEADSET_OFF)
        self.assertEqual(got[1], "Speakers (Realtek USB2.0 Audio)")
        enumerate_.assert_not_called()


class LooksLikeCorsairVoidTests(unittest.TestCase):
    def test_matches_void_names_in_any_case(self):
        for frag in ("CORSAIR VOID ELITE", "void elite", "Corsair VOID RGB", "VOID"):
            self.assertTrue(A.looks_like_corsair_void(frag), frag)

    def test_does_not_match_unrelated_or_substring_names(self):
        for frag in ("Realtek USB2.0 Audio", "Blue Snowball", "avoid", "Devoid", ""):
            self.assertFalse(A.looks_like_corsair_void(frag), frag)


class HeadsetPoweredTests(unittest.TestCase):
    """The load-bearing fix: power state comes from the dongle, not the endpoint."""

    def test_void_off_is_reported_off_even_though_endpoints_read_active(self):
        # THE regression. Same rows, two answers: the old signal is wrong.
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF):
            self.assertIsNotNone(
                A.find_active("CORSAIR VOID ELITE"),
                "measurement check: the OFF headset really does read Active")
            with mock.patch.object(A, "_void_link",
                                   return_value=_FakeVoidLink(answer=False)):
                self.assertIs(A.headset_powered("CORSAIR VOID ELITE"), False)

    def test_void_on_is_reported_on(self):
        with mock.patch.object(A, "_void_link", return_value=_FakeVoidLink(answer=True)):
            self.assertIs(A.headset_powered("CORSAIR VOID ELITE"), True)

    def test_void_unknown_stays_unknown_and_never_falls_back_to_endpoint(self):
        """void_link saying "I don't know" must NOT be rescued by the endpoint
        check — that check is known-false for this headset, so falling back
        would reintroduce the exact bug.

        The rows supplied here are the measured OFF capture, in which both VOID
        endpoints read Active: any fallback to them would answer True. Asserting
        on the RESULT (not on an injected exception, which a defensive
        `except Exception` would quietly swallow) is what makes this test bite."""
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF) as enumerate_, \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink(answer=None)):
            self.assertIsNone(A.headset_powered("CORSAIR VOID ELITE"))
        enumerate_.assert_not_called()

    def test_void_link_unavailable_is_unknown_not_off(self):
        with mock.patch.object(A, "_void_link", return_value=None):
            self.assertIsNone(A.headset_powered("VOID ELITE"))

    def test_void_link_raising_is_unknown_not_off(self):
        with mock.patch.object(A, "_void_link",
                               return_value=_FakeVoidLink(raises=True)), \
             mock.patch.object(A, "_log"):
            self.assertIsNone(A.headset_powered("VOID ELITE"))

    def test_non_void_headset_uses_the_endpoint_check(self):
        rows = [("{0.0.0.00000000}.{ren}", "Headphones (SteelSeries Arctis)", "Active")]
        with mock.patch.object(A, "list_render", return_value=rows), \
             mock.patch.object(A, "_void_link") as probe:
            self.assertIs(A.headset_powered("SteelSeries Arctis"), True)
        probe.assert_not_called()   # a VOID dongle says nothing about other hardware
        rows = [("{0.0.0.00000000}.{ren}", "Headphones (SteelSeries Arctis)", "NotPresent")]
        with mock.patch.object(A, "list_render", return_value=rows):
            self.assertIs(A.headset_powered("SteelSeries Arctis"), False)

    def test_failed_enumeration_is_unknown_not_off(self):
        """Zero endpoints means COM/pycaw broke, not that a headset powered down."""
        with mock.patch.object(A, "list_render", return_value=[]):
            self.assertIsNone(A.headset_powered("SteelSeries Arctis"))

    def test_empty_fragment_is_unknown(self):
        self.assertIsNone(A.headset_powered(""))


class TickTransitionTests(unittest.TestCase):
    def _sw(self):
        return A.AudioAutoSwitch("VOID ELITE", "Realtek", poll_s=1.0, announce=lambda m: None)

    def test_off_to_on_switches_to_headset_and_remembers_prior(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="SPK"), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd:
            label = sw.tick(was_on=False)
        self.assertEqual(label, "to_headset")
        setd.assert_called_once_with("HS")
        self.assertEqual(sw._prior_default, "SPK")
        self.assertIs(sw._believed_on, True)

    def test_on_to_off_restores_remembered_prior(self):
        sw = self._sw()
        sw._prior_default = "SPK"
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd:
            label = sw.tick(was_on=True)
        self.assertEqual(label, "away")
        setd.assert_called_once_with("SPK")
        self.assertIsNone(sw._prior_default)

    def test_on_to_off_uses_fallback_when_no_prior(self):
        sw = self._sw()
        sw._prior_default = None
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd:
            label = sw.tick(was_on=True)
        self.assertEqual(label, "away")
        setd.assert_called_once_with("{0.0.0.00000000}.{spk}")

    def test_already_default_is_noop(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="HS"), \
             mock.patch.object(A, "set_default_render") as setd:
            label = sw.tick(was_on=True)          # on->on, headset already default
        self.assertIsNone(label)
        setd.assert_not_called()

    def test_powered_on_but_no_endpoint_says_so_instead_of_silence(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "find_active", return_value=None), \
             mock.patch.object(A, "set_default_render") as setd, \
             mock.patch.object(A, "_log") as log:
            self.assertIsNone(sw.tick(was_on=False))
        setd.assert_not_called()
        self.assertIn("no ACTIVE playback", " ".join(str(c.args[0]) for c in log.call_args_list))


class UnknownHoldsTests(unittest.TestCase):
    """UNKNOWN is not OFF — the 105-second pairing silence, replayed."""

    def _sw(self):
        return A.AudioAutoSwitch("CORSAIR VOID ELITE", "Realtek", poll_s=1.0,
                                 announce=lambda m: None)

    def test_hundred_unknowns_hold_state_and_never_switch(self):
        sw = self._sw()
        sw._believed_on = True
        with mock.patch.object(A, "headset_powered", return_value=None), \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink()), \
             mock.patch.object(A, "set_default_render") as setd, \
             mock.patch.object(A, "_log"):
            for _ in range(100):
                self.assertIsNone(sw.tick())
        setd.assert_not_called()
        self.assertIs(sw._believed_on, True, "belief must survive the silence")

    def test_on_then_silence_then_on_never_fires_a_spurious_switch_away(self):
        sw = self._sw()
        sw._believed_on = True
        sw._prior_default = "SPK"
        answers = [None] * 35 + [True]        # the measured ~105 s at a 3 s poll
        with mock.patch.object(A, "headset_powered", side_effect=answers), \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink()), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="HS"), \
             mock.patch.object(A, "set_default_render") as setd, \
             mock.patch.object(A, "_log"):
            labels = [sw.tick() for _ in answers]
        self.assertEqual(set(labels), {None})
        setd.assert_not_called()
        self.assertEqual(sw._prior_default, "SPK")

    def test_unknown_before_anything_is_known_does_nothing(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", return_value=None), \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink()), \
             mock.patch.object(A, "set_default_render") as setd, \
             mock.patch.object(A, "_log"):
            self.assertIsNone(sw.tick())
        setd.assert_not_called()
        self.assertIsNone(sw._believed_on)

    def test_first_unknown_logs_once_and_recovery_logs_once(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", side_effect=[None, None, None, True]), \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink()), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="HS"), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "_log") as log:
            for _ in range(4):
                sw.tick()
        msgs = [str(c.args[0]) for c in log.call_args_list]
        holding = [m for m in msgs if "HOLDING" in m]
        recovered = [m for m in msgs if "readable again" in m]
        self.assertEqual(len(holding), 1, msgs)      # once per silent run, not per sample
        self.assertEqual(len(recovered), 1, msgs)
        self.assertIn("3 unknown sample", recovered[0])

    def test_startup_unknown_then_on_grabs_the_headset(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", side_effect=[None, True]), \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink()), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "default_render_id", return_value="SPK"), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd, \
             mock.patch.object(A, "_log"):
            self.assertIsNone(sw.tick())
            self.assertEqual(sw.tick(), "to_headset")
        setd.assert_called_once_with("HS")

    def test_startup_off_records_but_does_not_yank_the_default(self):
        """Nothing believed yet + measured OFF must not move the owner's chosen
        device just because JARVIS happened to start with the headset off."""
        sw = self._sw()
        # list_render is stubbed only so a REGRESSION here fails fast instead of
        # reaching real COM: correct code never gets as far as resolving a
        # fallback on this path.
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF), \
             mock.patch.object(A, "set_default_render") as setd, \
             mock.patch.object(A, "_log"):
            self.assertIsNone(sw.tick())
        setd.assert_not_called()
        self.assertIs(sw._believed_on, False)


class FallbackDiagnosisTests(unittest.TestCase):
    """Defect (2): the configured fallback was a MICROPHONE and nothing said so."""

    def test_microphone_fallback_is_named_as_the_problem(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF):
            problem = A.fallback_problem("Blue Snowball")
        self.assertIsNotNone(problem)
        self.assertIn("RECORDING device", problem)
        self.assertIn("Blue Snowball", problem)
        self.assertIn("AUDIO_AUTOSWITCH_FALLBACK", problem)

    def test_render_device_in_a_non_active_state_is_named(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF):
            problem = A.fallback_problem("LC32G7xT")
        self.assertIn("NotPresent", problem)
        self.assertIn("not Active", problem)

    def test_unmatched_name_is_named(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF):
            problem = A.fallback_problem("Nonexistent Device")
        self.assertIn("matches NO audio device", problem)

    def test_usable_fallback_has_no_problem_and_logs_nothing(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF), \
             mock.patch.object(A, "_log") as log:
            self.assertIsNone(A.fallback_problem("Realtek USB2.0 Audio"))
            self.assertEqual(A.resolve_fallback("Realtek USB2.0 Audio"),
                             ("{0.0.0.00000000}.{spk}", "Speakers (Realtek USB2.0 Audio)"))
        log.assert_not_called()

    def test_empty_fallback_is_not_a_problem(self):
        self.assertIsNone(A.fallback_problem(""))
        self.assertIsNone(A.resolve_fallback(""))

    def test_resolve_fallback_logs_loudly_when_unusable(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF), \
             mock.patch.object(A, "_log") as log:
            self.assertIsNone(A.resolve_fallback("Blue Snowball"))
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("FALLBACK UNUSABLE", msgs)

    def test_switch_away_with_microphone_fallback_is_loud_not_silent(self):
        sw = A.AudioAutoSwitch("VOID ELITE", "Blue Snowball", announce=lambda m: None)
        sw._prior_default = None
        with mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF), \
             mock.patch.object(A, "set_default_render") as setd, \
             mock.patch.object(A, "_log") as log:
            self.assertIsNone(sw._switch_away())
        setd.assert_not_called()
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("FALLBACK UNUSABLE", msgs)
        self.assertIn("nothing to switch back to", msgs)


class SetDefaultGuardTests(unittest.TestCase):
    def test_empty_id_returns_false(self):
        self.assertFalse(A.set_default_render(""))


class StatusAndBatteryTests(unittest.TestCase):
    def _sw(self, **kw):
        return A.AudioAutoSwitch("VOID ELITE", "Realtek",
                                 announce=kw.get("announce", lambda m: None))

    def test_status_includes_battery_when_on(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(sw, "battery_pct", return_value=72.0):
            s = sw.status()
        self.assertIn("ON", s)
        self.assertIn("72% battery", s)

    def test_status_off_omits_battery(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(sw, "battery_pct", return_value=None):
            s = sw.status()
        self.assertIn("off", s.lower())
        self.assertNotIn("battery", s.lower())

    def test_status_says_unknown_instead_of_claiming_off(self):
        sw = self._sw()
        with mock.patch.object(A, "headset_powered", return_value=None):
            s = sw.status()
        self.assertIn("can't tell", s.lower())
        self.assertNotIn("is off", s.lower())

    def test_battery_prefers_void_link(self):
        sw = self._sw()
        with mock.patch.object(A, "void_battery_pct", return_value=85):
            self.assertEqual(sw.battery_pct(), 85.0)

    def test_battery_falls_back_to_hwinfo_for_non_void(self):
        sw = A.AudioAutoSwitch("SteelSeries Arctis", "Realtek", announce=lambda m: None)
        with mock.patch("audio.hwinfo.battery", return_value=41.0) as hw:
            self.assertEqual(sw.battery_pct(), 41.0)
        hw.assert_called_once_with("SteelSeries Arctis")

    def test_void_battery_pct_is_none_for_non_void_names(self):
        self.assertIsNone(A.void_battery_pct("Realtek USB2.0 Audio"))

    def test_low_battery_warns_once_then_rearms_after_recharge(self):
        msgs = []
        sw = self._sw(announce=msgs.append)
        with mock.patch.object(sw, "battery_pct", return_value=10.0):
            sw._check_low_battery()
            sw._check_low_battery()                 # still low -> only ONE warning
        self.assertEqual(len(msgs), 1)
        self.assertIn("low", msgs[0].lower())
        with mock.patch.object(sw, "battery_pct", return_value=80.0):
            sw._check_low_battery()                 # recharged -> re-arm
        with mock.patch.object(sw, "battery_pct", return_value=8.0):
            sw._check_low_battery()                 # low again -> warns again
        self.assertEqual(len(msgs), 2)

    def test_low_battery_no_source_is_silent(self):
        msgs = []
        sw = self._sw(announce=msgs.append)
        with mock.patch.object(sw, "battery_pct", return_value=None):
            sw._check_low_battery()
        self.assertEqual(msgs, [])


class SkillWiringTests(unittest.TestCase):
    """skills/audio_autoswitch.py must ask the same three-valued question."""

    def _actions(self, headset="CORSAIR VOID ELITE", fallback="Blue Snowball"):
        from skills import audio_autoswitch as S
        cfg = {"AUDIO_AUTOSWITCH_HEADSET": headset,
               "AUDIO_AUTOSWITCH_FALLBACK": fallback,
               "AUDIO_AUTOSWITCH_ENABLED": False}
        actions = {}
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)):
            S.register(actions)
        return S, actions, cfg

    def test_use_headset_refuses_when_measured_off(self):
        S, actions, cfg = self._actions()
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)), \
             mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "set_default_render") as setd:
            reply = actions["use_headset"]()
        self.assertIn("powered off", reply.lower())
        setd.assert_not_called()

    def test_use_headset_switches_but_admits_uncertainty_on_unknown(self):
        S, actions, cfg = self._actions()
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)), \
             mock.patch.object(A, "headset_powered", return_value=None), \
             mock.patch.object(A, "find_active", return_value=("HS", "Headset")), \
             mock.patch.object(A, "set_default_render", return_value=True) as setd:
            reply = actions["use_headset"]()
        setd.assert_called_once_with("HS")
        self.assertIn("couldn't confirm", reply.lower())

    def test_use_speakers_explains_the_microphone_fallback(self):
        S, actions, cfg = self._actions()
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)), \
             mock.patch.object(A, "list_render", return_value=ROWS_HEADSET_OFF), \
             mock.patch.object(A, "set_default_render") as setd, \
             mock.patch.object(A, "_log"):
            reply = actions["use_speakers"]()
        setd.assert_not_called()
        self.assertIn("recording device", reply.lower())

    def test_status_without_daemon_reports_unknown_honestly(self):
        S, actions, cfg = self._actions()
        S._DAEMON[0] = None
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)), \
             mock.patch.object(A, "headset_powered", return_value=None):
            reply = actions["audio_autoswitch_status"]()
        self.assertIn("can't tell", reply.lower())


# ════════════════════════════════════════════════════════════════════════════
# THE INPUT HALF — make the microphone follow the headset's power
# ════════════════════════════════════════════════════════════════════════════
# THE OUTAGE THESE EXIST FOR (measured 2026-09-05 00:48-00:52 on this machine).
# The owner powered the headset off. The OUTPUT switched correctly:
#     [audio] speakers -> [8] Speakers (Realtek USB2.0 Audio)
# The INPUT did not move at all:
#     [audio] mic      -> [1] Headset Microphone (CORSAIR VOID ELITE ...)
#     [vad] timeout - silent peak RMS=0.0000 (threshold 0.008)   x2, 30 s apart
# EXACTLY zero twice. Nothing in the repo had ever written the default CAPTURE
# endpoint, so Windows went on pointing the microphone at a headset that was
# switched off. ROWS_LIVE below is that machine state, read 2026-09-05 01:0x.
#
# Every device call is mocked. Nothing here enumerates real hardware, opens a
# stream, or touches the real default capture device — the owner lost time to a
# dead microphone tonight and a test must not be able to repeat that.

CORSAIR_MIC = "{0.0.1.00000000}.{da8ac56b-d877-4c10-8599-05afa21630ea}"
CORSAIR_EAR = "{0.0.0.00000000}.{30c225e7-0000-0000-0000-000000000000}"
SNOWBALL    = "{0.0.1.00000000}.{snowball}"
SPEAKERS    = "{0.0.0.00000000}.{b068b686-e9f7-4977-ba63-7f2105be812e}"
EMEET       = "{0.0.1.00000000}.{emeet}"
# A SECOND endpoint carrying the headset's name, left behind by a dongle
# replug: same friendly name, different id, no longer Active. Windows can go on
# pointing the default at one of these.
CORSAIR_MIC_STALE = "{0.0.1.00000000}.{corsair-stale}"
CORSAIR_MIC_NAME = "Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Headset)"

# The real 2026-09-05 enumeration, trimmed to the rows that carry meaning.
# Note the two traps it encodes:
#   * the CORSAIR earphone row comes BEFORE the CORSAIR mic row, which is why
#     find_active(..., render_only=False) hands back a PLAYBACK id;
#   * every "Realtek USB2.0 Audio" recording row is Unplugged or Disabled,
#     which is why the playback fallback cannot be borrowed for the mic side.
ROWS_LIVE = [
    (CORSAIR_EAR, "Headset Earphone (CORSAIR VOID ELITE Wireless Gaming Headset)", "Active"),
    (SPEAKERS, "Speakers (Realtek USB2.0 Audio)", "Active"),
    ("{0.0.1.00000000}.{dualsense}",
     "Headset Microphone (DualSense Wireless Controller)", "Active"),
    (CORSAIR_MIC, "Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Headset)", "Active"),
    (EMEET, "Microphone (2- HD Webcam eMeet C960)", "Active"),
    (SNOWBALL, "Microphone (Blue Snowball )", "Active"),
    ("{0.0.1.00000000}.{rtk-linein}", "Line In (Realtek USB2.0 Audio)", "Unplugged"),
    ("{0.0.1.00000000}.{rtk-mic}", "Microphone (Realtek USB2.0 Audio)", "Unplugged"),
    ("{0.0.1.00000000}.{rtk-aux}", "Internal AUX Jack (Realtek USB2.0 Audio)", "Disabled"),
]


# The same machine with every OTHER microphone gone: the powered-off CORSAIR
# is the only ACTIVE recording endpoint left. Since deaf-safety rule 8 this is
# the ONLY shape of "there is nothing to move to" - any other Active recording
# endpoint is a last-resort candidate, because unproven beats measured-dead.
# The Unplugged/Disabled Realtek rows stay: not-Active is not a candidate, and
# keeping them proves the filter reads STATE and not merely the name.
#
# ONE definition, referenced by both classes that need it. Two copies of a
# fixture drifting apart is this repo's #1 bug class and it does not get to
# start in the file that guards his hearing.
ROWS_NO_OTHER_MIC = [
    (did, name, state) for did, name, state in ROWS_LIVE
    if not (did.startswith("{0.0.1.") and state.lower() == "active"
            and "CORSAIR" not in name)
]
# One virtual microphone, for the ranking test. Measured on this machine
# 2026-09-05, "Microphone (Voicemod)" really is among the Active recording
# endpoints - a route rather than a microphone, which is why it ranks last.
VOICEMOD_ROW = ("{0.0.1.00000000}.{voicemod}", "Microphone (Voicemod)", "Active")


class _DebouncedVoidLink:
    """audio.void_link's REAL VoidLink debounce, driven by scripted RAW samples.

    Deliberately not a hand-written fake of the debounce: the property under
    test is that the SHIPPED debounce stops a single lying sample, and a
    re-implementation would only prove that the copy in this file works."""

    def __init__(self, samples):
        from audio import void_link as V
        self._V = V
        self._link = V.VoidLink()          # shipped DEFAULT_DEBOUNCE, not 1
        self._samples = list(samples)

    def is_headset_on(self):
        V = self._V
        nxt = self._samples.pop(0) if self._samples else V.LINK_UNKNOWN
        with mock.patch.object(V, "probe_once", return_value=(nxt, 61)):
            state, _batt = self._link.state()
        return {V.LINK_ON: True, V.LINK_OFF: False}.get(state)

    def battery_percent(self):
        return None

    def discover_device(self):
        return ("\\\\?\\hid#vid_1b1c&pid_0a51", 20, 5)


def _cur(device_id, rows=None):
    """(id, name) as `default_capture()` returns it, named from `rows`.

    default_capture() is the CHEAP single-device reader (8.5 ms measured, vs
    286 ms for a full enumeration) and default_capture_id() derives from it, so
    mocking this one covers both."""
    return device_id, dict((d, n) for d, n, _s in (rows or ROWS_LIVE)).get(device_id, "")


def _mic_sw(headset="CORSAIR VOID ELITE", mic_fallback="Blue Snowball",
            follow_mic=True, announce=None):
    return A.AudioAutoSwitch(headset, "Realtek USB2.0 Audio", poll_s=1.0,
                             announce=announce or (lambda m: None),
                             mic_fallback=mic_fallback, follow_mic=follow_mic)


class FindActiveCaptureTests(unittest.TestCase):
    """Resolving a MICROPHONE is its own function, not a flag on the speaker one."""

    def test_returns_the_capture_endpoint_not_the_earphone(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE):
            got = A.find_active_capture("CORSAIR VOID ELITE")
        self.assertEqual(got[0], CORSAIR_MIC)
        self.assertIn("Microphone", got[1])
        self.assertTrue(got[0].startswith(A.CAPTURE_PREFIX))

    def test_render_only_false_is_NOT_a_microphone_lookup(self):
        """The measured trap, pinned so nobody "simplifies" back into it.

        find_active(..., render_only=False) does not INVERT the filter, it
        DISABLES it — so on these rows it returns the EARPHONE, a playback id.
        Measured live 2026-09-05. Using it as the mic resolver would call
        SetDefaultEndpoint on a render id for the input direction."""
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE):
            loose = A.find_active("CORSAIR VOID ELITE", render_only=False)
            proper = A.find_active_capture("CORSAIR VOID ELITE")
        self.assertEqual(loose[0], CORSAIR_EAR)
        self.assertTrue(loose[0].startswith(A.RENDER_PREFIX),
                        "the measurement itself: the loose call yields PLAYBACK")
        self.assertNotEqual(loose[0], proper[0])

    def test_inactive_capture_is_not_found(self):
        rows = [(SNOWBALL, "Microphone (Blue Snowball )", "NotPresent")]
        with mock.patch.object(A, "list_render", return_value=rows):
            self.assertIsNone(A.find_active_capture("Snowball"))

    def test_empty_fragment_and_rows_passthrough(self):
        self.assertIsNone(A.find_active_capture(""))
        with mock.patch.object(A, "list_render") as enumerate_:
            got = A.find_active_capture("Snowball", rows=ROWS_LIVE)
        self.assertEqual(got[0], SNOWBALL)
        enumerate_.assert_not_called()

    def test_endpoint_name_resolves_and_admits_an_unknown_id(self):
        self.assertEqual(A.endpoint_name(SNOWBALL, rows=ROWS_LIVE),
                         "Microphone (Blue Snowball )")
        self.assertIsNone(A.endpoint_name("{0.0.1.00000000}.{gone}", rows=ROWS_LIVE))
        self.assertIsNone(A.endpoint_name("", rows=ROWS_LIVE))


class DirectionGuardTests(unittest.TestCase):
    """A render-id / capture-id mix-up must be IMPOSSIBLE at the last mile.

    The picker is not the only place this can go wrong. These guards sit on the
    COM call itself, so even a future caller that resolves the wrong device
    cannot write it — it gets a logged refusal and the device stays put."""

    def test_capture_setter_refuses_a_render_id(self):
        with mock.patch.object(A, "set_default_endpoint") as com, \
             mock.patch.object(A, "_log") as log:
            self.assertFalse(A.set_default_capture(SPEAKERS))
        com.assert_not_called()
        self.assertIn("REFUSED", " ".join(str(c.args[0]) for c in log.call_args_list))

    def test_render_setter_refuses_a_capture_id(self):
        with mock.patch.object(A, "set_default_endpoint") as com, \
             mock.patch.object(A, "_log") as log:
            self.assertFalse(A.set_default_render(CORSAIR_MIC))
        com.assert_not_called()
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("REFUSED", msgs)
        self.assertIn("RECORDING", msgs)

    def test_correct_directions_pass_through(self):
        # default_capture_id is mocked because set_default_capture now READS
        # THE DEFAULT BACK before it will return True (see
        # CaptureReadBackTests). Without the mock this would call live COM and
        # return the machine's real default microphone — a unit test must
        # never touch the owner's actual input device.
        with mock.patch.object(A, "set_default_endpoint", return_value=True) as com, \
             mock.patch.object(A, "default_capture_id", return_value=SNOWBALL):
            self.assertTrue(A.set_default_capture(SNOWBALL))
            self.assertTrue(A.set_default_render(SPEAKERS))
        self.assertEqual([c.args[0] for c in com.call_args_list], [SNOWBALL, SPEAKERS])

    def test_capture_setter_is_strict_about_the_prefix(self):
        """Unlike the render setter it REQUIRES {0.0.1., because every id it is
        handed comes from a prefix-filtered source, so strictness costs nothing
        and buys a hard guarantee."""
        with mock.patch.object(A, "set_default_endpoint") as com, \
             mock.patch.object(A, "_log"):
            self.assertFalse(A.set_default_capture("not-an-endpoint-id"))
            self.assertFalse(A.set_default_capture(""))
        com.assert_not_called()


class MicFallbackDiagnosisTests(unittest.TestCase):
    """An unusable microphone fallback must SAY SO — silence here means deaf."""

    def test_the_playback_fallback_cannot_be_borrowed_and_says_why(self):
        """AUDIO_AUTOSWITCH_FALLBACK's real value on this machine, measured:
        'Realtek USB2.0 Audio' has three recording rows and not one is Active."""
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE):
            problem = A.mic_fallback_problem("Realtek USB2.0 Audio")
        self.assertIsNotNone(problem)
        self.assertIn("none is Active", problem)
        self.assertIn("Unplugged", problem)

    def test_a_speaker_in_the_mic_slot_is_named_as_the_problem(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE):
            problem = A.mic_fallback_problem("Speakers (Realtek")
        self.assertIn("PLAYBACK device", problem)
        self.assertIn("AUDIO_AUTOSWITCH_MIC_FALLBACK", problem)

    def test_unmatched_name_is_named(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE):
            self.assertIn("matches NO audio device",
                          A.mic_fallback_problem("Nonexistent Mic"))

    def test_usable_fallback_has_no_problem_and_logs_nothing(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "_log") as log:
            self.assertIsNone(A.mic_fallback_problem("Blue Snowball"))
            self.assertEqual(A.resolve_mic_fallback("Blue Snowball"),
                             (SNOWBALL, "Microphone (Blue Snowball )"))
        log.assert_not_called()

    def test_empty_fallback_is_not_a_fault(self):
        self.assertIsNone(A.mic_fallback_problem(""))
        self.assertIsNone(A.resolve_mic_fallback(""))

    def test_resolve_logs_loudly_when_unusable(self):
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "_log") as log:
            self.assertIsNone(A.resolve_mic_fallback("Realtek USB2.0 Audio"))
        self.assertIn("MIC FALLBACK UNUSABLE",
                      " ".join(str(c.args[0]) for c in log.call_args_list))


class MicFollowTransitionTests(unittest.TestCase):
    """The two transitions the owner actually asked for."""

    def test_on_to_off_moves_the_microphone_to_the_desk_mic(self):
        """THE OUTAGE, fixed. Headset powers down while the mic is on it."""
        sw = _mic_sw()
        sw._prior_capture = None
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            sw.tick(was_on=True)
        setc.assert_called_once_with(SNOWBALL)
        self.assertEqual(sw.last_mic_label, "mic_away")

    def test_off_to_on_moves_the_microphone_to_the_headset(self):
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Earphone")), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            sw.tick(was_on=False)
        setc.assert_called_once_with(CORSAIR_MIC)
        self.assertEqual(sw.last_mic_label, "mic_to_headset")
        self.assertEqual(sw._prior_capture, SNOWBALL, "must remember the desk mic")

    def test_round_trip_returns_him_to_the_mic_he_started_on(self):
        sw = _mic_sw(mic_fallback="")          # no fallback: memory only
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            with mock.patch.object(A, "headset_powered", return_value=True), \
                 mock.patch.object(A, "default_capture", return_value=_cur(EMEET)):
                sw.tick(was_on=False)
            with mock.patch.object(A, "headset_powered", return_value=False), \
                 mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)):
                sw.tick(was_on=True)
        self.assertEqual([c.args[0] for c in setc.call_args_list],
                         [CORSAIR_MIC, EMEET])

    def test_already_on_the_target_writes_nothing(self):
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_capture") as setc:
            sw.tick(was_on=False)
        setc.assert_not_called()

    def test_steady_state_on_does_not_fight_a_manual_choice(self):
        """Once it has moved once, the owner may override by hand. ON is a
        preference, so it fires on the TRANSITION only."""
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)), \
             mock.patch.object(A, "set_default_capture") as setc:
            sw.tick(was_on=True)          # on -> on, mic manually on the Snowball
        setc.assert_not_called()

    def test_follow_mic_off_never_touches_the_input_at_all(self):
        """The feature is opt-in. With it off, not one capture call is made."""
        sw = _mic_sw(follow_mic=False)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture_id") as cur, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc:
            sw.tick(was_on=True)
        setc.assert_not_called()
        cur.assert_not_called()
        self.assertIsNone(sw.last_mic_label)


class MicDeafSafetyTests(unittest.TestCase):
    """The hard constraint: never end up on a microphone that cannot hear.

    A wrong-but-working mic beats a right-but-silent one, so every branch here
    asserts that an unverifiable target leaves the working device alone."""

    def test_never_remembers_the_headsets_own_mic_as_the_thing_to_restore(self):
        """The trap that would have shipped the outage back as a "fix".

        A naive "remember whatever was default" would record the HEADSET, and
        the next power-down would then "restore" him straight onto the device
        that had just powered off.

        The fixture is a dongle replug: a stale endpoint carrying the same
        friendly name sits alongside the live one, and Windows' default still
        points at the stale one. That is what makes the guard load-bearing
        rather than decorative — when the default is simply the live headset
        mic, _mic_to_headset returns early at the already-there check and the
        remember step is never reached at all. An earlier version of this test
        used exactly that setup and passed while the guard was DELETED; the
        mutation run on 2026-09-05 is what caught it."""
        rows = ROWS_LIVE + [(CORSAIR_MIC_STALE, CORSAIR_MIC_NAME, "Unplugged")]
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=rows), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "default_capture",
                               return_value=(CORSAIR_MIC_STALE, CORSAIR_MIC_NAME)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            sw.tick(was_on=False)
        setc.assert_called_once_with(CORSAIR_MIC)     # moved to the LIVE headset mic
        self.assertIsNone(
            sw._prior_capture,
            f"remembered {sw._prior_capture} — an endpoint carrying the "
            f"headset's own name — so the next power-down would restore him "
            f"onto the device that had just powered off")

    def test_already_on_the_headset_mic_remembers_nothing_either(self):
        """The plain case, kept as a separate assertion now that the test above
        no longer covers it: nothing to move, nothing to remember."""
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc:
            sw.tick(was_on=False)
        setc.assert_not_called()
        self.assertIsNone(sw._prior_capture)

    def test_a_stale_remembered_mic_is_re_verified_and_falls_back(self):
        """A remembered id is a claim about the past. If it is no longer an
        ACTIVE recording endpoint it must not be restored."""
        sw = _mic_sw()
        sw._prior_capture = "{0.0.1.00000000}.{unplugged-since}"
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        setc.assert_called_once_with(SNOWBALL)
        self.assertIn("no longer", " ".join(str(c.args[0]) for c in log.call_args_list))

    def test_headset_off_but_the_mic_is_elsewhere_changes_nothing(self):
        """The OFF branch is a rescue, not a policy. It only ever moves away
        from the device it has MEASURED as powered off."""
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(EMEET)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc:
            sw.tick(was_on=True)
        setc.assert_not_called()

    def test_startup_with_the_headset_off_RESCUES_the_dead_default(self):
        """Nothing believed yet + measured OFF + the default mic IS the dead
        headset. The OUTPUT side deliberately does nothing here (the owner's
        speakers are his business); the INPUT side must act, because that state
        is the deafness itself and it is exactly how tonight started."""
        sw = _mic_sw()
        self.assertIsNone(sw._believed_on)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render") as setr, \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            label = sw.tick()
        self.assertIsNone(label, "the OUTPUT must still be left alone at startup")
        setr.assert_not_called()
        setc.assert_called_once_with(SNOWBALL)

    def test_no_verified_target_leaves_the_dead_default_alone_and_shouts(self):
        """Nothing resolvable ANYWHERE => DO NOT SWITCH, and say so.

        REWRITTEN for deaf-safety rule 8, and the old version is worth quoting
        because it is the defect in test form. It read "Nothing resolvable =>
        DO NOT SWITCH. Moving the input somewhere unverified is how a 'fix'
        becomes a second outage" - and it ran against ROWS_LIVE, an
        enumeration holding three other ACTIVE recording endpoints. So it never
        pinned "there is nothing to move to". It pinned "nothing the OWNER
        CONFIGURED resolved, therefore stay on the microphone we have just
        measured POWERED OFF" - the device that reads peak RMS exactly 0.0000.
        The unverified move was never the outage; refusing to move was.

        The invariant that survives needs a fixture that actually states it:
        with no other ACTIVE recording endpoint in the enumeration there is
        genuinely nowhere to go, so nothing moves and DEAF RISK is shouted."""
        sw = _mic_sw(mic_fallback="Nonexistent Mic")
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_NO_OTHER_MIC), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        setc.assert_not_called()
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("DEAF RISK", msgs)
        self.assertIn("MIC FALLBACK UNUSABLE", msgs)
        self.assertIn("NO other ACTIVE recording", msgs,
                      "the reason has to be named as a HARDWARE fact - with a "
                      "candidate available, staying put would be the bug")

    def test_a_fallback_that_IS_the_headset_mic_is_refused_not_announced(self):
        """The fallback slot is where this module's bugs live, and on the input
        direction the failure is silent-and-lying rather than merely silent.

        AUDIO_AUTOSWITCH_MIC_FALLBACK is a fragment the owner types by hand,
        directly under a setting whose value is the headset's own name, so
        "CORSAIR VOID ELITE" landing in it is the obvious slip. With the
        headset POWERED OFF its microphone endpoint is still ACTIVE (the
        measured 2026-09-04 fact this whole module is built around), so the
        fragment RESOLVES - to the very device we are running away from.

        Without the guard every one of those steps succeeds: it is a {0.0.1.
        id so set_default_capture accepts it, SetDefaultEndpoint on the current
        default returns S_OK, _prior_capture is cleared, and he is told he was
        moved. The default never moved. He is deaf and has been told he is not.

        Required: refuse the write, and say DEAF RISK instead of announcing."""
        spoken = []
        sw = _mic_sw(mic_fallback="CORSAIR VOID ELITE", announce=spoken.append)
        # Prove the fixture really is the trap before asserting about it: the
        # fallback must genuinely resolve, and resolve to the headset's own mic.
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE):
            resolved = A.resolve_mic_fallback("CORSAIR VOID ELITE")
        self.assertEqual(resolved[0], CORSAIR_MIC,
                         "fixture is not the trap - the fallback did not "
                         "resolve to the headset's own microphone")
        with mock.patch.object(A, "headset_powered", return_value=False),              mock.patch.object(A, "list_render", return_value=ROWS_LIVE),              mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)),              mock.patch.object(A, "set_default_render", return_value=True),              mock.patch.object(A, "set_default_capture", return_value=True) as setc,              mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        # Rule 8 changed the ENDING, not the invariant. The refusal still
        # stands - the headset's own microphone is never written and never
        # announced - but the pass no longer stops there and sits on the dead
        # device: it drops to the last-resort rung and moves him elsewhere.
        # Assert the invariant, not the old ending.
        for call in setc.call_args_list:
            self.assertNotIn(call.args[0], (CORSAIR_MIC, CORSAIR_MIC_STALE),
                             f"wrote the powered-off headset's own microphone: "
                             f"{call.args[0]}")
        setc.assert_called_once_with(EMEET)
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("MIC FALLBACK REFUSED", msgs)
        self.assertNotIn(
            "CORSAIR", " ".join(spoken),
            f"announced a rescue onto the powered-off headset: {spoken}")

    def test_the_refused_fallback_does_not_repeat_forever(self):
        """The OFF branch runs on EVERY measured-off sample - that is what
        makes the rescue self-healing - so a target that "succeeds" without
        moving anything does not fail once, it fails every poll_s seconds
        forever. Reproduced before the fix: three ticks, three writes to the
        dead headset, three identical spoken rescues, _prior_capture cleared.

        Pinned as its own test because the single-tick test above would still
        pass if a future change made the refusal announce once per poll."""
        spoken = []
        sw = _mic_sw(mic_fallback="CORSAIR VOID ELITE", announce=spoken.append)
        # The fixture now MOVES when a write succeeds. It used to pin
        # default_capture() at the CORSAIR forever while set_default_capture
        # returned True - a pair that cannot happen since the read-back landed
        # (a write that does not move the default returns False), and one that
        # makes any correct rescue look like it is flapping.
        cur = {"v": _cur(CORSAIR_MIC)}

        def _moved(did):
            cur["v"] = _cur(did)
            return True

        with mock.patch.object(A, "headset_powered", return_value=False),              mock.patch.object(A, "list_render", return_value=ROWS_LIVE),              mock.patch.object(A, "default_capture", side_effect=lambda: cur["v"]),              mock.patch.object(A, "set_default_render", return_value=True),              mock.patch.object(A, "set_default_capture", side_effect=_moved) as setc:
            for _ in range(3):
                sw.tick(was_on=True)
        # ONE write, on the first poll, and never to the headset: polls 2 and 3
        # see a default that is no longer the CORSAIR and short-circuit.
        setc.assert_called_once_with(EMEET)
        self.assertEqual(
            [m for m in spoken if "CORSAIR" in m], [],
            f"spoke a rescue onto the powered-off headset: {spoken}")
        self.assertEqual(
            len([m for m in spoken if "listening on" in m.lower()]), 1,
            f"a rescue that repeats every poll is the flap this test exists "
            f"for: {spoken}")

    def test_a_fallback_naming_a_STALE_headset_endpoint_is_refused_too(self):
        """_is_headset_capture has two ways in, and the fallback must get both.

        After a dongle replug a second endpoint carrying the headset's friendly
        name can be the one that is Active. Matching only the id that
        find_active_capture resolves would let that one through, so the guard
        has to fall through to the NAME the enumeration records - the same
        both-ways test rule 5 already gets."""
        rows = [r for r in ROWS_LIVE if r[0] != CORSAIR_MIC] + [
            (CORSAIR_MIC_STALE, CORSAIR_MIC_NAME, "Active")]
        sw = _mic_sw(mic_fallback="CORSAIR VOID ELITE")
        with mock.patch.object(A, "headset_powered", return_value=False),              mock.patch.object(A, "list_render", return_value=rows),              mock.patch.object(A, "default_capture",
                               return_value=(CORSAIR_MIC_STALE, CORSAIR_MIC_NAME)),              mock.patch.object(A, "set_default_render", return_value=True),              mock.patch.object(A, "set_default_capture", return_value=True) as setc,              mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        # Same as above: refusing the STALE headset endpoint is the invariant.
        # Where he ends up afterwards is rule 8's business, and it is anywhere
        # but there.
        for call in setc.call_args_list:
            self.assertNotIn(call.args[0], (CORSAIR_MIC, CORSAIR_MIC_STALE),
                             f"wrote a headset endpoint: {call.args[0]}")
        setc.assert_called_once_with(EMEET)
        self.assertIn("MIC FALLBACK REFUSED",
                      " ".join(str(c.args[0]) for c in log.call_args_list))

    def test_a_GOOD_fallback_is_still_used_after_the_guard(self):
        """The guard must reject the headset and NOTHING else. A guard that
        refused every fallback would also 'never make him deaf' and would be
        useless - this is the test that stops the fix from being that."""
        spoken = []
        sw = _mic_sw(announce=spoken.append)          # fallback = "Blue Snowball"
        with mock.patch.object(A, "headset_powered", return_value=False),              mock.patch.object(A, "list_render", return_value=ROWS_LIVE),              mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)),              mock.patch.object(A, "set_default_render", return_value=True),              mock.patch.object(A, "set_default_capture", return_value=True) as setc,              mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        setc.assert_called_once_with(SNOWBALL)
        self.assertNotIn("MIC FALLBACK REFUSED",
                         " ".join(str(c.args[0]) for c in log.call_args_list))
        self.assertIn("Blue Snowball", " ".join(spoken))

    def test_headset_on_but_no_active_headset_mic_leaves_the_working_one(self):
        """Target missing on the ON side: he stays on the mic that works."""
        rows = [r for r in ROWS_LIVE if r[0] != CORSAIR_MIC]
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=rows), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=False)
        setc.assert_not_called()
        self.assertIn("no ACTIVE RECORDING",
                      " ".join(str(c.args[0]) for c in log.call_args_list))

    def test_a_failed_capture_write_says_he_may_be_deaf(self):
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=False), \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("FAILED", msgs)
        self.assertIn("may be deaf", msgs)

    def test_unreadable_default_capture_holds_instead_of_guessing(self):
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=None), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log"):
            sw.tick(was_on=True)
        setc.assert_not_called()

    def test_an_empty_enumeration_never_moves_the_microphone(self):
        for powered, was_on in ((False, True), (True, False)):
            sw = _mic_sw()
            with mock.patch.object(A, "headset_powered", return_value=powered), \
                 mock.patch.object(A, "list_render", return_value=[]), \
                 mock.patch.object(A, "find_active", return_value=None), \
                 mock.patch.object(A, "set_default_render"), \
                 mock.patch.object(A, "set_default_capture") as setc, \
                 mock.patch.object(A, "_log"):
                sw.tick(was_on=was_on)
            setc.assert_not_called()

    def test_the_off_side_does_not_enumerate_when_the_mic_is_not_the_headset(self):
        """The OFF branch runs on EVERY measured-off poll, forever. Measured
        2026-09-05, list_render() costs 286 ms on this machine against 8.5 ms
        for default_capture(), so the common answer - "no, the microphone is
        fine" - must be reached WITHOUT a full enumeration. ~10% of a core
        burned continuously would be a real regression on a box whose disk
        contention is already a standing problem."""
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)), \
             mock.patch.object(A, "list_render") as enumerate_, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc:
            for _ in range(20):
                sw.tick(was_on=False)
        enumerate_.assert_not_called()
        setc.assert_not_called()

    def test_a_nameless_default_that_is_NOT_the_headset_still_changes_nothing(self):
        """The second, enumeration-confirmed half of the OFF guard.

        The fast path bails out on the NAME, so with a name present the
        confirmed check is never reached and a deleted guard goes unnoticed —
        the mutation run on 2026-09-05 proved exactly that. A nameless default
        forces the slow path, where the id must still be confirmed against the
        enumeration before anything moves. Without that confirmation this
        would yank a perfectly good desk mic."""
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=(EMEET, "")), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE) as enumerate_, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc:
            sw.tick(was_on=True)
        enumerate_.assert_called()          # the fast path did NOT short-circuit
        setc.assert_not_called()            # and the confirmed check held

    def test_a_nameless_default_still_gets_the_full_check(self):
        """If Windows hands back no friendly name, fall through to the
        enumeration rather than assuming the safe case. Skipping the rescue
        because a name could not be read would be exactly the kind of
        unverified assumption this whole feature exists to stop."""
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=(CORSAIR_MIC, "")), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE) as enumerate_, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            sw.tick(was_on=True)
        enumerate_.assert_called()
        setc.assert_called_once_with(SNOWBALL)

    def test_default_capture_id_is_derived_from_default_capture(self):
        with mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)):
            self.assertEqual(A.default_capture_id(), SNOWBALL)
        with mock.patch.object(A, "default_capture", return_value=None):
            self.assertIsNone(A.default_capture_id())

    def test_the_input_rescue_still_runs_when_the_output_half_raises(self):
        """DEAF-SAFETY OUTRANKS THE SPEAKERS.

        Found by this test on 2026-09-05: with the mic step written as a plain
        statement after the render step, one raise inside _switch_away() —
        here, an enumeration that dies mid-pass — skipped the microphone half
        entirely, so an output fault could hide the input outage. It is a
        `finally` now. Both halves see the broken enumeration, so neither can
        move anything; what this pins is that the input half is REACHED and
        reports honestly, and that the output half's exception still
        propagates to _run() unchanged rather than being swallowed."""
        sw = _mic_sw()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", side_effect=RuntimeError("COM died")), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "set_default_render") as setr, \
             mock.patch.object(A, "_log") as log:
            with self.assertRaises(RuntimeError):
                sw.tick(was_on=True)
        setc.assert_not_called()
        setr.assert_not_called()
        self.assertIsNone(sw.last_mic_label)
        # Assert the BEHAVIOUR the docstring names — the input half was reached
        # and reported honestly — not one particular sentence. Which honest
        # sentence it reaches depends on the environment: locally the
        # enumeration dies and it logs "left exactly as"; on the CI runner
        # ctypes.wintypes is absent, so default_capture() fails FIRST and it
        # logs "would not name the default recording device ... leaving the
        # microphone alone" instead. Both are the input half running and
        # declining to move anything, which is the whole point. Pinning the
        # exact wording made this fail on CI while the code was correct
        # (2026-09-05).
        said = " ".join(str(c.args[0]) for c in log.call_args_list).lower()
        self.assertIn("microphone", said,
                      "the microphone half was never reached")
        for overclaim in ("listening on", "switched", "moved to"):
            self.assertNotIn(overclaim, said,
                             f"claimed a microphone change that never happened: {said!r}")


class MicUnknownHoldsTests(unittest.TestCase):
    """UNKNOWN must HOLD. It is not "off", and it is not a reason to switch."""

    def test_a_hundred_unknown_samples_never_move_the_microphone(self):
        sw = _mic_sw()
        sw._believed_on = True
        with mock.patch.object(A, "headset_powered", return_value=None), \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink()), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log"):
            for _ in range(100):
                sw.tick()
        setc.assert_not_called()
        self.assertIs(sw._believed_on, True)
        self.assertIsNone(sw.last_mic_label)

    def test_the_pairing_silence_does_not_yank_the_headset_mic_away(self):
        """35 unknowns is the measured ~105 s pairing window at a 3 s poll."""
        sw = _mic_sw()
        sw._believed_on = True
        sw._prior_capture = SNOWBALL
        answers = [None] * 35 + [True]
        with mock.patch.object(A, "headset_powered", side_effect=answers), \
             mock.patch.object(A, "_void_link", return_value=_FakeVoidLink()), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log"):
            for _ in answers:
                sw.tick()
        setc.assert_not_called()
        self.assertEqual(sw._prior_capture, SNOWBALL)

    def test_one_lying_sample_mid_connect_does_not_flip_the_microphone(self):
        """Hard-won behaviour 2, proven END TO END through the SHIPPED debounce.

        At 20:21:34 (measured 2026-09-04) a single sample disagreed with the
        link state three seconds before it actually changed. Driving the real
        VoidLink here, rather than a hand-rolled stand-in, is the point: it
        proves the protection is in the code that ships, not in this file."""
        link = _DebouncedVoidLink(["on", "on", "off", "on", "on"])
        sw = _mic_sw()
        sw._believed_on = True
        with mock.patch.object(A, "_void_link", return_value=link), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render") as setr, \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log"):
            beliefs = []
            for _ in range(5):
                sw.tick()
                beliefs.append(sw._believed_on)
        setc.assert_not_called()
        setr.assert_not_called()
        self.assertNotIn(False, beliefs, f"the lie reached the state machine: {beliefs}")

    def test_the_lying_sample_test_is_not_passing_blind(self):
        """Floor for the test above: TWO agreeing OFF samples MUST get through,
        or "nothing ever switches" would satisfy it."""
        link = _DebouncedVoidLink(["on", "on", "off", "off"])
        sw = _mic_sw()
        sw._believed_on = True
        with mock.patch.object(A, "_void_link", return_value=link), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log"):
            for _ in range(4):
                sw.tick()
        setc.assert_called_once_with(SNOWBALL)
        self.assertIs(sw._believed_on, False)


class MicSkillWiringTests(unittest.TestCase):
    """The voice actions, so he is never STUCK on the wrong microphone."""

    CFG = {
        "AUDIO_AUTOSWITCH_HEADSET": "CORSAIR VOID ELITE",
        "AUDIO_AUTOSWITCH_FALLBACK": "Realtek USB2.0 Audio",
        "AUDIO_AUTOSWITCH_MIC_FALLBACK": "Blue Snowball",
        "AUDIO_AUTOSWITCH_MIC": True,
        "AUDIO_AUTOSWITCH_ENABLED": False,
        "PREFERRED_INPUT_DEVICES": [],
        "MICROPHONE_INDEX": None,
    }

    def _actions(self, **over):
        from skills import audio_autoswitch as S
        cfg = dict(self.CFG, **over)
        actions = {}
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)):
            S.register(actions)
        return S, actions, cfg

    def _run(self, name, cfg, S, actions):
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)):
            return actions[name]()

    def test_every_new_name_is_registered_and_declared_speakable(self):
        S, actions, _ = self._actions()
        for name in ("use_headset_mic", "switch_to_headset_mic", "use_desk_mic",
                     "switch_to_desk_mic", "which_mic_is_active"):
            self.assertIn(name, actions, name)
            self.assertIn(name, S.SPEAK_VERBATIM_ACTIONS,
                          f"{name} would be computed, logged and never voiced")
            self.assertIn(name, S.PROMPT_EXAMPLES,
                          f"{name} is registered but the model can never emit it")

    def test_use_headset_mic_refuses_when_the_headset_measures_off(self):
        S, actions, cfg = self._actions()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "set_default_capture") as setc:
            reply = self._run("use_headset_mic", cfg, S, actions)
        setc.assert_not_called()
        self.assertIn("powered off", reply.lower())

    def test_use_headset_mic_obeys_on_unknown_but_admits_it(self):
        """UNKNOWN is what the dongle reports for the ~105 s it spends pairing —
        i.e. exactly when he would ask. Refusing then would leave him stuck,
        and it is self-healing: a genuinely-off headset gets measured off on
        the next poll and the watcher moves him back."""
        S, actions, cfg = self._actions()
        with mock.patch.object(A, "headset_powered", return_value=None), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            reply = self._run("use_headset_mic", cfg, S, actions)
        setc.assert_called_once_with(CORSAIR_MIC)
        self.assertIn("couldn't confirm", reply.lower())

    def test_use_desk_mic_switches_to_the_configured_fallback(self):
        S, actions, cfg = self._actions()
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc:
            reply = self._run("use_desk_mic", cfg, S, actions)
        setc.assert_called_once_with(SNOWBALL)
        self.assertIn("Blue Snowball", reply)

    def test_use_desk_mic_explains_an_unusable_fallback_instead_of_no_op(self):
        S, actions, cfg = self._actions(
            AUDIO_AUTOSWITCH_MIC_FALLBACK="Realtek USB2.0 Audio")
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log"):
            reply = self._run("use_desk_mic", cfg, S, actions)
        setc.assert_not_called()
        self.assertIn("none is Active", reply)

    def test_which_mic_names_the_preferred_list_override(self):
        """With PREFERRED_INPUT_DEVICES set, JARVIS does NOT follow the Windows
        default — an answer that named the default alone would be true and
        useless, and would hide why the headset can never win."""
        S, actions, cfg = self._actions(
            PREFERRED_INPUT_DEVICES=["Blue Snowball", "CORSAIR VOID"])
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)):
            reply = self._run("which_mic_is_active", cfg, S, actions)
        self.assertIn("not following it", reply.lower())
        self.assertIn("Blue Snowball", reply)

    def test_which_mic_reports_the_hard_off_setting(self):
        S, actions, cfg = self._actions(MICROPHONE_INDEX=-1)
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)):
            reply = self._run("which_mic_is_active", cfg, S, actions)
        self.assertIn("switched off entirely", reply.lower())

    def test_which_mic_names_the_device_and_the_follow_state(self):
        S, actions, cfg = self._actions()
        with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)), \
             mock.patch.object(A, "headset_powered", return_value=False):
            reply = self._run("which_mic_is_active", cfg, S, actions)
        self.assertIn("Blue Snowball", reply)
        self.assertIn("headset is off", reply.lower())
        self.assertNotIn("hearing", reply.lower())   # never claims signal

    def test_the_daemon_is_built_with_the_mic_knobs(self):
        S, _actions, cfg = self._actions()
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)):
            d = S._make_daemon()
        self.assertEqual(d.mic_fallback, "Blue Snowball")
        self.assertIs(d.follow_mic, True)

    def test_the_input_half_is_off_unless_it_is_asked_for(self):
        S, _actions, cfg = self._actions(AUDIO_AUTOSWITCH_MIC=False,
                                         AUDIO_AUTOSWITCH_MIC_FALLBACK="")
        with mock.patch.object(S, "_cfg", side_effect=lambda n, d=None: cfg.get(n, d)):
            d = S._make_daemon()
        self.assertIs(d.follow_mic, False)
        self.assertEqual(d.mic_fallback, "")


class _StubMonolith:
    """Stands in for bobert_companion in sys.modules.

    The two attributes below are the SAME globals the real capture loop reads
    (bobert_companion gets them from `from core.config import *`), so setting
    them here reproduces the precedence exactly."""

    def __init__(self, preferred=None, index=None):
        self.PREFERRED_INPUT_DEVICES = list(preferred or [])
        self.MICROPHONE_INDEX = index


class _monolith:
    """`with _monolith(...)`: install the stub, always restore what was there."""

    def __init__(self, preferred=None, index=None, absent=False):
        self._absent = absent
        self._stub = None if absent else _StubMonolith(preferred, index)

    def __enter__(self):
        import sys as _s
        self._sentinel = object()
        self._prev = _s.modules.get("bobert_companion", self._sentinel)
        if self._absent:
            _s.modules.pop("bobert_companion", None)
        else:
            _s.modules["bobert_companion"] = self._stub
        return self._stub

    def __exit__(self, *exc):
        import sys as _s
        if self._prev is self._sentinel:
            _s.modules.pop("bobert_companion", None)
        else:
            _s.modules["bobert_companion"] = self._prev
        return False


# The owner's ACTUAL live setting, read out of data/user_settings.json
# 2026-09-05. This exact list is why the capture auto-switch could move the
# Windows default and change nothing at all about what JARVIS heard.
LIVE_PREFERRED = ["Blue Snowball", "eMeet C960", "CORSAIR VOID"]


class MicSpokenClaimTests(unittest.TestCase):
    """The daemon may say the WINDOWS DEFAULT moved. It may only say JARVIS is
    "listening on" a device when JARVIS actually follows that default.

    THE DEFECT (measured 2026-09-05; this class is what would have caught it):
    with PREFERRED_INPUT_DEVICES set, bobert_companion's capture loop takes the
    first connected preferred device and NEVER evaluates the branch that
    resolves the live default endpoint. set_default_capture() moved the Windows
    default, JARVIS kept recording through the Blue Snowball, and
    _mic_to_headset announced "headset on - listening on Headset Microphone
    (CORSAIR VOID ELITE ...)". He puts the headset on, is told the boom mic is
    live, and is heard through a desk mic across the room.

    Note what is NOT asserted anywhere below: that any microphone produces
    audio. These tests pin WHICH SELECTION RULE WINS - a config fact - because
    that is the only thing the announcement claims after the fix. Proving a mic
    is live needs signal (peak RMS); that needs the input stream the monolith
    owns, and nothing here opens one."""

    def _on_transition(self, **monolith):
        """Drive one off->on tick. Returns (spoken, set_default_capture, sw)."""
        said = []
        sw = _mic_sw(announce=said.append)
        with _monolith(**monolith), \
             mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log"):
            sw.tick(was_on=False)
        return " ".join(said), setc, sw

    def test_THE_DEFECT_headset_on_never_claims_to_be_listening_on_it(self):
        """The exact live configuration. This assertion fails on the old code."""
        said, setc, _sw = self._on_transition(preferred=LIVE_PREFERRED)
        setc.assert_called_once_with(CORSAIR_MIC)      # the default DID move
        self.assertNotIn("listening on", said.lower(),
                         "spoke an unverified claim about what JARVIS records from")
        self.assertIn("not recording from it", said.lower())
        self.assertIn("Blue Snowball", said,
                      "must name the override so he knows what he IS heard through")
        self.assertIn(CORSAIR_MIC_NAME, said,
                      "must still say where the Windows default went")

    def test_with_no_override_it_does_say_listening_on_the_headset(self):
        """The feature's real behaviour once PREFERRED_INPUT_DEVICES is cleared.

        Without this the fix could 'pass' by never making the claim at all."""
        said, setc, _sw = self._on_transition(preferred=[], index=None)
        setc.assert_called_once_with(CORSAIR_MIC)
        self.assertIn("headset on - listening on " + CORSAIR_MIC_NAME, said)

    def test_the_headset_off_direction_is_gated_the_same_way(self):
        said = []
        sw = _mic_sw(announce=said.append)
        sw._prior_capture = None
        with _monolith(preferred=LIVE_PREFERRED), \
             mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log"):
            sw.tick(was_on=True)
        setc.assert_called_once_with(SNOWBALL)         # the rescue still ran
        spoken = " ".join(said)
        self.assertNotIn("listening on", spoken.lower())
        self.assertIn("Microphone (Blue Snowball )", spoken)

    def test_a_pinned_microphone_index_also_blocks_the_claim(self):
        said, _setc, _sw = self._on_transition(preferred=[], index=6)
        self.assertNotIn("listening on", said.lower())
        self.assertIn("index 6", said)

    def test_the_hard_off_microphone_setting_blocks_the_claim(self):
        """MICROPHONE_INDEX -1 means JARVIS has no microphone at all. Saying
        "listening on the headset" there is the loudest possible version of
        this bug."""
        said, _setc, _sw = self._on_transition(preferred=[], index=-1)
        self.assertNotIn("listening on", said.lower())
        self.assertIn("switched off", said.lower())

    def test_the_override_NEVER_suppresses_the_device_move(self):
        """DEAF-SAFETY. The gate is on the SENTENCE, never on the switch.

        If an override made the daemon skip set_default_capture, the headset
        powering off would leave the Windows default parked on a dead endpoint
        - which is the original outage. The write must still happen and the
        state machine's label and memory must be unchanged."""
        _said, setc, sw = self._on_transition(preferred=LIVE_PREFERRED)
        setc.assert_called_once_with(CORSAIR_MIC)
        self.assertEqual(sw.last_mic_label, "mic_to_headset")
        self.assertEqual(sw._prior_capture, SNOWBALL)

    def test_the_log_says_exactly_how_to_make_the_feature_work(self):
        with _monolith(preferred=LIVE_PREFERRED), \
             mock.patch.object(A, "_log") as log:
            claim = A.capture_claim(CORSAIR_MIC_NAME, "headset on")
        logged = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("PREFERRED_INPUT_DEVICES", logged)
        self.assertIn("MICROPHONE_INDEX", logged)
        self.assertNotIn("listening on", claim.lower())

    def test_capture_override_never_imports_the_monolith(self):
        """It must stay usable from `python -m audio.audio_switch` and on the
        light CI tier, where importing bobert_companion is neither cheap nor
        safe. Absent module == no capture loop to override == None."""
        import sys as _s
        with _monolith(absent=True):
            self.assertIsNone(A.capture_override())
            self.assertNotIn("bobert_companion", _s.modules,
                             "capture_override imported the monolith")

    def test_capture_override_hedges_instead_of_raising(self):
        """It runs on the poll thread. An exception here must degrade the
        sentence, never kill the microphone follow - and 'unreadable' must
        never be reported as 'follows the default'."""
        class _Boom:
            @property
            def MICROPHONE_INDEX(self):
                raise RuntimeError("boom")
        import sys as _s
        sentinel = object()
        prev = _s.modules.get("bobert_companion", sentinel)
        _s.modules["bobert_companion"] = _Boom()
        try:
            why = A.capture_override()
            with mock.patch.object(A, "_log"):
                claim = A.capture_claim("X", "headset on")
        finally:
            if prev is sentinel:
                _s.modules.pop("bobert_companion", None)
            else:
                _s.modules["bobert_companion"] = prev
        self.assertIsNotNone(why, "an unreadable config must not read as 'follows'")
        self.assertNotIn("listening on", claim.lower())

    def test_the_daemon_and_the_voice_action_agree_about_the_override(self):
        """ANTI-STALE-DUPLICATE. `which_mic_is_active` carries its own copy of
        this precedence rule. The two are allowed to word it differently; they
        are NOT allowed to disagree about whether an override exists, which is
        how one copy gets fixed and the other rots."""
        from skills import audio_autoswitch as S
        for pref, idx in (([], None), (LIVE_PREFERRED, None), ([], 6), ([], -1)):
            with self.subTest(preferred=pref, index=idx):
                cfg = dict(MicSkillWiringTests.CFG,
                           PREFERRED_INPUT_DEVICES=pref, MICROPHONE_INDEX=idx)
                actions = {}
                with mock.patch.object(S, "_cfg",
                                       side_effect=lambda n, d=None: cfg.get(n, d)):
                    S.register(actions)
                    with mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
                         mock.patch.object(A, "default_capture",
                                           return_value=_cur(SNOWBALL)), \
                         mock.patch.object(A, "headset_powered", return_value=False):
                        reply = actions["which_mic_is_active"]()
                with _monolith(preferred=pref, index=idx):
                    daemon_says_override = A.capture_override() is not None
                voice_says_override = ("not following" in reply.lower()
                                       or "pinned to" in reply.lower()
                                       or "switched off" in reply.lower())
                self.assertEqual(daemon_says_override, voice_says_override,
                                 "the two copies disagree; voice said: " + reply)


class CaptureReadBackTests(unittest.TestCase):
    """S_OK IS NOT "SELECTED". A capture write is believed only after a read-back.

    THE HOLE THESE CLOSE (2026-09-05). `set_default_endpoint`'s own docstring
    records that a capture default was never observed MOVING — the single live
    call wrote the id that was ALREADY default for all three roles, a proven
    no-op — yet `set_default_capture` returned True on that HRESULT and every
    one of the four deaf-safety rules then reasoned about the WRITE. None of
    them looked at the RESULT. So if IPolicyConfigVista accepts a {0.0.1. id
    without moving the endpoint, the headset powers off, the desk mic is
    resolved and verified Active, the write "succeeds", `_prior_capture` is
    cleared and JARVIS SAYS "headset off - listening on Microphone (Blue
    Snowball )" — while the default recording endpoint is still the dead
    CORSAIR mic and the VAD reads 0.0000. Every rule survives that intact.

    `default_capture_id()` already existed and cost a measured 7.8 / 8.2 /
    26.3 ms (min / median / max of nine live reads, 2026-09-05) against 286 ms
    for a full enumeration. It was called nowhere in the switch path.

    WHAT THESE TESTS DO NOT PROVE, and it matters: that the microphone hears
    anything. A read-back proves SELECTION, one level below the old claim, and
    "selected" is still not "producing audio" — only the VAD's peak RMS can
    say that, and a dead device reads exactly 0.0000."""

    def setUp(self):
        # The retry budget is real code; the wall-clock wait is not what is
        # under test and must never slow the suite.
        p = mock.patch.object(A, "_CAPTURE_READBACK_WAIT_S", 0)
        p.start()
        self.addCleanup(p.stop)

    def test_s_ok_without_a_move_is_reported_as_a_failure(self):
        """THE DEFECT ITSELF: the COM call is accepted, the device never moves."""
        with mock.patch.object(A, "set_default_endpoint", return_value=True) as com, \
             mock.patch.object(A, "default_capture_id",
                               return_value=CORSAIR_MIC) as readback, \
             mock.patch.object(A, "_log") as log:
            self.assertFalse(A.set_default_capture(SNOWBALL))
        com.assert_called_once_with(SNOWBALL)      # the write really was attempted
        self.assertEqual(readback.call_count, A._CAPTURE_READBACK_TRIES)
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("NOT MOVED", msgs)
        self.assertIn(CORSAIR_MIC, msgs,
                      "the log must name what Windows actually reports, not just "
                      "that something went wrong")

    def test_a_confirmed_move_returns_true_without_retrying(self):
        with mock.patch.object(A, "set_default_endpoint", return_value=True), \
             mock.patch.object(A, "default_capture_id",
                               return_value=SNOWBALL) as readback, \
             mock.patch.object(A, "_log") as log:
            self.assertTrue(A.set_default_capture(SNOWBALL))
        self.assertEqual(readback.call_count, 1,
                         "a device that moved is confirmed on the FIRST read - "
                         "nothing may sleep on the happy path")
        log.assert_not_called()

    def test_an_unreadable_read_back_is_not_a_verified_move(self):
        """None means "Windows would not tell me", which is not "it failed" and
        is certainly not "it worked". It counts as NOT moved - the caller then
        stays quiet instead of announcing - but the log must not claim the
        device stayed put either."""
        with mock.patch.object(A, "set_default_endpoint", return_value=True), \
             mock.patch.object(A, "default_capture_id", return_value=None), \
             mock.patch.object(A, "_log") as log:
            self.assertFalse(A.set_default_capture(SNOWBALL))
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("UNVERIFIED", msgs)
        self.assertNotIn("NOT MOVED", msgs,
                         "'I cannot prove it moved' is a different fact from "
                         "'it did not move' and must not be reported as it")

    def test_a_late_default_is_confirmed_rather_than_called_a_failure(self):
        """A propagation lag must not be mistaken for a refusal. Whether Windows
        ever actually lags here was NOT measured; the retry is a bounded
        insurance policy (3 reads, 50 ms apart => ~125 ms worst case)."""
        with mock.patch.object(A, "set_default_endpoint", return_value=True), \
             mock.patch.object(A, "default_capture_id",
                               side_effect=[CORSAIR_MIC, SNOWBALL]), \
             mock.patch.object(A, "_log") as log:
            self.assertTrue(A.set_default_capture(SNOWBALL))
        self.assertIn("attempt 2", " ".join(str(c.args[0]) for c in log.call_args_list))

    def test_a_write_that_failed_is_not_read_back_at_all(self):
        with mock.patch.object(A, "set_default_endpoint", return_value=False), \
             mock.patch.object(A, "default_capture_id") as readback:
            self.assertFalse(A.set_default_capture(SNOWBALL))
        readback.assert_not_called()

    def test_a_refused_id_never_reaches_com_or_the_read_back(self):
        with mock.patch.object(A, "set_default_endpoint") as com, \
             mock.patch.object(A, "default_capture_id") as readback, \
             mock.patch.object(A, "_log"):
            self.assertFalse(A.set_default_capture(SPEAKERS))
        com.assert_not_called()
        readback.assert_not_called()


def _claims_a_move(message: str) -> bool:
    """Does this sentence assert that the microphone actually MOVED?

    The thing that must never be said on an unproven write. Deliberately
    matches the two shapes capture_claim() can produce and nothing else, so a
    WARNING that says the opposite ("I may not be able to hear you") is not
    mistaken for a claim."""
    m = message.lower()
    return "listening on" in m or "windows' microphone is now" in m


class UnprovenCaptureIsNeverAnnouncedTests(unittest.TestCase):
    """End to end, at the level the owner experiences: what JARVIS SAYS.

    These drive tick() through the real _follow_mic_state / _mic_off_headset /
    _mic_to_headset code with COM stubbed at `set_default_endpoint` - i.e. the
    exact failure mode the module's own docstring says was never ruled out:
    the interface accepts the capture id and returns success while the default
    recording endpoint does not budge."""

    def setUp(self):
        p = mock.patch.object(A, "_CAPTURE_READBACK_WAIT_S", 0)
        p.start()
        self.addCleanup(p.stop)

    @staticmethod
    def _mic_only_sw(said):
        """A watcher whose OUTPUT half is silent BY CONSTRUCTION.

        `announce` is shared by both halves, so a playback fallback would put
        "headset off - audio back to Speakers ..." in the same list and the
        assertions below could not tell a spoken MIC claim from a spoken
        SPEAKER one. No fallback and no remembered prior default means
        _switch_away has nothing to move to: it logs and returns without
        speaking, leaving `said` to hold microphone claims only."""
        return A.AudioAutoSwitch("CORSAIR VOID ELITE", "", poll_s=1.0,
                                 announce=said.append,
                                 mic_fallback="Blue Snowball", follow_mic=True)

    def test_the_off_rescue_never_announces_a_move_that_did_not_happen(self):
        """THE OUTAGE, one level deeper than the first fix reached.

        Headset off, the default IS its microphone, a remembered desk mic
        resolves and re-verifies Active - and the write is accepted without
        moving anything. Before the read-back this announced "headset off -
        listening on Microphone (2- HD Webcam eMeet C960)" and cleared the
        memory, while the peak RMS stayed at 0.0000."""
        said = []
        sw = self._mic_only_sw(said)
        sw._prior_capture = EMEET
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "default_capture_id", return_value=CORSAIR_MIC), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_endpoint", return_value=True) as com, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        com.assert_called_once_with(EMEET)          # the rescue really was attempted
        # NOT `said == []`. Nothing here may CLAIM a move that was not proven -
        # that is the invariant - but this exit must still SPEAK, because a
        # silent failure here is the 00:48-00:52 outage itself. See
        # MicDeafRiskIsSPOKENTests.
        self.assertFalse([m for m in said if _claims_a_move(m)],
                         f"announced a rescue it could not prove: {said}")
        self.assertTrue([m for m in said if "may not be able to hear" in m],
                        f"failed silently instead of warning him: {said}")
        self.assertIsNone(sw.last_mic_label)
        self.assertEqual(sw._prior_capture, EMEET,
                         "the remembered mic must survive an unproven write, or "
                         "the next poll has nothing left to retry with")
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("NOT MOVED", msgs)
        self.assertIn("may be deaf", msgs)

    def test_an_unproven_rescue_is_retried_on_every_measured_off_poll(self):
        """The OFF branch is a self-healing rescue, so a write that did not take
        must leave the state that triggers it untouched. Three polls, three
        attempts, and not one spoken claim."""
        said = []
        sw = self._mic_only_sw(said)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "default_capture_id", return_value=CORSAIR_MIC), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_endpoint", return_value=True) as com, \
             mock.patch.object(A, "_log"):
            for _ in range(3):
                sw.tick(was_on=True)
        self.assertEqual([c.args[0] for c in com.call_args_list],
                         [SNOWBALL, SNOWBALL, SNOWBALL])
        # Three attempts, not one spoken CLAIM - and, because the voice is
        # rate-limited by _say_deaf's backoff rather than by the poll, exactly
        # one spoken WARNING rather than three.
        self.assertFalse([m for m in said if _claims_a_move(m)], said)
        self.assertEqual(len([m for m in said if "may not be able to hear" in m]),
                         1, f"the alert must not repeat every poll: {said}")

    def test_the_on_side_does_not_announce_an_unproven_move_either(self):
        said = []
        sw = self._mic_only_sw(said)
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "default_capture_id", return_value=SNOWBALL), \
             mock.patch.object(A, "set_default_endpoint", return_value=True) as com, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=False)
        com.assert_called_once_with(CORSAIR_MIC)
        self.assertEqual(said, [], f"announced an unproven move: {said}")
        self.assertIsNone(sw.last_mic_label)
        self.assertIn("could not be VERIFIED",
                      " ".join(str(c.args[0]) for c in log.call_args_list))

    def test_a_PROVEN_move_still_announces_normally(self):
        """The floor under all of the above: "nothing is ever announced" would
        satisfy every assertion here, and would itself be a deaf-safety
        regression. When the read-back agrees, the rescue speaks as before."""
        said = []
        sw = self._mic_only_sw(said)
        moved = {"id": CORSAIR_MIC}
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "default_capture_id",
                               side_effect=lambda: moved["id"]), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_endpoint",
                               side_effect=lambda did, *a: moved.__setitem__("id", did) or True), \
             mock.patch.object(A, "_log"):
            label = sw.tick(was_on=True)
        self.assertIsNone(label, "the OUTPUT half is not what this is about")
        self.assertEqual(sw.last_mic_label, "mic_away")
        self.assertEqual(len(said), 1, f"expected exactly one spoken rescue: {said}")
        self.assertIn("Blue Snowball", said[0])
        self.assertIsNone(sw._prior_capture)


class MicRescueCostTests(unittest.TestCase):
    """The OFF rescue must be CHEAP in the very state it exists for.

    THE DEFECT, measured with mocks 2026-09-05 against the v2.0.98 working
    tree. `_mic_off_headset` opens with a fast path that short-circuits when
    the default recording device is NOT the headset:

        if cur_name and self.headset.lower() not in cur_name.lower():
            return None

    That covers the case the rescue is not needed for. In the case it IS
    needed for - the default recording device IS the powered-off headset - the
    name DOES contain the headset fragment, so the test cannot fire and every
    poll fell through to `rows = list_render()`. Measured on this machine the
    same day, list_render() costs 334.9 / 307.4 / 305.3 / 290.6 ms across 51
    endpoints, against 8.0 ms for default_capture().

    Nothing about that state changes on a pass with no verified target, so it
    repeated forever. Simulated at the SHIPPED defaults (AUDIO_AUTOSWITCH_MIC
    just switched on, AUDIO_AUTOSWITCH_MIC_FALLBACK still "", headset off):
    1,200 polls in an hour produced 1,200 enumerations and 1,200 log lines -
    ~10% of a core and ~1,200 lines an hour into JARVIS's stdout, for as long
    as the headset stayed off. The same loop runs in a CORRECTLY configured
    setup whenever the desk mic is unplugged or asleep.

    THE THING THESE TESTS MUST NOT LET ANYONE BUY THAT CHEAPLY: a rescue that
    stops rescuing. Everything after the first three tests is about the fix
    NOT costing him his hearing - the hold is a TIMER, not a latch, and
    anything observable for 8 ms cancels it on the very next poll."""

    # Every ACTIVE recording endpoint stripped out EXCEPT the headset's own
    # microphone. Since the last-resort ladder landed, this is the ONLY shape
    # of the world in which the rescue genuinely has nothing to move to - with
    # any other Active mic present it now switches to one and the loop ends by
    # itself. So this is the fixture that reaches the hold at all, and the
    # tests below use it deliberately rather than for convenience.
    # The module-level fixture, not a second copy of it. Same rows, same
    # meaning, one place to correct when the machine changes.
    ROWS_NO_OTHER_MIC = ROWS_NO_OTHER_MIC

    def _pinned(self, sw):
        """Pin the watcher's clock. Returns a list whose [0] is 'now'."""
        t = [1000.0]
        sw._now = lambda: t[0]
        return t

    def test_the_stuck_rescue_stops_enumerating_every_poll(self):
        """The defect itself: 20 measured-off polls with nothing to move to."""
        sw = _mic_sw(mic_fallback="")          # the SHIPPED default: blank
        t = self._pinned(sw)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "list_render", return_value=self.ROWS_NO_OTHER_MIC) as enum_, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log") as log:
            for _ in range(20):
                sw.tick()
                t[0] += 3.0                    # the default poll interval
        setc.assert_not_called()               # still nothing verified to move to
        self.assertLessEqual(
            enum_.call_count, 3,
            f"{enum_.call_count} enumerations in 60 s of polling. At ~300 ms each "
            f"that is {enum_.call_count * 0.3 / 60 * 100:.0f}% of a core burned to "
            f"keep re-deciding the same thing")
        self.assertLessEqual(
            log.call_count, 2,
            f"{log.call_count} log lines in 60 s = {log.call_count * 60:.0f}/hour "
            f"of the same sentence")

    def test_the_first_sighting_still_shouts_in_full(self):
        """The rate limit must not buy quiet by hiding the fault.

        DEAF RISK is the line that tells him JARVIS cannot hear. It has to be
        there the FIRST time, in full, before any window applies."""
        sw = _mic_sw(mic_fallback="Nonexistent Mic")
        self._pinned(sw)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "list_render", return_value=self.ROWS_NO_OTHER_MIC), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture"), \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("DEAF RISK", msgs)
        self.assertIn("MIC FALLBACK UNUSABLE", msgs)

    def test_it_says_so_again_and_admits_what_it_suppressed(self):
        """A fault still true after MIC_RESCUE_RELOG_S says so again, and the
        repeat CONFESSES the rate limit rather than pretending the polls in
        between never happened. A limit that hides itself is the same silence
        this module was rewritten to remove."""
        sw = _mic_sw(mic_fallback="")
        t = self._pinned(sw)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "list_render", return_value=self.ROWS_NO_OTHER_MIC), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture"), \
             mock.patch.object(A, "_log") as log:
            for _ in range(240):               # 12 minutes at a 3 s poll
                sw.tick()
                t[0] += 3.0
        deaf = [str(c.args[0]) for c in log.call_args_list
                if "DEAF RISK" in str(c.args[0])]
        self.assertGreaterEqual(len(deaf), 2, "the fault went quiet and never came back")
        self.assertLessEqual(len(deaf), 5, f"still shouting {len(deaf)} times in 12 min")
        self.assertIn("short-circuited", deaf[-1],
                      "the repeat hid the fact that polls had been skipped")

    # ── everything below here is deaf-safety, not cost ───────────────────────

    def test_a_desk_mic_appearing_is_still_rescued(self):
        """THE ONE THAT MATTERS. The hold is a TIMER, not a latch.

        He is deaf on the powered-off headset with nothing to move to, so the
        rescue holds. Then he plugs the Blue Snowball back in. Nothing CHEAP
        changed - same default capture id, same fallback fragment, same
        remembered mic - so the only thing that can notice is the hold
        expiring. If it ever became permanent this is the test that fails, and
        he would stay deaf until he restarted JARVIS."""
        sw = _mic_sw(mic_fallback="Blue Snowball")
        t = self._pinned(sw)
        rows = {"now": self.ROWS_NO_OTHER_MIC}   # the Snowball is unplugged
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "list_render", side_effect=lambda: rows["now"]), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log"):
            for _ in range(10):                # 30 s of holding
                sw.tick()
                t[0] += 3.0
            setc.assert_not_called()
            rows["now"] = ROWS_LIVE            # the Snowball is plugged back in
            fired_after = None
            for i in range(1, 41):             # give it 2 minutes to notice
                sw.tick()
                t[0] += 3.0
                if setc.call_args_list:
                    fired_after = i * 3.0
                    break
        self.assertIsNotNone(
            fired_after,
            "the desk mic came back and the rescue never fired - he would still "
            "be deaf on a powered-off headset")
        setc.assert_called_once_with(SNOWBALL)
        self.assertLessEqual(
            fired_after, A.AudioAutoSwitch.MIC_RESCUE_RETRY_S + 3.0,
            f"took {fired_after}s to notice a working microphone")

    def test_windows_re_electing_the_default_cancels_the_hold_at_once(self):
        """A change JARVIS can see for 8 ms must never have to WAIT out the
        retry window. The default capture id is in the hold key, so a Windows
        re-election is acted on by the very next poll."""
        sw = _mic_sw(mic_fallback="Blue Snowball")
        t = self._pinned(sw)
        rows = self.ROWS_NO_OTHER_MIC
        cur = {"v": _cur(CORSAIR_MIC)}
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", side_effect=lambda: cur["v"]), \
             mock.patch.object(A, "list_render", side_effect=lambda: rows) as enum_, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log"):
            for _ in range(5):
                sw.tick()
                t[0] += 3.0
            held = enum_.call_count
            # A dongle replug: same friendly name, a DIFFERENT endpoint id.
            cur["v"] = (CORSAIR_MIC_STALE, CORSAIR_MIC_NAME)
            sw.tick()
        self.assertEqual(enum_.call_count, held + 1,
                         "the default recording device changed and the rescue sat "
                         "on a stale verdict instead of looking")
        setc.assert_not_called()               # there is still nothing to move TO

    def test_a_power_cycle_re_arms_the_rescue_immediately(self):
        """Headset off (holding) -> ON -> off again. The second OFF must pay
        for a full check on its FIRST poll. Anything else means a power cycle
        can land him in up to MIC_RESCUE_RETRY_S of unexamined deafness."""
        sw = _mic_sw(mic_fallback="")
        t = self._pinned(sw)
        power = {"on": False}
        with mock.patch.object(A, "headset_powered", side_effect=lambda f: power["on"]), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "default_capture_id", return_value=CORSAIR_MIC), \
             mock.patch.object(A, "list_render",
                               return_value=self.ROWS_NO_OTHER_MIC) as enum_, \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True), \
             mock.patch.object(A, "_log"):
            for _ in range(5):                 # settle into the hold
                sw.tick()
                t[0] += 3.0
            power["on"] = True                 # headset powers up
            sw.tick()
            t[0] += 3.0
            power["on"] = False                # and straight back off
            before = enum_.call_count
            sw.tick()
        self.assertEqual(enum_.call_count, before + 1,
                         "the first measured-off poll after a power cycle reused a "
                         "stale verdict instead of looking")

    def test_a_failed_write_is_never_held_and_retries_every_poll(self):
        """The one path deliberately NOT rate limited.

        Everywhere else the rescue looked and decided not to switch. Here it
        TRIED and could not show it worked, so he may be deaf RIGHT NOW and the
        retry is the thing that fixes it. Paying 300 ms every poll to keep
        trying to restore his hearing is the correct trade.

        Asserted as invariants rather than call counts on purpose: how MANY
        writes one pass attempts is the last-resort ladder's business and it
        may grow another rung. What must not change is that the pass is paid
        for in full every time and no verdict is ever held."""
        sw = _mic_sw(mic_fallback="Blue Snowball")
        t = self._pinned(sw)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE) as enum_, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=False) as setc, \
             mock.patch.object(A, "_log") as log:
            for _ in range(6):
                sw.tick()
                t[0] += 3.0
        self.assertEqual(enum_.call_count, 6, "a failing rescue stopped retrying")
        self.assertGreaterEqual(setc.call_count, 6,
                                "a failing rescue stopped trying to write")
        self.assertIsNone(sw._mic_hold_key,
                          "a pass that TRIED and failed was rate limited - he may be "
                          "deaf right now and the retry is what fixes it")
        deaf = [c for c in log.call_args_list if "may be deaf" in str(c.args[0])]
        self.assertGreaterEqual(len(deaf), 6, "it stopped saying he may be deaf")

    def test_the_cheap_fast_path_still_short_circuits_and_clears_the_hold(self):
        """The original fast path must survive: when the default is somebody
        else's microphone, no enumeration at all - and any hold left over from
        an earlier fault is dropped rather than left to expire."""
        sw = _mic_sw(mic_fallback="")
        t = self._pinned(sw)
        cur = {"v": _cur(CORSAIR_MIC)}
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", side_effect=lambda: cur["v"]), \
             mock.patch.object(A, "list_render",
                               return_value=self.ROWS_NO_OTHER_MIC) as enum_, \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture"), \
             mock.patch.object(A, "_log"):
            sw.tick()                          # enters the hold
            t[0] += 3.0
            self.assertIsNotNone(sw._mic_hold_key)
            # He picked his own desk mic. Named explicitly because it is NOT in
            # this fixture's enumeration - a nameless default is required to
            # fall through to the slow path, and that is a different test.
            cur["v"] = (EMEET, "Microphone (2- HD Webcam eMeet C960)")
            held = enum_.call_count
            for _ in range(10):
                sw.tick()
                t[0] += 3.0
        self.assertEqual(enum_.call_count, held,
                         "the cheap fast path enumerated anyway")
        self.assertIsNone(sw._mic_hold_key,
                          "a stale verdict was left behind for a state that no "
                          "longer applies")

    def test_a_held_verdict_never_suppresses_a_switch_the_same_pass_would_make(self):
        """The invariant that makes the whole thing safe, stated as a test.

        A hold can only be ENTERED by a pass that already enumerated and
        already decided not to switch. So with a resolvable target present from
        the very first poll the rescue fires immediately and no hold is ever
        taken - the timer cannot get between him and a working microphone."""
        sw = _mic_sw(mic_fallback="Blue Snowball")
        self._pinned(sw)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log"):
            sw.tick(was_on=True)
        setc.assert_called_once_with(SNOWBALL)
        self.assertIsNone(sw._mic_hold_key, "a successful rescue left a hold behind")


class MicVoiceActionClaimTests(unittest.TestCase):
    """`use_headset_mic` / `use_desk_mic` may only say "Listening on X" when
    JARVIS actually follows the Windows default capture device.

    THE SAME DEFECT AS MicSpokenClaimTests, AT ITS SECOND SITE (2026-09-05).
    The daemon's two transitions were gated by capture_claim(); these two VOICE
    actions were not - and they answer a DIRECT ORDER, which is the worst place
    to say something unverified. With the owner's live PREFERRED_INPUT_DEVICES
    he says "use my headset mic", hears "Listening on Headset Microphone
    (CORSAIR VOID ELITE ...) now, sir", talks into the boom mic, and is
    recorded through the Snowball on the desk. Every assertion below fails on
    the pre-fix sentence.

    Nothing here asserts that any microphone PRODUCES AUDIO. These pin WHICH
    SELECTION RULE WINS - a config fact - because that is the only thing the
    sentence claims after the fix. Proving a mic is live needs signal (peak
    RMS), and that needs the input stream the monolith owns; nothing in this
    file opens one, and no real device is switched."""

    SNOWBALL_NAME = "Microphone (Blue Snowball )"

    def _say(self, action, preferred=None, index=None, powered=True,
             absent=False):
        """Run one voice action against a STUBBED monolith. Nothing real: the
        device calls are mocked and set_default_capture never runs."""
        from skills import audio_autoswitch as S
        cfg = dict(MicSkillWiringTests.CFG,
                   PREFERRED_INPUT_DEVICES=list(preferred or []),
                   MICROPHONE_INDEX=index)
        actions = {}
        printed = []
        with mock.patch.object(S, "_cfg",
                               side_effect=lambda n, d=None: cfg.get(n, d)):
            S.register(actions)
            with _monolith(preferred=preferred, index=index, absent=absent), \
                 mock.patch.object(A, "headset_powered", return_value=powered), \
                 mock.patch.object(A, "find_active_capture",
                                   return_value=(CORSAIR_MIC, CORSAIR_MIC_NAME)), \
                 mock.patch.object(A, "resolve_mic_fallback",
                                   return_value=(SNOWBALL, self.SNOWBALL_NAME)), \
                 mock.patch.object(A, "set_default_capture",
                                   return_value=True) as setc, \
                 mock.patch("builtins.print", side_effect=printed.append):
                said = actions[action]()
        return said, setc, " ".join(str(p) for p in printed)

    def test_THE_DEFECT_use_headset_mic_does_not_claim_to_be_listening(self):
        """The exact live configuration. This assertion fails on the old code."""
        said, setc, _p = self._say("use_headset_mic", preferred=LIVE_PREFERRED)
        setc.assert_called_once_with(CORSAIR_MIC)      # the default DID move
        self.assertNotIn("listening on", said.lower(),
                         "spoke an unverified claim in answer to a direct order")
        self.assertIn(CORSAIR_MIC_NAME, said,
                      "must still say where the Windows default went")
        self.assertIn("Blue Snowball", said,
                      "must name the rule that actually decides")

    def test_use_desk_mic_is_gated_the_same_way(self):
        said, setc, _p = self._say("use_desk_mic", preferred=LIVE_PREFERRED)
        setc.assert_called_once_with(SNOWBALL)
        self.assertNotIn("listening on", said.lower())
        self.assertIn(self.SNOWBALL_NAME, said)

    def test_with_no_override_both_actions_say_listening_plainly(self):
        """The floor. Without it the fix could 'pass' by never claiming
        anything, which would make the feature useless instead of dishonest."""
        said, _s, _p = self._say("use_headset_mic", preferred=[])
        self.assertEqual(said, f"Listening on {CORSAIR_MIC_NAME} now, sir.")
        said, _s, _p = self._say("use_desk_mic", preferred=[])
        self.assertEqual(said, f"Listening on {self.SNOWBALL_NAME} now, sir.")

    def test_no_monolith_loaded_still_answers_plainly(self):
        """`python -m audio.audio_switch` and the light CI tier: no capture
        loop is running, so there is nothing to be overridden by."""
        said, _s, _p = self._say("use_headset_mic", absent=True)
        self.assertEqual(said, f"Listening on {CORSAIR_MIC_NAME} now, sir.")

    def test_the_pairing_unknown_branch_is_gated_too(self):
        """powered=None is the measured ~105 s pairing window - exactly when he
        asks. The power caveat and the capture caveat are different facts and
        he must get both."""
        said, _s, _p = self._say("use_headset_mic", preferred=LIVE_PREFERRED,
                                 powered=None)
        self.assertNotIn("listening on", said.lower())
        self.assertIn("powered on", said, "dropped the power caveat")
        self.assertIn("Blue Snowball", said, "dropped the capture caveat")

    def test_a_pinned_microphone_index_blocks_the_voice_claim(self):
        said, _s, _p = self._say("use_headset_mic", preferred=[], index=6)
        self.assertNotIn("listening on", said.lower())
        self.assertIn("index 6", said)

    def test_the_hard_off_microphone_setting_blocks_the_voice_claim(self):
        """MICROPHONE_INDEX -1 means JARVIS has no microphone at all. "Listening
        on the headset" there is the loudest possible version of this bug."""
        said, _s, _p = self._say("use_headset_mic", preferred=[], index=-1)
        self.assertNotIn("listening on", said.lower())
        self.assertIn("switched off", said.lower())

    def test_the_override_NEVER_suppresses_the_device_move(self):
        """DEAF-SAFETY. The gate is on the SENTENCE, never on the switch.

        If it ever gated the SWITCH, "use my desk mic" would stop rescuing him
        from a headset microphone that has powered off - which is the outage
        this whole feature exists for."""
        for action, want in (("use_headset_mic", CORSAIR_MIC),
                             ("use_desk_mic", SNOWBALL)):
            with self.subTest(action=action):
                _said, setc, _p = self._say(action, preferred=LIVE_PREFERRED,
                                            index=6)
                setc.assert_called_once_with(want)

    def test_the_hedge_never_claims_he_is_unheard_on_the_chosen_device(self):
        """His desk mic IS the first preferred entry, so when he asks for it he
        really is heard on it. The hedge must report WHICH RULE DECIDES and must
        NOT assert the negative - that would be this same bug pointing the other
        way, and it would talk him off a microphone that is working."""
        said, _s, _p = self._say("use_desk_mic", preferred=LIVE_PREFERRED)
        self.assertNotIn("not recording from it", said.lower())
        self.assertNotIn("i am not recording", said.lower())
        self.assertIn("Blue Snowball", said)

    def test_the_console_line_says_exactly_how_to_make_the_feature_work(self):
        """The sentence is heard once and gone; the fault is a config fault he
        has to be able to find afterwards."""
        _said, _s, printed = self._say("use_headset_mic",
                                       preferred=LIVE_PREFERRED)
        self.assertIn("PREFERRED_INPUT_DEVICES", printed)
        self.assertIn("MICROPHONE_INDEX", printed)

    def test_the_voice_actions_use_the_DAEMONS_override_rule_not_a_copy(self):
        """ANTI-STALE-DUPLICATE, and this defect WAS that: the daemon got
        capture_claim() while these two actions kept the old sentence. They are
        allowed to word it differently; they are NOT allowed to disagree with
        capture_override() about whether an override exists - on today's config
        or any other."""
        for pref, idx in (([], None), (LIVE_PREFERRED, None), ([], 6), ([], -1),
                          (["eMeet C960"], None)):
            with _monolith(preferred=pref, index=idx):
                daemon_hedges = A.capture_override() is not None
            for action in ("use_headset_mic", "use_desk_mic"):
                with self.subTest(preferred=pref, index=idx, action=action):
                    said, _s, _p = self._say(action, preferred=pref, index=idx)
                    voice_hedges = "listening on" not in said.lower()
                    self.assertEqual(
                        voice_hedges, daemon_hedges,
                        f"{action} disagrees with capture_override for "
                        f"preferred={pref} index={idx}: {said!r}")


class LastResortMicTests(unittest.TestCase):
    """DEAF-SAFETY RULE 8: never park him on a microphone measured DEAD.

    THE DEFECT THESE WERE WRITTEN FOR, reproduced with mocks 2026-09-05 before
    the fix. follow_mic on, AUDIO_AUTOSWITCH_MIC_FALLBACK "Blue Snowball". The
    headset was ON, so the watcher had moved the default capture to its
    microphone and remembered the Snowball. The Snowball was then unplugged
    (his USB tree is documented as marginal) and the headset powered off:

        the microphone I remembered (...snowball) is no longer an active
          recording device - falling back to 'Blue Snowball'
        MIC FALLBACK UNUSABLE: 'Blue Snowball' matches NO audio device ...
        DEAF RISK: ... there is nothing VERIFIED to move to and I am leaving it.
        set_default_capture calls: []

    Nothing moved. The default recording device stayed the CORSAIR headset
    microphone with the headset powered off - the device that read peak RMS
    exactly 0.0000 twice, thirty seconds apart, at 00:50:40 - while the SAME
    enumeration held "Microphone (2- HD Webcam eMeet C960)" (his own #2
    PREFERRED_INPUT_DEVICES entry) and "Headset Microphone (DualSense Wireless
    Controller)", both Active. The code called that outcome the safe direction
    and logged that it was "leaving him on what is working". It had already
    measured that the device it was leaving him on was not working.

    WHAT THESE TESTS DO NOT CLAIM. Not one of them asserts that the microphone
    picked can hear anything, because nothing in audio_switch can establish
    that: signal is peak RMS, peak RMS needs a capture stream, and the monolith
    owns the only one this process may have. They asserts what IS checkable -
    that he is moved OFF the proven-dead device, onto an endpoint verified
    Active and verified not-the-headset, and that the sentence he hears says
    exactly that much and no more."""

    def _rows_without(self, *ids):
        return [r for r in ROWS_LIVE if r[0] not in ids]

    # ── the reported outage ─────────────────────────────────────────────────
    def test_the_reported_outage_now_moves_off_the_dead_headset(self):
        """The exact scenario above: remembered mic gone, fallback unresolvable."""
        spoken = []
        sw = _mic_sw(announce=spoken.append)          # fallback = "Blue Snowball"
        sw._prior_capture = SNOWBALL                  # remembered while ON
        rows = self._rows_without(SNOWBALL)           # ... then unplugged
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=rows), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        setc.assert_called_once_with(EMEET)
        self.assertEqual(sw.last_mic_label, "mic_last_resort")
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("MIC FALLBACK UNUSABLE", msgs)  # still diagnosed, loudly
        self.assertIn("LAST RESORT", msgs)
        self.assertEqual(sw._prior_capture, None)

    def test_nothing_configured_at_all_is_still_rescued(self):
        """The SHIPPED default - AUDIO_AUTOSWITCH_MIC_FALLBACK blank, nothing
        remembered because the watcher started with the headset already off.
        This is the state the owner was actually in at 00:50."""
        sw = _mic_sw(mic_fallback="")
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log"):
            sw.tick()                                 # nothing believed yet
        setc.assert_called_once_with(EMEET)

    # ── what it will never pick ─────────────────────────────────────────────
    def test_it_never_picks_the_powered_off_headset_itself(self):
        """Including the stale replug endpoint, which is ACTIVE and carries the
        headset's own friendly name. Picking either would be the rescue
        rescuing him onto the thing he is being rescued from."""
        rows = ROWS_LIVE + [(CORSAIR_MIC_STALE, CORSAIR_MIC_NAME, "Active")]
        sw = _mic_sw(mic_fallback="")
        got = sw._last_resort_captures(rows)
        self.assertTrue(got, "no candidates at all on the live enumeration")
        for did, name in got:
            self.assertNotIn(did, (CORSAIR_MIC, CORSAIR_MIC_STALE))
            self.assertNotIn("CORSAIR", name)

    def test_it_never_picks_a_playback_or_an_inactive_endpoint(self):
        """The two structural filters, on the real rows: every Realtek
        recording endpoint here is Unplugged or Disabled, and the speakers and
        the headset earphone are {0.0.0. render ids."""
        sw = _mic_sw()
        for did, name in sw._last_resort_captures(ROWS_LIVE):
            self.assertTrue(did.startswith(A.CAPTURE_PREFIX), name)
            state = dict((d, s) for d, _n, s in ROWS_LIVE)[did]
            self.assertEqual(state.lower(), "active", name)

    def test_no_other_active_recording_endpoint_is_the_only_reason_to_stay(self):
        """The one honest hold. It is a HARDWARE statement - nothing is
        plugged in - not a configuration one."""
        sw = _mic_sw(mic_fallback="")
        self.assertEqual(sw._last_resort_captures(ROWS_NO_OTHER_MIC), [])
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_NO_OTHER_MIC), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        setc.assert_not_called()
        self.assertIn("NO other ACTIVE recording",
                      " ".join(str(c.args[0]) for c in log.call_args_list))

    # ── the ordering, and what it is worth ──────────────────────────────────
    def test_a_mains_powered_mic_outranks_a_battery_one_and_a_virtual_one(self):
        """MEASURED ordering on this machine's own rows. The eMeet is a USB
        webcam mic - it cannot be flat - so it goes first; the DualSense
        carries a battery and can be asleep in exactly the way the CORSAIR
        was; Voicemod is a route rather than a microphone and produces nothing
        unless its app is up. A RANKING, never an exclusion: all three stay
        candidates, because all three beat a device measured dead."""
        rows = self._rows_without(SNOWBALL) + [VOICEMOD_ROW]
        sw = _mic_sw(mic_fallback="")
        names = [n for _d, n in sw._last_resort_captures(rows)]
        self.assertEqual(names, ["Microphone (2- HD Webcam eMeet C960)",
                                 "Headset Microphone (DualSense Wireless Controller)",
                                 "Microphone (Voicemod)"])

    def test_the_ranking_is_a_preference_and_never_an_exclusion(self):
        """A virtual mic ALONE is still chosen. "It is probably not a real
        microphone" is a reason to try it last, not a reason to stay on one
        that has been measured silent."""
        rows = [r for r in ROWS_NO_OTHER_MIC] + [VOICEMOD_ROW]
        sw = _mic_sw(mic_fallback="")
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=rows), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log"):
            sw.tick(was_on=True)
        setc.assert_called_once_with(VOICEMOD_ROW[0])

    def test_the_number_of_write_attempts_is_bounded(self):
        """The rescue re-runs on EVERY measured-off poll, so an unbounded
        ladder on a box where every write fails would multiply a 300 ms pass by
        the endpoint count, forever."""
        extra = [(f"{{0.0.1.00000000}}.{{spare{i}}}", f"Microphone (Spare {i})",
                  "Active") for i in range(12)]
        sw = _mic_sw(mic_fallback="")
        self.assertLessEqual(len(sw._last_resort_captures(ROWS_LIVE + extra)),
                             A.AudioAutoSwitch.LAST_RESORT_MAX)

    # ── the rungs above it still win ────────────────────────────────────────
    def test_the_remembered_microphone_still_wins(self):
        """Rule 8 is the BOTTOM rung. What he was demonstrably using outranks
        anything this module guesses at."""
        sw = _mic_sw()
        sw._prior_capture = SNOWBALL
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        setc.assert_called_once_with(SNOWBALL)
        self.assertNotIn("LAST RESORT",
                         " ".join(str(c.args[0]) for c in log.call_args_list))

    def test_the_configured_fallback_still_wins(self):
        """And so does the name he typed. A guess must never quietly replace a
        choice - that would be a different bug wearing this fix's clothes."""
        spoken = []
        sw = _mic_sw(announce=spoken.append)          # fallback = "Blue Snowball"
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True) as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        setc.assert_called_once_with(SNOWBALL)
        self.assertNotIn("LAST RESORT",
                         " ".join(str(c.args[0]) for c in log.call_args_list))
        self.assertNotIn("last resort", " ".join(spoken).lower())

    # ── it says what it does not know ───────────────────────────────────────
    def test_the_announcement_never_claims_the_microphone_works(self):
        """The defining bug class of this project is claiming a state nobody
        verified, and tonight produced three fresh instances in this stack
        alone. "Selected" is not "producing audio", and the sentence he hears
        has to keep the two apart - out loud, not only in a log line he will
        never read at one in the morning."""
        spoken = []
        sw = _mic_sw(mic_fallback="", announce=spoken.append)
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True), \
             mock.patch.object(A, "_log"):
            sw.tick(was_on=True)
        said = " ".join(spoken)
        self.assertIn("eMeet", said, f"never named the device it moved to: {spoken}")
        self.assertIn("last resort", said.lower())
        self.assertIn("can't confirm it hears you", said.lower())

    def test_the_log_line_separates_what_was_verified_from_what_was_not(self):
        sw = _mic_sw(mic_fallback="")
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True), \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        line = [str(c.args[0]) for c in log.call_args_list
                if "LAST RESORT" in str(c.args[0])]
        self.assertEqual(len(line), 1, f"expected one LAST RESORT line: {line}")
        self.assertIn("VERIFIED:", line[0])
        self.assertIn("NOT VERIFIED:", line[0])

    # ── failure handling ────────────────────────────────────────────────────
    def test_a_last_resort_that_will_not_take_tries_the_next_one(self):
        """These candidates are guesses of equal standing, so a guess whose
        write cannot be VERIFIED (S_OK without a move - the thing the read-back
        exists to catch) is a reason to try the next guess, not a reason to
        stop on the dead device."""
        rows = self._rows_without(SNOWBALL)
        sw = _mic_sw(mic_fallback="")
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=rows), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture",
                               side_effect=lambda did: did != EMEET) as setc, \
             mock.patch.object(A, "_log"):
            sw.tick(was_on=True)
        self.assertEqual([c.args[0] for c in setc.call_args_list],
                         [EMEET, "{0.0.1.00000000}.{dualsense}"])
        self.assertEqual(sw.last_mic_label, "mic_last_resort")

    def test_every_candidate_refusing_is_reported_as_a_deaf_risk(self):
        sw = _mic_sw(mic_fallback="")
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=False), \
             mock.patch.object(A, "_log") as log:
            sw.tick(was_on=True)
        self.assertIsNone(sw.last_mic_label)
        msgs = " ".join(str(c.args[0]) for c in log.call_args_list)
        self.assertIn("DEAF RISK", msgs)
        self.assertIn("may be deaf", msgs)



class _PinnedClock:
    """A monotonic clock the test drives by hand.

    The alert's spacing is the thing under test and it must be testable
    without sleeping — 1,200 polls is an hour of wall time at poll_s=3.0."""

    def __init__(self, step=3.0):
        self.t = 0.0
        self.step = step

    def __call__(self, _self=None):     # bound as AudioAutoSwitch._now
        return self.t

    def tick(self):
        self.t += self.step


class MicDeafRiskIsSPOKENTests(unittest.TestCase):
    """Every exit that can leave him on a POWERED-OFF microphone must SPEAK.

    THE DEFECT THESE TESTS EXIST FOR (reproduced with mocks 2026-09-05).
    `_log()` is a print(). Every failure exit of `_mic_off_headset` used only
    that, while `announce()` — which reaches proactive_announce ->
    pending_speech.json, and is provably working, because the OUTPUT half's
    success line is what moved him to speakers he could hear — was wired to
    SUCCESSES ONLY.

    Six deaf-making exits, five ticks each, headset already ON at start so no
    ON transition had ever run and `_prior_capture` was None: spoken sentences
    about the microphone, ZERO, in every case. And in five of the six the one
    thing he heard was the OUTPUT half saying "headset off - audio back to
    Speakers (Realtek USB2.0 Audio)" — i.e. the only thing JARVIS said out loud
    during its own deafness was that the switch had worked.

    The existing `test_no_verified_target_leaves_the_dead_default_alone_and_
    shouts` did not catch it: it asserts on `_log`, so it passes with the voice
    disconnected. That is the shape of this bug — a test named "shouts" that
    only checks the console.

    WHAT THESE TESTS DO NOT CLAIM. None of them proves JARVIS can hear. That
    needs signal (peak RMS off an open input stream) and this module must never
    open one — the monolith owns the input stream and PortAudio's close path
    here is a heap-corruption crash. They assert the ALERT reaches the spoken
    channel, and that its wording never overclaims.
    """

    # Only the headset's own microphone is an ACTIVE capture endpoint, so the
    # last-resort ladder is empty: the "nowhere to go" state.
    ROWS_NO_OTHER_MIC = [r for r in ROWS_LIVE
                         if not (r[0].startswith(A.CAPTURE_PREFIX)
                                 and "CORSAIR" not in r[1])]

    def _drive(self, rows, default_cap, write_ok, mic_fallback="", ticks=5,
               raise_enum=False, clock=None):
        """One headset-is-OFF run. Returns the sentences that were SPOKEN.

        `_prior_default` is pre-set so the OUTPUT half needs no enumeration —
        otherwise the raise_enum case blows up in the speaker half instead of
        the microphone half being tested."""
        spoken = []
        clock = clock or _PinnedClock()
        sw = A.AudioAutoSwitch("CORSAIR VOID ELITE", "Realtek USB2.0 Audio",
                               poll_s=3.0, announce=spoken.append,
                               mic_fallback=mic_fallback, follow_mic=True)
        sw._believed_on = True          # already ON -> no ON transition ever ran
        sw._prior_default = SPEAKERS
        enum = (mock.patch.object(A, "list_render",
                                  side_effect=RuntimeError("enumeration blew up"))
                if raise_enum else
                mock.patch.object(A, "list_render", return_value=rows))
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "default_capture", return_value=default_cap), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=write_ok), \
             mock.patch.object(A.AudioAutoSwitch, "_now", clock), \
             mock.patch.object(A, "_log"), enum:
            for _ in range(ticks):
                try:
                    sw.tick()
                except Exception:
                    pass                # _run() swallows these in production
                clock.tick()
        return sw, spoken

    @staticmethod
    def _about_hearing(spoken):
        return [m for m in spoken
                if "hear" in m.lower() or "microphone" in m.lower()]

    # ── the six deaf-making exits ───────────────────────────────────────────

    def test_nothing_anywhere_to_move_to_is_SPOKEN_not_just_printed(self):
        """THE reported defect, in the shipped default configuration:
        AUDIO_AUTOSWITCH_MIC on, AUDIO_AUTOSWITCH_MIC_FALLBACK still ""."""
        _, spoken = self._drive(self.ROWS_NO_OTHER_MIC, _cur(CORSAIR_MIC), True)
        said = self._about_hearing(spoken)
        self.assertTrue(said, "the DEAF RISK exit printed and said NOTHING")
        self.assertIn("may not be able to hear you", said[0])

    def test_a_failed_capture_write_is_SPOKEN(self):
        _, spoken = self._drive(ROWS_LIVE, _cur(CORSAIR_MIC), False,
                                mic_fallback="Blue Snowball")
        said = self._about_hearing(spoken)
        self.assertTrue(said, "a failed capture write said nothing out loud")
        self.assertIn("cannot show that it took", " ".join(said))

    def test_every_last_resort_rung_refusing_is_SPOKEN(self):
        _, spoken = self._drive(ROWS_LIVE, _cur(CORSAIR_MIC), False)
        said = self._about_hearing(spoken)
        self.assertTrue(said, "an exhausted last-resort ladder said nothing")
        self.assertIn("refused the switch", " ".join(said))

    def test_an_unreadable_default_is_SPOKEN_after_confirmation(self):
        """confirm=2 — one transient COM failure must not raise a false alarm,
        so the FIRST sample is silent and the second speaks."""
        _, one = self._drive(ROWS_LIVE, None, True, ticks=1)
        self.assertEqual(self._about_hearing(one), [],
                         "a single unreadable sample must not cry deaf")
        _, spoken = self._drive(ROWS_LIVE, None, True, ticks=2)
        self.assertTrue(self._about_hearing(spoken),
                        "a PERSISTENT unreadable default must be spoken")

    def test_an_empty_enumeration_is_SPOKEN(self):
        # Held 30 s between attempts, so this needs more than two polls of
        # simulated time to reach its second confirming sample.
        _, spoken = self._drive([], _cur(CORSAIR_MIC), True, ticks=20)
        said = self._about_hearing(spoken)
        self.assertTrue(said, "an empty enumeration said nothing out loud")
        self.assertIn("could not list the audio devices", " ".join(said))

    def test_an_exception_in_the_input_half_is_SPOKEN_when_the_headset_is_off(self):
        _, spoken = self._drive(ROWS_LIVE, _cur(CORSAIR_MIC), True,
                                raise_enum=True)
        said = self._about_hearing(spoken)
        self.assertTrue(said, "the input half died silently")
        self.assertIn("something went wrong", " ".join(said))

    # ── it must not become a nuisance, and must not go quiet forever ────────

    def test_an_hour_of_deafness_is_a_handful_of_sentences_not_1200(self):
        """The OFF branch runs every poll by design. Before the backoff this
        exit printed ~1,200 lines an hour; the VOICE must not now do the same."""
        clock = _PinnedClock()
        _, spoken = self._drive(self.ROWS_NO_OTHER_MIC, _cur(CORSAIR_MIC), True,
                                ticks=1200, clock=clock)
        said = self._about_hearing(spoken)
        self.assertGreaterEqual(len(said), 1, "an hour deaf and never told")
        self.assertLessEqual(len(said), 6, f"far too talkative: {len(said)}")

    def test_it_never_goes_permanently_silent_about_an_ongoing_fault(self):
        """A JARVIS that cannot hear cannot be TOLD to stop saying so, so the
        alert backs off rather than stopping. Eight hours must still produce
        repeats, or a fault he stepped away from becomes silent again."""
        clock = _PinnedClock()
        _, spoken = self._drive(self.ROWS_NO_OTHER_MIC, _cur(CORSAIR_MIC), True,
                                ticks=9600, clock=clock)
        said = self._about_hearing(spoken)
        self.assertGreaterEqual(len(said), 8, f"went quiet: only {len(said)} in 8h")
        self.assertTrue(any("Still no change" in m for m in said),
                        "a repeat must say it is a repeat")

    def test_a_recovery_re_arms_and_says_so(self):
        clock = _PinnedClock()
        sw, spoken = self._drive(self.ROWS_NO_OTHER_MIC, _cur(CORSAIR_MIC), True,
                                 clock=clock)
        self.assertTrue(self._about_hearing(spoken))
        spoken.clear()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(SNOWBALL)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True), \
             mock.patch.object(A, "_log"):
            sw.tick()
        self.assertTrue(any("off the powered-off headset now" in m for m in spoken),
                        f"recovery was never announced: {spoken}")
        # ... and it must not claim it can hear, which nothing here measured.
        self.assertTrue(any("cannot prove" in m for m in spoken), spoken)

    def test_the_recovery_line_is_not_doubled_when_the_switch_itself_speaks(self):
        """capture_claim() already reports a successful move. Saying the
        re-arm line as well would tell him the same good news twice."""
        clock = _PinnedClock()
        sw, spoken = self._drive(self.ROWS_NO_OTHER_MIC, _cur(CORSAIR_MIC), True,
                                 clock=clock)
        self.assertTrue(self._about_hearing(spoken))
        spoken.clear()
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True), \
             mock.patch.object(A, "_log"):
            sw.tick()
        self.assertEqual(len([m for m in spoken if "cannot prove" in m]), 0,
                         f"recovery said twice: {spoken}")

    # ── the alert must not itself overclaim ────────────────────────────────

    def test_it_does_NOT_claim_deafness_when_something_overrides_the_default(self):
        """The whole point of capture_override(). Measured in the owner's LIVE
        settings 2026-09-05, PREFERRED_INPUT_DEVICES = ['Blue Snowball',
        'eMeet C960', 'CORSAIR VOID'] and _refresh_devices consults that list
        BEFORE the Windows default - so "the default is the dead headset" does
        NOT imply JARVIS is deaf, and saying it did would be a fourth
        unverified claim rather than a fix for the first three."""
        fake = types.ModuleType("bobert_companion")
        fake.MICROPHONE_INDEX = None
        fake.PREFERRED_INPUT_DEVICES = ["Blue Snowball", "eMeet C960", "CORSAIR VOID"]
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            _, spoken = self._drive(self.ROWS_NO_OTHER_MIC, _cur(CORSAIR_MIC), True)
        said = " ".join(self._about_hearing(spoken))
        self.assertTrue(said, "it must still SAY something")
        self.assertNotIn("I may not be able to hear you", said)
        self.assertIn("cannot tell from here whether I can still hear you", said)
        self.assertIn("Blue Snowball", said)

    def test_no_sentence_ever_claims_a_microphone_WORKS(self):
        """Nothing in this module can measure signal, so nothing in it may
        claim a device is producing audio."""
        banned = ("is working", "works now", "i can hear you now",
                  "you are being heard", "microphone is fine")
        for rows, cap, ok in ((self.ROWS_NO_OTHER_MIC, _cur(CORSAIR_MIC), True),
                              (ROWS_LIVE, _cur(CORSAIR_MIC), False),
                              (ROWS_LIVE, None, True)):
            _, spoken = self._drive(rows, cap, ok, ticks=3)
            for m in spoken:
                for phrase in banned:
                    self.assertNotIn(phrase, m.lower(), f"overclaimed: {m!r}")

    def test_a_raising_announce_cannot_break_the_rescue(self):
        """The alert is the LAST thing each exit does, so a broken speech
        channel must never cost him the device switch."""
        def boom(_msg):
            raise RuntimeError("pending_speech.json is unwritable")
        # fallback="" and no remembered prior default, so the OUTPUT half has
        # nothing to announce: `boom` can then only be reached from the input
        # half, which is what is under test. (_switch_away does NOT guard
        # announce - documented there, it propagates to _run.)
        sw = A.AudioAutoSwitch("CORSAIR VOID ELITE", "", poll_s=3.0,
                               announce=boom, mic_fallback="Blue Snowball",
                               follow_mic=True)
        sw._believed_on = True
        with mock.patch.object(A, "headset_powered", return_value=False), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture", return_value=_cur(CORSAIR_MIC)), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=False) as setc, \
             mock.patch.object(A, "_log") as log:
            sw.tick()                       # must NOT raise
        setc.assert_called_once_with(SNOWBALL)      # the rescue was still tried
        self.assertIn("could not SPEAK",
                      " ".join(str(c.args[0]) for c in log.call_args_list))



# ═══════════════════════════════════════════════════════════════════════════
# STEADY-STATE ON — "selected" is not "producing audio", and now it is checked
# ═══════════════════════════════════════════════════════════════════════════
import sys as _sys
import time as _time
import contextlib as _contextlib


class _FakeAudioProcessor:
    """Stand-in for core.audio_processor's VAD counters, modelling the exact
    contract record_speech writes and audio_switch reads.

    Only the three numbers matter, and each is written by a named call in the
    monolith's per-chunk loop (bobert_companion.record_speech):
        note_vad_poll()  -> last_vad_poll_ts   (EVERY chunk it inspects)
        note_vad_poll()  -> vad_session_start  (first chunk of the session)
        note_raw_rms()   -> last_audible_chunk_ts, and ONLY when the RAW rms
                            crosses core.audio_processor._AUDIBLE_RMS_FLOOR
                            (1e-5) — a device handing back null samples never
                            moves it.

    Ages are seconds-ago; None means "never set" (a literal 0.0 in the real
    dict), which is what a cold start and a stalled pipeline both look like."""

    _AUDIBLE_RMS_FLOOR = 1.0e-5

    def __init__(self, poll_age=1.0, audible_age=None, session_age=None,
                 raises=False):
        now = _time.time()
        self._raises = raises
        self._state = {
            "last_vad_poll_ts": 0.0 if poll_age is None else now - poll_age,
            "last_audible_chunk_ts": (0.0 if audible_age is None
                                      else now - audible_age),
            "vad_session_start": 0.0 if session_age is None else now - session_age,
            "total_vad_trips": 0,
        }

    def get_vad_state(self):
        if self._raises:
            raise RuntimeError("simulated audio_processor failure")
        return dict(self._state)


@_contextlib.contextmanager
def _signal(**kw):
    """Install a fake core.audio_processor for the duration of the block.

    audio_switch reads it out of sys.modules and never imports it (the same
    discipline capture_override() uses), so putting one there is the whole
    wiring — and putting NOTHING there is a real scenario the tests also
    exercise, because it is what a cold process looks like."""
    with mock.patch.dict(_sys.modules, {"core.audio_processor":
                                        _FakeAudioProcessor(**kw)}):
        yield


# Rows where the CORSAIR headset microphone is the ONLY Active recording
# endpoint on the machine — the one state in which staying put is the honest
# answer, and a hardware statement rather than a configuration one.
ROWS_ONLY_THE_HEADSET_MIC = [
    (CORSAIR_EAR, "Headset Earphone (CORSAIR VOID ELITE Wireless Gaming Headset)",
     "Active"),
    (SPEAKERS, "Speakers (Realtek USB2.0 Audio)", "Active"),
    (CORSAIR_MIC, CORSAIR_MIC_NAME, "Active"),
    (SNOWBALL, "Microphone (Blue Snowball )", "Unplugged"),
    (EMEET, "Microphone (2- HD Webcam eMeet C960)", "NotPresent"),
]


class CaptureSignalStateTests(unittest.TestCase):
    """The probe itself. THREE-VALUED, and the UNKNOWN row is load-bearing.

    JARVIS is not listening between turns — record_speech holds the mic only
    while it is capturing — so "no audible chunk lately" and "nobody was
    listening lately" look identical unless the poll timestamp is consulted.
    Reading the second as the first would yank a WORKING microphone every time
    the owner asked a long question, which is a new way to be deaf rather than
    a fix for the old one."""

    def test_a_running_loop_that_hears_nothing_is_SILENT(self):
        with _signal(poll_age=1.0, audible_age=95.0, session_age=600.0):
            state, why = A.capture_signal_state(60.0)
        self.assertEqual(state, "silent")
        self.assertIn("95 seconds", why)

    def test_a_device_that_has_never_produced_audio_is_SILENT_from_cold_start(self):
        """The boot case: headset already on, boom mic already flipped up. The
        audible timestamp is never set at all, so the session start is what the
        age has to be measured from — otherwise the one scenario with no
        recovery path would be the one scenario the probe cannot see."""
        with _signal(poll_age=0.5, audible_age=None, session_age=120.0):
            state, why = A.capture_signal_state(60.0)
        self.assertEqual(state, "silent")
        self.assertIn("since I started", why)

    def test_a_stale_poll_is_UNKNOWN_not_silent(self):
        """JARVIS spent two minutes speaking and thinking. Nothing polled the
        microphone, so the silence is about the LOOP and not the DEVICE."""
        with _signal(poll_age=120.0, audible_age=130.0, session_age=600.0):
            state, why = A.capture_signal_state(60.0)
        self.assertEqual(state, "unknown")
        self.assertIn("not listening", why)

    def test_recent_audio_is_AUDIBLE(self):
        with _signal(poll_age=0.4, audible_age=2.0, session_age=600.0):
            state, _why = A.capture_signal_state(60.0)
        self.assertEqual(state, "audible")

    def test_a_loop_that_has_never_polled_is_UNKNOWN(self):
        with _signal(poll_age=None, audible_age=None, session_age=None):
            state, _why = A.capture_signal_state(60.0)
        self.assertEqual(state, "unknown")

    def test_no_capture_pipeline_in_the_process_is_UNKNOWN(self):
        """`python -m audio.audio_switch` and the unit tests have no monolith.
        No capture loop means no evidence, and no evidence must never act."""
        with mock.patch.dict(_sys.modules):
            _sys.modules.pop("core.audio_processor", None)
            state, _why = A.capture_signal_state(60.0)
        self.assertEqual(state, "unknown")

    def test_a_non_positive_threshold_switches_the_probe_OFF(self):
        with _signal(poll_age=1.0, audible_age=9999.0, session_age=9999.0):
            for off in (0.0, -1.0):
                state, why = A.capture_signal_state(off)
                self.assertEqual(state, "unknown")
                self.assertIn("switched off", why)

    def test_a_raising_probe_is_UNKNOWN_and_never_silent(self):
        """An error must be able to make the watchdog inert. It must never be
        able to make it ACT — that direction ends in a spurious device switch
        on a machine whose microphone was fine."""
        with _signal(raises=True):
            state, why = A.capture_signal_state(60.0)
        self.assertEqual(state, "unknown")
        self.assertIn("could not read", why)


class MicSilentOnHeadsetTests(unittest.TestCase):
    """THE DEFECT (audio/audio_switch.py, the steady-state ON branch).

    `if was_on is True: return None` was the WHOLE steady-state ON branch, so
    the ON side fired once on the power-up transition and never re-evaluated.

    HOW THAT LEAVES HIM DEAF. He powers the VOID ELITE on for game audio with
    the boom mic flipped up — the headset's own hardware mute, and its resting
    position. void_link reports the link ON, the transition fires, the default
    capture endpoint moves to the headset microphone, and that microphone reads
    exactly 0.0000 RMS. The headset stays powered on for hours. `was_on` is
    True on every subsequent tick, so the input half returned immediately; the
    OFF rescue could not fire because the headset MEASURES on; and nothing
    consulted the VAD. He is deaf until he physically powers the headset down.
    Same shape if the endpoint is muted in Windows, or he walks out of range.

    Every device call here is mocked. Nothing enumerates real hardware, opens a
    stream, or touches the real default capture device."""

    # A distinct sentinel, because `None` is a REAL value here: it is what
    # default_capture() returns when Windows will not name the default
    # recording device, and that case has its own hold-and-say-so branch.
    _DEFAULT_CAP = object()

    def _drive(self, *, rows=ROWS_LIVE, cap=_DEFAULT_CAP, ok=True, ticks=1,
               override=None, prior=None, mic_fallback="Blue Snowball",
               silent_s=60.0, signal_kw=None):
        """One or more steady-state-ON polls with every device call mocked.

        Returns (watcher, set_default_capture mock, spoken sentences, logs)."""
        spoken = []
        sw = _mic_sw(mic_fallback=mic_fallback, announce=spoken.append)
        sw.mic_silent_s = silent_s
        sw._prior_capture = prior
        cap = _cur(CORSAIR_MIC, rows) if cap is self._DEFAULT_CAP else cap
        kw = signal_kw or dict(poll_age=1.0, audible_age=None, session_age=120.0)
        with _signal(**kw), \
             mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "capture_override", return_value=override), \
             mock.patch.object(A, "list_render", return_value=rows), \
             mock.patch.object(A, "default_capture", return_value=cap), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=ok) as setc, \
             mock.patch.object(A, "_log") as log:
            for _ in range(ticks):
                sw.tick(was_on=True)
        return sw, setc, spoken, [str(c.args[0]) for c in log.call_args_list]

    # ── the reproduction, and the pin that fails without the fix ────────────
    def test_a_powered_on_headset_that_hears_NOTHING_is_rescued(self):
        """THE OUTAGE THIS FIXES. Steady-state ON, capture default on the
        headset's own microphone, capture loop demonstrably running, and not
        one chunk above the audible floor in 120 s."""
        sw, setc, spoken, _log = self._drive()
        setc.assert_called_once_with(SNOWBALL)
        self.assertEqual(sw.last_mic_label, "mic_silent_rescue")
        self.assertTrue(any("Blue Snowball" in m for m in spoken),
                        f"the rescue was not announced: {spoken}")

    def test_the_SAME_sample_did_nothing_before_the_watchdog(self):
        """The regression pin. mic_silent_s <= 0 restores the pre-2026-09-05
        steady-state ON branch EXACTLY — fire once, never look again — and on
        byte-identical inputs it moves nothing and says nothing. If the
        watchdog is ever removed, defaulted off, or wired to a knob that does
        not reach it, the test above fails and this one still passes, which is
        what tells you which of the two happened."""
        sw, setc, spoken, _log = self._drive(silent_s=0.0, ticks=5)
        setc.assert_not_called()
        self.assertIsNone(sw.last_mic_label)
        self.assertEqual(spoken, [])

    # ── everything that must NOT move a device ──────────────────────────────
    def test_silence_while_NOBODY_IS_LISTENING_never_switches(self):
        """record_speech does not hold the microphone while JARVIS is speaking
        or thinking. Reading that gap as a dead device would yank a WORKING
        microphone in the middle of every long question."""
        sw, setc, spoken, _log = self._drive(
            ticks=5,
            signal_kw=dict(poll_age=120.0, audible_age=200.0, session_age=900.0))
        setc.assert_not_called()
        self.assertEqual(spoken, [])

    def test_a_microphone_that_IS_hearing_is_left_alone_and_silent(self):
        sw, setc, spoken, _log = self._drive(
            ticks=5,
            signal_kw=dict(poll_age=0.4, audible_age=2.0, session_age=900.0))
        setc.assert_not_called()
        self.assertEqual(spoken, [])

    def test_the_input_half_being_OFF_disarms_the_watchdog_too(self):
        """The feature is opt-in, and the new ON-side branch must be inside
        that opt-in like everything else. With follow_mic off, a measured-dead
        headset microphone is still not this daemon's business."""
        spoken = []
        sw = _mic_sw(follow_mic=False, announce=spoken.append)
        with _signal(poll_age=1.0, audible_age=None, session_age=300.0), \
             mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "capture_override", return_value=None), \
             mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
             mock.patch.object(A, "default_capture",
                               return_value=_cur(CORSAIR_MIC)) as cap, \
             mock.patch.object(A, "default_render_id", return_value=CORSAIR_EAR), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture") as setc:
            for _ in range(5):
                sw.tick(was_on=True)
        setc.assert_not_called()
        cap.assert_not_called()
        self.assertIsNone(sw.last_mic_label)
        self.assertEqual(spoken, [])

    def test_it_never_takes_away_a_microphone_the_owner_CHOSE(self):
        """The narrowness that makes this safe: the only device the ON side
        will ever move away from is the configured headset's own. A silent desk
        mic is record_speech's own alert to raise (it speaks at 30 s, before
        this watchdog's 60 s) — two voices on one fault is worse than one."""
        sw, setc, spoken, _log = self._drive(cap=_cur(SNOWBALL), ticks=5)
        setc.assert_not_called()
        self.assertEqual(spoken, [])

    def test_an_OVERRIDE_holds_because_the_silence_is_about_another_device(self):
        """capture_override() is the gate, not decoration. With
        PREFERRED_INPUT_DEVICES set — the owner's LIVE config, measured
        2026-09-05 — the monolith does not record from the Windows default at
        all, so a silent capture says NOTHING about the headset's microphone
        and moving the default would not change what JARVIS hears."""
        why = ("I take the first connected match from my preferred microphone "
               "list instead (Blue Snowball, eMeet C960, CORSAIR VOID)")
        sw, setc, spoken, _log = self._drive(override=why, ticks=4)
        setc.assert_not_called()
        said = " ".join(spoken)
        self.assertTrue(spoken, "an override must still be reported, not hidden")
        self.assertIn("preferred microphone list", said)
        self.assertNotIn("I may not be able to hear you", said,
                         "asserting deafness from a stuck default that JARVIS "
                         "does not follow would be an unverified claim")

    def test_an_unreadable_default_holds_instead_of_guessing(self):
        sw, setc, spoken, _log = self._drive(cap=None, ticks=3)
        setc.assert_not_called()
        self.assertTrue(any("will not tell me" in m for m in spoken), spoken)

    def test_an_empty_enumeration_never_moves_the_microphone(self):
        sw, setc, spoken, _log = self._drive(rows=[], cap=_cur(CORSAIR_MIC,
                                                              ROWS_LIVE), ticks=3)
        setc.assert_not_called()
        self.assertTrue(any("came back empty" in m for m in spoken), spoken)

    # ── the exits that leave him deaf must SAY so ───────────────────────────
    def test_nowhere_to_go_leaves_him_put_and_SPEAKS(self):
        """The only state where staying is honest: no other ACTIVE recording
        endpoint exists at all. That is a hardware statement, so it is spoken
        immediately rather than after a confirmation poll."""
        sw, setc, spoken, log = self._drive(rows=ROWS_ONLY_THE_HEADSET_MIC,
                                            cap=_cur(CORSAIR_MIC, ROWS_LIVE))
        setc.assert_not_called()
        self.assertTrue(any("nowhere for me to move" in m for m in spoken), spoken)
        self.assertTrue(any("DEAF" in m for m in log), log)

    def test_a_failed_write_tries_every_rung_then_SAYS_he_may_be_deaf(self):
        sw, setc, spoken, log = self._drive(ok=False)
        self.assertGreaterEqual(len(setc.call_args_list), 2,
                                "a refused write is a reason to try the next "
                                "candidate, not to stop")
        self.assertNotIn(CORSAIR_MIC, [c.args[0] for c in setc.call_args_list],
                         "never 'rescue' him onto the device measured silent")
        self.assertTrue(any("refused the switch" in m for m in spoken), spoken)

    # ── the claims it is and is not allowed to make ─────────────────────────
    def test_it_never_claims_the_NEW_microphone_works(self):
        """The rule this whole module exists for. Moving OFF a device measured
        silent is justified by what was measured about the OLD device; nothing
        here has measured the new one."""
        banned = ("is working", "works now", "i can hear you now",
                  "you are being heard", "microphone is fine")
        for kw in (dict(), dict(ok=False), dict(rows=ROWS_ONLY_THE_HEADSET_MIC,
                                                cap=_cur(CORSAIR_MIC, ROWS_LIVE))):
            _sw, _setc, spoken, _log = self._drive(ticks=3, **kw)
            for m in spoken:
                for phrase in banned:
                    self.assertNotIn(phrase, m.lower(), f"overclaimed: {m!r}")

    def test_MEASURED_recovery_is_the_one_thing_it_may_announce(self):
        """A chunk crossing the audible floor is SIGNAL, not selection, so this
        is the single place in the module entitled to say a microphone is
        picking sound up — and it is said once, not on every poll."""
        spoken = []
        sw = _mic_sw(announce=spoken.append)
        with mock.patch.object(A, "headset_powered", return_value=True), \
             mock.patch.object(A, "capture_override", return_value=None), \
             mock.patch.object(A, "list_render",
                               return_value=ROWS_ONLY_THE_HEADSET_MIC), \
             mock.patch.object(A, "default_capture",
                               return_value=_cur(CORSAIR_MIC, ROWS_LIVE)), \
             mock.patch.object(A, "default_render_id", return_value=SPEAKERS), \
             mock.patch.object(A, "find_active", return_value=(CORSAIR_EAR, "Ear")), \
             mock.patch.object(A, "set_default_render", return_value=True), \
             mock.patch.object(A, "set_default_capture", return_value=True), \
             mock.patch.object(A, "_log"):
            with _signal(poll_age=1.0, audible_age=None, session_age=120.0):
                sw.tick(was_on=True)            # deaf: nowhere to go, speaks
            self.assertTrue(spoken, "the standing fault was never announced")
            with _signal(poll_age=0.3, audible_age=1.0, session_age=900.0):
                sw.tick(was_on=True)            # signal came back
                sw.tick(was_on=True)            # ...and is not re-announced
        recovery = [m for m in spoken if "picking up sound again" in m]
        self.assertEqual(len(recovery), 1, f"said {len(recovery)} times: {spoken}")
        self.assertIsNone(sw._deaf_key, "the alert must re-arm on recovery")

    # ── the anti-stale-duplicate guard ──────────────────────────────────────
    def test_the_two_rescues_choose_the_same_target(self):
        """This repo's most expensive recurring bug is a rule fixed in one copy
        while another rots, and the ON watchdog and the OFF rescue are asking
        the identical question about the identical device — only the trigger
        differs. `_silence_ladder` exists so the policy is written once; this
        fails the build the moment its first choice diverges from what
        `_mic_off_headset` actually writes.

        The cases are the ones where the policy has real branches: a remembered
        mic, a usable fallback, no fallback at all, an unresolvable fallback,
        and a fallback that resolves to the headset itself (which must be
        refused by BOTH, not just by the one that was patched)."""
        for prior, fb in ((None, "Blue Snowball"),
                          (EMEET, "Blue Snowball"),
                          (None, ""),
                          (None, "no such microphone"),
                          (None, "CORSAIR VOID ELITE")):
            with self.subTest(prior=prior, fallback=fb):
                off = _mic_sw(mic_fallback=fb)
                off._prior_capture = prior
                with mock.patch.object(A, "headset_powered", return_value=False), \
                     mock.patch.object(A, "list_render", return_value=ROWS_LIVE), \
                     mock.patch.object(A, "default_capture",
                                       return_value=_cur(CORSAIR_MIC)), \
                     mock.patch.object(A, "set_default_render", return_value=True), \
                     mock.patch.object(A, "set_default_capture",
                                       return_value=True) as setc, \
                     mock.patch.object(A, "_log"):
                    off.tick(was_on=True)
                off_target = (setc.call_args_list[0].args[0]
                              if setc.call_args_list else None)

                on = _mic_sw(mic_fallback=fb)
                on._prior_capture = prior
                with mock.patch.object(A, "_log"):
                    ladder = on._silence_ladder(ROWS_LIVE)
                on_target = ladder[0][0] if ladder else None
                self.assertEqual(on_target, off_target,
                                 "the ON watchdog and the OFF rescue disagree "
                                 "about where to move him")
                self.assertNotIn(CORSAIR_MIC, [d for d, _n in ladder],
                                 "the headset's own microphone is never a target")


if __name__ == "__main__":
    unittest.main()
