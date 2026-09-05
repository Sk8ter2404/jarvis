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


if __name__ == "__main__":
    unittest.main()
