"""Tests for skills/headset_status.py — "is my headset on?" answered honestly.

THE BUG THESE PIN DOWN
======================
Asking about the headset had TWO failure modes, and the second is the nastier
one:

  1. No action for the battery question at all, so the local brain emitted a
     wrong-but-plausible neighbour (the 2026-09-04 "what microphone are you
     using" -> [ACTION: system_pulse] report).

  2. Worse: "is my headset on" DID route somewhere — to
     `audio_autoswitch_status`, whose headset reading comes from
     `audio/audio_switch.py::find_active()`. That function documents, as fact,
     that a powered-off wireless headset reads NotPresent/Unplugged. It does
     not. Windows enumerates the DONGLE, which is plugged in either way, so
     both VOID ELITE endpoints read Active with the headset off. The question
     was answered confidently by a sensor that structurally cannot tell.

So these tests cover three halves of one fix:

  * the actions exist and are registered under every phrasing;
  * the answers are DECLARED SPEAKABLE (an answer computed and dropped is the
    same defect wearing a different hat);
  * the names are ROUTABLE — documented in core/prompts.py, and specifically
    steered AWAY from audio_autoswitch_status. Registering a handler does not
    teach the brain the name exists; that lesson cost the audio_devices skill a
    round of "fixed" that fixed nothing.

AND THE POINT OF THE WHOLE FEATURE: the state is THREE-valued. The dongle went
silent for 105 consecutive seconds while the owner was turning the headset ON.
"unknown" spoken as "off" is the same lie as the old detector, from a better
sensor. `UnknownIsNotOffTests` exists for exactly that and nothing else.

Every test installs a FAKE `audio.void_link` in sys.modules for the whole test.
Nothing here opens a HID handle, touches the dongle, or reads the real headset
— which matters, because the real module imports and answers fine on this box,
so a test that forgot to patch would "pass" against live hardware instead of
its fixture.

stdlib unittest + mock, matching tests/skills/test_audio_devices.py.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

# Imported, not re-implemented: this repo's #1 bug class is the stale duplicate
# — one copy of a rule fixed while the others rot. If that helper moves, this
# import fails loudly instead of drifting.
from tests.skills.test_audio_devices import _prompt_source

# A real captured ON battery reading: 0x58 & 0x7F = 88, measured 2026-09-04 on
# the live dongle, continuing the recorded drain curve 96 -> 94 -> 92 -> 90.
_REAL_BATTERY = 88


def _fake_vl(on=True, battery=_REAL_BATTERY, on_raises=False,
             battery_raises=False, drop_is_on=False, drop_battery=False):
    """A stand-in for audio.void_link.

    `drop_battery` models `battery_percent` not existing — it is a convenience
    on top of the required contract, so the skill must not depend on it.
    """
    vl = types.ModuleType("audio.void_link")
    vl.LINK_ON, vl.LINK_OFF, vl.LINK_UNKNOWN = "on", "off", "unknown"

    if not drop_is_on:
        def _is_headset_on():
            if on_raises:
                raise RuntimeError("hid.dll went away mid-read")
            return on
        vl.is_headset_on = _is_headset_on

    if not drop_battery:
        def _battery_percent():
            if battery_raises:
                raise RuntimeError("battery read blew up")
            return battery
        vl.battery_percent = _battery_percent

    return vl


class _Base(unittest.TestCase):
    """Loads the skill with a fake void_link that OUTLIVES the helper.

    The skill resolves void_link lazily inside `_read()`, not at import, so a
    `with` block that exited here would leave the actions reading the REAL
    audio.void_link — which on this machine answers correctly from the actual
    dongle and would make every assertion below a measurement of the owner's
    headset rather than of this code.
    """

    def _load(self, vl=None):
        patcher = mock.patch.dict(
            sys.modules,
            {"audio.void_link": vl if vl is not None else _fake_vl()})
        patcher.start()
        self.addCleanup(patcher.stop)
        return load_skill_isolated("headset_status")

    def _actions(self, vl=None):
        return self._load(vl)[1]

    def assertSpokenSentence(self, out):
        """Every reply must be a finished, addressed sentence — not a fragment,
        not a status token, not a dict."""
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip(), "empty reply")
        self.assertIn(", sir", out, f"not addressed to him: {out!r}")
        self.assertTrue(out.rstrip().endswith("."), f"unfinished: {out!r}")


# ─────────────────────────────────────────────────────────────────────────
# 1. Registration + the speak-set declaration
# ─────────────────────────────────────────────────────────────────────────
class RegistrationTests(_Base):

    ALL_NAMES = ("headset_status", "is_headset_on", "headset_on",
                 "is_my_headset_on", "headset_battery",
                 "how_much_battery_headset", "headset_battery_level")

    def test_every_phrasing_is_registered(self):
        actions = self._actions()
        for name in self.ALL_NAMES:
            self.assertIn(name, actions, name)

    def test_the_answers_are_declared_speakable(self):
        # Without this the answer is computed, logged and silently dropped.
        mod, actions = self._load()
        declared = set(getattr(mod, "SPEAK_VERBATIM_ACTIONS", ()))
        for name in self.ALL_NAMES:
            self.assertIn(name, declared, f"{name} would never be spoken")

    def test_declared_names_all_exist(self):
        # A declaration naming an unregistered action is a stale duplicate.
        mod, actions = self._load()
        for name in getattr(mod, "SPEAK_VERBATIM_ACTIONS", ()):
            self.assertIn(name, actions, f"declared but not registered: {name}")

    def test_registration_survives_void_link_being_unimportable(self):
        """The dead-on-arrival guard.

        If the skill imported void_link at module scope, a broken or absent
        detector would take all seven action names down with it — and a missing
        action is precisely what makes this brain emit a wrong-but-plausible
        one instead. It must still load and still answer, honestly."""
        sys.modules.pop("audio.void_link", None)
        imp = mock.patch("importlib.import_module",
                         side_effect=ImportError("no void_link"))
        imp.start()
        self.addCleanup(imp.stop)
        _mod, actions = load_skill_isolated("headset_status")
        for name in self.ALL_NAMES:
            self.assertIn(name, actions, name)
        self.assertIn("can't tell", actions["headset_status"](""))


# ─────────────────────────────────────────────────────────────────────────
# 2. Reachability — a registered name the model cannot emit is dead
# ─────────────────────────────────────────────────────────────────────────
class ReachabilityTests(_Base):
    """Registering a handler does NOT teach the brain the name exists.

    Found the hard way hours after skills/audio_devices.py was first written:
    nine actions registered, correctly declared speakable, and the reported bug
    100% unfixed — because none of the names appeared in core/prompts.py, so
    the LLM had no token to emit. This skill is tracked, so core/prompts.py is
    the right home (gitignored personal skills use a module-level
    PROMPT_EXAMPLES instead, to keep names out of tracked source).
    """

    def test_every_registered_action_is_routable(self):
        from core import prompts
        src = _prompt_source(prompts)
        actions = self._actions()
        missing = sorted(n for n in actions if n not in src)
        self.assertEqual(
            missing, [],
            "registered but undocumented — the model can never emit these, so "
            f"they are dead on arrival: {missing}")

    def test_is_my_headset_on_routes_here(self):
        from core import prompts
        src = _prompt_source(prompts).lower()
        self.assertRegex(src, r"'is my headset on'[^\n]*\[action: headset_status\]")

    def test_is_my_headset_on_no_longer_routes_to_the_broken_detector(self):
        """The live mis-route this skill exists to correct.

        core/prompts.py used to say:
            'is my headset on' / 'audio status' -> [ACTION: audio_autoswitch_status]
        and audio_autoswitch_status answers from audio_switch.find_active(),
        which cannot distinguish a powered-off headset from a powered-on one
        because it is looking at the always-present dongle."""
        from core import prompts
        src = _prompt_source(prompts).lower()
        self.assertNotRegex(
            src, r"'is my headset on'[^\n]*audio_autoswitch_status",
            "the headset question is still steered at the endpoint-state "
            "detector, which reads Active whether the headset is on or off")

    def test_the_battery_question_is_routable(self):
        from core import prompts
        src = _prompt_source(prompts).lower()
        self.assertRegex(
            src, r"battery[^\n]*headset[^\n]*\[action: headset_battery\]")

    def test_the_prompt_warns_off_the_plausible_neighbours(self):
        # Documenting the names is necessary but not sufficient: both
        # audio_autoswitch_status and system_pulse are plausible-looking
        # answers to "is my headset on", and one of them used to win.
        from core import prompts
        src = _prompt_source(prompts).lower()
        self.assertRegex(src, r"not\s+.{0,30}audio_autoswitch_status")
        self.assertRegex(src, r"not\s+.{0,30}system_pulse")


# ─────────────────────────────────────────────────────────────────────────
# 3. The three-valued state — the whole point of the feature
# ─────────────────────────────────────────────────────────────────────────
class UnknownIsNotOffTests(_Base):
    """`unknown` must be spoken as genuinely unknown. NEVER as "off".

    Collapsing the two is the exact defect that makes this feature misfire: the
    dongle was silent for 105 consecutive seconds during a pairing handshake,
    i.e. while the owner was turning the headset ON. Reporting "off" there is
    the same wrong answer the old endpoint detector gave, just from a better
    sensor.
    """

    #: every way the truth can be "I don't know"
    UNKNOWN_FAKES = {
        "is_headset_on returned None": _fake_vl(on=None),
        "is_headset_on raised": _fake_vl(on_raises=True),
        "is_headset_on missing": _fake_vl(drop_is_on=True),
        "contract breach (a string)": _fake_vl(on="yes"),
    }

    def _assert_honestly_unknown(self, out):
        """Unknown must READ as unknown.

        NOTE the shape of these assertions, and the two drafts it took to get
        here. Draft one banned the substrings "off" and "is on"; draft two
        banned "headset is on". Both failed nine times on the CORRECT answer —
        "I can't tell whether your headset is on, sir" — which is the honest
        reply naming the very thing it cannot determine. Those were bad tests,
        not a bad product; keeping them would have pushed the wording around to
        satisfy a proxy. What actually matters is that the reply never ASSERTS
        a state, so the ban is on the declarative form ("your headset is on"),
        with the hedged mention ("whether your headset is on") allowed."""
        self.assertSpokenSentence(out)
        low = out.lower()
        self.assertIn("can't tell", low)
        # The load-bearing assertions of this entire file.
        self.assertNotRegex(
            low, r"(?<!whether )your headset is off\b",
            f"unknown was collapsed into OFF — the exact defect: {out!r}")
        self.assertNotRegex(
            low, r"(?<!whether )your headset is on\b",
            f"unknown was asserted as ON: {out!r}")
        self.assertNotRegex(out, r"\d+\s*percent", f"invented a number: {out!r}")

    def test_status_reports_unknown_as_unknown(self):
        for label, vl in self.UNKNOWN_FAKES.items():
            with self.subTest(label):
                self._assert_honestly_unknown(self._actions(vl)["headset_status"](""))

    def test_battery_reports_unknown_as_unknown(self):
        for label, vl in self.UNKNOWN_FAKES.items():
            with self.subTest(label):
                self._assert_honestly_unknown(self._actions(vl)["headset_battery"](""))

    def test_missing_module_is_unknown_not_off(self):
        sys.modules.pop("audio.void_link", None)
        imp = mock.patch("importlib.import_module",
                         side_effect=ImportError("no void_link"))
        imp.start()
        self.addCleanup(imp.stop)
        _mod, actions = load_skill_isolated("headset_status")
        self._assert_honestly_unknown(actions["headset_status"](""))
        self._assert_honestly_unknown(actions["headset_battery"](""))

    def test_unknown_and_off_are_different_sentences(self):
        # A regression guard with teeth: if someone ever "simplifies" the two
        # branches into one, this fails even if the wording changes.
        unknown = self._actions(_fake_vl(on=None))["headset_status"]("")
        off = self._actions(_fake_vl(on=False))["headset_status"]("")
        self.assertNotEqual(unknown, off)


# ─────────────────────────────────────────────────────────────────────────
# 4. The ON / OFF answers
# ─────────────────────────────────────────────────────────────────────────
class AnswerTests(_Base):

    def test_on_with_battery(self):
        out = self._actions()["headset_status"]("")
        self.assertSpokenSentence(out)
        self.assertIn("is on", out)
        self.assertIn(f"{_REAL_BATTERY} percent", out)

    def test_off_says_off(self):
        out = self._actions(_fake_vl(on=False))["headset_status"]("")
        self.assertSpokenSentence(out)
        self.assertIn("is off", out)
        self.assertNotRegex(out, r"\d+\s*percent",
                            "a powered-down headset has no battery reading")

    def test_battery_action_reports_the_number(self):
        out = self._actions()["headset_battery"]("")
        self.assertSpokenSentence(out)
        self.assertIn(f"{_REAL_BATTERY} percent", out)

    def test_battery_action_when_off_does_not_invent_zero(self):
        out = self._actions(_fake_vl(on=False))["headset_battery"]("")
        self.assertSpokenSentence(out)
        self.assertIn("off", out)
        self.assertNotIn("0 percent", out)

    def test_low_battery_is_flagged(self):
        mod, actions = self._load(_fake_vl(battery=7))
        out = actions["headset_battery"]("")
        self.assertIn("7 percent", out)
        self.assertIn("charg", out.lower())
        self.assertLessEqual(7, mod.LOW_BATTERY_PCT)

    def test_healthy_battery_is_not_flagged(self):
        out = self._actions()["headset_battery"]("")
        self.assertNotIn("charg", out.lower())

    def test_every_alias_answers_identically(self):
        actions = self._actions()
        for a, b in (("headset_status", "is_headset_on"),
                     ("headset_status", "headset_on"),
                     ("headset_status", "is_my_headset_on"),
                     ("headset_battery", "how_much_battery_headset"),
                     ("headset_battery", "headset_battery_level")):
            with self.subTest(f"{a} == {b}"):
                self.assertEqual(actions[a](""), actions[b](""))

    def test_actions_tolerate_an_argument(self):
        # The dispatcher passes the [ACTION: name, arg] tail; these take none.
        actions = self._actions()
        for name in ("headset_status", "headset_battery"):
            with self.subTest(name):
                self.assertSpokenSentence(actions[name]("some stray argument"))


# ─────────────────────────────────────────────────────────────────────────
# 5. On, but the battery is unknown — its own case, not a guess
# ─────────────────────────────────────────────────────────────────────────
class OnButBatteryUnknownTests(_Base):

    #: every way the link can be up while the charge is not readable
    FAKES = {
        "battery_percent returned None": _fake_vl(battery=None),
        "battery_percent raised": _fake_vl(battery_raises=True),
        "battery_percent missing": _fake_vl(drop_battery=True),
        # void_link reports None for a zero byte precisely because a powered
        # down headset reads 0x00 there; if a 0 ever reaches this skill it is
        # an absence of a reading, not a reading of zero.
        "battery_percent returned 0": _fake_vl(battery=0),
        "battery out of range (200)": _fake_vl(battery=200),
        "battery negative": _fake_vl(battery=-5),
        "battery not a number": _fake_vl(battery="ninety"),
    }

    def _assert_on_without_a_number(self, out):
        self.assertSpokenSentence(out)
        self.assertIn("is on", out)
        self.assertIn("couldn't read", out)
        self.assertNotRegex(out, r"\d+\s*percent",
                            f"spoke a battery level it never read: {out!r}")

    def test_status_says_on_and_admits_the_battery_is_unreadable(self):
        for label, vl in self.FAKES.items():
            with self.subTest(label):
                self._assert_on_without_a_number(
                    self._actions(vl)["headset_status"](""))

    def test_battery_action_says_on_and_admits_it(self):
        for label, vl in self.FAKES.items():
            with self.subTest(label):
                self._assert_on_without_a_number(
                    self._actions(vl)["headset_battery"](""))

    def test_boundary_values_are_still_spoken(self):
        # 1 and 100 are legitimate; only 0 and out-of-range are refused.
        for pct in (1, 100):
            with self.subTest(pct):
                out = self._actions(_fake_vl(battery=pct))["headset_battery"]("")
                self.assertIn(f"{pct} percent", out)


# ─────────────────────────────────────────────────────────────────────────
# 6. Never raises, whatever the detector does
# ─────────────────────────────────────────────────────────────────────────
class NeverRaisesTests(_Base):

    def test_every_action_returns_a_sentence_for_every_fake(self):
        fakes = [_fake_vl(on=True), _fake_vl(on=False), _fake_vl(on=None),
                 _fake_vl(on_raises=True), _fake_vl(battery_raises=True),
                 _fake_vl(drop_is_on=True), _fake_vl(drop_battery=True),
                 _fake_vl(on="yes"), _fake_vl(battery=object()),
                 types.ModuleType("audio.void_link")]   # empty module
        for i, vl in enumerate(fakes):
            actions = self._actions(vl)
            for name in ("headset_status", "headset_battery"):
                with self.subTest(f"fake{i}.{name}"):
                    self.assertSpokenSentence(actions[name](""))

    def test_a_detector_that_returns_nonsense_never_leaks_a_repr(self):
        out = self._actions(_fake_vl(on=object()))["headset_status"]("")
        self.assertNotIn("object at 0x", out)
        self.assertNotIn("<", out)


# ─────────────────────────────────────────────────────────────────────────
# 7. It reads the real contract, and only that contract
# ─────────────────────────────────────────────────────────────────────────
class ContractTests(unittest.TestCase):

    def test_the_real_void_link_exposes_what_the_skill_calls(self):
        """Against the REAL module (no fake): the names this skill relies on
        exist. Pure attribute check — nothing is called, so no HID handle is
        opened and the owner's headset is not touched."""
        from audio import void_link
        self.assertTrue(callable(void_link.is_headset_on))
        self.assertIn(void_link.LINK_UNKNOWN, ("unknown",))

    def test_the_skill_does_not_fall_back_to_the_endpoint_detector(self):
        """audio_switch.find_active() is one import away and would make
        "unknown" rarer — by answering with the sensor already known to be
        wrong. An honest unknown beats a confident guess, so the source must
        not reference it at all."""
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
            "skills", "headset_status.py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith(("#", "*")))
        # Strip the module docstring, which discusses find_active on purpose.
        body = code.split('"""', 2)[-1]
        self.assertNotIn("find_active", body)
        self.assertNotIn("audio_switch", body)
        self.assertNotIn("pycaw", body)

    def test_low_battery_threshold_matches_the_autoswitch_daemon(self):
        """Not an independent invention: the skill copies
        AudioAutoSwitch.low_pct so the two do not disagree about "low".

        Read out of the SOURCE rather than by constructing AudioAutoSwitch.
        low_pct is assigned in __init__, so instantiating it would couple this
        test to that constructor's signature — which is being actively reworked
        by other work in this tree — and a break there would look like a break
        here. The regex depends only on the attribute existing."""
        import os
        import re as _re
        mod, _ = load_skill_isolated("headset_status")
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, "audio", "audio_switch.py"),
                  encoding="utf-8") as fh:
            m = _re.search(r"low_pct\s*=\s*(\d+)", fh.read())
        self.assertIsNotNone(
            m, "AudioAutoSwitch.low_pct is gone — the anchor this skill's "
               "LOW_BATTERY_PCT was copied from no longer exists; re-derive it "
               "rather than leaving two thresholds drifting apart")
        self.assertEqual(mod.LOW_BATTERY_PCT, int(m.group(1)))


if __name__ == "__main__":
    unittest.main()
