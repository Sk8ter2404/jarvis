"""Tests for skills/audio_devices.py — the "what microphone are you using?" gap.

2026-09-04, owner-reported: he asked "what microphone are you using right now"
and JARVIS emitted [ACTION: system_pulse], answering with CPU/memory/GPU. There
was no action bound to the current input device, so the local brain picked a
wrong-but-plausible one. These tests pin the two halves of the fix:

  1. The actions exist, are registered under every phrasing, and return a
     finished spoken sentence naming the REAL device.
  2. They fail HONESTLY. This repo's defining bug class is claiming a result
     that was never established, so a skill that cannot read the device must
     say so rather than inventing or guessing one.

The names must also be VOICED — an answer that is computed and dropped is the
same defect wearing a different hat — so the speak-set declaration is pinned too.

stdlib unittest + a fake monolith module; nothing here touches a real device.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_REAL_MIC = "Microphone (Blue Snowball )"
_REAL_SPK = "Speakers (Realtek USB2.0 Audio)"


def _fake_bc(mic=_REAL_MIC, spk=_REAL_SPK, friendly=True,
             mic_raises=False, drop_helpers=False):
    """A stand-in for the loaded monolith. `drop_helpers` models the standalone
    case where the module exists but the getters do not."""
    bc = types.ModuleType("bobert_companion")
    if not drop_helpers:
        def _mic():
            if mic_raises:
                raise RuntimeError("device enumeration blew up")
            return mic
        bc.get_current_mic_name = _mic
        bc.get_current_speaker_name = lambda: spk
    if friendly:
        bc._friendly_device_name = (
            lambda raw: "the Blue Snowball" if "Snowball" in raw else raw)
    return bc


class _Base(unittest.TestCase):
    def _load(self, bc=None):
        """Load the skill with a fake monolith in sys.modules and KEEP it there
        for the whole test.

        The patch must outlive this helper: the skill resolves the monolith
        lazily inside each action (`_bc()`), not at import, so a `with` block
        that exits here would leave the actions reading the REAL bobert_companion
        — which is importable on this box and would answer with the machine's
        actual devices. That is exactly how the first draft of these tests
        "passed" against live hardware instead of the fixture.
        """
        patcher = mock.patch.dict(
            sys.modules,
            {"bobert_companion": bc if bc is not None else _fake_bc()})
        patcher.start()
        self.addCleanup(patcher.stop)
        return load_skill_isolated("audio_devices")


class RegistrationTests(_Base):
    def test_every_phrasing_is_registered(self):
        _mod, actions = self._load()
        for name in ("current_mic", "what_microphone", "which_microphone",
                     "what_mic", "current_speaker", "what_speakers",
                     "which_speakers", "audio_devices", "what_audio_devices"):
            self.assertIn(name, actions, name)

    def test_the_answers_are_declared_speakable(self):
        # A computed-then-dropped answer is the same defect as no answer.
        mod, actions = self._load()
        declared = set(getattr(mod, "SPEAK_VERBATIM_ACTIONS", ()))
        for name in ("current_mic", "what_microphone", "current_speaker",
                     "audio_devices"):
            self.assertIn(name, declared, f"{name} would never be spoken")

    def test_declared_names_all_exist(self):
        # A declaration naming an unregistered action is a stale duplicate.
        mod, actions = self._load()
        for name in getattr(mod, "SPEAK_VERBATIM_ACTIONS", ()):
            self.assertIn(name, actions, f"declared but not registered: {name}")


class ReachabilityTests(_Base):
    """Registering a handler does NOT teach the brain the name exists.

    2026-09-04, found by audit hours after this skill was first written: all
    nine actions below were registered and correctly declared speakable, and
    the reported bug was still 100%% unfixed — because none of the names
    appeared in core/prompts.py, so the LLM had no token to emit and kept
    routing "what microphone are you using" to system_pulse. The handler was
    fixed and the routing was left exactly as broken as it was found.

    The monolith offers two routes: a tracked skill documents its actions in
    core/prompts.py (bobert_companion.py notes that gitignored personal skills
    use a module-level PROMPT_EXAMPLES instead, to keep PII out of the tracked
    file). This skill is tracked, so prompts.py is the right home — and this
    test is what stops the two halves drifting apart again.
    """

    def test_every_registered_action_is_routable(self):
        from core import prompts
        src = _prompt_source(prompts)
        _mod, actions = self._load()
        missing = sorted(n for n in actions if n not in src)
        self.assertEqual(
            missing, [],
            "registered but undocumented — the model can never emit these, so "
            f"they are dead on arrival: {missing}")

    def test_the_reported_phrasing_is_steered_away_from_system_pulse(self):
        # The actual owner-reported failure. Documenting the names is necessary
        # but not sufficient: system_pulse is a plausible neighbour, so the
        # prompt must say outright that it is the wrong answer here.
        from core import prompts
        src = _prompt_source(prompts).lower()
        self.assertIn("what microphone are you using", src)
        self.assertRegex(src, r"not\s+.{0,20}system_pulse")


def _prompt_source(prompts_mod):
    """Every string constant in core/prompts.py, concatenated.

    Reads the module's own attributes rather than the file, so a name that is
    only ever written into a dynamically-built prompt still counts as routable
    — the question is what reaches the model, not what a grep finds."""
    parts = []
    for name in dir(prompts_mod):
        if name.startswith("__"):
            continue
        val = getattr(prompts_mod, name)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, (list, tuple)):
            parts.extend(v for v in val if isinstance(v, str))
    return "\n".join(parts)


class AnswerTests(_Base):
    def test_mic_answer_names_the_real_device(self):
        _mod, actions = self._load()
        out = actions["what_microphone"]("")
        self.assertIn("Blue Snowball", out)
        self.assertTrue(out.rstrip().endswith("sir."), out)

    def test_speaker_answer_names_the_real_device(self):
        _mod, actions = self._load()
        self.assertIn("Realtek", actions["what_speakers"](""))

    def test_combined_answer_names_both(self):
        _mod, actions = self._load()
        out = actions["audio_devices"]("")
        self.assertIn("Blue Snowball", out)
        self.assertIn("Realtek", out)

    def test_raw_name_is_used_when_no_friendly_helper(self):
        _mod, actions = self._load(_fake_bc(friendly=False))
        self.assertIn(_REAL_MIC.strip(), actions["current_mic"](""))


class HonestFailureTests(_Base):
    """The point of the skill: never claim a device it did not read."""

    def _assert_honest(self, out):
        self.assertIn("couldn't determine", out)
        for invented in ("Blue Snowball", "Realtek", "CORSAIR"):
            self.assertNotIn(invented, out)

    def test_missing_helpers_say_so(self):
        _mod, actions = self._load(_fake_bc(drop_helpers=True))
        self._assert_honest(actions["current_mic"](""))

    def test_getter_raising_says_so(self):
        _mod, actions = self._load(_fake_bc(mic_raises=True))
        self._assert_honest(actions["current_mic"](""))

    def test_unknown_device_is_not_read_back_as_a_name(self):
        # get_current_mic_name() returns the literal "unknown" on failure —
        # speaking "I'm listening on unknown, sir" would be worse than useless.
        _mod, actions = self._load(_fake_bc(mic="unknown"))
        self._assert_honest(actions["current_mic"](""))

    def test_empty_device_name_says_so(self):
        _mod, actions = self._load(_fake_bc(mic="   "))
        self._assert_honest(actions["current_mic"](""))

    def test_no_monolith_at_all_says_so(self):
        patcher = mock.patch.dict(sys.modules, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("bobert_companion", None)
        imp = mock.patch("importlib.import_module",
                         side_effect=ImportError("no monolith"))
        imp.start()
        self.addCleanup(imp.stop)
        _mod, actions = load_skill_isolated("audio_devices")
        self._assert_honest(actions["current_mic"](""))

    def test_partial_read_is_reported_as_partial(self):
        bc = _fake_bc()
        bc.get_current_speaker_name = lambda: "unknown"
        _mod, actions = self._load(bc)
        out = actions["audio_devices"]("")
        self.assertIn("Blue Snowball", out)
        self.assertIn("couldn't determine", out)


if __name__ == "__main__":
    unittest.main()
