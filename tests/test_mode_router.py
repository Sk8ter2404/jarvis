"""Tests for core.mode_router — conversation mode routing (controlled / smart /
agent) and the follow-up loop depth. followup_loop_depth() is about to gain
complexity-aware gating, so these pin the existing contract first."""
import unittest

import core.mode_router as mr


class _ModePreserving(unittest.TestCase):
    """Snapshot + restore the persisted mode so tests never leave the user's
    conversation_mode.json mutated."""
    def setUp(self):
        self._orig_mode = mr.current_mode()

    def tearDown(self):
        try:
            mr.set_mode(self._orig_mode)
        except Exception:
            pass


class FollowupDepthTests(_ModePreserving):
    def test_smart_returns_default(self):
        mr.set_mode(mr.MODE_SMART)
        self.assertEqual(mr.followup_loop_depth(), 5)
        self.assertEqual(mr.followup_loop_depth(default=3), 3)

    def test_agent_boosts_and_caps(self):
        mr.set_mode(mr.MODE_AGENT)
        self.assertEqual(mr.followup_loop_depth(default=5), 15)   # 3x, capped 15
        self.assertEqual(mr.followup_loop_depth(default=2), 6)    # 3x under cap
        self.assertLessEqual(mr.followup_loop_depth(default=100), 15)

    def test_controlled_returns_default(self):
        mr.set_mode(mr.MODE_CONTROLLED)
        self.assertEqual(mr.followup_loop_depth(default=5), 5)


class ToggleDetectionTests(_ModePreserving):
    def test_non_mode_text_is_none(self):
        self.assertIsNone(mr.maybe_handle_mode_toggle("what's the weather?"))
        self.assertIsNone(mr.maybe_handle_mode_toggle(""))
        self.assertIsNone(mr.maybe_handle_mode_toggle("play some music"))

    def test_status_query(self):
        mr.set_mode(mr.MODE_SMART)
        out = mr.maybe_handle_mode_toggle("what mode are you in?")
        self.assertIsNotNone(out)
        self.assertIn("smart", out.lower())

    def test_switch_to_agent(self):
        mr.set_mode(mr.MODE_SMART)
        out = mr.maybe_handle_mode_toggle("switch to agent mode")
        self.assertIsNotNone(out)
        self.assertEqual(mr.current_mode(), mr.MODE_AGENT)

    def test_switch_with_lead_filler(self):
        mr.set_mode(mr.MODE_SMART)
        out = mr.maybe_handle_mode_toggle("JARVIS, please switch to controlled mode.")
        self.assertIsNotNone(out)
        self.assertEqual(mr.current_mode(), mr.MODE_CONTROLLED)

    def test_already_in_mode(self):
        mr.set_mode(mr.MODE_AGENT)
        out = mr.maybe_handle_mode_toggle("agent mode")
        self.assertIn("already", out.lower())


class ControlledDispatchTests(_ModePreserving):
    def test_returns_none_when_not_controlled(self):
        mr.set_mode(mr.MODE_SMART)
        self.assertIsNone(mr.controlled_dispatch("anything", {}))

    def test_refuses_unknown_in_controlled(self):
        mr.set_mode(mr.MODE_CONTROLLED)
        out = mr.controlled_dispatch("zxcvbnm qwerty nonsense", {})
        self.assertIsInstance(out, str)
        self.assertIn("controlled mode", out.lower())


class AddendumTests(_ModePreserving):
    def test_agent_addendum_present(self):
        mr.set_mode(mr.MODE_AGENT)
        self.assertIn("AGENT MODE", mr.system_prompt_addendum())

    def test_smart_addendum_empty(self):
        mr.set_mode(mr.MODE_SMART)
        self.assertEqual(mr.system_prompt_addendum(), "")


if __name__ == "__main__":
    unittest.main()
