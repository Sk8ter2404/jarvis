"""Tests for core.mode_router — conversation mode routing (controlled / smart /
agent) and the follow-up loop depth. followup_loop_depth() is about to gain
complexity-aware gating, so these pin the existing contract first."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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
        self.assertEqual(mr.followup_loop_depth(default=5), 15)   # 3x under cap
        self.assertEqual(mr.followup_loop_depth(default=8), 24)   # 3x, at cap
        self.assertEqual(mr.followup_loop_depth(default=2), 6)    # 3x under cap
        self.assertLessEqual(mr.followup_loop_depth(default=100), 24)

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


# ─────────────────────────────────────────────────────────────────────────
# Coverage-completion: state persistence, set_mode validation, the toggle
# error branches, and the full controlled_dispatch resolution chain.
# ─────────────────────────────────────────────────────────────────────────

def _fake_step(action="play_music", arg="jazz", confirmation="music queued"):
    """A stand-in for core.dispatcher.ChainStep with the 3 attributes
    controlled_dispatch reads."""
    return types.SimpleNamespace(action=action, arg=arg,
                                 confirmation=confirmation)


class _FakeDispatcher(types.ModuleType):
    """Injectable replacement for core.dispatcher. resolve_and_dispatch and
    match_single_intent are plain attributes the test sets per scenario."""
    def __init__(self, chain_result=None, single_result=None,
                 chain_exc=None, single_exc=None):
        super().__init__("core.dispatcher")
        self._chain_result = chain_result
        self._single_result = single_result
        self._chain_exc = chain_exc
        self._single_exc = single_exc

    def resolve_and_dispatch(self, text, actions):
        if self._chain_exc is not None:
            raise self._chain_exc
        return self._chain_result

    def match_single_intent(self, text, action_keys):
        if self._single_exc is not None:
            raise self._single_exc
        return self._single_result


class _ControlledBase(_ModePreserving):
    def _use_dispatcher(self, disp):
        """Inject `disp` as core.dispatcher for the duration of one test."""
        self.addCleanup(
            lambda old=sys.modules.get("core.dispatcher"):
            sys.modules.__setitem__("core.dispatcher", old) if old is not None
            else sys.modules.pop("core.dispatcher", None))
        sys.modules["core.dispatcher"] = disp


class StatePersistenceTests(_ModePreserving):
    def test_load_state_reads_valid_file(self):
        # Point _STATE_FILE at a temp file holding a valid mode, force a load.
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "conversation_mode.json"
            f.write_text(json.dumps({"mode": "agent"}), encoding="utf-8")
            with mock.patch.object(mr, "_STATE_FILE", f):
                mr._state["mode"] = mr._DEFAULT_MODE
                mr._load_state()
                self.assertEqual(mr._state["mode"], "agent")

    def test_load_state_missing_file_returns(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "does_not_exist.json"
            with mock.patch.object(mr, "_STATE_FILE", f):
                mr._state["mode"] = "smart"
                mr._load_state()       # file absent → early return, unchanged
                self.assertEqual(mr._state["mode"], "smart")

    def test_load_state_corrupt_file_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "conversation_mode.json"
            f.write_text("{ not json", encoding="utf-8")
            with mock.patch.object(mr, "_STATE_FILE", f):
                mr._state["mode"] = "smart"
                mr._load_state()       # corrupt → except → unchanged
                self.assertEqual(mr._state["mode"], "smart")

    def test_load_state_invalid_mode_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "conversation_mode.json"
            f.write_text(json.dumps({"mode": "banana"}), encoding="utf-8")
            with mock.patch.object(mr, "_STATE_FILE", f):
                mr._state["mode"] = "smart"
                mr._load_state()
                self.assertEqual(mr._state["mode"], "smart")  # invalid → ignored

    def test_ensure_data_dir_failure_swallowed(self):
        # Replace _DATA_DIR with a stub whose mkdir raises (can't patch mkdir
        # on a real WindowsPath instance — it's read-only).
        stub = mock.MagicMock()
        stub.mkdir.side_effect = OSError("denied")
        with mock.patch.object(mr, "_DATA_DIR", stub):
            mr._ensure_data_dir()      # must not raise
        stub.mkdir.assert_called_once()

    def test_save_state_uses_atomic_writer(self):
        # Inject a fake core.atomic_io exposing _atomic_write_json; assert it's used.
        recorded = {}
        fake_io = types.ModuleType("core.atomic_io")
        fake_io._atomic_write_json = lambda path, data: recorded.update(
            {"path": path, "data": dict(data)})
        self.addCleanup(
            lambda old=sys.modules.get("core.atomic_io"):
            sys.modules.__setitem__("core.atomic_io", old) if old is not None
            else sys.modules.pop("core.atomic_io", None))
        sys.modules["core.atomic_io"] = fake_io
        with mock.patch.object(mr, "_ensure_data_dir"):
            mr._save_state()
        self.assertIn("path", recorded)
        self.assertEqual(recorded["data"]["mode"], mr._state["mode"])

    def test_save_state_falls_back_to_plain_write(self):
        # atomic_io import raises → fall through to _STATE_FILE.write_text.
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "conversation_mode.json"
            with mock.patch.object(mr, "_STATE_FILE", f), \
                    mock.patch.dict(sys.modules, {"core.atomic_io": None}), \
                    mock.patch.object(mr, "_ensure_data_dir"):
                # core.atomic_io = None → `from core.atomic_io import ...` raises.
                mr._save_state()
            self.assertTrue(f.exists())
            self.assertEqual(json.loads(f.read_text(encoding="utf-8"))["mode"],
                             mr._state["mode"])

    def test_save_state_plain_write_failure_swallowed(self):
        with mock.patch.dict(sys.modules, {"core.atomic_io": None}), \
                mock.patch.object(mr, "_ensure_data_dir"), \
                mock.patch.object(type(mr._STATE_FILE), "write_text",
                                  side_effect=OSError("disk full")):
            mr._save_state()           # must not raise


class SetModeValidationTests(_ModePreserving):
    def test_unknown_mode_raises_valueerror(self):
        with self.assertRaises(ValueError):
            mr.set_mode("turbo")

    def test_blank_mode_raises(self):
        with self.assertRaises(ValueError):
            mr.set_mode("")


class ToggleBranchTests(_ModePreserving):
    def test_filler_only_text_returns_none(self):
        # A pure lead-filler ("please ") strips to "" and then the punctuation
        # trim leaves an empty string → the `if not s: return None` guard (217).
        self.assertIsNone(mr.maybe_handle_mode_toggle("please "))

    def test_punctuation_only_returns_none(self):
        # All-punctuation input → after _PUNCT_TAIL strip, s is empty → 217.
        self.assertIsNone(mr.maybe_handle_mode_toggle("?!"))

    def test_status_query_short_form(self):
        mr.set_mode(mr.MODE_SMART)
        self.assertIn("smart", mr.maybe_handle_mode_toggle("current mode").lower())

    def test_set_mode_failure_during_toggle(self):
        mr.set_mode(mr.MODE_SMART)
        # Force set_mode to raise so the toggle's except branch returns the
        # "Could not switch modes" line.
        with mock.patch.object(mr, "set_mode",
                               side_effect=RuntimeError("disk gone")):
            out = mr.maybe_handle_mode_toggle("switch to agent mode")
        self.assertIn("could not switch modes", out.lower())

    def test_smart_confirmation_line(self):
        mr.set_mode(mr.MODE_AGENT)
        out = mr.maybe_handle_mode_toggle("smart mode")
        self.assertIn("smart mode engaged", out.lower())

    def test_controlled_confirmation_line(self):
        mr.set_mode(mr.MODE_SMART)
        out = mr.maybe_handle_mode_toggle("controlled mode")
        self.assertIn("controlled mode engaged", out.lower())


class ControlledDispatchChainTests(_ControlledBase):
    def setUp(self):
        super().setUp()
        mr.set_mode(mr.MODE_CONTROLLED)

    def test_empty_text_refuses(self):
        out = mr.controlled_dispatch("   ", {})
        self.assertIn("controlled mode", out.lower())

    def test_dispatcher_import_failure_refuses(self):
        # core.dispatcher = None in sys.modules → the `from core.dispatcher
        # import ...` raises → refusal line.
        self._use_dispatcher(None)
        out = mr.controlled_dispatch("play music", {"play_music": lambda a: "ok"})
        self.assertIn("controlled mode", out.lower())

    def test_chain_result_returned(self):
        self._use_dispatcher(_FakeDispatcher(chain_result="two things, sir: a, b."))
        out = mr.controlled_dispatch("do a and b", {"a": lambda x: "1"})
        self.assertEqual(out, "two things, sir: a, b.")

    def test_chain_exception_falls_to_single(self):
        # chain raises → swallowed (chain_reply=None) → single match runs.
        disp = _FakeDispatcher(chain_exc=RuntimeError("chain boom"),
                               single_result=_fake_step("play_music", "jazz"))
        self._use_dispatcher(disp)
        actions = {"play_music": lambda arg: f"playing {arg}"}
        out = mr.controlled_dispatch("play jazz", actions)
        self.assertEqual(out, "playing jazz")

    def test_single_match_none_refuses(self):
        self._use_dispatcher(_FakeDispatcher(chain_result=None, single_result=None))
        out = mr.controlled_dispatch("unknown command", {"x": lambda a: "y"})
        self.assertIn("controlled mode", out.lower())

    def test_single_match_exception_refuses(self):
        disp = _FakeDispatcher(chain_result=None,
                               single_exc=RuntimeError("match boom"))
        self._use_dispatcher(disp)
        out = mr.controlled_dispatch("play jazz", {"play_music": lambda a: "ok"})
        self.assertIn("controlled mode", out.lower())

    def test_matched_action_not_registered_refuses(self):
        # single_match returns a step whose action isn't in the actions dict
        # (registration race) → refusal.
        disp = _FakeDispatcher(chain_result=None,
                               single_result=_fake_step("ghost_action", ""))
        self._use_dispatcher(disp)
        out = mr.controlled_dispatch("do ghost", {"play_music": lambda a: "ok"})
        self.assertIn("controlled mode", out.lower())

    def test_action_executes_and_returns_string(self):
        disp = _FakeDispatcher(chain_result=None,
                               single_result=_fake_step("play_music", "rock"))
        self._use_dispatcher(disp)
        out = mr.controlled_dispatch("play rock",
                                     {"play_music": lambda arg: f"now: {arg}"})
        self.assertEqual(out, "now: rock")

    def test_action_raises_returns_failure_line(self):
        def boom(arg):
            raise ValueError("kaboom")
        disp = _FakeDispatcher(chain_result=None,
                               single_result=_fake_step("play_music", "x"))
        self._use_dispatcher(disp)
        out = mr.controlled_dispatch("play x", {"play_music": boom})
        self.assertIn("that action failed", out.lower())
        self.assertIn("valueerror", out.lower())

    def test_action_empty_result_uses_confirmation_fallback(self):
        # Action returns "" → fall back to the resolver's canned confirmation.
        disp = _FakeDispatcher(
            chain_result=None,
            single_result=_fake_step("play_music", "x", confirmation="music queued"))
        self._use_dispatcher(disp)
        out = mr.controlled_dispatch("play x", {"play_music": lambda a: ""})
        self.assertEqual(out, "Music queued, sir.")

    def test_action_none_result_uses_confirmation_fallback(self):
        disp = _FakeDispatcher(
            chain_result=None,
            single_result=_fake_step("play_music", "x", confirmation="done"))
        self._use_dispatcher(disp)
        out = mr.controlled_dispatch("play x", {"play_music": lambda a: None})
        self.assertEqual(out, "Done, sir.")


# ─────────────────────────────────────────────────────────────────────────
# Stale-duplicate invariant (2026-07-21 audit): the lead-filler stripper
# existed as two drifting copies — this module's fixed version (comma wake
# variants + loop-until-stable) and core.dispatcher's stale one (neither),
# which made Controlled mode refuse "JARVIS, take a screenshot". There must
# be exactly ONE implementation, shared from core.lead_fillers.
# ─────────────────────────────────────────────────────────────────────────

class LeadFillerSingleImplementationTests(unittest.TestCase):
    def test_one_shared_strip_function(self):
        # Function-object IDENTITY — stronger than any source grep. If either
        # module grows a private def again, this fails immediately.
        import core.dispatcher as disp
        import core.lead_fillers as lf
        self.assertIs(mr._strip_lead_filler, lf.strip_lead_filler)
        self.assertIs(disp._strip_lead_filler, lf.strip_lead_filler)

    def test_one_shared_filler_table(self):
        import core.dispatcher as disp
        import core.lead_fillers as lf
        self.assertIs(mr._LEAD_FILLERS, lf.LEAD_FILLERS)
        # dispatcher must never regrow its own table; if the name exists at
        # all it must BE the shared tuple, not a fork.
        self.assertIs(getattr(disp, "_LEAD_FILLERS", lf.LEAD_FILLERS),
                      lf.LEAD_FILLERS)

    def test_every_wake_filler_has_comma_sibling(self):
        # The comma-variant rule can never silently regress in the shared
        # copy: every filler ending in "jarvis " needs its ", "-suffixed
        # sibling (Whisper renders the wake word with a comma).
        import core.lead_fillers as lf
        wake = [f for f in lf.LEAD_FILLERS if f.endswith("jarvis ")]
        self.assertTrue(wake, "expected wake-word fillers in LEAD_FILLERS")
        for f in wake:
            self.assertIn(f[:-1] + ", ", lf.LEAD_FILLERS,
                          f"missing comma variant for {f!r}")

    def test_strip_is_loop_until_stable_with_comma_variants(self):
        import core.lead_fillers as lf
        self.assertEqual(lf.strip_lead_filler("JARVIS, please switch to agent mode"),
                         "switch to agent mode")
        self.assertEqual(lf.strip_lead_filler("hey jarvis, could you play jazz"),
                         "play jazz")


if __name__ == "__main__":
    unittest.main()
