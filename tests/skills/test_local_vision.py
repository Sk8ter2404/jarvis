"""Logic tests for skills/local_vision.py.

local_vision mirrors the cloud see_screen / click actions but forces queries
through a local Ollama VLM. Tests cover:
  • the monitor:NAME| prefix parser (local-regex fallback path),
  • the graceful-degradation message ladder in _missing_local_vision_msg
    (no bobert / disabled / no model / Ollama down / model not pulled),
  • coordinate parsing from the VLM reply (incl. NOT_FOUND + out-of-bounds),
  • local_describe_screen happy + capture-failure paths,
  • local_click_target_by_description: empty arg, self-close refusal, and the
    success path with the VLM + click mocked.

A fake bobert_companion is injected into sys.modules so the skill's _bobert()
resolves to a controllable stub — no real screenshots, Ollama, or clicks.
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


@contextlib.contextmanager
def _quiet():
    """Swallow stdout — local_describe_screen prints emoji status lines
    (📸/👁) that raise UnicodeEncodeError on this cp1252 console. We assert
    on the returned string, not the prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _real_parse_monitor_prefix(text):
    """Mirror the skill's own regex fallback so a MagicMock bc doesn't return
    a non-tuple from b._parse_monitor_prefix and break the unpack."""
    import re
    m = re.match(r"^\s*monitor:([A-Za-z0-9_-]+)\s*\|\s*(.*)$", text or "")
    if m:
        return m.group(1).lower(), m.group(2)
    return None, text or ""


def _fake_bc(**attrs):
    """A bobert stub. By default local vision is fully healthy so the
    happy paths work; override attrs to exercise the degradation ladder."""
    bc = mock.MagicMock()
    bc.LOCAL_VISION_FALLBACK = attrs.get("LOCAL_VISION_FALLBACK", True)
    bc.LOCAL_VISION_MODEL = attrs.get("LOCAL_VISION_MODEL", "qwen2.5vl:7b")
    bc._ollama_alive.return_value = attrs.get("ollama_alive", True)
    bc._ollama_has_model.return_value = attrs.get("has_model", True)
    bc.MONITORS = attrs.get("MONITORS", {"left": (0, 0, 1920, 1080)})
    bc._is_self_close_attempt.return_value = attrs.get("self_close", False)
    # The skill's _parse_monitor_prefix delegates to bc when bc has the
    # attribute; provide a real implementation so it returns a (mon, rest)
    # tuple instead of a MagicMock.
    bc._parse_monitor_prefix.side_effect = _real_parse_monitor_prefix
    return bc


class LocalVisionHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("local_vision")

    # ── _parse_monitor_prefix (local fallback when bc has no parser) ──────
    def test_parse_monitor_prefix_present(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            mon, rest = self.mod._parse_monitor_prefix("monitor:LEFT| what is this")
        self.assertEqual(mon, "left")
        self.assertEqual(rest, "what is this")

    def test_parse_monitor_prefix_absent(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            mon, rest = self.mod._parse_monitor_prefix("just describe the screen")
        self.assertIsNone(mon)
        self.assertEqual(rest, "just describe the screen")

    # ── _missing_local_vision_msg ladder ─────────────────────────────────
    def test_missing_msg_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertIn("bobert_companion isn't loaded", self.mod._missing_local_vision_msg())

    def test_missing_msg_disabled(self):
        bc = _fake_bc(LOCAL_VISION_FALLBACK=False)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIn("disabled in config", self.mod._missing_local_vision_msg())

    def test_missing_msg_no_model(self):
        bc = _fake_bc(LOCAL_VISION_MODEL="")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIn("no model configured", self.mod._missing_local_vision_msg())

    def test_missing_msg_ollama_down(self):
        bc = _fake_bc(ollama_alive=False)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIn("Ollama isn't running", self.mod._missing_local_vision_msg())

    def test_missing_msg_model_not_pulled(self):
        bc = _fake_bc(has_model=False)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            msg = self.mod._missing_local_vision_msg()
        self.assertIn("hasn't finished", msg)
        self.assertIn("qwen2.5vl:7b", msg)

    def test_missing_msg_generic_call_failure(self):
        bc = _fake_bc()  # all healthy → the "call failed" tail
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIn("failed", self.mod._missing_local_vision_msg())

    # ── _local_query_coords ──────────────────────────────────────────────
    def test_query_coords_parses_xy(self):
        with mock.patch.object(self.mod, "_call_local_vision", return_value="432, 718"):
            self.assertEqual(self.mod._local_query_coords("btn", b"png", 1000, 1000), (432, 718))

    def test_query_coords_not_found(self):
        with mock.patch.object(self.mod, "_call_local_vision", return_value="NOT_FOUND"):
            self.assertIsNone(self.mod._local_query_coords("btn", b"png", 1000, 1000))

    def test_query_coords_out_of_bounds_rejected(self):
        with mock.patch.object(self.mod, "_call_local_vision", return_value="5000,5000"):
            self.assertIsNone(self.mod._local_query_coords("btn", b"png", 100, 100))

    def test_query_coords_unparseable(self):
        with mock.patch.object(self.mod, "_call_local_vision", return_value="dunno"):
            self.assertIsNone(self.mod._local_query_coords("btn", b"png", 100, 100))


class LocalVisionActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("local_vision")

    # ── local_describe_screen ────────────────────────────────────────────
    def test_describe_all_monitors_happy(self):
        bc = _fake_bc()
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_all_monitor_screenshots",
                               return_value={"left": b"png1", "right": b"png2"}), \
             mock.patch.object(self.mod, "_call_local_vision",
                               return_value="A code editor on the left."):
            out = self.actions["local_describe_screen"]("what's open")
        self.assertIn("[local-vision]", out)
        self.assertIn("code editor", out)

    def test_describe_all_monitors_capture_fail(self):
        bc = _fake_bc()
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_all_monitor_screenshots", return_value={}):
            out = self.actions["local_describe_screen"]("hi")
        self.assertIn("could not capture", out)

    def test_describe_single_monitor_vlm_unavailable(self):
        bc = _fake_bc(ollama_alive=False)
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_screenshot", return_value=b"png"), \
             mock.patch.object(self.mod, "_call_local_vision", return_value=None):
            out = self.actions["local_describe_screen"]("monitor:left| what is this")
        # Falls back to the degradation message (Ollama down).
        self.assertIn("Ollama isn't running", out)

    # ── local_click_target_by_description ────────────────────────────────
    def test_click_requires_description(self):
        bc = _fake_bc()
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIn("describe what I should click",
                          self.actions["local_click_target_by_description"](""))

    def test_click_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            out = self.actions["local_click_target_by_description"]("the play button")
        self.assertIn("not available", out)

    def test_click_refuses_self_close(self):
        bc = _fake_bc(self_close=True)
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["local_click_target_by_description"]("close powershell")
        self.assertTrue(out.startswith("REFUSED:"))
        self.assertIn("kill my session", out)

    def test_click_happy_path(self):
        bc = _fake_bc()
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_find_click_target_local", return_value=(120, 240)):
            out = self.actions["local_click_target_by_description"]("the play button")
        self.assertIn("clicked 'the play button' at (120, 240)", out)
        bc.ui_click.assert_called_once_with(120, 240)

    def test_click_target_not_found(self):
        bc = _fake_bc()  # healthy VLM → "could not find" rather than degradation msg
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_find_click_target_local", return_value=None):
            out = self.actions["local_click_target_by_description"]("a unicorn icon")
        self.assertIn("could not find 'a unicorn icon'", out)

    def test_click_surfaces_click_failure(self):
        bc = _fake_bc()
        bc.ui_click.side_effect = RuntimeError("failsafe: cursor in corner")
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_find_click_target_local", return_value=(10, 20)):
            out = self.actions["local_click_target_by_description"]("ok button")
        self.assertIn("click failed", out)
        self.assertIn("failsafe", out)


if __name__ == "__main__":
    unittest.main()
