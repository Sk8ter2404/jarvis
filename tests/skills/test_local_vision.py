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
import sys
import types
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

    # ── _bobert resolution (2026-07-11 sweep crash) ───────────────────────
    def test_bobert_skips_foreign_main(self):
        # In any host that isn't the monolith (harness, sweep, driver),
        # __main__ exists but has no take_screenshot — _bobert must fall
        # through to the imported bobert_companion module instead of
        # returning the host's own module (the old `get("__main__") or ...`
        # short-circuited on __main__ ALWAYS, so unguarded
        # b.take_screenshot(...) call sites crashed with AttributeError).
        import sys, types
        fake_main = types.ModuleType("__main__")          # no take_screenshot
        fake_bc = types.ModuleType("bobert_companion")
        fake_bc.take_screenshot = lambda **kw: b"png"
        with mock.patch.dict(sys.modules, {"__main__": fake_main,
                                           "bobert_companion": fake_bc}):
            self.assertIs(self.mod._bobert(), fake_bc)

    def test_bobert_prefers_monolith_main(self):
        # Live process: the monolith IS __main__ — it must win.
        import sys, types
        fake_main = types.ModuleType("__main__")
        fake_main.take_screenshot = lambda **kw: b"png"
        with mock.patch.dict(sys.modules, {"__main__": fake_main}):
            self.assertIs(self.mod._bobert(), fake_main)

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

    def test_click_not_found_returns_degradation_when_vlm_down(self):
        # _find_click_target_local returns None AND the VLM is actually down →
        # surface the degradation message (covers line 285's `return msg`).
        bc = _fake_bc(ollama_alive=False)
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_find_click_target_local", return_value=None):
            out = self.actions["local_click_target_by_description"]("the play button")
        self.assertIn("Ollama isn't running", out)


# ── module-resolver helpers (_bobert / _take_screenshot / _call_local_vision)
class ResolverHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("local_vision")

    def test_bobert_returns_none_when_neither_loaded(self):
        # Remove both __main__ and bobert_companion → _bobert() yields None.
        with mock.patch.dict(sys.modules, {}, clear=False):
            saved_main = sys.modules.pop("__main__", None)
            saved_bc = sys.modules.pop("bobert_companion", None)
            try:
                self.assertIsNone(self.mod._bobert())
            finally:
                if saved_main is not None:
                    sys.modules["__main__"] = saved_main
                if saved_bc is not None:
                    sys.modules["bobert_companion"] = saved_bc

    def test_bobert_prefers_monolith_looking_module(self):
        # 2026-07-11: resolution is by CAPABILITY, not name order. A bare
        # __main__ (a harness/driver host, no take_screenshot) must NOT win
        # over the imported monolith — the old name-order preference made
        # unguarded b.take_screenshot(...) call sites crash in any non-live
        # host. When neither candidate looks like the monolith, fall back to
        # bobert_companion (the named import beats an arbitrary host script).
        fake_main = types.ModuleType("__main__")
        fake_bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"__main__": fake_main,
                                           "bobert_companion": fake_bc}):
            self.assertIs(self.mod._bobert(), fake_bc)
        # Live shape: the monolith IS __main__ (has take_screenshot) — wins.
        fake_main.take_screenshot = lambda **kw: b"png"
        with mock.patch.dict(sys.modules, {"__main__": fake_main,
                                           "bobert_companion": fake_bc}):
            self.assertIs(self.mod._bobert(), fake_main)
        # With __main__ removed entirely, falls through to bobert_companion.
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            saved = sys.modules.pop("__main__", None)
            try:
                self.assertIs(self.mod._bobert(), fake_bc)
            finally:
                if saved is not None:
                    sys.modules["__main__"] = saved

    def test_take_screenshot_delegates_to_bobert(self):
        bc = types.SimpleNamespace(
            take_screenshot=mock.MagicMock(return_value=b"PNG"))
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._take_screenshot("left"), b"PNG")
        bc.take_screenshot.assert_called_once_with(monitor="left")

    def test_take_screenshot_none_when_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertIsNone(self.mod._take_screenshot("left"))

    def test_take_screenshot_none_when_bobert_lacks_method(self):
        bc = types.SimpleNamespace()   # no take_screenshot attr
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIsNone(self.mod._take_screenshot(None))

    def test_take_all_monitors_delegates(self):
        bc = types.SimpleNamespace(
            take_all_monitor_screenshots=mock.MagicMock(
                return_value={"left": b"a"}))
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._take_all_monitor_screenshots(), {"left": b"a"})

    def test_take_all_monitors_empty_when_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertEqual(self.mod._take_all_monitor_screenshots(), {})

    def test_call_local_vision_delegates(self):
        bc = types.SimpleNamespace(
            _call_local_vision=mock.MagicMock(return_value="answer"))
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.mod._call_local_vision("q", [b"png"], max_tokens=42)
        self.assertEqual(out, "answer")
        bc._call_local_vision.assert_called_once_with("q", [b"png"], max_tokens=42)

    def test_call_local_vision_none_when_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertIsNone(self.mod._call_local_vision("q", [b"png"]))

    def test_call_local_vision_none_when_bobert_lacks_method(self):
        bc = types.SimpleNamespace()
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIsNone(self.mod._call_local_vision("q", []))

    def test_parse_monitor_prefix_delegates_to_bobert(self):
        # When bc HAS _parse_monitor_prefix, the skill defers to it.
        bc = types.SimpleNamespace(
            _parse_monitor_prefix=lambda t: ("right", "stripped"))
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._parse_monitor_prefix("monitor:right| x"),
                             ("right", "stripped"))


# ── describe: single-monitor success + _push_screen_context hooks ─────────
class DescribePushContextTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("local_vision")

    def test_describe_single_monitor_success_pushes_context(self):
        bc = _fake_bc()
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_screenshot", return_value=b"png"), \
             mock.patch.object(self.mod, "_call_local_vision",
                               return_value="a terminal window"):
            out = self.actions["local_describe_screen"]("monitor:left| what is this")
        self.assertIn("[local-vision]", out)
        self.assertIn("terminal", out)
        bc._push_screen_context.assert_called_once()
        # Args: (monitor, question, result, {monitor: png})
        args = bc._push_screen_context.call_args[0]
        self.assertEqual(args[0], "left")
        self.assertEqual(args[3], {"left": b"png"})

    def test_describe_single_monitor_capture_none(self):
        bc = _fake_bc()
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_screenshot", return_value=None):
            out = self.actions["local_describe_screen"]("monitor:left| hi")
        self.assertIn("could not capture screen", out)

    def test_describe_single_monitor_push_context_error_swallowed(self):
        # Single-monitor path: _push_screen_context raises → swallowed, answer
        # still returned (covers the 161-162 except branch).
        bc = _fake_bc()
        bc._push_screen_context.side_effect = RuntimeError("ctx boom")
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_screenshot", return_value=b"png"), \
             mock.patch.object(self.mod, "_call_local_vision", return_value="desc"):
            out = self.actions["local_describe_screen"]("monitor:left| what")
        self.assertIn("[local-vision]", out)
        self.assertIn("desc", out)

    def test_describe_all_monitors_vlm_returns_empty(self):
        # All-monitors path: VLM yields falsy → degradation message (line 139).
        bc = _fake_bc(ollama_alive=False)
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_all_monitor_screenshots",
                               return_value={"left": b"p1", "right": b"p2"}), \
             mock.patch.object(self.mod, "_call_local_vision", return_value=None):
            out = self.actions["local_describe_screen"]("what's open")
        self.assertIn("Ollama isn't running", out)

    def test_describe_all_monitors_push_context_error_swallowed(self):
        bc = _fake_bc()
        bc._push_screen_context.side_effect = RuntimeError("ctx boom")
        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_all_monitor_screenshots",
                               return_value={"left": b"p1"}), \
             mock.patch.object(self.mod, "_call_local_vision",
                               return_value="desc"):
            out = self.actions["local_describe_screen"]("what")
        # The push error is swallowed; the answer is still returned.
        self.assertIn("[local-vision]", out)
        self.assertIn("desc", out)

    def test_describe_default_question_when_blank(self):
        # Empty question → the skill substitutes its default describe prompt.
        bc = _fake_bc()
        captured = {}

        def _call(prompt, pngs, max_tokens=600):
            captured["prompt"] = prompt
            return "ok"

        with _quiet(), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_take_all_monitor_screenshots",
                               return_value={"main": b"p"}), \
             mock.patch.object(self.mod, "_call_local_vision", side_effect=_call):
            self.actions["local_describe_screen"]("")
        self.assertIn("Describe in detail", captured["prompt"])


# ── _find_click_target_local two-pass coordinate refinement ───────────────
class _FakeImg:
    """Minimal PIL.Image stand-in: carries a size, supports crop()/save()."""
    def __init__(self, size):
        self.size = size

    def crop(self, box):
        left, top, right, bottom = box
        return _FakeImg((right - left, bottom - top))

    def save(self, buf, format=None):   # noqa: A002 — mirror PIL signature
        buf.write(b"CROP_PNG")


def _fake_pil(open_sizes):
    """Build a fake `PIL` package whose Image.open returns images of the given
    sizes in call order (last repeats). ``PIL`` and ``PIL.Image`` are both
    registered so ``from PIL import Image`` resolves to our stub."""
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    seq = list(open_sizes)
    state = {"i": 0}

    def _open(_buf):
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return _FakeImg(seq[i])

    image_mod.open = _open
    pil.Image = image_mod
    return pil, image_mod


class FindClickTargetLocalTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("local_vision")

    @contextlib.contextmanager
    def _pil(self, open_sizes):
        pil, image_mod = _fake_pil(open_sizes)
        with mock.patch.dict(sys.modules, {"PIL": pil, "PIL.Image": image_mod}):
            yield

    def test_find_pil_missing_returns_none(self):
        with mock.patch.dict(sys.modules, {"PIL": None}):
            self.assertIsNone(
                self.mod._find_click_target_local("btn", monitor=None))

    def test_find_no_bobert_returns_none(self):
        with self._pil([(100, 100)]), \
             mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertIsNone(
                self.mod._find_click_target_local("btn", monitor=None))

    def test_find_first_screenshot_none(self):
        bc = mock.MagicMock()
        bc.take_screenshot.return_value = None
        with self._pil([(100, 100)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertIsNone(
                self.mod._find_click_target_local("btn", monitor=None))

    def test_find_pass1_not_found_returns_none(self):
        bc = mock.MagicMock()
        bc.take_screenshot.return_value = b"png"
        with self._pil([(800, 600)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords", return_value=None):
            self.assertIsNone(
                self.mod._find_click_target_local("btn", monitor=None))

    def test_find_full_png_none_uses_low_res(self):
        # Second take_screenshot (full res) returns None → fall back to img1's
        # size; no refinement crop (full_w == w1). Returns the pass-1 coords.
        bc = mock.MagicMock()
        bc.take_screenshot.side_effect = [b"low", None]
        bc.MONITORS = {}
        with self._pil([(800, 600)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords", return_value=(100, 200)):
            coords = self.mod._find_click_target_local("btn", monitor=None)
        self.assertEqual(coords, (100, 200))

    def test_find_two_pass_refines_and_offsets_monitor(self):
        # Low-res 800x600, full-res 1600x1200 (2x) → pass1 (100,200) scales to
        # (200,400); pass2 in the crop refines to crop-local (10,20) → absolute
        # (left+10, top+20). Monitor offset (50,60) is then added.
        bc = mock.MagicMock()
        bc.take_screenshot.side_effect = [b"low", b"full"]
        bc.MONITORS = {"left": (50, 60, 1600, 1200)}
        # pass1 returns (100,200); pass2 (in crop) returns (10,20). _quiet()
        # swallows the 🔍 refinement print (cp1252 console can't encode it).
        with _quiet(), self._pil([(800, 600), (1600, 1200)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords",
                               side_effect=[(100, 200), (10, 20)]):
            coords = self.mod._find_click_target_local("play", monitor="left")
        # full coords: 100*2=200, 200*2=400. CROP=500 → left=max(0,200-250)=0,
        #   top=max(0,400-250)=150. refined = (0+10, 150+20) = (10, 170).
        # + monitor offset (50, 60) → (60, 230).
        self.assertEqual(coords, (60, 230))

    def test_find_two_pass_pass2_none_keeps_scaled(self):
        # When pass2 returns None, the scaled pass-1 coords are kept.
        bc = mock.MagicMock()
        bc.take_screenshot.side_effect = [b"low", b"full"]
        bc.MONITORS = {}
        with _quiet(), self._pil([(800, 600), (1600, 1200)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords",
                               side_effect=[(100, 200), None]):
            coords = self.mod._find_click_target_local("x", monitor=None)
        self.assertEqual(coords, (200, 400))   # scaled, unrefined

    def test_find_monitor_not_in_MONITORS_no_offset(self):
        bc = mock.MagicMock()
        bc.take_screenshot.side_effect = [b"low", None]
        bc.MONITORS = {"right": (0, 0, 100, 100)}   # 'left' absent
        with self._pil([(400, 300)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords", return_value=(40, 30)):
            coords = self.mod._find_click_target_local("x", monitor="left")
        self.assertEqual(coords, (40, 30))   # no offset applied

    def test_find_monitor_dpi_scale_applied(self):
        # Display at 200% DPI: native capture 3200x2400, logical monitor
        # extent 1600x1200 → native offsets must be halved before the
        # logical origin is added. The old code added raw native offsets
        # (would return (160, 570) here) — off by 2x on every scaled
        # display, which is why tiny targets like bookmark-bar items missed.
        bc = mock.MagicMock()
        bc.take_screenshot.side_effect = [b"low", b"full"]
        bc.MONITORS = {"left": (0, 0, 1600, 1200)}
        with _quiet(), self._pil([(800, 600), (3200, 2400)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords",
                               side_effect=[(100, 200), (10, 20)]):
            coords = self.mod._find_click_target_local("btn", monitor="left")
        # pass1 (100,200) ×4 → (400,800). CROP=500 → crop origin (150,550);
        # pass2 (10,20) → refined native (160,570). ×0.5 DPI → (80,285).
        self.assertEqual(coords, (80, 285))

    def test_find_virtual_origin_added_for_no_monitor(self):
        # monitor=None on a negative-origin virtual desktop: the virtual
        # origin must be added or the click lands on the wrong monitor.
        bc = mock.MagicMock()
        bc.take_screenshot.side_effect = [b"low", b"full"]
        bc.MONITORS = {}
        bc._virtual_screen_bounds.return_value = (-2560, -1440, 7680, 2880)
        with _quiet(), self._pil([(768, 288), (7680, 2880)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords",
                               side_effect=[(76, 28), None]):
            coords = self.mod._find_click_target_local("btn", monitor=None)
        # pass1 (76,28) ×10 → (760,280); pass2 None keeps it; virtual
        # origin (-2560,-1440) added at 1:1 DPI → (-1800,-1160).
        self.assertEqual(coords, (-1800, -1160))

    def test_find_full_png_none_uses_native_size(self):
        # Pass-2 capture failure must NOT treat the downscaled pass-1 size
        # as native: with a native size available, pass-1 coords scale up
        # and the logical translate scales them back.
        bc = mock.MagicMock()
        bc.take_screenshot.side_effect = [b"low", None]
        bc.MONITORS = {"left": (100, 0, 1600, 1200)}
        bc._native_capture_size.return_value = (3200, 2400)
        with _quiet(), self._pil([(800, 600)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords",
                               return_value=(100, 200)):
            coords = self.mod._find_click_target_local("btn", monitor="left")
        # (100,200) ×4 to native (400,800), ×0.5 to logical + origin (100,0)
        # → (300, 400).
        self.assertEqual(coords, (300, 400))

    def test_click_happy_path_through_real_find(self):
        # End-to-end click: drive the REAL _find_click_target_local (not mocked)
        # so the action→finder→click chain is covered together.
        bc = _fake_bc()
        bc.take_screenshot.side_effect = [b"low", None]
        bc.MONITORS = {}
        with _quiet(), self._pil([(800, 600)]), \
             mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_local_query_coords", return_value=(11, 22)):
            out = self.actions["local_click_target_by_description"]("the OK button")
        self.assertIn("clicked 'the OK button' at (11, 22)", out)
        bc.ui_click.assert_called_once_with(11, 22)


# ── register() ───────────────────────────────────────────────────────────
class RegisterTests(unittest.TestCase):
    def test_register_wires_both_actions(self):
        mod, actions = load_skill_isolated("local_vision")
        self.assertIs(actions["local_describe_screen"], mod.local_describe_screen)
        self.assertIs(actions["local_click_target_by_description"],
                      mod.local_click_target_by_description)


if __name__ == "__main__":
    unittest.main()
