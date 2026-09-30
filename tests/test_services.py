"""Unit tests for ``core/services.py`` — the typed JarvisServices seam (M2 Phase 1).

CI-SAFE: pure stdlib ``unittest`` + a fake ``skill_utils`` dict. The monolith is
never imported (these run on the bare-Linux light-deps runner too), so every
backing capability is a plain Python callable recording its calls.

What's pinned here:
  * ``from_skill_utils`` returns a JarvisServices that delegates each method to
    the correspondingly-named lambda, with the right positional/keyword args.
  * Missing-key degradation per method (no-op side-effects; ``""`` / ``None`` /
    ``False`` sentinels for value-returning ones) — exactly the monolith's own
    fallbacks.
  * A present-but-raising backing callable propagates (a wired capability that
    throws is a real error, surfaced just as ``skill_utils[k](...)`` would).
  * The structural ``JarvisServicesProtocol`` shape.
"""
from __future__ import annotations

import unittest
from unittest import mock

from core.services import JarvisServices, JarvisServicesProtocol


def _recording_utils():
    """A full fake ``skill_utils`` — every key a MagicMock with a sensible
    return value — so delegation can be asserted per key."""
    keys = (
        "ask_vision", "take_screenshot", "find_click_target", "click",
        "type_text", "press_key", "hotkey", "scroll", "sleep", "launch_app",
        "open_url", "write_hud_state", "make_promise",
        "register_promise_condition", "fulfil_promise",
    )
    utils = {k: mock.MagicMock(name=k) for k in keys}
    utils["ask_vision"].return_value = "an answer"
    utils["take_screenshot"].return_value = b"PNG"
    utils["find_click_target"].return_value = (12, 34)
    utils["launch_app"].return_value = "launched notepad"
    utils["open_url"].return_value = "opened"
    utils["make_promise"].return_value = 7
    utils["fulfil_promise"].return_value = True
    return utils


# ──────────────────────────────────────────────────────────────────────────
#  Construction + shape
# ──────────────────────────────────────────────────────────────────────────
class ConstructionTests(unittest.TestCase):
    def test_from_skill_utils_returns_jarvis_services(self):
        svc = JarvisServices.from_skill_utils({})
        self.assertIsInstance(svc, JarvisServices)

    def test_holds_dict_by_reference_not_copy(self):
        # The facade must stay in lockstep with the live dict (single source of
        # truth) — a lambda swapped in the dict after construction is seen.
        d = {}
        svc = JarvisServices.from_skill_utils(d)
        calls = []
        d["open_url"] = lambda u: calls.append(u) or "ok"
        self.assertEqual(svc.open_url("https://example.test"), "ok")
        self.assertEqual(calls, ["https://example.test"])

    def test_non_dict_degrades_to_empty(self):
        # Defensive: a non-dict (None / unexpected type) must not raise at
        # construction or on use — every method no-ops / returns its sentinel.
        svc = JarvisServices.from_skill_utils(None)  # type: ignore[arg-type]
        self.assertIsNone(svc.take_screenshot())
        self.assertEqual(svc.ask_vision("q"), "")
        self.assertFalse(svc.fulfil_promise(1))
        svc.write_hud_state(x=1)  # must not raise

    def test_satisfies_protocol(self):
        svc = JarvisServices.from_skill_utils({})
        self.assertIsInstance(svc, JarvisServicesProtocol)


# ──────────────────────────────────────────────────────────────────────────
#  Delegation — each method reaches the right key with the right args
# ──────────────────────────────────────────────────────────────────────────
class DelegationTests(unittest.TestCase):
    def setUp(self):
        self.utils = _recording_utils()
        self.svc = JarvisServices.from_skill_utils(self.utils)

    def test_ask_vision_question_only(self):
        # png_bytes defaulting to None must NOT be forwarded, so we hit the
        # backing ask_vision's own default (matches the monolith lambda).
        out = self.svc.ask_vision("what is on screen?")
        self.assertEqual(out, "an answer")
        self.utils["ask_vision"].assert_called_once_with("what is on screen?")

    def test_ask_vision_with_png(self):
        out = self.svc.ask_vision("describe", b"\x89PNG")
        self.assertEqual(out, "an answer")
        self.utils["ask_vision"].assert_called_once_with("describe", b"\x89PNG")

    def test_take_screenshot(self):
        self.assertEqual(self.svc.take_screenshot(), b"PNG")
        self.utils["take_screenshot"].assert_called_once_with()

    def test_find_click_target(self):
        self.assertEqual(self.svc.find_click_target("the OK button"), (12, 34))
        self.utils["find_click_target"].assert_called_once_with("the OK button")

    def test_click_default_button(self):
        self.svc.click(100, 200)
        self.utils["click"].assert_called_once_with(100, 200, "left")

    def test_click_explicit_button(self):
        self.svc.click(1, 2, "right")
        self.utils["click"].assert_called_once_with(1, 2, "right")

    def test_type_text(self):
        self.svc.type_text("hello world")
        self.utils["type_text"].assert_called_once_with("hello world")

    def test_press_key(self):
        self.svc.press_key("enter")
        self.utils["press_key"].assert_called_once_with("enter")

    def test_hotkey_varargs(self):
        self.svc.hotkey("ctrl", "shift", "p")
        self.utils["hotkey"].assert_called_once_with("ctrl", "shift", "p")

    def test_scroll(self):
        self.svc.scroll(-3)
        self.utils["scroll"].assert_called_once_with(-3)

    def test_sleep(self):
        self.svc.sleep(0.25)
        self.utils["sleep"].assert_called_once_with(0.25)

    def test_launch_app(self):
        self.assertEqual(self.svc.launch_app("notepad"), "launched notepad")
        self.utils["launch_app"].assert_called_once_with("notepad")

    def test_open_url(self):
        self.assertEqual(self.svc.open_url("https://a.test"), "opened")
        self.utils["open_url"].assert_called_once_with("https://a.test")

    def test_write_hud_state_passes_kwargs(self):
        self.svc.write_hud_state(pulse_strip="GPU 60C", updated_at=123)
        self.utils["write_hud_state"].assert_called_once_with(
            pulse_strip="GPU 60C", updated_at=123)

    def test_make_promise_forwards_message_condition_and_kwargs(self):
        pid = self.svc.make_promise(
            "Sir, the print is done.", "print_complete",
            params={"job": 1}, source="bambu")
        self.assertEqual(pid, 7)
        self.utils["make_promise"].assert_called_once_with(
            "Sir, the print is done.", "print_complete",
            params={"job": 1}, source="bambu")

    def test_register_promise_condition_forwards_all(self):
        pred = lambda d: True  # noqa: E731 - tiny inline predicate for the test
        self.svc.register_promise_condition("is_hot", pred)
        self.utils["register_promise_condition"].assert_called_once_with(
            "is_hot", pred)

    def test_fulfil_promise_returns_bool(self):
        self.assertIs(self.svc.fulfil_promise(99), True)
        self.utils["fulfil_promise"].assert_called_once_with(99)

    def test_fulfil_promise_coerces_truthy_to_bool(self):
        # Backing returns a truthy non-bool; the facade normalises to a real
        # bool so callers can rely on the annotated -> bool contract.
        self.utils["fulfil_promise"].return_value = "yes"
        self.assertIs(self.svc.fulfil_promise(1), True)


# ──────────────────────────────────────────────────────────────────────────
#  Missing-key degradation — partial dict
# ──────────────────────────────────────────────────────────────────────────
class MissingKeyDegradationTests(unittest.TestCase):
    """Every method must tolerate its key being absent from the dict and
    degrade to the monolith's own fallback rather than raising KeyError."""

    def setUp(self):
        self.svc = JarvisServices.from_skill_utils({})  # nothing wired

    def test_value_methods_return_safe_sentinels(self):
        self.assertEqual(self.svc.ask_vision("q"), "")
        self.assertIsNone(self.svc.take_screenshot())
        self.assertIsNone(self.svc.find_click_target("x"))
        self.assertIsNone(self.svc.launch_app("notepad"))
        self.assertIsNone(self.svc.open_url("https://a.test"))
        self.assertIsNone(self.svc.make_promise("m", "c"))
        self.assertIsNone(self.svc.register_promise_condition("n", lambda d: True))
        self.assertIs(self.svc.fulfil_promise(1), False)

    def test_side_effect_methods_are_silent_noops(self):
        # None of these may raise when unwired.
        self.svc.click(1, 2)
        self.svc.type_text("x")
        self.svc.press_key("enter")
        self.svc.hotkey("ctrl", "c")
        self.svc.scroll(1)
        self.svc.sleep(0)
        self.svc.write_hud_state(a=1, b=2)

    def test_non_callable_value_treated_as_absent(self):
        # A malformed dict whose value isn't callable (e.g. {"click": None})
        # must degrade, not blow up trying to call None.
        svc = JarvisServices.from_skill_utils(
            {"click": None, "fulfil_promise": "not-callable",
             "take_screenshot": 123})
        svc.click(1, 2)  # no-op, no TypeError
        self.assertIs(svc.fulfil_promise(1), False)
        self.assertIsNone(svc.take_screenshot())


# ──────────────────────────────────────────────────────────────────────────
#  A wired-but-raising callable propagates (don't mask real errors)
# ──────────────────────────────────────────────────────────────────────────
class WiredCallableRaisesTests(unittest.TestCase):
    def test_exception_from_backing_callable_propagates(self):
        boom = mock.MagicMock(side_effect=RuntimeError("vision down"))
        svc = JarvisServices.from_skill_utils({"ask_vision": boom})
        with self.assertRaises(RuntimeError):
            svc.ask_vision("q")

    def test_write_hud_state_exception_propagates(self):
        # The facade itself does not swallow — a consumer that wants
        # best-effort wraps its own call (the migrated status_panel skill does).
        boom = mock.MagicMock(side_effect=RuntimeError("hud locked"))
        svc = JarvisServices.from_skill_utils({"write_hud_state": boom})
        with self.assertRaises(RuntimeError):
            svc.write_hud_state(x=1)



class DialogueWrapperTests(unittest.TestCase):
    """The device-dialogue wrappers (2026-09-29) delegate to the dict and
    degrade to "no dialogue" when unwired."""

    def test_delegates(self):
        cm = object()
        u = {
            "dialogue_ready": mock.MagicMock(return_value=""),
            "dialogue_session": mock.MagicMock(return_value=cm),
            "speak_line": mock.MagicMock(return_value="spoken"),
            "listen_for_stop": mock.MagicMock(return_value="cap"),
            "local_complete": mock.MagicMock(return_value="{}"),
            "register_self_voiced": mock.MagicMock(return_value=True),
            "is_self_voiced": mock.MagicMock(return_value=True),
        }
        svc = JarvisServices.from_skill_utils(u)
        self.assertEqual(svc.dialogue_ready(), "")
        self.assertIs(svc.dialogue_session("desk device"), cm)
        u["dialogue_session"].assert_called_with("desk device")
        svc.dialogue_session("desk device", max_s=20)
        u["dialogue_session"].assert_called_with("desk device", max_s=20)
        self.assertEqual(svc.speak_line("Hi.", "wry"), "spoken")
        u["speak_line"].assert_called_with("Hi.", "wry")
        until = lambda: True  # noqa: E731
        self.assertEqual(svc.listen_for_stop(until, beat_s=0.6), "cap")
        u["listen_for_stop"].assert_called_with(until, beat_s=0.6)
        self.assertEqual(svc.local_complete("s", [], max_tokens=9), "{}")
        u["local_complete"].assert_called_with("s", [], max_tokens=9)
        self.assertTrue(svc.register_self_voiced("x"))
        self.assertTrue(svc.is_self_voiced("x"))

    def test_utterance_route_is_forwarded_and_degrades(self):
        route = lambda text: None  # noqa: E731
        u = {"register_utterance_route": mock.MagicMock(return_value=True)}
        svc = JarvisServices.from_skill_utils(u)
        self.assertTrue(svc.register_utterance_route(route, "desk device"))
        u["register_utterance_route"].assert_called_with(route, "desk device")
        self.assertFalse(JarvisServices.from_skill_utils({})
                         .register_utterance_route(route))
        self.assertTrue(hasattr(JarvisServicesProtocol,
                                "register_utterance_route"))

    def test_unwired_degrades(self):
        svc = JarvisServices.from_skill_utils({})
        self.assertEqual(svc.dialogue_ready(), "disabled")
        with self.assertRaises(RuntimeError):
            svc.dialogue_session("desk device")
        self.assertEqual(svc.speak_line("Hi."), "failed")
        self.assertIsNone(svc.listen_for_stop(lambda: True))
        self.assertIsNone(svc.local_complete("s", []))
        self.assertFalse(svc.register_self_voiced("x"))
        self.assertFalse(svc.is_self_voiced("x"))

    def test_protocol_lists_the_dialogue_methods(self):
        for name in ("dialogue_ready", "dialogue_session", "speak_line",
                     "listen_for_stop", "local_complete",
                     "register_self_voiced", "is_self_voiced"):
            self.assertTrue(hasattr(JarvisServicesProtocol, name), name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
