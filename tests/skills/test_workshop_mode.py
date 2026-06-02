"""Logic tests for skills/workshop_mode.py.

Auto-detects CAD/slicer windows and toggles a low-key "workshop mode" (scaled
TTS + a one-sentence prompt addendum). We test:

  • _find_cad_window — longest-hint-first substring match over window titles,
    with pygetwindow mocked
  • _friendly_app_name — hint → speakable display name
  • _maybe_get_print_status_line — reads bambu_monitor state, only speaks while
    RUNNING
  • _enter_workshop_mode / _exit_workshop_mode — flips the active flag, installs
    + restores the play_with_lipsync wrapper and the _system_prompt addendum on a
    *fake* bobert_companion (never the real monolith)
  • the workshop_status action

register() starts a poll thread (neutered by the harness) and prints; no real
windows or audio are touched. _enqueue_speech is patched so nothing is spoken.
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
    """Swallow stdout while driving _enter/_exit directly.

    NOTE: skills/workshop_mode.py:219 prints a status line containing a U+2192
    arrow ('→'); on the user's cp1252 Windows console that print raises
    UnicodeEncodeError. The harness only redirects stdout during skill *load*,
    so when we call _enter_workshop_mode() ourselves we redirect here to keep
    the (separately reported) latent print bug from masking the logic under
    test. We assert on return values / state, never on the print.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _fake_window(title):
    w = mock.MagicMock()
    w.title = title
    return w


def _make_fake_bc():
    """A minimal fake bobert_companion with the two attributes workshop_mode
    mutates: play_with_lipsync (callable) and _system_prompt (str)."""
    bc = types.ModuleType("bobert_companion")
    bc.play_with_lipsync = lambda audio, sr: ("PLAYED", audio, sr)
    bc._system_prompt = "BASE PROMPT"
    return bc


class WorkshopFindWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")

    def _patch_windows(self, titles):
        fake_gw = types.ModuleType("pygetwindow")
        fake_gw.getAllWindows = lambda: [_fake_window(t) for t in titles]
        return mock.patch.dict(sys.modules, {"pygetwindow": fake_gw})

    def test_finds_bambu_studio(self):
        with self._patch_windows(["Untitled - Bambu Studio", "Notepad"]):
            hint, title = self.mod._find_cad_window()
        self.assertEqual(hint, "bambu studio")
        self.assertIn("Bambu Studio", title)

    def test_longest_hint_wins(self):
        # "autodesk fusion 360" is checked before a bare "fusion 360".
        with self._patch_windows(["Design - Autodesk Fusion 360"]):
            hint, _ = self.mod._find_cad_window()
        self.assertEqual(hint, "autodesk fusion 360")

    def test_no_cad_window(self):
        with self._patch_windows(["Chrome", "Slack", "Notepad"]):
            self.assertEqual(self.mod._find_cad_window(), (None, None))

    def test_no_pygetwindow_degrades(self):
        with mock.patch.dict(sys.modules, {"pygetwindow": None}):
            # import inside the function raises ImportError → (None, None)
            self.assertEqual(self.mod._find_cad_window(), (None, None))

    def test_friendly_app_names(self):
        self.assertEqual(self.mod._friendly_app_name("bambu studio"),
                         "Bambu Studio")
        self.assertEqual(self.mod._friendly_app_name("fusion360"), "Fusion 360")
        # Unknown hint passes through.
        self.assertEqual(self.mod._friendly_app_name("weirdcad"), "weirdcad")


class WorkshopPrintStatusLineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")

    def _fake_bambu(self, **state):
        import threading
        m = types.ModuleType("skill_bambu_monitor")
        m._state_lock = threading.Lock()
        base = {"last_update": 0.0}
        base.update(state)
        m._state = base
        m._format_minutes = lambda mins: (f"{int(mins)} minutes" if mins else "")
        return m

    def test_none_when_bambu_absent(self):
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": None}):
            self.assertIsNone(self.mod._maybe_get_print_status_line())

    def test_none_when_no_fresh_state(self):
        fake = self._fake_bambu(last_update=0.0, gcode_state="RUNNING")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertIsNone(self.mod._maybe_get_print_status_line())

    def test_none_when_not_running(self):
        import time
        fake = self._fake_bambu(last_update=time.time(), gcode_state="IDLE")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertIsNone(self.mod._maybe_get_print_status_line())

    def test_line_when_running(self):
        import time
        fake = self._fake_bambu(last_update=time.time(), gcode_state="RUNNING",
                                layer_num=47, total_layer=312, mc_remaining=18)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            line = self.mod._maybe_get_print_status_line()
        self.assertIsNotNone(line)
        self.assertIn("layer 47 of 312", line)
        self.assertIn("18 minutes", line)


class WorkshopEnterExitTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")
        # Start from a clean, disengaged mode each test.
        self.mod._workshop_active[0] = False
        self.mod._current_app_title[0] = None
        self.mod._saved_play_with_lipsync[0] = None
        self.mod._saved_system_prompt[0] = None

    def test_enter_sets_flag_and_installs_hooks(self):
        bc = _make_fake_bc()
        original_play = bc.play_with_lipsync
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value=None), _quiet():
            self.mod._enter_workshop_mode("bambu studio", "Doc - Bambu Studio")
        self.assertTrue(self.mod._workshop_active[0])
        # The prompt got the addendum appended.
        self.assertIn(self.mod.WORKSHOP_PROMPT_ADDENDUM, bc._system_prompt)
        self.assertTrue(bc._system_prompt.startswith("BASE PROMPT"))
        # play_with_lipsync was wrapped (no longer the original object).
        self.assertIsNot(bc.play_with_lipsync, original_play)
        self.assertIs(self.mod._saved_play_with_lipsync[0], original_play)
        # Announced entry.
        self.assertTrue(any("Workshop mode engaged" in c.args[0]
                            for c in enq.call_args_list))

    def test_enter_is_idempotent(self):
        bc = _make_fake_bc()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value=None), _quiet():
            self.mod._enter_workshop_mode("bambu studio", "T1")
            prompt_after_first = bc._system_prompt
            self.mod._enter_workshop_mode("bambu studio", "T2")
        # Second enter is a no-op — addendum not doubled.
        self.assertEqual(bc._system_prompt, prompt_after_first)
        self.assertEqual(bc._system_prompt.count(self.mod.WORKSHOP_PROMPT_ADDENDUM), 1)

    def test_scaled_wrapper_multiplies_audio(self):
        bc = _make_fake_bc()
        seen = {}
        bc.play_with_lipsync = lambda audio, sr: seen.update(audio=audio, sr=sr)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value=None), _quiet():
            self.mod._enter_workshop_mode("bambu studio", "T")
            # Call the installed wrapper with a numeric "audio" sample.
            bc.play_with_lipsync(10.0, 44100)
        # 10.0 * WORKSHOP_TTS_SCALE (0.7) = 7.0
        self.assertAlmostEqual(seen["audio"], 10.0 * self.mod.WORKSHOP_TTS_SCALE)
        self.assertEqual(seen["sr"], 44100)

    def test_enter_then_exit_restores(self):
        bc = _make_fake_bc()
        original_play = bc.play_with_lipsync
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value=None), _quiet():
            self.mod._enter_workshop_mode("bambu studio", "T")
            self.mod._exit_workshop_mode()
        self.assertFalse(self.mod._workshop_active[0])
        # Restored to originals.
        self.assertEqual(bc._system_prompt, "BASE PROMPT")
        self.assertIs(bc.play_with_lipsync, original_play)
        self.assertTrue(any("disengaged" in c.args[0].lower()
                            for c in enq.call_args_list))

    def test_exit_when_inactive_is_noop(self):
        bc = _make_fake_bc()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._exit_workshop_mode()
        enq.assert_not_called()


class WorkshopStatusActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")
        self.mod._workshop_active[0] = False
        self.mod._current_app_title[0] = None

    def test_status_inactive(self):
        out = self.actions["workshop_status"]("")
        self.assertIn("not currently engaged", out.lower())

    def test_status_active_with_title(self):
        self.mod._workshop_active[0] = True
        self.mod._current_app_title[0] = "Doc - Bambu Studio"
        out = self.actions["workshop_status"]("")
        self.assertIn("engaged", out.lower())
        self.assertIn("Bambu Studio", out)

    def test_status_active_no_title(self):
        self.mod._workshop_active[0] = True
        self.mod._current_app_title[0] = None
        out = self.actions["workshop_status"]("")
        self.assertIn("engaged", out.lower())


class WorkshopEnqueueSpeechTests(unittest.TestCase):
    """_enqueue_speech routes through bobert_companion.proactive_announce, with
    a console fallback when it's absent / returns falsy / raises."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")

    def test_routes_through_proactive_announce(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(return_value=True)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), _quiet():
            self.mod._enqueue_speech("hello sir")
        bc.proactive_announce.assert_called_once()
        # source kwarg identifies the originating skill.
        self.assertEqual(bc.proactive_announce.call_args.kwargs.get("source"),
                         "workshop")

    def test_falls_back_when_announce_returns_falsy(self):
        # announcer present but reports it didn't queue → console fallback line.
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(return_value=False)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._enqueue_speech("queued?")
        self.assertIn("speech-queue unavailable", buf.getvalue())

    def test_falls_back_when_import_raises(self):
        # import_module raising is caught → the except-branch console line.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=RuntimeError("no module")):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._enqueue_speech("boom")
        self.assertIn("speech-queue write failed", buf.getvalue())


class WorkshopFindWindowDegradeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")

    def test_get_all_windows_raises_degrades(self):
        fake_gw = types.ModuleType("pygetwindow")
        fake_gw.getAllWindows = mock.MagicMock(side_effect=RuntimeError("x11"))
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw}):
            self.assertEqual(self.mod._find_cad_window(), (None, None))


class WorkshopPrintStatusLineBranchTests(unittest.TestCase):
    """The reachable branches inside _maybe_get_print_status_line that the
    sibling happy-path test doesn't hit: mid-print without layer/total, a
    raising _format_minutes, and an outer exception."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")

    def _fake_bambu(self, fmt=None, **state):
        import threading
        m = types.ModuleType("skill_bambu_monitor")
        m._state_lock = threading.Lock()
        base = {"last_update": 0.0}
        base.update(state)
        m._state = base
        m._format_minutes = fmt if fmt is not None else (
            lambda mins: (f"{int(mins)} minutes" if mins else ""))
        return m

    def test_midprint_without_layers(self):
        import time
        fake = self._fake_bambu(last_update=time.time(), gcode_state="RUNNING",
                                layer_num=None, total_layer=None, mc_remaining=0)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            line = self.mod._maybe_get_print_status_line()
        self.assertIsNotNone(line)
        self.assertIn("mid-print", line)
        # remaining=0 → no "remaining" tail.
        self.assertNotIn("remaining", line)

    def test_format_minutes_raises_is_swallowed(self):
        import time

        def _boom(_mins):
            raise ValueError("bad minutes")

        fake = self._fake_bambu(fmt=_boom, last_update=time.time(),
                                gcode_state="RUNNING", layer_num=5,
                                total_layer=50, mc_remaining=42)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            line = self.mod._maybe_get_print_status_line()
        # The formatter blew up → rem_str stays "" → still returns a line.
        self.assertIsNotNone(line)
        self.assertIn("layer 5 of 50", line)
        self.assertNotIn("remaining", line)

    def test_outer_exception_returns_none(self):
        # A _state_lock that isn't a context manager makes `with state_lock:`
        # raise inside the outer try → None.
        import time
        fake = types.ModuleType("skill_bambu_monitor")
        fake._state_lock = object()  # not a context manager
        fake._state = {"last_update": time.time(), "gcode_state": "RUNNING"}
        fake._format_minutes = lambda m: ""
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertIsNone(self.mod._maybe_get_print_status_line())


class WorkshopEnterExitBranchTests(unittest.TestCase):
    """Defensive / secondary branches of enter/exit not covered by the happy
    path: the audio-scale wrapper swallowing a bad audio object, a hook-install
    failure, the print-status offer firing, and a restore failure on exit."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("workshop_mode")
        self.mod._workshop_active[0] = False
        self.mod._current_app_title[0] = None
        self.mod._saved_play_with_lipsync[0] = None
        self.mod._saved_system_prompt[0] = None

    def test_scaled_wrapper_swallows_bad_audio(self):
        bc = _make_fake_bc()
        seen = {}
        bc.play_with_lipsync = lambda audio, sr: seen.update(audio=audio, sr=sr)
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value=None), _quiet():
            self.mod._enter_workshop_mode("bambu studio", "T")

            class _BadAudio:
                def __mul__(self, _other):
                    raise TypeError("can't scale")

            bad = _BadAudio()
            # The wrapper catches the multiply error and passes the original
            # object straight through.
            bc.play_with_lipsync(bad, 44100)
        self.assertIs(seen["audio"], bad)

    def test_enter_hook_install_failure_is_logged(self):
        # bobert_companion whose play_with_lipsync attribute access raises →
        # the install try/except prints a failure but enter still proceeds.
        class _ExplodingBC(types.ModuleType):
            @property
            def play_with_lipsync(self):
                raise RuntimeError("attr boom")

        bc = _ExplodingBC("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value=None):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._enter_workshop_mode("bambu studio", "T")
        self.assertIn("failed to install hooks", buf.getvalue())
        # Despite the hook failure, the mode still engaged + announced.
        self.assertTrue(self.mod._workshop_active[0])
        self.assertTrue(any("engaged" in c.args[0].lower()
                            for c in enq.call_args_list))

    def test_enter_offers_print_status_when_running(self):
        bc = _make_fake_bc()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value="By the way, mid-print, sir."), \
             _quiet():
            self.mod._enter_workshop_mode("bambu studio", "T")
        # Both the engaged line and the print-status offer were queued.
        spoken = [c.args[0] for c in enq.call_args_list]
        self.assertTrue(any("engaged" in m.lower() for m in spoken))
        self.assertTrue(any("mid-print" in m for m in spoken))

    def test_exit_restore_failure_is_logged(self):
        # Engage on a healthy bc, then swap in a bc whose attribute *assignment*
        # raises so the restore path hits its except branch.
        bc = _make_fake_bc()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_maybe_get_print_status_line",
                               return_value=None), _quiet():
            self.mod._enter_workshop_mode("bambu studio", "T")

        class _ReadOnlyBC(types.ModuleType):
            @property
            def play_with_lipsync(self):
                return None

            @play_with_lipsync.setter
            def play_with_lipsync(self, _v):
                raise RuntimeError("read only")

        ro = _ReadOnlyBC("bobert_companion")
        ro._system_prompt = "BASE PROMPT"
        with mock.patch.dict(sys.modules, {"bobert_companion": ro}), \
             mock.patch.object(self.mod, "_enqueue_speech"):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.mod._exit_workshop_mode()
        self.assertIn("failed to restore hooks", buf.getvalue())
        self.assertFalse(self.mod._workshop_active[0])


if __name__ == "__main__":
    unittest.main()
