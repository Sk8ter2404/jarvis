"""Logic tests for skills/dnd_focus_mode.py.

Covers the duration parser, the minutes formatter, the rate/registry-free
parts of the focus-mode state machine (enter → extend → exit), the
is_focus_mode_active() helper other skills poll, and the registered actions.

The second half drives the OS / network / monkeypatch side effects the first
half stubs out, so the whole skill — reg.exe Focus Assist, Graph Teams
presence, the critical-announcer-preserving nudge suppression, the prompt
addendum, the expiry thread, and the workshop_mode auto-trigger hook — is
exercised offline and deterministically. No real registry write, Graph call,
network, audio, or unbounded thread ever runs: subprocess/urllib are mocked,
fakes live only in the with-block / setUp and are torn down, and the critical
allow-list (only SUPPRESSED_SKILLS get silenced; bambu/timer-style announcers
are left untouched) is asserted directly.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules (restoring prior
    state — including absence — on exit). For dotted names the leaf is ALSO
    set as an attribute on the already-imported parent package so
    ``from pkg import leaf`` / ``import a.b.c`` resolve to the fake. Mirrors
    the helper in test_self_diagnostic.py so each test file stays self-
    contained and isolated."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


@contextlib.contextmanager
def _neutered_side_effects(mod):
    """Patch every external side effect of enter/exit so only the state
    machine runs."""
    with mock.patch.object(mod, "_set_focus_assist", return_value=True), \
         mock.patch.object(mod, "_set_teams_presence", return_value=True), \
         mock.patch.object(mod, "_install_nudge_suppressors"), \
         mock.patch.object(mod, "_restore_nudge_suppressors"), \
         mock.patch.object(mod, "_apply_prompt_addendum"), \
         mock.patch.object(mod, "_restore_prompt_addendum"), \
         mock.patch.object(mod, "_start_expiry_thread"), \
         mock.patch.object(mod, "_enqueue_speech"):
        yield


class FocusDurationParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dnd_focus_mode")

    def test_parse_units(self):
        p = self.mod._parse_duration_to_seconds
        self.assertEqual(p("90 minutes"), 5400)
        self.assertEqual(p("1 hour 30 min"), 5400)
        self.assertEqual(p("45m"), 2700)
        self.assertEqual(p("2 hours"), 7200)
        self.assertEqual(p("30 seconds"), 30)

    def test_parse_bare_number_is_minutes(self):
        self.assertEqual(self.mod._parse_duration_to_seconds("90"), 5400)

    def test_parse_invalid(self):
        self.assertIsNone(self.mod._parse_duration_to_seconds("soon"))
        self.assertIsNone(self.mod._parse_duration_to_seconds(""))

    def test_parse_matched_unit_but_zero_total(self):
        # A unit matches but sums to 0 → None (found-but-empty branch).
        self.assertIsNone(self.mod._parse_duration_to_seconds("0 minutes"))

    def test_parse_seconds_abbreviations(self):
        self.assertEqual(self.mod._parse_duration_to_seconds("15 sec"), 15)
        self.assertEqual(self.mod._parse_duration_to_seconds("90s"), 90)

    def test_format_minutes(self):
        f = self.mod._format_minutes
        self.assertEqual(f(30), "30 seconds")
        self.assertEqual(f(90), "2 minutes")
        self.assertEqual(f(60), "1 minute")
        self.assertEqual(f(3600), "1 hour")
        self.assertEqual(f(5400), "1 hour 30 minutes")

    def test_format_minutes_multi_hour(self):
        # h != 1 plural branch and the hour+remainder path.
        self.assertEqual(self.mod._format_minutes(7200), "2 hours")
        self.assertEqual(self.mod._format_minutes(2 * 3600 + 30 * 60),
                         "2 hours 30 minutes")


class FocusStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dnd_focus_mode")
        # Ensure a clean inactive baseline (module-global lists).
        self.mod._focus_active[0] = False
        self.mod._focus_ends_at[0] = 0.0
        self.mod._focus_trigger[0] = ""

    def test_initially_inactive(self):
        self.assertFalse(self.mod.is_focus_mode_active())

    def test_enter_sets_active_and_returns_summary(self):
        with _neutered_side_effects(self.mod):
            already, msg = self.mod._enter_focus_mode(5400, trigger="voice")
        self.assertFalse(already)
        self.assertTrue(self.mod.is_focus_mode_active())
        self.assertIn("Holding all non-critical interruptions", msg)
        self.assertIn("1 hour 30 minutes", msg)

    def test_reenter_extends_not_restarts(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
            already, msg = self.mod._enter_focus_mode(3600, trigger="voice")
        self.assertTrue(already)
        self.assertIn("Already in focus mode", msg)
        self.assertIn("extended", msg.lower())

    def test_duration_floor_enforced(self):
        # < 60s is clamped up to 60s minimum.
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(5, trigger="voice")
        remaining = self.mod._focus_ends_at[0] - self.mod._focus_started_at[0]
        self.assertGreaterEqual(remaining, 60)

    def test_duration_cap_enforced(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(99 * 3600, trigger="voice")
        remaining = self.mod._focus_ends_at[0] - self.mod._focus_started_at[0]
        self.assertLessEqual(remaining, self.mod.MAX_DURATION_SECONDS)

    def test_exit_clears_active(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
            msg = self.mod._exit_focus_mode(reason="manual")
        self.assertFalse(self.mod.is_focus_mode_active())
        self.assertIn("disengaged", msg.lower())

    def test_exit_when_inactive_is_graceful(self):
        msg = self.mod._exit_focus_mode(reason="manual")
        self.assertIn("was not active", msg.lower())

    def test_exit_expired_message(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
            msg = self.mod._exit_focus_mode(reason="expired")
        self.assertIn("complete", msg.lower())

    def test_exit_workshop_exit_message(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="workshop")
            msg = self.mod._exit_focus_mode(reason="workshop_exit")
        self.assertIn("Workshop closed", msg)
        self.assertIn("released", msg.lower())

    def test_enter_invokes_side_effects_and_records_results(self):
        # Verify the enter path actually calls each side effect and stores the
        # focus_assist / teams success flags it gets back.
        with mock.patch.object(self.mod, "_set_focus_assist",
                               return_value=True) as fa, \
             mock.patch.object(self.mod, "_set_teams_presence",
                               return_value=True) as tp, \
             mock.patch.object(self.mod, "_install_nudge_suppressors") as ins, \
             mock.patch.object(self.mod, "_apply_prompt_addendum") as apa, \
             mock.patch.object(self.mod, "_start_expiry_thread") as expt, \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._enter_focus_mode(600, trigger="voice")
        fa.assert_called_once_with(enable_dnd=True)
        tp.assert_called_once_with("DoNotDisturb")
        ins.assert_called_once()
        apa.assert_called_once()
        expt.assert_called_once()
        enq.assert_called_once()
        self.assertTrue(self.mod._focus_assist_was_set[0])
        self.assertTrue(self.mod._teams_was_set[0])

    def test_exit_restores_only_when_was_set(self):
        # When focus-assist/teams were NOT set on entry, exit must not try to
        # restore them (guards the `if _focus_assist_was_set[0]` branches).
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
        self.mod._focus_assist_was_set[0] = False
        self.mod._teams_was_set[0] = False
        with mock.patch.object(self.mod, "_set_focus_assist") as fa, \
             mock.patch.object(self.mod, "_set_teams_presence") as tp, \
             mock.patch.object(self.mod, "_restore_nudge_suppressors"), \
             mock.patch.object(self.mod, "_restore_prompt_addendum"), \
             mock.patch.object(self.mod, "_enqueue_speech"):
            self.mod._exit_focus_mode(reason="manual")
        fa.assert_not_called()
        tp.assert_not_called()

    def test_exit_restores_focus_assist_and_teams_when_set(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
        self.mod._focus_assist_was_set[0] = True
        self.mod._teams_was_set[0] = True
        with mock.patch.object(self.mod, "_set_focus_assist",
                               return_value=True) as fa, \
             mock.patch.object(self.mod, "_set_teams_presence",
                               return_value=True) as tp, \
             mock.patch.object(self.mod, "_restore_nudge_suppressors"), \
             mock.patch.object(self.mod, "_restore_prompt_addendum"), \
             mock.patch.object(self.mod, "_enqueue_speech"):
            self.mod._exit_focus_mode(reason="manual")
        fa.assert_called_once_with(enable_dnd=False)
        tp.assert_called_once_with("Available")
        self.assertFalse(self.mod._focus_assist_was_set[0])
        self.assertFalse(self.mod._teams_was_set[0])


class FocusActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dnd_focus_mode")
        self.mod._focus_active[0] = False
        self.mod._focus_ends_at[0] = 0.0
        self.mod._focus_trigger[0] = ""

    def test_focus_mode_action_default_duration(self):
        with _neutered_side_effects(self.mod):
            out = self.actions["focus_mode"]("")   # blank → default 60 min
        self.assertIn("Holding all non-critical", out)
        self.assertTrue(self.mod.is_focus_mode_active())

    def test_focus_mode_action_custom_duration(self):
        with _neutered_side_effects(self.mod):
            out = self.actions["focus_mode"]("90 minutes")
        self.assertIn("1 hour 30 minutes", out)

    def test_focus_mode_action_unparseable_falls_back_to_default(self):
        with _neutered_side_effects(self.mod):
            out = self.actions["focus_mode"]("whenever")
        # Unparseable duration → DEFAULT_DURATION_SECONDS (3600s → "1 hour").
        self.assertIn("1 hour", out)
        self.assertTrue(self.mod.is_focus_mode_active())

    def test_end_focus_mode_action(self):
        with _neutered_side_effects(self.mod):
            self.actions["focus_mode"]("30 minutes")
            out = self.actions["end_focus_mode"]("")
        self.assertIn("disengaged", out.lower())
        self.assertFalse(self.mod.is_focus_mode_active())

    def test_status_inactive(self):
        self.assertIn("not currently engaged",
                      self.actions["focus_mode_status"]("").lower())

    def test_status_active_reports_remaining_and_trigger(self):
        with _neutered_side_effects(self.mod):
            self.actions["focus_mode"]("90 minutes")
        out = self.actions["focus_mode_status"]("")
        self.assertIn("engaged", out.lower())
        self.assertIn("by voice", out)
        self.assertIn("remaining", out.lower())

    def test_status_workshop_trigger_label(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="workshop")
        out = self.actions["focus_mode_status"]("")
        self.assertIn("by workshop activity", out)

    def test_register_wires_three_actions(self):
        # register() (called by the harness) must expose exactly these.
        for name in ("focus_mode", "end_focus_mode", "focus_mode_status"):
            self.assertIn(name, self.actions)
            self.assertTrue(callable(self.actions[name]))


# ─── _enqueue_speech (announcer route + atomic fallback) ─────────────────
class FocusEnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")

    def test_routes_through_proactive_announce(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(return_value=True)
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("hello sir")
        bc.proactive_announce.assert_called_once()
        # Source tag identifies the focus skill to the canonical writer.
        self.assertEqual(bc.proactive_announce.call_args.kwargs.get("source"),
                         "focus")

    def test_falls_back_to_atomic_write_when_no_announcer(self):
        # bobert_companion present but proactive_announce missing → local write.
        bc = types.ModuleType("bobert_companion")  # no proactive_announce attr
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        tmp.close()
        self.addCleanup(lambda: __import__("os").unlink(tmp.name))
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", tmp.name), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("fallback message")
        wj.assert_called_once()
        # The payload carries our message.
        written = wj.call_args[0][1]
        self.assertEqual(written[-1]["message"], "fallback message")

    def test_fallback_reads_existing_queue_then_appends(self):
        bc = types.ModuleType("bobert_companion")
        import json
        import os
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        json.dump([{"ts": 1.0, "message": "old"}], tmp)
        tmp.close()
        self.addCleanup(lambda: os.unlink(tmp.name))
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", tmp.name), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("new")
        written = wj.call_args[0][1]
        self.assertEqual([d["message"] for d in written], ["old", "new"])

    def test_fallback_write_failure_is_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE",
                               "C:/nonexistent_dir_zzz/q.json"), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk gone")):
            # Must not raise — the print-and-continue branch handles it.
            self.mod._enqueue_speech("doomed")

    def test_announcer_raises_then_falls_back(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(side_effect=RuntimeError("boom"))
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_atomic_write_json") as wj:
            self.mod._enqueue_speech("still delivered")
        wj.assert_called_once()


# ─── _set_focus_assist (reg.exe) ─────────────────────────────────────────
class FocusAssistRegTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")

    def test_success_returns_true_and_disables_toasts(self):
        proc = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(self.mod.subprocess, "run",
                               return_value=proc) as run:
            self.assertTrue(self.mod._set_focus_assist(enable_dnd=True))
        # DND → ToastEnabled value "0".
        argv = run.call_args[0][0]
        self.assertIn("0", argv)
        self.assertIn(self.mod._FA_VAL, argv)

    def test_enable_dnd_false_writes_one(self):
        proc = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(self.mod.subprocess, "run",
                               return_value=proc) as run:
            self.assertTrue(self.mod._set_focus_assist(enable_dnd=False))
        self.assertIn("1", run.call_args[0][0])

    def test_nonzero_returncode_returns_false(self):
        proc = types.SimpleNamespace(returncode=1, stdout="", stderr="ACCESS DENIED")
        with mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertFalse(self.mod._set_focus_assist(enable_dnd=True))

    def test_subprocess_raises_returns_false(self):
        with mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("reg.exe missing")):
            self.assertFalse(self.mod._set_focus_assist(enable_dnd=True))


# ─── _set_teams_presence (Graph) ─────────────────────────────────────────
class TeamsPresenceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")

    def _graph(self, token="tok"):
        mod = types.ModuleType("skills.ms_graph")
        mod.get_access_token = mock.MagicMock(return_value=token)
        return mod

    def test_no_graph_module_returns_false(self):
        # Both import paths fail → False, no network attempted.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no ms_graph")), \
             mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_ms_graph", None)
            self.assertFalse(self.mod._set_teams_presence("DoNotDisturb"))

    def test_no_token_returns_false(self):
        with inject_modules(**{"skills.ms_graph": self._graph(token=None)}):
            self.assertFalse(self.mod._set_teams_presence("DoNotDisturb"))

    def test_token_getter_raises_returns_false(self):
        g = types.ModuleType("skills.ms_graph")
        g.get_access_token = mock.MagicMock(side_effect=RuntimeError("expired"))
        with inject_modules(**{"skills.ms_graph": g}):
            self.assertFalse(self.mod._set_teams_presence("DoNotDisturb"))

    def test_success_posts_and_returns_true(self):
        resp = mock.MagicMock()
        resp.status = 200
        cm = mock.MagicMock()
        cm.__enter__ = mock.MagicMock(return_value=resp)
        cm.__exit__ = mock.MagicMock(return_value=False)
        with inject_modules(**{"skills.ms_graph": self._graph()}), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               return_value=cm):
            self.assertTrue(self.mod._set_teams_presence("DoNotDisturb"))

    def test_http_error_returns_false(self):
        import urllib.error
        import io
        err = urllib.error.HTTPError(
            "url", 403, "Forbidden", {}, io.BytesIO(b"denied"))
        with inject_modules(**{"skills.ms_graph": self._graph()}), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=err):
            self.assertFalse(self.mod._set_teams_presence("DoNotDisturb"))

    def test_generic_error_returns_false(self):
        with inject_modules(**{"skills.ms_graph": self._graph()}), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=OSError("connection reset")):
            self.assertFalse(self.mod._set_teams_presence("DoNotDisturb"))

    def test_http_error_body_read_failure_swallowed(self):
        # On an HTTPError, the handler tries to read the response body for the
        # log line; if THAT read() raises, the inner `except Exception: pass`
        # swallows it and the function still returns False cleanly.
        import urllib.error

        class _BadBodyHTTPError(urllib.error.HTTPError):
            def read(self, *a, **k):
                raise OSError("socket already closed")

        err = _BadBodyHTTPError("url", 500, "Server Error", {}, None)
        with inject_modules(**{"skills.ms_graph": self._graph()}), \
             mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=err):
            self.assertFalse(self.mod._set_teams_presence("DoNotDisturb"))

    def test_falls_back_to_sys_modules_skill_ms_graph(self):
        # When `skills.ms_graph` is unimportable, the helper consults the
        # already-loaded skill_ms_graph entry in sys.modules.
        g = types.ModuleType("skill_ms_graph")
        g.get_access_token = mock.MagicMock(return_value=None)
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no pkg path")), \
             mock.patch.dict(sys.modules, {"skill_ms_graph": g}):
            # token is None → returns False, but it DID find the module.
            self.assertFalse(self.mod._set_teams_presence("Available"))
        g.get_access_token.assert_called_once()


# ─── nudge suppression (critical allow-list) ─────────────────────────────
class FocusNudgeSuppressionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")
        # Always start from a clean saved-state map.
        self.mod._saved_enqueues.clear()
        self.addCleanup(self.mod._saved_enqueues.clear)

    def _fake_skill(self, name):
        # A REAL function (not MagicMock) so getattr(fn, "_is_nudge_sink",
        # False) is genuinely falsy — a MagicMock would auto-vivify that
        # attribute as a truthy mock and defeat the sink-detection logic.
        m = types.ModuleType(f"skill_{name}")
        calls = []

        def _orig_enqueue(message, _calls=calls):
            _calls.append(message)
        _orig_enqueue.calls = calls
        m._enqueue_speech = _orig_enqueue
        return m

    def test_suppresses_listed_skills_only(self):
        # Build a fake module per suppressed skill PLUS a critical announcer
        # that must NOT be touched.
        fakes = {f"skill_{s}": self._fake_skill(s)
                 for s in self.mod.SUPPRESSED_SKILLS}
        critical = self._fake_skill("bambu_monitor")   # not in the allow-list
        original_critical = critical._enqueue_speech
        fakes["skill_bambu_monitor"] = critical
        with inject_modules(**fakes):
            self.mod._install_nudge_suppressors()
            # Every suppressed skill now has a sink.
            for s in self.mod.SUPPRESSED_SKILLS:
                sink = sys.modules[f"skill_{s}"]._enqueue_speech
                self.assertTrue(getattr(sink, "_is_nudge_sink", False))
            # The critical announcer is untouched — VIPs/emergencies still pass.
            self.assertIs(critical._enqueue_speech, original_critical)
            self.assertFalse(getattr(critical._enqueue_speech,
                                     "_is_nudge_sink", False))
            # The sink drops (just logs) and does not raise.
            sys.modules[f"skill_{self.mod.SUPPRESSED_SKILLS[0]}"]._enqueue_speech(
                "banter line")
            self.mod._restore_nudge_suppressors()
            # Restored to the genuine original callables.
            for s in self.mod.SUPPRESSED_SKILLS:
                restored = sys.modules[f"skill_{s}"]._enqueue_speech
                self.assertFalse(getattr(restored, "_is_nudge_sink", False))

    def test_missing_skill_module_is_skipped(self):
        # No skill_* modules present → install is a no-op, nothing saved.
        for s in self.mod.SUPPRESSED_SKILLS:
            sys.modules.pop(f"skill_{s}", None)
        self.mod._install_nudge_suppressors()
        self.assertEqual(self.mod._saved_enqueues, {})

    def test_non_callable_enqueue_is_skipped(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        m = types.ModuleType(f"skill_{s0}")
        m._enqueue_speech = "not callable"
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
        self.assertNotIn(s0, self.mod._saved_enqueues)

    def test_existing_sink_not_double_wrapped_or_owned(self):
        # Simulate an overlapping mode having already installed a sink: we must
        # record None (don't own it) and leave the sink in place; restore must
        # be a no-op for it.
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        m = types.ModuleType(f"skill_{s0}")

        def _other_sink(msg):
            pass
        _other_sink._is_nudge_sink = True
        m._enqueue_speech = _other_sink
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
            self.assertIsNone(self.mod._saved_enqueues[s0])
            self.assertIs(m._enqueue_speech, _other_sink)   # untouched
            self.mod._restore_nudge_suppressors()
            # Still the other mode's sink — we did not strand or overwrite it.
            self.assertIs(m._enqueue_speech, _other_sink)

    def test_double_install_does_not_rewrap(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        m = self._fake_skill(s0)
        real_original = m._enqueue_speech
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
            first_sink = m._enqueue_speech
            self.mod._install_nudge_suppressors()   # second call: already ours
            self.assertIs(m._enqueue_speech, first_sink)
            self.assertIs(self.mod._saved_enqueues[s0], real_original)

    def test_restore_skips_when_module_gone(self):
        # Saved an original, but the module vanished from sys.modules → restore
        # must not raise.
        s0 = self.mod.SUPPRESSED_SKILLS[0]
        self.mod._saved_enqueues[s0] = lambda m: None
        sys.modules.pop(f"skill_{s0}", None)
        self.mod._restore_nudge_suppressors()
        self.assertEqual(self.mod._saved_enqueues, {})


# ─── prompt addendum ─────────────────────────────────────────────────────
class FocusPromptAddendumTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")
        self.mod._saved_system_prompt[0] = None
        self.addCleanup(lambda: self.mod._saved_system_prompt.__setitem__(0, None))

    def test_apply_then_restore_roundtrip(self):
        bc = types.ModuleType("bobert_companion")
        bc._system_prompt = "BASE PROMPT"
        with inject_modules(bobert_companion=bc):
            self.mod._apply_prompt_addendum()
            self.assertTrue(bc._system_prompt.startswith("BASE PROMPT"))
            self.assertIn("[Focus mode]", bc._system_prompt)
            self.assertIs(self.mod._saved_system_prompt[0], "BASE PROMPT")
            self.mod._restore_prompt_addendum()
            self.assertEqual(bc._system_prompt, "BASE PROMPT")
            self.assertIsNone(self.mod._saved_system_prompt[0])

    def test_apply_import_failure_is_silent(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            self.mod._apply_prompt_addendum()   # no raise
        self.assertIsNone(self.mod._saved_system_prompt[0])

    def test_restore_noop_when_nothing_saved(self):
        bc = types.ModuleType("bobert_companion")
        bc._system_prompt = "UNCHANGED"
        with inject_modules(bobert_companion=bc):
            self.mod._restore_prompt_addendum()   # saved is None → no-op
        self.assertEqual(bc._system_prompt, "UNCHANGED")

    def test_restore_import_failure_is_silent(self):
        self.mod._saved_system_prompt[0] = "X"
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("gone")):
            self.mod._restore_prompt_addendum()   # no raise


# ─── expiry thread ───────────────────────────────────────────────────────
class FocusExpiryThreadTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")
        self.mod._focus_active[0] = False
        self.mod._focus_ends_at[0] = 0.0

    def test_expiry_thread_exits_when_deadline_passed(self):
        # Active with a deadline already in the past → the worker breaks out of
        # its sleep loop on the first tick and calls _exit_focus_mode("expired").
        self.mod._focus_active[0] = True
        self.mod._focus_ends_at[0] = self.mod.time.time() - 1
        exited = threading.Event()

        def _fake_exit(reason="manual"):
            self.assertEqual(reason, "expired")
            exited.set()
            return "done"
        with mock.patch.object(self.mod, "_exit_focus_mode",
                               side_effect=_fake_exit), \
             mock.patch.object(self.mod.time, "sleep",
                               side_effect=lambda *_a, **_k: None):
            self.mod._start_expiry_thread()
            t = self.mod._expiry_thread[0]
            t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertTrue(exited.is_set())

    def test_expiry_thread_returns_early_if_cancelled(self):
        # Not active when the worker wakes → it returns WITHOUT calling exit.
        self.mod._focus_active[0] = False
        with mock.patch.object(self.mod, "_exit_focus_mode") as ex, \
             mock.patch.object(self.mod.time, "sleep",
                               side_effect=lambda *_a, **_k: None):
            self.mod._start_expiry_thread()
            self.mod._expiry_thread[0].join(timeout=5)
        ex.assert_not_called()


# ─── workshop_mode auto-trigger hook ─────────────────────────────────────
class FocusWorkshopHookTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")
        self.mod._workshop_hook_installed[0] = False
        self.mod._focus_active[0] = False
        self.mod._focus_trigger[0] = ""
        self.addCleanup(lambda: self.mod._workshop_hook_installed.__setitem__(0, False))

    def _workshop_mod(self):
        m = types.ModuleType("skill_workshop_mode")
        m.entered = []
        m.exited = []

        def _enter(matched_hint, full_title):
            m.entered.append((matched_hint, full_title))
            return "ws-entered"

        def _exit():
            m.exited.append(True)
            return "ws-exited"
        m._enter_workshop_mode = _enter
        m._exit_workshop_mode = _exit
        return m

    def test_hook_not_installed_when_module_absent(self):
        sys.modules.pop("skill_workshop_mode", None)
        self.mod._install_workshop_hook()
        self.assertFalse(self.mod._workshop_hook_installed[0])

    def test_hook_skipped_if_already_installed(self):
        self.mod._workshop_hook_installed[0] = True
        ws = self._workshop_mod()
        original_enter = ws._enter_workshop_mode
        with inject_modules(skill_workshop_mode=ws):
            self.mod._install_workshop_hook()
        # Unchanged — early-return guard hit.
        self.assertIs(ws._enter_workshop_mode, original_enter)

    def test_hook_skipped_when_targets_not_callable(self):
        ws = types.ModuleType("skill_workshop_mode")
        ws._enter_workshop_mode = None
        ws._exit_workshop_mode = None
        with inject_modules(skill_workshop_mode=ws):
            self.mod._install_workshop_hook()
        self.assertFalse(self.mod._workshop_hook_installed[0])

    def test_wrapped_enter_auto_engages_focus(self):
        ws = self._workshop_mod()
        with inject_modules(skill_workshop_mode=ws), \
             _neutered_side_effects(self.mod):
            self.mod._install_workshop_hook()
            self.assertTrue(self.mod._workshop_hook_installed[0])
            # Calling the now-wrapped enter triggers workshop AND focus.
            result = ws._enter_workshop_mode("cad", "Fusion 360")
        self.assertEqual(result, "ws-entered")        # original return passes through
        self.assertEqual(ws.entered, [("cad", "Fusion 360")])
        self.assertTrue(self.mod.is_focus_mode_active())
        self.assertEqual(self.mod._focus_trigger[0], "workshop")

    def test_wrapped_enter_skips_focus_when_already_active(self):
        ws = self._workshop_mod()
        with inject_modules(skill_workshop_mode=ws), \
             _neutered_side_effects(self.mod):
            # Pre-engage focus by voice; wrapped enter must NOT re-enter.
            self.mod._enter_focus_mode(600, trigger="voice")
            self.mod._install_workshop_hook()
            with mock.patch.object(self.mod, "_enter_focus_mode") as ent:
                ws._enter_workshop_mode("cad", "Fusion")
            ent.assert_not_called()
        self.assertEqual(self.mod._focus_trigger[0], "voice")

    def test_wrapped_exit_releases_workshop_triggered_focus(self):
        ws = self._workshop_mod()
        with inject_modules(skill_workshop_mode=ws), \
             _neutered_side_effects(self.mod):
            self.mod._install_workshop_hook()
            ws._enter_workshop_mode("cad", "Fusion")   # focus engaged via workshop
            self.assertTrue(self.mod.is_focus_mode_active())
            result = ws._exit_workshop_mode()
        self.assertEqual(result, "ws-exited")
        self.assertFalse(self.mod.is_focus_mode_active())

    def test_wrapped_exit_keeps_voice_triggered_focus(self):
        ws = self._workshop_mod()
        with inject_modules(skill_workshop_mode=ws), \
             _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
            self.mod._install_workshop_hook()
            with mock.patch.object(self.mod, "_exit_focus_mode") as ex:
                ws._exit_workshop_mode()
            ex.assert_not_called()
        self.assertTrue(self.mod.is_focus_mode_active())

    def test_delayed_workshop_hook_installs_after_sleep(self):
        ws = self._workshop_mod()
        with inject_modules(skill_workshop_mode=ws), \
             mock.patch.object(self.mod.time, "sleep",
                               side_effect=lambda *_a, **_k: None), \
             mock.patch.object(self.mod, "_install_workshop_hook") as ins:
            self.mod._delayed_workshop_hook()
        ins.assert_called_once()

    def test_delayed_workshop_hook_swallows_exception(self):
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=RuntimeError("interrupted")), \
             mock.patch.object(self.mod.logging, "exception"):
            # Exception inside is logged, never propagated.
            self.mod._delayed_workshop_hook()

    def test_wrapped_enter_exception_is_caught(self):
        ws = self._workshop_mod()
        with inject_modules(skill_workshop_mode=ws), \
             _neutered_side_effects(self.mod):
            self.mod._install_workshop_hook()
            # Make the inner auto-engage raise; the wrapper must swallow it and
            # still return the original workshop-enter result.
            with mock.patch.object(self.mod, "_enter_focus_mode",
                                   side_effect=RuntimeError("engage boom")):
                result = ws._enter_workshop_mode("cad", "Fusion")
        self.assertEqual(result, "ws-entered")

    def test_wrapped_exit_exception_is_caught(self):
        ws = self._workshop_mod()
        with inject_modules(skill_workshop_mode=ws), \
             _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="workshop")
            self.mod._install_workshop_hook()
            with mock.patch.object(self.mod, "_exit_focus_mode",
                                   side_effect=RuntimeError("release boom")):
                result = ws._exit_workshop_mode()
        self.assertEqual(result, "ws-exited")


class FocusExceptionBranchTests(unittest.TestCase):
    """The deep best-effort exception handlers — each side effect is designed
    to fail soft, so we force the failure and assert no propagation."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("dnd_focus_mode")
        self.mod._saved_enqueues.clear()
        self.mod._saved_system_prompt[0] = None
        self.addCleanup(self.mod._saved_enqueues.clear)
        self.addCleanup(lambda: self.mod._saved_system_prompt.__setitem__(0, None))

    def test_nudge_wrap_assignment_failure_pops_saved(self):
        # If setting mod._enqueue_speech raises, the skill removes the cached
        # original so a later restore is a no-op.
        s0 = self.mod.SUPPRESSED_SKILLS[0]

        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_enqueue_speech" and getattr(v, "_is_nudge_sink", False):
                    raise RuntimeError("read-only module")
                super().__setattr__(k, v)
        m = _Locked(f"skill_{s0}")
        super(_Locked, m).__setattr__("_enqueue_speech", lambda msg: None)
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._install_nudge_suppressors()
        self.assertNotIn(s0, self.mod._saved_enqueues)

    def test_nudge_restore_assignment_failure_is_swallowed(self):
        s0 = self.mod.SUPPRESSED_SKILLS[0]

        def _orig(msg):
            pass

        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_enqueue_speech":
                    raise RuntimeError("cannot restore")
                super().__setattr__(k, v)
        m = _Locked(f"skill_{s0}")
        self.mod._saved_enqueues[s0] = _orig
        with inject_modules(**{f"skill_{s0}": m}):
            self.mod._restore_nudge_suppressors()   # must not raise
        self.assertEqual(self.mod._saved_enqueues, {})

    def test_workshop_hook_install_assignment_failure_is_swallowed(self):
        # The final wrap-assignment is wrapped in try/except so a frozen
        # workshop module can't crash register's deferred hook installer.
        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k in ("_enter_workshop_mode", "_exit_workshop_mode") \
                        and callable(v) and getattr(v, "__name__", "") \
                        .startswith("_wrapped"):
                    raise RuntimeError("frozen workshop module")
                super().__setattr__(k, v)
        ws = _Locked("skill_workshop_mode")
        super(_Locked, ws).__setattr__(
            "_enter_workshop_mode", lambda h, t: "e")
        super(_Locked, ws).__setattr__("_exit_workshop_mode", lambda: "x")
        with inject_modules(skill_workshop_mode=ws):
            self.mod._workshop_hook_installed[0] = False
            self.mod._install_workshop_hook()   # must not raise
        self.assertFalse(self.mod._workshop_hook_installed[0])
        self.addCleanup(lambda: self.mod._workshop_hook_installed.__setitem__(0, False))

    def test_apply_prompt_addendum_assignment_failure_is_swallowed(self):
        class _Locked(types.ModuleType):
            def __setattr__(self, k, v):
                if k == "_system_prompt" and v != "BASE":
                    raise RuntimeError("frozen prompt")
                super().__setattr__(k, v)
        bc = _Locked("bobert_companion")
        super(_Locked, bc).__setattr__("_system_prompt", "BASE")
        with inject_modules(bobert_companion=bc):
            self.mod._apply_prompt_addendum()   # logs, no raise

    def test_expiry_thread_iteration_exception_then_recovers(self):
        self.mod._focus_active[0] = True
        self.mod._focus_ends_at[0] = self.mod.time.time() + 100   # remaining > 0
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("sleep interrupted")
            # On the recovery sleep, cancel so the worker returns next loop.
            self.mod._focus_active[0] = False
        with mock.patch.object(self.mod, "_exit_focus_mode") as ex, \
             mock.patch.object(self.mod.logging, "exception") as logexc, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep):
            self.mod._start_expiry_thread()
            self.mod._expiry_thread[0].join(timeout=5)
        logexc.assert_called()       # the iteration-failed branch ran
        ex.assert_not_called()       # cancelled before deadline → no expiry exit


if __name__ == "__main__":
    unittest.main()
