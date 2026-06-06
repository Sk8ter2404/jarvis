"""Unit tests for bobert_companion.py, section 6 (source lines ~10903-13152).

This band of the monolith is the *action-dispatch + speech + boot-preflight*
layer. It contains:

  * Pure / near-pure helpers — the unsaved-window blurb composer, the
    destructive-shell + sketchy-URL classifiers, the LAN-host whitelist, the
    JARVIS-pushback objection builder, the mission-narration intro/cue
    formatters, the markdown-for-speech stripper, the dropped-step detector,
    and the orchestration-request matcher.
  * The big ``parse_and_run_actions`` dispatcher (runs whitelisted [ACTION:]
    tokens, defers confirm/pushback ones, detects hallucinated + dropped
    steps) and its follow-up-LLM companion ``get_followup_response``.
  * The two verbal-resolution handlers (``handle_confirmation_response`` /
    ``handle_autocorrect_disambig_response``).
  * The ``_speak`` TTS front (tag parsing -> markdown strip -> mute/staging
    short-circuits -> locked synth+play) and ``_apply_quip_layer``.
  * Boot-time self-heal: power-plan switch helpers, the cuBLAS DLL probe, the
    Claude-API preflight ping, and the injected-command / pending-speech queue
    drainers.
  * The orchestrator gate (``_orchestrator_enabled`` / ``_is_orchestration_
    request`` / ``_maybe_orchestrate``) and the singleton release.

Out of scope (boot-only, infinite-loop, or hardware-bound; covered only at
their guard clauses where practical): ``_overnight_upgrade_thread`` (``while
True`` engine), ``_do_proactive_turn`` (spins real animation threads),
``_startup_preflight`` (fatal ``sys.exit`` orchestration), ``_preflight_
cameras`` (fans real camera-probe threads), ``_enforce_singleton`` (forks
``tasklist`` / can ``sys.exit``), and ``main()`` / the ``__main__`` block.

Strategy: import the monolith ONCE via the harness (cached), then patch the
specific module globals each function reads via ``mock.patch.object(bc, ...)``.
ALL external I/O (LLM/anthropic, subprocess, sounddevice via synthesise/
play_with_lipsync, threads/timers, filesystem) is mocked or redirected to a
TemporaryDirectory. Lazily-imported sibling modules (``staging_instance``,
``core.orchestrator``) are faked via ``mock.patch.dict(sys.modules, ...)`` for
the duration of one test — the cached harness module object is never swapped,
and directly-mutated globals (the two pending-action lists, ``conversation_
history``) are restored in tearDown / addCleanup.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_sec6
On the light-deps CI runner these all skip via @requires_monolith.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


# A minimal stand-in for the window objects _jarvis_pushback inspects: it only
# ever reads ``.title``.
_FakeWin = collections.namedtuple("_FakeWin", ["title"])


@requires_monolith
class SectionSixBase(MonolithGlobalsTestCase):
    # setUpClass (loads the cached monolith) + a per-test deep-restore of the
    # mutated bobert_companion globals are inherited from
    # MonolithGlobalsTestCase; the lightweight queue snapshot below is kept as
    # a fast first line of defence for this section's own dispatch tests.

    def setUp(self):
        bc = self.bc
        # Snapshot the directly-mutated module-level queues so a test that
        # appends to them can't leak into the next one even if it raises.
        self._pc_snapshot = list(bc._pending_confirmation)
        self._ac_snapshot = list(bc._pending_autocorrect_choice)
        self._hist_len = len(bc.conversation_history)
        self.addCleanup(self._restore_globals)

    def _restore_globals(self):
        bc = self.bc
        bc._pending_confirmation[:] = self._pc_snapshot
        bc._pending_autocorrect_choice[:] = self._ac_snapshot
        del bc.conversation_history[self._hist_len:]

    # Convenience: start a patch and auto-stop it at test teardown.
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m


# ════════════════════════════════════════════════════════════════════════════
#  _unsaved_window_blurb
# ════════════════════════════════════════════════════════════════════════════
class UnsavedWindowBlurbTests(SectionSixBase):
    def test_leading_bullet_extracts_app_name(self):
        # "* notes.txt - Notepad" -> strong unsaved signal, app is last segment.
        out = self.bc._unsaved_window_blurb(["* notes.txt - Notepad"])
        self.assertEqual(out, "your unsaved Notepad project")

    def test_untitled_hint_matches(self):
        out = self.bc._unsaved_window_blurb(["Untitled - Notepad"])
        self.assertEqual(out, "your unsaved Notepad project")

    def test_embedded_bullet_treated_as_unsaved(self):
        out = self.bc._unsaved_window_blurb(["draft ● - SomeEditor"])
        self.assertEqual(out, "your unsaved SomeEditor project")

    def test_no_separator_quotes_trimmed_title(self):
        # Strong signal (leading *) but no " - "/" — " separator -> quoted form.
        out = self.bc._unsaved_window_blurb(["*scratchpad"])
        self.assertEqual(out, 'your unsaved "scratchpad"')

    def test_live_edit_fallback_when_no_strong_signal(self):
        # ".md - Cursor": no unsaved hint + no bullet -> live-edit fallback.
        out = self.bc._unsaved_window_blurb(["report.md - Cursor"])
        self.assertEqual(out, "your open Cursor window")

    def test_vs_code_live_edit_marker(self):
        out = self.bc._unsaved_window_blurb(["main.py - Visual Studio Code"])
        self.assertEqual(out, "your open VS Code window")

    def test_no_match_returns_none(self):
        self.assertIsNone(self.bc._unsaved_window_blurb(["Calculator"]))

    def test_empty_and_blank_titles_return_none(self):
        self.assertIsNone(self.bc._unsaved_window_blurb([]))
        self.assertIsNone(self.bc._unsaved_window_blurb(["", "   ", None]))

    def test_strong_signal_wins_over_earlier_fallback(self):
        # First title is a live-edit fallback; a later strong signal should
        # take precedence and return the unsaved-project form.
        out = self.bc._unsaved_window_blurb(
            ["readme.md - Google Docs", "* important.txt - Notepad"])
        self.assertEqual(out, "your unsaved Notepad project")


# ════════════════════════════════════════════════════════════════════════════
#  Destructive-shell + sketchy-URL + LAN-host classifiers
# ════════════════════════════════════════════════════════════════════════════
class ShellAndUrlClassifierTests(SectionSixBase):
    def test_destructive_shell_positive(self):
        for cmd in ("rm -rf /tmp/x", "git reset --hard", "drop table users",
                    "remove-item -recurse foo", "del /s /q c:\\bar"):
            self.assertTrue(self.bc._looks_like_destructive_shell(cmd), cmd)

    def test_destructive_shell_negative(self):
        for cmd in ("ls -la", "git status", "python build.py", "echo hi"):
            self.assertFalse(self.bc._looks_like_destructive_shell(cmd), cmd)

    def test_local_or_lan_hosts(self):
        bc = self.bc
        for h in ("localhost", "mybox.local", "127.0.0.1", "192.168.1.5",
                  "10.0.0.3", "169.254.1.1", "172.16.0.1", "172.31.255.1"):
            self.assertTrue(bc._is_local_or_lan_host(h), h)

    def test_non_lan_hosts(self):
        bc = self.bc
        for h in ("example.com", "8.8.8.8", "172.15.0.1", "172.32.0.1", ""):
            self.assertFalse(bc._is_local_or_lan_host(h), h)

    def test_lan_host_malformed_172_octet(self):
        # "172.x" with a non-numeric second octet must not raise.
        self.assertFalse(self.bc._is_local_or_lan_host("172.foo.0.1"))

    def test_sketchy_url_bare_ip_over_http(self):
        self.assertTrue(self.bc._looks_like_sketchy_url("http://203.0.113.5/x"))

    def test_sketchy_url_phishing_tld(self):
        self.assertTrue(self.bc._looks_like_sketchy_url("http://evil.tk/path"))
        # No scheme -> normalized to https, still caught by the TLD pattern.
        self.assertTrue(self.bc._looks_like_sketchy_url("evil.tk/path"))

    def test_sketchy_url_shortener_and_tunnel(self):
        self.assertTrue(self.bc._looks_like_sketchy_url("https://bit.ly/abc"))
        self.assertTrue(
            self.bc._looks_like_sketchy_url("https://foo.ngrok-free.app/x"))

    def test_plain_http_public_host_is_sketchy(self):
        self.assertTrue(self.bc._looks_like_sketchy_url("http://example.com"))

    def test_https_public_host_is_clean(self):
        self.assertFalse(self.bc._looks_like_sketchy_url("https://example.com"))

    def test_lan_url_whitelisted(self):
        # LAN host short-circuits before the bare-IP / http-without-tls checks.
        self.assertFalse(
            self.bc._looks_like_sketchy_url("http://192.168.1.5:8080"))
        self.assertFalse(
            self.bc._looks_like_sketchy_url("http://localhost:3000/app"))

    def test_empty_url_is_not_sketchy(self):
        self.assertFalse(self.bc._looks_like_sketchy_url(""))
        self.assertFalse(self.bc._looks_like_sketchy_url("   "))


# ════════════════════════════════════════════════════════════════════════════
#  _jarvis_pushback
# ════════════════════════════════════════════════════════════════════════════
class JarvisPushbackTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        # Pushback logic is gated on PUSHBACK_ENABLED — force it on regardless
        # of the live config so these tests are deterministic.
        self._p(self.bc, "PUSHBACK_ENABLED", True)

    def test_disabled_returns_none(self):
        with mock.patch.object(self.bc, "PUSHBACK_ENABLED", False):
            self.assertIsNone(
                self.bc._jarvis_pushback("run_shell", "rm -rf /"))

    def test_reset_memory_always_pushes_back(self):
        res = self.bc._jarvis_pushback("reset_memory", "")
        self.assertIsNotNone(res)
        phrase, reason = res
        self.assertIn("erase my entire memory", phrase)
        self.assertIn("reset_memory", reason)

    def test_forget_last_hour_always_pushes_back(self):
        phrase, reason = self.bc._jarvis_pushback("forget_last_hour", "")
        self.assertIn("last hour", phrase)
        self.assertIn("forget_last_hour", reason)

    def test_destructive_run_shell(self):
        phrase, reason = self.bc._jarvis_pushback(
            "run_shell", "rm -rf node_modules")
        self.assertIn("inadvisable", phrase)
        self.assertIn("destructive shell", reason)

    def test_non_destructive_run_shell_passes(self):
        self.assertIsNone(self.bc._jarvis_pushback("run_shell", "ls -la"))

    def test_sketchy_open_url(self):
        phrase, reason = self.bc._jarvis_pushback("open_url", "http://evil.tk")
        self.assertIn("unsavoury", phrase)
        self.assertIn("sketchy URL", reason)

    def test_safe_open_url_passes(self):
        self.assertIsNone(
            self.bc._jarvis_pushback("open_url", "https://example.com"))

    def test_close_many_windows_with_unsaved_blurb(self):
        bc = self.bc
        n = bc.PUSHBACK_MAX_CLOSE_WINDOWS + 1
        wins = [_FakeWin(f"* doc{i}.txt - Notepad") for i in range(n)]
        with mock.patch.object(bc, "_find_windows_by_title", return_value=wins):
            phrase, reason = bc._jarvis_pushback("close_window", "notepad")
        self.assertIn(f"close {n} windows", phrase)
        self.assertIn("unsaved", phrase)  # blurb embellishment present
        self.assertIn(str(n), reason)

    def test_close_few_windows_no_pushback(self):
        bc = self.bc
        wins = [_FakeWin("a"), _FakeWin("b")]  # under the threshold
        with mock.patch.object(bc, "_find_windows_by_title", return_value=wins):
            self.assertIsNone(bc._jarvis_pushback("close_window", "x"))

    def test_close_window_find_raises_is_swallowed(self):
        bc = self.bc
        with mock.patch.object(bc, "_find_windows_by_title",
                               side_effect=RuntimeError("boom")):
            # 0 matches after the swallow -> no pushback, no propagation.
            self.assertIsNone(bc._jarvis_pushback("close_window", "x"))

    def test_bulk_queue_task_pushback(self):
        bc = self.bc
        n = bc.PUSHBACK_MAX_QUEUE_TASKS_BULK + 5
        big = "\n".join(f"task {i}" for i in range(n))
        phrase, reason = bc._jarvis_pushback("queue_task", big)
        self.assertIn(f"{n} tasks at once", phrase)
        self.assertIn("bulk queue", reason)

    def test_small_queue_task_no_pushback(self):
        self.assertIsNone(self.bc._jarvis_pushback("queue_task", "one\ntwo"))

    def test_clear_tasks_over_threshold(self):
        bc = self.bc
        n = bc.PUSHBACK_MAX_CLEAR_PENDING + 3
        # _jarvis_pushback opens TODO_FILE and counts "- [ ]" lines.
        with tempfile.TemporaryDirectory() as d:
            todo = os.path.join(d, "todo.md")
            with open(todo, "w", encoding="utf-8") as f:
                f.write("\n".join("- [ ] pending item" for _ in range(n)))
            with mock.patch.object(bc, "TODO_FILE", todo):
                phrase, reason = bc._jarvis_pushback("clear_tasks", "")
        self.assertIn(f"wipe {n} pending", phrase)
        self.assertIn("clear_tasks", reason)

    def test_clear_tasks_missing_file_no_pushback(self):
        bc = self.bc
        with mock.patch.object(bc, "TODO_FILE",
                               os.path.join(tempfile.gettempdir(),
                                            "definitely_missing_todo.md")):
            self.assertIsNone(bc._jarvis_pushback("clear_tasks", ""))

    def test_unrelated_action_no_pushback(self):
        self.assertIsNone(self.bc._jarvis_pushback("get_time", ""))


# ════════════════════════════════════════════════════════════════════════════
#  Mission narration
# ════════════════════════════════════════════════════════════════════════════
class MissionNarrationTests(SectionSixBase):
    def test_intro_spelled_numbers(self):
        self.assertEqual(self.bc._mission_narration_intro(3), "Three steps, sir.")
        self.assertEqual(self.bc._mission_narration_intro(2), "Two steps, sir.")
        self.assertEqual(self.bc._mission_narration_intro(9), "Nine steps, sir.")

    def test_intro_large_count_uses_digits(self):
        self.assertEqual(self.bc._mission_narration_intro(12), "12 steps, sir.")

    def test_cue_known_action_with_arg(self):
        self.assertEqual(
            self.bc._mission_narration_cue("web_search", "cats", 1, 3),
            "Searching for cats…")

    def test_cue_known_action_no_arg_strips_placeholder(self):
        # "open_url" template has no {arg}; ends with ellipsis cleanly.
        self.assertEqual(
            self.bc._mission_narration_cue("open_url", "", 1, 3),
            "Opening the page…")

    def test_cue_long_arg_is_truncated(self):
        out = self.bc._mission_narration_cue("web_search", "x" * 80, 1, 2)
        self.assertTrue(out.startswith("Searching for "))
        self.assertIn("…", out)
        # The arg is trimmed to 37 chars + ellipsis well under the raw length.
        self.assertLess(len(out), 80)

    def test_cue_unknown_action_generic_form(self):
        self.assertEqual(
            self.bc._mission_narration_cue("frobnicate_thing", "", 2, 4),
            "Step 2 of 4: frobnicate thing…")


# ════════════════════════════════════════════════════════════════════════════
#  _strip_markdown_for_speech
# ════════════════════════════════════════════════════════════════════════════
class StripMarkdownTests(SectionSixBase):
    def test_bold_italic_code_link(self):
        bc = self.bc
        self.assertEqual(bc._strip_markdown_for_speech("**bold**"), "bold")
        self.assertEqual(bc._strip_markdown_for_speech("a *italic* b"),
                         "a italic b")
        self.assertEqual(bc._strip_markdown_for_speech("run `code` now"),
                         "run code now")
        self.assertEqual(
            bc._strip_markdown_for_speech("see [the docs](http://x)"),
            "see the docs")

    def test_headings_and_bullets_stripped(self):
        bc = self.bc
        self.assertEqual(bc._strip_markdown_for_speech("# Title"), "Title")
        self.assertEqual(bc._strip_markdown_for_speech("- item"), "item")

    def test_snake_case_underscores_become_spaces(self):
        self.assertEqual(
            self.bc._strip_markdown_for_speech("BAMBU_PRINTER_IP"),
            "BAMBU PRINTER IP")

    def test_done_sentinel_removed(self):
        out = self.bc._strip_markdown_for_speech("✓ DONE — wired the thing")
        self.assertNotIn("DONE", out)
        self.assertIn("wired the thing", out)

    def test_whitespace_collapsed(self):
        self.assertEqual(
            self.bc._strip_markdown_for_speech("a    b\n\nc"), "a b c")

    def test_empty_passthrough(self):
        self.assertEqual(self.bc._strip_markdown_for_speech(""), "")
        self.assertIsNone(self.bc._strip_markdown_for_speech(None))


# ════════════════════════════════════════════════════════════════════════════
#  _detect_dropped_steps
# ════════════════════════════════════════════════════════════════════════════
class DetectDroppedStepsTests(SectionSixBase):
    def test_promised_read_after_future_marker(self):
        bc = self.bc
        # "see_screen" must be a live ACTION for this to fire.
        self.assertIn("see_screen", bc.ACTIONS)
        dropped = bc._detect_dropped_steps(
            "I'll open the page and then read it to you.", set())
        self.assertEqual(dropped, [("see_screen", "read what's on screen")])

    def test_no_future_marker_returns_empty(self):
        # Past-tense narration -> no marker -> nothing flagged.
        self.assertEqual(
            self.bc._detect_dropped_steps("I opened the page and read it.",
                                          set()),
            [])

    def test_already_emitted_action_not_flagged(self):
        self.assertEqual(
            self.bc._detect_dropped_steps(
                "I'll open the page and then read it to you.",
                {"see_screen"}),
            [])

    def test_action_absent_from_registry_not_flagged(self):
        bc = self.bc
        # Remove see_screen from ACTIONS so the (a) registered guard trips.
        acts = {k: v for k, v in bc.ACTIONS.items() if k != "see_screen"}
        with mock.patch.object(bc, "ACTIONS", acts):
            self.assertEqual(
                bc._detect_dropped_steps(
                    "I'll open the page and then read it to you.", set()),
                [])

    def test_target_outside_window_not_flagged(self):
        bc = self.bc
        # Push the intent verb far past the ~120-char window after the marker.
        filler = "x " * 120
        text = f"I'll do something. {filler} and read it to you."
        self.assertEqual(bc._detect_dropped_steps(text, set()), [])


# ════════════════════════════════════════════════════════════════════════════
#  parse_and_run_actions
# ════════════════════════════════════════════════════════════════════════════
class ParseAndRunActionsTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        # Quiet + deterministic defaults for every dispatch test.
        self._p(bc, "_speak", lambda *a, **k: None)
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        # Disable the fuzzy-autocorrect layer so unknown-action tests don't try
        # to score embeddings (which can hang / hit a 60s Levenshtein fallback).
        self._p(bc, "_cmd_autocorrect", None)
        # PC control must be on for the dispatcher to do anything.
        self._p(bc, "PC_CONTROL_ENABLED", True)

    def _with_action(self, name, fn, informative=False):
        """Return a patched ACTIONS dict (+ INFORMATIVE set) with `name`->fn."""
        bc = self.bc
        acts = dict(bc.ACTIONS)
        acts[name] = fn
        self._p(bc, "ACTIONS", acts)
        if informative:
            info = set(bc.INFORMATIVE_ACTIONS) | {name}
            self._p(bc, "INFORMATIVE_ACTIONS", info)

    def test_pc_control_off_short_circuits(self):
        with mock.patch.object(self.bc, "PC_CONTROL_ENABLED", False):
            cleaned, results = self.bc.parse_and_run_actions(
                "hi [ACTION: get_time]")
            self.assertEqual(cleaned, "hi [ACTION: get_time]")
            self.assertEqual(results, [])

    def test_runs_informative_action_and_strips_token(self):
        calls = []
        self._with_action("testecho", lambda a: calls.append(a) or f"R-{a}",
                          informative=True)
        with mock.patch.object(self.bc, "_needs_confirmation",
                               lambda n, a: False), \
             mock.patch.object(self.bc, "_jarvis_pushback", lambda n, a: None):
            cleaned, results = self.bc.parse_and_run_actions(
                "Sure. [ACTION: testecho, hello]")
        self.assertEqual(cleaned, "Sure.")
        self.assertEqual(calls, ["hello"])
        self.assertEqual(results, [("testecho", "R-hello", True)])

    def test_unknown_action_recorded(self):
        cleaned, results = self.bc.parse_and_run_actions(
            "[ACTION: zzz_no_such_action]")
        self.assertEqual(cleaned, "")
        self.assertEqual(results,
                         [("zzz_no_such_action",
                           "unknown action: zzz_no_such_action", False)])

    def test_confirmation_defers_registered_action(self):
        bc = self.bc
        ran = []
        self._with_action("dangeract", lambda a: ran.append(a) or "ran")
        with mock.patch.object(bc, "_needs_confirmation",
                               lambda n, a: n == "dangeract"):
            cleaned, results = bc.parse_and_run_actions(
                "Okay. [ACTION: dangeract, foo]")
        # Deferred — NOT executed; queued on _pending_confirmation.
        self.assertEqual(ran, [])
        self.assertEqual(cleaned, "Okay.")
        self.assertEqual(len(results), 1)
        self.assertIn("REQUIRES CONFIRMATION", results[0][1])
        self.assertIn(("dangeract", "foo"), list(bc._pending_confirmation))

    def test_pushback_replaces_prose_and_defers(self):
        bc = self.bc
        ran = []
        self._with_action("run_shell", lambda a: ran.append(a) or "shell")
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "PUSHBACK_ENABLED", True):
            cleaned, results = bc.parse_and_run_actions(
                "Right away, sir. [ACTION: run_shell, rm -rf node_modules]")
        # The LLM's prose is dropped; the objection becomes the spoken reply.
        self.assertEqual(ran, [])
        self.assertIn("inadvisable", cleaned)
        self.assertIn("PUSHBACK", results[0][1])
        self.assertIn(("run_shell", "rm -rf node_modules"),
                      list(bc._pending_confirmation))

    def test_hallucinated_claim_appended_when_no_action(self):
        cleaned, results = self.bc.parse_and_run_actions("Restarting now, sir.")
        self.assertEqual(cleaned, "Restarting now, sir.")
        self.assertEqual(len(results), 1)
        name, msg, informative = results[0]
        self.assertEqual(name, "_unverified_claim")
        self.assertTrue(informative)
        self.assertIn("hallucinated execution", msg)

    def test_preemptive_refuse_drops_reply(self):
        # "panning the camera" maps to a None action -> refuse before TTS.
        cleaned, results = self.bc.parse_and_run_actions(
            "Panning the camera to the left now, sir.")
        self.assertEqual(cleaned, "")
        self.assertEqual(results[0][0], "_preemptive_hallucinated_claim")
        self.assertIn("refused before TTS", results[0][1])

    def test_preemptive_inject_adds_token_and_runs(self):
        bc = self.bc
        ran = []
        # ambient_mode_on must exist for the inject branch (vs refuse).
        self._with_action("ambient_mode_on", lambda a: ran.append(a) or "on")
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None):
            cleaned, results = bc.parse_and_run_actions(
                "Switching to ambient mode, sir.")
        # The missing [ACTION: ambient_mode_on] was injected and executed.
        self.assertEqual(ran, [""])
        self.assertEqual(results[0][0], "ambient_mode_on")

    def test_action_exception_yields_failure_result(self):
        bc = self.bc

        def boom(_arg):
            raise ValueError("kaboom")

        self._with_action("boomer", boom)
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None):
            cleaned, results = bc.parse_and_run_actions("[ACTION: boomer, x]")
        name, msg, informative = results[0]
        self.assertEqual(name, "boomer")
        # The result string keeps the word "failed" (for the _is_failure route).
        self.assertIn("failed", msg.lower())
        self.assertFalse(informative)

    def test_action_exception_files_scrubbed_bug_report(self):
        # The dispatcher's except block also self-files a scrubbed, rate-limited
        # bug report (core.bug_reporter.auto_capture) for the failing action.
        bc = self.bc
        import core.bug_reporter as br

        def boom(_arg):
            raise ValueError("kaboom")

        self._with_action("boomer2", boom)
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.dict(bc.os.environ, {"JARVIS_BUG_AUTO_CAPTURE": "1"}), \
             mock.patch.object(br, "auto_capture") as cap:
            bc.parse_and_run_actions("[ACTION: boomer2, x]")
        cap.assert_called_once()
        self.assertIsInstance(cap.call_args.args[0], ValueError)
        self.assertEqual(cap.call_args.kwargs.get("where"), "boomer2")

    def test_action_exception_bug_report_suppressible(self):
        # JARVIS_BUG_AUTO_CAPTURE=0 disables the self-file hook.
        bc = self.bc
        import core.bug_reporter as br

        def boom(_arg):
            raise ValueError("kaboom")

        self._with_action("boomer3", boom)
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.dict(bc.os.environ, {"JARVIS_BUG_AUTO_CAPTURE": "0"}), \
             mock.patch.object(br, "auto_capture") as cap:
            bc.parse_and_run_actions("[ACTION: boomer3, x]")
        cap.assert_not_called()

    def test_action_exception_auto_submits_when_enabled(self):
        # With both JARVIS_BUG_AUTO_CAPTURE and JARVIS_BUG_AUTO_SUBMIT on, the
        # captured report is also POSTed via the API.
        bc = self.bc
        import core.bug_reporter as br

        def boom(_arg):
            raise ValueError("kaboom")

        self._with_action("boomer4", boom)
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.dict(bc.os.environ, {"JARVIS_BUG_AUTO_CAPTURE": "1",
                                             "JARVIS_BUG_AUTO_SUBMIT": "1"}), \
             mock.patch.object(br, "auto_capture", return_value={"kind": "auto"}), \
             mock.patch.object(br, "api_submit_issue") as sub:
            bc.parse_and_run_actions("[ACTION: boomer4, x]")
        sub.assert_called_once()

    def test_dropped_step_appended_after_real_action(self):
        bc = self.bc
        # Run a real registered informative action AND promise a see_screen read
        # that's never emitted -> _dropped_step synthetic result.
        self._with_action("testecho", lambda a: "ok", informative=True)
        self.assertIn("see_screen", bc.ACTIONS)
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None):
            cleaned, results = bc.parse_and_run_actions(
                "[ACTION: testecho, go] I'll then read it to you.")
        names = [r[0] for r in results]
        self.assertIn("testecho", names)
        self.assertIn("_dropped_step", names)

    def test_autocorrect_silent_reroutes_unknown_action(self):
        bc = self.bc
        ran = []
        self._with_action("real_target", lambda a: ran.append(a) or "ok")
        # Fake the autocorrect layer: a typo cleanly resolves to real_target.
        fake_ac = types.SimpleNamespace(
            autocorrect_command_choice=lambda name, keys, **kw: {
                "status": "silent",
                "primary": ("real_target", 0.95),
                "secondary": None,
            })
        with mock.patch.object(bc, "_cmd_autocorrect", fake_ac), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None):
            cleaned, results = bc.parse_and_run_actions(
                "[ACTION: realtarget, hi]")
        # The misspelled token was routed to real_target and executed.
        self.assertEqual(ran, ["hi"])
        self.assertEqual(results[0][0], "real_target")

    def test_autocorrect_ambiguous_defers_with_choice(self):
        bc = self.bc
        self._with_action("opt_a", lambda a: "A")
        self._with_action("opt_b", lambda a: "B")
        fake_ac = types.SimpleNamespace(
            autocorrect_command_choice=lambda name, keys, **kw: {
                "status": "ambiguous",
                "primary": ("opt_a", 0.80),
                "secondary": ("opt_b", 0.78),
            })
        with mock.patch.object(bc, "_cmd_autocorrect", fake_ac):
            bc._pending_autocorrect_choice.clear()
            cleaned, results = bc.parse_and_run_actions("[ACTION: opt, x]")
        self.assertIn("AMBIGUOUS", results[0][1])
        # The disambiguation choice was queued for the next utterance.
        self.assertEqual(len(bc._pending_autocorrect_choice), 1)
        queued = bc._pending_autocorrect_choice[0]
        self.assertEqual(queued["primary"], ("opt_a", "x"))
        self.assertEqual(queued["secondary"], ("opt_b", "x"))

    def test_draft_preview_gate_routes_send_action(self):
        bc = self.bc

        class FakeGate:
            def should_gate(self, name):
                return name == "send_email"

            def run_with_gate(self, name, arg, fn):
                return "GATED:" + fn(arg)

        self._with_action("send_email", lambda a: "sent-" + a)
        with mock.patch.object(bc, "_draft_preview_gate", FakeGate()), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None):
            cleaned, results = bc.parse_and_run_actions(
                "[ACTION: send_email, hi]")
        self.assertEqual(results, [("send_email", "GATED:sent-hi", False)])

    def test_mission_narration_speaks_intro_when_chained(self):
        bc = self.bc
        # 3 simple non-confirm actions -> narration threshold (3) reached.
        self._with_action("act_one", lambda a: "1")
        # Reuse act_one for all three tokens; capture spoken cues.
        spoken = []
        with mock.patch.object(bc, "MISSION_NARRATION_ENABLED", True), \
             mock.patch.object(bc, "MISSION_NARRATION_THRESHOLD", 3), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: spoken.append(a)):
            cleaned, results = bc.parse_and_run_actions(
                "[ACTION: act_one, a] [ACTION: act_one, b] [ACTION: act_one, c]")
        # Narration fired: intro spoken, and the cleaned prose is dropped.
        self.assertEqual(cleaned, "")
        self.assertTrue(any("steps, sir." in c[0] for c in spoken))
        self.assertEqual(len(results), 3)


# ════════════════════════════════════════════════════════════════════════════
#  get_followup_response
# ════════════════════════════════════════════════════════════════════════════
class GetFollowupResponseTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_last_voice_route", [{"addendum": ""}])
        self._p(bc, "_last_user_tone", [None])

    def test_claude_via_llm_client(self):
        bc = self.bc

        class FakeClient:
            def __init__(self):
                self.kwargs = None

            def complete(self, **kwargs):
                self.kwargs = kwargs
                return "Follow-up answer, sir."

        client = FakeClient()
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "_llm_client", client):
            out = bc.get_followup_response([("get_time", "3pm")])
        self.assertEqual(out, "Follow-up answer, sir.")
        # The action-results summary is woven into the user message.
        msgs = client.kwargs["messages"]
        self.assertIn("get_time", msgs[-1]["content"])
        self.assertIn("3pm", msgs[-1]["content"])

    def test_cloud_failure_falls_back_to_local(self):
        bc = self.bc
        import anthropic
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "_llm_client", None), \
             mock.patch.object(anthropic, "Anthropic",
                               side_effect=Exception("cap hit")), \
             mock.patch.object(bc, "_call_local_llm",
                               lambda *a, **k: "local reply"):
            out = bc.get_followup_response([("x", "y")])
        self.assertEqual(out, "local reply")

    def test_cloud_and_local_both_fail_returns_empty(self):
        bc = self.bc
        import anthropic
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "_llm_client", None), \
             mock.patch.object(anthropic, "Anthropic",
                               side_effect=Exception("down")), \
             mock.patch.object(bc, "_call_local_llm", lambda *a, **k: ""):
            out = bc.get_followup_response([("x", "y")])
        self.assertEqual(out, "")


# ════════════════════════════════════════════════════════════════════════════
#  handle_confirmation_response
# ════════════════════════════════════════════════════════════════════════════
class HandleConfirmationResponseTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_speak", lambda *a, **k: None)

    def test_no_pending_returns_false(self):
        self.bc._pending_confirmation.clear()
        self.assertFalse(self.bc.handle_confirmation_response("yes"))

    def test_affirmative_runs_pending(self):
        bc = self.bc
        ran = []
        acts = dict(bc.ACTIONS)
        acts["cfa"] = lambda a: ran.append(a) or "ok"
        with mock.patch.object(bc, "ACTIONS", acts):
            bc._pending_confirmation.clear()
            bc._pending_confirmation.append(("cfa", "X"))
            consumed = bc.handle_confirmation_response("yes please")
        self.assertTrue(consumed)
        self.assertEqual(ran, ["X"])
        self.assertEqual(list(bc._pending_confirmation), [])

    def test_negative_cancels_pending(self):
        bc = self.bc
        ran = []
        acts = dict(bc.ACTIONS)
        acts["cfa"] = lambda a: ran.append(a) or "ok"
        with mock.patch.object(bc, "ACTIONS", acts):
            bc._pending_confirmation.clear()
            bc._pending_confirmation.append(("cfa", "Y"))
            consumed = bc.handle_confirmation_response("no thanks")
        self.assertTrue(consumed)
        self.assertEqual(ran, [])  # not executed
        self.assertEqual(list(bc._pending_confirmation), [])  # cleared

    def test_action_exception_does_not_propagate(self):
        bc = self.bc

        def boom(_a):
            raise RuntimeError("nope")

        acts = dict(bc.ACTIONS)
        acts["boomer"] = boom
        with mock.patch.object(bc, "ACTIONS", acts):
            bc._pending_confirmation.clear()
            bc._pending_confirmation.append(("boomer", "z"))
            consumed = bc.handle_confirmation_response("yes")
        self.assertTrue(consumed)
        self.assertEqual(list(bc._pending_confirmation), [])


# ════════════════════════════════════════════════════════════════════════════
#  handle_autocorrect_disambig_response
# ════════════════════════════════════════════════════════════════════════════
class HandleAutocorrectDisambigTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_speak", lambda *a, **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self.ran = []
        acts = dict(bc.ACTIONS)
        acts["alpha"] = lambda a: self.ran.append(("alpha", a)) or "A"
        acts["beta"] = lambda a: self.ran.append(("beta", a)) or "B"
        self._p(bc, "ACTIONS", acts)

    def _queue(self, primary=("alpha", "q"), secondary=("beta", "q")):
        self.bc._pending_autocorrect_choice.clear()
        self.bc._pending_autocorrect_choice.append(
            {"primary": primary, "secondary": secondary, "original": "alfa"})

    def test_no_pending_returns_false(self):
        self.bc._pending_autocorrect_choice.clear()
        self.assertFalse(
            self.bc.handle_autocorrect_disambig_response("alpha"))

    def test_named_primary(self):
        self._queue()
        self.assertTrue(self.bc.handle_autocorrect_disambig_response("alpha"))
        self.assertEqual(self.ran, [("alpha", "q")])
        self.assertEqual(list(self.bc._pending_autocorrect_choice), [])

    def test_named_secondary(self):
        self._queue()
        self.assertTrue(self.bc.handle_autocorrect_disambig_response("beta"))
        self.assertEqual(self.ran, [("beta", "q")])

    def test_affirmative_picks_primary(self):
        self._queue()
        self.assertTrue(
            self.bc.handle_autocorrect_disambig_response("first one"))
        self.assertEqual(self.ran, [("alpha", "q")])

    def test_destructive_pick_refused_not_run(self):
        # A guessed disambig pick that resolves to a destructive wipe must be
        # REFUSED (handled, but NOT executed) so a plain confirmation cannot
        # erase memory/tasks. Guards reset_memory / forget_last_hour / clear_tasks.
        bc = self.bc
        acts = dict(bc.ACTIONS)
        acts["reset_memory"] = lambda a: self.ran.append(("reset_memory", a)) or "wiped"
        self._p(bc, "ACTIONS", acts)
        self._queue(primary=("reset_memory", "x"), secondary=("beta", "q"))
        self.assertTrue(
            self.bc.handle_autocorrect_disambig_response("first one"))
        self.assertEqual(self.ran, [])

    def test_explicit_second_keyword(self):
        self._queue()
        self.assertTrue(self.bc.handle_autocorrect_disambig_response("second"))
        self.assertEqual(self.ran, [("beta", "q")])

    def test_negative_cancels_and_consumes(self):
        self._queue()
        self.assertTrue(self.bc.handle_autocorrect_disambig_response("neither"))
        self.assertEqual(self.ran, [])
        self.assertEqual(list(self.bc._pending_autocorrect_choice), [])

    def test_unrelated_reply_does_not_consume(self):
        self._queue()
        # Returns False (fall through to normal dispatch) and clears the queue.
        self.assertFalse(
            self.bc.handle_autocorrect_disambig_response("what time is it"))
        self.assertEqual(self.ran, [])
        self.assertEqual(list(self.bc._pending_autocorrect_choice), [])

    def test_destructive_pick_refused_without_confirmation(self):
        bc = self.bc
        dr = sorted(bc._DESTRUCTIVE_REPLAY_ACTIONS)[0]
        ran = []
        acts = dict(bc.ACTIONS)
        acts[dr] = lambda a: ran.append(a) or "ran"
        with mock.patch.object(bc, "ACTIONS", acts):
            bc._pending_autocorrect_choice.clear()
            bc._pending_autocorrect_choice.append(
                {"primary": (dr, "q"), "secondary": ("beta", "q"),
                 "original": "x"})
            consumed = bc.handle_autocorrect_disambig_response("yes")
        self.assertTrue(consumed)
        self.assertEqual(ran, [])  # refused — destructive picks need re-issue

    def test_picked_action_vanished_is_handled(self):
        bc = self.bc
        # primary names an action not present in ACTIONS -> clean bail.
        self._queue(primary=("ghost_action", "q"))
        consumed = bc.handle_autocorrect_disambig_response("first")
        self.assertTrue(consumed)
        self.assertEqual(self.ran, [])


# ════════════════════════════════════════════════════════════════════════════
#  _apply_quip_layer
# ════════════════════════════════════════════════════════════════════════════
class ApplyQuipLayerTests(SectionSixBase):
    def test_no_layer_returns_input(self):
        with mock.patch.object(self.bc, "_tts_layer", None):
            self.assertEqual(
                self.bc._apply_quip_layer("Done.", [("get_time", "x", True)]),
                "Done.")

    def test_empty_text_passthrough(self):
        layer = types.SimpleNamespace(
            jarvis_quip_layer=lambda t, p: t + "!")
        with mock.patch.object(self.bc, "_tts_layer", layer):
            self.assertEqual(
                self.bc._apply_quip_layer("", [("get_time", "x", True)]), "")

    def test_layer_applied_with_primary_action(self):
        seen = {}

        def quip(text, primary):
            seen["primary"] = primary
            return text + f" [{primary}]"

        layer = types.SimpleNamespace(jarvis_quip_layer=quip)
        with mock.patch.object(self.bc, "_tts_layer", layer):
            out = self.bc._apply_quip_layer(
                "Done.", [("get_time", "x", True)])
        self.assertEqual(out, "Done. [get_time]")
        self.assertEqual(seen["primary"], "get_time")

    def test_underscore_actions_skipped_for_primary(self):
        # Synthetic _-prefixed results must not be chosen as the quip category.
        captured = {}

        def quip(text, primary):
            captured["primary"] = primary
            return text

        layer = types.SimpleNamespace(jarvis_quip_layer=quip)
        with mock.patch.object(self.bc, "_tts_layer", layer):
            self.bc._apply_quip_layer(
                "Done.",
                [("_unverified_claim", "x", True), ("get_time", "y", True)])
        self.assertEqual(captured["primary"], "get_time")

    def test_fire_and_exit_action_skips_layer(self):
        bc = self.bc
        fa = sorted(bc._FIRE_AND_EXIT_ACTIONS)[0]
        layer = types.SimpleNamespace(
            jarvis_quip_layer=lambda t, p: t + " QUIP")
        with mock.patch.object(bc, "_tts_layer", layer):
            out = bc._apply_quip_layer("Restarting, sir.", [(fa, "x", True)])
        self.assertEqual(out, "Restarting, sir.")  # untouched

    def test_layer_exception_returns_original(self):
        def boom(_t, _p):
            raise RuntimeError("quip down")

        layer = types.SimpleNamespace(jarvis_quip_layer=boom)
        with mock.patch.object(self.bc, "_tts_layer", layer):
            out = self.bc._apply_quip_layer("Done.", [("get_time", "x", True)])
        self.assertEqual(out, "Done.")


# ════════════════════════════════════════════════════════════════════════════
#  _speak
# ════════════════════════════════════════════════════════════════════════════
class SpeakTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        # Defeat the 30s boot-window throttle so _speak never time.sleeps.
        self._p(bc, "_session_start_time", 0.0)
        self._p(bc, "_write_hud_state", lambda **k: None)

    def test_muted_short_circuits_before_synth(self):
        bc = self.bc
        with mock.patch.object(bc, "_tts_muted", [True]), \
             mock.patch.object(bc, "synthesise",
                               side_effect=AssertionError("must not synth")):
            self.assertIsNone(bc._speak("hello sir"))

    def test_empty_after_strip_skips_synth(self):
        bc = self.bc
        with mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise",
                               side_effect=AssertionError("must not synth")):
            # Whitespace-only -> nothing audible -> early return.
            self.assertIsNone(bc._speak("   "))

    def test_staging_records_reply_without_audio(self):
        bc = self.bc
        recorded = []
        fake_stg = types.ModuleType("staging_instance")
        fake_stg.record_reply = lambda text, **k: recorded.append((text, k))
        with mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: True), \
             mock.patch.dict(sys.modules, {"staging_instance": fake_stg}), \
             mock.patch.object(bc, "synthesise",
                               side_effect=AssertionError("must not synth")):
            bc._speak("Hello, sir.")
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0][0], "Hello, sir.")
        self.assertEqual(recorded[0][1]["kind"], "tts")

    def test_full_path_synthesises_and_plays(self):
        bc = self.bc
        import numpy as np
        synth_args, play_args, states = [], [], []
        with mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(
                 bc, "synthesise",
                 lambda t: (synth_args.append(t)
                            or np.zeros(8, dtype=np.float32), 22050)), \
             mock.patch.object(
                 bc, "play_with_lipsync",
                 lambda a, sr: play_args.append((len(a), sr))), \
             mock.patch.object(bc, "set_state", lambda s: states.append(s)):
            bc._speak("Speak this line.")
        self.assertEqual(synth_args, ["Speak this line."])
        self.assertEqual(play_args, [(8, 22050)])
        self.assertEqual(states, ["speaking", "idle"])

    def test_intent_tag_stripped_before_synth(self):
        bc = self.bc
        import numpy as np
        synth_args = []
        with mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(
                 bc, "synthesise",
                 lambda t: (synth_args.append(t)
                            or np.zeros(4, dtype=np.float32), 16000)), \
             mock.patch.object(bc, "play_with_lipsync", lambda a, sr: None), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("[intent:urgent] Look out, sir.")
        # The leading [intent:...] tag is never passed to synthesis.
        self.assertEqual(len(synth_args), 1)
        self.assertNotIn("intent", synth_args[0].lower())
        self.assertIn("Look out", synth_args[0])

    def test_wry_and_mood_tags_stripped_before_synth(self):
        bc = self.bc
        import numpy as np
        synth_args = []
        sink = (lambda t: (synth_args.append(t)
                           or np.zeros(4, dtype=np.float32), 16000))
        with mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise", sink), \
             mock.patch.object(bc, "play_with_lipsync", lambda a, sr: None), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("[wry] Oh, splendid.")
            bc._speak("[mood:dry_amused] As you wish, sir.")
        self.assertEqual(synth_args, ["Oh, splendid.", "As you wish, sir."])

    def test_volume_scale_applied_without_error(self):
        bc = self.bc
        import numpy as np
        played = []
        sink = lambda t: (np.ones(6, dtype=np.float32), 16000)
        with mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise", sink), \
             mock.patch.object(bc, "play_with_lipsync",
                               lambda a, sr: played.append(a)), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("Quietly, sir.", volume_scale=0.5)
        # Audio was attenuated in place before playback.
        self.assertEqual(len(played), 1)
        self.assertTrue((played[0] <= 1.0).all())

    def test_synthesis_error_recovers_to_idle(self):
        bc = self.bc
        states = []
        with mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise",
                               side_effect=RuntimeError("device gone")), \
             mock.patch.object(bc, "set_state", lambda s: states.append(s)):
            # Must not propagate; recovers to idle.
            bc._speak("Anything.")
        self.assertIn("idle", states)
        # The playback-active guard is forced back to False on the way out.
        self.assertFalse(bc._tts_playback_active[0])


# ════════════════════════════════════════════════════════════════════════════
#  Power-plan helpers
# ════════════════════════════════════════════════════════════════════════════
class PowerPlanTests(SectionSixBase):
    def test_get_active_plan_parses_guid(self):
        bc = self.bc
        out = ("Power Scheme GUID: "
               "11111111-2222-3333-4444-555555555555  (Balanced)")
        with mock.patch.object(bc.subprocess, "check_output", return_value=out):
            self.assertEqual(bc._get_active_power_plan_guid(),
                             "11111111-2222-3333-4444-555555555555")

    def test_get_active_plan_failure_returns_none(self):
        bc = self.bc
        with mock.patch.object(bc.subprocess, "check_output",
                               side_effect=Exception("powercfg gone")):
            self.assertIsNone(bc._get_active_power_plan_guid())

    def test_get_active_plan_no_guid_in_output(self):
        bc = self.bc
        with mock.patch.object(bc.subprocess, "check_output",
                               return_value="no guid here"):
            self.assertIsNone(bc._get_active_power_plan_guid())

    def test_set_power_plan_success(self):
        bc = self.bc
        with mock.patch.object(bc.subprocess, "run", return_value=None) as run:
            self.assertTrue(bc._set_power_plan("abc-guid"))
        run.assert_called_once()

    def test_set_power_plan_failure(self):
        bc = self.bc
        with mock.patch.object(bc.subprocess, "run",
                               side_effect=Exception("denied")):
            self.assertFalse(bc._set_power_plan("abc-guid"))

    def test_activate_high_perf_switches_and_remembers(self):
        bc = self.bc
        orig = bc._prior_power_plan_guid
        self.addCleanup(setattr, bc, "_prior_power_plan_guid", orig)
        prior_guid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with mock.patch.object(bc, "_get_active_power_plan_guid",
                               return_value=prior_guid), \
             mock.patch.object(bc, "_set_power_plan", return_value=True):
            bc._activate_high_performance_plan()
        self.assertEqual(bc._prior_power_plan_guid, prior_guid)

    def test_activate_high_perf_already_active_noop(self):
        bc = self.bc
        orig = bc._prior_power_plan_guid
        self.addCleanup(setattr, bc, "_prior_power_plan_guid", orig)
        with mock.patch.object(bc, "_get_active_power_plan_guid",
                               return_value=bc._HIGH_PERF_GUID), \
             mock.patch.object(bc, "_set_power_plan",
                               side_effect=AssertionError("must not switch")):
            bc._activate_high_performance_plan()
        self.assertEqual(bc._prior_power_plan_guid, bc._HIGH_PERF_GUID)

    def test_restore_prior_plan_when_set(self):
        bc = self.bc
        orig = bc._prior_power_plan_guid
        self.addCleanup(setattr, bc, "_prior_power_plan_guid", orig)
        prior_guid = "99999999-8888-7777-6666-555555555555"
        bc._prior_power_plan_guid = prior_guid
        with mock.patch.object(bc, "_set_power_plan",
                               return_value=True) as setp:
            bc._restore_prior_power_plan()
        setp.assert_called_once_with(prior_guid)

    def test_restore_noop_when_unset(self):
        bc = self.bc
        orig = bc._prior_power_plan_guid
        self.addCleanup(setattr, bc, "_prior_power_plan_guid", orig)
        bc._prior_power_plan_guid = None
        with mock.patch.object(bc, "_set_power_plan",
                               side_effect=AssertionError("must not run")):
            bc._restore_prior_power_plan()  # no exception == pass


# ════════════════════════════════════════════════════════════════════════════
#  cuBLAS preflight helpers
# ════════════════════════════════════════════════════════════════════════════
class CublasPreflightTests(SectionSixBase):
    def test_find_cublas_dll_in_site_packages(self):
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            bindir = os.path.join(d, "nvidia", "cublas", "bin")
            os.makedirs(bindir)
            dll = os.path.join(bindir, bc._CUBLAS_DLL_NAME)
            open(dll, "w").close()
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: [d]
            fake_site.getusersitepackages = lambda: ""
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ, {"PATH": "", "CUDA_PATH": ""},
                                 clear=False):
                self.assertEqual(bc._find_cublas_dll(), dll)

    def test_find_cublas_dll_on_path(self):
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            dll = os.path.join(d, bc._CUBLAS_DLL_NAME)
            open(dll, "w").close()
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: []
            fake_site.getusersitepackages = lambda: ""
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": d, "CUDA_PATH": ""}, clear=False):
                self.assertEqual(bc._find_cublas_dll(), dll)

    def test_find_cublas_dll_via_cuda_path(self):
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            bindir = os.path.join(d, "bin")
            os.makedirs(bindir)
            dll = os.path.join(bindir, bc._CUBLAS_DLL_NAME)
            open(dll, "w").close()
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: []
            fake_site.getusersitepackages = lambda: ""
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": "", "CUDA_PATH": d}, clear=False):
                self.assertEqual(bc._find_cublas_dll(), dll)

    def test_find_cublas_dll_via_program_files_glob(self):
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            # Build the exact NVIDIA Toolkit layout the step-4 glob expects.
            bindir = os.path.join(d, "NVIDIA GPU Computing Toolkit",
                                  "CUDA", "v12.4", "bin")
            os.makedirs(bindir)
            dll = os.path.join(bindir, bc._CUBLAS_DLL_NAME)
            open(dll, "w").close()
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: []
            fake_site.getusersitepackages = lambda: ""
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": "", "CUDA_PATH": "",
                                  "ProgramFiles": d,
                                  "ProgramFiles(x86)": ""}, clear=False):
                self.assertEqual(bc._find_cublas_dll(), dll)

    def test_find_cublas_dll_absent_returns_none(self):
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: [d]
            fake_site.getusersitepackages = lambda: ""
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": d, "CUDA_PATH": "",
                                  "ProgramFiles": d, "ProgramFiles(x86)": d},
                                 clear=False):
                self.assertIsNone(bc._find_cublas_dll())

    def test_ctranslate2_sees_cuda_true(self):
        bc = self.bc
        fake_ct2 = types.ModuleType("ctranslate2")
        fake_ct2.get_cuda_device_count = lambda: 1
        with mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
            self.assertTrue(bc._ctranslate2_sees_cuda())

    def test_ctranslate2_sees_cuda_false_on_zero(self):
        bc = self.bc
        fake_ct2 = types.ModuleType("ctranslate2")
        fake_ct2.get_cuda_device_count = lambda: 0
        with mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
            self.assertFalse(bc._ctranslate2_sees_cuda())

    def test_ctranslate2_import_error_false(self):
        bc = self.bc
        fake_ct2 = types.ModuleType("ctranslate2")

        def boom():
            raise RuntimeError("no cuda runtime")

        fake_ct2.get_cuda_device_count = boom
        with mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
            self.assertFalse(bc._ctranslate2_sees_cuda())

    def test_preflight_cublas_found_returns_true(self):
        bc = self.bc
        with mock.patch.object(bc, "_find_cublas_dll",
                               return_value=r"C:\fake\cublas64_12.dll"):
            self.assertTrue(bc._preflight_cublas_check())

    def test_preflight_cublas_missing_no_gpu_returns_false(self):
        bc = self.bc
        orig = bc._force_whisper_cpu_int8
        self.addCleanup(setattr, bc, "_force_whisper_cpu_int8", orig)
        with mock.patch.object(bc, "_find_cublas_dll", return_value=None), \
             mock.patch.object(bc, "_ctranslate2_sees_cuda", return_value=False):
            self.assertFalse(bc._preflight_cublas_check())

    def test_preflight_cublas_missing_with_gpu_forces_cpu(self):
        bc = self.bc
        orig = bc._force_whisper_cpu_int8
        self.addCleanup(setattr, bc, "_force_whisper_cpu_int8", orig)
        bc._force_whisper_cpu_int8 = False
        # The function tries to import overnight_upgrade to queue a task; fake
        # it so no real file write / import side effect occurs.
        fake_ou = types.ModuleType("overnight_upgrade")
        fake_ou._append_tasks = lambda tasks: 1
        with mock.patch.object(bc, "_find_cublas_dll", return_value=None), \
             mock.patch.object(bc, "_ctranslate2_sees_cuda",
                               return_value=True), \
             mock.patch.dict(sys.modules, {"overnight_upgrade": fake_ou}):
            self.assertFalse(bc._preflight_cublas_check())
        self.assertTrue(bc._force_whisper_cpu_int8)


# ════════════════════════════════════════════════════════════════════════════
#  _preflight_api_key
# ════════════════════════════════════════════════════════════════════════════
class PreflightApiKeyTests(SectionSixBase):
    def test_non_claude_backend_skipped(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"):
            self.assertEqual(self.bc._preflight_api_key(), (True, ""))

    def test_missing_key_reported(self):
        bc = self.bc
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            ok, reason = bc._preflight_api_key()
        self.assertFalse(ok)
        self.assertIn("ANTHROPIC_API_KEY", reason)

    def test_successful_ping(self):
        bc = self.bc
        fake_anthropic = types.ModuleType("anthropic")

        class FakeClient:
            def __init__(self, *a, **k):
                self.messages = self

            def create(self, **k):
                return types.SimpleNamespace(content=[])

        fake_anthropic.Anthropic = FakeClient
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(os.environ,
                             {"ANTHROPIC_API_KEY": "sk-test"}, clear=False), \
             mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            ok, reason = bc._preflight_api_key(timeout_sec=2.0)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_ping_timeout_reported(self):
        bc = self.bc
        import threading as _t
        release = _t.Event()
        self.addCleanup(release.set)  # never strand the worker thread
        fake_anthropic = types.ModuleType("anthropic")

        class FakeClient:
            def __init__(self, *a, **k):
                self.messages = self

            def create(self, **k):
                # Block past the join window so the preflight reports a timeout.
                release.wait(timeout=5.0)
                return types.SimpleNamespace(content=[])

        fake_anthropic.Anthropic = FakeClient
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(os.environ,
                             {"ANTHROPIC_API_KEY": "sk-test"}, clear=False), \
             mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            ok, reason = bc._preflight_api_key(timeout_sec=0.2)
        self.assertFalse(ok)
        self.assertIn("timed out", reason)

    def test_ping_failure_reported(self):
        bc = self.bc
        fake_anthropic = types.ModuleType("anthropic")

        class FakeClient:
            def __init__(self, *a, **k):
                self.messages = self

            def create(self, **k):
                raise RuntimeError("401 unauthorized")

        fake_anthropic.Anthropic = FakeClient
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(os.environ,
                             {"ANTHROPIC_API_KEY": "sk-test"}, clear=False), \
             mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            ok, reason = bc._preflight_api_key(timeout_sec=2.0)
        self.assertFalse(ok)
        self.assertIn("ping failed", reason)


# ════════════════════════════════════════════════════════════════════════════
#  _drain_injected_command
# ════════════════════════════════════════════════════════════════════════════
class DrainInjectedCommandTests(SectionSixBase):
    def _redirect(self, path):
        self._p(self.bc, "INJECTED_COMMANDS_PATH", path)

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self._redirect(os.path.join(d, "inject.json"))
            self.assertIsNone(self.bc._drain_injected_command())

    def test_pops_head_and_requeues_tail(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["first cmd", "second cmd"], f)
            self.assertEqual(self.bc._drain_injected_command(), "first cmd")
            # Tail persisted for the next pass.
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), ["second cmd"])
            self.assertEqual(self.bc._drain_injected_command(), "second cmd")
            # Now drained; file removed.
            self.assertFalse(os.path.exists(path))

    def test_three_items_requeues_two(self):
        # Exercises the tempfile-backed atomic tail-requeue write path.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["a", "b", "c"], f)
            self.assertEqual(self.bc._drain_injected_command(), "a")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), ["b", "c"])

    def test_dict_item_text_extracted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"text": "hello world"}], f)
            self.assertEqual(self.bc._drain_injected_command(), "hello world")

    def test_corrupt_json_discarded(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            self.assertIsNone(self.bc._drain_injected_command())
            # Snapshot dropped so the same garbage can't re-trip.
            self.assertFalse(os.path.exists(path))

    def test_empty_list_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
            self.assertIsNone(self.bc._drain_injected_command())


# ════════════════════════════════════════════════════════════════════════════
#  _speak_pending
# ════════════════════════════════════════════════════════════════════════════
class SpeakPendingTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoke = []
        self._p(bc, "_speak", lambda msg, **k: self.spoke.append(msg))
        self._p(bc, "_mark_speech_spoken", lambda m: None)
        self._p(bc, "_speech_was_recently_spoken", lambda m: False)

    def _redirect(self, path):
        self._p(self.bc, "PENDING_SPEECH_PATH", path)

    def test_missing_file_returns_false(self):
        with tempfile.TemporaryDirectory() as d:
            self._redirect(os.path.join(d, "pending.json"))
            self.assertFalse(self.bc._speak_pending())

    def test_speaks_each_reminder_and_dedupes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "one"}, {"message": "two"},
                           {"message": "one"}], f)
            self.assertTrue(self.bc._speak_pending())
        # The duplicate "one" within the batch is suppressed.
        self.assertEqual(self.spoke, ["one", "two"])

    def test_recently_spoken_suppressed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "stale"}], f)
            with mock.patch.object(self.bc, "_speech_was_recently_spoken",
                                   lambda m: True):
                result = self.bc._speak_pending()
        # Nothing spoken (suppressed) -> returns False.
        self.assertFalse(result)
        self.assertEqual(self.spoke, [])

    def test_corrupt_queue_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write("not json at all")
            self.assertFalse(self.bc._speak_pending())
            self.assertFalse(os.path.exists(path))

    def test_speak_exception_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "boom"}], f)
            with mock.patch.object(self.bc, "_speak",
                                   side_effect=RuntimeError("tts dead")):
                # Must swallow the TTS failure and not propagate.
                self.bc._speak_pending()
            self.assertFalse(os.path.exists(path))


# ════════════════════════════════════════════════════════════════════════════
#  Orchestrator gate
# ════════════════════════════════════════════════════════════════════════════
class OrchestratorGateTests(SectionSixBase):
    def test_enabled_via_config_flag(self):
        with mock.patch.object(self.bc, "ENABLE_ORCHESTRATOR", True):
            self.assertTrue(self.bc._orchestrator_enabled())

    def test_enabled_via_env_override(self):
        bc = self.bc
        with mock.patch.object(bc, "ENABLE_ORCHESTRATOR", False), \
             mock.patch.dict(os.environ,
                             {"JARVIS_ENABLE_ORCHESTRATOR": "1"}, clear=False):
            self.assertTrue(bc._orchestrator_enabled())

    def test_disabled_when_off_everywhere(self):
        bc = self.bc
        with mock.patch.object(bc, "ENABLE_ORCHESTRATOR", False), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_ENABLE_ORCHESTRATOR", None)
            self.assertFalse(bc._orchestrator_enabled())

    def test_undefined_config_flag_falls_back_to_env(self):
        # 13086-13087: if the ENABLE_ORCHESTRATOR global is undefined the bare
        # `if ENABLE_ORCHESTRATOR` raises NameError, which is caught so the
        # function falls through to the env override. Delete the attribute for
        # the duration of the test and restore it afterwards.
        bc = self.bc
        had = hasattr(bc, "ENABLE_ORCHESTRATOR")
        saved = getattr(bc, "ENABLE_ORCHESTRATOR", None)
        if had:
            delattr(bc, "ENABLE_ORCHESTRATOR")
        try:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("JARVIS_ENABLE_ORCHESTRATOR", None)
                self.assertFalse(bc._orchestrator_enabled())
                os.environ["JARVIS_ENABLE_ORCHESTRATOR"] = "1"
                self.assertTrue(bc._orchestrator_enabled())
        finally:
            if had:
                bc.ENABLE_ORCHESTRATOR = saved

    def test_is_orchestration_request_positive(self):
        for t in ("morning briefing", "orchestrate this",
                  "give me the daily rundown", "system brief"):
            self.assertTrue(self.bc._is_orchestration_request(t), t)

    def test_is_orchestration_request_too_long(self):
        # >9 words -> never an orchestration request even if it matches.
        self.assertFalse(self.bc._is_orchestration_request(
            "please could you kindly give me the full morning briefing now ok"))

    def test_is_orchestration_request_empty_and_unrelated(self):
        self.assertFalse(self.bc._is_orchestration_request(""))
        self.assertFalse(
            self.bc._is_orchestration_request("what is the weather"))

    def test_maybe_orchestrate_disabled_returns_false(self):
        # Force disabled so it never fans out to real sub-agents / LLMs.
        with mock.patch.object(self.bc, "_orchestrator_enabled",
                               return_value=False):
            self.assertFalse(self.bc._maybe_orchestrate("morning briefing"))

    def test_maybe_orchestrate_not_a_request_returns_false(self):
        with mock.patch.object(self.bc, "_orchestrator_enabled",
                               return_value=True), \
             mock.patch.object(self.bc, "_is_orchestration_request",
                               return_value=False):
            self.assertFalse(self.bc._maybe_orchestrate("hello there"))

    def test_maybe_orchestrate_handled_speaks_merged_brief(self):
        bc = self.bc
        spoke = []
        fake_orch = types.ModuleType("core.orchestrator")
        fake_orch.orchestrate = lambda *a, **k: "Here is your brief, sir."
        with mock.patch.object(bc, "_orchestrator_enabled",
                               return_value=True), \
             mock.patch.object(bc, "_is_orchestration_request",
                               return_value=True), \
             mock.patch.dict(sys.modules, {"core.orchestrator": fake_orch}), \
             mock.patch.object(bc, "set_state", lambda s: None), \
             mock.patch.object(bc, "_trim_conversation_history", lambda: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: spoke.append(a)):
            handled = bc._maybe_orchestrate("morning briefing")
        self.assertTrue(handled)
        self.assertEqual(spoke, [("Here is your brief, sir.",)])

    def test_maybe_orchestrate_strips_stray_action_token(self):
        bc = self.bc
        spoke = []
        fake_orch = types.ModuleType("core.orchestrator")
        fake_orch.orchestrate = (
            lambda *a, **k: "Your brief, sir. [ACTION: see_screen]")
        with mock.patch.object(bc, "_orchestrator_enabled",
                               return_value=True), \
             mock.patch.object(bc, "_is_orchestration_request",
                               return_value=True), \
             mock.patch.dict(sys.modules, {"core.orchestrator": fake_orch}), \
             mock.patch.object(bc, "set_state", lambda s: None), \
             mock.patch.object(bc, "_trim_conversation_history", lambda: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: spoke.append(a)):
            bc._maybe_orchestrate("morning briefing")
        self.assertEqual(len(spoke), 1)
        self.assertNotIn("ACTION", spoke[0][0])

    def test_maybe_orchestrate_empty_result_falls_through(self):
        bc = self.bc
        fake_orch = types.ModuleType("core.orchestrator")
        fake_orch.orchestrate = lambda *a, **k: "   "
        with mock.patch.object(bc, "_orchestrator_enabled",
                               return_value=True), \
             mock.patch.object(bc, "_is_orchestration_request",
                               return_value=True), \
             mock.patch.dict(sys.modules, {"core.orchestrator": fake_orch}), \
             mock.patch.object(bc, "set_state", lambda s: None), \
             mock.patch.object(bc, "_speak",
                               side_effect=AssertionError("must not speak")):
            self.assertFalse(bc._maybe_orchestrate("morning briefing"))

    def test_maybe_orchestrate_orchestrate_raises_falls_through(self):
        bc = self.bc
        fake_orch = types.ModuleType("core.orchestrator")

        def boom(*a, **k):
            raise RuntimeError("planner down")

        fake_orch.orchestrate = boom
        with mock.patch.object(bc, "_orchestrator_enabled",
                               return_value=True), \
             mock.patch.object(bc, "_is_orchestration_request",
                               return_value=True), \
             mock.patch.dict(sys.modules, {"core.orchestrator": fake_orch}), \
             mock.patch.object(bc, "set_state", lambda s: None), \
             mock.patch.object(bc, "_speak",
                               side_effect=AssertionError("must not speak")):
            self.assertFalse(bc._maybe_orchestrate("morning briefing"))


# ════════════════════════════════════════════════════════════════════════════
#  Misc boot helpers (guard clauses only)
# ════════════════════════════════════════════════════════════════════════════
class MiscBootHelperTests(SectionSixBase):
    def test_move_console_unknown_monitor_noop(self):
        # Unknown / empty monitor name returns immediately without ctypes.
        self.assertIsNone(self.bc._move_console_to_monitor("no_such_monitor"))
        self.assertIsNone(self.bc._move_console_to_monitor(""))

    def test_release_singleton_no_fd_noop(self):
        bc = self.bc
        orig = bc._SINGLETON_HELD_FD
        self.addCleanup(setattr, bc, "_SINGLETON_HELD_FD", orig)
        bc._SINGLETON_HELD_FD = None
        self.assertIsNone(bc._release_singleton())


# ════════════════════════════════════════════════════════════════════════════
#  Coverage-extension pass — error/edge branches under-exercised above.
#  Each class targets a specific uncovered span in the 10903-13152 band; all
#  I/O (LLM/anthropic/subprocess/threads/filesystem) is mocked or redirected.
# ════════════════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────────────────
#  _jarvis_pushback — close_window threshold WITHOUT an unsaved-work blurb
#  (line 11065: the else arm of the blurb conditional).
# ────────────────────────────────────────────────────────────────────────────
class JarvisPushbackCloseWindowTests(SectionSixBase):
    def test_close_window_over_threshold_no_blurb(self):
        bc = self.bc
        # N+1 windows whose titles carry NO unsaved hint -> blurb is None ->
        # the plain "...windows. Are you certain?" phrase fires (line 11065).
        n = bc.PUSHBACK_MAX_CLOSE_WINDOWS + 2
        wins = [_FakeWin(title=f"Plain Window {i}") for i in range(n)]
        with mock.patch.object(bc, "PUSHBACK_ENABLED", True), \
             mock.patch.object(bc, "_find_windows_by_title", lambda low: wins):
            out = bc._jarvis_pushback("close_window", "window")
        self.assertIsNotNone(out)
        phrase, reason = out
        self.assertIn(f"close {n} windows", phrase)
        self.assertNotIn("including", phrase)  # no blurb appended
        self.assertIn("Are you certain?", phrase)
        self.assertIn("close_window matched", reason)

    def test_close_window_with_blurb_takes_other_branch(self):
        bc = self.bc
        # A clearly-unsaved title yields a blurb -> the "...including <blurb>"
        # arm fires instead (keeps the two arms distinguished).
        n = bc.PUSHBACK_MAX_CLOSE_WINDOWS + 1
        wins = [_FakeWin(title="* draft.txt - Notepad")] + [
            _FakeWin(title=f"Plain {i}") for i in range(n)]
        with mock.patch.object(bc, "PUSHBACK_ENABLED", True), \
             mock.patch.object(bc, "_find_windows_by_title", lambda low: wins):
            out = bc._jarvis_pushback("close_window", "draft")
        self.assertIsNotNone(out)
        self.assertIn("including", out[0])

    def test_find_windows_raising_is_swallowed(self):
        bc = self.bc
        # _find_windows_by_title throwing -> matches=[] -> below threshold ->
        # returns None (exercises the except arm at 11056-11057).
        with mock.patch.object(bc, "PUSHBACK_ENABLED", True), \
             mock.patch.object(bc, "_find_windows_by_title",
                               side_effect=RuntimeError("win32 down")):
            self.assertIsNone(bc._jarvis_pushback("close_window", "anything"))


# ────────────────────────────────────────────────────────────────────────────
#  _mission_narration_cue — bad-template branch (11222-11223): a template
#  whose .format(arg=...) raises falls back to the raw template body.
# ────────────────────────────────────────────────────────────────────────────
class MissionNarrationCueErrorTests(SectionSixBase):
    def test_bad_template_falls_back_to_raw(self):
        bc = self.bc
        # A template referencing an unknown field raises KeyError in .format(),
        # so the except restores tpl.rstrip() (line 11223).
        bad = dict(bc._MISSION_NARRATION_CUES)
        bad["weird_action"] = "Doing {nonexistent_field}"
        with mock.patch.object(bc, "_MISSION_NARRATION_CUES", bad):
            out = bc._mission_narration_cue("weird_action", "x", 1, 2)
        self.assertEqual(out, "Doing {nonexistent_field}…")


# ────────────────────────────────────────────────────────────────────────────
#  parse_and_run_actions — TTS-failure + autocorrect + UIFailsafe + mid-task
#  branches the happy-path tests don't hit.
# ────────────────────────────────────────────────────────────────────────────
class ParseAndRunActionsBranchTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        self._p(bc, "_cmd_autocorrect", None)
        self._p(bc, "PC_CONTROL_ENABLED", True)

    def _with_action(self, name, fn, informative=False):
        bc = self.bc
        acts = dict(bc.ACTIONS)
        acts[name] = fn
        self._p(bc, "ACTIONS", acts)
        if informative:
            self._p(bc, "INFORMATIVE_ACTIONS",
                    set(bc.INFORMATIVE_ACTIONS) | {name})

    def test_narration_intro_tts_failure_is_caught(self):
        # _speak raising on the intro must not abort the dispatch (11375-11376).
        bc = self.bc
        self._with_action("act_one", lambda a: "1")

        def speak(*a, **k):
            raise RuntimeError("intro tts dead")

        with mock.patch.object(bc, "MISSION_NARRATION_ENABLED", True), \
             mock.patch.object(bc, "MISSION_NARRATION_THRESHOLD", 3), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", speak):
            cleaned, results = bc.parse_and_run_actions(
                "[ACTION: act_one, a] [ACTION: act_one, b] [ACTION: act_one, c]")
        # All three still ran despite the intro TTS blowing up.
        self.assertEqual(len(results), 3)

    def test_narration_cue_tts_failure_is_caught(self):
        # _speak raises ONLY on the per-step cue (not the intro), exercising
        # 11473-11474 while still letting the action run.
        bc = self.bc
        ran = []
        self._with_action("act_one", lambda a: ran.append(a) or "1")
        calls = {"n": 0}

        def speak(*a, **k):
            calls["n"] += 1
            if calls["n"] > 1:   # let the intro through, fail on cues
                raise RuntimeError("cue tts dead")

        with mock.patch.object(bc, "MISSION_NARRATION_ENABLED", True), \
             mock.patch.object(bc, "MISSION_NARRATION_THRESHOLD", 3), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", speak):
            cleaned, results = bc.parse_and_run_actions(
                "[ACTION: act_one, a] [ACTION: act_one, b] [ACTION: act_one, c]")
        self.assertEqual(ran, ["a", "b", "c"])

    def test_autocorrect_scoring_exception_treated_as_no_match(self):
        # autocorrect_command_choice raising -> choice defaults to status=none
        # -> unknown-action path (11400-11402 + 11442-11444).
        bc = self.bc
        fake_ac = types.SimpleNamespace(
            autocorrect_command_choice=mock.Mock(
                side_effect=RuntimeError("scorer exploded")))
        with mock.patch.object(bc, "_cmd_autocorrect", fake_ac):
            cleaned, results = bc.parse_and_run_actions("[ACTION: zzzbad, x]")
        self.assertEqual(results[0][0], "zzzbad")
        self.assertIn("unknown action", results[0][1])

    def test_autocorrect_silent_confirm_tts_failure_is_caught(self):
        # silent reroute whose confirmation _speak raises (11411-11412): the
        # action still runs on the corrected name.
        bc = self.bc
        ran = []
        self._with_action("real_target", lambda a: ran.append(a) or "ok")
        fake_ac = types.SimpleNamespace(
            autocorrect_command_choice=lambda name, keys, **kw: {
                "status": "silent",
                "primary": ("real_target", 0.95),
                "secondary": None,
            })

        def speak(*a, **k):
            raise RuntimeError("confirm tts dead")

        with mock.patch.object(bc, "_cmd_autocorrect", fake_ac), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", speak):
            cleaned, results = bc.parse_and_run_actions("[ACTION: realtarget, hi]")
        self.assertEqual(ran, ["hi"])
        self.assertEqual(results[0][0], "real_target")

    def test_autocorrect_disambig_tts_failure_is_caught(self):
        # ambiguous branch whose 'did you mean' _speak raises (11430-11431):
        # the choice is still queued and an AMBIGUOUS result returned.
        bc = self.bc
        self._with_action("opt_a", lambda a: "A")
        self._with_action("opt_b", lambda a: "B")
        fake_ac = types.SimpleNamespace(
            autocorrect_command_choice=lambda name, keys, **kw: {
                "status": "ambiguous",
                "primary": ("opt_a", 0.80),
                "secondary": ("opt_b", 0.78),
            })

        def speak(*a, **k):
            raise RuntimeError("disambig tts dead")

        with mock.patch.object(bc, "_cmd_autocorrect", fake_ac), \
             mock.patch.object(bc, "_speak", speak):
            bc._pending_autocorrect_choice.clear()
            cleaned, results = bc.parse_and_run_actions("[ACTION: opt, x]")
        self.assertIn("AMBIGUOUS", results[0][1])
        self.assertEqual(len(bc._pending_autocorrect_choice), 1)

    def test_autocorrect_no_candidate_falls_to_unknown(self):
        # status='none' WITH a sub-threshold primary -> the else 'no match'
        # branch prints best-conf (11439-11440) then unknown-action path.
        bc = self.bc
        fake_ac = types.SimpleNamespace(
            autocorrect_command_choice=lambda name, keys, **kw: {
                "status": "none",
                "primary": ("close_thing", 0.40),
                "secondary": None,
            })
        with mock.patch.object(bc, "_cmd_autocorrect", fake_ac):
            cleaned, results = bc.parse_and_run_actions("[ACTION: clsthing, x]")
        self.assertIn("unknown action", results[0][1])

    def test_autocorrect_silent_primary_not_in_actions_falls_through(self):
        # silent status but primary[0] not in ACTIONS -> skips the reroute,
        # lands on unknown-action (covers the guard on 11406).
        bc = self.bc
        fake_ac = types.SimpleNamespace(
            autocorrect_command_choice=lambda name, keys, **kw: {
                "status": "silent",
                "primary": ("ghost_never_registered", 0.99),
                "secondary": None,
            })
        with mock.patch.object(bc, "_cmd_autocorrect", fake_ac):
            cleaned, results = bc.parse_and_run_actions("[ACTION: ghosty, x]")
        self.assertIn("unknown action", results[0][1])

    def test_uifailsafe_error_yields_clean_message(self):
        # fn raising UIFailsafeError -> the dedicated except arm (11520-11521)
        # surfaces the message verbatim, NOT routed through _jfl.
        bc = self.bc

        def trip(_a):
            raise bc.UIFailsafeError("fail-safe tripped, sir")

        self._with_action("clicker", trip)
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: None):
            cleaned, results = bc.parse_and_run_actions("[ACTION: clicker, here]")
        name, msg, informative = results[0]
        self.assertEqual(name, "clicker")
        self.assertEqual(msg, "fail-safe tripped, sir")
        self.assertFalse(informative)

    def test_action_exception_with_traceback_capture_failing(self):
        # The inner traceback.format_exc() raising (11538-11539) must not stop
        # the JARVIS-voice failure result from being recorded.
        bc = self.bc

        def boom(_a):
            raise ValueError("primary boom")

        self._with_action("boomer", boom)
        with mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: None), \
             mock.patch.object(bc.traceback, "format_exc",
                               side_effect=RuntimeError("tb unavailable")):
            cleaned, results = bc.parse_and_run_actions("[ACTION: boomer, x]")
        self.assertEqual(results[0][0], "boomer")
        self.assertIn("failed", results[0][1].lower())

    def test_mid_task_timer_started_and_cancelled(self):
        # Long-running action with MID_TASK_STATUS_ENABLED -> the threading.Timer
        # block (11488-11498) runs and the finally cancels it (11547-11553).
        # threading.Timer is faked so NO real thread/timer is created.
        bc = self.bc
        events = []

        class FakeTimer:
            def __init__(self, delay, fn, args=()):
                self.delay, self.fn, self.args = delay, fn, args
                self.daemon = False

            def start(self):
                events.append("start")

            def cancel(self):
                events.append("cancel")

            def join(self, timeout=None):
                events.append("join")

        # Pick a real long-running action name and register a fast handler.
        long_name = sorted(bc.LONG_RUNNING_ACTIONS)[0]
        self._with_action(long_name, lambda a: "done")
        with mock.patch.object(bc, "MID_TASK_STATUS_ENABLED", True), \
             mock.patch.object(bc.threading, "Timer", FakeTimer), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: None):
            cleaned, results = bc.parse_and_run_actions(
                f"[ACTION: {long_name}, thing]")
        self.assertEqual(results[0][0], long_name)
        # Timer was started then cancelled in the finally.
        self.assertIn("start", events)
        self.assertIn("cancel", events)

    def test_mid_task_timer_start_failure_is_caught(self):
        # threading.Timer construction raising (11496-11498) leaves _mid_task_timer
        # None and the action still runs.
        bc = self.bc

        def bad_timer(*a, **k):
            raise RuntimeError("no threads left")

        long_name = sorted(bc.LONG_RUNNING_ACTIONS)[0]
        self._with_action(long_name, lambda a: "done")
        with mock.patch.object(bc, "MID_TASK_STATUS_ENABLED", True), \
             mock.patch.object(bc.threading, "Timer", bad_timer), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: None):
            cleaned, results = bc.parse_and_run_actions(
                f"[ACTION: {long_name}, thing]")
        self.assertEqual(results[0][1], "done")

    def test_mid_task_timer_already_fired_is_joined(self):
        # When the timer "fired" before fn returned, the finally join()s it
        # (11550-11551). FakeTimer.start() flips the shared fired-flag (the
        # third positional arg) to simulate the bridge having fired — no real
        # thread runs.
        bc = self.bc
        events = []

        class FakeTimer:
            def __init__(self, delay, fn, args=()):
                self.args = args
                self.daemon = False

            def start(self):
                events.append("start")
                # args == (name, arg, _mid_task_fired) — mark it fired.
                self.args[2][0] = True

            def cancel(self):
                events.append("cancel")

            def join(self, timeout=None):
                events.append("join")

        long_name = sorted(bc.LONG_RUNNING_ACTIONS)[0]
        self._with_action(long_name, lambda a: "done")
        with mock.patch.object(bc, "MID_TASK_STATUS_ENABLED", True), \
             mock.patch.object(bc.threading, "Timer", FakeTimer), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: None):
            cleaned, results = bc.parse_and_run_actions(
                f"[ACTION: {long_name}, thing]")
        self.assertEqual(results[0][1], "done")
        # cancel() AND join() both ran because the fired-flag was set.
        self.assertIn("cancel", events)
        self.assertIn("join", events)

    def test_mid_task_timer_cancel_exception_is_swallowed(self):
        # 11552-11553: the finally's _mid_task_timer.cancel() raises -> the
        # except swallows it so a flaky timer can't break action dispatch.
        bc = self.bc
        events = []

        class FakeTimer:
            def __init__(self, delay, fn, args=()):
                self.args = args
                self.daemon = False

            def start(self):
                events.append("start")

            def cancel(self):
                events.append("cancel")
                raise RuntimeError("cancel exploded")

            def join(self, timeout=None):
                events.append("join")

        long_name = sorted(bc.LONG_RUNNING_ACTIONS)[0]
        self._with_action(long_name, lambda a: "done")
        with mock.patch.object(bc, "MID_TASK_STATUS_ENABLED", True), \
             mock.patch.object(bc.threading, "Timer", FakeTimer), \
             mock.patch.object(bc, "_needs_confirmation", lambda n, a: False), \
             mock.patch.object(bc, "_jarvis_pushback", lambda n, a: None), \
             mock.patch.object(bc, "_speak", lambda *a, **k: None):
            cleaned, results = bc.parse_and_run_actions(
                f"[ACTION: {long_name}, thing]")
        # Action result still returned despite the cancel() blowing up.
        self.assertEqual(results[0][1], "done")
        self.assertIn("cancel", events)


# ────────────────────────────────────────────────────────────────────────────
#  get_followup_response — ollama backend + mode-router addendum import path.
# ────────────────────────────────────────────────────────────────────────────
class GetFollowupResponseExtraTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_last_voice_route", [{"addendum": ""}])
        self._p(bc, "_last_user_tone", [None])

    def test_ollama_backend_path(self):
        # AI_BACKEND == 'ollama' -> the bounded-ollama branch (P1-2). The call
        # now flows through _ollama_chat_bounded so a wedged runner can't hang
        # the follow-up turn forever.
        bc = self.bc
        captured = {}

        def _bounded(model, messages):
            captured["model"] = model
            captured["messages"] = messages
            return {"message": {"content": "ollama follow-up"}}

        with mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "_ollama_chat_bounded", side_effect=_bounded):
            out = bc.get_followup_response([("get_time", "noon")])
        self.assertEqual(out, "ollama follow-up")
        # The system prompt is the first message and the action summary the last.
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertIn("get_time", captured["messages"][-1]["content"])

    def test_mode_router_addendum_failure_is_swallowed(self):
        # The mode_router import / call raising (11675-11676) must not break
        # the follow-up; the claude path still returns.
        bc = self.bc

        class FakeClient:
            def complete(self, **kwargs):
                return "ok, sir."

        import core.mode_router as _mr
        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "_llm_client", FakeClient()), \
             mock.patch.object(_mr, "system_prompt_addendum",
                               side_effect=RuntimeError("mode router down")):
            out = bc.get_followup_response([("x", "y")])
        self.assertEqual(out, "ok, sir.")

    def test_claude_without_llm_client_uses_anthropic_sdk(self):
        # _llm_client is None on the claude path -> the raw anthropic SDK
        # branch (11694-11698) is taken.
        bc = self.bc
        import anthropic
        fake_msg = types.SimpleNamespace(
            content=[types.SimpleNamespace(text="sdk follow-up")])

        class FakeAnthropic:
            def __init__(self, *a, **k):
                self.messages = self

            def create(self, **k):
                return fake_msg

        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.object(bc, "_llm_client", None), \
             mock.patch.object(anthropic, "Anthropic", FakeAnthropic):
            out = bc.get_followup_response([("get_time", "3")])
        self.assertEqual(out, "sdk follow-up")

    def test_unknown_backend_returns_empty(self):
        # AI_BACKEND that is neither 'claude' nor 'ollama' falls past both
        # branches to the trailing `return ""` (11720).
        bc = self.bc
        with mock.patch.object(bc, "AI_BACKEND", "some_other_backend"):
            out = bc.get_followup_response([("x", "y")])
        self.assertEqual(out, "")


# ────────────────────────────────────────────────────────────────────────────
#  handle_autocorrect_disambig_response — the TTS-failure + vanished-action +
#  successful-run branches not covered by the primary suite.
# ────────────────────────────────────────────────────────────────────────────
class HandleAutocorrectDisambigBranchTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self.ran = []
        acts = dict(bc.ACTIONS)
        acts["alpha"] = lambda a: self.ran.append(("alpha", a)) or "A"
        acts["beta"] = lambda a: self.ran.append(("beta", a)) or "B"
        self._p(bc, "ACTIONS", acts)

    def _queue(self, primary=("alpha", "q"), secondary=("beta", "q")):
        self.bc._pending_autocorrect_choice.clear()
        self.bc._pending_autocorrect_choice.append(
            {"primary": primary, "secondary": secondary, "original": "alfa"})

    def test_cancel_tts_failure_still_consumes(self):
        # 'no' -> cancel; the "Cancelled, sir." _speak raising (11792-11793)
        # must still consume the turn.
        self._queue()
        with mock.patch.object(self.bc, "_speak",
                               side_effect=RuntimeError("tts dead")):
            self.assertTrue(
                self.bc.handle_autocorrect_disambig_response("no"))
        self.assertEqual(self.ran, [])

    def test_vanished_action_tts_failure_still_consumes(self):
        # picked action missing AND the "can't run" _speak raises (11803-11804).
        self._queue(primary=("ghost_action", "q"))
        with mock.patch.object(self.bc, "_speak",
                               side_effect=RuntimeError("tts dead")):
            self.assertTrue(
                self.bc.handle_autocorrect_disambig_response("first"))
        self.assertEqual(self.ran, [])

    def test_destructive_refusal_tts_failure_still_consumes(self):
        # destructive pick refusal whose _speak raises (11819-11820).
        bc = self.bc
        dr = sorted(bc._DESTRUCTIVE_REPLAY_ACTIONS)[0]
        ran = []
        acts = dict(bc.ACTIONS)
        acts[dr] = lambda a: ran.append(a) or "ran"
        with mock.patch.object(bc, "ACTIONS", acts), \
             mock.patch.object(bc, "_speak",
                               side_effect=RuntimeError("tts dead")):
            bc._pending_autocorrect_choice.clear()
            bc._pending_autocorrect_choice.append(
                {"primary": (dr, "q"), "secondary": ("beta", "q"),
                 "original": "x"})
            self.assertTrue(
                bc.handle_autocorrect_disambig_response("yes"))
        self.assertEqual(ran, [])

    def test_successful_run_confirm_tts_failure_then_action_raises(self):
        # The happy 'run' path with BOTH the confirm _speak failing (11825-11826)
        # AND the action itself raising (11833-11834) — both swallowed, consumed.
        bc = self.bc

        def boom(_a):
            raise RuntimeError("action exploded")

        acts = dict(bc.ACTIONS)
        acts["gamma"] = boom
        with mock.patch.object(bc, "ACTIONS", acts), \
             mock.patch.object(bc, "_speak",
                               side_effect=RuntimeError("confirm tts dead")), \
             mock.patch.object(bc, "record_session_action", lambda *a, **k: None), \
             mock.patch.object(bc, "record_action_history", lambda *a, **k: None):
            bc._pending_autocorrect_choice.clear()
            bc._pending_autocorrect_choice.append(
                {"primary": ("gamma", "q"), "secondary": ("beta", "q"),
                 "original": "gama"})
            self.assertTrue(
                bc.handle_autocorrect_disambig_response("first"))

    def test_successful_run_records_history(self):
        # Plain successful pick -> fn runs, history recorded (11827-11832).
        self._queue()
        recorded = []
        with mock.patch.object(self.bc, "_speak", lambda *a, **k: None), \
             mock.patch.object(self.bc, "record_action_history",
                               lambda *a, **k: recorded.append(a)):
            self.assertTrue(
                self.bc.handle_autocorrect_disambig_response("alpha"))
        self.assertEqual(self.ran, [("alpha", "q")])
        self.assertEqual(len(recorded), 1)


# ────────────────────────────────────────────────────────────────────────────
#  handle_confirmation_response — the fn-missing 'continue' skip (11855-11856).
# ────────────────────────────────────────────────────────────────────────────
class HandleConfirmationResponseBranchTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_speak", lambda *a, **k: None)

    def test_missing_action_skipped_during_confirm(self):
        bc = self.bc
        ran = []
        acts = dict(bc.ACTIONS)
        acts["realone"] = lambda a: ran.append(a) or "ok"
        # 'ghost' is queued but absent from ACTIONS -> the `if not fn: continue`
        # branch (11855-11856) skips it; 'realone' still runs.
        with mock.patch.object(bc, "ACTIONS", acts):
            bc._pending_confirmation.clear()
            bc._pending_confirmation.append(("ghost", "x"))
            bc._pending_confirmation.append(("realone", "y"))
            consumed = bc.handle_confirmation_response("yes")
        self.assertTrue(consumed)
        self.assertEqual(ran, ["y"])
        self.assertEqual(list(bc._pending_confirmation), [])


# ────────────────────────────────────────────────────────────────────────────
#  _apply_quip_layer — the no-jarvis_quip_layer-attr early return (11886-11887)
#  and the all-underscore-results -> primary stays None path (11890-11894).
# ────────────────────────────────────────────────────────────────────────────
class ApplyQuipLayerBranchTests(SectionSixBase):
    def test_layer_without_quip_attr_returns_input(self):
        bc = self.bc
        # A layer object lacking jarvis_quip_layer -> early return (11886-11887).
        layer = types.SimpleNamespace(something_else=lambda: None)
        with mock.patch.object(bc, "_tts_layer", layer):
            out = bc._apply_quip_layer("Done.", [("get_time", "x", True)])
        self.assertEqual(out, "Done.")

    def test_all_synthetic_results_leave_primary_none(self):
        bc = self.bc
        seen = {}

        def quip(text, primary):
            seen["primary"] = primary
            return text + "!"

        layer = types.SimpleNamespace(jarvis_quip_layer=quip)
        # Every result is _-prefixed/unknown -> the loop never sets primary,
        # so it stays None (covers the continue arms at 11891 + 11894).
        with mock.patch.object(bc, "_tts_layer", layer):
            out = bc._apply_quip_layer(
                "Sure.",
                [("_unverified_claim", "x", True),
                 ("unknown action: zz", "y", False)])
        self.assertEqual(out, "Sure.!")
        self.assertIsNone(seen["primary"])

    def test_empty_action_name_skipped(self):
        bc = self.bc
        seen = {}

        def quip(text, primary):
            seen["primary"] = primary
            return text

        layer = types.SimpleNamespace(jarvis_quip_layer=quip)
        # Iteration is REVERSED, so the falsy name must be LAST to make the
        # `if not n: continue` arm (11890-11891) fire before the real action.
        with mock.patch.object(bc, "_tts_layer", layer):
            bc._apply_quip_layer(
                "Hi.", [("get_time", "y", True), ("", "x", True)])
        self.assertEqual(seen["primary"], "get_time")


# ────────────────────────────────────────────────────────────────────────────
#  _speak — under-exercised guards: boot throttle, HUD-write failure, the
#  [wry] then late [mood] re-parse + wry-parse exception, staging-record
#  failure, volume astype failure, and the inner set_state failure.
# ────────────────────────────────────────────────────────────────────────────
class SpeakBranchTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_write_hud_state", lambda **k: None)

    def test_boot_window_throttle_sleeps(self):
        # Within 30s of boot AND <1s since last speech -> time.sleep is called
        # (12007-12009). time.sleep is mocked so the test stays instant.
        bc = self.bc
        import numpy as np
        slept = []
        with mock.patch.object(bc, "_session_start_time", bc.time.time()), \
             mock.patch.object(bc, "last_speech_time", bc.time.time()), \
             mock.patch.object(bc.time, "sleep", lambda s: slept.append(s)), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise",
                               lambda t: (np.zeros(4, dtype=np.float32), 16000)), \
             mock.patch.object(bc, "play_with_lipsync", lambda a, sr: None), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("Quick line, sir.")
        self.assertEqual(len(slept), 1)
        self.assertGreater(slept[0], 0.0)

    def test_hud_write_failure_is_swallowed(self):
        # _write_hud_state raising (12029-12030) must not abort the mute path.
        bc = self.bc
        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_write_hud_state",
                               side_effect=RuntimeError("hud down")), \
             mock.patch.object(bc, "_tts_muted", [True]), \
             mock.patch.object(bc, "synthesise",
                               side_effect=AssertionError("must not synth")):
            self.assertIsNone(bc._speak("muted line"))

    def test_wry_first_then_late_mood_tag_reparsed(self):
        # "[wry][mood:xxx]" — wry is parsed first, then the mood re-parse arm
        # (12049-12052) picks up the trailing mood tag. chosen_mood ends up set,
        # both tags stripped before synthesis.
        bc = self.bc
        import numpy as np
        synth_args = []
        captured = {}

        def synth(t):
            synth_args.append(t)
            captured["mood"] = bc._last_mood[0]
            return (np.zeros(4, dtype=np.float32), 16000)

        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise", synth), \
             mock.patch.object(bc, "play_with_lipsync", lambda a, sr: None), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("[wry][mood:dry_amused] Oh, splendid, sir.")
        self.assertEqual(synth_args, ["Oh, splendid, sir."])
        self.assertEqual(captured["mood"], "dry_amused")

    def test_wry_parse_exception_defaults_false(self):
        # _tts_layer.parse_wry_tag raising (12053-12054) -> wry_flag=False and
        # _speak proceeds normally.
        bc = self.bc
        import numpy as np
        synth_args = []
        bad_layer = types.SimpleNamespace(
            parse_wry_tag=mock.Mock(side_effect=RuntimeError("wry parser down")))
        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_tts_layer", bad_layer), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(
                 bc, "synthesise",
                 lambda t: (synth_args.append(t)
                            or np.zeros(4, dtype=np.float32), 16000)), \
             mock.patch.object(bc, "play_with_lipsync", lambda a, sr: None), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("Plain line, sir.")
        self.assertEqual(synth_args, ["Plain line, sir."])

    def test_staging_record_failure_is_swallowed(self):
        # staging record_reply raising (12089-12090) must not propagate; the
        # staging path still returns None without opening audio.
        bc = self.bc
        fake_stg = types.ModuleType("staging_instance")
        fake_stg.record_reply = mock.Mock(side_effect=RuntimeError("disk full"))
        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: True), \
             mock.patch.dict(sys.modules, {"staging_instance": fake_stg}), \
             mock.patch.object(bc, "synthesise",
                               side_effect=AssertionError("must not synth")):
            self.assertIsNone(bc._speak("Staged line, sir."))
        fake_stg.record_reply.assert_called_once()

    def test_volume_scale_astype_failure_is_swallowed(self):
        # A non-ndarray audio buffer makes the .astype attenuation raise
        # (12110-12112); the except passes and playback still receives the
        # original buffer.
        bc = self.bc
        played = []
        # synthesise returns a plain list (no .astype) so the scale block throws.
        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise",
                               lambda t: ([0.0, 0.0, 0.0], 16000)), \
             mock.patch.object(bc, "play_with_lipsync",
                               lambda a, sr: played.append(a)), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("Quietly now, sir.", volume_scale=0.25)
        self.assertEqual(played, [[0.0, 0.0, 0.0]])

    def test_recovery_setstate_idle_also_failing(self):
        # synthesise raises AND the recovery set_state('idle') ALSO raises
        # (12123-12126) — both swallowed, _tts_playback_active forced False.
        bc = self.bc
        calls = []

        def flaky_set_state(s):
            calls.append(s)
            if s == "idle":
                raise RuntimeError("state machine wedged")

        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise",
                               side_effect=RuntimeError("synth boom")), \
             mock.patch.object(bc, "set_state", flaky_set_state):
            bc._speak("Anything at all, sir.")
        # 'speaking' was set, the idle recovery was attempted, and the guard
        # was still forced back to False in the finally.
        self.assertIn("speaking", calls)
        self.assertFalse(bc._tts_playback_active[0])

    def test_no_tts_layer_defaults_wry_false(self):
        # _tts_layer is None -> the else arm (12056) sets wry_flag=False and
        # _speak still synthesises normally.
        bc = self.bc
        import numpy as np
        synth_args = []
        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_tts_layer", None), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(
                 bc, "synthesise",
                 lambda t: (synth_args.append(t)
                            or np.zeros(4, dtype=np.float32), 16000)), \
             mock.patch.object(bc, "play_with_lipsync", lambda a, sr: None), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("No layer here, sir.")
        self.assertEqual(synth_args, ["No layer here, sir."])

    def test_mood_kwarg_overrides_mood_tag(self):
        # mood= kwarg wins over a [mood:xxx] tag (12022 precedence).
        bc = self.bc
        import numpy as np
        captured = {}

        def synth(t):
            captured["mood"] = bc._last_mood[0]
            return (np.zeros(4, dtype=np.float32), 16000)

        with mock.patch.object(bc, "_session_start_time", 0.0), \
             mock.patch.object(bc, "_tts_muted", [False]), \
             mock.patch.object(bc, "_is_staging", lambda: False), \
             mock.patch.object(bc, "synthesise", synth), \
             mock.patch.object(bc, "play_with_lipsync", lambda a, sr: None), \
             mock.patch.object(bc, "set_state", lambda s: None):
            bc._speak("[mood:calm_efficient] Noted, sir.", mood="urgent_clipped")
        self.assertEqual(captured["mood"], "urgent_clipped")


# ────────────────────────────────────────────────────────────────────────────
#  Power-plan helpers — the failure-print arms not hit by the happy paths.
# ────────────────────────────────────────────────────────────────────────────
class PowerPlanBranchTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, self.bc, "_prior_power_plan_guid",
                        self.bc._prior_power_plan_guid)

    def test_activate_no_active_plan_skips(self):
        # _get_active_power_plan_guid returns None -> "could not read" skip
        # (12535-12537); _set_power_plan never called.
        bc = self.bc
        bc._prior_power_plan_guid = None
        with mock.patch.object(bc, "_get_active_power_plan_guid",
                               return_value=None), \
             mock.patch.object(bc, "_set_power_plan",
                               side_effect=AssertionError("must not switch")):
            bc._activate_high_performance_plan()
        self.assertIsNone(bc._prior_power_plan_guid)

    def test_activate_set_plan_failure_does_not_remember(self):
        # switch attempted but _set_power_plan returns False -> the failure
        # print arm (12546-12547); prior guid stays unset.
        bc = self.bc
        bc._prior_power_plan_guid = None
        prior = "12121212-3434-5656-7878-909090909090"
        with mock.patch.object(bc, "_get_active_power_plan_guid",
                               return_value=prior), \
             mock.patch.object(bc, "_set_power_plan", return_value=False):
            bc._activate_high_performance_plan()
        self.assertIsNone(bc._prior_power_plan_guid)

    def test_activate_unexpected_exception_is_caught(self):
        # _get_active_power_plan_guid raising -> outer except (12548-12549).
        bc = self.bc
        with mock.patch.object(bc, "_get_active_power_plan_guid",
                               side_effect=RuntimeError("powercfg exploded")):
            bc._activate_high_performance_plan()  # no exception == pass

    def test_restore_skips_when_prior_is_high_perf(self):
        # prior == High Performance GUID -> nothing-to-restore early out
        # (12557-12558); _set_power_plan never called.
        bc = self.bc
        bc._prior_power_plan_guid = bc._HIGH_PERF_GUID
        with mock.patch.object(bc, "_set_power_plan",
                               side_effect=AssertionError("must not run")):
            bc._restore_prior_power_plan()

    def test_restore_set_plan_failure_print_arm(self):
        # _set_power_plan returns False on restore -> failure print (12561-12562).
        bc = self.bc
        prior = "abababab-cdcd-efef-0101-202020202020"
        bc._prior_power_plan_guid = prior
        with mock.patch.object(bc, "_set_power_plan", return_value=False) as sp:
            bc._restore_prior_power_plan()
        sp.assert_called_once_with(prior)

    def test_restore_unexpected_exception_is_caught(self):
        # _set_power_plan raising on restore -> outer except (12563-12564).
        bc = self.bc
        bc._prior_power_plan_guid = "cccccccc-dddd-eeee-ffff-000011112222"
        with mock.patch.object(bc, "_set_power_plan",
                               side_effect=RuntimeError("boom")):
            bc._restore_prior_power_plan()  # no exception == pass


# ────────────────────────────────────────────────────────────────────────────
#  _find_cublas_dll — site.getsitepackages / getusersitepackages raising, and
#  a usersitepackages-only hit (the second site source).
# ────────────────────────────────────────────────────────────────────────────
class FindCublasDllBranchTests(SectionSixBase):
    def test_site_lookups_raise_then_fall_through(self):
        # Both site.* calls raise (12597-12598 + 12603-12604); with nothing on
        # PATH / CUDA_PATH / Program Files the walk returns None cleanly.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = mock.Mock(
                side_effect=RuntimeError("no sitepackages"))
            fake_site.getusersitepackages = mock.Mock(
                side_effect=RuntimeError("no usersite"))
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": "", "CUDA_PATH": "",
                                  "ProgramFiles": d, "ProgramFiles(x86)": ""},
                                 clear=False):
                self.assertIsNone(bc._find_cublas_dll())

    def test_found_in_user_site_packages(self):
        # getsitepackages empty but getusersitepackages holds the DLL
        # (covers the usersite append at 12601-12602 + the hit at 12607-12608).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            bindir = os.path.join(d, "nvidia", "cublas", "bin")
            os.makedirs(bindir)
            dll = os.path.join(bindir, bc._CUBLAS_DLL_NAME)
            open(dll, "w").close()
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: []
            fake_site.getusersitepackages = lambda: d
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": "", "CUDA_PATH": ""}, clear=False):
                self.assertEqual(bc._find_cublas_dll(), dll)

    def test_empty_path_entries_skipped(self):
        # A PATH with empty segments exercises the `if not p: continue` arm
        # while still finding the DLL in a populated segment.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            dll = os.path.join(d, bc._CUBLAS_DLL_NAME)
            open(dll, "w").close()
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: []
            fake_site.getusersitepackages = lambda: ""
            path_val = os.pathsep + d + os.pathsep   # leading + trailing empties
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": path_val, "CUDA_PATH": ""},
                                 clear=False):
                self.assertEqual(bc._find_cublas_dll(), dll)

    def test_site_dir_join_failure_hits_outer_except(self):
        # getsitepackages returns a non-string entry, so os.path.join inside the
        # loop raises -> the OUTER except (12609-12610) swallows it and the walk
        # continues to PATH where the DLL is found.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            dll = os.path.join(d, bc._CUBLAS_DLL_NAME)
            open(dll, "w").close()
            fake_site = types.ModuleType("site")
            fake_site.getsitepackages = lambda: [None]   # join(None,...) -> TypeError
            fake_site.getusersitepackages = lambda: ""
            with mock.patch.dict(sys.modules, {"site": fake_site}), \
                 mock.patch.dict(os.environ,
                                 {"PATH": d, "CUDA_PATH": ""}, clear=False):
                self.assertEqual(bc._find_cublas_dll(), dll)


# ────────────────────────────────────────────────────────────────────────────
#  _preflight_cublas_check — the "already queued" (n==0) and append-failure
#  branches of the CUDA-task queue (12734-12739).
# ────────────────────────────────────────────────────────────────────────────
class PreflightCublasQueueBranchTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, self.bc, "_force_whisper_cpu_int8",
                        self.bc._force_whisper_cpu_int8)

    def test_task_already_queued_duplicate_skipped(self):
        # _append_tasks returns 0 -> "already queued" branch (12734-12735).
        bc = self.bc
        bc._force_whisper_cpu_int8 = False
        fake_ou = types.ModuleType("overnight_upgrade")
        fake_ou._append_tasks = lambda tasks: 0
        with mock.patch.object(bc, "_find_cublas_dll", return_value=None), \
             mock.patch.object(bc, "_ctranslate2_sees_cuda", return_value=True), \
             mock.patch.dict(sys.modules, {"overnight_upgrade": fake_ou}):
            self.assertFalse(bc._preflight_cublas_check())
        self.assertTrue(bc._force_whisper_cpu_int8)

    def test_append_tasks_failure_is_caught(self):
        # _append_tasks raising -> inner except (12736-12737); still forces CPU.
        bc = self.bc
        bc._force_whisper_cpu_int8 = False
        fake_ou = types.ModuleType("overnight_upgrade")
        fake_ou._append_tasks = mock.Mock(
            side_effect=RuntimeError("todo file locked"))
        with mock.patch.object(bc, "_find_cublas_dll", return_value=None), \
             mock.patch.object(bc, "_ctranslate2_sees_cuda", return_value=True), \
             mock.patch.dict(sys.modules, {"overnight_upgrade": fake_ou}):
            self.assertFalse(bc._preflight_cublas_check())
        self.assertTrue(bc._force_whisper_cpu_int8)

    def test_overnight_upgrade_not_importable_is_caught(self):
        # overnight_upgrade import itself failing -> outer except (12738-12739).
        # Force the lazy `import overnight_upgrade` statement to raise.
        bc = self.bc
        bc._force_whisper_cpu_int8 = False
        real_import = __import__

        def boom_import(name, *a, **k):
            if name == "overnight_upgrade":
                raise ImportError("overnight_upgrade broken")
            return real_import(name, *a, **k)

        with mock.patch.object(bc, "_find_cublas_dll", return_value=None), \
             mock.patch.object(bc, "_ctranslate2_sees_cuda", return_value=True), \
             mock.patch("builtins.__import__", boom_import):
            self.assertFalse(bc._preflight_cublas_check())
        self.assertTrue(bc._force_whisper_cpu_int8)


# ────────────────────────────────────────────────────────────────────────────
#  _drain_injected_command — claim-failure, read-failure, non-list payload,
#  and the requeue-write-failure branches.
# ────────────────────────────────────────────────────────────────────────────
class DrainInjectedCommandBranchTests(SectionSixBase):
    def _redirect(self, path):
        self._p(self.bc, "INJECTED_COMMANDS_PATH", path)

    def test_claim_rename_filenotfound_returns_none(self):
        # os.replace raising FileNotFoundError (file vanished between the
        # exists() check and the rename) -> the dedicated arm (12906-12907).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["cmd"], f)
            with mock.patch.object(bc.os, "replace",
                                   side_effect=FileNotFoundError()):
                self.assertIsNone(bc._drain_injected_command())

    def test_claim_rename_failure_returns_none(self):
        # os.replace raising a non-FileNotFound error -> claim-failed branch
        # (12908-12910).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["cmd"], f)
            with mock.patch.object(bc.os, "replace",
                                   side_effect=PermissionError("locked")):
                self.assertIsNone(bc._drain_injected_command())

    def test_read_failure_discards_snapshot(self):
        # open() on the consumed snapshot raising -> read-failed branch
        # (12914-12918); the .consuming file is removed.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["cmd"], f)
            real_open = open

            def flaky_open(p, *a, **k):
                if str(p).endswith(".consuming"):
                    raise OSError("read error")
                return real_open(p, *a, **k)

            with mock.patch("builtins.open", flaky_open):
                self.assertIsNone(bc._drain_injected_command())

    def test_whitespace_only_file_returns_none(self):
        # A snapshot of only whitespace -> raw == '' branch (12919-12922).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write("   \n  ")
            self.assertIsNone(bc._drain_injected_command())
            self.assertFalse(os.path.exists(path))

    def test_non_list_payload_returns_none(self):
        # Valid JSON that isn't a list -> isinstance guard (12932-12935).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"text": "not a list"}, f)
            self.assertIsNone(bc._drain_injected_command())

    def test_non_string_non_dict_head_returns_none(self):
        # First item is an int -> text stays '' (12969-12972) -> None.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([123], f)
            self.assertIsNone(bc._drain_injected_command())

    def test_requeue_write_failure_still_returns_head(self):
        # The tail-requeue write failing (12961-12962) is logged but the head
        # is still returned and the snapshot cleaned up.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["head", "tail"], f)
            with mock.patch.object(bc.tempfile, "mkstemp",
                                   side_effect=OSError("no temp space")):
                self.assertEqual(bc._drain_injected_command(), "head")

    def test_snapshot_remove_failure_swallowed_single_item(self):
        # os.remove on the .consuming snapshot failing after a clean single-item
        # read is swallowed (the bare except at 12963-12964); the head still
        # returns.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["solo"], f)
            real_remove = bc.os.remove

            def flaky_remove(p, *a, **k):
                if str(p).endswith(".consuming"):
                    raise OSError("remove blocked")
                return real_remove(p, *a, **k)

            with mock.patch.object(bc.os, "remove", flaky_remove):
                self.assertEqual(bc._drain_injected_command(), "solo")

    def test_corrupt_json_remove_failure_swallowed(self):
        # Corrupt JSON path whose snapshot os.remove ALSO fails (12929-12930).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inject.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write("{bad json")
            real_remove = bc.os.remove

            def flaky_remove(p, *a, **k):
                if str(p).endswith(".consuming"):
                    raise OSError("remove blocked")
                return real_remove(p, *a, **k)

            with mock.patch.object(bc.os, "remove", flaky_remove):
                self.assertIsNone(bc._drain_injected_command())


# ────────────────────────────────────────────────────────────────────────────
#  _speak_pending — FileNotFound during rename, empty-list snapshot, blank
#  message skip, and the bad volume_scale / bad mood coercions.
# ────────────────────────────────────────────────────────────────────────────
class SpeakPendingBranchTests(SectionSixBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoke = []
        self._p(bc, "_speak",
                lambda msg, **k: self.spoke.append((msg, k)))
        self._p(bc, "_mark_speech_spoken", lambda m: None)
        self._p(bc, "_speech_was_recently_spoken", lambda m: False)

    def _redirect(self, path):
        self._p(self.bc, "PENDING_SPEECH_PATH", path)

    def test_rename_filenotfound_returns_false(self):
        # os.replace raising FileNotFoundError (race lost) -> False (12992-12993).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "x"}], f)
            with mock.patch.object(bc.os, "replace",
                                   side_effect=FileNotFoundError()):
                self.assertFalse(bc._speak_pending())

    def test_empty_list_snapshot_returns_false(self):
        # A valid empty-list snapshot -> not items branch (13010-13013).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
            self.assertFalse(bc._speak_pending())
            self.assertFalse(os.path.exists(path))

    def test_blank_message_entries_skipped(self):
        # Items whose message is empty are skipped (13026-13027); a real one
        # still speaks.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": ""}, {"no_message_key": 1},
                           {"message": "real one"}], f)
            self.assertTrue(bc._speak_pending())
        self.assertEqual([m for m, _ in self.spoke], ["real one"])

    def test_bad_volume_scale_defaults_to_one(self):
        # A non-numeric volume_scale -> ValueError -> vol=1.0 (13034-13036).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "hi", "volume_scale": "loud"}], f)
            self.assertTrue(bc._speak_pending())
        self.assertEqual(self.spoke[0][1]["volume_scale"], 1.0)

    def test_invalid_mood_coerced_to_none(self):
        # A mood not in _VOICE_MOOD_NAMES is coerced to None (13037-13039).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "hi", "mood": "bogus_mood"}], f)
            self.assertTrue(bc._speak_pending())
        self.assertIsNone(self.spoke[0][1]["mood"])

    def test_valid_mood_passed_through(self):
        # A recognised mood survives the coercion and reaches _speak.
        bc = self.bc
        good = sorted(bc._VOICE_MOOD_NAMES)[0]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "hi", "mood": good}], f)
            self.assertTrue(bc._speak_pending())
        self.assertEqual(self.spoke[0][1]["mood"], good)

    def _flaky_consuming_remove(self):
        """os.remove that raises only for the .consuming snapshot."""
        real_remove = self.bc.os.remove

        def flaky_remove(p, *a, **k):
            if str(p).endswith(".consuming"):
                raise OSError("remove blocked")
            return real_remove(p, *a, **k)

        return flaky_remove

    def test_blank_raw_remove_failure_swallowed(self):
        # Whitespace-only snapshot whose os.remove ALSO fails (12997-12999).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write("   ")
            with mock.patch.object(bc.os, "remove", self._flaky_consuming_remove()):
                self.assertFalse(bc._speak_pending())

    def test_corrupt_queue_remove_failure_swallowed(self):
        # Corrupt JSON whose snapshot os.remove ALSO fails (13005-13006).
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write("not json")
            with mock.patch.object(bc.os, "remove", self._flaky_consuming_remove()):
                self.assertFalse(bc._speak_pending())

    def test_final_remove_failure_swallowed(self):
        # A normal spoke-something run whose FINAL os.remove fails (13053-13055):
        # the failure is swallowed and the True result still returns.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "ping"}], f)
            with mock.patch.object(bc.os, "remove", self._flaky_consuming_remove()):
                self.assertTrue(bc._speak_pending())
        self.assertEqual([m for m, _ in self.spoke], ["ping"])

    def test_empty_list_remove_failure_swallowed(self):
        # 13011-13012: a valid empty-list snapshot whose os.remove ALSO fails ->
        # the failure is swallowed and the function still returns False.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
            with mock.patch.object(bc.os, "remove", self._flaky_consuming_remove()):
                self.assertFalse(bc._speak_pending())

    def test_open_after_rename_failure_returns_false(self):
        # 13008-13009: os.replace succeeds but reopening the .consuming snapshot
        # raises (e.g. the file vanished under us) -> the outer except returns
        # False without touching the speak loop.
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pending.json")
            self._redirect(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"message": "ping"}], f)
            real_open = open

            def _flaky_open(p, *a, **k):
                if str(p).endswith(".consuming"):
                    raise OSError("snapshot vanished")
                return real_open(p, *a, **k)

            with mock.patch("builtins.open", _flaky_open):
                self.assertFalse(bc._speak_pending())
        self.assertEqual(self.spoke, [])


# ────────────────────────────────────────────────────────────────────────────
#  _maybe_orchestrate — the import-failure branch (13113-13115): core.orchestrator
#  unimportable while the gate is open -> falls through to a normal turn.
# ────────────────────────────────────────────────────────────────────────────
class MaybeOrchestrateImportFailureTests(SectionSixBase):
    def test_orchestrator_import_failure_falls_through(self):
        bc = self.bc
        # Force the lazy `from core.orchestrator import orchestrate` to raise.
        real_import = __import__

        def boom_import(name, *a, **k):
            if name == "core.orchestrator":
                raise ImportError("orchestrator module missing")
            return real_import(name, *a, **k)

        with mock.patch.object(bc, "_orchestrator_enabled", return_value=True), \
             mock.patch.object(bc, "_is_orchestration_request",
                               return_value=True), \
             mock.patch.object(bc, "_speak",
                               side_effect=AssertionError("must not speak")), \
             mock.patch("builtins.__import__", boom_import):
            self.assertFalse(bc._maybe_orchestrate("morning briefing"))


# ────────────────────────────────────────────────────────────────────────────
#  _release_singleton (fd-unlock body, 12251-12269) + _move_console_to_monitor
#  (the ctypes move path, 12486-12496).
# ────────────────────────────────────────────────────────────────────────────
class SingletonAndConsoleTests(SectionSixBase):
    def test_release_singleton_unlocks_and_closes_fd(self):
        # A real held fd drives the msvcrt unlock + os.close backstop
        # (12251-12269). _SINGLETON_HELD_FD is restored by the harness, but we
        # also null it defensively so a later test can't double-close.
        bc = self.bc
        orig = bc._SINGLETON_HELD_FD
        self.addCleanup(setattr, bc, "_SINGLETON_HELD_FD", orig)
        fd, path = tempfile.mkstemp()
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        os.write(fd, b"x")
        bc._SINGLETON_HELD_FD = fd
        bc._release_singleton()
        # The held-fd slot is cleared and the descriptor is closed (re-closing
        # raises OSError, proving it was already closed by the helper).
        self.assertIsNone(bc._SINGLETON_HELD_FD)
        with self.assertRaises(OSError):
            os.close(fd)

    def test_release_singleton_unlock_oserror_swallowed(self):
        # msvcrt.locking raising OSError is swallowed (12257-12258); the fd is
        # still closed in the finally.
        bc = self.bc
        orig = bc._SINGLETON_HELD_FD
        self.addCleanup(setattr, bc, "_SINGLETON_HELD_FD", orig)
        fd, path = tempfile.mkstemp()
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        bc._SINGLETON_HELD_FD = fd
        import msvcrt
        with mock.patch.object(msvcrt, "locking",
                               side_effect=OSError("not locked")):
            bc._release_singleton()
        self.assertIsNone(bc._SINGLETON_HELD_FD)
        with self.assertRaises(OSError):
            os.close(fd)

    def test_move_console_known_monitor_calls_setwindowpos(self):
        # A known monitor name drives the ctypes GetConsoleWindow ->
        # SetWindowPos path (12486-12494). ctypes is faked so nothing touches
        # the real window manager.
        bc = self.bc
        calls = {}
        fake_ctypes = types.ModuleType("ctypes")

        class _User32:
            def SetWindowPos(self, hwnd, after, x, y, cx, cy, flags):
                calls["args"] = (hwnd, x, y, cx, cy, flags)
                return True

        class _Kernel32:
            def GetConsoleWindow(self):
                return 4242   # truthy hwnd

        fake_ctypes.windll = types.SimpleNamespace(
            kernel32=_Kernel32(), user32=_User32())
        monitors = {"left": (-2560, 0, 2560, 1440)}
        with mock.patch.object(bc, "MONITORS", monitors), \
             mock.patch.dict(sys.modules, {"ctypes": fake_ctypes}):
            bc._move_console_to_monitor("left")
        self.assertIn("args", calls)
        # hwnd forwarded; target x/y match the monitor origin; SWP_NOSIZE set.
        self.assertEqual(calls["args"][0], 4242)
        self.assertEqual(calls["args"][1], -2560)
        self.assertEqual(calls["args"][2], 0)
        self.assertEqual(calls["args"][5], 0x0001)

    def test_move_console_no_hwnd_returns_early(self):
        # GetConsoleWindow returning 0 (no console) -> early return before
        # SetWindowPos (12490-12491).
        bc = self.bc
        fake_ctypes = types.ModuleType("ctypes")

        class _User32:
            def SetWindowPos(self, *a):
                raise AssertionError("must not move when hwnd is 0")

        class _Kernel32:
            def GetConsoleWindow(self):
                return 0

        fake_ctypes.windll = types.SimpleNamespace(
            kernel32=_Kernel32(), user32=_User32())
        monitors = {"left": (-2560, 0, 2560, 1440)}
        with mock.patch.object(bc, "MONITORS", monitors), \
             mock.patch.dict(sys.modules, {"ctypes": fake_ctypes}):
            bc._move_console_to_monitor("left")  # no exception == pass

    def test_move_console_ctypes_failure_swallowed(self):
        # An exception from the ctypes path (12495-12496) is swallowed.
        bc = self.bc
        fake_ctypes = types.ModuleType("ctypes")

        class _Kernel32:
            def GetConsoleWindow(self):
                raise RuntimeError("ctypes exploded")

        fake_ctypes.windll = types.SimpleNamespace(kernel32=_Kernel32())
        monitors = {"left": (-2560, 0, 2560, 1440)}
        with mock.patch.object(bc, "MONITORS", monitors), \
             mock.patch.dict(sys.modules, {"ctypes": fake_ctypes}):
            bc._move_console_to_monitor("left")  # no exception == pass


# ────────────────────────────────────────────────────────────────────────────
#  _preflight_api_key — the anthropic-not-importable branch (12669-12670).
# ────────────────────────────────────────────────────────────────────────────
class PreflightApiKeyImportFailureTests(SectionSixBase):
    def test_anthropic_import_failure_reported(self):
        bc = self.bc
        real_import = __import__

        def boom_import(name, *a, **k):
            if name == "anthropic":
                raise ImportError("no anthropic wheel")
            return real_import(name, *a, **k)

        with mock.patch.object(bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(os.environ,
                             {"ANTHROPIC_API_KEY": "sk-test"}, clear=False), \
             mock.patch("builtins.__import__", boom_import):
            ok, reason = bc._preflight_api_key(timeout_sec=1.0)
        self.assertFalse(ok)
        self.assertIn("not importable", reason)


if __name__ == "__main__":
    unittest.main()
