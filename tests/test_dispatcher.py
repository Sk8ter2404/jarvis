"""Logic tests for core/dispatcher.py — the command-chain resolver.

Despite the filename, this module is the multi-step intent resolver
(command_chain_resolver). It (a) splits an utterance into command segments,
(b) matches each segment against an ordered intent-rule table, building
ChainStep/ChainResult plan objects, and (c) optionally dispatches the plan
against a live ACTIONS dict, catching per-step failures.

Tests drive the public surface — match_single_intent, command_chain_resolver,
resolve_and_dispatch — plus the parsing/segmentation/formatting helpers, and
assert concrete actions, args, confirmations, and consolidated text. Action
callables are plain in-process mocks (no hardware, no network).
"""
from __future__ import annotations

import unittest

from core import dispatcher as d


# Superset of actions the rules can target, so every rule is eligible.
ACTIONS = [
    "play_music", "pause_music", "resume_music", "next_song", "previous_song",
    "focus_mode", "set_timer", "volume_up", "volume_down", "volume_mute",
    "screenshot", "show_tasks",
]


# ── Dataclasses ──────────────────────────────────────────────────────────────
class DataclassTests(unittest.TestCase):
    def test_chainstep_fields(self):
        step = d.ChainStep(action="play_music", arg="jazz",
                           confirmation="music queued", source="play jazz")
        self.assertEqual(step.action, "play_music")
        self.assertEqual(step.arg, "jazz")
        self.assertEqual(step.confirmation, "music queued")
        self.assertEqual(step.source, "play jazz")

    def test_chainresult_defaults_are_independent_lists(self):
        a = d.ChainResult()
        b = d.ChainResult()
        self.assertEqual(a.steps, [])
        self.assertEqual(a.unknown, [])
        a.steps.append("x")
        # default_factory must give each instance its own list.
        self.assertEqual(b.steps, [])


# ── match_single_intent (single-segment matching) ───────────────────────────
class MatchSingleIntentTests(unittest.TestCase):
    def _mi(self, utterance, acts=ACTIONS):
        step = d.match_single_intent(utterance, acts)
        return None if step is None else (step.action, step.arg,
                                          step.confirmation)

    def test_play_music_extracts_query(self):
        self.assertEqual(self._mi("play Michael Jackson"),
                         ("play_music", "Michael Jackson", "music queued"))

    def test_play_some_strips_filler(self):
        self.assertEqual(self._mi("play some jazz"),
                         ("play_music", "jazz", "music queued"))

    def test_focus_timer_routes_to_focus_not_timer(self):
        # The focus rule precedes the generic timer rule, so
        # '45 minute focus timer' must become focus_mode with a duration arg.
        self.assertEqual(self._mi("start a 45 minute focus timer"),
                         ("focus_mode", "45 minutes", "focus mode armed"))

    def test_plain_focus_mode_no_duration(self):
        self.assertEqual(self._mi("focus mode"),
                         ("focus_mode", "", "focus mode armed"))

    def test_set_timer_default_message(self):
        # No 'for X' clause → message defaults to 'timer'.
        self.assertEqual(self._mi("set a 10 minute timer"),
                         ("set_timer", "10 minutes | timer", "timer set"))

    def test_set_timer_with_message(self):
        self.assertEqual(self._mi("set a 5 minute timer for tea"),
                         ("set_timer", "5 minutes | tea", "timer set"))

    def test_remind_me_maps_to_timer(self):
        self.assertEqual(self._mi("remind me in 20 minutes to stretch"),
                         ("set_timer", "20 minutes | stretch", "timer set"))

    def test_volume_and_mute(self):
        self.assertEqual(self._mi("turn it up"),
                         ("volume_up", "", "volume up"))
        self.assertEqual(self._mi("mute"),
                         ("volume_mute", "", "muted"))

    def test_screenshot(self):
        self.assertEqual(self._mi("take a screenshot"),
                         ("screenshot", "", "screenshot captured"))

    def test_lead_filler_stripped_before_match(self):
        self.assertEqual(self._mi("hey jarvis play Bowie"),
                         ("play_music", "Bowie", "music queued"))

    def test_no_rule_match_returns_none(self):
        self.assertIsNone(self._mi("what is the meaning of life"))

    def test_empty_and_whitespace_return_none(self):
        self.assertIsNone(self._mi(""))
        self.assertIsNone(self._mi("   "))

    def test_rule_skipped_when_action_not_registered(self):
        # play_music absent and no fallback present → no match at all.
        self.assertIsNone(self._mi("play jazz", ["screenshot"]))

    def test_fallback_action_used_when_primary_absent(self):
        # play_music absent but 'spotify' fallback present → routes to spotify.
        step = d.match_single_intent("play jazz", ["spotify"])
        self.assertIsNotNone(step)
        self.assertEqual(step.action, "spotify")
        self.assertEqual(step.arg, "jazz")


# ── command_chain_resolver (pure plan) ───────────────────────────────────────
class ChainResolverTests(unittest.TestCase):
    def test_two_step_chain(self):
        r = d.command_chain_resolver(
            "play Michael Jackson and start a 45 minute focus timer", ACTIONS)
        self.assertIsNotNone(r)
        self.assertEqual([(s.action, s.arg) for s in r.steps],
                         [("play_music", "Michael Jackson"),
                          ("focus_mode", "45 minutes")])
        self.assertEqual(r.unknown, [])

    def test_three_step_chain_with_strong_separators(self):
        r = d.command_chain_resolver(
            "play jazz, then take a screenshot, also turn it up", ACTIONS)
        self.assertIsNotNone(r)
        self.assertEqual([s.action for s in r.steps],
                         ["play_music", "screenshot", "volume_up"])

    def test_single_segment_returns_none(self):
        # No chain separator → not a chain; fall through to the LLM.
        self.assertIsNone(d.command_chain_resolver("play jazz", ACTIONS))

    def test_one_match_one_unknown_returns_none(self):
        # Needs >=2 MATCHED steps; one match + one unknown is just a sentence.
        self.assertIsNone(
            d.command_chain_resolver("play jazz and ponder the universe",
                                     ACTIONS))

    def test_entity_with_and_not_split(self):
        # 'Earth Wind and Fire' must stay intact (RHS 'Fire' isn't a verb),
        # leaving a single segment → None.
        self.assertIsNone(
            d.command_chain_resolver("play Earth Wind and Fire", ACTIONS))

    def test_unknown_segment_captured_via_comma(self):
        # A comma forces a 3-way split; the middle non-command becomes unknown
        # while the two real commands still form a chain.
        r = d.command_chain_resolver(
            "play jazz, ponder the universe, take a screenshot", ACTIONS)
        self.assertIsNotNone(r)
        self.assertEqual([s.action for s in r.steps],
                         ["play_music", "screenshot"])
        self.assertEqual(r.unknown, ["ponder the universe"])

    def test_empty_utterance_returns_none(self):
        self.assertIsNone(d.command_chain_resolver("", ACTIONS))
        self.assertIsNone(d.command_chain_resolver("   ", ACTIONS))


# ── Segmentation + helpers ───────────────────────────────────────────────────
class SegmentationTests(unittest.TestCase):
    def test_strong_separator_splits(self):
        self.assertEqual(
            d._split_chain("play jazz, then take a screenshot"),
            ["play jazz", "take a screenshot"])

    def test_bare_and_splits_only_before_command_verb(self):
        # RHS opens with the verb 'take' → split.
        self.assertEqual(
            d._split_chain("play jazz and take a screenshot"),
            ["play jazz", "take a screenshot"])
        # RHS 'the Jackson 5' is not a command → stays one segment.
        self.assertEqual(
            d._split_chain("play Michael Jackson and the Jackson 5"),
            ["play Michael Jackson and the Jackson 5"])

    def test_strip_lead_filler_single_pass(self):
        # Strips the first matching lead-in only (does not recurse), so
        # 'please' remains after 'could you ' is removed.
        self.assertEqual(d._strip_lead_filler("Could you please play jazz"),
                         "please play jazz")
        self.assertEqual(d._strip_lead_filler("okay jarvis screenshot"),
                         "screenshot")

    def test_looks_like_command_start(self):
        self.assertTrue(d._looks_like_command_start("play jazz"))
        # Leading article is skipped to peek at the real verb.
        self.assertTrue(d._looks_like_command_start("the volume up"))
        self.assertFalse(d._looks_like_command_start("banana split"))
        self.assertFalse(d._looks_like_command_start(""))


# ── _resolve_action ──────────────────────────────────────────────────────────
class ResolveActionTests(unittest.TestCase):
    RULE = {"action": "play_music", "fallbacks": ["apple_music", "spotify"]}

    def test_primary_preferred(self):
        self.assertEqual(d._resolve_action(self.RULE, ["play_music", "spotify"]),
                         "play_music")

    def test_first_available_fallback(self):
        self.assertEqual(d._resolve_action(self.RULE, ["spotify", "screenshot"]),
                         "spotify")

    def test_none_available(self):
        self.assertIsNone(d._resolve_action(self.RULE, ["screenshot"]))


# ── Failure-result detection + phrase compression ────────────────────────────
class FailureResultTests(unittest.TestCase):
    def test_failure_markers_detected(self):
        self.assertTrue(d._is_failure_result("Could not find any tracks"))
        self.assertTrue(d._is_failure_result("operation failed"))
        self.assertTrue(d._is_failure_result("COM refused the call"))

    def test_non_failures_and_non_strings(self):
        self.assertFalse(d._is_failure_result("all good"))
        self.assertFalse(d._is_failure_result(""))
        self.assertFalse(d._is_failure_result(None))
        self.assertFalse(d._is_failure_result(123))

    def test_phrase_takes_first_sentence_lowercased(self):
        self.assertEqual(
            d._failure_phrase("Could not find tracks. More detail here.",
                              "music queued"),
            "could not find tracks")

    def test_phrase_preserves_jarvis_token(self):
        self.assertEqual(d._failure_phrase("JARVIS could not comply", "fb"),
                         "JARVIS could not comply")

    def test_phrase_empty_uses_fallback(self):
        self.assertEqual(d._failure_phrase("", "timer set"), "timer set failed")

    def test_phrase_truncated_to_80_chars(self):
        out = d._failure_phrase("x" * 100, "fb")
        self.assertEqual(len(out), 80)
        self.assertTrue(out.endswith("..."))


# ── Canonical failure-marker list (shared with the monolith) ─────────────────
class CanonicalFailureMarkersTests(unittest.TestCase):
    """The marker list was de-duplicated into core/failure_markers.py and is now
    imported by BOTH core/dispatcher.py and bobert_companion._is_failure. These
    tests pin the canonical contents and prove the extraction didn't change how
    any historical result string classifies."""

    # The exact set both sites carried before the merge (bobert_companion used
    # "REFUSED"; the dispatcher used "refused" — identical under the
    # case-insensitive match both apply). This is the behavioural contract.
    _EXPECTED = {
        "could not", "failed", "refused", "no tracks found",
        "no window matching", "unknown ", "format:",
    }

    def test_canonical_contents(self):
        from core.failure_markers import FAILURE_MARKERS
        self.assertEqual(set(FAILURE_MARKERS), self._EXPECTED)
        # Markers are stored lower-case so the case-insensitive compare is a
        # plain substring test on the dispatcher side.
        self.assertTrue(all(m == m.lower() for m in FAILURE_MARKERS))

    def test_dispatcher_uses_the_canonical_list(self):
        from core.failure_markers import FAILURE_MARKERS
        # The dispatcher's module-level alias must BE the shared tuple, so the
        # two can never drift again.
        self.assertIs(d._FAIL_MARKERS, FAILURE_MARKERS)

    def test_every_marker_classifies_as_failure(self):
        # Each marker, embedded in a realistic result string, trips the check —
        # covers both the dispatcher's lower-cased markers and the monolith's
        # historical upper-case "REFUSED" (matched case-insensitively).
        samples = {
            "could not": "Could not capture screen",
            "failed": "operation failed",
            "refused": "COM REFUSED the call",       # upper in the wild
            "no tracks found": "no tracks found matching 'x' in iTunes library",
            "no window matching": "no window matching 'Spotify'",
            "unknown ": "unknown command foo",
            "format:": "bad format: expected HH:MM",
        }
        for marker, text in samples.items():
            self.assertTrue(d._is_failure_result(text),
                            f"marker {marker!r} should flag {text!r}")

    def test_union_preserves_non_failure_classification(self):
        # Strings that legitimately are NOT failures must still pass clean —
        # the extraction added no new marker, so nothing newly trips.
        for ok in ("all good", "music queued", "screenshot captured",
                   "done, sir", "playing your focus mix"):
            self.assertFalse(d._is_failure_result(ok), ok)


# ── _format_consolidated ─────────────────────────────────────────────────────
class FormatConsolidatedTests(unittest.TestCase):
    def _steps(self, *confs):
        return [d.ChainStep(action="a", arg="", confirmation=c, source="s")
                for c in confs]

    def test_count_word_and_confirmations(self):
        line = d._format_consolidated(
            self._steps("music queued", "timer set", "muted"), [])
        self.assertEqual(
            line, "Three things, sir: music queued, timer set, muted.")

    def test_single_unknown_noted(self):
        line = d._format_consolidated(
            self._steps("music queued", "timer set"), ["banana"])
        self.assertEqual(
            line,
            "Two things, sir: music queued, timer set."
            " I didn't catch 'banana', though.")

    def test_multiple_unknown_counted(self):
        line = d._format_consolidated(
            self._steps("music queued", "timer set"), ["x", "y"])
        self.assertTrue(line.endswith("2 other items I couldn't place."))


# ── resolve_and_dispatch (plan + execute) ────────────────────────────────────
class ResolveAndDispatchTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def _action(self, name, rv="ok"):
        def fn(arg):
            self.calls.append((name, arg))
            return rv
        return fn

    def test_happy_two_step_runs_both_and_summarises(self):
        actions = {"play_music": self._action("play_music"),
                   "focus_mode": self._action("focus_mode")}
        out = d.resolve_and_dispatch(
            "play jazz and start a 45 minute focus timer", actions)
        self.assertEqual(out, "Two things, sir: music queued, focus mode armed.")
        self.assertEqual(self.calls,
                         [("play_music", "jazz"), ("focus_mode", "45 minutes")])

    def test_no_chain_returns_none(self):
        actions = {"play_music": self._action("play_music")}
        self.assertIsNone(d.resolve_and_dispatch("play jazz", actions))
        self.assertEqual(self.calls, [])

    def test_failure_result_surfaced_in_line(self):
        # An action that returns a failure marker (without raising) has its
        # own message compressed into the consolidated reply.
        actions = {
            "play_music": self._action("play_music", "Could not find any tracks."),
            "screenshot": self._action("screenshot", "captured"),
        }
        out = d.resolve_and_dispatch("play jazz, take a screenshot", actions)
        self.assertEqual(
            out,
            "Two things, sir: could not find any tracks, screenshot captured.")

    def test_exception_in_step_is_caught_and_flagged(self):
        def boom(arg):
            raise RuntimeError("nope")
        actions = {"play_music": boom,
                   "screenshot": self._action("screenshot", "captured")}
        out = d.resolve_and_dispatch("play jazz, take a screenshot", actions)
        # The failing step keeps its slot but is flagged with the exc type;
        # the other step still runs.
        self.assertEqual(
            out,
            "Two things, sir: music queued failed (RuntimeError),"
            " screenshot captured.")
        self.assertEqual(self.calls, [("screenshot", "")])

    def test_missing_action_demoted_below_chain_floor(self):
        # Both resolved actions are absent from the dict at dispatch time, so
        # <2 steps survive → fall through to the LLM (None).
        out = d.resolve_and_dispatch(
            "play jazz and take a screenshot", {})
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
