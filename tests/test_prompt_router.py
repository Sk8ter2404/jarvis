"""Tests for core/prompt_router — dynamic local-prompt slimming (2026-07-15).

The local brain's context is capped at 12-16k tokens but the full system prompt
is ~30k, so it was TRUNCATED. The router keeps the core + only the sections a
turn needs, so the relevant instructions fit uncut. These pin: correct parsing,
relevant-section selection, the always-present core, the drop INDEX, big size
reduction, and never-raises.
"""
from __future__ import annotations

import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import prompts, prompt_router as pr   # noqa: E402

FULL = prompts.PC_CONTROL_PROMPT


class SplitTests(unittest.TestCase):
    def test_parses_core_and_sections(self):
        core, sections = pr.split_pc_control(FULL)
        self.assertTrue(len(core) > 500, "core preamble must be substantial")
        self.assertGreaterEqual(len(sections), 8,
                                "PC_CONTROL should split into many named sections")
        names = [h for h, _ in sections]
        self.assertIn("MUSIC CONTROLS", names)
        self.assertTrue(any("BAMBU" in n for n in names))

    def test_no_headers_returns_whole_as_core(self):
        core, sections = pr.split_pc_control("just some text, no headers here")
        self.assertEqual(sections, [])
        self.assertIn("just some text", core)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)

    def test_music_query_includes_music_excludes_printer(self):
        inc, drop = pr.select_sections("play some relaxing jazz", self.sections)
        self.assertIn("MUSIC CONTROLS", inc)
        self.assertTrue(any("BAMBU" in d for d in drop),
                        "a music query must NOT load the huge 3D-printer section")

    def test_printer_query_includes_printer(self):
        inc, _drop = pr.select_sections("is my 3d print finished", self.sections)
        self.assertTrue(any("BAMBU" in i for i in inc))

    def test_app_launching_is_always_included(self):
        # even an unrelated query keeps the fundamental app-launch grammar
        inc, _drop = pr.select_sections("tell me a joke", self.sections)
        self.assertIn("MULTI-MONITOR APP LAUNCHING", inc)

    def test_health_query_includes_health(self):
        inc, _drop = pr.select_sections("what's my cpu temperature", self.sections)
        self.assertIn("SYSTEM HEALTH", inc)


class SlimTests(unittest.TestCase):
    def test_slim_is_much_smaller_for_common_turn(self):
        slim = pr.slim_pc_control("what time is it", FULL)
        self.assertLess(len(slim), len(FULL) * 0.55,
                        "a common turn should drop well over 40% of the prompt")

    def test_slim_keeps_core_and_names_dropped_sections(self):
        slim = pr.slim_pc_control("play music", FULL)
        # the drop INDEX advertises what was left out so the model still knows
        self.assertIn("ADDITIONAL CAPABILITIES", slim)
        # the huge printer section is dropped but named in the index
        self.assertNotIn("BAMBU 3D PRINTER (H2D):\n", slim.replace(
            "ADDITIONAL CAPABILITIES", ""))  # body not present
        self.assertIn("BAMBU", slim)  # but named in the index

    def test_printer_turn_actually_loads_printer_body(self):
        slim = pr.slim_pc_control("start the 3d printer", FULL)
        # the section BODY (its original header line, not just the head name in
        # the INDEX) must be present — proves the section loaded, not merely
        # got listed. (Size is no longer a proxy: after the 2026-07-15 header-
        # regex fix the printer section stopped absorbing 4 unrelated blocks.)
        self.assertIn("BAMBU 3D PRINTER (H2D):", slim)
        self.assertGreater(len(slim), 4000, "printer body content is included")

    def test_never_raises_returns_full_on_bad_input(self):
        # a prompt with no sections just comes back whole
        self.assertEqual(pr.slim_pc_control("x", "no sections in here"),
                         "no sections in here")

    def test_slim_fits_local_window_for_common_turns(self):
        # BASE identity (~4.6k tok) + slim PC + ~800 tok rules/phrasebook must
        # clear the 16k local window for a representative spread of turns.
        base = len(prompts.BASE_SYSTEM_PROMPT) // 4
        for q in ("who are you", "open chrome", "set a timer for 10 minutes",
                  "what's my gpu temp", "remind me to call mom"):
            total = base + len(pr.slim_pc_control(q, FULL)) // 4 + 800
            self.assertLess(total, 16000,
                            f"{q!r} slim prompt must fit the local window: {total}")


class HeaderRegexRegressionTests(unittest.TestCase):
    """Locks in the 2026-07-15 fix. The header regex used to require the WHOLE
    line be uppercase, so it matched only 12 of ~54 real headers — the dominant
    style is 'TITLE (lowercase parenthetical):'. The other ~42 capability blocks
    were silently folded into the preceding matched section (bloating it) and
    vanished from BOTH keyword routing and the INDEX safety net. This suite fails
    if that regression returns."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.names = [n for n, _ in self.sections]

    def test_recognizes_many_sections(self):
        self.assertGreaterEqual(
            len(self.sections), 50,
            "parenthetical headers must be recognized (was a broken 13)")

    def test_recognizes_parenthetical_headers(self):
        for want in ("SMART HOME", "SCREEN VISION", "SELF-PRESERVATION",
                     "EMAIL TRIAGE", "IMAGE GENERATION", "AUDIO OUTPUT DEVICE",
                     "MORNING BRIEFING", "NEWS BRIEFING", "SCHEDULING",
                     "PHONE NOTIFICATIONS", "BROWSER AGENT", "LOCAL MODEL SELECTION"):
            self.assertIn(want, self.names, f"{want!r} header not recognized")

    def test_head_name_strips_parenthetical(self):
        # the section NAME is the uppercase head, not the descriptive paren
        self.assertIn("BAMBU 3D PRINTER", self.names)
        self.assertNotIn("BAMBU 3D PRINTER (H2D)", self.names)

    def test_every_section_has_routing_or_is_always(self):
        # No section may be stranded index-only with zero keywords: a turn that
        # names the capability must be able to load its full instructions.
        uncovered = [n for n in self.names
                     if n.upper() not in pr._ALWAYS and not pr._keywords_for(n)]
        self.assertEqual(uncovered, [],
                         f"sections without keyword routing: {uncovered}")

    def test_previously_merged_queries_route_correctly(self):
        for q, want in (("turn on the living room lights", "SMART HOME"),
                        ("what's in the news", "NEWS BRIEFING"),
                        ("what's on my screen", "SCREEN VISION"),
                        ("check my email", "EMAIL TRIAGE"),
                        ("switch audio to my headset", "AUDIO OUTPUT DEVICE"),
                        ("which local model are you using", "LOCAL MODEL SELECTION")):
            inc, _drop = pr.select_sections(q, self.sections)
            self.assertIn(want, inc, f"{q!r} should load {want!r}, got {inc}")

    def test_core_preamble_keeps_the_action_grammar(self):
        # the universal [ACTION: ...] grammar must stay in the always-included
        # core preamble, not get demoted into a keyword-gated section.
        self.assertIn("[ACTION:", self.core)
        self.assertIn("open_url", self.core)


class WrappedHeaderRegressionTests(unittest.TestCase):
    """Locks in the 2026-07-15 SECOND header fix. 14 real headers wrap their
    descriptive parenthetical across lines, and 3 more used '+'/em-dash/lowercase
    in the head — all invisible to the single-line uppercase-only matcher, so
    those 17 capability blocks were folded into a neighbour and dropped from the
    INDEX. This suite fails if any of that regresses."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.names = [n for n, _ in self.sections]

    def test_previously_wrapped_headers_now_recognized(self):
        for want in ("AIR CONTROL", "STREAMING SERVICES", "TASTE-AWARE MUSIC",
                     "FOCUS MODE / DO-NOT-DISTURB", "WEB INTERFACE", "CALENDAR",
                     "WEATHER BRIEFING", "PATTERN LEARNING", "REPO ROBOT PROJECT",
                     "SUIT DIAGNOSTICS", "LOCAL VOICE CLONE", "WAKE-WORD MODE",
                     "WELLNESS / FOCUS NUDGES"):
            self.assertIn(want, self.names, f"wrapped header {want!r} not recognized")

    def test_punctuated_and_versioned_headers_recognized(self):
        for want in ("MUSIC + VIDEO PLAYBACK", "SMART HOME — PER-BRAND LIST",
                     "KINECT DEPTH SENSOR", "MULTI-STEP TASKS"):
            self.assertIn(want, self.names, f"{want!r} not recognized")

    def test_no_wrapped_header_left_unmatched(self):
        # after join, no line should look like a header TAIL (ends in '):' with
        # its '(' on an earlier line) — that shape means a header still wrapped.
        joined = pr._join_wrapped_headers(FULL.split("\n"))
        orphan_tails = [l.strip() for l in joined
                        if l.strip().endswith("):") and "(" not in l]
        self.assertEqual(orphan_tails, [],
                         f"headers still wrapping across lines: {orphan_tails}")

    def test_weather_briefing_body_loads_on_weather_turn(self):
        # the concrete symptom that started this: a weather turn must load the
        # real WEATHER BRIEFING instructions, not just see its name in the INDEX.
        slim = pr.slim_pc_control("what's the weather going to be", FULL)
        self.assertIn("WEATHER BRIEFING", slim)
        self.assertIn("weather_briefing", slim)  # the action name from its body

    def test_every_section_including_new_ones_has_routing(self):
        uncovered = [n for n in self.names
                     if n.upper() not in pr._ALWAYS and not pr._keywords_for(n)]
        self.assertEqual(uncovered, [],
                         f"recognized sections without keyword routing: {uncovered}")


# A single-quoted phrase whose opening quote isn't a contraction apostrophe
# (preceded by a word char) and whose closing quote isn't one either (followed
# by a word char). Phrases containing an apostrophe or <placeholders> are
# deliberately un-extractable — the invariant tests silently skip them.
_QUOTED_PHRASE_RE = re.compile(r"(?<!\w)'([^'<>]{2,60})'(?!\w)")
# Only plain utterance-looking phrases (letters/digits/spaces/hyphens) — drops
# prompt-internal quoted fragments like 'overnight first?' or 'shuffle '.
_PLAIN_PHRASE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 \-]*$")


def _quoted_phrases(text: str) -> list[str]:
    """Well-formed single-quoted utterances in `text` (whitespace-normalized)."""
    out = []
    for ph in _QUOTED_PHRASE_RE.findall(" ".join(text.split())):
        ph = ph.strip()
        if _PLAIN_PHRASE_RE.match(ph):
            out.append(ph)
    return out


class VolumeRoutingRegressionTests(unittest.TestCase):
    """2026-07-21 audit: the 'Volume control:' block sat inside STREAMING
    SERVICES (whose keywords are netflix/hulu/…), MUSIC CONTROLS had no volume
    actions in its body, and 'mute' was in NO keyword list — so on the default
    slim local path every volume turn shipped a prompt with no volume grammar
    at all. The existing VolumeGrammarTests only checked the FULL prompt, which
    is exactly why this regressed invisibly."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)

    def test_volume_turns_ship_volume_grammar_in_slim_prompt(self):
        for q in ("set the volume to 30 percent", "turn the music down a bit",
                  "mute the audio", "make it louder"):
            slim = pr.slim_pc_control(q, FULL)
            for want in ("set_volume", "volume_up", "volume_mute"):
                self.assertIn(want, slim,
                              f"{q!r}: SLIM prompt must document {want!r}")
            self.assertIn("[ACTION: set_volume,", slim,
                          f"{q!r}: absolute-volume example must be present")

    def test_volume_doc_placement_invariant(self):
        # Survives future section reshuffles: WHEREVER the set_volume doc
        # lives, volume/mute turns must load that section. If someone moves the
        # volume block into a section whose keywords don't fire on volume
        # turns, this fails regardless of which section it landed in.
        homes = [h for h, b in self.sections if "set_volume" in b]
        self.assertTrue(homes, "some section must document set_volume")
        for q in ("set the volume to 30 percent", "mute the audio"):
            inc, _ = pr.select_sections(q, self.sections)
            for h in homes:
                self.assertIn(h, inc,
                              f"{q!r} must load {h!r} (the set_volume home)")


class UnifiedCameraRoutingRegressionTests(unittest.TestCase):
    """2026-07-21 audit: the indented 'UNIFIED (…)' sub-header is promoted to
    its own section (the parser matches headers on the stripped line), but its
    keyword list covered only the 'all cameras' phrasings — every trigger
    phrase its body documents ('where am I', 'camera status', 'look around',
    …) loaded nothing."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.bodies = dict(self.sections)

    def test_unified_section_exists_with_camera_actions(self):
        names = [n for n, _ in self.sections]
        self.assertIn("UNIFIED", names)
        body = self.bodies["UNIFIED"]
        for act in ("camera_status", "where_am_i", "look_around"):
            self.assertIn(act, body)

    def test_documented_trigger_phrases_load_unified(self):
        for q in ("camera status", "what cameras do you have", "where am I",
                  "what am I doing", "what's my status", "look around",
                  "what do you see everywhere", "all cameras"):
            inc, _ = pr.select_sections(q, self.sections)
            self.assertIn("UNIFIED", inc, f"{q!r} must load UNIFIED")

    def test_no_cameras_guard_lives_in_webcam_awareness(self):
        # 'can you see me' turns load only WEBCAM AWARENESS — the do-not-claim-
        # blindness guard must ride along with it, not sit in UNIFIED.
        self.assertIn("do not claim 'I have no cameras'",
                      self.bodies["WEBCAM AWARENESS"])

    def test_every_promoted_indented_subheader_has_keywords(self):
        # Bug-class invariant: ANY indented line the parser promotes to its own
        # section must have a keyword route — otherwise its documented phrases
        # can outrun its (empty) keyword list and the whole block goes dark.
        joined = pr._join_wrapped_headers(FULL.split("\n"))
        promoted = []
        for ln in joined:
            if ln and ln != ln.lstrip():
                m = pr._HEADER_RE.match(ln.strip())
                if m:
                    promoted.append(m.group("head").strip())
        self.assertTrue(promoted, "expected some indented sub-headers")
        missing = [h for h in promoted
                   if h.upper() not in pr._ALWAYS and not pr._keywords_for(h)]
        self.assertEqual(missing, [],
                         f"promoted sub-headers without keywords: {missing}")

    def test_unified_documented_phrases_route_back(self):
        # Every extractable quoted phrase the UNIFIED body documents must pull
        # UNIFIED back in — the exact outage this card fixed.
        phrases = _quoted_phrases(self.bodies["UNIFIED"])
        self.assertTrue(phrases, "UNIFIED body should document trigger phrases")
        for ph in phrases:
            inc, _ = pr.select_sections(ph, self.sections)
            self.assertIn("UNIFIED", inc, f"documented phrase {ph!r} must "
                          f"load UNIFIED, got {inc}")


class LifecycleRoutingRegressionTests(unittest.TestCase):
    """2026-07-21 audit: restart/hide_hud/show_hud/toggle_hud/arc_reactor/
    holographic-overlay docs live in TASK QUEUE, whose keywords were only
    queue-shaped — so 'restart yourself' loaded the SHUTDOWN aliases (power-
    off!) with no restart doc, and 'hide the HUD' loaded nothing at all."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.bodies = dict(self.sections)

    def test_restart_phrases_load_restart_doc(self):
        for q in ("restart yourself", "reboot", "restart", "relaunch"):
            slim = pr.slim_pc_control(q, FULL)
            inc, _ = pr.select_sections(q, self.sections)
            self.assertIn("TASK QUEUE", inc, f"{q!r} must load TASK QUEUE")
            self.assertIn("relaunch JARVIS immediately", slim,
                          f"{q!r}: the restart doc must be in the slim prompt")

    def test_never_shutdown_only_power_prompt_for_restart_phrases(self):
        # Invariant phrasing of the exact failure: the slim prompt may never
        # advertise the power-off aliases while omitting the restart action.
        # Survives future section reshuffles.
        for q in ("restart yourself", "reboot"):
            slim = pr.slim_pc_control(q, FULL)
            self.assertFalse(
                "exit_jarvis" in slim and
                "relaunch JARVIS immediately" not in slim,
                f"{q!r}: slim prompt advertises shutdown aliases without the "
                f"restart action — the model's only documented move is power-off")

    def test_hud_and_reactor_phrases_load_their_docs(self):
        for q, want in (("hide the HUD", "hide_hud"),
                        ("show the HUD", "show_hud"),
                        ("toggle hud", "toggle_hud"),
                        ("show the arc reactor", "arc_reactor"),
                        ("show the holographic overlay",
                         "show_holographic_overlay")):
            slim = pr.slim_pc_control(q, FULL)
            self.assertIn(want, slim, f"{q!r} must ship the {want!r} doc")

    def test_task_queue_documented_says_phrases_route_back(self):
        # Bug-class invariant: every "…says 'phrase'…" trigger the TASK QUEUE
        # body documents must route TASK QUEUE back in. Any future action
        # documented here whose phrases outrun the keyword list fails this,
        # regardless of which copy of the rule was edited.
        text = " ".join(self.bodies["TASK QUEUE"].split())
        phrases = []
        for sent in re.split(r"(?<=\.)\s+", text):
            if not re.search(r"says\s+'", sent):
                continue
            seg = sent.split("says", 1)[1]
            # stop at parenthetical asides / em-dash explanations — the quoted
            # run of trigger phrases always precedes them.
            seg = seg.split("(", 1)[0].split(" — ", 1)[0]
            phrases.extend(_quoted_phrases(seg))
        self.assertGreater(len(phrases), 20,
                           "extraction should find the documented triggers")
        misses = []
        for ph in phrases:
            inc, _ = pr.select_sections(ph, self.sections)
            if "TASK QUEUE" not in inc:
                misses.append(ph)
        self.assertEqual(misses, [],
                         f"documented TASK QUEUE phrases that no longer route "
                         f"back: {misses}")


class SafetyTrailerRegressionTests(unittest.TestCase):
    """2026-07-21 audit: the SAFETY (confirmation-hold) + ASK FIRST rules
    trailed the last section header, so split_pc_control folded them into
    SHUTDOWN ALIASES — and 'buy…'/'delete…' turns on the default slim local
    path shipped a prompt with no confirmation-hold rule at all."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)

    def test_safety_rules_live_in_always_shipped_core(self):
        self.assertIn("held for confirmation", self.core)
        self.assertIn("ASK FIRST", self.core)

    def test_purchase_and_delete_turns_ship_the_rules(self):
        for q in ("buy me a new keyboard on amazon", "delete that file",
                  "what time is it"):
            slim = pr.slim_pc_control(q, FULL)
            self.assertIn("held for confirmation", slim,
                          f"{q!r}: slim prompt must carry the confirmation hold")
            self.assertIn("ASK FIRST", slim,
                          f"{q!r}: slim prompt must carry the ask-first rule")

    def test_no_section_swallows_the_trailer(self):
        swallowed = [h for h, b in self.sections if "ASK FIRST" in b]
        self.assertEqual(swallowed, [],
                         f"safety trailer folded into section(s): {swallowed}")

    def test_rules_are_single_sourced(self):
        self.assertEqual(FULL.count("ASK FIRST"), 1,
                         "exactly one copy of the safety rules in the prompt")
        self.assertIn(prompts.PC_CONTROL_SAFETY_RULES, FULL,
                      "the prompt copy must BE the shared constant")

    def test_local_cheatsheet_references_the_constant(self):
        # Stale-duplicate guard for the second local-prompt path: source-scan
        # the monolith (no import) and require _local_cheatsheet to reference
        # PC_CONTROL_SAFETY_RULES, so the JARVIS_DYNAMIC_LOCAL_PROMPT=0
        # fallback can never silently drop the rules again.
        path = os.path.join(_PROJECT, "bobert_companion.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        start = src.index("def _local_cheatsheet")
        end = src.index("\ndef ", start)
        body = src[start:end]
        self.assertIn("PC_CONTROL_SAFETY_RULES", body,
                      "_local_cheatsheet must reference the shared safety-rules "
                      "constant (not a pasted copy, not nothing)")


class AmbientLearningRegressionTests(unittest.TestCase):
    """2026-07-21 audit: the AMBIENT-LEARNING MODE header wrapped its
    parenthetical over 4 lines and ended in a bare ':' (not '):'), so the
    wrapped-header joiner never fired, the header never parsed, and the whole
    block silently folded into WAKE LISTENER — invisible to routing AND the
    INDEX."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.names = [n for n, _ in self.sections]

    def test_ambient_learning_is_a_parsed_section(self):
        self.assertIn("AMBIENT-LEARNING MODE", self.names)

    def test_ambient_turns_load_the_body(self):
        inc, _ = pr.select_sections("go into ambient learning mode",
                                    self.sections)
        self.assertIn("AMBIENT-LEARNING MODE", inc)
        slim = pr.slim_pc_control("just listen and learn quietly", FULL)
        self.assertIn("ambient_learning_mode_on", slim,
                      "the body must load, not just appear in the INDEX")

    def test_no_header_shaped_line_fails_to_parse(self):
        # Bug-class invariant replacing the '):'-tail heuristic: after the
        # wrapped-header join, ANY line whose stripped text still looks like a
        # section header (CAPS head + '(' or CAPS head ending in ':') must have
        # become a parsed section. Catches wrapped, bare-colon, or otherwise
        # malformed headers regardless of tail shape — this flagged exactly
        # AMBIENT-LEARNING MODE before the fix, and nothing else.
        h_paren = re.compile(r"^([A-Z][A-Z0-9 +/&.'\-—]{1,60}?)\s*\(")
        h_colon = re.compile(r"^([A-Z][A-Z0-9 +/&.'\-—]{2,60})\s*:\s*$")
        parsed = set(self.names)
        orphans = []
        for ln in pr._join_wrapped_headers(FULL.split("\n")):
            s = ln.strip()
            m = h_paren.match(s) or h_colon.match(s)
            if m and m.group(1).strip() not in parsed:
                orphans.append(s[:80])
        self.assertEqual(orphans, [],
                         f"header-shaped lines that never became sections: "
                         f"{orphans}")


if __name__ == "__main__":
    unittest.main()
