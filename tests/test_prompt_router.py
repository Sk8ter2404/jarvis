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


class ReadThisPageRoutingTests(unittest.TestCase):
    """Live 2026-10-01 20:45: "read this page for me and see if there's any
    issues" ran see_screen with no SCREEN VISION section loaded - its keywords
    only knew "screen" phrasings, so the v2.0.159 reading rules in that section
    never reached the local brain on a "page" request."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.homes = [h for h, b in self.sections if "see_screen" in b]

    def test_page_reading_requests_load_the_see_screen_section(self):
        self.assertTrue(self.homes, "some section must document see_screen")
        for q in ("read this page for me and see if there's any issues",
                  "can you read this page",
                  "what does this article say",
                  "read this for me",
                  "summarize what's on the page"):
            inc, _ = pr.select_sections(q, self.sections)
            self.assertTrue(set(self.homes) & set(inc),
                            f"{q!r} must load a see_screen section {self.homes}")


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


class BrowserAgentRoutingRegressionTests(unittest.TestCase):
    """2026-09-04 reachability work documented the rest of the browser-agent
    surface (browse_for / find_cheapest / book_appointment / fill_form and the
    status/stop/open/reset_profile controls) in core/prompts.py but left
    _SECTION_KEYWORDS["BROWSER AGENT"] at its 2026-07 shape. Result: all four
    sentences the section itself prints as "'X' -> [ACTION: Y]" routed the
    section OUT, so on the default slim local path the model saw those actions
    as an INDEX line and nothing else. "book a" is not a substring of "book me
    a haircut"; "fill the form" is not a substring of "fill that form in with".

    This is worse than an ordinary keyword miss because the always-shipped
    PC_CONTROL_SAFETY_RULES tell the model that a near-miss is worse than
    nothing: with no browser action visible, the closed list steers a turn like
    "find me the cheapest 2TB NVMe" toward "I've no way to check that, sir."
    """

    # The four flagship examples the BROWSER AGENT body prints verbatim, and
    # the action each one is documented to produce.
    FLAGSHIP = (
        ("find me the cheapest 2tb nvme", "find_cheapest"),
        ("go read up on petg nozzle temps and summarise it", "browse_for"),
        ("book me a haircut friday afternoon", "book_appointment"),
        ("fill that form in with my name and email", "fill_form"),
    )

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.bodies = dict(self.sections)

    def test_browser_agent_is_a_parsed_section(self):
        self.assertIn("BROWSER AGENT", self.bodies)

    def test_flagship_examples_load_the_browser_section(self):
        misses = [q for q, _ in self.FLAGSHIP
                  if "BROWSER AGENT" not in pr.select_sections(q, self.sections)[0]]
        self.assertEqual(misses, [],
                         f"the section's own documented examples must load it: "
                         f"{misses}")

    def test_flagship_examples_ship_their_action_name(self):
        # The INDEX line is not enough: the model must see the action itself.
        for q, action in self.FLAGSHIP:
            slim = pr.slim_pc_control(q, FULL)
            self.assertIn(action, slim,
                          f"{q!r} must ship the {action!r} doc, not just the "
                          f"capability INDEX")

    def test_documented_arrow_examples_route_back(self):
        # Bug-class invariant: in the BROWSER AGENT body every utterance
        # printed immediately before an "-> [ACTION: ...]" arrow must pull the
        # section back in. Any action documented here in future whose trigger
        # phrasing outruns the keyword list fails this automatically.
        text = " ".join(self.bodies["BROWSER AGENT"].split())
        phrases = []
        for seg in text.split("→")[:-1]:
            # everything after the previous example's closing "]" is this
            # arrow's trigger utterance(s), quoted and "/"-separated.
            phrases.extend(_quoted_phrases(seg.rsplit("]", 1)[-1]))
        self.assertGreaterEqual(
            len(phrases), 8,
            f"extraction should find the documented triggers, got {phrases}")
        misses = []
        for ph in phrases:
            inc, _ = pr.select_sections(ph, self.sections)
            if "BROWSER AGENT" not in inc:
                misses.append(ph)
        self.assertEqual(misses, [],
                         f"documented BROWSER AGENT triggers that no longer "
                         f"route back: {misses}")

    def test_no_browser_turn_ships_a_browserless_prompt(self):
        # Phrasing of the failure that survives future section reshuffles: a
        # turn the prompt itself answers with a browser action may never reach
        # the model with the closed-list rule and no browser action at all.
        for q, action in self.FLAGSHIP:
            slim = pr.slim_pc_control(q, FULL)
            self.assertFalse(
                "emit no action at all" in slim and action not in slim,
                f"{q!r}: slim prompt carries the no-action rule but not the "
                f"{action!r} it is documented to fire")


class WebsiteBuilderRoutingTests(unittest.TestCase):
    """skills/site_builder.py's build_website must reach the LOCAL model: the
    WEBSITE BUILDER section loads (and ships the action name) for the phrasings
    it documents, and stays out of an unrelated turn."""

    PHRASES = (
        "build a website for Blue Door Bakery",
        "make a landing page for my friend's shop",
        "jarvis, design a web page for the corner cafe",
        "mock up a homepage for the garage down the road",
        "can you make a site for my friend's shop in Springfield",
    )

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)

    def test_website_builder_is_a_parsed_section_documenting_the_action(self):
        bodies = dict(self.sections)
        self.assertIn("WEBSITE BUILDER", bodies)
        self.assertIn("build_website", bodies["WEBSITE BUILDER"])

    def test_documented_phrasings_load_the_section_and_ship_the_action(self):
        for q in self.PHRASES:
            inc, _ = pr.select_sections(q, self.sections)
            self.assertIn("WEBSITE BUILDER", inc, q)
            self.assertIn("build_website", pr.slim_pc_control(q, FULL), q)

    def test_unrelated_turn_leaves_it_out(self):
        inc, _ = pr.select_sections("what's the weather tomorrow",
                                    self.sections)
        self.assertNotIn("WEBSITE BUILDER", inc)


# A "'phrase' -> [ACTION: name]" example line, whitespace-normalized first so
# the many examples that wrap the arrow onto the next line still extract.
_ARROW_EXAMPLE_RE = re.compile(
    r"'([^']{2,80})'\s*\u2192\s*\[ACTION:\s*([A-Za-z_][A-Za-z0-9_]*)\]")

# A line at column 0 opening with a run of two or more ALL-CAPS words. Every
# such line in PC_CONTROL_PROMPT is a section header -- there is no column-0
# prose in this prompt -- so one that did NOT parse is a pseudo-header.
_CAPS_RUN_RE = re.compile(
    r"^([A-Z][A-Z0-9'\-]{1,}(?:[ /+&\-\u2014]+[A-Z][A-Z0-9'\-]*){1,6})")


class StatusReadBackRoutingRegressionTests(unittest.TestCase):
    """2026-09-05: the STATUS READ-BACKS block (20 <feature>_status actions)
    opened with a PROSE line -- mixed case, no trailing colon -- so _HEADER_RE
    could not match it and split_pc_control folded the whole block into SUIT
    DIAGNOSTICS, whose keywords are diagnostics-only. Measured before the fix:
    0 of the 17 trigger phrases the block itself documents selected it, and the
    dropped-section INDEX could not help because the INDEX lists section NAMES.
    That gap was not a quiet degrade: PC_CONTROL_SAFETY_RULES ships its
    closed-list rule on 100% of turns, so 'is the workshop HUD showing?' came
    back as 'I've no way to check that, sir' while workshop_hud_status was
    registered and documented. Same bug class as the audio_devices incident --
    being IN the prompt is not the same as REACHING the model."""

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.names = [n for n, _ in self.sections]
        self.bodies = dict(self.sections)

    def test_status_read_backs_is_a_parsed_section(self):
        self.assertIn("STATUS READ-BACKS", self.names,
                      "the read-back block must be its own parsed section, not "
                      "prose folded into whatever section precedes it")

    def test_status_read_backs_has_keyword_routing(self):
        self.assertTrue(pr._keywords_for("STATUS READ-BACKS"),
                        "a parsed section with no keywords is reachable only by "
                        "its header words -- which no user ever says")

    def test_documented_phrases_load_the_block_and_ship_the_action(self):
        body = self.bodies.get("STATUS READ-BACKS", "")
        self.assertTrue(body, "no STATUS READ-BACKS section to check")
        text = " ".join(body.split())
        pairs = _ARROW_EXAMPLE_RE.findall(text)
        self.assertGreaterEqual(len(pairs), 15,
                                "extraction should find the documented examples")
        misses = []
        for phrase, action in pairs:
            inc, _ = pr.select_sections(phrase, self.sections)
            slim = pr.slim_pc_control(phrase, FULL)
            if "STATUS READ-BACKS" not in inc or action not in slim:
                misses.append((phrase, action))
        self.assertEqual(misses, [],
                         f"documented read-back phrases that do not reach their "
                         f"own action: {misses}")

    def test_read_back_home_loads_wherever_it_lives(self):
        # Placement-independent, like the set_volume invariant: WHEREVER
        # workshop_hud_status / ambient_listen_status are documented, the
        # questions the owner actually asks must load that section.
        for q, action in (("is the workshop HUD showing", "workshop_hud_status"),
                          ("are you listening in the background",
                           "ambient_listen_status"),
                          ("is the weekly digest still running",
                           "weekly_digest_status")):
            homes = [h for h, b in self.sections if action in b]
            self.assertTrue(homes, f"some section must document {action!r}")
            inc, _ = pr.select_sections(q, self.sections)
            for h in homes:
                self.assertIn(h, inc,
                              f"{q!r} must load {h!r} (the {action} home)")

    def test_read_backs_are_not_folded_into_suit_diagnostics(self):
        # The concrete mis-fold: the block's actions must not be answerable
        # only through a section whose keywords are diagnostics-only.
        self.assertNotIn("workshop_hud_status",
                         self.bodies.get("SUIT DIAGNOSTICS", ""),
                         "read-back actions folded back into SUIT DIAGNOSTICS")

    def test_no_column_zero_caps_line_fails_to_parse(self):
        # Bug-class invariant, and the net that would have caught this one:
        # every column-0 line that opens with a run of ALL-CAPS words must have
        # become a parsed section. The existing header-shaped guard only looks
        # for a '(' or a trailing ':', which a prose pseudo-header has neither
        # of. Verified 2026-09-05 to flag exactly the defective line and
        # nothing else across the whole prompt.
        parsed = {h.strip() for h, _ in self.sections}
        orphans = []
        for ln in pr._join_wrapped_headers(FULL.split("\n")):
            if not ln or ln != ln.lstrip():
                continue
            m = _CAPS_RUN_RE.match(ln)
            if not m:
                continue
            head = m.group(1).strip()
            if any(head.startswith(p) or p.startswith(head) for p in parsed):
                continue
            orphans.append(ln[:80])
        self.assertEqual(orphans, [],
                         f"column-0 ALL-CAPS lines that never became sections "
                         f"(pseudo-headers -- their block is unreachable): "
                         f"{orphans}")


if __name__ == "__main__":
    unittest.main()


class LivenessIsNotUnreadableStateRegressionTests(unittest.TestCase):
    """2026-09-05 over-refusal review of the 2026-09-04 reachability work.

    That work added 'can you hear me' and 'are you deaf' to the keyword list
    for MUTE / DEAF / SLOW / WHISPER - UNREADABLE STATE, a section whose whole
    body says "there is no token to emit ... Say plainly that you cannot check
    it". So the most basic question a voice assistant ever receives was routed
    straight into a decline block, reinforced by the always-shipped preamble
    rule ("Say plainly you cannot check it: 'I'm afraid I've no way to check
    that, sir.'"). Grepping core/prompts.py for 'hear me' returned NOTHING --
    there was no counter-instruction anywhere saying the true answer is yes.

    Deleting the two keywords would not have fixed it: 'deaf' and 'slow' are
    HEADER words, so select_sections pulls this section in on 'are you deaf'
    with the keyword gone. The fix has to live in the prompt TEXT, which is
    what these tests pin."""

    LIVENESS_TURNS = ("jarvis, can you hear me", "can you hear me now",
                      "are you deaf", "hey jarvis are you there")
    # Questions that genuinely have no read-back and must still be declined.
    UNREADABLE_TURNS = ("are you muted", "is echo cancellation on",
                        "what whisper model are you using")

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.bodies = dict(self.sections)
        self.homes = [n for n in self.bodies if "UNREADABLE STATE" in n]

    def test_unreadable_state_section_is_parsed(self):
        self.assertEqual(len(self.homes), 1,
                         f"expected exactly one unreadable-state section, "
                         f"got {self.homes}")

    def test_liveness_exception_is_read_before_the_decline(self):
        body = self.bodies[self.homes[0]]
        low = body.lower()
        self.assertIn("can you hear me", low,
                      "the section that 'can you hear me' routes into must "
                      "say what the answer is")
        self.assertLess(low.index("can you hear me"),
                        body.index("No action reads any of the following"),
                        "the liveness exception must come BEFORE the "
                        "no-handler decline, not after it")

    def test_liveness_answer_ships_on_every_turn(self):
        # Not every liveness phrasing routes here -- 'are you there' loads
        # WAKE LISTENER -- so the carve-out must also ride in the core
        # preamble, which is where the competing decline rule already lives.
        self.assertIn("whether you HEARD him", self.core,
                      "the always-shipped safety rules must carve liveness "
                      "out of the no-way-to-check answer")

    def test_liveness_turns_carry_the_answer_not_just_the_refusal(self):
        for q in self.LIVENESS_TURNS:
            slim = pr.slim_pc_control(q, FULL)
            self.assertIn("whether you HEARD him", slim,
                          f"{q!r}: slim prompt must tell the model it CAN "
                          f"answer this")

    def test_carveout_is_narrow_genuine_unreadable_state_still_declines(self):
        for q in self.UNREADABLE_TURNS:
            slim = pr.slim_pc_control(q, FULL)
            self.assertIn("there is no token to emit", slim,
                          f"{q!r}: must still get the honest no-handler "
                          f"answer -- the carve-out is liveness only")

    def test_microphone_turn_still_reaches_audio_devices(self):
        # Guard the change that this review is auditing: the carve-out must
        # not disturb the 2026-09-04 what_microphone routing fix.
        slim = pr.slim_pc_control("what microphone are you using", FULL)
        self.assertIn("what_microphone", slim)


# ──────────────────────────────────────────────────────────────────────────
#  CACHE-STABLE SPLIT  (2026-09-06 latency work)
# ──────────────────────────────────────────────────────────────────────────
#
# stable_pc_block() + turn_pc_block() replace slim_pc_control() ON THE WIRE for
# the local route: the stable half stays in the system prompt (byte-identical
# every turn, so the KV prefix survives) and the volatile half rides with the
# user's message. That is a POSITION change and must never become a CONTENT
# change — the whole point is that the model still sees everything slimming
# would have given it. These tests are what make that claim checkable rather
# than asserted, because the existing suite above exercises slim_pc_control
# directly and would stay green even if the split silently dropped a section.
#
# ⚠ SCOPE — READ BEFORE TRUSTING THIS SECTION (added 2026-09-06 after the
# review of the unverified latency work). Everything below models the PRIMARY
# turn and NOTHING ELSE. `stable + turn_pc_block(u)` is the message pair
# _call_llm builds; it is not what any other call site sends. The split has
# THREE live call sites, and the ratchet below was green at 65/65 while one of
# the other two was silently shipping a fraction of the vocabulary:
#
#   get_followup_response() reuses the primary turn's system prompt (so
#   stable_pc_block, no PC_CONTROL_PROMPT) and originally added no turn block
#   at all. Because _call_local_llm's cheatsheet swap is gated on
#   `PC_CONTROL_PROMPT in sys_prompt`, that swap stopped firing too — so the
#   round lost BOTH sources of action names at once. MEASURED on this box:
#   registered action names reachable by a follow-up round fell from 135/135
#   (legacy — the full _local_cheatsheet) to 11/135, and the model answered
#   "…and then mute the volume" with the unregistered [ACTION: mute_volume]
#   while claiming success.
#
# The lesson is about THIS FILE, not that bug: a ratchet is only a ratchet for
# the call site it models. Re-deriving prompt strings here can never catch a
# WIRING regression, because `stable + turn_pc_block(u)` is the same expression
# for every round — the question is only whether a call site still sends it.
# That question is answered by driving the real function, which
# tests/test_followup_turn_context.py does; the guard below keeps this file's
# claim honest by failing if that coverage disappears.

# An action-shaped token: lowercase words joined by underscores. Deliberately
# NOT bare words — 'search'/'play' are ordinary English inside descriptions and
# would make this test pass (or fail) for reasons that have nothing to do with
# the split. Same evidence tier tests/test_audit_action_reachability.py uses.
_ACTION_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![A-Za-z0-9_])")

# Utterances spanning every routing shape the suite above cares about, plus the
# four regression classes fixed in the week before this change (webcam health,
# GPU temperature, microphone identity, browser-agent flagships).
_SPLIT_CORPUS = [
    "what time is it", "play some music", "mute the music",
    "how hot is my graphics card", "are both webcams ok",
    "what microphone are you using", "find me the cheapest 2tb nvme",
    "book me a haircut friday afternoon", "is the workshop hud showing",
    "set a timer for ten minutes", "how's the printer doing",
    "open chrome on the left monitor", "restart yourself", "shut down",
    "turn on the lights", "read me my unread email", "what version are you",
    "who am i", "run diagnostics", "back up your memory", "test the mic",
    "remind me to call mom at six", "what's the weather going to be",
    "start recording in obs", "how many people are in the room",
    "what's on my screen", "check the network", "tell me the news",
]


class CacheStableSplitTests(unittest.TestCase):
    def test_stable_block_is_byte_identical_regardless_of_turn(self):
        """The entire point: if this ever varies, the prefix cache dies and the
        ~2.9 s full prompt re-evaluation comes straight back."""
        first = pr.stable_pc_block(FULL)
        for u in _SPLIT_CORPUS:
            pr.turn_pc_block(u, FULL)          # must not mutate shared state
            self.assertEqual(pr.stable_pc_block(FULL), first,
                             f"stable_pc_block changed after {u!r}")

    def test_stable_block_keeps_the_action_grammar_and_safety_rules(self):
        stable = pr.stable_pc_block(FULL)
        self.assertIn("[ACTION:", stable,
                      "the action grammar lives in the core preamble and must "
                      "stay in the always-cached half")
        # Single-sourced safety trailer — same anchor SafetyTrailerRegression
        # Tests uses, so a reworded rule fails in one place, not two.
        self.assertIn(prompts.PC_CONTROL_SAFETY_RULES.strip()[:60], stable)

    def test_index_names_every_section(self):
        stable = pr.stable_pc_block(FULL)
        _core, sections = pr.split_pc_control(FULL)
        missing = [h.strip() for h, _b in sections if h.strip() not in stable]
        self.assertEqual(missing, [],
                         "every capability must stay named in the always-"
                         "shipped index, or the model stops knowing it exists")

    def test_split_loses_no_action_name_slimming_would_have_shipped(self):
        """THE ratchet FOR THE PRIMARY TURN. For every utterance, stable+turn
        must be a content SUPERSET of slim_pc_control — position may change,
        coverage may not.

        Scope is load-bearing, not pedantry: this models the message pair
        _call_llm builds and says nothing about the other two call sites. See
        the ⚠ SCOPE note above and FollowupRoundIsRatchetedElsewhereTests."""
        stable = pr.stable_pc_block(FULL)
        losses = []
        for u in _SPLIT_CORPUS:
            slim = pr.slim_pc_control(u, FULL)
            combined = stable + "\n" + pr.turn_pc_block(u, FULL)
            missing = (set(_ACTION_NAME_RE.findall(slim))
                       - set(_ACTION_NAME_RE.findall(combined)))
            if missing:
                losses.append(f"{u!r} loses {sorted(missing)[:8]}")
        self.assertEqual(losses, [],
                         "the cache-stable split must never ship the local "
                         "model LESS than slim_pc_control did:\n  "
                         + "\n  ".join(losses))

    def test_always_sections_are_hoisted_and_not_duplicated(self):
        """_ALWAYS sections are in every selection, so they belong in the
        stable half — and must then NOT also be in the volatile tail, or every
        turn pays for them twice."""
        _core, sections = pr.split_pc_control(FULL)
        always_bodies = [b for h, b in sections
                         if h.strip().upper() in pr._ALWAYS]
        self.assertTrue(always_bodies, "no _ALWAYS section parsed — this test "
                                       "would pass blind")
        stable = pr.stable_pc_block(FULL)
        for body in always_bodies:
            probe = body.strip().split("\n")[0]
            self.assertIn(probe, stable)
            for u in _SPLIT_CORPUS:
                self.assertNotIn(probe, pr.turn_pc_block(u, FULL),
                                 f"{probe!r} shipped twice on {u!r}")

    def test_turn_block_is_much_smaller_than_the_slim_prompt(self):
        """It rides in the uncached tail, so its size is the thing that decides
        whether the NEXT turn hits the prefix cache."""
        for u in _SPLIT_CORPUS:
            self.assertLess(len(pr.turn_pc_block(u, FULL)),
                            len(pr.slim_pc_control(u, FULL)))

    def test_never_raises_on_garbage(self):
        for bad in ("", "no headers here at all", None):
            try:
                pr.stable_pc_block(bad if bad is not None else "")
                pr.turn_pc_block("hello", bad if bad is not None else "")
            except Exception as e:            # pragma: no cover - guard
                self.fail(f"raised on {bad!r}: {e}")


class FollowupRoundIsRatchetedElsewhereTests(unittest.TestCase):
    """Keep the ⚠ SCOPE note above TRUE.

    The ratchet in CacheStableSplitTests models the primary turn only, and that
    narrowness is exactly what let the follow-up round regress to 11/135 action
    names while the suite sat at 65/65 green. Re-deriving prompt strings in
    this file cannot fix that: `stable + turn_pc_block(u)` is the same
    expression for every round, so a content assertion here is a tautology and
    would pass just as happily with the follow-up unwired. Only driving the
    real function catches a WIRING regression.

    So this class asserts the two things that ARE checkable from here:
      1. the premise — the stable half alone is NOT a usable action reference,
         which is WHY a call site that ships only the stable half is broken;
      2. the coverage — a guard that drives the real get_followup_response
         still exists, and the runtime still carries a turn context into it.
    Both are cheap; neither imports the monolith, so this file stays runnable
    on the light-deps CI tier.
    """

    #: The companion guard. Named here so deleting it fails loudly rather than
    #: silently restoring the exact blind spot this file's SCOPE note denies.
    FOLLOWUP_GUARD = "test_followup_turn_context.py"

    def test_the_stable_half_alone_is_not_a_usable_action_reference(self):
        """The premise of the whole follow-up fix, pinned.

        If a refactor ever makes stable_pc_block self-sufficient this goes red
        — which is GOOD NEWS, not a failure: it means the follow-up's carried
        bodies stopped being load-bearing and the guard can be simplified. It
        must be a deliberate decision, not a silent drift, because the
        opposite drift is what shipped [ACTION: mute_volume] to a dispatcher
        that has no such action.
        """
        stable_names = set(_ACTION_NAME_RE.findall(pr.stable_pc_block(FULL)))
        carried = set()
        for u in _SPLIT_CORPUS:
            carried |= set(_ACTION_NAME_RE.findall(pr.turn_pc_block(u, FULL)))
        carried -= stable_names
        # Measured 2026-09-06: 10 in the stable half, 263 only in the bodies.
        self.assertLess(len(stable_names), 40,
                        "stable_pc_block now names a lot of actions on its "
                        "own — re-check whether the follow-up round still "
                        "needs turn_pc_block carried into it")
        self.assertGreaterEqual(
            len(carried), 5 * max(1, len(stable_names)),
            f"only {len(carried)} action names live exclusively in the turn "
            f"bodies vs {len(stable_names)} in the stable half; the premise "
            "of the follow-up fix (bodies carry the vocabulary) has changed")

    def test_a_guard_drives_the_real_get_followup_response(self):
        """Source-scan, no import: the companion file must exist and must
        actually CALL get_followup_response while intercepting the last hop
        before the model. A guard that only re-derived strings would have
        stayed green through the regression, so 'a file exists' is not
        enough — it has to drive the function and inspect what it hands over."""
        path = os.path.join(_PROJECT, "tests", self.FOLLOWUP_GUARD)
        self.assertTrue(
            os.path.isfile(path),
            f"{self.FOLLOWUP_GUARD} is gone. The ⚠ SCOPE note in this file "
            "says the follow-up call site is ratcheted THERE; without it "
            "nothing covers it and this file's ratchet silently over-claims "
            "again. Restore it or move the coverage and update the note.")
        with open(path, "r", encoding="utf-8") as f:
            guard = f.read()
        for probe, why in (
            ("get_followup_response(", "must call the real follow-up entry "
                                       "point, not a re-derived copy"),
            ("_local_then_cloud_or_honest", "must intercept the last hop "
                                            "before the model, so what it "
                                            "asserts on is what the model "
                                            "actually sees"),
        ):
            self.assertIn(probe, guard,
                          f"{self.FOLLOWUP_GUARD} no longer contains {probe!r} "
                          f"— it {why}")

    def test_the_followup_call_site_still_carries_a_turn_context(self):
        """The runtime half of the same claim.

        get_followup_response reuses the primary turn's system prompt, which
        has no PC_CONTROL_PROMPT in it — so _call_local_llm's cheatsheet swap
        cannot fire and the ONLY remaining source of action names is a turn
        context. If that branch ever stops passing one, the round is back to
        the index alone."""
        path = os.path.join(_PROJECT, "bobert_companion.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        start = src.index("def get_followup_response")
        end = src.index("\ndef ", start)
        # CODE ONLY. Probing the raw body was this guard's own first draft and
        # it was toothless: get_followup_response *explains* the mechanism in a
        # comment ("See _last_turn_pc_block."), so deleting the actual carry
        # left the probe green. Strip comments before matching — the whole
        # point of this file's SCOPE note is that a guard which passes for the
        # wrong reason is worse than no guard.
        body = "\n".join(line.split("#", 1)[0]
                         for line in src[start:end].split("\n"))
        # Since the local prompt budget (2026-10-01) the follow-up attaches
        # its turn context through _fit_local_messages, which must itself
        # attach with _with_turn_context.
        fit_start = src.index("def _fit_local_messages")
        fit_body = "\n".join(
            line.split("#", 1)[0] for line in
            src[fit_start:src.index("\ndef ", fit_start + 1)].split("\n"))
        self.assertIn("attach=_with_turn_context", fit_body,
                      "_fit_local_messages no longer attaches the turn "
                      "context with _with_turn_context")
        self.assertTrue(
            "_with_turn_context(" in body
            or "_fit_local_messages(" in body,
            "get_followup_response no longer sends a turn context "
            "at all — the follow-up round is back to whatever the "
            "reused system prompt happens to carry")
        self.assertIn("_last_turn_pc_block[0]", body,
                      "get_followup_response stopped carrying the primary "
                      "turn's turn_pc_block() bodies (_last_turn_pc_block). "
                      "That is the regression that took the follow-up round "
                      "from 135/135 action names to 11/135. If the slot was "
                      "renamed, update this probe; if the mechanism changed, "
                      "prove the new one in " + self.FOLLOWUP_GUARD)


class GenericHeaderWordRoutingRegressionTests(unittest.TestCase):
    """2026-09-29: select_sections' header-word fallback fired on ANY header
    word longer than three letters, with a loose prefix/suffix test. Generic
    words ("mode", "control", "status", ...) sit in many headers, so one of
    them in a turn loaded every section carrying it. Measured before the fix:

      'turn off wake word mode'  -> 9 sections, ~8.5k chars of volatile tail
                                    (GUARD / FOCUS x2 / NIGHT-OWL / AMBIENT
                                    MODE on "mode", WAKE LISTENER on "wake",
                                    SHUTDOWN ALIASES on "turn off")
      'Jarvis now has full control over the robot'
                                 -> AIR CONTROL + POINT-TO-CONTROL on "control"
      'how much space is left on my C drive'
                                 -> nothing beyond the always-on launcher, so
                                    check_system never reached the model

    Every test here fails on the pre-fix router."""

    WAKE_MODE_TURN = "turn off wake word mode"
    CONTROL_TURN = "Jarvis now has full control over the robot"
    DISK_TURN = "how much space is left on my C drive"

    def setUp(self):
        self.core, self.sections = pr.split_pc_control(FULL)
        self.bodies = dict(self.sections)
        self.always = {h.strip() for h, _b in self.sections
                       if h.strip().upper() in pr._ALWAYS}
        self.assertTrue(self.always, "no _ALWAYS section parsed -- the "
                                     "exact-set assertion below would be blind")

    def test_wake_word_mode_turn_selects_only_its_section(self):
        inc, _ = pr.select_sections(self.WAKE_MODE_TURN, self.sections)
        self.assertEqual(set(inc), {"WAKE-WORD MODE"} | self.always,
                         f"{self.WAKE_MODE_TURN!r} must load WAKE-WORD MODE "
                         f"and the always-on sections only, got {inc}")

    def test_wake_word_mode_turn_ships_the_off_action_not_its_neighbours(self):
        slim = pr.slim_pc_control(self.WAKE_MODE_TURN, FULL)
        self.assertIn("wake_word_mode_off", slim)
        for near_miss in ("wake_listener_stop", "turn_off_jarvis",
                          "guard_off", "ambient_learning_mode_off"):
            self.assertNotIn(near_miss, slim,
                             f"{self.WAKE_MODE_TURN!r} must not hand the "
                             f"model the near-miss {near_miss!r}")

    def test_wake_word_mode_turn_tail_is_small(self):
        # ~8.5k chars before the fix; WAKE-WORD MODE's own body is under 1k.
        tail = pr.turn_pc_block(self.WAKE_MODE_TURN, FULL)
        self.assertIn("wake_word_mode_off", tail)
        self.assertLess(len(tail), 2000,
                        f"volatile tail is {len(tail)} chars -- generic header "
                        f"words are pulling unrelated sections again")

    def test_full_control_sentence_does_not_load_air_control(self):
        inc, _ = pr.select_sections(self.CONTROL_TURN, self.sections)
        self.assertNotIn("AIR CONTROL", inc)
        self.assertNotIn("POINT-TO-CONTROL", inc)

    def test_c_drive_space_question_reaches_check_system(self):
        # Located by CONTENT so a rename or move cannot make this pass blind:
        # whichever section DEFINES check_system (an indented action line, not
        # a cross-reference) must load, and the action must be in the tail.
        homes = [h for h, b in self.sections
                 if re.search(r"^\s+check_system\s+\S", b, re.M)]
        self.assertTrue(homes, "no section defines check_system")
        inc, _ = pr.select_sections(self.DISK_TURN, self.sections)
        for h in homes:
            self.assertIn(h, inc, f"{self.DISK_TURN!r} must load {h!r}")
        self.assertIn("[ACTION: check_system]",
                      pr.turn_pc_block(self.DISK_TURN, FULL))

    def test_generic_word_alone_selects_nothing_by_header(self):
        # Bug-class invariant: a turn whose only content word is a generic
        # header word may load a section ONLY through that section's explicit
        # keyword list, never through its header.
        for w in sorted(pr._GENERIC_HEADER_WORDS):
            turn = f"the {w} please"
            low = " " + turn + " "
            inc, _ = pr.select_sections(turn, self.sections)
            by_header = [n for n in inc if n not in self.always and
                         not any(k in low for k in pr._keywords_for(n))]
            self.assertEqual(by_header, [],
                             f"{turn!r} loaded {by_header} on the generic "
                             f"header word {w!r} alone")

    def test_ordinary_turn_off_does_not_offer_jarvis_power_off(self):
        # Bare "turn off" routed SHUTDOWN ALIASES on every "turn off the X".
        slim = pr.slim_pc_control("turn off the lights", FULL)
        self.assertNotIn("turn_off_jarvis", slim)
        # ...while the self-directed form still reaches the aliases.
        inc, _ = pr.select_sections("turn yourself off", self.sections)
        self.assertIn("SHUTDOWN ALIASES", inc)

    def test_phrases_that_routed_only_on_generic_words_keep_their_home(self):
        # These documented triggers reached their own section -- or, for
        # 'system status', its status_panel action via STATUS READ-BACKS --
        # ONLY through a now-generic header word; each home section now has
        # an explicit keyword for it instead.
        for phrase, home in (("music mode", "WAKE-WORD MODE"),
                             ("quiet mode", "FOCUS MODE / DO-NOT-DISTURB"),
                             ("list my Hue lights",
                              "SMART HOME \u2014 PER-BRAND LIST"),
                             ("list my Tuya plugs",
                              "SMART HOME \u2014 PER-BRAND LIST"),
                             ("JARVIS, system status", "SUIT DIAGNOSTICS")):
            self.assertIn(home, self.bodies, f"fixture drift: no {home!r}")
            inc, _ = pr.select_sections(phrase, self.sections)
            self.assertIn(home, inc, f"{phrase!r} must still load {home!r}")


class UnitConversionRoutingRegressionTests(unittest.TestCase):
    """2026-09-29 live (v2.0.131): "convert 100 degrees fahrenheit to celsius"
    loaded SYSTEM HEALTH (gpu_usage / hardware-temperature grammar) on
    "degrees" + "celsius" — arithmetic handed hardware and weather actions.
    On a conversion turn the unit words no longer route; a real hardware or
    weather question that names a unit still does."""

    CONVERSIONS = ("convert 100 degrees fahrenheit to celsius",
                   "what's 30 celsius in fahrenheit",
                   "how many degrees celsius is 100 fahrenheit",
                   "convert 20 degrees c to f")

    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)

    def test_conversion_turns_load_no_health_or_weather(self):
        for text in self.CONVERSIONS:
            with self.subTest(text=text):
                inc, _ = pr.select_sections(text, self.sections)
                self.assertNotIn("SYSTEM HEALTH", inc)
                self.assertNotIn("WEATHER BRIEFING", inc)

    def test_conversion_prompt_does_not_carry_the_weather_action(self):
        slim = pr.slim_pc_control(self.CONVERSIONS[0], FULL)
        weather_body = dict(self.sections)["WEATHER BRIEFING"]
        self.assertNotIn(weather_body, slim)

    def test_unit_words_still_route_real_questions(self):
        for text, home in (("how hot is my graphics card", "SYSTEM HEALTH"),
                           ("what's the gpu temperature in celsius",
                            "SYSTEM HEALTH"),
                           ("convert the gpu temperature to fahrenheit",
                            "SYSTEM HEALTH"),
                           ("how many degrees is it outside",
                            "SYSTEM HEALTH"),
                           ("what's the weather in celsius", "WEATHER BRIEFING"),
                           ("will it rain tomorrow", "WEATHER BRIEFING")):
            with self.subTest(text=text):
                inc, _ = pr.select_sections(text, self.sections)
                self.assertIn(home, inc)


class ReplayRoutingRegressionTests(unittest.TestCase):
    """B032 (2026-10-01): "repeat that" / "come again" / "what did you say"
    ask JARVIS to SAY something again, yet they loaded the one section that
    documents replay_last_action, steering the model to re-run the owner's
    last action. Only action-replay phrasing loads it now."""

    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)
        self.homes = [h for h, b in self.sections if "replay_last_action" in b]

    def test_replay_doc_exists(self):
        self.assertTrue(self.homes, "some section must document replay_last_action")

    def test_say_it_again_requests_do_not_load_replay(self):
        for q in ("repeat that", "can you repeat that", "come again?",
                  "what did you say", "say again", "say that one more time"):
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                for h in self.homes:
                    self.assertNotIn(h, inc)

    def test_action_replay_requests_still_load_replay(self):
        for q in ("replay that", "do that again", "do it again on the left monitor"):
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                for h in self.homes:
                    self.assertIn(h, inc)


class WordBoundaryKeywordRoutingTests(unittest.TestCase):
    """2026-10-01: keywords matched as bare SUBSTRINGS, so a keyword that
    started mid-word routed a section the turn never named: "phone" inside
    "microphone" loaded PHONE NOTIFICATIONS + PHONE BRIDGE on every mic
    question (the 2026-09-04 what_microphone near-miss), "face" inside
    "interface" loaded FACE RECOGNITION on every web-interface turn, "hot"
    inside "hotword" / "screenshot" loaded SYSTEM HEALTH. A keyword now has to
    START a word; it may still run on ("print" -> "printing", "auto switch"
    -> "auto switching"), except a short (<= 3 character) one, which may only
    take a plural ("tv" -> "tvs", never "hot" -> "hotword")."""

    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)

    def _inc(self, q):
        return pr.select_sections(q, self.sections)[0]

    def test_microphone_does_not_load_the_phone_sections(self):
        for q in ("what microphone are you using", "is my microphone working",
                  "test your microphone"):
            with self.subTest(q=q):
                inc = self._inc(q)
                self.assertNotIn("PHONE NOTIFICATIONS", inc)
                self.assertNotIn("PHONE BRIDGE", inc)
                self.assertTrue(any(h.startswith("AUDIO DEVICES") for h in inc),
                                f"{q!r} lost its own section: {inc}")

    def test_a_real_phone_turn_still_loads_the_phone_sections(self):
        for q in ("send that to my phone", "notify my phone when it finishes",
                  "push it to my phones"):
            with self.subTest(q=q):
                self.assertIn("PHONE NOTIFICATIONS", self._inc(q))

    def test_mid_word_keywords_no_longer_fire(self):
        for q, wrong in (("is the web interface on", "FACE RECOGNITION"),
                         ("stop the hotword", "SYSTEM HEALTH"),
                         ("screenshot the browser", "SYSTEM HEALTH"),
                         ("any plans this weekend", "NETWORK / LAN PRESENCE"),
                         ("turn on the web dashboard", "STREAMING SERVICES"),
                         ("check Teams", "BAMBU 3D PRINTER"),
                         ("unit conversion", "CHANGELOG / VERSION"),
                         ("pick a random song", "SYSTEM HEALTH"),
                         ("put on the Michael Jackson playlist",
                          "AUDIO DEVICES — WHICH MICROPHONE AND SPEAKERS ARE IN USE")):
            with self.subTest(q=q):
                self.assertNotIn(wrong, self._inc(q))

    def test_word_start_matches_keep_their_suffixes(self):
        for q, want in (("is it printing", "BAMBU 3D PRINTER"),
                        ("is auto switching on", "AUDIO OUTPUT DEVICE"),
                        ("cancel my reminders", "TIMERS / REMINDERS"),
                        ("is the gpu throttling", "SYSTEM HEALTH"),
                        ("are the tvs on", "TV DETECTION"),
                        ("what's eating my ram", "SYSTEM HEALTH"),
                        ("what's using my ram's headroom", "SYSTEM HEALTH")):
            with self.subTest(q=q):
                self.assertIn(want, self._inc(q))

    def test_keywords_that_relied_on_a_mid_word_hit_got_their_own_entry(self):
        # The only two intended matches the substring rule made: "unmute"
        # (via "mute") and "xtts" (via "tts"). Both are arrow examples.
        self.assertIn("MUSIC CONTROLS", self._inc("unmute"))
        self.assertIn("TTS BACKEND SWITCHING", self._inc("switch to xtts"))
        self.assertIn("TTS BACKEND SWITCHING", self._inc("use pyttsx3"))

    def test_header_words_also_need_a_word_start(self):
        # The header-word fallback accepted a match at EITHER edge, so
        # "yourself" loaded every SELF-* section and "tonight" NIGHT-OWL MODE.
        for q, wrong in (("restart yourself", "SELF-PRESERVATION"),
                         ("restart yourself", "SELF-TEST PROBES"),
                         ("is it going to rain tonight", "NIGHT-OWL MODE")):
            with self.subTest(q=q, wrong=wrong):
                self.assertNotIn(wrong, self._inc(q))
        self.assertIn("NIGHT-OWL MODE", self._inc("night owl mode on"))

    def test_keyword_hit_edges(self):
        hit = pr._keyword_hit
        self.assertTrue(hit("phone", " my phone "))
        self.assertFalse(hit("phone", " microphone "))
        self.assertTrue(hit("phone", " phone, please "))
        self.assertTrue(hit("print", " printing "))
        self.assertTrue(hit("tv", " tvs "))
        self.assertFalse(hit("hot", " hotword "))
        self.assertTrue(hit("hot", " too hot! "))
        self.assertTrue(hit(" c drive", " my c drive "))
        self.assertFalse(hit(" c drive", " music drive "))
        self.assertTrue(hit("keep apple music", " keep apple music open "))
        self.assertTrue(hit("disney+", " watch disney+ now "))
        self.assertFalse(hit("", " anything "))
        # A later word-start occurrence still counts after a mid-word one.
        self.assertTrue(hit("phone", " microphone or phone "))


class RunningCostsRoutingTests(unittest.TestCase):
    """'How much does it cost to run you' asks what JARVIS COSTS to run
    (running_costs: electricity + session cloud spend), not the account
    BALANCE (check_credits, which drives a browser to the billing page). Both
    directions must reach the local model with the right action, and the
    prompt must teach which phrase is which."""

    COSTS = ("how much does it cost to run you", "what do you cost",
             "running costs", "how much do you cost per month")
    CREDITS = ("how many credits do I have", "check my Anthropic balance")

    # The always-shipped core preamble names check_credits in passing, so a
    # bare name proves nothing; the section's own example must be there.
    def test_cost_phrases_ship_running_costs(self):
        for q in self.COSTS:
            with self.subTest(q=q):
                self.assertIn("[ACTION: running_costs]",
                              pr.slim_pc_control(q, FULL))

    def test_credit_phrases_still_ship_check_credits(self):
        for q in self.CREDITS:
            with self.subTest(q=q):
                self.assertIn("[ACTION: check_credits]",
                              pr.slim_pc_control(q, FULL))

    def test_prompt_examples_map_each_phrase_to_its_own_action(self):
        examples = {phrase.lower(): action for phrase, action in
                    _ARROW_EXAMPLE_RE.findall(" ".join(FULL.split()))}
        self.assertEqual(examples.get("how much does it cost to run you"),
                         "running_costs")
        self.assertEqual(examples.get("how much do you cost per month"),
                         "running_costs")
        for q in self.CREDITS:
            self.assertEqual(examples.get(q.lower()), "check_credits", q)


class GlobeRoutingTests(unittest.TestCase):
    """skills/globe.py (2026-10-02). Both directions: the globe's own phrases
    must load the GLOBE section with the action they ask for, and the
    existing "where is ..." / location / weather turns must NOT pick it up
    (the router has no bare "where is" keyword on purpose) and must keep their
    own section."""

    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)

    def _inc(self, q):
        return pr.select_sections(q, self.sections)[0]

    def test_globe_is_a_parsed_section(self):
        self.assertIn("GLOBE", [h for h, _b in self.sections])

    def test_globe_phrases_load_the_globe_and_ship_their_action(self):
        for q, action in (("show me the globe", "show_globe"),
                          ("put the globe on the left monitor", "show_globe"),
                          ("show me where Tokyo is", "globe_pin"),
                          ("pin London and New York", "globe_pin"),
                          ("drop a pin on Paris", "globe_pin"),
                          ("clear the pins", "globe_clear"),
                          ("hide the globe", "hide_globe")):
            with self.subTest(q=q):
                self.assertIn("GLOBE", self._inc(q))
                self.assertIn(action, pr.turn_pc_block(q, FULL))

    def test_where_and_weather_turns_keep_their_home_and_skip_the_globe(self):
        for q, home in (("where's my package", "AMAZON ORDER TRACKER"),
                        ("where is my package", "AMAZON ORDER TRACKER"),
                        ("where am I", "UNIFIED"),
                        ("where is the print at", "BAMBU 3D PRINTER"),
                        ("where is the robot build at", "REPO ROBOT PROJECT"),
                        ("what's the weather in Tokyo", "WEATHER BRIEFING"),
                        ("will it rain tomorrow in London", "WEATHER BRIEFING"),
                        ("is it going to rain", "WEATHER BRIEFING")):
            with self.subTest(q=q):
                inc = self._inc(q)
                self.assertNotIn("GLOBE", inc)
                self.assertIn(home, inc)

    def test_pin_keyword_does_not_fire_inside_other_words(self):
        for q in ("keep the window pinned", "spin up the reactor",
                  "ping the router", "what's on the spinner"):
            with self.subTest(q=q):
                self.assertNotIn("GLOBE", self._inc(q))



def _names_in(text: str, action: str) -> bool:
    """True when ``action`` appears as a whole name in ``text`` — the same
    test the brain eval applies (the model can only emit what it was shown)."""
    return re.search(r"(?<![a-z0-9_])" + re.escape(action) + r"(?![a-z0-9_])",
                     text) is not None


class BrainEvalRouterMissRegressionTests(unittest.TestCase):
    """2026-10-01 brain eval: eight utterances whose correct action never
    reached the local prompt, so no model could pick it. Each case below is
    the eval's own utterance (and, for a follow-up, its history) and fails
    when the expected action's name is absent from what the router ships
    (turn_pc_block + stable_pc_block — the wire layout)."""

    def setUp(self):
        self.stable = pr.stable_pc_block(FULL)

    def _ships(self, utterance, action, history=None):
        if history is None:
            turn = pr.turn_pc_block(utterance, FULL)
        else:
            turn = pr.turn_pc_block(utterance, FULL, history=history)
        self.assertTrue(
            _names_in(turn + self.stable, action),
            f"{utterance!r} ships no {action}: the router selected "
            f"{pr.select_sections(utterance, pr.split_pc_control(FULL)[1])[0]}")

    # -- single-turn misses: vocabulary the keyword lists lacked -----------
    def test_media05_kill_the_sound_ships_volume_mute(self):
        self._ships("Jarvis, kill the sound completely.", "volume_mute")

    def test_print04_chamber_cam_ships_show_printer_camera(self):
        # The action was registered but documented in NO section, so this
        # needed a prompt line as well as routing.
        self._ships("Jarvis, let me see the chamber cam on the printer.",
                    "show_printer_camera")

    def test_sys02_video_memory_ships_gpu_usage(self):
        self._ships("Jarvis, how much video memory is free right now?",
                    "gpu_usage")

    def test_cam01_where_am_i_sitting_ships_situational_awareness(self):
        self._ships("Jarvis, can you tell where I'm sitting right now?",
                    "situational_awareness")

    def test_mem01_chatting_about_ships_session_memory_recall(self):
        self._ships("Jarvis, sum up what we've been chatting about today.",
                    "session_memory_recall")

    def test_mem03_scrub_the_past_hour_ships_forget_last_hour(self):
        self._ships("Jarvis, scrub everything from the past hour.",
                    "forget_last_hour")

    # -- follow-ups: the action lives in the PREVIOUS user turn -----------
    TIMER_HISTORY = [
        {"role": "user", "content": "Jarvis, set a timer for ten minutes."},
        {"role": "assistant",
         "content": "Ten minutes, starting now, sir. [ACTION: set_timer, 10 minutes]"},
    ]
    PRINT_HISTORY = [
        {"role": "user", "content": "Jarvis, how's the print going?"},
        {"role": "assistant", "content": "Checking, sir. [ACTION: check_print]"},
        {"role": "assistant",
         "content": "Layer 212 of 480, sir, about an hour and ten minutes left."},
    ]

    def test_fu02_cancel_that_after_a_timer_ships_cancel_timer(self):
        self._ships("Never mind, cancel that.", "cancel_timer",
                    history=self.TIMER_HISTORY)

    def test_fu08_pause_it_after_a_print_check_ships_pause_print(self):
        self._ships("Pause it.", "pause_print", history=self.PRINT_HISTORY)
        # ...without losing what the words alone route to.
        turn = pr.turn_pc_block("Pause it.", FULL, history=self.PRINT_HISTORY)
        self.assertTrue(_names_in(turn, "pause_music"))

    def test_history_may_or_may_not_already_hold_the_current_turn(self):
        # Live: _call_llm appends the turn to conversation_history BEFORE it
        # builds the prompt. The eval harness passes the prior turns only.
        u = "Pause it."
        live = self.PRINT_HISTORY + [{"role": "user", "content": u}]
        self.assertEqual(pr.turn_pc_block(u, FULL, history=live),
                         pr.turn_pc_block(u, FULL, history=self.PRINT_HISTORY))
        self.assertNotEqual(pr.turn_pc_block(u, FULL, history=live),
                            pr.turn_pc_block(u, FULL))

    def test_slim_prompt_routes_follow_ups_the_same_way(self):
        # slim_pc_control is the non-split fallback of the same call site.
        slim = pr.slim_pc_control("Never mind, cancel that.", FULL,
                                  history=self.TIMER_HISTORY)
        self.assertTrue(_names_in(slim, "cancel_timer"))

    def test_what_about_tomorrow_inherits_the_weather_turn(self):
        hist = [{"role": "user", "content": "Jarvis, what's it like outside?"},
                {"role": "assistant", "content": "One moment, sir."}]
        self._ships("And what about tomorrow?", "weather_briefing", history=hist)

    def test_a_chain_of_follow_ups_reaches_back_one_more_turn(self):
        hist = self.TIMER_HISTORY + [
            {"role": "user", "content": "Make it fifteen."},
            {"role": "assistant", "content": "Fifteen it is, sir."},
        ]
        self._ships("Actually, cancel it.", "cancel_timer", history=hist)

    def test_a_self_contained_turn_does_not_inherit_the_last_topic(self):
        # Only a SHORT, elliptical turn borrows the previous turn's routing;
        # a turn that names its own subject must not drag the printer in.
        for u in ("Jarvis, what's the weather like this afternoon?",
                  "Jarvis, open Notepad for me.",
                  "Jarvis, turn on the office lights and set them to fifty percent."):
            with self.subTest(u=u):
                self.assertEqual(
                    pr.turn_pc_block(u, FULL, history=self.PRINT_HISTORY),
                    pr.turn_pc_block(u, FULL))

    def test_no_history_routes_exactly_as_before(self):
        for u in ("Pause it.", "Never mind, cancel that.", "turn it off"):
            with self.subTest(u=u):
                self.assertEqual(pr.turn_pc_block(u, FULL, history=None),
                                 pr.turn_pc_block(u, FULL))
                self.assertEqual(pr.turn_pc_block(u, FULL, history=[]),
                                 pr.turn_pc_block(u, FULL))

    def test_elliptical_detection(self):
        ell = pr.is_elliptical_followup
        for u in ("what about tomorrow", "And what about tomorrow?",
                  "and the other one", "Pause it.", "Never mind, cancel that.",
                  "Make them a bit dimmer.", "Jarvis, turn it off.",
                  "Skip this one."):
            with self.subTest(u=u):
                self.assertTrue(ell(u))
        for u in ("", "Jarvis, open Notepad for me.", "what time is the game",
                  "set a timer for ten minutes",
                  # a demonstrative before a noun names its own subject
                  "what's the weather like this afternoon",
                  "cancel that timer", "set a timer for one minute",
                  "turn it off and then open the browser on the left monitor "
                  "and play some music"):
            with self.subTest(u=u):
                self.assertFalse(ell(u))

    def test_the_turns_never_fuse_into_a_phrase(self):
        # Current + previous are routed as separate lines: a keyword may not
        # match across the seam ("...that" + "jarvis ..." is not a phrase).
        text = pr.routing_text("cancel that", [
            {"role": "user", "content": "timer for ten minutes"}])
        self.assertIn("\n", text)
        self.assertTrue(text.startswith("cancel that"))
        self.assertFalse(pr._keyword_hit("that timer", " " + text + " "))

    def test_the_live_call_site_passes_the_history(self):
        """Source scan, no import (keeps this file on the light-deps tier):
        _call_llm must hand the conversation to both router entry points,
        or the follow-up routing above exists only in tests."""
        path = os.path.join(_PROJECT, "bobert_companion.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        start = src.index("def _call_llm(")
        end = src.index("\ndef ", start)
        body = "\n".join(line.split("#", 1)[0]
                         for line in src[start:end].split("\n"))
        self.assertRegex(body, r"turn_pc_block\([^)]*history=")
        self.assertRegex(body, r"slim_pc_control\([^)]*history=")


class BambuCameraDocumentedTests(unittest.TestCase):
    """show_printer_camera was registered (skills/holographic_overlay) but no
    PC_CONTROL section documented it, so 'show me the printer camera' could
    only ever reach the status read-back (bambu_camera_status)."""

    def test_printer_section_documents_show_and_hide(self):
        _core, sections = pr.split_pc_control(FULL)
        body = dict(sections)["BAMBU 3D PRINTER"]
        self.assertTrue(_names_in(body, "show_printer_camera"))
        self.assertTrue(_names_in(body, "hide_printer_camera"))

    def test_camera_phrasings_route_to_the_printer_section(self):
        _core, sections = pr.split_pc_control(FULL)
        for u in ("show me the printer camera", "pull up the chamber cam",
                  "close the printer camera"):
            with self.subTest(u=u):
                self.assertIn("BAMBU 3D PRINTER",
                              pr.select_sections(u, sections)[0])


class SplitTurnBlockTests(unittest.TestCase):
    """split_turn_block hands the prompt budget (core/prompt_budget) the turn's
    sections one by one so an overflowing turn can drop whole ones. It must
    give back exactly the sections turn_pc_block selected, and join back to
    the identical bytes, or a turn that fits would be sent changed."""

    # Every _SPLIT_CORPUS shape plus a many-section turn (the live overflow
    # shape: a long list of device / service names) and a runtime-rendered
    # section (SELF-KNOWLEDGE).
    _CORPUS = _SPLIT_CORPUS + [
        "kinect depth, the 3d printer, netflix and hulu, browser agent, "
        "queue a task",
        "how smart are you",
    ]

    def test_round_trips_to_the_same_bytes(self):
        for u in self._CORPUS:
            with self.subTest(u=u):
                block = pr.turn_pc_block(u, FULL)
                parts = pr.split_turn_block(block)
                self.assertEqual("\n".join(t for _h, t in parts), block)

    def test_headers_are_the_selected_sections_in_order(self):
        _core, sections = pr.split_pc_control(FULL)
        for u in self._CORPUS:
            with self.subTest(u=u):
                inc, _drop = pr.select_sections(u, sections)
                inc = set(inc)
                want = [h.strip() for h, _b in sections
                        if h.strip() in inc
                        and h.strip().upper() not in pr._ALWAYS]
                got = [h for h, _t in
                       pr.split_turn_block(pr.turn_pc_block(u, FULL))]
                self.assertEqual(got, want)

    def test_the_many_section_turn_really_has_many(self):
        parts = pr.split_turn_block(pr.turn_pc_block(self._CORPUS[-2], FULL))
        self.assertGreaterEqual(len(parts), 4)

    def test_text_without_headers_is_one_unnamed_part(self):
        self.assertEqual(pr.split_turn_block(""), [])
        self.assertEqual(pr.split_turn_block("B" * 50), [("", "B" * 50)])
        lead = "loose line\nMUSIC CONTROLS:\nbody"
        self.assertEqual(pr.split_turn_block(lead),
                         [("", "loose line"), ("MUSIC CONTROLS", "MUSIC CONTROLS:\nbody")])

    def test_never_raises(self):
        self.assertEqual(pr.split_turn_block(None), [])
        self.assertEqual(pr.split_turn_block(12345), [("", 12345)])
