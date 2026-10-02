"""Tests for core/brain_glow.py — the HUD colour + brief label of the brain
that is answering (local model / Claude Sonnet / Opus / any other model).

What is pinned here (light tier — stdlib only, no monolith, no Qt):

  * the colour MAPPING: local = blue, Haiku = teal (2026-10-02: apart from
    the HUD's listening green), Sonnet = gold, Opus = violet, Fable = rose,
    any other cloud model = silver; overrides from
    BRAIN_GLOW_COLORS are validated (a bad hex or an unknown tier never
    reaches a HUD);
  * the display NAMES the HUD label shows ("Sonnet 5.5", "gemma4 12B");
  * the hud_state ``brain`` dict the main process WRITES (brain_state);
  * the HUD-side READER (hud_brain): a missing / None / garbage key or a bad
    colour returns None so a HUD falls back to its normal look, and the
    label fades out after ``label_until``;
  * the PUBLISHER: one hud_state write per brain CHANGE (never per turn when
    nothing changed), a disabled flag clears a glow it published, a writer
    that raises never escapes;
  * expected_brain(): the brain the NEXT chat turn tries first, decided the
    way _call_llm decides it (the route predicate first, then AI_BACKEND);
  * describe_for_voice(): what current_model appends ("the reactor's glowing
    gold for Sonnet 5.5"), honest when the last answer came from a different
    brain, silent when the glow is off.

stdlib unittest + unittest.mock only (no pytest).
"""
from __future__ import annotations

import ast
import colorsys
import os
import threading
import types
import unittest
from unittest import mock

from core import brain_glow as BG

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-4-5"
FABLE = "claude-fable-5-1"
GEMMA = "gemma4:12b"
QWEN = "qwen2.5:14b-instruct-q5_K_M"


def _settings(enabled=True, label_s=4.0, colors=None):
    """Patch BG.settings() so tests never depend on core.config values."""
    return mock.patch.object(BG, "settings",
                             return_value=(enabled, label_s, dict(colors or {})))


class TierMappingTests(unittest.TestCase):
    def test_local_route_is_local_whatever_the_tag(self):
        self.assertEqual(BG.tier_for(GEMMA, "local"), "local")
        self.assertEqual(BG.tier_for(QWEN, "local"), "local")
        # An ollama tag that happens to contain a cloud family word is still
        # local when it is served locally.
        self.assertEqual(BG.tier_for("opus-coder:7b", "local"), "local")

    def test_cloud_families(self):
        self.assertEqual(BG.tier_for(SONNET, "cloud"), "sonnet")
        self.assertEqual(BG.tier_for(OPUS, "cloud"), "opus")
        self.assertEqual(BG.tier_for(HAIKU, "cloud"), "haiku")
        self.assertEqual(BG.tier_for(FABLE, "cloud"), "fable")
        self.assertEqual(BG.tier_for("claude-sonnet-4-5-20250929", "cloud"),
                         "sonnet")

    def test_unknown_cloud_model_is_generic_cloud(self):
        self.assertEqual(BG.tier_for("claude-mythos-1", "cloud"), "cloud")

    def test_route_inferred_from_model_when_missing(self):
        self.assertEqual(BG.tier_for(SONNET), "sonnet")
        self.assertEqual(BG.tier_for(GEMMA), "local")
        self.assertEqual(BG.normalize_route("", SONNET), "cloud")
        self.assertEqual(BG.normalize_route(None, GEMMA), "local")

    def test_route_aliases(self):
        self.assertEqual(BG.normalize_route("ollama", GEMMA), "local")
        self.assertEqual(BG.normalize_route("claude", SONNET), "cloud")
        self.assertEqual(BG.normalize_route("anthropic", SONNET), "cloud")
        self.assertEqual(BG.normalize_route("LOCAL", SONNET), "local")


class ColorMappingTests(unittest.TestCase):
    def test_defaults_are_the_requested_colours(self):
        # The owner's ask: local = blue, Sonnet = gold, Opus = red/violet.
        self.assertEqual(BG.color_word(BG.color_for(GEMMA, "local")), "blue")
        self.assertEqual(BG.color_word(BG.color_for(SONNET, "cloud")), "gold")
        self.assertEqual(BG.color_word(BG.color_for(OPUS, "cloud")), "violet")
        # Haiku is TEAL (owner's call 2026-10-02): its old green read as the
        # HUD's "listening" green. HaikuTealTests pins the distance.
        self.assertEqual(BG.color_word(BG.color_for(HAIKU, "cloud")), "teal")
        self.assertEqual(BG.color_word(BG.color_for(FABLE, "cloud")), "rose")
        self.assertEqual(
            BG.color_word(BG.color_for("claude-mythos-1", "cloud")), "silver")

    def test_every_tier_has_a_distinct_valid_colour(self):
        cols = [BG.DEFAULT_COLORS[t] for t in BG.TIERS]
        self.assertEqual(len(set(c.upper() for c in cols)), len(BG.TIERS))
        for c in cols:
            self.assertEqual(BG.valid_color(c), c.upper())

    def test_the_glow_never_reuses_the_hud_alert_red(self):
        # The HUDs paint alerts #ff5b5b; a brain colour equal to it would
        # read as an alarm.
        self.assertNotIn("#FF5B5B", {c.upper() for c in BG.DEFAULT_COLORS.values()})

    def test_overrides_merge_and_bad_values_are_ignored(self):
        cols = BG.resolve_colors({"opus": "#ff0000", "sonnet": "gold!!",
                                  "nonsense": "#00ff00", "local": 12})
        self.assertEqual(cols["opus"], "#FF0000")
        self.assertEqual(cols["sonnet"], BG.DEFAULT_COLORS["sonnet"])
        self.assertEqual(cols["local"], BG.DEFAULT_COLORS["local"])
        self.assertNotIn("nonsense", cols)
        self.assertEqual(BG.color_for(OPUS, "cloud", cols), "#FF0000")
        self.assertEqual(BG.color_word("#FF0000"), "red")

    def test_overrides_tolerate_none_and_non_dict(self):
        self.assertEqual(BG.resolve_colors(None), BG.resolve_colors({}))
        self.assertEqual(BG.resolve_colors("opus=#ff0000"), BG.resolve_colors({}))

    def test_valid_color(self):
        self.assertEqual(BG.valid_color("#3d8bff"), "#3D8BFF")
        self.assertEqual(BG.valid_color("  #3D8BFF "), "#3D8BFF")
        for bad in (None, "", "3D8BFF", "#3D8BF", "#3D8BFFAA", "#GGGGGG",
                    "blue", 0x3D8BFF, ["#3D8BFF"]):
            with self.subTest(bad=bad):
                self.assertIsNone(BG.valid_color(bad))

    def test_color_word_for_arbitrary_hues(self):
        self.assertEqual(BG.color_word("#00FF00"), "green")
        self.assertEqual(BG.color_word("#0000FF"), "blue")
        self.assertEqual(BG.color_word("#FF8800"), "orange")
        self.assertEqual(BG.color_word("#FFFFFF"), "white")
        self.assertEqual(BG.color_word("#808080"), "silver")
        self.assertEqual(BG.color_word("not a colour"), "")


def _hue_deg(color: str) -> float:
    r, g, b = (int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360.0


def _hue_gap(a: str, b: str) -> float:
    d = abs(_hue_deg(a) - _hue_deg(b)) % 360.0
    return min(d, 360.0 - d)


def _unified_hud_listening_color() -> str:
    """The unified HUD's "listening" accent as "#RRGGBB", read from its SOURCE
    (the light tier has no PyQt6): the name _accent() maps "listening" to,
    resolved to its ``NAME = QColor(r, g, b)`` assignment."""
    path = os.path.join(_PROJECT_DIR, "hud", "jarvis_unified_hud.py")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    accent = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_accent")
    table = next(n for n in ast.walk(accent) if isinstance(n, ast.Dict))
    name = next(v.id for k, v in zip(table.keys, table.values)
                if isinstance(k, ast.Constant) and k.value == "listening"
                and isinstance(v, ast.Name))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            continue
        call = node.value.body if isinstance(node.value, ast.IfExp) else node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "QColor" and len(call.args) == 3):
            rgb = [a.value for a in call.args]
            return "#" + "".join(f"{int(c):02X}" for c in rgb)
    raise AssertionError(f"no {name} = QColor(r, g, b) in {path}")


class HaikuTealTests(unittest.TestCase):
    """Owner's call (2026-10-02): Haiku's green (#36D399) sat ~13 degrees of
    hue from the unified HUD's listening green, so a Haiku ring read as
    "listening". Haiku is TEAL now — clearly apart from that green AND from
    the local model's blue."""

    #: Smallest hue gap (degrees) that reads as a different colour on the ring.
    MIN_GAP = 25.0

    def test_haiku_is_teal(self):
        haiku = BG.DEFAULT_COLORS["haiku"]
        self.assertEqual(BG.color_word(haiku), "teal")
        self.assertEqual(BG.brain_state(HAIKU, "cloud")["color"], haiku.upper())

    def test_teal_is_named_teal_by_hue_too(self):
        # The spoken word must not hang on the _DEFAULT_WORDS table alone: a
        # copy of the colour in BRAIN_GLOW_COLORS is named by its hue.
        haiku = BG.DEFAULT_COLORS["haiku"]
        with mock.patch.dict(BG._DEFAULT_WORDS, clear=True):
            self.assertEqual(BG.color_word(haiku), "teal")

    def test_haiku_is_apart_from_the_listening_green(self):
        listening = _unified_hud_listening_color()
        self.assertEqual(BG.color_word(listening), "green")
        haiku = BG.DEFAULT_COLORS["haiku"]
        self.assertGreaterEqual(_hue_gap(haiku, listening), self.MIN_GAP,
                                (haiku, listening))

    def test_haiku_is_apart_from_the_local_blue(self):
        haiku, local = BG.DEFAULT_COLORS["haiku"], BG.DEFAULT_COLORS["local"]
        self.assertGreaterEqual(_hue_gap(haiku, local), self.MIN_GAP,
                                (haiku, local))

    def test_the_voice_line_says_teal_for_haiku(self):
        BG.PUBLISHER.reset()
        self.addCleanup(BG.PUBLISHER.reset)
        bc = _bc(takes_local=False, backend="claude", claude=HAIKU)
        with _settings():
            self.assertEqual(BG.describe_for_voice(bc, route="cloud"),
                             " The reactor's glowing teal for Haiku 4.5.")


class DisplayNameTests(unittest.TestCase):
    def test_cloud_names(self):
        self.assertEqual(BG.display_name(SONNET, "cloud"), "Sonnet 5.5")
        self.assertEqual(BG.display_name(OPUS, "cloud"), "Opus 5.5")
        self.assertEqual(BG.display_name(HAIKU, "cloud"), "Haiku 4.5")
        self.assertEqual(BG.display_name("claude-sonnet-5", "cloud"), "Sonnet 5")
        # A dated snapshot never turns its date into a version number.
        self.assertEqual(BG.display_name("claude-sonnet-4-5-20250929", "cloud"),
                         "Sonnet 4.5")

    def test_local_names(self):
        self.assertEqual(BG.display_name(GEMMA, "local"), "gemma4 12B")
        self.assertEqual(BG.display_name(QWEN, "local"), "qwen2.5 14B")
        self.assertEqual(BG.display_name("gemma4:26b-a4b-it-qat", "local"),
                         "gemma4 26B")
        self.assertEqual(BG.display_name("laguna-xs-2.1", "local"), "laguna-xs-2.1")
        self.assertEqual(BG.display_name("hf.co/org/Some-Model-GGUF:Q4_K_M",
                                         "local"), "Some-Model-GGUF")

    def test_names_are_bounded_and_never_empty(self):
        self.assertLessEqual(len(BG.display_name("x" * 200, "local")), 24)
        self.assertEqual(BG.display_name("", "local"), "local model")
        self.assertEqual(BG.display_name("", "cloud"), "Claude")


class BrainStateTests(unittest.TestCase):
    def test_cloud_state_shape(self):
        st = BG.brain_state(SONNET, "cloud", source="turn", label_s=4.0, now=100.0)
        self.assertEqual(st["name"], "Sonnet 5.5")
        self.assertEqual(st["tier"], "sonnet")
        self.assertEqual(st["color"], BG.DEFAULT_COLORS["sonnet"].upper())
        self.assertEqual(st["model"], SONNET)
        self.assertEqual(st["route"], "cloud")
        self.assertEqual(st["source"], "turn")
        self.assertEqual(st["changed_at"], 100.0)
        self.assertEqual(st["label_until"], 104.0)
        self.assertEqual(st["label"], "SONNET 5.5")

    def test_local_state_label_says_local(self):
        st = BG.brain_state(GEMMA, "local", now=0.0)
        self.assertEqual(st["tier"], "local")
        self.assertEqual(st["label"], "LOCAL · GEMMA4 12B")

    def test_label_seconds_zero_means_no_label(self):
        st = BG.brain_state(SONNET, "cloud", label_s=0, now=50.0)
        self.assertEqual(st["label_until"], 50.0)
        self.assertEqual(BG.hud_brain({"brain": st}, 50.0).label_alpha, 0.0)

    def test_state_is_json_serialisable(self):
        import json
        json.dumps(BG.brain_state(OPUS, "cloud", colors={"opus": "#123456"}))


class HudReaderTests(unittest.TestCase):
    """The HUD side: never raise, never draw a bad colour."""

    def test_missing_or_garbage_key_returns_none(self):
        for hud in (None, {}, {"brain": None}, {"brain": "sonnet"},
                    {"brain": 5}, {"brain": []}, {"brain": {}},
                    {"brain": {"color": "gold"}},
                    {"brain": {"color": None, "name": "Sonnet 5.5"}},
                    "not a dict", 42):
            with self.subTest(hud=hud):
                self.assertIsNone(BG.hud_brain(hud, 10.0))

    def test_valid_state_round_trips(self):
        st = BG.brain_state(OPUS, "cloud", label_s=4.0, now=100.0)
        hb = BG.hud_brain({"brain": st}, 101.0)
        self.assertEqual(hb.color, st["color"])
        self.assertEqual(hb.name, "Opus 5.5")
        self.assertEqual(hb.label, "OPUS 5.5")
        self.assertEqual(hb.tier, "opus")
        self.assertEqual(hb.label_alpha, 1.0)

    def test_label_fades_then_disappears(self):
        st = BG.brain_state(SONNET, "cloud", label_s=4.0, now=100.0)
        hud = {"brain": st}
        self.assertEqual(BG.hud_brain(hud, 102.9).label_alpha, 1.0)
        mid = BG.hud_brain(hud, 103.5).label_alpha
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)
        self.assertEqual(BG.hud_brain(hud, 104.0).label_alpha, 0.0)
        gone = BG.hud_brain(hud, 500.0)
        self.assertEqual(gone.label_alpha, 0.0)
        # The colour (the glow) stays after the label is gone.
        self.assertEqual(gone.color, st["color"])

    def test_garbage_fields_degrade_not_raise(self):
        hb = BG.hud_brain({"brain": {"color": "#3d8bff", "name": 7,
                                     "label_until": "soon", "label": None,
                                     "tier": ["x"]}}, 1.0)
        self.assertEqual(hb.color, "#3D8BFF")
        self.assertEqual(hb.label_alpha, 0.0)
        self.assertIsInstance(hb.name, str)
        self.assertIsInstance(hb.label, str)
        self.assertIsInstance(hb.tier, str)

    def test_now_defaults_to_wall_clock(self):
        st = BG.brain_state(SONNET, "cloud", label_s=60.0)
        self.assertEqual(BG.hud_brain({"brain": st}).label_alpha, 1.0)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.pub = BG.BrainGlowPublisher()
        self.writes = []
        self.writer = lambda **kw: self.writes.append(kw)

    def test_first_publish_writes_the_brain(self):
        self.assertTrue(self.pub.publish(self.writer, SONNET, "cloud",
                                         source="boot", now=10.0))
        self.assertEqual(len(self.writes), 1)
        brain = self.writes[0]["brain"]
        self.assertEqual(brain["tier"], "sonnet")
        self.assertEqual(brain["source"], "boot")
        self.assertEqual(self.pub.last(), ("cloud", SONNET))

    def test_same_brain_again_writes_nothing(self):
        # Cheap per turn: a turn on the same brain must NOT rewrite
        # hud_state.json (the label would also re-flash every turn).
        self.pub.publish(self.writer, SONNET, "cloud", now=1.0)
        for _ in range(5):
            self.assertFalse(self.pub.publish(self.writer, SONNET, "cloud", now=2.0))
        self.assertEqual(len(self.writes), 1)

    def test_change_writes_again(self):
        self.pub.publish(self.writer, SONNET, "cloud", now=1.0)
        self.assertTrue(self.pub.publish(self.writer, GEMMA, "local", now=2.0))
        self.assertTrue(self.pub.publish(self.writer, SONNET, "cloud", now=3.0))
        self.assertEqual([w["brain"]["tier"] for w in self.writes],
                         ["sonnet", "local", "sonnet"])

    def test_disabled_never_writes_and_clears_once(self):
        self.assertFalse(self.pub.publish(self.writer, SONNET, "cloud",
                                          enabled=False))
        self.assertEqual(self.writes, [])
        self.pub.publish(self.writer, SONNET, "cloud")
        self.assertTrue(self.pub.publish(self.writer, OPUS, "cloud",
                                         enabled=False))
        self.assertEqual(self.writes[-1], {"brain": None})
        self.assertFalse(self.pub.publish(self.writer, OPUS, "cloud",
                                          enabled=False))
        self.assertEqual(len(self.writes), 2)
        self.assertIsNone(self.pub.last())
        # Re-enabled: publishes afresh even for the brain seen before.
        self.assertTrue(self.pub.publish(self.writer, SONNET, "cloud"))

    def test_empty_model_or_no_writer_is_a_noop(self):
        self.assertFalse(self.pub.publish(self.writer, "", "cloud"))
        self.assertFalse(self.pub.publish(self.writer, None, "local"))
        self.assertFalse(self.pub.publish(None, SONNET, "cloud"))
        self.assertEqual(self.writes, [])

    def test_raising_writer_never_escapes(self):
        def boom(**_kw):
            raise OSError("disk gone")
        self.assertFalse(self.pub.publish(boom, SONNET, "cloud"))
        # A failed write is retried on the next publish, not remembered.
        self.assertTrue(self.pub.publish(self.writer, SONNET, "cloud"))

    def test_concurrent_publishers_leave_the_last_key_on_disk(self):
        # The writer runs under the publisher lock, so the file and last()
        # can never disagree (which would strand a stale colour forever).
        on_disk = []
        writer = lambda **kw: on_disk.append(kw["brain"])
        models = [(SONNET, "cloud"), (GEMMA, "local"), (OPUS, "cloud")] * 30
        threads = [threading.Thread(target=self.pub.publish,
                                    args=(writer, m, r)) for m, r in models]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        last_written = on_disk[-1]
        self.assertEqual((last_written["route"], last_written["model"]),
                         self.pub.last())

    def test_reset(self):
        self.pub.publish(self.writer, SONNET, "cloud")
        self.pub.reset()
        self.assertIsNone(self.pub.last())
        self.assertTrue(self.pub.publish(self.writer, SONNET, "cloud"))

    def test_module_publish_reads_the_settings(self):
        BG.PUBLISHER.reset()
        self.addCleanup(BG.PUBLISHER.reset)
        with _settings(enabled=True, label_s=9.0, colors={"sonnet": "#010203"}):
            self.assertTrue(BG.publish(self.writer, SONNET, "cloud", now=0.0))
        self.assertEqual(self.writes[-1]["brain"]["color"], "#010203")
        self.assertEqual(self.writes[-1]["brain"]["label_until"], 9.0)
        with _settings(enabled=False):
            self.assertTrue(BG.publish(self.writer, SONNET, "cloud"))
        self.assertEqual(self.writes[-1], {"brain": None})

    def test_settings_reads_core_config_live(self):
        import core.config as cfg
        with mock.patch.object(cfg, "BRAIN_GLOW_ENABLED", False, create=True), \
             mock.patch.object(cfg, "BRAIN_GLOW_LABEL_S", 2.5, create=True), \
             mock.patch.object(cfg, "BRAIN_GLOW_COLORS", {"opus": "#ABCDEF"},
                               create=True):
            enabled, label_s, colors = BG.settings()
        self.assertFalse(enabled)
        self.assertEqual(label_s, 2.5)
        self.assertEqual(colors, {"opus": "#ABCDEF"})

    def test_settings_survive_garbage_config(self):
        import core.config as cfg
        with mock.patch.object(cfg, "BRAIN_GLOW_LABEL_S", "four", create=True), \
             mock.patch.object(cfg, "BRAIN_GLOW_COLORS", "nope", create=True):
            enabled, label_s, colors = BG.settings()
        self.assertEqual(label_s, BG.DEFAULT_LABEL_S)
        self.assertEqual(colors, {})

    def test_config_defaults(self):
        import core.config as cfg
        self.assertIs(cfg.BRAIN_GLOW_ENABLED, True)
        self.assertEqual(cfg.BRAIN_GLOW_LABEL_S, 4.0)
        self.assertEqual(cfg.BRAIN_GLOW_COLORS, {})


def _bc(*, takes_local=None, routing=None, backend="claude",
        resolved=None, local_model=GEMMA, claude=SONNET, writer=None):
    """A fake monolith with only the attributes expected_brain reads."""
    bc = types.SimpleNamespace()
    if takes_local is not None:
        bc._chat_takes_local_branch = lambda: takes_local
    if routing is not None:
        bc.MODEL_ROUTING = dict(routing)
    bc.AI_BACKEND = backend
    bc._RESOLVED_LOCAL_LLM_MODEL = [resolved]
    bc.LOCAL_LLM_MODEL = local_model
    bc.CLAUDE_MODEL = claude
    if writer is not None:
        bc._write_hud_state = writer
    return bc


class ExpectedBrainTests(unittest.TestCase):
    def setUp(self):
        import os
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)

    def test_env_override_wins_while_the_resolver_cache_is_cold(self):
        """The resolver puts JARVIS_LOCAL_LLM_MODEL first; at boot its cache
        is still cold, so the boot glow must not name the configured tag
        instead (2026-10-02 review). A warm cache still wins."""
        import os
        os.environ["JARVIS_LOCAL_LLM_MODEL"] = "llama3.2:3b"
        bc = _bc(takes_local=True, local_model=GEMMA)
        self.assertEqual(BG.expected_brain(bc), ("local", "llama3.2:3b"))
        bc = _bc(takes_local=True, resolved=QWEN, local_model=GEMMA)
        self.assertEqual(BG.expected_brain(bc), ("local", QWEN))

    def test_local_route_predicate_wins(self):
        self.assertEqual(BG.expected_brain(_bc(takes_local=True, backend="claude")),
                         ("local", GEMMA))

    def test_resolved_cache_beats_the_configured_tag(self):
        bc = _bc(takes_local=True, resolved=QWEN, local_model=GEMMA)
        self.assertEqual(BG.expected_brain(bc), ("local", QWEN))

    def test_cloud_when_not_local_and_backend_claude(self):
        self.assertEqual(BG.expected_brain(_bc(takes_local=False, backend="claude",
                                               claude=OPUS)), ("cloud", OPUS))

    def test_ollama_backend_is_local(self):
        self.assertEqual(BG.expected_brain(_bc(takes_local=False, backend="ollama")),
                         ("local", GEMMA))

    def test_unknown_backend_is_none(self):
        # _call_llm answers "AI backend not configured" — no brain to show.
        self.assertIsNone(BG.expected_brain(_bc(takes_local=False, backend="gpt")))

    def test_missing_backend_attribute_falls_back_to_core_config(self):
        import core.config as cfg
        bc = _bc(takes_local=False)
        del bc.AI_BACKEND
        with mock.patch.object(cfg, "AI_BACKEND", "claude"):
            self.assertEqual(BG.expected_brain(bc), ("cloud", SONNET))

    def test_falls_back_to_the_monolith_routing_dict(self):
        # No route predicate on the fake: MODEL_ROUTING['chat'] decides.
        self.assertEqual(
            BG.expected_brain(_bc(routing={"chat": "local"}, backend="claude")),
            ("local", GEMMA))
        self.assertEqual(
            BG.expected_brain(_bc(routing={"chat": "auto"}, backend="claude")),
            ("cloud", SONNET))

    def test_mock_attributes_never_leak_into_the_state(self):
        # core/actions tests hand switch_llm a mock.Mock() monolith: every
        # attribute exists and is a Mock. Nothing non-str may be published.
        bc = mock.Mock()
        bc.AI_BACKEND = "claude"
        bc._RESOLVED_LOCAL_LLM_MODEL = ["qwen2.5:14b"]
        got = BG.expected_brain(bc)
        self.assertIsNotNone(got)
        self.assertIsInstance(got[0], str)
        self.assertIsInstance(got[1], str)

    def test_none_bc_uses_core_config(self):
        import core.config as cfg
        with mock.patch.object(cfg, "MODEL_ROUTING", {"chat": "local"}), \
             mock.patch.object(cfg, "LOCAL_LLM_MODEL", "phi4:14b"):
            self.assertEqual(BG.expected_brain(None), ("local", "phi4:14b"))

    def test_publish_expected_writes_through_the_monolith_writer(self):
        BG.PUBLISHER.reset()
        self.addCleanup(BG.PUBLISHER.reset)
        writes = []
        bc = _bc(takes_local=False, backend="claude", claude=OPUS,
                 writer=lambda **kw: writes.append(kw))
        with _settings():
            self.assertTrue(BG.publish_expected(bc, source="switch"))
            self.assertFalse(BG.publish_expected(bc, source="switch"))
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["brain"]["tier"], "opus")
        self.assertEqual(writes[0]["brain"]["source"], "switch")

    def test_publish_expected_without_writer_is_noop(self):
        self.assertFalse(BG.publish_expected(_bc(takes_local=True)))
        self.assertFalse(BG.publish_expected(None))


class SwitchLlmPublishesTests(unittest.TestCase):
    """core/actions._act_switch_llm (voice "switch to claude/ollama/<tag>"
    and the tray's AI submenu) recolours the HUD the moment it switches."""

    def setUp(self):
        import core.config as cfg
        import core.actions as A
        self.A = A
        routing = cfg.MODEL_ROUTING
        saved = dict(routing)
        self.addCleanup(lambda: (routing.clear(), routing.update(saved)))
        BG.PUBLISHER.reset()
        self.addCleanup(BG.PUBLISHER.reset)
        p = mock.patch.object(BG, "settings", return_value=(True, 4.0, {}))
        p.start()
        self.addCleanup(p.stop)
        self.writes = []
        installed = {"gemma4:12b", "qwen2.5:14b"}
        self.bc = types.SimpleNamespace(
            AI_BACKEND="ollama", OLLAMA_MODEL="qwen2.5:14b",
            _KNOWN_OLLAMA_MODELS={"qwen2.5:14b"},
            _RESOLVED_LOCAL_LLM_MODEL=["qwen2.5:14b"],
            LOCAL_LLM_MODEL="qwen2.5:14b", CLAUDE_MODEL=OPUS,
            MODEL_ROUTING={"chat": "local"},
            _get_local_llm_model=lambda: self.bc._RESOLVED_LOCAL_LLM_MODEL[0],
            _ollama_resolve_model=lambda t: t if t in installed else None,
            _ollama_pull_async=lambda t: None,
            _write_hud_state=lambda **kw: self.writes.append(kw),
        )

    def _brains(self):
        return [w["brain"] for w in self.writes if "brain" in w]

    def test_switch_to_claude_glows_the_cloud_model(self):
        with mock.patch.object(self.A, "_bc", return_value=self.bc):
            self.A._act_switch_llm("claude")
        self.assertEqual(self._brains()[-1]["tier"], "opus")

    def test_switch_to_a_local_tag_glows_local(self):
        self.bc.AI_BACKEND = "claude"
        self.bc.MODEL_ROUTING = {"chat": "cloud"}
        with mock.patch.object(self.A, "_bc", return_value=self.bc):
            self.A._act_switch_llm("gemma4:12b")
        last = self._brains()[-1]
        self.assertEqual((last["route"], last["model"]), ("local", "gemma4:12b"))

    def test_status_query_does_not_publish(self):
        with mock.patch.object(self.A, "_bc", return_value=self.bc):
            self.A._act_switch_llm("")
        self.assertEqual(self._brains(), [])


class VoiceRoutingTests(unittest.TestCase):
    """'what brain are you on' reaches current_model on the LOCAL route too:
    the slim prompt must carry the LOCAL MODEL SELECTION section (and its
    example) for the phrasings the owner uses."""

    PHRASES = ("what brain are you on", "which brain are you using right now",
               "what colour is your brain glowing", "what brain is this")

    def test_phrases_load_the_model_section(self):
        from core import prompts, prompt_router as pr
        _core, sections = pr.split_pc_control(prompts.PC_CONTROL_PROMPT)
        for phrase in self.PHRASES:
            with self.subTest(phrase=phrase):
                inc, _drop = pr.select_sections(phrase, sections)
                self.assertIn("LOCAL MODEL SELECTION", inc)

    def test_prompt_has_the_brain_example(self):
        from core import prompts
        self.assertIn("'what brain are you on?'", prompts.PC_CONTROL_PROMPT)
        i = prompts.PC_CONTROL_PROMPT.index("'what brain are you on?'")
        line = prompts.PC_CONTROL_PROMPT[i:].split("\n", 1)[0]
        self.assertIn("[ACTION: current_model]", line)

    def test_slim_prompt_keeps_current_model_for_the_phrase(self):
        from core import prompts, prompt_router as pr
        slim = pr.slim_pc_control("what brain are you on", prompts.PC_CONTROL_PROMPT)
        self.assertIn("'what brain are you on?'", slim)


class VoiceDescriptionTests(unittest.TestCase):
    def setUp(self):
        BG.PUBLISHER.reset()
        self.addCleanup(BG.PUBLISHER.reset)

    def test_names_the_colour_of_the_expected_brain(self):
        with _settings():
            out = BG.describe_for_voice(_bc(takes_local=False, backend="claude"),
                                        route="cloud")
        self.assertIn("gold", out)
        self.assertIn("Sonnet 5.5", out)

    def test_local_brain_is_blue(self):
        with _settings():
            out = BG.describe_for_voice(_bc(takes_local=True), route="local")
        self.assertIn("blue", out)
        self.assertIn("gemma4 12B", out)

    def test_auto_mentions_the_fallback_colour(self):
        with _settings():
            out = BG.describe_for_voice(_bc(takes_local=False, backend="claude"),
                                        route="auto")
        self.assertIn("gold", out)
        self.assertIn("blue", out)

    def test_last_answer_from_another_brain_is_reported_honestly(self):
        bc = _bc(takes_local=False, backend="claude", writer=lambda **kw: None)
        with _settings():
            BG.publish(bc._write_hud_state, GEMMA, "local")   # a fallback turn
            out = BG.describe_for_voice(bc, route="cloud")
        self.assertIn("blue", out)
        self.assertIn("gemma4 12B", out)
        self.assertIn("answered last", out)

    def test_silent_when_the_glow_is_off(self):
        with _settings(enabled=False):
            self.assertEqual(
                BG.describe_for_voice(_bc(takes_local=True), route="local"), "")

    def test_override_colour_is_spoken_by_its_hue(self):
        with _settings(colors={"sonnet": "#FF0000"}):
            out = BG.describe_for_voice(_bc(takes_local=False, backend="claude"),
                                        route="cloud")
        self.assertIn("red", out)

    def test_never_raises(self):
        with mock.patch.object(BG, "settings", side_effect=RuntimeError("x")):
            self.assertEqual(BG.describe_for_voice(_bc(takes_local=True)), "")


if __name__ == "__main__":
    unittest.main()
