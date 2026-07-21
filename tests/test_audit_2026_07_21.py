"""Regression tests for the 2026-07-21 audit (w2-models cluster: local-model
identity defects).

Each class pins one confirmed finding:

  * PersistedModelPickSurvivesRestartTests — the persisted LOCAL_LLM_MODEL
    (Settings GUI / voice set_model) was ignored on every restart because the
    hardcoded preference chain ran first in _get_local_llm_model().
  * ShippedDefaultCaptureTests — the "did the owner override the default?"
    gate compares against core.config._SHIPPED_LOCAL_LLM_MODEL, captured ONCE
    before _apply_user_settings() and un-clobberable by user_settings.json —
    so the gate can never rot into a second hardcoded copy of the tag.
  * ExactModelMatchTests / SwitchLlmPinsConcreteTagTests — _ollama_has_model
    matched by base name, defeating _act_switch_llm's "only if installed"
    guard: the tray's short 'qwen2.5:14b' pinned an uninstalled tag that
    404'd every turn with no recovery. Switches now resolve to the CONCRETE
    installed tag via _ollama_resolve_model.
  * TrayBackendTagInvariantTests — source-scan guard: tray.py may only send
    switch_llm backends the monolith can resolve.
  * VisionLockstepSourceInvariantTests — source-scan guard for the
    stale-duplicate class: every switch site outside the monolith that
    repoints the resolver cache must also run the vision lockstep
    (skills.model_picker._sync_vision_to_chat).

w2-settings cluster (card 50 — acoustic barge-in can never fire):

  * BargeInHelpHonestyTests — the BARGE_IN_ENABLED schema help must state the
    wake-detector prerequisite (the GUI/web panel promised barge-in that the
    never-autostarted detector could not deliver).
  * BargeInBootHintTests — register() must print a one-line session-log hint
    when barge-in is on but the detector will not autostart, and must stay
    silent (and thread-free) otherwise.

Monolith-touching classes are @requires_monolith (local full tier only);
the source-scan invariants run everywhere.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (                  # noqa: E402
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


class _FakeResp:
    """Minimal stand-in for a requests.Response (tests/monolith style)."""

    def __init__(self, ok=True, status_code=200, json_data=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json


def _fake_tags(*names):
    """A mock `requests` whose GET /api/tags lists exactly `names`."""
    fake_req = mock.Mock()
    fake_req.get.return_value = _FakeResp(
        ok=True, json_data={"models": [{"name": n} for n in names]})
    return fake_req


GEMMA_26B = "gemma4:26b-a4b-it-qat"
GEMMA_12B = "gemma4:12b"
QWEN_14B = "qwen2.5:14b-instruct-q5_K_M"
QWEN_32B = "qwen2.5:32b-instruct-q4_K_M"


# ===========================================================================
# Finding: persisted LOCAL_LLM_MODEL ignored on every restart
# ===========================================================================
@requires_monolith
class PersistedModelPickSurvivesRestartTests(MonolithGlobalsTestCase):
    """_get_local_llm_model() must honour the owner's persisted pick (step 2:
    LOCAL_LLM_MODEL differing from the shipped default) BEFORE the hardcoded
    preference chain — the simulated-restart regression for the audit's
    'saved model silently reverts' failure."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_cache = list(self.bc._RESOLVED_LOCAL_LLM_MODEL)
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = None
        self._saved_env = os.environ.get("JARVIS_LOCAL_LLM_MODEL")
        os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)

    def tearDown(self):
        self.bc._RESOLVED_LOCAL_LLM_MODEL[:] = self._saved_cache
        if self._saved_env is None:
            os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)
        else:
            os.environ["JARVIS_LOCAL_LLM_MODEL"] = self._saved_env

    def _resolve(self, tags, local_llm_model):
        with mock.patch.object(self.bc, "requests", _fake_tags(*tags)), \
                mock.patch.object(self.bc, "LOCAL_LLM_MODEL", local_llm_model), \
                mock.patch.object(self.bc, "_log_gpu_state"):
            return self.bc._get_local_llm_model()

    def test_persisted_pick_beats_the_preference_chain(self):
        # gemma (chain head) IS installed, but the owner saved the qwen —
        # the saved pick must win and prime the cache (survives the session).
        out = self._resolve([GEMMA_26B, QWEN_14B], QWEN_14B)
        self.assertEqual(out, QWEN_14B)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], QWEN_14B)

    def test_uninstalled_pick_falls_through_to_the_chain(self):
        out = self._resolve([GEMMA_26B, QWEN_14B], "mystery:99b")
        self.assertEqual(out, GEMMA_26B)

    def test_env_override_still_beats_the_user_pick(self):
        os.environ["JARVIS_LOCAL_LLM_MODEL"] = "my:custom"
        out = self._resolve([GEMMA_26B, QWEN_14B], QWEN_14B)
        self.assertEqual(out, "my:custom")

    def test_default_pick_keeps_chain_behavior(self):
        # No override (LOCAL_LLM_MODEL == the shipped default): behaviour is
        # byte-identical to the old chain walk — guards default installs.
        import core.config as cfg
        out = self._resolve([GEMMA_26B, QWEN_14B],
                            cfg._SHIPPED_LOCAL_LLM_MODEL)
        self.assertEqual(out, GEMMA_26B)

    def test_base_name_match_selects_the_quantised_variant(self):
        # The saved 'qwen2.5:14b' counts as installed via the same base-name
        # matching the chain uses, selecting the concrete quantised tag.
        out = self._resolve([GEMMA_26B, QWEN_14B], "qwen2.5:14b")
        self.assertEqual(out, QWEN_14B)


# ===========================================================================
# Finding (support): the shipped-default capture is override-proof
# ===========================================================================
class ShippedDefaultCaptureTests(unittest.TestCase):
    """core.config._SHIPPED_LOCAL_LLM_MODEL is captured before
    _apply_user_settings() and the apply-loop skips `_`-prefixed keys — so the
    'differs from default' gate can never be clobbered by the settings file
    (and never needs a second hardcoded copy of the tag)."""

    def test_apply_user_settings_overrides_public_not_shipped(self):
        import core.config as config
        saved_llm = config.LOCAL_LLM_MODEL
        saved_shipped = config._SHIPPED_LOCAL_LLM_MODEL
        fake = json.dumps({"LOCAL_LLM_MODEL": "custom:tag",
                           "_SHIPPED_LOCAL_LLM_MODEL": "evil:tag"})
        try:
            with mock.patch("core.config.os.path.exists", return_value=True), \
                    mock.patch("builtins.open",
                               mock.mock_open(read_data=fake)):
                config._apply_user_settings()
            # The public constant takes the override…
            self.assertEqual(config.LOCAL_LLM_MODEL, "custom:tag")
            # …the shipped-default capture does NOT.
            self.assertEqual(config._SHIPPED_LOCAL_LLM_MODEL, saved_shipped)
        finally:
            config.LOCAL_LLM_MODEL = saved_llm

    def test_shipped_default_is_a_nonempty_private_string(self):
        import core.config as config
        self.assertIsInstance(config._SHIPPED_LOCAL_LLM_MODEL, str)
        self.assertTrue(config._SHIPPED_LOCAL_LLM_MODEL.strip())


# ===========================================================================
# Finding: base-name _ollama_has_model defeated the installed guard
# ===========================================================================
@requires_monolith
class ExactModelMatchTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_sibling_tag_no_longer_counts_as_installed(self):
        # The audit's exact fingerprint: only the quantised sibling is pulled.
        with mock.patch.object(self.bc, "requests", _fake_tags(QWEN_14B)):
            self.assertFalse(self.bc._ollama_has_model("qwen2.5:14b"))

    def test_resolve_model_finds_the_concrete_sibling(self):
        with mock.patch.object(self.bc, "requests", _fake_tags(QWEN_14B)):
            self.assertEqual(self.bc._ollama_resolve_model("qwen2.5:14b"),
                             QWEN_14B)

    def test_resolve_model_exact_tag_wins(self):
        with mock.patch.object(self.bc, "requests",
                               _fake_tags(QWEN_14B, GEMMA_12B)):
            self.assertEqual(self.bc._ollama_resolve_model(GEMMA_12B),
                             GEMMA_12B)

    def test_resolve_model_latest_equivalence_both_ways(self):
        with mock.patch.object(self.bc, "requests",
                               _fake_tags("qwen2.5:latest")):
            self.assertEqual(self.bc._ollama_resolve_model("qwen2.5"),
                             "qwen2.5:latest")
        with mock.patch.object(self.bc, "requests", _fake_tags("qwen2.5")):
            self.assertEqual(self.bc._ollama_resolve_model("qwen2.5:latest"),
                             "qwen2.5")

    def test_resolve_model_never_crosses_sizes(self):
        # 'qwen2.5:14b' must NOT resolve to the 32B — different sizes of the
        # same family are different models (the whole point of the picker).
        with mock.patch.object(self.bc, "requests", _fake_tags(QWEN_32B)):
            self.assertIsNone(self.bc._ollama_resolve_model("qwen2.5:14b"))

    def test_resolve_model_unreachable_returns_none(self):
        fake_req = mock.Mock()
        fake_req.get.side_effect = OSError("conn refused")
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertIsNone(self.bc._ollama_resolve_model("qwen2.5:14b"))


@requires_monolith
class SwitchLlmPinsConcreteTagTests(MonolithGlobalsTestCase):
    """_act_switch_llm must never pin a tag Ollama does not have: the request
    is resolved to the CONCRETE installed tag (via the REAL
    _ollama_resolve_model against a mocked /api/tags) before the resolver
    cache is written; an unresolvable request pulls in the background and
    keeps the working model."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _fake_actions_bc(self, resolved=GEMMA_12B):
        fake = mock.Mock()
        fake.AI_BACKEND = "claude"
        fake.OLLAMA_MODEL = resolved
        fake._KNOWN_OLLAMA_MODELS = {"qwen2.5:14b"}
        fake._get_local_llm_model.return_value = resolved
        fake._RESOLVED_LOCAL_LLM_MODEL = [resolved]
        # A pinned separate VLM string → the vision-lockstep helper no-ops
        # (and never probes the network) on these switches.
        fake.LOCAL_VISION_MODEL = "qwen2.5vl:7b"
        # The REAL resolver, driven by the mocked bc.requests in each test.
        fake._ollama_resolve_model = self.bc._ollama_resolve_model
        return fake

    def test_switch_pins_a_tag_ollama_actually_has(self):
        import core.actions as A
        installed = [QWEN_14B]
        fake = self._fake_actions_bc()
        with mock.patch.object(self.bc, "requests", _fake_tags(*installed)), \
                mock.patch.object(A, "_bc", return_value=fake):
            out = A._act_switch_llm("qwen2.5:14b")
        pinned = fake._RESOLVED_LOCAL_LLM_MODEL[0]
        # THE invariant: never pin a tag Ollama does not have.
        self.assertIn(pinned, installed)
        self.assertEqual(pinned, QWEN_14B)
        self.assertIn(QWEN_14B, out)          # the reply names the CONCRETE tag

    def test_unresolvable_switch_pulls_and_keeps_the_working_model(self):
        import core.actions as A
        fake = self._fake_actions_bc()
        with mock.patch.object(self.bc, "requests", _fake_tags(GEMMA_12B)), \
                mock.patch.object(A, "_bc", return_value=fake):
            A._act_switch_llm("qwen2.5:14b")
        self.assertEqual(fake._RESOLVED_LOCAL_LLM_MODEL[0], GEMMA_12B)
        fake._ollama_pull_async.assert_called_once_with("qwen2.5:14b")


@requires_monolith
class TrayBackendTagInvariantTests(unittest.TestCase):
    """Source-scan guard: every switch_llm backend tray.py sends must be a
    sentinel the monolith resolves itself ('anthropic' / 'ollama') or denote a
    model in the resolver preference chain — so a future tray tag that drifts
    from the tree's constants fails CI instead of shipping another
    unresolvable pin."""

    def test_every_tray_switch_backend_is_resolvable(self):
        bc = load_monolith()
        from skills.model_picker import _same_model
        with open(os.path.join(_PROJECT, "tray.py"), encoding="utf-8") as f:
            src = f.read()
        tags = re.findall(
            r'_send_command\(\s*"switch_llm"\s*,\s*backend="([^"]+)"', src)
        self.assertTrue(
            tags, "tray.py no longer sends switch_llm — update this invariant")
        for t in tags:
            if t in ("anthropic", "ollama"):
                continue
            self.assertTrue(
                any(_same_model(t, p) for p in bc._LOCAL_LLM_PREFERENCE),
                f'tray.py sends switch_llm backend="{t}", which denotes no '
                f"model in _LOCAL_LLM_PREFERENCE — it would pin an "
                f"unresolvable tag (the 2026-07-21 404-every-turn defect)")


# ===========================================================================
# Finding: chat switch forked vision off the shared multimodal brain
# ===========================================================================
class VisionLockstepSourceInvariantTests(unittest.TestCase):
    """Stale-duplicate guard for the vision-lockstep rule (this codebase's #1
    bug class): every file under skills/ and core/ that WRITES the monolith's
    _RESOLVED_LOCAL_LLM_MODEL cache (a chat-model switch site) must also
    reference _sync_vision_to_chat — a future third switch site fails the
    suite unless it keeps LOCAL_VISION_MODEL in lockstep."""

    _WRITE_RES = (
        re.compile(r"cache\[0\]\s*="),
        re.compile(r"setattr\([^)\n]*_RESOLVED_LOCAL_LLM_MODEL"),
    )

    def test_every_external_cache_writer_syncs_vision(self):
        offenders = []
        for root in ("skills", "core"):
            for dirpath, dirnames, filenames in os.walk(
                    os.path.join(_PROJECT, root)):
                dirnames[:] = [d for d in dirnames
                               if d not in ("__pycache__", ".claude")]
                for fn in filenames:
                    if not fn.endswith(".py") or fn.startswith("test_"):
                        continue
                    path = os.path.join(dirpath, fn)
                    with open(path, encoding="utf-8", errors="replace") as f:
                        src = f.read()
                    if "_RESOLVED_LOCAL_LLM_MODEL" not in src:
                        continue
                    if not any(rx.search(src) for rx in self._WRITE_RES):
                        continue          # read-only reference
                    if "_sync_vision_to_chat" not in src:
                        offenders.append(os.path.relpath(path, _PROJECT))
        self.assertEqual(
            offenders, [],
            f"these files repoint the local-LLM resolver cache without the "
            f"vision lockstep (_sync_vision_to_chat): {offenders}")

    def test_known_switch_sites_are_covered_by_the_scan(self):
        # Self-check: the two known external switch sites must be visible to
        # the scanner (guards against the invariant rotting into a no-op).
        for rel in (os.path.join("skills", "model_picker.py"),
                    os.path.join("core", "actions.py")):
            with open(os.path.join(_PROJECT, rel), encoding="utf-8") as f:
                src = f.read()
            self.assertIn("_RESOLVED_LOCAL_LLM_MODEL", src, rel)
            self.assertTrue(any(rx.search(src) for rx in self._WRITE_RES), rel)
            self.assertIn("_sync_vision_to_chat", src, rel)


# ===========================================================================
# Finding (card 50): acoustic barge-in can never fire — the only caller of
# request_tts_interrupt lives in a detector hard-coded never to autostart
# ===========================================================================
def _load_settings_window():
    """Load tools/settings_window.py by path (mirrors the other schema tests
    so `tools` needn't be an importable package on the runner)."""
    path = os.path.join(_PROJECT, "tools", "settings_window.py")
    spec = importlib.util.spec_from_file_location(
        "jarvis_settings_window_audit50", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class BargeInHelpHonestyTests(unittest.TestCase):
    """The Settings GUI (and the web panel, which serves the SAME
    settings_window.SCHEMA) must not overpromise: barge-in only works while
    the wake-word detector is running, and the detector deliberately never
    autostarts (skills/wake_listener.py — its InputStream collides with
    record_speech() on the same WASAPI device)."""

    def test_barge_in_help_names_the_prerequisite(self):
        sw = _load_settings_window()
        help_text = sw.SCHEMA["BARGE_IN_ENABLED"]["help"].lower()
        self.assertTrue(
            "start listening" in help_text
            or "wake_listener_start" in help_text,
            msg=("BARGE_IN_ENABLED's help no longer tells the owner the "
                 "wake-word detector must be running ('start listening for "
                 "the wake word') — the toggle reads as working out of the "
                 "box while acoustic barge-in is silently dead "
                 "(2026-07-21 audit, card 50)"))

    def test_web_panel_has_no_independent_barge_in_copy(self):
        # Stale-duplicate guard: settings_window.SCHEMA is the single source
        # of truth for BOTH UIs. If web_interface ever grows its own copy of
        # the barge-in wording, fixing the help in one place stops fixing
        # both — the exact rot class this audit keeps finding.
        with open(os.path.join(_PROJECT, "tools", "web_interface.py"),
                  encoding="utf-8", errors="replace") as f:
            src = f.read().lower()
        for literal in ("interrupt him", "cut him off"):
            self.assertNotIn(
                literal, src,
                msg=(f"tools/web_interface.py contains {literal!r} — an "
                     "independent copy of the barge-in help text. Serve "
                     "settings_window.SCHEMA instead of duplicating it."))


class BargeInBootHintTests(unittest.TestCase):
    """register() must surface the barge-in/detector mismatch in the session
    log: BARGE_IN_ENABLED on + no autostart → one hint line; barge-in off →
    silence. Never a real thread either way."""

    def _register(self, barge_in: bool):
        from tests._skill_harness import load_skill_isolated
        mod, _ = load_skill_isolated("wake_listener", register=False)
        self.assertFalse(mod.WAKE_WORD_AUTOSTART)   # the deliberate default
        bc_stub = types.ModuleType("bobert_companion")
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc_stub}), \
                mock.patch("core.config.BARGE_IN_ENABLED", barge_in), \
                mock.patch.object(mod.threading, "Thread") as thread_cls, \
                contextlib.redirect_stdout(buf):
            actions: dict = {}
            mod.register(actions)
        return buf.getvalue(), thread_cls, actions

    def test_hint_printed_when_barge_in_on_but_no_autostart(self):
        out, thread_cls, actions = self._register(barge_in=True)
        self.assertIn("barge-in is enabled", out)
        self.assertIn("start listening for the wake word", out)
        # No autostart thread was spawned — the hint replaces silence, it
        # must NOT replace the deliberate no-autostart policy (the WASAPI
        # device-collision hazard documented at WAKE_WORD_AUTOSTART).
        thread_cls.assert_not_called()
        self.assertIn("wake_listener_start", actions)   # register still whole

    def test_no_hint_when_barge_in_off(self):
        out, thread_cls, _ = self._register(barge_in=False)
        self.assertNotIn("barge-in", out)
        thread_cls.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
