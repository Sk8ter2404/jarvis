"""Tests for skills/game_mode.py — the game low-power mode.

WHAT THESE PIN, AND WHY EACH ONE EXISTS

This feature's failure modes are not hypothetical; every test below is anchored
to something MEASURED on the owner's machine on 2026-09-04 while he was playing:

  * Four Fortnite processes run at once (Bootstrapper, Launcher,
    ...Shipping_EAC_EOS, ...Shipping). A substring match fires on the EAC
    wrapper. -> test_false_positive_*
  * He alt-tabs mid-match constantly. A focus-driven exit would flap the brain
    between a 9 GB and a 15 GB model, which is the crash. -> test_alt_tab_*
  * core/vram_budget.py says gemma4:latest is 4 GB; live `ollama list` says
    9.6 GB. "Escalating" to it loads MORE. -> test_refuses_to_shrink_upward
  * `keep_alive: "20m"` is hardcoded at bobert_companion.py:11148, so a refresh
    that fires against an EVICTED model IS the cold load it was meant to avoid.
    -> test_keep_warm_never_loads
  * This repo's #1 defect is claiming a result that was never established, so a
    mode that reports freeing memory it did not free is the purest form of it.
    -> test_below_floor_is_reported_as_failure
  * There is NO cloud on this box (AI_BACKEND=ollama, no ANTHROPIC_API_KEY), so
    a half-applied transition that left the brain unset would MUTE him.
    -> test_crash_at_every_step_never_mutes
  * pause_face_tracking() is a bare set() on a SHARED un-ref-counted Event, and
    the main voice loop clears it on every listen iteration
    (bobert_companion.py:23917, :23893), so a one-shot pause from the mode's
    largest CPU lever died inside ~20 s while the mode kept reporting it
    stopped. -> TestFaceDetectionPauseIsActuallyHeld

Nothing here touches a real process, a real GPU, a real model, or the disk.
stdlib unittest + unittest.mock only (pytest is not installed).
"""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

BIG = "gemma4:26b-a4b-it-qat"
SMALL = "gemma4:12b"
GAME = "fortniteclient-win64-shipping.exe"
EAC_WRAPPER = "fortniteclient-win64-shipping_eac_eos.exe"

# Live `ollama list`, 2026-09-04. Note gemma4:latest is LARGER than gemma4:12b —
# the stale-table trap the skill must refuse.
LIVE_SIZES_MB = {
    BIG: 15360,
    SMALL: 7782,
    "gemma4:latest": 9830,
    "qwen2.5vl:7b": 6144,
}


def _fake_bc():
    """A stand-in monolith exposing exactly the cells the skill repoints."""
    bc = types.ModuleType("bobert_companion")
    bc._RESOLVED_LOCAL_LLM_MODEL = [BIG]
    bc.LOCAL_LLM_MODEL = BIG
    bc.LOCAL_VISION_MODEL = BIG
    bc.HUD_CAMERA_PREVIEW = True
    bc.proactive_announce = mock.MagicMock(return_value=True)
    # Face tracking is modelled FAITHFULLY rather than as four independent
    # MagicMocks, because the shape IS the defect: pause/resume are a bare
    # set()/clear() on ONE shared, un-ref-counted threading.Event
    # (bobert_companion.py:7924-7936), and every reader — the detection loop at
    # :5835 included — resolves that Event as a MODULE GLOBAL at call time.
    # Independent mocks record that pause() was called and can never show that
    # the pause did not survive the next listen cycle.
    bc._face_track_pause = threading.Event()
    bc._face_track_camera_off = threading.Event()
    bc.pause_face_tracking = lambda: bc._face_track_pause.set()
    bc.resume_face_tracking = lambda: bc._face_track_pause.clear()
    bc.set_face_tracking_camera_off = lambda: bc._face_track_camera_off.set()
    bc.clear_face_tracking_camera_off = lambda: bc._face_track_camera_off.clear()
    return bc


def _voice_loop_listen_cycles(bc, n=1):
    """Exactly what bobert_companion's main loop does on EVERY iteration:
    resume_face_tracking() immediately before record_speech(timeout=20)
    (bobert_companion.py:23917, and :23893 on the realtime path). n cycles is
    therefore up to n*20 seconds of a match."""
    for _ in range(n):
        bc.resume_face_tracking()


class _Base(unittest.TestCase):
    """Loads the skill with a fake monolith pinned in sys.modules and every
    outward-facing primitive (GPU, Ollama, psutil, config) stubbed.

    The monolith patch must OUTLIVE the loader: the skill resolves it lazily
    inside each call via _bc(), so a `with` block that exited here would leave
    the actions reading the REAL bobert_companion — which is importable on this
    box and would answer with the machine's live state. That is exactly how a
    first draft of a sibling test suite "passed" against live hardware."""

    SETTINGS = {
        "GAME_MODE_ENABLED": False,          # never autostart the watcher here
        "GAME_MODE_PROCESS_HINTS": [GAME],
        "GAME_MODE_BRAIN": SMALL,
        "GAME_MODE_POLL_SECONDS": 5.0,
        "GAME_MODE_ENTER_DWELL_SECONDS": 20.0,
        "GAME_MODE_REQUIRE_FOREGROUND": True,
        "GAME_MODE_EXIT_GRACE_SECONDS": 45.0,
        "GAME_MODE_MAX_SECONDS": 43200.0,
        "GAME_MODE_VERIFY_DELAY_SECONDS": 0.0,
        "GAME_MODE_MIN_VRAM_DELTA_MB": 3000,
        "GAME_MODE_KEEP_WARM_SECONDS": 600.0,
        "GAME_MODE_ANNOUNCE": False,
    }

    def _load(self, *, vram=(20000, 4000), after_vram=(9000, 15000),
              sizes=None, settings=None):
        self.bc = _fake_bc()
        imp = mock.patch.dict(sys.modules, {"bobert_companion": self.bc})
        imp.start()
        self.addCleanup(imp.stop)

        mod, actions = load_skill_isolated("game_mode")
        self.mod, self.actions = mod, actions

        cfg = dict(self.SETTINGS)
        cfg.update(settings or {})
        # _cfg reads the monolith first, so pinning the values onto the fake bc
        # exercises the real lookup path rather than stubbing _cfg away.
        for k, v in cfg.items():
            setattr(self.bc, k, v)

        # --- outward-facing primitives, all stubbed ---
        self.samples = [vram, after_vram]

        def _smi():
            return self.samples[0] if len(self.samples) == 1 else self.samples.pop(0)

        mod._nvidia_smi_mb = _smi
        mod._free_ram_mb = lambda: 9000
        mod._installed_sizes_mb = lambda: dict(
            LIVE_SIZES_MB if sizes is None else sizes)
        self.resident = [{"name": BIG, "size_vram_mb": 14524}]
        mod._resident_models = lambda: list(self.resident)
        self.unloaded = []
        mod._unload = lambda tag: (self.unloaded.append(tag), True)[1]
        self.warmed = []
        mod._is_multimodal = lambda tag: "gemma4" in tag or "vl" in tag
        mod._pid_alive = lambda pid: pid in self.alive_pids
        self.alive_pids = set()
        # No luxuries in the fake tree: every lever records "unavailable", which
        # is itself the honest behaviour under test.
        return mod, actions

    def setUp(self):
        self.alive_pids = set()

    def tearDown(self):
        """Wait out any deferred settle-and-verify worker BEFORE the fixture is
        unpicked. game_mode_on hands that wait to a thread (it runs on the main
        voice thread and must not block it), and a worker still alive when
        `bobert_companion` is unpatched resolves _bc() to the REAL monolith —
        whose import fails on the singleton lock and, in failing, DELETES the
        fake the next test just installed. That is how one leaked worker turned
        into a SystemExit inside an unrelated test's _load(). tearDown runs
        before addCleanup, so this lands ahead of the sys.modules unpatch."""
        mod = getattr(self, "mod", None)
        t = getattr(mod, "_verify_thread", [None])[0] if mod is not None else None
        if t is not None:
            with mod._lock:
                mod._st.active = False      # make the wake-up a no-op...
            t.join(timeout=15)              # ...then actually wait for it


# ══════════════════════════════════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════════════════════════════════
class TestDetection(_Base):
    def _fake_procs(self, names):
        procs = []
        for i, n in enumerate(names, start=100):
            p = mock.MagicMock()
            p.info = {"pid": i, "name": n}
            procs.append(p)
        fake = types.ModuleType("psutil")
        fake.process_iter = lambda attrs=None: iter(procs)
        fake.pid_exists = lambda pid: True
        fake.virtual_memory = lambda: types.SimpleNamespace(available=9 * 1024**3)
        return mock.patch.dict(sys.modules, {"psutil": fake})

    def test_detects_the_real_shipping_client(self):
        mod, _ = self._load()
        with self._fake_procs(["chrome.exe", "FortniteClient-Win64-Shipping.exe"]):
            pid, exe = mod._find_game_pid()
        self.assertIsNotNone(pid)
        self.assertEqual(exe, GAME)

    def test_false_positive_eac_wrapper_does_not_trigger(self):
        """The measured trap: FortniteClient-Win64-Shipping_EAC_EOS.exe was
        running alongside the real client. A substring match fires on it."""
        mod, _ = self._load()
        with self._fake_procs(["FortniteClient-Win64-Shipping_EAC_EOS.exe",
                               "FortniteLauncher.exe",
                               "FortniteBootstrapper.exe"]):
            pid, exe = mod._find_game_pid()
        self.assertIsNone(pid, f"matched a non-game process: {exe}")

    def test_false_positive_unrelated_apps(self):
        mod, _ = self._load()
        with self._fake_procs(["UnrealEditor.exe", "Fusion360.exe",
                               "blender.exe", "Teams.exe"]):
            self.assertEqual(mod._find_game_pid(), (None, None))

    def test_empty_allowlist_never_matches(self):
        """Fail closed: with nothing allowlisted the mode can never engage,
        even if something game-shaped is running."""
        mod, _ = self._load(settings={"GAME_MODE_PROCESS_HINTS": []})
        with self._fake_procs(["FortniteClient-Win64-Shipping.exe"]):
            self.assertEqual(mod._find_game_pid(), (None, None))


# ══════════════════════════════════════════════════════════════════════════
#  HYSTERESIS / ANTI-FLAP
# ══════════════════════════════════════════════════════════════════════════
class TestHysteresis(_Base):
    def _cfg(self, **over):
        c = {"dwell": 20.0, "grace": 45.0, "max_seconds": 43200.0,
             "require_foreground": True}
        c.update(over)
        return c

    def test_requires_continuous_foreground_dwell(self):
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        self.assertEqual(mod.decide(1000.0, 42, 42, st, cfg)[0], "none")
        self.assertEqual(mod.decide(1010.0, 42, 42, st, cfg)[0], "none")
        self.assertEqual(mod.decide(1021.0, 42, 42, st, cfg)[0], "enter")

    def test_dwell_resets_when_focus_breaks(self):
        """A game LAUNCHING behind his work must never trip the mode."""
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        mod.decide(1000.0, 42, 42, st, cfg)
        mod.decide(1015.0, 42, 99, st, cfg)          # he clicked Chrome
        self.assertEqual(mod.decide(1021.0, 42, 42, st, cfg)[0], "none")
        self.assertEqual(mod.decide(1042.0, 42, 42, st, cfg)[0], "enter")

    def test_running_but_never_focused_never_enters(self):
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        for t in range(1000, 1200, 5):
            self.assertEqual(mod.decide(float(t), 42, 99, st, cfg)[0], "none")

    def test_alt_tab_does_not_exit(self):
        """PROCESS LIFETIME owns the mode; focus only ever confirmed ENTRY.
        A focus-driven exit would flap the brain between 9 GB and 15 GB — the
        sawtooth that is the actual hazard."""
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        st.active, st.game_pid, st.entered_at, st.last_seen_at = True, 42, 1000.0, 1000.0
        self.alive_pids = {42}
        for t in (1005.0, 1100.0, 1500.0, 5000.0):
            action, _ = mod.decide(t, 42, 99, st, cfg)   # game not focused
            self.assertEqual(action, "none", f"exited on alt-tab at t={t}")

    def test_exit_waits_out_the_relaunch_grace(self):
        """A crash-and-relaunch must not thrash two big model loads."""
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        st.active, st.game_pid, st.entered_at, st.last_seen_at = True, 42, 1000.0, 1000.0
        self.alive_pids = set()
        self.assertEqual(mod.decide(1020.0, None, None, st, cfg)[0], "none")
        self.assertEqual(mod.decide(1044.0, None, None, st, cfg)[0], "none")
        self.assertEqual(mod.decide(1046.0, None, None, st, cfg)[0], "exit")

    def test_manual_off_inhibits_re_entry_for_that_game(self):
        """L5. The watcher must never undo what the owner just asked for —
        the escape-hatch/guard race that sank a rival design."""
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        st.inhibit_pid = 42
        for t in range(1000, 1200, 5):
            self.assertEqual(mod.decide(float(t), 42, 42, st, cfg)[0], "none")


# ══════════════════════════════════════════════════════════════════════════
#  STUCK-STATE EXPIRY  (L3 deadman)
# ══════════════════════════════════════════════════════════════════════════
class TestStuckStateExpiry(_Base):
    def _cfg(self, **over):
        c = {"dwell": 20.0, "grace": 45.0, "max_seconds": 100.0,
             "require_foreground": True}
        c.update(over)
        return c

    def test_deadman_expires_a_stale_mode(self):
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        st.active, st.game_pid, st.entered_at = True, 42, 1000.0
        st.last_seen_at = 1000.0
        self.alive_pids = set()
        action, reason = mod.decide(1000.0 + 500.0, None, None, st, cfg)
        self.assertEqual(action, "exit")
        self.assertIn("ceiling", reason.lower() + " gone")

    def test_absolute_ceiling_exits_even_a_manual_hold(self):
        """A manual 'game mode on' is owned by the owner — but not forever.
        Nothing in this design can outlive the ceiling."""
        mod, _ = self._load()
        st, cfg = mod._State(), self._cfg()
        st.active, st.manual, st.game_pid, st.entered_at = True, True, 42, 1000.0
        st.last_seen_at = 1000.0
        self.alive_pids = {42}
        self.assertEqual(mod.decide(1050.0, 42, 42, st, cfg)[0], "none")
        self.assertEqual(mod.decide(1101.0, 42, 42, st, cfg)[0], "exit")
        st2 = mod._State()
        st2.active, st2.manual, st2.game_pid, st2.entered_at = True, True, 42, 1000.0
        act, why = mod.decide(1000.0 + 201.0, 42, 42, st2, cfg)
        self.assertEqual(act, "exit")
        self.assertIn("absolute ceiling", why)

    def test_a_crashed_jarvis_needs_no_cleanup_code(self):
        """L1, the layer that requires no code to be correct: the skill writes
        NOTHING to disk, so there is no stale state a restart could read back.
        Pinned by AST, not by trust."""
        import ast
        import os
        src = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "skills", "game_mode.py")
        with open(src, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        writes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name in ("open", "write_text", "makedirs", "mkdir"):
                    writes.append(name)
        # `open` appears once, in a TEST-free read of nothing; assert no writer.
        self.assertEqual([w for w in writes if w != "open"], [],
                         f"game_mode writes to disk: {writes}")


# ══════════════════════════════════════════════════════════════════════════
#  THE DOWNSHIFT — and the stale-table refusal
# ══════════════════════════════════════════════════════════════════════════
class TestBrainDownshift(_Base):
    def test_picks_a_measurably_smaller_tag(self):
        mod, _ = self._load()
        tag, why = mod._pick_game_brain(BIG)
        self.assertEqual(tag, SMALL)
        self.assertIn("on disk", why)

    def test_refuses_to_shrink_upward(self):
        """core/vram_budget.py calls gemma4:latest 4 GB. Live `ollama list`
        says 9.6 GB — LARGER than gemma4:12b. Trusting the written table over
        the live box is this project's signature bug; the guard is the fix."""
        mod, _ = self._load(settings={"GAME_MODE_BRAIN": "gemma4:latest"})
        tag, why = mod._pick_game_brain(SMALL)
        self.assertIsNone(tag)
        self.assertIn("not a downshift", why)

    def test_refuses_when_tags_are_unreadable(self):
        """Entry fails CLOSED: repointing at a tag we cannot prove is installed
        makes _call_local_llm kick a background pull and return None — a mute."""
        mod, _ = self._load()
        mod._installed_sizes_mb = lambda: {}
        tag, why = mod._pick_game_brain(BIG)
        self.assertIsNone(tag)
        self.assertIn("refusing", why.lower())

    def test_refuses_an_uninstalled_tag(self):
        mod, _ = self._load(settings={"GAME_MODE_BRAIN": "nonexistent:70b"})
        tag, why = mod._pick_game_brain(BIG)
        self.assertIsNone(tag)
        self.assertIn("not installed", why)

    def test_repoint_hits_the_resolver_cache_chat_and_vision(self):
        """The whole no-restart mechanism: _call_local_llm reads
        _RESOLVED_LOCAL_LLM_MODEL[0] fresh every turn, _call_local_vision reads
        LOCAL_VISION_MODEL fresh every call. Repointing those covers EVERY
        consumer — no gated call sites, no daemon stop APIs, no restart."""
        mod, _ = self._load()
        notes = []
        mod._repoint_brain(SMALL, notes)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL)
        self.assertEqual(self.bc.LOCAL_LLM_MODEL, SMALL)
        self.assertEqual(self.bc.LOCAL_VISION_MODEL, SMALL,
                         "vision forked off the chat tag — a vision call would "
                         "reload the very model we just unloaded")

    def test_nothing_is_persisted(self):
        """persist=False is load-bearing: model_picker.set_model() ALWAYS
        persists, which would permanently rewrite his brain choice and let a
        crash strand him on the small model."""
        mod, _ = self._load()
        picker = types.ModuleType("skill_model_picker")
        picker._persist_setting = mock.MagicMock()
        picker._sync_vision_to_chat = mock.MagicMock(return_value=True)
        picker.set_model = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"skill_model_picker": picker}):
            mod._repoint_brain(SMALL, [])
        picker._persist_setting.assert_not_called()
        picker.set_model.assert_not_called()
        _old, _new = picker._sync_vision_to_chat.call_args[0][:2]
        self.assertIs(picker._sync_vision_to_chat.call_args[1]["persist"], False)


# ══════════════════════════════════════════════════════════════════════════
#  ENTER / EXIT — restore everything that was touched
# ══════════════════════════════════════════════════════════════════════════
class TestEnterExit(_Base):
    def test_enter_downshifts_unloads_and_reports_a_measured_delta(self):
        mod, actions = self._load(vram=(23269, 1307), after_vram=(8100, 16476))
        out = mod._enter(47036, GAME)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL)
        self.assertEqual(self.unloaded, [BIG])
        self.assertIn("15169 MB of VRAM", out)
        self.assertTrue(mod._st.active)

    def test_unload_is_skipped_when_the_downshift_did_not_take(self):
        """ORDERING IS LOAD-BEARING. Unloading without a successful downshift
        only schedules a full ~15 GB cold reload into a nearly-full card."""
        mod, _ = self._load(settings={"GAME_MODE_BRAIN": "nonexistent:70b"})
        mod._enter(47036, GAME)
        self.assertEqual(self.unloaded, [],
                         "unloaded the big brain with nothing smaller resolved")
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], BIG)

    def test_exit_restores_every_touched_setting(self):
        mod, _ = self._load()
        before = {
            "resolved": self.bc._RESOLVED_LOCAL_LLM_MODEL[0],
            "llm": self.bc.LOCAL_LLM_MODEL,
            "vision": self.bc.LOCAL_VISION_MODEL,
            "hud": self.bc.HUD_CAMERA_PREVIEW,
        }
        mod._enter(47036, GAME)
        self.assertNotEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], before["resolved"])
        self.alive_pids = set()
        mod._find_game_pid = lambda: (None, None)      # the game has exited
        mod._exit("game process exited")
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], before["resolved"])
        self.assertEqual(self.bc.LOCAL_LLM_MODEL, before["llm"])
        self.assertEqual(self.bc.LOCAL_VISION_MODEL, before["vision"])
        self.assertEqual(self.bc.HUD_CAMERA_PREVIEW, before["hud"])
        self.assertFalse(mod._st.active)

    def test_exit_does_not_rewarm_the_big_brain_while_a_game_still_runs(self):
        """A deadman exit with the game still up must NOT restore the 15 GB
        brain — that is the spike this feature exists to remove."""
        mod, _ = self._load()
        mod._enter(47036, GAME)
        mod._find_game_pid = lambda: (47036, GAME)     # still gaming
        out = mod._exit("deadman ceiling reached")
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL)
        self.assertFalse(mod._st.active)
        del out

    def test_below_floor_is_reported_as_failure_not_a_saving(self):
        """A mode that says it freed memory it did not free is this project's
        defining bug in its purest form."""
        mod, _ = self._load(vram=(23269, 1307), after_vram=(23000, 1576))
        out = mod._enter(47036, GAME)
        self.assertIn("did NOT free", out)
        self.assertEqual(mod._st.result, "failed")
        # The status must carry the failure forward rather than quietly
        # reporting an engaged mode — a "successful" tile over a 269 MB delta
        # is the exact defect this feature is written against.
        status = mod.game_mode_status("").lower()
        self.assertIn("failed to free", status)
        self.assertNotIn("freed by measurement", status)

    def test_unreadable_gpu_claims_nothing(self):
        mod, _ = self._load()
        mod._nvidia_smi_mb = lambda: (None, None)
        out = mod._enter(47036, GAME)
        self.assertIn("no measured saving", out)
        self.assertEqual(mod._st.result, "unverified")

    def test_game_mode_on_starts_the_watcher_so_it_ends_by_itself(self):
        """The no-restart path. _apply_user_settings runs at IMPORT time, so a
        GAME_MODE_ENABLED written to user_settings.json cannot reach a running
        JARVIS — register() would still see False. Firing the action must
        therefore also arm auto-exit, or he is left in a mode he has to leave by
        hand, which is the one thing the owner must never have to do."""
        mod, actions = self._load()
        mod._find_game_pid = lambda: (47036, GAME)
        started = []
        mod.start_watcher = lambda: (started.append(True), True)[1]
        out = actions["game_mode_on"]("")
        self.assertTrue(started, "game_mode_on did not arm the auto-exit watcher")
        self.assertIn("watch for the game closing", out)
        # A game that is already running OWNS the mode, so it auto-exits on
        # process death rather than becoming an open-ended manual hold.
        self.assertFalse(mod._st.manual)
        self.assertEqual(mod._st.game_pid, 47036)

    def test_manual_on_with_no_game_is_a_bounded_manual_hold(self):
        mod, actions = self._load()
        mod._find_game_pid = lambda: (None, None)
        mod.start_watcher = lambda: False
        actions["game_mode_on"]("")
        self.assertTrue(mod._st.manual)

    def test_explicit_on_clears_a_previous_off_inhibit(self):
        mod, actions = self._load()
        mod.start_watcher = lambda: False
        mod._find_game_pid = lambda: (47036, GAME)
        actions["game_mode_on"]("")
        actions["game_mode_off"]("")
        self.assertEqual(mod._st.inhibit_pid, 47036)
        actions["game_mode_on"]("")
        self.assertIsNone(mod._st.inhibit_pid)

    def test_manual_off_inhibits_and_status_is_honest(self):
        mod, actions = self._load()
        mod._enter(47036, GAME)
        mod._find_game_pid = lambda: (None, None)
        out = actions["game_mode_off"]("")
        self.assertIn("normal power", out)
        self.assertEqual(mod._st.inhibit_pid, 47036)
        self.assertIn("not engaged", actions["game_mode_status"](""))


# ══════════════════════════════════════════════════════════════════════════
#  THE VOICE THREAD IS NOT A WORKER
# ══════════════════════════════════════════════════════════════════════════
class TestVoiceThreadIsNeverBlocked(_Base):
    """Actions are invoked SYNCHRONOUSLY — `res = fn(arg)` inside _runner —
    from parse_and_run_actions on the MAIN LOOP thread: the same thread that
    runs record_speech() and ticks _heartbeat(). `_enter` used to park that
    thread for the fixed GAME_MODE_VERIFY_DELAY_SECONDS settle wait plus a
    second nvidia-smi + /api/ps sample, so "divert power" mid-match answered him
    with ten-plus seconds of DEAD AIR — no listening, no speaking, no heartbeat
    — at the exact moment he asked for help, and one slow Ollama socket on top
    of it walks toward the 60 s mic-reset watchdog. Nothing about that wait
    needs the caller: it exists only to let an unload show up in the driver's
    numbers before the 'after' sample."""

    def test_game_mode_on_returns_without_waiting_out_the_settle_delay(self):
        mod, actions = self._load(vram=(23269, 1307), after_vram=(8100, 16476),
                                  settings={"GAME_MODE_VERIFY_DELAY_SECONDS": 6.0})
        mod._find_game_pid = lambda: (47036, GAME)
        mod.start_watcher = lambda: False

        t0 = time.monotonic()
        out = actions["game_mode_on"]("")
        elapsed = time.monotonic() - t0

        # (_Base.tearDown waits the worker out, so a failure below cannot leak
        # a still-sleeping thread into the next test.)
        self.assertLess(elapsed, 3.0,
                        f"game_mode_on held the voice thread for {elapsed:.1f}s "
                        f"of a 6s settle wait — that is JARVIS deaf and silent "
                        f"mid-match")
        # The levers are applied BEFORE it returns, so the mode is real and
        # 'normal power' one second later still finds something to leave. Only
        # the MEASUREMENT is deferred, never the mode.
        self.assertTrue(mod._st.active)
        self.assertEqual(mod._st.game_pid, 47036)
        # And it reports NO number, because nothing has been measured yet —
        # claiming a saving that was never established is this repo's #1 defect.
        self.assertNotRegex(out, r"\d+\s*MB")

    def test_the_measured_verdict_still_reaches_him_afterwards(self):
        """Returning early must not LOSE the number. A mode that computes
        '15169 MB freed' and drops it on the floor is the same defect from the
        other side."""
        mod, actions = self._load(vram=(23269, 1307), after_vram=(8100, 16476),
                                  settings={"GAME_MODE_VERIFY_DELAY_SECONDS": 0.05})
        mod._find_game_pid = lambda: (47036, GAME)
        mod.start_watcher = lambda: False
        actions["game_mode_on"]("")

        t = mod._verify_thread[0]
        self.assertIsNotNone(t, "no settle-and-verify worker was started")
        t.join(timeout=15)
        self.assertFalse(t.is_alive(), "the verify worker never finished")

        self.assertEqual(mod._st.result, "ok")
        spoken = [c.args[0] for c in self.bc.proactive_announce.call_args_list]
        self.assertTrue(any("15169 MB of VRAM" in s for s in spoken),
                        f"the measured verdict never reached him: {spoken}")

    def test_a_deferred_verify_never_reports_a_session_he_already_ended(self):
        """'normal power' two seconds after 'divert power' is exactly what L5
        exists to allow. The in-flight verify must not then announce a delta for
        a mode that is already over."""
        mod, actions = self._load(settings={"GAME_MODE_VERIFY_DELAY_SECONDS": 1.0})
        mod._find_game_pid = lambda: (47036, GAME)
        mod.start_watcher = lambda: False
        actions["game_mode_on"]("")
        mod._find_game_pid = lambda: (None, None)
        actions["game_mode_off"]("")            # he changed his mind at once

        t = mod._verify_thread[0]
        self.assertIsNotNone(t, "no settle-and-verify worker was started")
        t.join(timeout=15)
        self.assertFalse(t.is_alive())
        self.assertFalse(mod._st.active)
        spoken = [c.args[0] for c in self.bc.proactive_announce.call_args_list]
        self.assertEqual(spoken, [],
                         f"spoke about a session he had already ended: {spoken}")

    def test_the_watcher_path_still_verifies_inline(self):
        """One implementation of the wait, two schedulers. The watcher is
        already a background thread, so it keeps the inline path — and must
        still come back with the measured sentence, not an acknowledgement."""
        mod, _ = self._load(vram=(23269, 1307), after_vram=(8100, 16476),
                            settings={"GAME_MODE_VERIFY_DELAY_SECONDS": 0.0})
        out = mod._enter(47036, GAME)
        self.assertIn("15169 MB of VRAM", out)
        self.assertIsNone(mod._verify_thread[0],
                          "the watcher path spawned a worker it did not need")


# ══════════════════════════════════════════════════════════════════════════
#  NEVER MUTE — the crash-mid-transition guarantee
# ══════════════════════════════════════════════════════════════════════════
class TestNeverMutes(_Base):
    def test_crash_at_every_step_never_mutes(self):
        """There is no cloud on this box, so an unset brain is a MUTE JARVIS.
        Inject a failure at each step of enter and assert the chat brain still
        points at a real, non-empty tag every time."""
        steps = ["_pick_game_brain", "_repoint_brain", "_unload",
                 "_suspend_luxuries", "_measure"]
        for step in steps:
            with self.subTest(step=step):
                mod, _ = self._load()
                def _boom(*a, **kw):
                    raise RuntimeError(f"injected failure in {step}")
                setattr(mod, step, _boom)
                try:
                    mod._enter(47036, GAME)
                except Exception:
                    pass                       # a raise is allowed; a mute is not
                tag = mod._current_brain()
                self.assertTrue(tag and tag.strip(),
                                f"{step} left the brain unset -> JARVIS is mute")
                self.assertNotIn(tag.strip().lower(), ("off", "none"))

    def test_invariant_repairs_an_unset_brain(self):
        mod, _ = self._load()
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = ""
        self.bc.LOCAL_LLM_MODEL = ""
        notes = []
        ok = mod._assert_brain_usable(notes)
        self.assertFalse(ok)
        self.assertTrue(any("INVARIANT VIOLATED" in n for n in notes))
        self.assertTrue(mod._current_brain())

    def test_the_mode_never_gates_the_llm(self):
        """The rejected design suppressed _call_local_llm and relied on a cloud
        fallback that does not exist here — producing 'my local model isn't
        responding' on every turn, which is both mute AND a false diagnosis.
        Assert this skill touches no suppression flag."""
        import os
        src = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "skills", "game_mode.py")
        with open(src, "r", encoding="utf-8") as f:
            body = f.read()
        for forbidden in ("LOCAL_LLM_FALLBACK", "llm_suspended",
                          "set_suppressed", "AI_BACKEND ="):
            self.assertNotIn(f"{forbidden} =", body,
                             f"game_mode assigns {forbidden} — that path mutes him")


# ══════════════════════════════════════════════════════════════════════════
#  KEEP-WARM MUST NEVER CAUSE A LOAD
# ══════════════════════════════════════════════════════════════════════════
class TestKeepWarm(_Base):
    def test_keep_warm_never_loads_an_evicted_model(self):
        """`keep_alive: "20m"` is hardcoded at bobert_companion.py:11148, so no
        long pin survives the owner's next utterance. The refresh therefore only
        fires when the model is ALREADY RESIDENT — otherwise the 'refresh' IS
        the ~9 GB cold load it was meant to prevent."""
        mod, _ = self._load()
        self.resident = []                       # evicted
        posted = []
        with mock.patch.object(mod.urllib.request, "urlopen",
                               side_effect=lambda *a, **k: posted.append(a)):
            out = mod._keep_warm(SMALL)
        self.assertIn("not resident", out)
        self.assertEqual(posted, [], "keep-warm fired a request against an "
                                     "evicted model — that is a cold load")

    def test_keep_warm_refreshes_a_resident_model(self):
        mod, _ = self._load()
        self.resident = [{"name": SMALL, "size_vram_mb": 8800}]
        fake = mock.MagicMock()
        fake.__enter__ = lambda s: s
        fake.__exit__ = lambda s, *a: False
        fake.read = lambda: b"{}"
        with mock.patch.object(mod.urllib.request, "urlopen", return_value=fake):
            out = mod._keep_warm(SMALL)
        self.assertIn("refreshed", out)
        self.assertIn("num_ctx", out)


# ══════════════════════════════════════════════════════════════════════════
#  WIRING — registered, routable, voiced, declared
# ══════════════════════════════════════════════════════════════════════════
class TestWiring(_Base):
    def test_actions_registered(self):
        _mod, actions = self._load()
        for name in ("game_mode_status", "game_mode_on", "game_mode_off",
                     "game_mode_learn_this", "normal_power", "full_power"):
            self.assertIn(name, actions)

    def test_actions_are_routable_via_prompt_examples(self):
        """Registering a handler does NOT teach the local brain the name. A name
        is only emittable if it appears in core/prompts.py or in a module-level
        PROMPT_EXAMPLES that load_skills collects. This exact mistake was made
        and caught on 2026-09-04."""
        mod, _ = self._load()
        pe = getattr(mod, "PROMPT_EXAMPLES", "")
        self.assertIsInstance(pe, str)
        for name in ("game_mode_status", "game_mode_on", "game_mode_off",
                     "game_mode_learn_this"):
            self.assertIn(name, pe, f"{name} is registered but unroutable")
        self.assertIn("[ACTION:", pe)

    def test_results_are_spoken_verbatim(self):
        mod, _ = self._load()
        declared = set(getattr(mod, "SPEAK_VERBATIM_ACTIONS", ()))
        for name in ("game_mode_status", "game_mode_on", "game_mode_off"):
            self.assertIn(name, declared,
                          f"{name} returns a finished sentence that would be "
                          f"computed, logged and dropped")

    def test_every_setting_is_declared_in_core_config(self):
        """_apply_user_settings SKIPS any key not already in core.config's
        globals(), so an undeclared setting is silently dropped and the owner's
        saved value never reaches the runtime."""
        import re
        import os
        import core.config as cfg
        src = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "skills", "game_mode.py")
        with open(src, "r", encoding="utf-8") as f:
            body = f.read()
        used = set(re.findall(r'_cfg(?:_float|_int)?\(\s*"(GAME_MODE_[A-Z_]+)"', body))
        self.assertTrue(used, "no GAME_MODE_* settings found to check")
        missing = [k for k in sorted(used) if not hasattr(cfg, k)]
        self.assertEqual(missing, [],
                         f"undeclared in core/config.py, silently dropped: {missing}")

    def test_not_enabled_by_default(self):
        import core.config as cfg
        self.assertFalse(cfg.GAME_MODE_ENABLED)

    def test_watcher_does_not_autostart_when_disabled(self):
        mod, _ = self._load()
        self.assertIsNone(mod._thread[0])

    def test_status_is_honest_when_not_engaged(self):
        mod, actions = self._load()
        mod._find_game_pid = lambda: (None, None)
        out = actions["game_mode_status"]("")
        self.assertIn("not engaged", out)
        self.assertIn(BIG, out)


# ══════════════════════════════════════════════════════════════════════════
#  THE OWNER'S OWN SETTINGS — game mode must never switch a feature ON,
#  and must never write one to disk
# ══════════════════════════════════════════════════════════════════════════
def _kinect_fake(mod_name, flag, off_name, on_name, persisted):
    """A stand-in skill module mirroring the REAL control flow of
    kinect_air_mouse / kinect_gestures, installed purely as a TRIPWIRE.

    game_mode must never drive these two through their spoken actions, because:

      * the OFF action EARLY-RETURNS when the flag is already clear — no
        exception, no state, nothing a caller can detect from the outside;
      * the ON action persists the flag to data/user_settings.json
        UNCONDITIONALLY, BEFORE its own "already on" early return.

    `persisted` collects every (key, value) that would have reached disk, so a
    test can assert on the file write without going anywhere near his live
    data/user_settings.json. If it is ever non-empty, someone reintroduced the
    action call."""
    m = types.ModuleType(mod_name)
    calls = []

    def _cfg_flag(key, default=False):
        import core.config as cfg
        return bool(getattr(cfg, key, default))

    def _set(on):                   # _set_enabled / _set_gestures_enabled
        import core.config as cfg
        setattr(cfg, flag, bool(on))            # live flip ...
        persisted.append((flag, bool(on)))      # ... AND a write to disk
        return True

    def _off(_=""):
        calls.append(off_name)
        if not _cfg_flag(flag):
            return "already off, sir."          # <- the invisible no-op
        _set(False)
        return "off, sir."

    def _on(_=""):
        calls.append(on_name)
        _set(True)                              # <- runs even when already on
        return "on, sir."

    m._cfg_flag = _cfg_flag
    setattr(m, off_name, _off)
    setattr(m, on_name, _on)
    m._calls = calls
    return m


class TestNeverEnablesWhatHeTurnedOff(_Base):
    """REGRESSION, 2026-09-04. Measured read-only from his live
    data/user_settings.json that evening: KINECT_AIR_MOUSE_ENABLED = False,
    KINECT_GESTURES_ENABLED = True.

    The defect: _suspend_luxuries recorded a restore for BOTH kinect levers
    regardless of what it had actually stopped, because the OFF actions
    early-return silently when the feature is already off. On exit it therefore
    called air_mouse_on(), whose unconditional _set_enabled(True) writes through
    _persist_setting -> tools.settings_window.save_settings ->
    data/user_settings.json. So the first time he closed Fortnite, the air-mouse
    he had deliberately switched off would turn ON, stay on through every future
    reboot, and the skill would log "kinect_air_mouse: restored" — this repo's
    defining defect (claiming a result it never verified) aimed at the one
    feature whose closed-fist click has closed his Chrome tabs.

    These two are also the only levers whose restore reached the DISK, which is
    why test_a_crashed_jarvis_needs_no_cleanup_code's AST scan of game_mode.py
    could not see it: the write happened two modules away.

    The assertions below are deliberately about OBSERVABLE end state — the live
    core.config flags, and whether anything was persisted — not about which
    helper does the work, so they hold across either fix and fail on a revert to
    either shape of the bug."""

    FLAGS = ("KINECT_AIR_MOUSE_ENABLED", "KINECT_GESTURES_ENABLED")

    def _flags(self, *, air_mouse, gestures):
        """Pin the two live flags for the duration of one test, restored by
        mock.patch on teardown so no test can leak into the next — or into the
        real process's config."""
        import core.config as cfg
        for name, val in zip(self.FLAGS, (air_mouse, gestures)):
            self.assertTrue(
                hasattr(cfg, name),
                f"{name} is undeclared in core/config.py — the lever would "
                f"silently do nothing")
            patch = mock.patch.object(cfg, name, val)
            patch.start()
            self.addCleanup(patch.stop)
        return cfg

    def _tripwires(self):
        """Install fake kinect skills. game_mode must NOT call them at all."""
        self.persisted = []
        air = _kinect_fake("skill_kinect_air_mouse", "KINECT_AIR_MOUSE_ENABLED",
                           "air_mouse_off", "air_mouse_on", self.persisted)
        ges = _kinect_fake("skill_kinect_gestures", "KINECT_GESTURES_ENABLED",
                           "gestures_off", "gestures_on", self.persisted)
        patch = mock.patch.dict(sys.modules, {
            "skill_kinect_air_mouse": air, "skill_kinect_gestures": ges})
        patch.start()
        self.addCleanup(patch.stop)
        return air, ges

    # ── the defect itself ────────────────────────────────────────────────
    def test_exit_does_not_switch_on_an_air_mouse_he_left_off(self):
        mod, _ = self._load()
        cfg = self._flags(air_mouse=False, gestures=True)
        self._tripwires()

        mod._enter(47036, GAME)
        mod._find_game_pid = lambda: (None, None)      # the game has exited
        mod._exit("game process exited")

        self.assertFalse(
            cfg.KINECT_AIR_MOUSE_ENABLED,
            "leaving game mode switched ON the air-mouse he deliberately left "
            "off — a global cursor takeover with synthetic click injection")
        self.assertNotIn(
            ("KINECT_AIR_MOUSE_ENABLED", True), self.persisted,
            "persisted KINECT_AIR_MOUSE_ENABLED=True to "
            "data/user_settings.json — that survives every reboot")

    def test_a_lever_already_off_records_no_undo(self):
        """Not merely restored correctly — never recorded at all. A lever that
        stopped nothing must leave nothing for the exit to 'restore'."""
        mod, _ = self._load()
        cfg = self._flags(air_mouse=False, gestures=True)
        self._tripwires()

        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)

        self.assertNotIn("kinect_air_mouse", applied,
                         "recorded an undo for a lever that stopped nothing")
        self.assertTrue(any("kinect_air_mouse" in n and "already off" in n
                            for n in notes), notes)

        mod._restore_luxuries(notes, applied)
        self.assertFalse(cfg.KINECT_AIR_MOUSE_ENABLED)

    # ── and the half that must keep working ──────────────────────────────
    def test_a_feature_that_was_on_is_still_idled_and_put_back(self):
        """The stray click mid-match is the reason this lever exists. Gating on
        prior state must not weaken it for a feature that IS on."""
        mod, _ = self._load()
        cfg = self._flags(air_mouse=True, gestures=True)
        self._tripwires()

        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        self.assertFalse(cfg.KINECT_AIR_MOUSE_ENABLED,
                         "left the air-mouse live in-game")
        self.assertFalse(cfg.KINECT_GESTURES_ENABLED,
                         "left gesture control live in-game")
        self.assertIn("kinect_air_mouse", applied)
        self.assertIn("kinect_gestures", applied)

        mod._restore_luxuries(notes, applied)
        self.assertTrue(cfg.KINECT_AIR_MOUSE_ENABLED,
                        "did not give the air-mouse back")
        self.assertTrue(cfg.KINECT_GESTURES_ENABLED,
                        "did not give gesture control back")

    def test_a_full_round_trip_leaves_both_flags_exactly_as_he_set_them(self):
        for air_on in (False, True):
            for ges_on in (False, True):
                with self.subTest(air_mouse=air_on, gestures=ges_on):
                    mod, _ = self._load()
                    cfg = self._flags(air_mouse=air_on, gestures=ges_on)
                    self._tripwires()
                    mod._enter(47036, GAME)
                    mod._find_game_pid = lambda: (None, None)
                    mod._exit("game process exited")
                    self.assertIs(cfg.KINECT_AIR_MOUSE_ENABLED, air_on)
                    self.assertIs(cfg.KINECT_GESTURES_ENABLED, ges_on)
                    self.assertEqual(self.persisted, [],
                                     "game mode wrote a kinect flag to "
                                     "data/user_settings.json")

    def test_the_persisting_actions_are_never_called_at_all(self):
        """L1: a crash mid-game must need no cleanup code. gestures_off()
        persists KINECT_GESTURES_ENABLED=false, and that write SURVIVES THE
        POWER BUTTON — the documented end state of the very scenario this skill
        exists for (lobby leak -> E_OUTOFMEMORY -> RenderThread hang -> hard
        reset). _restore_luxuries never runs on that path, so the next boot
        would read the persisted false and his gesture control would be dead for
        good, silently. The live config flag is the only correct lever."""
        mod, _ = self._load()
        self._flags(air_mouse=True, gestures=True)
        air, ges = self._tripwires()

        mod._enter(47036, GAME)
        mod._find_game_pid = lambda: (None, None)
        mod._exit("game process exited")

        self.assertEqual(air._calls, [], "game_mode called a persisting "
                                         "air-mouse action")
        self.assertEqual(ges._calls, [], "game_mode called a persisting "
                                         "gesture action")
        self.assertEqual(self.persisted, [])

    # ── the premise the whole design rests on ────────────────────────────
    def test_the_premise_that_makes_those_actions_unusable_here(self):
        """Pins the two facts in kinect_air_mouse / kinect_gestures that make
        driving them through their actions destructive, so nobody 'simplifies'
        game_mode back into calling them without first reading these:

          1. the ON action persists UNCONDITIONALLY, before any early return —
             so a blind restore ENABLES a feature that was off;
          2. the OFF action early-returns with no exception and no state — so
             the caller cannot tell a real stop from a no-op.

        If this fails, those actions changed — re-evaluate against the new
        behaviour rather than deleting the guard."""
        import ast
        import os
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        for fname, on_name, off_name, setter in (
                ("kinect_air_mouse.py", "air_mouse_on", "air_mouse_off",
                 "_set_enabled"),
                ("kinect_gestures.py", "gestures_on", "gestures_off",
                 "_set_gestures_enabled"),
        ):
            path = os.path.join(root, "skills", fname)
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            fns = {n.name: n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}
            self.assertIn(on_name, fns)
            self.assertIn(off_name, fns)

            # 1. the persisting setter is called at the FUNCTION's top level —
            #    i.e. unconditionally, whatever the flag already was.
            unconditional = any(
                any(isinstance(c, ast.Call)
                    and getattr(c.func, "id", None) == setter
                    for c in ast.walk(stmt))
                for stmt in fns[on_name].body
                if not isinstance(stmt, (ast.If, ast.Try)))
            self.assertTrue(
                unconditional,
                f"{fname}:{on_name} no longer persists unconditionally — the "
                f"reason game_mode must not call it may have changed")

            # 2. that setter really does reach the settings file.
            persists = any(
                isinstance(c, ast.Call)
                and getattr(c.func, "id", None) == "_persist_setting"
                for c in ast.walk(fns[setter]))
            self.assertTrue(
                persists,
                f"{fname}:{setter} no longer persists — re-check whether "
                f"game_mode's config-flag lever is still required")

            # 3. the OFF action's first real statement is a bare early return.
            body = [s for s in fns[off_name].body
                    if not (isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str))]
            self.assertIsInstance(
                body[0], ast.If,
                f"{fname}:{off_name} no longer opens with the already-off gate")
            self.assertTrue(
                any(isinstance(s, ast.Return) for s in body[0].body),
                f"{fname}:{off_name}'s already-off gate no longer returns "
                f"early — its no-op is no longer invisible to a caller")


# ══════════════════════════════════════════════════════════════════════════
#  THE FACE-DETECTION PAUSE MUST ACTUALLY HOLD
# ══════════════════════════════════════════════════════════════════════════
class TestFaceDetectionPauseIsActuallyHeld(_Base):
    """REGRESSION, 2026-09-04. The mode's single largest claimed CPU saving was
    cleared by JARVIS's own main loop within one listen cycle.

    bc.pause_face_tracking() is a bare `_face_track_pause.set()` on a SHARED
    threading.Event with no ref-counting (bobert_companion.py:7924-7936), and
    the main voice loop calls resume_face_tracking() — a bare `.clear()` — on
    EVERY iteration, immediately before record_speech(timeout=20)
    (bobert_companion.py:23917, and :23893 on the realtime path). So within at
    most ~20 s of engaging mid-match the Haar-cascade detection was back for the
    rest of the match — 2.31 cores of pythonw competing with Fortnite on a box
    with 2.5 GB free — while _st.notes still said "face_tracking_detect:
    stopped", game_mode_status implied the lever was held, and the exit path
    logged "restored" after calling a resume that was by then a no-op.

    A mode that says it stopped something it could not stop is the exact defect
    _lever's docstring says this module exists not to be, so these tests read
    the ONE thing the capture loop actually reads (bobert_companion.py:5835,
    `paused = _face_track_pause.is_set()`), never the notes alone."""

    def _detection_running(self):
        """What the capture loop asks once per camera frame: a falsey
        `_face_track_pause.is_set()` means the Haar cascade runs this frame
        (bobert_companion.py:5835, gating the `continue` at :6121 and :6158)."""
        return not self.bc._face_track_pause.is_set()

    # ── the defect itself ────────────────────────────────────────────────
    def test_the_pause_survives_the_main_loops_per_cycle_resume(self):
        mod, _ = self._load()
        mod._enter(47036, GAME)
        self.assertFalse(self._detection_running(),
                         "detection was never paused at all")
        _voice_loop_listen_cycles(self.bc, 5)      # five listen windows, ~100 s
        self.assertFalse(
            self._detection_running(),
            "the voice loop's resume_face_tracking() unpaused the Haar cascade "
            "while game mode still reported it stopped — 2.31 cores handed back "
            "to the box mid-match")

    def test_the_note_is_not_true_for_one_breath_only(self):
        """The note and the Event must agree AFTER the loop has run, not just
        at the instant of engaging."""
        mod, _ = self._load()
        mod._enter(47036, GAME)
        note = next(n for n in mod._st.notes
                    if n.startswith("face_tracking_detect"))
        self.assertIn("HELD", note, note)
        _voice_loop_listen_cycles(self.bc, 3)
        self.assertFalse(self._detection_running(), note)

    def test_suppressed_resumes_are_counted_not_assumed(self):
        """Measured, not inferred: the latch reports how many resumes it ate,
        so 'the hold held' is a number rather than a belief."""
        mod, _ = self._load()
        mod._enter(47036, GAME)
        _voice_loop_listen_cycles(self.bc, 4)
        self.assertEqual(self.bc._face_track_pause.suppressed, 4)

    # ── and the half that must keep working ──────────────────────────────
    def test_exit_puts_the_original_event_back_and_detection_returns(self):
        mod, _ = self._load()
        original = self.bc._face_track_pause
        mod._enter(47036, GAME)
        self.assertIsNot(self.bc._face_track_pause, original)
        _voice_loop_listen_cycles(self.bc, 2)

        mod._find_game_pid = lambda: (None, None)      # the game has exited
        mod._exit("game process exited")

        self.assertIs(self.bc._face_track_pause, original,
                      "exit left a latched Event installed — every later "
                      "resume_face_tracking() would be swallowed for the rest "
                      "of the session, so JARVIS would never look at him again")
        self.assertTrue(self._detection_running(),
                        "face detection did not come back after game mode")
        # ...and ordinary voice-time pause/resume works again, unlatched.
        self.bc.pause_face_tracking()
        self.assertFalse(self._detection_running())
        self.bc.resume_face_tracking()
        self.assertTrue(self._detection_running())

    def test_a_pause_the_speech_path_owned_is_handed_back_still_paused(self):
        """_lever's never-switch-on-what-was-off guarantee, kept: if the SPEECH
        path owned the pause when we engaged (JARVIS was talking), exit must not
        resume detection on its behalf. The restore is exact, not opinionated."""
        mod, _ = self._load()
        self.bc.pause_face_tracking()                  # JARVIS is mid-sentence
        original = self.bc._face_track_pause
        mod._enter(47036, GAME)
        _voice_loop_listen_cycles(self.bc, 2)
        self.assertFalse(self._detection_running())

        mod._find_game_pid = lambda: (None, None)
        mod._exit("game process exited")

        self.assertIs(self.bc._face_track_pause, original)
        self.assertFalse(self._detection_running(),
                         "cleared a pause game mode never set — the speech path "
                         "owned it")

    def test_an_unlatchable_build_claims_nothing(self):
        """Fails safe in the honest direction: with no latchable Event the mode
        must SKIP and say so, never record 'stopped' for a pause the voice loop
        would clear inside one listen window."""
        mod, _ = self._load()
        self.bc._face_track_pause = None               # not an Event here
        notes, applied = [], {}
        mod._suspend_luxuries(notes, applied)
        note = next(n for n in notes if n.startswith("face_tracking_detect"))
        self.assertIn("SKIPPED", note)
        self.assertNotIn("face_tracking_detect", applied,
                         "recorded an undo for a pause it never held")

    def test_the_premise_that_makes_the_latch_necessary(self):
        """Pins the three monolith facts this fix rests on, so nobody
        'simplifies' the latch back into a one-shot pause_face_tracking():

          1. resume_face_tracking() is a bare clear() with no ref-counting;
          2. the main voice loop calls it immediately before record_speech();
          3. the capture loop re-reads _face_track_pause as a module global
             (which is what lets an attribute swap work with no monolith edit).

        Substring checks, not an AST parse: the monolith is ~25k lines and this
        suite runs on a box that is short of RAM while he games.

        If this fails, the monolith changed — re-evaluate the latch against the
        new behaviour rather than deleting it."""
        import os
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, "bobert_companion.py"),
                  "r", encoding="utf-8") as f:
            body = f.read()

        self.assertIn(
            "def resume_face_tracking():\n    _face_track_pause.clear()", body,
            "resume_face_tracking() no longer bare-clears _face_track_pause — "
            "if it became ref-counted, game_mode's latch may be removable")

        rec = body.find("audio = record_speech(timeout=20)")
        self.assertNotEqual(rec, -1,
                            "the main listen loop's record_speech() moved")
        self.assertIn(
            "resume_face_tracking()", body[max(0, rec - 400):rec],
            "the main listen loop no longer calls resume_face_tracking() just "
            "before record_speech() — re-verify why the latch exists before "
            "touching it")

        self.assertIn(
            "paused = _face_track_pause.is_set()", body,
            "the capture loop no longer re-reads _face_track_pause as a module "
            "global — swapping the attribute may no longer reach it")


if __name__ == "__main__":
    unittest.main()
