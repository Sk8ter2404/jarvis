"""Logic tests for skills/weekly_digest_briefing.py.

Covers config clamping + the bobert_companion getattr fallbacks, the ISO-Monday
week label, cluster eligibility (in-band vs lead-window, day-of-week match,
confidence floor, malformed-band defence, ordering), the offer-line composer,
the once-per-week + max-cards throttle in _next_eligible (incl. the legacy
bare-string save format), atomic state persistence (_load_state / _save_state /
_mark_fired) against a temp data dir, state pruning, the digest loader, all four
hard gates (sleep/standby, in-call, away, window-title enumeration), the speech
enqueue (announcer path + pending_speech.json fallback + failure), the
background scheduler loop (each gate's early-continue + the firing path), the
register() entrypoint (enabled/disabled/no-digest/bad-announcer), both
registered actions in every branch, and the __main__ smoke block.

Isolation contract: every fake module lives only for the duration of a test via
a save/restore context manager (sys.modules + parent-package attribute), and the
on-disk state path is redirected to a per-test temp dir torn down afterwards.
The scheduler thread is neutered by the harness; pattern_learning /
bobert_companion / pygetwindow are mocked so nothing real runs, no network, no
LLM, no sleep, no threads. Stdlib unittest + unittest.mock only.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated, no_background_threads


# ─── fake-module save/restore (mirrors test_self_diagnostic.inject_modules) ──
_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules. For dotted names the
    leaf is ALSO set on its already-imported parent package (because
    ``from pkg import leaf`` resolves via ``getattr(parent, leaf)``). Restores
    the previous state — including absence — on exit, so tests stay isolated and
    real modules (numpy etc.) survive untouched. ``obj=None`` forces absence."""
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
                with contextlib.suppress(AttributeError):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


@contextlib.contextmanager
def block_import(*names):
    """Force ``import <name>`` to raise ImportError inside the block, even when
    the real module is installed and already cached. Patches builtins.__import__
    AND detaches any already-imported target from sys.modules for the duration,
    restoring both on exit. Used to exercise a skill's missing-optional-dep
    branch (e.g. pygetwindow absent) deterministically."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved_mod: dict[str, object] = {}
    for name in blocked:
        if name in sys.modules:
            saved_mod[name] = sys.modules.pop(name)
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for name, mod in saved_mod.items():
            sys.modules[name] = mod


def _silence_stdout(testcase):
    """Swallow stdout for the rest of TESTCASE's lifetime. Several failure/fire
    paths in the skill print diagnostics (and a '≥' banner that would crash a
    cp1252 console); tests assert on return values, not prints."""
    cm = contextlib.redirect_stdout(io.StringIO())
    cm.__enter__()
    testcase.addCleanup(cm.__exit__, None, None, None)


class _IsolatedSkillCase(unittest.TestCase):
    """Base case that quarantines sys.modules across every test.

    Loading the skill calls its ``register()``, which does
    ``importlib.import_module("bobert_companion")`` and leaves the REAL monolith
    (plus its transitive imports) cached in ``sys.modules``. Left there, that
    leak can perturb unrelated skill tests later in a full-discovery run. So we
    snapshot ``sys.modules`` before each test and, on teardown, drop any keys the
    test introduced and restore any whose object identity changed — guaranteeing
    a test here never exports module state to its neighbours. Subclasses that
    override setUp MUST call ``super().setUp()`` first."""

    def setUp(self):
        self._sysmod_before = dict(sys.modules)
        self.addCleanup(self._restore_sys_modules)

    def _restore_sys_modules(self):
        before = self._sysmod_before
        for name in list(sys.modules):
            if name not in before:
                del sys.modules[name]
        for name, obj in before.items():
            if sys.modules.get(name) is not obj:
                sys.modules[name] = obj


def _fake_bc(*, sleep=False, standby=False, announce=None,
             announce_raises=False, **config):
    """A bobert_companion stand-in. ``_sleep_mode``/``_standby_mode`` are the
    1-element lists the real module exposes. Config kwargs become module
    attributes the WEEKLY_DIGEST_* getattr reads consult."""
    bc = types.ModuleType("bobert_companion")
    bc._sleep_mode = [sleep]
    bc._standby_mode = [standby]
    if announce is not None or announce_raises:
        def proactive_announce(message, source="skill"):
            if announce_raises:
                raise RuntimeError("announce boom")
            announce(message, source)
            return True
        bc.proactive_announce = proactive_announce
    for k, v in config.items():
        setattr(bc, k, v)
    return bc


def _fake_pattern_learning(digest=None, raises=False, no_loader=False,
                           loader_not_callable=False):
    """A skill_pattern_learning stand-in exposing load_latest_weekly_digest."""
    pl = types.ModuleType("skill_pattern_learning")
    if no_loader:
        return pl
    if loader_not_callable:
        pl.load_latest_weekly_digest = "not-callable"
        return pl

    def load_latest_weekly_digest():
        if raises:
            raise RuntimeError("digest boom")
        return digest
    pl.load_latest_weekly_digest = load_latest_weekly_digest
    return pl


def _fake_face_tracker(monitor=_SENTINEL, last_sample_at=1.0, raises=False,
                       no_snapshot=False):
    """A skill_face_tracker stand-in exposing _snapshot_state."""
    ft = types.ModuleType("skill_face_tracker")
    if no_snapshot:
        return ft

    def _snapshot_state():
        if raises:
            raise RuntimeError("snap boom")
        snap = {"last_sample_at": last_sample_at}
        if monitor is not _SENTINEL:
            snap["current_monitor"] = monitor
        return snap
    ft._snapshot_state = _snapshot_state
    return ft


def _fake_pygetwindow(titles, raises_enum=False, title_raises=False):
    """A pygetwindow stand-in: getAllWindows() -> objects with .title."""
    gw = types.ModuleType("pygetwindow")

    class _Win:
        def __init__(self, t):
            self._t = t

        @property
        def title(self):
            if title_raises:
                raise RuntimeError("title boom")
            return self._t

    def getAllWindows():
        if raises_enum:
            raise RuntimeError("enum boom")
        return [_Win(t) for t in titles]
    gw.getAllWindows = getAllWindows
    return gw


# ════════════════════════════════════════════════════════════════════════════
#  Pure-logic tests (config, week label, eligibility, compose) — no I/O
# ════════════════════════════════════════════════════════════════════════════
class WeeklyDigestBriefingTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "poll_min": 15, "lead_min": 30,
                "conf_floor": 0.5, "max_cards": 3}
        base.update(over)
        return base

    # ── _clamp (pure) ────────────────────────────────────────────────────
    def test_clamp(self):
        c = self.mod._clamp
        self.assertEqual(c(15, 1, 60), 15)
        self.assertEqual(c(0, 1, 120), 1)
        self.assertEqual(c(999, 1, 10), 10)
        self.assertEqual(c(None, 1, 10), 1)

    def test_clamp_float_type_and_uncastable(self):
        c = self.mod._clamp
        # lo is a float → value is cast to float, bounds respected.
        self.assertEqual(c("0.7", 0.0, 1.0), 0.7)
        self.assertEqual(c(2.5, 0.0, 1.0), 1.0)
        self.assertEqual(c(-3, 0.0, 1.0), 0.0)
        # Uncastable → returns lo (the except branch).
        self.assertEqual(c(object(), 0.0, 1.0), 0.0)

    # ── _read_config ─────────────────────────────────────────────────────
    def test_read_config_no_bc_uses_defaults(self):
        # bobert_companion absent → every default, clamped.
        with inject_modules(bobert_companion=None):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": True, "poll_min": 15, "lead_min": 30,
                               "conf_floor": 0.5, "max_cards": 3})

    def test_read_config_reads_and_clamps_bc_values(self):
        bc = _fake_bc(WEEKLY_DIGEST_ENABLED=False,
                      WEEKLY_DIGEST_POLL_MINUTES=999,     # clamp → 60
                      WEEKLY_DIGEST_LEAD_MINUTES=0,       # clamp → 1
                      WEEKLY_DIGEST_CONFIDENCE_MIN=2.0,   # clamp → 1.0
                      WEEKLY_DIGEST_MAX_CARDS=-5)         # clamp → 1
        with inject_modules(bobert_companion=bc):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": False, "poll_min": 60, "lead_min": 1,
                               "conf_floor": 1.0, "max_cards": 1})

    def test_read_config_import_error_uses_defaults(self):
        # import_module raising (not just absent) hits the bc=None branch.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("nope")):
            cfg = self.mod._read_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["poll_min"], 15)

    # ── _week_label (pure) ───────────────────────────────────────────────
    def test_week_label_is_monday(self):
        ts = time.mktime(datetime.datetime(2026, 6, 3, 10, 0).timetuple())
        self.assertEqual(self.mod._week_label(ts), "2026-06-01")
        ts_mon = time.mktime(datetime.datetime(2026, 6, 1, 0, 0).timetuple())
        self.assertEqual(self.mod._week_label(ts_mon), "2026-06-01")

    def test_week_label_default_now(self):
        # Called with no arg → uses time.time(); just assert it's a valid Monday.
        lbl = self.mod._week_label()
        d = datetime.date.fromisoformat(lbl)
        self.assertEqual(d.weekday(), 0)

    # ── _eligible_clusters ───────────────────────────────────────────────
    def test_eligible_in_band(self):
        now = time.localtime()
        digest = {"clusters": [
            {"key": "c1", "confidence": 0.9, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2,
             "offer": "Netflix?"},
        ]}
        out = self.mod._eligible_clusters(digest, self._cfg())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["key"], "c1")

    def test_eligible_wrong_weekday_excluded(self):
        now = time.localtime()
        other_dow = (now.tm_wday + 1) % 7
        digest = {"clusters": [
            {"key": "c1", "confidence": 0.9, "dow": other_dow,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2},
        ]}
        self.assertEqual(self.mod._eligible_clusters(digest, self._cfg()), [])

    def test_eligible_below_confidence_excluded(self):
        now = time.localtime()
        digest = {"clusters": [
            {"key": "c1", "confidence": 0.1, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2},
        ]}
        self.assertEqual(self.mod._eligible_clusters(digest, self._cfg(conf_floor=0.5)), [])

    def test_eligible_lead_window(self):
        now = time.localtime()
        cur_min = now.tm_hour * 60 + now.tm_min
        start_hour = (cur_min + 20) // 60
        digest = {"clusters": [
            {"key": "soon", "confidence": 0.8, "dow": now.tm_wday,
             "hour_start": start_hour, "hour_end": start_hour + 2, "offer": "x"},
        ]}
        out = self.mod._eligible_clusters(digest, self._cfg(lead_min=120))
        self.assertEqual([c["key"] for c in out], ["soon"])

    def test_eligible_empty_digest(self):
        self.assertEqual(self.mod._eligible_clusters({}, self._cfg()), [])

    def test_eligible_no_clusters_key(self):
        # digest truthy but no "clusters" → the `if not clusters` guard.
        self.assertEqual(self.mod._eligible_clusters({"week_start": "x"}, self._cfg()), [])

    def test_eligible_non_dict_and_missing_hour_skipped(self):
        now = time.localtime()
        digest = {"clusters": [
            "not-a-dict",                                   # skipped (not dict)
            {"key": "nohour", "confidence": 0.9, "dow": now.tm_wday},  # hour_start<0
        ]}
        self.assertEqual(self.mod._eligible_clusters(digest, self._cfg()), [])

    def test_eligible_malformed_hour_end_defends_to_two_hour_band(self):
        # hour_end non-int → except → hour_start+2; and hour_end<=hour_start path.
        now = time.localtime()
        digest = {"clusters": [
            {"key": "bad_end", "confidence": 0.9, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": "garbage", "offer": "o1"},
            {"key": "rev_end", "confidence": 0.9, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour - 5, "offer": "o2"},
        ]}
        out = self.mod._eligible_clusters(digest, self._cfg())
        # Both get a canonical 2h band starting at the current hour → in-band now.
        self.assertEqual(sorted(c["key"] for c in out), ["bad_end", "rev_end"])

    def test_eligible_ordering_in_band_before_lead_then_confidence(self):
        now = time.localtime()
        # The lead-window cluster must start at a FUTURE hour boundary so it is
        # upcoming (never in-band) regardless of the current minute. The next
        # hour is 1..60 min away — always within lead_min=120 yet never in-band.
        # (The old (cur_min+20)//60 floor landed on the *current* hour whenever
        # tm_min < 40, making this cluster in-band and, since it has the highest
        # confidence, sorting it first — flipping the ordering assertion.)
        lead_hour = now.tm_hour + 1
        digest = {"clusters": [
            {"key": "lead", "confidence": 0.95, "dow": now.tm_wday,
             "hour_start": lead_hour, "hour_end": lead_hour + 2, "offer": "L"},
            {"key": "band_lo", "confidence": 0.6, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2, "offer": "B1"},
            {"key": "band_hi", "confidence": 0.9, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2, "offer": "B2"},
        ]}
        out = [c["key"] for c in self.mod._eligible_clusters(digest, self._cfg(lead_min=120))]
        # In-band first, ordered by confidence desc; lead-window last.
        self.assertEqual(out[0], "band_hi")
        self.assertEqual(out[1], "band_lo")
        self.assertEqual(out[2], "lead")

    def test_eligible_past_band_not_selected(self):
        # A band that already ended today is neither in-band nor upcoming.
        now = time.localtime()
        if now.tm_hour < 3:
            self.skipTest("too early in the day to have a 'past' 2h band")
        digest = {"clusters": [
            {"key": "past", "confidence": 0.9, "dow": now.tm_wday,
             "hour_start": 0, "hour_end": 2, "offer": "x"},
        ]}
        self.assertEqual(self.mod._eligible_clusters(digest, self._cfg()), [])

    # ── _compose_line ────────────────────────────────────────────────────
    def test_compose_line_prefers_offer(self):
        self.assertEqual(self.mod._compose_line({"offer": "Shall I queue Netflix, sir?"}),
                         "Shall I queue Netflix, sir?")

    def test_compose_line_fallback_label(self):
        out = self.mod._compose_line({"label": "Friday Netflix"})
        self.assertIn("Friday Netflix", out)
        self.assertIn("Shall I proceed", out)

    def test_compose_line_empty(self):
        self.assertEqual(self.mod._compose_line({}), "")

    def test_compose_line_whitespace_offer_falls_through_to_label(self):
        out = self.mod._compose_line({"offer": "   ", "label": "Gym time"})
        self.assertIn("Gym time", out)


# ════════════════════════════════════════════════════════════════════════════
#  _next_eligible throttle + max-cards (mostly mocked helpers)
# ════════════════════════════════════════════════════════════════════════════
class NextEligibleTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "poll_min": 15, "lead_min": 30,
                "conf_floor": 0.5, "max_cards": 3}
        base.update(over)
        return base

    def test_next_eligible_skips_already_fired_this_week(self):
        week = self.mod._week_label()
        cluster = {"key": "k1", "offer": "Netflix?"}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]), \
             mock.patch.object(self.mod, "_load_state",
                               return_value={"k1": {"week": week, "day": "x"}}):
            line, c = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "")
        self.assertEqual(c, {})

    def test_next_eligible_skips_legacy_bare_string_save(self):
        # Backward-compat: a pre-2026-05 save stored the week as a bare string.
        week = self.mod._week_label()
        cluster = {"key": "k1", "offer": "Netflix?"}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]), \
             mock.patch.object(self.mod, "_load_state", return_value={"k1": week}):
            line, c = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "")

    def test_next_eligible_fires_when_prev_week_differs(self):
        cluster = {"key": "k1", "offer": "Netflix?"}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]), \
             mock.patch.object(self.mod, "_load_state",
                               return_value={"k1": {"week": "1999-01-04", "day": "x"}}):
            line, c = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "Netflix?")
        self.assertEqual(c["key"], "k1")

    def test_next_eligible_respects_max_cards(self):
        today = time.strftime("%Y-%m-%d", time.localtime())
        cluster = {"key": "k2", "offer": "Netflix?"}
        state = {f"c{i}": {"week": "old", "day": today} for i in range(3)}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg(max_cards=3)), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]), \
             mock.patch.object(self.mod, "_load_state", return_value=state):
            line, c = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "")

    def test_next_eligible_bypass_returns_line(self):
        cluster = {"key": "k3", "offer": "Shall I queue Netflix, sir?"}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]):
            line, c = self.mod._next_eligible(bypass_throttle=True)
        self.assertEqual(line, "Shall I queue Netflix, sir?")
        self.assertEqual(c["key"], "k3")

    def test_next_eligible_skips_keyless_then_lineless_clusters(self):
        # First candidate has no key (skipped), second has key but empty line
        # (skipped), third fires. Exercises the `continue` arms in the loop.
        c_nokey = {"offer": "x"}
        c_noline = {"key": "kk", "offer": "", "label": ""}
        c_ok = {"key": "kok", "offer": "Real offer, sir."}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [1]}), \
             mock.patch.object(self.mod, "_eligible_clusters",
                               return_value=[c_nokey, c_noline, c_ok]):
            line, c = self.mod._next_eligible(bypass_throttle=True)
        self.assertEqual(line, "Real offer, sir.")
        self.assertEqual(c["key"], "kok")

    def test_next_eligible_no_digest(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={}):
            self.assertEqual(self.mod._next_eligible(bypass_throttle=True), ("", {}))

    def test_next_eligible_no_candidates(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [1]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[]):
            self.assertEqual(self.mod._next_eligible(bypass_throttle=True), ("", {}))

    def test_next_eligible_all_already_fired_returns_empty(self):
        # candidates exist but every one is throttled this week → loop exhausts.
        week = self.mod._week_label()
        cluster = {"key": "konly", "offer": "Netflix?"}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]), \
             mock.patch.object(self.mod, "_load_state",
                               return_value={"konly": {"week": week, "day": "x"}}):
            self.assertEqual(self.mod._next_eligible(bypass_throttle=False), ("", {}))


# ════════════════════════════════════════════════════════════════════════════
#  State persistence against a real temp dir (_load_state/_save_state/_mark_fired)
# ════════════════════════════════════════════════════════════════════════════
class StatePersistenceTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")
        _silence_stdout(self)
        self.tmp = tempfile.mkdtemp(prefix="weekdigest_state_")
        self._orig_data = self.mod._DATA_DIR
        self._orig_state = self.mod._STATE_FILE
        self.mod._DATA_DIR = self.tmp
        self.mod._STATE_FILE = os.path.join(self.tmp, "weekly_digest_briefing_state.json")

    def tearDown(self):
        self.mod._DATA_DIR = self._orig_data
        self.mod._STATE_FILE = self._orig_state
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_state_missing_file_returns_empty(self):
        self.assertEqual(self.mod._load_state(), {})

    def test_save_then_load_roundtrip(self):
        self.mod._save_state({"k": {"week": "2026-06-01", "day": "2026-06-01"}})
        self.assertTrue(os.path.exists(self.mod._STATE_FILE))
        self.assertEqual(self.mod._load_state(),
                         {"k": {"week": "2026-06-01", "day": "2026-06-01"}})
        # No stray .tmp files left behind after a successful replace.
        leftovers = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_load_state_corrupt_json_returns_empty(self):
        with open(self.mod._STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertEqual(self.mod._load_state(), {})

    def test_load_state_non_dict_top_level_returns_empty(self):
        with open(self.mod._STATE_FILE, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.mod._load_state(), {})

    def test_save_state_failure_is_swallowed(self):
        # mkstemp raising → the except prints and returns; no exception escapes.
        with mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("disk full")):
            self.mod._save_state({"k": "v"})   # must not raise
        self.assertFalse(os.path.exists(self.mod._STATE_FILE))

    def test_save_state_cleans_up_fd_and_tmp_on_fdopen_failure(self):
        # fdopen raising AFTER mkstemp keeps fd>=0 and tmp set, exercising BOTH
        # finally cleanup arms (os.close(fd) + os.unlink(tmp)). The real temp
        # file must be gone afterward and no descriptor leaked.
        real_fdopen = self.mod.os.fdopen
        with mock.patch.object(self.mod.os, "fdopen",
                               side_effect=OSError("fdopen boom")):
            self.mod._save_state({"k": "v"})   # must not raise
        self.assertIs(self.mod.os.fdopen, real_fdopen)   # patch unwound
        # The .tmp file created by mkstemp was unlinked by the cleanup arm.
        leftovers = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertFalse(os.path.exists(self.mod._STATE_FILE))

    def test_save_state_swallows_errors_in_cleanup_arms(self):
        # Force cleanup itself to fail: fdopen raises (enter cleanup), then both
        # os.close(fd) AND os.unlink(tmp) raise → the nested `except: pass`
        # swallow-arms execute. _save_state must still not raise. We capture the
        # real fd/tmp from mkstemp and clean them up ourselves to avoid a leak.
        real_close = os.close
        real_unlink = os.unlink
        captured = {}
        real_mkstemp = self.mod.tempfile.mkstemp

        def spy_mkstemp(*a, **k):
            fd, path = real_mkstemp(*a, **k)
            captured["fd"], captured["path"] = fd, path
            return fd, path

        with mock.patch.object(self.mod.tempfile, "mkstemp", side_effect=spy_mkstemp), \
             mock.patch.object(self.mod.os, "fdopen", side_effect=OSError("fdopen")), \
             mock.patch.object(self.mod.os, "close", side_effect=OSError("close")), \
             mock.patch.object(self.mod.os, "unlink", side_effect=OSError("unlink")):
            self.mod._save_state({"k": "v"})   # must not raise
        # Tidy up the descriptor + temp file the swallowed cleanup left behind.
        with contextlib.suppress(OSError):
            real_close(captured["fd"])
        with contextlib.suppress(OSError, KeyError):
            real_unlink(captured["path"])

    def test_save_state_tmp_cleanup_when_replace_fails(self):
        # os.replace raising AFTER fdopen succeeds: fd is already -1 (ownership
        # passed to fdopen), so only the tmp-unlink arm runs. Temp must be gone.
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("replace boom")):
            self.mod._save_state({"k": "v"})   # must not raise
        leftovers = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_ensure_data_dir_swallows_errors(self):
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("perm")):
            self.mod._ensure_data_dir()   # must not raise

    def test_mark_fired_writes_entry(self):
        cluster = {"key": "fri_netflix"}
        frozen = 1717200000.0   # 2024-05-31 (UTC) — deterministic
        expected_week = self.mod._week_label(frozen)
        with mock.patch.object(self.mod.time, "time", return_value=frozen):
            self.mod._mark_fired(cluster)
        state = self.mod._load_state()
        self.assertIn("fri_netflix", state)
        entry = state["fri_netflix"]
        self.assertEqual(entry["week"], expected_week)
        self.assertEqual(entry["ts"], frozen)
        self.assertIn("day", entry)

    def test_mark_fired_keyless_is_noop(self):
        self.mod._mark_fired({"offer": "no key"})
        self.assertEqual(self.mod._load_state(), {})
        self.assertFalse(os.path.exists(self.mod._STATE_FILE))

    def test_mark_fired_prunes_stale_before_saving(self):
        old_week = (datetime.date.fromisoformat(self.mod._week_label())
                    - datetime.timedelta(weeks=20)).isoformat()
        self.mod._save_state({"stale": old_week})
        self.mod._mark_fired({"key": "fresh"})
        state = self.mod._load_state()
        self.assertIn("fresh", state)
        self.assertNotIn("stale", state)

    # ── _prune_state ─────────────────────────────────────────────────────
    def test_prune_state_drops_old_weeks(self):
        old_week = (datetime.date.fromisoformat(self.mod._week_label())
                    - datetime.timedelta(weeks=20)).isoformat()
        state = {"stale": old_week, "fresh": self.mod._week_label()}
        self.mod._prune_state(state)
        self.assertNotIn("stale", state)
        self.assertIn("fresh", state)

    def test_prune_state_keeps_dict_values_and_recent(self):
        # Dict-valued entries (current format) aren't bare strings → untouched.
        recent = (datetime.date.fromisoformat(self.mod._week_label())
                  - datetime.timedelta(weeks=2)).isoformat()
        state = {"d": {"week": "old"}, "recent_str": recent}
        self.mod._prune_state(state)
        self.assertIn("d", state)
        self.assertIn("recent_str", state)

    def test_prune_state_swallows_errors(self):
        # _week_label raising inside prune → except branch, no crash.
        with mock.patch.object(self.mod, "_week_label",
                               side_effect=RuntimeError("clock")):
            self.mod._prune_state({"x": "2000-01-03"})   # must not raise


# ════════════════════════════════════════════════════════════════════════════
#  Digest loader (_load_digest) via injected skill_pattern_learning
# ════════════════════════════════════════════════════════════════════════════
class DigestLoaderTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")

    def test_load_digest_from_cached_module(self):
        pl = _fake_pattern_learning(digest={"clusters": [{"key": "a"}]})
        with inject_modules(skill_pattern_learning=pl):
            out = self.mod._load_digest()
        self.assertEqual(out, {"clusters": [{"key": "a"}]})

    def test_load_digest_import_when_not_cached(self):
        pl = _fake_pattern_learning(digest={"clusters": []})
        # Not in sys.modules → falls to importlib.import_module; patch that.
        with inject_modules(skill_pattern_learning=None), \
             mock.patch.object(self.mod.importlib, "import_module", return_value=pl):
            out = self.mod._load_digest()
        self.assertEqual(out, {"clusters": []})

    def test_load_digest_import_failure_returns_empty(self):
        with inject_modules(skill_pattern_learning=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no pl")):
            self.assertEqual(self.mod._load_digest(), {})

    def test_load_digest_no_loader_attr_returns_empty(self):
        pl = _fake_pattern_learning(no_loader=True)
        with inject_modules(skill_pattern_learning=pl):
            self.assertEqual(self.mod._load_digest(), {})

    def test_load_digest_loader_not_callable_returns_empty(self):
        pl = _fake_pattern_learning(loader_not_callable=True)
        with inject_modules(skill_pattern_learning=pl):
            self.assertEqual(self.mod._load_digest(), {})

    def test_load_digest_loader_raises_returns_empty(self):
        pl = _fake_pattern_learning(raises=True)
        with inject_modules(skill_pattern_learning=pl):
            self.assertEqual(self.mod._load_digest(), {})

    def test_load_digest_non_dict_result_returns_empty(self):
        pl = _fake_pattern_learning(digest=["not", "a", "dict"])
        with inject_modules(skill_pattern_learning=pl):
            self.assertEqual(self.mod._load_digest(), {})


# ════════════════════════════════════════════════════════════════════════════
#  Hard gates: sleep/standby, in-call (window titles), user-at-desk
# ════════════════════════════════════════════════════════════════════════════
class HardGateTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")

    # ── _is_sleep_or_standby ─────────────────────────────────────────────
    def test_sleep_gate_no_bc(self):
        with inject_modules(bobert_companion=None):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_sleep_gate_sleep_active(self):
        with inject_modules(bobert_companion=_fake_bc(sleep=True)):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_sleep_gate_standby_active(self):
        with inject_modules(bobert_companion=_fake_bc(standby=True)):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_sleep_gate_neither_active(self):
        with inject_modules(bobert_companion=_fake_bc()):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_sleep_gate_attribute_error_returns_false(self):
        # bc present but missing _sleep_mode → getattr raises → except → False.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            self.assertFalse(self.mod._is_sleep_or_standby())

    # ── _all_window_titles ───────────────────────────────────────────────
    def test_window_titles_no_pygetwindow(self):
        # pygetwindow is installed on the dev box; block the import so the
        # missing-dep branch (returns []) is what executes.
        with block_import("pygetwindow"):
            self.assertEqual(self.mod._all_window_titles(), [])

    def test_window_titles_collected_and_stripped(self):
        gw = _fake_pygetwindow(["Spotify", "   ", "", "Code"])
        with inject_modules(pygetwindow=gw):
            titles = self.mod._all_window_titles()
        self.assertEqual(titles, ["Spotify", "Code"])

    def test_window_titles_enum_failure_returns_empty(self):
        gw = _fake_pygetwindow([], raises_enum=True)
        with inject_modules(pygetwindow=gw):
            self.assertEqual(self.mod._all_window_titles(), [])

    # ── _is_in_call ──────────────────────────────────────────────────────
    def test_in_call_gate(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=["Zoom Meeting"]):
            self.assertTrue(self.mod._is_in_call())
        with mock.patch.object(self.mod, "_all_window_titles", return_value=["Spotify"]):
            self.assertFalse(self.mod._is_in_call())

    def test_in_call_no_titles(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=[]):
            self.assertFalse(self.mod._is_in_call())

    def test_in_call_matches_each_hint_variant(self):
        for title in ("Standup | Microsoft Teams Meeting", "Webex Meetings",
                      "Google Meet - room", "Discord Call with friends"):
            with mock.patch.object(self.mod, "_all_window_titles", return_value=[title]):
                self.assertTrue(self.mod._is_in_call(), title)

    def test_in_call_integration_through_pygetwindow(self):
        # Exercise the real _all_window_titles path feeding _is_in_call.
        gw = _fake_pygetwindow(["Daily Sync | Microsoft Teams Meeting"])
        with inject_modules(pygetwindow=gw):
            self.assertTrue(self.mod._is_in_call())

    # ── _user_at_desk ────────────────────────────────────────────────────
    def test_user_at_desk_no_module(self):
        with inject_modules(skill_face_tracker=None):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_no_snapshot_fn(self):
        with inject_modules(skill_face_tracker=_fake_face_tracker(no_snapshot=True)):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_snapshot_raises(self):
        with inject_modules(skill_face_tracker=_fake_face_tracker(raises=True)):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_no_sample_yet(self):
        ft = _fake_face_tracker(monitor="left", last_sample_at=None)
        with inject_modules(skill_face_tracker=ft):
            self.assertIsNone(self.mod._user_at_desk())

    def test_user_at_desk_present_variants_true(self):
        for mon in ("left", "right", "middle_or_top"):
            ft = _fake_face_tracker(monitor=mon)
            with inject_modules(skill_face_tracker=ft):
                self.assertIs(self.mod._user_at_desk(), True, mon)

    def test_user_at_desk_away_false(self):
        ft = _fake_face_tracker(monitor="away")
        with inject_modules(skill_face_tracker=ft):
            self.assertIs(self.mod._user_at_desk(), False)

    def test_user_at_desk_unknown_monitor_none(self):
        ft = _fake_face_tracker(monitor="somewhere_else")
        with inject_modules(skill_face_tracker=ft):
            self.assertIsNone(self.mod._user_at_desk())


# ════════════════════════════════════════════════════════════════════════════
#  _enqueue_speech: announcer path + pending_speech.json fallback + failure
# ════════════════════════════════════════════════════════════════════════════
class EnqueueSpeechTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")
        _silence_stdout(self)
        self.tmp = tempfile.mkdtemp(prefix="weekdigest_queue_")
        self._orig_proj = self.mod._PROJECT_DIR
        self.mod._PROJECT_DIR = self.tmp
        self.queue_path = os.path.join(self.tmp, "pending_speech.json")

    def tearDown(self):
        self.mod._PROJECT_DIR = self._orig_proj
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_enqueue_uses_announcer_when_available(self):
        seen = {}

        def announce(msg, src):
            seen["msg"] = msg
            seen["src"] = src
        bc = _fake_bc(announce=announce)
        with inject_modules(bobert_companion=bc):
            ok = self.mod._enqueue_speech("Hello sir")
        self.assertTrue(ok)
        self.assertEqual(seen["msg"], "Hello sir")
        self.assertEqual(seen["src"], "weekly_digest_briefing")
        # Announcer path: no fallback file written.
        self.assertFalse(os.path.exists(self.queue_path))

    def test_enqueue_falls_back_to_file_when_no_announcer(self):
        bc = types.ModuleType("bobert_companion")   # no proactive_announce attr
        with inject_modules(bobert_companion=bc):
            ok = self.mod._enqueue_speech("Queued line")
        self.assertTrue(ok)
        with open(self.queue_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "Queued line")
        self.assertIn("ts", data[0])

    def test_enqueue_fallback_when_bc_import_fails(self):
        with inject_modules(bobert_companion=None), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")):
            ok = self.mod._enqueue_speech("Line A")
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(self.queue_path))

    def test_enqueue_fallback_when_announcer_raises(self):
        bc = _fake_bc(announce_raises=True)
        with inject_modules(bobert_companion=bc):
            ok = self.mod._enqueue_speech("Line B")
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(self.queue_path))

    def test_enqueue_appends_to_existing_queue(self):
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            self.mod._enqueue_speech("new")
        with open(self.queue_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new"])

    def test_enqueue_corrupt_existing_queue_is_reset(self):
        with open(self.queue_path, "w", encoding="utf-8") as f:
            f.write("{garbage")
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc):
            ok = self.mod._enqueue_speech("fresh")
        self.assertTrue(ok)
        with open(self.queue_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["fresh"])

    def test_enqueue_write_failure_returns_false(self):
        # mkstemp raising in the fallback writer → outer except → False.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.tempfile, "mkstemp",
                               side_effect=OSError("disk full")):
            ok = self.mod._enqueue_speech("doomed")
        self.assertFalse(ok)

    def test_enqueue_cleans_up_fd_and_tmp_on_fdopen_failure(self):
        # fdopen raising AFTER mkstemp keeps fd>=0 and tmp set → both inner
        # cleanup arms run (close fd + unlink tmp), then re-raise → False. No
        # leaked descriptor and no stray .tmp file in the project dir.
        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.os, "fdopen",
                               side_effect=OSError("fdopen boom")):
            ok = self.mod._enqueue_speech("doomed")
        self.assertFalse(ok)
        leftovers = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertFalse(os.path.exists(self.queue_path))

    def test_enqueue_swallows_errors_in_cleanup_arms(self):
        # fdopen raises (enter inner cleanup), then os.close AND os.unlink raise
        # → the nested `except: pass` swallow-arms run; outer except → False.
        real_close = os.close
        real_unlink = os.unlink
        captured = {}
        real_mkstemp = self.mod.tempfile.mkstemp

        def spy_mkstemp(*a, **k):
            fd, path = real_mkstemp(*a, **k)
            captured["fd"], captured["path"] = fd, path
            return fd, path

        bc = types.ModuleType("bobert_companion")
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.tempfile, "mkstemp", side_effect=spy_mkstemp), \
             mock.patch.object(self.mod.os, "fdopen", side_effect=OSError("fdopen")), \
             mock.patch.object(self.mod.os, "close", side_effect=OSError("close")), \
             mock.patch.object(self.mod.os, "unlink", side_effect=OSError("unlink")):
            ok = self.mod._enqueue_speech("doomed")
        self.assertFalse(ok)
        with contextlib.suppress(OSError):
            real_close(captured["fd"])
        with contextlib.suppress(OSError, KeyError):
            real_unlink(captured["path"])


# ════════════════════════════════════════════════════════════════════════════
#  Scheduler loop: each gate's early-continue + the firing path
# ════════════════════════════════════════════════════════════════════════════
class _StopLoop(BaseException):
    """Raised from a patched time.sleep to break out of the infinite loop.

    Subclasses BaseException (NOT Exception) on purpose: _scheduler_loop wraps
    its body in ``except Exception``, so an Exception-derived sentinel raised by
    a sleep INSIDE that try would be swallowed and the loop would spin forever.
    BaseException slips past the handler and unwinds cleanly to assertRaises."""


class SchedulerLoopTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")
        _silence_stdout(self)

    def _run_one_iteration(self, *, sleeps_before_stop=2):
        """Drive _scheduler_loop for a single pass: the first time.sleep is the
        INITIAL_DELAY_SECONDS, and we raise _StopLoop on the Nth sleep call so
        exactly one loop body executes, then control returns."""
        calls = {"n": 0}

        def fake_sleep(_secs):
            calls["n"] += 1
            if calls["n"] >= sleeps_before_stop:
                raise _StopLoop
        with mock.patch.object(self.mod.time, "sleep", side_effect=fake_sleep):
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        return calls["n"]

    def test_loop_disabled_continues_without_work(self):
        cfg = {"enabled": False, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_next_eligible") as nxt:
            # Stop on the 3rd sleep so the gate's continue-sleep (2nd) RUNS and
            # the `continue` after it executes before we break on the next pass.
            self._run_one_iteration(sleeps_before_stop=3)
        nxt.assert_not_called()

    def test_loop_sleep_gate_skips(self):
        cfg = {"enabled": True, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible") as nxt:
            self._run_one_iteration(sleeps_before_stop=3)
        nxt.assert_not_called()

    def test_loop_in_call_gate_skips(self):
        cfg = {"enabled": True, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible") as nxt:
            self._run_one_iteration(sleeps_before_stop=3)
        nxt.assert_not_called()

    def test_loop_user_away_gate_skips(self):
        cfg = {"enabled": True, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=False), \
             mock.patch.object(self.mod, "_next_eligible") as nxt:
            self._run_one_iteration(sleeps_before_stop=3)
        nxt.assert_not_called()

    def test_loop_fires_and_marks(self):
        cfg = {"enabled": True, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Fire this, sir.", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=True) as enq, \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            self._run_one_iteration()
        enq.assert_called_once_with("Fire this, sir.")
        mark.assert_called_once_with({"key": "k"})

    def test_loop_fire_but_enqueue_fails_does_not_mark(self):
        cfg = {"enabled": True, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Line", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=False), \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            self._run_one_iteration()
        mark.assert_not_called()

    def test_loop_no_line_skips_enqueue(self):
        cfg = {"enabled": True, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("", {})), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self._run_one_iteration()
        enq.assert_not_called()

    def test_loop_body_exception_is_logged_not_fatal(self):
        # An exception inside the try is caught by logging.exception; the loop
        # then reaches the trailing sleep, where we stop it.
        cfg = {"enabled": True, "poll_min": 15, "lead_min": 30,
               "conf_floor": 0.5, "max_cards": 3}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_is_sleep_or_standby",
                               side_effect=RuntimeError("gate boom")), \
             mock.patch.object(self.mod.logging, "exception") as logexc:
            # initial-delay sleep (1) + trailing sleep (2) → stop on 2nd.
            self._run_one_iteration(sleeps_before_stop=2)
        logexc.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════
#  Registered actions: weekly_digest_now / weekly_digest_status, all branches
# ════════════════════════════════════════════════════════════════════════════
class ActionTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "poll_min": 15, "lead_min": 30,
                "conf_floor": 0.5, "max_cards": 3}
        base.update(over)
        return base

    def test_actions_registered(self):
        self.assertIn("weekly_digest_now", self.actions)
        self.assertIn("weekly_digest_status", self.actions)

    # ── weekly_digest_now ────────────────────────────────────────────────
    def test_action_now_suppressed_sleep(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True):
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("sleep", out.lower())

    def test_action_now_suppressed_in_call(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True):
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("call", out.lower())

    def test_action_now_suppressed_away(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=False):
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("desk", out.lower())

    def test_action_now_no_match(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("", {})):
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("No weekly habit matches", out)

    def test_action_now_fires(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Netflix, sir?", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=True), \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.actions["weekly_digest_now"]("")
        self.assertEqual(out, "Netflix, sir?")
        mark.assert_called_once()

    def test_action_now_enqueue_failure_message(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Netflix, sir?", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=False), \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("Could not enqueue", out)
        self.assertIn("Netflix, sir?", out)
        mark.assert_not_called()

    def test_action_now_user_at_desk_none_does_not_suppress(self):
        # _user_at_desk() is None (unknown) → NOT `is False` → proceeds.
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=None), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("", {})):
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("No weekly habit matches", out)

    # ── weekly_digest_status ─────────────────────────────────────────────
    def test_action_status(self):
        digest = {"clusters": [{}, {}], "computed_at": time.time()}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value=digest), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[]), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["weekly_digest_status"]("")
        self.assertIn("2 clusters", out)
        self.assertIn("eligible right now", out)
        self.assertIn("0 cards surfaced today", out)

    def test_action_status_disabled(self):
        with mock.patch.object(self.mod, "_read_config",
                               return_value=self._cfg(enabled=False)), \
             mock.patch.object(self.mod, "_load_digest", return_value={}), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["weekly_digest_status"]("")
        self.assertIn("disabled in config", out)
        self.assertIn("no weekly digest cached yet", out)

    def test_action_status_no_digest_branch(self):
        # digest falsy → clusters/candidates default empty, "no digest" line.
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={}), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["weekly_digest_status"]("")
        self.assertIn("no weekly digest cached yet", out)
        self.assertIn("0 eligible right now", out)

    def test_action_status_reports_all_suppressors_and_age(self):
        # computed_at in the past → age line; all three suppressors active.
        digest = {"clusters": [{}], "computed_at": time.time() - 7200.0}
        today = time.strftime("%Y-%m-%d", time.localtime())
        state = {"a": {"day": today}, "b": {"day": today}}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value=digest), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[{}]), \
             mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=False):
            out = self.actions["weekly_digest_status"]("")
        self.assertIn("1 clusters", out)
        self.assertIn("h old", out)
        self.assertIn("2 cards surfaced today", out)
        self.assertIn("sleep/standby active", out)
        self.assertIn("in a call", out)
        self.assertIn("user away", out)

    def test_action_status_singular_card_grammar(self):
        digest = {"clusters": [], "computed_at": 0.0}
        today = time.strftime("%Y-%m-%d", time.localtime())
        state = {"only": {"day": today}}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value=digest), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[]), \
             mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["weekly_digest_status"]("")
        self.assertIn("1 card surfaced today", out)   # singular, no trailing 's'


# ════════════════════════════════════════════════════════════════════════════
#  register(): enabled / disabled / no-digest / bad-announcer entrypoints
# ════════════════════════════════════════════════════════════════════════════
class RegisterTests(_IsolatedSkillCase):
    def setUp(self):
        super().setUp()
        # Load WITHOUT auto-register so each test drives register() explicitly
        # under controlled module injection and a neutered Thread.start.
        self.mod, _ = load_skill_isolated("weekly_digest_briefing", register=False)
        # register() prints a '≥' banner that crashes a cp1252 stdout; swallow
        # all register-time prints for the duration of every test here.
        _silence_stdout(self)

    def _cfg(self, **over):
        base = {"enabled": True, "poll_min": 15, "lead_min": 30,
                "conf_floor": 0.5, "max_cards": 3}
        base.update(over)
        return base

    def test_register_disabled_no_thread(self):
        actions = {}
        cfg = self._cfg(enabled=False)
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_ensure_data_dir"), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        # Actions still registered, but scheduler thread NOT constructed/started.
        self.assertIn("weekly_digest_now", actions)
        self.assertIn("weekly_digest_status", actions)
        Thread.assert_not_called()

    def test_register_enabled_with_digest_starts_thread(self):
        actions = {}
        digest = {"clusters": [{"key": "a"}], "computed_at": time.time()}
        bc = _fake_bc(announce=lambda *a: None)
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_ensure_data_dir"), \
             mock.patch.object(self.mod, "_load_digest", return_value=digest), \
             inject_modules(bobert_companion=bc), \
             no_background_threads(), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        Thread.assert_called_once()
        # daemon thread, started.
        _, kwargs = Thread.call_args
        self.assertTrue(kwargs.get("daemon"))
        Thread.return_value.start.assert_called_once()

    def test_register_enabled_no_digest_warns_but_starts(self):
        actions = {}
        bc = _fake_bc(announce=lambda *a: None)
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_ensure_data_dir"), \
             mock.patch.object(self.mod, "_load_digest", return_value={}), \
             inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)
        Thread.assert_called_once()

    def test_register_announcer_not_callable_warns(self):
        actions = {}
        digest = {"clusters": [], "computed_at": 0.0}
        bc = types.ModuleType("bobert_companion")   # no proactive_announce
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_ensure_data_dir"), \
             mock.patch.object(self.mod, "_load_digest", return_value=digest), \
             inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)   # must not raise
        Thread.assert_called_once()

    def test_register_bc_import_failure_warns(self):
        actions = {}
        digest = {"clusters": [], "computed_at": 0.0}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_ensure_data_dir"), \
             mock.patch.object(self.mod, "_load_digest", return_value=digest), \
             mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod.register(actions)   # must not raise
        Thread.assert_called_once()

    def test_register_via_harness_threads_neutered(self):
        # Smoke: full register() through the harness (threads neutered) with a
        # digest present, asserting both actions land and nothing real spawns.
        _, actions = load_skill_isolated("weekly_digest_briefing")
        self.assertIn("weekly_digest_now", actions)
        self.assertIn("weekly_digest_status", actions)


# ════════════════════════════════════════════════════════════════════════════
#  __main__ smoke block (runpy with injected fakes)
# ════════════════════════════════════════════════════════════════════════════
class MainBlockTests(_IsolatedSkillCase):
    def test_main_block_runs_offline(self):
        # Execute the module as __main__ with fakes so the smoke block's
        # _read_config/_load_digest/_eligible_clusters/_next_eligible all run
        # against deterministic data and never touch real deps.
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "skills", "weekly_digest_briefing.py")
        now = time.localtime()
        digest = {"week_start": "2026-06-01", "clusters": [
            {"key": "m1", "confidence": 0.9, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2,
             "offer": "Smoke offer, sir."},
        ]}
        bc = _fake_bc()
        pl = _fake_pattern_learning(digest=digest)
        buf = []
        with inject_modules(bobert_companion=bc, skill_pattern_learning=pl), \
             mock.patch("builtins.print", side_effect=lambda *a, **k: buf.append(a)), \
             no_background_threads():
            runpy.run_path(path, run_name="__main__")
        # The smoke block printed config + digest + eligible lines.
        flat = " ".join(str(a) for tup in buf for a in tup)
        self.assertIn("config:", flat)
        self.assertIn("eligible right now", flat)


if __name__ == "__main__":
    unittest.main()
