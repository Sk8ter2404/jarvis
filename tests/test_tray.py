"""Tests for tray.py — the pystray system-tray applet that fronts JARVIS.

tray.py is a *root* module (not under a package). It imports pystray + PIL at
module load (both are on CI) and otherwise pulls in only stdlib, so a plain
``import tray`` is safe under tools/run_tests_ci_sim.py. We never run a real
tray: pystray.Icon.run() / icon.stop() are mocked, the animation loop is driven
exactly one iteration via a stop-event sentinel, and every dialog/explorer/
subprocess shell-out is patched out.

Isolation contract (so the real C:\\JARVIS tree is never read or written):
  • Every absolute path constant tray.py resolved at import (HUD_STATE_FILE,
    TRAY_COMMANDS_FILE, TODO_FILE, LOGS_DIR, CHANGELOG_FILE, …) is repointed
    into a per-test TemporaryDirectory via mock.patch.object, auto-restored.
  • _send_command writes a tempfile into PROJECT_DIR then os.replace()s it onto
    TRAY_COMMANDS_FILE, so PROJECT_DIR is redirected too.
  • Module-level caches/globals (_base_icon, _icon_path, _FONT_CACHE,
    _queue_cache, _parent_pid, _stop_event) are snapshotted and restored in
    tearDown so tests can't leak state into each other.

pystray facts these tests rely on (verified against the installed pystray):
  • MenuItem(icon_arg) — calling a MenuItem invokes its action as
    action(icon, item).
  • item.text / item.checked / item.enabled evaluate the lambdas passed at
    construction, handing the MenuItem itself in as the argument.
  • pystray.Menu is iterable through ``.items``; pystray.Menu.SEPARATOR is the
    sentinel separator object.
"""
import functools
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


# --------------------------------------------------------------------------- #
# Headless-safe pystray shim.
#
# ``tray.py`` does ``import pystray`` at module top. The real pystray selects a
# GUI backend AT IMPORT TIME: on the Linux CI runner it tries the X11 backend,
# which connects to ``$DISPLAY`` and raises
#     Xlib.error.DisplayNameError: Bad display name ""
# on a headless host — so merely importing ``tray`` (hence collecting this test
# module) explodes on CI even though pystray is installed.
#
# We sidestep that by injecting a FAKE ``pystray`` into ``sys.modules`` BEFORE
# importing ``tray``, so ``tray.py`` binds its module-level ``pystray`` name to
# the fake and the real X11 backend is never touched. The fake faithfully
# re-implements the two backend-independent classes the tray tests actually use
# — ``pystray._base.Menu`` and ``pystray.MenuItem`` (text/checked/enabled
# lambda evaluation, ``MenuItem(icon)`` -> ``action(icon, item)``, ``Menu.items``
# iteration, ``Menu.SEPARATOR`` sentinel, ``.submenu`` for nested menus) — and a
# do-nothing ``Icon`` placeholder (every test that reaches ``pystray.Icon``
# already patches it via ``mock.patch.object(tray.pystray, "Icon")``).
#
# The fake is installed only for the duration of ``import tray`` and then the
# previous ``sys.modules['pystray']`` (if any) is restored, so it can't leak
# into other test modules that may want the real package.
# --------------------------------------------------------------------------- #
class _FakeMenuItem:
    """Behavioural twin of ``pystray._base.MenuItem`` (the parts tray uses)."""

    def __init__(self, text, action, checked=None, radio=False, default=False,
                 visible=True, enabled=True):
        self.__name__ = str(text)
        self._text = self._wrap(text or "")
        self._action = self._assert_action(action)
        self._checked = self._assert_callable(checked, lambda _: None)
        self._radio = self._wrap(radio)
        self._default = self._wrap(default)
        self._visible = self._wrap(visible)
        self._enabled = self._wrap(enabled)

    def __call__(self, icon):
        if not isinstance(self._action, _FakeMenu):
            return self._action(icon, self)

    def __str__(self):
        if isinstance(self._action, _FakeMenu):
            return "%s =>\n%s" % (self.text, str(self._action))
        return self.text

    @property
    def text(self):
        return self._text(self)

    @property
    def checked(self):
        return self._checked(self)

    @property
    def radio(self):
        return self._radio(self) if self.checked is not None else False

    @property
    def default(self):
        return self._default(self)

    @property
    def visible(self):
        if isinstance(self._action, _FakeMenu):
            return self._visible(self) and self._action.visible
        return self._visible(self)

    @property
    def enabled(self):
        return self._enabled(self)

    @property
    def submenu(self):
        return self._action if isinstance(self._action, _FakeMenu) else None

    @staticmethod
    def _assert_action(action):
        if action is None:
            return lambda *_: None
        if not hasattr(action, "__code__"):
            return action
        argcount = action.__code__.co_argcount - (
            1 if inspect.ismethod(action) else 0)
        if argcount == 0:
            @functools.wraps(action)
            def wrapper0(*args):
                return action()
            return wrapper0
        if argcount == 1:
            @functools.wraps(action)
            def wrapper1(icon, *args):
                return action(icon)
            return wrapper1
        if argcount == 2:
            return action
        raise ValueError(action)

    @staticmethod
    def _assert_callable(value, default):
        if value is None:
            return default
        if callable(value):
            return value
        raise ValueError(value)

    @staticmethod
    def _wrap(value):
        return value if callable(value) else lambda _: value


class _FakeMenu:
    """Behavioural twin of ``pystray._base.Menu`` (the parts tray uses)."""

    SEPARATOR = _FakeMenuItem("- - - -", None)

    def __init__(self, *items):
        self._items = tuple(items)

    @property
    def items(self):
        if (len(self._items) == 1
                and not isinstance(self._items[0], _FakeMenuItem)
                and callable(self._items[0])):
            return self._items[0]()
        return self._items

    @property
    def visible(self):
        return bool(self)

    def __call__(self, icon):
        try:
            return next(mi for mi in self.items if mi.default)(icon)
        except StopIteration:
            pass

    def __iter__(self):
        return iter(self._visible_items())

    def __bool__(self):
        return len(self._visible_items()) > 0

    def __str__(self):
        return "\n".join(
            "\n".join("    %s" % l for l in str(i).splitlines()) for i in self)

    def _visible_items(self):
        def cleaned(items):
            was_separator = False
            for i in items:
                if not i.visible:
                    continue
                if i is self.SEPARATOR:
                    if was_separator:
                        continue
                    was_separator = True
                else:
                    was_separator = False
                yield i

        def strip_head(items):
            import itertools
            return itertools.dropwhile(lambda i: i is self.SEPARATOR, items)

        def strip_tail(items):
            return reversed(list(strip_head(reversed(list(items)))))

        return tuple(strip_tail(strip_head(cleaned(self.items))))


class _FakeIcon:
    """Placeholder for ``pystray.Icon``. Tests that reach the real icon path
    patch this out via ``mock.patch.object(tray.pystray, "Icon")``; it exists
    only so the attribute is present and patchable."""

    def __init__(self, *a, **k):
        self.name = a[0] if a else k.get("name")
        self.icon = k.get("icon")
        self.title = k.get("title")
        self.menu = k.get("menu")

    def run(self, *a, **k):
        pass

    def stop(self, *a, **k):
        pass


def _make_fake_pystray():
    import types
    mod = types.ModuleType("pystray")
    mod.Menu = _FakeMenu
    mod.MenuItem = _FakeMenuItem
    mod.Icon = _FakeIcon
    # Some pystray consumers import the private base; expose a matching submodule
    # so ``import pystray._base`` (should tray ever do it) also resolves to fakes.
    base = types.ModuleType("pystray._base")
    base.Menu = _FakeMenu
    base.MenuItem = _FakeMenuItem
    mod._base = base
    return mod, base


# Install the fake, import tray so it binds to the fake, then restore whatever
# (if anything) previously occupied the ``pystray`` slot — keeping the fake from
# leaking into sibling test modules.
_saved_pystray = sys.modules.get("pystray")
_saved_pystray_base = sys.modules.get("pystray._base")
_fake_pystray, _fake_pystray_base = _make_fake_pystray()
sys.modules["pystray"] = _fake_pystray
sys.modules["pystray._base"] = _fake_pystray_base
try:
    import tray
finally:
    if _saved_pystray is not None:
        sys.modules["pystray"] = _saved_pystray
    else:
        sys.modules.pop("pystray", None)
    if _saved_pystray_base is not None:
        sys.modules["pystray._base"] = _saved_pystray_base
    else:
        sys.modules.pop("pystray._base", None)

# Real implementations, captured before TrayTestBase patches them per test (no
# test may reach the live Ollama / media session / git by accident). getattr
# so the module still imports against an older tray.py (fail-on-old proof).
_REAL_GIT_COMMIT = getattr(tray, "_git_commit", None)
_REAL_FETCH_LOCAL_MODELS = getattr(tray, "_fetch_local_models", None)
_REAL_NOW_PLAYING_LOOKUP = getattr(tray, "_now_playing_lookup", None)
_REAL_SETUP_TRAY_LOGGING = getattr(tray, "_setup_tray_logging", None)
# The real repo root (TrayTestBase repoints tray.PROJECT_DIR at a temp dir).
_REAL_PROJECT_DIR = tray.PROJECT_DIR


# --------------------------------------------------------------------------- #
# Shared base: redirect every path constant into a temp dir + reset globals.
# --------------------------------------------------------------------------- #
class TrayTestBase(unittest.TestCase):
    # tray path constants that point at the real repo — all redirected per-test.
    _PATH_ATTRS = (
        "PROJECT_DIR", "HUD_STATE_FILE", "TRAY_COMMANDS_FILE", "TODO_FILE",
        "LOGS_DIR", "ASSETS_DIR", "DEFAULT_ICON_PATH", "DATA_DIR",
        "CHANGELOG_FILE", "RELEASE_VERSION_FILE", "VERSION_FILE",
        "INSTANCES_FILE", "PIPELINE_LOCK_FILE",
        "OVERNIGHT_FLAG", "MEMORY_FACTS_FILE", "SETTINGS_WINDOW", "SHOW_LOG_PS1",
        "HUD_SCRIPT", "TRAY_RESULTS_FILE", "TRAY_LOG_FILE",
        "SETTINGS_LOG_FILE", "TRAY_RESULTS_DIR", "CRASH_TRACES_LOG",
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

        # Map each path constant to a sibling under the temp dir, preserving the
        # original basename so behaviour that keys off the filename still holds.
        for attr in self._PATH_ATTRS:
            # getattr/create=True: tolerate an older tray.py (fail-on-old proof).
            orig = getattr(tray, attr, None)
            base = os.path.basename(orig) if orig else attr
            patcher = mock.patch.object(tray, attr, os.path.join(self.dir, base),
                                        create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        # PROJECT_DIR itself must be the temp dir (mkstemp(dir=PROJECT_DIR)).
        p = mock.patch.object(tray, "PROJECT_DIR", self.dir)
        p.start()
        self.addCleanup(p.stop)
        # LOGS_DIR / DATA_DIR as real subdirs we can populate.
        for attr in ("LOGS_DIR", "DATA_DIR", "ASSETS_DIR"):
            sub = os.path.join(self.dir, attr.lower())
            q = mock.patch.object(tray, attr, sub)
            q.start()
            self.addCleanup(q.stop)

        # Snapshot mutable module globals so each test starts clean and can't
        # leak into the next (tearDown restores the originals).
        self._saved_base_icon = tray._base_icon
        self._saved_icon_path = tray._icon_path
        self._saved_font_cache = dict(tray._FONT_CACHE)
        self._saved_queue_cache = dict(tray._queue_cache)
        self._saved_parent_pid = list(tray._parent_pid)
        self._saved_stop_event = tray._stop_event

        tray._base_icon = None
        tray._icon_path = tray.DEFAULT_ICON_PATH
        tray._FONT_CACHE.clear()
        tray._queue_cache.clear()
        tray._queue_cache.update({"count": 0, "at": 0.0})
        tray._parent_pid[0] = 0
        tray._stop_event = threading.Event()
        self._reset_tray_runtime_state()
        self.addCleanup(self._reset_tray_runtime_state)
        # No test may reach the live Ollama or the OS media session: the model
        # picker and the now-playing header refresh on background threads.
        # And main() must never re-point THIS process's stdout/stderr at a tray
        # log (the dedicated test calls the real one on private streams).
        # create=True keeps this base usable against an older tray.py (the
        # fail-on-old-code proof runs these tests there).
        for name, value in (("_fetch_local_models", []),
                            ("_now_playing_lookup", "Apple Music: closed"),
                            ("_setup_tray_logging", False)):
            p = mock.patch.object(tray, name, return_value=value, create=True)
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _reset_tray_runtime_state():
        """Module-level runtime state added with the 2026-09-30 tray fixes.
        Tolerant of its absence (older tray.py)."""
        def _get(name):
            return getattr(tray, name, None)
        for name in ("_pending", "_confirm_armed"):
            if _get(name) is not None:
                _get(name).clear()
        if _get("_results_state") is not None:
            tray._results_state.update({"checked_at": 0.0, "mtime": None,
                                        "read_at": 0.0, "shown": set()})
        if _get("_menu_state") is not None:
            tray._menu_state.update({"sig": None, "at": 0.0})
        if _get("_menu_open") is not None:
            tray._menu_open.clear()
        if _get("_icon_state") is not None:
            tray._icon_state.update({"key": None, "title": None})
        if _get("_icon_ref") is not None:
            tray._icon_ref[0] = None
        if _get("_np_cache") is not None:
            tray._np_cache.update({"text": "", "at": 0.0, "busy": False})
        if _get("_models_cache") is not None:
            tray._models_cache.update({"tags": [], "at": 0.0, "busy": False})
        if _get("_hud_snapshot") is not None:
            tray._hud_snapshot.data = None

    def tearDown(self):
        self._tmp.cleanup()
        tray._base_icon = self._saved_base_icon
        tray._icon_path = self._saved_icon_path
        tray._FONT_CACHE.clear()
        tray._FONT_CACHE.update(self._saved_font_cache)
        tray._queue_cache.clear()
        tray._queue_cache.update(self._saved_queue_cache)
        tray._parent_pid[:] = self._saved_parent_pid
        tray._stop_event = self._saved_stop_event

    # -- helpers ---------------------------------------------------------- #
    def _write(self, path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _write_hud(self, **fields):
        self._write(tray.HUD_STATE_FILE, json.dumps(fields))

    def _read_commands(self):
        if not os.path.exists(tray.TRAY_COMMANDS_FILE):
            return []
        with open(tray.TRAY_COMMANDS_FILE, encoding="utf-8") as f:
            return json.load(f)

    def _last_command(self):
        cmds = self._read_commands()
        self.assertTrue(cmds, "no command was written")
        return cmds[-1]

    def _bust_queue_cache(self):
        """Force _count_pending_tasks to re-read the file (skip the 2s TTL)."""
        tray._queue_cache["at"] = 0.0


# --------------------------------------------------------------------------- #
# _read_hud_state
# --------------------------------------------------------------------------- #
class ReadHudStateTests(TrayTestBase):
    def test_reads_valid_json(self):
        self._write_hud(state="speaking", tts_amplitude=0.3)
        self.assertEqual(tray._read_hud_state()["state"], "speaking")

    def test_missing_file_returns_empty(self):
        self.assertEqual(tray._read_hud_state(), {})

    def test_malformed_json_returns_empty(self):
        self._write(tray.HUD_STATE_FILE, "{not valid json")
        self.assertEqual(tray._read_hud_state(), {})

    def test_json_null_returns_empty_dict(self):
        # "null" parses to None; `or {}` must coerce to {}.
        self._write(tray.HUD_STATE_FILE, "null")
        self.assertEqual(tray._read_hud_state(), {})


# --------------------------------------------------------------------------- #
# _send_command — the command-file IPC writer
# --------------------------------------------------------------------------- #
class SendCommandTests(TrayTestBase):
    def test_writes_single_command(self):
        tray._send_command("restart")
        cmds = self._read_commands()
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0]["cmd"], "restart")
        self.assertIn("ts", cmds[0])

    def test_kwargs_are_merged(self):
        tray._send_command("switch_llm", backend="anthropic")
        self.assertEqual(self._last_command()["backend"], "anthropic")

    def test_appends_to_existing_list(self):
        tray._send_command("a")
        tray._send_command("b")
        tray._send_command("c")
        self.assertEqual([c["cmd"] for c in self._read_commands()], ["a", "b", "c"])

    def test_leaves_no_tempfiles(self):
        tray._send_command("x")
        strays = [f for f in os.listdir(self.dir) if f.endswith(".tmp")]
        self.assertEqual(strays, [])

    def test_corrupt_existing_file_is_discarded(self):
        self._write(tray.TRAY_COMMANDS_FILE, "garbage{{")
        tray._send_command("recover")
        cmds = self._read_commands()
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0]["cmd"], "recover")

    def test_existing_non_list_payload_is_reset(self):
        # raw_decode would yield a dict, not a list -> drop it, start fresh.
        self._write(tray.TRAY_COMMANDS_FILE, json.dumps({"cmd": "stale"}))
        tray._send_command("fresh")
        cmds = self._read_commands()
        self.assertEqual([c["cmd"] for c in cmds], ["fresh"])

    def test_trailing_garbage_after_list_is_tolerated(self):
        # raw_decode stops at the end of the first JSON value.
        self._write(tray.TRAY_COMMANDS_FILE,
                    json.dumps([{"cmd": "old", "ts": 1}]) + "\n<<junk>>")
        tray._send_command("new")
        self.assertEqual([c["cmd"] for c in self._read_commands()], ["old", "new"])

    def test_write_failure_is_swallowed(self):
        # If the atomic write blows up, _send_command must not raise.
        with mock.patch.object(tray.tempfile, "mkstemp",
                               side_effect=OSError("disk full")):
            tray._send_command("boom")  # should not raise

    def test_empty_existing_file_treated_as_no_commands(self):
        self._write(tray.TRAY_COMMANDS_FILE, "   ")
        tray._send_command("first")
        self.assertEqual([c["cmd"] for c in self._read_commands()], ["first"])

    def test_replace_failure_cleans_tmp_and_swallows(self):
        # os.replace failing after the tmp is written: the inner handler removes
        # the tmp and re-raises, the outer handler swallows. No tmp left behind.
        with mock.patch.object(tray.os, "replace",
                               side_effect=OSError("rename denied")):
            tray._send_command("nope")  # must not raise
        strays = [f for f in os.listdir(self.dir) if f.endswith(".tmp")]
        self.assertEqual(strays, [])
        # The destination file was never created.
        self.assertFalse(os.path.exists(tray.TRAY_COMMANDS_FILE))

    def test_replace_failure_tmp_remove_also_failing_still_swallowed(self):
        # Both os.replace AND the cleanup os.remove fail — the bare
        # `except Exception: pass` around remove keeps us from crashing.
        with mock.patch.object(tray.os, "replace",
                               side_effect=OSError("rename denied")), \
             mock.patch.object(tray.os, "remove",
                               side_effect=OSError("remove denied")):
            tray._send_command("nope")  # must not raise


# --------------------------------------------------------------------------- #
# _load_base_icon + icon rendering
# --------------------------------------------------------------------------- #
class LoadBaseIconTests(TrayTestBase):
    def _make_png(self, path, size=(64, 64)):
        from PIL import Image
        Image.new("RGBA", size, (10, 20, 30, 255)).save(path)

    def test_missing_path_leaves_base_none(self):
        tray._load_base_icon(os.path.join(self.dir, "nope.png"))
        self.assertIsNone(tray._base_icon)

    def test_empty_path_leaves_base_none(self):
        tray._load_base_icon("")
        self.assertIsNone(tray._base_icon)

    def test_loads_and_resizes_to_canvas(self):
        p = os.path.join(self.dir, "icon.png")
        self._make_png(p, size=(128, 128))
        tray._load_base_icon(p)
        self.assertIsNotNone(tray._base_icon)
        self.assertEqual(tray._base_icon.size, (tray.SIZE, tray.SIZE))

    def test_already_correct_size_kept(self):
        p = os.path.join(self.dir, "icon.png")
        self._make_png(p, size=(tray.SIZE, tray.SIZE))
        tray._load_base_icon(p)
        self.assertEqual(tray._base_icon.size, (tray.SIZE, tray.SIZE))

    def test_corrupt_file_falls_back_to_none(self):
        p = os.path.join(self.dir, "icon.png")
        self._write(p, "this is not a PNG")
        tray._load_base_icon(p)
        self.assertIsNone(tray._base_icon)


class RenderIconTests(TrayTestBase):
    def _assert_image(self, img):
        from PIL import Image
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (tray.SIZE, tray.SIZE))
        self.assertEqual(img.mode, "RGBA")

    def test_procedural_render_when_no_base(self):
        tray._base_icon = None
        self._assert_image(tray._render_icon("idle", 0, queue_count=0))

    def test_render_with_base(self):
        from PIL import Image
        tray._base_icon = Image.new("RGBA", (tray.SIZE, tray.SIZE), (0, 0, 0, 255))
        self._assert_image(tray._render_icon("speaking", 2, tts_amplitude=0.4))

    def test_render_with_queue_badge(self):
        # queue_count > 0 exercises the numeric-badge text path.
        self._assert_image(tray._render_icon("idle", 0, queue_count=7))

    def test_render_queue_overflow_badge(self):
        # >= 100 -> "99+" overflow rule.
        self._assert_image(tray._render_icon("idle", 0, queue_count=250))

    def test_render_with_base_and_badge(self):
        from PIL import Image
        tray._base_icon = Image.new("RGBA", (tray.SIZE, tray.SIZE), (5, 5, 5, 255))
        self._assert_image(tray._render_icon("idle", 0, queue_count=42))

    def test_procedural_badge_textbbox_failure_swallowed(self):
        # If measuring the badge glyph raises, the badge is silently skipped and
        # the procedural icon still renders (the bare except around textbbox).
        from PIL import ImageDraw
        tray._base_icon = None
        with mock.patch.object(ImageDraw.ImageDraw, "textbbox",
                               side_effect=RuntimeError("no metrics")):
            self._assert_image(tray._render_icon("idle", 0, queue_count=5))

    def test_base_badge_textbbox_failure_swallowed(self):
        from PIL import Image, ImageDraw
        tray._base_icon = Image.new("RGBA", (tray.SIZE, tray.SIZE), (0, 0, 0, 255))
        with mock.patch.object(ImageDraw.ImageDraw, "textbbox",
                               side_effect=RuntimeError("no metrics")):
            self._assert_image(tray._render_icon("idle", 0, queue_count=5))

    def test_badge_skipped_when_font_unavailable(self):
        # When _get_font yields None, both renderers skip the badge entirely.
        with mock.patch.object(tray, "_get_font", return_value=None):
            self._assert_image(tray._render_icon("idle", 0, queue_count=9))
            from PIL import Image
            tray._base_icon = Image.new("RGBA", (tray.SIZE, tray.SIZE), (0, 0, 0, 255))
            self._assert_image(tray._render_icon("idle", 0, queue_count=9))

    def test_signal_compute_failure_uses_neutral(self):
        # If _compute_signal_colors raises, _render_icon must still return an
        # image built from synthesised neutral signals.
        with mock.patch.object(tray, "_compute_signal_colors",
                               side_effect=RuntimeError("boom")):
            self._assert_image(tray._render_icon("idle", 0))

    def test_base_composite_failure_falls_back_to_procedural(self):
        from PIL import Image
        tray._base_icon = Image.new("RGBA", (tray.SIZE, tray.SIZE), (0, 0, 0, 255))
        with mock.patch.object(tray, "_render_icon_with_base",
                               side_effect=RuntimeError("composite fail")):
            # Should swallow + fall through to the procedural renderer.
            self._assert_image(tray._render_icon("idle", 0, queue_count=3))

    def test_muted_state_renders(self):
        self._assert_image(tray._render_icon("listening", 1, muted=True))

    def test_bambu_active_renders(self):
        self._assert_image(tray._render_icon("idle", 0, bambu_active=True))


class IconRedesignTests(TrayTestBase):
    """Behavioural guarantees of the redesigned icon (legible at 16/24 px):
    full-reactor listen tint as the primary signal, a speaking halo, a large
    corner queue badge, and a bambu print-mark — all over the arc-reactor base
    with a procedural disc fallback that mirrors the same overlays."""

    def _img(self, *a, **k):
        return tray._render_icon(*a, **k)

    def _reactor_base(self):
        """A luminance-varied stand-in for the real arc-reactor PNG so that
        tinting produces visibly different pixels (a flat fill would tint to the
        same value for every state and defeat the comparison)."""
        from PIL import Image, ImageDraw
        b = Image.new("RGBA", (tray.SIZE, tray.SIZE), (0, 0, 0, 0))
        d = ImageDraw.Draw(b)
        d.ellipse([6, 6, tray.SIZE - 6, tray.SIZE - 6], fill=(0, 190, 255, 255))
        d.ellipse([22, 22, tray.SIZE - 22, tray.SIZE - 22], fill=(200, 240, 255, 255))
        return b

    def _nonblank_px(self, img):
        """Count pixels with any opacity — a quick 'something rendered' gauge.
        Uses the alpha channel's histogram (getdata() is deprecated in Pillow)."""
        alpha = img.getchannel("A")
        hist = alpha.histogram()       # 256 buckets, index == alpha value
        return sum(hist[1:])           # everything with alpha > 0

    # -- primary signal: full-icon listen tint reads as different colours ---- #
    def test_procedural_muted_differs_from_awake(self):
        awake = self._img("listening", 0, muted=False)
        muted = self._img("listening", 0, muted=True)
        self.assertNotEqual(awake.tobytes(), muted.tobytes())

    def test_procedural_standby_differs_from_awake(self):
        awake = self._img("listening", 0)
        standby = self._img("standby", 0)
        self.assertNotEqual(awake.tobytes(), standby.tobytes())

    def test_base_muted_differs_from_awake(self):
        tray._base_icon = self._reactor_base()
        awake = self._img("listening", 0, muted=False)
        muted = self._img("listening", 0, muted=True)
        self.assertNotEqual(awake.tobytes(), muted.tobytes())

    def test_tint_preserves_size_and_alpha_shape(self):
        # Tinting must keep the canvas size and not fill the transparent
        # surround (the reactor identity / shape survives).
        base = self._reactor_base()
        out = tray._tint_image(base, tray.LISTEN_RED, tray.TINT_STRENGTH_MUTED)
        self.assertEqual(out.size, (tray.SIZE, tray.SIZE))
        self.assertEqual(out.mode, "RGBA")
        # A corner pixel of the base is fully transparent; it must stay so.
        self.assertEqual(out.getpixel((0, 0))[3], 0)

    def test_tint_failure_returns_copy(self):
        # _tint_image must never raise — on an internal error it returns a copy.
        base = self._reactor_base()
        with mock.patch.object(tray.ImageChops, "multiply",
                               side_effect=RuntimeError("chops boom")):
            out = tray._tint_image(base, tray.LISTEN_GREEN, 0.5)
        self.assertEqual(out.size, (tray.SIZE, tray.SIZE))

    # -- speaking halo: pulsing ring appears only while speaking ------------- #
    def test_speaking_changes_pixels_vs_quiet(self):
        tray._base_icon = self._reactor_base()
        quiet = self._img("idle", 0)
        speaking = self._img("speaking", 1, tts_amplitude=0.6)
        self.assertNotEqual(quiet.tobytes(), speaking.tobytes())

    def test_halo_noop_when_not_speaking(self):
        # speak_t == 0 -> the halo draws nothing (image unchanged).
        base = self._reactor_base()
        before = base.tobytes()
        tray._draw_speaking_halo(base, 0.0, tray.SPEAK_BLUE)
        self.assertEqual(base.tobytes(), before)

    def test_halo_draws_when_speaking(self):
        base = self._reactor_base()
        before = base.tobytes()
        tray._draw_speaking_halo(base, 0.9, tray.SPEAK_BLUE)
        self.assertNotEqual(base.tobytes(), before)

    def test_halo_failure_swallowed(self):
        base = self._reactor_base()
        with mock.patch.object(tray.ImageFilter, "GaussianBlur",
                               side_effect=RuntimeError("blur boom")):
            tray._draw_speaking_halo(base, 0.9, tray.SPEAK_BLUE)  # must not raise

    # -- queue badge: large, high-contrast, only when count > 0 ------------- #
    def test_queue_badge_absent_when_zero(self):
        base = self._reactor_base()
        before = base.tobytes()
        tray._draw_queue_badge(base, 0, tray.QUEUE_YELLOW)
        self.assertEqual(base.tobytes(), before)

    def test_queue_badge_appears_when_count_positive(self):
        base = self._reactor_base()
        before = base.tobytes()
        tray._draw_queue_badge(base, 3, tray.QUEUE_YELLOW)
        self.assertNotEqual(base.tobytes(), before)

    def test_queue_badge_in_render_changes_image(self):
        tray._base_icon = self._reactor_base()
        none = self._img("idle", 0, queue_count=0)
        some = self._img("idle", 0, queue_count=5)
        self.assertNotEqual(none.tobytes(), some.tobytes())

    def test_queue_badge_overflow_renders(self):
        # >= 100 uses the "99+" string (smaller glyph) and still renders.
        base = self._reactor_base()
        tray._draw_queue_badge(base, 250, tray.QUEUE_YELLOW)  # must not raise
        self._assert_image_like(self._img("idle", 0, queue_count=250))

    def test_queue_badge_is_large(self):
        # The badge must be a LARGE corner mark (legibility at 24px), i.e. a
        # meaningful fraction of the canvas — guard against silent shrink.
        self.assertGreaterEqual(tray.BADGE_FRAC, 0.35)

    def test_queue_badge_font_none_skips_digit(self):
        base = self._reactor_base()
        with mock.patch.object(tray, "_get_font", return_value=None):
            tray._draw_queue_badge(base, 7, tray.QUEUE_YELLOW)  # disc only, no raise

    def test_queue_badge_failure_swallowed(self):
        base = self._reactor_base()
        with mock.patch.object(tray.ImageDraw.ImageDraw, "ellipse",
                               side_effect=RuntimeError("ellipse boom")):
            tray._draw_queue_badge(base, 7, tray.QUEUE_YELLOW)  # must not raise

    # -- bambu print-mark: secondary corner mark, only when printing -------- #
    def test_bambu_mark_changes_image(self):
        tray._base_icon = self._reactor_base()
        idle = self._img("idle", 0, bambu_active=False)
        printing = self._img("idle", 0, bambu_active=True)
        self.assertNotEqual(idle.tobytes(), printing.tobytes())

    def test_bambu_mark_failure_swallowed(self):
        base = self._reactor_base()
        with mock.patch.object(tray.ImageDraw.ImageDraw, "polygon",
                               side_effect=RuntimeError("poly boom")):
            tray._draw_bambu_mark(base)  # must not raise

    # -- procedural fallback mirrors the design ----------------------------- #
    def test_procedural_disc_renders_and_tints(self):
        green = tray._render_reactor_disc(tray.LISTEN_GREEN)
        red = tray._render_reactor_disc(tray.LISTEN_RED)
        self._assert_image_like(green)
        self._assert_image_like(red)
        # Different tint colour -> different disc pixels.
        self.assertNotEqual(green.tobytes(), red.tobytes())
        # The disc actually draws something (not a blank canvas).
        self.assertGreater(self._nonblank_px(green), 0)

    def test_render_never_raises_on_garbage_state(self):
        # Bad/oddball inputs degrade rather than crash (watchdog regression risk).
        for st in (None, "", "???", 12345, object()):
            self._assert_image_like(self._img(st, 0, queue_count=-3))

    def test_flat_fallback_when_both_renderers_fail(self):
        # If BOTH the base composite and the procedural renderer blow up, the
        # final guard still returns a valid 64px RGBA image.
        from PIL import Image
        tray._base_icon = Image.new("RGBA", (tray.SIZE, tray.SIZE), (0, 0, 0, 255))
        with mock.patch.object(tray, "_render_icon_with_base",
                               side_effect=RuntimeError("base boom")), \
             mock.patch.object(tray, "_render_icon_procedural",
                               side_effect=RuntimeError("proc boom")):
            self._assert_image_like(self._img("idle", 0))

    def _assert_image_like(self, img):
        from PIL import Image
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (tray.SIZE, tray.SIZE))
        self.assertEqual(img.mode, "RGBA")


class ComputeSignalColorsTests(TrayTestBase):
    def test_muted_is_red(self):
        s = tray._compute_signal_colors("listening", 0, 0.0, 0, True, False)
        self.assertEqual(s["listen"], tray.LISTEN_RED)

    def test_standby_is_gray(self):
        for st in ("standby", "sleeping", "sleep"):
            s = tray._compute_signal_colors(st, 0, 0.0, 0, False, False)
            self.assertEqual(s["listen"], tray.LISTEN_GRAY, st)

    def test_awake_is_green(self):
        s = tray._compute_signal_colors("listening", 0, 0.0, 0, False, False)
        self.assertEqual(s["listen"], tray.LISTEN_GREEN)

    def test_speaking_by_state(self):
        s = tray._compute_signal_colors("speaking", 0, 0.0, 0, False, False)
        self.assertNotEqual(s["speak"], tray.SPEAK_DIM)

    def test_speaking_by_amplitude(self):
        s = tray._compute_signal_colors("idle", 0, 0.9, 0, False, False)
        self.assertNotEqual(s["speak"], tray.SPEAK_DIM)

    def test_quiet_is_dim(self):
        s = tray._compute_signal_colors("idle", 0, 0.0, 0, False, False)
        self.assertEqual(s["speak"], tray.SPEAK_DIM)

    def test_queue_count_clamped_nonneg(self):
        s = tray._compute_signal_colors("idle", 0, 0.0, -5, False, False)
        self.assertEqual(s["queue_count"], 0)
        self.assertEqual(s["queue"], tray.QUEUE_DIM)

    def test_queue_yellow_when_pending(self):
        s = tray._compute_signal_colors("idle", 0, 0.0, 3, False, False)
        self.assertEqual(s["queue"], tray.QUEUE_YELLOW)
        self.assertEqual(s["queue_count"], 3)

    def test_bambu_orange_vs_white(self):
        on = tray._compute_signal_colors("idle", 0, 0.0, 0, False, True)
        off = tray._compute_signal_colors("idle", 0, 0.0, 0, False, False)
        self.assertEqual(on["bambu"], tray.BAMBU_ORANGE)
        self.assertEqual(off["bambu"], tray.BAMBU_WHITE)

    def test_none_state_is_green(self):
        s = tray._compute_signal_colors(None, 0, 0.0, 0, False, False)
        self.assertEqual(s["listen"], tray.LISTEN_GREEN)

    def test_muted_tint_is_stronger(self):
        # Muted pushes the tint harder than awake/standby so RED is unmistakable
        # at 16 px — the redesign's primary-signal guarantee.
        muted = tray._compute_signal_colors("listening", 0, 0.0, 0, True, False)
        awake = tray._compute_signal_colors("listening", 0, 0.0, 0, False, False)
        self.assertEqual(muted["tint_strength"], tray.TINT_STRENGTH_MUTED)
        self.assertEqual(awake["tint_strength"], tray.TINT_STRENGTH)
        self.assertGreater(muted["tint_strength"], awake["tint_strength"])

    def test_speak_t_zero_when_quiet_positive_when_speaking(self):
        quiet = tray._compute_signal_colors("idle", 0, 0.0, 0, False, False)
        loud = tray._compute_signal_colors("speaking", 0, 0.0, 0, False, False)
        self.assertEqual(quiet["speak_t"], 0.0)
        self.assertGreater(loud["speak_t"], 0.0)


class BlendTests(TrayTestBase):
    def test_midpoint(self):
        self.assertEqual(tray._blend((0, 0, 0), (100, 200, 255), 0.5),
                         (50, 100, 127))

    def test_clamps_high(self):
        self.assertEqual(tray._blend((0, 0, 0), (10, 10, 10), 5.0), (10, 10, 10))

    def test_clamps_low(self):
        self.assertEqual(tray._blend((0, 0, 0), (10, 10, 10), -1.0), (0, 0, 0))


class GetFontTests(TrayTestBase):
    def test_returns_font_and_caches(self):
        f1 = tray._get_font(12)
        self.assertIn(12, tray._FONT_CACHE)
        f2 = tray._get_font(12)
        self.assertIs(f1, f2)

    def test_truetype_failure_uses_default(self):
        # Force every named-font truetype lookup to fail so the code falls
        # through to load_default(). (NB: on Pillow 11+ load_default() itself
        # calls truetype() with an embedded font, so we stub load_default to a
        # sentinel rather than relying on a side_effect that would break it.)
        from PIL import ImageFont
        sentinel = object()
        with mock.patch.object(ImageFont, "truetype",
                               side_effect=OSError("no font")), \
             mock.patch.object(ImageFont, "load_default",
                               return_value=sentinel):
            f = tray._get_font(14)
        self.assertIs(f, sentinel)
        # Cached under the requested size for reuse.
        self.assertIs(tray._FONT_CACHE.get(14), sentinel)

    def test_imagefont_import_failure_returns_none(self):
        # If ImageFont can't even be imported, _get_font swallows it and
        # returns None (renderers guard against a None font).
        import builtins
        real_import = builtins.__import__

        def boom(name, *a, **k):
            if name == "PIL" and a and "ImageFont" in (a[2] or ()):
                raise ImportError("no PIL.ImageFont")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=boom):
            self.assertIsNone(tray._get_font(99))


# --------------------------------------------------------------------------- #
# Parent watchdog
# --------------------------------------------------------------------------- #
class ParentAliveTests(TrayTestBase):
    def setUp(self):
        super().setUp()
        # 2026-07-12: _parent_alive consults the AUTHORITATIVE
        # core.parent_watch layer first (real Win32 syscalls — fake pids
        # 4321/99 read dead for real). Make it RAISE so these tests keep
        # exercising the psutil / os.kill fallbacks beneath it;
        # parent_watch has its own suite.
        import core.parent_watch as _pw
        p = mock.patch.object(_pw, "parent_is_alive",
                              side_effect=RuntimeError("stubbed out"))
        p.start()
        self.addCleanup(p.stop)

    def test_no_pid_is_alive(self):
        tray._parent_pid[0] = 0
        self.assertTrue(tray._parent_alive())

    def test_psutil_path_true(self):
        tray._parent_pid[0] = 4321
        with mock.patch.object(tray, "_HAS_PSUTIL", True), \
             mock.patch.object(tray, "psutil", create=True) as ps:
            ps.pid_exists.return_value = True
            self.assertTrue(tray._parent_alive())
            ps.pid_exists.assert_called_once_with(4321)

    def test_psutil_path_false(self):
        tray._parent_pid[0] = 4321
        with mock.patch.object(tray, "_HAS_PSUTIL", True), \
             mock.patch.object(tray, "psutil", create=True) as ps:
            ps.pid_exists.return_value = False
            self.assertFalse(tray._parent_alive())

    def test_psutil_raises_defaults_alive(self):
        tray._parent_pid[0] = 99
        with mock.patch.object(tray, "_HAS_PSUTIL", True), \
             mock.patch.object(tray, "psutil", create=True) as ps:
            ps.pid_exists.side_effect = RuntimeError("boom")
            self.assertTrue(tray._parent_alive())

    def test_oskill_path_alive(self):
        tray._parent_pid[0] = 777
        with mock.patch.object(tray, "_HAS_PSUTIL", False), \
             mock.patch.object(tray.os, "kill", return_value=None) as k:
            self.assertTrue(tray._parent_alive())
            k.assert_called_once_with(777, 0)

    def test_oskill_path_dead(self):
        tray._parent_pid[0] = 777
        with mock.patch.object(tray, "_HAS_PSUTIL", False), \
             mock.patch.object(tray.os, "kill",
                               side_effect=ProcessLookupError()):
            self.assertFalse(tray._parent_alive())


# --------------------------------------------------------------------------- #
# _classify_state
# --------------------------------------------------------------------------- #
class ClassifyStateTests(TrayTestBase):
    def test_full_mapping(self):
        out = tray._classify_state({
            "state": "SPEAKING", "mic_level": "0.5", "tts_amplitude": 0.3,
            "mic_muted": True, "bambu_active": 1,
        })
        self.assertEqual(out["state"], "speaking")
        self.assertEqual(out["mic_level"], 0.5)
        self.assertEqual(out["tts_amplitude"], 0.3)
        self.assertTrue(out["muted"])
        self.assertTrue(out["bambu_active"])

    def test_muted_via_muted_key(self):
        self.assertTrue(tray._classify_state({"muted": True})["muted"])

    def test_empty_defaults(self):
        out = tray._classify_state({})
        self.assertEqual(out["state"], "")
        self.assertEqual(out["mic_level"], 0.0)
        self.assertFalse(out["muted"])
        self.assertFalse(out["bambu_active"])


# --------------------------------------------------------------------------- #
# _count_pending_tasks + queue cache
# --------------------------------------------------------------------------- #
class CountPendingTasksTests(TrayTestBase):
    def test_counts_unchecked_only(self):
        self._write(tray.TODO_FILE,
                    "# Queue\n- [ ] one\n- [x] done\n- [ ] two\n  - [ ] indented\n")
        self._bust_queue_cache()
        self.assertEqual(tray._count_pending_tasks(), 3)

    def test_missing_file_zero(self):
        self._bust_queue_cache()
        self.assertEqual(tray._count_pending_tasks(), 0)

    def test_cache_returns_stale_within_ttl(self):
        self._write(tray.TODO_FILE, "- [ ] a\n")
        self._bust_queue_cache()
        self.assertEqual(tray._count_pending_tasks(), 1)
        # Rewrite with more tasks, but the 2s TTL should keep the cached 1.
        self._write(tray.TODO_FILE, "- [ ] a\n- [ ] b\n- [ ] c\n")
        self.assertEqual(tray._count_pending_tasks(), 1)

    def test_recheck_after_cache_bust(self):
        self._write(tray.TODO_FILE, "- [ ] a\n")
        self._bust_queue_cache()
        self.assertEqual(tray._count_pending_tasks(), 1)
        self._write(tray.TODO_FILE, "- [ ] a\n- [ ] b\n")
        self._bust_queue_cache()
        self.assertEqual(tray._count_pending_tasks(), 2)

    def test_read_error_keeps_last_good(self):
        tray._queue_cache.update({"count": 9, "at": 0.0})
        with mock.patch.object(tray.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(tray._count_pending_tasks(), 9)


# --------------------------------------------------------------------------- #
# Command-firing menu callbacks — each should write exactly one command.
# --------------------------------------------------------------------------- #
class CommandCallbackTests(TrayTestBase):
    # Items that answer visibly: the command carries a request id (rid).
    _REQUESTS = {"trigger_overnight", "stop_pipeline", "force_backup",
                 "reload_skills", "run_smoke_test", "switch_llm",
                 "show_llm_stats", "show_recent_facts", "reset_memory",
                 "export_memory", "forget_last_hour", "run_diagnostic",
                 "show_last_diagnostic", "test_mic", "test_tts", "test_vision",
                 "test_each_skill", "latency_benchmark"}

    def _assert_cmd(self, fn, expected_cmd, confirm=False, **expected_kw):
        if confirm:
            # Destructive one-click: the FIRST click only arms the gate.
            with mock.patch.object(tray, "_notify", create=True) as note:
                fn(mock.Mock(), mock.Mock())
            self.assertEqual(self._read_commands(), [],
                             "a confirm-gated item fired on the first click")
            note.assert_called_once()
        fn(mock.Mock(), mock.Mock())
        cmd = self._last_command()
        self.assertEqual(cmd["cmd"], expected_cmd)
        self.assertTrue(cmd.get("cid"), "every command carries a cid")
        if expected_cmd in self._REQUESTS:
            self.assertTrue(cmd.get("rid"), f"{expected_cmd} must ask for an answer")
            self.assertIn(cmd["rid"], tray._pending)
        for k, v in expected_kw.items():
            self.assertEqual(cmd[k], v)

    def test_open_hud(self):
        self._assert_cmd(tray._on_open_hud, "open_hud")

    def test_restart(self):
        self._assert_cmd(tray._on_restart, "restart", confirm=True)

    def test_mute_tts(self):
        self._assert_cmd(tray._on_mute_tts, "mute_tts_toggle")

    def test_mute_mic(self):
        # New mic-mute toggle — must emit EXACTLY this command name, which the
        # bobert capture-loop handler keys off of.
        self._assert_cmd(tray._on_mute_mic, "mic_mute_toggle")

    def test_ambient_mode(self):
        self._assert_cmd(tray._on_ambient_mode, "ambient_mode_toggle")

    def test_force_upgrade(self):
        self._write_hud(overnight_upgrade_enabled=True)
        self._assert_cmd(tray._on_force_upgrade, "trigger_overnight")

    def test_shutdown(self):
        self._assert_cmd(tray._on_shutdown_jarvis, "shutdown_jarvis", confirm=True)

    def test_stop_pipeline(self):
        self._assert_cmd(tray._on_stop_pipeline, "stop_pipeline")

    def test_force_backup(self):
        self._assert_cmd(tray._on_force_backup, "force_backup")

    def test_reload_skills(self):
        self._assert_cmd(tray._on_reload_skills, "reload_skills")

    def test_run_smoke_test(self):
        self._assert_cmd(tray._on_run_smoke_test, "run_smoke_test")

    def test_pause_daemons(self):
        self._assert_cmd(tray._on_pause_daemons, "pause_daemons_toggle")

    def test_switch_anthropic(self):
        # Claude is the PAID backend — a second click is required.
        self._assert_cmd(tray._on_switch_anthropic, "switch_llm", confirm=True,
                         backend="anthropic")

    def test_switch_local(self):
        # 2026-07-21 audit: the tray sends the "ollama" sentinel and lets the
        # monolith resolve the live local default — its old hard-coded short
        # tags ('qwen2.5:14b') drifted from the installed quantised tags and
        # pinned the resolver cache at a model Ollama 404'd on every turn.
        self._assert_cmd(tray._on_switch_local, "switch_llm", backend="ollama")

    def test_no_menu_label_hardcodes_a_local_tag(self):
        # Same-rule stale-duplicate guard: no tray menu LABEL may embed a
        # concrete Ollama tag (family:size) that can drift from the installed
        # reality — the monolith owns model identity.
        import re
        src_path = os.path.abspath(tray.__file__)
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        labels = re.findall(r'MenuItem\(\s*"([^"]*)"', src)
        self.assertTrue(labels, "expected to find pystray MenuItem labels")
        offenders = [l for l in labels
                     if re.search(r"(qwen|llama|gemma|mistral|phi|deepseek)"
                                  r"[\w.\-]*:", l, re.IGNORECASE)]
        self.assertEqual(offenders, [],
                         f"tray menu labels hard-code local model tags: {offenders}")

    def test_picker_model_item_switches_to_that_tag(self):
        # The "Local Model" picker replaces the dead "other…" item: each
        # entry sends the exact installed tag.
        self._assert_cmd(tray._switch_to_model("gemma4:12b"), "switch_llm",
                         backend="gemma4:12b")

    def test_toggle_debug(self):
        self._assert_cmd(tray._on_toggle_debug_mode, "debug_mode_toggle")

    def test_show_llm_stats(self):
        self._assert_cmd(tray._on_show_llm_stats, "show_llm_stats")

    def test_toggle_audio_processing(self):
        self._assert_cmd(tray._on_toggle_audio_processing, "audio_processing_toggle")

    def test_toggle_echo_cancel(self):
        self._assert_cmd(tray._on_toggle_echo_cancel, "audio_echo_cancel_toggle")

    def test_toggle_noise_suppress(self):
        self._assert_cmd(tray._on_toggle_noise_suppress, "audio_noise_suppress_toggle")

    def test_toggle_agc(self):
        self._assert_cmd(tray._on_toggle_agc, "audio_agc_toggle")

    def test_recent_facts(self):
        self._assert_cmd(tray._on_recent_facts, "show_recent_facts")

    def test_reset_memory(self):
        self._assert_cmd(tray._on_reset_memory, "reset_memory", confirm=True)

    def test_export_memory(self):
        self._assert_cmd(tray._on_export_memory, "export_memory")

    def test_forget_last_hour(self):
        self._assert_cmd(tray._on_forget_last_hour, "forget_last_hour",
                         confirm=True)

    def test_run_diagnostic(self):
        self._assert_cmd(tray._on_run_diagnostic, "run_diagnostic")

    def test_show_last_diagnostic(self):
        self._assert_cmd(tray._on_show_last_diagnostic, "show_last_diagnostic")

    def test_test_mic(self):
        self._assert_cmd(tray._on_test_mic, "test_mic")

    def test_test_tts(self):
        self._assert_cmd(tray._on_test_tts, "test_tts")

    def test_test_vision(self):
        self._assert_cmd(tray._on_test_vision, "test_vision")

    def test_test_each_skill(self):
        self._assert_cmd(tray._on_test_each_skill, "test_each_skill")

    def test_latency_benchmark(self):
        self._assert_cmd(tray._on_latency_benchmark, "latency_benchmark")


# --------------------------------------------------------------------------- #
# Pause-listening is a stateful toggle (reads hud_state to decide direction).
# --------------------------------------------------------------------------- #
class PauseListeningToggleTests(TrayTestBase):
    def test_when_awake_enters_standby(self):
        self._write_hud(state="listening")
        tray._on_pause_listening(mock.Mock(), mock.Mock())
        self.assertEqual(self._last_command()["cmd"], "enter_standby")

    def test_when_standby_forces_wake(self):
        self._write_hud(state="standby")
        tray._on_pause_listening(mock.Mock(), mock.Mock())
        self.assertEqual(self._last_command()["cmd"], "force_wake")


# --------------------------------------------------------------------------- #
# Toggle / status state readers (all read hud_state.json)
# --------------------------------------------------------------------------- #
class StateReaderTests(TrayTestBase):
    def test_is_standby_true(self):
        for st in ("standby", "sleeping", "sleep"):
            self._write_hud(state=st)
            self.assertTrue(tray._is_standby(), st)

    def test_is_standby_false(self):
        self._write_hud(state="listening")
        self.assertFalse(tray._is_standby())

    def test_is_listen_paused_tracks_standby(self):
        self._write_hud(state="sleep")
        self.assertTrue(tray._is_listen_paused())

    def test_is_tts_muted(self):
        self._write_hud(tts_muted=True)
        self.assertTrue(tray._is_tts_muted())
        self._write_hud(tts_muted=False)
        self.assertFalse(tray._is_tts_muted())

    def test_is_mic_muted(self):
        self._write_hud(mic_muted=True)
        self.assertTrue(tray._is_mic_muted())
        self._write_hud(mic_muted=False)
        self.assertFalse(tray._is_mic_muted())

    def test_is_mic_muted_absent_is_false(self):
        # Until bobert publishes the field, the toggle reads unchecked.
        self._write_hud()
        self.assertFalse(tray._is_mic_muted())

    def test_is_ambient_mode(self):
        self._write_hud(ambient_mode_active=True)
        self.assertTrue(tray._is_ambient_mode())

    def test_is_debug_mode(self):
        self._write_hud(debug_mode=True)
        self.assertTrue(tray._is_debug_mode())

    def test_is_daemons_paused(self):
        self._write_hud(daemons_paused=True)
        self.assertTrue(tray._is_daemons_paused())

    def test_active_llm_backend(self):
        self._write_hud(llm_backend="Qwen2.5:14B")
        self.assertEqual(tray._active_llm_backend(), "qwen2.5:14b")

    def test_active_llm_backend_absent(self):
        self._write_hud()
        self.assertEqual(tray._active_llm_backend(), "")


class AudioFieldReaderTests(TrayTestBase):
    def test_absent_field_defaults_true(self):
        self._write_hud()  # no audio_* keys
        self.assertTrue(tray._is_audio_processing_enabled())
        self.assertTrue(tray._is_echo_cancel_enabled())
        self.assertTrue(tray._is_noise_suppress_enabled())
        self.assertTrue(tray._is_agc_enabled())

    def test_explicit_false_respected(self):
        self._write_hud(audio_processing_enabled=False, echo_cancel_enabled=False,
                        noise_suppress_enabled=False, agc_enabled=False)
        self.assertFalse(tray._is_audio_processing_enabled())
        self.assertFalse(tray._is_echo_cancel_enabled())
        self.assertFalse(tray._is_noise_suppress_enabled())
        self.assertFalse(tray._is_agc_enabled())

    def test_explicit_true_respected(self):
        self._write_hud(audio_processing_enabled=True)
        self.assertTrue(tray._is_audio_processing_enabled())


class PipelineRunningTests(TrayTestBase):
    def test_false_when_no_flags(self):
        self.assertFalse(tray._is_pipeline_running())

    def test_true_when_lock_present(self):
        self._write(tray.PIPELINE_LOCK_FILE, "{}")
        self.assertTrue(tray._is_pipeline_running())

    def test_true_when_overnight_flag_present(self):
        self._write(tray.OVERNIGHT_FLAG, "")
        self.assertTrue(tray._is_pipeline_running())

    def test_exception_returns_false(self):
        with mock.patch.object(tray.os.path, "exists",
                               side_effect=OSError("boom")):
            self.assertFalse(tray._is_pipeline_running())


# --------------------------------------------------------------------------- #
# Menu status-header text builders
# --------------------------------------------------------------------------- #
class StatusTextTests(TrayTestBase):
    def test_listen_muted(self):
        self._write_hud(state="listening", mic_muted=True)
        self.assertEqual(tray._status_text_listen(), "● Listening: muted")

    def test_listen_standby(self):
        self._write_hud(state="standby")
        self.assertEqual(tray._status_text_listen(), "● Listening: standby")

    def test_listen_awake(self):
        self._write_hud(state="listening")
        self.assertEqual(tray._status_text_listen(), "● Listening: awake")

    def test_tts_speaking_by_state(self):
        self._write_hud(state="speaking")
        self.assertEqual(tray._status_text_tts(), "● TTS: speaking")

    def test_tts_speaking_by_amplitude(self):
        self._write_hud(state="idle", tts_amplitude=0.5)
        self.assertEqual(tray._status_text_tts(), "● TTS: speaking")

    def test_tts_quiet(self):
        self._write_hud(state="idle", tts_amplitude=0.0)
        self.assertEqual(tray._status_text_tts(), "● TTS: quiet")

    def test_queue_text(self):
        self._write(tray.TODO_FILE, "- [ ] a\n- [ ] b\n")
        self._bust_queue_cache()
        self.assertEqual(tray._status_text_queue(), "● Queue: 2 task(s)")

    def test_bambu_printing(self):
        self._write_hud(bambu_active=True)
        self.assertEqual(tray._status_text_bambu(), "● Bambu: printing")

    def test_bambu_idle(self):
        self._write_hud(bambu_active=False)
        self.assertEqual(tray._status_text_bambu(), "● Bambu: idle")


# --------------------------------------------------------------------------- #
# Task queue append + the todo-open callback
# --------------------------------------------------------------------------- #
class AppendQueuedTaskTests(TrayTestBase):
    def test_creates_file_with_header(self):
        tray._append_queued_task("build a thing")
        with open(tray.TODO_FILE, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("# JARVIS Task Queue", content)
        self.assertIn("- [ ]", content)
        self.assertIn("build a thing", content)

    def test_appends_to_existing(self):
        self._write(tray.TODO_FILE, "# JARVIS Task Queue\n\n- [ ] existing\n")
        tray._append_queued_task("second")
        with open(tray.TODO_FILE, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("existing", content)
        self.assertIn("second", content)

    def test_blank_text_is_noop(self):
        tray._append_queued_task("   ")
        self.assertFalse(os.path.exists(tray.TODO_FILE))

    def test_none_text_is_noop(self):
        tray._append_queued_task(None)
        self.assertFalse(os.path.exists(tray.TODO_FILE))

    def test_write_error_swallowed(self):
        with mock.patch("builtins.open", side_effect=OSError("ro fs")):
            tray._append_queued_task("x")  # must not raise


class OpenTodoCallbackTests(TrayTestBase):
    def test_creates_then_opens(self):
        with mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._on_open_todo(mock.Mock(), mock.Mock())
        self.assertTrue(os.path.exists(tray.TODO_FILE))
        sf.assert_called_once_with(tray.TODO_FILE)

    def test_opens_existing(self):
        self._write(tray.TODO_FILE, "# JARVIS Task Queue\n")
        with mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._on_open_todo(mock.Mock(), mock.Mock())
        sf.assert_called_once_with(tray.TODO_FILE)

    def test_create_failure_returns_without_open(self):
        with mock.patch("builtins.open", side_effect=OSError("ro")), \
             mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._on_open_todo(mock.Mock(), mock.Mock())
        sf.assert_not_called()

    def test_startfile_failure_swallowed(self):
        self._write(tray.TODO_FILE, "x")
        with mock.patch.object(tray.os, "startfile", create=True,
                               side_effect=OSError("no shell")):
            tray._on_open_todo(mock.Mock(), mock.Mock())  # must not raise


# --------------------------------------------------------------------------- #
# "Open X" explorer/shell callbacks
# --------------------------------------------------------------------------- #
class OpenPathCallbackTests(TrayTestBase):
    def test_open_path_success(self):
        with mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._open_path("C:/some/file", "label")
        sf.assert_called_once_with("C:/some/file")

    def test_open_path_failure_swallowed(self):
        with mock.patch.object(tray.os, "startfile", create=True,
                               side_effect=OSError("x")):
            tray._open_path("C:/some/file", "label")  # no raise

    def test_open_logs_makedirs_and_open(self):
        with mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._on_open_logs(mock.Mock(), mock.Mock())
        self.assertTrue(os.path.isdir(tray.LOGS_DIR))
        sf.assert_called_once_with(tray.LOGS_DIR)

    def test_open_logs_startfile_error_swallowed(self):
        with mock.patch.object(tray.os, "startfile", create=True,
                               side_effect=OSError("x")):
            tray._on_open_logs(mock.Mock(), mock.Mock())  # no raise

    def test_open_logs_makedirs_error_swallowed(self):
        # a makedirs failure (e.g. read-only volume) is swallowed; the callback
        # still attempts to open the folder afterwards.
        with mock.patch.object(tray.os, "makedirs", side_effect=OSError("ro")), \
             mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._on_open_logs(mock.Mock(), mock.Mock())  # no raise
        sf.assert_called_once_with(tray.LOGS_DIR)

    def test_open_project_folder(self):
        with mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._on_open_project_folder(mock.Mock(), mock.Mock())
        sf.assert_called_once_with(tray.PROJECT_DIR)

    def test_open_project_folder_error_swallowed(self):
        with mock.patch.object(tray.os, "startfile", create=True,
                               side_effect=OSError("x")):
            tray._on_open_project_folder(mock.Mock(), mock.Mock())

    def test_open_changelog_opens_the_release_notes(self):
        # CHANGELOG.md is the self-upgrade pipeline's log (stale since the last
        # overnight run); the release notes are the GitHub releases page.
        self._write(tray.CHANGELOG_FILE, "## v1.0.0 — 2026-01-01 00:00\n")
        with mock.patch("webbrowser.open", return_value=True) as wb, \
             mock.patch.object(tray, "_open_path") as op:
            tray._on_open_changelog(mock.Mock(), mock.Mock())
        wb.assert_called_once()
        self.assertTrue(wb.call_args.args[0].endswith("/jarvis/releases"))
        op.assert_not_called()

    def test_open_changelog_falls_back_to_the_file(self):
        self._write(tray.CHANGELOG_FILE, "## v1.0.0 — 2026-01-01 00:00\n")
        with mock.patch("webbrowser.open", return_value=False), \
             mock.patch.object(tray, "_open_path") as op:
            tray._on_open_changelog(mock.Mock(), mock.Mock())
        op.assert_called_once()
        self.assertEqual(op.call_args.args[0], tray.CHANGELOG_FILE)

    def test_open_changelog_absent_no_open(self):
        with mock.patch("webbrowser.open", side_effect=OSError("no browser")), \
             mock.patch.object(tray, "_open_path") as op:
            tray._on_open_changelog(mock.Mock(), mock.Mock())
        op.assert_not_called()

    def test_open_memory_file_primary(self):
        os.makedirs(os.path.dirname(tray.MEMORY_FACTS_FILE), exist_ok=True)
        self._write(tray.MEMORY_FACTS_FILE, "[]")
        with mock.patch.object(tray, "_open_path") as op:
            tray._on_open_memory_file(mock.Mock(), mock.Mock())
        self.assertEqual(op.call_args.args[0], tray.MEMORY_FACTS_FILE)

    def test_open_memory_file_legacy_fallback(self):
        legacy = os.path.join(tray.PROJECT_DIR, "memory.json")
        self._write(legacy, "{}")
        with mock.patch.object(tray, "_open_path") as op:
            tray._on_open_memory_file(mock.Mock(), mock.Mock())
        self.assertEqual(op.call_args.args[0], legacy)

    def test_open_memory_file_dir_fallback(self):
        # Neither primary nor legacy exists -> open the memory dir.
        with mock.patch.object(tray, "_open_path") as op:
            tray._on_open_memory_file(mock.Mock(), mock.Mock())
        self.assertEqual(op.call_args.args[0],
                         os.path.dirname(tray.MEMORY_FACTS_FILE))


# --------------------------------------------------------------------------- #
# Threaded callbacks that spawn helpers — assert they start a thread and the
# helper they target runs without error (thread target invoked directly).
# --------------------------------------------------------------------------- #
class ThreadedCallbackTests(TrayTestBase):
    def _run_thread_target(self, start_mock):
        """Pull the target= off the patched Thread and run it synchronously."""
        self.assertTrue(start_mock.called)
        _, kwargs = start_mock.call_args
        target = kwargs.get("target")
        self.assertIsNotNone(target, "Thread created without target=")
        target()

    def test_open_live_log_spawns_thread(self):
        with mock.patch.object(tray.threading, "Thread") as T:
            inst = T.return_value
            tray._on_open_live_log(mock.Mock(), mock.Mock())
        T.assert_called_once()
        inst.start.assert_called_once()
        self.assertTrue(T.call_args.kwargs.get("daemon"))

    def test_open_crashes_spawns_thread(self):
        with mock.patch.object(tray.threading, "Thread") as T:
            inst = T.return_value
            tray._on_open_crashes(mock.Mock(), mock.Mock())
        inst.start.assert_called_once()

    def test_show_dossier_spawns_thread_and_runs_subprocess(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run") as run:
            tray._on_show_dossier(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        run.assert_called_once()

    def test_about_spawns_thread_and_runs_subprocess(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run") as run:
            tray._on_about(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        run.assert_called_once()

    def test_summary_spawns_thread_and_runs_subprocess(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run") as run:
            tray._on_show_today_summary(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        run.assert_called_once()

    def test_dossier_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run",
                               side_effect=OSError("spawn fail")):
            tray._on_show_dossier(mock.Mock(), mock.Mock())
            self._run_thread_target(T)  # must not raise

    def test_about_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run",
                               side_effect=OSError("spawn fail")):
            tray._on_about(mock.Mock(), mock.Mock())
            self._run_thread_target(T)  # must not raise

    def test_summary_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run",
                               side_effect=OSError("spawn fail")):
            tray._on_show_today_summary(mock.Mock(), mock.Mock())
            self._run_thread_target(T)  # must not raise

    def test_queue_task_spawns_and_appends_result(self):
        proc = mock.Mock()
        proc.stdout = "do the dishes"
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run", return_value=proc):
            tray._on_queue_task(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        with open(tray.TODO_FILE, encoding="utf-8") as f:
            self.assertIn("do the dishes", f.read())

    def test_queue_task_empty_result_no_append(self):
        proc = mock.Mock()
        proc.stdout = "   "
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run", return_value=proc):
            tray._on_queue_task(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        self.assertFalse(os.path.exists(tray.TODO_FILE))

    def test_queue_task_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run",
                               side_effect=OSError("spawn fail")):
            tray._on_queue_task(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        self.assertFalse(os.path.exists(tray.TODO_FILE))


class SettingsCallbackTests(TrayTestBase):
    """The top-level "Settings…" item spawns a daemon thread targeting
    _open_settings_window on the window's first tab."""

    def test_open_settings_targets_first_tab(self):
        with mock.patch.object(tray.threading, "Thread") as T:
            inst = T.return_value
            tray._on_open_settings(mock.Mock(), mock.Mock())
        inst.start.assert_called_once()
        self.assertEqual(T.call_args.kwargs.get("target"),
                         tray._open_settings_window)
        self.assertEqual(T.call_args.kwargs.get("args"), ("",))


# --------------------------------------------------------------------------- #
# Helper shell-outs: live log viewer, event viewer, settings window
# --------------------------------------------------------------------------- #
class OpenLiveLogViewerTests(TrayTestBase):
    def test_missing_script_no_popen(self):
        with mock.patch.object(tray.subprocess, "Popen") as P:
            tray._open_live_log_viewer()
        P.assert_not_called()

    def test_present_script_spawns_powershell(self):
        self._write(tray.SHOW_LOG_PS1, "echo hi")
        with mock.patch.object(tray.subprocess, "Popen") as P:
            tray._open_live_log_viewer()
        P.assert_called_once()
        argv = P.call_args.args[0]
        self.assertEqual(argv[0], "powershell.exe")
        self.assertIn(tray.SHOW_LOG_PS1, argv)

    def test_popen_error_swallowed(self):
        self._write(tray.SHOW_LOG_PS1, "echo hi")
        with mock.patch.object(tray.subprocess, "Popen",
                               side_effect=OSError("no shell")):
            tray._open_live_log_viewer()  # no raise


class OpenEventViewerTests(TrayTestBase):
    def test_startfile_called(self):
        with mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._open_event_viewer_crashes()
        sf.assert_called_once_with("eventvwr.msc")

    def test_error_swallowed(self):
        with mock.patch.object(tray.os, "startfile", create=True,
                               side_effect=OSError("x")):
            tray._open_event_viewer_crashes()  # no raise


class _FakeProc:
    """Popen stand-in: .wait() returns ``rc`` or raises TimeoutExpired."""

    def __init__(self, rc=None, pid=4242):
        self.rc = rc
        self.pid = pid
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.rc is None:
            raise subprocess.TimeoutExpired("settings", timeout)
        return self.rc


class OpenSettingsWindowTests(TrayTestBase):
    def _launch(self, tab, proc=None, **patches):
        self._write(tray.SETTINGS_WINDOW, "# settings")
        proc = proc or _FakeProc(rc=None)
        with mock.patch.object(tray.subprocess, "Popen",
                               return_value=proc) as P, \
             mock.patch.object(tray, "_notify", create=True) as note:
            tray._open_settings_window(tab)
        return P, note, proc

    def test_spawns_window_when_present(self):
        P, note, _ = self._launch("voice")
        P.assert_called_once()
        argv = P.call_args.args[0]
        self.assertIn("--tab", argv)
        self.assertIn("voice", argv)
        note.assert_not_called()          # still running after the grace = OK

    def test_launch_is_a_module_run_from_the_project_root(self):
        # THE Settings bug: `pythonw tools\settings_window.py` put tools\ on
        # sys.path[0] and gave no cwd, so the window's `from core import …`
        # failed. A module launch from cwd=PROJECT_DIR resolves core.
        P, _, _ = self._launch("ai")
        argv = P.call_args.args[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:3], ["-m", "tools.settings_window"])
        self.assertEqual(argv[3:], ["--tab", "ai"])
        self.assertEqual(os.path.normcase(P.call_args.kwargs.get("cwd") or ""),
                         os.path.normcase(tray.PROJECT_DIR))
        self.assertEqual(argv, tray._settings_launch_argv("ai"))

    def test_window_output_goes_to_the_settings_log(self):
        P, _, _ = self._launch("voice")
        kw = P.call_args.kwargs
        self.assertIs(kw.get("stderr"), subprocess.STDOUT)
        out = kw.get("stdout")
        self.assertTrue(hasattr(out, "write"),
                        "the window's stdout must go to a log file, not nowhere")
        self.assertEqual(os.path.normcase(out.name),
                         os.path.normcase(tray.SETTINGS_LOG_FILE))
        with open(tray.SETTINGS_LOG_FILE, encoding="utf-8") as f:
            self.assertIn("launch tab=voice", f.read())

    def test_window_that_dies_at_start_raises_a_balloon(self):
        log_path = tray.SETTINGS_LOG_FILE

        class _DyingProc(_FakeProc):
            def wait(self, timeout=None):
                # The child's traceback lands in the log AFTER the launch banner.
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("Traceback (most recent call last):\n"
                            "ModuleNotFoundError: No module named 'core'\n")
                return super().wait(timeout)
        P, note, proc = self._launch("ai", proc=_DyingProc(rc=1))
        note.assert_called_once()
        msg = note.call_args.args[0]
        self.assertIn("exit 1", msg)
        self.assertIn("No module named 'core'", msg)
        self.assertIn("settings_window.log", msg)
        self.assertEqual(proc.wait_timeouts, [tray.SETTINGS_LAUNCH_GRACE_S])

    def test_clean_exit_is_not_an_error(self):
        _, note, _ = self._launch("ai", proc=_FakeProc(rc=0))
        note.assert_not_called()

    def test_no_tab_arg_when_blank(self):
        P, _, _ = self._launch("")
        argv = P.call_args.args[0]
        self.assertNotIn("--tab", argv)

    def test_popen_error_falls_through_to_json(self):
        self._write(tray.SETTINGS_WINDOW, "# settings")
        fallback = os.path.join(tray.DATA_DIR, "user_settings.json")
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(fallback, "{}")
        with mock.patch.object(tray.subprocess, "Popen",
                               side_effect=OSError("spawn fail")), \
             mock.patch.object(tray, "_notify", create=True) as note, \
             mock.patch.object(tray, "_open_path") as op:
            tray._open_settings_window("ai")
        op.assert_called_once()
        self.assertEqual(op.call_args.args[0], fallback)
        # …and the failure is no longer silent.
        note.assert_called_once()
        self.assertIn("spawn fail", note.call_args.args[0])

    def test_fallback_to_user_settings_json(self):
        # No settings window installed; fall back to opening the JSON.
        fallback = os.path.join(tray.DATA_DIR, "user_settings.json")
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(fallback, "{}")
        with mock.patch.object(tray, "_open_path") as op:
            tray._open_settings_window("advanced")
        op.assert_called_once_with(fallback, "user_settings.json")

    def test_no_window_no_json_is_noop(self):
        # Nothing installed at all — must not raise and must not open anything.
        with mock.patch.object(tray, "_open_path") as op, \
             mock.patch.object(tray.subprocess, "Popen") as P:
            tray._open_settings_window("voice")
        op.assert_not_called()
        P.assert_not_called()


# --------------------------------------------------------------------------- #
# About-dialog text builders
# --------------------------------------------------------------------------- #
class VersionAndUptimeTests(TrayTestBase):
    def setUp(self):
        super().setUp()
        # About's Commit line shells out to git; tests that want it patch it.
        p = mock.patch.object(tray, "_git_commit", return_value="", create=True)
        p.start()
        self.addCleanup(p.stop)

    def test_version_parsed_from_changelog(self):
        self._write(tray.CHANGELOG_FILE,
                    "# Changelog\n\n## v1.2.3 — 2026-05-28 22:33\n- did stuff\n")
        ver, at = tray._read_version_and_upgrade()
        self.assertEqual(ver, "v1.2.3")
        self.assertEqual(at, "2026-05-28 22:33")

    def test_version_falls_back_to_version_json(self):
        # No changelog header -> read data/version.json.
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.VERSION_FILE,
                    json.dumps({"version": "9.9.9", "last_upgrade_at": "yesterday"}))
        ver, at = tray._read_version_and_upgrade()
        self.assertEqual(ver, "v9.9.9")
        self.assertEqual(at, "yesterday")

    def test_version_unknown_when_nothing(self):
        ver, at = tray._read_version_and_upgrade()
        self.assertEqual(ver, "unknown")
        self.assertEqual(at, "unknown")

    def _no_psutil_start(self):
        return mock.patch.object(tray, "_parent_started_at", return_value=0.0, create=True)

    def test_uptime_from_the_live_parent_process(self):
        # The authority: the OS start time of the JARVIS this tray belongs to.
        tray._parent_pid[0] = 4321
        with mock.patch.object(tray, "_parent_started_at", create=True,
                               return_value=tray.time.time() - 600):
            self.assertGreaterEqual(tray._read_uptime_seconds(), 590)

    def test_uptime_from_our_own_instance_entry(self):
        tray._parent_pid[0] = 4321
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({
            "4321": {"pid": 4321, "role": "prod",
                     "started_at": tray.time.time() - 120},
        }))
        with self._no_psutil_start():
            self.assertGreaterEqual(tray._read_uptime_seconds(), 100)

    def test_uptime_ignores_a_dead_first_prod_entry(self):
        # THE About bug: instances.json keeps dead PIDs and the FIRST prod entry
        # (a JARVIS from weeks ago) was reported as the uptime.
        tray._parent_pid[0] = 4321
        now = tray.time.time()
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({
            "111": {"pid": 111, "role": "prod", "started_at": now - 30 * 86400},
            "4321": {"pid": 4321, "role": "prod", "started_at": now - 300},
        }))
        with self._no_psutil_start():
            up = tray._read_uptime_seconds()
        self.assertGreaterEqual(up, 290)
        self.assertLess(up, 3600)

    def test_uptime_unknown_rather_than_a_stranger(self):
        # Parent known, but no record of it anywhere: say unknown (0.0), never
        # borrow another instance's start time.
        tray._parent_pid[0] = 4321
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({
            "x": {"pid": 111, "role": "prod", "started_at": tray.time.time() - 60},
        }))
        with self._no_psutil_start():
            self.assertEqual(tray._read_uptime_seconds(), 0.0)

    def test_uptime_skips_non_dict_entries(self):
        # A non-dict instance entry must be skipped (the `continue` branch) and
        # the scan must still find the dict entry that follows.
        tray._parent_pid[0] = 77
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({
            "bad": "not-a-dict",
            "77": {"role": "prod", "started_at": tray.time.time() - 70},
        }))
        with self._no_psutil_start():
            self.assertGreaterEqual(tray._read_uptime_seconds(), 50)

    def test_uptime_from_published_boot_when_it_is_ours(self):
        tray._parent_pid[0] = 55
        self._write_hud(boot_started_at=tray.time.time() - 45, jarvis_pid=55)
        with self._no_psutil_start():
            self.assertGreaterEqual(tray._read_uptime_seconds(), 40)
        self._write_hud(boot_started_at=tray.time.time() - 45, jarvis_pid=56)
        with self._no_psutil_start():
            self.assertEqual(tray._read_uptime_seconds(), 0.0)

    def test_uptime_falls_back_to_hud_when_instances_lack_started_at(self):
        # instances.json present but no usable started_at -> hud boot fallback.
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({"x": {"role": "prod"}}))
        self._write_hud(boot_started_at=tray.time.time() - 40)
        self.assertGreaterEqual(tray._read_uptime_seconds(), 30)

    def test_uptime_falls_back_to_hud_boot(self):
        self._write_hud(boot_started_at=tray.time.time() - 30)
        self.assertGreaterEqual(tray._read_uptime_seconds(), 20)

    def test_uptime_zero_when_nothing(self):
        self.assertEqual(tray._read_uptime_seconds(), 0.0)

    def test_version_changelog_read_error_swallowed(self):
        # CHANGELOG.md exists but can't be opened -> outer except swallows it and
        # the function falls through to the version.json fallback (also absent),
        # yielding "unknown".
        self._write(tray.CHANGELOG_FILE, "## v1.0.0 — 2026-01-01 00:00\n")
        with mock.patch("builtins.open", side_effect=OSError("locked")):
            ver, at = tray._read_version_and_upgrade()
        self.assertEqual(ver, "unknown")
        self.assertEqual(at, "unknown")

    def test_version_json_load_error_swallowed(self):
        # No changelog header, version.json present but malformed -> the
        # fallback's except fires and version stays "unknown".
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.VERSION_FILE, "{ not valid json")
        ver, at = tray._read_version_and_upgrade()
        self.assertEqual(ver, "unknown")

    def test_uptime_instances_parse_error_falls_back_to_hud(self):
        # instances.json present but malformed -> the parse except fires and the
        # function falls back to the hud boot timestamp.
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, "{ not valid json")
        self._write_hud(boot_started_at=tray.time.time() - 45)
        self.assertGreaterEqual(tray._read_uptime_seconds(), 30)

    def test_uptime_hud_boot_non_numeric_returns_zero(self):
        # a non-numeric hud boot_started_at makes float() raise -> the final
        # except returns 0.0 rather than propagating.
        with mock.patch.object(tray, "_read_hud_state",
                               return_value={"boot_started_at": "not-a-number"}):
            self.assertEqual(tray._read_uptime_seconds(), 0.0)

    def test_format_uptime_variants(self):
        self.assertEqual(tray._format_uptime(0), "0m")
        self.assertEqual(tray._format_uptime(90), "1m")
        self.assertEqual(tray._format_uptime(3661), "1h 1m")
        self.assertEqual(tray._format_uptime(90061), "1d 1h 1m")
        self.assertEqual(tray._format_uptime(-50), "0m")

    # -- release version (single source == GitHub) --------------------------
    def test_release_version_read_from_version_file(self):
        self._write(tray.RELEASE_VERSION_FILE, "1.0.0-beta.1\n")
        self.assertEqual(tray._read_release_version(), "1.0.0-beta.1")

    def test_release_version_unknown_when_missing(self):
        # VERSION file absent (temp dir) -> defensive 'unknown'.
        self.assertEqual(tray._read_release_version(), "unknown")

    def test_release_version_blank_file_is_unknown(self):
        self._write(tray.RELEASE_VERSION_FILE, "   \n")
        self.assertEqual(tray._read_release_version(), "unknown")

    def test_about_lines_structure(self):
        # PRIMARY 'Version:' line is the RELEASE version (== GitHub + git tag),
        # NOT the self-upgrade pipeline's CHANGELOG counter — that's the drift
        # the user caught (tray said v1.0.17, GitHub said 1.0.0-beta.1).
        self._write(tray.RELEASE_VERSION_FILE, "1.0.0-beta.1\n")
        self._write(tray.CHANGELOG_FILE, "## v2.0.0 — 2026-06-01 10:00\n")
        lines = tray._about_lines()
        self.assertEqual(lines[0], "J.A.R.V.I.S.")
        joined = "\n".join(lines)
        self.assertIn("Version:       1.0.0-beta.1", joined)
        # The pipeline counter is still surfaced, but clearly RELABELLED so it
        # can't be mistaken for the release version.
        self.assertIn("Upgrade build: v2.0.0", joined)
        self.assertIn("Uptime:", joined)
        # The pipeline counter must NOT appear on the primary Version line.
        self.assertNotIn("Version:       v2.0.0", joined)

    def test_about_lines_fresh_clone_hides_pipeline_build(self):
        # A fresh share has the tracked VERSION file but no pipeline history
        # (data/ + CHANGELOG upgrade entries are gitignored / absent), so the
        # build + last-upgrade lines are hidden — no confusing 'unknown'.
        self._write(tray.RELEASE_VERSION_FILE, "1.0.0-beta.1\n")
        joined = "\n".join(tray._about_lines())
        self.assertIn("Version:       1.0.0-beta.1", joined)
        self.assertNotIn("Upgrade build", joined)
        self.assertNotIn("Last upgrade", joined)

    def test_about_lines_build_hidden_when_equal_to_release(self):
        # If the CHANGELOG top entry == the release version, don't show a
        # redundant 'Upgrade build' line (matches with or without a 'v').
        self._write(tray.RELEASE_VERSION_FILE, "1.0.0-beta.1\n")
        self._write(tray.CHANGELOG_FILE, "## v1.0.0-beta.1 — 2026-06-01 10:00\n")
        joined = "\n".join(tray._about_lines())
        self.assertNotIn("Upgrade build", joined)

    # -- Last updated = the release's git date, not the stale CHANGELOG ------
    # The pipeline's CHANGELOG header is the only thing the old line read, and
    # no git release writes it: About said "Last upgrade: 2026-05-30 07:03"
    # on 2.0.159 (2026-10-02).
    def _git_release(self, version, when):
        import shutil
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        env = dict(os.environ, GIT_COMMITTER_DATE=f"{int(when)} +0000",
                   GIT_AUTHOR_DATE=f"{int(when)} +0000")

        def g(*args):
            subprocess.run(["git", "-C", self.dir, *args], check=True,
                           capture_output=True, text=True, env=env)
        g("init", "-q")
        g("config", "user.email", "t@t.com")
        g("config", "user.name", "t")
        self._write(tray.RELEASE_VERSION_FILE, version + "\n")
        g("add", "VERSION")
        g("commit", "-qm", f"v{version}")
        g("tag", f"v{version}")

    def test_about_last_updated_is_the_git_release(self):
        from datetime import datetime
        released = datetime(2026, 9, 28, 10, 0)
        self._git_release("2.0.159", released.timestamp())
        self._write(tray.CHANGELOG_FILE, "## v1.0.17 — 2026-05-30 07:03\n")
        joined = "\n".join(tray._about_lines())
        self.assertIn("Last updated:  2026-09-28 10:00", joined)
        self.assertNotIn("Last upgrade", joined)
        # The pipeline counter keeps its own date, on its own line.
        self.assertIn("Upgrade build: v1.0.17, last run 2026-05-30 07:03", joined)

    def test_about_last_updated_outside_a_checkout_uses_version_mtime(self):
        from datetime import datetime
        self._write(tray.RELEASE_VERSION_FILE, "2.0.159\n")
        when = datetime(2026, 9, 27, 8, 15).timestamp()
        os.utime(tray.RELEASE_VERSION_FILE, (when, when))
        self.assertIn("Last updated:  2026-09-27 08:15",
                      "\n".join(tray._about_lines()))

    def test_about_last_updated_takes_a_newer_pipeline_run(self):
        from datetime import datetime
        self._git_release("2.0.159", datetime(2026, 9, 28, 10, 0).timestamp())
        self._write(tray.CHANGELOG_FILE, "## v1.0.18 — 2026-09-29 03:10\n")
        self.assertIn("Last updated:  2026-09-29 03:10",
                      "\n".join(tray._about_lines()))

    def test_about_shows_the_running_version_and_flags_a_newer_disk(self):
        # After a `git pull` without a restart the VERSION file is ahead of the
        # process. About reports what is RUNNING and says a restart applies it.
        tray._parent_pid[0] = 900
        self._write(tray.RELEASE_VERSION_FILE, "2.0.141\n")
        self._write_hud(jarvis_version="2.0.140", jarvis_pid=900)
        with mock.patch.object(tray, "_parent_started_at", create=True,
                               return_value=tray.time.time() - 7200):
            joined = "\n".join(tray._about_lines())
        self.assertIn("Version:       2.0.140", joined)
        self.assertIn("On disk:       2.0.141 (restart to apply)", joined)
        self.assertIn("Uptime:        2h 0m", joined)

    def test_about_ignores_another_jarvis_version(self):
        tray._parent_pid[0] = 900
        self._write(tray.RELEASE_VERSION_FILE, "2.0.141\n")
        self._write_hud(jarvis_version="1.9.0", jarvis_pid=12)
        joined = "\n".join(tray._about_lines())
        self.assertIn("Version:       2.0.141", joined)
        self.assertNotIn("1.9.0", joined)
        self.assertIn("Uptime:        unknown", joined)

    def test_about_shows_the_commit(self):
        self._write(tray.RELEASE_VERSION_FILE, "2.0.141\n")
        with mock.patch.object(tray, "_git_commit", return_value="b0dcd1c"):
            self.assertIn("Commit:        b0dcd1c",
                          "\n".join(tray._about_lines()))

    def test_git_commit_failure_is_blank(self):
        real = _REAL_GIT_COMMIT
        with mock.patch.object(tray.subprocess, "run",
                               side_effect=OSError("no git")):
            self.assertEqual(real(), "")
        fake = mock.Mock(returncode=128, stdout="")
        with mock.patch.object(tray.subprocess, "run", return_value=fake):
            self.assertEqual(real(), "")
        ok = mock.Mock(returncode=0, stdout="abc1234\n")
        with mock.patch.object(tray.subprocess, "run", return_value=ok):
            self.assertEqual(real(), "abc1234")

    def test_parent_started_at_reads_the_os(self):
        tray._parent_pid[0] = os.getpid()
        started = tray._parent_started_at()
        if tray._HAS_PSUTIL:
            self.assertGreater(started, 0)
            self.assertLessEqual(started, tray.time.time())
        tray._parent_pid[0] = 0
        self.assertEqual(tray._parent_started_at(), 0.0)

    def test_about_dialog_is_told_which_jarvis(self):
        tray._parent_pid[0] = 31337
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray, "_tracked_dialog_run") as run:
            tray._on_about(mock.Mock(), mock.Mock())
            T.call_args.kwargs["target"]()
        argv = run.call_args.args[0]
        self.assertIn("--about-dialog", argv)
        self.assertEqual(argv[argv.index("--parent-pid") + 1], "31337")


# --------------------------------------------------------------------------- #
# Dossier text builder
# --------------------------------------------------------------------------- #
class DossierLinesTests(TrayTestBase):
    def _facts_path(self):
        os.makedirs(os.path.dirname(tray.MEMORY_FACTS_FILE), exist_ok=True)
        return tray.MEMORY_FACTS_FILE

    def test_no_file(self):
        lines = tray._dossier_lines()
        self.assertIn("(no memory file found yet)", lines)

    def test_list_of_fact_dicts(self):
        self._write(self._facts_path(),
                    json.dumps([{"text": "likes coffee"}, {"fact": "has a dog"}]))
        lines = tray._dossier_lines()
        joined = "\n".join(lines)
        self.assertIn("2 fact(s) on file.", joined)
        self.assertIn("likes coffee", joined)
        self.assertIn("has a dog", joined)

    def test_facts_wrapped_in_dict(self):
        self._write(self._facts_path(),
                    json.dumps({"facts": [{"content": "wakes at 7"}]}))
        self.assertIn("wakes at 7", "\n".join(tray._dossier_lines()))

    def test_generic_keyvalue_dump(self):
        self._write(self._facts_path(),
                    json.dumps({"name": "Tony", "city": "NYC"}))
        joined = "\n".join(tray._dossier_lines())
        self.assertIn("name: Tony", joined)
        self.assertIn("city: NYC", joined)

    def test_empty_list_message(self):
        self._write(self._facts_path(), "[]")
        self.assertIn("(no facts learned yet)", tray._dossier_lines())

    def test_unreadable_file(self):
        self._write(self._facts_path(), "{bad json")
        self.assertTrue(any("could not read memory" in ln
                            for ln in tray._dossier_lines()))

    def test_legacy_fallback_path(self):
        legacy = os.path.join(tray.PROJECT_DIR, "memory.json")
        self._write(legacy, json.dumps([{"text": "legacy fact"}]))
        self.assertIn("legacy fact", "\n".join(tray._dossier_lines()))

    def test_long_fact_truncated(self):
        long_text = "x" * 300
        self._write(self._facts_path(), json.dumps([{"text": long_text}]))
        joined = "\n".join(tray._dossier_lines())
        self.assertIn("…", joined)
        self.assertNotIn("x" * 200, joined)

    def test_plain_string_facts(self):
        self._write(self._facts_path(), json.dumps(["just a string fact"]))
        self.assertIn("just a string fact", "\n".join(tray._dossier_lines()))


# --------------------------------------------------------------------------- #
# Today's-summary text builder
# --------------------------------------------------------------------------- #
class TodaySummaryLinesTests(TrayTestBase):
    def test_header_has_date(self):
        os.makedirs(tray.LOGS_DIR, exist_ok=True)
        lines = tray._today_summary_lines()
        self.assertTrue(lines[0].startswith("J.A.R.V.I.S."))

    def test_counts_sessions_today(self):
        os.makedirs(tray.LOGS_DIR, exist_ok=True)
        today = tray.time.strftime("%Y-%m-%d")
        self._write(os.path.join(tray.LOGS_DIR, f"session_{today}_001.log"), "x" * 2048)
        self._write(os.path.join(tray.LOGS_DIR, f"session_{today}_002.log"), "y" * 1024)
        # An old session that should NOT be counted.
        self._write(os.path.join(tray.LOGS_DIR, "session_1999-01-01_001.log"), "z")
        joined = "\n".join(tray._today_summary_lines())
        self.assertIn("Sessions today:   2", joined)

    def test_no_logs_dir(self):
        # LOGS_DIR doesn't exist.
        joined = "\n".join(tray._today_summary_lines())
        self.assertIn("logs/ not found", joined)

    def test_task_counts_and_completions(self):
        os.makedirs(tray.LOGS_DIR, exist_ok=True)
        today = tray.time.strftime("%Y-%m-%d")
        self._write(tray.TODO_FILE,
                    f"- [ ] pending one\n- [ ] pending two\n"
                    f"- [x] **{today}** finished alpha\n"
                    f"- [x] **1999-01-01** old done\n")
        joined = "\n".join(tray._today_summary_lines())
        self.assertIn("Pending tasks:    2", joined)
        self.assertIn("Completed today:  1", joined)
        self.assertIn("finished alpha", joined)
        self.assertIn("Recent completions:", joined)

    def test_todo_read_error_branch(self):
        os.makedirs(tray.LOGS_DIR, exist_ok=True)
        # File exists but open() raises -> the except branch appends an error.
        self._write(tray.TODO_FILE, "- [ ] a\n")
        real_open = open

        def flaky_open(path, *a, **k):
            if os.path.abspath(path) == os.path.abspath(tray.TODO_FILE):
                raise OSError("locked")
            return real_open(path, *a, **k)

        with mock.patch("builtins.open", side_effect=flaky_open):
            joined = "\n".join(tray._today_summary_lines())
        self.assertIn("Todo:", joined)

    def test_session_getsize_failure_is_skipped(self):
        # A getsize() that raises for one session file is swallowed (the inner
        # bare except) — the session is still counted, total bytes just omits it.
        os.makedirs(tray.LOGS_DIR, exist_ok=True)
        today = tray.time.strftime("%Y-%m-%d")
        self._write(os.path.join(tray.LOGS_DIR, f"session_{today}_001.log"), "data")
        with mock.patch.object(tray.os.path, "getsize",
                               side_effect=OSError("stat fail")):
            joined = "\n".join(tray._today_summary_lines())
        self.assertIn("Sessions today:   1", joined)

    def test_sessions_listdir_failure_branch(self):
        # os.listdir blowing up drives the outer sessions `except` -> error line.
        os.makedirs(tray.LOGS_DIR, exist_ok=True)
        with mock.patch.object(tray.os, "listdir",
                               side_effect=OSError("io error")):
            joined = "\n".join(tray._today_summary_lines())
        self.assertIn("Sessions today:   (error:", joined)


# --------------------------------------------------------------------------- #
# Subprocess dialog entry points (run on the dialog subprocess's main thread)
# --------------------------------------------------------------------------- #
class DialogEntryPointTests(TrayTestBase):
    def test_queue_dialog_no_tk_returns_2(self):
        with mock.patch.object(tray, "_HAS_TK", False):
            self.assertEqual(tray._run_queue_task_dialog(), 2)

    def test_summary_dialog_no_tk_returns_2(self):
        with mock.patch.object(tray, "_HAS_TK", False):
            self.assertEqual(tray._run_summary_dialog(), 2)

    def test_about_dialog_no_tk_returns_2(self):
        with mock.patch.object(tray, "_HAS_TK", False):
            self.assertEqual(tray._run_about_dialog(), 2)

    def test_dossier_dialog_no_tk_returns_2(self):
        with mock.patch.object(tray, "_HAS_TK", False):
            self.assertEqual(tray._run_dossier_dialog(), 2)

    def test_queue_dialog_with_tk_prints_text(self):
        fake_tk, fake_sd = self._fake_tk(askstring_result="walk the dog")
        out = io.StringIO()
        with mock.patch.object(tray, "_HAS_TK", True), \
             mock.patch.object(tray, "tk", fake_tk, create=True), \
             mock.patch.object(tray, "simpledialog", fake_sd, create=True), \
             mock.patch.object(sys, "stdout", out):
            rc = tray._run_queue_task_dialog()
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue(), "walk the dog")

    def test_queue_dialog_cancelled_prints_nothing(self):
        fake_tk, fake_sd = self._fake_tk(askstring_result=None)
        out = io.StringIO()
        with mock.patch.object(tray, "_HAS_TK", True), \
             mock.patch.object(tray, "tk", fake_tk, create=True), \
             mock.patch.object(tray, "simpledialog", fake_sd, create=True), \
             mock.patch.object(sys, "stdout", out):
            rc = tray._run_queue_task_dialog()
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue(), "")

    def test_summary_dialog_with_tk(self):
        os.makedirs(tray.LOGS_DIR, exist_ok=True)
        fake_tk, _ = self._fake_tk()
        with mock.patch.object(tray, "_HAS_TK", True), \
             mock.patch.object(tray, "tk", fake_tk, create=True):
            self.assertEqual(tray._run_summary_dialog(), 0)
        self.assertTrue(fake_tk.Tk.return_value.mainloop.called)

    def test_about_dialog_with_tk(self):
        fake_tk, _ = self._fake_tk()
        with mock.patch.object(tray, "_HAS_TK", True), \
             mock.patch.object(tray, "tk", fake_tk, create=True):
            self.assertEqual(tray._run_about_dialog(), 0)
        self.assertTrue(fake_tk.Tk.return_value.mainloop.called)

    def test_dossier_dialog_with_tk(self):
        fake_tk, _ = self._fake_tk()
        with mock.patch.object(tray, "_HAS_TK", True), \
             mock.patch.object(tray, "tk", fake_tk, create=True):
            self.assertEqual(tray._run_dossier_dialog(), 0)
        self.assertTrue(fake_tk.Tk.return_value.mainloop.called)

    def test_summary_dialog_text_widget_fallback_to_label(self):
        self._text_widget_fallback(tray._run_summary_dialog)

    def test_about_dialog_text_widget_fallback_to_label(self):
        self._text_widget_fallback(tray._run_about_dialog)

    def test_dossier_dialog_text_widget_fallback_to_label(self):
        self._text_widget_fallback(tray._run_dossier_dialog)

    def _text_widget_fallback(self, dialog_fn):
        # tk.Text() raising drives the `except -> tk.Label(...)` simple-layout
        # fallback inside every dialog builder. Still returns 0 cleanly.
        fake_tk, _ = self._fake_tk(text_raises=True)
        with mock.patch.object(tray, "_HAS_TK", True), \
             mock.patch.object(tray, "tk", fake_tk, create=True):
            self.assertEqual(dialog_fn(), 0)
        fake_tk.Label.assert_called()  # the fallback widget was used

    def test_dialog_root_destroy_error_swallowed(self):
        # The finally-block root.destroy() raising must not surface, for every
        # dialog builder that has one.
        for dialog_fn, extra in (
            (tray._run_about_dialog, {}),
            (tray._run_summary_dialog, {}),
            (tray._run_dossier_dialog, {}),
            (tray._run_queue_task_dialog, {"askstring_result": "x"}),
        ):
            fake_tk, fake_sd = self._fake_tk(**extra)
            fake_tk.Tk.return_value.destroy.side_effect = RuntimeError("gone")
            with mock.patch.object(tray, "_HAS_TK", True), \
                 mock.patch.object(tray, "tk", fake_tk, create=True), \
                 mock.patch.object(tray, "simpledialog", fake_sd, create=True), \
                 mock.patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(dialog_fn(), 0, dialog_fn.__name__)

    # -- fake tkinter ------------------------------------------------------ #
    def _fake_tk(self, askstring_result="", text_raises=False):
        """Build a stand-in `tk` module + `simpledialog` whose widgets are all
        Mocks. Widget constructors (Text/Label/Button/Frame/Scrollbar) return
        Mocks so .pack()/.insert()/.configure() are no-ops. mainloop returns at
        once so the dialog 'closes' immediately. When ``text_raises`` is set the
        tk.Text constructor raises, forcing the tk.Label simple-layout path."""
        fake_tk = mock.MagicMock(name="tk")
        root = fake_tk.Tk.return_value
        root.mainloop.return_value = None
        if text_raises:
            fake_tk.Text.side_effect = RuntimeError("no Text widget")
        fake_sd = mock.MagicMock(name="simpledialog")
        fake_sd.askstring.return_value = askstring_result
        return fake_tk, fake_sd


# --------------------------------------------------------------------------- #
# _on_quit + _animate (the polling loop)
# --------------------------------------------------------------------------- #
class QuitTests(TrayTestBase):
    def test_quit_sets_stop_and_stops_icon(self):
        icon = mock.Mock()
        with mock.patch.object(tray, "_notify", create=True) as note:
            tray._on_quit(icon, mock.Mock())      # first click only arms
        self.assertFalse(tray._stop_event.is_set())
        icon.stop.assert_not_called()
        # The balloon tells the owner the way back (voice show_tray).
        self.assertIn("show the tray icon", note.call_args.args[0])
        tray._on_quit(icon, mock.Mock())          # second click quits
        self.assertTrue(tray._stop_event.is_set())
        icon.stop.assert_called_once()

    def test_quit_icon_stop_error_swallowed(self):
        icon = mock.Mock()
        icon.stop.side_effect = RuntimeError("already stopped")
        with mock.patch.object(tray, "_notify", create=True):
            tray._on_quit(icon, mock.Mock())
            tray._on_quit(icon, mock.Mock())  # must not raise
        self.assertTrue(tray._stop_event.is_set())


class AnimateTests(TrayTestBase):
    def _one_shot_event(self):
        """An Event whose .wait() flips it set, so _animate runs exactly one
        loop iteration then exits on the next condition check."""
        ev = threading.Event()
        real_wait = ev.wait

        def wait(timeout=None):
            ev.set()
            return real_wait(0)

        ev.wait = wait
        return ev

    def test_single_iteration_updates_icon(self):
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state",
                               return_value={"state": "speaking",
                                             "tts_amplitude": 0.5,
                                             "bambu_active": True,
                                             "tray_ready_pid": 1,
                                             "overnight_upgrade_enabled": True}), \
             mock.patch.object(tray, "_count_pending_tasks", return_value=4):
            tray._animate(icon)
        self.assertIsNotNone(icon.icon)
        # Plain-English tooltip (P2): what the icon shows, in words.
        self.assertIn("Listening", icon.title)
        self.assertIn("speaking", icon.title)
        self.assertIn("4 queued", icon.title)
        self.assertIn("printing", icon.title)

    def test_title_reflects_muted_and_standby(self):
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state",
                               return_value={"state": "standby",
                                             "mic_muted": True,
                                             "tray_ready_pid": 1}), \
             mock.patch.object(tray, "_count_pending_tasks", return_value=0):
            tray._animate(icon)
        self.assertIn("Mic muted", icon.title)
        self.assertNotIn("printing", icon.title)

    def test_title_standby_label_when_not_muted(self):
        # Non-muted standby must reach the "standby" listen label branch.
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state",
                               return_value={"state": "sleeping",
                                             "tray_ready_pid": 1}), \
             mock.patch.object(tray, "_count_pending_tasks", return_value=0):
            tray._animate(icon)
        self.assertIn("Paused", icon.title)
        self.assertNotIn("speaking", icon.title)

    def test_title_says_starting_until_the_drainer_is_up(self):
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state", return_value={}):
            tray._animate(icon)
        self.assertIn("Starting", icon.title)

    def test_parent_dead_stops_immediately(self):
        icon = mock.Mock()
        with mock.patch.object(tray, "_parent_alive", return_value=False):
            tray._animate(icon)
        icon.stop.assert_called_once()
        # No render happened because we bailed before the icon update.
        self.assertFalse(isinstance(icon.icon, type(tray._render_icon("idle", 0))))

    def test_parent_dead_icon_stop_error_swallowed(self):
        # On a dead parent, icon.stop() raising must not propagate.
        icon = mock.Mock()
        icon.stop.side_effect = RuntimeError("stop failed")
        with mock.patch.object(tray, "_parent_alive", return_value=False):
            tray._animate(icon)  # must not raise
        icon.stop.assert_called_once()

    def test_render_exception_does_not_kill_loop(self):
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state", return_value={}), \
             mock.patch.object(tray, "_count_pending_tasks", return_value=0), \
             mock.patch.object(tray, "_render_icon",
                               side_effect=RuntimeError("render boom")):
            # The inner try/except swallows the render failure; the loop then
            # exits via the one-shot event. Must not raise.
            tray._animate(icon)
        self.assertTrue(ev.is_set())

    def test_outer_exception_path_logs_and_continues(self):
        # Make _read_hud_state raise so the OUTER try/except fires; its
        # _stop_event.wait must still set the event and end the loop.
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state",
                               side_effect=RuntimeError("hud boom")):
            tray._animate(icon)
        self.assertTrue(ev.is_set())


# --------------------------------------------------------------------------- #
# main() — menu construction + boot icon, with the run-loop mocked out.
# --------------------------------------------------------------------------- #
class MainMenuConstructionTests(TrayTestBase):
    def _build_icon(self, argv):
        """Run main() with pystray.Icon mocked so icon.run() never blocks.
        Returns the (mocked) Icon instance and the captured kwargs."""
        captured = {}

        def fake_thread(*a, **k):
            # Don't actually start the animation thread; return a stub.
            return mock.Mock()

        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(tray, "_load_base_icon"), \
             mock.patch.object(tray.threading, "Thread", side_effect=fake_thread), \
             mock.patch.object(tray.pystray, "Icon") as Icon:
            inst = Icon.return_value
            inst.run.return_value = None

            def capture(*a, **k):
                captured["args"] = a
                captured["kwargs"] = k
                return inst

            Icon.side_effect = capture
            tray.main()
        return inst, captured

    def _flatten(self, menu):
        """Recursively collect all non-separator MenuItems from a pystray.Menu."""
        items = []
        for it in menu.items:
            if it is tray.pystray.Menu.SEPARATOR:
                continue
            items.append(it)
            sub = getattr(it, "submenu", None)
            if sub is not None:
                items.extend(self._flatten(sub))
        return items

    def test_main_builds_icon_and_runs(self):
        inst, captured = self._build_icon(["tray.py", "--parent-pid", "1234"])
        inst.run.assert_called_once()
        self.assertEqual(captured["args"][0], "jarvis-tray")
        self.assertEqual(tray._parent_pid[0], 1234)

    def test_menu_has_all_top_level_entries(self):
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        texts = [getattr(it, "text", None) for it in menu.items
                 if it is not tray.pystray.Menu.SEPARATOR]
        for expected in ("Open Dashboard", "Settings…",
                         "Pause Listening", "Mute TTS", "Mute Mic",
                         "Ambient Mode", "Open HUD",
                         "Restart JARVIS", "Shut Down JARVIS",
                         "Power tools", "AI", "Audio", "Apple Music", "Memory",
                         "Diagnostics", "About JARVIS", "Show Today's Summary",
                         "Queue Task…", "Quit Tray Only"):
            self.assertIn(expected, texts, expected)
        # "Run Upgrade Now" moved into Power tools (greyed while upgrades are off).
        self.assertNotIn("Run Upgrade Now", texts)
        power = next(it for it in menu.items
                     if getattr(it, "text", None) == "Power tools")
        self.assertIn("Run Upgrade Now",
                      [getattr(it, "text", None) for it in power.submenu.items])

    def test_open_hud_menu_item_wired(self):
        # _on_open_hud used to be dead (defined, never put in a menu). Assert the
        # "Open HUD" item now exists and its action fires the open_hud command.
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        item = next(it for it in self._flatten(menu)
                    if getattr(it, "text", None) == "Open HUD")
        with mock.patch.object(tray, "_send_command") as sc:
            item(mock.Mock())
        sc.assert_called_once_with("open_hud")

    def test_mute_mic_menu_item_wired_and_checks_state(self):
        # The "Mute Mic" toggle fires mic_mute_toggle and its checkmark reflects
        # hud_state.mic_muted (written true here).
        self._write_hud(mic_muted=True)
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        item = next(it for it in self._flatten(menu)
                    if getattr(it, "text", None) == "Mute Mic")
        self.assertTrue(item.checked)
        with mock.patch.object(tray, "_send_command") as sc:
            item(mock.Mock())
        sc.assert_called_once_with("mic_mute_toggle")

    def test_status_header_items_are_disabled(self):
        # The first four items are read-only status lines (enabled=False) whose
        # dynamic text comes from _status_text_* (which read hud_state).
        self._write_hud(state="listening")
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        header = list(menu.items)[:4]
        for item in header:
            self.assertFalse(item.enabled)
            # Evaluating .text must not raise and must yield the bullet prefix.
            self.assertTrue(str(item.text).startswith("●"))

    def test_every_menu_action_is_wired(self):
        # Each actionable MenuItem either fires a command, opens something, or
        # spawns a dialog. Invoke them all with everything that touches the OS
        # mocked, and assert none raise.
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        items = self._flatten(menu)
        icon = mock.Mock()
        invoked = 0
        with mock.patch.object(tray, "_send_command"), \
             mock.patch.object(tray.threading, "Thread"), \
             mock.patch.object(tray.os, "startfile", create=True), \
             mock.patch.object(tray.subprocess, "Popen"), \
             mock.patch.object(tray.subprocess, "run"), \
             mock.patch("webbrowser.open", return_value=True):
            for it in items:
                # pystray stores the supplied callback on the private _action
                # attribute; the status-header items pass action=None.
                if getattr(it, "_action", None) is None:
                    continue
                # Invoking the MenuItem calls its action(icon, item).
                it(icon)
                invoked += 1
        # Sanity: we exercised a large number of distinct callbacks.
        self.assertGreater(invoked, 40)

    def test_checkable_items_evaluate_without_error(self):
        # The checked= / enabled= lambdas read hud_state; evaluating them on a
        # populated state must not raise for any item.
        self._write_hud(state="standby", tts_muted=True, ambient_mode_active=True,
                        daemons_paused=True, debug_mode=True, llm_backend="anthropic",
                        audio_processing_enabled=True)
        self._write(tray.PIPELINE_LOCK_FILE, "{}")
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        for it in self._flatten(menu):
            # Touch both dynamic properties; they invoke the lambdas.
            _ = it.checked
            _ = it.enabled

    def test_pipeline_item_disabled_when_idle(self):
        # With no pipeline flags, "Stop Running Pipeline" must be disabled.
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        stop_item = next(it for it in self._flatten(menu)
                         if getattr(it, "text", None) == "Stop Running Pipeline")
        self.assertFalse(stop_item.enabled)

    def test_audio_sublayer_disabled_when_master_off(self):
        self._write_hud(audio_processing_enabled=False)
        inst, captured = self._build_icon(["tray.py"])
        menu = captured["kwargs"]["menu"]
        echo = next(it for it in self._flatten(menu)
                    if getattr(it, "text", None) == "Echo Cancellation")
        self.assertFalse(echo.enabled)


class MainDialogModeTests(TrayTestBase):
    """main() short-circuits into a dialog entry point when an internal
    --*-dialog flag is present, calling sys.exit with the dialog's return code."""

    def _run_main_expecting_exit(self, argv, dialog_attr):
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(tray, dialog_attr, return_value=0) as d, \
             self.assertRaises(SystemExit) as cm:
            tray.main()
        d.assert_called_once()
        # sys.exit(0) -> code 0
        self.assertEqual(cm.exception.code, 0)

    def test_queue_task_dialog_mode(self):
        self._run_main_expecting_exit(["tray.py", "--queue-task-dialog"],
                                      "_run_queue_task_dialog")

    def test_summary_dialog_mode(self):
        self._run_main_expecting_exit(["tray.py", "--summary-dialog"],
                                      "_run_summary_dialog")

    def test_about_dialog_mode(self):
        self._run_main_expecting_exit(["tray.py", "--about-dialog"],
                                      "_run_about_dialog")

    def test_dossier_dialog_mode(self):
        self._run_main_expecting_exit(["tray.py", "--dossier-dialog"],
                                      "_run_dossier_dialog")


# --------------------------------------------------------------------------- #
# Apple Music tray controls — now-playing label, transport commands, submenu.
#
# JARVIS hosts these because the UWP Apple Music app has no tray of its own.
# Transport goes through the command IPC (the same _send_command path every
# other tray verb uses); the now-playing label is the one thing read in-process
# via the lazy audio.apple_music_app bridge, which the tests fake.
# --------------------------------------------------------------------------- #
class _FakeAppleMusicBridge:
    """Stand-in for audio.apple_music_app used by the now-playing label."""
    def __init__(self, running=False, now=None, running_raises=False,
                 now_raises=False):
        self._running = running
        self._now = now
        self.running_raises = running_raises
        self.now_raises = now_raises

    def is_running(self):
        if self.running_raises:
            raise RuntimeError("psutil boom")
        return self._running

    def now_playing(self):
        if self.now_raises:
            raise RuntimeError("pygetwindow boom")
        return self._now


class AppleMusicLabelTests(TrayTestBase):
    def setUp(self):
        super().setUp()
        # The bridge accessor caches on a module global; clear it each test so a
        # prior test's fake/sentinel doesn't leak.
        self._saved_am = tray.__dict__.get("_apple_music_app_mod")
        tray.__dict__.pop("_apple_music_app_mod", None)
        self.addCleanup(self._restore_am)
        # The label consults SMTC first; pin it off so these cases exercise the
        # window-title bridge below. A dedicated test covers the SMTC path.
        _smtc_p = mock.patch("core.media_now_playing.now_playing_text", return_value=None)
        _smtc_p.start()
        self.addCleanup(_smtc_p.stop)

    def _restore_am(self):
        if self._saved_am is None:
            tray.__dict__.pop("_apple_music_app_mod", None)
        else:
            tray.__dict__["_apple_music_app_mod"] = self._saved_am

    def _patch_bridge(self, bridge):
        return mock.patch.object(tray, "_apple_music_app", return_value=bridge)

    def test_label_prefers_smtc_session(self):
        with mock.patch("core.media_now_playing.now_playing_text",
                        return_value="The Lady in My Life — Michael Jackson"):
            self.assertEqual(_REAL_NOW_PLAYING_LOOKUP(),
                             "♪ The Lady in My Life — Michael Jackson")

    def test_label_shows_now_playing_title(self):
        with self._patch_bridge(_FakeAppleMusicBridge(running=True,
                                                      now="Earth Song — MJ")):
            self.assertEqual(_REAL_NOW_PLAYING_LOOKUP(),
                             "Apple Music: Earth Song — MJ")

    def test_label_idle_when_running_quiet(self):
        with self._patch_bridge(_FakeAppleMusicBridge(running=True, now=None)):
            self.assertEqual(_REAL_NOW_PLAYING_LOOKUP(), "Apple Music: idle")

    def test_label_closed_when_not_running(self):
        with self._patch_bridge(_FakeAppleMusicBridge(running=False)):
            self.assertEqual(_REAL_NOW_PLAYING_LOOKUP(), "Apple Music: closed")

    def test_label_unavailable_when_bridge_absent(self):
        with self._patch_bridge(None):
            self.assertEqual(_REAL_NOW_PLAYING_LOOKUP(),
                             "Apple Music: unavailable")

    def test_label_truncates_long_title(self):
        long_title = "x" * 100
        with self._patch_bridge(_FakeAppleMusicBridge(running=True, now=long_title)):
            out = _REAL_NOW_PLAYING_LOOKUP()
        self.assertIn("…", out)
        self.assertLessEqual(len(out), len("Apple Music: ") + 60 + 1)

    def test_label_is_running_raise_degrades_to_closed(self):
        with self._patch_bridge(_FakeAppleMusicBridge(running_raises=True)):
            self.assertEqual(_REAL_NOW_PLAYING_LOOKUP(), "Apple Music: closed")

    def test_label_now_playing_raise_degrades_to_idle(self):
        with self._patch_bridge(_FakeAppleMusicBridge(running=True, now_raises=True)):
            self.assertEqual(_REAL_NOW_PLAYING_LOOKUP(), "Apple Music: idle")

    def test_bridge_accessor_caches_unavailable_on_import_failure(self):
        # When the bridge can't be resolved, the accessor remembers it (sentinel)
        # so repeated menu opens don't re-import. Force BOTH lookup paths to miss:
        # drop it from sys.modules AND make the `from audio import ...` raise.
        tray.__dict__.pop("_apple_music_app_mod", None)
        real_import = __import__

        def boom(name, *a, **k):
            if name == "audio" and a and a[2] and "apple_music_app" in a[2]:
                raise ImportError("no apple_music_app")
            return real_import(name, *a, **k)

        with mock.patch.dict(sys.modules), \
             mock.patch("builtins.__import__", side_effect=boom):
            sys.modules.pop("audio.apple_music_app", None)
            self.assertIsNone(tray._apple_music_app())
        self.assertIs(tray.__dict__.get("_apple_music_app_mod"), tray._AM_UNAVAILABLE)
        # Second call returns None from the cached sentinel (no re-import).
        self.assertIsNone(tray._apple_music_app())

    def test_bridge_accessor_returns_cached_sentinel_without_reimport(self):
        # Once the sentinel is cached, the accessor short-circuits to None and
        # never touches the import machinery again.
        tray.__dict__["_apple_music_app_mod"] = tray._AM_UNAVAILABLE
        with mock.patch("builtins.__import__",
                        side_effect=AssertionError("must not import")):
            self.assertIsNone(tray._apple_music_app())

    def test_bridge_accessor_returns_module_from_sys_modules(self):
        # The happy path: the already-imported bridge in sys.modules is returned
        # and cached, so the label can call is_running()/now_playing() on it.
        tray.__dict__.pop("_apple_music_app_mod", None)
        sentinel_mod = _FakeAppleMusicBridge(running=True, now="X")
        with mock.patch.dict(sys.modules,
                             {"audio.apple_music_app": sentinel_mod}):
            self.assertIs(tray._apple_music_app(), sentinel_mod)
        self.assertIs(tray.__dict__.get("_apple_music_app_mod"), sentinel_mod)


class AppleMusicCommandTests(TrayTestBase):
    """Each transport verb writes exactly the media/open command bobert's
    drainer routes to the existing ACTIONS (the OS media-key path)."""
    def _assert_cmd(self, fn, expected_cmd):
        fn(mock.Mock(), mock.Mock())
        self.assertEqual(self._last_command()["cmd"], expected_cmd)

    def test_playpause_sends_media_playpause(self):
        self._assert_cmd(tray._on_apple_music_playpause, "media_playpause")

    def test_next_sends_media_next(self):
        self._assert_cmd(tray._on_apple_music_next, "media_next")

    def test_prev_sends_media_prev(self):
        self._assert_cmd(tray._on_apple_music_prev, "media_prev")

    def test_open_opens_the_web_player_in_the_browser(self):
        # The owner uses Apple Music in Chrome; the Store app (AUMID launch via
        # open_apple_music) is not his player.
        with mock.patch("webbrowser.open", return_value=True) as wb:
            tray._on_open_apple_music(mock.Mock(), mock.Mock())
        wb.assert_called_once_with("https://music.apple.com/")
        self.assertEqual(self._read_commands(), [])

    def test_open_url_is_configurable(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_APPLE_MUSIC_URL": "https://example.test/m"}), \
             mock.patch("webbrowser.open", return_value=True) as wb:
            tray._on_open_apple_music(mock.Mock(), mock.Mock())
        wb.assert_called_once_with("https://example.test/m")

    def test_open_failure_is_reported(self):
        with mock.patch("webbrowser.open", return_value=False), \
             mock.patch.object(tray, "_notify", create=True) as note:
            tray._on_open_apple_music(mock.Mock(), mock.Mock())
        note.assert_called_once()


class AppleMusicMenuTests(TrayTestBase):
    """The Apple Music submenu renders, exposes the transport verbs + a dynamic
    now-playing header, and wires each verb to the right command — even when the
    bridge is absent (the items still build; the label degrades)."""

    def setUp(self):
        super().setUp()
        # The now-playing header consults SMTC (core.media_now_playing) first;
        # pin it off so these menu tests deterministically exercise the
        # window-title bridge path they assert on, regardless of whatever is
        # actually playing on the test box.
        _smtc_p = mock.patch("core.media_now_playing.now_playing_text", return_value=None)
        _smtc_p.start()
        self.addCleanup(_smtc_p.stop)

    def _build_menu(self):
        captured = {}

        def fake_thread(*a, **k):
            return mock.Mock()

        with mock.patch.object(sys, "argv", ["tray.py"]), \
             mock.patch.object(tray, "_load_base_icon"), \
             mock.patch.object(tray.threading, "Thread", side_effect=fake_thread), \
             mock.patch.object(tray.pystray, "Icon") as Icon:
            inst = Icon.return_value
            inst.run.return_value = None

            def capture(*a, **k):
                captured["kwargs"] = k
                return inst

            Icon.side_effect = capture
            tray.main()
        return captured["kwargs"]["menu"]

    def _flatten(self, menu):
        items = []
        for it in menu.items:
            if it is tray.pystray.Menu.SEPARATOR:
                continue
            items.append(it)
            sub = getattr(it, "submenu", None)
            if sub is not None:
                items.extend(self._flatten(sub))
        return items

    def test_apple_music_submenu_present_in_top_level(self):
        menu = self._build_menu()
        texts = [getattr(it, "text", None) for it in menu.items
                 if it is not tray.pystray.Menu.SEPARATOR]
        self.assertIn("Apple Music", texts)

    def test_submenu_has_transport_items(self):
        menu = self._build_menu()
        texts = [getattr(it, "text", None) for it in self._flatten(menu)]
        for expected in ("Play / Pause", "Next", "Previous", "Open Apple Music"):
            self.assertIn(expected, texts, expected)

    def test_transport_items_fire_right_commands(self):
        menu = self._build_menu()
        items = {getattr(it, "text", None): it for it in self._flatten(menu)}
        for label, cmd in (("Play / Pause", "media_playpause"),
                           ("Next", "media_next"),
                           ("Previous", "media_prev")):
            with mock.patch.object(tray, "_send_command") as sc:
                items[label](mock.Mock())
            sc.assert_called_once_with(cmd)
        with mock.patch("webbrowser.open", return_value=True) as wb, \
             mock.patch.object(tray, "_send_command") as sc:
            items["Open Apple Music"](mock.Mock())
        wb.assert_called_once()
        sc.assert_not_called()

    def _prime_header(self):
        """Run one real (off-thread, time-boxed) now-playing refresh."""
        with mock.patch.object(tray, "_now_playing_lookup",
                               side_effect=_REAL_NOW_PLAYING_LOOKUP):
            tray._refresh_now_playing(timeout=5.0)

    def test_now_playing_header_present_and_dynamic(self):
        # The header is a disabled item whose text is the CACHED now-playing
        # label; a refresh with a fake bridge reflects the track.
        with mock.patch.object(tray, "_apple_music_app",
                               return_value=_FakeAppleMusicBridge(running=True,
                                                                  now="A Song")):
            self._prime_header()
            menu = self._build_menu()
            am_item = next(it for it in menu.items
                           if getattr(it, "text", "") == "Apple Music")
            header = list(am_item.submenu.items)[0]
            self.assertFalse(header.enabled)
            self.assertEqual(header.text, "Apple Music: A Song")

    def test_submenu_builds_when_bridge_absent(self):
        # With the bridge unavailable, the submenu must still build and its label
        # must render the 'unavailable' text without raising.
        with mock.patch.object(tray, "_apple_music_app", return_value=None):
            self._prime_header()
            menu = self._build_menu()
            am_item = next(it for it in menu.items
                           if getattr(it, "text", "") == "Apple Music")
            header = list(am_item.submenu.items)[0]
            self.assertEqual(header.text, "Apple Music: unavailable")
            # Transport verbs still present + invokable (no-op send is fine).
            texts = [getattr(it, "text", None) for it in am_item.submenu.items]
            self.assertIn("Play / Pause", texts)


class SpawnedDialogTrackingTests(TrayTestBase):
    """Finding #40: modal dialog subprocesses must be tracked and reaped on
    quit / parent-death so they don't outlive JARVIS."""

    def test_tracked_dialog_run_registers_then_cleans_up(self):
        self.addCleanup(tray._dialog_procs.clear)

        class _FakePopen:
            def __init__(self, args, stdout=None, stderr=None, text=False,
                         creationflags=0):
                self.returncode = 0
                self.in_list_mid_call = None

            def communicate(self, timeout=None):
                # The child must be registered for the duration of the call.
                self.in_list_mid_call = any(p is self for p in tray._dialog_procs)
                return ("queued text", "")

            def poll(self):
                return self.returncode

        captured = {}

        def _make(*a, **k):
            p = _FakePopen(*a, **k)
            captured["p"] = p
            return p

        with mock.patch.object(tray.subprocess, "Popen", side_effect=_make):
            res = tray._tracked_dialog_run(["x"], capture_output=True,
                                           text=True, timeout=5)
        self.assertEqual(res.stdout, "queued text")
        self.assertEqual(res.returncode, 0)
        self.assertTrue(captured["p"].in_list_mid_call)     # tracked during call
        self.assertNotIn(captured["p"], tray._dialog_procs)  # cleaned up after

    def test_tracked_dialog_run_kills_child_on_timeout(self):
        self.addCleanup(tray._dialog_procs.clear)

        class _WedgedPopen:
            def __init__(self, *a, **k):
                self.killed = False
                self._first = True

            def communicate(self, timeout=None):
                if self._first:
                    self._first = False
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return ("", "")

            def kill(self):
                self.killed = True

            def poll(self):
                return None

        holder = {}

        def _make(*a, **k):
            holder["p"] = _WedgedPopen(*a, **k)
            return holder["p"]

        with mock.patch.object(tray.subprocess, "Popen", side_effect=_make):
            with self.assertRaises(subprocess.TimeoutExpired):
                tray._tracked_dialog_run(["x"], timeout=1)
        self.assertTrue(holder["p"].killed)                 # orphan killed
        self.assertNotIn(holder["p"], tray._dialog_procs)   # unregistered

    def test_terminate_dialog_procs_kills_live_only_and_clears(self):
        live = mock.Mock(); live.poll.return_value = None    # still running
        dead = mock.Mock(); dead.poll.return_value = 0       # already exited
        tray._dialog_procs[:] = [live, dead]
        self.addCleanup(tray._dialog_procs.clear)
        tray._terminate_dialog_procs()
        live.terminate.assert_called_once()
        dead.terminate.assert_not_called()
        self.assertEqual(tray._dialog_procs, [])

    def test_terminate_swallows_terminate_errors(self):
        proc = mock.Mock(); proc.poll.return_value = None
        proc.terminate.side_effect = OSError("gone")
        tray._dialog_procs[:] = [proc]
        self.addCleanup(tray._dialog_procs.clear)
        tray._terminate_dialog_procs()   # must not raise
        self.assertEqual(tray._dialog_procs, [])

    def test_quit_reaps_open_dialogs(self):
        proc = mock.Mock(); proc.poll.return_value = None
        tray._dialog_procs[:] = [proc]
        self.addCleanup(tray._dialog_procs.clear)
        with mock.patch.object(tray, "_notify", create=True):
            tray._on_quit(mock.Mock(), mock.Mock())   # arm (second-click gate)
            proc.terminate.assert_not_called()
            tray._on_quit(mock.Mock(), mock.Mock())   # confirm
        proc.terminate.assert_called_once()
        self.assertEqual(tray._dialog_procs, [])

    def test_parent_death_reaps_open_dialogs(self):
        proc = mock.Mock(); proc.poll.return_value = None
        tray._dialog_procs[:] = [proc]
        self.addCleanup(tray._dialog_procs.clear)
        icon = mock.Mock()
        with mock.patch.object(tray, "_parent_alive", return_value=False):
            tray._animate(icon)
        proc.terminate.assert_called_once()
        icon.stop.assert_called_once()


# =========================================================================== #
# 2026-09-30 tray audit fixes.
#
# The fake pystray above evaluates every lambda LIVE on each access, which is
# why the old suite never saw the stale-checkmark bug: the real win32 backend
# freezes the whole menu into an HMENU at update_menu() time and rebuilds it
# only (a) when Icon.run() marks itself ready and (b) right AFTER a click —
# before the monolith's 2 Hz drainer has applied the toggle. _CachingFakeIcon
# reproduces exactly that (proven against pystray 0.19.5 by the audit probe
# probe_pystray_staleness.py).
# =========================================================================== #
class _CachingFakeIcon:
    """pystray.Icon stand-in that CACHES the menu like the win32 backend."""

    HAS_NOTIFICATION = True

    def __init__(self, name, icon=None, title=None, menu=None, **_kw):
        self.name = name
        self.icon = icon
        self.title = title
        self.menu = menu
        self.notifications = []
        self.builds = 0
        self._snapshot = []
        self.update_menu()          # pystray's menu setter does this too

    # -- what pystray does ------------------------------------------------- #
    def update_menu(self):
        self.builds += 1
        self._snapshot = self._freeze(self.menu)

    def _freeze(self, menu):
        out = []
        for item in menu:            # visible items only, like pystray
            if item is tray.pystray.Menu.SEPARATOR:
                out.append(None)
                continue
            sub = item.submenu
            out.append({"text": item.text, "checked": bool(item.checked),
                        "enabled": bool(item.enabled),
                        "default": bool(item.default), "item": item,
                        "sub": self._freeze(sub) if sub else None})
        return out

    def notify(self, message, title=None):
        self.notifications.append((title, message))

    def run(self, *a, **k):
        pass

    def stop(self, *a, **k):
        pass

    # -- what the owner sees / does --------------------------------------- #
    def shown(self, *path):
        """The FROZEN entry a right-click would display (by label path)."""
        entries = self._snapshot
        entry = None
        for label in path:
            entry = next(e for e in entries
                         if e is not None and e["text"] == label)
            entries = entry["sub"] or []
        return entry

    def click(self, *path):
        """pystray win32 _on_notify: run the frozen item's callback, then
        (via Icon._handler) rebuild the menu immediately."""
        entry = self.shown(*path)
        try:
            entry["item"](self)
        finally:
            self.update_menu()


class _TrayFixBase(TrayTestBase):
    def _n_shot_event(self, n=1):
        """An Event whose n-th .wait() sets it: _animate runs n iterations."""
        ev = threading.Event()
        real_wait = ev.wait
        left = [n]

        def wait(timeout=None):
            left[0] -= 1
            if left[0] <= 0:
                ev.set()
            return real_wait(0)
        ev.wait = wait
        return ev

    def _tick(self, icon, n=1):
        """Exactly n _animate iterations (the tray's 5 Hz loop)."""
        with mock.patch.object(tray, "_stop_event", self._n_shot_event(n)), \
             mock.patch.object(tray, "_parent_alive", return_value=True):
            tray._animate(icon)

    def _main_icon(self, argv=("tray.py",)):
        """Run the REAL main() with the caching fake as pystray.Icon."""
        made = {}

        def factory(*a, **k):
            made["icon"] = _CachingFakeIcon(*a, **k)
            return made["icon"]
        with mock.patch.object(sys, "argv", list(argv)), \
             mock.patch.object(tray, "_load_base_icon"), \
             mock.patch.object(tray.threading, "Thread",
                               side_effect=lambda *a, **k: mock.Mock()), \
             mock.patch.object(tray.pystray, "Icon", side_effect=factory):
            tray.main()
        return made["icon"]

    def _ready_hud(self, **fields):
        fields.setdefault("tray_ready_pid", 1)
        self._write_hud(**fields)


class StaleMenuTests(_TrayFixBase):
    """P0: checkmarks ran one click behind."""

    def _mute_then_drainer_applies(self):
        self._ready_hud(mic_muted=False)
        icon = self._main_icon()
        self.assertFalse(icon.shown("Mute Mic")["checked"])
        icon.click("Mute Mic")
        self.assertEqual(self._last_command()["cmd"], "mic_mute_toggle")
        # pystray rebuilt BEFORE the monolith applied the toggle:
        self.assertFalse(icon.shown("Mute Mic")["checked"])
        # ~0.5 s later the 2 Hz drainer applies it and republishes hud_state.
        self._ready_hud(mic_muted=True)
        return icon

    def test_checkmark_catches_up_on_the_next_tick(self):
        icon = self._mute_then_drainer_applies()
        self._tick(icon)
        self.assertTrue(icon.shown("Mute Mic")["checked"],
                        "the menu still shows the state from before the click")

    def test_status_line_catches_up_too(self):
        icon = self._mute_then_drainer_applies()
        self._tick(icon)
        self.assertIn("● Listening: muted",
                      [e["text"] for e in icon._snapshot if e])

    def test_submenu_checkmark_catches_up(self):
        self._ready_hud(daemons_paused=False)
        icon = self._main_icon()
        icon.click("Power tools", "Pause All Daemons")
        self._ready_hud(daemons_paused=True)
        self._tick(icon)
        self.assertTrue(icon.shown("Power tools", "Pause All Daemons")["checked"])

    def test_no_rebuild_when_nothing_changed(self):
        self._ready_hud(mic_muted=False)
        icon = self._main_icon()
        self._tick(icon)                       # first tick records the sig
        builds = icon.builds
        tray._menu_state["at"] = 0.0           # debounce is not the reason
        self._tick(icon)
        self.assertEqual(icon.builds, builds)

    def test_rebuilds_are_debounced(self):
        self._ready_hud(mic_muted=False)
        icon = self._main_icon()
        self._tick(icon)
        builds = icon.builds
        self._ready_hud(mic_muted=True)
        tray._menu_state["at"] = time.time()   # a rebuild just happened
        self._tick(icon)
        self.assertEqual(icon.builds, builds, "rebuilt inside the debounce")
        tray._menu_state["at"] = time.time() - tray.MENU_REFRESH_MIN_S - 0.01
        self._tick(icon)
        self.assertEqual(icon.builds, builds + 1)
        self.assertTrue(icon.shown("Mute Mic")["checked"])

    def test_no_rebuild_while_the_menu_is_open(self):
        self._ready_hud(mic_muted=False)
        icon = self._main_icon()
        self._tick(icon)
        builds = icon.builds
        self._ready_hud(mic_muted=True)
        tray._menu_open.set()
        self._tick(icon)
        self.assertEqual(icon.builds, builds)
        tray._menu_open.clear()
        tray._menu_state["at"] = 0.0
        self._tick(icon)
        self.assertEqual(icon.builds, builds + 1)

    def test_menu_lambdas_read_one_snapshot_per_tick(self):
        # Evaluating the whole menu must not re-read hud_state.json per item.
        self._ready_hud(mic_muted=True)
        icon = self._main_icon()
        real_open = open
        reads = []

        def counting_open(path, *a, **k):
            if os.path.normcase(str(path)) == os.path.normcase(tray.HUD_STATE_FILE):
                reads.append(path)
            return real_open(path, *a, **k)
        with mock.patch("builtins.open", side_effect=counting_open):
            self._tick(icon)
        self.assertLessEqual(len(reads), 1)

    def test_menu_open_guard_wraps_the_win32_notify_handler(self):
        seen = []

        class _Win32ish:
            def __init__(self):
                self._message_handlers = {0x401: self._on_notify, 2: lambda *a: 0}

            def _on_notify(self, wparam, lparam):
                seen.append(tray._menu_open.is_set())
                return 7
        icon = _Win32ish()
        self.assertTrue(tray._install_menu_open_guard(icon))
        self.assertEqual(icon._message_handlers[0x401](0, 0), 7)
        self.assertEqual(seen, [True])
        self.assertFalse(tray._menu_open.is_set())
        self.assertFalse(tray._install_menu_open_guard(mock.Mock()))

    def test_signature_failure_forces_a_rebuild_not_a_crash(self):
        bad = mock.Mock()
        bad.__iter__ = mock.Mock(side_effect=RuntimeError("boom"))
        sig = tray._menu_signature(bad)
        self.assertEqual(sig[0][0], "error")


class SettingsMenuTests(_TrayFixBase):
    """Settings shortcut: (1) a real clickable item, no duplicate."""

    def test_settings_is_a_clickable_top_level_item(self):
        icon = self._main_icon()
        entry = icon.shown("Settings…")
        self.assertIsNone(entry["sub"], "Settings must not be a submenu header")
        with mock.patch.object(tray, "_open_settings_window") as osw:
            with mock.patch.object(tray.threading, "Thread",
                                   side_effect=lambda target, args=(), **k:
                                   mock.Mock(start=lambda: target(*args))):
                icon.click("Settings…")
        osw.assert_called_once_with("")

    def test_no_settings_duplicate_left_in_audio(self):
        icon = self._main_icon()
        audio = [e["text"] for e in icon.shown("Audio")["sub"] if e]
        self.assertFalse([t for t in audio if "settings" in t.lower()], audio)
        top = [e["text"] for e in icon._snapshot if e]
        self.assertEqual(sum(1 for t in top if t.lower().startswith("settings")), 1)


class SettingsLaunchSubprocessTests(TrayTestBase):
    """(2) Run the tray's EXACT launch command for real, window-free: the
    module form must import and resolve `core` from the tray's cwd."""

    def _captured_launch(self):
        self._write(tray.SETTINGS_WINDOW, "# settings")
        with mock.patch.object(tray.subprocess, "Popen",
                               return_value=_FakeProc(rc=None)) as P, \
             mock.patch.object(tray, "_notify", create=True):
            tray._open_settings_window("ai")
        return P.call_args.args[0], P.call_args.kwargs

    def test_exact_launch_command_imports_from_the_project_root(self):
        argv, kw = self._captured_launch()
        self.assertEqual(os.path.normcase(kw.get("cwd") or ""),
                         os.path.normcase(tray.PROJECT_DIR),
                         "the launch must set cwd to the project root")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)            # the tray has none
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        # The same interpreter + launch form, with --help swapped in for the
        # tab so argparse exits before any window could be built.
        help_argv = [a for a in argv if a not in ("--tab", "ai")] + ["--help"]
        r = subprocess.run(help_argv, cwd=_REAL_PROJECT_DIR, env=env,
                           capture_output=True, text=True, timeout=60,
                           creationflags=flags)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--tab", r.stdout)
        # And the window's lazy `from core import …` (what failed live) works
        # under the same sys.path rule a module run gets (cwd first).
        probe = ("import sys, runpy; sys.argv = ['settings_window', '--help'];\n"
                 "import tools.settings_window as sw\n"
                 "sys.exit(0 if sw._model_lockstep() is not None else 3)\n")
        r2 = subprocess.run([argv[0], "-c", probe], cwd=_REAL_PROJECT_DIR,
                            env=env, capture_output=True, text=True,
                            timeout=60, creationflags=flags)
        self.assertEqual(r2.returncode, 0, r2.stderr)


class ResultRoundTripTests(_TrayFixBase):
    """P0: ~18 items showed no result."""

    def _write_results(self, *entries):
        self._write(tray.TRAY_RESULTS_FILE, json.dumps({"results": list(entries)}))
        tray._results_state["checked_at"] = 0.0
        tray._results_state["mtime"] = None

    def test_click_to_balloon(self):
        self._ready_hud()
        icon = self._main_icon()
        icon.click("AI", "Show LLM Call Stats")
        cmd = self._last_command()
        self.assertEqual(cmd["cmd"], "show_llm_stats")
        rid = cmd["rid"]
        self._write_results({"rid": rid, "cmd": "show_llm_stats", "seq": 1,
                             "final": True, "text": "backend=ollama model=x"})
        self._tick(icon)
        self.assertEqual(icon.notifications,
                         [("JARVIS — LLM Call Stats", "backend=ollama model=x")])
        self.assertNotIn(rid, tray._pending)
        self._tick(icon)                       # shown once, never again
        self.assertEqual(len(icon.notifications), 1)

    def test_interim_then_final(self):
        tray._icon_ref[0] = icon = _CachingFakeIcon("t", menu=tray.pystray.Menu())
        rid = tray._send_request("force_backup", "Force Backup")
        self._write_results({"rid": rid, "seq": 1, "final": False,
                             "text": "backup started"})
        tray._poll_results()
        self.assertIn(rid, tray._pending)
        self._write_results({"rid": rid, "seq": 1, "final": False,
                             "text": "backup started"},
                            {"rid": rid, "seq": 2, "final": True,
                             "text": "backup -> 20260930_101500"})
        tray._poll_results()
        self.assertEqual([m for _, m in icon.notifications],
                         ["backup started", "backup -> 20260930_101500"])
        self.assertNotIn(rid, tray._pending)

    def test_final_before_interim_shows_only_the_final(self):
        # The monolith's worker can finish before the dispatcher files the
        # interim "started" line; the late interim must not pop up after.
        tray._icon_ref[0] = icon = _CachingFakeIcon("t", menu=tray.pystray.Menu())
        rid = tray._send_request("force_backup", "Force Backup")
        self._write_results({"rid": rid, "seq": 2, "final": True,
                             "text": "backup -> X"},
                            {"rid": rid, "seq": 1, "final": False,
                             "text": "backup started"})
        tray._poll_results()
        self.assertEqual([m for _, m in icon.notifications], ["backup -> X"])

    def test_a_second_write_with_the_same_mtime_is_still_read(self):
        # Windows file times tick coarsely: interim + final can share an
        # mtime. The size differs, and a pending request re-reads anyway.
        tray._icon_ref[0] = icon = _CachingFakeIcon("t", menu=tray.pystray.Menu())
        rid = tray._send_request("force_backup", "Force Backup")
        self._write_results({"rid": rid, "seq": 1, "final": False,
                             "text": "backup started"})
        tray._poll_results()
        st = os.stat(tray.TRAY_RESULTS_FILE)
        self._write(tray.TRAY_RESULTS_FILE, json.dumps({"results": [
            {"rid": rid, "seq": 1, "final": False, "text": "backup started"},
            {"rid": rid, "seq": 2, "final": True, "text": "backup -> Y"}]}))
        os.utime(tray.TRAY_RESULTS_FILE, ns=(st.st_atime_ns, st.st_mtime_ns))
        tray._results_state["checked_at"] = 0.0      # next 5 Hz poll
        tray._poll_results()
        self.assertEqual([m for _, m in icon.notifications],
                         ["backup started", "backup -> Y"])

    def test_answers_to_other_requests_are_ignored(self):
        tray._icon_ref[0] = icon = _CachingFakeIcon("t", menu=tray.pystray.Menu())
        tray._send_request("test_mic", "Test Mic")
        self._write_results({"rid": "r999-1", "seq": 1, "final": True,
                             "text": "not ours"})
        self.assertEqual(tray._poll_results(), 0)
        self.assertEqual(icon.notifications, [])

    def test_long_answer_opens_as_a_text_file(self):
        tray._icon_ref[0] = icon = _CachingFakeIcon("t", menu=tray.pystray.Menu())
        rid = tray._send_request("show_last_diagnostic", "Last Diagnostic Run")
        body = "\n".join(f"probe {i}: OK" for i in range(40))
        self._write_results({"rid": rid, "seq": 1, "final": True, "text": body})
        with mock.patch.object(tray, "_open_path") as op:
            tray._poll_results()
        path = op.call_args.args[0]
        self.assertTrue(path.endswith("show_last_diagnostic.txt"))
        with open(path, encoding="utf-8") as f:
            self.assertIn("probe 39: OK", f.read())
        self.assertIn("full output opened", icon.notifications[0][1])

    def test_balloon_text_is_clipped_to_the_shell_limit(self):
        icon = _CachingFakeIcon("t", menu=tray.pystray.Menu())
        tray._icon_ref[0] = icon
        self.assertTrue(tray._notify("x" * 1000, "T" * 200))
        title, msg = icon.notifications[0]
        self.assertLessEqual(len(msg), 256)
        self.assertLessEqual(len(title), 64)

    def test_pending_requests_expire(self):
        rid = tray._send_request("run_diagnostic", "Diagnostic")
        tray._pending[rid]["sent_at"] = time.time() - tray.RESULT_WAIT_S - 1
        tray._poll_results()
        self.assertNotIn(rid, tray._pending)

    def test_every_answering_item_sends_a_request(self):
        # The 18 audit items: each must ask for an answer (rid).
        self._ready_hud(overnight_upgrade_enabled=True)
        icon = self._main_icon()
        paths = [("Power tools", "Force Backup Now"),
                 ("Power tools", "Reload All Skills"),
                 ("Power tools", "Run Smoke Test"),
                 ("AI", "Show LLM Call Stats"),
                 ("AI", "Switch to Local LLM (default)"),
                 ("Memory", "Recent Facts Learned (last 24h)"),
                 ("Memory", "Export Memory (JSON)"),
                 ("Diagnostics", "Run Diagnostic Now"),
                 ("Diagnostics", "Show Last Diagnostic Run"),
                 ("Diagnostics", "Test Mic"), ("Diagnostics", "Test TTS"),
                 ("Diagnostics", "Test Vision"),
                 ("Diagnostics", "Test Each Skill"),
                 ("Diagnostics", "Latency Benchmark")]
        for path in paths:
            icon.click(*path)
            self.assertTrue(self._last_command().get("rid"), path)
        # Confirm-gated ones answer too (after the second click).
        with mock.patch.object(tray, "_notify", create=True):
            for path in (("Memory", "Forget Last Hour"), ("Memory", "Reset Memory…")):
                icon.click(*path)
                armed = "⚠ Click again: " + path[-1]
                icon.click(path[0], armed)
                self.assertTrue(self._last_command().get("rid"), path)


class ConfirmGateTests(_TrayFixBase):
    """P1: destructive one-clicks need a second click (no modal popups)."""

    def test_first_click_arms_and_relabels(self):
        icon = self._main_icon()
        tray._icon_ref[0] = icon
        icon.click("Restart JARVIS")
        self.assertEqual(self._read_commands(), [])
        self.assertIn("again", icon.notifications[-1][1])
        # The rebuilt menu names the pending confirmation.
        self.assertIsNotNone(icon.shown("⚠ Click again: Restart JARVIS"))
        icon.click("⚠ Click again: Restart JARVIS")
        self.assertEqual(self._last_command()["cmd"], "restart")
        self.assertIsNotNone(icon.shown("Restart JARVIS"))   # disarmed

    def test_confirmation_expires(self):
        with mock.patch.object(tray, "_notify", create=True):
            self.assertFalse(tray._confirmed("k", "Thing"))
            tray._confirm_armed["k"] = time.time() - 1      # window passed
            self.assertFalse(tray._confirmed("k", "Thing"))   # re-armed
            self.assertTrue(tray._confirmed("k", "Thing"))

    def test_gated_items(self):
        # Every item the audit named needs the second click.
        cases = [(tray._on_reset_memory, "reset_memory"),
                 (tray._on_forget_last_hour, "forget_last_hour"),
                 (tray._on_switch_anthropic, "switch_llm"),
                 (tray._on_restart, "restart"),
                 (tray._on_shutdown_jarvis, "shutdown_jarvis")]
        for fn, cmd in cases:
            with self.subTest(cmd=cmd), mock.patch.object(tray, "_notify", create=True):
                before = len(self._read_commands())
                fn(mock.Mock(), mock.Mock())
                self.assertEqual(len(self._read_commands()), before, cmd)
                fn(mock.Mock(), mock.Mock())
                self.assertEqual(self._last_command()["cmd"], cmd)
                tray._confirm_armed.clear()


class UpgradeGatingTests(_TrayFixBase):
    """P0: Run Upgrade Now + the queue badge while upgrades are off."""

    def test_run_upgrade_greyed_when_upgrades_off(self):
        self._ready_hud(overnight_upgrade_enabled=False)
        icon = self._main_icon()
        self.assertFalse(icon.shown("Power tools", "Run Upgrade Now")["enabled"])
        self._ready_hud(overnight_upgrade_enabled=True)
        self._tick(icon)
        self.assertTrue(icon.shown("Power tools", "Run Upgrade Now")["enabled"])

    def test_run_upgrade_callback_refuses_when_off(self):
        self._ready_hud()
        with mock.patch.object(tray, "_notify", create=True) as note:
            tray._on_force_upgrade(mock.Mock(), mock.Mock())
        self.assertEqual(self._read_commands(), [])
        note.assert_called_once()

    def test_queue_badge_and_line_hidden_when_upgrades_off(self):
        self._write(tray.TODO_FILE, "".join(f"- [ ] t{i}\n" for i in range(150)))
        self._bust_queue_cache()
        self._ready_hud(overnight_upgrade_enabled=False)
        icon = self._main_icon()
        texts = [e["text"] for e in icon._snapshot if e]
        self.assertFalse([t for t in texts if t.startswith("● Queue")])
        captured = {}
        real = tray._render_icon

        def spy(*a, **k):
            captured["queue_count"] = k.get("queue_count")
            return real(*a, **k)
        with mock.patch.object(tray, "_render_icon", side_effect=spy):
            self._tick(icon)
        self.assertEqual(captured.get("queue_count"), 0)
        self.assertNotIn("queued", icon.title)

    def test_queue_dialog_wording_follows_the_upgrade_switch(self):
        seen = {}

        def ask(title, prompt, parent=None):
            seen["prompt"] = prompt
            return None
        fake_tk = mock.Mock()
        with mock.patch.object(tray, "_HAS_TK", True), \
             mock.patch.object(tray, "tk", fake_tk, create=True), \
             mock.patch.object(tray, "simpledialog",
                               mock.Mock(askstring=ask), create=True):
            self._ready_hud(overnight_upgrade_enabled=False)
            tray._run_queue_task_dialog()
            self.assertNotIn("overnight upgrade:", seen["prompt"])
            self.assertIn("Claude Code", seen["prompt"])
            self._ready_hud(overnight_upgrade_enabled=True)
            tray._run_queue_task_dialog()
            self.assertIn("next overnight upgrade", seen["prompt"])


class RealStateCheckmarkTests(_TrayFixBase):
    """P1: Pause Listening + Ambient Mode read the real state."""

    def test_pause_listening_reads_the_standby_flags(self):
        # In standby the main loop keeps state='idle' — the label lies.
        self._ready_hud(state="idle", sleep_mode=True, standby_mode=True)
        icon = self._main_icon()
        self.assertTrue(icon.shown("Pause Listening")["checked"])
        self.assertIn("● Listening: standby", [e["text"] for e in icon._snapshot if e])
        icon.click("Pause Listening")
        self.assertEqual(self._last_command()["cmd"], "force_wake")

    def test_pause_listening_unchecked_when_awake_even_if_label_says_standby(self):
        self._ready_hud(state="standby", sleep_mode=False, standby_mode=False)
        self.assertFalse(tray._is_listen_paused())

    def test_state_label_is_only_a_fallback(self):
        self._write_hud(state="standby")
        self.assertTrue(tray._is_listen_paused())

    def test_ambient_checkmark_reads_the_running_daemon(self):
        self._ready_hud(ambient_mode_active=False, ambient_listening=True)
        icon = self._main_icon()
        self.assertTrue(icon.shown("Ambient Mode")["checked"])

    def test_standby_icon_is_gray(self):
        s = tray._classify_state({"state": "idle", "standby_mode": True})
        self.assertEqual(s["state"], "standby")


class LeftClickDashboardTests(_TrayFixBase):
    """P1: a default (left-click) action."""

    def test_dashboard_is_the_default_item(self):
        icon = self._main_icon()
        defaults = [e["text"] for e in icon._snapshot if e and e["default"]]
        self.assertEqual(defaults, ["Open Dashboard"])

    def test_left_click_opens_the_dashboard_when_it_runs(self):
        self._ready_hud(web_port=8766)
        icon = self._main_icon()
        with mock.patch("webbrowser.open", return_value=True) as wb:
            icon.menu(icon)                    # pystray: left-click = menu(icon)
        wb.assert_called_once_with("http://127.0.0.1:8766/")

    def test_left_click_shows_status_when_the_dashboard_is_off(self):
        self._ready_hud(web_port=0, mic_muted=True)
        icon = self._main_icon()
        tray._icon_ref[0] = icon
        with mock.patch("webbrowser.open") as wb:
            icon.menu(icon)
        wb.assert_not_called()
        self.assertIn("Mic muted", icon.notifications[-1][1])


class RemovedDeadItemsTests(_TrayFixBase):
    def test_no_fake_cache_items_and_no_dead_picker(self):
        icon = self._main_icon()

        def walk(entries):
            for e in entries:
                if e is None:
                    continue
                yield e["text"]
                if e["sub"]:
                    yield from walk(e["sub"])
        texts = list(walk(icon._snapshot))
        for gone in ("Clear LLM Cache", "Reset Local LLM Cache",
                     "Switch to Local LLM (other…)"):
            self.assertNotIn(gone, texts)
        self.assertIn("Open Logs Folder", texts)       # dead callback now wired


class LocalModelPickerTests(_TrayFixBase):
    def test_picker_lists_installed_models_with_the_active_one_checked(self):
        tray._models_cache.update({"tags": ["gemma4:12b", "qwen3:8b"],
                                   "at": time.time()})
        self._ready_hud(llm_backend="qwen3:8b")
        icon = self._main_icon()
        sub = icon.shown("AI", "Local Model")["sub"]
        self.assertEqual([e["text"] for e in sub], ["gemma4:12b", "qwen3:8b"])
        self.assertEqual([e["checked"] for e in sub], [False, True])
        icon.click("AI", "Local Model", "gemma4:12b")
        cmd = self._last_command()
        self.assertEqual((cmd["cmd"], cmd["backend"]), ("switch_llm", "gemma4:12b"))

    def test_picker_says_so_when_ollama_is_down(self):
        icon = self._main_icon()
        sub = icon.shown("AI", "Local Model")["sub"]
        self.assertEqual(len(sub), 1)
        self.assertFalse(sub[0]["enabled"])

    def test_fetch_parses_api_tags_and_skips_embedders(self):
        body = json.dumps({"models": [{"name": "qwen3:8b"},
                                      {"name": "nomic-embed-text:latest"},
                                      {"model": "gemma4:12b"}]}).encode()
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = body
        with mock.patch("urllib.request.urlopen", return_value=resp) as uo:
            self.assertEqual(_REAL_FETCH_LOCAL_MODELS(timeout=1),
                             ["gemma4:12b", "qwen3:8b"])
        self.assertEqual(uo.call_args.kwargs.get("timeout"), 1)
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            self.assertEqual(_REAL_FETCH_LOCAL_MODELS(timeout=1), [])


class NowPlayingCacheTests(_TrayFixBase):
    """P1: the now-playing lookup inside every menu rebuild had no timeout."""

    def test_a_hung_lookup_never_blocks_the_menu(self):
        gate = threading.Event()
        self.addCleanup(gate.set)

        def hang():
            gate.wait(10)
            return "never"
        with mock.patch.object(tray, "_now_playing_lookup", side_effect=hang):
            t0 = time.time()
            label = tray._status_text_apple_music()     # menu path
            self.assertLess(time.time() - t0, 0.5)
            self.assertEqual(label, "♪ …")
            t0 = time.time()
            tray._refresh_now_playing(timeout=0.2)       # bounded refresh
            self.assertLess(time.time() - t0, 2.0)

    def test_label_is_cached_between_refreshes(self):
        calls = []
        with mock.patch.object(tray, "_now_playing_lookup",
                               side_effect=lambda: calls.append(1) or "♪ Song"):
            tray._refresh_now_playing(timeout=2)
            for _ in range(20):
                self.assertEqual(tray._status_text_apple_music(), "♪ Song")
        self.assertEqual(len(calls), 1)


class TrayLogTests(TrayTestBase):
    """(3) The tray's output goes to logs\\tray.log."""

    def test_prints_and_tracebacks_land_in_the_log(self):
        path = os.path.join(self.dir, "logs", "tray.log")
        saved = (sys.stdout, sys.stderr, threading.excepthook,
                 list(tray.logging.getLogger().handlers))
        self.addCleanup(tray.logging.getLogger().setLevel,
                        tray.logging.getLogger().level)
        orig_out, orig_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = orig_out, orig_err
        try:
            self.assertTrue(_REAL_SETUP_TRAY_LOGGING(path))
            print("[tray] hello from the tray")
            tray.logging.getLogger().error("boom %s", 42)
        finally:
            fh = tray._tray_log_handle[0]
            sys.stdout, sys.stderr, threading.excepthook, handlers = saved
            root = tray.logging.getLogger()
            for h in list(root.handlers):
                if h not in handlers:
                    root.removeHandler(h)
            if fh is not None:
                fh.close()
            tray._tray_log_handle[0] = None
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("[tray] hello from the tray", text)
        self.assertIn("boom 42", text)
        self.assertIn("hello from the tray", orig_out.getvalue())   # tee

    def test_log_is_rolled_when_big(self):
        path = os.path.join(self.dir, "logs", "tray.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("x" * (tray.TRAY_LOG_MAX_BYTES + 10))
        saved = (sys.stdout, sys.stderr, threading.excepthook,
                 list(tray.logging.getLogger().handlers))
        self.addCleanup(tray.logging.getLogger().setLevel,
                        tray.logging.getLogger().level)
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            _REAL_SETUP_TRAY_LOGGING(path)
        finally:
            fh = tray._tray_log_handle[0]
            sys.stdout, sys.stderr, threading.excepthook, handlers = saved
            root = tray.logging.getLogger()
            for h in list(root.handlers):
                if h not in handlers:
                    root.removeHandler(h)
            if fh is not None:
                fh.close()
            tray._tray_log_handle[0] = None
        self.assertTrue(os.path.exists(path + ".1"))
        self.assertLess(os.path.getsize(path), 1000)

    def test_main_attaches_the_log(self):
        with mock.patch.object(sys, "argv", ["tray.py"]), \
             mock.patch.object(tray, "_load_base_icon"), \
             mock.patch.object(tray.threading, "Thread"), \
             mock.patch.object(tray.pystray, "Icon"), \
             mock.patch.object(tray, "_setup_tray_logging") as setup:
            tray.main()
        setup.assert_called_once()


class IconRenderTests(_TrayFixBase):
    """P2: unchanged icons aren't re-pushed; alert + voice-mute are shown."""

    def test_static_icon_is_pushed_once(self):
        icon = mock.MagicMock()
        icon.menu = None
        self._ready_hud(state="idle")
        with mock.patch.object(tray, "_render_icon",
                               wraps=tray._render_icon) as r:
            self._tick(icon)
            self._tick(icon)
            self._tick(icon)
        self.assertEqual(r.call_count, 1)

    def test_speaking_still_animates(self):
        icon = mock.MagicMock()
        icon.menu = None
        self._ready_hud(state="speaking", tts_amplitude=0.0)
        with mock.patch.object(tray, "_render_icon",
                               wraps=tray._render_icon) as r:
            self._tick(icon, n=2)              # two frames of one loop
        self.assertEqual(r.call_count, 2)

    def test_alert_and_tts_mute_change_the_icon(self):
        base = tray._render_icon("idle", 0).tobytes()
        self.assertNotEqual(tray._render_icon("idle", 0, alert=True).tobytes(), base)
        self.assertNotEqual(tray._render_icon("idle", 0, tts_muted=True).tobytes(),
                            base)
        s = tray._classify_state({"alert_active": True, "tts_muted": True})
        self.assertTrue(s["alert"])
        self.assertTrue(s["tts_muted"])


class CrashReportsTests(TrayTestBase):
    def test_opens_jarvis_crash_log_when_present(self):
        os.makedirs(os.path.dirname(tray.CRASH_TRACES_LOG), exist_ok=True)
        self._write(tray.CRASH_TRACES_LOG, "Fatal Python error\n")
        with mock.patch.object(tray, "_open_path") as op, \
             mock.patch.object(tray.os, "startfile", create=True) as sf:
            tray._open_event_viewer_crashes()
        op.assert_called_once_with(tray.CRASH_TRACES_LOG, "crash_traces.log")
        sf.assert_not_called()


class SendCommandConcurrencyTests(TrayTestBase):
    """P2: concurrent tray writers must not lose a click."""

    def test_parallel_senders_lose_nothing(self):
        threads = [threading.Thread(target=tray._send_command, args=(f"c{i}",))
                   for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        cmds = self._read_commands()
        self.assertEqual(sorted(c["cmd"] for c in cmds),
                         sorted(f"c{i}" for i in range(12)))
        self.assertEqual(len({c["cid"] for c in cmds}), 12)


if __name__ == "__main__":
    unittest.main()
