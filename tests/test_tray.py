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
import sys
import tempfile
import threading
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


# --------------------------------------------------------------------------- #
# Shared base: redirect every path constant into a temp dir + reset globals.
# --------------------------------------------------------------------------- #
class TrayTestBase(unittest.TestCase):
    # tray path constants that point at the real repo — all redirected per-test.
    _PATH_ATTRS = (
        "PROJECT_DIR", "HUD_STATE_FILE", "TRAY_COMMANDS_FILE", "TODO_FILE",
        "LOGS_DIR", "ASSETS_DIR", "DEFAULT_ICON_PATH", "DATA_DIR",
        "CHANGELOG_FILE", "VERSION_FILE", "INSTANCES_FILE", "PIPELINE_LOCK_FILE",
        "OVERNIGHT_FLAG", "MEMORY_FACTS_FILE", "SETTINGS_WINDOW", "SHOW_LOG_PS1",
        "HUD_SCRIPT",
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

        # Map each path constant to a sibling under the temp dir, preserving the
        # original basename so behaviour that keys off the filename still holds.
        for attr in self._PATH_ATTRS:
            orig = getattr(tray, attr)
            base = os.path.basename(orig) if orig else attr
            patcher = mock.patch.object(tray, attr, os.path.join(self.dir, base))
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
    def _assert_cmd(self, fn, expected_cmd, **expected_kw):
        fn(mock.Mock(), mock.Mock())
        cmd = self._last_command()
        self.assertEqual(cmd["cmd"], expected_cmd)
        for k, v in expected_kw.items():
            self.assertEqual(cmd[k], v)

    def test_open_hud(self):
        self._assert_cmd(tray._on_open_hud, "open_hud")

    def test_restart(self):
        self._assert_cmd(tray._on_restart, "restart")

    def test_mute_tts(self):
        self._assert_cmd(tray._on_mute_tts, "mute_tts_toggle")

    def test_mute_mic(self):
        # New mic-mute toggle — must emit EXACTLY this command name, which the
        # bobert capture-loop handler keys off of.
        self._assert_cmd(tray._on_mute_mic, "mic_mute_toggle")

    def test_ambient_mode(self):
        self._assert_cmd(tray._on_ambient_mode, "ambient_mode_toggle")

    def test_force_upgrade(self):
        self._assert_cmd(tray._on_force_upgrade, "trigger_overnight")

    def test_shutdown(self):
        self._assert_cmd(tray._on_shutdown_jarvis, "shutdown_jarvis")

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

    def test_reset_llm_cache(self):
        self._assert_cmd(tray._on_reset_llm_cache, "reset_llm_cache")

    def test_switch_anthropic(self):
        self._assert_cmd(tray._on_switch_anthropic, "switch_llm", backend="anthropic")

    def test_switch_qwen(self):
        self._assert_cmd(tray._on_switch_qwen, "switch_llm", backend="qwen2.5:14b")

    def test_switch_llama(self):
        self._assert_cmd(tray._on_switch_llama, "switch_llm", backend="llama3.1:8b")

    def test_switch_other(self):
        self._assert_cmd(tray._on_switch_other_llm, "switch_llm_picker")

    def test_toggle_debug(self):
        self._assert_cmd(tray._on_toggle_debug_mode, "debug_mode_toggle")

    def test_show_llm_stats(self):
        self._assert_cmd(tray._on_show_llm_stats, "show_llm_stats")

    def test_clear_llm_cache(self):
        self._assert_cmd(tray._on_clear_llm_cache, "clear_llm_cache")

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
        self._assert_cmd(tray._on_reset_memory, "reset_memory")

    def test_export_memory(self):
        self._assert_cmd(tray._on_export_memory, "export_memory")

    def test_forget_last_hour(self):
        self._assert_cmd(tray._on_forget_last_hour, "forget_last_hour")

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

    def test_open_changelog_present(self):
        self._write(tray.CHANGELOG_FILE, "## v1.0.0 — 2026-01-01 00:00\n")
        with mock.patch.object(tray, "_open_path") as op:
            tray._on_open_changelog(mock.Mock(), mock.Mock())
        op.assert_called_once()
        self.assertEqual(op.call_args.args[0], tray.CHANGELOG_FILE)

    def test_open_changelog_absent_no_open(self):
        with mock.patch.object(tray, "_open_path") as op:
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
             mock.patch.object(tray.subprocess, "run") as run:
            tray._on_show_dossier(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        run.assert_called_once()

    def test_about_spawns_thread_and_runs_subprocess(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run") as run:
            tray._on_about(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        run.assert_called_once()

    def test_summary_spawns_thread_and_runs_subprocess(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run") as run:
            tray._on_show_today_summary(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        run.assert_called_once()

    def test_dossier_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run",
                               side_effect=OSError("spawn fail")):
            tray._on_show_dossier(mock.Mock(), mock.Mock())
            self._run_thread_target(T)  # must not raise

    def test_about_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run",
                               side_effect=OSError("spawn fail")):
            tray._on_about(mock.Mock(), mock.Mock())
            self._run_thread_target(T)  # must not raise

    def test_summary_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run",
                               side_effect=OSError("spawn fail")):
            tray._on_show_today_summary(mock.Mock(), mock.Mock())
            self._run_thread_target(T)  # must not raise

    def test_queue_task_spawns_and_appends_result(self):
        proc = mock.Mock()
        proc.stdout = "do the dishes"
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run", return_value=proc):
            tray._on_queue_task(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        with open(tray.TODO_FILE, encoding="utf-8") as f:
            self.assertIn("do the dishes", f.read())

    def test_queue_task_empty_result_no_append(self):
        proc = mock.Mock()
        proc.stdout = "   "
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run", return_value=proc):
            tray._on_queue_task(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        self.assertFalse(os.path.exists(tray.TODO_FILE))

    def test_queue_task_subprocess_error_swallowed(self):
        with mock.patch.object(tray.threading, "Thread") as T, \
             mock.patch.object(tray.subprocess, "run",
                               side_effect=OSError("spawn fail")):
            tray._on_queue_task(mock.Mock(), mock.Mock())
            self._run_thread_target(T)
        self.assertFalse(os.path.exists(tray.TODO_FILE))


class SettingsCallbackTests(TrayTestBase):
    """Each Settings tab callback spawns a daemon thread targeting
    _open_settings_window with the tab name as a positional arg."""

    def _assert_settings(self, fn, expected_tab):
        with mock.patch.object(tray.threading, "Thread") as T:
            inst = T.return_value
            fn(mock.Mock(), mock.Mock())
        inst.start.assert_called_once()
        self.assertEqual(T.call_args.kwargs.get("target"),
                         tray._open_settings_window)
        self.assertEqual(T.call_args.kwargs.get("args"), (expected_tab,))

    def test_voice(self):
        self._assert_settings(tray._on_settings_voice, "voice")

    def test_ai(self):
        self._assert_settings(tray._on_settings_ai, "ai")

    def test_privacy(self):
        self._assert_settings(tray._on_settings_privacy, "privacy")

    def test_integrations(self):
        self._assert_settings(tray._on_settings_integrations, "integrations")

    def test_advanced(self):
        self._assert_settings(tray._on_settings_advanced, "advanced")


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


class OpenSettingsWindowTests(TrayTestBase):
    def test_spawns_window_when_present(self):
        self._write(tray.SETTINGS_WINDOW, "# settings")
        with mock.patch.object(tray.subprocess, "Popen") as P:
            tray._open_settings_window("voice")
        P.assert_called_once()
        argv = P.call_args.args[0]
        self.assertIn(tray.SETTINGS_WINDOW, argv)
        self.assertIn("--tab", argv)
        self.assertIn("voice", argv)

    def test_no_tab_arg_when_blank(self):
        self._write(tray.SETTINGS_WINDOW, "# settings")
        with mock.patch.object(tray.subprocess, "Popen") as P:
            tray._open_settings_window("")
        argv = P.call_args.args[0]
        self.assertNotIn("--tab", argv)

    def test_popen_error_falls_through_to_json(self):
        self._write(tray.SETTINGS_WINDOW, "# settings")
        fallback = os.path.join(tray.DATA_DIR, "user_settings.json")
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(fallback, "{}")
        with mock.patch.object(tray.subprocess, "Popen",
                               side_effect=OSError("spawn fail")), \
             mock.patch.object(tray, "_open_path") as op:
            tray._open_settings_window("ai")
        op.assert_called_once()
        self.assertEqual(op.call_args.args[0], fallback)

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

    def test_uptime_from_prod_instance(self):
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({
            "x": {"role": "prod", "started_at": tray.time.time() - 120},
        }))
        self.assertGreaterEqual(tray._read_uptime_seconds(), 100)

    def test_uptime_any_instance_when_no_prod(self):
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({
            "x": {"role": "dev", "started_at": tray.time.time() - 60},
        }))
        self.assertGreaterEqual(tray._read_uptime_seconds(), 50)

    def test_uptime_skips_non_dict_entries(self):
        # A non-dict instance entry must be skipped (the `continue` branch) and
        # the scan must still find the dict entry that follows.
        os.makedirs(tray.DATA_DIR, exist_ok=True)
        self._write(tray.INSTANCES_FILE, json.dumps({
            "bad": "not-a-dict",
            "good": {"role": "prod", "started_at": tray.time.time() - 70},
        }))
        self.assertGreaterEqual(tray._read_uptime_seconds(), 50)

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

    def test_about_lines_structure(self):
        self._write(tray.CHANGELOG_FILE, "## v2.0.0 — 2026-06-01 10:00\n")
        lines = tray._about_lines()
        self.assertEqual(lines[0], "J.A.R.V.I.S.")
        joined = "\n".join(lines)
        self.assertIn("Version:", joined)
        self.assertIn("v2.0.0", joined)
        self.assertIn("Uptime:", joined)


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
        tray._on_quit(icon, mock.Mock())
        self.assertTrue(tray._stop_event.is_set())
        icon.stop.assert_called_once()

    def test_quit_icon_stop_error_swallowed(self):
        icon = mock.Mock()
        icon.stop.side_effect = RuntimeError("already stopped")
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
                                             "bambu_active": True}), \
             mock.patch.object(tray, "_count_pending_tasks", return_value=4):
            tray._animate(icon)
        self.assertIsNotNone(icon.icon)
        self.assertIn("listen:awake", icon.title)
        self.assertIn("tts:speaking", icon.title)
        self.assertIn("queue:4", icon.title)
        self.assertIn("bambu:printing", icon.title)

    def test_title_reflects_muted_and_standby(self):
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state",
                               return_value={"state": "standby",
                                             "mic_muted": True}), \
             mock.patch.object(tray, "_count_pending_tasks", return_value=0):
            tray._animate(icon)
        self.assertIn("listen:muted", icon.title)
        self.assertIn("bambu:idle", icon.title)

    def test_title_standby_label_when_not_muted(self):
        # Non-muted standby must reach the "standby" listen label branch.
        icon = mock.Mock()
        ev = self._one_shot_event()
        with mock.patch.object(tray, "_stop_event", ev), \
             mock.patch.object(tray, "_parent_alive", return_value=True), \
             mock.patch.object(tray, "_read_hud_state",
                               return_value={"state": "sleeping"}), \
             mock.patch.object(tray, "_count_pending_tasks", return_value=0):
            tray._animate(icon)
        self.assertIn("listen:standby", icon.title)
        self.assertIn("tts:quiet", icon.title)

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
        for expected in ("Pause Listening", "Mute TTS", "Mute Mic",
                         "Ambient Mode", "Open HUD",
                         "Run Upgrade Now", "Restart JARVIS", "Shut Down JARVIS",
                         "Power tools", "AI", "Audio", "Memory", "Diagnostics",
                         "Settings", "About JARVIS", "Show Today's Summary",
                         "Queue Task…", "Quit Tray Only"):
            self.assertIn(expected, texts, expected)

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
             mock.patch.object(tray.subprocess, "run"):
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


if __name__ == "__main__":
    unittest.main()
