"""core/screen_memory.py - continuous screen memory, TEXT ONLY and cheap
(2026-10-05), against a FAKE environment: synthetic 160x90 thumbnails,
fake windows showing the synthetic research pages, a fake clock / CPU /
notification state. No real window, pixel, UIA or OCR is touched.

    python -m unittest tests.test_screen_memory
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from core import config as cfg
from core import screen_privacy as P
from core import screen_timeline as T
from core import screen_memory as SW
from tests import _screen_fakes as F


def _thumb(level=40, block=None, block_level=220):
    """BGRA 160x90 bytes: a flat grey, with an optional bright block
    (x0, y0, x1, y1) in thumbnail pixels."""
    a = np.full((90, 160, 4), level, np.uint8)
    a[..., 3] = 255
    if block:
        x0, y0, x1, y1 = block
        a[y0:y1, x0:x1, :3] = block_level
    return a.tobytes()


class FakeEnv:
    def __init__(self, windows, monitors=None):
        self.wins = list(windows)
        self.fg = self.wins[0].hwnd if self.wins else 0
        self.mons = monitors or {"middle": F.MONITORS["middle"]}
        self.thumbs = {}
        self.input_ms = 0
        self.state = 5                        # QUNS_ACCEPTS_NOTIFICATIONS
        self.cpu = 5.0
        self.game = False
        self.snapshots = []
        self.urls = {}
        self.ocr_lines = None
        self.captures = 0

    def windows(self):
        return [w.win(z) for z, w in enumerate(self.wins) if w.alive]

    def foreground(self):
        return self.fg

    def monitors(self):
        return self.mons

    def read_url(self, hwnd):
        w = next((x for x in self.wins if x.hwnd == hwnd), None)
        return w.url if w else ""

    def snapshot(self, win):
        self.snapshots.append(win.hwnd)
        w = next((x for x in self.wins if x.hwnd == win.hwnd), None)
        if w is None or not w.page:
            return None
        return F.page_snapshot(w.page, w.hwnd, w.rect, w.monitor,
                               title=w.title, url=w.url)

    def last_input_ms(self):
        return self.input_ms

    def notification_state(self):
        return self.state

    def cpu_percent(self):
        return self.cpu

    def game_mode(self):
        return self.game

    def grab_thumb(self, rect):
        for mon, r in self.mons.items():
            if tuple(r) == tuple(rect):
                return self.thumbs.get(mon, _thumb())
        return _thumb()

    def capture(self, rect, windows, urls=None):
        self.captures += 1
        self.capture_windows = list(windows)
        self.capture_urls = dict(urls or {})
        from PIL import Image
        return Image.new("RGB", (int(rect[2]), int(rect[3])))

    def ocr(self, img):
        return self.ocr_lines

    def ocr_cpu_s(self):
        return 0.0

    def health(self):
        return {"rss_mb": 1.0, "threads": 1, "handles": 1}

    def set_idle_priority(self):
        pass


class _Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="swatch_")
        self.addCleanup(shutil.rmtree, self.td, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td})
        env.start()
        self.addCleanup(env.stop)
        for k, v in (("MONITORS", F.MONITORS),
                     ("SCREENSHOT_PRIVACY_BLOCKLIST", ["bankingsite"]),
                     ("SCREEN_MEMORY_INTERVAL_S", 5.0),
                     ("SCREEN_MEMORY_CPU_PAUSE_PCT", 60.0),
                     ("SCREEN_MEMORY_MAX_CORE_PCT", 1.0)):
            p = mock.patch.object(cfg, k, v, create=True)
            p.start()
            self.addCleanup(p.stop)
        P.clear_exclusions()
        self.addCleanup(P.clear_exclusions)
        out = mock.patch("builtins.print")
        out.start()
        self.addCleanup(out.stop)
        self.tl = T.Timeline(os.path.join(self.td, "screen_timeline.db"))
        self.now = [1_000_000.0]

    def watcher(self, env):
        w = SW.Watcher(env=env, timeline=self.tl, clock=lambda: self.now[0])
        w.running = True
        return w

    def tick(self, w, dt=5.0):
        self.now[0] += dt
        return w.tick()


class DetectorTests(unittest.TestCase):
    def test_a_changed_block_maps_to_its_native_region(self):
        d = SW.ChangeDetector()
        rect = (0, 0, 2560, 1440)
        d.update("m", SW.ChangeDetector.to_grey(_thumb()), rect)
        # cells are 10x10 thumb px; change cells x 2..4, y 1..2
        n, region = d.update("m", SW.ChangeDetector.to_grey(
            _thumb(block=(20, 10, 50, 30))), rect)
        self.assertEqual(n, 6)
        x, y, w, h = region
        self.assertEqual((x, y), (2 * 160.0, 1 * 160.0))
        self.assertEqual((w, h), (3 * 160.0, 2 * 160.0))

    def test_a_tiny_change_is_not_a_change(self):
        d = SW.ChangeDetector()
        d.update("m", SW.ChangeDetector.to_grey(_thumb()), (0, 0, 2560, 1440))
        n, region = d.update("m", SW.ChangeDetector.to_grey(
            _thumb(block=(0, 0, 10, 10))), (0, 0, 2560, 1440))
        self.assertIsNone(region)

    def test_a_playing_video_stops_triggering(self):
        d = SW.ChangeDetector()
        rect = (0, 0, 2560, 1440)
        d.update("m", SW.ChangeDetector.to_grey(_thumb()), rect)
        regions = []
        for i in range(8):
            lvl = 220 if i % 2 == 0 else 120          # a block that keeps changing
            regions.append(d.update("m", SW.ChangeDetector.to_grey(
                _thumb(block=(40, 20, 80, 60), block_level=lvl)), rect)[1])
        self.assertIsNotNone(regions[0])
        self.assertTrue(all(r is None for r in regions[5:]), regions)

    def test_processing_cost_per_tick(self):
        d = SW.ChangeDetector()
        frames = [SW.ChangeDetector.to_grey(_thumb(block=(i, 5, i + 40, 50)))
                  for i in range(0, 120, 6)]
        t0 = time.perf_counter()
        for f in frames:
            for mon in ("a", "b", "c", "d"):
                d.update(mon, f, (0, 0, 2560, 1440))
        ms = (time.perf_counter() - t0) * 1000 / len(frames)
        print(f"\n  [screen-memory] detector: {ms:.2f} ms per 4-monitor tick")
        self.assertLess(ms, 50.0)


class TickTests(_Base):
    def test_windows_focus_and_the_address_are_recorded(self):
        home = F.FakeWindow(101, "home_dark", "middle",
                            url="https://www.youtube.com/watch?v=abc&si=TRACK")
        env = FakeEnv([home])
        w = self.watcher(env)
        self.tick(w)
        rows = self.tl.query()
        self.assertTrue(any(r["source"] == "title" for r in rows))
        self.assertTrue(any(r["source"] == "focus" for r in rows))
        urls = {r["url"] for r in rows if r["url"]}
        self.assertEqual(urls, {"https://www.youtube.com/watch?v=abc"})

    def test_page_text_is_read_on_change_and_new_lines_only(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        env = FakeEnv([home])
        w = self.watcher(env)
        self.tick(w)                                   # first grab (baseline)
        env.input_ms += 100
        env.thumbs["middle"] = _thumb(block=(20, 10, 90, 60))
        self.tick(w)
        uia = [r for r in self.tl.query(sources=("uia",))]
        self.assertEqual(len(uia), 1)
        self.assertIn("I Survived 7 Days In An Abandoned City", uia[0]["text"])
        # The same page changes again 15 s later: nothing new to store.
        env.input_ms += 100
        env.thumbs["middle"] = _thumb()
        self.tick(w, dt=15.0)
        self.assertEqual(len(self.tl.query(sources=("uia",))), 1)

    def test_no_input_no_grab_until_30_s(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        env = FakeEnv([home])
        w = self.watcher(env)
        r1 = self.tick(w)
        self.assertEqual(r1["grabs"], 1)
        r2 = self.tick(w)
        self.assertEqual(r2["grabs"], 0)
        r3 = self.tick(w, dt=30.0)
        self.assertEqual(r3["grabs"], 1)

    def test_a_private_window_is_never_read_or_stored(self):
        bank = F.FakeWindow(101, "home_dark", "middle",
                            title="Accounts - BankingSite - Google Chrome")
        env = FakeEnv([bank])
        w = self.watcher(env)
        self.tick(w)
        env.input_ms += 1
        env.thumbs["middle"] = _thumb(block=(20, 10, 90, 60))
        self.tick(w)
        self.assertEqual(env.snapshots, [])
        self.assertEqual(self.tl.count(), 0)

    def test_an_excluded_app_is_never_read(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        env = FakeEnv([home])
        w = self.watcher(env)
        w.exclude_app("chrome")
        self.assertIn("chrome", json.load(open(SW.state_path()))["apps"])
        with mock.patch.object(SW, "load_state",
                               return_value={"apps": ["chrome"]}):
            P.set_exclusions_provider(SW.load_state)
            self.tick(w)
            env.input_ms += 1
            env.thumbs["middle"] = _thumb(block=(20, 10, 90, 60))
            self.tick(w)
        P.set_exclusions_provider(None)
        self.assertEqual(env.snapshots, [])
        self.assertEqual(self.tl.count(), 0)

    def test_dont_watch_this_forgets_the_last_five_minutes_of_it(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        env = FakeEnv([home])
        w = self.watcher(env)
        self.tick(w)
        self.assertGreater(self.tl.count(), 0)
        with mock.patch.object(w, "clock", lambda: time.time()):
            self.tl.add_now(ts=time.time(), hwnd=101, title="x", text="y")
            line = w.exclude_foreground()
        self.assertIn("won't watch", line)
        self.assertEqual(self.tl.query(since=time.time() - 300), [])
        self.assertTrue(P.excluded({"hwnd": 101}))

    def test_a_password_page_is_skipped(self):
        signin = F.FakeWindow(101, "signin_light", "middle",
                              url="https://accounts.example.com/signin")
        env = FakeEnv([signin])
        w = self.watcher(env)
        self.tick(w)
        env.input_ms += 1
        env.thumbs["middle"] = _thumb(block=(20, 10, 90, 60))
        self.tick(w)
        self.assertEqual(self.tl.query(sources=("uia", "ocr")), [])

    def test_ocr_only_when_uia_is_thin_and_at_most_every_10_s(self):
        other = F.FakeWindow(101, None, "middle", title="Game Launcher",
                             process="launcher.exe")
        env = FakeEnv([other])
        env.ocr_lines = [{"t": "Play Now", "rect": (0, 0, 10, 10)}]
        w = self.watcher(env)
        self.tick(w)
        env.input_ms += 1
        env.thumbs["middle"] = _thumb(block=(20, 10, 90, 60))
        self.tick(w)
        self.assertEqual(env.captures, 1)
        rows = self.tl.query(sources=("ocr",))
        self.assertEqual(rows[0]["text"], "Play Now")
        env.input_ms += 1
        env.thumbs["middle"] = _thumb()
        self.tick(w, dt=5.0)                       # < 10 s: no second OCR
        self.assertEqual(env.captures, 1)


class PauseTests(_Base):
    def _env(self):
        return FakeEnv([F.FakeWindow(101, "home_dark", "middle")])

    def test_game_fullscreen_locked_busy(self):
        for attr, val, want in (("game", True, "a game is running"),
                                ("state", SW.QUNS_D3D_FULL, "full-screen"),
                                ("state", SW.QUNS_PRESENTATION, "full-screen"),
                                ("state", SW.QUNS_NOT_PRESENT, "locked")):
            env = self._env()
            setattr(env, attr, val)
            w = self.watcher(env)
            r = self.tick(w)
            self.assertIn(want, r["paused"], (attr, val))
            self.assertEqual(self.tl.count(), 0)

    def test_busy_with_a_browser_in_front_stores_titles_only(self):
        env = self._env()
        env.state = SW.QUNS_BUSY
        w = self.watcher(env)
        r = self.tick(w)
        self.assertEqual(r["paused"], "")
        self.assertEqual(r["grabs"], 0)
        self.assertTrue(self.tl.query(sources=("title",)))

    def test_cpu_busy_over_three_ticks(self):
        env = self._env()
        env.cpu = 95.0
        w = self.watcher(env)
        self.assertEqual(self.tick(w)["paused"], "")
        self.assertEqual(self.tick(w)["paused"], "")
        self.assertIn("CPU", self.tick(w)["paused"])

    def test_owner_pause_and_resume(self):
        env = self._env()
        w = self.watcher(env)
        line = w.pause(10)
        self.assertIn("10 minutes", line)
        self.assertIn("asked", self.tick(w)["paused"])
        self.assertIn("paused", w.status())
        self.now[0] += 601
        self.assertEqual(self.tick(w)["paused"], "")
        w.pause()
        self.assertIn("asked", self.tick(w, dt=3600)["paused"])
        self.assertIn("Watching again", w.unpause())
        self.assertEqual(self.tick(w)["paused"], "")

    def test_governor_backs_off_then_pauses(self):
        env = self._env()
        w = self.watcher(env)
        w.interval = 5.0
        for _ in range(3):
            self.now[0] += 5
            w._govern(0.2, 0.0)              # 4% of a core
        self.assertGreater(w.gov_pause_until, self.now[0])
        w2 = self.watcher(env)
        w2.interval = 5.0
        for _ in range(3):
            self.now[0] += 5
            w2._govern(0.08, 0.0)            # 1.6%: back off, no pause
        self.assertEqual(w2.interval, 10.0)
        self.assertEqual(w2.gov_pause_until, 0.0)

    def test_backing_off_before_the_first_interval_is_set(self):
        # Full-suite run 2026-10-05: a slow machine backed off while
        # interval was still None and the log line raised TypeError.
        w = self.watcher(self._env())
        w.interval = None
        for _ in range(3):
            self.now[0] += 5
            w._govern(0.08, 0.0)
        self.assertEqual(w.interval, 2 * w.base_interval())


class LifecycleTests(_Base):
    def test_never_autostarts_in_a_test_process(self):
        w = SW.Watcher()                     # the REAL environment
        self.assertFalse(w.start())
        self.assertFalse(w.running)

    def test_forget_purges_rows_trace_and_scenes(self):
        now = time.time()
        tl = T.get()
        tl.add_now(ts=now - 30, title="recent", text="x")
        tl.add_now(ts=now - 7200, title="old", text="y")
        from core import grounded_click as G
        G.reset_state()
        line = SW.forget({"seconds": 3600}, now=now)
        self.assertIn("forgotten the last hour", line)
        self.assertIn("1 note", line)
        self.assertEqual([r["title"] for r in tl.query()], ["old"])

    def test_status_line(self):
        w = self.watcher(FakeEnv([]))
        self.assertIn("text-only record", w.status())
        w.running = False
        self.assertIn("off", w.status())


class BudgetTests(_Base):
    def test_measured_cost_on_synthetic_input(self):
        """The watcher's own CPU on synthetic input (the GDI grab itself is
        not measured here - a test never grabs the screen)."""
        wins = [F.FakeWindow(100 + i, "home_dark", m)
                for i, m in enumerate(("left", "middle", "right", "top"))]
        env = FakeEnv(wins, monitors=dict(F.MONITORS))
        w = self.watcher(env)
        ticks = 60
        t0 = time.thread_time()
        w0 = time.perf_counter()
        for i in range(ticks):
            env.input_ms += 1
            for mon in env.mons:
                env.thumbs[mon] = _thumb(block=((i * 7) % 120, 10,
                                                (i * 7) % 120 + 30, 50))
            self.tick(w)
        cpu = time.thread_time() - t0
        wall = time.perf_counter() - w0
        per_tick_ms = cpu * 1000 / ticks
        # at one tick per 5 s, % of one core:
        pct = per_tick_ms / 5000 * 100
        import sys
        sys.__stdout__.write(
            f"\n  [screen-memory] synthetic 4-monitor ticks: {per_tick_ms:.1f} "
            f"ms CPU/tick (wall {wall * 1000 / ticks:.1f} ms) = {pct:.3f}% "
            f"of one core at 5 s; rows={self.tl.count()}\n")
        self.assertLess(pct, 1.0)


if __name__ == "__main__":
    unittest.main()
