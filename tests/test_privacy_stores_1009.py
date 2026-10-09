"""The stores the 2026-10-05 branches added, against guest mode, the memory
wipes and a shutdown (review of the v2.0.182 integration, 2026-10-09).

  * guest mode (core.guest_mode: visitors are in the room, nothing said is
    kept) - the vision trace keeps only a bare marker, the screen timeline
    and its watcher keep nothing, a developer note is refused, and the
    clone voice keeps no take on disk and counts no line;
  * reset_memory / forget_last_hour also forget the screen memory and the
    clone voice's cache (its takes and the line ledger's plain text);
  * the teardown lands the queued writes (_flush_persistent_stores).

    python -m unittest tests.test_privacy_stores_1009
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

import core.actions as A
import core.long_term_memory as LTM
from core import clone_render_cache as crc
from core import clone_seed as cseed
from core import clone_voice_client as cvc
from core import config as cfg
from core import dev_notes as DN
from core import guest_mode as GM
from core import screen_timeline as T
from core import vision_trace as VT
from tests.test_actions_sec3 import _base_bc, _patch_bc
from tests.test_clone_render_cache import _ClientBase, fit_wav
from tests.test_screen_memory import FakeEnv
from tests.test_screen_memory import _Base as _WatchBase
from tests import _screen_fakes as F


def _guest_on(test):
    GM.set_on(True)
    test.addCleanup(GM.set_on, False)


class _DataDir(unittest.TestCase):
    """A throwaway JARVIS_DATA_DIR, the vision trace on and fresh."""

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="privacy1009_")
        self.addCleanup(shutil.rmtree, self.td, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td})
        env.start()
        self.addCleanup(env.stop)
        p = mock.patch.object(cfg, "VISION_TRACE", "on", create=True)
        p.start()
        self.addCleanup(p.stop)
        VT._reset_for_tests()
        self.addCleanup(VT._reset_for_tests)
        out = mock.patch("builtins.print")
        out.start()
        self.addCleanup(out.stop)

    def timeline(self):
        tl = T.get()
        self.addCleanup(tl.close, 2.0)
        return tl


# ════════════════════════════════════════════════════════════════════════════
#  Guest mode
# ════════════════════════════════════════════════════════════════════════════
class GuestModeScreenStoresTests(_DataDir):
    def test_a_traced_step_keeps_only_the_bare_marker(self):
        _guest_on(self)
        VT.record("see_screen", utterance="my guest asked what the memo says",
                  outcome="read", raw_answer="the board memo text")
        self.assertTrue(VT.flush(5.0))
        entries = VT.read_index()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["privacy"], VT.GUEST_SKIP)
        blob = json.dumps(e)
        self.assertNotIn("guest asked", blob)
        self.assertNotIn("memo", blob)

    def test_out_of_guest_mode_the_step_is_traced_as_before(self):
        VT.record("see_screen", utterance="what does the memo say",
                  outcome="read", raw_answer="the memo text")
        self.assertTrue(VT.flush(5.0))
        e = VT.read_index()[-1]
        self.assertEqual(e["privacy"], "")
        self.assertEqual(e["utterance"], "what does the memo say")

    def test_the_timeline_keeps_no_row(self):
        tl = self.timeline()
        _guest_on(self)
        self.assertFalse(T.add(source="look", title="see_screen: middle",
                               text="what the visitor was shown"))
        tl.flush(3.0)
        self.assertEqual(tl.count(), 0)
        GM.set_on(False)
        self.assertTrue(T.add(source="look", title="see_screen: middle",
                              text="the owner's own page"))
        self.assertTrue(tl.flush(3.0))
        self.assertEqual(tl.count(), 1)

    def test_a_developer_note_is_refused_and_nothing_is_written(self):
        _guest_on(self)
        self.assertIsNone(DN.add_note("fix the click", utterance="tell Claude "
                                      "to fix the click"))
        self.assertEqual(A._act_note_for_claude("fix the click"),
                         DN.GUEST_LINE)
        self.assertFalse(os.path.exists(DN.notes_path()))
        GM.set_on(False)
        self.assertEqual(A._act_note_for_claude("fix the click"),
                         DN.SPOKEN_LINE)
        self.assertEqual(len(DN.read_notes()), 1)


class GuestModeWatcherTests(_WatchBase):
    def test_the_watcher_holds_off_while_guests_are_in(self):
        env = FakeEnv([F.FakeWindow(101, "home_dark", "middle")])
        w = self.watcher(env)
        _guest_on(self)
        r = self.tick(w)
        self.assertEqual(r["paused"], "guest mode is on")
        self.assertEqual(r["rows"], 0)
        self.assertEqual(self.tl.count(), 0)
        self.assertIn("guest mode is on", w.status())
        GM.set_on(False)
        self.assertEqual(self.tick(w)["paused"], "")


class GuestModeCloneTests(_ClientBase):
    TEXT = "Your appointment with Doctor Example is at nine, sir."

    def test_no_take_on_disk_and_no_ledger_line(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        _guest_on(self)
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(self.files(), [])
        # Said again: from the memory tier (this run only), still no line.
        self.assertEqual(c.render(self.TEXT, 2.5).cache, "mem")
        self.assertEqual(c.ledger.count(self.TEXT), 0)
        self.assertEqual(c.ledger.seeds(), [])
        self.assertEqual(c.cache_stats().get("guest_not_kept"), 1)

    def test_out_of_guest_mode_the_take_is_kept(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(c.ledger.count(self.TEXT), 1)


# ════════════════════════════════════════════════════════════════════════════
#  The wipes' share: ledger, render cache, client
# ════════════════════════════════════════════════════════════════════════════
class LedgerWipeTests(unittest.TestCase):
    def test_forget_since_drops_the_window_only(self):
        now = [1_000_000.0]
        led = cseed.LineLedger(None, wall=lambda: now[0])
        led.record("Certainly, sir.")
        led.record("Certainly, sir.")                 # two hours ago
        now[0] += 7200
        led.record("Your table is booked for eight, sir.")
        led.record("Your table is booked for eight, sir.")   # just now
        self.assertEqual(led.forget_since(now[0] - 3600, now[0]), 1)
        self.assertEqual(led.count("Your table is booked for eight, sir."), 0)
        self.assertEqual(led.seeds(), ["Certainly, sir."])

    def test_a_line_with_no_time_is_older_and_kept(self):
        led = cseed.LineLedger(None)
        led.merge_counts({"One moment, sir.": 4})       # the history bootstrap
        self.assertEqual(led.forget_since(0.0), 0)
        self.assertEqual(led.count("One moment, sir."), 4)

    def test_clear_keeps_only_the_never_seed_marks_and_saves_no_text(self):
        td = tempfile.mkdtemp(prefix="ledger1009_")
        self.addCleanup(shutil.rmtree, td, True)
        path = os.path.join(td, cseed.LEDGER_FILE)
        led = cseed.LineLedger(path)
        for t in ("Your appointment is at nine, sir.",) * 2 + (
                "That line sounded wrong.",):
            led.record(t)
        led.forget("That line sounded wrong.")
        self.assertTrue(led.save_if_dirty())
        self.assertEqual(led.clear(), 2)
        self.assertTrue(led.save_if_dirty())
        raw = open(path, encoding="utf-8").read()
        self.assertNotIn("appointment", raw)
        again = cseed.LineLedger(path)
        self.assertEqual(again.count("Your appointment is at nine, sir."), 0)
        self.assertTrue(again.is_forgotten("That line sounded wrong."))

    def test_the_time_survives_a_reload(self):
        td = tempfile.mkdtemp(prefix="ledger1009_")
        self.addCleanup(shutil.rmtree, td, True)
        path = os.path.join(td, cseed.LEDGER_FILE)
        led = cseed.LineLedger(path, wall=lambda: 1_000_000.0)
        led.record("Certainly, sir.")
        self.assertTrue(led.save_if_dirty())
        again = cseed.LineLedger(path)
        self.assertEqual(again.forget_since(999_999.0, 1_000_001.0), 1)


class RenderCacheWipeTests(unittest.TestCase):
    PREFIX = "a" * 16

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="crc1009_")
        self.addCleanup(shutil.rmtree, self.td, True)
        self.store = crc.CloneRenderCache()
        self.addCleanup(self.store.close)
        self.assertTrue(self.store.attach(self.td))

    def _key(self, n):
        return f"{n:064x}"

    def _put(self, n):
        self.assertTrue(self.store.disk_put(self._key(n), self.PREFIX,
                                            np.ones(800, np.int16)))
        self.assertTrue(self.store.flush(5.0))
        return os.path.join(self.td, f"{self.PREFIX}_{self._key(n)}.npy")

    def test_a_windowed_wipe_removes_that_hours_takes_only(self):
        old = self._put(1)
        os.utime(old, (time.time() - 7200, time.time() - 7200))
        new = self._put(2)
        self.store.mem_put(self._key(3), np.ones(100, np.float32), 24000,
                           self.PREFIX)
        n = self.store.wipe(time.time() - 3600, time.time() + 1)
        self.assertEqual(n, 1)                 # the memory tier is not counted
        self.assertTrue(os.path.exists(old))
        self.assertFalse(os.path.exists(new))
        self.assertEqual(self.store.mem_len(), 0)

    def test_a_full_wipe_removes_every_take(self):
        self._put(1)
        self.store.mem_put(self._key(3), np.ones(100, np.float32), 24000,
                           self.PREFIX)
        self.assertEqual(self.store.wipe(None), 2)
        self.assertEqual([n for n in os.listdir(self.td)
                          if n.endswith(".npy")], [])

    def test_a_write_queued_before_the_wipe_never_lands(self):
        go = threading.Event()
        started = threading.Event()

        def keep():
            started.set()
            go.wait(5.0)
            return True

        self.assertTrue(self.store.disk_put(self._key(7), self.PREFIX,
                                            np.ones(800, np.int16),
                                            keep=keep))
        self.assertTrue(started.wait(5.0))
        self.store.wipe(None)
        go.set()
        self.assertTrue(self.store.flush(5.0))
        self.assertEqual([n for n in os.listdir(self.td)
                          if n.endswith(".npy")], [])
        # A write queued AFTER the wipe lands as usual.
        self.assertTrue(os.path.exists(self._put(8)))


class ClientWipeTests(_ClientBase):
    OLD = "Certainly, sir."
    NEW = "Your appointment with Doctor Example is at nine, sir."

    def _say(self, c, text):
        """Voiced once (a take on disk) and counted once more: a line said
        twice, its text in the ledger (two renders inside LEDGER_DEDUPE_S
        count once)."""
        self.assertTrue(c.render(text, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        c.ledger.record(text)
        self.assertEqual(c.ledger.seeds().count(text), 1)

    def test_the_last_hour_goes_and_older_takes_stay(self):
        srv = self.server(wav_for={self.OLD: fit_wav(self.OLD),
                                   self.NEW: fit_wav(self.NEW)})
        c = self.client(srv)
        self._say(c, self.OLD)
        # ... two hours ago.
        for name in self.files():
            p = os.path.join(self.dir, name)
            os.utime(p, (time.time() - 7200, time.time() - 7200))
        h = cseed.text_hash(self.OLD)
        c.ledger._lines[h]["w"] = int(time.time() - 7200)
        self._say(c, self.NEW)
        self.assertEqual(len(self.files()), 2)
        out = c.wipe(time.time() - 3600)
        self.assertEqual(out, {"takes": 1, "lines": 1})
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(c.ledger.count(self.NEW), 0)
        self.assertEqual(c.ledger.count(self.OLD), 2)
        raw = open(os.path.join(self.dir, cseed.LEDGER_FILE),
                   encoding="utf-8").read()
        self.assertNotIn("Doctor Example", raw)       # saved at once
        self.assertEqual(c.forget_last_reply(), [])  # its record went too

    def test_a_full_wipe_leaves_no_take_and_no_line(self):
        srv = self.server(wav_for={self.NEW: fit_wav(self.NEW)})
        c = self.client(srv)
        self._say(c, self.NEW)
        out = c.wipe(None)
        self.assertEqual(out, {"takes": 1, "lines": 1})
        self.assertEqual(self.files(), [])
        self.assertEqual(len(c.ledger), 0)
        self.assertFalse(c.is_cached(self.NEW))


# ════════════════════════════════════════════════════════════════════════════
#  reset_memory / forget_last_hour reach the new stores
# ════════════════════════════════════════════════════════════════════════════
class MemoryWipeActionTests(_ClientBase):
    TEXT = "Your appointment with Doctor Example is at nine, sir."

    def setUp(self):
        super().setUp()
        self.data = tempfile.mkdtemp(prefix="wipe1009_")
        self.addCleanup(shutil.rmtree, self.data, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.data})
        env.start()
        self.addCleanup(env.stop)
        p = mock.patch.object(cfg, "VISION_TRACE", "on", create=True)
        p.start()
        self.addCleanup(p.stop)
        VT._reset_for_tests()
        self.addCleanup(VT._reset_for_tests)
        out = mock.patch("builtins.print")
        out.start()
        self.addCleanup(out.stop)
        self.tl = T.get()
        self.addCleanup(self.tl.close, 2.0)
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        self.c = self.client(srv)
        p = mock.patch.object(cvc, "CLIENT", self.c)
        p.start()
        self.addCleanup(p.stop)
        # What the screen and the voice kept in the last hour.
        self.assertTrue(self.tl.add_now(source="look", title="see_screen: "
                                        "middle", text="salary sheet Q3"))
        VT.record("see_screen", utterance="read me the salary sheet",
                  outcome="read", raw_answer="Q3 salaries")
        self.assertTrue(VT.flush(5.0))
        self.assertTrue(self.c.render(self.TEXT, 2.5).ok)
        self.assertTrue(self.c.store.flush(5.0))
        self.c.ledger.record(self.TEXT)              # said twice: its text
        self.assertTrue(self.c.ledger.save_if_dirty())
        self.bc = _base_bc(self.data)
        self.bc._memory_lock = mock.MagicMock()
        # The live monolith's own cached looks are not this test's.
        mods = mock.patch.dict(sys.modules, {"bobert_companion": None})
        mods.start()
        self.addCleanup(mods.stop)

    def _left(self):
        raw = open(os.path.join(self.dir, cseed.LEDGER_FILE),
                   encoding="utf-8").read()
        return (self.tl.count(), len(VT.read_index()), len(self.files()),
                "Doctor Example" in raw)

    def test_forget_last_hour(self):
        self.assertEqual(self._left(), (1, 1, 1, True))
        self.bc.load_memory.return_value = {"topics": [], "sessions": []}
        self.bc.pattern_memory.forget_voice_commands_since.return_value = 0
        with _patch_bc(self.bc), \
                mock.patch.object(LTM, "forget_since",
                                  return_value={"episodes": 0, "facts": 0,
                                                "working": 0}):
            out = A._act_forget_last_hour()
        self.assertEqual(self._left(), (0, 0, 0, False))
        self.assertIn("2 screen record(s)", out)
        self.assertIn("1 cached voice line(s)", out)
        self.assertTrue(out.endswith("from the last hour"), out)

    def test_reset_memory(self):
        self.assertEqual(self._left(), (1, 1, 1, True))
        self.bc.MEMORY_FILE = os.path.join(self.data, "bobert_memory.json")
        self.bc._empty_memory.return_value = {"facts": []}
        with _patch_bc(self.bc), \
                mock.patch.object(LTM, "reset_all", return_value=0):
            out = A._act_reset_memory()
        self.assertEqual(self._left(), (0, 0, 0, False))
        self.assertIn("2 screen record(s) + 1 cached voice line(s) cleared",
                      out)
        self.assertNotIn("NOT forgotten", out)
        self.assertNotIn("cache was NOT", out)


# ════════════════════════════════════════════════════════════════════════════
#  Shutdown: the queued writes land
# ════════════════════════════════════════════════════════════════════════════
class TeardownFlushTests(_ClientBase):
    TEXT = "Certainly, sir."

    def test_the_ledger_and_the_queued_takes_land_before_termination(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        c.ledger.record(self.TEXT)                   # said twice: its text
        path = os.path.join(self.dir, cseed.LEDGER_FILE)
        self.assertFalse(os.path.exists(path))     # saved only when quiet
        self.assertTrue(c.ledger.dirty)
        with mock.patch.object(cvc, "CLIENT", c):
            t0 = time.time()
            A._flush_persistent_stores()
        self.assertLess(time.time() - t0, 3.0)
        self.assertIn(self.TEXT, open(path, encoding="utf-8").read())
        self.assertEqual(len(self.files()), 1)

    def test_release_native_resources_flushes_last(self):
        from audio import kinect_bridge as kb
        order = []
        bc = mock.Mock()
        with mock.patch.object(kb, "close"), \
                mock.patch("core.voice_clone.unload"), \
                mock.patch.object(A, "_release_audio_streams",
                                  side_effect=lambda b: order.append(
                                      "audio")), \
                mock.patch.object(A, "_flush_persistent_stores",
                                  side_effect=lambda: order.append(
                                      "flush")) as fl:
            A._release_native_resources(bc)
        fl.assert_called_once_with()
        self.assertEqual(order, ["audio", "flush"])


if __name__ == "__main__":
    unittest.main()
