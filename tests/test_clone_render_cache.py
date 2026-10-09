"""core/clone_render_cache.py and the clone client's persistent cache, the
fast-decode alert and "forget that line" (voice architecture C3 / C4,
2026-10-05).

Light tier (stdlib + numpy). Every request goes to a FAKE loopback server
(tests/_clone_voice_fake.py); the disk tier is a temporary folder; nothing
is played, nothing touches a GPU. The voice profiles are synthetic.

Pins:
  * the key: the voice hash, model, T3 precision, sample rate and exact text
    all change it; no usable hash or text -> no key;
  * the disk tier: int16 round trip; a corrupt / wrong-type / wrong-shape /
    absurd-length file is never served and is deleted; a crash's temporary
    file is swept at attach (and by a purge of its voice); writes are
    fsynced before the rename; an over-long take is never written; a write
    queued before its take is forgotten (or its voice purged) never lands;
    the folder is trimmed to VOICE_CLONE_CACHE_MB, least recently used
    first; one writer thread however many writes;
  * the take gate: a fixed band until GATE_MIN_SAMPLES takes, then the
    voice's own p1-p99 clamped INSIDE the fixed band (runaway takes in the
    history can never widen it); a take outside it plays but is never
    persisted; a disk take that does not fit its text is never served;
  * the serve rules: 'on' serves a take written by an earlier run from disk
    with no request; never while the client is not ready (down, cooling
    down: no voice change mid-reply); never from disk unless the server's
    voice is the active consented profile's; a new reference / another
    model is a miss; a take OPENING a reply only while the clone is healthy
    and /health answers ready from the same server (a loading or dead server
    refuses it; a recent check stands for LIVE_MEMO_S), while a look-ahead
    line of a reply already in the clone is served; 'shadow' serves from
    memory only and logs would-hit; 'off' writes nothing;
  * identity: a take is kept on disk only once /health confirms the server
    process, voice and model that made it -- a server restarted with
    another reference never gets its takes filed under the consented key,
    and the client stops using it (or follows a restart in the same voice);
  * purges: a withdrawn consent, a replaced reference or a deleted profiles
    folder deletes that voice's takes (memory and disk), in every mode; a
    folder that cannot be listed purges nothing;
  * C3: one log line when the server is on its slow decoder (/health or a
    line's X-T3-Engine), one when it is back; a long line on the slow loop
    by design raises nothing; decode_note for voice_clone_status;
  * C8 inputs: a rendered line carries its T3 ms per token;
  * "forget that line": the last burst of lines goes from memory and disk
    and is never seeded; an older reply is kept; a line voiced after the
    request was accepted (its own acknowledgement) is not "that line"; a
    write of one still queued never lands;
  * the cache-aware planner (core.sentence_tts): a cached first sentence is
    played whole even when long; a first line the plan splits anyway is
    split at its cached first sentence; a short one-piece reply is never
    split for the cache; nothing cached -> the plan is unchanged.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

import core.config as cfg
from core import clone_render_cache as crc
from core import clone_voice_client as cvc
from core import sentence_tts as st
from core import voice_clone as vc
from tests._clone_voice_fake import FakeCloneServer, ProfileDir, make_wav

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def fit_wav(text: str, amp: float = 0.3) -> bytes:
    """A WAV whose length is the length the take gate expects for `text`."""
    ms = crc.GATE_FIXED_MS + crc.GATE_MS_PER_CHAR * len(text)
    return make_wav(lead_s=0.05, speech_s=ms / 1000.0 - 0.15, tail_s=0.1,
                    amp=amp)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ════════════════════════════════════════════════════════════════════════════
#  Pure helpers
# ════════════════════════════════════════════════════════════════════════════
class KeyTests(unittest.TestCase):
    def test_every_component_changes_the_key(self):
        base = crc.make_key(_SHA_A, "chatterbox-turbo", "fp16", 24000, "Hi.")
        self.assertRegex(base, r"^[0-9a-f]{64}$")
        others = [crc.make_key(_SHA_B, "chatterbox-turbo", "fp16", 24000,
                               "Hi."),
                  crc.make_key(_SHA_A, "other-model", "fp16", 24000, "Hi."),
                  crc.make_key(_SHA_A, "chatterbox-turbo", "fp32", 24000,
                               "Hi."),
                  crc.make_key(_SHA_A, "chatterbox-turbo", "fp16", 22050,
                               "Hi."),
                  crc.make_key(_SHA_A, "chatterbox-turbo", "fp16", 24000,
                               "Hi!"),
                  crc.make_key(_SHA_A, "chatterbox-turbo", "fp16", 24000,
                               "hi.")]
        self.assertEqual(len({base, *others}), 7)
        self.assertEqual(base, crc.make_key(_SHA_A.upper(), "chatterbox-turbo",
                                            "fp16", 24000, "Hi."))

    def test_no_usable_hash_or_text_is_no_key(self):
        for sha, text in (("", "Hi."), ("abc", "Hi."), (None, "Hi."),
                          (_SHA_A, ""), (_SHA_A, None)):
            self.assertIsNone(crc.make_key(sha, "m", "fp16", 24000, text))

    def test_prefix_and_normalise(self):
        self.assertEqual(crc.voice_prefix(_SHA_A), "a" * 16)
        self.assertEqual(crc.voice_prefix("nothex"), "")
        self.assertEqual(crc.normalise_text("  Right   away,\n sir. "),
                         "Right away, sir.")

    def test_mode_and_cap_read_config_at_call_time(self):
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE", "SHADOW"):
            self.assertEqual(crc.mode(), "shadow")
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE", "bogus"):
            self.assertEqual(crc.mode(), "off")
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE_MB", 2):
            self.assertEqual(crc.disk_cap_bytes(), 2 * 1024 * 1024)
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE_MB", -5):
            self.assertEqual(crc.disk_cap_bytes(), 0)
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE_MB", "x"):
            self.assertEqual(crc.disk_cap_bytes(),
                             crc.DEFAULT_DISK_MB * 1024 * 1024)

    def test_shipped_default_is_on(self):
        self.assertEqual(cfg.VOICE_CLONE_CACHE, "on")
        self.assertEqual(cfg.VOICE_CLONE_CACHE_MB, 64)
        self.assertEqual(crc.DEFAULT_MODE, cfg.VOICE_CLONE_CACHE)


class TakeGateTests(unittest.TestCase):
    def test_default_band_until_enough_samples_then_own_range(self):
        g = crc.TakeGate()
        p = "a" * 16
        lo, hi, n = g.band(p)
        self.assertEqual((lo, hi, n), (*crc.GATE_DEFAULT_BAND, 0))
        exp = crc.GATE_FIXED_MS + crc.GATE_MS_PER_CHAR * 40
        ok, r, _ = g.admit(p, exp, 40)
        self.assertTrue(ok)
        self.assertAlmostEqual(r, 1.0, places=3)
        self.assertFalse(g.admit(p, exp * 3.0, 40)[0])       # runaway
        self.assertFalse(g.admit(p, exp * 0.3, 40)[0])       # truncated
        # Every measurable take joins the history (only admitted ones would
        # make the band shrink a little more each time), so judge a voice
        # whose history is all ordinary takes.
        g = crc.TakeGate()
        for i in range(crc.GATE_MIN_SAMPLES + 10):
            g.admit(p, exp * (0.95 + 0.001 * i), 40)
        lo, hi, n = g.band(p)
        self.assertGreaterEqual(n, crc.GATE_MIN_SAMPLES)
        # Now its own, tighter range: 1.3x was fine by the fixed band.
        self.assertFalse(g.admit(p, exp * 1.3, 40)[0])
        self.assertTrue(g.admit(p, exp * 0.97, 40)[0])

    def test_runaway_takes_in_the_history_never_widen_the_band(self):
        # Every measured take joins the history (rejected ones too), so a
        # few runaways would stretch p99; the voice's own band may only
        # tighten the fixed one.
        p = "a" * 16
        exp = crc.GATE_FIXED_MS + crc.GATE_MS_PER_CHAR * 40
        g = crc.TakeGate()
        for i in range(47):
            self.assertTrue(g.admit(p, exp * (0.98 + 0.001 * i), 40)[0])
        for _ in range(3):
            self.assertFalse(g.admit(p, exp * 2.5, 40)[0])
        lo, hi, n = g.band(p)
        self.assertEqual(n, 50)
        self.assertLessEqual(hi, crc.GATE_DEFAULT_BAND[1])
        self.assertGreaterEqual(lo, crc.GATE_DEFAULT_BAND[0])
        self.assertFalse(g.admit(p, exp * 2.5, 40)[0])
        # Steady state: 2 % runaways in a full history.
        g = crc.TakeGate()
        for i in range(392):
            g.admit(p, exp * (0.9 + 0.0005 * i), 40)
        for _ in range(8):
            g.admit(p, exp * 2.2, 40)
        self.assertFalse(g.admit(p, exp * 2.2, 40)[0])
        self.assertTrue(g.admit(p, exp, 40)[0])
        # A voice whose own range is narrower keeps it.
        g = crc.TakeGate()
        for i in range(60):
            g.admit(p, exp * (0.99 + 0.0002 * i), 40)
        lo, hi, _n = g.band(p)
        self.assertGreater(lo, crc.GATE_DEFAULT_BAND[0])
        self.assertLess(hi, 1.1)

    def test_unusable_inputs_are_refused(self):
        g = crc.TakeGate()
        self.assertFalse(g.admit("a" * 16, 0, 40)[0])
        self.assertFalse(g.admit("a" * 16, 1000, 0)[0])
        self.assertFalse(g.admit("", 1000, 10)[0])

    def test_json_round_trip(self):
        g = crc.TakeGate()
        g.admit("a" * 16, 1500, 20)
        h = crc.TakeGate()
        h.load_json(g.to_json())
        self.assertEqual(h.band("a" * 16)[2], 1)
        h.load_json({"hist": {"bad": [1.0], "b" * 16: ["x", 1.2, -3]}})
        self.assertEqual(h.band("b" * 16)[2], 1)


# ════════════════════════════════════════════════════════════════════════════
#  The store
# ════════════════════════════════════════════════════════════════════════════
class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = os.path.join(self._tmp.name, "clone_cache")
        self.s = crc.CloneRenderCache(sync_writes=True)
        self.assertTrue(self.s.attach(self.dir))
        self.key = crc.make_key(_SHA_A, "m", "fp16", 24000, "Hello there.")
        self.pfx = crc.voice_prefix(_SHA_A)

    def pcm(self, n=24000, v=1000):
        return np.full(n, v, dtype=np.int16)

    def test_round_trip_int16(self):
        self.assertFalse(self.s.disk_has(self.key, self.pfx))
        self.assertTrue(self.s.disk_put(self.key, self.pfx, self.pcm()))
        self.assertTrue(self.s.disk_has(self.key, self.pfx))
        a = self.s.disk_get(self.key, self.pfx, 24000)
        self.assertEqual(a.dtype, np.int16)
        np.testing.assert_array_equal(a, self.pcm())
        names = os.listdir(self.dir)
        self.assertEqual(names, [f"{self.pfx}_{self.key}.npy"])
        # A fresh store over the same folder finds it.
        s2 = crc.CloneRenderCache(sync_writes=True)
        s2.attach(self.dir)
        self.assertTrue(s2.disk_has(self.key, self.pfx))

    def test_corrupt_and_foreign_files_are_ignored(self):
        cases = {
            "garbage": b"this is not numpy",
            "float32": None, "2d": None, "empty": None, "huge": None,
            "object": None,
        }
        for i, kind in enumerate(cases):
            key = crc.make_key(_SHA_A, "m", "fp16", 24000, f"line {i}")
            path = os.path.join(self.dir, f"{self.pfx}_{key}.npy")
            if kind == "garbage":
                with open(path, "wb") as f:
                    f.write(cases[kind])
            elif kind == "float32":
                np.save(path, np.zeros(100, dtype=np.float32))
            elif kind == "2d":
                np.save(path, np.zeros((10, 10), dtype=np.int16))
            elif kind == "empty":
                np.save(path, np.zeros(0, dtype=np.int16))
            elif kind == "huge":
                np.save(path, np.zeros(int(crc.MAX_TAKE_S * 100) + 1,
                                       dtype=np.int16))
            elif kind == "object":
                np.save(path, np.array([{"x": 1}], dtype=object),
                        allow_pickle=True)
            s = crc.CloneRenderCache(sync_writes=True)
            s.attach(self.dir)
            self.assertTrue(s.disk_has(key, self.pfx), kind)
            self.assertIsNone(s.disk_get(key, self.pfx, 100), kind)
            self.assertFalse(s.disk_has(key, self.pfx), kind)  # never again
            # ...and gone from the disk: nothing sits outside the cap.
            self.assertFalse(os.path.exists(path), kind)
        # Names that are not ours are never indexed at all.
        with open(os.path.join(self.dir, "notes.txt"), "w") as f:
            f.write("x")
        s = crc.CloneRenderCache()
        self.addCleanup(s.close)
        s.attach(self.dir)
        self.assertNotIn("notes.txt", s._index)

    def test_close_stops_the_writer_after_its_queued_writes(self):
        # rel-182 (2026-10-09): a writer per test client, never stopped,
        # left 50 threads alive in one suite.
        s = crc.CloneRenderCache()
        self.addCleanup(s.close)
        s.attach(self.dir)
        self.assertTrue(s.disk_put(self.key, self.pfx, self.pcm()))
        th = s._writer
        self.assertTrue(th is not None and th.is_alive())
        s.close()
        self.assertFalse(th.is_alive())
        self.assertTrue(s.disk_has(self.key, self.pfx))   # it landed first
        # A later write starts a new writer, and it lands too.
        k2 = crc.make_key(_SHA_A, "m", "fp16", 24000, "And again.")
        self.assertTrue(s.disk_put(k2, self.pfx, self.pcm()))
        self.assertTrue(s.flush(5.0))
        self.assertTrue(s.disk_has(k2, self.pfx))
        s.close()
        s.close()                                        # idempotent

    def test_folder_is_trimmed_least_recently_used_first(self):
        keys = [crc.make_key(_SHA_A, "m", "fp16", 24000, f"l{i}")
                for i in range(5)]
        size = None
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE_MB", 10.0):
            for i, k in enumerate(keys):
                self.s.disk_put(k, self.pfx, self.pcm(n=100_000))
                path = os.path.join(self.dir, f"{self.pfx}_{k}.npy")
                size = os.path.getsize(path)
                os.utime(path, ns=(10**18 + i * 10**9, 10**18 + i * 10**9))
            self.assertEqual(self.s.disk_len(), 5)
            # Read the oldest: it becomes the most recently used.
            self.assertIsNotNone(self.s.disk_get(keys[0], self.pfx, 24000))
        cap_files = 3
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE_MB",
                               (cap_files * size + 10) / 1024 / 1024):
            k5 = crc.make_key(_SHA_A, "m", "fp16", 24000, "l5")
            self.s.disk_put(k5, self.pfx, self.pcm(n=100_000))
        self.assertLessEqual(self.s.disk_bytes(), cap_files * size + 10)
        self.assertTrue(self.s.disk_has(keys[0], self.pfx))   # read lately
        self.assertTrue(self.s.disk_has(k5, self.pfx))
        self.assertFalse(self.s.disk_has(keys[1], self.pfx))
        self.assertEqual(len(os.listdir(self.dir)), self.s.disk_len())

    def test_purge_except_drops_other_voices_in_both_tiers(self):
        kb = crc.make_key(_SHA_B, "m", "fp16", 24000, "Hello there.")
        pb = crc.voice_prefix(_SHA_B)
        self.s.disk_put(self.key, self.pfx, self.pcm())
        self.s.disk_put(kb, pb, self.pcm())
        self.s.mem_put(self.key, np.ones(10, np.float32), 24000, self.pfx)
        self.s.mem_put(kb, np.ones(10, np.float32), 24000, pb)
        # A stray file of voice B that the index never saw.
        stray = os.path.join(self.dir, f"{pb}_{'c' * 64}.npy")
        np.save(stray, self.pcm(10))
        n = self.s.purge_except({self.pfx})
        self.assertEqual(n, 3)                # memory B, disk B, the stray
        self.assertTrue(self.s.disk_has(self.key, self.pfx))
        self.assertTrue(self.s.mem_has(self.key))
        self.assertFalse(self.s.mem_has(kb))
        self.assertEqual(os.listdir(self.dir), [f"{self.pfx}_{self.key}.npy"])
        self.assertEqual(self.s.purge_except(set()), 2)
        self.assertEqual(os.listdir(self.dir), [])

    def test_forget(self):
        self.s.disk_put(self.key, self.pfx, self.pcm())
        self.s.mem_put(self.key, np.ones(10, np.float32), 24000, self.pfx)
        self.assertEqual(self.s.forget([self.key]), 2)
        self.assertFalse(self.s.mem_has(self.key))
        self.assertFalse(self.s.disk_has(self.key, self.pfx))
        self.assertEqual(os.listdir(self.dir), [])

    def test_memory_tier_is_bounded_and_copies(self):
        s = crc.CloneRenderCache(mem_cap_fn=lambda: 4000)
        self.addCleanup(s.close)
        a = np.ones(200, np.float32)          # 800 bytes
        for i in range(10):
            s.mem_put(f"k{i}", a, 24000)
        self.assertLessEqual(s.stats()["mem_bytes"], 4000)
        self.assertFalse(s.mem_put("big", np.ones(2000, np.float32), 24000))
        got, sr = s.mem_get("k9")
        got[:] = 0
        self.assertTrue(np.all(s.mem_get("k9")[0] == 1.0))

    def test_one_writer_thread_however_many_writes(self):
        s = crc.CloneRenderCache()
        self.addCleanup(s.close)
        s.attach(self.dir)

        def writers():
            return sum(1 for t in threading.enumerate()
                       if t.name == "clone-cache-writer")
        before = writers()
        for i in range(20):
            k = crc.make_key(_SHA_A, "m", "fp16", 24000, f"w{i}")
            self.assertTrue(s.disk_put(k, self.pfx, self.pcm(n=500)))
        self.assertTrue(s.flush(5.0))
        self.assertEqual(writers() - before, 1)
        self.assertTrue(s._writer.is_alive())
        self.assertEqual(s.disk_len(), 20)
        self.assertEqual(len(os.listdir(self.dir)), 20)

    def test_gate_history_persists_with_the_folder(self):
        self.s.gate.admit(self.pfx, 1500.0, 20)
        self.assertTrue(self.s.save_gate())
        s2 = crc.CloneRenderCache()
        self.addCleanup(s2.close)
        s2.attach(self.dir)
        self.assertEqual(s2.gate.band(self.pfx)[2], 1)

    def test_crash_leftovers_are_swept_at_attach(self):
        tmp = os.path.join(self.dir, f"{self.pfx}_{self.key}.npy.999.123.tmp")
        with open(tmp, "wb") as f:
            f.write(b"half a take")
        other = os.path.join(self.dir, "notes.txt")
        with open(other, "w") as f:
            f.write("x")
        s = crc.CloneRenderCache(sync_writes=True)
        self.assertTrue(s.attach(self.dir))
        self.assertFalse(os.path.exists(tmp))
        self.assertTrue(os.path.exists(other))       # not ours: left alone
        self.assertEqual(s.stats()["leftovers"], 1)

    def test_a_purge_takes_a_purged_voices_leftovers_too(self):
        pb = crc.voice_prefix(_SHA_B)
        kb = crc.make_key(_SHA_B, "m", "fp16", 24000, "Hello there.")
        tmp_b = os.path.join(self.dir, f"{pb}_{kb}.npy.1.2.tmp")
        tmp_a = os.path.join(self.dir, f"{self.pfx}_{self.key}.npy.1.2.tmp")
        for t in (tmp_a, tmp_b):
            with open(t, "wb") as f:
                f.write(b"x")
        self.s.purge_except({self.pfx})
        self.assertFalse(os.path.exists(tmp_b))
        self.assertTrue(os.path.exists(tmp_a))

    def test_an_over_long_take_is_never_written(self):
        big = np.zeros(int(crc.MAX_TAKE_S * 24000) + 1, dtype=np.int16)
        self.assertFalse(self.s.disk_put(self.key, self.pfx, big, sr=24000))
        self.assertEqual(os.listdir(self.dir), [])
        self.assertTrue(self.s.disk_put(self.key, self.pfx, self.pcm(),
                                        sr=24000))

    def test_a_write_reaches_the_disk_before_its_rename(self):
        calls = []
        real_replace = os.replace

        def replace(a, b_):
            calls.append("replace")
            return real_replace(a, b_)
        with mock.patch.object(crc.os, "fsync",
                               side_effect=lambda fd: calls.append("fsync")), \
                mock.patch.object(crc.os, "replace", side_effect=replace):
            self.assertTrue(self.s.disk_put(self.key, self.pfx, self.pcm()))
        self.assertEqual(calls, ["fsync", "replace"])

    def test_keep_runs_on_the_writer_and_can_refuse(self):
        self.assertTrue(self.s.disk_put(self.key, self.pfx, self.pcm(),
                                        keep=lambda: False))
        self.assertEqual(os.listdir(self.dir), [])
        self.assertEqual(self.s.stats()["not_kept"], 1)

        def boom():
            raise RuntimeError("x")
        self.s.disk_put(self.key, self.pfx, self.pcm(), keep=boom)
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_write_queued_before_a_forget_or_a_purge_never_lands(self):
        s = crc.CloneRenderCache()
        self.addCleanup(s.close)
        s.attach(self.dir)
        held = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def hold():
            held.set()
            return release.wait(5.0)
        k1 = crc.make_key(_SHA_A, "m", "fp16", 24000, "one")
        k2 = crc.make_key(_SHA_A, "m", "fp16", 24000, "two")
        kb = crc.make_key(_SHA_B, "m", "fp16", 24000, "one")
        pb = crc.voice_prefix(_SHA_B)
        asked = []
        self.assertTrue(s.disk_put(k1, self.pfx, self.pcm(), keep=hold))
        self.assertTrue(held.wait(5.0))              # the writer is busy
        self.assertTrue(s.disk_put(k2, self.pfx, self.pcm(),
                                   keep=lambda: asked.append("k2") or True))
        self.assertTrue(s.disk_put(kb, pb, self.pcm(),
                                   keep=lambda: asked.append("kb") or True))
        s.forget([k2])
        s.purge_except({self.pfx})
        release.set()
        self.assertTrue(s.flush(5.0))
        self.assertEqual(os.listdir(self.dir), [f"{self.pfx}_{k1}.npy"])
        self.assertEqual(s.stats()["stale_writes"], 2)
        # Dropped before their keep() (a /health read for the client) and
        # before any file was written.
        self.assertEqual(asked, [])
        self.assertEqual(s.stats()["writes"], 1)
        # A take said afresh AFTER the forget is written as usual.
        self.assertTrue(s.disk_put(k2, self.pfx, self.pcm()))
        self.assertTrue(s.flush(5.0))
        self.assertTrue(s.disk_has(k2, self.pfx))

    def test_a_forget_during_the_write_itself_wins(self):
        # The forget lands between the writer's check and its rename.
        s = crc.CloneRenderCache(sync_writes=True)
        s.attach(self.dir)
        real_replace = os.replace

        def replace(a, b_):
            s.forget([self.key])
            return real_replace(a, b_)
        with mock.patch.object(crc.os, "replace", side_effect=replace):
            s.disk_put(self.key, self.pfx, self.pcm())
        self.assertEqual(os.listdir(self.dir), [])
        self.assertFalse(s.disk_has(self.key, self.pfx))

    def test_nothing_raises_without_a_disk_tier(self):
        s = crc.CloneRenderCache()
        self.addCleanup(s.close)
        self.assertFalse(s.disk_put(self.key, self.pfx, self.pcm()))
        self.assertIsNone(s.disk_get(self.key, self.pfx, 24000))
        self.assertFalse(s.disk_has(self.key, self.pfx))
        self.assertEqual(s.purge_except(set()), 0)
        self.assertFalse(s.save_gate())


# ════════════════════════════════════════════════════════════════════════════
#  The client with a disk tier
# ════════════════════════════════════════════════════════════════════════════
class _ClientBase(unittest.TestCase):
    MODE = "on"

    def setUp(self):
        self.prof = ProfileDir("butler")
        self.addCleanup(self.prof.cleanup)
        for target, name, value in ((vc, "PROFILES_DIR", self.prof.root),
                                    (cvc, "PROFILE_TTL_S", 0.0),
                                    (cfg, "VOICE_CLONE_CACHE", self.MODE)):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = os.path.join(self._tmp.name, "clone_cache")
        self.logs: list = []

    def server(self, **kw) -> FakeCloneServer:
        kw.setdefault("ref_sha", self.prof.sha)
        s = FakeCloneServer(**kw).start()
        self.addCleanup(s.stop)
        return s

    def client(self, srv, attach=True, start=True, **kw):
        kw.setdefault("log", self.logs.append)
        kw.setdefault("boot_wait_s", 3.0)
        kw.setdefault("sleep", lambda s: time.sleep(min(s, 0.02)))
        c = cvc.CloneVoiceClient(**kw)
        if attach:
            self.assertTrue(c.attach_cache(self.dir))
        if start:
            self.assertEqual(c.start(url=srv.url, cmd="", profile="butler"),
                             "ready", self.logs)
        self.addCleanup(lambda: c.store.flush(5.0))
        self.addCleanup(c.store.close)      # its writer thread (runs first)
        return c

    def rendered(self, srv, text):
        """An earlier run: a client renders `text`, its write lands."""
        c = self.client(srv)
        self.assertTrue(c.render(text, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        return c

    def files(self):
        try:
            return sorted(n for n in os.listdir(self.dir) if n.endswith(".npy"))
        except OSError:
            return []


class OnModeTests(_ClientBase):
    TEXT = "Certainly, sir."

    def test_a_take_from_an_earlier_run_is_served_from_disk(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c1 = self.client(srv)
        a = c1.render(self.TEXT, 2.5)
        self.assertTrue(a.ok and not a.cached)
        self.assertEqual(a.cache, "")
        self.assertTrue(c1.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)
        self.assertTrue(self.files()[0].startswith(self.prof.sha[:16] + "_"))
        # "Restart": a new client over the same folder.
        c2 = self.client(srv)
        self.assertTrue(c2.is_cached(self.TEXT))
        b = c2.render(self.TEXT, 2.5)
        self.assertTrue(b.ok and b.cached)
        self.assertEqual((b.cache, b.ms), ("disk", 0))
        np.testing.assert_allclose(a.audio, b.audio, atol=1e-6)
        self.assertEqual(srv.tts_texts(), [self.TEXT])     # one request
        # ...and the next one comes from memory.
        self.assertEqual(c2.render(self.TEXT, 2.5).cache, "mem")
        self.assertEqual(c2.cache_stats()["disk_hits"], 1)

    def test_no_cached_take_while_the_clone_is_not_ready(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        self.rendered(srv, self.TEXT)
        c2 = self.client(srv)
        c2._status = "cooldown"
        c2._cool_until = c2._clock() + 3600.0
        out = c2.render(self.TEXT, 2.5)
        self.assertEqual((out.ok, out.reason), (False, "not-ready"))
        self.assertFalse(c2.is_cached(self.TEXT))
        c3 = self.client(srv, start=False)          # never came up
        self.assertEqual(c3.render(self.TEXT, 2.5).reason, "not-ready")
        c2._down("test")
        self.assertEqual(c2.render(self.TEXT, 2.5).reason, "not-ready")
        self.assertEqual(srv.tts_texts(), [self.TEXT])

    def test_disk_serves_only_the_consented_profiles_voice(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        self.rendered(srv, self.TEXT)
        c2 = self.client(srv)
        with mock.patch.object(c2, "profile_sha", return_value=_SHA_B):
            self.assertFalse(c2.is_cached(self.TEXT))
            out = c2.render(self.TEXT, 2.5)
        self.assertFalse(out.cached)
        self.assertEqual(len(srv.tts_texts()), 2)

    def test_another_model_or_precision_is_a_miss(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        self.rendered(srv, self.TEXT)
        srv2 = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)},
                           t3_dtype="fp32")
        c2 = self.client(srv2)
        self.assertFalse(c2.is_cached(self.TEXT))
        self.assertFalse(c2.render(self.TEXT, 2.5).cached)
        self.assertEqual(srv2.tts_texts(), [self.TEXT])

    def test_a_replaced_reference_is_a_miss_and_its_takes_are_purged(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c1 = self.client(srv)
        c1.render(self.TEXT, 2.5)
        self.assertTrue(c1.store.flush(5.0))
        old = self.files()
        self.assertEqual(len(old), 1)
        # The owner picks a new reference.wav; the server restarts with it.
        with open(self.prof.ref, "wb") as f:
            f.write(b"RIFF a different, longer reference clip")
        import hashlib
        with open(self.prof.ref, "rb") as f:
            new_sha = hashlib.sha256(f.read()).hexdigest()
        srv2 = self.server(ref_sha=new_sha,
                           wav_for={self.TEXT: fit_wav(self.TEXT)})
        c2 = self.client(srv2)           # attach + ready both purge
        self.assertEqual(self.files(), [])
        self.assertTrue(any("no longer a consented profile" in m
                            for m in self.logs), self.logs)
        out = c2.render(self.TEXT, 2.5)
        self.assertFalse(out.cached)
        self.assertTrue(c2.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)
        self.assertTrue(self.files()[0].startswith(new_sha[:16]))

    def test_attaching_the_folder_alone_purges_a_replaced_voice(self):
        # At boot, before (or without) a server: the takes of a voice that
        # is no longer a consented profile's go when the folder is attached.
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        self.rendered(srv, self.TEXT)
        self.assertEqual(len(self.files()), 1)
        with open(self.prof.ref, "wb") as f:
            f.write(b"RIFF a replaced reference clip, longer")
        self.client(srv, start=False)
        self.assertEqual(self.files(), [])

    def test_a_withdrawn_consent_purges_in_every_mode(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        c.render(self.TEXT, 2.5)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)
        meta = os.path.join(self.prof.root, "butler", "meta.json")
        with open(meta, "w", encoding="utf-8") as f:
            f.write('{"name": "butler", "source": "character", '
                    '"consent": false}')
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE", "off"):
            self.assertEqual(c.purge_unconsented(), 2)     # memory + disk
        self.assertEqual(self.files(), [])
        self.assertEqual(c.cache_len(), 0)

    def test_a_deleted_profiles_folder_purges_every_take(self):
        # Deleting data/voice_profiles is the most complete way to withdraw
        # consent: every take goes, now and at the next boot.
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        c.render(self.TEXT, 2.5)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)
        gone = os.path.join(self._tmp.name, "missing")
        with mock.patch.object(vc, "PROFILES_DIR", gone):
            self.assertEqual(c.purge_unconsented(), 2)     # memory + disk
        self.assertEqual(self.files(), [])
        self.assertEqual(c.cache_len(), 0)
        # At the next boot, attaching the folder alone purges.
        self.rendered(srv, self.TEXT)
        self.assertEqual(len(self.files()), 1)
        with mock.patch.object(vc, "PROFILES_DIR", gone):
            self.client(srv, start=False)
        self.assertEqual(self.files(), [])

    def test_a_profiles_folder_that_cannot_be_listed_purges_nothing(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        c.render(self.TEXT, 2.5)
        self.assertTrue(c.store.flush(5.0))
        real_listdir = os.listdir

        def listdir(path="."):
            if os.path.normcase(str(path)) == os.path.normcase(self.prof.root):
                raise PermissionError("denied")
            return real_listdir(path)
        with mock.patch.object(os, "listdir", side_effect=listdir):
            self.assertIsNone(c.consented_prefixes())
            self.assertEqual(c.purge_unconsented(), 0)
        self.assertEqual(len(self.files()), 1)

    def test_an_outlier_take_plays_but_is_not_kept_on_disk(self):
        runaway = "Hi."
        srv = self.server(wav_for={
            runaway: make_wav(lead_s=0.0, speech_s=12.0, tail_s=0.0,
                              amp=0.3),
            self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        out = c.render(runaway, 2.5)
        self.assertTrue(out.ok)                       # it still plays
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)        # only the normal take
        self.assertEqual(c.cache_stats()["rejected"], 1)
        self.assertTrue(any("take not kept on disk" in m for m in self.logs))
        # Memory still has it for this run (as before the disk tier).
        self.assertEqual(c.render(runaway, 2.5).cache, "mem")

    def test_a_take_at_another_sample_rate_is_not_persisted(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)},
                          sample_rate=22050)        # /health disagrees
        c = self.client(srv)
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(self.files(), [])

    def test_whitespace_runs_are_one_key_and_one_request(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        c.render("Certainly,   sir. ", 2.5)
        self.assertTrue(c.render(" Certainly, sir.", 2.5).cached)
        self.assertEqual(srv.tts_texts(), [self.TEXT])

    def test_t3_ms_per_token_comes_from_the_headers(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)},
                          t3_ms_per_token=4.5)
        c = self.client(srv)
        out = c.render(self.TEXT, 2.5)
        self.assertEqual(out.engine, "graph")
        self.assertGreater(out.tokens, 10)
        self.assertAlmostEqual(out.t3_ms_per_token, 4.5, places=1)
        self.assertGreater(out.audio_ms, 1000)
        self.assertIsNone(c.render(self.TEXT, 2.5).t3_ms_per_token)

    def test_ledger_counts_listener_lines_not_background_ones(self):
        clock = _Clock()
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv, clock=clock)
        c.render(self.TEXT, 2.5)
        clock.t += 2.0
        c.render(self.TEXT, 2.5)       # moments later: the same saying
        self.assertEqual(c.ledger.count(self.TEXT), 1)
        clock.t += cvc.LEDGER_DEDUPE_S + 1.0
        c.render(self.TEXT, 2.5)                    # a cache hit counts too
        c.render("One moment, sir.", 2.5, count=False)   # a filler warm
        self.assertEqual(c.ledger.count(self.TEXT), 2)
        self.assertEqual(c.ledger.count("One moment, sir."), 0)
        self.assertEqual(c.ledger.seeds(), [self.TEXT])

    def test_no_cached_take_opens_a_reply_once_the_server_is_gone(self):
        # Died since the last request (still 'ready'): its voice must not
        # OPEN a reply from the cache while the rest falls to Kokoro.
        clock = _Clock()
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        earlier = self.rendered(srv, self.TEXT)
        c = self.client(srv, clock=clock)
        self.assertTrue(c.render("Right away, sir.", 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        srv.stop()
        clock.t += cvc.LIVE_MEMO_S + 1.0
        for text in ("Right away, sir.", self.TEXT):    # memory, disk
            c._fails = 0      # each its own reply opening on a healthy clone
            out = c.render(text, 2.5)
            self.assertFalse(out.ok, text)
            self.assertIn("not answering ready", out.reason)
            self.assertEqual(out.cache, "refused")
            self.assertTrue(out.counted)
        self.assertEqual(c.failures(), 1)
        # With that miss pending, a cached opener is not even tried: the
        # line renders, and misses like any other.
        out = c.render("Right away, sir.", 2.5)
        self.assertFalse(out.ok)
        self.assertNotEqual(out.cache, "refused")
        self.assertEqual(c.failures(), 2)
        # A background render (the filler warm) is not a listener's line.
        self.assertTrue(c.render("Right away, sir.", 2.5, count=False).ok)
        # A look-ahead line belongs to a reply already in the clone: its
        # cached take keeps more of that reply in the one voice.
        out = c.render("Right away, sir.", 2.5, needed_by=clock.t + 5.0)
        self.assertTrue(out.ok and out.cached)
        self.assertIsNotNone(earlier)

    def test_the_server_gpu_index(self):
        srv = self.server()
        c = self.client(srv)
        self.assertEqual(c.server_gpu_index(), 0)
        c._server_info["device"] = "cuda:1 (physical, PCI order)"
        self.assertEqual(c.server_gpu_index(), 1)
        c._server_info["device"] = "cpu"
        self.assertIsNone(c.server_gpu_index())


class ServerIdentityTests(_ClientBase):
    """/tts does not say which process or voice made a take: it is kept on
    disk only once /health confirms the server the key names."""
    TEXT = "The lights are off, sir."
    NEXT = "Right away, sir."

    def test_a_server_restarted_in_another_voice_never_files_its_takes(self):
        srv = self.server(wav_for={t: fit_wav(t) for t in (self.TEXT,
                                                            self.NEXT)})
        c = self.client(srv)
        # It restarts with another (unconsented) reference while JARVIS is
        # up; the client cannot tell from the /tts reply.
        srv.ref_sha = "c" * 64
        srv.pid = 5151
        out = c.render(self.TEXT, 2.5)
        self.assertTrue(out.ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(self.files(), [])           # never on disk ...
        self.assertEqual(c.cache_len(), 0)           # ... nor in memory
        self.assertEqual(c.status()[0], "down")      # and no longer used
        self.assertEqual(c.render(self.NEXT, 2.5).reason, "not-ready")
        # The next run, the server back on the consented voice: the line is
        # rendered afresh, never served from a wrong-voice file.
        srv.ref_sha = self.prof.sha
        srv.pid = 4242
        c2 = self.client(srv)
        n = len(srv.tts_texts())
        again = c2.render(self.TEXT, 2.5)
        self.assertTrue(again.ok and not again.cached)
        self.assertEqual(len(srv.tts_texts()), n + 1)

    def test_a_restart_in_the_same_voice_is_followed(self):
        srv = self.server(wav_for={t: fit_wav(t) for t in (self.TEXT,
                                                            self.NEXT)})
        c = self.client(srv)
        srv.pid = 5151                   # restarted: same reference, model
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(self.files(), [])   # the take across the restart
        self.assertEqual(c.status()[0], "ready")
        self.assertEqual(c.server_pid(), 5151)
        self.assertTrue(any("restarted" in m for m in self.logs), self.logs)
        self.assertTrue(c.render(self.NEXT, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)          # the next one is kept

    def test_a_take_the_server_cannot_vouch_for_is_not_kept(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv, clock=_Clock())
        srv.health_code = 503                # /health stops answering ready
        srv.ok = False
        c._clock.t += 1.0
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(self.files(), [])
        self.assertEqual(c.status()[0], "ready")    # nothing else changes
        self.assertEqual(c.cache_stats()["not_kept"], 1)

    def test_a_take_of_a_voice_no_longer_consented_is_not_kept(self):
        # Consent withdrawn (or the profile switched) while a take was on
        # its way to the disk: the writer re-checks the active profile.
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c = self.client(srv)
        with mock.patch.object(c, "profile_sha", return_value=_SHA_B):
            self.assertTrue(c.render(self.TEXT, 2.5).ok)
            self.assertTrue(c.store.flush(5.0))
        self.assertEqual(self.files(), [])
        self.assertEqual(c.cache_stats()["not_kept"], 1)

    def test_one_health_read_vouches_for_earlier_takes(self):
        srv = self.server(wav_for={t: fit_wav(t) for t in (self.TEXT,
                                                            self.NEXT)})
        clock = _Clock()
        c = self.client(srv, clock=clock)
        gets = srv.count("GET", "/health")
        clock.t += 1.0
        self.assertTrue(c.render(self.TEXT, 2.5).ok)
        self.assertTrue(c.render(self.NEXT, 2.5).ok)    # same instant
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(srv.count("GET", "/health"), gets + 1)
        self.assertEqual(len(self.files()), 2)
        # A take that arrives after that read needs a read of its own.
        clock.t += 1.0
        self.assertTrue(c.render("A third line, sir.", 2.5).ok)
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(srv.count("GET", "/health"), gets + 2)


class CachedOpenerTests(_ClientBase):
    """A cached take that OPENS a reply: only while the clone is healthy and
    the server answers ready in the same process, voice and model."""
    TEXT = "Certainly, sir."

    def cached_client(self, **srv_kw):
        srv_kw.setdefault("wav_for", {self.TEXT: fit_wav(self.TEXT)})
        srv = self.server(**srv_kw)
        clock = _Clock()
        c = self.client(srv, clock=clock)
        self.assertTrue(c.render(self.TEXT, 2.5).ok)    # now in memory
        self.assertTrue(c.store.flush(5.0))
        return srv, c, clock

    def test_a_loading_server_never_opens_a_reply_from_the_cache(self):
        srv, c, clock = self.cached_client()
        # It died; a new one bound the port and loads (~13 s of 503s).
        srv.ok = False
        srv.health_code = 503
        srv.tts_status = 503
        srv.pid = 5151
        clock.t += cvc.LIVE_MEMO_S + 1.0
        out = c.render(self.TEXT, 2.5)
        self.assertFalse(out.ok)
        self.assertTrue(out.counted)
        self.assertEqual(out.cache, "refused")
        self.assertIn("not answering ready", out.reason)
        self.assertEqual(c.cache_stats()["refused"], 1)

    def test_a_recent_check_stands_for_the_server_then_it_is_probed(self):
        srv, c, clock = self.cached_client()
        gets = srv.count("GET", "/health")
        self.assertEqual(c.render(self.TEXT, 2.5).cache, "mem")
        self.assertEqual(srv.count("GET", "/health"), gets)      # memo
        clock.t += cvc.LIVE_MEMO_S + 1.0
        self.assertEqual(c.render(self.TEXT, 2.5).cache, "mem")
        self.assertEqual(srv.count("GET", "/health"), gets + 1)  # probed
        self.assertEqual(c.render(self.TEXT, 2.5).cache, "mem")
        self.assertEqual(srv.count("GET", "/health"), gets + 1)  # fresh

    def test_a_pending_voice_recheck_renders_instead(self):
        srv, c, clock = self.cached_client()
        c._recheck = True                 # a cool-down just ended
        self.assertFalse(c.is_cached(self.TEXT))
        n = len(srv.tts_texts())
        out = c.render(self.TEXT, 2.5)
        self.assertTrue(out.ok and not out.cached)
        self.assertEqual(len(srv.tts_texts()), n + 1)
        self.assertFalse(c._recheck)

    def test_a_struggling_clone_renders_the_opener_instead(self):
        srv, c, clock = self.cached_client()
        for state in ({"_fails": 1}, {"_probation": True},
                      {"_decode": "eager"}):
            for k, v in state.items():
                setattr(c, k, v)
            self.assertFalse(c.is_cached(self.TEXT), state)
            n = len(srv.tts_texts())
            out = c.render(self.TEXT, 2.5)
            self.assertTrue(out.ok and not out.cached, state)
            self.assertEqual(len(srv.tts_texts()), n + 1, state)
            # A look-ahead line of a reply already in the clone still is.
            la = c.render(self.TEXT, 2.5, needed_by=clock.t + 5.0)
            self.assertTrue(la.ok and la.cached, state)
            c._fails, c._probation, c._decode = 0, False, "cuda-graph"
        self.assertTrue(c.is_cached(self.TEXT))

    def test_a_probe_that_finds_the_slow_decoder_renders_the_opener(self):
        srv, c, clock = self.cached_client()
        srv.t3_decode = "eager"              # it fell back since the last line
        srv.engine = "eager"
        clock.t += cvc.LIVE_MEMO_S + 1.0
        n = len(srv.tts_texts())
        out = c.render(self.TEXT, 2.5)
        self.assertTrue(out.ok and not out.cached)
        self.assertEqual(len(srv.tts_texts()), n + 1)
        self.assertEqual(c.decode_state(), "eager")

    def test_a_server_now_in_the_profiles_new_voice_renders_the_opener(self):
        srv, c, clock = self.cached_client()
        with open(self.prof.ref, "wb") as f:
            f.write(b"RIFF the owner's new reference")
        with open(self.prof.ref, "rb") as f:
            new_sha = hashlib.sha256(f.read()).hexdigest()
        srv.ref_sha = new_sha
        srv.pid = 5151
        clock.t += cvc.LIVE_MEMO_S + 1.0
        out = c.render(self.TEXT, 2.5)
        self.assertTrue(out.ok and not out.cached)   # the old voice's take
        self.assertEqual(c.status()[0], "ready")     # is not played
        self.assertEqual(c.voice_prefix(), new_sha[:16])

    def test_a_disk_take_that_does_not_fit_its_text_is_never_served(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c1 = self.client(srv, start=True)
        key = c1._key(self.prof.sha, c1._server_info,
                      crc.normalise_text(cvc._normalize_text(self.TEXT)))
        # A 12 s take for a 15-character line, filed by an older rule.
        c1.store.disk_put(key, self.prof.sha[:16],
                          np.full(12 * 24000, 900, dtype=np.int16))
        self.assertTrue(c1.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)
        c2 = self.client(srv)
        n = len(srv.tts_texts())
        out = c2.render(self.TEXT, 2.5)
        self.assertFalse(out.cached)
        self.assertEqual(len(srv.tts_texts()), n + 1)
        self.assertEqual(c2.cache_stats()["rejected_on_read"], 1)


class ShadowAndOffTests(_ClientBase):
    MODE = "shadow"
    TEXT = "Very good, sir."

    def test_shadow_serves_memory_only_and_logs_would_hit(self):
        srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
        c1 = self.client(srv)
        first = c1.render(self.TEXT, 2.5)
        self.assertEqual(first.shadow, "would-miss")
        self.assertTrue(c1.store.flush(5.0))
        self.assertEqual(len(self.files()), 1)           # written
        self.assertEqual(c1.render(self.TEXT, 2.5).cache, "mem")
        c2 = self.client(srv)
        self.assertFalse(c2.is_cached(self.TEXT))       # planner: unchanged
        out = c2.render(self.TEXT, 2.5)
        self.assertFalse(out.cached)                     # NOT served
        self.assertEqual(out.shadow, "would-hit")
        self.assertEqual(len(srv.tts_texts()), 2)
        self.assertTrue(any("[clone-cache] shadow would-hit (1/1" in m
                            for m in self.logs), self.logs)

    def test_off_is_memory_only(self):
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE", "off"):
            srv = self.server(wav_for={self.TEXT: fit_wav(self.TEXT)})
            c = self.client(srv)
            a = c.render(self.TEXT, 2.5)
            self.assertIsNone(a.shadow)
            self.assertEqual(c.render(self.TEXT, 2.5).cache, "mem")
            self.assertFalse(c.is_cached(self.TEXT))
            self.assertTrue(c.store.flush(5.0))
            self.assertEqual(self.files(), [])
            self.assertEqual(srv.tts_texts(), [self.TEXT])

    def test_without_a_disk_tier_shadow_and_on_are_the_old_cache(self):
        for m in ("shadow", "on"):
            with mock.patch.object(cfg, "VOICE_CLONE_CACHE", m):
                srv = self.server()
                c = self.client(srv, attach=False)
                a = c.render("Right away, sir.", 2.5)
                self.assertTrue(a.ok and not a.cached)
                self.assertIsNone(a.shadow)
                self.assertTrue(c.render("Right away, sir.", 2.5).cached)
                self.assertEqual(srv.tts_texts(), ["Right away, sir."])


# ════════════════════════════════════════════════════════════════════════════
#  C3: the fast-decode alert
# ════════════════════════════════════════════════════════════════════════════
class FastDecodeAlertTests(_ClientBase):
    def alerts(self):
        return [m for m in self.logs if "fast decode" in m]

    def test_health_on_the_slow_decoder_is_one_line_and_a_status_note(self):
        srv = self.server(t3_decode="eager")
        c = self.client(srv)
        self.assertEqual(len(self.alerts()), 1, self.logs)
        self.assertIn("OFF", self.alerts()[0])
        self.assertIn("nothing is restarted", self.alerts()[0])
        self.assertEqual(c.decode_state(), "eager")
        self.assertIn("slower", c.decode_note())
        c.refresh_health()
        self.assertEqual(len(self.alerts()), 1)          # no repeat
        srv.t3_decode = "cuda-graph"
        self.assertTrue(c.refresh_health())
        self.assertEqual(len(self.alerts()), 2)
        self.assertIn("back on", self.alerts()[1])
        self.assertEqual(c.decode_note(), "")

    def test_a_line_from_the_slow_decoder_raises_it_once(self):
        srv = self.server(engine="eager")
        c = self.client(srv)
        self.assertEqual(self.alerts(), [])
        c.render("First line, sir.", 2.5)
        c.render("Second line, sir.", 2.5)
        self.assertEqual(len(self.alerts()), 1)
        srv.engine = "graph"
        c.render("Third line, sir.", 2.5)
        self.assertEqual(len(self.alerts()), 2)

    def test_a_long_line_on_the_slow_loop_by_design_raises_nothing(self):
        srv = self.server(engine="eager")
        c = self.client(srv)
        c.render("word " * 120, 2.5)                 # > GRAPH_MAX_CHARS
        self.assertEqual(self.alerts(), [])
        self.assertEqual(c.decode_state(), "cuda-graph")

    def test_a_server_that_changed_voice(self):
        srv = self.server()
        c = self.client(srv)
        srv.ref_sha = _SHA_B                        # not any profile's
        self.assertFalse(c.refresh_health())
        self.assertEqual(c.status()[0], "down")
        # ...but one that now speaks the profile's NEW reference is followed.
        srv2 = self.server()
        c2 = self.client(srv2)
        with open(self.prof.ref, "wb") as f:
            f.write(b"RIFF the owner's new reference")
        import hashlib
        with open(self.prof.ref, "rb") as f:
            new_sha = hashlib.sha256(f.read()).hexdigest()
        srv2.ref_sha = new_sha
        self.assertTrue(c2.refresh_health())
        self.assertEqual(c2.status()[0], "ready")
        self.assertEqual(c2.voice_prefix(), new_sha[:16])


# ════════════════════════════════════════════════════════════════════════════
#  "Forget that line"
# ════════════════════════════════════════════════════════════════════════════
class ForgetTests(_ClientBase):
    A = "Certainly, sir."
    B = "The lamp is off now."
    OLD = "Good evening, sir."

    def test_the_last_reply_goes_and_an_older_one_stays(self):
        clock = _Clock()
        srv = self.server(wav_for={t: fit_wav(t) for t in
                                   (self.A, self.B, self.OLD)})
        c = self.client(srv, clock=clock)
        c.render(self.OLD, 2.5)
        clock.t += 120.0
        c.render(self.A, 2.5)
        clock.t += 2.0
        c.render(self.B, 2.5)
        c.render("A filler line.", 2.5, count=False)     # not "that line"
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(len(self.files()), 4)
        clock.t += 5.0
        self.assertEqual(c.forget_last_reply(), [self.A, self.B])
        self.assertEqual(len(self.files()), 2)
        self.assertTrue(c.ledger.is_forgotten(self.A))
        self.assertFalse(c.ledger.is_forgotten(self.OLD))
        n = len(srv.tts_texts())
        self.assertFalse(c.render(self.A, 2.5).cached)  # afresh
        self.assertTrue(c.render(self.OLD, 2.5).cached)
        self.assertEqual(len(srv.tts_texts()), n + 1)

    def test_a_line_voiced_after_the_request_is_not_that_line(self):
        # "Forget that line" -> the reply streams "Certainly, sir." before
        # the action runs; that acknowledgement is not the line he meant.
        clock = _Clock()
        bad = "The parcel arrived at the side door."
        srv = self.server(wav_for={t: fit_wav(t) for t in (bad, self.A)})
        c = self.client(srv, clock=clock)
        c.render(bad, 2.5)
        clock.t += 20.0
        asked = clock.t                     # his request accepted ("You:")
        clock.t += 6.0
        c.render(self.A, 2.5)               # the streamed acknowledgement
        clock.t += 1.0
        self.assertTrue(c.store.flush(5.0))
        self.assertEqual(c.forget_last_reply(before=asked), [bad])
        self.assertTrue(c.ledger.is_forgotten(bad))
        self.assertFalse(c.ledger.is_forgotten(self.A))
        self.assertTrue(c.render(self.A, 2.5).cached)
        self.assertFalse(c.render(bad, 2.5).cached)

    def test_a_take_still_queued_when_forgotten_never_lands(self):
        clock = _Clock()
        srv = self.server(wav_for={self.A: fit_wav(self.A)})
        c = self.client(srv, clock=clock)
        held = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        real_verify = c._verify_take

        def slow_verify(*a):
            held.set()
            release.wait(5.0)
            return real_verify(*a)
        with mock.patch.object(c, "_verify_take", side_effect=slow_verify):
            self.assertTrue(c.render(self.A, 2.5).ok)
            self.assertTrue(held.wait(5.0))      # its write is in flight
            clock.t += 1.0
            self.assertEqual(c.forget_last_reply(), [self.A])
            release.set()
            self.assertTrue(c.store.flush(5.0))
        self.assertEqual(self.files(), [])

    def test_nothing_recent_forgets_nothing(self):
        clock = _Clock()
        srv = self.server(wav_for={self.A: fit_wav(self.A)})
        c = self.client(srv, clock=clock)
        self.assertEqual(c.forget_last_reply(), [])
        c.render(self.A, 2.5)
        clock.t += cvc.FORGET_WINDOW_S + 1.0
        self.assertEqual(c.forget_last_reply(), [])
        self.assertTrue(c.render(self.A, 2.5).cached)


# ════════════════════════════════════════════════════════════════════════════
#  The cache-aware planner (core.sentence_tts)
# ════════════════════════════════════════════════════════════════════════════
class CacheAwarePlanTests(unittest.TestCase):
    LONG_S1 = ("The forecast for tomorrow looks mild, with a gentle breeze "
               "from the west and grey skies all afternoon.")
    SHORT = "Very good, sir. The lamp is off now."

    def test_nothing_cached_is_exactly_the_old_plan(self):
        for text in (self.SHORT, self.LONG_S1,
                     "Certainly, sir. " + self.LONG_S1,
                     self.LONG_S1 + " " + self.LONG_S1):
            self.assertEqual(st.plan_clone_chunks(text, is_cached=lambda t:
                                                  False),
                             st.plan_clone_chunks(text), text)

    def test_a_cached_long_first_sentence_is_played_whole(self):
        self.assertGreater(len(self.LONG_S1), st.CLAUSE_SPLIT_MIN_CHARS)
        plain = st.plan_clone_chunks(self.LONG_S1)
        self.assertEqual(getattr(plain[0], "clause", ""), "head")
        cached = st.plan_clone_chunks(self.LONG_S1,
                                      is_cached=lambda t: t == self.LONG_S1)
        self.assertEqual(cached, [self.LONG_S1])

    def test_a_short_one_piece_reply_is_never_split_for_the_cache(self):
        # Each piece plays on a new stream: a boundary is ~0.4-0.7 s of
        # silence until C5, a pause one render never had.
        self.assertLessEqual(len(self.SHORT), st.CLAUSE_SPLIT_MIN_CHARS)
        self.assertEqual(st.plan_clone_chunks(self.SHORT), [self.SHORT])
        self.assertEqual(st.plan_clone_chunks(
            self.SHORT, is_cached=lambda t: t == "Very good, sir."),
            [self.SHORT])
        # The whole short reply cached: one chunk, as before.
        self.assertEqual(st.plan_clone_chunks(
            self.SHORT, is_cached=lambda t: True), [self.SHORT])

    def test_a_first_line_split_anyway_splits_at_its_cached_opener(self):
        text = ("Very good, sir. The lamp in the study is off now and the "
                "heating is down to sixty eight.")
        self.assertGreater(len(text), st.CLAUSE_SPLIT_MIN_CHARS)
        self.assertLess(len(text), st.MIN_CHARS)
        plain = st.plan_clone_chunks(text)
        self.assertGreater(len(plain), 1)              # split anyway
        plan = st.plan_clone_chunks(
            text, is_cached=lambda t: t == "Very good, sir.")
        self.assertEqual(plan[0], "Very good, sir.")
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[1].budget_chars, len(text))
        self.assertLessEqual(len(plan), len(plain))   # never more pieces

    def test_a_cached_opener_of_a_long_reply(self):
        text = "Certainly, sir. " + self.LONG_S1 + " And that is all."
        plan = st.plan_clone_chunks(
            text, is_cached=lambda t: t == "Certainly, sir.")
        self.assertEqual(plan[0], "Certainly, sir.")
        self.assertEqual(" ".join(plan), text)

    def test_a_raising_probe_is_no_cache(self):
        def boom(_t):
            raise RuntimeError("x")
        self.assertEqual(st.plan_clone_chunks(self.LONG_S1, is_cached=boom),
                         st.plan_clone_chunks(self.LONG_S1))


if __name__ == "__main__":
    unittest.main()
