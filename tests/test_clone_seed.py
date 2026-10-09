"""core/clone_seed.py -- the line ledger, the one-time history bootstrap, the
once-per-voice seed budget and the cache keeper's quiet-time seeding (voice
architecture C6, 2026-10-05).

Light tier. The keeper drives a REAL CloneVoiceClient against a FAKE
loopback server (tests/_clone_voice_fake.py) with a temporary disk tier;
every reply text is synthetic. Nothing is played, nothing touches a GPU.

Pins:
  * the ledger keeps only a HASH of a line said once and its text from the
    second time; most said first, then shortest; a forgotten line is never
    a seed; it survives a reload and stays bounded;
  * the bootstrap merges sources by MAXIMUM (a reply in both the logs and
    the episode store is one reply), counts each unit once per reply, adds
    the first sentence (what the cache-aware planner looks for) and cleans
    each reply first;
  * the budget is per voice IN TOTAL (the owner approved seeding once,
    <= 60 s of GPU): it survives restarts and days, a new voice starts at
    zero, and a line is tried at most once per voice;
  * the keeper: nothing while the gate is closed or the cache is off (the
    purge still runs); a one-time bootstrap; one seed render per tick into
    the disk tier, never counted toward the clone's cool-down and never
    entered in the ledger; no seeding on the slow decoder, past the budget,
    in an unconsented voice, while the clone is on probation, or for a
    forgotten or already cached line; a seed the take gate rejects is never
    rendered again, nor one the cache later trims; a render abandoned the
    moment the owner starts talking leaves nothing behind and is tried
    again later; the ledger is saved while the owner is quiet (or at most
    every PERSIST_MAX_S), never on every tick mid-conversation.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import core.config as cfg
from core import clone_render_cache as crc
from core import clone_seed as cs
from core import clone_voice_client as cvc
from core import sentence_tts as st
from core import voice_clone as vc
from tests._clone_voice_fake import FakeCloneServer, ProfileDir, make_wav


def fit_wav(text: str) -> bytes:
    ms = crc.GATE_FIXED_MS + crc.GATE_MS_PER_CHAR * len(text)
    return make_wav(lead_s=0.05, speech_s=ms / 1000.0 - 0.15, tail_s=0.1,
                    amp=0.3)


class _Wall:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


# Noon local time on two consecutive days (never near midnight).
_DAY1 = time.mktime((2026, 10, 5, 12, 0, 0, 0, 0, -1))
_DAY2 = _DAY1 + 86400.0


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "lines.json")

    def test_a_one_off_line_is_only_a_hash(self):
        led = cs.LineLedger(self.path, wall=_Wall(_DAY1))
        led.record("Your parcel arrived at the side door.")
        self.assertTrue(led.save_if_dirty())
        with open(self.path, encoding="utf-8") as f:
            raw = f.read()
        self.assertNotIn("parcel", raw)
        self.assertEqual(led.seeds(), [])
        led.record("Your parcel arrived at the side door.")
        self.assertEqual(led.seeds(), ["Your parcel arrived at the side door."])

    def test_order_forget_and_reload(self):
        led = cs.LineLedger(self.path, wall=_Wall(_DAY1))
        for t, n in (("Certainly, sir.", 5), ("On it, sir.", 3),
                     ("Right away, sir.", 3), ("Anything further?", 2)):
            for _ in range(n):
                led.record(t)
        self.assertEqual(led.seeds(), ["Certainly, sir.", "On it, sir.",
                                       "Right away, sir.",
                                       "Anything further?"])
        led.forget("On it, sir.")
        self.assertNotIn("On it, sir.", led.seeds())
        led.save_if_dirty()
        again = cs.LineLedger(self.path)
        self.assertEqual(again.seeds(), led.seeds())
        self.assertTrue(again.is_forgotten("On it, sir."))
        self.assertEqual(again.count("Certainly, sir."), 5)

    def test_a_tampered_text_is_dropped_on_load(self):
        led = cs.LineLedger(self.path)
        led.record("Certainly, sir.")
        led.record("Certainly, sir.")
        led.save_if_dirty()
        with open(self.path, encoding="utf-8") as f:
            obj = json.load(f)
        for e in obj["lines"].values():
            e["t"] = "Something else entirely."
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        self.assertEqual(cs.LineLedger(self.path).seeds(), [])

    def test_bounded(self):
        led = cs.LineLedger(None)
        with mock.patch.object(cs, "LEDGER_MAX", 50):
            led.record("Kept, sir.")
            led.record("Kept, sir.")
            for i in range(200):
                led.record(f"one-off line number {i}")
        self.assertLessEqual(len(led), 50)
        self.assertEqual(led.seeds(), ["Kept, sir."])

    def test_long_lines_never_become_seeds(self):
        led = cs.LineLedger(None)
        long = "word " * 80
        led.record(long)
        led.record(long)
        self.assertEqual(led.seeds(), [])


class HistoryTests(unittest.TestCase):
    def test_log_and_episode_readers(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = os.path.join(tmp.name, "session_x.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write("[10:00:00]   You:    lights off\n"
                    "[10:00:01]   JARVIS: [intent:ack] Very good, sir.\n"
                    "[10:00:02]   JARVIS (spoken): A joke, sir.\n"
                    "[10:00:03]   JARVIS (retry): not this one\n"
                    "[10:00:04]   [tts] clone voice 500 ms\n")
        self.assertEqual(cs.log_replies([log]),
                         ["[intent:ack] Very good, sir.", "A joke, sir."])
        ep = os.path.join(tmp.name, "episodes.jsonl")
        with open(ep, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "text": "hi"}) + "\n")
            f.write(json.dumps({"role": "assistant", "text": "Hello, sir."})
                    + "\n")
            f.write("not json\n")
        self.assertEqual(cs.episode_replies(ep), ["Hello, sir."])
        self.assertEqual(cs.log_replies([os.path.join(tmp.name, "nope")]), [])

    def test_units_are_the_planned_chunks_plus_the_first_sentence(self):
        text = ("Certainly, sir. The forecast for tomorrow looks mild, with "
                "a gentle breeze from the west and grey skies all afternoon.")
        units = cs.reply_units(text, plan=st.plan_clone_chunks)
        self.assertEqual(units[0], "Certainly, sir.")
        self.assertEqual(len(units), len(set(units)))
        short = "Very good, sir. The lamp is off now."
        self.assertEqual(cs.reply_units(short, plan=st.plan_clone_chunks),
                         [short, "Very good, sir."])
        self.assertEqual(cs.reply_units(
            "[intent:x] Done, sir.", clean=lambda t: t.split("] ", 1)[1]),
            ["Done, sir."])

    def test_sources_merge_by_maximum(self):
        logs = ["Very good, sir.", "Very good, sir.", "Once only, sir."]
        episodes = ["Very good, sir.", "Once only, sir.", "Very good, sir.",
                    "Very good, sir."]
        counts = cs.bootstrap_counts([logs, episodes])
        self.assertEqual(counts["Very good, sir."], 3)    # max(2, 3), not 5
        self.assertEqual(counts["Once only, sir."], 1)    # not 2
        led = cs.LineLedger(None)
        self.assertEqual(led.merge_counts(counts), 1)
        self.assertTrue(led.bootstrapped)
        self.assertEqual(led.seeds(), ["Very good, sir."])


class BudgetTests(unittest.TestCase):
    def test_per_voice_in_total_across_restarts_and_days(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "seed_state.json")
        b = cs.SeedBudget(path)
        b.add("a" * 16, 1500)
        b.mark_tried("a" * 16, "Certainly, sir.")
        self.assertAlmostEqual(b.used_s("a" * 16), 1.5)
        self.assertEqual(b.used_s("b" * 16), 0.0)        # a new voice
        self.assertTrue(b.tried("a" * 16, "Certainly, sir."))
        self.assertFalse(b.tried("b" * 16, "Certainly, sir."))
        b.save_if_dirty()
        # A restart -- on any later day -- finds the same totals.
        with mock.patch.object(time, "time", return_value=_DAY2 + 86400.0):
            again = cs.SeedBudget(path)
            self.assertAlmostEqual(again.used_s("a" * 16), 1.5)
            self.assertTrue(again.tried("a" * 16, "Certainly, sir."))
        again.unmark_tried("a" * 16, "Certainly, sir.")
        self.assertFalse(again.tried("a" * 16, "Certainly, sir."))

    def test_an_older_per_day_file_starts_over(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "seed_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"date": "2026-10-05",
                       "voices": {"a" * 16: {"gpu_ms": 5000.0,
                                             "tried": []}}}, f)
        self.assertEqual(cs.SeedBudget(path).used_s("a" * 16), 0.0)

    def test_shipped_budget_is_the_owners_sixty_seconds(self):
        self.assertEqual(cs.DEFAULT_SEED_GPU_S, 60.0)
        self.assertEqual(cfg.VOICE_CLONE_SEED_GPU_S, cs.DEFAULT_SEED_GPU_S)
        from tools import settings_window as sw
        self.assertEqual(sw.SCHEMA["VOICE_CLONE_SEED_GPU_S"]["default"],
                         cs.DEFAULT_SEED_GPU_S)
        ex = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tools", "user_settings.example.json")
        with open(ex, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["VOICE_CLONE_SEED_GPU_S"],
                             cs.DEFAULT_SEED_GPU_S)


# ════════════════════════════════════════════════════════════════════════════
#  The keeper
# ════════════════════════════════════════════════════════════════════════════
SEEDS = ("Certainly, sir.", "Right away, sir.", "Anything further?")


class KeeperTests(unittest.TestCase):
    def setUp(self):
        self.prof = ProfileDir("butler")
        self.addCleanup(self.prof.cleanup)
        for target, name, value in ((vc, "PROFILES_DIR", self.prof.root),
                                    (cvc, "PROFILE_TTL_S", 0.0),
                                    (cfg, "VOICE_CLONE_CACHE", "on"),
                                    (cfg, "VOICE_CLONE_SEED_GPU_S", 90.0)):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = os.path.join(self._tmp.name, "clone_cache")
        self.logs: list = []
        self.gate = [None]
        self.abort = [False]
        self.util = [5]
        p = mock.patch.object(cs, "gpu_util_pct",
                              side_effect=lambda idx=None: self.util[0])
        self.gpu = p.start()
        self.addCleanup(p.stop)

    def setup(self, history=None, **srv_kw):
        srv_kw.setdefault("wav_for", {t: fit_wav(t) for t in SEEDS})
        self.srv = FakeCloneServer(ref_sha=self.prof.sha, **srv_kw).start()
        self.addCleanup(self.srv.stop)
        self.c = cvc.CloneVoiceClient(log=self.logs.append, boot_wait_s=3.0,
                                      sleep=lambda s: time.sleep(min(s, 0.02)))
        self.assertTrue(self.c.attach_cache(self.dir))
        self.addCleanup(lambda: self.c.store.flush(5.0))
        self.addCleanup(self.c.store.close)  # its writer thread (runs first)
        self.assertEqual(self.c.start(url=self.srv.url, cmd="",
                                      profile="butler"), "ready")
        self.clock = _Wall(1000.0)
        self.k = cs.CacheKeeper(
            self.c, gate_fn=lambda: self.gate[0],
            abort_fn=lambda: self.abort[0],
            history_fn=(lambda: history) if history is not None else None,
            plan=st.plan_clone_chunks, log=self.logs.append,
            clock=self.clock)
        return self.k

    def ledger_with(self, *texts, n=2):
        for t in texts:
            for _ in range(n):
                self.c.ledger.record(t)

    def test_nothing_while_the_gate_is_closed(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        self.gate[0] = "owner active"
        self.assertEqual(k.tick(), "owner active")
        self.assertEqual(self.srv.tts_texts(), [])

    def test_off_purges_but_never_seeds(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE", "off"), \
                mock.patch.object(self.c, "purge_unconsented",
                                  wraps=self.c.purge_unconsented) as purge:
            self.assertEqual(k.tick(), "off")
            self.assertEqual(purge.call_count, 1)
        with mock.patch.object(cfg, "VOICE_CLONE_CACHE", "shadow"):
            self.assertEqual(k.tick(), "shadow")
        self.assertEqual(self.srv.tts_texts(), [])

    def test_bootstrap_once_then_seed_one_line_per_tick(self):
        history = [["[intent:x] Certainly, sir. I'll see to it.",
                    "Certainly, sir. I'll see to it.",
                    "Right away, sir.", "Right away, sir.",
                    "A line said once, sir."],
                   ["Right away, sir."]]
        k = self.setup(history=history, wav_for={
            t: fit_wav(t) for t in SEEDS + ("Certainly, sir. I'll see to it.",)})
        k._clean = lambda t: t.replace("[intent:x] ", "")
        self.assertEqual(k.tick(), "bootstrapped")
        self.assertTrue(any("line history" in m for m in self.logs))
        self.assertTrue(self.c.ledger.bootstrapped)
        seeds = self.c.ledger.seeds()
        self.assertIn("Right away, sir.", seeds)
        self.assertIn("Certainly, sir.", seeds)        # first sentence
        self.assertNotIn("A line said once, sir.", seeds)
        done = []
        for _ in range(10):
            r = k.tick()
            done.append(r)
            if r != "seeded":
                break
        self.assertEqual(done[-1], "done", done)
        self.assertEqual(done.count("seeded"), len(seeds))
        self.assertEqual(sorted(self.srv.tts_texts()), sorted(seeds))
        self.assertTrue(self.c.store.flush(5.0))
        for t in seeds:
            self.assertTrue(self.c.is_cached(t), t)
        # Seeds never enter the ledger, never touch the miss count.
        self.assertEqual(self.c.ledger.count("Right away, sir."), 2)
        self.assertEqual(self.c.failures(), 0)
        self.assertGreater(self.c.budget.used_s(self.c.voice_prefix()), 0.0)
        msgs = [m for m in self.logs if "[clone-cache] seed" in m]
        self.assertEqual(len(msgs), 2, self.logs)    # start + end of burst
        # A second keeper over the same folder has nothing left to do.
        self.assertEqual(k.tick(), "done")
        self.assertFalse(any(k.tick() == "bootstrapped" for _ in range(2)))

    def test_no_seeding_on_the_slow_decoder(self):
        k = self.setup(t3_decode="eager")
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        self.assertEqual(k.tick(), "fast decode off")
        self.assertEqual(self.srv.tts_texts(), [])

    def test_no_seeding_while_the_gpu_is_busy(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        self.util[0] = 85                             # a game
        self.assertEqual(k.tick(), "gpu busy")
        self.gpu.assert_called_with(0)                # the server's GPU
        self.assertEqual(self.srv.tts_texts(), [])
        self.util[0] = None                           # unreadable: no veto
        self.assertEqual(k.tick(), "seeded")

    def test_the_budget(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        self.c.budget.add(self.c.voice_prefix(), 90_000)
        self.assertEqual(k.tick(), "budget")
        with mock.patch.object(cfg, "VOICE_CLONE_SEED_GPU_S", 0):
            self.assertEqual(k.tick(), "seed-off")
        self.assertEqual(self.srv.tts_texts(), [])

    def test_a_forgotten_line_is_never_seeded(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        self.c.ledger.forget("Certainly, sir.")
        while k.tick() == "seeded":
            pass
        self.assertNotIn("Certainly, sir.", self.srv.tts_texts())
        self.assertEqual(len(self.srv.tts_texts()), 2)

    def test_the_owner_talking_abandons_a_seed_render(self):
        k = self.setup(tts_delay=1.0)
        self.ledger_with("Certainly, sir.")
        self.c.ledger.bootstrapped = True
        threading.Timer(0.2, lambda: self.abort.__setitem__(0, True)).start()
        t0 = time.monotonic()
        self.assertEqual(k.tick(), "aborted")
        self.assertLess(time.monotonic() - t0, 0.9)
        self.assertTrue(self.c.store.flush(5.0))
        self.assertFalse(self.c.is_cached("Certainly, sir."))
        self.assertEqual(self.c.failures(), 0)
        self.assertEqual(self.c.status()[0], "ready")

    def test_a_slow_seed_never_starts_a_cool_down(self):
        k = self.setup(tts_delay=0.6)
        with mock.patch.object(cs, "SEED_TIMEOUT_S", 0.2):
            for t in SEEDS:
                self.ledger_with(t)
            self.c.ledger.bootstrapped = True
            for _ in range(len(SEEDS)):
                k.tick()
        self.assertEqual(self.c.failures(), 0)
        self.assertEqual(self.c.status()[0], "ready")

    def test_a_seed_the_take_gate_rejects_is_rendered_once(self):
        runaway = "Certainly, sir."
        wavs = {t: fit_wav(t) for t in SEEDS}
        wavs[runaway] = make_wav(lead_s=0.0, speech_s=12.0, tail_s=0.0,
                                 amp=0.3)
        k = self.setup(wav_for=wavs)
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        for _ in range(8):
            k.tick()
            self.assertTrue(self.c.store.flush(5.0))
        self.assertEqual(self.srv.tts_texts().count(runaway), 1)
        self.assertEqual(sorted(self.srv.tts_texts()), sorted(SEEDS))

    def test_each_line_is_seeded_once_per_voice_ever(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        while k.tick() == "seeded":
            pass
        self.assertTrue(self.c.store.flush(5.0))
        n = len(self.srv.tts_texts())
        self.assertEqual(n, len(SEEDS))
        # The cache trims one later; nothing renders it again -- not on a
        # later day, not after a restart (a new budget over the same file).
        key = self.c._key(self.c._server_sha, self.c._server_info,
                          "Certainly, sir.")
        self.c.store.forget([key], count=False)
        self.assertFalse(self.c.is_cached("Certainly, sir."))
        self.c.budget.save_if_dirty()
        self.c.budget = cs.SeedBudget(os.path.join(self.dir, cs.STATE_FILE))
        with mock.patch.object(time, "time", return_value=_DAY2 + 86400.0):
            for _ in range(3):
                self.assertEqual(k.tick(), "done")
        self.assertEqual(len(self.srv.tts_texts()), n)

    def test_a_line_already_cached_is_not_seeded(self):
        k = self.setup()
        # Said for a listener (cached) before the keeper got to it.
        self.assertTrue(self.c.render("Certainly, sir.", 2.5).ok)
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        while k.tick() == "seeded":
            pass
        self.assertEqual(self.srv.tts_texts().count("Certainly, sir."), 1)
        self.assertEqual(len(self.srv.tts_texts()), len(SEEDS))

    def test_a_seed_abandoned_for_the_owner_is_tried_again_later(self):
        k = self.setup(tts_delay=1.0)
        self.ledger_with("Certainly, sir.")
        self.c.ledger.bootstrapped = True
        threading.Timer(0.2, lambda: self.abort.__setitem__(0, True)).start()
        self.assertEqual(k.tick(), "aborted")
        self.assertFalse(self.c.budget.tried(self.c.voice_prefix(),
                                             "Certainly, sir."))
        self.abort[0] = False
        self.srv.tts_delay = 0.0
        self.assertEqual(k.tick(), "seeded")
        self.assertTrue(self.c.store.flush(5.0))
        self.assertTrue(self.c.is_cached("Certainly, sir."))

    def test_no_seeding_in_a_voice_that_is_not_the_consented_one(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        with mock.patch.object(self.c, "profile_sha", return_value="b" * 64):
            self.assertEqual(k.tick(), "voice not the consented profile's")
        self.assertEqual(self.srv.tts_texts(), [])

    def test_no_seeding_while_the_clone_is_missing_lines(self):
        k = self.setup()
        self.ledger_with(*SEEDS)
        self.c.ledger.bootstrapped = True
        for state in ({"_probation": True}, {"_fails": 1},
                      {"_recheck": True}):
            for name, v in state.items():
                setattr(self.c, name, v)
            self.assertEqual(k.tick(), "clone missing lines", state)
            self.c._probation, self.c._fails, self.c._recheck = \
                False, 0, False
        self.assertEqual(self.srv.tts_texts(), [])

    def test_the_ledger_is_saved_while_quiet_not_on_every_busy_tick(self):
        k = self.setup()
        path = os.path.join(self.dir, cs.LEDGER_FILE)
        self.c.ledger.bootstrapped = True
        self.c.ledger.record("A line before the quiet tick.")
        self.gate[0] = None
        k.tick()                                  # quiet: saved
        self.assertTrue(os.path.exists(path))
        self.c.ledger.record("Mid-conversation line.")
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        self.gate[0] = "owner active"
        for _ in range(5):
            k.tick()
            self.clock.t += 30.0
        self.assertEqual((os.path.getmtime(path), os.path.getsize(path)),
                         (mtime, size))
        self.clock.t += cs.PERSIST_MAX_S
        k.tick()                                  # at most every 5 min
        self.assertNotEqual(os.path.getsize(path), size)

    def test_the_daemon_starts_once(self):
        k = self.setup()
        with mock.patch.object(k, "_loop", lambda: None):
            self.assertTrue(k.start())
            self.assertFalse(k.start())


if __name__ == "__main__":
    unittest.main()
