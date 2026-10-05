"""core/vision_trace.py - the owner-approved vision trace ("Yes, with
limits", 2026-10-05): local only, 7 days / 300 entries / 300 MB, private
windows text-only, never in the repo.

    python -m unittest tests.test_vision_trace
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from core import config as cfg
from core import vision_trace as VT

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _png(w=320, h=200, color=(200, 30, 30)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


class _Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="vtrace_")
        self.addCleanup(shutil.rmtree, self.td, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td})
        env.start()
        self.addCleanup(env.stop)
        self.cfg = {}
        self._set(VISION_TRACE="on", VISION_TRACE_DAYS=7,
                  VISION_TRACE_MAX_ENTRIES=300, VISION_TRACE_MAX_MB=300)
        VT._reset_for_tests()
        p = mock.patch("builtins.print")
        p.start()
        self.addCleanup(p.stop)

    def _set(self, **kw):
        for k, v in kw.items():
            p = mock.patch.object(cfg, k, v, create=True)
            p.start()
            self.addCleanup(p.stop)

    def _entries(self):
        VT.flush(5)
        return VT.read_index()


class RecordTests(_Base):
    def test_a_step_records_images_as_sent_prompt_and_answer(self):
        with VT.step("click", utterance="click that MrBeast video",
                     source="mark") as st:
            st.model_call("Which number is the MrBeast video?", [_png()],
                          "[local-vision] 3", ms=1900)
            st.finish("verified", evidence="the address is now ...",
                      action={"how": "mouse", "x": 10, "y": 20})
        e = self._entries()[-1]
        self.assertEqual(e["step"], "click")
        self.assertEqual(e["outcome"], "verified")
        self.assertEqual(e["raw_answer"], "[local-vision] 3")
        self.assertTrue(e["prompt_sha1"])
        self.assertEqual(len(e["images"]), 1)
        img = e["images"][0]
        self.assertEqual(img["as_sent"], "320x200")
        path = os.path.join(VT.trace_dir(), *img["file"].split("/"))
        self.assertTrue(os.path.exists(path))
        self.assertTrue(img["file"].endswith(".webp"))

    def test_the_chokepoint_hook_attaches_to_the_open_step(self):
        with VT.step("see_screen", utterance="what's on my screen") as st:
            VT.note_model_call("Describe", [_png()], "a page", ms=500)
            self.assertIs(VT.current(), st)
        self.assertIsNone(VT.current())
        e = self._entries()[-1]
        self.assertEqual(e["model_calls"][0]["raw_answer"], "a page")

    def test_a_background_call_is_not_traced(self):
        VT.note_model_call("credits?", [_png()], "x")      # no step open
        with VT.step("x", utterance="") as st:              # no owner words
            self.assertFalse(st)
        self.assertEqual(self._entries(), [])

    def test_text_mode_stores_no_images(self):
        self._set(VISION_TRACE="text")
        with VT.step("click", utterance="click it") as st:
            st.model_call("q", [_png()], "a")
        e = self._entries()[-1]
        self.assertEqual(e["images"], [])
        self.assertEqual(e["raw_answer"], "a")

    def test_off_mode_stores_nothing(self):
        self._set(VISION_TRACE="off")
        with VT.step("click", utterance="click it") as st:
            st.model_call("q", [_png()], "a")
        self.assertEqual(self._entries(), [])
        self.assertFalse(os.path.exists(os.path.join(self.td, "vision_trace",
                                                     "index.jsonl")))

    def test_private_entry_carries_no_title_url_prompt_answer_or_image(self):
        with VT.step("click", utterance="click the vault",
                     scope={"window_title": "Vault - 1Password",
                            "url_host": "my.1password.com"}) as st:
            st.model_call("Where is the vault button?", [_png()], "here")
            st.mark_private("1password")
            st.finish("refused_private")
        e = self._entries()[-1]
        self.assertEqual(e["privacy"], VT.PRIVATE_SKIP)
        self.assertEqual(set(e), {"id", "ts", "step", "outcome", "privacy"})
        blob = json.dumps(e)
        for secret in ("Vault", "1password", "vault button", "here"):
            self.assertNotIn(secret, blob)
        webps = [f for _r, _d, fs in os.walk(self.td) for f in fs
                 if f.endswith(".webp")]
        self.assertEqual(webps, [])


class RetentionTests(_Base):
    def _write(self, n, ts, images=0):
        for i in range(n):
            with VT.step("click", utterance=f"u{i}") as st:
                st.entry["ts"] = ts + i
                for _ in range(images):
                    st.add_image(_png())
                st.finish("verified")
        VT.flush(5)

    def test_prune_by_days(self):
        now = time.time()
        self._write(3, now - 9 * 86400, images=1)
        self._write(2, now - 60)
        VT.prune(now=now)
        es = VT.read_index()
        self.assertEqual(len(es), 2)
        webps = [f for _r, _d, fs in os.walk(self.td) for f in fs
                 if f.endswith(".webp")]
        self.assertEqual(webps, [])          # files went with their entries

    def test_prune_by_entries_oldest_first(self):
        self._set(VISION_TRACE_MAX_ENTRIES=5)
        now = time.time()
        self._write(8, now - 600)
        VT.prune(now=now)
        es = VT.read_index()
        self.assertEqual(len(es), 5)
        self.assertEqual([e["utterance"] for e in es],
                         ["u3", "u4", "u5", "u6", "u7"])

    def test_the_writer_rotates_by_itself(self):
        self._set(VISION_TRACE_MAX_ENTRIES=4)
        self._write(9, time.time() - 100)
        self.assertLessEqual(len(VT.read_index()), 4)

    def test_prune_by_size(self):
        self._set(VISION_TRACE_MAX_MB=0.0005)          # ~500 bytes
        now = time.time()
        self._write(4, now - 600, images=1)
        VT.prune(now=now)
        total = sum(os.path.getsize(os.path.join(r, f))
                    for r, _d, fs in os.walk(VT.trace_dir()) for f in fs)
        self.assertLess(len(VT.read_index()), 4)
        self.assertLess(total, 40000)

    def test_compaction_uses_a_unique_temp_file(self):
        self._write(2, time.time() - 10)
        names = []
        real = os.replace

        def spy(src, dst):
            names.append(os.path.basename(src))
            return real(src, dst)
        with mock.patch.object(VT.os, "replace", spy):
            VT.prune()
            VT.prune()
        self.assertEqual(len(names), 2)
        self.assertNotEqual(names[0], names[1])
        self.assertTrue(all(n.startswith("index.") and n.endswith(".tmp")
                            for n in names))

    def test_purge_a_span(self):
        now = time.time()
        self._write(3, now - 7200, images=1)
        self._write(2, now - 60, images=1)
        gone = VT.purge(now - 3600, now)
        self.assertEqual(gone, 2)
        self.assertEqual(len(VT.read_index()), 3)


class WriterTests(_Base):
    def test_overflow_is_counted_never_blocking(self):
        VT.flush(5)
        before = VT.stats()["overflow"]
        with mock.patch.object(VT._q, "put_nowait",
                               side_effect=VT.queue.Full):
            t0 = time.monotonic()
            for i in range(5):
                VT.record("click", utterance="x", outcome="verified")
            self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(VT.stats()["overflow"] - before, 5)


class RepoTests(unittest.TestCase):
    def test_the_trace_lives_under_the_gitignored_data_dir(self):
        with mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": ""}):
            d = VT.trace_dir(create=False)
        rel = os.path.relpath(d, _ROOT).replace("\\", "/")
        self.assertTrue(rel.startswith("data/") or rel.startswith(
            "data_staging/"), rel)
        with open(os.path.join(_ROOT, ".gitignore"), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        self.assertIn("data/*", lines)
        for path in ("data/vision_trace/index.jsonl",
                     "data/vision_trace/20261005/vt-1_1.webp",
                     "data/screen_timeline.db", "data/notes_for_claude.jsonl",
                     "data/screen_memory_state.json"):
            try:
                r = subprocess.run(["git", "check-ignore", "-q", path],
                                   cwd=_ROOT, capture_output=True, timeout=20)
            except Exception as e:  # no git on a runner: the line check holds
                self.skipTest(f"git unavailable: {e}")
            if r.returncode == 128:
                self.skipTest(f"git check-ignore unusable here: {r.stderr!r}")
            self.assertEqual(r.returncode, 0, f"{path} is not gitignored")

    def test_the_writer_never_touches_the_network(self):
        src = open(VT.__file__, encoding="utf-8").read()
        for word in ("requests", "urllib.request", "socket", "http.client",
                     "anthropic"):
            self.assertNotIn(f"import {word}", src)


if __name__ == "__main__":
    unittest.main()
