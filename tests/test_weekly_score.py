"""tools/weekly_score.py and skills/weekly_score.py: synthetic logs only."""
import json
import os
import tempfile
import time
import unittest

from tools import weekly_score as ws


def _t(outcome="ok", kind="voice", you=400, fp=1400):
    return (f"[08:20:01]   [turn-timing] kind={kind} outcome={outcome} "
            f"you={you} first_play={fp} end=9000")


class ComputeTests(unittest.TestCase):
    def test_counts(self):
        lines = [
            _t(), _t(fp=2400), _t("shortcut", fp=900), _t("error"),
            _t(kind="inject"),
            "[08:21:00]   [action] open_on_monitor: could not launch x",
            "[08:21:01]   [action] open_on_monitor: couldn't find y",
            "[08:21:02]   [action] volume_up: volume 50%",
            "[08:21:03]   [action] close_window: failed (final)",
            "[08:21:04]   [turn-check] said_no_action conf=0.9",
        ]
        r = ws.compute(lines)
        self.assertEqual(r["owner_turns"], 4)
        self.assertEqual(r["turns_answered"], 3)
        self.assertEqual(r["injected_turns"], 1)
        self.assertEqual(r["actions_run"], 4)
        self.assertEqual(r["actions_failed"], 3)
        self.assertEqual(r["top_failing_actions"][0],
                         {"action": "open_on_monitor", "failures": 2})
        self.assertEqual(r["median_latency_ms"], 1000)
        self.assertEqual(r["turn_check_flags"], 1)

    def test_empty_and_malformed(self):
        r = ws.compute([])
        self.assertEqual(r["owner_turns"], 0)
        self.assertIsNone(r["answer_rate"])
        self.assertIsNone(r["median_latency_ms"])
        r = ws.compute(["[turn-timing] garbage", "[action] ???",
                        "[turn-timing] kind=voice outcome=ok you=x",
                        "\x00\x01 [action]", ""])
        self.assertEqual(r["actions_run"], 0)
        self.assertEqual(r["owner_turns"], 1)
        self.assertIsNone(r["median_latency_ms"])

    def test_summary(self):
        self.assertIn("nothing to score", ws.summary_text(ws.compute([])))
        txt = ws.summary_text(ws.compute([_t(), "[x]   [action] a_b: failed"]))
        self.assertIn("1 times", txt)
        self.assertIn("a b", txt)


class FileTests(unittest.TestCase):
    def test_window_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.time()
            new = os.path.join(d, "session_2099-01-01_00-00-00.log")
            old = os.path.join(d, "session_2020-01-01_00-00-00.log")
            for p, body in ((new, _t() + "\n"), (old, _t() + "\n" + _t())):
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(body)
            os.utime(old, (now - 40 * 86400, now - 40 * 86400))
            r = ws.build(d, now=now)
            self.assertEqual(r["sessions"], 1)
            self.assertEqual(r["owner_turns"], 1)
            out = ws.write_json(r, os.path.join(d, "w.json"))
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data
                             ["owner_turns"], 1)

    def test_missing_dir(self):
        r = ws.build(os.path.join(tempfile.gettempdir(), "nope_ws_dir"))
        self.assertEqual(r["sessions"], 0)


class SkillTests(unittest.TestCase):
    def test_register(self):
        from skills import weekly_score as sk
        acts = {}
        sk.register(acts)
        self.assertIn("weekly_score", acts)
        self.assertTrue(set(acts) <= set(sk.SPEAK_VERBATIM_ACTIONS))


if __name__ == "__main__":
    unittest.main()
