"""core/screen_resolve.py - spoken target -> on-screen element, from text.

The pages are the SYNTHETIC research pages (tests/_screen_pages.json: UIA
trees and OCR lines of pages rendered in a throw-away Chrome profile on a
hidden desktop, 2026-10-05). Dev set (written with the resolver) and the
held-out set (written BEFORE the resolver ever saw that page):

  * dev:      UIA 30/30, OCR 33/33 (the research prototype's result) must hold;
  * held-out: UIA >= 17/20 and OCR >= 17/20 with ZERO wrong targets - a miss
    must be a decline ("which one?" / not found), never a wrong click.

Note on honesty: the held-out page also showed the three general fixes this
module adds (cards counted in reading order without the sidebar, light
stemming, compound containment), so after this build it is no longer a
truly held-out set; tools/vision_bench writes fresh pages for that.

    python -m unittest tests.test_screen_resolve
"""
from __future__ import annotations

import json
import os
import unittest

from core import screen_resolve as R

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "_screen_pages.json"), encoding="utf-8") as _f:
    PAGES = json.load(_f)

DEV = [
    ("home_dark", "click that MrBeast video", "v01"),
    ("home_dark", "click that Mr. Beast video", "v01"),
    ("home_dark", "play the mister beast one", "v01"),
    ("home_dark", "play the Kai Cenat one", "v02"),
    ("home_dark", "click the kai senat video", "v02"),
    ("home_dark", "open the veritasium video about bridges", "v03"),
    ("home_dark", "click the one about helicopters", "v09"),
    ("home_dark", "click the pizza video", "v02"),
    ("home_dark", "play the lofi one", "v10"),
    ("home_dark", "open the burger ranking video", "v15"),
    ("home_dark", "click the beast mode one", "v07"),
    ("home_dark", "click the Mark Rober video", "v06"),
    ("home_dark", "click the video with the glitter bomb", "v06"),
    ("home_dark", "open the linus tech tips video", "v05"),
    ("home_dark", "click the second video", "v02"),
    ("home_dark", "play the first one", "v01"),
    ("home_dark", "click the Formula 1 video", "v14"),
    ("home_dark", "click the cabin video", "v11"),
    ("home_dark", "click the marques brownlee video", "v04"),
    ("home_dark", "click subscriptions", ("kind", "side-Subscriptions")),
    ("home_dark", "click recently uploaded", ("kind", "chip-Recently uploaded")),
    ("home2beast_dark", "click that MrBeast video", ("amb", {"v01", "v12"})),
    ("home2beast_dark", "click the second MrBeast video", "v12"),
    ("home2beast_dark", "click the MrBeast circle video", "v12"),
    ("watch_dark", "click the one that's playing", "w00"),
    ("watch_dark", "click the Dude Perfect video", "v08"),
    ("watch_dark", "click the cookbook video", "v12"),
    ("watch_dark", "click subscribe", ("kind", "subscribe")),
    ("watch_dark", "click share", ("kind", "share")),
    ("console_light", "click create key", ("kind", "create-key")),
    ("console_light", "click usage", ("kind", "nav-Usage")),
    ("console_light", "click billing", ("kind", "nav-Billing")),
    ("chooser_light", "click use another account", ("kind", "other")),
]
HOLDOUT = [
    ("holdout_light", "click the IShowSpeed video", "h03"),
    ("holdout_light", "play the i show speed one", "h03"),
    ("holdout_light", "the speed video in japan", "h03"),
    ("holdout_light", "click the sidemen video", "h05"),
    ("holdout_light", "open the one about the deepest hole", "h07"),
    ("holdout_light", "play the minecraft one", "h02"),
    ("holdout_light", "click the third video", "h03"),
    ("holdout_light", "click the last video", "h18"),
    ("holdout_light", "click the ludwig one", "h09"),
    ("holdout_light", "open the video about the James Webb telescope", "h11"),
    ("holdout_light", "click the cooking video", "h12"),
    ("holdout_light", "click the jidion video", "h14"),
    ("holdout_light", "click the one with the tesla", "h15"),
    ("holdout_light", "click history", ("kind", "side-History")),
    ("holdout_light", "click the shorts tab", ("kind", "side-Shorts")),
    ("holdout_light", "play the video from smosh", "h16"),
    ("holdout_light", "click the airplane one", "h17"),
    ("holdout_light", "click the danny duncan video", "h18"),
    ("holdout_light", "click the music chip", ("kind", "chip-Music")),
    ("holdout_light", "play the rocket landing video", "h10"),
]
# Negative set: nothing on the page is this - the resolver must decline.
NEGATIVE = [
    ("watch_dark", "the MrBeast video"),
    ("watch_dark", "the mister beast one"),
    ("home_dark", "the dragon video"),
    ("home_dark", "the minecraft video"),
    ("holdout_light", "the MrBeast video"),
    ("console_light", "the delete account button"),
]


def _inside(pt, r):
    return r[0] <= pt[0] <= r[0] + r[2] and r[1] <= pt[1] <= r[1] + r[3]


def uia_page(name):
    pg = PAGES["uia"][name]
    els = pg["elements"]
    doc = next(e for e in els if e["type"] == "Document"
               and e["fw"] == "Chrome")["rect"]
    cands = [{"text": e["name"], "rect": e["rect"], "type": e["type"]}
             for e in els if e["fw"] == "Chrome" and e["rect"] and e["name"]
             and e["type"] != "Document" and e["rect"][1] >= doc[1]]
    gt = [dict(g, rect=[doc[0] + g["rect"][0], doc[1] + g["rect"][1],
                        g["rect"][2], g["rect"][3]]) for g in pg["gt"]]
    return cands, gt, pg["title"].rsplit(" - ", 1)[0]


def ocr_page(name):
    pg = PAGES["ocr"][name]
    cands = [{"text": ln["t"], "rect": ln["rect"], "type": "ocr"}
             for ln in pg["lines"] if ln["rect"][1] >= 88]
    return cands, pg["gt"], pg["title"].rsplit(" - ", 1)[0]


def _ok_rects(gt, exp):
    if isinstance(exp, str):
        return [g["rect"] for g in gt if g["id"] == exp
                and g["kind"] in ("title", "thumb")]
    if exp[0] == "kind":
        return [g["rect"] for g in gt if g["id"] == exp[1]]
    return []


def judge(res, gt, exp) -> str:
    """'ok' | 'wrong' | 'declined'."""
    if isinstance(exp, tuple) and exp[0] == "amb":
        if res["status"] == "ok":
            return "wrong"
        if res["status"] != "ambiguous":
            return "declined"
        got = set()
        for o in res["options"]:
            c = (o["rect"][0] + o["rect"][2] / 2, o["rect"][1] + o["rect"][3] / 2)
            for vid in exp[1]:
                if any(_inside(c, r) for r in _ok_rects(gt, vid)):
                    got.add(vid)
        return "ok" if got == exp[1] else "wrong"
    if res["status"] != "ok":
        return "declined"
    r = res["target"]["rect"]
    c = (r[0] + r[2] / 2, r[1] + r[3] / 2)
    return "ok" if any(_inside(c, rr) for rr in _ok_rects(gt, exp)) else "wrong"


def score(source, queries):
    out = {"ok": 0, "wrong": 0, "declined": 0, "n": 0, "fails": []}
    for page, q, exp in queries:
        if page not in PAGES[source]:
            continue
        cands, gt, playing = (uia_page if source == "uia" else ocr_page)(page)
        v = judge(R.resolve(q, cands, playing_title=playing), gt, exp)
        out[v] += 1
        out["n"] += 1
        if v != "ok":
            out["fails"].append((v, q))
    return out


class DevSetTests(unittest.TestCase):
    def test_uia_dev_set_30_of_30(self):
        r = score("uia", DEV)
        self.assertEqual(r["n"], 30)
        self.assertEqual((r["ok"], r["wrong"]), (30, 0), r["fails"])

    def test_ocr_dev_set_33_of_33(self):
        r = score("ocr", DEV)
        self.assertEqual(r["n"], 33)
        self.assertEqual((r["ok"], r["wrong"]), (33, 0), r["fails"])


class HeldOutTests(unittest.TestCase):
    def test_uia_held_out_at_least_17_with_zero_wrong(self):
        r = score("uia", HOLDOUT)
        self.assertEqual(r["wrong"], 0, r["fails"])
        self.assertGreaterEqual(r["ok"], 17, r["fails"])

    def test_ocr_held_out_at_least_17_with_zero_wrong(self):
        r = score("ocr", HOLDOUT)
        self.assertEqual(r["wrong"], 0, r["fails"])
        self.assertGreaterEqual(r["ok"], 17, r["fails"])

    def test_the_third_video_counts_cards_not_the_sidebar(self):
        # The research prototype's one wrong answer: the stacked sidebar was
        # counted as card #1, so "the third video" clicked the second.
        cands, gt, _p = uia_page("holdout_light")
        self.assertEqual(judge(R.resolve("click the third video", cands), gt,
                               "h03"), "ok")

    def test_every_decline_names_the_candidates_or_none(self):
        for source in ("uia", "ocr"):
            for page, q, exp in HOLDOUT:
                cands, gt, playing = (uia_page if source == "uia"
                                      else ocr_page)(page)
                res = R.resolve(q, cands, playing_title=playing)
                if res["status"] == "ambiguous":
                    self.assertGreaterEqual(len(res["options"]), 2)
                    self.assertTrue(R.legend(res["options"], cands))


class NegativeSetTests(unittest.TestCase):
    def test_nothing_matching_is_never_resolved(self):
        for page, q in NEGATIVE:
            cands, _gt, _p = uia_page(page)
            res = R.resolve(q, cands)
            self.assertNotEqual(res["status"], "ok",
                                f"{q!r} on {page} -> {res.get('target')}")

    def test_mrbeast_is_not_beast_mode(self):
        # "mrbeast" vs "beast" is a missing prefix, not a speech slip.
        self.assertEqual(R.tok_sim("mrbeast", "beast"), 0.0)
        self.assertGreater(R.tok_sim("beast", "mrbeast"), 0.0)

    def test_the_one_playing_with_no_playing_title_is_nothing(self):
        # Live UIA bench 2026-10-05: with the playing title unknown (not a
        # YouTube window title), "the one that's playing" picked a sidebar
        # video - the "s" of "that's" and "playing" scored against titles.
        cands, _gt, playing = uia_page("watch_dark")
        res = R.resolve("click the one that's playing", cands)
        self.assertNotEqual(res["status"], "ok", res.get("target"))
        res = R.resolve("click the one that's playing", cands,
                        playing_title=playing)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res.get("why"), "playing")
        # ... and a title that merely CONTAINS the word is not "the one
        # that's playing".
        titles = [{"text": "Playing Minecraft For 100 Days",
                   "rect": [100, 300, 300, 20], "type": "Hyperlink"},
                  {"text": "Cooking With Lava", "rect": [500, 300, 300, 20],
                   "type": "Hyperlink"}]
        res = R.resolve("click the one that's playing", titles)
        self.assertNotEqual(res["status"], "ok", res.get("target"))
        res = R.resolve("click the minecraft one that's playing", titles)
        self.assertEqual(R.label_of(res.get("target") or {}),
                         "Playing Minecraft For 100 Days")


class TokenTests(unittest.TestCase):
    def test_mr_beast_spellings_are_one_token(self):
        for s in ("Mr. Beast", "mr beast", "Mister Beast", "MrBeast"):
            self.assertIn("mrbeast", R.toks(s), s)

    def test_speech_slips(self):
        self.assertGreaterEqual(R.tok_sim("senat", "cenat"), 0.8)
        self.assertGreaterEqual(R.tok_sim("veritasiam", "veritasium"), 0.8)
        self.assertEqual(R.tok_sim("cat", "car"), 0.0)

    def test_stems_and_compounds(self):
        self.assertGreater(R.tok_sim("landing", "land"), 0.0)
        self.assertGreater(R.tok_sim("rockets", "rocket"), 0.0)
        self.assertGreater(R.tok_sim("airplane", "plane"), 0.0)

    def test_stop_words_and_ordinals_are_not_content(self):
        self.assertEqual(R.content_tokens("click that video please"), [])
        self.assertNotIn("s", R.content_tokens("click the one that's up"))
        self.assertEqual(R.content_tokens("part 2 video"), ["part", "2"])
        self.assertEqual(R.content_tokens("the second one"), [])
        self.assertEqual(R.ordinals("the second one"), [2])


class LabelTests(unittest.TestCase):
    def test_long_youtube_label_is_split(self):
        t, ch = R.split_label("How Rockets Land Themselves by Space Explained "
                              "4.4M views 9 months ago 15 minutes 48 seconds")
        self.assertEqual((t, ch), ("How Rockets Land Themselves",
                                   "Space Explained"))

    def test_a_plain_label_is_untouched(self):
        self.assertEqual(R.split_label("Subscriptions"), ("Subscriptions", ""))

    def test_the_spoken_label_is_the_title(self):
        cands, _gt, _p = uia_page("holdout_light")
        res = R.resolve("click the ludwig one", cands)
        self.assertEqual(R.label_of(res["target"]),
                         "Ludwig Reacts To His Old Streams")


class CardTests(unittest.TestCase):
    def test_that_mrbeast_video_is_the_title_not_the_channel(self):
        cands, gt, _p = uia_page("home_dark")
        res = R.resolve("click that MrBeast video", cands)
        self.assertEqual(res["status"], "ok")
        r = res["target"]["rect"]
        c = (r[0] + r[2] / 2, r[1] + r[3] / 2)
        chan = [g["rect"] for g in gt if g["id"] == "v01"
                and g["kind"] == "channel"]
        self.assertTrue(chan)
        self.assertFalse(any(_inside(c, rr) for rr in chan))

    def test_two_mrbeast_videos_are_a_question(self):
        cands, _gt, _p = uia_page("home2beast_dark") if "home2beast_dark" in \
            PAGES["uia"] else ocr_page("home2beast_dark")
        res = R.resolve("click that MrBeast video", cands)
        self.assertEqual(res["status"], "ambiguous")
        self.assertEqual(len(res["options"]), 2)

    def test_reading_order_is_rows_then_columns(self):
        units = [[{"text": "b", "rect": [500, 100, 10, 10]}],
                 [{"text": "c", "rect": [0, 400, 10, 10]}],
                 [{"text": "a", "rect": [0, 110, 10, 10]}]]
        self.assertEqual([u[0]["text"] for u in R.reading_order(units)],
                         ["a", "b", "c"])

    def test_legend_names_positions(self):
        cands, _gt, _p = uia_page("home_dark")
        res = R.resolve("click the beast video", cands)
        opts = res.get("options") or [res["target"]]
        text = R.legend(opts, cands)
        self.assertIn("'", text)


class NeverRaisesTests(unittest.TestCase):
    def test_garbage_in(self):
        self.assertEqual(R.resolve(None, None), {"status": "none"})
        self.assertEqual(R.resolve("x", [{"text": None}])["status"], "none")
        self.assertEqual(R.legend(None), "")
        self.assertEqual(R.describe_position({}, None), "")


if __name__ == "__main__":
    unittest.main()
