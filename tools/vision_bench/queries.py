"""tools/vision_bench/queries.py - the screen-vision bench's query sets.

Each row is (page, query, expected):
  expected  "v01" / "h03" / "f04"  - a video id: the click must land on that
                                     video's title or thumbnail;
            ("kind", "<data-id>")  - a non-video control (chip, side link,
                                     button);
            ("amb", ("v01", "v12")) - must ASK, naming exactly these;
            None                   - nothing like it is on the page: must
                                     not pick anything.

  DEV       - the research dev set (2026-10-05), used while building.
  HOLDOUT1  - the research held-out page. Written before the research
              resolver ran, but the shipped resolver was tuned AFTER seeing
              its misses, so it is reported as contaminated.
  FRESH2    - written for this build before anything was run on it
              (tools/vision_bench/pages.FRESH2_QUERIES) - the honest number.
"""
from tools.vision_bench.pages import FRESH2_QUERIES

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
    ("home2beast_dark", "click that MrBeast video", ("amb", ("v01", "v12"))),
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

HOLDOUT1 = [
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

FRESH2 = [("fresh2_light", q, e) for q, e in FRESH2_QUERIES]

SETS = {"dev": DEV, "holdout1": HOLDOUT1, "fresh2": FRESH2}


def expected_texts(expected, meta, page="") -> list:
    """The label text(s) a right answer carries (for text-judged runs)."""
    if expected is None:
        return []
    if isinstance(expected, str):
        if expected == "w00":
            return ["Eat Everything In This Store, Win $10,000"]
        keys = (("two_beasts", "videos") if page.startswith("home2beast")
                else ("videos", "two_beasts"))
        for key in keys + ("holdout", "fresh2"):
            for row in meta.get(key, ()):
                if row[0] == expected:
                    return [row[1]]
        return []
    kind, ident = expected[0], expected[1]
    if kind == "amb":
        out = []
        for vid in ident:
            out += expected_texts(vid, meta, page)
        return out
    fixed = {"subscribe": "Subscribe", "share": "Share",
             "create-key": "Create key", "other": "Use another account"}
    if ident in fixed:
        return [fixed[ident]]
    return [ident.split("-", 1)[1]] if "-" in ident else [ident]


def _squash(s) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def label_matches(label, texts) -> bool:
    """A label names one of ``texts``: equal, contains it, or (an OCR line
    of a wrapped title) is a long enough piece of it."""
    lab = _squash(label)
    if not lab:
        return False
    for t in texts:
        want = _squash(t)
        if not want:
            continue
        if lab == want or want in lab or (len(lab) >= 8 and lab in want):
            return True
    return False


def placed_right(expected, rect, origin) -> bool:
    """Side links live in the 240 px sidebar, chips in the row under the
    top bar (the pages share "Music", "Gaming" and "Live" between both)."""
    if not (isinstance(expected, tuple) and expected[0] == "kind"):
        return True
    ident = expected[1]
    x = rect[0] + rect[2] / 2 - origin[0]
    y = rect[1] + rect[3] / 2 - origin[1]
    if ident.startswith("side-"):
        return x < 250
    if ident.startswith("chip-"):
        return x >= 240 and y < 130
    return True


def judge(outcome, label, rect, options, expected, page, meta,
          origin=(0.0, 0.0)) -> str:
    """'ok' | 'wrong' | 'declined'. ``outcome``: "ok" (a target was
    picked), "ambiguous" (asked, ``options`` = labels) or anything else
    (nothing picked)."""
    if expected is None:
        return "wrong" if outcome == "ok" else "ok"
    texts = expected_texts(expected, meta, page)
    if isinstance(expected, tuple) and expected[0] == "amb":
        if outcome == "ambiguous":
            hit = {t for t in texts for o in options if label_matches(o, [t])}
            return "ok" if len(hit) == len(texts) else "wrong"
        return "wrong" if outcome == "ok" else "declined"
    if outcome != "ok":
        return "declined"
    return ("ok" if label_matches(label, texts)
            and placed_right(expected, rect, origin) else "wrong")
