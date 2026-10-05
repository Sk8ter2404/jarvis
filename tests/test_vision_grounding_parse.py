"""core/vision_grounding.py - image size for the local model, and reading
its answer back (2026-10-05).

The answers are the REAL shape ask_vision returns: every local answer is
prefixed "[local-vision] " (the green-by-mock lesson: a test that mocked
ask_vision without that prefix hid a drift that made every local YES parse
as NO).

    python -m unittest tests.test_vision_grounding_parse
"""
from __future__ import annotations

import unittest

from core import vision_grounding as V

TAG = "[local-vision] "


class ParseBox2dTests(unittest.TestCase):
    def test_the_real_prefix_is_stripped_before_any_rule(self):
        r = V.parse_reply(TAG + '{"box_2d": [100, 200, 300, 400]}', "box2d")
        self.assertEqual(r, {"kind": "box", "box": (100.0, 200.0, 300.0, 400.0)})

    def test_code_fence(self):
        r = V.parse_reply(TAG + '```json\n{"box_2d": [10, 20, 30, 40]}\n```')
        self.assertEqual(r["kind"], "box")

    def test_json_list_of_objects(self):
        r = V.parse_reply(TAG + '[{"box_2d": [10, 20, 30, 40], "label": "x"}]')
        self.assertEqual(r["box"], (10.0, 20.0, 30.0, 40.0))

    def test_box_2d_in_prose(self):
        r = V.parse_reply(TAG + 'Sure: "box_2d": [500, 100, 560, 400] is it')
        self.assertEqual(r["box"], (500.0, 100.0, 560.0, 400.0))

    def test_bare_four_numbers_are_y_first_0_1000(self):
        r = V.parse_reply(TAG + "[120, 40, 180, 300]")
        self.assertEqual(r["box"], (120.0, 40.0, 180.0, 300.0))
        x, y, w, h = V.box_to_rect(r["box"], 1000, 500)
        self.assertEqual((round(x), round(y), round(w), round(h)),
                         (40, 60, 260, 30))

    def test_none_and_not_found(self):
        for a in ("NONE", "None.", "NOT_FOUND", "not found", TAG + "NONE"):
            self.assertEqual(V.parse_reply(a)["kind"], "none", a)

    def test_several_boxes_are_ambiguous(self):
        r = V.parse_reply(TAG + '[{"box_2d":[1,2,30,40]},{"box_2d":[50,60,70,80]}]')
        self.assertEqual(r["kind"], "ambiguous")

    def test_invalid_boxes(self):
        for a in ("[300, 200, 100, 400]",        # y1 > y2
                  "[100, 400, 300, 200]",        # x1 > x2
                  "[100, 200, 300, 1400]",       # outside 0-1000
                  "[0, 0, 900, 900]"):           # > 60% of the image
            self.assertEqual(V.parse_reply(TAG + a)["kind"], "invalid", a)

    def test_a_pixel_pair_is_not_a_box(self):
        r = V.parse_reply(TAG + "432,718", "box2d")
        self.assertNotEqual(r["kind"], "point")
        self.assertNotEqual(r["kind"], "box")


class ParsePixelTests(unittest.TestCase):
    def test_pixel_pair_with_the_real_prefix(self):
        # The live trap: with the 15-char prefix the old <= 32-char rule was
        # the only path to a bare pair.
        self.assertEqual(V.parse_reply(TAG + "432,718", "pixel"),
                         {"kind": "point", "xy": (432, 718)})
        self.assertEqual(V.parse_reply(TAG + "(432, 718).", "pixel")["xy"],
                         (432, 718))

    def test_prose_pairs_are_refused(self):
        r = V.parse_reply(TAG + "I can see 2 buttons, 3 tabs and a long list "
                          "of other things on this page", "pixel")
        self.assertEqual(r["kind"], "invalid")

    def test_short_trailing_pair(self):
        self.assertEqual(V.parse_reply("at 12,34", "pixel")["xy"], (12, 34))


class ParseMarkTests(unittest.TestCase):
    def test_a_number(self):
        for a in ("7", TAG + "7", "Number 7.", "#7", TAG + "**7**"):
            self.assertEqual(V.parse_reply(a, "mark"), {"kind": "mark",
                                                        "mark": 7}, a)

    def test_none(self):
        self.assertEqual(V.parse_reply(TAG + "NONE", "mark")["kind"], "none")

    def test_two_numbers_are_ambiguous(self):
        self.assertEqual(V.parse_reply("3 or 5", "mark")["kind"], "ambiguous")


class FitTests(unittest.TestCase):
    def test_token_formula(self):
        self.assertEqual(V.est_tokens(1024, 576), 22 * 12)
        self.assertEqual(V.est_tokens(1920, 1080), 40 * 23)

    def test_a_2560x1440_monitor_goes_at_0_75(self):
        w, h, s = V.fit_size(2560, 1440, 960)
        self.assertLessEqual(V.est_tokens(w, h), 960)
        # The largest scale under the cap: ~0.75-0.77 (1920-1966 px wide).
        self.assertGreaterEqual(s, 0.74)
        self.assertLessEqual(s, 0.78)

    def test_other_shapes_stay_under_the_cap(self):
        for size in ((1920, 1200), (3840, 1080), (2560, 2880), (7680, 2880)):
            w, h, _s = V.fit_size(*size, 960)
            self.assertLessEqual(V.est_tokens(w, h), 960, size)

    def test_a_small_crop_goes_native(self):
        self.assertEqual(V.fit_size(800, 600, 960), (800, 600, 1.0))

    def test_fit_for_vlm_resizes_a_pil_image(self):
        from PIL import Image
        img, scale = V.fit_for_vlm(Image.new("RGB", (2560, 1440)), 960)
        self.assertLessEqual(V.est_tokens(*img.size), 960)
        self.assertLess(scale, 1.0)

    def test_the_cap_follows_the_setting(self):
        from unittest import mock
        from core import config as cfg
        with mock.patch.object(cfg, "VLM_MAX_IMAGE_TOKENS", 300, create=True):
            w, h, _s = V.fit_size(2560, 1440)
        self.assertLessEqual(V.est_tokens(w, h), 300)


class MarksTests(unittest.TestCase):
    def test_numbers_are_drawn_and_capped(self):
        from PIL import Image
        img = Image.new("RGB", (800, 600), (30, 30, 30))
        rects = [(10 + i * 20, 40, 15, 15) for i in range(35)]
        out, numbers = V.draw_marks(img, rects, max_marks=30)
        self.assertEqual(numbers[:3], [1, 2, 3])
        self.assertEqual(numbers[30:], [None] * 5)
        self.assertNotEqual(out.tobytes(), img.tobytes(),
                            "no mark was drawn")

    def test_prompts_ask_for_the_right_format(self):
        self.assertIn("number only", V.prompt_mark("the red button"))
        self.assertIn("box_2d", V.prompt_box2d("the red button"))
        self.assertIn("0-1000", V.prompt_box2d("x"))


if __name__ == "__main__":
    unittest.main()
