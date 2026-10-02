"""core/monitor_geometry.py - which monitor a point is on, where a point in a
screenshot lands on the desktop, and the ONE label scheme for several monitor
images (S4, 2026-10-02).

Live (paraphrased): a page JARVIS opened on the MIDDLE monitor was named
"Image #4 (TOP monitor)" by local vision in one round and "the MIDDLE
monitor" in the next, and a description click landed at (-2325, 1165) - on
the LEFT monitor. The layout below is the shape of the owner's rig (four
2560x1440 monitors: left, middle at the origin, right, and one above the
middle), so the virtual desktop is 7680x2880 at (-2560, -1440).

    python -m unittest tests.test_monitor_geometry
"""
from __future__ import annotations

import unittest

from core import monitor_geometry as G

MONS = {
    "left":   (-2560, 0, 2560, 1440),
    "middle": (0, 0, 2560, 1440),
    "right":  (2560, 0, 2560, 1440),
    "top":    (0, -1440, 2560, 1440),
}


class MonitorAtTests(unittest.TestCase):
    def test_each_monitor_and_its_edges(self):
        for (x, y), want in (((-2560, 0), "left"), ((-1, 1439), "left"),
                             ((0, 0), "middle"), ((2559, 1439), "middle"),
                             ((2560, 0), "right"), ((5119, 700), "right"),
                             ((0, -1440), "top"), ((1280, -1), "top"),
                             ((-1, -1), None),          # the empty top-left corner
                             ((3000, -500), None),      # the empty top-right corner
                             ((5120, 0), None), ((0, 1440), None)):
            with self.subTest(point=(x, y)):
                self.assertEqual(G.monitor_at(x, y, MONS), want)

    def test_a_window_is_on_the_monitor_of_its_centre(self):
        # A maximised window overhangs its monitor by a few pixels.
        self.assertEqual(G.monitor_for_rect(-8, -8, 2576, 1456, MONS), "middle")
        self.assertEqual(G.monitor_for_rect(-2568, -8, 2576, 1456, MONS), "left")
        self.assertEqual(G.monitor_for_rect(100, -1400, 800, 600, MONS), "top")

    def test_virtual_bounds(self):
        self.assertEqual(G.virtual_bounds(MONS), (-2560, -1440, 7680, 2880))
        self.assertEqual(G.virtual_bounds({}), (0, 0, 2560, 1440))
        self.assertEqual(G.virtual_bounds({"bad": (1, 2)}), (0, 0, 2560, 1440))


class ImagePointToScreenTests(unittest.TestCase):
    """A point in a screenshot -> absolute desktop coordinates, for the four
    monitors, including the negative-x LEFT one and the negative-y TOP one."""

    def test_a_full_resolution_shot_of_one_monitor_maps_one_to_one(self):
        for name, (x, y, w, h) in MONS.items():
            with self.subTest(monitor=name):
                got = G.image_point_to_screen(100, 200, (w, h), (x, y, w, h))
                self.assertEqual(got, (x + 100, y + 200))
                self.assertEqual(G.monitor_at(*got, MONS), name)

    def test_the_left_monitor_lands_at_negative_x(self):
        # The centre of the LEFT monitor's own (downscaled) 1568x882 shot.
        got = G.image_point_to_screen(784, 441, (1568, 882), MONS["left"])
        self.assertEqual(got, (-1280, 720))
        self.assertEqual(G.monitor_at(*got, MONS), "left")

    def test_the_whole_desktop_shot_maps_back_onto_the_right_monitors(self):
        vb = G.virtual_bounds(MONS)                      # 7680x2880
        img = (1568, 588)                                 # its 1568-px shot
        cases = {
            "left":   (160, 441),     # x 0..522, y 294..588 in the image
            "middle": (784, 441),
            "right":  (1400, 441),
            "top":    (784, 100),
        }
        for name, (px, py) in cases.items():
            with self.subTest(monitor=name):
                got = G.image_point_to_screen(px, py, img, vb)
                self.assertEqual(G.monitor_at(*got, MONS), name)
        # The live click's spot is on the LEFT monitor: a click that should
        # have gone to the page on the MIDDLE monitor cannot land there once
        # it is pinned to the middle monitor's own shot.
        self.assertEqual(G.monitor_at(-2325, 1165, MONS), "left")
        for px in (0, 1567):
            got = G.image_point_to_screen(px, 881, (1568, 882), MONS["middle"])
            self.assertEqual(G.monitor_at(*got, MONS), "middle")

    def test_dpi_scaled_native_shot(self):
        # 150 % scaling: the native grab is 3840x2160 for a 2560x1440 logical
        # monitor, so the native offset is scaled down before the origin.
        got = G.image_point_to_screen(1920, 1080, (3840, 2160), MONS["left"])
        self.assertEqual(got, (-1280, 720))

    def test_two_pass_scaling(self):
        self.assertEqual(G.scale_point(784, 441, (1568, 882), (2560, 1440)),
                         (1280, 720))
        self.assertEqual(G.scale_point(5, 5, (0, 0), (10, 10)), (5, 5))


class LabelTests(unittest.TestCase):
    NAMES = ["left", "middle", "right", "top"]

    def test_one_label_scheme(self):
        self.assertEqual(G.monitor_image_labels(self.NAMES),
                         ["Image 1 = LEFT monitor", "Image 2 = MIDDLE monitor",
                          "Image 3 = RIGHT monitor", "Image 4 = TOP monitor"])
        intro = G.multi_monitor_intro(self.NAMES)
        for label in G.monitor_image_labels(self.NAMES):
            self.assertIn(label, intro)
        self.assertIn("by its NAME", intro)

    def test_image_numbers_become_the_monitor_they_are(self):
        for answer, want in (
                ("The page is on the **TOP monitor** (Image #4).",
                 "The page is on the **TOP monitor** (the TOP monitor)."),
                ("It is on Image #2 (MIDDLE).", "It is on the MIDDLE monitor."),
                ("Shown in Image 2.", "Shown in the MIDDLE monitor."),
                # The model's own name for image 4 was wrong: the number wins.
                ("It is on Image #4 (MIDDLE monitor).", "It is on the TOP monitor."),
                ("Image #9 is blank.", "Image #9 is blank."),
                ("Nothing to rename.", "Nothing to rename.")):
            with self.subTest(answer=answer):
                self.assertEqual(G.canonical_monitor_refs(answer, self.NAMES), want)

    def test_monitor_named_in_words(self):
        for text, want in (("click it on the left monitor", "left"),
                           ("the main screen", "middle"),
                           ("my primary display", "middle"),
                           ("the TOP monitor", "top"),
                           ("open it", None), ("", None)):
            with self.subTest(text=text):
                self.assertEqual(G.monitor_named_in(text, MONS), want)


if __name__ == "__main__":
    unittest.main()
