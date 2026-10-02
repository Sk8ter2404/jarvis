"""Pure-geometry tests for ``hud/globe_geometry.py`` (the holographic globe).

Everything the globe HUD computes — the orthographic projection and its
inverse, back-face culling (including arcs lifted above the surface), the
horizon-split screen runs, great-circle arcs and the rotate-to-pin easing —
lives in that one tkinter-free module, so it is tested here with no display
and no window. Loaded by file path with ``importlib`` like the other hud/
tests (hud/ is not a package).

stdlib ``unittest`` only (no pytest).
"""
from __future__ import annotations

import importlib.util
import math
import os
import unittest

_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud")


def _load():
    spec = importlib.util.spec_from_file_location(
        "globe_geometry_under_test", os.path.join(_HUD_DIR, "globe_geometry.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


geo = _load()

_VIEWS = ((0.0, 0.0), (18.0, 40.0), (-35.0, -120.0), (60.0, 179.0),
          (-70.0, -179.5))


def _norm(p):
    return math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)


def _lon_close(a, b, tol=1e-6):
    return abs(geo.wrap_lon(a - b)) < tol


class ProjectionTests(unittest.TestCase):
    def test_latlon_xyz_round_trip_on_the_unit_sphere(self):
        for lat in range(-80, 81, 20):
            for lon in range(-180, 180, 30):
                p = geo.latlon_to_xyz(lat, lon)
                self.assertAlmostEqual(_norm(p), 1.0, places=12)
                lat2, lon2 = geo.xyz_to_latlon(*p)
                self.assertAlmostEqual(lat2, lat, places=9)
                self.assertTrue(_lon_close(lon2, lon), (lon, lon2))

    def test_view_centre_projects_to_the_disc_centre(self):
        for vlat, vlon in _VIEWS:
            e, n, visible = geo.project(vlat, vlon, vlat, vlon)
            self.assertAlmostEqual(e, 0.0, places=12)
            self.assertAlmostEqual(n, 0.0, places=12)
            self.assertTrue(visible)

    def test_projection_round_trip_for_every_visible_point(self):
        checked = 0
        for vlat, vlon in _VIEWS:
            for lat in range(-85, 86, 5):
                for lon in range(-180, 180, 10):
                    e, n, visible = geo.project(lat, lon, vlat, vlon)
                    self.assertLessEqual(e * e + n * n, 1.0 + 1e-9)
                    if not visible:
                        continue
                    if geo.angular_distance(lat, lon, vlat, vlon) > 89.0:
                        continue   # on the limb the inverse is ill-conditioned
                    lat2, lon2 = geo.unproject(e, n, vlat, vlon)
                    self.assertAlmostEqual(lat2, lat, places=6,
                                           msg=(lat, lon, vlat, vlon))
                    self.assertTrue(_lon_close(lon2, lon, 1e-5),
                                    (lat, lon, lon2, vlat, vlon))
                    checked += 1
        self.assertGreater(checked, 2000)   # the sweep is not blind

    def test_unproject_outside_the_disc_is_none(self):
        self.assertIsNone(geo.unproject(0.8, 0.8, 10.0, 20.0))

    def test_north_is_up_and_east_is_right(self):
        e, n, _ = geo.project(10.0, 0.0, 0.0, 0.0)
        self.assertGreater(n, 0.0)
        self.assertAlmostEqual(e, 0.0, places=12)
        e, n, _ = geo.project(0.0, 10.0, 0.0, 0.0)
        self.assertGreater(e, 0.0)
        self.assertAlmostEqual(n, 0.0, places=12)


class CullingTests(unittest.TestCase):
    def test_far_hemisphere_is_culled(self):
        for vlat, vlon in _VIEWS:
            anti_lat, anti_lon = -vlat, geo.wrap_lon(vlon + 180.0)
            self.assertFalse(geo.project(anti_lat, anti_lon, vlat, vlon)[2])
            basis = geo.view_basis(vlat, vlon)
            self.assertFalse(geo.is_visible(
                geo.latlon_to_xyz(anti_lat, anti_lon), basis))
            self.assertIsNone(geo.to_screen(
                geo.latlon_to_xyz(anti_lat, anti_lon), basis, 0, 0, 100))

    def test_visibility_agrees_with_the_hemisphere(self):
        for vlat, vlon in _VIEWS:
            basis = geo.view_basis(vlat, vlon)
            for lat in range(-80, 81, 20):
                for lon in range(-180, 180, 20):
                    dist = geo.angular_distance(lat, lon, vlat, vlon)
                    if abs(dist - 90.0) < 0.5:
                        continue              # on the limb: either is fine
                    p = geo.latlon_to_xyz(lat, lon)
                    self.assertEqual(geo.is_visible(p, basis), dist < 90.0,
                                     (lat, lon, vlat, vlon, dist))

    def test_a_lifted_point_shows_past_the_limb_but_not_through_the_globe(self):
        basis = geo.view_basis(0.0, 0.0)
        # Just behind the limb on the far side, raised 20 %: outside the
        # disc's silhouette, so it is seen over the edge of the globe.
        over_edge = geo.latlon_to_xyz(0.0, 100.0, radius=1.2)
        self.assertTrue(geo.is_visible(over_edge, basis))
        # Straight behind the globe, also raised: hidden by the sphere.
        behind = geo.latlon_to_xyz(0.0, 180.0, radius=1.2)
        self.assertFalse(geo.is_visible(behind, basis))


class ProjectRunsTests(unittest.TestCase):
    CX, CY, R = 400.0, 300.0, 200.0

    def _runs(self, lines, vlat=0.0, vlon=0.0):
        return geo.project_runs(lines, geo.view_basis(vlat, vlon),
                                self.CX, self.CY, self.R)

    def _points(self, run):
        return list(zip(run[0::2], run[1::2]))

    def test_a_front_line_is_one_run_with_every_point(self):
        line = [geo.latlon_to_xyz(0.0, lon) for lon in range(-40, 41, 5)]
        runs = self._runs([line])
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]), 2 * len(line))
        # Screen y runs down: the view centre lands on (cx, cy).
        mid = self._points(runs[0])[len(line) // 2]
        self.assertAlmostEqual(mid[0], self.CX)
        self.assertAlmostEqual(mid[1], self.CY)

    def test_a_back_line_has_no_runs(self):
        line = [geo.latlon_to_xyz(0.0, lon) for lon in range(140, 221, 5)]
        self.assertEqual(self._runs([line]), [])

    def test_the_equator_is_clipped_at_the_horizon(self):
        line = [geo.latlon_to_xyz(0.0, lon) for lon in range(-180, 181, 7)]
        runs = self._runs([line])
        self.assertEqual(len(runs), 1)
        pts = self._points(runs[0])
        for x, y in pts:          # nothing from the far side leaks in
            self.assertLessEqual(math.hypot(x - self.CX, y - self.CY),
                                 self.R + 1e-6)
        # ...and both ends were interpolated onto the limb, not left short.
        for x, y in (pts[0], pts[-1]):
            self.assertAlmostEqual(math.hypot(x - self.CX, y - self.CY),
                                   self.R, delta=self.R * 0.01)

    def test_a_line_leaving_and_returning_splits_into_two_runs(self):
        # Along the equator from 60E round the back to 60W, then on to 0:
        # visible, hidden, visible again.
        line = [geo.latlon_to_xyz(0.0, lon) for lon in range(60, 361, 10)]
        self.assertEqual(len(self._runs([line])), 2)

    def test_graticule_line_counts(self):
        self.assertEqual(len(geo.meridians()), 360 // geo.GRATICULE_STEP_DEG)
        # Parallels every 15 degrees, poles excluded: -75 .. 75.
        self.assertEqual(len(geo.parallels()), 180 // geo.GRATICULE_STEP_DEG - 1)

    def test_coastline_decoding_reads_lon_before_lat(self):
        lines = geo.coastline_polylines([[139.7, 35.7, 0.0, 51.5], [1.0]])
        self.assertEqual(len(lines), 1)       # a one-number line is dropped
        lat, lon = geo.xyz_to_latlon(*lines[0][0])
        self.assertAlmostEqual(lat, 35.7)
        self.assertAlmostEqual(lon, 139.7)


class GreatCircleTests(unittest.TestCase):
    def test_arc_endpoints_are_exactly_the_pins(self):
        for (a, b) in (((51.5, -0.12), (40.75, -73.98)),
                       ((35.69, 139.75), (-33.87, 151.21)),
                       ((10.0, 170.0), (-10.0, -170.0))):
            for lift in (0.0, 0.1):
                pts = geo.great_circle_arc(a[0], a[1], b[0], b[1], lift=lift)
                self.assertEqual(pts[0], geo.latlon_to_xyz(*a))
                self.assertEqual(pts[-1], geo.latlon_to_xyz(*b))

    def test_arc_stays_on_the_sphere_and_on_the_great_circle(self):
        pts = geo.great_circle_arc(51.5, -0.12, 35.69, 139.75, step_deg=2.0)
        a, b = pts[0], pts[-1]
        normal = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0])
        for p in pts:
            self.assertAlmostEqual(_norm(p), 1.0, places=9)
            # In the plane through both endpoints and the centre.
            self.assertAlmostEqual(
                p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2], 0.0,
                places=9)
        total = geo.angular_distance(51.5, -0.12, 35.69, 139.75)
        for p, q in zip(pts, pts[1:]):
            step = math.degrees(math.acos(min(1.0, sum(x * y for x, y in zip(p, q)))))
            self.assertLessEqual(step, 2.0 + 1e-9)
        self.assertGreater(total, 80.0)   # London-Tokyo is ~86 degrees

    def test_lift_raises_the_middle_only(self):
        pts = geo.great_circle_arc(0.0, 0.0, 0.0, 90.0, step_deg=1.0, lift=0.1)
        self.assertAlmostEqual(_norm(pts[0]), 1.0)
        self.assertAlmostEqual(_norm(pts[-1]), 1.0)
        self.assertAlmostEqual(max(_norm(p) for p in pts), 1.1, places=3)

    def test_identical_and_antipodal_pins(self):
        same = geo.great_circle_arc(20.0, 30.0, 20.0, 30.0)
        self.assertEqual(len(same), 2)
        anti = geo.great_circle_arc(0.0, 0.0, 0.0, 180.0, step_deg=10.0)
        self.assertGreater(len(anti), 10)
        self.assertEqual(anti[0], geo.latlon_to_xyz(0.0, 0.0))
        self.assertEqual(anti[-1], geo.latlon_to_xyz(0.0, 180.0))
        for p in anti:
            self.assertAlmostEqual(_norm(p), 1.0, places=9)

    def test_angular_distance_known_pair(self):
        # London - New York: ~5570 km on a 6371 km sphere, ~50.1 degrees.
        d = geo.angular_distance(51.507, -0.128, 40.713, -74.006)
        self.assertAlmostEqual(d, 50.1, delta=0.3)


class EasingTests(unittest.TestCase):
    def test_ease_stays_in_bounds_and_hits_the_ends(self):
        self.assertEqual(geo.ease_in_out_cubic(0.0), 0.0)
        self.assertEqual(geo.ease_in_out_cubic(1.0), 1.0)
        self.assertAlmostEqual(geo.ease_in_out_cubic(0.5), 0.5)
        prev = -1.0
        for i in range(-50, 151):
            v = geo.ease_in_out_cubic(i / 100.0)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
            self.assertGreaterEqual(v, prev)      # monotonic
            prev = v

    def test_ease_starts_and_ends_slowly(self):
        self.assertLess(geo.ease_in_out_cubic(0.1), 0.1)
        self.assertGreater(geo.ease_in_out_cubic(0.9), 0.9)

    def test_tween_turns_the_short_way_round(self):
        start, end = (10.0, 170.0), (30.0, -170.0)
        self.assertEqual(geo.tween_view(start, end, 0.0), (10.0, 170.0))
        lat, lon = geo.tween_view(start, end, 1.0)
        self.assertAlmostEqual(lat, 30.0)
        self.assertTrue(_lon_close(lon, -170.0))
        _lat, mid = geo.tween_view(start, end, 0.5)
        self.assertTrue(_lon_close(mid, 180.0), mid)  # via the date line, not 0
        for t in (-1.0, 2.0):                          # clamped
            lat, _ = geo.tween_view(start, end, t)
            self.assertTrue(10.0 <= lat <= 30.0)

    def test_approach_never_overshoots(self):
        self.assertEqual(geo.approach(10.0, 18.0, 3.0), 13.0)
        self.assertEqual(geo.approach(17.0, 18.0, 3.0), 18.0)
        self.assertEqual(geo.approach(30.0, 18.0, 5.0), 25.0)

    def test_wrap_and_shortest_delta(self):
        self.assertEqual(geo.wrap_lon(190.0), -170.0)
        self.assertEqual(geo.wrap_lon(-180.0), -180.0)
        self.assertEqual(geo.shortest_lon_delta(170.0, -170.0), 20.0)
        self.assertEqual(geo.shortest_lon_delta(-170.0, 170.0), -20.0)

    def test_degrees_per_pixel_shrinks_with_a_bigger_globe(self):
        self.assertGreater(geo.degrees_per_pixel(100), geo.degrees_per_pixel(400))
        self.assertAlmostEqual(geo.degrees_per_pixel(400), math.degrees(1 / 400))


if __name__ == "__main__":
    unittest.main()
