"""Sanity tests for the holographic globe's bundled data (hud/data/).

The globe ships its own map so nothing is downloaded at runtime:
  * hud/data/globe_coastline.json — Natural Earth 110m coastline, simplified
    to a point budget (<= 4000) the HUD can re-project every frame;
  * hud/data/world_cities.json — ~260 major cities and capitals (Natural
    Earth populated places) for globe_pin's lookup.
Both are public domain; the attribution travels inside each file.

These pin the budget, the coordinate ranges (a lat/lon swap or a sign slip
puts Sydney in the northern hemisphere — spot-checked below) and that the
globe's code never reaches for the network.

stdlib ``unittest`` only (no pytest).
"""
from __future__ import annotations

import ast
import json
import os
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "hud", "data")


def _load(name):
    with open(os.path.join(_DATA, name), "r", encoding="utf-8") as f:
        return json.load(f)


class CoastlineDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = _load("globe_coastline.json")
        cls.lines = cls.doc["lines"]

    def test_point_budget(self):
        total = sum(len(line) // 2 for line in self.lines)
        self.assertLessEqual(total, 4000)
        # Floor: a truncated or mis-decoded file would pass the budget blind.
        self.assertGreater(total, 2000)
        self.assertGreater(len(self.lines), 100)

    def test_every_line_is_lon_lat_pairs_in_range(self):
        for line in self.lines:
            self.assertEqual(len(line) % 2, 0)
            self.assertGreaterEqual(len(line), 4)
            for lon, lat in zip(line[0::2], line[1::2]):
                self.assertTrue(-180.0 <= lon <= 180.0, lon)
                self.assertTrue(-90.0 <= lat <= 90.0, lat)

    def test_it_reaches_both_poles_and_the_date_line(self):
        lats = [v for line in self.lines for v in line[1::2]]
        lons = [v for line in self.lines for v in line[0::2]]
        self.assertLess(min(lats), -75.0)      # Antarctica
        self.assertGreater(max(lats), 75.0)    # the Arctic coast
        self.assertLess(min(lons), -170.0)
        self.assertGreater(max(lons), 170.0)

    def test_attribution(self):
        self.assertIn("Made with Natural Earth", self.doc["_source"])
        self.assertIn("public domain", self.doc["_source"])


class CitiesDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = _load("world_cities.json")
        cls.rows = cls.doc["cities"]

    def test_about_250_cities(self):
        self.assertGreaterEqual(len(self.rows), 200)
        self.assertLessEqual(len(self.rows), 300)

    def test_rows_are_name_country_lat_lon(self):
        for row in self.rows:
            name, country, lat, lon = row[:4]
            self.assertTrue(isinstance(name, str) and name.strip(), row)
            self.assertTrue(isinstance(country, str) and country.strip(), row)
            self.assertTrue(-90.0 <= lat <= 90.0, row)
            self.assertTrue(-180.0 <= lon <= 180.0, row)
            if len(row) > 4:
                self.assertTrue(all(isinstance(a, str) and a for a in row[4]),
                                row)
            self.assertLessEqual(len(row), 5)

    def test_names_are_unique(self):
        names = [row[0].casefold() for row in self.rows]
        self.assertEqual(len(names), len(set(names)))

    def test_hemispheres_spot_check(self):
        by_name = {row[0]: row for row in self.rows}
        for name, lat_sign, lon_sign in (("Tokyo", 1, 1), ("London", 1, -1),
                                         ("New York", 1, -1),
                                         ("Sydney", -1, 1),
                                         ("São Paulo", -1, -1)):
            with self.subTest(name=name):
                _n, _c, lat, lon = by_name[name][:4]
                self.assertEqual(lat > 0, lat_sign > 0)
                self.assertEqual(lon > 0, lon_sign > 0)
        _n, _c, lat, lon = by_name["Tokyo"][:4]
        self.assertAlmostEqual(lat, 35.7, delta=0.3)
        self.assertAlmostEqual(lon, 139.7, delta=0.3)

    def test_attribution(self):
        self.assertIn("Made with Natural Earth", self.doc["_source"])
        self.assertIn("public domain", self.doc["_source"])


class NoRuntimeDownloadTests(unittest.TestCase):
    """The globe draws only from hud/data/: none of its code may import a
    network module."""

    _FILES = ("hud/globe_hud.py", "hud/globe_geometry.py", "skills/globe.py")
    _NETWORK = {"urllib", "requests", "http", "socket", "urllib3", "httpx",
                "aiohttp", "ftplib"}

    def test_no_network_imports(self):
        for rel in self._FILES:
            with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    roots = {(node.module or "").split(".")[0]}
                else:
                    continue
                with self.subTest(file=rel, line=node.lineno):
                    self.assertFalse(roots & self._NETWORK, roots)


if __name__ == "__main__":
    unittest.main()
