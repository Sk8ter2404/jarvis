"""Tests for core.media_now_playing — the SMTC now-playing reader.

The real WinRT read is platform-only (pragma'd in the module). These tests pin
``_available = False`` so the cache / format / parse logic is deterministic on
any machine (CI / Linux or the Windows dev box) and never performs a real SMTC
read or spawns work.
"""
import unittest

import core.media_now_playing as m


class CleanAppTests(unittest.TestCase):
    def test_known_sources(self):
        self.assertEqual(m._clean_app("Chrome"), "Chrome")
        self.assertEqual(m._clean_app("Microsoft.MicrosoftEdge_8wekyb!App"), "Edge")
        self.assertEqual(m._clean_app("308046B0AF4A39CB"), "Firefox")
        self.assertEqual(
            m._clean_app("SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"), "Spotify")
        self.assertEqual(
            m._clean_app("AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"), "Apple Music")
        self.assertEqual(m._clean_app("iTunes"), "iTunes")
        self.assertEqual(m._clean_app("VLC media player"), "VLC")
        self.assertEqual(m._clean_app("Microsoft.ZuneMusic_8wekyb!App"), "Media Player")

    def test_blank_and_unknown(self):
        self.assertEqual(m._clean_app(""), "media")
        self.assertEqual(m._clean_app(None), "media")
        self.assertEqual(m._clean_app("SomeRandomApp"), "SomeRandomApp")


class _PinnedNoWinrt(unittest.TestCase):
    """Base: force the no-winrt path so injected snapshots flow through."""

    def setUp(self):
        self._save = (m._available, m._snapshot, m._last_read)
        m._available = False
        m._snapshot = None
        m._last_read = 0.0

    def tearDown(self):
        m._available, m._snapshot, m._last_read = self._save


class NowPlayingTextTests(_PinnedNoWinrt):
    def test_playing(self):
        m._snapshot = {"app": "Chrome", "title": "The Lady in My Life",
                       "artist": "Michael Jackson", "status": "playing", "playing": True}
        self.assertEqual(m.now_playing_text(), "The Lady in My Life — Michael Jackson")

    def test_paused_suffix(self):
        m._snapshot = {"title": "X", "artist": "Y", "status": "paused", "playing": False}
        self.assertEqual(m.now_playing_text(), "X — Y (paused)")

    def test_no_artist(self):
        m._snapshot = {"title": "Solo", "artist": "", "status": "playing", "playing": True}
        self.assertEqual(m.now_playing_text(), "Solo")

    def test_no_title_returns_none(self):
        m._snapshot = {"title": "", "artist": "Z", "status": "playing", "playing": True}
        self.assertIsNone(m.now_playing_text())

    def test_none_snapshot(self):
        m._snapshot = None
        self.assertIsNone(m.now_playing_text())
        self.assertIsNone(m.get_now_playing())

    def test_truncation(self):
        m._snapshot = {"title": "A" * 80, "artist": "B" * 80,
                       "status": "playing", "playing": True}
        out = m.now_playing_text(max_len=30)
        self.assertEqual(len(out), 30)
        self.assertTrue(out.endswith("…"))

    def test_get_returns_copy(self):
        m._snapshot = {"title": "T", "artist": "A", "status": "playing", "playing": True}
        got = m.get_now_playing()
        got["title"] = "MUTATED"
        self.assertEqual(m._snapshot["title"], "T")


class RefreshOnceTests(_PinnedNoWinrt):
    def test_reader_dict_sets_snapshot(self):
        snap = {"title": "T", "artist": "A", "status": "playing", "playing": True}
        out = m._refresh_once(reader=lambda: snap)
        self.assertEqual(out, snap)
        self.assertEqual(m._snapshot, snap)
        self.assertGreater(m._last_read, 0)

    def test_reader_raises_clears_snapshot(self):
        m._snapshot = {"title": "old"}

        def boom():
            raise RuntimeError("smtc down")

        self.assertIsNone(m._refresh_once(reader=boom))
        self.assertIsNone(m._snapshot)

    def test_reader_non_dict_ignored(self):
        self.assertIsNone(m._refresh_once(reader=lambda: "not a dict"))
        self.assertIsNone(m._snapshot)

    def test_reader_none(self):
        self.assertIsNone(m._refresh_once(reader=lambda: None))
        self.assertIsNone(m._snapshot)


class WinrtAvailableTests(unittest.TestCase):
    def test_returns_bool_and_caches(self):
        save = m._available
        try:
            m._available = None
            v = m._winrt_available()
            self.assertIsInstance(v, bool)
            self.assertEqual(m._winrt_available(), v)  # cached
        finally:
            m._available = save

    def test_pinned_value_used(self):
        save = m._available
        try:
            m._available = True
            self.assertTrue(m._winrt_available())
            m._available = False
            self.assertFalse(m._winrt_available())
        finally:
            m._available = save


if __name__ == "__main__":
    unittest.main()
