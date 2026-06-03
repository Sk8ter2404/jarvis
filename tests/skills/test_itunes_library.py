"""Tests for skills.itunes_library — playlist/library voice control over iTunes COM.

All iTunes COM access is faked; itunes_bridge.get_client is patched so no real
COM or iTunes process is touched. Playlist names here are deliberately generic
(no personal data) so the PII pre-commit gate stays clean.
"""
from __future__ import annotations

import unittest
from unittest import mock

from skills import itunes_library as M


# ─── fake iTunes COM object graph ──────────────────────────────────────────

class _Countable:
    def __init__(self, count):
        self.Count = count


class _FakePlaylist:
    def __init__(self, name, kind=2, special=0, tracks=10):
        self.Name = name
        self.Kind = kind
        self.SpecialKind = special
        self._tracks = tracks
        self.Shuffle = False
        self.played = False

    @property
    def Tracks(self):
        return _Countable(self._tracks)

    def PlayFirstTrack(self):
        self.played = True


class _ExplodingPlaylist(_FakePlaylist):
    def PlayFirstTrack(self):
        raise RuntimeError("COM boom")


class _Collection:
    def __init__(self, items):
        self._items = items

    @property
    def Count(self):
        return len(self._items)

    def Item(self, i):  # 1-based, like iTunes COM
        return self._items[i - 1]


class _Source:
    def __init__(self, playlists):
        self.Playlists = _Collection(playlists)


class _FakeApp:
    def __init__(self, playlists, library=None):
        self.LibrarySource = _Source(playlists)
        self.LibraryPlaylist = library or _FakePlaylist("Library", kind=1, tracks=6349)


def _patch_client(app=None, err=None):
    return mock.patch.object(M.itunes_bridge, "get_client", return_value=(app, err))


def _library():
    """Library + two auto special lists + three real user playlists."""
    return [
        _FakePlaylist("Library", kind=1, tracks=6349),
        _FakePlaylist("Music", kind=2, special=4, tracks=6269),   # auto special
        _FakePlaylist("Movies", kind=2, special=3, tracks=51),    # auto special
        _FakePlaylist("Road Trip", kind=2, special=0, tracks=97),
        _FakePlaylist("90s Rock", kind=2, special=0, tracks=115),
        _FakePlaylist("Driver’s Picks", kind=2, special=0, tracks=48),  # curly apostrophe
    ]


# ─── _norm / matching ──────────────────────────────────────────────────────

class NormalizationTests(unittest.TestCase):
    def test_folds_smart_quotes_and_drops_apostrophe(self):
        self.assertEqual(M._norm("Driver’s  Picks"), "drivers picks")

    def test_collapses_whitespace_and_case(self):
        self.assertEqual(M._norm("  ROAD   trip "), "road trip")

    def test_strip_shuffle_leading_only(self):
        self.assertEqual(M._strip_shuffle("shuffle 90s rock"), ("90s rock", True))
        self.assertEqual(M._strip_shuffle("Evening Shuffle"), ("Evening Shuffle", False))


# ─── _user_playlists filtering ─────────────────────────────────────────────

class UserPlaylistFilterTests(unittest.TestCase):
    def test_excludes_library_and_special(self):
        names = [n for _, n in M._user_playlists(_FakeApp(_library()))]
        self.assertEqual(names, ["Road Trip", "90s Rock", "Driver’s Picks"])

    def test_bad_item_is_skipped(self):
        class _Boom:
            @property
            def Kind(self):
                raise RuntimeError("x")
        good = _FakePlaylist("Keep", kind=2, special=0)
        names = [n for _, n in M._user_playlists(_FakeApp([_Boom(), good]))]
        self.assertEqual(names, ["Keep"])


# ─── play_playlist ─────────────────────────────────────────────────────────

class PlayPlaylistTests(unittest.TestCase):
    def setUp(self):
        self.trip = _FakePlaylist("Road Trip", kind=2, special=0)
        self.rock = _FakePlaylist("90s Rock", kind=2, special=0)
        self.smart = _FakePlaylist("Driver’s Picks", kind=2, special=0)
        self.app = _FakeApp([
            _FakePlaylist("Library", kind=1),
            _FakePlaylist("Music", kind=2, special=4),
            self.trip, self.rock, self.smart,
        ])

    def test_empty_arg_prompts(self):
        with _patch_client(self.app):
            self.assertIn("Which playlist", M.play_playlist(""))

    def test_exact_match_plays(self):
        with _patch_client(self.app):
            out = M.play_playlist("road trip")
        self.assertTrue(self.trip.played)
        self.assertIn("Road Trip", out)

    def test_substring_match(self):
        with _patch_client(self.app):
            M.play_playlist("rock")
        self.assertTrue(self.rock.played)

    def test_smart_apostrophe_match(self):
        with _patch_client(self.app):
            M.play_playlist("drivers picks")
        self.assertTrue(self.smart.played)

    def test_shuffle_prefix_sets_shuffle(self):
        with _patch_client(self.app):
            out = M.play_playlist("shuffle 90s rock")
        self.assertTrue(self.rock.played)
        self.assertTrue(self.rock.Shuffle)
        self.assertIn("shuffled", out)

    def test_not_found(self):
        with _patch_client(self.app):
            out = M.play_playlist("nonexistent playlist")
        self.assertIn("couldn't find", out.lower())

    def test_itunes_unreachable_returns_bridge_error(self):
        with _patch_client(None, "iTunes isn't running, sir."):
            out = M.play_playlist("road trip")
        self.assertIn("iTunes isn't running", out)

    def test_play_error_is_graceful(self):
        boom = _ExplodingPlaylist("Boom", kind=2, special=0)
        with _patch_client(_FakeApp([boom])):
            out = M.play_playlist("boom")
        self.assertIn("couldn't start", out.lower())

    def test_read_error_is_graceful(self):
        class _BadColl:
            @property
            def Count(self):
                raise RuntimeError("read boom")

        class _BadSource:
            Playlists = _BadColl()

        class _BadApp:
            LibrarySource = _BadSource()

        with _patch_client(_BadApp()):
            out = M.play_playlist("road trip")
        self.assertIn("couldn't read", out.lower())


# ─── list_playlists ────────────────────────────────────────────────────────

class ListPlaylistsTests(unittest.TestCase):
    def test_lists_only_user_playlists(self):
        with _patch_client(_FakeApp(_library())):
            out = M.list_playlists()
        self.assertIn("3 playlists", out)
        self.assertIn("Road Trip", out)
        self.assertIn("90s Rock", out)
        self.assertNotIn("Movies", out)
        self.assertNotIn("Music,", out)

    def test_empty_library(self):
        app = _FakeApp([_FakePlaylist("Library", kind=1),
                        _FakePlaylist("Music", kind=2, special=4)])
        with _patch_client(app):
            self.assertIn("don't see any", M.list_playlists().lower())

    def test_truncates_over_twenty(self):
        many = [_FakePlaylist(f"PL {i}", kind=2, special=0) for i in range(30)]
        with _patch_client(_FakeApp(many)):
            out = M.list_playlists()
        self.assertIn("30 playlists", out)
        self.assertIn("10 more", out)

    def test_unreachable(self):
        with _patch_client(None, "iTunes isn't running, sir."):
            self.assertIn("iTunes isn't running", M.list_playlists())


# ─── shuffle_library ───────────────────────────────────────────────────────

class ShuffleLibraryTests(unittest.TestCase):
    def test_prefers_music_playlist(self):
        music = _FakePlaylist("Music", kind=2, special=4, tracks=6269)
        app = _FakeApp([_FakePlaylist("Library", kind=1), music,
                        _FakePlaylist("Road Trip", kind=2, special=0)])
        with _patch_client(app):
            out = M.shuffle_library()
        self.assertTrue(music.played)
        self.assertTrue(music.Shuffle)
        self.assertIn("shuffling", out.lower())

    def test_falls_back_to_library_playlist(self):
        lib = _FakePlaylist("Library", kind=1, tracks=6349)
        app = _FakeApp([_FakePlaylist("Road Trip", kind=2, special=0)], library=lib)
        with _patch_client(app):
            M.shuffle_library()
        self.assertTrue(lib.played)

    def test_unreachable(self):
        with _patch_client(None, "nope, sir."):
            self.assertIn("nope", M.shuffle_library())


# ─── register ──────────────────────────────────────────────────────────────

class RegisterTests(unittest.TestCase):
    def test_registers_three_callables(self):
        actions = {}
        M.register(actions)
        self.assertEqual(set(actions),
                         {"play_playlist", "list_playlists", "shuffle_library"})
        for fn in actions.values():
            self.assertTrue(callable(fn))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
