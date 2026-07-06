"""Tests for skills.itunes_library — playlist/library voice control over iTunes COM.

All iTunes COM access is faked; itunes_bridge.get_client is patched so no real
COM or iTunes process is touched. Playlist names here are deliberately generic
(no personal data) so the PII pre-commit gate stays clean.
"""
from __future__ import annotations

import sys
import threading
import types
import unittest
from unittest import mock

from audio import apple_music_keeper as K
from skills import itunes_library as M


def _keeper_thread_alive() -> bool:
    """True iff a live keep-alive watchdog daemon is running."""
    return any(t.name == K._THREAD_NAME and t.is_alive()
               for t in threading.enumerate())


def tearDownModule():  # noqa: N802 - unittest hook name
    # Safety net for the whole module: ``keep_music_open`` can reach the real
    # ``start_keeper`` (and thus spawn the non-terminating keep-alive daemon) if
    # a test ever fails to neuter it; that stray thread would otherwise leak into
    # later test files (notably tests/test_structural's compile sweep, which
    # flakes when a stray daemon survives into it). Stop + join any survivor.
    # Generous join: once _STOP is set the loop exits after one wait() wake, so
    # this is ample even under the heavy concurrent CI-sim load.
    K.stop_keeper(timeout=10.0)


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

    def test_itunes_unreachable_falls_back_to_apple_music(self):
        # COM is dead → get_client returns (None, err). play_playlist must NOT
        # surface the bridge error; it falls back to the browser apple_music
        # action (here mocked to return a streaming line).
        with _patch_client(None, "iTunes isn't running, sir."), \
                mock.patch.object(M, "_apple_music_fallback",
                                  return_value="Playing it on Apple Music, sir."):
            out = M.play_playlist("road trip")
        self.assertEqual(out, "Playing it on Apple Music, sir.")

    def test_fallback_query_includes_playlist_keyword(self):
        # The Apple Music fallback must search for a PLAYLIST, not a song of the
        # same name (2026-07-06 audit: the keyword was dropped, so a playlist
        # request played a random track).
        with _patch_client(None, "iTunes isn't running, sir."), \
                mock.patch.object(M, "_apple_music_fallback",
                                  return_value="ok, sir.") as fb:
            M.play_playlist("workout")
        self.assertIn("playlist", fb.call_args[0][0].lower())

    def test_itunes_unreachable_and_no_fallback_returns_bridge_error(self):
        # If even the browser fallback is unreachable (None), the bridge error
        # is the last resort.
        with _patch_client(None, "iTunes isn't running, sir."), \
                mock.patch.object(M, "_apple_music_fallback", return_value=None):
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


# ─── Apple Music fallback (iTunes-preferred, browser fallback) ──────────────

def _fake_monolith(apple_music_fn):
    """A stand-in __main__ module exposing an ACTIONS dict whose 'apple_music'
    entry is `apple_music_fn` — mirrors how the live monolith is reached."""
    bc = types.ModuleType("__main__")
    bc.ACTIONS = {"apple_music": apple_music_fn}
    return bc


class AppleMusicFallbackTests(unittest.TestCase):
    """play_playlist prefers local iTunes COM and only reaches the browser
    apple_music action when the playlist isn't owned (or iTunes is down)."""

    def setUp(self):
        self.trip = _FakePlaylist("Road Trip", kind=2, special=0)
        self.rock = _FakePlaylist("90s Rock", kind=2, special=0)
        self.app = _FakeApp([
            _FakePlaylist("Library", kind=1),
            _FakePlaylist("Music", kind=2, special=4),
            self.trip, self.rock,
        ])

    def test_found_playlist_does_not_call_fallback(self):
        fb = mock.Mock(return_value="should not be used")
        with _patch_client(self.app), \
                mock.patch.object(M, "_apple_music_fallback") as patched_fb:
            out = M.play_playlist("road trip")
        self.assertTrue(self.trip.played)
        self.assertIn("Road Trip", out)
        patched_fb.assert_not_called()
        fb.assert_not_called()

    def test_not_found_calls_fallback_and_returns_its_string(self):
        am = mock.Mock(return_value="Queueing Michael Jackson Essentials on Apple Music, sir.")
        with _patch_client(self.app), \
                mock.patch.dict(sys.modules, {"__main__": _fake_monolith(am)}):
            sys.modules.pop("bobert_companion", None)
            out = M.play_playlist("Michael Jackson Essentials")
        am.assert_called_once()
        self.assertEqual(out, "Queueing Michael Jackson Essentials on Apple Music, sir.")

    def test_itunes_unreachable_calls_fallback(self):
        am = mock.Mock(return_value="Playing it on Apple Music, sir.")
        with _patch_client(None, "iTunes isn't running, sir."), \
                mock.patch.dict(sys.modules, {"__main__": _fake_monolith(am)}):
            sys.modules.pop("bobert_companion", None)
            out = M.play_playlist("some streaming mix")
        am.assert_called_once()
        self.assertEqual(out, "Playing it on Apple Music, sir.")

    def test_fallback_query_carries_shuffle(self):
        am = mock.Mock(return_value="Shuffling it on Apple Music, sir.")
        with _patch_client(self.app), \
                mock.patch.dict(sys.modules, {"__main__": _fake_monolith(am)}):
            sys.modules.pop("bobert_companion", None)
            M.play_playlist("shuffle Curated Hype")
        am.assert_called_once()
        sent = am.call_args.args[0]
        self.assertIn("Curated Hype", sent)
        self.assertTrue(sent.endswith(" shuffle"))

    def test_fallback_unavailable_returns_not_found(self):
        # No ACTIONS at all on the monolith → fallback yields None → original
        # not-found message, no exception.
        empty = types.ModuleType("__main__")  # no ACTIONS attribute
        with _patch_client(self.app), \
                mock.patch.dict(sys.modules, {"__main__": empty}):
            sys.modules.pop("bobert_companion", None)
            out = M.play_playlist("nope nope")
        self.assertIn("couldn't find", out.lower())

    def test_fallback_swallows_action_exception(self):
        # apple_music action raises → fallback returns None → not-found message.
        boom = mock.Mock(side_effect=RuntimeError("vision broke"))
        with _patch_client(self.app), \
                mock.patch.dict(sys.modules, {"__main__": _fake_monolith(boom)}):
            sys.modules.pop("bobert_companion", None)
            out = M.play_playlist("still missing")
        self.assertIn("couldn't find", out.lower())

    def test_fallback_non_string_result_returns_not_found(self):
        # apple_music returns a non-string → treated as no result → not-found.
        weird = mock.Mock(return_value=object())
        with _patch_client(self.app), \
                mock.patch.dict(sys.modules, {"__main__": _fake_monolith(weird)}):
            sys.modules.pop("bobert_companion", None)
            out = M.play_playlist("missing too")
        self.assertIn("couldn't find", out.lower())


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

    def test_unreachable_routes_to_apple_music_app(self):
        # COM dead → no local library to enumerate → point the user to the new
        # Apple Music app instead of surfacing a COM error.
        with _patch_client(None, "iTunes isn't running, sir."):
            out = M.list_playlists()
        self.assertIn("Apple Music app", out)
        self.assertNotIn("iTunes isn't running", out)


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

    def test_unreachable_gives_honest_line_not_random_song(self):
        # COM dead → must NOT hand the bare word "shuffle" to the browser
        # apple_music action (that plays a random song titled "Shuffle",
        # 2026-07-06 audit). We give the honest guidance line and never invoke
        # the search fallback with a meaningless query.
        with _patch_client(None, "nope, sir."), \
                mock.patch.object(M, "_apple_music_fallback") as fb:
            out = M.shuffle_library()
        fb.assert_not_called()          # no bogus "shuffle" search
        self.assertIn("Apple Music app", out)
        self.assertNotIn("nope", out)


# ─── register ──────────────────────────────────────────────────────────────

class RegisterTests(unittest.TestCase):
    def test_registers_expected_callables(self):
        actions = {}
        M.register(actions)
        self.assertEqual(
            set(actions),
            {"play_playlist", "list_playlists", "shuffle_library",
             "keep_music_open", "stop_keeping_music_open"})
        for fn in actions.values():
            self.assertTrue(callable(fn))


# ─── keep Apple Music always open (voice toggle + persistence) ──────────────
# The two keeper actions flip core.config.APPLE_MUSIC_AUTOSTART / _KEEP_OPEN,
# persist via the hardened settings writer, and launch the app — all mocked.
# No real launching, no real settings file, no hardware.

class _FakeAppleMusic:
    """Stand-in for audio.apple_music_app: records launches, never shells out."""
    def __init__(self, running=False, launch_ok=True, launch_err=None):
        self._running = running
        self.launch_ok = launch_ok
        self.launch_err = launch_err
        self.launches = 0

    def is_running(self):
        return self._running

    def launch(self):
        self.launches += 1
        self._running = True
        return self.launch_ok, self.launch_err

    def now_playing(self):
        return None


class KeepMusicOpenTests(unittest.TestCase):
    def setUp(self):
        # Live config object the actions mutate; restore the originals after.
        import core.config as cfg
        self.cfg = cfg
        self._saved = (getattr(cfg, "APPLE_MUSIC_AUTOSTART", False),
                       getattr(cfg, "APPLE_MUSIC_KEEP_OPEN", False))
        cfg.APPLE_MUSIC_AUTOSTART = False
        cfg.APPLE_MUSIC_KEEP_OPEN = False
        # Capture persisted settings in-memory (never touch user_settings.json).
        self.saved_settings = {}

        def _save(d):
            self.saved_settings = dict(d)

        self._persist_patch = mock.patch.object(
            M, "_persist_setting",
            side_effect=lambda k, v: (self.saved_settings.__setitem__(k, v) or True))
        self._persist_patch.start()
        self.addCleanup(self._persist_patch.stop)
        # Never look like staging here (we want the launch branch to run).
        self._stage_patch = mock.patch.object(M, "_is_staging", return_value=False)
        self._stage_patch.start()
        self.addCleanup(self._stage_patch.stop)
        # DEFENSE-IN-DEPTH: neuter the REAL keeper's start_keeper for EVERY test
        # in this class. keep_music_open does a runtime ``from audio import
        # apple_music_keeper; _amk.start_keeper()`` — which binds the attribute on
        # the real module — so without this a test that reaches that line (e.g.
        # one that doesn't run its own _patch_keeper) would spawn the real,
        # non-terminating keep-alive daemon and LEAK it into later tests. The
        # tests that assert call-counts layer their own mock on top via
        # _patch_keeper(); a second patch is harmless.
        self._global_keeper_patch = mock.patch.object(
            K, "start_keeper", mock.MagicMock(return_value=True))
        self._global_keeper_patch.start()
        self.addCleanup(self._global_keeper_patch.stop)
        # Final safety net: after every test, stop + join any keep-alive watchdog
        # that somehow escaped, and assert none survives into the next test.
        self.addCleanup(self._assert_no_keeper_leak)

    def _assert_no_keeper_leak(self):
        K.stop_keeper(timeout=10.0)
        self.assertFalse(_keeper_thread_alive(),
                         "a real apple-music-keeper thread leaked out of the test")

    def tearDown(self):
        self.cfg.APPLE_MUSIC_AUTOSTART, self.cfg.APPLE_MUSIC_KEEP_OPEN = self._saved

    def _patch_app(self, app):
        return mock.patch.object(M, "_apple_music_app", return_value=app)

    def _patch_keeper(self):
        # keep_music_open does `from audio import apple_music_keeper`, which
        # resolves the SUBMODULE ATTRIBUTE on the audio package (not a sys.modules
        # dict entry once the real module is imported). So patch start_keeper on
        # the real module object — that's what the import binds, and it also stops
        # any real keeper daemon from starting (which would leak across tests).
        from audio import apple_music_keeper as real_keeper
        sk = mock.MagicMock(return_value=True)
        return mock.patch.object(real_keeper, "start_keeper", sk), sk

    def test_keep_open_sets_both_flags_and_persists(self):
        app = _FakeAppleMusic(running=False)
        km, _ = self._patch_keeper()
        with self._patch_app(app), km:
            out = M.keep_music_open("")
        self.assertTrue(self.cfg.APPLE_MUSIC_AUTOSTART)
        self.assertTrue(self.cfg.APPLE_MUSIC_KEEP_OPEN)
        self.assertTrue(self.saved_settings.get("APPLE_MUSIC_AUTOSTART"))
        self.assertTrue(self.saved_settings.get("APPLE_MUSIC_KEEP_OPEN"))
        # Honest messaging: tray + mini-player reality, no false mini-player claim.
        self.assertIn("tray", out.lower())
        self.assertIn("mini-player", out.lower())

    def test_keep_open_launches_when_not_running(self):
        app = _FakeAppleMusic(running=False)
        km, start_keeper_mock = self._patch_keeper()
        with self._patch_app(app), km:
            M.keep_music_open("")
        self.assertEqual(app.launches, 1)
        start_keeper_mock.assert_called_once()

    def test_keep_open_does_not_relaunch_when_running(self):
        app = _FakeAppleMusic(running=True)
        km, _ = self._patch_keeper()
        with self._patch_app(app), km:
            M.keep_music_open("")
        self.assertEqual(app.launches, 0)   # already up — no focus-stealing launch

    def test_keep_open_persist_failure_noted(self):
        app = _FakeAppleMusic(running=True)
        km, _ = self._patch_keeper()
        with mock.patch.object(M, "_persist_setting", return_value=False), \
             self._patch_app(app), km:
            out = M.keep_music_open("")
        self.assertIn("revert on restart", out)

    def test_keep_open_no_launch_in_staging(self):
        app = _FakeAppleMusic(running=False)
        with mock.patch.object(M, "_is_staging", return_value=True), \
             self._patch_app(app):
            M.keep_music_open("")
        self.assertEqual(app.launches, 0)             # staging never launches
        self.assertTrue(self.saved_settings.get("APPLE_MUSIC_KEEP_OPEN"))  # still persists

    def test_stop_disables_and_persists(self):
        self.cfg.APPLE_MUSIC_AUTOSTART = True
        self.cfg.APPLE_MUSIC_KEEP_OPEN = True
        out = M.stop_keeping_music_open("")
        self.assertFalse(self.cfg.APPLE_MUSIC_AUTOSTART)
        self.assertFalse(self.cfg.APPLE_MUSIC_KEEP_OPEN)
        self.assertFalse(self.saved_settings.get("APPLE_MUSIC_AUTOSTART"))
        self.assertFalse(self.saved_settings.get("APPLE_MUSIC_KEEP_OPEN"))
        self.assertIn("stop", out.lower())

    def test_stop_actually_stops_the_watchdog(self):
        # Disabling must STOP the keep-alive watchdog (call stop_keeper), not
        # merely flip the flag — otherwise the loop keeps spinning forever.
        self.cfg.APPLE_MUSIC_AUTOSTART = True
        self.cfg.APPLE_MUSIC_KEEP_OPEN = True
        with mock.patch.object(K, "stop_keeper", return_value=True) as stopper:
            M.stop_keeping_music_open("")
        stopper.assert_called_once()

    def test_stop_when_already_off_is_noop_message(self):
        out = M.stop_keeping_music_open("")
        self.assertIn("wasn't keeping", out)
        # Nothing persisted because we short-circuited before the writer.
        self.assertEqual(self.saved_settings, {})

    def test_keep_open_bridge_absent_still_persists(self):
        # apple_music_app unavailable -> no launch, but flags still persist and
        # the reply still renders (graceful).
        km, _ = self._patch_keeper()
        with self._patch_app(None), km:
            out = M.keep_music_open("")
        self.assertTrue(self.saved_settings.get("APPLE_MUSIC_KEEP_OPEN"))
        self.assertIn("tray", out.lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
