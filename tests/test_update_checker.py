"""Tests for core.update_checker — the GitHub-release comparison core.

CI-safe: stdlib only, no real network (the one HTTP helper is patched). Env is
restored after every test via mock.patch.dict so token/owner overrides can't
leak between cases.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import core.update_checker as uc


def _no_token():
    """Context manager: env with BOTH token vars removed (restored on exit)."""
    ctx = mock.patch.dict(os.environ, {}, clear=False)

    class _C:
        def __enter__(self):
            ctx.start()
            os.environ.pop("JARVIS_GITHUB_TOKEN", None)
            os.environ.pop("GITHUB_TOKEN", None)
            return self

        def __exit__(self, *a):
            ctx.stop()
            return False

    return _C()


class ParseVersionTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(uc.parse_version("1.2.3"), (1, 2, 3, None))

    def test_leading_v_and_prerelease(self):
        self.assertEqual(uc.parse_version("v1.0.0-beta.1"), (1, 0, 0, "beta.1"))

    def test_surrounding_whitespace(self):
        self.assertEqual(uc.parse_version("  2.5.9\n"), (2, 5, 9, None))

    def test_invalid_string(self):
        self.assertIsNone(uc.parse_version("latest"))
        self.assertIsNone(uc.parse_version("1.2"))
        self.assertIsNone(uc.parse_version(""))

    def test_non_string(self):
        self.assertIsNone(uc.parse_version(None))
        self.assertIsNone(uc.parse_version(123))


class CompareVersionsTests(unittest.TestCase):
    def test_less_and_greater(self):
        self.assertEqual(uc.compare_versions("1.1.0", "1.2.0"), -1)
        self.assertEqual(uc.compare_versions("1.2.0", "1.1.0"), 1)

    def test_equal(self):
        self.assertEqual(uc.compare_versions("1.1.0", "v1.1.0"), 0)

    def test_patch_and_minor_and_major(self):
        self.assertEqual(uc.compare_versions("1.0.0", "1.0.1"), -1)
        self.assertEqual(uc.compare_versions("2.0.0", "1.9.9"), 1)

    def test_prerelease_below_release(self):
        self.assertEqual(uc.compare_versions("1.0.0-beta.1", "1.0.0"), -1)
        self.assertEqual(uc.compare_versions("1.0.0", "1.0.0-beta.1"), 1)

    def test_prerelease_ordering(self):
        # numeric identifiers compare by value
        self.assertEqual(uc.compare_versions("1.0.0-beta.1", "1.0.0-beta.2"), -1)
        # numeric ranks below alphanumeric
        self.assertEqual(uc.compare_versions("1.0.0-1", "1.0.0-alpha"), -1)
        self.assertEqual(uc.compare_versions("1.0.0-beta", "1.0.0-beta"), 0)

    def test_real_bump(self):
        # the exact case this feature exists for
        self.assertEqual(uc.compare_versions("1.0.0-beta.1", "1.1.0"), -1)

    def test_unparseable_returns_none(self):
        self.assertIsNone(uc.compare_versions("1.1.0", "garbage"))
        self.assertIsNone(uc.compare_versions("garbage", "1.1.0"))


class TokenAndRepoTests(unittest.TestCase):
    def test_jarvis_token_wins(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_GITHUB_TOKEN": "a", "GITHUB_TOKEN": "b"},
                             clear=False):
            self.assertEqual(uc._token(), "a")

    def test_github_token_fallback(self):
        with _no_token():
            with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "b"}, clear=False):
                self.assertEqual(uc._token(), "b")

    def test_blank_token_is_absent(self):
        with _no_token():
            with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "   "},
                                 clear=False):
                self.assertIsNone(uc._token())

    def test_no_token(self):
        with _no_token():
            self.assertIsNone(uc._token())

    def test_owner_repo_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_GITHUB_OWNER", None)
            os.environ.pop("JARVIS_GITHUB_REPO", None)
            self.assertEqual(uc._owner_repo(), ("Sk8ter2404", "jarvis"))

    def test_owner_repo_override(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_GITHUB_OWNER": "acme",
                              "JARVIS_GITHUB_REPO": "bot"}, clear=False):
            self.assertEqual(uc._owner_repo(), ("acme", "bot"))

    def test_owner_repo_whitespace_falls_back(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_GITHUB_OWNER": "   ",
                              "JARVIS_GITHUB_REPO": "   "}, clear=False):
            self.assertEqual(uc._owner_repo(), ("Sk8ter2404", "jarvis"))


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class HttpAndFetchTests(unittest.TestCase):
    def test_http_get_json_parses(self):
        payload = json.dumps({"tag_name": "v1.2.0"}).encode("utf-8")
        with mock.patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
            out = uc._http_get_json("https://x", {}, 1.0)
        self.assertEqual(out["tag_name"], "v1.2.0")

    def test_fetch_no_token(self):
        with _no_token():
            self.assertIsNone(uc.fetch_latest_release())

    def test_fetch_success(self):
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "_http_get_json",
                                   return_value={"tag_name": "v1.2.0"}):
                self.assertEqual(uc.fetch_latest_release()["tag_name"], "v1.2.0")

    def test_fetch_network_error_returns_none(self):
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "_http_get_json",
                                   side_effect=OSError("network down")):
                self.assertIsNone(uc.fetch_latest_release())

    def test_fetch_non_dict_returns_none(self):
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "_http_get_json", return_value=["nope"]):
                self.assertIsNone(uc.fetch_latest_release())


class CheckForUpdateTests(unittest.TestCase):
    def test_no_token_skips(self):
        with _no_token():
            r = uc.check_for_update(current="1.1.0")
        self.assertFalse(r["checked"])
        self.assertFalse(r["update_available"])
        self.assertIn("no GitHub token", r["detail"])

    def test_api_unreachable(self):
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "fetch_latest_release", return_value=None):
                r = uc.check_for_update(current="1.1.0")
        self.assertFalse(r["checked"])
        self.assertIn("couldn't reach", r["detail"])

    def test_unparseable_latest_tag(self):
        rel = {"tag_name": "nightly", "html_url": "u", "name": "Nightly"}
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "fetch_latest_release", return_value=rel):
                r = uc.check_for_update(current="1.1.0")
        self.assertFalse(r["checked"])
        self.assertEqual(r["latest"], "nightly")
        self.assertIn("couldn't compare", r["detail"])

    def test_update_available(self):
        rel = {"tag_name": "v1.2.0", "html_url": "https://gh/r/v1.2.0",
               "name": "JARVIS 1.2.0", "published_at": "2026-06-04T00:00:00Z"}
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "fetch_latest_release", return_value=rel):
                r = uc.check_for_update(current="1.1.0")
        self.assertTrue(r["checked"])
        self.assertTrue(r["update_available"])
        self.assertEqual(r["latest"], "v1.2.0")
        self.assertEqual(r["release_url"], "https://gh/r/v1.2.0")
        self.assertEqual(r["release_name"], "JARVIS 1.2.0")
        self.assertIn("update available", r["detail"])

    def test_up_to_date(self):
        rel = {"tag_name": "v1.1.0", "html_url": "u", "name": None}
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "fetch_latest_release", return_value=rel):
                r = uc.check_for_update(current="1.1.0")
        self.assertTrue(r["checked"])
        self.assertFalse(r["update_available"])
        # name falls back to the tag when the release has no title
        self.assertEqual(r["release_name"], "v1.1.0")
        self.assertIn("up to date", r["detail"])

    def test_local_ahead_is_up_to_date(self):
        rel = {"tag_name": "v1.0.0", "html_url": "u", "name": "old"}
        with mock.patch.dict(os.environ, {"JARVIS_GITHUB_TOKEN": "t"}, clear=False):
            with mock.patch.object(uc, "fetch_latest_release", return_value=rel):
                r = uc.check_for_update(current="1.1.0")
        self.assertTrue(r["checked"])
        self.assertFalse(r["update_available"])

    def test_default_current_is_local_version(self):
        with _no_token():
            r = uc.check_for_update()
        self.assertEqual(r["current"], uc.LOCAL_VERSION)


class UpdateMessageTests(unittest.TestCase):
    def test_available(self):
        msg = uc.update_message(
            {"update_available": True, "latest": "v1.2.0", "current": "1.1.0"})
        self.assertIn("new version", msg)
        self.assertIn("v1.2.0", msg)

    def test_up_to_date(self):
        msg = uc.update_message(
            {"update_available": False, "checked": True, "current": "1.1.0"})
        self.assertIn("up to date", msg)

    def test_could_not_check(self):
        msg = uc.update_message(
            {"update_available": False, "checked": False, "current": "1.1.0",
             "detail": "no token"})
        self.assertIn("couldn't check", msg)
        self.assertIn("no token", msg)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.dir, ignore_errors=True))
        self.path = os.path.join(self.dir, "uc.json")

    def test_default_cache_path(self):
        p = uc.default_cache_path().replace("\\", "/")
        self.assertTrue(p.endswith("data/update_check.json"))

    def test_read_cache_missing(self):
        self.assertIsNone(uc.read_cache(self.path))

    def test_read_write_roundtrip(self):
        self.assertTrue(uc.write_cache(self.path, {"a": 1, "checked_at": 5}))
        self.assertEqual(uc.read_cache(self.path), {"a": 1, "checked_at": 5})

    def test_read_cache_non_dict(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([1, 2], f)
        self.assertIsNone(uc.read_cache(self.path))

    def test_read_cache_bad_json(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{ not json")
        self.assertIsNone(uc.read_cache(self.path))

    def test_write_cache_failure_returns_false(self):
        with mock.patch.object(uc.os, "makedirs", side_effect=OSError("ro")):
            self.assertFalse(uc.write_cache(self.path, {"a": 1}))


class CachedCheckTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.dir, ignore_errors=True))
        self.path = os.path.join(self.dir, "uc.json")

    def test_fresh_cache_reused(self):
        uc.write_cache(self.path, {"current": "1.1.0", "update_available": False,
                                   "checked_at": 1000.0, "detail": "x"})
        with mock.patch.object(uc, "check_for_update") as cfu:
            out = uc.cached_check(ttl_hours=24, path=self.path, now=1000.0 + 3600)
        cfu.assert_not_called()
        self.assertTrue(out["cached"])

    def test_stale_cache_refetches(self):
        uc.write_cache(self.path, {"current": "1.1.0", "checked_at": 1000.0})
        with mock.patch.object(uc, "check_for_update",
                               return_value={"current": "1.1.0", "update_available": False}):
            out = uc.cached_check(ttl_hours=1, path=self.path, now=1000.0 + 7200)
        self.assertFalse(out["cached"])
        self.assertEqual(out["checked_at"], 1000.0 + 7200)
        self.assertIsNotNone(uc.read_cache(self.path))

    def test_future_cache_refetches(self):
        # clock went backwards (negative age) -> don't trust the cache
        uc.write_cache(self.path, {"checked_at": 9999.0})
        with mock.patch.object(uc, "check_for_update", return_value={"current": "1.1.0"}):
            out = uc.cached_check(path=self.path, now=1000.0)
        self.assertFalse(out["cached"])

    def test_no_cache_fetches(self):
        with mock.patch.object(uc, "check_for_update",
                               return_value={"current": "1.1.0", "update_available": True}):
            out = uc.cached_check(path=self.path, now=5000.0)
        self.assertFalse(out["cached"])

    def test_bad_checked_at_refetches(self):
        uc.write_cache(self.path, {"current": "1.1.0", "checked_at": "nope"})
        with mock.patch.object(uc, "check_for_update", return_value={"current": "1.1.0"}):
            out = uc.cached_check(path=self.path, now=5000.0)
        self.assertFalse(out["cached"])

    def test_now_defaults_to_time(self):
        with mock.patch.object(uc, "check_for_update", return_value={"current": "1.1.0"}):
            out = uc.cached_check(path=self.path)
        self.assertIn("checked_at", out)


class BootNudgeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.dir, ignore_errors=True))
        self.path = os.path.join(self.dir, "uc.json")

    def test_disabled_returns_none(self):
        calls = []
        self.assertIsNone(uc.boot_nudge(calls.append, enabled=False, path=self.path))
        self.assertEqual(calls, [])

    def test_update_available_announces(self):
        calls = []
        with mock.patch.object(uc, "cached_check",
                               return_value={"update_available": True, "latest": "v1.2.0",
                                             "current": "1.1.0"}):
            res = uc.boot_nudge(calls.append, path=self.path)
        self.assertEqual(len(calls), 1)
        self.assertIn("v1.2.0", calls[0])
        self.assertTrue(res["update_available"])

    def test_no_update_no_announce(self):
        calls = []
        with mock.patch.object(uc, "cached_check",
                               return_value={"update_available": False, "current": "1.1.0"}):
            uc.boot_nudge(calls.append, path=self.path)
        self.assertEqual(calls, [])

    def test_cached_check_raises_returns_none(self):
        with mock.patch.object(uc, "cached_check", side_effect=RuntimeError("boom")):
            self.assertIsNone(uc.boot_nudge(lambda m: None, path=self.path))

    def test_announce_raising_is_swallowed(self):
        def _boom(_m):
            raise RuntimeError("tts down")
        with mock.patch.object(uc, "cached_check",
                               return_value={"update_available": True, "latest": "v1.2.0",
                                             "current": "1.1.0"}):
            res = uc.boot_nudge(_boom, path=self.path)
        self.assertTrue(res["update_available"])


class CheckForUpdatesActionTests(unittest.TestCase):
    """The core.actions surface (_act_check_for_updates) — light tier; the
    action delegates to update_checker so it's importable without the monolith."""

    def test_action_renders_update_message(self):
        import core.actions as A
        with mock.patch("core.update_checker.check_for_update",
                        return_value={"update_available": True, "latest": "v1.2.0",
                                      "current": "1.1.0", "checked": True, "detail": "x"}):
            out = A._act_check_for_updates()
        self.assertIn("new version", out)
        self.assertIn("v1.2.0", out)

    def test_action_up_to_date(self):
        import core.actions as A
        with mock.patch("core.update_checker.check_for_update",
                        return_value={"update_available": False, "checked": True,
                                      "current": "1.1.0", "detail": "x"}):
            out = A._act_check_for_updates()
        self.assertIn("up to date", out)


if __name__ == "__main__":
    unittest.main()
