"""Smart-home router: the Alexa COMMAND path uses the saved OAuth sign-in.

2026-10-13. Discovery restores the owner's phone-app OAuth sign-in
(data/alexa_oauth.json) but the router used to drive devices with the legacy
cookie pickle only. The router now owns ONE long-lived event loop, builds the
login on it (OAuth first, cookie second, None when neither works) and runs
every control call on that same loop.

Hermetic: a fake ``skills.smart_home_discover`` (just the sign-in builders and
paths) and a fake ``alexapy`` (AlexaAPI.set_light_state). No network, no real
token / cookie file, no real device names - every name is made up.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import types
import unittest
from unittest import mock

from core import smart_home_router as router
from tests.test_smart_home_router import _inject_modules


def _ok(entity_id):
    return {"controlResponses": [{"entityId": entity_id, "code": "SUCCESS"}],
            "errors": []}


class _Login:
    """A fake logged-in AlexaLogin; remembers which loop built it."""

    def __init__(self, kind):
        self.kind = kind
        self.status = {"login_successful": True}
        self.built_on = asyncio.get_running_loop()
        self.closed = False


class _Env:
    """Fake discover module + fake alexapy for one test."""

    def __init__(self, tmp, oauth=True, cookie=True):
        self.tmp = tmp
        self.oauth_ok = oauth
        self.cookie_ok = cookie
        self.built = []          # [(kind, login)]
        self.calls = []          # [(login, entity, power_on, brightness, loop)]
        self.replies = []        # queued replies for set_light_state
        self.oauth_file = os.path.join(tmp, "alexa_oauth.json")
        self.cookie_file = os.path.join(tmp, "alexa_cookie.pickle")
        env = self

        async def _oauth():
            if not env.oauth_ok:
                return None
            lg = _Login("oauth")
            env.built.append(("oauth", lg))
            return lg

        async def _cookie():
            if not env.cookie_ok:
                return None
            lg = _Login("cookie")
            env.built.append(("cookie", lg))
            return lg

        async def _close(lg):
            lg.closed = True

        disc = types.ModuleType("skills.smart_home_discover")
        disc._oauth_login_async = _oauth
        disc._cookie_login_async = _cookie
        disc._close_login = _close
        disc._oauth_path = lambda: env.oauth_file
        disc._COOKIE_PICKLE_PATH = env.cookie_file
        self.disc = disc

        async def set_light_state(login, entity, power_on=True,
                                  brightness=None):
            env.calls.append((login, entity, power_on, brightness,
                              asyncio.get_running_loop()))
            if env.replies:
                return env.replies.pop(0)
            return _ok(entity)

        alexapy = types.ModuleType("alexapy")
        alexapy.AlexaAPI = types.SimpleNamespace(set_light_state=set_light_state)
        self.alexapy = alexapy

    def modules(self):
        return {"skills.smart_home_discover": self.disc,
                "alexapy": self.alexapy}

    def write_signin(self, bump=0):
        """Create / replace the (fake, token-free) sign-in file with a new
        mtime so the router sees a NEW sign-in."""
        with open(self.oauth_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"placeholder": bump}))
        t = time.time() + bump
        os.utime(self.oauth_file, (t, t))


def _dev(eid="ent-lamp"):
    return {"name": "Fake Lamp", "alexa_entity_id": eid}


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="shrouter_oauth_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._reset()
        self.addCleanup(self._reset)

    @staticmethod
    def _reset():
        with router._state_lock:
            router._state["alexapy_login"] = None
            router._state["alexapy_login_loop"] = None
            router._state["alexapy_failed_sig"] = None

    def env(self, **kw):
        e = _Env(self.tmp, **kw)
        cm = _inject_modules(**e.modules())
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return e


class SignInPriorityTests(_Base):
    def test_oauth_is_preferred_over_cookie(self):
        e = self.env()
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertTrue(out.get("ok"), out)
        self.assertEqual([k for k, _ in e.built], ["oauth"])
        self.assertEqual(e.calls[0][0].kind, "oauth")

    def test_cookie_is_the_fallback(self):
        e = self.env(oauth=False)
        out = router._alexa_set_state(_dev(), {"on": False})
        self.assertTrue(out.get("ok"), out)
        self.assertEqual([k for k, _ in e.built], ["cookie"])
        self.assertIs(e.calls[0][2], False)

    def test_neither_is_an_honest_not_signed_in_naming_reauth(self):
        e = self.env(oauth=False, cookie=False)
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertNotIn("ok", out)
        self.assertIn("alexa fallback unavailable", out["error"])
        self.assertIn("not signed in", out["error"])
        self.assertIn("python -m skills.smart_home_discover reauth",
                      out["error"])
        self.assertEqual(e.calls, [])           # nothing was claimed or sent

    def test_failed_sentinel_stops_retrying_every_command(self):
        e = self.env(oauth=False, cookie=False)
        router._alexa_set_state(_dev(), {"on": True})
        with mock.patch.object(router, "_build_login_async") as build:
            out = router._alexa_set_state(_dev(), {"on": True})
        build.assert_not_called()
        self.assertIn("not signed in", out["error"])
        with router._state_lock:
            self.assertEqual(router._state["alexapy_login"], "_failed")
        self.assertEqual(e.built, [])

    def test_new_oauth_file_clears_the_sentinel(self):
        e = self.env(oauth=False, cookie=False)
        self.assertIn("error", router._alexa_set_state(_dev(), {"on": True}))
        with router._state_lock:
            self.assertEqual(router._state["alexapy_login"], "_failed")
        # The owner re-runs reauth: a new sign-in file appears.
        e.oauth_ok = True
        e.write_signin()
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertTrue(out.get("ok"), out)
        self.assertEqual([k for k, _ in e.built], ["oauth"])

    def test_same_unchanged_file_keeps_the_sentinel(self):
        e = self.env(oauth=False, cookie=False)
        e.write_signin()
        router._alexa_set_state(_dev(), {"on": True})
        e.oauth_ok = True               # would work, but the file is unchanged
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertIn("not signed in", out["error"])
        self.assertEqual(e.built, [])


class SingleLoopTests(_Base):
    def test_login_and_control_share_one_loop_across_commands(self):
        e = self.env()
        a = router._alexa_set_state(_dev("ent-1"), {"on": True})
        b = router._alexa_set_state(_dev("ent-2"), {"on": False})
        self.assertTrue(a.get("ok") and b.get("ok"), (a, b))
        self.assertEqual(len(e.built), 1, "the login must be reused")
        login = e.built[0][1]
        self.assertEqual(len(e.calls), 2)
        for call in e.calls:
            self.assertIs(call[0], login)
            self.assertIs(call[4], login.built_on,
                          "control ran on a different loop than the login")

    def test_the_loop_is_not_the_callers_loop_and_is_a_daemon(self):
        self.env()
        router._alexa_set_state(_dev(), {"on": True})
        th = router._loop_box["thread"]
        self.assertTrue(th.daemon and th.is_alive())

    def test_router_run_is_bounded(self):
        async def _hang():
            await asyncio.sleep(30)
        t0 = time.monotonic()
        with self.assertRaises(TimeoutError):
            router._router_run(_hang(), 0.2)
        self.assertLess(time.monotonic() - t0, 5.0)


class ExpiredSessionTests(_Base):
    def test_expired_session_drops_the_login_and_retries_once(self):
        e = self.env()
        # First control call: Amazon says the session is unauthorised.
        e.replies = [{"errors": [{"code": "401 Unauthorized"}],
                      "controlResponses": []}]
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(len(e.built), 2, "one fresh sign-in after the 401")
        self.assertTrue(e.built[0][1].closed, "the dead login is released")
        self.assertEqual(len(e.calls), 2)
        self.assertIs(e.calls[0][0], e.built[0][1])
        self.assertIs(e.calls[1][0], e.built[1][1])

    def test_non_json_reply_counts_as_a_dead_session(self):
        e = self.env()
        e.replies = [None]
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(len(e.calls), 2)

    def test_a_second_failure_is_reported_not_retried_again(self):
        e = self.env()
        bad = {"errors": [{"code": "401"}], "controlResponses": []}
        e.replies = [bad, bad, bad]
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertNotIn("ok", out)
        self.assertIn("error", out)
        self.assertEqual(len(e.calls), 2, "exactly one retry")
        with router._state_lock:
            self.assertEqual(router._state["alexapy_login"], "_failed")

    def test_amazon_refusing_the_command_is_not_retried(self):
        e = self.env()
        e.replies = [{"errors": [{"code": "ENDPOINT_UNREACHABLE",
                                  "message": "device offline"}],
                      "controlResponses": []}]
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertIn("ENDPOINT_UNREACHABLE", out["error"])
        self.assertEqual(len(e.calls), 1)
        self.assertEqual(len(e.built), 1)

    def test_retry_when_resignin_fails_is_honest(self):
        e = self.env()
        e.replies = [{"errors": [{"code": "401"}], "controlResponses": []}]
        orig = e.disc._oauth_login_async

        async def _once():
            if e.built:
                return None
            return await orig()
        e.disc._oauth_login_async = _once
        e.cookie_ok = False
        out = router._alexa_set_state(_dev(), {"on": True})
        self.assertNotIn("ok", out)
        self.assertIn("not signed in", out["error"])
        self.assertEqual(len(e.calls), 1)

    def test_unsupported_color_is_still_a_polite_refusal(self):
        e = self.env()
        out = router._alexa_set_state(_dev(), {"color": (1, 2, 3)})
        self.assertIn("no compatible alexapy call", out["error"])
        self.assertEqual(e.calls, [])


class CatalogReloadTests(_Base):
    """The router already re-reads the catalog from disk once the in-memory
    copy is older than the TTL, so a discovery run that rewrites it needs no
    JARVIS restart. Pinned here with the new 'bigger catalog' shape."""

    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.tmp, "smart_home_devices.json")
        saved = router._CATALOG_PATH
        router._CATALOG_PATH = self.path
        self.addCleanup(setattr, router, "_CATALOG_PATH", saved)
        with router._state_lock:
            router._state["catalog"] = None
            router._state["loaded_at"] = 0.0
        self.addCleanup(self._clear)

    @staticmethod
    def _clear():
        with router._state_lock:
            router._state["catalog"] = None
            router._state["loaded_at"] = 0.0

    def _write(self, n):
        devs = [{"name": f"Fake Device {i}", "brand": "Alexa"}
                for i in range(n)]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"device_count": n, "devices": devs}, f)

    def test_rewritten_catalog_is_used_without_a_restart(self):
        self._write(3)
        self.assertEqual(len(router._ensure_catalog()["devices"]), 3)
        self._write(38)                       # discovery rewrote it
        with router._state_lock:              # the TTL elapses
            router._state["loaded_at"] = time.monotonic() - (
                router._CATALOG_TTL_SECS + 1)
        self.assertEqual(len(router._ensure_catalog()["devices"]), 38)

    def test_refresh_clears_the_login_state_too(self):
        self.env()
        router._alexa_set_state(_dev(), {"on": True})
        self._write(2)
        router.refresh_smart_home_router("")
        with router._state_lock:
            self.assertIsNone(router._state["alexapy_login"])
            self.assertIsNone(router._state["alexapy_failed_sig"])


if __name__ == "__main__":
    unittest.main()
