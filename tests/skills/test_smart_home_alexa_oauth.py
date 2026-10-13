"""Hermetic tests for the Alexa OAuth device-registration sign-in in
``skills/smart_home_discover.py`` (2026-10-12).

Background: the old browser fallback signed in through the RETIRED
alexa.amazon.com web app and produced plain amazon.com cookies, so the Alexa
device list never came back (and alexapy logged the whole cookie jar). The
fix uses alexapy's phone-app OAuth flow: browser sign-in at
``AlexaLogin(oauth_login=True).start_url`` -> ``/ap/maplanding`` carrying
``openid.oa2.authorization_code`` -> device registration -> refresh token
persisted to ``data/alexa_oauth.json``.

Everything here is offline: alexapy and Playwright are fakes, the data dir is
a throwaway temp dir, no browser opens, no socket is touched. Every secret in
a fixture is an obvious placeholder.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

MAPLANDING = "https://www.amazon.com/ap/maplanding"
FAKE_CODE = "FAKE-AUTH-CODE-0001"
FAKE_JAR_REPR = (
    "<RequestsCookieJar[Cookie(version=0, name='at-main', "
    "value='Atza|FAKEACCESSTOKEN', port=None, domain='.amazon.com', "
    "path='/'), Cookie(version=0, name='session-id', value='111-2222222-3333333', "
    "port=None, domain='.amazon.com', path='/')]>")


def run(coro):
    return asyncio.run(coro)


@contextlib.contextmanager
def captured_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class _Base(unittest.TestCase):
    """Fresh module copy + a temp data dir (so `_oauth_path()` and the
    catalog/cookie paths can never reach the owner's live data/)."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")
        self.tmp = tempfile.mkdtemp(prefix="shoauth_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = os.path.join(self.tmp, "data")
        os.makedirs(self.data)
        for name, val in (
            ("_DATA_DIR", self.data),
            ("_PROJECT_DIR", self.tmp),
            ("_COOKIE_JSON_PATH", os.path.join(self.data, "alexa_cookie.json")),
            ("_COOKIE_PICKLE_PATH",
             os.path.join(self.data, "alexa_cookie.pickle")),
            ("_CATALOG_PATH", os.path.join(self.data, "smart_home_devices.json")),
            ("_TODO_PATH", os.path.join(self.tmp, "jarvis_todo.md")),
        ):
            p = mock.patch.object(self.mod, name, val)
            p.start()
            self.addCleanup(p.stop)


# ── fakes ───────────────────────────────────────────────────────────────────
class FakeOAuthLogin:
    """Duck-typed AlexaLogin covering the OAuth surface the skill drives."""
    instances: list = []
    registers = True            # does test_loggedin() succeed?
    refresh_ok = True
    exchange_ok = True

    def __init__(self, **kw):
        self.kw = kw
        oauth = kw.get("oauth") or {}
        self.email = kw.get("email")
        self.uuid = kw.get("uuid") or "UUID-NEW"
        self.deviceid = "DEVID-" + self.uuid
        for k in ("access_token", "refresh_token", "mac_dms", "expires_in",
                  "code_verifier", "code_challenge", "authorization_code"):
            setattr(self, k, oauth.get(k))
        self.code_verifier = self.code_verifier or "VERIFIER-PLACEHOLDER"
        self.code_challenge = self.code_challenge or "CHALLENGE-PLACEHOLDER"
        self.status = {}
        self.calls: list[str] = []
        try:                                  # the skill builds it inside a loop
            self.built_in_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.built_in_loop = None
        self.closed = False
        self.start_url = "https://www.amazon.com/ap/register?placeholder=1"
        FakeOAuthLogin.instances.append(self)

    async def test_loggedin(self, cookies=None):
        self.calls.append("test_loggedin")
        if not (FakeOAuthLogin.registers and self.authorization_code):
            return False
        self.access_token = "Atza|FAKE-NEW-ACCESS"
        self.refresh_token = "Atnr|FAKE-NEW-REFRESH"
        self.mac_dms = "FAKE-MAC-DMS"
        self.expires_in = 4102444800.0
        self.status = {"login_successful": True}
        return True

    async def refresh_access_token(self):
        self.calls.append("refresh_access_token")
        if not FakeOAuthLogin.refresh_ok:
            return False
        self.access_token = "Atza|FAKE-REFRESHED-ACCESS"
        self.expires_in = 4102444900.0
        return True

    async def exchange_token_for_cookies(self):
        self.calls.append("exchange_token_for_cookies")
        return FakeOAuthLogin.exchange_ok

    async def get_csrf(self):
        self.calls.append("get_csrf")
        return True

    async def finalize_login(self):
        self.calls.append("finalize_login")
        self.status = {"login_successful": True}

    async def close(self):
        self.closed = True


class FakeAPI:
    echo = [{"serialNumber": "ECHO1", "accountName": "Kitchen Echo"}]
    appliances: list = []
    network_details_raises = False

    @classmethod
    async def get_devices(cls, login):
        return list(cls.echo)

    @classmethod
    async def get_network_details(cls, login):
        if cls.network_details_raises:
            raise RuntimeError("graphql down")
        return list(cls.appliances)


def fake_alexapy():
    m = types.ModuleType("alexapy")
    m.AlexaLogin = FakeOAuthLogin
    m.AlexaAPI = FakeAPI
    return m


def _reset_fakes():
    FakeOAuthLogin.instances = []
    FakeOAuthLogin.registers = True
    FakeOAuthLogin.refresh_ok = True
    FakeOAuthLogin.exchange_ok = True
    FakeAPI.echo = [{"serialNumber": "ECHO1", "accountName": "Kitchen Echo"}]
    FakeAPI.appliances = []
    FakeAPI.network_details_raises = False


class _FakeLoginBase(_Base):
    def setUp(self):
        super().setUp()
        _reset_fakes()
        p = mock.patch.object(self.mod, "_alexapy", return_value=fake_alexapy())
        p.start()
        self.addCleanup(p.stop)


# ── 1. authorization-code extraction ────────────────────────────────────────
class CodeExtractionTests(_Base):
    def test_code_from_maplanding_url(self):
        url = (MAPLANDING + "?openid.mode=id_res&openid.oa2.authorization_code="
               + FAKE_CODE + "&openid.ns=x")
        self.assertEqual(self.mod._authorization_code_from_url(url), FAKE_CODE)
        self.assertTrue(self.mod._is_maplanding_url(url))

    def test_code_url_decoded(self):
        url = MAPLANDING + "?openid.oa2.authorization_code=ab%2Bcd%3D"
        self.assertEqual(self.mod._authorization_code_from_url(url), "ab+cd=")

    def test_code_on_a_non_maplanding_amazon_url_still_counts(self):
        url = "https://www.amazon.com/x?openid.oa2.authorization_code=" + FAKE_CODE
        self.assertEqual(self.mod._authorization_code_from_url(url), FAKE_CODE)
        self.assertFalse(self.mod._is_maplanding_url(url))

    def test_maplanding_without_code_is_not_a_code(self):
        url = MAPLANDING + "?openid.mode=cancel"
        self.assertIsNone(self.mod._authorization_code_from_url(url))
        self.assertTrue(self.mod._is_maplanding_url(url))

    def test_untrusted_hosts_and_schemes_rejected(self):
        for url in (
            "https://evil.example.com/ap/maplanding?openid.oa2.authorization_code=X1",
            "https://amazon.com.evil.example.com/ap/maplanding?"
            "openid.oa2.authorization_code=X1",
            "http://www.amazon.com/ap/maplanding?openid.oa2.authorization_code=X1",
            "ftp://www.amazon.com/ap/maplanding?openid.oa2.authorization_code=X1",
        ):
            self.assertIsNone(self.mod._authorization_code_from_url(url), url)
            self.assertFalse(self.mod._is_maplanding_url(url), url)

    def test_garbage_inputs_never_raise(self):
        for url in ("", None, "not a url", "https://", 5):
            self.assertIsNone(self.mod._authorization_code_from_url(url))

    def test_empty_or_oversized_code_rejected(self):
        self.assertIsNone(self.mod._authorization_code_from_url(
            MAPLANDING + "?openid.oa2.authorization_code="))
        self.assertIsNone(self.mod._authorization_code_from_url(
            MAPLANDING + "?openid.oa2.authorization_code=" + "A" * 5000))


# ── 2. the browser wait loop (fake Playwright) ──────────────────────────────
class _Page:
    def __init__(self, urls, closed_after=None, goto_error=None, fire=None):
        self._urls = list(urls)
        self._closed_after = closed_after
        self._reads = 0
        self.goto_calls: list[str] = []
        self.handlers: dict = {}
        self._goto_error = goto_error
        self._fire = list(fire or [])

    @property
    def url(self):
        if len(self._urls) > 1:
            return self._urls.pop(0)
        return self._urls[0] if self._urls else ""

    def on(self, evt, cb):
        self.handlers[evt] = cb

    def is_closed(self):
        if self._closed_after is None:
            return False
        self._reads += 1
        return self._reads > self._closed_after

    async def goto(self, url):
        self.goto_calls.append(url)
        if self._goto_error:
            raise self._goto_error
        # simulate transit navigations that the poll could miss
        for u in self._fire:
            cb = self.handlers.get("framenavigated")
            if cb:
                cb(types.SimpleNamespace(url=u))


class _Context:
    def __init__(self, page, cookies=None):
        self._page = page
        self._cookies = cookies if cookies is not None else [
            {"name": "x", "value": "fake", "domain": ".amazon.com"}]

    async def new_page(self):
        return self._page

    async def cookies(self):
        return list(self._cookies)


class _Browser:
    def __init__(self, context):
        self._context = context
        self.closed = False

    async def new_context(self):
        return self._context

    async def close(self):
        self.closed = True


class _Chromium:
    def __init__(self, browser, fail_channels=()):
        self._browser = browser
        self._fail = set(fail_channels)
        self.launch_calls: list = []

    async def launch(self, headless=False, channel=None):
        self.launch_calls.append(channel)
        if channel in self._fail:
            raise RuntimeError("launch failed")
        return self._browser


class _PWManager:
    def __init__(self, chromium):
        self.chromium = chromium

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _fake_playwright(manager):
    api = types.ModuleType("playwright.async_api")
    api.async_playwright = lambda: manager
    root = types.ModuleType("playwright")
    root.async_api = api
    return mock.patch.dict(sys.modules, {"playwright": root,
                                         "playwright.async_api": api})


class BrowserOauthCodeTests(_Base):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(self.mod.asyncio, "sleep",
                              new=mock.AsyncMock(return_value=None))
        p.start()
        self.addCleanup(p.stop)

    def _go(self, page, timeout=5.0, fail_channels=(), cookies=None):
        ctx = _Context(page, cookies)
        browser = _Browser(ctx)
        chromium = _Chromium(browser, fail_channels)
        with _fake_playwright(_PWManager(chromium)), captured_stdout() as out:
            res = run(self.mod._browser_oauth_code(
                "https://www.amazon.com/ap/register?start=1", timeout))
        return res, browser, chromium, page, out.getvalue()

    def test_success_when_url_reaches_maplanding_with_code(self):
        page = _Page(["https://www.amazon.com/ap/register?start=1",
                      "https://www.amazon.com/ap/signin",
                      MAPLANDING + "?openid.oa2.authorization_code=" + FAKE_CODE])
        res, browser, chromium, page, out = self._go(page)
        self.assertEqual(res[0], FAKE_CODE)
        self.assertEqual(res[1][0]["name"], "x")           # cookies handed on
        self.assertEqual(page.goto_calls,
                         ["https://www.amazon.com/ap/register?start=1"])
        self.assertTrue(browser.closed)
        self.assertEqual(chromium.launch_calls, ["chrome"])  # first choice
        self.assertNotIn(FAKE_CODE, out)                    # never printed

    def test_navigation_event_catches_a_transit_url_the_poll_misses(self):
        # page.url never shows the code; only the framenavigated event does.
        page = _Page(["https://www.amazon.com/ap/signin"], closed_after=0,
                     fire=[MAPLANDING + "?openid.oa2.authorization_code="
                           + FAKE_CODE])
        res, *_ = self._go(page)
        self.assertEqual(res[0], FAKE_CODE)

    def test_window_closed_after_the_code_was_seen_still_succeeds(self):
        page = _Page(["https://www.amazon.com/ap/signin"], closed_after=0,
                     fire=[MAPLANDING + "?openid.oa2.authorization_code="
                           + FAKE_CODE])
        res, browser, *_ = self._go(page)
        self.assertEqual(res[0], FAKE_CODE)
        self.assertTrue(browser.closed)

    def test_window_closed_before_sign_in_finishes(self):
        page = _Page(["https://www.amazon.com/ap/signin"], closed_after=1)
        res, browser, _c, _p, out = self._go(page)
        self.assertIsNone(res)
        self.assertTrue(browser.closed)
        self.assertIn("closed", out)

    def test_timeout(self):
        page = _Page(["https://www.amazon.com/ap/signin"])
        res, browser, _c, _p, out = self._go(page, timeout=0.0)
        self.assertIsNone(res)
        self.assertTrue(browser.closed)
        self.assertIn("timed out", out)

    def test_maplanding_without_a_code_is_a_refusal(self):
        page = _Page([MAPLANDING + "?openid.mode=cancel"])
        res, _b, _c, _p, out = self._go(page)
        self.assertIsNone(res)
        self.assertIn("without granting", out)

    def test_code_from_an_untrusted_host_is_ignored(self):
        page = _Page(["https://evil.example.com/ap/maplanding?"
                      "openid.oa2.authorization_code=" + FAKE_CODE],
                     closed_after=1)
        res, *_ = self._go(page)
        self.assertIsNone(res)

    def test_launch_chain_chrome_then_bundled_then_edge(self):
        page = _Page([MAPLANDING + "?openid.oa2.authorization_code=" + FAKE_CODE])
        res, _b, chromium, *_ = self._go(page, fail_channels=("chrome", None))
        self.assertEqual(chromium.launch_calls, ["chrome", None, "msedge"])
        self.assertEqual(res[0], FAKE_CODE)

    def test_no_launchable_browser(self):
        page = _Page(["x"])
        res, _b, chromium, _p, out = self._go(
            page, fail_channels=("chrome", None, "msedge"))
        self.assertIsNone(res)
        self.assertIn("Could not launch any headed browser", out)

    def test_page_load_failure(self):
        page = _Page(["x"], goto_error=RuntimeError("net down"))
        res, browser, _c, _p, out = self._go(page)
        self.assertIsNone(res)
        self.assertTrue(browser.closed)
        self.assertIn("Failed to load", out)

    def test_playwright_missing(self):
        broken = types.ModuleType("playwright")
        with mock.patch.dict(sys.modules, {"playwright": broken}):
            sys.modules.pop("playwright.async_api", None)
            with captured_stdout():
                res = run(self.mod._browser_oauth_code("https://x", 1.0))
        self.assertIsNone(res)

    def test_login_view_of_the_retired_url_is_gone_from_the_new_flow(self):
        # The new sign-in must open alexapy's start_url, never the retired
        # alexa.amazon.com web sign-in constant.
        _reset_fakes()
        page = _Page([MAPLANDING + "?openid.oa2.authorization_code=" + FAKE_CODE])
        ctx = _Context(page)
        chromium = _Chromium(_Browser(ctx))
        with mock.patch.object(self.mod, "_alexapy", return_value=fake_alexapy()), \
             _fake_playwright(_PWManager(chromium)), captured_stdout():
            run(self.mod._oauth_signin_and_fetch_async(5.0))
        self.assertEqual(page.goto_calls,
                         ["https://www.amazon.com/ap/register?placeholder=1"])
        self.assertNotIn("alexa.amazon.com", page.goto_calls[0])


# ── 3. persistence ──────────────────────────────────────────────────────────
class OauthPersistenceTests(_FakeLoginBase):
    def _logged_in(self):
        lg = FakeOAuthLogin(email="jarvis-owner", uuid="UUID-ABC")
        lg.access_token = "Atza|FAKE-ACCESS"
        lg.refresh_token = "Atnr|FAKE-REFRESH"
        lg.mac_dms = "FAKE-MAC"
        lg.expires_in = 4102444800.0
        lg.authorization_code = FAKE_CODE
        return lg

    def test_round_trip(self):
        lg = self._logged_in()
        with captured_stdout() as out:
            self.assertTrue(self.mod._save_oauth(lg))
        path = os.path.join(self.data, "alexa_oauth.json")
        self.assertTrue(os.path.isfile(path))
        rec = self.mod._load_oauth()
        self.assertEqual(rec["refresh_token"], "Atnr|FAKE-REFRESH")
        self.assertEqual(rec["access_token"], "Atza|FAKE-ACCESS")
        self.assertEqual(rec["mac_dms"], "FAKE-MAC")
        self.assertEqual(rec["expires_in"], 4102444800.0)
        self.assertEqual(rec["code_verifier"], "VERIFIER-PLACEHOLDER")
        self.assertEqual(rec["authorization_code"], FAKE_CODE)
        self.assertEqual(rec["uuid"], "UUID-ABC")
        self.assertEqual(rec["deviceid"], "DEVID-UUID-ABC")
        self.assertNotIn("FAKE-REFRESH", out.getvalue())     # values never printed

    def test_nothing_saved_without_a_refresh_token(self):
        lg = self._logged_in()
        lg.refresh_token = None
        self.assertFalse(self.mod._save_oauth(lg))
        self.assertFalse(os.path.exists(os.path.join(self.data,
                                                      "alexa_oauth.json")))

    def test_load_rejects_missing_corrupt_and_tokenless_files(self):
        self.assertIsNone(self.mod._load_oauth())
        path = os.path.join(self.data, "alexa_oauth.json")
        for body in ("{not json", "[]", json.dumps({"access_token": "x"}),
                     json.dumps({"refresh_token": ""})):
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            self.assertIsNone(self.mod._load_oauth(), body)

    def test_path_follows_the_current_data_dir(self):
        self.assertEqual(self.mod._oauth_path(),
                         os.path.join(self.data, "alexa_oauth.json"))

    def test_purge_removes_the_oauth_file_too(self):
        self.mod._save_oauth(self._logged_in())
        out = self.mod.smart_home_purge_cookie("")
        self.assertIn("cleared", out)
        self.assertIsNone(self.mod._load_oauth())

    def test_oauth_file_is_gitignored(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, ".gitignore"), encoding="utf-8") as f:
            text = f.read()
        self.assertTrue("data/*" in text or "alexa_oauth" in text)


# ── 4. restore path ordering ────────────────────────────────────────────────
class RestoreTests(_FakeLoginBase):
    def _save_fake_oauth(self):
        lg = FakeOAuthLogin(email="jarvis-owner", uuid="UUID-ABC")
        lg.refresh_token = "Atnr|FAKE-REFRESH"
        lg.access_token = "Atza|FAKE-OLD"
        lg.mac_dms = "FAKE-MAC"
        lg.expires_in = 1.0
        self.mod._save_oauth(lg)
        FakeOAuthLogin.instances.clear()

    def test_restore_sequence_is_refresh_exchange_csrf_then_fetch(self):
        self._save_fake_oauth()
        FakeAPI.appliances = [{"applianceId": "A1", "entityId": "E1",
                               "friendlyName": "Desk Lamp"}]
        with captured_stdout():
            devices = run(self.mod._oauth_restore_and_fetch_async())
        lg = FakeOAuthLogin.instances[-1]
        self.assertEqual(lg.calls, ["refresh_access_token",
                                    "exchange_token_for_cookies", "get_csrf",
                                    "finalize_login"])
        self.assertEqual(devices["smarthome"][0]["friendlyName"], "Desk Lamp")
        self.assertEqual(devices["echo"][0]["serialNumber"], "ECHO1")
        self.assertTrue(lg.closed)
        # restored from the saved material, under the saved device identity
        self.assertEqual(lg.kw["uuid"], "UUID-ABC")
        self.assertTrue(lg.kw["oauth_login"])
        self.assertEqual(lg.kw["oauth"]["refresh_token"], "Atnr|FAKE-REFRESH")
        self.assertEqual(lg.kw["oauth"]["mac_dms"], "FAKE-MAC")
        # the refreshed access token was written back
        self.assertEqual(self.mod._load_oauth()["access_token"],
                         "Atza|FAKE-REFRESHED-ACCESS")

    def test_built_and_used_in_one_running_loop(self):
        self._save_fake_oauth()
        seen = {}

        async def probe(login):
            seen["fetch_loop"] = asyncio.get_running_loop()
            return {"echo": [], "smarthome": [], "groups": []}

        with mock.patch.object(self.mod, "_fetch_devices_async", new=probe), \
             captured_stdout():
            run(self.mod._oauth_restore_and_fetch_async())
        lg = FakeOAuthLogin.instances[-1]
        self.assertIs(lg.built_in_loop, seen["fetch_loop"])

    def test_refused_refresh_returns_none_and_keeps_the_file(self):
        self._save_fake_oauth()
        FakeOAuthLogin.refresh_ok = False
        with captured_stdout():
            self.assertIsNone(run(self.mod._oauth_restore_and_fetch_async()))
        self.assertIsNotNone(self.mod._load_oauth())     # never deleted
        self.assertTrue(FakeOAuthLogin.instances[-1].closed)

    def test_login_async_returns_the_open_logged_in_login(self):
        # 2026-10-13: the router drives devices with this login, on a loop it
        # owns - so it must come back OPEN (not closed) and built in the
        # running loop.
        self._save_fake_oauth()
        with captured_stdout():
            lg = run(self._login_and_loop())
        login, loop, running = lg
        self.assertIs(login.built_in_loop, running)
        self.assertFalse(login.closed)
        self.assertEqual(login.calls, ["refresh_access_token",
                                       "exchange_token_for_cookies",
                                       "get_csrf", "finalize_login"])
        self.assertEqual(self.mod._load_oauth()["access_token"],
                         "Atza|FAKE-REFRESHED-ACCESS")

    async def _login_and_loop(self):
        login = await self.mod._oauth_login_async()
        return login, None, asyncio.get_running_loop()

    def test_login_async_refused_refresh_closes_and_returns_none(self):
        self._save_fake_oauth()
        FakeOAuthLogin.refresh_ok = False
        with captured_stdout():
            self.assertIsNone(run(self.mod._oauth_login_async()))
        self.assertIsNotNone(self.mod._load_oauth())
        self.assertTrue(FakeOAuthLogin.instances[-1].closed)

    def test_login_async_without_a_saved_file_builds_nothing(self):
        self.assertIsNone(run(self.mod._oauth_login_async()))
        self.assertEqual(FakeOAuthLogin.instances, [])

    def test_refused_cookie_exchange_returns_none(self):
        self._save_fake_oauth()
        FakeOAuthLogin.exchange_ok = False
        with captured_stdout():
            self.assertIsNone(run(self.mod._oauth_restore_and_fetch_async()))

    def test_no_saved_file_means_none_and_no_login_is_built(self):
        self.assertIsNone(run(self.mod._oauth_restore_and_fetch_async()))
        self.assertEqual(FakeOAuthLogin.instances, [])

    # -- ordering: OAuth first, legacy cookie pickle second -------------------
    def test_oauth_success_never_touches_the_cookie_pickle(self):
        order = []

        async def oauth():
            order.append("oauth")
            return {"echo": [1], "smarthome": [], "groups": []}

        async def legacy():
            order.append("legacy")
            return {"echo": [2], "smarthome": [], "groups": []}

        self._save_fake_oauth()
        with mock.patch.object(self.mod, "_oauth_restore_and_fetch_async", new=oauth), \
             mock.patch.object(self.mod, "_restore_and_fetch_async", new=legacy):
            out = run(self.mod._restore_any_and_fetch_async())
        self.assertEqual(order, ["oauth"])
        self.assertEqual(out["echo"], [1])

    def test_oauth_failure_falls_back_to_the_cookie_pickle(self):
        order = []

        async def oauth():
            order.append("oauth")
            return None

        async def legacy():
            order.append("legacy")
            return {"echo": [2], "smarthome": [], "groups": []}

        self._save_fake_oauth()
        with mock.patch.object(self.mod, "_oauth_restore_and_fetch_async", new=oauth), \
             mock.patch.object(self.mod, "_restore_and_fetch_async", new=legacy):
            out = run(self.mod._restore_any_and_fetch_async())
        self.assertEqual(order, ["oauth", "legacy"])
        self.assertEqual(out["echo"], [2])

    def test_oauth_exception_falls_back_to_the_cookie_pickle(self):
        order = []

        async def oauth():
            order.append("oauth")
            raise RuntimeError("boom")

        async def legacy():
            order.append("legacy")
            return None

        self._save_fake_oauth()
        with mock.patch.object(self.mod, "_oauth_restore_and_fetch_async", new=oauth), \
             mock.patch.object(self.mod, "_restore_and_fetch_async", new=legacy), \
             captured_stdout():
            self.assertIsNone(run(self.mod._restore_any_and_fetch_async()))
        self.assertEqual(order, ["oauth", "legacy"])

    def test_no_oauth_file_goes_straight_to_the_cookie_pickle(self):
        order = []

        async def oauth():
            order.append("oauth")
            return None

        async def legacy():
            order.append("legacy")
            return None

        with mock.patch.object(self.mod, "_oauth_restore_and_fetch_async", new=oauth), \
             mock.patch.object(self.mod, "_restore_and_fetch_async", new=legacy):
            run(self.mod._restore_any_and_fetch_async())
        self.assertEqual(order, ["legacy"])


# ── 5. the interactive first sign-in ────────────────────────────────────────
class SigninFlowTests(_FakeLoginBase):
    def _patch_browser(self, result):
        async def fake_browser(start_url, timeout=300.0):
            self.seen_start_url = start_url
            self.browser_loop = asyncio.get_running_loop()
            return result
        return mock.patch.object(self.mod, "_browser_oauth_code", new=fake_browser)

    def test_full_flow_registers_saves_and_fetches(self):
        FakeAPI.appliances = [{"applianceId": "A1", "entityId": "E1",
                               "friendlyName": "Desk Lamp"}]
        with self._patch_browser((FAKE_CODE, [{"name": "n", "value": "v"}])), \
             captured_stdout() as out:
            devices = run(self.mod._oauth_signin_and_fetch_async())
        lg = FakeOAuthLogin.instances[-1]
        self.assertEqual(self.seen_start_url, lg.start_url)
        self.assertEqual(lg.authorization_code, FAKE_CODE)
        self.assertIn("test_loggedin", lg.calls)
        self.assertEqual(lg.status, {"login_successful": True})
        self.assertEqual(devices["smarthome"][0]["friendlyName"], "Desk Lamp")
        self.assertTrue(lg.closed)
        rec = self.mod._load_oauth()
        self.assertEqual(rec["refresh_token"], "Atnr|FAKE-NEW-REFRESH")
        self.assertEqual(rec["authorization_code"], FAKE_CODE)
        for secret in ("FAKE-NEW-REFRESH", "FAKE-NEW-ACCESS", FAKE_CODE):
            self.assertNotIn(secret, out.getvalue())
        # constructed as an OAuth (phone-app) login, not the retired web one
        self.assertTrue(lg.kw["oauth_login"])

    def test_browser_login_and_fetch_share_one_event_loop(self):
        seen = {}

        async def probe(login):
            seen["fetch"] = asyncio.get_running_loop()
            return {"echo": [], "smarthome": [], "groups": []}

        with self._patch_browser((FAKE_CODE, [])), \
             mock.patch.object(self.mod, "_fetch_devices_async", new=probe), \
             captured_stdout():
            run(self.mod._oauth_signin_and_fetch_async())
        lg = FakeOAuthLogin.instances[-1]
        self.assertIs(lg.built_in_loop, self.browser_loop)
        self.assertIs(lg.built_in_loop, seen["fetch"])

    def test_cancelled_browser_returns_none_and_saves_nothing(self):
        with self._patch_browser(None), captured_stdout():
            self.assertIsNone(run(self.mod._oauth_signin_and_fetch_async()))
        self.assertIsNone(self.mod._load_oauth())
        self.assertNotIn("test_loggedin", FakeOAuthLogin.instances[-1].calls)
        self.assertTrue(FakeOAuthLogin.instances[-1].closed)

    def test_registration_refused_returns_none_and_saves_nothing(self):
        FakeOAuthLogin.registers = False
        with self._patch_browser((FAKE_CODE, [])), captured_stdout():
            self.assertIsNone(run(self.mod._oauth_signin_and_fetch_async()))
        self.assertIsNone(self.mod._load_oauth())

    def test_alexapy_missing(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            self.assertIsNone(run(self.mod._oauth_signin_and_fetch_async()))

    def test_registration_exception_is_reported_without_secrets(self):
        async def boom(self_, cookies=None):
            raise RuntimeError("failed with cookies=" + FAKE_JAR_REPR)

        with self._patch_browser((FAKE_CODE, [])), \
             mock.patch.object(FakeOAuthLogin, "test_loggedin", new=boom), \
             captured_stdout() as out:
            self.assertIsNone(run(self.mod._oauth_signin_and_fetch_async()))
        self.assertNotIn("FAKEACCESSTOKEN", out.getvalue())
        self.assertNotIn("111-2222222-3333333", out.getvalue())


# ── 6. wizard / voice wiring ────────────────────────────────────────────────
class WizardPhaseTests(_Base):
    def _runner(self, results):
        """_run_async stand-in: records the coroutine function name, returns
        (or raises) the scripted value."""
        calls = []

        def _run(coro, timeout=None):
            name = getattr(coro, "__qualname__", "?")
            calls.append(name)
            try:
                coro.close()
            except Exception:
                pass
            val = results.pop(0)
            if isinstance(val, Exception):
                raise val
            return val
        _run.calls = calls
        return _run

    GOOD = {"echo": [{"serialNumber": "S"}], "smarthome": [{"entityId": "e"}],
            "groups": []}

    def _phase(self, force, results, oauth=True, meta=None):
        runner = self._runner(results)
        with mock.patch.object(self.mod, "_run_async", side_effect=runner), \
             mock.patch.object(self.mod, "_load_oauth",
                               return_value={"refresh_token": "x"} if oauth else None), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=meta), \
             captured_stdout():
            out = self.mod._wizard_alexa_phase(force)
        return out, runner.calls

    def test_cached_oauth_is_used_without_a_browser(self):
        (devices, status), calls = self._phase(False, [self.GOOD])
        self.assertEqual(devices, self.GOOD)
        self.assertTrue(status["ok"])
        self.assertEqual(len(calls), 1)
        self.assertIn("_restore_any_and_fetch_async", calls[0])

    def test_reauth_skips_the_cache_and_goes_to_the_browser(self):
        (devices, status), calls = self._phase(True, [self.GOOD])
        self.assertEqual(len(calls), 1)
        self.assertIn("_oauth_signin_and_fetch_async", calls[0])
        self.assertTrue(status["ok"])

    def test_unusable_cached_result_falls_through_to_the_browser(self):
        empty = {"echo": [], "smarthome": [], "groups": []}
        (devices, status), calls = self._phase(False, [empty, self.GOOD])
        self.assertEqual([c.split(".")[-1] for c in calls],
                         ["_restore_any_and_fetch_async",
                          "_oauth_signin_and_fetch_async"])
        self.assertEqual(devices, self.GOOD)

    def test_cached_exception_falls_through_to_the_browser(self):
        (devices, status), calls = self._phase(
            False, [RuntimeError("cached boom"), self.GOOD])
        self.assertEqual(len(calls), 2)
        self.assertTrue(status["ok"])

    def test_nothing_cached_goes_straight_to_the_browser(self):
        (devices, status), calls = self._phase(False, [self.GOOD], oauth=False)
        self.assertEqual(len(calls), 1)
        self.assertIn("_oauth_signin_and_fetch_async", calls[0])

    def test_cancelled_signin_reports_honestly_and_names_the_command(self):
        (devices, status), _ = self._phase(True, [None])
        self.assertIsNone(devices)
        self.assertFalse(status["ok"])
        self.assertIn("did not complete", status["error"])
        self.assertIn("skills.smart_home_discover reauth", status["error"])

    def test_signin_exception_is_reported_not_raised(self):
        (devices, status), _ = self._phase(True, [RuntimeError("kaboom")])
        self.assertIsNone(devices)
        self.assertIn("failed during sign-in", status["error"])

    def test_wizard_never_asks_for_a_password(self):
        with mock.patch.object(self.mod, "_prompt_credentials") as pc, \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=self._runner([None])), \
             mock.patch.object(self.mod, "_load_oauth", return_value=None), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             captured_stdout():
            self.mod._wizard_alexa_phase(False)
        pc.assert_not_called()


class VoicePhaseTests(_Base):
    def _call(self, oauth, meta, pickle_exists, run_result):
        seen = []

        def _run(coro, timeout=None):
            seen.append((getattr(coro, "__qualname__", "?"), timeout))
            coro.close()
            if isinstance(run_result, Exception):
                raise run_result
            return run_result

        with mock.patch.object(self.mod, "_alexapy", return_value=object()), \
             mock.patch.object(self.mod, "_load_oauth", return_value=oauth), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=meta), \
             mock.patch.object(self.mod.os.path, "exists",
                               return_value=pickle_exists), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run), \
             captured_stdout():
            res = self.mod._alexa_voice_phase(False)
        return res, seen

    def test_oauth_alone_is_enough_and_is_bounded(self):
        good = {"echo": [], "smarthome": [{"entityId": "e"}], "groups": []}
        (devices, status), seen = self._call({"refresh_token": "x"}, None, False, good)
        self.assertEqual(devices, good)
        self.assertTrue(status["ok"])
        self.assertIn("_restore_any_and_fetch_async", seen[0][0])
        self.assertEqual(seen[0][1], self.mod._DISCOVERY_TIMEOUT_SEC)

    def test_nothing_cached_never_runs_a_fetch(self):
        (devices, status), seen = self._call(None, None, False, None)
        self.assertIsNone(devices)
        self.assertEqual(status["error"], "no cached Amazon sign-in")
        self.assertTrue(status["console"])
        self.assertEqual(seen, [])

    def test_legacy_cookie_still_triggers_a_fetch(self):
        (devices, status), seen = self._call(None, {"email": "e"}, True, None)
        self.assertEqual(len(seen), 1)
        self.assertEqual(status["error"], "the cached sign-in was unusable")

    def test_voice_path_never_opens_a_browser(self):
        with mock.patch.object(self.mod, "_browser_oauth_code") as b:
            self._call({"refresh_token": "x"}, None, False, None)
        b.assert_not_called()

    def test_error_text_is_redacted(self):
        (devices, status), _ = self._call(
            {"refresh_token": "x"}, None, False,
            RuntimeError("login(cookies=" + FAKE_JAR_REPR + ")"))
        self.assertIsNone(devices)
        self.assertNotIn("FAKEACCESSTOKEN", status["error"])
        self.assertNotIn("111-2222222-3333333", status["error"])


# ── 7. device fetch: smart-home appliances, not just Echos ──────────────────
APPLIANCES = [
    {"applianceId": "AP1", "entityId": "EN1", "friendlyName": "Living Room Fan",
     "manufacturerName": "SmartLife", "modelName": "fan1",
     "applianceTypes": ["FAN"],
     "capabilities": [{"interfaceName": "Alexa.PowerController"}]},
    {"applianceId": "AP2", "entityId": "EN2", "friendlyName": "Desk Light",
     "manufacturerName": "Govee", "modelName": "H6000",
     "applianceTypes": ["LIGHT"],
     "capabilities": [{"interfaceName": "Alexa.PowerController"},
                      {"interfaceName": "Alexa.BrightnessController"}]},
    {"applianceId": "AP3", "entityId": "EN3", "friendlyName": "Hallway Thermostat",
     "manufacturerName": "Google Nest", "modelName": "T3",
     "applianceTypes": ["THERMOSTAT"],
     "capabilities": [{"interfaceName": "Alexa.ThermostatController"}]},
    {"applianceId": "AP4", "entityId": "EN4", "friendlyName": "Porch Camera",
     "manufacturerName": "Google Nest", "applianceTypes": ["CAMERA"],
     "capabilities": [{"interfaceName": "Alexa.CameraStreamController"}]},
    {"applianceId": "AP5", "entityId": "EN5", "friendlyName": "Coffee Plug",
     "manufacturerName": "SmartLife", "applianceTypes": ["SMARTPLUG"],
     "capabilities": [{"interfaceName": "Alexa.PowerController"}]},
    {"applianceId": "AP6", "entityId": "EN6", "friendlyName": "Kitchen Echo",
     "manufacturerName": "Amazon", "applianceTypes": ["ALEXA_VOICE_ENABLED"],
     "capabilities": []},
]


class FetchDevicesTests(_FakeLoginBase):
    def test_smart_home_list_comes_from_get_network_details(self):
        FakeAPI.appliances = APPLIANCES
        out = run(self.mod._fetch_devices_async(FakeOAuthLogin(email="x")))
        self.assertEqual(len(out["smarthome"]), len(APPLIANCES))
        self.assertEqual(out["echo"][0]["serialNumber"], "ECHO1")

    def test_catalog_types_for_the_owners_device_kinds(self):
        cat = self.mod._build_catalog(
            {"echo": [], "smarthome": APPLIANCES, "groups": []}, [])
        types_by_name = {d["name"]: d["type"] for d in cat["devices"]}
        self.assertEqual(types_by_name["Living Room Fan"], "fan")
        self.assertEqual(types_by_name["Desk Light"], "light")
        self.assertEqual(types_by_name["Hallway Thermostat"], "thermostat")
        self.assertEqual(types_by_name["Porch Camera"], "camera")
        self.assertEqual(types_by_name["Coffee Plug"], "plug")
        self.assertEqual(types_by_name["Kitchen Echo"], "unknown")
        caps = {d["name"]: d["capabilities"] for d in cat["devices"]}
        self.assertEqual(caps["Desk Light"], ["dim", "on_off"])    # interfaceName shape
        self.assertEqual(cat["device_count"], len(APPLIANCES))

    def test_falls_back_to_the_raw_graphql_query(self):
        FakeAPI.network_details_raises = True

        async def gql(cls, login):
            return [{"legacyAppliance": APPLIANCES[0]}, {"other": 1}]

        with mock.patch.object(FakeAPI, "get_devices_gql",
                               new=classmethod(gql), create=True), \
             captured_stdout():
            out = run(self.mod._fetch_devices_async(FakeOAuthLogin(email="x")))
        self.assertEqual([a["friendlyName"] for a in out["smarthome"]],
                         ["Living Room Fan"])

    def test_failures_are_swallowed_and_redacted(self):
        async def boom(cls, login):
            raise RuntimeError("dead (cookies=" + FAKE_JAR_REPR + ")")

        with mock.patch.object(FakeAPI, "get_devices", new=classmethod(boom)), \
             mock.patch.object(FakeAPI, "get_network_details",
                               new=classmethod(boom)), \
             captured_stdout() as out:
            res = run(self.mod._fetch_devices_async(FakeOAuthLogin(email="x")))
        self.assertEqual(res, {"echo": [], "smarthome": [], "groups": []})
        self.assertNotIn("FAKEACCESSTOKEN", out.getvalue())
        self.assertNotIn("111-2222222-3333333", out.getvalue())


class ConstructLoginTests(_Base):
    def test_output_path_builder_returns_a_string_never_none(self):
        # None here made alexapy's finalize_login raise
        # "'NoneType' object has no attribute 'replace'".
        seen = {}

        class L:
            def __init__(self, **kw):
                seen.update(kw)

        m = types.ModuleType("alexapy")
        m.AlexaLogin = L
        with mock.patch.object(self.mod, "_alexapy", return_value=m), \
             captured_stdout() as out:
            self.mod._construct_login("jarvis-owner", "")
            path = seen["outputpath"](".storage/alexa_media.jarvis-owner.pickle")
        self.assertIsInstance(path, str)
        self.assertTrue(path.startswith(self.data))
        self.assertEqual(out.getvalue(), "")            # and it no longer prints
        self.assertFalse(os.path.exists(os.path.dirname(path)))   # never created

    def test_oauth_kwargs_are_forwarded_only_when_given(self):
        seen = {}

        class L:
            def __init__(self, **kw):
                seen.clear()
                seen.update(kw)

        m = types.ModuleType("alexapy")
        m.AlexaLogin = L
        with mock.patch.object(self.mod, "_alexapy", return_value=m):
            self.mod._construct_login("e", "")
            self.assertNotIn("oauth", seen)
            self.assertNotIn("uuid", seen)
            self.assertNotIn("oauth_login", seen)
            self.mod._construct_login("e", "", oauth={"refresh_token": "r"},
                                      uuid="U1", oauth_login=True)
            self.assertEqual(seen["oauth"], {"refresh_token": "r"})
            self.assertEqual(seen["uuid"], "U1")
            self.assertIs(seen["oauth_login"], True)


# ── 8. log redaction ────────────────────────────────────────────────────────
class RedactionTests(_Base):
    SAMPLE = ("alexalogin.login(<AlexaLogin>, {'cookies': " + FAKE_JAR_REPR
              + "}): An error occurred accessing AlexaAPI: KeyError('x')")

    def _assert_clean(self, text):
        for secret in ("FAKEACCESSTOKEN", "111-2222222-3333333", "Atza|FAKE"):
            self.assertNotIn(secret, text)

    def test_redact_cookie_jar_repr(self):
        out = self.mod._redact(self.SAMPLE)
        self._assert_clean(out)
        self.assertIn("[REDACTED]", out)
        self.assertIn("An error occurred accessing AlexaAPI", out)  # context kept

    def test_redact_other_secret_shapes(self):
        text = ("tokens {'access_token': 'Atza|FAKE1', 'refresh_token': "
                "'Atnr|FAKE2', 'mac_dms': 'FAKEMAC', 'csrf': '12345'} "
                "Cookie: session-id=111-2222222-3333333; at-main=Atza|FAKE3\n"
                "<Cookie sst-main=FAKESST for .amazon.com/> "
                "https://www.amazon.com/ap/maplanding?"
                "openid.oa2.authorization_code=FAKECODE&x=1 "
                "value='FAKEVAL' bare Atnr|FAKEBARE tail")
        out = self.mod._redact(text)
        for secret in ("FAKE1", "FAKE2", "FAKEMAC", "12345", "111-2222222",
                       "FAKE3", "FAKESST", "FAKECODE", "FAKEVAL", "FAKEBARE"):
            self.assertNotIn(secret, out, secret)
        self.assertIn("tail", out)
        self.assertIn("x=1", out)

    def test_redact_is_idempotent_and_total(self):
        once = self.mod._redact(self.SAMPLE)
        self.assertEqual(self.mod._redact(once), once)
        self.assertEqual(self.mod._redact(None), "None")
        self.assertEqual(self.mod._redact("nothing secret here"),
                         "nothing secret here")

    def test_filter_scrubs_a_log_record_message_args_and_traceback(self):
        flt = self.mod._RedactingFilter()
        try:
            raise RuntimeError("cookies=" + FAKE_JAR_REPR)
        except RuntimeError:
            exc_info = sys.exc_info()
        rec = logging.LogRecord("alexapy.helpers", logging.WARNING, __file__, 1,
                                "failed (%s, %s)", (FAKE_JAR_REPR, "ok"),
                                exc_info)
        self.assertTrue(flt.filter(rec))
        text = rec.getMessage() + (rec.exc_text or "")
        self._assert_clean(text)
        self.assertIn("ok", rec.getMessage())

    def test_filter_leaves_clean_records_untouched(self):
        flt = self.mod._RedactingFilter()
        rec = logging.LogRecord("alexapy", logging.INFO, __file__, 1,
                                "hello %s", ("world",), None)
        flt.filter(rec)
        self.assertEqual(rec.msg, "hello %s")
        self.assertEqual(rec.args, ("world",))

    def test_alexapy_child_logger_output_is_redacted_by_install(self):
        # The leak path: alexapy.helpers (a CHILD logger) warns with the jar
        # and the handler prints it. The logger-level filter alone would miss
        # a child; the installed record factory must not.
        self.mod._install_log_redaction()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)       # NO filter on purpose
        logger = logging.getLogger("alexapy.helpers_test_child")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        logger.warning("x.login(%s, %s): An error occurred: %s",
                       "<AlexaLogin>", {"cookies": FAKE_JAR_REPR}, "KeyError('e')")
        text = stream.getvalue()
        self.assertIn("An error occurred", text)
        self._assert_clean(text)

    def test_unrelated_loggers_are_not_rewritten(self):
        self.mod._install_log_redaction()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("jarvis_unrelated_test")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        logger.info("session-id=keep-me")
        self.assertIn("keep-me", stream.getvalue())

    def test_install_is_idempotent_and_covers_last_resort_and_root(self):
        self.mod._install_log_redaction()
        self.mod._install_log_redaction()
        flt = logging._jarvis_alexa_redactor
        self.assertIsInstance(flt, logging.Filter)
        self.assertEqual(logging.lastResort.filters.count(flt), 1)
        self.assertEqual(logging.getLogger("alexapy").filters.count(flt), 1)
        factory_before = logging.getLogRecordFactory()
        self.mod._install_log_redaction()
        self.assertIs(logging.getLogRecordFactory(), factory_before)

    def test_load_skill_does_not_stack_factories(self):
        before = logging.getLogRecordFactory()
        load_skill_isolated("smart_home_discover")
        self.assertIs(logging.getLogRecordFactory(), before)

    def test_login_exception_text_is_redacted_before_printing(self):
        class L:
            def __init__(self, **kw):
                raise RuntimeError("ctor failed cookies=" + FAKE_JAR_REPR)

        m = types.ModuleType("alexapy")
        m.AlexaLogin = L
        with mock.patch.object(self.mod, "_alexapy", return_value=m), \
             captured_stdout() as out:
            self.assertIsNone(self.mod._construct_login("e", ""))
        self._assert_clean(out.getvalue())


if __name__ == "__main__":
    unittest.main()
