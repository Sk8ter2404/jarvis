"""core/url_guard.py (v3 overhaul W18): the guard in front of the link reader.

Every case is a fixture: no DNS, no sockets to the outside. The fetch is driven
through an injected ``opener`` (what the redirect loop and the pinning do) and,
for the real http.client path, through a socketpair that stands in for the
server.

    python -m unittest tests.test_url_guard
"""
from __future__ import annotations

import socket
import threading
import unittest
from unittest import mock

from core import url_guard as G


def _dns(table):
    """A resolver over a fixed table; records every lookup."""
    calls = []

    def resolve(host, port):
        calls.append(host)
        if host not in table:
            raise G.UrlRefused("the name did not resolve", "dns")
        ans = table[host]
        return list(ans() if callable(ans) else ans)

    resolve.calls = calls
    return resolve


PUBLIC = "93.184.216.34"


class RefusedAddressTests(unittest.TestCase):
    def assertRefused(self, url, code=None, resolver=None):
        with self.assertRaises(G.UrlRefused, msg=url) as cm:
            G.check_url(url, resolver=resolver or _dns({}))
        if code:
            self.assertEqual(cm.exception.code, code, url)

    def test_loopback_lan_tailnet_and_link_local_literals(self):
        for url in (
                "http://127.0.0.1/", "http://127.0.0.1:8080/x", "https://127.1.2.3/",
                "http://192.168.6" "8.52/", "http://192.168.69.7/", "http://192.168.71.250/",
                "http://192.168.1.1/", "http://10.0.0.5/", "http://172.16.4.4/",
                "http://100.86.3" "5.99/", "http://100.64" ".0.1/", "http://100.127" ".255.254/",
                "http://169.254.169.254/latest/meta-data/", "http://0.0.0.0/",
                "http://198.18.0.9/", "http://224.0.0.1/", "http://255.255.255.255/"):
            self.assertRefused(url, "private")

    def test_ipv6_equivalents(self):
        for url in (
                "http://[::1]/", "http://[::]/", "http://[fe80::1]/",
                "http://[fd00::1]/", "http://[fd7a:115c:a1e0::1]/",
                "http://[::ffff:127.0.0.1]/", "http://[::ffff:192.168.6" "8.52]/",
                "http://[::ffff:100.86.3" "5.99]/", "http://[64:ff9b::7f00:1]/",
                "http://[2002:7f00:1::]/", "http://[::127.0.0.1]/",
                "http://[2001:0:4136:e378:8000:63bf:3fff:fdd2]/"):
            self.assertRefused(url, "private")

    def test_public_literals_pass(self):
        for url in ("http://8.8.8.8/", "https://[2606:4700:4700::1111]/",
                    "http://[::ffff:8.8.8.8]/", "https://1.1.1.1/dns"):
            t = G.check_url(url, resolver=_dns({}))
            self.assertTrue(t.ips)

    def test_schemes_other_than_http_and_https(self):
        for url in ("file:///etc/passwd", "file://C:/Windows/win.ini",
                    "ftp://example.com/x", "gopher://example.com/",
                    "data:text/html,hi", "javascript:alert(1)",
                    "chrome://settings", "ssh://example.com"):
            self.assertRefused(url, resolver=_dns({"example.com": [PUBLIC]}))

    def test_inside_names(self):
        for url in ("http://localhost/", "http://LOCALHOST:8080/", "http://app.localhost/",
                    "http://router.local/", "http://http://router.local",
                    "http://nas.lan/", "http://x.home/", "http://x.home.arpa/",
                    "http://db.internal/", "http://pc.ts" ".net/", "http://printer/",
                    "http://bradendesktop/", "http://foo.test/"):
            self.assertRefused(url, "name")

    def test_numeric_host_tricks(self):
        for url in ("http://2130706433/", "http://0x7f.1/", "http://0x7f000001/",
                    "http://017700000001/", "http://127.1/", "http://3232253988/",
                    "http://192.168.6" "8/"):
            self.assertRefused(url, "numeric")

    def test_unicode_disguises_of_a_private_address(self):
        for url in ("http://\uff11\uff12\uff17.\uff10.\uff10.\uff11/",       # full-width digits
                    "http://127\u3002" + "0\u3002" + "0\u30021/",               # ideographic dots
                    "http://\uff4c\uff4f\uff43\uff41\uff4c\uff48\uff4f\uff53\uff54/"):
            self.assertRefused(url, resolver=_dns({}))

    def test_credentials_ports_and_odd_forms(self):
        self.assertRefused("http://user:pw@" "example.com/", "credentials")
        self.assertRefused("http://example.com@" "evil.com/", "credentials")
        self.assertRefused("http://example.com:22/", "port")
        self.assertRefused("http://example.com:6379/", "port")
        self.assertRefused("http://example.com\\@10.0.0.1/", "form")
        self.assertRefused("http://exa mple.com/", "form")
        self.assertRefused("http://example.com/\x00", "form")
        self.assertRefused("", "form")
        self.assertRefused("http://" + "a" * 3000 + ".com/", "form")

    def test_name_resolving_to_the_lan_is_refused(self):
        r = _dns({"evil.example.com": ["192.168.6" "8.52"]})
        self.assertRefused("https://evil.example.com/", "private", r)
        r = _dns({"ts.example.com": ["100.86.3" "5.99"]})
        self.assertRefused("https://ts.example.com/", "private", r)
        r = _dns({"v6.example.com": ["fd7a:115c:a1e0::5"]})
        self.assertRefused("https://v6.example.com/", "private", r)
        r = _dns({"lo.example.com": ["::ffff:127.0.0.1"]})
        self.assertRefused("https://lo.example.com/", "private", r)

    def test_split_answers_are_refused_if_any_is_private(self):
        r = _dns({"split.example.com": [PUBLIC, "10.0.0.5"]})
        self.assertRefused("https://split.example.com/", "private", r)

    def test_unresolvable_name(self):
        self.assertRefused("https://nope.example.com/", "dns", _dns({}))

    def test_public_hosts_pass_and_normalise(self):
        r = _dns({"www.example.com": [PUBLIC], "xn--bcher-kva.example.com": [PUBLIC]})
        t = G.check_url("https://www.example.com/a/b?q=1#frag", resolver=r)
        self.assertEqual((t.scheme, t.host, t.port, t.target, t.ips),
                         ("https", "www.example.com", 443, "/a/b?q=1", (PUBLIC,)))
        t = G.check_url("http://WWW.Example.COM.:8080", resolver=r)
        self.assertEqual((t.host, t.port, t.target), ("www.example.com", 8080, "/"))
        # a scheme-less address is taken as https
        self.assertEqual(G.check_url("www.example.com/x", resolver=r).scheme, "https")
        # an internationalised name is encoded before it is resolved
        self.assertEqual(G.check_url("https://bücher.example.com/", resolver=r).host,
                         "xn--bcher-kva.example.com")

    def test_ip_block_reason_direct(self):
        self.assertIsNone(G.ip_block_reason("8.8.8.8"))
        self.assertIsNone(G.ip_block_reason("2606:4700:4700::1111"))
        for ip in ("127.0.0.1", "192.168.6" "8.52", "100.86.3" "5.99", "10.1.1.1",
                   "169.254.1.1", "::1", "fe80::1", "::ffff:10.0.0.1", "garbage"):
            self.assertIsNotNone(G.ip_block_reason(ip), ip)


class RedirectAndPinningTests(unittest.TestCase):
    def _opener(self, script):
        """script: {url-host: (status, headers, body)}; records (host, ip)."""
        seen = []

        def opener(t, ip, headers, deadline, max_bytes, clock):
            seen.append((t.host, ip, dict(headers), deadline, max_bytes))
            return script[t.host]

        opener.seen = seen
        return opener

    def test_plain_get(self):
        r = _dns({"example.com": [PUBLIC]})
        op = self._opener({"example.com": (200, {"content-type": "text/html; charset=utf-8"},
                                           b"<p>hi</p>", False)})
        out = G.safe_get("https://example.com/a", resolver=r, opener=op)
        self.assertEqual((out.status, out.body, out.hops, out.content_type),
                         (200, b"<p>hi</p>", 0, "text/html"))
        host, ip, headers, deadline, cap = op.seen[0]
        self.assertEqual((host, ip), ("example.com", PUBLIC))
        self.assertEqual(cap, G.DEFAULT_MAX_BYTES)

    def test_request_is_plain_no_cookies_no_login(self):
        r = _dns({"example.com": [PUBLIC]})
        op = self._opener({"example.com": (200, {}, b"x", False)})
        G.safe_get("https://example.com/", resolver=r, opener=op)
        h = {k.lower(): v for k, v in op.seen[0][2].items()}
        self.assertEqual(h["user-agent"], G.USER_AGENT)
        self.assertEqual(h["accept-encoding"], "identity")
        self.assertEqual(h["host"], "example.com")
        for banned in ("cookie", "authorization", "proxy-authorization", "referer",
                       "x-api-key"):
            self.assertNotIn(banned, h)

    def test_redirect_to_the_lan_is_refused_and_never_connected(self):
        r = _dns({"example.com": [PUBLIC]})
        op = self._opener({"example.com": (302, {"location": "http://192.168.6" "8.52/admin"},
                                           b"", False)})
        with self.assertRaises(G.UrlRefused) as cm:
            G.safe_get("https://example.com/", resolver=r, opener=op)
        self.assertEqual(cm.exception.code, "private")
        self.assertEqual([s[0] for s in op.seen], ["example.com"])   # no second connect

    def test_redirect_to_loopback_tailnet_and_other_schemes(self):
        for loc in ("http://127.0.0.1:8080/", "http://localhost/", "http://100.86.3" "5.99/",
                    "http://[::1]/", "file:///etc/passwd", "http://router.local/",
                    "http://2130706433/"):
            r = _dns({"example.com": [PUBLIC]})
            op = self._opener({"example.com": (301, {"location": loc}, b"", False)})
            with self.assertRaises(G.UrlRefused, msg=loc):
                G.safe_get("https://example.com/", resolver=r, opener=op)
            self.assertEqual(len(op.seen), 1, loc)

    def test_redirect_to_a_name_that_resolves_to_the_lan(self):
        r = _dns({"example.com": [PUBLIC], "sneaky.example.org": ["192.168.69.9"]})
        op = self._opener({"example.com": (307, {"location": "https://sneaky.example.org/x"},
                                           b"", False)})
        with self.assertRaises(G.UrlRefused) as cm:
            G.safe_get("https://example.com/", resolver=r, opener=op)
        self.assertEqual(cm.exception.code, "private")
        self.assertIn("sneaky.example.org", r.calls)       # the hop WAS looked up
        self.assertEqual(len(op.seen), 1)                  # and never connected to

    def test_every_hop_is_resolved_and_followed_with_the_checked_ip(self):
        r = _dns({"a.example.com": ["1.1.1.1"], "b.example.com": ["8.8.4.4"]})
        op = self._opener({
            "a.example.com": (302, {"location": "https://b.example.com/next"}, b"", False),
            "b.example.com": (200, {"content-type": "text/plain"}, b"done", False)})
        out = G.safe_get("http://a.example.com/", resolver=r, opener=op)
        self.assertEqual((out.status, out.hops, out.final_url, out.body),
                         (200, 1, "https://b.example.com/next", b"done"))
        self.assertEqual([(s[0], s[1]) for s in op.seen],
                         [("a.example.com", "1.1.1.1"), ("b.example.com", "8.8.4.4")])
        self.assertEqual(r.calls, ["a.example.com", "b.example.com"])

    def test_relative_redirect(self):
        r = _dns({"example.com": [PUBLIC]})
        hops = iter([(302, {"location": "/moved?x=1"}, b"", False),
                     (200, {"content-type": "text/html"}, b"ok", False)])
        targets = []

        def op(t, ip, headers, deadline, cap, clock):
            targets.append(t.target)
            return next(hops)

        out = G.safe_get("https://example.com/old", resolver=r, opener=op)
        self.assertEqual(targets, ["/old", "/moved?x=1"])
        self.assertEqual(out.final_url, "https://example.com/moved?x=1")

    def test_dns_rebinding_one_lookup_per_hop_and_connect_to_that_answer(self):
        # The name answers public to the check and private to a SECOND lookup.
        # There is no second lookup: the connect goes to the checked address.
        answers = iter([[PUBLIC], ["10.0.0.5"], ["10.0.0.5"]])
        r = _dns({"rebind.example.com": lambda: next(answers)})
        op = self._opener({"rebind.example.com": (200, {"content-type": "text/plain"},
                                                  b"ok", False)})
        G.safe_get("https://rebind.example.com/", resolver=r, opener=op)
        self.assertEqual(r.calls, ["rebind.example.com"])
        self.assertEqual(op.seen[0][1], PUBLIC)
        # and when the FIRST answer is private the connect never happens
        answers = iter([["10.0.0.5"], [PUBLIC]])
        r = _dns({"rebind.example.com": lambda: next(answers)})
        op = self._opener({"rebind.example.com": (200, {}, b"ok", False)})
        with self.assertRaises(G.UrlRefused):
            G.safe_get("https://rebind.example.com/", resolver=r, opener=op)
        self.assertEqual(op.seen, [])

    def test_a_redirect_that_rebinds_on_the_second_hop(self):
        # hop 1 is a public name; hop 2 is a name that answers private
        r = _dns({"one.example.com": [PUBLIC], "two.example.com": ["10.9.8.7"]})
        op = self._opener({"one.example.com": (302, {"location": "https://two.example.com/"},
                                               b"", False)})
        with self.assertRaises(G.UrlRefused):
            G.safe_get("https://one.example.com/", resolver=r, opener=op)

    def test_redirect_loop_and_missing_location(self):
        r = _dns({"example.com": [PUBLIC]})
        op = self._opener({"example.com": (302, {"location": "https://example.com/"},
                                           b"", False)})
        with self.assertRaises(G.FetchError) as cm:
            G.safe_get("https://example.com/", resolver=r, opener=op, max_redirects=5)
        self.assertEqual(cm.exception.code, "redirect")
        self.assertEqual(len(op.seen), 6)
        op = self._opener({"example.com": (302, {}, b"", False)})
        with self.assertRaises(G.FetchError):
            G.safe_get("https://example.com/", resolver=r, opener=op)

    def test_first_address_that_connects_serves(self):
        r = _dns({"example.com": ["1.1.1.1", "2606:4700:4700::1111"]})
        tried = []

        def op(t, ip, headers, deadline, cap, clock):
            tried.append(ip)
            if ip == "1.1.1.1":
                raise G.FetchError("could not connect", "connect")
            return 200, {"content-type": "text/plain"}, b"v6", False

        out = G.safe_get("https://example.com/", resolver=r, opener=op)
        self.assertEqual((tried, out.body), (["1.1.1.1", "2606:4700:4700::1111"], b"v6"))

    def test_a_timeout_is_not_retried_on_another_address(self):
        r = _dns({"example.com": ["1.1.1.1", "8.8.8.8"]})
        tried = []

        def op(t, ip, headers, deadline, cap, clock):
            tried.append(ip)
            raise G.FetchError("timed out", "timeout")

        with self.assertRaises(G.FetchError) as cm:
            G.safe_get("https://example.com/", resolver=r, opener=op)
        self.assertEqual((cm.exception.code, tried), ("timeout", ["1.1.1.1"]))

    def test_the_deadline_is_one_for_the_whole_fetch(self):
        r = _dns({"a.example.com": [PUBLIC], "b.example.com": [PUBLIC]})
        deadlines = []
        hops = iter([(302, {"location": "https://b.example.com/"}, b"", False),
                     (200, {}, b"x", False)])

        def op(t, ip, headers, deadline, cap, clock):
            deadlines.append(deadline)
            return next(hops)

        G.safe_get("https://a.example.com/", resolver=r, opener=op, timeout=15,
                   clock=lambda: 100.0)
        self.assertEqual(deadlines, [115.0, 115.0])


class _FakeServer(threading.Thread):
    """Answers one request on one end of a socketpair."""

    def __init__(self, sock, response: bytes, hold: float = 0.0, chunked_body=None):
        super().__init__(daemon=True)
        self.sock = sock
        self.response = response
        self.hold = hold
        self.request = b""

    def run(self):
        try:
            self.sock.settimeout(5)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            self.request = data
            if self.hold:
                threading.Event().wait(self.hold)
            self.sock.sendall(self.response)
        except OSError:
            pass
        finally:
            try:
                self.sock.close()
            except OSError:
                pass


class RealOpenerTests(unittest.TestCase):
    """The default opener over a socketpair: pinning, caps, content types."""

    def _get(self, response: bytes, *, url="http://example.com/p?q=1", timeout=5.0,
             max_bytes=1000, hold=0.0, ip="93.184.216.34"):
        a, b = socket.socketpair()
        srv = _FakeServer(b, response, hold=hold)
        srv.start()
        connects = []

        def fake_connect(addr, timeout=None, *args, **kw):
            connects.append(addr)
            return a

        r = _dns({"example.com": [ip]})
        with mock.patch("socket.create_connection", fake_connect):
            try:
                out = G.safe_get(url, resolver=r, max_bytes=max_bytes, timeout=timeout)
            finally:
                srv.join(2)
        return out, connects, srv.request

    def test_pinned_connect_and_request_shape(self):
        resp = (b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: 5\r\n\r\nhello")
        out, connects, req = self._get(resp)
        self.assertEqual(connects, [("93.184.216.34", 80)])        # the checked IP
        self.assertEqual((out.status, out.body, out.truncated), (200, b"hello", False))
        text = req.decode("latin-1")
        self.assertTrue(text.startswith("GET /p?q=1 HTTP/1.1\r\n"), text)
        self.assertIn("Host: example.com\r\n", text)
        self.assertIn("User-Agent: " + G.USER_AGENT, text)
        self.assertIn("Accept-Encoding: identity", text)
        self.assertNotIn("Cookie", text)
        self.assertNotIn("Authorization", text)

    def test_size_cap_truncates(self):
        body = b"x" * 5000
        resp = (b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5000\r\n\r\n"
                + body)
        out, _c, _r = self._get(resp, max_bytes=1000)
        self.assertTrue(out.truncated)
        self.assertEqual(len(out.body), 1000)

    def test_a_non_text_body_is_not_read(self):
        resp = (b"HTTP/1.1 200 OK\r\nContent-Type: video/mp4\r\nContent-Length: 9\r\n\r\n"
                b"123456789")
        out, _c, _r = self._get(resp)
        self.assertEqual((out.status, out.body, out.content_type), (200, b"", "video/mp4"))

    def test_a_server_that_never_answers_times_out(self):
        with self.assertRaises(G.FetchError) as cm:
            self._get(b"HTTP/1.1 200 OK\r\n\r\n", timeout=0.4, hold=3.0)
        self.assertEqual(cm.exception.code, "timeout")

    def test_https_names_the_host_for_sni_and_verification(self):
        a, b = socket.socketpair()
        resp = (b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nok")
        srv = _FakeServer(b, resp)
        srv.start()
        ctx = mock.Mock()
        ctx.wrap_socket.side_effect = lambda s, server_hostname=None: s
        r = _dns({"example.com": [PUBLIC]})
        with mock.patch("socket.create_connection", lambda addr, timeout=None: a), \
                mock.patch("ssl.create_default_context", return_value=ctx):
            out = G.safe_get("https://example.com/", resolver=r, timeout=5)
        srv.join(2)
        ctx.wrap_socket.assert_called_once()
        self.assertEqual(ctx.wrap_socket.call_args.kwargs["server_hostname"], "example.com")
        self.assertEqual(out.body, b"ok")

    def test_redirect_response_carries_no_body_read(self):
        resp = (b"HTTP/1.1 302 Found\r\nLocation: http://192.168.6" b"8.52/\r\n"
                b"Content-Length: 0\r\n\r\n")
        with self.assertRaises(G.UrlRefused):
            self._get(resp)


if __name__ == "__main__":
    unittest.main()
