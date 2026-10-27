"""core/url_guard.py - the URL guard and the guarded fetch behind the link reader
(v3 overhaul W18, 2026-10-27).

THE RULE
  A link that comes from outside JARVIS (the clipboard, a browser tab, a page
  a page pointed to) is a request to open a door. The only doors this opens are
  PUBLIC web pages. Never the home network, never this PC, never the tailnet.

WHAT IS REFUSED (check_url)
  * any scheme but http / https (file:, ftp:, javascript:, data:, gopher: ...);
  * user:password@ in the address, backslashes, spaces, control characters;
  * ports other than 80, 443, 8080, 8443;
  * localhost and the names that only mean "inside": *.localhost, *.local,
    *.lan, *.home, *.home.arpa, *.internal, *.localdomain, *.ts.net (Tailscale
    MagicDNS), *.corp, *.test ...; single-label hosts ("router");
  * numeric host tricks: 2130706433, 0x7f.1, 017700000001, 127.1;
  * IPv4 loopback 127/8, "this host" 0/8, private 10/8 172.16/12 192.168/16
    (which holds the home LAN, whatever its subnet),
    CGNAT / Tailscale 100.64/10, link-local 169.254/16 (cloud metadata),
    benchmarking 198.18/15, multicast, reserved, broadcast;
  * the IPv6 equivalents: ::1, ::, fc00::/7 (Tailscale's fd7a:115c:a1e0::/48
    included), fe80::/10, ff00::/8, and the forms that WRAP an IPv4 address
    (::ffff:a.b.c.d, 64:ff9b::/96, 2002::/16, ::a.b.c.d) judged by the IPv4
    inside; Teredo is refused outright;
  * a NAME is resolved and refused when ANY address it answers with is not
    public (a name pointing at the LAN, or "split" answers).

REDIRECTS AND DNS REBINDING (safe_get)
  Every hop is a new request: the URL is re-parsed and re-checked and the name
  re-resolved. The connection is then made to the IP address that was just
  CHECKED (TLS still names the original host for SNI and certificate
  verification), so a name that answers "public" to the check and "10.0.0.5"
  to the connect cannot happen: there is only one lookup per hop and the
  socket goes to its result. Proxies from the environment are not used (they
  would make the pinning moot).

CAPS
  15 s for the whole fetch (all hops), 5 MB of body, 5 redirects, no cookies
  sent or kept, no login, no Authorization, a plain User-Agent, identity
  encoding only (no decompression bombs), text-like content types only.

The fetch is injectable (``resolver=``, ``opener=``) so tests never touch the
network. The module is stdlib only and never starts a thread.
"""
from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
from typing import Callable, NamedTuple, Optional
from urllib.parse import urljoin, urlsplit

USER_AGENT = "Mozilla/5.0 (compatible; JARVIS-LinkReader/1.0)"
ALLOWED_PORTS = (80, 443, 8080, 8443)
MAX_URL_CHARS = 2048
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)

# Content types the reader can use; anything else is not fetched (the opener
# is told, reads no body and the caller gets the type back).
TEXT_TYPES = ("text/", "application/json", "application/xhtml+xml",
              "application/xml", "application/ld+json", "application/rss+xml",
              "application/atom+xml")

# ── what is "not public" ─────────────────────────────────────────────────
_BLOCKED_V4 = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8",         # "this host"
    "10.0.0.0/8",        # private
    "100.64" ".0.0/10",     # CGNAT, and every Tailscale address
    "127.0.0.0/8",       # loopback
    "169.254.0.0/16",    # link-local, cloud metadata
    "172.16.0.0/12",     # private
    "192.0.0.0/24",      # IETF protocol assignments
    "192.168.0.0/16",    # private
    "198.18.0.0/15",     # benchmarking
    "224.0.0.0/4",       # multicast
    "240.0.0.0/4",       # reserved
    "255.255.255.255/32",
))
_BLOCKED_V6 = tuple(ipaddress.ip_network(n) for n in (
    "::/8", "fc00::/7", "fd7a:115c:a1e0::/48", "fe80::/10",
    "ff00::/8", "100::/64", "2001:db8::/32", "2001::/32",   # Teredo
))

_BAD_HOST_SUFFIXES = (
    ".localhost", ".local", ".localdomain", ".lan", ".home", ".home.arpa",
    ".internal", ".intranet", ".corp", ".private", ".ts.net", ".test",
    ".invalid", ".example", ".onion", ".arpa",
)
_HOST_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
_NUMERIC_LABEL = re.compile(r"^(\d+|0x[0-9a-f]*)$")


class UrlRefused(ValueError):
    """The address is not a public web page. ``code`` is machine-readable
    (scheme, port, credentials, form, name, numeric, private, dns, redirect);
    ``reason`` is a short phrase for a log line, never for speech."""

    def __init__(self, reason: str, code: str = "refused"):
        super().__init__(reason)
        self.reason = reason
        self.code = code


class FetchError(RuntimeError):
    """A public address that could not be fetched: timeout, connection,
    too many redirects, a redirect with no destination."""

    def __init__(self, reason: str, code: str = "fetch"):
        super().__init__(reason)
        self.reason = reason
        self.code = code


class Target(NamedTuple):
    url: str
    scheme: str
    host: str          # ASCII (IDNA) host, lower case, no brackets
    port: int
    target: str        # path + query as sent on the request line
    ips: tuple         # every address the name resolved to, all public


class Fetched(NamedTuple):
    status: int
    final_url: str
    headers: dict      # lower-cased names
    body: bytes
    truncated: bool
    hops: int
    content_type: str


# ── addresses ────────────────────────────────────────────────────────────
def _embedded_v4(ip6: ipaddress.IPv6Address) -> Optional[ipaddress.IPv4Address]:
    """The IPv4 address an IPv6 address WRAPS (mapped, NAT64, 6to4, the old
    ::a.b.c.d compatible form), or None."""
    if ip6.ipv4_mapped is not None:
        return ip6.ipv4_mapped
    if ip6.sixtofour is not None:
        return ip6.sixtofour
    n = int(ip6)
    if (n >> 32) == int(ipaddress.IPv6Address("64:ff9b::")) >> 32:
        return ipaddress.IPv4Address(n & 0xFFFFFFFF)          # 64:ff9b::/96
    if (n >> 32) == 0 and n > 1:                               # ::a.b.c.d
        return ipaddress.IPv4Address(n & 0xFFFFFFFF)
    return None


def ip_block_reason(ip) -> Optional[str]:
    """None when ``ip`` is a public address, else why it is refused. A string
    that is not an IP address is refused too."""
    try:
        addr = ip if isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)) \
            else ipaddress.ip_address(str(ip).strip().split("%", 1)[0])
    except ValueError:
        return "not an IP address"
    if isinstance(addr, ipaddress.IPv6Address):
        inner = _embedded_v4(addr)
        if inner is not None:
            why = ip_block_reason(inner)
            return ("wraps a non-public IPv4 address (%s)" % why) if why else None
        for net in _BLOCKED_V6:
            if addr in net:
                return "non-public IPv6 range %s" % net
        return None if addr.is_global else "not a global IPv6 address"
    for net in _BLOCKED_V4:
        if addr in net:
            return "non-public IPv4 range %s" % net
    return None if addr.is_global else "not a global IPv4 address"


# ── the URL ──────────────────────────────────────────────────────────────
def _clean_text(url) -> str:
    s = str(url or "").strip()
    if not s or len(s) > MAX_URL_CHARS:
        raise UrlRefused("empty or over-long address", "form")
    if "\\" in s or any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in s):
        raise UrlRefused("whitespace, control character or backslash in address",
                         "form")
    return s


def _host_ascii(host: str) -> str:
    h = host.lower().rstrip(".")
    if not h or "%" in h:
        raise UrlRefused("no usable host", "name")
    if ":" in h:                       # an IPv6 literal (brackets already gone)
        return h
    try:
        h = h.encode("idna").decode("ascii")
    except UnicodeError:
        raise UrlRefused("host name does not encode", "name") from None
    if len(h) > 253:
        raise UrlRefused("host name too long", "name")
    return h


def resolve_host(host: str, port: int) -> list:
    """Every address ``host`` answers with (strings). Raises UrlRefused('dns')
    when there are none. The default resolver; tests pass their own."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM,
                                   proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        raise UrlRefused("the name did not resolve", "dns") from None
    out: list = []
    for fam, _t, _p, _c, sa in infos:
        if fam in (socket.AF_INET, socket.AF_INET6):
            ip = sa[0]
            if ip not in out:
                out.append(ip)
    if not out:
        raise UrlRefused("the name did not resolve", "dns")
    return out


def check_url(url, *, resolver: Optional[Callable] = None) -> Target:
    """Parse and vet ``url``. Returns a Target whose ``ips`` are ALL public.
    Raises UrlRefused otherwise. ``resolver(host, port) -> [ip, ...]`` is the
    DNS lookup (default: the system resolver)."""
    s = _clean_text(url)
    if "://" not in s:
        s = "https://" + s
    try:
        parts = urlsplit(s)
        host_raw = parts.hostname or ""
        port = parts.port
    except ValueError:
        raise UrlRefused("malformed address", "form") from None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise UrlRefused("only http and https links are read", "scheme")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UrlRefused("an address with a login in it", "credentials")
    host = _host_ascii(host_raw)
    port = port if port is not None else (443 if scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise UrlRefused("port %s is not one a web page uses" % port, "port")

    literal = None
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        pass
    if literal is not None:
        why = ip_block_reason(literal)
        if why:
            raise UrlRefused(why, "private")
        ips = [str(literal)]
    else:
        labels = host.split(".")
        if all(_NUMERIC_LABEL.match(x) for x in labels) or _NUMERIC_LABEL.match(labels[-1]):
            raise UrlRefused("numeric host form", "numeric")
        if host == "localhost" or "." not in host or any(
                host.endswith(suf) for suf in _BAD_HOST_SUFFIXES):
            raise UrlRefused("a name that only means 'inside'", "name")
        if not all(_HOST_LABEL.match(x) for x in labels):
            raise UrlRefused("host name has odd characters", "name")
        ips = list((resolver or resolve_host)(host, port))
        if not ips:
            raise UrlRefused("the name did not resolve", "dns")
        for ip in ips:
            why = ip_block_reason(ip)
            if why:
                raise UrlRefused("%s resolves to %s" % (host, why), "private")

    path = parts.path or "/"
    target = path + (("?" + parts.query) if parts.query else "")
    shown_host = "[%s]" % host if ":" in host else host
    default = 443 if scheme == "https" else 80
    netloc = shown_host if port == default else "%s:%d" % (shown_host, port)
    return Target("%s://%s%s" % (scheme, netloc, target), scheme, host, port,
                  target, tuple(ips))


# ── the fetch ────────────────────────────────────────────────────────────
def _type_ok(content_type: str) -> bool:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    return not ct or any(ct.startswith(t) for t in TEXT_TYPES)


def _open_pinned(t: Target, ip: str, headers: dict, deadline: float,
                 max_bytes: int, clock: Callable[[], float]):
    """One GET to the CHECKED address ``ip`` (default opener). Returns
    (status, headers, body, truncated). TLS verifies the certificate against
    the host name, not the address."""
    left = deadline - clock()
    if left <= 0:
        raise FetchError("timed out", "timeout")
    sock = None
    try:
        sock = socket.create_connection((ip, t.port), timeout=max(0.2, left))
        if t.scheme == "https":
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=t.host)
        conn = http.client.HTTPConnection(t.host, t.port)
        conn.sock = sock                      # already connected, to the checked IP
        conn.request("GET", t.target, headers=headers)
        resp = conn.getresponse()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        status = resp.status
        if status in _REDIRECT_STATUSES or not _type_ok(hdrs.get("content-type", "")):
            return status, hdrs, b"", False
        chunks: list = []
        total = 0
        truncated = False
        while True:
            left = deadline - clock()
            if left <= 0:
                raise FetchError("timed out", "timeout")
            sock.settimeout(max(0.2, left))
            chunk = resp.read(65536)
            if not chunk:
                break
            total += len(chunk)
            chunks.append(chunk)
            if total > max_bytes:
                truncated = True
                break
        body = b"".join(chunks)
        return status, hdrs, body[:max_bytes], truncated
    except FetchError:
        raise
    except socket.timeout:
        raise FetchError("timed out", "timeout") from None
    except ssl.SSLError as e:
        raise FetchError("the site's certificate or TLS failed (%s)" % type(e).__name__,
                         "tls") from None
    except (OSError, http.client.HTTPException) as e:
        raise FetchError("could not connect (%s)" % type(e).__name__, "connect") from None
    finally:
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass


def safe_get(url, *, max_bytes: int = DEFAULT_MAX_BYTES,
             timeout: float = DEFAULT_TIMEOUT_S,
             max_redirects: int = DEFAULT_MAX_REDIRECTS,
             accept: str = "text/html,application/xhtml+xml,application/json;q=0.9,"
                           "text/plain;q=0.8,*/*;q=0.1",
             resolver: Optional[Callable] = None,
             opener: Optional[Callable] = None,
             clock: Callable[[], float] = time.monotonic) -> Fetched:
    """GET ``url`` as a public web page, following up to ``max_redirects``
    redirects with EVERY hop vetted again by check_url and the connection made
    to the address that was vetted. No cookies, no credentials, plain
    User-Agent. Raises UrlRefused (a hop is not public) or FetchError.

    ``opener(target, ip, headers, deadline, max_bytes, clock)`` returns
    ``(status, headers, body, truncated)``; the default makes the pinned
    socket connection. ``resolver`` as in check_url."""
    deadline = clock() + float(timeout)
    open_one = opener or _open_pinned
    current = url
    for hop in range(int(max_redirects) + 1):
        t = check_url(current, resolver=resolver)
        default = 443 if t.scheme == "https" else 80
        headers = {
            "Host": t.host if t.port == default else "%s:%d" % (t.host, t.port),
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Accept-Encoding": "identity",
            "Accept-Language": "en",
            "Connection": "close",
        }
        status, hdrs, body, truncated = _try_ips(open_one, t, headers, deadline,
                                                 int(max_bytes), clock)
        if status in _REDIRECT_STATUSES:
            loc = hdrs.get("location", "")
            if not loc:
                raise FetchError("a redirect with nowhere to go", "redirect")
            current = urljoin(t.url, loc)
            continue
        return Fetched(status, t.url, hdrs, body, truncated, hop,
                       (hdrs.get("content-type", "") or "").split(";", 1)[0].strip().lower())
    raise FetchError("too many redirects", "redirect")


def _try_ips(open_one, t: Target, headers: dict, deadline: float,
             max_bytes: int, clock):
    """Try the vetted addresses in order; the first that connects serves.
    Only a failure to CONNECT moves on; a timeout or a response ends it."""
    last: Optional[FetchError] = None
    for ip in t.ips:
        try:
            return open_one(t, ip, headers, deadline, max_bytes, clock)
        except FetchError as e:
            last = e
            if e.code != "connect":
                raise
    raise last or FetchError("no address to connect to", "connect")


def public_url(url) -> bool:
    """True when check_url accepts ``url`` (resolving the name). Never raises."""
    try:
        check_url(url)
        return True
    except Exception:
        return False
