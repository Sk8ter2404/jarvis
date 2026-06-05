"""
News briefing skill for JARVIS.

Fetches headlines from a configurable list of RSS feeds (tech, world, local
weather alerts by default), optionally summarises each in one sentence via
the Claude backend, and returns a JARVIS-style spoken briefing.

Designed to be pulled into skills/morning_briefing.py and
skills/evening_briefing.py — turning the one-line weather greeting into an
actual morning/evening intelligence briefing.

Actions registered:
  news_briefing  — manually trigger. Returns the briefing text and queues
                   it for spoken delivery (with the briefing TTS preset).

Public helper for other skills:
  get_news_text() -> str
      Returns the briefing paragraph (no leading [intent:] tag), or "" if
      every feed failed. Callers that enqueue it themselves should prepend
      "[intent:briefing] " if they want the briefing TTS preset.

Config knobs (read live from bobert_companion at call time):
  NEWS_BRIEFING_ENABLED         bool, default True
  NEWS_BRIEFING_FEEDS           list[str | dict], default = built-in BBC + NWS
  NEWS_BRIEFING_HEADLINE_COUNT  int, default 3       (total across all feeds)
  NEWS_BRIEFING_TIMEOUT         float, default 6.0   (per-feed HTTP timeout)
  NEWS_BRIEFING_SUMMARIZE       bool, default True   (LLM rewrite; off = use feed title verbatim)
  NEWS_BRIEFING_CACHE_MINUTES   int, default 30      (in-process feed cache TTL)

Feed entries may be either a URL string ("https://...") or a dict
{"name": "tech", "url": "https://..."} — the optional name is used in the
spoken intro ("from technology...") when multiple feeds contribute.
"""
from __future__ import annotations

import html
import importlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from xml.etree import ElementTree as ET

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

# Ensure the project root is importable so `core.atomic_io` resolves whether
# this module is loaded as `skills.news_briefing` or run directly.
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

# Default feed list — picked because they're free, no API key, and stable.
# Each entry has a friendly name used in the spoken briefing.
_DEFAULT_FEEDS: list[dict] = [
    {"name": "technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    {"name": "world",      "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "science",    "url": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"},
]

_DEFAULT_HEADLINE_COUNT  = 3
_DEFAULT_TIMEOUT         = 6.0
_DEFAULT_SUMMARIZE       = True
_DEFAULT_CACHE_MINUTES   = 30

# Hard wall-clock cap on each per-headline summarisation call. Without it a
# single hung Anthropic request stalls the whole briefing (each headline is
# summarised sequentially). Matches phone_bridge / email_triage passing an
# explicit timeout= to messages.create().
_LLM_TIMEOUT_SECONDS     = 30.0

_USER_AGENT = "jarvis/1.0 (news_briefing)"

_speech_lock = threading.Lock()
_cache_lock  = threading.Lock()
_feed_cache: dict[str, dict] = {}   # url -> {"fetched_at": ts, "items": [...]}


# ─── config ──────────────────────────────────────────────────────────────

def _config(name: str, default):
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return default
    return getattr(bc, name, default)


def _read_config() -> dict:
    feeds_raw = _config("NEWS_BRIEFING_FEEDS", _DEFAULT_FEEDS)
    # Normalise each feed entry to {"name": str, "url": str}.
    feeds: list[dict] = []
    for f in (feeds_raw or []):
        if isinstance(f, str):
            feeds.append({"name": "", "url": f})
        elif isinstance(f, dict) and f.get("url"):
            feeds.append({"name": (f.get("name") or "").strip(), "url": f["url"]})
    return {
        "enabled":   bool(_config("NEWS_BRIEFING_ENABLED",       True)),
        "feeds":     feeds,
        "count":     max(1, int(_config("NEWS_BRIEFING_HEADLINE_COUNT", _DEFAULT_HEADLINE_COUNT))),
        "timeout":   float(_config("NEWS_BRIEFING_TIMEOUT",      _DEFAULT_TIMEOUT)),
        "summarize": bool(_config("NEWS_BRIEFING_SUMMARIZE",     _DEFAULT_SUMMARIZE)),
        "ttl":       max(0, int(_config("NEWS_BRIEFING_CACHE_MINUTES", _DEFAULT_CACHE_MINUTES))) * 60,
    }


# ─── speech queue (atomic write, matches morning/evening) ────────────────

def _enqueue_speech(message: str) -> None:
    """Route a proactive briefing through bobert_companion's public
    proactive_announce() API — the canonical helper for pending_speech.json —
    falling back to a direct atomic write if the parent module hasn't loaded
    yet (e.g. unit test, import-time skill registration before bobert_companion
    finishes initialising)."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="news")
            return
    except Exception:
        pass

    with _speech_lock:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw:
                    try:
                        decoded, _ = json.JSONDecoder().raw_decode(raw)
                        if isinstance(decoded, list):
                            data = decoded
                    except Exception:
                        data = []
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            print(f"  [news] speech-queue write failed ({e}); briefing: {message}")


# ─── feed fetching ───────────────────────────────────────────────────────

_TAG_RE       = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _parse_with_feedparser(text: str) -> list[dict]:
    try:
        import feedparser  # type: ignore
    except Exception:
        return []
    try:
        parsed = feedparser.parse(text)
    except Exception:
        return []
    out: list[dict] = []
    for entry in (parsed.entries or []):
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        desc = (entry.get("summary") or entry.get("description") or "")
        out.append({"title": _strip_html(title), "description": _strip_html(desc)})
    return out


def _parse_with_stdlib(text: str) -> list[dict]:
    """Tolerant RSS/Atom parser using stdlib ElementTree. Handles both
    plain RSS 2.0 (<item><title/><description/></item>) and Atom
    (<entry><title/><summary/></entry>)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    out: list[dict] = []

    # Strip XML namespaces from tags so we can match by localname.
    def localname(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for el in root.iter():
        name = localname(el.tag).lower()
        if name not in ("item", "entry"):
            continue
        title_text = ""
        desc_text = ""
        for child in el:
            cname = localname(child.tag).lower()
            text = (child.text or "").strip()
            if not text and cname in ("title", "description", "summary", "content"):
                # Atom <content type="html">…</content> may have HTML children
                text = "".join(child.itertext()).strip()
            if cname == "title" and not title_text:
                title_text = text
            elif cname in ("description", "summary", "content") and not desc_text:
                desc_text = text
        if not title_text:
            continue
        out.append({"title": _strip_html(title_text), "description": _strip_html(desc_text)})

    return out


def _fetch_feed(url: str, timeout: float) -> list[dict]:
    """Return a list of {title, description} entries from `url`, newest
    first. Returns [] on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        print(f"  [news] feed http {e.code} from {url}: {e.reason}")
        return []
    except Exception as e:
        print(f"  [news] feed fetch failed for {url}: {e}")
        return []

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return []

    items = _parse_with_feedparser(text)
    if not items:
        items = _parse_with_stdlib(text)
    return items


def _fetch_feed_cached(url: str, timeout: float, ttl: float) -> list[dict]:
    with _cache_lock:
        cached = _feed_cache.get(url)
    if cached and (time.time() - cached["fetched_at"]) < ttl:
        return list(cached["items"])
    items = _fetch_feed(url, timeout)
    if items:
        with _cache_lock:
            _feed_cache[url] = {"fetched_at": time.time(), "items": items}
    elif cached:
        # Fetch failed but we have a stale cache — better that than nothing.
        return list(cached["items"])
    return items


# ─── LLM summarisation ───────────────────────────────────────────────────

_uname = os.getenv("JARVIS_USER_NAME", "").strip() or "the user"
_SUMMARY_SYSTEM = (
    f"You are JARVIS distilling news headlines for {_uname}. Rewrite the given "
    "headline plus optional description into ONE concise spoken sentence "
    "(maximum 20 words) in JARVIS's measured, slightly formal register. "
    "No preamble, no quoting, no '[intent:]' tag — just the sentence. If the "
    "headline is purely a teaser ('Click here'), return the title verbatim."
)


def _summarize_via_llm(title: str, description: str) -> str:
    """Use the same Anthropic SDK call pattern the rest of JARVIS uses.
    Returns title verbatim on any error, so a flaky LLM never blocks news."""
    try:
        bc = sys.modules.get("bobert_companion") or importlib.import_module("bobert_companion")
    except Exception:
        return title

    user_text = f"Headline: {title}"
    if description:
        # Keep the prompt small — long descriptions burn tokens without
        # improving a one-sentence summary.
        user_text += f"\nDescription: {description[:500]}"

    def _clean(out: str) -> str:
        out = (out or "").strip()
        # Defensive: strip any stray [intent:xxx] tag in case the model
        # disregarded the system prompt.
        out = re.sub(r"^\s*\[\s*intent\s*:\s*[a-z_]+\s*\]\s*", "", out, flags=re.IGNORECASE)
        return out or title

    # Primary path: Claude (keeps cost low). Only when the backend is Claude
    # and the SDK imports.
    if getattr(bc, "AI_BACKEND", "") == "claude":
        try:
            import anthropic  # type: ignore
            model = getattr(bc, "CLAUDE_MODEL", "claude-sonnet-4-6")
            msg = anthropic.Anthropic().messages.create(
                model=model,
                max_tokens=120,
                system=_SUMMARY_SYSTEM,
                messages=[{"role": "user", "content": user_text}],
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            return _clean(msg.content[0].text)
        except Exception as e:
            print(f"  [news] LLM summary failed, trying local: {e}")

    # Fallback path: local Ollama model (used while the Claude API is capped,
    # when the backend isn't Claude, or on any error). Same cleaning + return
    # shape; on None we keep the degraded raw-title fallback.
    try:
        local = bc._call_local_llm(
            _SUMMARY_SYSTEM, [{"role": "user", "content": user_text}], max_tokens=120)
        if local:
            return _clean(local)
    except Exception as e:
        print(f"  [news] local summary failed: {e}")
    return title


# ─── briefing assembly ───────────────────────────────────────────────────

def _gather_headlines(cfg: dict) -> list[dict]:
    """Round-robin one headline per feed until we hit cfg['count']."""
    per_feed: list[list[dict]] = []
    for f in cfg["feeds"]:
        items = _fetch_feed_cached(f["url"], cfg["timeout"], cfg["ttl"])
        per_feed.append([{**it, "feed_name": f["name"]} for it in items])

    out: list[dict] = []
    idx = 0
    while len(out) < cfg["count"] and any(per_feed):
        if per_feed[idx]:
            out.append(per_feed[idx].pop(0))
        idx = (idx + 1) % len(per_feed)
        # Stop if all feeds are exhausted
        if not any(per_feed):
            break
    return out


def get_news_text() -> str:
    """Build the spoken news briefing paragraph. No [intent:] tag — callers
    decide whether to prepend one. Returns '' if every feed failed or news
    is disabled."""
    cfg = _read_config()
    if not cfg["enabled"]:
        return ""
    if not cfg["feeds"]:
        return ""

    headlines = _gather_headlines(cfg)
    if not headlines:
        return ""

    sentences: list[str] = []
    for h in headlines:
        if cfg["summarize"]:
            line = _summarize_via_llm(h["title"], h.get("description", ""))
        else:
            line = h["title"]
        line = line.strip().rstrip(".")
        if not line:
            continue
        sentences.append(line + ".")

    if not sentences:
        return ""

    intro = "Today's headlines, sir. "
    return intro + " ".join(sentences)


# ─── action registration ─────────────────────────────────────────────────

def news_briefing(_: str = "") -> str:
    text = get_news_text()
    if not text:
        return "No news feeds responded, sir."
    _enqueue_speech("[intent:briefing] " + text)
    return text


def register(actions):
    actions["news_briefing"] = news_briefing
    cfg = _read_config()
    if cfg["enabled"]:
        feed_count = len(cfg["feeds"])
        print(f"  [news] briefing ready ({feed_count} feeds, {cfg['count']} headlines, "
              f"summarize={cfg['summarize']})")
    else:
        print("  [news] NEWS_BRIEFING_ENABLED is False — briefing disabled")


# ─── manual smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover - manual smoke entry; run by hand, not under unittest
    print(get_news_text() or "(empty briefing)")
