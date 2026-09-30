"""Deterministic "what time is it in <place>" answers, and a guard for replies
that state a clock time for a place (2026-09-29).

Live v2.0.140 (Tue Sep 29 2026, 22:17 US Central): "what time is it in London"
ran get_time, which reads the LOCAL clock, and the LLM voiced its result as
"It is 10:17 PM in London, sir." London was at 4:17 AM on Wednesday. Nothing
here asks a model: the time there is computed with zoneinfo from a built-in
map of common cities, countries, US states and zones.

Entry points:

  * ``answer(text, now) -> ClockAnswer | None`` — the fast path. Understands
    "what time is it in <place>", "what's the time in <place>", "time in
    <place>" (plus "current" / "local", "right now", "can you tell me" and
    the other politeness date_math.normalize strips). Replies "It's 4:17 AM
    in London, sir. That's Wednesday there." — the day note only when the
    date there differs from the local date. A zone is named without "in":
    "It's 11:17 PM Eastern time, sir."
  * ``check_time_claim(reply, now, question=None) -> ClaimCheck | None`` —
    the reply guard (bobert_companion.parse_and_run_actions). When a reply
    STATES the time right now for a known place ("It is 10:17 PM in London",
    "In Tokyo, it's ...", "The time in Paris is ...", "It's 4:17 AM London
    time"), the time is checked against that place's real time. None = no
    such claim. Never a claim: a conversion, a condition or a plan ("When
    it's 9 AM here, it's 3 PM in London", "it'll be 2 AM in London", "at
    9 AM here it's 3 PM in London": a sentence with a second clock time that
    is not the time here now), an event's time ("your call ... it's 3 PM
    Eastern time"), another day, a place this map does not know even when it
    starts with a known word ("Eastern Europe", "Mountain View", "New South
    Wales time") or carries a qualifier that makes it another town
    ("Athens, Georgia", "Athens, GA", "London, Ontario", "Paris, Tenn."), or
    a clause in another sentence or clause ("It's 10:17 PM here, whereas in
    London ..."). The owner's question can rule the check out too: a
    conversion, a time difference, a condition or an event's time ("if it's
    9 AM here ...", "how far ahead is Tokyo", "what time does the match
    start") — but a plain "what time is it in London" is checked even when
    he says why ("... I want to call my mom", "is it too late to call").
    When in doubt the reply is left exactly as the LLM wrote it.
  * ``has_time_claim(text)`` — the same claim shapes, truth not checked (the
    streaming early-speech gate holds such a sentence back so the guard sees
    it before it is voiced).
  * ``resolve_place(name) -> Place | None``.

An unknown place (or "there", or two places at once) returns None, so the
normal turn handles it. Countries and US states with more than one zone are
mapped to the zone most of them keep (Texas -> Central, Florida -> Eastern);
the ones split near evenly (Tennessee, South Dakota), ambiguous names
(Georgia, Washington) and multi-zone countries ("the US", "Australia",
"Canada") are not answered on their own; a multi-zone country still works as
the second half of "Sydney, Australia". If zoneinfo or its time zone data is
unavailable (Windows needs the ``tzdata`` package), everything returns None.
``now`` is supplied by the caller (tests freeze it); a naive datetime is taken
as the machine's local time. Stdlib only, no I/O, never raises.
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from typing import NamedTuple, Optional

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except Exception:   # pragma: no cover - zoneinfo is stdlib since 3.9
    _ZoneInfo = None

from core.date_math import normalize


class Place(NamedTuple):
    key: str      # the normalised name that matched ("new york city")
    zone: str     # IANA zone ("America/New_York")
    label: str    # how the reply names it ("New York", "the UK", "UTC")
    is_zone: bool  # a time zone ("Eastern time", "UTC"), not a place


class ClockAnswer(NamedTuple):
    kind: str     # "world-clock"
    reply: str    # "It's 4:17 AM in London, sir. That's Wednesday there."


class ClaimCheck(NamedTuple):
    reply: str        # the reply to speak (the input when nothing was wrong)
    corrected: bool   # True when a stated time was wrong and was replaced
    places: tuple     # labels of the places whose time was stated
    # ``reply`` with the checked place-time claims blanked out: what the
    # caller's own local-time check should still scan. A correct "in London
    # it's 4 AM" must not vouch for a wrong "It's 10:02 PM here" in the same
    # reply. On a correction the true lines are the guard's own and are
    # blanked too (only leading [tags] remain).
    masked: str = ""


_WEEKDAY_TITLE = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday")

# ── the map ────────────────────────────────────────────────────────────────
# zone -> {spoken label: [aliases after _key(): lower case, no accents or
# punctuation]}. The label is how the reply names the place.
_PLACES = {
    "Europe/London": {
        "London": ["london", "manchester", "liverpool",
                   "edinburgh", "glasgow", "cardiff", "belfast"],
        "the UK": ["uk", "the uk", "u k", "the u k", "united kingdom",
                   "the united kingdom", "britain", "great britain"],
        "England": ["england"], "Scotland": ["scotland"],
        "Wales": ["wales"]},
    "Europe/Dublin": {"Dublin": ["dublin"], "Ireland": ["ireland"]},
    "Europe/Lisbon": {"Lisbon": ["lisbon"], "Portugal": ["portugal"]},
    "Europe/Paris": {"Paris": ["paris"], "France": ["france"]},
    "Europe/Berlin": {"Berlin": ["berlin", "munich", "frankfurt", "hamburg"],
                      "Germany": ["germany"]},
    "Europe/Madrid": {"Madrid": ["madrid", "barcelona"], "Spain": ["spain"]},
    "Europe/Rome": {"Rome": ["rome", "milan"], "Italy": ["italy"]},
    "Europe/Amsterdam": {"Amsterdam": ["amsterdam"],
                         "the Netherlands": ["netherlands", "the netherlands",
                                             "holland"]},
    "Europe/Brussels": {"Brussels": ["brussels"], "Belgium": ["belgium"]},
    "Europe/Zurich": {"Zurich": ["zurich"], "Geneva": ["geneva"],
                      "Switzerland": ["switzerland"]},
    "Europe/Vienna": {"Vienna": ["vienna"], "Austria": ["austria"]},
    "Europe/Prague": {"Prague": ["prague"],
                      "the Czech Republic": ["czech republic",
                                             "the czech republic", "czechia"]},
    "Europe/Warsaw": {"Warsaw": ["warsaw"], "Poland": ["poland"]},
    "Europe/Budapest": {"Budapest": ["budapest"], "Hungary": ["hungary"]},
    "Europe/Stockholm": {"Stockholm": ["stockholm"], "Sweden": ["sweden"]},
    "Europe/Oslo": {"Oslo": ["oslo"], "Norway": ["norway"]},
    "Europe/Copenhagen": {"Copenhagen": ["copenhagen"],
                          "Denmark": ["denmark"]},
    "Europe/Helsinki": {"Helsinki": ["helsinki"], "Finland": ["finland"]},
    "Europe/Athens": {"Athens": ["athens"], "Greece": ["greece"]},
    "Europe/Istanbul": {"Istanbul": ["istanbul"], "Turkey": ["turkey"]},
    "Europe/Kyiv": {"Kyiv": ["kyiv", "kiev"], "Ukraine": ["ukraine"]},
    "Europe/Moscow": {"Moscow": ["moscow", "saint petersburg",
                                "st petersburg"]},
    "Atlantic/Reykjavik": {"Reykjavik": ["reykjavik"],
                           "Iceland": ["iceland"]},
    "Africa/Cairo": {"Cairo": ["cairo"], "Egypt": ["egypt"]},
    "Africa/Lagos": {"Lagos": ["lagos"], "Nigeria": ["nigeria"]},
    "Africa/Nairobi": {"Nairobi": ["nairobi"], "Kenya": ["kenya"]},
    "Africa/Johannesburg": {"Johannesburg": ["johannesburg"],
                            "Cape Town": ["cape town"],
                            "South Africa": ["south africa"]},
    "Asia/Jerusalem": {"Jerusalem": ["jerusalem"], "Tel Aviv": ["tel aviv"],
                       "Israel": ["israel"]},
    "Asia/Riyadh": {"Riyadh": ["riyadh"], "Saudi Arabia": ["saudi arabia"]},
    "Asia/Dubai": {"Dubai": ["dubai"], "Abu Dhabi": ["abu dhabi"],
                   "the UAE": ["uae", "the uae", "united arab emirates",
                               "the united arab emirates"]},
    "Asia/Tehran": {"Tehran": ["tehran"], "Iran": ["iran"]},
    "Asia/Karachi": {"Karachi": ["karachi"], "Pakistan": ["pakistan"]},
    "Asia/Kolkata": {"Mumbai": ["mumbai", "bombay"],
                     "Delhi": ["delhi", "new delhi"],
                     "Bangalore": ["bangalore", "bengaluru"],
                     "Kolkata": ["kolkata", "calcutta"],
                     "Chennai": ["chennai"], "India": ["india"]},
    "Asia/Kathmandu": {"Kathmandu": ["kathmandu"], "Nepal": ["nepal"]},
    "Asia/Dhaka": {"Dhaka": ["dhaka"], "Bangladesh": ["bangladesh"]},
    "Asia/Bangkok": {"Bangkok": ["bangkok"], "Thailand": ["thailand"]},
    "Asia/Ho_Chi_Minh": {"Vietnam": ["vietnam", "viet nam"],
                         "Hanoi": ["hanoi"],
                         "Ho Chi Minh City": ["ho chi minh city", "saigon"]},
    "Asia/Singapore": {"Singapore": ["singapore"]},
    "Asia/Kuala_Lumpur": {"Kuala Lumpur": ["kuala lumpur"],
                          "Malaysia": ["malaysia"]},
    "Asia/Jakarta": {"Jakarta": ["jakarta"]},
    "Asia/Manila": {"Manila": ["manila"], "the Philippines": [
        "philippines", "the philippines"]},
    "Asia/Hong_Kong": {"Hong Kong": ["hong kong"]},
    "Asia/Shanghai": {"Beijing": ["beijing", "peking"],
                      "Shanghai": ["shanghai"], "China": ["china"]},
    "Asia/Taipei": {"Taipei": ["taipei"], "Taiwan": ["taiwan"]},
    "Asia/Seoul": {"Seoul": ["seoul"],
                   "South Korea": ["south korea", "korea"]},
    "Asia/Tokyo": {"Tokyo": ["tokyo", "osaka", "kyoto"], "Japan": ["japan"]},
    "Australia/Perth": {"Perth": ["perth"]},
    "Australia/Adelaide": {"Adelaide": ["adelaide"]},
    "Australia/Brisbane": {"Brisbane": ["brisbane"]},
    "Australia/Sydney": {"Sydney": ["sydney"], "Canberra": ["canberra"]},
    "Australia/Melbourne": {"Melbourne": ["melbourne"]},
    "Pacific/Auckland": {"Auckland": ["auckland"],
                         "Wellington": ["wellington"],
                         "New Zealand": ["new zealand"]},
    "Pacific/Honolulu": {"Honolulu": ["honolulu"], "Hawaii": ["hawaii"]},
    "America/Anchorage": {"Anchorage": ["anchorage"], "Alaska": ["alaska"]},
    "America/Los_Angeles": {
        "Los Angeles": ["los angeles", "la", "l a"],
        "San Francisco": ["san francisco"], "San Diego": ["san diego"],
        "Seattle": ["seattle"], "Las Vegas": ["las vegas"],
        "Sacramento": ["sacramento"], "California": ["california"],
        "Nevada": ["nevada"], "Oregon": ["oregon"],
        "Washington State": ["washington state"]},
    "America/Phoenix": {"Phoenix": ["phoenix"], "Tucson": ["tucson"],
                        "Arizona": ["arizona"]},
    "America/Denver": {"Denver": ["denver"], "Salt Lake City": [
        "salt lake city"], "Albuquerque": ["albuquerque"],
        "Colorado": ["colorado"], "Utah": ["utah"],
        "New Mexico": ["new mexico"], "Montana": ["montana"],
        "Wyoming": ["wyoming"], "Idaho": ["idaho"]},
    "America/Chicago": {
        "Chicago": ["chicago"], "Dallas": ["dallas"], "Houston": ["houston"],
        "Austin": ["austin"], "San Antonio": ["san antonio"],
        "Minneapolis": ["minneapolis"], "St. Louis": ["st louis",
                                                      "saint louis"],
        "Kansas City": ["kansas city"], "New Orleans": ["new orleans"],
        "Nashville": ["nashville"], "Memphis": ["memphis"],
        "Milwaukee": ["milwaukee"], "Texas": ["texas"],
        "Illinois": ["illinois"], "Minnesota": ["minnesota"],
        "Wisconsin": ["wisconsin"], "Iowa": ["iowa"], "Missouri": ["missouri"],
        "Arkansas": ["arkansas"], "Louisiana": ["louisiana"],
        "Mississippi": ["mississippi"], "Alabama": ["alabama"],
        "Oklahoma": ["oklahoma"], "Kansas": ["kansas"],
        "Nebraska": ["nebraska"], "North Dakota": ["north dakota"]},
    "America/New_York": {
        "New York": ["new york", "new york city", "nyc", "manhattan",
                     "brooklyn"],
        "Boston": ["boston"], "Philadelphia": ["philadelphia"],
        "Washington, D.C.": ["washington dc", "washington d c", "dc", "d c"],
        "Atlanta": ["atlanta"], "Miami": ["miami"], "Orlando": ["orlando"],
        "Detroit": ["detroit"], "Pittsburgh": ["pittsburgh"],
        "Charlotte": ["charlotte"], "Florida": ["florida"],
        "Massachusetts": ["massachusetts"],
        "Pennsylvania": ["pennsylvania"], "New Jersey": ["new jersey"],
        "Connecticut": ["connecticut"], "Rhode Island": ["rhode island"],
        "Vermont": ["vermont"], "New Hampshire": ["new hampshire"],
        "Maine": ["maine"], "Maryland": ["maryland"],
        "Delaware": ["delaware"], "Virginia": ["virginia"],
        "West Virginia": ["west virginia"],
        "North Carolina": ["north carolina"],
        "South Carolina": ["south carolina"], "Ohio": ["ohio"],
        "Michigan": ["michigan"], "Indiana": ["indiana"],
        "Kentucky": ["kentucky"]},
    "America/Toronto": {"Toronto": ["toronto"], "Montreal": ["montreal"],
                        "Ottawa": ["ottawa"]},
    "America/Vancouver": {"Vancouver": ["vancouver"]},
    "America/Edmonton": {"Calgary": ["calgary"], "Edmonton": ["edmonton"]},
    "America/Winnipeg": {"Winnipeg": ["winnipeg"]},
    "America/Halifax": {"Halifax": ["halifax"]},
    "America/Mexico_City": {"Mexico City": ["mexico city"]},
    "America/Cancun": {"Cancun": ["cancun"]},
    "America/Puerto_Rico": {"Puerto Rico": ["puerto rico"]},
    "America/Havana": {"Havana": ["havana"], "Cuba": ["cuba"]},
    "America/Bogota": {"Bogota": ["bogota"], "Colombia": ["colombia"]},
    "America/Lima": {"Lima": ["lima"], "Peru": ["peru"]},
    "America/Caracas": {"Caracas": ["caracas"], "Venezuela": ["venezuela"]},
    "America/Santiago": {"Santiago": ["santiago"]},
    "America/Argentina/Buenos_Aires": {"Buenos Aires": ["buenos aires"],
                                       "Argentina": ["argentina"]},
    "America/Sao_Paulo": {"Sao Paulo": ["sao paulo"],
                          "Rio de Janeiro": ["rio de janeiro", "rio"]},
}

# Zones spoken as zones ("It's 11:17 PM Eastern time, sir."): alias -> label.
_ZONES = {
    "America/New_York": ("Eastern time", ["eastern", "eastern time", "est",
                                          "edt", "us eastern"]),
    "America/Chicago": ("Central time", ["central", "central time", "cst",
                                         "cdt", "us central"]),
    "America/Denver": ("Mountain time", ["mountain", "mountain time", "mst",
                                         "mdt", "us mountain"]),
    "America/Los_Angeles": ("Pacific time", ["pacific", "pacific time", "pst",
                                             "pdt", "us pacific"]),
    "America/Anchorage": ("Alaska time", ["alaska time", "akst", "akdt"]),
    "Pacific/Honolulu": ("Hawaii time", ["hawaii time", "hst"]),
    "America/Halifax": ("Atlantic time", ["atlantic time"]),
    "Europe/Paris": ("Central European time", ["central european time",
                                                "cet", "cest"]),
    "Asia/Tokyo": ("Japan time", ["japan time", "jst"]),
    "UTC": ("UTC", ["utc", "u t c", "coordinated universal time", "zulu",
                    "zulu time"]),
    # GMT as a ZONE is UTC+0 all year (London itself is on BST in summer).
    "Etc/GMT": ("GMT", ["gmt", "g m t", "greenwich mean time"]),
}

# More than one zone: never answered alone, only as "<city>, <country>" —
# and only when the city's zone is one the country actually keeps. "London,
# Canada" is London, Ontario (Eastern), not the UK; "Perth, Russia" is not
# Western Australia. A mapped city whose zone is not in the country's set
# returns None, so the LLM answers instead of a confident foreign time.
_US_ZONES = frozenset((
    "America/New_York", "America/Chicago", "America/Denver", "America/Phoenix",
    "America/Los_Angeles", "America/Anchorage", "Pacific/Honolulu",
    "America/Puerto_Rico"))
_CONTAINERS = {
    **dict.fromkeys(
        ("us", "u s", "usa", "u s a", "the us", "the usa", "united states",
         "the united states", "united states of america", "america"),
        _US_ZONES),
    "canada": frozenset((
        "America/Toronto", "America/Vancouver", "America/Edmonton",
        "America/Winnipeg", "America/Halifax", "America/St_Johns",
        "America/Regina")),
    "australia": frozenset((
        "Australia/Perth", "Australia/Adelaide", "Australia/Brisbane",
        "Australia/Sydney", "Australia/Melbourne", "Australia/Darwin",
        "Australia/Hobart")),
    "brazil": frozenset((
        "America/Sao_Paulo", "America/Manaus", "America/Recife",
        "America/Fortaleza", "America/Belem", "America/Cuiaba",
        "America/Rio_Branco", "America/Noronha")),
    "russia": frozenset((
        "Europe/Moscow", "Europe/Kaliningrad", "Europe/Samara",
        "Asia/Yekaterinburg", "Asia/Omsk", "Asia/Novosibirsk",
        "Asia/Krasnoyarsk", "Asia/Irkutsk", "Asia/Yakutsk",
        "Asia/Vladivostok", "Asia/Magadan", "Asia/Kamchatka")),
    "mexico": frozenset((
        "America/Mexico_City", "America/Cancun", "America/Tijuana",
        "America/Monterrey", "America/Chihuahua", "America/Mazatlan",
        "America/Hermosillo")),
    "indonesia": frozenset((
        "Asia/Jakarta", "Asia/Pontianak", "Asia/Makassar", "Asia/Jayapura")),
    "tennessee": frozenset(("America/Chicago", "America/New_York")),
    "south dakota": frozenset(("America/Chicago", "America/Denver")),
}

# Region names that can qualify a city in a reply ("Athens, Georgia",
# "London, Ontario") even though most are not answerable on their own. Only
# the reply guard reads them: a city followed by one of these is checked as
# the pair, never as the bare city.
_US_STATES = frozenset((
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming"))
_CA_PROVINCES = frozenset((
    "ontario", "quebec", "british columbia", "alberta", "manitoba",
    "saskatchewan", "nova scotia", "new brunswick", "newfoundland",
    "prince edward island", "yukon", "nunavut", "northwest territories"))
# The postal abbreviations the LLM writes after a comma ("Athens, GA",
# "London, ON", "Chicago, IL"), mapped to the names above. Read only when
# written in capitals, so "in", "or", "me", "oh", "ok" in ordinary text are
# never a state. A dotted short form ("Ga.", "Tenn.", "Ont.") is not listed:
# the guard skips any capitalised qualifier it cannot place (fail closed).
_REGION_ABBREVIATIONS = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut",
    "DE": "delaware", "FL": "florida", "GA": "georgia", "HI": "hawaii",
    "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa",
    "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine",
    "MD": "maryland", "MA": "massachusetts", "MI": "michigan",
    "MN": "minnesota", "MS": "mississippi", "MO": "missouri",
    "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new hampshire",
    "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio",
    "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont",
    "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming",
    "ON": "ontario", "QC": "quebec", "BC": "british columbia",
    "AB": "alberta", "MB": "manitoba", "SK": "saskatchewan",
    "NS": "nova scotia", "NB": "new brunswick", "NL": "newfoundland",
    "PE": "prince edward island", "YT": "yukon", "NU": "nunavut",
    "NT": "northwest territories"}

# Zone renames between tzdata releases: try the newer name, then the older.
_ZONE_FALLBACKS = {"Europe/Kyiv": ("Europe/Kyiv", "Europe/Kiev")}


def _build_index() -> dict:
    idx = {}
    for zone, labels in _PLACES.items():
        for label, aliases in labels.items():
            for a in aliases:
                idx[a] = Place(a, zone, label, False)
    for zone, (label, aliases) in _ZONES.items():
        for a in aliases:
            idx[a] = Place(a, zone, label, True)
    return idx


_INDEX = _build_index()


def _key(name) -> str:
    """Lower case, accents off ("São Paulo" -> "sao paulo"), punctuation
    to spaces, "the " kept (aliases carry it where it belongs)."""
    s = _fold(name).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.replace("'", ""))
    return re.sub(r"\s+", " ", s).strip()


def resolve_place(name) -> Optional[Place]:
    """The Place for a spoken name, or None when it is unknown, ambiguous or
    a multi-zone country on its own. "<city> <region>" ("Paris France",
    "Sydney Australia", "Chicago Illinois") resolves to the city when the
    region is a known single-zone place in the same zone, or a multi-zone
    country that keeps the city's zone ("London Canada", "Moscow USA" and
    "Perth Russia" are None: a same-named town, not the mapped city)."""
    k = _key(name)
    if not k:
        return None
    for cand in (k, k[4:] if k.startswith("the ") else None):
        if cand and cand in _INDEX:
            return _INDEX[cand]
    words = k.split()
    for i in range(1, len(words)):
        city, region = " ".join(words[:i]), " ".join(words[i:])
        place = _INDEX.get(city)
        if place is None or place.is_zone:
            continue
        zones = _CONTAINERS.get(region)
        if zones is None and region.startswith("the "):
            zones = _CONTAINERS.get(region[4:])
        if zones is not None:
            return place if place.zone in zones else None
        other = _INDEX.get(region)
        if other is not None and not other.is_zone \
                and other.zone == place.zone:
            return place
    return None


def _zoneinfo(zone: str):
    if _ZoneInfo is None:
        return None
    for name in _ZONE_FALLBACKS.get(zone, (zone,)):
        try:
            return _ZoneInfo(name)
        except Exception:
            continue
    return None


def _aware(now):
    """``now`` as an aware datetime (naive = this machine's local time)."""
    if not isinstance(now, _dt.datetime):
        return None
    return now if now.tzinfo is not None else now.astimezone()


def time_at(place: Place, now) -> Optional[_dt.datetime]:
    """The wall-clock time at ``place`` for the instant ``now``, or None."""
    local = _aware(now)
    tz = _zoneinfo(place.zone) if isinstance(place, Place) else None
    if local is None or tz is None:
        return None
    return local.astimezone(tz)


def _clock(t: _dt.datetime) -> str:
    """ "4:17 AM" (no leading zero)."""
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def reply_for(place: Place, now) -> Optional[str]:
    """ "It's 4:17 AM in London, sir. That's Wednesday there." or None."""
    there = time_at(place, now)
    local = _aware(now)
    if there is None or local is None:
        return None
    where = place.label if place.is_zone else f"in {place.label}"
    out = f"It's {_clock(there)} {where}, sir."
    d_there, d_here = there.date(), local.date()
    if d_there != d_here:
        day = _WEEKDAY_TITLE[there.weekday()]
        if place.is_zone:
            out += f" That's {day} on {place.label}."
        elif d_there > d_here:
            out += f" That's {day} there."
        else:
            out += f" That's still {day} there."
    return out


# ── the question ───────────────────────────────────────────────────────────

_ASK_LEAD_RE = re.compile(
    r"^(?:(?:can|could|would|will) you(?: please)? tell me|"
    r"(?:please )?tell me|do you (?:happen to )?know|"
    r"i (?:want|need|would like|d like) to know|any idea)\b\s*")
_P = r"(?P<p>[a-z][a-z0-9 ]*?)"
_WHEN = r"(?:(?:right now|now|currently|at the moment) )?"
_THERE = r"(?:(?:over|out|down|up) )?"
_QUESTION_RES = tuple(re.compile(p) for p in (
    rf"what time (?:is it|would it be|will it be|it is) {_WHEN}{_THERE}"
    rf"in {_P}",
    rf"what is the (?:(?:current|local|exact) )?time {_WHEN}{_THERE}in {_P}",
    rf"(?:the )?(?:(?:current|local|exact) )?time in {_P}",
    rf"what (?:time is it|is the time) {_P} time",
    r"what time is it (?P<z>[a-z ]+?)",
))


def _fold(text) -> str:
    """Accents off ("São Paulo" -> "Sao Paulo"); non-strings -> ""."""
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKD", text)
    return "".join(c for c in s if not unicodedata.combining(c))


def _question_place(text) -> Optional[Place]:
    t = normalize(_fold(text))
    if not t or len(t) > 100:
        return None
    m = _ASK_LEAD_RE.match(t)
    if m:
        t = t[m.end():].strip()
    for rx in _QUESTION_RES:
        m = rx.fullmatch(t)
        if not m:
            continue
        if "z" in rx.groupindex:
            # "what time is it eastern" / "... pacific time": zones only.
            place = resolve_place(m.group("z"))
            if place is not None and place.is_zone:
                return place
            continue
        place = resolve_place(m.group("p"))
        if place is not None:
            return place
    return None


def answer(text, now) -> Optional[ClockAnswer]:
    """The spoken time at the place ``text`` asks about, or None (not a
    world-clock question, an unknown place, or no zone data). Never raises."""
    try:
        place = _question_place(text)
        if place is None:
            return None
        reply = reply_for(place, now)
        return ClockAnswer("world-clock", reply) if reply else None
    except Exception:
        return None


# ── the reply guard ────────────────────────────────────────────────────────
# A clause that STATES the time right now at a place. Four shapes; each needs
# a present-tense statement ("it's", "the time in X is"), so "the flight lands
# in London at 6 AM" is never a claim, and neither is "it'll be 2 AM in
# London" / "it would be": a conversion or a plan, not the time there now.
# The guard only ever REPLACES a reply that says what time it is now; anything
# that reads as a hypothetical, a conversion or an event's time is left alone
# (fail-open: the LLM's own words, exactly as before this guard existed).

_TIME = (r"(?P<h>\d{1,2})(?::(?P<m>[0-5]\d))?"
         r"(?:\s*(?P<ap>[ap])\.?\s?m\b\.?)?")
_PLACE = r"(?P<p>[A-Za-z][\w.'-]*(?:\s+[A-Za-z][\w.'-]*){0,3})"
_IT_IS = r"(?:it'?s|it\s+is)"
_HEDGE = (r"(?:(?:now|currently|already|just|about|around|approximately|"
          r"roughly|nearly|almost|exactly|precisely)\s+)*")
# A connective starts another clause: "It's 10:17 PM here, whereas in London
# ..." is two statements, never 10:17 PM for London.
_STOP = (r"(?:when|while|whilst|whereas|meanwhile|though|although|however|"
         r"but|and|yet|where|as|before|after|until|if|so|versus|vs|"
         r"compared|plus)")
# The LOCAL clock: "It's 10:17 PM here / for you / your time, over in London
# ..." states the time here, whatever place comes after it. "here in
# Chicago" (no break) names the place itself and stays a claim on it.
_LOCAL = r"(?:(?:you|your|yours|locally)\b|here\b(?!\s+in\b))"
# A free word between the time and "in <place>" ("on Tuesday", "right now")
# is never a connective, a local marker or ANOTHER clock time: "It's 10 PM
# here, 4 AM in London" must not credit 10 PM to London.
_FREE = (rf"(?!{_STOP}\b)(?!{_LOCAL})(?!\d{{1,2}}\s*[ap]\.?\s?m\b)"
         rf"(?![ap]\.?m\b)[\w']+")
# "in <place>, it's <time>" opens its OWN claim (shape 1): the time before it
# ("It's 10:17 PM, in London, it's 4:17 AM") is not that place's.
_OPENS_CLAIM = (rf"(?!\s*,?\s+(?:{_IT_IS}|the\s+(?:local\s+)?time\s+is)"
                rf"\s+{_HEDGE}\d)")
_CLAIM_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    # "It is 10:17 PM in London", "it's 4:17 AM on Wednesday in Tokyo"
    rf"\b{_IT_IS}\s+{_HEDGE}{_TIME}(?:,?\s+{_FREE}){{0,6}}?,?"
    rf"\s+in\s+(?:the\s+)?{_PLACE}{_OPENS_CLAIM}",
    # "In London, it's 4:17 AM"
    rf"\bin\s+(?:the\s+)?{_PLACE}\s*,?\s+(?:{_IT_IS}|the\s+(?:local\s+)?"
    rf"time\s+is)\s+{_HEDGE}{_TIME}",
    # "The (current / local) time in London is 4:17 AM"
    rf"\btime\s+in\s+(?:the\s+)?{_PLACE}\s+is\s+{_HEDGE}{_TIME}",
    # "It's 4:17 AM London time", "it's 11:17 PM Eastern time"
    rf"\b{_IT_IS}\s+{_HEDGE}{_TIME}\s+{_PLACE}\s+time\b",
))

# The claim's SENTENCE is about another moment: a condition, a conversion, a
# plan ("When it's 9 AM here, it's 3 PM in London", "... once you land").
_OTHER_MOMENT_RE = re.compile(
    r"\b(?:when|whenever|if|once|until|till|unless|before|after|"
    r"by\s+the\s+time|by\s+then|at\s+that\s+(?:point|time|moment)|"
    r"at\s+which\s+(?:point|time)|tomorrow|yesterday)\b", re.IGNORECASE)
# Any clock time in a sentence ("9 AM", "10:17", "4 o'clock", "noon"). A
# claim's sentence holding ANOTHER one that is not the time here now is a
# conversion ("at 9 AM here, it's 3 PM in London"; "Tokyo is 14 hours ahead,
# so at 8 AM your time it's 10 PM there"), never the time there now. The
# time here now beside it ("It's 10:17 PM here and in London it's 4:17 AM")
# keeps the claim: that is the two-place answer to "here and in London".
_ANY_TIME_RE = re.compile(
    r"\b(?P<h>\d{1,2})(?::(?P<m>[0-5]\d))?\s*"
    r"(?:(?P<ap>[ap])\.?\s?m\b|o'?clock\b)"
    r"|\b(?P<h2>\d{1,2}):(?P<m2>[0-5]\d)\b"
    r"|\b(?P<word>noon|midday|midnight)\b", re.IGNORECASE)
# The REPLY is about an event's time ("Your call with the Boston office?
# It's 3 PM Eastern time"): never read as the time there now.
_EVENT_RE = re.compile(
    r"\b(?:calls?|meetings?|appointments?|match|game|kick(?:s|ing)?\s+off|"
    r"kickoff|flights?|lands?|landing|arrives?|arrival|departs?|departure|"
    r"stream|remind|reminder|alarm|deadline|starts?|begins?|opens?|closes?|"
    r"scheduled?)\b", re.IGNORECASE)
# The owner's QUESTION asks about another moment, so the times in the reply
# are not "now" either. Three tiers (question_is_about_another_moment):
#   1. _QUESTION_CONVERSION_RE — always another moment: a clock time in the
#      question ("what time is 3 PM Eastern in London", "if it's 9 AM here
#      ..."), an offset or a difference ("in 3 hours", "time difference",
#      "how far ahead is Tokyo", "hours behind"), a conversion, a condition
#      on a moment ("when I land", "if it's ..."), or an event's time ("what
#      time does the match start", "what time is my call", "what time will
#      it be").
#   2. _QUESTION_NOW_RE — otherwise, a question that asks the time NOW
#      somewhere ("what time is it in London", "what's the time in Tokyo",
#      "is it too late to call London") is checked, whatever reason he adds
#      ("... I want to call my mom", "did the game start"). Live v2.0.140
#      this is the bug the guard exists for.
#   3. _QUESTION_OTHER_MOMENT_RE — anything else that talks about an event,
#      a schedule or another day is left alone.
_QUESTION_CONVERSION_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:[ap]\.?\s?m\b|o'?clock\b)|\b\d{1,2}:\d{2}\b"
    r"|\bin \d+ (?:hours?|minutes?|mins?)\b"
    r"|\b\d+ (?:hours?|minutes?|mins?) (?:from now|later|ago|earlier)\b"
    r"|\bconvert(?:ed|ing)?\b|\bconversion\b"
    r"|\b(?:time )?differences?\b|\btime ?zones?\b"
    r"|\b(?:hours?|how far|how much|how many hours) (?:ahead|behind)\b"
    r"|\b(?:ahead|behind) of\b|\boffset\b"
    r"|\b(?:if|when|whenever|once|by the time|until|till|unless)\s+"
    r"(?:it'?s|it\s+is|it\s+was|it'?ll|it\s+will|i|i'?m|we|you|he|she|they|"
    r"my|our|his|her|the)\b"
    r"|\bwhat\s+time\s+(?:will|would|should|could|can|did|does|do|was|"
    r"were|are)\b|\bwhat\s+time\s+is\s+(?!it\b)(?:the|my|our|his|her|your|"
    r"their|that|this)\b", re.IGNORECASE)
_QUESTION_NOW_RE = re.compile(
    r"\bwhat\s+time\s+(?:is\s+it|it\s+is)\b"
    r"|\bwhat(?:'?s|\s+is)\s+the\s+(?:(?:current|local|exact)\s+)?time\b"
    r"|\b(?:current|local)\s+time\b|\btime\s+(?:right\s+)?now\b"
    r"|\bthe\s+time\s+(?:(?:over|out|down|up)\s+)?(?:in|there)\b"
    r"|\btoo\s+(?:late|early)\s+(?:to|for)\b"
    r"|\bis\s+it\s+(?:still\s+)?(?:morning|afternoon|evening|night|"
    r"nighttime|daytime|late|early|dark|light)\b", re.IGNORECASE)
_QUESTION_OTHER_MOMENT_RE = re.compile(
    r"\b(?:if|when|whenever|once|until|till|before|after|by the time|"
    r"convert|converted|converting|conversion|tomorrow|tonight|yesterday|"
    r"next|last|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"calls?|meetings?|appointments?|flights?|land|lands|landing|arrive|"
    r"arrives|arrival|depart|departs|departure|remind|reminder|alarm|"
    r"schedule|scheduled|kick off|kicks off|kickoff|match|game|stream|"
    r"starts?|begins?|opens?|closes?)\b"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:[ap]\.?\s?m\b|o'?clock)|\b\d{1,2}:\d{2}\b"
    r"|\bin \d+ (?:hours?|minutes?|mins?)\b", re.IGNORECASE)
_WEEKDAY_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE)

# Words a place capture may run on into ("London right now", "London, sir"):
# dropping them keeps the place. Any OTHER dropped word means the capture
# named a longer place this map does not know ("Eastern Europe", "Mountain
# View", "Rio Grande Valley"), so the claim is not checked at all.
_TRAIL_FILLER = frozenset((
    "right", "now", "currently", "presently", "today", "tonight", "sir", "at",
    "the", "moment", "this", "morning", "afternoon", "evening", "night",
    "there", "already", "just", "and", "so", "which", "where", "while", "but",
    "or", "as", "over", "then", "yet", "still", "local"))
# A bare US-zone word after "in" is a region ("in Central Europe", "in the
# Pacific"), not a zone: only "<word> time" or an abbreviation names a zone.
_BARE_ZONE_WORDS = frozenset(("eastern", "central", "mountain", "pacific"))
# ", <Region>" right after the place: "Athens, Georgia", "London, Ontario".
_QUALIFIER_RE = re.compile(
    r"\s*,\s*(?:the\s+)?([A-Za-z][\w.'-]*(?:\s+[A-Za-z][\w.'-]*){0,2})")
# What may follow the place after a comma without making it another town
# ("London, sir", "London, Wednesday morning", "London, which is ...").
# Anything else that is capitalised and not a region this map can pair with
# the city ("Athens, Ga.", "Sydney, New South Wales") skips the claim.
_QUALIFIER_FILLER = _TRAIL_FILLER | frozenset((
    "madam", "its", "it", "that", "whereas", "meanwhile", "though",
    "although", "however", "give", "roughly", "about", "around",
    "approximately", "nearly", "almost", "exactly", "early", "late", "if",
    "when", "i", "you", "we", "they", "he", "she", "is", "was", "on", "in",
    "for", "to", "by", "of", "with", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december"))
# AM / PM as the "<X> time" shape may see them ("PM Eastern time").
_MERIDIEM_WORDS = frozenset(("am", "pm", "a m", "p m"))
# A sentence break: . ! ? then space and a capital (or a [tag]).
_SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*\s+")
_ABBREVIATIONS = frozenset(("st", "mt", "ft", "dr", "mr", "mrs", "ms", "jr",
                            "sr", "vs", "etc", "no", "approx"))


class _Claim(NamedTuple):
    place: Place
    minute: int        # claimed minute of the day
    precise: bool      # minutes were stated
    meridiem: bool     # AM/PM was stated
    start: int         # span of the claim in the text
    end: int


def _sentences(text: str):
    """(start, end) spans of the sentences in ``text``. A claim never crosses
    one: "It's 10:17 PM. In London, it's 4:17 AM." is two statements, and
    "PM." must not hand the local time to "In London". No break after an
    abbreviation ("in St. Louis") or an initial ("D.C."), but always after
    a.m. / p.m. followed by a capital ("It is 10:17 p.m. In London ...")."""
    spans = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt == "["):
            continue
        if text[m.start()] == ".":
            before = text[start:m.start()]
            word = re.search(r"([A-Za-z]+)$", before)
            meridiem = re.search(r"\b[ap]\.\s?m$", before, re.IGNORECASE)
            if word and not meridiem and (
                    word.group(1).lower() in _ABBREVIATIONS
                    or len(word.group(1)) == 1):
                continue
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _is_region(key: str) -> bool:
    if key in _US_STATES or key in _CA_PROVINCES or key in _CONTAINERS:
        return True
    place = _INDEX.get(key)
    return place is not None and not place.is_zone


def _claim_place(p: str, suffix_zone: bool):
    """(Place, the words that named it) for the capture ``p``, or None.

    The capture may run on past the place ("London right now"): the longest
    leading run of words that names a place wins, but only when every word
    dropped after it is filler — "Eastern Europe" is not "Eastern" and
    "Mountain View" is not "Mountain". ``suffix_zone`` (the "<X> time"
    shape) tries the trailing words instead ("PM Eastern" -> Eastern), under
    the same rule mirrored: only AM / PM or filler may be dropped in front,
    so "New South Wales time" is not Wales and "New England time" is not
    England."""
    words = p.split()
    if suffix_zone:
        for i in range(len(words)):
            place = resolve_place(" ".join(words[i:]))
            if place is None:
                continue
            if not all(_key(w) in _TRAIL_FILLER or _key(w) in _MERIDIEM_WORDS
                       for w in words[:i] if _key(w)):
                return None
            return place, words[i:]
        return None
    for i in range(len(words), 0, -1):
        place = resolve_place(" ".join(words[:i]))
        if place is None:
            continue
        if not all(_key(w) in _TRAIL_FILLER for w in words[i:] if _key(w)):
            return None
        if place.is_zone and place.key in _BARE_ZONE_WORDS:
            return None
        return place, words[:i]
    return None


def _qualifier_agrees(place: Place, span, after: str) -> bool:
    """False when the place is followed by ", <Region>" that makes it a
    different town ("Athens, Georgia", "Athens, GA", "Paris, Texas",
    "London, Ontario", "London, ON", "London, Canada"): the pair must
    resolve to the same zone ("Chicago, IL", "Paris, France" keep the
    claim). Fail closed: a capitalised qualifier that is neither filler
    ("London, Wednesday") nor a region the pair can be checked against
    ("Athens, Ga.", "Paris, Tenn.", "Sydney, New South Wales") also returns
    False, so the claim is skipped rather than read as the mapped city."""
    m = _QUALIFIER_RE.match(after)
    if not m:
        return True
    raw = m.group(1).split()
    abbr = _REGION_ABBREVIATIONS.get(raw[0]) if raw else None
    if abbr is not None:
        both = resolve_place(" ".join(list(span) + [abbr]))
        return both is not None and both.zone == place.zone
    words = _key(m.group(1)).split()
    for n in range(len(words), 0, -1):
        region = " ".join(words[:n])
        if _is_region(region):
            both = resolve_place(" ".join(list(span) + [region]))
            return both is not None and both.zone == place.zone
    first = _key(raw[0]) if raw else ""
    if not first or first in _QUALIFIER_FILLER:
        return True
    return not raw[0][:1].isupper()


def _other_day(sentence: str, place: Place, now) -> bool:
    """The sentence names a weekday that is neither today here nor today
    there ("It is 3 PM in London on Saturday"): another moment."""
    days = {d.lower() for d in _WEEKDAY_RE.findall(sentence)}
    if not days or now is None:
        return False
    local, there = _aware(now), time_at(place, now)
    if local is None or there is None:
        return False
    today = {_WEEKDAY_TITLE[local.weekday()].lower(),
             _WEEKDAY_TITLE[there.weekday()].lower()}
    return bool(days - today)


def _is_off(true_minute: int, claimed: int, precise: bool,
            meridiem: bool) -> bool:
    """True when ``claimed`` (minute of the day) is not ``true_minute``:
    more than 3 minutes off (45 without minutes), either half of the day
    when no AM/PM was said (and the hour is 12 or less)."""
    period = 1440 if meridiem or claimed >= 780 else 720
    diff = abs(true_minute - claimed) % period
    diff = min(diff, period - diff)
    return diff > (3 if precise else 45)


def _mention_minute(m):
    """(minute of the day, precise, meridiem) for an _ANY_TIME_RE match, or
    None when it is not a valid clock time."""
    word = (m.group("word") or "").lower()
    if word:
        return (0 if word == "midnight" else 720), True, True
    if m.group("h2") is not None:
        h, mins, ap = int(m.group("h2")), int(m.group("m2")), ""
    else:
        h, mins = int(m.group("h")), int(m.group("m") or 0)
        ap = (m.group("ap") or "").lower()
    if ap:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ap == "p" else 0)
    elif h > 23:
        return None
    return h * 60 + mins, (m.group("m") is not None
                           or m.group("h2") is not None), bool(ap)


def _another_clock_time(sentence: str, spans, now) -> bool:
    """True when ``sentence`` states a clock time outside every claim span
    in ``spans`` that is not the time here at ``now``: the sentence is a
    conversion ("at 9 AM here, it's 3 PM in London"), not the time there
    now. False without ``now`` (has_time_claim then errs toward a claim, so
    the early-speech gate still holds the sentence for the guard)."""
    local = _aware(now)
    if local is None:
        return False
    here = local.hour * 60 + local.minute
    for m in _ANY_TIME_RE.finditer(sentence):
        if any(a < m.end() and m.start() < b for a, b in spans):
            continue
        got = _mention_minute(m)
        if got is not None and _is_off(here, *got):
            return True
    return False


def _claims(text: str, now=None) -> list:
    """Every statement of the time NOW at a known place in ``text``, as
    _Claim tuples. ``now`` (optional) lets a named weekday that is not today
    here or there, or a second clock time that is not the time here now,
    disqualify its sentence."""
    out: list = []
    if not isinstance(text, str) or not text or _EVENT_RE.search(text):
        return out
    for s0, s1 in _sentences(text):
        sentence = text[s0:s1]
        if _OTHER_MOMENT_RE.search(sentence):
            continue
        found = _sentence_claims(sentence, s0, now)
        if found and _another_clock_time(
                sentence, [(c.start - s0, c.end - s0) for c in found], now):
            continue
        out.extend(found)
    return out


def _sentence_claims(sentence: str, s0: int, now) -> list:
    """The claims in one sentence (spans offset by ``s0``)."""
    out: list = []
    for i, rx in enumerate(_CLAIM_RES):
        for m in rx.finditer(sentence):
            if m.group("m") is None and m.group("ap") is None:
                continue    # "it's 4 in London": not a clock time
            got = _claim_place(m.group("p"), suffix_zone=(i == 3))
            if got is None:
                continue
            place, span = got
            if i != 3 and not _qualifier_agrees(
                    place, span, sentence[m.end("p"):]):
                continue
            if _other_day(sentence, place, now):
                continue
            h = int(m.group("h"))
            mins = int(m.group("m") or 0)
            ap = (m.group("ap") or "").lower()
            if ap:
                if not 1 <= h <= 12:
                    continue
                h = h % 12 + (12 if ap == "p" else 0)
            elif h > 23:
                continue
            out.append(_Claim(place, h * 60 + mins,
                              m.group("m") is not None, bool(ap),
                              s0 + m.start(), s0 + m.end()))
    return out


def _mask(text: str, claims) -> str:
    """``text`` with every claim's span blanked (same length)."""
    chars = list(text)
    for c in claims:
        for k in range(c.start, min(c.end, len(chars))):
            chars[k] = " "
    return "".join(chars)


def has_time_claim(text) -> bool:
    """True when ``text`` states the time now for a known place (true or
    not). Never raises."""
    try:
        return bool(_claims(text)) if isinstance(text, str) else False
    except Exception:
        return False


# Leading delivery tags ("[intent:confirmation]") survive a correction; an
# [ACTION: ...] token is never carried over from here (the caller decides).
_LEADING_TAGS_RE = re.compile(r"^(?:\s*\[(?!\s*action\s*:)[^\]]*\])+",
                              re.IGNORECASE)


def question_is_about_another_moment(question) -> bool:
    """True when the owner's question is a conversion, a time difference, a
    condition or an event ("if it's 9 AM here what time is it in London",
    "what time is 3 PM Eastern in London", "how far ahead is Tokyo", "remind
    me when it's 9 AM in London", "what time does the match start"): the
    times in the reply are not the time now. False for a question that asks
    the time now somewhere, whatever reason comes with it ("what time is it
    in London? I want to call my mom", "is it too late to call London") —
    see the tiers above _QUESTION_CONVERSION_RE. Never raises."""
    try:
        if not isinstance(question, str) or not question.strip():
            return False
        if _QUESTION_CONVERSION_RE.search(question):
            return True
        if _QUESTION_NOW_RE.search(question):
            return False
        return bool(_QUESTION_OTHER_MOMENT_RE.search(question))
    except Exception:
        return False


def check_time_claim(reply, now, question=None) -> Optional[ClaimCheck]:
    """Check every time-NOW ``reply`` states for a known place against that
    place's real time at ``now``. None when there is no such claim (or no
    zone data, or ``question`` — the owner's words this turn, optional — asks
    about another moment). Otherwise a ClaimCheck: ``corrected`` False and
    the reply untouched when every stated time is right (within 3 minutes;
    45 without minutes; either half of the day when no AM/PM was said), or
    ``corrected`` True and the reply replaced by the true line for each
    place, keeping any leading delivery [tags] (never an [ACTION:] token).
    ``masked`` is always set (see ClaimCheck). Never raises."""
    try:
        if not isinstance(reply, str) or not reply.strip():
            return None
        if question_is_about_another_moment(question):
            return None
        found = _claims(reply, now)
        if not found:
            return None
        wrong = False
        labels: list = []
        for place, claimed, precise, meridiem, _s, _e in found:
            there = time_at(place, now)
            if there is None:
                return None
            if place.label not in labels:
                labels.append(place.label)
            if _is_off(there.hour * 60 + there.minute, claimed, precise,
                       meridiem):
                wrong = True
        if not wrong:
            return ClaimCheck(reply, False, tuple(labels),
                              _mask(reply, found))
        lines = []
        for label in labels:
            place = next(c.place for c in found if c.place.label == label)
            line = reply_for(place, now)
            if line is None:
                return None
            lines.append(line)
        tags = _LEADING_TAGS_RE.match(reply)
        lead = tags.group(0).strip() + " " if tags else ""
        return ClaimCheck(lead + " ".join(lines), True, tuple(labels),
                          lead.strip())
    except Exception:
        return None
