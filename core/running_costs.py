"""Running costs — what it actually costs to keep JARVIS going.

Answers "how much does it cost to run you" with a real breakdown instead of
reading the Anthropic billing page (that is ``check_credits``: the BALANCE):

  * Electricity, an ESTIMATE: the live GPU power draw from ``nvidia-smi
    --query-gpu=power.draw`` (summed over every card) plus a CPU figure
    estimated from utilisation, times the hours JARVIS has been running today
    and this month, times ``core.config.ELECTRICITY_RATE_PER_KWH``. The hours
    come from what JARVIS already keeps: the live session's start
    (``bobert_companion._session_start_time``) plus the start/end stamps of
    past sessions in memory.py's checkpointed session-summary index.
  * Cloud: this session's real Claude token usage (``core.llm_client``'s
    ``session_usage`` tally) and the calendar month's (``core.llm_usage``'s
    persisted tally), both priced from ``core.model_catalog``. Until a month
    tally exists the reply says so and points at check_credits.
  * A one-line verdict over the month when the month tally exists, else over
    this session; electricity and cloud always on the same window.

Import-light + CI-safe: stdlib only at import; psutil, memory.py and the
monolith are read lazily and every reader degrades instead of raising. The
nvidia-smi spawn has a short timeout and CREATE_NO_WINDOW on Windows (the
core/gpu_usage.py pattern).
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
from typing import Iterable, List, Optional, Tuple

_SMI_TIMEOUT = 2.0      # nvidia-smi cold-start can be ~1 s on Windows

_NO_WINDOW = (subprocess.CREATE_NO_WINDOW
              if sys.platform == "win32" else 0)  # type: ignore[attr-defined]

# Fallback when core.config's ELECTRICITY_RATE_PER_KWH is unreadable or
# nonsense; mirrors its shipped default.
_DEFAULT_RATE_PER_KWH = 0.14

# CPU package power has no portable counter, so it is estimated linearly from
# utilisation between an idle floor and a desktop part's full-load figure.
CPU_IDLE_WATTS = 20.0
CPU_FULL_WATTS = 125.0

# Prompt-cache pricing relative to the model's input price (Anthropic list
# multipliers: cache reads 0.1x, 5-minute cache writes 1.25x).
_CACHE_READ_MULT = 0.10
_CACHE_WRITE_MULT = 1.25


# ─── electricity ─────────────────────────────────────────────────────────

def _run_power_query() -> Optional[str]:
    """nvidia-smi's power.draw CSV (one line per GPU), or None when the binary
    is missing, hangs past the timeout, or errors. Never raises."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_SMI_TIMEOUT,
            creationflags=_NO_WINDOW,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout or ""


def parse_power_draw(text: Optional[str]) -> Optional[float]:
    """Total watts over every GPU line ("85.32" per card). Lines a card cannot
    report ("[N/A]", "[Not Supported]") are skipped; None when no card gave a
    number."""
    total = 0.0
    seen = False
    for line in (text or "").splitlines():
        try:
            watts = float(line.strip())
        except ValueError:
            continue
        if not math.isfinite(watts) or watts < 0:
            continue
        total += watts
        seen = True
    return total if seen else None


def gpu_watts() -> Optional[float]:
    """Live draw of all NVIDIA GPUs in watts, or None with no reading."""
    return parse_power_draw(_run_power_query())


def _read_cpu_percent() -> Optional[float]:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=0.2))
    except Exception:
        return None


def cpu_watts(util_pct: Optional[float]) -> float:
    """Estimated CPU draw for a utilisation percent; the idle floor when the
    utilisation is unknown."""
    if util_pct is None:
        return CPU_IDLE_WATTS
    u = min(100.0, max(0.0, float(util_pct)))
    return CPU_IDLE_WATTS + (CPU_FULL_WATTS - CPU_IDLE_WATTS) * u / 100.0


def electricity_rate() -> float:
    """core.config.ELECTRICITY_RATE_PER_KWH, read live so a Settings /
    user_settings.json change applies. Missing, non-numeric or negative falls
    back to the default; 0 is honoured (free power)."""
    try:
        from core import config as _cfg
        rate = float(getattr(_cfg, "ELECTRICITY_RATE_PER_KWH",
                             _DEFAULT_RATE_PER_KWH))
    except Exception:
        return _DEFAULT_RATE_PER_KWH
    if not math.isfinite(rate) or rate < 0:
        return _DEFAULT_RATE_PER_KWH
    return rate


# ─── hours running ───────────────────────────────────────────────────────

def _session_start() -> Optional[float]:
    """When this JARVIS session started: the monolith's _session_start_time
    when it is loaded (never imported fresh — that has side effects), else
    this process's start time."""
    mod = sys.modules.get("bobert_companion")
    start = getattr(mod, "_session_start_time", None) if mod else None
    if isinstance(start, (int, float)) and not isinstance(start, bool):
        return float(start)
    try:
        import psutil
        return float(psutil.Process().create_time())
    except Exception:
        return None


def _parse_local(stamp) -> Optional[float]:
    try:
        return time.mktime(time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None


def _entry_span(entry) -> Optional[Tuple[float, float]]:
    """(start, end) epoch seconds of one session-summary entry, or None."""
    if not isinstance(entry, dict):
        return None
    start = _parse_local(entry.get("iso_start"))
    try:
        end = float(entry.get("ts"))
    except (TypeError, ValueError):
        end = _parse_local(entry.get("iso_end"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def persisted_session_spans() -> List[Tuple[float, float]]:
    """(start, end) of past sessions from memory.py's session-summary index
    (memory/session_summaries.json, checkpointed every few minutes). A session
    with no summary is absent, so this undercounts rather than guesses."""
    try:
        import memory as _pm
    except Exception:
        return []
    loader = getattr(_pm, "_load_sessions_file", None)
    if not callable(loader):
        return []
    try:
        entries = loader() or []
    except Exception:
        return []
    return [s for s in (_entry_span(e) for e in entries) if s]


def _overlap_hours(spans: Iterable[Tuple[float, float]],
                   lo: float, hi: float) -> float:
    return sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in spans) / 3600.0


def running_hours(now: float, session_start: Optional[float],
                  past_spans: Iterable[Tuple[float, float]]
                  ) -> Tuple[float, float, float]:
    """(this session, today, this month) hours JARVIS has been running, by the
    local calendar. Spans are merged first, so the live session's own
    checkpoint entry (or two overlapping instances) is not counted twice."""
    spans = [(s, min(e, now)) for s, e in past_spans if s < now]
    session_h = 0.0
    if session_start is not None and session_start < now:
        spans.append((session_start, now))
        session_h = (now - session_start) / 3600.0
    merged: List[List[float]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    lt = time.localtime(now)
    day0 = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    month0 = time.mktime((lt.tm_year, lt.tm_mon, 1, 0, 0, 0, 0, 0, -1))
    return (session_h, _overlap_hours(merged, day0, now),
            _overlap_hours(merged, month0, now))


# ─── cloud ───────────────────────────────────────────────────────────────

def _session_usage() -> dict:
    try:
        from core import llm_client
        return llm_client.session_usage_snapshot()
    except Exception:
        return {}


def _month_usage(now: Optional[float] = None) -> Optional[dict]:
    try:
        from core import llm_usage
        return llm_usage.month_usage(now)
    except Exception:
        return None


def cloud_cost(usage: dict) -> Tuple[float, int, int]:
    """(USD, priced calls, unpriced calls) for a {model: token row} tally as
    kept by core.llm_client.session_usage, priced from core.model_catalog.
    Cache reads/writes are billed at their multiples of the input price; a
    model the catalog does not list is counted but not priced."""
    from core import model_catalog
    usd = 0.0
    priced = unpriced = 0
    for model, row in (usage or {}).items():
        try:
            calls = int(row.get("calls", 0))
            entry = model_catalog.by_id(str(model))
            if entry is None:
                unpriced += calls
                continue
            eff_in = (row.get("input", 0)
                      + row.get("cache_read", 0) * _CACHE_READ_MULT
                      + row.get("cache_write", 0) * _CACHE_WRITE_MULT)
            usd += (eff_in * entry.in_price
                    + row.get("output", 0) * entry.out_price) / 1_000_000.0
            priced += calls
        except Exception:
            continue
    return usd, priced, unpriced


# ─── spoken report ───────────────────────────────────────────────────────

def _money(usd: float) -> str:
    if usd <= 0:
        return "nothing"
    if usd < 0.005:
        return "under a cent"
    cents = max(1, round(usd * 100))
    if cents < 100:
        return "about 1 cent" if cents == 1 else f"about {cents} cents"
    if usd < 100:
        return f"about ${usd:.2f}"
    return f"about ${usd:,.0f}"


def _rate(rate: float) -> str:
    if rate >= 1:
        return f"${rate:.2f}"
    cents = round(rate * 100, 1)
    return f"{cents:g} cents"


def _hours(h: float) -> str:
    if h < 1:
        mins = round(h * 60)
        if mins < 1:
            return "under a minute"
        return "1 minute" if mins == 1 else f"{mins} minutes"
    n = f"{round(h, 1):g}" if h < 10 else f"{round(h)}"
    return "1 hour" if n == "1" else f"{n} hours"


def _calls(n: int) -> str:
    return f"{n} call{'' if n == 1 else 's'}"


def compose(*, gpu_w: Optional[float], cpu_w: float, rate: float,
            session_h: float, today_h: float, month_h: float,
            cloud_usd: float, cloud_calls: int, unpriced_calls: int = 0,
            month_cloud_usd: Optional[float] = None, month_calls: int = 0,
            month_unpriced: int = 0) -> str:
    """The three spoken sentences: electricity, cloud, verdict. Plain text,
    rounded numbers, no markdown, and none of the failure-marker words that
    would stop the verbatim speaker voicing it. ``month_cloud_usd`` is None
    when there is no persisted month tally."""
    kw = ((gpu_w or 0.0) + cpu_w) / 1000.0
    if gpu_w is None:
        draw = (f"with no GPU power reading, the CPU alone is roughly "
                f"{round(cpu_w)} watts")
    else:
        draw = (f"the GPU is drawing about {round(gpu_w)} watts and the CPU "
                f"roughly {round(cpu_w)}")
    power = (f"Electricity is an estimate: {draw}, so at {_rate(rate)} a "
             f"kilowatt-hour that's {_money(kw * today_h * rate)} for "
             f"{_hours(today_h)} today and {_money(kw * month_h * rate)} for "
             f"{_hours(month_h)} this month.")
    calls = cloud_calls + unpriced_calls
    if month_cloud_usd is not None:
        if calls == 0:
            cloud = "No Claude calls yet this session"
        else:
            cloud = (f"Claude calls this session come to {_money(cloud_usd)} "
                     f"across {_calls(calls)}")
        m_calls = month_calls + month_unpriced
        if m_calls == 0:
            cloud += ", and none so far this month"
        else:
            cloud += (f", and {_money(month_cloud_usd)} across "
                      f"{_calls(m_calls)} this month at list prices")
        if month_unpriced:
            cloud += (f", {month_unpriced} of them on a model I have no "
                      f"price for")
        cloud += "."
        window, cloud_usd_w, power_usd = (
            "this month", month_cloud_usd, kw * month_h * rate)
    else:
        if calls == 0:
            cloud = ("The cloud has cost nothing this session, no Claude "
                     "calls yet")
        else:
            cloud = (f"Claude calls this session come to {_money(cloud_usd)} "
                     f"across {_calls(calls)}")
            if unpriced_calls:
                cloud += (f", {unpriced_calls} of them on a model I have no "
                          f"price for")
        cloud += ("; there's no month-to-date cloud tally yet, so for the "
                  "bill ask me to check your credits.")
        window, cloud_usd_w, power_usd = (
            "this session", cloud_usd, kw * session_h * rate)
    total = power_usd + cloud_usd_w
    if total < 0.005:
        verdict = f"Verdict: next to nothing so far {window}, sir."
    elif cloud_usd_w <= 0:
        verdict = (f"Verdict: {_money(total)} {window} so far, all of it "
                   f"electricity, sir.")
    else:
        main = "the cloud" if cloud_usd_w >= power_usd else "electricity"
        verdict = (f"Verdict: {_money(total)} {window} so far, mostly "
                   f"{main}, sir.")
    return f"{power} {cloud} {verdict}"


def report(now: Optional[float] = None) -> str:
    """The running_costs answer, from live readings."""
    now = time.time() if now is None else now
    session_h, today_h, month_h = running_hours(
        now, _session_start(), persisted_session_spans())
    usd, priced, unpriced = cloud_cost(_session_usage())
    month = _month_usage(now)
    m_usd, m_priced, m_unpriced = (cloud_cost(month) if month is not None
                                   else (None, 0, 0))
    return compose(gpu_w=gpu_watts(), cpu_w=cpu_watts(_read_cpu_percent()),
                   rate=electricity_rate(), session_h=session_h,
                   today_h=today_h, month_h=month_h, cloud_usd=usd,
                   cloud_calls=priced, unpriced_calls=unpriced,
                   month_cloud_usd=m_usd, month_calls=m_priced,
                   month_unpriced=m_unpriced)
