"""Shared tiny text/number helpers for skills (clamp, titlecase, spoken duration).

Canonical copies of helpers that were duplicated verbatim. Modules import them
under the same local names so `mock.patch.object(mod, "_name")` keeps working.
"""


def _clamp(v, lo, hi):
    try:
        v = type(lo)(v)
    except Exception:
        return lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _titlecase(s: str) -> str:
    small = {"and", "or", "of", "the", "in", "on", "for", "to", "a", "an"}
    parts = (s or "").split()
    out = []
    for i, w in enumerate(parts):
        if i > 0 and w in small:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _format_seconds(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f"{s} second{'s' if s != 1 else ''}"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}"
    h, m = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"
