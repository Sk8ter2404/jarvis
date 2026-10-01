"""Spoken arithmetic: "what's 12 times 7", "144 divided by 12", "2 to the
power of 10" — recognised, and answered exactly.

WHY THIS EXISTS (2026-10-01, the 09-05 live diagnostic)
=======================================================
"X times Y" never reached the calculator. The local prompt router loads the
PYTHON SANDBOX section (run_python, the calculator) only on the words
"calculate" / "compute" / "math" / "python" / "what's the square", so a plain
"what's 12 times 7" shipped no calculator grammar at all and the local model
answered from its head — the one thing it is unreliable at. Operator WORDS
were in no keyword list, and a bare "times" cannot be one: "what times does
the store open" is not arithmetic. What makes a turn arithmetic is an operator
with a NUMBER ON BOTH SIDES.

Two consumers:

* ``is_arithmetic_request(text)`` — a number, an operator, a number anywhere
  in the turn. core.prompt_router loads PYTHON SANDBOX on it, so a longer
  turn ("what's 12 times 7 plus the 20 percent tip") still puts run_python in
  front of the model.
* ``answer(text)`` — the whole utterance is one arithmetic question (an
  optional "what's" / "how much is" / "calculate" lead, the expression, an
  optional "equal" / "make" tail). It is evaluated EXACTLY (fractions, the
  usual precedence, right-associative powers) and returned as a finished
  spoken line. core.fast_paths runs it before the LLM, so the answer is
  instant and cannot be wrong.

Operators: times / multiplied by / x / * / ×, divided by / over / / / ÷,
plus / +, minus / -, to the power of / raised to (the power of) / to the Nth
(power) / ^ / **, and the postfix squared / cubed. Numbers: digits (with
decimals and thousands commas) or number words ("twelve", "two hundred and
fifty", "one point five", "a thousand"), optionally "negative".

Stdlib only, no I/O, never raises.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import List, NamedTuple, Optional, Tuple, Union

# ── number words ───────────────────────────────────────────────────────────

_UNITS = {w: i for i, w in enumerate(
    ("zero one two three four five six seven eight nine ten eleven twelve "
     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
_DIGIT_WORDS = {w: i for w, i in _UNITS.items() if i <= 9}
_NUMBER_WORDS = (set(_UNITS) | set(_TENS) | set(_SCALES)
                 | {"hundred", "and", "a", "point", "oh"})


def _words_to_int(words: List[str]) -> Optional[int]:
    """A whole number from number words ("two hundred and fifty", "a
    thousand", "twenty one"), or None when the run is not a well-formed
    number ("seven seven", "and five")."""
    if not words or words[-1] in ("and", "a"):
        return None
    total = 0
    group = 0            # the current < 1000 group
    last = None          # kind of the previous word
    for i, w in enumerate(words):
        if w == "a":
            # only "a hundred" / "a thousand" / "a million"
            if i + 1 >= len(words) or words[i + 1] not in (
                    "hundred", *tuple(_SCALES)):
                return None
            group = 1
            last = "a"
            continue
        if w == "and":
            if last not in ("hundred", "scale"):
                return None
            last = "and"
            continue
        if w in _UNITS:
            v = _UNITS[w]
            if last in ("unit", "teen"):
                return None
            if last == "tens" and v >= 10:
                return None
            if last == "tens" and v == 0:
                return None
            group += v
            last = "teen" if v >= 10 else "unit"
            continue
        if w in _TENS:
            if last in ("unit", "teen", "tens"):
                return None
            group += _TENS[w]
            last = "tens"
            continue
        if w == "hundred":
            if last not in ("unit", "a"):
                return None
            group = (group or 1) * 100
            last = "hundred"
            continue
        if w in _SCALES:
            if last in (None, "and", "scale"):
                return None
            total += (group or 1) * _SCALES[w]
            group = 0
            last = "scale"
            continue
        return None
    return total + group


def _parse_word_number(words: List[str]) -> Optional[Fraction]:
    """Number words, with an optional "point" + digit words decimal part
    ("one point five", "point two five")."""
    if "point" in words:
        k = words.index("point")
        whole_w, frac_w = words[:k], words[k + 1:]
        whole = _words_to_int(whole_w) if whole_w else 0
        if whole is None or not frac_w:
            return None
        digits = []
        for w in frac_w:
            if w == "oh":
                digits.append("0")
            elif w in _DIGIT_WORDS:
                digits.append(str(_DIGIT_WORDS[w]))
            else:
                return None
        return Fraction(whole) + Fraction(int("".join(digits)),
                                          10 ** len(digits))
    if "oh" in words:
        return None
    n = _words_to_int(words)
    return None if n is None else Fraction(n)


# ── tokens ─────────────────────────────────────────────────────────────────

_ADD, _SUB, _MUL, _DIV, _POW, _SQUARED, _CUBED = (
    "+", "-", "*", "/", "^", "squared", "cubed")
# Canonical spoken word for each operator, for the reply line.
_OP_WORDS = {_ADD: "plus", _SUB: "minus", _MUL: "times", _DIV: "divided by",
             _POW: "to the power of", _SQUARED: "squared", _CUBED: "cubed"}

# Multi-word operator phrases, longest first so "raised to the power of" wins
# over "raised to".
_OP_PHRASES: Tuple[Tuple[Tuple[str, ...], str], ...] = tuple(sorted((
    (("raised", "to", "the", "power", "of"), _POW),
    (("raised", "to", "the", "power"), _POW),
    (("to", "the", "power", "of"), _POW),
    (("to", "the", "power"), _POW),
    (("raised", "to", "the"), _POW),
    (("raised", "to"), _POW),
    (("multiplied", "by"), _MUL),
    (("divided", "by"), _DIV),
    (("times",), _MUL),
    (("x",), _MUL),
    (("*",), _MUL),
    (("over",), _DIV),
    (("/",), _DIV),
    (("plus",), _ADD),
    (("+",), _ADD),
    (("minus",), _SUB),
    (("-",), _SUB),
    (("^",), _POW),
    (("squared",), _SQUARED),
    (("cubed",), _CUBED),
), key=lambda p: -len(p[0])))

_DIGITS_RE = re.compile(r"^\d+(?:\.\d+)?$")
_ORDINAL_DIGITS_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")

Token = Union[Fraction, str]


def _prepare(text: str) -> str:
    """Lower-case and space the operator symbols out so they tokenise."""
    t = text.lower()
    t = t.translate(str.maketrans({"×": " x ", "÷": " / ", "−": "-",
                                   "’": "'", "‘": "'"}))
    t = t.replace("**", " ^ ")
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)          # 1,000 -> 1000
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " x ", t)          # 12x7 -> 12 x 7
    t = re.sub(r"([+*/^])", r" \1 ", t)
    # A hyphen between two numbers is a minus ("10-4"); a hyphen inside a
    # word ("twenty-one") is a space.
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)
    t = re.sub(r"\s*-\s*", " - ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokenize(words: List[str]) -> Optional[List[Token]]:
    """Numbers and operators, or None when any word is neither."""
    out: List[Token] = []
    i, n = 0, len(words)
    while i < n:
        w = words[i]
        # "(raised) to the 10th (power)" -> ^ 10
        ordinal = False
        for lead in (("raised", "to", "the"), ("to", "the")):
            k = len(lead)
            if tuple(words[i:i + k]) == lead and i + k < n:
                m = _ORDINAL_DIGITS_RE.match(words[i + k])
                if m:
                    out.append(_POW)
                    out.append(Fraction(int(m.group(1))))
                    i += k + 1
                    if i < n and words[i] == "power":
                        i += 1
                    ordinal = True
                    break
        if ordinal:
            continue
        matched = False
        for phrase, op in _OP_PHRASES:
            k = len(phrase)
            if tuple(words[i:i + k]) == phrase:
                out.append(op)
                i += k
                matched = True
                break
        if matched:
            continue
        neg = False
        if w == "negative" and i + 1 < n:
            neg = True
            i += 1
            w = words[i]
        if _DIGITS_RE.match(w):
            val = Fraction(w)
            i += 1
        else:
            j = i
            while j < n and words[j] in _NUMBER_WORDS:
                j += 1
            if j == i:
                return None
            val = _parse_word_number(words[i:j])
            if val is None:
                return None
            i = j
        out.append(-val if neg else val)
    return out


def _fold_signs(tokens: List[Token]) -> Optional[List[Token]]:
    """Turn a '-' that opens the expression or follows another operator into
    the sign of the number after it ("-5 times 3", "10 times -2")."""
    out: List[Token] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        prev_is_value = bool(out) and (isinstance(out[-1], Fraction)
                                       or out[-1] in (_SQUARED, _CUBED))
        if t == _SUB and not prev_is_value:
            if i + 1 < len(tokens) and isinstance(tokens[i + 1], Fraction):
                out.append(-tokens[i + 1])
                i += 2
                continue
            return None
        out.append(t)
        i += 1
    return out


def _well_formed(tokens: List[Token]) -> bool:
    """NUM (OP NUM | POSTFIX)* with at least one operator."""
    if not tokens or not isinstance(tokens[0], Fraction):
        return False
    expect_value = False
    ops = 0
    for t in tokens[1:]:
        if expect_value:
            if not isinstance(t, Fraction):
                return False
            expect_value = False
            continue
        if t in (_SQUARED, _CUBED):
            ops += 1
            continue
        if t in (_ADD, _SUB, _MUL, _DIV, _POW):
            ops += 1
            expect_value = True
            continue
        return False
    return ops > 0 and not expect_value


# ── evaluation ─────────────────────────────────────────────────────────────

class _Undefined(Exception):
    """Division by zero (or 0 to a negative power)."""


class _TooBig(Exception):
    """A result too large to say out loud."""


_MAX_EXPONENT = 1000
# Past this an answer is said as "M times 10 to the power of E", not as
# two dozen digits read one group at a time.
_HUGE = Fraction(10) ** 15         # a non-integer
_HUGE_INT = Fraction(10) ** 21     # an exact integer (seven comma groups)


def _power(base: Fraction, exp: Fraction) -> Fraction:
    if exp.denominator != 1:
        if base < 0:
            raise _TooBig("complex")
        try:
            return Fraction(float(base) ** float(exp))
        except (OverflowError, ValueError):
            raise _TooBig("overflow")
    e = exp.numerator
    if abs(e) > _MAX_EXPONENT:
        raise _TooBig("exponent")
    if base == 0 and e < 0:
        raise _Undefined()
    # Rough size check before computing: digits(base) * e.
    approx = max(len(str(abs(base.numerator))),
                 len(str(base.denominator))) * abs(e)
    if approx > 400:
        raise _TooBig("size")
    return base ** e


def _evaluate(tokens: List[Token]) -> Fraction:
    """Precedence: postfix squared/cubed and ^ (right-assoc) bind tightest,
    then * /, then + -."""
    # 1. postfix powers
    seq: List[Token] = []
    for t in tokens:
        if t in (_SQUARED, _CUBED):
            seq[-1] = _power(seq[-1], Fraction(2 if t == _SQUARED else 3))
        else:
            seq.append(t)
    # 2. ^ right-associative
    while _POW in seq:
        k = len(seq) - 1 - seq[::-1].index(_POW)
        seq[k - 1:k + 2] = [_power(seq[k - 1], seq[k + 1])]
    # 3. * / left to right
    out: List[Token] = [seq[0]]
    i = 1
    while i < len(seq):
        op, val = seq[i], seq[i + 1]
        if op == _MUL:
            out[-1] = out[-1] * val
        elif op == _DIV:
            if val == 0:
                raise _Undefined()
            out[-1] = out[-1] / val
        else:
            out.extend([op, val])
        i += 2
    # 4. + - left to right
    acc = out[0]
    for j in range(1, len(out), 2):
        acc = acc + out[j + 1] if out[j] == _ADD else acc - out[j + 1]
    return acc


# ── formatting ─────────────────────────────────────────────────────────────

def _say_number(v: Fraction) -> Tuple[str, bool]:
    """(spoken digits, exact). Integers exactly (comma-grouped from five
    digits); other values to at most four decimals, exact=False when
    rounded."""
    neg = v < 0
    a = -v if neg else v
    if a >= (_HUGE_INT if a.denominator == 1 else _HUGE):
        # Too long to read out digit by digit: "1.27 times 10 to the power
        # of 30" (exact=False -> "is about").
        e = len(str(int(a))) - 1
        m = float(a / Fraction(10) ** e)
        if m >= 9.995:
            m, e = m / 10, e + 1
        mant = f"{m:.2f}".rstrip("0").rstrip(".")
        exact = Fraction(mant) * Fraction(10) ** e == a
        s = f"{mant} times 10 to the power of {e}"
        return ("negative " + s if neg else s), exact
    if a.denominator == 1:
        n = a.numerator
        s = f"{n:,}" if n >= 10000 else str(n)
        exact = True
    else:
        r = round(a, 4)
        exact = (r == a)
        whole = int(r)
        s = f"{float(r):.4f}".rstrip("0").rstrip(".")
        if whole >= 10000:
            head, _, tail = s.partition(".")
            s = f"{int(head):,}" + (f".{tail}" if tail else "")
        if s in ("0", ""):
            s = "0"
    return ("negative " + s if neg and s != "0" else s), exact


def _say_operand(v: Fraction) -> str:
    return _say_number(v)[0]


def _say_expression(tokens: List[Token]) -> str:
    parts = []
    for t in tokens:
        parts.append(_OP_WORDS[t] if isinstance(t, str) else _say_operand(t))
    return " ".join(parts)


# ── grammar ────────────────────────────────────────────────────────────────

_LEAD_RE = re.compile(
    r"^(?:(?:hey|ok|okay|so|um|uh|well|oh|please|jarvis|sir|quick question)"
    r"\b[\s,]*)+")
_ASK_RE = re.compile(
    r"^(?:(?:can|could|would) you(?: please)? (?:tell me|work out|calculate|"
    r"compute|figure out)|(?:please )?tell me|do you know|"
    r"what(?:'s| is|s)|whats|how much is|how much|calculate|compute|"
    r"work out|figure out|solve|what does|what do|what will|what would)"
    r"\b\s*")
_TAIL_RE = re.compile(
    r"\s*(?:\b(?:equal|equals|make|makes|come to|comes to|give me|give|be|is|"
    r"please|jarvis|sir|exactly|again|for me|then)\b\s*)+$")
_SENT_PUNCT_RE = re.compile(r"[?!.,;:]+(?=\s|$)")


class MathAnswer(NamedTuple):
    expression: str     # canonical spoken form, "12 times 7"
    value: Optional[Fraction]   # None when undefined
    reply: str          # the finished spoken line


def _core_expression(text: str) -> Optional[str]:
    t = _prepare(text)
    t = _SENT_PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    prev = None
    while prev != t:
        prev = t
        t = _LEAD_RE.sub("", t).strip()
        t = _ASK_RE.sub("", t, count=1).strip()
        t = _TAIL_RE.sub("", t).strip()
    return t or None


def answer(text) -> Optional[MathAnswer]:
    """The exact answer to a whole-utterance arithmetic question, or None
    when ``text`` is not one (anything left over that is neither a number
    nor an operator means it is not plain arithmetic, and the turn goes to
    the LLM — which now sees run_python via is_arithmetic_request)."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        core = _core_expression(text)
        if not core:
            return None
        tokens = _tokenize(core.split())
        if tokens is None:
            return None
        tokens = _fold_signs(tokens)
        if tokens is None or not _well_formed(tokens):
            return None
        spoken = _say_expression(tokens)
        try:
            value = _evaluate(tokens)
        except _Undefined:
            return MathAnswer(spoken, None,
                              f"{spoken[:1].upper()}{spoken[1:]} is undefined, "
                              f"sir — division by zero has no answer.")
        said, exact = _say_number(value)
        verb = "is" if exact else "is about"
        reply = f"{spoken[:1].upper()}{spoken[1:]} {verb} {said}, sir."
        return MathAnswer(spoken, value, reply)
    except (_TooBig, ValueError, ZeroDivisionError, OverflowError,
            IndexError, TypeError):
        return None
    except Exception:
        return None


# ── routing ────────────────────────────────────────────────────────────────

_NUM_WORD_ALT = "|".join(sorted(set(_UNITS) | set(_TENS) | {"hundred"},
                                key=len, reverse=True))
_NUM_TOKEN = rf"(?:\d+(?:\.\d+)?(?:st|nd|rd|th)?|(?:{_NUM_WORD_ALT})\b)"
_OP_ALT = (r"(?:times|multiplied by|divided by|over|plus|minus|x|\*|/|\+|-|"
           r"\^|(?:raised )?to the power(?: of)?|raised to(?: the)?|to the)")
_ARITH_RE = re.compile(
    rf"(?<![\w.]){_NUM_TOKEN}\s*{_OP_ALT}\s*(?:negative\s+|-\s*)?{_NUM_TOKEN}"
    rf"|(?<![\w.]){_NUM_TOKEN}\s+(?:squared|cubed)\b")


def is_arithmetic_request(text) -> bool:
    """True when the turn holds an operator with a number on BOTH sides
    ("12 times 7", "twelve times seven", "2 to the power of 10", "9 squared")
    — never on an operator word alone ("what times does the store open",
    "three times a day"). Never raises."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        return _ARITH_RE.search(_prepare(text)) is not None
    except Exception:
        return False
