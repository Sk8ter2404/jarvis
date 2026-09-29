"""core/topic_hygiene.py — quality gates for auto-learned TOPICS and PROJECTS.

WHY THIS EXISTS (live sweep, 2026-09-29)
========================================
Asked "what am I working on lately", JARVIS said the owner had been focusing
on a "mystery" named by a made-up-sounding word, and that he had mentioned a
project he had never mentioned. Asked what to have for dinner, it volunteered
a "restaurant project" that does not exist. None of the three was real. They
were Whisper mis-transcriptions of TV / room audio that the per-turn extractor
(``bobert_companion.learn_from_turn``) labelled as a TOPIC or a PROJECT,
``merge_memory`` stored verbatim, and ``build_system_prompt`` then rendered
into EVERY system prompt ("Projects they've mentioned" / "Recent topics
you've discussed"), where the model treated them as established fact.

Nothing on that path asked whether an item was real:
  * the ambient paths (``_ambient_learn_from_gated`` and the multimodal
    extractor skill) learned topics/projects from speech that was never
    addressed to JARVIS;
  * a low-confidence transcript taught exactly as much as a clear one;
  * a label made of non-words was stored as readily as "garden shed";
  * ONE sighting was enough — a single mis-heard line became a standing
    "project" for months.

THE RULES — a candidate topic or project is:
  1. REJECTED unless it came from an OWNER-DIRECTED turn: a turn addressed to
     JARVIS and answered. Ambient / background / overheard speech never
     teaches a topic or a project (it may still teach facts, as before).
  2. REJECTED when the transcript itself is low-confidence or garbled: the
     Whisper scores when the caller has them (no_speech_prob, avg_logprob,
     compression_ratio), a text compression-ratio check (Whisper's own
     repetition signal, computed with zlib), and a mostly-non-words check.
  3. HELD — recorded as a sighting in ``memory["topic_candidates"]``, never
     rendered — until seen in at least ``MIN_OWNER_TURNS`` separate owner
     turns (distinct utterances, so a hallucination loop repeating one
     transcript counts once).
  4. Never surfaced while its label is made MOSTLY of non-dictionary words,
     unless the owner has used those words himself in two or more separate
     turns — which is what lets real proper nouns and jargon he actually says
     through.

``tools/audit_learned_topics.py`` applies rules 2-4 to what is ALREADY stored
(``find_suspects``) — one home for the heuristics, so the audit and the write
gate can never disagree about what "suspect" means (this repo's #1 bug class
is the stale duplicate).

Pure stdlib. The only I/O is the read-only owner-vocabulary loader.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
import zlib
from typing import Optional

from core.common_words import lexicon as _lexicon

# ── tuning ──────────────────────────────────────────────────────────────────
# A topic/project must be seen in this many SEPARATE owner turns before it is
# surfaced to the model.
MIN_OWNER_TURNS = 2

# Learn-time transcript thresholds. Deliberately STRICTER than the response
# gate in core/speech_filter.py (no_speech 0.85 / logprob -1.5, and a loud-RMS
# bypass): a turn that is good enough to ANSWER may still be too shaky to
# LEARN a standing topic from, and a loud TV is loud. These are Whisper's own
# decode-failure defaults (no_speech_threshold 0.6, logprob_threshold -1.0,
# compression_ratio_threshold 2.4).
LEARN_MAX_NO_SPEECH_PROB = 0.6
LEARN_MIN_AVG_LOGPROB = -1.0
LEARN_MAX_COMPRESSION_RATIO = 2.4
# zlib's fixed overhead makes short strings compress to a ratio < 1, so the
# text-side repetition check only means something on a longer transcript.
_COMPRESSION_MIN_CHARS = 40

# Bounded hidden sighting store (a cache, like MAX_TOPICS: oldest evicted).
MAX_CANDIDATES = 60
MAX_TURNS_PER_CANDIDATE = 8

# Keys in the bobert_memory.json dict.
CANDIDATES_KEY = "topic_candidates"
QUARANTINE_KEY = "quarantined"

# Function words, fillers and assistant-domain words. Never "content": they
# say nothing about WHICH subject a label names.
_STOPWORDS = frozenset("""
a an the and or but nor so yet if then than that this these those there here
of to in on at by for with from into onto about above below over under after
before between through during without within upon via per as
i me my mine myself you your yours yourself we us our ours he him his she her
hers it its they them their theirs who whom whose which what when where why
how
is am are was were be been being do does did doing done have has had having
will would shall should can could may might must
not no yes very just also too only really quite some any all each every both
few more most much many other another such own same
up down out off again once still even ever never always now today tonight
don't doesn't didn't can't couldn't won't wouldn't shouldn't isn't aren't
wasn't weren't i'm i've i'll i'd you're you've you'll we're they're it's
that's what's there's let's
jarvis sir user owner assistant please thanks thank okay ok hey hi hello
um uh hmm oh yeah yep
""".split())

# Words that describe the KIND of item rather than its subject. "restaurant
# project" is about a restaurant; "project" alone identifies nothing, and
# letting it count would make every "... project" label corroborate every
# other one.
_GENERIC = frozenset("""
project projects topic topics discussion discussions conversation
conversations chat chats talk talking question questions request requests
inquiry inquiries query queries thing things stuff idea ideas plan plans
general misc miscellaneous update updates status info information help issue
issues work working mention mentioned mentions involving related regarding
new recent recently lately current currently ongoing some various possible
asked asking ask wants wanted want tell told say said check checking
""".split())

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


# ── tokenising ──────────────────────────────────────────────────────────────
def tokens(text) -> list:
    """Lowercase alphabetic word tokens. Possessive 's is dropped ("user's" ->
    "user"); other contractions stay whole so the stopword list can own them.
    Digits and punctuation are ignored. Never raises."""
    if not isinstance(text, str) or not text:
        return []
    low = text.lower().replace("’", "'")
    out = []
    for w in _WORD_RE.findall(low):
        if w.endswith("'s"):
            w = w[:-2]
        if w:
            out.append(w)
    return out


def content_words(text) -> list:
    """The words that name a SUBJECT: tokens minus stopwords, generic
    item-kind words, contractions and one- or two-letter tokens (too short to
    judge, and "tv" / "de" say nothing about which subject is meant)."""
    return [w for w in tokens(text)
            if len(w) > 2 and "'" not in w
            and w not in _STOPWORDS and w not in _GENERIC]


def stem(word: str) -> str:
    """Light, deterministic suffix folding used ONLY for matching two labels
    against each other ("plans for the weekend" ~ "weekend plans"). Both sides
    go through the same function, so consistency matters, not linguistics."""
    w = word
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[:-len(suf)]
            if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def topic_key(text) -> frozenset:
    """The set of stemmed content words — what two labels are compared on."""
    return frozenset(stem(w) for w in content_words(text))


def keys_match(a, b) -> bool:
    """Do two topic keys name the same subject? Equal, one contained in the
    other ("shed" ~ "garden shed"), or Jaccard overlap >= 0.5."""
    a, b = frozenset(a or ()), frozenset(b or ())
    if not a or not b:
        return False
    if a == b or a <= b or b <= a:
        return True
    inter = len(a & b)
    return bool(inter) and inter / len(a | b) >= 0.5


def _norm_label(text) -> str:
    return " ".join(tokens(text))


# ── the dictionary check ────────────────────────────────────────────────────
_SUFFIXES = (
    "ations", "ation", "itions", "ition", "ings", "ing", "ers", "er", "est",
    "ed", "es", "s", "ly", "ness", "ments", "ment", "fully", "ful", "less",
    "ables", "able", "ible", "ities", "ity", "ions", "ion", "ally", "al",
    "ives", "ive", "izes", "ized", "izing", "ize", "ises", "ised", "ising",
    "ise", "ists", "ist", "isms", "ism", "ships", "ship", "hood", "ery", "ish",
    "ous", "atic", "tic", "ic", "ics", "en", "y",
)
_PREFIXES = (
    "un", "re", "pre", "dis", "mis", "non", "over", "under", "out", "sub",
    "super", "inter", "anti", "auto", "co", "de", "multi", "micro", "mini",
    "self", "semi", "counter", "cross", "back", "up", "down", "mega", "ultra",
)


_INFLECTIONS = frozenset({"s", "es", "ed", "ing", "er", "ers", "ings"})
_PARTICLES = frozenset({"up", "out", "off", "on", "in", "down", "over",
                        "back", "away"})


def _one_level(w: str) -> set:
    """``w`` plus every base form one affix-strip away."""
    forms = {w}
    if len(w) > 4 and w.endswith(("ies", "ied", "ier")):
        forms.add(w[:-3] + "y")
    if len(w) > 5 and w.endswith("iest"):
        forms.add(w[:-4] + "y")
    for suf in _SUFFIXES:
        # Inflections may leave a two-letter base ("going" -> "go", "does" ->
        # "do"); derivational suffixes need three, or noise creeps in.
        floor = 2 if suf in _INFLECTIONS else 3
        if w.endswith(suf) and len(w) - len(suf) >= floor:
            base = w[:-len(suf)]
            forms.add(base)
            forms.add(base + "e")
            if len(base) >= 2 and base[-1] == base[-2]:
                forms.add(base[:-1])
            if base.endswith("i"):
                forms.add(base[:-1] + "y")
    return forms


def _base_forms(w: str) -> set:
    """Base forms up to two affix-strips deep ("carefully" -> "careful" ->
    "care", "supplementation" -> "supplement")."""
    forms = set()
    for f in _one_level(w):
        forms |= _one_level(f)
    return forms


def _lexical(w: str, vocab) -> bool:
    lex = _lexicon()
    return any(f in lex or f in vocab for f in _base_forms(w))


def is_known_word(word, vocab=frozenset()) -> bool:
    """Is ``word`` a real word — in the bundled list (with inflections and
    affixes), a simple compound of two listed words, or a word the owner has
    used in separate turns (``vocab``)? One- and two-letter tokens and
    contractions are never judged (always True)."""
    if not isinstance(word, str):
        return True
    w = word.lower().strip()
    if len(w) <= 2 or "'" in w or not w.isalpha():
        return True
    vocab = vocab or frozenset()
    if _lexical(w, vocab):
        return True
    for pre in _PREFIXES:
        if w.startswith(pre) and len(w) - len(pre) >= 3 \
                and _lexical(w[len(pre):], vocab):
            return True
    # Negating in-/im- ("inorganic", "impractical"): longer remainder only,
    # or every word starting "in" would find some tail to hide behind.
    if w.startswith(("in", "im")) and len(w) >= 7 and _lexical(w[2:], vocab):
        return True
    # Simple two-part compounds ("toolbox", "workbench", "spreadsheet"), and
    # a word plus a particle ("cleanup", "payoff", "workout").
    for i in range(3, len(w) - 1):
        head, tail = w[:i], w[i:]
        if not (head in _lexicon() or head in vocab):
            continue
        if tail in _PARTICLES or (len(tail) >= 3 and _lexical(tail, vocab)):
            return True
    return False


def unknown_words(text, vocab=frozenset()) -> list:
    """Content words of ``text`` that are not known words (order kept,
    duplicates dropped)."""
    return list(dict.fromkeys(
        w for w in content_words(text) if not is_known_word(w, vocab)))


# An ALL-CAPS token of 2-6 letters in the original text ("PID", "AJR", "PCBs")
# is an acronym or initialism, never a mis-heard word.
_ACRONYM_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,6})s?(?![A-Za-z])")
# Shortest unknown token that can look like a Whisper nonce. An unknown 2-3
# letter token ("ajr", "pid" once the label is lower-cased) is far more often
# an abbreviation than a mis-hearing, so it may count toward "mostly
# unknown" but can never be the reason a label is flagged.
NONCE_MIN_LEN = 4


def acronyms(text) -> frozenset:
    """Lower-cased acronyms written in capitals in ``text``."""
    if not isinstance(text, str):
        return frozenset()
    return frozenset(m.lower() for m in _ACRONYM_RE.findall(text))


# How many separate stored facts must use a word before the facts vouch for
# it. One is not enough: the extractor that learned a mis-heard topic often
# learned a sibling FACT from the same overheard line (live store,
# 2026-09-29: the one fact containing the mis-heard name was itself junk),
# and a single junk fact must not certify the junk topic.
MIN_FACTS_FOR_WORD = 2


def facts_vocab(memory, min_facts: int = MIN_FACTS_FOR_WORD) -> frozenset:
    """Words the store's own ``facts`` vouch for: used in at least
    ``min_facts`` separate facts, or written as an ACRONYM in capitals in any
    fact ("works with the QZX team"). That is how a client name the owner
    has stated becomes a known word everywhere else."""
    facts = memory.get("facts") if isinstance(memory, dict) else None
    counts: dict = {}
    caps: set = set()
    for f in facts if isinstance(facts, list) else ():
        if not isinstance(f, str):
            continue
        caps |= acronyms(f)
        for w in set(tokens(f)):
            counts[w] = counts.get(w, 0) + 1
    return frozenset(caps | {w for w, n in counts.items() if n >= min_facts})


def garbled_reason(text, vocab=frozenset(), *, utterance: bool = False) -> str:
    """'' when ``text`` reads as real words, else a short reason.

    A word is KNOWN when it is in the bundled lexicon (with inflections and
    affixes), is in ``vocab`` (the owner's own words: used in 2+ logged turns,
    or present in his stored facts), or is written as an ACRONYM in capitals.

    A topic/project LABEL (the thing that would be surfaced) is flagged only
    when BOTH hold:
      * at least half of its content words are unknown, and
      * at least one of those unknown words is NONCE-SHAPED — NONCE_MIN_LEN+
        letters and not an acronym. That is what separates "<non-word>
        mystery" from "PID controller explanation" or "<client> work".
    (Half, not a strict majority, on purpose: the typical mis-hearing is a
    two-word label, one real word plus one non-word, and a strict majority
    would never flag it.)

    A whole owner UTTERANCE (``utterance=True``) needs more: two or more
    unknown words, more than half, one of them nonce-shaped. A short real
    request naming one proper noun ("how's <name> going") is normal speech;
    a run of non-words is a mis-transcription."""
    content = content_words(text)
    if not content:
        return ""
    caps = acronyms(text)
    unknown = [w for w in content
               if w not in caps and not is_known_word(w, vocab)]
    if not any(len(w) >= NONCE_MIN_LEN for w in unknown):
        return ""
    if utterance:
        bad = len(unknown) >= 2 and 2 * len(unknown) > len(content)
    else:
        bad = 2 * len(unknown) >= len(content)
    if bad:
        return ("mostly unrecognised words ("
                + ", ".join(dict.fromkeys(unknown)) + ")")
    return ""


# ── transcript quality ──────────────────────────────────────────────────────
def compression_ratio(text) -> float:
    """Whisper's repetition signal: raw bytes / zlib-compressed bytes. A
    hallucination loop ("thank you thank you thank you ...") compresses far
    better than speech. 0.0 for empty input."""
    raw = (text or "").encode("utf-8") if isinstance(text, str) else b""
    if not raw:
        return 0.0
    return len(raw) / len(zlib.compress(raw))


def _num(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def transcript_quality_reason(text, conf=None, vocab=frozenset()) -> str:
    """'' when the owner's transcript is clean enough to learn a topic from,
    else the reason it is not. ``conf`` is the per-turn Whisper metadata dict
    when the caller has it (typed/injected turns carry synthetic high-confidence
    values); every score is optional."""
    if isinstance(conf, dict):
        nsp = _num(conf.get("no_speech_prob"))
        if nsp is not None and nsp > LEARN_MAX_NO_SPEECH_PROB:
            return f"no_speech_prob={nsp:.2f}"
        lp = _num(conf.get("avg_logprob"))
        if lp is not None and lp < LEARN_MIN_AVG_LOGPROB:
            return f"low transcription confidence (avg_logprob={lp:.2f})"
        cr = _num(conf.get("compression_ratio"))
        if cr is not None and cr > LEARN_MAX_COMPRESSION_RATIO:
            return f"repetitive transcript (compression_ratio={cr:.2f})"
    t = text.strip() if isinstance(text, str) else ""
    if len(t) >= _COMPRESSION_MIN_CHARS:
        cr = compression_ratio(t)
        if cr > LEARN_MAX_COMPRESSION_RATIO:
            return f"repetitive transcript (compression_ratio={cr:.2f})"
    g = garbled_reason(t, vocab, utterance=True)
    if g:
        return "garbled transcript: " + g
    return ""


def screen_reason(*, owner_directed, turn_text="", conf=None,
                  vocab=frozenset()) -> str:
    """Rules 1 + 2: '' when a turn may contribute topic/project sightings at
    all, else why not."""
    if not owner_directed:
        return "not owner-directed speech (ambient / background)"
    return transcript_quality_reason(turn_text, conf, vocab)


# ── owner vocabulary ────────────────────────────────────────────────────────
def owner_vocab_from_texts(texts, min_turns: int = MIN_OWNER_TURNS) -> frozenset:
    """Words the owner used in at least ``min_turns`` DISTINCT utterances.
    Identical utterances count once, so a repeated hallucination cannot vouch
    for its own non-words."""
    counts: dict = {}
    seen: set = set()
    for t in texts or ():
        norm = _norm_label(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        for w in set(norm.split()):
            counts[w] = counts.get(w, 0) + 1
    return frozenset(w for w, n in counts.items() if n >= min_turns)


def read_owner_texts(path, limit: int = 5000) -> Optional[list]:
    """Owner utterances from the voice-command log (memory/voice_commands.jsonl,
    written by memory.record_voice_command for every accepted owner turn). None
    when the log is absent or unreadable — callers must tell "no evidence" from
    "no mentions". Read-only; never raises."""
    if not path or not os.path.isfile(path):
        return None
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                txt = rec.get("text") if isinstance(rec, dict) else None
                if isinstance(txt, str) and txt.strip():
                    out.append(txt.strip())
    except OSError:
        return None
    return out[-limit:] if limit else out


_VOCAB_CACHE: dict = {}
_VOCAB_LOCK = threading.Lock()


def load_owner_vocab(path) -> frozenset:
    """``owner_vocab_from_texts`` over the voice-command log at ``path``,
    cached on (path, mtime, size) so the per-turn learner does not re-parse an
    unchanged log. Empty on any problem."""
    try:
        st = os.stat(path)
        sig = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except (OSError, TypeError, ValueError):
        return frozenset()
    with _VOCAB_LOCK:
        hit = _VOCAB_CACHE.get("sig")
        if hit == sig:
            return _VOCAB_CACHE["vocab"]
    texts = read_owner_texts(path) or []
    vocab = owner_vocab_from_texts(texts)
    with _VOCAB_LOCK:
        _VOCAB_CACHE["sig"] = sig
        _VOCAB_CACHE["vocab"] = vocab
    return vocab


# ── sightings (rules 3 + 4) ─────────────────────────────────────────────────
def turn_id(text) -> str:
    """Stable id of one owner utterance: two identical transcripts are the
    same turn for counting purposes. A caller with no text gets a unique id
    (every such call is its own turn)."""
    norm = _norm_label(text)
    if not norm:
        return "anon-" + uuid.uuid4().hex[:12]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _entry_ts(entry) -> float:
    try:
        return float(entry.get("ts", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _candidates(memory) -> list:
    cands = memory.get(CANDIDATES_KEY)
    if not isinstance(cands, list):
        cands = []
        memory[CANDIDATES_KEY] = cands
    return cands


def _same_subject(label_a, key_a, label_b, key_b) -> bool:
    if key_a and key_b:
        return keys_match(key_a, key_b)
    # All-generic labels ("general chat") have no key: compare whole labels.
    return bool(label_a) and _norm_label(label_a) == _norm_label(label_b)


def observe(memory, *, kind: str, label: str, turn: str, vocab=frozenset(),
            now: Optional[float] = None) -> tuple:
    """Record ONE owner-turn sighting of ``label`` in ``memory`` (mutated in
    place — the caller holds the memory lock and saves) and decide whether it
    may be surfaced now. Returns ``(surface, reason)``.

    Surfaced only once the matching candidate has been seen in
    MIN_OWNER_TURNS distinct turns AND its label is not mostly non-words."""
    label = label.strip() if isinstance(label, str) else ""
    if not label:
        return False, "empty label"
    now = time.time() if now is None else float(now)
    key = topic_key(label)
    cands = _candidates(memory)
    match = None
    for c in cands:
        if not isinstance(c, dict) or c.get("kind") != kind:
            continue
        if _same_subject(label, key, c.get("label", ""),
                         frozenset(c.get("key") or ())):
            match = c
            break
    if match is None:
        match = {"kind": kind, "label": label, "key": sorted(key),
                 "turns": [], "first_seen": now}
        cands.append(match)
    turns = [t for t in (match.get("turns") or []) if isinstance(t, dict)]
    if turn not in {t.get("id") for t in turns}:
        turns.append({"id": turn, "ts": now})
    match["turns"] = turns[-MAX_TURNS_PER_CANDIDATE:]
    match["label"] = label
    match["last_seen"] = now
    if len(cands) > MAX_CANDIDATES:
        cands.sort(key=lambda c: (c.get("last_seen", 0.0)
                                  if isinstance(c, dict) else 0.0))
        del cands[:len(cands) - MAX_CANDIDATES]
    n = len(match["turns"])
    if n < MIN_OWNER_TURNS:
        return False, f"held: seen in {n} of {MIN_OWNER_TURNS} owner turns"
    g = garbled_reason(label, vocab)
    if g:
        return False, "held: " + g
    return True, f"seen in {n} owner turns"


def matches_any(label, items) -> bool:
    """Does ``label`` name the same subject as any string in ``items``? Used
    to stop a paraphrase of an existing project being added beside it."""
    key = topic_key(label)
    for it in items or ():
        if isinstance(it, str) and _same_subject(label, key, it, topic_key(it)):
            return True
    return False


def forget_sightings_since(memory, cutoff: float) -> int:
    """Drop every candidate sighting at or after ``cutoff`` (forget_last_hour
    must reach this store too, or a forgotten hour could still promote a topic
    later). Candidates left with no sightings go. Returns sightings removed."""
    cands = memory.get(CANDIDATES_KEY)
    if not isinstance(cands, list):
        return 0
    removed = 0
    kept = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        turns = [t for t in (c.get("turns") or []) if isinstance(t, dict)]
        keep = [t for t in turns if _entry_ts(t) < cutoff]
        removed += len(turns) - len(keep)
        if keep:
            c["turns"] = keep
            kept.append(c)
    if removed:
        memory[CANDIDATES_KEY] = kept
    return removed


# ── audit of what is ALREADY stored (tools/audit_learned_topics.py) ─────────
def _mentions(key, owner_keys) -> int:
    """How many owner utterances mention the subject ``key`` (at least half of
    its content stems, and at least one)."""
    if not key:
        return 0
    need = max(1, (len(key) + 1) // 2)
    return sum(1 for ok in owner_keys if len(key & ok) >= need)


def find_suspects(memory, owner_texts=None) -> list:
    """Suspect entries among the SURFACED lists (``topics`` and ``projects``),
    judged by the same rules as the write gate.

    ``owner_texts`` — the owner's logged utterances (``read_owner_texts``), or
    None when that log is unavailable. Words in the store's ``facts`` count as
    known words. Without it only the non-word rule runs:
    the corroboration rule needs evidence of what the owner actually said, and
    "no log" must not read as "never said it".

    Returns ``[{"kind", "index", "text", "reason"}, ...]`` in list order.
    """
    topics = memory.get("topics") if isinstance(memory, dict) else None
    projects = memory.get("projects") if isinstance(memory, dict) else None
    topics = topics if isinstance(topics, list) else []
    projects = projects if isinstance(projects, list) else []

    have_log = owner_texts is not None
    distinct = list(dict.fromkeys(t for t in (owner_texts or [])
                                  if isinstance(t, str) and t.strip()))
    vocab = owner_vocab_from_texts(distinct) | facts_vocab(memory)
    owner_keys = [topic_key(t) for t in distinct]
    facts = memory.get("facts") if isinstance(memory, dict) else None
    fact_keys = [topic_key(f) for f in (facts if isinstance(facts, list) else [])
                 if isinstance(f, str)]
    topic_labels = [(t.get("topic") if isinstance(t, dict) else None)
                    for t in topics]
    topic_keys = [topic_key(lbl) if isinstance(lbl, str) else frozenset()
                  for lbl in topic_labels]

    def _judge(text, key, topic_hits) -> str:
        g = garbled_reason(text, vocab)
        if g:
            return g
        if not have_log or not key:
            return ""
        said = _mentions(key, owner_keys)
        if topic_hits >= MIN_OWNER_TURNS or said >= MIN_OWNER_TURNS:
            return ""
        # Word-based evidence, so say exactly that: an abstract label
        # ("<thing> overview") can be real even when no turn uses its words.
        if said == 0:
            heard = "no logged owner turn uses its words"
        else:
            heard = "only 1 logged owner turn uses its words"
        why = (f"not corroborated: {heard}; in {topic_hits} stored topic(s) "
               f"(needs {MIN_OWNER_TURNS} of either)")
        in_facts = _mentions(key, fact_keys)
        if in_facts:
            # Reported, never counted: a fact is often learned from the very
            # same line as the topic, so it is not a separate sighting.
            why += (f"; its words appear in {in_facts} stored fact(s), which "
                    "do not count as a separate turn")
        return why

    out = []
    for i, lbl in enumerate(topic_labels):
        if not isinstance(lbl, str) or not lbl.strip():
            continue
        key = topic_keys[i]
        hits = sum(1 for k in topic_keys if keys_match(key, k))
        why = _judge(lbl, key, hits)
        if why:
            out.append({"kind": "topic", "index": i, "text": lbl,
                        "reason": why})
    for i, p in enumerate(projects):
        if not isinstance(p, str) or not p.strip():
            continue
        key = topic_key(p)
        hits = sum(1 for k in topic_keys if keys_match(key, k))
        why = _judge(p, key, hits)
        if why:
            out.append({"kind": "project", "index": i, "text": p,
                        "reason": why})
    return out
