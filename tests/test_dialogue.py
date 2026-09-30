"""Tests for core/dialogue.py — a scripted back-and-forth between JARVIS and a
talking device, played strictly in turn.

GENERIC fixtures only ("self" is JARVIS, "device" is a made-up desk device).
Every callback is a fake; no audio, no network, no monolith.

stdlib unittest only; CI-safe (light tier).
    python -m unittest tests.test_dialogue
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from concurrent.futures import Future

from core import dialogue as dlg

_WHO = {"device": "device", "jarvis": "self"}


def _clean_self(t):
    t = dlg.strip_lead_words(t, ("indeed", "ah", "well", "oh", "quite"))
    if len(t.split()) > 16 or len(t) > 95:
        return None
    return t


def _clean_device(t):
    return t


def _check_self(t):
    return dlg.conflicts(t, words=tuple(dlg._dsf.STOP_WORDS) + ("jarvis",),
                         phrases=dlg.STOP_PHRASES, max_gap=0)


def _check_device(t):
    return dlg.conflicts(t, words=tuple(dlg._dsf.STOP_WORDS)
                         + ("jarvis", "sir"), phrases=dlg.STOP_PHRASES,
                         max_gap=0)


def _script(*pairs):
    return json.dumps({"lines": [{"who": w, "text": t} for w, t in pairs]})


def _v(raw, **kw):
    args = dict(who_map=_WHO, min_lines=3, max_lines=6,
                clean_self=_clean_self, clean_device=_clean_device,
                check_self=_check_self, check_device=_check_device,
                closer=lambda: "And there we shall leave it.")
    args.update(kw)
    return dlg.validate_script(raw, **args)


GOOD = _script(("device", "The toast is late again."),
               ("jarvis", "It keeps its own hours."),
               ("device", "I respect that about toast."),
               ("jarvis", "A kindred spirit. Barely."))


class ConstantsTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(dlg.STOP_PHRASES, (
            "that's enough", "enough", "quiet", "be quiet", "shut up",
            "knock it off", "never mind", "okay okay", "that'll do",
            "that will do"))
        self.assertEqual(dlg.REASONS, (
            "done", "owner_stop", "wake", "interrupted", "device_lost",
            "device_busy", "device_muted", "device_heap", "device_no_answer",
            "tts_muted", "mic_muted", "sleep", "expired", "error"))

    def test_reasons_carry_no_failure_marker(self):
        from core.failure_markers import FAILURE_MARKERS
        for r in dlg.REASONS:
            for m in FAILURE_MARKERS:
                self.assertNotIn(m.lower(), r.replace("_", " "), (r, m))


class ValidateScriptTests(unittest.TestCase):
    def test_good_script(self):
        lines = _v(GOOD)
        self.assertEqual([l.who for l in lines],
                         ["device", "self", "device", "self"])
        self.assertTrue(lines[-1].final)
        self.assertFalse(any(l.final for l in lines[:-1]))
        self.assertEqual(lines[0].chunks, ("The toast is late again.",))

    def test_fenced_json_with_prose(self):
        raw = "Here you go:\n```json\n" + GOOD + "\n```\nEnjoy."
        self.assertEqual(len(_v(raw)), 4)

    def test_bare_list(self):
        raw = json.dumps(json.loads(GOOD)["lines"])
        self.assertEqual(len(_v(raw)), 4)

    def test_parsed_dict_accepted(self):
        self.assertEqual(len(_v(json.loads(GOOD))), 4)

    def test_garbage_is_none(self):
        self.assertIsNone(_v("no json here"))
        self.assertIsNone(_v("{not json"))
        self.assertIsNone(_v(json.dumps({"lines": "nope"})))
        self.assertIsNone(_v(None))

    def test_unknown_who_truncates(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "I respect that about toast."),
                      ("narrator", "Meanwhile."),
                      ("jarvis", "Never spoken."))
        lines = _v(raw)
        self.assertEqual(len(lines), 4)          # 3 + the closer
        self.assertEqual(lines[-1].text, "And there we shall leave it.")
        self.assertNotIn("Never spoken.", [l.text for l in lines])

    def test_tagged_self_line_survives_with_tags_stripped(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "[wry] [mood:dry_amused] It keeps its own "
                                 "hours."),
                      ("device", "I respect that about toast."),
                      ("jarvis", "[intent:calm] A kindred spirit. Barely."))
        lines = _v(raw)
        self.assertEqual(lines[1].text, "It keeps its own hours.")
        self.assertEqual(lines[3].text, "A kindred spirit. Barely.")

    def test_action_token_rejects_the_line_and_the_rest(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "I respect that about toast."),
                      ("jarvis", "Fine. [ACTION: shutdown_pc]"),
                      ("device", "Oh no."),
                      ("jarvis", "Quite."))
        lines = _v(raw)
        self.assertFalse(any("[ACTION" in l.text for l in lines))
        self.assertEqual(lines[-1].text, "And there we shall leave it.")

    def test_leading_indeed_stripped(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "Indeed, it keeps its own hours."),
                      ("device", "I respect that about toast."),
                      ("jarvis", "Indeed."))
        lines = _v(raw)
        self.assertEqual(lines[1].text, "It keeps its own hours.")
        # Too short to strip: kept as is.
        self.assertEqual(lines[3].text, "Indeed.")

    def test_non_ascii_device_text_rejected(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "Café toast is best."),
                      ("jarvis", "Never spoken."))
        self.assertIsNone(_v(raw))     # only 2 lines survive < min 3

    def test_long_device_line_salvaged_into_two_chunks(self):
        long = ("I have studied toast for many long years in the kitchen. "
                "It is always late and it is always warm.")
        self.assertGreater(len(long), 90)
        self.assertLessEqual(len(long), 120)
        raw = _script(("device", long), ("jarvis", "Remarkable dedication."),
                      ("device", "Thank you."), ("jarvis", "Mostly."))
        lines = _v(raw)
        self.assertEqual(len(lines[0].chunks), 2)
        self.assertTrue(all(len(c) <= 90 for c in lines[0].chunks))
        self.assertEqual(" ".join(lines[0].chunks), long)

    def test_too_long_device_line_rejected(self):
        raw = _script(("device", "word " * 40), ("jarvis", "Too much."),
                      ("device", "Sorry."), ("jarvis", "Quite."))
        self.assertIsNone(_v(raw))

    def test_same_speaker_neighbours_merge(self):
        raw = _script(("device", "The toast is late."),
                      ("device", "Again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "I respect that."),
                      ("jarvis", "Naturally."))
        lines = _v(raw)
        self.assertEqual(lines[0].text, "The toast is late. Again.")
        self.assertEqual(len(lines), 4)

    def test_merge_that_breaks_caps_drops_the_second(self):
        raw = _script(("device", "The toast is late."),
                      ("jarvis", "It keeps its own hours, as all fine "
                                 "breads eventually do."),
                      ("jarvis", "Every single one of them, without "
                                 "exception, sir."),
                      ("device", "I respect that."),
                      ("jarvis", "Naturally."))
        lines = _v(raw)
        self.assertEqual([l.who for l in lines],
                         ["device", "self", "device", "self"])
        self.assertNotIn("exception", lines[1].text)

    def test_conflict_truncates_the_rest(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "I respect that about toast."),
                      ("jarvis", "Enough of this, I think."),
                      ("device", "Never said."),
                      ("jarvis", "Never said either."))
        lines = _v(raw)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[-1].text, "And there we shall leave it.")

    def test_stop_word_in_device_line_truncates(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "Is this a toast emergency?"),
                      ("jarvis", "Never said."))
        self.assertIsNone(_v(raw))

    def test_closer_appended_when_device_is_last(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "I respect that about toast."))
        lines = _v(raw)
        self.assertEqual(lines[-1].who, "self")
        self.assertTrue(lines[-1].final)
        self.assertEqual(lines[-1].text, "And there we shall leave it.")

    def test_no_closer_drops_trailing_device_lines(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."),
                      ("device", "I respect that about toast."),
                      ("jarvis", "Naturally."),
                      ("device", "Ha."))
        lines = _v(raw, closer=None, min_lines=2)
        self.assertEqual(lines[-1].text, "Naturally.")
        self.assertTrue(lines[-1].final)

    def test_below_min_lines_is_none(self):
        raw = _script(("device", "The toast is late again."),
                      ("jarvis", "It keeps its own hours."))
        self.assertIsNone(_v(raw))

    def test_max_lines_cut(self):
        pairs = []
        for i in range(5):
            pairs += [("device", f"Toast fact number {'x' * (i + 1)}."),
                      ("jarvis", f"Fascinating, truly {'y' * (i + 1)}.")]
        lines = _v(_script(*pairs), max_lines=4)
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[-1].final)

    def test_check_script_hook(self):
        lines = _v(GOOD, check_script=lambda ls: ls[:3])
        self.assertEqual(len(lines), 4)     # 3 + closer
        self.assertEqual(lines[-1].text, "And there we shall leave it.")

    def test_never_raises(self):
        def boom(_t):
            raise RuntimeError("x")
        self.assertIsNone(_v(GOOD, clean_self=boom))


class VocabularyTests(unittest.TestCase):
    def test_is_stop_utterance(self):
        self.assertEqual(dlg.is_stop_utterance("stop"), "stop")
        self.assertEqual(dlg.is_stop_utterance("okay, that's enough"), "stop")
        self.assertEqual(dlg.is_stop_utterance("please be quiet now"), "stop")
        self.assertEqual(dlg.is_stop_utterance("knock it off you lot"),
                         "stop")
        self.assertEqual(dlg.is_stop_utterance("That will do."), "stop")
        self.assertEqual(dlg.is_stop_utterance("hey jarvis",
                                                ("jarvis", "hey jarvis")),
                         "wake")
        self.assertEqual(dlg.is_stop_utterance("jarvis stop", ("jarvis",)),
                         "stop")
        self.assertEqual(dlg.is_stop_utterance("the toast is late"), "")
        self.assertEqual(dlg.is_stop_utterance(""), "")
        self.assertEqual(dlg.is_stop_utterance(None), "")
        # A word merely containing a stop word is not one.
        self.assertEqual(dlg.is_stop_utterance("the bus stopped"), "")

    def test_conflicts_words_prefix_and_phrases_with_gap(self):
        self.assertEqual(dlg.conflicts("let us dance", words=("dance*",)),
                         "word:dance*")
        self.assertEqual(dlg.conflicts("they were dancing",
                                       words=("dance*",)), None)
        self.assertEqual(dlg.conflicts("they were dancing",
                                       words=("danc*",)), "word:danc*")
        self.assertEqual(dlg.conflicts("turn hard left",
                                       phrases=("turn left",), max_gap=1),
                         "phrase:turn left")
        self.assertIsNone(dlg.conflicts("turn very hard left",
                                        phrases=("turn left",), max_gap=1))
        self.assertIsNone(dlg.conflicts("turn hard left",
                                        phrases=("turn left",), max_gap=0))
        self.assertIsNone(dlg.conflicts("quite right sir",
                                        words=("left",),
                                        phrases=("turn right",)))

    def test_conflicts_near(self):
        near = ((("turn", "spin", "go"), ("left", "right", "around"), 2),)
        self.assertEqual(dlg.conflicts("go all around", near=near),
                         "near:go~around")
        self.assertEqual(dlg.conflicts("left, then spin", near=near),
                         "near:spin~left")
        self.assertIsNone(dlg.conflicts("go and get the salt dish right",
                                        near=near))
        self.assertIsNone(dlg.conflicts("", words=("x",)))

    def test_strip_speech_tags(self):
        self.assertEqual(dlg.strip_speech_tags("[wry] [mood:dry_amused]Hi."),
                         "Hi.")
        self.assertEqual(dlg.strip_speech_tags("[intent:calm][wry] Hi."),
                         "Hi.")
        self.assertEqual(dlg.strip_speech_tags("Hi [wry]."), "Hi [wry].")
        self.assertEqual(dlg.strip_speech_tags(None), "")

    def test_strip_lead_words(self):
        self.assertEqual(dlg.strip_lead_words("Well, the toast is late.",
                                              ("well",)),
                         "The toast is late.")
        self.assertEqual(dlg.strip_lead_words("Well, fine.", ("well",)),
                         "Well, fine.")
        self.assertEqual(dlg.strip_lead_words("Wellington is far.",
                                              ("well",)),
                         "Wellington is far.")

    def test_chunk_device_text(self):
        self.assertEqual(dlg.chunk_device_text("Short.", 90), ["Short."])
        two = dlg.chunk_device_text("One two three. Four five six.", 16)
        self.assertEqual(two, ["One two three.", "Four five six."])
        words = dlg.chunk_device_text("aaa bbb ccc ddd eee fff", 12)
        self.assertEqual(words, ["aaa bbb ccc", "ddd eee fff"])
        self.assertEqual(dlg.chunk_device_text("a b c d e f g h i", 3, 2), [])
        self.assertEqual(dlg.chunk_device_text("x" * 20, 10), [])
        self.assertEqual(dlg.chunk_device_text("", 10), [])


# ── Runner ──────────────────────────────────────────────────────────────
class FakeSession:
    def __init__(self):
        self.reason = None
        self.cut = threading.Event()
        self._lock = threading.Lock()

    def stopped(self):
        return self.reason

    def stop(self, reason):
        with self._lock:
            if self.reason:
                return False
            self.reason = reason
        self.cut.set()
        return True


class FakeCap:
    def __init__(self, available=True, beat_voiced=False, delay=0.0,
                 kind="", log=None):
        self.available = available
        self.beat_voiced = beat_voiced
        self._delay = delay
        self._kind = kind
        self._t0 = time.monotonic()
        self.waited = None

    def result(self, timeout=0.0):
        self.waited = timeout
        remaining = self._delay - (time.monotonic() - self._t0)
        if remaining > timeout:
            time.sleep(timeout)
            return ("pending", "")
        time.sleep(max(0.0, remaining))
        return (self._kind, "")


class Rig:
    """Fake speaker / device / capture with an ordered event log."""

    def __init__(self, *, device_s=0.05, say_results=None, speak_block=None,
                 cap=None, speak_results=None):
        self.log = []
        self.session = FakeSession()
        self.device_s = device_s
        self.device_until = 0.0
        self.say_results = list(say_results or [])
        self.speak_results = list(speak_results or [])
        self.speak_block = speak_block     # text -> block until cut
        self.cap = cap
        self.self_speaking = False
        self.device_talking_during_self = False
        self.listen_during_self = False

    def speak_self(self, text, final):
        if not self.device_done():
            self.device_talking_during_self = True
        self.self_speaking = True
        self.log.append(("self", text, final))
        try:
            if self.speak_results:
                return self.speak_results.pop(0)
            if self.speak_block and text == self.speak_block:
                if self.session.cut.wait(5.0):
                    return "interrupted"
            return "spoken"
        finally:
            self.self_speaking = False

    def device_say(self, chunk):
        res = self.say_results.pop(0) if self.say_results else "ok"
        self.log.append(("say", chunk, res))
        if res == "ok":
            self.device_until = time.monotonic() + self.device_s
        return res

    def device_done(self):
        return time.monotonic() >= self.device_until

    def listen(self, until, *, beat_s, max_s):
        # Keyword-only, like skill_utils["listen_for_stop"].
        if self.self_speaking:
            self.listen_during_self = True
        self.log.append(("listen_start",))
        t0 = time.monotonic()
        while not until() and time.monotonic() - t0 < max_s:
            time.sleep(0.005)
        time.sleep(beat_s)
        self.log.append(("listen_end", self.device_done()))
        return self.cap if self.cap is not None else FakeCap()

    def runner(self, **kw):
        args = dict(speak_self=self.speak_self, device_say=self.device_say,
                    device_done=self.device_done, listen=self.listen,
                    session=self.session, beat_s=0.01, busy_step_s=0.02,
                    busy_retry_s=0.1, device_wait_s=2.0)
        args.update(kw)
        return dlg.Runner(**args)


def _lines():
    return [dlg.Line("device", "The toast is late.", ("The toast is late.",)),
            dlg.Line("self", "It keeps its own hours."),
            dlg.Line("device", "I respect that.", ("I respect that.",)),
            dlg.Line("self", "A kindred spirit. Barely.", final=True)]


def _ready(value):
    f = Future()
    f.set_result(value)
    return f


class RunnerTests(unittest.TestCase):
    def run_rig(self, rig, script=None, **kw):
        runner = rig.runner()
        return runner.run("Opening line.", script or _ready(_lines()),
                          lambda: [dlg.Line("self", "Canned.", final=True)],
                          script_deadline=time.monotonic() + 2.0, **kw)

    def test_happy_path_order_and_count(self):
        rig = Rig()
        out = self.run_rig(rig)
        self.assertEqual(out, dlg.Outcome(5, "done"))
        kinds = [e[0] for e in rig.log]
        self.assertEqual(kinds, ["self", "say", "listen_start", "listen_end",
                                 "self", "say", "listen_start", "listen_end",
                                 "self"])
        self.assertFalse(rig.device_talking_during_self)
        self.assertFalse(rig.listen_during_self)
        self.assertTrue(rig.log[-1][2])       # final flag passed through

    def test_listen_always_between_say_ok_and_next_self_line(self):
        rig = Rig(device_s=0.08)
        self.run_rig(rig)
        for i, e in enumerate(rig.log):
            if e[0] == "say" and e[2] == "ok":
                self.assertEqual(rig.log[i + 1][0], "listen_start")
                self.assertEqual(rig.log[i + 2], ("listen_end", True))

    def test_blocking_speak_is_cut_from_another_thread(self):
        rig = Rig(speak_block="It keeps its own hours.")
        threading.Timer(0.15, rig.session.stop, args=("device_lost",)).start()
        t0 = time.monotonic()
        out = self.run_rig(rig, closings={"device_lost": None})
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(out.reason, "device_lost")
        self.assertEqual(out.lines_spoken, 2)     # opener + first device line
        self.assertEqual(rig.log[-1][0], "self")  # nothing after the cut

    def test_final_device_done_awaited_before_return(self):
        lines = [dlg.Line("device", "Last word is mine.",
                          ("Last word is mine.",))]
        rig = Rig(device_s=0.3, cap=FakeCap(available=False))
        out = self.run_rig(rig, script=_ready(lines))
        self.assertEqual(out.reason, "done")
        self.assertTrue(rig.device_done())

    def test_unavailable_capture_still_waits_for_device_then_beat(self):
        rig = Rig(device_s=0.2, cap=FakeCap(available=False))
        out = self.run_rig(rig)
        self.assertEqual(out.reason, "done")
        self.assertFalse(rig.device_talking_during_self)

    def test_stall_line_when_script_is_late(self):
        rig = Rig()
        fut = Future()
        threading.Timer(0.2, fut.set_result, args=(_lines(),)).start()
        out = self.run_rig(rig, script=fut, stall=lambda: "Thinking hard.")
        self.assertEqual(out.reason, "done")
        says = [e[1] for e in rig.log if e[0] == "say"]
        self.assertEqual(says[0], "Thinking hard.")

    def test_no_stall_when_script_is_ready(self):
        rig = Rig()
        self.run_rig(rig, stall=lambda: "Thinking hard.")
        self.assertNotIn("Thinking hard.",
                         [e[1] for e in rig.log if e[0] == "say"])

    def test_deadline_falls_back_to_canned(self):
        rig = Rig()
        runner = rig.runner()
        out = runner.run("Opening line.", Future(),
                         lambda: [dlg.Line("self", "Canned.", final=True)],
                         script_deadline=time.monotonic() + 0.1)
        self.assertEqual(out, dlg.Outcome(2, "done"))
        self.assertEqual(rig.log[-1][1], "Canned.")

    def test_none_script_falls_back(self):
        rig = Rig()
        out = self.run_rig(rig, script=_ready(None))
        self.assertEqual(out, dlg.Outcome(2, "done"))

    def test_busy_retried_then_ok(self):
        rig = Rig(say_results=["busy", "busy", "ok"])
        out = self.run_rig(rig)
        self.assertEqual(out.reason, "done")
        self.assertEqual([e[2] for e in rig.log if e[0] == "say"][:3],
                         ["busy", "busy", "ok"])

    def test_busy_gives_up_with_its_closing(self):
        rig = Rig(say_results=["busy"] * 50)
        out = self.run_rig(rig, closings={"device_busy": "Lost interest."})
        self.assertEqual(out.reason, "device_busy")
        self.assertEqual(rig.log[-1][1], "Lost interest.")
        self.assertTrue(rig.log[-1][2])

    def test_device_muted_and_heap(self):
        for res, reason in (("muted", "device_muted"),
                            ("heap", "device_heap"),
                            ("no_answer", "device_no_answer"),
                            ("refused", "error")):
            rig = Rig(say_results=[res])
            out = self.run_rig(rig)
            self.assertEqual(out.reason, reason)

    def test_owner_stop_after_device_line_speaks_closing(self):
        rig = Rig()
        orig = rig.listen

        def listen(until, *, beat_s, max_s):
            cap = orig(until, beat_s=beat_s, max_s=max_s)
            rig.session.stop("owner_stop")
            return cap
        rig.listen = listen
        out = self.run_rig(rig, closings={"owner_stop": "Of course."})
        self.assertEqual(out.reason, "owner_stop")
        self.assertEqual(rig.log[-1][1], "Of course.")

    def test_voiced_beat_waits_for_the_verdict(self):
        cap = FakeCap(beat_voiced=True, delay=0.3)
        rig = Rig(cap=cap)
        t0 = time.monotonic()
        self.run_rig(rig)
        self.assertEqual(cap.waited, 1.5)
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)

    def test_unvoiced_beat_does_not_wait(self):
        cap = FakeCap(beat_voiced=False, delay=5.0)
        rig = Rig(cap=cap)
        t0 = time.monotonic()
        self.run_rig(rig)
        self.assertIsNone(cap.waited)
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_tts_muted_speak_ends_run(self):
        rig = Rig(speak_results=["spoken", "muted"])
        out = self.run_rig(rig)
        self.assertEqual(out.reason, "tts_muted")

    def test_interrupted_opener(self):
        rig = Rig(speak_results=["interrupted"])
        out = self.run_rig(rig)
        self.assertEqual(out, dlg.Outcome(0, "interrupted"))
        self.assertEqual([e[0] for e in rig.log], ["self"])

    def test_already_stopped_speaks_nothing(self):
        rig = Rig()
        rig.session.stop("wake")
        out = self.run_rig(rig)
        self.assertEqual(out, dlg.Outcome(0, "wake"))
        self.assertEqual(rig.log, [])

    def test_preflight_failure_after_opener(self):
        rig = Rig()
        out = self.run_rig(rig, preflight=_ready("device_no_answer"),
                           closings={"device_no_answer": "No answer."})
        self.assertEqual(out, dlg.Outcome(1, "device_no_answer"))
        self.assertEqual([e[1] for e in rig.log],
                         ["Opening line.", "No answer."])

    def test_preflight_ok_continues(self):
        rig = Rig()
        out = self.run_rig(rig, preflight=_ready(""))
        self.assertEqual(out.reason, "done")

    def test_session_expiry_between_lines(self):
        rig = Rig()
        orig = rig.speak_self

        def speak(text, final):
            r = orig(text, final)
            if text == "It keeps its own hours.":
                rig.session.reason = "expired"
            return r
        rig.speak_self = speak
        out = self.run_rig(rig)
        self.assertEqual(out.reason, "expired")
        self.assertEqual(out.lines_spoken, 3)

    def test_callbacks_raising_never_escape(self):
        rig = Rig()

        def bad_say(_c):
            raise RuntimeError("x")
        out = rig.runner(device_say=bad_say).run(
            "Opening line.", _ready(_lines()), lambda: [],
            script_deadline=time.monotonic() + 1.0)
        self.assertEqual(out.reason, "error")


class ListenCaptureTests(unittest.TestCase):
    def test_pending_then_result(self):
        cap = dlg.ListenCapture(available=True, beat_voiced=True)
        self.assertEqual(cap.result(0), ("pending", ""))
        cap.set_result("stop", "ignored text")
        self.assertEqual(cap.result(0), ("stop", ""))

    def test_speech_carries_text_and_unknown_kind_is_empty(self):
        cap = dlg.ListenCapture()
        cap.set_result("speech", "hello there")
        self.assertEqual(cap.result(), ("speech", "hello there"))
        cap2 = dlg.ListenCapture()
        cap2.set_result("bogus", "x")
        self.assertEqual(cap2.result(), ("", ""))

    def test_unavailable(self):
        cap = dlg.ListenCapture.unavailable()
        self.assertFalse(cap.available)
        self.assertEqual(cap.result(0), ("", ""))


if __name__ == "__main__":
    unittest.main()
