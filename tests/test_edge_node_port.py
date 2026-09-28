"""Edge-node fixes ported from the Dell's local-only patches (v2.0.109).

The Dell laptop runs JARVIS as an edge node whose LLM "brain" is another
machine over the network. It carried three local patches against v2.0.104; this
file pins how each was ported to main:

  A. Remote-brain Ollama probes -- a REAL FIX, ported and GENERALISED. The Dell
     fixed one probe (``_ollama_alive``); five more hit the same server with the
     same hardcoded 2 s timeout and false-fail the same way against a busy
     remote GPU. All six now take their timeout from one helper, and a static
     scan below fails if a hardcoded ``timeout=2`` probe ever reappears.
  B. The wake-word follow-up window -- a FEATURE, ported OFF by default so the
     desktop's behaviour is unchanged; an install opts in via user_settings.
  C. Speech-filter thresholds -- DEVICE TUNING for the Dell's microphone, so main
     keeps its defaults and gains a validated per-install override instead.

Test data uses documentation address ranges (RFC 5737), never a real host.

The first four classes need only the stdlib and run on the light-deps CI
runner. The last class drives the real monolith wiring and runs in the local
full-deps tier (``@requires_monolith`` skips it on CI).
"""
from __future__ import annotations

import os
import re
import unittest
from unittest import mock

from core import ollama_opts, speech_filter
from core.followup_window import FollowupWindow
from tests._monolith_harness import load_monolith, requires_monolith

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOOD_CONF = {"no_speech_prob": 0.1, "avg_logprob": -0.5}


# ── A. endpoint locality → probe timeout ────────────────────────────────────
class EndpointLocalityTests(unittest.TestCase):
    LOCAL = [
        "http://127.0.0.1:11434",
        "http://127.0.0.2:11434",          # the whole 127/8 block is loopback
        "http://localhost:11434",
        "http://LOCALHOST:11434/",
        "http://[::1]:11434",
        "http://0.0.0.0:11434",
        "127.0.0.1:11434",                 # scheme-less
        "http://localhost",                # no port
    ]
    REMOTE = [
        "http://192.0.2.10:11434",         # RFC 5737 TEST-NET-1
        "http://203.0.113.7:11434/",       # RFC 5737 TEST-NET-3
        "http://gpu-box.example:11434",
        "http://[2001:db8::1]:11434",      # RFC 3849 documentation v6
        "https://brain.example/ollama",
    ]

    def test_local_endpoints_are_not_remote(self):
        for url in self.LOCAL:
            with self.subTest(url=url):
                self.assertFalse(ollama_opts.endpoint_is_remote(url))
                self.assertEqual(ollama_opts.probe_timeout(url),
                                 ollama_opts.PROBE_TIMEOUT_LOCAL_S)

    def test_remote_endpoints_are_remote(self):
        for url in self.REMOTE:
            with self.subTest(url=url):
                self.assertTrue(ollama_opts.endpoint_is_remote(url))
                self.assertEqual(ollama_opts.probe_timeout(url),
                                 ollama_opts.PROBE_TIMEOUT_REMOTE_S)

    def test_unparseable_falls_back_to_local_legacy_behaviour(self):
        # Treating junk as local keeps the pre-port behaviour (2 s probe,
        # self-heal allowed) rather than inventing a remote brain.
        for bad in ("", None, "http://", "::::"):
            with self.subTest(bad=bad):
                self.assertFalse(ollama_opts.endpoint_is_remote(bad))

    def test_remote_timeout_is_longer_than_local(self):
        self.assertGreater(ollama_opts.PROBE_TIMEOUT_REMOTE_S,
                           ollama_opts.PROBE_TIMEOUT_LOCAL_S)


class NoHardcodedProbeTimeoutTests(unittest.TestCase):
    """Stale-duplicate guard: the Dell fixed ONE of six probes. Any probe of the
    LLM server that pins its own short timeout again would silently re-open the
    remote false-failure for that call site. Reads the source as TEXT, so it
    runs on CI where the monolith cannot be imported."""

    def test_no_llm_probe_pins_a_literal_timeout(self):
        with open(os.path.join(_REPO, "bobert_companion.py"), encoding="utf-8") as fh:
            src = fh.read()
        pinned = re.findall(
            r'LOCAL_LLM_BASE_URL\}/api/(?:tags|ps)"\s*,\s*timeout\s*=\s*\d', src)
        self.assertEqual(pinned, [], "a /api/tags or /api/ps probe pins a literal "
                         "timeout -- use _ollama_probe_timeout()")

    def test_the_helper_is_actually_used(self):
        # The guard above passes vacuously if every probe were deleted; make
        # sure the probes still exist and route through the helper.
        with open(os.path.join(_REPO, "bobert_companion.py"), encoding="utf-8") as fh:
            src = fh.read()
        routed = re.findall(
            r'LOCAL_LLM_BASE_URL\}/api/(?:tags|ps)"\s*,\s*timeout\s*=\s*_ollama_probe_timeout\(\)',
            src)
        self.assertGreaterEqual(len(routed), 6)


class ShippedDefaultsTests(unittest.TestCase):
    """What main SHIPS, read from source -- never from the live config, which
    merges this machine's gitignored user_settings.json and would make these
    pass or fail depending on the box (the v2.0.108 lesson)."""

    def _src(self, rel):
        with open(os.path.join(_REPO, rel), encoding="utf-8") as fh:
            return fh.read()

    def test_new_knobs_default_to_legacy_behaviour(self):
        cfg = self._src(os.path.join("core", "config.py"))
        self.assertRegex(cfg, r"(?m)^FOLLOWUP_WINDOW_S = 0\.0\s*$")
        self.assertRegex(cfg, r"(?m)^SPEECH_FILTER_OVERRIDES = \{\}\s*$")

    def test_trust_rms_is_reread_after_overrides_are_applied(self):
        # The monolith imports WHISPER_TRUST_RMS BY VALUE near the top; the
        # overrides are applied later, after config loads. If the re-read is
        # missing or comes first, that copy goes stale on any box with overrides.
        src = self._src("bobert_companion.py")
        apply_at = src.find("_speech_filter_mod.apply_overrides(SPEECH_FILTER_OVERRIDES)")
        reread_at = src.find("\nWHISPER_TRUST_RMS = _speech_filter_mod.WHISPER_TRUST_RMS\n")
        self.assertGreater(apply_at, 0, "overrides are never applied")
        self.assertGreater(reread_at, apply_at,
                           "WHISPER_TRUST_RMS is not re-read after apply_overrides")


# ── B. follow-up window ─────────────────────────────────────────────────────
class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FollowupWindowTests(unittest.TestCase):
    def test_default_is_disabled_and_never_admits(self):
        c = _Clock()
        w = FollowupWindow(clock=c)
        self.assertFalse(w.enabled)
        w.note_addressed()
        self.assertFalse(w.admit())
        self.assertEqual(w.remaining_s(), 0.0)

    def test_zero_and_negative_disable(self):
        for v in (0, 0.0, -5, "0"):
            with self.subTest(v=v):
                w = FollowupWindow(v, clock=_Clock())
                w.note_addressed()
                self.assertFalse(w.admit())

    def test_malformed_setting_disables_rather_than_widens(self):
        for v in ("abc", None, object()):
            with self.subTest(v=v):
                self.assertFalse(FollowupWindow(v, clock=_Clock()).enabled)

    def test_nothing_admitted_before_the_user_addresses_jarvis(self):
        w = FollowupWindow(45, clock=_Clock())
        self.assertFalse(w.admit())

    def test_opens_on_address_and_lapses(self):
        c = _Clock()
        w = FollowupWindow(45, clock=c)
        w.note_addressed()
        c.t += 44.9
        self.assertTrue(w.admit())          # inside; and this EXTENDS it
        c.t += 44.9
        self.assertTrue(w.admit())          # still inside thanks to the extension
        c.t += 45.1
        self.assertFalse(w.admit())         # conversation stopped -> lapsed

    def test_admission_extends_the_window(self):
        # Pins the KNOWN RISK documented in the module: admission alone keeps
        # it open. If this behaviour is ever capped, this test should change
        # deliberately, not by accident.
        c = _Clock()
        w = FollowupWindow(10, clock=c)
        w.note_addressed()
        for _ in range(20):                 # 20 * 9 s = 180 s of "crosstalk"
            c.t += 9
            self.assertTrue(w.admit())


# ── C. speech-filter per-install overrides ──────────────────────────────────
class SpeechFilterOverrideTests(unittest.TestCase):
    def setUp(self):
        speech_filter.reset_overrides()     # start from shipped defaults

    def tearDown(self):
        speech_filter.reset_overrides()     # module globals must not leak

    def test_defaults_unchanged_on_main(self):
        self.assertEqual(speech_filter.WHISPER_MIN_WORDS, 2)
        self.assertEqual(speech_filter.WHISPER_MIN_AVG_LOGPROB, -1.5)
        self.assertEqual(speech_filter.WHISPER_TRUST_RMS, 0.025)

    def test_two_word_command_passes_on_main_defaults(self):
        self.assertEqual(speech_filter.is_valid_speech("lights off", _GOOD_CONF, 0.01),
                         (True, ""))

    def test_dell_min_words_would_drop_a_two_word_command(self):
        # Why the Dell's MIN_WORDS=3 is device tuning and NOT a main default.
        applied = speech_filter.apply_overrides({"WHISPER_MIN_WORDS": 3})
        self.assertEqual(applied, {"WHISPER_MIN_WORDS": 3})
        ok, reason = speech_filter.is_valid_speech("lights off", _GOOD_CONF, 0.01)
        self.assertFalse(ok)
        self.assertIn("too short", reason)

    def test_dell_trust_rms_and_logprob_change_the_verdict(self):
        conf = {"no_speech_prob": 0.1, "avg_logprob": -1.3}
        text = "please turn the lights off"
        # main: peak 0.05 >= 0.025 -> trusted on loudness
        self.assertTrue(speech_filter.is_valid_speech(text, conf, 0.05)[0])
        speech_filter.apply_overrides({"WHISPER_TRUST_RMS": 0.15,
                                       "WHISPER_MIN_AVG_LOGPROB": -1.15})
        ok, reason = speech_filter.is_valid_speech(text, conf, 0.05)
        self.assertFalse(ok)
        self.assertIn("low confidence", reason)

    def test_reset_restores_every_default(self):
        speech_filter.apply_overrides({"WHISPER_MIN_WORDS": 3,
                                       "WHISPER_MIN_AVG_LOGPROB": -1.15,
                                       "WHISPER_TRUST_RMS": 0.15,
                                       "WHISPER_MAX_NO_SPEECH_PROB": 0.5})
        speech_filter.reset_overrides()
        self.test_defaults_unchanged_on_main()
        self.assertEqual(speech_filter.WHISPER_MAX_NO_SPEECH_PROB, 0.85)

    def test_rejects_unknown_wrong_type_and_out_of_range(self):
        applied = speech_filter.apply_overrides({
            "WAKE_WORD": "hey",                      # not overridable
            "WHISPER_HALLUCINATIONS": [],            # not overridable
            "WHISPER_MIN_WORDS": True,               # bool is not an int here
            "WHISPER_TRUST_RMS": 5.0,                # out of range
            "WHISPER_MIN_AVG_LOGPROB": "abc",        # not a number
        })
        self.assertEqual(applied, {})
        self.test_defaults_unchanged_on_main()

    def test_non_integral_min_words_rejected_not_truncated(self):
        self.assertEqual(speech_filter.apply_overrides({"WHISPER_MIN_WORDS": 2.7}), {})
        self.assertEqual(speech_filter.WHISPER_MIN_WORDS, 2)

    def test_non_dict_input_is_ignored(self):
        for bad in (None, [], "WHISPER_MIN_WORDS=3", 3):
            with self.subTest(bad=bad):
                self.assertEqual(speech_filter.apply_overrides(bad), {})


# ── wiring through the real monolith (local full-deps tier) ─────────────────
@requires_monolith
class MonolithWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_reexported_trust_rms_is_in_lockstep(self):
        # Runtime half. On its own this CANNOT fail on a box with no overrides
        # (both sides hold the default) -- mutation-tested 2026-09-28 -- so the
        # source-order guard in ShippedDefaultsTests carries the real weight.
        self.assertEqual(self.bc.WHISPER_TRUST_RMS, speech_filter.WHISPER_TRUST_RMS)

    def test_alive_probe_uses_the_remote_timeout(self):
        for url, want in (("http://127.0.0.1:11434", ollama_opts.PROBE_TIMEOUT_LOCAL_S),
                          ("http://192.0.2.10:11434", ollama_opts.PROBE_TIMEOUT_REMOTE_S)):
            with self.subTest(url=url), \
                    mock.patch.object(self.bc, "LOCAL_LLM_BASE_URL", url), \
                    mock.patch.object(self.bc.requests, "get") as get:
                get.return_value.ok = True
                self.assertTrue(self.bc._ollama_alive())
                self.assertEqual(get.call_args.kwargs["timeout"], want)

    def _self_heal(self, url):
        """Run _ensure_ollama_running with every EARLIER exit neutralised
        (staging, ollama.exe lookup, the three 4 s rechecks, the deadline loop),
        so reaching Popen depends only on the remote-brain gate."""
        with mock.patch.object(self.bc, "LOCAL_LLM_BASE_URL", url), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=False), \
                mock.patch.object(self.bc, "_is_staging", return_value=False), \
                mock.patch.dict(os.environ, {"JARVIS_STAGING": ""}), \
                mock.patch("shutil.which", return_value=r"C:\fake\ollama.exe"), \
                mock.patch.object(self.bc.time, "sleep"), \
                mock.patch.object(self.bc, "_reap_wedged_ollama") as reap, \
                mock.patch.object(self.bc.subprocess, "Popen") as popen:
            ok = self.bc._ensure_ollama_running(timeout_sec=0)
        return ok, popen, reap

    def test_positive_control_a_local_brain_is_self_healed(self):
        # Without this, "Popen not called" below would pass even if the gate were
        # deleted -- it did, mutation-tested 2026-09-28, because an earlier exit
        # returned first. This proves the harness genuinely reaches Popen.
        ok, popen, reap = self._self_heal("http://127.0.0.1:11434")
        popen.assert_called_once()
        reap.assert_called_once()

    def test_remote_brain_is_never_self_healed_locally(self):
        ok, popen, reap = self._self_heal("http://192.0.2.10:11434")
        self.assertFalse(ok)
        popen.assert_not_called()
        reap.assert_not_called()

    def _gate(self, text):
        return self.bc._should_refuse_background_audio(text)

    def test_wake_mode_without_window_refuses_follow_ups(self):
        with mock.patch.object(self.bc, "_require_wake_runtime", True), \
                mock.patch.object(self.bc, "_followup_window", FollowupWindow(0)):
            self.assertEqual(self._gate("jarvis what time is it"), (False, ""))
            self.assertEqual(self._gate("and tomorrow"), (True, "wake-word mode"))

    def test_wake_mode_with_window_admits_then_lapses(self):
        c = _Clock()
        with mock.patch.object(self.bc, "_require_wake_runtime", True), \
                mock.patch.object(self.bc, "_followup_window", FollowupWindow(45, clock=c)):
            self.assertEqual(self._gate("jarvis what time is it"), (False, ""))
            c.t += 30
            self.assertEqual(self._gate("and tomorrow"), (False, "follow-up window"))
            c.t += 46
            self.assertEqual(self._gate("and the day after"), (True, "wake-word mode"))


if __name__ == "__main__":
    unittest.main()
