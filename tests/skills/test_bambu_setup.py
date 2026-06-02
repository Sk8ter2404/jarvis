"""Logic tests for skills/bambu_setup.py.

The first-time printer wizard is mostly pure parsing helpers plus a guarded
inline-args entry point and a multicast-discovery + voice flow. We cover:

  • _parse_bambu_packet  — NOTIFY-shaped UDP payload → {ip, serial, model, name}
  • discover_printers     — the multicast socket listen loop (socket fully faked:
                            no real bind/recv), dedupe-by-IP, fallback-to-sender,
                            and the OSError "no multicast" path.
  • _voice_to_digits     — spoken digits / numbers → digit string
  • _voice_to_ip         — dotted / spoken IPv4 extraction
  • _affirmative / _negative — yes/no intent
  • _format_digits_for_speech / _humanise_printer
  • _say / _listen        — the TTS/STT bridge (bobert_companion faked) + fallbacks
  • _parse_inline_args    — the non-voice "ip access [serial]" shortcut
  • _wizard_pick_printer  — single / multi / index / model-name selection + misses
  • _wizard_prompt_ip / _wizard_prompt_access_code / _wizard_prompt_serial —
                            the re-prompt state machines (3-attempt loops).
  • _persist_credentials  — live-attr patch + source rewrite, every branch, with
                            a FAKE bobert_companion module and a TEMP config file
                            so NO real bobert_companion.py source is ever rewritten.
  • _restart_monitor      — skill_bambu_monitor.start_monitor hot-restart paths.
  • setup_printer         — inline-args + the "already running" lock + the full
                            discovery/voice flow (every terminal branch).

register() only wires actions (no thread, no I/O), so it loads cleanly. The
wizard's voice helpers (_say/_listen) are patched wherever a path reaches them.
Every IP/serial/access-code fixture is generic (192.168.1.x / 12345678 / 01P...).
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── fake-module helpers ─────────────────────────────────────────────────
_SENTINEL = object()


class _FakeBC(types.ModuleType):
    """A stand-in bobert_companion module exposing only what bambu_setup
    touches: the three BAMBU_* attrs (_persist patches these live), plus the
    speech/record/transcribe bridge used by _say/_listen."""

    def __init__(self):
        super().__init__("bobert_companion")
        self.BAMBU_PRINTER_IP = ""
        self.BAMBU_ACCESS_CODE = ""
        self.BAMBU_SERIAL = ""


def _inject_module(test, name, obj):
    """Install ``obj`` as ``sys.modules[name]`` for the duration of ``test``,
    restoring the prior state (including absence) on cleanup. Mirrors the
    save/restore contract used across the suite — never a bare module-level
    write."""
    saved = sys.modules.get(name, _SENTINEL)

    def _restore():
        if saved is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved

    test.addCleanup(_restore)
    if obj is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = obj
    return obj


class BambuSetupParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    # ── _parse_bambu_packet ──────────────────────────────────────────────
    def test_parse_packet_extracts_fields(self):
        payload = (
            b"NOTIFY * HTTP/1.1\r\n"
            b"Location: 192.168.1.42\r\n"
            b"USN: 01P00A123456789\r\n"
            b"DevModel.bambu.com: BL-P001\r\n"
            b"DevName.bambu.com: My H2D\r\n"
            b"From: bambulab\r\n"
        )
        out = self.mod._parse_bambu_packet(payload)
        self.assertEqual(out["ip"], "192.168.1.42")
        self.assertEqual(out["serial"], "01P00A123456789")
        self.assertEqual(out["model"], "BL-P001")
        self.assertEqual(out["name"], "My H2D")

    def test_parse_packet_rejects_non_bambu(self):
        self.assertEqual(
            self.mod._parse_bambu_packet(b"Location: 10.0.0.1\r\nrandom upnp"),
            {})

    def test_parse_packet_requires_ip(self):
        # bambu signature present but no Location → useless, empty dict.
        self.assertEqual(
            self.mod._parse_bambu_packet(b"USN: abc\r\nfrom bambulab device"),
            {})

    # ── _voice_to_digits ─────────────────────────────────────────────────
    def test_voice_to_digits_words(self):
        self.assertEqual(self.mod._voice_to_digits("one two three four"), "1234")

    def test_voice_to_digits_mixed_and_homophones(self):
        # "to"/"for" homophones map to 2/4; literal digits pass through.
        self.assertEqual(self.mod._voice_to_digits("to for 5 6"), "2456")

    def test_voice_to_digits_compound_numbers(self):
        # "twenty" → "20", "three" → "3"
        self.assertEqual(self.mod._voice_to_digits("twenty three"), "203")

    def test_voice_to_digits_empty(self):
        self.assertEqual(self.mod._voice_to_digits(""), "")
        self.assertEqual(self.mod._voice_to_digits("hello there"), "")

    # ── _voice_to_ip ─────────────────────────────────────────────────────
    def test_voice_to_ip_dotted_direct(self):
        self.assertEqual(self.mod._voice_to_ip("it's 192.168.1.42 sir"),
                         "192.168.1.42")

    def test_voice_to_ip_spoken_with_dot(self):
        out = self.mod._voice_to_ip("one nine two dot one six eight dot one dot four two")
        self.assertEqual(out, "192.168.1.42")

    def test_voice_to_ip_rejects_out_of_range_octet(self):
        # 300 is not a valid octet → no four-octet result.
        self.assertEqual(
            self.mod._voice_to_ip("three zero zero dot one dot one dot one"), "")

    def test_voice_to_ip_empty_when_unparseable(self):
        self.assertEqual(self.mod._voice_to_ip("no numbers here"), "")

    # ── intent helpers ───────────────────────────────────────────────────
    def test_affirmative(self):
        for w in ("yes", "Yeah", "correct", "that's right", "go ahead"):
            self.assertTrue(self.mod._affirmative(w))
        self.assertFalse(self.mod._affirmative("no"))
        self.assertFalse(self.mod._affirmative(""))

    def test_negative(self):
        for w in ("no", "Nope", "wrong", "cancel"):
            self.assertTrue(self.mod._negative(w))
        self.assertFalse(self.mod._negative("yes"))
        self.assertFalse(self.mod._negative(""))

    # ── small formatters ─────────────────────────────────────────────────
    def test_format_digits_for_speech(self):
        self.assertEqual(self.mod._format_digits_for_speech("12345678"),
                         "1-2-3-4-5-6-7-8")

    def test_humanise_printer(self):
        self.assertEqual(
            self.mod._humanise_printer({"name": "My H2D", "ip": "10.0.0.5"}),
            "My H2D at 10.0.0.5")
        # Falls back to model, then "printer".
        self.assertEqual(
            self.mod._humanise_printer({"model": "BL-P001", "ip": "10.0.0.5"}),
            "BL-P001 at 10.0.0.5")
        self.assertEqual(self.mod._humanise_printer({"ip": "10.0.0.5"}),
                         "printer at 10.0.0.5")

    # ── _parse_inline_args ───────────────────────────────────────────────
    def test_inline_args_three_tokens(self):
        self.assertEqual(
            self.mod._parse_inline_args("192.168.1.5 12345678 01P00A99"),
            ("192.168.1.5", "12345678", "01P00A99"))

    def test_inline_args_two_tokens_serial_blank(self):
        self.assertEqual(self.mod._parse_inline_args("192.168.1.5 12345678"),
                         ("192.168.1.5", "12345678", ""))

    def test_inline_args_rejects_bad_ip(self):
        self.assertIsNone(self.mod._parse_inline_args("not-an-ip 12345678"))

    def test_inline_args_rejects_bad_access_code(self):
        # access code must be 6-12 digits
        self.assertIsNone(self.mod._parse_inline_args("192.168.1.5 abc"))

    def test_inline_args_rejects_wrong_token_count(self):
        self.assertIsNone(self.mod._parse_inline_args("192.168.1.5"))
        self.assertIsNone(self.mod._parse_inline_args(""))


class BambuSetupWizardPickTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_single_printer_confirmed(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="yes"):
            chosen = self.mod._wizard_pick_printer([{"ip": "10.0.0.1",
                                                     "name": "H2D"}])
        self.assertEqual(chosen["ip"], "10.0.0.1")

    def test_single_printer_declined(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="no"):
            self.assertIsNone(
                self.mod._wizard_pick_printer([{"ip": "10.0.0.1"}]))

    def test_multiple_pick_by_index(self):
        printers = [{"ip": "10.0.0.1", "name": "A"},
                    {"ip": "10.0.0.2", "name": "B"}]
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="two"):
            chosen = self.mod._wizard_pick_printer(printers)
        self.assertEqual(chosen["ip"], "10.0.0.2")

    def test_multiple_pick_by_model_name(self):
        printers = [{"ip": "10.0.0.1", "name": "Prusa"},
                    {"ip": "10.0.0.2", "name": "H2D"}]
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               return_value="the h2d please"):
            chosen = self.mod._wizard_pick_printer(printers)
        self.assertEqual(chosen["ip"], "10.0.0.2")


class BambuSetupActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_setup_inline_args_persists_and_restarts(self):
        # Stub the two side-effecting helpers so NO real source file is
        # rewritten and no monitor is actually started.
        with mock.patch.object(self.mod, "_persist_credentials",
                               return_value=True) as persist, \
             mock.patch.object(self.mod, "_restart_monitor", return_value=True):
            out = self.actions["setup_printer"](
                "192.168.1.50 12345678 01P00A0001")
        persist.assert_called_once_with("192.168.1.50", "12345678", "01P00A0001")
        self.assertIn("online", out.lower())

    def test_setup_inline_args_persist_ok_monitor_idle(self):
        with mock.patch.object(self.mod, "_persist_credentials",
                               return_value=True), \
             mock.patch.object(self.mod, "_restart_monitor", return_value=False):
            out = self.actions["setup_printer"]("192.168.1.50 12345678")
        self.assertIn("next launch", out.lower())

    def test_setup_inline_args_persist_failure(self):
        with mock.patch.object(self.mod, "_persist_credentials",
                               return_value=False), \
             mock.patch.object(self.mod, "_restart_monitor", return_value=True):
            out = self.actions["setup_printer"]("192.168.1.50 12345678")
        self.assertIn("couldn't write the credentials", out.lower())

    def test_setup_lock_blocks_concurrent_run(self):
        # Hold the wizard lock so the action reports the busy message instead of
        # entering the voice flow.
        self.assertTrue(self.mod._wizard_lock.acquire(blocking=False))
        try:
            out = self.actions["setup_printer"]("")
        finally:
            self.mod._wizard_lock.release()
        self.assertIn("already running", out.lower())

    def test_all_action_aliases_registered(self):
        for name in ("setup_printer", "setup_bambu", "configure_printer",
                     "bambu_setup", "first_time_printer_setup"):
            self.assertIn(name, self.actions)


# ─────────────────────────────────────────────────────────────────────────
# _parse_bambu_packet — extra branches (decode error, sender fallback feed).
# ─────────────────────────────────────────────────────────────────────────
class BambuPacketEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_packet_3dprinter_signature(self):
        # The alternate signature ("3dprinter") also passes the filter.
        payload = b"NOTIFY\r\nLocation: 192.168.1.7\r\nServer: 3DPrinter/1.0\r\n"
        out = self.mod._parse_bambu_packet(payload)
        self.assertEqual(out["ip"], "192.168.1.7")

    def test_packet_lines_without_colon_skipped(self):
        payload = (b"NOTIFY bambulab\r\nthislinehasnocolon\r\n"
                   b"Location: 192.168.1.8\r\n")
        out = self.mod._parse_bambu_packet(payload)
        self.assertEqual(out["ip"], "192.168.1.8")

    def test_packet_decode_failure_returns_empty(self):
        # A payload whose .decode raises (not just produces mojibake) → {}.
        class _Boom(bytes):
            def decode(self, *a, **k):
                raise UnicodeError("boom")

        self.assertEqual(self.mod._parse_bambu_packet(_Boom()), {})


# ─────────────────────────────────────────────────────────────────────────
# discover_printers — the multicast socket listen loop, fully faked.
# ─────────────────────────────────────────────────────────────────────────
class _FakeSocket:
    """A drop-in for socket.socket: records option/bind calls and yields a
    scripted sequence of recvfrom results. Each script item is either a
    (payload_bytes, (sender_ip, port)) tuple OR the socket.timeout class to
    raise a timeout for that iteration."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.closed = False
        self.opts = []
        self.bound = None
        self.timeout = None

    def setsockopt(self, *a):
        self.opts.append(a)

    def bind(self, addr):
        self.bound = addr

    def settimeout(self, t):
        self.timeout = t

    def recvfrom(self, _n):
        if self._i >= len(self._script):
            raise socket.timeout()
        item = self._script[self._i]
        self._i += 1
        if item is socket.timeout:
            raise socket.timeout()
        return item

    def close(self):
        self.closed = True


class DiscoverPrintersTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")
        # Freeze the clock so the duration loop runs a deterministic number of
        # iterations: time.time() ticks one "second" per call.
        self._clock = [1000.0]

        def _tick():
            self._clock[0] += 1.0
            return self._clock[0]

        mock.patch.object(self.mod.time, "time", _tick).start()
        self.addCleanup(mock.patch.stopall)

    def _run_with_socket(self, sock, duration=2.0):
        with mock.patch.object(self.mod.socket, "socket", return_value=sock):
            return self.mod.discover_printers(duration=duration)

    def test_discovers_and_dedupes_by_ip(self):
        pkt = (b"NOTIFY bambulab\r\nLocation: 192.168.1.10\r\n"
               b"USN: 01P00A000000001\r\nDevName.bambu.com: Shop H2D\r\n")
        # Same printer announced twice → one entry; bind/opt calls recorded.
        sock = _FakeSocket([(pkt, ("192.168.1.10", 2021)),
                            (pkt, ("192.168.1.10", 2021))])
        found = self._run_with_socket(sock, duration=4.0)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["ip"], "192.168.1.10")
        self.assertEqual(found[0]["serial"], "01P00A000000001")
        self.assertEqual(found[0]["name"], "Shop H2D")
        self.assertTrue(sock.closed)
        self.assertEqual(sock.bound, ("", self.mod.BAMBU_MCAST_PORT))

    def test_timeout_iterations_then_packet(self):
        pkt = b"NOTIFY bambulab\r\nLocation: 192.168.1.11\r\n"
        sock = _FakeSocket([socket.timeout, (pkt, ("192.168.1.11", 2021))])
        found = self._run_with_socket(sock, duration=5.0)
        self.assertEqual([p["ip"] for p in found], ["192.168.1.11"])

    def test_non_bambu_packet_ignored(self):
        good = b"NOTIFY bambulab\r\nLocation: 192.168.1.12\r\n"
        sock = _FakeSocket([(b"random upnp chatter", ("10.0.0.9", 1900)),
                            (good, ("192.168.1.12", 2021))])
        found = self._run_with_socket(sock, duration=5.0)
        self.assertEqual([p["ip"] for p in found], ["192.168.1.12"])

    def test_location_missing_falls_back_to_sender_addr(self):
        # Bambu signature present, USN present, but NO Location line — the
        # parser returns {} so the packet is skipped (Location is required by
        # _parse_bambu_packet). NOTE: discover_printers has a sender-addr
        # fallback (ip = parsed.get('ip') or addr[0]) that is therefore
        # unreachable for signature-but-no-Location packets, since the parser
        # already filtered them. Documented, not a test of the fallback.
        sock = _FakeSocket([(b"NOTIFY bambulab\r\nUSN: 01P00A2\r\n",
                             ("192.168.1.99", 2021))])
        found = self._run_with_socket(sock, duration=3.0)
        self.assertEqual(found, [])

    def test_reuseport_unsupported_is_swallowed(self):
        # The source wraps the SO_REUSEPORT setsockopt in except
        # (AttributeError, OSError). On a host where SO_REUSEPORT EXISTS (Linux
        # CI) we make that one setsockopt raise OSError; on a host where it's
        # ABSENT (some Windows builds) the attribute access itself raises
        # AttributeError before our fake is reached — either way discovery must
        # continue and still return the find.
        reuseport = getattr(socket, "SO_REUSEPORT", None)
        pkt = b"NOTIFY bambulab\r\nLocation: 192.168.1.13\r\n"
        sock = _FakeSocket([(pkt, ("192.168.1.13", 2021))])
        real_setsockopt = sock.setsockopt

        def _maybe_raise(level, optname, *rest):
            if reuseport is not None and optname == reuseport:
                raise OSError("REUSEPORT unsupported")
            return real_setsockopt(level, optname, *rest)

        sock.setsockopt = _maybe_raise
        found = self._run_with_socket(sock, duration=3.0)
        self.assertEqual([p["ip"] for p in found], ["192.168.1.13"])

    def test_socket_oserror_returns_empty(self):
        # bind() / socket() raising OSError (no multicast) → [] and the wizard
        # falls back to voice prompting.
        with mock.patch.object(self.mod.socket, "socket",
                               side_effect=OSError("no multicast")):
            self.assertEqual(self.mod.discover_printers(duration=2.0), [])

    def test_close_failure_swallowed(self):
        pkt = b"NOTIFY bambulab\r\nLocation: 192.168.1.14\r\n"
        sock = _FakeSocket([(pkt, ("192.168.1.14", 2021))])
        sock.close = mock.MagicMock(side_effect=OSError("already closed"))
        # Must still return the find despite close() blowing up in finally.
        found = self._run_with_socket(sock, duration=3.0)
        self.assertEqual([p["ip"] for p in found], ["192.168.1.14"])


# ─────────────────────────────────────────────────────────────────────────
# _voice_to_ip / _voice_to_digits — remaining branches.
# ─────────────────────────────────────────────────────────────────────────
class VoiceParseEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_voice_to_ip_point_separator(self):
        # "point" is accepted as a separator word alongside "dot".
        out = self.mod._voice_to_ip(
            "one nine two point one six eight point one point five zero")
        self.assertEqual(out, "192.168.1.50")

    def test_voice_to_ip_too_few_octets(self):
        # Only three dot-pieces → not a valid IPv4 → "".
        self.assertEqual(self.mod._voice_to_ip("one dot two dot three"), "")

    def test_voice_to_ip_chunk_without_digits_skipped(self):
        # A 'dot'-piece that has no digit tokens is skipped, leaving 4 good
        # octets from the rest.
        out = self.mod._voice_to_ip(
            "192 dot 168 dot uh 1 dot 42")
        self.assertEqual(out, "192.168.1.42")

    def test_voice_to_ip_empty_string(self):
        self.assertEqual(self.mod._voice_to_ip(""), "")


# ─────────────────────────────────────────────────────────────────────────
# _say / _listen — the TTS/STT bridge.
# ─────────────────────────────────────────────────────────────────────────
class SpeechBridgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_say_routes_to_bc_speak(self):
        bc = _FakeBC()
        bc._speak = mock.MagicMock()
        _inject_module(self, "bobert_companion", bc)
        self.mod._say("hello sir")
        bc._speak.assert_called_once_with("hello sir")

    def test_say_falls_back_to_console_on_error(self):
        bc = _FakeBC()
        bc._speak = mock.MagicMock(side_effect=RuntimeError("tts down"))
        _inject_module(self, "bobert_companion", bc)
        # Should not raise — it prints to console instead.
        self.mod._say("fallback please")

    def test_listen_returns_transcript(self):
        bc = _FakeBC()
        bc.record_speech = mock.MagicMock(return_value=[1, 2, 3])
        bc.transcribe = mock.MagicMock(return_value=("  yes sir  ", 0.9))
        _inject_module(self, "bobert_companion", bc)
        self.assertEqual(self.mod._listen(), "yes sir")

    def test_listen_empty_audio_returns_blank(self):
        bc = _FakeBC()
        bc.record_speech = mock.MagicMock(return_value=[])
        bc.transcribe = mock.MagicMock()
        _inject_module(self, "bobert_companion", bc)
        self.assertEqual(self.mod._listen(), "")
        bc.transcribe.assert_not_called()

    def test_listen_none_audio_returns_blank(self):
        bc = _FakeBC()
        bc.record_speech = mock.MagicMock(return_value=None)
        _inject_module(self, "bobert_companion", bc)
        self.assertEqual(self.mod._listen(), "")

    def test_listen_exception_returns_blank(self):
        bc = _FakeBC()
        bc.record_speech = mock.MagicMock(side_effect=RuntimeError("mic gone"))
        _inject_module(self, "bobert_companion", bc)
        self.assertEqual(self.mod._listen(), "")


# ─────────────────────────────────────────────────────────────────────────
# _wizard_pick_printer — the remaining miss / fallthrough branches.
# ─────────────────────────────────────────────────────────────────────────
class WizardPickEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_multiple_index_out_of_range_then_no_name_match(self):
        printers = [{"ip": "192.168.1.1", "name": "A"},
                    {"ip": "192.168.1.2", "name": "B"}]
        # "nine" → index 8, out of range; no model/name substring → None.
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="nine"):
            self.assertIsNone(self.mod._wizard_pick_printer(printers))

    def test_multiple_unparseable_reply_returns_none(self):
        printers = [{"ip": "192.168.1.1", "name": "Alpha"},
                    {"ip": "192.168.1.2", "name": "Beta"}]
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="hmm dunno"):
            self.assertIsNone(self.mod._wizard_pick_printer(printers))

    def test_multiple_match_by_model_field(self):
        # Name absent, but the spoken word matches the 'model' field. The reply
        # has no digit tokens, so the index-parse path yields nothing and the
        # model-substring fallback decides. (Model strings here are purely
        # alphabetic so _voice_to_digits can't turn them into an index.)
        printers = [{"ip": "192.168.1.1", "model": "prusa"},
                    {"ip": "192.168.1.2", "model": "bambu"}]
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               return_value="the bambu please"):
            chosen = self.mod._wizard_pick_printer(printers)
        self.assertEqual(chosen["ip"], "192.168.1.2")


# ─────────────────────────────────────────────────────────────────────────
# _wizard_prompt_ip — the 3-attempt re-prompt loop.
# ─────────────────────────────────────────────────────────────────────────
class WizardPromptIpTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_first_try_confirmed(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["192.168.1.42", "yes"]):
            self.assertEqual(self.mod._wizard_prompt_ip(), "192.168.1.42")

    def test_heard_ip_but_user_says_no_then_succeeds(self):
        # attempt 0: parse ok, user declines; attempt 1: parse ok, confirms.
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["192.168.1.10", "no",
                                            "192.168.1.20", "yes"]):
            self.assertEqual(self.mod._wizard_prompt_ip(), "192.168.1.20")

    def test_empty_then_unparseable_then_give_up(self):
        # attempt 0: blank; attempt 1: no octets; attempt 2: blank → "".
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["", "no numbers", ""]):
            self.assertEqual(self.mod._wizard_prompt_ip(), "")

    def test_all_three_unconfirmed_returns_blank(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["192.168.1.1", "no",
                                            "192.168.1.2", "no",
                                            "192.168.1.3", "no"]):
            self.assertEqual(self.mod._wizard_prompt_ip(), "")


# ─────────────────────────────────────────────────────────────────────────
# _wizard_prompt_access_code — the 3-attempt re-prompt loop.
# ─────────────────────────────────────────────────────────────────────────
class WizardPromptAccessCodeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_eight_digits_confirmed(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["one two three four five six seven eight",
                                            "yes"]):
            self.assertEqual(self.mod._wizard_prompt_access_code(), "12345678")

    def test_wrong_length_then_correct(self):
        # attempt 0: 4 digits (too short) → re-prompt; attempt 1: 8 → confirm.
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["1234", "12345678", "yes"]):
            self.assertEqual(self.mod._wizard_prompt_access_code(), "12345678")

    def test_heard_eight_but_declined_then_succeeds(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["11112222", "no",
                                            "12345678", "yes"]):
            self.assertEqual(self.mod._wizard_prompt_access_code(), "12345678")

    def test_blank_then_give_up(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["", "", ""]):
            self.assertEqual(self.mod._wizard_prompt_access_code(), "")

    def test_all_three_declined_returns_blank(self):
        # 8 digits every time but declined on all 3 — incl. the final attempt
        # where `attempt < 2` is False and the loop simply ends.
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["12345678", "no",
                                            "12345678", "no",
                                            "12345678", "no"]):
            self.assertEqual(self.mod._wizard_prompt_access_code(), "")

    def test_wrong_length_on_final_attempt(self):
        # Wrong length on the 3rd (last) attempt — exercises the else-branch
        # falling through without a re-prompt.
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["123", "1234", "12345"]):
            self.assertEqual(self.mod._wizard_prompt_access_code(), "")


# ─────────────────────────────────────────────────────────────────────────
# _wizard_prompt_serial — optional, skip-able, best-effort capture.
# ─────────────────────────────────────────────────────────────────────────
class WizardPromptSerialTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_blank_reply_returns_empty(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value=""):
            self.assertEqual(self.mod._wizard_prompt_serial(), "")

    def test_skip_word_returns_empty(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="skip it sir"):
            self.assertEqual(self.mod._wizard_prompt_serial(), "")

    def test_captured_and_confirmed(self):
        # The serial path does NOT word→digit map — it just strips non-alnum
        # and upper-cases the raw transcript. A typical Whisper read of a
        # printed serial comes through with the alphanumerics intact.
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["zero one P 00 A 99", "yes"]):
            out = self.mod._wizard_prompt_serial()
        # "zero one P 00 A 99" → strip spaces/non-alnum, upper → ZEROONEP00A99.
        self.assertEqual(out, "ZEROONEP00A99")

    def test_captured_but_declined_returns_blank(self):
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen",
                               side_effect=["serial zero one two three four five",
                                            "no"]):
            self.assertEqual(self.mod._wizard_prompt_serial(), "")

    def test_too_short_candidate_returns_blank(self):
        # Fewer than 6 alphanumerics → not confirmed, returns "".
        with mock.patch.object(self.mod, "_say"), \
             mock.patch.object(self.mod, "_listen", return_value="ab12"):
            self.assertEqual(self.mod._wizard_prompt_serial(), "")


# ─────────────────────────────────────────────────────────────────────────
# _persist_credentials — live-attr patch + source rewrite, FAKE bc + TEMP file.
# ─────────────────────────────────────────────────────────────────────────
_SAMPLE_CONFIG = (
    "# bobert_companion config\n"
    'BAMBU_PRINTER_IP  = "192.168.1.1"  # printer on LAN\n'
    'BAMBU_ACCESS_CODE = "00000000"\n'
    "BAMBU_SERIAL      = 'OLDSERIAL01'\n"
    "OTHER = 1\n"
)


class PersistCredentialsTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")
        self.bc = _FakeBC()
        _inject_module(self, "bobert_companion", self.bc)
        # Redirect the source-rewrite target at a throwaway temp file so the
        # real bobert_companion.py is NEVER touched.
        self.tmpdir = tempfile.mkdtemp(prefix="bambu_persist_")
        self.cfg = os.path.join(self.tmpdir, "bobert_companion.py")
        with open(self.cfg, "w", encoding="utf-8") as f:
            f.write(_SAMPLE_CONFIG)
        mock.patch.object(self.mod, "_CONFIG_PATH", self.cfg).start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for fn in os.listdir(self.tmpdir):
            try:
                os.unlink(os.path.join(self.tmpdir, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def test_happy_path_patches_attrs_and_rewrites_source(self):
        ok = self.mod._persist_credentials("192.168.1.77", "12345678",
                                           "01P00A0001")
        self.assertTrue(ok)
        # Live attrs updated.
        self.assertEqual(self.bc.BAMBU_PRINTER_IP, "192.168.1.77")
        self.assertEqual(self.bc.BAMBU_ACCESS_CODE, "12345678")
        self.assertEqual(self.bc.BAMBU_SERIAL, "01P00A0001")
        # Source rewritten — all three, double-quoted, comment preserved.
        with open(self.cfg, encoding="utf-8") as f:
            body = f.read()
        self.assertIn('BAMBU_PRINTER_IP  = "192.168.1.77"  # printer on LAN', body)
        self.assertIn('BAMBU_ACCESS_CODE = "12345678"', body)
        self.assertIn('BAMBU_SERIAL      = "01P00A0001"', body)
        self.assertNotIn("OLDSERIAL01", body)

    def test_live_attr_patch_failure_returns_false(self):
        # _bc() raising → can't patch live attrs → False, no file write.
        with mock.patch.object(self.mod, "_bc",
                               side_effect=RuntimeError("import boom")):
            self.assertFalse(
                self.mod._persist_credentials("192.168.1.5", "12345678", ""))

    def test_config_read_failure_returns_false(self):
        # Live attrs patch fine, but the source read explodes → False.
        with mock.patch("builtins.open", side_effect=OSError("no read")):
            self.assertFalse(
                self.mod._persist_credentials("192.168.1.6", "12345678", ""))

    def test_unmatched_vars_keep_session_live_only(self):
        # A config file with NONE of the BAMBU_* lines → no rewrite happens
        # but live attrs are still patched → returns True (session-only).
        with open(self.cfg, "w", encoding="utf-8") as f:
            f.write("NOTHING_HERE = 1\n")
        ok = self.mod._persist_credentials("192.168.1.8", "12345678", "S1")
        self.assertTrue(ok)
        self.assertEqual(self.bc.BAMBU_PRINTER_IP, "192.168.1.8")
        with open(self.cfg, encoding="utf-8") as f:
            self.assertNotIn("192.168.1.8", f.read())

    def test_backslash_and_quote_in_value_escaped(self):
        # An access code with a backslash/quote stays syntactically valid.
        ok = self.mod._persist_credentials('192.168.1.9', r'12\3"4', "S")
        self.assertTrue(ok)
        with open(self.cfg, encoding="utf-8") as f:
            body = f.read()
        self.assertIn(r'\\', body)
        self.assertIn(r'\"', body)

    def test_source_write_failure_returns_false(self):
        # Read succeeds and lines match, but the tempfile write/replace fails.
        with mock.patch.object(self.mod.os, "replace",
                               side_effect=OSError("disk full")):
            self.assertFalse(
                self.mod._persist_credentials("192.168.1.11", "12345678", "S"))


# ─────────────────────────────────────────────────────────────────────────
# _restart_monitor — hot-restart of the bambu_monitor poller.
# ─────────────────────────────────────────────────────────────────────────
class RestartMonitorTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")

    def test_monitor_not_loaded_returns_false(self):
        # importlib.import_module raising (module absent) → False.
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("not loaded")):
            self.assertFalse(self.mod._restart_monitor())

    def test_start_monitor_true(self):
        fake = types.SimpleNamespace(start_monitor=lambda: True)
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=fake):
            self.assertTrue(self.mod._restart_monitor())

    def test_start_monitor_false(self):
        fake = types.SimpleNamespace(start_monitor=lambda: 0)
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=fake):
            self.assertFalse(self.mod._restart_monitor())

    def test_start_monitor_raises_returns_false(self):
        def _boom():
            raise RuntimeError("monitor exploded")
        fake = types.SimpleNamespace(start_monitor=_boom)
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=fake):
            self.assertFalse(self.mod._restart_monitor())


# ─────────────────────────────────────────────────────────────────────────
# setup_printer — the full discovery / voice flow (inline path already covered).
# ─────────────────────────────────────────────────────────────────────────
class SetupPrinterVoiceFlowTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("bambu_setup")
        # Patch the speech bridge + side-effecting helpers used by every flow.
        self._say = mock.patch.object(self.mod, "_say").start()
        self.addCleanup(mock.patch.stopall)

    def test_discovery_then_full_success_polling(self):
        chosen = {"ip": "192.168.1.30", "serial": "01P00A0030", "name": "H2D"}
        with mock.patch.object(self.mod, "discover_printers",
                               return_value=[chosen]), \
             mock.patch.object(self.mod, "_wizard_pick_printer",
                               return_value=chosen), \
             mock.patch.object(self.mod, "_wizard_prompt_access_code",
                               return_value="12345678"), \
             mock.patch.object(self.mod, "_persist_credentials",
                               return_value=True) as persist, \
             mock.patch.object(self.mod, "_restart_monitor", return_value=True):
            out = self.actions["setup_printer"]("")
        # Serial came from discovery → prompt_serial NOT called; persist got it.
        persist.assert_called_once_with("192.168.1.30", "12345678", "01P00A0030")
        self.assertIn("192.168.1.30", out)
        self.assertIn("polling", out.lower())

    def test_discovery_pick_cancelled(self):
        with mock.patch.object(self.mod, "discover_printers",
                               return_value=[{"ip": "192.168.1.31"}]), \
             mock.patch.object(self.mod, "_wizard_pick_printer",
                               return_value=None):
            out = self.actions["setup_printer"]("")
        self.assertIn("cancelled", out.lower())

    def test_no_discovery_prompts_ip_then_serial_then_idle(self):
        # No broadcast → prompt for IP; no serial from discovery → prompt it;
        # persist ok but monitor idle.
        with mock.patch.object(self.mod, "discover_printers", return_value=[]), \
             mock.patch.object(self.mod, "_wizard_prompt_ip",
                               return_value="192.168.1.40"), \
             mock.patch.object(self.mod, "_wizard_prompt_access_code",
                               return_value="12345678"), \
             mock.patch.object(self.mod, "_wizard_prompt_serial",
                               return_value="01P00A0040") as ser, \
             mock.patch.object(self.mod, "_persist_credentials",
                               return_value=True), \
             mock.patch.object(self.mod, "_restart_monitor", return_value=False):
            out = self.actions["setup_printer"]("")
        ser.assert_called_once()
        self.assertIn("192.168.1.40", out)
        self.assertIn("idle", out.lower())

    def test_no_discovery_no_ip_aborts(self):
        with mock.patch.object(self.mod, "discover_printers", return_value=[]), \
             mock.patch.object(self.mod, "_wizard_prompt_ip", return_value=""):
            out = self.actions["setup_printer"]("")
        self.assertIn("no ip captured", out.lower())

    def test_no_access_code_aborts(self):
        with mock.patch.object(self.mod, "discover_printers", return_value=[]), \
             mock.patch.object(self.mod, "_wizard_prompt_ip",
                               return_value="192.168.1.41"), \
             mock.patch.object(self.mod, "_wizard_prompt_access_code",
                               return_value=""):
            out = self.actions["setup_printer"]("")
        self.assertIn("no access code captured", out.lower())

    def test_persist_failure_in_voice_flow(self):
        chosen = {"ip": "192.168.1.42", "serial": "01P00A0042"}
        with mock.patch.object(self.mod, "discover_printers",
                               return_value=[chosen]), \
             mock.patch.object(self.mod, "_wizard_pick_printer",
                               return_value=chosen), \
             mock.patch.object(self.mod, "_wizard_prompt_access_code",
                               return_value="12345678"), \
             mock.patch.object(self.mod, "_persist_credentials",
                               return_value=False):
            out = self.actions["setup_printer"]("")
        self.assertIn("persist error", out.lower())

    def test_lock_released_after_voice_flow(self):
        # After a full run the wizard lock must be free for the next call.
        with mock.patch.object(self.mod, "discover_printers", return_value=[]), \
             mock.patch.object(self.mod, "_wizard_prompt_ip", return_value=""):
            self.actions["setup_printer"]("")
        self.assertTrue(self.mod._wizard_lock.acquire(blocking=False))
        self.mod._wizard_lock.release()


if __name__ == "__main__":
    unittest.main()
