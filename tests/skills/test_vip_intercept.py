"""Logic tests for skills/vip_intercept.py — the Wayne draft-and-send flow.

The regression these exist for (2026-08-20 bug-hunt, H-2):

  ``send_vip_reply`` typed the queued reply into Teams with pyautogui,
  pressed Ctrl+Enter, then returned **"Sent the reply to Wayne, sir."** and
  destroyed the draft with ``_write_pending(None)`` — on nothing but "the
  keystroke calls did not raise". Its own helper docstring conceded the
  keystrokes "DOES NOT verify Teams accepted them" and then promised "a
  failed send surfaces as a spoken error and the draft stays queued", a
  contract the code could not keep: it held only for failures the helper
  could SEE. If focus moved, a ringing-call toast (which has no compose
  box) won the window race, or Teams swallowed the keys, the message never
  went out, the result string is SPOKEN VERBATIM to the owner
  (bobert_companion's verbatim speak set), and the only copy of his reply
  to his boss was gone.

  The codebase's honest-failure contract says an action must never claim it
  did something it did not. So:

    • a dispatch that could not be positively confirmed must NOT say "Sent"
      and must NOT destroy the draft, and
    • every not-confirmed line must carry a live FAILURE_MARKER so it routes
      to the failure follow-up instead of reading like a success.

Nothing here touches the live ``data/wayne_pending_reply.json`` — every test
repoints ``_PENDING_FILE`` at a temp dir.

The background monitor thread never starts (harness neuters threads).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from core.failure_markers import FAILURE_MARKERS
from tests._skill_harness import load_skill_isolated

# skills/vip_intercept.py is a GITIGNORED personal skill (it carries the
# owner's VIP names) — it exists on the owner's box but not in the public
# repo, so on GitHub CI there is nothing to load. Skip cleanly there.
_SKILL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "skills", "vip_intercept.py")

_SKIP_REASON = "vip_intercept is a gitignored personal skill (absent on CI)"


def _is_failure_line(text: str) -> bool:
    """Mirror of the classification bobert_companion / core.dispatcher apply
    to a free-text action result (lower-cased substring match)."""
    low = (text or "").lower()
    return any(m in low for m in FAILURE_MARKERS)


class _FakeWindow:
    """Stand-in for a pygetwindow Win32Window."""

    def __init__(self, title, hwnd=1, minimized=False, activate_raises=False):
        self.title = title
        self._hWnd = hwnd
        self.isMinimized = minimized
        self._activate_raises = activate_raises
        self.activated = False
        self.restored = False

    def restore(self):
        self.restored = True

    def activate(self):
        if self._activate_raises:
            raise RuntimeError("SetForegroundWindow refused")
        self.activated = True


@unittest.skipUnless(os.path.exists(_SKILL_PATH), _SKIP_REASON)
class _VipInterceptBase(unittest.TestCase):
    """Loads the skill and redirects the pending-draft file to a temp dir.

    NEVER let a test write under C:\\JARVIS\\data — that is live runtime
    state for a running assistant."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("vip_intercept")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pending_file = os.path.join(self._tmp.name, "wayne_pending_reply.json")
        patcher = mock.patch.object(self.mod, "_PENDING_FILE", self.pending_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ── helpers ──────────────────────────────────────────────────────────
    def queue(self, body="Sorry Wayne — in a 1:1, call you at 3.", **extra):
        rec = {
            "ts": self.mod.time.time(),
            "to": "Wayne",
            "subject": "Teams reply",
            "body": body,
            "kind": "dm",
            "original": "got 5 min?",
            "state": self.mod.DRAFT_STATE_QUEUED,
            "attempts": 0,
        }
        rec.update(extra)
        self.assertTrue(self.mod._write_pending(rec))
        return rec

    def on_disk(self):
        if not os.path.exists(self.pending_file):
            return None
        with open(self.pending_file, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("active")


# ─────────────────────────────────────────────────────────────────────────
# The headline regression: an unverified dispatch is not a send.
# ─────────────────────────────────────────────────────────────────────────
class UnverifiedSendTests(_VipInterceptBase):

    def _send(self, dispatch, verdict):
        with mock.patch.object(self.mod, "_dispatch_via_keyboard",
                               return_value=dispatch) as disp, \
             mock.patch.object(self.mod, "_verify_send_landed",
                               return_value=verdict) as ver:
            out = self.actions["send_vip_reply"]("")
        return out, disp, ver

    def test_unverified_dispatch_does_not_claim_a_send(self):
        # Keystrokes went out; nothing corroborated them. The old code said
        # "Sent the reply to Wayne, sir." here.
        self.queue()
        out, _, _ = self._send(self.mod.SEND_DISPATCHED, self.mod.SEND_UNVERIFIED)
        self.assertNotIn("Sent the reply", out)
        self.assertIn("typed", out.lower())
        self.assertTrue(_is_failure_line(out),
                        f"unverified send must carry a FAILURE_MARKER: {out!r}")

    def test_unverified_dispatch_does_not_destroy_the_draft(self):
        rec = self.queue()
        self._send(self.mod.SEND_DISPATCHED, self.mod.SEND_UNVERIFIED)
        kept = self.on_disk()
        self.assertIsNotNone(kept, "an unverified send must NOT clear the draft")
        self.assertEqual(kept["body"], rec["body"])
        self.assertEqual(kept["state"], self.mod.DRAFT_STATE_UNVERIFIED)
        self.assertEqual(kept["attempts"], 1)

    def test_vision_says_not_sent_keeps_draft_and_says_so(self):
        rec = self.queue()
        out, _, _ = self._send(self.mod.SEND_DISPATCHED, self.mod.SEND_NOT_VISIBLE)
        self.assertNotIn("Sent the reply", out)
        self.assertTrue(_is_failure_line(out),
                        f"a negative verdict must carry a FAILURE_MARKER: {out!r}")
        kept = self.on_disk()
        self.assertIsNotNone(kept)
        self.assertEqual(kept["body"], rec["body"])
        self.assertEqual(kept["state"], self.mod.DRAFT_STATE_UNVERIFIED)

    def test_focus_failure_keeps_draft_and_never_verifies(self):
        rec = self.queue()
        out, _, ver = self._send(self.mod.SEND_FAILED, self.mod.SEND_CONFIRMED)
        # Nothing was typed, so there is nothing to look for.
        ver.assert_not_called()
        self.assertNotIn("Sent the reply", out)
        self.assertTrue(_is_failure_line(out))
        kept = self.on_disk()
        self.assertIsNotNone(kept)
        self.assertEqual(kept["body"], rec["body"])
        self.assertEqual(kept["state"], self.mod.DRAFT_STATE_QUEUED)

    def test_confirmed_send_clears_the_draft_and_claims_it(self):
        self.queue()
        out, _, _ = self._send(self.mod.SEND_DISPATCHED, self.mod.SEND_CONFIRMED)
        self.assertEqual(out, "Sent the reply to Wayne, sir.")
        self.assertIsNone(self.on_disk(),
                          "a CONFIRMED send is the only thing that may clear it")
        self.assertFalse(_is_failure_line(out),
                         "the genuine success line must not read as a failure")

    def test_confirmed_send_reports_a_failed_clear(self):
        # Sent for real but the queue file survived — saying only "Sent"
        # would set up a silent duplicate to the owner's boss.
        self.queue()
        with mock.patch.object(self.mod, "_dispatch_via_keyboard",
                               return_value=self.mod.SEND_DISPATCHED), \
             mock.patch.object(self.mod, "_verify_send_landed",
                               return_value=self.mod.SEND_CONFIRMED), \
             mock.patch.object(self.mod, "_write_pending", return_value=False):
            out = self.actions["send_vip_reply"]("")
        self.assertIn("Sent the reply", out)
        self.assertTrue(_is_failure_line(out),
                        f"an uncleared queue must surface as a failure: {out!r}")

    def test_no_draft_is_reported_plainly(self):
        self.assertEqual(self.actions["send_vip_reply"](""),
                         "No Wayne reply queued, sir.")

    def test_every_not_confirmed_outcome_carries_a_marker(self):
        combos = [
            (self.mod.SEND_FAILED, self.mod.SEND_UNVERIFIED),
            (self.mod.SEND_DISPATCHED, self.mod.SEND_UNVERIFIED),
            (self.mod.SEND_DISPATCHED, self.mod.SEND_NOT_VISIBLE),
        ]
        for dispatch, verdict in combos:
            with self.subTest(dispatch=dispatch, verdict=verdict):
                self.queue()
                out, _, _ = self._send(dispatch, verdict)
                self.assertTrue(
                    _is_failure_line(out),
                    f"{dispatch}/{verdict} must route to the failure "
                    f"follow-up, got {out!r}")
                self.assertNotIn("Sent the reply", out)


# ─────────────────────────────────────────────────────────────────────────
# The keystroke helper itself must not report a send.
# ─────────────────────────────────────────────────────────────────────────
class KeystrokeDispatchTests(_VipInterceptBase):

    def _fake_pyautogui(self, typewrite_raises=False):
        fake = mock.MagicMock(name="pyautogui")
        if typewrite_raises:
            fake.typewrite.side_effect = RuntimeError("no display")
        return fake

    def test_old_boolean_helper_is_gone(self):
        # A bool return let `if _send_via_keyboard(body):` read "keys were
        # dispatched" as "message sent". The rename makes any stale caller
        # fail loudly instead of treating a failure status as truthy.
        self.assertFalse(hasattr(self.mod, "_send_via_keyboard"))
        self.assertTrue(callable(self.mod._dispatch_via_keyboard))

    def test_dispatch_reports_dispatched_not_sent(self):
        fake = self._fake_pyautogui()
        with mock.patch.object(self.mod, "_focus_teams_for_send", return_value=True), \
             mock.patch.dict(sys.modules, {"pyautogui": fake}):
            out = self.mod._dispatch_via_keyboard("hello")
        self.assertEqual(out, self.mod.SEND_DISPATCHED)
        self.assertNotEqual(out, self.mod.SEND_CONFIRMED)

    def test_dispatch_fails_when_no_teams_window(self):
        fake = self._fake_pyautogui()
        with mock.patch.object(self.mod, "_focus_teams_for_send", return_value=False), \
             mock.patch.dict(sys.modules, {"pyautogui": fake}):
            out = self.mod._dispatch_via_keyboard("hello")
        self.assertEqual(out, self.mod.SEND_FAILED)
        fake.typewrite.assert_not_called()

    def test_dispatch_fails_when_typing_raises(self):
        fake = self._fake_pyautogui(typewrite_raises=True)
        with mock.patch.object(self.mod, "_focus_teams_for_send", return_value=True), \
             mock.patch.dict(sys.modules, {"pyautogui": fake}):
            out = self.mod._dispatch_via_keyboard("hello")
        self.assertEqual(out, self.mod.SEND_FAILED)

    def test_dispatch_fails_when_send_hotkey_raises(self):
        fake = self._fake_pyautogui()
        self.mod.skill_utils["hotkey"].side_effect = RuntimeError("blocked")
        with mock.patch.object(self.mod, "_focus_teams_for_send", return_value=True), \
             mock.patch.dict(sys.modules, {"pyautogui": fake}):
            out = self.mod._dispatch_via_keyboard("hello")
        self.assertEqual(out, self.mod.SEND_FAILED)

    def test_docstring_no_longer_promises_a_contract_it_cannot_keep(self):
        # The pre-fix docstring said "Returns True iff the keystrokes were
        # dispatched" and then "a failed send surfaces as a spoken error and
        # the draft stays queued for a manual re-attempt" — true only for the
        # failures the helper could SEE. Stale comments of exactly this shape
        # are the #1 rot vector in this repo.
        src = open(_SKILL_PATH, "r", encoding="utf-8").read()
        self.assertNotIn("Returns True iff the keystrokes were dispatched", src)
        self.assertNotIn("the draft stays queued for a manual re-attempt", src)


# ─────────────────────────────────────────────────────────────────────────
# Window selection — the concrete route to a false "sent".
# ─────────────────────────────────────────────────────────────────────────
class SendTargetTests(_VipInterceptBase):

    def _candidates(self, windows, proc=""):
        with mock.patch.object(self.mod, "_window_proc_name", return_value=proc):
            return self.mod._teams_send_candidates(windows)

    def test_ring_toast_is_never_a_send_target(self):
        # A ringing-call window has no compose box: typing there dispatches
        # keystrokes that go nowhere while every call still "succeeds". The
        # designed flow drafts a reply WHILE Wayne is ringing, so this is the
        # common case, not an exotic one.
        ring = _FakeWindow("Wayne A. Example is calling you | Microsoft Teams")
        chat = _FakeWindow("Chat | Wayne A. Example | Microsoft Teams", hwnd=2)
        got = self._candidates([ring, chat], proc="ms-teams.exe")
        self.assertEqual(got, [chat])

    def test_every_ring_hint_is_excluded(self):
        for hint in self.mod.TEAMS_RING_TITLE_HINTS:
            with self.subTest(hint=hint):
                w = _FakeWindow(f"Wayne {hint} | Microsoft Teams")
                self.assertEqual(self._candidates([w], proc="ms-teams.exe"), [])

    def test_bare_teams_title_is_not_enough(self):
        # skills/teams_screener's TEAMS_TITLE_HINTS accepts a bare "teams";
        # this path must not inherit that looseness.
        w = _FakeWindow("Teams standup notes - Notepad")
        self.assertEqual(self._candidates([w], proc=""), [])

    def test_browser_window_titled_teams_is_vetoed_by_process(self):
        w = _FakeWindow("Chat | Wayne | Microsoft Teams - Google Chrome")
        self.assertEqual(self._candidates([w], proc="chrome.exe"), [])

    def test_unattributable_window_falls_back_to_the_title_rule(self):
        # '' from _window_proc_name means "could not resolve", not "not
        # Teams" — it must not silently disqualify a real Teams window.
        w = _FakeWindow("Chat | Wayne | Microsoft Teams")
        self.assertEqual(self._candidates([w], proc=""), [w])

    def test_refused_foreground_change_is_not_treated_as_focused(self):
        # pygetwindow's activate() can return without raising while Windows
        # refuses the foreground change (and teams_screener's fallback path
        # returns True from a bare minimize/restore). Typing after that lands
        # the owner's reply in whatever is actually in front.
        w = _FakeWindow("Chat | Wayne | Microsoft Teams")
        fake_gw = mock.MagicMock(name="pygetwindow")
        fake_gw.getAllWindows.return_value = [w]
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw}), \
             mock.patch.object(self.mod, "_window_proc_name", return_value="ms-teams.exe"), \
             mock.patch.object(self.mod, "_wait_for_foreground", return_value=False):
            self.assertFalse(self.mod._focus_teams_for_send())

    def test_confirmed_foreground_change_is_accepted(self):
        w = _FakeWindow("Chat | Wayne | Microsoft Teams", minimized=True)
        fake_gw = mock.MagicMock(name="pygetwindow")
        fake_gw.getAllWindows.return_value = [w]
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw}), \
             mock.patch.object(self.mod, "_window_proc_name", return_value="ms-teams.exe"), \
             mock.patch.object(self.mod, "_wait_for_foreground", return_value=True):
            self.assertTrue(self.mod._focus_teams_for_send())
        self.assertTrue(w.restored)
        self.assertTrue(w.activated)

    def test_unverifiable_foreground_is_allowed_through(self):
        # None = "this build can't tell". We still type, but the send is then
        # gated by the vision check, so an unknown focus can only ever produce
        # an honest "couldn't confirm" — never a false "Sent".
        w = _FakeWindow("Chat | Wayne | Microsoft Teams")
        fake_gw = mock.MagicMock(name="pygetwindow")
        fake_gw.getAllWindows.return_value = [w]
        with mock.patch.dict(sys.modules, {"pygetwindow": fake_gw}), \
             mock.patch.object(self.mod, "_window_proc_name", return_value="ms-teams.exe"), \
             mock.patch.object(self.mod, "_wait_for_foreground", return_value=None):
            self.assertTrue(self.mod._focus_teams_for_send())

    def test_focus_does_not_delegate_to_the_screener(self):
        # skill_teams_screener is by definition loaded in production (it is
        # the module that calls handle_call), and its _focus_teams PREFERS a
        # ringing window. Delegating to it is how the reply got typed into a
        # call toast.
        screener = mock.MagicMock(name="skill_teams_screener")
        screener._focus_teams.return_value = True
        fake_gw = mock.MagicMock(name="pygetwindow")
        fake_gw.getAllWindows.return_value = []
        with mock.patch.dict(sys.modules, {"skill_teams_screener": screener,
                                           "pygetwindow": fake_gw}):
            self.assertFalse(self.mod._focus_teams_for_send())
        screener._focus_teams.assert_not_called()

    def test_no_pygetwindow_is_a_clean_failure(self):
        with mock.patch.dict(sys.modules, {"pygetwindow": None}):
            self.assertFalse(self.mod._focus_teams_for_send())

    def test_process_veto_is_not_dead_code(self):
        # LIVE API, deliberately not mocked. The first cut of this veto used
        # psapi.GetModuleBaseNameW (copied from ambient_listen) against a
        # handle opened with PROCESS_QUERY_LIMITED_INFORMATION — measured
        # 2026-08-20, that combination fails even for our OWN process, so the
        # veto would have silently never fired and every mocked test would
        # still have passed. Resolving the running interpreter proves the
        # win32 path actually returns a name.
        if sys.platform != "win32":
            self.skipTest("win32-only process lookup")
        name = self.mod._proc_name_for_pid(os.getpid())
        self.assertTrue(name, "process-name lookup returned '' for our own pid "
                              "— the browser veto would be dead code")
        self.assertTrue(name.endswith(".exe"), name)
        self.assertEqual(name, name.lower(), "callers compare against lower-case")

    def test_process_veto_survives_a_bogus_pid(self):
        self.assertEqual(self.mod._proc_name_for_pid(0), "")
        self.assertEqual(self.mod._proc_name_for_pid(0x7FFFFFFF), "")

    def test_window_without_a_handle_is_unattributable(self):
        self.assertEqual(self.mod._window_proc_name(object()), "")

    def test_wait_for_foreground_tri_state(self):
        w = _FakeWindow("Chat | Wayne | Microsoft Teams", hwnd=4242)
        with mock.patch.object(self.mod, "_foreground_hwnd", return_value=4242):
            self.assertIs(self.mod._wait_for_foreground(w, timeout_s=0.0), True)
        with mock.patch.object(self.mod, "_foreground_hwnd", return_value=99):
            self.assertIs(self.mod._wait_for_foreground(w, timeout_s=0.0), False)
        with mock.patch.object(self.mod, "_foreground_hwnd", return_value=None):
            self.assertIsNone(self.mod._wait_for_foreground(w, timeout_s=0.0))
        # No handle to compare against → unknown, never a confident True.
        self.assertIsNone(self.mod._wait_for_foreground(object(), timeout_s=0.0))


# ─────────────────────────────────────────────────────────────────────────
# The post-send corroboration.
# ─────────────────────────────────────────────────────────────────────────
class VerificationTests(_VipInterceptBase):

    def _verify(self, answer, screenshot=b"png", has_vision=True):
        bc = mock.MagicMock(name="bobert_companion")
        bc.take_screenshot.return_value = screenshot
        if has_vision:
            bc.ask_vision.return_value = answer
        else:
            del bc.ask_vision
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            return self.mod._verify_send_landed("hello Wayne")

    def test_yes_confirms(self):
        self.assertEqual(self._verify("YES"), self.mod.SEND_CONFIRMED)
        self.assertEqual(self._verify("  yes  "), self.mod.SEND_CONFIRMED)
        self.assertEqual(self._verify("[local-vision] YES"),
                         self.mod.SEND_CONFIRMED)

    def test_no_is_a_negative_verdict(self):
        self.assertEqual(self._verify("NO"), self.mod.SEND_NOT_VISIBLE)
        self.assertEqual(self._verify("[local-vision] no"),
                         self.mod.SEND_NOT_VISIBLE)

    def test_anything_unreadable_is_unverified_never_confirmed(self):
        for answer in ("(could not capture screen)",
                       "(screen vision is disabled — set SCREEN_VISION_ENABLED = True)",
                       "I'm not able to look at that right now.",
                       "", None, "maybe?"):
            with self.subTest(answer=answer):
                self.assertEqual(self._verify(answer), self.mod.SEND_UNVERIFIED)

    def test_no_screenshot_or_no_vision_is_unverified(self):
        self.assertEqual(self._verify("YES", screenshot=None),
                         self.mod.SEND_UNVERIFIED)
        self.assertEqual(self._verify("YES", has_vision=False),
                         self.mod.SEND_UNVERIFIED)

    def test_no_companion_is_unverified(self):
        with mock.patch.object(self.mod, "_import_companion", return_value=None):
            self.assertEqual(self.mod._verify_send_landed("x"),
                             self.mod.SEND_UNVERIFIED)

    def test_vision_failure_is_unverified_not_confirmed(self):
        bc = mock.MagicMock(name="bobert_companion")
        bc.take_screenshot.return_value = b"png"
        bc.ask_vision.side_effect = RuntimeError("rate limited")
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            self.assertEqual(self.mod._verify_send_landed("x"),
                             self.mod.SEND_UNVERIFIED)


# ─────────────────────────────────────────────────────────────────────────
# Draft store: the other "claim a side effect we never established" sites.
# ─────────────────────────────────────────────────────────────────────────
class DraftStoreHonestyTests(_VipInterceptBase):

    def test_write_pending_reports_disk_failure(self):
        with mock.patch.object(self.mod, "_PENDING_FILE",
                               os.path.join(self._tmp.name, "nope\x00", "x.json")):
            self.assertFalse(self.mod._write_pending({"body": "x"}))

    def test_queue_draft_reports_disk_failure(self):
        with mock.patch.object(self.mod, "_write_pending", return_value=False):
            self.assertFalse(self.mod._queue_draft("body", "dm"))

    def test_handle_call_does_not_announce_a_draft_it_could_not_queue(self):
        spoken = []
        with mock.patch.object(self.mod, "_maybe_engage", return_value=True), \
             mock.patch.object(self.mod, "_draft_reply", return_value="a reply"), \
             mock.patch.object(self.mod, "_queue_draft", return_value=False), \
             mock.patch.object(self.mod, "_speak_now",
                               side_effect=lambda t, **k: spoken.append(t)):
            self.assertFalse(self.mod.handle_call({"name": "Wayne"}))
        self.assertTrue(spoken)
        self.assertNotIn("I've drafted a reply for Wayne", " ".join(spoken))

    def test_handle_dm_does_not_announce_a_draft_it_could_not_queue(self):
        spoken = []
        with mock.patch.object(self.mod, "_maybe_engage", return_value=True), \
             mock.patch.object(self.mod, "_extract_wayne_message",
                               return_value="got 5 min?"), \
             mock.patch.object(self.mod, "_draft_reply", return_value="a reply"), \
             mock.patch.object(self.mod, "_queue_draft", return_value=False), \
             mock.patch.object(self.mod, "_speak_now",
                               side_effect=lambda t, **k: spoken.append(t)):
            self.assertFalse(self.mod.handle_dm({"name": "Wayne"}))
        self.assertNotIn("I've drafted a reply, sir", " ".join(spoken))

    def test_handle_dm_announces_a_draft_it_did_queue(self):
        spoken = []
        with mock.patch.object(self.mod, "_maybe_engage", return_value=True), \
             mock.patch.object(self.mod, "_extract_wayne_message",
                               return_value="got 5 min?"), \
             mock.patch.object(self.mod, "_draft_reply", return_value="a reply"), \
             mock.patch.object(self.mod, "_speak_now",
                               side_effect=lambda t, **k: spoken.append(t)):
            self.assertTrue(self.mod.handle_dm({"name": "Wayne"}))
        self.assertIn("I've drafted a reply, sir", " ".join(spoken))
        self.assertEqual(self.on_disk()["body"], "a reply")

    def test_scrap_does_not_claim_a_clear_the_disk_refused(self):
        self.queue()
        with mock.patch.object(self.mod, "_write_pending", return_value=False):
            out = self.actions["scrap_vip_reply"]("")
        self.assertNotIn("Scrapped", out)
        self.assertTrue(_is_failure_line(out))
        self.assertIsNotNone(self.on_disk())

    def test_scrap_clears_and_says_so(self):
        self.queue()
        self.assertEqual(self.actions["scrap_vip_reply"](""),
                         "Scrapped the Wayne reply, sir.")
        self.assertIsNone(self.on_disk())

    def test_ttl_never_reaps_an_unverified_send(self):
        # This record is the ONLY trace that a message may or may not have
        # reached Wayne. Letting the 5-minute TTL eat it silently would put
        # us right back to losing the draft.
        old = self.mod.time.time() - (self.mod.DRAFT_TTL_SECONDS + 60)
        self.queue(ts=old, state=self.mod.DRAFT_STATE_UNVERIFIED)
        rec = self.mod._get_active_draft()
        self.assertIsNotNone(rec)
        self.assertEqual(rec["state"], self.mod.DRAFT_STATE_UNVERIFIED)
        self.assertIsNotNone(self.on_disk())

    def test_ttl_still_reaps_an_ordinary_stale_draft(self):
        old = self.mod.time.time() - (self.mod.DRAFT_TTL_SECONDS + 60)
        self.queue(ts=old)
        self.assertIsNone(self.mod._get_active_draft())
        self.assertIsNone(self.on_disk())

    def test_status_surfaces_the_unverified_state(self):
        self.queue(state=self.mod.DRAFT_STATE_UNVERIFIED, attempts=1)
        out = self.actions["vip_intercept_status"]("")
        self.assertIn("NOT confirmed", out)
        # vip_intercept_status is an INFORMATIVE action — it reports, it does
        # not fail, so it must not trip the failure follow-up.
        self.assertFalse(_is_failure_line(out), out)

    def test_status_reads_an_ordinary_draft_normally(self):
        self.queue()
        out = self.actions["vip_intercept_status"]("")
        self.assertIn("Wayne reply queued", out)
        self.assertNotIn("NOT confirmed", out)

    def test_queue_draft_stamps_the_queued_state(self):
        self.assertTrue(self.mod._queue_draft("hi", "dm"))
        rec = self.on_disk()
        self.assertEqual(rec["state"], self.mod.DRAFT_STATE_QUEUED)
        self.assertEqual(rec["attempts"], 0)

    def test_gate_adapter_still_sees_the_shape_it_expects(self):
        # core/draft_preview_gate reads to/subject/body — the new lifecycle
        # fields must not disturb it.
        self.queue(body="hello Wayne")
        self.assertEqual(self.mod.get_pending_draft(),
                         {"to": "Wayne", "subject": "Teams reply",
                          "body": "hello Wayne"})


if __name__ == "__main__":
    unittest.main()
