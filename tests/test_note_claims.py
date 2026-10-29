"""Spoken note / remember / list / save claims (v3.0.7, "noted-claim").

THE LIVE TURN (2026-10-28 19:12, 3.0.5): a remark not meant for JARVIS was
heard inside the follow-up window and answered "Certainly, sir. I've noted
that for your presentation." No note, memory or list action ran - the claim
was false. core.claim_validator had no note / remember family, its reactive
check only reads replies with no [ACTION:] token, and "noted" is one of its
courtesy lead-ins, so nothing caught it.

Pinned here (core/claim_validator.find_note_claim / drop_note_claims /
note_claim_honest_line):
  * the live shape and its relatives are caught when no matching action ran
    ("I've noted that", "noted for your X", "I'll remember that", "I've added
    it to your list", "Saved, sir.") - with or without some OTHER action;
  * a true claim is untouched: note_for_claude / a list / reminder / save
    action that ran grounds its own kind;
  * the owner handing JARVIS something to keep grounds note / remember only;
  * acknowledgements, questions, offers, negations, observations, advice and
    dated reports are never claims;
  * the drop keeps the leading tag and other content, and leaves "" when
    only courtesy is left (the caller then speaks the honest line).

The monolith wiring (parse_and_run_actions, the stream flush, the setting)
is in tests/monolith/test_monolith_note_claims.py. Neutral fixtures only.
stdlib unittest; light tier.
"""
from __future__ import annotations

import unittest

from core import claim_validator as cv

# The live reply's shape with a neutral owner line (room speech, not a
# request to note anything).
_LIVE_REPLY = ("[intent:confirmation] Certainly, sir. I've noted that for "
               "your presentation.")
_ROOM_LINE = "I'll get to the next slide after lunch."


class LiveShapeTests(unittest.TestCase):
    def test_the_live_reply_is_a_note_claim(self):
        got = cv.find_note_claim(_LIVE_REPLY, user_text=_ROOM_LINE)
        self.assertIsNotNone(got)
        self.assertIn("noted that", got)

    def test_the_old_checks_never_saw_it(self):
        # Why a new check is needed: the reactive and completed-claim
        # checks pass the live reply.
        self.assertIsNone(cv.find_unverified_claim(_LIVE_REPLY,
                                                   user_text=_ROOM_LINE))
        self.assertIsNone(cv.find_completed_claim(_LIVE_REPLY,
                                                  user_text=_ROOM_LINE))

    def test_an_unrelated_action_does_not_ground_it(self):
        for ran in (["play_music"], ["see_screen"], ["get_time", "web_search"]):
            with self.subTest(ran=ran):
                self.assertIsNotNone(cv.find_note_claim(
                    _LIVE_REPLY, ran_actions=ran, user_text=_ROOM_LINE))

    def test_the_drop_leaves_nothing_but_the_honest_line(self):
        kept, bad = cv.drop_note_claims(_LIVE_REPLY, user_text=_ROOM_LINE)
        self.assertEqual(kept, "")
        self.assertEqual(len(bad), 1)
        self.assertEqual(
            cv.note_claim_honest_line(_LIVE_REPLY, user_text=_ROOM_LINE),
            "I haven't made a note of that, sir.")


class CaughtTests(unittest.TestCase):
    CASES = (
        ("I've noted that for your presentation.", "note"),
        ("Very good, sir, noted for your presentation.", "note"),
        ("Noted for later, sir.", "note"),
        ("I've made a note of it, sir.", "note"),
        ("I'll note that for posterity.", "note"),
        ("I'll jot that down, sir.", "note"),
        ("I'm noting that down now.", "note"),
        ("Consider it noted, sir.", "note"),
        ("It has been noted, sir.", "note"),
        ("I'll remember that, sir.", "remember"),
        ("I'll keep that in mind, sir.", "remember"),
        ("I won't forget, sir.", "remember"),
        ("I've committed it to memory.", "remember"),
        ("I've added it to your shopping list, sir.", "list"),
        ("Added to your reminders, sir.", "list"),
        ("I'll put that on your to-do list.", "list"),
        ("It's been added to your list.", "list"),
        ("Saved, sir.", "save"),
        ("I've saved it for you.", "save"),
        ("It has been saved to your desktop.", "save"),
    )

    def test_each_is_caught_with_nothing_run(self):
        for text, kind in self.CASES:
            with self.subTest(text=text):
                self.assertIsNotNone(
                    cv.find_note_claim(text, user_text=_ROOM_LINE))
                self.assertEqual(
                    cv.note_claim_honest_line(text, user_text=_ROOM_LINE),
                    cv.NOTE_CLAIM_HONEST_LINES[kind])

    def test_with_an_action_token_too(self):
        # The reply's own token is no exemption (the old reactive check
        # skipped every reply that carried one).
        text = "I've noted that, sir. [ACTION: play_music]"
        self.assertIsNotNone(cv.find_note_claim(
            text, ran_actions=["play_music"], user_text=_ROOM_LINE))


class TrueClaimTests(unittest.TestCase):
    def test_a_matching_action_grounds_its_kind(self):
        cases = (
            ("Noted for Claude, sir.", ["note_for_claude"]),
            ("I've noted that for your presentation.", ["note_for_claude"]),
            ("I've added it to your shopping list, sir.",
             ["add_shopping_item"]),
            ("Added to your reminders, sir.", ["add_reminder"]),
            ("I've put it in the queue.", ["queue_task"]),
            ("Saved, sir.", ["screenshot"]),
            ("I'll remember your face, sir.", ["enroll_face"]),
        )
        for text, ran in cases:
            with self.subTest(text=text):
                self.assertIsNone(cv.find_note_claim(
                    text, ran_actions=ran, user_text="x"))
                self.assertEqual(cv.drop_note_claims(
                    text, ran_actions=ran, user_text="x"), (text, []))

    def test_a_failed_note_action_is_not_passed_in(self):
        # The monolith passes only SUCCESSFUL actions (_turn_actions_ran):
        # with none, the claim stands out.
        self.assertIsNotNone(cv.find_note_claim("Noted for Claude, sir.",
                                                user_text="x"))

    def test_the_owner_handing_something_to_keep(self):
        for owner in ("Remember that my locker is number 12.",
                      "Don't forget the meeting moved to Friday.",
                      "I prefer metric units for printing.",
                      "Could you remember that I'm on leave next week?"):
            for text in ("I'll remember that, sir.", "I've noted that, sir.",
                         "Noted, sir - I'll keep that in mind."):
                with self.subTest(owner=owner, text=text):
                    self.assertIsNone(cv.find_note_claim(text,
                                                         user_text=owner))

    def test_a_statement_about_himself_grounds_remember_only(self):
        owner = "I'm off to the gym at six tomorrow."
        self.assertIsNone(cv.find_note_claim("I'll keep that in mind, sir.",
                                             user_text=owner))
        self.assertIsNotNone(cv.find_note_claim(
            "I've noted that for your schedule.", user_text=owner))

    def test_the_owner_words_never_ground_a_list_or_a_file(self):
        owner = "Remember to buy milk."
        for text in ("I've added it to your shopping list, sir.",
                     "I've saved it to your desktop, sir."):
            with self.subTest(text=text):
                self.assertIsNotNone(cv.find_note_claim(text,
                                                        user_text=owner))

    def test_a_remember_question_is_not_a_keep(self):
        # "do you remember ..." asks for something; it hands nothing over.
        self.assertFalse(cv._nc_owner_keeps(
            "Do you remember what the forecast said?"))
        self.assertTrue(cv._nc_owner_keeps("Remember the code is 4417."))
        # A plain remark is not a keep either (the live room line).
        self.assertFalse(cv._nc_owner_keeps(_ROOM_LINE))


class WriterGroundingTests(unittest.TestCase):
    """Only an action that really writes grounds a claim; a read-only or
    opposite action sharing a word with the claim does not (review)."""
    UNGROUNDED = (
        ("I've added eggs to your shopping list, sir.",
         ("show_shopping_list", "remove_shopping_item",
          "clear_shopping_list")),
        ("I've added the dentist to your calendar, sir.",
         ("calendar_today", "calendar_next", "ms_graph_calendar")),
        ("I've added that to your reminders, sir.",
         ("list_timers", "show_reminders_list", "remove_reminder_item",
          "clear_reminders_list")),
        ("I've noted that for your presentation.",
         ("browser_task", "list_schedule", "show_tasks")),
        ("I'll remember that, sir.",
         ("forget_face", "list_enrolled_faces", "session_memory_recall",
          "voice_clone_status")),
        ("Saved, sir.", ("list_windows", "see_screen")),
    )

    def test_a_read_only_or_opposite_action_grounds_nothing(self):
        for text, names in self.UNGROUNDED:
            for name in names:
                with self.subTest(text=text, ran=name):
                    self.assertIsNotNone(cv.find_note_claim(
                        text, ran_actions=[name], user_text=_ROOM_LINE))
                    kept, bad = cv.drop_note_claims(
                        text, ran_actions=[name], user_text=_ROOM_LINE)
                    self.assertEqual(kept, "")
                    self.assertTrue(bad)

    def test_the_real_writer_still_grounds_it(self):
        cases = (
            ("I've added eggs to your shopping list, sir.",
             "add_shopping_item"),
            ("I've added that to your reminders, sir.", "add_reminder"),
            ("I've noted that for your presentation.", "note_for_claude"),
            ("I'll remember that, sir.", "enroll_face"),
            ("Saved, sir.", "export_memory"),
        )
        for text, name in cases:
            with self.subTest(text=text, ran=name):
                self.assertIsNone(cv.find_note_claim(
                    text, ran_actions=[name], user_text=_ROOM_LINE))


class MemoryTargetTests(unittest.TestCase):
    """A claim that names memory as its target is a remember claim, so the
    owner's "remember ..." turn grounds it whatever the verb."""
    OWNER_TURNS = ("remember that my car is in bay four",
                   "from now on call me boss",
                   "don't forget my locker code is 4417")

    def test_memory_target_with_any_verb_is_kept_on_a_handover(self):
        for owner in self.OWNER_TURNS:
            for text in ("Saved to memory, sir.",
                         "I've added that to my memory, sir.",
                         "I've stored that in memory, sir.",
                         "I've memorised that, sir.",
                         "I've saved that, sir."):
                with self.subTest(owner=owner, text=text):
                    self.assertEqual(cv.drop_note_claims(
                        text, user_text=owner), (text, []))

    def test_memory_target_without_a_handover_is_a_remember_claim(self):
        for text in ("Saved to memory, sir.",
                     "I've added that to my memory, sir.",
                     "I've stored that in memory, sir."):
            with self.subTest(text=text):
                self.assertIsNotNone(cv.find_note_claim(
                    text, user_text=_ROOM_LINE))
                self.assertEqual(
                    cv.note_claim_honest_line(text, user_text=_ROOM_LINE),
                    cv.NOTE_CLAIM_HONEST_LINES["remember"])

    def test_a_handover_never_grounds_a_save_with_a_destination(self):
        self.assertIsNotNone(cv.find_note_claim(
            "I've saved that to your desktop, sir.",
            user_text="remember that my car is in bay four"))


class ReportNotClaimTests(unittest.TestCase):
    """A diagnosis or a report that happens to use the words is content."""
    def test_putting_it_down_to_a_cause_is_not_a_note(self):
        for text in ("I'm putting that down to a driver glitch, sir.",
                     "Putting it down to a driver glitch, sir.",
                     "Put that down to a driver glitch, sir.",
                     "I'll put that down to a driver glitch."):
            with self.subTest(text=text):
                self.assertIsNone(cv.find_note_claim(
                    text, user_text="why did my screen flicker just now"))

    def test_noted_the_following_keeps_the_report(self):
        text = ("I've noted the following, sir: memory is at 82 percent and "
                "the GPU is busy.")
        self.assertEqual(cv.drop_note_claims(
            text, ran_actions=["system_pulse"],
            user_text="why is the ram so high"), (text, []))

    def test_only_the_claim_clause_goes_when_a_report_shares_the_sentence(self):
        kept, bad = cv.drop_note_claims(
            "I've noted that: memory is at 82 percent and the GPU is busy.",
            user_text="why is the ram so high")
        self.assertEqual(kept, "Memory is at 82 percent and the GPU is busy.")
        self.assertEqual(len(bad), 1)
        kept, bad = cv.drop_note_claims(
            "Memory is at 82 percent, sir; I've noted that.",
            user_text="why is the ram so high")
        self.assertEqual(kept, "Memory is at 82 percent, sir.")
        self.assertEqual(len(bad), 1)

    def test_a_claim_with_only_courtesy_beside_it_is_dropped_whole(self):
        kept, bad = cv.drop_note_claims(
            "Certainly, sir; I've noted that.", user_text=_ROOM_LINE)
        self.assertEqual(kept, "")
        self.assertEqual(len(bad), 1)


class NotAClaimTests(unittest.TestCase):
    TEXTS = (
        "Noted, sir.",
        "Duly noted, sir.",
        "Noted, sir. I'll keep it brief.",
        "Your dedication to the craft is noted.",
        "I noted that your CPU was running hot.",
        "Shall I note that for Claude?",
        "I can add it to your shopping list if you'd like.",
        "Would you like me to save it?",
        "I haven't saved it, sir.",
        "I have not added anything to your list.",
        "You could add it to your calendar.",
        "Saving money is hard work.",
        "I've saved you a trip, sir.",
        "I remember that film well.",
        "Logging you in now.",
        "I've logged in.",
        "I've put that down to experience.",
        "Recorded in 1969, the album sold well.",
        "Opening it now, sir.",
        "Noted, sir, for what it's worth.",
    )

    def test_none_is_a_claim(self):
        for text in self.TEXTS:
            with self.subTest(text=text):
                self.assertIsNone(cv.find_note_claim(text,
                                                     user_text=_ROOM_LINE))
                self.assertEqual(cv.drop_note_claims(
                    text, user_text=_ROOM_LINE), (text, []))

    def test_a_dated_report_is_not_a_claim(self):
        self.assertIsNone(cv.find_note_claim(
            "I noted it down earlier, sir.", user_text="x"))

    def test_the_owner_asking_whether_it_was_noted(self):
        for owner in ("Did you add milk to the list?",
                      "Have you noted that for Claude?",
                      "Is it saved?"):
            with self.subTest(owner=owner):
                self.assertIsNone(cv.find_note_claim(
                    "Yes, sir, I've added it to your list.", user_text=owner))


class DropTests(unittest.TestCase):
    def test_other_content_is_kept_with_the_tag(self):
        text = ("[intent:confirmation] Certainly, sir. I've noted that for "
                "your presentation. The next slide covers the budget.")
        kept, bad = cv.drop_note_claims(text, user_text=_ROOM_LINE)
        self.assertEqual(kept, "[intent:confirmation] Certainly, sir. The "
                               "next slide covers the budget.")
        self.assertEqual(len(bad), 1)

    def test_a_dropped_first_sentence_recapitalises(self):
        kept, _bad = cv.drop_note_claims(
            "I've saved it. the file is on your desktop.", user_text="x")
        self.assertEqual(kept, "The file is on your desktop.")

    def test_garbage_never_raises(self):
        for bad in (None, "", "   ", 42, "[intent:x]"):
            with self.subTest(bad=bad):
                cv.find_note_claim(bad)
                cv.drop_note_claims(bad)
                self.assertIn(cv.note_claim_honest_line(bad),
                              cv.NOTE_CLAIM_HONEST_LINES.values())


if __name__ == "__main__":
    unittest.main()
