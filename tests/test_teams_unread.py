"""core/teams_unread.py (v3.0.11): which words ask for the Teams reading, and the
sentence it speaks from what Graph and notification triage hold. Live 2026-11-05
14:43:39: "I've no way to check that, sir ... you have 24 unread messages".
"""
import unittest

from core import teams_unread as tu

NOW = 1_800_000_000.0


class AsksTests(unittest.TestCase):
    def test_natural_asks_route(self):
        for line in (
            "Jarvis, any messages I missed on Teams while I was gone?",
            "any messages I missed",
            "any messages I might have missed?",
            "messages I missed",
            "any Teams messages",
            "any new Teams messages?",
            "do I have any unread Teams messages",
            "did I miss any messages",
            "have I got any messages?",
            "are there any messages for me",
            "were there any messages while I was away",
            "any messages",
            "any pings?",
            "did anyone message me",
            "did anybody message me on Teams?",
            "has anyone messaged me today",
            "did someone ping me",
            "anyone message me?",
            "who messaged me",
            "Jarvis, who's messaged me?",
            "who has messaged me",
            "who's been messaging me",
            "who sent me a message",
            "how many unread messages do I have",
            "how many Teams messages",
            "anything new on Teams",
            "what's new on Teams",
            "what did I miss on Teams?",
            "check Teams",
            "check my Teams messages",
            "can you check if I have any messages",
            "tell me if there are any messages I missed",
        ):
            with self.subTest(line=line):
                self.assertTrue(tu.asks_teams_messages(line))

    def test_other_channels_stay_away(self):
        for line in (
            "any email?",
            "any new email",
            "any mail",
            "any unread mail",
            "did anyone email me",
            "any messages in my inbox",
            "any email messages",
            "any text messages",
            "any messages on WhatsApp",
            "who messaged me on Discord",
            "any Slack messages",
            "any voicemail",
            "any pings to the server",
            "any chats with Claude",
            "any messages in the log",
        ):
            with self.subTest(line=line):
                self.assertFalse(tu.asks_teams_messages(line))

    def test_ordinary_speech_stays_away(self):
        for line in (
            "",
            None,
            "what's on TV",
            "any good movies on TV",
            "what did I miss",
            "open Teams",
            "open Teams on the top monitor",
            "close Teams",
            "the messages I missed were boring",
            "I missed a lot of messages yesterday",
            "send a message to the team",
            "message him on Teams",
            "don't send any messages",
            "there aren't any messages",
            "if anyone messages me let me know",
            "who messaged me about the invoice last Tuesday afternoon",
            "any messages from the landlord",
            "who messaged you",
            "I think the teams are playing tonight",
            "which teams are playing",
            "the chat was fun",
            "check the weather",
            "any news",
        ):
            with self.subTest(line=line):
                self.assertFalse(tu.asks_teams_messages(line))


class ComposeTests(unittest.TestCase):
    def toast(self, title="Sam Example", age=720.0, app="Microsoft Teams",
              action="read_aloud", body="the secret words"):
        return {"ts": NOW - age, "app": app, "title": title, "body": body,
                "action": action}

    def test_a_count_and_the_newest_sender(self):
        self.assertEqual(
            tu.compose({"count": 24, "top_sender": "Sam"}, 200, [], NOW),
            "24 unread Teams chats, sir, the newest from Sam.")
        self.assertEqual(
            tu.compose({"count": 1, "top_sender": "Sam"}, 200, None, NOW),
            "One unread Teams chat, sir, from Sam.")
        self.assertEqual(
            tu.compose({"count": 3, "top_sender": ""}, 200, None, NOW),
            "3 unread Teams chats, sir.")

    def test_nothing_unread(self):
        self.assertEqual(tu.compose({"count": 0, "top_sender": ""}, 200,
                                    [self.toast()], NOW),
                         "Nothing unread on Teams, sir.")

    def test_toasts_add_when_they_add_something(self):
        same = tu.compose({"count": 2, "top_sender": "Sam"}, 200, [self.toast()], NOW)
        self.assertEqual(same, "2 unread Teams chats, sir, the newest from Sam. "
                               "The latest Teams notification reached me 12 minutes ago.")
        other = tu.compose({"count": 2, "top_sender": "Sam"}, 200,
                           [self.toast("Pat Example")], NOW)
        self.assertIn("notification from Pat Example reached me", other)

    def test_graph_unavailable_says_why_and_never_guesses_a_count(self):
        for status, words in (("unconfigured", "because I'm not signed in to Microsoft"),
                              (0, "because I'm not signed in to Microsoft"),
                              (403, "won't let me read your Teams chats"),
                              (401, "won't let me read your Teams chats"),
                              (500, "couldn't reach"),
                              (-1, "couldn't reach"),
                              (None, "couldn't reach")):
            with self.subTest(status=status):
                out = tu.compose(None, status, None, NOW)
                self.assertIn(words, out)
                self.assertIn("can't say what's unread", out)
                self.assertNotRegex(out, r"\d")
                self.assertNotIn("Graph", out)

    def test_graph_unavailable_still_gives_the_toasts(self):
        out = tu.compose(None, 0, [self.toast(age=300), self.toast("Pat", age=60)], NOW)
        self.assertIn("not signed in to Microsoft", out)
        self.assertIn("2 Teams notifications have reached me", out)
        self.assertIn("the latest from Pat, just now", out)
        one = tu.compose(None, 0, [self.toast()], NOW)
        self.assertIn("One Teams notification has reached me", one)
        self.assertIn("from Sam Example, 12 minutes ago", one)

    def test_a_recent_restart_never_claims_twelve_hours(self):
        hour = 3600.0
        toast = [self.toast()]
        self.assertIn("No Teams notification has reached me since I started,",
                      tu.compose(None, 0, [], NOW, since=NOW - 2 * hour))
        self.assertIn("One Teams notification has reached me since I started, from",
                      tu.compose(None, 0, toast, NOW, since=NOW - 2 * hour))
        for since in (NOW - 12 * hour, NOW - 40 * hour, None, "junk"):
            with self.subTest(since=since):
                self.assertIn("in the last 12 hours",
                              tu.compose(None, 0, [], NOW, since=since))

    def test_no_toasts_is_not_proof_of_nothing(self):
        out = tu.compose(None, 0, [], NOW)
        self.assertIn("doesn't prove nothing is unread", out)

    def test_a_message_body_is_never_spoken(self):
        for graph in (None, {"count": 2, "top_sender": "Sam"}):
            out = tu.compose(graph, 0, [self.toast("Sam: the secret words")], NOW)
            self.assertNotIn("secret", out)
            self.assertIn("Sam", out)

    def test_other_apps_old_and_dropped_toasts_do_not_count(self):
        recs = [self.toast(app="Outlook"), self.toast(age=13 * 3600),
                self.toast(action="drop"), {"junk": 1}, "not a dict"]
        self.assertEqual(tu.teams_toasts(recs, NOW), [])

    def test_toast_name(self):
        self.assertEqual(tu.toast_name("Sam Example"), "Sam Example")
        self.assertEqual(tu.toast_name("Sam Example: are you around?"), "Sam Example")
        for bad in ("", None, "a@b.co", "1234567 is your code",
                    "this title is much too long to be a name"):
            with self.subTest(bad=bad):
                self.assertEqual(tu.toast_name(bad), "")

    def test_long_ago(self):
        self.assertIn("3 hours ago", tu.compose(None, 0, [self.toast(age=3 * 3600)], NOW))


if __name__ == "__main__":
    unittest.main()
