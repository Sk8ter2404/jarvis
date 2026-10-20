"""skills.server_watch - READ-ONLY home-server watch (2026-10-19). No socket,
ssh or file outside a temp dir is touched: the port opener and the runner are
fakes.

    python -m unittest tests.skills.test_server_watch
"""
from __future__ import annotations

import os
import tempfile
import time
import types
import unittest
from unittest import mock

from skills import server_watch as W

# Live 2026-10-19 13:05, two disks rebuilding (values verbatim).
MDCMD_REBUILD = """sbSynced=1792428815
mdState=STARTED
mdNumDisks=18
mdNumDisabled=0
mdNumInvalid=2
mdNumMissing=0
mdResyncAction=recon D10 D14
mdResyncSize=1953514552
mdResync=1953514552
mdResyncPos=1455489016
mdResyncDt=32
mdResyncDb=2896320
"""
MDCMD_QUIET = "mdState=STARTED\nmdNumDisabled=0\nmdResync=0\nmdResyncPos=0\nmdResyncSize=1953514552\n"


def _ran(stdout="", rc=0):
    return types.SimpleNamespace(stdout=stdout, returncode=rc)


class _Cfg(unittest.TestCase):
    def setUp(self):
        from core import config as cfg
        for k, v in (("SERVER_WATCH_HOST", "nas.example"), ("SERVER_WATCH_NAME", "server"),
                     ("SERVER_WATCH_SSH", "root@nas.example")):
            p = mock.patch.object(cfg, k, v, create=True)
            p.start()
            self.addCleanup(p.stop)


class CheckTests(_Cfg):
    def test_any_answering_service_means_up_and_the_array_is_read_once(self):
        run = mock.Mock(return_value=_ran(MDCMD_REBUILD))
        r = W.check(opener=lambda h, port: port in (22, 32400), run=run)
        self.assertTrue(r["up"])
        self.assertEqual(r["services"], ["SSH", "Plex"])
        self.assertEqual(r["array"]["mdResyncAction"], "recon D10 D14")
        argv = run.call_args[0][0]
        self.assertEqual(argv[-1], "mdcmd status")          # the ONLY remote command
        self.assertIn("BatchMode=yes", argv)
        run.assert_called_once()

    def test_down_never_tries_ssh(self):
        run = mock.Mock()
        r = W.check(opener=lambda h, port: False, run=run)
        self.assertFalse(r["up"])
        run.assert_not_called()

    def test_an_unreadable_array_is_empty_not_a_guess(self):
        r = W.check(opener=lambda h, p: True, run=mock.Mock(return_value=_ran("", 255)))
        self.assertEqual(r["array"], {})
        self.assertIn("couldn't read the array", W.describe(r, {}))


class DescribeTests(_Cfg):
    def test_rebuild_is_named_with_disks_progress_and_time_left(self):
        r = {"up": True, "services": ["SSH"], "array": W.parse_mdcmd(MDCMD_REBUILD)}
        line = W.describe(r, {})
        self.assertIn("The server's up, sir: SSH answering.", line)
        self.assertIn("no disabled disks", line)
        self.assertIn("rebuilding disk 10 and disk 14, 75 percent done, about 1.5 hours left", line)

    def test_parity_check_and_quiet_array(self):
        a = W.parse_mdcmd(MDCMD_REBUILD.replace("recon D10 D14", "check P"))
        self.assertTrue(W.resync_line(a).startswith("A parity check is running"))
        self.assertEqual(W.resync_line(W.parse_mdcmd(MDCMD_QUIET)), "")

    def test_offline_says_since_when_and_last_up(self):
        now = time.mktime((2026, 10, 19, 13, 0, 0, 0, 0, -1))
        st = {"believed": "down", "since": now - 3600, "last_up": now - 4000}
        line = W.describe({"up": False}, st, now=now)
        self.assertTrue(line.startswith("The server's offline, sir: none of its services answered since 12:00 PM today"))
        self.assertIn("last up", line)


class StepTests(_Cfg):
    def test_a_change_needs_two_polls_and_a_fresh_start_is_silent(self):
        st, msg = W.step({}, {"up": False, "at": 1.0})
        self.assertEqual(msg, "")
        st, msg = W.step(st, {"up": False, "at": 2.0})
        self.assertEqual((st["believed"], msg), ("down", ""))        # first belief: no announcement
        st, msg = W.step(st, {"up": True, "at": 3.0})
        self.assertEqual(msg, "")                                     # one poll is not enough
        st, msg = W.step(st, {"up": True, "at": 4.0})
        self.assertEqual(msg, "The server is back online, sir.")
        st, msg = W.step(st, {"up": False, "at": 5.0})
        st, msg = W.step(st, {"up": False, "at": 6.0})
        self.assertEqual(msg, "Sir, the server just went offline.")

    def test_more_disabled_disks_and_a_finished_rebuild_are_announced_once(self):
        st = {"believed": "up"}
        busy = W.parse_mdcmd(MDCMD_REBUILD)
        st, msg = W.step(st, {"up": True, "at": 1.0, "array": busy})
        self.assertEqual(msg, "")
        st, msg = W.step(st, {"up": True, "at": 2.0, "array": W.parse_mdcmd(MDCMD_QUIET)})
        self.assertEqual(msg, "The server finished its rebuild, sir.")
        bad = dict(W.parse_mdcmd(MDCMD_QUIET), mdNumDisabled=1)
        st, msg = W.step(st, {"up": True, "at": 3.0, "array": bad})
        self.assertEqual(msg, "Sir, the server reports 1 disabled disk.")
        st, msg = W.step(st, {"up": True, "at": 4.0, "array": bad})
        self.assertEqual(msg, "")


class StateFileTests(unittest.TestCase):
    def test_state_round_trips_and_a_bad_file_is_empty(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "s.json")
        W.save_state({"believed": "up"}, p)
        self.assertEqual(W.load_state(p), {"believed": "up"})
        open(p, "w").write("{bad")
        self.assertEqual(W.load_state(p), {})


class RouteAndRegisterTests(unittest.TestCase):
    def test_server_questions_route_and_other_servers_do_not(self):
        for t in ("Jarvis, is the server up?", "how's the server", "check on unraid",
                  "is the NAS online", "any news on the server"):
            with self.subTest(t=t):
                self.assertEqual(W._route(t), "[ACTION: server_status]")
        for t in ("is the minecraft server up", "how's the discord server",
                  "check the voice server", "turn on the fan", ""):
            with self.subTest(t=t):
                self.assertIsNone(W._route(t))

    def test_rebuild_questions_route_when_a_server_is_watched(self):
        from core import config as cfg
        with mock.patch.object(cfg, "SERVER_WATCH_HOST", "nas.example", create=True):
            for t in ("Jarvis, how long until that data rebuild's done?",
                      "how far along is the parity check", "is the rebuild finished"):
                with self.subTest(t=t):
                    self.assertEqual(W._route(t), "[ACTION: server_status]")
            for t in ("rebuild the website", "how long until the index rebuild is done"):
                with self.subTest(t=t):
                    self.assertIsNone(W._route(t))
        with mock.patch.object(cfg, "SERVER_WATCH_HOST", "", create=True):
            self.assertIsNone(W._route("how long until the rebuild is done"))

    def test_rebuild_wording_routes_the_prompt_examples_to_the_brain(self):
        # 2026-10-20 brain test set (c033, c037): a rebuild question the
        # route above does not catch still reached the brain without the
        # server_status grammar, so it guessed. The examples' own quoted
        # triggers are what core.prompt_router routes on.
        from core import prompt_router as pr
        parts = [("server_watch", W.PROMPT_EXAMPLES.strip())]
        for t in ("Jarvis, how is the rebuild looking? Is it healthy?",
                  "Jarvis, how long until that data rebuild's done?",
                  "how far along is the parity check"):
            with self.subTest(t=t):
                self.assertEqual(len(pr.select_skill_examples(t, parts)), 1)
        self.assertEqual(pr.select_skill_examples("play some jazz", parts), [])

    def test_registered_actions_and_no_thread_under_tests(self):
        actions = {}
        with mock.patch.object(W.threading, "Thread") as thread:
            W.register(actions)
        for a in ("server_status",):
            self.assertIn(a, actions)
        thread.assert_not_called()

    def test_off_without_a_host(self):
        from core import config as cfg
        with mock.patch.object(cfg, "SERVER_WATCH_HOST", "", create=True):
            self.assertIn("isn't in my settings", W.server_status(""))


if __name__ == "__main__":
    unittest.main()
