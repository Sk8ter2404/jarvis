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
        cmds = [c[0][0][-1] for c in run.call_args_list]
        # mdcmd once, plus the three 3.0.6 health reads - nothing else, ever.
        self.assertEqual(sorted(cmds), sorted(W.READ_COMMANDS))
        for c in run.call_args_list:
            self.assertIn("BatchMode=yes", c[0][0])

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


# ── 3.0.6: health reads + the restricted key (scope D / D2) ────────────
# disks.ini in Unraid's own layout (["name"] sections, key="value"); the
# values are made up, the shape is what /var/local/emhttp/disks.ini holds.
DISKS_OK = """["parity"]
idx="0"
name="parity"
device="sdb"
status="DISK_OK"
temp="38"
numReads="1234"
numWrites="5678"
numErrors="0"
["disk1"]
idx="1"
name="disk1"
temp="*"
numErrors="0"
["cache"]
name="cache"
temp="41"
numErrors="0"
"""
# `apcaccess -u` (units stripped); LOADPCT/BCHARGE/TIMELEFT are never read.
UPS_ONLINE = ("APC      : 001,036,0857\nSTATUS   : ONLINE \nLOADPCT  : 112.0\n"
              "BCHARGE  : 100.0\nTIMELEFT : 4.5\n")
UPS_ONBATT = UPS_ONLINE.replace("ONLINE ", "ONBATT ")
DOCKER_UP = "plex Up 3 days\nswag Up 3 days\nadguard Up 2 hours\n"
DOCKER_SWAG_EXITED = "plex Up 3 days\nswag Exited (1) 4 minutes ago\nadguard Up 2 hours\n"

KEY = r"C:\Users\someone\.ssh\jarvis_ro_ed25519"


def _runner(disks=DISKS_OK, ups=UPS_ONLINE, docker=DOCKER_UP, mdcmd=MDCMD_QUIET):
    by_cmd = {W.CMD_MDCMD: mdcmd, W.CMD_DISKS: disks, W.CMD_UPS: ups, W.CMD_DOCKER: docker}

    def run(argv, **kw):
        out = by_cmd.get(argv[-1])
        return _ran(out, 0) if out is not None else _ran("", 255)
    return mock.Mock(side_effect=run)


class ReadCommandTests(_Cfg):
    def test_the_command_tuple_is_byte_for_byte_the_allow_list(self):
        # Home Lab hub 2026-10-28: jarvis-ro.sh refuses anything else.
        self.assertEqual(W.READ_COMMANDS, (
            "mdcmd status",
            "cat /var/local/emhttp/disks.ini",
            "apcaccess -u",
            "docker ps -a --format '{{.Names}} {{.Status}}'"))

    def test_no_key_keeps_todays_exact_argv(self):
        from core import config as cfg
        with mock.patch.object(cfg, "SERVER_WATCH_SSH_KEY", "", create=True):
            run = _runner()
            W.check(opener=lambda h, p: True, run=run)
        argvs = {c[0][0][-1]: c[0][0] for c in run.call_args_list}
        self.assertEqual(argvs["mdcmd status"],
                         ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                          "-o", "StrictHostKeyChecking=accept-new", "root@nas.example",
                          "mdcmd status"])

    def test_the_key_adds_identities_only_in_the_home_lab_order(self):
        from core import config as cfg
        with mock.patch.object(cfg, "SERVER_WATCH_SSH_KEY", KEY, create=True),                 mock.patch.object(W.os.path, "isfile", lambda p: p == KEY):
            run = _runner()
            W.check(opener=lambda h, p: True, run=run)
        self.assertEqual(run.call_count, 4)
        for c in run.call_args_list:
            cmd = c[0][0][-1]
            with self.subTest(cmd=cmd):
                self.assertEqual(c[0][0], [
                    "ssh", "-n", "-i", KEY, "-o", "IdentitiesOnly=yes",
                    "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                    "-o", "StrictHostKeyChecking=accept-new", "root@nas.example", cmd])
                self.assertIn(cmd, W.READ_COMMANDS)

    def test_a_bad_key_runs_nothing_rather_than_fall_back_to_the_root_key(self):
        from core import config as cfg
        for bad in ("-oProxyCommand=calc", "C:\\k\nx"):
            with self.subTest(bad=bad), \
                    mock.patch.object(cfg, "SERVER_WATCH_SSH_KEY", bad, create=True):
                run = _runner()
                r = W.check(opener=lambda h, p: True, run=run)
                run.assert_not_called()
                self.assertEqual((r["array"], r["health"]), ({}, {}))

    def test_a_missing_key_file_runs_nothing_rather_than_fall_back_to_the_root_key(self):
        # Review 2026-10-28: ssh only warns about an unreadable `-i` file and
        # then offers ~/.ssh/id_ed25519 (full root; IdentitiesOnly does not
        # stop the defaults). A typo / deleted / not-yet-deployed key must
        # therefore run NOTHING, checked on every call.
        from core import config as cfg
        missing = KEY + ".missing"
        with mock.patch.object(cfg, "SERVER_WATCH_SSH_KEY", missing, create=True),                 mock.patch.object(W.os.path, "isfile", lambda p: False):
            self.assertIs(W.ssh_key(), W._INVALID_KEY)
            self.assertIsNone(W.ssh_argv("root@nas.example", W.CMD_UPS))
            run = _runner()
            r = W.check(opener=lambda h, p: True, run=run)
            run.assert_not_called()
            self.assertEqual((r["array"], r["health"]), ({}, {}))

    def test_the_key_is_rechecked_on_every_call(self):
        from core import config as cfg
        present = [True]
        with mock.patch.object(cfg, "SERVER_WATCH_SSH_KEY", KEY, create=True),                 mock.patch.object(W.os.path, "isfile", lambda p: present[0]):
            self.assertIsNotNone(W.ssh_argv("root@nas.example", W.CMD_UPS))
            present[0] = False      # removed while JARVIS runs
            self.assertIsNone(W.ssh_argv("root@nas.example", W.CMD_UPS))

    def test_a_command_outside_the_tuple_is_never_run(self):
        self.assertIsNone(W.ssh_argv("root@nas.example", "rm -rf /mnt/user"))
        self.assertIsNone(W.ssh_argv("root@nas.example", "apcaccess -u; reboot"))
        run = mock.Mock()
        self.assertIsNone(W._run_read("root@nas.example", "mdcmd stop", run))
        run.assert_not_called()

    def test_spoken_text_never_reaches_the_argv(self):
        from core import config as cfg
        run = _runner()
        with mock.patch.object(W, "load_state", return_value={}), \
                mock.patch.object(W, "save_state"), \
                mock.patch.object(cfg, "SERVER_WATCH_SSH_KEY", "", create=True):
            # check()'s defaults are bound at import; route the real
            # server_status through a check that uses the fake runner.
            real = W.check
            with mock.patch.object(W, "check", lambda: real(opener=lambda h, p: True, run=run)):
                W.server_status("mdcmd stop; docker rm -f plex")
        for c in run.call_args_list:
            self.assertIn(c[0][0][-1], W.READ_COMMANDS)

    def test_health_reads_off_runs_only_mdcmd(self):
        from core import config as cfg
        with mock.patch.object(cfg, "SERVER_WATCH_HEALTH_READS", False, create=True):
            run = _runner()
            r = W.check(opener=lambda h, p: True, run=run)
        run.assert_called_once()
        self.assertEqual(run.call_args[0][0][-1], "mdcmd status")
        self.assertEqual(r["health"], {})

    def test_a_failed_read_is_left_out_not_guessed(self):
        run = _runner(docker=None, ups=None)
        r = W.check(opener=lambda h, p: True, run=run)
        self.assertEqual(set(r["health"]), {"disks"})


class ParseHealthTests(unittest.TestCase):
    def test_disks_ini(self):
        d = W.parse_disks_ini(DISKS_OK)
        self.assertEqual(d["parity"], {"temp": 38, "errors": 0})
        self.assertEqual(d["disk1"], {"temp": None, "errors": 0})   # spun down
        self.assertEqual(W.parse_disks_ini("garbage"), {})

    def test_ups_status_only(self):
        self.assertEqual(W.parse_ups_status(UPS_ONLINE), "ONLINE")
        self.assertEqual(W.parse_ups_status(UPS_ONBATT), "ONBATT")
        self.assertEqual(W.parse_ups_status("STATUS   : ONBATT LOWBATT \n"), "ONBATT")
        self.assertEqual(W.parse_ups_status("STATUS   : COMMLOST \n"), "")
        self.assertEqual(W.parse_ups_status(""), "")

    def test_docker(self):
        self.assertEqual(W.parse_docker(DOCKER_SWAG_EXITED)["swag"], "Exited (1) 4 minutes ago")


class StepHealthTests(_Cfg):
    def _h(self, **kw):
        return W.read_health("root@nas.example", _runner(**kw))

    def test_first_read_is_a_silent_baseline(self):
        st, lines = W.step_health({}, self._h(disks=DISKS_OK.replace('temp="41"', 'temp="55"'),
                                              ups=UPS_ONBATT, docker=DOCKER_SWAG_EXITED),
                                  claim_held=lambda: False)
        self.assertEqual(lines, [])
        st, lines = W.step_health(st, self._h(disks=DISKS_OK.replace('temp="41"', 'temp="55"'),
                                              ups=UPS_ONBATT, docker=DOCKER_SWAG_EXITED),
                                  claim_held=lambda: False)
        self.assertEqual(lines, [])

    def test_disk_errors_rising_and_temp_at_50_speak_once(self):
        st, _ = W.step_health({}, self._h())
        st, lines = W.step_health(st, self._h(disks=DISKS_OK.replace('temp="41"', 'temp="49"')))
        self.assertEqual(lines, [])                                  # 49 C: silent
        st, lines = W.step_health(st, self._h(disks=DISKS_OK.replace('temp="41"', 'temp="50"')))
        self.assertEqual(lines, ["Sir, cache on the server is at 50 degrees Celsius."])
        st, lines = W.step_health(st, self._h(disks=DISKS_OK.replace('temp="41"', 'temp="51"')))
        self.assertEqual(lines, [])                                  # said once
        st, lines = W.step_health(st, self._h(disks=DISKS_OK.replace('numErrors="0"\n["disk1"]',
                                                                     'numErrors="3"\n["disk1"]')))
        self.assertEqual(lines, ["Sir, parity on the server logged 3 new errors."])
        st, lines = W.step_health(st, self._h(disks=DISKS_OK.replace('numErrors="0"\n["disk1"]',
                                                                     'numErrors="3"\n["disk1"]')))
        self.assertEqual(lines, [])

    def test_ups_speaks_only_on_battery_and_back(self):
        st, _ = W.step_health({}, self._h())
        # Load 112 % -> 130 %, charge and runtime changes: never said.
        st, lines = W.step_health(st, self._h(ups=UPS_ONLINE.replace("112.0", "130.0")
                                              .replace("4.5", "1.0")))
        self.assertEqual(lines, [])
        st, lines = W.step_health(st, self._h(ups=UPS_ONBATT))
        self.assertEqual(lines, ["Sir, the server is running on battery."])
        st, lines = W.step_health(st, self._h(ups=UPS_ONBATT.replace("ONBATT ", "ONBATT LOWBATT ")))
        self.assertEqual(lines, [])                                  # low battery: never
        st, lines = W.step_health(st, self._h(ups=UPS_ONLINE))
        self.assertEqual(lines, ["The server is back on mains power, sir."])
        for line in ("load", "overload", "low battery", "runtime", "percent"):
            self.assertNotIn(line, " ".join(lines).lower())

    def test_an_exited_container_is_said_once(self):
        st, _ = W.step_health({}, self._h(), claim_held=lambda: False)
        st, lines = W.step_health(st, self._h(docker=DOCKER_SWAG_EXITED), claim_held=lambda: False)
        self.assertEqual(lines, ["Sir, the swag container on the server stopped."])
        st, lines = W.step_health(st, self._h(docker=DOCKER_SWAG_EXITED), claim_held=lambda: False)
        self.assertEqual(lines, [])
        st, _ = W.step_health(st, self._h(), claim_held=lambda: False)            # back up
        st, lines = W.step_health(st, self._h(docker=DOCKER_SWAG_EXITED), claim_held=lambda: False)
        self.assertEqual(len(lines), 1)                                            # exits again

    def test_silent_while_a_server_claim_is_held(self):
        st, _ = W.step_health({}, self._h(), claim_held=lambda: True)
        st, lines = W.step_health(st, self._h(docker=DOCKER_SWAG_EXITED), claim_held=lambda: True)
        self.assertEqual(lines, [])
        # Claim released, swag still down: the work stopped it - not said later.
        st, lines = W.step_health(st, self._h(docker=DOCKER_SWAG_EXITED), claim_held=lambda: False)
        self.assertEqual(lines, [])

    def test_describe_names_battery_and_hot_disks_only(self):
        r = {"up": True, "services": ["SSH"], "array": W.parse_mdcmd(MDCMD_QUIET),
             "health": self._h(ups=UPS_ONBATT, disks=DISKS_OK.replace('temp="41"', 'temp="52"'))}
        line = W.describe(r, {})
        self.assertIn("It's running on battery.", line)
        self.assertIn("Cache is at 52 degrees Celsius.", line)
        self.assertNotIn("112", line)


class BoardClaimTests(unittest.TestCase):
    def test_a_live_server_claim_is_held(self):
        now = 1793227380.0
        board = {"claims": [
            {"id": "c-1", "chat": "x", "what": "C:/JARVIS", "until": now + 60},
            {"id": "c-2", "chat": "Home Lab hub", "what": "Server:Unraid", "until": now + 600}]}
        fetch = mock.Mock(return_value=board)
        self.assertTrue(W.board_claim_held(fetch=fetch, now=now))
        url, timeout = fetch.call_args[0]
        self.assertEqual((url, timeout), ("http://127.0.0.1:8795/api/board", 2.0))

    def test_expired_other_or_unreachable_is_not_held(self):
        now = 1793227380.0
        self.assertFalse(W.board_claim_held(
            fetch=lambda u, t: {"claims": [{"what": "server:unraid", "until": now - 1}]}, now=now))
        self.assertFalse(W.board_claim_held(
            fetch=lambda u, t: {"claims": [{"what": "dell", "until": now + 60}]}, now=now))

        def down(u, t):
            raise OSError("connection refused")
        self.assertFalse(W.board_claim_held(fetch=down, now=now))
        self.assertFalse(W.board_claim_held(fetch=lambda u, t: {"claims": "bad"}, now=now))


class PollLoopTests(_Cfg):
    def test_one_poll_announces_the_health_line(self):
        # The live path: _poll_loop -> check -> step -> step_health -> _announce.
        state = {"believed": "up", "disabled": 0, "resync": "", "disk_errors": {"parity": 0},
                 "disk_hot": [], "ups": "ONLINE", "exited": []}
        result = {"up": True, "services": ["SSH"], "array": W.parse_mdcmd(MDCMD_QUIET),
                  "health": W.read_health("root@nas.example", _runner(ups=UPS_ONBATT)), "at": 1.0}
        saved = []
        with mock.patch.object(W, "check", return_value=result), \
                mock.patch.object(W, "load_state", return_value=state), \
                mock.patch.object(W, "save_state", side_effect=saved.append), \
                mock.patch.object(W, "board_claim_held", return_value=False), \
                mock.patch.object(W, "_announce") as ann, \
                mock.patch.object(W._stop, "wait", side_effect=[False, True]):
            W._poll_loop()
        ann.assert_called_once_with("Sir, the server is running on battery.")
        self.assertEqual(saved[-1]["ups"], "ONBATT")


if __name__ == "__main__":
    unittest.main()
