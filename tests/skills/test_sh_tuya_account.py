"""skills.sh_tuya smart_home_account - "which account are my smart devices
on?" (live 2026-10-18 21:57 the model said it had "no way to check"). Tuya is
never contacted: the cloud is a fake and the wizard file is a temp copy.

    python -m unittest tests.skills.test_sh_tuya_account
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from skills import sh_tuya as T


class _Cloud:
    def __init__(self, result=None, success=True, boom=False):
        self.result, self.success, self.boom, self.asked = result, success, boom, []

    def cloudrequest(self, url):
        self.asked.append(url)
        if self.boom:
            raise RuntimeError("no network")
        return {"success": self.success, "result": self.result}


class AccountTests(unittest.TestCase):
    def setUp(self):
        T._account_cache.update(at=0.0, info=None)
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_phone_login_is_spoken_by_its_last_four_digits_only(self):
        cloud = _Cloud({"username": "1-5550001234", "create_time": 1553376212})
        with mock.patch.object(T, "_all_devices", return_value=[{}] * 7):
            line = T.smart_home_account("", cloud=cloud, uid="az123")
        self.assertEqual(cloud.asked, ["/v1.0/users/az123/infos"])
        self.assertIn("Your 7 Tuya devices are in the Smart Life app, sir", line)
        self.assertIn("phone number ending 1234", line)
        self.assertNotIn("5550001234", line)
        self.assertIn("March 2019", line)
        self.assertIn("Tuya Smart app is a separate account", line)

    def test_email_login_and_one_device(self):
        cloud = _Cloud({"username": "owner@example.com"})
        with mock.patch.object(T, "_all_devices", return_value=[{}]):
            line = T.smart_home_account("", cloud=cloud, uid="az1")
        self.assertTrue(line.startswith("Your Tuya device is in the Smart Life app"))
        self.assertIn("the email owner@example.com", line)

    def test_tuya_unreachable_is_said_not_guessed(self):
        for cloud in (_Cloud(boom=True), _Cloud(success=False)):
            with self.subTest(cloud=cloud.boom), \
                    mock.patch.object(T, "_all_devices", return_value=[]):
                line = T.smart_home_account("", cloud=cloud, uid="az1")
            self.assertTrue(line.startswith("I couldn't reach Tuya"))

    def test_no_wizard_file_means_no_linked_account(self):
        self.assertEqual(T._linked_uid(os.path.join(self.dir, "missing.json")), "")
        self.assertIn("hasn't been run", T.smart_home_account("", cloud=_Cloud(), uid=""))

    def test_the_uid_comes_from_the_wizard_output(self):
        p = os.path.join(self.dir, "tuya-raw.json")
        with open(p, "w") as f:
            json.dump({"result": [{"name": "Lamp", "uid": "azUID"}]}, f)
        self.assertEqual(T._linked_uid(p), "azUID")


class RouteTests(unittest.TestCase):
    def test_account_questions_route(self):
        for t in ("Jarvis, what account do I use for my 2UA account?",
                  "which account are my smart life devices on",
                  "what login is my tuya on"):
            with self.subTest(t=t):
                self.assertEqual(T._account_route(t), "[ACTION: smart_home_account]")

    def test_other_smart_home_lines_do_not(self):
        for t in ("add a device to smart life", "turn on the fan", "link my tuya account",
                  "log out of smart life", ""):
            with self.subTest(t=t):
                self.assertIsNone(T._account_route(t))

    def test_registered_with_the_list_actions(self):
        actions = {}
        T.register(actions)
        for a in ("tuya_list", "smart_life_list", "smart_home_account"):
            self.assertIn(a, actions)


if __name__ == "__main__":
    unittest.main()
