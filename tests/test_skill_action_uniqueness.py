"""Guard: one owner per action name across skills/ (stale-duplicate class).

The monolith loader calls every skill's register(ACTIONS) in sorted file order
and later writes win, so two skills registering the same action name make the
winner depend on file naming. 2026-10-14: sh_kasa and the smart-home router
both registered smart_home_control; the router only won because
"sh_kasa" sorts before "smart_home_router_skill".
"""
import unittest

from tests._skill_harness import list_skill_names, load_skill_isolated

# Known duplicates still to be resolved by their owners: name -> skills.
# Found by this test on 2026-10-14, deliberately NOT fixed here (own change).
_KNOWN_DUPLICATES: dict[str, set[str]] = {
    "focus_mode": {"dnd_focus_mode", "focus_mode"},
    "end_focus_mode": {"dnd_focus_mode", "focus_mode"},
    "focus_mode_status": {"dnd_focus_mode", "focus_mode"},
}


def _collect():
    owners: dict[str, set[str]] = {}
    for name in list_skill_names():
        try:
            _mod, acts = load_skill_isolated(name)
        except unittest.SkipTest:
            continue
        except Exception:
            continue
        for action in acts:
            owners.setdefault(action, set()).add(name)
    return owners


class SkillActionUniquenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owners = _collect()

    def test_no_action_registered_by_two_skills(self):
        dups = {a: sorted(s) for a, s in self.owners.items()
                if len(s) > 1 and s != _KNOWN_DUPLICATES.get(a)}
        self.assertEqual(dups, {}, f"action names registered by >1 skill: {dups}")

    def test_smart_home_control_owned_by_router_only(self):
        for action in ("smart_home_control", "control_device"):
            self.assertEqual(self.owners.get(action), {"smart_home_router_skill"},
                             action)

    def test_router_action_is_router_function(self):
        from core import smart_home_router
        acts: dict = {}
        smart_home_router.register(acts)
        self.assertIs(acts["smart_home_control"],
                      smart_home_router.smart_home_control)
        _m, kasa = load_skill_isolated("sh_kasa")
        self.assertNotIn("smart_home_control", kasa)
        self.assertIn("kasa_smart_home_control", kasa)


if __name__ == "__main__":
    unittest.main()
