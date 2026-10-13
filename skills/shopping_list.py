"""
Shopping list + untimed reminders.

Actions added:
  add_reminder, <what>            a reminder that names NO time: goes on the
                                  reminders list. Says where it went. If the
                                  text is about buying something it goes on
                                  the shopping list instead.
  show_reminders_list             read the reminders list back
  remove_reminder_item, <item>    take one reminder off
  clear_reminders_list            empty the reminders list (a .bak is kept)
  add_shopping_item, <a, b, c>    add items to the shopping list
  show_shopping_list              read the list back
  remove_shopping_item, <item>    take one item off
  clear_shopping_list             empty the list (a .bak copy is kept)

"Remind me to buy milk" (no time) lands on the shopping list; any other
reminder with no time lands on the reminders list; a reminder WITH a time stays
a timer / schedule. The reminders list is its own file (data/reminders_list.json),
NEVER jarvis_todo.md: that is the developer task queue the overnight upgrade
pipeline reads as coding tasks, and "the to-do list" already names it by voice. The grammar lives in core/untimed_reminders.py; the monolith
routes whole-utterance requests here without the model, and the prompt teaches
the model the same actions for the phrasings the grammar misses.

Every reply says where the item went, and a failed save says so instead of
claiming success. State: data/shopping_list.json (core.paths, so a staging
process or a test never touches the live file).
"""
from __future__ import annotations

from core import untimed_reminders as _ur

# The replies are finished sentences that say where the item went: spoken
# verbatim (bobert_companion._collect_skill_speak_sets), never re-worded.
SPEAK_VERBATIM_ACTIONS = ("add_reminder", "add_shopping_item",
                          "show_shopping_list", "remove_shopping_item",
                          "clear_shopping_list", "show_reminders_list",
                          "remove_reminder_item", "clear_reminders_list")


def _count_words(n: int) -> str:
    return f"{n} item" + ("" if n == 1 else "s")


def register(actions: dict) -> None:
    def add_shopping_item(args: str = "") -> str:
        raw = (args or "").strip()
        if not raw:
            return ("What should I add to your shopping list, sir? Say "
                    "'add milk to my shopping list'.")
        texts = _ur._split_items(raw) or [raw]
        try:
            added, dup = _ur.add_items(texts)
        except Exception as e:
            return (f"I couldn't save your shopping list, sir "
                    f"({type(e).__name__}) - nothing was added.")
        if not added and dup:
            return (f"{_ur.spoken_join(dup).capitalize()} "
                    f"{'is' if len(dup) == 1 else 'are'} already on your "
                    f"shopping list, sir.")
        if not added:
            return "I didn't catch an item to add, sir."
        line = f"Added {_ur.spoken_join(added)} to your shopping list, sir."
        if dup:
            line += (f" {_ur.spoken_join(dup).capitalize()} "
                     f"{'was' if len(dup) == 1 else 'were'} already there.")
        return line

    def show_shopping_list(_: str = "") -> str:
        items = [i["text"] for i in _ur.load_items()]
        if not items:
            return "Your shopping list is empty, sir."
        return (f"Your shopping list has {_count_words(len(items))}, sir: "
                f"{_ur.spoken_join(items)}.")

    def remove_shopping_item(args: str = "") -> str:
        what = (args or "").strip()
        if not what:
            return "Which item should I take off your shopping list, sir?"
        if what.lower() in _ur._ALL_WORDS:
            return clear_shopping_list("")
        try:
            removed, candidates = _ur.remove_matching(what)
        except Exception as e:
            return (f"I couldn't update your shopping list, sir "
                    f"({type(e).__name__}) - nothing was removed.")
        if removed:
            return f"Took {removed} off your shopping list, sir."
        if candidates:
            return (f"More than one item matches '{what}', sir: "
                    f"{_ur.spoken_join(candidates)}. Which one?")
        return f"I don't see '{what}' on your shopping list, sir."

    def clear_shopping_list(_: str = "") -> str:
        try:
            n = _ur.clear_all()
        except Exception as e:
            return (f"I couldn't clear your shopping list, sir "
                    f"({type(e).__name__}) - it's unchanged.")
        if not n:
            return "Your shopping list was already empty, sir."
        return f"Cleared your shopping list, sir - {_count_words(n)} removed."

    def add_reminder(args: str = "") -> str:
        """A reminder with no time. Shopping goes to the shopping list, the
        rest to the reminders list - never to a timer, never to the
        developer queue."""
        task = (args or "").strip(" ,.")
        if not task:
            return ("What should I remind you about, sir? Say 'remind me to "
                    "call the dentist', or give me a time and I'll set a "
                    "timer.")
        items = _ur.shopping_items(task)
        if items:
            return add_shopping_item(", ".join(items))
        try:
            added, dup = _ur.add_items([task], name=_ur.REMINDERS_LIST)
        except Exception as e:
            return (f"I couldn't save that to your reminders list, sir "
                    f"({type(e).__name__}) - nothing was saved.")
        if not added and dup:
            return f"That's already on your reminders list, sir: {dup[0]}."
        if not added:
            return "I didn't catch what to remind you about, sir."
        return (f"Added to your reminders list, sir: {added[0]}. You gave me "
                f"no time, so I won't alert you.")

    def show_reminders_list(_: str = "") -> str:
        items = [i["text"] for i in _ur.load_items(_ur.REMINDERS_LIST)]
        if not items:
            return "Your reminders list is empty, sir."
        return (f"Your reminders list has {_count_words(len(items))}, sir: "
                f"{_ur.spoken_join(items)}.")

    def remove_reminder_item(args: str = "") -> str:
        what = (args or "").strip()
        if not what:
            return "Which reminder should I take off your list, sir?"
        if what.lower() in _ur._ALL_WORDS:
            return clear_reminders_list("")
        try:
            removed, candidates = _ur.remove_matching(
                what, name=_ur.REMINDERS_LIST)
        except Exception as e:
            return (f"I couldn't update your reminders list, sir "
                    f"({type(e).__name__}) - nothing was removed.")
        if removed:
            return f"Took '{removed}' off your reminders list, sir."
        if candidates:
            return (f"More than one reminder matches '{what}', sir: "
                    f"{_ur.spoken_join(candidates)}. Which one?")
        return f"I don't see '{what}' on your reminders list, sir."

    def clear_reminders_list(_: str = "") -> str:
        try:
            n = _ur.clear_all(name=_ur.REMINDERS_LIST)
        except Exception as e:
            return (f"I couldn't clear your reminders list, sir "
                    f"({type(e).__name__}) - it's unchanged.")
        if not n:
            return "Your reminders list was already empty, sir."
        return f"Cleared your reminders list, sir - {_count_words(n)} removed."

    actions["add_shopping_item"] = add_shopping_item
    actions["show_shopping_list"] = show_shopping_list
    actions["remove_shopping_item"] = remove_shopping_item
    actions["clear_shopping_list"] = clear_shopping_list
    actions["add_reminder"] = add_reminder
    actions["show_reminders_list"] = show_reminders_list
    actions["remove_reminder_item"] = remove_reminder_item
    actions["clear_reminders_list"] = clear_reminders_list
