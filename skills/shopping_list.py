"""
Shopping list + untimed reminders.

Actions added:
  add_reminder, <what>            a reminder that names NO time: goes on the
                                  to-do list (queue_task). Says where it went.
                                  If the text is about buying something it goes
                                  on the shopping list instead.
  add_shopping_item, <a, b, c>    add items to the shopping list
  show_shopping_list              read the list back
  remove_shopping_item, <item>    take one item off
  clear_shopping_list             empty the list (a .bak copy is kept)

"Remind me to buy milk" (no time) lands on the shopping list; any other
reminder with no time lands on the to-do list; a reminder WITH a time stays a
timer / schedule. The grammar lives in core/untimed_reminders.py; the monolith
routes whole-utterance requests here without the model, and the prompt teaches
the model the same actions for the phrasings the grammar misses.

Every reply says where the item went, and a failed save says so instead of
claiming success. State: data/shopping_list.json (core.paths, so a staging
process or a test never touches the live file).
"""
from __future__ import annotations

from core import untimed_reminders as _ur

_TODO_TAG = "[reminder] "

# The replies are finished sentences that say where the item went: spoken
# verbatim (bobert_companion._collect_skill_speak_sets), never re-worded.
SPEAK_VERBATIM_ACTIONS = ("add_reminder", "add_shopping_item",
                          "show_shopping_list", "remove_shopping_item",
                          "clear_shopping_list")


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
        rest to the to-do list - never to a timer."""
        task = (args or "").strip(" ,.")
        if not task:
            return ("What should I remind you about, sir? Say 'remind me to "
                    "call the dentist', or give me a time and I'll set a "
                    "timer.")
        items = _ur.shopping_items(task)
        if items:
            return add_shopping_item(", ".join(items))
        queue = actions.get("queue_task")
        if not callable(queue):
            return ("I can't reach the to-do list just now, sir - nothing was "
                    "saved.")
        try:
            result = queue(_TODO_TAG + task)
        except Exception as e:
            return (f"I couldn't add that to your to-do list, sir "
                    f"({type(e).__name__}) - nothing was saved.")
        if isinstance(result, str) and result.lower().startswith("format:"):
            return "I couldn't add that to your to-do list, sir."
        return (f"Added to your to-do list, sir: {task}. You gave me no "
                f"time, so I won't alert you.")

    actions["add_shopping_item"] = add_shopping_item
    actions["show_shopping_list"] = show_shopping_list
    actions["remove_shopping_item"] = remove_shopping_item
    actions["clear_shopping_list"] = clear_shopping_list
    actions["add_reminder"] = add_reminder
