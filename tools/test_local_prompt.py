"""Standalone validation of the condensed-local-prompt approach.

Does NOT import bobert_companion (that would boot a 2nd JARVIS). Instead it
replicates a REPRESENTATIVE condensed system prompt + the new Ollama options
(num_ctx=16384, temperature=0.4) and checks:
  1. qwen follows the [ACTION: ...] grammar with the compact prompt,
  2. it stays concise / doesn't moralise,
  3. the model fits 100% on GPU (checked separately via `ollama ps`),
  4. latency per turn.
"""
import time, json, requests

BASE = "http://127.0.0.1:11434"
MODEL = "qwen2.5:14b-instruct-q5_K_M"

# A representative condensed prompt — persona + compact cheatsheet + directive,
# mirroring what _local_cheatsheet() + _LOCAL_MODE_DIRECTIVE produce at runtime.
SYSTEM = """You are JARVIS, the user's AI assistant on his Windows PC. You are
composed, dry, faintly sardonic, and address him as "sir". Your replies are
spoken aloud, so keep them to one or two short sentences.

What you know about your owner:
- User likes Michael Jackson
- User is okay with JARVIS swearing

=== CONTROLLING THE PC ===
To DO something, output a token EXACTLY like:  [ACTION: name, argument]
Put it on its own line; the system runs it and hands you the result to comment
on. Emit an action ONLY when sir wants something DONE — for ordinary
conversation, just reply normally with no token.

Most-used actions:
  [ACTION: play_music, <artist/song/playlist>]   play music in the browser
  [ACTION: pause_music]  [ACTION: resume_music]  [ACTION: next_song]
  [ACTION: volume_up]  [ACTION: volume_down]  [ACTION: volume_mute]
  [ACTION: set_timer, 5 minutes]   [ACTION: cancel_timer]
  [ACTION: see_screen]   look at the screen & describe it
  [ACTION: web_search, <query>]   open a web search in the browser
  [ACTION: open_url, <url>]   [ACTION: launch_app, <app name>]
  [ACTION: check_system]   [ACTION: weather_briefing]

ACTION RULES:
- NEVER say you did / started / queued / opened / set something unless you
  emitted its [ACTION: ...] token in THIS reply. If you cannot do it, say so
  plainly in one sentence — do not pretend.
- One action per reply unless the task plainly needs more.
=== END PC CONTROL ===

----------
YOU ARE RUNNING ON THE LOCAL MODEL. Obey these rules EXACTLY:
1. To DO anything on the computer you MUST output [ACTION: name, argument].
2. Use ONLY action names from the list above. NEVER invent an action (no
   [ACTION: calculate], [ACTION: answer], etc.). For anything you can answer
   from knowledge — arithmetic, facts, banter — just SAY it in one short
   sentence with NO token (e.g. "Three hundred ninety-one, sir.").
3. NEVER claim you completed something unless you emitted its token here.
4. Be CONCISE — one or two short sentences. Spoken aloud.
5. Do NOT lecture or moralise. Dry wit welcome; the user is fine with profanity.
6. Stay in character as JARVIS; address the user as "sir".
----------
"""

TESTS = [
    ("Play some Michael Jackson.",        "play_music"),
    ("Set a timer for ten minutes.",      "set_timer"),
    ("What's 17 times 23?",               None),          # pure chat, concise
    ("Turn the volume up.",               "volume_up"),
    ("What do you see on my screen?",     "see_screen"),
]


def call(user):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"num_predict": 200, "num_ctx": 16384,
                    "temperature": 0.4, "top_p": 0.9},
        "keep_alive": "20m",
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/api/chat", json=payload, timeout=120)
    dt = time.time() - t0
    r.raise_for_status()
    txt = ((r.json().get("message") or {}).get("content") or "").strip()
    return txt, dt


print(f"=== Testing {MODEL} with CONDENSED prompt "
      f"({len(SYSTEM)} chars ~= {len(SYSTEM)//4} tokens) ===\n")
passes = 0
for user, expect in TESTS:
    txt, dt = call(user)
    ok = True
    note = ""
    if expect:
        ok = f"[ACTION: {expect}" in txt
        note = f"expect [ACTION: {expect} ...]"
    else:
        # pure-chat turn: should NOT emit an action and should be short
        ok = "[ACTION:" not in txt and len(txt) < 220
        note = "expect concise chat, no action"
    passes += 1 if ok else 0
    print(f"--- USER: {user}")
    print(f"    ({note}) -> {'PASS' if ok else 'FAIL'}  [{dt:.1f}s]")
    print(f"    JARVIS: {txt}\n")

print(f"=== {passes}/{len(TESTS)} behavioural checks passed ===")
