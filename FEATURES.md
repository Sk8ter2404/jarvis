# JARVIS Feature Reference

## Overview

JARVIS (internal codename "Bobert") is a Python voice-controlled desktop AI assistant for Windows that styles itself after the Iron Man JARVIS — dry, British, composed. It is a single long-running script (`C:\JARVIS\bobert_companion.py`, ~15,000 lines) that listens via microphone, transcribes with Whisper, routes the transcript through Claude as the language model, and lets Claude emit `[ACTION: name, arg]` tokens that map to ~150 registered handler functions. Around the core sits a skills system (`C:\JARVIS\skills\`, 78 modules), a HUD subsystem (`C:\JARVIS\hud\`, 10 overlay subprocesses), and several proactive background daemons (printer monitor, credit monitor, briefings, wellness nudges, banter, anticipation engine).

## How to invoke

There is no push-to-talk hotkey by default — JARVIS listens continuously. Speak normally and Claude maps your phrasing to one of the registered ACTION names; you can also use the **wake/sleep phrases** below to put it into standby. A system-tray applet (`tray.py`) shows live status and a grouped right-click menu — common toggles plus five submenus (Power tools / AI / Memory / Diagnostics / Settings), the last opening the standalone **Settings GUI** (`tools/settings_window.py`).

- **Wake phrases**: `jarvis`, `hey jarvis`, `wake up`, `start listening`, `i need you`, `come back`, `resume listening`, `wake`
- **Sleep phrases**: `stop listening`, `go to sleep`, `sleep mode`, `stand by`, `go on standby`, `be quiet`, `mute yourself`, `take a break`, `go idle`, `pause listening`
- **Standby (work-mode)**: `wake work mode`, `work mode`, `standby mode`, `enter standby`, `go to standby`

## Capabilities by category

---

### Category 1: PC control & web navigation

- **Open a URL or app** — opens any website or launches a Windows app by name.
  - "open YouTube", "launch Bambu Studio", "open notepad", "fire up Chrome"
  - Actions: `open_url`, `launch_app`
- **Web / YouTube search** — Google or YouTube search in default browser.
  - "search for Bambu nozzle replacement", "google how to flash ESP32", "YouTube TIG welding basics"
  - Actions: `web_search` / `search`, `youtube`
- **Screenshot** — full-virtual-screen capture saved to `screenshots/`.
  - "take a screenshot", "grab a screenshot of everything", "snapshot the screen"
  - Action: `screenshot`
- **Get current time**
  - "what time is it", "what's the time"
  - Action: `get_time`
- **Shell command execution** — runs PowerShell, captures stdout/stderr, hard-blocks `format`, `shutdown`, `rm -rf`, `taskkill python`; pushes back on `rm -rf`, `git reset --hard`, `git clean`, etc.
  - "run git status", "show me the running Python processes", "what does `pip list` say"
  - Action: `run_shell`

### Category 2: Window & multi-monitor management

- **List / focus / minimize / close windows**
  - "minimize Chrome", "close the Bambu Studio window", "bring VS Code to the front", "list every open window"
  - Actions: `list_windows`, `focus_window`, `minimize_window`, `close_window`
- **Open app on a specific monitor / move existing window** — uses win32 to set position directly, no fragile hotkeys.
  - "open YouTube on the right monitor", "put Chrome on the left monitor", "move Bambu Studio to the top monitor"
  - Actions: `open_on_monitor`, `move_window_to_monitor`
- **UI automation primitives** — click by coordinates OR element description (vision), type text, press keys, hotkeys, scroll.
  - "click the Save button", "type my email", "press enter", "send Ctrl+L", "scroll down five"
  - Actions: `click`, `type`, `press`, `hotkey`, `scroll`

### Category 3: Screen vision & webcam awareness

- **Ask about the screen** — captures all monitors by default and asks a vision-Claude question across them.
  - "what's on screen", "what error is Bambu Studio showing", "see screen what tab am I on"
  - Action: `see_screen` (prefix `monitor:left|` etc. to target one)
- **Recall the last vision snapshot** — re-uses cached image from the last 5 minutes (near-instant, preserves dismissed errors).
  - "what did that error say again", "what was on the second tab", "remind me what we just looked at"
  - Actions: `recall_screen`, `last_screen`, `previous_screen`, `screen_history`
- **Locate a UI element by description** — for downstream clicking on the primary monitor.
  - "find the Save button on screen", "find the search box"
  - Action: `find_on_screen`
- **Webcam awareness** — JARVIS has cameras pointed at the desk and knows where you're looking. System prompt forbids "I have no cameras."
  - "which camera can see me", "what monitor am I looking at", "what do I look like right now", "have I been at the desk"
  - Actions: `where_is_user`, `see_user`, `which_monitor`, `gaze_status`, `gaze_stats`, `face_track_status`
  - Skill: `skills/face_tracker.py`

### Category 4: Music & media

- **iTunes COM library playback** — for owned songs.
  - "play Earth Song", "play artist:Michael Jackson", "play album:Thriller", "play library:Bohemian Rhapsody" (force iTunes)
  - Actions: `play_music`, `pause_music`, `resume_music`, `next_song`, `previous_song`, `now_playing`
- **Streaming services** — opens the service, clicks first result, clicks play in one action.
  - "play Stranger Things on Netflix", "put on The Bear on Hulu", "play Bohemian Rhapsody on Spotify", "Succession on Max"
  - Actions: `netflix`, `prime_video`, `disney_plus`, `hulu`, `max`, `spotify`, `apple_music`, `youtube_play`, `play_streaming`
- **Media keys** — go to whatever app holds media focus; use these when playing in a browser.
  - "skip this song", "pause", "play"
  - Actions: `media_next`, `media_prev`, `media_playpause`
- **Volume control** — system-wide.
  - "turn it down", "volume up", "mute"
  - Actions: `volume_up`, `volume_down`, `volume_mute`
- **Auto-routing note**: if Apple Music is open in a browser tab/PWA, iTunes COM actions auto-route to the browser. Use `library:` prefix on `play_music` to force iTunes.

### Category 5: Apple Music intelligence (taste-aware music)

- **Play something you haven't heard in a while** — uses iTunes `PlayedDate`.
  - "play something I haven't heard in months", "put on a track I haven't played in weeks"
  - Action: `play_unheard` (optional integer days, default 14)
- **Vibe-based playback** — picks dominant artist for a day/time slot.
  - "play my Friday night vibe", "put on something for tonight", "play my Sunday morning vibe"
  - Action: `play_vibe`
- **Taste-rejection skip** — records skip to session log, advances.
  - "skip, I'm not feeling this one", "no, something else", "not this one"
  - Action: `skip_track`
- **Listening history & taste summary**
  - "what have I been listening to lately", "what's my music taste", "music history"
  - Actions: `music_history`, `music_taste`, `music_aggregate`
- Skill: `skills/apple_music_intel.py`

### Category 6: Bambu H2D 3D-printer integration

- **Print status / details**
  - "check the print", "how is the print", "what's the printer doing"
  - Actions: `check_print`, `how_is_the_print`, `print_details`
  - Skill: `skills/bambu_monitor.py`
- **Pause / resume the active print**
  - "pause the print", "hold the printer", "resume the print", "continue printing"
  - Actions: `pause_print`, `resume_print`
- **First-time setup wizard** — voice-driven LAN discovery via SSDP, walks through reading the access code, hot-restarts poller without JARVIS restart.
  - "set up the printer", "configure the printer", "first time printer setup"
  - Actions: `setup_printer`, `configure_printer`, `bambu_setup`, `setup_bambu`, `first_time_printer_setup`
- **H2D corner overlay** — pinned ~280×140 always-on-top widget on top monitor.
  - "show the printer overlay", "bring up the print widget", "hide the printer overlay", "bambu overlay status"
  - Actions: `bambu_overlay_on`, `bambu_overlay_off`, `bambu_overlay_toggle`, `bambu_overlay_status`
- **Proactive announcements** — print start/complete/fail, layer-1 adhesion, 10/25/50/75/95 % milestones, AMS faults, filament runout, "your part is ready, sir." Gated by focus mode and rate-limited.

### Category 7: Briefings & daily intelligence

- **Morning briefing** — auto 06:00–11:00 once per day. Weather + tasks + late-night remark.
  - "morning briefing"
  - Action: `morning_briefing`
- **Morning handoff** — bigger chained version: weather → calendar/unread mail → Teams unread (VIP-aware) → overnight print → news → "anything else?"
  - "morning handoff", "what's on for today"
  - Actions: `morning_handoff`, `predictive_morning_setup`, `setup_workspace`, `workspace_setup` (opens Chrome+Apple Music, Teams, optionally Bambu Studio; drops master volume to ~30%)
- **Daily briefing** — 08:00 auto. Weather + first meeting + active print.
  - "daily briefing"
  - Action: `daily_briefing`
- **Evening briefing** — 22:00 auto. Interactions today + tasks shipped + active print + tomorrow's weather + tomorrow's first meeting + one dry observation.
  - "evening briefing", "evening recap"
  - Action: `evening_briefing`
- **Daily recap** — 22:30 auto end-of-day summary, ends with "Shall I queue the same morning briefing for tomorrow?"
  - "recap my day", "how did today go", "end of day summary"
  - Action: `daily_recap`
- **Weather forecast (hourly + umbrella alert)** — warns of rain transitions ~2h out.
  - "weather forecast", "do I need an umbrella", "what's the weather looking like later"
  - Actions: `weather_briefing`, `weather_forecast`
- **News briefing** — RSS headlines, optionally LLM-summarised.
  - "news briefing", "headlines", "what's in the news"
  - Action: `news_briefing`

### Category 8: Memory, recall & task queue

- **Session memory recall** — query prior sessions; resolves "yesterday" / "last night" / weekday names.
  - "what did we do yesterday", "remind me what happened this morning", "what did we work on Tuesday"
  - Action: `session_memory_recall`
- **Session resume** — pick up the unfinished thread from most recent prior session. Auto-fires once at startup if last session ended within 18h.
  - "where did we leave off", "pick up where we left off", "where were we", "resume", "continue from earlier"
  - Action: `session_resume`
- **Task queue → Claude Code handoff** — adds work items to `jarvis_todo.md`.
  - "add to the to-do list", "queue this for Claude Code", "put it on the doorless" (Whisper misheard 'to-do list')
  - Actions: `queue_task`, `show_tasks`, `clear_tasks` (requires confirmation)
- **Dossier — "pull up the file on X"** — aggregates memory facts + queued tasks + recent log mentions + DuckDuckGo abstract, slides a card titled "DOSSIER — X" onto the top monitor.
  - "pull up the file on a contact", "what do you have on Apple Music", "dossier on Bambu", "tell me what you know about X"
  - Actions: `dossier`, `pull_up_file`, `pull_up_dossier`, `file_on`, `dossier_on`, `what_do_you_have_on`, `whats_on_file`

### Category 9: HUD overlays

- **Arc-reactor corner HUD** (`hud/jarvis_hud.py`) — animated reactor ring. State-driven color/pulse (idle cyan, listening white, thinking blue spin, speaking amplitude-modulated, standby violet, alert red). Right-click for hide/resize/reset; Ctrl+wheel resize; drag to move; double-click reset.
  - "hide the HUD", "show the HUD", "toggle HUD"
  - Actions: `hide_hud`, `show_hud`, `toggle_hud`
- **Fullscreen holographic overlay** (`hud/jarvis_holo.py`) — big cinematic reactor on top monitor.
  - "show the holographic overlay", "holo on", "hud on", "hide holo", "dismiss holo"
  - Actions: `show_holographic_overlay`/`show_holo`/`hud_on`/`holographic_on`, `hide_holographic_overlay`/`hide_holo`/`hud_off`/`dismiss_holo`/`holographic_off`, `toggle_holographic_overlay`/`toggle_holo`, `holographic_status`
- **Arc-reactor workshop canvas** (`hud/holo_workshop_canvas.py`) — small rotating-blade reactor in top monitor corner. Auto-shows while thinking/speaking.
  - "show the arc reactor", "hide the arc reactor", "pulse the reactor", "arc reactor"
  - Actions: `arc_reactor`, `arc_reactor_on`, `arc_reactor_off`, `arc_reactor_pulse`, `holo_workshop_canvas`, `holo_workshop`, `workshop_canvas`
- **Workshop HUD** (`hud/workshop_hud.py`) — slim top-right corner panel: arc-reactor "power %" from CPU/RAM headroom, CPU/RAM bars.
  - "show the workshop HUD", "hide workshop HUD", "workshop HUD status"
  - Actions: `workshop_hud`, `workshop_hud_on`, `workshop_hud_off`, `workshop_hud_toggle`, `workshop_hud_status`, `hide_workshop_hud`, `show_workshop_hud`
- **Reticle overlay** (`hud/jarvis_reticle.py`) — translucent target reticle at click/type coordinates during UI automation. No voice commands.
- **System tray applet** (`tray.py`) — arc-reactor icon tinted by listen state (green awake / gray standby / red muted) + speaking halo + upgrade-queue badge + Bambu print mark. Grouped right-click menu: common toggles + five submenus (Power tools / AI / Memory / Diagnostics / Settings) + About; the Settings submenu opens the standalone **Settings GUI** (`tools/settings_window.py`).

### Category 10: System monitoring & status

- **Quick health check** — CPU, RAM, top processes, disk, network rates. Background alert if CPU or RAM crosses threshold.
  - "check system", "how's the system"
  - Action: `check_system`
- **System pulse** — wider read: CPU, RAM, GPU temp, disk, network, battery, uptime, window count, Bambu %, Anthropic credits. Proactive every 15 min if anything abnormal.
  - "system pulse", "status report"
  - Actions: `system_pulse`, `status_report`
- **Status panel ("suit diagnostics")** — full multi-line readout (CPU, RAM, GPU temp, network latency, credit balance, print %, focused app, Apple Music track).
  - "status panel", "system status", "suit diagnostics"
  - Actions: `status_panel`, `system_status`, `suit_diagnostics`
- **Screen-watch wellness nudge** — 25 min of same-window + idle input triggers a 5-min stretch offer.
  - "screen watch status"
  - Action: `screen_watch_status`
- **Presence wellness** — after 90 min continuous desk presence, volunteers a hydration / eye-break line. 60-min cooldown.
  - "wellness status"
  - Action: `wellness_status`

### Category 11: Focus / workshop / night-owl modes

- **Do-not-disturb focus mode** — mutes Windows toasts, sets Teams DND, pauses banter/wellness/teams-nudge, shortens LLM replies. Auto-restores at duration.
  - "focus mode for 90 minutes", "do not disturb for an hour", "DND for 2 hours", "go heads-down", "end focus mode", "cancel DND"
  - Actions: `focus_mode`, `end_focus_mode`, `focus_mode_status`
- **Workshop mode (auto-engaged)** — fires when Bambu Studio / Fusion 360 / SolidWorks / FreeCAD / OnShape / Blender / OpenSCAD / Orca / Prusa / Cura appears. TTS to 70% volume, single-sentence prompt addendum, announces print status if mid-flight.
  - "workshop status" (manual query)
  - Action: `workshop_status`
- **Night-owl mode** — auto 23:00–06:00. TTS softer + quieter, holographic overlay dimmed to 40%, non-critical nudges suppressed. Critical alerts (print fail, VIP call, timer) still fire. Auto-disengages 06:00 OR "good morning."
  - "night owl on", "night owl off", "good morning", "night owl status"
  - Actions: `night_owl_on`/`night_owl_mode`/`enable_night_owl`, `night_owl_off`/`end_night_owl`/`disable_night_owl`, `good_morning`, `night_owl_status`
- **Suit-up sequence** — cinematic 6–8s warm-restart boot. Arc-reactor spin-up, sequential system-check TTS lines ("Diagnostics: nominal." "Network: online." "Welcome back, sir. Systems are yours."). Fires once per day on first warm restart.
  - "suit up", "run the suit up sequence"
  - Actions: `suit_up`, `suit_up_sequence`

### Category 12: Microsoft Teams integration

- **Teams call screener** — polls window titles for VIP-caller patterns (e.g. "a VIP contact is calling you"). Speaks caller's name and arms answer/decline.
  - "yes, patch him through", "answer it", "take it", "no, send my regards", "decline the call"
  - Actions: `screen_teams_calls`, `answer_call`, `decline_call`, `vip_priority_handler`
- **Teams unread nudge** — vision-based: full-screen capture every N seconds, Claude vision looks for unread badge, queues a JARVIS nudge if not snoozed.
  - "check Teams", "any Teams messages"
  - Action: `check_teams`

### Category 13: Timers & reminders

- **Set / list / cancel timers** — fires write reminder to `pending_speech.json`.
  - "set a 5 minute timer to check the oven", "set a timer for 20 minutes", "list timers", "cancel timer 3"
  - Actions: `set_timer`, `list_timers`, `cancel_timer`

### Category 14: Self-management & learning

- **Claude credits** — opens Anthropic billing in hidden Chrome, reads balance via vision, closes window. Background monitor hourly; speaks if balance below $5.
  - "check my credits", "how many credits do I have", "what's my Claude balance", "am I running low on credits"
  - Action: `check_credits`
- **Restart yourself**
  - "restart", "reboot", "restart yourself", "relaunch", "start over"
  - Action: `restart`
- **Upgrade pipeline** — hands task queue to Claude Code: kills JARVIS, spawns Claude Code CLI to implement pending tasks, relaunches when done.
  - "upgrade yourself", "restart and upgrade", "apply the changes"
  - Action: `upgrade`
- **Overnight upgrade engine** — autonomous improvement loop. Generates new ideas, appends as tasks, runs upgrade pipeline, loops while idle. Auto-puts JARVIS in silent standby.
  - "start overnight upgrade", "improve yourself while I sleep", "run overnight"
  - **Goodnight phrasings ALWAYS fire this**: "goodnight", "good night", "I'm going to bed", "heading to bed", "going to sleep", "I'm off to bed", "time to sleep"
  - Action: `start_overnight_upgrade`
- **Pattern learning** — nightly aggregator over `data/usage_patterns.jsonl`; produces broad-window and precise-clock predictions. Feeds anticipation engine.
  - "pattern predictions", "what are my patterns", "pattern stats", "force a pattern offer", "pattern aggregate"
  - Actions: `pattern_predictions`, `pattern_offer_now`, `pattern_aggregate`, `pattern_stats`
- **Anticipation engine** — proactive 60-sec poll; volunteers one in-character line when pattern + dwell + time-of-day + gaze line up. 20-min cooldown, 35% fire probability, silent in calls / sleep / standby.
  - "anticipation status"
  - Action: `anticipation_status`
- **Banter engine** — dry zingers when JARVIS notices a tell (repeat question, 5x+ same target opened today, >40 tabs, "play music while music already playing"). 30-min cooldown, 50% fire probability.
  - "banter status"
  - Action: `banter_status`
- **Skills system** — self-extension.
  - "list skills", "what skills do you have"
  - Action: `list_skills`
  - "create a skill called X that does Y" → writes module to `pending_skills/`, you review and move to `skills/` to activate.
  - Action: `create_skill, <name> | <description>`

### Category 15: REPO Robot project tracker

- **Robot status / blockers / next step** — reads `data/repo_robot_state.json` + `jarvis_todo.md` + recent logs.
  - "robot status", "what's blocking the robot", "what's the next step on the robot"
  - Actions: `robot_status`, `robot_blocker`, `next_robot_step`

### Category 16: Ambient audio detection

- **Standby audio detector** — spectral classifier on raw mic chunks; sets internal "music currently playing" state so a wake-word buried in a lyric won't flip JARVIS out of standby.
  - "audio music status"
  - Action: `audio_music_status`

---

## Special voice patterns / proactive behaviors

- **Time-aware wake greetings** — JARVIS picks context-aware response from time-of-day, recent wake frequency, gaze, print state, and tone:
  - 01:00–05:00 with ≥3 wakes in 10 min → "Still up, sir?"
  - 05:00–12:00 first wake of the day → "Good morning, sir."
  - Bambu actively printing → "At your service — the print is at 47%, by the way."
  - Out of view on every camera → quieter "Yes, sir?"
  - Otherwise one of 12 phrases tagged formal / terse / playful / soft / general.
- **Barge-in** — on a headset, TTS is killed mid-sentence if mic detects sustained speech. Silent on speakers to avoid feedback.
- **Goodnight = overnight upgrade** — any bedtime phrasing fires `start_overnight_upgrade` and silences JARVIS until morning.
- **Workshop auto-engagement** — opening any CAD/slicer app drops TTS volume 30%, shortens replies, auto-engages focus mode for an hour (auto-releases when app closes).
- **Promises ("I'll let you know when…")** — skills register deferred announcement via `skill_utils["make_promise"]` (e.g. "tell me when the print finishes", "let me know when the bed cools").

---

## High-risk actions (require confirmation)

Hard-blocked until you say "yes" / "confirm" / "do it" — matched in action name + arg: **purchase, buy, pay, checkout, delete, format, transfer**.

Additional pushback / refusal layers:

- **`clear_tasks`** — wipes whole task queue; held for confirmation.
- **`run_shell` with destructive patterns** — `rm -rf`, `git reset --hard`, `git clean -fd/-fx`, `git push --force`, `del /s|/q|/f`, `rmdir /s`, `drop table`, `drop database`, `truncate table`, `remove-item -recurse` — held for "yes".
- **`run_shell` with hard blocklist** — `format`, `shutdown`, `taskkill python` — REFUSED outright (cannot be confirmed away).
- **Self-preservation** — forbidden from killing own host:
  - `close_window` where title contains `powershell`, `python`, `terminal`, or `bobert`
  - `alt+f4` while PowerShell / python is focused
  - clicking "close button on PowerShell" via vision
  - Refused with explanation, no retry.
- **Sketchy URLs in `open_url`** — bare-IP HTTP, free-phishing TLDs (.tk/.ml/.cf/.gq/.top/.xyz/.click/.country/.zip/.mov), `.onion`, tunnel hosts (ngrok / trycloudflare / loca.lt / serveo), link shorteners (bit.ly / tinyurl / goo.gl / t.co / is.gd / buff.ly / ow.ly / rebrand.ly) — pushback before opening. Local/LAN URLs whitelisted.
- **Bulk window-close with unsaved work** — pushes back when titles look unsaved (`*` / `●` / `•`, "Untitled", "(modified)") or live-edit apps (VS Code, Cursor, Google Docs/Sheets).
- **Skill code is forbidden** from including final purchase / payment confirmation steps (must stop one step BEFORE money-spending click).

---

## Key files

- `C:\JARVIS\bobert_companion.py` — main loop, prompt, ACTIONS dict, all core handlers
- `C:\JARVIS\skills\` — 78 skill modules
- `C:\JARVIS\hud\` — 10 overlay subprocesses driven by `hud_state.json`
- `C:\JARVIS\upgrade_jarvis.py` — voice-triggered Claude Code pipeline
- `C:\JARVIS\overnight_upgrade.py` — idle-watch autonomous improvement loop
- `C:\JARVIS\iron_man_boot.py` — boot-sequence sting + HUD animation + spoken greeting
- `C:\JARVIS\tray.py` — system-tray applet
- `C:\JARVIS\jarvis_todo.md` — task queue Claude Code consumes
- `C:\JARVIS\hud_state.json`, `hud_card_state.json`, `pending_speech.json` — inter-process state
