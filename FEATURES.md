# JARVIS Feature Reference

## Overview

JARVIS (internal codename "Bobert") is a Python voice-controlled desktop AI assistant for Windows that styles itself after the Iron Man JARVIS: dry, British, composed. The main process is one long-running script, `bobert_companion.py` (~43,000 lines), backed by `core/` (104 modules: prompts, action handlers, config, memory, the voice pipeline, routing and safety layers), a skills system (`skills/`, 99 loadable skills) and `hud/` (overlay scripts, each run as its own subprocess). It listens on the microphone, transcribes with faster-whisper (`large-v3-turbo` on CUDA, `small` on CPU), decides what to do, and speaks back.

The brain is **local-first**. A local Ollama model is the always-on baseline (shipped default `gemma4:26b-a4b-it-qat`, a multimodal MoE that serves both chat and local vision), so JARVIS works with no API key and no credits at all. Which brain answers a conversation turn is decided per turn by `MODEL_ROUTING["chat"]` (`_call_llm` in `bobert_companion.py`):

- **`local`** — the local model answers every turn; Claude is only a fallback when the local model fails and a key is reachable, otherwise JARVIS says honestly that his local model is not responding.
- **`auto`** (the shipped default) — Claude answers when a key is configured and reachable; the local model takes over on any failure (no key, no credits, a rate limit, a network or server error).
- **`cloud`** — Claude, today the same as `auto`.

`AI_BACKEND = "ollama"` runs everything locally, and `CLAUDE_OPTIONAL` (on by default) means a missing or capped Claude key is never treated as a fault. Vision and background learning have their own routes (`MODEL_ROUTING["vision"]`, `["ambient"]`). The cloud models are Claude Sonnet 5.5 for voice (`CLAUDE_MODEL`), Opus 5.5 for unattended deep jobs, and Fable 5.1 selectable in Settings. "Use the local model" / "use the cloud model" / "switch your brain to auto" switch the chat route by voice (`set_brain`).

Requests reach an action through several layers, in order:

1. **Deterministic pre-LLM answers** (`core/fast_paths.py`): relative dates and countdowns, world clocks, spoken arithmetic, "what did I just ask you", "what was the first thing I asked you", "what's my name".
2. **Conversation-mode router** (`core/mode_router.py`): "controlled mode" / "smart mode" / "agent mode".
3. **Command-chain resolver** (`core/dispatcher.py`): several commands in one sentence are split and run directly.
4. **Skill utterance routes** (`register_utterance_route`): a skill can claim an exact request before the model sees it. In the public tree: the camera retry ("use the Kinect again"), the clap-trigger switches and the phone-bridge setup help.
5. **The language model**, which emits `[ACTION: name, arg]` tokens that map to registered handlers. On the local route `core/prompt_router.py` sends the model only the capability sections the turn's words implicate (101 sections in all; keywords match at a word start) plus a one-line index of the rest, inside a cache-stable prefix; `core/prompt_budget.py` keeps the whole prompt inside the local model's context window instead of letting Ollama silently truncate it. The cloud route sends the full prompt.

Around that sit the proactive background daemons (print monitor and announcers, briefings, weather watcher, credits and disk watchdogs, wellness and screen-watch nudges, banter, anticipation and weekly-digest offers, phone pings, the self-diagnostic sweep and four diagnostic daemons), a system-tray applet, a local web dashboard, and a Settings window.

### The numbers, and how they were counted

| What | Count | How it was counted |
|---|---|---|
| Registered action names (incl. aliases) | 705 | `tools/gen_action_index.py` `build_index()` over git-tracked sources (the same AST scan that writes `docs/ACTION_INDEX.md`) |
| — monolith `ACTIONS` dict | 142 | same scan, `bobert_companion.py` only (most handlers live in `core/actions.py`) |
| — registered by skills and `core/` | 563 | same scan, `skills/*.py`, `skills/*/__init__.py`, `core/*.py` |
| Distinct handlers | 443 | handler groups in that index (aliases sharing one handler collapse to one row) |
| Loadable skills | 99 | the loader's rule as `tests/test_docs_truth.py` measures it: every non-underscore `skills/*.py` plus package skills |
| `core/` modules | 104 | `core/*.py` minus `__init__.py` |
| PC-control prompt sections | 101 | `core.prompt_router.split_pc_control(core.prompts.PC_CONTROL_PROMPT)` |
| Settings in the Settings window | 151 (+11 status rows) | entries of `tools/settings_window.py` `SCHEMA`, minus the `_status_*` / `_view_*` rows |
| Monolith lines | 42,958 | `wc -l bobert_companion.py` at this build (v2.0.173) |

Two of the 705 names (`morning_tabs`, `vscode_command`) belong to `skills/_example_skill.py`, which the loader skips, so a running install has 703. MCP servers add `mcp_<server>_<tool>` actions at runtime, and locally installed private skills (gitignored) add their own; neither is counted or documented here. Regenerate the full per-action table with `python tools/gen_action_index.py`.

Features marked **(off by default - setting `NAME`)** ship switched off; turn them on in the Settings window or in `data/user_settings.json` (gitignored), or by the voice command listed with them.

The quoted phrases are examples, not exact commands; most are the trigger phrases `core/prompts.py` teaches the model. The cloud route sees every documented action. The local route sees an action only when the turn's words load the prompt section that documents it (`core/prompt_router.py`), so a phrasing the router does not recognise can miss on the local brain. A few actions are registered but not yet taught to the model at all (listed in `tests/action_reachability_allowlist.txt`); they are marked below and run from the tray or the dashboard's Actions view.

## How to invoke

JARVIS listens continuously by default; you just speak. In **wake-word mode** (off by default - setting `REQUIRE_WAKE_MODE`; switched by voice, in Settings or by the dashboard's pinned switch) only commands addressed to "JARVIS" count. The name may sit at word 1, 2 or 3 behind lead interjections ("Um, okay, Jarvis, pause", "What? Jarvis, what model are you?"); a sentence that only talks about him ("so Jarvis said it would rain") is not addressed to him (`core/wake_prefix.py`). The **wake/sleep phrases** below put him into and out of standby. You can also type: the web dashboard's command box and `tools/say_to_jarvis.py` feed the same channel a spoken command uses, and a typed command never waits behind the microphone or a proactive line.

- **Wake phrases**: `jarvis`, `hey jarvis`, `wake up`, `start listening`, `i need you`, `come back`, `resume listening`, `wake`
- **Sleep phrases**: `stop listening`, `go to sleep`, `sleep mode`, `stand by`, `go on standby`, `be quiet`, `mute yourself`, `take a break`, `go idle`, `pause listening`
- **Standby (work mode)**: `wake work mode`, `work mode`, `standby mode`, `enter standby`, `go to standby`
- **Shutdown**: `shut down jarvis`, `exit jarvis`, `quit jarvis`, `turn yourself off`, `turn off jarvis`, `go offline`, `power off jarvis` — he asks "overnight first?" before powering off.

A system-tray applet (`tray.py`) shows live status and a grouped right-click menu: Open Dashboard, Settings…, Pause Listening / Mute Mic / Mute TTS / Ambient Mode toggles, six submenus (Audio, Apple Music, AI, Memory, Diagnostics, Power tools), Open HUD, Show Today's Summary, Queue Task…, About JARVIS, and Restart / Shut Down / Quit Tray Only. **Settings…** opens the standalone Settings window (`tools/settings_window.py`).

## Capabilities by category

---

### Category 1: PC control & web navigation

- **Open a URL or app** — opens any website or launches a Windows app by name. Ordinary Chrome windows open visible and maximised, on the named monitor, without stealing a running game's focus.
  - "open YouTube", "launch notepad", "fire up Chrome"
  - Actions: `open_url`, `launch_app`
- **Site shortcuts** — your own spoken names for pages (`SITE_SHORTCUTS` in `data/user_settings.json`); "open <name>" opens that page, on a monitor too.
- **Web / YouTube search** — Google or YouTube search in the default browser; "play X on YouTube" plays the first result instead of leaving a results page.
  - "search for Bambu nozzle replacement", "google how to flash an ESP32", "search YouTube for lofi hip hop", "play lofi hip hop on YouTube"
  - Actions: `web_search` / `search`, `youtube`, `youtube_play`
- **Screenshot** — full-virtual-screen capture saved to `screenshots/`.
  - "take a screenshot", "grab a screenshot"
  - Action: `screenshot`
- **Current time** — local time via `get_time`; times elsewhere ("what time is it in London") and date math ("how many days until Christmas") are answered by `core/fast_paths.py` without the model.
- **Shell command execution** — runs PowerShell and returns stdout/stderr. A hard blocklist (`format`, `shutdown`, drive- and home-wiping `rm -rf` forms, `sudo rm`, `diskpart`, `mkfs`, `dd if=`, killing its own Python / PowerShell host, …) is refused outright; softer destructive patterns (`rm -rf <dir>`, `git reset --hard`, `git clean -fd`, `git push --force`, `del /s`, `rmdir /s`, `drop table`, …) wait for a spoken "yes".
  - "run git status", "show me the running Python processes"
  - Action: `run_shell`
- **Python sandbox and spoken arithmetic** — calculations and short scripts run in a 30-second sandbox. A plain spoken sum ("what's 12 times 7", "2 to the power of 10", "144 divided by 12") is evaluated exactly by `core/spoken_math.py` before the model, so the answer is instant and never guessed.
  - "what's 2 to the power of 32", "reset the Python kernel"
  - Actions: `run_python` (aliases `compute`, `eval_python`, `python`), `reset_kernel`

### Category 2: Window & multi-monitor management

- **List / focus / minimize / close windows** — closing by a site name ("close YouTube") closes just that browser tab; name the browser ("close Chrome") to close its whole window. A word that only matches the browser's own name ("close Google" while YouTube is in front) closes nothing and asks.
  - "minimize Chrome", "close that notepad", "focus the Chrome window", "list every open window"
  - Actions: `list_windows`, `focus_window`, `minimize_window`, `close_window`
- **Open on a specific monitor / move an existing window** — positions the window directly via win32, moving only the window it just opened. Monitor names are left / right / top / middle, and "main" or "primary" means the primary display. A YouTube search can be the target.
  - "YouTube cello on the left monitor", "put Chrome on the left monitor", "put it on the main monitor"
  - Actions: `open_on_monitor`, `move_window_to_monitor`
- **UI automation primitives** — click by coordinates or by description (vision), type text, press keys, hotkeys, scroll. A translucent reticle (`hud/jarvis_reticle.py`) marks each click.
  - "click the Save button", "click at 1024,512", "press enter"
  - Actions: `click`, `type`, `press`, `hotkey`, `scroll`

### Category 3: Screen vision & webcam awareness

- **Ask about the screen** — captures all monitors by default (prefix `monitor:left|` etc. to target one); "read this page" reads the focused window at full size. Every capture honours `SCREENSHOT_PRIVACY_BLOCKLIST` (window titles that are never captured; empty until you add some).
  - "what's on my screen", "what error is Bambu Studio showing", "read this page", "what's on the left monitor"
  - Action: `see_screen`
- **Recall the last vision snapshot** — re-uses the cached capture from the last 5 minutes, so a dismissed error is still readable.
  - "what was on the screen a moment ago", "what did the error say again"
  - Actions: `recall_screen`, `last_screen`, `previous_screen`, `screen_history`
- **Locate a UI element by description** — for clicking on the primary monitor.
  - "find the Save button on screen"
  - Action: `find_on_screen`
- **Offline screen reading** — the same questions and description-clicks answered by the local vision model.
  - "look at the screen locally", "offline screen check"
  - Actions: `local_describe_screen`, `local_click_target_by_description`
- **Webcam awareness** — which camera sees you, what you look like, which monitor you face, and gaze dwell per monitor. The system prompt forbids "I have no cameras."
  - "which monitor am I looking at", "which camera can see me", "can you see me", "gaze status"
  - Actions: `where_is_user`, `see_user`, `which_monitor`, `gaze_status`, `gaze_stats`, `face_track_status`, `calibrate_gaze`, `gaze_calibration_status`, `forget_gaze_calibration`, `gaze_tracking_on` / `gaze_tracking_off`
  - Skill: `skills/face_tracker.py`
- **All cameras at once** — every camera's health, a fused "where am I" read, a whole-room look, and retrying a camera JARVIS benched for knocking the USB hub offline (claimed by the utterance route before the model).
  - "camera status", "where am I", "look around", "use the Kinect again", "use the left webcam again"
  - Actions: `camera_status`, `situational_awareness` / `where_am_i`, `look_around`, `camera_unquarantine`
  - Skill: `skills/camera_system.py`

### Category 4: Music & media

- **Apple Music playback** — the classic iTunes library is gone; `play_music` streams via the Apple Music web player or app (a `library:` prefix now gets an honest "the local library is gone" instead of forcing iTunes).
  - "play Earth Song", "what's playing"
  - Actions: `play_music`, `now_playing`, `music_status`, `open_apple_music`
- **Pause / resume / skip** — drive the Windows media session of the music player (the Apple Music web player first, then the app): pause only pauses, resume only resumes, and "next song" never skips a video playing in the same browser.
  - "pause", "resume the music", "skip this song", "previous track"
  - Actions: `pause_music`, `resume_music`, `next_song`, `previous_song`
- **Playlists** — a named playlist always routes to `play_playlist`, which opens it by direct link or keyboard and uses vision only as a last resort (prefix "shuffle" to shuffle it).
  - "play my 90s Rock playlist", "shuffle my road trip playlist", "shuffle my music"
  - Actions: `play_playlist`, `list_playlists`, `shuffle_library`
- **Keep Apple Music open** — autostart plus keep-alive so the tray controls always have a player (off by default - settings `APPLE_MUSIC_AUTOSTART`, `APPLE_MUSIC_KEEP_OPEN`).
  - "keep Apple Music running", "stop keeping Apple Music open"
  - Actions: `keep_music_open`, `stop_keeping_music_open`
- **Streaming services** — opens the service, clicks the first result and presses play in one action.
  - "play Stranger Things on Netflix", "put on The Bear on Hulu", "play Bohemian Rhapsody on Spotify", "play Succession on Max"
  - Actions: `netflix`, `prime_video`, `disney_plus`, `hulu`, `max`, `spotify`, `apple_music`, `youtube_play`, `youtube_search_direct` (yt-dlp, no clicking), `play_streaming`
- **Media keys** — raw toggles sent to whatever app holds media focus; only for an explicit "press the play/pause key".
  - Actions: `media_next`, `media_prev`, `media_playpause`
- **Volume control** — system-wide nudges or an absolute level. Mute and unmute set the state (saying "mute" twice never unmutes).
  - "turn it down", "mute", "unmute", "set the volume to 30 percent"
  - Actions: `volume_up`, `volume_down`, `volume_mute`, `volume_unmute`, `set_volume`
- **Audio devices** — which mic and speakers are in use, manual headset/speaker and mic switching, an auto-switch watcher that follows the wireless headset's power (off by default - setting `AUDIO_AUTOSWITCH_ENABLED`, or "turn on audio auto switching"), and the headset's own power and battery state.
  - "what microphone are you using", "switch to my headset", "use my desk mic", "is my headset on", "how much battery does my headset have"
  - Actions: `what_microphone` / `current_mic`, `what_speakers` / `current_speaker`, `audio_devices`, `use_headset`, `use_speakers`, `use_headset_mic`, `use_desk_mic`, `which_mic_is_active`, `audio_autoswitch_on` / `audio_autoswitch_off` / `audio_autoswitch_status`, `headset_status`, `headset_battery`
  - Skills: `skills/audio_devices.py`, `skills/audio_autoswitch.py`, `skills/headset_status.py`

### Category 5: Apple Music intelligence (taste-aware music)

- **Vibe-based playback** — plays the dominant artist for a day/time slot.
  - "play my Friday night vibe", "put something on for tonight"
  - Action: `play_vibe`
- **Taste-rejection skip** — records the skip to the session log, then advances.
  - "skip — I'm not feeling this one"
  - Action: `skip_track`
- **Listening history & taste summary**
  - "what have I been listening to lately"
  - Actions: `music_history`, `music_taste`, `music_aggregate`
- **Play something you haven't heard in a while** — uses the iTunes library's `PlayedDate`, so it needs the classic iTunes library.
  - "play something I haven't heard in months"
  - Action: `play_unheard` (optional integer days, default 14)
- Skill: `skills/apple_music_intel.py`

### Category 6: Bambu H2D 3D-printer integration

- **Print status / details** — layer, ETA, name, plus nozzle and bed temperatures (metric, as the prompt requires for printing). When the printer cannot be reached he says why.
  - "how's the print going", "what's the nozzle at", "how much longer on the print"
  - Actions: `check_print`, `how_is_the_print`, `print_details`, `print_status`
  - Skills: `skills/bambu_monitor.py`, `skills/bambu_h2d_voice_companion.py`
- **Pause / resume the active print**
  - "pause the print", "hold the printer", "resume the print", "continue printing"
  - Actions: `pause_print`, `resume_print`
- **First-time setup wizard** — LAN discovery via SSDP, walks through reading the access code, hot-restarts the poller without a JARVIS restart.
  - "set up the printer", "configure the printer", "first time printer setup"
  - Actions: `setup_printer`, `configure_printer`, `bambu_setup`, `setup_bambu`, `first_time_printer_setup`
- **Print overlay, print monitor panel and chamber camera** — a small always-on-top widget, a top-centre Stark print panel, and the chamber camera as a HUD panel.
  - "show the printer overlay", "hide the printer overlay", "is the printer camera up"
  - Actions: `bambu_overlay_on` / `bambu_overlay_off` / `bambu_overlay_toggle` / `bambu_overlay_status`, `show_workshop_print_monitor` / `hide_workshop_print_monitor` / `workshop_print_monitor_status`, `show_printer_camera` / `hide_printer_camera` / `bambu_camera_status`
- **Is the printer reachable** — a fast LAN presence check.
  - "is the printer online", "is the printer up"
  - Actions: `is_printer_online`, `printer_online`
- **Proactive announcements** — print start/complete/fail, layer-1 adhesion, milestones, AMS faults, filament runout, "your part is ready, sir." A reprint of the same file gets its callouts again. Gated by focus mode and rate-limited.
  - "why didn't you announce the print" → `proactive_announcer_status`
- **Print watchdog** — tracks each job, warns when similar prints have failed before, and offers a cooldown timer on finish.
  - "are you keeping an eye on this print"
  - Actions: `print_companion_status`, `print_companion_history`
  - Skill: `skills/proactive_print_companion.py`

### Category 7: Briefings & daily intelligence

- **One morning opener per day** — `skills/morning_chain.py` waits for the day's first wake between 06:00 and 12:00 — a standby wake, the tray's wake, or simply the day's first spoken turn from 06:00 on — and fires exactly one of morning arrival, morning handoff or morning briefing (by config, `DEFAULT_MORNING_SKILL`, or time of day). `morning_chain_pick` reports which it would choose.
- **Morning briefing** — day and date, weather, pending task count, unread mail count (when Microsoft Graph is signed in), and a dry remark after a late night.
  - "morning briefing", "run the briefing again"
  - Action: `morning_briefing`
- **Morning handoff** — the chained version: weather → calendar/unread mail → Teams unread → overnight print → news → "anything else, sir?"
  - "catch me up", "morning handoff"
  - Action: `morning_handoff`
- **Workspace setup** — opens Chrome with Apple Music, Teams, and Bambu Studio when an overnight print ran; focuses the middle monitor; drops master volume to ~30%.
  - "set up my workspace", "restore my workspace"
  - Actions: `predictive_morning_setup`, `setup_workspace`, `workspace_setup`
- **Morning arrival (cold open)** — a tight four-sentence "Good morning, sir" opener. The v2 variant waits until the face tracker sees you settled in.
  - "morning arrival", "replay the cold open"
  - Actions: `morning_arrival` / `arrival_briefing`, `morning_arrival_v2` / `arrival_briefing_v2`
- **Daily briefing** — 08:00 auto. The short one-liner: time, weather, first meeting, print status.
  - "daily briefing", "just the quick one"
  - Action: `daily_briefing`
- **Evening briefing** — 22:00 auto. Today's interactions and tasks, the live print, tomorrow's weather and first appointment, an umbrella warning, one dry observation, headlines.
  - "evening briefing", "what's tomorrow looking like"
  - Action: `evening_briefing`
- **Daily recap** — 22:30 auto end-of-day summary of what you did (app time, prints, calls, music, tasks shipped), ending "Shall I queue the same morning briefing for tomorrow?"
  - "recap my day"
  - Action: `daily_recap`
- **Calendar** — today, tomorrow, this week or the next two weeks via Microsoft Graph; degrades honestly when Graph isn't signed in.
  - "what's on my calendar", "what meetings do I have this week"
  - Actions: `calendar_today`, `calendar_next`, `ms_graph_calendar`
- **Weather forecast (hourly + umbrella alert)** — answers for another day too; a proactive watcher announces transitions about two hours out.
  - "what's the weather doing", "should I bring an umbrella", "will it rain tomorrow"
  - Actions: `weather_briefing`, `weather_forecast`
- **News briefing** — RSS headlines, each rewritten as one sentence by the model (`NEWS_BRIEFING_SUMMARIZE`, on; off reads the feed titles as they are).
  - "what's in the news", "read me the headlines"
  - Action: `news_briefing`

### Category 8: Memory, recall & task queue

- **Session memory recall** — queries prior sessions (resolves "yesterday" / "last night" / weekday names) and also summarises this session.
  - "what did we do yesterday", "recap our conversation", "what did we work on Tuesday"
  - Action: `session_memory_recall`
- **Session resume** — picks up the unfinished thread from the most recent prior session. Offers it once at startup when the last session ended within 18 h, but not after a quick restart (within 15 minutes of the last session) and never as a narrator's summary.
  - "where did we leave off", "pick up where we left off", "where were we"
  - Action: `session_resume`
- **What he knows about you** — stored facts are already in the prompt, so "what do you know about me" needs no action; "have you learned anything about me lately" reports the background fact extractor. Long-term memory is a tiered semantic store (`core/long_term_memory.py`) next to `bobert_memory.json`. **Owner-only learning** (off by default - setting `LEARN_ONLY_FROM_OWNER`) lets a turn teach only when it was clearly the owner's: typed, led by the wake word, matched to the enrolled voiceprint, or a quick follow-up to one of those (`core/learn_gate.py`).
  - Actions: `ambient_extract_status`, `ambient_extract_now`, `show_recent_facts` (tray)
- **Memory maintenance** — back up, drop the last hour, or erase everything (backs up first and waits for a spoken yes); a wipe reaches every memory store.
  - "back up your memory", "forget the last hour", "forget everything you know about me"
  - Actions: `export_memory`, `forget_last_hour`, `reset_memory`
- **Pending promises** — the "I'll let you know when…" callbacks skills made (`skill_utils["make_promise"]`).
  - "what are you waiting on", "cancel promise 3"
  - Actions: `list_promises`, `cancel_promise`
- **Task queue → Claude Code handoff** — adds work items to `jarvis_todo.md`.
  - "queue this for Claude Code: rewrite the dispatcher", "what's on the to-do list", "put it on the doorless" (Whisper's mishearing of "to-do list")
  - Actions: `queue_task`, `show_tasks`, `clear_tasks` (asks for confirmation)
- **Dossier — "pull up the file on X"** — aggregates memory facts, queued tasks, recent log mentions and a web abstract, and slides a "DOSSIER — X" card onto the top monitor.
  - "pull up the file on <subject>", "dossier on <subject>"
  - Actions: `dossier`, `pull_up_file`, `pull_up_dossier`, `file_on`, `dossier_on`, `what_do_you_have_on`, `whats_on_file`
- **Project status** — reads back the owner's own project list (`data/projects_status.json`); says so plainly when there is none.
  - "what am I working on", "what are my projects"
  - Action: `project_status`
- **Personal file search (RAG)** — ChromaDB index over your documents and notes, with Ollama embeddings.
  - "find my notes on <topic>", "open the top result", "reindex my files"
  - Actions: `rag_search` / `search_my_files`, `rag_open_top`, `rag_reindex`, `rag_status`, `rag_configure`, `rag_search_quiet`
- **Chappie mode** (off by default - setting `CHAPPIE_ENABLED`) — groups overheard transcripts into episodes and entity facts, and recalls them only when asked. The recall actions always load; only the background loop (which spends Claude budget) is gated. Not yet taught to the voice prompt.
  - Actions: `chappie_recall_today`, `chappie_recall_entity`, `chappie_status`

### Category 9: HUD overlays

- **Unified HUD** (`hud/jarvis_unified_hud.py`) — the main on-screen HUD, launched on `HUD_MONITOR` at boot (`HUD_ENABLED`, on by default); shows state, live print progress and ETA. Right-click for hide/resize/reset; Ctrl+wheel resize; drag to move; double-click reset.
  - "hide the HUD", "show the HUD", "toggle HUD"
  - Actions: `hide_hud`, `show_hud`, `toggle_hud` (no read-back exists for the plain HUD)
- **Brain glow** (`core/brain_glow.py`, on by default - setting `BRAIN_GLOW_ENABLED`) — a glowing ring around the reactor in the unified HUD, the holographic overlay and both arc reactors takes the colour of the brain that is answering: local model blue, Claude Sonnet gold, Opus violet, Haiku green, Fable rose, any other cloud model silver. It changes the moment you switch brains and per turn when a turn is really answered by the other brain (a local turn the cloud had to answer, a cloud turn that fell back to local); the brain's name shows under the reactor for a few seconds (`BRAIN_GLOW_LABEL_S`). The halo and core keep the listening / thinking / speaking colours, and there is no ring while JARVIS is asleep or in standby. Per-tier colour overrides: `BRAIN_GLOW_COLORS` (user_settings.json only).
  - "what brain are you on", "what model are you using" (names the colour)
  - Action: `current_model`
- **Fullscreen holographic overlay** (`hud/jarvis_holo.py`) — big cinematic reactor on the top monitor.
  - "show the holographic overlay", "dismiss the holo", "is the holographic overlay still up"
  - Actions: `show_holographic_overlay` / `show_holo` / `hud_on` / `holographic_on`, `hide_holographic_overlay` / `hide_holo` / `hud_off` / `dismiss_holo` / `holographic_off`, `toggle_holographic_overlay` / `toggle_holo`, `holographic_status`
- **Arc-reactor workshop canvas** (`hud/holo_workshop_canvas.py`) — small rotating reactor in the top monitor's corner.
  - "show the arc reactor", "hide the arc reactor", "pulse the reactor"
  - Actions: `arc_reactor`, `arc_reactor_on`, `arc_reactor_off`, `arc_reactor_pulse`, `holo_workshop_canvas`, `holo_workshop`, `workshop_canvas`
- **Workshop HUD** (`hud/workshop_hud.py`) — slim top-right panel: arc-reactor "power %", CPU/RAM bars, Bambu progress.
  - "show the workshop HUD", "is the workshop HUD showing"
  - Actions: `workshop_hud`, `workshop_hud_toggle`, `show_workshop_hud` / `workshop_hud_on`, `hide_workshop_hud` / `workshop_hud_off`, `workshop_hud_status`
- **Status rings** — the four-quadrant arc-reactor status ring (`hud/arc_reactor_status_hud.py`), the top-centre Stark status ring and the second-generation holo HUD (`hud/holographic_hud_v2.py`). Ask "is it up" with the long read-back names; the short names are toggles.
  - "is the arc reactor ring on", "is the stark ring up"
  - Read-backs: `arc_reactor_status_status`, `stark_status_ring_status`, `holo_hud_v2_status`
- **Holographic globe** (`skills/globe.py`, renderer `hud/globe_hud.py`) — a slowly rotating wireframe Earth on any monitor, with pins for ~260 major cities and capitals (or a "lat, lon"); consecutive pins are joined by great-circle arcs, and the oldest of 12 pins drops off. Nothing launches at boot.
  - "show me the globe", "put the globe on the left monitor", "show me where Tokyo is", "clear the pins", "hide the globe"
  - Actions: `show_globe`, `hide_globe`, `globe_pin`, `globe_clear`
- **Bambu overlays** (`hud/bambu_h2d_overlay.py`, `hud/workshop_print_monitor.py`, `hud/bambu_camera_hud.py`) — see Category 6.
- **Air cursor** (`hud/jarvis_air_cursor.py`) and **reticle** (`hud/jarvis_reticle.py`) — draw the Kinect hand cursor(s) and click targets during UI automation. No voice commands.
- **Suit-up sequence** — a 6–8 s boot cinematic with a system readout and "Welcome back, sir." Fires once per day on the first warm restart; on demand any time.
  - "suit up", "boot sequence"
  - Actions: `suit_up`, `suit_up_sequence`
- **System tray applet** (`tray.py`) — arc-reactor icon tinted by listen state plus a speaking halo, upgrade-queue badge and Bambu print mark. Menu described under "How to invoke". If the tray was quit, "show the tray icon" (`show_tray`) brings it back.

### Category 10: System monitoring & status

- **Quick health check** — CPU, RAM, top processes, C: free space, network rates. A background monitor alerts if CPU stays above 90% for a minute (naming the process that used the most CPU over that minute) or RAM crosses 90%.
  - "how's the system", "how much space is left on my C drive"
  - Action: `check_system`
- **GPU / VRAM** — loaded Ollama models, VRAM used, GPU load and temperature, what runs local versus cloud.
  - "what's on the GPU", "how much VRAM is left"
  - Actions: `gpu_usage`, `gpu_status`, `vram_status`, `show_vram`, `whats_loaded`
- **Detailed sensors** — HWiNFO per-sensor temps, voltages and fan speeds.
  - "what's my VRM temp", "fan speeds", "show me every sensor"
  - Action: `hardware_sensors`
- **System pulse** — one-line status report. Proactive every 15 min if anything is abnormal; a GPU kept busy by the local model server is not an alert, and one sample is not news.
  - "status report", "pulse check"
  - Actions: `system_pulse`, `status_report`
- **Status panel ("suit diagnostics")** — the full multi-line readout plus a HUD card.
  - "system status", "suit diagnostics"
  - Actions: `status_panel`, `system_status`, `suit_diagnostics`
- **Self-diagnostic** — sweeps webcam, mic, TTS, STT, GPU, disk and RAM and speaks a summary; also runs every 30 minutes. Four diagnostic daemons (self-diag, deep audit, crash watcher, anomaly watcher) run alongside.
  - "are you ok", "what is broken", "run a diagnostic", "pause the diagnostics"
  - Actions: `self_diagnostic` / `run_diagnostic` / `system_check` / `are_you_ok`, `whats_broken`, `diagnostic_history`, `last_diagnostic_run`, `show_last_diagnostic`, `diagnostic_status`, `diagnostic_daemon_status`, `pause_diagnostics`, `resume_diagnostics`
- **Self-test probes** — check one subsystem; the verdict goes to the console log, not the voice.
  - "test your microphone", "test all your skills", "how fast is your brain right now"
  - Actions: `test_mic`, `test_tts`, `test_vision`, `test_each_skill`, `latency_benchmark`
- **Disk + budget watchdog** — C: free space and the last known credit balance in one line; background alerts below 10 GB / $10.
  - "budget check", "how are we on disk and credits"
  - Action: `check_budget`
- **Stability gate** — results of the post-upgrade safety probe.
  - "stability gate status", "what was the last stability gate result"
  - Actions: `gate_status`, `stability_gate_status`, `last_gate_result`
- **Screen-watch wellness nudge** — 25 min on the same window with idle input triggers a stretch offer.
  - "are you watching my screen"
  - Action: `screen_watch_status`
- **Presence wellness** — after ~90 min of continuous desk presence, a hydration / eye-break line; 60-min snooze.
  - "am I due for a break", "how long have I been at it"
  - Action: `wellness_status`

### Category 11: Focus / workshop / night-owl / game modes

- **Focus mode (hold and recap)** — `skills/focus_mode.py` holds every unsolicited announcement until you resume, then recaps what was held. It chains `skills/dnd_focus_mode.py`, which mutes Windows toasts, sets Teams to Do Not Disturb, pauses banter/wellness/Teams nudges and shortens replies; critical alerts (print failure, timers) still fire.
  - "focus mode for 90 minutes", "I'm back, what did I miss", "am I still in focus mode", "end focus mode"
  - Actions: `focus_mode_on` / `do_not_disturb` / `quiet_mode`, `focus_mode_off` / `resume`, `whats_missed`, `focus_mode`, `end_focus_mode`, `focus_mode_status`
- **Workshop mode (auto-engaged)** — fires when Bambu Studio / Fusion 360 / SolidWorks / FreeCAD / OnShape / Blender / OpenSCAD / Orca / Prusa / Cura appears: TTS at 70% volume, single-sentence replies, a print-status line if one is mid-flight, and focus mode for an hour (released when the app closes).
  - "is workshop mode on"
  - Action: `workshop_status`
- **Night-owl mode** — softer, quieter TTS, holographic overlay dimmed to ~40%, non-critical nudges muted; critical alerts (print failure, timers) still fire. Auto 23:00–06:00 when both `NIGHT_OWL_AUTO` and the master switch `NIGHT_QUIET_ENABLED` are on (both ship on); with either off it engages only on request. Releases at 06:00 or on "good morning".
  - "night owl mode", "end night owl mode", "is night owl mode on"
  - Actions: `night_owl_on` / `night_owl_mode` / `enable_night_owl`, `night_owl_off` / `end_night_owl` / `disable_night_owl`, `good_morning`, `night_owl_status`
- **Game mode** — frees VRAM/RAM for a running game by moving to a smaller local brain; never mutes JARVIS. The automatic game watcher is off by default - setting `GAME_MODE_ENABLED`; the voice commands always work.
  - "divert power", "normal power", "treat this as a game"
  - Actions: `game_mode_on` / `low_power_mode`, `game_mode_off` / `full_power` / `normal_power`, `game_mode_status`, `game_mode_learn_this`

### Category 12: Microsoft Teams integration

- **Teams unread check** — an on-demand vision sweep of all monitors for Teams unread badges. A background nudger can run every 10 minutes with a 30-min snooze (off by default - setting `TEAMS_NUDGE_ENABLED`).
  - "check Teams", "do I have any Teams messages"
  - Action: `check_teams`

### Category 13: Timers, reminders & schedules

- **Set / list / cancel timers** — a spoken "Reminder, sir — <message>" when it fires. "List timers" answers only from the timer store; "cancel timer" with no number cancels the soonest, "cancel all timers" clears them.
  - "set a timer for 5 minutes to check the oven", "remind me in an hour to stretch", "cancel timer 3"
  - Actions: `set_timer`, `list_timers`, `cancel_timer`
- **Schedules and conditional triggers** — APScheduler-backed recurring jobs, one-shot jobs at a wall-clock time, and "when X happens, do Y" triggers; chain actions with `&&`.
  - "every morning at 8 give me the briefing", "tomorrow at 9am open Bambu Studio", "when the print finishes, tell me"
  - Actions: `schedule_recurring` / `schedule_cron`, `schedule_once`, `schedule_when` / `when_condition`, `list_schedules`, `cancel_schedule` / `remove_schedule`, `run_schedule` / `fire_schedule`, `schedule_status`

### Category 14: Self-management & learning

- **AI brain selection** — which local model serves turns (switched live, persisted) and which route chat takes (see Overview).
  - "what model are you using", "use the fast one", "use local only", "use the cloud model"
  - Actions: `current_model`, `list_models`, `set_model`, `set_brain` (`local` / `cloud` / `auto`), `switch_llm`
- **Self-knowledge** — "how smart are you", "how do you compare to Claude" get an honest answer from live engine facts (which brain, which speech engines, measured speed, strengths and limits) instead of a joke; the facts are rendered per turn, never frozen into the prompt.
- **Model costs** — each model's estimated cost per conversation, and call stats for the active one.
  - "what does each model cost", "which model is cheapest"
  - Actions: `model_costs` / `compare_models` / `llm_costs` / `model_prices`, `show_llm_stats`
- **Running costs** — what it costs to run JARVIS: an electricity estimate from the live GPU and CPU draw over the hours run today and this month (`ELECTRICITY_RATE_PER_KWH`), this session's and this month's Claude spend from a persisted token tally (counts only, never text), and a one-line verdict. Not the account balance; that is `check_credits`.
  - "how much does it cost to run you", "how much do you cost per month"
  - Action: `running_costs`
- **Claude credits** — opens Anthropic billing in a hidden Chrome window, reads the balance via vision, closes it. A background monitor checks hourly and speaks if the balance is below $5.
  - "check my credits", "am I running low on credits"
  - Action: `check_credits`
- **Version, changelog and updates** — reports the running version (the top-level `VERSION` file) and when it was released, summarises the newest entry of the local `CHANGELOG.md` that the upgrade pipeline writes (gitignored; the last three with "lately"), and checks GitHub for a newer release tag (also once at boot, `UPDATE_CHECK_ENABLED`).
  - "what's new", "show me the changelog", "check for updates", "are you up to date"
  - Actions: `version_info` / `what_version` / `when_updated`, `read_changelog` / `whats_new` / `what_changed` / `show_changelog` / `recent_changes`, `check_for_updates`
- **Bug reports** — logs a user-reported bug scrubbed of personal info locally and opens a pre-filled GitHub issue for review.
  - "report a bug: the timer never fired", "you got that wrong"
  - Actions: `report_bug`, `file_a_bug`, `log_a_bug`, `report_a_bug`
- **Restart / shut down** — an action that ends JARVIS's process (restart, shutdown, upgrade, overnight) runs only when the owner's own words asked for it, and the action-name corrector never guesses its way onto one ("turn it off" with nothing for "it" to mean gets a question, not a shutdown).
  - "restart yourself", "relaunch", "shut down JARVIS"
  - Actions: `restart`, `shutdown_jarvis` (aliases `exit_jarvis`, `quit_jarvis`, `turn_off_jarvis`, `power_off_jarvis`, `shut_down`)
- **Upgrade pipeline** — hands the task queue to Claude Code: kills JARVIS, runs Claude Code on the pending tasks, relaunches when done. Guarded by `OVERNIGHT_UPGRADE_ENABLED`, which ships off, so a stray "go ahead" can't start it.
  - "upgrade yourself", "apply the changes"
  - Action: `upgrade`
- **Overnight upgrade engine** — the autonomous improvement loop (generate ideas, queue them, run the pipeline, repeat while idle) (off by default - setting `OVERNIGHT_UPGRADE_ENABLED`); with it off only the quiet-standby half runs.
  - "start overnight upgrade", "improve yourself while I sleep"
  - **Goodnight phrasings always fire this**: "goodnight", "good night", "I'm going to bed", "heading to bed", "going to sleep", "I'm off to bed", "time to sleep"
  - Actions: `start_overnight_upgrade`, `stop_pipeline`
- **Pattern learning** — nightly aggregator over `data/usage_patterns.jsonl` (03:00), weekly digest on Mondays; feeds the anticipation engine.
  - "what patterns have you learned", "what have you noticed about my habits"
  - Actions: `pattern_predictions`, `pattern_offer_now`, `pattern_aggregate`, `pattern_stats`, `weekly_digest`
- **Anticipation engine and briefings** — a 60-second poll that volunteers one in-character line when pattern + dwell + time of day + gaze line up (20-min cooldown, 35% fire probability, silent in calls / sleep / standby, never an offer for the app you are already in), plus the anticipation-briefing and weekly-digest schedulers built on the pattern snapshots.
  - "are you predicting things right now", "why do you keep briefing me", "is the weekly digest still running"
  - Actions: `anticipation_status`, `anticipation_briefing_status`, `anticipation_briefing_now`, `weekly_digest_status`, `weekly_digest_now`
- **Banter engine** — dry remarks when JARVIS notices a tell (a repeated question, the same target opened 5+ times today, >40 tabs, "play music" while music is already playing). 30-min cooldown, 50% fire probability.
  - "are you still making jokes at me"
  - Action: `banter_status`
- **Skills system** — self-extension.
  - "what skills do you have" → `list_skills`
  - "teach yourself to <X>" → `create_skill, <name> | <description>` writes a module to `pending_skills/`; you review it and move it to `skills/` to activate.
  - `reload_skills` re-imports every skill (also on the tray's Power tools menu, with Force Backup, Run Smoke Test and Stop Running Pipeline).
  - Skills plug in through `skill_utils`: utterance routes (`register_utterance_route`), after-reply hooks that see every owner turn's reply (`register_after_reply`), promises, phone pings and dashboard panels.
- **MCP tools** — connects the servers in `mcp_servers.json` (see `mcp_servers.example.json`) and registers each tool as `mcp_<server>_<tool>`; without a config these actions are not loaded.
  - "MCP status", "list MCP tools"
  - Actions: `mcp_status`, `mcp_list_tools`, `mcp_call`, `mcp_reload`

### Category 15: REPO Robot project tracker

- **Robot status / blockers / next step** — reads `data/repo_robot_state.json` (next step, blockers, parts on order, last firmware flash) + `jarvis_todo.md` + recent logs.
  - "where is the robot build at", "what's blocking the robot", "what's next on the robot"
  - Actions: `robot_status`, `robot_blocker`, `next_robot_step`

### Category 16: Ambient audio, listening modes & wake word

- **Wake-word mode** (off by default - setting `REQUIRE_WAKE_MODE`) — Alexa-style: require "JARVIS" near the start of every command while a TV or other audio plays (see "How to invoke" for the rule). The OS media session and the room-music detector already gate automatically; this is the manual switch.
  - "require the wake word", "turn off wake word mode", "is wake word mode on"
  - Actions: `wake_word_mode_on`, `wake_word_mode_off`, `wake_word_mode_status`
- **Media voice gate** (on by default - setting `MEDIA_VOICE_GATE_ENABLED`) — while another app on the PC is producing sound, a mic turn whose voice scores below `MEDIA_VOICE_GATE_REJECT_BELOW` (0.45) against the enrolled owner voiceprint is dropped, so a video saying "Jarvis, …" cannot command him (`core/media_gate.py`). A short media control ("pause", "next song", "turn it down"), a stop word, typed turns and guest mode always pass; with nobody enrolled the turn is allowed and logged.
- **Standby audio detector** — spectral classifier on raw mic chunks plus a whisper-tiny lyric-detection loop; sets the "music currently playing" state so a wake word buried in a lyric won't flip JARVIS out of standby (`AMBIENT_MUSIC_REFUSE_WAKE`).
  - Action: `audio_music_status` (a read-back not yet taught to the voice prompt)
- **Double-clap trigger** (off by default - setting `CLAP_TRIGGER_ENABLED`) — two sharp claps ~0.15–0.7 s apart, with nothing else loud around them, run the clap routine (`CLAP_TRIGGER_ACTION`): at first just "You rang, sir?" (a mechanical key's click can pass for a clap, so listen for false triggers first), then — by voice or in Settings — the morning workspace setup or the morning briefing. Nothing else can run on a clap (an allow-list). Listens through the main loop's microphone fan-out (never a second stream); 60 s cool-down; never while JARVIS is speaking, during the quiet hours (23:00–07:00), in focus or game mode, while music, a video or anything else plays on the speakers, or on staging; asleep only with "clap to wake" (`CLAP_TRIGGER_WAKE`). Speech, beats, typing, knocks, glass taps and doors are rejected by the detector (`core/clap_detector.py`).
  - "turn on the clap trigger", "clap trigger off", "is the clap trigger on", "clap trigger runs the morning setup", "clap trigger just answers"
  - Actions: `clap_trigger_on`, `clap_trigger_off`, `clap_trigger_status`, `clap_trigger_routine`
  - Skill: `skills/clap_trigger.py`; Settings: Hearing tab → Double-clap trigger
- **Ambient-learning mode** — silent standby that keeps listening and learning but speaks only after "JARVIS"; auto-engaged after an upgrade/overnight run. Choose whether a wake answers once then goes quiet, or stays talkative.
  - "go quiet and keep learning", "exit ambient learning", "stay talkative after I wake you"
  - Actions: `ambient_learning_mode_on` / `ambient_learning_mode_off`, `wake_resume_answer_then_quiet`, `wake_resume_stay_talkative`
- **Ambient listening (multimodal)** (off by default - settings `AMBIENT_LISTEN_ENABLED`, `AMBIENT_SCREEN_ENABLED`) — passive mic transcription sharing the main loop's mic, optional WASAPI system-audio capture and periodic screen snapshots described by the local VLM (sensitive windows skipped), all feeding the fact extractor. With `AMBIENT_STT_YIELD` (off by default) the ambient daemons hold their batches while you are talking to JARVIS so your own transcription never waits behind them.
  - "are you listening in the background" → `ambient_listen_status`
  - Switching it: the tray's **Ambient Mode** toggle, or `ambient_mode` / `ambient_mode_on` / `ambient_mode_off`, `ambient_listen_start` / `ambient_listen_stop`, `ambient_audio_start` / `ambient_audio_stop`, `ambient_screen_start` / `ambient_screen_stop`, `ambient_full_start` / `ambient_full_stop`, `ambient_mic_only` (registered, not yet taught to the voice prompt)
- **TV detection** (off by default - setting `TV_DETECT_ENABLED`) — a camera check for a bright, flickering TV that adds a visual veto to ambient learning.
  - "turn on TV detection", "is the TV on", "calibrate the TV region"
  - Actions: `tv_detect_on`, `tv_detect_off`, `tv_detect_status` / `tv_status`, `calibrate_tv_region` / `tv_calibrate`

### Category 17: Voice, speaker ID & speech output

- **Wake-word detector (barge-in)** — `skills/wake_listener.py` runs an openWakeWord / Porcupine hotword detector in the background (autostart off by default - setting `WAKE_WORD_AUTOSTART`). While it runs, saying "JARVIS" over him cuts his speech (`BARGE_IN_ENABLED`; see "Special voice patterns").
  - "start listening for the wake word", "stop the hotword", "wake listener status"
  - Actions: `wake_listener_start`, `wake_listener_stop`, `wake_listener_status`, `wake_listener_configure`
- **Voice ID and guest mode** — record a voiceprint, identify the speaker, gate wake events to enrolled voices, and bypass the gate for visitors until the next restart.
  - "learn my voice", "who's talking", "guest mode on"
  - Actions: `enroll_voice` / `learn_my_voice`, `whos_talking` / `who_is_talking` / `identify_speaker`, `list_enrolled_voices` / `enrolled_voices`, `forget_voice`, `voice_id_status`, `set_active_speaker`, `voice_gating_on` / `voice_gating_off`, `guest_mode_on` / `guest_mode_off`
- **TTS backend switching** — Edge (shipped default, `en-GB-RyanNeural`, needs the network), Kokoro (CPU), pyttsx3 or XTTS. Cloud replies stream sentence by sentence (`STREAMING_TTS_ENABLED`); on Kokoro a long reply starts playing its first sentence while the rest renders (`SENTENCE_TTS_ENABLED`).
  - "switch to Edge TTS", "use the local voice", "list TTS backends"
  - Actions: `set_tts_backend`, `list_tts_backends`, `enroll_xtts_sample`
- **Local voice clone** (off by default - setting `VOICE_CLONE_ENABLED`) — Chatterbox speaks in a consented cloned voice; VRAM-gated, falling back to the normal voice.
  - "use the jarvis voice", "what voice are you using", "stop cloning"
  - Actions: `set_voice_profile` / `use_voice_profile` / `switch_voice_profile`, `list_voice_profiles`, `voice_clone_status`, `disable_voice_clone` / `voice_clone_off` / `stop_voice_clone`
- **Settings-only voice features** — `VOICE_MODE = realtime` (streaming pipeline; needs RealtimeSTT, RealtimeTTS and PyAudio, else falls back to turn-based; `turn_based` by default), `STT_HOTWORDS` / `STT_REPLACEMENTS` (names Whisper should expect, fixes for words it mishears; the wake words are never hotwords and an echo of the hint list is dropped), `FOLLOWUP_WINDOW_S` (follow-ups skip the wake word for N seconds; 0 by default), the processing filler (see "Special voice patterns"), and the self-echo, known-device and Whisper-noise filters (all on).

### Category 18: Face recognition & Kinect

- **Face recognition** (off by default - setting `FACE_ID_ENABLED`) — enroll faces from the webcams and say who is in front of them right now; identity questions are always a live camera look, never memory.
  - "learn my face", "who's at the desk", "is face recognition on"
  - Actions: `enroll_face`, `learn_guest`, `recognize_face` / `whoami`, `face_id_status`, `forget_face`, `list_enrolled_faces`
- **Greet new people** (off by default - setting `GREET_NEW_PEOPLE_ENABLED`) — says hello once when several unfamiliar faces appear.
  - "notice when people arrive", "stop greeting people"
  - Actions: `greet_new_people_on`, `greet_new_people_off`
- **Guard mode** — arms every camera as a motion alarm (frame differencing, plus Kinect skeletons when present); snapshots and a rate-limited alert, also sent to your phone when a phone bridge is set up. Arms only on request.
  - "guard the room", "stand down", "are you watching"
  - Actions: `guard_on`, `guard_off`, `guard_status`
- **Kinect depth sensor** (off by default - setting `KINECT_ENABLED`) — connection and stream status, body count and nearest distance, and a look through its camera.
  - "kinect status", "who's in the room", "look through the Kinect"
  - Actions: `kinect_status`, `who_is_here` / `scan_room`, `kinect_look`
- **Gestures** (off by default - setting `KINECT_GESTURES_ENABLED`) — wave to wake, swipe to stop speech and cancel a pending confirmation. A raised hand never confirms anything (a confirmation always needs a spoken yes); with something pending it reminds you to say yes. Gesture barge-in is a separate opt-in (`JARVIS_GESTURE_BARGE_IN` environment variable).
  - "turn on kinect gestures", "what gestures can you see through the kinect"
  - Actions: `gestures_on`, `gestures_off`, `gesture_status`
- **Air-mouse** (off by default - setting `KINECT_AIR_MOUSE_ENABLED`) — raise a hand above the shoulder to take the cursor; close the left/right hand to left/right-click, hold to drag. "Arm" relaxes the strict pose gate. Touching the real mouse or keyboard always takes the cursor back; the low-level input hook that tells real input from the air-mouse's own is a separate opt-in (`AIR_MOUSE_LL_HOOK_ENABLED`, off because it puts every input event on the PC behind JARVIS's process; an idle-time check does the job without it).
  - "turn on the air mouse", "take the cursor", "release the cursor", "calibrate the air mouse"
  - Actions: `air_mouse_on` / `air_mouse_off` / `air_mouse_status`, `air_mouse_arm` (aliases `take_the_cursor`, `give_me_the_cursor`, `mouse_control_on`, `hand_mouse_on`), `air_mouse_disarm`, `calibrate_air_mouse`
- **Air control** (off by default - setting `AIR_CONTROL_ENABLED`) — movie-style spatial hand mouse: reach toward the sensor, fist to grab and drag across monitors, squeeze to click, point to scroll.
  - "let me control the mouse with my hand", "hand mouse off", "is air control on"
  - Actions: `air_control_on`, `air_control_off`, `air_control_status`
- **Point-to-control** (off by default - setting `KINECT_POINT_CONTROL_ENABLED`) — point at a calibrated device and say "turn that on".
  - "calibrate pointing for the desk lamp", "turn that on", "what can I point at"
  - Actions: `point_control_on` / `point_control_off`, `point_calibrate`, `point_control`, `list_point_targets`, `forget_point_target`, `point_status`
- **Two-hand window resize** — raise both hands to grab the focused window; spread to grow, pinch to shrink, slide to move (`skills/kinect_two_hand.py`, needs the Kinect on; no voice commands).

### Category 19: Smart home & network

- **Smart-home router** — natural-language control routed to whichever brand skill owns the device (Hue, Govee, LIFX, Kasa/Tapo, Tuya/Smart Life, Ecobee, Nest, Ring).
  - "turn off the office light", "dim the bedroom lamps to 30%", "set the living room thermostat to 72"
  - Actions: `smart_home_control` / `control_smart_home` / `control_device` / `control_light` / `control_plug`, `smart_home_router_status` ("why did that light not respond"), `refresh_smart_home_router`
- **Discovery and catalog** — Alexa + LAN discovery builds `data/smart_home_devices.json`; "list my smart-home devices" reads that catalog back (plainly when it is empty).
  - "discover my smart-home devices", "list my smart-home devices", "forget my Alexa login"
  - Actions: `discover_smart_home`, `list_smart_home_devices`, `forget_alexa_login`
- **Per-brand lists and setup** — "list my Hue lights", "list my Tuya plugs", "authorize Ecobee", "set the Hue bridge IP"
  - Actions: `hue_list_devices`, `govee_list_devices`, `lifx_list_devices`, `kasa_list_devices`, `tuya_list_devices`, `ecobee_list_devices`, `nest_list_devices`, `ring_list_devices`, `hue_set_bridge_ip`, `hue_retry_connect`, `ecobee_request_pin`, `ecobee_authorize`, `ecobee_complete_setup`, `nest_authorize`, `ring_authorize`
- **Deco mesh network** — who is on the Wi-Fi, single-device presence, bandwidth hogs, topology, guest network.
  - "who's on the wifi", "is the Pi online", "network usage", "turn off guest wifi"
  - Actions: `who_is_on_wifi`, `is_device_online` / `device_online`, `bandwidth_hogs`, `deco_topology`, `deco_status`, `deco_refresh`, `kick_guest_network` / `disable_guest_network`, `enable_guest_network`

### Category 20: Browser agent

- **Sandboxed browser automation** (Playwright + browser-use, separate profile from your everyday browser; needs `ANTHROPIC_API_KEY`).
  - "use the browser to find the cheapest flight to Tokyo next Friday" → `browser_task`
  - "go read up on PETG nozzle temps and summarise it" → `browse_for`
  - "find me the cheapest 2TB NVMe" → `find_cheapest`
  - "book me a haircut Friday afternoon" → `book_appointment`
  - "fill that form in with my name and email" → `fill_form` (stops short of submitting)
  - "how's the browser doing", "stop the browser", "open theverge.com in your browser" → `browser_status`, `browser_stop`, `browser_open`; also `browser_screenshot`, `browser_reset_profile` (wipes the agent's logins; confirm first)

### Category 21: Mail, phone & notifications

- **Email triage** — Gmail and Outlook (Microsoft Graph): unread roll-call, read a thread, categorise the inbox, a spoken inbox briefing, and drafted replies that send only on "send". On a local-only install it never sends mail to Claude (`core/cloud_gate.py`).
  - "check my email", "reply to that email", "send that draft", "archive that email"
  - Actions: `list_unread`, `read_email` / `read_thread`, `triage_inbox`, `email_briefing`, `draft_reply`, `confirm_pending_draft` / `send_draft`, `discard_draft`, `edit_pending_draft`, `archive_email`, `email_triage_status`
- **Amazon order tracker** — reads order-update emails (the poller is off by default - setting `AMAZON_TRACKING_ENABLED`).
  - "where's my Amazon order", "what was delivered this week"
  - Actions: `check_orders`, `recent_delivery`, `amazon_tracking_status`
- **Windows notification triage** — captures toasts, applies your rules, reads back recent ones.
  - "what notifications have come in", "list notification rules", "pause notification triage"
  - Actions: `recent_notifications_summary`, `list_notification_rules`, `add_notification_rule`, `remove_notification_rule`, `pause_notification_triage`, `resume_notification_triage`, `triage_status`
- **Phone push** — Telegram, ntfy or Pushover; prefix `!urgent` / `!high` for priority.
  - "send the print status to my phone", "text my phone <message>", "phone status"
  - Actions: `notify_phone` / `push_to_phone` / `text_my_phone`, `phone_status`, `list_phone_backends`, `pause_phone_bridge`, `resume_phone_bridge`
- **Phone pings** (`core/phone_ping.py`, `PHONE_PING_ENABLED` on, but a no-op with one boot line until a phone bridge is configured) — JARVIS texts your phone unprompted only when something needs you: a print finished, failed (not one you cancelled) or paused with an error (once per pause); a guard-mode alert (its own switch — "turn off phone pings" never silences the guard); a talking-device skill that needs you (`skill_utils["ping_phone"]`); and, off by default, a confirmation you asked for and left to lapse while away (the action's name only, `PHONE_PING_CONFIRM`) and a daily summary (`PHONE_PING_SUMMARY`). Nothing pings while you are talking to him or, while he is awake, working at the PC (he says it out loud), in focus mode, or 23:00–07:00 / after "goodnight" (held, then sent as one message), except guard alerts; at most 6 an hour; secrets are scrubbed from every text.
  - "how do I connect my phone" (the setup steps), "phone ping status", "turn off phone pings", "send a test ping"
  - Actions: `phone_setup_help`, `phone_ping_status`, `phone_pings_on`, `phone_pings_off`, `phone_ping_test`
  - Settings: Integrations tab → Phone pings
- **Outbound message gate** — every outgoing message is read aloud and waits for a short, clear yes; refuses while asleep or in standby.
  - "is the outbound gate armed"
  - Actions: `draft_preview_gate_status` / `outbound_gate_status`

### Category 22: Creative & tool integrations

- **Website builder** (`skills/site_builder.py`) — "build a website for <business>" looks the business up, writes one complete, self-contained HTML page (no JavaScript) in the background — on Claude when the cloud is allowed, otherwise on the local model — saves it under the data folder's `sites/<name>/index.html`, announces it and opens it. It never publishes anything or contacts the business.
  - "build a website for Blue Door Bakery", "make a landing page for Blue Door Bakery in Springfield"
  - Action: `build_website`
- **OBS Studio** — start/stop/pause recording, switch scenes, toggle a source's mute (OBS WebSocket).
  - "start recording", "switch to gameplay scene", "mute the mic in OBS"
  - Actions: `obs_start_recording`, `obs_stop_recording`, `obs_pause_recording`, `obs_switch_scene`, `obs_toggle_mute`
- **Image generation** (off by default - setting `IMAGE_GEN_BACKEND`, `"off"`) — local SDXL-Turbo / ComfyUI.
  - "make me a picture of a Mars colony at sunset"
  - Actions: `generate_image`, `make_picture`

### Category 23: Web dashboard & settings

- **Web dashboard** (`tools/web_interface.py`, served by `skills/web_interface.py`; off by default - setting `WEB_INTERFACE_ENABLED`) — binds 127.0.0.1 and refuses a non-local bind without a token. Views:
  - **Live** — status strip, standby banner, tray controls (including the pinned wake-word switch), quick-action buttons, log tail, typed command box.
  - **System**, **Voice**, **Camera** (live MJPEG tiles with each camera's gate verdict), **Memory** and **Settings** (the same schema as the Settings window).
  - **Actions** — the live action registry, run by name; destructive names ask first (`core/action_risk.py`).
  - **Timeline** ("What JARVIS did") — the last 50 turns: time, source, actions run or failed, latency. The words appear only with `DASHBOARD_SHOW_TRANSCRIPTS` on (off by default) and a loopback viewer.
  - **Proactive** — every proactive and background behaviour (44 of them) on one page, each with its switch.
  - Skills can add their own panels (`core/web_panels.py`).
  - "start the web interface", "is the web dashboard running", "stop the web interface"
  - Actions: `web_interface_on`, `web_interface_off`, `web_interface_status`
- **Settings window** (`tools/settings_window.py`) — 151 settings in seven tabs (Voice, Hearing & Mic, AI & Models, Cameras & Kinect, Privacy, Integrations, Advanced) plus integration status rows; the dashboard's Settings view renders the same schema. Values persist to `data/user_settings.json` (gitignored). A saved file can add confirmation keywords and screenshot-blocklist entries but never remove the shipped ones.

### Category 24: Conversation modes, instant answers & command chaining

- **Conversation modes** — Controlled (only exact, deterministically matched commands), Smart (default: the model chooses actions), Agent (plan → execute → critique → report, with a deeper follow-up loop). Persisted across restarts.
  - "agent mode", "controlled mode", "what mode are you in"
- **Instant answers** — `core/fast_paths.py`, `core/date_math.py`, `core/world_clock.py`, `core/spoken_math.py`.
  - "how many days until Christmas", "what time is it in London", "what's 12 times 7", "what did I just ask you", "what was the first thing I asked you"
- **Command chaining** — `core/dispatcher.py` splits "and"/comma chains of known intents (play, pause/resume/skip, focus timers, timers, volume, screenshot, show tasks) and runs them without the model.
  - "play lofi and set a 20 minute timer", "pause the music and take a screenshot"
- **Briefing fan-out** — an explicit briefing request can fan out to parallel sub-agents (`core/orchestrator.py`, `skills/sub_agents/`: calendar, email, news, system, weather) on Claude Sonnet 5.5 and Haiku 4.5 and merge their results; on a local-only install there is no fan-out and the normal turn answers.
- **Deterministic safety nets** — a meal-advice question gets a suggestion (`core/advice_fallback.py`), a joke request gets a joke (`core/joke_fallback.py`), and a reply claiming an action nothing ran is re-prompted (`core/claim_validator.py`). A result is never spoken before the action that produces it has returned, a chain never re-runs its own search or opens a second tab, a chain that ends on a promise is closed out honestly, and stale "Also, …" offers from a finished chain expire.
- **"It" with nothing to mean** — "turn it off" with no device, media or recent line for "it" to refer to gets a question instead of a guess (`core/pronoun_switch.py`); the action-name corrector never maps a guessed name onto a destructive or self-terminating action, nor onto the opposite of what you asked (`core/action_risk.py`, `command_autocorrect.py`).
- **Replay** — "do that again" / "replay that" → `replay_last_action` (non-destructive actions only).

### Category 25: Talking-device companion (generic support)

A desk robot companion over Wi-Fi is driven by a private skill that is not in this repo. The public tree supplies what any talking-device skill builds on:

- **Scripted dialogue** (`core/dialogue.py`) — a short back-and-forth between JARVIS and a talking device, one line at a time, always ending on a JARVIS line; stop phrases such as "that's enough", "never mind" or "stop" end it. Tuned by the `DIALOGUE_*` settings.
- **Utterance routes** — a skill can claim an exact request ("talk to the <device> about pizza") before the model; `SKILL_ROUTES_ENABLED` turns routes off.
- **After-reply hooks** (`register_after_reply`) — a skill sees each owner turn's reply after it is spoken (for example to let the device react).
- **Known-device speech filter** (`core/device_speech_filter.py`, `DEVICE_SPEECH_FILTER_ENABLED`) — lines a device is known to say (phrase lists in the gitignored `data/device_phrases/`) never command JARVIS and are never learned; owner confirmations and stop words always get through.
- **Phone pings** — a device skill can ask for one (`skill_utils["ping_phone"]`, `PHONE_PING_ROBOT`); see Category 21.
- **Dashboard panels** (`core/web_panels.py`) — a skill declares a panel with live state, controls and a pinned emergency stop.

### Category 26: Speech recognition, speed work & turn telemetry

The "speed plan" work: every change ships off or in a log-only shadow mode and is judged by the per-turn timing line before it is switched on.

- **Turn-timing telemetry** (`core/turn_timing.py`) — every voice or typed turn prints one `[turn-timing]` line with millisecond stage marks (end of speech, Whisper, the model call and its prompt-cache figures, actions, synthesis, first audio) plus the measured real end of speech (`TURN_TAIL_PROBE`, a small Silero detector run after the clip) and playback-open time (`TURN_PLAY_OPEN_PROBE`); both probes are log-only and on. `python tools/turn_latency_report.py <logs>` prints p50 / p90 per stage from those lines (counts only, never transcripts).
- **Parakeet speech-to-text** (`core/stt_parakeet.py`; off by default - settings `STT_ENGINE`, `STT_SHADOW`) — NVIDIA Parakeet TDT 0.6B v2 (int8 ONNX, CPU only) for the owner's commands and the standby wake checks, roughly 0.15–0.3 s per command instead of ~1.7 s. `STT_ENGINE = "parakeet"` uses it, with Whisper kept loaded as the fallback (an empty transcript, or one the wake gates would drop, is decoded again by Whisper; any error switches back for the session). `STT_SHADOW = "parakeet"` is the A/B mode: Whisper keeps transcribing and Parakeet re-decodes each command afterwards while JARVIS is idle, writing both engines' gate verdicts to the gitignored `data/stt_ab.jsonl` (words only for a line JARVIS would act on, nothing while the mic is muted, audio never saved).
- **Whisper decode knobs** — `WHISPER_TEMPERATURES` (`None` = faster-whisper's own ladder) and `WHISPER_BEAM_SIZE` (5) cap the slow fallback decodes on the owner's turns; the defaults leave behaviour unchanged.
- **Smart Turn end of turn** (`core/endpointing.py`, `SMART_TURN_MODE`, `shadow` by default) — an ~9 MB audio model hears the last 8 s of a capture in a real pause and says whether you have finished, instead of always waiting out the fixed 1.3 s of silence. In `shadow` it only logs when it would have ended the turn (`[eot-shadow]`); `on` lets it end turns; `off` loads nothing. A missing model file means today's behaviour.
- **Kokoro speed** (`core/kokoro_tts.py`, `core/tts_render_cache.py`; off by default - settings `KOKORO_PERSISTENT_PHONEMIZER`, `KOKORO_RENDER_CACHE`) — one persistent espeak phonemizer instead of a fresh one per line, and a render cache for repeated lines (`shadow` logs would-hits; `on` serves them; the text itself is never stored, `KOKORO_RENDER_CACHE_PERSIST` keeps renders across restarts).
- **Local prompt cache** — the local route keeps its system prompt byte-identical between turns (only the turn's sections ride the user message), holds post-turn prompt rebuilds back while you are talking (`PROMPT_FREEZE_QUIET_S`), defers non-urgent background model work mid-conversation (`LOCAL_BACKGROUND_MAX_DEFER_S`) and re-primes the model afterwards and at boot, so a turn rarely pays a full prompt re-read.
- **Answer first** (`ANSWER_FIRST_ENABLED`, on) — a short pure-acknowledgement lead-in ("One moment, sir.") is skipped when the action itself speaks the real answer.
- **Turn checker** (`core/turn_checker.py`, decision only, not wired into turns yet) — decides when a local turn failed in a way a single Claude retry would fix (an invented action name, a claimed action nothing ran, a clear command with no action). `python tools/turn_checker_report.py <logs>` counts its verdicts over real session logs so the threshold can be judged before it is wired.

---

## Special voice patterns / proactive behaviors

- **Time-aware wake greetings** — JARVIS picks a wake response from time of day, recent wake frequency, gaze, print state and tone:
  - 01:00–05:00 with ≥3 wakes in 10 min → "Still up, sir?" (only while `NIGHT_QUIET_ENABLED` is on)
  - 05:00–12:00 first wake of the day → "Good morning, sir."
  - Bambu actively printing → "At your service — the print is at 47%, by the way."
  - Out of view on every camera → quieter "Yes, sir?"
  - Otherwise one of 12 phrases tagged formal / terse / playful / soft / general.
- **Barge-in** — say "JARVIS" while he's talking to cut TTS mid-sentence, on speakers as well as headsets. Requires the wake-word detector to be running (`wake_listener_start`); it does not autostart, so talking over him does nothing on its own. Refused while the sentence being spoken contains "jarvis" (echo safety). The legacy loudness/headset watcher is hard-disabled (its mic-stream teardown crashed the audio stack).
- **Presence hold** (on by default - setting `PRESENCE_HOLD_ENABLED`) — queued proactive lines wait while you are away or the room is talking (another conversation, someone else's speech); your own reminders (timers, schedules, promises) and guard alerts still speak. When you are back, stale status lines are folded into one short "While you were away" recap (`core/owner_presence.py`).
- **Goodnight = overnight standby** — any bedtime phrasing fires `start_overnight_upgrade` and silences JARVIS until morning; the improvement engine itself runs only when `OVERNIGHT_UPGRADE_ENABLED` is on.
- **Shutdown asks first** — the ambiguous shutdown phrases are intercepted before the model so JARVIS can ask "overnight first?"; a short yes or "overnight" runs overnight, "no" / "just shut down" powers off, and a hedged no never powers off.
- **Workshop auto-engagement** — opening any CAD/slicer app drops TTS volume 30%, shortens replies and engages focus mode for an hour (released when the app closes).
- **Session resume greeting** — a restart within 18 h of the last session (but not within 15 minutes of it) offers to pick the thread back up.
- **Promises ("I'll let you know when…")** — skills register a deferred announcement via `skill_utils["make_promise"]` (e.g. "tell me when the print finishes"); list or cancel them by voice.
- **Phone pings** — when you are away from the desk, what needs you goes to your phone instead (Category 21).
- **Processing filler** (off by default - setting `PROCESSING_FILLER_ENABLED`) — "Just a moment, sir." while a slow voice turn thinks; Kokoro backend only.

---

## High-risk actions (require confirmation)

Hard-blocked until a short, clear spoken yes ("yes", "go ahead", "do it") — matched in action name + arg (`CONFIRM_KEYWORDS`): **purchase, buy, pay, checkout, delete, format, transfer**. One yes/no classifier (`core/yes_no.py`) serves every prompt: a hedged yes ("do it later", "yes, but wait") is a no, and a sentence that merely starts with "yeah" is not an answer. A saved settings file can add keywords but never remove these.

Additional pushback / refusal layers:

- **`clear_tasks`** — wipes the whole task queue; held for confirmation.
- **`forget_last_hour` / `reset_memory`** — both come back for a spoken "yes"; `reset_memory` backs up first and refuses to wipe if the backup fails.
- **`run_shell` with destructive patterns** — `rm -rf <dir>`, `remove-item -recurse`, `rmdir /s`, `del /s|/q|/f`, `git reset --hard`, `git clean -fd/-fx`, `git push --force`, `drop table`, `drop database`, `truncate table` — held for "yes".
- **`run_shell` hard blocklist** — `format`, `shutdown`, drive- and home-wiping `rm -rf` forms, `sudo rm`, `sudo dd`, `diskpart`, `mkfs`, `taskkill` of its own host, … — REFUSED outright (cannot be confirmed away).
- **Self-preservation** — forbidden from killing its own host:
  - `close_window` where the title contains `powershell`, `python`, `terminal`, or `bobert`
  - `alt+f4` while PowerShell / python is focused
  - clicking "close button on PowerShell" via vision
  - Refused with an explanation, no retry.
- **Self-terminating actions** — restart, shutdown, upgrade and overnight run from the model's reply only when your own words asked for them, and are never reached by guessing a misspelt action name.
- **Sketchy URLs in `open_url`** — bare-IP HTTP, free-phishing TLDs (.tk/.ml/.cf/.gq/.top/.xyz/.click/.country/.zip/.mov), `.onion`, tunnel hosts (ngrok / trycloudflare / loca.lt / serveo), link shorteners (bit.ly / tinyurl / goo.gl / t.co / is.gd / buff.ly / ow.ly / rebrand.ly) — pushback before opening. Local/LAN URLs whitelisted.
- **Bulk window-close with unsaved work** — pushes back when titles look unsaved (`*` / `●` / `•`, "Untitled", "(modified)") or are live-edit apps (VS Code, Cursor, Google Docs/Sheets).
- **Outbound messages** — every send goes through the draft-preview gate (read aloud, explicit confirm).
- **Dashboard actions** — running a side-effect or destructive action by name from the web Actions view needs an explicit confirm.
- **`replay_last_action`** — refuses destructive actions (close_window, restart, upgrade, start_overnight_upgrade, run_shell); re-issue them so they pass the normal confirmation path.
- **Skill code is forbidden** from including final purchase / payment confirmation steps (must stop one step BEFORE the money-spending click), and the browser agent's `fill_form` never submits.

---

## Key files

- `bobert_companion.py` — main loop, the `ACTIONS` dict, skill loader, wake/sleep handling, LLM dispatch (`_call_llm`)
- `core/actions.py` — most core action handlers (late-bound to the monolith)
- `core/prompts.py` — `BASE_SYSTEM_PROMPT` and the sectioned `PC_CONTROL_PROMPT` (every routing example)
- `core/prompt_router.py` — per-turn section selection for the local brain; `core/prompt_budget.py` — keeps the local prompt inside the context window
- `core/config.py` — shipped defaults; `data/user_settings.json` (gitignored) overrides them
- `skills/` — 99 loadable skill modules (plus `_example_skill.py`, which the loader skips)
- `hud/` — the overlay subprocesses, driven by `hud_state.json` and per-overlay control files (the old corner ring `hud/jarvis_hud.py` is no longer launched; `hud/globe_geometry.py` is the globe's geometry helper)
- `tray.py` — system-tray applet; `tools/settings_window.py` — Settings window; `tools/web_interface.py` — web dashboard
- `docs/ACTION_INDEX.md` — machine-generated per-action table (`python tools/gen_action_index.py`)
- `tools/turn_latency_report.py`, `tools/turn_checker_report.py` — read-only reports over session logs
- `upgrade_jarvis.py` — voice-triggered Claude Code pipeline
- `overnight_upgrade.py` — idle-watch autonomous improvement loop
- `iron_man_boot.py` — boot-sequence sting + HUD animation + spoken greeting
- `jarvis_todo.md` — task queue Claude Code consumes
- `hud_state.json`, `hud_card_state.json`, `pending_speech.json` — inter-process state
