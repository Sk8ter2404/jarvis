# Release history

Every tagged release, newest first. Generated; do not edit by hand.

## v2.0.209 (2026-10-16)

Hand tracking on the Kinect no longer runs backwards: the Kinect's mirrored picture (and its depth map) is flipped back once before the hand tracker sees it.

## v2.0.208 (2026-10-16)

Memory clean-up calibrated for the new memory search model: duplicate and 'might contradict' checks use cut-offs measured on your own facts (0.97 / 0.50 on voyage-4-nano), so the contradiction check works again and nothing is deleted that the old model would have kept.

## v2.0.207 (2026-10-15)

Fixes from tonight's log review: playing an artist works again (no such playlist -> played as music; the Apple Music song page now presses its Play button or Space when the song row can't be seen; 'play it on YouTube' plays instead of only searching), 'YouTube, <search>' on a monitor no longer runs as a program, a song name sent to the vibe action is played, 'delete / close that website' closes the tab in front, the TV detector stops flickering on people moving at the desk (sustained evidence until a TV area is calibrated, 20 s hold), and no Hugging Face progress bars in the log.

## v2.0.206 (2026-10-15)

Test-only (no change to the running code): the last hardware-dependent test now fakes the Windows default microphone, so the full suite passes completely on this PC (25,111 of 25,111) - which the safe self-upgrade's gate needs.

## v2.0.205 (2026-10-15)

Code health, no behaviour change: helper functions that existed as identical copies in several skills (window titles, sleep/standby checks, desk presence, Kinect bridge lookup, protected-data decryption, small text formatters, a JSON writer) now live once in shared modules, so a future fix lands everywhere at once.

## v2.0.204 (2026-10-15)

Hologram phase 3, ready for a headset: a mixed-reality page (/holo/xr) for the Quest 3 browser with passthrough and the headset's own hand tracking - pinch to grab and move the reactor or your own models in 3D, two hands to scale and turn, pull apart to explode, clap to rebuild, pinch-and-hold a part for its name and size. It needs a secure address to run on the headset; until one is set up the page explains the two ways to get one, and 'open the hologram on the Quest' says the same.

## v2.0.203 (2026-10-15)

Hologram phase 2, off by default: 'open the hologram with the Kinect' uses the Kinect's colour camera for hand tracking and its depth sensor so the hologram hides behind your real hands when they are in front of it, and a fist pushed toward or pulled away from the sensor moves it nearer or farther. It reads the Kinect only through JARVIS's existing connection. A new test fails if any module ever defines the same function twice (a scan found no stale duplicates today).

## v2.0.202 (2026-10-15)

Fixes from the self-upgrade queue review: the local brain adopts a better model once it finishes downloading (checked at most every 5 minutes, between turns); turning the workshop hologram off hides it so 'on' brings it back; web pages opened by 'open', 'search' and 'YouTube' land on a monitor instead of off-screen; a stale memory note corrected. Three queued items were already fixed and one was not a bug.

## v2.0.201 (2026-10-15)

The self-upgrade is rebuilt to be safe (still switched off). It runs only when asked, in a separate git worktree off the published code, never touching or restarting the running JARVIS. The editing step cannot push, tag, merge or change data files; the planning and review steps are read-only. Each task is its own commit, a failing task is reverted, the full test suite and the privacy check run at the end, and the result is a branch plus a report for review - nothing ships without a normal release. Overnight ideas go to a proposals file, never straight into the work queue.

## v2.0.200 (2026-10-15)

Hand mouse (air mouse v2) on the new hand tracker, off by default: a deliberate pinch held still takes the cursor, a quick pinch clicks, a second held pinch drags, an open palm or the hand leaving lets go, and it never engages in games, fullscreen apps or calls. Every brain call now says who is calling and whether it is a live answer or background work (a test fails on any untagged call). The wake-word listener follows the current microphone when the default input changes. Old session logs are compressed into an archive instead of deleted. The every-action sweep counts its own sandbox blocks separately from real crashes. Fixes a Linux-only test failure from v2.0.199.

## v2.0.199 (2026-10-15)

A hard stop before irreversible clicks: when screen control is about to click a button that sends, submits, posts, publishes, pays, buys, transfers, deletes or accepts terms - or press Enter in a message box that would send - it asks first and acts only on an explicit yes for that one action. The hologram can show your own 3D models (STL, 3MF, OBJ from chosen folders): a 3MF's objects become named parts that explode and show their size, and "show me the <model>" picks one by name. Model files are served only by id from the configured folders.

## v2.0.198 (2026-10-14)

Claude Code jobs can run on a second Claude subscription first. With a last-day date set, every Claude Code job JARVIS starts (upgrade loop, multi-agent pipeline, dry audit, overnight ideas) runs on a second Claude Code profile - its own sign-in folder - through that date, and on the main account whenever the second one is at its 5-hour or weekly limit, switching back when the limit resets. The sign-in stays inside Claude Code's own folder; JARVIS never reads or stores a token. The upgrade pipeline also recognises Claude Code's current limit wording again.

## v2.0.197 (2026-10-14)

Test tooling only (no change to the running code): the GitHub CI job's time limit goes from 20 to 40 minutes. The job runs the whole suite twice (plain, then with coverage) and had grown to the edge; v2.0.196's run passed its tests and was cut off mid-coverage at exactly 20 minutes.

## v2.0.196 (2026-10-14)

The point-and-flick gesture works at the webcam's real frame rate. Live on v2.0.195 the tracker got about 12 frames a second: the webcam can do 30 but slows itself in room light, and frames arrive in short bursts. The flick now adapts its time window to the measured frame rate (the speed and angle bar is unchanged, so jitter and slow swings still never count). 'Is hand tracking on' mentions when more light would help.

## v2.0.195 (2026-10-14)

The hand tracker now gets the camera's full frame rate: live on v2.0.194 it received only about 6 frames a second (the face-tracking loop's own pace), too few for a quick flick. While the tracker runs, the camera loop keeps reading the tracker's camera for a moment after each normal read and hands every frame over; face tracking keeps its cadence, and nothing changes while the tracker is off. Voice: 'open up the hologram', 'close the hologram' and 'hand menu' work without the language model; 'show me the arc reactor', 'explode it' and 'put it back together' go to the hologram only while it is open, so a plain 'show the arc reactor' stays the small corner reactor.

## v2.0.194 (2026-10-14)

Hand tracking v2, phase 1: a separate MediaPipe hand tracker (21 points per finger-level hand, about 29 frames a second) reads the camera frames JARVIS already captures, and a hologram window (a three.js arc reactor) is driven by it: pinch and hold for the menu, a fist to move, point and flick to spin, two fists apart to explode it into labelled parts, pinch a part and hold it for its info, clap to rebuild. Two views: a solid dark 'hand menu' with no camera picture, and 'open up the hologram' with the mirrored camera behind it. The hologram starts the tracker it needs and stops it when closed. 'Hand tracking on' now means this tracker, which never moves the cursor (it used to switch on the Kinect air mouse); 'hand tracking off' also stops the air mouse. All of it is off until asked for.

## v2.0.193 (2026-10-14)

Test tooling only (no change to the running code): the test watchdog's kill now really ends a wedged test run on Windows (TerminateProcess instead of os._exit, which could hang on leftover CUDA/COM threads - one run sat 26 hours), tests can no longer fake its clock, and a new gate wrapper ends a test run that outlives the watchdog.

## v2.0.192 (2026-10-14)

A weekly success score ('how did you do this week': turns, answer rate, median response time, failing actions, from this PC's logs only). The smart-home command now has exactly one owner (a duplicate registration that only lost by alphabetical load order is gone) and a test fails if any action is ever registered twice. An echo-cancellation fallback for protected video audio (ducks and tightens the wake gate when the echo reference is missing while media plays), shipped log-only.

## v2.0.191 (2026-10-14)

Direct smart-home control on the home network, with the voice-assistant cloud only as the fallback: Smart Life / Tuya lights and switches (the right on/off and brightness data points, confirmed by reading the device back), Magic Home LED controllers (on/off, brightness, colour, confirmed), and the BroadLink IR/RF hub (learns each remote button once by voice, then sends it locally; honest about toggle codes). Router and discovery brand tables are pinned to agree.

## v2.0.190 (2026-10-13)

Smart-home commands that go through the voice assistant cloud now use the saved phone-app style sign-in (the old cookie is only a fallback), keep one connection open across commands instead of signing in each time, retry once on an expired session, report success only when the cloud confirms it, and pick up a new sign-in or a rewritten device list without a restart.

## v2.0.189 (2026-10-13)

Queued speech can no longer be lost: a line whose speech fails is retried (up to 3 tries) and a queue file cut off mid-write keeps every intact line. The printer self-check reads the live monitor instead of opening a second connection (no more false FAILED). Cameras that drop off USB and come back are counted per camera with one 'check its cable' line. The last four open July-audit findings are fixed (overnight yes is answered, empty speech no longer clears another line's voice style, the local action list follows new actions, wake-word pause drops leftover audio). Dashboard accessibility fixes, an up-to-date FEATURES.md, and a test-order leak fixed.

## v2.0.188 (2026-10-12)

Test-only lint fix: clears the unused imports and one unused variable that made the GitHub CI lint step fail on v2.0.187. No change to the running code.

## v2.0.187 (2026-10-12)

Two-way chat bot for alerts and commands; a shopping list and a separate reminders list (reminders with no time never land in the developer queue); optional comments about what is on screen; easier air-mouse engage and useful Kinect gestures; batched background learning; game mode, TV and clap tweaks plus an order tracker; privacy guards on test data; repo cleanup (one daily briefing skill).

## v2.0.186 (2026-10-12)

Alexa sign-in rebuilt on the phone-app style OAuth device registration (sign in once, saved token, the real smart-home device list, no session secrets in logs); audio auto-switch waits while media plays; CLOUD_BOOST_UNTIL sends hard turns and screen reading to Claude until a date; PROMPT_SLIM_MODE keeps the cached brain prompt smaller.

## v2.0.185 (2026-10-12)

Public status docs on GitHub: docs/STATUS.md and docs/RELEASE_HISTORY.md, generated from the master sheet with all personal data stripped. No code change; the running JARVIS code is the same as v2.0.184.

## v2.0.184 (2026-10-12)

JARVIS only announces a finished print he actually watched run (no "print complete" for a print finished while he was off, none when a new print's name arrives before the printer leaves FINISH, no replay after a restart, a stale queued finish line is ...

## v2.0.183 (2026-10-09)

"turn on/off hand tracking" works, "close that window" closes one, "u2"/"you two" means YouTube only inside open/close/play commands, a bare website name opens the site (no exe launch), the cold-start warm-up message tells the truth; "why can't I hear X" runs ...

## v2.0.182 (2026-10-09)

JARVIS shuts down when asked in plain words ("shut down, no overnight protocol", "yes, that is what I want") and still never on someone else's words; screen vision rebuilt (reads real buttons and links through UI Automation, OCR as backup, checks every click ...

## v2.0.181 (2026-10-05)

a named close closes that app (not "what I opened"), close-all keeps the running app and says when a title alone kept a window, "you forgot X" closes X, a misheard keep/close name gets "did you mean ...?", JARVIS never clicks or types on a sign-in page unless ...

## v2.0.180 (2026-10-04)

fixes the v2.0.179 native abort - the CUDA driver starts before ctranslate2 again, and every CTranslate2 CUDA build, decode and free runs on one never-exiting host thread (a short-lived self-check thread exiting with CTranslate2 CUDA state aborted the ...

## v2.0.179 (2026-10-04)

one voice per reply (the clone never switches mid-reply, a sentence prepared ahead waits until it is needed, a long first line is split at a clause, misses cool down instead of switching the clone off for the session), steady Kinect hand tracking (one shared ...

## v2.0.178 (2026-10-03)

the cloned JARVIS voice runs from its own GPU server with Kokoro covering any line it misses (off until switched on), and "close all windows except X" really closes the rest - never JARVIS's own HUD/Reticle or the Windows shell, asks first above five, never ...

## v2.0.177 (2026-10-02)

a yes to JARVIS's own offer carries it out, slow apps are moved when their window appears, streaming search uses verified links and stops at a sign-in wall, 'close that' closes only what he opened, fresh 'diagnostic status', facts remember who said them + ...

## v2.0.176 (2026-10-02)

the big 10-02 batch - JARVIS never voices a claim he didn't carry out, honest about admin windows and repeated failures, instant time; turn checker + instant actions (shadow), answer pre-render (R3), July audit fixes, local-first features with a retired-model ...

## v2.0.175 (2026-10-02)

'power off the tv' never arms JARVIS's own shutdown, a reply saying 'always listening' never leaves wake-word mode, and standby wakes only on the name or a bare wake phrase (July audit A28, A37, A29)

## v2.0.174 (2026-10-02)

speed plan R7 - Smart Turn listens to every owner turn in shadow mode and logs when it would have ended it (SMART_TURN_MODE=shadow; 'on' is the owner's call)

## v2.0.173 (2026-10-02)

the lyric detector skips 64 ms mic scraps; a browser tab closed mid-reply logs one line, not a traceback

## v2.0.172 (2026-10-02)

phone-ping tests stay green on a freshly booted CI runner

## v2.0.171 (2026-10-02)

fix(ci): the Linux memory ceiling bounds memory in use, not glibc's per-thread reservations

## v2.0.170 (2026-10-02)

speed plan R6 - Parakeet TDT speech-to-text on the CPU for the owner's commands, with a shadow A/B mode (both off by default)

## v2.0.169 (2026-10-02)

speed plan R11 - Whisper beam/temperature knobs (defaults unchanged)

## v2.0.168 (2026-10-02)

fixes from the 10-01 live conversations - no made-up results spoken before a lookup returns, media audio can't command JARVIS, honest chain close-outs, a presence hold for proactive lines, the hotword-echo re-decode

## v2.0.167 (2026-10-02)

test(monolith): "hours ago" monotonic stamps stay valid on a freshly booted PC

## v2.0.166 (2026-10-02)

the dashboard shows what JARVIS did and gathers every proactive switch on one page; a turn checker decides when a local turn deserves a Claude retry (not wired yet)

## v2.0.165 (2026-10-02)

the to-do backlog - privacy scrubs, a safety-list floor, crash-safe overlay files, honest camera and VAD-stall checks, Apple Music playlists by name, a bounded STT retry

## v2.0.164 (2026-10-02)

test(turn-timing): the omitted-flag check removes the flag instead of assuming it is undefined

## v2.0.163 (2026-10-02)

test(chappie): today's recall fixtures sit at noon, not "now - N s"

## v2.0.162 (2026-10-01)

after-reply skill hooks (unblocks the robot banter mode) and ambient Whisper that can yield to the owner's turn (off by default)

## v2.0.161 (2026-10-01)

speed plan R1 - every turn line now measures the real end-of-speech wait, the STT lock wait, playback open time and the filler clip; boot warmers; a latency report tool

## v2.0.160 (2026-10-01)

morning chain wakes on the day's first real turn, the air-mouse yield hook installs on 64-bit Python (off by default), the credit check survives Chrome swapping its hidden window

## v2.0.159 (2026-10-01)

Revert "test: pyflakes-clean three test files (pre-existing on the 157 base)"

## v2.0.158 (2026-10-01)

test(anticipation): pin the monotonic clock ,  v2.0.157's CI went red on a fresh Linux runner

## v2.0.157 (2026-10-01)

the 2026-10-01 bug-hunt release - nine fix batches, controls that do what they report, live-log defects, merge-audit repairs

## v2.0.156 (2026-10-01)

"use the Kinect again" loads the prompt section that documents the retry

## v2.0.155 (2026-10-01)

the owner's speech vocabulary and named site shortcuts (STT_HOTWORDS, STT_REPLACEMENTS, SITE_SHORTCUTS)

## v2.0.154 (2026-10-01)

the monitor actions take the form the local model writes; "main monitor" is the primary display; a YouTube search opens on a monitor

## v2.0.153 (2026-10-01)

a banter never fails on its first line; "use the Kinect again" retries it; the repeat quip is for questions only

## v2.0.152 (2026-09-30)

owner-only learning - other people in the room no longer teach JARVIS facts about the user

## v2.0.151 (2026-09-30)

small backlog - overlay fallbacks match config, more live state guarded, desktop launchers guarded, DIALOGUE_LOST_HOLD_S wired, Runner closings, one DialogueUnavailable

## v2.0.150 (2026-09-30)

hermetic test guards (network, input, hardware probes); web-started dialogues get the mic; no "Done." after a confirmed self-voiced action; panel image placeholder

## v2.0.149 (2026-09-30)

the monolith test tier is green on the owner's PC; power-mode aliases speak; presence everywhere needs a sustained face

## v2.0.148 (2026-09-30)

next-weekday dates, a world clock, "who am I", "what did I ask first", repeat requests are not frustration

## v2.0.147 (2026-09-30)

one switch for all clock-driven night quieting (NIGHT_QUIET_ENABLED); NIGHT_OWL_AUTO

## v2.0.146 (2026-09-30)

Settings window fixes - readable, honest, never clobbers the settings file

## v2.0.145 (2026-09-30)

web dashboard fixes, a live Actions/controls surface, and a generic skill-panel hub; action index no longer leaks private skills

## v2.0.144 (2026-09-30)

tray overhaul - a clickable Settings, live checkmarks, visible results, confirmations, instant mic mute

## v2.0.143 (2026-09-30)

no confident lead-in before a failure; camera lift says what it lifted; tests never touch live state; log spam filtered

## v2.0.142 (2026-09-30)

proactive remarks need a real presence, never copy an example, never repeat; repeated-token noise is dropped

## v2.0.141 (2026-09-30)

skills can claim exact requests before the LLM (utterance routes)

## v2.0.140 (2026-09-29)

a different request is not "repeating himself"

## v2.0.139 (2026-09-29)

the first turn after a start is as quick as the rest

## v2.0.138 (2026-09-29)

no mic hot loop when the input device vanishes; noise "Bye." is not a turn

## v2.0.137 (2026-09-29)

a camera that dies on every open is retried slowly, and said once

## v2.0.136 (2026-09-29)

generic device-dialogue hooks; fallbacks log what was spoken

## v2.0.135 (2026-09-29)

a request for a suggestion gets one (advice fallback)

## v2.0.134 (2026-09-29)

camera culprit quarantine, JARVIS ignores his own voice, open-ended answers get delivered

## v2.0.133 (2026-09-29)

webcam privacy-log paths parse as Windows paths on every host; fast-path replies are logged; the driver stops waking JARVIS

## v2.0.132 (2026-09-29)

fix the CI coverage crash (a leaked test thread re-ran the monolith's stdio reconfigure)

## v2.0.131 (2026-09-29)

one camera gate for every opener; USB-storm circuit breaker

## v2.0.130 (2026-09-29)

the background Teams nudger is opt-in (TEAMS_NUDGE_ENABLED, default off)

## v2.0.129 (2026-09-29)

learned topics need real evidence; "what am I working on" comes from the owner's own list

## v2.0.128 (2026-09-29)

background gate tracks its holder by thread object, not a reusable ident

## v2.0.127 (2026-09-29)

audio-device flap damping; never open a stale MME index

## v2.0.126 (2026-09-29)

background brain work waits for a quiet moment; the cache is re-warmed after it

## v2.0.125 (2026-09-29)

load upgrade_jarvis by path; never touch sys.path (CI flake, for good)

## v2.0.124 (2026-09-29)

exact date math and instant answers; no false retractions or double answers

## v2.0.123 (2026-09-29)

tomorrow's weather, the real live microphone, conversation summaries

## v2.0.122 (2026-09-29)

match the opener to the content

## v2.0.121 (2026-09-29)

run-jarvis driver captures the real answer of informative actions

## v2.0.120 (2026-09-29)

per-sentence speech -- start talking after the first sentence renders

## v2.0.119 (2026-09-29)

answer first -- drop the model's filler lead-in when an action speaks the answer

## v2.0.118 (2026-09-29)

the run-jarvis driver never boots over a live JARVIS; CI flake fixed

## v2.0.117 (2026-09-29)

speech from known devices never commands JARVIS

## v2.0.116 (2026-09-29)

keep the local prompt prefix byte-stable during a conversation

## v2.0.115 (2026-09-29)

generic header words and a bare "still" no longer misroute turns

## v2.0.114 (2026-09-29)

one [turn-timing] line per turn, and Ollama stats on every call

## v2.0.113 (2026-09-29)

"Processing, sir." -- the owner's wording, ready from the first turn

## v2.0.112 (2026-09-29)

fix the test_bambu_camera CI flake at its cause

## v2.0.111 (2026-09-29)

"just a moment, sir" while a slow voice turn is processing (off by default)

## v2.0.110 (2026-09-29)

the ambient privacy blocklist could never match a process name

## v2.0.109 (2026-09-28)

the Dell fixed one of six probes that false-fail against a remote brain

## v2.0.108 (2026-09-26)

the last 3 CI failures were all "passes only on the author's machine"

## v2.0.107 (2026-09-26)

GitHub CI has been red for 38 releases, and the local suite could never see why

## v2.0.106 (2026-09-26)

a fresh `pip install -r requirements.txt` installs a broken OpenCV

## v2.0.105 (2026-09-26)

a 2.0s timeout that never returned, and three tests that agreed with each other instead of the product

## v2.0.104 (2026-09-06)

the invariant held ,  and it was the wrong invariant

## v2.0.103 (2026-09-06)

the near-miss defect recurred three times, one release after "fixing" it

## v2.0.102 (2026-09-06)

the camera was never sick ,  JARVIS was breaking it every 30 minutes

## v2.0.101 (2026-09-05)

DirectShow camera enumeration leaks 1 thread + 103 handles per call, forever

## v2.0.100 (2026-09-05)

the camera loop never died ,  it wedged, holding the lock, for 2 hours

## v2.0.99 (2026-09-05)

the microphone half of the same lie ,  JARVIS was listening to a powered-off headset

## v2.0.98 (2026-09-05)

being IN the prompt is not the same as REACHING the model

## v2.0.97 (2026-09-04)

the fix was in the wrong copy ,  six features that were already written and never reached

## v2.0.96 (2026-08-20)

the guards were lying ,  an adversarial review of v2.0.94/95 found 28 defects in them

## v2.0.95 (2026-08-20)

stop reporting success that was never verified

## v2.0.94 (2026-08-20)

honest failure ,  close 10 verified defects + make the test suite unable to damage the machine

## v2.0.93 (2026-07-21)

close all 51 verified defects from the 2026-07-21 adversarial audit

## v2.0.92 (2026-07-21)

a crash no longer disables the watchdog + staging really isolates state

## v2.0.91 (2026-07-21)

local brain no longer bricked by its own vision calls

## v2.0.90 (2026-07-15)

Kinect ghost-skeleton suppression (adversarially-audited)

## v2.0.89 (2026-07-15)

air-mouse raise-hand gate + local-prompt header fix + env-only secrets

## v2.0.88 (2026-07-15)

unify memory ,  sync learned facts into the semantic store + pin embedder to CPU

## v2.0.87 (2026-07-15)

gesture (Kinect-swipe) barge-in OFF by default ,  fixes the crash-twice-today bug

## v2.0.86 (2026-07-15)

promote local brain to gemma4:26b-a4b (P2 phase B) + VRAM-gate the clone

## v2.0.85 (2026-07-15)

add 'kokoro' to the TTS_BACKEND settings enum

## v2.0.84 (2026-07-15)

add CPU Kokoro TTS backend (P2 phase A ,  frees the 3090)

## v2.0.83 (2026-07-15)

fix prompt-router header regex ,  was hiding ~42 capabilities from the local brain

## v2.0.82 (2026-07-15)

ignore non-acoustic barge-ins during boot grace (fix boot-greeting native crash)

## v2.0.81 (2026-07-15)

correct stale gemma4:26b 'broken quant' comment (on-box re-bench)

## v2.0.80 (2026-07-15)

anchor JARVIS identity in BASE prompt (stop vendor-identity bleed)

## v2.0.79 (2026-07-15)

P1: dynamic local prompt (fix the truncated-brain linchpin)

## v2.0.78 (2026-07-14)

bug-hunt batch 18: the tail (#24, #25, #26 + #12, #23) ,  backlog cleared

## v2.0.77 (2026-07-14)

bug-hunt batch 17: scene activation + email bound + audio stats (#11, #16, #17)

## v2.0.76 (2026-07-14)

bug-hunt batch 16: core routing (#9, #10, #20)

## v2.0.75 (2026-07-14)

bug-hunt batch 15: camera/device reliability (#3, #4)

## v2.0.74 (2026-07-14)

bug-hunt batch 14: voice-loop correctness (#2, #5, #18)

## v2.0.73 (2026-07-14)

bug-hunt batch 13: media/streaming confirmation (#13, #14, #22)

## v2.0.72 (2026-07-14)

bug-hunt batch 12: local-brain routing cluster (#6, #7, #19)

## v2.0.71 (2026-07-14)

fresh bug-hunt: both HIGH (incl. a security fix) + stale-duplicate cluster

## v2.0.70 (2026-07-14)

audit batch 10: dev-tooling + guards (closes the confirmed-open set)

## v2.0.69 (2026-07-14)

audit batch 9: honest results + user-facing correctness

## v2.0.68 (2026-07-14)

audit batch 8: four ways the voice thread could hang or crash

## v2.0.67 (2026-07-14)

audit batch 7: two bugs that made JARVIS ignore you

## v2.0.66 (2026-07-14)

audit batch 6: the stale-gate / dead-config cluster

## v2.0.65 (2026-07-14)

audit batch 5: five silent-wrong-behaviour findings

## v2.0.64 (2026-07-14)

LLM-independent control plane: you could not restart JARVIS when the brain was down

## v2.0.63 (2026-07-14)

audit batch 4: the PII guard failed OPEN, guard mode couldn't work, a blind toggle paused the video it started

## v2.0.62 (2026-07-14)

teardown must WAIT for in-flight GPU work (a restart 83s into boot still corpsed)

## v2.0.61 (2026-07-14)

audit batch 2+3: the camera "failed to open" mystery was a lock-queue billing bug; the chat path had no wedge detector; the ollama backend was DOA

## v2.0.60 (2026-07-14)

audit batch 1: two ways to destroy the owner's browser window, a heap-corruption TOCTOU, the last os._exit, a resurrectable Kinect close, and 10 corpse-blind overlays

## v2.0.59 (2026-07-14)

restart spawned NOTHING (my own v2.0.57 ordering bug) + the watchdog was corpse-blind

## v2.0.58 (2026-07-14)

detect a VISION-WEDGED ollama runner (the self-heal was blind to it)

## v2.0.57 (2026-07-13)

release CUDA/Kinect/WASAPI before exit: restarts were leaking ~5GB VRAM per corpse

## v2.0.56 (2026-07-13)

left camera is now the eMeet C960 (Logi C270 retired)

## v2.0.55 (2026-07-13)

the [local-vision] tag made every local YES parse as NO

## v2.0.54 (2026-07-12)

stop noisereduce from pinning the main loop on idle silence

## v2.0.53 (2026-07-12)

pin Ollama to 127.0.0.1: "localhost" cost every local call a hidden ~2.05s

## v2.0.52 (2026-07-12)

youtube verify false-negative, slow-warm camera drop, corpse-held web port

## v2.0.51 (2026-07-12)

kill the immortal-zombie exit paths + authoritative parent liveness + CI-freeze fix

## v2.0.50 (2026-07-11)

sweep denylist tail: youtube_search opened real tabs, keep_music_open pins the app

## v2.0.49 (2026-07-11)

sweep desktop-hygiene (screen is not staged) + local_vision __main__ crash

## v2.0.48 (2026-07-11)

youtube verify was dead config + settings writer ignored staging (2nd live-pollution)

## v2.0.47 (2026-07-10)

action_smoke runs under JARVIS_STAGING isolation (it polluted live settings)

## v2.0.46 (2026-07-10)

youtube_play: deterministic watch-URL activation + honest vision verify

## v2.0.45 (2026-07-10)

ollama gets a HIDDEN console (the last ghost-window source) + action-smoke harness

## v2.0.44 (2026-07-10)

vision-click works on the LOCAL route: retire the two Claude-only gates

## v2.0.43 (2026-07-10)

local cheatsheet advertises youtube_play (the find-and-click autoplay)

## v2.0.42 (2026-07-10)

the by-name rescue had its own stale copy of the KINECT_AS_CAMERA veto

## v2.0.41 (2026-07-10)

camera probe retries reads until its deadline (warming cams failed the single read)

## v2.0.40 (2026-07-10)

preflight by-name rescue gets the retry-pass probe budget

## v2.0.39 (2026-07-10)

KINECT_AS_CAMERA no longer hijacks named webcam entries

## v2.0.38 (2026-07-10)

per-camera live streams on the web Camera tab (owner request)

## v2.0.37 (2026-07-10)

TOTAL AUDIT batch: process-wide no-window net, resurrection watchdog, web UI fixes, camera sides

## v2.0.36 (2026-07-10)

set_volume: support modern pycaw AudioDevice.EndpointVolume

## v2.0.35 (2026-07-10)

full-feature live-battery fixes: set_volume action + wedged-Kinect fail-fast

## v2.0.34 (2026-07-10)

spoken-number normalisation for the clone voice (20.07 read as "two thousand seven")

## v2.0.33 (2026-07-10)

LOCAL BRAIN OVERHAUL: gemma4:12b primary (chat+vision), runtime ollama self-heal, think-param, embedder off GPU

## v2.0.32 (2026-07-10)

kill ghost console windows: CREATE_NO_WINDOW on 9 verified spawn sites

## v2.0.31 (2026-07-10)

local-brain resilience: kill the broken-model landmine + empty-response failover

## v2.0.30 (2026-07-09)

ollama self-heal timeout fix + public-repo voice-file safety

## v2.0.29 (2026-07-09)

fix broken local brain ,  working default model + Ollama self-heal

## v2.0.28 (2026-07-08)

bug-hunt-2 batch ,  ambient-refcount crash regression + 2 leaks + 11 findings

## v2.0.27 (2026-07-08)

stop air-mouse telemetry flooding the log / web Live view

## v2.0.26 (2026-07-08)

camera preflight retry for a Kinect-starved slow USB webcam

## v2.0.25 (2026-07-08)

fix face-track camera index (left webcam is index 2, not the Kinect)

## v2.0.24 (2026-07-08)

scan-tail fixes (29) + imperial distances + real 3-camera roster

## v2.0.23 (2026-07-08)

STT-OOM crash root cause + 3 HIGH crash vectors + currency speech

## v2.0.22 (2026-07-08)

scan batch ,  fix HIGH mic-ownership race (live-crash root cause) + 18 non-web findings

## v2.0.21 (2026-07-08)

web interface full control panel + web security fixes

## v2.0.20 (2026-07-08)

Whisper STT can target a specific GPU (WHISPER_DEVICE=cuda:N)

## v2.0.19 (2026-07-07)

fix LTM-embedder hot-retry churn (stressed-PC bug) ,  stdout tee isatty + loader backoff

## v2.0.18 (2026-07-07)

air-mouse fist-release OFF by default (closing hand no longer stops tracking)

## v2.0.17 (2026-07-07)

fix air-mouse on/off flicker when hand is closed (fist-release latch)

## v2.0.16 (2026-07-07)

harden web interface against localhost CSRF + DNS rebinding

## v2.0.15 (2026-07-07)

full settings control panel on the web interface + wake-word toggle

## v2.0.14 (2026-07-07)

fix camera preview going dark when the primary camera is dropped

## v2.0.13 (2026-07-07)

fix: fabricated version/time behind an [intent:] tag now intercepted

## v2.0.12 (2026-07-07)

voice-pipeline review fixes: clone timeout + focus resume race

## v2.0.11 (2026-07-07)

hand tracking: stop an invalid pose from pre-charging the dwell

## v2.0.10 (2026-07-07)

fix the intermittent flaky web-interface test

## v2.0.9 (2026-07-07)

test hardening: config default-safety test respects owner opt-in

## v2.0.8 (2026-07-07)

Focus Mode: "focus mode" now actually holds notifications

## v2.0.7 (2026-07-07)

Focus Mode (do-not-disturb + recap)

## v2.0.6 (2026-07-07)

bug-hunt batch 3: core internals (history bloat, LTM cascade, promise latch)

## v2.0.5 (2026-07-07)

voicing batch (74 dropped-result actions now spoken + failure-marker contractions)

## v2.0.4 (2026-07-07)

bug-hunt batch 2: mic-ownership race, console flashes, misc hardening

## v2.0.3 (2026-07-07)

web dashboard controls + richer status

## v2.0.2 (2026-07-07)

test-suite cleanup (7 stale local-tier failures)

## v2.0.1 (2026-07-07)

bug-hunt batch: smart-home questions, spoken numbers, diag counts

## v2.0.0 (2026-07-07)

Smart hand tracking (multi-signal engagement + priming + calibration)
