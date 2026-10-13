# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.189
- Live since: 2026-10-13
- Latest change: v2.0.189 (10-13): queued speech never lost, printer self-check false alarm fixed, camera flap counter, last July-audit findings fixed, dashboard accessibility, FEATURES.md current.

## In progress

- Smart home control without the voice assistant cloud: direct control of local plugs and lights, doorbell announcements, IR remote
- Echo-cancellation tuning for hearing over videos
- Pinch-to-click calibration for Kinect hand tracking
- Per-camera flap counter
- Boot proof and long live diagnostics
- Speed plan: enabling each speed feature one at a time with before and after measurements

## Roadmap

Open, planned and idea items. Wording is generic on purpose.

### In progress (4)

- No more bugs on JARVIS or the robot. In progress.
- Prepare the answer while 'processing' plays. In progress.
- Stop the robot repeating the same jokes. In progress.
- Turn checker: retry a bad local answer on Claude. In progress.

### Planned next (3)

- Small engineering follow-ups. Planned next.
- Kinect sees ghost skeletons. Planned next.
- Better memory search models. Planned next.

### Ideas (23)

- Leaner local brain prompt. Idea, not started.
- Speak the first sentence sooner. Idea, not started.
- Stream the local brain's words straight into speech. Idea, not started.
- Trim the gap between 'play' and actual sound. Idea, not started.
- Native tool calling, loading only the tools needed. Idea, not started.
- Whole-conversation tests and a personality-drift test. Idea, not started.
- Memory that updates facts instead of deleting them. Idea, not started.
- Speak first, finish the work in the background. Idea, not started.
- Tell apart several people talking in the room. Idea, not started.
- Long-running missions with check-ins. Idea, not started.
- Fine-tune the brain on its own logs. Idea, not started.
- Turn on by itself when USB is plugged in. Idea, not started.
- Drive safety tests on the stand. Idea, not started.
- Let the robot explore the desk on command. Idea, not started.
- Banter mode that keeps them chatting. Idea, not started.
- JARVIS reacts when the robot is picked up or shaken. Idea, not started.
- Robot mood and 'age', with eyes on the web page. Idea, not started.
- Games where the robot is a sore loser. Idea, not started.
- Distance, gesture and screen-eye modules. Idea, not started.
- Motion emotes (nod, wiggle, sulk-turn). Idea, not started.
- JARVIS's voice through the robot's speaker. Idea, not started.
- A character voice for the robot. Idea, not started.
- JARVIS mishears its name (Jerry, Gary, Jersey). Idea, not started.

## Recent releases

- v2.0.189 (2026-10-13): Queued speech can no longer be lost: a line whose speech fails is retried (up to 3 tries) and a queue file cut off mid-write keeps every intact line. The printer self-check reads the live monitor instead of opening a second connection (no more false FAILED). Cameras that drop off USB and come back are counted per camera with one 'check its cable' line. The last four open July-audit findings are fixed (overnight yes is answered, empty speech no longer clears another line's voice style, the local action list follows new actions, wake-word pause drops leftover audio). Dashboard accessibility fixes, an up-to-date FEATURES.md, and a test-order leak fixed.
- v2.0.188 (2026-10-12): Test-only lint fix: clears the unused imports and one unused variable that made the GitHub CI lint step fail on v2.0.187. No change to the running code.
- v2.0.187 (2026-10-12): Two-way chat bot for alerts and commands; a shopping list and a separate reminders list (reminders with no time never land in the developer queue); optional comments about what is on screen; easier air-mouse engage and useful Kinect gestures; batched background learning; game mode, TV and clap tweaks plus an order tracker; privacy guards on test data; repo cleanup (one daily briefing skill).
- v2.0.186 (2026-10-12): Alexa sign-in rebuilt on the phone-app style OAuth device registration (sign in once, saved token, the real smart-home device list, no session secrets in logs); audio auto-switch waits while media plays; CLOUD_BOOST_UNTIL sends hard turns and screen reading to Claude until a date; PROMPT_SLIM_MODE keeps the cached brain prompt smaller.
- v2.0.185 (2026-10-12): Public status docs on GitHub: docs/STATUS.md and docs/RELEASE_HISTORY.md, generated from the master sheet with all personal data stripped. No code change; the running JARVIS code is the same as v2.0.184.
- v2.0.184 (2026-10-12): JARVIS only announces a finished print he actually watched run (no "print complete" for a print finished while he was off, none when a new print's name arrives before the printer leaves FINISH, no replay after a restart, a stale queued finish line is ...
- v2.0.183 (2026-10-09): "turn on/off hand tracking" works, "close that window" closes one, "u2"/"you two" means YouTube only inside open/close/play commands, a bare website name opens the site (no exe launch), the cold-start warm-up message tells the truth; "why can't I hear X" runs ...
- v2.0.182 (2026-10-09): JARVIS shuts down when asked in plain words ("shut down, no overnight protocol", "yes, that is what I want") and still never on someone else's words; screen vision rebuilt (reads real buttons and links through UI Automation, OCR as backup, checks every click ...
- v2.0.181 (2026-10-05): a named close closes that app (not "what I opened"), close-all keeps the running app and says when a title alone kept a window, "you forgot X" closes X, a misheard keep/close name gets "did you mean ...?", JARVIS never clicks or types on a sign-in page unless ...
- v2.0.180 (2026-10-04): fixes the v2.0.179 native abort - the CUDA driver starts before ctranslate2 again, and every CTranslate2 CUDA build, decode and free runs on one never-exiting host thread (a short-lived self-check thread exiting with CTranslate2 CUDA state aborted the ...

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
