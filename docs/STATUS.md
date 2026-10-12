# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.184
- Live since: 2026-10-12
- Latest change: v2.0.184: no false 'print complete' announcements; new HUMOR_MODE setting (dry or sarcastic).

## In progress

- Echo-cancellation tuning for hearing over videos
- Pinch-to-click calibration for Kinect hand tracking
- Screen memory and remarks about what is on screen (built, off by default)
- Per-camera flap counter
- Boot proof and long live diagnostics
- Speed plan: enabling each speed feature one at a time with before and after measurements

## Roadmap

Open, planned and idea items. Wording is generic on purpose.

### In progress (8)

- No more bugs on JARVIS or the robot. In progress.
- Make FEATURES.md match reality. In progress.
- Fix the leftover findings from the July 4 audit. In progress.
- Prepare the answer while 'processing' plays. In progress.
- Searchable screen memory. In progress.
- Stop the robot repeating the same jokes. In progress.
- Turn checker: retry a bad local answer on Claude. In progress.
- Test-order leak between two test files. In progress.

### Planned next (6)

- Make sure queued speech is never lost. Planned next.
- Small engineering follow-ups. Planned next.
- Kinect sees ghost skeletons. Planned next.
- Refresh the big mission prompt. Planned next.
- Better memory search models. Planned next.
- Show clearly on the VRAM bar when settings go over the limit. Planned next.

### Open, needs a decision (2)

- Keep the new Ollama setup or go back. Open: waiting on a design decision.
- JARVIS reacts when the robot is picked up or shaken. Open: waiting on a design decision.

### Ideas (18)

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
- Robot mood and 'age', with eyes on the web page. Idea, not started.
- Games where the robot is a sore loser. Idea, not started.
- Distance, gesture and screen-eye modules. Idea, not started.
- Motion emotes (nod, wiggle, sulk-turn). Idea, not started.
- JARVIS's voice through the robot's speaker. Idea, not started.
- A character voice for the robot. Idea, not started.
- JARVIS mishears its name (Jerry, Gary, Jersey). Idea, not started.

## Recent releases

- v2.0.184 (2026-10-12): JARVIS only announces a finished print he actually watched run (no "print complete" for a print finished while he was off, none when a new print's name arrives before the printer leaves FINISH, no replay after a restart, a stale queued finish line is ...
- v2.0.183 (2026-10-09): "turn on/off hand tracking" works, "close that window" closes one, "u2"/"you two" means YouTube only inside open/close/play commands, a bare website name opens the site (no exe launch), the cold-start warm-up message tells the truth; "why can't I hear X" runs ...
- v2.0.182 (2026-10-09): JARVIS shuts down when asked in plain words ("shut down, no overnight protocol", "yes, that is what I want") and still never on someone else's words; screen vision rebuilt (reads real buttons and links through UI Automation, OCR as backup, checks every click ...
- v2.0.181 (2026-10-05): a named close closes that app (not "what I opened"), close-all keeps the running app and says when a title alone kept a window, "you forgot X" closes X, a misheard keep/close name gets "did you mean ...?", JARVIS never clicks or types on a sign-in page unless ...
- v2.0.180 (2026-10-04): fixes the v2.0.179 native abort - the CUDA driver starts before ctranslate2 again, and every CTranslate2 CUDA build, decode and free runs on one never-exiting host thread (a short-lived self-check thread exiting with CTranslate2 CUDA state aborted the ...
- v2.0.179 (2026-10-04): one voice per reply (the clone never switches mid-reply, a sentence prepared ahead waits until it is needed, a long first line is split at a clause, misses cool down instead of switching the clone off for the session), steady Kinect hand tracking (one shared ...
- v2.0.178 (2026-10-03): the cloned JARVIS voice runs from its own GPU server with Kokoro covering any line it misses (off until switched on), and "close all windows except X" really closes the rest - never JARVIS's own HUD/Reticle or the Windows shell, asks first above five, never ...
- v2.0.177 (2026-10-02): a yes to JARVIS's own offer carries it out, slow apps are moved when their window appears, streaming search uses verified links and stops at a sign-in wall, 'close that' closes only what he opened, fresh 'diagnostic status', facts remember who said them + ...
- v2.0.176 (2026-10-02): the big 10-02 batch - JARVIS never voices a claim he didn't carry out, honest about admin windows and repeated failures, instant time; turn checker + instant actions (shadow), answer pre-render (R3), July audit fixes, local-first features with a retired-model ...
- v2.0.175 (2026-10-02): 'power off the tv' never arms JARVIS's own shutdown, a reply saying 'always listening' never leaves wake-word mode, and standby wakes only on the name or a bare wake phrase (July audit A28, A37, A29)

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
