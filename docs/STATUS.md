# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.208
- Live since: 2026-10-16
- Latest change: v2.0.209 (10-16): Kinect hand tracking no longer runs backwards (applies at the next restart).

## In progress

- Smart home control without the voice assistant cloud: direct control of local plugs and lights, doorbell announcements, IR remote
- Echo-cancellation tuning for hearing over videos
- Pinch-to-click calibration for Kinect hand tracking
- Boot proof and long live diagnostics
- Speed plan: enabling each speed feature one at a time with before and after measurements

## Roadmap

Open, planned and idea items. Wording is generic on purpose.

### In progress (3)

- Prepare the answer while 'processing' plays. In progress.
- Stop the robot repeating the same jokes. In progress.
- Turn checker: retry a bad local answer on Claude. In progress.

### Planned next (1)

- Kinect sees ghost skeletons. Planned next.

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

- v2.0.209 (2026-10-16): Hand tracking on the Kinect no longer runs backwards: the Kinect's mirrored picture (and its depth map) is flipped back once before the hand tracker sees it.
- v2.0.208 (2026-10-16): Memory clean-up calibrated for the new memory search model: duplicate and 'might contradict' checks use cut-offs measured on your own facts (0.97 / 0.50 on voyage-4-nano), so the contradiction check works again and nothing is deleted that the old model would have kept.
- v2.0.207 (2026-10-15): Fixes from tonight's log review: playing an artist works again (no such playlist -> played as music; the Apple Music song page now presses its Play button or Space when the song row can't be seen; 'play it on YouTube' plays instead of only searching), 'YouTube, <search>' on a monitor no longer runs as a program, a song name sent to the vibe action is played, 'delete / close that website' closes the tab in front, the TV detector stops flickering on people moving at the desk (sustained evidence until a TV area is calibrated, 20 s hold), and no Hugging Face progress bars in the log.
- v2.0.206 (2026-10-15): Test-only (no change to the running code): the last hardware-dependent test now fakes the Windows default microphone, so the full suite passes completely on this PC (25,111 of 25,111) - which the safe self-upgrade's gate needs.
- v2.0.205 (2026-10-15): Code health, no behaviour change: helper functions that existed as identical copies in several skills (window titles, sleep/standby checks, desk presence, Kinect bridge lookup, protected-data decryption, small text formatters, a JSON writer) now live once in shared modules, so a future fix lands everywhere at once.
- v2.0.204 (2026-10-15): Hologram phase 3, ready for a headset: a mixed-reality page (/holo/xr) for the Quest 3 browser with passthrough and the headset's own hand tracking - pinch to grab and move the reactor or your own models in 3D, two hands to scale and turn, pull apart to explode, clap to rebuild, pinch-and-hold a part for its name and size. It needs a secure address to run on the headset; until one is set up the page explains the two ways to get one, and 'open the hologram on the Quest' says the same.
- v2.0.203 (2026-10-15): Hologram phase 2, off by default: 'open the hologram with the Kinect' uses the Kinect's colour camera for hand tracking and its depth sensor so the hologram hides behind your real hands when they are in front of it, and a fist pushed toward or pulled away from the sensor moves it nearer or farther. It reads the Kinect only through JARVIS's existing connection. A new test fails if any module ever defines the same function twice (a scan found no stale duplicates today).
- v2.0.202 (2026-10-15): Fixes from the self-upgrade queue review: the local brain adopts a better model once it finishes downloading (checked at most every 5 minutes, between turns); turning the workshop hologram off hides it so 'on' brings it back; web pages opened by 'open', 'search' and 'YouTube' land on a monitor instead of off-screen; a stale memory note corrected. Three queued items were already fixed and one was not a bug.
- v2.0.201 (2026-10-15): The self-upgrade is rebuilt to be safe (still switched off). It runs only when asked, in a separate git worktree off the published code, never touching or restarting the running JARVIS. The editing step cannot push, tag, merge or change data files; the planning and review steps are read-only. Each task is its own commit, a failing task is reverted, the full test suite and the privacy check run at the end, and the result is a branch plus a report for review - nothing ships without a normal release. Overnight ideas go to a proposals file, never straight into the work queue.
- v2.0.200 (2026-10-15): Hand mouse (air mouse v2) on the new hand tracker, off by default: a deliberate pinch held still takes the cursor, a quick pinch clicks, a second held pinch drags, an open palm or the hand leaving lets go, and it never engages in games, fullscreen apps or calls. Every brain call now says who is calling and whether it is a live answer or background work (a test fails on any untagged call). The wake-word listener follows the current microphone when the default input changes. Old session logs are compressed into an archive instead of deleted. The every-action sweep counts its own sandbox blocks separately from real crashes. Fixes a Linux-only test failure from v2.0.199.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
