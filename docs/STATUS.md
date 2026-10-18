# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.212
- Live since: 2026-10-18
- Latest change: v2.0.212 (10-18): log-review fixes - sees his own-launched browser, echo-cancel capture buffer, honest shell failures, pass-to-Claude notes.

## In progress

- Smart home control without the voice assistant cloud: direct control of local plugs and lights, doorbell announcements, IR remote
- Echo-cancellation tuning for hearing over videos
- Pinch-to-click calibration for Kinect hand tracking
- Boot proof and long live diagnostics
- Speed plan: enabling each speed feature one at a time with before and after measurements

## Roadmap

Open, planned and idea items. Wording is generic on purpose.

### In progress (4)

- Prepare the answer while 'processing' plays. In progress.
- Stop the robot repeating the same jokes. In progress.
- Turn checker: retry a bad local answer on Claude. In progress.
- Echo-cancel reference drops data (55,000 warnings a night). In progress.

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

- v2.0.212 (2026-10-18): Fixes from the 10-18 log review: JARVIS sees and clicks windows of a browser or app he opened himself (Chrome started by him was hidden from his clicks); the echo canceller's copy of the PC's audio gets a 200 ms buffer instead of 10 ms and marks any gap (55,000 lost-audio warnings a night); a failed shell command is reported as failed, not 'As you wish'; 'pass that over to Claude' leaves a real note.
- v2.0.211 (2026-10-18): Web search's 'top results' are real results: only links below the page's 'Search Results' heading, never Google's own controls (the doodle, search by voice, filter chips) and never a link carrying an email address (the owner's Google Account link was among them).
- v2.0.210 (2026-10-17): Shutting down in wake-word mode works: after 'shut down' asks about the overnight protocol, your next sentence counts without saying 'Jarvis' (once), a heard 'now' / 'know' / 'nah' means no, and a garbled one- or two-word answer is asked once more instead of cancelled. Also: 'I've opened the search' is no longer blocked as a made-up claim when a search really just ran.
- v2.0.209 (2026-10-16): Hand tracking on the Kinect no longer runs backwards: the Kinect's mirrored picture (and its depth map) is flipped back once before the hand tracker sees it.
- v2.0.208 (2026-10-16): Memory clean-up calibrated for the new memory search model: duplicate and 'might contradict' checks use cut-offs measured on your own facts (0.97 / 0.50 on voyage-4-nano), so the contradiction check works again and nothing is deleted that the old model would have kept.
- v2.0.207 (2026-10-15): Fixes from tonight's log review: playing an artist works again (no such playlist -> played as music; the Apple Music song page now presses its Play button or Space when the song row can't be seen; 'play it on YouTube' plays instead of only searching), 'YouTube, <search>' on a monitor no longer runs as a program, a song name sent to the vibe action is played, 'delete / close that website' closes the tab in front, the TV detector stops flickering on people moving at the desk (sustained evidence until a TV area is calibrated, 20 s hold), and no Hugging Face progress bars in the log.
- v2.0.206 (2026-10-15): Test-only (no change to the running code): the last hardware-dependent test now fakes the Windows default microphone, so the full suite passes completely on this PC (25,111 of 25,111) - which the safe self-upgrade's gate needs.
- v2.0.205 (2026-10-15): Code health, no behaviour change: helper functions that existed as identical copies in several skills (window titles, sleep/standby checks, desk presence, Kinect bridge lookup, protected-data decryption, small text formatters, a JSON writer) now live once in shared modules, so a future fix lands everywhere at once.
- v2.0.204 (2026-10-15): Hologram phase 3, ready for a headset: a mixed-reality page (/holo/xr) for the Quest 3 browser with passthrough and the headset's own hand tracking - pinch to grab and move the reactor or your own models in 3D, two hands to scale and turn, pull apart to explode, clap to rebuild, pinch-and-hold a part for its name and size. It needs a secure address to run on the headset; until one is set up the page explains the two ways to get one, and 'open the hologram on the Quest' says the same.
- v2.0.203 (2026-10-15): Hologram phase 2, off by default: 'open the hologram with the Kinect' uses the Kinect's colour camera for hand tracking and its depth sensor so the hologram hides behind your real hands when they are in front of it, and a fist pushed toward or pulled away from the sensor moves it nearer or farther. It reads the Kinect only through JARVIS's existing connection. A new test fails if any module ever defines the same function twice (a scan found no stale duplicates today).

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
