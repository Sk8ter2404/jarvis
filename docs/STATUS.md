# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.214
- Live since: 2026-10-19
- Latest change: v2.0.213 (10-18): test-only fix for Linux CI.

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

- v2.0.215 (2026-10-19): One JARVIS at a time across the owner's machines: JARVIS starting on one machine asks the other machine's copy to stand down over SSH (it says "Handing over to the Dell laptop, sir" and shuts down cleanly), so the laptop gets the desktop's GPU to itself. With the desktop's resume setting on, the desktop boots JARVIS again by itself once the laptop's copy stops (quit there, or the laptop unreachable for about 3 minutes). Off until each machine's private settings name the other.
- v2.0.214 (2026-10-19): Store apps open by voice: "open Teams on the top monitor" failed with "could not launch Microsoft Teams" because the new Teams (and new Outlook, Notepad) are Microsoft Store apps with no program file to start by name. JARVIS now looks the app up in the Start menu list and launches it by its app ID, only when exactly one app fits the name.
- v2.0.213 (2026-10-18): Test-only (no change to the running code): v2.0.212's new own-helper test built Windows-style paths, which failed on GitHub's Linux CI; it now uses os.path.join.
- v2.0.212 (2026-10-18): Fixes from the 10-18 log review: JARVIS sees and clicks windows of a browser or app he opened himself (Chrome started by him was hidden from his clicks); the echo canceller's copy of the PC's audio gets a 200 ms buffer instead of 10 ms and marks any gap (55,000 lost-audio warnings a night); a failed shell command is reported as failed, not 'As you wish'; 'pass that over to Claude' leaves a real note.
- v2.0.211 (2026-10-18): Web search's 'top results' are real results: only links below the page's 'Search Results' heading, never Google's own controls (the doodle, search by voice, filter chips) and never a link carrying an email address (the owner's Google Account link was among them).
- v2.0.210 (2026-10-17): Shutting down in wake-word mode works: after 'shut down' asks about the overnight protocol, your next sentence counts without saying 'Jarvis' (once), a heard 'now' / 'know' / 'nah' means no, and a garbled one- or two-word answer is asked once more instead of cancelled. Also: 'I've opened the search' is no longer blocked as a made-up claim when a search really just ran.
- v2.0.209 (2026-10-16): Hand tracking on the Kinect no longer runs backwards: the Kinect's mirrored picture (and its depth map) is flipped back once before the hand tracker sees it.
- v2.0.208 (2026-10-16): Memory clean-up calibrated for the new memory search model: duplicate and 'might contradict' checks use cut-offs measured on your own facts (0.97 / 0.50 on voyage-4-nano), so the contradiction check works again and nothing is deleted that the old model would have kept.
- v2.0.207 (2026-10-15): Fixes from tonight's log review: playing an artist works again (no such playlist -> played as music; the Apple Music song page now presses its Play button or Space when the song row can't be seen; 'play it on YouTube' plays instead of only searching), 'YouTube, <search>' on a monitor no longer runs as a program, a song name sent to the vibe action is played, 'delete / close that website' closes the tab in front, the TV detector stops flickering on people moving at the desk (sustained evidence until a TV area is calibrated, 20 s hold), and no Hugging Face progress bars in the log.
- v2.0.206 (2026-10-15): Test-only (no change to the running code): the last hardware-dependent test now fakes the Windows default microphone, so the full suite passes completely on this PC (25,111 of 25,111) - which the safe self-upgrade's gate needs.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
