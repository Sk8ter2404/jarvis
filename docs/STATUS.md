# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.201
- Live since: 2026-10-15
- Latest change: v2.0.201 (10-15): the self-upgrade is rebuilt to be safe (request-only, isolated, reviewed); still switched off.

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

### Planned next (2)

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

- v2.0.201 (2026-10-15): The self-upgrade is rebuilt to be safe (still switched off). It runs only when asked, in a separate git worktree off the published code, never touching or restarting the running JARVIS. The editing step cannot push, tag, merge or change data files; the planning and review steps are read-only. Each task is its own commit, a failing task is reverted, the full test suite and the privacy check run at the end, and the result is a branch plus a report for review - nothing ships without a normal release. Overnight ideas go to a proposals file, never straight into the work queue.
- v2.0.200 (2026-10-15): Hand mouse (air mouse v2) on the new hand tracker, off by default: a deliberate pinch held still takes the cursor, a quick pinch clicks, a second held pinch drags, an open palm or the hand leaving lets go, and it never engages in games, fullscreen apps or calls. Every brain call now says who is calling and whether it is a live answer or background work (a test fails on any untagged call). The wake-word listener follows the current microphone when the default input changes. Old session logs are compressed into an archive instead of deleted. The every-action sweep counts its own sandbox blocks separately from real crashes. Fixes a Linux-only test failure from v2.0.199.
- v2.0.199 (2026-10-15): A hard stop before irreversible clicks: when screen control is about to click a button that sends, submits, posts, publishes, pays, buys, transfers, deletes or accepts terms - or press Enter in a message box that would send - it asks first and acts only on an explicit yes for that one action. The hologram can show your own 3D models (STL, 3MF, OBJ from chosen folders): a 3MF's objects become named parts that explode and show their size, and "show me the <model>" picks one by name. Model files are served only by id from the configured folders.
- v2.0.198 (2026-10-14): Claude Code jobs can run on a second Claude subscription first. With a last-day date set, every Claude Code job JARVIS starts (upgrade loop, multi-agent pipeline, dry audit, overnight ideas) runs on a second Claude Code profile - its own sign-in folder - through that date, and on the main account whenever the second one is at its 5-hour or weekly limit, switching back when the limit resets. The sign-in stays inside Claude Code's own folder; JARVIS never reads or stores a token. The upgrade pipeline also recognises Claude Code's current limit wording again.
- v2.0.197 (2026-10-14): Test tooling only (no change to the running code): the GitHub CI job's time limit goes from 20 to 40 minutes. The job runs the whole suite twice (plain, then with coverage) and had grown to the edge; v2.0.196's run passed its tests and was cut off mid-coverage at exactly 20 minutes.
- v2.0.196 (2026-10-14): The point-and-flick gesture works at the webcam's real frame rate. Live on v2.0.195 the tracker got about 12 frames a second: the webcam can do 30 but slows itself in room light, and frames arrive in short bursts. The flick now adapts its time window to the measured frame rate (the speed and angle bar is unchanged, so jitter and slow swings still never count). 'Is hand tracking on' mentions when more light would help.
- v2.0.195 (2026-10-14): The hand tracker now gets the camera's full frame rate: live on v2.0.194 it received only about 6 frames a second (the face-tracking loop's own pace), too few for a quick flick. While the tracker runs, the camera loop keeps reading the tracker's camera for a moment after each normal read and hands every frame over; face tracking keeps its cadence, and nothing changes while the tracker is off. Voice: 'open up the hologram', 'close the hologram' and 'hand menu' work without the language model; 'show me the arc reactor', 'explode it' and 'put it back together' go to the hologram only while it is open, so a plain 'show the arc reactor' stays the small corner reactor.
- v2.0.194 (2026-10-14): Hand tracking v2, phase 1: a separate MediaPipe hand tracker (21 points per finger-level hand, about 29 frames a second) reads the camera frames JARVIS already captures, and a hologram window (a three.js arc reactor) is driven by it: pinch and hold for the menu, a fist to move, point and flick to spin, two fists apart to explode it into labelled parts, pinch a part and hold it for its info, clap to rebuild. Two views: a solid dark 'hand menu' with no camera picture, and 'open up the hologram' with the mirrored camera behind it. The hologram starts the tracker it needs and stops it when closed. 'Hand tracking on' now means this tracker, which never moves the cursor (it used to switch on the Kinect air mouse); 'hand tracking off' also stops the air mouse. All of it is off until asked for.
- v2.0.193 (2026-10-14): Test tooling only (no change to the running code): the test watchdog's kill now really ends a wedged test run on Windows (TerminateProcess instead of os._exit, which could hang on leftover CUDA/COM threads - one run sat 26 hours), tests can no longer fake its clock, and a new gate wrapper ends a test run that outlives the watchdog.
- v2.0.192 (2026-10-14): A weekly success score ('how did you do this week': turns, answer rate, median response time, failing actions, from this PC's logs only). The smart-home command now has exactly one owner (a duplicate registration that only lost by alphabetical load order is gone) and a test fails if any action is ever registered twice. An echo-cancellation fallback for protected video audio (ducks and tightens the wake gate when the echo reference is missing while media plays), shipped log-only.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
