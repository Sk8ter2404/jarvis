# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.220
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

- v2.0.221 (2026-10-19): 'Which account are my smart devices on?' is answered from Tuya itself: the JARVIS project's linked user and Tuya's user record give 'Your 7 Tuya devices are in the Smart Life app, signed in with your phone number ending 6707 (an account from March 2019); the Tuya Smart app is a separate account.' Only the last 4 digits are spoken. Also fixes v2.0.220's red GitHub check: the new skills' unused alias names are gone.
- v2.0.220 (2026-10-19): JARVIS watches the home server (read-only): 'how's the server?' answers from a fresh check (services answering; over SSH the array state, disabled disks, and a rebuild/parity check with percent and time left), and a 5-minute check announces once when it goes down, comes back, reports a disabled disk or finishes a rebuild. Plus 'how's the Backblaze backup?' from Backblaze's own status files.
- v2.0.219 (2026-10-19): OpenCV stays off the graphics card in every thread: the face tracker threw 'OpenCL CL_OUT_OF_RESOURCES' (86 times in 40 minutes) because cv2.ocl.setUseOpenCL(False) only covers the thread that calls it (measured: main thread off, others on). OPENCV_OPENCL_RUNTIME=disabled is now set before OpenCV loads, so the Haar cascades run on the CPU everywhere, as the original comment intended.
- v2.0.218 (2026-10-19): The VOID headset is no longer called off while you wear it: at 10-13% battery its dongle reports link byte 0x02, a value never seen before; JARVIS read it as unknown and held a stale 'off' (and moved the mic to the desk mic). 0x02 now counts as on (measured 15 of 15 samples with the headset in use), and every on/off change logs the dongle's raw reply.
- v2.0.217 (2026-10-19): Room noise from a real reading: "how bad is the background noise?" got an invented "roughly 42% of the current audio is non-essential interference". JARVIS now listens to the room for one second on the shared mic and checks whether the PC is playing sound, then says only that (quiet / some noise under the speech level / noisy, with the dB reading), with no model guess.
- v2.0.216 (2026-10-19): One memory across the owner's machines: each JARVIS trades a small bundle of its own recent session summaries and facts with the other machine over the same SSH link (pushed every 10 minutes and when it stands down, pulled at start). Imported sessions read "On the Dell laptop: ...", so asking the desktop what happened on the laptop today gets "while I was over on the laptop...". Imported items are never sent back.
- v2.0.215 (2026-10-19): One JARVIS at a time across the owner's machines: JARVIS starting on one machine asks the other machine's copy to stand down over SSH (it says "Handing over to the Dell laptop, sir" and shuts down cleanly), so the laptop gets the desktop's GPU to itself. With the desktop's resume setting on, the desktop boots JARVIS again by itself once the laptop's copy stops (quit there, or the laptop unreachable for about 3 minutes). Off until each machine's private settings name the other.
- v2.0.214 (2026-10-19): Store apps open by voice: "open Teams on the top monitor" failed with "could not launch Microsoft Teams" because the new Teams (and new Outlook, Notepad) are Microsoft Store apps with no program file to start by name. JARVIS now looks the app up in the Start menu list and launches it by its app ID, only when exactly one app fits the name.
- v2.0.213 (2026-10-18): Test-only (no change to the running code): v2.0.212's new own-helper test built Windows-style paths, which failed on GitHub's Linux CI; it now uses os.path.join.
- v2.0.212 (2026-10-18): Fixes from the 10-18 log review: JARVIS sees and clicks windows of a browser or app he opened himself (Chrome started by him was hidden from his clicks); the echo canceller's copy of the PC's audio gets a 200 ms buffer instead of 10 ms and marks any gap (55,000 lost-audio warnings a night); a failed shell command is reported as failed, not 'As you wish'; 'pass that over to Claude' leaves a real note.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
