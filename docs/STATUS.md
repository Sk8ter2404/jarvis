# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v2.0.229
- Live since: 2026-10-20
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

- v2.0.230 (2026-10-20): JARVIS knows he is one copy that moves between your machines. On the desktop right after a day on the Dell he said 'I've never quite made myself at home in the Dell, sir' and then 'a bit of a squeeze in this laptop': his prompt said 'there may be other versions of you running in other rooms' and never named the machine. With a handoff partner he is now told he is ONE JARVIS, which machine he is running on (new machine-local MACHINE_NAME: desktop / Dell laptop), and that 'On the Dell laptop' memories are his own time there.
- v2.0.229 (2026-10-20): Test-only fix for GitHub CI: v2.0.228's new two-hand test imported a module it never used, which the pyflakes lint step rejects (the local suite does not run pyflakes). No change to JARVIS itself.
- v2.0.228 (2026-10-20): The webcam hand mouse catches up with the Kinect one: a steady frame pump feeds the hand tracker at the camera's own rate (it was getting clumps between ~6 fps face-detection passes - the lag), the on-screen reticle, fist-drag scrolling, a two-fist grab that moves / resizes the window under your hands (HAND_TWO_HAND_ENABLED, off by default), and no false triggers: minimum tracker score and hand size, a settle time for a new hand, poses confirmed over several frames, no engage right after you type or move the mouse, the cursor locked to your hand. The HUD camera preview runs ~15 fps while the tracker runs and the HUD reads it every 33 ms.
- v2.0.227 (2026-10-20): JARVIS moves between your machines as one copy. Every setting that isn't tied to the hardware is now shared: a change on the desktop or the Dell reaches the other one (newest change wins; cameras, Kinect, hands, monitors, audio devices, mic levels, folders and tunnels stay per machine), and a machine boots already matching the other. At a handoff the conversation in progress travels too - 'Picking up where we left off on the desktop, sir.' - and a version difference between the two is logged. Also: a brain reached through a tunnel on the laptop is treated as remote, so the Dell's boot no longer starts its own empty Ollama and waits ~90 s.
- v2.0.226 (2026-10-20): A device renamed in Alexa, Tuya or Kasa now gets its new name in JARVIS: the catalog merge kept any stored name, so the first name discovery ever saw stuck for good. Each device now also records the name and controller that discovery gave it, so a name you edited by hand still survives, but a rename at the source arrives on the next discovery.
- v2.0.225 (2026-10-20): Stops the made-up numbers at the source. His personality rules TOLD him to put a number on everything ('Tag the certainty with a percentage or probability whenever you can', with sample lines like 'A 78% probability' and 'CPU at 34%, RAM at 61%' that he copied with new numbers); now a number comes only from a reading this turn, and confidence is said in words. The number checker also learns CPU, memory, GPU, time left, percent done and percentages on a confidence; a guess written before a reading that the same reply runs ("CPU at 12%" then system_pulse) is dropped quietly so only the real reading is spoken.
- v2.0.224 (2026-10-20): The action router hands the brain the right instructions more often (measured on 50 real bad turns: 27 -> 34 of 39 expected actions, +17 prompt tokens a turn on average): 'how bad is the background noise', 'dampen the audio', 'anything to do for school tonight', 'what's linked to my Tuya account', 'it's downloading something' and rebuild / parity questions now reach room_noise, volume_down, calendar_today, smart_home_account, the bandwidth actions and server_status. An embedding-search router was tried and lost (30 of 39 for ~1,400 extra tokens).
- v2.0.223 (2026-10-20): Echo cancelling waits for a late PC-audio reference (up to 0.25 s while the loopback is alive): the 10-19 room recording showed the reference arriving >= ~58 ms after the echo, so the canceller had been cancelling against silence (simulated 4.6 -> ~17 dB). And the Claude safety net now works with local-first settings: a turn the checker catches is redone once on Claude Sonnet 5.5 while the Claude boost is on (TURN_CHECK_MODE on).
- v2.0.222 (2026-10-20): Stops answering the room TV (camera sees the TV on + a full 30 s capture of nonstop sound + a voice that isn't clearly his -> dropped, with a spoken 'couldn't tell that was you'); the voice server restarts itself after 3 CUDA-error renders (a GPU driver reset left it broken); the headset's charging status (0x05) counts as on, so the deaf-mic rescue can run; rebuild/parity questions go to the server skill; only facts the owner said himself are shared or imported between the machines.
- v2.0.221 (2026-10-19): 'Which account are my smart devices on?' is answered from Tuya itself: the JARVIS project's linked user and Tuya's user record give 'Your 7 Tuya devices are in the Smart Life app, signed in with your phone number ending 6707 (an account from March 2019); the Tuya Smart app is a separate account.' Only the last 4 digits are spoken. Also fixes v2.0.220's red GitHub check: the new skills' unused alias names are gone.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
