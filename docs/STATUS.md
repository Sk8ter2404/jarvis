# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v3.0.2
- Live since: 2026-10-28
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

- v3.0.2 (2026-10-28): Stability. The CI simulator is fully green for the first time (its 3 long-standing failures were simulator gaps, and a test left a real 5-minute timer thread running into later tests). A timing-flaky speech test is fixed (9% failure under heavy CPU load -> 0 in 600 runs). Short compliments ('Very nice', 'Sounds great') no longer get 'Sorry, I didn't catch that', while garbled lines still do. A comment on something he just reported ('Yay, version 3 is done') no longer re-runs the reading. 'Read up on / do some research on X' searches the web; 'look into the camera' does not. The news test no longer touches the network. Tonight's mistakes are now permanent offline checks (169 replay items; 9 brain-test items).
- v3.0.1 (2026-10-27): Fixes from the first v3 evening. 'What's (all) new' now reads the real release history (it was reading an old upgrade log). 'Is <chat> online / what is <chat> doing' answers about one Agent Office chat by name. Face recognition runs only when you ask who someone is or about a face ('he doesn't sound happy' no longer runs it). The canned 'A bold choice ... I'll note that for posterity' is dropped wherever it is unearned, and each signature line is limited to once per 10 turns and 20 minutes. After a screen read, a date or 'it's out' claim must match what was on the screen; a third unchanged screen read in one turn is refused; 'is it out yet' goes to a web search. Garbled short lines get 'Sorry, I didn't catch that' instead of an action, while short real replies still work. Streaming sites without a search link get searched through their own page (only when that site is the window in front), and his own typed text is never offered as a click target. CI fix: drive roots are recognised on any OS.
- v3.0.0 (2026-10-27): Version 3, a full overhaul. A slimmer, clearer prompt (43,875 to 40,815 characters) so he picks the right action more often (94.0% vs 90.1% on the test set; unverified claims 3 to 0). One name per job: near-miss and invented action names map to the real one, and a verb-clash guard keeps mute and pause apart. New 'move this file to ...' and 'undo the last move' (your own folders only; he asks first). Learning works again (packed learning replies were being dropped). Proactive remarks no longer claim actions he didn't take, and 'shut down, no overnight protocol' asks instead of guessing. 'What are my agents doing / anything waiting on me?' reads the Agent Office. Watch-only guards for wrong-but-allowed actions and action ping-pong, plus a per-turn trace report. Off until switched on: link reader, faster boot, background jobs on another computer. A security review found 13 issues; all fixed.
- v2.0.231 (2026-10-26): Fixes from tonight's live log: habit offers no longer announce a wrong time ('Around this time on Mondays, sir' instead of 'It's Monday 7 PM' at 8 PM); 'play X on YouTube on the left monitor' plays it and moves it there instead of opening a search page; 'play my song from earlier / play that again' replays your last music or video request instead of searching for a song called 'Earlier'; 'tell Claude to ...', '..., or tell Claude to do it' and 'pass it over to Claude' save a note for Claude instead of 'I can't' and a web search.
- v2.0.230 (2026-10-20): JARVIS knows he is one copy that moves between your machines. On the desktop right after a day on the Dell he said 'I've never quite made myself at home in the Dell, sir' and then 'a bit of a squeeze in this laptop': his prompt said 'there may be other versions of you running in other rooms' and never named the machine. With a handoff partner he is now told he is ONE JARVIS, which machine he is running on (new machine-local MACHINE_NAME: desktop / Dell laptop), and that 'On the Dell laptop' memories are his own time there.
- v2.0.229 (2026-10-20): Test-only fix for GitHub CI: v2.0.228's new two-hand test imported a module it never used, which the pyflakes lint step rejects (the local suite does not run pyflakes). No change to JARVIS itself.
- v2.0.228 (2026-10-20): The webcam hand mouse catches up with the Kinect one: a steady frame pump feeds the hand tracker at the camera's own rate (it was getting clumps between ~6 fps face-detection passes - the lag), the on-screen reticle, fist-drag scrolling, a two-fist grab that moves / resizes the window under your hands (HAND_TWO_HAND_ENABLED, off by default), and no false triggers: minimum tracker score and hand size, a settle time for a new hand, poses confirmed over several frames, no engage right after you type or move the mouse, the cursor locked to your hand. The HUD camera preview runs ~15 fps while the tracker runs and the HUD reads it every 33 ms.
- v2.0.227 (2026-10-20): JARVIS moves between your machines as one copy. Every setting that isn't tied to the hardware is now shared: a change on the desktop or the Dell reaches the other one (newest change wins; cameras, Kinect, hands, monitors, audio devices, mic levels, folders and tunnels stay per machine), and a machine boots already matching the other. At a handoff the conversation in progress travels too - 'Picking up where we left off on the desktop, sir.' - and a version difference between the two is logged. Also: a brain reached through a tunnel on the laptop is treated as remote, so the Dell's boot no longer starts its own empty Ollama and waits ~90 s.
- v2.0.226 (2026-10-20): A device renamed in Alexa, Tuya or Kasa now gets its new name in JARVIS: the catalog merge kept any stored name, so the first name discovery ever saw stuck for good. Each device now also records the name and controller that discovery gave it, so a name you edited by hand still survives, but a rename at the source arrives on the next discovery.
- v2.0.225 (2026-10-20): Stops the made-up numbers at the source. His personality rules TOLD him to put a number on everything ('Tag the certainty with a percentage or probability whenever you can', with sample lines like 'A 78% probability' and 'CPU at 34%, RAM at 61%' that he copied with new numbers); now a number comes only from a reading this turn, and confidence is said in words. The number checker also learns CPU, memory, GPU, time left, percent done and percentages on a confidence; a guess written before a reading that the same reply runs ("CPU at 12%" then system_pulse) is dropped quietly so only the real reading is spoken.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
