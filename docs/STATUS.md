# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v3.0.7
- Live since: 2026-10-28
- Latest change: v3.0.7 (10-28): the RAM question, other voices in the room, and honest note claims.

## In progress

- Smart home control without the voice assistant cloud: direct control of local plugs and lights, doorbell announcements, IR remote
- Echo-cancellation tuning for hearing over videos
- Pinch-to-click calibration for Kinect hand tracking
- Boot proof and long live diagnostics
- Speed plan: enabling each speed feature one at a time with before and after measurements

## Roadmap

Open, planned and idea items. Wording is generic on purpose.

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

- v3.0.7 (2026-10-28): Three fixes from tonight's live log. 'Why is the RAM so high?' or 'what's using all my memory?' is answered at once with the number and the biggest memory users, even below the high mark, and JARVIS's own thinking on the GPU is said as such. In the follow-up window after he answers, a voice that does not clearly match the owner no longer keeps the window open, and its remarks that are not a command or a question for JARVIS are ignored, so he stops replying to other people in the room. A spoken 'I've noted that' when no note was made is replaced with an honest 'I haven't made a note of that, sir.'
- v3.0.6 (2026-10-28): Honesty, routing and the server, from today's log review and live test. Plain timers no longer get a made-up 'check the oven' label, stock lines that claim work ('I'll note that for posterity') are gone, and when a chain of steps ends on a promise he did not keep, JARVIS now says so. Ideas and suggestions are answered in words instead of opening a web search, and no tool names are spoken. 'Can you open it up?' with nothing to open gets 'Open what, sir?'; developer to-dos ('we need to give the robot more voice lines') become notes for Claude; a flat 'I've no way to check that' offers the nearest read-only action or a note. 'How long is left on that timer?' and 'what can you do?' answer instantly, 'what's new' starts with the version, and his own GPU load while thinking is called that. The server watch also reads disk errors and temperatures, the UPS on-battery status and stopped server apps, read-only with fixed commands. Audio heard while the PC plays media, with an unsure voice, no longer keeps the follow-up window open. YouTube on a named monitor no longer searches for the monitor words.
- v3.0.5 (2026-10-28): 'Pass it over to Claude' and the RAM answer, from this afternoon's live log. At 15:58 'No, just pass over to Clog for me' was misheard and matched nothing, so a browser robot searched the web for 'Clog' for 47 seconds. The hand-off now needs no 'it', allows 'for me', 'please' or 'now' at the end, and accepts the speech-to-text's own mishearings of Claude (clog, claw, clawed, claud) only inside the hand-off phrase, so nozzle clogs and 'send it to the cloud' keep their meaning. Questions about a hand-off ('what did you send to Claude?') are no longer taken as one, and a bare hand-off that still reaches the browser robot gets a quick question instead of a web search. 'Why is the RAM so high?' no longer answers 'All systems nominal' at 92 percent: an asked-for system report opens with the reading that is out of line and names the biggest memory users (for example: my local brain model at 8 gigabytes, Claude at 6.5 and Chrome at 4.5); JARVIS's own GPU use is not reported as a problem, and the PC check names the memory users too.
- v3.0.4 (2026-10-28): Morning greeting and printer speech, from today's live log. The morning greeting no longer opens with the Amazon tracker's sign-in command ('Good morning, sir. I need a Google sign-in first, sir. Run python -m ...'); every arrival since 10-13 had. With the mailbox unreadable it says nothing about orders, and any briefing section that reads like setup instructions is left out. The print heads-up no longer says 'past UNKNOWN prints around this size have failed 33% of the time': with neither the material nor the size known there is no heads-up, and materials are said properly ('PLA plus', 'of about 220 layers'). A print failure or a low credit balance is said alone, without 'CPU 38 percent ... 12 windows open' behind it.
- v3.0.3 (2026-10-28): Smarter and lighter, every change measured on 6 brain-test seeds against v3.0.2. Garbled words ('Enough and awesome') get 'Sorry, sir, I missed that - what would you like?' instead of opening an app (0/6 -> 6/6). 'Is it time to watch <show> yet' / 'is <show> out yet' searches for the release date even when the title has a word like 'Vision' (3/6 -> 6/6). 'Skip this one' plays the next song (0/6 -> 6/6). Overall golden accuracy 92.9% -> 94.4%, unverified claims 1 -> 0, invented actions 9 -> 7. After a screen read, a made-up 'it dropped early' follow-on sentence is dropped along with the claim it continues. Idle CPU: the face detector ran four full passes on every camera frame even with nobody in the room (about 3.5 CPU cores); after 15 s without a face it now checks once a second, and the first face brings back full speed.
- v3.0.2 (2026-10-28): Stability. The CI simulator is fully green for the first time (its 3 long-standing failures were simulator gaps, and a test left a real 5-minute timer thread running into later tests). A timing-flaky speech test is fixed (9% failure under heavy CPU load -> 0 in 600 runs). Short compliments ('Very nice', 'Sounds great') no longer get 'Sorry, I didn't catch that', while garbled lines still do. A comment on something he just reported ('Yay, version 3 is done') no longer re-runs the reading. 'Read up on / do some research on X' searches the web; 'look into the camera' does not. The news test no longer touches the network. Tonight's mistakes are now permanent offline checks (169 replay items; 9 brain-test items).
- v3.0.1 (2026-10-27): Fixes from the first v3 evening. 'What's (all) new' now reads the real release history (it was reading an old upgrade log). 'Is <chat> online / what is <chat> doing' answers about one Agent Office chat by name. Face recognition runs only when you ask who someone is or about a face ('he doesn't sound happy' no longer runs it). The canned 'A bold choice ... I'll note that for posterity' is dropped wherever it is unearned, and each signature line is limited to once per 10 turns and 20 minutes. After a screen read, a date or 'it's out' claim must match what was on the screen; a third unchanged screen read in one turn is refused; 'is it out yet' goes to a web search. Garbled short lines get 'Sorry, I didn't catch that' instead of an action, while short real replies still work. Streaming sites without a search link get searched through their own page (only when that site is the window in front), and his own typed text is never offered as a click target. CI fix: drive roots are recognised on any OS.
- v3.0.0 (2026-10-27): Version 3, a full overhaul. A slimmer, clearer prompt (43,875 to 40,815 characters) so he picks the right action more often (94.0% vs 90.1% on the test set; unverified claims 3 to 0). One name per job: near-miss and invented action names map to the real one, and a verb-clash guard keeps mute and pause apart. New 'move this file to ...' and 'undo the last move' (your own folders only; he asks first). Learning works again (packed learning replies were being dropped). Proactive remarks no longer claim actions he didn't take, and 'shut down, no overnight protocol' asks instead of guessing. 'What are my agents doing / anything waiting on me?' reads the Agent Office. Watch-only guards for wrong-but-allowed actions and action ping-pong, plus a per-turn trace report. Off until switched on: link reader, faster boot, background jobs on another computer. A security review found 13 issues; all fixed.
- v2.0.231 (2026-10-26): Fixes from tonight's live log: habit offers no longer announce a wrong time ('Around this time on Mondays, sir' instead of 'It's Monday 7 PM' at 8 PM); 'play X on YouTube on the left monitor' plays it and moves it there instead of opening a search page; 'play my song from earlier / play that again' replays your last music or video request instead of searching for a song called 'Earlier'; 'tell Claude to ...', '..., or tell Claude to do it' and 'pass it over to Claude' save a note for Claude instead of 'I can't' and a web search.
- v2.0.230 (2026-10-20): JARVIS knows he is one copy that moves between your machines. On the desktop right after a day on the Dell he said 'I've never quite made myself at home in the Dell, sir' and then 'a bit of a squeeze in this laptop': his prompt said 'there may be other versions of you running in other rooms' and never named the machine. With a handoff partner he is now told he is ONE JARVIS, which machine he is running on (new machine-local MACHINE_NAME: desktop / Dell laptop), and that 'On the Dell laptop' memories are his own time there.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
