# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v3.0.13
- Live since: 2026-11-10
- Latest change: v3.0.11 (11-05, LIVE since 18:55): looks through the camera before answering, 'look again' really looks, no invented Teams counts, idle remarks stay grounded, stale queued lines stay quiet.

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

- v3.0.14 (2026-11-10): From today's live log. A sworn "what did you just say?" now gets his last line back word for word (it started a robot banter action instead). "Shut up, stop listening to me" now puts him to sleep: insults and courtesy no longer count toward the six-word limit that keeps casual mentions from triggering it.
- v3.0.13 (2026-11-06): The webcam two-hand window grab stops dropping the window. From a recording of the real tracker on the Dell: a held fist often reads as a pinch, which ended the grab, and a tiny phantom second hand kept replacing the real one. A grab now holds through both, and opening a hand or losing one still lets go. The hand mouse ignores the phantom too.
- v3.0.12 (2026-11-05): Test tooling only, nothing he says or does changes. GitHub's test run for v3.0.11 failed, and every local and server run passes, including a copy of GitHub's setup. GitHub's log needs a sign-in, so a failed run now names its failing tests in the public annotations.
- v3.0.11 (2026-11-05): From today's live log. Asked what you're holding, wearing or showing him, JARVIS now looks through the camera before he answers, and says so when he couldn't get a look or the picture isn't current, instead of inventing an object. 'Look again' looks again through the camera, and a plain 'try again' only does so right after a look. He no longer promises to update a visual database he doesn't have. 'What did you say?' repeats his last line word for word, 'Apologies, sir' isn't said three times in one turn, and a plain 'look again' no longer counts as stress. 'Messages on TV' is corrected to Teams when you're talking about messages. A new Teams check says what it can really read, and says plainly when it isn't signed in, so he no longer gives a count nothing read. Idle remarks about the silence or how productive you've been are gone, and screen remarks that could never be spoken no longer cost a brain call. Lines left in the queue from days ago no longer play at startup: that was the 'Good evening' briefing in the morning. The greeting now comes before the recap, a failed news check no longer becomes 'nothing of note overnight', and the news gets time while the brain warms up. Claps during video or TV are ignored, a late 'You rang?' is dropped, and old memory warnings and his own microphone choice stay out of the recap. The log now also counts how often the second speech check actually helps.
- v3.0.10 (2026-11-03): From a full sweep of the live log. While a movie or video plays, JARVIS holds his own remarks and status lines and says them when it ends; anything you asked for still comes through. Proactive lines no longer borrow an hours-old tone of voice. The morning arrival no longer piles a nudge on top of the recap and briefing, and a screen nudge never fires for a screen nobody is at. The memory alert names what is using the memory, repeats only when it gets worse, and stays quiet while JARVIS starts up. One swear word no longer counts as stress, and the 'calming lights' step is off, since it only read the device list and never dimmed anything here. Pause and resume answer at once without a second model round. A misheard line no longer runs a server or printer check nobody asked for, while 'give me a status report' still gets one. Asked about something he just did, JARVIS says plainly what ran. 'Print complete' is no longer dropped as stale, each print milestone is one line, and a held 25% is replaced by the newer reading. The echo canceller's timing estimate is kept within real-world limits. Until a TV area is calibrated, the TV detector ignores people moving around the room.
- v3.0.9 (2026-10-29): Fixes from tonight's live log, while you watched a movie. 'Jarvis, say X' is now said by JARVIS himself. The robot speaks only when you name it or ask him to talk ('have the bot say hi', 'make him talk'), and that rule is enforced, not just logged. 'All right' or 'No, not right now' never re-runs the last action, and a stray 'On it' after a no is dropped instead of starting a second round. When JARVIS is about to promise a change to how he works ('I'll build in a pause'), he now says he can't change that himself and notes it for Claude. 'Resume the movie' resumes it; it used to search Apple Music for a song called Resume. A song you name ('play Carry On') still plays. MakerWorld, Printables, Thingiverse and Cults open by name. A yes to 'search for it?' runs one search, and a browser task only runs on top when you asked for the page to be read. 'How much longer on my print?' gives the time left and says finished when it is, and the printer is asked for a full status as soon as JARVIS connects. With the robot skill update that loads at the same restart, JARVIS waits for the robot to finish its line before he speaks.
- v3.0.8 (2026-10-28): Faster answers and fewer freezes, from tonight's live checks. 'What's new' says the version and reads the summary once, without a second model round. 'What can you do' names seven skills and offers the rest. Web-browsing tasks run in the background: JARVIS says 'Working on it, sir', stays free to talk, and tells you the result when it is done ('stop' or 'never mind' cancels it). The morning briefing says 'sir' once, and 'I'll hand it to Claude' after a question refers back to that question. A cheaper at-desk face detector runs side by side with the current one and only logs how often they agree, so it can be switched on once it proves itself. A Kinect ghost-skeleton filter and a switch to keep the Claude safety net after November 5 are built but off by default.
- v3.0.7 (2026-10-28): Three fixes from tonight's live log. 'Why is the RAM so high?' or 'what's using all my memory?' is answered at once with the number and the biggest memory users, even below the high mark, and JARVIS's own thinking on the GPU is said as such. In the follow-up window after he answers, a voice that does not clearly match the owner no longer keeps the window open, and its remarks that are not a command or a question for JARVIS are ignored, so he stops replying to other people in the room. A spoken 'I've noted that' when no note was made is replaced with an honest 'I haven't made a note of that, sir.'
- v3.0.6 (2026-10-28): Honesty, routing and the server, from today's log review and live test. Plain timers no longer get a made-up 'check the oven' label, stock lines that claim work ('I'll note that for posterity') are gone, and when a chain of steps ends on a promise he did not keep, JARVIS now says so. Ideas and suggestions are answered in words instead of opening a web search, and no tool names are spoken. 'Can you open it up?' with nothing to open gets 'Open what, sir?'; developer to-dos ('we need to give the robot more voice lines') become notes for Claude; a flat 'I've no way to check that' offers the nearest read-only action or a note. 'How long is left on that timer?' and 'what can you do?' answer instantly, 'what's new' starts with the version, and his own GPU load while thinking is called that. The server watch also reads disk errors and temperatures, the UPS on-battery status and stopped server apps, read-only with fixed commands. Audio heard while the PC plays media, with an unsure voice, no longer keeps the follow-up window open. YouTube on a named monitor no longer searches for the monitor words.
- v3.0.5 (2026-10-28): 'Pass it over to Claude' and the RAM answer, from this afternoon's live log. At 15:58 'No, just pass over to Clog for me' was misheard and matched nothing, so a browser robot searched the web for 'Clog' for 47 seconds. The hand-off now needs no 'it', allows 'for me', 'please' or 'now' at the end, and accepts the speech-to-text's own mishearings of Claude (clog, claw, clawed, claud) only inside the hand-off phrase, so nozzle clogs and 'send it to the cloud' keep their meaning. Questions about a hand-off ('what did you send to Claude?') are no longer taken as one, and a bare hand-off that still reaches the browser robot gets a quick question instead of a web search. 'Why is the RAM so high?' no longer answers 'All systems nominal' at 92 percent: an asked-for system report opens with the reading that is out of line and names the biggest memory users (for example: my local brain model at 8 gigabytes, Claude at 6.5 and Chrome at 4.5); JARVIS's own GPU use is not reported as a problem, and the PC check names the memory users too.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
