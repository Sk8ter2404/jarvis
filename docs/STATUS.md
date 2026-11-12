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

### In progress (1)

- Dead-microphone warning never fires. In progress.

### Planned next (2)

- Fact-check chain: wrong answer, then 75 seconds of fumbling. Planned next.
- Every sound is transcribed twice for nothing. Planned next.

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

- v3.0.18 (2026-11-12): Hand tracking and conversation fixes from the 11-12 checks. Hands: camera frames are no longer thrown away (22 a second becomes about 30), a closed fist is never read as a pinch, and a hand lost for a moment mid-grab freezes the window instead of dropping it. Talking: "repeat yourself" gives only his last line, no jokes while you are annoyed or mid-conversation, the wasted second transcription is gone (2-3 seconds saved on most sounds), and web search waits for real results. Hand changes are proven on recordings and simulated hands, not yet live.
- v3.0.17 (2026-11-12): A real alarm. "Wake me up at 7" or "alarm in 20 minutes" now rings a loud tone on the desk speakers at full volume and he talks over it ("Sir, you need to wake up.") until you say stop, say snooze, or touch the keyboard or mouse. It gives up after 10 minutes, puts the volume back, and saved alarms survive a restart. Before this, asking for an alarm set a timer that spoke one quiet line.
- v3.0.16 (2026-11-12): Finishes the v3.0.15 microphone-switch fix. Live at 11:10 he followed a switch but the device-list refresh it asked for kept being postponed, because his speaker-side checks ran first and used up the slot. His listening loop now makes room for an owed refresh before each listen.
- v3.0.15 (2026-11-12): He went deaf after a microphone switch on 11-12: Windows moved the default mic, he announced the new one, and then heard nothing until a restart. A followed switch now also refreshes his device list, which is what the restart did. The dead-microphone warning also fires for a muted mic now; its threshold sat exactly on a muted mic's background level.
- v3.0.14 (2026-11-10): From today's live log. A sworn "what did you just say?" now gets his last line back word for word (it started a robot banter action instead). "Shut up, stop listening to me" now puts him to sleep: insults and courtesy no longer count toward the six-word limit that keeps casual mentions from triggering it.
- v3.0.13 (2026-11-06): The webcam two-hand window grab stops dropping the window. From a recording of the real tracker on the Dell: a held fist often reads as a pinch, which ended the grab, and a tiny phantom second hand kept replacing the real one. A grab now holds through both, and opening a hand or losing one still lets go. The hand mouse ignores the phantom too.
- v3.0.12 (2026-11-05): Test tooling only, nothing he says or does changes. GitHub's test run for v3.0.11 failed, and every local and server run passes, including a copy of GitHub's setup. GitHub's log needs a sign-in, so a failed run now names its failing tests in the public annotations.
- v3.0.11 (2026-11-05): From today's live log. Asked what you're holding, wearing or showing him, JARVIS now looks through the camera before he answers, and says so when he couldn't get a look or the picture isn't current, instead of inventing an object. 'Look again' looks again through the camera, and a plain 'try again' only does so right after a look. He no longer promises to update a visual database he doesn't have. 'What did you say?' repeats his last line word for word, 'Apologies, sir' isn't said three times in one turn, and a plain 'look again' no longer counts as stress. 'Messages on TV' is corrected to Teams when you're talking about messages. A new Teams check says what it can really read, and says plainly when it isn't signed in, so he no longer gives a count nothing read. Idle remarks about the silence or how productive you've been are gone, and screen remarks that could never be spoken no longer cost a brain call. Lines left in the queue from days ago no longer play at startup: that was the 'Good evening' briefing in the morning. The greeting now comes before the recap, a failed news check no longer becomes 'nothing of note overnight', and the news gets time while the brain warms up. Claps during video or TV are ignored, a late 'You rang?' is dropped, and old memory warnings and his own microphone choice stay out of the recap. The log now also counts how often the second speech check actually helps.
- v3.0.10 (2026-11-03): From a full sweep of the live log. While a movie or video plays, JARVIS holds his own remarks and status lines and says them when it ends; anything you asked for still comes through. Proactive lines no longer borrow an hours-old tone of voice. The morning arrival no longer piles a nudge on top of the recap and briefing, and a screen nudge never fires for a screen nobody is at. The memory alert names what is using the memory, repeats only when it gets worse, and stays quiet while JARVIS starts up. One swear word no longer counts as stress, and the 'calming lights' step is off, since it only read the device list and never dimmed anything here. Pause and resume answer at once without a second model round. A misheard line no longer runs a server or printer check nobody asked for, while 'give me a status report' still gets one. Asked about something he just did, JARVIS says plainly what ran. 'Print complete' is no longer dropped as stale, each print milestone is one line, and a held 25% is replaced by the newer reading. The echo canceller's timing estimate is kept within real-world limits. Until a TV area is calibrated, the TV detector ignores people moving around the room.
- v3.0.9 (2026-10-29): Fixes from tonight's live log, while you watched a movie. 'Jarvis, say X' is now said by JARVIS himself. The robot speaks only when you name it or ask him to talk ('have the bot say hi', 'make him talk'), and that rule is enforced, not just logged. 'All right' or 'No, not right now' never re-runs the last action, and a stray 'On it' after a no is dropped instead of starting a second round. When JARVIS is about to promise a change to how he works ('I'll build in a pause'), he now says he can't change that himself and notes it for Claude. 'Resume the movie' resumes it; it used to search Apple Music for a song called Resume. A song you name ('play Carry On') still plays. MakerWorld, Printables, Thingiverse and Cults open by name. A yes to 'search for it?' runs one search, and a browser task only runs on top when you asked for the page to be read. 'How much longer on my print?' gives the time left and says finished when it is, and the printer is asked for a full status as soon as JARVIS connects. With the robot skill update that loads at the same restart, JARVIS waits for the robot to finish its line before he speaks.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
