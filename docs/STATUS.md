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

### In progress (3)

- Dead-microphone warning never fires. In progress.
- Fact-check chain: wrong answer, then 75 seconds of fumbling. In progress.
- Every sound is transcribed twice for nothing. In progress.

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

- v3.0.21 (2026-11-13): Web search can read its results again. Asked about the Outlook outage (11-13 11:12), he opened the search, read nothing off the page, offered "headlines" and read the news briefing instead. Google's result links had grown taller than the size his page reader accepted, so every real result was thrown out. Results are read again, with the web address trimmed off each title, and "what are the results" / "read the results" now has a real action behind it instead of a second search.
- v3.0.20 (2026-11-12): Alarm made dependable after its first live try and two review passes. A plain "set an alarm for 4:22" or "wake me up at 6:30" is set straight away with no brain call (the brain had answered "422 what?"). Times in words work ("seven thirty", "half past six", "7 am tomorrow"), and the time confirmed is the time asked (7:00 was confirmed as 6:59). Ringing: the volume raise and speaker switch now work from the alarm's own thread, a second alarm due during a ring is no longer lost, an unanswered alarm rings again after 5 minutes instead of giving up, an alarm survives a restart mid-ring, and "stop" works in standby. Room talk such as "the TV is off" no longer ends a ring. "Cancel the 7 am alarm" works. Not yet rung live.
- v3.0.19 (2026-11-12): Alarm fix from the first live try. "Set an alarm for 4:22" was heard as "422" and set 422 minutes ahead (11:23 PM). A spoken clock time ("422", "4 22", "1030") is now read as a time, and one with no AM or PM means the next time the clock reads it. A number with a unit ("100 seconds") or after "in" is still a length of time.
- v3.0.18 (2026-11-12): Hand tracking and conversation fixes from the 11-12 checks. Hands: camera frames are no longer thrown away (22 a second becomes about 30), a closed fist is never read as a pinch, and a hand lost for a moment mid-grab freezes the window instead of dropping it. Talking: "repeat yourself" gives only his last line, no jokes while you are annoyed or mid-conversation, the wasted second transcription is gone (2-3 seconds saved on most sounds), and web search waits for real results. Hand changes are proven on recordings and simulated hands, not yet live.
- v3.0.17 (2026-11-12): A real alarm. "Wake me up at 7" or "alarm in 20 minutes" now rings a loud tone on the desk speakers at full volume and he talks over it ("Sir, you need to wake up.") until you say stop, say snooze, or touch the keyboard or mouse. It gives up after 10 minutes, puts the volume back, and saved alarms survive a restart. Before this, asking for an alarm set a timer that spoke one quiet line.
- v3.0.16 (2026-11-12): Finishes the v3.0.15 microphone-switch fix. Live at 11:10 he followed a switch but the device-list refresh it asked for kept being postponed, because his speaker-side checks ran first and used up the slot. His listening loop now makes room for an owed refresh before each listen.
- v3.0.15 (2026-11-12): He went deaf after a microphone switch on 11-12: Windows moved the default mic, he announced the new one, and then heard nothing until a restart. A followed switch now also refreshes his device list, which is what the restart did. The dead-microphone warning also fires for a muted mic now; its threshold sat exactly on a muted mic's background level.
- v3.0.14 (2026-11-10): From today's live log. A sworn "what did you just say?" now gets his last line back word for word (it started a robot banter action instead). "Shut up, stop listening to me" now puts him to sleep: insults and courtesy no longer count toward the six-word limit that keeps casual mentions from triggering it.
- v3.0.13 (2026-11-06): The webcam two-hand window grab stops dropping the window. From a recording of the real tracker on the Dell: a held fist often reads as a pinch, which ended the grab, and a tiny phantom second hand kept replacing the real one. A grab now holds through both, and opening a hand or losing one still lets go. The hand mouse ignores the phantom too.
- v3.0.12 (2026-11-05): Test tooling only, nothing he says or does changes. GitHub's test run for v3.0.11 failed, and every local and server run passes, including a copy of GitHub's setup. GitHub's log needs a sign-in, so a failed run now names its failing tests in the public annotations.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
