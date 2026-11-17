# JARVIS status

Generated from the maintainer's private master sheet. Do not edit by hand.

## Live now

- Version: v3.0.23
- Live since: 2026-11-16
- Latest change: v3.0.24 (11-16 evening, pushed): the camera opens after the boot (the Dell's 3.75-min boot freeze), and face detection slows while a hand is in view (spotty hand tracking). Desktop has it on disk, running v3.0.23 until the next restart; the Dell has the files, unproven (shut down).

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

### Open, needs a decision (1)

- Mic heard nothing from 4:25 to 7:03 AM. Open: waiting on a design decision.

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

- v3.0.25 (2026-11-17): Fixes from a full review of the 11-16 and 11-17 logs, where he misread about a third of the turns. "Open YouTube" opens the YouTube home page instead of a random Short. "Time to play Fortnite" turns on game mode and starts Fortnite through the Epic launcher (it used to fail). "Did anything happen while I was at school?" gets the real away recap instead of a made-up "I've checked the logs". Game mode can no longer switch on from a misheard line: it needs you to mention a game, gaming or power (a mishearing on the Dell had turned off the camera and the hand mouse). "Etsy" is heard correctly. In wake-word mode the camera now opens as soon as he is listening asleep, not after 45 seconds. Hand tracking on the Kinect: frames reach the tracker at the camera's own speed (about 30 a second instead of about 8), a grab survives a missed frame or a left/right mix-up, the size limits fit the Kinect's wide view, dim frames are brightened before tracking, and the tracker logs its real frame rate and brightness.
- v3.0.24 (2026-11-16): Two fixes from the Dell tonight. The boot no longer freezes behind the webcam: both first boots there sat about 3.5 minutes with no voice, no listening and no hand tracking, because the webcam's first open hung inside its driver and held up the boot line until it came back. The camera now opens once he is already listening (45 seconds at most), and the sound library it was stuck on loads before any camera opens. Hand tracking is steadier: with his face in view the face detector ran on every frame and took half the Dell's processor, so the hand tracker kept losing his hand. While a hand is in view, face detection now runs once a second.
- v3.0.23 (2026-11-16): The alarm starts quiet and gets louder, with a ceiling. The first real ring (11-16 09:20) jumped the PC from 18 % straight to 100 %, clipped the speakers and played through the whole house. Now it starts at 15 %, rises a step every 3 seconds, reaches 50 % after 45 seconds and never goes above it, even when the volume was higher before. His wake-up lines ride the same volume. Your own volume and mute come back when it stops.
- v3.0.22 (2026-11-13): "What are the results?" now gets a spoken answer. In the v3.0.21 live check he picked the right command and read the page, then said the results were "too long" because that path handed the brain an oversized prompt. The command now speaks its own finished answer: the top three results, numbered, straight from the page. The search also waits up to 4 seconds for results to appear instead of 2.
- v3.0.21 (2026-11-13): Web search can read its results again. Asked about the Outlook outage (11-13 11:12), he opened the search, read nothing off the page, offered "headlines" and read the news briefing instead. Google's result links had grown taller than the size his page reader accepted, so every real result was thrown out. Results are read again, with the web address trimmed off each title, and "what are the results" / "read the results" now has a real action behind it instead of a second search.
- v3.0.20 (2026-11-12): Alarm made dependable after its first live try and two review passes. A plain "set an alarm for 4:22" or "wake me up at 6:30" is set straight away with no brain call (the brain had answered "422 what?"). Times in words work ("seven thirty", "half past six", "7 am tomorrow"), and the time confirmed is the time asked (7:00 was confirmed as 6:59). Ringing: the volume raise and speaker switch now work from the alarm's own thread, a second alarm due during a ring is no longer lost, an unanswered alarm rings again after 5 minutes instead of giving up, an alarm survives a restart mid-ring, and "stop" works in standby. Room talk such as "the TV is off" no longer ends a ring. "Cancel the 7 am alarm" works. Not yet rung live.
- v3.0.19 (2026-11-12): Alarm fix from the first live try. "Set an alarm for 4:22" was heard as "422" and set 422 minutes ahead (11:23 PM). A spoken clock time ("422", "4 22", "1030") is now read as a time, and one with no AM or PM means the next time the clock reads it. A number with a unit ("100 seconds") or after "in" is still a length of time.
- v3.0.18 (2026-11-12): Hand tracking and conversation fixes from the 11-12 checks. Hands: camera frames are no longer thrown away (22 a second becomes about 30), a closed fist is never read as a pinch, and a hand lost for a moment mid-grab freezes the window instead of dropping it. Talking: "repeat yourself" gives only his last line, no jokes while you are annoyed or mid-conversation, the wasted second transcription is gone (2-3 seconds saved on most sounds), and web search waits for real results. Hand changes are proven on recordings and simulated hands, not yet live.
- v3.0.17 (2026-11-12): A real alarm. "Wake me up at 7" or "alarm in 20 minutes" now rings a loud tone on the desk speakers at full volume and he talks over it ("Sir, you need to wake up.") until you say stop, say snooze, or touch the keyboard or mouse. It gives up after 10 minutes, puts the volume back, and saved alarms survive a restart. Before this, asking for an alarm set a timer that spoke one quiet line.
- v3.0.16 (2026-11-12): Finishes the v3.0.15 microphone-switch fix. Live at 11:10 he followed a switch but the device-list refresh it asked for kept being postponed, because his speaker-side checks ran first and used up the slot. His listening loop now makes room for an owed refresh before each listen.

Full history: [RELEASE_HISTORY.md](RELEASE_HISTORY.md).
