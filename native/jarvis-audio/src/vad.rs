//! Energy-based voice-activity detection + endpointing. Consumes i16 PCM frames
//! (from the capture ring buffer), tracks short-term energy (RMS), and emits an
//! endpointed utterance once speech is followed by enough trailing silence.
//!
//! This is the deterministic, fully-tested baseline. A neural VAD (Silero) +
//! openWakeWord wake detection swap in behind the same `push_frame` interface in
//! a later increment; the wake-word gate is what finally kills the
//! "Whisper-as-wake-word" GPU cost the M1 design calls out.

/// Events emitted as frames are fed in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadEvent {
    /// Voice activity just began.
    SpeechStart,
    /// Voice activity ended; payload is the endpointed utterance PCM.
    SpeechEnd(Vec<i16>),
}

/// A small energy-gate VAD + endpointer.
pub struct Vad {
    threshold: f32,       // normalized RMS (0..~1) above which a frame is "speech"
    hangover_frames: u32, // trailing silent frames required to end an utterance
    in_speech: bool,
    silent_run: u32,
    utterance: Vec<i16>,
}

impl Vad {
    /// `threshold` is normalized RMS (e.g. 0.01–0.05); `hangover_frames` is how
    /// many consecutive silent frames end an utterance.
    pub fn new(threshold: f32, hangover_frames: u32) -> Self {
        Vad {
            threshold,
            hangover_frames,
            in_speech: false,
            silent_run: 0,
            utterance: Vec::new(),
        }
    }

    /// Root-mean-square energy of a frame, normalized to ~0..1 for i16 PCM.
    pub fn rms(frame: &[i16]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
        ((sum_sq / frame.len() as f64).sqrt() / i16::MAX as f64) as f32
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Feed one frame; returns any events it produced.
    pub fn push_frame(&mut self, frame: &[i16]) -> Vec<VadEvent> {
        let mut events = Vec::new();
        let loud = Self::rms(frame) >= self.threshold;
        if !self.in_speech {
            if loud {
                self.in_speech = true;
                self.silent_run = 0;
                self.utterance.clear();
                self.utterance.extend_from_slice(frame);
                events.push(VadEvent::SpeechStart);
            }
        } else {
            self.utterance.extend_from_slice(frame);
            if loud {
                self.silent_run = 0;
            } else {
                self.silent_run += 1;
                if self.silent_run >= self.hangover_frames {
                    let utt = std::mem::take(&mut self.utterance);
                    self.in_speech = false;
                    self.silent_run = 0;
                    events.push(VadEvent::SpeechEnd(utt));
                }
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loud() -> Vec<i16> {
        vec![10_000; 160] // rms ~0.305
    }
    fn quiet() -> Vec<i16> {
        vec![0; 160] // rms 0
    }

    #[test]
    fn rms_basics() {
        assert_eq!(Vad::rms(&[]), 0.0);
        assert_eq!(Vad::rms(&quiet()), 0.0);
        assert!(Vad::rms(&loud()) > 0.2);
    }

    #[test]
    fn silence_emits_nothing() {
        let mut v = Vad::new(0.1, 2);
        assert!(v.push_frame(&quiet()).is_empty());
        assert!(!v.in_speech());
    }

    #[test]
    fn detects_start_and_endpoints_after_hangover() {
        let mut v = Vad::new(0.1, 2);
        assert_eq!(v.push_frame(&loud()), vec![VadEvent::SpeechStart]);
        assert!(v.in_speech());
        assert!(v.push_frame(&loud()).is_empty()); // still talking
        assert!(v.push_frame(&quiet()).is_empty()); // silent_run = 1 < 2
        let evs = v.push_frame(&quiet()); // silent_run = 2 -> endpoint
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            VadEvent::SpeechEnd(utt) => {
                // accumulated 2 loud + 2 quiet frames of 160 samples each
                assert_eq!(utt.len(), 4 * 160);
            }
            other => panic!("expected SpeechEnd, got {other:?}"),
        }
        assert!(!v.in_speech());
    }

    #[test]
    fn brief_dip_does_not_endpoint() {
        let mut v = Vad::new(0.1, 3);
        v.push_frame(&loud());
        v.push_frame(&quiet()); // dip 1
        let evs = v.push_frame(&loud()); // back to speech -> resets silent_run
        assert!(evs.is_empty());
        assert!(v.in_speech());
    }
}
