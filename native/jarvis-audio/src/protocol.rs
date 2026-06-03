//! Wire protocol for the native audio service <-> Python brain IPC.
//!
//! Framing: every message is `[1-byte tag][4-byte big-endian length][payload]`.
//! `Event`s flow service -> brain; `Control`s flow brain -> service.
//! (Architecture: `docs/design/M1-native-audio-service.md` §3.)

/// Protocol version announced in the `Hello` handshake.
pub const PROTOCOL_VERSION: u8 = 1;

// ── Event tags (service -> brain) ──
const TAG_HELLO: u8 = 0x01;
const TAG_WAKE: u8 = 0x02;
const TAG_VAD_START: u8 = 0x03;
const TAG_AUDIO_FRAME: u8 = 0x04;
const TAG_VAD_END: u8 = 0x05;
const TAG_BARGE_IN: u8 = 0x06;
const TAG_LOG: u8 = 0x07;

// ── Control tags (brain -> service) ──
const TAG_MUTE: u8 = 0x10;
const TAG_UNMUTE: u8 = 0x11;
const TAG_DUCK: u8 = 0x12;
const TAG_START: u8 = 0x13;
const TAG_STOP: u8 = 0x14;

/// A message from the audio service to the Python brain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Handshake: the protocol version the service speaks.
    Hello(u8),
    /// The wake word was detected.
    WakeDetected,
    /// Voice activity started.
    VadStart,
    /// A chunk of 16 kHz / 16-bit mono PCM.
    AudioFrame(Vec<u8>),
    /// Voice activity ended; payload is the endpointed utterance PCM.
    VadEnd(Vec<u8>),
    /// The user spoke over JARVIS (barge-in).
    BargeIn,
    /// A diagnostic line (used heavily in shadow mode).
    Log(String),
}

/// A control message from the Python brain to the audio service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Mute,
    Unmute,
    Duck,
    StartListening,
    Stop,
}

fn frame(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(tag);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

impl Event {
    /// Encode to a framed byte buffer.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Event::Hello(v) => frame(TAG_HELLO, &[*v]),
            Event::WakeDetected => frame(TAG_WAKE, &[]),
            Event::VadStart => frame(TAG_VAD_START, &[]),
            Event::AudioFrame(pcm) => frame(TAG_AUDIO_FRAME, pcm),
            Event::VadEnd(pcm) => frame(TAG_VAD_END, pcm),
            Event::BargeIn => frame(TAG_BARGE_IN, &[]),
            Event::Log(s) => frame(TAG_LOG, s.as_bytes()),
        }
    }

    /// Decode one framed message from the front of `buf`. Returns
    /// `(event, bytes_consumed)`, or `None` if `buf` doesn't yet hold a complete
    /// frame, the length is implausible, or the tag is unknown.
    pub fn decode(buf: &[u8]) -> Option<(Event, usize)> {
        if buf.len() < 5 {
            return None;
        }
        let tag = buf[0];
        let len = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
        let end = 5usize.checked_add(len)?;
        if buf.len() < end {
            return None;
        }
        let payload = &buf[5..end];
        let ev = match tag {
            TAG_HELLO if len == 1 => Event::Hello(payload[0]),
            TAG_WAKE => Event::WakeDetected,
            TAG_VAD_START => Event::VadStart,
            TAG_AUDIO_FRAME => Event::AudioFrame(payload.to_vec()),
            TAG_VAD_END => Event::VadEnd(payload.to_vec()),
            TAG_BARGE_IN => Event::BargeIn,
            TAG_LOG => Event::Log(String::from_utf8_lossy(payload).into_owned()),
            _ => return None,
        };
        Some((ev, end))
    }
}

impl Control {
    /// The single-byte tag for this control message.
    pub fn tag(self) -> u8 {
        match self {
            Control::Mute => TAG_MUTE,
            Control::Unmute => TAG_UNMUTE,
            Control::Duck => TAG_DUCK,
            Control::StartListening => TAG_START,
            Control::Stop => TAG_STOP,
        }
    }

    /// Parse a control message from its tag byte, or `None` if unknown.
    pub fn from_tag(tag: u8) -> Option<Control> {
        match tag {
            TAG_MUTE => Some(Control::Mute),
            TAG_UNMUTE => Some(Control::Unmute),
            TAG_DUCK => Some(Control::Duck),
            TAG_START => Some(Control::StartListening),
            TAG_STOP => Some(Control::Stop),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(ev: Event) {
        let bytes = ev.encode();
        let (decoded, n) = Event::decode(&bytes).expect("decodes");
        assert_eq!(decoded, ev);
        assert_eq!(n, bytes.len());
    }

    #[test]
    fn roundtrips_all_events() {
        roundtrip(Event::Hello(PROTOCOL_VERSION));
        roundtrip(Event::WakeDetected);
        roundtrip(Event::VadStart);
        roundtrip(Event::AudioFrame(vec![1, 2, 3, 4]));
        roundtrip(Event::VadEnd(vec![9, 8, 7]));
        roundtrip(Event::BargeIn);
        roundtrip(Event::Log("hello sir".into()));
    }

    #[test]
    fn decode_needs_a_full_frame() {
        let bytes = Event::AudioFrame(vec![0u8; 10]).encode();
        assert!(Event::decode(&bytes[..4]).is_none()); // header incomplete
        assert!(Event::decode(&bytes[..bytes.len() - 1]).is_none()); // payload short
        assert!(Event::decode(&bytes).is_some());
    }

    #[test]
    fn decode_rejects_unknown_tag() {
        let bad = [0xFFu8, 0, 0, 0, 0];
        assert!(Event::decode(&bad).is_none());
    }

    #[test]
    fn decode_two_back_to_back() {
        let mut buf = Event::WakeDetected.encode();
        buf.extend(Event::Log("x".into()).encode());
        let (e1, n1) = Event::decode(&buf).unwrap();
        assert_eq!(e1, Event::WakeDetected);
        let (e2, _) = Event::decode(&buf[n1..]).unwrap();
        assert_eq!(e2, Event::Log("x".into()));
    }

    #[test]
    fn control_tags_roundtrip() {
        for c in [
            Control::Mute,
            Control::Unmute,
            Control::Duck,
            Control::StartListening,
            Control::Stop,
        ] {
            assert_eq!(Control::from_tag(c.tag()), Some(c));
        }
        assert_eq!(Control::from_tag(0x99), None);
    }
}
