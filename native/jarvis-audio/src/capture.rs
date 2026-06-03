//! Microphone capture via cpal. The default input device is opened and its
//! samples are converted to 16-bit PCM and pushed into a shared `RingBuffer`.
//!
//! The live input stream is hardware-dependent (WASAPI on Windows), so it's
//! exercised manually (`cargo run`); the pure helpers (sample conversion) are
//! unit-tested. Wake-word + VAD + endpointing consume the ring buffer in the
//! next increment.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::ring_buffer::RingBuffer;

/// Convert one f32 sample (nominally -1.0..=1.0) to 16-bit PCM, clamping.
pub fn f32_to_i16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// Convert a slice of f32 samples to 16-bit PCM.
pub fn f32_slice_to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&x| f32_to_i16(x)).collect()
}

/// A short human description of the default input device + its config, or an
/// error string. Touches the audio host, so it's a smoke helper, not a unit
/// test target.
pub fn default_input_description() -> Result<String, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "no default input device".to_string())?;
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let cfg = device
        .default_input_config()
        .map_err(|e| format!("no default input config: {e}"))?;
    Ok(format!(
        "{name} @ {} Hz, {} ch, {:?}",
        cfg.sample_rate().0,
        cfg.channels(),
        cfg.sample_format()
    ))
}

/// Open the default input device and stream its samples (converted to i16 PCM)
/// into `ring`. Returns the live cpal stream — KEEP IT ALIVE to keep capturing;
/// dropping it stops capture. Hardware-dependent; run via `cargo run`, not unit
/// tests. Total: returns an error string rather than panicking on any failure.
pub fn start_capture(ring: Arc<Mutex<RingBuffer>>) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "no default input device".to_string())?;
    let cfg = device
        .default_input_config()
        .map_err(|e| format!("no default input config: {e}"))?;
    let err_fn = |e| eprintln!("[capture] stream error: {e}");
    let config: cpal::StreamConfig = cfg.clone().into();

    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut r) = ring.lock() {
                    for &x in data {
                        r.push(f32_to_i16(x));
                    }
                }
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                if let Ok(mut r) = ring.lock() {
                    r.push_slice(data);
                }
            },
            err_fn,
            None,
        ),
        other => return Err(format!("unsupported sample format {other:?}")),
    }
    .map_err(|e| format!("build_input_stream failed: {e}"))?;

    stream.play().map_err(|e| format!("stream play failed: {e}"))?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_conversion_clamps_and_scales() {
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(2.0), i16::MAX); // clamped
        assert_eq!(f32_to_i16(-2.0), -i16::MAX); // clamped to -32767
    }

    #[test]
    fn f32_slice_conversion() {
        let out = f32_slice_to_i16(&[0.0, 1.0, -1.0]);
        assert_eq!(out, vec![0, i16::MAX, -i16::MAX]);
    }
}
