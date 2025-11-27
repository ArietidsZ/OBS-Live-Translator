//! Audio processing pipeline

pub mod buffer;
pub mod mel_spectrogram;
pub mod preprocessing;
pub mod resampling;
pub mod vad;

pub use buffer::AudioBuffer;
pub use preprocessing::AudioPreprocessor;
pub use vad::SileroVAD;

use crate::Result;

/// Audio sample type
pub type Sample = f32;

/// Audio frame for processing
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Audio samples
    pub samples: Vec<Sample>,

    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Timestamp (optional)
    pub timestamp: Option<std::time::Instant>,
}

impl AudioFrame {
    /// Create a new audio frame
    pub fn new(samples: Vec<Sample>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            timestamp: Some(std::time::Instant::now()),
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    /// Convert to mono if stereo
    pub fn to_mono(&mut self) {
        if self.channels == 2 {
            let mut mono_samples = Vec::with_capacity(self.samples.len() / 2);
            for chunk in self.samples.chunks_exact(2) {
                mono_samples.push((chunk[0] + chunk[1]) / 2.0);
            }
            self.samples = mono_samples;
            self.channels = 1;
        }
    }

    /// Resample to target sample rate
    pub fn resample(&mut self, target_rate: u32) -> Result<()> {
        if self.sample_rate == target_rate {
            return Ok(());
        }

        let resampled =
            resampling::resample(&self.samples, self.sample_rate, target_rate, self.channels)?;

        self.samples = resampled;
        self.sample_rate = target_rate;

        Ok(())
    }
}
