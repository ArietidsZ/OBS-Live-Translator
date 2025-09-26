//! Audio processing module with real-time DSP capabilities

pub mod processor;
pub mod resampler;
pub mod vad;
pub mod features;
pub mod whisper_mel;

pub use processor::AudioProcessor;
pub use resampler::Resampler;
pub use vad::VoiceActivityDetector;
pub use features::FeatureExtractor;
pub use whisper_mel::{audio_to_whisper_mel, prepare_mel_for_onnx, WhisperMelConfig};

/// Audio configuration parameters
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,  // Whisper standard
            channels: 1,         // Mono
            frame_size: 480,     // 30ms at 16kHz
            hop_length: 160,     // 10ms at 16kHz
            n_fft: 1024,
            n_mels: 80,          // Whisper standard
            f_min: 0.0,
            f_max: 8000.0,       // Nyquist for 16kHz
        }
    }
}

/// Audio sample format
pub type Sample = f32;

/// Audio buffer for streaming
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub data: Vec<Sample>,
    pub sample_rate: u32,
    pub channels: u16,
    pub timestamp: std::time::Instant,
}

impl AudioBuffer {
    pub fn new(capacity: usize, sample_rate: u32, channels: u16) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            sample_rate,
            channels,
            timestamp: std::time::Instant::now(),
        }
    }

    pub fn duration_ms(&self) -> f32 {
        (self.data.len() as f32 / self.sample_rate as f32) * 1000.0
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.timestamp = std::time::Instant::now();
    }

    pub fn append(&mut self, samples: &[Sample]) {
        self.data.extend_from_slice(samples);
    }
}