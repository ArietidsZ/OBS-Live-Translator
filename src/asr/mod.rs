//! ASR (Automatic Speech Recognition) engines

pub mod beam_search;
pub mod canary;
pub mod distil_whisper;
pub mod parakeet;

use crate::{types::Transcription, Result};
use async_trait::async_trait;

/// ASR engine trait
#[async_trait]
pub trait ASREngine {
    /// Transcribe audio samples to text
    async fn transcribe(&self, samples: &[f32]) -> Result<Transcription>;

    /// Get model name
    fn model_name(&self) -> &str;

    /// Get supported sample rate
    fn sample_rate(&self) -> u32;
}
