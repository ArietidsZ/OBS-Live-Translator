//! Canary Qwen 2.5B - High Profile ASR
//!
//! #1 on Hugging Face Open ASR Leaderboard (5.63% WER)
//! Speech-Augmented Language Model architecture

use crate::{
    asr::ASREngine,
    execution_provider::ExecutionProviderConfig,
    inference::{InferenceEngine, Precision},
    platform_detect::Platform,
    types::{Transcription, TranslatorConfig},
    Error, Result,
};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

/// Canary 180M ASR engine (Encoder-Decoder architecture)
#[allow(dead_code)]
pub struct Canary180M {
    encoder: Arc<InferenceEngine>,
    decoder: Arc<InferenceEngine>,
    sample_rate: u32,
    precision: Precision,
}

impl Canary180M {
    /// Create a new Canary 180M instance
    pub async fn new(
        config: &TranslatorConfig,
        cache: Arc<crate::models::session_cache::SessionCache>,
    ) -> Result<Self> {
        tracing::info!("Loading Canary 180M (INT8)");

        let model_path = &config.model_path;
        let encoder_path = format!("{model_path}/canary-180m-encoder.int8.onnx");
        let decoder_path = format!("{model_path}/canary-180m-decoder.int8.onnx");

        // Check if models exist
        if !Path::new(&encoder_path).exists() || !Path::new(&decoder_path).exists() {
            return Err(Error::ModelLoad(
                "Canary 180M model files not found. Please run scripts/download_canary_int8.py"
                    .to_string(),
            ));
        }

        // Detect platform and configure execution provider
        let platform = Platform::detect().map_err(|e| Error::Acceleration(e.to_string()))?;
        let provider_config = ExecutionProviderConfig::from_platform(platform);
        tracing::info!("Execution Provider: {}", provider_config.summary());

        // Load encoder and decoder
        let encoder = InferenceEngine::new(
            &encoder_path,
            Precision::INT8,
            &provider_config,
            &config.acceleration,
            cache.clone(),
        )?;
        let decoder = InferenceEngine::new(
            &decoder_path,
            Precision::INT8,
            &provider_config,
            &config.acceleration,
            cache,
        )?;

        tracing::info!("Canary 180M loaded successfully");

        Ok(Self {
            encoder: Arc::new(encoder),
            decoder: Arc::new(decoder),
            sample_rate: 16000,
            precision: Precision::INT8,
        })
    }
}

#[async_trait]
impl ASREngine for Canary180M {
    async fn transcribe(&self, _samples: &[f32]) -> Result<Transcription> {
        // TODO: Implement Canary 180M inference (Encoder-Decoder)
        // 1. Extract features (FastConformer expects 80-channel mel spectrogram)
        // 2. Run Encoder
        // 3. Run Decoder (autoregressive generation with task tokens)

        tracing::warn!("Canary 180M inference not yet implemented");

        Ok(Transcription {
            text: String::new(),
            confidence: 0.0,
            timestamps: None,
            detected_language: None, // TODO: Extract from decoder output (first token)
        })
    }

    fn model_name(&self) -> &str {
        "Canary 180M (INT8)"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
