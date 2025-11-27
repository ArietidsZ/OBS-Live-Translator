//! Distil-Whisper Large v3 - Low Profile ASR
//!
//! 6x faster than base Whisper, 49% smaller, <1% WER degradation
//! INT8 quantization for minimal memory footprint

use crate::{
    asr::ASREngine,
    execution_provider::ExecutionProviderConfig,
    inference::{InferenceEngine, Precision},
    platform_detect::Platform,
    types::{Transcription, TranslatorConfig},
    Error, Result,
};
use async_trait::async_trait;
use ndarray::Array3;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Distil-Whisper Large v3 ASR engine
#[allow(dead_code)]
pub struct DistilWhisper {
    encoder: Arc<InferenceEngine>,
    decoder: Arc<InferenceEngine>,
    tokenizer: Arc<Tokenizer>,
    mel_extractor: crate::audio::mel_spectrogram::MelSpectrogramExtractor,
    sample_rate: u32,
}

impl DistilWhisper {
    /// Create a new Distil-Whisper instance
    pub async fn new(
        config: &TranslatorConfig,
        cache: Arc<crate::models::session_cache::SessionCache>,
    ) -> Result<Self> {
        tracing::info!("Loading Distil-Whisper Large v3 (INT8)");

        let model_path = &config.model_path;
        let encoder_path = format!("{model_path}/distil-whisper-large-v3-encoder-int8.onnx");
        let decoder_path = format!("{model_path}/distil-whisper-large-v3-decoder-int8.onnx");
        let tokenizer_path = format!("{model_path}/distil-whisper-large-v3-tokenizer.json");

        // Use CPU for low profile (memory efficient)
        // Note: Even though we prefer CPU for this specific model profile,
        // we'll use the platform detection to be consistent, but we could enforce CPU if needed.
        // For now, let's use the standard platform detection which might pick CoreML/TensorRT
        // if available, which is actually better than forcing CPU.
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

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenization(format!("Failed to load tokenizer: {e}")))?;

        let mel_extractor = crate::audio::mel_spectrogram::whisper_mel_extractor()?;

        tracing::info!("Distil-Whisper loaded successfully");

        Ok(Self {
            encoder: Arc::new(encoder),
            decoder: Arc::new(decoder),
            tokenizer: Arc::new(tokenizer),
            mel_extractor,
            sample_rate: 16000,
        })
    }

    /// Extract mel spectrogram features
    fn extract_features(&self, samples: &[f32]) -> Result<Array3<f32>> {
        self.mel_extractor.extract(samples)
    }
}

#[async_trait]
impl ASREngine for DistilWhisper {
    async fn transcribe(&self, samples: &[f32]) -> Result<Transcription> {
        // 1. Extract mel spectrogram features
        let _features = self.extract_features(samples)?;

        // 2. Create sessions for encoder/decoder
        let _encoder_session = self.encoder.get_session().await?;
        let _decoder_session = self.decoder.get_session().await?;

        // TODO: Run encoder and decoder inference
        // TODO: Implement beam search decoding

        // Placeholder transcription
        Ok(Transcription {
            text: String::new(),
            confidence: 0.0,
            timestamps: None,
            detected_language: None,
        })
    }

    fn model_name(&self) -> &str {
        "Distil-Whisper Large v3 (INT8)"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
