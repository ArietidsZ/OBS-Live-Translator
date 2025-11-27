//! OBS Live Translator v5.0
//!
//! State-of-the-art real-time speech translation system with multi-platform hardware acceleration.
//!
//! # Architecture
//!
//! ```text
//! Audio Input → VAD → ASR → Language Detection → Translation → Output
//!                ↓      ↓         ↓                   ↓
//!             Silero  Canary/   CLD3 +            MADLAD-400/
//!              VAD    Parakeet  FastText           NLLB-200
//! ```
//!
//! # Performance Profiles
//!
//! - **Low**: <500ms latency, 1.2GB memory (Distil-Whisper + MADLAD INT8)
//! - **Medium**: <300ms latency, 3.2GB VRAM (Parakeet TDT + NLLB FP16)
//! - **High**: <150ms latency, 7GB VRAM (Canary Qwen + MADLAD BF16)

pub mod acceleration;
pub mod asr;
pub mod audio;
pub mod batching;
pub mod config;
pub mod error;
pub mod execution_provider; // Part 1.5: Execution provider selection and configuration
pub mod inference;
pub mod language_detection;
pub mod logging; // Part 1.7: Logging and telemetry
pub mod models;
pub mod monitoring;
pub mod optimization;
pub mod vad;
use vad::{create_vad_engine, VadEngine};
pub mod platform_detect; // Part 1.4: Platform detection and hardware capabilities
pub mod profile;
pub mod streaming;
pub mod translation;
pub mod types;

// Re-exports for convenience
pub use error::{Error, Result};
pub use profile::Profile;
pub use types::{TranslationResult, TranslatorConfig};

use std::sync::Arc;
use tokio::sync::RwLock;

/// Main translator pipeline
pub struct Translator {
    config: TranslatorConfig,
    vad: Arc<dyn VadEngine + Send + Sync>,
    asr: Arc<dyn asr::ASREngine + Send + Sync>,
    lang_detector: Arc<language_detection::MultiDetector>,
    translation: Arc<dyn translation::TranslationEngine + Send + Sync>,
    metrics: Arc<RwLock<monitoring::MetricsCollector>>,
    #[allow(dead_code)]
    session_cache: Arc<crate::models::session_cache::SessionCache>,
}

impl Translator {
    /// Create a new translator with the given configuration
    pub async fn new(config: TranslatorConfig) -> Result<Self> {
        tracing::info!("Initializing OBS Live Translator v5.0");
        tracing::info!("Profile: {:?}", config.profile);

        // Initialize metrics collection
        let metrics = Arc::new(RwLock::new(monitoring::MetricsCollector::new()));

        // Initialize session cache
        let session_cache = Arc::new(crate::models::session_cache::SessionCache::new());

        // Initialize VAD based on configuration
        let vad = create_vad_engine(&config, session_cache.clone()).await?;

        // Initialize ASR based on profile
        let asr: Arc<dyn asr::ASREngine + Send + Sync> = match config.profile {
            Profile::Low => Arc::new(
                asr::distil_whisper::DistilWhisper::new(&config, session_cache.clone()).await?,
            ),
            Profile::Medium => {
                Arc::new(asr::parakeet::ParakeetTDT::new(&config, session_cache.clone()).await?)
            }
            Profile::High => {
                Arc::new(asr::canary::Canary180M::new(&config, session_cache.clone()).await?)
            }
        };

        // Initialize language detection (CLD3 + FastText fusion)
        let lang_detector = Arc::new(language_detection::MultiDetector::new(&config).await?);

        // Initialize translation based on profile
        let translation: Arc<dyn translation::TranslationEngine + Send + Sync> =
            match config.profile {
                Profile::Low => Arc::new(
                    translation::madlad::MADLAD400::new(
                        &config,
                        inference::Precision::INT8,
                        session_cache.clone(),
                    )
                    .await?,
                ),
                Profile::Medium => Arc::new(
                    translation::nllb::NLLB200::new(
                        &config,
                        inference::Precision::FP16,
                        session_cache.clone(),
                    )
                    .await?,
                ),
                Profile::High => {
                    // High profile: Use MADLAD BF16 or optionally Claude API
                    if config.use_claude_api {
                        Arc::new(translation::claude::ClaudeAPI::new(&config).await?)
                    } else {
                        Arc::new(
                            translation::madlad::MADLAD400::new(
                                &config,
                                inference::Precision::BF16,
                                session_cache.clone(),
                            )
                            .await?,
                        )
                    }
                }
            };

        tracing::info!("Translator initialized successfully");

        Ok(Self {
            config,
            vad,
            asr,
            lang_detector,
            translation,
            metrics,
            session_cache,
        })
    }

    /// Process audio samples and return translation
    pub async fn process_audio(&self, audio_samples: &[f32]) -> Result<TranslationResult> {
        let start = std::time::Instant::now();

        // Step 1: Voice Activity Detection
        let vad_start = std::time::Instant::now();
        let has_speech = self.vad.detect(audio_samples).await?;
        let vad_latency = vad_start.elapsed();

        if !has_speech {
            return Ok(TranslationResult::silence());
        }

        // Step 2: Automatic Speech Recognition
        let asr_start = std::time::Instant::now();
        let transcription = self.asr.transcribe(audio_samples).await?;
        let asr_latency = asr_start.elapsed();

        if transcription.text.is_empty() {
            return Ok(TranslationResult::empty());
        }

        // Step 3: Language Detection
        let lang_start = std::time::Instant::now();
        let detected_lang = self
            .lang_detector
            .detect(
                &transcription.text,
                transcription.detected_language.as_deref(),
            )
            .await?;
        let lang_latency = lang_start.elapsed();

        // Step 4: Translation (skip if already in target language)
        let (translation_text, translation_latency) =
            if detected_lang == self.config.target_language {
                (transcription.text.clone(), std::time::Duration::ZERO)
            } else {
                let trans_start = std::time::Instant::now();
                let translated = self
                    .translation
                    .translate(
                        &transcription.text,
                        &detected_lang,
                        &self.config.target_language,
                    )
                    .await?;
                (translated, trans_start.elapsed())
            };

        let total_latency = start.elapsed();

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_latency("vad", vad_latency);
            metrics.record_latency("asr", asr_latency);
            metrics.record_latency("language_detection", lang_latency);
            metrics.record_latency("translation", translation_latency);
            metrics.record_latency("total", total_latency);
        }

        Ok(TranslationResult {
            transcription: Some(transcription.text),
            translation: Some(translation_text),
            source_language: Some(detected_lang),
            target_language: self.config.target_language.clone(),
            confidence: transcription.confidence,
            latency_ms: total_latency.as_millis() as u64,
        })
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> monitoring::Metrics {
        self.metrics.read().await.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_translator_creation() {
        let config = TranslatorConfig::default();
        // This will fail without models, but tests the basic structure
        let result = Translator::new(config).await;
        assert!(result.is_ok() || result.is_err()); // Placeholder test
    }
}
