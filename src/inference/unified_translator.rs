//! Unified translation manager that intelligently routes between different models
//!
//! Provides optimal model selection based on language pairs, quality requirements,
//! and performance constraints.

use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

use super::whisper_translate::{WhisperTranslator, WhisperCTranslate2, WhisperTask};
use super::nllb::{NLLB200Translator, NLLBModelSize};

/// Translation strategy
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TranslationStrategy {
    /// Use Whisper for speech-to-English translation
    WhisperDirect,
    /// Use Whisper for ASR + NLLB for translation
    WhisperPlusNLLB,
    /// Use SeamlessM4T for end-to-end
    SeamlessM4T,
    /// Use CTranslate2 optimized Whisper
    WhisperCTranslate2,
    /// Text-only translation with NLLB
    NLLBOnly,
}

/// Translation request
#[derive(Debug, Clone)]
pub struct TranslationRequest {
    /// Input type
    pub input_type: InputType,
    /// Source language (None for auto-detect)
    pub source_language: Option<String>,
    /// Target language
    pub target_language: String,
    /// Quality preference
    pub quality: QualityLevel,
    /// Maximum latency in milliseconds
    pub max_latency_ms: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputType {
    /// Raw audio samples
    Audio(Vec<f32>),
    /// Mel spectrogram features
    MelSpectrogram(Vec<f32>),
    /// Text input
    Text(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    /// Fastest possible, may sacrifice quality
    Draft,
    /// Balanced speed and quality
    Standard,
    /// Highest quality, may be slower
    Premium,
}

/// Unified translation response
#[derive(Debug, Clone)]
pub struct TranslationResponse {
    pub original_text: String,
    pub translated_text: String,
    pub source_language: String,
    pub target_language: String,
    pub confidence: f32,
    pub strategy_used: TranslationStrategy,
    pub processing_time_ms: f32,
    pub model_info: String,
}

/// Unified translator that manages multiple translation models
pub struct UnifiedTranslator {
    /// Whisper models
    whisper_standard: Option<WhisperTranslator>,
    whisper_ctranslate2: Option<WhisperCTranslate2>,

    /// NLLB models
    nllb_600m: Option<Arc<RwLock<NLLB200Translator>>>,
    nllb_1_3b: Option<Arc<RwLock<NLLB200Translator>>>,
    nllb_3_3b: Option<Arc<RwLock<NLLB200Translator>>>,

    /// Model capabilities cache
    #[allow(dead_code)]
    capabilities: ModelCapabilities,

    /// Performance statistics
    stats: Arc<RwLock<PerformanceStats>>,
}

impl UnifiedTranslator {
    /// Create a new unified translator
    pub fn new() -> Result<Self> {
        Ok(Self {
            whisper_standard: None,
            whisper_ctranslate2: None,
            nllb_600m: None,
            nllb_1_3b: None,
            nllb_3_3b: None,
            capabilities: ModelCapabilities::default(),
            stats: Arc::new(RwLock::new(PerformanceStats::default())),
        })
    }

    /// Initialize Whisper model
    pub fn init_whisper(&mut self, model_size: &str, use_ctranslate2: bool) -> Result<()> {
        if use_ctranslate2 {
            let model = WhisperCTranslate2::new(
                &format!("models/whisper-{}-ct2", model_size),
                "auto"
            )?;
            self.whisper_ctranslate2 = Some(model);
        } else {
            let model = WhisperTranslator::new(model_size)?;
            self.whisper_standard = Some(model);
        }
        Ok(())
    }

    /// Initialize NLLB model
    pub fn init_nllb(&mut self, model_size: NLLBModelSize) -> Result<()> {
        let model = NLLB200Translator::new(model_size)?;
        let model = Arc::new(RwLock::new(model));

        match model_size {
            NLLBModelSize::Distilled600M => self.nllb_600m = Some(model),
            NLLBModelSize::Distilled1_3B => self.nllb_1_3b = Some(model),
            NLLBModelSize::Base3_3B => self.nllb_3_3b = Some(model),
        }
        Ok(())
    }

    /// Translate with automatic model selection
    pub async fn translate(&self, request: TranslationRequest) -> Result<TranslationResponse> {
        let start = std::time::Instant::now();

        // Select optimal strategy
        let strategy = self.select_strategy(&request)?;

        // Execute translation based on strategy
        let response = match strategy {
            TranslationStrategy::WhisperDirect => {
                self.translate_whisper_direct(request).await?
            },
            TranslationStrategy::WhisperPlusNLLB => {
                self.translate_whisper_nllb(request).await?
            },
            TranslationStrategy::WhisperCTranslate2 => {
                self.translate_whisper_ct2(request).await?
            },
            TranslationStrategy::NLLBOnly => {
                self.translate_nllb_only(request).await?
            },
            TranslationStrategy::SeamlessM4T => {
                return Err(anyhow!("SeamlessM4T not implemented yet"));
            }
        };

        // Update statistics
        let processing_time = start.elapsed().as_secs_f32() * 1000.0;
        self.update_stats(&strategy, processing_time).await;

        Ok(TranslationResponse {
            processing_time_ms: processing_time,
            strategy_used: strategy,
            ..response
        })
    }

    /// Select optimal translation strategy
    fn select_strategy(&self, request: &TranslationRequest) -> Result<TranslationStrategy> {
        // If target is English and we have audio, prefer Whisper direct translation
        if request.target_language == "en" && matches!(request.input_type, InputType::Audio(_)) {
            if self.whisper_ctranslate2.is_some() && request.quality != QualityLevel::Premium {
                return Ok(TranslationStrategy::WhisperCTranslate2);
            }
            if self.whisper_standard.is_some() {
                return Ok(TranslationStrategy::WhisperDirect);
            }
        }

        // For text-only, use NLLB
        if matches!(request.input_type, InputType::Text(_)) {
            return Ok(TranslationStrategy::NLLBOnly);
        }

        // For non-English targets, use Whisper + NLLB
        if self.whisper_standard.is_some() && self.has_nllb() {
            return Ok(TranslationStrategy::WhisperPlusNLLB);
        }

        Err(anyhow!("No suitable translation strategy available"))
    }

    /// Translate using Whisper direct translation
    async fn translate_whisper_direct(&self, request: TranslationRequest) -> Result<TranslationResponse> {
        let whisper = self.whisper_standard.as_ref()
            .ok_or_else(|| anyhow!("Whisper model not initialized"))?;

        let mel_features = match request.input_type {
            InputType::MelSpectrogram(ref features) => features.clone(),
            InputType::Audio(ref audio) => {
                // Convert audio to mel spectrogram
                self.audio_to_mel(audio)?
            },
            InputType::Text(_) => return Err(anyhow!("Whisper requires audio input")),
        };

        let result = whisper.process(&mel_features)?;

        Ok(TranslationResponse {
            original_text: result.text.clone(),
            translated_text: result.translated_text.unwrap_or(result.text),
            source_language: result.language,
            target_language: "en".to_string(),
            confidence: result.confidence,
            strategy_used: TranslationStrategy::WhisperDirect,
            processing_time_ms: result.processing_time_ms,
            model_info: "Whisper".to_string(),
        })
    }

    /// Translate using Whisper ASR + NLLB translation
    async fn translate_whisper_nllb(&self, request: TranslationRequest) -> Result<TranslationResponse> {
        // First, transcribe with Whisper
        let whisper = self.whisper_standard.as_ref()
            .ok_or_else(|| anyhow!("Whisper model not initialized"))?;

        let mel_features = match request.input_type {
            InputType::MelSpectrogram(ref features) => features.clone(),
            InputType::Audio(ref audio) => self.audio_to_mel(audio)?,
            InputType::Text(ref text) => {
                // Skip Whisper, go directly to NLLB
                return self.translate_nllb_only(TranslationRequest {
                    input_type: InputType::Text(text.clone()),
                    ..request
                }).await;
            }
        };

        // Transcribe
        let transcription = whisper.process(&mel_features)?;

        // Then translate with NLLB
        let nllb = self.get_best_nllb(&request.quality)?;
        let translation = nllb.read().await.translate(
            &transcription.text,
            &transcription.language,
            &request.target_language
        ).await?;

        Ok(TranslationResponse {
            original_text: transcription.text,
            translated_text: translation.translated_text,
            source_language: transcription.language,
            target_language: request.target_language,
            confidence: (transcription.confidence + translation.confidence) / 2.0,
            strategy_used: TranslationStrategy::WhisperPlusNLLB,
            processing_time_ms: transcription.processing_time_ms + translation.processing_time_ms,
            model_info: format!("Whisper + NLLB"),
        })
    }

    /// Translate using CTranslate2 Whisper
    async fn translate_whisper_ct2(&self, request: TranslationRequest) -> Result<TranslationResponse> {
        let whisper_ct2 = self.whisper_ctranslate2.as_ref()
            .ok_or_else(|| anyhow!("Whisper CTranslate2 not initialized"))?;

        let audio = match request.input_type {
            InputType::Audio(ref audio) => audio,
            _ => return Err(anyhow!("CTranslate2 requires audio input")),
        };

        let result = whisper_ct2.process(
            audio,
            WhisperTask::Translate,
            request.source_language.as_deref()
        )?;

        Ok(TranslationResponse {
            original_text: result.text.clone(),
            translated_text: result.translated_text.unwrap_or(result.text),
            source_language: result.language,
            target_language: "en".to_string(),
            confidence: result.confidence,
            strategy_used: TranslationStrategy::WhisperCTranslate2,
            processing_time_ms: result.processing_time_ms,
            model_info: "Whisper-CTranslate2".to_string(),
        })
    }

    /// Translate text using NLLB only
    async fn translate_nllb_only(&self, request: TranslationRequest) -> Result<TranslationResponse> {
        let text = match request.input_type {
            InputType::Text(ref text) => text,
            _ => return Err(anyhow!("NLLB requires text input")),
        };

        let source_lang = request.source_language
            .as_ref()
            .ok_or_else(|| anyhow!("Source language required for NLLB"))?;

        let nllb = self.get_best_nllb(&request.quality)?;
        let result = nllb.read().await.translate(
            text,
            source_lang,
            &request.target_language
        ).await?;

        Ok(TranslationResponse {
            original_text: text.clone(),
            translated_text: result.translated_text,
            source_language: source_lang.clone(),
            target_language: request.target_language,
            confidence: result.confidence,
            strategy_used: TranslationStrategy::NLLBOnly,
            processing_time_ms: result.processing_time_ms,
            model_info: "NLLB-200".to_string(),
        })
    }

    /// Convert audio to mel spectrogram
    fn audio_to_mel(&self, _audio: &[f32]) -> Result<Vec<f32>> {
        // Stub implementation - would use actual mel spectrogram computation
        Ok(vec![0.0; 80 * 100]) // 80 mel bins, 100 time frames
    }

    /// Check if any NLLB model is available
    fn has_nllb(&self) -> bool {
        self.nllb_600m.is_some() || self.nllb_1_3b.is_some() || self.nllb_3_3b.is_some()
    }

    /// Get best NLLB model based on quality requirements
    fn get_best_nllb(&self, quality: &QualityLevel) -> Result<Arc<RwLock<NLLB200Translator>>> {
        match quality {
            QualityLevel::Draft => {
                self.nllb_600m.clone()
                    .or_else(|| self.nllb_1_3b.clone())
                    .or_else(|| self.nllb_3_3b.clone())
            },
            QualityLevel::Standard => {
                self.nllb_1_3b.clone()
                    .or_else(|| self.nllb_3_3b.clone())
                    .or_else(|| self.nllb_600m.clone())
            },
            QualityLevel::Premium => {
                self.nllb_3_3b.clone()
                    .or_else(|| self.nllb_1_3b.clone())
                    .or_else(|| self.nllb_600m.clone())
            },
        }.ok_or_else(|| anyhow!("No NLLB model available"))
    }

    /// Update performance statistics
    async fn update_stats(&self, strategy: &TranslationStrategy, latency_ms: f32) {
        let mut stats = self.stats.write().await;
        stats.record_translation(strategy.clone(), latency_ms);
    }

    /// Get performance statistics
    pub async fn get_stats(&self) -> PerformanceStats {
        self.stats.read().await.clone()
    }
}

/// Model capabilities registry
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ModelCapabilities {
    whisper_languages: Vec<String>,
    nllb_languages: Vec<String>,
    seamless_languages: Vec<String>,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            whisper_languages: vec!["en", "es", "fr", "de", "it", "ja", "ko", "zh"]
                .iter().map(|s| s.to_string()).collect(),
            nllb_languages: vec!["en", "es", "fr", "de", "it", "ja", "ko", "zh", "ar", "hi", "ru"]
                .iter().map(|s| s.to_string()).collect(),
            seamless_languages: vec!["en", "es", "fr", "de", "it"]
                .iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    /// Total number of translations performed
    pub total_translations: u64,
    /// Count of translations per strategy
    pub strategy_counts: HashMap<TranslationStrategy, u64>,
    /// Average latency per strategy in milliseconds
    pub average_latencies: HashMap<TranslationStrategy, f32>,
    /// Minimum observed latency across all translations
    pub min_latency_ms: f32,
    /// Maximum observed latency across all translations
    pub max_latency_ms: f32,
}

impl PerformanceStats {
    fn record_translation(&mut self, strategy: TranslationStrategy, latency_ms: f32) {
        self.total_translations += 1;

        *self.strategy_counts.entry(strategy.clone()).or_insert(0) += 1;

        let count = self.strategy_counts[&strategy] as f32;
        let current_avg = self.average_latencies.get(&strategy).copied().unwrap_or(0.0);
        let new_avg = (current_avg * (count - 1.0) + latency_ms) / count;
        self.average_latencies.insert(strategy, new_avg);

        if self.min_latency_ms == 0.0 || latency_ms < self.min_latency_ms {
            self.min_latency_ms = latency_ms;
        }
        if latency_ms > self.max_latency_ms {
            self.max_latency_ms = latency_ms;
        }
    }
}

