//! Multi-tier Neural Machine Translation System
//!
//! This module provides profile-aware translation implementations:
//! - Low Profile: MarianNMT INT8 quantized for CPU efficiency
//! - Medium Profile: M2M-100 FP16 GPU for balanced performance
//! - High Profile: NLLB-200/Tower-7B for maximum quality
//! - Translation pipeline optimization and caching

pub mod marian_translator;
pub mod m2m_translator;
pub mod nllb_translator;
pub mod translation_pipeline;
pub mod cache;

use crate::profile::Profile;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, debug};

/// Translation result with confidence metrics
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Translated text
    pub translated_text: String,
    /// Source language (ISO 639-1 code)
    pub source_language: String,
    /// Target language (ISO 639-1 code)
    pub target_language: String,
    /// Translation confidence score (0.0-1.0)
    pub confidence: f32,
    /// Word-level alignments (optional)
    pub word_alignments: Vec<WordAlignment>,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Model used for translation
    pub model_name: String,
    /// Quality metrics
    pub metrics: TranslationMetrics,
}

/// Word-level alignment between source and target
#[derive(Debug, Clone)]
pub struct WordAlignment {
    /// Source word
    pub source_word: String,
    /// Target word
    pub target_word: String,
    /// Alignment confidence (0.0-1.0)
    pub alignment_confidence: f32,
    /// Source word position
    pub source_position: usize,
    /// Target word position
    pub target_position: usize,
}

/// Translation configuration for different profiles
#[derive(Debug, Clone)]
pub struct TranslationConfig {
    /// Profile for model selection
    pub profile: Profile,
    /// Source language (None for auto-detection)
    pub source_language: Option<String>,
    /// Target language
    pub target_language: String,
    /// Model precision (INT8, FP16, FP32)
    pub precision: ModelPrecision,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Beam search width
    pub beam_size: usize,
    /// Length penalty for beam search
    pub length_penalty: f32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Enable caching for repeated phrases
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable batch processing
    pub enable_batching: bool,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Medium,
            source_language: None, // Auto-detect
            target_language: "en".to_string(),
            precision: ModelPrecision::FP16,
            max_sequence_length: 512,
            beam_size: 5,
            length_penalty: 1.0,
            temperature: 0.0, // Deterministic
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            enable_batching: false,
            batch_timeout_ms: 100,
        }
    }
}

/// Model precision options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelPrecision {
    INT8,
    FP16,
    FP32,
    INT4,  // For high-end quantization
    FP8,   // For modern hardware
}

/// Translation performance and quality metrics
#[derive(Debug, Clone, Default)]
pub struct TranslationMetrics {
    /// Processing latency (ms)
    pub latency_ms: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: f32,
    /// Throughput (tokens per second)
    pub tokens_per_second: f32,
    /// Model confidence score
    pub model_confidence: f32,
    /// Estimated BLEU score
    pub estimated_bleu: f32,
    /// Translation quality assessment
    pub quality_score: f32,
}

/// Language pair specification
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LanguagePair {
    /// Source language code
    pub source: String,
    /// Target language code
    pub target: String,
}

impl LanguagePair {
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source: source.to_string(),
            target: target.to_string(),
        }
    }

    pub fn reverse(&self) -> Self {
        Self {
            source: self.target.clone(),
            target: self.source.clone(),
        }
    }
}

/// Trait for translation implementations
pub trait TranslationEngine: Send + Sync {
    /// Initialize the translation engine with configuration
    fn initialize(&mut self, config: TranslationConfig) -> Result<()>;

    /// Translate text from source to target language
    fn translate(&mut self, text: &str, source_lang: &str, target_lang: &str) -> Result<TranslationResult>;

    /// Translate multiple texts in batch
    fn translate_batch(&mut self, texts: &[String], source_lang: &str, target_lang: &str) -> Result<Vec<TranslationResult>>;

    /// Check if language pair is supported
    fn supports_language_pair(&self, source_lang: &str, target_lang: &str) -> bool;

    /// Get supported language pairs
    fn supported_language_pairs(&self) -> Vec<LanguagePair>;

    /// Get current profile
    fn profile(&self) -> Profile;

    /// Get engine capabilities
    fn get_capabilities(&self) -> TranslationCapabilities;

    /// Reset engine state
    fn reset(&mut self) -> Result<()>;

    /// Get processing statistics
    fn get_stats(&self) -> TranslationStats;

    /// Update configuration at runtime
    fn update_config(&mut self, config: TranslationConfig) -> Result<()>;
}

/// Translation engine capabilities
#[derive(Debug, Clone)]
pub struct TranslationCapabilities {
    /// Supported profiles
    pub supported_profiles: Vec<Profile>,
    /// Supported language pairs
    pub supported_language_pairs: Vec<LanguagePair>,
    /// Supported precisions
    pub supported_precisions: Vec<ModelPrecision>,
    /// Maximum text length (characters)
    pub max_text_length: usize,
    /// Supports batch translation
    pub supports_batching: bool,
    /// Real-time processing capability
    pub supports_real_time: bool,
    /// GPU acceleration available
    pub has_gpu_acceleration: bool,
    /// Model size in MB
    pub model_size_mb: f64,
    /// Memory requirements in MB
    pub memory_requirement_mb: f64,
}

/// Translation processing statistics
#[derive(Debug, Clone, Default)]
pub struct TranslationStats {
    pub total_translations: u64,
    pub total_tokens_processed: u64,
    pub average_latency_ms: f32,
    pub peak_latency_ms: f32,
    pub average_confidence: f32,
    pub cache_hit_rate: f32,
    pub success_rate: f32,
    pub average_bleu_score: f32,
    pub language_pair_distribution: HashMap<LanguagePair, u64>,
}

/// Multi-tier translation manager that selects appropriate engine based on profile
pub struct TranslationManager {
    profile: Profile,
    engine: Box<dyn TranslationEngine>,
    config: TranslationConfig,
    stats: TranslationStats,
}

impl TranslationManager {
    /// Create a new translation manager for the given profile
    pub fn new(profile: Profile, config: TranslationConfig) -> Result<Self> {
        info!("🌍 Initializing Translation Manager for profile {:?}", profile);

        let engine: Box<dyn TranslationEngine> = match profile {
            Profile::Low => {
                info!("📊 Creating MarianNMT INT8 Engine (Low Profile)");
                Box::new(marian_translator::MarianTranslator::new()?)
            }
            Profile::Medium => {
                info!("📊 Creating M2M-100 FP16 Engine (Medium Profile)");
                Box::new(m2m_translator::M2MTranslator::new()?)
            }
            Profile::High => {
                info!("📊 Creating NLLB-200 Engine (High Profile)");
                Box::new(nllb_translator::NLLBTranslator::new()?)
            }
        };

        info!("✅ Translation Manager initialized for profile {:?}", profile);

        Ok(Self {
            profile,
            engine,
            config,
            stats: TranslationStats::default(),
        })
    }

    /// Initialize with the given configuration
    pub fn initialize(&mut self, config: TranslationConfig) -> Result<()> {
        info!("🔧 Initializing translation engine: {} profile, {} -> {}",
              self.profile_name(),
              config.source_language.as_deref().unwrap_or("auto"),
              config.target_language);

        self.config = config.clone();
        self.engine.initialize(config)?;

        Ok(())
    }

    /// Translate text with the profile-specific engine
    pub fn translate(&mut self, text: &str, source_lang: Option<&str>, target_lang: &str) -> Result<TranslationResult> {
        let start_time = Instant::now();

        // Use configured source language if none provided
        let source = source_lang.unwrap_or(
            self.config.source_language.as_deref().unwrap_or("auto")
        );

        let mut result = self.engine.translate(text, source, target_lang)?;
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update result timing
        result.processing_time_ms = processing_time;

        // Update statistics
        self.update_stats(&result);

        debug!("Translation completed in {:.2}ms: \"{}\" -> \"{}\" ({} profile)",
               processing_time,
               if text.len() > 50 { format!("{}...", &text[..50]) } else { text.to_string() },
               if result.translated_text.len() > 50 {
                   format!("{}...", &result.translated_text[..50])
               } else {
                   result.translated_text.clone()
               },
               self.profile_name());

        Ok(result)
    }

    /// Translate multiple texts in batch
    pub fn translate_batch(&mut self, texts: &[String], source_lang: Option<&str>, target_lang: &str) -> Result<Vec<TranslationResult>> {
        let start_time = Instant::now();

        let source = source_lang.unwrap_or(
            self.config.source_language.as_deref().unwrap_or("auto")
        );

        let mut results = self.engine.translate_batch(texts, source, target_lang)?;
        let total_processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update timing for each result
        let time_per_translation = total_processing_time / texts.len() as f32;
        for result in &mut results {
            result.processing_time_ms = time_per_translation;
            self.update_stats(result);
        }

        debug!("Batch translation completed: {} texts in {:.2}ms ({:.2}ms avg)",
               texts.len(), total_processing_time, time_per_translation);

        Ok(results)
    }

    /// Check if language pair is supported
    pub fn supports_language_pair(&self, source_lang: &str, target_lang: &str) -> bool {
        self.engine.supports_language_pair(source_lang, target_lang)
    }

    /// Get supported language pairs
    pub fn supported_language_pairs(&self) -> Vec<LanguagePair> {
        self.engine.supported_language_pairs()
    }

    /// Get the current profile
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Get the current configuration
    pub fn config(&self) -> &TranslationConfig {
        &self.config
    }

    /// Get processing statistics
    pub fn stats(&self) -> &TranslationStats {
        &self.stats
    }

    /// Check if translation is meeting performance targets
    pub fn is_meeting_performance_targets(&self) -> bool {
        let target_latency = match self.profile {
            Profile::Low => 180.0,    // 180ms for Low Profile
            Profile::Medium => 120.0, // 120ms for Medium Profile
            Profile::High => 80.0,    // 80ms for High Profile
        };

        let target_bleu = match self.profile {
            Profile::Low => 25.0,     // BLEU 25-30 for Low Profile
            Profile::Medium => 32.0,  // BLEU 32-37 for Medium Profile
            Profile::High => 38.0,    // BLEU 38-42 for High Profile
        };

        self.stats.average_latency_ms <= target_latency &&
        self.stats.average_bleu_score >= target_bleu &&
        self.stats.success_rate >= 0.95
    }

    /// Reset translation state and statistics
    pub fn reset(&mut self) -> Result<()> {
        self.engine.reset()?;
        self.stats = TranslationStats::default();
        info!("🔄 Translation Manager reset");
        Ok(())
    }

    /// Get quality assessment for current settings
    pub fn assess_quality(&self) -> QualityAssessment {
        let performance_score = match self.profile {
            Profile::Low => {
                // Focus on efficiency and basic quality
                let latency_score = if self.stats.average_latency_ms <= 180.0 { 0.8 } else { 0.6 };
                let bleu_score = (self.stats.average_bleu_score / 30.0).min(1.0);
                (latency_score + bleu_score) / 2.0
            }
            Profile::Medium => {
                // Balanced quality and performance
                let latency_score = if self.stats.average_latency_ms <= 120.0 { 0.8 } else { 0.6 };
                let bleu_score = (self.stats.average_bleu_score / 37.0).min(1.0);
                let confidence_score = self.stats.average_confidence;
                (latency_score + bleu_score + confidence_score) / 3.0
            }
            Profile::High => {
                // Maximum quality focus
                let latency_score = if self.stats.average_latency_ms <= 80.0 { 0.9 } else { 0.7 };
                let bleu_score = (self.stats.average_bleu_score / 42.0).min(1.0);
                let confidence_score = self.stats.average_confidence;
                let quality_score = self.stats.average_bleu_score / 50.0; // Normalize to 50 BLEU max
                (latency_score + bleu_score + confidence_score + quality_score) / 4.0
            }
        };

        QualityAssessment {
            overall_score: performance_score,
            meets_profile_requirements: performance_score >= 0.7,
            profile: self.profile,
            average_latency_ms: self.stats.average_latency_ms,
            average_confidence: self.stats.average_confidence,
            average_bleu_score: self.stats.average_bleu_score,
            success_rate: self.stats.success_rate,
        }
    }

    /// Update internal statistics
    fn update_stats(&mut self, result: &TranslationResult) {
        // Update totals
        self.stats.total_translations += 1;
        self.stats.total_tokens_processed += result.translated_text.split_whitespace().count() as u64;

        // Update averages
        let total_translations = self.stats.total_translations as f32;

        self.stats.average_latency_ms =
            (self.stats.average_latency_ms * (total_translations - 1.0) + result.processing_time_ms) / total_translations;

        // Update peak latency
        if result.processing_time_ms > self.stats.peak_latency_ms {
            self.stats.peak_latency_ms = result.processing_time_ms;
        }

        // Update confidence (moving average)
        self.stats.average_confidence =
            (self.stats.average_confidence * 0.9) + (result.confidence * 0.1);

        // Update BLEU score (moving average)
        self.stats.average_bleu_score =
            (self.stats.average_bleu_score * 0.9) + (result.metrics.estimated_bleu * 0.1);

        // Update success rate (assume success if confidence > 0.5)
        let current_success = if result.confidence > 0.5 { 1.0 } else { 0.0 };
        self.stats.success_rate =
            (self.stats.success_rate * 0.95) + (current_success * 0.05);

        // Update language pair distribution
        let pair = LanguagePair::new(&result.source_language, &result.target_language);
        *self.stats.language_pair_distribution.entry(pair).or_insert(0) += 1;
    }

    /// Get profile name as string
    fn profile_name(&self) -> &'static str {
        match self.profile {
            Profile::Low => "Low",
            Profile::Medium => "Medium",
            Profile::High => "High",
        }
    }

    /// Get engine capabilities
    pub fn get_capabilities(&self) -> TranslationCapabilities {
        self.engine.get_capabilities()
    }

    /// Update configuration at runtime
    pub fn update_config(&mut self, config: TranslationConfig) -> Result<()> {
        self.config = config.clone();
        self.engine.update_config(config)
    }
}

/// Quality assessment result for translation
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall performance score (0.0-1.0)
    pub overall_score: f32,
    /// Whether quality meets profile requirements
    pub meets_profile_requirements: bool,
    /// Profile being assessed
    pub profile: Profile,
    /// Average processing latency (ms)
    pub average_latency_ms: f32,
    /// Average translation confidence
    pub average_confidence: f32,
    /// Average BLEU score
    pub average_bleu_score: f32,
    /// Translation success rate
    pub success_rate: f32,
}

impl QualityAssessment {
    /// Get quality grade as string
    pub fn grade(&self) -> &'static str {
        if self.overall_score >= 0.9 {
            "Excellent"
        } else if self.overall_score >= 0.8 {
            "Very Good"
        } else if self.overall_score >= 0.7 {
            "Good"
        } else if self.overall_score >= 0.6 {
            "Fair"
        } else {
            "Poor"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translation_config_default() {
        let config = TranslationConfig::default();
        assert_eq!(config.profile, Profile::Medium);
        assert_eq!(config.precision, ModelPrecision::FP16);
        assert_eq!(config.target_language, "en");
    }

    #[test]
    fn test_language_pair() {
        let pair = LanguagePair::new("en", "es");
        assert_eq!(pair.source, "en");
        assert_eq!(pair.target, "es");

        let reversed = pair.reverse();
        assert_eq!(reversed.source, "es");
        assert_eq!(reversed.target, "en");
    }

    #[test]
    fn test_quality_assessment_grade() {
        let assessment = QualityAssessment {
            overall_score: 0.95,
            meets_profile_requirements: true,
            profile: Profile::High,
            average_latency_ms: 60.0,
            average_confidence: 0.92,
            average_bleu_score: 40.5,
            success_rate: 0.98,
        };
        assert_eq!(assessment.grade(), "Excellent");

        let poor_assessment = QualityAssessment {
            overall_score: 0.5,
            meets_profile_requirements: false,
            profile: Profile::Low,
            average_latency_ms: 250.0,
            average_confidence: 0.4,
            average_bleu_score: 20.0,
            success_rate: 0.8,
        };
        assert_eq!(poor_assessment.grade(), "Poor");
    }

    #[test]
    fn test_translation_metrics() {
        let metrics = TranslationMetrics {
            latency_ms: 85.0,
            estimated_bleu: 35.2,
            quality_score: 0.87,
            ..TranslationMetrics::default()
        };

        assert_eq!(metrics.latency_ms, 85.0);
        assert_eq!(metrics.estimated_bleu, 35.2);
        assert_eq!(metrics.quality_score, 0.87);
    }
}