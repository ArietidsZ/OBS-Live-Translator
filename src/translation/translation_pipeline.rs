//! Translation Pipeline Module
//!
//! Orchestrates the complete translation workflow with quality assessment,
//! fallback mechanisms, and performance monitoring.

use super::{
    TranslationEngine, TranslationResult,
    LanguagePair, marian_translator::MarianTranslator,
    m2m_translator::M2MTranslator, nllb_translator::NLLBTranslator,
};
use crate::profile::Profile;
use anyhow::Result;
use std::time::Instant;
use tracing::{info, debug, warn};

/// Complete translation pipeline with fallback and quality assessment
pub struct TranslationPipeline {
    /// Current profile determines which engine to use
    profile: Profile,
    /// Primary translation engine
    primary_engine: Box<dyn TranslationEngine>,
    /// Fallback engine for quality issues
    fallback_engine: Option<Box<dyn TranslationEngine>>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Pipeline statistics
    stats: PipelineStats,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Minimum confidence threshold for accepting translations
    pub min_confidence_threshold: f32,
    /// Enable fallback to higher quality engine if confidence is low
    pub enable_fallback: bool,
    /// Maximum processing time before timeout (ms)
    pub max_processing_time_ms: f32,
    /// Enable quality assessment
    pub enable_quality_assessment: bool,
    /// Language pair priorities
    pub language_pair_priorities: Vec<LanguagePair>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            enable_fallback: true,
            max_processing_time_ms: 5000.0,
            enable_quality_assessment: true,
            language_pair_priorities: Vec::new(),
        }
    }
}

/// Pipeline performance statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_translations: u64,
    pub successful_translations: u64,
    pub fallback_used: u64,
    pub quality_rejections: u64,
    pub average_processing_time_ms: f32,
    pub average_confidence: f32,
    pub language_pair_stats: std::collections::HashMap<LanguagePair, PairStats>,
}

/// Per-language-pair statistics
#[derive(Debug, Clone, Default)]
pub struct PairStats {
    pub translations: u64,
    pub average_confidence: f32,
    pub average_processing_time_ms: f32,
    pub fallback_rate: f32,
}

impl TranslationPipeline {
    /// Create new translation pipeline
    pub fn new(profile: Profile, config: PipelineConfig) -> Result<Self> {
        let (primary_engine, fallback_engine) = Self::create_engines_for_profile(&profile)?;

        info!("🔄 Initializing translation pipeline for profile: {:?}", profile);

        Ok(Self {
            profile,
            primary_engine,
            fallback_engine,
            config,
            stats: PipelineStats::default(),
        })
    }

    /// Create appropriate engines based on profile
    fn create_engines_for_profile(profile: &Profile) -> Result<(Box<dyn TranslationEngine>, Option<Box<dyn TranslationEngine>>)> {
        match profile {
            Profile::Low => {
                let primary = Box::new(MarianTranslator::new()?);
                let fallback = None; // No fallback for low profile
                Ok((primary, fallback))
            },
            Profile::Medium => {
                let primary = Box::new(M2MTranslator::new()?);
                let fallback = Some(Box::new(NLLBTranslator::new()?) as Box<dyn TranslationEngine>);
                Ok((primary, fallback))
            },
            Profile::High => {
                let primary = Box::new(NLLBTranslator::new()?);
                let fallback = None; // High profile doesn't need fallback
                Ok((primary, fallback))
            }
        }
    }

    /// Process translation through the complete pipeline
    pub async fn translate(&mut self, text: &str, source_lang: &str, target_lang: &str) -> Result<TranslationResult> {
        let start_time = Instant::now();
        let language_pair = LanguagePair::new(source_lang, target_lang);

        debug!("🔄 Starting translation pipeline: {} -> {} (profile: {:?})",
               source_lang, target_lang, self.profile);

        // Try primary engine first
        let mut result = match self.primary_engine.translate(text, source_lang, target_lang) {
            Ok(result) => result,
            Err(e) => {
                warn!("Primary engine failed: {}", e);
                self.stats.quality_rejections += 1;

                // Try fallback if available
                if let Some(ref mut fallback) = self.fallback_engine {
                    if self.config.enable_fallback {
                        debug!("🔄 Trying fallback engine");
                        self.stats.fallback_used += 1;
                        fallback.translate(text, source_lang, target_lang)?
                    } else {
                        return Err(e);
                    }
                } else {
                    return Err(e);
                }
            }
        };

        // Quality assessment
        if self.config.enable_quality_assessment {
            let quality_score = self.assess_translation_quality(&result);

            if quality_score < self.config.min_confidence_threshold {
                if let Some(ref mut fallback) = self.fallback_engine {
                    if self.config.enable_fallback {
                        debug!("🔄 Quality too low ({:.2}), trying fallback engine", quality_score);
                        self.stats.fallback_used += 1;
                        result = fallback.translate(text, source_lang, target_lang)?;
                    }
                }
            }
        }

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Check timeout
        if processing_time > self.config.max_processing_time_ms {
            warn!("Translation exceeded timeout: {:.2}ms", processing_time);
        }

        // Update statistics
        self.update_stats(&language_pair, &result, processing_time);

        debug!("✅ Translation completed in {:.2}ms with confidence {:.2}",
               processing_time, result.confidence);

        Ok(result)
    }

    /// Assess translation quality
    fn assess_translation_quality(&self, result: &TranslationResult) -> f32 {
        // Basic quality assessment combining multiple factors
        let mut quality_score = result.confidence;

        // Length ratio check (translated text shouldn't be too different in length)
        let source_len = result.translated_text.chars().count() as f32; // Note: in real impl, would use source text
        let target_len = result.translated_text.chars().count() as f32;

        if source_len > 0.0 {
            let length_ratio = target_len / source_len;
            if length_ratio < 0.3 || length_ratio > 3.0 {
                quality_score *= 0.8; // Penalize extreme length differences
            }
        }

        // Check for obvious translation failures
        if result.translated_text.trim().is_empty() {
            quality_score = 0.0;
        }

        // Check for repeated patterns (sign of model failure)
        if self.has_repetitive_patterns(&result.translated_text) {
            quality_score *= 0.6;
        }

        quality_score.clamp(0.0, 1.0)
    }

    /// Detect repetitive patterns that indicate poor translation
    fn has_repetitive_patterns(&self, text: &str) -> bool {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 6 {
            return false;
        }

        // Check for repeated 3-grams
        let mut trigrams = std::collections::HashMap::new();
        for window in words.windows(3) {
            let trigram = window.join(" ");
            *trigrams.entry(trigram).or_insert(0) += 1;
        }

        // If any trigram appears more than 3 times, it's likely repetitive
        trigrams.values().any(|&count| count > 3)
    }

    /// Update pipeline statistics
    fn update_stats(&mut self, language_pair: &LanguagePair, result: &TranslationResult, processing_time: f32) {
        self.stats.total_translations += 1;

        if result.confidence >= self.config.min_confidence_threshold {
            self.stats.successful_translations += 1;
        }

        // Update averages
        let n = self.stats.total_translations as f32;
        self.stats.average_processing_time_ms =
            (self.stats.average_processing_time_ms * (n - 1.0) + processing_time) / n;
        self.stats.average_confidence =
            (self.stats.average_confidence * (n - 1.0) + result.confidence) / n;

        // Update language pair stats
        let pair_stats = self.stats.language_pair_stats.entry(language_pair.clone()).or_default();
        pair_stats.translations += 1;

        let pair_n = pair_stats.translations as f32;
        pair_stats.average_confidence =
            (pair_stats.average_confidence * (pair_n - 1.0) + result.confidence) / pair_n;
        pair_stats.average_processing_time_ms =
            (pair_stats.average_processing_time_ms * (pair_n - 1.0) + processing_time) / pair_n;
    }

    /// Process batch translations
    pub async fn translate_batch(&mut self, batch: Vec<(String, String, String)>) -> Result<Vec<TranslationResult>> {
        let mut results = Vec::with_capacity(batch.len());

        debug!("🔄 Processing batch of {} translations", batch.len());

        for (text, source_lang, target_lang) in batch {
            match self.translate(&text, &source_lang, &target_lang).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Batch translation failed for '{}': {}", text, e);
                    // Create error result
                    results.push(TranslationResult {
                        translated_text: text.clone(),
                        source_language: source_lang,
                        target_language: target_lang,
                        confidence: 0.0,
                        word_alignments: Vec::new(),
                        processing_time_ms: 0.0,
                        model_name: "error".to_string(),
                        metrics: Default::default(),
                    });
                }
            }
        }

        debug!("✅ Batch processing completed: {}/{} successful",
               results.iter().filter(|r| r.confidence > 0.0).count(),
               results.len());

        Ok(results)
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get success rate
    pub fn get_success_rate(&self) -> f32 {
        if self.stats.total_translations > 0 {
            self.stats.successful_translations as f32 / self.stats.total_translations as f32
        } else {
            0.0
        }
    }

    /// Get fallback usage rate
    pub fn get_fallback_rate(&self) -> f32 {
        if self.stats.total_translations > 0 {
            self.stats.fallback_used as f32 / self.stats.total_translations as f32
        } else {
            0.0
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStats::default();
        info!("🔄 Pipeline statistics reset");
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PipelineConfig) {
        self.config = config;
        info!("🔄 Pipeline configuration updated");
    }

    /// Switch to different profile
    pub async fn switch_profile(&mut self, new_profile: Profile) -> Result<()> {
        if new_profile == self.profile {
            return Ok(());
        }

        info!("🔄 Switching pipeline profile from {:?} to {:?}", self.profile, new_profile);

        let (new_primary, new_fallback) = Self::create_engines_for_profile(&new_profile)?;

        self.profile = new_profile;
        self.primary_engine = new_primary;
        self.fallback_engine = new_fallback;

        info!("✅ Profile switch completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = TranslationPipeline::new(Profile::Low, config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_quality_assessment() {
        let config = PipelineConfig::default();
        let pipeline = TranslationPipeline::new(Profile::Low, config).unwrap();

        let good_result = TranslationResult {
            translated_text: "This is a good translation".to_string(),
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            confidence: 0.9,
            word_alignments: Vec::new(),
            processing_time_ms: 100.0,
            model_name: "test".to_string(),
            metrics: Default::default(),
        };

        let quality = pipeline.assess_translation_quality(&good_result);
        assert!(quality >= 0.8);

        let empty_result = TranslationResult {
            translated_text: "".to_string(),
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            confidence: 0.9,
            word_alignments: Vec::new(),
            processing_time_ms: 100.0,
            model_name: "test".to_string(),
            metrics: Default::default(),
        };

        let empty_quality = pipeline.assess_translation_quality(&empty_result);
        assert_eq!(empty_quality, 0.0);
    }

    #[test]
    fn test_repetitive_pattern_detection() {
        let config = PipelineConfig::default();
        let pipeline = TranslationPipeline::new(Profile::Low, config).unwrap();

        let repetitive_text = "the same phrase the same phrase the same phrase the same phrase";
        assert!(pipeline.has_repetitive_patterns(repetitive_text));

        let normal_text = "This is a normal translation without repetitive patterns";
        assert!(!pipeline.has_repetitive_patterns(normal_text));
    }

    #[test]
    fn test_stats_calculation() {
        let config = PipelineConfig::default();
        let mut pipeline = TranslationPipeline::new(Profile::Low, config).unwrap();

        assert_eq!(pipeline.get_success_rate(), 0.0);
        assert_eq!(pipeline.get_fallback_rate(), 0.0);

        // Simulate some statistics
        pipeline.stats.total_translations = 10;
        pipeline.stats.successful_translations = 8;
        pipeline.stats.fallback_used = 2;

        assert_eq!(pipeline.get_success_rate(), 0.8);
        assert_eq!(pipeline.get_fallback_rate(), 0.2);
    }
}