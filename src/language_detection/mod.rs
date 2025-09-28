//! Multi-tier Language Detection System
//!
//! This module provides profile-aware language detection implementations:
//! - FastText-based detection for all profiles with model size optimization
//! - Text preprocessing and normalization
//! - Audio-text fusion for high-accuracy detection
//! - Real-time language switching support

pub mod fasttext_detector;
pub mod language_fusion;
pub mod pipeline;

use crate::profile::Profile;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, debug};

/// Language detection result with confidence metrics
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    /// ISO 639-1 language code (e.g., "en", "es", "fr")
    pub language: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Alternative candidates with scores
    pub alternatives: Vec<LanguageCandidate>,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Detection method used
    pub detection_method: DetectionMethod,
}

/// Language candidate with confidence
#[derive(Debug, Clone)]
pub struct LanguageCandidate {
    /// ISO 639-1 language code
    pub language: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Detection methods available
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DetectionMethod {
    /// Text-only FastText detection
    TextFastText,
    /// Audio-based detection (from ASR language hints)
    AudioBased,
    /// Fusion of text and audio signals
    MultiModal,
    /// Cached result from previous detection
    Cached,
}

/// Language detector configuration
#[derive(Debug, Clone)]
pub struct LanguageDetectorConfig {
    /// Profile for model selection
    pub profile: Profile,
    /// Minimum confidence threshold for detection
    pub confidence_threshold: f32,
    /// Maximum number of alternative candidates
    pub max_alternatives: usize,
    /// Enable caching of recent detections
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Minimum text length for reliable detection
    pub min_text_length: usize,
    /// Enable multimodal fusion (text + audio)
    pub enable_multimodal: bool,
}

impl Default for LanguageDetectorConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Medium,
            confidence_threshold: 0.7,
            max_alternatives: 3,
            enable_caching: true,
            cache_ttl_seconds: 300, // 5 minutes
            min_text_length: 10,
            enable_multimodal: false, // Enable for High Profile
        }
    }
}

/// Trait for language detection implementations
pub trait LanguageDetector: Send + Sync {
    /// Initialize the detector with configuration
    fn initialize(&mut self, config: LanguageDetectorConfig) -> Result<()>;

    /// Detect language from text
    fn detect_text(&mut self, text: &str) -> Result<LanguageDetection>;

    /// Detect language with audio context (multimodal)
    fn detect_multimodal(&mut self, text: &str, audio_language_hint: Option<&str>) -> Result<LanguageDetection>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<String>;

    /// Get current profile
    fn profile(&self) -> Profile;

    /// Update configuration at runtime
    fn update_config(&mut self, config: LanguageDetectorConfig) -> Result<()>;

    /// Reset detector state
    fn reset(&mut self) -> Result<()>;

    /// Get processing statistics
    fn get_stats(&self) -> DetectionStats;
}

/// Language detection statistics
#[derive(Debug, Clone, Default)]
pub struct DetectionStats {
    pub total_detections: u64,
    pub average_processing_time_ms: f32,
    pub cache_hit_rate: f32,
    pub average_confidence: f32,
    pub language_distribution: HashMap<String, u64>,
    pub method_distribution: HashMap<DetectionMethod, u64>,
}

/// Language detector capabilities
#[derive(Debug, Clone)]
pub struct DetectorCapabilities {
    /// Supported profiles
    pub supported_profiles: Vec<Profile>,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Maximum text length for processing
    pub max_text_length: usize,
    /// Processing latency (ms)
    pub typical_latency_ms: f32,
    /// Memory requirements (MB)
    pub memory_requirement_mb: f64,
    /// Model size (MB)
    pub model_size_mb: f64,
    /// Supports multimodal detection
    pub supports_multimodal: bool,
}

/// Language detection manager that selects appropriate detector based on profile
pub struct LanguageDetectionManager {
    profile: Profile,
    detector: Box<dyn LanguageDetector>,
    config: LanguageDetectorConfig,
    stats: DetectionStats,
    detection_cache: HashMap<String, CachedDetection>,
}

/// Cached detection result with timestamp
#[derive(Debug, Clone)]
struct CachedDetection {
    result: LanguageDetection,
    timestamp: Instant,
}

impl LanguageDetectionManager {
    /// Create a new language detection manager for the given profile
    pub fn new(profile: Profile, config: LanguageDetectorConfig) -> Result<Self> {
        info!("🌐 Initializing Language Detection Manager for profile {:?}", profile);

        let detector: Box<dyn LanguageDetector> = match profile {
            Profile::Low => {
                info!("📊 Creating FastText Compact Detector (Low Profile)");
                Box::new(fasttext_detector::FastTextCompactDetector::new()?)
            }
            Profile::Medium => {
                info!("📊 Creating FastText Standard Detector (Medium Profile)");
                Box::new(fasttext_detector::FastTextStandardDetector::new()?)
            }
            Profile::High => {
                info!("📊 Creating FastText Multimodal Detector (High Profile)");
                Box::new(language_fusion::MultimodalDetector::new()?)
            }
        };

        info!("✅ Language Detection Manager initialized for profile {:?}", profile);

        Ok(Self {
            profile,
            detector,
            config,
            stats: DetectionStats::default(),
            detection_cache: HashMap::new(),
        })
    }

    /// Initialize with the given configuration
    pub fn initialize(&mut self, config: LanguageDetectorConfig) -> Result<()> {
        info!("🔧 Initializing language detector: {} profile, threshold {:.2}",
              self.profile_name(), config.confidence_threshold);

        self.config = config.clone();
        self.detector.initialize(config)?;

        Ok(())
    }

    /// Detect language from text with caching
    pub fn detect(&mut self, text: &str) -> Result<LanguageDetection> {
        let start_time = Instant::now();

        // Check cache first if enabled
        if self.config.enable_caching {
            if let Some(cached) = self.check_cache(text) {
                self.update_cache_stats();
                return Ok(cached);
            }
        }

        // Preprocess text
        let preprocessed_text = self.preprocess_text(text);

        // Check minimum length requirement
        if preprocessed_text.len() < self.config.min_text_length {
            return Ok(LanguageDetection {
                language: "unknown".to_string(),
                confidence: 0.0,
                alternatives: Vec::new(),
                processing_time_ms: start_time.elapsed().as_secs_f32() * 1000.0,
                detection_method: DetectionMethod::TextFastText,
            });
        }

        // Perform detection
        let mut result = self.detector.detect_text(&preprocessed_text)?;
        result.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Cache result if enabled
        if self.config.enable_caching {
            self.cache_result(text, &result);
        }

        // Update statistics
        self.update_stats(&result);

        debug!("Language detected in {:.2}ms: {} ({:.3} confidence)",
               result.processing_time_ms, result.language, result.confidence);

        Ok(result)
    }

    /// Detect language with multimodal support (text + audio hint)
    pub fn detect_multimodal(&mut self, text: &str, audio_language_hint: Option<&str>) -> Result<LanguageDetection> {
        let start_time = Instant::now();

        // Use multimodal detection if enabled and supported
        if self.config.enable_multimodal && self.profile == Profile::High {
            let preprocessed_text = self.preprocess_text(text);

            let mut result = self.detector.detect_multimodal(&preprocessed_text, audio_language_hint)?;
            result.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

            self.update_stats(&result);

            debug!("Multimodal language detected in {:.2}ms: {} ({:.3} confidence)",
                   result.processing_time_ms, result.language, result.confidence);

            Ok(result)
        } else {
            // Fallback to text-only detection
            self.detect(text)
        }
    }

    /// Get supported languages
    pub fn supported_languages(&self) -> Vec<String> {
        self.detector.supported_languages()
    }

    /// Get current profile
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Get detector capabilities
    pub fn get_capabilities(&self) -> DetectorCapabilities {
        match self.profile {
            Profile::Low => DetectorCapabilities {
                supported_profiles: vec![Profile::Low],
                supported_languages: self.get_core_languages(),
                max_text_length: 1000,
                typical_latency_ms: 3.0,
                memory_requirement_mb: 50.0,
                model_size_mb: 1.0, // Compact FastText model
                supports_multimodal: false,
            },
            Profile::Medium => DetectorCapabilities {
                supported_profiles: vec![Profile::Medium],
                supported_languages: self.get_extended_languages(),
                max_text_length: 5000,
                typical_latency_ms: 4.0,
                memory_requirement_mb: 150.0,
                model_size_mb: 126.0, // Standard FastText model
                supports_multimodal: false,
            },
            Profile::High => DetectorCapabilities {
                supported_profiles: vec![Profile::High],
                supported_languages: self.get_comprehensive_languages(),
                max_text_length: 10000,
                typical_latency_ms: 2.0,
                memory_requirement_mb: 200.0,
                model_size_mb: 126.0, // FastText model + fusion components
                supports_multimodal: true,
            },
        }
    }

    /// Get processing statistics
    pub fn stats(&self) -> &DetectionStats {
        &self.stats
    }

    /// Reset detector state and statistics
    pub fn reset(&mut self) -> Result<()> {
        self.detector.reset()?;
        self.stats = DetectionStats::default();
        self.detection_cache.clear();
        info!("🔄 Language Detection Manager reset");
        Ok(())
    }

    /// Update configuration at runtime
    pub fn update_config(&mut self, config: LanguageDetectorConfig) -> Result<()> {
        self.config = config.clone();
        self.detector.update_config(config)
    }

    /// Preprocess text for better detection
    fn preprocess_text(&self, text: &str) -> String {
        // Remove excessive whitespace and normalize
        let cleaned = text
            .trim()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");

        // Truncate if too long
        if cleaned.len() > self.get_capabilities().max_text_length {
            cleaned.chars().take(self.get_capabilities().max_text_length).collect()
        } else {
            cleaned
        }
    }

    /// Check cache for existing detection
    fn check_cache(&self, text: &str) -> Option<LanguageDetection> {
        if let Some(cached) = self.detection_cache.get(text) {
            // Check if cache entry is still valid
            if cached.timestamp.elapsed().as_secs() < self.config.cache_ttl_seconds {
                let mut result = cached.result.clone();
                result.detection_method = DetectionMethod::Cached;
                return Some(result);
            }
        }
        None
    }

    /// Cache detection result
    fn cache_result(&mut self, text: &str, result: &LanguageDetection) {
        // Only cache confident results
        if result.confidence >= self.config.confidence_threshold {
            let cached = CachedDetection {
                result: result.clone(),
                timestamp: Instant::now(),
            };
            self.detection_cache.insert(text.to_string(), cached);

            // Limit cache size
            if self.detection_cache.len() > 1000 {
                // Remove oldest entries (simplified cleanup)
                let keys_to_remove: Vec<String> = self.detection_cache
                    .iter()
                    .filter(|(_, cached)| cached.timestamp.elapsed().as_secs() > self.config.cache_ttl_seconds)
                    .map(|(key, _)| key.clone())
                    .collect();

                for key in keys_to_remove {
                    self.detection_cache.remove(&key);
                }
            }
        }
    }

    /// Update processing statistics
    fn update_stats(&mut self, result: &LanguageDetection) {
        self.stats.total_detections += 1;

        // Update average processing time
        let total_operations = self.stats.total_detections as f32;
        self.stats.average_processing_time_ms =
            (self.stats.average_processing_time_ms * (total_operations - 1.0) + result.processing_time_ms) / total_operations;

        // Update average confidence
        self.stats.average_confidence =
            (self.stats.average_confidence * (total_operations - 1.0) + result.confidence) / total_operations;

        // Update language distribution
        *self.stats.language_distribution.entry(result.language.clone()).or_insert(0) += 1;

        // Update method distribution
        *self.stats.method_distribution.entry(result.detection_method.clone()).or_insert(0) += 1;
    }

    /// Update cache hit statistics
    fn update_cache_stats(&mut self) {
        let total_detections = self.stats.total_detections as f32;
        let cache_hits = self.stats.method_distribution.get(&DetectionMethod::Cached).copied().unwrap_or(0) as f32;

        self.stats.cache_hit_rate = if total_detections > 0.0 {
            cache_hits / total_detections
        } else {
            0.0
        };
    }

    /// Get profile name as string
    fn profile_name(&self) -> &'static str {
        match self.profile {
            Profile::Low => "Low",
            Profile::Medium => "Medium",
            Profile::High => "High",
        }
    }

    /// Get core languages for Low Profile
    fn get_core_languages(&self) -> Vec<String> {
        vec![
            "en".to_string(), "es".to_string(), "fr".to_string(), "de".to_string(),
            "it".to_string(), "pt".to_string(), "zh".to_string(), "ja".to_string(),
            "ko".to_string(), "ru".to_string(),
        ]
    }

    /// Get extended languages for Medium Profile
    fn get_extended_languages(&self) -> Vec<String> {
        let mut languages = self.get_core_languages();
        languages.extend(vec![
            "ar".to_string(), "hi".to_string(), "th".to_string(), "vi".to_string(),
            "tr".to_string(), "pl".to_string(), "nl".to_string(), "sv".to_string(),
            "da".to_string(), "no".to_string(), "fi".to_string(), "he".to_string(),
        ]);
        languages
    }

    /// Get comprehensive languages for High Profile
    fn get_comprehensive_languages(&self) -> Vec<String> {
        let mut languages = self.get_extended_languages();
        languages.extend(vec![
            "cs".to_string(), "sk".to_string(), "hu".to_string(), "ro".to_string(),
            "bg".to_string(), "hr".to_string(), "sl".to_string(), "et".to_string(),
            "lv".to_string(), "lt".to_string(), "uk".to_string(), "be".to_string(),
            "ca".to_string(), "eu".to_string(), "gl".to_string(), "mt".to_string(),
        ]);
        languages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detector_config_default() {
        let config = LanguageDetectorConfig::default();
        assert_eq!(config.profile, Profile::Medium);
        assert_eq!(config.confidence_threshold, 0.7);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_language_detection_result() {
        let detection = LanguageDetection {
            language: "en".to_string(),
            confidence: 0.95,
            alternatives: vec![
                LanguageCandidate {
                    language: "es".to_string(),
                    confidence: 0.03,
                }
            ],
            processing_time_ms: 2.5,
            detection_method: DetectionMethod::TextFastText,
        };

        assert_eq!(detection.language, "en");
        assert!(detection.confidence > 0.9);
        assert_eq!(detection.alternatives.len(), 1);
    }

    #[test]
    fn test_detection_stats() {
        let mut stats = DetectionStats::default();
        assert_eq!(stats.total_detections, 0);
        assert_eq!(stats.average_processing_time_ms, 0.0);

        stats.total_detections = 10;
        stats.average_processing_time_ms = 3.2;

        assert_eq!(stats.total_detections, 10);
        assert_eq!(stats.average_processing_time_ms, 3.2);
    }
}