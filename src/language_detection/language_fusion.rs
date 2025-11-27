//! Multimodal language detection with text-audio fusion
//!
//! This module provides advanced language detection for High Profile:
//! - Fusion of text-based and audio-based language signals
//! - Confidence weighting and decision fusion
//! - Real-time language switching detection
//! - Advanced preprocessing and normalization

use super::fasttext_detector::FastTextStandardDetector;
use super::{
    DetectionMethod, DetectionStats, LanguageCandidate, LanguageDetection, LanguageDetector,
    LanguageDetectorConfig,
};
use crate::profile::Profile;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

/// Multimodal detector that fuses text and audio language signals
pub struct MultimodalDetector {
    config: Option<LanguageDetectorConfig>,
    stats: DetectionStats,

    // Text-based detection component
    text_detector: FastTextStandardDetector,

    // Audio-based language hints processing
    audio_language_weights: HashMap<String, f32>,
    language_switching_detector: LanguageSwitchingDetector,

    // Fusion parameters
    text_weight: f32,
    audio_weight: f32,
    temporal_smoothing: f32,

    // Language history for temporal consistency
    language_history: Vec<HistoricalDetection>,
    max_history_size: usize,
}

/// Language switching detection for real-time adaptation
struct LanguageSwitchingDetector {
    switch_threshold: f32,
    min_confidence_for_switch: f32,
    switch_cooldown_ms: u64,
    last_switch_time: Option<Instant>,
    current_stable_language: Option<String>,
}

/// Historical detection for temporal smoothing
#[derive(Debug, Clone)]
struct HistoricalDetection {
    language: String,
    confidence: f32,
    timestamp: Instant,
    detection_method: DetectionMethod,
}

/// Fusion strategy for combining text and audio signals
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStrategy {
    /// Weighted average of text and audio confidences
    WeightedAverage,
    /// Maximum confidence across modalities
    MaxConfidence,
    /// Multiplicative fusion with normalization
    Multiplicative,
    /// Adaptive fusion based on confidence levels
    Adaptive,
}

impl MultimodalDetector {
    /// Create a new multimodal detector
    pub fn new() -> Result<Self> {
        info!("🚀 Initializing Multimodal Language Detector (High Profile)");

        let text_detector = FastTextStandardDetector::new()?;
        let language_switching_detector = LanguageSwitchingDetector::new();
        let mut detector = Self {
            config: None,
            stats: DetectionStats::default(),
            text_detector,
            audio_language_weights: HashMap::new(),
            language_switching_detector,
            text_weight: 0.7, // Default: prioritize text slightly
            audio_weight: 0.3,
            temporal_smoothing: 0.8,
            language_history: Vec::new(),
            max_history_size: 10,
        };

        detector.initialize_audio_models()?;

        Ok(detector)
    }

    /// Initialize audio language weighting models
    fn initialize_audio_models(&mut self) -> Result<()> {
        info!("📊 Initializing audio-based language detection models...");

        let languages = self.text_detector.supported_languages();
        if languages.is_empty() {
            return Ok(());
        }

        let base_weight = 1.0 / languages.len() as f32;
        self.audio_language_weights = languages
            .into_iter()
            .map(|lang| (lang, base_weight))
            .collect();

        // Provide gentle boosts for languages commonly encountered in audio hints
        for (lang, weight) in self.audio_language_weights.iter_mut() {
            match lang.as_str() {
                "en" => *weight *= 1.4,
                "es" | "zh" => *weight *= 1.2,
                _ => {}
            }
        }

        // Normalize weights to sum to 1.0
        let total: f32 = self.audio_language_weights.values().sum();
        if total > 0.0 {
            for weight in self.audio_language_weights.values_mut() {
                *weight /= total;
            }
        }

        info!("✅ Audio language models initialized");
        Ok(())
    }

    /// Perform multimodal language detection
    fn detect_multimodal_fusion(
        &mut self,
        text: &str,
        audio_language_hint: Option<&str>,
    ) -> Result<LanguageDetection> {
        let start_time = Instant::now();

        // Step 1: Get text-based detection
        let text_result = self.text_detector.detect_text(text)?;

        // Step 2: Process audio language hint
        let audio_signal = self.process_audio_language_hint(audio_language_hint);

        // Step 3: Apply fusion strategy
        let fusion_result = self.fuse_text_audio_signals(&text_result, &audio_signal)?;

        // Step 4: Apply temporal smoothing
        let smoothed_result = self.apply_temporal_smoothing(fusion_result)?;

        // Step 5: Check for language switching
        let final_result = self.process_language_switching(smoothed_result)?;

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;
        let mut result = final_result;
        result.processing_time_ms = processing_time;
        result.detection_method = DetectionMethod::MultiModal;

        // Update history
        self.update_language_history(&result);

        debug!(
            "Multimodal detection: {} ({:.3} confidence) in {:.2}ms",
            result.language, result.confidence, processing_time
        );

        Ok(result)
    }

    /// Process audio language hint into probability distribution
    fn process_audio_language_hint(&self, audio_hint: Option<&str>) -> AudioLanguageSignal {
        let mut signal = AudioLanguageSignal {
            language_probabilities: HashMap::new(),
            confidence: 0.0,
            has_signal: false,
        };

        if let Some(hint_lang) = audio_hint {
            signal.has_signal = true;
            signal.confidence = 0.8; // Assume good audio quality

            let mut remaining = 0.2;
            signal
                .language_probabilities
                .insert(hint_lang.to_string(), 0.8);

            let related_languages = self.get_related_languages(hint_lang);
            if !related_languages.is_empty() {
                let share = remaining / related_languages.len() as f32;
                for lang in related_languages {
                    signal
                        .language_probabilities
                        .entry(lang)
                        .and_modify(|weight| *weight += share)
                        .or_insert(share);
                }
                remaining = 0.0;
            }

            if remaining > 0.0 {
                let base_weight = remaining / self.audio_language_weights.len().max(1) as f32;
                for (lang, _) in &self.audio_language_weights {
                    signal
                        .language_probabilities
                        .entry(lang.clone())
                        .and_modify(|w| *w += base_weight)
                        .or_insert(base_weight);
                }
            }
        } else {
            // Use baseline audio weighting to provide gentle prior
            signal.has_signal = !self.audio_language_weights.is_empty();
            signal.confidence = 0.25;
            signal.language_probabilities = self.audio_language_weights.clone();
        }

        signal
    }

    /// Fuse text and audio language signals
    fn fuse_text_audio_signals(
        &self,
        text_result: &LanguageDetection,
        audio_signal: &AudioLanguageSignal,
    ) -> Result<LanguageDetection> {
        if !audio_signal.has_signal {
            // No audio signal, return text result
            return Ok(text_result.clone());
        }

        let fusion_strategy = FusionStrategy::Adaptive;

        match fusion_strategy {
            FusionStrategy::WeightedAverage => {
                self.fuse_weighted_average(text_result, audio_signal)
            }
            FusionStrategy::MaxConfidence => self.fuse_max_confidence(text_result, audio_signal),
            FusionStrategy::Multiplicative => self.fuse_multiplicative(text_result, audio_signal),
            FusionStrategy::Adaptive => self.fuse_adaptive(text_result, audio_signal),
        }
    }

    /// Weighted average fusion
    fn fuse_weighted_average(
        &self,
        text_result: &LanguageDetection,
        audio_signal: &AudioLanguageSignal,
    ) -> Result<LanguageDetection> {
        let mut language_scores: HashMap<String, f32> = HashMap::new();

        // Add text signal
        language_scores.insert(
            text_result.language.clone(),
            text_result.confidence * self.text_weight,
        );

        for alt in &text_result.alternatives {
            let current_score = language_scores.get(&alt.language).copied().unwrap_or(0.0);
            language_scores.insert(
                alt.language.clone(),
                current_score + alt.confidence * self.text_weight,
            );
        }

        // Add audio signal
        for (lang, prob) in &audio_signal.language_probabilities {
            let current_score = language_scores.get(lang).copied().unwrap_or(0.0);
            language_scores.insert(lang.clone(), current_score + prob * self.audio_weight);
        }

        // Find best language
        let best_match = language_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let confidence = *best_match.1;
        let language = best_match.0.clone();

        // Generate alternatives
        let mut alternatives = Vec::new();
        for (lang, score) in &language_scores {
            if lang != &language && *score > 0.05 {
                alternatives.push(LanguageCandidate {
                    language: lang.clone(),
                    confidence: *score,
                });
            }
        }

        alternatives.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        alternatives.truncate(3);

        Ok(LanguageDetection {
            language,
            confidence,
            alternatives,
            processing_time_ms: 0.0, // Will be set by caller
            detection_method: DetectionMethod::MultiModal,
        })
    }

    /// Maximum confidence fusion
    fn fuse_max_confidence(
        &self,
        text_result: &LanguageDetection,
        audio_signal: &AudioLanguageSignal,
    ) -> Result<LanguageDetection> {
        let text_confidence = text_result.confidence;
        let audio_confidence = audio_signal.confidence;

        if text_confidence >= audio_confidence {
            Ok(text_result.clone())
        } else {
            // Use audio signal if it's more confident
            let best_audio_lang = audio_signal
                .language_probabilities
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            Ok(LanguageDetection {
                language: best_audio_lang.0.clone(),
                confidence: *best_audio_lang.1,
                alternatives: Vec::new(),
                processing_time_ms: 0.0,
                detection_method: DetectionMethod::MultiModal,
            })
        }
    }

    /// Multiplicative fusion
    fn fuse_multiplicative(
        &self,
        text_result: &LanguageDetection,
        audio_signal: &AudioLanguageSignal,
    ) -> Result<LanguageDetection> {
        let mut language_scores: HashMap<String, f32> = HashMap::new();

        // Calculate multiplicative scores
        for (audio_lang, audio_prob) in &audio_signal.language_probabilities {
            if audio_lang == &text_result.language {
                let fused_score = text_result.confidence * audio_prob;
                language_scores.insert(audio_lang.clone(), fused_score);
            } else {
                // Check alternatives
                for alt in &text_result.alternatives {
                    if alt.language == *audio_lang {
                        let fused_score = alt.confidence * audio_prob;
                        language_scores.insert(audio_lang.clone(), fused_score);
                        break;
                    }
                }
            }
        }

        // Normalize scores
        let total_score: f32 = language_scores.values().sum();
        if total_score > 0.0 {
            for score in language_scores.values_mut() {
                *score /= total_score;
            }
        }

        // Find best match
        let best_match = language_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((&text_result.language, &text_result.confidence));

        Ok(LanguageDetection {
            language: best_match.0.clone(),
            confidence: *best_match.1,
            alternatives: Vec::new(),
            processing_time_ms: 0.0,
            detection_method: DetectionMethod::MultiModal,
        })
    }

    /// Adaptive fusion based on confidence levels
    fn fuse_adaptive(
        &self,
        text_result: &LanguageDetection,
        audio_signal: &AudioLanguageSignal,
    ) -> Result<LanguageDetection> {
        let text_confidence = text_result.confidence;
        let _audio_confidence = audio_signal.confidence;

        // Adaptive weighting based on confidence levels
        let adaptive_text_weight = if text_confidence > 0.8 {
            0.8
        } else if text_confidence > 0.6 {
            0.6
        } else {
            0.4
        };

        let adaptive_audio_weight = 1.0 - adaptive_text_weight;

        // Temporarily update weights
        let _original_text_weight = self.text_weight;
        let _original_audio_weight = self.audio_weight;

        // Use adaptive weights for this detection
        let temp_detector = MultimodalDetector {
            config: self.config.clone(),
            stats: self.stats.clone(),
            text_detector: FastTextStandardDetector::new()?, // Placeholder
            audio_language_weights: self.audio_language_weights.clone(),
            language_switching_detector: LanguageSwitchingDetector::new(),
            text_weight: adaptive_text_weight,
            audio_weight: adaptive_audio_weight,
            temporal_smoothing: self.temporal_smoothing,
            language_history: self.language_history.clone(),
            max_history_size: self.max_history_size,
        };

        temp_detector.fuse_weighted_average(text_result, audio_signal)
    }

    /// Apply temporal smoothing using language history
    fn apply_temporal_smoothing(
        &self,
        current_result: LanguageDetection,
    ) -> Result<LanguageDetection> {
        if self.language_history.is_empty() {
            return Ok(current_result);
        }

        // Get recent stable language
        let recent_detections: Vec<&HistoricalDetection> =
            self.language_history.iter().rev().take(5).collect();

        if recent_detections.is_empty() {
            return Ok(current_result);
        }

        // Check for consistent language
        let most_common_lang = self.find_most_common_language(&recent_detections);

        if let Some((stable_lang, stability_score)) = most_common_lang {
            if stability_score > 0.6 && stable_lang != current_result.language {
                // Apply temporal smoothing
                let smoothed_confidence = current_result.confidence
                    * (1.0 - self.temporal_smoothing)
                    + stability_score * self.temporal_smoothing;

                if smoothed_confidence < current_result.confidence * 0.8 {
                    // Keep stable language if new detection isn't significantly more confident
                    return Ok(LanguageDetection {
                        language: stable_lang,
                        confidence: smoothed_confidence,
                        alternatives: current_result.alternatives,
                        processing_time_ms: current_result.processing_time_ms,
                        detection_method: current_result.detection_method,
                    });
                }
            }
        }

        Ok(current_result)
    }

    /// Process language switching detection
    fn process_language_switching(
        &mut self,
        result: LanguageDetection,
    ) -> Result<LanguageDetection> {
        self.language_switching_detector.process_detection(&result)
    }

    /// Update language history with new detection
    fn update_language_history(&mut self, result: &LanguageDetection) {
        let historical_detection = HistoricalDetection {
            language: result.language.clone(),
            confidence: result.confidence,
            timestamp: Instant::now(),
            detection_method: result.detection_method.clone(),
        };

        self.language_history.push(historical_detection);

        // Limit history size
        if self.language_history.len() > self.max_history_size {
            self.language_history.remove(0);
        }
    }

    /// Find most common language in recent history
    fn find_most_common_language(
        &self,
        detections: &[&HistoricalDetection],
    ) -> Option<(String, f32)> {
        let mut language_counts: HashMap<String, f32> = HashMap::new();

        for detection in detections {
            let current_score = language_counts
                .get(&detection.language)
                .copied()
                .unwrap_or(0.0);
            language_counts.insert(
                detection.language.clone(),
                current_score + detection.confidence,
            );
        }

        let total_score: f32 = language_counts.values().sum();
        if total_score == 0.0 {
            return None;
        }

        let best_match = language_counts
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        Some((best_match.0.clone(), best_match.1 / total_score))
    }

    /// Get related languages for a given language
    fn get_related_languages(&self, language: &str) -> Vec<String> {
        match language {
            "en" => vec!["de".to_string(), "nl".to_string(), "sv".to_string()],
            "es" => vec!["pt".to_string(), "it".to_string(), "fr".to_string()],
            "fr" => vec!["es".to_string(), "it".to_string(), "pt".to_string()],
            "de" => vec!["en".to_string(), "nl".to_string(), "sv".to_string()],
            "zh" => vec!["ja".to_string(), "ko".to_string()],
            "ja" => vec!["zh".to_string(), "ko".to_string()],
            "ko" => vec!["zh".to_string(), "ja".to_string()],
            _ => vec!["en".to_string()],
        }
    }
}

/// Audio language signal representation
struct AudioLanguageSignal {
    language_probabilities: HashMap<String, f32>,
    confidence: f32,
    has_signal: bool,
}

impl LanguageSwitchingDetector {
    fn new() -> Self {
        Self {
            switch_threshold: 0.8,
            min_confidence_for_switch: 0.7,
            switch_cooldown_ms: 2000, // 2 seconds
            last_switch_time: None,
            current_stable_language: None,
        }
    }

    fn process_detection(&mut self, result: &LanguageDetection) -> Result<LanguageDetection> {
        let now = Instant::now();

        // Check if we're in cooldown period
        if let Some(last_switch) = self.last_switch_time {
            if now.duration_since(last_switch).as_millis() < self.switch_cooldown_ms as u128 {
                // In cooldown, maintain stable language if confidence is reasonable
                if let Some(ref stable_lang) = self.current_stable_language {
                    if result.confidence < self.switch_threshold {
                        return Ok(LanguageDetection {
                            language: stable_lang.clone(),
                            confidence: result.confidence * 0.9, // Slightly reduce confidence
                            alternatives: result.alternatives.clone(),
                            processing_time_ms: result.processing_time_ms,
                            detection_method: result.detection_method.clone(),
                        });
                    }
                }
            }
        }

        // Check for language switch
        if let Some(ref current_lang) = self.current_stable_language {
            if &result.language != current_lang
                && result.confidence >= self.min_confidence_for_switch
            {
                // Language switch detected
                info!(
                    "🔄 Language switch detected: {} -> {} ({:.3} confidence)",
                    current_lang, result.language, result.confidence
                );

                self.current_stable_language = Some(result.language.clone());
                self.last_switch_time = Some(now);
            }
        } else {
            // First detection
            self.current_stable_language = Some(result.language.clone());
        }

        Ok(result.clone())
    }
}

impl LanguageDetector for MultimodalDetector {
    fn initialize(&mut self, config: LanguageDetectorConfig) -> Result<()> {
        self.text_detector.initialize(config.clone())?;
        self.initialize_audio_models()?;
        self.config = Some(config);

        debug!("Multimodal detector initialized");
        Ok(())
    }

    fn detect_text(&mut self, text: &str) -> Result<LanguageDetection> {
        // Fallback to text-only detection
        self.text_detector.detect_text(text)
    }

    fn detect_multimodal(
        &mut self,
        text: &str,
        audio_language_hint: Option<&str>,
    ) -> Result<LanguageDetection> {
        self.detect_multimodal_fusion(text, audio_language_hint)
    }

    fn supported_languages(&self) -> Vec<String> {
        self.text_detector.supported_languages()
    }

    fn profile(&self) -> Profile {
        Profile::High
    }

    fn update_config(&mut self, config: LanguageDetectorConfig) -> Result<()> {
        self.text_detector.update_config(config.clone())?;
        self.config = Some(config);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.text_detector.reset()?;
        self.stats = DetectionStats::default();
        self.language_history.clear();
        self.language_switching_detector = LanguageSwitchingDetector::new();
        Ok(())
    }

    fn get_stats(&self) -> DetectionStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_detector_creation() {
        let detector = MultimodalDetector::new();
        assert!(detector.is_ok());
    }

    #[test]
    fn test_language_switching_detector() {
        let mut detector = LanguageSwitchingDetector::new();

        let result1 = LanguageDetection {
            language: "en".to_string(),
            confidence: 0.9,
            alternatives: Vec::new(),
            processing_time_ms: 2.0,
            detection_method: DetectionMethod::MultiModal,
        };

        let processed = detector.process_detection(&result1);
        assert!(processed.is_ok());
        assert_eq!(detector.current_stable_language, Some("en".to_string()));
    }

    #[test]
    fn test_audio_language_signal() {
        let signal = AudioLanguageSignal {
            language_probabilities: HashMap::from([
                ("en".to_string(), 0.8),
                ("es".to_string(), 0.2),
            ]),
            confidence: 0.8,
            has_signal: true,
        };

        assert!(signal.has_signal);
        assert_eq!(signal.confidence, 0.8);
        assert_eq!(signal.language_probabilities.len(), 2);
    }

    #[test]
    fn test_fusion_strategies() {
        // Test that different fusion strategies are defined
        assert_eq!(
            FusionStrategy::WeightedAverage,
            FusionStrategy::WeightedAverage
        );
        assert_ne!(
            FusionStrategy::WeightedAverage,
            FusionStrategy::MaxConfidence
        );
    }
}
