//! Adaptive VAD framework for enhanced performance
//!
//! This module provides adaptive enhancements over base VAD implementations:
//! - Confidence scoring system
//! - Temporal smoothing with majority voting
//! - Noise-aware threshold adaptation
//! - Speech/music discrimination
//! - Frame-level and segment-level decisions

use super::{VadConfig, VadProcessor, VadResult, VadStats};
use anyhow::Result;
use std::collections::VecDeque;
use tracing::{debug, info};

/// Configuration for adaptive VAD processing
#[derive(Debug, Clone)]
pub struct AdaptiveVadConfig {
    /// Base VAD configuration
    pub base_config: VadConfig,

    /// Temporal smoothing window size
    pub smoothing_window_size: usize,

    /// Minimum consecutive frames for speech onset
    pub speech_onset_frames: usize,

    /// Minimum consecutive frames for speech offset
    pub speech_offset_frames: usize,

    /// Confidence threshold for high-confidence decisions
    pub high_confidence_threshold: f32,

    /// Confidence threshold for low-confidence decisions
    pub low_confidence_threshold: f32,

    /// Enable noise floor adaptation
    pub enable_noise_adaptation: bool,

    /// Enable music discrimination
    pub enable_music_discrimination: bool,

    /// Segment analysis window size (for segment-level decisions)
    pub segment_window_size: usize,
}

impl Default for AdaptiveVadConfig {
    fn default() -> Self {
        Self {
            base_config: VadConfig::default(),
            smoothing_window_size: 7,
            speech_onset_frames: 2,
            speech_offset_frames: 3,
            high_confidence_threshold: 0.8,
            low_confidence_threshold: 0.3,
            enable_noise_adaptation: true,
            enable_music_discrimination: true,
            segment_window_size: 50, // ~0.5 seconds at 10ms hop
        }
    }
}

/// Enhanced VAD result with segment-level information
#[derive(Debug, Clone)]
pub struct AdaptiveVadResult {
    /// Base VAD result
    pub base_result: VadResult,

    /// Smoothed speech decision
    pub smoothed_speech: bool,

    /// Segment-level speech decision
    pub segment_speech: bool,

    /// Confidence category
    pub confidence_category: ConfidenceCategory,

    /// Detected content type
    pub content_type: ContentType,

    /// Noise floor estimate
    pub noise_floor: f32,

    /// Signal-to-noise ratio estimate
    pub snr_estimate: f32,
}

/// Confidence categories for decision making
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfidenceCategory {
    High,
    Medium,
    Low,
    VeryLow,
}

/// Detected content types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContentType {
    Speech,
    Music,
    Noise,
    Silence,
    Unknown,
}

/// Adaptive VAD wrapper that enhances any base VAD implementation
pub struct AdaptiveVad {
    /// Base VAD processor
    base_vad: Box<dyn VadProcessor>,

    /// Configuration
    config: AdaptiveVadConfig,

    /// Frame-level history for temporal smoothing
    frame_history: VecDeque<VadResult>,

    /// Confidence history for adaptation
    confidence_history: VecDeque<f32>,

    /// Energy history for noise floor estimation
    energy_history: VecDeque<f32>,

    /// Spectral features history for music discrimination
    spectral_history: VecDeque<f32>,

    /// Current noise floor estimate
    noise_floor: f32,

    /// Current speech state tracking
    speech_state: SpeechState,

    /// Segment-level analysis buffer
    segment_buffer: VecDeque<bool>,

    /// Enhanced statistics
    adaptive_stats: AdaptiveStats,
}

/// Speech state tracking for onset/offset detection
#[derive(Debug, Clone)]
struct SpeechState {
    is_speaking: bool,
    consecutive_speech_frames: usize,
    consecutive_nonspeech_frames: usize,
    last_transition_frame: u64,
}

/// Enhanced statistics for adaptive VAD
#[derive(Debug, Clone, Default)]
pub struct AdaptiveStats {
    pub base_stats: VadStats,
    pub smoothing_corrections: u64,
    pub noise_adaptations: u64,
    pub music_detections: u64,
    pub false_positive_reductions: u64,
    pub false_negative_corrections: u64,
    pub average_snr: f32,
    pub confidence_distribution: [u64; 4], // [VeryLow, Low, Medium, High]
}

impl AdaptiveVad {
    /// Create a new adaptive VAD wrapper
    pub fn new(base_vad: Box<dyn VadProcessor>, config: AdaptiveVadConfig) -> Self {
        info!("🧠 Initializing Adaptive VAD framework");

        let history_size = config.smoothing_window_size.max(config.segment_window_size);

        Self {
            base_vad,
            config: config.clone(),
            frame_history: VecDeque::with_capacity(history_size),
            confidence_history: VecDeque::with_capacity(history_size),
            energy_history: VecDeque::with_capacity(history_size * 2),
            spectral_history: VecDeque::with_capacity(history_size),
            noise_floor: 0.001,
            speech_state: SpeechState {
                is_speaking: false,
                consecutive_speech_frames: 0,
                consecutive_nonspeech_frames: 0,
                last_transition_frame: 0,
            },
            segment_buffer: VecDeque::with_capacity(config.segment_window_size),
            adaptive_stats: AdaptiveStats::default(),
        }
    }

    /// Process audio frame with adaptive enhancements
    pub fn process_frame(&mut self, audio_frame: &[f32]) -> Result<AdaptiveVadResult> {
        // Get base VAD result
        let base_result = self.base_vad.process_frame(audio_frame)?;

        // Update histories
        self.update_histories(&base_result);

        // Adapt noise floor if enabled
        if self.config.enable_noise_adaptation {
            self.adapt_noise_floor();
        }

        // Classify confidence category
        let confidence_category = self.classify_confidence(base_result.confidence);

        // Detect content type
        let content_type = self.detect_content_type(&base_result);

        // Apply temporal smoothing
        let smoothed_speech = self.apply_temporal_smoothing(&base_result);

        // Update speech state tracking
        self.update_speech_state(smoothed_speech, base_result.frame_number);

        // Perform segment-level analysis
        let segment_speech = self.analyze_segment_level();

        // Calculate SNR estimate
        let snr_estimate = self.estimate_snr(&base_result);

        // Update adaptive statistics
        self.update_adaptive_stats(&base_result, confidence_category, content_type);

        // Create enhanced result
        let adaptive_result = AdaptiveVadResult {
            base_result,
            smoothed_speech,
            segment_speech,
            confidence_category,
            content_type,
            noise_floor: self.noise_floor,
            snr_estimate,
        };

        debug!(
            "Adaptive VAD: base={}, smoothed={}, segment={}, type={:?}, SNR={:.1}dB",
            adaptive_result.base_result.is_speech,
            smoothed_speech,
            segment_speech,
            content_type,
            20.0 * snr_estimate.log10()
        );

        Ok(adaptive_result)
    }

    /// Update internal histories with new frame data
    fn update_histories(&mut self, result: &VadResult) {
        // Update frame history
        self.frame_history.push_back(result.clone());
        if self.frame_history.len() > self.config.smoothing_window_size {
            self.frame_history.pop_front();
        }

        // Update confidence history
        self.confidence_history.push_back(result.confidence);
        if self.confidence_history.len() > self.config.smoothing_window_size {
            self.confidence_history.pop_front();
        }

        // Update energy history
        self.energy_history.push_back(result.metadata.energy);
        if self.energy_history.len() > self.config.smoothing_window_size * 2 {
            self.energy_history.pop_front();
        }

        // Update spectral history if available
        if let Some(spectral_centroid) = result.metadata.spectral_centroid {
            self.spectral_history.push_back(spectral_centroid);
            if self.spectral_history.len() > self.config.smoothing_window_size {
                self.spectral_history.pop_front();
            }
        }
    }

    /// Adapt noise floor based on recent low-energy frames
    fn adapt_noise_floor(&mut self) {
        if self.energy_history.len() < 10 {
            return;
        }

        // Find minimum energy values (likely noise)
        let mut sorted_energies: Vec<f32> = self.energy_history.iter().cloned().collect();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use 10th percentile as noise floor estimate
        let percentile_10_idx = sorted_energies.len() / 10;
        let new_noise_floor = sorted_energies[percentile_10_idx];

        // Smooth the adaptation
        const ADAPTATION_RATE: f32 = 0.05;
        self.noise_floor =
            self.noise_floor * (1.0 - ADAPTATION_RATE) + new_noise_floor * ADAPTATION_RATE;

        // Ensure minimum noise floor
        self.noise_floor = self.noise_floor.max(0.0001);

        self.adaptive_stats.noise_adaptations += 1;
    }

    /// Classify confidence into categories
    fn classify_confidence(&self, confidence: f32) -> ConfidenceCategory {
        let high = self.config.high_confidence_threshold;
        let low = self.config.low_confidence_threshold;
        let medium_band = low + (high - low) / 3.0;

        if confidence >= high {
            ConfidenceCategory::High
        } else if confidence >= medium_band {
            ConfidenceCategory::Medium
        } else if confidence >= low {
            ConfidenceCategory::Low
        } else {
            ConfidenceCategory::VeryLow
        }
    }

    /// Detect content type based on signal characteristics
    fn detect_content_type(&mut self, result: &VadResult) -> ContentType {
        let energy = result.metadata.energy;
        let zcr = result.metadata.zero_crossing_rate;

        // Silence detection
        if energy < self.noise_floor * 2.0 {
            return ContentType::Silence;
        }

        // Music vs speech discrimination using spectral features
        if self.config.enable_music_discrimination {
            if let Some(spectral_centroid) = result.metadata.spectral_centroid {
                // Music typically has higher spectral centroid and lower ZCR variability
                if spectral_centroid > 3000.0 && zcr < 0.1 {
                    self.adaptive_stats.music_detections += 1;
                    return ContentType::Music;
                }
            }
        }

        // Speech detection based on VAD result and characteristics
        if result.is_speech && zcr > 0.02 && zcr < 0.4 {
            ContentType::Speech
        } else if energy > self.noise_floor * 5.0 {
            ContentType::Noise
        } else {
            ContentType::Unknown
        }
    }

    /// Apply temporal smoothing with majority voting
    fn apply_temporal_smoothing(&mut self, current_result: &VadResult) -> bool {
        if self.frame_history.len() < 3 {
            return current_result.is_speech;
        }

        // Count speech decisions in recent history
        let speech_count = self.frame_history.iter().filter(|r| r.is_speech).count();

        let total_frames = self.frame_history.len();
        let majority_threshold = (total_frames + 1) / 2;

        let smoothed_decision = speech_count >= majority_threshold;

        // Track corrections
        if smoothed_decision != current_result.is_speech {
            self.adaptive_stats.smoothing_corrections += 1;

            if smoothed_decision && !current_result.is_speech {
                self.adaptive_stats.false_negative_corrections += 1;
            } else if !smoothed_decision && current_result.is_speech {
                self.adaptive_stats.false_positive_reductions += 1;
            }
        }

        smoothed_decision
    }

    /// Update speech state tracking for onset/offset detection
    fn update_speech_state(&mut self, is_speech: bool, frame_number: u64) {
        if is_speech {
            self.speech_state.consecutive_speech_frames += 1;
            self.speech_state.consecutive_nonspeech_frames = 0;

            // Speech onset detection
            if !self.speech_state.is_speaking
                && self.speech_state.consecutive_speech_frames >= self.config.speech_onset_frames
            {
                self.speech_state.is_speaking = true;
                self.speech_state.last_transition_frame = frame_number;
                debug!("Speech onset detected at frame {}", frame_number);
            }
        } else {
            self.speech_state.consecutive_nonspeech_frames += 1;
            self.speech_state.consecutive_speech_frames = 0;

            // Speech offset detection
            if self.speech_state.is_speaking
                && self.speech_state.consecutive_nonspeech_frames
                    >= self.config.speech_offset_frames
            {
                self.speech_state.is_speaking = false;
                self.speech_state.last_transition_frame = frame_number;
                debug!("Speech offset detected at frame {}", frame_number);
            }
        }
    }

    /// Analyze segment-level speech activity
    fn analyze_segment_level(&mut self) -> bool {
        // Add current frame decision to segment buffer
        let current_speech = self.speech_state.is_speaking;
        self.segment_buffer.push_back(current_speech);

        if self.segment_buffer.len() > self.config.segment_window_size {
            self.segment_buffer.pop_front();
        }

        if self.segment_buffer.len() < self.config.segment_window_size / 2 {
            return current_speech;
        }

        // Calculate speech ratio in segment
        let speech_frames = self.segment_buffer.iter().filter(|&&s| s).count();
        let speech_ratio = speech_frames as f32 / self.segment_buffer.len() as f32;

        // Segment is considered speech if >30% of frames contain speech
        speech_ratio > 0.3
    }

    /// Estimate signal-to-noise ratio
    fn estimate_snr(&self, result: &VadResult) -> f32 {
        let signal_energy = result.metadata.energy;
        let noise_energy = self.noise_floor;

        if noise_energy > 0.0 {
            signal_energy / noise_energy
        } else {
            1000.0 // Very high SNR if no noise detected
        }
    }

    /// Update adaptive statistics
    fn update_adaptive_stats(
        &mut self,
        result: &VadResult,
        confidence_category: ConfidenceCategory,
        _content_type: ContentType,
    ) {
        // Update base stats
        self.adaptive_stats.base_stats = self.base_vad.get_stats();

        // Update confidence distribution
        let conf_idx = match confidence_category {
            ConfidenceCategory::VeryLow => 0,
            ConfidenceCategory::Low => 1,
            ConfidenceCategory::Medium => 2,
            ConfidenceCategory::High => 3,
        };
        self.adaptive_stats.confidence_distribution[conf_idx] += 1;

        // Update average SNR
        let snr = self.estimate_snr(result);
        let total_frames = self.adaptive_stats.base_stats.total_frames as f32;
        self.adaptive_stats.average_snr =
            (self.adaptive_stats.average_snr * (total_frames - 1.0) + snr) / total_frames;
    }

    /// Get comprehensive statistics
    pub fn get_adaptive_stats(&self) -> &AdaptiveStats {
        &self.adaptive_stats
    }

    /// Reset all adaptive state
    pub fn reset(&mut self) {
        self.base_vad.reset();
        self.frame_history.clear();
        self.confidence_history.clear();
        self.energy_history.clear();
        self.spectral_history.clear();
        self.segment_buffer.clear();
        self.noise_floor = 0.001;
        self.speech_state = SpeechState {
            is_speaking: false,
            consecutive_speech_frames: 0,
            consecutive_nonspeech_frames: 0,
            last_transition_frame: 0,
        };
        self.adaptive_stats = AdaptiveStats::default();
    }
}

impl VadProcessor for AdaptiveVad {
    fn process_frame(&mut self, audio_frame: &[f32]) -> Result<VadResult> {
        // For compatibility, return the base result
        // Users should call process_frame directly for enhanced results
        self.base_vad.process_frame(audio_frame)
    }

    fn frame_size(&self) -> usize {
        self.base_vad.frame_size()
    }

    fn sample_rate(&self) -> u32 {
        self.base_vad.sample_rate()
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn get_stats(&self) -> VadStats {
        self.adaptive_stats.base_stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::vad::webrtc_vad::WebRtcVad;
    use crate::audio::vad::VadMetadata;

    #[test]
    fn test_adaptive_vad_creation() {
        let base_config = VadConfig::default();
        let adaptive_config = AdaptiveVadConfig::default();
        let base_vad = Box::new(WebRtcVad::new(base_config).unwrap());

        let adaptive_vad = AdaptiveVad::new(base_vad, adaptive_config);
        assert_eq!(adaptive_vad.frame_size(), 480);
    }

    #[test]
    fn test_confidence_classification() {
        let base_config = VadConfig::default();
        let adaptive_config = AdaptiveVadConfig::default();
        let base_vad = Box::new(WebRtcVad::new(base_config).unwrap());
        let adaptive_vad = AdaptiveVad::new(base_vad, adaptive_config);

        assert_eq!(
            adaptive_vad.classify_confidence(0.9),
            ConfidenceCategory::High
        );
        assert_eq!(
            adaptive_vad.classify_confidence(0.5),
            ConfidenceCategory::Medium
        );
        assert_eq!(
            adaptive_vad.classify_confidence(0.2),
            ConfidenceCategory::VeryLow
        );
    }

    #[test]
    fn test_snr_estimation() {
        let base_config = VadConfig::default();
        let adaptive_config = AdaptiveVadConfig::default();
        let base_vad = Box::new(WebRtcVad::new(base_config).unwrap());
        let adaptive_vad = AdaptiveVad::new(base_vad, adaptive_config);

        let test_result = VadResult {
            is_speech: true,
            confidence: 0.8,
            processing_time_ms: 1.0,
            frame_number: 0,
            metadata: VadMetadata {
                energy: 0.1,
                zero_crossing_rate: 0.2,
                spectral_centroid: None,
                model_scores: vec![0.8],
            },
        };

        let snr = adaptive_vad.estimate_snr(&test_result);
        assert!(snr > 0.0);
    }
}
