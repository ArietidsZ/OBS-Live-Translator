//! WebRTC VAD implementation for Low Profile
//!
//! This module implements a Rust-native WebRTC-inspired VAD with:
//! - 30ms frame processing (480 samples at 16kHz)
//! - Energy-based detection with ZCR fallback
//! - Sub-millisecond processing target
//! - 158KB memory footprint target

use super::{VadProcessor, VadResult, VadConfig, VadMetadata, VadStats};
use anyhow::Result;
use std::collections::VecDeque;
use tracing::{debug, warn};

/// WebRTC-style VAD implementation optimized for low resource usage
pub struct WebRtcVad {
    config: VadConfig,

    // Signal processing state
    _frame_buffer: Vec<f32>,
    energy_history: VecDeque<f32>,
    zcr_history: VecDeque<f32>,

    // Adaptive thresholds
    energy_threshold: f32,
    zcr_threshold: f32,
    noise_floor: f32,

    // Processing statistics
    stats: VadStats,
    frame_count: u64,

    // Temporal smoothing
    speech_history: VecDeque<bool>,
    smoothing_enabled: bool,
}

impl WebRtcVad {
    /// Create a new WebRTC VAD instance
    pub fn new(config: VadConfig) -> Result<Self> {
        // Validate configuration
        if config.sample_rate != 16000 {
            warn!("WebRTC VAD optimized for 16kHz, got {}Hz", config.sample_rate);
        }

        if config.frame_size != 480 {
            warn!("WebRTC VAD optimized for 480-sample frames (30ms), got {}", config.frame_size);
        }

        let history_size = config.smoothing_window;

        Ok(Self {
            config: config.clone(),
            _frame_buffer: Vec::with_capacity(480),
            energy_history: VecDeque::with_capacity(history_size),
            zcr_history: VecDeque::with_capacity(history_size),
            energy_threshold: 0.01,  // Initial threshold
            zcr_threshold: 0.3,      // Initial ZCR threshold
            noise_floor: 0.001,      // Minimum energy level
            stats: VadStats::default(),
            frame_count: 0,
            speech_history: VecDeque::with_capacity(history_size),
            smoothing_enabled: config.enable_smoothing,
        })
    }

    /// Compute signal energy (RMS)
    fn compute_energy(&self, samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Compute zero crossing rate
    fn compute_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        zero_crossings as f32 / (samples.len() - 1) as f32
    }

    /// Apply pre-emphasis filter to enhance signal characteristics
    fn apply_pre_emphasis(&self, samples: &[f32]) -> Vec<f32> {
        let mut filtered = Vec::with_capacity(samples.len());
        filtered.push(samples[0]); // First sample unchanged

        const ALPHA: f32 = 0.97;
        for i in 1..samples.len() {
            filtered.push(samples[i] - ALPHA * samples[i - 1]);
        }

        filtered
    }

    /// Update adaptive thresholds based on signal history
    fn update_adaptive_thresholds(&mut self) {
        if self.energy_history.len() < 10 {
            return; // Need enough history
        }

        // Calculate statistics from recent history
        let energies: Vec<f32> = self.energy_history.iter().cloned().collect();
        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;

        // Calculate standard deviation
        let variance = energies.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>() / energies.len() as f32;
        let std_dev = variance.sqrt();

        // Update noise floor (minimum of recent energies)
        self.noise_floor = energies.iter().cloned().fold(f32::INFINITY, f32::min)
            .max(0.0001); // Prevent zero noise floor

        // Adaptive energy threshold: mean + sensitivity * std_dev
        self.energy_threshold = (mean_energy + self.config.sensitivity * std_dev)
            .max(self.noise_floor * 3.0); // At least 3x noise floor

        // ZCR threshold based on historical variation
        if self.zcr_history.len() >= 5 {
            let zcr_values: Vec<f32> = self.zcr_history.iter().cloned().collect();
            let mean_zcr = zcr_values.iter().sum::<f32>() / zcr_values.len() as f32;
            self.zcr_threshold = mean_zcr + 0.1; // Slightly above average
        }

        debug!("Updated thresholds: energy={:.4}, zcr={:.3}, noise_floor={:.4}",
               self.energy_threshold, self.zcr_threshold, self.noise_floor);
    }

    /// Classify frame as speech or non-speech using multiple features
    fn classify_frame(&self, energy: f32, zcr: f32) -> (bool, f32) {
        let mut confidence = 0.0;
        let mut speech_indicators = 0;
        let total_indicators = 3;

        // Feature 1: Energy above threshold
        if energy > self.energy_threshold {
            speech_indicators += 1;
            confidence += 0.4;
        }

        // Feature 2: Energy significantly above noise floor
        let snr = energy / self.noise_floor.max(0.0001);
        if snr > 10.0 { // 20dB SNR
            speech_indicators += 1;
            confidence += 0.3;
        }

        // Feature 3: ZCR in speech range (not too low, not too high)
        if zcr > 0.05 && zcr < self.zcr_threshold {
            speech_indicators += 1;
            confidence += 0.3;
        }

        // Decision based on majority voting
        let is_speech = speech_indicators >= 2;

        // Scale confidence based on how many indicators agree
        confidence *= speech_indicators as f32 / total_indicators as f32;

        // Apply sensitivity adjustment
        let sensitivity_factor = (self.config.sensitivity - 0.5) * 0.2;
        confidence = (confidence + sensitivity_factor).clamp(0.0, 1.0);

        (is_speech, confidence)
    }

    /// Apply temporal smoothing to reduce false positives/negatives
    fn apply_temporal_smoothing(&self, current_decision: bool) -> bool {
        if !self.smoothing_enabled || self.speech_history.len() < 3 {
            return current_decision;
        }

        // Count recent speech decisions
        let recent_speech_count = self.speech_history.iter()
            .filter(|&&is_speech| is_speech)
            .count();

        // Majority voting over recent frames
        let majority_threshold = (self.speech_history.len() + 1) / 2;

        if current_decision {
            // If current frame is speech, confirm if recent history supports it
            recent_speech_count >= majority_threshold.saturating_sub(1)
        } else {
            // If current frame is not speech, confirm if recent history supports it
            recent_speech_count < majority_threshold
        }
    }
}

impl VadProcessor for WebRtcVad {
    fn process_frame(&mut self, audio_frame: &[f32]) -> Result<VadResult> {
        let start_time = std::time::Instant::now();

        // Validate input
        if audio_frame.len() != self.config.frame_size {
            return Err(anyhow::anyhow!(
                "Invalid frame size: expected {}, got {}",
                self.config.frame_size,
                audio_frame.len()
            ));
        }

        // Apply pre-emphasis to enhance signal characteristics
        let pre_emphasized = self.apply_pre_emphasis(audio_frame);

        // Compute frame features
        let energy = self.compute_energy(&pre_emphasized);
        let zcr = self.compute_zero_crossing_rate(audio_frame);

        // Update feature history
        self.energy_history.push_back(energy);
        self.zcr_history.push_back(zcr);

        // Maintain history size
        let max_history = self.config.smoothing_window.max(20);
        if self.energy_history.len() > max_history {
            self.energy_history.pop_front();
        }
        if self.zcr_history.len() > max_history {
            self.zcr_history.pop_front();
        }

        // Update adaptive thresholds periodically
        if self.frame_count % 10 == 0 {
            self.update_adaptive_thresholds();
        }

        // Classify current frame
        let (is_speech_raw, confidence) = self.classify_frame(energy, zcr);

        // Apply temporal smoothing
        let is_speech = self.apply_temporal_smoothing(is_speech_raw);

        // Update speech history
        self.speech_history.push_back(is_speech);
        if self.speech_history.len() > self.config.smoothing_window {
            self.speech_history.pop_front();
        }

        // Calculate processing time
        let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        self.stats.total_frames += 1;
        if is_speech {
            self.stats.speech_frames += 1;
        }

        // Update timing statistics
        let frame_count = self.stats.total_frames as f32;
        self.stats.average_processing_time_ms =
            (self.stats.average_processing_time_ms * (frame_count - 1.0) + processing_time_ms) / frame_count;

        if processing_time_ms > self.stats.peak_processing_time_ms {
            self.stats.peak_processing_time_ms = processing_time_ms;
        }

        // Update confidence statistics
        self.stats.average_confidence =
            (self.stats.average_confidence * (frame_count - 1.0) + confidence) / frame_count;

        self.frame_count += 1;

        // Create result with metadata
        let metadata = VadMetadata {
            energy,
            zero_crossing_rate: zcr,
            spectral_centroid: None, // Not computed in WebRTC VAD
            model_scores: vec![confidence],
        };

        Ok(VadResult {
            is_speech,
            confidence,
            processing_time_ms,
            frame_number: self.frame_count - 1,
            metadata,
        })
    }

    fn frame_size(&self) -> usize {
        self.config.frame_size
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn reset(&mut self) {
        self.energy_history.clear();
        self.zcr_history.clear();
        self.speech_history.clear();
        self.energy_threshold = 0.01;
        self.zcr_threshold = 0.3;
        self.noise_floor = 0.001;
        self.frame_count = 0;
        self.stats = VadStats::default();
    }

    fn get_stats(&self) -> VadStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webrtc_vad_creation() {
        let config = VadConfig::default();
        let vad = WebRtcVad::new(config).unwrap();
        assert_eq!(vad.frame_size(), 480);
        assert_eq!(vad.sample_rate(), 16000);
    }

    #[test]
    fn test_energy_computation() {
        let config = VadConfig::default();
        let vad = WebRtcVad::new(config).unwrap();

        // Test silence (should have very low energy)
        let silence = vec![0.0; 480];
        let energy = vad.compute_energy(&silence);
        assert!(energy < 0.001);

        // Test sine wave (should have moderate energy)
        let mut sine_wave = Vec::new();
        for i in 0..480 {
            sine_wave.push((2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5);
        }
        let energy = vad.compute_energy(&sine_wave);
        assert!(energy > 0.1 && energy < 1.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let config = VadConfig::default();
        let vad = WebRtcVad::new(config).unwrap();

        // Test DC signal (should have zero crossings)
        let dc_signal = vec![1.0; 480];
        let zcr = vad.compute_zero_crossing_rate(&dc_signal);
        assert_eq!(zcr, 0.0);

        // Test alternating signal (should have high ZCR)
        let mut alternating = Vec::new();
        for i in 0..480 {
            alternating.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let zcr = vad.compute_zero_crossing_rate(&alternating);
        assert!(zcr > 0.8);
    }

    #[test]
    fn test_frame_processing() {
        let config = VadConfig::default();
        let mut vad = WebRtcVad::new(config).unwrap();

        // Process a frame of silence
        let silence_frame = vec![0.0; 480];
        let result = vad.process_frame(&silence_frame).unwrap();

        assert_eq!(result.frame_number, 0);
        assert!(result.processing_time_ms < 10.0); // Should be very fast
        assert!(!result.is_speech); // Silence should not be detected as speech
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_invalid_frame_size() {
        let config = VadConfig::default();
        let mut vad = WebRtcVad::new(config).unwrap();

        // Try to process frame with wrong size
        let wrong_size_frame = vec![0.0; 100];
        let result = vad.process_frame(&wrong_size_frame);

        assert!(result.is_err());
    }

    #[test]
    fn test_reset_functionality() {
        let config = VadConfig::default();
        let mut vad = WebRtcVad::new(config).unwrap();

        // Process some frames
        let frame = vec![0.5; 480];
        for _ in 0..5 {
            let _ = vad.process_frame(&frame).unwrap();
        }

        // Check that state was accumulated
        let stats_before = vad.get_stats();
        assert!(stats_before.total_frames > 0);

        // Reset and check that state is cleared
        vad.reset();
        let stats_after = vad.get_stats();
        assert_eq!(stats_after.total_frames, 0);
        assert_eq!(stats_after.speech_frames, 0);
    }
}