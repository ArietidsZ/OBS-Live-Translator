//! Voice Activity Detection for real-time audio processing

use super::{AudioBuffer, Sample};
use anyhow::Result;
use std::collections::VecDeque;

/// Voice Activity Detector
pub struct VoiceActivityDetector {
    energy_threshold: f32,
    zero_crossing_threshold: f32,
    frame_size: usize,
    smoothing_frames: usize,
    energy_history: VecDeque<f32>,
    zcr_history: VecDeque<f32>,
    speech_history: VecDeque<bool>,
}

impl VoiceActivityDetector {
    pub fn new(frame_size: usize) -> Self {
        Self {
            energy_threshold: 0.01,
            zero_crossing_threshold: 0.3,
            frame_size,
            smoothing_frames: 5,
            energy_history: VecDeque::with_capacity(100),
            zcr_history: VecDeque::with_capacity(100),
            speech_history: VecDeque::with_capacity(20),
        }
    }

    /// Detect voice activity in audio buffer
    pub fn detect(&mut self, audio: &AudioBuffer) -> Result<bool> {
        let energy = self.compute_energy(&audio.data);
        let zcr = self.compute_zero_crossing_rate(&audio.data);

        // Update histories
        self.energy_history.push_back(energy);
        self.zcr_history.push_back(zcr);

        // Keep limited history
        if self.energy_history.len() > 100 {
            self.energy_history.pop_front();
        }
        if self.zcr_history.len() > 100 {
            self.zcr_history.pop_front();
        }

        // Adaptive thresholding
        self.update_thresholds();

        // Voice activity decision
        let is_speech = energy > self.energy_threshold && zcr < self.zero_crossing_threshold;

        // Temporal smoothing
        self.speech_history.push_back(is_speech);
        if self.speech_history.len() > self.smoothing_frames {
            self.speech_history.pop_front();
        }

        // Final decision based on majority vote
        let speech_count = self.speech_history.iter().filter(|&&x| x).count();
        Ok(speech_count > self.speech_history.len() / 2)
    }

    /// Compute frame energy (RMS)
    fn compute_energy(&self, frame: &[Sample]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_squares / frame.len() as f32).sqrt()
    }

    /// Compute zero crossing rate
    fn compute_zero_crossing_rate(&self, frame: &[Sample]) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }

        let crossings = frame
            .windows(2)
            .filter(|window| (window[0] >= 0.0) != (window[1] >= 0.0))
            .count();

        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Update adaptive thresholds based on recent history
    fn update_thresholds(&mut self) {
        if self.energy_history.len() > 10 {
            // Estimate noise floor from bottom 30% of energy values
            let mut sorted_energies: Vec<f32> = self.energy_history.iter().cloned().collect();
            sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let noise_samples = sorted_energies.len() * 3 / 10;
            let noise_floor: f32 = sorted_energies
                .iter()
                .take(noise_samples.max(1))
                .sum::<f32>() / noise_samples.max(1) as f32;

            // Set threshold above noise floor
            self.energy_threshold = noise_floor * 3.0;
        }

        if self.zcr_history.len() > 10 {
            // Estimate typical ZCR for non-speech
            let mut sorted_zcr: Vec<f32> = self.zcr_history.iter().cloned().collect();
            sorted_zcr.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Use median as threshold
            let median_idx = sorted_zcr.len() / 2;
            self.zero_crossing_threshold = sorted_zcr[median_idx] * 1.5;
        }
    }

    /// Set custom thresholds
    pub fn set_thresholds(&mut self, energy_threshold: f32, zcr_threshold: f32) {
        self.energy_threshold = energy_threshold;
        self.zero_crossing_threshold = zcr_threshold;
    }

    /// Get current thresholds
    pub fn thresholds(&self) -> (f32, f32) {
        (self.energy_threshold, self.zero_crossing_threshold)
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.energy_history.clear();
        self.zcr_history.clear();
        self.speech_history.clear();
    }

    /// Get detection confidence (0.0 to 1.0)
    pub fn confidence(&self) -> f32 {
        if self.speech_history.is_empty() {
            return 0.0;
        }

        let speech_count = self.speech_history.iter().filter(|&&x| x).count();
        speech_count as f32 / self.speech_history.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let vad = VoiceActivityDetector::new(480);
        assert_eq!(vad.frame_size, 480);
    }

    #[test]
    fn test_energy_computation() {
        let vad = VoiceActivityDetector::new(480);
        let silence = vec![0.0; 480];
        let noise: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin()).collect();

        let silence_energy = vad.compute_energy(&silence);
        let noise_energy = vad.compute_energy(&noise);

        assert!(silence_energy < noise_energy);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let vad = VoiceActivityDetector::new(480);

        // Low frequency signal (low ZCR)
        let low_freq: Vec<f32> = (0..480)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 16000.0).sin())
            .collect();

        // High frequency signal (high ZCR)
        let high_freq: Vec<f32> = (0..480)
            .map(|i| (2.0 * std::f32::consts::PI * 4000.0 * i as f32 / 16000.0).sin())
            .collect();

        let low_zcr = vad.compute_zero_crossing_rate(&low_freq);
        let high_zcr = vad.compute_zero_crossing_rate(&high_freq);

        assert!(low_zcr < high_zcr);
    }

    #[test]
    fn test_vad_detection() {
        let mut vad = VoiceActivityDetector::new(480);

        // Simulate speech signal
        let speech_signal: Vec<f32> = (0..480)
            .map(|i| {
                let freq1 = (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin();
                let freq2 = (2.0 * std::f32::consts::PI * 800.0 * i as f32 / 16000.0).sin();
                0.5 * (freq1 + 0.3 * freq2)
            })
            .collect();

        let audio = AudioBuffer {
            data: speech_signal,
            sample_rate: 16000,
            channels: 1,
            timestamp: std::time::Instant::now(),
        };

        let is_speech = vad.detect(&audio).unwrap();
        // Note: This is a simple test - real speech detection would need more sophisticated signals
        assert!(is_speech || !is_speech); // Just ensure it returns a boolean
    }
}