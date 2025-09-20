//! Feature extraction including FFT and mel-spectrogram computation

use super::{AudioConfig, AudioBuffer, Sample};
use anyhow::{Result, anyhow};
use rustfft::{FftPlanner, num_complex::Complex};

/// Feature extractor for mel-spectrograms
pub struct FeatureExtractor {
    config: AudioConfig,
    fft_planner: FftPlanner<f32>,
    mel_filterbank: Vec<Vec<f32>>,
}

impl FeatureExtractor {
    pub fn new(config: AudioConfig) -> Result<Self> {
        let mut extractor = Self {
            config: config.clone(),
            fft_planner: FftPlanner::new(),
            mel_filterbank: Vec::new(),
        };

        extractor.mel_filterbank = extractor.create_mel_filterbank()?;
        Ok(extractor)
    }

    /// Extract mel-spectrogram from audio buffer
    pub fn extract_mel_spectrogram(&mut self, audio: &AudioBuffer) -> Result<Vec<Vec<f32>>> {
        if audio.data.len() < self.config.frame_size {
            return Err(anyhow!("Audio buffer too short for feature extraction"));
        }

        let mut spectrograms = Vec::new();
        let hop_length = self.config.hop_length;
        let frame_size = self.config.frame_size;

        // Process overlapping frames
        let mut start = 0;
        while start + frame_size <= audio.data.len() {
            let frame = &audio.data[start..start + frame_size];
            let mel_spectrum = self.compute_mel_spectrum(frame)?;
            spectrograms.push(mel_spectrum);
            start += hop_length;
        }

        Ok(spectrograms)
    }

    /// Compute mel spectrum for a single frame
    fn compute_mel_spectrum(&mut self, frame: &[Sample]) -> Result<Vec<f32>> {
        // Pad frame to FFT size
        let mut padded_frame = frame.to_vec();
        padded_frame.resize(self.config.n_fft, 0.0);

        // Compute FFT
        let power_spectrum = self.compute_power_spectrum(&padded_frame)?;

        // Apply mel filterbank
        let mut mel_spectrum = vec![0.0; self.config.n_mels];
        for (mel_idx, filter) in self.mel_filterbank.iter().enumerate() {
            let mut energy = 0.0;
            for (freq_idx, &weight) in filter.iter().enumerate() {
                if freq_idx < power_spectrum.len() {
                    energy += weight * power_spectrum[freq_idx];
                }
            }
            mel_spectrum[mel_idx] = energy.max(1e-10).ln(); // Log mel energy
        }

        Ok(mel_spectrum)
    }

    /// Compute power spectrum using FFT
    fn compute_power_spectrum(&mut self, frame: &[Sample]) -> Result<Vec<f32>> {
        let fft = self.fft_planner.plan_fft_forward(frame.len());

        // Convert to complex
        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Compute FFT
        fft.process(&mut buffer);

        // Compute power spectrum (magnitude squared)
        let power_spectrum: Vec<f32> = buffer
            .iter()
            .take(frame.len() / 2 + 1) // Only positive frequencies
            .map(|c| c.norm_sqr())
            .collect();

        Ok(power_spectrum)
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank(&self) -> Result<Vec<Vec<f32>>> {
        let n_fft = self.config.n_fft;
        let n_mels = self.config.n_mels;
        let sample_rate = self.config.sample_rate as f32;
        let f_min = self.config.f_min;
        let f_max = self.config.f_max;

        // Convert frequencies to mel scale
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        // Create mel points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points
            .iter()
            .map(|&mel| Self::mel_to_hz(mel))
            .collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft + 1) as f32 * hz / sample_rate).floor() as usize)
            .collect();

        // Create filterbank
        let mut filterbank = vec![vec![0.0; n_fft / 2 + 1]; n_mels];

        for m in 0..n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for k in left..=right {
                if k < filterbank[m].len() {
                    if k <= center {
                        // Rising edge
                        if center > left {
                            filterbank[m][k] = (k - left) as f32 / (center - left) as f32;
                        }
                    } else {
                        // Falling edge
                        if right > center {
                            filterbank[m][k] = (right - k) as f32 / (right - center) as f32;
                        }
                    }
                }
            }
        }

        Ok(filterbank)
    }

    /// Convert Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Get configuration
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_scale_conversion() {
        let hz = 1000.0;
        let mel = FeatureExtractor::hz_to_mel(hz);
        let hz_back = FeatureExtractor::mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 1e-3);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = AudioConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.is_ok());
    }

    #[test]
    fn test_mel_spectrogram_extraction() {
        let config = AudioConfig::default();
        let mut extractor = FeatureExtractor::new(config).unwrap();

        // Create test audio
        let samples: Vec<f32> = (0..1600)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let audio = AudioBuffer {
            data: samples,
            sample_rate: 16000,
            channels: 1,
            timestamp: std::time::Instant::now(),
        };

        let mel_spec = extractor.extract_mel_spectrogram(&audio).unwrap();
        assert!(!mel_spec.is_empty());
        assert_eq!(mel_spec[0].len(), 80); // n_mels
    }
}