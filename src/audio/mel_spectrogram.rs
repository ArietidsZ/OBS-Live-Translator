//! Mel spectrogram feature extraction for ASR models

use crate::Result;
use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Mel spectrogram extractor
#[allow(dead_code)]
pub struct MelSpectrogramExtractor {
    sample_rate: u32,
    window_size: usize,
    hop_length: usize,
    n_mels: usize,
    mel_filters: Array2<f32>,
}

impl MelSpectrogramExtractor {
    /// Create a new mel spectrogram extractor
    pub fn new(
        sample_rate: u32,
        window_size: usize,
        hop_length: usize,
        n_mels: usize,
    ) -> Result<Self> {
        let mel_filters = Self::create_mel_filterbank(sample_rate, window_size, n_mels);

        Ok(Self {
            sample_rate,
            window_size,
            hop_length,
            n_mels,
            mel_filters,
        })
    }

    /// Extract mel spectrogram from audio samples
    pub fn extract(&self, samples: &[f32]) -> Result<Array3<f32>> {
        // 1. Apply windowing and compute STFT
        let stft = self.compute_stft(samples)?;

        // 2. Compute power spectrogram
        let power_spec = self.power_spectrogram(&stft);

        // 3. Apply mel filterbank
        let mel_spec = self.apply_mel_filters(&power_spec);

        // 4. Convert to log scale
        let log_mel = self.to_log_scale(&mel_spec);

        // 5. Reshape to (batch, n_mels, time)
        let n_frames = log_mel.ncols();
        let output =
            Array3::from_shape_fn((1, self.n_mels, n_frames), |(_, mel_idx, frame_idx)| {
                log_mel[[mel_idx, frame_idx]]
            });

        Ok(output)
    }

    /// Compute Short-Time Fourier Transform
    fn compute_stft(&self, samples: &[f32]) -> Result<Vec<Vec<Complex<f32>>>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.window_size);

        let window = self.hann_window(self.window_size);
        let n_frames = (samples.len() - self.window_size) / self.hop_length + 1;

        let mut stft = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.window_size).min(samples.len());

            // Apply window and zero-pad if necessary
            let mut buffer: Vec<Complex<f32>> = (0..self.window_size)
                .map(|i| {
                    let sample = if start + i < end {
                        samples[start + i] * window[i]
                    } else {
                        0.0
                    };
                    Complex::new(sample, 0.0)
                })
                .collect();

            // Compute FFT
            fft.process(&mut buffer);

            stft.push(buffer);
        }

        Ok(stft)
    }

    /// Compute power spectrogram from STFT
    fn power_spectrogram(&self, stft: &[Vec<Complex<f32>>]) -> Array2<f32> {
        let n_frames = stft.len();
        let n_freqs = self.window_size / 2 + 1;

        Array2::from_shape_fn((n_freqs, n_frames), |(freq_idx, frame_idx)| {
            let c = stft[frame_idx][freq_idx];
            c.norm_sqr()
        })
    }

    /// Apply mel filterbank to power spectrogram
    fn apply_mel_filters(&self, power_spec: &Array2<f32>) -> Array2<f32> {
        // mel_filters: (n_mels, n_freqs)
        // power_spec: (n_freqs, n_frames)
        // output: (n_mels, n_frames)

        let n_frames = power_spec.ncols();
        let mut mel_spec = Array2::zeros((self.n_mels, n_frames));

        for mel_idx in 0..self.n_mels {
            for frame_idx in 0..n_frames {
                let mut sum = 0.0;
                for freq_idx in 0..power_spec.nrows() {
                    sum +=
                        self.mel_filters[[mel_idx, freq_idx]] * power_spec[[freq_idx, frame_idx]];
                }
                mel_spec[[mel_idx, frame_idx]] = sum;
            }
        }

        mel_spec
    }

    /// Convert to log scale
    fn to_log_scale(&self, mel_spec: &Array2<f32>) -> Array2<f32> {
        mel_spec.mapv(|x| (x.max(1e-10)).log10() * 10.0)
    }

    /// Create Hann window
    fn hann_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
            .collect()
    }

    /// Create mel filterbank
    fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Array2<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filterbank = Array2::zeros((n_mels, n_freqs));

        // Convert frequency to mel scale
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let fmin = 0.0;
        let fmax = sample_rate as f32 / 2.0;
        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);

        // Create mel points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert mel points to frequencies
        let freq_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Convert frequencies to FFT bin numbers
        let fft_bins: Vec<usize> = freq_points
            .iter()
            .map(|&f| ((n_fft + 1) as f32 * f / sample_rate as f32).floor() as usize)
            .collect();

        // Create triangular filters
        for mel_idx in 0..n_mels {
            let left = fft_bins[mel_idx];
            let center = fft_bins[mel_idx + 1];
            let right = fft_bins[mel_idx + 2];

            // Rising slope
            for freq_idx in left..center {
                let weight = (freq_idx - left) as f32 / (center - left) as f32;
                filterbank[[mel_idx, freq_idx]] = weight;
            }

            // Falling slope
            for freq_idx in center..right {
                let weight = (right - freq_idx) as f32 / (right - center) as f32;
                filterbank[[mel_idx, freq_idx]] = weight;
            }
        }

        filterbank
    }
}

/// Default mel spectrogram configuration for Whisper models
pub fn whisper_mel_extractor() -> Result<MelSpectrogramExtractor> {
    MelSpectrogramExtractor::new(
        16000, // sample_rate
        400,   // n_fft (25ms window at 16kHz)
        160,   // hop_length (10ms hop at 16kHz)
        80,    // n_mels (Whisper uses 80 mel bins)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_extractor_creation() {
        let extractor = whisper_mel_extractor().unwrap();
        assert_eq!(extractor.n_mels, 80);
        assert_eq!(extractor.sample_rate, 16000);
    }

    #[test]
    fn test_mel_extraction() {
        let extractor = whisper_mel_extractor().unwrap();
        let samples = vec![0.0f32; 16000]; // 1 second of silence
        let mel_spec = extractor.extract(&samples).unwrap();

        assert_eq!(mel_spec.shape(), &[1, 80, 98]); // 98 frames for 1 second with current window/hop
    }
}
