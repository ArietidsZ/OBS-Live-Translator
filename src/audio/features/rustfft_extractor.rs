//! RustFFT-based feature extractor for Low Profile
//!
//! This module implements efficient mel-spectrogram extraction using RustFFT:
//! - Optimized Rust-native FFT computation
//! - Basic triangular mel filterbank
//! - Memory-efficient streaming processing
//! - Target: 5% CPU, 10ms latency, 128-point mel features

use super::{ExtractorStats, FeatureConfig, FeatureExtractor, FeatureMetrics, FeatureResult};
use anyhow::Result;
use rustfft::{num_complex::Complex, FftPlanner};
use std::time::Instant;
use tracing::{debug, info};

/// RustFFT feature extractor for memory-efficient processing (Low Profile)
pub struct RustFFTExtractor {
    config: Option<FeatureConfig>,
    fft_planner: FftPlanner<f32>,
    mel_filterbank: Vec<Vec<f32>>,
    window: Vec<f32>,
    stats: ExtractorStats,
    frame_buffer: Vec<f32>,
}

impl RustFFTExtractor {
    /// Create a new RustFFT feature extractor
    pub fn new() -> Result<Self> {
        info!("🔧 Initializing RustFFT Feature Extractor (Low Profile)");

        Ok(Self {
            config: None,
            fft_planner: FftPlanner::new(),
            mel_filterbank: Vec::new(),
            window: Vec::new(),
            stats: ExtractorStats::default(),
            frame_buffer: Vec::new(),
        })
    }

    /// Initialize the extractor with configuration
    fn initialize_rustfft(&mut self, config: &FeatureConfig) -> Result<()> {
        // Create Hann window for signal windowing
        self.window = Self::create_hann_window(config.frame_size);

        // Create mel filterbank matrix
        self.mel_filterbank = self.create_mel_filterbank(config)?;

        // Pre-allocate frame buffer
        self.frame_buffer = vec![0.0; config.n_fft];

        info!(
            "📊 RustFFT initialized: FFT size={}, mel filters={}, window=Hann",
            config.n_fft, config.n_mels
        );

        Ok(())
    }

    /// Create Hann window for signal preprocessing
    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
                0.5 * (1.0 - angle.cos())
            })
            .collect()
    }

    /// Extract mel-spectrogram features using RustFFT
    fn extract_rustfft(&mut self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        let config = self.config.as_ref().unwrap();
        let mut spectrograms = Vec::new();

        let hop_length = config.hop_length;
        let frame_size = config.frame_size;

        // Process overlapping frames
        let mut start = 0;
        while start + frame_size <= audio.len() {
            let frame = &audio[start..start + frame_size];
            let mel_features = self.compute_mel_frame(frame)?;
            spectrograms.push(mel_features);
            start += hop_length;
        }

        Ok(spectrograms)
    }

    /// Compute mel-spectrogram for a single frame
    fn compute_mel_frame(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        // Apply windowing
        let windowed_frame: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&sample, &window_val)| sample * window_val)
            .collect();

        // Zero-pad to FFT size
        self.frame_buffer.fill(0.0);
        self.frame_buffer[..windowed_frame.len()].copy_from_slice(&windowed_frame);

        // Compute power spectrum
        let frame_buffer_copy = self.frame_buffer.clone();
        let power_spectrum = self.compute_power_spectrum(&frame_buffer_copy)?;

        // Apply mel filterbank
        let config = self.config.as_ref().unwrap();
        let mut mel_features = vec![0.0; config.n_mels];
        for (mel_idx, filter) in self.mel_filterbank.iter().enumerate() {
            let mut energy = 0.0;
            for (freq_idx, &weight) in filter.iter().enumerate() {
                if freq_idx < power_spectrum.len() && weight > 0.0 {
                    energy += weight * power_spectrum[freq_idx];
                }
            }
            // Apply log compression with small epsilon to avoid log(0)
            mel_features[mel_idx] = (energy + 1e-8).ln();
        }

        Ok(mel_features)
    }

    /// Compute power spectrum using RustFFT
    fn compute_power_spectrum(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        let fft = self.fft_planner.plan_fft_forward(frame.len());

        // Convert real signal to complex
        let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Compute FFT in-place
        fft.process(&mut buffer);

        // Compute power spectrum (magnitude squared)
        let power_spectrum: Vec<f32> = buffer
            .iter()
            .take(frame.len() / 2 + 1) // Only positive frequencies
            .map(|c| c.norm_sqr())
            .collect();

        Ok(power_spectrum)
    }

    /// Create triangular mel filterbank matrix
    fn create_mel_filterbank(&self, config: &FeatureConfig) -> Result<Vec<Vec<f32>>> {
        let n_fft = config.n_fft;
        let n_mels = config.n_mels;
        let sample_rate = config.sample_rate as f32;
        let f_min = config.f_min;
        let f_max = config.f_max;

        // Convert frequency range to mel scale
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        // Create equally spaced mel points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert mel points back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| Self::mel_to_hz(mel)).collect();

        // Convert Hz to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft + 1) as f32 * hz / sample_rate).round() as usize)
            .map(|bin| bin.min(n_fft / 2)) // Clamp to Nyquist
            .collect();

        // Create triangular filterbank
        let mut filterbank = vec![vec![0.0; n_fft / 2 + 1]; n_mels];

        for m in 0..n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            // Create triangular filter
            for k in left..=right.min(n_fft / 2) {
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

            // Normalize filter to unit area (optional for energy conservation)
            let filter_sum: f32 = filterbank[m].iter().sum();
            if filter_sum > 0.0 {
                for weight in &mut filterbank[m] {
                    *weight /= filter_sum;
                }
            }
        }

        debug!(
            "Created mel filterbank: {} filters, {} FFT bins, {:.1}-{:.1} Hz",
            n_mels,
            n_fft / 2 + 1,
            f_min,
            f_max
        );

        Ok(filterbank)
    }

    /// Convert Hz to mel scale (using natural log formula)
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Calculate expected output dimensions
    fn calculate_output_dims(&self, input_length: usize) -> (usize, usize) {
        let config = self.config.as_ref().unwrap();
        let n_frames = if input_length >= config.frame_size {
            (input_length - config.frame_size) / config.hop_length + 1
        } else {
            0
        };
        (n_frames, config.n_mels)
    }

    /// Estimate processing metrics for RustFFT
    fn estimate_metrics(&self, input_length: usize, processing_time_ms: f32) -> FeatureMetrics {
        let (n_frames, n_mels) = self.calculate_output_dims(input_length);
        let total_features = n_frames * n_mels;

        // RustFFT provides good accuracy with reasonable computational cost
        let computational_complexity = 2.0; // Lower complexity than advanced methods
        let memory_usage_mb = (input_length as f64 * 4.0 + 1024.0 * 1024.0) / (1024.0 * 1024.0);

        // Feature quality is good but not as high as specialized libraries
        let feature_quality = 0.75; // Good quality for Low Profile
        let spectral_resolution = 1.0; // Standard FFT resolution

        // Calculate efficiency
        let efficiency = if processing_time_ms > 0.0 {
            (total_features as f64) / (processing_time_ms as f64 / 1000.0)
        } else {
            0.0
        };

        FeatureMetrics {
            n_frames,
            n_features: n_mels,
            computational_complexity,
            memory_usage_mb,
            feature_quality,
            spectral_resolution,
            efficiency_features_per_sec: efficiency,
        }
    }

    /// Get RustFFT implementation details
    pub fn get_implementation_info(&self) -> RustFFTInfo {
        RustFFTInfo {
            library: "RustFFT".to_string(),
            version: "6.1.0".to_string(),
            algorithm: "Cooley-Tukey FFT".to_string(),
            window_function: "Hann".to_string(),
            filterbank_type: "Triangular Mel".to_string(),
            optimization_level: "Rust Native".to_string(),
        }
    }
}

/// RustFFT implementation information
#[derive(Debug, Clone)]
pub struct RustFFTInfo {
    pub library: String,
    pub version: String,
    pub algorithm: String,
    pub window_function: String,
    pub filterbank_type: String,
    pub optimization_level: String,
}

impl FeatureExtractor for RustFFTExtractor {
    fn initialize(&mut self, config: FeatureConfig) -> Result<()> {
        // Initialize RustFFT components
        self.initialize_rustfft(&config)?;

        self.config = Some(config);

        debug!("RustFFT extractor initialized for Low Profile");
        Ok(())
    }

    fn extract_features(&mut self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        if self.config.is_none() {
            return Err(anyhow::anyhow!("Feature extractor not initialized"));
        }

        self.extract_rustfft(audio)
    }

    fn extract_with_metrics(&mut self, audio: &[f32]) -> Result<FeatureResult> {
        let start_time = Instant::now();
        let features = self.extract_features(audio)?;
        let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        let config = self.config.as_ref().unwrap();
        let metrics = self.estimate_metrics(audio.len(), processing_time_ms);

        Ok(FeatureResult {
            features,
            sample_rate: config.sample_rate,
            frame_size: config.frame_size,
            hop_length: config.hop_length,
            n_mels: config.n_mels,
            processing_time_ms,
            metrics,
        })
    }

    fn get_feature_dimensions(&self) -> (usize, usize) {
        if let Some(config) = &self.config {
            (0, config.n_mels) // n_frames depends on input length
        } else {
            (0, 0)
        }
    }

    fn reset(&mut self) {
        self.stats = ExtractorStats::default();
        if let Some(config) = &self.config {
            self.frame_buffer = vec![0.0; config.n_fft];
        }
    }

    fn get_stats(&self) -> ExtractorStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rustfft_extractor_creation() {
        let extractor = RustFFTExtractor::new().unwrap();
        assert!(extractor.mel_filterbank.is_empty()); // Not initialized yet
    }

    #[test]
    fn test_hann_window_creation() {
        let window = RustFFTExtractor::create_hann_window(512);
        assert_eq!(window.len(), 512);
        assert!((window[0] - 0.0).abs() < 1e-6); // First sample should be ~0
        assert!((window[256] - 1.0).abs() < 0.1); // Middle should be ~1
    }

    #[test]
    fn test_mel_scale_conversion() {
        let hz = 1000.0;
        let mel = RustFFTExtractor::hz_to_mel(hz);
        let hz_back = RustFFTExtractor::mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 1.0); // Should roundtrip accurately
    }

    #[test]
    fn test_rustfft_initialization() {
        let mut extractor = RustFFTExtractor::new().unwrap();
        let config = FeatureConfig::default();
        extractor.initialize(config).unwrap();

        assert!(extractor.config.is_some());
        assert!(!extractor.mel_filterbank.is_empty());
        assert!(!extractor.window.is_empty());
    }

    #[test]
    fn test_implementation_info() {
        let extractor = RustFFTExtractor::new().unwrap();
        let info = extractor.get_implementation_info();
        assert_eq!(info.library, "RustFFT");
        assert_eq!(info.window_function, "Hann");
    }
}
