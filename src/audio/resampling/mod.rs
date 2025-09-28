//! Multi-tier audio resampling system for different performance profiles
//!
//! This module provides high-quality audio resampling implementations:
//! - Low Profile: SIMD-optimized linear interpolation
//! - Medium Profile: Cubic Hermite (Catmull-Rom) spline interpolation
//! - High Profile: libsoxr integration with VHQ preset
//! - Adaptive Pipeline: Automatic quality and algorithm selection

pub mod linear_resampler;
pub mod cubic_resampler;
pub mod soxr_resampler;
pub mod adaptive_resampler;

use crate::profile::Profile;
use anyhow::Result;
use std::time::Instant;
use tracing::{info, debug};

/// Resampling result with quality metrics
#[derive(Debug, Clone)]
pub struct ResamplingResult {
    /// Resampled audio data
    pub samples: Vec<f32>,
    /// Input sample rate
    pub input_sample_rate: u32,
    /// Output sample rate
    pub output_sample_rate: u32,
    /// Resampling ratio
    pub ratio: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for resampling assessment
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio estimate (dB)
    pub snr_db: f32,
    /// Total harmonic distortion estimate
    pub thd_percent: f32,
    /// Frequency response flatness (dB deviation)
    pub frequency_response_deviation_db: f32,
    /// Alias suppression level (dB)
    pub alias_suppression_db: f32,
    /// Processing efficiency (samples per second)
    pub efficiency_samples_per_sec: f64,
}

/// Configuration for audio resampling
#[derive(Debug, Clone)]
pub struct ResamplingConfig {
    /// Input sample rate
    pub input_sample_rate: u32,
    /// Output sample rate
    pub output_sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Quality level (0.0-1.0, higher = better quality)
    pub quality: f32,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Buffer size for streaming
    pub buffer_size: usize,
    /// Enable real-time constraints
    pub real_time_mode: bool,
}

impl Default for ResamplingConfig {
    fn default() -> Self {
        Self {
            input_sample_rate: 44100,
            output_sample_rate: 16000,
            channels: 1,
            quality: 0.8,
            enable_simd: true,
            buffer_size: 4096,
            real_time_mode: true,
        }
    }
}

/// Trait for audio resampling implementations
pub trait AudioResampler: Send + Sync {
    /// Initialize the resampler with configuration
    fn initialize(&mut self, config: ResamplingConfig) -> Result<()>;

    /// Resample audio data
    fn resample(&mut self, input: &[f32]) -> Result<Vec<f32>>;

    /// Resample with detailed result information
    fn resample_with_metrics(&mut self, input: &[f32]) -> Result<ResamplingResult>;

    /// Get the current resampling ratio
    fn ratio(&self) -> f64;

    /// Get quality metrics for the last operation
    fn get_quality_metrics(&self) -> QualityMetrics;

    /// Reset internal state
    fn reset(&mut self);

    /// Get processing statistics
    fn get_stats(&self) -> ResamplerStats;
}

/// Resampler processing statistics
#[derive(Debug, Clone, Default)]
pub struct ResamplerStats {
    pub total_samples_processed: u64,
    pub total_processing_time_ms: f64,
    pub average_processing_time_ms: f32,
    pub peak_processing_time_ms: f32,
    pub average_quality_score: f32,
    pub efficiency_samples_per_sec: f64,
}

/// Multi-tier resampling manager that selects appropriate resampler based on profile
pub struct ResamplingManager {
    profile: Profile,
    resampler: Box<dyn AudioResampler>,
    config: ResamplingConfig,
    stats: ResamplerStats,
}

impl ResamplingManager {
    /// Create a new resampling manager for the given profile
    pub fn new(profile: Profile, config: ResamplingConfig) -> Result<Self> {
        info!("🎵 Initializing resampling manager for profile {:?}", profile);

        let resampler: Box<dyn AudioResampler> = match profile {
            Profile::Low => {
                info!("📊 Creating Linear Resampler (Low Profile)");
                Box::new(linear_resampler::LinearResampler::new()?)
            }
            Profile::Medium => {
                info!("📊 Creating Cubic Hermite Resampler (Medium Profile)");
                Box::new(cubic_resampler::CubicResampler::new()?)
            }
            Profile::High => {
                info!("📊 Creating SoXR Resampler (High Profile)");
                Box::new(soxr_resampler::SoxrResampler::new()?)
            }
        };

        info!("✅ Resampling manager initialized for profile {:?}", profile);

        Ok(Self {
            profile,
            resampler,
            config,
            stats: ResamplerStats::default(),
        })
    }

    /// Initialize with the given configuration
    pub fn initialize(&mut self, config: ResamplingConfig) -> Result<()> {
        info!("🔧 Initializing resampler: {}Hz → {}Hz (ratio: {:.3})",
              config.input_sample_rate,
              config.output_sample_rate,
              config.output_sample_rate as f64 / config.input_sample_rate as f64);

        self.config = config.clone();
        self.resampler.initialize(config)?;

        Ok(())
    }

    /// Resample audio with the profile-specific implementation
    pub fn resample(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let start_time = Instant::now();
        let result = self.resampler.resample(input)?;
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        self.update_stats(input.len(), processing_time);

        debug!("Resampled {} → {} samples in {:.2}ms ({} profile)",
               input.len(), result.len(), processing_time, self.profile_name());

        Ok(result)
    }

    /// Resample with detailed metrics
    pub fn resample_with_metrics(&mut self, input: &[f32]) -> Result<ResamplingResult> {
        let start_time = Instant::now();
        let mut result = self.resampler.resample_with_metrics(input)?;
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update timing in result
        result.processing_time_ms = processing_time;

        // Update statistics
        self.update_stats(input.len(), processing_time);

        debug!("Resampled {} → {} samples in {:.2}ms, SNR: {:.1}dB, THD: {:.3}%",
               input.len(), result.samples.len(), processing_time,
               result.quality_metrics.snr_db, result.quality_metrics.thd_percent);

        Ok(result)
    }

    /// Get the current profile
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Get the current configuration
    pub fn config(&self) -> &ResamplingConfig {
        &self.config
    }

    /// Get resampling ratio
    pub fn ratio(&self) -> f64 {
        self.resampler.ratio()
    }

    /// Get processing statistics
    pub fn stats(&self) -> &ResamplerStats {
        &self.stats
    }

    /// Reset resampler state and statistics
    pub fn reset(&mut self) {
        self.resampler.reset();
        self.stats = ResamplerStats::default();
        info!("🔄 Resampling manager reset");
    }

    /// Check if resampling is meeting performance targets
    pub fn is_meeting_targets(&self) -> bool {
        let target_latency = match self.profile {
            Profile::Low => 5.0,    // 5ms target
            Profile::Medium => 8.0, // 8ms target
            Profile::High => 6.0,   // 6ms target
        };

        self.stats.average_processing_time_ms <= target_latency
    }

    /// Get quality assessment for current settings
    pub fn assess_quality(&self) -> QualityAssessment {
        let metrics = self.resampler.get_quality_metrics();

        let overall_score = match self.profile {
            Profile::Low => {
                // Focus on efficiency for low profile
                if metrics.efficiency_samples_per_sec > 1_000_000.0 { 0.8 } else { 0.6 }
            }
            Profile::Medium => {
                // Balanced quality/performance
                let snr_score = (metrics.snr_db / 60.0).min(1.0);
                let thd_score = (1.0 - metrics.thd_percent / 5.0).max(0.0);
                (snr_score + thd_score) / 2.0
            }
            Profile::High => {
                // Maximum quality focus
                let snr_score = (metrics.snr_db / 120.0).min(1.0);
                let alias_score = (metrics.alias_suppression_db / 100.0).min(1.0);
                let freq_score = (1.0 - metrics.frequency_response_deviation_db / 3.0).max(0.0);
                (snr_score + alias_score + freq_score) / 3.0
            }
        };

        QualityAssessment {
            overall_score,
            meets_profile_requirements: overall_score >= 0.7,
            metrics,
        }
    }

    /// Update internal statistics
    fn update_stats(&mut self, input_samples: usize, processing_time_ms: f32) {
        let samples_processed = input_samples as u64;

        // Update totals
        self.stats.total_samples_processed += samples_processed;
        self.stats.total_processing_time_ms += processing_time_ms as f64;

        // Update averages
        let total_operations = if self.stats.total_samples_processed > 0 {
            self.stats.total_processing_time_ms / processing_time_ms as f64
        } else {
            1.0
        };

        self.stats.average_processing_time_ms =
            (self.stats.total_processing_time_ms / total_operations) as f32;

        // Update peak
        if processing_time_ms > self.stats.peak_processing_time_ms {
            self.stats.peak_processing_time_ms = processing_time_ms;
        }

        // Update efficiency
        if processing_time_ms > 0.0 {
            self.stats.efficiency_samples_per_sec =
                (samples_processed as f64) / (processing_time_ms as f64 / 1000.0);
        }
    }

    /// Get profile name as string
    fn profile_name(&self) -> &'static str {
        match self.profile {
            Profile::Low => "Low",
            Profile::Medium => "Medium",
            Profile::High => "High",
        }
    }
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall quality score (0.0-1.0)
    pub overall_score: f32,
    /// Whether quality meets profile requirements
    pub meets_profile_requirements: bool,
    /// Detailed quality metrics
    pub metrics: QualityMetrics,
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
    fn test_resampling_config_default() {
        let config = ResamplingConfig::default();
        assert_eq!(config.input_sample_rate, 44100);
        assert_eq!(config.output_sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert!(config.enable_simd);
    }

    #[test]
    fn test_quality_metrics_default() {
        let metrics = QualityMetrics::default();
        assert_eq!(metrics.snr_db, 0.0);
        assert_eq!(metrics.thd_percent, 0.0);
    }

    #[test]
    fn test_quality_assessment_grade() {
        let assessment = QualityAssessment {
            overall_score: 0.95,
            meets_profile_requirements: true,
            metrics: QualityMetrics::default(),
        };
        assert_eq!(assessment.grade(), "Excellent");

        let poor_assessment = QualityAssessment {
            overall_score: 0.5,
            meets_profile_requirements: false,
            metrics: QualityMetrics::default(),
        };
        assert_eq!(poor_assessment.grade(), "Poor");
    }
}