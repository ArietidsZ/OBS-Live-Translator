//! Multi-tier feature extraction system for mel-spectrograms
//!
//! This module provides profile-aware feature extraction implementations:
//! - Low Profile: RustFFT + basic mel filterbank for memory efficiency
//! - Medium Profile: Enhanced RustFFT pipeline with advanced windowing
//! - High Profile: Intel IPP + SIMD processing for maximum performance
//! - Adaptive Pipeline: Automatic algorithm selection and quality optimization

pub mod rustfft_extractor;
pub mod enhanced_extractor;
pub mod ipp_extractor;
pub mod adaptive_extractor;

use crate::profile::Profile;
use anyhow::Result;
use std::time::Instant;
use tracing::{info, debug};

/// Feature extraction result with quality metrics
#[derive(Debug, Clone)]
pub struct FeatureResult {
    /// Extracted mel-spectrogram features [frames x mels]
    pub features: Vec<Vec<f32>>,
    /// Input sample rate
    pub sample_rate: u32,
    /// Frame size used for extraction
    pub frame_size: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Number of mel filters
    pub n_mels: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Quality and performance metrics
    pub metrics: FeatureMetrics,
}

/// Feature extraction quality and performance metrics
#[derive(Debug, Clone, Default)]
pub struct FeatureMetrics {
    /// Number of frames extracted
    pub n_frames: usize,
    /// Number of features per frame
    pub n_features: usize,
    /// Computational complexity indicator (1.0 = baseline)
    pub computational_complexity: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Feature quality score (0.0-1.0)
    pub feature_quality: f32,
    /// Spectral resolution indicator (1.0 = standard FFT)
    pub spectral_resolution: f32,
    /// Processing efficiency (features per second)
    pub efficiency_features_per_sec: f64,
}

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Audio sample rate
    pub sample_rate: u32,
    /// Frame size for analysis (samples)
    pub frame_size: usize,
    /// Hop length between frames (samples)
    pub hop_length: usize,
    /// FFT size (usually same as frame_size or zero-padded)
    pub n_fft: usize,
    /// Number of mel filter banks
    pub n_mels: usize,
    /// Minimum frequency for mel scale (Hz)
    pub f_min: f32,
    /// Maximum frequency for mel scale (Hz)
    pub f_max: f32,
    /// Enable advanced processing features
    pub enable_advanced: bool,
    /// Real-time processing constraints
    pub real_time_mode: bool,
    /// Quality level (0.0-1.0)
    pub quality: f32,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 512,      // 32ms at 16kHz
            hop_length: 160,      // 10ms hop (75% overlap)
            n_fft: 512,
            n_mels: 80,           // Standard mel features
            f_min: 80.0,          // Low speech frequencies
            f_max: 8000.0,        // Nyquist for 16kHz
            enable_advanced: true,
            real_time_mode: true,
            quality: 0.8,
        }
    }
}

/// Trait for feature extraction implementations
pub trait FeatureExtractor: Send + Sync {
    /// Initialize the feature extractor with configuration
    fn initialize(&mut self, config: FeatureConfig) -> Result<()>;

    /// Extract mel-spectrogram features from audio
    fn extract_features(&mut self, audio: &[f32]) -> Result<Vec<Vec<f32>>>;

    /// Extract features with detailed metrics
    fn extract_with_metrics(&mut self, audio: &[f32]) -> Result<FeatureResult>;

    /// Get expected output dimensions (frames, features)
    fn get_feature_dimensions(&self) -> (usize, usize);

    /// Reset internal state
    fn reset(&mut self);

    /// Get processing statistics
    fn get_stats(&self) -> ExtractorStats;
}

/// Feature extractor processing statistics
#[derive(Debug, Clone, Default)]
pub struct ExtractorStats {
    pub total_frames_processed: u64,
    pub total_processing_time_ms: f64,
    pub average_processing_time_ms: f32,
    pub peak_processing_time_ms: f32,
    pub average_quality_score: f32,
    pub efficiency_features_per_sec: f64,
}

/// Multi-tier feature extraction manager
pub struct FeatureExtractionManager {
    profile: Profile,
    extractor: Box<dyn FeatureExtractor>,
    config: FeatureConfig,
    stats: ExtractorStats,
}

impl FeatureExtractionManager {
    /// Create a new feature extraction manager for the given profile
    pub fn new(profile: Profile, config: FeatureConfig) -> Result<Self> {
        info!("🎵 Initializing feature extraction manager for profile {:?}", profile);

        let extractor: Box<dyn FeatureExtractor> = match profile {
            Profile::Low => {
                info!("📊 Creating RustFFT Extractor (Low Profile)");
                Box::new(rustfft_extractor::RustFFTExtractor::new()?)
            }
            Profile::Medium => {
                info!("📊 Creating Enhanced RustFFT Extractor (Medium Profile)");
                Box::new(enhanced_extractor::EnhancedExtractor::new()?)
            }
            Profile::High => {
                info!("📊 Creating Intel IPP Extractor (High Profile)");
                Box::new(ipp_extractor::IppExtractor::new()?)
            }
        };

        info!("✅ Feature extraction manager initialized for profile {:?}", profile);

        Ok(Self {
            profile,
            extractor,
            config,
            stats: ExtractorStats::default(),
        })
    }

    /// Initialize with the given configuration
    pub fn initialize(&mut self, config: FeatureConfig) -> Result<()> {
        info!("🔧 Initializing feature extractor: {}Hz, {}ms frames, {} mel filters",
              config.sample_rate,
              (config.frame_size as f32 / config.sample_rate as f32 * 1000.0) as u32,
              config.n_mels);

        self.config = config.clone();
        self.extractor.initialize(config)?;

        Ok(())
    }

    /// Extract mel-spectrogram features with the profile-specific implementation
    pub fn extract_features(&mut self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        let start_time = Instant::now();
        let result = self.extractor.extract_features(audio)?;
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        self.update_stats(result.len(), processing_time);

        debug!("Extracted features: {} frames x {} mels in {:.2}ms ({} profile)",
               result.len(),
               if result.is_empty() { 0 } else { result[0].len() },
               processing_time,
               self.profile_name());

        Ok(result)
    }

    /// Extract features with detailed metrics
    pub fn extract_with_metrics(&mut self, audio: &[f32]) -> Result<FeatureResult> {
        let start_time = Instant::now();
        let mut result = self.extractor.extract_with_metrics(audio)?;
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update timing in result
        result.processing_time_ms = processing_time;

        // Update statistics
        self.update_stats(result.features.len(), processing_time);

        debug!("Extracted {} frames x {} mels in {:.2}ms, quality: {:.3}, complexity: {:.1}x",
               result.features.len(), result.n_mels, processing_time,
               result.metrics.feature_quality, result.metrics.computational_complexity);

        Ok(result)
    }

    /// Get the current profile
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Get the current configuration
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }

    /// Get expected output dimensions
    pub fn get_dimensions(&self) -> (usize, usize) {
        self.extractor.get_feature_dimensions()
    }

    /// Get processing statistics
    pub fn stats(&self) -> &ExtractorStats {
        &self.stats
    }

    /// Reset extractor state and statistics
    pub fn reset(&mut self) {
        self.extractor.reset();
        self.stats = ExtractorStats::default();
        info!("🔄 Feature extraction manager reset");
    }

    /// Check if extraction is meeting performance targets
    pub fn is_meeting_targets(&self) -> bool {
        let target_latency = match self.profile {
            Profile::Low => 10.0,   // 10ms target
            Profile::Medium => 12.0, // 12ms target
            Profile::High => 8.0,   // 8ms target
        };

        self.stats.average_processing_time_ms <= target_latency
    }

    /// Get quality assessment for current settings
    pub fn assess_quality(&self) -> QualityAssessment {
        let stats = &self.stats;

        let performance_score = match self.profile {
            Profile::Low => {
                // Focus on efficiency and memory usage
                if stats.efficiency_features_per_sec > 100_000.0 { 0.8 } else { 0.6 }
            }
            Profile::Medium => {
                // Balanced quality and performance
                let timing_score = if stats.average_processing_time_ms <= 12.0 { 0.8 } else { 0.5 };
                let efficiency_score = if stats.efficiency_features_per_sec > 50_000.0 { 0.8 } else { 0.6 };
                (timing_score + efficiency_score) / 2.0
            }
            Profile::High => {
                // Maximum quality focus
                let timing_score = if stats.average_processing_time_ms <= 8.0 { 0.9 } else { 0.7 };
                let quality_score = stats.average_quality_score;
                (timing_score + quality_score) / 2.0
            }
        };

        QualityAssessment {
            overall_score: performance_score,
            meets_profile_requirements: performance_score >= 0.7,
            profile: self.profile,
            latency_ms: stats.average_processing_time_ms,
            efficiency: stats.efficiency_features_per_sec,
        }
    }

    /// Update internal statistics
    fn update_stats(&mut self, n_frames: usize, processing_time_ms: f32) {
        let frames_processed = n_frames as u64;

        // Update totals
        self.stats.total_frames_processed += frames_processed;
        self.stats.total_processing_time_ms += processing_time_ms as f64;

        // Update averages
        let total_operations = if self.stats.total_frames_processed > 0 {
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
            let features_extracted = n_frames * self.config.n_mels;
            self.stats.efficiency_features_per_sec =
                (features_extracted as f64) / (processing_time_ms as f64 / 1000.0);
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

/// Quality assessment result for feature extraction
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall performance score (0.0-1.0)
    pub overall_score: f32,
    /// Whether quality meets profile requirements
    pub meets_profile_requirements: bool,
    /// Profile being assessed
    pub profile: Profile,
    /// Average processing latency (ms)
    pub latency_ms: f32,
    /// Processing efficiency (features/sec)
    pub efficiency: f64,
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
    fn test_feature_config_default() {
        let config = FeatureConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.frame_size, 512);
        assert!(config.enable_advanced);
    }

    #[test]
    fn test_feature_metrics_default() {
        let metrics = FeatureMetrics::default();
        assert_eq!(metrics.n_frames, 0);
        assert_eq!(metrics.computational_complexity, 0.0);
    }

    #[test]
    fn test_quality_assessment_grade() {
        let assessment = QualityAssessment {
            overall_score: 0.95,
            meets_profile_requirements: true,
            profile: Profile::High,
            latency_ms: 5.0,
            efficiency: 100_000.0,
        };
        assert_eq!(assessment.grade(), "Excellent");

        let poor_assessment = QualityAssessment {
            overall_score: 0.5,
            meets_profile_requirements: false,
            profile: Profile::Low,
            latency_ms: 20.0,
            efficiency: 10_000.0,
        };
        assert_eq!(poor_assessment.grade(), "Poor");
    }
}