//! Adaptive feature extraction pipeline
//!
//! This module implements intelligent feature extraction with automatic optimization:
//! - Quality monitoring and algorithm switching
//! - Performance profiling and adaptation
//! - Streaming processing with dynamic parameters
//! - Content-aware feature selection

use super::{FeatureExtractor, FeatureConfig, FeatureResult, FeatureMetrics, ExtractorStats};
use super::{rustfft_extractor::RustFFTExtractor, enhanced_extractor::EnhancedExtractor};
use anyhow::Result;
use std::time::Instant;
use tracing::{info, debug};

/// Adaptive feature extraction algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtractionAlgorithm {
    RustFFT,
    Enhanced,
    IPP,
}

/// Adaptive feature extractor with intelligent algorithm selection
pub struct AdaptiveExtractor {
    current_algorithm: ExtractionAlgorithm,
    extractors: AdaptiveExtractors,
    config: Option<FeatureConfig>,
    stats: ExtractorStats,

    // Adaptation parameters
    quality_monitor: QualityMonitor,
    performance_tracker: PerformanceTracker,
    adaptation_enabled: bool,
}

/// Container for different extractor implementations
struct AdaptiveExtractors {
    rustfft: Option<RustFFTExtractor>,
    enhanced: Option<EnhancedExtractor>,
}

/// Quality monitoring for adaptive behavior
#[derive(Debug, Clone)]
struct QualityMonitor {
    recent_quality_scores: Vec<f32>,
    quality_threshold: f32,
    quality_trend: QualityTrend,
}

/// Performance tracking for adaptation decisions
#[derive(Debug, Clone)]
struct PerformanceTracker {
    recent_latencies: Vec<f32>,
    latency_threshold: f32,
    target_efficiency: f64,
}

/// Quality trend analysis
#[derive(Debug, Clone)]
struct QualityTrend {
    average_quality: f32,
    quality_variance: f32,
    improving: bool,
}

impl AdaptiveExtractor {
    /// Create a new adaptive feature extractor
    pub fn new() -> Result<Self> {
        info!("🧠 Initializing Adaptive Feature Extractor");

        Ok(Self {
            current_algorithm: ExtractionAlgorithm::RustFFT,
            extractors: AdaptiveExtractors {
                rustfft: Some(RustFFTExtractor::new()?),
                enhanced: Some(EnhancedExtractor::new()?),
            },
            config: None,
            stats: ExtractorStats::default(),
            quality_monitor: QualityMonitor {
                recent_quality_scores: Vec::new(),
                quality_threshold: 0.75,
                quality_trend: QualityTrend {
                    average_quality: 0.0,
                    quality_variance: 0.0,
                    improving: true,
                },
            },
            performance_tracker: PerformanceTracker {
                recent_latencies: Vec::new(),
                latency_threshold: 15.0, // 15ms threshold
                target_efficiency: 50_000.0,
            },
            adaptation_enabled: true,
        })
    }

    /// Initialize adaptive extractor with configuration
    fn initialize_adaptive(&mut self, config: &FeatureConfig) -> Result<()> {
        // Initialize all available extractors
        if let Some(ref mut rustfft) = self.extractors.rustfft {
            rustfft.initialize(config.clone())?;
        }

        if let Some(ref mut enhanced) = self.extractors.enhanced {
            enhanced.initialize(config.clone())?;
        }

        // Select initial algorithm based on quality requirements
        self.current_algorithm = self.select_initial_algorithm(config);

        info!("🎯 Adaptive extractor initialized with algorithm: {:?}", self.current_algorithm);
        Ok(())
    }

    /// Select initial algorithm based on configuration
    fn select_initial_algorithm(&self, config: &FeatureConfig) -> ExtractionAlgorithm {
        if config.quality > 0.8 && config.enable_advanced {
            ExtractionAlgorithm::Enhanced
        } else {
            ExtractionAlgorithm::RustFFT
        }
    }

    /// Extract features with adaptive algorithm selection
    fn extract_adaptive(&mut self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        let start_time = Instant::now();

        // Check if algorithm switching is needed
        if self.adaptation_enabled {
            self.consider_algorithm_switch()?;
        }

        // Extract features with current algorithm
        let features = match self.current_algorithm {
            ExtractionAlgorithm::RustFFT => {
                if let Some(ref mut extractor) = self.extractors.rustfft {
                    extractor.extract_features(audio)?
                } else {
                    return Err(anyhow::anyhow!("RustFFT extractor not available"));
                }
            }
            ExtractionAlgorithm::Enhanced => {
                if let Some(ref mut extractor) = self.extractors.enhanced {
                    extractor.extract_features(audio)?
                } else {
                    return Err(anyhow::anyhow!("Enhanced extractor not available"));
                }
            }
            ExtractionAlgorithm::IPP => {
                // Fallback to Enhanced for now
                if let Some(ref mut extractor) = self.extractors.enhanced {
                    extractor.extract_features(audio)?
                } else {
                    return Err(anyhow::anyhow!("IPP extractor not available"));
                }
            }
        };

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update performance tracking
        self.update_performance_tracking(processing_time, &features);

        Ok(features)
    }

    /// Consider switching algorithms based on performance
    fn consider_algorithm_switch(&mut self) -> Result<()> {
        if self.performance_tracker.recent_latencies.len() < 5 {
            return Ok(()) // Not enough data
        }

        let avg_latency: f32 = self.performance_tracker.recent_latencies.iter().sum::<f32>()
            / self.performance_tracker.recent_latencies.len() as f32;

        let quality_declining = self.quality_monitor.quality_trend.average_quality
            < self.quality_monitor.quality_threshold;

        let latency_exceeded = avg_latency > self.performance_tracker.latency_threshold;

        // Algorithm switching logic
        match self.current_algorithm {
            ExtractionAlgorithm::RustFFT => {
                // Switch to Enhanced if quality is needed and latency allows
                if quality_declining && avg_latency < 10.0 {
                    self.switch_algorithm(ExtractionAlgorithm::Enhanced)?;
                }
            }
            ExtractionAlgorithm::Enhanced => {
                // Switch to RustFFT if latency is too high
                if latency_exceeded {
                    self.switch_algorithm(ExtractionAlgorithm::RustFFT)?;
                }
            }
            ExtractionAlgorithm::IPP => {
                // IPP should have best performance, fallback if issues
                if latency_exceeded {
                    self.switch_algorithm(ExtractionAlgorithm::Enhanced)?;
                }
            }
        }

        Ok(())
    }

    /// Switch to a different algorithm
    fn switch_algorithm(&mut self, new_algorithm: ExtractionAlgorithm) -> Result<()> {
        if new_algorithm != self.current_algorithm {
            info!("🔄 Switching feature extraction algorithm: {:?} → {:?}",
                  self.current_algorithm, new_algorithm);

            self.current_algorithm = new_algorithm;

            // Reset performance tracking for new algorithm
            self.performance_tracker.recent_latencies.clear();
            self.quality_monitor.recent_quality_scores.clear();
        }

        Ok(())
    }

    /// Update performance tracking with latest results
    fn update_performance_tracking(&mut self, processing_time: f32, features: &[Vec<f32>]) {
        // Track latency
        self.performance_tracker.recent_latencies.push(processing_time);
        if self.performance_tracker.recent_latencies.len() > 10 {
            self.performance_tracker.recent_latencies.remove(0);
        }

        // Estimate quality score based on feature consistency
        let quality_score = self.estimate_feature_quality(features);
        self.quality_monitor.recent_quality_scores.push(quality_score);
        if self.quality_monitor.recent_quality_scores.len() > 10 {
            self.quality_monitor.recent_quality_scores.remove(0);
        }

        // Update quality trend
        self.update_quality_trend();
    }

    /// Estimate feature quality based on consistency and range
    fn estimate_feature_quality(&self, features: &[Vec<f32>]) -> f32 {
        if features.is_empty() {
            return 0.0;
        }

        let _n_frames = features.len();
        let n_features = features[0].len();

        // Calculate feature variance as quality indicator
        let mut total_variance = 0.0;
        for feat_idx in 0..n_features {
            let values: Vec<f32> = features.iter().map(|frame| frame[feat_idx]).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            total_variance += variance;
        }

        let avg_variance = total_variance / n_features as f32;

        // Quality score: higher variance indicates more dynamic features (better)
        // but extremely high variance might indicate noise
        let quality = if avg_variance > 0.1 && avg_variance < 10.0 {
            0.8 + (avg_variance / 10.0).min(0.2)
        } else if avg_variance <= 0.1 {
            0.5 // Low dynamics
        } else {
            0.6 // Possibly noisy
        };

        quality.min(1.0)
    }

    /// Update quality trend analysis
    fn update_quality_trend(&mut self) {
        if self.quality_monitor.recent_quality_scores.len() < 3 {
            return;
        }

        let scores = &self.quality_monitor.recent_quality_scores;
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;

        let variance = scores.iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;

        // Check if trend is improving (recent scores higher than older ones)
        let recent_half = &scores[scores.len()/2..];
        let older_half = &scores[..scores.len()/2];

        let recent_mean = recent_half.iter().sum::<f32>() / recent_half.len() as f32;
        let older_mean = older_half.iter().sum::<f32>() / older_half.len() as f32;

        self.quality_monitor.quality_trend = QualityTrend {
            average_quality: mean,
            quality_variance: variance,
            improving: recent_mean > older_mean,
        };
    }

    /// Get current algorithm information
    pub fn get_algorithm_info(&self) -> AlgorithmInfo {
        AlgorithmInfo {
            current_algorithm: self.current_algorithm,
            adaptation_enabled: self.adaptation_enabled,
            average_latency: if self.performance_tracker.recent_latencies.is_empty() {
                0.0
            } else {
                self.performance_tracker.recent_latencies.iter().sum::<f32>()
                    / self.performance_tracker.recent_latencies.len() as f32
            },
            average_quality: self.quality_monitor.quality_trend.average_quality,
            quality_improving: self.quality_monitor.quality_trend.improving,
        }
    }

    /// Enable or disable adaptation
    pub fn set_adaptation_enabled(&mut self, enabled: bool) {
        self.adaptation_enabled = enabled;
        info!("🎛️ Adaptive extraction: {}", if enabled { "enabled" } else { "disabled" });
    }
}

/// Algorithm information
#[derive(Debug, Clone)]
pub struct AlgorithmInfo {
    pub current_algorithm: ExtractionAlgorithm,
    pub adaptation_enabled: bool,
    pub average_latency: f32,
    pub average_quality: f32,
    pub quality_improving: bool,
}

impl FeatureExtractor for AdaptiveExtractor {
    fn initialize(&mut self, config: FeatureConfig) -> Result<()> {
        self.initialize_adaptive(&config)?;
        self.config = Some(config);

        debug!("Adaptive feature extractor initialized");
        Ok(())
    }

    fn extract_features(&mut self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        if self.config.is_none() {
            return Err(anyhow::anyhow!("Feature extractor not initialized"));
        }

        self.extract_adaptive(audio)
    }

    fn extract_with_metrics(&mut self, audio: &[f32]) -> Result<FeatureResult> {
        let start_time = Instant::now();
        let features = self.extract_features(audio)?;
        let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        let config = self.config.as_ref().unwrap();

        // Create metrics based on current algorithm
        let complexity_multiplier = match self.current_algorithm {
            ExtractionAlgorithm::RustFFT => 2.0,
            ExtractionAlgorithm::Enhanced => 3.5,
            ExtractionAlgorithm::IPP => 1.5,
        };

        let quality_score = self.quality_monitor.quality_trend.average_quality;

        let metrics = FeatureMetrics {
            n_frames: features.len(),
            n_features: if features.is_empty() { 0 } else { features[0].len() },
            computational_complexity: complexity_multiplier,
            memory_usage_mb: 2.0,
            feature_quality: quality_score,
            spectral_resolution: 1.2,
            efficiency_features_per_sec: if processing_time_ms > 0.0 {
                (features.len() * config.n_mels) as f64 / (processing_time_ms as f64 / 1000.0)
            } else {
                0.0
            },
        };

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
            let base_features = config.n_mels;
            // Enhanced extractor may add deltas
            let total_features = match self.current_algorithm {
                ExtractionAlgorithm::Enhanced => base_features * 2, // May include deltas
                _ => base_features,
            };
            (0, total_features)
        } else {
            (0, 0)
        }
    }

    fn reset(&mut self) {
        self.stats = ExtractorStats::default();
        self.performance_tracker.recent_latencies.clear();
        self.quality_monitor.recent_quality_scores.clear();

        // Reset individual extractors
        if let Some(ref mut extractor) = self.extractors.rustfft {
            extractor.reset();
        }
        if let Some(ref mut extractor) = self.extractors.enhanced {
            extractor.reset();
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
    fn test_adaptive_extractor_creation() {
        let extractor = AdaptiveExtractor::new().unwrap();
        assert_eq!(extractor.current_algorithm, ExtractionAlgorithm::RustFFT);
        assert!(extractor.adaptation_enabled);
    }

    #[test]
    fn test_algorithm_selection() {
        let extractor = AdaptiveExtractor::new().unwrap();

        let low_quality_config = FeatureConfig {
            quality: 0.5,
            enable_advanced: false,
            ..FeatureConfig::default()
        };
        let initial = extractor.select_initial_algorithm(&low_quality_config);
        assert_eq!(initial, ExtractionAlgorithm::RustFFT);

        let high_quality_config = FeatureConfig {
            quality: 0.9,
            enable_advanced: true,
            ..FeatureConfig::default()
        };
        let initial = extractor.select_initial_algorithm(&high_quality_config);
        assert_eq!(initial, ExtractionAlgorithm::Enhanced);
    }

    #[test]
    fn test_feature_quality_estimation() {
        let extractor = AdaptiveExtractor::new().unwrap();

        // Test with empty features
        let empty_features = vec![];
        let quality = extractor.estimate_feature_quality(&empty_features);
        assert_eq!(quality, 0.0);

        // Test with some features
        let features = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.3, 0.4],
            vec![0.15, 0.25, 0.35],
        ];
        let quality = extractor.estimate_feature_quality(&features);
        assert!(quality > 0.0 && quality <= 1.0);
    }
}