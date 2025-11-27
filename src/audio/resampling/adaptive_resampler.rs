//! Adaptive resampling pipeline with automatic quality and algorithm selection
//!
//! This module provides intelligent resampling that:
//! - Automatically detects optimal resampling ratios
//! - Selects quality-based algorithms dynamically
//! - Provides buffered streaming support
//! - Monitors real-time quality metrics

use super::cubic_resampler::CubicResampler;
use super::linear_resampler::LinearResampler;
use super::soxr_resampler::SoxrResampler;
use super::{AudioResampler, QualityMetrics, ResamplerStats, ResamplingConfig, ResamplingResult};
use crate::profile::Profile;
use anyhow::Result;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, info};

/// Adaptive resampler that automatically selects optimal algorithms
pub struct AdaptiveResampler {
    /// Current resampling configuration
    config: Option<ResamplingConfig>,

    /// Available resampler implementations
    linear_resampler: LinearResampler,
    cubic_resampler: CubicResampler,
    soxr_resampler: SoxrResampler,

    /// Current active resampler
    active_resampler: ResamplerType,

    /// Quality monitoring
    quality_monitor: QualityMonitor,

    /// Streaming buffer management
    stream_buffer: StreamBuffer,

    /// Adaptive configuration
    adaptive_config: AdaptiveConfig,

    /// Processing statistics
    stats: ResamplerStats,
}

/// Type of resampler currently active
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResamplerType {
    Linear,
    Cubic,
    Soxr,
}

/// Configuration for adaptive behavior
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Enable automatic algorithm switching
    pub auto_algorithm_selection: bool,

    /// Quality threshold for algorithm selection (0.0-1.0)
    pub quality_threshold: f32,

    /// Enable real-time monitoring
    pub enable_monitoring: bool,

    /// Buffer size for streaming
    pub stream_buffer_size: usize,

    /// Maximum latency tolerance (ms)
    pub max_latency_ms: f32,

    /// Quality vs performance preference (0.0=performance, 1.0=quality)
    pub quality_preference: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            auto_algorithm_selection: true,
            quality_threshold: 0.7,
            enable_monitoring: true,
            stream_buffer_size: 8192,
            max_latency_ms: 10.0,
            quality_preference: 0.5,
        }
    }
}

/// Quality monitoring system
struct QualityMonitor {
    /// Recent quality measurements
    quality_history: VecDeque<f32>,

    /// Performance history (processing time)
    performance_history: VecDeque<f32>,

    /// Quality trend analysis
    quality_trend: QualityTrend,

    /// Monitoring enabled
    enabled: bool,
}

/// Quality trend information
#[derive(Debug, Clone)]
pub struct QualityTrend {
    /// Average quality over recent samples
    pub average_quality: f32,

    /// Quality improvement/degradation rate
    pub trend_slope: f32,

    /// Stability measure (lower = more stable)
    pub stability: f32,

    /// Recommendation for algorithm change
    pub recommendation: AlgorithmRecommendation,
}

/// Algorithm change recommendation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgorithmRecommendation {
    /// Keep current algorithm
    Keep,
    /// Switch to faster algorithm
    SwitchToFaster,
    /// Switch to higher quality algorithm
    SwitchToHigherQuality,
    /// Switch based on specific criteria
    SwitchConditional,
}

/// Streaming buffer for continuous processing
struct StreamBuffer {
    /// Internal buffer for overlap processing
    buffer: VecDeque<f32>,

    /// Maximum buffer size
    max_size: usize,

    /// Overlap size for continuity
    overlap_size: usize,
}

impl AdaptiveResampler {
    /// Create a new adaptive resampler
    pub fn new(adaptive_config: AdaptiveConfig) -> Result<Self> {
        info!("🧠 Initializing Adaptive Resampling Pipeline");

        let linear_resampler = LinearResampler::new()?;
        let cubic_resampler = CubicResampler::new()?;
        let soxr_resampler = SoxrResampler::new()?;

        let quality_monitor = QualityMonitor {
            quality_history: VecDeque::with_capacity(50),
            performance_history: VecDeque::with_capacity(50),
            quality_trend: QualityTrend {
                average_quality: 0.0,
                trend_slope: 0.0,
                stability: 1.0,
                recommendation: AlgorithmRecommendation::Keep,
            },
            enabled: adaptive_config.enable_monitoring,
        };

        let stream_buffer = StreamBuffer {
            buffer: VecDeque::with_capacity(adaptive_config.stream_buffer_size),
            max_size: adaptive_config.stream_buffer_size,
            overlap_size: 256, // Default overlap
        };

        Ok(Self {
            config: None,
            linear_resampler,
            cubic_resampler,
            soxr_resampler,
            active_resampler: ResamplerType::Linear, // Start with fastest
            quality_monitor,
            stream_buffer,
            adaptive_config,
            stats: ResamplerStats::default(),
        })
    }

    /// Initialize with configuration and automatic algorithm selection
    pub fn initialize_adaptive(
        &mut self,
        mut config: ResamplingConfig,
        profile: Profile,
    ) -> Result<()> {
        info!(
            "🔧 Initializing adaptive resampler for profile {:?}",
            profile
        );

        // Select initial algorithm based on profile and config
        self.active_resampler = self.select_optimal_algorithm(&config, profile)?;

        // Adjust configuration based on adaptive settings
        self.adjust_config_for_adaptive(&mut config);

        // Initialize all resamplers
        self.linear_resampler.initialize(config.clone())?;
        self.cubic_resampler.initialize(config.clone())?;
        self.soxr_resampler.initialize(config.clone())?;

        self.config = Some(config);

        info!(
            "✅ Adaptive resampler initialized with {:?} algorithm",
            self.active_resampler
        );

        Ok(())
    }

    /// Select optimal algorithm based on requirements
    fn select_optimal_algorithm(
        &self,
        config: &ResamplingConfig,
        profile: Profile,
    ) -> Result<ResamplerType> {
        let ratio = config.output_sample_rate as f64 / config.input_sample_rate as f64;

        // Consider multiple factors for algorithm selection
        let quality_factor = config.quality;
        let realtime_factor = if config.real_time_mode { 1.0 } else { 0.5 };
        let ratio_factor = if ratio > 2.0 || ratio < 0.5 { 1.0 } else { 0.7 }; // Complex ratios need better algorithms

        let algorithm_score = quality_factor * self.adaptive_config.quality_preference
            + realtime_factor * (1.0 - self.adaptive_config.quality_preference)
            + ratio_factor * 0.3;

        let selected = match profile {
            Profile::Low => {
                // Always use linear for low profile
                ResamplerType::Linear
            }
            Profile::Medium => {
                // Choose between linear and cubic based on requirements
                if algorithm_score > 0.6 || ratio_factor > 0.8 {
                    ResamplerType::Cubic
                } else {
                    ResamplerType::Linear
                }
            }
            Profile::High => {
                // Choose between cubic and SoXR based on quality requirements
                if algorithm_score > 0.8 {
                    ResamplerType::Soxr
                } else {
                    ResamplerType::Cubic
                }
            }
        };

        debug!(
            "Algorithm selection: score={:.3}, selected={:?}",
            algorithm_score, selected
        );

        Ok(selected)
    }

    /// Adjust configuration for adaptive processing
    fn adjust_config_for_adaptive(&self, config: &mut ResamplingConfig) {
        // Adjust buffer size for streaming
        config.buffer_size = self.adaptive_config.stream_buffer_size;

        // Enable SIMD if adaptive monitoring is enabled
        if self.adaptive_config.enable_monitoring {
            config.enable_simd = true;
        }

        // Adjust quality based on adaptive preference
        config.quality = config.quality * self.adaptive_config.quality_preference
            + (1.0 - self.adaptive_config.quality_preference) * 0.5;
    }

    /// Process audio with adaptive algorithm selection
    pub fn resample_adaptive(&mut self, input: &[f32]) -> Result<ResamplingResult> {
        let start_time = Instant::now();

        // Add input to stream buffer for continuous processing
        self.stream_buffer.add_input(input);

        // Check if we need to switch algorithms
        if self.adaptive_config.auto_algorithm_selection && self.should_switch_algorithm()? {
            self.switch_algorithm()?;
        }

        // Get buffered input for processing
        let buffered_input = self.stream_buffer.get_processing_buffer();

        // Process with current algorithm
        let mut result = match self.active_resampler {
            ResamplerType::Linear => self
                .linear_resampler
                .resample_with_metrics(&buffered_input)?,
            ResamplerType::Cubic => self
                .cubic_resampler
                .resample_with_metrics(&buffered_input)?,
            ResamplerType::Soxr => self.soxr_resampler.resample_with_metrics(&buffered_input)?,
        };

        // Update timing
        result.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Monitor quality if enabled
        if self.quality_monitor.enabled {
            self.update_quality_monitoring(&result);
        }

        // Update stream buffer with processed output
        self.stream_buffer.update_with_output(&result.samples);

        // Update statistics
        self.update_adaptive_stats(&result);

        debug!(
            "Adaptive resampling: algorithm={:?}, quality={:.3}, time={:.2}ms",
            self.active_resampler,
            self.calculate_overall_quality(&result.quality_metrics),
            result.processing_time_ms
        );

        Ok(result)
    }

    /// Check if algorithm should be switched
    fn should_switch_algorithm(&self) -> Result<bool> {
        if !self.adaptive_config.auto_algorithm_selection {
            return Ok(false);
        }

        // Don't switch too frequently
        if self.quality_monitor.quality_history.len() < 10 {
            return Ok(false);
        }

        // Check if current performance is below threshold
        let current_quality = self.quality_monitor.quality_trend.average_quality;
        let quality_stable = self.quality_monitor.quality_trend.stability > 0.8;

        // Switch if quality is consistently below threshold
        let should_switch =
            current_quality < self.adaptive_config.quality_threshold && quality_stable;

        // Or if latency is too high
        let average_latency = self.quality_monitor.performance_history.iter().sum::<f32>()
            / self.quality_monitor.performance_history.len() as f32;
        let latency_exceeded = average_latency > self.adaptive_config.max_latency_ms;

        Ok(should_switch || latency_exceeded)
    }

    /// Switch to a different algorithm
    fn switch_algorithm(&mut self) -> Result<()> {
        let recommendation = self.quality_monitor.quality_trend.recommendation;

        let new_algorithm = match recommendation {
            AlgorithmRecommendation::SwitchToFaster => {
                match self.active_resampler {
                    ResamplerType::Soxr => ResamplerType::Cubic,
                    ResamplerType::Cubic => ResamplerType::Linear,
                    ResamplerType::Linear => ResamplerType::Linear, // Already fastest
                }
            }
            AlgorithmRecommendation::SwitchToHigherQuality => {
                match self.active_resampler {
                    ResamplerType::Linear => ResamplerType::Cubic,
                    ResamplerType::Cubic => ResamplerType::Soxr,
                    ResamplerType::Soxr => ResamplerType::Soxr, // Already highest quality
                }
            }
            _ => return Ok(()), // No switch needed
        };

        if new_algorithm != self.active_resampler {
            info!(
                "🔄 Switching algorithm: {:?} → {:?}",
                self.active_resampler, new_algorithm
            );
            self.active_resampler = new_algorithm;

            // Reset quality monitoring after switch
            self.quality_monitor.quality_history.clear();
            self.quality_monitor.performance_history.clear();
        }

        Ok(())
    }

    /// Update quality monitoring with new results
    fn update_quality_monitoring(&mut self, result: &ResamplingResult) {
        if !self.quality_monitor.enabled {
            return;
        }

        let overall_quality = self.calculate_overall_quality(&result.quality_metrics);

        // Add to history
        self.quality_monitor
            .quality_history
            .push_back(overall_quality);
        self.quality_monitor
            .performance_history
            .push_back(result.processing_time_ms);

        // Maintain history size
        if self.quality_monitor.quality_history.len() > 50 {
            self.quality_monitor.quality_history.pop_front();
        }
        if self.quality_monitor.performance_history.len() > 50 {
            self.quality_monitor.performance_history.pop_front();
        }

        // Update trend analysis
        self.update_quality_trend();
    }

    /// Calculate overall quality score from metrics
    fn calculate_overall_quality(&self, metrics: &QualityMetrics) -> f32 {
        // Weighted combination of quality factors
        let snr_score = (metrics.snr_db / 100.0).min(1.0);
        let thd_score = (1.0 - metrics.thd_percent / 1.0).max(0.0);
        let freq_score = (1.0 - metrics.frequency_response_deviation_db / 3.0).max(0.0);

        (snr_score * 0.4 + thd_score * 0.3 + freq_score * 0.3).clamp(0.0, 1.0)
    }

    /// Update quality trend analysis
    fn update_quality_trend(&mut self) {
        let history = &self.quality_monitor.quality_history;

        if history.len() < 5 {
            return;
        }

        // Calculate average quality
        let average_quality = history.iter().sum::<f32>() / history.len() as f32;

        // Calculate trend slope (simple linear regression)
        let n = history.len() as f32;
        let sum_x = (0..history.len()).sum::<usize>() as f32;
        let sum_y = history.iter().sum::<f32>();
        let sum_xy = history
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum::<f32>();
        let sum_x2 = (0..history.len()).map(|i| (i * i) as f32).sum::<f32>();

        let trend_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Calculate stability (inverse of variance)
        let variance = history
            .iter()
            .map(|&q| (q - average_quality).powi(2))
            .sum::<f32>()
            / history.len() as f32;
        let stability = if variance > 0.0 {
            1.0 / (1.0 + variance)
        } else {
            1.0
        };

        // Determine recommendation
        let recommendation = if average_quality < self.adaptive_config.quality_threshold {
            if trend_slope < -0.01 {
                AlgorithmRecommendation::SwitchToHigherQuality
            } else {
                AlgorithmRecommendation::SwitchConditional
            }
        } else if average_quality > 0.9 && trend_slope > 0.01 {
            AlgorithmRecommendation::SwitchToFaster
        } else {
            AlgorithmRecommendation::Keep
        };

        self.quality_monitor.quality_trend = QualityTrend {
            average_quality,
            trend_slope,
            stability,
            recommendation,
        };
    }

    /// Update adaptive processing statistics
    fn update_adaptive_stats(&mut self, result: &ResamplingResult) {
        // Update basic stats
        self.stats.total_samples_processed += result.samples.len() as u64;
        self.stats.total_processing_time_ms += result.processing_time_ms as f64;

        // Update averages
        let num_operations =
            (self.stats.total_processing_time_ms / result.processing_time_ms as f64) as f32;
        self.stats.average_processing_time_ms =
            (self.stats.total_processing_time_ms / num_operations as f64) as f32;

        // Update peak
        if result.processing_time_ms > self.stats.peak_processing_time_ms {
            self.stats.peak_processing_time_ms = result.processing_time_ms;
        }

        // Update quality
        let overall_quality = self.calculate_overall_quality(&result.quality_metrics);
        self.stats.average_quality_score =
            (self.stats.average_quality_score * (num_operations - 1.0) + overall_quality)
                / num_operations;

        // Update efficiency
        self.stats.efficiency_samples_per_sec = result.quality_metrics.efficiency_samples_per_sec;
    }

    /// Get current quality trend
    pub fn get_quality_trend(&self) -> &QualityTrend {
        &self.quality_monitor.quality_trend
    }

    /// Get current active algorithm
    pub fn get_active_algorithm(&self) -> ResamplerType {
        self.active_resampler
    }

    /// Force algorithm switch for testing/manual control
    pub fn force_algorithm_switch(&mut self, algorithm: ResamplerType) -> Result<()> {
        if algorithm != self.active_resampler {
            info!(
                "🔧 Manual algorithm switch: {:?} → {:?}",
                self.active_resampler, algorithm
            );
            self.active_resampler = algorithm;
        }
        Ok(())
    }
}

impl StreamBuffer {
    /// Add new input to the buffer
    fn add_input(&mut self, input: &[f32]) {
        self.buffer.extend(input.iter());

        // Maintain buffer size
        while self.buffer.len() > self.max_size {
            self.buffer.pop_front();
        }
    }

    /// Get buffered data for processing
    fn get_processing_buffer(&self) -> Vec<f32> {
        self.buffer.iter().cloned().collect()
    }

    /// Update buffer after processing
    fn update_with_output(&mut self, _output: &[f32]) {
        // Remove processed samples, keeping overlap
        let remove_count = self.buffer.len().saturating_sub(self.overlap_size);
        for _ in 0..remove_count {
            self.buffer.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_resampler_creation() {
        let config = AdaptiveConfig::default();
        let resampler = AdaptiveResampler::new(config).unwrap();
        assert_eq!(resampler.active_resampler, ResamplerType::Linear);
    }

    #[test]
    fn test_algorithm_selection() {
        let config = AdaptiveConfig::default();
        let resampler = AdaptiveResampler::new(config).unwrap();

        let resampling_config = ResamplingConfig {
            quality: 0.9,
            real_time_mode: false,
            ..Default::default()
        };

        // High quality should prefer better algorithms
        let algorithm = resampler
            .select_optimal_algorithm(&resampling_config, Profile::High)
            .unwrap();
        assert!(matches!(
            algorithm,
            ResamplerType::Soxr | ResamplerType::Cubic
        ));
    }

    #[test]
    fn test_quality_calculation() {
        let config = AdaptiveConfig::default();
        let resampler = AdaptiveResampler::new(config).unwrap();

        let metrics = QualityMetrics {
            snr_db: 60.0,
            thd_percent: 0.1,
            frequency_response_deviation_db: 0.5,
            alias_suppression_db: 40.0,
            efficiency_samples_per_sec: 1000000.0,
        };

        let quality = resampler.calculate_overall_quality(&metrics);
        assert!(quality > 0.5);
        assert!(quality <= 1.0);
    }

    #[test]
    fn test_stream_buffer() {
        let mut buffer = StreamBuffer {
            buffer: VecDeque::new(),
            max_size: 100,
            overlap_size: 10,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        buffer.add_input(&input);

        let processing_buffer = buffer.get_processing_buffer();
        assert_eq!(processing_buffer.len(), 4);
        assert_eq!(processing_buffer, input);
    }
}
