//! Adaptive ASR Pipeline with automatic optimization
//!
//! This module provides intelligent ASR pipeline management:
//! - Automatic profile selection based on system resources
//! - Dynamic quality vs performance optimization
//! - Real-time adaptation to processing constraints
//! - Fallback mechanisms for resource limitations

use super::{AsrEngine, AsrConfig, TranscriptionResult, AsrCapabilities, AsrStats, ModelPrecision, AsrMetrics};
use crate::profile::Profile;
use anyhow::Result;
use std::time::Instant;
use tracing::{info, warn, debug};

/// Adaptive ASR pipeline that optimizes based on system resources and requirements
pub struct AdaptiveAsrPipeline {
    config: Option<AsrConfig>,
    stats: AsrStats,

    // Available engines (loaded based on system capabilities)
    low_engine: Option<Box<dyn AsrEngine>>,
    medium_engine: Option<Box<dyn AsrEngine>>,
    high_engine: Option<Box<dyn AsrEngine>>,

    // Current active engine
    active_engine: Option<Box<dyn AsrEngine>>,
    current_profile: Profile,

    // Adaptation state
    system_monitor: SystemResourceMonitor,
    quality_tracker: QualityTracker,
    adaptation_history: Vec<AdaptationEvent>,
    last_adaptation: Instant,
}

/// System resource monitoring for adaptive decisions
struct SystemResourceMonitor {
    // CPU metrics
    cpu_usage_history: Vec<f32>,
    available_cpu_cores: u32,

    // Memory metrics
    available_memory_mb: f64,
    memory_pressure: f32,

    // GPU metrics (if available)
    has_gpu: bool,
    gpu_memory_mb: f64,
    gpu_utilization: f32,

    // Real-time constraints
    target_latency_ms: f32,
    latency_violations: u32,
}

/// Quality tracking for adaptive optimization
struct QualityTracker {
    recent_confidences: Vec<f32>,
    recent_latencies: Vec<f32>,
    recent_wer_estimates: Vec<f32>,
    quality_trend: QualityTrend,
    minimum_acceptable_quality: f32,
}

/// Quality trend analysis
#[derive(Debug, Clone, Copy, PartialEq)]
enum QualityTrend {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Adaptation event for learning
#[derive(Debug, Clone)]
struct AdaptationEvent {
    timestamp: Instant,
    from_profile: Profile,
    to_profile: Profile,
    trigger: AdaptationTrigger,
    outcome: AdaptationOutcome,
}

/// Reasons for adaptation
#[derive(Debug, Clone, Copy, PartialEq)]
enum AdaptationTrigger {
    HighLatency,
    LowQuality,
    ResourceConstraint,
    QualityRequirement,
    UserRequest,
}

/// Adaptation results
#[derive(Debug, Clone, Copy, PartialEq)]
enum AdaptationOutcome {
    Success,
    Failed,
    NoChange,
    Partial,
}

impl AdaptiveAsrPipeline {
    /// Create a new adaptive ASR pipeline
    pub fn new() -> Result<Self> {
        info!("🧠 Initializing Adaptive ASR Pipeline");

        let system_monitor = SystemResourceMonitor::new()?;
        let quality_tracker = QualityTracker::new();

        Ok(Self {
            config: None,
            stats: AsrStats::default(),
            low_engine: None,
            medium_engine: None,
            high_engine: None,
            active_engine: None,
            current_profile: Profile::Medium,
            system_monitor,
            quality_tracker,
            adaptation_history: Vec::new(),
            last_adaptation: Instant::now(),
        })
    }

    /// Initialize with system capability detection
    pub fn initialize_adaptive(&mut self, base_config: AsrConfig) -> Result<()> {
        info!("🔧 Initializing adaptive ASR with capability detection");

        // Detect system capabilities
        let capabilities = self.detect_system_capabilities()?;

        // Initialize available engines based on capabilities
        self.initialize_available_engines(&capabilities, &base_config)?;

        // Select optimal initial profile
        let initial_profile = self.select_optimal_profile(&capabilities, &base_config)?;

        // Activate the selected engine
        self.switch_to_profile(initial_profile)?;

        self.config = Some(base_config);

        info!("✅ Adaptive ASR initialized with {} profile", self.profile_name(initial_profile));
        Ok(())
    }

    /// Transcribe with adaptive optimization
    pub fn transcribe_adaptive(&mut self, mel_features: &[Vec<f32>]) -> Result<TranscriptionResult> {
        let start_time = Instant::now();

        // Check if adaptation is needed before processing
        self.check_adaptation_needed()?;

        // Process with current engine
        let result = if let Some(engine) = &mut self.active_engine {
            engine.transcribe(mel_features)?
        } else {
            return Err(anyhow::anyhow!("No active ASR engine"));
        };

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update quality tracking
        self.quality_tracker.update(&result, processing_time);

        // Update system monitoring
        self.system_monitor.update(processing_time);

        // Check if post-processing adaptation is needed
        self.evaluate_post_processing_adaptation(&result)?;

        debug!("Adaptive ASR processed {} frames in {:.2}ms with {} profile",
               mel_features.len(), processing_time, self.profile_name(self.current_profile));

        Ok(result)
    }

    /// Detect system capabilities for optimal engine selection
    fn detect_system_capabilities(&self) -> Result<SystemCapabilities> {
        info!("🔍 Detecting system capabilities...");

        // CPU detection
        let cpu_cores = num_cpus::get() as u32;
        let available_memory = self.get_available_memory_mb()?;

        // GPU detection (placeholder - would use actual GPU detection)
        let has_gpu = self.detect_gpu_availability();
        let gpu_memory = if has_gpu {
            self.get_gpu_memory_mb()?
        } else {
            0.0
        };

        let capabilities = SystemCapabilities {
            cpu_cores,
            available_memory_mb: available_memory,
            has_gpu,
            gpu_memory_mb: gpu_memory,
            supports_fp16: has_gpu,
            supports_int8: true,
            has_tensorrt: false, // Would detect TensorRT availability
            has_onnx_runtime: true, // Assume ONNX Runtime is available
        };

        info!("📊 System capabilities: {} cores, {:.1}GB RAM, GPU: {}, VRAM: {:.1}GB",
              capabilities.cpu_cores,
              capabilities.available_memory_mb / 1024.0,
              if capabilities.has_gpu { "Yes" } else { "No" },
              capabilities.gpu_memory_mb / 1024.0);

        Ok(capabilities)
    }

    /// Initialize engines based on system capabilities
    fn initialize_available_engines(&mut self, capabilities: &SystemCapabilities, config: &AsrConfig) -> Result<()> {
        // Always initialize Low Profile (CPU-only)
        if capabilities.available_memory_mb >= 200.0 {
            info!("🔧 Initializing Low Profile engine (Whisper-tiny INT8)");
            let mut low_engine = Box::new(super::whisper_tiny::WhisperTinyEngine::new()?) as Box<dyn AsrEngine>;
            low_engine.initialize(config.clone())?;
            self.low_engine = Some(low_engine);
        }

        // Initialize Medium Profile if GPU available
        if capabilities.has_gpu && capabilities.gpu_memory_mb >= 1000.0 {
            info!("🔧 Initializing Medium Profile engine (Whisper-small FP16)");
            let mut medium_engine = Box::new(super::whisper_small::WhisperSmallEngine::new()?) as Box<dyn AsrEngine>;
            medium_engine.initialize(config.clone())?;
            self.medium_engine = Some(medium_engine);
        }

        // Initialize High Profile if sufficient GPU memory
        if capabilities.has_gpu && capabilities.gpu_memory_mb >= 3000.0 {
            info!("🔧 Initializing High Profile engine (Parakeet-TDT streaming)");
            let mut high_engine = Box::new(super::parakeet_streaming::ParakeetEngine::new()?) as Box<dyn AsrEngine>;
            high_engine.initialize(config.clone())?;
            self.high_engine = Some(high_engine);
        }

        Ok(())
    }

    /// Select optimal profile based on capabilities and requirements
    fn select_optimal_profile(&self, capabilities: &SystemCapabilities, config: &AsrConfig) -> Result<Profile> {
        // Priority order: meet quality requirements while respecting constraints

        // If High Profile requested and available
        if config.profile == Profile::High && self.high_engine.is_some() {
            return Ok(Profile::High);
        }

        // If Medium Profile requested and available
        if config.profile == Profile::Medium && self.medium_engine.is_some() {
            return Ok(Profile::Medium);
        }

        // Fallback logic based on capabilities
        if capabilities.has_gpu && capabilities.gpu_memory_mb >= 3000.0 && self.high_engine.is_some() {
            Ok(Profile::High)
        } else if capabilities.has_gpu && capabilities.gpu_memory_mb >= 1000.0 && self.medium_engine.is_some() {
            Ok(Profile::Medium)
        } else if self.low_engine.is_some() {
            Ok(Profile::Low)
        } else {
            Err(anyhow::anyhow!("No suitable ASR engine available for system capabilities"))
        }
    }

    /// Switch to a different profile
    fn switch_to_profile(&mut self, profile: Profile) -> Result<()> {
        let engine = match profile {
            Profile::Low => self.low_engine.take(),
            Profile::Medium => self.medium_engine.take(),
            Profile::High => self.high_engine.take(),
        };

        if let Some(new_engine) = engine {
            self.active_engine = Some(new_engine);
            self.current_profile = profile;
            info!("🔄 Switched to {} profile", self.profile_name(profile));
            Ok(())
        } else {
            Err(anyhow::anyhow!("Profile {:?} not available", profile))
        }
    }

    /// Check if adaptation is needed based on current conditions
    fn check_adaptation_needed(&mut self) -> Result<()> {
        // Don't adapt too frequently
        if self.last_adaptation.elapsed().as_secs() < 5 {
            return Ok(());
        }

        // Check latency constraints
        if self.system_monitor.latency_violations > 3 {
            self.consider_downgrade_for_performance()?;
        }

        // Check quality requirements
        if self.quality_tracker.is_quality_degrading() {
            self.consider_upgrade_for_quality()?;
        }

        // Check resource constraints
        if self.system_monitor.memory_pressure > 0.8 {
            self.consider_downgrade_for_resources()?;
        }

        Ok(())
    }

    /// Consider downgrading for better performance
    fn consider_downgrade_for_performance(&mut self) -> Result<()> {
        let target_profile = match self.current_profile {
            Profile::High => Profile::Medium,
            Profile::Medium => Profile::Low,
            Profile::Low => return Ok(()), // Already at lowest
        };

        if self.is_profile_available(target_profile) {
            warn!("📉 Downgrading to {} profile due to latency violations", self.profile_name(target_profile));
            self.switch_to_profile(target_profile)?;
            self.record_adaptation(AdaptationTrigger::HighLatency, AdaptationOutcome::Success)?;
        }

        Ok(())
    }

    /// Consider upgrading for better quality
    fn consider_upgrade_for_quality(&mut self) -> Result<()> {
        let target_profile = match self.current_profile {
            Profile::Low => Profile::Medium,
            Profile::Medium => Profile::High,
            Profile::High => return Ok(()), // Already at highest
        };

        if self.is_profile_available(target_profile) && self.can_afford_upgrade(target_profile) {
            info!("📈 Upgrading to {} profile for better quality", self.profile_name(target_profile));
            self.switch_to_profile(target_profile)?;
            self.record_adaptation(AdaptationTrigger::QualityRequirement, AdaptationOutcome::Success)?;
        }

        Ok(())
    }

    /// Consider downgrading due to resource constraints
    fn consider_downgrade_for_resources(&mut self) -> Result<()> {
        let target_profile = match self.current_profile {
            Profile::High => Profile::Medium,
            Profile::Medium => Profile::Low,
            Profile::Low => return Ok(()), // Already at lowest
        };

        if self.is_profile_available(target_profile) {
            warn!("💾 Downgrading to {} profile due to resource constraints", self.profile_name(target_profile));
            self.switch_to_profile(target_profile)?;
            self.record_adaptation(AdaptationTrigger::ResourceConstraint, AdaptationOutcome::Success)?;
        }

        Ok(())
    }

    /// Check if a profile is available
    fn is_profile_available(&self, profile: Profile) -> bool {
        match profile {
            Profile::Low => self.low_engine.is_some(),
            Profile::Medium => self.medium_engine.is_some(),
            Profile::High => self.high_engine.is_some(),
        }
    }

    /// Check if system can afford an upgrade
    fn can_afford_upgrade(&self, _target_profile: Profile) -> bool {
        // Simplified check - in real implementation, consider:
        // - Available GPU memory
        // - CPU utilization
        // - Memory pressure
        // - Real-time constraints
        self.system_monitor.memory_pressure < 0.6 &&
        self.system_monitor.latency_violations == 0
    }

    /// Evaluate adaptation needs after processing
    fn evaluate_post_processing_adaptation(&mut self, result: &TranscriptionResult) -> Result<()> {
        // Check if quality is below threshold
        if result.confidence < self.quality_tracker.minimum_acceptable_quality {
            debug!("Quality below threshold: {:.3} < {:.3}",
                   result.confidence, self.quality_tracker.minimum_acceptable_quality);
        }

        // Check if latency exceeded target
        if result.processing_time_ms > self.system_monitor.target_latency_ms {
            self.system_monitor.latency_violations += 1;
            debug!("Latency violation: {:.1}ms > {:.1}ms",
                   result.processing_time_ms, self.system_monitor.target_latency_ms);
        } else {
            // Reset violations on successful processing
            self.system_monitor.latency_violations = 0;
        }

        Ok(())
    }

    /// Record adaptation event for learning
    fn record_adaptation(&mut self, trigger: AdaptationTrigger, outcome: AdaptationOutcome) -> Result<()> {
        let event = AdaptationEvent {
            timestamp: Instant::now(),
            from_profile: self.current_profile,
            to_profile: self.current_profile, // Will be updated by caller
            trigger,
            outcome,
        };

        self.adaptation_history.push(event);
        self.last_adaptation = Instant::now();

        // Keep only recent history
        if self.adaptation_history.len() > 100 {
            self.adaptation_history.remove(0);
        }

        Ok(())
    }

    /// Get profile name
    fn profile_name(&self, profile: Profile) -> &'static str {
        match profile {
            Profile::Low => "Low",
            Profile::Medium => "Medium",
            Profile::High => "High",
        }
    }

    /// Placeholder system methods
    fn get_available_memory_mb(&self) -> Result<f64> {
        // Placeholder - would use system APIs
        Ok(8192.0) // Assume 8GB
    }

    fn detect_gpu_availability(&self) -> bool {
        // Placeholder - would detect actual GPU
        true
    }

    fn get_gpu_memory_mb(&self) -> Result<f64> {
        // Placeholder - would query GPU memory
        Ok(4096.0) // Assume 4GB
    }
}

/// System capability detection results
struct SystemCapabilities {
    cpu_cores: u32,
    available_memory_mb: f64,
    has_gpu: bool,
    gpu_memory_mb: f64,
    supports_fp16: bool,
    supports_int8: bool,
    has_tensorrt: bool,
    has_onnx_runtime: bool,
}

impl SystemResourceMonitor {
    fn new() -> Result<Self> {
        Ok(Self {
            cpu_usage_history: Vec::new(),
            available_cpu_cores: num_cpus::get() as u32,
            available_memory_mb: 8192.0, // Placeholder
            memory_pressure: 0.0,
            has_gpu: true, // Placeholder
            gpu_memory_mb: 4096.0, // Placeholder
            gpu_utilization: 0.0,
            target_latency_ms: 200.0, // Default target
            latency_violations: 0,
        })
    }

    fn update(&mut self, processing_time_ms: f32) {
        // Update latency tracking
        if processing_time_ms > self.target_latency_ms {
            self.latency_violations += 1;
        }

        // Update resource usage (placeholder)
        self.memory_pressure = 0.3; // Would calculate actual pressure
        self.gpu_utilization = 0.7; // Would query actual GPU usage
    }
}

impl QualityTracker {
    fn new() -> Self {
        Self {
            recent_confidences: Vec::new(),
            recent_latencies: Vec::new(),
            recent_wer_estimates: Vec::new(),
            quality_trend: QualityTrend::Unknown,
            minimum_acceptable_quality: 0.7,
        }
    }

    fn update(&mut self, result: &TranscriptionResult, processing_time_ms: f32) {
        // Track recent metrics
        self.recent_confidences.push(result.confidence);
        self.recent_latencies.push(processing_time_ms);
        self.recent_wer_estimates.push(result.metrics.estimated_wer);

        // Keep only recent history
        let max_history = 20;
        if self.recent_confidences.len() > max_history {
            self.recent_confidences.remove(0);
            self.recent_latencies.remove(0);
            self.recent_wer_estimates.remove(0);
        }

        // Update quality trend
        self.quality_trend = self.calculate_quality_trend();
    }

    fn is_quality_degrading(&self) -> bool {
        matches!(self.quality_trend, QualityTrend::Degrading)
    }

    fn calculate_quality_trend(&self) -> QualityTrend {
        if self.recent_confidences.len() < 5 {
            return QualityTrend::Unknown;
        }

        let recent_avg = self.recent_confidences.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = self.recent_confidences.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;

        let change = recent_avg - older_avg;

        if change > 0.05 {
            QualityTrend::Improving
        } else if change < -0.05 {
            QualityTrend::Degrading
        } else {
            QualityTrend::Stable
        }
    }
}

impl AsrEngine for AdaptiveAsrPipeline {
    fn initialize(&mut self, config: AsrConfig) -> Result<()> {
        self.initialize_adaptive(config)
    }

    fn transcribe(&mut self, mel_features: &[Vec<f32>]) -> Result<TranscriptionResult> {
        self.transcribe_adaptive(mel_features)
    }

    fn transcribe_streaming(&mut self, mel_chunk: &[Vec<f32>]) -> Result<Option<TranscriptionResult>> {
        if let Some(engine) = &mut self.active_engine {
            engine.transcribe_streaming(mel_chunk)
        } else {
            Err(anyhow::anyhow!("No active ASR engine"))
        }
    }

    fn profile(&self) -> Profile {
        self.current_profile
    }

    fn supports_streaming(&self) -> bool {
        if let Some(engine) = &self.active_engine {
            engine.supports_streaming()
        } else {
            false
        }
    }

    fn get_capabilities(&self) -> AsrCapabilities {
        // Return combined capabilities of all available engines
        AsrCapabilities {
            supported_profiles: vec![Profile::Low, Profile::Medium, Profile::High],
            supported_languages: vec![
                "en".to_string(), "es".to_string(), "fr".to_string(), "de".to_string(),
                "it".to_string(), "pt".to_string(), "zh".to_string(), "ja".to_string(),
                "ko".to_string(), "ru".to_string(), "ar".to_string(), "hi".to_string(),
            ],
            supported_precisions: vec![ModelPrecision::INT8, ModelPrecision::FP16, ModelPrecision::FP32],
            max_audio_duration_s: 300.0, // Maximum of all engines
            supports_streaming: true,
            supports_real_time: true,
            has_gpu_acceleration: self.system_monitor.has_gpu,
            model_size_mb: 1200.0, // Largest model size
            memory_requirement_mb: 2500.0, // Highest requirement
        }
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(engine) = &mut self.active_engine {
            engine.reset()?;
        }
        self.stats = AsrStats::default();
        self.quality_tracker = QualityTracker::new();
        self.adaptation_history.clear();
        Ok(())
    }

    fn get_stats(&self) -> AsrStats {
        self.stats.clone()
    }

    fn update_config(&mut self, config: AsrConfig) -> Result<()> {
        self.config = Some(config.clone());
        if let Some(engine) = &mut self.active_engine {
            engine.update_config(config)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_asr_creation() {
        let pipeline = AdaptiveAsrPipeline::new();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_quality_tracker() {
        let mut tracker = QualityTracker::new();
        assert_eq!(tracker.quality_trend, QualityTrend::Unknown);

        // Simulate improving quality
        for confidence in [0.7, 0.75, 0.8, 0.85, 0.9] {
            let result = TranscriptionResult {
                text: "test".to_string(),
                language: "en".to_string(),
                confidence,
                word_segments: Vec::new(),
                processing_time_ms: 100.0,
                model_name: "test".to_string(),
                metrics: AsrMetrics::default(),
            };
            tracker.update(&result, 100.0);
        }
    }

    #[test]
    fn test_system_monitor() {
        let monitor = SystemResourceMonitor::new();
        assert!(monitor.is_ok());

        let mut monitor = monitor.unwrap();
        assert_eq!(monitor.latency_violations, 0);

        // Simulate latency violation
        monitor.update(300.0); // Above default 200ms target
        assert_eq!(monitor.latency_violations, 1);
    }
}