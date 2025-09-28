//! AudioProcessor trait for profile abstraction and component integration
//!
//! This module defines the core abstraction for audio processing components,
//! enabling profile-aware processing and seamless component integration.

use crate::profile::Profile;
use super::pipeline::{AudioPipelineResult, AudioPipelineConfig, PipelineMetrics};
use anyhow::Result;
use std::time::Duration;

/// Core trait for audio processing components
pub trait AudioProcessor: Send + Sync {
    /// Initialize the processor with the given configuration
    fn initialize(&mut self, config: AudioPipelineConfig) -> Result<()>;

    /// Process audio data through the complete pipeline
    fn process_audio(&mut self, audio_data: &[f32]) -> Result<AudioPipelineResult>;

    /// Process audio in streaming mode with continuous buffering
    fn process_streaming(&mut self, audio_chunk: &[f32]) -> Result<Option<AudioPipelineResult>>;

    /// Get the current profile
    fn profile(&self) -> Profile;

    /// Check if the processor is meeting real-time constraints
    fn is_realtime_compliant(&self) -> bool;

    /// Get current processing statistics
    fn get_metrics(&self) -> ProcessorMetrics;

    /// Reset processor state
    fn reset(&mut self) -> Result<()>;

    /// Shutdown the processor gracefully
    fn shutdown(&mut self) -> Result<()>;

    /// Get processor capabilities
    fn get_capabilities(&self) -> ProcessorCapabilities;

    /// Update processor configuration at runtime
    fn update_config(&mut self, config: AudioPipelineConfig) -> Result<()>;

    /// Get estimated processing latency for given input size
    fn estimate_latency(&self, input_samples: usize) -> Duration;

    /// Check if processor can handle the given workload
    fn can_handle_workload(&self, sample_rate: u32, channels: u16, frame_size: usize) -> bool;
}

/// Audio processor metrics for monitoring and optimization
#[derive(Debug, Clone, Default)]
pub struct ProcessorMetrics {
    /// Total frames processed
    pub frames_processed: u64,
    /// Average processing latency (ms)
    pub average_latency_ms: f32,
    /// Peak processing latency (ms)
    pub peak_latency_ms: f32,
    /// Current memory usage (MB)
    pub memory_usage_mb: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Processing efficiency (samples/sec)
    pub efficiency_samples_per_sec: f64,
    /// Real-time constraint violations
    pub realtime_violations: u32,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
    /// Component-specific metrics
    pub component_metrics: ComponentMetrics,
}

/// Component-specific performance metrics
#[derive(Debug, Clone, Default)]
pub struct ComponentMetrics {
    /// VAD processing metrics
    pub vad_metrics: ComponentPerformance,
    /// Resampling metrics
    pub resampling_metrics: ComponentPerformance,
    /// Feature extraction metrics
    pub feature_extraction_metrics: ComponentPerformance,
}

/// Individual component performance data
#[derive(Debug, Clone, Default)]
pub struct ComponentPerformance {
    /// Average processing time (ms)
    pub average_time_ms: f32,
    /// Peak processing time (ms)
    pub peak_time_ms: f32,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
}

/// Processor capabilities for feature discovery
#[derive(Debug, Clone)]
pub struct ProcessorCapabilities {
    /// Supported profiles
    pub supported_profiles: Vec<Profile>,
    /// Maximum supported sample rate
    pub max_sample_rate: u32,
    /// Supported channel configurations
    pub supported_channels: Vec<u16>,
    /// Minimum frame size (samples)
    pub min_frame_size: usize,
    /// Maximum frame size (samples)
    pub max_frame_size: usize,
    /// Real-time processing capability
    pub supports_realtime: bool,
    /// Zero-copy processing capability
    pub supports_zero_copy: bool,
    /// SIMD acceleration available
    pub has_simd_acceleration: bool,
    /// GPU acceleration available
    pub has_gpu_acceleration: bool,
    /// Streaming processing capability
    pub supports_streaming: bool,
    /// Supported audio formats
    pub supported_formats: Vec<AudioFormat>,
}

/// Supported audio formats
#[derive(Debug, Clone, PartialEq)]
pub enum AudioFormat {
    F32,
    I16,
    I24,
    I32,
}

/// Audio processor factory for profile-aware instantiation
pub struct AudioProcessorFactory;

impl AudioProcessorFactory {
    /// Create an audio processor for the specified profile
    pub fn create_processor(profile: Profile) -> Result<Box<dyn AudioProcessor>> {
        match profile {
            Profile::Low => Ok(Box::new(LowProfileProcessor::new()?)),
            Profile::Medium => Ok(Box::new(MediumProfileProcessor::new()?)),
            Profile::High => Ok(Box::new(HighProfileProcessor::new()?)),
        }
    }

    /// Create the best processor for the given constraints
    pub fn create_best_fit_processor(
        max_latency_ms: f32,
        max_memory_mb: f64,
        _require_realtime: bool,
    ) -> Result<Box<dyn AudioProcessor>> {
        // Profile selection logic based on constraints
        let profile = if max_latency_ms < 20.0 && max_memory_mb > 2000.0 {
            Profile::High
        } else if max_latency_ms < 50.0 && max_memory_mb > 1000.0 {
            Profile::Medium
        } else {
            Profile::Low
        };

        Self::create_processor(profile)
    }

    /// Get available processor capabilities
    pub fn get_available_capabilities() -> Vec<ProcessorCapabilities> {
        vec![
            LowProfileProcessor::capabilities(),
            MediumProfileProcessor::capabilities(),
            HighProfileProcessor::capabilities(),
        ]
    }
}

/// Profile-specific processor implementations
use super::pipeline::AudioPipeline;

/// Low Profile Audio Processor
pub struct LowProfileProcessor {
    pipeline: Option<AudioPipeline>,
    metrics: ProcessorMetrics,
    config: Option<AudioPipelineConfig>,
}

impl LowProfileProcessor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            pipeline: None,
            metrics: ProcessorMetrics::default(),
            config: None,
        })
    }

    pub fn capabilities() -> ProcessorCapabilities {
        ProcessorCapabilities {
            supported_profiles: vec![Profile::Low],
            max_sample_rate: 48000,
            supported_channels: vec![1, 2],
            min_frame_size: 160,   // 10ms at 16kHz
            max_frame_size: 1024,  // 64ms at 16kHz
            supports_realtime: true,
            supports_zero_copy: true,
            has_simd_acceleration: true,
            has_gpu_acceleration: false,
            supports_streaming: true,
            supported_formats: vec![AudioFormat::F32, AudioFormat::I16],
        }
    }
}

impl AudioProcessor for LowProfileProcessor {
    fn initialize(&mut self, config: AudioPipelineConfig) -> Result<()> {
        let pipeline_config = AudioPipelineConfig {
            profile: Profile::Low,
            max_latency_ms: 30.0,  // Relaxed constraints for Low Profile
            ..config
        };

        self.pipeline = Some(AudioPipeline::new(pipeline_config.clone())?);
        self.config = Some(pipeline_config);
        Ok(())
    }

    fn process_audio(&mut self, audio_data: &[f32]) -> Result<AudioPipelineResult> {
        if let Some(ref mut pipeline) = self.pipeline {
            let result = pipeline.process_audio(audio_data)?;
            self.update_metrics(&result.metrics);
            Ok(result)
        } else {
            Err(anyhow::anyhow!("Processor not initialized"))
        }
    }

    fn process_streaming(&mut self, audio_chunk: &[f32]) -> Result<Option<AudioPipelineResult>> {
        // For low profile, process every chunk
        Ok(Some(self.process_audio(audio_chunk)?))
    }

    fn profile(&self) -> Profile {
        Profile::Low
    }

    fn is_realtime_compliant(&self) -> bool {
        if let Some(ref pipeline) = self.pipeline {
            pipeline.is_meeting_realtime_constraints()
        } else {
            false
        }
    }

    fn get_metrics(&self) -> ProcessorMetrics {
        self.metrics.clone()
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.reset()?;
        }
        self.metrics = ProcessorMetrics::default();
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.pipeline = None;
        Ok(())
    }

    fn get_capabilities(&self) -> ProcessorCapabilities {
        Self::capabilities()
    }

    fn update_config(&mut self, config: AudioPipelineConfig) -> Result<()> {
        self.shutdown()?;
        self.initialize(config)
    }

    fn estimate_latency(&self, input_samples: usize) -> Duration {
        let sample_rate = self.config.as_ref().map(|c| c.input_sample_rate).unwrap_or(16000);
        let audio_duration_ms = (input_samples as f64 / sample_rate as f64) * 1000.0;
        Duration::from_secs_f64((audio_duration_ms + 25.0) / 1000.0) // 25ms processing overhead
    }

    fn can_handle_workload(&self, sample_rate: u32, channels: u16, frame_size: usize) -> bool {
        sample_rate <= 48000 && channels <= 2 && frame_size >= 160 && frame_size <= 1024
    }
}

impl LowProfileProcessor {
    fn update_metrics(&mut self, pipeline_metrics: &PipelineMetrics) {
        self.metrics.frames_processed += 1;
        self.metrics.average_latency_ms = pipeline_metrics.total_processing_time_ms;

        if pipeline_metrics.total_processing_time_ms > self.metrics.peak_latency_ms {
            self.metrics.peak_latency_ms = pipeline_metrics.total_processing_time_ms;
        }

        self.metrics.memory_usage_mb = pipeline_metrics.memory_usage_mb;
        self.metrics.efficiency_samples_per_sec = pipeline_metrics.efficiency_samples_per_sec;

        // Update component metrics
        self.metrics.component_metrics.vad_metrics.average_time_ms = pipeline_metrics.vad_time_ms;
        self.metrics.component_metrics.resampling_metrics.average_time_ms = pipeline_metrics.resampling_time_ms;
        self.metrics.component_metrics.feature_extraction_metrics.average_time_ms = pipeline_metrics.feature_extraction_time_ms;
    }
}

/// Medium Profile Audio Processor
pub struct MediumProfileProcessor {
    pipeline: Option<AudioPipeline>,
    metrics: ProcessorMetrics,
    config: Option<AudioPipelineConfig>,
}

impl MediumProfileProcessor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            pipeline: None,
            metrics: ProcessorMetrics::default(),
            config: None,
        })
    }

    pub fn capabilities() -> ProcessorCapabilities {
        ProcessorCapabilities {
            supported_profiles: vec![Profile::Medium],
            max_sample_rate: 48000,
            supported_channels: vec![1, 2],
            min_frame_size: 160,
            max_frame_size: 2048,
            supports_realtime: true,
            supports_zero_copy: true,
            has_simd_acceleration: true,
            has_gpu_acceleration: true,
            supports_streaming: true,
            supported_formats: vec![AudioFormat::F32, AudioFormat::I16, AudioFormat::I24],
        }
    }

    fn update_metrics(&mut self, pipeline_metrics: &PipelineMetrics) {
        self.metrics.frames_processed += 1;
        self.metrics.average_latency_ms = pipeline_metrics.total_processing_time_ms;

        if pipeline_metrics.total_processing_time_ms > self.metrics.peak_latency_ms {
            self.metrics.peak_latency_ms = pipeline_metrics.total_processing_time_ms;
        }

        self.metrics.memory_usage_mb = pipeline_metrics.memory_usage_mb;
        self.metrics.efficiency_samples_per_sec = pipeline_metrics.efficiency_samples_per_sec;

        // Update component metrics
        self.metrics.component_metrics.vad_metrics.average_time_ms = pipeline_metrics.vad_time_ms;
        self.metrics.component_metrics.resampling_metrics.average_time_ms = pipeline_metrics.resampling_time_ms;
        self.metrics.component_metrics.feature_extraction_metrics.average_time_ms = pipeline_metrics.feature_extraction_time_ms;
    }
}

impl AudioProcessor for MediumProfileProcessor {
    fn initialize(&mut self, config: AudioPipelineConfig) -> Result<()> {
        let pipeline_config = AudioPipelineConfig {
            profile: Profile::Medium,
            max_latency_ms: 25.0,  // Tighter constraints for Medium Profile
            ..config
        };

        self.pipeline = Some(AudioPipeline::new(pipeline_config.clone())?);
        self.config = Some(pipeline_config);
        Ok(())
    }

    fn process_audio(&mut self, audio_data: &[f32]) -> Result<AudioPipelineResult> {
        if let Some(ref mut pipeline) = self.pipeline {
            let result = pipeline.process_audio(audio_data)?;
            self.update_metrics(&result.metrics);
            Ok(result)
        } else {
            Err(anyhow::anyhow!("Processor not initialized"))
        }
    }

    fn process_streaming(&mut self, audio_chunk: &[f32]) -> Result<Option<AudioPipelineResult>> {
        Ok(Some(self.process_audio(audio_chunk)?))
    }

    fn profile(&self) -> Profile {
        Profile::Medium
    }

    fn is_realtime_compliant(&self) -> bool {
        if let Some(ref pipeline) = self.pipeline {
            pipeline.is_meeting_realtime_constraints()
        } else {
            false
        }
    }

    fn get_metrics(&self) -> ProcessorMetrics {
        self.metrics.clone()
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.reset()?;
        }
        self.metrics = ProcessorMetrics::default();
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.pipeline = None;
        Ok(())
    }

    fn get_capabilities(&self) -> ProcessorCapabilities {
        Self::capabilities()
    }

    fn update_config(&mut self, config: AudioPipelineConfig) -> Result<()> {
        self.shutdown()?;
        self.initialize(config)
    }

    fn estimate_latency(&self, input_samples: usize) -> Duration {
        let sample_rate = self.config.as_ref().map(|c| c.input_sample_rate).unwrap_or(16000);
        let audio_duration_ms = (input_samples as f64 / sample_rate as f64) * 1000.0;
        Duration::from_secs_f64((audio_duration_ms + 20.0) / 1000.0) // 20ms processing overhead
    }

    fn can_handle_workload(&self, sample_rate: u32, channels: u16, frame_size: usize) -> bool {
        sample_rate <= 48000 && channels <= 2 && frame_size >= 160 && frame_size <= 2048
    }
}

/// High Profile Audio Processor
pub struct HighProfileProcessor {
    pipeline: Option<AudioPipeline>,
    metrics: ProcessorMetrics,
    config: Option<AudioPipelineConfig>,
}

impl HighProfileProcessor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            pipeline: None,
            metrics: ProcessorMetrics::default(),
            config: None,
        })
    }

    pub fn capabilities() -> ProcessorCapabilities {
        ProcessorCapabilities {
            supported_profiles: vec![Profile::High],
            max_sample_rate: 96000,
            supported_channels: vec![1, 2, 4, 6, 8],
            min_frame_size: 128,
            max_frame_size: 4096,
            supports_realtime: true,
            supports_zero_copy: true,
            has_simd_acceleration: true,
            has_gpu_acceleration: true,
            supports_streaming: true,
            supported_formats: vec![AudioFormat::F32, AudioFormat::I16, AudioFormat::I24, AudioFormat::I32],
        }
    }

    fn update_metrics(&mut self, pipeline_metrics: &PipelineMetrics) {
        self.metrics.frames_processed += 1;
        self.metrics.average_latency_ms = pipeline_metrics.total_processing_time_ms;

        if pipeline_metrics.total_processing_time_ms > self.metrics.peak_latency_ms {
            self.metrics.peak_latency_ms = pipeline_metrics.total_processing_time_ms;
        }

        self.metrics.memory_usage_mb = pipeline_metrics.memory_usage_mb;
        self.metrics.efficiency_samples_per_sec = pipeline_metrics.efficiency_samples_per_sec;

        // Update component metrics
        self.metrics.component_metrics.vad_metrics.average_time_ms = pipeline_metrics.vad_time_ms;
        self.metrics.component_metrics.resampling_metrics.average_time_ms = pipeline_metrics.resampling_time_ms;
        self.metrics.component_metrics.feature_extraction_metrics.average_time_ms = pipeline_metrics.feature_extraction_time_ms;
    }
}

impl AudioProcessor for HighProfileProcessor {
    fn initialize(&mut self, config: AudioPipelineConfig) -> Result<()> {
        let pipeline_config = AudioPipelineConfig {
            profile: Profile::High,
            max_latency_ms: 15.0,  // Strictest constraints for High Profile
            ..config
        };

        self.pipeline = Some(AudioPipeline::new(pipeline_config.clone())?);
        self.config = Some(pipeline_config);
        Ok(())
    }

    fn process_audio(&mut self, audio_data: &[f32]) -> Result<AudioPipelineResult> {
        if let Some(ref mut pipeline) = self.pipeline {
            let result = pipeline.process_audio(audio_data)?;
            self.update_metrics(&result.metrics);
            Ok(result)
        } else {
            Err(anyhow::anyhow!("Processor not initialized"))
        }
    }

    fn process_streaming(&mut self, audio_chunk: &[f32]) -> Result<Option<AudioPipelineResult>> {
        Ok(Some(self.process_audio(audio_chunk)?))
    }

    fn profile(&self) -> Profile {
        Profile::High
    }

    fn is_realtime_compliant(&self) -> bool {
        if let Some(ref pipeline) = self.pipeline {
            pipeline.is_meeting_realtime_constraints()
        } else {
            false
        }
    }

    fn get_metrics(&self) -> ProcessorMetrics {
        self.metrics.clone()
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.reset()?;
        }
        self.metrics = ProcessorMetrics::default();
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.pipeline = None;
        Ok(())
    }

    fn get_capabilities(&self) -> ProcessorCapabilities {
        Self::capabilities()
    }

    fn update_config(&mut self, config: AudioPipelineConfig) -> Result<()> {
        self.shutdown()?;
        self.initialize(config)
    }

    fn estimate_latency(&self, input_samples: usize) -> Duration {
        let sample_rate = self.config.as_ref().map(|c| c.input_sample_rate).unwrap_or(16000);
        let audio_duration_ms = (input_samples as f64 / sample_rate as f64) * 1000.0;
        Duration::from_secs_f64((audio_duration_ms + 12.0) / 1000.0) // 12ms processing overhead
    }

    fn can_handle_workload(&self, sample_rate: u32, channels: u16, frame_size: usize) -> bool {
        sample_rate <= 96000 && channels <= 8 && frame_size >= 128 && frame_size <= 4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_factory() {
        let processor = AudioProcessorFactory::create_processor(Profile::Low);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_processor_capabilities() {
        let capabilities = AudioProcessorFactory::get_available_capabilities();
        assert_eq!(capabilities.len(), 3);
    }

    #[test]
    fn test_best_fit_processor() {
        let processor = AudioProcessorFactory::create_best_fit_processor(10.0, 3000.0, true);
        assert!(processor.is_ok());
        assert_eq!(processor.unwrap().profile(), Profile::High);
    }

    #[test]
    fn test_low_profile_processor() {
        let mut processor = LowProfileProcessor::new().unwrap();
        let config = AudioPipelineConfig::default();

        assert!(processor.initialize(config).is_ok());
        assert_eq!(processor.profile(), Profile::Low);
        assert!(processor.can_handle_workload(16000, 1, 480));
    }
}