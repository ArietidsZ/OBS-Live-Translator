//! Unified audio processing pipeline with profile-aware component integration
//!
//! This module provides a complete audio processing pipeline that integrates:
//! - Voice Activity Detection (VAD)
//! - Audio Resampling
//! - Feature Extraction (Mel-Spectrograms)
//! - Real-time streaming with ring buffers
//! - Zero-copy frame processing

use super::{
    features::{FeatureConfig, FeatureExtractionManager},
    resampling::{ResamplingConfig, ResamplingManager},
    vad::{VadConfig, VadManager, VadResult},
};
use crate::profile::Profile;
use anyhow::Result;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Audio processing pipeline result
#[derive(Debug, Clone)]
pub struct AudioPipelineResult {
    /// Voice activity detection results
    pub vad_results: Vec<VadResult>,
    /// Resampled audio data
    pub resampled_audio: Vec<f32>,
    /// Extracted mel-spectrogram features
    pub mel_features: Vec<Vec<f32>>,
    /// Overall processing metrics
    pub metrics: PipelineMetrics,
    /// Processing timestamps
    pub timestamps: ProcessingTimestamps,
}

/// Audio processing pipeline configuration
#[derive(Debug, Clone)]
pub struct AudioPipelineConfig {
    /// Profile for component selection
    pub profile: Profile,
    /// Input audio sample rate
    pub input_sample_rate: u32,
    /// Target sample rate for processing
    pub target_sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Frame size for processing (samples)
    pub frame_size: usize,
    /// Enable real-time processing constraints
    pub real_time_mode: bool,
    /// Enable zero-copy optimizations
    pub enable_zero_copy: bool,
    /// Maximum processing latency (ms)
    pub max_latency_ms: f32,
}

impl Default for AudioPipelineConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Medium,
            input_sample_rate: 48000,  // Common audio interface rate
            target_sample_rate: 16000, // Whisper standard
            channels: 1,               // Mono processing
            frame_size: 480,           // 30ms at 16kHz
            real_time_mode: true,
            enable_zero_copy: true,
            max_latency_ms: 50.0, // 50ms max latency
        }
    }
}

/// Processing performance metrics
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    /// Total processing time (ms)
    pub total_processing_time_ms: f32,
    /// VAD processing time (ms)
    pub vad_time_ms: f32,
    /// Resampling time (ms)
    pub resampling_time_ms: f32,
    /// Feature extraction time (ms)
    pub feature_extraction_time_ms: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Processing efficiency (samples/sec)
    pub efficiency_samples_per_sec: f64,
}

/// Processing timestamps for latency analysis
#[derive(Debug, Clone)]
pub struct ProcessingTimestamps {
    /// Input received timestamp
    pub input_received: Instant,
    /// VAD completed timestamp
    pub vad_completed: Instant,
    /// Resampling completed timestamp
    pub resampling_completed: Instant,
    /// Feature extraction completed timestamp
    pub features_completed: Instant,
    /// Pipeline completed timestamp
    pub pipeline_completed: Instant,
}

impl Default for ProcessingTimestamps {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            input_received: now,
            vad_completed: now,
            resampling_completed: now,
            features_completed: now,
            pipeline_completed: now,
        }
    }
}

/// Unified audio processing pipeline
pub struct AudioPipeline {
    config: AudioPipelineConfig,

    // Component managers
    vad_manager: VadManager,
    resampling_manager: ResamplingManager,
    feature_manager: FeatureExtractionManager,

    // Ring buffer for streaming
    audio_ring_buffer: AudioRingBuffer,

    // Performance monitoring
    metrics_history: VecDeque<PipelineMetrics>,
    processing_stats: PipelineStats,

    // Real-time constraints
    deadline_monitor: DeadlineMonitor,

    // Zero-copy frame processing
    frame_processor: ZeroCopyFrameProcessor,
}

impl AudioPipeline {
    /// Create a new audio processing pipeline
    pub fn new(config: AudioPipelineConfig) -> Result<Self> {
        info!(
            "🎵 Initializing Audio Processing Pipeline for profile {:?}",
            config.profile
        );

        // Initialize component managers with profile-specific configurations
        let vad_config = VadConfig {
            sample_rate: config.target_sample_rate,
            frame_size: 480, // 30ms at 16kHz
            ..VadConfig::default()
        };
        let vad_manager = VadManager::new(config.profile, vad_config)?;

        let resampling_config = ResamplingConfig {
            input_sample_rate: config.input_sample_rate,
            output_sample_rate: config.target_sample_rate,
            channels: config.channels,
            enable_simd: true,
            real_time_mode: config.real_time_mode,
            ..ResamplingConfig::default()
        };
        let mut resampling_manager =
            ResamplingManager::new(config.profile, resampling_config.clone())?;
        resampling_manager.initialize(resampling_config)?;

        let feature_config = FeatureConfig {
            sample_rate: config.target_sample_rate,
            frame_size: config.frame_size,
            hop_length: config.frame_size / 4, // 75% overlap
            enable_advanced: config.profile != Profile::Low,
            real_time_mode: config.real_time_mode,
            ..FeatureConfig::default()
        };
        let mut feature_manager =
            FeatureExtractionManager::new(config.profile, feature_config.clone())?;
        feature_manager.initialize(feature_config)?;

        // Initialize pipeline components
        let audio_ring_buffer = AudioRingBuffer::new(config.input_sample_rate, config.channels)?;
        let deadline_monitor = DeadlineMonitor::new(config.max_latency_ms);
        let frame_processor =
            ZeroCopyFrameProcessor::new(config.frame_size, config.enable_zero_copy)?;

        info!("✅ Audio Processing Pipeline initialized successfully");

        Ok(Self {
            config,
            vad_manager,
            resampling_manager,
            feature_manager,
            audio_ring_buffer,
            metrics_history: VecDeque::with_capacity(100),
            processing_stats: PipelineStats::default(),
            deadline_monitor,
            frame_processor,
        })
    }

    /// Process audio through the complete pipeline
    pub fn process_audio(&mut self, audio_data: &[f32]) -> Result<AudioPipelineResult> {
        let timestamps = ProcessingTimestamps {
            input_received: Instant::now(),
            ..Default::default()
        };

        // Check real-time constraints
        if self.config.real_time_mode {
            self.deadline_monitor.start_processing()?;
        }

        let start_time = Instant::now();

        // Step 1: Add audio to ring buffer for continuous processing
        self.audio_ring_buffer.write_audio(audio_data)?;

        // Step 2: Extract frames for processing
        let frames = self
            .frame_processor
            .extract_frames(&mut self.audio_ring_buffer)?;

        if frames.is_empty() {
            return Ok(AudioPipelineResult {
                vad_results: Vec::new(),
                resampled_audio: Vec::new(),
                mel_features: Vec::new(),
                metrics: PipelineMetrics::default(),
                timestamps,
            });
        }

        // Process all frames through the pipeline
        let mut all_vad_results = Vec::new();
        let mut all_resampled_audio = Vec::new();
        let mut all_mel_features = Vec::new();

        let mut pipeline_metrics = PipelineMetrics::default();

        for frame in frames {
            // Step 3: Voice Activity Detection
            let vad_start = Instant::now();
            let vad_result = self.vad_manager.process_frame(&frame)?;
            let vad_time = vad_start.elapsed().as_secs_f32() * 1000.0;
            pipeline_metrics.vad_time_ms += vad_time;

            all_vad_results.push(vad_result.clone());

            // Only process further if speech is detected (optimization)
            if vad_result.is_speech {
                // Step 4: Audio Resampling
                let resample_start = Instant::now();
                let resampled = self.resampling_manager.resample(&frame)?;
                let resample_time = resample_start.elapsed().as_secs_f32() * 1000.0;
                pipeline_metrics.resampling_time_ms += resample_time;

                // Step 5: Feature Extraction
                let feature_start = Instant::now();
                let features = self.feature_manager.extract_features(&resampled)?;
                let feature_time = feature_start.elapsed().as_secs_f32() * 1000.0;
                pipeline_metrics.feature_extraction_time_ms += feature_time;

                all_resampled_audio.extend(resampled);
                all_mel_features.extend(features);
            } else {
                // For non-speech frames, add silence markers
                let silence_length = (frame.len() as f32 * self.config.target_sample_rate as f32
                    / self.config.input_sample_rate as f32)
                    as usize;
                all_resampled_audio.extend(vec![0.0; silence_length]);
            }
        }

        let total_time = start_time.elapsed().as_secs_f32() * 1000.0;
        pipeline_metrics.total_processing_time_ms = total_time;

        // Calculate efficiency metrics
        pipeline_metrics.efficiency_samples_per_sec = if total_time > 0.0 {
            (audio_data.len() as f64) / (total_time as f64 / 1000.0)
        } else {
            0.0
        };

        // Update memory usage estimate
        pipeline_metrics.memory_usage_mb = self.estimate_memory_usage();

        // Check deadline compliance
        if self.config.real_time_mode {
            self.deadline_monitor.check_deadline(&pipeline_metrics)?;
        }

        // Update statistics
        self.update_processing_stats(&pipeline_metrics);

        let final_timestamps = ProcessingTimestamps {
            input_received: timestamps.input_received,
            vad_completed: timestamps.input_received
                + std::time::Duration::from_secs_f32(pipeline_metrics.vad_time_ms / 1000.0),
            resampling_completed: timestamps.input_received
                + std::time::Duration::from_secs_f32(
                    (pipeline_metrics.vad_time_ms + pipeline_metrics.resampling_time_ms) / 1000.0,
                ),
            features_completed: timestamps.input_received
                + std::time::Duration::from_secs_f32(
                    (pipeline_metrics.vad_time_ms
                        + pipeline_metrics.resampling_time_ms
                        + pipeline_metrics.feature_extraction_time_ms)
                        / 1000.0,
                ),
            pipeline_completed: Instant::now(),
        };

        debug!(
            "Audio pipeline processed {} samples in {:.2}ms ({:.1}x real-time)",
            audio_data.len(),
            total_time,
            (audio_data.len() as f64 / self.config.input_sample_rate as f64) * 1000.0
                / total_time as f64
        );

        Ok(AudioPipelineResult {
            vad_results: all_vad_results,
            resampled_audio: all_resampled_audio,
            mel_features: all_mel_features,
            metrics: pipeline_metrics,
            timestamps: final_timestamps,
        })
    }

    /// Get pipeline processing statistics
    pub fn get_stats(&self) -> &PipelineStats {
        &self.processing_stats
    }

    /// Check if pipeline is meeting real-time constraints
    pub fn is_meeting_realtime_constraints(&self) -> bool {
        if self.metrics_history.is_empty() {
            return true;
        }

        let recent_avg_latency: f32 = self
            .metrics_history
            .iter()
            .rev()
            .take(10)
            .map(|m| m.total_processing_time_ms)
            .sum::<f32>()
            / 10.0;

        recent_avg_latency <= self.config.max_latency_ms
    }

    /// Reset pipeline state
    pub fn reset(&mut self) -> Result<()> {
        self.vad_manager.reset();
        self.resampling_manager.reset();
        self.feature_manager.reset();
        self.audio_ring_buffer.reset()?;
        self.metrics_history.clear();
        self.processing_stats = PipelineStats::default();

        info!("🔄 Audio pipeline reset");
        Ok(())
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> f64 {
        let ring_buffer_mb = self.audio_ring_buffer.memory_usage_mb();
        let frame_processor_mb = self.frame_processor.memory_usage_mb();
        let component_overhead_mb = 5.0; // Estimated overhead for VAD, resampling, features

        ring_buffer_mb + frame_processor_mb + component_overhead_mb
    }

    /// Update processing statistics
    fn update_processing_stats(&mut self, metrics: &PipelineMetrics) {
        // Add to history
        self.metrics_history.push_back(metrics.clone());
        if self.metrics_history.len() > 100 {
            self.metrics_history.pop_front();
        }

        // Update running statistics
        self.processing_stats.total_frames_processed += 1;
        self.processing_stats.total_processing_time_ms += metrics.total_processing_time_ms as f64;

        // Calculate averages
        let frame_count = self.processing_stats.total_frames_processed as f64;
        self.processing_stats.average_latency_ms =
            (self.processing_stats.total_processing_time_ms / frame_count) as f32;

        // Update peak latency
        if metrics.total_processing_time_ms > self.processing_stats.peak_latency_ms {
            self.processing_stats.peak_latency_ms = metrics.total_processing_time_ms;
        }

        // Update efficiency
        self.processing_stats.average_efficiency_samples_per_sec = self
            .metrics_history
            .iter()
            .map(|m| m.efficiency_samples_per_sec)
            .sum::<f64>()
            / self.metrics_history.len() as f64;
    }

    /// Get current configuration
    pub fn config(&self) -> &AudioPipelineConfig {
        &self.config
    }

    /// Get component manager references
    pub fn get_vad_manager(&self) -> &VadManager {
        &self.vad_manager
    }

    pub fn get_resampling_manager(&self) -> &ResamplingManager {
        &self.resampling_manager
    }

    pub fn get_feature_manager(&self) -> &FeatureExtractionManager {
        &self.feature_manager
    }
}

/// Pipeline processing statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_frames_processed: u64,
    pub total_processing_time_ms: f64,
    pub average_latency_ms: f32,
    pub peak_latency_ms: f32,
    pub average_efficiency_samples_per_sec: f64,
    pub realtime_violations: u32,
}

/// Ring buffer for continuous audio streaming
struct AudioRingBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    /// Sample rate stored for validation and debugging
    #[allow(dead_code)]
    sample_rate: u32,
    /// Number of channels stored for validation and debugging
    #[allow(dead_code)]
    channels: u16,
    capacity: usize,
}

impl AudioRingBuffer {
    fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        // Buffer for 1 second of audio
        let capacity = (sample_rate as usize) * (channels as usize);

        Ok(Self {
            buffer: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            sample_rate,
            channels,
            capacity,
        })
    }

    fn write_audio(&mut self, audio: &[f32]) -> Result<()> {
        for &sample in audio {
            self.buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
        Ok(())
    }

    fn read_frame(&mut self, frame_size: usize) -> Option<Vec<f32>> {
        let available = self.available_samples();
        if available < frame_size {
            return None;
        }

        let mut frame = Vec::with_capacity(frame_size);
        for _ in 0..frame_size {
            frame.push(self.buffer[self.read_pos]);
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }

        Some(frame)
    }

    fn available_samples(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.write_pos = 0;
        self.read_pos = 0;
        self.buffer.fill(0.0);
        Ok(())
    }

    fn memory_usage_mb(&self) -> f64 {
        (self.buffer.len() * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)
    }
}

/// Zero-copy frame processor for efficient memory usage
struct ZeroCopyFrameProcessor {
    frame_size: usize,
    enable_zero_copy: bool,
    frame_buffer: Vec<f32>,
}

impl ZeroCopyFrameProcessor {
    fn new(frame_size: usize, enable_zero_copy: bool) -> Result<Self> {
        Ok(Self {
            frame_size,
            enable_zero_copy,
            frame_buffer: Vec::with_capacity(frame_size),
        })
    }

    fn extract_frames(&mut self, ring_buffer: &mut AudioRingBuffer) -> Result<Vec<Vec<f32>>> {
        let mut frames = Vec::new();

        while let Some(frame) = ring_buffer.read_frame(self.frame_size) {
            if self.enable_zero_copy {
                // In a real zero-copy implementation, we'd use references or slices
                // For now, we still copy but optimize the process
                frames.push(frame);
            } else {
                frames.push(frame);
            }
        }

        Ok(frames)
    }

    fn memory_usage_mb(&self) -> f64 {
        (self.frame_buffer.capacity() * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)
    }
}

/// Real-time deadline monitoring
struct DeadlineMonitor {
    max_latency_ms: f32,
    processing_start: Option<Instant>,
}

impl DeadlineMonitor {
    fn new(max_latency_ms: f32) -> Self {
        Self {
            max_latency_ms,
            processing_start: None,
        }
    }

    fn start_processing(&mut self) -> Result<()> {
        self.processing_start = Some(Instant::now());
        Ok(())
    }

    fn check_deadline(&mut self, _metrics: &PipelineMetrics) -> Result<()> {
        if let Some(start) = self.processing_start {
            let elapsed = start.elapsed().as_secs_f32() * 1000.0;

            if elapsed > self.max_latency_ms {
                warn!(
                    "⚠️ Real-time deadline exceeded: {:.2}ms > {:.2}ms",
                    elapsed, self.max_latency_ms
                );
                return Err(anyhow::anyhow!("Real-time deadline exceeded"));
            }

            self.processing_start = None;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_pipeline_creation() {
        let config = AudioPipelineConfig::default();
        let pipeline = AudioPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = AudioRingBuffer::new(16000, 1).unwrap();

        // Write some audio
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.write_audio(&audio).unwrap();

        // Read a frame
        let frame = buffer.read_frame(3);
        assert!(frame.is_some());
        assert_eq!(frame.unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zero_copy_frame_processor() {
        let mut processor = ZeroCopyFrameProcessor::new(480, true).unwrap();
        let mut buffer = AudioRingBuffer::new(16000, 1).unwrap();

        // Add enough audio for multiple frames
        let audio: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        buffer.write_audio(&audio).unwrap();

        let frames = processor.extract_frames(&mut buffer).unwrap();
        assert!(!frames.is_empty());
        assert_eq!(frames[0].len(), 480);
    }

    #[test]
    fn test_deadline_monitor() {
        let mut monitor = DeadlineMonitor::new(50.0);

        monitor.start_processing().unwrap();

        let metrics = PipelineMetrics {
            total_processing_time_ms: 25.0,
            ..Default::default()
        };

        // Should pass deadline check
        assert!(monitor.check_deadline(&metrics).is_ok());
    }
}
