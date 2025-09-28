//! Voice Activity Detection (VAD) implementations for different performance profiles
//!
//! This module provides multi-tier VAD systems:
//! - Low Profile: WebRTC VAD with energy/ZCR fallback
//! - Medium Profile: TEN VAD ONNX model
//! - High Profile: Silero VAD with GPU acceleration

pub mod webrtc_vad;
pub mod ten_vad;
pub mod silero_vad;
pub mod adaptive_vad;

use crate::profile::Profile;
use anyhow::Result;
use std::time::Instant;
use tracing::{info, debug};

/// Voice activity detection result
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Whether speech is detected
    pub is_speech: bool,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Frame number for tracking
    pub frame_number: u64,
    /// Additional metadata
    pub metadata: VadMetadata,
}

/// Additional VAD metadata
#[derive(Debug, Clone, Default)]
pub struct VadMetadata {
    /// Energy level of the frame
    pub energy: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Spectral properties (for advanced VADs)
    pub spectral_centroid: Option<f32>,
    /// Model-specific confidence scores
    pub model_scores: Vec<f32>,
}

/// Configuration for VAD processing
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Frame size in samples
    pub frame_size: usize,
    /// Hop length in samples
    pub hop_length: usize,
    /// Sensitivity threshold (0.0-1.0)
    pub sensitivity: f32,
    /// Enable temporal smoothing
    pub enable_smoothing: bool,
    /// Smoothing window size
    pub smoothing_window: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 480,  // 30ms at 16kHz
            hop_length: 160,  // 10ms hop
            sensitivity: 0.5,
            enable_smoothing: true,
            smoothing_window: 5,
        }
    }
}

/// Trait for Voice Activity Detection implementations
pub trait VadProcessor: Send + Sync {
    /// Process audio frame and detect voice activity
    fn process_frame(&mut self, audio_frame: &[f32]) -> Result<VadResult>;

    /// Get the required frame size in samples
    fn frame_size(&self) -> usize;

    /// Get the sample rate this VAD expects
    fn sample_rate(&self) -> u32;

    /// Reset internal state
    fn reset(&mut self);

    /// Get processing statistics
    fn get_stats(&self) -> VadStats;
}

/// VAD processing statistics
#[derive(Debug, Clone, Default)]
pub struct VadStats {
    pub total_frames: u64,
    pub speech_frames: u64,
    pub average_processing_time_ms: f32,
    pub peak_processing_time_ms: f32,
    pub average_confidence: f32,
}

/// Multi-tier VAD manager that selects appropriate VAD based on profile
pub struct VadManager {
    profile: Profile,
    processor: Box<dyn VadProcessor>,
    config: VadConfig,
    frame_counter: u64,
    stats: VadStats,
}

impl VadManager {
    /// Create a new VAD manager for the given profile
    pub fn new(profile: Profile, config: VadConfig) -> Result<Self> {
        info!("🎤 Initializing VAD manager for profile {:?}", profile);

        let processor: Box<dyn VadProcessor> = match profile {
            Profile::Low => {
                info!("📊 Creating WebRTC VAD (Low Profile)");
                Box::new(webrtc_vad::WebRtcVad::new(config.clone())?)
            }
            Profile::Medium => {
                info!("📊 Creating TEN VAD (Medium Profile)");
                Box::new(ten_vad::TenVad::new(config.clone())?)
            }
            Profile::High => {
                info!("📊 Creating Silero VAD (High Profile)");
                Box::new(silero_vad::SileroVad::new(config.clone())?)
            }
        };

        info!("✅ VAD manager initialized for profile {:?}", profile);

        Ok(Self {
            profile,
            processor,
            config,
            frame_counter: 0,
            stats: VadStats::default(),
        })
    }

    /// Process audio frame with the appropriate VAD
    pub fn process_frame(&mut self, audio_frame: &[f32]) -> Result<VadResult> {
        let start_time = Instant::now();

        // Validate frame size
        if audio_frame.len() != self.processor.frame_size() {
            return Err(anyhow::anyhow!(
                "Invalid frame size: expected {}, got {}",
                self.processor.frame_size(),
                audio_frame.len()
            ));
        }

        // Process with the profile-specific VAD
        let mut result = self.processor.process_frame(audio_frame)?;

        // Update frame number
        result.frame_number = self.frame_counter;
        self.frame_counter += 1;

        // Update timing
        result.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        self.update_stats(&result);

        debug!("VAD result: speech={}, confidence={:.3}, time={:.2}ms",
               result.is_speech, result.confidence, result.processing_time_ms);

        Ok(result)
    }

    /// Process multiple frames in batch
    pub fn process_batch(&mut self, audio_frames: &[Vec<f32>]) -> Result<Vec<VadResult>> {
        let mut results = Vec::with_capacity(audio_frames.len());

        for frame in audio_frames {
            results.push(self.process_frame(frame)?);
        }

        Ok(results)
    }

    /// Get the current profile
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Get VAD configuration
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Get processing statistics
    pub fn stats(&self) -> &VadStats {
        &self.stats
    }

    /// Reset VAD state and statistics
    pub fn reset(&mut self) {
        self.processor.reset();
        self.frame_counter = 0;
        self.stats = VadStats::default();
        info!("🔄 VAD manager reset");
    }

    /// Update internal statistics
    fn update_stats(&mut self, result: &VadResult) {
        self.stats.total_frames += 1;

        if result.is_speech {
            self.stats.speech_frames += 1;
        }

        // Update average processing time
        let frame_count = self.stats.total_frames as f32;
        self.stats.average_processing_time_ms =
            (self.stats.average_processing_time_ms * (frame_count - 1.0) + result.processing_time_ms) / frame_count;

        // Update peak processing time
        if result.processing_time_ms > self.stats.peak_processing_time_ms {
            self.stats.peak_processing_time_ms = result.processing_time_ms;
        }

        // Update average confidence
        self.stats.average_confidence =
            (self.stats.average_confidence * (frame_count - 1.0) + result.confidence) / frame_count;
    }

    /// Get speech activity ratio
    pub fn get_speech_ratio(&self) -> f32 {
        if self.stats.total_frames == 0 {
            0.0
        } else {
            self.stats.speech_frames as f32 / self.stats.total_frames as f32
        }
    }

    /// Check if VAD is meeting performance targets
    pub fn is_meeting_targets(&self) -> bool {
        let target_latency = match self.profile {
            Profile::Low => 1.0,    // <1ms
            Profile::Medium => 5.0, // <5ms
            Profile::High => 1.0,   // <1ms
        };

        self.stats.average_processing_time_ms <= target_latency
    }

    /// Get memory usage estimate in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        match self.profile {
            Profile::Low => 158_000,      // 158KB target
            Profile::Medium => 15_000_000, // 15MB
            Profile::High => 50_000_000,   // 50MB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.frame_size, 480); // 30ms at 16kHz
        assert_eq!(config.hop_length, 160); // 10ms at 16kHz
    }

    #[test]
    fn test_vad_result_creation() {
        let result = VadResult {
            is_speech: true,
            confidence: 0.85,
            processing_time_ms: 0.5,
            frame_number: 42,
            metadata: VadMetadata::default(),
        };

        assert!(result.is_speech);
        assert_eq!(result.confidence, 0.85);
        assert_eq!(result.frame_number, 42);
    }

    #[test]
    fn test_vad_stats_default() {
        let stats = VadStats::default();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.speech_frames, 0);
        assert_eq!(stats.average_processing_time_ms, 0.0);
    }
}