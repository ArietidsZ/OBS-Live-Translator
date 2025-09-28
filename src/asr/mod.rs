//! Multi-tier Automatic Speech Recognition (ASR) system
//!
//! This module provides profile-aware speech recognition implementations:
//! - Low Profile: Whisper-tiny INT8 quantized for CPU efficiency
//! - Medium Profile: Whisper-small FP16 GPU for balanced performance
//! - High Profile: Parakeet-TDT streaming for maximum accuracy
//! - Adaptive Pipeline: Automatic model selection and optimization

pub mod whisper_tiny;
pub mod whisper_small;
pub mod parakeet_streaming;
pub mod adaptive_asr;
pub mod inference_engine;

use crate::profile::Profile;
use anyhow::Result;
use std::time::Instant;
use tracing::{info, debug};

/// ASR transcription result with confidence metrics
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Transcribed text
    pub text: String,
    /// Language detected/used
    pub language: String,
    /// Overall confidence score (0.0-1.0)
    pub confidence: f32,
    /// Word-level timestamps and confidences
    pub word_segments: Vec<WordSegment>,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Model used for transcription
    pub model_name: String,
    /// Quality metrics
    pub metrics: AsrMetrics,
}

/// Word-level segment with timing and confidence
#[derive(Debug, Clone)]
pub struct WordSegment {
    /// Word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// ASR configuration for different profiles
#[derive(Debug, Clone)]
pub struct AsrConfig {
    /// Profile for model selection
    pub profile: Profile,
    /// Target language (auto-detect if None)
    pub language: Option<String>,
    /// Model precision (INT8, FP16, FP32)
    pub precision: ModelPrecision,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Beam search width
    pub beam_size: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Enable streaming mode
    pub enable_streaming: bool,
    /// Chunk size for streaming (seconds)
    pub chunk_duration_s: f32,
    /// Context overlap for streaming (seconds)
    pub context_overlap_s: f32,
    /// Real-time processing mode
    pub real_time_mode: bool,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Medium,
            language: None, // Auto-detect
            precision: ModelPrecision::FP16,
            max_sequence_length: 448, // Whisper default
            beam_size: 5,
            temperature: 0.0, // Deterministic
            enable_streaming: true,
            chunk_duration_s: 2.0,
            context_overlap_s: 0.5,
            real_time_mode: true,
        }
    }
}

/// Model precision options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelPrecision {
    INT8,
    FP16,
    FP32,
}

/// ASR performance and quality metrics
#[derive(Debug, Clone, Default)]
pub struct AsrMetrics {
    /// Processing latency (ms)
    pub latency_ms: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: f32,
    /// Real-time factor (processing_time / audio_duration)
    pub real_time_factor: f32,
    /// Words per second processed
    pub words_per_second: f32,
    /// Model confidence score
    pub model_confidence: f32,
    /// Quality assessment
    pub estimated_wer: f32, // Word Error Rate estimate
}

/// Trait for ASR implementations
pub trait AsrEngine: Send + Sync {
    /// Initialize the ASR engine with configuration
    fn initialize(&mut self, config: AsrConfig) -> Result<()>;

    /// Transcribe mel-spectrogram features
    fn transcribe(&mut self, mel_features: &[Vec<f32>]) -> Result<TranscriptionResult>;

    /// Transcribe with streaming support
    fn transcribe_streaming(&mut self, mel_chunk: &[Vec<f32>]) -> Result<Option<TranscriptionResult>>;

    /// Get current profile
    fn profile(&self) -> Profile;

    /// Check if engine supports streaming
    fn supports_streaming(&self) -> bool;

    /// Get engine capabilities
    fn get_capabilities(&self) -> AsrCapabilities;

    /// Reset engine state
    fn reset(&mut self) -> Result<()>;

    /// Get processing statistics
    fn get_stats(&self) -> AsrStats;

    /// Update configuration at runtime
    fn update_config(&mut self, config: AsrConfig) -> Result<()>;
}

/// ASR engine capabilities
#[derive(Debug, Clone)]
pub struct AsrCapabilities {
    /// Supported profiles
    pub supported_profiles: Vec<Profile>,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Supported precisions
    pub supported_precisions: Vec<ModelPrecision>,
    /// Maximum audio duration (seconds)
    pub max_audio_duration_s: f32,
    /// Streaming support
    pub supports_streaming: bool,
    /// Real-time processing capability
    pub supports_real_time: bool,
    /// GPU acceleration available
    pub has_gpu_acceleration: bool,
    /// Model size in MB
    pub model_size_mb: f64,
    /// Memory requirements in MB
    pub memory_requirement_mb: f64,
}

/// ASR processing statistics
#[derive(Debug, Clone, Default)]
pub struct AsrStats {
    pub total_audio_processed_s: f64,
    pub total_processing_time_ms: f64,
    pub average_latency_ms: f32,
    pub peak_latency_ms: f32,
    pub total_words_transcribed: u64,
    pub average_confidence: f32,
    pub real_time_violations: u32,
    pub success_rate: f32,
}

/// Multi-tier ASR manager that selects appropriate engine based on profile
pub struct AsrManager {
    profile: Profile,
    engine: Box<dyn AsrEngine>,
    config: AsrConfig,
    stats: AsrStats,
}

impl AsrManager {
    /// Create a new ASR manager for the given profile
    pub fn new(profile: Profile, config: AsrConfig) -> Result<Self> {
        info!("🎤 Initializing ASR Manager for profile {:?}", profile);

        let engine: Box<dyn AsrEngine> = match profile {
            Profile::Low => {
                info!("📊 Creating Whisper-tiny INT8 Engine (Low Profile)");
                Box::new(whisper_tiny::WhisperTinyEngine::new()?)
            }
            Profile::Medium => {
                info!("📊 Creating Whisper-small FP16 Engine (Medium Profile)");
                Box::new(whisper_small::WhisperSmallEngine::new()?)
            }
            Profile::High => {
                info!("📊 Creating Parakeet-TDT Streaming Engine (High Profile)");
                Box::new(parakeet_streaming::ParakeetEngine::new()?)
            }
        };

        info!("✅ ASR Manager initialized for profile {:?}", profile);

        Ok(Self {
            profile,
            engine,
            config,
            stats: AsrStats::default(),
        })
    }

    /// Initialize with the given configuration
    pub fn initialize(&mut self, config: AsrConfig) -> Result<()> {
        info!("🔧 Initializing ASR engine: {} profile, {} precision",
              self.profile_name(), self.precision_name(&config.precision));

        self.config = config.clone();
        self.engine.initialize(config)?;

        Ok(())
    }

    /// Transcribe mel-spectrogram features with the profile-specific engine
    pub fn transcribe(&mut self, mel_features: &[Vec<f32>]) -> Result<TranscriptionResult> {
        let start_time = Instant::now();
        let mut result = self.engine.transcribe(mel_features)?;
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update result timing
        result.processing_time_ms = processing_time;

        // Update statistics
        self.update_stats(&result);

        debug!("ASR transcribed {} frames in {:.2}ms: \"{}\" ({} profile)",
               mel_features.len(), processing_time,
               if result.text.len() > 50 {
                   format!("{}...", &result.text[..50])
               } else {
                   result.text.clone()
               },
               self.profile_name());

        Ok(result)
    }

    /// Transcribe with streaming support
    pub fn transcribe_streaming(&mut self, mel_chunk: &[Vec<f32>]) -> Result<Option<TranscriptionResult>> {
        if !self.engine.supports_streaming() {
            return self.transcribe(mel_chunk).map(Some);
        }

        let start_time = Instant::now();
        let result = self.engine.transcribe_streaming(mel_chunk)?;
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        if let Some(mut transcription) = result {
            transcription.processing_time_ms = processing_time;
            self.update_stats(&transcription);

            debug!("ASR streaming chunk processed in {:.2}ms: \"{}\"",
                   processing_time,
                   if transcription.text.len() > 30 {
                       format!("{}...", &transcription.text[..30])
                   } else {
                       transcription.text.clone()
                   });

            Ok(Some(transcription))
        } else {
            Ok(None)
        }
    }

    /// Get the current profile
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Get the current configuration
    pub fn config(&self) -> &AsrConfig {
        &self.config
    }

    /// Get processing statistics
    pub fn stats(&self) -> &AsrStats {
        &self.stats
    }

    /// Check if ASR is meeting real-time constraints
    pub fn is_meeting_realtime_constraints(&self) -> bool {
        let target_rtf = match self.profile {
            Profile::Low => 0.5,    // 0.5x real-time for Low Profile
            Profile::Medium => 0.3, // 0.3x real-time for Medium Profile
            Profile::High => 0.2,   // 0.2x real-time for High Profile
        };

        self.stats.average_latency_ms > 0.0 &&
        (self.stats.average_latency_ms / 1000.0) <= target_rtf
    }

    /// Reset ASR state and statistics
    pub fn reset(&mut self) -> Result<()> {
        self.engine.reset()?;
        self.stats = AsrStats::default();
        info!("🔄 ASR Manager reset");
        Ok(())
    }

    /// Get quality assessment for current settings
    pub fn assess_quality(&self) -> QualityAssessment {
        let performance_score = match self.profile {
            Profile::Low => {
                // Focus on efficiency and speed
                if self.stats.average_latency_ms <= 200.0 { 0.8 } else { 0.6 }
            }
            Profile::Medium => {
                // Balanced quality and performance
                let latency_score = if self.stats.average_latency_ms <= 150.0 { 0.8 } else { 0.6 };
                let confidence_score = self.stats.average_confidence;
                (latency_score + confidence_score) / 2.0
            }
            Profile::High => {
                // Maximum quality focus
                let latency_score = if self.stats.average_latency_ms <= 100.0 { 0.9 } else { 0.7 };
                let confidence_score = self.stats.average_confidence;
                let rtf_score = if self.is_meeting_realtime_constraints() { 0.9 } else { 0.6 };
                (latency_score + confidence_score + rtf_score) / 3.0
            }
        };

        QualityAssessment {
            overall_score: performance_score,
            meets_profile_requirements: performance_score >= 0.7,
            profile: self.profile,
            average_latency_ms: self.stats.average_latency_ms,
            average_confidence: self.stats.average_confidence,
            real_time_factor: if self.stats.total_audio_processed_s > 0.0 {
                (self.stats.total_processing_time_ms / 1000.0) / self.stats.total_audio_processed_s
            } else {
                0.0
            } as f32,
        }
    }

    /// Update internal statistics
    fn update_stats(&mut self, result: &TranscriptionResult) {
        // Estimate audio duration from mel features (rough approximation)
        let estimated_audio_duration = result.word_segments.last()
            .map(|seg| seg.end_time)
            .unwrap_or(2.0); // Default 2 seconds if no segments

        // Update totals
        self.stats.total_audio_processed_s += estimated_audio_duration as f64;
        self.stats.total_processing_time_ms += result.processing_time_ms as f64;
        self.stats.total_words_transcribed += result.word_segments.len() as u64;

        // Update averages
        let total_operations = if self.stats.total_audio_processed_s > 0.0 {
            self.stats.total_processing_time_ms / result.processing_time_ms as f64
        } else {
            1.0
        };

        self.stats.average_latency_ms =
            (self.stats.total_processing_time_ms / total_operations) as f32;

        // Update peak latency
        if result.processing_time_ms > self.stats.peak_latency_ms {
            self.stats.peak_latency_ms = result.processing_time_ms;
        }

        // Update confidence (moving average)
        self.stats.average_confidence =
            (self.stats.average_confidence * 0.9) + (result.confidence * 0.1);

        // Update success rate (assume success if confidence > 0.5)
        let current_success = if result.confidence > 0.5 { 1.0 } else { 0.0 };
        self.stats.success_rate =
            (self.stats.success_rate * 0.95) + (current_success * 0.05);
    }

    /// Get profile name as string
    fn profile_name(&self) -> &'static str {
        match self.profile {
            Profile::Low => "Low",
            Profile::Medium => "Medium",
            Profile::High => "High",
        }
    }

    /// Get precision name as string
    fn precision_name(&self, precision: &ModelPrecision) -> &'static str {
        match precision {
            ModelPrecision::INT8 => "INT8",
            ModelPrecision::FP16 => "FP16",
            ModelPrecision::FP32 => "FP32",
        }
    }

    /// Get engine capabilities
    pub fn get_capabilities(&self) -> AsrCapabilities {
        self.engine.get_capabilities()
    }

    /// Update configuration at runtime
    pub fn update_config(&mut self, config: AsrConfig) -> Result<()> {
        self.config = config.clone();
        self.engine.update_config(config)
    }
}

/// Quality assessment result for ASR
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall performance score (0.0-1.0)
    pub overall_score: f32,
    /// Whether quality meets profile requirements
    pub meets_profile_requirements: bool,
    /// Profile being assessed
    pub profile: Profile,
    /// Average processing latency (ms)
    pub average_latency_ms: f32,
    /// Average transcription confidence
    pub average_confidence: f32,
    /// Real-time processing factor
    pub real_time_factor: f32,
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
    fn test_asr_config_default() {
        let config = AsrConfig::default();
        assert_eq!(config.profile, Profile::Medium);
        assert_eq!(config.precision, ModelPrecision::FP16);
        assert!(config.enable_streaming);
    }

    #[test]
    fn test_quality_assessment_grade() {
        let assessment = QualityAssessment {
            overall_score: 0.95,
            meets_profile_requirements: true,
            profile: Profile::High,
            average_latency_ms: 80.0,
            average_confidence: 0.92,
            real_time_factor: 0.15,
        };
        assert_eq!(assessment.grade(), "Excellent");

        let poor_assessment = QualityAssessment {
            overall_score: 0.5,
            meets_profile_requirements: false,
            profile: Profile::Low,
            average_latency_ms: 300.0,
            average_confidence: 0.4,
            real_time_factor: 0.8,
        };
        assert_eq!(poor_assessment.grade(), "Poor");
    }

    #[test]
    fn test_word_segment() {
        let segment = WordSegment {
            word: "hello".to_string(),
            start_time: 1.0,
            end_time: 1.5,
            confidence: 0.95,
        };
        assert_eq!(segment.word, "hello");
        assert_eq!(segment.start_time, 1.0);
        assert!(segment.confidence > 0.9);
    }
}