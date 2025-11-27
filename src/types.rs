//! Core types and data structures

use serde::{Deserialize, Serialize};

/// Performance profile configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Profile {
    /// Low resource profile: <500ms latency, ~1.2GB memory
    /// - ASR: Distil-Whisper Large v3 INT8
    /// - Translation: MADLAD-400 INT8
    Low,

    /// Medium resource profile: <300ms latency, ~3.2GB VRAM
    /// - ASR: Parakeet TDT 0.6B FP16
    /// - Translation: NLLB-200 FP16
    #[default]
    Medium,

    /// High resource profile: <150ms latency, ~7GB VRAM
    /// - ASR: Canary Qwen 2.5B BF16
    /// - Translation: MADLAD-400 BF16 or Claude API
    High,
}


/// VAD Engine Type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum VadType {
    /// Silero VAD (v4/Legacy)
    Silero,
    /// Silero VAD v5 (New)
    #[default]
    SileroV5,
    /// TEN VAD (High Accuracy)
    TenVad,
    /// Cobra VAD (Enterprise)
    Cobra,
}


/// Language Detection Strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum LanguageDetectionStrategy {
    /// Prioritize ASR-detected language, fallback to text-based
    #[default]
    Hybrid,
    /// Use ASR-detected language only (if available)
    AsrOnly,
    /// Use text-based detection only (ignore ASR)
    TextOnly,
}


/// Main translator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslatorConfig {
    /// Performance profile
    pub profile: Profile,

    /// Source language (auto-detect if empty)
    pub source_language: String,

    /// Target language for translation
    pub target_language: String,

    /// Audio sample rate (Hz)
    pub sample_rate: u32,

    /// Audio chunk size for processing
    pub chunk_size: usize,

    /// Enable Claude API for translation (High profile only)
    pub use_claude_api: bool,

    /// Claude API key (if enabled)
    pub claude_api_key: Option<String>,

    /// Path to model files
    pub model_path: String,

    /// Hardware acceleration preferences
    pub acceleration: AccelerationConfig,

    /// KV Cache optimization settings
    /// KV Cache optimization settings
    pub kv_cache: KvCacheConfig,

    /// VAD Engine selection
    pub vad_type: VadType,

    /// Language detection strategy
    pub language_detection_strategy: LanguageDetectionStrategy,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Medium,
            source_language: String::new(), // Auto-detect
            target_language: "en".to_string(),
            sample_rate: 16000,
            chunk_size: 1600, // 100ms at 16kHz
            use_claude_api: false,
            claude_api_key: None,
            model_path: "./models".to_string(),
            acceleration: AccelerationConfig::default(),
            kv_cache: KvCacheConfig::default(),
            vad_type: VadType::default(),
            language_detection_strategy: LanguageDetectionStrategy::default(),
        }
    }
}

/// Hardware acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationConfig {
    /// Prefer CUDA (NVIDIA)
    pub cuda: bool,

    /// Prefer TensorRT (NVIDIA)
    pub tensorrt: bool,

    /// Prefer CoreML (Apple)
    pub coreml: bool,

    /// Prefer OpenVINO (Intel)
    pub openvino: bool,

    /// Prefer DirectML (Windows)
    pub directml: bool,

    /// Device ID for multi-GPU systems
    pub device_id: Option<u32>,

    /// Enable Flash Attention 3 (or 2) optimization if supported
    pub flash_attention: bool,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            // Auto-detect based on platform
            cuda: cfg!(feature = "cuda"),
            tensorrt: cfg!(feature = "tensorrt"),
            coreml: cfg!(feature = "coreml"),
            openvino: cfg!(feature = "openvino"),
            directml: cfg!(feature = "directml"),
            device_id: None,
            flash_attention: true, // Enable by default if supported
        }
    }
}

/// KV Cache optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Enable KV cache optimization
    pub enabled: bool,

    /// Use lower precision for KV cache (e.g., FP16/INT8)
    pub low_precision: bool,

    /// Maximum cache size (in tokens) before eviction/sliding window
    pub max_cache_size: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            low_precision: true,
            max_cache_size: 4096,
        }
    }
}

/// Translation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    /// Original transcription
    pub transcription: Option<String>,

    /// Translated text
    pub translation: Option<String>,

    /// Detected source language
    pub source_language: Option<String>,

    /// Target language
    pub target_language: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// End-to-end latency in milliseconds
    pub latency_ms: u64,
}

impl TranslationResult {
    /// Create a result indicating silence (no speech detected)
    pub fn silence() -> Self {
        Self {
            transcription: None,
            translation: None,
            source_language: None,
            target_language: String::new(),
            confidence: 0.0,
            latency_ms: 0,
        }
    }

    /// Create an empty result (speech detected but no transcription)
    pub fn empty() -> Self {
        Self {
            transcription: Some(String::new()),
            translation: Some(String::new()),
            source_language: None,
            target_language: String::new(),
            confidence: 0.0,
            latency_ms: 0,
        }
    }
}

/// ASR transcription result
#[derive(Debug, Clone)]
pub struct Transcription {
    /// Transcribed text
    pub text: String,

    /// Confidence score
    pub confidence: f32,

    /// Word-level timestamps (optional)
    pub timestamps: Option<Vec<WordTimestamp>>,

    /// Detected language (optional, from ASR model)
    pub detected_language: Option<String>,
}

/// Word-level timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// Word text
    pub word: String,

    /// Start time in seconds
    pub start: f32,

    /// End time in seconds
    pub end: f32,

    /// Confidence for this word
    pub confidence: f32,
}

/// Language detection result
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    /// Detected language code (ISO 639-1 or 639-3)
    pub language: String,

    /// Confidence score
    pub confidence: f32,

    /// Alternative detections
    pub alternatives: Vec<(String, f32)>,
}
