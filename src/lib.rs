//! OBS Live Translator Library
//!
//! Real-time speech recognition and translation for streaming applications.

#![deny(missing_docs)]
#![allow(clippy::module_name_repetitions)]

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod models;
pub mod obs;
pub mod hardware;

use anyhow::Result;
use std::sync::Arc;
use std::time::SystemTime;

/// Translation result
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Original transcribed text
    pub original_text: String,
    /// Translated text
    pub translated_text: String,
    /// Source language code
    pub source_language: String,
    /// Target language code
    pub target_language: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Whether result came from cache
    pub from_cache: bool,
    /// Timestamp of audio chunk
    pub timestamp: SystemTime,
    /// Voice-cloned audio data (optional)
    pub cloned_audio: Option<Vec<u8>>,
}

/// Audio chunk for processing
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw audio samples (f32, normalized)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Timestamp of the chunk
    pub timestamp: SystemTime,
    /// Unique fingerprint for caching
    pub fingerprint: u64,
}

impl AudioChunk {
    /// Calculate audio fingerprint for caching
    pub fn calculate_fingerprint(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash first and last few samples for fingerprint
        if self.samples.len() >= 64 {
            self.samples[..32].iter().for_each(|s| s.to_bits().hash(&mut hasher));
            self.samples[self.samples.len()-32..].iter().for_each(|s| s.to_bits().hash(&mut hasher));
        } else {
            self.samples.iter().for_each(|s| s.to_bits().hash(&mut hasher));
        }

        self.sample_rate.hash(&mut hasher);
        self.channels.hash(&mut hasher);

        self.fingerprint = hasher.finish();
    }
}

/// Main translator configuration
#[derive(Debug, Clone)]
pub struct TranslatorConfig {
    /// Target latency in milliseconds
    pub target_latency_ms: u32,
    /// Maximum GPU memory usage in MB
    pub max_gpu_memory_mb: u32,
    /// Number of processing threads
    pub thread_count: usize,
    /// Enable aggressive optimizations
    pub aggressive_mode: bool,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Audio processing buffer size
    pub buffer_size_samples: usize,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 100,
            max_gpu_memory_mb: 4096,
            thread_count: num_cpus::get(),
            aggressive_mode: false,
            batch_size: 8,
            buffer_size_samples: 1024,
        }
    }
}

/// Main translator instance
pub struct Translator {
    config: Arc<TranslatorConfig>,
    hardware: Arc<hardware::HardwareCapabilities>,
}

impl Translator {
    /// Create new translator instance
    pub async fn new(config: TranslatorConfig) -> Result<Self> {
        let hardware = Arc::new(hardware::HardwareDetector::detect().await?);

        tracing::info!(
            "Detected hardware: CPU {:?}, GPU {:?}",
            hardware.cpu.architecture,
            hardware.gpu.as_ref().map(|g| &g.architecture)
        );

        Ok(Self {
            config: Arc::new(config),
            hardware,
        })
    }

    /// Process audio chunk
    pub async fn process_audio(&self, chunk: AudioChunk) -> Result<TranslationResult> {
        let start = std::time::Instant::now();

        // Placeholder for actual processing
        let result = TranslationResult {
            original_text: "Sample text".to_string(),
            translated_text: "Translated text".to_string(),
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            confidence: 0.95,
            processing_time_ms: start.elapsed().as_millis() as f32,
            from_cache: false,
            timestamp: chunk.timestamp,
            cloned_audio: None,
        };

        Ok(result)
    }

    /// Get hardware capabilities
    pub fn get_hardware(&self) -> &hardware::HardwareCapabilities {
        &self.hardware
    }
}