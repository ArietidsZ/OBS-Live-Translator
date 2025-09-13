//! High-Performance OBS Live Translator Core
//! 
//! This library provides ultra-low-latency real-time speech recognition and translation
//! optimized for streaming and VTuber applications.

#![deny(missing_docs)]
#![allow(clippy::module_name_repetitions)]

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod audio;
pub mod gpu;
pub mod inference;
pub mod translation;
pub mod cache;
pub mod networking;
pub mod bindings;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Core translator instance for maximum performance
#[derive(Clone)]
pub struct HighPerformanceTranslator {
    audio_processor: Arc<audio::StreamingAudioProcessor>,
    gpu_manager: Arc<gpu::OptimizedGpuManager>,
    asr_engine: Arc<inference::ParallelAsrEngine>,
    translation_engine: Arc<translation::BatchedTranslationEngine>,
    cache_manager: Arc<cache::IntelligentCache>,
    config: Arc<RwLock<TranslatorConfig>>,
}

/// Configuration for maximum performance
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
            target_latency_ms: 50, // Ultra-low latency target
            max_gpu_memory_mb: 8192, // 8GB GPU memory
            thread_count: num_cpus::get(),
            aggressive_mode: true,
            batch_size: 8,
            buffer_size_samples: 1024,
        }
    }
}

impl HighPerformanceTranslator {
    /// Initialize translator with maximum performance optimizations
    pub async fn new(config: TranslatorConfig) -> Result<Self> {
        tracing::info!("Initializing high-performance translator");
        
        // Initialize GPU manager first for optimal VRAM allocation
        let gpu_manager = Arc::new(gpu::OptimizedGpuManager::new(&config).await?);
        
        // Initialize audio processor with SIMD optimizations
        let audio_processor = Arc::new(
            audio::StreamingAudioProcessor::new(&config, Arc::clone(&gpu_manager)).await?
        );
        
        // Initialize ASR engine with batching and parallel processing
        let asr_engine = Arc::new(
            inference::ParallelAsrEngine::new(&config, Arc::clone(&gpu_manager)).await?
        );
        
        // Initialize translation engine with optimized batching
        let translation_engine = Arc::new(
            translation::BatchedTranslationEngine::new(&config, Arc::clone(&gpu_manager)).await?
        );
        
        // Initialize intelligent cache for gaming terms and frequent phrases
        let cache_manager = Arc::new(cache::IntelligentCache::new(&config)?);
        
        let config = Arc::new(RwLock::new(config));
        
        Ok(Self {
            audio_processor,
            gpu_manager,
            asr_engine,
            translation_engine,
            cache_manager,
            config,
        })
    }
    
    /// Start real-time translation pipeline
    pub async fn start_translation_pipeline(&self) -> Result<tokio::sync::mpsc::Receiver<TranslationResult>> {
        let (tx, rx) = tokio::sync::mpsc::channel(1000);
        
        let audio_processor = Arc::clone(&self.audio_processor);
        let asr_engine = Arc::clone(&self.asr_engine);
        let translation_engine = Arc::clone(&self.translation_engine);
        let cache_manager = Arc::clone(&self.cache_manager);
        
        // Spawn high-priority processing task
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async move {
                let mut audio_stream = audio_processor.start_capture().await?;
                
                while let Some(audio_chunk) = audio_stream.recv().await {
                    // Check cache first for instant results
                    if let Some(cached_result) = cache_manager.get_translation(&audio_chunk.fingerprint).await {
                        let _ = tx.send(cached_result).await;
                        continue;
                    }
                    
                    // Parallel ASR and translation pipeline
                    let asr_task = asr_engine.transcribe_batch(vec![audio_chunk.clone()]);
                    
                    if let Ok(transcription_results) = asr_task.await {
                        for transcription in transcription_results {
                            if !transcription.text.is_empty() {
                                // Batch translation for efficiency
                                let translation_result = translation_engine
                                    .translate_with_context(transcription, &audio_chunk)
                                    .await?;
                                
                                // Cache successful translations
                                cache_manager.cache_translation(
                                    &audio_chunk.fingerprint,
                                    &translation_result
                                ).await;
                                
                                let _ = tx.send(translation_result).await;
                            }
                        }
                    }
                }
                
                anyhow::Ok(())
            })
        });
        
        Ok(rx)
    }
    
    /// Get real-time performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            audio_latency_ms: self.audio_processor.get_latency_ms().await,
            asr_latency_ms: self.asr_engine.get_average_latency_ms().await,
            translation_latency_ms: self.translation_engine.get_average_latency_ms().await,
            gpu_utilization_percent: self.gpu_manager.get_utilization_percent().await,
            memory_usage_mb: self.gpu_manager.get_memory_usage_mb().await,
            cache_hit_rate: self.cache_manager.get_hit_rate().await,
            total_latency_ms: self.get_total_latency_ms().await,
        }
    }
    
    async fn get_total_latency_ms(&self) -> f32 {
        self.audio_processor.get_latency_ms().await +
        self.asr_engine.get_average_latency_ms().await +
        self.translation_engine.get_average_latency_ms().await
    }
}

/// Real-time performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Audio processing latency in milliseconds
    pub audio_latency_ms: f32,
    /// ASR processing latency in milliseconds
    pub asr_latency_ms: f32,
    /// Translation processing latency in milliseconds
    pub translation_latency_ms: f32,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: f32,
    /// GPU memory usage in MB
    pub memory_usage_mb: f32,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,
    /// Total end-to-end latency in milliseconds
    pub total_latency_ms: f32,
}

/// Translation result with metadata
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
    pub timestamp: std::time::SystemTime,
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
    pub timestamp: std::time::SystemTime,
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