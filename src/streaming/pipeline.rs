use std::collections::VecDeque;
use std::sync::{Arc, atomic::{AtomicUsize, AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock, Semaphore};
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};
use futures::future::join_all;

use super::multi_stream::{AudioFrame, ProcessedSegment, StreamPriority};
use super::advanced_cache::{AdvancedCache, CacheKey, ModelType, CacheMetadata};
use super::resource_manager::ResourceManager;
use crate::models::{WhisperV3Turbo, NLLB600M};
use crate::gpu::AdaptiveMemoryManager;
use crate::acceleration::ONNXAccelerator;

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub max_parallel_stages: usize,
    pub batch_size: usize,
    pub stage_timeout_ms: u64,
    pub enable_caching: bool,
    pub enable_prefetch: bool,
    pub worker_threads: usize,
    pub queue_capacity: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_parallel_stages: 4,
            batch_size: 8,
            stage_timeout_ms: 1000,
            enable_caching: true,
            enable_prefetch: true,
            worker_threads: 8,
            queue_capacity: 1000,
        }
    }
}

#[derive(Debug, Clone)]
struct PipelineStage {
    name: String,
    stage_type: StageType,
    input_queue: Arc<Mutex<VecDeque<PipelineItem>>>,
    output_queue: Arc<Mutex<VecDeque<PipelineItem>>>,
    workers: Vec<Arc<Mutex<WorkerState>>>,
    semaphore: Arc<Semaphore>,
}

#[derive(Debug, Clone)]
enum StageType {
    AudioProcessing,
    SpeechRecognition,
    Translation,
    PostProcessing,
    Output,
}

#[derive(Debug, Clone)]
struct PipelineItem {
    id: String,
    stream_id: String,
    priority: StreamPriority,
    data: PipelineData,
    timestamp: Instant,
    stage_times: Vec<(String, Duration)>,
}

#[derive(Debug, Clone)]
enum PipelineData {
    Audio(Vec<f32>),
    Text(String),
    Translated(String, String),
    Processed(ProcessedSegment),
}

#[derive(Debug)]
struct WorkerState {
    id: usize,
    active: AtomicBool,
    items_processed: AtomicUsize,
    total_processing_time: Arc<Mutex<Duration>>,
}

pub struct ConcurrentPipeline {
    stages: Arc<RwLock<Vec<PipelineStage>>>,
    config: PipelineConfig,
    cache: Arc<AdvancedCache>,
    resource_manager: Arc<ResourceManager>,
    whisper_pool: Arc<Mutex<Vec<WhisperV3Turbo>>>,
    translator_pool: Arc<Mutex<Vec<NLLB600M>>>,
    memory_manager: Arc<AdaptiveMemoryManager>,
    accelerator: Arc<ONNXAccelerator>,
    shutdown_signal: Arc<AtomicBool>,
    worker_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    metrics: Arc<PipelineMetrics>,
}

struct PipelineMetrics {
    total_items: AtomicUsize,
    completed_items: AtomicUsize,
    failed_items: AtomicUsize,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    avg_latency_ms: Arc<RwLock<f64>>,
    throughput_per_sec: Arc<RwLock<f64>>,
}

impl ConcurrentPipeline {
    pub async fn new(
        config: PipelineConfig,
        cache: Arc<AdvancedCache>,
        resource_manager: Arc<ResourceManager>,
        memory_manager: Arc<AdaptiveMemoryManager>,
        accelerator: Arc<ONNXAccelerator>,
    ) -> Result<Self> {
        let stages = Self::create_stages(&config).await?;

        let whisper_pool = Self::create_whisper_pool(
            config.worker_threads / 2,
            memory_manager.clone(),
            accelerator.clone(),
        ).await?;

        let translator_pool = Self::create_translator_pool(
            config.worker_threads / 2,
            memory_manager.clone(),
            accelerator.clone(),
        ).await?;

        let pipeline = Self {
            stages: Arc::new(RwLock::new(stages)),
            config,
            cache,
            resource_manager,
            whisper_pool: Arc::new(Mutex::new(whisper_pool)),
            translator_pool: Arc::new(Mutex::new(translator_pool)),
            memory_manager,
            accelerator,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            worker_handles: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(PipelineMetrics::new()),
        };

        pipeline.start_workers().await?;

        Ok(pipeline)
    }

    async fn create_stages(config: &PipelineConfig) -> Result<Vec<PipelineStage>> {
        let stage_types = vec![
            StageType::AudioProcessing,
            StageType::SpeechRecognition,
            StageType::Translation,
            StageType::PostProcessing,
            StageType::Output,
        ];

        let mut stages = Vec::new();
        let mut prev_queue = Arc::new(Mutex::new(VecDeque::with_capacity(config.queue_capacity)));

        for stage_type in stage_types {
            let name = format!("{:?}", stage_type);
            let output_queue = Arc::new(Mutex::new(VecDeque::with_capacity(config.queue_capacity)));

            let mut workers = Vec::new();
            for i in 0..config.worker_threads {
                workers.push(Arc::new(Mutex::new(WorkerState {
                    id: i,
                    active: AtomicBool::new(true),
                    items_processed: AtomicUsize::new(0),
                    total_processing_time: Arc::new(Mutex::new(Duration::ZERO)),
                })));
            }

            let stage = PipelineStage {
                name: name.clone(),
                stage_type,
                input_queue: prev_queue.clone(),
                output_queue: output_queue.clone(),
                workers,
                semaphore: Arc::new(Semaphore::new(config.max_parallel_stages)),
            };

            stages.push(stage);
            prev_queue = output_queue;
        }

        Ok(stages)
    }

    async fn create_whisper_pool(
        size: usize,
        memory_manager: Arc<AdaptiveMemoryManager>,
        accelerator: Arc<ONNXAccelerator>,
    ) -> Result<Vec<WhisperV3Turbo>> {
        let mut pool = Vec::new();

        for _ in 0..size {
            let model = WhisperV3Turbo::new(
                memory_manager.clone(),
                accelerator.clone(),
            ).await?;
            pool.push(model);
        }

        Ok(pool)
    }

    async fn create_translator_pool(
        size: usize,
        memory_manager: Arc<AdaptiveMemoryManager>,
        accelerator: Arc<ONNXAccelerator>,
    ) -> Result<Vec<NLLB600M>> {
        let mut pool = Vec::new();

        for _ in 0..size {
            let model = NLLB600M::new(
                memory_manager.clone(),
                accelerator.clone(),
            ).await?;
            pool.push(model);
        }

        Ok(pool)
    }

    async fn start_workers(&self) -> Result<()> {
        let mut handles = Vec::new();
        let stages = self.stages.read().await;

        for (stage_idx, stage) in stages.iter().enumerate() {
            for worker_idx in 0..self.config.worker_threads {
                let stage_clone = stage.clone();
                let shutdown_signal = self.shutdown_signal.clone();
                let cache = self.cache.clone();
                let whisper_pool = self.whisper_pool.clone();
                let translator_pool = self.translator_pool.clone();
                let metrics = self.metrics.clone();

                let handle = tokio::spawn(async move {
                    Self::worker_loop(
                        stage_clone,
                        worker_idx,
                        shutdown_signal,
                        cache,
                        whisper_pool,
                        translator_pool,
                        metrics,
                    ).await;
                });

                handles.push(handle);
            }
        }

        *self.worker_handles.lock().await = handles;
        Ok(())
    }

    async fn worker_loop(
        stage: PipelineStage,
        worker_idx: usize,
        shutdown_signal: Arc<AtomicBool>,
        cache: Arc<AdvancedCache>,
        whisper_pool: Arc<Mutex<Vec<WhisperV3Turbo>>>,
        translator_pool: Arc<Mutex<Vec<NLLB600M>>>,
        metrics: Arc<PipelineMetrics>,
    ) {
        while !shutdown_signal.load(Ordering::Relaxed) {
            let permit = match stage.semaphore.try_acquire() {
                Ok(p) => p,
                Err(_) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
            };

            let item = {
                let mut queue = stage.input_queue.lock().await;
                queue.pop_front()
            };

            if let Some(mut item) = item {
                let start = Instant::now();

                let result = Self::process_stage_item(
                    &stage.stage_type,
                    &mut item,
                    &cache,
                    &whisper_pool,
                    &translator_pool,
                    &metrics,
                ).await;

                match result {
                    Ok(processed_item) => {
                        let mut output_queue = stage.output_queue.lock().await;
                        processed_item.stage_times.push((stage.name.clone(), start.elapsed()));
                        output_queue.push_back(processed_item);

                        if let Some(worker) = stage.workers.get(worker_idx) {
                            let worker = worker.lock().await;
                            worker.items_processed.fetch_add(1, Ordering::Relaxed);
                            let mut time = worker.total_processing_time.lock().await;
                            *time += start.elapsed();
                        }
                    }
                    Err(e) => {
                        log::error!("Pipeline stage {} error: {}", stage.name, e);
                        metrics.failed_items.fetch_add(1, Ordering::Relaxed);
                    }
                }

                drop(permit);
            } else {
                drop(permit);
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    async fn process_stage_item(
        stage_type: &StageType,
        item: &mut PipelineItem,
        cache: &Arc<AdvancedCache>,
        whisper_pool: &Arc<Mutex<Vec<WhisperV3Turbo>>>,
        translator_pool: &Arc<Mutex<Vec<NLLB600M>>>,
        metrics: &Arc<PipelineMetrics>,
    ) -> Result<PipelineItem> {
        match stage_type {
            StageType::AudioProcessing => {
                if let PipelineData::Audio(audio_data) = &item.data {
                    let processed = Self::process_audio(audio_data).await?;
                    item.data = PipelineData::Audio(processed);
                }
            }
            StageType::SpeechRecognition => {
                if let PipelineData::Audio(audio_data) = &item.data {
                    let cache_key = CacheKey {
                        content_hash: format!("{:?}", audio_data.len()),
                        model_type: ModelType::Whisper,
                        params: item.stream_id.clone(),
                    };

                    if let Some(cached) = cache.get(&cache_key).await {
                        metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                        let text = String::from_utf8_lossy(&cached.data).to_string();
                        item.data = PipelineData::Text(text);
                    } else {
                        metrics.cache_misses.fetch_add(1, Ordering::Relaxed);

                        let mut pool = whisper_pool.lock().await;
                        if let Some(mut whisper) = pool.pop() {
                            let transcription = whisper.transcribe_streaming(audio_data, 16000).await?;
                            pool.push(whisper);

                            let metadata = CacheMetadata {
                                model_type: "whisper".to_string(),
                                input_hash: format!("{:?}", audio_data.len()),
                                processing_time_ms: 0,
                                confidence: transcription.confidence,
                                language_pair: None,
                                compression_ratio: 1.0,
                            };

                            cache.put(
                                cache_key,
                                transcription.text.as_bytes().to_vec(),
                                metadata,
                            ).await?;

                            item.data = PipelineData::Text(transcription.text);
                        }
                    }
                }
            }
            StageType::Translation => {
                if let PipelineData::Text(text) = &item.data {
                    let cache_key = CacheKey {
                        content_hash: format!("{:?}", text),
                        model_type: ModelType::Translation,
                        params: "en-es".to_string(),
                    };

                    if let Some(cached) = cache.get(&cache_key).await {
                        metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                        let translated = String::from_utf8_lossy(&cached.data).to_string();
                        item.data = PipelineData::Translated(text.clone(), translated);
                    } else {
                        metrics.cache_misses.fetch_add(1, Ordering::Relaxed);

                        let mut pool = translator_pool.lock().await;
                        if let Some(mut translator) = pool.pop() {
                            let translation = translator.translate(text, "en", "es").await?;
                            pool.push(translator);

                            let metadata = CacheMetadata {
                                model_type: "translation".to_string(),
                                input_hash: format!("{:?}", text),
                                processing_time_ms: 0,
                                confidence: translation.confidence,
                                language_pair: Some(("en".to_string(), "es".to_string())),
                                compression_ratio: 1.0,
                            };

                            cache.put(
                                cache_key,
                                translation.text.as_bytes().to_vec(),
                                metadata,
                            ).await?;

                            item.data = PipelineData::Translated(text.clone(), translation.text);
                        }
                    }
                }
            }
            StageType::PostProcessing => {
                if let PipelineData::Translated(original, translated) = &item.data {
                    let segment = ProcessedSegment {
                        stream_id: item.stream_id.clone(),
                        original_text: original.clone(),
                        translated_text: translated.clone(),
                        timestamp: item.timestamp,
                        confidence: 0.95,
                        processing_time_ms: item.timestamp.elapsed().as_millis() as u64,
                    };
                    item.data = PipelineData::Processed(segment);
                }
            }
            StageType::Output => {
                metrics.completed_items.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(item.clone())
    }

    async fn process_audio(audio_data: &[f32]) -> Result<Vec<f32>> {
        Ok(audio_data.to_vec())
    }

    pub async fn submit(&self, stream_id: String, audio_frame: AudioFrame, priority: StreamPriority) -> Result<()> {
        let item = PipelineItem {
            id: uuid::Uuid::new_v4().to_string(),
            stream_id,
            priority,
            data: PipelineData::Audio(audio_frame.data),
            timestamp: Instant::now(),
            stage_times: Vec::new(),
        };

        let stages = self.stages.read().await;
        if let Some(first_stage) = stages.first() {
            let mut queue = first_stage.input_queue.lock().await;

            let insert_pos = match priority {
                StreamPriority::Critical => 0,
                StreamPriority::High => queue.len() / 4,
                StreamPriority::Normal => queue.len() / 2,
                StreamPriority::Low => queue.len(),
            };

            if insert_pos >= queue.len() {
                queue.push_back(item);
            } else {
                queue.insert(insert_pos, item);
            }

            self.metrics.total_items.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    pub async fn get_metrics(&self) -> PipelineMetricsReport {
        let total = self.metrics.total_items.load(Ordering::Relaxed);
        let completed = self.metrics.completed_items.load(Ordering::Relaxed);
        let failed = self.metrics.failed_items.load(Ordering::Relaxed);
        let cache_hits = self.metrics.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.metrics.cache_misses.load(Ordering::Relaxed);

        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f32 / (cache_hits + cache_misses) as f32
        } else {
            0.0
        };

        PipelineMetricsReport {
            total_items: total,
            completed_items: completed,
            failed_items: failed,
            pending_items: total - completed - failed,
            cache_hit_rate,
            avg_latency_ms: *self.metrics.avg_latency_ms.read().await,
            throughput_per_sec: *self.metrics.throughput_per_sec.read().await,
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        let mut handles = self.worker_handles.lock().await;
        for handle in handles.drain(..) {
            handle.await?;
        }

        log::info!("Pipeline shutdown complete");
        Ok(())
    }
}

impl PipelineMetrics {
    fn new() -> Self {
        Self {
            total_items: AtomicUsize::new(0),
            completed_items: AtomicUsize::new(0),
            failed_items: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            avg_latency_ms: Arc::new(RwLock::new(0.0)),
            throughput_per_sec: Arc::new(RwLock::new(0.0)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineMetricsReport {
    pub total_items: usize,
    pub completed_items: usize,
    pub failed_items: usize,
    pub pending_items: usize,
    pub cache_hit_rate: f32,
    pub avg_latency_ms: f64,
    pub throughput_per_sec: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let cache = Arc::new(AdvancedCache::new(100, super::super::advanced_cache::EvictionPolicy::LRU, true).await.unwrap());
        let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
        let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

        let resource_manager = Arc::new(ResourceManager::new(
            super::super::resource_manager::ResourceLimits {
                max_cpu_percent: 80.0,
                max_memory_mb: 8192,
                max_gpu_memory_mb: 4096,
                max_threads: 16,
                reserved_cpu_percent: 20.0,
                reserved_memory_mb: 1024,
            },
            memory_manager.clone(),
            true,
        ).await.unwrap());

        let pipeline = ConcurrentPipeline::new(
            config,
            cache,
            resource_manager,
            memory_manager,
            accelerator,
        ).await.unwrap();

        pipeline.shutdown().await.unwrap();
    }
}