use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};
use uuid::Uuid;

use crate::models::{WhisperV3Turbo, NLLB600M};
use crate::gpu::AdaptiveMemoryManager;
use crate::acceleration::ONNXAccelerator;

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub stream_id: String,
    pub language_pair: (String, String),
    pub priority: StreamPriority,
    pub buffer_size: usize,
    pub max_latency_ms: u32,
    pub enable_voice_activity_detection: bool,
    pub enable_noise_reduction: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

#[derive(Debug)]
pub struct AudioFrame {
    pub timestamp: Instant,
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u8,
}

#[derive(Debug, Clone)]
pub struct ProcessedSegment {
    pub stream_id: String,
    pub original_text: String,
    pub translated_text: String,
    pub timestamp: Instant,
    pub confidence: f32,
    pub processing_time_ms: u64,
}

struct StreamState {
    config: StreamConfig,
    audio_buffer: VecDeque<AudioFrame>,
    last_processed: Instant,
    total_frames: u64,
    total_processing_time: Duration,
    dropped_frames: u64,
    active: bool,
}

pub struct MultiStreamProcessor {
    streams: Arc<RwLock<HashMap<String, Arc<Mutex<StreamState>>>>>,
    whisper: Arc<Mutex<WhisperV3Turbo>>,
    translator: Arc<Mutex<NLLB600M>>,
    memory_manager: Arc<AdaptiveMemoryManager>,
    accelerator: Arc<ONNXAccelerator>,
    worker_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    shutdown_tx: Arc<Mutex<Option<mpsc::Sender<()>>>>,
    max_concurrent_streams: usize,
    thread_pool_size: usize,
    processing_queue: Arc<Mutex<VecDeque<(String, AudioFrame)>>>,
    results_tx: mpsc::UnboundedSender<ProcessedSegment>,
    results_rx: Arc<Mutex<mpsc::UnboundedReceiver<ProcessedSegment>>>,
}

impl MultiStreamProcessor {
    pub async fn new(
        memory_manager: Arc<AdaptiveMemoryManager>,
        accelerator: Arc<ONNXAccelerator>,
        max_concurrent_streams: usize,
        thread_pool_size: usize,
    ) -> Result<Self> {
        let whisper = Arc::new(Mutex::new(
            WhisperV3Turbo::new(memory_manager.clone(), accelerator.clone()).await?
        ));

        let translator = Arc::new(Mutex::new(
            NLLB600M::new(memory_manager.clone(), accelerator.clone()).await?
        ));

        let (results_tx, results_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let processor = Self {
            streams: Arc::new(RwLock::new(HashMap::new())),
            whisper,
            translator,
            memory_manager,
            accelerator,
            worker_handles: Arc::new(Mutex::new(Vec::new())),
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            max_concurrent_streams,
            thread_pool_size,
            processing_queue: Arc::new(Mutex::new(VecDeque::new())),
            results_tx,
            results_rx: Arc::new(Mutex::new(results_rx)),
        };

        processor.start_workers(shutdown_rx).await?;

        Ok(processor)
    }

    async fn start_workers(&self, mut shutdown_rx: mpsc::Receiver<()>) -> Result<()> {
        let mut handles = Vec::new();

        for worker_id in 0..self.thread_pool_size {
            let streams = self.streams.clone();
            let whisper = self.whisper.clone();
            let translator = self.translator.clone();
            let processing_queue = self.processing_queue.clone();
            let results_tx = self.results_tx.clone();
            let memory_manager = self.memory_manager.clone();

            let handle = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = shutdown_rx.recv() => {
                            log::info!("Worker {} shutting down", worker_id);
                            break;
                        }
                        _ = tokio::time::sleep(Duration::from_millis(10)) => {
                            if let Some((stream_id, frame)) = {
                                let mut queue = processing_queue.lock().await;
                                queue.pop_front()
                            } {
                                if let Err(e) = Self::process_frame(
                                    stream_id,
                                    frame,
                                    streams.clone(),
                                    whisper.clone(),
                                    translator.clone(),
                                    results_tx.clone(),
                                    memory_manager.clone(),
                                ).await {
                                    log::error!("Worker {} error processing frame: {}", worker_id, e);
                                }
                            }
                        }
                    }
                }
            });

            handles.push(handle);
        }

        *self.worker_handles.lock().await = handles;
        Ok(())
    }

    async fn process_frame(
        stream_id: String,
        frame: AudioFrame,
        streams: Arc<RwLock<HashMap<String, Arc<Mutex<StreamState>>>>>,
        whisper: Arc<Mutex<WhisperV3Turbo>>,
        translator: Arc<Mutex<NLLB600M>>,
        results_tx: mpsc::UnboundedSender<ProcessedSegment>,
        memory_manager: Arc<AdaptiveMemoryManager>,
    ) -> Result<()> {
        let start_time = Instant::now();

        let stream_state = {
            let streams_guard = streams.read().unwrap();
            streams_guard.get(&stream_id).cloned()
        };

        if let Some(state_arc) = stream_state {
            let mut state = state_arc.lock().await;

            if !state.active {
                return Ok(());
            }

            let memory_available = memory_manager.get_available_memory().await?;
            if memory_available < 512 * 1024 * 1024 {
                log::warn!("Low memory, dropping frame for stream {}", stream_id);
                state.dropped_frames += 1;
                return Ok(());
            }

            let transcription = {
                let mut whisper_lock = whisper.lock().await;
                whisper_lock.transcribe_streaming(&frame.data, frame.sample_rate).await?
            };

            if transcription.text.is_empty() {
                return Ok(());
            }

            let (source_lang, target_lang) = &state.config.language_pair;
            let translation = {
                let mut translator_lock = translator.lock().await;
                translator_lock.translate(
                    &transcription.text,
                    source_lang,
                    target_lang,
                ).await?
            };

            let processing_time = start_time.elapsed();
            state.total_processing_time += processing_time;
            state.last_processed = Instant::now();
            state.total_frames += 1;

            let segment = ProcessedSegment {
                stream_id: stream_id.clone(),
                original_text: transcription.text,
                translated_text: translation.text,
                timestamp: frame.timestamp,
                confidence: transcription.confidence,
                processing_time_ms: processing_time.as_millis() as u64,
            };

            results_tx.send(segment)?;
        }

        Ok(())
    }

    pub async fn add_stream(&self, config: StreamConfig) -> Result<String> {
        let mut streams = self.streams.write().unwrap();

        if streams.len() >= self.max_concurrent_streams {
            return Err(anyhow!("Maximum concurrent streams limit reached"));
        }

        let stream_id = config.stream_id.clone();

        let state = StreamState {
            config,
            audio_buffer: VecDeque::with_capacity(100),
            last_processed: Instant::now(),
            total_frames: 0,
            total_processing_time: Duration::ZERO,
            dropped_frames: 0,
            active: true,
        };

        streams.insert(stream_id.clone(), Arc::new(Mutex::new(state)));

        log::info!("Added stream: {}", stream_id);
        Ok(stream_id)
    }

    pub async fn remove_stream(&self, stream_id: &str) -> Result<()> {
        let mut streams = self.streams.write().unwrap();

        if let Some(state_arc) = streams.get(stream_id) {
            let mut state = state_arc.lock().await;
            state.active = false;
        }

        streams.remove(stream_id);
        log::info!("Removed stream: {}", stream_id);
        Ok(())
    }

    pub async fn submit_audio(&self, stream_id: String, frame: AudioFrame) -> Result<()> {
        let streams = self.streams.read().unwrap();

        if let Some(state_arc) = streams.get(&stream_id) {
            let state = state_arc.lock().await;

            if !state.active {
                return Err(anyhow!("Stream {} is not active", stream_id));
            }

            let priority = state.config.priority;
            drop(state);

            let mut queue = self.processing_queue.lock().await;

            let insert_pos = match priority {
                StreamPriority::Critical => 0,
                StreamPriority::High => queue.len() / 4,
                StreamPriority::Normal => queue.len() / 2,
                StreamPriority::Low => queue.len(),
            };

            if insert_pos >= queue.len() {
                queue.push_back((stream_id, frame));
            } else {
                queue.insert(insert_pos, (stream_id, frame));
            }

            Ok(())
        } else {
            Err(anyhow!("Stream {} not found", stream_id))
        }
    }

    pub async fn get_results(&self) -> Vec<ProcessedSegment> {
        let mut results = Vec::new();
        let mut rx = self.results_rx.lock().await;

        while let Ok(segment) = rx.try_recv() {
            results.push(segment);
        }

        results
    }

    pub async fn get_stream_stats(&self, stream_id: &str) -> Result<StreamStats> {
        let streams = self.streams.read().unwrap();

        if let Some(state_arc) = streams.get(stream_id) {
            let state = state_arc.lock().await;

            let avg_processing_time = if state.total_frames > 0 {
                state.total_processing_time.as_millis() as f64 / state.total_frames as f64
            } else {
                0.0
            };

            Ok(StreamStats {
                stream_id: stream_id.to_string(),
                total_frames: state.total_frames,
                dropped_frames: state.dropped_frames,
                avg_processing_time_ms: avg_processing_time,
                last_processed: state.last_processed,
                active: state.active,
            })
        } else {
            Err(anyhow!("Stream {} not found", stream_id))
        }
    }

    pub async fn get_all_stats(&self) -> Vec<StreamStats> {
        let mut all_stats = Vec::new();
        let streams = self.streams.read().unwrap();

        for (stream_id, _) in streams.iter() {
            if let Ok(stats) = self.get_stream_stats(stream_id).await {
                all_stats.push(stats);
            }
        }

        all_stats
    }

    pub async fn adjust_priority(&self, stream_id: &str, priority: StreamPriority) -> Result<()> {
        let streams = self.streams.read().unwrap();

        if let Some(state_arc) = streams.get(stream_id) {
            let mut state = state_arc.lock().await;
            state.config.priority = priority;
            log::info!("Adjusted priority for stream {} to {:?}", stream_id, priority);
            Ok(())
        } else {
            Err(anyhow!("Stream {} not found", stream_id))
        }
    }

    pub async fn pause_stream(&self, stream_id: &str) -> Result<()> {
        let streams = self.streams.read().unwrap();

        if let Some(state_arc) = streams.get(stream_id) {
            let mut state = state_arc.lock().await;
            state.active = false;
            log::info!("Paused stream: {}", stream_id);
            Ok(())
        } else {
            Err(anyhow!("Stream {} not found", stream_id))
        }
    }

    pub async fn resume_stream(&self, stream_id: &str) -> Result<()> {
        let streams = self.streams.read().unwrap();

        if let Some(state_arc) = streams.get(stream_id) {
            let mut state = state_arc.lock().await;
            state.active = true;
            log::info!("Resumed stream: {}", stream_id);
            Ok(())
        } else {
            Err(anyhow!("Stream {} not found", stream_id))
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _ = tx.send(()).await;
        }

        let mut handles = self.worker_handles.lock().await;
        for handle in handles.drain(..) {
            handle.await?;
        }

        log::info!("MultiStreamProcessor shutdown complete");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StreamStats {
    pub stream_id: String,
    pub total_frames: u64,
    pub dropped_frames: u64,
    pub avg_processing_time_ms: f64,
    pub last_processed: Instant,
    pub active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_stream_creation() {
        let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
        let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

        let processor = MultiStreamProcessor::new(
            memory_manager,
            accelerator,
            10,
            4,
        ).await.unwrap();

        let config = StreamConfig {
            stream_id: "test_stream".to_string(),
            language_pair: ("en".to_string(), "es".to_string()),
            priority: StreamPriority::Normal,
            buffer_size: 1024,
            max_latency_ms: 100,
            enable_voice_activity_detection: true,
            enable_noise_reduction: true,
        };

        let stream_id = processor.add_stream(config).await.unwrap();
        assert_eq!(stream_id, "test_stream");

        processor.shutdown().await.unwrap();
    }
}