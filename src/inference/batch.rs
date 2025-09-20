//! Batch processing for improved throughput

use super::{InferenceEngine, InferenceResult};
use crate::audio::AudioBuffer;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

/// Batch processing request
#[derive(Debug)]
pub struct BatchRequest {
    pub id: String,
    pub audio: AudioBuffer,
    pub response_tx: tokio::sync::oneshot::Sender<Result<InferenceResult>>,
}

/// Batch processor for efficient inference
pub struct BatchProcessor {
    engine: Arc<Mutex<InferenceEngine>>,
    batch_size: usize,
    max_wait_time: Duration,
    request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
}

impl BatchProcessor {
    pub fn new(engine: InferenceEngine, batch_size: usize, max_wait_ms: u64) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            batch_size,
            max_wait_time: Duration::from_millis(max_wait_ms),
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Start the batch processing loop
    pub async fn start(&self) -> Result<()> {
        let engine = Arc::clone(&self.engine);
        let queue = Arc::clone(&self.request_queue);
        let batch_size = self.batch_size;
        let max_wait_time = self.max_wait_time;

        tokio::spawn(async move {
            loop {
                // Collect batch of requests
                let batch = Self::collect_batch(&queue, batch_size, max_wait_time).await;

                if !batch.is_empty() {
                    // Process batch
                    let results = Self::process_batch(&engine, batch).await;

                    // Send results back to requesters
                    for (request, result) in results {
                        let _ = request.response_tx.send(result);
                    }
                }

                // Small delay to prevent busy waiting
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        Ok(())
    }

    /// Add a request to the batch queue
    pub async fn add_request(&self, request: BatchRequest) -> Result<()> {
        let mut queue = self.request_queue.lock().await;
        queue.push_back(request);
        Ok(())
    }

    /// Process a single request (bypasses batching)
    pub async fn process_single(&self, audio: &AudioBuffer) -> Result<InferenceResult> {
        let mut engine = self.engine.lock().await;

        // Convert audio to model input format
        let inputs = self.audio_to_inputs(audio)?;

        // Run inference
        engine.run(&inputs)
    }

    /// Collect a batch of requests
    async fn collect_batch(
        queue: &Arc<Mutex<VecDeque<BatchRequest>>>,
        batch_size: usize,
        max_wait_time: Duration,
    ) -> Vec<BatchRequest> {
        let mut batch = Vec::new();
        let start_time = tokio::time::Instant::now();

        loop {
            {
                let mut queue_guard = queue.lock().await;
                while batch.len() < batch_size {
                    if let Some(request) = queue_guard.pop_front() {
                        batch.push(request);
                    } else {
                        break;
                    }
                }
            }

            // Break if we have a full batch or exceeded wait time
            if batch.len() >= batch_size || start_time.elapsed() >= max_wait_time {
                break;
            }

            // Break if we have at least one request and some time has passed
            if !batch.is_empty() && start_time.elapsed() >= Duration::from_millis(10) {
                break;
            }

            // Short sleep to avoid busy waiting
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        batch
    }

    /// Process a batch of requests
    async fn process_batch(
        engine: &Arc<Mutex<InferenceEngine>>,
        batch: Vec<BatchRequest>,
    ) -> Vec<(BatchRequest, Result<InferenceResult>)> {
        let mut results = Vec::new();

        // For now, process each request individually
        // In a real implementation, this would batch the inputs together
        for request in batch {
            let result = {
                let mut engine_guard = engine.lock().await;
                Self::process_single_request(&mut engine_guard, &request.audio)
            };

            results.push((request, result));
        }

        results
    }

    /// Process a single request
    fn process_single_request(
        engine: &mut InferenceEngine,
        audio: &AudioBuffer,
    ) -> Result<InferenceResult> {
        // Convert audio to model inputs
        let inputs = Self::audio_to_inputs_sync(audio)?;

        // Run inference
        engine.run(&inputs)
    }

    /// Convert audio buffer to model inputs (async version)
    fn audio_to_inputs(&self, audio: &AudioBuffer) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        Self::audio_to_inputs_sync(audio)
    }

    /// Convert audio buffer to model inputs (sync version)
    fn audio_to_inputs_sync(audio: &AudioBuffer) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        let mut inputs = std::collections::HashMap::new();

        // For now, just pass raw audio
        // In a real implementation, this would extract mel-spectrograms
        inputs.insert("audio".to_string(), audio.data.clone());

        Ok(inputs)
    }

    /// Get current queue size
    pub async fn queue_size(&self) -> usize {
        let queue = self.request_queue.lock().await;
        queue.len()
    }

    /// Clear the queue
    pub async fn clear_queue(&self) -> usize {
        let mut queue = self.request_queue.lock().await;
        let size = queue.len();
        queue.clear();
        size
    }
}

/// Batch processing statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub total_requests: u64,
    pub total_batches: u64,
    pub average_batch_size: f32,
    pub average_processing_time_ms: f32,
    pub queue_length: usize,
}

/// Batch statistics collector
pub struct BatchStatsCollector {
    total_requests: u64,
    total_batches: u64,
    total_processing_time_ms: f64,
    processor: Arc<BatchProcessor>,
}

impl BatchStatsCollector {
    pub fn new(processor: Arc<BatchProcessor>) -> Self {
        Self {
            total_requests: 0,
            total_batches: 0,
            total_processing_time_ms: 0.0,
            processor,
        }
    }

    /// Record a processed batch
    pub fn record_batch(&mut self, batch_size: usize, processing_time_ms: f32) {
        self.total_requests += batch_size as u64;
        self.total_batches += 1;
        self.total_processing_time_ms += processing_time_ms as f64;
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> BatchStats {
        let queue_length = self.processor.queue_size().await;

        BatchStats {
            total_requests: self.total_requests,
            total_batches: self.total_batches,
            average_batch_size: if self.total_batches > 0 {
                self.total_requests as f32 / self.total_batches as f32
            } else {
                0.0
            },
            average_processing_time_ms: if self.total_batches > 0 {
                self.total_processing_time_ms as f32 / self.total_batches as f32
            } else {
                0.0
            },
            queue_length,
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.total_requests = 0;
        self.total_batches = 0;
        self.total_processing_time_ms = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferenceEngine, InferenceConfig};

    #[tokio::test]
    async fn test_batch_processor() {
        // Create test engine
        let config = InferenceConfig::default();
        let engine = InferenceEngine::new(config).unwrap();

        let processor = BatchProcessor::new(engine, 2, 100);

        // Test queue operations
        assert_eq!(processor.queue_size().await, 0);

        let cleared = processor.clear_queue().await;
        assert_eq!(cleared, 0);
    }

    #[test]
    fn test_audio_to_inputs() {
        let audio = AudioBuffer {
            data: vec![0.1, 0.2, 0.3],
            sample_rate: 16000,
            channels: 1,
            timestamp: std::time::Instant::now(),
        };

        let inputs = BatchProcessor::audio_to_inputs_sync(&audio).unwrap();
        assert!(inputs.contains_key("audio"));
        assert_eq!(inputs["audio"], vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_batch_stats() {
        let config = InferenceConfig::default();
        let engine = InferenceEngine::new(config).unwrap();
        let processor = Arc::new(BatchProcessor::new(engine, 2, 100));

        let mut stats_collector = BatchStatsCollector::new(processor);

        // Record some batches
        stats_collector.record_batch(2, 50.0);
        stats_collector.record_batch(3, 75.0);

        assert_eq!(stats_collector.total_requests, 5);
        assert_eq!(stats_collector.total_batches, 2);
    }
}