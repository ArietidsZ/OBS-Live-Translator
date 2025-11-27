use crate::{types::TranslationResult, Result};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info};

/// Request for inference processing
#[derive(Debug)]
pub struct InferenceRequest {
    pub audio_data: Vec<f32>,
    pub response_tx: oneshot::Sender<Result<TranslationResult>>,
    pub created_at: Instant,
}

/// Dynamic batch processor for inference requests
pub struct BatchProcessor {
    max_batch_size: usize,
    max_wait: Duration,
    request_rx: mpsc::Receiver<InferenceRequest>,
    pending_requests: Vec<InferenceRequest>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(max_batch_size: usize, max_wait_ms: u64) -> (Self, mpsc::Sender<InferenceRequest>) {
        let (tx, rx) = mpsc::channel(100);

        let processor = Self {
            max_batch_size,
            max_wait: Duration::from_millis(max_wait_ms),
            request_rx: rx,
            pending_requests: Vec::with_capacity(max_batch_size),
        };

        (processor, tx)
    }

    /// Run the batch processor loop
    pub async fn run(mut self) {
        info!(
            "Starting BatchProcessor (batch_size={}, wait={}ms)",
            self.max_batch_size,
            self.max_wait.as_millis()
        );

        loop {
            // Determine timeout for the current batch
            let timeout = if self.pending_requests.is_empty() {
                // If empty, wait indefinitely for first item
                Duration::from_secs(3600)
            } else {
                // If has items, wait remaining time until max_wait
                let oldest_age = self.pending_requests[0].created_at.elapsed();
                if oldest_age >= self.max_wait {
                    Duration::ZERO
                } else {
                    self.max_wait - oldest_age
                }
            };

            tokio::select! {
                // Receive new request
                Some(req) = self.request_rx.recv() => {
                    self.pending_requests.push(req);

                    // Check if batch is full
                    if self.pending_requests.len() >= self.max_batch_size {
                        self.process_batch().await;
                    }
                }

                // Timeout reached (batch latency limit)
                _ = tokio::time::sleep(timeout), if !self.pending_requests.is_empty() => {
                    debug!("Batch timeout reached, processing {} items", self.pending_requests.len());
                    self.process_batch().await;
                }

                else => break, // Channel closed
            }
        }
    }

    /// Process the current batch of requests
    async fn process_batch(&mut self) {
        if self.pending_requests.is_empty() {
            return;
        }

        // Drain requests to process
        let batch: Vec<_> = self.pending_requests.drain(..).collect();
        let batch_size = batch.len();
        let start = Instant::now();

        debug!("Processing batch of {} requests", batch_size);

        // TODO: Implement actual batched inference here
        // For now, process sequentially to simulate functionality
        for req in batch {
            // Simulate processing
            // In real implementation, we would stack tensors and run one inference

            // Placeholder result
            let result = TranslationResult::empty();

            // Send response back
            let _ = req.response_tx.send(Ok(result));
        }

        debug!(
            "Batch processed in {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
