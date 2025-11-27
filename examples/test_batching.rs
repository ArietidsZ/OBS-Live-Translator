use obs_live_translator::{
    batching::{BatchProcessor, InferenceRequest},
    Result,
};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    tracing::info!("Starting BatchProcessor test...");

    // Configure batch processor: max 5 items, max wait 100ms
    let (processor, tx) = BatchProcessor::new(5, 100);

    // Spawn processor in background
    tokio::spawn(async move {
        processor.run().await;
    });

    // Test Case 1: Fill batch immediately
    tracing::info!("Test 1: Sending 5 requests (should trigger immediate batch)");
    let start = Instant::now();
    let mut handles = vec![];

    for _i in 0..5 {
        let (resp_tx, resp_rx) = oneshot::channel();
        let req = InferenceRequest {
            audio_data: vec![0.0; 100], // Dummy audio
            response_tx: resp_tx,
            created_at: Instant::now(),
        };
        tx.send(req).await.unwrap();
        handles.push(resp_rx);
    }

    // Wait for all responses
    for h in handles {
        h.await.unwrap().unwrap();
    }

    let duration = start.elapsed();
    tracing::info!(
        "Test 1 completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Test Case 2: Timeout trigger
    tracing::info!("Test 2: Sending 2 requests (should trigger timeout after 100ms)");
    let start = Instant::now();
    let mut handles = vec![];

    for _i in 0..2 {
        let (resp_tx, resp_rx) = oneshot::channel();
        let req = InferenceRequest {
            audio_data: vec![0.0; 100],
            response_tx: resp_tx,
            created_at: Instant::now(),
        };
        tx.send(req).await.unwrap();
        handles.push(resp_rx);
    }

    // Wait for all responses
    for h in handles {
        h.await.unwrap().unwrap();
    }

    let duration = start.elapsed();
    tracing::info!(
        "Test 2 completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    assert!(
        duration >= Duration::from_millis(100),
        "Timeout should have waited at least 100ms"
    );

    Ok(())
}
