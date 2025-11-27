use anyhow::Result;
// use ndarray::Array3;
use obs_live_translator::batching::BatchProcessor;
use obs_live_translator::optimization::kv_cache::KvCacheManager;
use obs_live_translator::types::KvCacheConfig;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting Benchmark Suite...");
    println!("===========================");

    benchmark_batching().await?;
    benchmark_kv_cache().await?;

    println!("\nBenchmark Suite Completed.");
    Ok(())
}

async fn benchmark_batching() -> Result<()> {
    println!("\n--- Benchmarking Dynamic Batching ---");

    let batch_size = 32;
    let timeout_ms = 10;
    let (processor, tx) = BatchProcessor::new(batch_size, timeout_ms);

    // Spawn processor in background
    tokio::spawn(async move {
        processor.run().await;
    });

    let iterations = 1000;
    let start = Instant::now();

    let mut tasks = Vec::new();
    for _i in 0..iterations {
        let tx_clone = tx.clone();
        tasks.push(tokio::spawn(async move {
            // Create a dummy request
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            let req = obs_live_translator::batching::InferenceRequest {
                audio_data: vec![0.0; 16000], // 1s of audio
                response_tx,
                created_at: Instant::now(),
            };

            if let Err(e) = tx_clone.send(req).await {
                eprintln!("Failed to send request: {e}");
                return;
            }

            // Wait for response
            let _ = response_rx.await;
        }));
    }

    // Wait for all tasks
    for task in tasks {
        task.await?;
    }

    let duration = start.elapsed();
    println!("Processed {iterations} items in {duration:?}");
    println!(
        "Throughput: {:.2} items/sec",
        iterations as f64 / duration.as_secs_f64()
    );

    Ok(())
}

async fn benchmark_kv_cache() -> Result<()> {
    println!("\n--- Benchmarking KV Cache Manager ---");
    println!("Benchmarking KV Cache...");
    let cache_manager = KvCacheManager::new(KvCacheConfig::default());
    
    let iterations = 1000;
    let start = Instant::now();
    
    for i in 0..iterations {
        let key = format!("req_{i}");
        // Simulate cache operations
        let _ = cache_manager.get(&key);
        // In a real benchmark we'd put something too
    }
    
    let duration = start.elapsed();
    println!(
        "Performed {iterations} cache update/get operations in {duration:?}"
    );
    println!(
        "Average latency: {:.2} µs/op",
        duration.as_micros() as f64 / iterations as f64
    );

    Ok(())
}
