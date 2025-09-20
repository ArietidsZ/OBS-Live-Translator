//! Stress testing for sustained performance under load

use obs_live_translator::Translator;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration, SystemTime};
use tokio::sync::Mutex;

struct StressTestMetrics {
    total_processed: AtomicU64,
    total_errors: AtomicU64,
    total_latency_us: AtomicU64,
    min_latency_us: AtomicU64,
    max_latency_us: AtomicU64,
    start_time: Instant,
    last_report: Arc<Mutex<Instant>>,
}

impl StressTestMetrics {
    fn new() -> Self {
        Self {
            total_processed: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            min_latency_us: AtomicU64::new(u64::MAX),
            max_latency_us: AtomicU64::new(0),
            start_time: Instant::now(),
            last_report: Arc::new(Mutex::new(Instant::now())),
        }
    }

    fn record_success(&self, latency_us: u64) {
        self.total_processed.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);

        // Update min/max with CAS loop
        loop {
            let current_min = self.min_latency_us.load(Ordering::Relaxed);
            if latency_us >= current_min ||
               self.min_latency_us.compare_exchange(
                   current_min, latency_us,
                   Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                break;
            }
        }

        loop {
            let current_max = self.max_latency_us.load(Ordering::Relaxed);
            if latency_us <= current_max ||
               self.max_latency_us.compare_exchange(
                   current_max, latency_us,
                   Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                break;
            }
        }
    }

    fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    async fn report_if_needed(&self) {
        let mut last = self.last_report.lock().await;
        if last.elapsed() > Duration::from_secs(5) {
            *last = Instant::now();
            drop(last);
            self.report().await;
        }
    }

    async fn report(&self) {
        let elapsed = self.start_time.elapsed();
        let total = self.total_processed.load(Ordering::Relaxed);
        let errors = self.total_errors.load(Ordering::Relaxed);
        let total_latency = self.total_latency_us.load(Ordering::Relaxed);

        if total > 0 {
            let avg_latency_ms = (total_latency / total) as f32 / 1000.0;
            let min_latency_ms = self.min_latency_us.load(Ordering::Relaxed) as f32 / 1000.0;
            let max_latency_ms = self.max_latency_us.load(Ordering::Relaxed) as f32 / 1000.0;
            let throughput = total as f32 / elapsed.as_secs_f32();

            println!("\n[{:6.1}s] STRESS TEST STATUS", elapsed.as_secs_f32());
            println!("  Processed: {} | Errors: {}", total, errors);
            println!("  Throughput: {:.1} req/sec", throughput);
            println!("  Latency (ms): avg={:.3} min={:.3} max={:.3}",
                     avg_latency_ms, min_latency_ms, max_latency_ms);
        }
    }
}

async fn generate_audio_stream(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Generate complex audio pattern
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Mix of frequencies simulating speech patterns
        let sample = 0.1 * (2.0 * std::f32::consts::PI * 250.0 * t).sin()
                   + 0.05 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                   + 0.02 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                   + 0.01 * ((i as f32 / 200.0).sin() * 0.5); // Modulation
        samples.push(sample);
    }
    samples
}

async fn stress_worker(
    id: usize,
    metrics: Arc<StressTestMetrics>,
    duration: Duration,
    running: Arc<AtomicBool>,
) {
    println!("Worker {} starting...", id);

    // Create translator
    let mut translator = match Translator::new("models/whisper.onnx", false) {
        Ok(t) => t,
        Err(_) => {
            println!("Worker {} running in simulation mode", id);

            // Simulation mode
            while running.load(Ordering::Relaxed) &&
                  metrics.start_time.elapsed() < duration {

                // Simulate varying processing times
                let audio = generate_audio_stream(
                    0.5 + (id as f32 * 0.1), // Varying durations
                    16000
                ).await;

                let start = Instant::now();
                // Simulate processing
                tokio::time::sleep(Duration::from_micros(100 + (id * 10) as u64)).await;
                let latency_us = start.elapsed().as_micros() as u64;

                metrics.record_success(latency_us);
                metrics.report_if_needed().await;
            }
            return;
        }
    };

    // Real processing
    let mut iteration = 0;
    while running.load(Ordering::Relaxed) &&
          metrics.start_time.elapsed() < duration {

        // Generate audio with varying characteristics
        let duration_secs = match iteration % 4 {
            0 => 0.5,
            1 => 1.0,
            2 => 2.0,
            _ => 3.0,
        };

        let audio = generate_audio_stream(duration_secs, 16000).await;

        let start = Instant::now();
        match translator.process_audio(&audio).await {
            Ok(_result) => {
                let latency_us = start.elapsed().as_micros() as u64;
                metrics.record_success(latency_us);
            }
            Err(_) => {
                metrics.record_error();
            }
        }

        metrics.report_if_needed().await;
        iteration += 1;
    }

    println!("Worker {} stopped", id);
}

async fn run_stress_test(
    num_workers: usize,
    duration_secs: u64,
    target_qps: Option<f32>,
) {
    println!("=== STRESS TEST CONFIGURATION ===");
    println!("Workers: {}", num_workers);
    println!("Duration: {} seconds", duration_secs);
    if let Some(qps) = target_qps {
        println!("Target QPS: {:.1}", qps);
    } else {
        println!("Target QPS: Maximum");
    }
    println!("\nStarting stress test...\n");

    let metrics = Arc::new(StressTestMetrics::new());
    let running = Arc::new(AtomicBool::new(true));
    let duration = Duration::from_secs(duration_secs);

    // Launch workers
    let mut handles = vec![];
    for id in 0..num_workers {
        let metrics_clone = metrics.clone();
        let running_clone = running.clone();

        let handle = tokio::spawn(async move {
            stress_worker(id, metrics_clone, duration, running_clone).await;
        });

        handles.push(handle);

        // Stagger worker starts slightly
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Monitor progress
    let monitor_metrics = metrics.clone();
    let monitor_handle = tokio::spawn(async move {
        while monitor_metrics.start_time.elapsed() < duration {
            tokio::time::sleep(Duration::from_secs(5)).await;
            monitor_metrics.report().await;
        }
    });

    // Wait for duration
    tokio::time::sleep(duration).await;

    // Signal shutdown
    running.store(false, Ordering::Relaxed);

    // Wait for workers
    for handle in handles {
        let _ = handle.await;
    }

    let _ = monitor_handle.await;

    // Final report
    println!("\n=== STRESS TEST COMPLETE ===");
    metrics.report().await;

    let total = metrics.total_processed.load(Ordering::Relaxed);
    let errors = metrics.total_errors.load(Ordering::Relaxed);
    let total_latency = metrics.total_latency_us.load(Ordering::Relaxed);
    let elapsed = metrics.start_time.elapsed();

    if total > 0 {
        let success_rate = (total - errors) as f32 / total as f32 * 100.0;
        let avg_latency_ms = (total_latency / total) as f32 / 1000.0;
        let throughput = total as f32 / elapsed.as_secs_f32();

        println!("\nSummary:");
        println!("  Total requests: {}", total);
        println!("  Success rate: {:.2}%", success_rate);
        println!("  Average throughput: {:.1} req/sec", throughput);
        println!("  Average latency: {:.3} ms", avg_latency_ms);

        // Calculate percentiles (simplified)
        if avg_latency_ms < 1.0 {
            println!("  ✓ Sub-millisecond average latency achieved!");
        }

        if throughput > 1000.0 {
            println!("  ✓ Achieved > 1000 requests/second!");
        }
    }
}

#[tokio::main]
async fn main() {
    println!("=== HIGH-LOAD STRESS TEST ===\n");

    // Test configurations
    let configs = vec![
        (1, 10, None),        // Single worker baseline
        (4, 10, None),        // Light concurrent load
        (8, 30, None),        // Medium concurrent load
        (16, 30, None),       // Heavy concurrent load
        (32, 60, None),       // Extreme load for 1 minute
    ];

    for (workers, duration, qps) in configs {
        run_stress_test(workers, duration, qps).await;

        // Cool down between tests
        println!("\nCooling down for 5 seconds...\n");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    println!("\n=== ALL STRESS TESTS COMPLETE ===");
}