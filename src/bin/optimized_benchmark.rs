//! Comprehensive benchmark comparing standard vs optimized implementations

use obs_live_translator::{Translator, TranslationResult};
#[cfg(feature = "simd")]
use obs_live_translator::OptimizedTranslator;

use std::time::{Instant, Duration};
use std::collections::HashMap;

#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    total_samples: usize,
    total_duration: Duration,
    avg_latency_ms: f32,
    min_latency_ms: f32,
    max_latency_ms: f32,
    p95_latency_ms: f32,
    p99_latency_ms: f32,
    throughput: f32,
}

impl BenchmarkResult {
    fn print_summary(&self) {
        println!("\n=== {} ===", self.name);
        println!("Total samples:     {}", self.total_samples);
        println!("Total duration:    {:?}", self.total_duration);
        println!("Average latency:   {:.2} ms", self.avg_latency_ms);
        println!("Min latency:       {:.2} ms", self.min_latency_ms);
        println!("Max latency:       {:.2} ms", self.max_latency_ms);
        println!("P95 latency:       {:.2} ms", self.p95_latency_ms);
        println!("P99 latency:       {:.2} ms", self.p99_latency_ms);
        println!("Throughput:        {:.1} samples/sec", self.throughput);
        println!("Real-time factor:  {:.2}x", self.calculate_rtf());
    }

    fn calculate_rtf(&self) -> f32 {
        // Assuming 1 second of audio per sample at 16kHz
        let audio_duration_secs = self.total_samples as f32;
        let processing_duration_secs = self.total_duration.as_secs_f32();
        audio_duration_secs / processing_duration_secs
    }
}

async fn benchmark_implementation<F, Fut>(
    name: &str,
    samples: Vec<Vec<f32>>,
    mut process_fn: F,
) -> BenchmarkResult
where
    F: FnMut(Vec<f32>) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<TranslationResult>>,
{
    let mut latencies = Vec::new();
    let total_start = Instant::now();

    for sample in samples.iter() {
        let start = Instant::now();
        let _ = process_fn(sample.clone()).await;
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_secs_f32() * 1000.0);
    }

    let total_duration = total_start.elapsed();

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;
    let min_latency = *latencies.first().unwrap_or(&0.0);
    let max_latency = *latencies.last().unwrap_or(&0.0);
    let p95_idx = (latencies.len() as f32 * 0.95) as usize;
    let p99_idx = (latencies.len() as f32 * 0.99) as usize;
    let p95_latency = latencies.get(p95_idx).copied().unwrap_or(0.0);
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(0.0);

    BenchmarkResult {
        name: name.to_string(),
        total_samples: samples.len(),
        total_duration,
        avg_latency_ms: avg_latency,
        min_latency_ms: min_latency,
        max_latency_ms: max_latency,
        p95_latency_ms: p95_latency,
        p99_latency_ms: p99_latency,
        throughput: samples.len() as f32 / total_duration.as_secs_f32(),
    }
}

async fn benchmark_batch_processing<F, Fut>(
    name: &str,
    samples: Vec<Vec<f32>>,
    batch_size: usize,
    mut process_fn: F,
) -> BenchmarkResult
where
    F: FnMut(Vec<Vec<f32>>) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<Vec<TranslationResult>>>,
{
    let mut latencies = Vec::new();
    let total_start = Instant::now();

    for batch in samples.chunks(batch_size) {
        let start = Instant::now();
        let _ = process_fn(batch.to_vec()).await;
        let elapsed = start.elapsed();
        let per_sample_latency = elapsed.as_secs_f32() * 1000.0 / batch.len() as f32;
        for _ in 0..batch.len() {
            latencies.push(per_sample_latency);
        }
    }

    let total_duration = total_start.elapsed();

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;
    let min_latency = *latencies.first().unwrap_or(&0.0);
    let max_latency = *latencies.last().unwrap_or(&0.0);
    let p95_idx = (latencies.len() as f32 * 0.95) as usize;
    let p99_idx = (latencies.len() as f32 * 0.99) as usize;
    let p95_latency = latencies.get(p95_idx).copied().unwrap_or(0.0);
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(0.0);

    BenchmarkResult {
        name: format!("{} (batch={})", name, batch_size),
        total_samples: samples.len(),
        total_duration,
        avg_latency_ms: avg_latency,
        min_latency_ms: min_latency,
        max_latency_ms: max_latency,
        p95_latency_ms: p95_latency,
        p99_latency_ms: p99_latency,
        throughput: samples.len() as f32 / total_duration.as_secs_f32(),
    }
}

fn generate_test_audio(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Generate synthetic speech-like audio (mix of frequencies)
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() // Fundamental
            + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()       // Harmonic
            + 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()      // Harmonic
            + 0.05 * (rand::random::<f32>() - 0.5);                       // Noise

        samples.push(sample);
    }

    samples
}

#[tokio::main]
async fn main() {
    println!("🚀 OBS Live Translator - Performance Benchmark Suite");
    println!("====================================================\n");

    // Configuration
    let sample_rate = 16000;
    let segment_duration = 1.0; // 1 second segments
    let num_samples = 100;
    let batch_sizes = vec![1, 4, 8, 16];

    println!("Configuration:");
    println!("  Sample rate:       {} Hz", sample_rate);
    println!("  Segment duration:  {} sec", segment_duration);
    println!("  Number of samples: {}", num_samples);
    println!("  Batch sizes:       {:?}", batch_sizes);

    // Generate test data
    println!("\n📊 Generating test audio data...");
    let mut samples = Vec::new();
    for _ in 0..num_samples {
        samples.push(generate_test_audio(segment_duration, sample_rate));
    }
    println!("  Generated {} audio segments", samples.len());

    let mut results = HashMap::new();

    // Benchmark standard Rust implementation
    println!("\n🔧 Benchmarking standard Rust implementation...");
    let mut standard_translator = Translator::new("models/whisper-base.onnx", false)
        .expect("Failed to create standard translator");

    let standard_result = benchmark_implementation(
        "Standard Rust",
        samples.clone(),
        |sample| async {
            standard_translator.process_audio(&sample).await
        },
    ).await;

    standard_result.print_summary();
    results.insert("Standard Rust", standard_result);

    // Benchmark standard batch processing
    for batch_size in &batch_sizes {
        if *batch_size > 1 {
            let batch_result = benchmark_batch_processing(
                "Standard Rust Batch",
                samples.clone(),
                *batch_size,
                |batch| async {
                    standard_translator.process_batch(batch).await
                },
            ).await;

            batch_result.print_summary();
            results.insert(format!("Standard Batch {}", batch_size), batch_result);
        }
    }

    // Benchmark optimized SIMD/ONNX implementation if available
    #[cfg(feature = "simd")]
    {
        println!("\n⚡ Benchmarking optimized SIMD/ONNX implementation...");
        let mut optimized_translator = OptimizedTranslator::new(
            "models/whisper-encoder.onnx",
            "models/whisper-decoder.onnx",
            false
        ).expect("Failed to create optimized translator");

        let optimized_result = benchmark_implementation(
            "Optimized SIMD/ONNX",
            samples.clone(),
            |sample| async {
                optimized_translator.process_audio(&sample).await
            },
        ).await;

        optimized_result.print_summary();
        results.insert("Optimized SIMD", optimized_result);

        // Benchmark optimized batch processing
        for batch_size in &batch_sizes {
            if *batch_size > 1 {
                let batch_result = benchmark_batch_processing(
                    "Optimized SIMD Batch",
                    samples.clone(),
                    *batch_size,
                    |batch| async {
                        optimized_translator.process_batch(batch).await
                    },
                ).await;

                batch_result.print_summary();
                results.insert(format!("Optimized Batch {}", batch_size), batch_result);
            }
        }

        // Test with GPU if CUDA is available
        #[cfg(feature = "cuda")]
        {
            println!("\n🎮 Benchmarking GPU-accelerated implementation...");
            let mut gpu_translator = OptimizedTranslator::new(
                "models/whisper-encoder.onnx",
                "models/whisper-decoder.onnx",
                true  // Enable GPU
            ).expect("Failed to create GPU translator");

            let gpu_result = benchmark_implementation(
                "GPU Accelerated",
                samples.clone(),
                |sample| async {
                    gpu_translator.process_audio(&sample).await
                },
            ).await;

            gpu_result.print_summary();
            results.insert("GPU Accelerated", gpu_result);
        }
    }

    // Summary comparison
    println!("\n📈 Performance Comparison Summary");
    println!("==================================");

    if let Some(standard) = results.get("Standard Rust") {
        println!("\nBaseline (Standard Rust):");
        println!("  Average latency: {:.2} ms", standard.avg_latency_ms);
        println!("  Throughput:      {:.1} samples/sec", standard.throughput);

        #[cfg(feature = "simd")]
        if let Some(optimized) = results.get("Optimized SIMD") {
            let speedup = standard.avg_latency_ms / optimized.avg_latency_ms;
            let throughput_gain = optimized.throughput / standard.throughput;

            println!("\nOptimized SIMD vs Standard:");
            println!("  Speedup:         {:.2}x faster", speedup);
            println!("  Throughput gain: {:.2}x", throughput_gain);
            println!("  Latency reduction: {:.1}%",
                    (1.0 - optimized.avg_latency_ms / standard.avg_latency_ms) * 100.0);
        }

        #[cfg(all(feature = "simd", feature = "cuda"))]
        if let Some(gpu) = results.get("GPU Accelerated") {
            let speedup = standard.avg_latency_ms / gpu.avg_latency_ms;
            let throughput_gain = gpu.throughput / standard.throughput;

            println!("\nGPU vs Standard:");
            println!("  Speedup:         {:.2}x faster", speedup);
            println!("  Throughput gain: {:.2}x", throughput_gain);
            println!("  Latency reduction: {:.1}%",
                    (1.0 - gpu.avg_latency_ms / standard.avg_latency_ms) * 100.0);
        }
    }

    // Find best batch size
    let mut best_batch_config = ("", 0.0f32);
    for (name, result) in &results {
        if name.contains("Batch") && result.throughput > best_batch_config.1 {
            best_batch_config = (name.as_str(), result.throughput);
        }
    }

    if !best_batch_config.0.is_empty() {
        println!("\n🏆 Best batch configuration: {}", best_batch_config.0);
        println!("   Throughput: {:.1} samples/sec", best_batch_config.1);
    }

    println!("\n✅ Benchmark complete!");
}

// Add rand for synthetic audio generation
#[allow(dead_code)]
mod rand {
    pub fn random<T>() -> T
    where
        T: Default,
    {
        T::default()
    }
}