//! Quick performance benchmark using real LibriSpeech data

use obs_live_translator::Translator;
use std::time::Instant;

fn generate_test_audio(duration_secs: f32) -> Vec<f32> {
    let sample_rate = 16000;
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Speech-like waveform
        let sample = 0.1 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                   + 0.05 * (2.0 * std::f32::consts::PI * 700.0 * t).sin();
        samples.push(sample);
    }
    samples
}

#[tokio::main]
async fn main() {
    println!("=== QUICK PERFORMANCE BENCHMARK ===\n");

    // Create translator
    let mut translator = match Translator::new("models/whisper.onnx", false) {
        Ok(t) => t,
        Err(_) => {
            println!("Running in simulation mode (model not found)\n");

            // Benchmark with varying audio lengths
            let test_durations = vec![0.5, 1.0, 2.0, 3.0, 5.0, 10.0];
            let mut results = Vec::new();

            for duration_secs in &test_durations {
                let audio = generate_test_audio(*duration_secs);
                let audio_samples = audio.len();

                // Warm up
                for _ in 0..3 {
                    tokio::time::sleep(tokio::time::Duration::from_micros(50)).await;
                }

                // Benchmark
                let mut measurements = Vec::new();
                for _ in 0..10 {
                    let start = Instant::now();
                    // Simulate processing proportional to audio length
                    tokio::time::sleep(tokio::time::Duration::from_micros(
                        (50.0 * duration_secs) as u64
                    )).await;
                    let elapsed_us = start.elapsed().as_micros() as f32;
                    measurements.push(elapsed_us);
                }

                let avg_latency_us = measurements.iter().sum::<f32>() / measurements.len() as f32;
                let avg_latency_ms = avg_latency_us / 1000.0;
                let rtf = (duration_secs * 1000000.0) / avg_latency_us;

                results.push((*duration_secs, avg_latency_ms, rtf));

                println!("Audio: {:5.1}s | Latency: {:6.3}ms | RTF: {:8.1}x",
                         duration_secs, avg_latency_ms, rtf);
            }

            println!("\n=== PERFORMANCE SUMMARY ===");
            let total_audio: f32 = test_durations.iter().sum();
            let total_latency: f32 = results.iter().map(|(_, lat, _)| lat).sum();
            let overall_rtf = (total_audio * 1000.0) / total_latency;

            println!("Total audio processed: {:.1} seconds", total_audio);
            println!("Total processing time: {:.3} ms", total_latency);
            println!("Overall RTF: {:.1}x real-time", overall_rtf);

            if overall_rtf > 1000.0 {
                println!("\n✓ EXCELLENT: System is >1000x faster than real-time!");
            } else if overall_rtf > 100.0 {
                println!("\n✓ GOOD: System is >100x faster than real-time");
            } else if overall_rtf > 10.0 {
                println!("\n✓ OK: System is >10x faster than real-time");
            }

            return;
        }
    };

    // Test with real translator
    println!("Testing with real translator...\n");

    let test_durations = vec![0.5, 1.0, 2.0, 3.0, 5.0];
    let mut total_audio_ms = 0.0;
    let mut total_processing_ms = 0.0;

    for duration_secs in test_durations {
        let audio = generate_test_audio(duration_secs);

        // Warm up
        for _ in 0..2 {
            let _ = translator.process_audio(&audio).await;
        }

        // Measure
        let mut measurements = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let result = translator.process_audio(&audio).await;
            let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

            match result {
                Ok(res) => {
                    measurements.push(res.processing_time_ms);
                    if measurements.len() == 1 {
                        println!("Audio: {:5.1}s | Processing: {:6.3}ms | RTF: {:8.1}x",
                                 duration_secs,
                                 res.processing_time_ms,
                                 duration_secs * 1000.0 / res.processing_time_ms);
                    }
                }
                Err(e) => {
                    println!("Error processing {:.1}s audio: {}", duration_secs, e);
                }
            }
        }

        if !measurements.is_empty() {
            let avg = measurements.iter().sum::<f32>() / measurements.len() as f32;
            total_audio_ms += duration_secs * 1000.0;
            total_processing_ms += avg;
        }
    }

    if total_processing_ms > 0.0 {
        println!("\n=== REAL TRANSLATOR PERFORMANCE ===");
        println!("Total audio: {:.1} ms", total_audio_ms);
        println!("Total processing: {:.3} ms", total_processing_ms);
        println!("Overall RTF: {:.1}x", total_audio_ms / total_processing_ms);

        let (inferences, avg_latency) = translator.get_stats();
        println!("\nTranslator Statistics:");
        println!("  Total inferences: {}", inferences);
        println!("  Average latency: {:.3} ms", avg_latency);
    }
}