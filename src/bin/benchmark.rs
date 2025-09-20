use obs_live_translator::Translator;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("Running performance benchmarks...\n");

    // Generate test audio (1 second of silence for now)
    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let samples = (sample_rate as f32 * duration) as usize;
    let audio: Vec<f32> = vec![0.0; samples];

    // Test single inference
    println!("=== Single Inference Test ===");
    let mut translator = Translator::new("models/whisper-tiny", false)
        .expect("Failed to create translator");

    let start = Instant::now();
    let result = translator.process_audio(&audio).await;
    let elapsed = start.elapsed();

    println!("Audio duration: {:.2}s", duration);
    println!("Processing time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
    println!("Real-time factor: {:.2}x", duration / elapsed.as_secs_f32());

    if let Ok(result) = result {
        println!("Original text: {:?}", result.original_text);
        println!("Translated text: {:?}", result.translated_text);
        println!("Processing time: {:.2}ms", result.processing_time_ms);
    }

    // Test batch processing
    println!("\n=== Batch Processing Test ===");
    let batch_size = 4;
    let batch: Vec<Vec<f32>> = (0..batch_size).map(|_| audio.clone()).collect();

    let start = Instant::now();
    let results = translator.process_batch(batch).await;
    let elapsed = start.elapsed();

    let total_audio_duration = duration * batch_size as f32;
    println!("Batch size: {}", batch_size);
    println!("Total audio: {:.2}s", total_audio_duration);
    println!("Processing time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
    println!("Real-time factor: {:.2}x", total_audio_duration / elapsed.as_secs_f32());
    println!("Throughput: {:.1} samples/sec", batch_size as f32 / elapsed.as_secs_f32());

    // Memory test
    println!("\n=== Memory Usage ===");
    let stats = translator.get_stats();
    println!("Inference count: {}", stats.0);
    println!("Average latency: {:.2}ms", stats.1);
}