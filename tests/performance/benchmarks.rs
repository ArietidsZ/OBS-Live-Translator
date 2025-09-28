//! Performance Benchmarks

use obs_live_translator::{
    Translator, TranslatorConfig,
    profile::Profile,
    audio::AudioProcessorConfig,
    asr::AsrConfig,
    translation::TranslationConfig,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

/// Benchmark audio processing pipeline
pub fn bench_audio_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_processing");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(2));

    let sample_rates = vec![16000, 44100, 48000];
    let frame_sizes = vec![10.0, 20.0, 30.0, 40.0]; // ms

    for sample_rate in sample_rates {
        for frame_size in &frame_sizes {
            let id = BenchmarkId::new(
                format!("{}Hz", sample_rate),
                format!("{}ms", frame_size)
            );

            group.bench_with_input(id, frame_size, |b, &frame_size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = AudioProcessorConfig {
                    sample_rate,
                    frame_size_ms: frame_size,
                    enable_vad: true,
                    enable_noise_reduction: true,
                    enable_agc: true,
                };

                b.to_async(&rt).iter(|| async {
                    let processor = obs_live_translator::audio::AudioProcessor::new(config.clone())
                        .await
                        .unwrap();
                    
                    let audio_data = generate_audio_data(1000, sample_rate);
                    processor.process(&audio_data).await.unwrap()
                });
            });
        }
    }

    group.finish();
}

/// Benchmark ASR inference
pub fn bench_asr_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("asr_inference");
    group.measurement_time(Duration::from_secs(15));

    let models = vec![
        ("whisper_tiny", "Int8"),
        ("whisper_small", "Fp16"),
        ("parakeet", "Fp32"),
    ];

    for (model, precision) in models {
        group.bench_function(BenchmarkId::new(model, precision), |b| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let config = AsrConfig {
                model_type: model.to_string(),
                model_path: format!("models/{}.onnx", model),
                device: "cpu".to_string(),
                batch_size: 1,
                precision: match precision {
                    "Int8" => obs_live_translator::asr::ModelPrecision::Int8,
                    "Fp16" => obs_live_translator::asr::ModelPrecision::Fp16,
                    _ => obs_live_translator::asr::ModelPrecision::Fp32,
                },
                enable_timestamps: false,
            };

            b.to_async(&rt).iter(|| async {
                let mut engine = obs_live_translator::asr::AsrEngine::new(config.clone())
                    .await
                    .unwrap();
                
                let audio = generate_audio_data(1000, 16000);
                engine.transcribe(&audio).await.unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark translation performance
pub fn bench_translation(c: &mut Criterion) {
    let mut group = c.benchmark_group("translation");
    
    let test_texts = vec![
        ("short", "Hello world"),
        ("medium", "The quick brown fox jumps over the lazy dog. This is a test sentence."),
        ("long", "This is a longer piece of text that simulates a real translation scenario with multiple sentences. It contains various grammatical structures and vocabulary that would be typical in a live translation setting. The goal is to measure the performance of the translation engine under realistic conditions."),
    ];

    let language_pairs = vec![
        ("en", "es"),
        ("en", "fr"),
        ("en", "de"),
        ("en", "ja"),
        ("en", "zh"),
    ];

    for (size, text) in test_texts {
        for (source, target) in &language_pairs {
            let id = BenchmarkId::new(
                format!("{}_{}", source, target),
                size
            );

            group.bench_with_input(id, &text, |b, text| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = TranslationConfig {
                    model_type: "nllb".to_string(),
                    model_path: "models/nllb.onnx".to_string(),
                    source_language: source.to_string(),
                    target_language: target.to_string(),
                    device: "cpu".to_string(),
                    max_length: 512,
                };

                b.to_async(&rt).iter(|| async {
                    let mut engine = obs_live_translator::translation::TranslationEngine::new(config.clone())
                        .await
                        .unwrap();
                    
                    engine.translate(black_box(text)).await.unwrap()
                });
            });
        }
    }

    group.finish();
}

/// Benchmark end-to-end pipeline
pub fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    group.measurement_time(Duration::from_secs(20));

    let profiles = vec![Profile::Low, Profile::Medium, Profile::High];

    for profile in profiles {
        group.bench_function(BenchmarkId::new("pipeline", format!("{:?}", profile)), |b| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let config = TranslatorConfig::from_profile(profile);

            b.to_async(&rt).iter(|| async {
                let translator = Translator::new(config.clone()).await.unwrap();
                let audio = generate_audio_data(2000, 16000);
                translator.process_audio(&audio).await.unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark streaming performance
pub fn bench_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming");
    
    let chunk_sizes = vec![100, 200, 500, 1000]; // ms
    let protocols = vec!["websocket", "webrtc"];

    for protocol in protocols {
        for chunk_size in &chunk_sizes {
            let id = BenchmarkId::new(protocol, format!("{}ms", chunk_size));

            group.bench_with_input(id, chunk_size, |b, &chunk_size| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                
                b.to_async(&rt).iter(|| async {
                    let samples = (16000 * chunk_size / 1000) as usize;
                    let audio = generate_audio_data(chunk_size, 16000);
                    
                    // Simulate streaming processing
                    process_streaming_chunk(&audio).await
                });
            });
        }
    }

    group.finish();
}

/// Helper function to generate audio data
fn generate_audio_data(duration_ms: u32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (sample_rate * duration_ms / 1000) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
        samples.push(sample);
    }
    
    samples
}

async fn process_streaming_chunk(audio: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    // Simulate streaming processing
    tokio::time::sleep(Duration::from_micros(100)).await;
    Ok(())
}

criterion_group!(
    benches,
    bench_audio_processing,
    bench_asr_inference,
    bench_translation,
    bench_end_to_end,
    bench_streaming
);
criterion_main!(benches);