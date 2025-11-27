//! Real audio sample testing
//!
//! This program generates synthetic audio samples and tests the audio processing
//! pipeline with realistic data.

use anyhow::Result;
use hound::{WavSpec, WavWriter};
use std::f32::consts::PI;
// use std::path::Path;

/// Generate a sine wave at the specified frequency
fn generate_tone(freq: f32, duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * PI * freq * t).sin()
        })
        .collect()
}

/// Generate pure silence
fn generate_silence(duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    vec![0.0; num_samples]
}

/// Generate speech-like signal
fn generate_speech_sim(duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f0 = 150.0 + 50.0 * (2.0 * PI * 3.0 * t).sin();
            let signal = 0.5 * (2.0 * PI * f0 * t).sin()
                + 0.3 * (2.0 * PI * 2.0 * f0 * t).sin()
                + 0.2 * (2.0 * PI * 3.0 * f0 * t).sin();
            let envelope = 0.5 + 0.5 * (2.0 * PI * 4.0 * t).sin();
            signal * envelope
        })
        .collect()
}

/// Write samples to a WAV file
fn write_wav(filename: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in samples {
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample * amplitude) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎵 Generating test audio samples...");

    // Create directory
    std::fs::create_dir_all("test_audio")?;

    let sample_rate = 16000;

    // 1. Silence
    println!("  ✓ Generating silence.wav");
    let silence = generate_silence(2.0, sample_rate);
    write_wav("test_audio/silence.wav", &silence, sample_rate)?;

    // 2. Pure tone
    println!("  ✓ Generating tone_440hz.wav");
    let tone = generate_tone(440.0, 2.0, sample_rate);
    write_wav("test_audio/tone_440hz.wav", &tone, sample_rate)?;

    // 3. Speech simulation
    println!("  ✓ Generating speech_sim.wav");
    let speech = generate_speech_sim(3.0, sample_rate);
    write_wav("test_audio/speech_sim.wav", &speech, sample_rate)?;

    println!("✅ Test audio samples generated in test_audio/");
    println!("   - silence.wav (2s)");
    println!("   - tone_440hz.wav (2s)");
    println!("   - speech_sim.wav (3s)");

    // Now test audio processing modules
    println!("\n🔬 Testing audio processing modules...");
    test_audio_processing(&speech).await?;

    Ok(())
}

async fn test_audio_processing(samples: &[f32]) -> Result<()> {
    use obs_live_translator::types::TranslatorConfig;
    use obs_live_translator::vad::create_vad_engine;
    use std::sync::Arc;

    println!("\n📊 Testing VAD (Voice Activity Detection)...");
    println!("   Note: VAD expects 512+ samples per frame");

    let config = TranslatorConfig::default();
    let cache = Arc::new(obs_live_translator::models::session_cache::SessionCache::new());
    let vad = create_vad_engine(&config, cache).await?;

    // Test 1: Silence detection
    println!("\n  Test 1: Silence Detection");
    let silence_frame = &generate_silence(1.0, 16000)[0..512]; // Take first 512 samples
    let is_speech = vad.detect(silence_frame).await?;
    println!("    - Is Speech: {is_speech} (expected: false)");

    // 2. Test with tone (should be speech)
    println!("\n  Test 2: Pure Tone Detection");
    let tone = generate_tone(440.0, 1.0, 16000);
    let is_speech = vad.detect(&tone).await?;
    println!("    - Is Speech: {is_speech} (expected: true)");

    // 3. Test with mixed
    println!("\n  Test 3: Mixed Silence and Tone Detection");
    let silence_full = generate_silence(1.0, 16000);
    let mut mixed = silence_full.clone();
    mixed.extend_from_slice(&tone);
    let is_speech = vad.detect(&mixed).await?;
    println!("    - Is Speech: {is_speech} (expected: true)");

    // Test 4: Speech simulation detection
    println!("\n  Test 4: Speech Simulation Detection");
    let speech_frame = &samples[0..512.min(samples.len())]; // Take first 512 samples
    let is_speech = vad.detect(speech_frame).await?;
    println!("    - Is Speech: {is_speech} (expected: true)");

    println!("\n✅ VAD testing completed successfully!");
    println!("\nℹ️  Generated audio files available in test_audio/:");
    println!("   - silence.wav (2s,16kHz)");
    println!("   - tone_440hz.wav (2s, 16kHz)");
    println!("   - speech_sim.wav (3s, 16kHz)");

    Ok(())
}
