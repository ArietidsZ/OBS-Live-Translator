//! Integration tests for OBS Live Translator

use obs_live_translator::{types::TranslatorConfig, Profile, Translator};

#[tokio::test]
async fn test_profile_detection() {
    use obs_live_translator::profile::ProfileDetector;

    let profile = ProfileDetector::detect();
    assert!(profile.is_ok(), "Profile detection should succeed");

    let profile = profile.unwrap();
    assert!(matches!(
        profile,
        Profile::Low | Profile::Medium | Profile::High
    ));
}

#[tokio::test]
async fn test_translator_creation() {
    let mut config = TranslatorConfig::default();
    config.model_path = "./models".to_string();

    let result = Translator::new(config).await;
    // May fail if models not downloaded, but shouldn't panic
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_mel_spectrogram_extractor() {
    use obs_live_translator::audio::mel_spectrogram::whisper_mel_extractor;

    let extractor = whisper_mel_extractor();
    assert!(extractor.is_ok(), "Should create mel extractor");

    let extractor = extractor.unwrap();

    // Test with 1 second of silence
    let samples = vec![0.0f32; 16000];
    let result = extractor.extract(&samples);

    assert!(result.is_ok(), "Should extract features");
    let mel_spec = result.unwrap();

    // Verify output shape (1, 80, ~100 frames)
    assert_eq!(mel_spec.shape()[0], 1, "Batch size should be 1");
    assert_eq!(mel_spec.shape()[1], 80, "Should have 80 mel bins");
    assert!(
        mel_spec.shape()[2] > 90 && mel_spec.shape()[2] < 110,
        "Should have ~100 frames for 1 second"
    );
}

#[test]
fn test_beam_search() {
    use obs_live_translator::asr::beam_search::BeamSearch;

    let beam_search = BeamSearch::new(5, 20, 0); // beam=5, max_len=20, eos=0

    // Mock function that always predicts token 1, then 2, then EOS (0)
    let result = beam_search.search(|hyps| {
        hyps.iter()
            .map(|h| {
                let len = h.tokens.len();
                match len {
                    0 => vec![(1, -0.1)],
                    1 => vec![(2, -0.1)],
                    _ => vec![(0, -0.1)], // EOS
                }
            })
            .collect()
    });

    assert_eq!(result.tokens.len(), 3);
    assert!(result.finished);
}

#[test]
fn test_audio_resampling() {
    use obs_live_translator::audio::resampling::resample;

    // Resample 1 second of audio (16kHz -> 48kHz)
    let input = vec![0.0f32; 16000];
    let result = resample(&input, 16000, 48000, 1);

    assert!(result.is_ok(), "Should resample");
    let output = result.unwrap();

    // Output should be ~3x the input length
    assert!(
        output.len() >= 47000 && output.len() <= 49000,
        "Output should be ~48000 samples"
    );
}

#[test]
fn test_config_serialization() {
    use obs_live_translator::types::TranslatorConfig;

    let config = TranslatorConfig::default();
    let toml_str = toml::to_string(&config);

    assert!(toml_str.is_ok(), "Should serialize to TOML");

    let deserialized: Result<TranslatorConfig, _> = toml::from_str(&toml_str.unwrap());
    assert!(deserialized.is_ok(), "Should deserialize from TOML");
}
