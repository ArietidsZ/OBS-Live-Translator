//! Audio Processing Pipeline Unit Tests

#[cfg(test)]
mod tests {
    use obs_live_translator::audio::{
        processor_trait::{AudioProcessor, AudioProcessorConfig},
        vad::{VadEngine, VadConfig},
        resampling::{ResamplerEngine, ResamplerConfig, ResamplingQuality},
        features::{MelSpectrogramExtractor, MelSpectrogramConfig},
    };
    use crate::test_utils::*;

    #[tokio::test]
    async fn test_vad_detection() {
        // Test WebRTC VAD
        let config = VadConfig {
            engine_type: "webrtc".to_string(),
            threshold: 0.5,
            min_voice_duration_ms: 100,
            min_silence_duration_ms: 200,
        };

        let mut vad = VadEngine::new(config).await.unwrap();

        // Test with silence
        let silence = vec![0.0f32; 16000]; // 1 second of silence at 16kHz
        let result = vad.process_frame(&silence).await.unwrap();
        assert!(!result.is_voice, "VAD should not detect voice in silence");

        // Test with generated audio
        let audio = generate_test_audio(1000, 16000);
        let result = vad.process_frame(&audio).await.unwrap();
        // Note: Actual voice detection depends on VAD implementation
    }

    #[tokio::test]
    async fn test_resampling() {
        let config = ResamplerConfig {
            input_sample_rate: 48000,
            output_sample_rate: 16000,
            quality: ResamplingQuality::Medium,
            channels: 1,
        };

        let mut resampler = ResamplerEngine::new(config).await.unwrap();

        // Generate test audio at 48kHz
        let input_audio = generate_test_audio(100, 48000);
        let expected_output_len = (input_audio.len() as f32 * 16000.0 / 48000.0) as usize;

        let output = resampler.resample(&input_audio).await.unwrap();

        // Check output length is approximately correct
        let length_ratio = output.len() as f32 / expected_output_len as f32;
        assert_approx_eq(length_ratio, 1.0, 0.1);
    }

    #[tokio::test]
    async fn test_mel_spectrogram_extraction() {
        let config = MelSpectrogramConfig {
            sample_rate: 16000,
            n_fft: 1024,
            hop_length: 256,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8000.0,
        };

        let mut extractor = MelSpectrogramExtractor::new(config).await.unwrap();

        // Generate test audio
        let audio = generate_test_audio(1000, 16000);

        let mel_spectrogram = extractor.extract(&audio).await.unwrap();

        // Check dimensions
        assert_eq!(mel_spectrogram.n_mels, 80);
        assert!(mel_spectrogram.n_frames > 0);
        assert_eq!(mel_spectrogram.data.len(), 80 * mel_spectrogram.n_frames);
    }

    #[tokio::test]
    async fn test_audio_processor_pipeline() {
        let config = AudioProcessorConfig {
            sample_rate: 16000,
            frame_size_ms: 30.0,
            enable_vad: true,
            enable_noise_reduction: false,
            enable_agc: false,
        };

        let processor = AudioProcessor::new(config).await.unwrap();

        // Generate test audio
        let audio = generate_test_audio(1000, 16000);

        let processed = processor.process(&audio).await.unwrap();

        assert!(!processed.frames.is_empty());
        assert!(processed.sample_rate == 16000);
    }

    #[tokio::test]
    async fn test_frame_size_calculations() {
        let sample_rates = vec![16000, 22050, 44100, 48000];
        let frame_size_ms = 30.0;

        for sample_rate in sample_rates {
            let expected_frame_size = (sample_rate as f32 * frame_size_ms / 1000.0) as usize;
            let config = AudioProcessorConfig {
                sample_rate,
                frame_size_ms,
                enable_vad: false,
                enable_noise_reduction: false,
                enable_agc: false,
            };

            let processor = AudioProcessor::new(config).await.unwrap();
            assert_eq!(processor.get_frame_size(), expected_frame_size);
        }
    }

    #[tokio::test]
    async fn test_multi_channel_audio() {
        // Test stereo to mono conversion
        let stereo_audio = vec![
            0.5, -0.5,  // L, R sample 1
            0.3, -0.3,  // L, R sample 2
            0.7, -0.7,  // L, R sample 3
        ];

        let config = AudioProcessorConfig {
            sample_rate: 16000,
            frame_size_ms: 30.0,
            enable_vad: false,
            enable_noise_reduction: false,
            enable_agc: false,
        };

        let processor = AudioProcessor::new(config).await.unwrap();
        let mono = processor.stereo_to_mono(&stereo_audio).await.unwrap();

        assert_eq!(mono.len(), 3);
        assert_approx_eq(mono[0], 0.0, 0.01); // (0.5 + (-0.5)) / 2 = 0
        assert_approx_eq(mono[1], 0.0, 0.01); // (0.3 + (-0.3)) / 2 = 0
        assert_approx_eq(mono[2], 0.0, 0.01); // (0.7 + (-0.7)) / 2 = 0
    }

    #[tokio::test]
    async fn test_audio_normalization() {
        let audio = vec![0.1, 0.5, -0.3, 0.9, -0.8];

        let config = AudioProcessorConfig {
            sample_rate: 16000,
            frame_size_ms: 30.0,
            enable_vad: false,
            enable_noise_reduction: false,
            enable_agc: true,
        };

        let processor = AudioProcessor::new(config).await.unwrap();
        let normalized = processor.normalize_audio(&audio).await.unwrap();

        // Check that max absolute value is close to 1.0
        let max_val = normalized.iter()
            .map(|x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert_approx_eq(max_val, 1.0, 0.1);
    }

    #[tokio::test]
    async fn test_vad_energy_threshold() {
        let config = VadConfig {
            engine_type: "energy".to_string(),
            threshold: 0.01, // Energy threshold
            min_voice_duration_ms: 100,
            min_silence_duration_ms: 200,
        };

        let mut vad = VadEngine::new(config).await.unwrap();

        // Test with low energy (silence)
        let silence = vec![0.001f32; 1600]; // 100ms at 16kHz
        let result = vad.process_frame(&silence).await.unwrap();
        assert!(!result.is_voice);

        // Test with high energy
        let loud = vec![0.5f32; 1600]; // 100ms at 16kHz
        let result = vad.process_frame(&loud).await.unwrap();
        assert!(result.is_voice);
    }

    #[tokio::test]
    async fn test_adaptive_vad() {
        let config = VadConfig {
            engine_type: "adaptive".to_string(),
            threshold: 0.5,
            min_voice_duration_ms: 100,
            min_silence_duration_ms: 200,
        };

        let mut vad = VadEngine::new(config).await.unwrap();

        // Process multiple frames to test adaptation
        for i in 0..10 {
            let amplitude = 0.1 * i as f32;
            let audio = vec![amplitude; 1600];
            let result = vad.process_frame(&audio).await.unwrap();

            // VAD should adapt to changing energy levels
            if i > 5 {
                assert!(result.confidence > 0.3);
            }
        }
    }

    #[tokio::test]
    async fn test_profile_aware_audio_processing() {
        use obs_live_translator::profile::Profile;

        let profiles = vec![Profile::Low, Profile::Medium, Profile::High];

        for profile in profiles {
            let config = AudioProcessorConfig::from_profile(profile);
            let processor = AudioProcessor::new(config).await.unwrap();

            // Different profiles should have different configurations
            match profile {
                Profile::Low => {
                    assert_eq!(processor.get_sample_rate(), 16000);
                    assert!(!processor.is_gpu_enabled());
                },
                Profile::Medium => {
                    assert!(processor.get_sample_rate() >= 16000);
                },
                Profile::High => {
                    assert!(processor.get_sample_rate() >= 44100);
                    // High profile may enable GPU if available
                },
            }
        }
    }
}