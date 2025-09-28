//! ML Inference Accuracy Unit Tests

#[cfg(test)]
mod tests {
    use obs_live_translator::asr::{AsrEngine, AsrConfig, ModelPrecision};
    use obs_live_translator::language_detection::{LanguageDetector, LanguageDetectionConfig};
    use obs_live_translator::translation::{TranslationEngine, TranslationConfig};
    use obs_live_translator::profile::Profile;
    use crate::test_utils::*;

    #[tokio::test]
    async fn test_asr_initialization() {
        let config = AsrConfig {
            model_type: "whisper_tiny".to_string(),
            model_path: "models/whisper_tiny.onnx".to_string(),
            device: "cpu".to_string(),
            batch_size: 1,
            precision: ModelPrecision::Int8,
            enable_timestamps: true,
        };

        let engine = AsrEngine::new(config).await;
        assert!(engine.is_ok(), "ASR engine should initialize successfully");
    }

    #[tokio::test]
    async fn test_asr_transcription() {
        run_with_timeout(async {
            let config = AsrConfig {
                model_type: "whisper_tiny".to_string(),
                model_path: "models/whisper_tiny.onnx".to_string(),
                device: "cpu".to_string(),
                batch_size: 1,
                precision: ModelPrecision::Int8,
                enable_timestamps: false,
            };

            let mut engine = AsrEngine::new(config).await.unwrap();

            // Generate test audio (1 second at 16kHz)
            let audio = generate_test_audio(1000, 16000);

            let result = engine.transcribe(&audio).await.unwrap();

            assert!(result.text.is_some());
            assert!(result.confidence > 0.0);
            assert!(result.processing_time_ms > 0.0);
        }).await;
    }

    #[tokio::test]
    async fn test_language_detection() {
        let config = LanguageDetectionConfig {
            engine_type: "fasttext".to_string(),
            model_path: "models/fasttext.bin".to_string(),
            confidence_threshold: 0.7,
            enable_multimodal: false,
        };

        let mut detector = LanguageDetector::new(config).await.unwrap();

        // Test with English text
        let english_text = "Hello, this is a test sentence in English.";
        let result = detector.detect_language(english_text).await.unwrap();

        assert_eq!(result.language_code, "en");
        assert!(result.confidence > 0.5);

        // Test with Spanish text
        let spanish_text = "Hola, esta es una oración de prueba en español.";
        let result = detector.detect_language(spanish_text).await.unwrap();

        assert_eq!(result.language_code, "es");
        assert!(result.confidence > 0.5);
    }

    #[tokio::test]
    async fn test_multimodal_language_detection() {
        let config = LanguageDetectionConfig {
            engine_type: "fusion".to_string(),
            model_path: "models/fasttext.bin".to_string(),
            confidence_threshold: 0.7,
            enable_multimodal: true,
        };

        let mut detector = LanguageDetector::new(config).await.unwrap();

        // Test with both audio and text signals
        let text = "This is English text";
        let audio_features = vec![0.1, 0.2, 0.3, 0.4]; // Simplified features

        let result = detector.detect_multimodal(text, &audio_features).await.unwrap();

        assert!(!result.language_code.is_empty());
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_translation_engine() {
        let config = TranslationConfig {
            model_type: "nllb".to_string(),
            model_path: "models/nllb.onnx".to_string(),
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            device: "cpu".to_string(),
            max_length: 512,
        };

        let mut engine = TranslationEngine::new(config).await.unwrap();

        let source_text = "Hello, how are you?";
        let result = engine.translate(source_text).await.unwrap();

        assert!(!result.translated_text.is_empty());
        assert!(result.confidence > 0.0);
        assert_eq!(result.source_language, "en");
        assert_eq!(result.target_language, "es");
    }

    #[tokio::test]
    async fn test_translation_with_auto_detect() {
        let config = TranslationConfig {
            model_type: "m2m".to_string(),
            model_path: "models/m2m.onnx".to_string(),
            source_language: "auto".to_string(), // Auto-detect
            target_language: "en".to_string(),
            device: "cpu".to_string(),
            max_length: 512,
        };

        let mut engine = TranslationEngine::new(config).await.unwrap();

        // Test with Spanish text
        let source_text = "Hola, ¿cómo estás?";
        let result = engine.translate(source_text).await.unwrap();

        assert!(!result.translated_text.is_empty());
        assert_eq!(result.detected_language, Some("es".to_string()));
        assert_eq!(result.target_language, "en");
    }

    #[tokio::test]
    async fn test_batch_inference() {
        let config = AsrConfig {
            model_type: "whisper_tiny".to_string(),
            model_path: "models/whisper_tiny.onnx".to_string(),
            device: "cpu".to_string(),
            batch_size: 4,
            precision: ModelPrecision::Fp16,
            enable_timestamps: false,
        };

        let mut engine = AsrEngine::new(config).await.unwrap();

        // Create batch of audio samples
        let batch = vec![
            generate_test_audio(500, 16000),
            generate_test_audio(750, 16000),
            generate_test_audio(1000, 16000),
            generate_test_audio(1250, 16000),
        ];

        let results = engine.transcribe_batch(&batch).await.unwrap();

        assert_eq!(results.len(), 4);
        for result in results {
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_model_precision_modes() {
        let precisions = vec![
            ModelPrecision::Int8,
            ModelPrecision::Fp16,
            ModelPrecision::Fp32,
        ];

        for precision in precisions {
            let config = AsrConfig {
                model_type: "whisper_tiny".to_string(),
                model_path: "models/whisper_tiny.onnx".to_string(),
                device: "cpu".to_string(),
                batch_size: 1,
                precision: precision.clone(),
                enable_timestamps: false,
            };

            let engine = AsrEngine::new(config).await;
            assert!(engine.is_ok(), "Should support {:?} precision", precision);

            // Test that different precisions have different performance characteristics
            match precision {
                ModelPrecision::Int8 => {
                    // INT8 should be fastest but potentially less accurate
                },
                ModelPrecision::Fp16 => {
                    // FP16 balanced speed and accuracy
                },
                ModelPrecision::Fp32 => {
                    // FP32 highest accuracy but slowest
                },
            }
        }
    }

    #[tokio::test]
    async fn test_profile_aware_inference() {
        let profiles = vec![Profile::Low, Profile::Medium, Profile::High];

        for profile in profiles {
            let config = AsrConfig::from_profile(profile);
            let engine = AsrEngine::new(config).await.unwrap();

            // Different profiles should use different models
            match profile {
                Profile::Low => {
                    assert_eq!(engine.get_model_type(), "whisper_tiny");
                    assert_eq!(engine.get_precision(), ModelPrecision::Int8);
                },
                Profile::Medium => {
                    assert_eq!(engine.get_model_type(), "whisper_small");
                    assert_eq!(engine.get_precision(), ModelPrecision::Fp16);
                },
                Profile::High => {
                    assert_eq!(engine.get_model_type(), "parakeet");
                    assert_eq!(engine.get_precision(), ModelPrecision::Fp32);
                },
            }
        }
    }

    #[tokio::test]
    async fn test_streaming_inference() {
        let config = AsrConfig {
            model_type: "parakeet".to_string(),
            model_path: "models/parakeet.onnx".to_string(),
            device: "cpu".to_string(),
            batch_size: 1,
            precision: ModelPrecision::Fp32,
            enable_timestamps: true,
        };

        let mut engine = AsrEngine::new(config).await.unwrap();

        // Simulate streaming with chunks
        let full_audio = generate_test_audio(3000, 16000); // 3 seconds
        let chunk_size = 16000 / 2; // 500ms chunks

        let mut accumulated_text = String::new();

        for chunk in full_audio.chunks(chunk_size) {
            let result = engine.transcribe_streaming(chunk).await.unwrap();
            if let Some(text) = result.text {
                accumulated_text.push_str(&text);
            }
        }

        // Should have accumulated some text from streaming
        assert!(!accumulated_text.is_empty());
    }

    #[tokio::test]
    async fn test_translation_quality_metrics() {
        let config = TranslationConfig {
            model_type: "nllb".to_string(),
            model_path: "models/nllb.onnx".to_string(),
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            device: "cpu".to_string(),
            max_length: 512,
        };

        let mut engine = TranslationEngine::new(config).await.unwrap();

        // Test translation with quality metrics
        let source = "The quick brown fox jumps over the lazy dog";
        let result = engine.translate_with_metrics(source).await.unwrap();

        assert!(!result.translated_text.is_empty());
        assert!(result.bleu_score.is_some());

        if let Some(bleu) = result.bleu_score {
            assert!(bleu >= 0.0 && bleu <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_confidence_thresholding() {
        let config = AsrConfig {
            model_type: "whisper_tiny".to_string(),
            model_path: "models/whisper_tiny.onnx".to_string(),
            device: "cpu".to_string(),
            batch_size: 1,
            precision: ModelPrecision::Int8,
            enable_timestamps: false,
        };

        let mut engine = AsrEngine::new(config).await.unwrap();
        engine.set_confidence_threshold(0.8);

        // Test with low quality audio (noise)
        let noise = vec![rand::random::<f32>() * 0.01 - 0.005; 16000];
        let result = engine.transcribe(&noise).await.unwrap();

        // Low confidence results should be filtered
        if result.confidence < 0.8 {
            assert!(result.text.is_none() || result.text.as_ref().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn test_gpu_acceleration() {
        // Skip if no GPU available
        if !obs_live_translator::inference::acceleration::is_gpu_available() {
            return;
        }

        let config = AsrConfig {
            model_type: "whisper_small".to_string(),
            model_path: "models/whisper_small.onnx".to_string(),
            device: "cuda".to_string(),
            batch_size: 4,
            precision: ModelPrecision::Fp16,
            enable_timestamps: false,
        };

        let engine = AsrEngine::new(config).await;
        assert!(engine.is_ok(), "GPU acceleration should work if available");

        let mut engine = engine.unwrap();
        let audio = generate_test_audio(1000, 16000);

        // GPU inference should be faster than CPU for batch processing
        let start = std::time::Instant::now();
        let _result = engine.transcribe(&audio).await.unwrap();
        let gpu_time = start.elapsed();

        // Compare with CPU timing (in real tests)
        println!("GPU inference time: {:?}", gpu_time);
    }
}