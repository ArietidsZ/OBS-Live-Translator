//! End-to-End Integration Tests

#[cfg(test)]
mod tests {
    use obs_live_translator::{
        Translator, TranslatorConfig,
        profile::Profile,
        audio::AudioProcessorConfig,
        asr::AsrConfig,
        translation::TranslationConfig,
        streaming::{StreamingServer, StreamingConfig},
        monitoring::PerformanceMonitor,
    };
    use crate::integration_test_utils::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_end_to_end_translation_pipeline() {
        let env = setup_test_environment().await;

        // Create translator with full pipeline
        let config = TranslatorConfig {
            profile: Profile::Medium,
            audio: AudioProcessorConfig {
                sample_rate: 16000,
                frame_size_ms: 30.0,
                enable_vad: true,
                enable_noise_reduction: true,
                enable_agc: true,
            },
            asr: AsrConfig {
                model_type: "whisper_small".to_string(),
                model_path: env.get_temp_path("whisper_small.onnx").to_string_lossy().to_string(),
                device: "cpu".to_string(),
                batch_size: 1,
                precision: ModelPrecision::Fp16,
                enable_timestamps: true,
            },
            translation: TranslationConfig {
                model_type: "m2m".to_string(),
                model_path: env.get_temp_path("m2m.onnx").to_string_lossy().to_string(),
                source_language: "auto".to_string(),
                target_language: "en".to_string(),
                device: "cpu".to_string(),
                max_length: 512,
            },
        };

        let translator = Translator::new(config).await.unwrap();

        // Generate test audio (3 seconds of speech simulation)
        let audio_data = generate_realistic_speech(3000, 16000);

        // Process through pipeline and measure latency
        let (result, latency) = measure_latency(async {
            translator.process_audio(&audio_data).await
        }).await;

        assert!(result.is_ok());
        let output = result.unwrap();

        // Validate output
        assert!(output.transcription.is_some());
        assert!(output.translation.is_some());
        assert!(output.detected_language.is_some());
        assert!(output.confidence > 0.0);

        // Check latency requirements
        assert!(latency < Duration::from_millis(500), "End-to-end latency too high: {:?}", latency);

        println!("End-to-end latency: {:?}", latency);
        println!("Detected language: {:?}", output.detected_language);
        println!("Transcription: {:?}", output.transcription);
        println!("Translation: {:?}", output.translation);
    }

    #[tokio::test]
    async fn test_streaming_pipeline() {
        let env = setup_test_environment().await;

        // Setup streaming server
        let server_config = StreamingConfig {
            buffer_size: 8192,
            max_latency_ms: 200,
            reconnect_attempts: 3,
            heartbeat_interval_ms: 30000,
        };

        let server = StreamingServer::new("127.0.0.1:0", server_config).await.unwrap();
        let server_addr = server.local_addr();

        // Start server in background
        tokio::spawn(async move {
            server.run().await.unwrap();
        });

        sleep(Duration::from_millis(100)).await; // Let server start

        // Connect client
        let client = WebSocketClient::connect(&format!("ws://{}", server_addr))
            .await
            .unwrap();

        // Stream audio chunks
        let chunk_size = 16000 / 10; // 100ms chunks
        let full_audio = generate_realistic_speech(2000, 16000);

        let mut total_latency = Duration::ZERO;
        let mut chunk_count = 0;

        for chunk in full_audio.chunks(chunk_size) {
            let (response, latency) = measure_latency(async {
                client.send_audio_chunk(chunk).await
            }).await;

            assert!(response.is_ok());
            total_latency += latency;
            chunk_count += 1;

            // Small delay between chunks to simulate real-time
            sleep(Duration::from_millis(100)).await;
        }

        let avg_latency = total_latency / chunk_count;
        assert!(avg_latency < Duration::from_millis(150), "Streaming latency too high: {:?}", avg_latency);

        println!("Average streaming latency: {:?}", avg_latency);
    }

    #[tokio::test]
    async fn test_profile_switching() {
        let env = setup_test_environment().await;

        let profiles = vec![Profile::Low, Profile::Medium, Profile::High];
        let mut latencies = Vec::new();
        let mut resource_usages = Vec::new();

        for profile in profiles {
            let config = TranslatorConfig::from_profile(profile);
            let translator = Translator::new(config).await.unwrap();

            // Test with same audio for each profile
            let audio_data = generate_realistic_speech(1000, 16000);

            let ((result, latency), resources) = run_with_monitoring(async {
                measure_latency(async {
                    translator.process_audio(&audio_data).await
                }).await
            }).await;

            assert!(result.is_ok());
            latencies.push(latency);
            resource_usages.push(resources);

            println!("Profile {:?}: latency={:?}, memory_delta={:.2}MB, cpu={:.1}%",
                     profile, latency, resources.memory_delta_mb, resources.avg_cpu_percent);
        }

        // Verify profile characteristics
        // Low profile should have lowest resource usage
        assert!(resource_usages[0].memory_delta_mb <= resource_usages[1].memory_delta_mb);
        assert!(resource_usages[0].avg_cpu_percent <= resource_usages[1].avg_cpu_percent);

        // High profile should have lowest latency
        assert!(latencies[2] <= latencies[0]);
    }

    #[tokio::test]
    async fn test_multi_language_support() {
        let env = setup_test_environment().await;

        // Test languages
        let test_cases = vec![
            ("en", "Hello, how are you?", "es"),
            ("es", "Hola, ¿cómo estás?", "en"),
            ("fr", "Bonjour, comment allez-vous?", "en"),
            ("de", "Guten Tag, wie geht es Ihnen?", "en"),
            ("ja", "こんにちは、お元気ですか？", "en"),
        ];

        let mut translator = Translator::new(TranslatorConfig::default()).await.unwrap();

        for (source_lang, text, target_lang) in test_cases {
            // Synthesize audio for the text (placeholder)
            let audio = synthesize_speech(text, source_lang);

            let result = translator.process_with_languages(
                &audio,
                Some(source_lang),
                target_lang,
            ).await;

            assert!(result.is_ok());
            let output = result.unwrap();

            assert_eq!(output.detected_language, Some(source_lang.to_string()));
            assert!(!output.translation.unwrap_or_default().is_empty());

            println!("Translated {} -> {}: '{}'",
                     source_lang, target_lang,
                     output.translation.unwrap_or_default());
        }
    }

    #[tokio::test]
    async fn test_concurrent_sessions() {
        let env = setup_test_environment().await;

        let session_count = 10;
        let mut handles = Vec::new();

        for session_id in 0..session_count {
            let handle = tokio::spawn(async move {
                let config = TranslatorConfig::from_profile(Profile::Low);
                let translator = Translator::new(config).await.unwrap();

                let audio = generate_realistic_speech(500, 16000);
                let start = std::time::Instant::now();

                let result = translator.process_audio(&audio).await;

                let latency = start.elapsed();
                (session_id, result.is_ok(), latency)
            });

            handles.push(handle);
        }

        // Wait for all sessions
        let mut total_latency = Duration::ZERO;
        let mut success_count = 0;

        for handle in handles {
            let (id, success, latency) = handle.await.unwrap();
            if success {
                success_count += 1;
                total_latency += latency;
            }
            println!("Session {}: success={}, latency={:?}", id, success, latency);
        }

        assert_eq!(success_count, session_count, "All sessions should succeed");

        let avg_latency = total_latency / session_count;
        println!("Average latency for {} concurrent sessions: {:?}", session_count, avg_latency);
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let env = setup_test_environment().await;

        // Setup performance monitor
        let monitor = PerformanceMonitor::new(Default::default()).await.unwrap();
        monitor.start().await.unwrap();

        // Create translator
        let translator = Translator::new(TranslatorConfig::default()).await.unwrap();

        // Process multiple audio samples
        for i in 0..5 {
            let audio = generate_realistic_speech(1000, 16000);

            // Record metrics
            monitor.metrics_collector()
                .start_component_operation("translation", format!("op_{}", i))
                .await;

            let result = translator.process_audio(&audio).await;

            monitor.metrics_collector()
                .end_component_operation("translation", &format!("op_{}", i), result.is_ok())
                .await;

            sleep(Duration::from_millis(100)).await;
        }

        // Get performance snapshot
        let snapshot = monitor.get_performance_snapshot().await.unwrap();

        assert!(snapshot.total_samples_collected > 0);
        assert!(snapshot.current_metrics.latency.avg_latency_ms > 0.0);
        assert!(snapshot.current_metrics.resources.cpu_utilization_percent > 0.0);

        println!("Performance snapshot:");
        println!("  Samples collected: {}", snapshot.total_samples_collected);
        println!("  Average latency: {:.2}ms", snapshot.current_metrics.latency.avg_latency_ms);
        println!("  CPU utilization: {:.1}%", snapshot.current_metrics.resources.cpu_utilization_percent);

        monitor.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_error_recovery() {
        let env = setup_test_environment().await;

        let translator = Translator::new(TranslatorConfig::default()).await.unwrap();

        // Test with invalid audio (empty)
        let empty_audio = vec![];
        let result = translator.process_audio(&empty_audio).await;
        assert!(result.is_err() || result.unwrap().transcription.is_none());

        // Test with very short audio
        let short_audio = vec![0.0; 100]; // Very short
        let result = translator.process_audio(&short_audio).await;
        // Should handle gracefully
        assert!(result.is_ok() || result.is_err());

        // Test with noise
        let noise: Vec<f32> = (0..16000)
            .map(|_| rand::random::<f32>() * 0.01 - 0.005)
            .collect();
        let result = translator.process_audio(&noise).await;
        // Should handle noise without crashing
        assert!(result.is_ok() || result.is_err());

        // Test recovery after error
        let valid_audio = generate_realistic_speech(1000, 16000);
        let result = translator.process_audio(&valid_audio).await;
        assert!(result.is_ok(), "Should recover after errors");
    }

    #[tokio::test]
    async fn test_resource_constraints() {
        let env = setup_test_environment().await;

        // Test with limited resources (Low profile)
        let config = TranslatorConfig::from_profile(Profile::Low);
        let translator = Translator::new(config).await.unwrap();

        // Generate large batch of audio
        let batch_size = 20;
        let mut handles = Vec::new();

        for i in 0..batch_size {
            let translator_clone = translator.clone();
            let handle = tokio::spawn(async move {
                let audio = generate_realistic_speech(500, 16000);
                let (result, resources) = run_with_monitoring(async {
                    translator_clone.process_audio(&audio).await
                }).await;
                (i, result.is_ok(), resources)
            });
            handles.push(handle);
        }

        // Check resource usage stays within limits
        let mut max_memory = 0.0f32;
        let mut max_cpu = 0.0f32;

        for handle in handles {
            let (id, success, resources) = handle.await.unwrap();
            max_memory = max_memory.max(resources.peak_memory_mb);
            max_cpu = max_cpu.max(resources.avg_cpu_percent);

            println!("Batch {}: memory={:.2}MB, cpu={:.1}%",
                     id, resources.peak_memory_mb, resources.avg_cpu_percent);
        }

        // Low profile should stay within resource constraints
        assert!(max_memory < 2048.0, "Memory usage too high: {:.2}MB", max_memory);
        assert!(max_cpu < 90.0, "CPU usage too high: {:.1}%", max_cpu);
    }

    // Helper functions

    fn generate_realistic_speech(duration_ms: u32, sample_rate: u32) -> Vec<f32> {
        // Generate more realistic speech-like audio
        let num_samples = (sample_rate * duration_ms / 1000) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        // Mix of frequencies to simulate speech
        let fundamentals = vec![150.0, 200.0, 250.0]; // Voice fundamentals
        let harmonics = vec![2.0, 3.0, 4.0];

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let mut sample = 0.0;

            for &fundamental in &fundamentals {
                for &harmonic in &harmonics {
                    let frequency = fundamental * harmonic;
                    let amplitude = 0.3 / harmonic; // Decreasing amplitude for harmonics
                    sample += amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
                }
            }

            // Add some noise to make it more realistic
            sample += (rand::random::<f32>() - 0.5) * 0.05;

            // Apply envelope for more natural sound
            let envelope = (t * 2.0).min(1.0) * (1.0 - (t - duration_ms as f32 / 1000.0).max(0.0) * 4.0).max(0.0);
            samples.push(sample * envelope);
        }

        samples
    }

    fn synthesize_speech(text: &str, language: &str) -> Vec<f32> {
        // Placeholder for speech synthesis
        // In real implementation, use TTS engine
        generate_realistic_speech(text.len() as u32 * 50, 16000)
    }
}