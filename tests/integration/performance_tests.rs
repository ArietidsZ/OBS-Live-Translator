//! Integration tests for performance validation

#[cfg(test)]
mod tests {
    use obs_live_translator::{
        Translator, TranslatorConfig,
        profile::Profile,
        monitoring::PerformanceMonitor,
    };
    use std::time::{Duration, Instant};
    use crate::integration_test_utils::*;

    #[tokio::test]
    async fn test_real_time_processing_performance() {
        let env = setup_test_environment().await;

        // Test real-time factor for each profile
        let profiles = vec![Profile::Low, Profile::Medium, Profile::High];
        let mut results = Vec::new();

        for profile in profiles {
            let config = TranslatorConfig::from_profile(profile);
            let translator = Translator::new(config).await.unwrap();

            // Generate 10 seconds of audio
            let audio_duration_s = 10.0;
            let sample_rate = 16000;
            let audio = generate_realistic_speech(
                (audio_duration_s * 1000.0) as u32,
                sample_rate
            );

            // Process and measure time
            let start = Instant::now();
            let result = translator.process_audio(&audio).await;
            let processing_time = start.elapsed();

            assert!(result.is_ok());

            // Calculate real-time factor
            let rtf = processing_time.as_secs_f64() / audio_duration_s;
            results.push((profile, rtf));

            println!("Profile {:?}: RTF = {:.3}x ({})",
                     profile, rtf,
                     if rtf < 1.0 { "real-time" } else { "slower than real-time" });

            // Verify real-time constraints
            match profile {
                Profile::Low => assert!(rtf < 0.5, "Low profile should be < 0.5x real-time"),
                Profile::Medium => assert!(rtf < 0.75, "Medium profile should be < 0.75x real-time"),
                Profile::High => assert!(rtf < 1.0, "High profile should be < 1.0x real-time"),
            }
        }
    }

    #[tokio::test]
    async fn test_memory_consumption() {
        let env = setup_test_environment().await;

        // Monitor memory usage across profiles
        let profiles = vec![Profile::Low, Profile::Medium, Profile::High];
        let mut memory_usage = Vec::new();

        for profile in profiles {
            let (result, resources) = run_with_monitoring(async {
                let config = TranslatorConfig::from_profile(profile);
                let translator = Translator::new(config).await.unwrap();

                // Process multiple audio samples
                for _ in 0..10 {
                    let audio = generate_realistic_speech(1000, 16000);
                    let _ = translator.process_audio(&audio).await;
                }

                Ok::<(), Box<dyn std::error::Error>>(())
            }).await;

            assert!(result.is_ok());
            memory_usage.push((profile, resources.peak_memory_mb));

            println!("Profile {:?}: Peak memory = {:.2} MB", profile, resources.peak_memory_mb);

            // Verify memory constraints
            match profile {
                Profile::Low => assert!(resources.peak_memory_mb < 512.0, "Low profile should use < 512MB"),
                Profile::Medium => assert!(resources.peak_memory_mb < 1024.0, "Medium profile should use < 1GB"),
                Profile::High => assert!(resources.peak_memory_mb < 2048.0, "High profile should use < 2GB"),
            }
        }
    }

    #[tokio::test]
    async fn test_latency_targets() {
        let env = setup_test_environment().await;

        // Component latency targets (in milliseconds)
        let targets = vec![
            ("audio_processing", 10.0),
            ("asr_inference", 50.0),
            ("translation", 30.0),
            ("end_to_end", 100.0),
        ];

        let config = TranslatorConfig::from_profile(Profile::Medium);
        let translator = Translator::new(config).await.unwrap();

        // Process multiple samples and track latencies
        let mut latencies = std::collections::HashMap::new();

        for i in 0..20 {
            let audio = generate_realistic_speech(500, 16000);
            
            let start = Instant::now();
            let result = translator.process_audio(&audio).await;
            let e2e_latency = start.elapsed();

            assert!(result.is_ok());
            
            latencies.entry("end_to_end")
                .or_insert_with(Vec::new)
                .push(e2e_latency.as_secs_f64() * 1000.0);
        }

        // Check P95 latencies against targets
        for (component, target_ms) in targets {
            if let Some(measurements) = latencies.get(component) {
                let mut sorted = measurements.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let p95_index = (sorted.len() as f64 * 0.95) as usize;
                let p95_latency = sorted[p95_index.min(sorted.len() - 1)];

                println!("Component {}: P95 latency = {:.2}ms (target: {:.0}ms)",
                         component, p95_latency, target_ms);

                // Allow 20% tolerance
                assert!(p95_latency < target_ms * 1.2,
                        "{} P95 latency {:.2}ms exceeds target {:.0}ms",
                        component, p95_latency, target_ms);
            }
        }
    }

    #[tokio::test]
    async fn test_throughput_scaling() {
        let env = setup_test_environment().await;

        // Test how throughput scales with batch size
        let batch_sizes = vec![1, 2, 4, 8];
        let mut throughput_results = Vec::new();

        for batch_size in batch_sizes {
            let config = TranslatorConfig {
                profile: Profile::Medium,
                batch_size,
                ..Default::default()
            };

            let translator = Translator::new(config).await.unwrap();

            // Create batch of audio samples
            let audio_samples: Vec<_> = (0..batch_size)
                .map(|_| generate_realistic_speech(1000, 16000))
                .collect();

            let start = Instant::now();
            let mut success_count = 0;

            for audio in &audio_samples {
                if translator.process_audio(audio).await.is_ok() {
                    success_count += 1;
                }
            }

            let elapsed = start.elapsed();
            let throughput = success_count as f64 / elapsed.as_secs_f64();

            throughput_results.push((batch_size, throughput));

            println!("Batch size {}: {:.2} samples/second", batch_size, throughput);
        }

        // Verify throughput improves with batch size
        for i in 1..throughput_results.len() {
            assert!(throughput_results[i].1 >= throughput_results[i-1].1 * 0.9,
                    "Throughput should not degrade with larger batch sizes");
        }
    }

    #[tokio::test]
    async fn test_performance_under_load() {
        let env = setup_test_environment().await;

        // Create performance monitor
        let monitor = PerformanceMonitor::new(Default::default()).await.unwrap();
        monitor.start().await.unwrap();

        // Simulate high load
        let num_concurrent = 5;
        let mut handles = Vec::new();

        for i in 0..num_concurrent {
            let handle = tokio::spawn(async move {
                let config = TranslatorConfig::from_profile(Profile::Low);
                let translator = Translator::new(config).await.unwrap();

                let mut latencies = Vec::new();

                for _ in 0..10 {
                    let audio = generate_realistic_speech(500, 16000);
                    let start = Instant::now();
                    let _ = translator.process_audio(&audio).await;
                    latencies.push(start.elapsed());
                }

                latencies
            });
            handles.push(handle);
        }

        // Wait for all tasks
        let mut all_latencies = Vec::new();
        for handle in handles {
            let latencies = handle.await.unwrap();
            all_latencies.extend(latencies);
        }

        // Get performance metrics
        let snapshot = monitor.get_performance_snapshot().await.unwrap();
        monitor.stop().await.unwrap();

        // Calculate statistics
        let avg_latency: Duration = all_latencies.iter().sum::<Duration>() / all_latencies.len() as u32;
        let max_latency = all_latencies.iter().max().unwrap();

        println!("Under load ({} concurrent):", num_concurrent);
        println!("  Average latency: {:?}", avg_latency);
        println!("  Max latency: {:?}", max_latency);
        println!("  CPU utilization: {:.1}%", snapshot.current_metrics.resources.cpu_utilization_percent);
        println!("  Memory usage: {:.0} MB", snapshot.current_metrics.resources.memory_usage_mb);

        // Verify performance under load
        assert!(avg_latency < Duration::from_millis(200), "Average latency too high under load");
        assert!(max_latency < Duration::from_millis(500), "Max latency too high under load");
        assert!(snapshot.current_metrics.resources.cpu_utilization_percent < 90.0, "CPU utilization too high");
    }

    #[tokio::test] 
    async fn test_performance_regression_detection() {
        // This test would normally compare against baseline performance
        // For now, just verify that performance tracking works
        
        let env = setup_test_environment().await;
        let config = TranslatorConfig::from_profile(Profile::Low);
        let translator = Translator::new(config).await.unwrap();

        // Collect performance samples
        let mut samples = Vec::new();
        
        for i in 0..10 {
            let audio = generate_realistic_speech(500, 16000);
            let start = Instant::now();
            let result = translator.process_audio(&audio).await;
            let latency = start.elapsed();
            
            if result.is_ok() {
                samples.push(latency.as_secs_f64() * 1000.0);
            }
        }

        // Calculate basic statistics
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();

        println!("Performance baseline:");
        println!("  Mean latency: {:.2}ms", mean);
        println!("  Std deviation: {:.2}ms", std_dev);
        println!("  CV: {:.2}%", (std_dev / mean) * 100.0);

        // Verify consistency (coefficient of variation < 20%)
        assert!((std_dev / mean) < 0.2, "Performance too variable");
    }

    fn generate_realistic_speech(duration_ms: u32, sample_rate: u32) -> Vec<f32> {
        let num_samples = (sample_rate * duration_ms / 1000) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        // Generate speech-like audio with formants
        let fundamentals = vec![150.0, 200.0, 250.0];
        let harmonics = vec![2.0, 3.0, 4.0];

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let mut sample = 0.0;

            for &fundamental in &fundamentals {
                for &harmonic in &harmonics {
                    let frequency = fundamental * harmonic;
                    let amplitude = 0.3 / harmonic;
                    sample += amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
                }
            }

            // Add noise
            sample += (rand::random::<f32>() - 0.5) * 0.05;

            // Apply envelope
            let envelope = (t * 2.0).min(1.0) * (1.0 - (t - duration_ms as f32 / 1000.0).max(0.0) * 4.0).max(0.0);
            samples.push(sample * envelope);
        }

        samples
    }
}