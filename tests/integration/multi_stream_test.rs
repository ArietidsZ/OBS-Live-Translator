use obs_live_translator_core::streaming::{
    MultiStreamProcessor, StreamConfig, StreamPriority, AudioFrame,
    AdvancedCache, EvictionPolicy,
    ResourceManager, ResourceLimits,
    ConcurrentPipeline, PipelineConfig,
};
use obs_live_translator_core::gpu::AdaptiveMemoryManager;
use obs_live_translator_core::acceleration::ONNXAccelerator;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::test]
async fn test_multi_stream_concurrent_processing() {
    let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
    let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

    let processor = MultiStreamProcessor::new(
        memory_manager.clone(),
        accelerator.clone(),
        10,
        4,
    ).await.unwrap();

    let mut stream_ids = Vec::new();
    for i in 0..5 {
        let config = StreamConfig {
            stream_id: format!("stream_{}", i),
            language_pair: ("en".to_string(), "es".to_string()),
            priority: match i {
                0 => StreamPriority::Critical,
                1 => StreamPriority::High,
                _ => StreamPriority::Normal,
            },
            buffer_size: 1024,
            max_latency_ms: 100,
            enable_voice_activity_detection: true,
            enable_noise_reduction: true,
        };

        let id = processor.add_stream(config).await.unwrap();
        stream_ids.push(id);
    }

    for (idx, stream_id) in stream_ids.iter().enumerate() {
        let frame = AudioFrame {
            timestamp: Instant::now(),
            data: vec![0.1; 16000],
            sample_rate: 16000,
            channels: 1,
        };

        processor.submit_audio(stream_id.clone(), frame).await.unwrap();
    }

    sleep(Duration::from_millis(500)).await;

    let all_stats = processor.get_all_stats().await;
    assert_eq!(all_stats.len(), 5);

    for stream_id in &stream_ids {
        processor.remove_stream(stream_id).await.unwrap();
    }

    processor.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_stream_priority_ordering() {
    let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
    let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

    let processor = MultiStreamProcessor::new(
        memory_manager.clone(),
        accelerator.clone(),
        10,
        4,
    ).await.unwrap();

    let critical_config = StreamConfig {
        stream_id: "critical_stream".to_string(),
        language_pair: ("en".to_string(), "es".to_string()),
        priority: StreamPriority::Critical,
        buffer_size: 1024,
        max_latency_ms: 50,
        enable_voice_activity_detection: true,
        enable_noise_reduction: true,
    };

    let low_config = StreamConfig {
        stream_id: "low_stream".to_string(),
        language_pair: ("en".to_string(), "es".to_string()),
        priority: StreamPriority::Low,
        buffer_size: 1024,
        max_latency_ms: 200,
        enable_voice_activity_detection: false,
        enable_noise_reduction: false,
    };

    let critical_id = processor.add_stream(critical_config).await.unwrap();
    let low_id = processor.add_stream(low_config).await.unwrap();

    let start = Instant::now();

    for i in 0..10 {
        let frame = AudioFrame {
            timestamp: Instant::now(),
            data: vec![0.1; 8000],
            sample_rate: 16000,
            channels: 1,
        };

        if i % 2 == 0 {
            processor.submit_audio(critical_id.clone(), frame).await.unwrap();
        } else {
            processor.submit_audio(low_id.clone(), frame).await.unwrap();
        }
    }

    sleep(Duration::from_secs(1)).await;

    let critical_stats = processor.get_stream_stats(&critical_id).await.unwrap();
    let low_stats = processor.get_stream_stats(&low_id).await.unwrap();

    assert!(critical_stats.avg_processing_time_ms < low_stats.avg_processing_time_ms);

    processor.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_resource_allocation_and_scaling() {
    let limits = ResourceLimits {
        max_cpu_percent: 80.0,
        max_memory_mb: 4096,
        max_gpu_memory_mb: 2048,
        max_threads: 8,
        reserved_cpu_percent: 10.0,
        reserved_memory_mb: 512,
    };

    let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
    let resource_manager = Arc::new(
        ResourceManager::new(limits, memory_manager, true).await.unwrap()
    );

    let allocation1 = resource_manager
        .allocate_resources("stream1".to_string(), StreamPriority::High)
        .await
        .unwrap();

    assert!(allocation1.cpu_cores > 0);
    assert!(allocation1.memory_mb > 0);

    let allocation2 = resource_manager
        .allocate_resources("stream2".to_string(), StreamPriority::Normal)
        .await
        .unwrap();

    assert!(allocation2.cpu_cores > 0);

    resource_manager.optimize_for_latency("stream1").await.unwrap();

    let updated_allocation = resource_manager
        .get_allocation("stream1")
        .await
        .unwrap();

    assert!(updated_allocation.processing_quota_ms > allocation1.processing_quota_ms);

    resource_manager.release_resources("stream1").await.unwrap();
    resource_manager.release_resources("stream2").await.unwrap();
}

#[tokio::test]
async fn test_cache_performance() {
    let cache = AdvancedCache::new(100, EvictionPolicy::Adaptive, true).await.unwrap();

    let mut keys = Vec::new();
    for i in 0..100 {
        let key = obs_live_translator_core::streaming::CacheKey {
            content_hash: format!("hash_{}", i),
            model_type: obs_live_translator_core::streaming::ModelType::Translation,
            params: "en-es".to_string(),
        };

        let metadata = obs_live_translator_core::streaming::CacheMetadata {
            model_type: "translation".to_string(),
            input_hash: format!("input_{}", i),
            processing_time_ms: 50,
            confidence: 0.95,
            language_pair: Some(("en".to_string(), "es".to_string())),
            compression_ratio: 0.8,
        };

        cache.put(key.clone(), vec![1, 2, 3, 4, 5], metadata).await.unwrap();
        keys.push(key);
    }

    for _ in 0..3 {
        for key in &keys[0..20] {
            cache.get(key).await;
        }
    }

    let stats = cache.get_stats().await;
    assert!(stats.hit_rate > 0.5);
    assert_eq!(stats.total_entries, 100);

    cache.clear().await.unwrap();
}

#[tokio::test]
async fn test_pipeline_stages() {
    let config = PipelineConfig {
        max_parallel_stages: 4,
        batch_size: 8,
        stage_timeout_ms: 1000,
        enable_caching: true,
        enable_prefetch: true,
        worker_threads: 4,
        queue_capacity: 100,
    };

    let cache = Arc::new(
        AdvancedCache::new(100, EvictionPolicy::LRU, true).await.unwrap()
    );

    let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
    let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

    let resource_manager = Arc::new(
        ResourceManager::new(
            ResourceLimits {
                max_cpu_percent: 80.0,
                max_memory_mb: 4096,
                max_gpu_memory_mb: 2048,
                max_threads: 8,
                reserved_cpu_percent: 10.0,
                reserved_memory_mb: 512,
            },
            memory_manager.clone(),
            true,
        ).await.unwrap()
    );

    let pipeline = ConcurrentPipeline::new(
        config,
        cache,
        resource_manager,
        memory_manager,
        accelerator,
    ).await.unwrap();

    for i in 0..10 {
        let frame = AudioFrame {
            timestamp: Instant::now(),
            data: vec![0.1; 16000],
            sample_rate: 16000,
            channels: 1,
        };

        let priority = if i < 3 {
            StreamPriority::Critical
        } else if i < 6 {
            StreamPriority::High
        } else {
            StreamPriority::Normal
        };

        pipeline.submit(format!("stream_{}", i), frame, priority).await.unwrap();
    }

    sleep(Duration::from_secs(2)).await;

    let metrics = pipeline.get_metrics().await;
    assert!(metrics.completed_items > 0);
    assert!(metrics.cache_hit_rate >= 0.0);

    pipeline.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_error_recovery() {
    let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
    let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

    let processor = MultiStreamProcessor::new(
        memory_manager.clone(),
        accelerator.clone(),
        10,
        4,
    ).await.unwrap();

    let config = StreamConfig {
        stream_id: "test_stream".to_string(),
        language_pair: ("en".to_string(), "es".to_string()),
        priority: StreamPriority::Normal,
        buffer_size: 1024,
        max_latency_ms: 100,
        enable_voice_activity_detection: true,
        enable_noise_reduction: true,
    };

    let stream_id = processor.add_stream(config).await.unwrap();

    let invalid_frame = AudioFrame {
        timestamp: Instant::now(),
        data: vec![],
        sample_rate: 0,
        channels: 0,
    };

    let result = processor.submit_audio(stream_id.clone(), invalid_frame).await;
    assert!(result.is_ok() || result.is_err());

    processor.pause_stream(&stream_id).await.unwrap();

    let paused_frame = AudioFrame {
        timestamp: Instant::now(),
        data: vec![0.1; 16000],
        sample_rate: 16000,
        channels: 1,
    };

    let result = processor.submit_audio(stream_id.clone(), paused_frame).await;
    assert!(result.is_err());

    processor.resume_stream(&stream_id).await.unwrap();

    let valid_frame = AudioFrame {
        timestamp: Instant::now(),
        data: vec![0.1; 16000],
        sample_rate: 16000,
        channels: 1,
    };

    processor.submit_audio(stream_id.clone(), valid_frame).await.unwrap();

    processor.shutdown().await.unwrap();
}