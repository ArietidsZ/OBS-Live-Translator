use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use obs_live_translator_core::streaming::{
    MultiStreamProcessor, StreamConfig, StreamPriority, AudioFrame,
    AdvancedCache, EvictionPolicy, CacheKey, ModelType, CacheMetadata,
    ConcurrentPipeline, PipelineConfig,
};
use obs_live_translator_core::gpu::AdaptiveMemoryManager;
use obs_live_translator_core::acceleration::ONNXAccelerator;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

fn benchmark_multi_stream_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("multi_stream_throughput");

    for num_streams in [1, 5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*num_streams as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_streams),
            num_streams,
            |b, &num_streams| {
                b.to_async(&rt).iter(|| async move {
                    let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
                    let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

                    let processor = MultiStreamProcessor::new(
                        memory_manager.clone(),
                        accelerator.clone(),
                        20,
                        8,
                    ).await.unwrap();

                    let mut stream_ids = Vec::new();
                    for i in 0..num_streams {
                        let config = StreamConfig {
                            stream_id: format!("bench_stream_{}", i),
                            language_pair: ("en".to_string(), "es".to_string()),
                            priority: StreamPriority::Normal,
                            buffer_size: 1024,
                            max_latency_ms: 100,
                            enable_voice_activity_detection: false,
                            enable_noise_reduction: false,
                        };

                        let id = processor.add_stream(config).await.unwrap();
                        stream_ids.push(id);
                    }

                    for stream_id in &stream_ids {
                        let frame = AudioFrame {
                            timestamp: Instant::now(),
                            data: vec![0.1; 16000],
                            sample_rate: 16000,
                            channels: 1,
                        };

                        processor.submit_audio(stream_id.clone(), frame).await.unwrap();
                    }

                    processor.shutdown().await.unwrap();
                });
            },
        );
    }
    group.finish();
}

fn benchmark_cache_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("cache_operations");

    group.bench_function("cache_write", |b| {
        b.to_async(&rt).iter(|| async {
            let cache = AdvancedCache::new(1000, EvictionPolicy::LRU, false).await.unwrap();

            for i in 0..100 {
                let key = CacheKey {
                    content_hash: format!("bench_hash_{}", i),
                    model_type: ModelType::Translation,
                    params: "en-es".to_string(),
                };

                let metadata = CacheMetadata {
                    model_type: "translation".to_string(),
                    input_hash: format!("input_{}", i),
                    processing_time_ms: 50,
                    confidence: 0.95,
                    language_pair: Some(("en".to_string(), "es".to_string())),
                    compression_ratio: 0.8,
                };

                cache.put(key, black_box(vec![1, 2, 3, 4, 5]), metadata).await.unwrap();
            }
        });
    });

    group.bench_function("cache_read", |b| {
        b.to_async(&rt).iter(|| async {
            let cache = AdvancedCache::new(1000, EvictionPolicy::LRU, false).await.unwrap();

            let mut keys = Vec::new();
            for i in 0..100 {
                let key = CacheKey {
                    content_hash: format!("bench_hash_{}", i),
                    model_type: ModelType::Translation,
                    params: "en-es".to_string(),
                };

                let metadata = CacheMetadata {
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

            for key in &keys {
                black_box(cache.get(key).await);
            }
        });
    });

    group.bench_function("cache_with_compression", |b| {
        b.to_async(&rt).iter(|| async {
            let cache = AdvancedCache::new(1000, EvictionPolicy::LRU, true).await.unwrap();

            for i in 0..50 {
                let key = CacheKey {
                    content_hash: format!("compressed_{}", i),
                    model_type: ModelType::Translation,
                    params: "en-es".to_string(),
                };

                let metadata = CacheMetadata {
                    model_type: "translation".to_string(),
                    input_hash: format!("input_{}", i),
                    processing_time_ms: 50,
                    confidence: 0.95,
                    language_pair: Some(("en".to_string(), "es".to_string())),
                    compression_ratio: 0.8,
                };

                let data = vec![0u8; 1000];
                cache.put(key, black_box(data), metadata).await.unwrap();
            }
        });
    });

    group.finish();
}

fn benchmark_pipeline_stages(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("pipeline_stages");

    for batch_size in [1, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async move {
                    let config = PipelineConfig {
                        max_parallel_stages: 4,
                        batch_size,
                        stage_timeout_ms: 1000,
                        enable_caching: true,
                        enable_prefetch: false,
                        worker_threads: 4,
                        queue_capacity: 100,
                    };

                    let cache = Arc::new(
                        AdvancedCache::new(100, EvictionPolicy::LRU, false).await.unwrap()
                    );

                    let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
                    let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

                    let resource_manager = Arc::new(
                        obs_live_translator_core::streaming::ResourceManager::new(
                            obs_live_translator_core::streaming::ResourceLimits {
                                max_cpu_percent: 80.0,
                                max_memory_mb: 4096,
                                max_gpu_memory_mb: 2048,
                                max_threads: 8,
                                reserved_cpu_percent: 10.0,
                                reserved_memory_mb: 512,
                            },
                            memory_manager.clone(),
                            false,
                        ).await.unwrap()
                    );

                    let pipeline = ConcurrentPipeline::new(
                        config,
                        cache,
                        resource_manager,
                        memory_manager,
                        accelerator,
                    ).await.unwrap();

                    for i in 0..batch_size {
                        let frame = AudioFrame {
                            timestamp: Instant::now(),
                            data: vec![0.1; 8000],
                            sample_rate: 16000,
                            channels: 1,
                        };

                        pipeline.submit(
                            format!("bench_stream_{}", i),
                            frame,
                            StreamPriority::Normal,
                        ).await.unwrap();
                    }

                    tokio::time::sleep(Duration::from_millis(100)).await;
                    pipeline.shutdown().await.unwrap();
                });
            },
        );
    }
    group.finish();
}

fn benchmark_memory_allocation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_allocation");

    group.bench_function("adaptive_memory_allocation", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = AdaptiveMemoryManager::new().await.unwrap();

            for size in [1024, 4096, 16384, 65536].iter() {
                black_box(manager.allocate_memory(*size).await.unwrap());
            }
        });
    });

    group.bench_function("memory_pool_operations", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = AdaptiveMemoryManager::new().await.unwrap();
            let pool = manager.create_memory_pool(1024 * 1024).await.unwrap();

            for _ in 0..100 {
                let allocation = pool.allocate(1024).await.unwrap();
                black_box(&allocation);
                pool.deallocate(allocation).await.unwrap();
            }
        });
    });

    group.finish();
}

fn benchmark_priority_scheduling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("priority_scheduling");

    group.bench_function("mixed_priority_streams", |b| {
        b.to_async(&rt).iter(|| async {
            let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
            let accelerator = Arc::new(ONNXAccelerator::new(Default::default()).await.unwrap());

            let processor = MultiStreamProcessor::new(
                memory_manager.clone(),
                accelerator.clone(),
                10,
                4,
            ).await.unwrap();

            let priorities = [
                StreamPriority::Critical,
                StreamPriority::High,
                StreamPriority::Normal,
                StreamPriority::Low,
            ];

            let mut stream_ids = Vec::new();
            for (i, priority) in priorities.iter().enumerate() {
                let config = StreamConfig {
                    stream_id: format!("priority_stream_{}", i),
                    language_pair: ("en".to_string(), "es".to_string()),
                    priority: *priority,
                    buffer_size: 1024,
                    max_latency_ms: 100,
                    enable_voice_activity_detection: false,
                    enable_noise_reduction: false,
                };

                let id = processor.add_stream(config).await.unwrap();
                stream_ids.push(id);
            }

            for _ in 0..10 {
                for stream_id in &stream_ids {
                    let frame = AudioFrame {
                        timestamp: Instant::now(),
                        data: vec![0.1; 4000],
                        sample_rate: 16000,
                        channels: 1,
                    };

                    processor.submit_audio(stream_id.clone(), frame).await.unwrap();
                }
            }

            processor.shutdown().await.unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_multi_stream_throughput,
    benchmark_cache_operations,
    benchmark_pipeline_stages,
    benchmark_memory_allocation,
    benchmark_priority_scheduling
);

criterion_main!(benches);