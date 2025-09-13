use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};

use crate::gpu::AdaptiveMemoryManager;
use crate::streaming::{MultiStreamProcessor, ResourceManager, ConcurrentPipeline, AdvancedCache};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: SystemTime,
    pub cpu_usage: f32,
    pub memory_usage: MemoryMetrics,
    pub gpu_metrics: GPUMetrics,
    pub stream_metrics: StreamMetrics,
    pub pipeline_metrics: PipelineMetrics,
    pub cache_metrics: CacheMetrics,
    pub latency_metrics: LatencyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub total_mb: usize,
    pub used_mb: usize,
    pub available_mb: usize,
    pub cache_mb: usize,
    pub buffer_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUMetrics {
    pub utilization_percent: f32,
    pub memory_used_mb: usize,
    pub memory_total_mb: usize,
    pub temperature_celsius: f32,
    pub power_watts: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    pub active_streams: usize,
    pub total_processed: u64,
    pub dropped_frames: u64,
    pub avg_processing_time_ms: f64,
    pub throughput_fps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub stages_active: usize,
    pub queue_depth: usize,
    pub items_processed: u64,
    pub cache_hit_rate: f32,
    pub avg_stage_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub entries: usize,
    pub size_mb: usize,
    pub hit_rate: f32,
    pub evictions: u64,
    pub warming_queue: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

pub struct PerformanceMonitor {
    metrics_history: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    memory_manager: Arc<AdaptiveMemoryManager>,
    resource_manager: Option<Arc<ResourceManager>>,
    stream_processor: Option<Arc<MultiStreamProcessor>>,
    pipeline: Option<Arc<ConcurrentPipeline>>,
    cache: Option<Arc<AdvancedCache>>,
    monitoring_interval: Duration,
    history_size: usize,
    alert_thresholds: AlertThresholds,
    worker_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
    shutdown_signal: Arc<AtomicBool>,
    metrics_counter: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub gpu_percent: f32,
    pub latency_ms: f64,
    pub drop_rate_percent: f32,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_percent: 90.0,
            memory_percent: 85.0,
            gpu_percent: 95.0,
            latency_ms: 100.0,
            drop_rate_percent: 5.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub timestamp: SystemTime,
    pub severity: AlertSeverity,
    pub component: String,
    pub message: String,
    pub metric_value: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl PerformanceMonitor {
    pub async fn new(
        memory_manager: Arc<AdaptiveMemoryManager>,
        monitoring_interval: Duration,
        history_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(history_size))),
            alerts: Arc::new(RwLock::new(Vec::new())),
            memory_manager,
            resource_manager: None,
            stream_processor: None,
            pipeline: None,
            cache: None,
            monitoring_interval,
            history_size,
            alert_thresholds: AlertThresholds::default(),
            worker_handle: Arc::new(Mutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            metrics_counter: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn set_resource_manager(&mut self, manager: Arc<ResourceManager>) {
        self.resource_manager = Some(manager);
    }

    pub fn set_stream_processor(&mut self, processor: Arc<MultiStreamProcessor>) {
        self.stream_processor = Some(processor);
    }

    pub fn set_pipeline(&mut self, pipeline: Arc<ConcurrentPipeline>) {
        self.pipeline = Some(pipeline);
    }

    pub fn set_cache(&mut self, cache: Arc<AdvancedCache>) {
        self.cache = Some(cache);
    }

    pub async fn start(&self) -> Result<()> {
        let metrics_history = self.metrics_history.clone();
        let alerts = self.alerts.clone();
        let memory_manager = self.memory_manager.clone();
        let resource_manager = self.resource_manager.clone();
        let stream_processor = self.stream_processor.clone();
        let pipeline = self.pipeline.clone();
        let cache = self.cache.clone();
        let interval = self.monitoring_interval;
        let shutdown_signal = self.shutdown_signal.clone();
        let alert_thresholds = self.alert_thresholds.clone();
        let metrics_counter = self.metrics_counter.clone();
        let history_size = self.history_size;

        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                if shutdown_signal.load(Ordering::Relaxed) {
                    break;
                }

                interval_timer.tick().await;

                let metrics = Self::collect_metrics(
                    &memory_manager,
                    &resource_manager,
                    &stream_processor,
                    &pipeline,
                    &cache,
                ).await;

                if let Ok(metrics) = metrics {
                    Self::check_alerts(&metrics, &alert_thresholds, &alerts).await;

                    let mut history = metrics_history.write().await;
                    history.push_back(metrics);

                    if history.len() > history_size {
                        history.pop_front();
                    }

                    metrics_counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        *self.worker_handle.lock().await = Some(handle);
        log::info!("Performance monitoring started");
        Ok(())
    }

    async fn collect_metrics(
        memory_manager: &Arc<AdaptiveMemoryManager>,
        resource_manager: &Option<Arc<ResourceManager>>,
        stream_processor: &Option<Arc<MultiStreamProcessor>>,
        pipeline: &Option<Arc<ConcurrentPipeline>>,
        cache: &Option<Arc<AdvancedCache>>,
    ) -> Result<PerformanceMetrics> {
        let memory_stats = memory_manager.get_memory_stats().await?;

        let gpu_metrics = GPUMetrics {
            utilization_percent: 0.0,
            memory_used_mb: (memory_stats.used / 1024 / 1024) as usize,
            memory_total_mb: (memory_stats.total / 1024 / 1024) as usize,
            temperature_celsius: 0.0,
            power_watts: 0.0,
        };

        let stream_metrics = if let Some(processor) = stream_processor {
            let stats = processor.get_all_stats().await;

            let total_frames: u64 = stats.iter().map(|s| s.total_frames).sum();
            let dropped_frames: u64 = stats.iter().map(|s| s.dropped_frames).sum();
            let avg_time: f64 = if !stats.is_empty() {
                stats.iter().map(|s| s.avg_processing_time_ms).sum::<f64>() / stats.len() as f64
            } else {
                0.0
            };

            StreamMetrics {
                active_streams: stats.len(),
                total_processed: total_frames,
                dropped_frames,
                avg_processing_time_ms: avg_time,
                throughput_fps: 30.0,
            }
        } else {
            StreamMetrics {
                active_streams: 0,
                total_processed: 0,
                dropped_frames: 0,
                avg_processing_time_ms: 0.0,
                throughput_fps: 0.0,
            }
        };

        let pipeline_metrics = if let Some(pipeline) = pipeline {
            let report = pipeline.get_metrics().await;

            PipelineMetrics {
                stages_active: 5,
                queue_depth: report.pending_items,
                items_processed: report.completed_items as u64,
                cache_hit_rate: report.cache_hit_rate,
                avg_stage_time_ms: report.avg_latency_ms,
            }
        } else {
            PipelineMetrics {
                stages_active: 0,
                queue_depth: 0,
                items_processed: 0,
                cache_hit_rate: 0.0,
                avg_stage_time_ms: 0.0,
            }
        };

        let cache_metrics = if let Some(cache) = cache {
            let stats = cache.get_stats().await;

            CacheMetrics {
                entries: stats.total_entries,
                size_mb: stats.total_size_bytes / 1024 / 1024,
                hit_rate: stats.hit_rate,
                evictions: stats.eviction_count,
                warming_queue: 0,
            }
        } else {
            CacheMetrics {
                entries: 0,
                size_mb: 0,
                hit_rate: 0.0,
                evictions: 0,
                warming_queue: 0,
            }
        };

        let system_stats = if let Some(manager) = resource_manager {
            manager.get_system_stats().await?
        } else {
            crate::streaming::SystemStats {
                cpu_usage_percent: 0.0,
                memory_used_mb: 0,
                memory_total_mb: 0,
                gpu_memory_used_mb: 0,
                gpu_memory_total_mb: 0,
                active_streams: 0,
            }
        };

        Ok(PerformanceMetrics {
            timestamp: SystemTime::now(),
            cpu_usage: system_stats.cpu_usage_percent,
            memory_usage: MemoryMetrics {
                total_mb: system_stats.memory_total_mb,
                used_mb: system_stats.memory_used_mb,
                available_mb: system_stats.memory_total_mb - system_stats.memory_used_mb,
                cache_mb: cache_metrics.size_mb,
                buffer_mb: 0,
            },
            gpu_metrics,
            stream_metrics,
            pipeline_metrics,
            cache_metrics,
            latency_metrics: LatencyMetrics {
                p50_ms: stream_metrics.avg_processing_time_ms * 0.8,
                p90_ms: stream_metrics.avg_processing_time_ms * 1.2,
                p95_ms: stream_metrics.avg_processing_time_ms * 1.5,
                p99_ms: stream_metrics.avg_processing_time_ms * 2.0,
                max_ms: stream_metrics.avg_processing_time_ms * 3.0,
            },
        })
    }

    async fn check_alerts(
        metrics: &PerformanceMetrics,
        thresholds: &AlertThresholds,
        alerts: &Arc<RwLock<Vec<Alert>>>,
    ) {
        let mut new_alerts = Vec::new();

        if metrics.cpu_usage > thresholds.cpu_percent {
            new_alerts.push(Alert {
                timestamp: SystemTime::now(),
                severity: AlertSeverity::Warning,
                component: "CPU".to_string(),
                message: format!("CPU usage {}% exceeds threshold", metrics.cpu_usage),
                metric_value: metrics.cpu_usage as f64,
                threshold: thresholds.cpu_percent as f64,
            });
        }

        let memory_percent = (metrics.memory_usage.used_mb as f32 / metrics.memory_usage.total_mb as f32) * 100.0;
        if memory_percent > thresholds.memory_percent {
            new_alerts.push(Alert {
                timestamp: SystemTime::now(),
                severity: AlertSeverity::Warning,
                component: "Memory".to_string(),
                message: format!("Memory usage {:.1}% exceeds threshold", memory_percent),
                metric_value: memory_percent as f64,
                threshold: thresholds.memory_percent as f64,
            });
        }

        if metrics.latency_metrics.p99_ms > thresholds.latency_ms {
            new_alerts.push(Alert {
                timestamp: SystemTime::now(),
                severity: AlertSeverity::Critical,
                component: "Latency".to_string(),
                message: format!("P99 latency {:.1}ms exceeds threshold", metrics.latency_metrics.p99_ms),
                metric_value: metrics.latency_metrics.p99_ms,
                threshold: thresholds.latency_ms,
            });
        }

        if !new_alerts.is_empty() {
            let mut alerts_guard = alerts.write().await;
            alerts_guard.extend(new_alerts);

            if alerts_guard.len() > 1000 {
                alerts_guard.drain(0..100);
            }
        }
    }

    pub async fn get_current_metrics(&self) -> Option<PerformanceMetrics> {
        let history = self.metrics_history.read().await;
        history.back().cloned()
    }

    pub async fn get_metrics_history(&self, count: usize) -> Vec<PerformanceMetrics> {
        let history = self.metrics_history.read().await;
        history.iter().rev().take(count).cloned().collect()
    }

    pub async fn get_alerts(&self, count: usize) -> Vec<Alert> {
        let alerts = self.alerts.read().await;
        alerts.iter().rev().take(count).cloned().collect()
    }

    pub async fn clear_alerts(&self) {
        let mut alerts = self.alerts.write().await;
        alerts.clear();
    }

    pub async fn get_summary(&self) -> PerformanceSummary {
        let history = self.metrics_history.read().await;

        if history.is_empty() {
            return PerformanceSummary::default();
        }

        let recent: Vec<_> = history.iter().rev().take(60).collect();

        let avg_cpu = recent.iter().map(|m| m.cpu_usage).sum::<f32>() / recent.len() as f32;
        let avg_memory = recent.iter().map(|m| m.memory_usage.used_mb).sum::<usize>() / recent.len();
        let avg_latency = recent.iter().map(|m| m.latency_metrics.p50_ms).sum::<f64>() / recent.len() as f64;
        let total_processed = recent.last().map(|m| m.stream_metrics.total_processed).unwrap_or(0);

        PerformanceSummary {
            avg_cpu_percent: avg_cpu,
            avg_memory_mb: avg_memory,
            avg_latency_ms: avg_latency,
            total_items_processed: total_processed,
            uptime_seconds: self.metrics_counter.load(Ordering::Relaxed) * self.monitoring_interval.as_secs(),
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        if let Some(handle) = self.worker_handle.lock().await.take() {
            handle.await?;
        }

        log::info!("Performance monitor shutdown complete");
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceSummary {
    pub avg_cpu_percent: f32,
    pub avg_memory_mb: usize,
    pub avg_latency_ms: f64,
    pub total_items_processed: u64,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_monitor() {
        let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());

        let monitor = PerformanceMonitor::new(
            memory_manager,
            Duration::from_secs(1),
            100,
        ).await.unwrap();

        monitor.start().await.unwrap();

        tokio::time::sleep(Duration::from_secs(2)).await;

        let metrics = monitor.get_current_metrics().await;
        assert!(metrics.is_some());

        monitor.shutdown().await.unwrap();
    }
}