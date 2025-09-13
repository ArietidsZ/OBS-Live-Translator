use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::streaming::{MultiStreamProcessor, AdvancedCache, ResourceManager, ConcurrentPipeline};
use crate::gpu::AdaptiveMemoryManager;
use crate::monitoring::PerformanceMonitor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: ServiceStatus,
    pub timestamp: SystemTime,
    pub uptime_seconds: u64,
    pub version: String,
    pub components: Vec<ComponentHealth>,
    pub metrics: HealthMetrics,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: ServiceStatus,
    pub latency_ms: f64,
    pub error_rate: f32,
    pub last_check: SystemTime,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub active_streams: usize,
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: usize,
    pub gpu_usage_percent: f32,
    pub cache_hit_rate: f32,
    pub avg_processing_time_ms: f64,
    pub error_count_last_minute: usize,
}

pub struct HealthCheckService {
    start_time: Instant,
    stream_processor: Option<Arc<MultiStreamProcessor>>,
    cache: Option<Arc<AdvancedCache>>,
    resource_manager: Option<Arc<ResourceManager>>,
    pipeline: Option<Arc<ConcurrentPipeline>>,
    memory_manager: Arc<AdaptiveMemoryManager>,
    performance_monitor: Option<Arc<PerformanceMonitor>>,
    last_check_results: Arc<RwLock<Vec<ComponentHealth>>>,
    warning_threshold: HealthThresholds,
    critical_threshold: HealthThresholds,
}

#[derive(Debug, Clone)]
struct HealthThresholds {
    cpu_percent: f32,
    memory_percent: f32,
    gpu_percent: f32,
    error_rate: f32,
    latency_ms: f64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            cpu_percent: 80.0,
            memory_percent: 85.0,
            gpu_percent: 90.0,
            error_rate: 0.05,
            latency_ms: 100.0,
        }
    }
}

impl HealthCheckService {
    pub fn new(memory_manager: Arc<AdaptiveMemoryManager>) -> Self {
        Self {
            start_time: Instant::now(),
            stream_processor: None,
            cache: None,
            resource_manager: None,
            pipeline: None,
            memory_manager,
            performance_monitor: None,
            last_check_results: Arc::new(RwLock::new(Vec::new())),
            warning_threshold: HealthThresholds::default(),
            critical_threshold: HealthThresholds {
                cpu_percent: 95.0,
                memory_percent: 95.0,
                gpu_percent: 98.0,
                error_rate: 0.1,
                latency_ms: 500.0,
            },
        }
    }

    pub fn set_stream_processor(&mut self, processor: Arc<MultiStreamProcessor>) {
        self.stream_processor = Some(processor);
    }

    pub fn set_cache(&mut self, cache: Arc<AdvancedCache>) {
        self.cache = Some(cache);
    }

    pub fn set_resource_manager(&mut self, manager: Arc<ResourceManager>) {
        self.resource_manager = Some(manager);
    }

    pub fn set_pipeline(&mut self, pipeline: Arc<ConcurrentPipeline>) {
        self.pipeline = Some(pipeline);
    }

    pub fn set_performance_monitor(&mut self, monitor: Arc<PerformanceMonitor>) {
        self.performance_monitor = Some(monitor);
    }

    pub async fn check_health(&self) -> Result<HealthStatus> {
        let mut components = Vec::new();
        let mut warnings = Vec::new();

        components.push(self.check_memory_manager().await?);

        if let Some(processor) = &self.stream_processor {
            components.push(self.check_stream_processor(processor).await?);
        }

        if let Some(cache) = &self.cache {
            components.push(self.check_cache(cache).await?);
        }

        if let Some(manager) = &self.resource_manager {
            components.push(self.check_resource_manager(manager).await?);
        }

        if let Some(pipeline) = &self.pipeline {
            components.push(self.check_pipeline(pipeline).await?);
        }

        let metrics = self.collect_health_metrics().await?;

        if metrics.cpu_usage_percent > self.warning_threshold.cpu_percent {
            warnings.push(format!("High CPU usage: {:.1}%", metrics.cpu_usage_percent));
        }

        if metrics.memory_usage_mb as f32 >
           (self.memory_manager.get_total_memory().await? as f32 *
            self.warning_threshold.memory_percent / 100.0) {
            warnings.push(format!("High memory usage: {}MB", metrics.memory_usage_mb));
        }

        if metrics.avg_processing_time_ms > self.warning_threshold.latency_ms {
            warnings.push(format!("High latency: {:.1}ms", metrics.avg_processing_time_ms));
        }

        let overall_status = self.determine_overall_status(&components, &metrics);

        *self.last_check_results.write().await = components.clone();

        Ok(HealthStatus {
            status: overall_status,
            timestamp: SystemTime::now(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            components,
            metrics,
            warnings,
        })
    }

    async fn check_memory_manager(&self) -> Result<ComponentHealth> {
        let start = Instant::now();

        let memory_stats = self.memory_manager.get_memory_stats().await?;
        let latency = start.elapsed().as_millis() as f64;

        let usage_percent = (memory_stats.used as f32 / memory_stats.total as f32) * 100.0;

        let status = if usage_percent > 95.0 {
            ServiceStatus::Critical
        } else if usage_percent > 85.0 {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Healthy
        };

        Ok(ComponentHealth {
            name: "MemoryManager".to_string(),
            status,
            latency_ms: latency,
            error_rate: 0.0,
            last_check: SystemTime::now(),
            details: Some(format!("Memory usage: {:.1}%", usage_percent)),
        })
    }

    async fn check_stream_processor(&self, processor: &Arc<MultiStreamProcessor>) -> Result<ComponentHealth> {
        let start = Instant::now();

        let stats = processor.get_all_stats().await;
        let latency = start.elapsed().as_millis() as f64;

        let total_frames: u64 = stats.iter().map(|s| s.total_frames).sum();
        let dropped_frames: u64 = stats.iter().map(|s| s.dropped_frames).sum();

        let drop_rate = if total_frames > 0 {
            dropped_frames as f32 / total_frames as f32
        } else {
            0.0
        };

        let status = if drop_rate > 0.1 {
            ServiceStatus::Critical
        } else if drop_rate > 0.05 {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Healthy
        };

        Ok(ComponentHealth {
            name: "StreamProcessor".to_string(),
            status,
            latency_ms: latency,
            error_rate: drop_rate,
            last_check: SystemTime::now(),
            details: Some(format!("Active streams: {}, Drop rate: {:.2}%",
                                stats.len(), drop_rate * 100.0)),
        })
    }

    async fn check_cache(&self, cache: &Arc<AdvancedCache>) -> Result<ComponentHealth> {
        let start = Instant::now();

        let stats = cache.get_stats().await;
        let latency = start.elapsed().as_millis() as f64;

        let status = if stats.hit_rate < 0.5 {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Healthy
        };

        Ok(ComponentHealth {
            name: "Cache".to_string(),
            status,
            latency_ms: latency,
            error_rate: 1.0 - stats.hit_rate,
            last_check: SystemTime::now(),
            details: Some(format!("Hit rate: {:.1}%, Entries: {}",
                                stats.hit_rate * 100.0, stats.total_entries)),
        })
    }

    async fn check_resource_manager(&self, manager: &Arc<ResourceManager>) -> Result<ComponentHealth> {
        let start = Instant::now();

        let system_stats = manager.get_system_stats().await?;
        let latency = start.elapsed().as_millis() as f64;

        let cpu_critical = system_stats.cpu_usage_percent > self.critical_threshold.cpu_percent;
        let memory_critical = (system_stats.memory_used_mb as f32 /
                             system_stats.memory_total_mb as f32) * 100.0 >
                             self.critical_threshold.memory_percent;

        let status = if cpu_critical || memory_critical {
            ServiceStatus::Critical
        } else if system_stats.cpu_usage_percent > self.warning_threshold.cpu_percent {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Healthy
        };

        Ok(ComponentHealth {
            name: "ResourceManager".to_string(),
            status,
            latency_ms: latency,
            error_rate: 0.0,
            last_check: SystemTime::now(),
            details: Some(format!("CPU: {:.1}%, Memory: {}MB/{} MB",
                                system_stats.cpu_usage_percent,
                                system_stats.memory_used_mb,
                                system_stats.memory_total_mb)),
        })
    }

    async fn check_pipeline(&self, pipeline: &Arc<ConcurrentPipeline>) -> Result<ComponentHealth> {
        let start = Instant::now();

        let metrics = pipeline.get_metrics().await;
        let latency = start.elapsed().as_millis() as f64;

        let failure_rate = if metrics.total_items > 0 {
            metrics.failed_items as f32 / metrics.total_items as f32
        } else {
            0.0
        };

        let status = if failure_rate > 0.1 {
            ServiceStatus::Critical
        } else if failure_rate > 0.05 || metrics.pending_items > 100 {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Healthy
        };

        Ok(ComponentHealth {
            name: "Pipeline".to_string(),
            status,
            latency_ms: latency,
            error_rate: failure_rate,
            last_check: SystemTime::now(),
            details: Some(format!("Processed: {}, Pending: {}, Failed: {}",
                                metrics.completed_items,
                                metrics.pending_items,
                                metrics.failed_items)),
        })
    }

    async fn collect_health_metrics(&self) -> Result<HealthMetrics> {
        let mut active_streams = 0;
        let mut avg_processing_time = 0.0;
        let mut cache_hit_rate = 0.0;

        if let Some(processor) = &self.stream_processor {
            let stats = processor.get_all_stats().await;
            active_streams = stats.len();
            if !stats.is_empty() {
                avg_processing_time = stats.iter()
                    .map(|s| s.avg_processing_time_ms)
                    .sum::<f64>() / stats.len() as f64;
            }
        }

        if let Some(cache) = &self.cache {
            let stats = cache.get_stats().await;
            cache_hit_rate = stats.hit_rate;
        }

        let system_stats = if let Some(manager) = &self.resource_manager {
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

        Ok(HealthMetrics {
            active_streams,
            cpu_usage_percent: system_stats.cpu_usage_percent,
            memory_usage_mb: system_stats.memory_used_mb,
            gpu_usage_percent: if system_stats.gpu_memory_total_mb > 0 {
                (system_stats.gpu_memory_used_mb as f32 /
                 system_stats.gpu_memory_total_mb as f32) * 100.0
            } else {
                0.0
            },
            cache_hit_rate,
            avg_processing_time_ms: avg_processing_time,
            error_count_last_minute: 0,
        })
    }

    fn determine_overall_status(&self, components: &[ComponentHealth], metrics: &HealthMetrics) -> ServiceStatus {
        let critical_count = components.iter()
            .filter(|c| c.status == ServiceStatus::Critical)
            .count();

        let degraded_count = components.iter()
            .filter(|c| c.status == ServiceStatus::Degraded)
            .count();

        if critical_count > 0 {
            ServiceStatus::Critical
        } else if degraded_count > components.len() / 2 {
            ServiceStatus::Unhealthy
        } else if degraded_count > 0 {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Healthy
        }
    }

    pub async fn get_readiness(&self) -> ReadinessStatus {
        let health = match self.check_health().await {
            Ok(h) => h,
            Err(_) => {
                return ReadinessStatus {
                    ready: false,
                    reason: Some("Health check failed".to_string()),
                };
            }
        };

        let ready = match health.status {
            ServiceStatus::Healthy | ServiceStatus::Degraded => true,
            _ => false,
        };

        ReadinessStatus {
            ready,
            reason: if !ready {
                Some(format!("Service status: {:?}", health.status))
            } else {
                None
            },
        }
    }

    pub async fn get_liveness(&self) -> LivenessStatus {
        LivenessStatus {
            alive: true,
            timestamp: SystemTime::now(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessStatus {
    pub ready: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessStatus {
    pub alive: bool,
    pub timestamp: SystemTime,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
        let health_service = HealthCheckService::new(memory_manager);

        let health = health_service.check_health().await.unwrap();
        assert!(matches!(health.status, ServiceStatus::Healthy | ServiceStatus::Degraded));
        assert!(health.uptime_seconds >= 0);
    }
}