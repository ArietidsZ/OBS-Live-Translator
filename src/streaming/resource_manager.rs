use std::collections::{HashMap, BinaryHeap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use anyhow::{Result, anyhow};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};

use crate::gpu::AdaptiveMemoryManager;
use super::multi_stream::StreamPriority;

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub stream_id: String,
    pub cpu_cores: usize,
    pub memory_mb: usize,
    pub gpu_memory_mb: usize,
    pub thread_pool_size: usize,
    pub processing_quota_ms: u64,
    pub priority: StreamPriority,
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_cpu_percent: f32,
    pub max_memory_mb: usize,
    pub max_gpu_memory_mb: usize,
    pub max_threads: usize,
    pub reserved_cpu_percent: f32,
    pub reserved_memory_mb: usize,
}

#[derive(Debug)]
struct StreamResource {
    allocation: ResourceAllocation,
    usage: ResourceUsage,
    last_updated: Instant,
    performance_score: f32,
}

#[derive(Debug, Clone)]
struct ResourceUsage {
    cpu_percent: f32,
    memory_mb: usize,
    gpu_memory_mb: usize,
    active_threads: usize,
    processing_time_ms: u64,
}

pub struct ResourceManager {
    allocations: Arc<RwLock<HashMap<String, StreamResource>>>,
    limits: ResourceLimits,
    memory_manager: Arc<AdaptiveMemoryManager>,
    system: Arc<Mutex<System>>,
    cpu_semaphore: Arc<Semaphore>,
    memory_semaphore: Arc<Semaphore>,
    rebalance_interval: Duration,
    last_rebalance: Arc<Mutex<Instant>>,
    dynamic_scaling: bool,
}

impl ResourceManager {
    pub async fn new(
        limits: ResourceLimits,
        memory_manager: Arc<AdaptiveMemoryManager>,
        dynamic_scaling: bool,
    ) -> Result<Self> {
        let cpu_permits = (limits.max_cpu_percent * 100.0) as usize;
        let memory_permits = limits.max_memory_mb / 100;

        Ok(Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            limits,
            memory_manager,
            system: Arc::new(Mutex::new(System::new_all())),
            cpu_semaphore: Arc::new(Semaphore::new(cpu_permits)),
            memory_semaphore: Arc::new(Semaphore::new(memory_permits)),
            rebalance_interval: Duration::from_secs(5),
            last_rebalance: Arc::new(Mutex::new(Instant::now())),
            dynamic_scaling,
        })
    }

    pub async fn allocate_resources(
        &self,
        stream_id: String,
        priority: StreamPriority,
    ) -> Result<ResourceAllocation> {
        let base_allocation = self.calculate_base_allocation(priority).await?;

        let cpu_permits = base_allocation.cpu_cores;
        let memory_permits = base_allocation.memory_mb / 100;

        let cpu_permit = self.cpu_semaphore
            .try_acquire_many(cpu_permits as u32)
            .map_err(|_| anyhow!("Insufficient CPU resources"))?;

        let memory_permit = self.memory_semaphore
            .try_acquire_many(memory_permits as u32)
            .map_err(|_| {
                drop(cpu_permit);
                anyhow!("Insufficient memory resources")
            })?;

        std::mem::forget(cpu_permit);
        std::mem::forget(memory_permit);

        let allocation = ResourceAllocation {
            stream_id: stream_id.clone(),
            cpu_cores: base_allocation.cpu_cores,
            memory_mb: base_allocation.memory_mb,
            gpu_memory_mb: base_allocation.gpu_memory_mb,
            thread_pool_size: base_allocation.thread_pool_size,
            processing_quota_ms: base_allocation.processing_quota_ms,
            priority,
        };

        let resource = StreamResource {
            allocation: allocation.clone(),
            usage: ResourceUsage {
                cpu_percent: 0.0,
                memory_mb: 0,
                gpu_memory_mb: 0,
                active_threads: 0,
                processing_time_ms: 0,
            },
            last_updated: Instant::now(),
            performance_score: 1.0,
        };

        let mut allocations = self.allocations.write().unwrap();
        allocations.insert(stream_id, resource);

        if self.dynamic_scaling {
            self.trigger_rebalance().await?;
        }

        Ok(allocation)
    }

    async fn calculate_base_allocation(&self, priority: StreamPriority) -> Result<ResourceAllocation> {
        let (cpu_multiplier, memory_multiplier, quota_multiplier) = match priority {
            StreamPriority::Critical => (2.0, 2.0, 3.0),
            StreamPriority::High => (1.5, 1.5, 2.0),
            StreamPriority::Normal => (1.0, 1.0, 1.0),
            StreamPriority::Low => (0.5, 0.5, 0.5),
        };

        let base_cpu = 2;
        let base_memory = 512;
        let base_gpu_memory = 256;
        let base_threads = 2;
        let base_quota = 100;

        Ok(ResourceAllocation {
            stream_id: String::new(),
            cpu_cores: (base_cpu as f32 * cpu_multiplier) as usize,
            memory_mb: (base_memory as f32 * memory_multiplier) as usize,
            gpu_memory_mb: (base_gpu_memory as f32 * memory_multiplier) as usize,
            thread_pool_size: (base_threads as f32 * cpu_multiplier) as usize,
            processing_quota_ms: (base_quota as f32 * quota_multiplier) as u64,
            priority,
        })
    }

    pub async fn release_resources(&self, stream_id: &str) -> Result<()> {
        let mut allocations = self.allocations.write().unwrap();

        if let Some(resource) = allocations.remove(stream_id) {
            let cpu_permits = resource.allocation.cpu_cores;
            let memory_permits = resource.allocation.memory_mb / 100;

            self.cpu_semaphore.add_permits(cpu_permits);
            self.memory_semaphore.add_permits(memory_permits);

            log::info!("Released resources for stream: {}", stream_id);
        }

        Ok(())
    }

    pub async fn update_usage(
        &self,
        stream_id: &str,
        cpu_percent: f32,
        memory_mb: usize,
        processing_time_ms: u64,
    ) -> Result<()> {
        let mut allocations = self.allocations.write().unwrap();

        if let Some(resource) = allocations.get_mut(stream_id) {
            resource.usage.cpu_percent = cpu_percent;
            resource.usage.memory_mb = memory_mb;
            resource.usage.processing_time_ms = processing_time_ms;
            resource.last_updated = Instant::now();

            resource.performance_score = self.calculate_performance_score(
                &resource.allocation,
                &resource.usage,
            );
        }

        Ok(())
    }

    fn calculate_performance_score(
        &self,
        allocation: &ResourceAllocation,
        usage: &ResourceUsage,
    ) -> f32 {
        let cpu_efficiency = if allocation.cpu_cores > 0 {
            1.0 - (usage.cpu_percent / (allocation.cpu_cores as f32 * 100.0)).min(1.0)
        } else {
            0.0
        };

        let memory_efficiency = if allocation.memory_mb > 0 {
            1.0 - (usage.memory_mb as f32 / allocation.memory_mb as f32).min(1.0)
        } else {
            0.0
        };

        let processing_efficiency = if allocation.processing_quota_ms > 0 {
            1.0 - (usage.processing_time_ms as f32 / allocation.processing_quota_ms as f32).min(1.0)
        } else {
            0.0
        };

        (cpu_efficiency + memory_efficiency + processing_efficiency) / 3.0
    }

    async fn trigger_rebalance(&self) -> Result<()> {
        let mut last_rebalance = self.last_rebalance.lock().await;

        if last_rebalance.elapsed() < self.rebalance_interval {
            return Ok(());
        }

        *last_rebalance = Instant::now();
        drop(last_rebalance);

        self.rebalance_resources().await
    }

    async fn rebalance_resources(&self) -> Result<()> {
        let allocations = self.allocations.read().unwrap();

        let mut stream_scores: Vec<(String, StreamPriority, f32)> = allocations
            .iter()
            .map(|(id, resource)| {
                (
                    id.clone(),
                    resource.allocation.priority,
                    resource.performance_score,
                )
            })
            .collect();

        drop(allocations);

        stream_scores.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then(b.2.partial_cmp(&a.2).unwrap())
        });

        let total_cpu = self.limits.max_cpu_percent - self.limits.reserved_cpu_percent;
        let total_memory = self.limits.max_memory_mb - self.limits.reserved_memory_mb;

        let priority_weights = [
            (StreamPriority::Critical, 0.4),
            (StreamPriority::High, 0.3),
            (StreamPriority::Normal, 0.2),
            (StreamPriority::Low, 0.1),
        ];

        for (stream_id, priority, score) in stream_scores {
            if score < 0.3 {
                self.scale_down_resources(&stream_id).await?;
            } else if score > 0.8 {
                self.scale_up_resources(&stream_id).await?;
            }
        }

        log::debug!("Resource rebalancing completed");
        Ok(())
    }

    async fn scale_up_resources(&self, stream_id: &str) -> Result<()> {
        let mut allocations = self.allocations.write().unwrap();

        if let Some(resource) = allocations.get_mut(stream_id) {
            let available_cpu = self.cpu_semaphore.available_permits();
            let available_memory = self.memory_semaphore.available_permits() * 100;

            if available_cpu > 0 {
                resource.allocation.cpu_cores += 1;
                resource.allocation.thread_pool_size += 1;
                self.cpu_semaphore.try_acquire().ok();
            }

            if available_memory >= 256 {
                resource.allocation.memory_mb += 256;
                self.memory_semaphore.try_acquire_many(2).ok();
            }

            log::info!("Scaled up resources for stream: {}", stream_id);
        }

        Ok(())
    }

    async fn scale_down_resources(&self, stream_id: &str) -> Result<()> {
        let mut allocations = self.allocations.write().unwrap();

        if let Some(resource) = allocations.get_mut(stream_id) {
            if resource.allocation.cpu_cores > 1 {
                resource.allocation.cpu_cores -= 1;
                resource.allocation.thread_pool_size =
                    resource.allocation.thread_pool_size.saturating_sub(1).max(1);
                self.cpu_semaphore.add_permits(1);
            }

            if resource.allocation.memory_mb > 256 {
                resource.allocation.memory_mb -= 256;
                self.memory_semaphore.add_permits(2);
            }

            log::info!("Scaled down resources for stream: {}", stream_id);
        }

        Ok(())
    }

    pub async fn get_system_stats(&self) -> Result<SystemStats> {
        let mut system = self.system.lock().await;
        system.refresh_all();

        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let cpu_usage = system.global_cpu_info().cpu_usage();

        let gpu_stats = self.memory_manager.get_memory_stats().await?;

        Ok(SystemStats {
            cpu_usage_percent: cpu_usage,
            memory_used_mb: (used_memory / 1024 / 1024) as usize,
            memory_total_mb: (total_memory / 1024 / 1024) as usize,
            gpu_memory_used_mb: (gpu_stats.used / 1024 / 1024) as usize,
            gpu_memory_total_mb: (gpu_stats.total / 1024 / 1024) as usize,
            active_streams: self.allocations.read().unwrap().len(),
        })
    }

    pub async fn optimize_for_latency(&self, stream_id: &str) -> Result<()> {
        let mut allocations = self.allocations.write().unwrap();

        if let Some(resource) = allocations.get_mut(stream_id) {
            resource.allocation.processing_quota_ms =
                (resource.allocation.processing_quota_ms as f32 * 1.5) as u64;

            if self.cpu_semaphore.available_permits() > 0 {
                resource.allocation.cpu_cores += 1;
                self.cpu_semaphore.try_acquire().ok();
            }

            log::info!("Optimized stream {} for latency", stream_id);
        }

        Ok(())
    }

    pub async fn optimize_for_throughput(&self, stream_id: &str) -> Result<()> {
        let mut allocations = self.allocations.write().unwrap();

        if let Some(resource) = allocations.get_mut(stream_id) {
            if self.memory_semaphore.available_permits() >= 5 {
                resource.allocation.memory_mb += 512;
                self.memory_semaphore.try_acquire_many(5).ok();
            }

            resource.allocation.thread_pool_size =
                (resource.allocation.thread_pool_size + 2).min(8);

            log::info!("Optimized stream {} for throughput", stream_id);
        }

        Ok(())
    }

    pub async fn get_allocation(&self, stream_id: &str) -> Option<ResourceAllocation> {
        let allocations = self.allocations.read().unwrap();
        allocations.get(stream_id).map(|r| r.allocation.clone())
    }

    pub async fn enforce_quotas(&self) -> Result<()> {
        let allocations = self.allocations.read().unwrap();

        for (stream_id, resource) in allocations.iter() {
            if resource.usage.processing_time_ms > resource.allocation.processing_quota_ms {
                log::warn!(
                    "Stream {} exceeded processing quota: {}ms > {}ms",
                    stream_id,
                    resource.usage.processing_time_ms,
                    resource.allocation.processing_quota_ms
                );
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SystemStats {
    pub cpu_usage_percent: f32,
    pub memory_used_mb: usize,
    pub memory_total_mb: usize,
    pub gpu_memory_used_mb: usize,
    pub gpu_memory_total_mb: usize,
    pub active_streams: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_allocation() {
        let limits = ResourceLimits {
            max_cpu_percent: 80.0,
            max_memory_mb: 8192,
            max_gpu_memory_mb: 4096,
            max_threads: 16,
            reserved_cpu_percent: 20.0,
            reserved_memory_mb: 1024,
        };

        let memory_manager = Arc::new(AdaptiveMemoryManager::new().await.unwrap());
        let manager = ResourceManager::new(limits, memory_manager, true).await.unwrap();

        let allocation = manager.allocate_resources(
            "test_stream".to_string(),
            StreamPriority::High,
        ).await.unwrap();

        assert!(allocation.cpu_cores > 0);
        assert!(allocation.memory_mb > 0);

        manager.release_resources("test_stream").await.unwrap();
    }
}