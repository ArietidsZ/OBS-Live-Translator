//! CPU fallback for GPU operations when CUDA is not available

use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

use crate::TranslatorConfig;

/// Fallback GPU manager that uses CPU for all operations
pub struct OptimizedGpuManager {
    memory_usage: AtomicU64,
    max_memory: u64,
}

impl OptimizedGpuManager {
    /// Create CPU fallback manager
    pub async fn new(config: &TranslatorConfig) -> Result<Self> {
        info!("Using CPU fallback for GPU operations");
        
        Ok(Self {
            memory_usage: AtomicU64::new(0),
            max_memory: config.max_gpu_memory_mb as u64 * 1024 * 1024,
        })
    }

    /// Get memory utilization percentage (always returns 0 for CPU fallback)
    pub async fn get_utilization_percent(&self) -> f32 {
        0.0
    }

    /// Get memory usage in MB
    pub async fn get_memory_usage_mb(&self) -> f32 {
        self.memory_usage.load(Ordering::Relaxed) as f32 / (1024.0 * 1024.0)
    }

    /// Allocate CPU memory (simulated GPU allocation)
    pub fn allocate_memory(&self, size_bytes: u64) -> Result<()> {
        let current = self.memory_usage.load(Ordering::Relaxed);
        if current + size_bytes > self.max_memory {
            return Err(anyhow::anyhow!("Not enough memory available"));
        }
        
        self.memory_usage.store(current + size_bytes, Ordering::Relaxed);
        Ok(())
    }

    /// Free CPU memory (simulated GPU deallocation)
    pub fn free_memory(&self, size_bytes: u64) {
        let current = self.memory_usage.load(Ordering::Relaxed);
        self.memory_usage.store(current.saturating_sub(size_bytes), Ordering::Relaxed);
    }
}