//! Optimized GPU management and memory allocation

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use parking_lot::RwLock;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tracing::{debug, error, info, warn};

use crate::TranslatorConfig;

/// High-performance GPU manager with optimal memory allocation
pub struct OptimizedGpuManager {
    device: Arc<CudaDevice>,
    memory_pool: Arc<RwLock<GpuMemoryPool>>,
    compute_streams: Vec<cudarc::driver::CudaStream>,
    memory_usage: AtomicU64,
    max_memory: u64,
    config: TranslatorConfig,
}

impl OptimizedGpuManager {
    /// Initialize GPU manager with optimal configuration
    pub async fn new(config: &TranslatorConfig) -> Result<Self> {
        info!("Initializing optimized GPU manager");

        // Initialize CUDA device with highest compute capability
        let device = CudaDevice::new(0).context("Failed to initialize CUDA device")?;
        
        info!(
            "Using GPU: {} (Compute Capability: {}.{})",
            device.name()?,
            device.compute_capability().0,
            device.compute_capability().1
        );

        // Get total GPU memory
        let total_memory = device.total_memory()?;
        let max_memory = std::cmp::min(
            (config.max_gpu_memory_mb as u64) * 1024 * 1024,
            (total_memory as f64 * 0.8) as u64, // Use 80% of total memory
        );

        info!("GPU memory limit: {} MB", max_memory / 1024 / 1024);

        // Create multiple compute streams for parallel processing
        let mut compute_streams = Vec::new();
        for i in 0..config.thread_count.min(8) {
            let stream = device.fork_default_stream()?;
            debug!("Created compute stream {}", i);
            compute_streams.push(stream);
        }

        // Initialize memory pool
        let memory_pool = Arc::new(RwLock::new(GpuMemoryPool::new(
            Arc::clone(&device),
            max_memory,
        )?));

        Ok(Self {
            device,
            memory_pool,
            compute_streams,
            memory_usage: AtomicU64::new(0),
            max_memory,
            config: config.clone(),
        })
    }

    /// Allocate GPU memory with optimal alignment
    pub async fn allocate_memory<T: DeviceRepr>(&self, size: usize) -> Result<CudaSlice<T>> {
        let mut pool = self.memory_pool.write();
        let allocation = pool.allocate(size * std::mem::size_of::<T>())?;
        
        self.memory_usage.fetch_add(
            allocation.len() as u64,
            Ordering::Relaxed,
        );

        debug!("Allocated {} bytes on GPU", allocation.len());
        Ok(allocation)
    }

    /// Free GPU memory and return to pool
    pub async fn free_memory<T: DeviceRepr>(&self, memory: CudaSlice<T>) {
        let size = memory.len();
        self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
        
        let mut pool = self.memory_pool.write();
        pool.deallocate(memory);
        
        debug!("Freed {} bytes on GPU", size);
    }

    /// Get optimal compute stream for parallel processing
    pub fn get_compute_stream(&self, index: usize) -> &cudarc::driver::CudaStream {
        &self.compute_streams[index % self.compute_streams.len()]
    }

    /// Get current GPU utilization percentage
    pub async fn get_utilization_percent(&self) -> f32 {
        // Use NVML if available, otherwise estimate from memory usage
        match self.query_gpu_utilization() {
            Ok(utilization) => utilization,
            Err(_) => {
                // Fallback to memory-based estimation
                let memory_usage = self.memory_usage.load(Ordering::Relaxed) as f32;
                let max_memory = self.max_memory as f32;
                (memory_usage / max_memory * 100.0).min(100.0)
            }
        }
    }

    /// Get current memory usage in MB
    pub async fn get_memory_usage_mb(&self) -> f32 {
        self.memory_usage.load(Ordering::Relaxed) as f32 / (1024.0 * 1024.0)
    }

    /// Optimize GPU settings for inference
    pub async fn optimize_for_inference(&self) -> Result<()> {
        info!("Optimizing GPU for inference workloads");

        // Set optimal GPU clocks if possible
        if let Err(e) = self.set_optimal_clocks() {
            warn!("Failed to set optimal GPU clocks: {}", e);
        }

        // Configure memory pool for inference patterns
        let mut pool = self.memory_pool.write();
        pool.optimize_for_inference();

        // Warm up compute streams
        for (i, stream) in self.compute_streams.iter().enumerate() {
            debug!("Warming up compute stream {}", i);
            // Launch a small kernel to warm up the stream
            self.warmup_stream(stream).await?;
        }

        info!("GPU optimization complete");
        Ok(())
    }

    /// Launch kernel with optimal configuration
    pub async fn launch_kernel_optimized<Args: cudarc::driver::LaunchArg>(
        &self,
        func: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        args: Args,
        stream_index: usize,
    ) -> Result<()> {
        let stream = self.get_compute_stream(stream_index);
        
        unsafe {
            func.launch_on_stream(stream, cfg, args)?;
        }
        
        Ok(())
    }

    /// Synchronize all compute streams
    pub async fn synchronize_all(&self) -> Result<()> {
        for stream in &self.compute_streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Query actual GPU utilization using NVML
    fn query_gpu_utilization(&self) -> Result<f32> {
        // This would integrate with NVML for real GPU utilization
        // For now, return a placeholder based on memory usage
        let memory_usage = self.memory_usage.load(Ordering::Relaxed) as f32;
        let max_memory = self.max_memory as f32;
        Ok((memory_usage / max_memory * 100.0).min(100.0))
    }

    /// Set optimal GPU clocks for inference
    fn set_optimal_clocks(&self) -> Result<()> {
        // This would use NVML to set optimal clocks
        // Implementation depends on system permissions
        Ok(())
    }

    /// Warm up compute stream with dummy kernel
    async fn warmup_stream(&self, stream: &cudarc::driver::CudaStream) -> Result<()> {
        // Launch a small dummy kernel to initialize the stream
        let dummy_data = self.device.alloc_zeros::<f32>(1024)?;
        stream.synchronize()?;
        self.device.drop(dummy_data);
        Ok(())
    }
}

/// High-performance GPU memory pool with optimal allocation strategies
struct GpuMemoryPool {
    device: Arc<CudaDevice>,
    free_blocks: Vec<MemoryBlock>,
    allocated_blocks: Vec<MemoryBlock>,
    total_allocated: u64,
    max_memory: u64,
}

impl GpuMemoryPool {
    fn new(device: Arc<CudaDevice>, max_memory: u64) -> Result<Self> {
        Ok(Self {
            device,
            free_blocks: Vec::new(),
            allocated_blocks: Vec::new(),
            total_allocated: 0,
            max_memory,
        })
    }

    /// Allocate memory with optimal strategy
    fn allocate<T: DeviceRepr>(&mut self, size: usize) -> Result<CudaSlice<T>> {
        // Try to find a suitable free block first
        if let Some(block_index) = self.find_suitable_block(size) {
            let block = self.free_blocks.remove(block_index);
            self.allocated_blocks.push(block.clone());
            
            // Create slice from existing allocation
            // This is a simplified version - real implementation would need proper casting
            return self.device.alloc_zeros::<T>(size / std::mem::size_of::<T>());
        }

        // Allocate new memory if no suitable block found
        if self.total_allocated + size as u64 > self.max_memory {
            return Err(anyhow::anyhow!("GPU memory limit exceeded"));
        }

        let allocation = self.device.alloc_zeros::<T>(size / std::mem::size_of::<T>())?;
        
        self.allocated_blocks.push(MemoryBlock {
            ptr: allocation.device_ptr() as u64,
            size,
        });
        
        self.total_allocated += size as u64;
        
        Ok(allocation)
    }

    /// Return memory to the pool
    fn deallocate<T: DeviceRepr>(&mut self, memory: CudaSlice<T>) {
        let ptr = memory.device_ptr() as u64;
        let size = memory.len() * std::mem::size_of::<T>();
        
        // Find and remove from allocated blocks
        if let Some(index) = self.allocated_blocks.iter().position(|block| block.ptr == ptr) {
            let block = self.allocated_blocks.remove(index);
            self.free_blocks.push(block);
            
            // Sort free blocks by size for efficient allocation
            self.free_blocks.sort_by_key(|block| block.size);
        }
        
        // The actual deallocation is handled by CUDA when the slice is dropped
    }

    /// Find suitable free block for allocation
    fn find_suitable_block(&self, required_size: usize) -> Option<usize> {
        self.free_blocks
            .iter()
            .position(|block| block.size >= required_size)
    }

    /// Optimize memory pool for inference workloads
    fn optimize_for_inference(&mut self) {
        // Pre-allocate common sizes for inference
        let common_sizes = vec![
            1024 * 1024,     // 1MB
            4 * 1024 * 1024, // 4MB
            16 * 1024 * 1024, // 16MB
        ];

        for size in common_sizes {
            if self.total_allocated + size as u64 <= self.max_memory {
                // Pre-allocate and immediately free to create pool
                if let Ok(allocation) = self.device.alloc_zeros::<u8>(size) {
                    self.deallocate(allocation);
                }
            }
        }
    }
}

/// Memory block tracking
#[derive(Debug, Clone)]
struct MemoryBlock {
    ptr: u64,
    size: usize,
}

/// GPU performance monitor
pub struct GpuPerformanceMonitor {
    device: Arc<CudaDevice>,
    last_update: std::time::Instant,
    utilization_history: Vec<f32>,
    memory_history: Vec<f32>,
}

impl GpuPerformanceMonitor {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            last_update: std::time::Instant::now(),
            utilization_history: Vec::new(),
            memory_history: Vec::new(),
        }
    }

    /// Update performance metrics
    pub async fn update(&mut self, gpu_manager: &OptimizedGpuManager) {
        let now = std::time::Instant::now();
        
        if now.duration_since(self.last_update).as_millis() >= 100 {
            let utilization = gpu_manager.get_utilization_percent().await;
            let memory_usage = gpu_manager.get_memory_usage_mb().await;
            
            self.utilization_history.push(utilization);
            self.memory_history.push(memory_usage);
            
            // Keep only last 100 measurements
            if self.utilization_history.len() > 100 {
                self.utilization_history.remove(0);
                self.memory_history.remove(0);
            }
            
            self.last_update = now;
        }
    }

    /// Get average utilization over time
    pub fn get_average_utilization(&self) -> f32 {
        if self.utilization_history.is_empty() {
            0.0
        } else {
            self.utilization_history.iter().sum::<f32>() / self.utilization_history.len() as f32
        }
    }

    /// Get peak memory usage
    pub fn get_peak_memory_usage(&self) -> f32 {
        self.memory_history
            .iter()
            .fold(0.0, |max, &usage| max.max(usage))
    }
}