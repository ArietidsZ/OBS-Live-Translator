//! Memory management with tiered allocation strategy for different performance profiles
//!
//! This module implements a sophisticated memory management system:
//! - Global allocator: mimalloc for overall performance
//! - Audio processing: bumpalo for low-latency frame processing
//! - Real-time processing: linked_list_allocator for deterministic allocation
//! - GPU memory pools: custom CUDA/Metal memory management

use crate::profile::Profile;
pub mod audio_buffers;

use anyhow::Result;
use bumpalo::Bump;
use linked_list_allocator::Heap;
use std::sync::{Arc, Mutex};
use std::alloc::Layout;
use tracing::{info, warn, error};

/// Memory pool manager for profile-specific allocation strategies
pub struct MemoryPoolManager {
    profile: Profile,
    audio_bump_allocator: Arc<Mutex<Bump>>,
    realtime_heap: Arc<Mutex<Heap>>,
    gpu_memory_pools: Vec<GpuMemoryPool>,
    allocation_stats: Arc<Mutex<AllocationStats>>,
}

/// Statistics for memory allocation tracking
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub peak_memory_usage: u64,
    pub current_memory_usage: u64,
    pub audio_allocations: u64,
    pub realtime_allocations: u64,
    pub gpu_allocations: u64,
    pub allocation_failures: u64,
}

/// GPU memory pool for ML workloads
#[derive(Debug)]
pub struct GpuMemoryPool {
    pub device_id: u32,
    pub total_memory_mb: u64,
    pub allocated_memory_mb: u64,
    pub available_memory_mb: u64,
    pub pool_type: GpuPoolType,
}

#[derive(Debug, Clone, Copy)]
pub enum GpuPoolType {
    CUDA,
    Metal,
    OpenCL,
    DirectML,
}

/// Memory allocation strategy based on profile
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Low profile: Basic allocation with minimal overhead
    Basic,
    /// Medium profile: Balanced allocation with some optimizations
    Balanced,
    /// High profile: Maximum performance allocation with all optimizations
    HighPerformance,
}

impl MemoryPoolManager {
    /// Create a new memory pool manager for the given profile
    pub fn new(profile: Profile) -> Result<Self> {
        info!("🧠 Initializing memory pool manager for profile {:?}", profile);

        let strategy = Self::get_allocation_strategy(profile);

        // Initialize audio bump allocator
        let audio_bump_allocator = Arc::new(Mutex::new(Bump::new()));

        // Initialize real-time heap (1MB for low latency allocations)
        let mut realtime_heap = Heap::empty();
        let mut heap_memory = vec![0u8; 1024 * 1024]; // 1MB heap
        let heap_start = heap_memory.as_mut_ptr();
        let heap_size = heap_memory.len();

        // SAFETY: We're providing a valid memory range for the heap
        unsafe {
            realtime_heap.init(heap_start, heap_size);
        }

        // Keep the heap memory alive (we need to prevent it from being dropped)
        std::mem::forget(heap_memory);
        let realtime_heap = Arc::new(Mutex::new(realtime_heap));

        // Initialize GPU memory pools based on profile
        let gpu_memory_pools = Self::initialize_gpu_pools(profile)?;

        let allocation_stats = Arc::new(Mutex::new(AllocationStats::default()));

        info!("✅ Memory pool manager initialized with strategy {:?}", strategy);

        Ok(Self {
            profile,
            audio_bump_allocator,
            realtime_heap,
            gpu_memory_pools,
            allocation_stats,
        })
    }

    /// Get allocation strategy for profile
    fn get_allocation_strategy(profile: Profile) -> AllocationStrategy {
        match profile {
            Profile::Low => AllocationStrategy::Basic,
            Profile::Medium => AllocationStrategy::Balanced,
            Profile::High => AllocationStrategy::HighPerformance,
        }
    }

    /// Initialize GPU memory pools for the profile
    fn initialize_gpu_pools(profile: Profile) -> Result<Vec<GpuMemoryPool>> {
        let mut pools = Vec::new();

        match profile {
            Profile::Low => {
                // No GPU pools for low profile (CPU-only)
                info!("📊 Low profile: CPU-only allocation, no GPU pools");
            }
            Profile::Medium => {
                // Create small GPU pool for medium profile
                pools.push(GpuMemoryPool {
                    device_id: 0,
                    total_memory_mb: 2048, // 2GB
                    allocated_memory_mb: 0,
                    available_memory_mb: 2048,
                    pool_type: Self::detect_gpu_pool_type(),
                });
                info!("📊 Medium profile: 2GB GPU memory pool initialized");
            }
            Profile::High => {
                // Create large GPU pool for high profile
                pools.push(GpuMemoryPool {
                    device_id: 0,
                    total_memory_mb: 8192, // 8GB
                    allocated_memory_mb: 0,
                    available_memory_mb: 8192,
                    pool_type: Self::detect_gpu_pool_type(),
                });
                info!("📊 High profile: 8GB GPU memory pool initialized");
            }
        }

        Ok(pools)
    }

    /// Detect the appropriate GPU pool type for the system
    fn detect_gpu_pool_type() -> GpuPoolType {
        #[cfg(target_os = "macos")]
        return GpuPoolType::Metal;

        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        return GpuPoolType::CUDA;

        #[cfg(all(not(target_os = "macos"), not(feature = "cuda"), target_os = "windows"))]
        return GpuPoolType::DirectML;

        #[cfg(all(not(target_os = "macos"), not(feature = "cuda"), not(target_os = "windows")))]
        return GpuPoolType::OpenCL;
    }

    /// Allocate memory for audio frame processing (bump allocator)
    pub fn allocate_audio_frame(&self, size: usize) -> Result<AudioFrameAllocation> {
        let bump = self.audio_bump_allocator.lock().unwrap();
        let layout = Layout::from_size_align(size, 8).unwrap(); // 8-byte aligned

        // Allocate from bump allocator
        let ptr = bump.alloc_layout(layout);

        // Update stats
        {
            let mut stats = self.allocation_stats.lock().unwrap();
            stats.total_allocations += 1;
            stats.audio_allocations += 1;
            stats.current_memory_usage += size as u64;
            if stats.current_memory_usage > stats.peak_memory_usage {
                stats.peak_memory_usage = stats.current_memory_usage;
            }
        }

        Ok(AudioFrameAllocation {
            ptr: ptr.as_ptr(),
            size,
            _lifetime: std::marker::PhantomData,
        })
    }

    /// Allocate memory for real-time processing (TLSF allocator)
    pub fn allocate_realtime(&self, size: usize) -> Result<RealtimeAllocation> {
        let mut heap = self.realtime_heap.lock().unwrap();
        let layout = Layout::from_size_align(size, 8).unwrap();

        match heap.allocate_first_fit(layout) {
            Ok(ptr) => {
                // Update stats
                {
                    let mut stats = self.allocation_stats.lock().unwrap();
                    stats.total_allocations += 1;
                    stats.realtime_allocations += 1;
                    stats.current_memory_usage += size as u64;
                    if stats.current_memory_usage > stats.peak_memory_usage {
                        stats.peak_memory_usage = stats.current_memory_usage;
                    }
                }

                Ok(RealtimeAllocation {
                    ptr: ptr.as_ptr(),
                    size,
                    layout,
                    heap: Arc::clone(&self.realtime_heap),
                })
            }
            Err(_) => {
                // Update failure stats
                {
                    let mut stats = self.allocation_stats.lock().unwrap();
                    stats.allocation_failures += 1;
                }

                Err(anyhow::anyhow!("Real-time allocation failed for size {}", size))
            }
        }
    }

    /// Allocate GPU memory for ML workloads
    pub fn allocate_gpu_memory(&mut self, size_mb: u64, device_id: u32) -> Result<GpuAllocation> {
        // Find the appropriate GPU pool
        let pool = self.gpu_memory_pools
            .iter_mut()
            .find(|pool| pool.device_id == device_id)
            .ok_or_else(|| anyhow::anyhow!("GPU device {} not found", device_id))?;

        if pool.available_memory_mb < size_mb {
            return Err(anyhow::anyhow!(
                "Insufficient GPU memory: requested {}MB, available {}MB",
                size_mb,
                pool.available_memory_mb
            ));
        }

        // Update pool statistics
        pool.allocated_memory_mb += size_mb;
        pool.available_memory_mb -= size_mb;

        // Update global stats
        {
            let mut stats = self.allocation_stats.lock().unwrap();
            stats.total_allocations += 1;
            stats.gpu_allocations += 1;
            stats.current_memory_usage += size_mb * 1024 * 1024;
            if stats.current_memory_usage > stats.peak_memory_usage {
                stats.peak_memory_usage = stats.current_memory_usage;
            }
        }

        info!("🎯 Allocated {}MB GPU memory on device {}", size_mb, device_id);

        Ok(GpuAllocation {
            device_id,
            size_mb,
            ptr: std::ptr::null_mut(), // Placeholder - would be actual GPU pointer
            pool_type: pool.pool_type,
        })
    }

    /// Reset audio bump allocator (for new frame cycles)
    pub fn reset_audio_allocator(&self) {
        let mut bump = self.audio_bump_allocator.lock().unwrap();
        bump.reset();
        info!("🔄 Audio bump allocator reset for new frame cycle");
    }

    /// Get current allocation statistics
    pub fn get_allocation_stats(&self) -> AllocationStats {
        self.allocation_stats.lock().unwrap().clone()
    }

    /// Monitor memory usage and warn if approaching limits
    pub fn monitor_memory_usage(&self) -> Result<()> {
        let stats = self.get_allocation_stats();
        let profile_limits = self.get_profile_memory_limits();

        let usage_percent = (stats.current_memory_usage as f64 / profile_limits.max_memory_bytes as f64) * 100.0;

        if usage_percent > 90.0 {
            error!("🚨 Memory usage critical: {:.1}% of limit", usage_percent);
            return Err(anyhow::anyhow!("Memory usage exceeded 90% of profile limit"));
        } else if usage_percent > 80.0 {
            warn!("⚠️ Memory usage high: {:.1}% of limit", usage_percent);
        } else if usage_percent > 70.0 {
            info!("📊 Memory usage moderate: {:.1}% of limit", usage_percent);
        }

        // Check GPU memory usage
        for pool in &self.gpu_memory_pools {
            let gpu_usage_percent = (pool.allocated_memory_mb as f64 / pool.total_memory_mb as f64) * 100.0;
            if gpu_usage_percent > 85.0 {
                warn!("⚠️ GPU {} memory usage high: {:.1}%", pool.device_id, gpu_usage_percent);
            }
        }

        Ok(())
    }

    /// Get memory limits for current profile
    fn get_profile_memory_limits(&self) -> MemoryLimits {
        match self.profile {
            Profile::Low => MemoryLimits {
                max_memory_bytes: 1200 * 1024 * 1024, // 1.2GB
                max_gpu_memory_bytes: 0,
            },
            Profile::Medium => MemoryLimits {
                max_memory_bytes: 3500 * 1024 * 1024, // 3.5GB
                max_gpu_memory_bytes: 2048 * 1024 * 1024, // 2GB
            },
            Profile::High => MemoryLimits {
                max_memory_bytes: 8000 * 1024 * 1024, // 8GB
                max_gpu_memory_bytes: 8192 * 1024 * 1024, // 8GB
            },
        }
    }

    /// Perform garbage collection and cleanup
    pub fn cleanup(&mut self) -> Result<()> {
        info!("🧹 Performing memory cleanup for profile {:?}", self.profile);

        // Reset audio allocator
        self.reset_audio_allocator();

        // Clear GPU pools if switching to lower profile
        if self.profile == Profile::Low {
            self.gpu_memory_pools.clear();
            info!("🗑️ GPU memory pools cleared for Low profile");
        }

        // Reset stats
        {
            let mut stats = self.allocation_stats.lock().unwrap();
            stats.current_memory_usage = 0;
            stats.total_deallocations += stats.total_allocations;
        }

        info!("✅ Memory cleanup completed");
        Ok(())
    }
}

/// Memory limits for a profile
#[derive(Debug, Clone)]
struct MemoryLimits {
    max_memory_bytes: u64,
    max_gpu_memory_bytes: u64,
}

/// Audio frame allocation using bump allocator
pub struct AudioFrameAllocation {
    ptr: *mut u8,
    size: usize,
    _lifetime: std::marker::PhantomData<&'static ()>,
}

impl AudioFrameAllocation {
    /// Get raw pointer to allocated memory
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get size of allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get as slice (unsafe)
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.size)
    }

    /// Get as mutable slice (unsafe)
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, self.size)
    }
}

/// Real-time allocation using TLSF allocator
pub struct RealtimeAllocation {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
    heap: Arc<Mutex<Heap>>,
}

impl RealtimeAllocation {
    /// Get raw pointer to allocated memory
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get size of allocation
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for RealtimeAllocation {
    fn drop(&mut self) {
        let mut heap = self.heap.lock().unwrap();
        unsafe {
            heap.deallocate(std::ptr::NonNull::new_unchecked(self.ptr), self.layout);
        }
    }
}

/// GPU memory allocation
pub struct GpuAllocation {
    pub device_id: u32,
    pub size_mb: u64,
    pub ptr: *mut u8, // Would be actual GPU pointer in real implementation
    pub pool_type: GpuPoolType,
}

impl GpuAllocation {
    /// Get device ID
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Get size in MB
    pub fn size_mb(&self) -> u64 {
        self.size_mb
    }

    /// Get pool type
    pub fn pool_type(&self) -> GpuPoolType {
        self.pool_type
    }
}

impl Default for AllocationStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            audio_allocations: 0,
            realtime_allocations: 0,
            gpu_allocations: 0,
            allocation_failures: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_manager_creation() {
        let manager = MemoryPoolManager::new(Profile::Low).unwrap();
        assert_eq!(manager.profile, Profile::Low);
        assert_eq!(manager.gpu_memory_pools.len(), 0); // Low profile has no GPU pools
    }

    #[test]
    fn test_audio_frame_allocation() {
        let manager = MemoryPoolManager::new(Profile::Medium).unwrap();
        let allocation = manager.allocate_audio_frame(1024).unwrap();
        assert_eq!(allocation.size(), 1024);
        assert!(!allocation.as_ptr().is_null());
    }

    #[test]
    fn test_realtime_allocation() {
        let manager = MemoryPoolManager::new(Profile::High).unwrap();
        let allocation = manager.allocate_realtime(512).unwrap();
        assert_eq!(allocation.size(), 512);
        assert!(!allocation.as_ptr().is_null());
    }

    #[test]
    fn test_gpu_allocation() {
        let mut manager = MemoryPoolManager::new(Profile::High).unwrap();
        let allocation = manager.allocate_gpu_memory(100, 0).unwrap();
        assert_eq!(allocation.size_mb(), 100);
        assert_eq!(allocation.device_id(), 0);
    }

    #[test]
    fn test_allocation_stats() {
        let manager = MemoryPoolManager::new(Profile::Low).unwrap();
        let stats = manager.get_allocation_stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.current_memory_usage, 0);
    }
}